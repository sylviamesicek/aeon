use crate::{
    geometry::{faces, regions, AxisMask, Region, Side},
    prelude::{Face, FaceMask, IndexSpace, Rectangle},
};
use bitvec::{order::Lsb0, slice::BitSlice, vec::BitVec};
use std::{array, ops::Range, slice};

/// Null index, used internally to make storage of `Option<usize>`` more efficent
const NULL: usize = usize::MAX;

/// Index into active cells in tree.
///
/// This is the primary representation of cells in a `Tree`, as degrees
/// of freedom are only assigned to active cells. Can be converted to generic `CellIndex` via
/// `tree.cell_from_active_index(`
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct ActiveCellIndex(pub usize);

/// Index into cells in a tree.
///
/// A tree stores non-active cells to facilitate O(log n) point -> cell and cell -> neighbor
/// searches. These cells are generated after refinement/coarsening and are therefore not
/// the "source of truth" for the dataset.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct CellIndex(pub usize);

impl CellIndex {
    /// The root cell in a tree is also stored at index 0.
    pub const ROOT: CellIndex = CellIndex(0);

    pub fn child<const N: usize>(offset: Self, split: AxisMask<N>) -> Self {
        Self(offset.0 + split.to_linear())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct Cell<const N: usize> {
    /// Physical bounds of this node
    bounds: Rectangle<N>,
    /// Parent Node
    parent: usize,
    /// Child nodes
    children: usize,
    /// Which active cells are children of this cell?
    active_offset: usize,
    /// Number of active cells which are children of this cell.
    active_count: usize,
    /// Level of cell
    level: usize,
}

/// An `N`-dimensional hypertree, which subdives each axis in two in
/// each refinement step.
///
/// Used as a basis for axes aligned adaptive finite difference
/// meshes. The tree is
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Tree<const N: usize> {
    domain: Rectangle<N>,
    #[serde(with = "crate::array")]
    periodic: [bool; N],
    // *********************
    // Active Cells
    //
    /// Stores structure of the quadtree using `zindex` ordering.
    active_values: BitVec<usize, Lsb0>,
    /// Offsets into `active_indices` (stride of `N`).
    active_offsets: Vec<usize>,
    /// Map from active cell index to general cells.
    active_to_cell: Vec<usize>,
    // *********************
    // All cells
    //
    /// Map from level to cells.
    level_offsets: Vec<usize>,
    /// Bounds of each individual cell.
    cells: Vec<Cell<N>>,
}

impl<const N: usize> Tree<N> {
    /// Constructs a new tree consisting of a single root cell, covering the given
    /// domain.
    pub fn new(domain: Rectangle<N>) -> Self {
        let mut result = Self {
            domain,
            periodic: [false; N],
            active_values: BitVec::new(),
            active_offsets: vec![0, 0],
            active_to_cell: Vec::new(),
            level_offsets: Vec::new(),
            cells: Vec::new(),
        };
        result.build();
        result
    }

    pub fn set_periodic(&mut self, axis: usize, periodic: bool) {
        self.periodic[axis] = periodic;
    }

    /// The number of active (leaf) cells in this tree.
    pub fn num_active_cells(&self) -> usize {
        self.active_offsets.len() - 1
    }

    /// The total number of cells in this tree (including )
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// The maximum depth of this tree.
    pub fn num_levels(&self) -> usize {
        self.level_offsets.len() - 1
    }

    pub fn level_cells(&self, level: usize) -> impl Iterator<Item = CellIndex> + ExactSizeIterator {
        (self.level_offsets[level]..self.level_offsets[level + 1]).map(CellIndex)
    }

    /// Returns the numerical bounds of a given cell.
    pub fn bounds(&self, cell: CellIndex) -> Rectangle<N> {
        self.cells[cell.0].bounds
    }

    /// Returns the level of a given cell.
    pub fn level(&self, cell: CellIndex) -> usize {
        self.cells[cell.0].level
    }

    pub fn active_level(&self, cell: ActiveCellIndex) -> usize {
        self.active_offsets[cell.0 + 1] - self.active_offsets[cell.0]
    }

    /// Returns the children of a given node. Node must not be leaf.
    pub fn children(&self, cell: CellIndex) -> Option<CellIndex> {
        if self.cells[cell.0].children == NULL {
            return None;
        }
        Some(CellIndex(self.cells[cell.0].children))
    }

    /// Returns a child of a give node.
    pub fn child(&self, cell: CellIndex, child: AxisMask<N>) -> Option<CellIndex> {
        if self.cells[cell.0].children == NULL {
            return None;
        }
        Some(CellIndex(self.cells[cell.0].children + child.to_linear()))
    }

    /// The parent node of a given node.
    pub fn parent(&self, cell: CellIndex) -> Option<CellIndex> {
        if self.cells[cell.0].parent == NULL {
            return None;
        }

        Some(CellIndex(self.cells[cell.0].parent))
    }

    /// Returns the zvalue of the given active cell.
    pub fn active_zvalue(&self, active: ActiveCellIndex) -> &BitSlice<usize, Lsb0> {
        &self.active_values
            [N * self.active_offsets[active.0]..N * self.active_offsets[active.0 + 1]]
    }

    pub fn active_split(&self, active: ActiveCellIndex, level: usize) -> AxisMask<N> {
        AxisMask::pack(array::from_fn(|axis| {
            self.active_zvalue(active)[N * level + axis]
        }))
    }

    pub fn refine(&mut self, flags: &[bool]) {
        assert!(self.num_active_cells() == flags.len());

        let num_flags = flags.iter().copied().filter(|&p| p).count();
        let total_active_cells = self.num_active_cells() + (AxisMask::<N>::COUNT - 1) * num_flags;

        let mut active_values = BitVec::with_capacity(total_active_cells * N);
        let mut active_offsets = Vec::with_capacity(total_active_cells);
        active_offsets.push(0);

        for active in 0..self.num_active_cells() {
            if flags[active] {
                for split in AxisMask::<N>::enumerate() {
                    active_values.extend_from_bitslice(self.active_zvalue(ActiveCellIndex(active)));
                    for axis in 0..N {
                        active_values.push(split.is_set(axis));
                    }
                    active_offsets.push(active_values.len() / N);
                }
            } else {
                active_values.extend_from_bitslice(self.active_zvalue(ActiveCellIndex(active)));
                active_offsets.push(active_values.len() / N);
            }
        }

        self.active_values.clone_from(&active_values);
        self.active_offsets.clone_from(&active_offsets);
    }

    pub fn coarsen(&mut self, flags: &[bool]) {
        assert!(flags.len() == self.num_active_cells());

        // Compute number of cells after coarsening
        let num_flags = flags.iter().copied().filter(|&p| p).count();
        debug_assert!(num_flags % AxisMask::<N>::COUNT == 0);
        let total_active = self.num_active_cells() - num_flags / AxisMask::<N>::COUNT;

        let mut active_values = BitVec::with_capacity(total_active * N);
        let mut active_offsets = Vec::new();
        active_offsets.push(0);

        // Loop over cells
        let mut cursor = 0;

        while cursor < self.num_active_cells() {
            // Retrieve zvalue of cursor
            let zvalue = self.active_zvalue(ActiveCellIndex(cursor));

            if flags[cursor] {
                #[cfg(debug_assertions)]
                for split in AxisMask::<N>::enumerate() {
                    assert!(flags[cursor + split.to_linear()])
                }

                active_values.extend_from_bitslice(&zvalue[0..zvalue.len().saturating_sub(N)]);
                // Skip next `Count` cells
                cursor += AxisMask::<N>::COUNT;
            } else {
                active_values.extend_from_bitslice(zvalue);
                cursor += 1;
            }

            active_offsets.push(active_values.len() / N);
        }

        self.active_values.clone_from(&active_values);
        self.active_offsets.clone_from(&active_offsets);
    }

    pub fn build(&mut self) {
        // Reset tree
        self.active_to_cell.resize(self.num_active_cells(), 0);
        self.level_offsets.clear();
        self.cells.clear();

        // Add root cell
        self.cells.push(Cell {
            bounds: self.domain,
            parent: NULL,
            children: NULL,
            active_offset: 0,
            active_count: self.num_active_cells(),
            level: 0,
        });
        self.level_offsets.push(0);
        self.level_offsets.push(1);

        // Recursively subdivide existing nodes using `active_indices`.
        loop {
            let level = self.level_offsets.len() - 2;
            let level_cells = self.level_offsets[level]..self.level_offsets[level + 1];

            // First node on current level
            let next_level_start = self.cells.len();
            // Loop over nodes on the current level
            for parent in level_cells {
                if self.cells[parent].active_count == 1 {
                    self.active_to_cell[self.cells[parent].active_offset] = parent;
                    continue;
                }

                // Update parent's children
                self.cells[parent].children = self.cells.len();
                // Iterate over constituent active cells
                let active_start = self.cells[parent].active_offset;
                let active_end = active_start + self.cells[parent].active_count;

                let mut cursor = active_start;

                let bounds = self.cells[parent].bounds;

                for mask in AxisMask::<N>::enumerate() {
                    let child_cell_start = cursor;

                    while cursor < active_end
                        && mask == self.active_split(ActiveCellIndex(cursor), level)
                    {
                        cursor += 1;
                    }

                    let child_cell_end = cursor;

                    self.cells.push(Cell {
                        bounds: bounds.split(mask),
                        parent,
                        children: NULL,
                        active_offset: child_cell_start,
                        active_count: child_cell_end - child_cell_start,
                        level: level + 1,
                    });
                }
            }

            let next_level_end = self.cells.len();

            if next_level_start >= next_level_end {
                break;
            }

            self.level_offsets.push(next_level_end);
        }
    }

    /// Computes the cell index corresponding to an active cell.
    pub fn cell_from_active_index(&self, active: ActiveCellIndex) -> CellIndex {
        debug_assert!(
            active.0 < self.num_active_cells(),
            "Active cell index is expected to be less that the number of active cells."
        );
        CellIndex(self.active_to_cell[active.0])
    }

    /// Computes active cell index from a cell, returning None if `cell` is
    /// not active.
    pub fn active_index_from_cell(&self, cell: CellIndex) -> Option<ActiveCellIndex> {
        debug_assert!(
            cell.0 < self.num_cells(),
            "Cell index is expected to be less that the number of cells."
        );

        if self.cells[cell.0].active_count != 1 {
            return None;
        }

        Some(ActiveCellIndex(self.cells[cell.0].active_offset))
    }

    /// Returns an iterator over active cells that are children of the given cell.
    /// If `is_active(cell) = true` then this iterator will be a singleton
    /// returning the same value as `tree.active_index_from_cell(cell)`.
    pub fn active_children(
        &self,
        cell: CellIndex,
    ) -> impl Iterator<Item = ActiveCellIndex> + ExactSizeIterator {
        let (offset, count) = (
            self.cells[cell.0].active_offset,
            self.cells[cell.0].active_count,
        );

        (offset..offset + count).map(ActiveCellIndex)
    }

    /// True if a node has no children.
    pub fn is_active(&self, node: CellIndex) -> bool {
        let result = self.cells[node.0].children == NULL;
        debug_assert!(!result || self.cells[node.0].active_count == 1);
        result
    }

    /// Returns the cell which owns the given point.
    /// Performs in O(log N).
    pub fn cell_from_point(&self, point: [f64; N]) -> CellIndex {
        debug_assert!(self.domain.contains(point));

        let mut node = CellIndex(0);

        while let Some(children) = self.children(node) {
            let bounds = self.bounds(node);
            let center = bounds.center();
            node = CellIndex::child(
                children,
                AxisMask::<N>::pack(array::from_fn(|axis| point[axis] > center[axis])),
            );
        }

        node
    }

    /// Returns the node which owns the given point, shortening this search
    /// with an initial guess. Rather than operating in O(log N) time, this approaches
    /// O(1) if the guess is sufficiently close.
    pub fn cell_from_point_cached(&self, point: [f64; N], mut cache: CellIndex) -> CellIndex {
        debug_assert!(self.domain.contains(point));

        while !self.bounds(cache).contains(point) {
            cache = self.parent(cache).unwrap();
        }

        let mut node = cache;

        while let Some(children) = self.children(node) {
            let bounds = self.bounds(node);
            let center = bounds.center();
            node = CellIndex::child(
                children,
                AxisMask::<N>::pack(array::from_fn(|axis| point[axis] > center[axis])),
            )
        }

        node
    }

    /// Returns the neighboring cell along the given face. If the neighboring cell is more refined, this
    /// returns the cell index of the adjacent cell with `tree.level(neighbor) == tree.level(cell)`.
    /// If this passes over a nonperiodic boundary then it returns `None`.
    pub fn neighbor(&self, cell: CellIndex, face: Face<N>) -> Option<CellIndex> {
        let mut region = Region::CENTRAL;
        region.set_side(face.axis, if face.side { Side::Right } else { Side::Left });
        self.neighbor_region(cell, region)
    }

    /// Returns the neighboring cell in the given region. If the neighboring cell is more refined, this
    /// returns the cell index of the adjacent cell with `tree.level(neighbor) == tree.level(cell)`.
    /// If this passes over a nonperiodic boundary then it returns `None`.
    pub fn neighbor_region(&self, cell: CellIndex, region: Region<N>) -> Option<CellIndex> {
        let is_periodic = (0..N)
            .map(|axis| region.side(axis) == Side::Middle || self.periodic[axis])
            .all(|b| b);

        if cell == CellIndex::ROOT && is_periodic {
            return Some(CellIndex::ROOT);
        }

        // Retrieve first active cell owned by `cell`.
        let active_index = ActiveCellIndex(self.cells[cell.0].active_offset);
        // Start at this cell
        let mut cursor = cell;
        // While this cell has a parent, recurse downwards and check whether the region is compatible.
        // If so, break.
        while let Some(parent) = self.parent(cursor) {
            cursor = parent;
            if self
                .active_split(active_index, self.level(cursor))
                .is_inner_region(region)
            {
                break;
            }
        }

        if self.children(cursor).is_some() {
            let split = self.active_split(active_index, self.level(cursor));

            if split.is_inner_region(region) {
                cursor = CellIndex::child(
                    self.children(cursor).unwrap(),
                    split.as_outer_region(region),
                )
            }
        }

        // If we are at root, we can proceed to do silliness (i.e. recurse back upwards)
        if cursor == CellIndex::ROOT {
            if !is_periodic {
                return None;
            }

            debug_assert!(self.level(cell) > 0);

            let split = self.active_split(active_index, self.level(cursor));
            cursor = CellIndex::child(
                self.children(cursor).unwrap(),
                split.as_inner_region(region),
            );
        }

        // Recurse back upwards
        while self.level(cursor) < self.level(cell) {
            let Some(children) = self.children(cursor) else {
                break;
            };

            let split = self
                .active_split(active_index, self.level(cursor))
                .as_inner_region(region);
            cursor = CellIndex::child(children, split);
        }

        // Algorithm complete
        Some(cursor)
    }

    /// Iterates over
    pub fn active_neighbors_in_region(
        &self,
        cell: CellIndex,
        region: Region<N>,
    ) -> impl Iterator<Item = ActiveCellIndex> + '_ {
        let level = self.level(cell);

        self.neighbor_region(cell, region)
            .into_iter()
            .flat_map(move |neighbor| {
                self.active_children(neighbor).filter(move |&active| {
                    for l in level..self.active_level(active) {
                        if !region.is_split_adjacent(self.active_split(active, l)) {
                            return false;
                        }
                    }

                    true
                })
            })
    }

    /// Returns true if a face lies on a boundary.
    pub fn is_boundary_face(&self, cell: CellIndex, face: Face<N>) -> bool {
        let mut region = Region::CENTRAL;
        region.set_side(face.axis, if face.side { Side::Right } else { Side::Left });
        self.boundary_region(cell, region) != Region::CENTRAL
    }

    /// Given a neighboring region to a cell, determines which global region that
    /// belongs to (usually)
    pub fn boundary_region(&self, cell: CellIndex, region: Region<N>) -> Region<N> {
        // Get the active cell owned by this cell.
        let Some(active) = self.active_index_from_cell(cell) else {
            return region;
        };

        let mut result = region;
        let mut level = self.level(cell);

        while level > 0 && result != Region::CENTRAL {
            let split = self.active_split(active, level - 1);
            // Terminate recursion
            if split.is_inner_region(region) {
                break;
            }

            // Mask region by
            for axis in 0..N {
                match (result.side(axis), split.is_set(axis)) {
                    (Side::Left, true) => result.set_side(axis, Side::Middle),
                    (Side::Right, false) => result.set_side(axis, Side::Middle),
                    _ => {}
                }
            }

            level -= 1;
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn neighbors() {
        let mut tree = Tree::<2>::new(Rectangle::UNIT);

        assert_eq!(tree.bounds(CellIndex::ROOT), Rectangle::UNIT);
        assert_eq!(tree.num_cells(), 1);
        assert_eq!(tree.num_active_cells(), 1);
        assert_eq!(tree.num_levels(), 1);

        assert_eq!(tree.neighbor(CellIndex::ROOT, Face::negative(0)), None);

        tree.refine(&[true]);
        tree.build();

        assert_eq!(tree.num_cells(), 5);
        assert_eq!(tree.num_active_cells(), 4);
        assert_eq!(tree.num_levels(), 2);
        for split in AxisMask::enumerate() {
            assert_eq!(
                tree.active_split(ActiveCellIndex(split.to_linear()), 0),
                split
            );
        }
        for i in 0..4 {
            assert_eq!(
                tree.cell_from_active_index(ActiveCellIndex(i)),
                CellIndex(i + 1)
            );
        }

        tree.refine(&[true, false, false, false]);
        tree.build();

        assert_eq!(
            tree.cell_from_active_index(ActiveCellIndex(0)),
            CellIndex(5)
        );

        assert!(tree.is_boundary_face(CellIndex(5), Face::negative(0)));
        assert!(tree.is_boundary_face(CellIndex(5), Face::negative(1)));
        assert_eq!(
            tree.boundary_region(CellIndex(5), Region::new([Side::Left, Side::Right])),
            Region::new([Side::Left, Side::Middle])
        );

        assert_eq!(
            tree.neighbor_region(CellIndex(5), Region::new([Side::Right, Side::Right])),
            Some(CellIndex(8))
        );

        assert_eq!(
            tree.neighbor_region(CellIndex(4), Region::new([Side::Left, Side::Left])),
            Some(CellIndex(1))
        );
    }

    #[test]
    fn periodic_neighbors() {
        let mut tree = Tree::<2>::new(Rectangle::UNIT);
        tree.set_periodic(0, true);
        tree.set_periodic(1, true);
        assert_eq!(
            tree.neighbor(CellIndex::ROOT, Face::negative(0)),
            Some(CellIndex::ROOT)
        );

        // Refine tree
        tree.refine(&[true]);
        tree.refine(&[true, false, false, false]);
        tree.build();

        assert_eq!(
            tree.neighbor(CellIndex(5), Face::negative(0)),
            Some(CellIndex(2))
        );
        assert_eq!(
            tree.neighbor_region(CellIndex(5), Region::new([Side::Left, Side::Left])),
            Some(CellIndex(4))
        );
    }
}

/// Stores neighbor of a cell on a tree.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeCellNeighbor<const N: usize> {
    /// Primary cell.
    pub cell: ActiveCellIndex,
    /// Neighbor cell.
    pub neighbor: ActiveCellIndex,
    /// Which region is the neighbor cell in?
    pub region: Region<N>,
    /// Which periodic region is the neighbor cell in?
    pub boundary_region: Region<N>,
}

/// Neighbor of block.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeBlockNeighbor<const N: usize> {
    /// Primary block.
    pub block: usize,
    /// Neighbor block.
    pub neighbor: usize,
    /// Leftmost cell neighbor.
    pub a: TreeCellNeighbor<N>,
    /// Rightmost cell neighbor.
    pub b: TreeCellNeighbor<N>,
}

impl<const N: usize> TreeBlockNeighbor<N> {
    /// If this is a face neighbor, return the corresponding face, otherwise return `None`.
    pub fn face(&self) -> Option<Face<N>> {
        regions_to_face(self.a.region, self.b.region)
    }
}

fn regions_to_face<const N: usize>(a: Region<N>, b: Region<N>) -> Option<Face<N>> {
    let mut adjacency = 0;
    let mut faxis = 0;
    let mut fside = false;

    for axis in 0..N {
        let aside = a.side(axis);
        let bside = b.side(axis);

        if aside == bside && aside != Side::Middle {
            adjacency += 1;
            faxis = axis;
            fside = aside == Side::Right;
        }
    }

    if adjacency == 1 {
        Some(Face {
            axis: faxis,
            side: fside,
        })
    } else {
        None
    }
}

/// Stores information about neighbors of blocks and cells.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TreeNeighbors<const N: usize> {
    /// Flattened list of lists of neighbors for each block.
    neighbors: Vec<TreeBlockNeighbor<N>>,
    /// Offset map for blocks -> neighbors.
    block_offsets: Vec<usize>,
    /// A cached list of all fine interfaces.
    fine: Vec<usize>,
    /// A cached list of all direct interfaces.
    direct: Vec<usize>,
    /// A cached list of all coarse interfaces.
    coarse: Vec<usize>,
}

impl<const N: usize> TreeNeighbors<N> {
    /// Iterates over all fine `BlockInterface`s.
    pub fn fine(&self) -> impl Iterator<Item = &TreeBlockNeighbor<N>> {
        self.fine.iter().map(|&i| &self.neighbors[i])
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn direct(&self) -> impl Iterator<Item = &TreeBlockNeighbor<N>> {
        self.direct.iter().map(|&i| &self.neighbors[i])
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn coarse(&self) -> impl Iterator<Item = &TreeBlockNeighbor<N>> {
        self.coarse.iter().map(|&i| &self.neighbors[i])
    }

    /// Iterates over all interfaces in the mesh.
    pub fn iter(&self) -> slice::Iter<'_, TreeBlockNeighbor<N>> {
        self.neighbors.iter()
    }

    pub fn fine_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.fine.iter().copied()
    }

    pub fn direct_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.direct.iter().copied()
    }

    pub fn coarse_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.coarse.iter().copied()
    }

    /// Iterates over all neighbors of a block.
    pub fn block(&self, block: usize) -> slice::Iter<'_, TreeBlockNeighbor<N>> {
        self.neighbors[self.block_offsets[block]..self.block_offsets[block + 1]].iter()
    }

    /// Returns the range of neighbor indices belonging to a given block.
    pub fn block_range(&self, block: usize) -> Range<usize> {
        self.block_offsets[block]..self.block_offsets[block + 1]
    }

    pub fn neighbor(&self, idx: usize) -> &TreeBlockNeighbor<N> {
        &self.neighbors[idx]
    }

    /// Rebuilds the block interface data.
    pub fn build(&mut self, tree: &Tree<N>, blocks: &TreeBlocks<N>) {
        self.neighbors.clear();
        self.block_offsets.clear();
        self.fine.clear();
        self.coarse.clear();
        self.direct.clear();

        // Reused memory for neighbors.
        let mut neighbors = Vec::new();

        for block in 0..blocks.len() {
            self.block_offsets.push(self.neighbors.len());

            // Build cell neighbors.
            neighbors.clear();
            Self::build_cell_neighbors(tree, blocks, block, &mut neighbors);

            // Sort neighbors (to group cells from the same block together).
            neighbors.sort_unstable_by(|left, right| {
                let lblock = blocks.active_cell_block(left.neighbor);
                let rblock = blocks.active_cell_block(right.neighbor);

                left.boundary_region
                    .cmp(&right.boundary_region)
                    .then(lblock.cmp(&rblock))
                    .then(left.neighbor.cmp(&right.neighbor))
                    .then(left.cell.cmp(&right.cell))
                    .then(left.region.cmp(&right.region))
            });

            Self::taverse_cell_neighbors(blocks, &mut neighbors, |neighbor, a, b| {
                let acell = tree.cell_from_active_index(a.cell);
                let aneighbor = tree.cell_from_active_index(a.neighbor);

                // Compute this boundary interface.
                let kind = InterfaceKind::from_levels(tree.level(acell), tree.level(aneighbor));
                let interface = TreeBlockNeighbor {
                    block,
                    neighbor,
                    a,
                    b,
                };

                let idx = self.neighbors.len();
                self.neighbors.push(interface);

                match kind {
                    InterfaceKind::Fine => self.fine.push(idx),
                    InterfaceKind::Direct => self.direct.push(idx),
                    InterfaceKind::Coarse => self.coarse.push(idx),
                }
            });
        }

        self.block_offsets.push(self.neighbors.len());
    }

    /// Iterates the cell neighbors of a block, and pushes them onto the memory stack.
    fn build_cell_neighbors(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        block: usize,
        neighbors: &mut Vec<TreeCellNeighbor<N>>,
    ) {
        let block_size = blocks.size(block);
        let block_active_cells = blocks.active_cells(block);
        let block_space = IndexSpace::new(block_size);

        debug_assert!(block_size.iter().product::<usize>() == block_active_cells.len());

        for region in regions::<N>() {
            if region == Region::CENTRAL {
                continue;
            }

            // Find all cells adjacent to the given region.
            for index in block_space.adjacent(region) {
                let active = block_active_cells[block_space.linear_from_cartesian(index)];
                let cell = tree.cell_from_active_index(active);
                let periodic = tree.boundary_region(cell, region);

                for neighbor in tree.active_neighbors_in_region(cell, region) {
                    neighbors.push(TreeCellNeighbor {
                        cell: active,
                        neighbor,
                        region,
                        boundary_region: periodic,
                    })
                }
            }
        }
    }

    /// Traverses a sorted list of cell neighbors, calling f once for each distinct block.
    fn taverse_cell_neighbors(
        blocks: &TreeBlocks<N>,
        neighbors: &mut [TreeCellNeighbor<N>],
        mut f: impl FnMut(usize, TreeCellNeighbor<N>, TreeCellNeighbor<N>),
    ) {
        let mut neighbors = neighbors.iter().cloned().peekable();

        while let Some(a) = neighbors.next() {
            let neighbor = blocks.active_cell_block(a.neighbor);

            // Next we walk through the iterator until we find the last neighbor that is still in this block.
            let mut b = a.clone();

            loop {
                if let Some(next) = neighbors.peek() {
                    if a.boundary_region == next.boundary_region
                        && neighbor == blocks.active_cell_block(next.neighbor)
                    {
                        b = neighbors.next().unwrap();
                        continue;
                    }
                }

                break;
            }

            f(neighbor, a, b)
        }
    }
}

/// Stores dataon how to fill coarse-fine interfaces.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeInterface<const N: usize> {
    /// Target block.
    pub block: usize,
    /// Source block.
    pub neighbor: usize,
    /// Source dof on neighbor.
    #[serde(with = "crate::array")]
    pub source: [isize; N],
    /// Destination dof on target.
    #[serde(with = "crate::array")]
    pub dest: [isize; N],
    /// Number of dofs to be filled along each axis.
    #[serde(with = "crate::array")]
    pub size: [usize; N],
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum InterfaceKind {
    Coarse,
    Direct,
    Fine,
}

impl InterfaceKind {
    fn from_levels(level: usize, neighbor: usize) -> Self {
        match level as isize - neighbor as isize {
            1 => InterfaceKind::Coarse,
            0 => InterfaceKind::Direct,
            -1 => InterfaceKind::Fine,
            _ => panic!("Unbalanced levels"),
        }
    }
}
