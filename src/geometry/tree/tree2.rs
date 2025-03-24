use std::{array, ops::Range, slice};

use bitvec::{order::Lsb0, slice::BitSlice, vec::BitVec};
use faer::linalg::qr;

use crate::{
    geometry::{faces, regions, AxisMask, Region, Side},
    prelude::{Face, FaceMask, IndexSpace, Rectangle},
};

pub const NULL: usize = usize::MAX;

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct ActiveCellIndex(pub usize);

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct CellIndex(pub usize);

impl CellIndex {
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
        // self.active_offsets[cell.0 + 1] - self.active_offsets[cell.0]
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

    // pub fn sibling(&self, cell: usize, split: AxisMask<N>) -> Option<usize> {
    //     Some(self.cells[self.parent(cell)?].children + split.to_linear())
    // }

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

    // pub fn cell_split(&self, cell: usize) -> AxisMask<N> {
    //     assert!(!self.is_root(cell));
    //     let parent = self.parent(cell).unwrap();
    //     AxisMask::from_linear(cell - self.children(parent).unwrap())
    // }

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

            if next_level_start == next_level_end {
                break;
            }

            self.level_offsets.push(next_level_end);
        }
    }

    pub fn cell_from_active_index(&self, active: ActiveCellIndex) -> CellIndex {
        debug_assert!(
            active.0 < self.num_active_cells(),
            "Active cell index is expected to be less that the number of active cells."
        );
        CellIndex(self.active_to_cell[active.0])
    }

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

    pub fn neighbor(&self, cell: CellIndex, face: Face<N>) -> Option<CellIndex> {
        let mut region = Region::CENTRAL;
        region.set_side(face.axis, if face.side { Side::Right } else { Side::Left });
        self.neighbor_region(cell, region)
    }

    pub fn neighbor_no_periodic(&self, cell: CellIndex, face: Face<N>) -> Option<CellIndex> {
        let mut region = Region::CENTRAL;
        region.set_side(face.axis, if face.side { Side::Right } else { Side::Left });
        self.neighbor_impl(cell, region, [false; N])
    }

    pub fn neighbor_region(&self, cell: CellIndex, region: Region<N>) -> Option<CellIndex> {
        self.neighbor_impl(cell, region, self.periodic)
    }

    fn neighbor_impl(
        &self,
        cell: CellIndex,
        region: Region<N>,
        periodic: [bool; N],
    ) -> Option<CellIndex> {
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
        // If we are at root, we can proceed to do silliness (i.e. recurse back upwards)
        if cursor == CellIndex(0)
            && !(0..N)
                .map(|axis| region.side(axis) == Side::Middle || self.periodic[axis])
                .all(|b| b)
        {
            return None;
        }
        // Recurse back upwards
        while let Some(children) = self.children(cursor) {
            cursor = CellIndex::child(
                children,
                self.active_split(active_index, self.level(cursor))
                    .as_inner_region(region),
            )
        }
        // Algorithm complete
        Some(cursor)
    }

    pub fn neighbors_in_region(
        &self,
        cell: CellIndex,
        region: Region<N>,
    ) -> impl Iterator<Item = CellIndex> + '_ {
        let level = self.level(cell);

        self.neighbor_region(cell, region)
            .into_iter()
            .flat_map(move |neighbor| {
                self.active_children(neighbor)
                    .filter(move |&active| {
                        for l in level..self.active_level(active) {
                            if !region.is_split_adjacent(self.active_split(active, l)) {
                                return false;
                            }
                        }

                        true
                    })
                    .map(|active| self.cell_from_active_index(active))
            })
    }

    pub fn periodic_region(&self, cell: CellIndex, region: Region<N>) -> Region<N> {
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

/// Groups cells of a `Tree` into uniform blocks, for more efficient inter-cell communication and multithreading.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeBlocks<const N: usize> {
    /// Stores each cell's position within its parent's block.
    #[serde(with = "crate::array::vec")]
    active_cell_positions: Vec<[usize; N]>,
    /// Maps cell to the block that contains it.
    active_cell_to_block: Vec<usize>,
    /// Stores the size of each block.
    #[serde(with = "crate::array::vec")]
    block_sizes: Vec<[usize; N]>,
    /// A flattened list of lists (for each block) that stores
    /// a local cell index to global cell index map.
    block_active_indices: Vec<ActiveCellIndex>,
    /// The offsets for the aforementioned flattened list of lists.
    block_active_offsets: Vec<usize>,
    /// The physical bounds of each block.
    block_bounds: Vec<Rectangle<N>>,
    /// The level of refinement of each block.
    block_levels: Vec<usize>,
    /// Stores whether block face is on physical boundary.
    boundaries: BitVec,
}

impl<const N: usize> TreeBlocks<N> {
    /// Rebuilds the tree block structure from existing geometric information. Performs greedy meshing
    /// to group cells into blocks.
    pub fn build(&mut self, tree: &Tree<N>) {
        self.build_blocks(tree);
        self.build_bounds(tree);
        self.build_boundaries(tree);
        self.build_levels(tree);
    }

    // Number of blocks in the mesh.
    pub fn len(&self) -> usize {
        self.block_sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the cells associated with the given block.
    pub fn active_cells(&self, block: usize) -> &[ActiveCellIndex] {
        &self.block_active_indices
            [self.block_active_offsets[block]..self.block_active_offsets[block + 1]]
    }

    /// Size of a given block, measured in cells.
    pub fn size(&self, block: usize) -> [usize; N] {
        self.block_sizes[block]
    }

    /// Returns the bounds of the given block.
    pub fn bounds(&self, block: usize) -> Rectangle<N> {
        self.block_bounds[block]
    }

    pub fn level(&self, block: usize) -> usize {
        self.block_levels[block]
    }

    /// Returns boundary flags for a block.
    pub fn boundary_flags(&self, block: usize) -> FaceMask<N> {
        let mut flags = [[false; 2]; N];

        for face in faces::<N>() {
            flags[face.axis][face.side as usize] =
                self.boundaries[block * 2 * N + face.to_linear()];
        }

        FaceMask::pack(flags)
    }

    /// Returns the position of the cell within the block.
    pub fn active_cell_position(&self, cell: ActiveCellIndex) -> [usize; N] {
        self.active_cell_positions[cell.0]
    }

    pub fn active_cell_block(&self, cell: ActiveCellIndex) -> usize {
        self.active_cell_to_block[cell.0]
    }

    fn build_blocks(&mut self, tree: &Tree<N>) {
        let num_active_cells = tree.num_active_cells();

        // Resize/reset various maps
        self.active_cell_positions.resize(num_active_cells, [0; N]);
        self.active_cell_positions.fill([0; N]);

        self.active_cell_to_block
            .resize(num_active_cells, usize::MAX);
        self.active_cell_to_block.fill(usize::MAX);

        self.block_sizes.clear();
        self.block_active_indices.clear();
        self.block_active_offsets.clear();

        // Loop over each cell in the tree
        for active in 0..num_active_cells {
            if self.active_cell_to_block[active] != usize::MAX {
                // This cell already belongs to a block, continue.
                continue;
            }

            // Get index of next block
            let block = self.block_sizes.len();

            self.active_cell_positions[active] = [0; N];
            self.active_cell_to_block[active] = block;

            self.block_sizes.push([1; N]);
            let block_cell_offset = self.block_active_indices.len();

            self.block_active_offsets.push(block_cell_offset);
            self.block_active_indices.push(ActiveCellIndex(active));

            // Try expanding the block along each axis.
            for axis in 0..N {
                // Perform greedy meshing.
                'expand: loop {
                    let face = Face::<N>::positive(axis);

                    let size = self.block_sizes[block];
                    let space = IndexSpace::new(size);

                    // Make sure every cell on face is suitable for expansion.
                    for index in space.face(Face::positive(axis)).iter() {
                        // Retrieves the cell on this face
                        let cell = tree.cell_from_active_index(
                            self.block_active_indices
                                [block_cell_offset + space.linear_from_cartesian(index)],
                        );
                        let level = tree.level(cell);
                        // We can only expand if
                        // 1. We are not on a boundary
                        // 2. The neighbor is the same level of refinement
                        // 3. The neighbor does not already belong to another block.
                        let Some(neighbor) = tree.neighbor_no_periodic(cell, face) else {
                            break 'expand;
                        };

                        if level != tree.level(neighbor) {
                            break 'expand;
                        }

                        if self.active_cell_to_block
                            [tree.active_index_from_cell(neighbor).unwrap().0]
                            != usize::MAX
                        {
                            break 'expand;
                        }
                    }

                    // We may now expand along this axis
                    for index in space.face(Face::positive(axis)).iter() {
                        let active = self.block_active_indices
                            [block_cell_offset + space.linear_from_cartesian(index)];

                        let cell = tree.cell_from_active_index(active);
                        let cell_neighbor = tree.neighbor(cell, face).unwrap();
                        let active_neighbor = tree.active_index_from_cell(cell_neighbor).unwrap();

                        self.active_cell_positions[active_neighbor.0] = index;
                        self.active_cell_positions[active_neighbor.0][axis] += 1;
                        self.active_cell_to_block[active_neighbor.0] = block;

                        self.block_active_indices.push(active_neighbor);
                    }

                    self.block_sizes[block][axis] += 1;
                }
            }
        }

        self.block_active_offsets
            .push(self.block_active_indices.len());
    }

    fn build_bounds(&mut self, tree: &Tree<N>) {
        self.block_bounds.clear();

        for block in 0..self.len() {
            let size = self.block_sizes[block];
            let a = *self.active_cells(block).first().unwrap();

            let cell_bounds = tree.bounds(tree.cell_from_active_index(a));

            self.block_bounds.push(Rectangle {
                origin: cell_bounds.origin,
                size: array::from_fn(|axis| cell_bounds.size[axis] * size[axis] as f64),
            })
        }
    }

    fn build_boundaries(&mut self, tree: &Tree<N>) {
        self.boundaries.clear();

        for block in 0..self.len() {
            let a = 0;
            let b: usize = self.active_cells(block).len() - 1;

            for face in faces::<N>() {
                let active = if face.side {
                    self.active_cells(block)[b]
                } else {
                    self.active_cells(block)[a]
                };
                let cell = tree.cell_from_active_index(active);
                self.boundaries.push(tree.neighbor(cell, face).is_none());
            }
        }
    }

    fn build_levels(&mut self, tree: &Tree<N>) {
        self.block_levels.resize(self.len(), 0);
        for block in 0..self.len() {
            let active = self.active_cells(block)[0];
            let cell = tree.cell_from_active_index(active);
            self.block_levels[block] = tree.level(cell);
        }
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
    pub periodic_region: Region<N>,
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

pub fn regions_to_face<const N: usize>(a: Region<N>, b: Region<N>) -> Option<Face<N>> {
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

                left.periodic_region
                    .cmp(&right.periodic_region)
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
                let periodic = tree.periodic_region(cell, region);

                for neighbor in tree.neighbors_in_region(cell, region) {
                    neighbors.push(TreeCellNeighbor {
                        cell: active,
                        neighbor: tree.active_index_from_cell(neighbor).unwrap(),
                        region,
                        periodic_region: periodic,
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
                    if a.periodic_region == next.periodic_region
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
