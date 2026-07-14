#![allow(clippy::needless_range_loop)]

use crate::{
    geometry::{Region, Side, Split, regions},
    prelude::{Face, HyperBox, IndexSpace},
};
use bitvec::{order::Lsb0, slice::BitSlice, vec::BitVec};
use datasize::DataSize;
use std::{array, ops::Range, slice};

mod blocks;
mod interfaces;
mod neighbors;

pub use blocks::{BlockId, TreeBlocks};
pub use interfaces::{TreeInterface, TreeInterfaces};
pub use neighbors::{NeighborId, TreeBlockNeighbor, TreeCellNeighbor, TreeNeighbors};

/// Null index, used internally to make storage of `Option<usize>`` more efficent
const NULL: usize = usize::MAX;

/// Index into leaves in tree.
///
/// This is the primary representation of cells in a `Tree`, as degrees
/// of freedom are only assigned to leaves. Can be converted to generic `CellId` via
/// `tree.cell_from_leaf_index(`
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    DataSize,
)]
pub struct LeafId(pub usize);

/// Index into cells in a tree.
///
/// A tree stores non-leaf cells to facilitate O(log n) point -> cell and cell -> neighbor
/// searches. These cells are generated after refinement/coarsening and are therefore not
/// the "source of truth" for the dataset.
#[derive(
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    DataSize,
)]
pub struct CellId(pub usize);

impl CellId {
    /// The root cell in a tree is also stored at index 0.
    pub const ROOT: CellId = CellId(0);

    /// Index to child located at `offset`.
    pub fn child<const N: usize>(offset: Self, split: Split<N>) -> Self {
        Self(offset.0 + split.to_linear())
    }
}

/// Index into a tree as a sequence of increasingly refined uniform grids.
#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct UniformId<const N: usize> {
    pub level: usize,
    /// Position on `level`-th uniform grid.
    #[serde(with = "crate::array")]
    pub position: [usize; N],
}

impl<const N: usize> UniformId<N> {
    /// Does self.position fit into the uniform grid at level self.level
    pub fn is_valid(self) -> bool {
        debug_assert!(self.level < 64);

        let max_grid_position = 1usize << self.level;
        self.position.iter().all(|&i| i < max_grid_position)
    }

    /// Returns true if this uniform index contains a more refined index.
    pub fn contains(self, other: Self) -> bool {
        if other.level < self.level {
            return false;
        }

        let diff = other.level - self.level;

        (0..N)
            .into_iter()
            .all(|axis| (other.position[axis] >> diff) == self.position[axis])
    }

    pub fn coarsened(self) -> Self {
        Self {
            level: self.level - 1,
            position: std::array::from_fn(|axis| self.position[axis] >> 1),
        }
    }
}

impl<const N: usize> DataSize for UniformId<N> {
    const IS_DYNAMIC: bool = false;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct Cell<const N: usize> {
    /// Physical bounds of this node
    bounds: HyperBox<N>,
    /// Parent Node
    parent: usize,
    /// Child nodes
    children: usize,
    /// Which leaves are children of this cell?
    leaf_offset: usize,
    /// Number of leaves which are children of this cell.
    leaf_count: usize,
    /// Level of cell
    level: usize,
}

impl<const N: usize> DataSize for Cell<N> {
    const IS_DYNAMIC: bool = false;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        0
    }
}

/// An `N`-dimensional hypertree, which subdives each axis in two in
/// each refinement step.
///
/// Used as a basis for axes aligned adaptive finite difference
/// meshes. The tree is
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(from = "TreeSer<N>", into = "TreeSer<N>")]
pub struct Tree<const N: usize> {
    domain: HyperBox<N>,
    /// Is this tree periodic along any axes?
    periodic: [bool; N],
    // *********************
    // Active Cells
    //
    /// Stores structure of the quadtree using `zindex` ordering.
    leaf_values: BitVec<usize, Lsb0>,
    /// Offsets into `active_indices` (stride of `N`).
    leaf_offsets: Vec<usize>,
    /// Map from leave indices to general cells.
    leaf_to_cell: Vec<usize>,
    // *********************
    // All cells
    //
    /// Bounds of each individual cell.
    cells: Vec<Cell<N>>,
    /// Map from level to cells.
    level_offsets: Vec<usize>,
}

impl<const N: usize> Tree<N> {
    /// Constructs a new tree consisting of a single root cell, covering the given
    /// domain.
    pub fn new(domain: HyperBox<N>) -> Self {
        let mut result = Self {
            domain,
            periodic: [false; N],
            leaf_values: BitVec::new(),
            leaf_offsets: vec![0, 0],
            leaf_to_cell: Vec::new(),
            level_offsets: Vec::new(),
            cells: Vec::new(),
        };
        result.build();
        result
    }

    pub fn set_periodic(&mut self, axis: usize, periodic: bool) {
        self.periodic[axis] = periodic;
    }

    pub fn domain(&self) -> HyperBox<N> {
        self.domain
    }

    /// The number of active (leaf) cells in this tree.
    pub fn num_leaves(&self) -> usize {
        self.leaf_offsets.len() - 1
    }

    /// The total number of cells in this tree (including )
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }

    /// The maximum depth of this tree.
    pub fn num_levels(&self) -> usize {
        self.level_offsets.len() - 1
    }

    pub fn level_cells(&self, level: usize) -> impl Iterator<Item = CellId> + ExactSizeIterator {
        (self.level_offsets[level]..self.level_offsets[level + 1]).map(CellId)
    }

    /// Iterator over all cells in the tree.
    pub fn cells(&self) -> impl Iterator<Item = CellId> {
        (0..self.num_cells()).map(CellId)
    }

    /// Iterator over all leaves in the tree.
    pub fn leaves(&self) -> impl Iterator<Item = LeafId> {
        (0..self.num_leaves()).map(LeafId)
    }

    /// Returns the numerical bounds of a given cell.
    pub fn bounds(&self, cell: CellId) -> HyperBox<N> {
        self.cells[cell.0].bounds
    }

    pub fn leaf_bounds(&self, active: LeafId) -> HyperBox<N> {
        self.bounds(self.cell_from_leaf(active))
    }

    /// Returns the level of a given cell.
    pub fn level(&self, cell: CellId) -> usize {
        self.cells[cell.0].level
    }

    pub fn leaf_level(&self, cell: LeafId) -> usize {
        self.leaf_offsets[cell.0 + 1] - self.leaf_offsets[cell.0]
    }

    /// Returns the children offset of a given node. Node must not be leaf.
    pub fn child_offset(&self, cell: CellId) -> Option<CellId> {
        if self.cells[cell.0].children == NULL {
            return None;
        }
        Some(CellId(self.cells[cell.0].children))
    }

    /// Returns a specific child of a give cell.
    pub fn child(&self, cell: CellId, child: Split<N>) -> Option<CellId> {
        if self.cells[cell.0].children == NULL {
            return None;
        }
        Some(CellId(self.cells[cell.0].children + child.to_linear()))
    }

    /// The parent node of a given node.
    pub fn parent(&self, cell: CellId) -> Option<CellId> {
        if self.cells[cell.0].parent == NULL {
            return None;
        }

        Some(CellId(self.cells[cell.0].parent))
    }

    /// Returns the zvalue of the given active cell.
    pub fn leaf_zvalue(&self, leaf: LeafId) -> &BitSlice<usize, Lsb0> {
        &self.leaf_values[N * self.leaf_offsets[leaf.0]..N * self.leaf_offsets[leaf.0 + 1]]
    }

    pub fn leaf_split(&self, leaf: LeafId, level: usize) -> Split<N> {
        Split::pack(array::from_fn(|axis| {
            self.leaf_zvalue(leaf)[N * level + axis]
        }))
    }

    pub fn most_recent_leaf_split(&self, active: LeafId) -> Option<Split<N>> {
        if self.num_cells() == 1 {
            return None;
        }

        Some(self.leaf_split(active, self.leaf_level(active) - 1))
    }

    /// Checks whether the given refinement flags are balanced.
    pub fn check_refine_flags(&self, flags: &[bool]) -> bool {
        assert!(flags.len() == self.num_leaves());

        for cell in self.leaves() {
            if !flags[cell.0] {
                continue;
            }

            for coarse in self.leaf_coarse_neighborhood(cell) {
                if !flags[coarse.0] {
                    return false;
                }
            }
        }

        true
    }

    /// Balances the given refinement flags, flagging additional cells
    /// for refinement to preserve the 2:1 fine coarse ratio between every
    /// two neighbors.
    pub fn balance_refine_flags(&self, flags: &mut [bool]) {
        assert!(flags.len() == self.num_leaves());

        loop {
            let mut is_balanced = true;

            for cell in self.leaves() {
                if !flags[cell.0] {
                    continue;
                }

                for coarse in self.leaf_coarse_neighborhood(cell) {
                    if !flags[coarse.0] {
                        is_balanced = false;
                        flags[coarse.0] = true;
                    }
                }
            }

            if is_balanced {
                break;
            }
        }
    }

    /// Fills the map with updated indices after refinement is performed.
    /// If a leaf is refined, this will point to the first leaf in that new subdivision.
    pub fn refine_leaf_index_map(&self, flags: &[bool], map: &mut [LeafId]) {
        assert!(flags.len() == self.num_leaves());
        assert!(map.len() == self.num_leaves());

        let mut cursor = 0;

        for cell in 0..self.num_leaves() {
            map[cell] = LeafId(cursor);

            if flags[cell] {
                cursor += Split::<N>::COUNT;
            } else {
                cursor += 1;
            }
        }
    }

    /// Performs tree refinement.
    pub fn refine(&mut self, flags: &[bool]) {
        assert!(self.num_leaves() == flags.len());

        let num_flags = flags.iter().copied().filter(|&p| p).count();
        let total_leaves = self.num_leaves() + (Split::<N>::COUNT - 1) * num_flags;

        let mut leaf_values = BitVec::with_capacity(total_leaves * N);
        let mut leaf_offsets = Vec::with_capacity(total_leaves);
        leaf_offsets.push(0);

        for leaf in 0..self.num_leaves() {
            if flags[leaf] {
                for split in Split::<N>::enumerate() {
                    leaf_values.extend_from_bitslice(self.leaf_zvalue(LeafId(leaf)));
                    for axis in 0..N {
                        leaf_values.push(split.is_set(axis));
                    }
                    leaf_offsets.push(leaf_values.len() / N);
                }
            } else {
                leaf_values.extend_from_bitslice(self.leaf_zvalue(LeafId(leaf)));
                leaf_offsets.push(leaf_values.len() / N);
            }
        }

        self.leaf_values.clone_from(&leaf_values);
        self.leaf_offsets.clone_from(&leaf_offsets);

        self.build();
    }

    /// Checks that the given coarsening flags are balanced and valid.
    pub fn check_coarsen_flags(&self, flags: &[bool]) -> bool {
        assert!(flags.len() == self.num_leaves());

        if flags.len() == 1 {
            return true;
        }

        // Short circuit if this mesh only has two levels.
        if flags.len() == Split::<N>::COUNT {
            return flags.iter().all(|&b| !b);
        }

        // First if any flagging would break 2:1 border, unmark it
        for leaf in self.leaves() {
            if !flags[leaf.0] {
                for neighbor in self.leaf_coarse_neighborhood(leaf) {
                    // Set any coarser cells to not be coarsened further.
                    if flags[neighbor.0] {
                        return false;
                    }
                }
            }
        }

        // Make sure only cells that can be coarsened are coarsened. And that every single child of such a cell
        // is flagged.
        let mut leaf = 0;

        while leaf < self.num_leaves() {
            if !flags[leaf] {
                leaf += 1;
                continue;
            }

            // if flags[cell] {
            let level = self.leaf_level(LeafId(leaf));
            let split = self.most_recent_leaf_split(LeafId(leaf)).unwrap();

            if split != Split::<N>::empty() {
                return false;
            }

            for offset in 0..Split::<N>::COUNT {
                if self.leaf_level(LeafId(leaf + offset)) != level {
                    return false;
                }
            }

            if !flags[leaf..leaf + Split::<N>::COUNT].iter().all(|&b| b) {
                return false;
            }
            // Skip forwards. We have considered all cases.
            leaf += Split::<N>::COUNT;
        }

        true
    }

    /// Balances the given coarsening flags
    pub fn balance_coarsen_flags(&self, flags: &mut [bool]) {
        assert!(flags.len() == self.num_leaves());

        if flags.len() == 1 {
            return;
        }

        // Short circuit if this mesh only has two levels.
        if flags.len() == Split::<N>::COUNT {
            flags.fill(false);
        }

        loop {
            let mut is_balanced = true;

            // First if any flagging would break 2:1 border, unmark it
            for leaf in self.leaves() {
                if !flags[leaf.0] {
                    for neighbor in self.leaf_coarse_neighborhood(leaf) {
                        // Set any coarser cells to not be coarsened further.
                        if flags[neighbor.0] {
                            is_balanced = false;
                        }
                        flags[neighbor.0] = false;
                    }
                }
            }

            // Make sure only cells that can be coarsened are coarsened. And that every single child of such a cell
            // is flagged.
            let mut leaf = 0;

            while leaf < self.num_leaves() {
                if !flags[leaf] {
                    leaf += 1;
                    continue;
                }

                let level = self.leaf_level(LeafId(leaf));
                let split = self.most_recent_leaf_split(LeafId(leaf)).unwrap();

                if split != Split::<N>::empty() {
                    flags[leaf] = false;
                    is_balanced = false;
                    leaf += 1;
                    continue;
                }

                for offset in 0..Split::<N>::COUNT {
                    if self.leaf_level(LeafId(leaf + offset)) != level {
                        flags[leaf] = false;
                        is_balanced = false;
                        leaf += 1;
                        continue;
                    }
                }

                if !flags[leaf..leaf + Split::<N>::COUNT].iter().all(|&b| b) {
                    flags[leaf..leaf + Split::<N>::COUNT].fill(false);
                    is_balanced = false;
                }
                // Skip forwards. We have considered all cases.
                leaf += Split::<N>::COUNT;
            }

            if is_balanced {
                break;
            }
        }
    }

    /// Maps current cells to indices after coarsening is performed.
    pub fn coarsen_leaf_index_map(&self, flags: &[bool], map: &mut [LeafId]) {
        assert!(flags.len() == self.num_leaves());
        assert!(map.len() == self.num_leaves());

        let mut cursor = 0;
        let mut cell = 0;

        while cell < self.num_leaves() {
            if flags[cell] {
                map[cell..cell + Split::<N>::COUNT].fill(LeafId(cursor));
                cell += Split::<N>::COUNT;
            } else {
                map[cell] = LeafId(cursor);
                cell += 1;
            }

            cursor += 1;
        }
    }

    pub fn coarsen(&mut self, flags: &[bool]) {
        assert!(flags.len() == self.num_leaves());

        // Compute number of cells after coarsening
        let num_flags = flags.iter().copied().filter(|&p| p).count();
        debug_assert!(num_flags % Split::<N>::COUNT == 0);
        let total_leaves = self.num_leaves() - num_flags / Split::<N>::COUNT;

        let mut leaf_values = BitVec::with_capacity(total_leaves * N);
        let mut leaf_offsets = Vec::new();
        leaf_offsets.push(0);

        // Loop over cells
        let mut cursor = 0;

        while cursor < self.num_leaves() {
            // Retrieve zvalue of cursor
            let zvalue = self.leaf_zvalue(LeafId(cursor));

            if flags[cursor] {
                #[cfg(debug_assertions)]
                for split in Split::<N>::enumerate() {
                    assert!(flags[cursor + split.to_linear()])
                }

                leaf_values.extend_from_bitslice(&zvalue[0..zvalue.len().saturating_sub(N)]);
                // Skip next `Count` cells
                cursor += Split::<N>::COUNT;
            } else {
                leaf_values.extend_from_bitslice(zvalue);
                cursor += 1;
            }

            leaf_offsets.push(leaf_values.len() / N);
        }

        self.leaf_values.clone_from(&leaf_values);
        self.leaf_offsets.clone_from(&leaf_offsets);

        self.build();
    }

    pub fn build(&mut self) {
        // Reset tree
        self.leaf_to_cell.resize(self.num_leaves(), 0);
        self.level_offsets.clear();
        self.cells.clear();

        // Add root cell
        self.cells.push(Cell {
            bounds: self.domain,
            parent: NULL,
            children: NULL,
            leaf_offset: 0,
            leaf_count: self.num_leaves(),
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
                if self.cells[parent].leaf_count == 1 {
                    debug_assert!(self.leaf_level(LeafId(self.cells[parent].leaf_offset)) == level);
                    self.leaf_to_cell[self.cells[parent].leaf_offset] = parent;
                    continue;
                }

                // Update parent's children
                self.cells[parent].children = self.cells.len();
                // Iterate over constituent active cells
                let active_start = self.cells[parent].leaf_offset;
                let active_end = active_start + self.cells[parent].leaf_count;

                let mut cursor = active_start;

                debug_assert!(self.leaf_level(LeafId(cursor)) > level);

                let bounds = self.cells[parent].bounds;

                for mask in Split::<N>::enumerate() {
                    let child_cell_start = cursor;

                    while cursor < active_end && mask == self.leaf_split(LeafId(cursor), level) {
                        cursor += 1;
                    }

                    let child_cell_end = cursor;

                    self.cells.push(Cell {
                        bounds: bounds.subdivide(mask),
                        parent,
                        children: NULL,
                        leaf_offset: child_cell_start,
                        leaf_count: child_cell_end - child_cell_start,
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

        #[cfg(debug_assertions)]
        for cell in self.cells() {
            let active = LeafId(self.cells[cell.0].leaf_offset);
            assert!(self.leaf_level(active) >= self.level(cell));
        }
    }

    /// Computes the cell index corresponding to an active cell.
    pub fn cell_from_leaf(&self, leaf: LeafId) -> CellId {
        debug_assert!(
            leaf.0 < self.num_leaves(),
            "Leaf index is expected to be less that the number of leaves."
        );
        CellId(self.leaf_to_cell[leaf.0])
    }

    /// Transforms a cell id into a leaf id, returning None if `cell` is
    /// not a leaf.
    pub fn leaf_from_cell(&self, cell: CellId) -> Option<LeafId> {
        debug_assert!(
            cell.0 < self.num_cells(),
            "Cell index is expected to be less that the number of cells."
        );

        if self.cells[cell.0].leaf_count != 1 {
            return None;
        }

        Some(LeafId(self.cells[cell.0].leaf_offset))
    }

    /// Returns an iterator over leaves that are children of the given cell.
    /// If `is_leaf(cell) = true` then this iterator will be a singleton
    /// returning the same value as `tree.active_index_from_cell(cell)`.
    pub fn contained_leaves(
        &self,
        cell: CellId,
    ) -> impl Iterator<Item = LeafId> + ExactSizeIterator {
        let (offset, count) = (
            self.cells[cell.0].leaf_offset,
            self.cells[cell.0].leaf_count,
        );

        (offset..offset + count).map(LeafId)
    }

    /// True if a cell has no children.
    pub fn is_leaf(&self, cell: CellId) -> bool {
        let result = self.cells[cell.0].children == NULL;
        debug_assert!(!result || self.cells[cell.0].leaf_count == 1);
        result
    }

    /// Returns the cell which owns the given point.
    /// Performs in O(log N).
    pub fn cell_from_point(&self, point: [f64; N]) -> CellId {
        debug_assert!(self.domain.contains(point));

        let mut cell = CellId(0);

        while let Some(children) = self.child_offset(cell) {
            let bounds = self.bounds(cell);
            let center = bounds.center();
            cell = CellId::child(
                children,
                Split::<N>::pack(array::from_fn(|axis| point[axis] >= center[axis])),
            );
        }

        cell
    }

    /// Returns the cells which owns the given point, shortening this search
    /// with an initial guess. Rather than operating in O(log N) time, this approaches
    /// O(1) if the guess is sufficiently close.
    pub fn cell_from_point_cached(&self, point: [f64; N], mut cache: CellId) -> CellId {
        debug_assert!(self.domain.contains(point));

        while !self.bounds(cache).contains(point) {
            cache = self.parent(cache).unwrap();
        }

        let mut cell = cache;

        while let Some(children) = self.child_offset(cell) {
            let bounds = self.bounds(cell);
            let center = bounds.center();
            cell = CellId::child(
                children,
                Split::<N>::pack(array::from_fn(|axis| point[axis] >= center[axis])),
            )
        }

        cell
    }

    /// Returns the uniform index corresponding to a cell
    pub fn uniform_index_from_cell(&self, cell: CellId) -> UniformId<N> {
        debug_assert!(cell.0 <= self.num_cells());

        let level = self.level(cell);
        // ZValue of cell stored as [[bool; N]; level]
        let zvalue = &self.leaf_zvalue(LeafId(self.cells[cell.0].leaf_offset))[0..N * level];

        let mut position = [0usize; N];

        for i in 0..level {
            for axis in 0..N {
                position[axis] |= (zvalue[N * i + axis] as usize) << (level - 1 - i);
            }
        }
        UniformId { level, position }
    }

    /// Returns the cell which owns the given uniform grid point.
    pub fn cell_from_uniform_index(&self, uniform: UniformId<N>) -> CellId {
        debug_assert!(uniform.is_valid());

        // Start with root.
        let mut cell = CellId(0);
        let mut position = [0usize; N];

        while let Some(children) = self.child_offset(cell) {
            // Level of bottom-left child.
            let level = self.level(children);
            // Position of bottom-left child.
            position = array::from_fn(|i| 2 * position[i]);
            // If we have progressed so far that this child is actually more refined than the grid,
            // just return the most recent cell.
            if uniform.level < level {
                break;
            }
            debug_assert!(uniform.level >= level);

            // Width of child on the uniform grid.
            let child_width = 1 << (uniform.level - level);

            let split = Split::<N>::pack(array::from_fn(|i| {
                uniform.position[i] >= (position[i] + 1) * child_width
            }));

            for i in 0..N {
                position[i] += split.is_set(i) as usize;
            }

            cell = CellId::child(children, split);
        }

        cell
    }

    /// Returns the cell which owns the given uniform grid point. shortening this search
    /// with an initial guess. Rather than operating in O(log N) time, this approaches
    /// O(1) if the guess is sufficiently close.
    pub fn cell_from_uniform_index_cached(
        &self,
        uniform: UniformId<N>,
        mut cache: CellId,
    ) -> CellId {
        debug_assert!(uniform.is_valid());

        let mut grid = self.uniform_index_from_cell(cache);

        while !grid.contains(uniform) {
            cache = self.parent(cache).unwrap();
            grid = grid.coarsened();
        }

        // Start with root.
        let mut cell = cache;
        let mut position = grid.position;

        while let Some(children) = self.child_offset(cell) {
            // Level of bottom-left child.
            let level = self.level(children);
            // Position of bottom-left child.
            position = array::from_fn(|i| 2 * position[i]);
            // If we have progressed so far that this child is actually more refined than the grid,
            // just return the most recent cell.
            if uniform.level < level {
                break;
            }
            debug_assert!(uniform.level >= level);

            // Width of child on the uniform grid.
            let child_width = 1 << (uniform.level - level);

            let split = Split::<N>::pack(array::from_fn(|i| {
                uniform.position[i] >= (position[i] + 1) * child_width
            }));

            for i in 0..N {
                position[i] += split.is_set(i) as usize;
            }

            cell = CellId::child(children, split);
        }

        cell
    }

    /// Returns the neighboring cell along the given face. If the neighboring cell is more refined, this
    /// returns the cell index of the adjacent cell with `tree.level(neighbor) == tree.level(cell)`.
    /// If this passes over a nonperiodic boundary then it returns `None`.
    pub fn neighbor_face(&self, cell: CellId, face: Face<N>) -> Option<CellId> {
        let mut region = Region::CENTRAL;
        region.set_side(face.axis, if face.side { Side::Right } else { Side::Left });
        self.neighbor_region(cell, region)
    }

    /// Returns the neighboring cell in the given region. If the neighboring cell is more refined, this
    /// returns the cell index of the adjacent cell with `tree.level(neighbor) == tree.level(cell)`.
    /// If this passes over a nonperiodic boundary then it returns `None`.
    pub fn neighbor_region(&self, cell: CellId, region: Region<N>) -> Option<CellId> {
        let leaf_offset = LeafId(self.cells[cell.0].leaf_offset);
        debug_assert!(self.leaf_level(leaf_offset) >= self.level(cell));

        let is_periodic =
            (0..N).all(|axis| region.side(axis) == Side::Middle || self.periodic[axis]);

        if cell == CellId::ROOT && is_periodic {
            return Some(CellId::ROOT);
        }

        let parent = self.parent(cell)?;
        debug_assert!(self.level(cell) > 0 && self.level(cell) == self.level(parent) + 1);
        let split = self.leaf_split(leaf_offset, self.level(parent));
        if split.is_inner_region(region) {
            let children = self.child_offset(parent).unwrap();
            return Some(CellId::child(children, split.as_outer_region(region)));
        }

        let mut parent_region = region;

        for axis in 0..N {
            // If on inside, set parent region to middle
            match (region.side(axis), split.is_set(axis)) {
                (Side::Left, true) | (Side::Right, false) => {
                    parent_region.set_side(axis, Side::Middle);
                }
                _ => {}
            }
        }

        let parent_neighbor = self.neighbor_region(parent, parent_region)?;

        let Some(parent_neighbor_children) = self.child_offset(parent_neighbor) else {
            return Some(parent_neighbor);
        };

        let mut neighbor_split = split;

        for axis in 0..N {
            match (region.side(axis), split.is_set(axis)) {
                (Side::Left, false) | (Side::Right, true) => {
                    neighbor_split = neighbor_split.toggled(axis);
                }
                (Side::Left, true) | (Side::Right, false) => {
                    neighbor_split = neighbor_split.toggled(axis);
                }
                _ => {}
            }
        }

        Some(CellId::child(parent_neighbor_children, neighbor_split))
    }

    pub fn neighbor_region_alt(&self, cell: CellId, region: Region<N>) -> Option<CellId> {
        let UniformId { level, position } = self.uniform_index_from_cell(cell);
        let mut position: [isize; N] = std::array::from_fn(|axis| position[axis] as isize);

        for axis in 0..N {
            match region.side(axis) {
                Side::Left => position[axis] -= 1,
                Side::Middle => {}
                Side::Right => position[axis] += 1,
            }
        }

        for axis in 0..N {
            if self.periodic[axis] {
                position[axis] = position[axis].rem_euclid(1 << level);
            } else if position[axis] < 0 || position[axis] >= (1 << level) {
                return None;
            }
        }

        Some(self.cell_from_uniform_index_cached(
            UniformId {
                level,
                position: std::array::from_fn(|axis| position[axis] as usize),
            },
            cell,
        ))
    }

    /// Iterates over
    pub fn leaf_neighbors_in_region(
        &self,
        cell: CellId,
        region: Region<N>,
    ) -> impl Iterator<Item = LeafId> + '_ {
        let level = self.level(cell);

        self.neighbor_region(cell, region)
            .into_iter()
            .flat_map(move |neighbor| {
                self.contained_leaves(neighbor).filter(move |&active| {
                    for l in level..self.leaf_level(active) {
                        if !region
                            .reverse()
                            .is_split_adjacent(self.leaf_split(active, l))
                        {
                            return false;
                        }
                    }

                    true
                })
            })
    }

    pub fn leaf_neighborhood(&self, cell: LeafId) -> impl Iterator<Item = LeafId> + '_ {
        regions().flat_map(move |region| {
            self.leaf_neighbors_in_region(self.cell_from_leaf(cell), region)
        })
    }

    pub fn leaf_coarse_neighborhood(&self, cell: LeafId) -> impl Iterator<Item = LeafId> + '_ {
        regions().flat_map(move |region| {
            let neighbor = self.neighbor_region(self.cell_from_leaf(cell), region)?;
            if self.level(neighbor) < self.leaf_level(cell) {
                return self.leaf_from_cell(neighbor);
            }
            None
        })
    }

    /// Returns true if a face lies on a boundary.
    pub fn is_boundary_face(&self, cell: CellId, face: Face<N>) -> bool {
        let mut region = Region::CENTRAL;
        region.set_side(face.axis, if face.side { Side::Right } else { Side::Left });
        self.boundary_region(cell, region) != Region::CENTRAL
    }

    /// Given a neighboring region to a cell, determines which global region that
    /// belongs to (usually)
    pub fn boundary_region(&self, cell: CellId, region: Region<N>) -> Region<N> {
        // Get the active cell owned by this cell.
        let Some(active) = self.leaf_from_cell(cell) else {
            return region;
        };

        let mut result = region;
        let mut level = self.level(cell);

        while level > 0 && result != Region::CENTRAL {
            let split = self.leaf_split(active, level - 1);

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

impl<const N: usize> DataSize for Tree<N> {
    const IS_DYNAMIC: bool = true;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        self.leaf_offsets.estimate_heap_size()
            + self.leaf_values.capacity() / size_of::<usize>()
            + self.leaf_to_cell.estimate_heap_size()
            + self.level_offsets.estimate_heap_size()
            + self.cells.estimate_heap_size()
    }
}

/// Helper struct for serializing a tree while avoiding saving redundent data.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TreeSer<const N: usize> {
    domain: HyperBox<N>,
    #[serde(with = "crate::array")]
    periodic: [bool; N],
    leaf_values: BitVec<usize, Lsb0>,
    leaf_offsets: Vec<usize>,
}

impl<const N: usize> From<TreeSer<N>> for Tree<N> {
    fn from(value: TreeSer<N>) -> Self {
        let mut result = Tree {
            domain: value.domain,
            periodic: value.periodic,
            leaf_values: value.leaf_values,
            leaf_offsets: value.leaf_offsets,
            leaf_to_cell: Vec::default(),
            level_offsets: Vec::default(),
            cells: Vec::default(),
        };
        result.build();
        result
    }
}

impl<const N: usize> From<Tree<N>> for TreeSer<N> {
    fn from(value: Tree<N>) -> Self {
        Self {
            domain: value.domain,
            periodic: value.periodic,
            leaf_values: value.leaf_values,
            leaf_offsets: value.leaf_offsets,
        }
    }
}

impl<const N: usize> Default for TreeSer<N> {
    fn default() -> Self {
        Self {
            domain: HyperBox::UNIT,
            periodic: [false; N],
            leaf_values: BitVec::default(),
            leaf_offsets: Vec::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbors() {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);

        assert_eq!(tree.bounds(CellId::ROOT), HyperBox::UNIT);
        assert_eq!(tree.num_cells(), 1);
        assert_eq!(tree.num_leaves(), 1);
        assert_eq!(tree.num_levels(), 1);

        assert_eq!(tree.neighbor_face(CellId::ROOT, Face::negative(0)), None);

        tree.refine(&[true]);
        tree.build();

        assert_eq!(tree.num_cells(), 5);
        assert_eq!(tree.num_leaves(), 4);
        assert_eq!(tree.num_levels(), 2);
        for split in Split::enumerate() {
            assert_eq!(tree.leaf_split(LeafId(split.to_linear()), 0), split);
        }
        for i in 0..4 {
            assert_eq!(tree.cell_from_leaf(LeafId(i)), CellId(i + 1));
        }

        tree.refine(&[true, false, false, false]);
        tree.build();

        assert_eq!(tree.cell_from_leaf(LeafId(0)), CellId(5));

        assert!(tree.is_boundary_face(CellId(5), Face::negative(0)));
        assert!(tree.is_boundary_face(CellId(5), Face::negative(1)));
        assert_eq!(
            tree.boundary_region(CellId(5), Region::new([Side::Left, Side::Right])),
            Region::new([Side::Left, Side::Middle])
        );

        assert_eq!(
            tree.neighbor_region(CellId(5), Region::new([Side::Right, Side::Right])),
            Some(CellId(8))
        );

        assert_eq!(
            tree.neighbor_region(CellId(4), Region::new([Side::Left, Side::Left])),
            Some(CellId(1))
        );
    }

    #[test]
    fn periodic_neighbors() {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);
        tree.set_periodic(0, true);
        tree.set_periodic(1, true);
        assert_eq!(
            tree.neighbor_face(CellId::ROOT, Face::negative(0)),
            Some(CellId::ROOT)
        );

        // Refine tree
        tree.refine(&[true]);
        tree.refine(&[true, false, false, false]);
        tree.build();

        assert_eq!(
            tree.neighbor_face(CellId(5), Face::negative(0)),
            Some(CellId(2))
        );
        assert_eq!(
            tree.neighbor_region(CellId(5), Region::new([Side::Left, Side::Left])),
            Some(CellId(4))
        );
    }

    #[test]
    fn active_neighbors_in_region() {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);
        // Refine tree
        tree.refine(&[true]);
        tree.refine(&[true, false, false, false]);
        tree.build();

        assert!(
            tree.leaf_neighbors_in_region(CellId(2), Region::new([Side::Left, Side::Middle]))
                .eq([LeafId(1), LeafId(3)].into_iter())
        );

        assert!(
            tree.leaf_neighbors_in_region(CellId(3), Region::new([Side::Middle, Side::Left]))
                .eq([LeafId(2), LeafId(3)].into_iter())
        );

        assert!(
            tree.leaf_neighbors_in_region(CellId(4), Region::new([Side::Left, Side::Left]))
                .eq([LeafId(3)].into_iter())
        );

        assert!(
            tree.leaf_neighbors_in_region(CellId(6), Region::new([Side::Right, Side::Right]))
                .eq([LeafId(4)].into_iter())
        );
    }

    #[test]
    fn refinement_and_coarsening() {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);
        tree.refine(&[true]);
        // Make initially asymmetric.
        tree.refine(&[true, false, false, false]);

        for _ in 0..1 {
            let mut flags: Vec<bool> = vec![true; tree.num_leaves()];
            tree.balance_refine_flags(&mut flags);
            tree.refine(&flags);
        }

        for _ in 0..2 {
            let mut flags = vec![true; tree.num_leaves()];
            tree.balance_coarsen_flags(&mut flags);
            let mut coarsen_map = vec![LeafId(0); tree.num_leaves()];
            tree.coarsen_leaf_index_map(&flags, &mut coarsen_map);
            tree.coarsen(&flags);
        }

        let mut other_tree = Tree::<2>::new(HyperBox::UNIT);
        other_tree.refine(&[true]);

        assert_eq!(tree, other_tree);
    }

    use rand::Rng;

    #[test]
    fn fuzz_serialize() -> eyre::Result<()> {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);

        // Randomly refine tree
        let mut rng = rand::rng();
        for _ in 0..4 {
            let mut flags = vec![false; tree.num_leaves()];
            rng.fill(flags.as_mut_slice());

            tree.balance_coarsen_flags(&mut flags);
            tree.refine(&mut flags);
        }

        // Serialize tree
        let data = ron::to_string(&tree)?;
        let tree2: Tree<2> = ron::from_str(data.as_str())?;

        assert_eq!(tree, tree2);

        Ok(())
    }

    #[test]
    fn cell_from_point() -> eyre::Result<()> {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);
        tree.refine(&[true]);
        tree.refine(&[true, false, false, false]);

        assert_eq!(tree.cell_from_point([0.0, 0.0]), CellId(5));
        assert_eq!(tree.leaf_from_cell(CellId(5)), Some(LeafId(0)));

        assert_eq!(tree.cell_from_point([0.51, 0.67]), CellId(4));
        assert_eq!(tree.leaf_from_cell(CellId(4)), Some(LeafId(6)));

        let mut rng = rand::rng();
        for _ in 0..50 {
            let x: f64 = rng.random_range(0.0..1.0);
            let y: f64 = rng.random_range(0.0..1.0);

            let cache: usize = rng.random_range(..tree.num_cells());

            assert_eq!(
                tree.cell_from_point_cached([x, y], CellId(cache)),
                tree.cell_from_point([x, y])
            );
        }

        Ok(())
    }

    #[test]
    fn cell_from_uniform_index() -> eyre::Result<()> {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);
        tree.refine(&[true]);
        tree.refine(&[true, true, false, true]);
        tree.refine(&[
            false, true, false, false, // Bottom left
            true, false, false, false, // Bottom right
            false, // Top left
            false, false, false, false, // Top right
        ]);

        // Level 0
        assert_eq!(
            tree.cell_from_uniform_index(UniformId {
                level: 0,
                position: [0, 0]
            }),
            CellId(0)
        );

        // Level 1
        let level1 = [1, 2, 3, 4];
        for column in 0..2 {
            for row in 0..2 {
                let position = [row, column];
                let index = column * 2 + row;

                assert_eq!(
                    tree.cell_from_uniform_index(UniformId { level: 1, position }),
                    CellId(level1[index])
                )
            }
        }

        // Level 2
        let level2 = [
            5, 6, 9, 10, //
            7, 8, 11, 12, //
            3, 3, 13, 14, //
            3, 3, 15, 16, //
        ];

        for row in 0..4 {
            for column in 0..4 {
                let position = [column, row];
                let index = row * 4 + column;

                assert_eq!(
                    tree.cell_from_uniform_index(UniformId { level: 2, position }),
                    CellId(level2[index])
                )
            }
        }

        // #[rustfmt::skip]
        let level3 = [
            5, 5, 17, 18, 21, 22, 10, 10, //
            5, 5, 19, 20, 23, 24, 10, 10, //
            7, 7, 8, 8, 11, 11, 12, 12, //
            7, 7, 8, 8, 11, 11, 12, 12, //
            3, 3, 3, 3, 13, 13, 14, 14, //
            3, 3, 3, 3, 13, 13, 14, 14, //
            3, 3, 3, 3, 15, 15, 16, 16, //
            3, 3, 3, 3, 15, 15, 16, 16, //
        ];

        for row in 0..8 {
            for column in 0..8 {
                let position = [column, row];
                let index = row * 8 + column;

                assert_eq!(
                    tree.cell_from_uniform_index(UniformId { level: 3, position }),
                    CellId(level3[index])
                )
            }
        }

        Ok(())
    }

    #[test]
    fn cell_from_uniform_index_cached() -> eyre::Result<()> {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);
        tree.refine(&[true]);
        tree.refine(&[true, true, true, true]);

        let mut rng = rand::rng();

        for _ in 0..3 {
            let mut flags: Vec<bool> = std::iter::repeat_with(|| rng.random())
                .take(tree.num_leaves())
                .collect();
            tree.balance_refine_flags(&mut flags);
            tree.refine(&flags);
        }

        for _ in 0..50 {
            let level = rng.random_range(0..tree.num_levels());
            let max = 1usize << level;
            let x = rng.random_range(0..max);
            let y = rng.random_range(0..max);

            let uniform = UniformId {
                level,
                position: [x, y],
            };

            let cache: usize = rng.random_range(0..tree.num_cells());

            assert_eq!(
                tree.cell_from_uniform_index_cached(uniform, CellId(cache)),
                tree.cell_from_uniform_index(uniform)
            );
        }

        Ok(())
    }

    #[test]
    fn uniform_index_from_cell() -> eyre::Result<()> {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);
        tree.refine(&[true]);
        tree.refine(&[true, true, false, true]);
        tree.refine(&[
            false, true, false, false, // Bottom left
            true, false, false, false, // Bottom right
            false, // Top left
            false, false, false, false, // Top right
        ]);

        let compare = [
            (0, [0, 0]), // Root
            // Level 1
            (1, [0, 0]),
            (1, [1, 0]),
            (1, [0, 1]),
            (1, [1, 1]),
            // *******************
            // Level 2
            (2, [0, 0]),
            (2, [1, 0]),
            (2, [0, 1]),
            (2, [1, 1]),
            (2, [2, 0]),
            (2, [3, 0]),
            (2, [2, 1]),
            (2, [3, 1]),
            (2, [2, 2]),
            (2, [3, 2]),
            (2, [2, 3]),
            (2, [3, 3]),
            // *******************
            // Level 3
            (3, [2, 0]),
            (3, [3, 0]),
            (3, [2, 1]),
            (3, [3, 1]),
            (3, [4, 0]),
            (3, [5, 0]),
            (3, [4, 1]),
            (3, [5, 1]),
        ];

        for (index, &(level, position)) in compare.iter().enumerate() {
            assert_eq!(
                tree.uniform_index_from_cell(CellId(index)),
                UniformId { level, position }
            );
        }

        Ok(())
    }

    #[test]
    fn uniform_index_contains() -> eyre::Result<()> {
        let base = UniformId {
            level: 2,
            position: [2, 3],
        };

        assert!(base.coarsened().contains(base));
        assert!(base.contains(UniformId {
            level: 3,
            position: [4, 6]
        }));
        assert!(base.contains(UniformId {
            level: 3,
            position: [5, 6]
        }));
        assert!(base.contains(UniformId {
            level: 3,
            position: [4, 7]
        }));
        assert!(base.contains(UniformId {
            level: 3,
            position: [5, 7]
        }));
        assert!(!base.contains(UniformId {
            level: 3,
            position: [6, 7]
        }));

        assert!(base.contains(UniformId {
            level: 4,
            position: [10, 14]
        }));

        Ok(())
    }

    #[test]
    fn neighbor_region_compare() -> eyre::Result<()> {
        let mut tree = Tree::<2>::new(HyperBox::UNIT);
        let mut rng = rand::rng();
        for axis in 0..2 {
            tree.set_periodic(axis, rng.random());
        }
        tree.refine(&[true]);
        tree.refine(&[true, true, true, true]);

        for _ in 0..4 {
            let mut flags: Vec<bool> = std::iter::repeat_with(|| rng.random())
                .take(tree.num_leaves())
                .collect();
            tree.balance_refine_flags(&mut flags);
            tree.refine(&flags);
        }

        for _ in 0..100 {
            let region = Region::from_linear(rng.random_range(0..Region::<2>::COUNT));
            let cell = CellId(rng.random_range(0..tree.num_cells()));

            assert_eq!(
                tree.neighbor_region(cell, region),
                tree.neighbor_region_alt(cell, region)
            );
        }

        Ok(())
    }
}
