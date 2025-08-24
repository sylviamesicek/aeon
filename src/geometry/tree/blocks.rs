use std::{array, ops::Range};

use super::{ActiveCellId, Tree};
use crate::geometry::{Face, FaceMask, HyperBox, IndexSpace};
use bitvec::prelude::*;
use datasize::DataSize;

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct BlockId(pub usize);

/// Groups cells of a `Tree` into uniform blocks, for more efficient inter-cell communication and multithreading.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    block_active_indices: Vec<ActiveCellId>,
    /// The offsets for the aforementioned flattened list of lists.
    block_active_offsets: Vec<usize>,
    /// The physical bounds of each block.
    block_bounds: Vec<HyperBox<N>>,
    /// The level of refinement of each block.
    block_levels: Vec<usize>,
    /// Stores whether block face is on physical boundary.
    boundaries: BitVec,
    /// Number of subdivisions for each axis.
    #[serde(with = "crate::array")]
    width: [usize; N],
    /// Ghost vertices along each face.
    ghost: usize,
    /// Stores a map from blocks to ranges of vertices.
    offsets: Vec<usize>,
}

impl<const N: usize> TreeBlocks<N> {
    pub fn new(width: [usize; N], ghost: usize) -> Self {
        Self {
            active_cell_positions: Default::default(),
            active_cell_to_block: Default::default(),
            block_sizes: Default::default(),
            block_active_indices: Default::default(),
            block_active_offsets: Default::default(),
            block_bounds: Default::default(),
            block_levels: Default::default(),
            boundaries: Default::default(),
            width,
            ghost,
            offsets: Default::default(),
        }
    }

    /// Rebuilds the tree block structure from existing geometric information. Performs greedy meshing
    /// to group cells into blocks.
    pub fn build(&mut self, tree: &Tree<N>) {
        self.build_blocks(tree);
        self.build_bounds(tree);
        self.build_boundaries(tree);
        self.build_levels(tree);
        self.build_nodes();
    }

    // Number of blocks in the mesh.
    pub fn len(&self) -> usize {
        self.block_sizes.len()
    }

    pub fn indices(&self) -> impl Iterator<Item = BlockId> + use<N> {
        (0..self.len()).map(BlockId)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the cells associated with the given block.
    pub fn active_cells(&self, block: BlockId) -> &[ActiveCellId] {
        &self.block_active_indices
            [self.block_active_offsets[block.0]..self.block_active_offsets[block.0 + 1]]
    }

    /// Size of a given block, measured in cells.
    pub fn size(&self, block: BlockId) -> [usize; N] {
        self.block_sizes[block.0]
    }

    /// Number of nodes along each axis of a given block, not including ghost nodes.
    pub fn node_size(&self, block: BlockId) -> [usize; N] {
        array::from_fn(|axis| self.block_sizes[block.0][axis] * self.width[axis] + 1)
    }

    /// Returns the bounds of the given block.
    pub fn bounds(&self, block: BlockId) -> HyperBox<N> {
        self.block_bounds[block.0]
    }

    /// Returns the level of a block.
    pub fn level(&self, block: BlockId) -> usize {
        self.block_levels[block.0]
    }

    /// Returns boundary flags for a block.
    pub fn boundary_flags(&self, block: BlockId) -> FaceMask<N> {
        let mut flags = [[false; 2]; N];

        for face in Face::<N>::iterate() {
            flags[face.axis][face.side as usize] =
                self.boundaries[block.0 * 2 * N + face.to_linear()];
        }

        FaceMask::pack(flags)
    }

    /// Returns the position of the cell within the block.
    pub fn active_cell_position(&self, cell: ActiveCellId) -> [usize; N] {
        self.active_cell_positions[cell.0]
    }

    /// Retrieves the block associated with a given active cell.
    pub fn active_cell_block(&self, cell: ActiveCellId) -> BlockId {
        BlockId(self.active_cell_to_block[cell.0])
    }

    /// The width of each cell along each axis.
    pub fn width(&self) -> [usize; N] {
        self.width
    }

    /// Number of ghost nodes on each face.
    pub fn ghost(&self) -> usize {
        self.ghost
    }

    // /// Sets the width of cells in the block.
    // pub fn set_width(&mut self, width: [usize; N]) {
    //     self.width = width;
    // }

    // /// Sets the number of ghost nodes in cells in the block.
    // pub fn set_ghost(&mut self, ghost: usize) {
    //     self.ghost = ghost;
    // }

    /// Returns the total number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        *self.offsets.last().unwrap()
    }

    /// The range of dofs associated with the given block.
    pub fn nodes(&self, block: BlockId) -> Range<usize> {
        self.offsets[block.0]..self.offsets[block.0 + 1]
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
        for active in tree.active_cell_indices() {
            if self.active_cell_to_block[active.0] != usize::MAX {
                // This cell already belongs to a block, continue.
                continue;
            }

            // Get index of next block
            let block = self.block_sizes.len();

            self.active_cell_positions[active.0] = [0; N];
            self.active_cell_to_block[active.0] = block;

            self.block_sizes.push([1; N]);
            let block_cell_offset = self.block_active_indices.len();

            self.block_active_offsets.push(block_cell_offset);
            self.block_active_indices.push(active);

            // Try expanding the block along each axis.
            for axis in 0..N {
                // Perform greedy meshing.
                'expand: loop {
                    let face = Face::<N>::positive(axis);

                    let size = self.block_sizes[block];
                    let space = IndexSpace::new(size);

                    // Make sure every cell on face is suitable for expansion.
                    for index in space.face_window(Face::positive(axis)).iter() {
                        // Retrieves the cell on this face
                        let cell = tree.cell_from_active_index(
                            self.block_active_indices
                                [block_cell_offset + space.linear_from_cartesian(index)],
                        );
                        let level = tree.level(cell);
                        // We can only expand if:
                        // 1. We are not on a physical boundary.
                        // 2. We did not pass over a periodic boundary.
                        // 3. The neighbor is the same level of refinement.
                        // 4. The neighbor does not already belong to another block.
                        let Some(neighbor) = tree.neighbor(cell, face) else {
                            break 'expand;
                        };

                        if tree.is_boundary_face(cell, face) {
                            break 'expand;
                        }

                        if level != tree.level(neighbor) || !tree.is_active(neighbor) {
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
                    for index in space.face_window(Face::positive(axis)).iter() {
                        let active = self.block_active_indices
                            [block_cell_offset + space.linear_from_cartesian(index)];

                        let cell = tree.cell_from_active_index(active);
                        let cell_neighbor = tree.neighbor(cell, face).unwrap();
                        debug_assert!(tree.is_active(cell_neighbor));
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

        for block in self.indices() {
            let size = self.size(block);
            let a = *self.active_cells(block).first().unwrap();

            let cell_bounds = tree.bounds(tree.cell_from_active_index(a));

            self.block_bounds.push(HyperBox {
                origin: cell_bounds.origin,
                size: array::from_fn(|axis| cell_bounds.size[axis] * size[axis] as f64),
            })
        }
    }

    fn build_boundaries(&mut self, tree: &Tree<N>) {
        self.boundaries.clear();

        for block in self.indices() {
            let a = 0;
            let b: usize = self.active_cells(block).len() - 1;

            for face in Face::<N>::iterate() {
                let active = if face.side {
                    self.active_cells(block)[b]
                } else {
                    self.active_cells(block)[a]
                };
                let cell = tree.cell_from_active_index(active);
                self.boundaries.push(tree.is_boundary_face(cell, face))
            }
        }
    }

    fn build_levels(&mut self, tree: &Tree<N>) {
        self.block_levels.resize(self.len(), 0);
        for block in self.indices() {
            let active = self.active_cells(block)[0];
            self.block_levels[block.0] = tree.active_level(active);
        }
    }

    fn build_nodes(&mut self) {
        for axis in 0..N {
            assert!(self.width[axis] % 2 == 0);
        }

        // Reset map
        self.offsets.clear();
        self.offsets.reserve(self.len() + 1);

        // Start cursor at 0.
        let mut cursor = 0;
        self.offsets.push(cursor);

        for block in self.indices() {
            let size = self.size(block);
            // Width of block in nodes.
            let block_width: [usize; N] =
                array::from_fn(|axis| self.width[axis] * size[axis] + 1 + 2 * self.ghost);

            cursor += block_width.iter().product::<usize>();
            self.offsets.push(cursor);
        }
    }
}

impl<const N: usize> DataSize for TreeBlocks<N> {
    const IS_DYNAMIC: bool = true;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        self.active_cell_positions.estimate_heap_size()
            + self.active_cell_to_block.estimate_heap_size()
            + self.block_sizes.estimate_heap_size()
            + self.block_active_offsets.estimate_heap_size()
            + self.block_active_indices.estimate_heap_size()
            + self.block_bounds.estimate_heap_size()
            + self.block_levels.estimate_heap_size()
            + self.boundaries.capacity() / size_of::<usize>()
            + self.offsets.estimate_heap_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{geometry::Tree, prelude::HyperBox};

    #[test]
    fn greedy_meshing() {
        let mut tree = Tree::new(HyperBox::<2>::UNIT);
        tree.refine(&[true]);
        tree.refine(&[true, false, false, false]);

        let mut blocks = TreeBlocks::new([4; 2], 2);
        blocks.build(&tree);

        assert!(blocks.len() == 3);
        assert_eq!(blocks.level(BlockId(0)), 2);
        assert_eq!(blocks.size(BlockId(0)), [2, 2]);
        assert_eq!(
            blocks.active_cells(BlockId(0)),
            [
                ActiveCellId(0),
                ActiveCellId(1),
                ActiveCellId(2),
                ActiveCellId(3)
            ]
        );
        assert_eq!(blocks.level(BlockId(1)), 1);
        assert_eq!(blocks.size(BlockId(1)), [1, 2]);
        assert_eq!(
            blocks.active_cells(BlockId(1)),
            [ActiveCellId(4), ActiveCellId(6),]
        );
        assert_eq!(blocks.level(BlockId(2)), 1);
        assert_eq!(blocks.size(BlockId(2)), [1, 1]);
        assert_eq!(blocks.active_cells(BlockId(2)), [ActiveCellId(5)]);

        tree.refine(&[false, false, false, false, true, false, false]);
        blocks.build(&tree);
        assert!(blocks.len() == 2);
        assert_eq!(blocks.level(BlockId(0)), 2);
        assert_eq!(blocks.size(BlockId(0)), [4, 2]);
        assert_eq!(
            blocks.active_cells(BlockId(0)),
            [
                ActiveCellId(0),
                ActiveCellId(1),
                ActiveCellId(4),
                ActiveCellId(5),
                ActiveCellId(2),
                ActiveCellId(3),
                ActiveCellId(6),
                ActiveCellId(7),
            ]
        );
        assert_eq!(blocks.level(BlockId(1)), 1);
        assert_eq!(blocks.size(BlockId(1)), [2, 1]);
        assert_eq!(
            blocks.active_cells(BlockId(1)),
            [ActiveCellId(8), ActiveCellId(9),]
        );
    }

    #[test]
    fn node_ranges() {
        let mut tree = Tree::new(HyperBox::<2>::UNIT);
        let mut blocks = TreeBlocks::new([8; 2], 3);
        tree.refine(&[true]);
        tree.refine(&[true, false, false, false]);
        tree.build();

        blocks.build(&tree);

        assert_eq!(blocks.len(), 3);

        assert_eq!(blocks.nodes(BlockId(0)), 0..529);
        assert_eq!(blocks.nodes(BlockId(1)), 529..874);
        assert_eq!(blocks.nodes(BlockId(2)), 874..1099);
    }
}
