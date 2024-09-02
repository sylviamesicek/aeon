use crate::{faces, Face, FaceMask, IndexSpace, Rectangle, Tree, NULL};
use bitvec::prelude::*;
use std::array::from_fn;

/// Groups cells of a `Tree` into uniform blocks, for more efficient inter-cell communication and multithreading.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeBlocks<const N: usize> {
    /// Stores each cell's position within its parent's block.
    #[serde(with = "aeon_array::vec")]
    cell_indices: Vec<[usize; N]>,
    /// Maps cell to the block that contains it.
    cell_to_block: Vec<usize>,
    /// Stores the size of each block.
    #[serde(with = "aeon_array::vec")]
    block_sizes: Vec<[usize; N]>,
    block_cell_indices: Vec<usize>,
    block_cell_offsets: Vec<usize>,
    /// The physical bounds of each block.
    block_bounds: Vec<Rectangle<N>>,
    /// Stores whether block face is on physical boundary.
    boundaries: BitVec,
}

impl<const N: usize> Default for TreeBlocks<N> {
    fn default() -> Self {
        Self {
            cell_indices: Vec::new(),
            cell_to_block: Vec::new(),

            block_sizes: Vec::new(),
            block_cell_indices: Vec::new(),
            block_cell_offsets: Vec::new(),

            block_bounds: Vec::new(),

            boundaries: BitVec::new(),
        }
    }
}

impl<const N: usize> TreeBlocks<N> {
    /// Rebuilds the tree block structure from existing geometric information. Performs greedy meshing
    /// to group cells into blocks.
    pub fn build(&mut self, tree: &Tree<N>) {
        self.build_blocks(tree);
        self.build_bounds(tree);
        self.build_boundaries(tree);
    }

    // Number of blocks in the mesh.
    pub fn len(&self) -> usize {
        self.block_sizes.len()
    }

    /// Size of a given block, measured in cells.
    pub fn size(&self, block: usize) -> [usize; N] {
        self.block_sizes[block]
    }

    /// Returns the cells associated with the given block.
    pub fn cells(&self, block: usize) -> &[usize] {
        &self.block_cell_indices[self.block_cell_offsets[block]..self.block_cell_offsets[block + 1]]
    }

    /// Finds the cell at the given position within the block.
    pub fn cell_from_position(&self, block: usize, position: [usize; N]) -> usize {
        let space = IndexSpace::new(self.size(block));
        let index = space.linear_from_cartesian(position);

        self.block_cell_indices[self.block_cell_offsets[block] + index]
    }

    /// Returns the bounds of the given block.
    pub fn bounds(&self, block: usize) -> Rectangle<N> {
        self.block_bounds[block].clone()
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
    pub fn cell_position(&self, cell: usize) -> [usize; N] {
        self.cell_indices[cell]
    }

    pub fn cell_block(&self, cell: usize) -> usize {
        self.cell_to_block[cell]
    }

    fn build_blocks(&mut self, tree: &Tree<N>) {
        let num_cells = tree.num_cells();

        // Resize/reset various maps
        self.cell_indices.resize(num_cells, [0; N]);
        self.cell_indices.fill([0; N]);

        self.cell_to_block.resize(num_cells, usize::MAX);
        self.cell_to_block.fill(usize::MAX);

        self.block_sizes.clear();
        self.block_cell_indices.clear();
        self.block_cell_offsets.clear();

        // Loop over each cell in the tree
        for cell in 0..num_cells {
            if self.cell_to_block[cell] != usize::MAX {
                // This cell already belongs to a block, continue.
                continue;
            }

            // Get index of next block
            let block = self.block_sizes.len();

            self.cell_indices[cell] = [0; N];
            self.cell_to_block[cell] = block;

            self.block_sizes.push([1; N]);
            let block_cell_offset = self.block_cell_indices.len();

            self.block_cell_offsets.push(block_cell_offset);
            self.block_cell_indices.push(cell);

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
                        let cell = self.block_cell_indices
                            [block_cell_offset + space.linear_from_cartesian(index)];
                        let level = tree.level(cell);
                        let neighbor = tree.neighbor(cell, face);

                        // We can only expand if
                        // 1. This is not a physical boundary
                        // 2. The neighbor is the same level of refinement
                        // 3. The neighbor does not already belong to another block.
                        let is_suitable = neighbor != NULL
                            && level == tree.level(neighbor)
                            && self.cell_to_block[neighbor] == usize::MAX;

                        if !is_suitable {
                            break 'expand;
                        }
                    }

                    // We may now expand along this axis
                    for index in space.face(Face::positive(axis)).iter() {
                        let cell = self.block_cell_indices
                            [block_cell_offset + space.linear_from_cartesian(index)];
                        let neighbor = tree.neighbor(cell, face);

                        self.cell_indices[neighbor] = index;
                        self.cell_indices[neighbor][axis] += 1;
                        self.cell_to_block[neighbor] = block;

                        self.block_cell_indices.push(neighbor);
                    }

                    self.block_sizes[block][axis] += 1;
                }
            }
        }

        self.block_cell_offsets.push(self.block_cell_indices.len());
    }

    fn build_bounds(&mut self, tree: &Tree<N>) {
        self.block_bounds.clear();

        for block in 0..self.len() {
            let size = self.block_sizes[block];
            let a = self.cells(block).first().unwrap().clone();

            let cell_bounds = tree.bounds(a);

            self.block_bounds.push(Rectangle {
                origin: cell_bounds.origin,
                size: from_fn(|axis| cell_bounds.size[axis] * size[axis] as f64),
            })
        }
    }

    fn build_boundaries(&mut self, tree: &Tree<N>) {
        self.boundaries.clear();

        for block in 0..self.len() {
            let a = 0;
            let b: usize = self.cells(block).len() - 1;

            for face in faces::<N>() {
                let cell = if face.side {
                    self.cells(block)[b]
                } else {
                    self.cells(block)[a]
                };
                let neighbor = tree.neighbor(cell, face);
                self.boundaries.push(neighbor == NULL);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_meshing() {
        let mut tree = Tree::new(Rectangle::UNIT);
        tree.refine(&[true, false, false, false]);

        let mut blocks = TreeBlocks::default();
        blocks.build(&tree);

        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks.size(0), [2, 2]);
        assert_eq!(blocks.size(1), [1, 2]);
        assert_eq!(blocks.size(2), [1, 1]);

        assert_eq!(blocks.cells(0), [0, 1, 2, 3]);
        assert_eq!(blocks.cells(1), [4, 6]);
        assert_eq!(blocks.cells(2), [5]);

        tree.refine(&[false, false, false, false, true, false, false]);
        blocks.build(&tree);

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks.size(0), [4, 2]);
        assert_eq!(blocks.size(1), [2, 1]);

        assert_eq!(blocks.cells(0), [0, 1, 4, 5, 2, 3, 6, 7]);
        assert_eq!(blocks.cells(1), [8, 9]);
    }
}
