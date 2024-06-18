use std::array::from_fn;

use crate::{
    geometry::{Rectangle, SpatialTree, SPATIAL_BOUNDARY},
    prelude::{Face, IndexSpace},
};

mod kernel;

// pub use kernel::{Derivative, Dissipation, Kernel, SecondDerivative};

/// Implementation of an axis aligned tree mesh using standard finite difference operators.
#[derive(Debug, Clone)]
pub struct TreeMesh<const N: usize> {
    /// Tree of leaf cells.
    tree: SpatialTree<N>,

    /// Number of additional ghost nodes on each face.
    ghost_nodes: usize,
    /// Number of dof cells per mesh cell (along each axis).
    cell_width: [usize; N],

    /// Stores each cell's position within its parent's block.
    cell_positions: Vec<[usize; N]>,
    /// Maps cell to the block that contains it.
    cell_to_block: Vec<usize>,
    /// Stores the size of each block.
    block_sizes: Vec<[usize; N]>,
    block_cell_indices: Vec<usize>,
    block_cell_offsets: Vec<usize>,
    /// Maps each block to the nodes it owns.
    block_node_offsets: Vec<usize>,
}

impl<const N: usize> TreeMesh<N> {
    pub fn new(bounds: Rectangle<N>, ghost_nodes: usize, cell_width: [usize; N]) -> Self {
        let mut result = Self {
            tree: SpatialTree::new(bounds),
            ghost_nodes,
            cell_width,

            cell_positions: Vec::new(),
            cell_to_block: Vec::new(),

            block_sizes: Vec::new(),
            block_cell_indices: Vec::new(),
            block_cell_offsets: Vec::new(),
            block_node_offsets: Vec::new(),
        };

        result.build_blocks();
        result.build_node_offsets();

        result
    }

    pub fn is_balanced(&self, flags: &[bool]) -> bool {
        self.tree.is_balanced(flags)
    }

    /// Balances the given refinement flags.
    pub fn balance(&self, flags: &mut [bool]) {
        self.tree.balance(flags)
    }

    pub fn refine(&mut self, flags: &[bool]) {
        self.tree.refine(flags);
        self.build_blocks();
        self.build_node_offsets();
    }

    /// Number of cells in the mesh.
    pub fn num_cells(&self) -> usize {
        self.tree.num_cells()
    }

    /// Number of blocks in the block.
    pub fn num_blocks(&self) -> usize {
        self.block_sizes.len()
    }

    /// Size of a given block, measured in cells.
    pub fn block_size(&self, block: usize) -> [usize; N] {
        self.block_sizes[block]
    }

    /// Map from cartesian indices within block to cell at the given positions
    pub fn block_cells(&self, block: usize) -> &[usize] {
        &self.block_cell_indices[self.block_cell_offsets[block]..self.block_cell_offsets[block + 1]]
    }

    fn build_blocks(&mut self) {
        let num_cells = self.tree.num_cells();

        // Resize/reset various maps
        self.cell_positions.resize(num_cells, [0; N]);
        self.cell_positions.fill([0; N]);

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

            self.cell_positions[cell] = [0; N];
            self.cell_to_block[cell] = block;

            self.block_sizes.push([1; N]);
            let block_cell_offset = self.block_cell_indices.len();

            self.block_cell_offsets.push(block_cell_offset);
            self.block_cell_indices.push(cell);

            // Try expanding the block along each axis.
            for axis in 0..N {
                // Perform greedy meshing.
                'expand: loop {
                    let face = Face::positive(axis);

                    let size = self.block_sizes[block];
                    let space = IndexSpace::new(size);

                    // Make sure every cell on face is suitable for expansion.
                    for index in space.face(Face::positive(axis)).iter() {
                        // Retrieves the cell on this face
                        let cell = self.block_cell_indices
                            [block_cell_offset + space.linear_from_cartesian(index)];
                        let level = self.tree.level(cell);
                        let neighbor = self.tree.neighbors(cell)[face.to_linear()];

                        // We can only expand if
                        // 1. This is not a physical boundary
                        // 2. The neighbor is the same level of refinement
                        // 3. The neighbor does not already belong to another block.
                        let is_suitable = neighbor != SPATIAL_BOUNDARY
                            && level == self.tree.level(neighbor)
                            && self.cell_to_block[neighbor] == usize::MAX;

                        if !is_suitable {
                            break 'expand;
                        }
                    }

                    // We may now expand along this axis
                    for index in space.face(Face::positive(axis)).iter() {
                        let cell = self.block_cell_indices
                            [block_cell_offset + space.linear_from_cartesian(index)];
                        let neighbor = self.tree.neighbors(cell)[face.to_linear()];

                        self.cell_positions[neighbor] = index;
                        self.cell_positions[neighbor][axis] += 1;
                        self.cell_to_block[neighbor] = block;

                        self.block_cell_indices.push(neighbor);
                    }

                    self.block_sizes[block][axis] += 1;
                }
            }
        }

        self.block_cell_offsets.push(self.block_cell_indices.len());
    }

    fn build_node_offsets(&mut self) {
        // Reset map
        self.block_node_offsets.clear();
        self.block_node_offsets.reserve(self.block_sizes.len() + 1);

        // Start cursor at 0.
        let mut cursor = 0;
        self.block_node_offsets.push(cursor);

        for block in self.block_sizes.iter() {
            // Width of block in nodes.
            let block_width: [usize; N] =
                from_fn(|i| block[i] * self.cell_width[i] + 1 + 2 * self.ghost_nodes);

            cursor += block_width.iter().product::<usize>();
            self.block_node_offsets.push(cursor);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn greedy_meshing() {
        let mut mesh = TreeMesh::new(Rectangle::UNIT, 3, [7; 2]);
        mesh.refine(&[true, false, false, false]);

        assert_eq!(mesh.num_blocks(), 3);
        assert_eq!(mesh.block_size(0), [2, 2]);
        assert_eq!(mesh.block_size(1), [1, 2]);
        assert_eq!(mesh.block_size(2), [1, 1]);

        assert_eq!(mesh.block_cells(0), [0, 1, 2, 3]);
        assert_eq!(mesh.block_cells(1), [4, 6]);
        assert_eq!(mesh.block_cells(2), [5]);

        mesh.refine(&[false, false, false, false, true, false, false]);

        assert_eq!(mesh.num_blocks(), 2);
        assert_eq!(mesh.block_size(0), [4, 2]);
        assert_eq!(mesh.block_size(1), [2, 1]);

        assert_eq!(mesh.block_cells(0), [0, 1, 4, 5, 2, 3, 6, 7]);
        assert_eq!(mesh.block_cells(1), [8, 9]);
    }
}
