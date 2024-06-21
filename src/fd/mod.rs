use std::{array::from_fn, ops::Range};

use crate::geometry::{faces, regions, Face, IndexSpace, Rectangle, SpatialTree, SPATIAL_BOUNDARY};

mod boundary;
mod engine;
mod kernel;
mod node;

use bitvec::vec::BitVec;
pub use boundary::{Boundary, BoundaryCondition};
pub use engine::{Engine, FdEngine};
pub use kernel::{Interpolation, Operator, Order, Support};
pub use node::{node_from_vertex, NodeSpace};

pub struct BlockBoundary<'a, const N: usize, B> {
    inner: &'a B,
    flags: [[bool; 2]; N],
}

impl<'a, const N: usize, B: Boundary> Boundary for BlockBoundary<'a, N, B> {
    fn face(&self, face: Face) -> BoundaryCondition {
        if self.flags[face.axis][face.side as usize] {
            self.inner.face(face)
        } else {
            BoundaryCondition::Custom
        }
    }
}

#[derive(Debug, Clone)]
pub struct TreeBlocks<const N: usize> {
    /// Stores each cell's position within its parent's block.
    cell_indices: Vec<[usize; N]>,
    /// Maps cell to the block that contains it.
    cell_to_block: Vec<usize>,
    /// Stores the size of each block.
    block_sizes: Vec<[usize; N]>,
    block_cell_indices: Vec<usize>,
    block_cell_offsets: Vec<usize>,
    /// Maps each block to the nodes it owns.
    block_node_offsets: Vec<usize>,
    /// Stores whether block face is on physical boundary.
    boundaries: BitVec,
}

impl<const N: usize> TreeBlocks<N> {
    pub fn new() -> Self {
        Self {
            cell_indices: Vec::new(),
            cell_to_block: Vec::new(),

            block_sizes: Vec::new(),
            block_cell_indices: Vec::new(),
            block_cell_offsets: Vec::new(),
            block_node_offsets: Vec::new(),

            boundaries: BitVec::new(),
        }
    }

    pub fn reinit(&mut self, tree: &SpatialTree<N>, cell_width: [usize; N], ghost_nodes: usize) {
        self.build_blocks(tree);
        self.build_node_offsets(cell_width, ghost_nodes);
        self.build_boundaries(tree);
    }

    // Number of blocks in the block.
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

    /// Returns the range of nodes that this block owns.
    pub fn block_nodes(&self, block: usize) -> Range<usize> {
        self.block_node_offsets[block]..self.block_node_offsets[block + 1]
    }

    /// Computes the boundary condition that should be applied to a block, given the
    /// physical boundary conditions.
    pub fn block_boundary<'a, B: Boundary>(
        &self,
        block: usize,
        physical: &'a B,
    ) -> BlockBoundary<'a, N, B> {
        let mut flags = [[false; 2]; N];

        for face in faces::<N>() {
            flags[face.axis][face.side as usize] =
                self.boundaries[block * 2 * N + face.to_linear()];
        }

        BlockBoundary {
            inner: physical,
            flags,
        }
    }

    pub fn cell_index(&self, cell: usize) -> [usize; N] {
        self.cell_indices[cell]
    }

    pub fn cell_block(&self, cell: usize) -> usize {
        self.cell_to_block[cell]
    }

    fn build_blocks(&mut self, tree: &SpatialTree<N>) {
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
                    let face = Face::positive(axis);

                    let size = self.block_sizes[block];
                    let space = IndexSpace::new(size);

                    // Make sure every cell on face is suitable for expansion.
                    for index in space.face(Face::positive(axis)).iter() {
                        // Retrieves the cell on this face
                        let cell = self.block_cell_indices
                            [block_cell_offset + space.linear_from_cartesian(index)];
                        let level = tree.level(cell);
                        let neighbor = tree.neighbors(cell)[face.to_linear()];

                        // We can only expand if
                        // 1. This is not a physical boundary
                        // 2. The neighbor is the same level of refinement
                        // 3. The neighbor does not already belong to another block.
                        let is_suitable = neighbor != SPATIAL_BOUNDARY
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
                        let neighbor = tree.neighbors(cell)[face.to_linear()];

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

    fn build_node_offsets(&mut self, cell_width: [usize; N], ghost_nodes: usize) {
        // Reset map
        self.block_node_offsets.clear();
        self.block_node_offsets.reserve(self.block_sizes.len() + 1);

        // Start cursor at 0.
        let mut cursor = 0;
        self.block_node_offsets.push(cursor);

        for block in self.block_sizes.iter() {
            // Width of block in nodes.
            let block_width: [usize; N] =
                from_fn(|i| block[i] * cell_width[i] + 1 + 2 * ghost_nodes);

            cursor += block_width.iter().product::<usize>();
            self.block_node_offsets.push(cursor);
        }
    }

    fn build_boundaries(&mut self, tree: &SpatialTree<N>) {
        self.boundaries.clear();

        for block in 0..self.num_blocks() {
            let size = self.block_size(block);
            let space = IndexSpace::new(size);

            let a = 0;
            let b: usize = space.size().map(|size| size - 1).iter().product();

            for face in faces::<N>() {
                let cell = if face.side {
                    self.block_cells(block)[b]
                } else {
                    self.block_cells(block)[a]
                };
                let neighbor = tree.neighbors(cell)[face.to_linear()];
                self.boundaries.push(neighbor == SPATIAL_BOUNDARY);
            }
        }
    }
}

/// Implementation of an axis aligned tree mesh using standard finite difference operators.
#[derive(Debug, Clone)]
pub struct TreeMesh<const N: usize> {
    /// Tree of leaf cells.
    tree: SpatialTree<N>,

    /// Number of additional ghost nodes on each face.
    ghost_nodes: usize,
    /// Number of dof cells per mesh cell (along each axis).
    cell_width: [usize; N],

    blocks: TreeBlocks<N>,
}

impl<const N: usize> TreeMesh<N> {
    pub fn new(bounds: Rectangle<N>, cell_width: [usize; N], ghost_nodes: usize) -> Self {
        let mut result = Self {
            tree: SpatialTree::new(bounds),
            ghost_nodes,
            cell_width,

            blocks: TreeBlocks::new(),
        };

        result.blocks.reinit(&result.tree, cell_width, ghost_nodes);

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
        self.blocks
            .reinit(&self.tree, self.cell_width, self.ghost_nodes);
    }

    /// Number of cells in the mesh.
    pub fn num_cells(&self) -> usize {
        self.tree.num_cells()
    }

    /// Number of blocks in the block.
    pub fn num_blocks(&self) -> usize {
        self.blocks.num_blocks()
    }

    /// Size of a given block, measured in cells.
    pub fn block_size(&self, block: usize) -> [usize; N] {
        self.blocks.block_size(block)
    }

    /// Map from cartesian indices within block to cell at the given positions
    pub fn block_cells(&self, block: usize) -> &[usize] {
        self.blocks.block_cells(block)
    }

    /// Computes the nodespace corresponding to a block.
    pub fn block_space<'a, B: Boundary>(
        &self,
        block: usize,
        physical: &'a B,
    ) -> NodeSpace<N, BlockBoundary<'a, N, B>> {
        let boundary = self.blocks.block_boundary(block, physical);
        let size = self.blocks.block_size(block);
        let cell_size = from_fn(|axis| size[axis] * self.cell_width[axis]);

        NodeSpace {
            size: cell_size,
            ghost: self.ghost_nodes,
            boundary,
        }
    }

    pub fn fill_boundary(&self, physical: &impl Boundary, field: &mut [f64]) {
        self.fill_direct(physical, field);
        self.fill_prolong(field);
    }

    fn fill_direct(&self, physical: &impl Boundary, field: &mut [f64]) {
        let cell_node_size = self.cell_node_size();

        for block in 0..self.blocks.num_blocks() {
            // Fill Physical Boundary conditions
            let space = self.block_space(block, physical);
            let nodes = self.blocks.block_nodes(block);
            space.fill_boundary(&mut field[nodes.clone()]);
            // Fill Injection boundary conditions

            let size = self.blocks.block_size(block);
            let cells = self.blocks.block_cells(block);
            let cell_space = IndexSpace::new(size);

            for region in regions::<N>() {
                let offset_dir = region.offset_dir();

                for cell_index in region.face_vertices(size) {
                    let cell = cells[cell_space.linear_from_cartesian(cell_index)];
                    let cell_origin: [isize; N] = self.cell_node_origin(cell_index);

                    // TODO
                    let neighbor = SPATIAL_BOUNDARY;

                    // If physical boundary we skip
                    if neighbor == SPATIAL_BOUNDARY {
                        continue;
                    }

                    // If neighbor is coarser we skip
                    if self.tree.level(neighbor) < self.tree.level(cell) {
                        continue;
                    }

                    if self.tree.level(neighbor) > self.tree.level(cell) {
                        // TODO
                        continue;
                    }

                    // Store various information about neighbor
                    let neighbor_index = self.blocks.cell_index(neighbor);
                    let neighbor_block = self.blocks.cell_block(neighbor);
                    let neighbor_origin: [isize; N] = self.cell_node_origin(neighbor_index);

                    let neighbor_nodes = self.blocks.block_nodes(neighbor_block);

                    let neighbor_offset: [isize; N] = from_fn(|axis| {
                        neighbor_origin[axis] - offset_dir[axis] * cell_node_size[axis] as isize
                    });

                    for node in region.nodes(self.ghost_nodes, cell_node_size) {
                        let source = from_fn(|axis| neighbor_offset[axis] + node[axis]);
                        let dest = from_fn(|axis| cell_origin[axis] + node[axis]);

                        let v = space.value(source, &field[neighbor_nodes.clone()]);
                        space.set_value(dest, v, &mut field[nodes.clone()])
                    }
                }
            }
        }
    }

    fn fill_prolong(&self, _field: &mut [f64]) {}

    fn cell_node_origin(&self, index: [usize; N]) -> [isize; N] {
        from_fn(|axis| (index[axis] * self.cell_width[axis]) as isize)
    }

    fn cell_node_size(&self) -> [usize; N] {
        from_fn(|axis| self.cell_width[axis] + 1)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn greedy_meshing() {
        let mut mesh = TreeMesh::new(Rectangle::UNIT, [7; 2], 3);
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
