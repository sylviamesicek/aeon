use bitvec::prelude::BitVec;
use std::cmp::Ordering;
use std::{array::from_fn, ops::Range};

use crate::fd::{Boundary, BoundaryCondition, NodeSpace};
use crate::geometry::{
    faces, regions, Face, IndexSpace, Rectangle, Region, Side, SpatialTree, SPATIAL_BOUNDARY,
};

use super::node_from_vertex;

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

impl<const N: usize> Default for TreeBlocks<N> {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn build(&mut self, tree: &SpatialTree<N>, cell_width: [usize; N], ghost_nodes: usize) {
        self.build_blocks(tree);
        self.build_node_offsets(cell_width, ghost_nodes);
        self.build_boundaries(tree);
    }

    // Number of blocks in the mesh.
    pub fn len(&self) -> usize {
        self.block_sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Size of a given block, measured in cells.
    pub fn size(&self, block: usize) -> [usize; N] {
        self.block_sizes[block]
    }

    /// Map from cartesian indices within block to cell at the given positions
    pub fn cells(&self, block: usize) -> &[usize] {
        &self.block_cell_indices[self.block_cell_offsets[block]..self.block_cell_offsets[block + 1]]
    }

    /// Returns the range of nodes that this block owns.
    pub fn nodes(&self, block: usize) -> Range<usize> {
        self.block_node_offsets[block]..self.block_node_offsets[block + 1]
    }

    /// Computes the boundary condition that should be applied to a block, given the
    /// physical boundary conditions.
    pub fn boundary<'a, B: Boundary>(
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

        for block in 0..self.len() {
            let size = self.size(block);
            let space = IndexSpace::new(size);

            let a = 0;
            let b: usize = space.size().map(|size| size - 1).iter().product();

            for face in faces::<N>() {
                let cell = if face.side {
                    self.cells(block)[b]
                } else {
                    self.cells(block)[a]
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
    /// Information about blocks on the tree mesh.
    blocks: TreeBlocks<N>,
    /// Stores the neighboring cell for each valid region (rather than face, as is stored
    /// in the geometric tree).
    neighbors: Vec<usize>,
}

impl<const N: usize> TreeMesh<N> {
    pub fn new(bounds: Rectangle<N>, cell_width: [usize; N], ghost_nodes: usize) -> Self {
        for axis in 0..N {
            assert!(cell_width[axis] % 2 == 0);
        }

        let mut result = Self {
            tree: SpatialTree::new(bounds),
            ghost_nodes,
            cell_width,

            blocks: TreeBlocks::new(),
            neighbors: Vec::new(),
        };

        result.build();

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
        self.build();
    }

    pub fn build(&mut self) {
        self.blocks
            .build(&self.tree, self.cell_width, self.ghost_nodes);
        self.build_neighbors();
    }

    fn build_neighbors(&mut self) {
        // Reset and reserve
        self.neighbors.clear();
        self.neighbors
            .reserve(self.num_cells() * Region::<N>::COUNT);

        // Loop through every cell
        for cell in 0..self.num_cells() {
            let split = self.tree.split(cell);
            let level = self.tree.level(cell);

            'regions: for region in regions::<N>() {
                // Start with self
                let mut neighbor = cell;

                let mut neighbor_fine: bool = false;
                let mut neighbor_coarse: bool = false;

                // Iterate over faces
                for face in region.adjacent_faces() {
                    // Make sure face is compatible with split.
                    if neighbor_coarse && split.is_set(face.axis) != face.side {
                        continue;
                    }

                    // Snap origin to be adjacent to face
                    if neighbor_fine {
                        let mut neighbor_split = self.tree.split(neighbor);
                        let neighbor_origin = neighbor - neighbor_split.into_linear();
                        neighbor_split.set_to(face.axis, face.side);
                        neighbor = neighbor_origin + neighbor_split.into_linear();
                    }

                    // Get neighbor of current cell
                    let nneighbor = self.tree.neighbors(neighbor)[face.to_linear()];

                    // Short circut if we have encountered a boundary
                    if nneighbor == SPATIAL_BOUNDARY {
                        self.neighbors.push(SPATIAL_BOUNDARY);
                        continue 'regions;
                    }

                    // Get level of neighbor
                    let nlevel = self.tree.level(nneighbor);

                    match nlevel.cmp(&level) {
                        Ordering::Less => {
                            debug_assert!(nlevel == level - 1);
                            neighbor_coarse = true;
                        }
                        Ordering::Greater => {
                            debug_assert!(nlevel == level + 1);
                            neighbor_fine = true;
                        }
                        _ => {}
                    }

                    neighbor = nneighbor;
                }

                self.neighbors.push(cell);
            }
        }
    }

    /// Number of cells in the mesh.
    pub fn num_cells(&self) -> usize {
        self.tree.num_cells()
    }

    /// Number of blocks in the block.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Size of a given block, measured in cells.
    pub fn block_size(&self, block: usize) -> [usize; N] {
        self.blocks.size(block)
    }

    /// Map from cartesian indices within block to cell at the given positions
    pub fn block_cells(&self, block: usize) -> &[usize] {
        self.blocks.cells(block)
    }

    /// Computes the nodespace corresponding to a block.
    pub fn block_space<'a, B: Boundary>(
        &self,
        block: usize,
        physical: &'a B,
    ) -> NodeSpace<N, BlockBoundary<'a, N, B>> {
        let boundary = self.blocks.boundary(block, physical);
        let size = self.blocks.size(block);
        let cell_size = from_fn(|axis| size[axis] * self.cell_width[axis]);

        NodeSpace {
            size: cell_size,
            ghost: self.ghost_nodes,
            boundary,
        }
    }

    /// Retrieves the neighbors of the given cell for each region.
    pub fn cell_neighbors(&self, cell: usize) -> &[usize] {
        &self.neighbors[cell * Region::<N>::COUNT..(cell + 1) * Region::<N>::COUNT]
    }

    /// Retrieves the neighbors of the given cell for each region.
    pub fn cell_neighbor(&self, cell: usize, region: Region<N>) -> usize {
        self.cell_neighbors(cell)[region.to_linear()]
    }

    pub fn fill_boundary<const ORDER: usize>(&self, physical: &impl Boundary, field: &mut [f64]) {
        self.fill_direct(physical, field);
        self.fill_prolong::<ORDER>(physical, field);
    }

    fn fill_direct(&self, physical: &impl Boundary, field: &mut [f64]) {
        for block in 0..self.blocks.len() {
            // Fill Physical Boundary conditions
            let space = self.block_space(block, physical);
            let nodes = self.blocks.nodes(block);
            space.fill_boundary(&mut field[nodes.clone()]);
            // Fill Injection boundary conditions

            let size = self.blocks.size(block);
            let cells = self.blocks.cells(block);
            let cell_space = IndexSpace::new(size);

            for region in regions::<N>() {
                let offset_dir = region.offset_dir();

                let mut region_size: [_; N] = from_fn(|axis| self.cell_width[axis]);
                let mut coarse_region_size: [_; N] = from_fn(|axis| self.cell_width[axis]);

                for axis in 0..N {
                    if region.side(axis) == Side::Right {
                        region_size[axis] += 1;
                        coarse_region_size[axis] += 1;
                    }
                }

                for cell_index in region.face_vertices(size) {
                    let cell = cells[cell_space.linear_from_cartesian(cell_index)];
                    let cell_origin: [isize; N] = self.cell_node_origin(cell_index);

                    let neighbor = self.cell_neighbor(cell, region.clone());

                    // If physical boundary we skip
                    if neighbor == SPATIAL_BOUNDARY {
                        continue;
                    }

                    // If neighbor is coarser we skip
                    if self.tree.level(neighbor) < self.tree.level(cell) {
                        continue;
                    }

                    // If neighbor is more refined we use injection
                    if self.tree.level(neighbor) > self.tree.level(cell) {
                        for mask in region.adjacent_splits() {
                            let mut nmask = mask;

                            for axis in 0..N {
                                if region.side(axis) != Side::Middle {
                                    nmask.toggle(axis);
                                }
                            }

                            let neighbor = neighbor + nmask.into_linear();
                            let neighbor_index = self.blocks.cell_index(neighbor);
                            let neighbor_block = self.blocks.cell_block(neighbor);
                            let neighbor_origin = self.cell_node_origin(neighbor_index);

                            let neighbor_nodes = self.blocks.nodes(neighbor_block);

                            let neighbor_offset: [isize; N] = from_fn(|axis| {
                                neighbor_origin[axis]
                                    - offset_dir[axis] * self.cell_width[axis] as isize
                            });

                            let mut origin = cell_origin;

                            for axis in 0..N {
                                if mask.is_set(axis) {
                                    origin[axis] += (self.cell_width[axis] / 2) as isize;
                                }
                            }

                            for node in region.nodes(self.ghost_nodes, coarse_region_size).chain(
                                region
                                    .face_vertices(coarse_region_size)
                                    .map(node_from_vertex),
                            ) {
                                let source = from_fn(|axis| neighbor_offset[axis] + 2 * node[axis]);
                                let dest = from_fn(|axis| origin[axis] + node[axis]);

                                let v = space.value(source, &field[neighbor_nodes.clone()]);
                                space.set_value(dest, v, &mut field[nodes.clone()])
                            }
                        }

                        continue;
                    }

                    // Store various information about neighbor
                    let neighbor_index = self.blocks.cell_index(neighbor);
                    let neighbor_block = self.blocks.cell_block(neighbor);
                    let neighbor_origin: [isize; N] = self.cell_node_origin(neighbor_index);

                    let neighbor_nodes = self.blocks.nodes(neighbor_block);

                    let neighbor_offset: [isize; N] = from_fn(|axis| {
                        neighbor_origin[axis] - offset_dir[axis] * self.cell_width[axis] as isize
                    });

                    for node in region
                        .nodes(self.ghost_nodes, region_size)
                        .chain(region.face_vertices(region_size).map(node_from_vertex))
                    {
                        let source = from_fn(|axis| neighbor_offset[axis] + node[axis]);
                        let dest = from_fn(|axis| cell_origin[axis] + node[axis]);

                        let v = space.value(source, &field[neighbor_nodes.clone()]);
                        space.set_value(dest, v, &mut field[nodes.clone()]);
                    }
                }
            }
        }
    }

    fn fill_prolong<const ORDER: usize>(&self, physical: &impl Boundary, field: &mut [f64]) {
        for block in 0..self.blocks.len() {
            // Cache node space
            let space = self.block_space(block, physical);
            let nodes = self.blocks.nodes(block);
            // Fill Injection boundary conditions
            let size = self.blocks.size(block);
            let cells = self.blocks.cells(block);
            let cell_space = IndexSpace::new(size);

            for region in regions::<N>() {
                let offset_dir = region.offset_dir();

                let mut region_size: [_; N] = from_fn(|axis| self.cell_width[axis]);

                for axis in 0..N {
                    if region.side(axis) == Side::Right {
                        region_size[axis] += 1;
                    }
                }

                for cell_index in region.face_vertices(size) {
                    let cell = cells[cell_space.linear_from_cartesian(cell_index)];
                    let cell_origin = self.cell_node_origin(cell_index);
                    let cell_split = self.tree.split(cell);

                    let neighbor = self.cell_neighbor(cell, region.clone());

                    // If physical boundary we skip
                    if neighbor == SPATIAL_BOUNDARY {
                        continue;
                    }

                    // We only consider this neighbor if it is coarser
                    if self.tree.level(neighbor) >= self.tree.level(cell) {
                        continue;
                    }

                    let neighbor_index = self.blocks.cell_index(neighbor);
                    let neighbor_block = self.blocks.cell_block(neighbor);
                    let neighbor_nodes = self.blocks.nodes(neighbor_block);

                    let mut neighbor_split = cell_split;
                    for axis in 0..N {
                        if region.side(axis) != Side::Middle {
                            neighbor_split.toggle(axis);
                        }
                    }

                    let mut neighbor_origin: [isize; N] = self.cell_node_origin(neighbor_index);
                    for axis in 0..N {
                        if neighbor_split.is_set(axis) {
                            neighbor_origin[axis] += self.cell_width[axis] as isize / 2;
                        }
                    }

                    let neighbor_offset: [isize; N] = from_fn(|axis| {
                        neighbor_origin[axis]
                            - offset_dir[axis] * self.cell_width[axis] as isize / 2
                    });

                    for node in region.nodes(self.ghost_nodes, region_size) {
                        let source =
                            from_fn(|axis| (2 * neighbor_offset[axis] + node[axis]) as usize);
                        let dest = from_fn(|axis| cell_origin[axis] + node[axis]);

                        let v = space.prolong::<ORDER>(source, &field[neighbor_nodes.clone()]);
                        space.set_value(dest, v, &mut field[nodes.clone()])
                    }
                }
            }
        }
    }

    fn cell_node_origin(&self, index: [usize; N]) -> [isize; N] {
        from_fn(|axis| (index[axis] * self.cell_width[axis]) as isize)
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
