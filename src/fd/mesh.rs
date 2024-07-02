#![allow(dead_code)]

use bitvec::prelude::BitVec;
use std::cmp::Ordering;
use std::{array::from_fn, ops::Range};

use crate::array::{unwrap_vec_of_arrays, wrap_vec_of_arrays, Array};
use crate::fd::{Boundary, BoundaryKind, NodeSpace};
use crate::geometry::{
    faces, regions, AxisMask, Face, FaceMask, IndexSpace, Rectangle, Region, SpatialTree, NULL,
};

/// Implementation of an axis aligned tree mesh using standard finite difference operators.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "MeshSerde<N>")]
#[serde(into = "MeshSerde<N>")]
pub struct Mesh<const N: usize> {
    /// Tree of leaf cells.
    tree: SpatialTree<N>,
    /// Number of additional ghost nodes on each face.
    pub(crate) ghost_nodes: usize,
    /// Number of dof cells per mesh cell (along each axis).
    pub(crate) cell_width: [usize; N],
    /// Information about blocks on the tree mesh.
    pub(crate) blocks: Blocks<N>,
    /// Stores the neighboring cell for each valid region (rather than face, as is stored
    /// in the geometric tree).
    neighbors: Vec<usize>,
}

impl<const N: usize> Mesh<N> {
    /// Constructs a new tree mesh, covering the given physical domain. Each cell has the given number of subdivisions
    /// per axis, and each block extends out an extra `ghost_nodes` distance to facilitate inter-cell communication.
    pub fn new(bounds: Rectangle<N>, cell_width: [usize; N], ghost_nodes: usize) -> Self {
        for axis in 0..N {
            assert!(cell_width[axis] % 2 == 0);
        }

        let mut result = Self {
            tree: SpatialTree::new(bounds),
            ghost_nodes,
            cell_width,

            blocks: Blocks::new(),
            neighbors: Vec::new(),
        };

        result.build();

        result
    }

    /// Checks if the given refinement flags are 2:1 balanced.
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

    /// Reconstructs interal structure of the TreeMesh, automatically called during refinement.
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
                        let neighbor_origin = neighbor - neighbor_split.to_linear();
                        neighbor_split.set_to(face.axis, face.side);
                        neighbor = neighbor_origin + neighbor_split.to_linear();
                    }

                    // Get neighbor of current cell
                    let nneighbor = self.tree.neighbors(neighbor)[face.to_linear()];

                    // Short circut if we have encountered a boundary
                    if nneighbor == NULL {
                        self.neighbors.push(NULL);
                        continue 'regions;
                    }

                    // Get level of neighbor
                    let nlevel = self.tree.level(nneighbor);

                    match nlevel.cmp(&level) {
                        Ordering::Less => {
                            debug_assert!(nlevel == level - 1);
                            debug_assert!(!neighbor_fine);
                            neighbor_coarse = true;
                        }
                        Ordering::Greater => {
                            debug_assert!(nlevel == level + 1);
                            debug_assert!(!neighbor_coarse);
                            neighbor_fine = true;
                        }
                        _ => {}
                    }

                    neighbor = nneighbor;
                }

                self.neighbors.push(neighbor);
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

    pub fn block_bounds(&self, block: usize) -> Rectangle<N> {
        self.blocks.bounds(block)
    }

    pub fn block_nodes(&self, block: usize) -> Range<usize> {
        self.blocks.nodes(block)
    }

    pub fn block_boundary_flags(&self, block: usize) -> FaceMask<N> {
        self.blocks.boundary_flags(block)
    }

    pub fn block_boundary<B: Boundary>(&self, block: usize, boundary: B) -> BlockBoundary<N, B> {
        BlockBoundary {
            mask: self.block_boundary_flags(block),
            inner: boundary,
        }
    }

    /// Computes the nodespace corresponding to a block.
    pub fn block_space(&self, block: usize) -> NodeSpace<N> {
        let size = self.blocks.size(block);
        let cell_size = from_fn(|axis| size[axis] * self.cell_width[axis]);

        NodeSpace {
            size: cell_size,
            ghost: self.ghost_nodes,
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

    pub fn cell_level(&self, cell: usize) -> usize {
        self.tree.level(cell)
    }

    pub fn cell_split(&self, cell: usize) -> AxisMask<N> {
        self.tree.split(cell)
    }

    pub fn cell_block(&self, cell: usize) -> usize {
        self.blocks.cell_block(cell)
    }

    pub(crate) fn cell_node_origin(&self, index: [usize; N]) -> [isize; N] {
        from_fn(|axis| (index[axis] * self.cell_width[axis]) as isize)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MeshSerde<const N: usize> {
    /// Tree of leaf cells.
    tree: SpatialTree<N>,
    /// Number of additional ghost nodes on each face.
    ghost_nodes: usize,
    /// Number of dof cells per mesh cell (along each axis).
    cell_width: Array<[usize; N]>,
    /// Information about blocks on the tree mesh.
    blocks: Blocks<N>,
    /// Stores the neighboring cell for each valid region (rather than face, as is stored
    /// in the geometric tree).
    neighbors: Vec<usize>,
}

impl<const N: usize> From<Mesh<N>> for MeshSerde<N> {
    fn from(value: Mesh<N>) -> Self {
        Self {
            tree: value.tree,
            ghost_nodes: value.ghost_nodes,
            cell_width: value.cell_width.into(),
            blocks: value.blocks,
            neighbors: value.neighbors,
        }
    }
}

impl<const N: usize> From<MeshSerde<N>> for Mesh<N> {
    fn from(value: MeshSerde<N>) -> Self {
        Self {
            tree: value.tree,
            ghost_nodes: value.ghost_nodes,
            cell_width: value.cell_width.inner(),
            blocks: value.blocks,
            neighbors: value.neighbors,
        }
    }
}

// ******************************
// Blocks ***********************
// ******************************

#[derive(Debug, Clone)]
pub struct BlockBoundary<const N: usize, B> {
    inner: B,
    mask: FaceMask<N>,
}

impl<const N: usize, B: Boundary> Boundary for BlockBoundary<N, B> {
    fn kind(&self, face: Face) -> BoundaryKind {
        if self.mask.is_set(face) {
            self.inner.kind(face)
        } else {
            BoundaryKind::Custom
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "BlocksSerde<N>")]
#[serde(into = "BlocksSerde<N>")]
pub(crate) struct Blocks<const N: usize> {
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
    /// The physical bounds of each block.
    block_bounds: Vec<Rectangle<N>>,
    /// Stores whether block face is on physical boundary.
    boundaries: BitVec,
}

impl<const N: usize> Default for Blocks<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Blocks<N> {
    pub fn new() -> Self {
        Self {
            cell_indices: Vec::new(),
            cell_to_block: Vec::new(),

            block_sizes: Vec::new(),
            block_cell_indices: Vec::new(),
            block_cell_offsets: Vec::new(),
            block_node_offsets: Vec::new(),

            block_bounds: Vec::new(),

            boundaries: BitVec::new(),
        }
    }

    /// Rebuilds the tree block structure from existing geometric information. Performs greedy meshing
    /// to group cells into blocks.
    pub fn build(&mut self, tree: &SpatialTree<N>, cell_width: [usize; N], ghost_nodes: usize) {
        self.build_blocks(tree);
        self.build_bounds(tree);
        self.build_node_offsets(cell_width, ghost_nodes);
        self.build_boundaries(tree);
    }

    // Number of blocks in the mesh.
    pub fn len(&self) -> usize {
        self.block_sizes.len()
    }

    /// Returns true if the mesh contains no blocks.
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

    pub fn bounds(&self, block: usize) -> Rectangle<N> {
        self.block_bounds[block].clone()
    }

    pub fn boundary_flags(&self, block: usize) -> FaceMask<N> {
        let mut flags = [[false; 2]; N];

        for face in faces::<N>() {
            flags[face.axis][face.side as usize] =
                self.boundaries[block * 2 * N + face.to_linear()];
        }

        FaceMask::pack(flags)
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

    fn build_bounds(&mut self, tree: &SpatialTree<N>) {
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
                self.boundaries.push(neighbor == NULL);
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlocksSerde<const N: usize> {
    cell_indices: Vec<Array<[usize; N]>>,
    cell_to_block: Vec<usize>,
    block_sizes: Vec<Array<[usize; N]>>,
    block_cell_indices: Vec<usize>,
    block_cell_offsets: Vec<usize>,
    block_node_offsets: Vec<usize>,
    block_bounds: Vec<Rectangle<N>>,
    boundaries: BitVec,
}

impl<const N: usize> From<Blocks<N>> for BlocksSerde<N> {
    fn from(value: Blocks<N>) -> Self {
        Self {
            cell_indices: wrap_vec_of_arrays(value.cell_indices),
            cell_to_block: value.cell_to_block,
            block_sizes: wrap_vec_of_arrays(value.block_sizes),
            block_cell_indices: value.block_cell_indices,
            block_cell_offsets: value.block_cell_offsets,
            block_node_offsets: value.block_node_offsets,
            block_bounds: value.block_bounds,
            boundaries: value.boundaries,
        }
    }
}

impl<const N: usize> From<BlocksSerde<N>> for Blocks<N> {
    fn from(value: BlocksSerde<N>) -> Self {
        Self {
            cell_indices: unwrap_vec_of_arrays(value.cell_indices),
            cell_to_block: value.cell_to_block,
            block_sizes: unwrap_vec_of_arrays(value.block_sizes),
            block_cell_indices: value.block_cell_indices,
            block_cell_offsets: value.block_cell_offsets,
            block_node_offsets: value.block_node_offsets,
            block_bounds: value.block_bounds,
            boundaries: value.boundaries,
        }
    }
}

// ****************************************
// Tests **********************************
// ****************************************

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_meshing() {
        let mut mesh = Mesh::new(Rectangle::UNIT, [8; 2], 3);
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

    #[test]
    fn neighbors() {
        let mut mesh = Mesh::new(Rectangle::UNIT, [8; 2], 3);
        mesh.refine(&[true, false, false, false]);

        assert_eq!(mesh.num_cells(), 7);

        assert_eq!(
            mesh.cell_neighbors(0),
            [NULL, NULL, NULL, NULL, 0, 1, NULL, 2, 3]
        );
        assert_eq!(mesh.cell_neighbors(1), [NULL, NULL, NULL, 0, 1, 4, 2, 3, 4]);
        assert_eq!(mesh.cell_neighbors(2), [NULL, 0, 1, NULL, 2, 3, NULL, 5, 5]);
        assert_eq!(mesh.cell_neighbors(3), [0, 1, 4, 2, 3, 4, 5, 5, 6]);
        assert_eq!(
            mesh.cell_neighbors(4),
            [NULL, NULL, NULL, 0, 4, NULL, 5, 6, NULL]
        );
        assert_eq!(
            mesh.cell_neighbors(5),
            [NULL, 0, 4, NULL, 5, 6, NULL, NULL, NULL]
        );
        assert_eq!(
            mesh.cell_neighbors(6),
            [0, 4, NULL, 5, 6, NULL, NULL, NULL, NULL]
        );
    }

    #[test]
    fn node_offsets() {
        let mut mesh = Mesh::new(Rectangle::UNIT, [8; 2], 3);
        mesh.refine(&[true, false, false, false]);

        assert_eq!(mesh.num_blocks(), 3);

        assert_eq!(mesh.block_nodes(0), 0..529);
        assert_eq!(mesh.block_nodes(1), 529..874);
        assert_eq!(mesh.block_nodes(2), 874..1099);
    }
}
