#![allow(clippy::needless_range_loop)]

use crate::array::{Array, ArrayLike};
use crate::geometry::{faces, AxisMask, Rectangle, Side};
use bitvec::prelude::*;
use std::array::from_fn;
use std::cmp::Ordering;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::slice;

use super::{regions, Face, FaceMask, IndexSpace, Region};

/// Denotes that the cell neighbors the physical boundary of a spatial domain.
pub const NULL: usize = usize::MAX;

/// An implementation of a spatial quadtree in any number of dimensions.
/// Only leaves are stored by this tree, and its hierarchical structure is implied.
/// To enable various optimisations, and avoid certain checks, the tree always contains
/// at least one level of refinement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "TreeSerde<N>")]
#[serde(into = "TreeSerde<N>")]
pub struct Tree<const N: usize> {
    /// Domain of the full tree
    domain: Rectangle<N>,
    /// Bounds of each cell in tree
    bounds: Vec<Rectangle<N>>,
    /// Stores neighbors along each face
    neighbors: Vec<usize>,
    /// Index within z-filling curve
    indices: [BitVec<usize, Lsb0>; N],
    /// Offsets into indices,
    offsets: Vec<usize>,
}

impl<const N: usize> Tree<N> {
    /// Constructs a new tree consisting of a single root cell that has been
    /// subdivided once.
    pub fn new(domain: Rectangle<N>) -> Self {
        let bounds = AxisMask::enumerate()
            .map(|mask| domain.split(mask))
            .collect();
        let neighbors = AxisMask::<N>::enumerate()
            .flat_map(|mask| {
                faces::<N>().map(move |face| {
                    if mask.is_inner_face(face) {
                        mask.toggled(face.axis).to_linear()
                    } else {
                        NULL
                    }
                })
            })
            .collect();

        let mut indices = from_fn(|_| BitVec::new());

        for mask in AxisMask::<N>::enumerate() {
            for axis in 0..N {
                indices[axis].push(mask.is_set(axis));
            }
        }

        let offsets = (0..=AxisMask::<N>::COUNT).collect();

        Self {
            domain,
            bounds,
            neighbors,
            indices,
            offsets,
        }
    }

    /// Number of cells in tree
    pub fn num_cells(&self) -> usize {
        self.bounds.len()
    }

    /// Returns neighbors for each face of the cell.
    pub fn neighbors(&self, cell: usize) -> &[usize] {
        &self.neighbors[cell * 2 * N..(cell + 1) * 2 * N]
    }

    /// Returns the neighbor of the cell along the given face.
    pub fn neighbor(&self, cell: usize, face: Face) -> usize {
        self.neighbors[cell * 2 * N + face.to_linear()]
    }

    /// Returns the neighbor of the cell in the given region.
    pub fn neighbor_region(&self, cell: usize, region: Region<N>) -> usize {
        let split = self.split(cell);
        let level = self.level(cell);

        // Start with self
        let mut neighbor = cell;

        let mut neighbor_fine: bool = false;
        let mut neighbor_coarse: bool = false;

        let mut rsplit = region.adjacent_split();

        // Iterate over faces
        for face in region.adjacent_faces() {
            // Make sure face is compatible with split.
            if neighbor_coarse && split.is_inner_face(face) {
                continue;
            }

            // Snap origin to be adjacent to face
            if neighbor_fine {
                if self.split(neighbor).is_inner_face(face) {
                    neighbor = self.neighbor(neighbor, face);
                }
                debug_assert!(self.split(neighbor).is_outer_face(face));
            }

            // Get neighbor of current cell
            let nneighbor = self.neighbor_after_refinement(neighbor, rsplit, face);
            rsplit.toggle(face.axis);

            // Short circut if we have encountered a boundary
            if nneighbor == NULL {
                return NULL;
            }

            // Get level of neighbor
            let nlevel = self.level(nneighbor);

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

        neighbor
    }

    /// Finds the outer neighbor along the face of the given subcell.
    pub fn neighbor_after_refinement(&self, cell: usize, split: AxisMask<N>, face: Face) -> usize {
        let mut result = self.neighbor(cell, face);

        if result == NULL {
            return NULL;
        }

        if self.level(result) <= self.level(cell) {
            return result;
        }

        (0..N)
            .into_iter()
            .filter(|&i| i != face.axis && split.is_set(i))
            .for_each(|i| {
                result = self.neighbor(result, Face::positive(i));
            });

        result
    }

    /// Computes the level of a cell.
    pub fn level(&self, cell: usize) -> usize {
        self.offsets[cell + 1] - self.offsets[cell]
    }

    /// Returns the domain of the full quadtree.
    pub fn domain(&self) -> Rectangle<N> {
        self.domain.clone()
    }

    /// Returns the bounds of a cell
    pub fn bounds(&self, cell: usize) -> Rectangle<N> {
        self.bounds[cell].clone()
    }

    /// If cell is not root, returns it most recent subdivision.
    pub fn split(&self, cell: usize) -> AxisMask<N> {
        AxisMask::pack(from_fn(|axis| self.indices[axis][self.offsets[cell]]))
    }

    /// Checks whether the given refinement flags are balanced.
    pub fn check_refine_flags(&self, flags: &[bool]) -> bool {
        assert!(flags.len() == self.num_cells());

        for block in 0..self.num_cells() {
            if !flags[block] {
                continue;
            }

            for coarse in self.coarse_neighborhood(block) {
                if !flags[coarse] {
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
        assert!(flags.len() == self.num_cells());

        loop {
            let mut is_balanced = true;

            for cell in 0..self.num_cells() {
                if !flags[cell] {
                    continue;
                }

                for coarse in self.coarse_neighborhood(cell) {
                    if !flags[coarse] {
                        is_balanced = false;
                        flags[coarse] = true;
                    }
                }
            }

            if is_balanced {
                break;
            }
        }
    }

    /// Fills the map with updated indices after refinement is performed.
    /// If a cell is refined, this will point to the base cell in that new subdivision.
    pub fn cell_map_after_refine(&self, flags: &[bool], map: &mut [usize]) {
        assert!(flags.len() == self.num_cells());
        assert!(map.len() == self.num_cells());

        let mut cursor = 0;

        for cell in 0..self.num_cells() {
            map[cell] = cursor;

            if flags[cell] {
                cursor += AxisMask::<N>::COUNT;
            } else {
                cursor += 1;
            }
        }
    }

    /// Refines the mesh using the given flags (temporary API).
    pub fn refine(&mut self, flags: &[bool]) {
        assert!(self.num_cells() == flags.len());
        assert!(self.check_refine_flags(flags));

        let num_flags = flags.iter().filter(|&&p| p).count();
        let total_blocks = self.num_cells() + (AxisMask::<N>::COUNT - 1) * num_flags;

        // ****************************************
        // Prepass to deteremine updated indices **

        // Maps current cell indices to new cell indices. If a cell is refined, this will point
        // to
        let mut update_map = Vec::with_capacity(self.num_cells());

        let mut cursor = 0;

        for &flag in flags.iter() {
            update_map.push(cursor);

            if flag {
                cursor += AxisMask::<N>::COUNT;
            } else {
                cursor += 1;
            }
        }

        let mut bounds = Vec::with_capacity(total_blocks);
        let mut neighbors = Vec::with_capacity(total_blocks * 2 * N);
        let mut indices = from_fn(|_| BitVec::with_capacity(total_blocks));
        let mut offsets = Vec::with_capacity(total_blocks + 1);
        offsets.push(0);

        for cell in 0..self.num_cells() {
            if flags[cell] {
                let parent = bounds.len();
                // Loop over subdivisions
                for split in AxisMask::<N>::enumerate() {
                    // Set physical bounds of block
                    bounds.push(self.bounds(cell).split(split));
                    // Update neighbors
                    for face in faces::<N>() {
                        if split.is_inner_face(face) {
                            // We can just reflect across axis within this particular group
                            // because we are uniformly refining this cell.
                            let neighbor = parent + split.toggled(face.axis).to_linear();
                            neighbors.push(neighbor);
                        } else {
                            // Gets the neighbor across this face, after any refinement is performed
                            let neighbor = self.neighbor_after_refinement(cell, split, face);

                            if neighbor == NULL {
                                // Propagate boundary information
                                neighbors.push(NULL);
                                continue;
                            }

                            let level = self.level(cell);
                            let neighbor_level = self.level(neighbor);

                            let neighbor =
                                match (flags[neighbor], level as isize - neighbor_level as isize) {
                                    (true, -1) => {
                                        // We refine both and the neighbor started finer
                                        update_map[neighbor]
                                            + face.reversed().adjacent_split::<N>().to_linear()
                                        // update_map[neighbor + mask.toggled(face.axis).to_linear()]
                                    }
                                    (false, -1) => {
                                        // Neighbor was more refined, but now they are on the same level
                                        update_map[neighbor]
                                    }
                                    (true, 0) => {
                                        // If we refine both and they started on the same level. simply flip
                                        // along axis.
                                        update_map[neighbor] + split.toggled(face.axis).to_linear()
                                    }
                                    (false, 0) => {
                                        // Started on same level, but neighbor was not refined
                                        update_map[neighbor]
                                    }
                                    (true, 1) => {
                                        // We refine both and this block starts finer than neighbor
                                        update_map[neighbor]
                                            + self.split(cell).toggled(face.axis).to_linear()
                                    }

                                    _ => panic!("Unbalanced quadtree."),
                                };

                            neighbors.push(neighbor);
                        }
                    }

                    // Compute index
                    for axis in 0..N {
                        // Multiply current index by 2 and set least significant bit
                        indices[axis].push(split.is_set(axis));
                        indices[axis].extend_from_bitslice(self.index_slice(cell, axis));
                    }
                    let previous = offsets[offsets.len() - 1];
                    offsets.push(previous + self.level(cell) + 1);
                }
            } else {
                // Set physical bounds of block
                bounds.push(self.bounds(cell));

                // Update neighbors
                for face in faces::<N>() {
                    let neighbor = self.neighbors(cell)[face.to_linear()];
                    if neighbor == NULL {
                        neighbors.push(NULL);
                        continue;
                    }

                    // Is the neighbor along this face being refined?
                    if flags[neighbor] {
                        // Find an adjacent split across face.
                        neighbors.push(
                            update_map[neighbor]
                                + face.reversed().adjacent_split::<N>().to_linear(),
                        );
                    } else {
                        neighbors.push(update_map[neighbor])
                    }
                }

                for axis in 0..N {
                    indices[axis].extend_from_bitslice(self.index_slice(cell, axis));
                }

                let previous = offsets[offsets.len() - 1];
                offsets.push(previous + self.level(cell));
            }
        }

        self.bounds = bounds;
        self.neighbors = neighbors;
        self.indices = indices;
        self.offsets = offsets;
    }

    /// Returns all coarse cells that neighbor the given cells.
    fn coarse_neighborhood(&self, cell: usize) -> impl Iterator<Item = usize> + '_ {
        let subdivision = self.split(cell);

        // Loop over possible directions
        AxisMask::<N>::enumerate()
            .skip(1)
            .map(move |dir| {
                // Get neighboring node
                let mut neighbor = cell;
                // Is this neighbor coarser than this node?
                let mut neighbor_coarse = false;
                // Loop over axes which can potentially be coarse
                for face in subdivision.outer_faces() {
                    if !dir.is_set(face.axis) {
                        continue;
                    }

                    // Get neighbor
                    let traverse = self.neighbors(neighbor)[face.to_linear()];

                    if traverse == NULL {
                        // No Update necessary if face is on boundary
                        neighbor_coarse = false;
                        neighbor = NULL;
                        break;
                    }

                    if self.level(traverse) < self.level(cell) {
                        neighbor_coarse = true;
                    }

                    // Update node
                    neighbor = traverse;
                }

                if neighbor_coarse {
                    neighbor
                } else {
                    NULL
                }
            })
            .filter(|&neighbor| neighbor != NULL)
    }

    /// Returns the z index for
    fn index_slice(&self, cell: usize, axis: usize) -> &BitSlice<usize, Lsb0> {
        &self.indices[axis][self.offsets[cell]..self.offsets[cell + 1]]
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TreeSerde<const N: usize> {
    domain: Rectangle<N>,
    /// Bounds of each cell in tree
    bounds: Vec<Rectangle<N>>,
    /// Stores neighbors along each face
    neighbors: Vec<usize>,
    /// Index within z-filling curve
    indices: Array<[BitVec<usize, Lsb0>; N]>,
    /// Offsets into indices,
    offsets: Vec<usize>,
}

impl<const N: usize> From<Tree<N>> for TreeSerde<N> {
    fn from(value: Tree<N>) -> Self {
        Self {
            domain: value.domain,
            bounds: value.bounds,
            neighbors: value.neighbors,
            indices: value.indices.into(),
            offsets: value.offsets,
        }
    }
}

impl<const N: usize> From<TreeSerde<N>> for Tree<N> {
    fn from(value: TreeSerde<N>) -> Self {
        Self {
            domain: value.domain,
            bounds: value.bounds,
            neighbors: value.neighbors,
            indices: value.indices.inner(),
            offsets: value.offsets,
        }
    }
}

// ******************************
// Blocks ***********************
// ******************************

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "TreeBlocksSerde<N>")]
#[serde(into = "TreeBlocksSerde<N>")]
pub struct TreeBlocks<const N: usize> {
    /// Stores each cell's position within its parent's block.
    cell_indices: Vec<[usize; N]>,
    /// Maps cell to the block that contains it.
    cell_to_block: Vec<usize>,
    /// Stores the size of each block.
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

    pub fn block_bounds(&self, block: usize) -> Rectangle<N> {
        self.block_bounds[block].clone()
    }

    pub fn block_boundary_flags(&self, block: usize) -> FaceMask<N> {
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

    fn build_bounds(&mut self, tree: &Tree<N>) {
        self.block_bounds.clear();

        for block in 0..self.num_blocks() {
            let size = self.block_sizes[block];
            let a = self.block_cells(block).first().unwrap().clone();

            let cell_bounds = tree.bounds(a);

            self.block_bounds.push(Rectangle {
                origin: cell_bounds.origin,
                size: from_fn(|axis| cell_bounds.size[axis] * size[axis] as f64),
            })
        }
    }

    fn build_boundaries(&mut self, tree: &Tree<N>) {
        self.boundaries.clear();

        for block in 0..self.num_blocks() {
            let a = 0;
            let b: usize = self.block_cells(block).len() - 1;

            for face in faces::<N>() {
                let cell = if face.side {
                    self.block_cells(block)[b]
                } else {
                    self.block_cells(block)[a]
                };
                let neighbor = tree.neighbors(cell)[face.to_linear()];
                self.boundaries.push(neighbor == NULL);
            }
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeBlocksSerde<const N: usize> {
    cell_indices: Vec<Array<[usize; N]>>,
    cell_to_block: Vec<usize>,
    block_sizes: Vec<Array<[usize; N]>>,
    block_cell_indices: Vec<usize>,
    block_cell_offsets: Vec<usize>,
    block_bounds: Vec<Rectangle<N>>,
    boundaries: BitVec,
}

impl<const N: usize> From<TreeBlocks<N>> for TreeBlocksSerde<N> {
    fn from(value: TreeBlocks<N>) -> Self {
        Self {
            cell_indices: wrap_vec_of_arrays(value.cell_indices),
            cell_to_block: value.cell_to_block,
            block_sizes: wrap_vec_of_arrays(value.block_sizes),
            block_cell_indices: value.block_cell_indices,
            block_cell_offsets: value.block_cell_offsets,
            block_bounds: value.block_bounds,
            boundaries: value.boundaries,
        }
    }
}

impl<const N: usize> From<TreeBlocksSerde<N>> for TreeBlocks<N> {
    fn from(value: TreeBlocksSerde<N>) -> Self {
        Self {
            cell_indices: unwrap_vec_of_arrays(value.cell_indices),
            cell_to_block: value.cell_to_block,
            block_sizes: unwrap_vec_of_arrays(value.block_sizes),
            block_cell_indices: value.block_cell_indices,
            block_cell_offsets: value.block_cell_offsets,
            block_bounds: value.block_bounds,
            boundaries: value.boundaries,
        }
    }
}

// ************************
// Neighbors **************
// ************************

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InterfaceKind {
    Coarse,
    Direct,
    Fine,
}

impl InterfaceKind {
    pub fn from_levels(level: usize, neighbor: usize) -> Self {
        match level as isize - neighbor as isize {
            1 => InterfaceKind::Coarse,
            0 => InterfaceKind::Direct,
            -1 => InterfaceKind::Fine,
            _ => panic!("Unbalanced levels"),
        }
    }
}

/// An interface between two blocks on a quad tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockInterface<const N: usize> {
    /// Target block.
    block: usize,
    /// Source block.
    neighbor: usize,
    /// Type of interface between blocks.
    interface: InterfaceKind,
    /// Source node on neighbor.
    source: [isize; N],
    /// Destination node on target.
    dest: [isize; N],
    /// Number of blocks to be filled.
    size: [usize; N],
}

/// Caches information about interior interfaces on quadtrees, specifically
/// storing information necessary for transferring data between blocks.
#[derive(Default, Clone)]
pub struct TreeInterfaces<const N: usize> {
    interfaces: Vec<BlockInterface<N>>,
}

impl<const N: usize> TreeInterfaces<N> {
    /// Iterates over all `BlockInterface`s.
    pub fn iter(&self) -> slice::Iter<'_, BlockInterface<N>> {
        self.interfaces.iter()
    }

    /// Rebuilds the block interface data.
    pub fn build(&mut self, tree: &Tree<N>, blocks: &TreeBlocks<N>, nodes: &TreeNodes<N>) {
        // Reused memory for neighbors.
        let mut neighbors = Vec::new();

        for block in 0..blocks.num_blocks() {
            // Cache block info
            let block_size = blocks.block_size(block);
            // Build cell neighbors.
            neighbors.clear();
            Self::build_cell_neighbors(tree, blocks, block, &mut neighbors);

            // Sort neighbors (to group cells from the same block together).
            neighbors.sort_unstable_by(|left, right| {
                let lblock = blocks.cell_block(left.neighbor);
                let rblock = blocks.cell_block(right.neighbor);

                lblock
                    .cmp(&rblock)
                    .then(left.neighbor.cmp(&right.neighbor))
                    .then(left.cell.cmp(&right.cell))
                    .then(left.region.cmp(&right.region))
            });

            Self::taverse_cell_neighbors(blocks, &mut neighbors, |neighbor, a, b| {
                // Find active region.
                let (anode, bnode) = Self::block_ghost_aabb(blocks, nodes, a, b);
                let mut source = Self::neighbor_origin(tree, blocks, nodes, a);
                let (mut dest, mut size) = Self::space_from_aabb(anode, bnode);

                // Avoid overlaps between aabbs on this block.
                let aorigin = blocks.cell_index(a.cell);
                let borigin = blocks.cell_index(b.cell);
                let flags = blocks.block_boundary_flags(block);

                for axis in 0..N {
                    let right_boundary = flags.is_set(Face::positive(axis));
                    let left_boundary = flags.is_set(Face::negative(axis));

                    // If the right edge doesn't extend all the way to the right,
                    // shrink by one.
                    if b.region.side(axis) == Side::Middle
                        && !(borigin[axis] == block_size[axis] - 1 && right_boundary)
                    {
                        size[axis] -= 1;
                    }

                    // If we do not extend further left, don't include
                    if a.region.side(axis) == Side::Middle && aorigin[axis] == 0 && !left_boundary {
                        source[axis] += 1;
                        dest[axis] += 1;
                        size[axis] -= 1;
                    }
                }

                // Compute this boundary interface.
                let interface =
                    InterfaceKind::from_levels(tree.level(a.cell), tree.level(a.neighbor));

                self.interfaces.push(BlockInterface {
                    block,
                    neighbor,
                    interface,
                    source,
                    dest,
                    size,
                });
            });
        }
    }

    /// Iterates the cell neighbors of a block, and pushes them onto the memory stack.
    fn build_cell_neighbors(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        block: usize,
        neighbors: &mut Vec<CellNeighbor<N>>,
    ) {
        let block_size = blocks.block_size(block);
        let block_cells = blocks.block_cells(block);

        for region in regions::<N>() {
            if region == Region::CENTRAL {
                continue;
            }

            let block_space = IndexSpace::new(block_size);

            // Find all cells adjacent to the given region.
            for index in block_space.adjacent(region) {
                let cell = block_cells[block_space.linear_from_cartesian(index)];
                let neighbor = tree.neighbor_region(cell, region);

                if neighbor == NULL {
                    continue;
                }

                neighbors.push(CellNeighbor {
                    cell,
                    neighbor,
                    region: region.clone(),
                })
            }
        }
    }

    /// Traverses a sorted list of cell neighbors, calling f once for each distinct block.
    fn taverse_cell_neighbors(
        blocks: &TreeBlocks<N>,
        neighbors: &mut [CellNeighbor<N>],
        mut f: impl FnMut(usize, CellNeighbor<N>, CellNeighbor<N>),
    ) {
        let mut neighbors = neighbors.iter().map(|n| n.clone()).peekable();

        while let Some(a) = neighbors.next() {
            let neighbor = blocks.cell_block(a.neighbor);

            // Next we walk through the iterator until we find the last neighbor that is still in this block.
            let mut b = a.clone();

            loop {
                if let Some(next) = neighbors.peek() {
                    if neighbor == blocks.cell_block(next.neighbor) {
                        b = neighbors.next().unwrap();
                        continue;
                    }
                }

                break;
            }

            f(neighbor, a, b)
        }
    }

    /// Computes the nodes that ghost nodes adjacent to the AABB of cells
    /// defined by A and B.
    fn block_ghost_aabb(
        blocks: &TreeBlocks<N>,
        nodes: &TreeNodes<N>,
        a: CellNeighbor<N>,
        b: CellNeighbor<N>,
    ) -> ([isize; N], [isize; N]) {
        // Find ghost region that must be filled
        let aindex = blocks.cell_index(a.cell);
        let bindex = blocks.cell_index(b.cell);

        // Compute bottom right corner of A cell.
        let mut anode: [_; N] = from_fn(|axis| (aindex[axis] * nodes.cell_width[axis]) as isize);

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match a.region.side(axis) {
                Side::Left => {
                    anode[axis] -= nodes.ghost_nodes as isize;
                }
                Side::Right => {
                    anode[axis] += nodes.cell_width[axis] as isize;
                }
                Side::Middle => {}
            }
        }

        // Compute top left corner of B cell
        let mut bnode: [_; N] =
            from_fn(|axis| ((bindex[axis] + 1) * nodes.cell_width[axis]) as isize);

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match b.region.side(axis) {
                Side::Right => {
                    bnode[axis] += nodes.ghost_nodes as isize;
                }
                Side::Left => {
                    bnode[axis] -= nodes.cell_width[axis] as isize;
                }
                Side::Middle => {}
            }
        }

        (anode, bnode)
    }

    /// Converts a window stored in aabb format to a window stored as an origin and a size.
    fn space_from_aabb(a: [isize; N], b: [isize; N]) -> ([isize; N], [usize; N]) {
        // Origin is just the bottom left corner of A.
        let dest = a;
        // Size is inclusive.
        let size = from_fn(|axis| (b[axis] - a[axis] + 1) as usize);

        (dest, size)
    }

    fn neighbor_origin(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        nodes: &TreeNodes<N>,
        a: CellNeighbor<N>,
    ) -> [isize; N] {
        // I think it works?

        // Compute this boundary interface.
        let interface = InterfaceKind::from_levels(tree.level(a.cell), tree.level(a.neighbor));

        // Find source node
        let index = blocks.cell_index(a.neighbor);
        let mut source: [isize; N] =
            from_fn(|axis| (index[axis] * nodes.cell_width[axis]) as isize);

        match interface {
            InterfaceKind::Direct => {
                for axis in 0..N {
                    if a.region.side(axis) == Side::Left {
                        source[axis] += (nodes.cell_width[axis] - nodes.ghost_nodes) as isize;
                    }
                }
            }
            InterfaceKind::Coarse => {
                // Source is stored in subnodes
                for axis in 0..N {
                    source[axis] *= 2;
                }

                let split = tree.split(a.cell);

                for axis in 0..N {
                    if split.is_set(axis) {
                        match a.region.side(axis) {
                            Side::Left => {
                                source[axis] +=
                                    nodes.cell_width[axis] as isize - nodes.ghost_nodes as isize
                            }
                            Side::Middle => source[axis] += nodes.cell_width[axis] as isize,
                            Side::Right => {}
                        }
                    } else {
                        match a.region.side(axis) {
                            Side::Left => {
                                source[axis] +=
                                    2 * nodes.cell_width[axis] as isize - nodes.ghost_nodes as isize
                            }
                            Side::Middle => {}
                            Side::Right => source[axis] += nodes.cell_width[axis] as isize,
                        }
                    }
                }
            }
            InterfaceKind::Fine => {
                // Source is stored in supernodes
                for axis in 0..N {
                    source[axis] /= 2;
                }

                for axis in 0..N {
                    if a.region.side(axis) == Side::Left {
                        source[axis] +=
                            nodes.cell_width[axis] as isize / 2 - nodes.ghost_nodes as isize;
                    }
                }
            }
        }

        source
    }
}

#[derive(Clone, Copy)]
struct CellNeighbor<const N: usize> {
    cell: usize,
    neighbor: usize,
    region: Region<N>,
}

/// Caches information about every neighbor touching each cell (including non-adjacent neighbors).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TreeNeighbors<const N: usize> {
    neighbors: Vec<usize>,
}

impl<const N: usize> TreeNeighbors<N> {
    /// Rebuilds the set of tree neighbors.
    pub fn build(&mut self, tree: &Tree<N>) {
        // Reset and reserve
        self.neighbors.clear();
        self.neighbors
            .reserve(tree.num_cells() * Region::<N>::COUNT);

        // Loop through every cell
        for cell in 0..tree.num_cells() {
            let split = tree.split(cell);
            let level = tree.level(cell);

            'regions: for region in regions::<N>() {
                // Start with self
                let mut neighbor = cell;

                let mut neighbor_fine: bool = false;
                let mut neighbor_coarse: bool = false;

                let mut rsplit = region.adjacent_split();

                // Iterate over faces
                for face in region.adjacent_faces() {
                    // Make sure face is compatible with split.
                    if neighbor_coarse && split.is_inner_face(face) {
                        continue;
                    }

                    // Snap origin to be adjacent to face
                    if neighbor_fine {
                        if tree.split(neighbor).is_inner_face(face) {
                            neighbor = tree.neighbor(neighbor, face);
                        }
                        debug_assert!(tree.split(neighbor).is_outer_face(face));
                    }

                    // Get neighbor of current cell
                    let nneighbor = tree.neighbor_after_refinement(neighbor, rsplit, face);
                    rsplit.toggle(face.axis);

                    // Short circut if we have encountered a boundary
                    if nneighbor == NULL {
                        self.neighbors.push(NULL);
                        continue 'regions;
                    }

                    // Get level of neighbor
                    let nlevel = tree.level(nneighbor);

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

    pub fn num_cells(&self) -> usize {
        self.neighbors.len() / Region::<N>::COUNT
    }

    /// Retrieves the neighbors of the given cell for each region.
    pub fn cell_neighbors(&self, cell: usize) -> &[usize] {
        &self.neighbors[cell * Region::<N>::COUNT..(cell + 1) * Region::<N>::COUNT]
    }

    /// Retrieves the neighbors of the given cell for each region.
    pub fn cell_neighbor(&self, cell: usize, region: Region<N>) -> usize {
        self.cell_neighbors(cell)[region.to_linear()]
    }
}

// ************************
// Nodes ******************
// ************************

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "TreeNodesSerde<N>")]
#[serde(into = "TreeNodesSerde<N>")]
pub struct TreeNodes<const N: usize> {
    pub cell_width: [usize; N],
    pub ghost_nodes: usize,
    node_offsets: Vec<usize>,
}

impl<const N: usize> TreeNodes<N> {
    pub fn new(cell_width: [usize; N], ghost_nodes: usize) -> Self {
        Self {
            cell_width,
            ghost_nodes,
            node_offsets: Vec::new(),
        }
    }

    pub fn cell_width(&self) -> [usize; N] {
        self.cell_width
    }

    pub fn ghost(&self) -> usize {
        self.ghost_nodes
    }

    /// Rebuilds the set of tree nodes.
    pub fn build(&mut self, blocks: &TreeBlocks<N>) {
        for axis in 0..N {
            assert!(self.cell_width[axis] % 2 == 0);
        }

        // Reset map
        self.node_offsets.clear();
        self.node_offsets.reserve(blocks.num_blocks() + 1);

        // Start cursor at 0.
        let mut cursor = 0;
        self.node_offsets.push(cursor);

        for block in 0..blocks.num_blocks() {
            let size = blocks.block_size(block);
            // Width of block in nodes.
            let block_width: [usize; N] =
                from_fn(|axis| self.cell_width[axis] * size[axis] + 1 + 2 * self.ghost_nodes);

            cursor += block_width.iter().product::<usize>();
            self.node_offsets.push(cursor);
        }
    }

    /// Returns the total number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.node_offsets.last().unwrap().clone()
    }

    /// The range of nodes associated with the given block.
    pub fn block_nodes(&self, block: usize) -> Range<usize> {
        self.node_offsets[block]..self.node_offsets[block + 1]
    }
}

impl<const N: usize> Default for TreeNodes<N> {
    fn default() -> Self {
        Self {
            cell_width: [2; N],
            ghost_nodes: 0,
            node_offsets: Default::default(),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct TreeNodesSerde<const N: usize> {
    cell_width: Array<[usize; N]>,
    ghost_nodes: usize,
    node_offsets: Vec<usize>,
}

impl<const N: usize> From<TreeNodes<N>> for TreeNodesSerde<N> {
    fn from(value: TreeNodes<N>) -> Self {
        Self {
            cell_width: value.cell_width.into(),
            ghost_nodes: value.ghost_nodes,
            node_offsets: value.node_offsets,
        }
    }
}

impl<const N: usize> From<TreeNodesSerde<N>> for TreeNodes<N> {
    fn from(value: TreeNodesSerde<N>) -> Self {
        Self {
            cell_width: value.cell_width.inner(),
            ghost_nodes: value.ghost_nodes,
            node_offsets: value.node_offsets,
        }
    }
}

/// Converts a vector of arraylike values to a vector of those values wrapped in the newtype
pub fn wrap_vec_of_arrays<I: ArrayLike>(vec: Vec<I>) -> Vec<Array<I>> {
    unsafe {
        let mut v = ManuallyDrop::new(vec);
        Vec::from_raw_parts(v.as_mut_ptr() as *mut Array<I>, v.len(), v.capacity())
    }
}

/// Converts a vector of wrapped arrays into a vector of their underlying arraylike values.
pub fn unwrap_vec_of_arrays<I: ArrayLike>(vec: Vec<Array<I>>) -> Vec<I> {
    unsafe {
        let mut v = ManuallyDrop::new(vec);
        Vec::from_raw_parts(v.as_mut_ptr() as *mut I, v.len(), v.capacity())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform() {
        let tree = Tree::new(Rectangle::<2>::UNIT);

        assert_eq!(tree.num_cells(), 4);
        assert_eq!(tree.neighbors(0), &[NULL, 1, NULL, 2]);
        assert_eq!(tree.neighbors(1), &[0, NULL, NULL, 3]);
        assert_eq!(tree.neighbors(2), &[NULL, 3, 0, NULL]);
        assert_eq!(tree.neighbors(3), &[2, NULL, 1, NULL]);

        assert_eq!(
            tree.bounds(0),
            Rectangle {
                origin: [0.0, 0.0],
                size: [0.5, 0.5]
            }
        );

        assert_eq!(
            tree.bounds(1),
            Rectangle {
                origin: [0.5, 0.0],
                size: [0.5, 0.5]
            }
        );

        assert_eq!(
            tree.bounds(2),
            Rectangle {
                origin: [0.0, 0.5],
                size: [0.5, 0.5]
            }
        );

        assert_eq!(
            tree.bounds(3),
            Rectangle {
                origin: [0.5, 0.5],
                size: [0.5, 0.5]
            }
        );

        let mut blocks = TreeBlocks::default();
        blocks.build(&tree);

        assert_eq!(blocks.num_blocks(), 1);
        assert_eq!(blocks.block_size(0), [2, 2]);
        assert_eq!(blocks.block_cells(0), &[0, 1, 2, 3]);

        assert_eq!(
            blocks.block_boundary_flags(0),
            FaceMask::pack([[true; 2]; 2])
        );
    }

    #[test]
    fn balancing() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);

        tree.refine(&[true, false, false, false]);

        // Produce unbalanced flags
        let mut flags = vec![false, false, false, true, false, false, false];
        assert!(!tree.check_refine_flags(&flags));

        // Perform rebalancing
        tree.balance_refine_flags(&mut flags);
        assert_eq!(flags, &[false, false, false, true, true, true, true]);

        assert!(tree.check_refine_flags(&flags));
    }

    #[test]
    fn refinement() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);

        assert_eq!(tree.num_cells(), 4);
        assert_eq!(tree.split(0).unpack(), [false, false]);
        assert_eq!(tree.split(1).unpack(), [true, false]);
        assert_eq!(tree.split(2).unpack(), [false, true]);
        assert_eq!(tree.split(3).unpack(), [true, true]);

        tree.refine(&[false, false, false, true]);

        assert_eq!(tree.num_cells(), 7);

        for block in 0..3 {
            assert_eq!(tree.level(block), 1);
        }
        for block in 3..7 {
            assert_eq!(tree.level(block), 2);
        }

        tree.refine(&[false, false, true, false, false, true, false]);

        assert_eq!(tree.num_cells(), 13);
        assert_eq!(tree.neighbors(0), &[NULL, 1, NULL, 2]);
        assert_eq!(tree.neighbors(2), &[NULL, 3, 0, 4]);
        assert_eq!(tree.neighbors(5), &[4, 8, 3, NULL]);
        assert_eq!(tree.neighbors(10), &[5, 11, 8, NULL]);
    }

    #[test]
    fn greedy_meshing() {
        let mut tree = Tree::new(Rectangle::UNIT);
        tree.refine(&[true, false, false, false]);

        let mut blocks = TreeBlocks::default();
        blocks.build(&tree);

        assert_eq!(blocks.num_blocks(), 3);
        assert_eq!(blocks.block_size(0), [2, 2]);
        assert_eq!(blocks.block_size(1), [1, 2]);
        assert_eq!(blocks.block_size(2), [1, 1]);

        assert_eq!(blocks.block_cells(0), [0, 1, 2, 3]);
        assert_eq!(blocks.block_cells(1), [4, 6]);
        assert_eq!(blocks.block_cells(2), [5]);

        tree.refine(&[false, false, false, false, true, false, false]);
        blocks.build(&tree);

        assert_eq!(blocks.num_blocks(), 2);
        assert_eq!(blocks.block_size(0), [4, 2]);
        assert_eq!(blocks.block_size(1), [2, 1]);

        assert_eq!(blocks.block_cells(0), [0, 1, 4, 5, 2, 3, 6, 7]);
        assert_eq!(blocks.block_cells(1), [8, 9]);
    }

    #[test]
    fn neighbor_regions() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut neighbors = TreeNeighbors::default();

        tree.refine(&[true, false, false, false]);
        neighbors.build(&tree);

        assert_eq!(tree.num_cells(), 7);

        let assert_eq_regions = |cell, values: [usize; 9]| {
            for region in regions::<2>() {
                assert_eq!(
                    tree.neighbor_region(cell, region),
                    values[region.to_linear()]
                );
            }
        };

        assert_eq_regions(0, [NULL, NULL, NULL, NULL, 0, 1, NULL, 2, 3]);
        assert_eq_regions(1, [NULL, NULL, NULL, 0, 1, 4, 2, 3, 4]);
        assert_eq_regions(2, [NULL, 0, 1, NULL, 2, 3, NULL, 5, 5]);
        assert_eq_regions(3, [0, 1, 4, 2, 3, 4, 5, 5, 6]);
        assert_eq_regions(4, [NULL, NULL, NULL, 1, 4, NULL, 5, 6, NULL]);
        assert_eq_regions(5, [NULL, 2, 4, NULL, 5, 6, NULL, NULL, NULL]);
        assert_eq_regions(6, [3, 4, NULL, 5, 6, NULL, NULL, NULL, NULL]);

        assert_eq!(
            tree.neighbor_after_refinement(4, AxisMask::pack([true, true]), Face::negative(0)),
            3
        );

        assert_eq!(
            tree.neighbor_after_refinement(5, AxisMask::pack([true, false]), Face::negative(1)),
            3
        );

        assert_eq!(
            tree.neighbor_after_refinement(6, AxisMask::pack([false, false]), Face::negative(0)),
            5
        );
    }

    #[test]
    fn node_offsets() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut nodes = TreeNodes::new([8; 2], 3);

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        nodes.build(&blocks);

        assert_eq!(blocks.num_blocks(), 3);

        assert_eq!(nodes.block_nodes(0), 0..529);
        assert_eq!(nodes.block_nodes(1), 529..874);
        assert_eq!(nodes.block_nodes(2), 874..1099);
    }

    #[test]
    fn neighbors2() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut nodes = TreeNodes::new([4; 2], 2);
        let mut interfaces = TreeInterfaces::default();

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        nodes.build(&blocks);
        interfaces.build(&tree, &blocks, &nodes);

        let mut interfaces = interfaces.iter();

        assert_eq!(
            interfaces.next(),
            Some(&BlockInterface {
                block: 0,
                neighbor: 1,
                interface: InterfaceKind::Coarse,
                source: [0, 0],
                dest: [8, 0],
                size: [3, 11],
            })
        );

        assert_eq!(
            interfaces.next(),
            Some(&BlockInterface {
                block: 0,
                neighbor: 2,
                interface: InterfaceKind::Coarse,
                source: [0, 0],
                dest: [0, 8],
                size: [8, 3],
            })
        );

        assert_eq!(
            interfaces.next(),
            Some(&BlockInterface {
                block: 1,
                neighbor: 0,
                interface: InterfaceKind::Fine,
                source: [2, 0],
                dest: [-2, 0],
                size: [3, 4],
            })
        );

        assert_eq!(
            interfaces.next(),
            Some(&BlockInterface {
                block: 1,
                neighbor: 2,
                interface: InterfaceKind::Direct,
                source: [2, 0],
                dest: [-2, 4],
                size: [3, 5],
            })
        );

        assert_eq!(
            interfaces.next(),
            Some(&BlockInterface {
                block: 2,
                neighbor: 0,
                interface: InterfaceKind::Fine,
                source: [0, 2],
                dest: [0, -2],
                size: [4, 3],
            })
        );

        assert_eq!(
            interfaces.next(),
            Some(&BlockInterface {
                block: 2,
                neighbor: 1,
                interface: InterfaceKind::Direct,
                source: [0, 2],
                dest: [4, -2],
                size: [3, 7],
            })
        );

        assert_eq!(interfaces.next(), None);
    }
}
