#![allow(clippy::needless_range_loop)]

use crate::array::Array;
use crate::geometry::{faces, AxisMask, Rectangle};
use bitvec::prelude::*;
use std::array::from_fn;

/// Denotes that the cell neighbors the physical boundary of a spatial domain.
pub const SPATIAL_BOUNDARY: usize = usize::MAX;

/// An implementation of a spatial quadtree in any number of dimensions.
/// Only leaves are stored by this tree, and its hierarchical structure is implied.
/// To enable various optimisations, and avoid certain checks, the tree always contains
/// at least one level of refinement.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(from = "SpatialTreeSerde<N>")]
#[serde(into = "SpatialTreeSerde<N>")]
pub struct SpatialTree<const N: usize> {
    /// Bounds of each cell in tree
    bounds: Vec<Rectangle<N>>,
    /// Stores neighbors along each face
    neighbors: Vec<usize>,
    /// Index within z-filling curve
    indices: [BitVec<usize, Lsb0>; N],
    /// Offsets into indices,
    offsets: Vec<usize>,
}

impl<const N: usize> SpatialTree<N> {
    /// Constructs a new tree consisting of a single root cell that has been
    /// subdivided once.
    pub fn new(bounds: Rectangle<N>) -> Self {
        let bounds = AxisMask::enumerate()
            .map(|mask| bounds.split(mask))
            .collect();
        let neighbors = AxisMask::<N>::enumerate()
            .map(|mask| {
                faces::<N>().map(move |face| {
                    if mask.is_inner_face(face) {
                        mask.toggled(face.axis).into_linear()
                    } else {
                        SPATIAL_BOUNDARY
                    }
                })
            })
            .flatten()
            .collect();

        let mut indices = from_fn(|_| BitVec::new());

        for mask in AxisMask::<N>::enumerate() {
            for axis in 0..N {
                indices[axis].push(mask.is_set(axis));
            }
        }

        let offsets = (0..=AxisMask::<N>::COUNT).into_iter().collect();

        Self {
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

    /// Computes the level of a cell.
    pub fn level(&self, cell: usize) -> usize {
        self.offsets[cell + 1] - self.offsets[cell]
    }

    /// Returns the bounds of a cell
    pub fn bounds(&self, cell: usize) -> Rectangle<N> {
        self.bounds[cell].clone()
    }

    /// If block is not root, returns it most recent subdivision.
    pub fn subdivision(&self, block: usize) -> AxisMask<N> {
        AxisMask::pack(from_fn(|axis| self.indices[axis][self.offsets[block]]))
    }

    /// Checks whether the given refinement flags are balanced.
    pub fn is_balanced(&self, flags: &[bool]) -> bool {
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
    pub fn balance(&self, flags: &mut [bool]) {
        assert!(flags.len() == self.num_cells());

        loop {
            let mut is_balanced = true;

            for block in 0..self.num_cells() {
                if !flags[block] {
                    continue;
                }

                for coarse in self.coarse_neighborhood(block) {
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

    /// Refines the mesh using the given flags (temporary API).
    pub fn refine(&mut self, flags: &[bool]) {
        assert!(self.num_cells() == flags.len());
        assert!(self.is_balanced(flags));

        let num_flags = flags.iter().filter(|&&p| p).count();
        let total_blocks = self.num_cells() + (AxisMask::<N>::COUNT - 1) * num_flags;

        // ****************************************
        // Prepass to deteremine updated indices **

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

        for block in 0..self.num_cells() {
            if flags[block] {
                let parent = bounds.len();
                // Loop over subdivisions
                for mask in AxisMask::<N>::enumerate() {
                    // Set physical bounds of block
                    bounds.push(self.bounds(block).split(mask));
                    // Update neighbors
                    for face in faces::<N>() {
                        if mask.is_inner_face(face) {
                            // We can just reflect across axis within this particular group
                            let neighbor = parent + mask.toggled(face.axis).into_linear();
                            neighbors.push(neighbor);
                        } else {
                            let neighbor = self.neighbors(block)[face.to_linear()];
                            if neighbor == SPATIAL_BOUNDARY {
                                // Propagate boundary information
                                neighbors.push(SPATIAL_BOUNDARY);
                                continue;
                            }

                            let level = self.level(block);
                            let neighbor_level = self.level(neighbor);

                            let neighbor =
                                match (flags[neighbor], level as isize - neighbor_level as isize) {
                                    (true, -1) => {
                                        // We refine both and the neighbor started finer
                                        update_map[neighbor + mask.toggled(face.axis).into_linear()]
                                    }
                                    (false, -1) => {
                                        // Neighbor was more refined, but now they are on the same level
                                        update_map[neighbor + mask.toggled(face.axis).into_linear()]
                                    }
                                    (true, 0) => {
                                        // If we refine both and they started on the same level. simply flip
                                        // along axis.
                                        update_map[neighbor] + mask.toggled(face.axis).into_linear()
                                    }
                                    (false, 0) => {
                                        // Started on same level, but neighbor was not refined
                                        update_map[neighbor]
                                    }
                                    (true, 1) => {
                                        // We refine both and this block starts finer than neighbor
                                        update_map[neighbor]
                                            + self
                                                .subdivision(block)
                                                .toggled(face.axis)
                                                .into_linear()
                                    }

                                    _ => panic!("Unbalanced quadtree."),
                                };

                            neighbors.push(neighbor);
                        }
                    }

                    // Compute index
                    for axis in 0..N {
                        // Multiply current index by 2 and set least significant bit
                        indices[axis].push(mask.is_set(axis));
                        indices[axis].extend_from_bitslice(self.index_slice(block, axis));
                    }
                    let previous = offsets[offsets.len() - 1];
                    offsets.push(previous + self.level(block) + 1);
                }
            } else {
                // Set physical bounds of block
                bounds.push(self.bounds(block));

                // Update neighbors
                for face in faces::<N>() {
                    let neighbor = self.neighbors(block)[face.to_linear()];
                    if neighbor == SPATIAL_BOUNDARY {
                        neighbors.push(SPATIAL_BOUNDARY);
                        continue;
                    }

                    neighbors.push(update_map[neighbor])
                }

                for axis in 0..N {
                    indices[axis].extend_from_bitslice(self.index_slice(block, axis));
                }

                let previous = offsets[offsets.len() - 1];
                offsets.push(previous + self.level(block));
            }
        }

        self.bounds = bounds;
        self.neighbors = neighbors;
        self.indices = indices;
        self.offsets = offsets;
    }

    /// Returns all coarse blocks that neighbor the given block.
    fn coarse_neighborhood(&self, block: usize) -> impl Iterator<Item = usize> + '_ {
        let subdivision = self.subdivision(block);

        // Loop over possible directions
        AxisMask::<N>::enumerate()
            .skip(1)
            .map(move |dir| {
                // Get neighboring node
                let mut neighbor = block;
                // Is this neighbor coarser than this node?
                let mut neighbor_coarse = false;
                // Loop over axes which can potentially be coarse
                for face in subdivision.outer_faces() {
                    if !dir.is_set(face.axis) {
                        continue;
                    }

                    // Get neighbor
                    let traverse = self.neighbors(neighbor)[face.to_linear()];

                    if traverse == SPATIAL_BOUNDARY {
                        // No Update necessary if face is on boundary
                        neighbor_coarse = false;
                        neighbor = SPATIAL_BOUNDARY;
                        break;
                    }

                    if self.level(traverse) < self.level(block) {
                        neighbor_coarse = true;
                    }

                    // Update node
                    neighbor = traverse;
                }

                if neighbor_coarse {
                    neighbor
                } else {
                    SPATIAL_BOUNDARY
                }
            })
            .filter(|&neighbor| neighbor != SPATIAL_BOUNDARY)
    }

    /// Returns the z index for
    fn index_slice(&self, cell: usize, axis: usize) -> &BitSlice<usize, Lsb0> {
        &self.indices[axis][self.offsets[cell]..self.offsets[cell + 1]]
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SpatialTreeSerde<const N: usize> {
    /// Bounds of each cell in tree
    bounds: Vec<Rectangle<N>>,
    /// Stores neighbors along each face
    neighbors: Vec<usize>,
    /// Index within z-filling curve
    indices: Array<[BitVec<usize, Lsb0>; N]>,
    /// Offsets into indices,
    offsets: Vec<usize>,
}

impl<const N: usize> From<SpatialTree<N>> for SpatialTreeSerde<N> {
    fn from(value: SpatialTree<N>) -> Self {
        Self {
            bounds: value.bounds,
            neighbors: value.neighbors,
            indices: value.indices.into(),
            offsets: value.offsets,
        }
    }
}

impl<const N: usize> From<SpatialTreeSerde<N>> for SpatialTree<N> {
    fn from(value: SpatialTreeSerde<N>) -> Self {
        Self {
            bounds: value.bounds,
            neighbors: value.neighbors,
            indices: value.indices.inner(),
            offsets: value.offsets,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balancing() {
        let mut tree = SpatialTree::new(Rectangle::<2>::UNIT);

        tree.refine(&[true, false, false, false]);

        // Produce unbalanced flags
        let mut flags = vec![false, false, false, true, false, false, false];
        assert!(!tree.is_balanced(&flags));

        // Perform rebalancing
        tree.balance(&mut flags);
        assert_eq!(flags, &[false, false, false, true, true, true, true]);

        assert!(tree.is_balanced(&flags));
    }

    #[test]
    fn refinement() {
        let mut tree = SpatialTree::new(Rectangle::<2>::UNIT);

        assert_eq!(tree.num_cells(), 4);
        assert_eq!(tree.subdivision(0).unpack(), [false, false]);
        assert_eq!(tree.subdivision(1).unpack(), [true, false]);
        assert_eq!(tree.subdivision(2).unpack(), [false, true]);
        assert_eq!(tree.subdivision(3).unpack(), [true, true]);

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
        assert_eq!(
            tree.neighbors(0),
            &[SPATIAL_BOUNDARY, 1, SPATIAL_BOUNDARY, 2]
        );
        assert_eq!(tree.neighbors(2), &[SPATIAL_BOUNDARY, 3, 0, 4]);
        assert_eq!(tree.neighbors(5), &[4, 8, 3, SPATIAL_BOUNDARY]);
        assert_eq!(tree.neighbors(10), &[5, 11, 8, SPATIAL_BOUNDARY]);
    }
}
