use std::{array::from_fn, iter::repeat};

use crate::geometry::{faces, AxisMask, Face, Rectangle};
use bitvec::prelude::*;

pub const BOUNDARY: usize = usize::MAX;

pub struct Tree<const N: usize> {
    /// Bounds of each block in tree
    bounds: Vec<Rectangle<N>>,
    /// Stores neighbors along each face
    neighbors: Vec<usize>,
    /// Index within z-filling curve
    indices: [BitVec<usize, Lsb0>; N],
    /// Offsets into indices,
    offsets: Vec<usize>,
}

impl<const N: usize> Tree<N> {
    /// Constructs a new tree with a single root node.
    pub fn new(bounds: Rectangle<N>) -> Self {
        Self {
            bounds: vec![bounds],
            neighbors: vec![BOUNDARY; 2 * N],
            indices: from_fn(|_| BitVec::new()),
            offsets: vec![0, 0],
        }
    }

    /// Returns neighbors for each face of the block.
    pub fn neighbors(&self, block: usize) -> &[usize] {
        &self.neighbors[block * 2 * N..(block + 1) * 2 * N]
    }

    /// Computes the level of the block.
    pub fn level(&self, block: usize) -> usize {
        self.offsets[block + 1] - self.offsets[block]
    }

    pub fn bounds(&self, block: usize) -> Rectangle<N> {
        self.bounds[block].clone()
    }

    pub fn num_blocks(&self) -> usize {
        self.bounds.len()
    }

    pub fn index_slice(&self, block: usize, axis: usize) -> &BitSlice<usize, Lsb0> {
        &self.indices[axis][self.offsets[block]..self.offsets[block + 1]]
    }

    pub fn refine(&mut self, flags: &[bool]) {
        assert!(self.num_blocks() == flags.len());

        let num_flags = flags.iter().filter(|&&p| p).count();
        let total_blocks = self.num_blocks() + (AxisMask::<N>::COUNT - 1) * num_flags;

        let mut bounds = Vec::with_capacity(total_blocks);
        let mut indices = from_fn(|_| BitVec::with_capacity(total_blocks));
        let mut offsets = Vec::with_capacity(total_blocks + 1);
        offsets.push(0);

        // Accumulate map from old to new indices.
        let mut old_to_new = Vec::with_capacity(self.num_blocks());

        for block in 0..self.num_blocks() {
            old_to_new.push(offsets.len() - 1);

            if flags[block] {
                for mask in AxisMask::<N>::enumerate() {
                    // Set physical bounds of block
                    bounds.push(self.bounds(block).split(mask));
                    // Compute index
                    for axis in 0..N {
                        // Multiply current index by 2 and set least significant bit
                        indices[axis].push(mask.is_set(axis));
                        indices[axis].extend_from_bitslice(self.index_slice(block, axis));
                    }
                    let previous = offsets.len();
                    offsets.push(previous + self.level(block) + 1);
                }
            } else {
                // Set physical bounds of block
                bounds.push(self.bounds(block));

                for axis in 0..N {
                    indices[axis].extend_from_bitslice(self.index_slice(block, axis));
                }

                let previous = offsets.len();
                offsets.push(previous + self.level(block) + 1);
            }
        }

        // Compute neighborhood
        let mut neighbors = vec![0; total_blocks * 2 * N];

        let mut cursor = 0;

        for block in 0..self.num_blocks() {
            if flags[block] {
                for mask in AxisMask::<N>::enumerate() {
                    let coffset = cursor + mask.into_linear();
                    // Update inner indices
                    for face in mask.inner_faces() {
                        let mut nmask = mask.clone();
                        nmask.toggle(face.axis);
                        neighbors[2 * N * coffset + face.to_linear()] =
                            cursor + nmask.into_linear();
                    }

                    // Update outer indices
                    for face in mask.outer_faces() {
                        // Old outer neighbor
                        let oneighbor = self.neighbors(block)[face.to_linear()];
                        // Propogate boundary information
                        if oneighbor == BOUNDARY {
                            neighbors[2 * N * coffset + face.to_linear()] = BOUNDARY;
                            continue;
                        }

                        let nneighbor = old_to_new[oneighbor];
                    }
                }

                cursor += AxisMask::<N>::COUNT;
            } else {
                for face in faces::<N>() {}

                neighbors.extend_from_slice(self.neighbors(block));
                cursor += 1;
            }
        }

        self.neighbors = neighbors;
        self.indices = indices;
        self.offsets = offsets;
    }
}
