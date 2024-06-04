use std::array::from_fn;

use crate::geometry::{faces, AxisMask, Face, Rectangle};
use bitvec::prelude::*;

pub const BOUNDARY: usize = usize::MAX;

pub struct Tree<const N: usize> {
    bounds: Rectangle<N>,

    /// Stores neighbors along each face
    neighbors: Vec<usize>,
    /// Index within z-filling curve
    indices: Vec<[u64; N]>,
    /// Level of each block
    levels: Vec<usize>,
}

impl<const N: usize> Tree<N> {
    /// Constructs a new tree with a single root node.
    pub fn new(bounds: Rectangle<N>) -> Self {
        Self {
            bounds,

            neighbors: vec![BOUNDARY; 2 * N],
            indices: vec![[0; N]],
            levels: vec![0],
        }
    }

    /// Retrieves the physical bounds of the tree
    pub fn bounds(&self) -> Rectangle<N> {
        self.bounds.clone()
    }

    /// Retrieves the neighbor on the given face of the block.
    pub fn neighbor(&self, block: usize, face: Face) -> usize {
        self.neighbors[block * 2 * N + face.to_linear()]
    }

    /// Returns neighbors for each face of the block.
    pub fn neighbors(&self, block: usize) -> &[usize] {
        &self.neighbors[block * 2 * N..(block + 1) * 2 * N]
    }

    /// Computes the level of the block.
    pub fn level(&self, block: usize) -> usize {
        self.levels[block]
    }

    pub fn spacing(&self, level: usize) -> [f64; N] {
        from_fn(|i| self.bounds.size[i] / 2usize.pow(level as u32) as f64)
    }

    pub fn position(&self, block: usize) -> [f64; N] {
        let spacing = self.spacing(self.level(block));
        let index = self.indices[block];

        from_fn(|i| self.bounds.origin[i] + spacing[i] * index[i] as f64)
    }

    pub fn num_blocks(&self) -> usize {
        self.levels.len()
    }

    pub fn refine(&mut self, flags: &[bool]) {
        assert!(self.num_blocks() == flags.len());

        let num_flags = flags.iter().filter(|&&p| p).count();
        let total_blocks = self.num_blocks() + (AxisMask::<N>::COUNT - 1) * num_flags;

        let mut neighbors = Vec::with_capacity(total_blocks * 2 * N);
        let mut indices = Vec::with_capacity(total_blocks);
        let mut levels = Vec::with_capacity(total_blocks);

        for block in 0..self.num_blocks() {
            let level = self.level(block);
            let index = self.indices[block];

            if flags[block] {
                for mask in AxisMask::<N>::enumerate() {
                    let mut nindex = [0; N];
                    for axis in 0..N {
                        // Multiply current index by 2 and set least significant bit
                        nindex[axis] = index[axis] << 1 | mask.is_set(axis) as u64;
                    }
                    indices.push(nindex);
                    levels.push(level + 1);
                }
            } else {
                indices.push(index);
                levels.push(level);
            }
        }

        self.neighbors = neighbors;
        self.indices = indices;
    }
}
