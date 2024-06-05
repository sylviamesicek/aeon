use std::array::from_fn;

use crate::geometry::{faces, AxisMask, Rectangle};
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

    /// Number of blocks in mesh
    pub fn num_blocks(&self) -> usize {
        self.bounds.len()
    }

    /// Returns neighbors for each face of the block.
    pub fn neighbors(&self, block: usize) -> &[usize] {
        &self.neighbors[block * 2 * N..(block + 1) * 2 * N]
    }

    /// Computes the level of the block.
    pub fn level(&self, block: usize) -> usize {
        self.offsets[block + 1] - self.offsets[block]
    }

    /// Returns the bounds of a block
    pub fn bounds(&self, block: usize) -> Rectangle<N> {
        self.bounds[block].clone()
    }

    pub fn index_slice(&self, block: usize, axis: usize) -> &BitSlice<usize, Lsb0> {
        &self.indices[axis][self.offsets[block]..self.offsets[block + 1]]
    }

    /// If block is not root, returns it most recent subdivision.
    pub fn subdivision(&self, block: usize) -> Option<AxisMask<N>> {
        if self.level(block) == 0 {
            return None;
        }

        Some(AxisMask::pack(from_fn(|axis| {
            self.indices[axis][self.offsets[block]]
        })))
    }

    /// Returns all coarse blocks that neighbor the given block.
    fn coarse_neighborhood(&self, block: usize) -> impl Iterator<Item = usize> + '_ {
        assert!(self.level(block) != 0);

        let subdivision = self.subdivision(block).unwrap();

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
                    if dir.is_set(face.axis) == false {
                        continue;
                    }

                    // Get neighbor
                    let traverse = self.neighbors(neighbor)[face.to_linear()];

                    if traverse == BOUNDARY {
                        // No Update necessary if face is on boundary
                        neighbor_coarse = false;
                        neighbor = BOUNDARY;
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
                    BOUNDARY
                }
            })
            .filter(|&neighbor| neighbor != BOUNDARY)
    }

    pub fn is_balanced(&self, flags: &[bool]) -> bool {
        assert!(flags.len() == self.num_blocks());

        for block in 0..self.num_blocks() {
            if !flags[block] || self.level(block) == 0 {
                continue;
            }

            for coarse in self.coarse_neighborhood(block) {
                if !flags[coarse] {
                    return false;
                }
            }
        }

        return true;
    }

    pub fn balance(&self, flags: &mut [bool]) {
        assert!(flags.len() == self.num_blocks());

        loop {
            let mut is_balanced = true;

            for block in 0..self.num_blocks() {
                if !flags[block] || self.level(block) == 0 {
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
        assert!(self.num_blocks() == flags.len());
        assert!(self.is_balanced(flags));

        let num_flags = flags.iter().filter(|&&p| p).count();
        let total_blocks = self.num_blocks() + (AxisMask::<N>::COUNT - 1) * num_flags;

        // ****************************************
        // Prepass to deteremine updated indices **

        let mut update_map = Vec::with_capacity(self.num_blocks());

        let mut cursor = 0;
        for block in 0..self.num_blocks() {
            update_map.push(cursor);

            if flags[block] {
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

        for block in 0..self.num_blocks() {
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
                            if neighbor == BOUNDARY {
                                // Propagate boundary information
                                neighbors.push(BOUNDARY);
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
                                                .unwrap()
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
                    if neighbor == BOUNDARY {
                        neighbors.push(BOUNDARY);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balancing() {
        let mut mesh = Tree::new(Rectangle::<2>::UNIT);

        mesh.refine(&[true]);
        mesh.refine(&[true, false, false, false]);

        // Produce unbalanced flags
        let mut flags = vec![false, false, false, true, false, false, false];
        assert!(!mesh.is_balanced(&flags));

        // Perform rebalancing
        mesh.balance(&mut flags);
        assert_eq!(flags, &[false, false, false, true, true, true, true]);

        assert!(mesh.is_balanced(&flags));
    }

    #[test]
    fn refinement() {
        let mut mesh = Tree::new(Rectangle::<2>::UNIT);

        mesh.refine(&[true]);

        assert_eq!(mesh.num_blocks(), 4);
        assert_eq!(mesh.subdivision(0).unwrap().unpack(), [false, false]);
        assert_eq!(mesh.subdivision(1).unwrap().unpack(), [true, false]);
        assert_eq!(mesh.subdivision(2).unwrap().unpack(), [false, true]);
        assert_eq!(mesh.subdivision(3).unwrap().unpack(), [true, true]);

        mesh.refine(&[false, false, false, true]);

        assert_eq!(mesh.num_blocks(), 7);

        for block in 0..3 {
            assert_eq!(mesh.level(block), 1);
        }
        for block in 3..7 {
            assert_eq!(mesh.level(block), 2);
        }

        mesh.refine(&[false, false, true, false, false, true, false]);

        assert_eq!(mesh.num_blocks(), 13);
        assert_eq!(mesh.neighbors(0), &[BOUNDARY, 1, BOUNDARY, 2]);
        assert_eq!(mesh.neighbors(2), &[BOUNDARY, 3, 0, 4]);
        assert_eq!(mesh.neighbors(5), &[4, 8, 3, BOUNDARY]);
        assert_eq!(mesh.neighbors(10), &[5, 11, 8, BOUNDARY]);
    }
}
