// ************************
// Block Dofs *************
// ************************

use crate::TreeBlocks;
use std::{array, ops::Range};

/// Associates vertices with each block in the `Tree`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeVertices<const N: usize> {
    /// Number of subdivisions for each axis.
    #[serde(with = "aeon_array")]
    pub width: [usize; N],
    /// Ghost vertices along each face.
    pub ghost: usize,
    /// Stores a map from blocks to ranges of vertices.
    offsets: Vec<usize>,
}

impl<const N: usize> TreeVertices<N> {
    pub fn new(width: [usize; N], ghost: usize) -> Self {
        Self {
            width,
            ghost,
            offsets: Vec::new(),
        }
    }

    // pub fn width(&self) -> [usize; N] {
    //     self.width
    // }

    // pub fn ghost(&self) -> usize {
    //     self.ghost
    // }

    /// Returns the total number of nodes in the tree.
    pub fn num_vertices(&self) -> usize {
        self.offsets.last().unwrap().clone()
    }

    /// The range of nodes associated with the given block.
    pub fn block_vertices(&self, block: usize) -> Range<usize> {
        self.offsets[block]..self.offsets[block + 1]
    }

    /// Rebuilds the set of tree nodes.
    pub fn build(&mut self, blocks: &TreeBlocks<N>) {
        for axis in 0..N {
            assert!(self.width[axis] % 2 == 0);
        }

        // Reset map
        self.offsets.clear();
        self.offsets.reserve(blocks.num_blocks() + 1);

        // Start cursor at 0.
        let mut cursor = 0;
        self.offsets.push(cursor);

        for block in 0..blocks.num_blocks() {
            let size = blocks.block_size(block);
            // Width of block in nodes.
            let block_width: [usize; N] =
                array::from_fn(|axis| self.width[axis] * size[axis] + 1 + 2 * self.ghost);

            cursor += block_width.iter().product::<usize>();
            self.offsets.push(cursor);
        }
    }
}

impl<const N: usize> Default for TreeVertices<N> {
    fn default() -> Self {
        Self {
            width: [2; N],
            ghost: 0,
            offsets: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Rectangle, Tree, TreeBlocks, TreeVertices};
    #[test]
    fn dof_offsets() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut nodes = TreeVertices::new([8; 2], 3);

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        nodes.build(&blocks);

        assert_eq!(blocks.num_blocks(), 3);

        assert_eq!(nodes.block_vertices(0), 0..529);
        assert_eq!(nodes.block_vertices(1), 529..874);
        assert_eq!(nodes.block_vertices(2), 874..1099);
    }
}
