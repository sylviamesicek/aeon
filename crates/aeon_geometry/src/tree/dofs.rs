// ************************
// Block Dofs *************
// ************************

use crate::TreeBlocks;
use std::{array, ops::Range};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeDofs<const N: usize> {
    #[serde(with = "aeon_array")]
    pub width: [usize; N],
    pub ghost: usize,
    node_offsets: Vec<usize>,
}

impl<const N: usize> TreeDofs<N> {
    pub fn new(width: [usize; N], ghost: usize) -> Self {
        Self {
            width,
            ghost,
            node_offsets: Vec::new(),
        }
    }

    pub fn width(&self) -> [usize; N] {
        self.width
    }

    pub fn ghost(&self) -> usize {
        self.ghost
    }

    /// Returns the total number of nodes in the tree.
    pub fn num_nodes(&self) -> usize {
        self.node_offsets.last().unwrap().clone()
    }

    /// The range of nodes associated with the given block.
    pub fn block_nodes(&self, block: usize) -> Range<usize> {
        self.node_offsets[block]..self.node_offsets[block + 1]
    }

    /// Rebuilds the set of tree nodes.
    pub fn build(&mut self, blocks: &TreeBlocks<N>) {
        for axis in 0..N {
            assert!(self.width[axis] % 2 == 0);
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
                array::from_fn(|axis| self.width[axis] * size[axis] + 1 + 2 * self.ghost);

            cursor += block_width.iter().product::<usize>();
            self.node_offsets.push(cursor);
        }
    }
}

impl<const N: usize> Default for TreeDofs<N> {
    fn default() -> Self {
        Self {
            width: [2; N],
            ghost: 0,
            node_offsets: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Rectangle, Tree, TreeBlocks, TreeDofs};
    #[test]
    fn dof_offsets() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut nodes = TreeDofs::new([8; 2], 3);

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        nodes.build(&blocks);

        assert_eq!(blocks.num_blocks(), 3);

        assert_eq!(nodes.block_nodes(0), 0..529);
        assert_eq!(nodes.block_nodes(1), 529..874);
        assert_eq!(nodes.block_nodes(2), 874..1099);
    }
}
