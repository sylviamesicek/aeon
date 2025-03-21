use std::{array::from_fn, ops::Range};

use crate::geometry::{AxisMask, Rectangle};

use super::{Tree, NULL};

#[derive(Clone, Debug)]
struct TreeNode<const N: usize> {
    /// Physical bounds of this node
    bounds: Rectangle<N>,
    /// Parent Node
    parent: usize,
    /// Child nodes
    children: usize,
    /// Offset map to owned cells
    cell_offset: usize,
    /// Number of cells owned by this node
    cell_count: usize,
}

#[derive(Clone, Debug, Default)]
pub struct TreeGraph<const N: usize> {
    /// Nodes that constitute this directed acyclic graph.
    nodes: Vec<TreeNode<N>>,
    /// Map from node level to nodes.
    level_offsets: Vec<usize>,
    /// Map from cells to leaf nodes
    cell_to_node: Vec<usize>,
}

impl<const N: usize> TreeGraph<N> {
    /// Constructs the tree graph of a given tree.
    pub fn build(&mut self, tree: &Tree<N>) {
        // Reset tree
        self.nodes.clear();
        self.level_offsets.clear();
        self.cell_to_node.clear();

        self.cell_to_node.resize(tree.num_cells(), 0);

        // Add root node (owns all cells in tree).
        self.nodes.push(TreeNode {
            bounds: tree.domain(),
            parent: NULL,
            children: NULL,
            cell_offset: 0,
            cell_count: tree.num_cells(),
        });
        self.level_offsets.push(0);
        self.level_offsets.push(1);

        // Recursively subdivide existing nodes using `tree.indices`.
        loop {
            let level = self.level_offsets.len() - 2;
            let level_nodes = self.level_offsets[level]..self.level_offsets[level + 1];

            // First node on the current level.
            let level_start = self.nodes.len();
            // Loop over nodes on this level.
            for node_index in level_nodes {
                let children = self.nodes.len();
                let node = &mut self.nodes[node_index];
                // Continue, we have reached a leaf.
                if node.cell_count == 1 {
                    self.cell_to_node[node.cell_offset] = node_index;
                    continue;
                }
                // Update parent's children.
                node.children = children;
                // Iterate over constituent cells.
                let cell_start = node.cell_offset;
                let cell_end = node.cell_offset + node.cell_count;

                let mut cursor = cell_start;

                for mask in AxisMask::<N>::enumerate() {
                    let child_cell_start = cursor;

                    // Iterate all things which belong in this subdivision
                    'cells: while cursor < cell_end {
                        // We cell_level > node_level
                        for axis in 0..N {
                            let slice = tree.index_slice(cursor, axis);
                            let bit = slice[slice.len() - 1 - level];
                            if bit != mask.is_set(axis) {
                                break 'cells;
                            }
                        }

                        cursor += 1;
                    }

                    let child_cell_end = cursor;

                    self.nodes.push(TreeNode {
                        bounds: Rectangle::from_aabb(
                            tree.bounds(child_cell_start).aa(),
                            tree.bounds(child_cell_end - 1).bb(),
                        ),
                        parent: node_index,
                        children: NULL,
                        cell_offset: child_cell_start,
                        cell_count: child_cell_end - child_cell_start,
                    });
                }

                debug_assert!(self.nodes.len() - children == AxisMask::<N>::COUNT);
            }

            // Last node on current level.
            let level_end = self.nodes.len();
            if level_start == level_end {
                break;
            }

            self.level_offsets.push(self.nodes.len());
        }
    }

    /// Number of nodes in the tree graph.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of levels in the tree.
    pub fn num_levels(&self) -> usize {
        self.level_offsets.len() - 1
    }

    /// Returns the bounds of a given node.
    pub fn bounds(&self, index: usize) -> Rectangle<N> {
        self.nodes[index].bounds
    }

    /// Returns the children of a given node. Node must not be leaf.
    pub fn children(&self, index: usize) -> Range<usize> {
        debug_assert!(self.nodes[index].children != NULL);
        let offset = self.nodes[index].children;
        offset..offset + AxisMask::<N>::COUNT
    }

    pub fn child(&self, index: usize, child: AxisMask<N>) -> usize {
        debug_assert!(self.nodes[index].children != NULL);
        self.nodes[index].children + child.to_linear()
    }

    pub fn parent(&self, index: usize) -> usize {
        self.nodes[index].parent
    }

    /// True if a node has no children.
    pub fn is_leaf(&self, index: usize) -> bool {
        let result = self.nodes[index].children == NULL;
        debug_assert!(!result || self.nodes[index].cell_count == 1);
        result
    }

    /// True if a node has no parents.
    pub fn is_root(&self, index: usize) -> bool {
        if index > 0 {
            debug_assert!(self.nodes[index].parent != NULL);
        }
        index == 0
    }

    /// Returns the nodes on the given level.
    pub fn level_nodes(&self, level: usize) -> Range<usize> {
        self.level_offsets[level]..self.level_offsets[level + 1]
    }

    /// Computes the node corresponding to a given cell.
    pub fn cell_to_node(&self, cell: usize) -> usize {
        self.cell_to_node[cell]
    }

    /// Returns the cell which owns the given point.
    pub fn owner(&self, point: [f64; N]) -> usize {
        debug_assert!(self.bounds(0).contains(point));

        let mut node = 0;

        while !self.is_leaf(node) {
            let bounds = self.bounds(node);
            let center = bounds.center();

            let mask = AxisMask::pack(from_fn(|axis| point[axis] > center[axis]));
            node = self.child(node, mask);
        }

        self.nodes[node].cell_offset
    }

    /// Returns the cell which owns the given point.
    pub fn owner_with_guess(&self, point: [f64; N], guess: usize) -> usize {
        debug_assert!(self.bounds(0).contains(point));

        let mut guess = self.cell_to_node(guess);
        while !self.bounds(guess).contains(point) {
            guess = self.parent(guess);
        }

        let mut node = guess;

        while !self.is_leaf(node) {
            let bounds = self.bounds(node);
            let center = bounds.center();

            let mask = AxisMask::pack(from_fn(|axis| point[axis] > center[axis]));
            node = self.child(node, mask);
        }

        self.nodes[node].cell_offset
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::{Rectangle, Tree, NULL};

    use super::TreeGraph;

    fn construct_tree() -> Tree<2> {
        let mut tree = Tree::new(Rectangle::UNIT, [false; 2]);
        tree.refine(&[true, false, true, false]);
        tree.refine(&[
            true, false, false, false, false, false, false, true, false, false,
        ]);

        tree
    }

    #[test]
    fn basic_graph() {
        // Build tree
        let tree = construct_tree();
        // Build graph
        let mut graph = TreeGraph::default();
        graph.build(&tree);
        // Assert

        assert_eq!(graph.num_levels(), 4);
        assert_eq!(graph.num_nodes(), 21);
        assert_eq!(graph.parent(0), NULL);
        assert_eq!(graph.children(0), 1..5);

        assert_eq!(graph.level_nodes(0), 0..1);
        assert_eq!(graph.level_nodes(1), 1..5);
        assert_eq!(graph.level_nodes(2), 5..13);
        assert_eq!(graph.level_nodes(3), 13..21);

        for node in graph.level_nodes(3) {
            assert!(graph.is_leaf(node));
        }

        assert_eq!(graph.parent(17), 11);

        dbg!(graph);
    }
}
