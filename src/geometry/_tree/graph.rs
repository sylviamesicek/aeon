use std::{array::from_fn, ops::Range};

use crate::geometry::{AxisMask, Rectangle, Tree, NULL};

#[derive(Clone, Debug)]
struct GraphNode<const N: usize> {
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
    nodes: Vec<GraphNode<N>>,
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
        self.nodes.push(GraphNode {
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

                    self.nodes.push(GraphNode {
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
    pub fn bounds(&self, node: usize) -> Rectangle<N> {
        self.nodes[node].bounds
    }

    /// Returns the children of a given node. Node must not be leaf.
    pub fn children(&self, node: usize) -> Option<usize> {
        if self.nodes[node].children == NULL {
            return None;
        }
        Some(self.nodes[node].children)
    }

    /// Returns a child of a give node.
    pub fn child(&self, node: usize, child: AxisMask<N>) -> Option<usize> {
        if self.nodes[node].children == NULL {
            return None;
        }
        Some(self.nodes[node].children + child.to_linear())
    }

    /// The parent node of a given node.
    pub fn parent(&self, node: usize) -> Option<usize> {
        if self.nodes[node].parent == NULL {
            return None;
        }

        Some(self.nodes[node].parent)
    }

    /// True if a node has no children.
    pub fn is_leaf(&self, node: usize) -> bool {
        let result = self.nodes[node].children == NULL;
        debug_assert!(!result || self.nodes[node].cell_count == 1);
        result
    }

    /// True if a node has no parents.
    pub fn is_root(&self, node: usize) -> bool {
        if node > 0 {
            debug_assert!(self.nodes[node].parent != NULL);
        }
        node == 0
    }

    /// Returns the nodes on the given level.
    pub fn level_nodes(&self, level: usize) -> Range<usize> {
        self.level_offsets[level]..self.level_offsets[level + 1]
    }

    /// Computes the leaf node corresponding to a given cell.
    pub fn node_from_cell(&self, cell: usize) -> usize {
        self.cell_to_node[cell]
    }

    /// Computes the cell that a leaf node corresponds to (None if the node
    /// is not leaf).
    pub fn cell_from_node(&self, node: usize) -> Option<usize> {
        if self.nodes[node].cell_count != 1 {
            return None;
        }

        Some(self.nodes[node].cell_offset)
    }

    /// Returns the cell which owns the given point.
    /// Performs in O(log N).
    pub fn node_from_point(&self, point: [f64; N]) -> usize {
        debug_assert!(self.bounds(0).contains(point));

        let mut node = 0;

        while let Some(children) = self.children(node) {
            let bounds = self.bounds(node);
            let center = bounds.center();

            let mask = AxisMask::<N>::pack(from_fn(|axis| point[axis] > center[axis]));
            node = children + mask.to_linear();
        }

        node
    }

    /// Returns the node which owns the given point, shortening this search
    /// with an initial guess. Rather than operating in O(log N) time, this approaches
    /// O(1) if the guess is sufficiently close.
    pub fn node_from_point_cached(&self, point: [f64; N], mut cache: usize) -> usize {
        debug_assert!(self.bounds(0).contains(point));

        while !self.bounds(cache).contains(point) {
            cache = self.parent(cache).unwrap();
        }

        let mut node = cache;

        while let Some(children) = self.children(node) {
            let bounds = self.bounds(node);
            let center = bounds.center();

            let mask = AxisMask::<N>::pack(from_fn(|axis| point[axis] > center[axis]));
            node = children + mask.to_linear();
        }

        node
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::{Rectangle, Tree};

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
        assert_eq!(graph.parent(0), None);
        assert_eq!(graph.children(0), Some(1));

        assert_eq!(graph.children(1), Some(5));
        assert_eq!(graph.children(2), None);
        assert_eq!(graph.children(3), Some(9));
        assert_eq!(graph.children(4), None);

        assert_eq!(graph.level_nodes(0), 0..1);
        assert_eq!(graph.level_nodes(1), 1..5);
        assert_eq!(graph.level_nodes(2), 5..13);
        assert_eq!(graph.level_nodes(3), 13..21);

        for node in graph.level_nodes(3) {
            assert!(graph.is_leaf(node));
        }

        assert_eq!(graph.parent(17), Some(11));

        dbg!(graph);
    }

    #[test]
    fn graph_node_from_point() {
        // Build tree
        let tree = construct_tree();
        // Build graph
        let mut graph = TreeGraph::default();
        graph.build(&tree);
        // Asserts
        assert_eq!(graph.node_from_point([0.6, 0.75]), 4);
        assert_eq!(graph.node_from_point([0.4, 0.4]), 8);
        assert_eq!(graph.node_from_point([0.1, 0.1]), 13);
        assert_eq!(graph.node_from_point([0.1, 0.9]), 19);
    }
}
