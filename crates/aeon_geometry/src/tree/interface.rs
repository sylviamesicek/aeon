use crate::{regions, IndexSpace, Region, NULL};
use crate::{Face, Side, Tree, TreeBlocks};
use std::{array, slice};

use super::TreeDofs;

/// Stores neighbor of a cell on a tree.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeNeighbor<const N: usize> {
    pub cell: usize,
    pub neighbor: usize,
    pub region: Region<N>,
}

/// An interface between two blocks on a tree.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeInterface<const N: usize> {
    /// Target block.
    pub block: usize,
    /// Source block.
    pub neighbor: usize,
    pub a: TreeNeighbor<N>,
    pub b: TreeNeighbor<N>,
}

/// Caches information about interior interfaces on trees.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TreeInterfaces<const N: usize> {
    fine: Vec<TreeInterface<N>>,
    direct: Vec<TreeInterface<N>>,
    coarse: Vec<TreeInterface<N>>,
}

impl<const N: usize> TreeInterfaces<N> {
    /// Iterates over all fine `BlockInterface`s.
    pub fn fine(&self) -> slice::Iter<'_, TreeInterface<N>> {
        self.fine.iter()
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn direct(&self) -> slice::Iter<'_, TreeInterface<N>> {
        self.direct.iter()
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn coarse(&self) -> slice::Iter<'_, TreeInterface<N>> {
        self.coarse.iter()
    }

    /// Rebuilds the block interface data.
    pub fn build(&mut self, tree: &Tree<N>, blocks: &TreeBlocks<N>) {
        // Reused memory for neighbors.
        let mut neighbors = Vec::new();

        for block in 0..blocks.num_blocks() {
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
                // Compute this boundary interface.
                let kind = InterfaceKind::from_levels(tree.level(a.cell), tree.level(a.neighbor));
                let interface = TreeInterface {
                    block,
                    neighbor,
                    a,
                    b,
                };

                match kind {
                    InterfaceKind::Fine => self.fine.push(interface),
                    InterfaceKind::Direct => self.direct.push(interface),
                    InterfaceKind::Coarse => self.coarse.push(interface),
                }
            });
        }
    }

    /// Iterates the cell neighbors of a block, and pushes them onto the memory stack.
    fn build_cell_neighbors(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        block: usize,
        neighbors: &mut Vec<TreeNeighbor<N>>,
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

                for neighbor in tree.neighbors_in_region(cell, region) {
                    debug_assert!(neighbor != NULL);

                    neighbors.push(TreeNeighbor {
                        cell,
                        neighbor,
                        region: region.clone(),
                    })
                }
            }
        }
    }

    /// Traverses a sorted list of cell neighbors, calling f once for each distinct block.
    fn taverse_cell_neighbors(
        blocks: &TreeBlocks<N>,
        neighbors: &mut [TreeNeighbor<N>],
        mut f: impl FnMut(usize, TreeNeighbor<N>, TreeNeighbor<N>),
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
}

// *******************************
// Dofs **************************
// *******************************

/// Stores data on which dofs border one another, and how to fill interior interfaces.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeOverlap<const N: usize> {
    /// Target block.
    pub block: usize,
    /// Source block.
    pub neighbor: usize,
    /// Source dof on neighbor.
    #[serde(with = "aeon_array")]
    pub source: [isize; N],
    /// Destination dof on target.
    #[serde(with = "aeon_array")]
    pub dest: [isize; N],
    /// Number of dofs to be filled along each axis.
    #[serde(with = "aeon_array")]
    pub size: [usize; N],
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeOverlaps<const N: usize> {
    fine: Vec<TreeOverlap<N>>,
    direct: Vec<TreeOverlap<N>>,
    coarse: Vec<TreeOverlap<N>>,
}

impl<const N: usize> TreeOverlaps<N> {
    /// Iterates over all fine `BlockInterface`s.
    pub fn fine(&self) -> slice::Iter<'_, TreeOverlap<N>> {
        self.fine.iter()
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn direct(&self) -> slice::Iter<'_, TreeOverlap<N>> {
        self.direct.iter()
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn coarse(&self) -> slice::Iter<'_, TreeOverlap<N>> {
        self.coarse.iter()
    }

    pub fn build(
        &mut self,
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        interfaces: &TreeInterfaces<N>,
        dofs: &TreeDofs<N>,
    ) {
        for fine in interfaces.fine() {
            self.fine
                .push(Self::process_interface(tree, blocks, dofs, fine));
        }

        for direct in interfaces.direct() {
            self.fine
                .push(Self::process_interface(tree, blocks, dofs, direct));
        }

        for coarse in interfaces.coarse() {
            self.fine
                .push(Self::process_interface(tree, blocks, dofs, coarse));
        }
    }

    fn process_interface(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        dofs: &TreeDofs<N>,
        interface: &TreeInterface<N>,
    ) -> TreeOverlap<N> {
        let a = interface.a.clone();
        let b = interface.b.clone();

        // Find active region.
        let (anode, bnode) = Self::block_ghost_aabb(tree, blocks, dofs, interface);
        let mut source = Self::neighbor_origin(tree, blocks, dofs, a.clone());
        let (mut dest, mut size) = Self::space_from_aabb(anode, bnode);

        // Avoid overlaps between aabbs on this block.
        // let aorigin = blocks.cell_index(a.cell);
        // let borigin = blocks.cell_index(b.cell);
        let flags = blocks.block_boundary_flags(interface.block);
        let block_size = blocks.block_size(interface.block);

        for axis in 0..N {
            let right_boundary = flags.is_set(Face::positive(axis));
            let left_boundary = flags.is_set(Face::negative(axis));

            // // If the right edge doesn't extend all the way to the right,
            // // shrink by one.
            // if b.region.side(axis) == Side::Middle
            //     && !(borigin[axis] == block_size[axis] - 1 && right_boundary)
            // {
            //     size[axis] -= 1;
            // }

            // // If we do not extend further left, don't include
            // if a.region.side(axis) == Side::Middle && aorigin[axis] == 0 && !left_boundary {
            //     source[axis] += 1;
            //     dest[axis] += 1;
            //     size[axis] -= 1;
            // }

            // If the right edge doesn't extend all the way to the right,
            // shrink by one.
            if b.region.side(axis) == Side::Middle
                && !(bnode[axis] == (block_size[axis] * dofs.width[axis]) as isize
                    && right_boundary)
            {
                size[axis] -= 1;
            }

            // If we do not extend further left, don't include
            if a.region.side(axis) == Side::Middle && anode[axis] == 0 && !left_boundary {
                source[axis] += 1;
                dest[axis] += 1;
                size[axis] -= 1;
            }
        }

        TreeOverlap {
            block: interface.block,
            neighbor: interface.neighbor,
            source,
            dest,
            size,
        }
    }

    /// Computes the nodes that ghost nodes adjacent to the AABB of cells
    /// defined by A and B.
    fn block_ghost_aabb(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        dofs: &TreeDofs<N>,
        interface: &TreeInterface<N>,
    ) -> ([isize; N], [isize; N]) {
        let a = interface.a.clone();
        let b = interface.b.clone();

        // Find ghost region that must be filled
        let aindex = blocks.cell_index(a.cell);
        let bindex = blocks.cell_index(b.cell);

        // Compute bottom right corner of A cell.
        let mut anode: [_; N] = array::from_fn(|axis| (aindex[axis] * dofs.width[axis]) as isize);

        if tree.level(a.cell) < tree.level(a.neighbor) {
            let split = tree.split(a.neighbor);
            (0..N)
                .filter(|&axis| a.region.side(axis) == Side::Middle && split.is_set(axis))
                .for_each(|axis| anode[axis] += (dofs.width[axis] / 2) as isize);
        }

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match a.region.side(axis) {
                Side::Left => {
                    anode[axis] -= dofs.ghost as isize;
                }
                Side::Right => {
                    anode[axis] += dofs.width[axis] as isize;
                }
                Side::Middle => {}
            }
        }

        // Compute top left corner of B cell
        let mut bnode: [_; N] =
            array::from_fn(|axis| ((bindex[axis] + 1) * dofs.width[axis]) as isize);

        if tree.level(b.cell) < tree.level(b.neighbor) {
            let split = tree.split(b.neighbor);
            (0..N)
                .filter(|&axis| b.region.side(axis) == Side::Middle && !split.is_set(axis))
                .for_each(|axis| bnode[axis] -= (dofs.width[axis] / 2) as isize);
        }

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match b.region.side(axis) {
                Side::Right => {
                    bnode[axis] += dofs.ghost as isize;
                }
                Side::Left => {
                    bnode[axis] -= dofs.width[axis] as isize;
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
        let size = array::from_fn(|axis| (b[axis] - a[axis] + 1) as usize);

        (dest, size)
    }

    fn neighbor_origin(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        nodes: &TreeDofs<N>,
        a: TreeNeighbor<N>,
    ) -> [isize; N] {
        // Compute this boundary interface.
        let interface = InterfaceKind::from_levels(tree.level(a.cell), tree.level(a.neighbor));

        // Find source node
        let index = blocks.cell_index(a.neighbor);
        let mut source: [isize; N] =
            array::from_fn(|axis| (index[axis] * nodes.width[axis]) as isize);

        match interface {
            InterfaceKind::Direct => {
                for axis in 0..N {
                    if a.region.side(axis) == Side::Left {
                        source[axis] += (nodes.width[axis] - nodes.ghost) as isize;
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
                                source[axis] += nodes.width[axis] as isize - nodes.ghost as isize
                            }
                            Side::Middle => source[axis] += nodes.width[axis] as isize,
                            Side::Right => {}
                        }
                    } else {
                        match a.region.side(axis) {
                            Side::Left => {
                                source[axis] +=
                                    2 * nodes.width[axis] as isize - nodes.ghost as isize
                            }
                            Side::Middle => {}
                            Side::Right => source[axis] += nodes.width[axis] as isize,
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
                        source[axis] += nodes.width[axis] as isize / 2 - nodes.ghost as isize;
                    }
                }
            }
        }

        source
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
enum InterfaceKind {
    Coarse,
    Direct,
    Fine,
}

impl InterfaceKind {
    fn from_levels(level: usize, neighbor: usize) -> Self {
        match level as isize - neighbor as isize {
            1 => InterfaceKind::Coarse,
            0 => InterfaceKind::Direct,
            -1 => InterfaceKind::Fine,
            _ => panic!("Unbalanced levels"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Rectangle;

    use super::*;

    #[test]
    fn interfaces() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut interfaces = TreeInterfaces::default();

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        interfaces.build(&tree, &blocks);

        let mut coarse = interfaces.coarse();

        assert_eq!(
            coarse.next(),
            Some(&TreeInterface {
                block: 0,
                neighbor: 1,
                a: TreeNeighbor {
                    cell: 1,
                    neighbor: 4,
                    region: Region::new([Side::Right, Side::Middle])
                },
                b: TreeNeighbor {
                    cell: 1,
                    neighbor: 6,
                    region: Region::new([Side::Right, Side::Right])
                }
            })
        )
    }

    #[test]
    fn overlaps() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut dofs = TreeDofs::new([4; 2], 2);
        let mut interfaces = TreeInterfaces::default();
        let mut overlaps = TreeOverlaps::default();

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        dofs.build(&blocks);
        interfaces.build(&tree, &blocks);
        overlaps.build(&tree, &blocks, &interfaces, &dofs);

        let mut coarse = overlaps.coarse();

        assert_eq!(
            coarse.next(),
            Some(&TreeOverlap {
                block: 0,
                neighbor: 1,
                source: [0, 0],
                dest: [8, 0],
                size: [3, 11],
            })
        );

        assert_eq!(
            coarse.next(),
            Some(&TreeOverlap {
                block: 0,
                neighbor: 2,
                source: [0, 0],
                dest: [0, 8],
                size: [8, 3],
            })
        );

        assert_eq!(coarse.next(), None);

        let mut fine = overlaps.fine();

        assert_eq!(
            fine.next(),
            Some(&TreeOverlap {
                block: 1,
                neighbor: 0,
                source: [2, 0],
                dest: [-2, 0],
                size: [3, 4],
            })
        );

        assert_eq!(
            fine.next(),
            Some(&TreeOverlap {
                block: 2,
                neighbor: 0,
                source: [0, 2],
                dest: [0, -2],
                size: [4, 3],
            })
        );

        assert_eq!(fine.next(), None);

        let mut direct = overlaps.direct();

        assert_eq!(
            direct.next(),
            Some(&TreeOverlap {
                block: 1,
                neighbor: 2,
                source: [2, 0],
                dest: [-2, 4],
                size: [3, 5],
            })
        );

        assert_eq!(
            direct.next(),
            Some(&TreeOverlap {
                block: 2,
                neighbor: 1,
                source: [0, 2],
                dest: [4, -2],
                size: [3, 7],
            })
        );

        assert_eq!(direct.next(), None);
    }
}
