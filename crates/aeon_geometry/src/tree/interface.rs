use crate::{regions, IndexSpace, Region, NULL};
use crate::{Face, Side, Tree, TreeBlocks, TreeNodes};
use std::{array, slice};

/// Stores neighbor of a cell on a tree.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeCellNeighbor<const N: usize> {
    /// Primary cell.
    pub cell: usize,
    /// Neighbor cell.
    pub neighbor: usize,
    /// Which region is the neighbor cell in?
    pub region: Region<N>,
}

/// Neighbor of block.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeBlockNeighbor<const N: usize> {
    /// Primary block.
    pub block: usize,
    /// Neighbor block.
    pub neighbor: usize,
    /// Leftmost cell neighbor.
    pub a: TreeCellNeighbor<N>,
    /// Rightmost cell neighbor.
    pub b: TreeCellNeighbor<N>,
}

impl<const N: usize> TreeBlockNeighbor<N> {
    /// If this is a face neighbor, return the corresponding face, otherwise return `None`.
    pub fn face(&self) -> Option<Face<N>> {
        regions_to_face(self.a.region, self.b.region)
    }
}

pub fn regions_to_face<const N: usize>(a: Region<N>, b: Region<N>) -> Option<Face<N>> {
    let mut adjacency = 0;
    let mut faxis = 0;
    let mut fside = false;

    for axis in 0..N {
        let aside = a.side(axis);
        let bside = b.side(axis);

        if aside == bside && aside != Side::Middle {
            adjacency += 1;
            faxis = axis;
            fside = aside == Side::Right;
        }
    }

    if adjacency == 1 {
        Some(Face {
            axis: faxis,
            side: fside,
        })
    } else {
        None
    }
}

/// Stores information about neighbors of blocks and cells.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TreeNeighbors<const N: usize> {
    fine: Vec<TreeBlockNeighbor<N>>,
    direct: Vec<TreeBlockNeighbor<N>>,
    coarse: Vec<TreeBlockNeighbor<N>>,
}

impl<const N: usize> TreeNeighbors<N> {
    /// Iterates over all fine `BlockInterface`s.
    pub fn fine(&self) -> slice::Iter<'_, TreeBlockNeighbor<N>> {
        self.fine.iter()
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn direct(&self) -> slice::Iter<'_, TreeBlockNeighbor<N>> {
        self.direct.iter()
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn coarse(&self) -> slice::Iter<'_, TreeBlockNeighbor<N>> {
        self.coarse.iter()
    }

    /// Rebuilds the block interface data.
    pub fn build(&mut self, tree: &Tree<N>, blocks: &TreeBlocks<N>) {
        self.fine.clear();
        self.coarse.clear();
        self.direct.clear();

        // Reused memory for neighbors.
        let mut neighbors = Vec::new();

        for block in 0..blocks.len() {
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
                let interface = TreeBlockNeighbor {
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
        neighbors: &mut Vec<TreeCellNeighbor<N>>,
    ) {
        let block_size = blocks.size(block);
        let block_cells = blocks.cells(block);
        let block_space = IndexSpace::new(block_size);

        debug_assert!(block_size.iter().product::<usize>() == block_cells.len());

        for region in regions::<N>() {
            if region == Region::CENTRAL {
                continue;
            }

            // Find all cells adjacent to the given region.
            for index in block_space.adjacent(region) {
                let cell = block_cells[block_space.linear_from_cartesian(index)];

                for neighbor in tree.neighbors_in_region(cell, region) {
                    debug_assert!(neighbor != NULL);

                    neighbors.push(TreeCellNeighbor {
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
        neighbors: &mut [TreeCellNeighbor<N>],
        mut f: impl FnMut(usize, TreeCellNeighbor<N>, TreeCellNeighbor<N>),
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

/// Stores dataon how to fill coarse-fine interfaces.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeInterface<const N: usize> {
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

    /// Constructs interfaces from neighbors and vertices.
    pub fn build(
        &mut self,
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        neighbors: &TreeNeighbors<N>,
        dofs: &TreeNodes<N>,
    ) {
        self.fine.clear();
        self.direct.clear();
        self.coarse.clear();

        for fine in neighbors.fine() {
            self.fine
                .push(Self::interface_from_neighbor(tree, blocks, dofs, fine));
        }

        for direct in neighbors.direct() {
            self.direct
                .push(Self::interface_from_neighbor(tree, blocks, dofs, direct));
        }

        for coarse in neighbors.coarse() {
            self.coarse
                .push(Self::interface_from_neighbor(tree, blocks, dofs, coarse));
        }
    }

    fn interface_from_neighbor(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        dofs: &TreeNodes<N>,
        interface: &TreeBlockNeighbor<N>,
    ) -> TreeInterface<N> {
        let a = interface.a.clone();
        let b = interface.b.clone();

        // Find active region.
        let (anode, bnode) = Self::block_ghost_aabb(tree, blocks, dofs, interface);
        let mut source = Self::neighbor_origin(tree, blocks, dofs, a.clone());
        let (mut dest, mut size) = Self::space_from_aabb(anode, bnode);

        // Avoid overlaps between aabbs on this block.
        // let aorigin = blocks.cell_index(a.cell);
        // let borigin = blocks.cell_index(b.cell);
        let flags = blocks.boundary_flags(interface.block);
        let block_size = blocks.size(interface.block);

        let kind = InterfaceKind::from_levels(tree.level(a.cell), tree.level(a.neighbor));

        for axis in 0..N {
            let right_boundary = flags.is_set(Face::positive(axis));
            let left_boundary = flags.is_set(Face::negative(axis));

            // // If the right edge doesn't extend all the way to the right,
            // // shrink by one.
            // if b.region.side(axis) == Side::Middle
            //     && !(bnode[axis] == (block_size[axis] * dofs.width[axis]) as isize
            //         && right_boundary)
            // {
            //     size[axis] -= 1;
            // }

            // // If we do not extend further left, don't include
            // if a.region.side(axis) == Side::Middle && anode[axis] == 0 && !left_boundary {
            //     source[axis] += 1;
            //     dest[axis] += 1;
            //     size[axis] -= 1;
            // }

            // if b.region.side(axis) == Side::Middle
            //     && bnode[axis] < (block_size[axis] * dofs.width[axis]) as isize
            // {
            //     size[axis] -= 1;
            // }

            // if a.region.side(axis) == Side::Left
            //     && matches!(kind, InterfaceKind::Fine | InterfaceKind::Direct)
            // {
            //     size[axis] -= 1;
            // }
        }

        TreeInterface {
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
        dofs: &TreeNodes<N>,
        interface: &TreeBlockNeighbor<N>,
    ) -> ([isize; N], [isize; N]) {
        let a = interface.a.clone();
        let b = interface.b.clone();

        // Find ghost region that must be filled
        let aindex = blocks.cell_position(a.cell);
        let bindex = blocks.cell_position(b.cell);

        // Compute bottom left corner of A cell.
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

        // Compute top right corner of B cell
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
        nodes: &TreeNodes<N>,
        a: TreeCellNeighbor<N>,
    ) -> [isize; N] {
        // Compute this boundary interface.
        let interface = InterfaceKind::from_levels(tree.level(a.cell), tree.level(a.neighbor));

        // Find source node
        let index = blocks.cell_position(a.neighbor);
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
    fn regions_and_faces() {
        assert_eq!(regions_to_face::<2>(Region::CENTRAL, Region::CENTRAL), None);
        assert_eq!(
            regions_to_face(
                Region::new([Side::Left, Side::Middle]),
                Region::new([Side::Left, Side::Left])
            ),
            Some(Face::negative(0))
        );
        assert_eq!(
            regions_to_face(
                Region::new([Side::Left, Side::Right]),
                Region::new([Side::Left, Side::Right])
            ),
            None
        );
        assert_eq!(
            regions_to_face(
                Region::new([Side::Left, Side::Right]),
                Region::new([Side::Middle, Side::Right])
            ),
            Some(Face::positive(1))
        );
        assert_eq!(
            regions_to_face(
                Region::new([Side::Middle, Side::Right]),
                Region::new([Side::Middle, Side::Right])
            ),
            Some(Face::positive(1))
        );
    }

    #[test]
    fn neighbors() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut interfaces = TreeNeighbors::default();

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        interfaces.build(&tree, &blocks);

        let mut coarse = interfaces.coarse();

        assert_eq!(
            coarse.next(),
            Some(&TreeBlockNeighbor {
                block: 0,
                neighbor: 1,
                a: TreeCellNeighbor {
                    cell: 1,
                    neighbor: 4,
                    region: Region::new([Side::Right, Side::Middle])
                },
                b: TreeCellNeighbor {
                    cell: 3,
                    neighbor: 6,
                    region: Region::new([Side::Right, Side::Right])
                }
            })
        );

        assert_eq!(
            coarse.next(),
            Some(&TreeBlockNeighbor {
                block: 0,
                neighbor: 2,
                a: TreeCellNeighbor {
                    cell: 2,
                    neighbor: 5,
                    region: Region::new([Side::Middle, Side::Right])
                },
                b: TreeCellNeighbor {
                    cell: 3,
                    neighbor: 5,
                    region: Region::new([Side::Middle, Side::Right])
                }
            })
        );
        assert_eq!(coarse.next(), None);
    }

    #[test]
    fn interfaces() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut dofs = TreeNodes::new([4; 2], 2);
        let mut neighbors = TreeNeighbors::default();
        let mut interfaces = TreeInterfaces::default();

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        dofs.build(&blocks);
        neighbors.build(&tree, &blocks);
        interfaces.build(&tree, &blocks, &neighbors, &dofs);

        let mut coarse = interfaces.coarse();

        assert_eq!(
            coarse.next(),
            Some(&TreeInterface {
                block: 0,
                neighbor: 1,
                source: [0, 0],
                dest: [8, 0],
                size: [3, 11],
            })
        );

        assert_eq!(
            coarse.next(),
            Some(&TreeInterface {
                block: 0,
                neighbor: 2,
                source: [0, 0],
                dest: [0, 8],
                size: [8, 3],
            })
        );

        assert_eq!(coarse.next(), None);

        let mut fine = interfaces.fine();

        assert_eq!(
            fine.next(),
            Some(&TreeInterface {
                block: 1,
                neighbor: 0,
                source: [2, 0],
                dest: [-2, 0],
                size: [3, 4],
            })
        );

        assert_eq!(
            fine.next(),
            Some(&TreeInterface {
                block: 2,
                neighbor: 0,
                source: [0, 2],
                dest: [0, -2],
                size: [4, 3],
            })
        );

        assert_eq!(fine.next(), None);

        let mut direct = interfaces.direct();

        assert_eq!(
            direct.next(),
            Some(&TreeInterface {
                block: 1,
                neighbor: 2,
                source: [2, 0],
                dest: [-2, 4],
                size: [3, 5],
            })
        );

        assert_eq!(
            direct.next(),
            Some(&TreeInterface {
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
