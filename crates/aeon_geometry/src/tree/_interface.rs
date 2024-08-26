use crate::{regions, Face, IndexSpace, Region, Side, NULL};
use std::array::from_fn;
use std::slice;

use super::{Tree, TreeBlocks, TreeNodes};

/// Caches information about interior interfaces on quadtrees, specifically
/// storing information necessary for transferring data between blocks.
#[derive(Default, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TreeInterfaces<const N: usize> {
    fine: Vec<BlockInterface<N>>,
    direct: Vec<BlockInterface<N>>,
    coarse: Vec<BlockInterface<N>>,
}

impl<const N: usize> TreeInterfaces<N> {
    /// Iterates over all fine `BlockInterface`s.
    pub fn fine(&self) -> slice::Iter<'_, BlockInterface<N>> {
        self.fine.iter()
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn direct(&self) -> slice::Iter<'_, BlockInterface<N>> {
        self.direct.iter()
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn coarse(&self) -> slice::Iter<'_, BlockInterface<N>> {
        self.coarse.iter()
    }

    /// Rebuilds the block interface data.
    pub fn build(&mut self, tree: &Tree<N>, blocks: &TreeBlocks<N>, nodes: &TreeNodes<N>) {
        // Reused memory for neighbors.
        let mut neighbors = Vec::new();

        for block in 0..blocks.num_blocks() {
            // Cache block info
            let block_size = blocks.block_size(block);
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
                // Find active region.
                let (anode, bnode) = Self::block_ghost_aabb(tree, blocks, nodes, a, b);
                let mut source = Self::neighbor_origin(tree, blocks, nodes, a);
                let (mut dest, mut size) = Self::space_from_aabb(anode, bnode);

                // Avoid overlaps between aabbs on this block.
                // let aorigin = blocks.cell_index(a.cell);
                // let borigin = blocks.cell_index(b.cell);
                let flags = blocks.block_boundary_flags(block);

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
                        && !(bnode[axis] == (block_size[axis] * nodes.cell_width[axis]) as isize
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

                // Compute this boundary interface.
                let kind = InterfaceKind::from_levels(tree.level(a.cell), tree.level(a.neighbor));
                let interface = BlockInterface {
                    block,
                    neighbor,
                    source,
                    dest,
                    size,
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
        neighbors: &mut Vec<CellNeighbor<N>>,
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

                    neighbors.push(CellNeighbor {
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
        neighbors: &mut [CellNeighbor<N>],
        mut f: impl FnMut(usize, CellNeighbor<N>, CellNeighbor<N>),
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

    /// Computes the nodes that ghost nodes adjacent to the AABB of cells
    /// defined by A and B.
    fn block_ghost_aabb(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        nodes: &TreeNodes<N>,
        a: CellNeighbor<N>,
        b: CellNeighbor<N>,
    ) -> ([isize; N], [isize; N]) {
        // Find ghost region that must be filled
        let aindex = blocks.cell_index(a.cell);
        let bindex = blocks.cell_index(b.cell);

        // Compute bottom right corner of A cell.
        let mut anode: [_; N] = from_fn(|axis| (aindex[axis] * nodes.cell_width[axis]) as isize);

        if tree.level(a.cell) < tree.level(a.neighbor) {
            let split = tree.split(a.neighbor);
            (0..N)
                .filter(|&axis| a.region.side(axis) == Side::Middle && split.is_set(axis))
                .for_each(|axis| anode[axis] += (nodes.cell_width[axis] / 2) as isize);
        }

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match a.region.side(axis) {
                Side::Left => {
                    anode[axis] -= nodes.ghost as isize;
                }
                Side::Right => {
                    anode[axis] += nodes.cell_width[axis] as isize;
                }
                Side::Middle => {}
            }
        }

        // Compute top left corner of B cell
        let mut bnode: [_; N] =
            from_fn(|axis| ((bindex[axis] + 1) * nodes.cell_width[axis]) as isize);

        if tree.level(b.cell) < tree.level(b.neighbor) {
            let split = tree.split(b.neighbor);
            (0..N)
                .filter(|&axis| b.region.side(axis) == Side::Middle && !split.is_set(axis))
                .for_each(|axis| bnode[axis] -= (nodes.cell_width[axis] / 2) as isize);
        }

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match b.region.side(axis) {
                Side::Right => {
                    bnode[axis] += nodes.ghost as isize;
                }
                Side::Left => {
                    bnode[axis] -= nodes.cell_width[axis] as isize;
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
        let size = from_fn(|axis| (b[axis] - a[axis] + 1) as usize);

        (dest, size)
    }

    fn neighbor_origin(
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        nodes: &TreeNodes<N>,
        a: CellNeighbor<N>,
    ) -> [isize; N] {
        // I think it works?

        // Compute this boundary interface.
        let interface = InterfaceKind::from_levels(tree.level(a.cell), tree.level(a.neighbor));

        // Find source node
        let index = blocks.cell_index(a.neighbor);
        let mut source: [isize; N] =
            from_fn(|axis| (index[axis] * nodes.cell_width[axis]) as isize);

        match interface {
            InterfaceKind::Direct => {
                for axis in 0..N {
                    if a.region.side(axis) == Side::Left {
                        source[axis] += (nodes.cell_width[axis] - nodes.ghost) as isize;
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
                                source[axis] +=
                                    nodes.cell_width[axis] as isize - nodes.ghost as isize
                            }
                            Side::Middle => source[axis] += nodes.cell_width[axis] as isize,
                            Side::Right => {}
                        }
                    } else {
                        match a.region.side(axis) {
                            Side::Left => {
                                source[axis] +=
                                    2 * nodes.cell_width[axis] as isize - nodes.ghost as isize
                            }
                            Side::Middle => {}
                            Side::Right => source[axis] += nodes.cell_width[axis] as isize,
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
                        source[axis] += nodes.cell_width[axis] as isize / 2 - nodes.ghost as isize;
                    }
                }
            }
        }

        source
    }
}

/// An interface between two blocks on a quad tree.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlockInterface<const N: usize> {
    /// Target block.
    pub block: usize,
    /// Source block.
    pub neighbor: usize,
    /// Source node on neighbor.
    #[serde(with = "aeon_array")]
    pub source: [isize; N],
    /// Destination node on target.
    #[serde(with = "aeon_array")]
    pub dest: [isize; N],
    /// Number of blocks to be filled.
    #[serde(with = "aeon_array")]
    pub size: [usize; N],
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

#[derive(Clone, Copy)]
struct CellNeighbor<const N: usize> {
    cell: usize,
    neighbor: usize,
    region: Region<N>,
}

#[cfg(test)]
mod tests {
    use crate::Rectangle;

    use super::*;

    #[test]
    fn interfaces() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut nodes = TreeNodes::new([4; 2], 2);
        let mut interfaces = TreeInterfaces::default();

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        nodes.build(&blocks);
        interfaces.build(&tree, &blocks, &nodes);

        let mut coarse = interfaces.coarse();

        assert_eq!(
            coarse.next(),
            Some(&BlockInterface {
                block: 0,
                neighbor: 1,
                source: [0, 0],
                dest: [8, 0],
                size: [3, 11],
            })
        );

        assert_eq!(
            coarse.next(),
            Some(&BlockInterface {
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
            Some(&BlockInterface {
                block: 1,
                neighbor: 0,
                source: [2, 0],
                dest: [-2, 0],
                size: [3, 4],
            })
        );

        assert_eq!(
            fine.next(),
            Some(&BlockInterface {
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
            Some(&BlockInterface {
                block: 1,
                neighbor: 2,
                source: [2, 0],
                dest: [-2, 4],
                size: [3, 5],
            })
        );

        assert_eq!(
            direct.next(),
            Some(&BlockInterface {
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
