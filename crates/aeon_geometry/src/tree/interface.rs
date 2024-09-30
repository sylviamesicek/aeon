use crate::{regions, IndexSpace, Region, NULL};
use crate::{Face, Side, Tree, TreeBlocks};
use std::ops::Range;
use std::slice;

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
    neighbors: Vec<TreeBlockNeighbor<N>>,
    block_offsets: Vec<usize>,
    fine: Vec<usize>,
    direct: Vec<usize>,
    coarse: Vec<usize>,
}

impl<const N: usize> TreeNeighbors<N> {
    /// Iterates over all fine `BlockInterface`s.
    pub fn fine(&self) -> impl Iterator<Item = &TreeBlockNeighbor<N>> {
        self.fine.iter().map(|&i| &self.neighbors[i])
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn direct(&self) -> impl Iterator<Item = &TreeBlockNeighbor<N>> {
        self.direct.iter().map(|&i| &self.neighbors[i])
    }

    /// Iterates over all fine `BlockInterface`s.
    pub fn coarse(&self) -> impl Iterator<Item = &TreeBlockNeighbor<N>> {
        self.coarse.iter().map(|&i| &self.neighbors[i])
    }

    /// Iterates over all interfaces in the mesh.
    pub fn iter(&self) -> slice::Iter<'_, TreeBlockNeighbor<N>> {
        self.neighbors.iter()
    }

    pub fn fine_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.fine.iter().copied()
    }

    pub fn direct_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.direct.iter().copied()
    }

    pub fn coarse_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.coarse.iter().copied()
    }

    /// Iterates over all neighbors of a block.
    pub fn block(&self, block: usize) -> slice::Iter<'_, TreeBlockNeighbor<N>> {
        self.neighbors[self.block_offsets[block]..self.block_offsets[block + 1]].iter()
    }

    /// Returns the range of neighbor indices belonging to a given block.
    pub fn block_range(&self, block: usize) -> Range<usize> {
        self.block_offsets[block]..self.block_offsets[block + 1]
    }

    pub fn neighbor(&self, idx: usize) -> &TreeBlockNeighbor<N> {
        &self.neighbors[idx]
    }

    /// Rebuilds the block interface data.
    pub fn build(&mut self, tree: &Tree<N>, blocks: &TreeBlocks<N>) {
        self.neighbors.clear();
        self.block_offsets.clear();
        self.fine.clear();
        self.coarse.clear();
        self.direct.clear();

        // Reused memory for neighbors.
        let mut neighbors = Vec::new();

        for block in 0..blocks.len() {
            self.block_offsets.push(self.neighbors.len());

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

                let idx = self.neighbors.len();
                self.neighbors.push(interface);

                match kind {
                    InterfaceKind::Fine => self.fine.push(idx),
                    InterfaceKind::Direct => self.direct.push(idx),
                    InterfaceKind::Coarse => self.coarse.push(idx),
                }
            });
        }

        self.block_offsets.push(self.neighbors.len());
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

    // #[ignore = "Outdated interface test."]
    // #[test]
    // fn interfaces() {
    //     let mut tree = Tree::new(Rectangle::<2>::UNIT);
    //     let mut blocks = TreeBlocks::default();
    //     let mut dofs = TreeNodes::new([4; 2], 2);
    //     let mut neighbors = TreeNeighbors::default();
    //     let mut interfaces = TreeInterfaces::default();

    //     tree.refine(&[true, false, false, false]);
    //     blocks.build(&tree);
    //     dofs.build(&blocks);
    //     neighbors.build(&tree, &blocks);
    //     interfaces.build(&tree, &blocks, &neighbors, &dofs);

    //     let mut coarse = interfaces.coarse();

    //     assert_eq!(
    //         coarse.next(),
    //         Some(&TreeInterface {
    //             block: 0,
    //             neighbor: 1,
    //             source: [0, 0],
    //             dest: [8, 0],
    //             size: [3, 11],
    //         })
    //     );

    //     assert_eq!(
    //         coarse.next(),
    //         Some(&TreeInterface {
    //             block: 0,
    //             neighbor: 2,
    //             source: [0, 0],
    //             dest: [0, 8],
    //             size: [8, 3],
    //         })
    //     );

    //     assert_eq!(coarse.next(), None);

    //     let mut fine = interfaces.fine();

    //     assert_eq!(
    //         fine.next(),
    //         Some(&TreeInterface {
    //             block: 1,
    //             neighbor: 0,
    //             source: [2, 0],
    //             dest: [-2, 0],
    //             size: [3, 4],
    //         })
    //     );

    //     assert_eq!(
    //         fine.next(),
    //         Some(&TreeInterface {
    //             block: 2,
    //             neighbor: 0,
    //             source: [0, 2],
    //             dest: [0, -2],
    //             size: [4, 3],
    //         })
    //     );

    //     assert_eq!(fine.next(), None);

    //     let mut direct = interfaces.direct();

    //     assert_eq!(
    //         direct.next(),
    //         Some(&TreeInterface {
    //             block: 1,
    //             neighbor: 2,
    //             source: [2, 0],
    //             dest: [-2, 4],
    //             size: [3, 5],
    //         })
    //     );

    //     assert_eq!(
    //         direct.next(),
    //         Some(&TreeInterface {
    //             block: 2,
    //             neighbor: 1,
    //             source: [0, 2],
    //             dest: [4, -2],
    //             size: [3, 7],
    //         })
    //     );

    //     assert_eq!(direct.next(), None);
    // }
}
