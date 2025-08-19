use super::{blocks::BlockId, *};

#[derive(
    Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, serde::Serialize, serde::Deserialize,
)]
pub struct NeighborId(pub usize);

/// Stores neighbor of a cell on a tree.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeCellNeighbor<const N: usize> {
    /// Primary cell.
    pub cell: ActiveCellId,
    /// Neighbor cell.
    pub neighbor: ActiveCellId,
    /// Which region is the neighbor cell in?
    pub region: Region<N>,
    /// Which periodic region is the neighbor cell in?
    pub boundary_region: Region<N>,
}

/// Neighbor of block.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TreeBlockNeighbor<const N: usize> {
    /// Primary block.
    pub block: BlockId,
    /// Neighbor block.
    pub neighbor: BlockId,
    /// Leftmost cell neighbor.
    pub a: TreeCellNeighbor<N>,
    /// Rightmost cell neighbor.
    pub b: TreeCellNeighbor<N>,
}

impl<const N: usize> DataSize for TreeBlockNeighbor<N> {
    const IS_DYNAMIC: bool = false;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        0
    }
}

impl<const N: usize> TreeBlockNeighbor<N> {
    /// If this is a face neighbor, return the corresponding face, otherwise return `None`.
    pub fn face(&self) -> Option<Face<N>> {
        regions_to_face(self.a.region, self.b.region)
    }
}

fn regions_to_face<const N: usize>(a: Region<N>, b: Region<N>) -> Option<Face<N>> {
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
#[derive(Default, Clone, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub struct TreeNeighbors<const N: usize> {
    /// Flattened list of lists of neighbors for each block.
    neighbors: Vec<TreeBlockNeighbor<N>>,
    /// Offset map for blocks -> neighbors.
    block_offsets: Vec<usize>,
    /// A cached list of all fine interfaces.
    fine: Vec<usize>,
    /// A cached list of all direct interfaces.
    direct: Vec<usize>,
    /// A cached list of all coarse interfaces.
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

    pub fn indices(&self) -> impl Iterator<Item = NeighborId> {
        (0..self.neighbors.len()).map(NeighborId)
    }

    pub fn fine_indices(&self) -> impl Iterator<Item = NeighborId> + '_ {
        self.fine.iter().copied().map(NeighborId)
    }

    pub fn direct_indices(&self) -> impl Iterator<Item = NeighborId> + '_ {
        self.direct.iter().copied().map(NeighborId)
    }

    pub fn coarse_indices(&self) -> impl Iterator<Item = NeighborId> + '_ {
        self.coarse.iter().copied().map(NeighborId)
    }

    /// Iterates over all neighbors of a block.
    pub fn block(&self, block: BlockId) -> slice::Iter<'_, TreeBlockNeighbor<N>> {
        self.neighbors[self.block_offsets[block.0]..self.block_offsets[block.0 + 1]].iter()
    }

    /// Returns the range of neighbor indices belonging to a given block.
    pub fn block_range(&self, block: BlockId) -> Range<usize> {
        self.block_offsets[block.0]..self.block_offsets[block.0 + 1]
    }

    pub fn neighbor(&self, idx: NeighborId) -> &TreeBlockNeighbor<N> {
        &self.neighbors[idx.0]
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

        for block in blocks.indices() {
            self.block_offsets.push(self.neighbors.len());

            // Build cell neighbors.
            neighbors.clear();
            Self::build_cell_neighbors(tree, blocks, block, &mut neighbors);

            // Sort neighbors (to group cells from the same block together).
            neighbors.sort_unstable_by(|left, right| {
                let lblock = blocks.active_cell_block(left.neighbor);
                let rblock = blocks.active_cell_block(right.neighbor);

                left.boundary_region
                    .cmp(&right.boundary_region)
                    .then(lblock.cmp(&rblock))
                    .then(left.neighbor.cmp(&right.neighbor))
                    .then(left.cell.cmp(&right.cell))
                    .then(left.region.cmp(&right.region))
            });

            Self::taverse_cell_neighbors(blocks, &mut neighbors, |neighbor, a, b| {
                let acell = tree.cell_from_active_index(a.cell);
                let aneighbor = tree.cell_from_active_index(a.neighbor);

                // Compute this boundary interface.
                let kind = InterfaceKind::from_levels(tree.level(acell), tree.level(aneighbor));
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
        block: BlockId,
        neighbors: &mut Vec<TreeCellNeighbor<N>>,
    ) {
        let block_size = blocks.size(block);
        let block_active_cells = blocks.active_cells(block);
        let block_space = IndexSpace::new(block_size);

        debug_assert!(block_size.iter().product::<usize>() == block_active_cells.len());

        for region in regions::<N>() {
            if region == Region::CENTRAL {
                continue;
            }

            // Find all cells adjacent to the given region.
            for index in block_space.region_adjacent_window(region) {
                let active = block_active_cells[block_space.linear_from_cartesian(index)];
                let cell = tree.cell_from_active_index(active);
                let periodic = tree.boundary_region(cell, region);

                for neighbor in tree.active_neighbors_in_region(cell, region) {
                    neighbors.push(TreeCellNeighbor {
                        cell: active,
                        neighbor,
                        region,
                        boundary_region: periodic,
                    })
                }
            }
        }
    }

    /// Traverses a sorted list of cell neighbors, calling f once for each distinct block.
    fn taverse_cell_neighbors(
        blocks: &TreeBlocks<N>,
        neighbors: &mut [TreeCellNeighbor<N>],
        mut f: impl FnMut(BlockId, TreeCellNeighbor<N>, TreeCellNeighbor<N>),
    ) {
        let mut neighbors = neighbors.iter().cloned().peekable();

        while let Some(a) = neighbors.next() {
            let neighbor = blocks.active_cell_block(a.neighbor);

            // Next we walk through the iterator until we find the last neighbor that is still in this block.
            let mut b = a.clone();

            loop {
                if let Some(next) = neighbors.peek() {
                    if a.boundary_region == next.boundary_region
                        && neighbor == blocks.active_cell_block(next.neighbor)
                    {
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

impl<const N: usize> DataSize for TreeNeighbors<N> {
    const IS_DYNAMIC: bool = true;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        self.neighbors.estimate_heap_size()
            + self.block_offsets.estimate_heap_size()
            + self.fine.estimate_heap_size()
            + self.direct.estimate_heap_size()
            + self.coarse.estimate_heap_size()
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
    use super::*;
    use crate::geometry::HyperBox;

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
        let mut tree = Tree::new(HyperBox::<2>::UNIT);
        let mut blocks = TreeBlocks::new([4; 2], 2);
        let mut interfaces = TreeNeighbors::default();
        tree.refine(&[true]);
        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        interfaces.build(&tree, &blocks);

        let mut coarse = interfaces.coarse();

        assert_eq!(
            coarse.next(),
            Some(&TreeBlockNeighbor {
                block: BlockId(0),
                neighbor: BlockId(1),
                a: TreeCellNeighbor {
                    cell: ActiveCellId(1),
                    neighbor: ActiveCellId(4),
                    region: Region::new([Side::Right, Side::Middle]),
                    boundary_region: Region::CENTRAL,
                },
                b: TreeCellNeighbor {
                    cell: ActiveCellId(3),
                    neighbor: ActiveCellId(6),
                    region: Region::new([Side::Right, Side::Right]),
                    boundary_region: Region::CENTRAL,
                }
            })
        );

        assert_eq!(
            coarse.next(),
            Some(&TreeBlockNeighbor {
                block: BlockId(0),
                neighbor: BlockId(2),
                a: TreeCellNeighbor {
                    cell: ActiveCellId(2),
                    neighbor: ActiveCellId(5),
                    region: Region::new([Side::Middle, Side::Right]),
                    boundary_region: Region::CENTRAL,
                },
                b: TreeCellNeighbor {
                    cell: ActiveCellId(3),
                    neighbor: ActiveCellId(5),
                    region: Region::new([Side::Middle, Side::Right]),
                    boundary_region: Region::CENTRAL,
                }
            })
        );
        assert_eq!(coarse.next(), None);
    }
}
