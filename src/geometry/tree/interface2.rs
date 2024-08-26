use crate::geometry::{regions, IndexSpace, Region, NULL};
use std::slice;

use super::{Tree, TreeBlocks};

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
                let aindex = blocks.cell_index(a.cell);
                let bindex = blocks.cell_index(b.cell);

                // Compute this boundary interface.
                let kind = InterfaceKind::from_levels(tree.level(a.cell), tree.level(a.neighbor));
                let interface = BlockInterface {
                    block,
                    neighbor,
                    acell: aindex,
                    aregion: a.region.clone(),
                    bcell: bindex,
                    bregion: b.region.clone(),
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
}

/// An interface between two blocks on a quad tree.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct BlockInterface<const N: usize> {
    /// Target block.
    pub block: usize,
    /// Source block.
    pub neighbor: usize,

    #[serde(with = "aeon_array")]
    pub acell: [usize; N],
    pub aregion: Region<N>,

    #[serde(with = "aeon_array")]
    pub bcell: [usize; N],
    pub bregion: Region<N>,
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
    use crate::geometry::Rectangle;

    use super::*;

    #[test]
    fn interfaces() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        let mut blocks = TreeBlocks::default();
        let mut interfaces = TreeInterfaces::default();

        tree.refine(&[true, false, false, false]);
        blocks.build(&tree);
        interfaces.build(&tree, &blocks);
    }
}
