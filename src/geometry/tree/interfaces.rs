use std::{array, cmp::Ordering, ops::Range};

use super::{
    NeighborId, Tree, TreeBlockNeighbor, TreeBlocks, TreeCellNeighbor, TreeNeighbors,
    blocks::BlockId,
};
use crate::{
    geometry::{IndexSpace, Side},
    kernel::NodeSpace,
    prelude::{FaceArray, HyperBox},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TreeInterface<const N: usize> {
    /// Block to be filled
    pub block: BlockId,
    /// Neighbor block.
    pub neighbor: BlockId,
    /// Source node in neighbor block.
    pub source: [isize; N],
    /// Destination node in target block.
    pub dest: [isize; N],
    /// Number of nodes to be filled along each axis.
    pub size: [usize; N],
}

impl<const N: usize> datasize::DataSize for TreeInterface<N> {
    const IS_DYNAMIC: bool = false;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        0
    }
}

struct TransferAABB<const N: usize> {
    /// Source node in neighbor block.
    source: [isize; N],
    /// Destination node in target block.
    dest: [isize; N],
    /// Number of nodes to be filled along each axis.
    size: [usize; N],
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct TreeInterfaces<const N: usize> {
    /// Interfaces build from neighbors.
    interfaces: Vec<TreeInterface<N>>,
    /// Offsets linking each interface to a range of ghost/face nodes.
    interface_node_offsets: Vec<usize>,
    /// Mask that keeps different interfaces disjoint.
    interface_masks: Vec<bool>,
}

impl<const N: usize> TreeInterfaces<N> {
    pub fn iter(&self) -> impl Iterator<Item = &TreeInterface<N>> {
        self.interfaces.iter()
    }

    /// Returns the total number of interface nodes.
    pub fn _num_interface_nodes(&self) -> usize {
        *self.interface_node_offsets.last().unwrap()
    }

    /// Retrieves the interface corresponding to the given neighbor.
    pub fn interface(&self, neighbor: NeighborId) -> &TreeInterface<N> {
        &self.interfaces[neighbor.0]
    }

    /// Returns the range of nodes associated with a given interface.
    pub fn interface_nodes(&self, interface: NeighborId) -> Range<usize> {
        self.interface_node_offsets[interface.0]..self.interface_node_offsets[interface.0 + 1]
    }

    /// Returns a mask of nodes for a given interface.
    pub fn interface_mask(&self, interface: NeighborId) -> &[bool] {
        &self.interface_masks[self.interface_nodes(interface)]
    }

    /// Returns an index space corresponding to a given interface.
    pub fn interface_space(&self, interface: NeighborId) -> IndexSpace<N> {
        IndexSpace::new(self.interfaces[interface.0].size)
    }

    /// Iterates over active index nodes.
    pub fn interface_nodes_active(
        &self,
        interface: NeighborId,
        _extent: usize,
    ) -> impl Iterator<Item = [usize; N]> + '_ {
        let space = self.interface_space(interface);
        let mask = self.interface_mask(interface);

        space
            .iter()
            .filter(move |&offset| mask[space.linear_from_cartesian(offset)])
    }

    pub fn build(&mut self, tree: &Tree<N>, blocks: &TreeBlocks<N>, neighbors: &TreeNeighbors<N>) {
        self.interfaces.clear();
        self.interface_node_offsets.clear();
        self.interface_masks.clear();

        for neighbor in neighbors.iter() {
            let aabb = self.transfer_aabb(tree, blocks, neighbor);

            self.interfaces.push(TreeInterface {
                block: neighbor.block,
                neighbor: neighbor.neighbor,
                source: aabb.source,
                dest: aabb.dest,
                size: aabb.size,
            });
        }

        // ************************
        // Compute node offsets

        let mut cursor = 0;

        for interface in self.interfaces.iter() {
            self.interface_node_offsets.push(cursor);
            cursor += IndexSpace::new(interface.size).index_count();
        }

        self.interface_node_offsets.push(cursor);

        // **************************
        // Compute interface masks

        self.interface_masks.resize(cursor, false);

        let mut buffer = Vec::default();

        for block in blocks.indices() {
            let block_size = blocks.size(block);
            let block_level = blocks.level(block);
            let block_space = NodeSpace {
                size: array::from_fn(|axis| block_size[axis] * blocks.width()[axis]),
                ghost: blocks.ghost(),
                bounds: HyperBox::<N>::UNIT,
                boundary: FaceArray::default(),
            };

            buffer.resize(block_space.num_nodes(), usize::MAX);
            buffer.fill(usize::MAX);

            // Fill buffer
            for iidx in neighbors.block_range(block) {
                let interface = &self.interfaces[iidx];
                let interface_level = blocks.level(interface.neighbor);

                debug_assert!(interface.block == block);

                for offset in IndexSpace::new(interface.size).iter() {
                    let node = array::from_fn(|axis| interface.dest[axis] + offset[axis] as isize);
                    let index = block_space.index_from_node(node);

                    if buffer[index] == usize::MAX {
                        buffer[index] = iidx;
                        continue;
                    }

                    let other_interface = &self.interfaces[buffer[index]];
                    let other_neighbor = other_interface.neighbor;

                    debug_assert!(other_interface.block == interface.block);

                    let other_level = blocks.level(other_neighbor);

                    // If new interface is coarser than old interface, prefer it
                    //
                    // Otherwise if they are the same level, but the other interface
                    // points to a block with a smaller index, prefer it.
                    if interface_level < other_level
                        || (interface_level == other_level && interface.neighbor < other_neighbor)
                    {
                        buffer[index] = iidx;
                    }
                }
            }

            // Now compute masks
            for iidx in neighbors.block_range(block) {
                let interface = &self.interfaces[iidx];
                let interface_level = blocks.level(interface.neighbor);

                debug_assert!(interface.block == block);

                let interface_mask_offset = self.interface_node_offsets[iidx];

                for (dst, offset) in IndexSpace::new(interface.size).iter().enumerate() {
                    let node = array::from_fn(|axis| interface.dest[axis] + offset[axis] as isize);
                    let src = block_space.index_from_node(node);

                    // Only fill if this point belongs to interface.
                    let mut mask = buffer[src] == iidx;

                    if interface_level == block_level
                        && interface.block <= interface.neighbor
                        && block_space.is_interior(node)
                    {
                        mask = false;
                    }

                    // Set mask value
                    self.interface_masks[interface_mask_offset + dst] = mask;
                }
            }
        }
    }

    /// Computes transfer aabb for nodes that lie outside the node space.
    fn transfer_aabb(
        &self,
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        neighbor: &TreeBlockNeighbor<N>,
    ) -> TransferAABB<N> {
        let a = neighbor.a.clone();
        let b = neighbor.b.clone();

        let block_level = tree.active_level(a.cell);
        let neighbor_level = tree.active_level(a.neighbor);

        // ********************************
        // Dest

        let (mut anode, mut bnode) = self.transfer_dest(tree, blocks, neighbor);

        // **************************
        // Source

        let mut source = self.transfer_source(tree, blocks, &neighbor.a);

        for axis in 0..N {
            if b.region.side(axis) == Side::Left && block_level < neighbor_level {
                debug_assert!(a.region.side(axis) == Side::Left);
                bnode[axis] -= 1;
            }

            if a.region.side(axis) == Side::Right && block_level < neighbor_level {
                debug_assert!(b.region.side(axis) == Side::Right);

                anode[axis] += 1;
                source[axis] += 1;
            }
        }

        let size = array::from_fn(|axis| {
            if bnode[axis] >= anode[axis] {
                (bnode[axis] - anode[axis] + 1) as usize
            } else {
                0
            }
        });

        TransferAABB {
            source,
            dest: anode,
            size,
        }
    }

    #[inline]
    fn transfer_dest(
        &self,
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        interface: &TreeBlockNeighbor<N>,
    ) -> ([isize; N], [isize; N]) {
        let a = interface.a.clone();
        let b = interface.b.clone();

        // Find ghost region that must be filled
        let aindex = blocks.active_cell_position(a.cell);
        let bindex = blocks.active_cell_position(b.cell);

        let block_level = tree.active_level(a.cell);
        let neighbor_level = tree.active_level(a.neighbor);

        // ********************************
        // A node

        // Compute bottom left corner of A cell.
        let mut anode: [_; N] =
            array::from_fn(|axis| (aindex[axis] * blocks.width()[axis]) as isize);

        if block_level < neighbor_level {
            let split = tree.most_recent_active_split(a.neighbor).unwrap();
            (0..N)
                .filter(|&axis| a.region.side(axis) == Side::Middle && split.is_set(axis))
                .for_each(|axis| anode[axis] += (blocks.width()[axis] / 2) as isize);
        }

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match a.region.side(axis) {
                Side::Left => {
                    anode[axis] -= blocks.ghost() as isize;
                }
                Side::Right => {
                    anode[axis] += blocks.width()[axis] as isize;
                }
                Side::Middle => {}
            }
        }

        // ***********************************
        // B Node

        // Compute top right corner of B cell
        let mut bnode: [_; N] =
            array::from_fn(|axis| ((bindex[axis] + 1) * blocks.width()[axis]) as isize);

        if block_level < neighbor_level {
            let split = tree.most_recent_active_split(b.neighbor).unwrap();
            (0..N)
                .filter(|&axis| b.region.side(axis) == Side::Middle && !split.is_set(axis))
                .for_each(|axis| bnode[axis] -= (blocks.width()[axis] / 2) as isize);
        }

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match b.region.side(axis) {
                Side::Right => {
                    bnode[axis] += blocks.ghost() as isize;
                }
                Side::Left => {
                    bnode[axis] -= blocks.width()[axis] as isize;
                }
                Side::Middle => {}
            }
        }

        (anode, bnode)
    }

    /// Computes the source node on the neighboring block from which we fill the current block.
    #[inline]
    fn transfer_source(
        &self,
        tree: &Tree<N>,
        blocks: &TreeBlocks<N>,
        a: &TreeCellNeighbor<N>,
    ) -> [isize; N] {
        let block_level = tree.active_level(a.cell);
        let neighbor_level = tree.active_level(a.neighbor);

        // Find source node
        let nindex = blocks.active_cell_position(a.neighbor);
        let mut source: [isize; N] =
            array::from_fn(|axis| (nindex[axis] * blocks.width()[axis]) as isize);

        match block_level.cmp(&neighbor_level) {
            Ordering::Equal => {
                for axis in 0..N {
                    if a.region.side(axis) == Side::Left {
                        source[axis] += (blocks.width()[axis] - blocks.ghost()) as isize;
                    }
                }
            }
            Ordering::Greater => {
                // Source is stored in subnodes
                for axis in 0..N {
                    source[axis] *= 2;
                }

                let split = tree.most_recent_active_split(a.cell).unwrap();

                for axis in 0..N {
                    if split.is_set(axis) {
                        match a.region.side(axis) {
                            Side::Left => {
                                source[axis] +=
                                    blocks.width()[axis] as isize - blocks.ghost() as isize
                            }
                            Side::Middle => source[axis] += blocks.width()[axis] as isize,
                            Side::Right => {}
                        }
                    } else {
                        match a.region.side(axis) {
                            Side::Left => {
                                source[axis] +=
                                    2 * blocks.width()[axis] as isize - blocks.ghost() as isize
                            }
                            Side::Middle => {}
                            Side::Right => source[axis] += blocks.width()[axis] as isize,
                        }
                    }
                }
            }
            Ordering::Less => {
                // Source is stored in supernodes
                for axis in 0..N {
                    source[axis] /= 2;
                }

                for axis in 0..N {
                    if a.region.side(axis) == Side::Left {
                        source[axis] += blocks.width()[axis] as isize / 2 - blocks.ghost() as isize;
                    }
                }
            }
        }

        source
    }
}

impl<const N: usize> datasize::DataSize for TreeInterfaces<N> {
    const IS_DYNAMIC: bool = true;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        self.interfaces.estimate_heap_size()
            + self.interface_node_offsets.estimate_heap_size()
            + self.interface_masks.estimate_heap_size()
    }
}
