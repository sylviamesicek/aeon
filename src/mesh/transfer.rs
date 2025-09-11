use crate::geometry::{ActiveCellId, IndexSpace, NeighborId, Split};
use crate::image::ImageShared;
use crate::kernel::Interpolation;
use rayon::iter::{ParallelBridge, ParallelIterator as _};
use reborrow::ReborrowMut;
use std::array;

use crate::{
    image::{ImageMut, ImageRef},
    kernel::SystemBoundaryConds,
    mesh::Mesh,
    shared::SharedSlice,
};

impl<const N: usize> Mesh<N> {
    pub fn transfer_system(&mut self, order: usize, source: ImageRef, dest: ImageMut) {
        assert_eq!(source.num_channels(), dest.num_channels());
        assert_eq!(dest.num_nodes(), self.num_nodes());
        assert_eq!(source.num_nodes(), self.num_old_nodes());

        match order {
            2 => self.transfer_system_impl::<2>(source, dest),
            4 => self.transfer_system_impl::<4>(source, dest),
            6 => self.transfer_system_impl::<6>(source, dest),
            _ => unimplemented!("Order unimplemented"),
        }
    }

    /// Transfers data from an old version of the mesh to the new refined version.
    fn transfer_system_impl<const ORDER: usize>(&mut self, source: ImageRef, dest: ImageMut) {
        assert!(self.num_nodes() == dest.num_nodes());
        assert!(self.num_old_nodes() == source.num_nodes());

        let dest: ImageShared = dest.into();

        self.old_block_compute(|mesh, _store, block| {
            let size = mesh.old_blocks.size(block);
            let level = mesh.old_block_level(block);
            let nodes = mesh.old_block_nodes(block);
            let space = mesh.old_block_space(block);

            let block_source = source.slice(nodes.clone());

            for (i, offset) in IndexSpace::new(size).iter().enumerate() {
                let cell = mesh.old_blocks.active_cells(block)[i];
                let cell_origin: [_; N] = array::from_fn(|axis| offset[axis] * mesh.width);

                let new_cell = mesh.regrid_map[cell.0];
                let new_level = mesh.tree.active_level(new_cell);

                #[allow(clippy::comparison_chain)]
                if new_level > level {
                    // Loop over every child of the recently refined cell.
                    for child in Split::<N>::enumerate() {
                        // Retrieves new cell.
                        let new_cell = ActiveCellId(new_cell.0 + child.to_linear());

                        let new_block = mesh.blocks.active_cell_block(new_cell);
                        let new_nodes = mesh.block_nodes(new_block);
                        let new_space = mesh.block_space(new_block);

                        let mut block_dest = unsafe { dest.slice_mut(new_nodes.clone()) };

                        let mut cell_origin: [_; N] = array::from_fn(|axis| 2 * cell_origin[axis]);

                        for axis in 0..N {
                            if child.is_set(axis) {
                                cell_origin[axis] += mesh.width;
                            }
                        }

                        let node_size = mesh.cell_node_size(new_cell);
                        let node_origin = mesh.active_node_origin(new_cell);

                        for node_offset in IndexSpace::new(node_size).iter() {
                            let source_node = array::from_fn(|axis| {
                                (cell_origin[axis] + node_offset[axis]) as isize
                            });
                            let dest_node = array::from_fn(|axis| {
                                (node_origin[axis] + node_offset[axis]) as isize
                            });

                            for field in dest.channels() {
                                let v = space.prolong(
                                    Interpolation::<ORDER>,
                                    source_node,
                                    block_source.channel(field),
                                );
                                block_dest.channel_mut(field)
                                    [new_space.index_from_node(dest_node)] = v;
                            }
                        }
                    }
                } else if new_level == level {
                    // Direct copy
                    let new_block = mesh.blocks.active_cell_block(new_cell);
                    let new_nodes = mesh.block_nodes(new_block);
                    let new_space = mesh.block_space(new_block);

                    let mut block_dest = unsafe { dest.slice_mut(new_nodes.clone()) };

                    let node_size = mesh.cell_node_size(new_cell);
                    let node_origin = mesh.active_node_origin(new_cell);

                    for node_offset in IndexSpace::new(node_size).iter() {
                        let source_node =
                            array::from_fn(|axis| (cell_origin[axis] + node_offset[axis]) as isize);
                        let dest_node =
                            array::from_fn(|axis| (node_origin[axis] + node_offset[axis]) as isize);

                        for field in dest.channels() {
                            let v = block_source.channel(field)[space.index_from_node(source_node)];
                            block_dest.channel_mut(field)[new_space.index_from_node(dest_node)] = v;
                        }
                    }
                } else {
                    // Coarsening
                    let split = mesh.old_cell_splits[cell.0];

                    let new_block = mesh.blocks.active_cell_block(new_cell);
                    let new_offset = mesh.blocks.active_cell_position(new_cell);
                    let new_size = mesh.blocks.size(new_block);
                    let new_nodes = mesh.block_nodes(new_block);
                    let new_space = mesh.block_space(new_block);

                    let mut block_dest = unsafe { dest.slice_mut(new_nodes.clone()) };

                    let cell_origin: [_; N] = array::from_fn(|axis| cell_origin[axis] / 2);

                    let mut node_size = [mesh.width / 2; N];

                    for axis in 0..N {
                        if new_offset[axis] == new_size[axis] - 1 && split.is_set(axis) {
                            node_size[axis] += 1;
                        }
                    }

                    let mut node_origin: [_; N] =
                        array::from_fn(|axis| new_offset[axis] * mesh.width);

                    for axis in 0..N {
                        if split.is_set(axis) {
                            node_origin[axis] += mesh.width / 2;
                        }
                    }

                    for node_offset in IndexSpace::new(node_size).iter() {
                        let source_node = array::from_fn(|axis| {
                            2 * (cell_origin[axis] + node_offset[axis]) as isize
                        });
                        let dest_node =
                            array::from_fn(|axis| (node_origin[axis] + node_offset[axis]) as isize);

                        for field in dest.channels() {
                            let v = block_source.channel(field)[space.index_from_node(source_node)];
                            block_dest.channel_mut(field)[new_space.index_from_node(dest_node)] = v;
                        }
                    }
                }
            }
        });
    }

    /// Enforces strong boundary conditions. This includes strong physical boundary conditions, as well
    /// as handling interior boundaries (same level, coarse-fine, or fine-coarse).
    pub fn fill_boundary<BCs: SystemBoundaryConds<N> + Sync>(
        &mut self,
        order: usize,
        bcs: BCs,
        system: ImageMut,
    ) {
        assert_eq!(system.num_nodes(), self.num_nodes());

        self.fill_boundary_to_extent(order, self.ghost, bcs, system);
    }

    /// Enforces strong boundary conditions, only filling ghost nodes if those nodes are within `extent`
    /// of a physical node. This is useful if one is using Kriss-Olgier dissipation, where dissipation
    /// and derivatives use different order stencils.
    pub fn fill_boundary_to_extent<C: SystemBoundaryConds<N> + Sync>(
        &mut self,
        order: usize,
        extent: usize,
        bcs: C,
        mut system: ImageMut,
    ) {
        assert_eq!(system.num_nodes(), self.num_nodes());
        assert!(extent <= self.ghost);

        self.fill_fine(extent, system.rb_mut());
        self.fill_direct(extent, system.rb_mut());

        self.fill_physical(extent, &bcs, system.rb_mut());
        match order {
            2 => self.fill_prolong::<2>(extent, system.rb_mut()),
            4 => self.fill_prolong::<4>(extent, system.rb_mut()),
            6 => self.fill_prolong::<6>(extent, system.rb_mut()),
            _ => unimplemented!("Fill order not implemented"),
        }
        self.fill_physical(extent, &bcs, system.rb_mut());
    }

    fn fill_physical<C: SystemBoundaryConds<N> + Sync>(
        &mut self,
        extent: usize,
        bcs: &C,
        dest: ImageMut,
    ) {
        debug_assert!(dest.num_nodes() == self.num_nodes());

        let shared: ImageShared = dest.into();

        self.blocks.indices().par_bridge().for_each(|block| {
            // Fill Physical Boundary conditions
            let nodes = self.block_nodes(block);
            let space = self.block_space(block);
            let bcs = self.block_bcs(block, bcs.clone());

            let mut block_system = unsafe { shared.slice_mut(nodes) };

            for field in shared.channels() {
                space.fill_boundary(extent, bcs.field(field), block_system.channel_mut(field));
            }
        });
    }

    fn fill_direct(&mut self, extent: usize, result: ImageMut) {
        let shared: ImageShared = result.into();

        // Fill direct neighbors
        self.neighbors
            .direct_indices()
            .par_bridge()
            .for_each(|interface| {
                let info = self.interfaces.interface(interface);

                let block_space = self.block_space(info.block);
                let block_nodes = self.block_nodes(info.block);
                let neighbor_space = self.block_space(info.neighbor);
                let neighbor_nodes = self.block_nodes(info.neighbor);

                let dest = info.dest;
                let source = info.source;

                let mut block_system = unsafe { shared.slice_mut(block_nodes) };
                let neighbor_system = unsafe { shared.slice(neighbor_nodes) };

                for node in self.interfaces.interface_nodes_active(interface, extent) {
                    let block_node = array::from_fn(|axis| node[axis] as isize + dest[axis]);
                    let neighbor_node = array::from_fn(|axis| node[axis] as isize + source[axis]);

                    let block_index = block_space.index_from_node(block_node);
                    let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                    for field in shared.channels() {
                        let value = neighbor_system.channel(field)[neighbor_index];
                        block_system.channel_mut(field)[block_index] = value;
                    }
                }
            });
    }

    fn fill_fine(&mut self, extent: usize, result: ImageMut) {
        let shared: ImageShared = result.into();

        self.neighbors
            .fine_indices()
            .par_bridge()
            .for_each(|interface| {
                let info = self.interfaces.interface(interface);

                let block_space = self.block_space(info.block);
                let block_nodes = self.block_nodes(info.block);
                let neighbor_space = self.block_space(info.neighbor);
                let neighbor_nodes = self.block_nodes(info.neighbor);

                let mut block_system = unsafe { shared.slice_mut(block_nodes) };
                let neighbor_system = unsafe { shared.slice(neighbor_nodes) };

                for node in self.interfaces.interface_nodes_active(interface, extent) {
                    let block_node = array::from_fn(|axis| node[axis] as isize + info.dest[axis]);
                    let neighbor_node =
                        array::from_fn(|axis| 2 * (node[axis] as isize + info.source[axis]));

                    let block_index = block_space.index_from_node(block_node);
                    let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                    for field in shared.channels() {
                        let value = neighbor_system.channel(field)[neighbor_index];
                        block_system.channel_mut(field)[block_index] = value;
                    }
                }
            });
    }

    fn fill_prolong<const ORDER: usize>(&mut self, extent: usize, result: ImageMut) {
        let shared: ImageShared = result.into();

        self.neighbors
            .coarse_indices()
            .par_bridge()
            .for_each(|interface| {
                let info = self.interfaces.interface(interface);

                let block_nodes = self.block_nodes(info.block);
                let block_space = self.block_space(info.block);
                let neighbor_nodes = self.block_nodes(info.neighbor);
                let neighbor_space = self.block_space(info.neighbor);

                let mut block_system = unsafe { shared.slice_mut(block_nodes) };
                let neighbor_system = unsafe { shared.slice(neighbor_nodes) };

                for node in self.interfaces.interface_nodes_active(interface, extent) {
                    let block_node = array::from_fn(|axis| node[axis] as isize + info.dest[axis]);

                    let neighbor_node =
                        array::from_fn(|axis| node[axis] as isize + info.source[axis]);
                    let block_index = block_space.index_from_node(block_node);

                    for field in shared.channels() {
                        let value = neighbor_space.prolong(
                            Interpolation::<ORDER>,
                            neighbor_node,
                            neighbor_system.channel(field),
                        );
                        block_system.channel_mut(field)[block_index] = value;
                    }
                }
            });
    }

    /// Stores the index of the block which owns each individual node in a debug vector.
    pub fn block_debug(&mut self, debug: &mut [i64]) {
        assert!(debug.len() == self.num_nodes());

        let debug = SharedSlice::new(debug);

        self.block_compute(|mesh, _, block| {
            let block_nodes = mesh.block_nodes(block);

            for node in block_nodes {
                unsafe {
                    *debug.get_mut(node) = block.0 as i64;
                }
            }
        });
    }

    /// Stores the index of the cell which owns each individual node in a debug vector.
    pub fn cell_debug(&mut self, debug: &mut [i64]) {
        assert!(debug.len() == self.num_nodes());

        let debug = SharedSlice::new(debug);

        self.block_compute(|mesh, _, block| {
            let block_nodes = mesh.block_nodes(block);
            let block_space = mesh.block_space(block);
            let block_size = mesh.blocks.size(block);
            let cells = mesh.blocks.active_cells(block);

            for (i, position) in IndexSpace::new(block_size).iter().enumerate() {
                let cell = cells[i];
                let origin: [_; N] = array::from_fn(|axis| (position[axis] * mesh.width) as isize);

                for offset in IndexSpace::new([mesh.width + 1; N]).iter() {
                    let node = array::from_fn(|axis| origin[axis] + offset[axis] as isize);

                    let idx = block_nodes.start + block_space.index_from_node(node);

                    unsafe {
                        *debug.get_mut(idx) = cell.0 as i64;
                    }
                }
            }
        });
    }

    /// Stores the index of the neighbor along each interface.
    pub fn interface_neighbor_debug(&mut self, extent: usize, debug: &mut [i64]) {
        debug.fill(-1);

        let debug = SharedSlice::new(debug);

        self.interfaces
            .iter()
            .enumerate()
            .for_each(|(iidx, interface)| {
                let block_nodes = self.block_nodes(interface.block);
                let block_space = self.block_space(interface.block);

                for offset in self
                    .interfaces
                    .interface_nodes_active(NeighborId(iidx), extent)
                {
                    let node = array::from_fn(|axis| interface.dest[axis] + offset[axis] as isize);
                    let index = block_space.index_from_node(node);

                    unsafe {
                        *debug.get_mut(block_nodes.start + index) = interface.neighbor.0 as i64;
                    }
                }
            });
    }

    /// Stores the index of each interface for nodes along that interface.
    pub fn interface_index_debug(&mut self, extent: usize, debug: &mut [i64]) {
        debug.fill(-1);

        let debug = SharedSlice::new(debug);

        self.interfaces
            .iter()
            .enumerate()
            .for_each(|(iidx, interface)| {
                let block_nodes = self.block_nodes(interface.block);
                let block_space = self.block_space(interface.block);

                for offset in self
                    .interfaces
                    .interface_nodes_active(NeighborId(iidx), extent)
                {
                    let node = array::from_fn(|axis| interface.dest[axis] + offset[axis] as isize);
                    let index = block_space.index_from_node(node);

                    unsafe {
                        *debug.get_mut(block_nodes.start + index) = iidx as i64;
                    }
                }
            });
    }
}
