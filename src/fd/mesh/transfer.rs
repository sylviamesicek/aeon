use aeon_basis::{Boundary, BoundaryKind, Condition, Kernels};
use aeon_geometry::{faces, AxisMask, Face, IndexSpace, Side, TreeBlockNeighbor, TreeCellNeighbor};
use reborrow::ReborrowMut;
use std::{array, cmp::Ordering, ops::Range};

use crate::{
    fd::{Conditions, Engine, FdEngine, Mesh, ScalarConditions, SystemBC},
    shared::SharedSlice,
    system::{System, SystemSlice, SystemSliceMut},
};

#[derive(Clone, Debug)]
pub(super) struct TreeInterface<const N: usize> {
    /// Block to be filled
    block: usize,
    /// Neighbor block.
    neighbor: usize,
    /// Source node in neighbor block.
    source: [isize; N],
    /// Destination node in target block.
    dest: [isize; N],
    /// Number of nodes to be filled along each axis.
    size: [usize; N],
}

struct TransferAABB<const N: usize> {
    /// Source node in neighbor block.
    source: [isize; N],
    /// Destination node in target block.
    dest: [isize; N],
    /// Number of nodes to be filled along each axis.
    size: [usize; N],
}

impl<const N: usize> Mesh<N> {
    /// Retrieves the nth interface.
    fn interface(&self, interface: usize) -> &TreeInterface<N> {
        &self.interfaces[interface]
    }

    /// Returns the total number of interface nodes.
    fn _num_interface_nodes(&self) -> usize {
        *self.interface_node_offsets.last().unwrap()
    }

    /// Returns the range of nodes associated with a given interface.
    fn interface_nodes(&self, interface: usize) -> Range<usize> {
        self.interface_node_offsets[interface]..self.interface_node_offsets[interface + 1]
    }

    /// Returns a mask of nodes for a given interface.
    fn interface_mask(&self, interface: usize) -> &[bool] {
        &self.interface_masks[self.interface_nodes(interface)]
    }

    fn interface_space(&self, interface: usize) -> IndexSpace<N> {
        IndexSpace::new(self.interfaces[interface].size)
    }

    /// Iterates over active index nodes.
    fn interface_nodes_active(
        &self,
        interface: usize,
        _extent: usize,
    ) -> impl Iterator<Item = [usize; N]> + '_ {
        let space = self.interface_space(interface);
        let mask = self.interface_mask(interface);

        space
            .iter()
            .filter(move |&offset| mask[space.linear_from_cartesian(offset)])
    }

    /// Transfers data from an old version of the mesh to the new refined version.
    pub fn transfer_system<K: Kernels, B: Boundary<N> + Sync, S: System>(
        &mut self,
        _order: K,
        boundary: B,
        source: SystemSlice<S>,
        dest: SystemSliceMut<S>,
    ) {
        assert!(self.num_nodes() == dest.len());
        assert!(self.num_old_nodes() == source.len());

        let dest = dest.into_shared();

        self.old_block_compute(|mesh, _store, block| {
            let size = mesh.old_blocks.size(block);
            let level = mesh.old_block_level(block);
            let nodes = mesh.old_block_nodes(block);
            let space = mesh.old_block_space(block);
            let boundary = mesh.old_block_boundary(block, boundary.clone());

            let block_source = source.slice(nodes.clone());

            for (i, offset) in IndexSpace::new(size).iter().enumerate() {
                let cell = mesh.old_blocks.cells(block)[i];
                let cell_origin: [_; N] = array::from_fn(|axis| offset[axis] * mesh.width);

                let new_cell = mesh.regrid_map[cell];
                let new_level = mesh.tree.level(new_cell);

                #[allow(clippy::comparison_chain)]
                if new_level > level {
                    // Loop over every child of the recently refined cell.
                    for child in AxisMask::<N>::enumerate() {
                        // Retrieves new cell.
                        let new_cell = new_cell + child.to_linear();

                        let new_block = mesh.blocks.cell_block(new_cell);
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
                        let node_origin = mesh.cell_node_origin(new_cell);

                        for node_offset in IndexSpace::new(node_size).iter() {
                            let source_vertex =
                                array::from_fn(|axis| cell_origin[axis] + node_offset[axis]);
                            let dest_node = array::from_fn(|axis| {
                                (node_origin[axis] + node_offset[axis]) as isize
                            });

                            for field in dest.system().enumerate() {
                                let v = space.prolong(
                                    boundary.clone(),
                                    K::interpolation().clone(),
                                    source_vertex,
                                    block_source.field(field),
                                );
                                block_dest.field_mut(field)[new_space.index_from_node(dest_node)] =
                                    v;
                            }
                        }
                    }
                } else if new_level == level {
                    // Direct copy
                    let new_block = mesh.blocks.cell_block(new_cell);
                    let new_nodes = mesh.block_nodes(new_block);
                    let new_space = mesh.block_space(new_block);

                    let mut block_dest = unsafe { dest.slice_mut(new_nodes.clone()) };

                    let node_size = mesh.cell_node_size(new_cell);
                    let node_origin = mesh.cell_node_origin(new_cell);

                    for node_offset in IndexSpace::new(node_size).iter() {
                        let source_node =
                            array::from_fn(|axis| (cell_origin[axis] + node_offset[axis]) as isize);
                        let dest_node =
                            array::from_fn(|axis| (node_origin[axis] + node_offset[axis]) as isize);

                        for field in dest.system().enumerate() {
                            let v = block_source.field(field)[space.index_from_node(source_node)];
                            block_dest.field_mut(field)[new_space.index_from_node(dest_node)] = v;
                        }
                    }
                } else {
                    // Coarsening
                    let split = mesh.old_cell_splits[cell];

                    let new_block = mesh.blocks.cell_block(new_cell);
                    let new_offset = mesh.blocks.cell_position(new_cell);
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

                        for field in dest.system().enumerate() {
                            let v = block_source.field(field)[space.index_from_node(source_node)];
                            block_dest.field_mut(field)[new_space.index_from_node(dest_node)] = v;
                        }
                    }
                }
            }
        });
    }

    /// Enforces strong boundary conditions. This includes strong physical boundary conditions, as well
    /// as handling interior boundaries (same level, coarse-fine, or fine-coarse).
    pub fn fill_boundary<K: Kernels, B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        order: K,
        boundary: B,
        conditions: C,
        system: SystemSliceMut<'_, C::System>,
    ) where
        C::System: Clone,
    {
        self.fill_boundary_to_extent(order, self.ghost, boundary, conditions, system);
    }

    pub fn fill_boundary_scalar<K: Kernels, B: Boundary<N> + Sync, C: Condition<N> + Sync>(
        &mut self,
        order: K,
        boundary: B,
        condition: C,
        system: &mut [f64],
    ) {
        self.fill_boundary_to_extent(
            order,
            self.ghost,
            boundary,
            ScalarConditions(condition),
            system.into(),
        );
    }

    pub fn fill_boundary_to_extent<K: Kernels, B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        order: K,
        extent: usize,
        boundary: B,
        conditions: C,
        mut system: SystemSliceMut<'_, C::System>,
    ) where
        C::System: Clone,
    {
        self.fill_fine(extent, system.rb_mut());
        self.fill_direct(extent, system.rb_mut());

        self.fill_physical(extent, &boundary, &conditions, system.rb_mut());
        self.fill_prolong(order, extent, &boundary, &conditions, system.rb_mut());
        self.fill_physical(extent, &boundary, &conditions, system.rb_mut());
    }

    pub fn fill_boundary_zeros<S: System>(&mut self, dest: SystemSliceMut<S>) {
        let shared = dest.into_shared();

        (0..self.interfaces.len()).for_each(|interface| {
            let info = self.interface(interface);

            let block_space = self.block_space(info.block);
            let block_nodes = self.block_nodes(info.block);

            let dest = info.dest;

            let mut block_system = unsafe { shared.slice_mut(block_nodes) };

            for node in self.interface_nodes_active(interface, self.ghost) {
                let block_node = array::from_fn(|axis| node[axis] as isize + dest[axis]);
                let block_index = block_space.index_from_node(block_node);

                for field in shared.system().enumerate() {
                    block_system.field_mut(field)[block_index] = 0.0;
                }
            }
        });
    }

    fn fill_physical<B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        extent: usize,
        boundary: &B,
        conditions: &C,
        dest: SystemSliceMut<'_, C::System>,
    ) {
        debug_assert!(dest.len() == self.num_nodes());

        let shared = dest.into_shared();

        (0..self.blocks.len()).for_each(|block| {
            // Fill Physical Boundary conditions
            let nodes = self.block_nodes(block);
            let space = self.block_space(block);
            let boundary = self.block_boundary(block, boundary.clone());

            let mut block_system = unsafe { shared.slice_mut(nodes) };

            for field in shared.system().enumerate() {
                let bcs = SystemBC::new(boundary.clone(), conditions.clone(), field);
                space.fill_boundary(extent, bcs, block_system.field_mut(field));
            }
        });
    }

    fn fill_direct<S: System>(&mut self, extent: usize, result: SystemSliceMut<S>) {
        let shared = result.into_shared();

        // Fill direct neighbors
        self.neighbors.direct_indices().for_each(|interface| {
            let info = self.interface(interface);

            let block_space = self.block_space(info.block);
            let block_nodes = self.block_nodes(info.block);
            let neighbor_space = self.block_space(info.neighbor);
            let neighbor_nodes = self.block_nodes(info.neighbor);

            let dest = info.dest;
            let source = info.source;

            let mut block_system = unsafe { shared.slice_mut(block_nodes) };
            let neighbor_system = unsafe { shared.slice(neighbor_nodes) };

            for node in self.interface_nodes_active(interface, extent) {
                let block_node = array::from_fn(|axis| node[axis] as isize + dest[axis]);
                let neighbor_node = array::from_fn(|axis| node[axis] as isize + source[axis]);

                let block_index = block_space.index_from_node(block_node);
                let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                for field in shared.system().enumerate() {
                    let value = neighbor_system.field(field)[neighbor_index];
                    block_system.field_mut(field)[block_index] = value;
                }
            }
        });
    }

    fn fill_fine<S: System + Clone>(&mut self, extent: usize, result: SystemSliceMut<S>) {
        let system = result.system().clone();
        let shared = result.into_shared();

        self.neighbors.fine_indices().for_each(|interface| {
            let info = self.interface(interface);

            let block_space = self.block_space(info.block);
            let block_nodes = self.block_nodes(info.block);
            let neighbor_space = self.block_space(info.neighbor);
            let neighbor_nodes = self.block_nodes(info.neighbor);

            let mut block_system = unsafe { shared.slice_mut(block_nodes) };
            let neighbor_system = unsafe { shared.slice(neighbor_nodes) };

            for node in self.interface_nodes_active(interface, extent) {
                let block_node = array::from_fn(|axis| node[axis] as isize + info.dest[axis]);
                let neighbor_node =
                    array::from_fn(|axis| 2 * (node[axis] as isize + info.source[axis]));

                let block_index = block_space.index_from_node(block_node);
                let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                for field in system.enumerate() {
                    let value = neighbor_system.field(field)[neighbor_index];
                    block_system.field_mut(field)[block_index] = value;
                }
            }
        });
    }

    fn fill_prolong<K: Kernels, B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        _order: K,
        extent: usize,
        boundary: &B,
        conditions: &C,
        result: SystemSliceMut<C::System>,
    ) where
        C::System: Clone,
    {
        let system = result.system().clone();
        let shared = result.into_shared();

        self.neighbors.coarse_indices().for_each(|interface| {
            let info = self.interface(interface);

            let block_nodes = self.block_nodes(info.block);
            let block_space = self.block_space(info.block);
            let neighbor_nodes = self.block_nodes(info.neighbor);
            let neighbor_boundary = self.block_boundary(info.neighbor, boundary.clone());
            let neighbor_space = self.block_space(info.neighbor);

            let mut block_system = unsafe { shared.slice_mut(block_nodes) };
            let neighbor_system = unsafe { shared.slice(neighbor_nodes) };

            for node in self.interface_nodes_active(interface, extent) {
                let block_node = array::from_fn(|axis| node[axis] as isize + info.dest[axis]);

                let neighbor_vertex =
                    array::from_fn(|axis| (node[axis] as isize + info.source[axis]) as usize);

                let block_index = block_space.index_from_node(block_node);

                for field in system.enumerate() {
                    let value = neighbor_space.prolong(
                        SystemBC::new(neighbor_boundary.clone(), conditions.clone(), field),
                        K::interpolation().clone(),
                        neighbor_vertex,
                        neighbor_system.field(field),
                    );
                    block_system.field_mut(field)[block_index] = value;
                }
            }
        });
    }

    /// Enforce weak boundary conditions on the time derivative of a system.
    pub fn weak_boundary<O: Kernels + Sync, B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        order: O,
        boundary: B,
        conditions: C,
        source: SystemSlice<'_, C::System>,
        deriv: SystemSliceMut<'_, C::System>,
    ) where
        C::System: Clone + Sync,
    {
        let system = deriv.system().clone();
        let deriv = deriv.into_shared();

        self.block_compute(|mesh, store, block| {
            let boundary = mesh.block_boundary(block, boundary.clone());
            let bounds = mesh.blocks.bounds(block);
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);
            let vertex_size = space.inner_size();

            let block_source = source.slice(nodes.clone());
            let mut block_deriv = unsafe { deriv.slice_mut(nodes.clone()) };

            let engine = FdEngine {
                space: space.clone(),
                bounds,
                order,
                boundary: boundary.clone(),
                store,
                range: nodes.clone(),
            };
            let spacing = engine.min_spacing();

            for face in faces::<N>() {
                if boundary.kind(face) != BoundaryKind::Radiative {
                    continue;
                }

                // Sommerfeld radiative boundary conditions.
                for vertex in IndexSpace::new(vertex_size).face(face).iter() {
                    // *************************
                    // At vertex

                    let position: [f64; N] = engine.position(vertex);
                    let r = position.iter().map(|&v| v * v).sum::<f64>().sqrt();
                    let index = engine.index_from_vertex(vertex);

                    // *************************
                    // Inner

                    let mut inner = vertex;

                    // Find innter vertex for approximating higher order r dependence
                    for axis in 0..N {
                        if boundary.kind(Face::negative(axis)) == BoundaryKind::Radiative
                            && vertex[axis] == 0
                        {
                            inner[axis] += 1;
                        }

                        if boundary.kind(Face::positive(axis)) == BoundaryKind::Radiative
                            && vertex[axis] == vertex_size[axis] - 1
                        {
                            inner[axis] -= 1;
                        }
                    }

                    let inner_position = engine.position(inner);
                    let inner_r = inner_position.iter().map(|&v| v * v).sum::<f64>().sqrt();
                    let inner_index = engine.index_from_vertex(inner);

                    for field in system.enumerate() {
                        let source = block_source.field(field);
                        let deriv = block_deriv.field_mut(field);

                        let params = conditions.radiative(field, position, spacing);

                        // Inner R dependence
                        let mut inner_advection = engine.value(source, inner) - params.target;

                        for axis in 0..N {
                            let derivative = engine.derivative(source, axis, inner);
                            inner_advection += inner_position[axis] * derivative;
                        }

                        inner_advection *= params.speed;

                        let k = inner_r
                            * inner_r
                            * inner_r
                            * (deriv[inner_index] + inner_advection / inner_r);

                        // Vertex
                        let mut advection = engine.value(source, vertex) - params.target;

                        for axis in 0..N {
                            let derivative = engine.derivative(source, axis, vertex);
                            advection += position[axis] * derivative;
                        }

                        advection *= params.speed;
                        deriv[index] = -advection / r + k / (r * r * r);
                    }
                }
            }
        })
    }

    /// Stores the index of the block which owns each individual node in a debug vector.
    pub fn block_debug(&mut self, debug: &mut [i64]) {
        assert!(debug.len() == self.num_nodes());

        let debug = SharedSlice::new(debug);

        self.block_compute(|mesh, _, block| {
            let block_nodes = mesh.block_nodes(block);

            for node in block_nodes {
                unsafe {
                    *debug.get_mut(node) = block as i64;
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
            let cells = mesh.blocks.cells(block);

            for (i, position) in IndexSpace::new(block_size).iter().enumerate() {
                let cell = cells[i];
                let origin: [_; N] = array::from_fn(|axis| (position[axis] * mesh.width) as isize);

                for offset in IndexSpace::new([mesh.width + 1; N]).iter() {
                    let node = array::from_fn(|axis| origin[axis] + offset[axis] as isize);

                    let idx = block_nodes.start + block_space.index_from_node(node);

                    unsafe {
                        *debug.get_mut(idx) = cell as i64;
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

                for offset in self.interface_nodes_active(iidx, extent) {
                    let node = array::from_fn(|axis| interface.dest[axis] + offset[axis] as isize);
                    let index = block_space.index_from_node(node);

                    unsafe {
                        *debug.get_mut(block_nodes.start + index) = interface.neighbor as i64;
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

                for offset in self.interface_nodes_active(iidx, extent) {
                    let node = array::from_fn(|axis| interface.dest[axis] + offset[axis] as isize);
                    let index = block_space.index_from_node(node);

                    unsafe {
                        *debug.get_mut(block_nodes.start + index) = iidx as i64;
                    }
                }
            });
    }

    pub(super) fn build_interfaces(&mut self) {
        self.interfaces.clear();
        self.interface_node_offsets.clear();
        self.interface_masks.clear();

        for neighbor in self.neighbors.iter() {
            let aabb = self.transfer_aabb(neighbor);

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

        // Retrieve index masks
        let mut interface_masks = std::mem::take(&mut self.interface_masks);
        interface_masks.resize(cursor, false);

        // Construct shared slice
        let masks = SharedSlice::new(&mut interface_masks);

        self.block_compute(|mesh, store, block| {
            let block_space = mesh.block_space(block);
            let block_level = mesh.block_level(block);

            let buffer = store.scratch(block_space.num_nodes());
            buffer.fill(usize::MAX);

            // Fill buffer
            for iidx in mesh.neighbors.block_range(block) {
                let interface = &mesh.interfaces[iidx];
                let interface_level = mesh.block_level(interface.neighbor);

                debug_assert!(interface.block == block);

                for offset in IndexSpace::new(interface.size).iter() {
                    let node = array::from_fn(|axis| interface.dest[axis] + offset[axis] as isize);
                    let index = block_space.index_from_node(node);

                    if buffer[index] == usize::MAX {
                        buffer[index] = iidx;
                        continue;
                    }

                    let other_interface = &mesh.interfaces[buffer[index]];
                    let other_neighbor = other_interface.neighbor;

                    debug_assert!(other_interface.block == interface.block);

                    let other_level = mesh.block_level(other_neighbor);

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
            for iidx in mesh.neighbors.block_range(block) {
                let interface = &mesh.interfaces[iidx];
                let interface_level = mesh.block_level(interface.neighbor);

                debug_assert!(interface.block == block);

                let interface_mask_offset = mesh.interface_node_offsets[iidx];

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
                    unsafe {
                        *masks.get_mut(interface_mask_offset + dst) = mask;
                    }
                }
            }
        });

        // Now replace interface masks in mesh.
        let _ = std::mem::replace(&mut self.interface_masks, interface_masks);
    }

    /// Computes transfer aabb for nodes that lie outside the node space.
    fn transfer_aabb(&self, interface: &TreeBlockNeighbor<N>) -> TransferAABB<N> {
        let a = interface.a.clone();
        let b = interface.b.clone();

        let block_level = self.tree.level(a.cell);
        let neighbor_level = self.tree.level(a.neighbor);

        // ********************************
        // Dest

        let (mut anode, mut bnode) = self.transfer_dest(interface);

        // **************************
        // Source

        let mut source = self.transfer_source(&interface.a);

        // // Ensure regions are disjoint
        // for axis in 0..N {
        //     if b.region.side(axis) == Side::Middle
        //         && bnode[axis] < (block_size[axis] * self.width) as isize
        //     {
        //         bnode[axis] -= 1;
        //     }
        // }

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
    fn transfer_dest(&self, interface: &TreeBlockNeighbor<N>) -> ([isize; N], [isize; N]) {
        let a = interface.a.clone();
        let b = interface.b.clone();

        // Find ghost region that must be filled
        let aindex = self.blocks.cell_position(a.cell);
        let bindex = self.blocks.cell_position(b.cell);

        let block_level = self.tree.level(a.cell);
        let neighbor_level = self.tree.level(a.neighbor);

        // ********************************
        // A node

        // Compute bottom left corner of A cell.
        let mut anode: [_; N] = array::from_fn(|axis| (aindex[axis] * self.width) as isize);

        if block_level < neighbor_level {
            let split = self.tree.split(a.neighbor);
            (0..N)
                .filter(|&axis| a.region.side(axis) == Side::Middle && split.is_set(axis))
                .for_each(|axis| anode[axis] += (self.width / 2) as isize);
        }

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match a.region.side(axis) {
                Side::Left => {
                    anode[axis] -= self.ghost as isize;
                }
                Side::Right => {
                    anode[axis] += self.width as isize;
                }
                Side::Middle => {}
            }
        }

        // ***********************************
        // B Node

        // Compute top right corner of B cell
        let mut bnode: [_; N] = array::from_fn(|axis| ((bindex[axis] + 1) * self.width) as isize);

        if block_level < neighbor_level {
            let split = self.tree.split(b.neighbor);
            (0..N)
                .filter(|&axis| b.region.side(axis) == Side::Middle && !split.is_set(axis))
                .for_each(|axis| bnode[axis] -= (self.width / 2) as isize);
        }

        // Offset by appropriate ghost nodes/width
        for axis in 0..N {
            match b.region.side(axis) {
                Side::Right => {
                    bnode[axis] += self.ghost as isize;
                }
                Side::Left => {
                    bnode[axis] -= self.width as isize;
                }
                Side::Middle => {}
            }
        }

        (anode, bnode)
    }

    /// Computes the source node on the neighboring block from which we fill the current block.
    #[inline]
    fn transfer_source(&self, a: &TreeCellNeighbor<N>) -> [isize; N] {
        let block_level = self.tree.level(a.cell);
        let neighbor_level = self.tree.level(a.neighbor);

        // Find source node
        let nindex = self.blocks.cell_position(a.neighbor);
        let mut source: [isize; N] = array::from_fn(|axis| (nindex[axis] * self.width) as isize);

        match block_level.cmp(&neighbor_level) {
            Ordering::Equal => {
                for axis in 0..N {
                    if a.region.side(axis) == Side::Left {
                        source[axis] += (self.width - self.ghost) as isize;
                    }
                }
            }
            Ordering::Greater => {
                // Source is stored in subnodes
                for axis in 0..N {
                    source[axis] *= 2;
                }

                let split = self.tree.split(a.cell);

                for axis in 0..N {
                    if split.is_set(axis) {
                        match a.region.side(axis) {
                            Side::Left => source[axis] += self.width as isize - self.ghost as isize,
                            Side::Middle => source[axis] += self.width as isize,
                            Side::Right => {}
                        }
                    } else {
                        match a.region.side(axis) {
                            Side::Left => {
                                source[axis] += 2 * self.width as isize - self.ghost as isize
                            }
                            Side::Middle => {}
                            Side::Right => source[axis] += self.width as isize,
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
                        source[axis] += self.width as isize / 2 - self.ghost as isize;
                    }
                }
            }
        }

        source
    }
}
