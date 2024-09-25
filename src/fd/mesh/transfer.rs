use aeon_basis::{Boundary, BoundaryKind, Condition, Kernels};
use aeon_geometry::{faces, Face, IndexSpace, Region, Side, TreeBlockNeighbor, TreeCellNeighbor};
use rayon::iter::{ParallelBridge, ParallelIterator};
use reborrow::Reborrow as _;
use std::{array, iter};

use crate::{
    fd::{Conditions, Engine, FdEngine, Mesh, SystemBC},
    shared::SharedSlice,
    system::{SystemLabel, SystemSlice, SystemSliceMut},
};

#[derive(Clone, Debug)]
pub(super) struct TreeInterface<const N: usize> {
    /// Block to be filled
    block: usize,
    /// Neighbor block.
    neighbor: usize,
    aregion: Region<N>,
    bregion: Region<N>,
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
    /// Enforces strong boundary conditions. This includes strong physical boundary conditions, as well
    /// as handling interior boundaries (same level, coarse-fine, or fine-coarse).
    pub fn fill_boundary<K: Kernels, B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        order: K,
        boundary: B,
        conditions: C,
        system: SystemSliceMut<'_, C::System>,
    ) {
        self.fill_boundary_to_extent(order, K::MAX_BORDER, boundary, conditions, system);
    }

    pub fn fill_boundary_to_extent<K: Kernels, B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        order: K,
        extent: usize,
        boundary: B,
        conditions: C,
        mut system: SystemSliceMut<'_, C::System>,
    ) {
        self.fill_direct(extent, &mut system);
        self.fill_fine(extent, &mut system);

        self.fill_physical(extent, &boundary, &conditions, &mut system);
        self.fill_prolong(order, extent, &boundary, &conditions, &mut system);
        self.fill_physical(extent, &boundary, &conditions, &mut system);
    }

    fn fill_physical<B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        extent: usize,
        boundary: &B,
        conditions: &C,
        system: &mut SystemSliceMut<'_, C::System>,
    ) {
        let system = system.as_range();

        (0..self.blocks.len()).for_each(|block| {
            // Fill Physical Boundary conditions
            let nodes = self.block_nodes(block);
            let space = self.block_space(block);
            let boundary = self.block_boundary(block, boundary.clone());

            let mut block_system = unsafe { system.slice_mut(nodes) };

            for field in C::System::fields() {
                let bcs = SystemBC::new(field.clone(), boundary.clone(), conditions.clone());
                space.fill_boundary(extent, bcs, block_system.field_mut(field.clone()));
            }
        });
    }

    fn fill_direct<System: SystemLabel>(
        &mut self,
        extent: usize,
        system: &mut SystemSliceMut<'_, System>,
    ) {
        let system = system.as_range();

        // Fill direct neighbors
        self.neighbors.direct().for_each(|interface| {
            let block_space = self.block_space(interface.block);
            let block_nodes = self.block_nodes(interface.block);
            let neighbor_space = self.block_space(interface.neighbor);
            let neighbor_nodes = self.block_nodes(interface.neighbor);

            let mut block_system = unsafe { system.slice_mut(block_nodes).fields_mut() };
            let neighbor_system = unsafe { system.slice(neighbor_nodes).fields() };

            for aabb in self.padding_aabbs(interface, extent) {
                for node in IndexSpace::new(aabb.size).iter() {
                    let block_node = array::from_fn(|axis| node[axis] as isize + aabb.dest[axis]);
                    let neighbor_node =
                        array::from_fn(|axis| node[axis] as isize + aabb.source[axis]);

                    let block_index = block_space.index_from_node(block_node);
                    let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                    for field in System::fields() {
                        let value = neighbor_system.field(field.clone())[neighbor_index];
                        block_system.field_mut(field.clone())[block_index] = value;
                    }
                }
            }
        });
    }

    fn fill_fine<System: SystemLabel>(
        &mut self,
        extent: usize,
        system: &mut SystemSliceMut<'_, System>,
    ) {
        let system = system.as_range();

        self.neighbors.fine().for_each(|interface| {
            let block_space = self.block_space(interface.block);
            let block_nodes = self.block_nodes(interface.block);
            let neighbor_space = self.block_space(interface.neighbor);
            let neighbor_nodes = self.block_nodes(interface.neighbor);

            let mut block_system = unsafe { system.slice_mut(block_nodes).fields_mut() };
            let neighbor_system = unsafe { system.slice(neighbor_nodes).fields() };

            for aabb in self.padding_aabbs(interface, extent) {
                for node in IndexSpace::new(aabb.size).iter() {
                    let block_node = array::from_fn(|axis| node[axis] as isize + aabb.dest[axis]);
                    let neighbor_node =
                        array::from_fn(|axis| 2 * (node[axis] as isize + aabb.source[axis]));

                    let block_index = block_space.index_from_node(block_node);
                    let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                    for field in System::fields() {
                        let value = neighbor_system.field(field.clone())[neighbor_index];
                        block_system.field_mut(field.clone())[block_index] = value;
                    }
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
        system: &mut SystemSliceMut<'_, C::System>,
    ) {
        let system = system.as_range();
        self.neighbors.coarse().for_each(|interface| {
            let block_nodes = self.block_nodes(interface.block);
            let block_space = self.block_space(interface.block);
            let neighbor_nodes = self.block_nodes(interface.neighbor);
            let neighbor_boundary = self.block_boundary(interface.neighbor, boundary.clone());
            let neighbor_space = self.block_space(interface.neighbor);

            let mut block_system = unsafe { system.slice_mut(block_nodes).fields_mut() };
            let neighbor_system = unsafe { system.slice(neighbor_nodes).fields() };

            for aabb in self.padding_aabbs(interface, extent) {
                for node in IndexSpace::new(aabb.size).iter() {
                    let block_node = array::from_fn(|axis| node[axis] as isize + aabb.dest[axis]);

                    let neighbor_vertex =
                        array::from_fn(|axis| (node[axis] as isize + aabb.source[axis]) as usize);

                    let block_index = block_space.index_from_node(block_node);

                    for field in C::System::fields() {
                        let bcs = SystemBC::new(
                            field.clone(),
                            neighbor_boundary.clone(),
                            conditions.clone(),
                        );

                        let value = neighbor_space.prolong(
                            bcs,
                            K::interpolation().clone(),
                            neighbor_vertex,
                            neighbor_system.field(field.clone()),
                        );
                        block_system.field_mut(field.clone())[block_index] = value;
                    }
                }
            }
        });
    }

    pub fn weak_boundary<O: Kernels, B: Boundary<N>, C: Conditions<N>>(
        &mut self,
        order: O,
        boundary: B,
        conditions: C,
        system: SystemSlice<'_, C::System>,
        deriv: SystemSliceMut<'_, C::System>,
    ) {
        let system = system.as_range();
        let deriv = deriv.as_range();

        (0..self.blocks.len()).for_each(|block| {
            let boundary = self.block_boundary(block, boundary.clone());
            let bounds = self.blocks.bounds(block);
            let space = self.block_space(block);
            let nodes = self.block_nodes(block);
            let vertex_size = space.inner_size();

            let block_system = unsafe { system.slice(nodes.clone()).fields() };
            let mut block_deriv = unsafe { deriv.slice_mut(nodes.clone()).fields_mut() };

            for face in faces::<N>() {
                if boundary.kind(face) != BoundaryKind::Radiative {
                    continue;
                }

                // Sommerfeld radiative boundary conditions.
                for vertex in IndexSpace::new(vertex_size).face(face).iter() {
                    // *************************
                    // At vertex

                    let engine = FdEngine {
                        space: space.clone(),
                        vertex,
                        bounds,
                        fields: block_system.rb(),
                        order,
                        boundary: boundary.clone(),
                        conditions: conditions.clone(),
                    };

                    let position: [f64; N] = engine.position();
                    let r = position.iter().map(|&v| v * v).sum::<f64>().sqrt();
                    let index = space.index_from_vertex(vertex);

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

                    let inner_engine = FdEngine {
                        space: space.clone(),
                        vertex: inner,
                        bounds,
                        fields: block_system.rb(),
                        order,
                        boundary: boundary.clone(),
                        conditions: conditions.clone(),
                    };

                    let inner_position = inner_engine.position();
                    let inner_r = inner_position.iter().map(|&v| v * v).sum::<f64>().sqrt();
                    let inner_index = space.index_from_vertex(inner);

                    for field in C::System::fields() {
                        let field_derivs = block_deriv.field_mut(field.clone());
                        let bcs =
                            SystemBC::new(field.clone(), boundary.clone(), conditions.clone());

                        let target = Condition::radiative(&bcs, position);

                        // Inner R dependence
                        let mut inner_advection = inner_engine.value(field.clone()) - target;

                        for axis in 0..N {
                            let derivative = inner_engine.derivative(field.clone(), axis);
                            inner_advection += inner_position[axis] * derivative;
                        }

                        let k = inner_r
                            * inner_r
                            * inner_r
                            * (field_derivs[inner_index] + inner_advection / inner_r);

                        // Vertex
                        let mut advection = engine.value(field.clone()) - target;

                        for axis in 0..N {
                            let derivative = engine.derivative(field.clone(), axis);
                            advection += position[axis] * derivative;
                        }

                        field_derivs[index] = -advection / r + k / (r * r * r);
                        // field_derivs[index] = -advection / r;
                    }
                }
            }
        });
    }

    pub fn interface_indices(&mut self, extent: usize, info: &mut [i64]) {
        assert!(info.len() == self.num_nodes());
        assert!(extent <= self.ghost);

        info.fill(0);

        let info = SharedSlice::new(info);

        self.neighbors
            .iter()
            .enumerate()
            .par_bridge()
            .for_each(|(i, interface)| {
                let block_nodes = self.block_nodes(interface.block);
                let block_space = self.block_space(interface.block);

                for aabb in self.padding_aabbs(interface, extent) {
                    for node in IndexSpace::new(aabb.size).iter() {
                        let block_node =
                            array::from_fn(|axis| node[axis] as isize + aabb.dest[axis]);

                        let block_index = block_space.index_from_node(block_node);

                        unsafe {
                            *info.get_mut(block_nodes.start + block_index) = i as i64;
                        }
                    }
                }
            });
    }

    pub fn block_interface_indices(&mut self, extent: usize, info: &mut [i64]) {
        assert!(info.len() == self.num_nodes());
        assert!(extent <= self.ghost);

        info.fill(0);

        let info = SharedSlice::new(info);

        self.neighbors
            .iter()
            .enumerate()
            .par_bridge()
            .for_each(|(i, interface)| {
                let block_nodes = self.block_nodes(interface.block);
                let block_space = self.block_space(interface.block);

                for aabb in self.padding_aabbs(interface, extent) {
                    for node in IndexSpace::new(aabb.size).iter() {
                        let block_node =
                            array::from_fn(|axis| node[axis] as isize + aabb.dest[axis]);

                        let block_index = block_space.index_from_node(block_node);

                        unsafe {
                            *info.get_mut(block_nodes.start + block_index) =
                                interface.neighbor as i64;
                        }
                    }
                }
            });
    }

    pub fn block_debug(&mut self, debug: &mut [i64]) {
        assert!(debug.len() == self.num_nodes());

        let debug = SharedSlice::new(debug);

        self.block_compute(|mesh, store, block| {
            let block_nodes = mesh.block_nodes(block);

            for node in block_nodes {
                unsafe {
                    *debug.get_mut(node) = block as i64;
                }
            }
        });
    }

    pub(super) fn build_interfaces(&mut self) {
        self.interfaces.clear();
        self.interface_node_offsets.clear();
        self.interface_masks.clear();

        for neighbor in self.neighbors.iter() {
            let aabb = self.region_aabb(neighbor);

            self.interfaces.push(TreeInterface {
                block: neighbor.block,
                neighbor: neighbor.neighbor,
                aregion: neighbor.a.region.clone(),
                bregion: neighbor.b.region.clone(),
                source: aabb.source,
                dest: aabb.dest,
                size: aabb.size,
            });
        }

        // Compute node offsets

        let mut cursor = 0;

        for interface in self.interfaces.iter() {
            self.interface_node_offsets.push(cursor);

            cursor += IndexSpace::new(interface.size).index_count();
        }

        self.interface_node_offsets.push(cursor);

        // Compute interface masks

        // Retrieve index masks
        let mut interface_masks = std::mem::take(&mut self.interface_masks);
        self.interface_masks.resize(cursor, false);

        // Construct shared slice
        let masks = SharedSlice::new(&mut interface_masks);

        self.block_compute(|mesh, store, block| {});

        let _ = std::mem::replace(&mut self.interface_masks, interface_masks);
    }

    /// Computes transfer aabb for nodes that lie outside the node space.
    fn region_aabb(&self, interface: &TreeBlockNeighbor<N>) -> TransferAABB<N> {
        let a = interface.a.clone();
        let b = interface.b.clone();

        // Find ghost region that must be filled
        let aindex = self.blocks.cell_position(a.cell);
        let bindex = self.blocks.cell_position(b.cell);

        let block_level = self.tree.level(a.cell);
        let neighbor_level = self.tree.level(a.neighbor);

        let block_size = self.blocks.size(interface.block);

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

        // **************************
        // Origin

        let source = self.aabb_source(&a, self.ghost);

        // // Ensure regions are disjoint
        // for axis in 0..N {
        //     if b.region.side(axis) == Side::Middle
        //         && bnode[axis] < (block_size[axis] * self.width) as isize
        //     {
        //         bnode[axis] -= 1;
        //     }
        // }

        // for axis in 0..N {
        //     if b.region.side(axis) == Side::Left {
        //         bnode[axis] -= 1;
        //     }

        //     if a.region.side(axis) == Side::Right {
        //         anode[axis] += 1;
        //         source[axis] += 1;
        //     }
        // }

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

    // /// Computes transfer aabb for nodes on a face of the node space.
    // fn face_aabb(&self, interface: &TreeBlockNeighbor<N>) -> Option<TransferAABB<N>> {
    //     let face = interface.face()?;

    //     let a = interface.a.clone();
    //     let b = interface.b.clone();

    //     let aindex = self.blocks.cell_position(a.cell);
    //     let bindex = self.blocks.cell_position(b.cell);

    //     let block_size = self.blocks.size(interface.block);

    //     let block_level = self.tree.level(a.cell);
    //     let neighbor_level = self.tree.level(a.neighbor);

    //     let block_space = self.block_space(interface.block);

    //     // If coarser, we should not copy nodes from fine
    //     if block_level < neighbor_level {
    //         return None;
    //     }

    //     // If same level, only transfer if neighbor has
    //     if block_level == neighbor_level && interface.block < interface.neighbor {
    //         return None;
    //     }

    //     // Destination AABB
    //     let mut origin: [_; N] = array::from_fn(|axis| (aindex[axis] * self.width) as isize);
    //     if face.side {
    //         origin[face.axis] += self.width as isize;
    //     }

    //     let mut size: [usize; N] =
    //         array::from_fn(|axis| (bindex[axis] - aindex[axis] + 1) * self.width + 1);
    //     size[face.axis] = 1;

    //     // Offset if the neighbor is more fine.
    //     if block_level < neighbor_level {
    //         let split = self.tree.split(a.neighbor);
    //         (0..N)
    //             .filter(|&axis| a.region.side(axis) == Side::Middle && split.is_set(axis))
    //             .for_each(|axis| {
    //                 origin[axis] += (self.width / 2) as isize;
    //                 size[axis] -= self.width / 2;
    //             });

    //         let split = self.tree.split(b.neighbor);
    //         (0..N)
    //             .filter(|&axis| b.region.side(axis) == Side::Middle && !split.is_set(axis))
    //             .for_each(|axis| size[axis] -= self.width / 2);
    //     }

    //     let mut source = self.aabb_source(&a, 0);

    //     // Ensure that all aabbs on a given face don't overlap
    //     for axis in 0..N {
    //         if b.region.side(axis) == Side::Middle
    //             && (origin[axis] + size[axis] as isize)
    //                 < (block_size[axis] * self.width + 1) as isize
    //         {
    //             size[axis] -= 1;
    //         }
    //     }

    //     // // Intersect with disjoint face window.
    //     // let window = block_space.face_window_disjoint(face);

    //     // for axis in 0..N {
    //     //     if origin[axis] < window.origin[axis] {
    //     //         let diff = window.origin[axis] - origin[axis];

    //     //         debug_assert!(diff == 1);

    //     //         origin[axis] += diff;
    //     //         source[axis] += diff;
    //     //         size[axis] -= diff as usize;
    //     //     }

    //     //     let corner = origin[axis] + size[axis] as isize;
    //     //     let window_corner = window.origin[axis] + window.size[axis] as isize;

    //     //     if corner > window_corner {
    //     //         let diff = corner - window_corner;

    //     //         debug_assert!(diff == 1);

    //     //         size[axis] -= diff as usize;
    //     //     }
    //     // }

    //     Some(TransferAABB {
    //         source,
    //         dest: origin,
    //         size,
    //     })
    // }

    /// Computes the source node on the neighboring block from which we fill the current block.
    #[inline]
    fn aabb_source(&self, a: &TreeCellNeighbor<N>, extent: usize) -> [isize; N] {
        let block_level = self.tree.level(a.cell);
        let neighbor_level = self.tree.level(a.neighbor);

        // Find source node
        let nindex = self.blocks.cell_position(a.neighbor);
        let mut source: [isize; N] = array::from_fn(|axis| (nindex[axis] * self.width) as isize);

        if block_level == neighbor_level {
            for axis in 0..N {
                if a.region.side(axis) == Side::Left {
                    source[axis] += (self.width - extent) as isize;
                }
            }
        } else if block_level > neighbor_level {
            // Source is stored in subnodes
            for axis in 0..N {
                source[axis] *= 2;
            }

            let split = self.tree.split(a.cell);

            for axis in 0..N {
                if split.is_set(axis) {
                    match a.region.side(axis) {
                        Side::Left => source[axis] += self.width as isize - extent as isize,
                        Side::Middle => source[axis] += self.width as isize,
                        Side::Right => {}
                    }
                } else {
                    match a.region.side(axis) {
                        Side::Left => source[axis] += 2 * self.width as isize - extent as isize,
                        Side::Middle => {}
                        Side::Right => source[axis] += self.width as isize,
                    }
                }
            }
        } else if block_level < neighbor_level {
            // Source is stored in supernodes
            for axis in 0..N {
                source[axis] /= 2;
            }

            for axis in 0..N {
                if a.region.side(axis) == Side::Left {
                    source[axis] += self.width as isize / 2 - extent as isize;
                }
            }
        }

        source
    }
}
