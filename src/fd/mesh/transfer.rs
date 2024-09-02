use std::array;

use aeon_geometry::{faces, Face, IndexSpace};
use rayon::iter::{ParallelBridge, ParallelIterator};
use reborrow::Reborrow as _;

use crate::{
    fd::{
        kernel::Kernels, Boundary, BoundaryKind, Condition, Conditions, Engine, FdEngine, SystemBC,
    },
    system::{SystemLabel, SystemSlice, SystemSliceMut},
};

use super::Mesh;

impl<const N: usize> Mesh<N> {
    /// Enforces strong boundary conditions. This includes strong physical boundary conditions, as well
    /// as handling interior boundaries (same level, coarse-fine, or fine-coarse).
    pub fn fill_boundary<K: Kernels, B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        order: K,
        boundary: B,
        conditions: C,
        mut system: SystemSliceMut<'_, C::System>,
    ) {
        self.fill_physical(&boundary, &conditions, &mut system);
        self.fill_direct(&mut system);
        self.fill_fine(&mut system);
        self.fill_prolong(order, &boundary, &conditions, &mut system);
    }

    fn fill_physical<B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
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
                space.fill_boundary(bcs, block_system.field_mut(field.clone()));
            }
        });
    }

    fn fill_direct<System: SystemLabel>(&mut self, system: &mut SystemSliceMut<'_, System>) {
        let system = system.as_range();

        // Fill direct interfaces
        self.interfaces.direct().par_bridge().for_each(|interface| {
            let block_space = self.block_space(interface.block);
            let block_nodes = self.block_nodes(interface.block);
            let neighbor_space = self.block_space(interface.neighbor);
            let neighbor_nodes = self.block_nodes(interface.neighbor);

            let mut block_system = unsafe { system.slice_mut(block_nodes).fields_mut() };
            let neighbor_system = unsafe { system.slice(neighbor_nodes).fields() };

            for node in IndexSpace::new(interface.size).iter() {
                let block_node = array::from_fn(|axis| node[axis] as isize + interface.dest[axis]);
                let neighbor_node =
                    array::from_fn(|axis| node[axis] as isize + interface.source[axis]);

                let block_index = block_space.index_from_node(block_node);
                let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                for field in System::fields() {
                    let value = neighbor_system.field(field.clone())[neighbor_index];
                    block_system.field_mut(field.clone())[block_index] = value;
                }
            }
        });
    }

    fn fill_fine<System: SystemLabel>(&mut self, system: &mut SystemSliceMut<'_, System>) {
        let system = system.as_range();

        self.interfaces.fine().par_bridge().for_each(|interface| {
            let block_space = self.block_space(interface.block);
            let block_nodes = self.block_nodes(interface.block);
            let neighbor_space = self.block_space(interface.neighbor);
            let neighbor_nodes = self.block_nodes(interface.neighbor);

            let mut block_system = unsafe { system.slice_mut(block_nodes).fields_mut() };
            let neighbor_system = unsafe { system.slice(neighbor_nodes).fields() };

            for node in IndexSpace::new(interface.size).iter() {
                let block_node = array::from_fn(|axis| node[axis] as isize + interface.dest[axis]);
                let neighbor_node =
                    array::from_fn(|axis| 2 * (node[axis] as isize + interface.source[axis]));

                let block_index = block_space.index_from_node(block_node);
                let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                for field in System::fields() {
                    let value = neighbor_system.field(field.clone())[neighbor_index];
                    block_system.field_mut(field.clone())[block_index] = value;
                }
            }
        });
    }

    fn fill_prolong<K: Kernels, B: Boundary<N> + Sync, C: Conditions<N> + Sync>(
        &mut self,
        _order: K,
        boundary: &B,
        conditions: &C,
        system: &mut SystemSliceMut<'_, C::System>,
    ) {
        let system = system.as_range();
        self.interfaces.coarse().par_bridge().for_each(|interface| {
            let block_nodes = self.block_nodes(interface.block);
            let block_space = self.block_space(interface.block);
            let neighbor_nodes = self.block_nodes(interface.neighbor);
            let neighbor_boundary = self.block_boundary(interface.neighbor, boundary.clone());
            let neighbor_space = self.block_space(interface.neighbor);

            let mut block_system = unsafe { system.slice_mut(block_nodes).fields_mut() };
            let neighbor_system = unsafe { system.slice(neighbor_nodes).fields() };

            for node in IndexSpace::new(interface.size).iter() {
                let block_node = array::from_fn(|axis| node[axis] as isize + interface.dest[axis]);

                let neighbor_vertex =
                    array::from_fn(|axis| (node[axis] as isize + interface.source[axis]) as usize);

                let block_index = block_space.index_from_node(block_node);

                for field in C::System::fields() {
                    let bcs =
                        SystemBC::new(field.clone(), neighbor_boundary.clone(), conditions.clone());

                    let value = neighbor_space.prolong(
                        bcs,
                        K::interpolation().clone(),
                        neighbor_vertex,
                        neighbor_system.field(field.clone()),
                    );
                    block_system.field_mut(field.clone())[block_index] = value;
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

    pub fn transfer_to_coarse<System: SystemLabel>(
        &mut self,
        source: SystemSlice<System>,
        dest: SystemSliceMut<System>,
    ) {
        let source = source.as_range();
        let dest = dest.as_range();

        (0..self.blocks.len()).par_bridge().for_each(|block| {
            let source_dofs = self.block_nodes(block);
            let dest_dofs = self.block_coarse_nodes(block);

            let source = unsafe { source.slice(source_dofs.clone()).fields() };
            let mut dest = unsafe { dest.slice_mut(dest_dofs.clone()).fields_mut() };

            let source_space = self.block_space(block);
            let dest_space = self.block_coarse_space(block);

            for dest_node in dest_space.inner_window() {
                let source_node = array::from_fn(|axis| dest_node[axis] * 2);
                let source_index = source_space.index_from_node(source_node);
                let dest_index = dest_space.index_from_node(dest_node);

                for field in System::fields() {
                    dest.field_mut(field.clone())[dest_index] =
                        source.field(field.clone())[source_index];
                }
            }
        });
    }
}
