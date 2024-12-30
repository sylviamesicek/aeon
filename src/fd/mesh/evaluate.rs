use std::array;

use aeon_basis::{Boundary, BoundaryKind, Kernels};
use aeon_geometry::{faces, Face, FaceMask, IndexSpace, NULL};
// use rayon::iter::{ParallelBridge, ParallelIterator};
use reborrow::{Reborrow, ReborrowMut as _};

use crate::{
    fd::{Conditions, Engine, FdEngine, FdIntEngine, Function, Projection, ProjectionAsFunction},
    shared::SharedSlice,
    system::{Scalar, System, SystemSlice, SystemSliceMut},
};

use super::Mesh;

impl<const N: usize> Mesh<N> {
    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn evaluate<K: Kernels + Sync, B: Boundary<N> + Sync, P: Function<N> + Sync>(
        &mut self,
        order: K,
        boundary: B,
        function: P,
        source: SystemSlice<'_, P::Input>,
        dest: SystemSliceMut<'_, P::Output>,
    ) where
        P::Input: Sync,
        P::Output: Sync,
    {
        let dest = dest.into_shared();

        self.block_compute(|mesh, store, block| {
            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            let boundary = mesh.block_boundary(block, boundary.clone());

            let block_source = source.slice(nodes.clone());
            let block_dest = unsafe { dest.slice_mut(nodes.clone()) };

            if Self::is_interior(&boundary) {
                let engine = FdIntEngine {
                    space: space.clone(),
                    bounds,
                    order,
                    store,
                    range: nodes.clone(),
                };

                function.evaluate(engine, block_source, block_dest);
            } else {
                let engine = FdEngine {
                    space: space.clone(),
                    bounds,
                    order,
                    store,
                    boundary,
                    range: nodes.clone(),
                };

                function.evaluate(engine, block_source, block_dest);
            }
        });
    }

    pub(crate) fn is_interior(boundary: &impl Boundary<N>) -> bool {
        let mut result = true;

        for axis in 0..N {
            result &= boundary.kind(Face::negative(axis)).has_ghost();
            result &= boundary.kind(Face::positive(axis)).has_ghost();
        }

        result
    }

    /// Evaluates the given function on a system in place.
    fn evaluate_mut<
        S: System + Sync,
        K: Kernels + Sync,
        B: Boundary<N> + Sync,
        P: Function<N, Input = S, Output = S> + Sync,
    >(
        &mut self,
        order: K,
        boundary: B,
        function: P,
        dest: SystemSliceMut<'_, S>,
    ) {
        let dest = dest.into_shared();

        self.block_compute(|mesh, store, block| {
            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            let boundary = mesh.block_boundary(block, boundary.clone());

            let block_dest = unsafe { dest.slice_mut(nodes.clone()) };

            let num_nodes = block_dest.len() * dest.system().count();
            let mut block_source =
                SystemSliceMut::from_contiguous(store.scratch(num_nodes), dest.system());

            for field in dest.system().enumerate() {
                block_source
                    .field_mut(field)
                    .copy_from_slice(block_dest.field(field));
            }

            if Self::is_interior(&boundary) {
                let engine = FdIntEngine {
                    space: space.clone(),
                    bounds,
                    order,
                    store,
                    range: nodes.clone(),
                };

                function.evaluate(engine, block_source.rb(), block_dest);
            } else {
                let engine = FdEngine {
                    space: space.clone(),
                    bounds,
                    order,
                    store,
                    boundary,
                    range: nodes.clone(),
                };

                function.evaluate(engine, block_source.rb(), block_dest);
            }
        });
    }

    pub fn apply<
        O: Kernels + Sync,
        B: Boundary<N> + Sync,
        C: Conditions<N> + Sync,
        P: Function<N, Input = C::System, Output = C::System> + Sync,
    >(
        &mut self,
        order: O,
        boundary: B,
        conditions: C,
        op: P,
        f: SystemSliceMut<'_, C::System>,
    ) where
        C::System: Sync,
    {
        let f = f.into_shared();

        self.block_compute(|mesh, store, block| {
            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            let boundary = mesh.block_boundary(block, boundary.clone());

            let mut block_dest = unsafe { f.slice_mut(nodes.clone()) };

            let num_nodes = block_dest.len() * f.system().count();
            let mut block_source =
                SystemSliceMut::from_contiguous(store.scratch(num_nodes), f.system());

            for field in f.system().enumerate() {
                block_source
                    .field_mut(field)
                    .copy_from_slice(block_dest.field(field));
            }

            if Self::is_interior(&boundary) {
                let engine = FdIntEngine {
                    space: space.clone(),
                    bounds,
                    order,
                    store,
                    range: nodes.clone(),
                };

                op.evaluate(engine, block_source.rb(), block_dest.rb_mut());
            } else {
                let engine = FdEngine {
                    space: space.clone(),
                    bounds,
                    order,
                    store,
                    boundary: boundary.clone(),
                    range: nodes.clone(),
                };

                op.evaluate(&engine, block_source.rb(), block_dest.rb_mut());

                let spacing = mesh.block_spacing(block);

                // Weak boundary conditions.
                for face in faces::<N>() {
                    if boundary.kind(face) != BoundaryKind::Radiative {
                        continue;
                    }

                    // Sommerfeld radiative boundary conditions.
                    for vertex in IndexSpace::new(engine.vertex_size()).face(face).iter() {
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
                                && vertex[axis] == engine.vertex_size()[axis] - 1
                            {
                                inner[axis] -= 1;
                            }
                        }

                        let inner_position = engine.position(inner);
                        let inner_r = inner_position.iter().map(|&v| v * v).sum::<f64>().sqrt();
                        let inner_index = engine.index_from_vertex(inner);

                        for field in f.system().enumerate() {
                            let source = block_source.field(field);
                            let dest = block_dest.field_mut(field);
                            // Get condition parameters.
                            let params = conditions.radiative(field, position, spacing);
                            // Inner R dependence.
                            let mut inner_advection = source[inner_index] - params.target;

                            for axis in 0..N {
                                let derivative = engine.derivative(source, axis, inner);
                                inner_advection += inner_position[axis] * derivative;
                            }

                            inner_advection *= params.speed;

                            let k = inner_r
                                * inner_r
                                * inner_r
                                * (dest[inner_index] + inner_advection / inner_r);

                            // Vertex
                            let mut advection = source[index] - params.target;

                            for axis in 0..N {
                                let derivative = engine.derivative(source, axis, vertex);
                                advection += position[axis] * derivative;
                            }

                            advection *= params.speed;
                            dest[index] = -advection / r + k / (r * r * r);
                        }
                    }
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
        C::System: Sync,
    {
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

                    for field in deriv.system().enumerate() {
                        let source = block_source.field(field);
                        let deriv = block_deriv.field_mut(field);

                        let params = conditions.radiative(field, position, spacing);

                        // Inner R dependence
                        let mut inner_advection = source[inner_index] - params.target;

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
                        let mut advection = source[index] - params.target;

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

    /// Copies an immutable src slice into a mutable dest slice.
    pub fn copy_from_slice<S: System>(&mut self, mut dest: SystemSliceMut<S>, src: SystemSlice<S>) {
        for label in dest.system().enumerate() {
            dest.field_mut(label).copy_from_slice(src.field(label));
        }
    }

    pub fn project<K: Kernels + Sync, B: Boundary<N> + Sync, P: Projection<N> + Sync>(
        &mut self,
        order: K,
        boundary: B,
        projection: P,
        dest: &mut [f64],
    ) {
        self.evaluate(
            order,
            boundary,
            ProjectionAsFunction(projection),
            SystemSlice::empty(),
            SystemSliceMut::from_scalar(dest),
        );
    }

    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn dissipation<K: Kernels + Sync, B: Boundary<N> + Sync, S: System>(
        &mut self,
        order: K,
        boundary: B,
        amplitude: f64,
        mut dest: SystemSliceMut<S>,
    ) {
        #[derive(Clone)]
        struct Dissipation(f64);

        impl<const N: usize> Function<N> for Dissipation {
            type Input = Scalar;
            type Output = Scalar;

            fn evaluate(
                &self,
                engine: impl Engine<N>,
                input: SystemSlice<Self::Input>,
                mut output: SystemSliceMut<Self::Output>,
            ) {
                let input = input.field(());
                let output = output.field_mut(());

                for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                    let index = engine.index_from_vertex(vertex);

                    for axis in 0..N {
                        output[index] += self.0 * engine.dissipation(input, axis, vertex);
                    }
                }
            }
        }

        for field in dest.system().enumerate() {
            self.evaluate_mut(
                order,
                boundary.clone(),
                Dissipation(amplitude),
                SystemSliceMut::from_scalar(dest.field_mut(field)),
            );
        }
    }

    /// This function computes the distance between each vertex and its nearest
    /// neighbor.
    pub fn spacing_per_vertex(&mut self, dest: &mut [f64]) {
        assert!(dest.len() == self.num_nodes());

        let dest = SharedSlice::new(dest);

        self.block_compute(|mesh, _, block| {
            let nodes = mesh.block_nodes(block);
            let space = mesh.block_space(block);
            let bounds = mesh.block_bounds(block);

            let spacing = space.spacing(bounds);
            let min_spacing = spacing
                .iter()
                .min_by(|a, b| a.total_cmp(&b))
                .cloned()
                .unwrap_or(1.0);

            let vertex_size = space.inner_size();

            let block_dest = unsafe { dest.slice_mut(nodes) };

            for &cell in mesh.blocks.cells(block) {
                let cell_level = mesh.tree.level(cell);
                let node_size = mesh.cell_node_size(cell);
                let node_origin = mesh.cell_node_offset(cell);

                let mut flags = FaceMask::empty();

                for face in faces() {
                    let neighbor = mesh.tree.neighbor(cell, face);

                    if neighbor == NULL {
                        continue;
                    }

                    let neighbor_level = mesh.tree.level(neighbor);

                    if neighbor_level > cell_level {
                        flags.set(face);
                    }
                }

                for offset in IndexSpace::new(node_size).iter() {
                    let vertex: [_; N] = array::from_fn(|axis| node_origin[axis] + offset[axis]);
                    let index = space.index_from_vertex(vertex);

                    let mut refined = false;

                    for axis in 0..N {
                        refined |= vertex[axis] == 0 && flags.is_set(Face::negative(axis));
                        refined |= vertex[axis] == vertex_size[axis] - 1
                            && flags.is_set(Face::positive(axis));
                    }

                    if refined {
                        block_dest[index] = min_spacing / 2.0;
                    } else {
                        block_dest[index] = min_spacing;
                    }
                }
            }
        });
    }

    pub fn adaptive_cfl<S: System + Sync>(
        &mut self,
        spacing_per_vertex: &[f64],
        dest: SystemSliceMut<S>,
    ) {
        let dest = dest.into_shared();
        let min_spacing = self.min_spacing();

        self.block_compute(|mesh, _, block| {
            let block_space = mesh.block_space(block);
            let block_nodes = mesh.block_nodes(block);

            let block_spacings = &spacing_per_vertex[block_nodes.clone()];
            let mut block_dest = unsafe { dest.slice_mut(block_nodes.clone()) };

            for field in block_dest.system().enumerate() {
                let block_dest = block_dest.field_mut(field);

                for vertex in IndexSpace::new(block_space.inner_size()).iter() {
                    let index = block_space.index_from_vertex(vertex);
                    block_dest[index] *= block_spacings[index] / min_spacing;
                }
            }
        });
    }

    // /// Performs a fused-multiply-add operation `a + mb` and then assigns this value to `dest`.
    // pub fn fma<S: System + Sync>(
    //     &mut self,
    //     a: SystemSlice<'_, S>,
    //     m: f64,
    //     b: SystemSlice<'_, S>,
    //     dest: SystemSliceMut<'_, S>,
    // ) {
    //     assert!(a.len() == self.num_nodes());
    //     assert!(b.len() == self.num_nodes());
    //     assert!(dest.len() == self.num_nodes());

    //     let dest = dest.into_shared();

    //     (0..self.blocks.len()).par_bridge().for_each(|block| {
    //         let nodes = self.block_nodes(block);

    //         let a = a.slice(nodes.clone());
    //         let b = b.slice(nodes.clone());
    //         let mut dest = unsafe { dest.slice_mut(nodes.clone()) };

    //         for field in a.system().enumerate() {
    //             for i in 0..a.len() {
    //                 dest.field_mut(field.clone())[i] =
    //                     a.field(field.clone())[i] + m * b.field(field.clone())[i];
    //             }
    //         }
    //     });
    // }

    // /// Performs a fused-multiply-add operation `a + mb`, and then adds and assigns this value to `dest`.
    // pub fn add_assign_fma<S: System + Sync>(
    //     &mut self,
    //     a: SystemSlice<'_, S>,
    //     m: f64,
    //     b: SystemSlice<'_, S>,
    //     dest: SystemSliceMut<'_, S>,
    // ) {
    //     assert!(a.len() == self.num_nodes());
    //     assert!(b.len() == self.num_nodes());
    //     assert!(dest.len() == self.num_nodes());

    //     let dest = dest.into_shared();

    //     (0..self.blocks.len()).par_bridge().for_each(|block| {
    //         let nodes = self.block_nodes(block);

    //         let a = a.slice(nodes.clone());
    //         let b = b.slice(nodes.clone());
    //         let mut dest = unsafe { dest.slice_mut(nodes.clone()) };

    //         for field in a.system().enumerate() {
    //             for i in 0..a.len() {
    //                 dest.field_mut(field.clone())[i] +=
    //                     a.field(field.clone())[i] + m * b.field(field.clone())[i];
    //             }
    //         }
    //     });
    // }
}
