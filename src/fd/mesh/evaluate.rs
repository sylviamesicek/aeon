use std::array;

use aeon_basis::{Boundary, Kernels};
use aeon_geometry::{faces, Face, FaceMask, IndexSpace, NULL};
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    fd::{Engine, FdEngine, FdIntEngine, Function, Projection, ProjectionAsFunction},
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
    ) {
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

    pub fn evaluate_scalar<
        K: Kernels + Sync,
        B: Boundary<N> + Sync,
        P: Function<N, Input = Scalar, Output = Scalar> + Sync,
    >(
        &mut self,
        order: K,
        boundary: B,
        function: P,
        source: &[f64],
        dest: &mut [f64],
    ) {
        self.evaluate(order, boundary, function, source.into(), dest.into());
    }

    fn is_interior(boundary: &impl Boundary<N>) -> bool {
        let mut result = true;

        for axis in 0..N {
            result &= boundary.kind(Face::negative(axis)).has_ghost();
            result &= boundary.kind(Face::positive(axis)).has_ghost();
        }

        result
    }

    pub fn project_scalar<K: Kernels + Sync, B: Boundary<N> + Sync, P: Projection<N> + Sync>(
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
        source: SystemSlice<S>,
        mut dest: SystemSliceMut<S>,
    ) {
        #[derive(Clone)]
        pub struct Dissipation;

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

                    output[index] = 0.0;
                    for axis in 0..N {
                        output[index] += engine.dissipation(input, axis, vertex);
                    }
                }
            }
        }
        for field in source.system().enumerate() {
            self.evaluate(
                order,
                boundary.clone(),
                Dissipation,
                SystemSlice::from_scalar(source.field(field)),
                SystemSliceMut::from_scalar(dest.field_mut(field)),
            );
        }
    }

    /// This function computes the distance between each vertex and its nearest
    /// neighbor.
    pub fn min_spacing_per_vertex(&mut self, dest: &mut [f64]) {
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

    /// Performs a fused-multiply-add operation `a + mb` and then assigns this value to `dest`.
    pub fn fma<S: System + Clone>(
        &mut self,
        a: SystemSlice<'_, S>,
        m: f64,
        b: SystemSlice<'_, S>,
        dest: SystemSliceMut<'_, S>,
    ) {
        assert!(a.len() == self.num_nodes());
        assert!(b.len() == self.num_nodes());
        assert!(dest.len() == self.num_nodes());

        let dest = dest.into_shared();

        (0..self.blocks.len()).par_bridge().for_each(|block| {
            let nodes = self.block_nodes(block);

            let a = a.slice(nodes.clone());
            let b = b.slice(nodes.clone());
            let mut dest = unsafe { dest.slice_mut(nodes.clone()) };

            for field in a.system().enumerate() {
                for i in 0..a.len() {
                    dest.field_mut(field.clone())[i] =
                        a.field(field.clone())[i] + m * b.field(field.clone())[i];
                }
            }
        });
    }

    /// Performs a fused-multiply-add operation `a + mb`, and then adds and assigns this value to `dest`.
    pub fn add_assign_fma<S: System>(
        &mut self,
        a: SystemSlice<'_, S>,
        m: f64,
        b: SystemSlice<'_, S>,
        dest: SystemSliceMut<'_, S>,
    ) {
        assert!(a.len() == self.num_nodes());
        assert!(b.len() == self.num_nodes());
        assert!(dest.len() == self.num_nodes());

        let dest = dest.into_shared();

        (0..self.blocks.len()).par_bridge().for_each(|block| {
            let nodes = self.block_nodes(block);

            let a = a.slice(nodes.clone());
            let b = b.slice(nodes.clone());
            let mut dest = unsafe { dest.slice_mut(nodes.clone()) };

            for field in a.system().enumerate() {
                for i in 0..a.len() {
                    dest.field_mut(field.clone())[i] +=
                        a.field(field.clone())[i] + m * b.field(field.clone())[i];
                }
            }
        });
    }
}
