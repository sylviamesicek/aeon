use aeon_basis::{Boundary, BoundaryKind, Kernels, Order};
use aeon_geometry::{Face, IndexSpace};
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    fd::{
        Engine, FdEngine, FdIntEngine, Function, Operator, OperatorAsFunction, Projection,
        ProjectionAsFunction,
    },
    system::{LabelledSystem, Scalar, System, SystemSlice, SystemSliceMut, SystemSliceShared},
};

use super::Mesh;

impl<const N: usize> Mesh<N> {
    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn evaluate<'a, K: Kernels + Sync, B: Boundary<N> + Sync, P: Function<N> + Sync>(
        &mut self,
        order: K,
        boundary: B,
        function: P,
        source: impl SystemSlice<'a, Label = P::Input>,
        dest: impl SystemSliceMut<'a, Label = P::Output>,
    ) {
        let dest = dest.into_shared();

        self.block_compute(|mesh, store, block| {
            let nodes = mesh.block_nodes(block);

            let input = source.slice(nodes.clone());
            let output = unsafe { dest.slice_unsafe_mut(nodes.clone()) };

            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block);

            let boundary = mesh.block_boundary(block, boundary.clone());

            if Self::is_interior(&boundary) {
                let engine = FdEngine {
                    space,
                    boundary: boundary.clone(),
                    bounds,
                    order,
                    store,
                };

                function.evaluate(engine, input, output);
            } else {
                let engine = FdIntEngine {
                    space,
                    bounds,
                    order,
                    store,
                };

                function.evaluate(engine, input, output);
            }
        });
    }

    // fn evaluate_block<K: Kernels + Sync, B: Boundary<N> + Sync, P: Function<N> + Sync>(
    //     &self,
    //     order: K,
    //     boundary: &B,
    //     projection: &P,
    //     block: usize,
    //     input: SystemFields<'_, P::Input>,
    //     mut output: SystemFieldsMut<'_, P::Output>,
    // ) {
    //     let bounds = self.block_bounds(block);
    //     let space = self.block_space(block);
    //     let vertex_size = space.inner_size();

    //     let boundary = self.block_boundary(block, boundary.clone());

    //     for vertex in IndexSpace::new(vertex_size).iter() {
    //         let result = if Self::is_interior::<K>(&boundary, vertex_size, vertex) {
    //             let engine = FdIntEngine {
    //                 space: space.clone(),
    //                 vertex,
    //                 bounds,
    //                 fields: input.rb(),
    //                 order,
    //             };

    //             projection.evaluate(&engine)
    //         } else {
    //             let engine = FdEngine {
    //                 space: space.clone(),
    //                 vertex,
    //                 bounds,
    //                 fields: input.rb(),
    //                 order,
    //                 boundary: boundary.clone(),
    //             };

    //             projection.evaluate(&engine)
    //         };

    //         let index = space.index_from_vertex(vertex);

    //         for field in P::Output::fields() {
    //             output.field_mut(field.clone())[index] = result.field(field.clone());
    //         }
    //     }
    // }

    // /// Determines if a vertex is not within `ORDER` of any weakly enforced boundary.
    // fn is_interior<K: Kernels>(
    //     boundary: &impl Boundary<N>,
    //     vertex_size: [usize; N],
    //     vertex: [usize; N],
    // ) -> bool {
    //     let mut result = true;

    //     for axis in 0..N {
    //         result &=
    //             boundary.kind(Face::negative(axis)).has_ghost() || vertex[axis] >= K::MAX_BORDER;
    //         result &= boundary.kind(Face::positive(axis)).has_ghost()
    //             || vertex[axis] < vertex_size[axis] - K::MAX_BORDER;
    //     }

    //     result
    // }

    fn is_interior(boundary: &impl Boundary<N>) -> bool {
        let mut result = true;

        for axis in 0..N {
            result &= boundary.kind(Face::negative(axis)).has_ghost();
            result &= boundary.kind(Face::positive(axis)).has_ghost();
        }

        result
    }

    pub fn project<'a, K: Kernels + Sync, B: Boundary<N> + Sync, P: Projection<N> + Sync>(
        &mut self,
        projection: P,
        mut dest: impl SystemSliceMut<'a, Label = Scalar>,
    ) {
        #[derive(Clone)]
        struct TrivialBoundary;

        impl<const N: usize> Boundary<N> for TrivialBoundary {
            fn kind(&self, _face: Face<N>) -> BoundaryKind {
                BoundaryKind::Custom
            }
        }

        let system = LabelledSystem::empty();
        let slice = system.as_slice();

        self.evaluate(
            Order::<2>,
            TrivialBoundary,
            ProjectionAsFunction(projection),
            slice,
            dest.slice_mut(0..dest.len()),
        );
    }

    /// Applies the given operator to `source`, storing the result in `dest`, and utilizing `context` to store
    /// extra fields.
    pub fn apply<'a, K: Kernels + Sync, B: Boundary<N> + Sync, O: Operator<N> + Sync>(
        &'a mut self,
        order: K,
        boundary: B,
        operator: O,
        source: impl SystemSlice<'a, Label = O::System>,
        context: impl SystemSlice<'a, Label = O::Context>,
        dest: impl SystemSliceMut<'a, Label = O::System>,
    ) {
        self.evaluate(
            order,
            boundary,
            OperatorAsFunction(operator),
            (source, context),
            dest,
        );
    }

    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn dissipation<'a, K: Kernels + Sync, B: Boundary<N> + Sync, S: Clone>(
        &'a mut self,
        order: K,
        boundary: B,
        source: impl SystemSlice<'a, Label = S>,
        mut dest: impl SystemSliceMut<'a, Label = S>,
    ) {
        #[derive(Clone)]
        pub struct Dissipation;

        impl<const N: usize> Function<N> for Dissipation {
            type Input = Scalar;
            type Output = Scalar;

            fn evaluate<'a>(
                &'a self,
                engine: impl Engine<N>,
                input: impl SystemSlice<'a, Label = Self::Input>,
                mut output: impl SystemSliceMut<'a, Label = Self::Output>,
            ) {
                for vertex in IndexSpace::new(engine.size()).iter() {
                    let index = engine.index(vertex);

                    let mut result = 0.0;

                    for axis in 0..N {
                        result += engine.dissipation(input.field(Scalar), axis, vertex);
                    }

                    output.field_mut(Scalar)[index] = result;
                }
            }
        }

        for field in source.enumerate() {
            self.evaluate(
                order,
                boundary.clone(),
                Dissipation,
                source.field(field.clone()),
                dest.field_mut(field.clone()),
            );
        }
    }

    /// Performs a fused-multiply-add operation `a + mb` and then assigns this value to `dest`.
    pub fn fma<'a, S: Clone>(
        &'a mut self,
        a: impl SystemSlice<'a, Label = S>,
        m: f64,
        b: impl SystemSlice<'a, Label = S>,
        dest: impl SystemSliceMut<'a, Label = S>,
    ) {
        assert!(a.len() == self.num_nodes());
        assert!(b.len() == self.num_nodes());
        assert!(dest.len() == self.num_nodes());

        let dest = dest.into_shared();

        (0..self.blocks.len()).par_bridge().for_each(|block| {
            let nodes = self.block_nodes(block);

            let a = a.slice(nodes.clone());
            let b = b.slice(nodes.clone());
            let mut dest = unsafe { dest.slice_unsafe_mut(nodes.clone()) };

            for field in a.enumerate() {
                for i in 0..a.len() {
                    dest.field_mut(field.clone())[i] =
                        a.field(field.clone())[i] + m * b.field(field.clone())[i];
                }
            }
        });
    }

    /// Performs a fused-multiply-add operation `a + mb`, and then adds and assigns this value to `dest`.
    pub fn add_assign_fma<'a, S: Clone>(
        &'a mut self,
        a: impl SystemSlice<'a, Label = S>,
        m: f64,
        b: impl SystemSlice<'a, Label = S>,
        dest: impl SystemSliceMut<'a, Label = S>,
    ) {
        assert!(a.len() == self.num_nodes());
        assert!(b.len() == self.num_nodes());
        assert!(dest.len() == self.num_nodes());

        let dest = dest.into_shared();

        (0..self.blocks.len()).par_bridge().for_each(|block| {
            let nodes = self.block_nodes(block);

            let a = a.slice(nodes.clone());
            let b = b.slice(nodes.clone());
            let mut dest = unsafe { dest.slice_unsafe_mut(nodes.clone()) };

            for field in a.enumerate() {
                for i in 0..a.len() {
                    dest.field_mut(field.clone())[i] +=
                        a.field(field.clone())[i] + m * b.field(field.clone())[i];
                }
            }
        });
    }
}
