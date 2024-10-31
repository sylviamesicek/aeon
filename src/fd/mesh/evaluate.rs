use std::marker::PhantomData;

use aeon_basis::{Boundary, Kernels};
use aeon_geometry::{Face, IndexSpace};
use rayon::iter::{ParallelBridge, ParallelIterator};
use reborrow::Reborrow as _;

use crate::{
    fd::{
        Engine, FdEngine, FdIntEngine, Function, Operator, OperatorAsFunction, Projection,
        ProjectionAsFunction,
    },
    system::{
        SystemFields, SystemFieldsMut, SystemLabel, SystemSlice, SystemSliceMut, SystemValue,
    },
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
        let source = source.as_range();
        let dest = dest.as_range();

        (0..self.blocks.len()).par_bridge().for_each(|block| {
            let nodes = self.block_nodes(block);

            let input = unsafe { source.slice(nodes.clone()).fields() };
            let output = unsafe { dest.slice_mut(nodes.clone()).fields_mut() };

            self.evaluate_block(order, &boundary, &function, block, input, output);
        });
    }

    fn evaluate_block<K: Kernels + Sync, B: Boundary<N> + Sync, P: Function<N> + Sync>(
        &self,
        order: K,
        boundary: &B,
        projection: &P,
        block: usize,
        input: SystemFields<'_, P::Input>,
        mut output: SystemFieldsMut<'_, P::Output>,
    ) {
        let bounds = self.block_bounds(block);
        let space = self.block_space(block);
        let vertex_size = space.inner_size();

        let boundary = self.block_boundary(block, boundary.clone());

        for vertex in IndexSpace::new(vertex_size).iter() {
            let result = if Self::is_interior::<K>(&boundary, vertex_size, vertex) {
                let engine = FdIntEngine {
                    space: space.clone(),
                    vertex,
                    bounds,
                    fields: input.rb(),
                    order,
                };

                projection.evaluate(&engine)
            } else {
                let engine = FdEngine {
                    space: space.clone(),
                    vertex,
                    bounds,
                    fields: input.rb(),
                    order,
                    boundary: boundary.clone(),
                };

                projection.evaluate(&engine)
            };

            let index = space.index_from_vertex(vertex);

            for field in P::Output::fields() {
                output.field_mut(field.clone())[index] = result.field(field.clone());
            }
        }
    }

    /// Determines if a vertex is not within `ORDER` of any weakly enforced boundary.
    fn is_interior<K: Kernels>(
        boundary: &impl Boundary<N>,
        vertex_size: [usize; N],
        vertex: [usize; N],
    ) -> bool {
        let mut result = true;

        for axis in 0..N {
            result &=
                boundary.kind(Face::negative(axis)).has_ghost() || vertex[axis] >= K::MAX_BORDER;
            result &= boundary.kind(Face::positive(axis)).has_ghost()
                || vertex[axis] < vertex_size[axis] - K::MAX_BORDER;
        }

        result
    }

    pub fn project<K: Kernels + Sync, B: Boundary<N> + Sync, P: Projection<N> + Sync>(
        &mut self,
        order: K,
        boundary: B,
        projection: P,
        dest: SystemSliceMut<'_, P::Output>,
    ) {
        self.evaluate(
            order,
            boundary,
            ProjectionAsFunction(projection),
            SystemSlice::empty(),
            dest,
        );
    }

    /// Applies the given operator to `source`, storing the result in `dest`, and utilizing `context` to store
    /// extra fields.
    pub fn apply<K: Kernels + Sync, B: Boundary<N> + Sync, O: Operator<N> + Sync>(
        &mut self,
        order: K,
        boundary: B,
        operator: O,
        source: SystemSlice<'_, O::System>,
        context: SystemSlice<'_, O::Context>,
        dest: SystemSliceMut<'_, O::System>,
    ) {
        let source = source.as_range();
        let context = context.as_range();
        let dest = dest.as_range();

        (0..self.blocks.len()).par_bridge().for_each(|block| {
            let nodes = self.block_nodes(block);

            let source = unsafe { source.slice(nodes.clone()).fields() };
            let context = unsafe { context.slice(nodes.clone()).fields() };
            let dest = unsafe { dest.slice_mut(nodes.clone()).fields_mut() };

            let input = SystemFields::join_pair(source, context);
            let output = dest;

            self.evaluate_block(
                order,
                &boundary,
                &OperatorAsFunction(operator.clone()),
                block,
                input,
                output,
            );
        });
    }

    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn dissipation<K: Kernels + Sync, B: Boundary<N> + Sync, S: SystemLabel>(
        &mut self,
        order: K,
        boundary: B,
        source: SystemSlice<'_, S>,
        dest: SystemSliceMut<'_, S>,
    ) {
        #[derive(Clone)]
        pub struct Dissipation<S: SystemLabel>(PhantomData<S>);

        impl<const N: usize, S: SystemLabel> Function<N> for Dissipation<S> {
            type Input = S;
            type Output = S;

            fn evaluate(&self, engine: &impl Engine<N, Self::Input>) -> SystemValue<Self::Output> {
                SystemValue::from_fn(|field: S| engine.dissipation(field))
            }
        }

        self.evaluate(order, boundary, Dissipation(PhantomData), source, dest);
    }

    /// Performs a fused-multiply-add operation `a + mb` and then assigns this value to `dest`.
    pub fn fma<Label: SystemLabel>(
        &mut self,
        a: SystemSlice<'_, Label>,
        m: f64,
        b: SystemSlice<'_, Label>,
        dest: SystemSliceMut<'_, Label>,
    ) {
        assert!(a.len() == self.num_nodes());
        assert!(b.len() == self.num_nodes());
        assert!(dest.len() == self.num_nodes());

        let a = a.as_range();
        let b = b.as_range();
        let dest = dest.as_range();

        (0..self.blocks.len()).par_bridge().for_each(|block| {
            let nodes = self.block_nodes(block);

            let a = unsafe { a.slice(nodes.clone()).fields() };
            let b = unsafe { b.slice(nodes.clone()).fields() };
            let mut dest = unsafe { dest.slice_mut(nodes.clone()).fields_mut() };

            for field in Label::fields() {
                for i in 0..a.len() {
                    dest.field_mut(field.clone())[i] =
                        a.field(field.clone())[i] + m * b.field(field.clone())[i];
                }
            }
        });
    }

    /// Performs a fused-multiply-add operation `a + mb`, and then adds and assigns this value to `dest`.
    pub fn add_assign_fma<Label: SystemLabel>(
        &mut self,
        a: SystemSlice<'_, Label>,
        m: f64,
        b: SystemSlice<'_, Label>,
        dest: SystemSliceMut<'_, Label>,
    ) {
        assert!(a.len() == self.num_nodes());
        assert!(b.len() == self.num_nodes());
        assert!(dest.len() == self.num_nodes());

        let a = a.as_range();
        let b = b.as_range();
        let dest = dest.as_range();

        (0..self.blocks.len()).par_bridge().for_each(|block| {
            let nodes = self.block_nodes(block);

            let a = unsafe { a.slice(nodes.clone()).fields() };
            let b = unsafe { b.slice(nodes.clone()).fields() };
            let mut dest = unsafe { dest.slice_mut(nodes.clone()).fields_mut() };

            for field in Label::fields() {
                for i in 0..a.len() {
                    dest.field_mut(field.clone())[i] +=
                        a.field(field.clone())[i] + m * b.field(field.clone())[i];
                }
            }
        });
    }
}
