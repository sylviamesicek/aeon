use std::ops::Range;

use aeon_basis::{node_from_vertex, Boundary, Hessian, Kernels, NodeSpace, VertexKernel};
use aeon_geometry::Rectangle;

use super::mesh::MeshStore;

/// An interface for computing values, gradients, and hessians of fields.
pub trait Engine<const N: usize> {
    fn num_nodes(&self) -> usize;
    fn node_range(&self) -> Range<usize>;
    fn vertex_size(&self) -> [usize; N];

    fn alloc<T: Default>(&self, len: usize) -> &[T];

    fn position(&self, vertex: [usize; N]) -> [f64; N];
    fn index_from_vertex(&self, vertex: [usize; N]) -> usize;
    fn min_spacing(&self) -> f64;

    fn value(&self, field: &[f64], vertex: [usize; N]) -> f64;
    fn derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64;
    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64;
    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64;
    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64;
}

impl<'a, const N: usize, E: Engine<N>> Engine<N> for &'a E {
    fn num_nodes(&self) -> usize {
        (&**self).num_nodes()
    }

    fn node_range(&self) -> Range<usize> {
        (&**self).node_range()
    }

    fn vertex_size(&self) -> [usize; N] {
        (&**self).vertex_size()
    }

    fn alloc<T: Default>(&self, len: usize) -> &[T] {
        (&**self).alloc(len)
    }

    fn position(&self, vertex: [usize; N]) -> [f64; N] {
        (&**self).position(vertex)
    }

    fn index_from_vertex(&self, vertex: [usize; N]) -> usize {
        (&**self).index_from_vertex(vertex)
    }

    fn min_spacing(&self) -> f64 {
        (&**self).min_spacing()
    }

    fn value(&self, field: &[f64], vertex: [usize; N]) -> f64 {
        (&**self).value(field, vertex)
    }

    fn derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        (&**self).derivative(field, axis, vertex)
    }

    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        (&**self).second_derivative(field, axis, vertex)
    }

    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64 {
        (&**self).mixed_derivative(field, i, j, vertex)
    }

    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        (&**self).dissipation(field, axis, vertex)
    }
}

/// A finite difference engine of a given order, but potentially bordering a free boundary.
pub struct FdEngine<'store, const N: usize, K: Kernels, B: Boundary<N>> {
    pub space: NodeSpace<N>,
    pub bounds: Rectangle<N>,
    pub order: K,
    pub boundary: B,
    pub store: &'store MeshStore,
    pub range: Range<usize>,
}

impl<'store, const N: usize, K: Kernels, B: Boundary<N>> FdEngine<'store, N, K, B> {
    fn evaluate_axis(
        &self,
        field: &[f64],
        axis: usize,
        kernel: &impl VertexKernel,
        vertex: [usize; N],
    ) -> f64 {
        self.space.evaluate_axis(
            self.boundary.clone(),
            kernel,
            self.bounds,
            node_from_vertex(vertex),
            field,
            axis,
        )
    }
}

impl<'store, const N: usize, K: Kernels, B: Boundary<N>> Engine<N> for FdEngine<'store, N, K, B> {
    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn node_range(&self) -> Range<usize> {
        self.range.clone()
    }

    fn vertex_size(&self) -> [usize; N] {
        self.space.inner_size()
    }

    fn alloc<T: Default>(&self, len: usize) -> &[T] {
        self.store.scratch(len)
    }

    fn position(&self, vertex: [usize; N]) -> [f64; N] {
        self.space.position(node_from_vertex(vertex), self.bounds)
    }

    fn index_from_vertex(&self, vertex: [usize; N]) -> usize {
        self.space.index_from_vertex(vertex)
    }

    fn value(&self, field: &[f64], vertex: [usize; N]) -> f64 {
        let index = self.space.index_from_vertex(vertex);
        field[index]
    }

    fn derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate_axis(field, axis, K::derivative(), vertex)
    }

    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate_axis(field, axis, K::second_derivative(), vertex)
    }

    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64 {
        self.space.evaluate(
            self.boundary.clone(),
            Hessian::<K>::new(i, j),
            self.bounds,
            node_from_vertex(vertex),
            field,
        )
    }

    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate_axis(field, axis, K::dissipation(), vertex)
    }

    fn min_spacing(&self) -> f64 {
        let spacing = self.space.spacing(self.bounds);
        spacing
            .iter()
            .min_by(|a, b| a.total_cmp(&b))
            .cloned()
            .unwrap_or(1.0)
    }
}

/// A finite difference engine that only every relies on interior support (and can thus use better optimized stencils).
pub struct FdIntEngine<'store, const N: usize, K: Kernels> {
    pub space: NodeSpace<N>,
    pub bounds: Rectangle<N>,
    pub order: K,
    pub store: &'store MeshStore,
    pub range: Range<usize>,
}

impl<'store, const N: usize, K: Kernels> FdIntEngine<'store, N, K> {
    fn evaluate(
        &self,
        field: &[f64],
        axis: usize,
        kernel: &impl VertexKernel,
        vertex: [usize; N],
    ) -> f64 {
        self.space.evaluate_axis_interior(
            kernel,
            self.bounds,
            node_from_vertex(vertex),
            field,
            axis,
        )
    }
}

impl<'store, const N: usize, K: Kernels> Engine<N> for FdIntEngine<'store, N, K> {
    fn num_nodes(&self) -> usize {
        self.space.num_nodes()
    }

    fn node_range(&self) -> Range<usize> {
        self.range.clone()
    }

    fn vertex_size(&self) -> [usize; N] {
        self.space.inner_size()
    }

    fn alloc<T: Default>(&self, len: usize) -> &[T] {
        self.store.scratch(len)
    }

    fn position(&self, vertex: [usize; N]) -> [f64; N] {
        self.space.position(node_from_vertex(vertex), self.bounds)
    }

    fn index_from_vertex(&self, vertex: [usize; N]) -> usize {
        self.space.index_from_vertex(vertex)
    }

    fn value(&self, field: &[f64], vertex: [usize; N]) -> f64 {
        let index = self.space.index_from_vertex(vertex);
        field[index]
    }

    fn derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate(field, axis, K::derivative(), vertex)
    }

    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate(field, axis, K::second_derivative(), vertex)
    }

    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64 {
        self.space.evaluate_interior(
            Hessian::<K>::new(i, j),
            self.bounds,
            node_from_vertex(vertex),
            field,
        )
    }

    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        self.evaluate(field, axis, K::dissipation(), vertex)
    }

    fn min_spacing(&self) -> f64 {
        let spacing = self.space.spacing(self.bounds);
        spacing
            .iter()
            .min_by(|a, b| a.total_cmp(&b))
            .cloned()
            .unwrap_or(1.0)
    }
}
