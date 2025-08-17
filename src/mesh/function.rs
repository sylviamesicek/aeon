#![allow(clippy::needless_range_loop)]

use crate::mesh::Mesh;
use crate::{
    kernel::{NodeSpace, node_from_vertex},
    system::{System, SystemSlice, SystemSliceMut},
};
use std::ops::Range;

/// An interface for computing values, gradients, and hessians of fields.
pub trait Engine<const N: usize> {
    fn space(&self) -> &NodeSpace<N>;

    fn num_nodes(&self) -> usize {
        self.space().num_nodes()
    }

    fn vertex_size(&self) -> [usize; N] {
        self.space().vertex_size()
    }

    fn node_range(&self) -> Range<usize>;

    fn alloc<T: Default>(&self, len: usize) -> &mut [T];

    fn position(&self, vertex: [usize; N]) -> [f64; N] {
        self.space().position(node_from_vertex(vertex))
    }

    fn index_from_vertex(&self, vertex: [usize; N]) -> usize {
        self.space().index_from_vertex(vertex)
    }
    fn min_spacing(&self) -> f64 {
        let spacing = self.space().spacing();
        spacing
            .iter()
            .min_by(|a, b| a.total_cmp(&b))
            .cloned()
            .unwrap_or(1.0)
    }

    fn value(&self, field: &[f64], vertex: [usize; N]) -> f64;
    fn derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64;
    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64;
    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64;
    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64;
}

impl<'a, const N: usize, E: Engine<N>> Engine<N> for &'a E {
    fn space(&self) -> &NodeSpace<N> {
        (&**self).space()
    }
    fn num_nodes(&self) -> usize {
        (**self).num_nodes()
    }

    fn node_range(&self) -> Range<usize> {
        (**self).node_range()
    }

    fn vertex_size(&self) -> [usize; N] {
        (**self).vertex_size()
    }

    fn alloc<T: Default>(&self, len: usize) -> &mut [T] {
        (**self).alloc(len)
    }

    fn position(&self, vertex: [usize; N]) -> [f64; N] {
        (**self).position(vertex)
    }

    fn index_from_vertex(&self, vertex: [usize; N]) -> usize {
        (**self).index_from_vertex(vertex)
    }

    fn min_spacing(&self) -> f64 {
        (**self).min_spacing()
    }

    fn value(&self, field: &[f64], vertex: [usize; N]) -> f64 {
        (**self).value(field, vertex)
    }

    fn derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        (**self).derivative(field, axis, vertex)
    }

    fn second_derivative(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        (**self).second_derivative(field, axis, vertex)
    }

    fn mixed_derivative(&self, field: &[f64], i: usize, j: usize, vertex: [usize; N]) -> f64 {
        (**self).mixed_derivative(field, i, j, vertex)
    }

    fn dissipation(&self, field: &[f64], axis: usize, vertex: [usize; N]) -> f64 {
        (**self).dissipation(field, axis, vertex)
    }
}

/// A function maps one set of scalar fields to another.
pub trait Function<const N: usize> {
    type Input: System;
    type Output: System;
    type Error;

    /// Action of the function on an individual finite different block.
    fn evaluate(
        &self,
        engine: impl Engine<N>,
        input: SystemSlice<Self::Input>,
        output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Self::Error>;

    /// An (optional) preprocessing step run immediately before a function is applied, after
    /// boundary conditions have been filled.
    fn preprocess(
        &mut self,
        _mesh: &mut Mesh<N>,
        _input: SystemSliceMut<Self::Input>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// A projection takes in a position and returns a system of values.
pub trait Projection<const N: usize> {
    fn project(&self, position: [f64; N]) -> f64;
}

/// A Gaussian projection centered on a given point, with an amplitude and a sigma.
#[derive(Clone, Copy)]
pub struct Gaussian<const N: usize> {
    pub amplitude: f64,
    pub sigma: [f64; N],
    pub center: [f64; N],
}

impl<const N: usize> Projection<N> for Gaussian<N> {
    fn project(&self, position: [f64; N]) -> f64 {
        let offset: [_; N] =
            core::array::from_fn(|axis| (position[axis] - self.center[axis]) / self.sigma[axis]);
        let r2: f64 = offset.map(|v| v * v).iter().sum();
        self.amplitude * (-r2).exp()
    }
}

#[derive(Clone, Copy)]
pub struct TanH<const N: usize> {
    pub amplitude: f64,
    pub sigma: f64,
    pub center: [f64; N],
}

impl<const N: usize> Projection<N> for TanH<N> {
    fn project(&self, position: [f64; N]) -> f64 {
        let offset =
            core::array::from_fn::<_, N, _>(|axis| (position[axis] - self.center[axis]).powi(2))
                .iter()
                .sum::<f64>()
                .sqrt()
                / self.sigma;
        self.amplitude * offset.tanh()
    }
}

pub struct FunctionBorrowMut<'a, I>(pub &'a mut I);

impl<'a, const N: usize, F: Function<N>> Function<N> for FunctionBorrowMut<'a, F> {
    type Input = F::Input;
    type Output = F::Output;
    type Error = F::Error;

    fn preprocess(
        &mut self,
        mesh: &mut Mesh<N>,
        input: SystemSliceMut<Self::Input>,
    ) -> Result<(), Self::Error> {
        self.0.preprocess(mesh, input)
    }

    fn evaluate(
        &self,
        engine: impl Engine<N>,
        input: SystemSlice<Self::Input>,
        output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Self::Error> {
        self.0.evaluate(engine, input, output)
    }
}
