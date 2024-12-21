#![allow(clippy::needless_range_loop)]

mod boundary;
mod engine;
mod mesh;

use crate::system::{Empty, Scalar, System, SystemSlice, SystemSliceMut};
use std::array;

use aeon_geometry::IndexSpace;
pub use boundary::{
    BlockBoundary, Conditions, EmptyConditions, PairConditions, ScalarConditions, SystemBC,
    SystemCondition,
};
pub use engine::{Engine, FdEngine, FdIntEngine};
pub use mesh::{ExportVtuConfig, Mesh, MeshCheckpoint, SystemCheckpoint};

/// A function maps one set of scalar fields to another.
pub trait Function<const N: usize> {
    type Input: System;
    type Output: System;

    fn evaluate(
        &self,
        engine: impl Engine<N>,
        input: SystemSlice<Self::Input>,
        output: SystemSliceMut<Self::Output>,
    );

    // fn callback(
    //     &self,
    //     _mesh: &Mesh<N>,
    //     _input: SystemSlice<Self::Input>,
    //     _output: SystemSlice<Self::Output>,
    // ) {
    // }
}

/// A projection takes in a position and returns a system of values.
pub trait Projection<const N: usize> {
    fn project(&self, position: [f64; N]) -> f64;
}

/// Transforms a projection into a function.
#[derive(Clone)]
pub struct ProjectionAsFunction<P>(pub P);

impl<const N: usize, P: Projection<N>> Function<N> for ProjectionAsFunction<P> {
    type Input = Empty;
    type Output = Scalar;

    fn evaluate(
        &self,
        engine: impl Engine<N>,
        _input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        let dest = output.field_mut(());

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            dest[index] = self.0.project(engine.position(vertex))
        }
    }
}

/// A Gaussian projection centered on a given point, with an amplitude and a sigma.
#[derive(Clone, Copy)]
pub struct Gaussian<const N: usize> {
    pub amplitude: f64,
    pub sigma: f64,
    pub center: [f64; N],
}

impl<const N: usize> Projection<N> for Gaussian<N> {
    fn project(&self, position: [f64; N]) -> f64 {
        let offset: [_; N] = array::from_fn(|axis| position[axis] - self.center[axis]);
        let r2: f64 = offset.map(|v| v * v).iter().sum();
        let s2 = self.sigma * self.sigma;

        self.amplitude * (-r2 / s2).exp()
    }
}
