#![allow(clippy::needless_range_loop)]

mod boundary;
mod engine;
mod mesh;

use crate::system::{Empty, Pair, Scalar, SystemLabel, SystemSlice, SystemSliceMut};
use std::array;

use aeon_geometry::IndexSpace;
pub use boundary::{
    BlockBoundary, Conditions, EmptyConditions, PairConditions, ScalarConditions, SystemBC,
    SystemCondition,
};
pub use engine::{Engine, FdEngine, FdIntEngine};
pub use mesh::{ExportVtuConfig, Mesh, MeshCheckpoint, SystemCheckpoint};

/// A function maps one set of scalar fields to another.
pub trait Function<const N: usize>: Clone {
    type Input;
    type Output;

    fn evaluate<'a>(
        &'a self,
        engine: impl Engine<N>,
        input: impl SystemSlice<'a, Label = Self::Input>,
        output: impl SystemSliceMut<'a, Label = Self::Output>,
    );
}

/// A projection takes in a position and returns a system of values.
pub trait Projection<const N: usize>: Clone {
    fn project(&self, position: [f64; N]) -> f64;
}

/// An operator transforms a set of scalar fields, using an additional set of scalar fields
/// as context.
pub trait Operator<const N: usize>: Clone {
    type System;
    type Context;

    fn apply<'a>(
        &'a self,
        engine: impl Engine<N>,
        system: impl SystemSlice<'a, Label = Self::System>,
        context: impl SystemSlice<'a, Label = Self::Context>,
        dest: impl SystemSliceMut<'a, Label = Self::System>,
    );

    fn callback<'a>(
        &'a self,
        _mesh: &'a mut Mesh<N>,
        system: impl SystemSlice<'a, Label = Self::System>,
        context: impl SystemSlice<'a, Label = Self::Context>,
        _index: usize,
    ) {
    }
}

/// Transforms a projection into a function.
#[derive(Clone)]
pub struct ProjectionAsFunction<P>(pub P);

impl<const N: usize, P: Projection<N>> Function<N> for ProjectionAsFunction<P> {
    type Input = Empty;
    type Output = Scalar;

    fn evaluate<'a>(
        &'a self,
        engine: impl Engine<N>,
        _input: impl SystemSlice<'a, Label = Self::Input>,
        mut output: impl SystemSliceMut<'a, Label = Self::Output>,
    ) {
        let field = output.field_mut(Scalar);
        for vertex in IndexSpace::new(engine.size()).iter() {
            let index = engine.index(vertex);
            let position = engine.position(vertex);

            field[index] = self.0.project(position)
        }
    }
}

/// Transforms an operator into a function.
#[derive(Clone)]
pub struct OperatorAsFunction<O>(pub O);

impl<const N: usize, O: Operator<N>> Function<N> for OperatorAsFunction<O> {
    type Input = Pair<O::System, O::Context>;
    type Output = O::System;

    fn evaluate<'a>(
        &'a self,
        engine: impl Engine<N>,
        input: impl SystemSlice<'a, Label = Self::Input>,
        mut output: impl SystemSliceMut<'a, Label = Self::Output>,
    ) {
        // let field = output.field_mut(Scalar);
        // for vertex in IndexSpace::new(engine.size()).iter() {
        //     let index = engine.index(vertex);
        //     let position = engine.position(vertex);

        //     field[index] = self.0.project(position)
        // }
        todo!()
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
