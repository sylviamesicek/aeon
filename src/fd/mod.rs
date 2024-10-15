#![allow(clippy::needless_range_loop)]

mod boundary;
mod engine;
mod mesh;

use std::array;

pub use boundary::{
    BlockBoundary, Conditions, EmptyConditions, PairConditions, ScalarConditions, SystemBC,
    SystemCondition,
};
pub use engine::{Engine, FdEngine, FdIntEngine};
pub use mesh::{ExportVtuConfig, Mesh, MeshCheckpoint, SystemCheckpoint};

use crate::system::{Empty, Pair, Scalar, SystemLabel, SystemSlice, SystemValue};

/// A function maps one set of scalar fields to another.
pub trait Function<const N: usize>: Clone {
    type Input: SystemLabel;
    type Output: SystemLabel;

    type Conditions: Conditions<N, System = Self::Input>;
    fn conditions(&self) -> Self::Conditions;

    fn evaluate(&self, engine: &impl Engine<N, Self::Input>) -> SystemValue<Self::Output>;
}

/// A projection takes in a position and returns a system of values.
pub trait Projection<const N: usize>: Clone {
    type Output: SystemLabel;

    fn project(&self, position: [f64; N]) -> SystemValue<Self::Output>;
}

/// An operator transforms a set of scalar fields, using an additional set of scalar fields
/// as context.
pub trait Operator<const N: usize>: Clone {
    type System: SystemLabel;
    type Context: SystemLabel;

    type SystemConditions: Conditions<N, System = Self::System>;
    fn system_conditions(&self) -> Self::SystemConditions;

    type ContextConditions: Conditions<N, System = Self::Context>;
    fn context_conditions(&self) -> Self::ContextConditions;

    fn apply(
        &self,
        engine: &impl Engine<N, Pair<Self::System, Self::Context>>,
    ) -> SystemValue<Self::System>;

    fn callback(
        &self,
        _mesh: &mut Mesh<N>,
        _system: SystemSlice<Self::System>,
        _context: SystemSlice<Self::Context>,
        _index: usize,
    ) {
    }
}

#[derive(Clone)]
pub struct ProjectionAsFunction<P>(pub P);

impl<const N: usize, P: Projection<N>> Function<N> for ProjectionAsFunction<P> {
    type Input = Empty;
    type Output = P::Output;

    type Conditions = EmptyConditions;

    fn conditions(&self) -> Self::Conditions {
        EmptyConditions
    }

    fn evaluate(&self, engine: &impl Engine<N, Self::Input>) -> SystemValue<Self::Output> {
        self.0.project(engine.position())
    }
}

#[derive(Clone)]
pub struct OperatorAsFunction<O>(pub O);

impl<const N: usize, O: Operator<N>> Function<N> for OperatorAsFunction<O> {
    type Input = Pair<O::System, O::Context>;
    type Output = O::System;

    type Conditions = PairConditions<O::SystemConditions, O::ContextConditions>;

    fn conditions(&self) -> Self::Conditions {
        PairConditions {
            left: self.0.system_conditions(),
            right: self.0.context_conditions(),
        }
    }

    fn evaluate(&self, engine: &impl Engine<N, Self::Input>) -> SystemValue<Self::Output> {
        self.0.apply(engine)
    }
}

#[derive(Clone)]
pub struct DissipationFunction<C>(pub C);

impl<const N: usize, C: Conditions<N>> Function<N> for DissipationFunction<C> {
    type Input = C::System;
    type Output = C::System;

    type Conditions = C;

    fn conditions(&self) -> Self::Conditions {
        self.0.clone()
    }

    fn evaluate(&self, engine: &impl Engine<N, Self::Input>) -> SystemValue<Self::Output> {
        SystemValue::from_fn(|field: C::System| engine.dissipation(field))
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
    type Output = Scalar;

    fn project(&self, position: [f64; N]) -> SystemValue<Self::Output> {
        let offset: [_; N] = array::from_fn(|axis| position[axis] - self.center[axis]);
        let r2: f64 = offset.map(|v| v * v).iter().sum();
        let s2 = self.sigma * self.sigma;

        SystemValue::new([self.amplitude * (-r2 / s2).exp()])
    }
}
