#![allow(clippy::needless_range_loop)]

mod boundary;
mod driver;
mod engine;
mod kernel;
mod mesh;
mod model;
mod node;

pub use boundary::{Boundary, BoundaryKind, Condition, Conditions};
pub use driver::Driver;
pub use engine::{Engine, FdEngine};
pub use kernel::{BasisOperator, Interpolation, Order, Support};
pub use mesh::{BlockBoundary, Mesh};
pub use model::Model;
pub use node::{node_from_vertex, NodeSpace};

use crate::system::{SystemLabel, SystemSlice, SystemVal};

pub trait Projection<const N: usize> {
    type Input: SystemLabel;
    type Output: SystemLabel;

    fn project(
        &self,
        engine: &impl Engine<N>,
        input: SystemSlice<'_, Self::Input>,
    ) -> SystemVal<Self::Output>;
}

impl<const N: usize, Output, T> Projection<N> for T
where
    Output: SystemLabel,
    T: Fn([f64; N]) -> SystemVal<Output>,
{
    type Input = ();
    type Output = Output;

    fn project(
        &self,
        engine: &impl Engine<N>,
        _input: SystemSlice<'_, Self::Input>,
    ) -> SystemVal<Self::Output> {
        self(engine.position())
    }
}

pub trait Operator<const N: usize> {
    type System: SystemLabel;
    type Context: SystemLabel;

    fn evaluate(
        &self,
        engine: &impl Engine<N>,
        context: SystemSlice<'_, Self::Context>,
        input: SystemSlice<'_, Self::System>,
    ) -> SystemVal<Self::System>;
}
