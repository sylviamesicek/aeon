#![allow(clippy::needless_range_loop)]

mod boundary;
mod engine;
mod kernel;
mod mesh;
mod model;
mod node;

pub use boundary::{Boundary, BoundaryKind, Condition, Conditions, Domain, SystemBC, BC};
pub use engine::{Engine, FdEngine, FdIntEngine};
pub use kernel::{BasisOperator, Interpolation, Order, Support};
pub use mesh::Mesh;
pub use model::Model;
pub use node::{node_from_vertex, NodeSpace, NodeWindow};

use crate::system::{SystemFields, SystemLabel, SystemValue};

pub trait Projection<const N: usize>: Clone {
    type Input: SystemLabel;
    type Output: SystemLabel;

    fn project(
        &self,
        engine: &impl Engine<N>,
        input: SystemFields<'_, Self::Input>,
    ) -> SystemValue<Self::Output>;
}

// impl<const N: usize, Output, T> Projection<N> for T
// where
//     Output: SystemLabel,
//     T: Fn([f64; N]) -> SystemValue<Output>,
// {
//     type Input = Empty;
//     type Output = Output;

//     fn project(
//         &self,
//         engine: &impl Engine<N>,
//         _input: SystemFields<'_, Self::Input>,
//     ) -> SystemValue<Self::Output> {
//         self(engine.position())
//     }
// }

pub trait Operator<const N: usize>: Clone {
    type System: SystemLabel;
    type Context: SystemLabel;

    fn evaluate(
        &self,
        engine: &impl Engine<N>,
        input: SystemFields<'_, Self::System>,
        context: SystemFields<'_, Self::Context>,
    ) -> SystemValue<Self::System>;
}
