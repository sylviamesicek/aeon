#![allow(clippy::needless_range_loop)]

mod boundary;
mod discrete;
mod engine;
mod kernel;
mod mesh;
mod model;
mod node;

pub use boundary::{
    BlockBC, Boundary, BoundaryKind, Condition, Conditions, Domain, SystemBC, UnitBC, BC,
};
pub use discrete::{Discretization, DiscretizationOrder};
pub use engine::{Engine, FdEngine, FdIntEngine};
pub use kernel::{BasisOperator, Dissipation, Interpolation, Order, Support};
pub use mesh::Mesh;
pub use model::{ExportVtkConfig, Model};
pub use node::{node_from_vertex, NodeSpace, NodeWindow};

use crate::system::{SystemFields, SystemLabel, SystemValue};

/// A function takes in a position and returns a system of values.
pub trait Function<const N: usize>: Clone {
    type Output: SystemLabel;

    fn evaluate(&self, position: [f64; N]) -> SystemValue<Self::Output>;
}

// #[derive(Clone)]
// pub struct ProjectionFromFunction<const N: usize, I: Function<N>>(pub I);

// impl<const N: usize, I: Function<N>> Projection<N> for ProjectionFromFunction<N, I> {
//     type Output = I::Output;
//     type Input = Empty;

//     fn project(
//         &self,
//         engine: &impl Engine<N>,
//         _input: SystemFields<'_, Self::Input>,
//     ) -> SystemValue<Self::Output> {
//         let position = engine.position();
//         self.0.evaluate(position)
//     }
// }

/// A projection maps one set of scalar fields to another.
pub trait Projection<const N: usize>: Clone {
    type Input: SystemLabel;
    type Output: SystemLabel;

    fn project(
        &self,
        engine: &impl Engine<N>,
        input: SystemFields<'_, Self::Input>,
    ) -> SystemValue<Self::Output>;
}

/// An operator transforms a set of scalar fields, using an additional set of scalar fields
/// as context.
pub trait Operator<const N: usize>: Clone {
    type System: SystemLabel;
    type Context: SystemLabel;

    fn apply(
        &self,
        engine: &impl Engine<N>,
        input: SystemFields<'_, Self::System>,
        context: SystemFields<'_, Self::Context>,
    ) -> SystemValue<Self::System>;
}
