#![allow(clippy::needless_range_loop)]

mod boundary;
mod engine;
mod kernel;
mod mesh;
mod node;

pub use boundary::{
    BlockBoundary, Boundary, BoundaryKind, Condition, Conditions, PairConditions, ScalarConditions,
    SystemBC, BC,
};
pub use engine::{Engine, FdEngine, FdIntEngine};
pub use kernel::{
    Convolution, DissipationAxis, FourthOrder, Gradient, Hessian, Kernel, Kernels, Order,
    SecondOrder, SixthOrder, Value,
};
pub use mesh::{ExportVtkConfig, Mesh, MeshCheckpoint, SystemCheckpoint};
pub use node::{node_from_vertex, vertex_from_node, NodeSpace, NodeWindow};

use crate::system::{Pair, SystemLabel, SystemSlice, SystemValue};

/// A function maps one set of scalar fields to another.
pub trait Function<const N: usize>: Clone {
    type Input: SystemLabel;
    type Output: SystemLabel;

    fn evaluate(&self, engine: &impl Engine<N, Self::Input>) -> SystemValue<Self::Output>;
}

/// A projection takes in a position and returns a system of values.
pub trait Projection<const N: usize>: Clone {
    type Output: SystemLabel;

    fn project(&self, position: [f64; N]) -> SystemValue<Self::Output>;
}

// impl<const N: usize, F: Projection<N>> Function<N> for F {
//     type Input = Empty;
//     type Output = F::Output;

//     fn evaluate(&self, engine: &impl Engine<N, Self::Input>) -> SystemValue<Self::Output> {
//         self.project(engine.position())
//     }
// }
/// An operator transforms a set of scalar fields, using an additional set of scalar fields
/// as context.
pub trait Operator<const N: usize>: Clone {
    type System: SystemLabel;
    type Context: SystemLabel;

    type Boundary: Boundary<N>;
    fn boundary(&self) -> Self::Boundary;

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

impl<const N: usize, O: Operator<N>> Function<N> for O {
    type Input = Pair<O::System, O::Context>;
    type Output = O::System;

    fn evaluate(&self, engine: &impl Engine<N, Self::Input>) -> SystemValue<Self::Output> {
        self.apply(engine)
    }
}

// #[derive(Clone)]
// pub struct DissipationOperator<O, B>(pub B);

// impl<const N: usize, const ORDER: usize, B: Boundary<N> + Conditions<N>> Operator<N>
//     for DissipationOperator<ORDER, B>
// {
//     type System = B::System;
//     type Context = Empty;

//     type Boundary = B;
//     fn boundary(&self) -> Self::Boundary {
//         self.0.clone()
//     }

//     fn apply(
//         &self,
//         engine: &impl Engine<N>,
//         input: SystemFields<'_, Self::System>,
//         _context: SystemFields<'_, Self::Context>,
//     ) -> SystemValue<Self::System> {
//         SystemValue::<Self::System>::from_fn(|system| {
//             let mut result = 0.0;

//             for axis in 0..N {
//                 let value = engine.evaluate(
//                     Single(Dissipation::<ORDER>, axis),
//                     SystemBC::new(system.clone(), self.0.clone()),
//                     input.field(system.clone()),
//                 );

//                 result += value;
//             }

//             result
//         })
//     }
// }
