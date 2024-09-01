#![allow(clippy::needless_range_loop)]

mod boundary;
mod engine;
mod kernel;
mod mesh;
mod node;

pub use boundary::{BlockBC, Boundary, BoundaryKind, Condition, Conditions, SystemBC, UnitBC, BC};
pub use engine::{Engine, FdEngine, FdIntEngine};
pub use kernel::{
    Convolution, FourthOrder, Gradient, Hessian, Kernel, Kernels, Order, SecondOrder, SixthOrder,
    Value,
};
pub use mesh::{ExportVtkConfig, Mesh, MeshCheckpoint, SystemCheckpoint};
pub use node::{node_from_vertex, vertex_from_node, NodeSpace, NodeWindow};

use crate::system::{Empty, Pair, SystemFields, SystemLabel, SystemSlice, SystemValue};

/// A function takes in a position and returns a system of values.
pub trait Function<const N: usize>: Clone {
    type Output: SystemLabel;

    fn evaluate(&self, position: [f64; N]) -> SystemValue<Self::Output>;
}

/// A projection maps one set of scalar fields to another.
pub trait Projection<const N: usize>: Clone {
    type Input: SystemLabel;
    type Output: SystemLabel;

    type Boundary: Boundary<N>;
    fn boundary(&self) -> Self::Boundary;

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

    type Boundary: Boundary<N>;
    fn boundary(&self) -> Self::Boundary;

    fn apply(
        &self,
        engine: &impl Engine<N>,
        input: SystemFields<'_, Self::System>,
        context: SystemFields<'_, Self::Context>,
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

// impl<const N: usize, O: Operator<N>> Projection<N> for Operator<N> {
//     type Input = Pair<O::System, O::Context>;
//     type Output = O::System;

//     fn project(
//             &self,
//             engine: &impl Engine<N>,
//             input: SystemFields<'_, Self::Input>,
//         ) -> SystemValue<Self::Output> {

//     }
// }

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
