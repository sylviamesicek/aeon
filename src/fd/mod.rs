#![allow(clippy::needless_range_loop)]

mod boundary;
mod discrete;
mod engine;
// mod engine2;
mod kernel;
// mod kernel2;
mod mesh;
mod model;
mod node;
// mod node2;
mod vertex;

pub use boundary::{BlockBC, Boundary, BoundaryKind, Condition, Conditions, SystemBC, UnitBC, BC};
pub use discrete::{Discretization, DiscretizationOrder};
pub use engine::{Engine, FdEngine, FdIntEngine};
pub use kernel::{BasisOperator, Dissipation, Interpolation, Order, Support};
pub use mesh::Mesh;
pub use model::{ExportVtkConfig, Model};
pub use node::{node_from_vertex, NodeSpace, NodeWindow};
pub use vertex::VertexSpace;

use crate::system::{SystemFields, SystemLabel, SystemSlice, SystemValue};

/// A function takes in a position and returns a system of values.
pub trait Function<const N: usize>: Clone {
    type Output: SystemLabel;

    fn evaluate(&self, position: [f64; N]) -> SystemValue<Self::Output>;
}

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

    fn callback(
        &self,
        _discrete: &mut Discretization<N>,
        _system: SystemSlice<Self::System>,
        _context: SystemSlice<Self::Context>,
        _index: usize,
    ) {
    }
}
