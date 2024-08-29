#![allow(clippy::needless_range_loop)]

mod boundary;
mod engine;
// mod engine2;
mod kernel;
mod mesh;
mod node;

pub use boundary::{BlockBC, Boundary, BoundaryKind, Condition, Conditions, SystemBC, UnitBC, BC};
pub use engine::{Engine, FdEngine, FdIntEngine};
pub use kernel::{Derivative, Dissipation, Interpolation, SecondDerivative};
pub use mesh::{ExportVtkConfig, Mesh, MeshCheckpoint, MeshOrder, SystemCheckpoint};
pub use node::{NodeSpace, NodeWindow};

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
        _mesh: &mut Mesh<N>,
        _system: SystemSlice<Self::System>,
        _context: SystemSlice<Self::Context>,
        _index: usize,
    ) {
    }
}
