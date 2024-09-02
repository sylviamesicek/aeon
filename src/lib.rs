// Included so we can use custom derive macros from `aeon_macros` within this crate.
extern crate self as aeon;

pub mod elliptic;
pub mod fd;
pub mod lac;
pub mod ode;
pub mod system;

pub use aeon_geometry as geometry;

/// Provides common types used for most `aeon` applications.
pub mod prelude {
    // pub use crate::arena::Arena;
    pub use crate::fd::{
        Boundary, BoundaryKind, Condition, Conditions, Engine, ExportVtkConfig, Projection,
        Gradient, Hessian, Mesh, MeshCheckpoint, Operator, Function, SystemBC, SystemCheckpoint,
        ScalarConditions, Value,
    };
    pub use crate::lac::{BiCGStabConfig, BiCGStabSolver, IdentityMap, LinearMap, LinearSolver};
    pub use crate::ode::{ForwardEuler, Ode, Rk4};
    pub use crate::system::{
        Empty, Pair, Scalar, SystemFields, SystemFieldsMut, SystemLabel, SystemSlice,
        SystemSliceMut, SystemValue, SystemVec,
    };
    pub use aeon_geometry::{Face, IndexSpace, Rectangle};
}
