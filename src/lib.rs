#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

// Included so we can use custom derive macros from `aeon_macros` within this crate.
extern crate self as aeon;

pub mod elliptic;
pub mod fd;
pub mod lac;
pub mod ode;
pub mod shared;
pub mod system;
// pub mod system_old;

pub use aeon_basis as basis;
pub use aeon_geometry as geometry;

/// Provides common types used for most `aeon` applications.
pub mod prelude {
    pub use crate::fd::{
        Conditions, Engine, ExportVtuConfig, Function, Mesh, MeshCheckpoint, Projection,
        ScalarConditions, SystemBC, SystemCheckpoint,
    };
    pub use crate::lac::{BiCGStabConfig, BiCGStabSolver, IdentityMap, LinearMap, LinearSolver};
    pub use crate::ode::{ForwardEuler, Ode, Rk4};
    pub use crate::system::{Empty, Pair, Scalar, SystemSlice, SystemSliceMut, SystemVec};
    pub use aeon_basis::{Boundary, BoundaryKind, Condition, Gradient, Hessian, Order, Value};
    pub use aeon_geometry::{Face, IndexSpace, Rectangle};
}
