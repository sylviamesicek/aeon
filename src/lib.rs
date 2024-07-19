// Included so we can use custom derive macros from `aeon_macros` within this crate.
extern crate self as aeon;

pub mod array;
pub mod elliptic;
pub mod fd;
pub mod geometry;
pub mod lac;
pub mod ode;
pub mod system;

/// Provides common types used for most `aeon` applications.
pub mod prelude {
    // pub use crate::arena::Arena;
    pub use crate::fd::{Mesh, Model, Operator, Projection};
    pub use crate::geometry::{Face, IndexSpace, Rectangle};
    pub use crate::lac::{BiCGStabConfig, BiCGStabSolver, IdentityMap, LinearMap, LinearSolver};
    pub use crate::ode::{ForwardEuler, Ode, Rk4};
    pub use crate::system::{
        Empty, Scalar, SystemLabel, SystemSlice, SystemSliceMut, SystemValue, SystemVec,
    };
}
