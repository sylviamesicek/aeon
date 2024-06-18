// Included so we can use custom derive macros from `aeon_macros` within this crate.
extern crate self as aeon;

pub mod array;
pub mod common;
pub mod elliptic;
pub mod fd;
pub mod geometry;
pub mod lac;
pub mod mesh;
pub mod ode;
pub mod system;

/// Provides common types used for most `aeon` applications.
pub mod prelude {
    // pub use crate::arena::Arena;
    pub use crate::common::{Boundary, BoundaryCondition, Kernel};
    pub use crate::geometry::{Face, IndexSpace, Rectangle};
    pub use crate::lac::{BiCGStabConfig, BiCGStabSolver, IdentityMap, LinearMap, LinearSolver};
    pub use crate::mesh::{
        Block, BlockExt, Driver, MemPool, Mesh, Model, Operator, Projection, SystemBoundary,
        SystemOperator, SystemProjection,
    };
    pub use crate::ode::{ForwardEuler, Ode, Rk4};
    pub use crate::system::{Scalar, SystemLabel, SystemSlice, SystemSliceMut, SystemVec};
}
