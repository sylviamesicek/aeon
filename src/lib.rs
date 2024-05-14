pub mod array;
pub mod common;
pub mod geometry;
pub mod lac;
pub mod mesh;
pub mod ode;

/// Provides common types used for most `aeon` applications.
pub mod prelude {
    // pub use crate::arena::Arena;
    pub use crate::common::{Boundary, Kernel};
    pub use crate::geometry::{IndexSpace, Rectangle};
    pub use crate::lac::{BiCGStabConfig, BiCGStabSolver, IdentityMap, LinearMap, LinearSolver};
    pub use crate::mesh::{
        Block, Driver, MemPool, Mesh, Model, Operator, Projection, SystemLabel, SystemOperator,
        SystemProjection, SystemSlice, SystemSliceMut, SystemVec,
    };
    pub use crate::ode::{ForwardEuler, Ode, Rk4};
}
