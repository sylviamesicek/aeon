// pub mod arena;
pub mod array;
pub mod common;
pub mod geometry;
pub mod lac;
pub mod mesh;
pub mod ode;
pub mod system;
// pub mod uniform;

pub mod prelude {
    // pub use crate::arena::Arena;
    pub use crate::common::{Boundary, Kernel};
    pub use crate::geometry::{IndexSpace, Rectangle};
    pub use crate::lac::{BiCGStabConfig, BiCGStabSolver, IdentityMap, LinearMap, LinearSolver};
    pub use crate::mesh::{Driver, MemPool, Mesh};
    pub use crate::ode::{Ode, Rk4};
    pub use crate::system::{SystemLabel, SystemOwned, SystemSlice, SystemSliceMut};
    // pub use crate::uniform::{Model, UniformMesh, UniformMultigrid, UniformMultigridConfig};
}
