pub mod arena;
pub mod array;
pub mod common;
pub mod geometry;
pub mod lac;
pub mod uniform;

pub mod prelude {
    pub use crate::arena::Arena;
    pub use crate::common::{Block, Boundary, Kernel, Operator, Projection};
    pub use crate::geometry::{IndexSpace, Rectangle};
    pub use crate::lac::{BiCGStabConfig, BiCGStabSolver, IdentityMap, LinearMap, LinearSolver};
    pub use crate::uniform::{DataOut, UniformMesh, UniformMultigrid, UniformMultigridConfig};
}
