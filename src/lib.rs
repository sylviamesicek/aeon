pub mod common;
pub mod geometry;
pub mod lac;
pub mod uniform;

pub mod prelude {
    pub use crate::common::{Block, Boundary, Convolution, Kernel, Operator};
    pub use crate::geometry::{IndexSpace, Rectangle};
    pub use crate::lac::{BiCGStabConfig, BiCGStabSolver, IdentityMap, LinearMap, LinearSolver};
    pub use crate::uniform::{UniformMesh, UniformMultigrid};
}
