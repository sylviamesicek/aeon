#![allow(clippy::needless_range_loop)]

mod boundary;
mod engine;
mod kernel;
mod node;
mod tree;

pub use boundary::{Boundary, BoundaryCondition};
pub use engine::{Engine, FdEngine};
pub use kernel::{Interpolation, Operator, Order, Support};
pub use node::{node_from_vertex, NodeSpace};
pub use tree::{TreeBlocks, TreeMesh};
