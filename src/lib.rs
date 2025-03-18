#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

// Included so we can use custom derive macros from `aeon_macros` within this crate.
extern crate self as aeon;

pub mod array;
pub mod geometry;
pub mod kernel;
pub mod mesh;
pub mod shared;
pub mod solver;
pub mod system;

pub use aeon_macros as macros;

/// Provides common types used for most `aeon` applications.
pub mod prelude {
    pub use crate::geometry::{Face, FaceArray, FaceMask, IndexSpace, Rectangle};
    pub use crate::kernel::{
        BoundaryClass, BoundaryConds, BoundaryKind, DirichletParams, Order, RadiativeParams,
    };
    pub use crate::mesh::{
        Engine, ExportVtuConfig, Function, Mesh, MeshCheckpoint, Projection, SystemCheckpoint,
    };
    pub use crate::system::{
        Empty, EmptyConditions, Pair, PairConditions, Scalar, ScalarConditions, System,
        SystemBoundaryConds, SystemSlice, SystemSliceMut, SystemVec,
    };
    pub use aeon_macros::SystemLabel;
}
