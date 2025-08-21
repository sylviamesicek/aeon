#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

// Included so we can use custom derive macros from `aeon_macros` within this crate.
extern crate self as aeon;

pub mod array;
pub mod element;
pub mod geometry;
pub mod image;
pub mod kernel;
pub mod mesh;
pub mod shared;
pub mod solver;
// pub mod system;
// mod uniform;

pub use aeon_macros as macros;

/// A common helper wrapper for implementing a trait T for types of the form &impl T
pub struct IRef<'a, B>(pub &'a B);

/// Provides common types used for most `aeon` applications.
pub mod prelude {
    pub use crate::geometry::{
        ActiveCellId, BlockId, CellId, Face, FaceArray, FaceMask, HyperBox, IndexSpace, NeighborId,
    };
    pub use crate::image::{Image, ImageMut, ImageRef, ImageShared};
    pub use crate::kernel::{
        BoundaryClass, BoundaryConds, BoundaryKind, DirichletParams, RadiativeParams,
        SystemBoundaryConds,
    };
    pub use crate::mesh::{
        Checkpoint, Engine, ExportStride, ExportVtuConfig, Function, Mesh, Projection,
    };
    pub use aeon_macros::SystemLabel;
}
