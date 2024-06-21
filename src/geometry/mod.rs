//! A module for various geometric primatives (AABBs, working with cartesian indices, etc.).

#![allow(clippy::needless_range_loop)]

mod axis;
mod face;
mod indices;
mod rectangle;
mod region;
mod tree;

pub use axis::AxisMask;
pub use face::{faces, Face, FaceIter};
pub use indices::{CartesianIter, CartesianWindowIter, IndexSpace};
pub use rectangle::Rectangle;
pub use region::{
    regions, Region, RegionFaceVertexIter, RegionIter, RegionNodeIter, RegionOffsetNodeIter,
};
pub use tree::{SpatialTree, SPATIAL_BOUNDARY};
