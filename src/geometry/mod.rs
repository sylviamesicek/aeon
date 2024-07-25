//! A module for various geometric primatives (AABBs, working with cartesian indices, etc.).

#![allow(clippy::needless_range_loop)]

mod axis;
mod face;
mod index;
mod rectangle;
mod region;
mod tree;

pub use axis::AxisMask;
pub use face::{faces, Face, FaceIter, FaceMask};
pub use index::{CartesianIter, CartesianWindowIter, IndexSpace};
pub use rectangle::Rectangle;
pub use region::{
    regions, Region, RegionFaceVertexIter, RegionIter, RegionNodeIter, RegionOffsetNodeIter, Side,
};
pub use tree::{Tree, TreeBlocks, TreeInterfaces, TreeNeighbors, TreeNodes, NULL};
