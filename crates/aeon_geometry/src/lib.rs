//! A module for various geometric primatives (AABBs, working with cartesian indices, etc.).
//!
//! This includes several classes which handle associating dofs with N-dimensional quadtrees.

#![allow(clippy::needless_range_loop)]

mod axis;
mod face;
mod index;
mod rectangle;
mod region;
mod tree;

pub use axis::AxisMask;
pub use face::{faces, Face, FaceIter, FaceMask};
pub use index::{CartesianIter, CartesianWindowIter, IndexSpace, IndexWindow};
pub use rectangle::Rectangle;
pub use region::{regions, Region, RegionIter, Side};
pub use tree::{
    Tree, TreeBlockNeighbor, TreeBlocks, TreeInterface, TreeInterfaces, TreeNeighbors, TreeNodes,
    NULL,
};
