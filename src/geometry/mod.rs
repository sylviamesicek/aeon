//! A module for various geometric primatives (AABBs, working with cartesian indices, etc.).
//!
//! This includes several classes which handle associating dofs with N-dimensional quadtrees.

#![allow(clippy::needless_range_loop)]

mod axis;
mod r#box;
mod face;
mod index;
mod region;
mod tree;

pub use axis::Split;
pub use r#box::HyperBox;
pub use face::{Face, FaceArray, FaceIter, FaceMask};
pub use index::{CartesianIter, CartesianWindowIter, IndexSpace, IndexWindow};
pub use region::{Region, RegionIter, Side, regions};
pub use tree::{
    ActiveCellId, BlockId, CellId, NeighborId, Tree, TreeBlockNeighbor, TreeBlocks,
    TreeCellNeighbor, TreeInterface, TreeInterfaces, TreeNeighbors, TreeSer,
};
