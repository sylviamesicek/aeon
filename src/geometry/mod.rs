//! A module for various geometric primatives (AABBs, working with cartesian indices, etc.).

mod axis;
mod index_space;
mod region;

pub use axis::AxisMask;
pub use index_space::{CartesianIter, IndexSpace, PlaneIterator};
pub use region::{
    faces, regions, Face, FaceIter, Region, RegionFaceNodeIter, RegionIter, RegionNodeIter,
    RegionOffsetNodeIter,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Rectangle<const N: usize> {
    pub size: [f64; N],
    pub origin: [f64; N],
}

impl<const N: usize> Rectangle<N> {
    /// Unit rectangle.
    pub const UNIT: Self = Rectangle {
        size: [1.0; N],
        origin: [0.0; N],
    };
}
