//! A module for various geometric primatives (AABBs, working with cartesian indices, etc.).

mod axis;
mod face;
mod indices;
mod region;

pub use axis::AxisMask;
pub use face::{faces, Face, FaceIter};
pub use indices::{CartesianIter, IndexSpace, PlaneIterator};
pub use region::{
    regions, Region, RegionFaceNodeIter, RegionIter, RegionNodeIter, RegionOffsetNodeIter,
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
