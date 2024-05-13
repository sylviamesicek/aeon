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

use serde::{Deserialize, Serialize};
use std::array::from_fn;

use crate::array::Array;

/// Represents a rectangular physical domain.
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

    /// Computes center of rectangle.
    pub fn center(&self) -> [f64; N] {
        from_fn(|i| self.origin[i] + self.size[i] / 2.0)
    }
}

impl<const N: usize> Serialize for Rectangle<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        RectangleSerde {
            size: self.size.clone().into(),
            origin: self.origin.clone().into(),
        }
        .serialize(serializer)
    }
}

impl<'de, const N: usize> Deserialize<'de> for Rectangle<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let rect = RectangleSerde::deserialize(deserializer)?;

        Ok(Self {
            size: rect.size.inner(),
            origin: rect.origin.inner(),
        })
    }
}

#[derive(Serialize, Deserialize)]
struct RectangleSerde<const N: usize> {
    size: Array<[f64; N]>,
    origin: Array<[f64; N]>,
}
