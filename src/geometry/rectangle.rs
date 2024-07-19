use serde::{Deserialize, Serialize};
use std::array::from_fn;

use crate::array::Array;
use crate::geometry::AxisMask;

/// Represents a rectangular physical domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(from = "RectangleSerde<N>")]
#[serde(into = "RectangleSerde<N>")]
pub struct Rectangle<const N: usize> {
    /// Size of the rectangle along each axis.
    pub size: [f64; N],
    /// Origin of the rectangle (located at the bottom-left corner).
    pub origin: [f64; N],
}

impl<const N: usize> Rectangle<N> {
    /// Unit rectangle.
    pub const UNIT: Self = Rectangle {
        size: [1.0; N],
        origin: [0.0; N],
    };

    /// Computes the center of the rectangle.
    pub fn center(&self) -> [f64; N] {
        from_fn(|i| self.origin[i] + self.size[i] / 2.0)
    }

    /// Returns the subdivision
    pub fn split(&self, mask: AxisMask<N>) -> Self {
        let size = from_fn(|i| self.size[i] / 2.0);
        let origin = from_fn(|i| {
            if mask.is_set(i) {
                self.origin[i] + size[i] / 2.0
            } else {
                self.origin[i]
            }
        });

        Self { size, origin }
    }
}

#[derive(Serialize, Deserialize)]
struct RectangleSerde<const N: usize> {
    size: Array<[f64; N]>,
    origin: Array<[f64; N]>,
}

impl<const N: usize> From<Rectangle<N>> for RectangleSerde<N> {
    fn from(value: Rectangle<N>) -> Self {
        Self {
            size: value.size.into(),
            origin: value.origin.into(),
        }
    }
}

impl<const N: usize> From<RectangleSerde<N>> for Rectangle<N> {
    fn from(value: RectangleSerde<N>) -> Self {
        Self {
            size: value.size.inner(),
            origin: value.origin.inner(),
        }
    }
}
