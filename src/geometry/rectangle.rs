use serde::{Deserialize, Serialize};
use std::array::from_fn;

use crate::geometry::AxisMask;

/// Represents a rectangular physical domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rectangle<const N: usize> {
    /// Size of the rectangle along each axis.
    #[serde(with = "aeon_array")]
    pub size: [f64; N],
    /// Origin of the rectangle (located at the bottom-left corner).
    #[serde(with = "aeon_array")]
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
                self.origin[i] + self.size[i] / 2.0
            } else {
                self.origin[i]
            }
        });

        Self { size, origin }
    }
}
