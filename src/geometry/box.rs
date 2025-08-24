use std::array::{self, from_fn};

use datasize::DataSize;

use crate::geometry::Split;

/// Represents a rectangular physical domain.
#[derive(Debug, Copy, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HyperBox<const N: usize> {
    /// Size of the rectangle along each axis.
    #[serde(with = "crate::array")]
    pub size: [f64; N],
    /// Origin of the rectangle (located at the bottom-left corner).
    #[serde(with = "crate::array")]
    pub origin: [f64; N],
}

impl<const N: usize> DataSize for HyperBox<N> {
    const IS_DYNAMIC: bool = false;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        0
    }
}

impl<const N: usize> HyperBox<N> {
    /// Unit rectangle.
    pub const UNIT: Self = HyperBox {
        size: [1.0; N],
        origin: [0.0; N],
    };

    /// Constructs a rectangle from an aabb.
    pub fn from_aabb(aa: [f64; N], bb: [f64; N]) -> Self {
        let size = array::from_fn(|axis| (bb[axis] - aa[axis]).max(0.0));
        Self { size, origin: aa }
    }

    /// Computes the center of the rectangle.
    pub fn center(&self) -> [f64; N] {
        array::from_fn(|i| self.origin[i] + self.size[i] / 2.0)
    }

    /// Returns true if the rectangle contains a point.
    pub fn contains(&self, point: [f64; N]) -> bool {
        for axis in 0..N {
            if point[axis] < self.origin[axis] || point[axis] > self.origin[axis] + self.size[axis]
            {
                return false;
            }
        }

        true
    }

    /// Subdivides the rectangle.
    pub fn subdivide(&self, split: Split<N>) -> Self {
        let size = array::from_fn(|i| self.size[i] / 2.0);
        let origin = array::from_fn(|i| {
            if split.is_set(i) {
                self.origin[i] + self.size[i] / 2.0
            } else {
                self.origin[i]
            }
        });

        Self { size, origin }
    }

    pub fn aa(&self) -> [f64; N] {
        self.origin
    }

    pub fn bb(&self) -> [f64; N] {
        from_fn(|i| self.origin[i] + self.size[i])
    }

    pub fn local_to_global(&self, local: [f64; N]) -> [f64; N] {
        array::from_fn(|axis| local[axis] * self.size[axis] + self.origin[axis])
    }

    pub fn global_to_local(&self, global: [f64; N]) -> [f64; N] {
        array::from_fn(|axis| (global[axis] - self.origin[axis]) / self.size[axis])
    }
}
