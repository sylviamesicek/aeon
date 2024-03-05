mod boundary;
mod kernel;

pub use boundary::*;
pub use kernel::*;

use crate::geometry::{IndexSpace, Rectangle};

pub struct NodeSpace<const N: usize> {
    pub size: [usize; N],
    pub bounds: Rectangle<N>,
}

impl<const N: usize> NodeSpace<N> {
    pub fn total(self: &Self) -> usize {
        let mut result = 1;

        for i in 0..N {
            result *= self.size[i];
        }

        result
    }

    pub fn spacing(self: &Self, axis: usize) -> f64 {
        self.bounds.size[axis] / (self.size[axis] - 1) as f64
    }

    pub fn value(self: &Self, node: [usize; N], src: &[f64]) -> f64 {
        let linear = IndexSpace::new(self.size).linear_from_cartesian(node);
        src[linear]
    }

    pub fn set_value(self: &Self, node: [usize; N], v: f64, dest: &mut [f64]) {
        let linear = IndexSpace::new(self.size).linear_from_cartesian(node);
        dest[linear] = v;
    }

    pub fn evaluate<K: Kernel>(self: &Self, axis: usize, src: &[f64], dest: &mut [f64]) {
        assert!(src.len() == self.total() && dest.len() == self.total());

        // Get spacing along axis for covariant transformation
        let spacing = self.spacing(axis);
        let scale = K::scale(spacing);

        let length = self.size[axis];

        // let negative_edge = K::Stencil::NEGATIVE;
        let negative_boundary = 0;
        let positive_boundary = length - 1;
        let positive_edge = length - K::Stencil::EDGE - 1;

        // Loop over plane normal to axis
        for mut node in IndexSpace::new(self.size).plane(axis, 0) {
            // Fill negative boundary
            for left in 0..K::Stencil::NEGATIVE {
                let mut result = 0.0;

                for (i, w) in K::negative(left).into_iter().enumerate() {
                    node[axis] = i;
                    result += self.value(node, src) * w;
                }

                node[axis] = negative_boundary + left;
                self.set_value(node, scale * result, dest);
            }

            // Fill positive boundary
            for right in 0..K::Stencil::POSITIVE {
                let mut result = 0.0;

                for (i, w) in K::positive(right).into_iter().enumerate() {
                    node[axis] = positive_edge + i;
                    result += self.value(node, src) * w;
                }

                node[axis] = positive_boundary - right;
                self.set_value(node, scale * result, dest);
            }

            // Fill interior
            for middle in K::Stencil::NEGATIVE..(length - K::Stencil::POSITIVE) {
                let mut result = 0.0;

                for (i, w) in K::interior().into_iter().enumerate() {
                    node[axis] = middle - K::Stencil::NEGATIVE + i;
                    result += self.value(node, src) * w;
                }

                node[axis] = middle;
                self.set_value(node, scale * result, dest);
            }
        }
    }
}
