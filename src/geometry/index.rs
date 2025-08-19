#![allow(clippy::needless_range_loop)]

use std::array;

use super::{Face, Region, Side};

/// Describes an abstract index space. Allows for iteration of indices
/// in N dimensions, and transformations between cartesian and linear
/// indices.
#[derive(Debug, Clone, Copy)]
pub struct IndexSpace<const N: usize> {
    size: [usize; N],
}

impl<const N: usize> IndexSpace<N> {
    /// Constructs a new index space.
    pub const fn new(size: [usize; N]) -> Self {
        Self { size }
    }

    /// Returns the number of indices in the index space.
    pub fn index_count(&self) -> usize {
        let mut result = 1;

        for i in 0..N {
            result *= self.size[i]
        }

        result
    }

    /// Returns the dimensions of the index space along each axis.
    pub fn size(self) -> [usize; N] {
        self.size
    }

    /// Converts a linear index into a cartesian index. This
    /// will likely be an order of magnitude slower than
    /// `linear_from_cartesian()` due to several
    /// modulus operations.
    pub fn cartesian_from_linear(self, mut linear: usize) -> [usize; N] {
        debug_assert!(linear < self.size.iter().product());

        let mut result = [0; N];

        for i in 0..N {
            result[i] = linear % self.size[i];
            linear /= self.size[i];
        }

        result
    }

    /// Converts a cartesian index into a linear index.
    pub fn linear_from_cartesian(self, cartesian: [usize; N]) -> usize {
        for axis in 0..N {
            debug_assert!(cartesian[axis] < self.size[axis]);
        }

        let mut result = 0;
        let mut stride = 1;

        for i in 0..N {
            result += stride * cartesian[i];
            stride *= self.size[i];
        }

        result
    }

    /// Iterates all cartesian indices in the index space.
    pub const fn iter(self) -> CartesianIter<N> {
        CartesianIter {
            size: self.size,
            cursor: [0; N],
        }
    }

    /// Returns an index window corresponding to the entire IndexSpace.
    pub fn window(self) -> IndexWindow<N> {
        IndexWindow {
            origin: [0; N],
            size: self.size,
        }
    }

    /// Returns the window containing all points along a plane in the index space.
    pub fn plane_window(self, axis: usize, intercept: usize) -> IndexWindow<N> {
        debug_assert!(intercept < self.size[axis]);

        let mut origin = [0; N];
        origin[axis] = intercept;

        let mut size = self.size;
        size[axis] = 1;

        IndexWindow::new(origin, size)
    }

    /// Returns the window containing all points along a face in index space.
    pub fn face_window(self, face: Face<N>) -> IndexWindow<N> {
        let intercept = if face.side {
            self.size[face.axis] - 1
        } else {
            0
        };
        self.plane_window(face.axis, intercept)
    }

    /// The window of all indices that border the given region.
    pub fn region_adjacent_window(self, region: Region<N>) -> IndexWindow<N> {
        let origin = array::from_fn(|axis| match region.side(axis) {
            Side::Left | Side::Middle => 0,
            Side::Right => self.size[axis] - 1,
        });

        let size = array::from_fn(|axis| match region.side(axis) {
            Side::Left | Side::Right => 1,
            Side::Middle => self.size[axis],
        });

        IndexWindow { origin, size }
    }
}

impl<const N: usize> IntoIterator for IndexSpace<N> {
    type IntoIter = CartesianIter<N>;
    type Item = [usize; N];

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Represents a subset of an index space, and provides utilities for iterating over this window.
#[derive(Debug, Clone, Copy)]
pub struct IndexWindow<const N: usize> {
    /// Stores the origin (bottom-left corner) of the index window
    pub origin: [usize; N],
    /// Stores the size along each axis of the index window.
    pub size: [usize; N],
}

impl<const N: usize> IndexWindow<N> {
    /// Constructs a new index window.
    pub fn new(origin: [usize; N], size: [usize; N]) -> Self {
        Self { origin, size }
    }

    /// Iterates over indices in the index window.
    pub fn iter(&self) -> CartesianWindowIter<N> {
        CartesianWindowIter {
            origin: self.origin,
            inner: IndexSpace::new(self.size).iter(),
        }
    }
}

impl<const N: usize> IntoIterator for IndexWindow<N> {
    type IntoIter = CartesianWindowIter<N>;
    type Item = [usize; N];

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Debug, Clone)]
/// An iterator over the cartesian indices of an `IndexSpace`.
pub struct CartesianIter<const N: usize> {
    size: [usize; N],
    cursor: [usize; N],
}

impl<const N: usize> Iterator for CartesianIter<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        // Last index was incremented, iteration is complete
        if self.cursor[N - 1] == self.size[N - 1] {
            return None;
        }

        // Store current cursor value (this is what we will return)
        let result = self.cursor;

        for i in 0..N {
            if self.size[i] == 0 {
                // Short circuit if any of the dimensions are zero.
                return None;
            }

            // If we need to increment this axis, we add to the cursor value
            self.cursor[i] += 1;
            // If the cursor is equal to size, we wrap.
            // However, if we have reached the final axis,
            // this indicates we are at the end of iteration,
            // and will return None on the next call of next().
            if self.cursor[i] == self.size[i] && i < N - 1 {
                self.cursor[i] = 0;
                // Continue looping over axes
                continue;
            }

            break;
        }

        Some(result)
    }
}

#[derive(Debug, Clone)]
/// An iterator over the cartesian indices of an `IndexSpace`.
pub struct CartesianWindowIter<const N: usize> {
    origin: [usize; N],
    inner: CartesianIter<N>,
}

impl<const N: usize> Iterator for CartesianWindowIter<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let offset = self.inner.next()?;
        Some(array::from_fn(|i| self.origin[i] + offset[i]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_iteration() {
        let space = IndexSpace::new([3, 2]);
        let mut indices = space.iter();

        assert_eq!(indices.next(), Some([0, 0]));
        assert_eq!(indices.next(), Some([1, 0]));
        assert_eq!(indices.next(), Some([2, 0]));
        assert_eq!(indices.next(), Some([0, 1]));
        assert_eq!(indices.next(), Some([1, 1]));
        assert_eq!(indices.next(), Some([2, 1]));
        assert_eq!(indices.next(), None);

        let space = IndexSpace::new([0, 10]);
        let mut indices = space.iter();

        assert_eq!(indices.next(), None);

        let space = IndexSpace::new([2, 3, 10]);
        let mut plane = space.plane_window(2, 5).iter();

        assert_eq!(plane.next(), Some([0, 0, 5]));
        assert_eq!(plane.next(), Some([1, 0, 5]));
        assert_eq!(plane.next(), Some([0, 1, 5]));
        assert_eq!(plane.next(), Some([1, 1, 5]));
        assert_eq!(plane.next(), Some([0, 2, 5]));
        assert_eq!(plane.next(), Some([1, 2, 5]));
        assert_eq!(plane.next(), None);

        let space = IndexSpace::new([2, 3, 4]);
        for (i, index) in space.iter().enumerate() {
            assert_eq!(i, space.linear_from_cartesian(index));
        }
    }

    #[test]
    fn index_conversion() {
        let space = IndexSpace::new([2, 4, 3]);

        assert_eq!(space.linear_from_cartesian([1, 0, 0]), 1);
        assert_eq!(space.linear_from_cartesian([0, 1, 0]), 2);
        assert_eq!(space.linear_from_cartesian([0, 0, 2]), 8 * 2);
        assert_eq!(space.linear_from_cartesian([1, 1, 2]), 8 * 2 + 2 + 1);

        assert_eq!(space.cartesian_from_linear(1), [1, 0, 0]);
        assert_eq!(space.cartesian_from_linear(2), [0, 1, 0]);
        assert_eq!(space.cartesian_from_linear(8 * 2), [0, 0, 2]);
        assert_eq!(space.cartesian_from_linear(8 * 2 + 2 + 1), [1, 1, 2]);
    }
}
