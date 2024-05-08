#![allow(clippy::needless_range_loop)]

/// Describes an abstract index space. Allows for iteration of indices
/// in N dimensions, and transformations between cartesian and linear
/// indices.
#[derive(Debug, Clone, Copy)]
pub struct IndexSpace<const N: usize> {
    size: [usize; N],
}

impl<const N: usize> IndexSpace<N> {
    /// Constructs a new index space.
    pub fn new(size: [usize; N]) -> Self {
        Self { size }
    }

    pub fn len(&self) -> usize {
        let mut result = 1;

        for i in 0..N {
            result *= self.size[i]
        }

        result
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
        let mut result = [0; N];

        for i in 0..N {
            result[i] = linear % self.size[i];
            linear /= self.size[i];
        }

        result
    }

    /// Converts a cartesian index into a linear index.
    pub fn linear_from_cartesian(self, cartesian: [usize; N]) -> usize {
        let mut result = 0;
        let mut stride = 1;

        for i in 0..N {
            result += stride * cartesian[i];
            stride *= self.size[i];
        }

        result
    }

    pub fn iter(self) -> CartesianIter<N> {
        CartesianIter {
            size: self.size,
            cursor: [0; N],
        }
    }

    pub fn plane(self, axis: usize, intercept: usize) -> PlaneIterator<N> {
        let mut plane_size = self.size;
        plane_size[axis] = 1;

        PlaneIterator {
            inner: Self::new(plane_size).iter(),
            axis,
            intercept,
        }
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

        let mut increment = [false; N];
        increment[0] = true;

        for i in 0..N {
            if increment[i] {
                // If we need to increment this axis, we add to the cursor value
                self.cursor[i] += 1;
                // If the cursor is equal to size, we wrap.
                // However, if we have reached the final axis,
                // this indicates we are at the end of iteration,
                // and will return None on the next call of next().
                if self.cursor[i] == self.size[i] && i < N - 1 {
                    self.cursor[i] = 0;
                    increment[i + 1] = true;
                }
            }
        }

        Some(result)
    }
}

/// An iterator over a plane in an index space.
#[derive(Debug, Clone)]
pub struct PlaneIterator<const N: usize> {
    inner: CartesianIter<N>,
    axis: usize,
    intercept: usize,
}

impl<const N: usize> Iterator for PlaneIterator<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let mut index = self.inner.next()?;
        index[self.axis] = self.intercept;
        Some(index)
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

        let space = IndexSpace::new([2, 3, 10]);
        let mut plane = space.plane(2, 5);

        assert_eq!(plane.next(), Some([0, 0, 5]));
        assert_eq!(plane.next(), Some([1, 0, 5]));
        assert_eq!(plane.next(), Some([0, 1, 5]));
        assert_eq!(plane.next(), Some([1, 1, 5]));
        assert_eq!(plane.next(), Some([0, 2, 5]));
        assert_eq!(plane.next(), Some([1, 2, 5]));
        assert_eq!(plane.next(), None);
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
