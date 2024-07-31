use std::array::from_fn;

use super::{index::IndexWindow, AxisMask, Region, Side};

/// A face of a rectangular prism in `N` dimensional space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Face<const N: usize> {
    pub axis: usize,
    pub side: bool,
}

impl<const N: usize> Face<N> {
    /// Face on negative side of axis.
    pub fn negative(axis: usize) -> Self {
        assert!(axis < N);
        Self { axis, side: false }
    }

    /// Face on positive side of axis.
    pub fn positive(axis: usize) -> Self {
        assert!(axis < N);
        Self { axis, side: true }
    }

    pub fn reversed(self) -> Self {
        Self {
            axis: self.axis,
            side: !self.side,
        }
    }

    /// Transforms a face into a linear index.
    pub fn to_linear(self) -> usize {
        2 * self.axis + self.side as usize
    }

    /// Constructs a face from a linear index.
    pub fn from_linear(linear: usize) -> Self {
        assert!(linear < 2 * N);

        Self {
            axis: linear / 2,
            side: linear % 2 == 1,
        }
    }

    pub fn adjacent_split(self) -> AxisMask<N> {
        let mut result = AxisMask::empty();
        result.set_to(self.axis, self.side);
        result
    }

    pub fn adjacent_splits(self) -> impl Iterator<Item = AxisMask<N>> {
        AxisMask::<N>::enumerate().filter(move |split| split.is_set(self.axis) == self.side)
    }
}

/// Iterator over all faces in a given number of dimensions.
#[derive(Debug)]
pub struct FaceIter<const N: usize> {
    axis: usize,
    side: bool,
}

impl<const N: usize> Iterator for FaceIter<N> {
    type Item = Face<N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.axis >= N {
            return None;
        }

        let result = Face {
            axis: self.axis,
            side: self.side,
        };

        self.axis += self.side as usize;
        self.side = !self.side;

        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (2 * N, Some(2 * N))
    }
}

impl<const N: usize> ExactSizeIterator for FaceIter<N> {
    fn len(&self) -> usize {
        2 * N
    }
}

/// Iterates over all faces in a given number of dimensions.
pub fn faces<const N: usize>() -> FaceIter<N> {
    FaceIter {
        axis: 0,
        side: false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceMask<const N: usize>([[bool; 2]; N]);

impl<const N: usize> FaceMask<N> {
    pub fn pack(bits: [[bool; 2]; N]) -> Self {
        Self(bits)
    }

    pub fn empty() -> Self {
        Self([[false; 2]; N])
    }

    pub fn full() -> Self {
        Self([[true; 2]; N])
    }

    pub fn is_set(&self, face: Face<N>) -> bool {
        self.0[face.axis][face.side as usize]
    }

    pub fn set(&mut self, face: Face<N>) {
        self.0[face.axis][face.side as usize] = true;
    }

    pub fn clear(&mut self, face: Face<N>) {
        self.0[face.axis][face.side as usize] = false;
    }

    pub fn set_to(&mut self, face: Face<N>, val: bool) {
        self.0[face.axis][face.side as usize] = val;
    }

    pub fn adjacent_regions(&self) -> impl Iterator<Item = Region<N>> {
        let mut window = IndexWindow::new([1; N], [1; N]);

        for axis in 0..N {
            if self.is_set(Face::negative(axis)) {
                window.origin[axis] -= 1;
                window.size[axis] += 1;
            }

            if self.is_set(Face::positive(axis)) {
                window.size[axis] += 1;
            }
        }

        window
            .iter()
            .map(|index| Region::new(from_fn(|axis| Side::from_value(index[axis] as u8))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_iteration() {
        let mut list = faces::<3>();
        assert_eq!(list.next(), Some(Face::negative(0)));
        assert_eq!(list.next(), Some(Face::positive(0)));
        assert_eq!(list.next(), Some(Face::negative(1)));
        assert_eq!(list.next(), Some(Face::positive(1)));
        assert_eq!(list.next(), Some(Face::negative(2)));
        assert_eq!(list.next(), Some(Face::positive(2)));
        assert_eq!(list.next(), None);

        assert_eq!(Face::<4>::negative(1).to_linear(), 2);
        assert_eq!(Face::<4>::positive(3).to_linear(), 7);
        assert_eq!(Face::<4>::positive(3), Face::<4>::from_linear(7));
    }
}
