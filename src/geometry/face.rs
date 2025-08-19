use std::{
    array::from_fn,
    fmt::{Display, Write},
    ops::{Index, IndexMut},
};

use crate::array::ArrayWrap;

use super::{Region, Side, Split, index::IndexWindow};

/// A face of a rectangular prism in `N` dimensional space.
/// If `face.side` is true, than this face points in the positive direction along
/// `face.axis`, otherwise it points along the negative direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Face<const N: usize> {
    pub axis: usize,
    pub side: bool,
}

impl<const N: usize> Face<N> {
    pub fn iterate() -> FaceIter<N> {
        FaceIter {
            axis: 0,
            side: false,
        }
    }

    /// Face on negative side of axis.
    pub fn negative(axis: usize) -> Self {
        debug_assert!(axis < N);
        Self { axis, side: false }
    }

    /// Face on positive side of axis.
    pub fn positive(axis: usize) -> Self {
        debug_assert!(axis < N);
        Self { axis, side: true }
    }

    /// Reverses the direction of the face along its axis.
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

    /// Finds a split adjacent to the given face (all other axes default to negative).
    pub fn adjacent_split(self) -> Split<N> {
        let mut result = Split::empty();
        result.set_to(self.axis, self.side);
        result
    }

    /// Iterates over all splits adjacent to the given face.
    pub fn adjacent_splits(self) -> impl Iterator<Item = Split<N>> {
        Split::<N>::enumerate().filter(move |split| split.is_set(self.axis) == self.side)
    }
}

const AXIS_NAMES: [char; 4] = ['x', 'y', 'z', 'w'];

impl<const N: usize> Display for Face<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.side {
            f.write_char('+')?;
        } else {
            f.write_char('-')?;
        }

        if N < 4 {
            f.write_char(AXIS_NAMES[self.axis])
        } else {
            f.write_str(&self.axis.to_string())
        }
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

/// An array storing a value for each `Face<N>` in a N-dimensional space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceArray<const N: usize, T>([[T; 2]; N]);

impl<const N: usize, T> FaceArray<N, T> {
    /// Constructs a `FaceArray<N>` by calling `f` for each `Face<N>`.
    pub fn from_fn<F: FnMut(Face<N>) -> T>(mut f: F) -> Self {
        Self(core::array::from_fn(|axis| {
            [f(Face::negative(axis)), f(Face::positive(axis))]
        }))
    }

    /// Retrieves the inner representation of `FaceArray`, i.e. an array of type
    /// `[[T; 2]; N]` where the first index is axis and the second index is size.
    pub fn into_inner(self) -> [[T; 2]; N] {
        self.0
    }
}

impl<const N: usize, T: Clone> FaceArray<N, T> {
    /// Constructs a `FaceArray` by filling the whole array with `value`.
    pub fn splat(value: T) -> Self {
        Self::from_fn(|_| value.clone())
    }

    pub fn from_sides(negative: [T; N], positive: [T; N]) -> Self {
        Self::from_fn(|face| match face.side {
            true => positive[face.axis].clone(),
            false => negative[face.axis].clone(),
        })
    }
}

impl<const N: usize, T> From<[[T; 2]; N]> for FaceArray<N, T> {
    fn from(value: [[T; 2]; N]) -> Self {
        Self(value)
    }
}

impl<const N: usize, T> From<[(T, T); N]> for FaceArray<N, T> {
    fn from(value: [(T, T); N]) -> Self {
        Self(value.map(|(l, r)| [l, r]))
    }
}

impl<const N: usize, T: serde::Serialize + Clone> serde::Serialize for FaceArray<N, T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        ArrayWrap(self.0.clone()).serialize(serializer)
    }
}

impl<'de, const N: usize, T: serde::de::Deserialize<'de>> serde::de::Deserialize<'de>
    for FaceArray<N, T>
where
    T: serde::de::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(FaceArray(ArrayWrap::deserialize(deserializer)?.0))
    }
}

impl<const N: usize, T: Default> Default for FaceArray<N, T> {
    fn default() -> Self {
        Self::from_fn(|_| T::default())
    }
}

impl<const N: usize, T> Index<Face<N>> for FaceArray<N, T> {
    type Output = T;
    fn index(&self, index: Face<N>) -> &Self::Output {
        &self.0[index.axis][index.side as usize]
    }
}

impl<const N: usize, T> IndexMut<Face<N>> for FaceArray<N, T> {
    fn index_mut(&mut self, index: Face<N>) -> &mut Self::Output {
        &mut self.0[index.axis][index.side as usize]
    }
}

/// Stores a boolean flag for each face of a rectangular prism.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceMask<const N: usize>([[bool; 2]; N]);

impl<const N: usize> FaceMask<N> {
    pub fn from_fn<F: FnMut(Face<N>) -> bool>(mut f: F) -> Self {
        Self(core::array::from_fn(|axis| {
            [f(Face::negative(axis)), f(Face::positive(axis))]
        }))
    }

    pub fn pack(bits: [[bool; 2]; N]) -> Self {
        Self(bits)
    }

    /// Constructs a mask where all flags have been set to false.
    pub fn empty() -> Self {
        Self([[false; 2]; N])
    }

    /// Constructs a mask where all flags have been set to true.
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

    /// Iterates over regions adjacent to flagged faces.
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

impl<const N: usize> Default for FaceMask<N> {
    fn default() -> Self {
        Self::from_fn(|_| false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_iteration() {
        let mut list = Face::<3>::iterate();
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

    #[test]
    fn adjacent_regions() {
        let mut mask = FaceMask::<2>::empty();
        mask.set(Face::positive(0));
        mask.set(Face::negative(1));

        let mut regions = mask.adjacent_regions();

        assert_eq!(
            regions.next(),
            Some(Region::new([Side::Middle, Side::Left]))
        );
        assert_eq!(regions.next(), Some(Region::new([Side::Right, Side::Left])));
        assert_eq!(
            regions.next(),
            Some(Region::new([Side::Middle, Side::Middle]))
        );
        assert_eq!(
            regions.next(),
            Some(Region::new([Side::Right, Side::Middle]))
        );
        assert_eq!(regions.next(), None);
    }
}
