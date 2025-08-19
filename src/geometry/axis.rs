use crate::geometry::Face;

use super::{Region, Side};

/// Stores a bitset flag for each axis
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Split<const N: usize>(usize);

impl<const N: usize> Split<N> {
    /// Total permutations of axis maskes for a given dimension.
    pub const COUNT: usize = 2usize.pow(N as u32);

    /// Iterates of all axis maskes of a given dimension.
    pub const fn enumerate() -> SplitIter<N> {
        SplitIter { cursor: 0 }
    }

    /// Constructs an empty axis mask.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Constructs an axis mask with all axes set to true.
    pub const fn full() -> Self {
        Self(usize::MAX)
    }

    /// Converts a linear value into an axis mask.
    pub fn from_linear(linear: usize) -> Self {
        Self(linear)
    }

    /// Transforms this mask into a linear index.
    pub fn to_linear(self) -> usize {
        self.0.min(Self::COUNT - 1)
    }

    /// Transforms an axis mask into an array of bit values.
    pub fn unpack(self) -> [bool; N] {
        let mut result = [false; N];

        for axis in 0..N {
            result[axis] = self.is_set(axis);
        }

        result
    }

    /// Transforms an array of bit values into an axis mask.
    pub fn pack(bits: [bool; N]) -> Self {
        let mut result = Self::empty();

        for (i, bit) in bits.into_iter().enumerate() {
            result.set_to(i, bit);
        }

        result
    }

    /// Sets the given axis.
    pub fn set(&mut self, axis: usize) {
        self.0 |= 1 << axis
    }

    /// Unsets the given axis.
    pub fn clear(&mut self, axis: usize) {
        self.0 &= !(1 << axis)
    }

    /// Sets the axis to the given value.
    pub fn set_to(&mut self, axis: usize, value: bool) {
        self.0 &= !(1 << axis);
        self.0 |= (value as usize) << axis;
    }

    /// Toggles an axis.
    pub fn toggle(&mut self, axis: usize) {
        self.0 ^= 1 << axis
    }

    /// Returns a new axis mask with the given axis toggled.
    pub fn toggled(mut self, axis: usize) -> Self {
        self.0 ^= 1 << axis;
        self
    }

    /// Checks if the given axis is set.
    pub fn is_set(self, axis: usize) -> bool {
        (self.0 & (1 << axis)) != 0
    }

    pub fn is_inner_face(self, face: Face<N>) -> bool {
        self.is_set(face.axis) != face.side
    }

    pub fn is_outer_face(self, face: Face<N>) -> bool {
        self.is_set(face.axis) == face.side
    }

    pub fn inner_faces(self) -> impl Iterator<Item = Face<N>> {
        Face::<N>::iterate().filter(move |&face| self.is_inner_face(face))
    }

    pub fn outer_faces(self) -> impl Iterator<Item = Face<N>> {
        Face::<N>::iterate().filter(move |&face| self.is_outer_face(face))
    }

    pub fn as_inner_face(mut self, face: Face<N>) -> Self {
        self.set_to(face.axis, !face.side);
        self
    }

    pub fn as_outer_face(mut self, face: Face<N>) -> Self {
        self.set_to(face.axis, face.side);
        self
    }

    pub fn is_inner_region(self, region: Region<N>) -> bool {
        for axis in 0..N {
            match (region.side(axis), self.is_set(axis)) {
                (Side::Left, false) => return false,
                (Side::Right, true) => return false,
                _ => {}
            }
        }

        true
    }

    pub fn is_outer_region(self, region: Region<N>) -> bool {
        !self.is_inner_region(region)
    }

    pub fn as_inner_region(mut self, region: Region<N>) -> Self {
        for axis in 0..N {
            match region.side(axis) {
                Side::Left => self.set_to(axis, true),
                Side::Right => self.set_to(axis, false),
                _ => {}
            }
        }

        self
    }

    pub fn as_outer_region(mut self, region: Region<N>) -> Self {
        for axis in 0..N {
            match region.side(axis) {
                Side::Left => self.set_to(axis, false),
                Side::Right => self.set_to(axis, true),
                _ => {}
            }
        }

        self
    }
}

impl<const N: usize> datasize::DataSize for Split<N> {
    const IS_DYNAMIC: bool = false;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        0
    }
}

/// Iterates over all possible axis masks for a given dimension.
pub struct SplitIter<const N: usize> {
    cursor: usize,
}

impl<const N: usize> Iterator for SplitIter<N> {
    type Item = Split<N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= Split::<N>::COUNT {
            return None;
        }

        let result = self.cursor;
        self.cursor += 1;
        Some(Split::from_linear(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masks() {
        let mut splits = Split::<2>::enumerate();
        assert_eq!(splits.next(), Some(Split::pack([false, false])));
        assert_eq!(splits.next(), Some(Split::pack([true, false])));
        assert_eq!(splits.next(), Some(Split::pack([false, true])));
        assert_eq!(splits.next(), Some(Split::pack([true, true])));
        assert_eq!(splits.next(), None);

        let mut faces = Split::<2>::pack([false, true]).outer_faces();
        assert_eq!(faces.next(), Some(Face::<2>::negative(0)));
        assert_eq!(faces.next(), Some(Face::<2>::positive(1)));
        assert_eq!(faces.next(), None);
    }
}
