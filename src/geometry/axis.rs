use crate::geometry::{faces, Face};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AxisMask<const N: usize>(usize);

impl<const N: usize> AxisMask<N> {
    pub const COUNT: usize = 2usize.pow(N as u32);

    pub const fn enumerate() -> AxisMaskIter<N> {
        AxisMaskIter { cursor: 0 }
    }

    pub fn empty() -> Self {
        Self(0)
    }

    pub fn full() -> Self {
        Self(usize::MAX)
    }

    pub fn from_linear(linear: usize) -> Self {
        Self(linear)
    }

    pub fn to_linear(self) -> usize {
        self.0.min(Self::COUNT - 1)
    }

    pub fn unpack(self) -> [bool; N] {
        let mut result = [false; N];

        for axis in 0..N {
            result[axis] = self.is_set(axis);
        }

        result
    }

    pub fn pack(bits: [bool; N]) -> Self {
        let mut result = Self::empty();

        for (i, bit) in bits.into_iter().enumerate() {
            result.set_to(i, bit);
        }

        result
    }

    pub fn set(&mut self, axis: usize) {
        self.0 |= 1 << axis
    }

    pub fn clear(&mut self, axis: usize) {
        self.0 &= !(1 << axis)
    }

    pub fn set_to(&mut self, axis: usize, value: bool) {
        self.0 &= !(1 << axis);
        self.0 |= (value as usize) << axis;
    }

    pub fn toggle(&mut self, axis: usize) {
        self.0 ^= 1 << axis
    }

    pub fn toggled(mut self, axis: usize) -> Self {
        self.0 ^= 1 << axis;
        self
    }

    pub fn is_set(self, axis: usize) -> bool {
        (self.0 & (1 << axis)) != 0
    }

    pub fn is_inner_face(self, face: Face) -> bool {
        self.is_set(face.axis) != face.side
    }

    pub fn is_outer_face(self, face: Face) -> bool {
        self.is_set(face.axis) == face.side
    }

    pub fn inner_faces(self) -> impl Iterator<Item = Face> {
        faces::<N>().filter(move |&face| self.is_inner_face(face))
    }

    pub fn outer_faces(self) -> impl Iterator<Item = Face> {
        faces::<N>().filter(move |&face| self.is_outer_face(face))
    }

    pub fn is_compatible_with_face(self, face: Face) -> bool {
        self.is_set(face.axis) == face.side
    }
}

pub struct AxisMaskIter<const N: usize> {
    cursor: usize,
}

impl<const N: usize> Iterator for AxisMaskIter<N> {
    type Item = AxisMask<N>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= AxisMask::<N>::COUNT {
            return None;
        }

        let result = self.cursor;
        self.cursor += 1;
        Some(AxisMask::from_linear(result))
    }
}
