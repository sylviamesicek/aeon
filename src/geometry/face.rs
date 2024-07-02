/// A face of a rectangular prism in `N` dimensional space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Face {
    pub axis: usize,
    pub side: bool,
}

impl Face {
    /// Face on negative side of axis.
    pub fn negative(axis: usize) -> Self {
        Self { axis, side: false }
    }

    /// Face on positive side of axis.
    pub fn positive(axis: usize) -> Self {
        Self { axis, side: true }
    }

    /// Transforms a face into a linear index.
    pub fn to_linear(self) -> usize {
        2 * self.axis + self.side as usize
    }

    /// Constructs a face from a linear index.
    pub fn from_linear(linear: usize) -> Self {
        Self {
            axis: linear / 2,
            side: linear % 2 == 1,
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
    type Item = Face;

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

#[derive(Clone, Copy, Debug)]
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

    pub fn is_set(&self, face: Face) -> bool {
        self.0[face.axis][face.side as usize]
    }

    pub fn set(&mut self, face: Face) {
        self.0[face.axis][face.side as usize] = true;
    }

    pub fn clear(&mut self, face: Face) {
        self.0[face.axis][face.side as usize] = false;
    }

    pub fn set_to(&mut self, face: Face, val: bool) {
        self.0[face.axis][face.side as usize] = val;
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

        assert_eq!(Face::negative(1).to_linear(), 2);
        assert_eq!(Face::positive(3).to_linear(), 7);
        assert_eq!(Face::positive(3), Face::from_linear(7));
    }
}
