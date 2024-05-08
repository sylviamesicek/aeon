/// A face of a rectangular prism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

/// Iterator over all faces in a given number of dimensions.
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
}

/// Iterates over all faces in a given number of dimensions.
pub fn faces<const N: usize>() -> FaceIter<N> {
    FaceIter {
        axis: 0,
        side: false,
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
    }
}
