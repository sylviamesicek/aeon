use std::{
    array::from_fn,
    cmp::Ordering,
    fmt::{Display, Write},
};

use super::{index::IndexWindow, Split, CartesianIter, Face, IndexSpace};

/// Denotes where the region falls on a certain axis.
#[repr(u8)]
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left = 0,
    Middle = 1,
    Right = 2,
}

impl Side {
    pub fn reverse(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
            Self::Middle => Self::Middle,
        }
    }

    pub fn from_value(val: u8) -> Self {
        assert!(val < 3);
        // Safety. We have specified the memory representation of the
        // enum and checked the value, so this should be safe.
        unsafe { std::mem::transmute(val) }
    }
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region<const N: usize> {
    #[serde(with = "crate::array")]
    sides: [Side; N],
}

impl<const N: usize> Region<N> {
    /// Number of different regions in a given number of dimensions.
    pub const COUNT: usize = 3usize.pow(N as u32);
    /// The default "central" region.
    pub const CENTRAL: Self = Self::new([Side::Middle; N]);

    // Builds a new region from the given sides.
    pub const fn new(sides: [Side; N]) -> Self {
        Self { sides }
    }

    pub const fn sides(&self) -> [Side; N] {
        self.sides
    }

    pub const fn side(&self, axis: usize) -> Side {
        self.sides[axis]
    }

    pub fn set_side(&mut self, axis: usize, side: Side) {
        self.sides[axis] = side
    }

    /// Reverse every side in the region.
    pub fn reverse(&self) -> Self {
        let mut result = [Side::Left; N];

        for axis in 0..N {
            result[axis] = self.sides[axis].reverse();
        }

        Self::new(result)
    }

    /// Returns number of axes that are not `Side::Middle`.
    pub fn adjacency(&self) -> usize {
        self.sides
            .into_iter()
            .filter(|&s| s != Side::Middle)
            .count()
    }

    /// Iterates over all faces one would have to move over to get to the region.
    pub fn adjacent_faces(&self) -> impl Iterator<Item = Face<N>> + '_ {
        (0..N)
            .filter(|&axis| self.side(axis) != Side::Middle)
            .map(|axis| Face {
                axis,
                side: self.side(axis) == Side::Right,
            })
    }

    /// Iterates over all splits that can touch this region.
    pub fn adjacent_splits(self) -> impl Iterator<Item = Split<N>> {
        let origin: [_; N] = from_fn(|axis| match self.side(axis) {
            Side::Left | Side::Middle => 0,
            Side::Right => 1,
        });

        let size: [_; N] = from_fn(|axis| match self.side(axis) {
            Side::Middle => 2,
            _ => 1,
        });

        IndexWindow::new(origin, size)
            .iter()
            .map(|index| Split::pack(from_fn(|axis| index[axis] != 0)))
    }

    /// Computes a split which touches the given region.
    pub fn adjacent_split(&self) -> Split<N> {
        let mut result = Split::empty();
        for axis in 0..N {
            result.set_to(axis, self.side(axis) == Side::Right)
        }
        result
    }

    /// Checks whether a given split is adjacent to the region.
    pub fn is_split_adjacent(&self, split: Split<N>) -> bool {
        for axis in 0..N {
            match (self.side(axis), split.is_set(axis)) {
                (Side::Left, true) => return false,
                (Side::Right, false) => return false,
                _ => {}
            }
        }

        true
    }

    // /// Returns an index space with the same size as the region.
    // pub fn index_space(&self, support: usize, block: [usize; N]) -> IndexSpace<N> {
    //     let mut size = [0; N];

    //     for axis in 0..N {
    //         if self.sides[axis] != Side::Middle {
    //             size[axis] = support;
    //         } else {
    //             size[axis] = block[axis]
    //         }
    //     }

    //     IndexSpace::new(size)
    // }

    // /// Iterates nodes in the given region (including ghost nodes and nodes
    // /// on faces).
    // pub fn nodes(&self, support: usize, block: [usize; N]) -> RegionNodeIter<N> {
    //     RegionNodeIter {
    //         inner: self.index_space(support, block).iter(),
    //         block,
    //         sides: self.sides,
    //     }
    // }

    // pub fn face_vertices(&self, block: [usize; N]) -> RegionFaceVertexIter<N> {
    //     let mut size = [0; N];

    //     for axis in 0..N {
    //         size[axis] = match self.sides[axis] {
    //             Side::Left | Side::Right => 1,
    //             Side::Middle => block[axis],
    //         }
    //     }

    //     RegionFaceVertexIter {
    //         inner: IndexSpace::new(size).iter(),
    //         block,
    //         sides: self.sides,
    //     }
    // }

    // pub fn offset_nodes(&self, support: usize) -> RegionOffsetNodeIter<N> {
    //     let size = self.sides.map(|side| match side {
    //         Side::Left | Side::Right => support,
    //         Side::Middle => 1,
    //     });

    //     RegionOffsetNodeIter {
    //         inner: IndexSpace::new(size).iter(),
    //         sides: self.sides,
    //     }
    // }

    // pub fn offset_dir(&self) -> [isize; N] {
    //     self.sides.map(|side| match side {
    //         Side::Left => -1,
    //         Side::Right => 1,
    //         Side::Middle => 0,
    //     })
    // }

    // /// Returns a mask for which a given axis is set if and only if `self.sides[axis] != Side::Middle`.
    // pub fn to_mask(&self) -> AxisMask<N> {
    //     let mut result = AxisMask::empty();

    //     for axis in 0..N {
    //         result.set_to(axis, self.sides[axis] != Side::Middle);
    //     }

    //     result
    // }

    // pub fn masked(&self, mask: AxisMask<N>) -> Self {
    //     let mut sides = self.sides;

    //     for i in 0..N {
    //         if !mask.is_set(i) {
    //             sides[i] = Side::Middle;
    //         }
    //     }

    //     Self::new(sides)
    // }

    // pub fn masked_by_split(&self, split: AxisMask<N>) -> Self {
    //     let mut mask = AxisMask::empty();

    //     for axis in 0..N {
    //         match self.sides[axis] {
    //             Side::Left => mask.set_to(axis, !split.is_set(axis)),
    //             Side::Middle => mask.clear(axis),
    //             Side::Right => mask.set_to(axis, split.is_set(axis)),
    //         }
    //     }

    //     self.masked(mask)
    // }

    /// Converts the region into an integer value.
    pub fn to_linear(&self) -> usize {
        let space = IndexSpace::new([3; N]);
        let index = from_fn(|axis| self.side(axis) as usize);
        space.linear_from_cartesian(index)
    }

    /// Converts an integer value into a region.
    pub fn from_linear(val: usize) -> Self {
        let space = IndexSpace::new([3; N]);
        let index = space.cartesian_from_linear(val);
        Self::new(from_fn(|axis| Side::from_value(index[axis] as u8)))
    }
}

impl<const N: usize> Ord for Region<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        let space = IndexSpace::new([3; N]);
        let index = from_fn(|axis| self.side(axis) as usize);
        let oindex = from_fn(|axis| other.side(axis) as usize);

        space
            .linear_from_cartesian(index)
            .cmp(&space.linear_from_cartesian(oindex))
    }
}

impl<const N: usize> PartialOrd for Region<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<const N: usize> Display for Region<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for axis in 0..N {
            match self.side(axis) {
                Side::Left => f.write_char('-'),
                Side::Middle => f.write_char('='),
                Side::Right => f.write_char('+'),
            }?;
        }
        Ok(())
    }
}

// /// Allows iterating the nodes in a region.
// pub struct RegionNodeIter<const N: usize> {
//     inner: CartesianIter<N>,
//     block: [usize; N],
//     sides: [Side; N],
// }

// impl<const N: usize> Iterator for RegionNodeIter<N> {
//     type Item = [isize; N];

//     fn next(&mut self) -> Option<Self::Item> {
//         let cart = self.inner.next()?;

//         let mut result = [0isize; N];

//         for axis in 0..N {
//             result[axis] = match self.sides[axis] {
//                 Side::Left => -(cart[axis] as isize),
//                 Side::Right => (self.block[axis] + cart[axis]) as isize,
//                 Side::Middle => cart[axis] as isize,
//             }
//         }

//         Some(result)
//     }
// }

// /// Allows iterating the vertices on the face of a region.
// pub struct RegionFaceVertexIter<const N: usize> {
//     inner: CartesianIter<N>,
//     block: [usize; N],
//     sides: [Side; N],
// }

// impl<const N: usize> Iterator for RegionFaceVertexIter<N> {
//     type Item = [usize; N];

//     fn next(&mut self) -> Option<Self::Item> {
//         let cart = self.inner.next()?;

//         let mut result = [0; N];

//         for axis in 0..N {
//             result[axis] = match self.sides[axis] {
//                 Side::Left => 0,
//                 Side::Right => self.block[axis] - 1,
//                 Side::Middle => cart[axis],
//             }
//         }

//         Some(result)
//     }
// }

// pub struct RegionOffsetNodeIter<const N: usize> {
//     inner: CartesianIter<N>,
//     sides: [Side; N],
// }

// impl<const N: usize> Iterator for RegionOffsetNodeIter<N> {
//     type Item = [isize; N];

//     fn next(&mut self) -> Option<Self::Item> {
//         let cart = self.inner.next()?;

//         let mut result = [0isize; N];

//         for axis in 0..N {
//             result[axis] = match self.sides[axis] {
//                 Side::Left => -(cart[axis] as isize),
//                 Side::Right => cart[axis] as isize,
//                 Side::Middle => 0,
//             }
//         }

//         Some(result)
//     }
// }

pub struct RegionIter<const N: usize> {
    inner: CartesianIter<N>,
}

impl<const N: usize> Iterator for RegionIter<N> {
    type Item = Region<N>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(Region::new(self.inner.next()?.map(|idx| match idx {
            0 => Side::Left,
            1 => Side::Middle,
            2 => Side::Right,
            _ => unreachable!(),
        })))
    }
}

impl<const N: usize> ExactSizeIterator for RegionIter<N> {
    fn len(&self) -> usize {
        Region::<N>::COUNT
    }
}

/// Iterates over all regions in an N-dimensional space.
pub fn regions<const N: usize>() -> RegionIter<N> {
    RegionIter {
        inner: IndexSpace::new([3; N]).iter(),
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::{Split, Face};

    use super::{regions, Region, Side};

    #[test]
    fn region_iteration() {
        let comparison = [
            [Side::Left, Side::Left],
            [Side::Middle, Side::Left],
            [Side::Right, Side::Left],
            [Side::Left, Side::Middle],
            [Side::Middle, Side::Middle],
            [Side::Right, Side::Middle],
            [Side::Left, Side::Right],
            [Side::Middle, Side::Right],
            [Side::Right, Side::Right],
        ];

        for (region, compare) in regions().zip(comparison.into_iter()) {
            assert_eq!(region, Region::new(compare));
        }
    }

    #[test]
    fn adjacency() {
        let region = Region::new([Side::Left, Side::Right]);
        assert_eq!(region.adjacency(), 2);

        let mut faces = region.adjacent_faces();
        assert_eq!(faces.next(), Some(Face::negative(0)));
        assert_eq!(faces.next(), Some(Face::positive(1)));
        assert_eq!(faces.next(), None);

        let mut splits = region.adjacent_splits();
        assert_eq!(splits.next(), Some(Split::pack([false, true])));
        assert_eq!(splits.next(), None);

        assert_eq!(region.adjacent_split(), Split::pack([false, true]));
    }
}
