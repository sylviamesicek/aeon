use std::array::from_fn;

use super::{index::IndexWindow, AxisMask, CartesianIter, Face, IndexSpace};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region<const N: usize> {
    sides: [Side; N],
}

impl<const N: usize> Region<N> {
    /// Number of different regions in a given number of dimensions.
    pub const COUNT: usize = 3usize.pow(N as u32);

    pub const CENTRAL: Self = Self::new([Side::Middle; N]);

    pub const fn new(sides: [Side; N]) -> Self {
        Self { sides }
    }

    pub fn sides(&self) -> [Side; N] {
        self.sides
    }

    pub fn side(&self, axis: usize) -> Side {
        self.sides[axis]
    }

    pub fn reverse(&self) -> Self {
        let mut result = [Side::Left; N];

        for axis in 0..N {
            result[axis] = self.sides[axis].reverse();
        }

        Self::new(result)
    }

    pub fn adjacency(&self) -> usize {
        self.sides
            .into_iter()
            .filter(|&s| s != Side::Middle)
            .count()
    }

    /// Iterates over all faces one would have to move over to get to the region.
    pub fn adjacent_faces(&self) -> impl Iterator<Item = Face> + '_ {
        (0..N)
            .filter(|&axis| self.side(axis) != Side::Middle)
            .map(|axis| Face {
                axis,
                side: self.side(axis) == Side::Right,
            })
    }

    pub fn adjacent_splits(&self) -> impl Iterator<Item = AxisMask<N>> + '_ {
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
            .map(|index| AxisMask::pack(from_fn(|axis| index[axis] != 0)))
    }

    pub fn face_from_axis(&self, axis: usize) -> Face {
        Face {
            axis,
            side: self.sides[axis] == Side::Right,
        }
    }

    /// Returns an index space with the same size as the region.
    pub fn index_space(&self, support: usize, block: [usize; N]) -> IndexSpace<N> {
        let mut size = [0; N];

        for axis in 0..N {
            if self.sides[axis] != Side::Middle {
                size[axis] = support;
            } else {
                size[axis] = block[axis]
            }
        }

        IndexSpace::new(size)
    }

    /// Iterates nodes in the given region (including ghost nodes and nodes
    /// on faces).
    pub fn nodes(&self, support: usize, block: [usize; N]) -> RegionNodeIter<N> {
        RegionNodeIter {
            inner: self.index_space(support, block).iter(),
            block,
            sides: self.sides,
        }
    }

    pub fn face_vertices(&self, block: [usize; N]) -> RegionFaceVertexIter<N> {
        let mut size = [0; N];

        for axis in 0..N {
            size[axis] = match self.sides[axis] {
                Side::Left | Side::Right => 1,
                Side::Middle => block[axis],
            }
        }

        RegionFaceVertexIter {
            inner: IndexSpace::new(size).iter(),
            block,
            sides: self.sides,
        }
    }

    pub fn offset_nodes(&self, support: usize) -> RegionOffsetNodeIter<N> {
        let size = self.sides.map(|side| match side {
            Side::Left | Side::Right => support,
            Side::Middle => 1,
        });

        RegionOffsetNodeIter {
            inner: IndexSpace::new(size).iter(),
            sides: self.sides,
        }
    }

    pub fn offset_dir(&self) -> [isize; N] {
        self.sides.map(|side| match side {
            Side::Left => -1,
            Side::Right => 1,
            Side::Middle => 0,
        })
    }

    /// Returns a mask for which a given axis is set if and only if self.sides[axis] != Middle.
    pub fn to_mask(&self) -> AxisMask<N> {
        let mut result = AxisMask::empty();

        for axis in 0..N {
            result.set_to(axis, self.sides[axis] != Side::Middle);
        }

        result
    }

    pub fn masked(&self, mask: AxisMask<N>) -> Self {
        let mut sides = self.sides;

        for i in 0..N {
            if !mask.is_set(i) {
                sides[i] = Side::Middle;
            }
        }

        Self::new(sides)
    }

    pub fn masked_by_split(&self, split: AxisMask<N>) -> Self {
        let mut mask = AxisMask::empty();

        for axis in 0..N {
            match self.sides[axis] {
                Side::Left => mask.set_to(axis, !split.is_set(axis)),
                Side::Middle => mask.clear(axis),
                Side::Right => mask.set_to(axis, split.is_set(axis)),
            }
        }

        self.masked(mask)
    }

    pub fn to_linear(&self) -> usize {
        let space = IndexSpace::new([3; N]);
        let index = from_fn(|axis| self.side(axis) as usize);
        space.linear_from_cartesian(index)
    }
}

pub struct RegionNodeIter<const N: usize> {
    inner: CartesianIter<N>,
    block: [usize; N],
    sides: [Side; N],
}

impl<const N: usize> Iterator for RegionNodeIter<N> {
    type Item = [isize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let cart = self.inner.next()?;

        let mut result = [0isize; N];

        for axis in 0..N {
            result[axis] = match self.sides[axis] {
                Side::Left => -(cart[axis] as isize),
                Side::Right => (self.block[axis] + cart[axis]) as isize,
                Side::Middle => cart[axis] as isize,
            }
        }

        Some(result)
    }
}

pub struct RegionFaceVertexIter<const N: usize> {
    inner: CartesianIter<N>,
    block: [usize; N],
    sides: [Side; N],
}

impl<const N: usize> Iterator for RegionFaceVertexIter<N> {
    type Item = [usize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let cart = self.inner.next()?;

        let mut result = [0; N];

        for axis in 0..N {
            result[axis] = match self.sides[axis] {
                Side::Left => 0,
                Side::Right => self.block[axis] - 1,
                Side::Middle => cart[axis],
            }
        }

        Some(result)
    }
}

pub struct RegionOffsetNodeIter<const N: usize> {
    inner: CartesianIter<N>,
    sides: [Side; N],
}

impl<const N: usize> Iterator for RegionOffsetNodeIter<N> {
    type Item = [isize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let cart = self.inner.next()?;

        let mut result = [0isize; N];

        for axis in 0..N {
            result[axis] = match self.sides[axis] {
                Side::Left => -(cart[axis] as isize),
                Side::Right => cart[axis] as isize,
                Side::Middle => 0,
            }
        }

        Some(result)
    }
}

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

pub fn regions<const N: usize>() -> RegionIter<N> {
    RegionIter {
        inner: IndexSpace::new([3; N]).iter(),
    }
}
