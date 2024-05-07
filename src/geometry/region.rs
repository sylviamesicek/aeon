use super::{CartesianIter, IndexSpace};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Face {
    pub axis: usize,
    pub side: bool,
}

impl Face {
    pub fn negative(axis: usize) -> Self {
        Self { axis, side: false }
    }

    pub fn positive(axis: usize) -> Self {
        Self { axis, side: true }
    }
}

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

pub fn faces<const N: usize>() -> FaceIter<N> {
    FaceIter {
        axis: 0,
        side: false,
    }
}

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

    pub fn new(sides: [Side; N]) -> Self {
        Self { sides }
    }

    pub fn sides(&self) -> [Side; N] {
        self.sides
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

    pub fn face_from_axis(&self, axis: usize) -> Face {
        Face {
            axis,
            side: self.sides[axis] == Side::Right,
        }
    }

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

    pub fn face_nodes(&self, block: [usize; N]) -> RegionFaceNodeIter<N> {
        let mut size = [0; N];

        for axis in 0..N {
            size[axis] = match self.sides[axis] {
                Side::Left | Side::Right => 1,
                Side::Middle => block[axis],
            }
        }

        RegionFaceNodeIter {
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

pub struct RegionFaceNodeIter<const N: usize> {
    inner: CartesianIter<N>,
    block: [usize; N],
    sides: [Side; N],
}

impl<const N: usize> Iterator for RegionFaceNodeIter<N> {
    type Item = [isize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let cart = self.inner.next()?;

        let mut result = [0isize; N];

        for axis in 0..N {
            result[axis] = match self.sides[axis] {
                Side::Left => 0,
                Side::Right => (self.block[axis] - 1) as isize,
                Side::Middle => cart[axis] as isize,
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
