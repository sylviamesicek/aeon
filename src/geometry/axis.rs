#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct AxisMask<const N: usize>(u64);

impl<const N: usize> AxisMask<N> {
    pub const COUNT: usize = 2usize.pow(N as u32);

    pub fn empty() -> Self {
        Self(0)
    }

    pub fn full() -> Self {
        Self(u64::MAX)
    }

    pub fn new(linear: usize) -> Self {
        Self(linear as u64)
    }

    pub fn linear(self) -> usize {
        self.0 as usize
    }

    pub fn unpack(self) -> [bool; N] {
        let mut result = [false; N];

        for axis in 0..N {
            result[axis] = self.is_set(axis);
        }

        result
    }

    pub fn is_set(self, axis: usize) -> bool {
        (self.0 & (1 << axis) as u64) != 0
    }

    pub fn set(&mut self, axis: usize) {
        self.0 |= (1 << axis) as u64
    }
}
