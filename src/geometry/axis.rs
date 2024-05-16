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

    pub fn from_linear(linear: usize) -> Self {
        Self(linear as u64)
    }

    pub fn into_linear(self) -> usize {
        (self.0 as usize).min(Self::COUNT - 1)
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
        self.0 |= (1 << axis) as u64
    }

    pub fn clear(&mut self, axis: usize) {
        self.0 &= !((1 << axis) as u64)
    }

    pub fn set_to(&mut self, axis: usize, value: bool) {
        self.0 &= !((1 << axis) as u64);
        self.0 |= ((value as usize) << axis) as u64;
    }

    pub fn toggle(&mut self, axis: usize) {
        self.0 ^= (1 << axis) as u64
    }

    pub fn is_set(self, axis: usize) -> bool {
        (self.0 & (1 << axis) as u64) != 0
    }
}
