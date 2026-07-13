use crate::geometry::IndexSpace;

/// Support points for an element.
pub trait Support<const N: usize> {
    /// Total number of points in the support.
    fn num_points(&self) -> usize;
    /// Position of the `i`th support point.
    fn point(&self, i: usize) -> [f64; N];
}

/// A [-1, 1]ᴺ hypercube with uniformly placed support points along each axis.
pub struct Uniform<const N: usize>([usize; N]);

impl<const N: usize> Uniform<N> {
    pub fn new(width: [usize; N]) -> Self {
        Self(width)
    }
}

impl<const N: usize> Support<N> for Uniform<N> {
    fn num_points(&self) -> usize {
        self.0.iter().product()
    }

    fn point(&self, i: usize) -> [f64; N] {
        let spacing: [f64; N] = std::array::from_fn(|i| 2.0 / (self.0[i] - 1) as f64);
        let cartesian = IndexSpace::new(self.0).cartesian_from_linear(i);

        std::array::from_fn(|i| spacing[i] * cartesian[i] as f64 - 1.0)
    }
}
