use crate::element::BasisFunction;

pub trait LinearOperator<const N: usize> {
    fn num_operations(&self) -> usize;
    fn apply(&self, i: usize, basis: &impl BasisFunction<N>) -> f64;
}

pub struct Value<const N: usize>([f64; N]);

impl<const N: usize> Value<N> {
    pub fn new(position: [f64; N]) -> Self {
        Value(position)
    }
}

impl<const N: usize> LinearOperator<N> for Value<N> {
    fn num_operations(&self) -> usize {
        1
    }

    fn apply(&self, _: usize, basis: &impl BasisFunction<N>) -> f64 {
        basis.value(self.0)
    }
}

pub struct Values<'a, const N: usize>(pub &'a [[f64; N]]);

impl<'a, const N: usize> LinearOperator<N> for Values<'a, N> {
    fn num_operations(&self) -> usize {
        self.0.len()
    }

    fn apply(&self, i: usize, basis: &impl BasisFunction<N>) -> f64 {
        basis.value(self.0[i])
    }
}

pub struct ProductValue<const N: usize>([f64; N]);

impl<const N: usize> ProductValue<N> {
    pub fn new(position: [f64; N]) -> Self {
        ProductValue(position)
    }
}

impl<const N: usize> LinearOperator<1> for ProductValue<N> {
    fn num_operations(&self) -> usize {
        N
    }

    fn apply(&self, i: usize, basis: &impl BasisFunction<1>) -> f64 {
        basis.value([self.0[i]])
    }
}
