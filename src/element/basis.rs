use crate::geometry::IndexSpace;

pub trait Basis<const N: usize> {
    type Function: BasisFunction<N>;

    fn order(&self) -> usize;
    fn func(&self, deg: usize) -> Self::Function;
}

pub trait BasisFunction<const N: usize> {
    fn value(&self, point: [f64; N]) -> f64;
}

pub struct Monomials<const N: usize>([usize; N]);

impl<const N: usize> Monomials<N> {
    pub fn new(degree: [usize; N]) -> Self {
        Self(degree)
    }
}

impl<const N: usize> Basis<N> for Monomials<N> {
    type Function = Monomial<N>;

    fn order(&self) -> usize {
        self.0.iter().product()
    }

    fn func(&self, deg: usize) -> Self::Function {
        let degree = IndexSpace::new(self.0).cartesian_from_linear(deg);
        Monomial(degree)
    }
}

pub struct Monomial<const N: usize>([usize; N]);

impl<const N: usize> BasisFunction<N> for Monomial<N> {
    fn value(&self, point: [f64; N]) -> f64 {
        let mut result = 1.0;
        for i in 0..N {
            result *= point[i].powi(self.0[i] as i32);
        }
        result
    }
}
