use crate::array::Array;
use aeon_macros::derivative;

/// Boundary conditions for the positive and negative faces of some axis.
pub trait BoundarySet<const N: usize> {
    type NegativeBoundary: Boundary;
    type PositiveBoundary: Boundary;

    fn negative(self: &Self, position: [f64; N]) -> Self::NegativeBoundary;
    fn positive(self: &Self, position: [f64; N]) -> Self::PositiveBoundary;
}

/// A transformer used to combine different kinds of boundary set.
pub struct Mixed<const N: usize, NB: BoundarySet<N>, PB: BoundarySet<N>> {
    negative: NB,
    positive: PB,
}

impl<const N: usize, NB: BoundarySet<N>, PB: BoundarySet<N>> Mixed<N, NB, PB> {
    /// Builds a new mixed boundary set.
    pub const fn new(negative: NB, positive: PB) -> Self {
        Self { negative, positive }
    }
}

impl<const N: usize, NB: BoundarySet<N>, PB: BoundarySet<N>> BoundarySet<N> for Mixed<N, NB, PB> {
    type NegativeBoundary = NB::NegativeBoundary;
    type PositiveBoundary = PB::PositiveBoundary;

    fn negative(self: &Self, position: [f64; N]) -> Self::NegativeBoundary {
        self.negative.negative(position)
    }

    fn positive(self: &Self, position: [f64; N]) -> Self::PositiveBoundary {
        self.positive.positive(position)
    }
}

/// A simple wrapper around an ordinary boundary (thus a simple boundary set has no position
/// dependence).
pub struct Simple<B: Boundary> {
    boundary: B,
}

impl<B: Boundary> Simple<B> {
    /// Constructs a new simple boundary set.
    pub const fn new(boundary: B) -> Self {
        Self { boundary }
    }
}

impl<const N: usize, B: Boundary> BoundarySet<N> for Simple<B> {
    type NegativeBoundary = B;
    type PositiveBoundary = B;

    fn negative(self: &Self, _: [f64; N]) -> Self::NegativeBoundary {
        self.boundary.clone()
    }

    fn positive(self: &Self, _: [f64; N]) -> Self::PositiveBoundary {
        self.boundary.clone()
    }
}

/// Asymptotic flatness along a given axis, as used in ETK.
pub struct AsymptoticFlatness<const ORDER: usize> {
    axis: usize,
}

impl<const ORDER: usize> AsymptoticFlatness<ORDER> {
    /// Constructs a new asymptotic flatness boundary set.
    pub const fn new(axis: usize) -> Self {
        Self { axis: axis }
    }
}

impl<const N: usize> BoundarySet<N> for AsymptoticFlatness<2> {
    type PositiveBoundary = RobinBoundary<2>;
    type NegativeBoundary = RobinBoundary<2>;

    fn negative(self: &Self, position: [f64; N]) -> Self::NegativeBoundary {
        let mut r2 = 0.0;

        for i in 0..N {
            r2 += position[i] * position[i];
        }

        RobinBoundary {
            coefficient: -position[self.axis].abs() / r2,
        }
    }

    fn positive(self: &Self, position: [f64; N]) -> Self::PositiveBoundary {
        let mut r2 = 0.0;

        for i in 0..N {
            r2 += position[i] * position[i];
        }

        RobinBoundary {
            coefficient: -position[self.axis].abs() / r2,
        }
    }
}

impl<const N: usize> BoundarySet<N> for AsymptoticFlatness<4> {
    type PositiveBoundary = RobinBoundary<4>;
    type NegativeBoundary = RobinBoundary<4>;

    fn negative(self: &Self, position: [f64; N]) -> Self::NegativeBoundary {
        let mut r2 = 0.0;

        for i in 0..N {
            r2 += position[i] * position[i];
        }

        RobinBoundary {
            coefficient: -position[self.axis].abs() / r2,
        }
    }

    fn positive(self: &Self, position: [f64; N]) -> Self::PositiveBoundary {
        let mut r2 = 0.0;

        for i in 0..N {
            r2 += position[i] * position[i];
        }

        RobinBoundary {
            coefficient: -position[self.axis].abs() / r2,
        }
    }
}

pub trait Boundary: Clone {
    const EXTENT: usize;

    type Stencil: Array<f64>;

    fn stencil(self: &Self, extent: usize, spacing: f64) -> Self::Stencil;
}

#[derive(Debug, Clone, Copy)]
pub struct FreeBoundary;

impl Boundary for FreeBoundary {
    const EXTENT: usize = 0;
    type Stencil = [f64; 0];

    fn stencil(self: &Self, _: usize, _: f64) -> Self::Stencil {
        []
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SymmetricBoundary<const ORDER: usize>;

macro_rules! impl_symmetric_boundary {
    ($n:expr) => {
        impl Boundary for SymmetricBoundary<{ $n * 2 }> {
            const EXTENT: usize = $n;

            type Stencil = [f64; { $n + 1 }];

            fn stencil(self: &Self, extent: usize, _: f64) -> Self::Stencil {
                let mut result = [0.0; { $n + 1 }];

                result[extent] = 1.0;

                result
            }
        }
    };
}

impl_symmetric_boundary!(1);
impl_symmetric_boundary!(2);
impl_symmetric_boundary!(3);

#[derive(Debug, Clone, Copy)]
pub struct AntiSymmetricBoundary<const ORDER: usize>;

macro_rules! impl_antisymmetric_boundary {
    ($n:expr) => {
        impl Boundary for AntiSymmetricBoundary<{ $n * 2 }> {
            const EXTENT: usize = $n;

            type Stencil = [f64; { $n + 1 }];

            fn stencil(self: &Self, extent: usize, _: f64) -> Self::Stencil {
                let mut result = [0.0; { $n + 1 }];

                result[extent] = -1.0;

                result
            }
        }
    };
}

impl_antisymmetric_boundary!(1);
impl_antisymmetric_boundary!(2);
impl_antisymmetric_boundary!(3);

#[derive(Debug, Clone)]
pub struct RobinBoundary<const ORDER: usize> {
    pub coefficient: f64,
}

impl<const ORDER: usize> RobinBoundary<ORDER> {
    pub const fn new(coefficient: f64) -> Self {
        Self { coefficient }
    }

    pub const fn nuemann() -> Self {
        Self::new(0.0)
    }
}

// macro_rules! robin_boundary_impl {
//     ($n:expr, $t:expr, $($ts:expr),*) => {
//         impl Boundary for RobinBoundary<$n> {
//             const EXTENT: usize = 1;

//             type Stencil = BoundaryStencil<$n>;

//             fn stencil(self: &Self, _: usize) -> Self::Stencil {
//                 let mut derivative = boundary_derivative!(($n - 1), 0);

//                 for i in 0..derivative.len() {
//                     derivative[i] /= self.spacing;
//                 }

//                 let mut result = [0.0; $n];

//                 $(
//                     result[$ts] = -derivative[$ts];
//                 )*

//                 result[0] += self.coef;

//                 for i in 0..result.len() {
//                     result[i] /= derivative[$t];
//                 }

//                 BoundaryStencil(result)
//             }
//         }

//         robin_boundary_impl! { ($n - 1), $($ts)* }
//     };
//     ($n: expr) => {}
// }

// robin_boundary_impl!(5, 4, 3, 2, 1, 0);

impl Boundary for RobinBoundary<2> {
    const EXTENT: usize = 1;

    type Stencil = [f64; 2];

    fn stencil(self: &Self, _: usize, spacing: f64) -> Self::Stencil {
        let mut derivative = derivative!(2, 0, -1);
        derivative.reverse();

        for i in 0..derivative.len() {
            derivative[i] /= spacing;
        }

        let mut result = [0.0; 2];

        result[0] = -derivative[1];
        result[1] = -derivative[2];

        result[0] += self.coefficient;

        for i in 0..result.len() {
            result[i] /= derivative[0];
        }

        result
    }
}

impl Boundary for RobinBoundary<4> {
    const EXTENT: usize = 1;
    type Stencil = [f64; 4];

    fn stencil(self: &Self, _: usize, spacing: f64) -> Self::Stencil {
        let mut derivative = derivative!(4, 0, -1);
        derivative.reverse();

        for i in 0..derivative.len() {
            derivative[i] /= spacing;
        }

        let mut result = [0.0; 4];

        result[0] = -derivative[1];
        result[1] = -derivative[2];
        result[2] = -derivative[3];
        result[3] = -derivative[4];

        result[0] += self.coefficient;

        for i in 0..result.len() {
            result[i] /= derivative[0];
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPACING: f64 = 0.1;

    #[test]
    fn symmetric_boundary() {
        let boundary = SymmetricBoundary::<4>;

        assert_eq!(boundary.stencil(0, SPACING), [1.0, 0.0, 0.0]);
        assert_eq!(boundary.stencil(1, SPACING), [0.0, 1.0, 0.0]);
        assert_eq!(boundary.stencil(2, SPACING), [0.0, 0.0, 1.0]);

        let boundary = AntiSymmetricBoundary::<4>;

        assert_eq!(boundary.stencil(0, SPACING), [-1.0, 0.0, 0.0]);
        assert_eq!(boundary.stencil(1, SPACING), [0.0, -1.0, 0.0]);
        assert_eq!(boundary.stencil(2, SPACING), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn robin_boundary() {
        let boundary = RobinBoundary::<2>::nuemann();
        // Second order robin boundary condition should behave the same as symmetric boundary condition.
        assert_eq!(boundary.stencil(1, SPACING), [0.0, 1.0]);

        let boundary = RobinBoundary::<4>::nuemann();
        // Desmos agrees, this should work.
        assert_eq!(
            boundary.stencil(1, SPACING),
            [-3.3333333333333335, 6.0, -2.0, 0.3333333333333333]
        );
    }
}
