use crate::array::Array;
use aeon_macros::derivative;

/// Boundary conditions for the positive and negative faces of some axis.
pub trait BoundarySet<const N: usize> {
    type NegativeBoundary: Boundary;
    type PositiveBoundary: Boundary;

    fn negative(&self, position: [f64; N]) -> Self::NegativeBoundary;
    fn positive(&self, position: [f64; N]) -> Self::PositiveBoundary;
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

    fn negative(&self, position: [f64; N]) -> Self::NegativeBoundary {
        self.negative.negative(position)
    }

    fn positive(&self, position: [f64; N]) -> Self::PositiveBoundary {
        self.positive.positive(position)
    }
}

/// Asymptotic flatness along a given axis, as used in ETK.
pub struct AsymptoticFlatness<const ORDER: usize> {
    axis: usize,
}

impl<const ORDER: usize> AsymptoticFlatness<ORDER> {
    /// Constructs a new asymptotic flatness boundary set.
    pub const fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl<const N: usize, const ORDER: usize> BoundarySet<N> for AsymptoticFlatness<ORDER>
where
    RobinBoundary<ORDER>: Boundary,
{
    type PositiveBoundary = RobinBoundary<ORDER>;
    type NegativeBoundary = RobinBoundary<ORDER>;

    fn negative(&self, position: [f64; N]) -> Self::NegativeBoundary {
        let mut r2 = 0.0;

        for pos in position.iter() {
            r2 += pos * pos;
        }

        RobinBoundary {
            coefficient: -position[self.axis].abs() / r2,
        }
    }

    fn positive(&self, position: [f64; N]) -> Self::PositiveBoundary {
        let mut r2 = 0.0;

        for pos in position.iter() {
            r2 += pos * pos;
        }

        RobinBoundary {
            coefficient: -position[self.axis].abs() / r2,
        }
    }
}

/// Generic representation of a boundary.
pub trait Boundary: Clone {
    /// Number of ghost points which can be filled using this boundary.
    const GHOST: usize;
    /// Should the boundary be set to zero?.
    const IS_DIRITCHLET: bool = false;
    /// Interior support required for this boundary.
    type Stencil: Array<f64>;
    /// Produces the stencil used for computing the given ghost value.
    fn stencil(&self, ghost: usize, spacing: f64) -> Self::Stencil;
}

/// Blanket implementation of boundary set for all boundaries.
impl<T: Boundary, const N: usize> BoundarySet<N> for T {
    type NegativeBoundary = T;
    type PositiveBoundary = T;

    fn negative(&self, _: [f64; N]) -> Self::NegativeBoundary {
        self.clone()
    }

    fn positive(&self, _: [f64; N]) -> Self::PositiveBoundary {
        self.clone()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FreeBoundary;

impl Boundary for FreeBoundary {
    const GHOST: usize = 0;
    type Stencil = [f64; 0];

    fn stencil(&self, _: usize, _: f64) -> Self::Stencil {
        []
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SymmetricBoundary<const ORDER: usize>;

macro_rules! impl_symmetric_boundary {
    ($n:expr) => {
        impl Boundary for SymmetricBoundary<{ $n * 2 }> {
            const GHOST: usize = $n;

            type Stencil = [f64; { $n + 1 }];

            fn stencil(&self, ghost: usize, _: f64) -> Self::Stencil {
                let mut result = [0.0; { $n + 1 }];

                result[ghost + 1] = 1.0;

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
            const GHOST: usize = $n;
            const IS_DIRITCHLET: bool = true;

            type Stencil = [f64; { $n + 1 }];

            fn stencil(&self, ghost: usize, _: f64) -> Self::Stencil {
                let mut result = [0.0; { $n + 1 }];

                result[ghost + 1] = -1.0;

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

macro_rules! robin_boundary_impl {
    ($order:expr) => {
        impl Boundary for RobinBoundary<$order> {
            const GHOST: usize = 1;

            type Stencil = [f64; $order];

            fn stencil(&self, _: usize, spacing: f64) -> Self::Stencil {
                let mut derivative = derivative!($order, 0, -1);
                derivative.reverse();

                for d in derivative.iter_mut() {
                    *d /= spacing;
                }

                let mut result = [0.0; $order];

                for i in 0..($order) {
                    result[i] = -derivative[i + 1];
                }

                result[0] += self.coefficient;

                for res in result.iter_mut() {
                    *res /= derivative[0]
                }

                result
            }
        }
    };
}

robin_boundary_impl!(2);
robin_boundary_impl!(4);
robin_boundary_impl!(6);
robin_boundary_impl!(8);

#[cfg(test)]
mod tests {
    use super::*;

    const SPACING: f64 = 0.1;

    #[test]
    fn symmetric_boundary() {
        let boundary = SymmetricBoundary::<4>;

        assert_eq!(boundary.stencil(0, SPACING), [0.0, 1.0, 0.0]);
        assert_eq!(boundary.stencil(1, SPACING), [0.0, 0.0, 1.0]);

        let boundary = AntiSymmetricBoundary::<4>;

        assert_eq!(boundary.stencil(0, SPACING), [0.0, -1.0, 0.0]);
        assert_eq!(boundary.stencil(1, SPACING), [0.0, 0.0, -1.0]);
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
