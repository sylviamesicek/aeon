use crate::array::Array;
use aeon_macros::{derivative, second_derivative};

/// A seperable kernal used for approximating a derivative or numerical operator
/// to some order of accuracy.
pub trait Kernel<const N: usize> {
    type InteriorStencil: Array<f64>;
    type BoundaryStencil: Array<f64>;

    const POSITIVE_SUPPORT: usize = 0;
    const NEGATIVE_SUPPORT: usize = 0;

    /// Stencil weights for the interior of the domain.
    fn interior() -> Self::InteriorStencil;

    /// Stencil weights for the negative edge of the domain.
    fn negative(left: usize) -> Self::BoundaryStencil;

    /// Stencil weights for the positive edge of the domain.
    fn positive(right: usize) -> Self::BoundaryStencil;

    /// Scale factor given spacing.
    fn scale(spacing: f64) -> f64;
}

pub struct FDDerivative<const ORDER: usize>;

impl<const N: usize> Kernel<N> for FDDerivative<2> {
    type InteriorStencil = [f64; 3];
    type BoundaryStencil = [f64; 3];

    const POSITIVE_SUPPORT: usize = 1;
    const NEGATIVE_SUPPORT: usize = 1;

    fn interior() -> Self::InteriorStencil {
        derivative!(1, 1, 0)
    }

    fn negative(_: usize) -> Self::BoundaryStencil {
        derivative!(1, 1, -1)
    }

    fn positive(_: usize) -> Self::BoundaryStencil {
        derivative!(1, 1, 1)
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / spacing
    }
}

impl<const N: usize> Kernel<N> for FDDerivative<4> {
    type InteriorStencil = [f64; 5];
    type BoundaryStencil = [f64; 5];

    const POSITIVE_SUPPORT: usize = 2;
    const NEGATIVE_SUPPORT: usize = 2;

    fn interior() -> Self::InteriorStencil {
        derivative!(2, 2, 0)
    }

    fn negative(left: usize) -> Self::BoundaryStencil {
        if left == 0 {
            derivative!(0, 4, 0)
        } else {
            derivative!(0, 4, 1)
        }
    }

    fn positive(right: usize) -> Self::BoundaryStencil {
        if right == 0 {
            derivative!(4, 0, 0)
        } else {
            derivative!(4, 0, -1)
        }
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / spacing
    }
}

pub struct FDSecondDerivative<const ORDER: usize>;

impl<const N: usize> Kernel<N> for FDSecondDerivative<2> {
    type InteriorStencil = [f64; 3];
    type BoundaryStencil = [f64; 4];

    const NEGATIVE_SUPPORT: usize = 1;
    const POSITIVE_SUPPORT: usize = 1;

    fn interior() -> Self::InteriorStencil {
        second_derivative!(1, 1, 0)
    }

    fn negative(_: usize) -> Self::BoundaryStencil {
        second_derivative!(0, 3, 0)
    }

    fn positive(_: usize) -> Self::BoundaryStencil {
        second_derivative!(3, 0, 0)
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

impl<const N: usize> Kernel<N> for FDSecondDerivative<4> {
    type InteriorStencil = [f64; 5];
    type BoundaryStencil = [f64; 6];

    const NEGATIVE_SUPPORT: usize = 2;
    const POSITIVE_SUPPORT: usize = 2;

    fn interior() -> Self::InteriorStencil {
        second_derivative!(2, 2, 0)
    }

    fn negative(left: usize) -> Self::BoundaryStencil {
        if left == 0 {
            second_derivative!(0, 5, 0)
        } else {
            second_derivative!(0, 5, 1)
        }
    }

    fn positive(right: usize) -> Self::BoundaryStencil {
        if right == 0 {
            second_derivative!(5, 0, 0)
        } else {
            second_derivative!(5, 0, -1)
        }
    }

    fn scale(spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}
