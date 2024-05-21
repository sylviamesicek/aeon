use crate::array::ArrayLike;
use aeon_macros::{derivative, second_derivative};

/// A seperable kernel used for approximating a derivative or numerical operator
/// to some order of accuracy. All kernel weights are applied negative to positive.
pub trait Kernel {
    /// Weights on interior of domain. Length should equal `POSITIVE_SUPPORT + NEGATIVE_SUPPORT + 1`.
    type InteriorWeights: ArrayLike<Elem = f64>;
    /// Weights on edge of domain (for free boundaries).
    type BoundaryWeights: ArrayLike<Elem = f64>;

    const POSITIVE_SUPPORT: usize;
    const NEGATIVE_SUPPORT: usize;

    /// Returns the axis this kernel should be applied to.
    fn axis(&self) -> usize;

    /// Stencil weights for the interior of the domain.
    fn interior(&self) -> Self::InteriorWeights;

    /// Stencil weights for the negative edge of the domain.
    fn negative(&self, right: usize) -> Self::BoundaryWeights;

    /// Stencil weights for the positive edge of the domain.
    fn positive(&self, left: usize) -> Self::BoundaryWeights;

    /// Scale factor given spacing.
    fn scale(&self, spacing: f64) -> f64;
}

/// A finite difference approximation for a derivative.
pub struct FDDerivative<const ORDER: usize>(usize);

impl<const ORDER: usize> FDDerivative<ORDER> {
    pub fn new(axis: usize) -> Self {
        Self(axis)
    }
}

impl Kernel for FDDerivative<2> {
    type InteriorWeights = [f64; 3];
    type BoundaryWeights = [f64; 3];

    const POSITIVE_SUPPORT: usize = 1;
    const NEGATIVE_SUPPORT: usize = 1;

    fn axis(&self) -> usize {
        self.0
    }

    fn interior(&self) -> Self::InteriorWeights {
        derivative!(1, 1, 0)
    }

    fn negative(&self, _right: usize) -> Self::BoundaryWeights {
        derivative!(0, 2, 0)
    }

    fn positive(&self, _left: usize) -> Self::BoundaryWeights {
        derivative!(2, 0, 0)
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / spacing
    }
}

impl Kernel for FDDerivative<4> {
    type InteriorWeights = [f64; 5];
    type BoundaryWeights = [f64; 5];

    const POSITIVE_SUPPORT: usize = 2;
    const NEGATIVE_SUPPORT: usize = 2;

    fn axis(&self) -> usize {
        self.0
    }

    fn interior(&self) -> Self::InteriorWeights {
        derivative!(2, 2, 0)
    }

    fn negative(&self, right: usize) -> Self::BoundaryWeights {
        if right == 0 {
            derivative!(0, 4, 0)
        } else {
            derivative!(0, 4, 1)
        }
    }

    fn positive(&self, left: usize) -> Self::BoundaryWeights {
        if left == 0 {
            derivative!(4, 0, 0)
        } else {
            derivative!(4, 0, -1)
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / spacing
    }
}

/// A finite difference approximation for a second derivative.
pub struct FDSecondDerivative<const ORDER: usize>(usize);

impl<const ORDER: usize> FDSecondDerivative<ORDER> {
    pub fn new(axis: usize) -> Self {
        Self(axis)
    }
}

impl Kernel for FDSecondDerivative<2> {
    type InteriorWeights = [f64; 3];
    type BoundaryWeights = [f64; 4];

    const NEGATIVE_SUPPORT: usize = 1;
    const POSITIVE_SUPPORT: usize = 1;

    fn axis(&self) -> usize {
        self.0
    }

    fn interior(&self) -> Self::InteriorWeights {
        second_derivative!(1, 1, 0)
    }

    fn negative(&self, _: usize) -> Self::BoundaryWeights {
        second_derivative!(0, 3, 0)
    }

    fn positive(&self, _: usize) -> Self::BoundaryWeights {
        second_derivative!(3, 0, 0)
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

impl Kernel for FDSecondDerivative<4> {
    type InteriorWeights = [f64; 5];
    type BoundaryWeights = [f64; 6];

    const NEGATIVE_SUPPORT: usize = 2;
    const POSITIVE_SUPPORT: usize = 2;

    fn axis(&self) -> usize {
        self.0
    }

    fn interior(&self) -> Self::InteriorWeights {
        second_derivative!(2, 2, 0)
    }

    fn negative(&self, right: usize) -> Self::BoundaryWeights {
        if right == 0 {
            second_derivative!(0, 5, 0)
        } else {
            second_derivative!(0, 5, 1)
        }
    }

    fn positive(&self, left: usize) -> Self::BoundaryWeights {
        if left == 0 {
            second_derivative!(5, 0, 0)
        } else {
            second_derivative!(5, 0, -1)
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

/// Kriess-Oliger dissipation kernel.
pub struct FDDissipation<const ORDER: usize>(usize);

impl<const ORDER: usize> FDDissipation<ORDER> {
    pub fn new(axis: usize) -> Self {
        Self(axis)
    }
}

impl Kernel for FDDissipation<2> {
    type InteriorWeights = [f64; 5];
    type BoundaryWeights = [f64; 6];

    const NEGATIVE_SUPPORT: usize = 2;
    const POSITIVE_SUPPORT: usize = 2;

    fn axis(&self) -> usize {
        self.0
    }

    fn interior(&self) -> Self::InteriorWeights {
        [1.0, -4.0, 6.0, -4.0, 1.0]
    }

    fn negative(&self, right: usize) -> Self::BoundaryWeights {
        match right {
            0 => [3.0, -14.0, 26.0, -24.0, 11.0, -2.0],
            _ => [2.0, -9.0, 16.0, -14.0, 6.0, -1.0],
        }
    }

    fn positive(&self, left: usize) -> Self::BoundaryWeights {
        match left {
            0 => [-2.0, 11.0, -24.0, 26.0, -14.0, 3.0],
            _ => [-1.0, 6.0, -14.0, 16.0, -9.0, 2.0],
        }
    }

    fn scale(&self, _: f64) -> f64 {
        -1.0 / 16.0
    }
}

impl Kernel for FDDissipation<4> {
    type InteriorWeights = [f64; 7];
    type BoundaryWeights = [f64; 8];

    const NEGATIVE_SUPPORT: usize = 3;
    const POSITIVE_SUPPORT: usize = 3;

    fn axis(&self) -> usize {
        self.0
    }

    fn interior(&self) -> Self::InteriorWeights {
        [1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]
    }

    fn negative(&self, right: usize) -> Self::BoundaryWeights {
        if right == 0 {
            [4.0, -27.0, 78.0, -125.0, 120.0, -69.0, 22.0, -3.0]
        } else if right == 1 {
            [3.0, -20.0, 57.0, -90.0, 85.0, -48.0, 15.0, -2.0]
        } else {
            [2.0, -13.0, 36.0, -55.0, 50.0, -27.0, 8.0, -1.0]
        }
    }

    fn positive(&self, left: usize) -> Self::BoundaryWeights {
        if left == 0 {
            [-3.0, 22.0, -69.0, 120.0, -125.0, 78.0, -27.0, 4.0]
        } else if left == 1 {
            [-2.0, 15.0, -48.0, 85.0, -90.0, 57.0, -20.0, 3.0]
        } else {
            [-1.0, 8.0, -27.0, 50.0, -55.0, 36.0, -13.0, 2.0]
        }
    }

    fn scale(&self, _spacing: f64) -> f64 {
        1.0 / 64.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_weights() {
        let derivative = FDDerivative::<2>::new(0);
        assert_eq!(derivative.negative(0), [-1.5, 2.0, -0.5]);
        assert_eq!(derivative.interior(), [-0.5, 0.0, 0.5]);
        assert_eq!(derivative.positive(0), [0.5, -2.0, 1.5]);

        let derivative = FDDerivative::<4>::new(0);
        assert_eq!(
            derivative.negative(0),
            [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0]
        );
        assert_eq!(
            derivative.interior(),
            [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]
        );
        assert_eq!(
            derivative.positive(0),
            [1.0 / 4.0, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0]
        );

        let second_derivative = FDSecondDerivative::<2>::new(0);
        assert_eq!(second_derivative.negative(0), [2.0, -5.0, 4.0, -1.0]);
        assert_eq!(second_derivative.interior(), [1.0, -2.0, 1.0]);
        assert_eq!(second_derivative.positive(0), [-1.0, 4.0, -5.0, 2.0]);

        let second_derivative = FDSecondDerivative::<4>::new(0);
        assert_eq!(
            second_derivative.interior(),
            [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
        );
    }
}
