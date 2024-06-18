use std::{array::from_fn, borrow::Borrow};

use crate::{array::ArrayLike, prelude::IndexSpace};
use aeon_macros::{derivative, second_derivative};

#[derive(Debug, Clone, Copy)]
pub enum Domain {
    Interior,
    Negative(usize),
    Positive(usize),
}

#[derive(Clone, Debug, Copy)]
pub enum Kernel {
    Value,
    Derivative,
    SecondDerivative,
}

impl Kernel {
    pub const fn support<const ORDER: usize>(self) -> usize {
        match (ORDER, self) {
            (2, Self::Value) => 0,
            (2, Self::Derivative) => 1,
            (2, Self::SecondDerivative) => 1,

            (4, Self::Value) => 0,
            (4, Self::Derivative) => 2,
            (4, Self::SecondDerivative) => 2,

            _ => panic!("Order unsupport"),
        }
    }

    pub const fn interior<const ORDER: usize>(self) -> &'static [f64] {
        match (ORDER, self) {
            (2, Self::Value) => &[1.0],
            (2, Self::Derivative) => &derivative!(1, 1, 0),
            (2, Self::SecondDerivative) => &second_derivative!(1, 1, 0),

            (4, Self::Value) => &[1.0],
            (4, Self::Derivative) => &derivative!(2, 2, 0),
            (4, Self::SecondDerivative) => &second_derivative!(2, 2, 0),

            _ => panic!("Order Unsupported"),
        }
    }

    pub const fn negative<const ORDER: usize>(self, left: usize) -> &'static [f64] {
        match (ORDER, left, self) {
            (2, _, Self::Value) => &[],
            (2, 0, Self::Derivative) => &derivative!(0, 2, 0),
            (2, 0, Self::SecondDerivative) => &second_derivative!(0, 3, 0),

            (4, _, Self::Value) => &[],
            (4, 0, Self::Derivative) => &derivative!(0, 4, 0),
            (4, 1, Self::Derivative) => &derivative!(0, 4, 1),
            (4, 0, Self::SecondDerivative) => &second_derivative!(0, 5, 0),
            (4, 1, Self::SecondDerivative) => &second_derivative!(0, 5, 1),

            _ => panic!("Order Unsupported"),
        }
    }

    pub const fn positive<const ORDER: usize>(self, right: usize) -> &'static [f64] {
        match (ORDER, right, self) {
            (2, _, Self::Value) => &[],
            (2, 0, Self::Derivative) => &derivative!(2, 0, 0),
            (2, 0, Self::SecondDerivative) => &second_derivative!(3, 0, 0),

            (4, _, Self::Value) => &[],
            (4, 0, Self::Derivative) => &derivative!(4, 0, 0),
            (4, 1, Self::Derivative) => &derivative!(4, 0, 1),
            (4, 0, Self::SecondDerivative) => &second_derivative!(5, 0, 0),
            (4, 1, Self::SecondDerivative) => &second_derivative!(5, 0, 1),

            _ => panic!("Order Unsupported"),
        }
    }
}

// /// A seperable kernel used for approximating a derivative or numerical operator
// /// to some order of accuracy. All kernel weights are applied negative to positive.
// pub trait Kernel {
//     const POSITIVE_SUPPORT: usize;
//     const NEGATIVE_SUPPORT: usize;
//     const BOUNDARY_SUPPORT: usize = 3;

//     fn weights(&self, domain: AxisDomain) -> &[f64];

//     /// Scale factor given spacing.
//     fn scale(&self, spacing: f64) -> f64;
// }

// /// A finite difference "approximation" of the identity operator (just returns the value at the given point).
// pub struct Value;

// impl Kernel for Value {
//     const POSITIVE_SUPPORT: usize = 0;
//     const NEGATIVE_SUPPORT: usize = 0;
//     const BOUNDARY_SUPPORT: usize = 1;

//     fn weights(&self, _domain: AxisDomain) -> &[f64] {
//         &[1.0]
//     }

//     fn scale(&self, _spacing: f64) -> f64 {
//         1.0
//     }
// }

// /// A finite difference approximation for a derivative.
// pub struct Derivative<const ORDER: usize>;

// impl Kernel for Derivative<2> {
//     const POSITIVE_SUPPORT: usize = 1;
//     const NEGATIVE_SUPPORT: usize = 1;
//     const BOUNDARY_SUPPORT: usize = 3;

//     fn weights(&self, domain: AxisDomain) -> &[f64] {
//         match domain {
//             AxisDomain::Interior => &derivative!(1, 1, 0),
//             AxisDomain::Negative(_) => &derivative!(0, 2, 0),
//             AxisDomain::Positive(_) => &derivative!(2, 0, 0),
//         }
//     }

//     fn scale(&self, spacing: f64) -> f64 {
//         1.0 / spacing
//     }
// }

// impl Kernel for Derivative<4> {
//     const POSITIVE_SUPPORT: usize = 2;
//     const NEGATIVE_SUPPORT: usize = 2;
//     const BOUNDARY_SUPPORT: usize = 5;

//     fn weights(&self, domain: AxisDomain) -> &[f64] {
//         match domain {
//             AxisDomain::Interior => &derivative!(2, 2, 0),
//             AxisDomain::Negative(0) => &derivative!(0, 4, 0),
//             AxisDomain::Negative(_) => &derivative!(0, 4, 1),
//             AxisDomain::Positive(0) => &derivative!(4, 0, 0),
//             AxisDomain::Positive(_) => &derivative!(4, 0, -1),
//         }
//     }

//     fn scale(&self, spacing: f64) -> f64 {
//         1.0 / spacing
//     }
// }

// /// A finite difference approximation for a second derivative.
// pub struct SecondDerivative<const ORDER: usize>;

// impl Kernel for SecondDerivative<2> {
//     const NEGATIVE_SUPPORT: usize = 1;
//     const POSITIVE_SUPPORT: usize = 1;
//     const BOUNDARY_SUPPORT: usize = 4;

//     fn weights(&self, domain: AxisDomain) -> &[f64] {
//         match domain {
//             AxisDomain::Interior => &second_derivative!(1, 1, 0),
//             AxisDomain::Negative(_) => &second_derivative!(0, 3, 0),
//             AxisDomain::Positive(_) => &second_derivative!(3, 0, 0),
//         }
//     }

//     fn scale(&self, spacing: f64) -> f64 {
//         1.0 / (spacing * spacing)
//     }
// }

// impl Kernel for SecondDerivative<4> {
//     const NEGATIVE_SUPPORT: usize = 2;
//     const POSITIVE_SUPPORT: usize = 2;
//     const BOUNDARY_SUPPORT: usize = 6;

//     fn weights(&self, domain: AxisDomain) -> &[f64] {
//         match domain {
//             AxisDomain::Interior => &second_derivative!(2, 2, 0),
//             AxisDomain::Negative(0) => &second_derivative!(0, 5, 0),
//             AxisDomain::Negative(_) => &second_derivative!(0, 5, 1),
//             AxisDomain::Positive(0) => &second_derivative!(5, 0, 0),
//             AxisDomain::Positive(_) => &second_derivative!(5, 0, -1),
//         }
//     }

//     fn scale(&self, spacing: f64) -> f64 {
//         1.0 / (spacing * spacing)
//     }
// }

// /// Kriess-Oliger dissipation kernel.
// pub struct Dissipation<const ORDER: usize>;

// impl Kernel for Dissipation<2> {
//     const NEGATIVE_SUPPORT: usize = 2;
//     const POSITIVE_SUPPORT: usize = 2;
//     const BOUNDARY_SUPPORT: usize = 6;

//     fn weights(&self, domain: AxisDomain) -> &[f64] {
//         match domain {
//             AxisDomain::Interior => &[1.0, -4.0, 6.0, -4.0, 1.0],
//             AxisDomain::Negative(0) => &[3.0, -14.0, 26.0, -24.0, 11.0, -2.0],
//             AxisDomain::Negative(_) => &[2.0, -9.0, 16.0, -14.0, 6.0, -1.0],
//             AxisDomain::Positive(0) => &[-2.0, 11.0, -24.0, 26.0, -14.0, 3.0],
//             AxisDomain::Positive(_) => &[-1.0, 6.0, -14.0, 16.0, -9.0, 2.0],
//         }
//     }

//     fn scale(&self, _: f64) -> f64 {
//         -1.0 / 16.0
//     }
// }

// impl Kernel for Dissipation<4> {
//     const NEGATIVE_SUPPORT: usize = 3;
//     const POSITIVE_SUPPORT: usize = 3;
//     const BOUNDARY_SUPPORT: usize = 8;

//     fn weights(&self, domain: AxisDomain) -> &[f64] {
//         match domain {
//             AxisDomain::Interior => &[1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0],

//             AxisDomain::Negative(0) => &[4.0, -27.0, 78.0, -125.0, 120.0, -69.0, 22.0, -3.0],
//             AxisDomain::Negative(1) => &[3.0, -20.0, 57.0, -90.0, 85.0, -48.0, 15.0, -2.0],
//             AxisDomain::Negative(_) => &[2.0, -13.0, 36.0, -55.0, 50.0, -27.0, 8.0, -1.0],

//             AxisDomain::Positive(0) => &[-3.0, 22.0, -69.0, 120.0, -125.0, 78.0, -27.0, 4.0],
//             AxisDomain::Positive(1) => &[-2.0, 15.0, -48.0, 85.0, -90.0, 57.0, -20.0, 3.0],
//             AxisDomain::Positive(_) => &[-1.0, 8.0, -27.0, 50.0, -55.0, 36.0, -13.0, 2.0],
//         }
//     }

//     fn scale(&self, _spacing: f64) -> f64 {
//         1.0 / 64.0
//     }
// }

// pub trait Operator<const N: usize> {
//     fn offset(&self, domain: [AxisDomain; N], vertex: [usize; N], size: [usize; N]) -> [isize; N];
//     fn weights(&self, domain: [AxisDomain; N], func: impl FnMut([usize; N], f64));
//     fn scale(&self, spacing: [f64; N]) -> f64;
// }

// impl<A: Kernel, B: Kernel> Operator<2> for (A, B) {
//     fn offset(&self, domain: [AxisDomain; 2], vertex: [usize; 2], size: [usize; 2]) -> [isize; 2] {
//         let mut result = [0; 2];

//         result[0] = match domain[0] {
//             AxisDomain::Interior => vertex[0] as isize - A::NEGATIVE_SUPPORT as isize,
//             AxisDomain::Negative(_) => 0,
//             AxisDomain::Positive(_) => size[0] as isize - A::BOUNDARY_SUPPORT as isize,
//         };

//         // result[0] = match domain[0] {
//         //     AxisDomain::Interior => -(A::NEGATIVE_SUPPORT as isize),
//         //     AxisDomain::Negative(_) => 0,
//         //     AxisDomain::Positive(_) => -A::BoundaryWeights::
//         // };

//         result
//     }

//     fn weights(&self, domain: [AxisDomain; 2], mut func: impl FnMut([usize; 2], f64)) {
//         let weights = [self.0.weights(domain[0]), self.1.weights(domain[1])];
//         let size = [weights[0].len(), weights[1].len()];

//         for index in IndexSpace::new(size).iter() {
//             let mut weight = 1.0;
//             weight *= weights[0][index[0]];
//             weight *= weights[1][index[1]];

//             func(index, weight)
//         }
//     }

//     fn scale(&self, spacing: [f64; 2]) -> f64 {
//         let mut result = 1.0;
//         result *= self.0.scale(spacing[0]);
//         result *= self.1.scale(spacing[1]);
//         result
//     }
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn kernel_weights() {
//         let derivative = Derivative::<2>;
//         assert_eq!(derivative.negative(0), [-1.5, 2.0, -0.5]);
//         assert_eq!(derivative.interior(), [-0.5, 0.0, 0.5]);
//         assert_eq!(derivative.positive(0), [0.5, -2.0, 1.5]);

//         let derivative = Derivative::<4>;
//         assert_eq!(
//             derivative.negative(0),
//             [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0]
//         );
//         assert_eq!(
//             derivative.interior(),
//             [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]
//         );
//         assert_eq!(
//             derivative.positive(0),
//             [1.0 / 4.0, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0]
//         );

//         let second_derivative = SecondDerivative::<2>;
//         assert_eq!(second_derivative.negative(0), [2.0, -5.0, 4.0, -1.0]);
//         assert_eq!(second_derivative.interior(), [1.0, -2.0, 1.0]);
//         assert_eq!(second_derivative.positive(0), [-1.0, 4.0, -5.0, 2.0]);

//         let second_derivative = SecondDerivative::<4>;
//         assert_eq!(
//             second_derivative.interior(),
//             [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
//         );
//     }
// }
