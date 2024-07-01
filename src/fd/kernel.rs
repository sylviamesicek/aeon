use aeon_macros::{derivative, second_derivative};

/// The support along an axis for a given point in node space.
#[derive(Debug, Clone, Copy)]
pub enum Support {
    /// The point has the necessary support on both sides.
    Interior,
    /// The point lies near a negative boundary, and must use a boundary support.
    Negative(usize),
    /// The point lies near a positive boundary, and must use a boundary support.
    Positive(usize),
}

/// The order of accuracy of finite difference stencils.
#[derive(Debug, Clone, Copy)]
pub enum Order {
    /// Second order accurate kernels.
    Second,
    /// Fourth order accurate kernels.
    Fourth,
}

impl Order {
    pub const fn from_value(val: usize) -> Self {
        match val {
            2 => Order::Second,
            4 => Order::Fourth,
            _ => panic!("Unknown Order"),
        }
    }
}

/// The specific operation to be approximated.
#[derive(Clone, Debug, Copy)]
pub enum BasisOperator {
    /// Identity operator.
    Value,
    /// Approximation of derivative.
    Derivative,
    /// Approximation of second derivative.
    SecondDerivative,
}

impl BasisOperator {
    /// The number of points on either side of the interior stencil's support.
    pub const fn border(self, order: Order) -> usize {
        match (self, order) {
            (Self::Value, Order::Second) => 0,
            (Self::Derivative, Order::Second) => 1,
            (Self::SecondDerivative, Order::Second) => 1,

            (Self::Value, Order::Fourth) => 0,
            (Self::Derivative, Order::Fourth) => 2,
            (Self::SecondDerivative, Order::Fourth) => 2,
        }
    }

    /// Returns the weights necessary to approximate an operator to the given order
    /// at a point with the given support.
    pub const fn weights(self, order: Order, support: Support) -> &'static [f64] {
        match (self, order) {
            (Self::Value, Order::Second) => &[1.0],
            (Self::Derivative, Order::Second) => match support {
                Support::Interior => &derivative!(1, 1, 0),
                Support::Negative(_) => &derivative!(0, 2, 0),
                Support::Positive(_) => &derivative!(2, 0, 0),
            },
            (Self::SecondDerivative, Order::Second) => match support {
                Support::Interior => &second_derivative!(1, 1, 0),
                Support::Negative(_) => &second_derivative!(0, 3, 0),
                Support::Positive(_) => &second_derivative!(3, 0, 0),
            },

            (Self::Value, Order::Fourth) => &[1.0],
            (Self::Derivative, Order::Fourth) => match support {
                Support::Interior => &derivative!(2, 2, 0),
                Support::Negative(0) => &derivative!(0, 4, 0),
                Support::Negative(_) => &derivative!(0, 4, 1),
                Support::Positive(0) => &derivative!(4, 0, 0),
                Support::Positive(_) => &derivative!(4, 0, -1),
            },
            (Self::SecondDerivative, Order::Fourth) => match support {
                Support::Interior => &second_derivative!(2, 2, 0),
                Support::Negative(0) => &second_derivative!(0, 5, 0),
                Support::Negative(_) => &second_derivative!(0, 5, 1),
                Support::Positive(0) => &second_derivative!(5, 0, 0),
                Support::Positive(_) => &second_derivative!(5, 0, -1),
            },
        }
    }

    /// An overall scaling factor given the grid spacing along this axis.
    pub fn scale(self, spacing: f64) -> f64 {
        match self {
            Self::Value => 1.0,
            Self::Derivative => 1.0 / spacing,
            Self::SecondDerivative => 1.0 / (spacing * spacing),
        }
    }
}

/// Represents an interpolation operator.
pub struct Interpolation;

impl Interpolation {
    pub const fn border(order: Order) -> usize {
        match order {
            Order::Second => 2,
            Order::Fourth => 3,
        }
    }

    pub const fn weights(order: Order, support: Support) -> &'static [f64] {
        match order {
            Order::Second => match support {
                Support::Interior => &[-1.0, 9.0, 9.0, -1.0],
                Support::Negative(_) => &[5.0, 15.0, -5.0, 1.0],
                Support::Positive(_) => &[1.0, -5.0, 15.0, 5.0],
            },
            Order::Fourth => match support {
                Support::Interior => &[3.0, -25.0, 150.0, 150.0, -25.0, 3.0],
                Support::Negative(1) => &[63.0, 315.0, -210.0, 126.0, -45.0, 7.0],
                Support::Negative(_) => &[-7.0, 105.0, 210.0, -70.0, 21.0, -3.0],
                Support::Positive(1) => &[7.0, -45.0, 126.0, -210.0, 315.0, 63.0],
                Support::Positive(_) => &[-3.0, 21.0, -70.0, 210.0, 105.0, -7.0],
            },
        }
    }

    pub fn scale(order: Order) -> f64 {
        match order {
            Order::Second => 1.0 / 16.0,
            Order::Fourth => 1.0 / 256.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_weights() {
        let derivative = BasisOperator::Derivative;
        assert_eq!(
            derivative.weights(Order::Second, Support::Negative(0)),
            [-1.5, 2.0, -0.5]
        );
        assert_eq!(
            derivative.weights(Order::Second, Support::Interior),
            [-0.5, 0.0, 0.5]
        );
        assert_eq!(
            derivative.weights(Order::Second, Support::Positive(0)),
            [0.5, -2.0, 1.5]
        );

        assert_eq!(
            derivative.weights(Order::Fourth, Support::Negative(0)),
            [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0]
        );
        assert_eq!(
            derivative.weights(Order::Fourth, Support::Interior),
            [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]
        );
        assert_eq!(
            derivative.weights(Order::Fourth, Support::Positive(0)),
            [1.0 / 4.0, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0]
        );

        let second_derivative = BasisOperator::SecondDerivative;
        assert_eq!(
            second_derivative.weights(Order::Second, Support::Negative(0)),
            [2.0, -5.0, 4.0, -1.0]
        );
        assert_eq!(
            second_derivative.weights(Order::Second, Support::Interior),
            [1.0, -2.0, 1.0]
        );
        assert_eq!(
            second_derivative.weights(Order::Second, Support::Positive(0)),
            [-1.0, 4.0, -5.0, 2.0]
        );

        assert_eq!(
            second_derivative.weights(Order::Fourth, Support::Interior),
            [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
        );
    }
}
