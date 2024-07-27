use aeon_macros::{derivative, second_derivative};

/// The support along an axis for a given point in node space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// The point has the necessary support on both sides.
    Interior,
    /// The point lies near a negative boundary, and must use a boundary support.
    FreeNegative(usize),
    /// The point lies near a positive boundary, and must use a boundary support.
    FreePositive(usize),

    SymNegative(usize),
    SymPositive(usize),

    AntiSymNegative(usize),
    AntiSymPositive(usize),
}

/// The order of accuracy of finite difference stencils.
#[derive(Debug, Clone, Copy)]
pub enum Order {
    /// Second order accurate kernels.
    Second,
    /// Fourth order accurate kernels.
    Fourth,
    /// Sixth order accurate kernels.
    Sixth,
}

impl Order {
    pub const fn from_value(val: usize) -> Self {
        match val {
            2 => Order::Second,
            4 => Order::Fourth,
            _ => panic!("Unknown Order"),
        }
    }

    pub const fn support(self) -> usize {
        match self {
            Self::Second => 1,
            Self::Fourth => 2,
            Self::Sixth => 3,
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

            _ => panic!("Unknown order"),
        }
    }

    /// Returns the weights necessary to approximate an operator to the given order
    /// at a point with the given support.
    pub const fn weights(self, order: Order, support: Support) -> &'static [f64] {
        const SYM_NEG_FIRST_0_4: &'static [f64] = &[0.0];
        const SYM_NEG_FIRST_1_4: &'static [f64] = &[-2.0 / 3.0, 1.0 / 12.0, 2.0 / 3.0, -1.0 / 12.0];
        const SYM_POS_FIRST_0_4: &'static [f64] = &[0.0];
        const SYM_POS_FIRST_1_4: &'static [f64] = &[-2.0 / 3.0, 1.0 / 12.0, 2.0 / 3.0, -1.0 / 12.0];
        const ASYM_NEG_FIRST_0_4: &'static [f64] = &[0.0, 4.0 / 3.0, -2.0 / 12.0];
        const ASYM_NEG_FIRST_1_4: &'static [f64] =
            &[-2.0 / 3.0, -1.0 / 12.0, 2.0 / 3.0, -1.0 / 12.0];
        const ASYM_POS_FIRST_0_4: &'static [f64] = &[2.0 / 12.0, -4.0 / 3.0, 0.0];
        const ASYM_POS_FIRST_1_4: &'static [f64] = &[1.0 / 12.0, -2.0 / 3.0, 1.0 / 12.0, 2.0 / 3.0];

        const SYM_NEG_SECOND_0_4: &'static [f64] = &[-5.0 / 2.0, 8.0 / 3.0, -2.0 / 12.0];
        const SYM_NEG_SECOND_1_4: &'static [f64] =
            &[4.0 / 3.0, -5.0 / 2.0 - 1.0 / 12.0, 4.0 / 3.0, -1.0 / 12.0];
        const SYM_POS_SECOND_0_4: &'static [f64] = &[-2.0 / 12.0, 8.0 / 3.0, -5.0 / 2.0];
        const SYM_POS_SECOND_1_4: &'static [f64] =
            &[-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0 - 1.0 / 12.0, 4.0 / 3.0];
        const ASYM_NEG_SECOND_0_4: &'static [f64] = &[0.0];
        const ASYM_NEG_SECOND_1_4: &'static [f64] =
            &[4.0 / 3.0, -5.0 / 2.0 + 1.0 / 12.0, 4.0 / 3.0, -1.0 / 12.0];
        const ASYM_POS_SECOND_0_4: &'static [f64] = &[0.0];
        const ASYM_POS_SECOND_1_4: &'static [f64] =
            &[-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0 + 1.0 / 12.0, 4.0 / 3.0];

        match (self, order) {
            (Self::Value, Order::Second) => &[1.0],
            (Self::Derivative, Order::Second) => match support {
                Support::Interior => &derivative!(1, 1, 0),
                Support::FreeNegative(_) => &derivative!(0, 2, 0),
                Support::FreePositive(_) => &derivative!(2, 0, 0),
                Support::SymNegative(_) => &[0.0],
                Support::SymPositive(_) => &[0.0],
                Support::AntiSymNegative(_) => &[0.0, 1.0],
                Support::AntiSymPositive(_) => &[1.0, 0.0],
            },
            (Self::SecondDerivative, Order::Second) => match support {
                Support::Interior => &second_derivative!(1, 1, 0),
                Support::FreeNegative(_) => &second_derivative!(0, 3, 0),
                Support::FreePositive(_) => &second_derivative!(3, 0, 0),
                Support::SymNegative(_) => &[-2.0, 2.0],
                Support::SymPositive(_) => &[2.0, -2.0],
                Support::AntiSymNegative(_) => &[0.0],
                Support::AntiSymPositive(_) => &[0.0],
            },

            (Self::Value, Order::Fourth) => &[1.0],
            (Self::Derivative, Order::Fourth) => match support {
                // [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]
                Support::Interior => &derivative!(2, 2, 0),
                Support::SymNegative(0) => SYM_NEG_FIRST_0_4,
                Support::SymNegative(_) => SYM_NEG_FIRST_1_4,
                Support::SymPositive(0) => SYM_POS_FIRST_0_4,
                Support::SymPositive(_) => SYM_POS_FIRST_1_4,
                Support::AntiSymNegative(0) => ASYM_NEG_FIRST_0_4,
                Support::AntiSymNegative(_) => ASYM_NEG_FIRST_1_4,
                Support::AntiSymPositive(0) => ASYM_POS_FIRST_0_4,
                Support::AntiSymPositive(_) => ASYM_POS_FIRST_1_4,
                Support::FreeNegative(0) => &derivative!(0, 4, 0),
                Support::FreeNegative(_) => &derivative!(0, 4, 1),
                Support::FreePositive(0) => &derivative!(4, 0, 0),
                Support::FreePositive(_) => &derivative!(4, 0, -1),
            },
            (Self::SecondDerivative, Order::Fourth) => match support {
                // [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
                Support::Interior => &second_derivative!(2, 2, 0),
                Support::SymNegative(0) => SYM_NEG_SECOND_0_4,
                Support::SymNegative(_) => SYM_NEG_SECOND_1_4,
                Support::SymPositive(0) => SYM_POS_SECOND_0_4,
                Support::SymPositive(_) => SYM_POS_SECOND_1_4,
                Support::AntiSymNegative(0) => ASYM_NEG_SECOND_0_4,
                Support::AntiSymNegative(_) => ASYM_NEG_SECOND_1_4,
                Support::AntiSymPositive(0) => ASYM_POS_SECOND_0_4,
                Support::AntiSymPositive(_) => ASYM_POS_SECOND_1_4,
                Support::FreeNegative(0) => &second_derivative!(0, 5, 0),
                Support::FreeNegative(_) => &second_derivative!(0, 5, 1),
                Support::FreePositive(0) => &second_derivative!(5, 0, 0),
                Support::FreePositive(_) => &second_derivative!(5, 0, -1),
            },
            _ => panic!("Unknown order"),
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

pub struct Dissipation;

impl Dissipation {
    pub const fn border(order: Order) -> usize {
        match order {
            Order::Second => 1,
            Order::Fourth => 2,
            Order::Sixth => 3,
        }
    }

    pub const fn weights(order: Order, support: Support) -> &'static [f64] {
        match order {
            Order::Second => unimplemented!(),
            Order::Fourth => match support {
                Support::Interior => &[1.0, -4.0, 6.0, -4.0, 1.0],
                Support::SymNegative(0) => &[6.0, -8.0, 2.0],
                Support::SymNegative(_) => &[-4.0, 7.0, -4.0, 1.0],
                Support::SymPositive(0) => &[2.0, -8.0, 6.0],
                Support::SymPositive(_) => &[1.0, -4.0, 7.0, -4.0],
                Support::AntiSymNegative(0) => &[0.0],
                Support::AntiSymNegative(_) => &[-4.0, 5.0, -4.0, 1.0],
                Support::AntiSymPositive(0) => &[0.0],
                Support::AntiSymPositive(_) => &[1.0, -4.0, 5.0, -4.0],
                Support::FreeNegative(0) => &[3.0, -14.0, 26.0, -24.0, 11.0, -2.0],
                Support::FreeNegative(_) => &[2.0, -9.0, 16.0, -14.0, 6.0, -1.0],
                Support::FreePositive(0) => &[-2.0, 11.0, -24.0, 26.0, -14.0, 3.0],
                Support::FreePositive(_) => &[-1.0, 6.0, -14.0, 16.0, -9.0, 2.0],
            },
            Order::Sixth => match support {
                Support::Interior => &[1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0],
                Support::SymNegative(0) => &[-20.0, 15.0, -6.0, 1.0],
                Support::SymNegative(1) => &[15.0, -26.0, 16.0, -6.0, 1.0],
                Support::SymNegative(_) => &[-6.0, 16.0, -20.0, 15.0, -6.0, 1.0],
                Support::SymPositive(0) => &[1.0, -6.0, 15.0, -20.0],
                Support::SymPositive(1) => &[1.0, -6.0, 16.0, -26.0, 15.0],
                Support::SymPositive(_) => &[1.0, -6.0, 15.0, -20.0, 16.0, -6.0],

                Support::AntiSymNegative(0) => &[0.0],
                Support::AntiSymNegative(1) => &[15.0, -14.0, 14.0, -6.0, 1.0],
                Support::AntiSymNegative(_) => &[-6.0, 14.0, -20.0, 15.0, -6.0, 1.0],
                Support::AntiSymPositive(0) => &[0.0],
                Support::AntiSymPositive(1) => &[1.0, -6.0, 14.0, -14.0, 15.0],
                Support::AntiSymPositive(_) => &[1.0, -6.0, 15.0, -20.0, 14.0, -6.0],

                Support::FreeNegative(0) => &[4.0, -27.0, 78.0, -125.0, 120.0, -69.0, 22.0, -3.0],
                Support::FreeNegative(1) => &[3.0, -20.0, 57.0, -90.0, 85.0, -48.0, 15.0, -2.0],
                Support::FreeNegative(_) => &[2.0, -13.0, 36.0, -55.0, 50.0, -27.0, 8.0, -1.0],
                Support::FreePositive(0) => &[-3.0, 22.0, -69.0, 120.0, -125.0, 78.0, -27.0, 4.0],
                Support::FreePositive(1) => &[-2.0, 15.0, -48.0, 85.0, -90.0, 57.0, -20.0, 3.0],
                Support::FreePositive(_) => &[-1.0, 8.0, -27.0, 50.0, -55.0, 36.0, -13.0, 2.0],
            },
        }
    }

    pub fn scale(order: Order) -> f64 {
        match order {
            Order::Second => unimplemented!(),
            Order::Fourth => -1.0 / 16.0,
            Order::Sixth => 1.0 / 64.0,
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
            Order::Sixth => panic!(),
        }
    }

    pub const fn weights(order: Order, support: Support) -> &'static [f64] {
        match order {
            Order::Second => match support {
                Support::Interior => &[-1.0, 9.0, 9.0, -1.0],
                Support::SymNegative(_) => &[9.0, 8.0, -1.0],
                Support::SymPositive(_) => &[-1.0, 8.0, 9.0],
                Support::AntiSymNegative(_) => &[9.0, 10.0, -1.0],
                Support::AntiSymPositive(_) => &[-1.0, 10.0, 9.0],
                Support::FreeNegative(_) => &[5.0, 15.0, -5.0, 1.0],
                Support::FreePositive(_) => &[1.0, -5.0, 15.0, 5.0],
            },
            Order::Fourth => match support {
                Support::Interior => &[3.0, -25.0, 150.0, 150.0, -25.0, 3.0],
                Support::SymNegative(1) => &[150.0, 125.0, -22.0, 3.0],
                Support::SymNegative(_) => &[-25.0, 153.0, 150.0, -25.0, 3.0],
                Support::SymPositive(1) => &[3.0, -22.0, 125.0, 150.0],
                Support::SymPositive(_) => &[3.0, -25.0, 150.0, 153.0, -25.0],
                Support::AntiSymNegative(1) => &[150.0, 175.0, -28.0, 3.0],
                Support::AntiSymNegative(_) => &[-25.0, 147.0, 150.0, -25.0, 3.0],
                Support::AntiSymPositive(1) => &[3.0, -28.0, 175.0, 150.0],
                Support::AntiSymPositive(_) => &[3.0, -25.0, 150.0, 147.0, -28.0],
                Support::FreeNegative(1) => &[63.0, 315.0, -210.0, 126.0, -45.0, 7.0],
                Support::FreeNegative(_) => &[-7.0, 105.0, 210.0, -70.0, 21.0, -3.0],
                Support::FreePositive(1) => &[7.0, -45.0, 126.0, -210.0, 315.0, 63.0],
                Support::FreePositive(_) => &[-3.0, 21.0, -70.0, 210.0, 105.0, -7.0],
            },
            Order::Sixth => panic!(),
        }
    }

    pub fn scale(order: Order) -> f64 {
        match order {
            Order::Second => 1.0 / 16.0,
            Order::Fourth => 1.0 / 256.0,
            Order::Sixth => panic!(),
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
            derivative.weights(Order::Second, Support::FreeNegative(0)),
            [-1.5, 2.0, -0.5]
        );
        assert_eq!(
            derivative.weights(Order::Second, Support::Interior),
            [-0.5, 0.0, 0.5]
        );
        assert_eq!(
            derivative.weights(Order::Second, Support::FreePositive(0)),
            [0.5, -2.0, 1.5]
        );

        assert_eq!(
            derivative.weights(Order::Fourth, Support::FreeNegative(0)),
            [-25.0 / 12.0, 4.0, -3.0, 4.0 / 3.0, -1.0 / 4.0]
        );
        assert_eq!(
            derivative.weights(Order::Fourth, Support::Interior),
            [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]
        );
        assert_eq!(
            derivative.weights(Order::Fourth, Support::FreePositive(0)),
            [1.0 / 4.0, -4.0 / 3.0, 3.0, -4.0, 25.0 / 12.0]
        );

        let second_derivative = BasisOperator::SecondDerivative;
        assert_eq!(
            second_derivative.weights(Order::Second, Support::FreeNegative(0)),
            [2.0, -5.0, 4.0, -1.0]
        );
        assert_eq!(
            second_derivative.weights(Order::Second, Support::Interior),
            [1.0, -2.0, 1.0]
        );
        assert_eq!(
            second_derivative.weights(Order::Second, Support::FreePositive(0)),
            [-1.0, 4.0, -5.0, 2.0]
        );

        assert_eq!(
            second_derivative.weights(Order::Fourth, Support::Interior),
            [-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0]
        );
    }
}
