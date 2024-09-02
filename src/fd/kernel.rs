use std::marker::PhantomData;

use aeon_macros::{derivative, second_derivative};

/// Distance of a vertex from a boundary.
#[derive(Clone, Copy, Debug)]
pub enum Border {
    Negative(usize),
    Positive(usize),
}

impl Border {
    /// Returns false for negative borders and true for positive borders.
    pub fn side(self) -> bool {
        match self {
            Border::Negative(_) => false,
            Border::Positive(_) => true,
        }
    }
}

// *********************************
// Kernel **************************
// *********************************

pub trait Kernel: Clone {
    fn border_width(&self) -> usize;

    fn interior(&self) -> &[f64];
    fn free(&self, border: Border) -> &[f64];
    fn symmetric(&self, border: Border) -> &[f64];
    fn antisymmetric(&self, border: Border) -> &[f64];

    fn scale(&self, spacing: f64) -> f64;
}

/// A kernel which is used for prolonging values between levels.
pub trait ProlongKernel: Kernel {}

/// Value operation.
#[derive(Clone)]
pub struct Value;

impl Kernel for Value {
    fn border_width(&self) -> usize {
        0
    }

    fn interior(&self) -> &[f64] {
        &[1.0]
    }

    fn free(&self, _border: Border) -> &[f64] {
        &[1.0]
    }

    fn antisymmetric(&self, _border: Border) -> &[f64] {
        &[0.0]
    }

    fn symmetric(&self, _border: Border) -> &[f64] {
        &[1.0]
    }

    fn scale(&self, _spacing: f64) -> f64 {
        1.0
    }
}

#[derive(Clone)]
pub struct Unimplemented(usize);

impl Kernel for Unimplemented {
    fn border_width(&self) -> usize {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn interior(&self) -> &[f64] {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn free(&self, _border: Border) -> &[f64] {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn symmetric(&self, _border: Border) -> &[f64] {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn antisymmetric(&self, _border: Border) -> &[f64] {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }

    fn scale(&self, _spacing: f64) -> f64 {
        unimplemented!("Kernel is unimplemented for order {}", self.0)
    }
}

impl ProlongKernel for Unimplemented {}

/// Derivative operation of a given order.
#[derive(Clone)]
pub struct Derivative<const ORDER: usize>;

impl Kernel for Derivative<2> {
    fn border_width(&self) -> usize {
        1
    }

    fn interior(&self) -> &[f64] {
        &derivative!(1, 1, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(_) => &derivative!(0, 2, 0),
            Border::Positive(_) => &derivative!(2, 0, 0),
        }
    }

    fn symmetric(&self, _border: Border) -> &[f64] {
        &[0.0]
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(_) => &[0.0, 1.0],
            Border::Positive(_) => &[1.0, 0.0],
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / spacing
    }
}

impl Kernel for Derivative<4> {
    fn border_width(&self) -> usize {
        2
    }

    fn interior(&self) -> &[f64] {
        &derivative!(2, 2, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &derivative!(0, 4, 0),
            Border::Negative(_) => &derivative!(0, 4, 1),
            Border::Positive(0) => &derivative!(4, 0, 0),
            Border::Positive(_) => &derivative!(4, 0, -1),
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        const NEG_0: &'static [f64] = &[0.0];
        const NEG_1: &'static [f64] = &[-2.0 / 3.0, 1.0 / 12.0, 2.0 / 3.0, -1.0 / 12.0];
        const POS_0: &'static [f64] = &[0.0];
        const POS_1: &'static [f64] = &[1.0 / 12.0, -2.0 / 3.0, -1.0 / 12.0, 2.0 / 3.0];

        match border {
            Border::Negative(0) => NEG_0,
            Border::Negative(_) => NEG_1,
            Border::Positive(0) => POS_0,
            Border::Positive(_) => POS_1,
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        const NEG_0: &'static [f64] = &[0.0, 4.0 / 3.0, -2.0 / 12.0];
        const NEG_1: &'static [f64] = &[-2.0 / 3.0, -1.0 / 12.0, 2.0 / 3.0, -1.0 / 12.0];
        const POS_0: &'static [f64] = &[2.0 / 12.0, -4.0 / 3.0, 0.0];
        const POS_1: &'static [f64] = &[1.0 / 12.0, -2.0 / 3.0, 1.0 / 12.0, 2.0 / 3.0];

        match border {
            Border::Negative(0) => NEG_0,
            Border::Negative(_) => NEG_1,
            Border::Positive(0) => POS_0,
            Border::Positive(_) => POS_1,
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / spacing
    }
}

/// Second derivative operator of a given order.
#[derive(Clone)]
pub struct SecondDerivative<const ORDER: usize>;

impl Kernel for SecondDerivative<2> {
    fn border_width(&self) -> usize {
        1
    }

    fn interior(&self) -> &[f64] {
        &second_derivative!(1, 1, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(_) => &second_derivative!(0, 3, 0),
            Border::Positive(_) => &second_derivative!(3, 0, 0),
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(_) => &[-2.0, 2.0],
            Border::Positive(_) => &[2.0, -2.0],
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(_) => &[0.0],
            Border::Positive(_) => &[0.0],
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

impl Kernel for SecondDerivative<4> {
    fn border_width(&self) -> usize {
        2
    }

    fn interior(&self) -> &[f64] {
        &second_derivative!(2, 2, 0)
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &second_derivative!(0, 5, 0),
            Border::Negative(_) => &second_derivative!(0, 5, 1),
            Border::Positive(0) => &second_derivative!(5, 0, 0),
            Border::Positive(_) => &second_derivative!(5, 0, -1),
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        const NEG_0: &'static [f64] = &[-5.0 / 2.0, 8.0 / 3.0, -2.0 / 12.0];
        const NEG_1: &'static [f64] = &[4.0 / 3.0, -5.0 / 2.0 - 1.0 / 12.0, 4.0 / 3.0, -1.0 / 12.0];
        const POS_0: &'static [f64] = &[-2.0 / 12.0, 8.0 / 3.0, -5.0 / 2.0];
        const POS_1: &'static [f64] = &[-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0 - 1.0 / 12.0, 4.0 / 3.0];

        match border {
            Border::Negative(0) => NEG_0,
            Border::Negative(_) => NEG_1,
            Border::Positive(0) => POS_0,
            Border::Positive(_) => POS_1,
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        const NEG_0: &'static [f64] = &[0.0];
        const NEG_1: &'static [f64] = &[4.0 / 3.0, -5.0 / 2.0 + 1.0 / 12.0, 4.0 / 3.0, -1.0 / 12.0];
        const POS_0: &'static [f64] = &[0.0];
        const POS_1: &'static [f64] = &[-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0 + 1.0 / 12.0, 4.0 / 3.0];

        match border {
            Border::Negative(0) => NEG_0,
            Border::Negative(_) => NEG_1,
            Border::Positive(0) => POS_0,
            Border::Positive(_) => POS_1,
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

/// Kriss Olgier dissipation of the given order.
#[derive(Clone)]
pub struct Dissipation<const ORDER: usize>;

impl Kernel for Dissipation<4> {
    fn border_width(&self) -> usize {
        2
    }

    fn interior(&self) -> &[f64] {
        &[1.0, -4.0, 6.0, -4.0, 1.0]
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &[3.0, -14.0, 26.0, -24.0, 11.0, -2.0],
            Border::Negative(_) => &[2.0, -9.0, 16.0, -14.0, 6.0, -1.0],
            Border::Positive(0) => &[-2.0, 11.0, -24.0, 26.0, -14.0, 3.0],
            Border::Positive(_) => &[-1.0, 6.0, -14.0, 16.0, -9.0, 2.0],
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &[6.0, -8.0, 2.0],
            Border::Negative(_) => &[-4.0, 7.0, -4.0, 1.0],
            Border::Positive(0) => &[2.0, -8.0, 6.0],
            Border::Positive(_) => &[1.0, -4.0, 7.0, -4.0],
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &[0.0],
            Border::Negative(_) => &[-4.0, 5.0, -4.0, 1.0],
            Border::Positive(0) => &[0.0],
            Border::Positive(_) => &[1.0, -4.0, 5.0, -4.0],
        }
    }

    fn scale(&self, _spacing: f64) -> f64 {
        -1.0 / 16.0
    }
}

impl Kernel for Dissipation<6> {
    fn border_width(&self) -> usize {
        3
    }

    fn interior(&self) -> &[f64] {
        &[1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &[4.0, -27.0, 78.0, -125.0, 120.0, -69.0, 22.0, -3.0],
            Border::Negative(1) => &[3.0, -20.0, 57.0, -90.0, 85.0, -48.0, 15.0, -2.0],
            Border::Negative(_) => &[2.0, -13.0, 36.0, -55.0, 50.0, -27.0, 8.0, -1.0],
            Border::Positive(0) => &[-3.0, 22.0, -69.0, 120.0, -125.0, 78.0, -27.0, 4.0],
            Border::Positive(1) => &[-2.0, 15.0, -48.0, 85.0, -90.0, 57.0, -20.0, 3.0],
            Border::Positive(_) => &[-1.0, 8.0, -27.0, 50.0, -55.0, 36.0, -13.0, 2.0],
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &[-20.0, 30.0, -12.0, 2.0],
            Border::Negative(1) => &[15.0, -26.0, 16.0, -6.0, 1.0],
            Border::Negative(_) => &[-6.0, 16.0, -20.0, 15.0, -6.0, 1.0],
            Border::Positive(0) => &[2.0, -12.0, 30.0, -20.0],
            Border::Positive(1) => &[1.0, -6.0, 16.0, -26.0, 15.0],
            Border::Positive(_) => &[1.0, -6.0, 15.0, -20.0, 16.0, -6.0],
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(0) => &[0.0],
            Border::Negative(1) => &[15.0, -14.0, 14.0, -6.0, 1.0],
            Border::Negative(_) => &[-6.0, 14.0, -20.0, 15.0, -6.0, 1.0],
            Border::Positive(0) => &[0.0],
            Border::Positive(1) => &[1.0, -6.0, 14.0, -14.0, 15.0],
            Border::Positive(_) => &[1.0, -6.0, 15.0, -20.0, 14.0, -6.0],
        }
    }

    fn scale(&self, _spacing: f64) -> f64 {
        1.0 / 64.0
    }
}

#[derive(Clone)]
pub struct Interpolation<const ORDER: usize>;

impl Kernel for Interpolation<2> {
    fn border_width(&self) -> usize {
        2
    }

    fn interior(&self) -> &[f64] {
        &[-1.0, 9.0, 9.0, -1.0]
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Positive(_) => &[1.0, -5.0, 15.0, 5.0],
            Border::Negative(_) => &[5.0, 15.0, -5.0, 1.0],
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(_) => &[9.0, 8.0, -1.0],
            Border::Positive(_) => &[-1.0, 8.0, 9.0],
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(_) => &[9.0, 10.0, -1.0],
            Border::Positive(_) => &[-1.0, 10.0, 9.0],
        }
    }

    fn scale(&self, _spacing: f64) -> f64 {
        1.0 / 16.0
    }
}

impl Kernel for Interpolation<4> {
    fn border_width(&self) -> usize {
        3
    }

    fn interior(&self) -> &[f64] {
        &[3.0, -25.0, 150.0, 150.0, -25.0, 3.0]
    }

    fn free(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(1) => &[63.0, 315.0, -210.0, 126.0, -45.0, 7.0],
            Border::Negative(_) => &[-7.0, 105.0, 210.0, -70.0, 21.0, -3.0],
            Border::Positive(1) => &[7.0, -45.0, 126.0, -210.0, 315.0, 63.0],
            Border::Positive(_) => &[-3.0, 21.0, -70.0, 210.0, 105.0, -7.0],
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(1) => &[150.0, 125.0, -22.0, 3.0],
            Border::Negative(_) => &[-25.0, 153.0, 150.0, -25.0, 3.0],
            Border::Positive(1) => &[3.0, -22.0, 125.0, 150.0],
            Border::Positive(_) => &[3.0, -25.0, 150.0, 153.0, -25.0],
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        match border {
            Border::Negative(1) => &[150.0, 175.0, -28.0, 3.0],
            Border::Negative(_) => &[-25.0, 147.0, 150.0, -25.0, 3.0],
            Border::Positive(1) => &[3.0, -28.0, 175.0, 150.0],
            Border::Positive(_) => &[3.0, -25.0, 150.0, 147.0, -28.0],
        }
    }

    fn scale(&self, _spacing: f64) -> f64 {
        1.0 / 256.0
    }
}

impl<const ORDER: usize> ProlongKernel for Interpolation<ORDER> where Interpolation<ORDER>: Kernel {}

// ************************************
// Order ******************************
// ************************************

#[derive(Clone, Copy, Default)]
pub struct Order<const ORDER: usize>;

pub type SecondOrder = Order<2>;
pub type FourthOrder = Order<4>;
pub type SixthOrder = Order<6>;

mod private {
    pub trait Sealed {}
}

impl private::Sealed for Order<2> {}
impl private::Sealed for Order<4> {}
impl private::Sealed for Order<6> {}

/// Associates an order with a type. Commonly used to set the order of accuracy for certain
/// operators or boundary conditions.
pub trait Kernels: private::Sealed + Clone + Copy + Default + 'static {
    const ORDER: usize;
    const MAX_BORDER: usize;

    fn derivative() -> &'static impl Kernel;
    fn second_derivative() -> &'static impl Kernel;
    fn dissipation() -> &'static impl Kernel;
    fn interpolation() -> &'static impl ProlongKernel;
}

impl Kernels for SecondOrder {
    const ORDER: usize = 2;
    const MAX_BORDER: usize = 1;

    fn derivative() -> &'static impl Kernel {
        &Derivative::<2>
    }

    fn second_derivative() -> &'static impl Kernel {
        &SecondDerivative::<2>
    }

    fn dissipation() -> &'static impl Kernel {
        &Unimplemented(2)
    }

    fn interpolation() -> &'static impl ProlongKernel {
        &Interpolation::<2>
    }
}

impl Kernels for FourthOrder {
    const ORDER: usize = 4;
    const MAX_BORDER: usize = 2;

    fn derivative() -> &'static impl Kernel {
        &Derivative::<4>
    }

    fn second_derivative() -> &'static impl Kernel {
        &SecondDerivative::<4>
    }

    fn dissipation() -> &'static impl Kernel {
        &Unimplemented(4)
    }

    fn interpolation() -> &'static impl ProlongKernel {
        &Interpolation::<4>
    }
}

impl Kernels for SixthOrder {
    const ORDER: usize = 6;
    const MAX_BORDER: usize = 3;

    fn derivative() -> &'static impl Kernel {
        &Unimplemented(6)
    }

    fn second_derivative() -> &'static impl Kernel {
        &Unimplemented(6)
    }

    fn dissipation() -> &'static impl Kernel {
        &Dissipation::<6>
    }

    fn interpolation() -> &'static impl ProlongKernel {
        &Unimplemented(6)
    }
}

// ************************************
// Convolution ************************
// ************************************

/// A N-dimensional tensor product of several seperable kernels.
pub trait Convolution<const N: usize> {
    fn border_width(&self, axis: usize) -> usize;

    fn interior(&self, axis: usize) -> &[f64];
    fn free(&self, border: Border, axis: usize) -> &[f64];
    fn symmetric(&self, border: Border, axis: usize) -> &[f64];
    fn antisymmetric(&self, border: Border, axis: usize) -> &[f64];

    fn scale(&self, spacing: [f64; N]) -> f64;
}

macro_rules! impl_convolution_for_tuples {
    ($($N:literal => $($T:ident $i:tt)*)*) => {
        $(
            impl<$($T: Kernel,)*> Convolution<$N> for ($($T,)*) {
                fn border_width(&self, axis: usize) -> usize {
                    match axis {
                        $($i => self.$i.border_width(),)*
                        _ => panic!("Invalid Axis")
                    }
                }

                fn interior(&self, axis: usize) -> &[f64] {
                    match axis {
                        $($i => self.$i.interior(),)*
                        _ => panic!("Invalid Axis")
                    }
                }

                fn free(&self, border: Border, axis: usize) -> &[f64] {
                    match axis {
                        $($i => self.$i.free(border),)*
                        _ => panic!("Invalid Axis")
                    }
                }

                fn symmetric(&self, border: Border, axis: usize) -> &[f64] {
                    match axis {
                        $($i => self.$i.symmetric(border),)*
                        _ => panic!("Invalid Axis")
                    }
                }


                fn antisymmetric(&self, border: Border, axis: usize) -> &[f64] {
                    match axis {
                        $($i => self.$i.antisymmetric(border),)*
                        _ => panic!("Invalid Axis")
                    }
                }

                fn scale(&self, spacing: [f64; $N]) -> f64 {
                    let mut result = 1.0;

                    $(
                        result *= self.$i.scale(spacing[$i]);
                    )*

                    result
                }
            }
        )*

    };
}

impl_convolution_for_tuples! {
   1 => K0 0
   2 => K0 0 K1 1
   3 => K0 0 K1 1 K2 2
   4 => K0 0 K1 1 K2 2 K3 3
}

/// Computes the gradient along the given axis.
pub struct Gradient<O: Kernels>(usize, PhantomData<O>);

impl<O: Kernels> Gradient<O> {
    /// Constructs a convolution which computes the derivative along the given axis.
    pub const fn new(axis: usize) -> Self {
        Self(axis, PhantomData)
    }
}

impl<O: Kernels> Clone for Gradient<O> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<const N: usize, O: Kernels> Convolution<N> for Gradient<O> {
    fn border_width(&self, axis: usize) -> usize {
        if axis == self.0 {
            O::derivative().border_width()
        } else {
            Value.border_width()
        }
    }

    fn interior(&self, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::derivative().interior()
        } else {
            Value.interior()
        }
    }

    fn free(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::derivative().free(border)
        } else {
            Value.free(border)
        }
    }

    fn symmetric(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::derivative().symmetric(border)
        } else {
            Value.symmetric(border)
        }
    }

    fn antisymmetric(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::derivative().antisymmetric(border)
        } else {
            Value.antisymmetric(border)
        }
    }

    fn scale(&self, spacing: [f64; N]) -> f64 {
        1.0 / spacing[self.0]
    }
}

/// Computes the mixed derivative of the given axes.
pub struct Hessian<O: Kernels>(usize, usize, PhantomData<O>);

impl<O: Kernels> Hessian<O> {
    /// Constructs a convolution which computes the given entry of the hessian matrix.
    pub fn new(i: usize, j: usize) -> Self {
        Self(i, j, PhantomData)
    }
}

impl<O: Kernels> Hessian<O> {
    fn is_second(&self, axis: usize) -> bool {
        self.0 == self.1 && axis == self.0
    }

    fn is_first(&self, axis: usize) -> bool {
        self.0 == axis || self.1 == axis
    }
}

impl<const N: usize, O: Kernels> Convolution<N> for Hessian<O> {
    fn border_width(&self, axis: usize) -> usize {
        if self.is_second(axis) {
            O::second_derivative().border_width()
        } else if self.is_first(axis) {
            O::derivative().border_width()
        } else {
            Value.border_width()
        }
    }

    fn interior(&self, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            O::second_derivative().interior()
        } else if self.is_first(axis) {
            O::derivative().interior()
        } else {
            Value.interior()
        }
    }

    fn free(&self, border: Border, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            O::second_derivative().free(border)
        } else if self.is_first(axis) {
            O::derivative().free(border)
        } else {
            Value.free(border)
        }
    }

    fn symmetric(&self, border: Border, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            O::second_derivative().symmetric(border)
        } else if self.is_first(axis) {
            O::derivative().symmetric(border)
        } else {
            Value.symmetric(border)
        }
    }

    fn antisymmetric(&self, border: Border, axis: usize) -> &[f64] {
        if self.is_second(axis) {
            O::second_derivative().antisymmetric(border)
        } else if self.is_first(axis) {
            O::derivative().antisymmetric(border)
        } else {
            Value.antisymmetric(border)
        }
    }

    fn scale(&self, spacing: [f64; N]) -> f64 {
        1.0 / (spacing[self.0] * spacing[self.1])
    }
}

/// Computes the gradient along the given axis.
pub struct DissipationAxis<O: Kernels>(usize, PhantomData<O>);

impl<O: Kernels> DissipationAxis<O> {
    /// Constructs a convolution which computes the derivative along the given axis.
    pub const fn new(axis: usize) -> Self {
        Self(axis, PhantomData)
    }
}

impl<O: Kernels> Clone for DissipationAxis<O> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<const N: usize, O: Kernels> Convolution<N> for DissipationAxis<O> {
    fn border_width(&self, axis: usize) -> usize {
        if axis == self.0 {
            O::dissipation().border_width()
        } else {
            Value.border_width()
        }
    }

    fn interior(&self, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::dissipation().interior()
        } else {
            Value.interior()
        }
    }

    fn free(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::dissipation().free(border)
        } else {
            Value.free(border)
        }
    }

    fn symmetric(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::dissipation().symmetric(border)
        } else {
            Value.symmetric(border)
        }
    }

    fn antisymmetric(&self, border: Border, axis: usize) -> &[f64] {
        if axis == self.0 {
            O::dissipation().antisymmetric(border)
        } else {
            Value.antisymmetric(border)
        }
    }

    fn scale(&self, spacing: [f64; N]) -> f64 {
        1.0 / spacing[self.0]
    }
}
