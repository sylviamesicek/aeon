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

// ************************************
// Order ******************************
// ************************************

/// Associates an order with a type. Commonly used to set the order of accuracy for certain
/// operators or boundary conditions.
pub trait Order: 'static {
    const ORDER: usize;
    const BORDER: usize = Self::ORDER / 2;
}

/// Declares a struct which implements order
macro_rules! OrderStruct {
    ($name:ident<$order:literal>) => {
        pub struct $name;

        impl Order for $name {
            const ORDER: usize = $order;
        }
    };
}

OrderStruct! { SecondOrder<2> }
OrderStruct! { FourthOrder<2> }
OrderStruct! { SixthOrder<2> }

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

/// Derivative operation of a given order.
struct Derivative<O: Order>(PhantomData<O>);

impl<O: Order> Derivative<O> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<O: Order> Clone for Derivative<O> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<O: Order> Kernel for Derivative<O> {
    fn border_width(&self) -> usize {
        O::BORDER
    }

    fn interior(&self) -> &[f64] {
        if O::ORDER == 2 {
            &derivative!(1, 1, 0)
        } else if O::ORDER == 4 {
            &derivative!(2, 2, 0)
        } else {
            unimplemented!()
        }
    }

    fn free(&self, border: Border) -> &[f64] {
        if O::ORDER == 2 {
            match border {
                Border::Negative(_) => &derivative!(0, 2, 0),
                Border::Positive(_) => &derivative!(2, 0, 0),
            }
        } else if O::ORDER == 4 {
            match border {
                Border::Negative(0) => &derivative!(0, 4, 0),
                Border::Negative(_) => &derivative!(0, 4, 1),
                Border::Positive(0) => &derivative!(4, 0, 0),
                Border::Positive(_) => &derivative!(4, 0, -1),
            }
        } else {
            unimplemented!()
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        if O::ORDER == 2 {
            &[0.0]
        } else if O::ORDER == 4 {
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
        } else {
            unimplemented!()
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        if O::ORDER == 2 {
            match border {
                Border::Negative(_) => &[0.0, 1.0],
                Border::Positive(_) => &[1.0, 0.0],
            }
        } else if O::ORDER == 4 {
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
        } else {
            unimplemented!()
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / spacing
    }
}

/// Second derivative operation of a given order.
struct SecondDerivative<O: Order>(PhantomData<O>);

impl<O: Order> SecondDerivative<O> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<O: Order> Clone for SecondDerivative<O> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<O: Order> Kernel for SecondDerivative<O> {
    fn border_width(&self) -> usize {
        O::BORDER
    }

    fn interior(&self) -> &[f64] {
        if O::ORDER == 2 {
            &second_derivative!(1, 1, 0)
        } else if O::ORDER == 4 {
            &second_derivative!(2, 2, 0)
        } else {
            unimplemented!()
        }
    }

    fn free(&self, border: Border) -> &[f64] {
        if O::ORDER == 2 {
            match border {
                Border::Negative(_) => &second_derivative!(0, 3, 0),
                Border::Positive(_) => &second_derivative!(3, 0, 0),
            }
        } else if O::ORDER == 4 {
            match border {
                Border::Negative(0) => &second_derivative!(0, 5, 0),
                Border::Negative(_) => &second_derivative!(0, 5, 1),
                Border::Positive(0) => &second_derivative!(5, 0, 0),
                Border::Positive(_) => &second_derivative!(5, 0, -1),
            }
        } else {
            unimplemented!()
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        if O::ORDER == 2 {
            match border {
                Border::Negative(_) => &[-2.0, 2.0],
                Border::Positive(_) => &[2.0, -2.0],
            }
        } else if O::ORDER == 4 {
            const NEG_0: &'static [f64] = &[-5.0 / 2.0, 8.0 / 3.0, -2.0 / 12.0];
            const NEG_1: &'static [f64] =
                &[4.0 / 3.0, -5.0 / 2.0 - 1.0 / 12.0, 4.0 / 3.0, -1.0 / 12.0];
            const POS_0: &'static [f64] = &[-2.0 / 12.0, 8.0 / 3.0, -5.0 / 2.0];
            const POS_1: &'static [f64] =
                &[-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0 - 1.0 / 12.0, 4.0 / 3.0];

            match border {
                Border::Negative(0) => NEG_0,
                Border::Negative(_) => NEG_1,
                Border::Positive(0) => POS_0,
                Border::Positive(_) => POS_1,
            }
        } else {
            unimplemented!()
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        if O::ORDER == 2 {
            &[0.0]
        } else if O::ORDER == 4 {
            const NEG_0: &'static [f64] = &[0.0];
            const NEG_1: &'static [f64] =
                &[4.0 / 3.0, -5.0 / 2.0 + 1.0 / 12.0, 4.0 / 3.0, -1.0 / 12.0];
            const POS_0: &'static [f64] = &[0.0];
            const POS_1: &'static [f64] =
                &[-1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0 + 1.0 / 12.0, 4.0 / 3.0];

            match border {
                Border::Negative(0) => NEG_0,
                Border::Negative(_) => NEG_1,
                Border::Positive(0) => POS_0,
                Border::Positive(_) => POS_1,
            }
        } else {
            unimplemented!()
        }
    }

    fn scale(&self, spacing: f64) -> f64 {
        1.0 / (spacing * spacing)
    }
}

/// Dissipation operation of a given order.
struct Dissipation<O: Order>(PhantomData<O>);

impl<O: Order> Dissipation<O> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<O: Order> Clone for Dissipation<O> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<O: Order> Kernel for Dissipation<O> {
    fn border_width(&self) -> usize {
        O::BORDER
    }

    fn interior(&self) -> &[f64] {
        if O::ORDER == 4 {
            &[1.0, -4.0, 6.0, -4.0, 1.0]
        } else if O::ORDER == 6 {
            &[1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0]
        } else {
            unimplemented!()
        }
    }

    fn free(&self, border: Border) -> &[f64] {
        if O::ORDER == 4 {
            match border {
                Border::Negative(0) => &[3.0, -14.0, 26.0, -24.0, 11.0, -2.0],
                Border::Negative(_) => &[2.0, -9.0, 16.0, -14.0, 6.0, -1.0],
                Border::Positive(0) => &[-2.0, 11.0, -24.0, 26.0, -14.0, 3.0],
                Border::Positive(_) => &[-1.0, 6.0, -14.0, 16.0, -9.0, 2.0],
            }
        } else if O::ORDER == 6 {
            match border {
                Border::Negative(0) => &[4.0, -27.0, 78.0, -125.0, 120.0, -69.0, 22.0, -3.0],
                Border::Negative(1) => &[3.0, -20.0, 57.0, -90.0, 85.0, -48.0, 15.0, -2.0],
                Border::Negative(_) => &[2.0, -13.0, 36.0, -55.0, 50.0, -27.0, 8.0, -1.0],
                Border::Positive(0) => &[-3.0, 22.0, -69.0, 120.0, -125.0, 78.0, -27.0, 4.0],
                Border::Positive(1) => &[-2.0, 15.0, -48.0, 85.0, -90.0, 57.0, -20.0, 3.0],
                Border::Positive(_) => &[-1.0, 8.0, -27.0, 50.0, -55.0, 36.0, -13.0, 2.0],
            }
        } else {
            unimplemented!()
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        if O::ORDER == 4 {
            match border {
                Border::Negative(0) => &[6.0, -8.0, 2.0],
                Border::Negative(_) => &[-4.0, 7.0, -4.0, 1.0],
                Border::Positive(0) => &[2.0, -8.0, 6.0],
                Border::Positive(_) => &[1.0, -4.0, 7.0, -4.0],
            }
        } else if O::ORDER == 6 {
            match border {
                Border::Negative(0) => &[-20.0, 30.0, -12.0, 2.0],
                Border::Negative(1) => &[15.0, -26.0, 16.0, -6.0, 1.0],
                Border::Negative(_) => &[-6.0, 16.0, -20.0, 15.0, -6.0, 1.0],
                Border::Positive(0) => &[2.0, -12.0, 30.0, -20.0],
                Border::Positive(1) => &[1.0, -6.0, 16.0, -26.0, 15.0],
                Border::Positive(_) => &[1.0, -6.0, 15.0, -20.0, 16.0, -6.0],
            }
        } else {
            unimplemented!()
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        if O::ORDER == 4 {
            match border {
                Border::Negative(0) => &[0.0],
                Border::Negative(_) => &[-4.0, 5.0, -4.0, 1.0],
                Border::Positive(0) => &[0.0],
                Border::Positive(_) => &[1.0, -4.0, 5.0, -4.0],
            }
        } else if O::ORDER == 6 {
            match border {
                Border::Negative(0) => &[0.0],
                Border::Negative(1) => &[15.0, -14.0, 14.0, -6.0, 1.0],
                Border::Negative(_) => &[-6.0, 14.0, -20.0, 15.0, -6.0, 1.0],
                Border::Positive(0) => &[0.0],
                Border::Positive(1) => &[1.0, -6.0, 14.0, -14.0, 15.0],
                Border::Positive(_) => &[1.0, -6.0, 15.0, -20.0, 14.0, -6.0],
            }
        } else {
            unimplemented!()
        }
    }

    fn scale(&self, _spacing: f64) -> f64 {
        if O::ORDER == 4 {
            -1.0 / 16.0
        } else if O::ORDER == 6 {
            1.0 / 64.0
        } else {
            unimplemented!()
        }
    }
}

/// Interpolating operator of a given order.
struct Interpolation<O: Order>(PhantomData<O>);

impl<O: Order> Interpolation<O> {
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<O: Order> Clone for Interpolation<O> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<O: Order> Kernel for Interpolation<O> {
    fn border_width(&self) -> usize {
        O::BORDER
    }

    fn interior(&self) -> &[f64] {
        if O::ORDER == 2 {
            &[-1.0, 9.0, 9.0, -1.0]
        } else if O::ORDER == 4 {
            &[3.0, -25.0, 150.0, 150.0, -25.0, 3.0]
        } else {
            unimplemented!()
        }
    }

    fn free(&self, border: Border) -> &[f64] {
        if O::ORDER == 2 {
            match border {
                Border::Positive(_) => &[1.0, -5.0, 15.0, 5.0],
                Border::Negative(_) => &[5.0, 15.0, -5.0, 1.0],
            }
        } else if O::ORDER == 4 {
            match border {
                Border::Negative(1) => &[63.0, 315.0, -210.0, 126.0, -45.0, 7.0],
                Border::Negative(_) => &[-7.0, 105.0, 210.0, -70.0, 21.0, -3.0],
                Border::Positive(1) => &[7.0, -45.0, 126.0, -210.0, 315.0, 63.0],
                Border::Positive(_) => &[-3.0, 21.0, -70.0, 210.0, 105.0, -7.0],
            }
        } else {
            unimplemented!()
        }
    }

    fn symmetric(&self, border: Border) -> &[f64] {
        if O::ORDER == 2 {
            match border {
                Border::Negative(_) => &[9.0, 8.0, -1.0],
                Border::Positive(_) => &[-1.0, 8.0, 9.0],
            }
        } else if O::ORDER == 4 {
            match border {
                Border::Negative(1) => &[150.0, 125.0, -22.0, 3.0],
                Border::Negative(_) => &[-25.0, 153.0, 150.0, -25.0, 3.0],
                Border::Positive(1) => &[3.0, -22.0, 125.0, 150.0],
                Border::Positive(_) => &[3.0, -25.0, 150.0, 153.0, -25.0],
            }
        } else {
            unimplemented!()
        }
    }

    fn antisymmetric(&self, border: Border) -> &[f64] {
        if O::ORDER == 2 {
            match border {
                Border::Negative(_) => &[9.0, 10.0, -1.0],
                Border::Positive(_) => &[-1.0, 10.0, 9.0],
            }
        } else if O::ORDER == 4 {
            match border {
                Border::Negative(1) => &[150.0, 175.0, -28.0, 3.0],
                Border::Negative(_) => &[-25.0, 147.0, 150.0, -25.0, 3.0],
                Border::Positive(1) => &[3.0, -28.0, 175.0, 150.0],
                Border::Positive(_) => &[3.0, -25.0, 150.0, 147.0, -28.0],
            }
        } else {
            unimplemented!()
        }
    }

    fn scale(&self, _spacing: f64) -> f64 {
        if O::ORDER == 2 {
            1.0 / 16.0
        } else if O::ORDER == 6 {
            1.0 / 256.0
        } else {
            unimplemented!()
        }
    }
}

impl<O: Order> ProlongKernel for Interpolation<O> {}

mod private {
    pub trait Sealed {}
}

impl<T: Order> private::Sealed for T {}

/// Associates kernels for common operations with any type that implements Order.
pub trait Kernels: private::Sealed + 'static {
    fn derivative() -> &'static impl Kernel;
    fn second_derivative() -> &'static impl Kernel;
    fn dissipation() -> &'static impl Kernel;
    fn interpolation() -> &'static impl ProlongKernel;
}

impl<T: Order> Kernels for T {
    fn derivative() -> &'static impl Kernel {
        &Derivative::<T>(PhantomData)
    }

    fn second_derivative() -> &'static impl Kernel {
        &SecondDerivative::<T>(PhantomData)
    }

    fn dissipation() -> &'static impl Kernel {
        &Dissipation::<T>(PhantomData)
    }

    fn interpolation() -> &'static impl ProlongKernel {
        &Interpolation::<T>(PhantomData)
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
pub struct Gradient<O: Order>(usize, PhantomData<O>);

impl<O: Order> Gradient<O> {
    /// Constructs a convolution which computes the derivative along the given axis.
    pub const fn new(axis: usize) -> Self {
        Self(axis, PhantomData)
    }
}

impl<O: Order> Clone for Gradient<O> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<const N: usize, O: Order> Convolution<N> for Gradient<O> {
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
pub struct Hessian<O: Order>(usize, usize, PhantomData<O>);

impl<O: Order> Hessian<O> {
    /// Constructs a convolution which computes the given entry of the hessian matrix.
    pub fn new(i: usize, j: usize) -> Self {
        Self(i, j, PhantomData)
    }
}

impl<O: Order> Hessian<O> {
    fn is_second(&self, axis: usize) -> bool {
        self.0 == self.1 && axis == self.0
    }

    fn is_first(&self, axis: usize) -> bool {
        self.0 == axis || self.1 == axis
    }
}

impl<const N: usize, O: Order> Convolution<N> for Hessian<O> {
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
