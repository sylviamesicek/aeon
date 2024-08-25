// use crate::{fd::node2::NodeSpace, geometry::Rectangle};
// use std::array;

// use crate::fd::{Boundary, Condition, Order, BC};

// use super::{kernel2::{Convolution, Derivative, Kernel, Single}, vertex::Support};

// /// An interface for computing values, gradients, and hessians of fields.
// pub trait Engine<const N: usize> {
//     fn position(&self) -> [f64; N];
//     fn node(&self) -> [isize; N];
//     fn value(&self, field: &[f64]) -> f64;
//     fn derivative<C: Condition<N>>(&self, cond: C, field: &[f64], i: usize) -> f64;
//     fn hessian<C: Condition<N>>(&self, cond: C, field: &[f64], i: usize, j: usize) -> f64;
//     fn dissipation<C: Condition<N>>(&self, cond: C, field: &[f64]) -> f64;
// }

// /// A finite difference engine of a given order, but potentially bordering a free boundary.
// pub struct FdEngine<const N: usize, const ORDER: usize, B> {
//     space: NodeSpace<N, B>,
//     node: [isize; N],
//     bounds: Rectangle<N>,
//     support: [Support; N],
// }

// impl<const N: usize, const ORDER: usize, B: Boundary<N>> FdEngine<N, ORDER, B> {
//     pub fn new(space: NodeSpace<N, B>, node: [isize; N], bounds: Rectangle<N>) -> Self {
//         let support = array::from_fn(|axis| space.support(node, ORDER / 2, axis))

//         Self {
//             space,
//             node,
//             bounds,
//             support,
//         }
//     }

// }

// impl<const N: usize, const ORDER: usize, B: Boundary<N>> Engine<N> for FdEngine<N, ORDER, B>
// {
//     fn position(&self) -> [f64; N] {
//         self.space.position(self.bounds.clone(), self.node)
//     }

//     fn node(&self) -> [isize; N] {
//         self.node
//     }

//     fn value(&self, field: &[f64]) -> f64 {
//         let index = self.space.index_from_node(self.node);
//         field[index]
//     }

//     fn derivative<C: Condition<N>>(&self, cond: C, field: &[f64], i: usize) -> f64 {
//         let convolution = Single(Derivative::<ORDER>, i);

//         let spacing = self.space.spacing(self.bounds);
//         let scale = convolution.scale(spacing);
//         let support = self.space.support(node, border_width, axis)

//         let stencils = self.space.attach_condition(cond).stencils(support, convolution);
//     }

//     fn evaluate(&self, field: &[f64], convolution: impl Convolution<N>) -> f64 {
//         let support = array::from_fn(|axis| match convolution.border_width(axis)  {
//             0 => Support::Interior,
//             n if n == ORDER / 2 => self.support[axis],
//             n => self.space.support(self.node, convolution.border_width(axis) , axis)
//         });
//         let spacing = self.space.spacing(self.bounds);
//         let scale = convolution.scale(spacing);

//         let stencils = self.space.
//     }
// }

// /// A finite difference engine that only every relies on interior support (and can thus use better optimized stencils).
// pub struct FdIntEngine<const N: usize, const ORDER: usize> {
//     pub space: NodeSpace<N, Rectangle<N>>,
//     pub vertex: [usize; N],
// }

// impl<const N: usize, const ORDER: usize> Engine<N> for FdIntEngine<N, ORDER> {
//     fn position(&self) -> [f64; N] {
//         self.space.position(node_from_vertex(self.vertex))
//     }

//     fn vertex(&self) -> [usize; N] {
//         self.vertex
//     }

//     fn value(&self, field: &[f64]) -> f64 {
//         let linear = self.space.index_from_vertex(self.vertex);
//         field[linear]
//     }

//     fn gradient<C: Condition<N>>(&self, _cond: C, field: &[f64]) -> [f64; N] {
//         let spacing = self.space.spacing();

//         let (weights, border) = const {
//             let order = Order::from_value(ORDER);
//             let weights = BasisOperator::Derivative.weights(order, super::Support::Interior);
//             let border = BasisOperator::Derivative.border(order);

//             (weights, border)
//         };

//         array::from_fn(|axis| {
//             let mut corner = node_from_vertex(self.vertex);
//             corner[axis] -= border as isize;

//             self.space.weights_axis(corner, weights, axis, field) / spacing[axis]
//         })
//     }

//     fn hessian<C: Condition<N>>(&self, _cond: C, field: &[f64]) -> [[f64; N]; N] {
//         let spacing = self.space.spacing();

//         let (dweights, dborder) = const {
//             let order = Order::from_value(ORDER);
//             let weights = BasisOperator::Derivative.weights(order, super::Support::Interior);
//             let border = BasisOperator::Derivative.border(order);

//             (weights, border)
//         };

//         let (ddweights, ddborder) = const {
//             let order = Order::from_value(ORDER);
//             let weights = BasisOperator::SecondDerivative.weights(order, super::Support::Interior);
//             let border = BasisOperator::SecondDerivative.border(order);

//             (weights, border)
//         };

//         let mut result = [[0.0; N]; N];

//         for i in 0..N {
//             for j in i..N {
//                 if i == j {
//                     let mut corner = node_from_vertex(self.vertex);
//                     corner[i] -= ddborder as isize;
//                     result[i][j] = self.space.weights_axis(corner, ddweights, i, field);
//                     result[i][j] /= spacing[i] * spacing[j];
//                 } else {
//                     let mut corner = node_from_vertex(self.vertex);
//                     corner[i] -= dborder as isize;
//                     corner[j] -= dborder as isize;

//                     let mut weights: [&'static [f64]; N] = [&[1.0]; N];
//                     weights[i] = dweights;
//                     weights[j] = dweights;

//                     result[i][j] = self.space.weights(corner, weights, field);
//                     result[i][j] /= spacing[i] * spacing[j];
//                     result[j][i] = result[i][j]
//                 }
//             }
//         }

//         result
//     }
// }
