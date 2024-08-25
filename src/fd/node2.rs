// use crate::fd::{vertex::Support, BasisOperator, Interpolation, Order};
// use crate::geometry::{faces, CartesianIter, IndexSpace};
// use crate::prelude::{Face, Rectangle};
// use std::array::{self, from_fn};

// use super::boundary::{Boundary, Condition, Domain, DomainWithBC};
// use super::kernel2::{Border, Convolution};
// use super::{BoundaryKind, Dissipation, VertexSpace, BC};

// /// A uniform rectangular domain of nodes to which
// /// various derivative and interpolation kernels can be
// /// applied.
// #[derive(Debug, Clone)]
// pub struct NodeSpace<const N: usize, D> {
//     /// Number of cells along each axis (one less than then number of vertices).
//     size: [usize; N],
//     ghost: usize,
//     context: D,
// }

// impl<const N: usize, D> NodeSpace<N, D> {
//     pub fn new(size: [usize; N], ghost: usize, context: D) -> Self {
//         Self {
//             size,
//             ghost,
//             context,
//         }
//     }

//     pub fn cell_size(&self) -> [usize; N] {
//         self.size
//     }

//     /// Returns the spacing along each axis of the node space.
//     pub fn spacing(&self, bounds: Rectangle<N>) -> [f64; N] {
//         from_fn(|axis| bounds.size[axis] / self.size[axis] as f64)
//     }

//     /// Computes the position of the given vertex.
//     pub fn position(&self, bounds: Rectangle<N>, node: [isize; N]) -> [f64; N] {
//         let spacing: [_; N] = from_fn(|axis| bounds.size[axis] / self.size[axis] as f64);

//         let mut result = [0.0; N];

//         for i in 0..N {
//             result[i] = bounds.origin[i] + spacing[i] * node[i] as f64;
//         }

//         result
//     }

//     pub fn vertex_size(&self) -> [usize; N] {
//         array::from_fn(|axis| self.size[axis] + 1)
//     }

//     pub fn node_size(&self) -> [usize; N] {
//         array::from_fn(|axis| self.size[axis] + 1 + 2 * self.ghost)
//     }

//     pub fn stencil_space(&self) -> VertexSpace<N> {
//         VertexSpace::new(self.node_size())
//     }

//     pub fn vertex_from_node(&self, node: [isize; N]) -> [usize; N] {
//         array::from_fn(|axis| {
//             let mut vertex = node[axis];

//             // if self.context.kind(Face::negative(axis)).has_ghost() {
//             vertex += self.ghost as isize;
//             // }

//             vertex as usize
//         })
//     }

//     pub fn index_from_node(&self, node: [isize; N]) -> usize {
//         self.vertex_space()
//             .index_from_vertex(self.vertex_from_node(node))
//     }

//     pub fn support(&self, node: [isize; N], border_width: usize, axis: usize) -> Support {
//         self.vertex_space()
//             .support(self.vertex_from_node(node), border_width, axis)
//     }
// }

// impl<const N: usize, D: Clone> NodeSpace<N, D> {
//     pub fn map_context<E>(&self, f: impl FnOnce(D) -> E) -> NodeSpace<N, E> {
//         NodeSpace {
//             size: self.size,
//             ghost: self.ghost,
//             context: f(self.context.clone()),
//         }
//     }
// }

// impl<const N: usize, D: Boundary<N>> NodeSpace<N, D> {
//     pub fn attach_condition<E>(&self, condition: E) -> NodeSpace<N, BC<D, E>> {
//         self.map_context(|boundary| BC::new(boundary, condition))
//     }
// }

// impl<const N: usize, D: Boundary<N> + Condition<N>> NodeSpace<N, D> {
//     pub fn stencils<'a, C: Convolution<N>>(
//         &self,
//         support: [Support; N],
//         convolution: &'a C,
//     ) -> [&'a [f64]; N] {
//         array::from_fn(move |axis| {
//             let border = match support[axis] {
//                 Support::Negative(border) => Border::Negative(border),
//                 Support::Positive(border) => Border::Positive(border),
//                 Support::Interior => {
//                     return convolution.interior(axis);
//                 }
//             };

//             let face = Face {
//                 axis,
//                 side: border.side(),
//             };

//             let kind = self.context.kind(face);
//             let parity = self.context.parity(face);

//             match (kind, parity) {
//                 (BoundaryKind::Parity, false) => convolution.antisymmetric(border, axis),
//                 (BoundaryKind::Parity, true) => convolution.symmetric(border, axis),
//                 (BoundaryKind::Custom, _) => convolution.interior(axis),
//                 (BoundaryKind::Free | BoundaryKind::Radiative, _) => convolution.free(border, axis),
//             }
//         })
//     }
// }
