use crate::fd::{BasisOperator, Interpolation, Order};
use crate::geometry::{faces, CartesianIter, IndexSpace};
use crate::prelude::{Face, Rectangle};
use std::array::{self, from_fn};

use super::boundary::{Boundary, BoundaryKind, Condition};
use super::kernel2::{Border, Convolution};
use super::BC;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Support {
    Interior,
    Negative(usize),
    Positive(usize),
}

/// A uniform rectangular domain of nodes to which
/// various derivative and interpolation kernels can be
/// applied.
#[derive(Debug, Clone)]
pub struct NodeSpace<const N: usize, D> {
    /// Number of cells along each axis (one less than then number of vertices).
    size: [usize; N],
    ghost: usize,
    boundary: D,
}

impl<const N: usize, D> NodeSpace<N, D> {
    pub fn new(size: [usize; N], ghost: usize, boundary: D) -> Self {
        Self {
            size,
            ghost,
            boundary,
        }
    }

    pub fn cell_size(&self) -> [usize; N] {
        self.size
    }

    pub fn vertex_size(&self) -> [usize; N] {
        array::from_fn(|axis| self.size[axis] + 1)
    }

    pub fn node_size(&self) -> [usize; N] {
        array::from_fn(|axis| self.size[axis] + 1 + 2 * self.ghost)
    }

    /// Returns the spacing along each axis of the node space.
    pub fn spacing(&self, bounds: Rectangle<N>) -> [f64; N] {
        from_fn(|axis| bounds.size[axis] / self.size[axis] as f64)
    }

    /// Computes the position of the given vertex.
    pub fn position(&self, bounds: Rectangle<N>, node: [isize; N]) -> [f64; N] {
        let spacing: [_; N] = from_fn(|axis| bounds.size[axis] / self.size[axis] as f64);

        let mut result = [0.0; N];

        for i in 0..N {
            result[i] = bounds.origin[i] + spacing[i] * node[i] as f64;
        }

        result
    }

    pub fn index_from_node(&self, node: [isize; N]) -> usize {
        let cartesian = array::from_fn(|axis| {
            let mut vertex = node[axis];

            // if self.context.kind(Face::negative(axis)).has_ghost() {
            vertex += self.ghost as isize;
            // }

            vertex as usize
        });

        IndexSpace::new(self.node_size()).linear_from_cartesian(cartesian)
    }

    pub fn apply(&self, corner: [isize; N], stencils: [&[f64]; N], field: &[f64]) -> f64 {
        let ssize: [_; N] = array::from_fn(|axis| stencils[axis].len());

        let mut result = 0.0;

        for offset in IndexSpace::new(ssize).iter() {
            let mut weight = 1.0;

            for axis in 0..N {
                weight *= stencils[axis][offset[axis]];
            }

            let index =
                self.index_from_node(array::from_fn(|axis| corner[axis] + offset[axis] as isize));
            result += field[index] * weight;
        }

        result
    }

    pub fn apply_axis(
        &self,
        corner: [isize; N],
        stencil: &[f64],
        axis: usize,
        field: &[f64],
    ) -> f64 {
        let mut stencils: [&[f64]; N] = [&[1.0]; N];
        stencils[axis] = stencil;

        self.apply(corner, stencils, field)
    }

    pub fn evaluate_interior(
        &self,
        node: [usize; N],
        convolution: impl Convolution<N>,
        field: &[f64],
    ) -> f64 {
        let stencils = array::from_fn(|axis| convolution.interior(axis));
        let corner = array::from_fn(|axis| (node[axis] - convolution.border_width(axis)) as isize);
        self.apply(corner, stencils, field)
    }
}

impl<const N: usize, D: Clone> NodeSpace<N, D> {
    pub fn map_boundary<E>(&self, f: impl FnOnce(D) -> E) -> NodeSpace<N, E> {
        NodeSpace {
            size: self.size,
            ghost: self.ghost,
            boundary: f(self.boundary.clone()),
        }
    }
}

impl<const N: usize, D: Boundary<N>> NodeSpace<N, D> {
    pub fn attach_condition<E>(&self, condition: E) -> NodeSpace<N, BC<D, E>> {
        self.map_boundary(|boundary| BC::new(boundary, condition))
    }

    pub fn support(&self, node: [usize; N], border_width: usize, axis: usize) -> Support {
        debug_assert!(self.ghost >= border_width);

        let has_negative = self.boundary.kind(Face::negative(axis)).has_ghost();
        let has_positive = self.boundary.kind(Face::positive(axis)).has_ghost();

        if !has_negative && node[axis] < border_width {
            Support::Negative(node[axis])
        } else if !has_positive && node[axis] >= self.size[axis] + 1 - border_width {
            Support::Positive(self.size[axis] - 1 - node[axis])
        } else {
            Support::Interior
        }
    }
}

impl<const N: usize, D: Boundary<N> + Condition<N>> NodeSpace<N, D> {
    pub fn stencils<'a, C: Convolution<N>>(
        &self,
        support: [Support; N],
        convolution: &'a C,
    ) -> [&'a [f64]; N] {
        array::from_fn(move |axis| {
            let border = match support[axis] {
                Support::Negative(border) => Border::Negative(border),
                Support::Positive(border) => Border::Positive(border),
                Support::Interior => {
                    return convolution.interior(axis);
                }
            };

            let face = Face {
                axis,
                side: border.side(),
            };

            let kind = self.boundary.kind(face);
            let parity = self.boundary.parity(face);

            match (kind, parity) {
                (BoundaryKind::Parity, false) => convolution.antisymmetric(border, axis),
                (BoundaryKind::Parity, true) => convolution.symmetric(border, axis),
                (BoundaryKind::Custom, _) => convolution.interior(axis),
                (BoundaryKind::Free | BoundaryKind::Radiative, _) => convolution.free(border, axis),
            }
        })
    }

    pub fn evaluate(
        &self,
        node: [usize; N],
        convolution: impl Convolution<N>,
        field: &[f64],
    ) -> f64 {
        let support =
            array::from_fn(|axis| self.support(node, convolution.border_width(axis), axis));
        let stencils = self.stencils(support, &convolution);
        let corner = array::from_fn(|axis| match support[axis] {
            Support::Interior => node[axis] as isize - convolution.border_width(axis) as isize,
            Support::Negative(_) => 0,
            Support::Positive(_) => (self.size[axis] + 1) as isize - stencils[axis].len() as isize,
        });
        self.apply(corner, stencils, field)
    }
}
