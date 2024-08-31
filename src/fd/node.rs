use aeon_geometry::IndexWindow;

use crate::geometry::{faces, CartesianIter, IndexSpace};
use crate::prelude::{Face, Rectangle};
use std::array::{self, from_fn};

use super::boundary::{Boundary, BoundaryKind, Condition};
use super::kernel::{Border, Convolution, Kernel, OffsetKernel, Value};
use super::BC;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Support {
    Interior,
    Negative(usize),
    Positive(usize),
}

pub fn vertex_from_node<const N: usize>(node: [isize; N]) -> [usize; N] {
    array::from_fn(|axis| node[axis] as usize)
}

pub fn node_from_vertex<const N: usize>(vertex: [usize; N]) -> [isize; N] {
    array::from_fn(|axis| vertex[axis] as isize)
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

    pub fn size(&self) -> [usize; N] {
        self.size
    }

    pub fn ghost(&self) -> usize {
        self.ghost
    }

    pub fn inner_size(&self) -> [usize; N] {
        array::from_fn(|axis| self.size[axis] + 1)
    }

    pub fn full_size(&self) -> [usize; N] {
        array::from_fn(|axis| self.size[axis] + 1 + 2 * self.ghost)
    }

    pub fn num_nodes(&self) -> usize {
        self.full_size().iter().product()
    }

    /// Returns the spacing along each axis of the node space.
    pub fn spacing(&self, bounds: Rectangle<N>) -> [f64; N] {
        from_fn(|axis| bounds.size[axis] / self.size[axis] as f64)
    }

    /// Computes the position of the given vertex.
    pub fn position(&self, node: [isize; N], bounds: Rectangle<N>) -> [f64; N] {
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

        IndexSpace::new(self.full_size()).linear_from_cartesian(cartesian)
    }

    pub fn index_from_vertex(&self, vertex: [usize; N]) -> usize {
        self.index_from_node(node_from_vertex(vertex))
    }

    // pub fn value(&self, node: [isize; N], field: &[f64]) -> f64 {
    //     let index = self.index_from_node(node);
    //     field[index]
    // }

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
        vertex: [usize; N],
        convolution: impl Convolution<N>,
        field: &[f64],
    ) -> f64 {
        let stencils = array::from_fn(|axis| convolution.interior(axis));
        let corner =
            array::from_fn(|axis| (vertex[axis] - convolution.border_width(axis)) as isize);
        self.apply(corner, stencils, field)
    }

    pub fn vertices(&self) -> IndexWindow<N> {
        IndexWindow {
            origin: [0; N],
            size: self.inner_size(),
        }
    }

    /// Returns a window of just the interior and edges of the node space (no ghost nodes).
    pub fn inner_window(&self) -> NodeWindow<N> {
        NodeWindow {
            origin: [0isize; N],
            size: self.inner_size(),
        }
    }

    /// Returns the window which encompasses the whole node space
    pub fn full_window(&self) -> NodeWindow<N> {
        let mut origin = [0; N];

        for axis in 0..N {
            // if self.domain.kind(Face::negative(axis)).has_ghost() {
            origin[axis] = -(self.ghost as isize);
            // }
        }

        NodeWindow {
            origin,
            size: self.full_size(),
        }
    }

    pub fn attach_boundary<E>(&self, f: E) -> NodeSpace<N, E> {
        NodeSpace {
            size: self.size,
            ghost: self.ghost,
            boundary: f,
        }
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

    pub fn support(&self, vertex: [usize; N], border_width: usize, axis: usize) -> Support {
        debug_assert!(self.ghost >= border_width);

        let has_negative = self.boundary.kind(Face::negative(axis)).has_ghost();
        let has_positive = self.boundary.kind(Face::positive(axis)).has_ghost();

        if !has_negative && vertex[axis] < border_width {
            Support::Negative(vertex[axis])
        } else if !has_positive && vertex[axis] >= self.size[axis] + 1 - border_width {
            Support::Positive(self.size[axis] - 1 - vertex[axis])
        } else {
            Support::Interior
        }
    }

    pub fn support_cell(&self, node: [usize; N], border_width: usize, axis: usize) -> Support {
        if node[axis] < border_width.saturating_sub(1)
            && !self.boundary.kind(Face::negative(axis)).has_ghost()
        {
            let right = node[axis] + 1;
            Support::Negative(right)
        } else if node[axis] > self.size[axis] - border_width
            && !self.boundary.kind(Face::positive(axis)).has_ghost()
        {
            let left = self.size[axis] - node[axis];
            debug_assert!(left > 0);
            Support::Positive(left)
        } else {
            Support::Interior
        }
    }
}

impl<const N: usize, D: Boundary<N> + Condition<N>> NodeSpace<N, D> {
    /// Set strongly enforced boundary conditions.
    pub fn fill_boundary(&self, dest: &mut [f64]) {
        let vertex_size = self.inner_size();
        let active_window = self.inner_window();

        // Loop over faces
        for face in faces::<N>() {
            let axis = face.axis;
            let side = face.side;

            // As well as strongly enforce any diritchlet boundary conditions on axis.
            let intercept = if side { vertex_size[axis] - 1 } else { 0 } as isize;

            // Iterate over face
            for node in active_window.plane(axis, intercept) {
                if self.boundary.kind(face) == BoundaryKind::Parity
                    && self.boundary.parity(face) == false
                {
                    // For antisymmetric boundaries we set all values on axis to be 0.
                    let index = self.index_from_node(node);
                    dest[index] = 0.0;
                }
            }
        }
    }

    fn stencils<'a, C: Convolution<N>>(
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
        vertex: [usize; N],
        convolution: impl Convolution<N>,
        field: &[f64],
    ) -> f64 {
        let support =
            array::from_fn(|axis| self.support(vertex, convolution.border_width(axis), axis));
        let stencils = self.stencils(support, &convolution);
        let corner = array::from_fn(|axis| match support[axis] {
            Support::Interior => vertex[axis] as isize - convolution.border_width(axis) as isize,
            Support::Negative(_) => 0,
            Support::Positive(_) => (self.size[axis] + 1) as isize - stencils[axis].len() as isize,
        });
        self.apply(corner, stencils, field)
    }

    pub fn prolong(&self, supernode: [usize; N], kernel: impl OffsetKernel, field: &[f64]) -> f64 {
        let node: [_; N] = array::from_fn(|axis| supernode[axis] / 2);
        let flags: [_; N] = array::from_fn(|axis| supernode[axis] % 2 == 0);

        let border_width: [_; N] = array::from_fn(|axis| {
            if flags[axis] {
                kernel.border_width()
            } else {
                Value.border_width()
            }
        });
        let support: [_; N] =
            array::from_fn(|axis| self.support_cell(node, border_width[axis], axis));

        let stencils = array::from_fn(|axis| {
            if !flags[axis] {
                return Value.interior();
            }

            let border = match support[axis] {
                Support::Interior => return kernel.interior(),
                Support::Negative(i) => Border::Negative(i),
                Support::Positive(i) => Border::Positive(i),
            };

            let face = Face {
                axis,
                side: border.side(),
            };

            let kind = self.boundary.kind(face);
            let parity = self.boundary.parity(face);

            match (kind, parity) {
                (BoundaryKind::Parity, false) => kernel.antisymmetric(border),
                (BoundaryKind::Parity, true) => kernel.symmetric(border),
                (BoundaryKind::Custom, _) => kernel.interior(),
                (BoundaryKind::Free | BoundaryKind::Radiative, _) => kernel.free(border),
            }
        });

        let corner = array::from_fn(|axis| {
            if !flags[axis] {
                return node[axis] as isize;
            }

            match support[axis] {
                Support::Interior => node[axis] as isize - border_width[axis] as isize + 1,
                Support::Negative(_) => 0,
                Support::Positive(_) => (self.size[axis] + 1 - stencils[axis].len()) as isize,
            }
        });
        self.apply(corner, stencils, field)
    }
}

// ****************************
// Node Window ****************
// ****************************

/// Defines a rectagular region of a larger `NodeSpace`.
#[derive(Debug, Clone)]
pub struct NodeWindow<const N: usize> {
    pub origin: [isize; N],
    pub size: [usize; N],
}

impl<const N: usize> NodeWindow<N> {
    /// Iterate over all nodes in the window.
    pub fn iter(&self) -> NodeCartesianIter<N> {
        NodeCartesianIter::new(self.origin, self.size)
    }

    /// Iterate over nodes on a plane in the window.
    pub fn plane(&self, axis: usize, intercept: isize) -> NodePlaneIter<N> {
        debug_assert!(
            intercept >= self.origin[axis]
                && intercept < self.size[axis] as isize + self.origin[axis]
        );

        let mut size = self.size;
        size[axis] = 1;

        NodePlaneIter {
            axis,
            intercept,
            inner: NodeCartesianIter::new(self.origin, size),
        }
    }
}

impl<const N: usize> IntoIterator for NodeWindow<N> {
    type IntoIter = NodeCartesianIter<N>;
    type Item = [isize; N];

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A helper for iterating over a node window.
pub struct NodeCartesianIter<const N: usize> {
    origin: [isize; N],
    inner: CartesianIter<N>,
}

impl<const N: usize> NodeCartesianIter<N> {
    pub fn new(origin: [isize; N], size: [usize; N]) -> Self {
        Self {
            origin,
            inner: IndexSpace::new(size).iter(),
        }
    }
}

impl<const N: usize> Iterator for NodeCartesianIter<N> {
    type Item = [isize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.inner.next()?;

        let mut result = self.origin;

        for axis in 0..N {
            result[axis] += index[axis] as isize;
        }

        Some(result)
    }
}

/// A helper for iterating over a plane in a node window.
pub struct NodePlaneIter<const N: usize> {
    axis: usize,
    intercept: isize,
    inner: NodeCartesianIter<N>,
}

impl<const N: usize> Iterator for NodePlaneIter<N> {
    type Item = [isize; N];

    fn next(&mut self) -> Option<Self::Item> {
        let mut index = self.inner.next()?;
        index[self.axis] = self.intercept;
        Some(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        fd::kernel::{Derivative, Interpolation},
        geometry::Rectangle,
    };

    #[derive(Clone)]
    struct Quadrant;

    impl<const N: usize> Boundary<N> for Quadrant {
        fn kind(&self, face: Face<N>) -> BoundaryKind {
            match face.side {
                false => BoundaryKind::Parity,
                true => BoundaryKind::Free,
            }
        }
    }

    impl<const N: usize> Condition<N> for Quadrant {
        fn parity(&self, _face: Face<N>) -> bool {
            false
        }
    }

    #[test]
    fn support_vertex() {
        let space = NodeSpace {
            size: [8],
            ghost: 2,
            boundary: Quadrant,
        };

        const BORDER: usize = 3;

        let supports = [
            Support::Negative(0),
            Support::Negative(1),
            Support::Negative(2),
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Positive(2),
            Support::Positive(1),
            Support::Positive(0),
        ];

        for i in 0..9 {
            assert_eq!(space.support([i], BORDER, 0), supports[i]);
        }
    }

    fn eval_convergence(size: [usize; 2]) -> f64 {
        let space = NodeSpace {
            size,
            ghost: 2,
            boundary: Quadrant,
        };
        let bounds = Rectangle::UNIT;
        let spacing = space.spacing(bounds);

        let mut field = vec![0.0; space.num_nodes()];

        for node in space.full_window().iter() {
            let index = space.index_from_node(node);
            let [x, y] = space.position(node, bounds.clone());
            field[index] = x.sin() * y.sin();
        }

        let mut result: f64 = 0.0;

        for node in space.inner_window().iter() {
            let vertex = [node[0] as usize, node[1] as usize];
            let [x, y] = space.position(node, bounds);
            let numerical = space.evaluate(vertex, (Derivative::<4>, Derivative::<4>), &field)
                / (spacing[0] * spacing[1]);
            let analytical = x.cos() * y.cos();
            // dbg!(numerical - analytical);
            let error: f64 = (numerical - analytical).abs();
            result = result.max(error);
        }

        result
    }

    #[test]
    fn evaluate() {
        let error16 = eval_convergence([16, 16]);
        let error32 = eval_convergence([32, 32]);
        let error64 = eval_convergence([64, 64]);

        assert!(error16 / error32 >= 16.0);
        assert!(error32 / error64 >= 16.0);
    }

    fn prolong_convergence(size: [usize; 2]) -> f64 {
        let cspace = NodeSpace {
            size,
            ghost: 2,
            boundary: Quadrant,
        };
        let rspace = NodeSpace {
            size: size.map(|v| v * 2),
            ghost: 2,
            boundary: Quadrant,
        };

        let bounds = Rectangle::UNIT;

        let mut field = vec![0.0; cspace.num_nodes()];

        for node in cspace.full_window().iter() {
            let index = cspace.index_from_node(node);
            let [x, y] = cspace.position(node, bounds.clone());
            field[index] = x.sin() * y.sin();
        }

        let mut result: f64 = 0.0;

        for node in rspace.inner_window().iter() {
            let vertex = [node[0] as usize, node[1] as usize];
            let [x, y] = rspace.position(node, bounds.clone());
            let numerical = cspace.prolong(vertex, Interpolation::<4>, &field);
            let analytical = x.sin() * y.sin();
            let error: f64 = (numerical - analytical).abs();
            result = result.max(error);
        }

        result
    }

    #[test]
    fn prolong() {
        let error16 = prolong_convergence([16, 16]);
        let error32 = prolong_convergence([32, 32]);
        let error64 = prolong_convergence([64, 64]);

        assert!(error16 / error32 >= 32.0);
        assert!(error32 / error64 >= 32.0);
    }
}
