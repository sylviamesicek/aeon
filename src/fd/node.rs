use crate::fd::{BasisOperator, Interpolation, Order, Support};
use crate::geometry::{faces, CartesianIter, IndexSpace, Rectangle};
use crate::prelude::Face;
use std::array::from_fn;

use super::boundary::{Boundary, Condition};
use super::BoundaryKind;

/// Transforms a vertex into a node (just casts an array of `usize` -> `isize`).
pub fn node_from_vertex<const N: usize>(vertex: [usize; N]) -> [isize; N] {
    let mut result = [0isize; N];

    for axis in 0..N {
        result[axis] = vertex[axis] as isize;
    }

    result
}

/// A uniform rectangular domain of nodes to which
/// various derivative and interpolation kernels can be
/// applied.
#[derive(Debug, Clone)]
pub struct NodeSpace<const N: usize> {
    /// Number of cells along each axis (one less than then number of vertices).
    pub size: [usize; N],
    /// Number of ghost vertices in each direction.
    pub ghost: usize,
}

impl<const N: usize> NodeSpace<N> {
    /// Computes the total number of nodes in the space.
    pub fn node_count(&self) -> usize {
        self.node_size().iter().product()
    }

    /// Converts a node into a linear index.
    pub fn index_from_node(&self, node: [isize; N]) -> usize {
        let cart = from_fn(|i| (node[i] + self.ghost as isize) as usize);
        IndexSpace::new(self.node_size()).linear_from_cartesian(cart)
    }

    pub fn index_from_vertex(&self, vertex: [usize; N]) -> usize {
        self.index_from_node(node_from_vertex(vertex))
    }

    /// Computes the number of cells along each axis.
    pub fn cell_size(&self) -> [usize; N] {
        self.size
    }

    /// Returns the number of vertices along each axis.
    pub fn vertex_size(&self) -> [usize; N] {
        let mut size = self.size;

        for s in size.iter_mut() {
            *s += 1;
        }

        size
    }

    /// Returns the total number of indices (including ghost indices) along each axis.
    pub fn node_size(&self) -> [usize; N] {
        let mut size = self.size;

        for s in size.iter_mut() {
            *s += 1 + 2 * self.ghost;
        }

        size
    }

    /// Returns the spacing along each axis of the node space.
    pub fn spacing(&self, bounds: Rectangle<N>) -> [f64; N] {
        from_fn(|axis| bounds.size[axis] / self.size[axis] as f64)
    }

    /// Computes the position of the given vertex.
    pub fn position(&self, bounds: Rectangle<N>, node: [isize; N]) -> [f64; N] {
        let spacing = self.spacing(bounds.clone());
        let mut result = [0.0; N];

        for i in 0..N {
            result[i] = bounds.origin[i] + spacing[i] * node[i] as f64;
        }

        result
    }

    /// Returns the value of the field at the given node.
    pub fn value(&self, node: [isize; N], src: &[f64]) -> f64 {
        src[self.index_from_node(node)]
    }

    /// Sets the value of the field at the given node.
    pub fn set_value(&self, node: [isize; N], v: f64, dest: &mut [f64]) {
        dest[self.index_from_node(node)] = v;
    }

    /// Returns the window which encompasses the whole node space
    pub fn full_window(&self) -> NodeWindow<N> {
        NodeWindow {
            origin: [-(self.ghost as isize); N],
            size: self.node_size(),
        }
    }

    /// Returns a window of just the interior and edges of the node space (no ghost nodes).
    pub fn inner_window(&self) -> NodeWindow<N> {
        NodeWindow {
            origin: [0isize; N],
            size: self.vertex_size(),
        }
    }

    /// An optimized version of `weights()` for use when only one stencil needs to be applied along one axis.
    pub fn weights_axis(
        &self,
        corner: [isize; N],
        weights: &'static [f64],
        axis: usize,
        field: &[f64],
    ) -> f64 {
        let mut result = 0.0;

        let mut node = corner;
        for index in 0..weights.len() {
            node[axis] = corner[axis] + index as isize;
            result += self.value(node, field) * weights[index];
        }

        result
    }

    /// Applies a tensor product of weights to the given the field. Here corner is the corner of stencil
    /// formed by the weights.
    pub fn weights(&self, corner: [isize; N], weights: [&'static [f64]; N], field: &[f64]) -> f64 {
        let wsize: [_; N] = from_fn(|axis| weights[axis].len());

        let mut result = 0.0;

        for index in IndexSpace::new(wsize).iter() {
            let mut weight = 1.0;

            for axis in 0..N {
                weight *= weights[axis][index[axis]];
            }

            let node = from_fn(|axis| corner[axis] + index[axis] as isize);
            result += self.value(node, field) * weight;
        }

        result
    }

    /// Returns the covariant scaling that must be applied to the given operator.
    pub fn scale(&self, bounds: Rectangle<N>, operator: [BasisOperator; N]) -> f64 {
        let spacing = self.spacing(bounds);

        let mut result = 1.0;

        for axis in 0..N {
            result *= operator[axis].scale(spacing[axis])
        }

        result
    }

    /// Computes the window containing active nodes. Aka all nodes that will be used
    /// for kernel evaluation.
    pub fn active_window(&self, boundary: &impl Boundary) -> NodeWindow<N> {
        let mut origin = [0isize; N];
        let mut size = self.vertex_size();

        faces::<N>()
            .filter(|&face| !boundary.kind(face).is_weak())
            .for_each(|face| {
                if face.side {
                    size[face.axis] += self.ghost;
                } else {
                    origin[face.axis] -= self.ghost as isize;
                    size[face.axis] += self.ghost;
                }
            });

        NodeWindow { origin, size }
    }

    /// Set strongly enforced boundary conditions.
    pub fn fill_boundary(
        &self,
        boundary: &impl Boundary,
        condition: &impl Condition<N>,
        dest: &mut [f64],
    ) {
        let vertex_size = self.vertex_size();
        let active_window = self.active_window(boundary);

        // Loop over faces
        for face in faces::<N>() {
            let axis = face.axis;
            let side = face.side;

            // Window of all active ghost nodes adjacent to the given face.
            let mut face_window = active_window.clone();

            if side {
                // If on the right side we have to offset the origin such that it is equal to `vertex_size[axis]`
                let shift = vertex_size[axis] as isize - face_window.origin[axis];
                face_window.origin[axis] += shift;
                // Similarly we have to shrink the size, clamping the minimum to be 0.
                face_window.size[axis] = (face_window.size[axis] as isize - shift).max(0) as usize;
            } else {
                // If on the left side we shrink the size of the window to only include that face.
                face_window.size[axis] = (-face_window.origin[axis]).max(0) as usize;
            }

            // Now we fill the values of these nodes appropriately
            for node in face_window.iter() {
                match boundary.kind(face) {
                    BoundaryKind::Parity => {
                        let parity = condition.parity(face);
                        // Compute offset from nearest vertex
                        let offset = if side {
                            node[axis] - (vertex_size[axis] as isize - 1)
                        } else {
                            node[axis]
                        };

                        // Flip across axis
                        let mut source = node;
                        source[axis] -= 2 * offset;

                        // Get value at this inner node and set current node to the (anti)symmetric reflection
                        // of that value.
                        let v = self.value(source, dest);

                        if parity {
                            self.set_value(node, v, dest);
                        } else {
                            self.set_value(node, -v, dest);
                        }
                    }
                    BoundaryKind::Custom | BoundaryKind::Radiative | BoundaryKind::Free => {
                        // Do nothing for custom boundary conditions.
                    }
                }
            }

            // As well as strongly enforce any diritchlet boundary conditions on axis.
            let intercept = if side { vertex_size[axis] - 1 } else { 0 } as isize;

            // Iterate over face
            for node in active_window.plane(axis, intercept) {
                if boundary.kind(face) == BoundaryKind::Parity && condition.parity(face) == false {
                    // For antisymmetric boundaries we set all values on axis to be 0.
                    self.set_value(node, 0.0, dest);
                }
            }
        }
    }

    fn support_vertex_axis(
        &self,
        boundary: &impl Boundary,
        vertex: [usize; N],
        border: usize,
        axis: usize,
    ) -> Support {
        if vertex[axis] < border && boundary.kind(Face::negative(axis)).is_weak() {
            Support::Negative(vertex[axis])
        } else if vertex[axis] > self.size[axis] - border
            && boundary.kind(Face::positive(axis)).is_weak()
        {
            Support::Positive(self.size[axis] - vertex[axis])
        } else {
            Support::Interior
        }
    }

    fn support_cell_axis(
        &self,
        boundary: &impl Boundary,
        vertex: [usize; N],
        border: usize,
        axis: usize,
    ) -> Support {
        if vertex[axis] < border.saturating_sub(1) && boundary.kind(Face::negative(axis)).is_weak()
        {
            Support::Negative(vertex[axis] + 1)
        } else if vertex[axis] > self.size[axis] - border
            && boundary.kind(Face::positive(axis)).is_weak()
        {
            Support::Positive(self.size[axis] - vertex[axis])
        } else {
            Support::Interior
        }
    }

    /// Evaluates the tensor product of the given operators at the vertex.
    pub fn evaluate<const ORDER: usize>(
        &self,
        boundary: &impl Boundary,
        bounds: Rectangle<N>,
        vertex: [usize; N],
        operator: [BasisOperator; N],
        field: &[f64],
    ) -> f64 {
        let order = const { Order::from_value(ORDER) };

        let node = node_from_vertex(vertex);

        let mut weights: [&'static [f64]; N] = [&[1.0]; N];
        let mut corner = node;

        for axis in 0..N {
            let border = operator[axis].border(order);
            let support = self.support_vertex_axis(boundary, vertex, border, axis);

            weights[axis] = operator[axis].weights(order, support);
            corner[axis] = match support {
                Support::Interior => node[axis],
                Support::Negative(_) => 0,
                Support::Positive(_) => (self.size[axis] + 1 - weights[axis].len()) as isize,
            };
        }

        self.weights(corner, weights, field) * self.scale(bounds, operator)
    }

    /// Computes the interpolation of the underlying data to one increased level of refinement.
    pub fn prolong<const ORDER: usize>(
        &self,
        boundary: &impl Boundary,
        subvertex: [usize; N],
        field: &[f64],
    ) -> f64 {
        let order = const { Order::from_value(ORDER) };
        let vertex: [usize; N] = from_fn(|axis| subvertex[axis] / 2);
        let node = node_from_vertex(vertex);

        let mut weights: [&'static [f64]; N] = [&[1.0]; N];
        let mut corner = node;
        let mut scale = 1.0;

        for axis in 0..N {
            if subvertex[axis] % 2 == 1 {
                let border = Interpolation::border(order);
                let support = self.support_cell_axis(boundary, vertex, border, axis);

                weights[axis] = Interpolation::weights(order, support);
                corner[axis] = match support {
                    Support::Interior => node[axis] - border as isize + 1,
                    Support::Negative(_) => 0,
                    Support::Positive(_) => (self.size[axis] + 1 - weights[axis].len()) as isize,
                };

                scale *= Interpolation::scale(order);
            }
        }

        self.weights(corner, weights, field) * scale
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

    #[derive(Clone)]
    struct Quadrant;

    impl Boundary for Quadrant {
        fn kind(&self, face: Face) -> BoundaryKind {
            match (face.axis, face.side) {
                (_, false) => BoundaryKind::Parity,
                (_, true) => BoundaryKind::Radiative,
            }
        }
    }

    #[test]
    fn vertex_support() {
        let space = NodeSpace {
            size: [8],
            ghost: 2,
        };

        const BORDER: usize = 2;

        let supports = [
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Positive(1),
            Support::Positive(0),
        ];

        for i in 0..9 {
            assert_eq!(
                space.support_vertex_axis(&Quadrant, [i], BORDER, 0),
                supports[i]
            );
        }
    }
}
