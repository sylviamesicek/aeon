//! A module for applying finite difference operators to uniformly distributed nodes. This module
//! represents the intersection of several key abstractions.
//!
//! Firstly, one must differentiate between cells, vertices, and nodes. A cell is, quite simply,
//! a cell of a uniform grid. Most dimensions are stored in terms of cells because they scale easily
//! with refinement and coarsening (aka, a refined grid has twice the number of cells on each axis).
//! Vertices are the points in between cells (i.e. at cell intersections). Most coordinates are stored
//! in terms of vertex indices (a cartesian array of `usize`s) because each vertex represents
//! a "real" degree of freedom on the mesh. Finally nodes extend vertices to include a buffer
//! region around the grid consisting of _ghost nodes_. These ghost nodes are not actual dofs, but
//! rather are used for transfering data between blocks and enforcing simple boundary conditions
//! (parity, periodic, and embedded bcs). Nodes are indexed with arrays of `isizes`.

mod boundary;
mod kernel;
mod window;

pub use boundary::{Boundary, BoundaryKind};
pub use kernel::{FDDerivative, FDDissipation, FDSecondDerivative, Kernel};
pub use window::{NodeCartesianIter, NodePlaneIter, NodeWindow};

use crate::array::ArrayLike as _;
use crate::geometry::{faces, Face, IndexSpace, Rectangle};

/// A uniform rectangular domain of nodes to which
/// various derivative and interpolation kernels can be
/// applied.
#[derive(Debug, Clone)]
pub struct NodeSpace<const N: usize> {
    /// Number of cells along each axis (one less than then number of vertices).
    pub size: [usize; N],
    /// The physical bounds of the node space.
    pub bounds: Rectangle<N>,
    /// Number of ghost vertices in each direction
    pub ghost: usize,
}

impl<const N: usize> NodeSpace<N> {
    /// Computes the total number of nodes in the space.
    pub fn node_count(&self) -> usize {
        self.index_size().iter().product()
    }

    /// Converts a node into a unsigned cartesian index.
    pub fn index_from_node(&self, node: [isize; N]) -> [usize; N] {
        let mut result = [0; N];

        for i in 0..N {
            result[i] = (node[i] + self.ghost as isize) as usize;
        }

        result
    }

    /// Transforms a vertex into a node.
    pub fn node_from_vertex(vertex: [usize; N]) -> [isize; N] {
        let mut result = [0isize; N];

        for axis in 0..N {
            result[axis] = vertex[axis] as isize;
        }

        result
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
    pub fn index_size(&self) -> [usize; N] {
        let mut size = self.size;

        for s in size.iter_mut() {
            *s += 1 + 2 * self.ghost;
        }

        size
    }

    /// Returns an index space over the cells in this node space.
    pub fn cell_space(&self) -> IndexSpace<N> {
        IndexSpace::new(self.cell_size())
    }

    /// Returns an index space over the vertices in this node space.
    pub fn vertex_space(&self) -> IndexSpace<N> {
        IndexSpace::new(self.vertex_size())
    }

    /// Returns an index space over the vertices in this node space.
    pub fn index_space(&self) -> IndexSpace<N> {
        IndexSpace::new(self.index_size())
    }

    /// Returns the window which encompasses the whole node space
    pub fn full_window(&self) -> NodeWindow<N> {
        NodeWindow {
            origin: [-(self.ghost as isize); N],
            size: self.index_size(),
        }
    }

    /// Returns a window of just the interior and edges of the node space (no ghost nodes).
    pub fn inner_window(&self) -> NodeWindow<N> {
        NodeWindow {
            origin: [0; N],
            size: self.vertex_size(),
        }
    }

    /// Returns a window which encompasses all interior and custom nodes,
    /// but excluding unused ghost nodes.
    pub fn custom_window<B: Boundary<N>>(&self, boundary: &B) -> NodeWindow<N> {
        let mut origin = [0isize; N];
        let mut size = self.vertex_size();

        faces::<N>()
            .filter(|&face| boundary.kind(face).is_custom())
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

    /// Computes the position of the given vertex.
    pub fn position(&self, node: [isize; N]) -> [f64; N] {
        let mut result = [0.0; N];

        for i in 0..N {
            result[i] = self.bounds.origin[i] + self.spacing(i) * node[i] as f64;
        }

        result
    }

    /// Returns the spacing along a given axis.
    pub fn spacing(&self, axis: usize) -> f64 {
        self.bounds.size[axis] / self.size[axis] as f64
    }

    /// Returns the value of the field at the given node.
    pub fn value(&self, node: [isize; N], src: &[f64]) -> f64 {
        let linear = self
            .index_space()
            .linear_from_cartesian(self.index_from_node(node));
        src[linear]
    }

    /// Sets the value of the field at the given node.
    pub fn set_value(&self, node: [isize; N], v: f64, dest: &mut [f64]) {
        let linear = self
            .index_space()
            .linear_from_cartesian(self.index_from_node(node));
        dest[linear] = v;
    }

    /// Produces a node space with half the number of
    /// nodes along each direction.
    pub fn coarsened(&self) -> Self {
        let mut cells: [usize; N] = self.size;

        for c in cells.iter_mut() {
            *c /= 2;
        }

        Self {
            size: cells,
            bounds: self.bounds.clone(),
            ghost: self.ghost,
        }
    }

    /// Produces a node space with double the number of nodes
    /// along each direction.
    pub fn refined(&self) -> Self {
        let mut cells: [usize; N] = self.size;

        for c in cells.iter_mut() {
            *c *= 2;
        }

        Self {
            size: cells,
            bounds: self.bounds.clone(),
            ghost: self.ghost,
        }
    }

    /// Set strongly enforce diritchlet-type boundary conditions.
    pub fn diritchlet<B: Boundary<N>>(&self, boundary: &B, dest: &mut [f64]) {
        let window = self.custom_window(boundary);

        for face in faces::<N>() {
            let length = self.vertex_size()[face.axis];
            let intercept = if face.side { length - 1 } else { 0 } as isize;
            let kind = boundary.kind(face);

            // Iterate over face
            for node in window.plane(face.axis, intercept) {
                match kind {
                    BoundaryKind::Parity(false) => {
                        // For antisymmetric boundaries we set all values on axis to be 0.
                        self.set_value(node, 0.0, dest);
                    }
                    BoundaryKind::Custom | BoundaryKind::Free | BoundaryKind::Parity(true) => {}
                }
            }
        }
    }

    /// Evaluate a kernel acting on a node space with the given boundary conditions.
    pub fn evaluate<K: Kernel, B: Boundary<N>>(
        &self,
        kernel: &K,
        boundary: &B,
        src: &[f64],
        dest: &mut [f64],
    ) {
        // Check that support fits into ghost nodes.
        assert!(K::POSITIVE_SUPPORT <= self.ghost && K::NEGATIVE_SUPPORT <= self.ghost);

        let interior_support = K::InteriorWeights::LEN;
        let boundary_support = K::BoundaryWeights::LEN;

        let axis = kernel.axis();
        let spacing = self.spacing(axis);
        let scale = kernel.scale(spacing);
        let length = self.vertex_size()[axis];

        let boundary_negative = boundary.kind(Face::negative(axis));
        let boundary_positive = boundary.kind(Face::positive(axis));

        // Custom window
        let window = self.custom_window(boundary);

        // ****************************
        // Fill interior

        let int_start = if boundary_negative.is_custom() {
            0
        } else {
            K::NEGATIVE_SUPPORT
        };
        let int_end = if boundary_positive.is_custom() {
            length
        } else {
            length - K::POSITIVE_SUPPORT
        };

        for index in int_start..int_end {
            for mut node in window.plane(axis, 0) {
                let mut result: f64 = 0.0;

                for (off, w) in kernel.interior().into_iter().enumerate() {
                    node[axis] = index as isize - K::NEGATIVE_SUPPORT as isize + off as isize;
                    result += w * self.value(node, src);
                }

                node[axis] = index as isize;

                self.set_value(node, scale * result, dest);
            }
        }

        // *****************************
        // Fill negative boundary

        for index in 0..int_start {
            // Measures how far from left-most edge we are
            let left = index;
            // How many exterior points would we depend on if centered.
            let exterior = K::NEGATIVE_SUPPORT.saturating_sub(left);

            for mut node in window.plane(axis, 0) {
                let mut result: f64 = 0.0;

                // Apply specific boundary operators
                match boundary_negative {
                    BoundaryKind::Free => {
                        // Apply increasingly one-sided stencils
                        for (off, w) in kernel.negative(left).into_iter().enumerate() {
                            node[axis] = off as isize;
                            result += w * self.value(node, src);
                        }
                    }
                    BoundaryKind::Parity(sign) => {
                        // Reflect stencil appropriately
                        let sym = if sign { 1.0 } else { -1.0 };
                        let sym_edge = if sign == false { 0.0 } else { 1.0 };

                        let weights = kernel.interior();

                        for i in 0..exterior {
                            node[axis] = (exterior - i) as isize;
                            result += sym * weights[i] * self.value(node, src);
                        }

                        node[axis] = 0;
                        result += sym_edge * weights[exterior] * self.value(node, src);

                        for i in (exterior + 1)..interior_support {
                            node[axis] = (i - exterior) as isize;
                            result += weights[i] * self.value(node, src);
                        }
                    }
                    BoundaryKind::Custom => {
                        // We need not do anything, we already applied the centered stencil to every point
                    }
                }

                node[axis] = index as isize;
                self.set_value(node, scale * result, dest);
            }
        }

        // *****************************
        // Fill positive boundary

        for index in int_end..length {
            // Measures how far from right-most edge we are
            let right = length - 1 - index;

            // How many interior points do we depend on (centered case)
            let interior = K::NEGATIVE_SUPPORT + right;

            for mut node in window.plane(axis, 0) {
                let mut result: f64 = 0.0;

                // Apply specific boundary operators
                match boundary_positive {
                    BoundaryKind::Free => {
                        // Apply increasingly one-sided stencils
                        for (off, w) in kernel.positive(right).into_iter().enumerate() {
                            node[axis] = length as isize - boundary_support as isize + off as isize;
                            result += w * self.value(node, src);
                        }
                    }
                    BoundaryKind::Parity(sign) => {
                        // Reflect stencil appropriately
                        let sym = if sign { 1.0 } else { -1.0 };
                        let sym_edge = if sign == false { 0.0 } else { 1.0 };

                        let weights = kernel.interior();

                        for i in 0..interior {
                            node[axis] =
                                length as isize - 1isize + (i as isize - interior as isize);
                            result += weights[i] * self.value(node, src);
                        }

                        node[axis] = (length - 1) as isize;
                        result += sym_edge * weights[interior] * self.value(node, src);

                        for i in (interior + 1)..interior_support {
                            node[axis] =
                                length as isize - 1isize - (i as isize - interior as isize);
                            result += sym * weights[i] * self.value(node, src);
                        }
                    }
                    BoundaryKind::Custom => {
                        // We need not do anything, we already applied the centered stencil to every point
                    }
                }

                node[axis] = index as isize;
                self.set_value(node, scale * result, dest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MixedBoundary;

    impl Boundary<2> for MixedBoundary {
        fn kind(&self, face: Face) -> BoundaryKind {
            if face.side == false {
                BoundaryKind::Custom
            } else {
                BoundaryKind::Free
            }
        }
    }

    #[test]
    fn evaluate_deriv_2d() {
        let space: NodeSpace<2> = NodeSpace {
            size: [10, 10],
            bounds: Rectangle::UNIT,
            ghost: 1,
        };

        let xspacing = space.spacing(0);

        let kernel = FDDerivative::<2>::new(0);
        let boundary = MixedBoundary;

        // Create a source vector.
        let mut source = vec![0.0; space.node_count()].into_boxed_slice();

        for node in space.full_window().iter() {
            let position = space.position(node);
            space.set_value(node, position[0].sin() * position[1].sin(), &mut source);
        }

        // Allocate explicit vectors
        let mut explicit = vec![0.0; space.node_count()].into_boxed_slice();
        let mut evaluated = vec![0.0; space.node_count()].into_boxed_slice();

        for node in space.custom_window(&boundary).iter() {
            let xi = node[0];
            let yi = node[1];

            if xi < 0 {
                continue;
            }

            if xi == (space.vertex_size()[0] - 1) as isize {
                let x0 = space.value([xi - 2, yi], &source);
                let x1 = space.value([xi - 1, yi], &source);
                let x2 = space.value([xi, yi], &source);

                space.set_value(
                    node,
                    (3.0 * x2 - 4.0 * x1 + x0) / (2.0 * xspacing),
                    &mut explicit,
                );
            } else {
                let left = space.value([xi - 1, yi], &source);
                let right = space.value([xi + 1, yi], &source);

                space.set_value(node, (right - left) / (2.0 * xspacing), &mut explicit);
            }
        }

        space.evaluate(&kernel, &boundary, &source, &mut evaluated);

        for node in space.custom_window(&boundary).iter() {
            if node[0] < 0 {
                continue;
            }

            let diff = space.value(node, &explicit) - space.value(node, &evaluated);
            assert!(diff.abs() <= 10e-10);
        }
    }
}
