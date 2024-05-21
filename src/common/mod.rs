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

use std::array::from_fn;

pub use boundary::{Boundary, BoundaryCondition};
pub use kernel::{FDDerivative, FDDissipation, FDSecondDerivative, Kernel};
pub use window::{NodeCartesianIter, NodePlaneIter, NodeWindow};

use crate::array::ArrayLike as _;
use crate::geometry::{faces, Face, IndexSpace, Rectangle};

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

    /// Converts a node into a linear index.
    pub fn index_from_node(&self, node: [isize; N]) -> usize {
        let cart = from_fn(|i| (node[i] + self.ghost as isize) as usize);
        self.index_space().linear_from_cartesian(cart)
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
            origin: [0isize; N],
            size: self.vertex_size(),
        }
    }

    /// Computes the window containing active nodes. Aka all nodes that will be used
    /// for kernel evaluation.
    pub fn active_window<B: Boundary>(&self, boundary: &B) -> NodeWindow<N> {
        let mut origin = [0isize; N];
        let mut size = self.vertex_size();

        faces::<N>()
            .filter(|&face| !matches!(boundary.face(face), BoundaryCondition::Free))
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
            result[i] = self.bounds.origin[i] + self.spacing_axis(i) * node[i] as f64;
        }

        result
    }

    /// Returns the spacing along each axis of the node space.
    pub fn spacing(&self) -> [f64; N] {
        from_fn(|axis| self.spacing_axis(axis))
    }

    /// Returns the spacing along a given axis.
    pub fn spacing_axis(&self, axis: usize) -> f64 {
        self.bounds.size[axis] / self.size[axis] as f64
    }

    /// Returns the value of the field at the given node.
    pub fn value(&self, node: [isize; N], src: &[f64]) -> f64 {
        src[self.index_from_node(node)]
    }

    /// Sets the value of the field at the given node.
    pub fn set_value(&self, node: [isize; N], v: f64, dest: &mut [f64]) {
        dest[self.index_from_node(node)] = v;
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

    /// Set strongly enforced boundary conditions.
    pub fn fill_boundary<B: Boundary>(&self, boundary: &B, dest: &mut [f64]) {
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
                match boundary.face(face) {
                    BoundaryCondition::Parity(parity) => {
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
                    BoundaryCondition::Free | BoundaryCondition::Custom => {
                        // Do nothing for free or custom boundary conditions.
                    }
                }
            }

            // As well as strongly enforce any diritchlet boundary conditions on axis.
            let intercept = if side { vertex_size[axis] - 1 } else { 0 } as isize;

            // Iterate over face
            for node in active_window.plane(axis, intercept) {
                match boundary.face(face) {
                    BoundaryCondition::Parity(false) => {
                        // For antisymmetric boundaries we set all values on axis to be 0.
                        self.set_value(node, 0.0, dest);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Evaluate a kernel acting on a node space with the given boundary conditions.
    pub fn evaluate<K: Kernel, B: Boundary>(
        &self,
        kernel: &K,
        boundary: &B,
        src: &[f64],
        dest: &mut [f64],
    ) {
        // Check that support fits into ghost nodes.
        assert!(K::POSITIVE_SUPPORT <= self.ghost && K::NEGATIVE_SUPPORT <= self.ghost);

        // let interior_support = K::InteriorWeights::LEN;
        let boundary_support = K::BoundaryWeights::LEN;

        // Alias some commonly used values
        let axis = kernel.axis();
        let spacing = self.spacing_axis(axis);
        let scale = kernel.scale(spacing);
        let length = self.vertex_size()[axis];

        // Boundary condition on negative face.
        let boundary_negative = boundary.face(Face::negative(axis));
        // Boundary condition on positive face.
        let boundary_positive = boundary.face(Face::positive(axis));

        // Window of active nodes
        let window = self.active_window(boundary);

        // ****************************
        // Fill interior

        let int_start = if matches!(boundary_negative, BoundaryCondition::Free) {
            K::NEGATIVE_SUPPORT
        } else {
            0
        };

        let int_end = if matches!(boundary_positive, BoundaryCondition::Free) {
            length - K::POSITIVE_SUPPORT
        } else {
            length
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

            for mut node in window.plane(axis, 0) {
                let mut result: f64 = 0.0;

                // Apply increasingly one-sided stencils
                for (off, w) in kernel.negative(left).into_iter().enumerate() {
                    node[axis] = off as isize;
                    result += w * self.value(node, src);
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

            for mut node in window.plane(axis, 0) {
                let mut result: f64 = 0.0;

                // Apply increasingly one-sided stencils
                for (off, w) in kernel.positive(right).into_iter().enumerate() {
                    node[axis] = length as isize - boundary_support as isize + off as isize;
                    result += w * self.value(node, src);
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

    use std::f64::consts::PI;

    struct MixedBoundary;

    impl Boundary for MixedBoundary {
        fn face(&self, face: Face) -> BoundaryCondition {
            if face.side == false {
                BoundaryCondition::Parity(false)
            } else {
                BoundaryCondition::Free
            }
        }
    }

    #[test]
    fn compare_evaluate_to_explicit() {
        let space: NodeSpace<2> = NodeSpace {
            size: [10, 10],
            bounds: Rectangle {
                origin: [0.0, 0.0],
                size: [PI, PI],
            },
            ghost: 1,
        };

        let xspacing = space.spacing_axis(0);

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

        for node in space.active_window(&boundary).iter() {
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

        for node in space.active_window(&boundary).iter() {
            if node[0] < 0 {
                continue;
            }

            let diff = space.value(node, &explicit) - space.value(node, &evaluated);
            assert!(diff.abs() <= 10e-10);
        }
    }

    #[test]
    fn deriv_lte_2d() {
        fn convergence(nx: usize, ny: usize) -> f64 {
            let space: NodeSpace<2> = NodeSpace {
                size: [nx, ny],
                bounds: Rectangle {
                    origin: [0.0, 0.0],
                    size: [PI, PI],
                },
                ghost: 1,
            };

            // Create a source vector.
            let mut source = vec![0.0; space.node_count()].into_boxed_slice();

            for node in space.full_window().iter() {
                let position = space.position(node);
                space.set_value(node, position[0].sin() * position[1].sin(), &mut source);
            }

            // Allocate explicit vectors
            let mut evaluated = vec![0.0; space.node_count()].into_boxed_slice();
            space.evaluate(
                &FDDerivative::<2>::new(0),
                &MixedBoundary,
                &source,
                &mut evaluated,
            );

            let mut mixed = vec![0.0; space.node_count()].into_boxed_slice();
            space.evaluate(
                &FDDerivative::<2>::new(1),
                &MixedBoundary,
                &evaluated,
                &mut mixed,
            );

            let mut error = 0.0;

            for node in space.inner_window().iter() {
                let position = space.position(node);
                let term1 = space.value(node, &evaluated);
                let term2 = space.value(node, &mixed);

                let diff = term1 + term2
                    - position[0].cos() * position[1].sin()
                    - position[0].cos() * position[1].cos();

                error += diff * diff;
            }

            error.sqrt() * space.spacing_axis(0).sqrt() * space.spacing_axis(1).sqrt()
        }

        let f1 = convergence(50, 50);
        let f2 = convergence(100, 100);
        let f3 = convergence(200, 200);
        let f4 = convergence(400, 400);

        assert!(f1 / f2 >= 4.0);
        assert!(f2 / f3 >= 4.0);
        assert!(f3 / f4 >= 4.0);
    }

    struct ParityBoundary;

    impl Boundary for ParityBoundary {
        fn face(&self, face: Face) -> BoundaryCondition {
            BoundaryCondition::Parity(face.side)
        }
    }

    #[test]
    fn boundary_filling_parity() {
        let space: NodeSpace<2> = NodeSpace {
            size: [4, 4],
            bounds: Rectangle {
                origin: [0.0, 0.0],
                size: [PI / 2.0, PI / 2.0],
            },
            ghost: 2,
        };

        // Create a source vector.
        let mut field = vec![0.0; space.node_count()].into_boxed_slice();

        for node in space.full_window().iter() {
            let [rho, z] = space.position(node);
            space.set_value(node, rho.sin() * z.sin(), &mut field);
        }

        space.fill_boundary(&ParityBoundary, &mut field);

        for node in space.full_window().iter() {
            let [rho, z] = space.position(node);
            let index = space.index_from_node(node);
            assert!((field[index] - rho.sin() * z.sin()).abs() <= 10e-15);
        }
    }

    struct WindowBoundary;

    impl Boundary for WindowBoundary {
        fn face(&self, face: Face) -> BoundaryCondition {
            match (face.axis, face.side) {
                (0, true) => BoundaryCondition::Custom,
                (1, false) => BoundaryCondition::Parity(false),
                _ => BoundaryCondition::Free,
            }
        }
    }

    #[test]
    fn node_windows() {
        let node_space = NodeSpace {
            bounds: Rectangle::UNIT,
            size: [10, 10],
            ghost: 2,
        };

        let full_window = node_space.full_window();
        assert_eq!(full_window.origin, [-2, -2]);
        assert_eq!(full_window.size, [15, 15]);

        let active_window = node_space.active_window(&WindowBoundary);
        assert_eq!(active_window.origin, [0, -2]);
        assert_eq!(active_window.size, [13, 13]);

        let inner_window = node_space.inner_window();
        assert_eq!(inner_window.origin, [0, 0]);
        assert_eq!(inner_window.size, [11, 11]);
    }
}
