use crate::common::{Boundary, Kernel, Stencil, StencilIterator, VertexStencil};
use crate::geometry::{IndexSpace, Rectangle};

pub trait Convolution<const N: usize> {
    type Kernel: Kernel;
    type NegativeBoundary: Boundary;
    type PositiveBoundary: Boundary;

    fn negative(self: &Self, position: [f64; N]) -> Self::NegativeBoundary;
    fn positive(self: &Self, position: [f64; N]) -> Self::PositiveBoundary;
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
}

impl<const N: usize> NodeSpace<N> {
    /// Computes the total number of nodes in the space.
    pub fn len(self: &Self) -> usize {
        let mut result = 1;

        for i in 0..N {
            result *= self.size[i] + 1;
        }

        result
    }

    /// Returns the number of vertices along each axis.
    pub fn vertex_size(self: &Self) -> [usize; N] {
        let mut size = self.size;

        for i in 0..N {
            size[i] += 1;
        }

        size
    }

    /// Returns an index space over the vertices in this node space.
    pub fn vertex_space(self: &Self) -> IndexSpace<N> {
        IndexSpace::new(self.vertex_size())
    }

    /// Computes the position of the given node.
    pub fn position(self: &Self, node: [usize; N]) -> [f64; N] {
        let mut result = [0.0; N];

        for i in 0..N {
            result[i] = self.bounds.origin[i] + self.spacing(i) * node[i] as f64;
        }

        result
    }

    /// Returns the spacing along a given axis.
    pub fn spacing(self: &Self, axis: usize) -> f64 {
        self.bounds.size[axis] / self.size[axis] as f64
    }

    /// Returns the value of the field at the given node.
    pub fn value(self: &Self, node: [usize; N], src: &[f64]) -> f64 {
        let linear = self.vertex_space().linear_from_cartesian(node);
        src[linear]
    }

    /// Sets the value of the field at the given node.
    pub fn set_value(self: &Self, node: [usize; N], v: f64, dest: &mut [f64]) {
        let linear = self.vertex_space().linear_from_cartesian(node);
        dest[linear] = v;
    }

    /// Produces a node space with half the number of
    /// nodes along each direction.
    pub fn coarsened(self: &Self) -> Self {
        let mut cells: [usize; N] = self.size;

        for i in 0..N {
            cells[i] /= 2;
        }

        return Self {
            size: cells,
            bounds: self.bounds.clone(),
        };
    }

    /// Produces a node space with double the number of nodes
    /// along each direction.
    pub fn refined(self: &Self) -> Self {
        let mut cells: [usize; N] = self.size;

        for i in 0..N {
            cells[i] *= 2;
        }

        return Self {
            size: cells,
            bounds: self.bounds.clone(),
        };
    }

    pub fn axis<'a>(&'a self, axis: usize) -> NodeSpaceAxis<'a, N> {
        NodeSpaceAxis { space: self, axis }
    }

    /// Prolong values from src -> dest by simply copying values
    /// from aligned vertices, and ignorning intermediate fine vertices.
    pub fn prolong_inject(self: &Self, src: &[f64], dest: &mut [f64]) {
        let src_space = self.coarsened();

        assert!(src.len() == src_space.len() && dest.len() == self.len());

        for src_vertex in src_space.vertex_space().iter() {
            let mut vertex = src_vertex;

            for i in 0..N {
                vertex[i] *= 2;
            }

            let value = self.value(vertex, src);
            src_space.set_value(src_vertex, value, dest);
        }
    }

    /// Restrict values from src -> dest by copying values from the
    /// fine vertices directly to the corresponding coarse vertices.
    pub fn restrict_inject(self: &Self, src: &[f64], dest: &mut [f64]) {
        let dest_space = self.coarsened();

        assert!(src.len() == self.len() && dest.len() == dest_space.len());

        for dest_vertex in dest_space.vertex_space().iter() {
            let mut vertex = dest_vertex;

            for i in 0..N {
                vertex[i] *= 2;
            }

            let value = self.value(vertex, src);
            dest_space.set_value(dest_vertex, value, dest);
        }
    }
}

pub struct NodeSpaceAxis<'a, const N: usize> {
    space: &'a NodeSpace<N>,
    axis: usize,
}

impl<'a, const N: usize> NodeSpaceAxis<'a, N> {
    /// Evaluates the operation of a kernel on the node space.
    pub fn evaluate<K: Convolution<N>>(
        self: &Self,
        convolution: &K,
        src: &[f64],
        dest: &mut [f64],
    ) {
        let positive: usize = <K::Kernel as Kernel>::InteriorStencil::POSITIVE;
        let negative: usize = <K::Kernel as Kernel>::InteriorStencil::NEGATIVE;

        let boundary_support: usize = <K::Kernel as Kernel>::BoundaryStencil::SUPPORT;
        let interior_support = positive + negative + 1;

        let negative_extent = K::NegativeBoundary::EXTENT;
        let positive_extent = K::PositiveBoundary::EXTENT;

        // Source lengths and dest lengths must match node space.
        assert!(src.len() == self.space.len() && dest.len() == self.space.len());

        // Get spacing along axis for covariant transformation of kernel.
        let spacing = self.space.spacing(self.axis);
        let scale = K::Kernel::scale(spacing);

        // Number of nodes along this axis
        let length: usize = self.space.size[self.axis] + 1;

        // Loop over plane normal to axis
        for mut node in self.space.vertex_space().plane(self.axis, 0) {
            // Position of negative boundary vertex
            node[self.axis] = 0;
            let negative_position = self.space.position(node);

            // Position of positive boundary vertex
            node[self.axis] = length - 1;
            let positive_position = self.space.position(node);

            // *****************************
            // Fill left boundary

            let negative_boundary = convolution.negative(negative_position);

            for left in 0..(negative - negative_extent) {
                let mut result = 0.0;

                let stencil = K::Kernel::negative(left);

                for i in 0..negative_extent {
                    let w = stencil.weight(i);
                    let ghost = self.negative_ghost_value(node, src, &negative_boundary, i);

                    result += w * ghost;
                }

                for i in negative_extent..boundary_support {
                    node[self.axis] = i - negative_extent;
                    let w = stencil.weight(i);

                    result += w * self.space.value(node, src);
                }

                node[self.axis] = left;
                self.space.set_value(node, scale * result, dest);
            }

            for left in (negative - negative_extent)..negative {
                let mut result = 0.0;

                let stencil = K::Kernel::interior();

                for i in 0..negative_extent {
                    let w = stencil.weight(i);
                    let ghost = self.negative_ghost_value(node, src, &negative_boundary, i);

                    result += w * ghost;
                }

                for i in negative_extent..interior_support {
                    node[self.axis] = i - negative_extent;
                    let w = stencil.weight(i);

                    result += w * self.space.value(node, src);
                }

                node[self.axis] = left;
                self.space.set_value(node, scale * result, dest);
            }

            // *************************************
            // Fill right boundary

            let positive_boundary = convolution.positive(positive_position);

            for right in 0..(positive - positive_extent) {
                let mut result = 0.0;

                let stencil = K::Kernel::positive(right);

                for i in 0..positive_extent {
                    let w = stencil.weight(i);
                    let ghost = self.positive_ghost_value(node, src, &positive_boundary, i);

                    result += w * ghost;
                }

                for i in positive_extent..boundary_support {
                    node[self.axis] = length - 1 - (i - positive_extent);
                    let w = stencil.weight(i);

                    result += w * self.space.value(node, src);
                }

                node[self.axis] = length - 1 - right;
                self.space.set_value(node, scale * result, dest);
            }

            for right in (positive - positive_extent)..positive {
                let mut result = 0.0;

                let stencil = K::Kernel::interior();

                for i in 0..positive_extent {
                    let w = stencil.weight(i);
                    let ghost = self.positive_ghost_value(node, src, &positive_boundary, i);

                    result += w * ghost;
                }

                for i in positive_extent..interior_support {
                    node[self.axis] = length - 1 - (i - positive_extent);
                    let w = stencil.weight(i);

                    result += w * self.space.value(node, src);
                }

                node[self.axis] = length - 1 - right;
                self.space.set_value(node, scale * result, dest);
            }

            // *****************************
            // Fill interior

            for middle in negative..(length - positive) {
                let mut result = 0.0;

                for (i, w) in StencilIterator::new(K::Kernel::interior()).enumerate() {
                    node[self.axis] = middle - negative + i;
                    result += w * self.space.value(node, src);
                }

                node[self.axis] = middle;
                self.space.set_value(node, scale * result, dest);
            }
        }
    }

    pub fn evaluate_diag<K: Convolution<N>>(
        self: &Self,
        convolution: &K,
        src: &[f64],
        dest: &mut [f64],
    ) {
        let positive: usize = <K::Kernel as Kernel>::InteriorStencil::POSITIVE;
        let negative: usize = <K::Kernel as Kernel>::InteriorStencil::NEGATIVE;

        let negative_extent = K::NegativeBoundary::EXTENT;
        let positive_extent = K::PositiveBoundary::EXTENT;

        // Source lengths and dest lengths must match node space.
        assert!(src.len() == self.space.len() && dest.len() == self.space.len());

        // Get spacing along axis for covariant transformation of kernel.
        let spacing = self.space.spacing(self.axis);
        let scale = K::Kernel::scale(spacing);

        // Number of nodes along this axis
        let length: usize = self.space.size[self.axis] + 1;

        // Loop over plane normal to axis
        for mut node in self.space.vertex_space().plane(self.axis, 0) {
            // Position of negative boundary vertex
            node[self.axis] = 0;
            let negative_position = self.space.position(node);

            // Position of positive boundary vertex
            node[self.axis] = length - 1;
            let positive_position = self.space.position(node);

            // *****************************
            // Fill left boundary

            let negative_boundary = convolution.negative(negative_position);

            for left in 0..(negative - negative_extent) {
                let mut result = 0.0;

                let stencil = K::Kernel::negative(left);

                for i in 0..negative_extent {
                    let w = stencil.weight(i);

                    let ghost = negative_boundary.stencil(i, spacing).weight(0);

                    result += w * ghost;
                }

                result += stencil.weight(negative_extent);

                node[self.axis] = left;
                self.space.set_value(node, scale * result, dest);
            }

            for left in (negative - negative_extent)..negative {
                let mut result = 0.0;

                let stencil = K::Kernel::interior();

                for i in 0..negative_extent {
                    let w = stencil.weight(i);
                    let ghost = negative_boundary.stencil(i, spacing).weight(0);

                    result += w * ghost;
                }

                result += stencil.weight(negative_extent);

                node[self.axis] = left;
                self.space.set_value(node, scale * result, dest);
            }

            // *************************************
            // Fill right boundary

            let positive_boundary = convolution.positive(positive_position);

            for right in 0..(positive - positive_extent) {
                let mut result = 0.0;

                let stencil = K::Kernel::positive(right);

                for i in 0..positive_extent {
                    let w = stencil.weight(i);
                    let ghost = positive_boundary.stencil(i, spacing).weight(0);

                    result += w * ghost;
                }

                result += stencil.weight(positive_extent);

                node[self.axis] = length - 1 - right;
                self.space.set_value(node, scale * result, dest);
            }

            for right in (positive - positive_extent)..positive {
                let mut result = 0.0;

                let stencil = K::Kernel::interior();

                for i in 0..positive_extent {
                    let w = stencil.weight(i);
                    let ghost = positive_boundary.stencil(i, spacing).weight(0);

                    result += w * ghost;
                }

                result += stencil.weight(positive_extent);

                node[self.axis] = length - 1 - right;
                self.space.set_value(node, scale * result, dest);
            }

            // *****************************
            // Fill interior

            for middle in negative..(length - positive) {
                node[self.axis] = middle;
                self.space
                    .set_value(node, scale * K::Kernel::interior().weight(negative), dest);
            }
        }
    }

    fn negative_ghost_value<B: Boundary>(
        self: &Self,
        mut node: [usize; N],
        src: &[f64],
        boundary: &B,
        extent: usize,
    ) -> f64 {
        let spacing = self.space.spacing(self.axis);

        let mut result = 0.0;

        for (i, w) in StencilIterator::new(boundary.stencil(extent, spacing)).enumerate() {
            node[self.axis] = i;
            result += self.space.value(node, src) * w;
        }

        result
    }

    fn positive_ghost_value<B: Boundary>(
        self: &Self,
        mut node: [usize; N],
        src: &[f64],
        boundary: &B,
        extent: usize,
    ) -> f64 {
        let length: usize = self.space.size[self.axis] + 1;
        let spacing = self.space.spacing(self.axis);

        let mut result = 0.0;

        for (i, w) in StencilIterator::new(boundary.stencil(extent, spacing)).enumerate() {
            node[self.axis] = length - 1 - i;
            result += self.space.value(node, src) * w;
        }

        result
    }

    pub fn apply_diritchlet_bc(self: &Self, face: bool, field: &mut [f64]) {
        let slice = if face { self.space.size[self.axis] } else { 0 };

        for node in self.space.vertex_space().plane(self.axis, slice) {
            self.space.set_value(node, 0.0, field);
        }
    }

    /// Performs bilinear prolongation on the given field.
    pub fn prolong(self: &Self, dest: &mut [f64]) {
        let length = self.space.size[self.axis] + 1;

        for mut node in self.space.vertex_space().plane(self.axis, 0) {
            for i in (1..length - 1).step_by(2) {
                node[self.axis] = i - 1;
                let left = self.space.value(node, dest);
                node[self.axis] = i + 1;
                let right = self.space.value(node, dest);

                // Bilinear prolongation
                node[self.axis] = i;
                self.space.set_value(node, (left + right) / 2.0, dest)
            }
        }
    }

    /// Performs full weighted restriction on the given field.
    pub fn restrict(self: &Self, dest: &mut [f64]) {
        let length = self.space.size[self.axis] + 1;

        for mut node in self.space.vertex_space().plane(self.axis, 0) {
            // Fill left hand side
            {
                node[self.axis] = 0;
                let edge = self.space.value(node, dest);

                node[self.axis] = 1;
                let right = self.space.value(node, dest);

                node[self.axis] = 0;
                self.space.set_value(node, (2.0 * edge + right) / 4.0, dest);
            }

            // Fill right hand side
            {
                node[self.axis] = length - 1;
                let edge = self.space.value(node, dest);

                node[self.axis] = length - 2;
                let left = self.space.value(node, dest);

                node[self.axis] = length - 1;
                self.space.set_value(node, (left + 2.0 * edge) / 4.0, dest);
            }

            for i in (2..length - 1).step_by(2) {
                node[self.axis] = i - 1;
                let left = self.space.value(node, dest);
                node[self.axis] = i + 1;
                let right = self.space.value(node, dest);
                node[self.axis] = i;
                let middle = self.space.value(node, dest);
                // Full weighted restriction
                self.space
                    .set_value(node, (left + 2.0 * middle + right) / 4.0, dest)
            }
        }
    }
}
