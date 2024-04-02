use crate::array::Array;
use crate::common::{Boundary, Kernel};
use crate::geometry::{IndexSpace, Rectangle};

use super::BoundarySet;

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
    pub fn len(&self) -> usize {
        let mut result = 1;

        for i in 0..N {
            result *= self.size[i] + 1;
        }

        result
    }

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

    /// Returns an index space over the vertices in this node space.
    pub fn vertex_space(&self) -> IndexSpace<N> {
        IndexSpace::new(self.vertex_size())
    }

    /// Returns an index space over the cells in this node space.
    pub fn cell_space(&self) -> IndexSpace<N> {
        IndexSpace::new(self.cell_size())
    }

    /// Computes the position of the given node.
    pub fn position(&self, node: [usize; N]) -> [f64; N] {
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
    pub fn value(&self, node: [usize; N], src: &[f64]) -> f64 {
        let linear = self.vertex_space().linear_from_cartesian(node);
        src[linear]
    }

    /// Sets the value of the field at the given node.
    pub fn set_value(&self, node: [usize; N], v: f64, dest: &mut [f64]) {
        let linear = self.vertex_space().linear_from_cartesian(node);
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
        }
    }

    pub fn axis<'a, B: BoundarySet<N>>(
        &'a self,
        axis: usize,
        set: &'a B,
    ) -> NodeSpaceAxis<'a, N, B> {
        NodeSpaceAxis {
            space: self,
            set,
            axis,
        }
    }

    /// Prolong values from src -> dest by simply copying values
    /// from aligned vertices, and ignorning intermediate fine vertices.
    pub fn prolong_inject(&self, src: &[f64], dest: &mut [f64]) {
        let src_space = self.coarsened();

        assert!(src.len() == src_space.len() && dest.len() == self.len());

        for src_vertex in src_space.vertex_space().iter() {
            let mut vertex = src_vertex;

            for v in vertex.iter_mut() {
                *v *= 2;
            }

            let value = src_space.value(src_vertex, src);
            self.set_value(vertex, value, dest);
        }
    }

    /// Restrict values from src -> dest by copying values from the
    /// fine vertices directly to the corresponding coarse vertices.
    pub fn restrict_inject(&self, src: &[f64], dest: &mut [f64]) {
        let dest_space = self.coarsened();

        assert!(src.len() == self.len() && dest.len() == dest_space.len());

        for dest_vertex in dest_space.vertex_space().iter() {
            let mut vertex = dest_vertex;

            for v in vertex.iter_mut() {
                *v *= 2;
            }

            let value = self.value(vertex, src);
            dest_space.set_value(dest_vertex, value, dest);
        }
    }

    /// Performs bilinear prolongation on the given field.
    pub fn prolong(&self, dest: &mut [f64]) {
        let vertex_size = self.vertex_size();

        for axis in 0..N {
            let length = vertex_size[axis];

            for mut node in self.vertex_space().plane(axis, 0) {
                for i in (1..length - 1).step_by(2) {
                    node[axis] = i - 1;
                    let left = self.value(node, dest);
                    node[axis] = i + 1;
                    let right = self.value(node, dest);

                    // Bilinear prolongation
                    node[axis] = i;
                    self.set_value(node, (left + right) / 2.0, dest)
                }
            }
        }
    }

    /// Performs full weighted restriction on the given field.
    pub fn restrict(&self, dest: &mut [f64]) {
        let vertex_size = self.vertex_size();

        for axis in 0..N {
            let length = vertex_size[axis];

            for mut node in self.vertex_space().plane(axis, 0) {
                // Fill left hand side
                {
                    node[axis] = 0;
                    let edge = self.value(node, dest);

                    node[axis] = 1;
                    let right = self.value(node, dest);

                    node[axis] = 0;
                    self.set_value(node, (2.0 * edge + right) / 4.0, dest);
                }

                // Fill right hand side
                {
                    node[axis] = length - 1;
                    let edge = self.value(node, dest);

                    node[axis] = length - 2;
                    let left = self.value(node, dest);

                    node[axis] = length - 1;
                    self.set_value(node, (left + 2.0 * edge) / 4.0, dest);
                }

                for i in (2..=length - 3).step_by(2) {
                    node[axis] = i - 1;
                    let left = self.value(node, dest);
                    node[axis] = i + 1;
                    let right = self.value(node, dest);
                    node[axis] = i;
                    let middle = self.value(node, dest);
                    // Full weighted restriction
                    self.set_value(node, (left + 2.0 * middle + right) / 4.0, dest)
                }
            }
        }
    }
}

pub struct NodeSpaceAxis<'a, const N: usize, B: BoundarySet<N>> {
    space: &'a NodeSpace<N>,
    set: &'a B,
    axis: usize,
}

impl<'a, const N: usize, B: BoundarySet<N>> NodeSpaceAxis<'a, N, B> {
    /// Evaluates the operation of a kernel on the node space.
    pub fn evaluate<K: Kernel>(&self, src: &[f64], dest: &mut [f64]) {
        let positive: usize = K::POSITIVE_SUPPORT;
        let negative: usize = K::NEGATIVE_SUPPORT;

        // Source lengths and dest lengths must match node space.
        assert!(src.len() == self.space.len() && dest.len() == self.space.len());

        // Get spacing along axis for covariant transformation of kernel.
        let spacing = self.space.spacing(self.axis);
        let scale = K::scale(spacing);

        // Number of nodes along this axis
        let length: usize = self.space.size[self.axis] + 1;

        // Loop over plane normal to axis
        for mut node in self.space.vertex_space().plane(self.axis, 0) {
            // *****************************
            // Fill left boundary
            self.evaluate_negative::<K>(node, Some(src), dest);

            // *************************************
            // Fill right boundary
            self.evaluate_positive::<K>(node, Some(src), dest);

            // *****************************
            // Fill interior

            for middle in negative..(length - positive) {
                let mut result = 0.0;

                for (i, w) in K::interior().into_iter().enumerate() {
                    node[self.axis] = middle - negative + i;
                    result += w * self.space.value(node, src);
                }

                node[self.axis] = middle;
                self.space.set_value(node, scale * result, dest);
            }
        }
    }

    pub fn evaluate_diag<K: Kernel>(&self, dest: &mut [f64]) {
        let positive: usize = K::POSITIVE_SUPPORT;
        let negative: usize = K::NEGATIVE_SUPPORT;

        // Source lengths and dest lengths must match node space.
        assert!(dest.len() == self.space.len());

        // Get spacing along axis for covariant transformation of kernel.
        let spacing = self.space.spacing(self.axis);
        let scale = K::scale(spacing);

        // Number of nodes along this axis
        let length: usize = self.space.size[self.axis] + 1;

        // Loop over plane normal to axis
        for mut node in self.space.vertex_space().plane(self.axis, 0) {
            // *****************************
            // Fill left boundary

            self.evaluate_negative::<K>(node, None, dest);

            // *************************************
            // Fill right boundary

            self.evaluate_positive::<K>(node, None, dest);

            // *****************************
            // Fill interior

            for middle in negative..(length - positive) {
                node[self.axis] = middle;
                self.space
                    .set_value(node, scale * K::interior()[negative], dest);
            }
        }
    }

    fn evaluate_negative<K: Kernel>(
        &self,
        mut node: [usize; N],
        src: Option<&[f64]>,
        dest: &mut [f64],
    ) {
        let kernel_negative_support: usize = K::NEGATIVE_SUPPORT;
        let kernel_interior_support: usize = K::NEGATIVE_SUPPORT + K::POSITIVE_SUPPORT + 1;
        let kernel_boundary_support: usize = K::BoundaryStencil::LEN;
        let ghost = <B::NegativeBoundary as Boundary>::GHOST;

        // Get spacing along axis for covariant transformation of kernel.
        let spacing = self.space.spacing(self.axis);
        let scale = K::scale(spacing);

        // Position of negative boundary vertex
        node[self.axis] = 0;
        let negative_position = self.space.position(node);
        let negative_boundary = self.set.negative(negative_position);

        // First loop over points that require a one-sided stencil and ghost values
        for left in 0..kernel_negative_support.saturating_sub(ghost) {
            // Get negative stencil
            let stencil = K::negative(left + ghost);

            // Accumulate result
            let mut result = 0.0;

            if let Some(src) = src {
                for i in 0..ghost {
                    node[self.axis] = 0;
                    result += stencil[i] * self.negative_ghost_value(node, ghost - 1 - i, src);
                }

                for i in ghost..kernel_boundary_support {
                    node[self.axis] = i - ghost;
                    result += stencil[i] * self.space.value(node, src);
                }
            } else {
                for i in 0..ghost {
                    // Each ghost vertex could possibly contribute
                    result += stencil[i] * negative_boundary.stencil(ghost - 1 - i, spacing)[left];
                }

                result += stencil[ghost + left]
            }

            node[self.axis] = left;
            self.space.set_value(node, scale * result, dest);
        }

        // Next loop over points that require a central stencil and ghost values
        for left in kernel_negative_support.saturating_sub(ghost)..kernel_negative_support {
            let stencil = K::interior();

            // How many ghost points does this stencil require?
            let negative_edge = kernel_negative_support - left;

            // Accumulate result
            let mut result = 0.0;

            if let Some(src) = src {
                // Use ghost node values
                for i in 0..negative_edge {
                    node[self.axis] = 0;
                    result +=
                        stencil[i] * self.negative_ghost_value(node, negative_edge - 1 - i, src);
                }

                // Fill from interior
                for i in negative_edge..kernel_interior_support {
                    node[self.axis] = i - negative_edge;
                    result += stencil[i] * self.space.value(node, src);
                }
            } else {
                for i in 0..negative_edge {
                    result += stencil[i]
                        * negative_boundary.stencil(negative_edge - 1 - i, spacing)[left];
                }

                result += stencil[kernel_negative_support];
            }

            node[self.axis] = left;
            self.space.set_value(node, scale * result, dest);
        }
    }

    fn evaluate_positive<K: Kernel>(
        &self,
        mut node: [usize; N],
        src: Option<&[f64]>,
        dest: &mut [f64],
    ) {
        let kernel_positive_support: usize = K::POSITIVE_SUPPORT;
        let kernel_interior_support: usize = K::NEGATIVE_SUPPORT + K::POSITIVE_SUPPORT + 1;
        let kernel_boundary_support: usize = K::BoundaryStencil::LEN;
        let ghost = <B::PositiveBoundary as Boundary>::GHOST;

        // Get spacing along axis for covariant transformation of kernel.
        let spacing = self.space.spacing(self.axis);
        let scale = K::scale(spacing);

        let length = self.space.vertex_size()[self.axis];

        // Position of positive boundary vertex
        node[self.axis] = length - 1;
        let positive_position = self.space.position(node);
        let positive_boundary = self.set.positive(positive_position);

        // First loop over points that require a one-sided stencil and ghost values
        for right in 0..kernel_positive_support.saturating_sub(ghost) {
            let stencil = K::positive(right + ghost);

            // Accumulate result
            let mut result = 0.0;

            if let Some(src) = src {
                for i in 0..ghost {
                    node[self.axis] = length - 1;
                    result += stencil[kernel_boundary_support - 1 - i]
                        * self.positive_ghost_value(node, ghost - 1 - i, src);
                }

                for i in ghost..kernel_boundary_support {
                    node[self.axis] = length - 1 - (i - ghost);
                    result +=
                        stencil[kernel_boundary_support - 1 - i] * self.space.value(node, src);
                }
            } else {
                for i in 0..ghost {
                    result += stencil[kernel_boundary_support - 1 - i]
                        * positive_boundary.stencil(ghost - 1 - i, spacing)[right];
                }

                result += stencil[kernel_boundary_support - 1 - ghost - right];
            }

            node[self.axis] = length - 1 - right;
            self.space.set_value(node, scale * result, dest);
        }

        // Next loop over points that require a central stencil and ghost values
        for right in kernel_positive_support.saturating_sub(ghost)..kernel_positive_support {
            let stencil = K::interior();

            // How many ghost points does this stencil require?
            let positive_edge = kernel_positive_support - right;

            // Accumulate result
            let mut result = 0.0;

            if let Some(src) = src {
                // Use ghost node values
                for i in 0..positive_edge {
                    result += stencil[kernel_interior_support - 1 - i]
                        * self.positive_ghost_value(node, positive_edge - 1 - i, src);
                }

                // Fill from interior
                for i in positive_edge..kernel_interior_support {
                    node[self.axis] = length - 1 - (i - positive_edge);
                    result +=
                        stencil[kernel_interior_support - 1 - i] * self.space.value(node, src);
                }
            } else {
                // Use ghost node values
                for i in 0..positive_edge {
                    result += stencil[kernel_interior_support - 1 - i]
                        * positive_boundary.stencil(positive_edge - 1 - i, spacing)[right];
                }

                result += stencil[K::NEGATIVE_SUPPORT];
            }

            node[self.axis] = length - 1 - right;
            self.space.set_value(node, scale * result, dest);
        }
    }

    fn negative_ghost_value(&self, mut node: [usize; N], ghost: usize, src: &[f64]) -> f64 {
        let position = self.space.position(node);
        let spacing = self.space.spacing(self.axis);

        let boundary = self.set.negative(position);

        let mut result = 0.0;

        for (i, w) in boundary.stencil(ghost, spacing).into_iter().enumerate() {
            node[self.axis] = i;
            result += self.space.value(node, src) * w;
        }

        result
    }

    fn positive_ghost_value(&self, mut node: [usize; N], ghost: usize, src: &[f64]) -> f64 {
        let length: usize = self.space.vertex_size()[self.axis];

        let position = self.space.position(node);
        let spacing = self.space.spacing(self.axis);

        let boundary = self.set.positive(position);

        let mut result = 0.0;

        for (i, w) in boundary.stencil(ghost, spacing).into_iter().enumerate() {
            node[self.axis] = length - 1 - i;
            result += self.space.value(node, src) * w;
        }

        result
    }

    pub fn diritchlet(&self, field: &mut [f64]) {
        if B::NegativeBoundary::IS_DIRITCHLET {
            for node in self.space.vertex_space().plane(self.axis, 0) {
                self.space.set_value(node, 0.0, field);
            }
        }

        if B::PositiveBoundary::IS_DIRITCHLET {
            for node in self
                .space
                .vertex_space()
                .plane(self.axis, self.space.size[self.axis])
            {
                self.space.set_value(node, 0.0, field);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{FDDerivative, SymmetricBoundary};

    const CELLS: usize = 10;

    fn source_field() -> Vec<f64> {
        let space = NodeSpace {
            bounds: Rectangle::UNIT,
            size: [CELLS, CELLS],
        };

        let mut source = vec![0.0; space.len()];

        // Source should be filled with sin(x) * cos(y).
        for vert in space.vertex_space().iter() {
            let pos = space.position(vert);
            let value = pos[0].sin() * pos[1].cos();
            space.set_value(vert, value, &mut source);
        }

        source
    }

    #[test]
    fn positions_and_spacing() {
        let space = NodeSpace {
            bounds: Rectangle::UNIT,
            size: [CELLS, CELLS],
        };

        assert!(space.position([0, 0]) == [0.0, 0.0]);
        assert!(space.position([CELLS / 2, CELLS / 2]) == [0.5, 0.5]);
        assert!(space.position([CELLS, CELLS]) == [1.0, 1.0]);
        assert!(space.spacing(0) == 0.1);
    }

    #[test]
    fn evaluate_derivative() {
        let space = NodeSpace {
            bounds: Rectangle::UNIT,
            size: [CELLS, CELLS],
        };

        let hy = 1.0 / space.spacing(1);

        let src = source_field();

        let mut dest = vec![0.0; space.len()];

        let set = SymmetricBoundary::<2>;
        space
            .axis(1, &set)
            .evaluate::<FDDerivative<2>>(&src, &mut dest);

        for vert in space.vertex_space().iter() {
            let xi = vert[0];
            let yi = vert[1];

            if yi == CELLS {
                assert_eq!(space.value(vert, &dest), 0.0);
            } else if yi == 0 {
                assert_eq!(space.value(vert, &dest), 0.0);
            } else {
                let positive = space.value([xi, yi + 1], &src);
                let negative = space.value([xi, yi - 1], &src);

                assert_eq!(
                    space.value(vert, &dest),
                    (0.5 * positive - 0.5 * negative) * hy
                );
            }
        }
    }

    #[test]
    fn prolong() {
        let space = NodeSpace {
            bounds: Rectangle::UNIT,
            size: [CELLS, CELLS],
        };

        let source = source_field();
        let mut dest = source_field();
        space.prolong(&mut dest);

        for vert in space.vertex_space().iter() {
            let xi = vert[0];
            let yi = vert[1];

            let src = |xoff: isize, yoff: isize| {
                let xnode = xi as isize + xoff;
                let ynode = yi as isize + yoff;

                space.value([xnode as usize, ynode as usize], &source)
            };

            let value = match (xi % 2 == 1, yi % 2 == 1) {
                (false, false) => src(0, 0),
                (false, true) => 0.5 * src(0, -1) + 0.5 * src(0, 1),
                (true, false) => 0.5 * src(-1, 0) + 0.5 * src(1, 0),
                (true, true) => {
                    0.25 * src(-1, -1) + 0.25 * src(1, -1) + 0.25 * src(-1, 1) + 0.25 * src(1, 1)
                }
            };

            assert!((space.value([xi, yi], &dest) - value).abs() < 1e-10);
        }
    }

    #[test]
    fn restrict_full() {
        let space = NodeSpace {
            bounds: Rectangle::UNIT,
            size: [CELLS, CELLS],
        };

        let source = source_field();
        let mut dest = source_field();
        space.restrict(&mut dest);

        for vert in space.coarsened().vertex_space().iter() {
            let xi = vert[0];
            let yi = vert[1];

            let src = |xoff: isize, yoff: isize| {
                if xi == 0 && xoff < 0 {
                    return 0.0;
                } else if yi == 0 && yoff < 0 {
                    return 0.0;
                } else if xi == CELLS / 2 && xoff > 0 {
                    return 0.0;
                } else if yi == CELLS / 2 && yoff > 0 {
                    return 0.0;
                }

                let xnode = 2 * xi as isize + xoff;
                let ynode = 2 * yi as isize + yoff;

                space.value([xnode as usize, ynode as usize], &source)
            };

            let corners = src(-1, -1) + src(1, -1) + src(1, 1) + src(-1, 1);
            let edges = src(-1, 0) + src(1, 0) + src(0, 1) + src(0, -1);
            let value = 1.0 / 16.0 * corners + 1.0 / 8.0 * edges + 1.0 / 4.0 * src(0, 0);

            assert!((space.value([2 * xi, 2 * yi], &dest) - value).abs() < 1e-10);
        }
    }
}
