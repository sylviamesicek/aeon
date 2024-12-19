use aeon_geometry::{
    faces, regions, CartesianIter, Face, IndexSpace, IndexWindow, Rectangle, Region, Side,
};
use std::array::{self, from_fn};

use crate::{
    Border, Boundary, BoundaryKind, CellKernel, Condition, Convolution, Kernel, Value, VertexKernel,
};

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
pub struct NodeSpace<const N: usize> {
    /// Number of cells along each axis (one less than then number of vertices).
    size: [usize; N],
    ghost: usize,
}

impl<const N: usize> NodeSpace<N> {
    pub fn new(size: [usize; N], ghost: usize) -> Self {
        Self { size, ghost }
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

    pub fn spacing_axis(&self, bounds: Rectangle<N>, axis: usize) -> f64 {
        bounds.size[axis] / self.size[axis] as f64
    }

    /// Returns true if the node lies inside the interior of the nodespace (i.e. it is not a padding or ghost node).
    pub fn is_interior(&self, node: [isize; N]) -> bool {
        (0..N)
            .map(|axis| node[axis] >= 0 && node[axis] < self.size[axis] as isize)
            .all(|b| b)
    }

    /// Returns true of the node lies on the given face.
    pub fn is_on_face(&self, face: Face<N>, node: [isize; N]) -> bool {
        let intercept = if face.side { self.size[face.axis] } else { 0 };
        node[face.axis] == intercept as isize
    }

    /// Returns true if the node is owned by the given face.
    pub fn is_owned_by_face(&self, face: Face<N>, node: [isize; N]) -> bool {
        if face.side {
            // Check all negative faces
            for axis in 0..N {
                if node[axis] == 0 {
                    return false;
                }
            }

            // Check all lower postive faces
            for axis in 0..face.axis {
                if node[axis] == self.size[axis] as isize {
                    return false;
                }
            }

            node[face.axis] == self.size[face.axis] as isize
        } else {
            // Check all negative faces lower than this face
            for axis in 0..face.axis {
                if node[axis] == 0 {
                    return false;
                }
            }

            node[face.axis] == 0
        }
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
        for axis in 0..N {
            debug_assert!(node[axis] >= -(self.ghost as isize));
            debug_assert!(node[axis] <= (self.size[axis] + self.ghost) as isize);
        }

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

    pub fn apply(&self, corner: [isize; N], stencils: [&[f64]; N], field: &[f64]) -> f64 {
        for axis in 0..N {
            debug_assert!(corner[axis] >= -(self.ghost as isize));
            debug_assert!(
                corner[axis] <= (self.size[axis] + 1 + self.ghost - stencils[axis].len()) as isize
            );
        }

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
        field: &[f64],
        axis: usize,
    ) -> f64 {
        let mut result = 0.0;

        let mut node = corner;

        for (i, weight) in stencil.iter().enumerate() {
            node[axis] = corner[axis] + i as isize;

            let index = self.index_from_node(node);
            result += field[index] * weight;
        }

        result
    }

    pub fn evaluate_interior(
        &self,
        convolution: impl Convolution<N>,
        bounds: Rectangle<N>,
        node: [isize; N],
        field: &[f64],
    ) -> f64 {
        let spacing = self.spacing(bounds);
        let stencils = array::from_fn(|axis| convolution.interior(axis));
        let corner = array::from_fn(|axis| node[axis] - convolution.border_width(axis) as isize);
        self.apply(corner, stencils, field) * convolution.scale(spacing)
    }

    pub fn evaluate_axis_interior(
        &self,
        kernel: &impl VertexKernel,
        bounds: Rectangle<N>,
        node: [isize; N],
        field: &[f64],
        axis: usize,
    ) -> f64 {
        let spacing = self.spacing_axis(bounds, axis);
        let stencil = kernel.interior();
        let mut corner = node;
        corner[axis] -= kernel.border_width() as isize;
        self.apply_axis(corner, stencil, field, axis) * kernel.scale(spacing)
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
            origin[axis] = -(self.ghost as isize);
        }

        NodeWindow {
            origin,
            size: self.full_size(),
        }
    }

    pub fn region_window(&self, extent: usize, region: Region<N>) -> NodeWindow<N> {
        debug_assert!(extent <= self.ghost);

        let origin = array::from_fn(|axis| match region.side(axis) {
            Side::Left => -(extent as isize),
            Side::Right => self.size[axis] as isize + 1,
            Side::Middle => 0,
        });

        let size = array::from_fn(|axis: usize| match region.side(axis) {
            Side::Left | Side::Right => extent,
            Side::Middle => self.size[axis] + 1,
        });

        NodeWindow { origin, size }
    }

    pub fn face_window(&self, Face { axis, side }: Face<N>) -> NodeWindow<N> {
        let intercept = if side { self.size[axis] } else { 0 } as isize;

        let mut origin = [0; N];
        let mut size = array::from_fn(|axis| self.size[axis] + 1);

        origin[axis] = intercept;
        size[axis] = 1;

        NodeWindow { origin, size }
    }

    pub fn face_window_disjoint(&self, face: Face<N>) -> NodeWindow<N> {
        let intercept = if face.side { self.size[face.axis] } else { 0 } as isize;

        let mut origin = [0; N];
        let mut size = self.inner_size();

        origin[face.axis] = intercept;
        size[face.axis] = 1;

        if face.side {
            for axis in (0..N).filter(|&axis| axis != face.axis) {
                origin[axis] += 1;
                size[axis] -= 1;
            }

            for axis in 0..face.axis {
                size[axis] -= 1;
            }
        } else {
            for axis in 0..face.axis {
                origin[axis] += 1;
                size[axis] -= 1;
            }
        }

        NodeWindow { origin, size }
    }

    pub fn support(
        &self,
        boundary: &impl Boundary<N>,
        vertex: usize,
        border_width: usize,
        axis: usize,
    ) -> Support {
        debug_assert!(self.ghost >= border_width);

        let has_negative = boundary.kind(Face::negative(axis)).has_ghost();
        let has_positive = boundary.kind(Face::positive(axis)).has_ghost();

        if !has_negative && vertex < border_width {
            Support::Negative(vertex)
        } else if !has_positive && vertex >= self.size[axis] + 1 - border_width {
            Support::Positive(self.size[axis] - vertex)
        } else {
            Support::Interior
        }
    }

    pub fn support_cell(
        &self,
        boundary: &impl Boundary<N>,
        cell: usize,
        border_width: usize,
        axis: usize,
    ) -> Support {
        debug_assert!(self.ghost >= border_width);
        debug_assert!(cell < self.size[axis]);

        let has_negative = boundary.kind(Face::negative(axis)).has_ghost();
        let has_positive = boundary.kind(Face::positive(axis)).has_ghost();

        if !has_negative && cell < border_width {
            let right = cell;
            Support::Negative(right)
        } else if !has_positive && cell >= self.size[axis] - border_width {
            let left = self.size[axis] - 1 - cell;
            Support::Positive(left)
        } else {
            Support::Interior
        }
    }
}

impl<const N: usize> NodeSpace<N> {
    /// Set strongly enforced boundary conditions.
    pub fn fill_boundary(
        &self,
        extent: usize,
        boundary: impl Boundary<N> + Condition<N>,
        dest: &mut [f64],
    ) {
        // Loop over regions
        for region in regions::<N>() {
            if region == Region::CENTRAL {
                continue;
            }

            // Nodes to iterate over
            let window = self.region_window(extent, region);

            // Parity boundary conditions
            let mut parity = true;

            for face in region.adjacent_faces() {
                // Flip parity if boundary is antisymmetric
                parity ^= boundary.kind(face) == BoundaryKind::Parity && !boundary.parity(face);
            }

            let sign = if parity { 1.0 } else { -1.0 };

            for node in window {
                let mut source = node;

                for face in region.adjacent_faces() {
                    if boundary.kind(face) != BoundaryKind::Parity {
                        continue;
                    }

                    if face.side {
                        let dist = node[face.axis] - self.size[face.axis] as isize;
                        source[face.axis] = self.size[face.axis] as isize - dist;
                    } else {
                        source[face.axis] = -node[face.axis];
                    }
                }

                dest[self.index_from_node(node)] = sign * dest[self.index_from_node(source)];
            }
        }

        // Loop over faces
        for face in faces::<N>() {
            if boundary.kind(face) == BoundaryKind::Parity && !boundary.parity(face) {
                // Iterate over face
                for node in self.face_window_disjoint(face) {
                    // For antisymmetric boundaries we set all values on axis to be 0.
                    let index = self.index_from_node(node);
                    dest[index] = 0.0;
                }
            }
        }
    }

    fn stencils<'a, C: Convolution<N>>(
        &self,
        boundary: impl Boundary<N>,
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

            let kind = boundary.kind(face);

            match kind {
                // (BoundaryKind::Parity, false) => convolution.antisymmetric(border, axis),
                // (BoundaryKind::Parity, true) => convolution.symmetric(border, axis),
                BoundaryKind::Custom | BoundaryKind::Parity => convolution.interior(axis),
                BoundaryKind::Free | BoundaryKind::Radiative => convolution.free(border, axis),
            }
        })
    }

    fn stencil_axis<'a, K: Kernel>(
        &self,
        boundary: &impl Boundary<N>,
        support: Support,
        kernel: &'a K,
        axis: usize,
    ) -> &'a [f64] {
        let border = match support {
            Support::Negative(border) => Border::Negative(border),
            Support::Positive(border) => Border::Positive(border),
            Support::Interior => {
                return kernel.interior();
            }
        };

        let face = Face {
            axis,
            side: border.side(),
        };

        let kind = boundary.kind(face);

        match kind {
            // (BoundaryKind::Parity, false) => convolution.antisymmetric(border, axis),
            // (BoundaryKind::Parity, true) => convolution.symmetric(border, axis),
            BoundaryKind::Custom | BoundaryKind::Parity => kernel.interior(),
            BoundaryKind::Free | BoundaryKind::Radiative => kernel.free(border),
        }
    }

    /// Evaluates the operation of a convolution at a given vertex, assuming the given boundary conditions.
    pub fn evaluate(
        &self,
        boundary: impl Boundary<N>,
        convolution: impl Convolution<N>,
        bounds: Rectangle<N>,
        vertex: [usize; N],
        field: &[f64],
    ) -> f64 {
        for axis in 0..N {
            debug_assert!(vertex[axis] <= self.size[axis]);
        }

        let spacing = self.spacing(bounds);
        let support = array::from_fn(|axis| {
            self.support(
                &boundary,
                vertex[axis],
                convolution.border_width(axis),
                axis,
            )
        });
        let stencils = self.stencils(boundary, support, &convolution);
        let corner = array::from_fn(|axis| match support[axis] {
            Support::Interior => vertex[axis] as isize - convolution.border_width(axis) as isize,
            Support::Negative(_) => 0,
            Support::Positive(_) => (self.size[axis] + 1) as isize - stencils[axis].len() as isize,
        });
        self.apply(corner, stencils, field) * convolution.scale(spacing)
    }

    /// Evaluates the operation of a kernel along an axis at a given vertex, assuming the given boundary conditions.
    pub fn evaluate_axis(
        &self,
        boundary: impl Boundary<N>,
        kernel: &impl VertexKernel,
        bounds: Rectangle<N>,
        node: [isize; N],
        field: &[f64],
        axis: usize,
    ) -> f64 {
        #[cfg(debug_assertions)]
        for axis in 0..N {
            assert!(node[axis] <= (self.size[axis] + self.ghost) as isize);
            assert!(node[axis] >= -(self.ghost as isize));
        }

        debug_assert!(node[axis] >= 0);
        debug_assert!(node[axis] <= self.size[axis] as isize);

        let spacing = self.spacing_axis(bounds, axis);
        let support = self.support(&boundary, node[axis] as usize, kernel.border_width(), axis);
        let stencil = self.stencil_axis(&boundary, support, kernel, axis);

        let mut corner = node;
        corner[axis] = match support {
            Support::Interior => node[axis] - kernel.border_width() as isize,
            Support::Negative(_) => 0,
            Support::Positive(_) => (self.size[axis] + 1) as isize - stencil.len() as isize,
        };

        self.apply_axis(corner, stencil, field, axis) * kernel.scale(spacing)
    }

    pub fn prolong(
        &self,
        boundary: impl Boundary<N>,
        kernel: impl CellKernel,
        supervertex: [usize; N],
        field: &[f64],
    ) -> f64 {
        for axis in 0..N {
            debug_assert!(supervertex[axis] <= 2 * self.size[axis]);
        }

        let node: [_; N] = array::from_fn(|axis| supervertex[axis] / 2);
        let flags: [_; N] = array::from_fn(|axis| supervertex[axis] % 2 == 1);

        let mut scale = 1.0;

        for _ in (0..N).filter(|&axis| flags[axis]) {
            scale *= kernel.scale();
        }

        let support: [_; N] = array::from_fn(|axis| {
            if flags[axis] {
                self.support_cell(&boundary, node[axis], kernel.border_width(), axis)
            } else {
                Support::Interior
            }
        });

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

            let kind = boundary.kind(face);

            match kind {
                // (BoundaryKind::Parity, false) => kernel.antisymmetric(border),
                // (BoundaryKind::Parity, true) => kernel.symmetric(border),
                BoundaryKind::Custom | BoundaryKind::Parity => kernel.interior(),
                BoundaryKind::Free | BoundaryKind::Radiative => kernel.free(border),
            }
        });

        let corner = array::from_fn(|axis| {
            if !flags[axis] {
                return node[axis] as isize;
            }

            match support[axis] {
                Support::Interior => node[axis] as isize - kernel.border_width() as isize,
                Support::Negative(_) => 0,
                Support::Positive(_) => {
                    self.size[axis] as isize + 1 - stencils[axis].len() as isize
                }
            }
        });
        self.apply(corner, stencils, field) * scale
    }
}

// ****************************
// Node Window ****************
// ****************************

/// Defines a rectagular region of a larger `NodeSpace`.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    use aeon_geometry::Rectangle;

    use super::*;
    use crate::{Kernels as _, Order};

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
            ghost: 3,
        };

        const BORDER: usize = 3;

        let supports = [
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Positive(2),
            Support::Positive(1),
            Support::Positive(0),
        ];

        for i in 0..9 {
            assert_eq!(space.support(&Quadrant, i, BORDER, 0), supports[i]);
        }
    }

    #[test]
    fn support_cell() {
        let space = NodeSpace {
            size: [8],
            ghost: 3,
        };

        const BORDER: usize = 3;

        let supports = [
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Interior,
            Support::Positive(2),
            Support::Positive(1),
            Support::Positive(0),
        ];

        for i in 0..8 {
            assert_eq!(space.support_cell(&Quadrant, i, BORDER, 0), supports[i]);
        }
    }

    fn eval_convergence(size: [usize; 2]) -> f64 {
        let space = NodeSpace { size, ghost: 2 };
        let bounds = Rectangle::UNIT;

        let mut field = vec![0.0; space.num_nodes()];

        for node in space.full_window().iter() {
            let index = space.index_from_node(node);
            let [x, y] = space.position(node, bounds);
            field[index] = x.sin() * y.sin();
        }

        let mut result: f64 = 0.0;

        for node in space.inner_window().iter() {
            let vertex = [node[0] as usize, node[1] as usize];
            let [x, y] = space.position(node, bounds);
            let numerical = space.evaluate(
                Quadrant,
                (
                    Order::<4>::derivative().clone(),
                    Order::<4>::derivative().clone(),
                ),
                bounds,
                vertex,
                &field,
            );
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
        let cspace = NodeSpace { size, ghost: 2 };
        let rspace = NodeSpace {
            size: size.map(|v| v * 2),
            ghost: 2,
        };

        let bounds = Rectangle::UNIT;

        let mut field = vec![0.0; cspace.num_nodes()];

        for node in cspace.full_window().iter() {
            let index = cspace.index_from_node(node);
            let [x, y] = cspace.position(node, bounds);
            field[index] = x.sin() * y.sin();
        }

        let mut result: f64 = 0.0;

        for node in rspace.inner_window().iter() {
            let vertex = [node[0] as usize, node[1] as usize];
            let [x, y] = rspace.position(node, bounds);
            let numerical = cspace.prolong(
                Quadrant,
                Order::<4>::interpolation().clone(),
                vertex,
                &field,
            );
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

    #[test]
    fn windows() {
        let space = NodeSpace {
            size: [10; 2],
            ghost: 2,
        };

        // Regions
        assert_eq!(
            space.region_window(2, Region::new([Side::Left, Side::Right])),
            NodeWindow {
                origin: [-2, 11],
                size: [2, 2],
            }
        );
        assert_eq!(
            space.region_window(2, Region::new([Side::Middle, Side::Right])),
            NodeWindow {
                origin: [0, 11],
                size: [11, 2]
            }
        );

        // Faces
        assert_eq!(
            space.face_window(Face::positive(1)),
            NodeWindow {
                origin: [0, 10],
                size: [11, 1]
            }
        );

        // Faces disjoint
        assert_eq!(
            space.face_window_disjoint(Face::negative(0)),
            NodeWindow {
                origin: [0, 0],
                size: [1, 11],
            }
        );
        assert_eq!(
            space.face_window_disjoint(Face::negative(1)),
            NodeWindow {
                origin: [1, 0],
                size: [10, 1],
            }
        );
        assert_eq!(
            space.face_window_disjoint(Face::positive(0)),
            NodeWindow {
                origin: [10, 1],
                size: [1, 10],
            }
        );
        assert_eq!(
            space.face_window_disjoint(Face::positive(1)),
            NodeWindow {
                origin: [1, 10],
                size: [9, 1],
            }
        );
    }
}
