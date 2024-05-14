use crate::common::{
    node_from_vertex, Boundary, FDDerivative, FDDissipation, FDSecondDerivative, Kernel, NodeSpace,
};
use crate::geometry::{CartesianIter, Face, PlaneIterator};

use std::ops::Range;

/// An interface for interacting with blocks of an octtree `Mesh`.
///
/// This allows for operators (specifically `Projection`s) to operator on each block concurrently,
/// which in turn opens of the potential for better parallelization.
#[derive(Debug, Clone)]
pub struct Block<const N: usize> {
    pub(crate) space: NodeSpace<N>,
    range: Range<usize>,
}

impl<const N: usize> Block<N> {
    /// Creates a new block from a nodespace and range (usually only called internally).
    pub fn new(space: NodeSpace<N>, range: Range<usize>) -> Self {
        assert!(range.len() == space.node_count());

        Self { space, range }
    }

    /// Number of nodes in this block
    pub fn node_count(&self) -> usize {
        self.range.len()
    }

    pub fn vertex_size(&self) -> [usize; N] {
        self.space.vertex_size()
    }

    /// A map from global degrees of freedom to local nodes.
    pub fn local_from_global(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Iterates over the vertices in the block.
    pub fn iter(&self) -> CartesianIter<N> {
        self.space.vertex_space().iter()
    }

    /// Iterates over the vertices on a plane in the block.
    pub fn plane(&self, axis: usize, intercept: usize) -> PlaneIterator<N> {
        self.space.vertex_space().plane(axis, intercept)
    }

    /// Iterates over the vertices on a face in the block
    pub fn face_plane(&self, face: Face) -> PlaneIterator<N> {
        let intercept = if face.side {
            self.space.vertex_size()[face.axis] - 1
        } else {
            0
        };
        self.plane(face.axis, intercept)
    }

    /// Computes a linear index from a vertex.
    pub fn index_from_vertex(&self, vertex: [usize; N]) -> usize {
        self.space.index_from_node(node_from_vertex(vertex))
    }

    /// Computes the value of the local field at the vertex.
    pub fn value(&self, vertex: [usize; N], src: &[f64]) -> f64 {
        self.space.value(node_from_vertex(vertex), src)
    }

    /// Sets the value of a local field at the vertex.
    pub fn set_value(&self, vertex: [usize; N], v: f64, dest: &mut [f64]) {
        self.space.set_value(node_from_vertex(vertex), v, dest)
    }

    /// Computes the position of a vertex.
    pub fn position(&self, vertex: [usize; N]) -> [f64; N] {
        self.space.position(node_from_vertex(vertex))
    }

    /// Evaluates the operation of a kernel working on a field with the specified boundary conditions.
    pub fn evaluate<K: Kernel, B: Boundary>(
        &self,
        kernel: &K,
        boundary: &B,
        src: &[f64],
        dest: &mut [f64],
    ) {
        self.space.evaluate(kernel, boundary, src, dest)
    }
}

/// Extension methods for `Block`s.
pub trait BlockExt<const N: usize> {
    /// Aproximates a derivative to the given order of accuracy.
    fn derivative<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDDerivative<ORDER>: Kernel;

    /// Aproximates a second derivative to the given order of accuracy.
    fn second_derivative<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDSecondDerivative<ORDER>: Kernel;

    /// Computes a Kriess-Oliger dissipation of the given order of accuracy.
    fn dissipation<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDDissipation<ORDER>: Kernel;
}

impl<const N: usize> BlockExt<N> for Block<N> {
    fn derivative<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDDerivative<ORDER>: Kernel,
    {
        self.evaluate(&FDDerivative::<ORDER>::new(axis), boundary, src, dest)
    }

    fn second_derivative<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDSecondDerivative<ORDER>: Kernel,
    {
        self.evaluate(&FDSecondDerivative::<ORDER>::new(axis), boundary, src, dest)
    }

    fn dissipation<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDDissipation<ORDER>: Kernel,
    {
        self.evaluate(&FDDissipation::<ORDER>::new(axis), boundary, src, dest)
    }
}
