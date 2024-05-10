use crate::common::{
    node_from_vertex, Boundary, FDDerivative, FDDissipation, FDSecondDerivative, Kernel, NodeSpace,
};
use crate::geometry::CartesianIter;

use std::ops::Range;

#[derive(Debug)]
pub struct Block<const N: usize> {
    pub(crate) space: NodeSpace<N>,
    range: Range<usize>,
}

impl<const N: usize> Block<N> {
    pub fn new(space: NodeSpace<N>, range: Range<usize>) -> Self {
        assert!(range.len() == space.node_count());

        Self { space, range }
    }

    pub fn node_count(&self) -> usize {
        self.range.len()
    }

    pub fn local_from_global(&self) -> Range<usize> {
        self.range.clone()
    }

    /// Iterates over the vertices in the block.
    pub fn iter(&self) -> CartesianIter<N> {
        self.space.vertex_space().iter()
    }

    /// Computes a linear index from a vertex.
    pub fn index_from_vertex(&self, vertex: [usize; N]) -> usize {
        self.space.index_from_node(node_from_vertex(vertex))
    }

    pub fn value(&self, vertex: [usize; N], src: &[f64]) -> f64 {
        self.space.value(node_from_vertex(vertex), src)
    }

    pub fn set_value(&self, vertex: [usize; N], v: f64, dest: &mut [f64]) {
        self.space.set_value(node_from_vertex(vertex), v, dest)
    }

    pub fn position(&self, vertex: [usize; N]) -> [f64; N] {
        self.space.position(node_from_vertex(vertex))
    }

    pub fn evaluate<K: Kernel, B: Boundary<N>>(
        &self,
        kernel: &K,
        boundary: &B,
        src: &[f64],
        dest: &mut [f64],
    ) {
        self.space.evaluate(kernel, boundary, src, dest)
    }
}

pub trait BlockExt<const N: usize> {
    fn derivative<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary<N>,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDDerivative<ORDER>: Kernel;

    fn second_derivative<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary<N>,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDSecondDerivative<ORDER>: Kernel;

    fn dissipation<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary<N>,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDDissipation<ORDER>: Kernel;
}

impl<const N: usize> BlockExt<N> for Block<N> {
    fn derivative<const ORDER: usize>(
        &self,
        axis: usize,
        boundary: &impl Boundary<N>,
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
        boundary: &impl Boundary<N>,
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
        boundary: &impl Boundary<N>,
        src: &[f64],
        dest: &mut [f64],
    ) where
        FDDissipation<ORDER>: Kernel,
    {
        self.evaluate(&FDDissipation::<ORDER>::new(axis), boundary, src, dest)
    }
}
