use crate::common::NodeSpace;
use crate::geometry::CartesianIterator;

use super::{kernel::FDDissipation, BoundarySet, FDDerivative, FDSecondDerivative, Kernel};

#[derive(Debug, Clone)]
pub struct Block<const N: usize> {
    space: NodeSpace<N>,
    offset: usize,
    total: usize,
}

impl<const N: usize> Block<N> {
    pub fn new(space: NodeSpace<N>, offset: usize, total: usize) -> Self {
        assert!(total == space.len());
        Self {
            space,
            offset,
            total,
        }
    }

    /// Number of nodes in block.
    pub fn len(self: &Self) -> usize {
        self.total
    }

    /// Position of a given node in the block.
    pub fn position(self: &Self, node: [usize; N]) -> [f64; N] {
        self.space.position(node)
    }

    /// Iterates over the nodes in the block.
    pub fn iter(self: &Self) -> CartesianIterator<N> {
        self.space.vertex_space().iter()
    }

    // pub fn global_from_local(self: &Self, node: [usize; N]) -> usize {
    //     self.offset + self.space.vertex_space().linear_from_cartesian(node)
    // }

    pub fn auxillary<'a>(self: &Self, src: &'a [f64]) -> &'a [f64] {
        &src[self.offset..(self.offset + self.total)]
    }

    pub fn auxillary_mut<'a>(self: &Self, src: &'a mut [f64]) -> &'a mut [f64] {
        &mut src[self.offset..(self.offset + self.total)]
    }

    pub fn evaluate<K: Kernel, B: BoundarySet<N>>(
        self: &Self,
        axis: usize,
        set: &B,
        src: &[f64],
        dest: &mut [f64],
    ) {
        self.space.axis(axis).evaluate::<K, B>(set, src, dest);
    }

    pub fn evaluate_diag<K: Kernel, B: BoundarySet<N>>(
        self: &Self,
        axis: usize,
        set: &B,
        dest: &mut [f64],
    ) {
        self.space.axis(axis).evaluate_diag::<K, B>(set, dest);
    }

    pub fn diritchlet(self: &Self, axis: usize, face: bool, dest: &mut [f64]) {
        self.space.axis(axis).apply_diritchlet_bc(face, dest);
    }

    pub fn axis<'a, const ORDER: usize>(self: &'a Self, axis: usize) -> BlockAxis<'a, N, ORDER> {
        BlockAxis::<N, ORDER> { block: self, axis }
    }
}

pub struct BlockAxis<'a, const N: usize, const ORDER: usize> {
    block: &'a Block<N>,
    axis: usize,
}

impl<'a, const N: usize> BlockAxis<'a, N, 2> {
    pub fn derivative<B: BoundarySet<N>>(self: &Self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDDerivative<2>, B>(self.axis, set, src, dest)
    }

    pub fn derivative_diag<B: BoundarySet<N>>(self: &Self, set: &B, dest: &mut [f64]) {
        self.block
            .evaluate_diag::<FDDerivative<2>, B>(self.axis, set, dest)
    }

    pub fn second_derivative<B: BoundarySet<N>>(
        self: &Self,
        set: &B,
        src: &[f64],
        dest: &mut [f64],
    ) {
        self.block
            .evaluate::<FDSecondDerivative<2>, B>(self.axis, set, src, dest)
    }

    pub fn second_derivative_diag<B: BoundarySet<N>>(self: &Self, set: &B, dest: &mut [f64]) {
        self.block
            .evaluate_diag::<FDSecondDerivative<2>, B>(self.axis, set, dest)
    }

    pub fn dissipation<B: BoundarySet<N>>(self: &Self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDDissipation<2>, B>(self.axis, set, src, dest)
    }
}

impl<'a, const N: usize> BlockAxis<'a, N, 4> {
    pub fn derivative<B: BoundarySet<N>>(self: &Self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDDerivative<4>, B>(self.axis, set, src, dest)
    }

    pub fn derivative_diag<B: BoundarySet<N>>(self: &Self, set: &B, dest: &mut [f64]) {
        self.block
            .evaluate_diag::<FDDerivative<4>, B>(self.axis, set, dest)
    }

    pub fn second_derivative<B: BoundarySet<N>>(
        self: &Self,
        set: &B,
        src: &[f64],
        dest: &mut [f64],
    ) {
        self.block
            .evaluate::<FDSecondDerivative<4>, B>(self.axis, set, src, dest)
    }

    pub fn second_derivative_diag<B: BoundarySet<N>>(self: &Self, set: &B, dest: &mut [f64]) {
        self.block
            .evaluate_diag::<FDSecondDerivative<4>, B>(self.axis, set, dest)
    }

    pub fn dissipation<B: BoundarySet<N>>(self: &Self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDDissipation<4>, B>(self.axis, set, src, dest)
    }
}
