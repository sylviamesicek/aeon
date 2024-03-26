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

    pub fn size(&self) -> [usize; N] {
        self.space.vertex_size()
    }

    /// Number of nodes in block.
    pub fn len(&self) -> usize {
        self.total
    }

    /// Position of a given node in the block.
    pub fn position(&self, node: [usize; N]) -> [f64; N] {
        self.space.position(node)
    }

    /// Iterates over the nodes in the block.
    pub fn iter(&self) -> CartesianIterator<N> {
        self.space.vertex_space().iter()
    }

    // pub fn global_from_local(&self, node: [usize; N]) -> usize {
    //     self.offset + self.space.vertex_space().linear_from_cartesian(node)
    // }

    pub fn auxillary<'a>(&self, src: &'a [f64]) -> &'a [f64] {
        &src[self.offset..(self.offset + self.total)]
    }

    pub fn auxillary_mut<'a>(&self, src: &'a mut [f64]) -> &'a mut [f64] {
        &mut src[self.offset..(self.offset + self.total)]
    }

    pub fn evaluate<K: Kernel, B: BoundarySet<N>>(
        &self,
        axis: usize,
        set: &B,
        src: &[f64],
        dest: &mut [f64],
    ) {
        self.space.axis(axis).evaluate::<K, B>(set, src, dest);
    }

    pub fn evaluate_diag<K: Kernel, B: BoundarySet<N>>(
        &self,
        axis: usize,
        set: &B,
        dest: &mut [f64],
    ) {
        self.space.axis(axis).evaluate_diag::<K, B>(set, dest);
    }

    pub fn diritchlet(&self, axis: usize, face: bool, dest: &mut [f64]) {
        self.space.axis(axis).apply_diritchlet_bc(face, dest);
    }

    pub fn axis<const ORDER: usize>(&self, axis: usize) -> BlockAxis<'_, N, ORDER> {
        BlockAxis::<N, ORDER> { block: self, axis }
    }
}

pub struct BlockAxis<'a, const N: usize, const ORDER: usize> {
    block: &'a Block<N>,
    axis: usize,
}

impl<'a, const N: usize> BlockAxis<'a, N, 2> {
    pub fn derivative<B: BoundarySet<N>>(&self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDDerivative<2>, B>(self.axis, set, src, dest)
    }

    pub fn derivative_diag<B: BoundarySet<N>>(&self, set: &B, dest: &mut [f64]) {
        self.block
            .evaluate_diag::<FDDerivative<2>, B>(self.axis, set, dest)
    }

    pub fn second_derivative<B: BoundarySet<N>>(&self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDSecondDerivative<2>, B>(self.axis, set, src, dest)
    }

    pub fn second_derivative_diag<B: BoundarySet<N>>(&self, set: &B, dest: &mut [f64]) {
        self.block
            .evaluate_diag::<FDSecondDerivative<2>, B>(self.axis, set, dest)
    }

    pub fn dissipation<B: BoundarySet<N>>(&self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDDissipation<2>, B>(self.axis, set, src, dest)
    }
}

impl<'a, const N: usize> BlockAxis<'a, N, 4> {
    pub fn derivative<B: BoundarySet<N>>(&self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDDerivative<4>, B>(self.axis, set, src, dest)
    }

    pub fn derivative_diag<B: BoundarySet<N>>(&self, set: &B, dest: &mut [f64]) {
        self.block
            .evaluate_diag::<FDDerivative<4>, B>(self.axis, set, dest)
    }

    pub fn second_derivative<B: BoundarySet<N>>(&self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDSecondDerivative<4>, B>(self.axis, set, src, dest)
    }

    pub fn second_derivative_diag<B: BoundarySet<N>>(&self, set: &B, dest: &mut [f64]) {
        self.block
            .evaluate_diag::<FDSecondDerivative<4>, B>(self.axis, set, dest)
    }

    pub fn dissipation<B: BoundarySet<N>>(&self, set: &B, src: &[f64], dest: &mut [f64]) {
        self.block
            .evaluate::<FDDissipation<4>, B>(self.axis, set, src, dest)
    }
}

#[cfg(test)]
mod tests {
    use crate::common::*;
    use crate::prelude::Rectangle;

    /// A comprehensive test of 2nd order accurate stencils applied
    /// to one dimensional domains.
    #[test]
    fn evaluation() {
        // **********************
        // Set up Block

        let space = NodeSpace {
            bounds: Rectangle {
                size: [2.0],
                origin: [-1.0],
            },
            size: [100],
        };

        let offset = 0;
        let total = space.len();

        let spacing = 1.0 / 50.0;

        assert_eq!(space.spacing(0), spacing);

        let block = Block::new(space, offset, total);

        assert_eq!(block.len(), 101);

        // **********************
        // Initial data

        let mut field = vec![0.0; block.len()];
        let mut explicit = vec![0.0; block.len()];

        for (i, vertex) in block.iter().enumerate() {
            let position = block.position(vertex)[0];
            field[i] = (-position * position / 10.0).exp();
        }

        for i in 0..block.len() {
            let position = -1.0 + spacing * (i as f64);
            explicit[i] = (-position * position / 10.0).exp();
        }

        assert_eq!(field, explicit);

        // **********************
        // Evolution

        let mut derivative = vec![0.0; block.len()];

        for _ in 0..10 {
            // Field
            block.axis::<2>(0).derivative(
                &Mixed::new(
                    Simple::new(SymmetricBoundary::<2>),
                    Simple::new(FreeBoundary),
                ),
                &field,
                &mut derivative,
            );

            for i in 0..block.len() {
                field[i] -= 0.01 * derivative[i];
            }

            // Explicit
            derivative[0] = 0.0;

            for i in 1..block.len() - 1 {
                derivative[i] = -0.5 * explicit[i - 1] + 0.5 * explicit[i + 1];
            }

            derivative[block.len() - 1] = 0.5 * explicit[block.len() - 3]
                - 2.0 * explicit[block.len() - 2]
                + 1.5 * explicit[block.len() - 1];

            for i in derivative.iter_mut() {
                *i *= 1.0 / spacing;
            }

            for i in 0..block.len() {
                explicit[i] -= 0.01 * derivative[i];
            }
        }

        for i in 0..block.len() {
            assert!((explicit[i] - field[i]).abs() < 10e-15);
        }
    }
}
