use crate::common::{Convolution, NodeSpace, NodeSpaceAxis};
use crate::geometry::CartesianIterator;

#[derive(Debug, Clone)]
pub struct Block<const N: usize> {
    space: NodeSpace<N>,
    offset: usize,
    total: usize,
}

impl<const N: usize> Block<N> {
    pub fn new(space: NodeSpace<N>, offset: usize) -> Self {
        let total = space.len();

        Self {
            space,
            offset,
            total,
        }
    }

    pub fn len(self: &Self) -> usize {
        self.total
    }

    pub fn position(self: &Self, node: [usize; N]) -> [f64; N] {
        self.space.position(node)
    }

    pub fn iter(self: &Self) -> CartesianIterator<N> {
        self.space.vertex_space().iter()
    }

    pub fn global_from_local(self: &Self, node: [usize; N]) -> usize {
        self.offset + self.space.vertex_space().linear_from_cartesian(node)
    }

    pub fn axis<'a>(self: &'a Self, axis: usize) -> BlockAxis<'a, N> {
        BlockAxis {
            space: self.space.axis(axis),
            offset: self.offset,
            total: self.total,
        }
    }
}

pub struct BlockAxis<'a, const N: usize> {
    space: NodeSpaceAxis<'a, N>,
    offset: usize,
    total: usize,
}

impl<'a, const N: usize> BlockAxis<'a, N> {
    pub fn evaluate_auxillary<K: Convolution<N>>(
        self: &Self,
        convolution: &K,
        src: &[f64],
        dest: &mut [f64],
    ) {
        let block_src = &src[self.offset..self.offset + self.total];
        self.space.evaluate(convolution, block_src, dest);
    }

    pub fn evaluate<K: Convolution<N>>(
        self: &Self,
        convolution: &K,
        src: &[f64],
        dest: &mut [f64],
    ) {
        self.space.evaluate(convolution, src, dest);
    }

    pub fn evaluate_diag_auxillary<K: Convolution<N>>(
        self: &Self,
        convolution: &K,
        src: &[f64],
        dest: &mut [f64],
    ) {
        let block_src = &src[self.offset..self.offset + self.total];
        self.space.evaluate_diag(convolution, block_src, dest);
    }

    pub fn evaluate_diag<K: Convolution<N>>(
        self: &Self,
        convolution: &K,
        src: &[f64],
        dest: &mut [f64],
    ) {
        self.space.evaluate_diag(convolution, src, dest);
    }
}

pub trait Operator<const N: usize> {
    fn apply(self: &Self, block: Block<N>, src: &[f64], dest: &mut [f64]);
    fn apply_diag(self: &Self, block: Block<N>, src: &[f64], dest: &mut [f64]);
}
