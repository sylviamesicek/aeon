#![allow(clippy::len_without_is_empty)]

mod block;
mod boundary;
mod kernel;
mod space;

pub use block::{Block, BlockExt};
pub use boundary::{
    AntiSymmetricBoundary, AsymptoticFlatness, Boundary, BoundarySet, FreeBoundary, Mixed,
    RobinBoundary, SymmetricBoundary,
};
pub use kernel::{FDDerivative, FDSecondDerivative, Kernel};
pub use space::{NodeSpace, NodeSpaceAxis};

use crate::arena::Arena;

pub trait BoundaryCallback<const N: usize> {
    fn axis<B: BoundarySet<N>>(&mut self, axis: usize, boundary: &B);
}

pub trait Operator<const N: usize> {
    /// Applies the operator to the source function on the block, storing the result in dest.
    fn apply(&self, arena: &Arena, block: &Block<N>, src: &[f64], dest: &mut [f64]);
    fn apply_diag(&self, arena: &Arena, block: &Block<N>, dest: &mut [f64]);

    fn boundary(&self, callback: impl BoundaryCallback<N>);
}

pub trait Projection<const N: usize> {
    fn evaluate(&self, arena: &Arena, block: &Block<N>, dest: &mut [f64]);
}
