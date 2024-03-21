#![allow(clippy::len_without_is_empty)]

mod block;
mod boundary;
mod kernel;
mod space;

pub use block::{Block, BlockAxis};
pub use boundary::{
    AntiSymmetricBoundary, AsymptoticFlatness, Boundary, BoundarySet, FreeBoundary, Mixed,
    RobinBoundary, Simple, SymmetricBoundary,
};
pub use kernel::{FDDerivative, FDSecondDerivative, Kernel};
pub use space::{NodeSpace, NodeSpaceAxis};

use crate::arena::Arena;

pub trait Operator<const N: usize> {
    /// Applies the operator to the source function on the block, storing the result in dest.
    fn apply(&self, arena: &Arena, block: &Block<N>, src: &[f64], dest: &mut [f64]);
    fn apply_diag(&self, arena: &Arena, block: &Block<N>, dest: &mut [f64]);

    fn diritchlet(axis: usize, face: bool) -> bool;
}

pub trait Projection<const N: usize> {
    fn evaluate(&self, arena: &Arena, block: &Block<N>, dest: &mut [f64]);
}
