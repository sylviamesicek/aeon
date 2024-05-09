use std::array::from_fn;

use crate::common::{Boundary, NodeSpace};
use crate::geometry::Rectangle;
use crate::system::{SystemLabel, SystemSlice, SystemSliceMut};

mod block;
mod driver;
mod model;

pub use block::{Block, BlockExt};
pub use driver::{Driver, MemPool};
pub use model::Model;

pub trait Projection<const N: usize> {
    type Output: SystemLabel;

    fn evaluate(&self, block: Block<N>, pool: &MemPool, dest: SystemSliceMut<'_, Self::Output>);
}

pub trait Operator<const N: usize> {
    type Output: SystemLabel;

    fn apply(
        &self,
        block: Block<N>,
        pool: &MemPool,
        src: SystemSlice<'_, Self::Output>,
        dest: SystemSliceMut<'_, Self::Output>,
    );
}

#[derive(Debug, Clone)]
pub struct Mesh<const N: usize> {
    /// Uniform bounds for mesh
    bounds: Rectangle<N>,
    /// Number of cells on base block
    size: [usize; N],
    /// Number of ghost cells.
    ghost: usize,
}

impl<const N: usize> Mesh<N> {
    /// Constructs a new mesh.
    pub fn new(bounds: Rectangle<N>, size: [usize; N], ghost: usize) -> Self {
        Self {
            bounds,
            size,
            ghost,
        }
    }

    pub fn node_count(&self) -> usize {
        self.base_block().node_count()
    }

    /// Physical bounds of the mesh.
    pub fn bounds(&self) -> Rectangle<N> {
        self.bounds.clone()
    }

    pub fn fill_boundary<B: Boundary<N>>(&self, boundary: &B, dest: &mut [f64]) {
        let block = self.base_block();
        block.space.fill_boundary(boundary, dest);
    }

    pub fn project<P: Projection<N>>(
        &self,
        driver: &mut Driver,
        projection: &P,
        dest: SystemSliceMut<'_, P::Output>,
    ) {
        let block = self.base_block();
        projection.evaluate(block, &driver.pool, dest);
        driver.pool.reset();
    }

    pub fn apply<O: Operator<N>>(
        &self,
        driver: &mut Driver,
        operator: &O,
        src: SystemSlice<'_, O::Output>,
        dest: SystemSliceMut<'_, O::Output>,
    ) {
        let block = self.base_block();
        operator.apply(block, &driver.pool, src, dest);
        driver.pool.reset();
    }

    pub fn minimum_spacing(&self) -> [f64; N] {
        from_fn(|axis| self.base_block().space.spacing(axis))
    }

    fn base_block(&self) -> Block<N> {
        let space = NodeSpace {
            bounds: self.bounds.clone(),
            size: self.size,
            ghost: self.ghost,
        };

        let node_count = space.node_count();

        Block::new(space, 0..node_count)
    }
}
