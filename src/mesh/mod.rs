use crate::common::NodeSpace;
use crate::geometry::Rectangle;
use crate::system::{SystemLabel, SystemSliceMut};

mod block;
mod driver;
mod model;

pub use block::{Block, BlockExt};
pub use driver::{Driver, MemPool};
pub use model::Model;

pub trait Operator<const N: usize> {
    type Output: SystemLabel;

    fn eval(&self, block: Block<N>, pool: &MemPool, dest: SystemSliceMut<'_, Self::Output>);
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

    /// Physical bounds of the mesh.
    pub fn bounds(&self) -> Rectangle<N> {
        self.bounds.clone()
    }

    pub fn project<O: Operator<N>>(
        &self,
        driver: &mut Driver,
        operator: &O,
        dest: SystemSliceMut<'_, O::Output>,
    ) {
        let block = self.base_block();

        operator.eval(block, &driver.pool, dest);
        driver.pool.reset();
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
