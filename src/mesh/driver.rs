use crate::common::Boundary;
use crate::mesh::{
    Mesh, Operator, Projection, SystemOperator, SystemProjection, SystemSlice, SystemSliceMut,
};

use bumpalo::Bump;

#[derive(Debug, Default)]
pub struct Driver {
    pool: MemPool,
}

impl Driver {
    pub fn new() -> Self {
        Self {
            pool: MemPool::new(),
        }
    }

    pub fn fill_boundary<const N: usize, B: Boundary<N>>(
        &mut self,
        mesh: &Mesh<N>,
        boundary: &B,
        dest: &mut [f64],
    ) {
        let block = mesh.base_block();
        block.space.fill_boundary(boundary, dest);
    }

    pub fn project<const N: usize, P: Projection<N>>(
        &mut self,
        mesh: &Mesh<N>,
        projection: &P,
        dest: &mut [f64],
    ) {
        let block = mesh.base_block();
        projection.evaluate(block, &self.pool, dest);
        self.pool.reset();
    }

    pub fn apply<const N: usize, O: Operator<N>>(
        &mut self,
        mesh: &Mesh<N>,
        operator: &O,
        src: &[f64],
        dest: &mut [f64],
    ) {
        let block = mesh.base_block();
        operator.apply(block, &self.pool, src, dest);
        self.pool.reset();
    }

    pub fn project_system<const N: usize, P: SystemProjection<N>>(
        &mut self,
        mesh: &Mesh<N>,
        projection: &P,
        dest: SystemSliceMut<'_, P::Label>,
    ) {
        let block = mesh.base_block();
        projection.evaluate(block, &self.pool, dest);
        self.pool.reset();
    }

    pub fn apply_system<const N: usize, O: SystemOperator<N>>(
        &mut self,
        mesh: &Mesh<N>,
        operator: &O,
        src: SystemSlice<'_, O::Label>,
        dest: SystemSliceMut<'_, O::Label>,
    ) {
        let block = mesh.base_block();
        operator.apply(block, &self.pool, src, dest);
        self.pool.reset();
    }
}

/// A simple arena allocator, wrapping a `bumpalo::Bump`.
#[derive(Debug)]
pub struct MemPool {
    bump: Bump,
}

impl MemPool {
    /// Creates a new `MemPool`.
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    /// Allocates a slice of scalar values, used for intermediate values when, for example, computing derivatives in
    /// operators.
    pub fn alloc_scalar(&self, len: usize) -> &mut [f64] {
        self.bump.alloc_slice_fill_default(len)
    }

    /// Resets the mempool without.
    pub fn reset(&mut self) {
        self.bump.reset();
    }
}

impl Default for MemPool {
    fn default() -> Self {
        Self::new()
    }
}
