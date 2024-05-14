use crate::{
    common::GhostBoundary,
    mesh::{Boundary, Mesh, Operator, Projection, SystemLabel, SystemSlice, SystemSliceMut},
};

use bumpalo::Bump;

#[derive(Debug, Default)]
pub struct Driver {
    pub(crate) pool: MemPool,
}

impl Driver {
    pub fn new() -> Self {
        Self {
            pool: MemPool::new(),
        }
    }

    pub fn fill_boundary<const N: usize, B: Boundary>(
        &mut self,
        mesh: &Mesh<N>,
        boundary: &B,
        mut dest: SystemSliceMut<'_, B::Label>,
    ) {
        let block = mesh.base_block();
        let range = block.local_from_global();

        for field in B::Label::fields() {
            block.space.fill_boundary(
                &boundary.boundary(field.clone()),
                &mut dest.field_mut(field.clone())[range.clone()],
            );
        }
    }

    pub fn fill_boundary_scalar<const N: usize, B: GhostBoundary>(
        &mut self,
        mesh: &Mesh<N>,
        boundary: &B,
        dest: &mut [f64],
    ) {
        let block = mesh.base_block();
        let range = block.local_from_global();
        block.space.fill_boundary(boundary, &mut dest[range]);
    }

    pub fn project<const N: usize, P: Projection<N>>(
        &mut self,
        mesh: &Mesh<N>,
        projection: &P,
        dest: SystemSliceMut<'_, P::Label>,
    ) {
        let block = mesh.base_block();
        projection.evaluate(block, &self.pool, dest);
        self.pool.reset();
    }

    pub fn apply<const N: usize, O: Operator<N>>(
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
