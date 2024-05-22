use crate::{
    common::Boundary,
    mesh::{
        Mesh, Operator, Projection, SystemBoundary, SystemLabel, SystemOperator, SystemProjection,
        SystemSlice, SystemSliceMut,
    },
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
        dest: &mut [f64],
    ) {
        let block = mesh.base_block();
        let range = block.local_from_global();
        block.space.fill_boundary(boundary, &mut dest[range]);
    }

    pub fn fill_boundary_system<const N: usize, B: SystemBoundary>(
        &mut self,
        mesh: &Mesh<N>,
        boundary: &B,
        mut dest: SystemSliceMut<'_, B::Label>,
    ) {
        let block = mesh.base_block();
        let range = block.local_from_global();

        for field in B::Label::fields() {
            block.space.fill_boundary(
                &boundary.field(field.clone()),
                &mut dest.field_mut(field.clone())[range.clone()],
            );
        }
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

    /// Computes the approximate l^2 functional norm of the scalar field.
    pub fn norm<const N: usize>(&mut self, mesh: &Mesh<N>, src: &[f64]) -> f64 {
        let block = mesh.base_block();
        let src = &src[block.local_from_global()];

        let mut result = 0.0;

        for vertex in block.iter() {
            let index = block.index_from_vertex(vertex);

            result += src[index] * src[index];
        }

        // Scale by spacing in each direction to approximate
        // functional norm.
        for spacing in block.space.spacing() {
            result *= spacing;
        }

        result.sqrt()
    }

    /// Returns the maximum norm of any of the constituent scalar fields.
    pub fn norm_system<const N: usize, Label: SystemLabel>(
        &mut self,
        mesh: &Mesh<N>,
        src: SystemSlice<'_, Label>,
    ) -> f64 {
        Label::fields()
            .into_iter()
            .map(|label| self.norm(mesh, src.field(label)))
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
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
