use bumpalo::Bump;

pub struct Driver {
    pub(crate) pool: MemPool,
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
