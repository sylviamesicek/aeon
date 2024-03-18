use bumpalo::Bump;

/// A simple arena allocator, wrapping a `bumpalo::Bump`.
#[derive(Debug)]
pub struct Arena {
    bump: Bump,
}

impl Arena {
    /// Creates a new arrena.
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    /// Allocates a new slice of the given length, filled with the default value of a type.
    pub fn alloc<T: Default>(self: &Self, len: usize) -> &mut [T] {
        self.bump.alloc_slice_fill_default(len)
    }

    /// Resets the arena without calling drop.
    pub fn reset(self: &mut Self) {
        self.bump.reset();
    }
}
