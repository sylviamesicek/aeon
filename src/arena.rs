use bumpalo::Bump;

#[derive(Debug)]
pub struct Arena {
    bump: Bump,
}

impl Arena {
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    pub fn alloc<T: Default>(self: &Self, len: usize) -> &mut [T] {
        self.bump.alloc_slice_fill_default(len)
    }

    pub fn reset(self: &mut Self) {
        self.bump.reset();
    }
}
