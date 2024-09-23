use bumpalo::Bump;

#[derive(Debug, Default)]
pub struct MeshStore {
    arena: Bump,
}

impl MeshStore {
    /// Allocates scratch data for use by the current thread.
    pub fn scratch(&self, len: usize) -> &mut [f64] {
        self.arena.alloc_slice_fill_default(len)
    }

    /// Resets memory cached in mesh store.
    pub fn reset(&mut self) {
        self.arena.reset();
    }
}
