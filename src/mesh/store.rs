use std::cell::UnsafeCell;

use bumpalo::Bump;
use thread_local::ThreadLocal;

/// A per-thread store that contains memory pools and caches for multi-threaded
/// workloads.
#[derive(Debug, Default)]
pub struct MeshStore {
    /// Linear allocator for each thread.
    arena: Bump,
}

impl MeshStore {
    /// Allocates scratch data for use by the current thread.
    pub fn scratch<T: Default>(&self, len: usize) -> &mut [T] {
        self.arena.alloc_slice_fill_default(len)
    }

    /// Resets memory cached in mesh store.
    pub fn reset(&mut self) {
        self.arena.reset();
    }
}

#[derive(Debug)]
pub struct UnsafeThreadCache<T: Send> {
    pool: ThreadLocal<UnsafeCell<T>>,
}

impl<T: Send> UnsafeThreadCache<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<T: Send + Default> UnsafeThreadCache<T> {
    pub unsafe fn get_or_default(&self) -> &mut T {
        unsafe { &mut *self.pool.get_or_default().get() }
    }
}

impl<T: Send> Default for UnsafeThreadCache<T> {
    fn default() -> Self {
        Self {
            pool: ThreadLocal::default(),
        }
    }
}
