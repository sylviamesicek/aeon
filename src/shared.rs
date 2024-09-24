use std::cell::UnsafeCell;

/// Represents a reference to a slice which may be shared among threads. This uses `UnsafeCell` to
/// Uphold rust's immutability gaurentees, but it is the responsibility of the user that the values
/// at two different indices are not aliased improperly.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct SharedSlice<'a, T>(&'a [SyncUnsafeCell<T>]);

impl<'a, T> SharedSlice<'a, T> {
    pub fn new(data: &'a mut [T]) -> Self {
        Self(unsafe { &*(data as *mut [T] as *const [SyncUnsafeCell<T>]) })
    }

    pub unsafe fn get(self, index: usize) -> &'a T {
        &*self.0[index].get()
    }

    pub unsafe fn get_mut(self, index: usize) -> &'a mut T {
        &mut *self.0[index].get()
    }
}

/// Wrapper around `UnsafeCell` that also implements sync.
#[repr(transparent)]
struct SyncUnsafeCell<T: ?Sized> {
    value: UnsafeCell<T>,
}

unsafe impl<T: ?Sized + Sync> Sync for SyncUnsafeCell<T> {}

// Identical interface as UnsafeCell:
impl<T: ?Sized> SyncUnsafeCell<T> {
    pub const fn get(&self) -> *mut T {
        self.value.get()
    }
}
