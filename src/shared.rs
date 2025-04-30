use std::{cell::UnsafeCell, ops::Range, slice};

/// Represents a reference to a slice which may be shared among threads. This uses `UnsafeCell` to
/// Uphold rust's immutability gaurentees, but it is the responsibility of the user that the values
/// at two different indices are not aliased improperly.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct SharedSlice<'a, T>(&'a [SyncUnsafeCell<T>]);

impl<'a, T> SharedSlice<'a, T> {
    pub fn new(data: &'a mut [T]) -> Self {
        Self(unsafe {
            slice::from_raw_parts(data.as_ptr() as *const SyncUnsafeCell<T>, data.len())
        })
    }

    pub fn len(self) -> usize {
        self.0.len()
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Retrieves a reference to the `index`th element of the slice.
    ///
    /// # Safety
    /// No mutable references to this element may exist when this function is called.
    pub unsafe fn get(self, index: usize) -> &'a T {
        unsafe { &*self.0[index].get() }
    }

    /// Retrieves a mutable reference to the `index`th element of the slice.
    ///
    /// # Safety
    /// No mutable or immutable references to this element may exist when this function is called.
    pub unsafe fn get_mut(self, index: usize) -> &'a mut T {
        unsafe { &mut *self.0[index].get() }
    }

    pub unsafe fn slice(self, range: Range<usize>) -> &'a [T] {
        debug_assert!(range.start <= self.0.len());
        debug_assert!(range.end <= self.0.len());

        if range.start >= range.end {
            return &[];
        }

        unsafe { core::slice::from_raw_parts(self.get(range.start), range.end - range.start) }
    }

    pub unsafe fn slice_mut(self, range: Range<usize>) -> &'a mut [T] {
        debug_assert!(range.start <= self.0.len());
        debug_assert!(range.end <= self.0.len());

        if range.start >= range.end {
            return &mut [];
        }
        unsafe {
            core::slice::from_raw_parts_mut(self.get_mut(range.start), range.end - range.start)
        }
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
