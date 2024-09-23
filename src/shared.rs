use std::cell::UnsafeCell;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct SharedSlice<'a, T>(&'a [UnsafeCell<T>]);

impl<'a, T> SharedSlice<'a, T> {
    pub fn new(data: &'a mut [T]) -> Self {
        Self(unsafe { &*(data as *mut [T] as *const [UnsafeCell<T>]) })
    }
}
