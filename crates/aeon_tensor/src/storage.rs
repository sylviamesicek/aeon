pub trait TensorStorageRef {
    /// Reference to underlying buffer for tensor storage.
    /// This is only gaurenteed to be >= the most recent call to length.
    fn buffer(&self) -> &[f64];
}

pub trait TensorStorageMut: TensorStorageRef {
    fn buffer_mut(&mut self) -> &mut [f64];
}

pub trait TensorStorageOwned: TensorStorageMut {
    fn resize(&mut self, length: usize);
}

// *********************************
// Implementations

impl TensorStorageRef for f64 {
    fn buffer(&self) -> &[f64] {
        std::slice::from_ref(self)
    }
}

impl TensorStorageMut for f64 {
    fn buffer_mut(&mut self) -> &mut [f64] {
        std::slice::from_mut(self)
    }
}

impl TensorStorageOwned for f64 {
    fn resize(&mut self, length: usize) {
        assert!(
            length <= 1,
            "f64 storage does not have capacity for {} elements",
            length
        );
    }
}

impl<const L: usize> TensorStorageRef for [f64; L] {
    fn buffer(&self) -> &[f64] {
        self.as_slice()
    }
}

impl<const L: usize> TensorStorageMut for [f64; L] {
    fn buffer_mut(&mut self) -> &mut [f64] {
        self.as_mut_slice()
    }
}

impl<const L: usize> TensorStorageOwned for [f64; L] {
    fn resize(&mut self, length: usize) {
        assert!(
            length <= L,
            "static storage of length {} does not have capacity for {} elements",
            L,
            length
        );
    }
}

impl<'a> TensorStorageRef for &'a [f64] {
    fn buffer(&self) -> &[f64] {
        self
    }
}

impl<'a> TensorStorageRef for &'a mut [f64] {
    fn buffer(&self) -> &[f64] {
        self
    }
}

impl<'a> TensorStorageMut for &'a mut [f64] {
    fn buffer_mut(&mut self) -> &mut [f64] {
        self
    }
}

impl TensorStorageRef for Vec<f64> {
    fn buffer(&self) -> &[f64] {
        self
    }
}

impl<'a> TensorStorageMut for Vec<f64> {
    fn buffer_mut(&mut self) -> &mut [f64] {
        self
    }
}

impl<'a> TensorStorageOwned for Vec<f64> {
    fn resize(&mut self, length: usize) {
        assert!(
            length <= self.len(),
            "slice storage of length {} does not have capacity for {} elements",
            self.len(),
            length
        );
    }
}
