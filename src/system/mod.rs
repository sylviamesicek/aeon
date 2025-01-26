//! A module for handling coupled multi-variate systems defined on meshes.
//!
//! The primary abstraction of this module is the `System` trait, which defines
//! a system of scalar fields to be stored in a SoA format.

mod conditions;
mod prim;
mod vec;

pub use conditions::*;
pub use prim::*;
pub use vec::*;

/// Represents a system of fields that can be stored in a SoA format. This abstraction allows users to pass around
/// `SystemVec`s and `SystemSlice`s without having to worry about computing offsets to individual fields or allocations.
pub trait System {
    /// Name of the system, used for serialization.
    const NAME: &'static str = "Unknown";

    /// Label used to index fields.
    type Label: Clone + Copy + Send + Sync;

    /// Enumerates all fields in the system
    fn enumerate(&self) -> impl Iterator<Item = Self::Label> + Send + Sync;
    /// Returns the number of fields in the system
    fn count(&self) -> usize {
        self.enumerate().count()
    }

    /// Converts a system label to an index.
    fn label_index(&self, label: Self::Label) -> usize;
    /// Builds a system label from an index.
    fn label_from_index(&self, index: usize) -> Self::Label;
    /// Returns the name of an individual field for serialization.
    fn label_name(&self, _label: Self::Label) -> String {
        "Unknown".to_string()
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, aeon_macros::SystemLabel)]
    enum Test {
        First,
        Second,
        Third,
    }

    /// Test of basic system functionality.
    #[test]
    fn basic() {
        let mut fields = SystemVec::with_length(3, TestSystem);

        {
            let shared = fields.as_mut_slice().into_shared();
            let mut slice = unsafe { shared.slice_mut(1..2) };

            slice.field_mut(Test::First).fill(1.0);
            slice.field_mut(Test::Second).fill(2.0);
            slice.field_mut(Test::Third).fill(3.0);
        }

        let buffer = fields.into_contiguous();

        assert_eq!(&buffer, &[0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0]);

        let empty = SystemVec::new(Empty);
        assert!(empty.is_empty());
    }

    /// A simple test of creating composite (pair) systems, splitting them, and taking various references.
    #[test]
    fn pair() {
        let system = (Dynamic(3), Dynamic(2));

        let mut data = SystemVec::with_length(10, system);
        assert_eq!(data.len(), 10);
        data.field_mut(Pair::First(0)).fill(0.0);
        data.field_mut(Pair::First(1)).fill(1.0);
        data.field_mut(Pair::First(2)).fill(2.0);
        data.field_mut(Pair::Second(0)).fill(3.0);
        data.field_mut(Pair::Second(1)).fill(4.0);

        assert!(data.field(Pair::First(0)).iter().all(|v| *v == 0.0));
        assert!(data.field(Pair::First(1)).iter().all(|v| *v == 1.0));
        assert!(data.field(Pair::First(2)).iter().all(|v| *v == 2.0));

        assert!(data.field(Pair::Second(0)).iter().all(|v| *v == 3.0));
        assert!(data.field(Pair::Second(1)).iter().all(|v| *v == 4.0));

        let mut data = data.into_contiguous();

        let slice = SystemSliceMut::from_contiguous(&mut data, &system);

        assert!(slice.field(Pair::First(0)).iter().all(|v| *v == 0.0));
        assert!(slice.field(Pair::First(1)).iter().all(|v| *v == 1.0));
        assert!(slice.field(Pair::First(2)).iter().all(|v| *v == 2.0));

        assert!(slice.field(Pair::Second(0)).iter().all(|v| *v == 3.0));
        assert!(slice.field(Pair::Second(1)).iter().all(|v| *v == 4.0));

        let (slice1, slice2) = slice.split_pair();

        assert!(slice1.field(0).iter().all(|v| *v == 0.0));
        assert!(slice1.field(1).iter().all(|v| *v == 1.0));
        assert!(slice1.field(2).iter().all(|v| *v == 2.0));

        assert!(slice2.field(0).iter().all(|v| *v == 3.0));
        assert!(slice2.field(1).iter().all(|v| *v == 4.0));

        let mut data = (0..15).map(|i| i as f64).collect::<Vec<_>>();
        let mut slice = SystemSliceMut::from_contiguous(&mut data, &(Static::<2>, Scalar));

        assert_eq!(slice.field(Pair::First(0)), &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(slice.field(Pair::First(1)), &[5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(
            slice.field(Pair::Second(())),
            &[10.0, 11.0, 12.0, 13.0, 14.0]
        );

        let (slice1, slice2) = slice.slice_mut(2..4).split_pair();

        assert_eq!(slice1.field(0), &[2.0, 3.0]);
        assert_eq!(slice1.field(1), &[7.0, 8.0]);
        assert_eq!(slice2.field(()), &[12.0, 13.0]);
    }
}
