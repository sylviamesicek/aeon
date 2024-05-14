use crate::array::{Array, ArrayLike};

use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::{Bound, RangeBounds};
use std::slice::{self, SliceIndex};

/// This trait is used to define systems of fields.
pub trait SystemLabel: Sized {
    /// Name of the system (used for debugging and when serializing a system).
    const NAME: &'static str;

    /// Array type with same length as number of fields
    type FieldLike<T>: ArrayLike<Elem = T>;

    /// Returns an array of all possible system labels.
    fn fields() -> Array<Self::FieldLike<Self>>;

    /// Retrieves the index of an individual field.
    fn field_index(&self) -> usize;

    /// Retrieves the name of an individual field.
    fn field_name(&self) -> String;
}

pub type FieldArray<Label, T> = Array<<Label as SystemLabel>::FieldLike<T>>;

/// Number of fields in a given system label.
pub const fn field_count<Label: SystemLabel>() -> usize {
    Label::FieldLike::<()>::LEN
}

/// A default system representing one scalar field.
pub struct Scalar;

impl SystemLabel for Scalar {
    const NAME: &'static str = "Default";

    type FieldLike<T> = [T; 1];

    fn fields() -> Array<Self::FieldLike<Self>> {
        [Scalar].into()
    }

    fn field_index(&self) -> usize {
        0
    }

    fn field_name(&self) -> String {
        "scalar".to_string()
    }
}

/// Stores a system in memory as an structure of field vectors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemVec<Label: SystemLabel> {
    node_count: usize,
    fields: FieldArray<Label, Vec<f64>>,
}

impl<Label: SystemLabel> SystemVec<Label> {
    /// Constructs a new system with the given degrees of freedom.
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            fields: Label::FieldLike::<Vec<f64>>::from_fn(|_| vec![0.0; node_count]).into(),
        }
    }

    /// Copies the data for a system from a contigious slice of values.
    pub fn from_contigious(data: &[f64]) -> Self {
        let slice = SystemSlice::<'_, Label>::from_contiguous(data);
        let fields = Label::FieldLike::<_>::from_fn(|i| slice.fields[i].to_vec());

        Self {
            node_count: slice.node_count(),
            fields: fields.into(),
        }
    }

    /// Transforms a system into an owned array of values.
    pub fn into_contigious(&self) -> Vec<f64> {
        // Damn, iterator chaining is powerful
        self.fields.clone().into_iter().flatten().collect()
    }

    /// Number of degrees of freedom per field.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, label: Label) -> &[f64] {
        &self.fields[label.field_index()]
    }

    /// Retrieves a mutable reference to a field located at the given index.
    pub fn field_mut(&mut self, label: Label) -> &mut [f64] {
        &mut self.fields[label.field_index()]
    }

    pub fn as_slice<'a>(&'a self) -> SystemSlice<'a, Label> {
        SystemSlice {
            node_count: self.node_count,
            fields: Label::FieldLike::<&'a [f64]>::from_fn(|i| self.fields[i].as_slice()).into(),
        }
    }

    pub fn as_mut_slice<'a>(&'a mut self) -> SystemSliceMut<'a, Label> {
        // First map into an array of pointers.
        let ptrs = Label::FieldLike::<_>::from_fn(|i| self.fields[i].as_mut_ptr());

        // Next we transforms
        let fields = Label::FieldLike::<&'a mut [f64]>::from_fn(|i| unsafe {
            slice::from_raw_parts_mut::<'a, _>(ptrs[i], self.node_count)
        });

        SystemSliceMut {
            node_count: self.node_count,
            fields: fields.into(),
        }
    }
}

// impl<Label: SystemLabel> Index<Label> for SystemVec<Label> {
//     type Output = [f64];

//     fn index(&self, index: Label) -> &Self::Output {
//         self.field(index.field_index())
//     }
// }

// impl<Label: SystemLabel> IndexMut<Label> for SystemVec<Label> {
//     fn index_mut(&mut self, index: Label) -> &mut Self::Output {
//         self.field_mut(index.field_index())
//     }
// }

#[derive(Clone, Debug)]
pub struct SystemSlice<'a, Label: SystemLabel> {
    node_count: usize,
    fields: FieldArray<Label, &'a [f64]>,
}

impl<'a, Label: SystemLabel> SystemSlice<'a, Label> {
    /// Reinterprets the given data vector as a system of individual slices.
    /// data.len() must be equal to node_count * field_count.
    pub fn from_contiguous(data: &'a [f64]) -> Self {
        assert!(data.len() % field_count::<Label>() == 0);

        let node_count = data.len() / field_count::<Label>();

        // I am sure there is a safe way to do this, but this works and is quick.
        let ptr = data.as_ptr();
        let fields = Label::FieldLike::from_fn(|i| unsafe {
            slice::from_raw_parts::<'a, _>(ptr.offset((node_count * i) as isize), node_count)
        });

        Self {
            node_count,
            fields: fields.into(),
        }
    }

    /// Number of degrees of freedom per field.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, label: Label) -> &[f64] {
        self.fields[label.field_index()]
    }

    pub fn slice<R>(&self, range: R) -> Self
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let start_inc = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i + 1,
            Bound::Unbounded => 0,
        };

        let end_exc = match range.end_bound() {
            Bound::Included(&i) => i + 1,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.node_count,
        };

        Self {
            node_count: end_exc - start_inc,
            fields: Label::FieldLike::<&'a [f64]>::from_fn(|i| &self.fields[i][range.clone()])
                .into(),
        }
    }
}

// impl<Label: SystemLabel> Index<Label> for SystemSlice<'_, Label> {
//     type Output = [f64];

//     fn index(&self, index: Label) -> &Self::Output {
//         self.field(index.field_index())
//     }
// }

#[derive(Debug)]
pub struct SystemSliceMut<'a, Label: SystemLabel> {
    node_count: usize,
    fields: FieldArray<Label, &'a mut [f64]>,
}

impl<'a, Label: SystemLabel> SystemSliceMut<'a, Label> {
    /// Reinterprets the given data vector as a system of individual slices.
    /// data.len() must be equal to node_count * field_count.
    pub fn from_contiguous(data: &'a mut [f64]) -> Self {
        assert!(data.len() % field_count::<Label>() == 0);
        let node_count = data.len() / field_count::<Label>();

        // I am sure there is a safe way to do this, but this works and is quick.
        let ptr = data.as_mut_ptr();
        let fields = Label::FieldLike::from_fn(|i| unsafe {
            slice::from_raw_parts_mut::<'a, _>(ptr.offset((node_count * i) as isize), node_count)
        });

        Self {
            node_count,
            fields: fields.into(),
        }
    }

    /// Number of degrees of freedom per field.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, label: Label) -> &[f64] {
        self.fields[label.field_index()]
    }

    pub fn field_mut(&mut self, label: Label) -> &mut [f64] {
        self.fields[label.field_index()]
    }

    pub fn slice<R>(&self, range: R) -> SystemSlice<'a, Label>
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let start_inc = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i + 1,
            Bound::Unbounded => 0,
        };

        let end_exc = match range.end_bound() {
            Bound::Included(&i) => i + 1,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.node_count,
        };

        let node_count = end_exc - start_inc;

        let fields = Label::FieldLike::<&[f64]>::from_fn(|i| unsafe {
            slice::from_raw_parts(self.fields[i].as_ptr(), node_count)
        });

        SystemSlice {
            node_count,
            fields: fields.into(),
        }
    }

    pub fn slice_mut<R>(&mut self, range: R) -> Self
    where
        R: RangeBounds<usize> + SliceIndex<[f64], Output = [f64]> + Clone,
    {
        let start_inc = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i + 1,
            Bound::Unbounded => 0,
        };

        let end_exc = match range.end_bound() {
            Bound::Included(&i) => i + 1,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.node_count,
        };

        let node_count = end_exc - start_inc;

        let fields = Label::FieldLike::<&'a mut [f64]>::from_fn(|i| unsafe {
            slice::from_raw_parts_mut(self.fields[i].as_mut_ptr(), node_count)
        });

        Self {
            node_count,
            fields: fields.into(),
        }
    }
}

// impl<Label: SystemLabel> Index<Label> for SystemSliceMut<'_, Label> {
//     type Output = [f64];

//     fn index(&self, index: Label) -> &Self::Output {
//         self.field(index.field_index())
//     }
// }

// impl<Label: SystemLabel> IndexMut<Label> for SystemSliceMut<'_, Label> {
//     fn index_mut(&mut self, index: Label) -> &mut Self::Output {
//         self.field_mut(index.field_index())
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    pub enum MySystem {
        First,
        Second,
        Third,
    }

    impl SystemLabel for MySystem {
        const NAME: &'static str = "MySystem";

        type FieldLike<T> = [T; 3];

        fn fields() -> Array<Self::FieldLike<Self>> {
            [MySystem::First, MySystem::Second, MySystem::Third].into()
        }

        fn field_index(&self) -> usize {
            match self {
                Self::First => 0,
                Self::Second => 1,
                Self::Third => 2,
            }
        }

        fn field_name(&self) -> String {
            match self {
                Self::First => "First",
                Self::Second => "Second",
                Self::Third => "Third",
            }
            .to_string()
        }
    }

    #[test]
    fn systems() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let owned = SystemVec::<MySystem>::from_contigious(data.as_ref());

        assert_eq!(owned.node_count(), 3);
        assert_eq!(owned.field(MySystem::First), &[0.0, 1.0, 2.0]);
        assert_eq!(owned.field(MySystem::Second), &[3.0, 4.0, 5.0]);
        assert_eq!(owned.field(MySystem::Third), &[6.0, 7.0, 8.0]);

        let slice = owned.as_slice();
        let slice_cast = SystemSlice::<MySystem>::from_contiguous(&data);
        assert_eq!(
            slice.field(MySystem::First),
            slice_cast.field(MySystem::First)
        );
        assert_eq!(
            slice.field(MySystem::Second),
            slice_cast.field(MySystem::Second)
        );
        assert_eq!(
            slice.field(MySystem::Third),
            slice_cast.field(MySystem::Third)
        );
    }
}

// fn test() {
//     let mut v = [0; 10];
//     let rv = &mut v[..];
//     let bg = &mut rv[..];

//     rv[0] += 1;
//     bg[0] += 1;
// }

// /// Untyped representation of a system in memory (a structure of field vectors).
// #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
// pub struct SystemData {
//     pub node_count: usize,
//     pub field_count: usize,
//     pub data: Vec<f64>,
// }

// impl SystemData {
//     /// Retrieves an immutable reference to a field located at the given index.
//     pub fn field(&self, idx: usize) -> &[f64] {
//         let start = idx * self.node_count;
//         let end = idx * self.node_count + self.node_count;

//         &self.data[start..end]
//     }

//     /// Retrieves a mutable reference to a field located at the given index.
//     pub fn field_mut(&mut self, idx: usize) -> &mut [f64] {
//         let start = idx * self.node_count;
//         let end = idx * self.node_count + self.node_count;

//         &mut self.data[start..end]
//     }
// }
