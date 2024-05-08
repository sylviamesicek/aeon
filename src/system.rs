use crate::array::{Array, ArrayLike};

use std::fmt::Debug;
use std::ops::{Bound, Index, IndexMut, RangeBounds};
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

    // /// Constructs an individual field label from an index.
    // fn from_index(idx: usize) -> Self;

    // /// Iterates all fields in the system.
    // fn fields() -> impl Iterator<Item = Self> {
    //     (0..Self::FIELD_COUNT)
    //         .into_iter()
    //         .map(|idx| Self::from_index(idx))
    // }
}

pub type FieldArray<Label, T> = Array<<Label as SystemLabel>::FieldLike<T>>;

pub const fn field_count<Label: SystemLabel>() -> usize {
    Label::FieldLike::<()>::LEN
}

/// Stores a system in memory as an structure of field vectors. This SoA approach
/// allows us to compute derivatives faster and better utilize caching.
#[derive(Clone, Debug)]
pub struct SystemOwned<Label: SystemLabel> {
    node_count: usize,
    fields: FieldArray<Label, Vec<f64>>,
}

impl<Label: SystemLabel> SystemOwned<Label> {
    /// Constructs a new system with the given degrees of freedom.
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            fields: Label::FieldLike::<Vec<f64>>::from_fn(|_| vec![0.0; node_count]).into(),
        }
    }

    pub fn from_contigious(data: &[f64]) -> Self {
        let slice = SystemSlice::<'_, Label>::from_contiguous(data);
        let fields = Label::FieldLike::<_>::from_fn(|i| slice.field(i).to_vec());

        Self {
            node_count: slice.node_count(),
            fields: fields.into(),
        }
    }

    pub fn into_contigious(&self) -> Vec<f64> {
        // Damn, iterator chaining is powerful
        self.fields.clone().into_iter().flatten().collect()
    }

    // /// Casts an untyped system into a typed representation.
    // pub fn from_data(data: SystemData) -> Self {
    //     assert!(data.field_count == field_count::<Label>());

    //     Self {
    //         inner: data,
    //         _marker: PhantomData,
    //     }
    // }

    // /// Converts system into untyped representation.
    // pub fn into_data(self) -> SystemData {
    //     self.inner
    // }

    /// Number of degrees of freedom per field.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, idx: usize) -> &[f64] {
        &self.fields[idx]
    }

    /// Retrieves a mutable reference to a field located at the given index.
    pub fn field_mut(&mut self, idx: usize) -> &mut [f64] {
        &mut self.fields[idx]
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

impl<Label: SystemLabel> Index<Label> for SystemOwned<Label> {
    type Output = [f64];

    fn index(&self, index: Label) -> &Self::Output {
        self.field(index.field_index())
    }
}

impl<Label: SystemLabel> IndexMut<Label> for SystemOwned<Label> {
    fn index_mut(&mut self, index: Label) -> &mut Self::Output {
        self.field_mut(index.field_index())
    }
}

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
    pub fn field(&self, idx: usize) -> &[f64] {
        self.fields[idx]
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

impl<Label: SystemLabel> Index<Label> for SystemSlice<'_, Label> {
    type Output = [f64];

    fn index(&self, index: Label) -> &Self::Output {
        self.field(index.field_index())
    }
}

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
    pub fn field(&self, idx: usize) -> &[f64] {
        self.fields[idx]
    }

    pub fn field_mut(&mut self, idx: usize) -> &mut [f64] {
        self.fields[idx]
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

impl<Label: SystemLabel> Index<Label> for SystemSliceMut<'_, Label> {
    type Output = [f64];

    fn index(&self, index: Label) -> &Self::Output {
        self.field(index.field_index())
    }
}

impl<Label: SystemLabel> IndexMut<Label> for SystemSliceMut<'_, Label> {
    fn index_mut(&mut self, index: Label) -> &mut Self::Output {
        self.field_mut(index.field_index())
    }
}

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
        let _owned = SystemOwned::<MySystem>::new(100);
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
