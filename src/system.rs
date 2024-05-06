use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// This trait is used to define systems of fields.
pub trait SystemLabel: Sized {
    /// Name of the system (used for debugging and when serializing a system).
    const NAME: &'static str;
    /// Number of component fields in this system.
    const FIELDS: usize;

    /// Constructs an individual field label from an index.
    fn from_index(idx: usize) -> Self;

    /// Retrieves the index of an individual field.
    fn field_index(&self) -> usize;

    /// Retrieves the name of an individual field.
    fn field_name(&self) -> String;

    /// Iterates all fields in the system.
    fn fields() -> impl Iterator<Item = Self> {
        (0..Self::FIELDS)
            .into_iter()
            .map(|idx| Self::from_index(idx))
    }
}

// /// An iterator that allows enumerating system labels.
// pub struct SystemLabelIter<Label: SystemLabel>(Option<usize>, PhantomData<Label>);

// impl<Label: SystemLabel> SystemLabelIter<Label> {
//     pub fn new() -> Self {
//         if Label::FIELDS == 0 {
//             Self(None, PhantomData)
//         } else {
//             Self(Some(0), PhantomData)
//         }
//     }
// }

// impl<Label: SystemLabel> Iterator for SystemLabelIter<Label> {
//     type Item = Label;

//     fn next(&mut self) -> Option<Self::Item> {
//         let idx = self.0?;

//         // Retrieved index should always be valid
//         debug_assert!(idx < Label::FIELDS);

//         // Construct label
//         let result = Label::from_index(idx);

//         // Increment index, setting cursor to null if necessary.
//         if idx + 1 >= Label::FIELDS {
//             self.0 = None;
//         }

//         // Return result
//         Some(result)
//     }
// }

/// Stores a system in memory as an structure of data vectors. This SoA approach
/// allows us to compute derivatives faster and better utilize caching.
#[derive(Clone, Debug)]
pub struct System<Label: SystemLabel> {
    /// Number of dofs per field.
    ndofs: usize,
    /// Backing data
    data: Vec<f64>,
    /// Marker
    _marker: PhantomData<Label>,
}

impl<Label: SystemLabel> System<Label> {
    /// Constructs a new system with the given degrees of freedom.
    pub fn new(ndofs: usize) -> Self {
        let data = vec![0.0; ndofs * Label::FIELDS];

        Self {
            ndofs,
            data,
            _marker: PhantomData,
        }
    }

    /// Constructs a system of a certain number of dofs from a untyped data vector.
    /// `data.len()` must equal `ndofs * Label::FIELDS`.
    pub fn from_parts(ndofs: usize, data: Vec<f64>) -> Self {
        assert!(ndofs * Label::FIELDS == data.len());

        Self {
            ndofs,
            data,
            _marker: PhantomData,
        }
    }

    /// Retrieves an untyped representation of a system (for serialization).
    pub fn to_parts(self) -> (usize, Vec<f64>) {
        (self.ndofs, self.data)
    }

    /// Number of degrees of freedom per field.
    pub fn ndofs(&self) -> usize {
        self.ndofs
    }

    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, idx: usize) -> &[f64] {
        let start = idx * self.ndofs;
        let end = idx * self.ndofs + self.ndofs;

        &self.data[start..end]
    }

    /// Retrieves a mutable reference to a field located at the given index.
    pub fn field_mut(&mut self, idx: usize) -> &mut [f64] {
        let start = idx * self.ndofs;
        let end = idx * self.ndofs + self.ndofs;

        &mut self.data[start..end]
    }
}

impl<Label: SystemLabel> Index<Label> for System<Label> {
    type Output = [f64];

    fn index(&self, index: Label) -> &Self::Output {
        self.field(index.field_index())
    }
}

impl<Label: SystemLabel> IndexMut<Label> for System<Label> {
    fn index_mut(&mut self, index: Label) -> &mut Self::Output {
        self.field_mut(index.field_index())
    }
}
