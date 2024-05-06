use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// This trait is used to define systems of fields.
pub trait SystemLabel {
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
}

/// Stores a system in memory as an structure of data vectors. This SoA approach
/// allows us to compute derivatives faster and better utilize caching.
#[derive(Clone, Debug)]
pub struct System<Label: SystemLabel> {
    /// Number of dofs per field.
    len: usize,
    fields: Vec<Vec<f64>>,

    _marker: PhantomData<Label>,
}

impl<Label: SystemLabel> System<Label> {
    /// Constructs a new system with the given degrees of freedom.
    pub fn new(len: usize) -> Self {
        Self {
            fields: vec![vec![0.0; len]; Label::FIELDS],
            len,
            _marker: PhantomData,
        }
    }

    /// Number of degrees of freedom per field.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, idx: usize) -> &[f64] {
        &self.fields[idx]
    }

    /// Retrieves a mutable reference to a field located at the given index.
    pub fn field_mut(&mut self, idx: usize) -> &mut [f64] {
        &mut self.fields[idx]
    }

    pub fn into_untyped_fields(self) -> Vec<Vec<f64>> {
        self.fields
    }

    pub fn from_untyped_fields(fields: Vec<Vec<f64>>) -> Self {
        assert!(fields.len() == Label::FIELDS);

        let len = fields.first().map_or(0, |f| f.len());

        Self {
            fields: vec![vec![0.0; len]; Label::FIELDS],
            len,
            _marker: PhantomData,
        }
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
