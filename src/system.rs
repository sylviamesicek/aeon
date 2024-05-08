use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// This trait is used to define systems of fields.
pub trait SystemLabel: Sized {
    /// Name of the system (used for debugging and when serializing a system).
    const NAME: &'static str;
    /// Number of component fields in this system.
    const FIELD_COUNT: usize;

    /// Constructs an individual field label from an index.
    fn from_index(idx: usize) -> Self;

    /// Retrieves the index of an individual field.
    fn field_index(&self) -> usize;

    /// Retrieves the name of an individual field.
    fn field_name(&self) -> String;

    /// Iterates all fields in the system.
    fn fields() -> impl Iterator<Item = Self> {
        (0..Self::FIELD_COUNT)
            .into_iter()
            .map(|idx| Self::from_index(idx))
    }
}

/// Stores a system in memory as an structure of field vectors. This SoA approach
/// allows us to compute derivatives faster and better utilize caching.
#[derive(Clone, Debug)]
pub struct System<Label: SystemLabel> {
    inner: SystemData<Vec<f64>>,
    /// Marker
    _marker: PhantomData<Label>,
}

impl<Label: SystemLabel> System<Label> {
    /// Constructs a new system with the given degrees of freedom.
    pub fn new(dof_count: usize) -> Self {
        Self {
            inner: SystemData {
                dof_count,
                field_count: Label::FIELD_COUNT,
                data: vec![0.0; dof_count * Label::FIELD_COUNT],
            },
            _marker: PhantomData,
        }
    }

    /// Casts an untyped system into a typed representation.
    pub fn from_data(data: SystemData<Vec<f64>>) -> Self {
        assert!(data.field_count == Label::FIELD_COUNT);

        Self {
            inner: data,
            _marker: PhantomData,
        }
    }

    /// Converts system into untyped representation.
    pub fn into_data(self) -> SystemData<Vec<f64>> {
        self.inner
    }

    /// Number of degrees of freedom per field.
    pub fn dof_count(&self) -> usize {
        self.inner.dof_count
    }

    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, idx: usize) -> &[f64] {
        self.inner.field(idx)
    }

    /// Retrieves a mutable reference to a field located at the given index.
    pub fn field_mut(&mut self, idx: usize) -> &mut [f64] {
        self.inner.field_mut(idx)
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

/// Untyped representation of a system in memory (a structure of field vectorsr).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SystemData<T> {
    pub dof_count: usize,
    pub field_count: usize,
    pub data: T,
}

impl<T> SystemData<T>
where
    T: AsRef<[f64]> + AsMut<[f64]>,
{
    /// Retrieves an immutable reference to a field located at the given index.
    pub fn field(&self, idx: usize) -> &[f64] {
        let start = idx * self.dof_count;
        let end = idx * self.dof_count + self.dof_count;

        &self.data.as_ref()[start..end]
    }

    /// Retrieves a mutable reference to a field located at the given index.
    pub fn field_mut(&mut self, idx: usize) -> &mut [f64] {
        let start = idx * self.dof_count;
        let end = idx * self.dof_count + self.dof_count;

        &mut self.data.as_mut()[start..end]
    }
}
