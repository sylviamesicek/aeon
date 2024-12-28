//! A module for handling coupled multi-variate systems defined on meshes.
//!
//! The primary abstraction of this module is the `System` trait, which defines
//! a system of scalar fields to be stored in a SoA format.

mod prim;
mod vec;

pub use prim::*;
pub use vec::*;

/// Represents a system of fields that can be stored in a SoA format. This abstraction allows users to pass around
/// `SystemVec`s and `SystemSlice`s without having to worry about computing offsets to individual fields or allocations.
pub trait System {
    /// Name of the system, used for serialization.
    const NAME: &'static str = "Unknown";

    /// Label used to index fields.
    type Label: Clone + Copy;

    /// Enumerates all fields in the system
    fn enumerate(&self) -> impl Iterator<Item = Self::Label>;
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
