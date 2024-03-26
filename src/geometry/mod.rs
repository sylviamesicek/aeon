mod index_space;

pub use index_space::{CartesianIterator, IndexSpace};

#[derive(Debug, Clone, PartialEq)]
pub struct Rectangle<const N: usize> {
    pub size: [f64; N],
    pub origin: [f64; N],
}

impl<const N: usize> Rectangle<N> {
    /// Unit rectangle.
    pub const UNIT: Self = Rectangle {
        size: [1.0; N],
        origin: [0.0; N],
    };
}
