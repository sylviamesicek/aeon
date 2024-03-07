mod index_space;

pub use index_space::{CartesianIterator, IndexSpace};

pub struct IndexRectangle<const N: usize> {
    pub size: [f64; N],
    pub origin: [f64; N],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Rectangle<const N: usize> {
    pub size: [f64; N],
    pub origin: [f64; N],
}
