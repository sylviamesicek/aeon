/// A stencil consist of a series of weights defined on equispaced points
/// used to approximate some numerical operator.
pub trait Stencil {
    /// Number of support points.
    const SUPPORT: usize;

    /// Weight at `i`th point where `0 <= i < SUPPORT`.
    fn weight(&self, i: usize) -> f64;
}

/// A convience class for iterating over the weights in a stencil.
pub struct StencilIterator<S: Stencil> {
    stencil: S,
    cursor: usize,
}

impl<S: Stencil> StencilIterator<S> {
    /// Constructs a new stencil iterator.
    pub fn new(stencil: S) -> Self {
        Self { stencil, cursor: 0 }
    }
}

impl<S: Stencil> Iterator for StencilIterator<S> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor >= S::SUPPORT {
            return None;
        }

        let weight = self.stencil.weight(self.cursor);
        self.cursor += 1;
        Some(weight)
    }
}

pub trait VertexStencil: Stencil {
    const POSITIVE: usize;
    const NEGATIVE: usize;
}

pub trait CellStencil: Stencil {
    const POSITIVE: usize;
    const NEGATIVE: usize;
}

pub trait DirectedStencil: Stencil {}

#[derive(Debug, Clone)]
pub struct CenteredSecondOrder(pub [f64; 3]);

impl Stencil for CenteredSecondOrder {
    const SUPPORT: usize = 3;

    fn weight(self: &Self, i: usize) -> f64 {
        self.0[i]
    }
}

impl VertexStencil for CenteredSecondOrder {
    const NEGATIVE: usize = 1;
    const POSITIVE: usize = 1;
}

#[derive(Debug, Clone)]
pub struct CenteredFourthOrder(pub [f64; 5]);

impl Stencil for CenteredFourthOrder {
    const SUPPORT: usize = 5;

    fn weight(self: &Self, i: usize) -> f64 {
        self.0[i]
    }
}

impl VertexStencil for CenteredFourthOrder {
    const NEGATIVE: usize = 2;
    const POSITIVE: usize = 2;
}

#[derive(Debug, Clone, Copy)]
pub struct ProlongSecondOrder(pub [f64; 2]);

impl Stencil for ProlongSecondOrder {
    const SUPPORT: usize = 2;

    fn weight(&self, i: usize) -> f64 {
        self.0[i]
    }
}

impl CellStencil for ProlongSecondOrder {
    const NEGATIVE: usize = 1;
    const POSITIVE: usize = 1;
}

#[derive(Debug, Clone, Copy)]
pub struct ProlongFourthOrder(pub [f64; 4]);

impl Stencil for ProlongFourthOrder {
    const SUPPORT: usize = 4;

    fn weight(&self, i: usize) -> f64 {
        self.0[i]
    }
}

impl CellStencil for ProlongFourthOrder {
    const NEGATIVE: usize = 2;
    const POSITIVE: usize = 2;
}

#[derive(Debug, Clone, Copy)]
pub struct BoundaryStencil<const SUPPORT: usize>(pub [f64; SUPPORT]);

impl<const SUPPORT: usize> Stencil for BoundaryStencil<SUPPORT> {
    const SUPPORT: usize = SUPPORT;

    fn weight(&self, i: usize) -> f64 {
        self.0[i]
    }
}

impl<const SUPPORT: usize> DirectedStencil for BoundaryStencil<SUPPORT> {}
