use crate::{fd::NodeSpace, geometry::Rectangle};

use crate::fd::{Boundary, Condition};

use super::{kernel::Convolution, node_from_vertex};

/// An interface for computing values, gradients, and hessians of fields.
pub trait Engine<const N: usize> {
    fn position(&self) -> [f64; N];
    fn vertex(&self) -> [usize; N];
    fn value(&self, field: &[f64]) -> f64;
    fn evaluate<K: Convolution<N>, C: Condition<N>>(
        &self,
        convolution: K,
        condition: C,
        field: &[f64],
    ) -> f64;
}

/// A finite difference engine of a given order, but potentially bordering a free boundary.
pub struct FdEngine<const N: usize, B> {
    pub space: NodeSpace<N, B>,
    pub vertex: [usize; N],
    pub bounds: Rectangle<N>,
}

impl<const N: usize, B> FdEngine<N, B> {
    pub fn new(space: NodeSpace<N, B>, vertex: [usize; N], bounds: Rectangle<N>) -> Self {
        FdEngine {
            space,
            vertex,
            bounds,
        }
    }
}

impl<const N: usize, B: Boundary<N>> Engine<N> for FdEngine<N, B> {
    fn position(&self) -> [f64; N] {
        self.space
            .position(node_from_vertex(self.vertex), self.bounds.clone())
    }

    fn vertex(&self) -> [usize; N] {
        self.vertex
    }

    fn value(&self, field: &[f64]) -> f64 {
        let linear = self.space.index_from_vertex(self.vertex);
        field[linear]
    }

    fn evaluate<K: Convolution<N>, C: Condition<N>>(
        &self,
        convolution: K,
        condition: C,
        field: &[f64],
    ) -> f64 {
        let space = self.space.attach_condition(condition);
        let spacing = space.spacing(self.bounds.clone());
        let scale = convolution.scale(spacing);
        space.evaluate(self.vertex, convolution, field) * scale
    }
}

/// A finite difference engine that only every relies on interior support (and can thus use better optimized stencils).
pub struct FdIntEngine<const N: usize> {
    pub space: NodeSpace<N, ()>,
    pub vertex: [usize; N],
    pub bounds: Rectangle<N>,
}

impl<const N: usize> FdIntEngine<N> {
    pub fn new(space: NodeSpace<N, ()>, vertex: [usize; N], bounds: Rectangle<N>) -> Self {
        FdIntEngine {
            space,
            vertex,
            bounds,
        }
    }
}

impl<const N: usize> Engine<N> for FdIntEngine<N> {
    fn position(&self) -> [f64; N] {
        self.space
            .position(node_from_vertex(self.vertex), self.bounds.clone())
    }

    fn vertex(&self) -> [usize; N] {
        self.vertex
    }

    fn value(&self, field: &[f64]) -> f64 {
        let linear = self.space.index_from_vertex(self.vertex);
        field[linear]
    }

    fn evaluate<K: Convolution<N>, C: Condition<N>>(
        &self,
        convolution: K,
        _condition: C,
        field: &[f64],
    ) -> f64 {
        let spacing = self.space.spacing(self.bounds.clone());
        let scale = convolution.scale(spacing);
        self.space
            .evaluate_interior(self.vertex, convolution, field)
            * scale
    }
}
