use aeon_basis::{
    node_from_vertex, Boundary, Convolution, Dissipation, Gradient, Hessian, Kernels, NodeSpace,
};
use aeon_geometry::Rectangle;

use crate::fd::{Conditions, SystemBC};
use crate::system::{SystemFields, SystemLabel};

/// An interface for computing values, gradients, and hessians of fields.
pub trait Engine<const N: usize, S: SystemLabel> {
    fn position(&self) -> [f64; N];
    fn vertex(&self) -> [usize; N];
    fn value(&self, system: S) -> f64;
    fn derivative(&self, system: S, axis: usize) -> f64;
    fn second_derivative(&self, system: S, i: usize, j: usize) -> f64;
    fn dissipation(&self, system: S) -> f64;
}

/// A finite difference engine of a given order, but potentially bordering a free boundary.
pub struct FdEngine<'a, const N: usize, K: Kernels, B: Boundary<N>, C: Conditions<N>> {
    pub space: NodeSpace<N>,
    pub vertex: [usize; N],
    pub bounds: Rectangle<N>,
    pub fields: SystemFields<'a, C::System>,
    pub order: K,
    pub boundary: B,
    pub conditions: C,
}

impl<'a, const N: usize, K: Kernels, B: Boundary<N>, C: Conditions<N>> FdEngine<'a, N, K, B, C> {
    fn evaluate(&self, system: C::System, convolution: impl Convolution<N>) -> f64 {
        let result = self.space.evaluate(
            SystemBC::new(
                system.clone(),
                self.boundary.clone(),
                self.conditions.clone(),
            ),
            convolution,
            self.bounds.clone(),
            self.vertex,
            self.fields.field(system.clone()),
        );
        result
    }
}

impl<'a, const N: usize, K: Kernels, B: Boundary<N>, C: Conditions<N>> Engine<N, C::System>
    for FdEngine<'a, N, K, B, C>
{
    fn position(&self) -> [f64; N] {
        self.space
            .position(node_from_vertex(self.vertex), self.bounds.clone())
    }

    fn vertex(&self) -> [usize; N] {
        self.vertex
    }

    fn value(&self, system: C::System) -> f64 {
        let index = self.space.index_from_vertex(self.vertex);
        self.fields.field(system)[index]
    }

    fn derivative(&self, system: C::System, axis: usize) -> f64 {
        self.evaluate(system, Gradient::<K>::new(axis))
    }

    fn second_derivative(&self, system: C::System, i: usize, j: usize) -> f64 {
        self.evaluate(system, Hessian::<K>::new(i, j))
    }

    fn dissipation(&self, system: C::System) -> f64 {
        let mut result = 0.0;

        for axis in 0..N {
            result += self.evaluate(system.clone(), Dissipation::<K>::new(axis))
        }

        result
    }
}

/// A finite difference engine that only every relies on interior support (and can thus use better optimized stencils).
pub struct FdIntEngine<'a, const N: usize, K: Kernels, S: SystemLabel> {
    pub space: NodeSpace<N>,
    pub vertex: [usize; N],
    pub bounds: Rectangle<N>,
    pub fields: SystemFields<'a, S>,
    pub order: K,
}

impl<'a, const N: usize, K: Kernels, S: SystemLabel> FdIntEngine<'a, N, K, S> {
    fn evaluate(&self, system: S, convolution: impl Convolution<N>) -> f64 {
        let result = self.space.evaluate_interior(
            convolution,
            self.bounds.clone(),
            self.vertex,
            self.fields.field(system.clone()),
        );
        result
    }
}

impl<'a, const N: usize, K: Kernels, S: SystemLabel> Engine<N, S> for FdIntEngine<'a, N, K, S> {
    fn position(&self) -> [f64; N] {
        self.space
            .position(node_from_vertex(self.vertex), self.bounds.clone())
    }

    fn vertex(&self) -> [usize; N] {
        self.vertex
    }

    fn value(&self, system: S) -> f64 {
        let index = self.space.index_from_vertex(self.vertex);
        self.fields.field(system)[index]
    }

    fn derivative(&self, system: S, axis: usize) -> f64 {
        self.evaluate(system, Gradient::<K>::new(axis))
    }

    fn second_derivative(&self, system: S, i: usize, j: usize) -> f64 {
        self.evaluate(system, Hessian::<K>::new(i, j))
    }

    fn dissipation(&self, system: S) -> f64 {
        let mut result = 0.0;

        for axis in 0..N {
            result += self.evaluate(system.clone(), Dissipation::<K>::new(axis))
        }

        result
    }
}
