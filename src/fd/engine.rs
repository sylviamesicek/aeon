use crate::{
    fd::{node_from_vertex, BasisOperator, NodeSpace},
    geometry::Rectangle,
};
use std::array;

use crate::fd::{Boundary, Condition, Order, BC};

/// An interface for computing values, gradients, and hessians of fields.
pub trait Engine<const N: usize> {
    fn position(&self) -> [f64; N];
    fn vertex(&self) -> [usize; N];
    fn value(&self, field: &[f64]) -> f64;
    fn gradient<C: Condition<N>>(&self, cond: C, field: &[f64]) -> [f64; N];
    fn hessian<C: Condition<N>>(&self, cond: C, field: &[f64]) -> [[f64; N]; N];
}

/// A finite difference engine of a given order, but potentially bordering a free boundary.
pub struct FdEngine<const N: usize, const ORDER: usize, B> {
    pub space: NodeSpace<N, Rectangle<N>>,
    pub vertex: [usize; N],
    pub boundary: B,
}

impl<const N: usize, const ORDER: usize, B: Boundary<N>> Engine<N> for FdEngine<N, ORDER, B> {
    fn position(&self) -> [f64; N] {
        self.space.position(node_from_vertex(self.vertex))
    }

    fn vertex(&self) -> [usize; N] {
        self.vertex
    }

    fn value(&self, field: &[f64]) -> f64 {
        let linear = self.space.index_from_vertex(self.vertex);
        field[linear]
    }

    fn gradient<C: Condition<N>>(&self, cond: C, field: &[f64]) -> [f64; N] {
        let space = self.space.set_bc(BC::new(self.boundary.clone(), cond));

        array::from_fn(|axis| {
            let mut operators = [BasisOperator::Value; N];
            operators[axis] = BasisOperator::Derivative;
            space.evaluate::<ORDER>(self.vertex, operators, field)
        })
    }

    fn hessian<C: Condition<N>>(&self, cond: C, field: &[f64]) -> [[f64; N]; N] {
        let space = self.space.set_bc(BC::new(self.boundary.clone(), cond));

        let mut result = [[0.0; N]; N];

        for i in 0..N {
            for j in i..N {
                let mut operator = [BasisOperator::Value; N];
                operator[i] = BasisOperator::Derivative;
                operator[j] = BasisOperator::Derivative;

                if i == j {
                    operator[i] = BasisOperator::SecondDerivative;
                }

                result[i][j] = space.evaluate::<ORDER>(self.vertex, operator, field);
                result[j][i] = result[i][j]
            }
        }

        result
    }
}

/// A finite difference engine that only every relies on interior support (and can thus use better optimized stencils).
pub struct FdIntEngine<const N: usize, const ORDER: usize> {
    pub space: NodeSpace<N, Rectangle<N>>,
    pub vertex: [usize; N],
}

impl<const N: usize, const ORDER: usize> Engine<N> for FdIntEngine<N, ORDER> {
    fn position(&self) -> [f64; N] {
        self.space.position(node_from_vertex(self.vertex))
    }

    fn vertex(&self) -> [usize; N] {
        self.vertex
    }

    fn value(&self, field: &[f64]) -> f64 {
        let linear = self.space.index_from_vertex(self.vertex);
        field[linear]
    }

    fn gradient<C: Condition<N>>(&self, _cond: C, field: &[f64]) -> [f64; N] {
        let spacing = self.space.spacing();

        let (weights, border) = const {
            let order = Order::from_value(ORDER);
            let weights = BasisOperator::Derivative.weights(order, super::Support::Interior);
            let border = BasisOperator::Derivative.border(order);

            (weights, border)
        };

        array::from_fn(|axis| {
            let mut corner = node_from_vertex(self.vertex);
            corner[axis] -= border as isize;

            self.space.weights_axis(corner, weights, axis, field) / spacing[axis]
        })
    }

    fn hessian<C: Condition<N>>(&self, _cond: C, field: &[f64]) -> [[f64; N]; N] {
        let spacing = self.space.spacing();

        let (dweights, dborder) = const {
            let order = Order::from_value(ORDER);
            let weights = BasisOperator::Derivative.weights(order, super::Support::Interior);
            let border = BasisOperator::Derivative.border(order);

            (weights, border)
        };

        let (ddweights, ddborder) = const {
            let order = Order::from_value(ORDER);
            let weights = BasisOperator::SecondDerivative.weights(order, super::Support::Interior);
            let border = BasisOperator::SecondDerivative.border(order);

            (weights, border)
        };

        let mut result = [[0.0; N]; N];

        for i in 0..N {
            for j in i..N {
                if i == j {
                    let mut corner = node_from_vertex(self.vertex);
                    corner[i] -= ddborder as isize;
                    result[i][j] = self.space.weights_axis(corner, ddweights, i, field);
                    result[i][j] /= spacing[i] * spacing[j];
                } else {
                    let mut corner = node_from_vertex(self.vertex);
                    corner[i] -= dborder as isize;
                    corner[j] -= dborder as isize;

                    let mut weights: [&'static [f64]; N] = [&[1.0]; N];
                    weights[i] = dweights;
                    weights[j] = dweights;

                    result[i][j] = self.space.weights(corner, weights, field);
                    result[i][j] /= spacing[i] * spacing[j];
                    result[j][i] = result[i][j]
                }
            }
        }

        result
    }
}
