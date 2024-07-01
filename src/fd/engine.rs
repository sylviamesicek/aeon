use crate::{
    fd::{node_from_vertex, BasisOperator, BoundaryConditions, NodeSpace},
    geometry::Rectangle,
};
use std::array::{self, from_fn};

use super::Order;

/// An interface for computing values, gradients, and hessians of fields.
pub trait Engine<const N: usize> {
    fn position(&self) -> [f64; N];
    fn vertex(&self) -> [usize; N];
    fn value(&self, field: &[f64]) -> f64;
    fn gradient(&self, field: &[f64]) -> [f64; N];
    fn hessian(&self, field: &[f64]) -> [[f64; N]; N];
}

/// A finite difference engine of a given order, but potentially bordering a free boundary.
pub struct FdEngine<const N: usize, const ORDER: usize, B> {
    pub(crate) space: NodeSpace<N, B>,
    pub(crate) bounds: Rectangle<N>,
    pub(crate) vertex: [usize; N],
}

impl<const N: usize, const ORDER: usize, B: BoundaryConditions> Engine<N> for FdEngine<N, ORDER, B> {
    fn position(&self) -> [f64; N] {
        self.space
            .position(self.bounds.clone(), node_from_vertex(self.vertex))
    }

    fn vertex(&self) -> [usize; N] {
        self.vertex
    }

    fn value(&self, field: &[f64]) -> f64 {
        let linear = self.space.index_from_node(node_from_vertex(self.vertex));
        field[linear]
    }

    fn gradient(&self, field: &[f64]) -> [f64; N] {
        let spacing = self.space.spacing(self.bounds.clone());
        array::from_fn(|axis| {
            let mut operators = [BasisOperator::Value; N];
            operators[axis] = BasisOperator::Derivative;
            self.space.evaluate::<ORDER>(self.vertex, operators, field) / spacing[axis]
        })
    }

    fn hessian(&self, field: &[f64]) -> [[f64; N]; N] {
        let spacing = self.space.spacing(self.bounds.clone());

        let mut result = [[0.0; N]; N];

        for i in 0..N {
            for j in i..N {
                let mut operator = [BasisOperator::Value; N];
                operator[i] = BasisOperator::Derivative;
                operator[j] = BasisOperator::Derivative;

                if i == j {
                    operator[i] = BasisOperator::SecondDerivative;
                }

                result[i][j] = self.space.evaluate::<ORDER>(self.vertex, operator, field);
                result[i][j] /= spacing[i] * spacing[j];

                result[j][i] = result[i][j]
            }
        }

        result
    }
}

/// A finite difference engine that only every relies on interior support (and can thus use better optimized stencils).
pub struct FdIntEngine<const N: usize, const ORDER: usize, B>(pub(crate) FdEngine<N, ORDER, B>);

impl<const N: usize, const ORDER: usize, B: BoundaryConditions> Engine<N> for FdIntEngine<N, ORDER, B> {
    fn vertex(&self) -> [usize; N] {
        self.0.vertex()
    }

    fn position(&self) -> [f64; N] {
        self.0.position()
    }

    fn value(&self, field: &[f64]) -> f64 {
        self.0.value(field)
    }

    fn gradient(&self, field: &[f64]) -> [f64; N] {
        let spacing = self.0.space.spacing(self.0.bounds.clone());
        let (weights, border) = const {
            let order = Order::from_value(ORDER);
            let weights = BasisOperator::Derivative.weights(order, super::Support::Interior);
            let border = BasisOperator::Derivative.border(order);

            (weights, border)
        };

        from_fn(|axis| {
            let mut corner = node_from_vertex(self.0.vertex);
            corner[axis] -= border as isize;

            self.0.space.weights_axis(corner, weights, axis, field) / spacing[axis]
        })
    }

    fn hessian(&self, field: &[f64]) -> [[f64; N]; N] {
        let spacing = self.0.space.spacing(self.0.bounds.clone());

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
                    let mut corner = node_from_vertex(self.0.vertex);
                    corner[i] -= ddborder as isize;
                    result[i][j] = self.0.space.weights_axis(corner, ddweights, i, field);
                    result[i][j] /= spacing[i] * spacing[i];
                } else {
                    let mut corner = node_from_vertex(self.0.vertex);
                    corner[i] -= dborder as isize;
                    corner[j] -= dborder as isize;

                    let mut weights: [&'static [f64]; N] = [&[1.0]; N];
                    weights[i] = dweights;
                    weights[j] = dweights;

                    result[i][j] = self.0.space.weights(corner, weights, field);
                    result[i][j] /= spacing[i] * spacing[j];
                    result[j][i] = result[i][j]
                }
            }
        }

        result
    }
}
