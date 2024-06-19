use crate::fd::{Operator, Order, Support};
use crate::geometry::{IndexSpace, Rectangle};
use std::array::{self, from_fn};

/// Transforms a vertex into a node (just casts an array of `usize` -> `isize`).
pub fn node_from_vertex<const N: usize>(vertex: [usize; N]) -> [isize; N] {
    let mut result = [0isize; N];

    for axis in 0..N {
        result[axis] = vertex[axis] as isize;
    }

    result
}

/// A uniform rectangular domain of nodes to which
/// various derivative and interpolation kernels can be
/// applied.
#[derive(Debug, Clone)]
pub struct NodeSpace<const N: usize> {
    /// Number of cells along each axis (one less than then number of vertices).
    pub size: [usize; N],
    /// The physical bounds of the node space.
    pub bounds: Rectangle<N>,
    /// Number of ghost vertices in each direction
    pub ghost: usize,
}

impl<const N: usize> NodeSpace<N> {
    /// Computes the total number of nodes in the space.
    pub fn node_count(&self) -> usize {
        self.node_size().iter().product()
    }

    /// Converts a node into a linear index.
    pub fn index_from_node(&self, node: [isize; N]) -> usize {
        let cart = from_fn(|i| (node[i] + self.ghost as isize) as usize);
        IndexSpace::new(self.node_size()).linear_from_cartesian(cart)
    }

    /// Computes the number of cells along each axis.
    pub fn cell_size(&self) -> [usize; N] {
        self.size
    }

    /// Returns the number of vertices along each axis.
    pub fn vertex_size(&self) -> [usize; N] {
        let mut size = self.size;

        for s in size.iter_mut() {
            *s += 1;
        }

        size
    }

    /// Returns the total number of indices (including ghost indices) along each axis.
    pub fn node_size(&self) -> [usize; N] {
        let mut size = self.size;

        for s in size.iter_mut() {
            *s += 1 + 2 * self.ghost;
        }

        size
    }

    /// Returns the spacing along each axis of the node space.
    pub fn spacing(&self) -> [f64; N] {
        from_fn(|axis| self.spacing_axis(axis))
    }

    /// Returns the spacing along a given axis.
    pub fn spacing_axis(&self, axis: usize) -> f64 {
        self.bounds.size[axis] / self.size[axis] as f64
    }

    /// Computes the position of the given vertex.
    pub fn position(&self, node: [isize; N]) -> [f64; N] {
        let mut result = [0.0; N];

        for i in 0..N {
            result[i] = self.bounds.origin[i] + self.spacing_axis(i) * node[i] as f64;
        }

        result
    }

    /// Returns the value of the field at the given node.
    pub fn value(&self, node: [isize; N], src: &[f64]) -> f64 {
        src[self.index_from_node(node)]
    }

    /// Sets the value of the field at the given node.
    pub fn set_value(&self, node: [isize; N], v: f64, dest: &mut [f64]) {
        dest[self.index_from_node(node)] = v;
    }

    fn support<const ORDER: usize>(
        &self,
        vertex: [usize; N],
        operator: [Operator; N],
    ) -> [Support; N] {
        array::from_fn(|axis| self.support_axis::<ORDER>(vertex, operator[axis], axis))
    }

    fn support_axis<const ORDER: usize>(
        &self,
        vertex: [usize; N],
        operator: Operator,
        axis: usize,
    ) -> Support {
        let border = operator.border(const { Order::from_value(ORDER) });

        if vertex[axis] < border {
            Support::Negative(vertex[axis])
        } else if vertex[axis] > self.size[axis] - border {
            Support::Positive(self.size[axis] - vertex[axis])
        } else {
            Support::Interior
        }
    }

    fn corner<const ORDER: usize>(
        &self,
        vertex: [usize; N],
        operator: [Operator; N],
        wsize: [usize; N],
    ) -> [isize; N] {
        array::from_fn(|axis| self.corner_axis::<ORDER>(vertex, operator[axis], wsize[axis], axis))
    }

    fn corner_axis<const ORDER: usize>(
        &self,
        vertex: [usize; N],
        operator: Operator,
        wsize: usize,
        axis: usize,
    ) -> isize {
        let border = operator.border(const { Order::from_value(ORDER) });

        if vertex[axis] < border {
            0
        } else if vertex[axis] > self.size[axis] - border {
            vertex[axis] as isize - (border as isize)
        } else {
            self.size[axis] as isize + 1 - wsize as isize
        }
    }

    pub fn weights_axis(
        &self,
        corner: [isize; N],
        weights: &'static [f64],
        axis: usize,
        field: &[f64],
    ) -> f64 {
        let mut result = 0.0;

        let mut node = corner;
        for index in 0..weights.len() {
            node[axis] = corner[axis] + index as isize;
            result += self.value(node, field) * weights[index];
        }

        result
    }

    pub fn weights(&self, corner: [isize; N], weights: [&'static [f64]; N], field: &[f64]) -> f64 {
        let wsize: [_; N] = from_fn(|axis| weights[axis].len());

        let mut result = 0.0;

        for index in IndexSpace::new(wsize).iter() {
            let mut weight = 1.0;

            for axis in 0..N {
                weight *= weights[axis][index[axis]];
            }

            let node = from_fn(|axis| corner[axis] + index[axis] as isize);
            result += self.value(node, field) * weight;
        }

        result
    }

    pub fn evaluate_axis<const ORDER: usize>(
        &self,
        vertex: [usize; N],
        operator: Operator,
        axis: usize,
        field: &[f64],
    ) -> f64 {
        let spacing = self.spacing_axis(axis);
        let order = const { Order::from_value(ORDER) };
        let support = self.support_axis::<ORDER>(vertex, operator, axis);
        let weights = operator.weights(order, support);
        let wsize = weights.len();

        let mut corner = node_from_vertex(vertex);
        corner[axis] = self.corner_axis::<ORDER>(vertex, operator, wsize, axis);

        self.weights_axis(corner, weights, axis, field) * operator.scale(spacing)
    }

    pub fn evaluate<const ORDER: usize>(
        &self,
        vertex: [usize; N],
        operator: [Operator; N],
        field: &[f64],
    ) -> f64 {
        let spacing = self.spacing();
        let order = const { Order::from_value(ORDER) };
        let support = self.support::<ORDER>(vertex, operator);
        let weights: [_; N] = array::from_fn(|axis| operator[axis].weights(order, support[axis]));
        let wsize: [_; N] = array::from_fn(|axis| weights[axis].len());
        let corner = self.corner::<ORDER>(vertex, operator, wsize);

        let mut result = self.weights(corner, weights, field);

        for axis in 0..N {
            result *= operator[axis].scale(spacing[axis])
        }

        result
    }
}
