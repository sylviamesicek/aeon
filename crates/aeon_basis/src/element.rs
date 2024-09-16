use std::array;

use faer::{solvers::SpSolver, Mat, MatRef};

use aeon_geometry::IndexSpace;

pub trait Basis {
    fn vandermonde(grid: &[f64]) -> Mat<f64>;
    fn evaluate(&self, point: f64) -> f64;
    fn degree(degree: usize) -> Self;
}

pub struct Monomial(pub usize);

impl Basis for Monomial {
    fn vandermonde(grid: &[f64]) -> Mat<f64> {
        Mat::from_fn(grid.len(), grid.len(), |i, j| grid[i].powi(j as i32))
    }

    fn evaluate(&self, point: f64) -> f64 {
        point.powi(self.0 as i32)
    }

    fn degree(degree: usize) -> Self {
        Self(degree)
    }
}

/// A reference element defined on [-1, 1]^N.
pub struct RefElement<const N: usize> {
    order: usize,
    /// Grid of points within element.
    grid: Vec<f64>,
    /// Vandermonde matrix on grid.
    vandermonde: Mat<f64>,
    /// Interpolation stencils.
    interp: Mat<f64>,
}

impl<const N: usize> RefElement<N> {
    /// Constructs a reference element with uniformly placed
    /// support points with `order + 1` points along each axis.
    pub fn uniform(order: usize) -> Self {
        let spacing = 2.0 / order as f64;
        let grid = (0..=order)
            .map(|i| i as f64 * spacing - 1.0)
            .collect::<Vec<_>>();
        let vandermonde = Monomial::vandermonde(&grid);

        let m = vandermonde.transpose().qr();
        let rhs = Self::uniform_rhs::<Monomial>(order);
        let interp = m.solve(rhs);

        Self {
            order,
            grid,
            vandermonde,
            interp,
        }
    }

    fn uniform_rhs<B: Basis>(order: usize) -> Mat<f64> {
        let spacing = 2.0 / order as f64;

        let mut result = Mat::zeros(order + 1, order);

        for i in 0..=order {
            let basis = B::degree(i);

            for j in 0..order {
                let point = -1.0 + spacing * (j as f64 + 0.5);

                result[(i, j)] = basis.evaluate(point);
            }
        }

        result
    }

    /// Number of support points in the reference element.
    pub fn support(&self) -> usize {
        (self.order + 1).pow(N as u32)
    }

    pub fn order(&self) -> usize {
        self.order
    }

    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    pub fn vandermonde(&self) -> MatRef<f64> {
        self.vandermonde.as_ref()
    }

    pub fn prolong(&self, source: &[f64], dest: &mut [f64]) {
        self.refine(source, dest);
        self.prolong_in_place(dest);
    }

    /// Performs injection from source to a refined dest.
    pub fn refine(&self, source: &[f64], dest: &mut [f64]) {
        debug_assert!(source.len() == self.support());
        debug_assert!(dest.len() == (2 * self.order + 1).pow(N as u32));

        // Perform injection
        let source_space = IndexSpace::new([self.order + 1; N]);
        let dest_space = IndexSpace::new([2 * self.order + 1; N]);

        for (pindex, point) in source_space.iter().enumerate() {
            let refined: [_; N] = array::from_fn(|axis| 2 * point[axis]);
            let rindex = dest_space.linear_from_cartesian(refined);
            dest[rindex] = source[pindex];
        }
    }

    pub fn prolong_in_place(&self, dest: &mut [f64]) {
        debug_assert!(dest.len() == (2 * self.order + 1).pow(N as u32));

        let space = IndexSpace::new([2 * self.order + 1; N]);

        // And now perform interpolation
        for axis in (0..N).rev() {
            let mut psize = [0; N];

            for i in 0..axis {
                psize[i] = self.order + 1;
            }
            psize[axis] = self.order;
            for i in (axis + 1)..N {
                psize[i] = 2 * self.order + 1;
            }

            for mut point in IndexSpace::new(psize).iter() {
                for i in 0..axis {
                    point[i] *= 2;
                }

                let stencil = self.interp.col(point[axis]);

                point[axis] *= 2;
                point[axis] += 1;

                let index = space.linear_from_cartesian(point);
                dest[index] = 0.0;

                for i in 0..=self.order {
                    point[axis] = 2 * i;
                    dest[index] += stencil[i] * dest[space.linear_from_cartesian(point)];
                }
            }
        }
    }
}

pub struct WaveletElement<const N: usize> {
    element: RefElement<N>,
    order: usize,
}

impl<const N: usize> WaveletElement<N> {
    pub fn new(order: usize) -> Self {
        // Padding must be sufficiently large, and order must be even.
        assert!(order % 2 == 0);
        assert!(order > 0);

        WaveletElement {
            order,
            element: RefElement::uniform(order),
        }
    }

    pub fn support(&self) -> usize {
        (2 * self.order + 1).pow(N as u32)
    }

    pub fn padding(&self) -> usize {
        self.order / 2
    }

    /// Finds the point at the given cartesian index.
    pub fn index_to_point(&self, index: [isize; N]) -> usize {
        let space = IndexSpace::new([2 * self.order + 1; N]);
        let index = array::from_fn(|axis| (index[axis] + self.padding() as isize) as usize);
        space.linear_from_cartesian(index)
    }

    pub fn prolong(&self, source: &[f64], dest: &mut [f64]) {
        debug_assert!(source.len() == self.support());
        debug_assert!(dest.len() == self.support());

        let space = IndexSpace::new([2 * self.order + 1; N]);

        for index in IndexSpace::new([self.order + 1; N]).iter() {
            let point = space.linear_from_cartesian(array::from_fn(|axis| index[axis] * 2));
            dest[point] = source[point];
        }

        self.element.prolong_in_place(dest);
    }
}
