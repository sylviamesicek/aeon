use std::array;

use faer::{solvers::SpSolver, Mat, MatRef};

use aeon_geometry::IndexSpace;

fn monomial_vandermonde(grid: &[f64]) -> Mat<f64> {
    Mat::from_fn(grid.len(), grid.len(), |i, j| grid[i].powi(j as i32))
}

fn monomial_interpolation(order: usize, point: f64) -> f64 {
    point.powi(order as i32)
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
        let vandermonde = monomial_vandermonde(&grid);

        let basis = vandermonde.transpose().qr();
        let rhs = Mat::from_fn(order + 1, order, |i, j| {
            let point = -1.0 + spacing * (j as f64 + 0.5);
            monomial_interpolation(i, point)
        });

        let interp = basis.solve(rhs);

        Self {
            order,
            grid,
            vandermonde,
            interp,
        }
    }

    /// Number of support points in the reference element.
    pub fn num_points(&self) -> usize {
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
        debug_assert!(source.len() == self.num_points());
        debug_assert!(dest.len() == (2 * self.order + 1).pow(N as u32));

        // Perform injection
        let space = IndexSpace::new([self.order + 1; N]);
        let space_refined = IndexSpace::new([2 * self.order + 1; N]);

        for (pindex, point) in space.iter().enumerate() {
            let refined: [_; N] = array::from_fn(|axis| 2 * point[axis]);
            let rindex = space_refined.linear_from_cartesian(refined);
            dest[rindex] = source[pindex];
        }

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
