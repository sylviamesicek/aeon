use crate::{
    element::{ApproxOperator, Monomials, Uniform, Values},
    geometry::{IndexSpace, Split},
};
use faer::{ColRef, Mat};
use std::array;

/// A reference element defined on [-1, 1]^N. This element implements code
/// for wavelet transformations and general operator approximation, whereas
/// `NodeSpace` implements a much stricter subset of operations with fixed
/// precomputed weights.
#[derive(Clone, Debug)]
pub struct Element<const N: usize> {
    /// Order of basis functions used in element.
    order: usize,
    /// Width of element in nodes
    width: usize,
    /// Grid of points within element.
    grid: Vec<f64>,
    /// Positions of points within element
    /// after refinement
    grid_refined: Vec<f64>,
    /// Interpolation stencils.
    stencils: Mat<f64>,
}

impl<const N: usize> Element<N> {
    /// Constructs a reference element with uniformly placed
    /// support points with `width + 1` points along each axis.
    pub fn uniform(width: usize, order: usize) -> Self {
        assert!(width >= order);
        let (grid, grid_refined) = Self::uniform_grid(width);
        debug_assert!(grid.len() == width + 1);
        debug_assert!(grid_refined.len() == 2 * width + 1);

        let spacing = 2.0 / width as f64;
        let points = Vec::from_iter(
            (0..width)
                .into_iter()
                .map(|j| [-1.0 + spacing * (j as f64 + 0.5)]),
        );

        let mut approx = ApproxOperator::default();
        approx
            .build(
                &Uniform::new([width + 1]),
                &Monomials::new([order + 1]),
                &Values(points.as_slice()),
            )
            .unwrap();

        let stencils = approx.shape().to_owned();

        debug_assert!(stencils.nrows() == width + 1);
        debug_assert!(stencils.ncols() == width);

        Self {
            width,
            order,
            grid,
            grid_refined,
            stencils,
        }
    }

    fn uniform_grid(width: usize) -> (Vec<f64>, Vec<f64>) {
        let spacing = 2.0 / width as f64;

        let grid = (0..=width)
            .map(|i| i as f64 * spacing - 1.0)
            .collect::<Vec<_>>();

        let grid_refined = (0..=2 * width)
            .map(|i| i as f64 * spacing / 2.0 - 1.0)
            .collect::<Vec<_>>();

        (grid, grid_refined)
    }

    /// Number of support points in the reference element.
    pub fn support(&self) -> usize {
        (self.width + 1).pow(N as u32)
    }

    /// Number of supports points after the element has been refined.
    pub fn support_refined(&self) -> usize {
        (2 * self.width + 1).pow(N as u32)
    }

    /// Retrieves the order of the element.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Retrieves the width of the element.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Retrieves the grid along one axis.
    pub fn grid(&self) -> &[f64] {
        &self.grid
    }

    pub fn prolong_stencil(&self, target: usize) -> ColRef<'_, f64> {
        self.stencils.col(target)
    }

    /// Iterates the position of a support point in the element.
    pub fn position(&self, index: [usize; N]) -> [f64; N] {
        array::from_fn(|axis| self.grid[index[axis]])
    }

    pub fn position_refined(&self, index: [usize; N]) -> [f64; N] {
        array::from_fn(|axis| self.grid_refined[index[axis]])
    }

    // *********************
    // Point Iteration *****
    // *********************

    pub fn space(&self) -> IndexSpace<N> {
        IndexSpace::new([self.width + 1; N])
    }

    pub fn space_refined(&self) -> IndexSpace<N> {
        IndexSpace::new([2 * self.width + 1; N])
    }

    /// Iterates over all nodal coefficients in a wavelet representation on this element.
    pub fn nodal_indices(&self) -> impl Iterator<Item = [usize; N]> {
        IndexSpace::new([self.width + 1; N])
            .iter()
            .map(|v| array::from_fn(|axis| v[axis] * 2))
    }

    /// Iterates over all diagonal detail coefficients in a wavelet representation on this element.
    pub fn diagonal_indices(&self) -> impl Iterator<Item = [usize; N]> {
        IndexSpace::new([self.width; N])
            .iter()
            .map(|v| array::from_fn(|axis| v[axis] * 2 + 1))
    }

    /// Iterates over diagonal detail coefficients in a wavelet representation of this element,
    /// ignoring elements within a `buffer` around the edge of the refined support.
    pub fn diagonal_int_indices(&self, buffer: usize) -> impl Iterator<Item = [usize; N]> {
        debug_assert!(buffer % 2 == 0);

        IndexSpace::new([self.width - buffer; N])
            .iter()
            .map(move |v| array::from_fn(|axis| 2 * v[axis] + 1 + buffer))
    }

    /// Iterates over all detail coefficients in a wavelet representation on this element.
    pub fn detail_indices(&self) -> impl Iterator<Item = [usize; N]> {
        let cells = IndexSpace::new([self.width; N]).iter();

        cells.flat_map(|index| {
            Split::<N>::enumerate().skip(1).map(move |mask| {
                let mut point = index;

                for axis in 0..N {
                    point[axis] *= 2;

                    if mask.is_set(axis) {
                        point[axis] += 1;
                    }
                }

                point
            })
        })
    }

    pub fn nodal_points(&self) -> impl Iterator<Item = usize> {
        let space = IndexSpace::new([2 * self.width + 1; N]);
        self.nodal_indices()
            .map(move |index| space.linear_from_cartesian(index))
    }

    pub fn diagonal_points(&self) -> impl Iterator<Item = usize> {
        let space = IndexSpace::new([2 * self.width + 1; N]);
        self.diagonal_indices()
            .map(move |index| space.linear_from_cartesian(index))
    }

    pub fn diagonal_int_points(&self, buffer: usize) -> impl Iterator<Item = usize> {
        let space = IndexSpace::new([2 * self.width + 1; N]);
        self.diagonal_int_indices(buffer)
            .map(move |index| space.linear_from_cartesian(index))
    }

    pub fn detail_points(&self) -> impl Iterator<Item = usize> {
        let space = IndexSpace::new([2 * self.width + 1; N]);
        self.detail_indices()
            .map(move |index| space.linear_from_cartesian(index))
    }

    // ****************************
    // Prolongation ***************
    // ****************************

    /// Prolongs data from the element to a refined version of the element.
    pub fn prolong(&self, source: &[f64], dest: &mut [f64]) {
        self.inject(source, dest);
        self.prolong_in_place(dest);
    }

    /// Fills in-between points on dest using interpolation, assuming that nodal
    /// points on dest have been properly filled.
    pub fn prolong_in_place(&self, dest: &mut [f64]) {
        debug_assert!(dest.len() == self.support_refined());

        let space = IndexSpace::new([2 * self.width + 1; N]);

        // And now perform interpolation
        for axis in (0..N).rev() {
            let mut psize = [0; N];

            for i in 0..axis {
                psize[i] = self.width + 1;
            }
            psize[axis] = self.width;
            for i in (axis + 1)..N {
                psize[i] = 2 * self.width + 1;
            }

            for mut point in IndexSpace::new(psize).iter() {
                for i in 0..axis {
                    point[i] *= 2;
                }

                let stencil = self.stencils.col(point[axis]);

                point[axis] *= 2;
                point[axis] += 1;

                let center = space.linear_from_cartesian(point);
                dest[center] = 0.0;

                for i in 0..=self.width {
                    point[axis] = 2 * i;
                    dest[center] += stencil[i] * dest[space.linear_from_cartesian(point)];
                }
            }
        }
    }

    // *******************************
    // Injection *********************
    // *******************************

    /// Performs injection from source to a refined dest.
    pub fn inject(&self, source: &[f64], dest: &mut [f64]) {
        debug_assert!(source.len() == self.support());
        debug_assert!(dest.len() == self.support_refined());

        // Perform injection
        let source_space = IndexSpace::new([self.width + 1; N]);
        let dest_space = IndexSpace::new([2 * self.width + 1; N]);

        for (pindex, point) in source_space.iter().enumerate() {
            let refined: [_; N] = array::from_fn(|axis| 2 * point[axis]);
            let rindex = dest_space.linear_from_cartesian(refined);
            dest[rindex] = source[pindex];
        }
    }

    /// Restricts refined data from source onto dest.
    pub fn restrict(&self, source: &[f64], dest: &mut [f64]) {
        debug_assert!(source.len() == self.support_refined());
        debug_assert!(dest.len() == self.support());

        let source_space = IndexSpace::new([2 * self.width + 1; N]);
        let dest_space = IndexSpace::new([self.width + 1; N]);

        for (pindex, point) in source_space.iter().enumerate() {
            let refined: [_; N] = array::from_fn(|axis| point[axis] / 2);
            let rindex = dest_space.linear_from_cartesian(refined);
            dest[rindex] = source[pindex];
        }
    }

    // *******************************
    // Wavelet Expansion *************
    // *******************************

    /// Computes the wavelet coefficients for the given function.
    pub fn wavelet(&self, source: &[f64], dest: &mut [f64]) {
        debug_assert!(source.len() == self.support_refined());
        debug_assert!(dest.len() == self.support_refined());

        // Copies data from the source to noal points on dest.
        for point in self.nodal_points() {
            dest[point] = source[point];
        }

        self.prolong_in_place(dest);

        // Iterates over the detail coefficients.
        for point in self.detail_points() {
            dest[point] -= source[point];
        }
    }

    /// Computes the relative error between the wavelet's representation
    /// and a nodal approximation.
    pub fn wavelet_rel_error(&self, coefs: &[f64]) -> f64 {
        let scale = self
            .nodal_points()
            .map(|v| coefs[v].abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        self.wavelet_abs_error(coefs) / scale
    }

    /// Computes the absolute error between the wavelet's representation
    /// and a nodal approximation.
    pub fn wavelet_abs_error(&self, coefs: &[f64]) -> f64 {
        self.diagonal_points()
            .map(|v| coefs[v].abs())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::Element;

    #[test]
    fn iteration() {
        let element = Element::<2>::uniform(2, 1);

        let mut nodal = element.nodal_indices();
        assert_eq!(nodal.next(), Some([0, 0]));
        assert_eq!(nodal.next(), Some([2, 0]));
        assert_eq!(nodal.next(), Some([4, 0]));
        assert_eq!(nodal.next(), Some([0, 2]));
        assert_eq!(nodal.next(), Some([2, 2]));
        assert_eq!(nodal.next(), Some([4, 2]));
        assert_eq!(nodal.next(), Some([0, 4]));
        assert_eq!(nodal.next(), Some([2, 4]));
        assert_eq!(nodal.next(), Some([4, 4]));
        assert_eq!(nodal.next(), None);

        let mut diagonal = element.diagonal_indices();
        assert_eq!(diagonal.next(), Some([1, 1]));
        assert_eq!(diagonal.next(), Some([3, 1]));
        assert_eq!(diagonal.next(), Some([1, 3]));
        assert_eq!(diagonal.next(), Some([3, 3]));
        assert_eq!(diagonal.next(), None);

        let mut diagonal_int = element.diagonal_int_indices(0);
        assert_eq!(diagonal_int.next(), Some([1, 1]));
        assert_eq!(diagonal_int.next(), Some([3, 1]));
        assert_eq!(diagonal_int.next(), Some([1, 3]));
        assert_eq!(diagonal_int.next(), Some([3, 3]));
        assert_eq!(diagonal_int.next(), None);

        let mut detail = element.detail_indices();
        assert_eq!(detail.next(), Some([1, 0]));
        assert_eq!(detail.next(), Some([0, 1]));
        assert_eq!(detail.next(), Some([1, 1]));

        assert_eq!(detail.next(), Some([3, 0]));
        assert_eq!(detail.next(), Some([2, 1]));
        assert_eq!(detail.next(), Some([3, 1]));

        assert_eq!(detail.next(), Some([1, 2]));
        assert_eq!(detail.next(), Some([0, 3]));
        assert_eq!(detail.next(), Some([1, 3]));

        assert_eq!(detail.next(), Some([3, 2]));
        assert_eq!(detail.next(), Some([2, 3]));
        assert_eq!(detail.next(), Some([3, 3]));
        assert_eq!(detail.next(), None);
    }

    #[test]
    fn interior_indices() {
        let element = Element::<1>::uniform(6, 4);

        let mut indices = element.diagonal_int_points(2);
        assert_eq!(indices.next(), Some(3));
        assert_eq!(indices.next(), Some(5));
        assert_eq!(indices.next(), Some(7));
        assert_eq!(indices.next(), Some(9));
        assert_eq!(indices.next(), None);

        // Width 4, ghost 3
        let width = 6;
        let ghost = 3;

        let buffer = 2 * (ghost / 2); // 2
        let support = (width + 2 * buffer) / 2; // 5

        let element = Element::<1>::uniform(support, 4);

        let mut indices = element.diagonal_int_points(buffer);
        assert_eq!(indices.next(), Some(3));
        assert_eq!(indices.next(), Some(5));
        assert_eq!(indices.next(), Some(7));
        assert_eq!(indices.next(), None);
    }

    fn prolong(h: f64) -> f64 {
        let element = Element::<2>::uniform(6, 4);

        let space = element.space_refined();

        let mut values = vec![0.0; element.support_refined()];
        let mut coefs = vec![0.0; element.support_refined()];

        for index in space.iter() {
            let [x, y] = element.position_refined(index);
            let point = space.linear_from_cartesian(index);
            values[point] = (x * h).sin() * (y * h).exp();
        }

        element.wavelet(&values, &mut coefs);
        element.wavelet_abs_error(&coefs)
    }

    #[test]
    fn convergence() {
        let error1 = prolong(0.1);
        let error2 = prolong(0.05);
        let error4 = prolong(0.025);
        let error8 = prolong(0.0125);

        dbg!(error1 / error2);
        dbg!(error2 / error4);
        dbg!(error4 / error8);

        assert!(error1 / error2 >= 32.);
        assert!(error2 / error4 >= 32.);
        assert!(error4 / error8 >= 32.);
    }
}
