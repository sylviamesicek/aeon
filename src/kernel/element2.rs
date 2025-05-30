use faer::Mat;

#[derive(Clone, Copy, Debug)]
pub struct Monomial(pub usize);

impl Monomial {
    fn vandermonde(grid: &[f64], order: usize) -> Mat<f64> {
        Mat::from_fn(grid.len(), order, |i, j| grid[i].powi(j as i32))
    }

    fn evaluate(&self, point: f64) -> f64 {
        point.powi(self.0 as i32)
    }

    fn degree(degree: usize) -> Self {
        Self(degree)
    }
}
