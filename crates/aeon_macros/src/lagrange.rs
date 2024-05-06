use num::{rational::Rational64 as Ratio, One, Zero};
use std::ops::Index;

/// Represents a stencil: a set of nodal points from
/// which one  can derive stencil weights.
#[derive(Debug, Clone)]
pub struct Stencil {
    grid: Vec<Ratio>,
}

impl Stencil {
    /// Creates a vertex centered grid with left + right + 1 nodal points.
    pub fn vertex(left: u64, right: u64) -> Self {
        let range = -(left as i64)..=(right as i64);
        let grid = range.into_iter().map(|i| Ratio::new(i, 1)).collect();
        Self { grid }
    }

    /// Creates a cell centered grid with left + right nodal points.
    pub fn cell(left: u64, right: u64) -> Self {
        let range = -(left as i64)..(right as i64);
        let grid = range
            .into_iter()
            .map(|i| Ratio::new(2 * i + 1, 2))
            .collect();
        Self { grid }
    }

    /// Constructs weights for interpolation at the given point.
    pub fn value_weights(self, point: Ratio) -> Vec<Ratio> {
        let mut weights = vec![Ratio::one(); self.grid.len()];

        for i in 0..self.grid.len() {
            for j in 0..self.grid.len() {
                if i != j {
                    weights[i] *= (point - self[j]) / (self[i] - self[j])
                }
            }
        }

        weights
    }

    /// Constructs weights for approximating a derivative at the given point.
    pub fn derivative_weights(self, point: Ratio) -> Vec<Ratio> {
        let mut weights = vec![Ratio::zero(); self.grid.len()];

        for i in 0..self.grid.len() {
            for j in 0..self.grid.len() {
                if i != j {
                    let mut result = Ratio::one();

                    for k in 0..self.grid.len() {
                        if i != k && j != k {
                            result *= (point - self[k]) / (self[i] - self[k]);
                        }
                    }

                    weights[i] += result / (self[i] - self[j])
                }
            }
        }

        weights
    }

    pub fn second_derivative_weights(self, point: Ratio) -> Vec<Ratio> {
        let mut weights = vec![Ratio::zero(); self.grid.len()];

        for i in 0..self.grid.len() {
            for j in 0..self.grid.len() {
                if i != j {
                    let mut result1 = Ratio::zero();

                    for k in 0..self.grid.len() {
                        if i != k && j != k {
                            let mut result2 = Ratio::one();

                            for l in 0..self.grid.len() {
                                if l != i && l != j && l != k {
                                    result2 *= (point - self[l]) / (self[i] - self[l]);
                                }
                            }

                            result1 += result2 / (self[i] - self[k]);
                        }
                    }

                    weights[i] += result1 / (self[i] - self[j])
                }
            }
        }

        weights
    }
}

impl Index<usize> for Stencil {
    type Output = Ratio;

    fn index(&self, index: usize) -> &Self::Output {
        &self.grid[index]
    }
}
