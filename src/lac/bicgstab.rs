#![allow(clippy::needless_range_loop)]

use super::{LinearMap, LinearSolver};

/// Implementation of the stabilized bi-conjugate gradient method.
pub struct BiCGStabSolver {
    max_iterations: usize,
    tolerance: f64,
    dimension: usize,

    rg: Vec<f64>,
    rh: Vec<f64>,
    pg: Vec<f64>,
    ph: Vec<f64>,
    sg: Vec<f64>,
    sh: Vec<f64>,
    tg: Vec<f64>,
    vg: Vec<f64>,
    tp: Vec<f64>,

    mask: Vec<bool>,
}

impl BiCGStabSolver {
    /// Retrieves the tolerance of the multigrid solver.
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Retrieves the maximum number of iterations before the multigrid
    /// solver fails.
    pub fn max_iterations(&self) -> usize {
        self.max_iterations
    }
}

#[derive(Debug)]
pub enum BiCGStabError {
    Breakdown,
    FailedToConverge,
}

pub struct BiCGStabConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl LinearSolver for BiCGStabSolver {
    type Error = BiCGStabError;
    type Config = BiCGStabConfig;

    /// Builds a new BiCGStabSolver with the given dimension and
    /// configuration.
    fn new(dimension: usize, config: &BiCGStabConfig) -> Self {
        Self {
            dimension,
            max_iterations: config.max_iterations,
            tolerance: config.tolerance,

            rg: vec![0.0; dimension],
            rh: vec![0.0; dimension],
            pg: vec![0.0; dimension],
            ph: vec![0.0; dimension],
            sg: vec![0.0; dimension],
            sh: vec![0.0; dimension],
            tg: vec![0.0; dimension],
            vg: vec![0.0; dimension],
            tp: vec![0.0; dimension],

            mask: vec![true; dimension],
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn solve<M: LinearMap>(
        &mut self,
        mut map: M,
        rhs: &[f64],
        solution: &mut [f64],
    ) -> Result<(), Self::Error> {
        // Mask dofs
        map.mask(&mut self.mask);
        // Mask closure
        let mask = |dest: &mut [f64]| {
            for i in
                dest.iter_mut()
                    .zip(self.mask.iter())
                    .filter_map(|(v, &m)| if !m { Some(v) } else { None })
            {
                *i = 0.0;
            }
        };

        // Set termination tolerance
        map.apply(solution, &mut self.tp);
        mask(&mut self.tp);

        // let nrhs = norm(rhs);

        for i in 0..self.dimension {
            self.rg[i] = rhs[i] - self.tp[i];
        }
        mask(&mut self.rg);

        self.rh.clone_from_slice(&self.rg);
        self.sh.fill(0.0);
        self.ph.fill(0.0);

        let mut residual = norm(&self.rg);

        let tolerance = self.tolerance * residual;

        let mut iter = 0;
        let mut rho0 = 0.0;
        let mut rho1 = 0.0;
        let mut pra = 0.0;
        let mut prb = 0.0;
        let mut prc = 0.0;

        _ = residual;
        _ = rho1;
        _ = prb;

        while iter < self.max_iterations {
            rho1 = dot(&self.rg, &self.rh);

            if rho1 == 0.0 {
                return Err(BiCGStabError::Breakdown);
            }

            if iter == 0 {
                self.pg.clone_from_slice(&self.rg);
            } else {
                prb = (rho1 * pra) / (rho0 * prc);

                for i in 0..self.dimension {
                    self.pg[i] = self.rg[i] + prb * (self.pg[i] - prc * self.vg[i]);
                }
                mask(&mut self.pg);
            }

            rho0 = rho1;

            // Identity Preconditioner
            self.ph.clone_from_slice(&self.pg);
            map.apply(&self.ph, &mut self.vg);

            pra = rho1 / dot(&self.rh, &self.vg);

            // if dot(&self.rh, &self.vg).is_nan() {
            //     println!("DOT Hmm");
            //     println!("RH {:?}", &self.rh);
            //     println!("VG {:?}", &self.vg);

            //     panic!("Testing");
            // }

            for i in 0..self.dimension {
                self.sg[i] = self.rg[i] - pra * self.vg[i];
            }

            mask(&mut self.sg);

            if norm(&self.sg) <= 1e-60 {
                for i in 0..self.dimension {
                    solution[i] += pra * self.ph[i];
                }

                map.apply(solution, &mut self.tp);

                for i in 0..self.dimension {
                    self.rg[i] = rhs[i] - self.tp[i];
                }

                mask(&mut self.rg);

                // residual = norm(&self.rg);
                break;
            }

            self.sh.clone_from_slice(&self.sg);
            map.apply(&self.sh, &mut self.tg);

            prc = dot(&self.tg, &self.sg) / dot(&self.tg, &self.tg);

            for i in 0..self.dimension {
                solution[i] = solution[i] + pra * self.ph[i] + prc * self.sh[i];
                self.rg[i] = self.sg[i] - prc * self.tg[i];
            }

            mask(solution);
            mask(&mut self.rg);

            residual = norm(&self.rg);
            map.callback(iter, residual, solution);

            if residual <= tolerance {
                break;
            }

            iter += 1;
        }

        if iter < self.max_iterations {
            iter += 1;
        }

        if iter >= self.max_iterations {
            return Err(BiCGStabError::FailedToConverge);
        }

        Ok(())
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{BiCGStabConfig, BiCGStabError, BiCGStabSolver};
    use crate::lac::{IdentityMap, LinearSolver};

    #[test]
    fn identity_convergence() -> Result<(), BiCGStabError> {
        let mut solution = vec![0.0; 100];
        let mut rhs = vec![0.0; 100];

        for i in 0..100 {
            solution[i] = 0.0;
            rhs[i] = i as f64;
        }

        let mut solver: BiCGStabSolver = BiCGStabSolver::new(
            100,
            &BiCGStabConfig {
                max_iterations: 1000,
                tolerance: 10e-12,
            },
        );

        solver.solve(IdentityMap::new(100), &rhs, &mut solution)?;

        assert_eq!(solution == rhs, true);

        Ok(())
    }
}
