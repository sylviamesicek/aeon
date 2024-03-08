use crate::common::{Block, Operator};
use crate::lac::{LinearMap, LinearSolver};
use crate::uniform::UniformMesh;

pub struct UniformMultigrid<'mesh, const N: usize, Solver: LinearSolver> {
    mesh: &'mesh UniformMesh<N>,
    solver: Solver,
    max_iterations: usize,
    tolerance: f64,
    presmoothing: usize,
    postsmoothing: usize,

    scratch: Vec<f64>,
    diagonal: Vec<f64>,
    old: Vec<f64>,
    rhs: Vec<f64>,
}

impl<'mesh, const N: usize, Solver: LinearSolver> UniformMultigrid<'mesh, N, Solver> {
    pub fn new(
        mesh: &'mesh UniformMesh<N>,
        max_iterations: usize,
        tolerance: f64,
        presmoothing: usize,
        postsmoothing: usize,
        config: &Solver::Config,
    ) -> Self {
        let base_node_count = mesh.base_node_count();
        let node_count = mesh.node_count();

        let scratch = vec![0.0; node_count];
        let diag = vec![0.0; node_count];
        let old = vec![0.0; node_count];
        let rhs = vec![0.0; node_count];

        Self {
            mesh,
            solver: Solver::new(base_node_count, config),
            max_iterations,
            tolerance,

            presmoothing,
            postsmoothing,

            scratch,
            diagonal: diag,
            old,
            rhs,
        }
    }

    pub fn solve<O: Operator<N>>(self: &mut Self, operator: &O, b: &[f64], x: &mut [f64]) {
        self.scratch.fill(0.0);

        let irhs = self.mesh.norm(b);
        let tol = self.tolerance * irhs;

        if irhs <= 1e-60 {
            println!("Trivial Linear Problem.");

            x.fill(0.0);
            return;
        }

        for _ in 0..self.max_iterations {
            // Reset rhs
            self.rhs.copy_from_slice(b);

            // Cycle
            self.cycle(operator, self.mesh.level_count() - 1, x);

            // Compute residual
            for level in 0..self.mesh.level_count() {
                let range = self.mesh.level_node_range(level);

                let block = self.mesh.level_block(level);

                operator.apply(block, &x[range.clone()], &mut self.scratch[range.clone()]);

                for i in range {
                    self.scratch[i] = b[i] - self.scratch[i];
                }
            }

            let nres = self.mesh.norm(&self.scratch);

            if nres <= tol {
                return;
            }
        }

        println!("Multigrid Failed to Converge.");
    }

    fn cycle<O: Operator<N>>(self: &mut Self, operator: &O, level: usize, x: &mut [f64]) {
        if level == 0 {
            let base_node_count = self.mesh.level_node_offset(1);
            let block = self.mesh.level_block(level);

            let map = BaseLinearMap {
                dimension: base_node_count,
                operator,
                block,
            };

            self.solver
                .solve(
                    map,
                    &self.rhs[0..base_node_count],
                    &mut x[0..base_node_count],
                )
                .unwrap();

            return;
        }

        let block = self.mesh.level_block(level);
        let space = self.mesh.level_node_space(level);

        let fine = self.mesh.level_node_range(level);
        let coarse = self.mesh.level_node_range(level - 1);

        // ******************************
        // Presmoothing

        for _ in 0..self.presmoothing {
            operator.apply(
                block.clone(),
                &x[fine.clone()],
                &mut self.scratch[fine.clone()],
            );
            operator.apply_diag(
                block.clone(),
                &x[fine.clone()],
                &mut self.diagonal[fine.clone()],
            );

            for i in fine.clone() {
                x[i] += 2.0 / 3.0 * 1.0 / self.diagonal[i] * (self.rhs[i] - self.scratch[i]);
            }
        }

        // Restrict solution
        self.mesh.restrict_level(level, x);
        self.mesh.copy_level(level - 1, x, &mut self.old);

        // *****************************
        // Right hand side

        operator.apply(
            block.clone(),
            &x[fine.clone()],
            &mut self.scratch[fine.clone()],
        );

        for i in fine.clone() {
            self.scratch[i] = self.rhs[i] - self.scratch[i];
        }

        for i in 0..N {
            space.axis(i).restrict(&mut self.scratch[fine.clone()]);
        }

        self.mesh.restrict_level(level, &mut self.scratch);

        // *************************
        // Recurse

        self.cycle(operator, level - 1, x);

        // *************************
        // Error Correction

        for i in coarse.clone() {
            self.scratch[i] = x[i] - self.old[i]
        }

        self.mesh.prolong_level(level, &mut self.scratch);

        for i in 0..N {
            space.axis(i).prolong(&mut self.scratch[fine.clone()]);
        }

        for i in fine.clone() {
            x[i] += self.scratch[i];
        }

        // *************************
        // Postsmooth

        for _ in 0..self.postsmoothing {
            operator.apply(
                block.clone(),
                &x[fine.clone()],
                &mut self.scratch[fine.clone()],
            );
            operator.apply_diag(
                block.clone(),
                &x[fine.clone()],
                &mut self.diagonal[fine.clone()],
            );

            for i in fine.clone() {
                x[i] += 2.0 / 3.0 * 1.0 / self.diagonal[i] * (self.rhs[i] - self.scratch[i]);
            }
        }
    }
}

struct BaseLinearMap<'a, const N: usize, O: Operator<N>> {
    dimension: usize,
    operator: &'a O,
    block: Block<N>,
}

impl<'a, const N: usize, O: Operator<N>> LinearMap for BaseLinearMap<'a, N, O> {
    fn dimension(self: &Self) -> usize {
        self.dimension
    }

    fn apply(self: &Self, src: &[f64], dest: &mut [f64]) {
        self.operator.apply(self.block.clone(), src, dest);
    }
}
