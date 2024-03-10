use crate::arena::Arena;
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
        let rhs = vec![0.0; node_count];

        Self {
            mesh,
            solver: Solver::new(base_node_count, config),
            max_iterations,
            tolerance,

            presmoothing,
            postsmoothing,

            scratch,
            rhs,
        }
    }

    pub fn solve<O: Operator<N>>(
        self: &mut Self,
        arena: &mut Arena,
        operator: &O,
        b: &[f64],
        x: &mut [f64],
    ) {
        self.scratch.fill(0.0);

        let irhs = self.mesh.norm(b);
        let tol = self.tolerance * irhs;

        if irhs <= 1e-60 {
            println!("Trivial Linear Problem.");

            x.fill(0.0);
            return;
        }

        for i in 0..self.max_iterations {
            // Reset rhs
            self.rhs.copy_from_slice(b);

            // Cycle
            self.cycle(arena, operator, self.mesh.level_count() - 1, x);

            // Compute residual
            self.mesh
                .residual(arena, &self.rhs, operator, &x, &mut self.scratch);

            let nres = self.mesh.norm(&self.scratch);

            println!("Iteration {i}, Residual {nres}");

            if nres <= tol {
                return;
            }
        }

        println!("Multigrid Failed to Converge.");
    }

    fn cycle<O: Operator<N>>(
        self: &mut Self,
        arena: &mut Arena,
        operator: &O,
        level: usize,
        x: &mut [f64],
    ) {
        if level == 0 {
            let base_node_count = self.mesh.base_node_count();
            let block = self.mesh.level_block(level);

            println!("Solving BASE multigrid, node count {base_node_count}");
            // println!("{:?}", &self.rhs[0..base_node_count]);
            // println!("{:?}", &x[0..base_node_count]);

            let map = BaseLinearMap {
                dimension: base_node_count,
                operator,
                block,
                arena,
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

        let fine = self.mesh.level_node_range(level);
        let coarse = self.mesh.level_node_range(level - 1);

        // ******************************
        // Presmoothing

        for _ in 0..self.presmoothing {
            let diag: &mut [f64] = arena.alloc::<f64>(block.len());

            operator.apply(
                arena,
                &block,
                &x[fine.clone()],
                &mut self.scratch[fine.clone()],
            );
            operator.apply_diag(arena, &block, diag);

            for (local, global) in fine.clone().enumerate() {
                x[global] +=
                    2.0 / 3.0 * 1.0 / diag[local] * (self.rhs[global] - self.scratch[global]);
            }

            arena.reset();
        }

        // *****************************
        // Right hand side

        self.mesh
            .residual_level(level, arena, &self.rhs, operator, &x, &mut self.scratch);
        self.mesh.restrict_level_full(level, &mut self.scratch);
        self.mesh.restrict_level(level, x);

        // Tau correction
        for i in coarse.clone() {
            self.rhs[i] = x[i] + self.scratch[i];
        }

        // *************************
        // Recurse

        self.cycle(arena, operator, level - 1, x);

        // *************************
        // Error Correction

        self.mesh.copy_level(level, &x, &mut self.scratch);
        self.mesh.restrict_level(level, &mut self.scratch);

        for i in coarse.clone() {
            self.scratch[i] = x[i] - self.scratch[i];
        }

        self.mesh.prolong_level_full(level, &mut self.scratch);

        for i in fine.clone() {
            x[i] += self.scratch[i];
        }

        // *************************
        // Postsmooth

        for _ in 0..self.postsmoothing {
            let diag: &mut [f64] = arena.alloc::<f64>(block.len());

            operator.apply(
                arena,
                &block,
                &x[fine.clone()],
                &mut self.scratch[fine.clone()],
            );
            operator.apply_diag(arena, &block, diag);

            for (local, global) in fine.clone().enumerate() {
                x[global] +=
                    2.0 / 3.0 * 1.0 / diag[local] * (self.rhs[global] - self.scratch[global]);
            }

            arena.reset();
        }
    }
}

struct BaseLinearMap<'a, const N: usize, O: Operator<N>> {
    dimension: usize,
    operator: &'a O,
    block: Block<N>,
    arena: &'a mut Arena,
}

impl<'a, const N: usize, O: Operator<N>> LinearMap for BaseLinearMap<'a, N, O> {
    fn dimension(self: &Self) -> usize {
        self.dimension
    }

    fn apply(self: &mut Self, src: &[f64], dest: &mut [f64]) {
        self.operator.apply(&mut self.arena, &self.block, src, dest);
        self.arena.reset();
    }

    fn callback(self: &Self, iteration: usize, residual: f64, _: &[f64]) {
        println!("Base Iteration {iteration}, Residual {residual}");
    }
}
