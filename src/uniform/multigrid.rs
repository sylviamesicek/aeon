use crate::arena::Arena;
use crate::common::{Block, Boundary, BoundaryCallback, BoundarySet, Operator};
use crate::lac::{LinearMap, LinearSolver};
use crate::uniform::UniformMesh;

/// Configuration for uniform multigrid.
#[derive(Debug, Clone)]
pub struct UniformMultigridConfig {
    /// Maximum allowed iterations.
    pub max_iterations: usize,
    /// Tolerance from error.
    pub tolerance: f64,
    /// Number of Presmoothing
    pub presmoothing: usize,
    /// Number of postsmoothing steps
    pub postsmoothing: usize,
}

/// Implements the multigrid algorithm for a uniform mesh.
pub struct UniformMultigrid<'m, const N: usize, Solver: LinearSolver> {
    /// Mesh that the multigrid algorithm operators over.
    mesh: &'m UniformMesh<N>,
    /// Iterative Config
    config: UniformMultigridConfig,
    /// Base solver.
    solver: Solver,
    /// Scratch vector
    scratch: Vec<f64>,
    /// Right hand side scratch vector
    rhs: Vec<f64>,
}

impl<'m, const N: usize, Solver: LinearSolver> UniformMultigrid<'m, N, Solver> {
    /// Creates a new uniform multigrid manager.
    pub fn new(
        mesh: &'m UniformMesh<N>,
        base_config: &Solver::Config,
        grid_config: &UniformMultigridConfig,
    ) -> Self {
        let base_node_count = mesh.base_node_count();
        let node_count = mesh.node_count();

        let scratch = vec![0.0; node_count];
        let rhs = vec![0.0; node_count];

        Self {
            mesh,
            config: grid_config.clone(),
            solver: Solver::new(base_node_count, base_config),
            scratch,
            rhs,
        }
    }

    pub fn solve<O: Operator<N>>(
        &mut self,
        arena: &mut Arena,
        operator: &O,
        b: &[f64],
        x: &mut [f64],
    ) {
        self.scratch.fill(0.0);

        let irhs = self.mesh.norm(b);
        let tol = self.config.tolerance * irhs;

        if irhs <= 1e-60 {
            log::trace!("Trivial Linear Problem.");

            x.fill(0.0);
            return;
        }

        for i in 0..self.config.max_iterations {
            // Reset rhs
            self.rhs.copy_from_slice(b);

            // Cycle
            self.cycle(arena, operator, self.mesh.level_count() - 1, x);

            // Compute residual
            self.mesh
                .residual(arena, &self.rhs, operator, x, &mut self.scratch);

            self.mesh.diritchlet(operator, &mut self.scratch);

            let nres = self.mesh.norm(&self.scratch);

            log::trace!("Iteration {i}, Residual {nres:10.5e}");

            if nres <= tol {
                log::trace!("Multigrid Converged in {i} Iterations with Residual {nres:10.5e}");
                return;
            }
        }

        // Compute residual
        self.mesh
            .residual(arena, &self.rhs, operator, x, &mut self.scratch);

        self.mesh.diritchlet(operator, &mut self.scratch);

        let nres = self.mesh.norm(&self.scratch);

        log::error!(
            "Multigrid Failed to Converge. Final Residual {nres:10.5e}, Tolerance {tol:10.5e}"
        );
    }

    /// Runs a v-cycle.
    fn cycle<O: Operator<N>>(
        &mut self,
        arena: &mut Arena,
        operator: &O,
        level: usize,
        x: &mut [f64],
    ) {
        if level == 0 {
            let base_node_count = self.mesh.base_node_count();
            let block = self.mesh.level_block(level);

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

        self.mesh.diritchlet_level(level, operator, x);

        for _ in 0..self.config.presmoothing {
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

            self.mesh.diritchlet_level(level, operator, x);
        }

        // *****************************
        // Right hand side

        self.mesh
            .residual_level(level, arena, &self.rhs, operator, x, &mut self.scratch);
        self.mesh.restrict_level(level, &mut self.scratch);
        // Any diritchlet boundary should be set to zero for the residual
        self.mesh
            .diritchlet_level(level, operator, &mut self.scratch);
        // Must be some error in restrict implementation
        // self.mesh.restrict_level_full(level, &mut self.scratch);

        // Tau correction
        for i in coarse.clone() {
            self.rhs[i] = self.scratch[i];
        }

        self.mesh.restrict_level(level, x);
        self.mesh
            .apply_level(level - 1, arena, operator, x, &mut self.scratch);

        for i in coarse.clone() {
            self.rhs[i] += self.scratch[i];
        }

        // *************************
        // Recurse

        self.cycle(arena, operator, level - 1, x);

        // *************************
        // Error Correction

        self.mesh.copy_level(level, x, &mut self.scratch);
        self.mesh.restrict_level(level, &mut self.scratch);

        for i in coarse.clone() {
            self.scratch[i] = x[i] - self.scratch[i];
        }

        self.mesh.prolong_level_full(level, &mut self.scratch);

        for i in fine.clone() {
            x[i] += self.scratch[i];
        }

        self.mesh.diritchlet_level(level, operator, x);

        // *************************
        // Postsmooth

        for _ in 0..self.config.postsmoothing {
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

            self.mesh.diritchlet_level(level, operator, x);
        }
    }
}

/// A wrapper for applying the operator to the base level.
struct BaseLinearMap<'a, const N: usize, O: Operator<N>> {
    dimension: usize,
    operator: &'a O,
    block: Block<N>,
    arena: &'a mut Arena,
}

impl<'a, const N: usize, O: Operator<N>> LinearMap for BaseLinearMap<'a, N, O> {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn apply(&mut self, src: &[f64], dest: &mut [f64]) {
        self.operator.apply(self.arena, &self.block, src, dest);
        self.arena.reset();
    }

    fn callback(&self, _iteration: usize, _residual: f64, _: &[f64]) {
        // println!("Base Iteration {_iteration}, Residual {_residual}");
    }

    // fn dot(&self, src: &[f64], dest: &[f64]) -> f64 {
    //     let mut result = 0.0;

    //     let size = self.block.size();

    //     'nodes: for (i, node) in self.block.iter().enumerate() {
    //         for axis in 0..N {
    //             if O::diritchlet(axis, false) && node[axis] == 0 {
    //                 continue 'nodes;
    //             }

    //             if O::diritchlet(axis, false) && node[axis] == size[axis] - 1 {
    //                 continue 'nodes;
    //             }
    //         }

    //         result += src[i] * dest[i];
    //     }

    //     result
    // }

    fn mask(&self, mask: &mut [bool]) {
        pub struct Callback<'a, const N: usize> {
            block: &'a Block<N>,
            mask: &'a mut [bool],
        }

        impl<'a, const N: usize> BoundaryCallback<N> for Callback<'a, N> {
            fn axis<B: BoundarySet<N>>(&mut self, axis: usize, _: &B) {
                if B::PositiveBoundary::IS_DIRITCHLET {
                    let space = self.block.space.vertex_space();

                    let length = space.size()[axis];

                    for node in space.plane(axis, length - 1) {
                        let linear = space.linear_from_cartesian(node);

                        self.mask[linear] = false;
                    }
                }

                if B::NegativeBoundary::IS_DIRITCHLET {
                    let space = self.block.space.vertex_space();

                    for node in space.plane(axis, 0) {
                        let linear = space.linear_from_cartesian(node);

                        self.mask[linear] = false;
                    }
                }
            }
        }

        mask.fill(true);
        self.operator.boundary(Callback {
            block: &self.block,
            mask,
        });
    }
}

// fn compare() {
//     let domain = [0.0, 1.0];
//     let vertices = 101;
//     let spacing = (domain[1] - domain[0]) / (vertices as f64 - 1.0);

//     let solution = (0..vertices)
//         .into_iter()
//         .map(|i| {
//             let pos = i as f64 * spacing;
//             pos * (PI * pos).sin()
//         })
//         .collect::<Vec<_>>();

//     let rhs = (0..vertices)
//         .into_iter()
//         .map(|i| {
//             let pos = i as f64 * spacing;

//             2.0 * (PI * pos).cos() - PI * PI * pos * (PI * pos).sin()
//         })
//         .collect::<Vec<_>>();

//     let mut approx = vec![0.0; vertices];
// }
