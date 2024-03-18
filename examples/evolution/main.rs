use std::path::PathBuf;

// Global imports
use aeon::{common::BoundarySet, prelude::*};
use soa_derive::StructOfArray;

// Submodules
mod bcs;
mod config;
mod eqs;

use bcs::*;
use config::*;
use eqs::*;

// **********************************
// Settings

pub struct Dissipation<'a, Rho: BoundarySet<2>, Z: BoundarySet<2>> {
    src: &'a [f64],
    rho: &'a Rho,
    z: &'a Z,
}

impl<'a, Rho: BoundarySet<2>, Z: BoundarySet<2>> Projection<2> for Dissipation<'a, Rho, Z> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        // Bizarre rust analyzer warning.
        let mut _f = dest;

        let diss_r = arena.alloc(block.len());
        let diss_z = arena.alloc(block.len());
        let src = block.auxillary(self.src);

        block.axis::<ORDER>(0).dissipation(self.rho, src, diss_r);
        block.axis::<ORDER>(1).dissipation(self.z, src, diss_z);

        for (i, _) in block.iter().enumerate() {
            _f[i] = DISS * (diss_r[i] + diss_z[i]);
        }
    }
}

#[derive(Default, StructOfArray)]
pub struct Gauge {
    lapse: f64,
    shiftr: f64,
    shiftz: f64,
}

#[derive(Default, StructOfArray)]
pub struct Dynamic {
    psi: f64,
    seed: f64,
    u: f64,
    w: f64,
    x: f64,
}

fn gauge_new(node_count: usize) -> GaugeVec {
    let mut gauge = GaugeVec::new();

    gauge.reserve(node_count);

    for _ in 0..node_count {
        gauge.push(Default::default());
    }

    gauge
}

fn dynamic_new(node_count: usize) -> DynamicVec {
    let mut dynamic = DynamicVec::new();

    dynamic.reserve(node_count);

    for _ in 0..node_count {
        dynamic.push(Default::default());
    }

    dynamic
}

pub struct InitialSolver<'m> {
    mesh: &'m UniformMesh<2>,
    multigrid: UniformMultigrid<'m, 2, BiCGStabSolver>,
    rhs: Vec<f64>,
}

impl<'m> InitialSolver<'m> {
    pub fn new(mesh: &'m UniformMesh<2>) -> Self {
        let multigrid: UniformMultigrid<'_, 2, BiCGStabSolver> = UniformMultigrid::new(
            &mesh,
            &BiCGStabConfig {
                max_iterations: 1000,
                tolerance: 10e-12,
            },
            &UniformMultigridConfig {
                max_iterations: 100,
                tolerance: 10e-12,
                presmoothing: 5,
                postsmoothing: 5,
            },
        );

        Self {
            mesh,
            multigrid,
            rhs: vec![0.0; mesh.node_count()],
        }
    }

    pub fn solve(
        self: &mut Self,
        arena: &mut Arena,
        dynamic: DynamicSliceMut,
        gauge: GaugeSliceMut,
    ) {
        self.mesh.project(arena, &InitialSeed, dynamic.seed);

        let psi_rhs = InitialPsiRhs { seed: dynamic.seed };
        self.mesh.project(arena, &psi_rhs, &mut self.rhs);

        dynamic.psi.fill(0.0);
        let psi_op = InitialPsiOp { seed: dynamic.seed };
        self.multigrid.solve(arena, &psi_op, &self.rhs, dynamic.psi);

        dynamic.u.fill(0.0);
        dynamic.w.fill(0.0);
        dynamic.x.fill(0.0);

        gauge.lapse.fill(0.0);
        gauge.shiftr.fill(0.0);
        gauge.shiftz.fill(0.0);
    }
}

pub struct GaugeSolver<'m> {
    mesh: &'m UniformMesh<2>,
    multigrid: UniformMultigrid<'m, 2, BiCGStabSolver>,
    rhs: Vec<f64>,
}

impl<'m> GaugeSolver<'m> {
    pub fn new(mesh: &'m UniformMesh<2>) -> Self {
        let multigrid: UniformMultigrid<'_, 2, BiCGStabSolver> = UniformMultigrid::new(
            &mesh,
            &BiCGStabConfig {
                max_iterations: 10000,
                tolerance: 10e-12,
            },
            &UniformMultigridConfig {
                max_iterations: 100,
                tolerance: 10e-12,
                presmoothing: 5,
                postsmoothing: 5,
            },
        );

        Self {
            mesh,
            multigrid,
            rhs: vec![0.0; mesh.node_count()],
        }
    }

    pub fn solve(self: &mut Self, arena: &mut Arena, dynamic: DynamicSlice, gauge: GaugeSliceMut) {
        // **********************
        // Lapse

        let lapse_rhs = LapseRhs {
            psi: dynamic.psi,
            seed: dynamic.seed,
            u: dynamic.u,
            w: dynamic.w,
            x: dynamic.x,
        };
        self.mesh.project(arena, &lapse_rhs, &mut self.rhs);

        // Memset 0
        gauge.lapse.fill(0.0);

        let lapse_op = LapseOp {
            psi: dynamic.psi,
            seed: dynamic.seed,
            u: dynamic.u,
            w: dynamic.w,
            x: dynamic.x,
        };
        self.multigrid
            .solve(arena, &lapse_op, &self.rhs, gauge.lapse);

        // **********************
        // ShiftR

        let shiftr_rhs = ShiftRRhs {
            lapse: gauge.lapse,
            u: dynamic.u,
            x: dynamic.x,
        };
        self.mesh.project(arena, &shiftr_rhs, &mut self.rhs);

        gauge.shiftr.fill(0.0);

        self.multigrid
            .solve(arena, &ShiftROp {}, &self.rhs, gauge.shiftr);

        // **********************
        // ShiftZ

        let shiftz_rhs = ShiftZRhs {
            lapse: gauge.lapse,
            u: dynamic.u,
            x: dynamic.x,
        };

        self.mesh.project(arena, &shiftz_rhs, &mut self.rhs);

        gauge.shiftz.fill(0.0);

        self.multigrid
            .solve(arena, &ShiftZOp {}, &self.rhs, gauge.shiftz);
    }
}

pub struct ResidualSolver<'m> {
    mesh: &'m UniformMesh<2>,
    rhs: Vec<f64>,
}

impl<'m> ResidualSolver<'m> {
    pub fn new(mesh: &'m UniformMesh<2>) -> Self {
        Self {
            mesh,
            rhs: vec![0.0; mesh.node_count()],
        }
    }

    pub fn solve(
        self: &mut Self,
        arena: &mut Arena,
        dynamic: DynamicSlice,
        gauge: GaugeSlice,
        residuals: GaugeSliceMut,
    ) {
        // **********************
        // Lapse

        let lapse_rhs = LapseRhs {
            psi: dynamic.psi,
            seed: dynamic.seed,
            u: dynamic.u,
            w: dynamic.w,
            x: dynamic.x,
        };
        self.mesh.project(arena, &lapse_rhs, &mut self.rhs);

        let lapse_op = LapseOp {
            psi: dynamic.psi,
            seed: dynamic.seed,
            u: dynamic.u,
            w: dynamic.w,
            x: dynamic.x,
        };
        self.mesh
            .residual(arena, &self.rhs, &lapse_op, gauge.lapse, residuals.lapse);

        // **********************
        // ShiftR

        let shiftr_rhs = ShiftRRhs {
            lapse: gauge.lapse,
            u: dynamic.u,
            x: dynamic.x,
        };
        self.mesh.project(arena, &shiftr_rhs, &mut self.rhs);
        self.mesh
            .residual(arena, &self.rhs, &ShiftROp, gauge.shiftr, residuals.shiftr);

        // **********************
        // ShiftZ

        let shiftz_rhs = ShiftZRhs {
            lapse: gauge.lapse,
            u: dynamic.u,
            x: dynamic.x,
        };
        self.mesh.project(arena, &shiftz_rhs, &mut self.rhs);
        self.mesh
            .residual(arena, &self.rhs, &ShiftZOp, gauge.shiftz, residuals.shiftz);
    }
}

pub struct DynamicIntegrator<'a> {
    mesh: &'a UniformMesh<2>,
    solver: GaugeSolver<'a>,

    gauge: GaugeVec,
    dynamic: DynamicVec,

    scratch: DynamicVec,

    k1: DynamicVec,
    k2: DynamicVec,
    k3: DynamicVec,
    k4: DynamicVec,

    time: f64,
}

impl<'a> DynamicIntegrator<'a> {
    pub fn new(mesh: &'a UniformMesh<2>, dynamic: DynamicVec, gauge: GaugeVec) -> Self {
        let node_count = mesh.node_count();

        Self {
            mesh,
            solver: GaugeSolver::new(mesh),

            gauge,
            dynamic,

            scratch: dynamic_new(node_count),

            k1: dynamic_new(node_count),
            k2: dynamic_new(node_count),
            k3: dynamic_new(node_count),
            k4: dynamic_new(node_count),

            time: 0.0,
        }
    }

    pub fn step(self: &mut Self, arena: &mut Arena, k: f64) {
        // ************************
        // K1

        self.solver
            .solve(arena, self.dynamic.as_slice(), self.gauge.as_mut_slice());

        Self::compute_derivative(
            self.mesh,
            arena,
            self.dynamic.as_slice(),
            self.gauge.as_slice(),
            self.k1.as_mut_slice(),
        );

        // ********************************
        // K2

        for i in 0..self.dynamic.len() {
            self.scratch.psi[i] = self.dynamic.psi[i] + k / 2.0 * self.k1.psi[i];
            self.scratch.seed[i] = self.dynamic.seed[i] + k / 2.0 * self.k1.seed[i];
            self.scratch.u[i] = self.dynamic.u[i] + k / 2.0 * self.k1.u[i];
            self.scratch.w[i] = self.dynamic.w[i] + k / 2.0 * self.k1.w[i];
            self.scratch.x[i] = self.dynamic.x[i] + k / 2.0 * self.k1.x[i];
        }

        self.solver
            .solve(arena, self.scratch.as_slice(), self.gauge.as_mut_slice());

        Self::compute_derivative(
            self.mesh,
            arena,
            self.scratch.as_slice(),
            self.gauge.as_slice(),
            self.k2.as_mut_slice(),
        );

        // **************************************
        // K3

        for i in 0..self.dynamic.len() {
            self.scratch.psi[i] = self.dynamic.psi[i] + k / 2.0 * self.k2.psi[i];
            self.scratch.seed[i] = self.dynamic.seed[i] + k / 2.0 * self.k2.seed[i];
            self.scratch.u[i] = self.dynamic.u[i] + k / 2.0 * self.k2.u[i];
            self.scratch.w[i] = self.dynamic.w[i] + k / 2.0 * self.k2.w[i];
            self.scratch.x[i] = self.dynamic.x[i] + k / 2.0 * self.k2.x[i];
        }

        self.solver
            .solve(arena, self.scratch.as_slice(), self.gauge.as_mut_slice());

        Self::compute_derivative(
            self.mesh,
            arena,
            self.scratch.as_slice(),
            self.gauge.as_slice(),
            self.k3.as_mut_slice(),
        );

        // ****************************************
        // K4

        for i in 0..self.dynamic.len() {
            self.scratch.psi[i] = self.dynamic.psi[i] + k * self.k3.psi[i];
            self.scratch.seed[i] = self.dynamic.seed[i] + k * self.k3.seed[i];
            self.scratch.u[i] = self.dynamic.u[i] + k * self.k3.u[i];
            self.scratch.w[i] = self.dynamic.w[i] + k * self.k3.w[i];
            self.scratch.x[i] = self.dynamic.x[i] + k * self.k3.x[i];
        }

        self.solver
            .solve(arena, self.scratch.as_slice(), self.gauge.as_mut_slice());

        Self::compute_derivative(
            self.mesh,
            arena,
            self.scratch.as_slice(),
            self.gauge.as_slice(),
            self.k4.as_mut_slice(),
        );

        // ******************************
        // Update

        // Dissipation
        self.mesh.project(
            arena,
            &Dissipation {
                src: &self.dynamic.psi,
                rho: &PSI_RHO,
                z: &PSI_Z,
            },
            &mut self.scratch.psi,
        );

        self.mesh.project(
            arena,
            &Dissipation {
                src: &self.dynamic.seed,
                rho: &SEED_RHO,
                z: &SEED_Z,
            },
            &mut self.scratch.seed,
        );

        self.mesh.project(
            arena,
            &Dissipation {
                src: &self.dynamic.w,
                rho: &W_RHO,
                z: &W_Z,
            },
            &mut self.scratch.w,
        );

        self.mesh.project(
            arena,
            &Dissipation {
                src: &self.dynamic.u,
                rho: &U_RHO,
                z: &U_Z,
            },
            &mut self.scratch.u,
        );

        self.mesh.project(
            arena,
            &Dissipation {
                src: &self.dynamic.x,
                rho: &X_RHO,
                z: &X_Z,
            },
            &mut self.scratch.x,
        );

        for i in 0..self.dynamic.len() {
            self.dynamic.psi[i] += k / 6.0
                * (self.k1.psi[i] + 2.0 * self.k2.psi[i] + 2.0 * self.k3.psi[i] + self.k4.psi[i]);
            self.dynamic.seed[i] += k / 6.0
                * (self.k1.seed[i]
                    + 2.0 * self.k2.seed[i]
                    + 2.0 * self.k3.seed[i]
                    + self.k4.seed[i]);
            self.dynamic.u[i] +=
                k / 6.0 * (self.k1.u[i] + 2.0 * self.k2.u[i] + 2.0 * self.k3.u[i] + self.k4.u[i]);
            self.dynamic.w[i] +=
                k / 6.0 * (self.k1.w[i] + 2.0 * self.k2.w[i] + 2.0 * self.k3.w[i] + self.k4.w[i]);
            self.dynamic.x[i] +=
                k / 6.0 * (self.k1.x[i] + 2.0 * self.k2.x[i] + 2.0 * self.k3.x[i] + self.k4.x[i]);
        }

        for i in 0..self.dynamic.len() {
            self.dynamic.psi[i] += self.scratch.psi[i];
            self.dynamic.seed[i] += self.scratch.seed[i];
            self.dynamic.u[i] += self.scratch.u[i];
            self.dynamic.w[i] += self.scratch.w[i];
            self.dynamic.x[i] += self.scratch.x[i];
        }

        self.mesh.restrict(&mut self.dynamic.psi);
        self.mesh.restrict(&mut self.dynamic.seed);
        self.mesh.restrict(&mut self.dynamic.u);
        self.mesh.restrict(&mut self.dynamic.w);
        self.mesh.restrict(&mut self.dynamic.x);

        self.solver
            .solve(arena, self.dynamic.as_slice(), self.gauge.as_mut_slice());

        self.time += k;
    }

    fn compute_derivative(
        mesh: &'a UniformMesh<2>,
        arena: &mut Arena,
        dynamic: DynamicSlice,
        gauge: GaugeSlice,
        result: DynamicSliceMut,
    ) {
        let psi = PsiEvolution {
            lapse: gauge.lapse,
            shiftr: gauge.shiftr,
            shiftz: gauge.shiftz,
            psi: dynamic.psi,
            u: dynamic.u,
            w: dynamic.w,
        };

        let seed = SeedEvolution {
            lapse: gauge.lapse,
            shiftr: gauge.shiftr,
            shiftz: gauge.shiftz,
            seed: dynamic.seed,
            w: dynamic.w,
        };

        let u = UEvolution {
            lapse: gauge.lapse,
            shiftr: gauge.shiftr,
            shiftz: gauge.shiftz,
            psi: dynamic.psi,
            seed: dynamic.seed,
            u: dynamic.u,
            x: dynamic.x,
        };

        let w = WEvolution {
            lapse: gauge.lapse,
            shiftr: gauge.shiftr,
            shiftz: gauge.shiftz,
            psi: dynamic.psi,
            seed: dynamic.seed,
            w: dynamic.w,
            x: dynamic.x,
        };

        let x = XEvolution {
            lapse: gauge.lapse,
            shiftr: gauge.shiftr,
            shiftz: gauge.shiftz,
            psi: dynamic.psi,
            seed: dynamic.seed,
            u: dynamic.u,
            x: dynamic.x,
        };

        mesh.project(arena, &psi, result.psi);
        mesh.project(arena, &seed, result.seed);
        mesh.project(arena, &u, result.u);
        mesh.project(arena, &w, result.w);
        mesh.project(arena, &x, result.x);
    }
}

fn write_vtk_output(
    step: usize,
    mesh: &UniformMesh<2>,
    dynamic: DynamicSlice,
    gauge: GaugeSlice,
    residuals: GaugeSlice,
    constraint: &[f64],
) {
    let title = format!("evolution{step}");
    let file_path = PathBuf::from(format!("output/{title}.vtu"));

    let mut output = DataOut::new(mesh);

    output.attrib_scalar("psi", dynamic.psi);
    output.attrib_scalar("seed", dynamic.seed);
    output.attrib_scalar("u", dynamic.u);
    output.attrib_scalar("x", dynamic.x);
    output.attrib_scalar("w", dynamic.w);

    output.attrib_scalar("lapse", gauge.lapse);
    output.attrib_scalar("shiftr", gauge.shiftr);
    output.attrib_scalar("shiftz", gauge.shiftz);

    output.attrib_scalar("lapse_residual", residuals.lapse);
    output.attrib_scalar("shiftr_residual", residuals.shiftr);
    output.attrib_scalar("shiftz_residual", residuals.shiftz);

    output.attrib_scalar("contraint", constraint);

    output.export_vtk(&title, file_path).unwrap()
}

pub fn main() {
    env_logger::builder()
        .format_timestamp(None)
        .filter_level(log::LevelFilter::max())
        .init();

    // Scratch allocator
    let mut arena = Arena::new();

    let mesh = UniformMesh::new(
        Rectangle {
            size: [RADIUS, RADIUS],
            origin: [0.0, 0.0],
        },
        [8, 8],
        LEVELS,
    );

    log::info!("Node Count: {}", mesh.node_count());

    let mut dynamic = dynamic_new(mesh.node_count());
    let mut gauge = gauge_new(mesh.node_count());
    let mut residuals = gauge_new(mesh.node_count());
    let mut constraint = vec![0.0; mesh.node_count()];

    {
        let mut initial_solver = InitialSolver::new(&mesh);
        initial_solver.solve(&mut arena, dynamic.as_mut_slice(), gauge.as_mut_slice());
    }

    {
        let hamiltonian = Hamiltonian {
            psi: &dynamic.psi,
            seed: &dynamic.seed,
            u: &dynamic.u,
            x: &dynamic.x,
            w: &dynamic.w,
        };

        mesh.project(&mut arena, &hamiltonian, &mut constraint);
    }

    // Build residual solver
    let mut residual_solver = ResidualSolver::new(&mesh);
    // Compute residuals
    residual_solver.solve(
        &mut arena,
        dynamic.as_slice(),
        gauge.as_slice(),
        residuals.as_mut_slice(),
    );

    // Write output
    write_vtk_output(
        0,
        &mesh,
        dynamic.as_slice(),
        gauge.as_slice(),
        residuals.as_slice(),
        &constraint,
    );

    let mut system = DynamicIntegrator::new(&mesh, dynamic, gauge);

    let min_spacing = mesh.min_spacing();
    let k = CFL * min_spacing;

    log::info!("Step Size {k}");

    for i in 0..STEPS {
        log::info!(
            "Step {}, Residual {}, Time {}",
            i,
            mesh.norm(&constraint) / (constraint.len() as f64).sqrt(),
            k * i as f64,
        );
        // Step
        system.step(&mut arena, k);
        // Solve constraint
        let hamiltonian = Hamiltonian {
            psi: &system.dynamic.psi,
            seed: &system.dynamic.seed,
            u: &system.dynamic.u,
            x: &system.dynamic.x,
            w: &system.dynamic.w,
        };

        mesh.project(&mut arena, &hamiltonian, &mut constraint);

        residual_solver.solve(
            &mut arena,
            system.dynamic.as_slice(),
            system.gauge.as_slice(),
            residuals.as_mut_slice(),
        );

        // Write output
        write_vtk_output(
            i + 1,
            &mesh,
            system.dynamic.as_slice(),
            system.gauge.as_slice(),
            residuals.as_slice(),
            &constraint,
        );
    }

    // multigrid.solve(&Laplacian { bump: Bump::new() }, &rhs, &mut solution);

    log::info!("Evolved Brill Waves {STEPS} steps.");
}
