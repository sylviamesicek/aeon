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

        block.dissipation::<ORDER>(0, self.rho, src, diss_r);
        block.dissipation::<ORDER>(1, self.z, src, diss_z);

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
        derivatives: DynamicSliceMut,
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

        // Dynamic

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

        self.mesh.project(arena, &psi, derivatives.psi);
        self.mesh.project(arena, &seed, derivatives.seed);
        self.mesh.project(arena, &u, derivatives.u);
        self.mesh.project(arena, &w, derivatives.w);
        self.mesh.project(arena, &x, derivatives.x);
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

        self.mesh.restrict(&mut self.dynamic.psi);
        self.mesh.restrict(&mut self.dynamic.seed);
        self.mesh.restrict(&mut self.dynamic.u);
        self.mesh.restrict(&mut self.dynamic.w);
        self.mesh.restrict(&mut self.dynamic.x);

        self.solver
            .solve(arena, self.dynamic.as_slice(), self.gauge.as_mut_slice());

        Self::compute_derivative(
            self.mesh,
            arena,
            self.dynamic.as_slice(),
            self.gauge.as_slice(),
            self.k1.as_mut_slice(),
        );

        if !EULER {
            // ********************************
            // K2

            for i in 0..self.dynamic.len() {
                self.scratch.psi[i] = self.dynamic.psi[i] + k / 2.0 * self.k1.psi[i];
                self.scratch.seed[i] = self.dynamic.seed[i] + k / 2.0 * self.k1.seed[i];
                self.scratch.u[i] = self.dynamic.u[i] + k / 2.0 * self.k1.u[i];
                self.scratch.w[i] = self.dynamic.w[i] + k / 2.0 * self.k1.w[i];
                self.scratch.x[i] = self.dynamic.x[i] + k / 2.0 * self.k1.x[i];
            }

            self.mesh.restrict(&mut self.scratch.psi);
            self.mesh.restrict(&mut self.scratch.seed);
            self.mesh.restrict(&mut self.scratch.u);
            self.mesh.restrict(&mut self.scratch.w);
            self.mesh.restrict(&mut self.scratch.x);

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

            self.mesh.restrict(&mut self.scratch.psi);
            self.mesh.restrict(&mut self.scratch.seed);
            self.mesh.restrict(&mut self.scratch.u);
            self.mesh.restrict(&mut self.scratch.w);
            self.mesh.restrict(&mut self.scratch.x);

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

            self.mesh.restrict(&mut self.scratch.psi);
            self.mesh.restrict(&mut self.scratch.seed);
            self.mesh.restrict(&mut self.scratch.u);
            self.mesh.restrict(&mut self.scratch.w);
            self.mesh.restrict(&mut self.scratch.x);

            self.solver
                .solve(arena, self.scratch.as_slice(), self.gauge.as_mut_slice());

            Self::compute_derivative(
                self.mesh,
                arena,
                self.scratch.as_slice(),
                self.gauge.as_slice(),
                self.k4.as_mut_slice(),
            );
        }

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

        if EULER {
            for i in 0..self.dynamic.len() {
                self.dynamic.psi[i] += k * self.k1.psi[i];
                self.dynamic.seed[i] += k * self.k1.seed[i];
                self.dynamic.u[i] += k * self.k1.u[i];
                self.dynamic.w[i] += k * self.k1.w[i];
                self.dynamic.x[i] += k * self.k1.x[i];
            }
        } else {
            for i in 0..self.dynamic.len() {
                self.dynamic.psi[i] += k / 6.0
                    * (self.k1.psi[i]
                        + 2.0 * self.k2.psi[i]
                        + 2.0 * self.k3.psi[i]
                        + self.k4.psi[i]);
                self.dynamic.seed[i] += k / 6.0
                    * (self.k1.seed[i]
                        + 2.0 * self.k2.seed[i]
                        + 2.0 * self.k3.seed[i]
                        + self.k4.seed[i]);
                self.dynamic.u[i] += k / 6.0
                    * (self.k1.u[i] + 2.0 * self.k2.u[i] + 2.0 * self.k3.u[i] + self.k4.u[i]);
                self.dynamic.w[i] += k / 6.0
                    * (self.k1.w[i] + 2.0 * self.k2.w[i] + 2.0 * self.k3.w[i] + self.k4.w[i]);
                self.dynamic.x[i] += k / 6.0
                    * (self.k1.x[i] + 2.0 * self.k2.x[i] + 2.0 * self.k3.x[i] + self.k4.x[i]);
            }
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
    arena: &mut Arena,
    dynamic: DynamicSlice,
    gauge: GaugeSlice,
    derivatives: DynamicSlice,
    residuals: GaugeSlice,
    constraint: &[f64],
) {
    // Store debugging variables

    let mut lapse_r = vec![0.0; mesh.node_count()];
    let mut lapse_z = vec![0.0; mesh.node_count()];
    let mut lapse_rr = vec![0.0; mesh.node_count()];
    let mut lapse_zz = vec![0.0; mesh.node_count()];
    let mut lapse_rz = vec![0.0; mesh.node_count()];

    mesh.apply(arena, &LapseDR, gauge.lapse, &mut lapse_r);
    mesh.apply(arena, &LapseDZ, gauge.lapse, &mut lapse_z);
    mesh.apply(arena, &LapseDRR, gauge.lapse, &mut lapse_rr);
    mesh.apply(arena, &LapseDZZ, gauge.lapse, &mut lapse_zz);
    mesh.apply(arena, &LapseDRZ, gauge.lapse, &mut lapse_rz);

    let mut psi_r = vec![0.0; mesh.node_count()];
    let mut psi_z = vec![0.0; mesh.node_count()];
    let mut psi_rr = vec![0.0; mesh.node_count()];
    let mut psi_zz = vec![0.0; mesh.node_count()];
    let mut psi_rz = vec![0.0; mesh.node_count()];

    mesh.apply(arena, &PsiDR, dynamic.psi, &mut psi_r);
    mesh.apply(arena, &PsiDZ, dynamic.psi, &mut psi_z);
    mesh.apply(arena, &PsiDRR, dynamic.psi, &mut psi_rr);
    mesh.apply(arena, &PsiDZZ, dynamic.psi, &mut psi_zz);
    mesh.apply(arena, &PsiDRZ, dynamic.psi, &mut psi_rz);

    let mut seed_r = vec![0.0; mesh.node_count()];
    let mut seed_z = vec![0.0; mesh.node_count()];
    let mut seed_rr = vec![0.0; mesh.node_count()];
    let mut seed_zz = vec![0.0; mesh.node_count()];

    mesh.apply(arena, &SeedDR, dynamic.seed, &mut seed_r);
    mesh.apply(arena, &SeedDZ, dynamic.seed, &mut seed_z);
    mesh.apply(arena, &SeedDRR, dynamic.seed, &mut seed_rr);
    mesh.apply(arena, &SeedDZZ, dynamic.seed, &mut seed_zz);

    let mut shiftr_r = vec![0.0; mesh.node_count()];
    let mut shiftr_z = vec![0.0; mesh.node_count()];

    mesh.apply(arena, &ShiftRDR, gauge.shiftr, &mut shiftr_r);
    mesh.apply(arena, &ShiftRDZ, gauge.shiftr, &mut shiftr_z);

    let mut shiftz_r = vec![0.0; mesh.node_count()];
    let mut shiftz_z = vec![0.0; mesh.node_count()];

    mesh.apply(arena, &ShiftZDR, gauge.shiftz, &mut shiftz_r);
    mesh.apply(arena, &ShiftZDZ, gauge.shiftz, &mut shiftz_z);

    let mut w_r = vec![0.0; mesh.node_count()];
    let mut w_z = vec![0.0; mesh.node_count()];

    mesh.apply(arena, &WDR, dynamic.w, &mut w_r);
    mesh.apply(arena, &WDZ, dynamic.w, &mut w_z);

    let mut u_r = vec![0.0; mesh.node_count()];
    let mut u_z = vec![0.0; mesh.node_count()];

    mesh.apply(arena, &UDR, dynamic.u, &mut u_r);
    mesh.apply(arena, &UDZ, dynamic.u, &mut u_z);

    let mut x_r = vec![0.0; mesh.node_count()];
    let mut x_z = vec![0.0; mesh.node_count()];

    mesh.apply(arena, &XDR, dynamic.x, &mut x_r);
    mesh.apply(arena, &XDZ, dynamic.x, &mut x_z);

    let title = format!("evolution{step}");
    let file_path = PathBuf::from(format!("output/{title}.vtu"));

    log::trace!("Saving to {}.vtu", title);

    let mut output = Model::new(mesh.clone());

    output.attach_debug_field("psi", dynamic.psi.to_vec());
    output.attach_debug_field("seed", dynamic.seed.to_vec());
    output.attach_debug_field("u", dynamic.u.to_vec());
    output.attach_debug_field("x", dynamic.x.to_vec());
    output.attach_debug_field("w", dynamic.w.to_vec());

    output.attach_debug_field("lapse", gauge.lapse.to_vec());
    output.attach_debug_field("shiftr", gauge.shiftr.to_vec());
    output.attach_debug_field("shiftz", gauge.shiftz.to_vec());

    output.attach_debug_field("psi_deriv", derivatives.psi.to_vec());
    output.attach_debug_field("seed_deriv", derivatives.seed.to_vec());
    output.attach_debug_field("u_deriv", derivatives.u.to_vec());
    output.attach_debug_field("x_deriv", derivatives.x.to_vec());
    output.attach_debug_field("w_deriv", derivatives.w.to_vec());

    output.attach_debug_field("lapse_residual", residuals.lapse.to_vec());
    output.attach_debug_field("shiftr_residual", residuals.shiftr.to_vec());
    output.attach_debug_field("shiftz_residual", residuals.shiftz.to_vec());

    output.attach_debug_field("contraint", constraint.to_vec());

    output.attach_debug_field("lapse_r", lapse_r);
    output.attach_debug_field("lapse_z", lapse_z);
    output.attach_debug_field("lapse_rr", lapse_rr);
    output.attach_debug_field("lapse_zz", lapse_zz);
    output.attach_debug_field("lapse_rz", lapse_rz);

    output.attach_debug_field("psi_r", psi_r);
    output.attach_debug_field("psi_z", psi_z);
    output.attach_debug_field("psi_rr", psi_rr);
    output.attach_debug_field("psi_zz", psi_zz);
    output.attach_debug_field("psi_rz", psi_rz);

    output.attach_debug_field("seed_r", seed_r);
    output.attach_debug_field("seed_z", seed_z);
    output.attach_debug_field("seed_rr", seed_rr);
    output.attach_debug_field("seed_zz", seed_zz);

    output.attach_debug_field("shiftr_r", shiftr_r);
    output.attach_debug_field("shiftr_z", shiftr_z);

    output.attach_debug_field("shiftz_r", shiftz_r);
    output.attach_debug_field("shiftz_z", shiftz_z);

    output.attach_debug_field("w_r", w_r);
    output.attach_debug_field("w_z", w_z);

    output.attach_debug_field("u_r", u_r);
    output.attach_debug_field("u_z", u_z);

    output.attach_debug_field("x_r", x_r);
    output.attach_debug_field("x_z", x_z);

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
    let mut derivatives = dynamic_new(mesh.node_count());
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
        derivatives.as_mut_slice(),
        residuals.as_mut_slice(),
    );

    // Write output
    write_vtk_output(
        0,
        &mesh,
        &mut arena,
        dynamic.as_slice(),
        gauge.as_slice(),
        derivatives.as_slice(),
        residuals.as_slice(),
        &constraint,
    );

    let mut system = DynamicIntegrator::new(&mesh, dynamic, gauge);

    let min_spacing = mesh.min_spacing();
    let k = CFL * min_spacing;

    log::info!("Step Size {k:10.5e}");

    for i in 0..STEPS {
        let l2 = mesh.norm(&constraint) / (constraint.len() as f64).sqrt();
        let sup: f64 = constraint.iter().fold(0.0, |a, &b| a.max(b));

        log::info!(
            "Step {}, Residual (L2: {:10.5e}, Sup: {:10.5e}), Time {}",
            i,
            l2,
            sup,
            system.time,
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
            derivatives.as_mut_slice(),
            residuals.as_mut_slice(),
        );

        // Write output
        write_vtk_output(
            i + 1,
            &mesh,
            &mut arena,
            system.dynamic.as_slice(),
            system.gauge.as_slice(),
            derivatives.as_slice(),
            residuals.as_slice(),
            &constraint,
        );
    }

    log::info!("Evolved Brill Waves {STEPS} steps.");
}
