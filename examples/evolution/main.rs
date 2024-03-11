use std::path::PathBuf;

// Global imports
use aeon::prelude::*;
use soa_derive::StructOfArray;
use vtkio::model::*;

// Submodules
mod bcs;

pub use bcs::*;

// **********************************
// Settings

const RADIUS: f64 = 10.0;

fn is_approximately_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < 10e-10
}

struct InitialSeed;

impl Projection<2> for InitialSeed {
    fn evaluate(self: &Self, _: &Arena, block: &Block<2>, dest: &mut [f64]) {
        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);

            let rho = position[0];
            let z = position[1];

            let rho2 = rho * rho;
            let z2 = z * z;
            let sigma2 = 1.0;

            dest[i] = rho * (-(rho2 + z2) / sigma2).exp();
        }
    }
}

pub struct InitialPsiOp<'a> {
    seed: &'a [f64],
}

impl<'a> Operator<2> for InitialPsiOp<'a> {
    fn apply(self: &Self, arena: &Arena, block: &Block<2>, psi: &[f64], dest: &mut [f64]) {
        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());

        block
            .axis::<4>(0)
            .second_derivative(&PSI_INITIAL_RHO, psi, psi_rr);
        block
            .axis::<4>(1)
            .second_derivative(&PSI_INITIAL_Z, psi, psi_zz);
        block.axis::<4>(0).derivative(&PSI_INITIAL_RHO, psi, psi_r);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block.axis::<4>(0).second_derivative(&SEED_RHO, seed, s_rr);
        block.axis::<4>(1).second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<4>(0).derivative(&SEED_RHO, seed, s_r);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            let mut term1 = psi_rr[i] + psi_zz[i];

            if is_approximately_equal(rho, 0.0) {
                term1 += psi_rr[i];
            } else {
                term1 += psi_r[i] / rho;
            }

            let term2 = psi[i] / 4.0 * (rho * s_rr[i] + 2.0 * s_r[i] + rho * s_zz[i]);

            dest[i] = term1 + term2;
        }
    }

    fn apply_diag(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());

        block
            .axis::<4>(0)
            .second_derivative_diag(&PSI_INITIAL_RHO, psi_rr);
        block
            .axis::<4>(1)
            .second_derivative_diag(&PSI_INITIAL_Z, psi_zz);
        block.axis::<4>(0).derivative_diag(&PSI_INITIAL_RHO, psi_r);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block.axis::<4>(0).second_derivative(&SEED_RHO, seed, s_rr);
        block.axis::<4>(1).second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<4>(0).derivative(&SEED_RHO, seed, s_r);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            let mut term1 = psi_rr[i] + psi_zz[i];

            if is_approximately_equal(rho, 0.0) {
                term1 += psi_rr[i];
            } else {
                term1 += psi_r[i] / rho;
            }

            let term2 = 1.0 / 4.0 * (rho * s_rr[i] + 2.0 * s_r[i] + rho * s_zz[i]);

            dest[i] = term1 + term2;
        }
    }
}

pub struct InitialPsiRhs<'a> {
    seed: &'a [f64],
}

impl<'a> Projection<2> for InitialPsiRhs<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block.axis::<4>(0).second_derivative(&SEED_RHO, seed, s_rr);
        block.axis::<4>(1).second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<4>(0).derivative(&SEED_RHO, seed, s_r);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            dest[i] = -1.0 / 4.0 * (rho * s_rr[i] + 2.0 * s_r[i] + rho * s_zz[i]);
        }
    }
}

pub struct LapseOp<'a> {
    psi: &'a [f64],
    seed: &'a [f64],
    u: &'a [f64],
    w: &'a [f64],
    x: &'a [f64],
}

impl<'a> Operator<2> for LapseOp<'a> {
    fn apply(self: &Self, arena: &Arena, block: &Block<2>, lapse: &[f64], dest: &mut [f64]) {
        let lapse_rr = arena.alloc::<f64>(block.len());
        let lapse_zz = arena.alloc::<f64>(block.len());
        let lapse_r = arena.alloc::<f64>(block.len());
        let lapse_z = arena.alloc::<f64>(block.len());

        block
            .axis::<4>(0)
            .second_derivative(&LAPSE_RHO, lapse, lapse_rr);
        block
            .axis::<4>(1)
            .second_derivative(&LAPSE_Z, lapse, lapse_zz);
        block.axis::<4>(0).derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<4>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());
        let psi = block.auxillary(self.psi);

        block.axis::<4>(0).derivative(&PSI_RHO, self.psi, psi_r);
        block.axis::<4>(1).derivative(&PSI_Z, self.psi, psi_z);

        let seed = block.auxillary(self.seed);
        let w = block.auxillary(self.w);
        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            let psi = psi[i] + 1.0;
            let seed = seed[i];
            let w = w[i];
            let u = u[i];
            let x = x[i];

            let mut term1 = lapse_rr[i] + lapse_zz[i];

            if is_approximately_equal(rho, 0.0) {
                term1 += lapse_rr[i];
            } else {
                term1 += lapse_r[i] / rho;
            }

            let term2 = 2.0 / psi * (psi_r[i] * lapse_r[i] + psi_z[i] * lapse_z[i]);

            let scale = -lapse[i] * (2.0 * rho * seed).exp() * psi * psi * psi * psi;

            let term3 = 2.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            let term4 = 2.0 * x * x;

            dest[i] = term1 + term2 + scale * (term3 + term4);
        }
    }

    fn apply_diag(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse_rr = arena.alloc::<f64>(block.len());
        let lapse_zz = arena.alloc::<f64>(block.len());
        let lapse_r = arena.alloc::<f64>(block.len());
        let lapse_z = arena.alloc::<f64>(block.len());

        block
            .axis::<4>(0)
            .second_derivative_diag(&LAPSE_RHO, lapse_rr);
        block
            .axis::<4>(1)
            .second_derivative_diag(&LAPSE_Z, lapse_zz);
        block.axis::<4>(0).derivative_diag(&LAPSE_RHO, lapse_r);
        block.axis::<4>(1).derivative_diag(&LAPSE_Z, lapse_z);

        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());
        let psi = block.auxillary(self.psi);

        block.axis::<4>(0).derivative(&PSI_RHO, self.psi, psi_r);
        block.axis::<4>(1).derivative(&PSI_Z, self.psi, psi_z);

        let seed = block.auxillary(self.seed);
        let w = block.auxillary(self.w);
        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            let psi = psi[i] + 1.0;
            let seed = seed[i];
            let w = w[i];
            let u = u[i];
            let x = x[i];

            let mut term1 = lapse_rr[i] + lapse_zz[i];

            if is_approximately_equal(rho, 0.0) {
                term1 += lapse_rr[i];
            } else {
                term1 += lapse_r[i] / rho;
            }

            let term2 = 2.0 / psi * (psi_r[i] * lapse_r[i] + psi_z[i] * lapse_z[i]);

            let scale = -(2.0 * rho * seed).exp() * psi * psi * psi * psi;

            let term3 = 2.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            let term4 = 2.0 * x * x;

            dest[i] = term1 + term2 + scale * (term3 + term4);
        }
    }
}

pub struct LapseRhs<'a> {
    psi: &'a [f64],
    seed: &'a [f64],
    u: &'a [f64],
    w: &'a [f64],
    x: &'a [f64],
}

impl<'a> Projection<2> for LapseRhs<'a> {
    fn evaluate(self: &Self, _: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let psi = block.auxillary(self.psi);
        let seed = block.auxillary(self.seed);
        let w = block.auxillary(self.w);
        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            let psi = psi[i] + 1.0;
            let seed = seed[i];
            let w = w[i];
            let u = u[i];
            let x = x[i];

            let scale = (2.0 * rho * seed).exp() * psi * psi * psi * psi;
            let term1 = 2.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            let term2 = 2.0 * x * x;

            dest[i] = scale * (term1 + term2);
        }
    }
}

pub struct ShiftRRhs<'a> {
    lapse: &'a [f64],
    x: &'a [f64],
    u: &'a [f64],
}

impl<'a> Projection<2> for ShiftRRhs<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse = block.auxillary(self.lapse);
        let lapse_r = arena.alloc(block.len());
        let lapse_z = arena.alloc(block.len());

        block.axis::<4>(0).derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<4>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let x = block.auxillary(self.x);
        let x_z = arena.alloc(block.len());

        block.axis::<4>(1).derivative(&X_Z, x, x_z);

        let u = block.auxillary(self.u);
        let u_r = arena.alloc(block.len());

        block.axis::<4>(0).derivative(&U_RHO, u, u_r);

        for (i, _) in block.iter().enumerate() {
            let lapse = lapse[i] + 1.0;

            let term1 = 2.0 * (x[i] * lapse_z[i] + lapse * x_z[i]);
            let term2 = -u[i] * lapse_r[i] - lapse * u_r[i];

            dest[i] = term1 + term2;
        }
    }
}

pub struct ShiftZRhs<'a> {
    lapse: &'a [f64],
    x: &'a [f64],
    u: &'a [f64],
}

impl<'a> Projection<2> for ShiftZRhs<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse = block.auxillary(self.lapse);
        let lapse_r = arena.alloc(block.len());
        let lapse_z = arena.alloc(block.len());

        block.axis::<4>(0).derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<4>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let x = block.auxillary(self.x);
        let x_r = arena.alloc(block.len());

        block.axis::<4>(0).derivative(&X_RHO, x, x_r);

        let u = block.auxillary(self.u);
        let u_z = arena.alloc(block.len());

        block.axis::<4>(1).derivative(&U_Z, u, u_z);

        for (i, _) in block.iter().enumerate() {
            let lapse = lapse[i] + 1.0;

            let term1 = 2.0 * (x[i] * lapse_r[i] + lapse * x_r[i]);
            let term2 = u[i] * lapse_z[i] - lapse * u_z[i];

            dest[i] = term1 + term2;
        }
    }
}

pub struct ShiftROp;

impl Operator<2> for ShiftROp {
    fn apply(self: &Self, arena: &Arena, block: &Block<2>, shiftr: &[f64], dest: &mut [f64]) {
        let shiftr_rr = arena.alloc::<f64>(block.len());
        let shiftr_zz = arena.alloc::<f64>(block.len());

        block
            .axis::<4>(0)
            .second_derivative(&SHIFTR_RHO, shiftr, shiftr_rr);
        block
            .axis::<4>(1)
            .second_derivative(&SHIFTR_Z, shiftr, shiftr_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = shiftr_rr[i] + shiftr_zz[i];
        }
    }

    fn apply_diag(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let shiftr_rr = arena.alloc::<f64>(block.len());
        let shiftr_zz = arena.alloc::<f64>(block.len());

        block
            .axis::<4>(0)
            .second_derivative_diag(&SHIFTR_RHO, shiftr_rr);
        block
            .axis::<4>(1)
            .second_derivative_diag(&SHIFTR_Z, shiftr_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = shiftr_rr[i] + shiftr_zz[i];
        }
    }
}

pub struct ShiftZOp;

impl Operator<2> for ShiftZOp {
    fn apply(self: &Self, arena: &Arena, block: &Block<2>, shiftz: &[f64], dest: &mut [f64]) {
        let shiftz_rr = arena.alloc::<f64>(block.len());
        let shiftz_zz = arena.alloc::<f64>(block.len());

        block
            .axis::<4>(0)
            .second_derivative(&SHIFTZ_RHO, shiftz, shiftz_rr);
        block
            .axis::<4>(1)
            .second_derivative(&SHIFTZ_Z, shiftz, shiftz_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = shiftz_rr[i] + shiftz_zz[i];
        }
    }

    fn apply_diag(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let shiftz_rr = arena.alloc::<f64>(block.len());
        let shiftz_zz = arena.alloc::<f64>(block.len());

        block
            .axis::<4>(0)
            .second_derivative_diag(&SHIFTZ_RHO, shiftz_rr);
        block
            .axis::<4>(1)
            .second_derivative_diag(&SHIFTZ_Z, shiftz_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = shiftz_rr[i] + shiftz_zz[i];
        }
    }
}

pub struct Hamiltonian<'a> {
    psi: &'a [f64],
    seed: &'a [f64],
    u: &'a [f64],
    x: &'a [f64],
    w: &'a [f64],
}

impl<'a> Projection<2> for Hamiltonian<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block.axis::<4>(0).second_derivative(&PSI_RHO, psi, psi_rr);
        block.axis::<4>(1).second_derivative(&PSI_Z, psi, psi_zz);
        block.axis::<4>(0).derivative(&PSI_RHO, psi, psi_r);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block.axis::<4>(0).second_derivative(&SEED_RHO, seed, s_rr);
        block.axis::<4>(1).second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<4>(0).derivative(&SEED_RHO, seed, s_r);

        let w = block.auxillary(self.w);
        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            let psi: f64 = psi[i] + 1.0;

            let mut term1 = psi_rr[i] + psi_zz[i];

            if is_approximately_equal(rho, 0.0) {
                term1 += psi_rr[i];
            } else {
                term1 += psi_r[i] / rho;
            }

            let term2 = psi / 4.0 * (rho * s_rr[i] + 2.0 * s_r[i] + rho * s_zz[i]);

            let scale = psi * psi * psi * psi * psi * (2.0 * rho * seed[i]).exp() / 4.0;

            let term3 = 1.0 / 3.0 * (rho * rho * w[i] + rho * u[i] * w[i] + u[i] * u[i]);
            let term4 = x[i] * x[i];

            dest[i] = term1 + term2 + scale * (term3 + term4);
        }
    }
}

pub struct PsiEvolution<'a> {
    lapse: &'a [f64],
    shiftr: &'a [f64],
    shiftz: &'a [f64],
    psi: &'a [f64],
    u: &'a [f64],
    w: &'a [f64],
}

impl<'a> Projection<2> for PsiEvolution<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse = block.auxillary(self.lapse);
        let shiftr = block.auxillary(self.shiftr);
        let shiftz = block.auxillary(self.shiftz);

        let shiftr_r = arena.alloc::<f64>(block.len());
        block.axis::<4>(0).derivative(&SHIFTR_RHO, shiftr, shiftr_r);

        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block.axis::<4>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<4>(1).derivative(&PSI_Z, psi, psi_z);

        let u = block.auxillary(self.u);
        let w = block.auxillary(self.w);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let z = position[1];

            if is_approximately_equal(rho, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / rho * psi_r[i];
                let term2 = -psi[i] / r;

                dest[i] = term1 + term2;

                continue;
            } else if is_approximately_equal(z, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / z * psi_z[i];
                let term2 = -psi[i] / r;

                dest[i] = term1 + term2;

                continue;
            }

            let psi = psi[i] + 1.0;
            let lapse = lapse[i] + 1.0;

            let term1 = shiftr[i] * psi_r[i] + shiftz[i] * psi_z[i];
            let term2 = psi * lapse / 6.0 * (u[i] + 2.0 * rho * w[i]);

            if is_approximately_equal(rho, 0.0) {
                dest[i] = term1 + term2 + psi / 2.0 * shiftr_r[i];
            } else {
                dest[i] = term1 + term2 + psi * shiftr[i] / (2.0 * rho);
            }
        }
    }
}

pub struct SeedEvolution<'a> {
    lapse: &'a [f64],
    shiftr: &'a [f64],
    shiftz: &'a [f64],
    seed: &'a [f64],
    w: &'a [f64],
}

impl<'a> Projection<2> for SeedEvolution<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse = block.auxillary(self.lapse);
        let shiftr = block.auxillary(self.shiftr);
        let shiftz = block.auxillary(self.shiftz);

        let shiftr_r = arena.alloc::<f64>(block.len());
        block.axis::<4>(0).derivative(&SHIFTR_RHO, shiftr, shiftr_r);

        let s_r = arena.alloc::<f64>(block.len());
        let s_z = arena.alloc::<f64>(block.len());

        let s = block.auxillary(self.seed);

        block.axis::<4>(0).derivative(&SEED_RHO, s, s_r);
        block.axis::<4>(1).derivative(&SEED_Z, s, s_z);

        let w = block.auxillary(self.w);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let z = position[1];

            if is_approximately_equal(rho, 0.0) {
                dest[i] = 0.0;

                continue;
            }

            if is_approximately_equal(rho, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / rho * s_r[i];
                let term2 = -s[i] / r;

                dest[i] = term1 + term2;

                continue;
            } else if is_approximately_equal(z, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / z * s_z[i];
                let term2 = -s[i] / r;

                dest[i] = term1 + term2;

                continue;
            }

            let lapse = lapse[i] + 1.0;

            let term1 = shiftr[i] * s_r[i] + shiftz[i] * s_z[i];
            let term2 = -lapse * w[i] + shiftr[i] * s[i] / rho;
            let term3 = shiftr_r[i] / rho - shiftr[i] / (rho * rho);

            dest[i] = term1 + term2 + term3;
        }
    }
}

pub struct WEvolution<'a> {
    lapse: &'a [f64],
    shiftr: &'a [f64],
    shiftz: &'a [f64],
    psi: &'a [f64],
    seed: &'a [f64],
    x: &'a [f64],
    w: &'a [f64],
}

impl<'a> Projection<2> for WEvolution<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse_rr = arena.alloc(block.len());
        let lapse_zz = arena.alloc(block.len());
        let lapse_r = arena.alloc(block.len());
        let lapse_z = arena.alloc(block.len());
        let lapse = block.auxillary(self.lapse);

        block
            .axis::<4>(0)
            .second_derivative(&LAPSE_RHO, lapse, lapse_rr);
        block
            .axis::<4>(1)
            .second_derivative(&LAPSE_Z, lapse, lapse_zz);
        block.axis::<4>(0).derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<4>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let shiftr_z = arena.alloc(block.len());
        let shiftr = block.auxillary(self.shiftr);

        block.axis::<4>(1).derivative(&SHIFTR_Z, shiftr, shiftr_z);

        let shiftz_r = arena.alloc(block.len());
        let shiftz = block.auxillary(self.shiftz);

        block.axis::<4>(0).derivative(&SHIFTZ_RHO, shiftz, shiftz_r);

        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block.axis::<4>(0).second_derivative(&PSI_RHO, psi, psi_rr);
        block.axis::<4>(1).second_derivative(&PSI_Z, psi, psi_zz);
        block.axis::<4>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<4>(1).derivative(&PSI_Z, psi, psi_z);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());
        let s_z = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block.axis::<4>(0).second_derivative(&SEED_RHO, seed, s_rr);
        block.axis::<4>(1).second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<4>(0).derivative(&SEED_RHO, seed, s_r);
        block.axis::<4>(1).derivative(&SEED_Z, seed, s_z);

        let w = block.auxillary(self.w);
        let x = block.auxillary(self.x);

        let w_r = arena.alloc(block.len());
        let w_z = arena.alloc(block.len());

        block.axis::<4>(0).derivative(&W_RHO, w, w_r);
        block.axis::<4>(1).derivative(&W_Z, w, w_z);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let z = position[1];

            if is_approximately_equal(rho, 0.0) {
                dest[i] = 0.0;

                continue;
            }

            if is_approximately_equal(rho, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / rho * w_r[i];
                let term2 = -w[i] / r;

                dest[i] = term1 + term2;

                continue;
            } else if is_approximately_equal(z, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / z * w_z[i];
                let term2 = -w[i] / r;

                dest[i] = term1 + term2;

                continue;
            }

            let psi = psi[i] + 1.0;
            let lapse = lapse[i] + 1.0;

            let term1 = shiftr[i] * w_r[i] + shiftz[i] * w_z[i] + shiftr[i] * w[i] / rho;
            let term2 = x[i] / rho * (shiftz_r[i] - shiftr_z[i]);

            let scale1 = (-2.0 * rho * seed[i]).exp() / (psi * psi * psi * psi);
            let term3 = lapse_r[i] / rho * (rho * s_r[i] + seed[i] + 4.0 * psi_r[i] / psi);
            let term4 = lapse_r[i] / (rho * rho) - lapse_rr[i] / rho - lapse_z[i] * s_z[i];

            let scale2 = lapse * scale1;

            let term5 = 2.0 * psi_r[i] / (rho * psi)
                * (rho * s_r[i] + seed[i] + 3.0 * psi_r[i] / psi)
                - 2.0 * psi_z[i] / psi * s_z[i];

            let term6 = -s_rr[i] - s_r[i] / rho - s_zz[i] + seed[i] / (rho * rho);
            let term7 = 2.0 / psi * (psi_r[i] / (rho * rho) - psi_rr[i] / rho);

            dest[i] = term1 + term2 + scale1 * (term3 + term4) + scale2 * (term5 + term6 + term7);
        }
    }
}

pub struct UEvolution<'a> {
    lapse: &'a [f64],
    shiftr: &'a [f64],
    shiftz: &'a [f64],
    psi: &'a [f64],
    seed: &'a [f64],
    x: &'a [f64],
    u: &'a [f64],
}

impl<'a> Projection<2> for UEvolution<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse_rr = arena.alloc(block.len());
        let lapse_zz = arena.alloc(block.len());
        let lapse_r = arena.alloc(block.len());
        let lapse_z = arena.alloc(block.len());
        let lapse = block.auxillary(self.lapse);

        block
            .axis::<4>(0)
            .second_derivative(&LAPSE_RHO, lapse, lapse_rr);
        block
            .axis::<4>(1)
            .second_derivative(&LAPSE_Z, lapse, lapse_zz);
        block.axis::<4>(0).derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<4>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let shiftr_z = arena.alloc(block.len());
        let shiftr = block.auxillary(self.shiftr);

        block.axis::<4>(1).derivative(&SHIFTR_Z, shiftr, shiftr_z);

        let shiftz_r = arena.alloc(block.len());
        let shiftz = block.auxillary(self.shiftz);

        block.axis::<4>(0).derivative(&SHIFTZ_RHO, shiftz, shiftz_r);

        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block.axis::<4>(0).second_derivative(&PSI_RHO, psi, psi_rr);
        block.axis::<4>(1).second_derivative(&PSI_Z, psi, psi_zz);
        block.axis::<4>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<4>(1).derivative(&PSI_Z, psi, psi_z);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());
        let s_z = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block.axis::<4>(0).second_derivative(&SEED_RHO, seed, s_rr);
        block.axis::<4>(1).second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<4>(0).derivative(&SEED_RHO, seed, s_r);
        block.axis::<4>(1).derivative(&SEED_Z, seed, s_z);

        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        let u_r = arena.alloc(block.len());
        let u_z = arena.alloc(block.len());

        block.axis::<4>(0).derivative(&U_RHO, u, u_r);
        block.axis::<4>(1).derivative(&U_Z, u, u_z);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let z = position[1];

            if is_approximately_equal(rho, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / rho * u_r[i];
                let term2 = -u[i] / r;

                dest[i] = term1 + term2;

                continue;
            } else if is_approximately_equal(z, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / z * u_z[i];
                let term2 = -u[i] / r;

                dest[i] = term1 + term2;

                continue;
            }

            let psi = psi[i] + 1.0;
            let lapse = lapse[i] + 1.0;

            let term1 = shiftr[i] * u_r[i] + shiftz[i] * u_z[i];
            let term2 = 2.0 * x[i] * (shiftr_z[i] - shiftz_r[i]);

            let scale1 = (-2.0 * rho * seed[i]) / (psi * psi * psi * psi);
            let term3 = 2.0 * lapse_z[i] * (2.0 * psi_z[i] / psi + rho * s_z[i]);
            let term4 = -2.0 * lapse_r[i] * (rho * s_r[i] + seed[i] + 2.0 * psi_r[i] / psi);
            let term5 = lapse_rr[i] - lapse_zz[i];

            let scale2 = 2.0 * lapse * scale1;
            let term6 = psi_z[i] / psi * (2.0 * rho * s_z[i] + 3.0 * psi_z[i] / psi);
            let term7 =
                -psi_r[i] / psi * (2.0 * rho * s_r[i] + 2.0 * seed[i] + 3.0 * psi_r[i] / psi);

            let mut term8 = psi_rr[i] / psi - psi_zz[i] / psi - s_r[i];

            if is_approximately_equal(rho, 0.0) {
                term8 -= s_r[i];
            } else {
                term8 -= seed[i] / rho;
            }

            dest[i] =
                term1 + term2 + scale1 * (term3 + term4 + term5) + scale2 * (term6 + term7 + term8);
        }
    }
}

pub struct XEvolution<'a> {
    lapse: &'a [f64],
    shiftr: &'a [f64],
    shiftz: &'a [f64],
    psi: &'a [f64],
    seed: &'a [f64],
    x: &'a [f64],
    u: &'a [f64],
}

impl<'a> Projection<2> for XEvolution<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse_rr = arena.alloc(block.len());
        let lapse_zz = arena.alloc(block.len());
        let lapse_rz = arena.alloc(block.len());
        let lapse_r = arena.alloc(block.len());
        let lapse_z = arena.alloc(block.len());
        let lapse = block.auxillary(self.lapse);

        block
            .axis::<4>(0)
            .second_derivative(&LAPSE_RHO, lapse, lapse_rr);
        block
            .axis::<4>(1)
            .second_derivative(&LAPSE_Z, lapse, lapse_zz);
        block.axis::<4>(0).derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<4>(1).derivative(&LAPSE_Z, lapse, lapse_z);
        block.axis::<4>(1).derivative(&LAPSE_Z, lapse_r, lapse_rz);

        let shiftr_z = arena.alloc(block.len());
        let shiftr = block.auxillary(self.shiftr);

        block.axis::<4>(1).derivative(&SHIFTR_Z, shiftr, shiftr_z);

        let shiftz_r = arena.alloc(block.len());
        let shiftz = block.auxillary(self.shiftz);

        block.axis::<4>(0).derivative(&SHIFTZ_RHO, shiftz, shiftz_r);

        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_rz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block.axis::<4>(0).second_derivative(&PSI_RHO, psi, psi_rr);
        block.axis::<4>(1).second_derivative(&PSI_Z, psi, psi_zz);
        block.axis::<4>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<4>(1).derivative(&PSI_Z, psi, psi_z);
        block.axis::<4>(1).derivative(&PSI_Z, psi_r, psi_rz);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());
        let s_z = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block.axis::<4>(0).second_derivative(&SEED_RHO, seed, s_rr);
        block.axis::<4>(1).second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<4>(0).derivative(&SEED_RHO, seed, s_r);
        block.axis::<4>(1).derivative(&SEED_Z, seed, s_z);

        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        let x_r = arena.alloc(block.len());
        let x_z = arena.alloc(block.len());

        block.axis::<4>(0).derivative(&X_RHO, x, x_r);
        block.axis::<4>(1).derivative(&X_Z, x, x_z);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let z = position[1];

            if is_approximately_equal(rho, 0.0) {
                dest[i] = 0.0;

                continue;
            }

            if is_approximately_equal(rho, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / rho * x_r[i];
                let term2 = -x[i] / r;

                dest[i] = term1 + term2;

                continue;
            } else if is_approximately_equal(z, RADIUS) {
                let r = (rho * rho + z * z).sqrt();

                let term1 = -r / z * x_z[i];
                let term2 = -x[i] / r;

                dest[i] = term1 + term2;

                continue;
            }

            let psi = psi[i] + 1.0;
            let lapse = lapse[i] + 1.0;

            let term1 = shiftr[i] * x_r[i] + shiftz[i] * x_z[i];
            let term2 = 1.0 / 2.0 * u[i] * (shiftz_r[i] - shiftr_z[i]);

            let scale1 = (-2.0 * rho * seed[i]) / (psi * psi * psi * psi);
            let term3 = lapse_r[i] * (rho * s_z[i] + 2.0 * psi_z[i] / psi);
            let term4 = lapse_z[i] * (rho * s_r[i] + seed[i] + 2.0 * psi_r[i] / psi);
            let term5 = -lapse_rz[i];

            let scale2 = lapse * scale1;
            let term6 = psi_r[i] / psi * (2.0 * rho * s_z[i] + 6.0 * psi_z[i] / psi);
            let term7 = psi_z[i] / psi * (2.0 * rho * s_r[i] + 2.0 * seed[i]);
            let term8 = s_z[i] - 2.0 * psi_rz[i] / psi;

            dest[i] =
                term1 + term2 + scale1 * (term3 + term4 + term5) + scale2 * (term6 + term7 + term8);
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
            100,
            10e-12,
            5,
            5,
            &BiCGStabConfig {
                max_iterations: 1000,
                tolerance: 10e-12,
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
        let initial_seed = InitialSeed {};
        self.mesh.project(arena, &initial_seed, dynamic.seed);

        let psi_rhs = InitialPsiRhs { seed: dynamic.seed };
        self.mesh.project(arena, &psi_rhs, &mut self.rhs);

        println!("Solving Initial psi");

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
            100,
            10e-12,
            5,
            5,
            &BiCGStabConfig {
                max_iterations: 10000,
                tolerance: 10e-12,
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
    constraint: &[f64],
) {
    let title = format!("evolution{step}");

    let range = mesh.level_node_range(mesh.level_count() - 1);

    let psi = &dynamic.psi[range.clone()];
    let seed = &dynamic.seed[range.clone()];
    let u = &dynamic.u[range.clone()];
    let w = &dynamic.w[range.clone()];
    let x = &dynamic.x[range.clone()];

    let lapse = &gauge.lapse[range.clone()];
    let shiftr = &gauge.lapse[range.clone()];
    let shiftz = &gauge.lapse[range.clone()];

    let constraint = &constraint[range.clone()];

    let node_space = mesh.level_node_space(mesh.level_count() - 1);

    let cell_space = node_space.cell_space();
    let vertex_space = node_space.vertex_space();

    let cell_total = node_space.cell_space().len();

    // Generate Cells

    let mut connectivity = Vec::new();
    let mut offsets = Vec::new();

    for cell in cell_space.iter() {
        let v1 = vertex_space.linear_from_cartesian(cell);
        let v2 = vertex_space.linear_from_cartesian([cell[0], cell[1] + 1]);
        let v3 = vertex_space.linear_from_cartesian([cell[0] + 1, cell[1] + 1]);
        let v4 = vertex_space.linear_from_cartesian([cell[0] + 1, cell[1]]);

        connectivity.push(v1 as u64);
        connectivity.push(v2 as u64);
        connectivity.push(v3 as u64);
        connectivity.push(v4 as u64);

        offsets.push(connectivity.len() as u64);
    }

    let cell_verts = VertexNumbers::XML {
        connectivity,
        offsets,
    };

    let cell_types = vec![CellType::Quad; cell_total];

    let cells = Cells {
        cell_verts,
        types: cell_types,
    };

    // Generate points

    let mut vertices = Vec::new();

    for vertex in vertex_space.iter() {
        let position = node_space.position(vertex);
        vertices.extend([position[0], position[1], 0.0]);
    }

    let points = IOBuffer::new(vertices);

    // Attributes

    let psi_attr = Attribute::DataArray(DataArrayBase {
        name: "psi".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(psi.to_vec()),
    });

    let seed_attr = Attribute::DataArray(DataArrayBase {
        name: "seed".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(seed.to_vec()),
    });

    let u_attr = Attribute::DataArray(DataArrayBase {
        name: "u".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(u.to_vec()),
    });

    let w_attr = Attribute::DataArray(DataArrayBase {
        name: "w".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(w.to_vec()),
    });

    let x_attr = Attribute::DataArray(DataArrayBase {
        name: "x".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(x.to_vec()),
    });

    let lapse_attr = Attribute::DataArray(DataArrayBase {
        name: "lapse".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(lapse.to_vec()),
    });

    let shiftr_attr = Attribute::DataArray(DataArrayBase {
        name: "shiftr".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(shiftr.to_vec()),
    });

    let shiftz_attr = Attribute::DataArray(DataArrayBase {
        name: "shiftz".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(shiftz.to_vec()),
    });

    let constraint_attr = Attribute::DataArray(DataArrayBase {
        name: "constraint".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(constraint.to_vec()),
    });

    let attributes = Attributes {
        point: vec![
            psi_attr,
            seed_attr,
            u_attr,
            w_attr,
            x_attr,
            lapse_attr,
            shiftr_attr,
            shiftz_attr,
            constraint_attr,
        ],
        cell: Vec::new(),
    };

    let piece = UnstructuredGridPiece {
        points,
        cells,
        data: attributes,
    };

    let vtk = Vtk {
        version: (2, 2).into(),
        title: title.clone(),
        byte_order: ByteOrder::LittleEndian,
        data: DataSet::UnstructuredGrid {
            meta: None,
            pieces: vec![Piece::Inline(Box::new(piece))],
        },
        file_path: None,
    };

    // Write to output
    let file_path = PathBuf::from(format!("output/{title}.vtu"));
    vtk.export(&file_path).unwrap();
}

pub fn main() {
    // Scratch allocator
    let mut arena = Arena::new();

    let mesh = UniformMesh::new(
        Rectangle {
            size: [RADIUS, RADIUS],
            origin: [0.0, 0.0],
        },
        [8, 8],
        5,
    );

    let mut dynamic = dynamic_new(mesh.node_count());
    let mut gauge = gauge_new(mesh.node_count());
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

    // Write output
    write_vtk_output(0, &mesh, dynamic.as_slice(), gauge.as_slice(), &constraint);

    let mut system = DynamicIntegrator::new(&mesh, dynamic, gauge);

    let steps = 100;
    let cfl = 0.1;

    let k = cfl * 0.1;

    for i in 0..steps {
        println!("Step {}", i);
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
        // Write output
        write_vtk_output(
            i + 1,
            &mesh,
            system.dynamic.as_slice(),
            system.gauge.as_slice(),
            &constraint,
        );
    }

    // multigrid.solve(&Laplacian { bump: Bump::new() }, &rhs, &mut solution);

    println!("Solved")
}
