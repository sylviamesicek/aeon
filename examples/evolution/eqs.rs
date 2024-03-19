use crate::bcs::*;
use crate::config::*;
use aeon::prelude::*;

fn is_approximately_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < 10e-10
}

pub struct InitialSeed;

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
    pub seed: &'a [f64],
}

impl<'a> Operator<2> for InitialPsiOp<'a> {
    fn apply(self: &Self, arena: &Arena, block: &Block<2>, psi: &[f64], dest: &mut [f64]) {
        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());

        block
            .axis::<ORDER>(0)
            .second_derivative(&PSI_INITIAL_RHO, psi, psi_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&PSI_INITIAL_Z, psi, psi_zz);
        block
            .axis::<ORDER>(0)
            .derivative(&PSI_INITIAL_RHO, psi, psi_r);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block
            .axis::<ORDER>(0)
            .second_derivative(&SEED_RHO, seed, s_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<ORDER>(0).derivative(&SEED_RHO, seed, s_r);

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
            .axis::<ORDER>(0)
            .second_derivative_diag(&PSI_INITIAL_RHO, psi_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative_diag(&PSI_INITIAL_Z, psi_zz);
        block
            .axis::<ORDER>(0)
            .derivative_diag(&PSI_INITIAL_RHO, psi_r);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block
            .axis::<ORDER>(0)
            .second_derivative(&SEED_RHO, seed, s_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<ORDER>(0).derivative(&SEED_RHO, seed, s_r);

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

    fn diritchlet(_axis: usize, _face: bool) -> bool {
        false
    }
}

pub struct InitialPsiRhs<'a> {
    pub seed: &'a [f64],
}

impl<'a> Projection<2> for InitialPsiRhs<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block
            .axis::<ORDER>(0)
            .second_derivative(&SEED_RHO, seed, s_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<ORDER>(0).derivative(&SEED_RHO, seed, s_r);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            dest[i] = -1.0 / 4.0 * (rho * s_rr[i] + 2.0 * s_r[i] + rho * s_zz[i]);
        }
    }
}

pub struct LapseOp<'a> {
    pub psi: &'a [f64],
    pub seed: &'a [f64],
    pub u: &'a [f64],
    pub w: &'a [f64],
    pub x: &'a [f64],
}

impl<'a> Operator<2> for LapseOp<'a> {
    fn apply(self: &Self, arena: &Arena, block: &Block<2>, lapse: &[f64], dest: &mut [f64]) {
        assert!(lapse.len() == block.len());

        let lapse_rr = arena.alloc::<f64>(block.len());
        let lapse_zz = arena.alloc::<f64>(block.len());
        let lapse_r = arena.alloc::<f64>(block.len());
        let lapse_z = arena.alloc::<f64>(block.len());

        block
            .axis::<ORDER>(0)
            .second_derivative(&LAPSE_RHO, lapse, lapse_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&LAPSE_Z, lapse, lapse_zz);
        block
            .axis::<ORDER>(0)
            .derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<ORDER>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());
        let psi = block.auxillary(self.psi);

        block.axis::<ORDER>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<ORDER>(1).derivative(&PSI_Z, psi, psi_z);

        let seed = block.auxillary(self.seed);
        let w = block.auxillary(self.w);
        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            let psi = psi[i] + 1.0;

            let mut term1 = lapse_rr[i] + lapse_zz[i];

            if is_approximately_equal(rho, 0.0) {
                term1 += lapse_rr[i];
            } else {
                term1 += lapse_r[i] / rho;
            }

            let term2 = 2.0 / psi * (psi_r[i] * lapse_r[i] + psi_z[i] * lapse_z[i]);

            let scale = -lapse[i] * (2.0 * rho * seed[i]).exp() * psi * psi * psi * psi;

            let term3 = 2.0 / 3.0 * (rho * rho * w[i] * w[i] + rho * u[i] * w[i] + u[i] * u[i]);
            let term4 = 2.0 * x[i] * x[i];

            dest[i] = term1 + term2 + scale * (term3 + term4);
        }
    }

    fn apply_diag(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse_rr = arena.alloc::<f64>(block.len());
        let lapse_zz = arena.alloc::<f64>(block.len());
        let lapse_r = arena.alloc::<f64>(block.len());
        let lapse_z = arena.alloc::<f64>(block.len());

        block
            .axis::<ORDER>(0)
            .second_derivative_diag(&LAPSE_RHO, lapse_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative_diag(&LAPSE_Z, lapse_zz);
        block.axis::<ORDER>(0).derivative_diag(&LAPSE_RHO, lapse_r);
        block.axis::<ORDER>(1).derivative_diag(&LAPSE_Z, lapse_z);

        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());
        let psi = block.auxillary(self.psi);

        block.axis::<ORDER>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<ORDER>(1).derivative(&PSI_Z, psi, psi_z);

        let seed = block.auxillary(self.seed);
        let w = block.auxillary(self.w);
        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];

            let psi = psi[i] + 1.0;

            let mut term1 = lapse_rr[i] + lapse_zz[i];

            if is_approximately_equal(rho, 0.0) {
                term1 += lapse_rr[i];
            } else {
                term1 += lapse_r[i] / rho;
            }

            let term2 = 2.0 / psi * (psi_r[i] * lapse_r[i] + psi_z[i] * lapse_z[i]);

            let scale = -(2.0 * rho * seed[i]).exp() * psi * psi * psi * psi;

            let term3 = 2.0 / 3.0 * (rho * rho * w[i] * w[i] + rho * u[i] * w[i] + u[i] * u[i]);
            let term4 = 2.0 * x[i] * x[i];

            dest[i] = term1 + term2 + scale * (term3 + term4);
        }
    }

    fn diritchlet(_axis: usize, _face: bool) -> bool {
        false
    }
}

pub struct LapseRhs<'a> {
    pub psi: &'a [f64],
    pub seed: &'a [f64],
    pub u: &'a [f64],
    pub w: &'a [f64],
    pub x: &'a [f64],
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

            let scale = (2.0 * rho * seed[i]).exp() * psi * psi * psi * psi;
            let term1 = 2.0 / 3.0 * (rho * rho * w[i] * w[i] + rho * u[i] * w[i] + u[i] * u[i]);
            let term2 = 2.0 * x[i] * x[i];

            dest[i] = scale * (term1 + term2);
        }
    }
}

pub struct ShiftRRhs<'a> {
    pub lapse: &'a [f64],
    pub x: &'a [f64],
    pub u: &'a [f64],
}

impl<'a> Projection<2> for ShiftRRhs<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse = block.auxillary(self.lapse);
        let lapse_r = arena.alloc(block.len());
        let lapse_z = arena.alloc(block.len());

        block
            .axis::<ORDER>(0)
            .derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<ORDER>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let x = block.auxillary(self.x);
        let x_z = arena.alloc(block.len());

        block.axis::<ORDER>(1).derivative(&X_Z, x, x_z);

        let u = block.auxillary(self.u);
        let u_r = arena.alloc(block.len());

        block.axis::<ORDER>(0).derivative(&U_RHO, u, u_r);

        for (i, _) in block.iter().enumerate() {
            let lapse = lapse[i] + 1.0;

            let term1 = 2.0 * (x[i] * lapse_z[i] + lapse * x_z[i]);
            let term2 = -u[i] * lapse_r[i] - lapse * u_r[i];

            dest[i] = term1 + term2;
        }
    }
}

pub struct ShiftZRhs<'a> {
    pub lapse: &'a [f64],
    pub x: &'a [f64],
    pub u: &'a [f64],
}

impl<'a> Projection<2> for ShiftZRhs<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse = block.auxillary(self.lapse);
        let lapse_r = arena.alloc(block.len());
        let lapse_z = arena.alloc(block.len());

        block
            .axis::<ORDER>(0)
            .derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<ORDER>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let x = block.auxillary(self.x);
        let x_r = arena.alloc(block.len());

        block.axis::<ORDER>(0).derivative(&X_RHO, x, x_r);

        let u = block.auxillary(self.u);
        let u_z = arena.alloc(block.len());

        block.axis::<ORDER>(1).derivative(&U_Z, u, u_z);

        for (i, _) in block.iter().enumerate() {
            let lapse = lapse[i] + 1.0;

            let term1 = 2.0 * (x[i] * lapse_r[i] + lapse * x_r[i]);
            let term2 = u[i] * lapse_z[i] + lapse * u_z[i];

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
            .axis::<ORDER>(0)
            .second_derivative(&SHIFTR_RHO, shiftr, shiftr_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&SHIFTR_Z, shiftr, shiftr_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = shiftr_rr[i] + shiftr_zz[i];
        }
    }

    fn apply_diag(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let shiftr_rr = arena.alloc::<f64>(block.len());
        let shiftr_zz = arena.alloc::<f64>(block.len());

        block
            .axis::<ORDER>(0)
            .second_derivative_diag(&SHIFTR_RHO, shiftr_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative_diag(&SHIFTR_Z, shiftr_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = shiftr_rr[i] + shiftr_zz[i];
        }
    }

    fn diritchlet(axis: usize, face: bool) -> bool {
        match (axis, face) {
            (0, false) => true,
            _ => false,
        }
    }
}

pub struct ShiftZOp;

impl Operator<2> for ShiftZOp {
    fn apply(self: &Self, arena: &Arena, block: &Block<2>, shiftz: &[f64], dest: &mut [f64]) {
        let shiftz_rr = arena.alloc::<f64>(block.len());
        let shiftz_zz = arena.alloc::<f64>(block.len());

        block
            .axis::<ORDER>(0)
            .second_derivative(&SHIFTZ_RHO, shiftz, shiftz_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&SHIFTZ_Z, shiftz, shiftz_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = shiftz_rr[i] + shiftz_zz[i];
        }
    }

    fn apply_diag(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let shiftz_rr = arena.alloc::<f64>(block.len());
        let shiftz_zz = arena.alloc::<f64>(block.len());

        block
            .axis::<ORDER>(0)
            .second_derivative_diag(&SHIFTZ_RHO, shiftz_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative_diag(&SHIFTZ_Z, shiftz_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = shiftz_rr[i] + shiftz_zz[i];
        }
    }

    fn diritchlet(axis: usize, face: bool) -> bool {
        match (axis, face) {
            (1, false) => true,
            _ => false,
        }
    }
}

pub struct Hamiltonian<'a> {
    pub psi: &'a [f64],
    pub seed: &'a [f64],
    pub u: &'a [f64],
    pub x: &'a [f64],
    pub w: &'a [f64],
}

impl<'a> Projection<2> for Hamiltonian<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block
            .axis::<ORDER>(0)
            .second_derivative(&PSI_RHO, psi, psi_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&PSI_Z, psi, psi_zz);
        block.axis::<ORDER>(0).derivative(&PSI_RHO, psi, psi_r);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block
            .axis::<ORDER>(0)
            .second_derivative(&SEED_RHO, seed, s_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<ORDER>(0).derivative(&SEED_RHO, seed, s_r);

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

            let term3 = 1.0 / 3.0 * (rho * rho * w[i] * w[i] + rho * u[i] * w[i] + u[i] * u[i]);
            let term4 = x[i] * x[i];

            dest[i] = term1 + term2 + scale * (term3 + term4);
        }
    }
}

pub struct PsiEvolution<'a> {
    pub lapse: &'a [f64],
    pub shiftr: &'a [f64],
    pub shiftz: &'a [f64],
    pub psi: &'a [f64],
    pub u: &'a [f64],
    pub w: &'a [f64],
}

impl<'a> Projection<2> for PsiEvolution<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse = block.auxillary(self.lapse);
        let shiftr = block.auxillary(self.shiftr);
        let shiftz = block.auxillary(self.shiftz);

        let shiftr_r = arena.alloc::<f64>(block.len());
        block
            .axis::<ORDER>(0)
            .derivative(&SHIFTR_RHO, shiftr, shiftr_r);

        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block.axis::<ORDER>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<ORDER>(1).derivative(&PSI_Z, psi, psi_z);

        let u = block.auxillary(self.u);
        let w = block.auxillary(self.w);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let _z = position[1];

            // if is_approximately_equal(rho, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / rho * psi_r[i];
            //     let term2 = -psi[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // } else if is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / z * psi_z[i];
            //     let term2 = -psi[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = rho / r * psi_r[i];
            //     let term2 = z / r * psi_z[i];

            //     dest[i] = -term1 - term2 - psi[i] / r;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     dest[i] = shiftr[i] * psi_r[i] + shiftz[i] * psi_z[i];

            //     continue;
            // }

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
    pub lapse: &'a [f64],
    pub shiftr: &'a [f64],
    pub shiftz: &'a [f64],
    pub seed: &'a [f64],
    pub w: &'a [f64],
}

impl<'a> Projection<2> for SeedEvolution<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse = block.auxillary(self.lapse);
        let shiftr = block.auxillary(self.shiftr);
        let shiftz = block.auxillary(self.shiftz);

        let shiftr_r = arena.alloc::<f64>(block.len());
        block
            .axis::<ORDER>(0)
            .derivative(&SHIFTR_RHO, shiftr, shiftr_r);

        let s_r = arena.alloc::<f64>(block.len());
        let s_z = arena.alloc::<f64>(block.len());

        let s = block.auxillary(self.seed);

        block.axis::<ORDER>(0).derivative(&SEED_RHO, s, s_r);
        block.axis::<ORDER>(1).derivative(&SEED_Z, s, s_z);

        let w = block.auxillary(self.w);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let _z = position[1];

            if is_approximately_equal(rho, 0.0) {
                dest[i] = 0.0;

                continue;
            }

            // if is_approximately_equal(rho, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / rho * s_r[i];
            //     let term2 = -s[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // } else if is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / z * s_z[i];
            //     let term2 = -s[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = rho / r * s_r[i];
            //     let term2 = z / r * s_z[i];

            //     dest[i] = -term1 - term2 - s[i] / r;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     dest[i] = shiftr[i] * s_r[i] + shiftz[i] * s_z[i] + shiftr[i] * s[i] / rho;

            //     continue;
            // }

            let lapse = lapse[i] + 1.0;

            let term1 = shiftr[i] * s_r[i] + shiftz[i] * s_z[i];
            let term2 = -lapse * w[i] + shiftr[i] * s[i] / rho;
            let term3 = shiftr_r[i] / rho - shiftr[i] / (rho * rho);

            dest[i] = term1 + term2 + term3;
        }
    }
}

pub struct WEvolution<'a> {
    pub lapse: &'a [f64],
    pub shiftr: &'a [f64],
    pub shiftz: &'a [f64],
    pub psi: &'a [f64],
    pub seed: &'a [f64],
    pub x: &'a [f64],
    pub w: &'a [f64],
}

impl<'a> Projection<2> for WEvolution<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse_rr = arena.alloc(block.len());
        let lapse_zz = arena.alloc(block.len());
        let lapse_r = arena.alloc(block.len());
        let lapse_z = arena.alloc(block.len());
        let lapse = block.auxillary(self.lapse);

        block
            .axis::<ORDER>(0)
            .second_derivative(&LAPSE_RHO, lapse, lapse_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&LAPSE_Z, lapse, lapse_zz);
        block
            .axis::<ORDER>(0)
            .derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<ORDER>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let shiftr_z = arena.alloc(block.len());
        let shiftr = block.auxillary(self.shiftr);

        block
            .axis::<ORDER>(1)
            .derivative(&SHIFTR_Z, shiftr, shiftr_z);

        let shiftz_r = arena.alloc(block.len());
        let shiftz = block.auxillary(self.shiftz);

        block
            .axis::<ORDER>(0)
            .derivative(&SHIFTZ_RHO, shiftz, shiftz_r);

        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block
            .axis::<ORDER>(0)
            .second_derivative(&PSI_RHO, psi, psi_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&PSI_Z, psi, psi_zz);
        block.axis::<ORDER>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<ORDER>(1).derivative(&PSI_Z, psi, psi_z);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());
        let s_z = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block
            .axis::<ORDER>(0)
            .second_derivative(&SEED_RHO, seed, s_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<ORDER>(0).derivative(&SEED_RHO, seed, s_r);
        block.axis::<ORDER>(1).derivative(&SEED_Z, seed, s_z);

        let w = block.auxillary(self.w);
        let x = block.auxillary(self.x);

        let w_r = arena.alloc(block.len());
        let w_z = arena.alloc(block.len());

        block.axis::<ORDER>(0).derivative(&W_RHO, w, w_r);
        block.axis::<ORDER>(1).derivative(&W_Z, w, w_z);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let _z = position[1];

            if is_approximately_equal(rho, 0.0) {
                dest[i] = 0.0;

                continue;
            }

            // if is_approximately_equal(rho, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / rho * w_r[i];
            //     let term2 = -w[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // } else if is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / z * w_z[i];
            //     let term2 = -w[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = rho / r * w_r[i];
            //     let term2 = z / r * w_z[i];

            //     dest[i] = -term1 - term2 - w[i] / r;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     dest[i] = shiftr[i] * w_r[i] + shiftz[i] * w_z[i] + shiftr[i] * w[i] / rho;

            //     continue;
            // }

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
    pub lapse: &'a [f64],
    pub shiftr: &'a [f64],
    pub shiftz: &'a [f64],
    pub psi: &'a [f64],
    pub seed: &'a [f64],
    pub x: &'a [f64],
    pub u: &'a [f64],
}

impl<'a> Projection<2> for UEvolution<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let lapse_rr = arena.alloc(block.len());
        let lapse_zz = arena.alloc(block.len());
        let lapse_r = arena.alloc(block.len());
        let lapse_z = arena.alloc(block.len());
        let lapse = block.auxillary(self.lapse);

        block
            .axis::<ORDER>(0)
            .second_derivative(&LAPSE_RHO, lapse, lapse_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&LAPSE_Z, lapse, lapse_zz);
        block
            .axis::<ORDER>(0)
            .derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<ORDER>(1).derivative(&LAPSE_Z, lapse, lapse_z);

        let shiftr_z = arena.alloc(block.len());
        let shiftr = block.auxillary(self.shiftr);

        block
            .axis::<ORDER>(1)
            .derivative(&SHIFTR_Z, shiftr, shiftr_z);

        let shiftz_r = arena.alloc(block.len());
        let shiftz = block.auxillary(self.shiftz);

        block
            .axis::<ORDER>(0)
            .derivative(&SHIFTZ_RHO, shiftz, shiftz_r);

        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block
            .axis::<ORDER>(0)
            .second_derivative(&PSI_RHO, psi, psi_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&PSI_Z, psi, psi_zz);
        block.axis::<ORDER>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<ORDER>(1).derivative(&PSI_Z, psi, psi_z);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());
        let s_z = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block
            .axis::<ORDER>(0)
            .second_derivative(&SEED_RHO, seed, s_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<ORDER>(0).derivative(&SEED_RHO, seed, s_r);
        block.axis::<ORDER>(1).derivative(&SEED_Z, seed, s_z);

        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        let u_r = arena.alloc(block.len());
        let u_z = arena.alloc(block.len());

        block.axis::<4>(0).derivative(&U_RHO, u, u_r);
        block.axis::<4>(1).derivative(&U_Z, u, u_z);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let _z = position[1];

            // if is_approximately_equal(rho, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / rho * u_r[i];
            //     let term2 = -u[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // } else if is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / z * u_z[i];
            //     let term2 = -u[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = rho / r * u_r[i];
            //     let term2 = z / r * u_z[i];

            //     dest[i] = -term1 - term2 - u[i] / r;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     dest[i] = shiftr[i] * u_r[i] + shiftz[i] * u_z[i];

            //     continue;
            // }

            let psi = psi[i] + 1.0;
            let lapse = lapse[i] + 1.0;

            let term1 = shiftr[i] * u_r[i] + shiftz[i] * u_z[i];
            let term2 = 2.0 * x[i] * (shiftr_z[i] - shiftz_r[i]);

            let scale1 = (-2.0 * rho * seed[i]).exp() / (psi * psi * psi * psi);
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
    pub lapse: &'a [f64],
    pub shiftr: &'a [f64],
    pub shiftz: &'a [f64],
    pub psi: &'a [f64],
    pub seed: &'a [f64],
    pub x: &'a [f64],
    pub u: &'a [f64],
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
            .axis::<ORDER>(0)
            .second_derivative(&LAPSE_RHO, lapse, lapse_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&LAPSE_Z, lapse, lapse_zz);
        block
            .axis::<ORDER>(0)
            .derivative(&LAPSE_RHO, lapse, lapse_r);
        block.axis::<ORDER>(1).derivative(&LAPSE_Z, lapse, lapse_z);
        block
            .axis::<ORDER>(1)
            .derivative(&LAPSE_Z, lapse_r, lapse_rz);

        let shiftr_z = arena.alloc(block.len());
        let shiftr = block.auxillary(self.shiftr);

        block
            .axis::<ORDER>(1)
            .derivative(&SHIFTR_Z, shiftr, shiftr_z);

        let shiftz_r = arena.alloc(block.len());
        let shiftz = block.auxillary(self.shiftz);

        block
            .axis::<ORDER>(0)
            .derivative(&SHIFTZ_RHO, shiftz, shiftz_r);

        let psi_rr = arena.alloc::<f64>(block.len());
        let psi_zz = arena.alloc::<f64>(block.len());
        let psi_rz = arena.alloc::<f64>(block.len());
        let psi_r = arena.alloc::<f64>(block.len());
        let psi_z = arena.alloc::<f64>(block.len());

        let psi = block.auxillary(self.psi);

        block
            .axis::<ORDER>(0)
            .second_derivative(&PSI_RHO, psi, psi_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&PSI_Z, psi, psi_zz);
        block.axis::<ORDER>(0).derivative(&PSI_RHO, psi, psi_r);
        block.axis::<ORDER>(1).derivative(&PSI_Z, psi, psi_z);
        block.axis::<ORDER>(1).derivative(&PSI_Z, psi_r, psi_rz);

        let s_rr = arena.alloc::<f64>(block.len());
        let s_zz = arena.alloc::<f64>(block.len());
        let s_r = arena.alloc::<f64>(block.len());
        let s_z = arena.alloc::<f64>(block.len());

        let seed = block.auxillary(self.seed);

        block
            .axis::<ORDER>(0)
            .second_derivative(&SEED_RHO, seed, s_rr);
        block
            .axis::<ORDER>(1)
            .second_derivative(&SEED_Z, seed, s_zz);
        block.axis::<ORDER>(0).derivative(&SEED_RHO, seed, s_r);
        block.axis::<ORDER>(1).derivative(&SEED_Z, seed, s_z);

        let u = block.auxillary(self.u);
        let x = block.auxillary(self.x);

        let x_r = arena.alloc(block.len());
        let x_z = arena.alloc(block.len());

        block.axis::<ORDER>(0).derivative(&X_RHO, x, x_r);
        block.axis::<ORDER>(1).derivative(&X_Z, x, x_z);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let rho = position[0];
            let _z = position[1];

            if is_approximately_equal(rho, 0.0) {
                dest[i] = 0.0;

                continue;
            }

            // if is_approximately_equal(rho, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / rho * x_r[i];
            //     let term2 = -x[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // } else if is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = -r / z * x_z[i];
            //     let term2 = -x[i] / r;

            //     dest[i] = term1 + term2;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     let r = (rho * rho + z * z).sqrt();

            //     let term1 = rho / r * x_r[i];
            //     let term2 = z / r * x_z[i];

            //     dest[i] = -term1 - term2 - x[i] / r;

            //     continue;
            // }

            // if is_approximately_equal(rho, RADIUS) || is_approximately_equal(z, RADIUS) {
            //     dest[i] = shiftr[i] * x_r[i] + shiftz[i] * x_z[i];

            //     continue;
            // }

            let psi = psi[i] + 1.0;
            let lapse = lapse[i] + 1.0;

            let term1 = shiftr[i] * x_r[i] + shiftz[i] * x_z[i];
            let term2 = 1.0 / 2.0 * u[i] * (shiftz_r[i] - shiftr_z[i]);

            let scale1 = (-2.0 * rho * seed[i]).exp() / (psi * psi * psi * psi);
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
