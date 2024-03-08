use aeon::{
    common::{
        AntiSymmetricBoundary, FDDerivative, FDSecondDerivative, FreeBoundary, RobinBoundary,
        SymmetricBoundary,
    },
    prelude::*,
};
use bumpalo::Bump;

fn asymptotic_flatness_rho(pos: [f64; 2]) -> RobinBoundary<4> {
    RobinBoundary::new(-pos[0] / (pos[0] * pos[0] + pos[1] * pos[1]))
}

fn asymptotic_flatness_z(pos: [f64; 2]) -> RobinBoundary<4> {
    RobinBoundary::new(-pos[1] / (pos[0] * pos[0] + pos[1] * pos[1]))
}

pub struct SecondDerivativeRhoOdd;

impl Convolution<2> for SecondDerivativeRhoOdd {
    type Kernel = FDSecondDerivative<4>;

    type NegativeBoundary = AntiSymmetricBoundary<2>;
    type PositiveBoundary = RobinBoundary<4>;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        AntiSymmetricBoundary::<2>
    }

    fn positive(self: &Self, pos: [f64; 2]) -> Self::PositiveBoundary {
        asymptotic_flatness_rho(pos)
    }
}

pub struct SecondDerivativeRhoEven;

impl Convolution<2> for SecondDerivativeRhoEven {
    type Kernel = FDSecondDerivative<4>;

    type NegativeBoundary = SymmetricBoundary<2>;
    type PositiveBoundary = RobinBoundary<4>;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        SymmetricBoundary::<2>
    }

    fn positive(self: &Self, pos: [f64; 2]) -> Self::PositiveBoundary {
        asymptotic_flatness_rho(pos)
    }
}

pub struct SecondDerivativeZOdd;

impl Convolution<2> for SecondDerivativeZOdd {
    type Kernel = FDSecondDerivative<4>;

    type NegativeBoundary = AntiSymmetricBoundary<2>;
    type PositiveBoundary = RobinBoundary<4>;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        AntiSymmetricBoundary::<2>
    }

    fn positive(self: &Self, pos: [f64; 2]) -> Self::PositiveBoundary {
        asymptotic_flatness_z(pos)
    }
}

pub struct SecondDerivativeZEven;

impl Convolution<2> for SecondDerivativeZEven {
    type Kernel = FDSecondDerivative<4>;

    type NegativeBoundary = SymmetricBoundary<2>;
    type PositiveBoundary = RobinBoundary<4>;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        SymmetricBoundary::<2>
    }

    fn positive(self: &Self, pos: [f64; 2]) -> Self::PositiveBoundary {
        asymptotic_flatness_z(pos)
    }
}

pub struct DerivativeRhoOdd;

impl Convolution<2> for DerivativeRhoOdd {
    type Kernel = FDDerivative<4>;

    type NegativeBoundary = AntiSymmetricBoundary<2>;
    type PositiveBoundary = RobinBoundary<4>;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        AntiSymmetricBoundary::<2>
    }

    fn positive(self: &Self, pos: [f64; 2]) -> Self::PositiveBoundary {
        asymptotic_flatness_rho(pos)
    }
}

pub struct DerivativeRhoEven;

impl Convolution<2> for DerivativeRhoEven {
    type Kernel = FDDerivative<4>;

    type NegativeBoundary = SymmetricBoundary<2>;
    type PositiveBoundary = RobinBoundary<4>;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        SymmetricBoundary::<2>
    }

    fn positive(self: &Self, pos: [f64; 2]) -> Self::PositiveBoundary {
        asymptotic_flatness_rho(pos)
    }
}

pub struct DerivativeZOdd;

impl Convolution<2> for DerivativeZOdd {
    type Kernel = FDDerivative<4>;

    type NegativeBoundary = AntiSymmetricBoundary<2>;
    type PositiveBoundary = RobinBoundary<4>;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        AntiSymmetricBoundary::<2>
    }

    fn positive(self: &Self, pos: [f64; 2]) -> Self::PositiveBoundary {
        asymptotic_flatness_z(pos)
    }
}

pub struct DerivativeZEven;

impl Convolution<2> for DerivativeZEven {
    type Kernel = FDDerivative<4>;

    type NegativeBoundary = SymmetricBoundary<2>;
    type PositiveBoundary = RobinBoundary<4>;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        SymmetricBoundary::<2>
    }

    fn positive(self: &Self, pos: [f64; 2]) -> Self::PositiveBoundary {
        asymptotic_flatness_z(pos)
    }
}

pub struct SecondDerivativeOddFree;

impl Convolution<2> for SecondDerivativeOddFree {
    type Kernel = FDSecondDerivative<4>;

    type NegativeBoundary = AntiSymmetricBoundary<2>;
    type PositiveBoundary = FreeBoundary;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        AntiSymmetricBoundary::<2>
    }

    fn positive(self: &Self, _: [f64; 2]) -> Self::PositiveBoundary {
        FreeBoundary
    }
}

pub struct SecondDerivativeEvenFree;

impl Convolution<2> for SecondDerivativeEvenFree {
    type Kernel = FDSecondDerivative<4>;

    type NegativeBoundary = SymmetricBoundary<2>;
    type PositiveBoundary = FreeBoundary;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        SymmetricBoundary::<2>
    }

    fn positive(self: &Self, _: [f64; 2]) -> Self::PositiveBoundary {
        FreeBoundary
    }
}

pub struct DerivativeOddFree;

impl Convolution<2> for DerivativeOddFree {
    type Kernel = FDDerivative<4>;

    type NegativeBoundary = AntiSymmetricBoundary<2>;
    type PositiveBoundary = FreeBoundary;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        AntiSymmetricBoundary::<2>
    }

    fn positive(self: &Self, _: [f64; 2]) -> Self::PositiveBoundary {
        FreeBoundary
    }
}

pub struct DerivativeEvenFree;

impl Convolution<2> for DerivativeEvenFree {
    type Kernel = FDDerivative<4>;

    type NegativeBoundary = SymmetricBoundary<2>;
    type PositiveBoundary = FreeBoundary;

    fn negative(self: &Self, _: [f64; 2]) -> Self::NegativeBoundary {
        SymmetricBoundary::<2>
    }

    fn positive(self: &Self, _: [f64; 2]) -> Self::PositiveBoundary {
        FreeBoundary
    }
}

const LAPSE_DERIVATIVE_RHO: DerivativeRhoEven = DerivativeRhoEven {};
const LAPSE_SECOND_DERIVATIVE_RHO: SecondDerivativeRhoEven = SecondDerivativeRhoEven {};
const LAPSE_DERIVATIVE_Z: DerivativeZEven = DerivativeZEven {};
const LAPSE_SECOND_DERIVATIVE_Z: SecondDerivativeZEven = SecondDerivativeZEven {};

const SHIFTRHO_DERIVATIVE_RHO: DerivativeRhoOdd = DerivativeRhoOdd {};
const SHIFTRHO_SECOND_DERIVATIVE_RHO: SecondDerivativeRhoOdd = SecondDerivativeRhoOdd {};
const SHIFTRHO_DERIVATIVE_Z: DerivativeZOdd = DerivativeZOdd {};
const SHIFTRHO_SECOND_DERIVATIVE_Z: SecondDerivativeZOdd = SecondDerivativeZOdd {};

const SHIFTZ_DERIVATIVE_RHO: DerivativeRhoEven = DerivativeRhoEven {};
const SHIFTZ_SECOND_DERIVATIVE_RHO: SecondDerivativeRhoEven = SecondDerivativeRhoEven {};
const SHIFTZ_DERIVATIVE_Z: DerivativeZEven = DerivativeZEven {};
const SHIFTZ_SECOND_DERIVATIVE_Z: SecondDerivativeZEven = SecondDerivativeZEven {};

const PSI_INITIAL_DERIVATIVE_RHO: DerivativeRhoEven = DerivativeRhoEven {};
const PSI_INITIAL_SECOND_DERIVATIVE_RHO: SecondDerivativeRhoEven = SecondDerivativeRhoEven {};
const PSI_INITIAL_DERIVATIVE_Z: DerivativeZEven = DerivativeZEven {};
const PSI_INITIAL_SECOND_DERIVATIVE_Z: SecondDerivativeZEven = SecondDerivativeZEven {};

const PSI_SECOND_DERIVATIVE: SecondDerivativeEvenFree = SecondDerivativeEvenFree {};
const PSI_DERIVATIVE: DerivativeEvenFree = DerivativeEvenFree {};

const SEED_SECOND_DERIVATIVE: SecondDerivativeOddFree = SecondDerivativeOddFree {};
const SEED_DERIVATIVE: DerivativeOddFree = DerivativeOddFree {};

const W_SECOND_DERIVATIVE: SecondDerivativeOddFree = SecondDerivativeOddFree {};
const W_DERIVATIVE: DerivativeOddFree = DerivativeOddFree {};

const U_SECOND_DERIVATIVE: SecondDerivativeEvenFree = SecondDerivativeEvenFree {};
const U_DERIVATIVE: DerivativeEvenFree = DerivativeEvenFree {};

const X_SECOND_DERIVATIVE: SecondDerivativeOddFree = SecondDerivativeOddFree {};
const X_DERIVATIVE: DerivativeOddFree = DerivativeOddFree {};

fn is_approximately_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < 10e-10
}

fn initial_seed(mesh: &UniformMesh<2>, seed: &mut [f64]) {
    let block = mesh.level_block(mesh.level_count() - 1);

    for node in block.iter() {
        let index = block.global_from_local(node);
        let position = block.position(node);

        let rho = position[0];
        let z = position[1];

        let rho2 = rho * rho;
        let z2 = z * z;
        let sigma2 = 1.0;

        seed[index] = rho * (-(rho2 + z2) / sigma2).exp();
    }

    mesh.restrict(seed);
}

pub struct InitialPhs<'a> {
    seed: &'a [f64],
    bump: Bump,
}

impl<'a> InitialPhs<'a> {
    pub fn rhs(self: &mut Self, mesh: &UniformMesh<2>, rhs: &mut [f64]) {
        let block = mesh.level_block(mesh.level_count() - 1);

        let s_rr = self.bump.alloc_slice_fill_default(block.len());
        let s_zz = self.bump.alloc_slice_fill_default(block.len());
        let s_r = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_auxillary(&SEED_SECOND_DERIVATIVE, self.seed, s_rr);
        block
            .axis(1)
            .evaluate_auxillary(&SEED_SECOND_DERIVATIVE, self.seed, s_zz);
        block
            .axis(0)
            .evaluate_auxillary(&SEED_DERIVATIVE, self.seed, s_r);

        for (i, node) in block.iter().enumerate() {
            let index = block.global_from_local(node);
            let position = block.position(node);

            let rho = position[0];

            rhs[index] = -1.0 / 4.0 * (rho * s_rr[i] + 2.0 * s_r[i] + rho * s_zz[i]);
        }

        self.bump.reset();

        mesh.restrict(rhs);
    }
}

impl<'a> Operator<2> for InitialPhs<'a> {
    fn apply(self: &mut Self, block: &Block<2>, psi: &[f64], dest: &mut [f64]) {
        let s_rr = self.bump.alloc_slice_fill_default(block.len());
        let s_zz = self.bump.alloc_slice_fill_default(block.len());
        let s_r = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_auxillary(&SEED_SECOND_DERIVATIVE, self.seed, s_rr);
        block
            .axis(1)
            .evaluate_auxillary(&SEED_SECOND_DERIVATIVE, self.seed, s_zz);
        block
            .axis(0)
            .evaluate_auxillary(&SEED_DERIVATIVE, self.seed, s_r);

        let psi_rr = self.bump.alloc_slice_fill_default(block.len());
        let psi_zz = self.bump.alloc_slice_fill_default(block.len());
        let psi_r = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate(&PSI_INITIAL_SECOND_DERIVATIVE_RHO, self.seed, psi_rr);
        block
            .axis(1)
            .evaluate(&PSI_INITIAL_SECOND_DERIVATIVE_Z, self.seed, psi_zz);
        block
            .axis(0)
            .evaluate(&PSI_INITIAL_DERIVATIVE_RHO, self.seed, psi_r);

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

        self.bump.reset();
    }

    fn apply_diag(self: &mut Self, block: &Block<2>, psi: &[f64], dest: &mut [f64]) {
        let s_rr = self.bump.alloc_slice_fill_default(block.len());
        let s_zz = self.bump.alloc_slice_fill_default(block.len());
        let s_r = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_auxillary(&SEED_SECOND_DERIVATIVE, self.seed, s_rr);
        block
            .axis(1)
            .evaluate_auxillary(&SEED_SECOND_DERIVATIVE, self.seed, s_zz);
        block
            .axis(0)
            .evaluate_auxillary(&SEED_DERIVATIVE, self.seed, s_r);

        let psi_rr = self.bump.alloc_slice_fill_default(block.len());
        let psi_zz = self.bump.alloc_slice_fill_default(block.len());
        let psi_r = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_diag(&PSI_INITIAL_SECOND_DERIVATIVE_RHO, psi, psi_rr);
        block
            .axis(1)
            .evaluate_diag(&PSI_INITIAL_SECOND_DERIVATIVE_Z, psi, psi_zz);
        block
            .axis(0)
            .evaluate_diag(&PSI_INITIAL_DERIVATIVE_RHO, psi, psi_r);

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

        self.bump.reset();
    }
}

pub struct LapseOp<'a> {
    psi: &'a [f64],
    seed: &'a [f64],
    u: &'a [f64],
    w: &'a [f64],
    x: &'a [f64],
    bump: &'a mut Bump,
}

impl<'a> LapseOp<'a> {
    pub fn rhs(self: &mut Self, mesh: &UniformMesh<2>, rhs: &mut [f64]) {
        let block = mesh.level_block(mesh.level_count() - 1);

        for node in block.iter() {
            let i = block.global_from_local(node);
            let position = block.position(node);
            let rho = position[0];

            let psi = self.psi[i] + 1.0;
            let w = self.w[i];
            let u = self.u[i];
            let x = self.x[i];

            let scale = (2.0 * rho * self.seed[i]).exp() * psi * psi * psi * psi;

            let term1 = 2.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            let term2 = 2.0 * x * x;

            rhs[i] = scale * (term1 + term2);
        }

        self.bump.reset();

        mesh.restrict(rhs);
    }
}

impl<'a> Operator<2> for LapseOp<'a> {
    fn apply(self: &mut Self, block: &Block<2>, lapse: &[f64], dest: &mut [f64]) {
        let lapse_rr = self.bump.alloc_slice_fill_default(block.len());
        let lapse_zz = self.bump.alloc_slice_fill_default(block.len());
        let lapse_r = self.bump.alloc_slice_fill_default(block.len());
        let lapse_z = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate(&LAPSE_SECOND_DERIVATIVE_RHO, lapse, lapse_rr);
        block
            .axis(1)
            .evaluate(&LAPSE_SECOND_DERIVATIVE_Z, lapse, lapse_zz);
        block
            .axis(0)
            .evaluate(&LAPSE_DERIVATIVE_RHO, lapse, lapse_r);
        block.axis(1).evaluate(&LAPSE_DERIVATIVE_Z, lapse, lapse_z);

        let psi_r = self.bump.alloc_slice_fill_default(block.len());
        let psi_z = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_auxillary(&PSI_DERIVATIVE, self.psi, psi_r);

        block
            .axis(1)
            .evaluate_auxillary(&PSI_DERIVATIVE, self.psi, psi_z);

        for (local, node) in block.iter().enumerate() {
            let global = block.global_from_local(node);
            let position = block.position(node);
            let rho = position[0];

            let psi = self.psi[global] + 1.0;

            let seed = self.seed[global];
            let u = self.u[global];
            let x = self.x[global];
            let w = self.w[global];

            let mut term1 = lapse_rr[local] + lapse_zz[local];

            if is_approximately_equal(rho, 0.0) {
                term1 += lapse_rr[local];
            } else {
                term1 += lapse_r[local] / rho;
            }

            let term2 = 2.0 / psi * (psi_r[local] * lapse_r[local] + psi_z[local] * lapse_z[local]);

            let scale = -lapse[local] * (2.0 * rho * seed) * psi * psi * psi * psi;

            let term3 = 2.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            let term4 = 2.0 * x * x;

            dest[local] = term1 + term2 + scale * (term3 + term4);
        }

        self.bump.reset();
    }

    fn apply_diag(self: &mut Self, block: &Block<2>, lapse: &[f64], dest: &mut [f64]) {
        let lapse_rr = self.bump.alloc_slice_fill_default(block.len());
        let lapse_zz = self.bump.alloc_slice_fill_default(block.len());
        let lapse_r = self.bump.alloc_slice_fill_default(block.len());
        let lapse_z = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_diag(&LAPSE_SECOND_DERIVATIVE_RHO, lapse, lapse_rr);
        block
            .axis(1)
            .evaluate_diag(&LAPSE_SECOND_DERIVATIVE_Z, lapse, lapse_zz);
        block
            .axis(0)
            .evaluate_diag(&LAPSE_DERIVATIVE_RHO, lapse, lapse_r);
        block
            .axis(1)
            .evaluate_diag(&LAPSE_DERIVATIVE_Z, lapse, lapse_z);

        let psi_r = self.bump.alloc_slice_fill_default(block.len());
        let psi_z = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_auxillary(&PSI_DERIVATIVE, self.psi, psi_r);

        block
            .axis(1)
            .evaluate_auxillary(&PSI_DERIVATIVE, self.psi, psi_z);

        for (local, node) in block.iter().enumerate() {
            let global = block.global_from_local(node);
            let position = block.position(node);
            let rho = position[0];

            let psi = self.psi[global] + 1.0;

            let seed = self.seed[global];
            let u = self.u[global];
            let x = self.x[global];
            let w = self.w[global];

            let mut term1 = lapse_rr[local] + lapse_zz[local];

            if is_approximately_equal(rho, 0.0) {
                term1 += lapse_rr[local];
            } else {
                term1 += lapse_r[local] / rho;
            }

            let term2 = 2.0 / psi * (psi_r[local] * lapse_r[local] + psi_z[local] * lapse_z[local]);

            let scale = -lapse[local] * (2.0 * rho * seed) * psi * psi * psi * psi;

            let term3 = 2.0 / 3.0 * (rho * rho * w * w + rho * u * w + u * u);
            let term4 = 2.0 * x * x;

            dest[local] = term1 + term2 + scale * (term3 + term4);
        }

        self.bump.reset();
    }
}

pub struct ShiftRhoOp<'a> {
    lapse: &'a [f64],
    x: &'a [f64],
    u: &'a [f64],
    bump: &'a mut Bump,
}

impl<'a> ShiftRhoOp<'a> {
    pub fn rhs(self: &mut Self, mesh: &UniformMesh<2>, rhs: &mut [f64]) {
        let block = mesh.level_block(mesh.level_count() - 1);

        let lapse_r = self.bump.alloc_slice_fill_default(block.len());
        let lapse_z = self.bump.alloc_slice_fill_default(block.len());

        let x_z = self.bump.alloc_slice_fill_default(block.len());
        let u_r = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_auxillary(&LAPSE_DERIVATIVE_RHO, self.lapse, lapse_r);
        block
            .axis(1)
            .evaluate_auxillary(&LAPSE_DERIVATIVE_Z, self.lapse, lapse_z);

        block.axis(0).evaluate_auxillary(&U_DERIVATIVE, self.u, u_r);
        block.axis(1).evaluate_auxillary(&X_DERIVATIVE, self.x, x_z);

        for (local, node) in block.iter().enumerate() {
            let global = block.global_from_local(node);

            let lapse = self.lapse[global] + 1.0;
            let x = self.x[global];
            let u = self.u[global];

            rhs[global] = 2.0 * (x * lapse_z[local] + lapse * x_z[local])
                - u * lapse_r[local]
                - lapse * u_r[local];
        }

        self.bump.reset();

        mesh.restrict(rhs);
    }
}

impl<'a> Operator<2> for ShiftRhoOp<'a> {
    fn apply(self: &mut Self, block: &Block<2>, shift: &[f64], dest: &mut [f64]) {
        let shift_rr = self.bump.alloc_slice_fill_default(block.len());
        let shift_zz = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate(&SHIFTRHO_SECOND_DERIVATIVE_RHO, shift, shift_rr);
        block
            .axis(1)
            .evaluate(&SHIFTRHO_SECOND_DERIVATIVE_Z, shift, shift_zz);

        for (local, _) in block.iter().enumerate() {
            dest[local] = shift_rr[local] + shift_zz[local];
        }

        self.bump.reset();
    }

    fn apply_diag(self: &mut Self, block: &Block<2>, shift: &[f64], dest: &mut [f64]) {
        let shift_rr = self.bump.alloc_slice_fill_default(block.len());
        let shift_zz = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_diag(&SHIFTRHO_SECOND_DERIVATIVE_RHO, shift, shift_rr);
        block
            .axis(1)
            .evaluate_diag(&SHIFTRHO_SECOND_DERIVATIVE_Z, shift, shift_zz);

        for (local, _) in block.iter().enumerate() {
            dest[local] = shift_rr[local] + shift_zz[local];
        }
    }
}

pub struct ShiftZOp<'a> {
    lapse: &'a [f64],
    x: &'a [f64],
    u: &'a [f64],
    bump: &'a mut Bump,
}

impl<'a> ShiftZOp<'a> {
    pub fn rhs(self: &mut Self, mesh: &UniformMesh<2>, rhs: &mut [f64]) {
        let block = mesh.level_block(mesh.level_count() - 1);

        let lapse_r = self.bump.alloc_slice_fill_default(block.len());
        let lapse_z = self.bump.alloc_slice_fill_default(block.len());

        let x_r = self.bump.alloc_slice_fill_default(block.len());
        let u_z = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_auxillary(&LAPSE_DERIVATIVE_RHO, self.lapse, lapse_r);
        block
            .axis(1)
            .evaluate_auxillary(&LAPSE_DERIVATIVE_Z, self.lapse, lapse_z);

        block.axis(0).evaluate_auxillary(&U_DERIVATIVE, self.u, u_z);
        block.axis(1).evaluate_auxillary(&X_DERIVATIVE, self.x, x_r);

        for (local, node) in block.iter().enumerate() {
            let global = block.global_from_local(node);

            let lapse = self.lapse[global] + 1.0;
            let x = self.x[global];
            let u = self.u[global];

            rhs[global] = 2.0 * (x * lapse_r[local] + lapse * x_r[local])
                + u * lapse_z[local]
                + lapse * u_z[local];
        }

        self.bump.reset();

        mesh.restrict(rhs);
    }
}

impl<'a> Operator<2> for ShiftZOp<'a> {
    fn apply(self: &mut Self, block: &Block<2>, shift: &[f64], dest: &mut [f64]) {
        let shift_rr = self.bump.alloc_slice_fill_default(block.len());
        let shift_zz = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate(&SHIFTZ_SECOND_DERIVATIVE_RHO, shift, shift_rr);
        block
            .axis(1)
            .evaluate(&SHIFTZ_SECOND_DERIVATIVE_Z, shift, shift_zz);

        for (local, _) in block.iter().enumerate() {
            dest[local] = shift_rr[local] + shift_zz[local];
        }

        self.bump.reset();
    }

    fn apply_diag(self: &mut Self, block: &Block<2>, shift: &[f64], dest: &mut [f64]) {
        let shift_rr = self.bump.alloc_slice_fill_default(block.len());
        let shift_zz = self.bump.alloc_slice_fill_default(block.len());

        block
            .axis(0)
            .evaluate_diag(&SHIFTZ_SECOND_DERIVATIVE_RHO, shift, shift_rr);
        block
            .axis(1)
            .evaluate_diag(&SHIFTZ_SECOND_DERIVATIVE_Z, shift, shift_zz);

        for (local, _) in block.iter().enumerate() {
            dest[local] = shift_rr[local] + shift_zz[local];
        }
    }
}

pub fn main() {
    let mesh = UniformMesh::new(
        Rectangle {
            size: [1.0, 1.0],
            origin: [0.0, 0.0],
        },
        [8, 8],
        3,
    );

    // Allocate solution
    let mut solution = vec![0.0; mesh.node_count()];

    // Fill right hand side
    let mut rhs = vec![0.0; mesh.node_count()];

    let block = mesh.level_block(mesh.level_count() - 1);

    for node in block.iter() {
        let index = block.global_from_local(node);
        let position = block.position(node);

        rhs[index] = position[0] + position[1];
    }

    mesh.restrict(&mut rhs);

    // Run solver

    let mut multigrid: UniformMultigrid<'_, 2, BiCGStabSolver> = UniformMultigrid::new(
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

    _ = multigrid;

    // multigrid.solve(&Laplacian { bump: Bump::new() }, &rhs, &mut solution);

    println!("Solved")
}
