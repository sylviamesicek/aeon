use std::ops::Range;

use aeon_tensor::axisymmetry as axi;
use rand::Rng;

pub type Rank1 = [f64; 2];
pub type Rank2 = [[f64; 2]; 2];
pub type Rank3 = [[[f64; 2]; 2]; 2];
pub type Rank4 = [[[[f64; 2]; 2]; 2]; 2];

#[repr(C)]
#[derive(Clone, Debug)]
pub struct HyperbolicSystem {
    pub grr: f64,
    pub grr_r: f64,
    pub grr_z: f64,
    pub grr_rr: f64,
    pub grr_rz: f64,
    pub grr_zz: f64,

    pub grz: f64,
    pub grz_r: f64,
    pub grz_z: f64,
    pub grz_rr: f64,
    pub grz_rz: f64,
    pub grz_zz: f64,

    pub gzz: f64,
    pub gzz_r: f64,
    pub gzz_z: f64,
    pub gzz_rr: f64,
    pub gzz_rz: f64,
    pub gzz_zz: f64,

    pub s: f64,
    pub s_r: f64,
    pub s_z: f64,
    pub s_rr: f64,
    pub s_rz: f64,
    pub s_zz: f64,

    pub krr: f64,
    pub krr_r: f64,
    pub krr_z: f64,

    pub krz: f64,
    pub krz_r: f64,
    pub krz_z: f64,

    pub kzz: f64,
    pub kzz_r: f64,
    pub kzz_z: f64,

    pub y: f64,
    pub y_r: f64,
    pub y_z: f64,

    pub lapse: f64,
    pub lapse_r: f64,
    pub lapse_z: f64,
    pub lapse_rr: f64,
    pub lapse_rz: f64,
    pub lapse_zz: f64,

    pub shiftr: f64,
    pub shiftr_r: f64,
    pub shiftr_z: f64,

    pub shiftz: f64,
    pub shiftz_r: f64,
    pub shiftz_z: f64,

    pub theta: f64,
    pub theta_r: f64,
    pub theta_z: f64,

    pub zr: f64,
    pub zr_r: f64,
    pub zr_z: f64,

    pub zz: f64,
    pub zz_r: f64,
    pub zz_z: f64,

    pub phi: f64,
    pub phi_r: f64,
    pub phi_z: f64,
    pub phi_rr: f64,
    pub phi_rz: f64,
    pub phi_zz: f64,

    pub pi: f64,
    pub pi_r: f64,
    pub pi_z: f64,

    pub mass: f64,
}

impl HyperbolicSystem {
    /// Generates a valid hyperbolic system for fuzzy testing.
    pub fn fuzzy(rng: &mut impl Rng) -> Self {
        const NEAR_ONE: Range<f64> = 0.2..1.8;
        const NEAR_ZERO: Range<f64> = -1.0..1.0;

        let system = HyperbolicSystem {
            grr: rng.gen_range(NEAR_ONE),
            grr_r: rng.gen_range(NEAR_ZERO),
            grr_z: rng.gen_range(NEAR_ZERO),
            grr_rr: rng.gen_range(NEAR_ZERO),
            grr_rz: rng.gen_range(NEAR_ZERO),
            grr_zz: rng.gen_range(NEAR_ZERO),
            grz: rng.gen_range(NEAR_ZERO),
            grz_r: rng.gen_range(NEAR_ZERO),
            grz_z: rng.gen_range(NEAR_ZERO),
            grz_rr: rng.gen_range(NEAR_ZERO),
            grz_rz: rng.gen_range(NEAR_ZERO),
            grz_zz: rng.gen_range(NEAR_ZERO),
            gzz: rng.gen_range(NEAR_ONE),
            gzz_r: rng.gen_range(NEAR_ZERO),
            gzz_z: rng.gen_range(NEAR_ZERO),
            gzz_rr: rng.gen_range(NEAR_ZERO),
            gzz_rz: rng.gen_range(NEAR_ZERO),
            gzz_zz: rng.gen_range(NEAR_ZERO),
            s: rng.gen_range(NEAR_ZERO),
            s_r: rng.gen_range(NEAR_ZERO),
            s_z: rng.gen_range(NEAR_ZERO),
            s_rr: rng.gen_range(NEAR_ZERO),
            s_rz: rng.gen_range(NEAR_ZERO),
            s_zz: rng.gen_range(NEAR_ZERO),

            krr: rng.gen_range(NEAR_ZERO),
            krr_r: rng.gen_range(NEAR_ZERO),
            krr_z: rng.gen_range(NEAR_ZERO),
            krz: rng.gen_range(NEAR_ZERO),
            krz_r: rng.gen_range(NEAR_ZERO),
            krz_z: rng.gen_range(NEAR_ZERO),
            kzz: rng.gen_range(NEAR_ZERO),
            kzz_r: rng.gen_range(NEAR_ZERO),
            kzz_z: rng.gen_range(NEAR_ZERO),
            y: rng.gen_range(NEAR_ZERO),
            y_r: rng.gen_range(NEAR_ZERO),
            y_z: rng.gen_range(NEAR_ZERO),

            lapse: rng.gen_range(NEAR_ONE),
            lapse_r: rng.gen_range(NEAR_ZERO),
            lapse_z: rng.gen_range(NEAR_ZERO),
            lapse_rr: rng.gen_range(NEAR_ZERO),
            lapse_rz: rng.gen_range(NEAR_ZERO),
            lapse_zz: rng.gen_range(NEAR_ZERO),
            shiftr: rng.gen_range(NEAR_ZERO),
            shiftr_r: rng.gen_range(NEAR_ZERO),
            shiftr_z: rng.gen_range(NEAR_ZERO),
            shiftz: rng.gen_range(NEAR_ZERO),
            shiftz_r: rng.gen_range(NEAR_ZERO),
            shiftz_z: rng.gen_range(NEAR_ZERO),

            theta: rng.gen_range(NEAR_ZERO),
            theta_r: rng.gen_range(NEAR_ZERO),
            theta_z: rng.gen_range(NEAR_ZERO),
            zr: rng.gen_range(NEAR_ZERO),
            zr_r: rng.gen_range(NEAR_ZERO),
            zr_z: rng.gen_range(NEAR_ZERO),
            zz: rng.gen_range(NEAR_ZERO),
            zz_r: rng.gen_range(NEAR_ZERO),
            zz_z: rng.gen_range(NEAR_ZERO),

            phi: rng.gen_range(NEAR_ZERO),
            phi_r: rng.gen_range(NEAR_ZERO),
            phi_z: rng.gen_range(NEAR_ZERO),
            phi_rr: rng.gen_range(NEAR_ZERO),
            phi_rz: rng.gen_range(NEAR_ZERO),
            phi_zz: rng.gen_range(NEAR_ZERO),

            pi: rng.gen_range(NEAR_ZERO),
            pi_r: rng.gen_range(NEAR_ZERO),
            pi_z: rng.gen_range(NEAR_ZERO),

            mass: rng.gen_range(NEAR_ONE),
        };

        system
    }

    pub fn axisymmetric_system(&self) -> axi::System {
        use aeon_tensor::*;

        let metric = Metric::new(TensorFieldC2 {
            value: Tensor::from_storage(self.metric()),
            derivs: Tensor::from_storage(self.metric_derivs()),
            second_derivs: Tensor::from_storage(self.metric_second_derivs()),
        });

        let seed = TensorFieldC2 {
            value: Tensor::from_storage(self.seed()),
            derivs: Tensor::from_storage(self.seed_derivs()),
            second_derivs: Tensor::from_storage(self.seed_second_derivs()),
        };

        axi::System {
            metric,
            seed,
            k: TensorFieldC1 {
                value: Tensor::from_storage(self.extrinsic()),
                derivs: Tensor::from_storage(self.extrinsic_derivs()),
            },
            y: TensorFieldC1 {
                value: Tensor::from_storage(self.y),
                derivs: Tensor::from_storage([self.y_r, self.y_z]),
            },
            theta: TensorFieldC1 {
                value: Tensor::from_storage(self.theta),
                derivs: Tensor::from_storage([self.theta_r, self.theta_z]),
            },
            z: TensorFieldC1 {
                value: Tensor::from_storage([self.zr, self.zz]),
                derivs: Tensor::from_storage([[self.zr_r, self.zr_z], [self.zz_r, self.zz_z]]),
            },
            lapse: TensorFieldC2 {
                value: Tensor::from_storage(self.lapse),
                derivs: Tensor::from_storage([self.lapse_r, self.lapse_z]),
                second_derivs: Tensor::from_storage([
                    [self.lapse_rr, self.lapse_rz],
                    [self.lapse_rz, self.lapse_zz],
                ]),
            },
            shift: TensorFieldC1 {
                value: Tensor::from_storage([self.shiftr, self.shiftz]),
                derivs: Tensor::from_storage([
                    [self.shiftr_r, self.shiftr_z],
                    [self.shiftz_r, self.shiftz_z],
                ]),
            },
            source: axi::StressEnergy::vacuum(),
        }
    }

    pub fn scalar_field_system(&self) -> axi::ScalarFieldSystem {
        use aeon_tensor::*;

        axi::ScalarFieldSystem {
            phi: TensorFieldC2 {
                value: Tensor::from_storage(self.phi),
                derivs: Tensor::from_storage([self.phi_r, self.phi_z]),
                second_derivs: Tensor::from_storage([
                    [self.phi_rr, self.phi_rz],
                    [self.phi_rz, self.phi_zz],
                ]),
            },
            pi: TensorFieldC1 {
                value: Tensor::from_storage(self.pi),
                derivs: Tensor::from_storage([self.pi_r, self.pi_z]),
            },
            mass: self.mass,
        }
    }

    pub fn det(&self) -> f64 {
        self.grr * self.gzz - self.grz * self.grz
    }

    pub fn metric(&self) -> Rank2 {
        [[self.grr, self.grz], [self.grz, self.gzz]]
    }

    pub fn metric_derivs(&self) -> Rank3 {
        let grr_par = [self.grr_r, self.grr_z];
        let grz_par = [self.grz_r, self.grz_z];
        let gzz_par = [self.gzz_r, self.gzz_z];

        [[grr_par, grz_par], [grz_par, gzz_par]]
    }

    pub fn metric_second_derivs(&self) -> Rank4 {
        let grr_par2 = [[self.grr_rr, self.grr_rz], [self.grr_rz, self.grr_zz]];
        let grz_par2 = [[self.grz_rr, self.grz_rz], [self.grz_rz, self.grz_zz]];
        let gzz_par2 = [[self.gzz_rr, self.gzz_rz], [self.gzz_rz, self.gzz_zz]];

        [[grr_par2, grz_par2], [grz_par2, gzz_par2]]
    }

    pub fn seed(&self) -> f64 {
        self.s
    }

    pub fn seed_derivs(&self) -> Rank1 {
        [self.s_r, self.s_z]
    }

    pub fn seed_second_derivs(&self) -> Rank2 {
        [[self.s_rr, self.s_rz], [self.s_rz, self.s_zz]]
    }

    pub fn extrinsic(&self) -> Rank2 {
        [[self.krr, self.krz], [self.krz, self.kzz]]
    }

    pub fn extrinsic_derivs(&self) -> Rank3 {
        let krr_par = [self.krr_r, self.krr_z];
        let krz_par = [self.krz_r, self.krz_z];
        let kzz_par = [self.kzz_r, self.kzz_z];

        [[krr_par, krz_par], [krz_par, kzz_par]]
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct HyperbolicDerivs {
    pub grr_t: f64,
    pub grz_t: f64,
    pub gzz_t: f64,
    pub s_t: f64,

    pub krr_t: f64,
    pub krz_t: f64,
    pub kzz_t: f64,
    pub y_t: f64,

    pub lapse_t: f64,
    pub shiftr_t: f64,
    pub shiftz_t: f64,

    pub theta_t: f64,
    pub zr_t: f64,
    pub zz_t: f64,

    pub phi_t: f64,
    pub pi_t: f64,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Geometric {
    pub ricci_rr: f64,
    pub ricci_rz: f64,
    pub ricci_zz: f64,

    pub gamma_rrr: f64,
    pub gamma_rrz: f64,
    pub gamma_rzz: f64,

    pub gamma_zrr: f64,
    pub gamma_zrz: f64,
    pub gamma_zzz: f64,
}
