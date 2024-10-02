use std::ops::Range;

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
}

impl HyperbolicSystem {
    pub fn det(&self) -> f64 {
        self.grr * self.gzz - self.grz * self.grz
    }

    pub fn metric(&self) -> Rank2 {
        [[self.grr, self.grz], [self.grz, self.gzz]]
    }

    pub fn metric_par(&self) -> Rank3 {
        let grr_par = [self.grr_r, self.grr_z];
        let grz_par = [self.grz_r, self.grz_z];
        let gzz_par = [self.gzz_r, self.gzz_z];

        [[grr_par, grz_par], [grz_par, gzz_par]]
    }

    pub fn metric_par2(&self) -> Rank4 {
        let grr_par2 = [[self.grr_rr, self.grr_rz], [self.grr_rz, self.grr_zz]];
        let grz_par2 = [[self.grz_rr, self.grz_rz], [self.grz_rz, self.grz_zz]];
        let gzz_par2 = [[self.gzz_rr, self.gzz_rz], [self.gzz_rz, self.gzz_zz]];

        [[grr_par2, grz_par2], [grz_par2, gzz_par2]]
    }

    pub fn seed(&self) -> f64 {
        self.s
    }

    pub fn seed_par(&self) -> Rank1 {
        [self.s_r, self.s_z]
    }

    pub fn seed_par2(&self) -> Rank2 {
        [[self.s_rr, self.s_rz], [self.s_rz, self.s_zz]]
    }

    pub fn extrinsic(&self) -> Rank2 {
        [[self.krr, self.krz], [self.krz, self.kzz]]
    }

    pub fn extrinsic_par(&self) -> Rank3 {
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
