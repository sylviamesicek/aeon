//! This crate provides equations for evolution in axisymmetric
//! numerical relativity. This mainly exists because build scripts
//! must be attached to crates, rather than simple binaries.

#[repr(C)]
#[derive(Clone, Debug)]
pub struct HyperbolicSystem {
    grr: f64,
    grr_r: f64,
    grr_z: f64,
    grr_rr: f64,
    grr_rz: f64,
    grr_zz: f64,

    grz: f64,
    grz_r: f64,
    grz_z: f64,
    grz_rr: f64,
    grz_rz: f64,
    grz_zz: f64,

    gzz: f64,
    gzz_r: f64,
    gzz_z: f64,
    gzz_rr: f64,
    gzz_rz: f64,
    gzz_zz: f64,

    s: f64,
    s_r: f64,
    s_z: f64,
    s_rr: f64,
    s_rz: f64,
    s_zz: f64,

    krr: f64,
    krr_r: f64,
    krr_z: f64,

    krz: f64,
    krz_r: f64,
    krz_z: f64,

    kzz: f64,
    kzz_r: f64,
    kzz_z: f64,

    y: f64,
    y_r: f64,
    y_z: f64,

    lapse: f64,
    lapse_r: f64,
    lapse_z: f64,
    lapse_rr: f64,
    lapse_rz: f64,
    lapse_zz: f64,

    shiftr: f64,
    shiftr_r: f64,
    shiftr_z: f64,

    shiftz: f64,
    shiftz_r: f64,
    shiftz_z: f64,

    theta: f64,
    theta_r: f64,
    theta_z: f64,

    zr: f64,
    zr_r: f64,
    zr_z: f64,

    zz: f64,
    zz_r: f64,
    zz_z: f64,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct HyperbolicDerivs {
    grr_t: f64,
    grz_t: f64,
    gzz_t: f64,
    s_t: f64,

    krr_t: f64,
    krz_t: f64,
    kzz_t: f64,
    y_t: f64,

    theta_t: f64,
    zr_t: f64,
    zz_t: f64,
}

extern "C" {
    pub fn hyperbolic(vars: HyperbolicSystem, rho: f64, z: f64) -> HyperbolicDerivs;
    pub fn hyperbolic_regular(vars: HyperbolicSystem, rho: f64, z: f64) -> HyperbolicDerivs;
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct InitialSystem {
    psi: f64,
    psi_r: f64,
    psi_z: f64,
    psi_rr: f64,
    psi_rz: f64,
    psi_zz: f64,

    s: f64,
    s_r: f64,
    s_z: f64,
    s_rr: f64,
    s_rz: f64,
    s_zz: f64,
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct InitialDerivs {
    psi_t: f64,
}

extern "C" {
    pub fn initial(vars: InitialSystem, rho: f64, z: f64) -> InitialDerivs;
    pub fn initial_regular(vars: InitialSystem, rho: f64, z: f64) -> InitialDerivs;
}
