//! Rust bindings to automatiically generated C code.

use crate::shared::*;

extern "C" {
    fn hyperbolic_sys(vars: HyperbolicSystem, rho: f64, z: f64) -> HyperbolicDerivs;
    fn hyperbolic_regular_sys(vars: HyperbolicSystem, rho: f64, z: f64) -> HyperbolicDerivs;
    fn geometric_sys(vars: HyperbolicSystem, rho: f64, z: f64) -> Geometric;
}

pub fn hyperbolic(vars: HyperbolicSystem, [rho, z]: [f64; 2]) -> HyperbolicDerivs {
    let on_axis = rho.abs() <= 10e-10;
    unsafe {
        if on_axis {
            hyperbolic_regular_sys(vars, rho, z)
        } else {
            hyperbolic_sys(vars, rho, z)
        }
    }
}

pub fn geometric(vars: HyperbolicSystem, [rho, z]: [f64; 2]) -> Geometric {
    unsafe { geometric_sys(vars, rho, z) }
}
