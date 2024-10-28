use crate::shared::*;
use aeon_tensor::axisymmetry as axi;

pub fn hyperbolic(sys: HyperbolicSystem, pos: [f64; 2]) -> HyperbolicDerivs {
    let decomp = axi::Decomposition::new(pos, sys.axisymmetric_system());
    let evolve = axi::Decomposition::evolution(&decomp);
    let gauge = axi::Decomposition::gauge(&decomp);
    let scalar = decomp.scalar(sys.scalar_field_system());

    HyperbolicDerivs {
        grr_t: evolve.g[[0, 0]],
        grz_t: evolve.g[[0, 1]],
        gzz_t: evolve.g[[1, 1]],
        s_t: evolve.seed,
        krr_t: evolve.k[[0, 0]],
        krz_t: evolve.k[[0, 1]],
        kzz_t: evolve.k[[1, 1]],
        y_t: evolve.y,
        lapse_t: gauge.lapse,
        shiftr_t: gauge.shift[[0]],
        shiftz_t: gauge.shift[[1]],
        theta_t: evolve.theta,
        zr_t: evolve.z[[0]],
        zz_t: evolve.z[[1]],
        phi_t: scalar.phi,
        pi_t: scalar.pi,
    }
}
