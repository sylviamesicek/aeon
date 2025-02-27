use aeon_tensor::{Matrix, Tensor, Tensor3, Tensor4, Vector};
use sharedaxi::eqs as axi;

/// All degrees of freedom re
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
}

impl HyperbolicSystem {
    /// Transforms a struct containing individual field values into an `DynamicalSystem` (stored)
    /// with tensor fields.
    pub fn axisymmetric_system(&self, scalar_fields: &[ScalarFieldSystem]) -> axi::DynamicalSystem {
        use aeon_tensor::*;

        let metric = Metric::new(
            self.metric(),
            self.metric_derivs(),
            self.metric_second_derivs(),
        );

        let mut source = axi::StressEnergy::vacuum();

        for sf in scalar_fields {
            let sf = axi::StressEnergy::scalar(&metric, sf.axisymmetric_system());

            source.energy += sf.energy;
            source.momentum[0] += sf.momentum[0];
            source.momentum[1] += sf.momentum[1];

            source.stress[[0, 0]] += sf.stress[[0, 0]];
            source.stress[[0, 1]] += sf.stress[[0, 1]];
            source.stress[[1, 1]] += sf.stress[[1, 1]];
            source.stress[[1, 0]] += sf.stress[[1, 0]];

            source.angular_momentum += sf.angular_momentum;
            source.angular_shear[[0]] += sf.angular_shear[[0]];
            source.angular_shear[[1]] += sf.angular_shear[[1]];
            source.angular_stress += sf.angular_stress;
        }

        axi::DynamicalSystem {
            metric,
            seed: self.seed(),
            seed_partials: self.seed_derivs(),
            seed_second_partials: self.seed_second_derivs(),

            k: self.extrinsic(),
            k_partials: self.extrinsic_derivs(),
            y: self.y,
            y_partials: [self.y_r, self.y_z].into(),

            theta: self.theta,
            theta_partials: [self.theta_r, self.theta_z].into(),
            z: [self.zr, self.zz].into(),
            z_partials: [[self.zr_r, self.zr_z], [self.zz_r, self.zz_z]].into(),

            lapse: self.lapse,
            lapse_partials: [self.lapse_r, self.lapse_z].into(),
            lapse_second_partials: [
                [self.lapse_rr, self.lapse_rz],
                [self.lapse_rz, self.lapse_zz],
            ]
            .into(),

            shift: [self.shiftr, self.shiftz].into(),
            shift_partials: [
                [self.shiftr_r, self.shiftr_z],
                [self.shiftz_r, self.shiftz_z],
            ]
            .into(),

            source,
        }
    }

    /// Computes the determinant of the metric.
    pub fn det(&self) -> f64 {
        self.grr * self.gzz - self.grz * self.grz
    }

    pub fn metric(&self) -> Matrix<2> {
        Tensor::from([[self.grr, self.grz], [self.grz, self.gzz]])
    }

    pub fn metric_derivs(&self) -> Tensor3<2> {
        let grr_par = [self.grr_r, self.grr_z];
        let grz_par = [self.grz_r, self.grz_z];
        let gzz_par = [self.gzz_r, self.gzz_z];

        [[grr_par, grz_par], [grz_par, gzz_par]].into()
    }

    pub fn metric_second_derivs(&self) -> Tensor4<2> {
        let grr_par2 = [[self.grr_rr, self.grr_rz], [self.grr_rz, self.grr_zz]];
        let grz_par2 = [[self.grz_rr, self.grz_rz], [self.grz_rz, self.grz_zz]];
        let gzz_par2 = [[self.gzz_rr, self.gzz_rz], [self.gzz_rz, self.gzz_zz]];

        [[grr_par2, grz_par2], [grz_par2, gzz_par2]].into()
    }

    pub fn seed(&self) -> f64 {
        self.s
    }

    pub fn seed_derivs(&self) -> Vector<2> {
        [self.s_r, self.s_z].into()
    }

    pub fn seed_second_derivs(&self) -> Matrix<2> {
        [[self.s_rr, self.s_rz], [self.s_rz, self.s_zz]].into()
    }

    pub fn extrinsic(&self) -> Matrix<2> {
        [[self.krr, self.krz], [self.krz, self.kzz]].into()
    }

    pub fn extrinsic_derivs(&self) -> Tensor3<2> {
        let krr_par = [self.krr_r, self.krr_z];
        let krz_par = [self.krz_r, self.krz_z];
        let kzz_par = [self.kzz_r, self.kzz_z];

        [[krr_par, krz_par], [krz_par, kzz_par]].into()
    }
}

#[repr(C)]
#[derive(Clone, Default)]
pub struct ScalarFieldSystem {
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

impl ScalarFieldSystem {
    pub fn axisymmetric_system(&self) -> axi::ScalarFieldSystem {
        axi::ScalarFieldSystem {
            phi: self.phi,
            phi_derivs: [self.phi_r, self.phi_z].into(),
            phi_second_derivs: [[self.phi_rr, self.phi_rz], [self.phi_rz, self.phi_zz]].into(),

            pi: self.pi,
            pi_derivs: [self.pi_r, self.pi_z].into(),
            mass: self.mass,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
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
}

#[derive(Clone, Default)]
pub struct ScalarFieldDerivs {
    pub phi: f64,
    pub pi: f64,
}

pub fn hyperbolic(
    system: HyperbolicSystem,
    scalar_fields: &[ScalarFieldSystem],
    pos: [f64; 2],
    derivs: &mut HyperbolicDerivs,
    scalar_field_derivs: &mut [ScalarFieldDerivs],
) {
    debug_assert!(scalar_fields.len() == scalar_field_derivs.len());

    let decomp = axi::Decomposition::new(pos, system.axisymmetric_system(scalar_fields));
    let evolve = decomp.metric_evolution();
    let gauge = decomp.gauge_evolution();

    derivs.grr_t = evolve.g[[0, 0]];
    derivs.grz_t = evolve.g[[0, 1]];
    derivs.gzz_t = evolve.g[[1, 1]];
    derivs.s_t = evolve.seed;
    derivs.krr_t = evolve.k[[0, 0]];
    derivs.krz_t = evolve.k[[0, 1]];
    derivs.kzz_t = evolve.k[[1, 1]];
    derivs.y_t = evolve.y;
    derivs.lapse_t = gauge.lapse;
    derivs.shiftr_t = gauge.shift[[0]];
    derivs.shiftz_t = gauge.shift[[1]];
    derivs.theta_t = evolve.theta;
    derivs.zr_t = evolve.z[[0]];
    derivs.zz_t = evolve.z[[1]];

    for i in 0..scalar_fields.len() {
        let scalar = decomp.scalar_field_evolution(scalar_fields[i].axisymmetric_system());

        scalar_field_derivs[i].phi = scalar.phi;
        scalar_field_derivs[i].pi = scalar.pi;
    }
}

// pub fn hyperbolic(sys: HyperbolicSystem, pos: [f64; 2]) -> HyperbolicDerivs {
//     let decomp = axi::Decomposition::new(pos, sys.axisymmetric_system());
//     let evolve = axi::Decomposition::evolution(&decomp);
//     let gauge = axi::Decomposition::gauge(&decomp);

//     HyperbolicDerivs {
//         grr_t: evolve.g[[0, 0]],
//         grz_t: evolve.g[[0, 1]],
//         gzz_t: evolve.g[[1, 1]],
//         s_t: evolve.seed,
//         krr_t: evolve.k[[0, 0]],
//         krz_t: evolve.k[[0, 1]],
//         kzz_t: evolve.k[[1, 1]],
//         y_t: evolve.y,
//         lapse_t: gauge.lapse,
//         shiftr_t: gauge.shift[[0]],
//         shiftz_t: gauge.shift[[1]],
//         theta_t: evolve.theta,
//         zr_t: evolve.z[[0]],
//         zz_t: evolve.z[[1]],
//     }
// }
