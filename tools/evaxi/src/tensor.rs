use aeon_tensor::axisymmetry as axi;

pub type Rank1 = [f64; 2];
pub type Rank2 = [[f64; 2]; 2];
pub type Rank3 = [[[f64; 2]; 2]; 2];
pub type Rank4 = [[[[f64; 2]; 2]; 2]; 2];

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
    pub fn axisymmetric_system(&self, scalar_fields: &[ScalarFieldSystem]) -> axi::System {
        use aeon_tensor::*;

        let metric = Metric::new(MatrixFieldC2 {
            value: Tensor::from(self.metric()),
            derivs: Tensor::from(self.metric_derivs()),
            second_derivs: Tensor::from(self.metric_second_derivs()),
        });

        let seed = ScalarFieldC2 {
            value: self.seed(),
            derivs: Tensor::from(self.seed_derivs()),
            second_derivs: Tensor::from(self.seed_second_derivs()),
        };

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

        axi::System {
            metric,
            seed,
            k: MatrixFieldC1 {
                value: Tensor::from(self.extrinsic()),
                derivs: Tensor::from(self.extrinsic_derivs()),
            },
            y: ScalarFieldC1 {
                value: self.y,
                derivs: Tensor::from([self.y_r, self.y_z]),
            },
            theta: ScalarFieldC1 {
                value: self.theta,
                derivs: Tensor::from([self.theta_r, self.theta_z]),
            },
            z: VectorFieldC1 {
                value: Tensor::from([self.zr, self.zz]),
                derivs: Tensor::from([[self.zr_r, self.zr_z], [self.zz_r, self.zz_z]]),
            },
            lapse: ScalarFieldC2 {
                value: self.lapse,
                derivs: Tensor::from([self.lapse_r, self.lapse_z]),
                second_derivs: Tensor::from([
                    [self.lapse_rr, self.lapse_rz],
                    [self.lapse_rz, self.lapse_zz],
                ]),
            },
            shift: VectorFieldC1 {
                value: Tensor::from([self.shiftr, self.shiftz]),
                derivs: Tensor::from([
                    [self.shiftr_r, self.shiftr_z],
                    [self.shiftz_r, self.shiftz_z],
                ]),
            },
            source: axi::StressEnergy::vacuum(),
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
        use aeon_tensor::*;

        axi::ScalarFieldSystem {
            phi: ScalarFieldC2 {
                value: self.phi,
                derivs: Tensor::from([self.phi_r, self.phi_z]),
                second_derivs: Tensor::from([
                    [self.phi_rr, self.phi_rz],
                    [self.phi_rz, self.phi_zz],
                ]),
            },
            pi: ScalarFieldC1 {
                value: self.pi,
                derivs: Tensor::from([self.pi_r, self.pi_z]),
            },
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
    let evolve = decomp.evolution();
    let gauge = decomp.gauge();

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
        let scalar = decomp.scalar(scalar_fields[i].axisymmetric_system());

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
