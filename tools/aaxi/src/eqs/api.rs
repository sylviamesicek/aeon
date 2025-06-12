use crate::eqs::{Decomposition, DynamicalSystem, Manifold, ScalarFieldSystem, Twist};
use aeon_tensor::Tensor;
use aeon_tensor::metric::Space;
use aeon_tensor::metric::d2::{
    Matrix, Metric, ScalarC1, ScalarC2, Static as S, Symmetric, SymmetricC1, SymmetricDeriv,
    SymmetricSymmetric, Vector, VectorC1,
};

// ***********************************
// Public interface.

/// Gauge condition to use for evolution.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum GaugeCondition {
    /// Pure generalized harmonic gauge conditions.
    #[default]
    #[serde(rename = "harmonic")]
    Harmonic,
    /// Generalized harmonic gauge with no shift.
    #[serde(rename = "harmonic_zero_shift")]
    HarmonicZeroShift,
    /// Log + 1 slicing with no shift.
    #[serde(rename = "log_plus_one_zero_shift")]
    LogPlusOneZeroShift,
    /// Log + 1 slicing with harmonic shift.
    #[serde(rename = "log_plus_one")]
    LogPlusOne,
}

pub const ON_AXIS: f64 = 1e-10;
pub const KAPPA: f64 = 8.0 * std::f64::consts::PI;

#[repr(C)]
#[derive(Clone, Debug)]
pub struct DynamicalData {
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

impl DynamicalData {
    /// Transforms a struct containing individual field values into an `DynamicalSystem` (stored)
    /// with tensor fields.
    fn system(&self) -> DynamicalSystem {
        let metric = Metric::new(
            self.metric(),
            self.metric_derivs(),
            self.metric_second_derivs(),
        );

        DynamicalSystem {
            metric,
            seed: ScalarC2 {
                value: self.seed(),
                derivs: self.seed_derivs(),
                derivs2: self.seed_second_derivs(),
            },

            k: SymmetricC1 {
                value: self.extrinsic(),
                derivs: self.extrinsic_derivs(),
            },
            y: ScalarC1 {
                value: self.y,
                derivs: [self.y_r, self.y_z].into(),
            },

            theta: ScalarC1 {
                value: self.theta,
                derivs: [self.theta_r, self.theta_z].into(),
            },
            z: VectorC1 {
                value: [self.zr, self.zz].into(),
                derivs: [self.zr_r, self.zr_z, self.zz_r, self.zz_z].into(),
            },

            lapse: ScalarC2 {
                value: self.lapse,
                derivs: [self.lapse_r, self.lapse_z].into(),
                derivs2: [self.lapse_rr, self.lapse_rz, self.lapse_zz].into(),
            },
            shift: VectorC1 {
                value: [self.shiftr, self.shiftz].into(),
                derivs: [self.shiftr_r, self.shiftr_z, self.shiftz_r, self.shiftz_z].into(),
            },
        }
    }

    fn metric(&self) -> Symmetric {
        Symmetric::from([self.grr, self.grz, self.gzz])
    }

    fn metric_derivs(&self) -> SymmetricDeriv {
        Tensor::from([
            self.grr_r, self.grr_z, self.grz_r, self.grz_z, self.gzz_r, self.gzz_z,
        ])
    }

    fn metric_second_derivs(&self) -> SymmetricSymmetric {
        Tensor::from([
            self.grr_rr,
            self.grr_rz,
            self.grr_zz,
            self.grz_rr,
            self.grz_rz,
            self.grz_zz,
            self.gzz_rr,
            self.gzz_rz,
            self.gzz_zz,
        ])
    }

    fn seed(&self) -> f64 {
        self.s
    }

    fn seed_derivs(&self) -> Vector {
        Tensor::from([self.s_r, self.s_z])
    }

    fn seed_second_derivs(&self) -> Symmetric {
        Tensor::from([self.s_rr, self.s_rz, self.s_zz])
    }

    fn extrinsic(&self) -> Symmetric {
        Symmetric::from([self.krr, self.krz, self.kzz])
    }

    fn extrinsic_derivs(&self) -> SymmetricDeriv {
        Tensor::from([
            self.krr_r, self.krr_z, self.krz_r, self.krz_z, self.kzz_r, self.kzz_z,
        ])
    }
}

#[repr(C)]
#[derive(Clone, Default)]
pub struct ScalarFieldData {
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

impl ScalarFieldData {
    fn system(&self) -> ScalarFieldSystem {
        ScalarFieldSystem {
            phi: ScalarC2 {
                value: self.phi,
                derivs: [self.phi_r, self.phi_z].into(),
                derivs2: [self.phi_rr, self.phi_rz, self.phi_zz].into(),
            },
            pi: ScalarC1 {
                value: self.pi,
                derivs: [self.pi_r, self.pi_z].into(),
            },
            mass: self.mass,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct DynamicalDerivs {
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

pub fn evolution(
    system: DynamicalData,
    scalar_fields: &[ScalarFieldData],
    pos: [f64; 2],
    derivs: &mut DynamicalDerivs,
    scalar_field_derivs: &mut [ScalarFieldDerivs],
    gauge: GaugeCondition,
) {
    debug_assert!(scalar_fields.len() == scalar_field_derivs.len());

    let decomp = Decomposition::new(
        pos,
        system.system(),
        scalar_fields.iter().map(|sf| sf.system()),
    );
    let evolve = decomp.metric_evolution();
    let gauge = match gauge {
        GaugeCondition::Harmonic => decomp.harmonic(),
        GaugeCondition::HarmonicZeroShift => decomp.harmonic_gauge_zero_shift(),
        GaugeCondition::LogPlusOne => decomp.log_plus_one(),
        GaugeCondition::LogPlusOneZeroShift => decomp.log_plus_one_zero_shift(),
    };

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
        let scalar = decomp.scalar_field_evolution(scalar_fields[i].system());

        scalar_field_derivs[i].phi = scalar.phi;
        scalar_field_derivs[i].pi = scalar.pi;
    }
}

pub struct HorizonData {
    pub grr: f64,
    pub grz: f64,
    pub gzz: f64,
    pub grr_r: f64,
    pub grr_z: f64,
    pub grz_r: f64,
    pub grz_z: f64,
    pub gzz_r: f64,
    pub gzz_z: f64,
    pub s: f64,
    pub s_r: f64,
    pub s_z: f64,
    pub krr: f64,
    pub krz: f64,
    pub kzz: f64,
    pub y: f64,

    pub theta: f64,
    pub radius: f64,
    pub radius_deriv: f64,
    pub radius_second_deriv: f64,
}

impl HorizonData {
    fn metric(&self) -> Symmetric {
        Tensor::from([self.grr, self.grz, self.gzz])
    }

    fn metric_derivs(&self) -> SymmetricDeriv {
        Tensor::from([
            self.grr_r, self.grr_z, self.grz_r, self.grz_z, self.gzz_r, self.gzz_z,
        ])
    }

    fn metric_second_derivs(&self) -> SymmetricSymmetric {
        Tensor::zeros()
    }

    fn seed(&self) -> f64 {
        self.s
    }

    fn seed_derivs(&self) -> Vector {
        Tensor::from([self.s_r, self.s_z])
    }

    fn seed_second_derivs(&self) -> Symmetric {
        Tensor::zeros()
    }

    fn extrinsic(&self) -> Symmetric {
        Symmetric::from([self.krr, self.krz, self.kzz])
    }
}

pub fn horizon(system: HorizonData, [r, z]: [f64; 2]) -> f64 {
    // Perpare decomposition
    let metric = Metric::new(
        system.metric(),
        system.metric_derivs(),
        system.metric_second_derivs(),
    );
    let manifold = Manifold::new(metric);
    let twist = Twist::new(
        &manifold,
        [r, z],
        ScalarC2 {
            value: system.seed(),
            derivs: system.seed_derivs(),
            derivs2: system.seed_second_derivs(),
        },
    );
    let Manifold {
        metric,
        inv,
        det,
        symbols,
    } = &manifold;

    debug_assert!(det.value >= 0.0);

    // Build levi_civita tensor
    let levi_civita_factor = det.value.sqrt();
    let levi_civita = Matrix::from([0., levi_civita_factor, -levi_civita_factor, 0.]);
    // Decompose xᵐ(θ) into two derivatives
    let theta = system.theta;
    let radius = system.radius;
    let radius_dtheta = system.radius_deriv;
    let radius_ddtheta = system.radius_second_deriv;

    // ρ = R cos(θ)
    // z = R sin(θ)
    let tau = {
        let dr = radius_dtheta * theta.cos() - radius * theta.sin();
        let dz = radius_dtheta * theta.sin() + radius * theta.cos();

        Vector::from([dr, dz])
    };
    let tau2 = {
        let drr =
            radius_ddtheta * theta.cos() - 2.0 * radius_dtheta * theta.sin() - radius * theta.cos();
        let dzz =
            radius_ddtheta * theta.sin() + 2.0 * radius_dtheta * theta.cos() - radius * theta.sin();

        Vector::from([drr, dzz])
    };
    // Normalization factor
    let sigma = S::sum(|[i, j]| metric.value[[i, j]] * tau[[i]] * tau[[j]])
        .sqrt()
        .recip();

    // let tangent = S.vector(|i| sigma * tau[[i]]);
    // sᵅ = σ Hᵅᵝ εᵦᵧ (dxˠ/dθ)
    let normal = S::vector(|[a]| {
        sigma * S::sum(|[b, c]| inv.value[[a, b]] * levi_civita[[b, c]] * tau[[c]])
    });

    // Terms
    let mut div_term = {
        let term1 = S::sum(|[a, b]| levi_civita[[a, b]] * tau2[[a]] * tau[[b]]);
        let term2 = S::sum(|[a, b, c, d]| {
            levi_civita[[a, b]] * tau[[b]] * tau[[c]] * tau[[d]] * symbols.second_kind[[a, c, d]]
        });
        let term3 = S::sum(|[a]| twist.regular_co()[[a]] * normal[[a]]);

        -sigma.powi(3) * (term1 + term2) + term3
    };

    // sᵅ = σ Hᵅᵝ εᵦᵧ (dxˠ/dθ)
    // sʳ = σ Hʳᵝ εᵦᵧ (dxˠ/dθ) = σ / Hᵣᵣ εᵣᵧ (dxˠ/dθ) = σ / Hᵣᵣ εᵣₜ (dz/dθ)
    if r.abs() <= ON_AXIS {
        // // I think this reduces to zero
        // let sigma_r =
        //     -0.5 * sigma.powi(3) * S.sum(|[a, b]| metric.derivs()[[a, b, 0]] * tau[[a]] * tau[[b]]);

        // div_lambda_term +=
        //     sigma_r * S.sum(|[b, c]| metric.inv()[[0, b]] * levi_civita[[b, c]] * tau[[c]]);
        // div_lambda_term += metric.det_derivs()[[]]

        // I think everything becomes zero except for the dz/dθ / r term.
        // This in turn becomes
        // (dR/dθ sin θ + R cos θ) / R cos θ
        // 1 + (dR/dθ / cos θ) (sin θ / R)
        // Both dR/dθ and cos θ approach 0 on axis, therefore this term becomes d²R/dθ² / (-sin θ).
        // This gives
        // 1/r dz/dθ = 1 - d²R/dθ² / R

        div_term += sigma / metric.value[[0, 0]] * levi_civita[[0, 1]] * (1.0 - tau2[[1]] / radius);
    }

    // Extrinsic curvature terms
    let extrinsic = system.extrinsic();
    let l = extrinsic[[0, 0]] / metric.value[[0, 0]] + r * system.y;

    let ext_term = {
        let term1 = S::sum(|[a, b]| extrinsic[[a, b]] * normal[[a]] * normal[[b]]);
        let term2 = -inv.cotrace(&extrinsic) - l;

        term1 + term2
    };

    div_term + ext_term
}
