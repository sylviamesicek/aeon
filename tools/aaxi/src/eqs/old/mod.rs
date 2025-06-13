mod metric;
mod tensor;

use tensor::{Matrix, Metric, Space, Tensor, Tensor3, Tensor4, Vector, lie_derivative};

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

#[derive(Default, Clone)]
struct ScalarFieldSystem {
    /// φ
    pub phi: f64,
    /// ∂ₐφ
    pub phi_derivs: Vector<2>,
    /// ∂ₐ∂ᵦφ
    pub phi_second_derivs: Matrix<2>,
    /// π
    pub pi: f64,
    /// ∂ₐπ
    pub pi_derivs: Vector<2>,
    /// m
    pub mass: f64,
}

/// A decomposition of the stress energy tensor in axisymmetric units.
struct StressEnergy {
    /// σ
    pub energy: f64,
    /// Sₐ
    pub momentum: Vector<2>,
    /// Sₐᵦ
    pub stress: Matrix<2>,
    /// Jᶲ
    pub angular_momentum: f64,
    /// Jᵃ
    pub angular_shear: Vector<2>,
    /// τ
    pub angular_stress: f64,
}

impl StressEnergy {
    /// Stress energy momentum tensor for a vacuum is trivially entirely zeros.
    pub fn vacuum() -> Self {
        Self {
            energy: 0.0,
            momentum: Tensor::zeros(),
            stress: Tensor::zeros(),
            angular_momentum: 0.0,
            angular_shear: Tensor::zeros(),
            angular_stress: 0.0,
        }
    }

    /// Stress energy momentum tensor for a (potentially massive) scalar field.
    pub fn scalar(metric: &Metric<2>, field: ScalarFieldSystem) -> Self {
        let mass = field.mass;
        let pi = field.pi;
        let phi = field.phi;
        let phi_grad = metric.gradient(&field.phi, &field.phi_derivs);
        let phi_grad_trace = metric.cotrace(&(phi_grad * phi_grad));

        let angular_stress = 0.5 * (pi * pi - phi_grad_trace - mass * mass * phi * phi);
        let angular_shear = Tensor::zeros();
        let angular_momentum = 0.0;

        let energy = 0.5 * (pi * pi + phi_grad_trace + mass * mass * phi * phi);
        let momentum = -pi * phi_grad;
        let stress = phi_grad * phi_grad + *metric.value() * angular_stress;

        Self {
            energy,
            momentum,
            stress,
            angular_momentum,
            angular_shear,
            angular_stress,
        }
    }
}

/// All degrees of freedom wrapped in one struct.
struct DynamicalSystem {
    /// g
    metric: Metric<2>,
    /// s
    seed: f64,
    /// ∂ₐs
    seed_partials: Vector<2>,
    /// ∂ₐ∂ᵦs
    seed_second_partials: Matrix<2>,

    /// Kₐᵦ
    k: Matrix<2>,
    /// ∂ᵧKₐᵦ
    k_partials: Tensor3<2>,
    /// Y
    y: f64,
    /// ∂ₐY
    y_partials: Vector<2>,

    /// θ
    theta: f64,
    /// ∂ₐθ
    theta_partials: Vector<2>,
    z: Vector<2>,
    z_partials: Matrix<2>,

    /// α
    lapse: f64,
    /// ∂ₐα
    lapse_partials: Vector<2>,
    /// ∂ₐ∂ᵦα
    lapse_second_partials: Matrix<2>,
    shift: Vector<2>,
    shift_partials: Matrix<2>,

    source: StressEnergy,
}

/// Time derivatives for metric dynamical variables.
struct MetricEvolution {
    pub g: Matrix<2>,
    pub seed: f64,

    pub k: Matrix<2>,
    pub y: f64,

    pub theta: f64,
    pub z: Vector<2>,

    pub debug: Matrix<2>,
}

/// Time derivatives for gauge variables.
struct GaugeEvolution {
    pub lapse: f64,
    pub shift: Vector<2>,
}

/// Time derivative for scalar fields.
struct ScalarFieldEvolution {
    pub phi: f64,
    pub pi: f64,
}

struct Decomposition {
    pos: [f64; 2],

    metric: Metric<2>,
    twist: Twist,

    k: Matrix<2>,
    k_partials: Tensor3<2>,
    l: f64,
    l_partials: Vector<2>,

    theta: f64,
    theta_partials: Vector<2>,
    z: Vector<2>,
    z_partials: Matrix<2>,

    lapse: f64,
    lapse_partials: Vector<2>,
    lapse_second_partials: Matrix<2>,
    shift: Vector<2>,
    shift_partials: Matrix<2>,

    source: StressEnergy,

    y: f64,
}
impl Decomposition {
    /// Construct a decomposition from a position and dynamical system.
    fn new(
        pos: [f64; 2],
        DynamicalSystem {
            metric,
            seed,
            seed_partials,
            seed_second_partials,
            k,
            k_partials,
            y,
            y_partials,
            theta,
            theta_partials,
            z,
            z_partials,
            lapse,
            lapse_partials,
            lapse_second_partials,
            shift,
            shift_partials,
            source,
        }: DynamicalSystem,
    ) -> Self {
        let twist = Twist::new(&metric, pos, seed, seed_partials, seed_second_partials);

        let l = pos[0] * y + k[[0, 0]] / metric.value()[[0, 0]];
        let mut l_partials = Vector::from_fn(|[i]| {
            pos[0] * y_partials[i] + k_partials[[0, 0, i]] / metric.value()[[0, 0]]
                - k[[0, 0]] / metric.value()[[0, 0]] * metric.derivs()[[0, 0, i]]
                    / metric.value()[[0, 0]]
        });
        // From the `ρy` term.
        l_partials[0] += y;

        Self {
            pos,
            metric,
            twist,
            k,
            k_partials,
            l,
            l_partials,
            theta,
            theta_partials,
            z,
            z_partials,
            lapse,
            lapse_partials,
            lapse_second_partials,
            shift,
            shift_partials,
            source,
            y,
        }
    }

    fn on_axis(&self) -> bool {
        self.pos[0] <= ON_AXIS
    }

    fn metric_evolution(&self) -> MetricEvolution {
        // Destructure self
        let Self {
            pos,
            metric,
            twist,
            k,
            k_partials,
            l,
            l_partials,
            theta,
            theta_partials,
            z,
            z_partials,
            lapse,
            lapse_partials,
            lapse_second_partials,
            shift,
            shift_partials,
            source:
                StressEnergy {
                    energy,
                    momentum,
                    stress,
                    angular_momentum: _,
                    angular_shear: _,
                    angular_stress,
                },
            y,
        } = self;

        let on_axis = pos[0].abs() <= ON_AXIS;

        let s = Space::<2>;

        let ricci = metric.ricci();
        let ricci_trace = metric.cotrace(&ricci);

        let k_grad = metric.gradient(k, k_partials);
        let k_trace = metric.cotrace(k);
        let k_trace_grad = s.vector(|i| {
            s.sum(|[m, n]| {
                k_partials[[m, n, i]] * metric.inv()[[m, n]]
                    + k[[m, n]] * metric.inv_derivs()[[m, n, i]]
            })
        });

        let k_con = s
            .matrix(|i, j| s.sum(|[m, n]| metric.inv()[[i, m]] * metric.inv()[[j, n]] * k[[m, n]]));

        let l_grad = metric.gradient(l, l_partials);

        let theta_grad = metric.gradient(theta, theta_partials);
        let z_grad = metric.gradient(z, z_partials);
        let z_con = metric.raise(*z);

        let lapse_grad = metric.gradient(lapse, lapse_partials);
        let lapse_hess = metric.hessian(lapse, lapse_partials, lapse_second_partials);

        let stress_trace = metric.cotrace(stress);

        // *********************************
        // Hamiltonian *********************
        // *********************************

        let hamiltonian = {
            let term1 = 0.5 * (ricci_trace + k_trace * k_trace) + k_trace * l;
            let term2 = -0.5 * s.sum(|[i, j]| k[[i, j]] * k_con[[i, j]]);
            let term3 = -s.sum(|[i, j]| twist.hess()[[i, j]] * metric.inv()[[i, j]]);

            term1 + term2 + term3 - KAPPA * energy
        };

        // **********************************
        // Momentum *************************
        // **********************************

        let momentum = s.vector(|i| {
            let term1 = -k_trace_grad[[i]] - l_grad[[i]];
            let term2 = s.sum(|[m, n]| k_grad[[i, m, n]] * metric.inv()[[m, n]]);

            let mut regular =
                s.sum(|[m]| twist.regular_con()[[m]] * k[[m, i]]) - twist.regular_co()[[i]] * l;

            if on_axis && i == 0 {
                regular += -y;
            } else if on_axis && i == 1 {
                regular += k_partials[[0, 1, 0]] / metric.value()[[0, 0]];
            }

            term1 + term2 + regular - KAPPA * momentum[[i]]
        });

        // ***********************************
        // Metric ****************************
        // ***********************************

        let g_t = {
            let term1 = s.matrix(|i, j| -2.0 * lapse * k[[i, j]]);
            let term2 = metric.killing2(shift, shift_partials);

            term1 + term2
        };

        // (∂ₜλ) / λ
        let mut lam_lt = -lapse * l + s.sum(|[i]| twist.regular_co()[[i]] * shift[[i]]);
        if on_axis {
            lam_lt += shift_partials[[0, 0]];
        }

        // ***********************************
        // Extrinsic curvature ***************
        // ***********************************

        let k_t = {
            let k_lie_shift = lie_derivative(k, k_partials, shift, shift_partials);
            let term1 = *lapse * ricci - *lapse * *twist.hess() - lapse_hess;
            let term2 = lapse * (k_trace + *l) * *k;
            let term3 = -2.0
                * lapse
                * s.matrix(|i, j| s.sum(|[m, n]| k[[i, m]] * metric.inv()[[m, n]] * k[[n, j]]));
            let term4 =
                *lapse * s.matrix(|i, j| z_grad[[i, j]] + z_grad[[j, i]] - 2.0 * k[[i, j]] * theta);

            let term5 = -lapse
                * KAPPA
                * (*stress + 0.5 * (energy - stress_trace - angular_stress) * (*metric.value()));

            term1 + term2 + term3 + term4 + term5 + k_lie_shift
        };

        let debug = {
            //
            lie_derivative(k, k_partials, shift, shift_partials)
        };

        let l_t = {
            let term1 = lapse * l * (k_trace + l - 2.0 * theta);
            let term2 = -lapse * metric.cotrace(twist.hess());
            let term3 = s.sum(|m| shift[m] * l_partials[m]);

            let mut regular =
                s.sum(|[m]| twist.regular_con()[[m]] * (2.0 * lapse * z[m] - lapse_grad[m]));

            if on_axis {
                regular += (2.0 * lapse * z_partials[[0, 0]] - lapse_second_partials[[0, 0]])
                    / metric.value()[[0, 0]];
            };

            let term4 = -0.5 * lapse * KAPPA * (energy - stress_trace + angular_stress);

            term1 + term2 + term3 + term4 + regular
        };

        // ************************************
        // Constraint *************************
        // ************************************

        let theta_t = {
            let term1 = lapse * hamiltonian - lapse * (k_trace + l) * theta;
            let term2 = lapse * metric.cotrace(&z_grad);
            let term3 = -s.sum(|i| lapse_grad[i] * z_con[i]);
            let term4 = s.sum(|[i]| theta_partials[i] * shift[i]);

            let mut regular = lapse * s.sum(|[m]| twist.regular_con()[m] * z[[m]]);
            if on_axis {
                regular += lapse * z_partials[[0, 0]] / metric.value()[[0, 0]];
            }

            term1 + term2 + term3 + term4 + regular
        };

        let z_lie_shift = lie_derivative(z, z_partials, shift, shift_partials);

        let z_t = s.vector(|i| {
            let term1 = lapse * momentum[[i]];
            let term2 = -2.0 * lapse * s.sum(|[m]| k[[i, m]] * z_con[[m]]);
            let term3 = lapse * theta_grad[i] - lapse_grad[[i]] * theta;
            term1 + term2 + term3 + z_lie_shift[i]
        });

        let mut y_t = 0.0;
        let mut seed_t = 0.0;

        if !on_axis {
            y_t = (l_t - k_t[[0, 0]] / metric.value()[[0, 0]]
                + k[[0, 0]] / metric.value()[[0, 0]].powi(2) * g_t[[0, 0]])
                / pos[0];
            seed_t = (lam_lt - 0.5 * g_t[[0, 0]] / metric.value()[[0, 0]]) / pos[0];
        }

        MetricEvolution {
            g: g_t,
            seed: seed_t,

            k: k_t,
            y: y_t,

            theta: theta_t,
            z: z_t,

            debug,
        }
    }

    fn harmonic(&self) -> GaugeEvolution {
        let Self {
            pos,
            metric,
            twist,
            k,
            l,
            theta,
            z,
            lapse,
            lapse_partials,
            shift,
            shift_partials,
            ..
        } = self;

        let on_axis = pos[0].abs() <= ON_AXIS;

        const F: f64 = 1.0;
        const A: f64 = 1.0;
        const MU: f64 = 1.0;
        const D: f64 = 1.0;
        const M: f64 = 2.0;

        let s = Space::<2>;

        let k_trace = s.sum(|[i, j]| k[[i, j]] * metric.inv()[[i, j]]);
        let lapse2 = lapse * lapse;

        let lapse_t = {
            let term1 = -lapse2 * F * (k_trace + l - M * theta);
            let term2 = s.sum(|i| shift[i] * lapse_partials[i]);
            term1 + term2
        };

        let shift_t = s.vector(|i| {
            let lamg_term = 0.5 / metric.det()
                * s.sum(|[m]| metric.inv()[[i, m]] * metric.det_derivs()[[m]])
                + twist.regular_con()[[i]];

            let g_inv_term: f64 = s.sum(|[m]| metric.inv_derivs()[[i, m, m]]);

            let term1 = -lapse2 * MU * g_inv_term;
            let term2 = lapse2 * 2.0 * MU * s.sum(|[m]| metric.inv()[[i, m]] * z[[m]]);

            let term3 = -lapse * A * s.sum(|[m]| metric.inv()[[i, m]] * lapse_partials[m]);
            let term4 = s.sum(|[m]| shift[[m]] * shift_partials[[i, m]]);

            let mut regular = -lapse2 * (2.0 * MU - D) * lamg_term;
            if !on_axis && i == 0 {
                regular += lapse2 * (2.0 * MU - D) * metric.inv()[[0, 0]] / pos[0];
            }

            term1 + term2 + term3 + term4 + regular
        });

        GaugeEvolution {
            lapse: lapse_t,
            shift: shift_t,
        }
    }

    fn harmonic_gauge_zero_shift(&self) -> GaugeEvolution {
        let Self {
            metric,
            k,
            l,
            theta,
            lapse,
            lapse_partials,
            shift,
            ..
        } = self;

        const F: f64 = 1.0;
        const M: f64 = 2.0;

        let s = Space::<2>;

        let k_trace = s.sum(|[i, j]| k[[i, j]] * metric.inv()[[i, j]]);
        let lapse2 = lapse * lapse;
        // let lapse2 = lapse;

        let lapse_t = {
            let term1 = -lapse2 * F * (k_trace + l - M * theta);
            let term2 = s.sum(|i| shift[i] * lapse_partials[i]);
            term1 + term2
        };

        let shift_t = s.vector(|_| 0.0);

        GaugeEvolution {
            lapse: lapse_t,
            shift: shift_t,
        }
    }

    fn log_plus_one(&self) -> GaugeEvolution {
        let Self {
            pos,
            metric,
            twist,
            k,
            l,
            theta,
            z,
            lapse,
            lapse_partials,
            shift,
            shift_partials,
            ..
        } = self;

        let on_axis = pos[0].abs() <= ON_AXIS;

        const F: f64 = 1.0;
        const A: f64 = 1.0;
        const MU: f64 = 1.0;
        const D: f64 = 1.0;
        const M: f64 = 2.0;

        let s = Space::<2>;

        let k_trace = s.sum(|[i, j]| k[[i, j]] * metric.inv()[[i, j]]);
        let lapse2 = lapse * lapse;

        let lapse_t = {
            let term1 = -2.0 * lapse * F * (k_trace + l - M * theta);
            let term2 = s.sum(|i| shift[i] * lapse_partials[i]);
            term1 + term2
        };

        let shift_t = s.vector(|i| {
            let lamg_term = 0.5 / metric.det()
                * s.sum(|[m]| metric.inv()[[i, m]] * metric.det_derivs()[[m]])
                + twist.regular_con()[[i]];

            let g_inv_term: f64 = s.sum(|[m]| metric.inv_derivs()[[i, m, m]]);

            let term1 = -lapse2 * MU * g_inv_term;
            let term2 = lapse2 * 2.0 * MU * s.sum(|[m]| metric.inv()[[i, m]] * z[[m]]);

            let term3 = -lapse * A * s.sum(|[m]| metric.inv()[[i, m]] * lapse_partials[m]);
            let term4 = s.sum(|[m]| shift[[m]] * shift_partials[[i, m]]);

            let mut regular = -lapse2 * (2.0 * MU - D) * lamg_term;
            if !on_axis && i == 0 {
                regular += lapse2 * (2.0 * MU - D) * metric.inv()[[0, 0]] / pos[0];
            }

            term1 + term2 + term3 + term4 + regular
        });

        GaugeEvolution {
            lapse: lapse_t,
            shift: shift_t,
        }
    }

    fn log_plus_one_zero_shift(&self) -> GaugeEvolution {
        let Self {
            metric,
            k,
            l,
            theta,
            lapse,
            lapse_partials,
            shift,
            ..
        } = self;

        let s = Space::<2>;

        let k_trace = s.sum(|[i, j]| k[[i, j]] * metric.inv()[[i, j]]);

        let lapse_t = {
            let term1 = -2.0 * lapse * (k_trace + l - 2.0 * theta);
            let term2 = s.sum(|i| shift[i] * lapse_partials[i]);
            term1 + term2
        };

        let shift_t = s.vector(|_| 0.0);

        GaugeEvolution {
            lapse: lapse_t,
            shift: shift_t,
        }
    }

    fn scalar_field_evolution(&self, field: ScalarFieldSystem) -> ScalarFieldEvolution {
        let s = Space::<2>;

        let Self {
            metric,
            twist,
            k,
            l,
            lapse,
            lapse_partials,
            shift,
            ..
        } = self;
        let phi_hess = metric.hessian(&field.phi, &field.phi_derivs, &field.phi_second_derivs);
        let phi_grad = metric.gradient(&field.phi, &field.phi_derivs);
        let pi_grad = metric.gradient(&field.pi, &field.pi_derivs);

        let lapse_grad = metric.gradient(lapse, &lapse_partials);

        let phi_t = lapse * field.pi + s.sum(|[i]| phi_grad[[i]] * shift[[i]]);

        let pi_t = {
            let term1 =
                lapse * metric.cotrace(&phi_hess) + lapse * field.pi * (metric.cotrace(k) + l);
            let term2 = s.sum(|[i, j]| phi_grad[[i]] * metric.inv()[[i, j]] * lapse_grad[[j]]);

            let mut term3 = lapse * s.sum(|[i]| phi_grad[[i]] * twist.regular_con()[[i]]);
            if self.on_axis() {
                term3 += lapse * field.phi_second_derivs[[0, 0]] / metric.value()[[0, 0]];
            }

            let term4 = -lapse * field.mass * field.mass * field.phi;
            let term5 = s.sum(|[i]| pi_grad[[i]] * shift[[i]]);

            term1 + term2 + term3 + term4 + term5
        };

        ScalarFieldEvolution {
            phi: phi_t,
            pi: pi_t,
        }
    }
}

/// Various tensors relating to the twist degree of freedom in a metric.
struct Twist {
    /// λ⁻¹ ∂ₐ λ
    lam_regular_co: Vector<2>,
    /// λ⁻¹ ∂ᵃ λ
    lam_regular_con: Vector<2>,
    /// λ⁻¹ ∂ₐ∂ᵦ λ
    lam_hess: Matrix<2>,
}

impl Twist {
    /// Computes components of the twist vector from the seed tensor field.
    pub fn new(
        metric: &Metric<2>,
        [r, _z]: [f64; 2],
        seed: f64,
        seed_derivs: Vector<2>,
        seed_second_derivs: Matrix<2>,
    ) -> Self {
        let on_axis = r.abs() <= ON_AXIS;

        let space = Space::<2>;

        // Plus 1/r on axis for r component
        let mut lam_reg_co = Tensor::zeros();
        // Plus 1/(r * grr) on axis for r component
        let mut lam_reg_con = Tensor::zeros();
        // Fully regular
        let mut lam_hess = Tensor::zeros();

        {
            let g_derivs_term =
                Vector::<2>::from_fn(|[i]| metric.derivs()[[0, 0, i]] / metric.value()[[0, 0]]);

            let g_second_derivs_term = Matrix::<2>::from_fn(|[i, j]| {
                metric.second_derivs()[[0, 0, i, j]] / metric.value()[[0, 0]]
                    - metric.derivs()[[0, 0, i]] * metric.derivs()[[0, 0, j]]
                        / (metric.value()[[0, 0]] * metric.value()[[0, 0]])
            });

            // Decompose lam_r into a regular part and an Order(1/r) part.
            let lam_r = seed + r * seed_derivs[[0]] + 0.5 * g_derivs_term[[0]]; // + 1.0 / pos[0]
            let lam_z = r * seed_derivs[[1]] + 0.5 * g_derivs_term[[1]];

            lam_reg_co[[0]] = lam_r;
            lam_reg_co[[1]] = lam_z;

            lam_reg_con[[0]] = metric.inv()[[0, 0]] * lam_r + metric.inv()[[0, 1]] * lam_z;
            lam_reg_con[[1]] = metric.inv()[[1, 0]] * lam_r + metric.inv()[[1, 1]] * lam_z;

            if !on_axis {
                lam_reg_co[[0]] += 1.0 / r;

                lam_reg_con[[0]] += metric.inv()[[0, 0]] / r;
                lam_reg_con[[1]] += metric.inv()[[1, 0]] / r;
            } else {
                lam_reg_con[[1]] += -metric.derivs()[[0, 1, 0]] / metric.det();
            }

            let mut gamma_regular = space
                .matrix(|i, j| space.sum(|[m]| lam_reg_con[[m]] * metric.christoffel()[[m, i, j]]));

            if on_axis {
                gamma_regular[[0, 0]] +=
                    0.5 * metric.second_derivs()[[0, 0, 0, 0]] / metric.value()[[0, 0]];
                gamma_regular[[0, 1]] += 0.0; // + 0.5 * g_par[0][0][1] / (r * g[0][0])
                gamma_regular[[1, 1]] += 0.5
                    * (2.0 * metric.second_derivs()[[0, 1, 0, 1]]
                        - metric.second_derivs()[[1, 1, 0, 0]])
                    / metric.value()[[0, 0]];
            }

            let lam_rr = {
                // Plus a -1/r^2 term that gets cancelled by lam_r * lam_r
                let term1 = 2.0 * seed_derivs[[0]]
                    + r * seed_second_derivs[[0, 0]]
                    + 0.5 * g_second_derivs_term[[0, 0]]; // -1.0 / pos[0].powi(2)
                let term2 = lam_r * lam_r; // + 1.0 / pos[0].powi(2)
                let term3 = if on_axis {
                    // Use lhopital's rule to compute on axis lam_r / r and lam_z / r on axis.
                    let lam_r_lhopital = 2.0 * seed_derivs[[0]]
                        + 0.5 * metric.second_derivs()[[0, 0, 0, 0]] / metric.value()[[0, 0]];
                    2.0 * lam_r_lhopital
                } else {
                    2.0 * lam_r / r
                };

                term1 + term2 + term3 - gamma_regular[[0, 0]]
            };

            let lam_rz = {
                let term1: f64 = seed_derivs[[1]]
                    + r * seed_second_derivs[[0, 1]]
                    + 0.5 * g_second_derivs_term[[0, 1]];
                let term2 = lam_r * lam_z;
                let term3 = if on_axis {
                    seed_derivs[[1]] // + 0.5 * g_par[0][0][1] / (g[0][0] * pos[0]);
                } else {
                    lam_z / r
                };
                term1 + term2 + term3 - gamma_regular[[0, 1]]
            };

            let lam_zz =
                r * seed_second_derivs[[1, 1]] + 0.5 * g_second_derivs_term[[1, 1]] + lam_z * lam_z
                    - gamma_regular[[1, 1]];

            lam_hess[[0, 0]] = lam_rr;
            lam_hess[[0, 1]] = lam_rz;
            lam_hess[[1, 0]] = lam_rz;
            lam_hess[[1, 1]] = lam_zz;
        }

        Self {
            lam_regular_co: lam_reg_co,
            lam_regular_con: lam_reg_con,
            lam_hess,
        }
    }

    pub fn hess(&self) -> &Matrix<2> {
        &self.lam_hess
    }

    pub fn regular_co(&self) -> &Vector<2> {
        &self.lam_regular_co
    }

    pub fn regular_con(&self) -> &Vector<2> {
        &self.lam_regular_con
    }
}

// ***********************************
// Public interface.

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
    fn system(&self, scalar_fields: &[ScalarFieldData]) -> DynamicalSystem {
        let metric = Metric::new(
            self.metric(),
            self.metric_derivs(),
            self.metric_second_derivs(),
        );

        let mut source = StressEnergy::vacuum();

        for sf in scalar_fields {
            let sf = StressEnergy::scalar(&metric, sf.system());

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

        DynamicalSystem {
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
    fn _det(&self) -> f64 {
        self.grr * self.gzz - self.grz * self.grz
    }

    fn metric(&self) -> Matrix<2> {
        Tensor::from([[self.grr, self.grz], [self.grz, self.gzz]])
    }

    fn metric_derivs(&self) -> Tensor3<2> {
        let grr_par = [self.grr_r, self.grr_z];
        let grz_par = [self.grz_r, self.grz_z];
        let gzz_par = [self.gzz_r, self.gzz_z];

        [[grr_par, grz_par], [grz_par, gzz_par]].into()
    }

    fn metric_second_derivs(&self) -> Tensor4<2> {
        let grr_par2 = [[self.grr_rr, self.grr_rz], [self.grr_rz, self.grr_zz]];
        let grz_par2 = [[self.grz_rr, self.grz_rz], [self.grz_rz, self.grz_zz]];
        let gzz_par2 = [[self.gzz_rr, self.gzz_rz], [self.gzz_rz, self.gzz_zz]];

        [[grr_par2, grz_par2], [grz_par2, gzz_par2]].into()
    }

    fn seed(&self) -> f64 {
        self.s
    }

    fn seed_derivs(&self) -> Vector<2> {
        [self.s_r, self.s_z].into()
    }

    fn seed_second_derivs(&self) -> Matrix<2> {
        [[self.s_rr, self.s_rz], [self.s_rz, self.s_zz]].into()
    }

    fn extrinsic(&self) -> Matrix<2> {
        [[self.krr, self.krz], [self.krz, self.kzz]].into()
    }

    fn extrinsic_derivs(&self) -> Tensor3<2> {
        let krr_par = [self.krr_r, self.krr_z];
        let krz_par = [self.krz_r, self.krz_z];
        let kzz_par = [self.kzz_r, self.kzz_z];

        [[krr_par, krz_par], [krz_par, kzz_par]].into()
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

    pub debugrr: f64,
    pub debugrz: f64,
    pub debugzz: f64,
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

    let decomp = Decomposition::new(pos, system.system(scalar_fields));
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

    derivs.debugrr = evolve.debug[[0, 0]];
    derivs.debugrz = evolve.debug[[0, 1]];
    derivs.debugzz = evolve.debug[[1, 1]];

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
    fn metric(&self) -> Matrix<2> {
        Tensor::from([[self.grr, self.grz], [self.grz, self.gzz]])
    }

    fn metric_derivs(&self) -> Tensor3<2> {
        let grr_par = [self.grr_r, self.grr_z];
        let grz_par = [self.grz_r, self.grz_z];
        let gzz_par = [self.gzz_r, self.gzz_z];

        [[grr_par, grz_par], [grz_par, gzz_par]].into()
    }

    fn metric_second_derivs(&self) -> Tensor4<2> {
        Tensor::zeros()
    }

    pub fn seed(&self) -> f64 {
        self.s
    }

    pub fn seed_derivs(&self) -> Vector<2> {
        [self.s_r, self.s_z].into()
    }

    pub fn seed_second_derivs(&self) -> Matrix<2> {
        Tensor::zeros()
    }

    pub fn extrinsic(&self) -> Matrix<2> {
        [[self.krr, self.krz], [self.krz, self.kzz]].into()
    }
}

pub fn horizon(system: HorizonData, [r, z]: [f64; 2]) -> f64 {
    // Prepare space for tensor computations
    const S: Space<2> = Space::<2>;
    // Perpare decomposition
    let metric = Metric::new(
        system.metric(),
        system.metric_derivs(),
        system.metric_second_derivs(),
    );
    let twist = Twist::new(
        &metric,
        [r, z],
        system.seed(),
        system.seed_derivs(),
        system.seed_second_derivs(),
    );
    debug_assert!(metric.det() >= 0.0);
    // Build levi_civita tensor
    let levi_civita = metric.det().sqrt() * Matrix::from([[0., 1.], [-1., 0.]]);
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
    let sigma = S
        .sum(|[i, j]| metric.value()[[i, j]] * tau[[i]] * tau[[j]])
        .sqrt()
        .recip();

    // let tangent = S.vector(|i| sigma * tau[[i]]);
    // sᵅ = σ Hᵅᵝ εᵦᵧ (dxˠ/dθ)
    let normal =
        S.vector(|a| sigma * S.sum(|[b, c]| metric.inv()[[a, b]] * levi_civita[[b, c]] * tau[[c]]));

    // Terms
    let mut div_term = {
        let term1 = S.sum(|[a, b]| levi_civita[[a, b]] * tau2[[a]] * tau[[b]]);
        let term2 = S.sum(|[a, b, c, d]| {
            levi_civita[[a, b]]
                * tau[[b]]
                * tau[[c]]
                * tau[[d]]
                * metric.christoffel_2nd()[[a, c, d]]
        });
        let term3 = S.sum(|[a]| twist.regular_co()[[a]] * normal[[a]]);

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

        div_term +=
            sigma / metric.value()[[0, 0]] * levi_civita[[0, 1]] * (1.0 - tau2[[1]] / radius);
    }

    // Extrinsic curvature terms
    let extrinsic = system.extrinsic();
    let l = extrinsic[[0, 0]] / metric.value()[[0, 0]] + r * system.y;

    let ext_term = {
        let term1 = S.sum(|[a, b]| extrinsic[[a, b]] * normal[[a]] * normal[[b]]);
        let term2 = -metric.cotrace(&extrinsic) - l;

        term1 + term2
    };

    div_term + ext_term
}
