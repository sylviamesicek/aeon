use aeon_tensor::Tensor;
use aeon_tensor::metric::Space;
use aeon_tensor::metric::d2::{
    ChristoffelSymbol, Metric, MetricDet, MetricInv, ScalarC1, ScalarC2, Static as S, Symmetric,
    SymmetricC1, Vector, VectorC1,
};

mod api;
pub mod old;

// pub use api::*;
pub use old::{
    DynamicalData, DynamicalDerivs, GaugeCondition, HorizonData, KAPPA, ON_AXIS, ScalarFieldData,
    ScalarFieldDerivs, evolution, horizon,
};

/// All degrees of freedom wrapped in one struct.
struct DynamicalSystem {
    /// g
    metric: Metric,
    /// s
    seed: ScalarC2,

    /// Kₐᵦ
    k: SymmetricC1,
    /// Y
    y: ScalarC1,

    /// θ
    theta: ScalarC1,
    /// Z
    z: VectorC1,

    /// α
    lapse: ScalarC2,
    shift: VectorC1,
}

struct Manifold {
    metric: Metric,
    inv: MetricInv,
    det: MetricDet,
    symbols: ChristoffelSymbol,
}

impl Manifold {
    fn new(metric: Metric) -> Self {
        let det = metric.det();
        let inv = metric.inv(&det);
        let symbols = metric.chirstoffel_symbol(&inv);
        Self {
            metric,
            inv,
            det,
            symbols,
        }
    }
}

/// Various tensors relating to the twist degree of freedom in a metric.
struct Twist {
    /// λ⁻¹ ∂ₐ λ
    lam_regular_co: Vector,
    /// λ⁻¹ ∂ᵃ λ
    lam_regular_con: Vector,
    /// λ⁻¹ ∂ₐ∂ᵦ λ
    lam_hess: Symmetric,
}

impl Twist {
    /// Computes components of the twist vector from the seed tensor field.
    fn new(
        Manifold {
            metric,
            inv,
            symbols,
            ..
        }: &Manifold,
        [r, _z]: [f64; 2],
        seed: ScalarC2,
    ) -> Self {
        let on_axis = r.abs() <= ON_AXIS;

        // Plus 1/r on axis for r component
        let mut lam_reg_co = Tensor::zeros();
        // Plus 1/(r * grr) on axis for r component
        let mut lam_reg_con = Tensor::zeros();
        // Fully regular
        let mut lam_hess = Tensor::zeros();

        {
            let g_derivs_term =
                Vector::from_fn(|[i]| metric.derivs[[0, 0, i]] / metric.value[[0, 0]]);

            let g_second_derivs_term = Symmetric::from_fn(|[i, j]| {
                metric.derivs2[[0, 0, i, j]] / metric.value[[0, 0]]
                    - metric.derivs[[0, 0, i]] * metric.derivs[[0, 0, j]]
                        / (metric.value[[0, 0]].powi(2))
            });

            // Decompose lam_r into a regular part and an Order(1/r) part.
            let lam_r = seed.value + r * seed.derivs[[0]] + 0.5 * g_derivs_term[[0]]; // + 1.0 / pos[0]
            let lam_z = r * seed.derivs[[1]] + 0.5 * g_derivs_term[[1]];

            lam_reg_co[[0]] = lam_r;
            lam_reg_co[[1]] = lam_z;

            lam_reg_con[[0]] = inv.value[[0, 0]] * lam_r + inv.value[[0, 1]] * lam_z;
            lam_reg_con[[1]] = inv.value[[1, 0]] * lam_r + inv.value[[1, 1]] * lam_z;

            if !on_axis {
                lam_reg_co[[0]] += 1.0 / r;

                lam_reg_con[[0]] += inv.value[[0, 0]] / r;
                lam_reg_con[[1]] += inv.value[[1, 0]] / r;
            } else {
                lam_reg_con[[1]] += inv.derivs[[1, 0, 0]];
            }

            let mut gamma_regular = Symmetric::from_fn(|[i, j]| {
                S::sum(|[m]| lam_reg_con[[m]] * symbols.first_kind[[m, i, j]])
            });

            if on_axis {
                gamma_regular[[0, 0]] += 0.5 * metric.derivs2[[0, 0, 0, 0]] / metric.value[[0, 0]];
                gamma_regular[[1, 0]] += 0.0; // + 0.5 * g_par[0][0][1] / (r * g[0][0])
                gamma_regular[[1, 1]] += 0.5
                    * (2.0 * metric.derivs2[[0, 1, 0, 1]] - metric.derivs2[[1, 1, 0, 0]])
                    / metric.value[[0, 0]];
            }

            let lam_rr = {
                let term1 = 2.0 * seed.derivs[[0]]
                    + r * seed.derivs2[[0, 0]]
                    + 0.5 * g_second_derivs_term[[0, 0]]; // -1.0 / pos[0].powi(2)
                let term2 = lam_r * lam_r; // + 1.0 / pos[0].powi(2)
                let term3 = if on_axis {
                    // Use lhopital's rule to compute on axis lam_r / r and lam_z / r on axis.
                    let lam_r_lhopital = 2.0 * seed.derivs[[0]]
                        + 0.5 * metric.derivs2[[0, 0, 0, 0]] / metric.value[[0, 0]];
                    2.0 * lam_r_lhopital
                } else {
                    2.0 * lam_r / r
                };

                term1 + term2 + term3 - gamma_regular[[0, 0]]
            };

            let lam_rz = {
                let term1: f64 = seed.derivs[[1]]
                    + r * seed.derivs2[[0, 1]]
                    + 0.5 * g_second_derivs_term[[0, 1]];
                let term2 = lam_r * lam_z;
                let term3 = if on_axis {
                    seed.derivs[[1]] // + 0.5 * g_par[0][0][1] / (g[0][0] * pos[0]);
                } else {
                    lam_z / r
                };
                term1 + term2 + term3 - gamma_regular[[1, 0]]
            };

            let lam_zz =
                r * seed.derivs2[[1, 1]] + 0.5 * g_second_derivs_term[[1, 1]] + lam_z * lam_z
                    - gamma_regular[[1, 1]];

            lam_hess[[0, 0]] = lam_rr;
            lam_hess[[1, 0]] = lam_rz;
            lam_hess[[1, 1]] = lam_zz;
        }

        Self {
            lam_regular_co: lam_reg_co,
            lam_regular_con: lam_reg_con,
            lam_hess,
        }
    }

    fn hess(&self) -> &Symmetric {
        &self.lam_hess
    }

    fn regular_co(&self) -> &Vector {
        &self.lam_regular_co
    }

    fn regular_con(&self) -> &Vector {
        &self.lam_regular_con
    }
}

#[derive(Default, Clone)]
struct ScalarFieldSystem {
    /// φ
    phi: ScalarC2,
    /// π
    pi: ScalarC1,
    /// m
    mass: f64,
}

/// A decomposition of the stress energy tensor in axisymmetric units.
struct StressEnergy {
    /// σ
    energy: f64,
    /// Sₐ
    momentum: Vector,
    /// Sₐᵦ
    stress: Symmetric,
    /// Jᶲ
    angular_momentum: f64,
    /// Jᵃ
    angular_shear: Vector,
    /// τ
    angular_stress: f64,
}

impl StressEnergy {
    /// Stress energy momentum tensor for a vacuum is trivially entirely zeros.
    fn vacuum() -> Self {
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
    fn scalar(m: &Manifold, field: ScalarFieldSystem) -> Self {
        let mass = field.mass;
        let pi = field.pi.value;
        let phi = field.phi.value;
        let phi_grad = field.phi.gradient(&m.symbols);
        let phi_grad_trace = S::sum(|[a, b]| phi_grad[[a]] * m.inv.value[[a, b]] * phi_grad[[b]]);

        let angular_stress = 0.5 * (pi.powi(2) - phi_grad_trace - mass.powi(2) * phi.powi(2));
        let angular_shear = Tensor::zeros();
        let angular_momentum = 0.0;

        let energy = 0.5 * (pi * pi + phi_grad_trace + mass * mass * phi * phi);
        let momentum = S::vector(|idx| -pi * phi_grad[idx]);
        let stress = S::symmetric(|[a, b]| {
            phi_grad[[a]] * phi_grad[[b]] + m.metric.value[[a, b]] * angular_stress
        });

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

/// Time derivatives for metric dynamical variables.
struct MetricEvolution {
    g: Symmetric,
    seed: f64,

    k: Symmetric,
    y: f64,

    theta: f64,
    z: Vector,
}

/// Time derivatives for gauge variables.
struct GaugeEvolution {
    lapse: f64,
    shift: Vector,
}

/// Time derivative for scalar fields.
struct ScalarFieldEvolution {
    phi: f64,
    pi: f64,
}

struct Decomposition {
    pos: [f64; 2],

    manifold: Manifold,
    twist: Twist,

    k: SymmetricC1,
    l: ScalarC1,

    theta: ScalarC1,
    z: VectorC1,

    lapse: ScalarC2,
    shift: VectorC1,

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
            k,
            y,
            theta,
            z,
            lapse,
            shift,
        }: DynamicalSystem,
        scalar_fields: impl Iterator<Item = ScalarFieldSystem>,
    ) -> Self {
        let mut l = ScalarC1::default();
        l.value = pos[0] * y.value + k.value[[0, 0]] / metric.value[[0, 0]];
        l.derivs = Tensor::from_fn(|[i]| {
            pos[0] * y.derivs[[i]] + k.derivs[[0, 0, i]] / metric.value[[0, 0]]
                - k.value[[0, 0]] / metric.value[[0, 0]].powi(2) * metric.derivs[[0, 0, i]]
        });
        // From the `ρy` term.
        l.derivs[[0]] += y.value;

        let manifold = Manifold::new(metric);
        let twist = Twist::new(&manifold, pos, seed);

        let mut source = StressEnergy::vacuum();

        for sf in scalar_fields {
            let sf = StressEnergy::scalar(&manifold, sf);

            source.energy += sf.energy;
            source.momentum[[0]] += sf.momentum[[0]];
            source.momentum[[1]] += sf.momentum[[1]];

            source.stress[[0, 0]] += sf.stress[[0, 0]];
            source.stress[[0, 1]] += sf.stress[[0, 1]];
            source.stress[[1, 1]] += sf.stress[[1, 1]];
            source.stress[[1, 0]] += sf.stress[[1, 0]];

            source.angular_momentum += sf.angular_momentum;
            source.angular_shear[[0]] += sf.angular_shear[[0]];
            source.angular_shear[[1]] += sf.angular_shear[[1]];
            source.angular_stress += sf.angular_stress;
        }

        Self {
            pos,
            manifold,
            twist,
            k,
            l,
            theta,
            z,
            lapse,
            shift,
            source,
            y: y.value,
        }
    }

    fn on_axis(&self) -> bool {
        self.pos[0].abs() <= ON_AXIS
    }

    fn metric_evolution(&self) -> MetricEvolution {
        // Destructure self
        let Self {
            pos,
            manifold:
                Manifold {
                    metric,
                    inv,
                    symbols,
                    ..
                },
            twist,
            k,
            l,
            theta,
            z,
            lapse,
            shift,
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

        let ricci = symbols.ricci();
        let ricci_trace = inv.cotrace(&ricci);

        let k_grad = k.gradient(symbols);
        let k_trace = inv.cotrace(&k.value);
        let k_trace_grad = Vector::from_fn(|[i]| {
            S::sum(|[m, n]| {
                k.derivs[[m, n, i]] * inv.value[[m, n]] + k.value[[m, n]] * inv.derivs[[m, n, i]]
            })
        });
        let k_con = S::symmetric(|[a, b]| {
            S::sum(|[m, n]| inv.value[[a, m]] * inv.value[[b, n]] * k.value[[m, n]])
        });

        let l_grad = l.gradient(symbols);

        let theta_grad = theta.gradient(symbols);

        let z_grad = z.gradient(symbols);
        let z_con = inv.raise_first(&z.value);

        let lapse_grad = lapse.gradient(symbols);
        let lapse_hess = lapse.hessian(symbols);

        let stress_trace = inv.cotrace(stress);

        // *********************************
        // Hamiltonian *********************
        // *********************************

        let hamiltonian = {
            let term1 = 0.5 * (ricci_trace + k_trace.powi(2)) + k_trace * l.value;
            let term2 = -0.5 * S::sum(|[i, j]| k.value[[i, j]] * k_con[[i, j]]);
            let term3 = -S::sum(|[i, j]| twist.hess()[[i, j]] * inv.value[[i, j]]);

            term1 + term2 + term3 - KAPPA * energy
        };

        // **********************************
        // Momentum *************************
        // **********************************

        let momentum = Vector::from_fn(|[i]| {
            let term1 = -k_trace_grad[[i]] - l_grad[[i]];
            let term2 = S::sum(|[m, n]| k_grad[[i, m, n]] * inv.value[[m, n]]);

            let mut regular = S::sum(|[m]| twist.regular_con()[[m]] * k.value[[m, i]])
                - twist.regular_co()[[i]] * l.value;

            if on_axis && i == 0 {
                regular += -y;
            } else if on_axis && i == 1 {
                regular += k.derivs[[0, 1, 0]] / metric.value[[0, 0]];
            }

            term1 + term2 + regular - KAPPA * momentum[[i]]
        });

        // ***********************************
        // Metric ****************************
        // ***********************************

        let g_t = {
            let term1 = S::symmetric(|[i, j]| -2.0 * lapse.value * k.value[[i, j]]);
            let term2 = metric.killing(shift);

            S::symmetric(|idx| term1[idx] + term2[idx])
        };

        // (∂ₜλ) / λ
        let mut lam_lt =
            -lapse.value * l.value + S::sum(|[a]| twist.regular_co()[[a]] * shift.value[[a]]);
        if on_axis {
            lam_lt += shift.derivs[[0, 0]];
        }

        // ***********************************
        // Extrinsic curvature ***************
        // ***********************************

        let k_lie_shift = k.lie_derivative(shift);
        let k_t = S::symmetric(|[a, b]| {
            let term1 = ricci[[a, b]] - twist.hess()[[a, b]];
            let term2 = (k_trace + l.value) * k.value[[a, b]];
            let term3 =
                -2.0 * S::sum(|[m, n]| k.value[[a, m]] * inv.value[[m, n]] * k.value[[n, b]]);
            let term4 = z_grad[[a, b]] + z_grad[[b, a]] - 2.0 * theta.value * k.value[[a, b]];
            let term5 = -KAPPA
                * (stress[[a, b]]
                    + 0.5 * (energy - stress_trace - angular_stress) * metric.value[[a, b]]);

            lapse.value * (term1 + term2 + term3 + term4 + term5) - lapse_hess[[a, b]]
                + k_lie_shift[[a, b]]
        });

        let l_t = {
            let l_lie_shift = l.lie_derivative(shift);

            let term1 = l.value * (k_trace + l.value - 2.0 * theta.value);
            let term2 = -inv.cotrace(twist.hess());
            let term3 = -0.5 * KAPPA * (energy - stress_trace + angular_stress);

            let mut regular = S::sum(|[m]| {
                twist.regular_con()[[m]] * (2.0 * lapse.value * z.value[[m]] - lapse_grad[[m]])
            });

            if on_axis {
                regular += (2.0 * lapse.value * z.derivs[[0, 0]] - lapse.derivs2[[0, 0]])
                    / metric.value[[0, 0]];
            };

            lapse.value * (term1 + term2 + term3) + l_lie_shift + regular
        };

        // ************************************
        // Constraint *************************
        // ************************************

        let theta_t = {
            let theta_lie_shift = theta.lie_derivative(shift);

            let term1 = lapse.value * hamiltonian - lapse.value * (k_trace + l.value) * theta.value;
            let term2 = lapse.value * inv.cotrace(&z_grad);
            let term3 = -S::sum(|[i]| lapse_grad[[i]] * z_con[[i]]);

            let mut regular = lapse.value * S::sum(|[m]| twist.regular_con()[[m]] * z.value[[m]]);
            if on_axis {
                regular += lapse.value * z.derivs[[0, 0]] / metric.value[[0, 0]];
            }

            term1 + term2 + term3 + theta_lie_shift + regular
        };

        let z_lie_shift = z.lie_derivative(shift);
        let z_t = S::vector(|[i]| {
            let term1 = lapse.value * momentum[[i]];
            let term2 = -2.0 * lapse.value * S::sum(|[m]| k.value[[i, m]] * z_con[[m]]);
            let term3 = lapse.value * theta_grad[[i]] - lapse_grad[[i]] * theta.value;
            term1 + term2 + term3 + z_lie_shift[[i]]
        });

        let mut y_t = 0.0;
        let mut seed_t = 0.0;

        if !on_axis {
            y_t = (l_t - k_t[[0, 0]] / metric.value[[0, 0]]
                + k.value[[0, 0]] / metric.value[[0, 0]].powi(2) * g_t[[0, 0]])
                / pos[0];
            seed_t = (lam_lt - 0.5 * g_t[[0, 0]] / metric.value[[0, 0]]) / pos[0];
        }

        MetricEvolution {
            g: g_t,
            seed: seed_t,

            k: k_t,
            y: y_t,

            theta: theta_t,
            z: z_t,
        }
    }

    fn harmonic(&self) -> GaugeEvolution {
        let Self {
            pos,
            manifold: Manifold { inv, det, .. },
            twist,
            k,
            l,
            theta,
            z,
            lapse,
            shift,
            ..
        } = self;

        let on_axis = pos[0].abs() <= ON_AXIS;

        const F: f64 = 1.0;
        const A: f64 = 1.0;
        const MU: f64 = 1.0;
        const D: f64 = 1.0;
        const M: f64 = 2.0;

        let k_trace = inv.cotrace(&k.value);
        let lapse2 = lapse.value.powi(2);

        let lapse_t = {
            let term1 = -lapse2 * F * (k_trace + l.value - M * theta.value);
            let term2 = S::sum(|i| shift.value[i] * lapse.derivs[i]);
            term1 + term2
        };

        let shift_t = S::vector(|[i]| {
            let lamg_term = 0.5 / det.value * S::sum(|[m]| inv.value[[i, m]] * det.derivs[[m]])
                + twist.regular_con()[[i]];

            let g_inv_term: f64 = S::sum(|[m]| inv.derivs[[i, m, m]]);

            let term1 = -lapse2 * MU * g_inv_term;
            let term2 = lapse2 * 2.0 * MU * S::sum(|[m]| inv.value[[i, m]] * z.value[[m]]);

            let term3 = -lapse.value * A * S::sum(|[m]| inv.value[[i, m]] * lapse.derivs[[m]]);
            let term4 = S::sum(|[m]| shift.value[[m]] * shift.derivs[[i, m]]);

            let mut regular = -lapse2 * (2.0 * MU - D) * lamg_term;
            if !on_axis && i == 0 {
                regular += lapse2 * (2.0 * MU - D) * inv.value[[0, 0]] / pos[0];
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
            manifold: Manifold { inv, .. },

            k,
            l,
            theta,

            lapse,
            shift,
            ..
        } = self;

        const F: f64 = 1.0;
        const M: f64 = 2.0;

        let k_trace = inv.cotrace(&k.value);
        let lapse2 = lapse.value.powi(2);

        let lapse_t = {
            let term1 = -lapse2 * F * (k_trace + l.value - M * theta.value);
            let term2 = S::sum(|i| shift.value[i] * lapse.derivs[i]);
            term1 + term2
        };

        GaugeEvolution {
            lapse: lapse_t,
            shift: Tensor::zeros(),
        }
    }

    fn log_plus_one(&self) -> GaugeEvolution {
        let Self {
            pos,
            manifold: Manifold { inv, det, .. },
            twist,
            k,
            l,
            theta,
            z,
            lapse,
            shift,
            ..
        } = self;

        let on_axis = pos[0].abs() <= ON_AXIS;

        const F: f64 = 1.0;
        const A: f64 = 1.0;
        const MU: f64 = 1.0;
        const D: f64 = 1.0;
        const M: f64 = 2.0;

        let k_trace = inv.cotrace(&k.value);
        let lapse2 = lapse.value.powi(2);

        let lapse_t = {
            let term1 = -2.0 * lapse.value * F * (k_trace + l.value - M * theta.value);
            let term2 = S::sum(|i| shift.value[i] * lapse.derivs[i]);
            term1 + term2
        };

        let shift_t = S::vector(|[i]| {
            let lamg_term = 0.5 / det.value * S::sum(|[m]| inv.value[[i, m]] * det.derivs[[m]])
                + twist.regular_con()[[i]];

            let g_inv_term: f64 = S::sum(|[m]| inv.derivs[[i, m, m]]);

            let term1 = -lapse2 * MU * g_inv_term;
            let term2 = lapse2 * 2.0 * MU * S::sum(|[m]| inv.value[[i, m]] * z.value[[m]]);

            let term3 = -lapse.value * A * S::sum(|[m]| inv.value[[i, m]] * lapse.derivs[[m]]);
            let term4 = S::sum(|[m]| shift.value[[m]] * shift.derivs[[i, m]]);

            let mut regular = -lapse2 * (2.0 * MU - D) * lamg_term;
            if !on_axis && i == 0 {
                regular += lapse2 * (2.0 * MU - D) * inv.value[[0, 0]] / pos[0];
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
            manifold: Manifold { inv, .. },

            k,
            l,
            theta,

            lapse,
            shift,
            ..
        } = self;

        const F: f64 = 1.0;
        const M: f64 = 2.0;

        let k_trace = inv.cotrace(&k.value);

        let lapse_t = {
            let term1 = -2.0 * lapse.value * F * (k_trace + l.value - M * theta.value);
            let term2 = S::sum(|i| shift.value[i] * lapse.derivs[i]);
            term1 + term2
        };

        GaugeEvolution {
            lapse: lapse_t,
            shift: Tensor::zeros(),
        }
    }

    fn scalar_field_evolution(&self, field: ScalarFieldSystem) -> ScalarFieldEvolution {
        let Self {
            manifold:
                Manifold {
                    metric,
                    inv,
                    symbols,
                    ..
                },
            twist,
            k,
            l,
            lapse,
            shift,
            ..
        } = self;
        let phi_hess = field.phi.hessian(symbols);
        let phi_grad = field.phi.gradient(symbols);

        let lapse_grad = lapse.gradient(symbols);

        let phi_t =
            lapse.value * field.pi.value + S::sum(|[i]| field.phi.derivs[[i]] * shift.value[[i]]);

        let pi_t = {
            let term1 = lapse.value * inv.cotrace(&phi_hess)
                + lapse.value * field.pi.value * (inv.cotrace(&k.value) + l.value);
            let term2 = S::sum(|[i, j]| phi_grad[[i]] * inv.value[[i, j]] * lapse_grad[[j]]);

            let mut term3 = lapse.value * S::sum(|[i]| phi_grad[[i]] * twist.regular_con()[[i]]);
            if self.on_axis() {
                term3 += lapse.value * field.phi.derivs2[[0, 0]] / metric.value[[0, 0]];
            }

            let term4 = -lapse.value * field.mass.powi(2) * field.phi.value;
            let term5 = S::sum(|[i]| field.pi.derivs[[i]] * shift.value[[i]]);

            term1 + term2 + term3 + term4 + term5
        };

        ScalarFieldEvolution {
            phi: phi_t,
            pi: pi_t,
        }
    }
}
