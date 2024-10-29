use crate::{lie_derivative, Space};

use super::{Metric, Static, Tensor, Tensor1, Tensor2, TensorFieldC1, TensorFieldC2};

const ON_AXIS: f64 = 1e-10;

const KAPPA: f64 = 1.0;

pub struct Twist {
    lam_regular_co: Tensor1<2>,
    lam_regular_con: Tensor1<2>,
    lam_hess: Tensor2<2>,
}

impl Twist {
    pub fn new(metric: &Metric<2>, pos: [f64; 2], seed: TensorFieldC2<2, Static<0>>) -> Self {
        let on_axis = pos[0].abs() <= ON_AXIS;

        let space = Space::<2>;

        let s = seed.value[[]];
        let s_derivs = seed.derivs;
        let s_second_derivs = seed.second_derivs;

        // Plus 1/r on axis for r component
        let mut lam_reg_co = Tensor::zeros();
        // Plus 1/(r * grr) on axis for r component
        let mut lam_reg_con = Tensor::zeros();
        // Fully regular
        let mut lam_hess = Tensor::zeros();

        {
            let g_derivs_term =
                Tensor1::<2>::from_fn(|[i]| metric.derivs()[[0, 0, i]] / metric.value()[[0, 0]]);

            let g_second_derivs_term = Tensor2::<2>::from_fn(|[i, j]| {
                metric.second_derivs()[[0, 0, i, j]] / metric.value()[[0, 0]]
                    - metric.derivs()[[0, 0, i]] * metric.derivs()[[0, 0, j]]
                        / (metric.value()[[0, 0]] * metric.value()[[0, 0]])
            });

            // Decompose lam_r into a regular part and an Order(1/r) part.
            let lam_r = s + pos[0] * s_derivs[[0]] + 0.5 * g_derivs_term[[0]]; // + 1.0 / pos[0]
            let lam_z = pos[0] * s_derivs[[1]] + 0.5 * g_derivs_term[[1]];

            lam_reg_co[[0]] = lam_r;
            lam_reg_co[[1]] = lam_z;

            lam_reg_con[[0]] = metric.inv()[[0, 0]] * lam_r + metric.inv()[[0, 1]] * lam_z;
            lam_reg_con[[1]] = metric.inv()[[1, 0]] * lam_r + metric.inv()[[1, 1]] * lam_z;

            if !on_axis {
                lam_reg_co[[0]] += 1.0 / pos[0];

                lam_reg_con[[0]] += metric.inv()[[0, 0]] / pos[0];
                lam_reg_con[[1]] += metric.inv()[[1, 0]] / pos[0];
            } else {
                lam_reg_con[[1]] += -metric.derivs()[[0, 1, 0]] / metric.det();
            }

            let mut gamma_regular = space.tensor(|[i, j]| {
                space.sum(|[m]| lam_reg_con[[m]] * metric.christoffel()[[m, i, j]])
            });

            if on_axis {
                gamma_regular[[0, 0]] +=
                    0.5 * metric.second_derivs()[[0, 0, 0, 0]] / metric.value()[[0, 0]];
                gamma_regular[[0, 1]] += 0.0; // + 0.5 * g_par[0][0][1] / (pos[0] * g[0][0])
                gamma_regular[[1, 1]] += 0.5
                    * (2.0 * metric.second_derivs()[[0, 1, 0, 1]]
                        - metric.second_derivs()[[1, 1, 0, 0]])
                    / metric.value()[[0, 0]];
            }

            let lam_rr = {
                // Plus a -1/r^2 term that gets cancelled by lam_r * lam_r
                let term1 = 2.0 * s_derivs[[0]]
                    + pos[0] * s_second_derivs[[0, 0]]
                    + 0.5 * g_second_derivs_term[[0, 0]]; // -1.0 / pos[0].powi(2)
                let term2 = lam_r * lam_r; // + 1.0 / pos[0].powi(2)
                let term3 = if on_axis {
                    // Use lhopital's rule to compute on axis lam_r / r and lam_z / r on axis.
                    let lam_r_lhopital = 2.0 * s_derivs[[0]]
                        + 0.5 * metric.second_derivs()[[0, 0, 0, 0]] / metric.value()[[0, 0]];
                    2.0 * lam_r_lhopital
                } else {
                    2.0 * lam_r / pos[0]
                };

                term1 + term2 + term3 - gamma_regular[[0, 0]]
            };

            let lam_rz = {
                let term1: f64 = s_derivs[[1]]
                    + pos[0] * s_second_derivs[[0, 1]]
                    + 0.5 * g_second_derivs_term[[0, 1]];
                let term2 = lam_r * lam_z;
                let term3 = if on_axis {
                    s_derivs[[1]] // + 0.5 * g_par[0][0][1] / (g[0][0] * pos[0]);
                } else {
                    lam_z / pos[0]
                };
                term1 + term2 + term3 - gamma_regular[[0, 1]]
            };

            let lam_zz = pos[0] * s_second_derivs[[1, 1]]
                + 0.5 * g_second_derivs_term[[1, 1]]
                + lam_z * lam_z
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

    pub fn hess(&self) -> &Tensor2<2> {
        &self.lam_hess
    }

    pub fn regular_co(&self) -> &Tensor1<2> {
        &self.lam_regular_co
    }

    pub fn regular_con(&self) -> &Tensor1<2> {
        &self.lam_regular_con
    }
}

pub struct System {
    pub metric: Metric<2>,
    pub seed: TensorFieldC2<2, Static<0>>,

    pub k: TensorFieldC1<2, Static<2>>,
    pub y: TensorFieldC1<2, Static<0>>,

    pub theta: TensorFieldC1<2, Static<0>>,
    pub z: TensorFieldC1<2, Static<1>>,

    pub lapse: TensorFieldC2<2, Static<0>>,
    pub shift: TensorFieldC1<2, Static<1>>,

    pub source: StressEnergy,
}

pub struct ScalarFieldSystem {
    pub phi: TensorFieldC2<2, Static<0>>,
    pub pi: TensorFieldC1<2, Static<0>>,
    pub mass: f64,
}

/// A decomposition of the stress energy tensor in axisymmetric units.
pub struct StressEnergy {
    pub energy: f64,
    pub momentum: Tensor1<2>,
    pub stress: Tensor2<2>,

    pub angular_momentum: f64,
    pub angular_shear: Tensor1<2>,
    pub angular_stress: f64,
}

impl StressEnergy {
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

    pub fn scalar(
        metric: &Metric<2>,
        mass: f64,
        phi: TensorFieldC1<2, Static<0>>,
        pi: f64,
    ) -> Self {
        let phi_grad = metric.gradiant(phi.clone());

        let angular_stress = -0.5 * (mass * mass) * (phi.scalar() + phi.scalar() * phi.scalar());
        let angular_shear = Tensor::zeros();
        let angular_momentum = 0.0;

        let energy = pi * pi + 0.5 * (mass * mass) * (phi.scalar() + phi.scalar() * phi.scalar());
        let momentum = -pi * phi_grad.clone();
        let stress = phi_grad.clone() * phi_grad.clone()
            - metric.value().clone()
                * 0.5
                * (mass * mass)
                * (phi.scalar() + phi.scalar() * phi.scalar());

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

pub struct Evolution {
    pub g: Tensor2<2>,
    pub seed: f64,

    pub k: Tensor2<2>,
    pub y: f64,

    pub theta: f64,
    pub z: Tensor1<2>,
}

pub struct Gauge {
    pub lapse: f64,
    pub shift: Tensor1<2>,
}

pub struct ScalarField {
    pub phi: f64,
    pub pi: f64,
}

pub struct Decomposition {
    pub pos: [f64; 2],

    pub metric: Metric<2>,
    pub twist: Twist,

    pub k: TensorFieldC1<2, Static<2>>,
    pub l: TensorFieldC1<2, Static<0>>,

    pub theta: TensorFieldC1<2, Static<0>>,
    pub z: TensorFieldC1<2, Static<1>>,

    pub lapse: TensorFieldC2<2, Static<0>>,
    pub shift: TensorFieldC1<2, Static<1>>,

    pub source: StressEnergy,

    y: f64,
}

impl Decomposition {
    pub fn new(
        pos: [f64; 2],
        System {
            metric,
            seed,
            k,
            y,
            theta,
            z,
            lapse,
            shift,
            source,
        }: System,
    ) -> Self {
        let twist = Twist::new(&metric, pos, seed);

        // Build l from y and k
        let l = TensorFieldC1::<2, Static<0>> {
            value: Tensor::from_storage(
                pos[0] * y.scalar() + k.value[[0, 0]] / metric.value()[[0, 0]],
            ),
            derivs: {
                let mut result = Tensor1::from_fn(|[i]| {
                    pos[0] * y.derivs[[i]] + k.derivs[[0, 0, i]] / metric.value()[[0, 0]]
                        - k.value[[0, 0]] / metric.value()[[0, 0]] * metric.derivs()[[0, 0, i]]
                            / metric.value()[[0, 0]]
                });

                // From the rho * y term.
                result[[0]] += y.scalar();

                result
            },
        };

        Self {
            pos,
            metric,
            twist,
            k,
            y: y.scalar(),
            l,
            theta,
            z,
            lapse,
            shift,
            source,
        }
    }

    pub fn on_axis(&self) -> bool {
        self.pos[0] <= ON_AXIS
    }

    pub fn evolution(&self) -> Evolution {
        // Destructure self
        let Self {
            pos,
            metric,
            twist,
            k,
            y,
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
        } = self;

        let on_axis = pos[0].abs() <= ON_AXIS;

        let s = Space::<2>;

        let ricci = metric.ricci();
        let ricci_trace = metric.trace(ricci.clone());

        let k_grad = metric.gradiant(k.clone());
        let k_trace = s.sum(|[i, j]| k.value[[i, j]] * metric.inv()[[i, j]]);
        let k_trace_grad = s.vector(|i| {
            s.sum(|[m, n]| {
                k.derivs[[m, n, i]] * metric.inv()[[m, n]]
                    + k.value[[m, n]] * metric.inv_derivs()[[m, n, i]]
            })
        });

        let k_con = s.tensor(|[i, j]| {
            s.sum(|[m, n]| metric.inv()[[i, m]] * metric.inv()[[j, n]] * k.value[[m, n]])
        });

        let l_grad = metric.gradiant(l.clone());

        let theta_grad = metric.gradiant(theta.clone());
        let z_grad = metric.gradiant(z.clone());
        let z_con = metric.raise(z.value.clone());

        let lapse_grad = metric.gradiant(lapse.clone().into());
        let lapse_hess = metric.hessian(lapse.clone());

        let stress_trace = metric.trace(stress.clone());

        // *********************************
        // Hamiltonian *********************
        // *********************************

        let hamiltonian = {
            let term1 = 0.5 * (ricci_trace + k_trace * k_trace) + k_trace * l.scalar();
            let term2 = -0.5 * s.sum(|[i, j]| k.value[[i, j]] * k_con[[i, j]]);
            let term3 = -s.sum(|[i, j]| twist.hess()[[i, j]] * metric.inv()[[i, j]]);

            term1 + term2 + term3 - KAPPA * energy
        };

        // **********************************
        // Momentum *************************
        // **********************************

        let momentum = s.vector(|i| {
            let term1 = -k_trace_grad[[i]] - l_grad[[i]];
            let term2 = s.sum(|[m, n]| k_grad[[i, m, n]] * metric.inv()[[m, n]]);

            let mut regular = s.sum(|[m]| twist.regular_con()[[m]] * k.value[[m, i]])
                - twist.regular_co()[[i]] * l.scalar();

            if on_axis && i == 0 {
                regular += -y;
            } else if on_axis && i == 1 {
                regular += k.derivs[[0, 1, 0]] / metric.value()[[0, 0]];
            }

            term1 + term2 + regular - KAPPA * momentum[[i]]
        });

        // ***********************************
        // Metric ****************************
        // ***********************************

        let g_t = {
            let term1 = s.tensor(|[i, j]| -2.0 * lapse.scalar() * k.value[[i, j]]);
            let term2 = metric.killing(shift.clone());

            term1 + term2
        };

        // (partial_t lambda) / lambda
        let mut lam_lt =
            -lapse.scalar() * l.scalar() + s.sum(|[i]| twist.regular_co()[[i]] * shift.value[[i]]);
        if on_axis {
            lam_lt += shift.derivs[[0, 0]];
        }

        // ***********************************
        // Extrinsic curvature ***************
        // ***********************************

        let k_t = {
            let k_lie_shift = lie_derivative(shift.clone(), k.clone());
            let term1 = lapse.scalar() * ricci.clone()
                - lapse.scalar() * twist.hess().clone()
                - lapse_hess.clone();
            let term2 = lapse.scalar() * (k_trace + l.scalar()) * k.value.clone();
            let term3 = -2.0
                * lapse.scalar()
                * s.tensor(|[i, j]| {
                    s.sum(|[m, n]| k.value[[i, m]] * metric.inv()[[m, n]] * k.value[[n, j]])
                });
            let term4 = lapse.scalar()
                * s.tensor(|[i, j]| {
                    z_grad[[i, j]] + z_grad[[j, i]] - 2.0 * k.value[[i, j]] * theta.scalar()
                });

            let term5 = -lapse.scalar()
                * KAPPA
                * (stress.clone()
                    + 0.5 * (energy - stress_trace - angular_stress) * metric.value().clone());

            term1 + term2 + term3 + term4 + term5 + k_lie_shift
        };

        let l_t = {
            let term1 = lapse.scalar() * l.scalar() * (k_trace + l.scalar() - 2.0 * theta.scalar());
            let term2 = -lapse.scalar() * metric.trace(twist.hess().clone());
            let term3 = s.sum(|m| shift.value[m] * l.derivs[m]);

            let mut regular = s.sum(|[m]| {
                twist.regular_con()[[m]] * (2.0 * lapse.scalar() * z.value[[m]] - lapse_grad[[m]])
            });

            if on_axis {
                regular += (2.0 * lapse.scalar() * z.derivs[[0, 0]] - lapse.second_derivs[[0, 0]])
                    / metric.value()[[0, 0]];
            };

            let term4 = -0.5 * lapse.scalar() * KAPPA * (energy - stress_trace + angular_stress);

            term1 + term2 + term3 + term4 + regular
        };

        // ************************************
        // Constraint *************************
        // ************************************

        let theta_t = {
            let term1 = lapse.scalar() * hamiltonian
                - lapse.scalar() * (k_trace + l.scalar()) * theta.scalar();
            let term2 = lapse.scalar() * metric.trace(z_grad);
            let term3 = -s.sum(|i| lapse_grad[i] * z_con[i]);
            let term4 = s.sum(|[i]| theta.derivs[[i]] * shift.value[[i]]);

            let mut regular = lapse.scalar() * s.sum(|[m]| twist.regular_con()[[m]] * z.value[[m]]);
            if on_axis {
                regular += lapse.scalar() * z.derivs[[0, 0]] / metric.value()[[0, 0]];
            }

            term1 + term2 + term3 + term4 + regular
        };

        let z_lie_shift = lie_derivative(shift.clone(), z.clone());

        let z_t = s.tensor(|[i]| {
            let term1 = lapse.scalar() * momentum[[i]];
            let term2 = -2.0 * lapse.scalar() * s.sum(|[m]| k.value[[i, m]] * z_con[[m]]);
            let term3 = lapse.scalar() * theta_grad[[i]] - lapse_grad[[i]] * theta.scalar();
            term1 + term2 + term3 + z_lie_shift[[i]]
        });

        let mut y_t = 0.0;
        let mut seed_t = 0.0;

        if !on_axis {
            y_t = (l_t - k_t[[0, 0]] / metric.value()[[0, 0]]
                + k.value[[0, 0]] / metric.value()[[0, 0]].powi(2) * g_t[[0, 0]])
                / pos[0];
            seed_t = (lam_lt - 0.5 * g_t[[0, 0]] / metric.value()[[0, 0]]) / pos[0];
        }

        Evolution {
            g: g_t,
            seed: seed_t,

            k: k_t,
            y: y_t,

            theta: theta_t,
            z: z_t,
        }
    }

    pub fn gauge(
        Self {
            pos,
            metric,
            twist,
            k,
            l,
            theta,
            z,
            lapse,
            shift,
            ..
        }: &Self,
    ) -> Gauge {
        let on_axis = pos[0].abs() <= ON_AXIS;

        const F: f64 = 1.0;
        const A: f64 = 1.0;
        const MU: f64 = 1.0;
        const D: f64 = 1.0;
        const M: f64 = 2.0;

        let s = Space::<2>;

        let k_trace = s.sum(|[i, j]| k.value[[i, j]] * metric.inv()[[i, j]]);
        let lapse2 = lapse.scalar() * lapse.scalar();

        let lapse_t = {
            let term1 = -lapse2 * F * (k_trace + l.scalar() - M * theta.scalar());
            let term2 = s.sum(|i| shift.value[i] * lapse.derivs[i]);
            term1 + term2
        };

        let shift_t = s.tensor(|[i]| {
            let lamg_term = 0.5 / metric.det()
                * s.sum(|[m]| metric.inv()[[i, m]] * metric.det_derivs()[[m]])
                + twist.regular_con()[[i]];

            let g_inv_term: f64 = s.sum(|[m]| metric.inv_derivs()[[i, m, m]]);

            let term1 = -lapse2 * MU * g_inv_term;
            let term2 = lapse2 * 2.0 * MU * s.sum(|[m]| metric.inv()[[i, m]] * z.value[[m]]);

            let term3 = -lapse.scalar() * A * s.sum(|[m]| metric.inv()[[i, m]] * lapse.derivs[[m]]);
            let term4 = s.sum(|[m]| shift.value[[m]] * shift.derivs[[i, m]]);

            let mut regular = -lapse2 * (2.0 * MU - D) * lamg_term;
            if !on_axis && i == 0 {
                regular += lapse2 * (2.0 * MU - D) * metric.inv()[[0, 0]] / pos[0];
            }

            term1 + term2 + term3 + term4 + regular
        });

        Gauge {
            lapse: lapse_t,
            shift: shift_t,
        }
    }

    pub fn scalar(&self, field: ScalarFieldSystem) -> ScalarField {
        let s = Space::<2>;

        let Self {
            metric,
            twist,
            k,
            l,
            lapse,
            shift,
            ..
        } = self;

        let phi_hess = metric.hessian(field.phi.clone());
        let phi_grad = metric.gradiant(field.phi.clone().into());
        let pi_grad = metric.gradiant(field.pi.clone());

        let lapse_grad = metric.gradiant(lapse.clone().into());

        let phi_t =
            lapse.scalar() * field.pi.scalar() + s.sum(|[i]| phi_grad[[i]] * shift.value[[i]]);

        let pi_t = {
            let term1 = lapse.scalar() * metric.trace(phi_hess)
                + lapse.scalar() * field.pi.scalar() * (metric.trace(k.value.clone()) + l.scalar());
            let term2 = s.sum(|[i, j]| phi_grad[[i]] * metric.inv()[[i, j]] * lapse_grad[[j]]);

            let mut term3 = lapse.scalar() * s.sum(|[i]| phi_grad[[i]] * twist.regular_con()[[i]]);
            if self.on_axis() {
                term3 += lapse.scalar() * field.phi.second_derivs[[0, 0]] / metric.value()[[0, 0]];
            }

            let term4 = s.sum(|[i]| pi_grad[[i]] * shift.value[[i]]);

            term1 + term2 + term3 + term4
        };

        ScalarField {
            phi: phi_t,
            pi: pi_t,
        }
    }
}
