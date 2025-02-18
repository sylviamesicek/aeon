use aeon_tensor::{Matrix, Metric, Space, Tensor, Vector};

const ON_AXIS: f64 = 1e-10;
const KAPPA: f64 = 1.0;

#[derive(Default, Clone)]
pub struct ScalarFieldSystem {
    pub phi: f64,
    pub phi_derivs: Vector<2>,
    pub phi_second_derivs: Matrix<2>,
    pub pi: f64,
    pub pi_derivs: Vector<2>,
    pub mass: f64,
}

/// A decomposition of the stress energy tensor in axisymmetric units.
pub struct StressEnergy {
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
        let phi_grad_trace = metric.cotrace(phi_grad * phi_grad);

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

/// Various tensors relating to the twist degree of freedom in a metric.
pub struct Twist {
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
}

/// All degrees of freedom wrapped in one struct.
pub struct DynamicalSystem {
    pub metric: Metric<2>,
    pub seed: f64,
    pub seed_partials: Vector<2>,
    pub seed_second_partials: Matrix<2>,
}
