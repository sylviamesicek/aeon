#![allow(unused_assignments)]

const M: f64 = 1.0;
const KAPPA: f64 = 1.0;

// A very useful function
use crate::types::*;
use std::array::from_fn;

// *********************************
// Tensor Utils ********************
// *********************************

pub fn tensor1(f: impl FnMut(usize) -> f64) -> Rank1 {
    from_fn(f)
}

pub fn tensor2(mut f: impl FnMut(usize, usize) -> f64) -> Rank2 {
    from_fn(|i| from_fn(|j| f(i, j)))
}

pub fn sym_tensor2(mut f: impl FnMut(usize, usize) -> f64) -> Rank2 {
    let rr = f(0, 0);
    let rz = f(0, 1);
    let zz = f(1, 1);

    [[rr, rz], [rz, zz]]
}

pub fn tensor3(mut f: impl FnMut(usize, usize, usize) -> f64) -> Rank3 {
    from_fn(|i| from_fn(|j| from_fn(|k| f(i, j, k))))
}

pub fn tensor4(mut f: impl FnMut(usize, usize, usize, usize) -> f64) -> Rank4 {
    from_fn(|i| from_fn(|j| from_fn(|k| from_fn(|l| f(i, j, k, l)))))
}

pub fn sum1(mut f: impl FnMut(usize) -> f64) -> f64 {
    f(0) + f(1)
}

pub fn sum2(mut f: impl FnMut(usize, usize) -> f64) -> f64 {
    sum1(|i| sum1(|j| f(i, j)))
}

pub fn sum3(mut f: impl FnMut(usize, usize, usize) -> f64) -> f64 {
    sum1(|i| sum1(|j| sum1(|k| f(i, j, k))))
}

pub fn sum4(mut f: impl FnMut(usize, usize, usize, usize) -> f64) -> f64 {
    sum1(|i| sum1(|j| sum1(|k| sum1(|l| f(i, j, k, l)))))
}

pub fn det2(tensor: Rank2) -> f64 {
    tensor[0][0] * tensor[1][1] - tensor[0][1] * tensor[1][0]
}

pub fn det_par2(tensor: Rank2, par: Rank3) -> Rank1 {
    tensor1(|i| {
        tensor[0][0] * par[1][1][i] + par[0][0][i] * tensor[1][1]
            - tensor[0][1] * par[1][0][i]
            - par[0][1][i] * tensor[1][0]
    })
}

pub fn raise3(tensor: Rank3, g_inv: Rank2) -> Rank3 {
    tensor3(|i, j, k| sum1(|m| g_inv[i][m] * tensor[m][j][k]))
}

// ***********************
// Geometry **************
// ***********************

pub fn christoffel(g_par: Rank3) -> Rank3 {
    tensor3(|i, j, k| 0.5 * (g_par[i][j][k] + g_par[k][i][j] - g_par[j][k][i]))
}

pub fn christoffel_par(g_par2: Rank4) -> Rank4 {
    tensor4(|i, j, k, l| 0.5 * (g_par2[i][j][k][l] + g_par2[k][i][j][l] - g_par2[j][k][i][l]))
}

pub fn christoffel_2nd(gamma: Rank3, g_inv: Rank2) -> Rank3 {
    raise3(gamma, g_inv)
}

pub fn christoffel_2nd_par(
    gamma: Rank3,
    gamma_par: Rank4,
    g_inv: Rank2,
    g_inv_par: Rank3,
) -> Rank4 {
    tensor4(|i, j, k, l| {
        sum1(|m| g_inv[i][m] * gamma_par[m][j][k][l] + g_inv_par[i][m][l] * gamma[m][j][k])
    })
}

pub fn ricci(gamma_2nd: Rank3, gamma_2nd_par: Rank4) -> Rank2 {
    tensor2(|i, j| {
        let term1 = sum1(|m| gamma_2nd_par[m][i][j][m] - gamma_2nd_par[m][m][i][j]);
        let term2 = sum2(|m, n| {
            gamma_2nd[m][m][n] * gamma_2nd[n][i][j] - gamma_2nd[m][i][n] * gamma_2nd[n][m][j]
        });
        term1 + term2
    })
}

/// Takes the covariant derivative of a one form.
#[inline]
pub fn grad1(value: Rank1, pars: Rank2, gamma_2nd: Rank3) -> Rank2 {
    tensor2(|i, j| pars[i][j] - sum1(|k| gamma_2nd[k][i][j] * value[k]))
}

/// Takes the covariant derivative of a covariant rank 2 tensor.

#[inline]
pub fn grad2(value: Rank2, pars: Rank3, gamma_2nd: Rank3) -> Rank3 {
    tensor3(|i, j, k| {
        pars[i][j][k]
            - sum1(|l| gamma_2nd[l][i][k] * value[l][j] + gamma_2nd[l][j][k] * value[i][l])
    })
}

/// Takes the hessian of a scalar field.
#[inline]
pub fn hess0(pars: Rank1, pars2: Rank2, gamma_2nd: Rank3) -> Rank2 {
    tensor2(|i, j| pars2[i][j] - sum1(|k| gamma_2nd[k][i][j] * pars[k]))
}

#[inline]
pub fn lie1(t: Rank1, tpar: Rank2, v: Rank1, vpar: Rank2) -> Rank1 {
    tensor1(|i| {
        sum1(|m| {
            let term1 = v[m] * tpar[i][m];
            let term2 = vpar[m][i] * t[m];
            term1 + term2
        })
    })
}

#[inline]
pub fn lie2(t: Rank2, tpar: Rank3, v: Rank1, vpar: Rank2) -> Rank2 {
    tensor2(|i, j| {
        sum1(|m| {
            let term1 = v[m] * tpar[i][j][m];
            let term2 = vpar[m][i] * t[m][j] + vpar[m][j] * t[i][m];
            term1 + term2
        })
    })
}

pub fn hyperbolic(sys: HyperbolicSystem, pos: [f64; 2]) -> HyperbolicDerivs {
    let on_axis = pos[0].abs() <= 10e-10;

    // ******************************
    // Unpack variables
    // ******************************

    // Metric
    let g = sys.metric();
    let g_par = sys.metric_par();
    let g_par2 = sys.metric_par2();

    let s = sys.seed();
    let s_par = sys.seed_par();
    let s_par2 = sys.seed_par2();

    // Extrinsic curvature
    let k = sys.extrinsic();
    let k_par = sys.extrinsic_par();

    let y = sys.y;
    let y_par = [sys.y_r, sys.y_z];

    let l = pos[0] * y + k[0][0] / g[0][0];
    let l_par = {
        // Apply product rule to definition of l.
        let mut result = tensor1(|i| {
            pos[0] * y_par[i] + k_par[0][0][i] / g[0][0]
                - k[0][0] / g[0][0] * g_par[0][0][i] / g[0][0]
        });
        // Extra term from ρ * y
        result[0] += y;
        result
    };

    // Constraints
    let theta = sys.theta;
    let theta_par = [sys.theta_r, sys.theta_z];

    let z = [sys.zr, sys.zz];
    let z_par = [[sys.zr_r, sys.zr_z], [sys.zz_r, sys.zz_z]];

    // Gauge
    let lapse = sys.lapse;
    let lapse_par = [sys.lapse_r, sys.lapse_z];
    let lapse_par2 = [[sys.lapse_rr, sys.lapse_rz], [sys.lapse_rz, sys.lapse_zz]];

    let shift = [sys.shiftr, sys.shiftz];
    let shift_par = [[sys.shiftr_r, sys.shiftr_z], [sys.shiftz_r, sys.shiftz_z]];

    // Scalar field
    let phi = sys.phi;
    let phi_par = [sys.phi_r, sys.phi_z];
    let phi_par2 = [[sys.phi_rr, sys.phi_rz], [sys.phi_rz, sys.phi_zz]];

    let pi = sys.pi;
    let pi_par = [sys.pi_r, sys.pi_z];

    // **************************************
    // Geometry *****************************
    // **************************************

    let g_det = det2(g);
    let g_det_par = det_par2(g, g_par);

    let g_inv = [
        [sys.gzz / g_det, -sys.grz / g_det],
        [-sys.grz / g_det, sys.grr / g_det],
    ];
    let g_inv_par = {
        let grr_inv_par =
            tensor1(|i| g_par[1][1][i] / g_det - g[1][1] / (g_det * g_det) * g_det_par[i]);
        let grz_inv_par =
            tensor1(|i| -g_par[0][1][i] / g_det + g[0][1] / (g_det * g_det) * g_det_par[i]);
        let gzz_inv_par =
            tensor1(|i| g_par[0][0][i] / g_det - g[0][0] / (g_det * g_det) * g_det_par[i]);

        [[grr_inv_par, grz_inv_par], [grz_inv_par, gzz_inv_par]]
    };

    // Next compute Christoffel symbols
    let gamma = christoffel(g_par);
    let gamma_par = christoffel_par(g_par2);

    let gamma_2nd = christoffel_2nd(gamma, g_inv);
    let gamma_2nd_par = christoffel_2nd_par(gamma, gamma_par, g_inv, g_inv_par);

    // And now the Ricci tensor
    let ricci = ricci(gamma_2nd, gamma_2nd_par);
    // As well as the Ricci scalar
    let ricci_trace = sum2(|i, j| ricci[i][j] * g_inv[i][j]);

    // *********************************
    // Contractions and Derivatives ****
    // *********************************

    // Logrithmic derivatives of lambda

    // Λ is Order(1/r) on axis, so we split into a regular part and a 1/r part in the r component.
    // Off axis, this is the full gradient of lambda, including that 1/r term, but on axis this
    // term is set to zero, and one must apply lhopital's rule on a case by case basis.

    // Plus 1/r on axis for r component
    let mut lam_reg_co = [0.0; 2];
    // Plus 1/(r * grr) on axis for r component
    let mut lam_reg_con = [0.0; 2];
    // Fully regular
    let mut lam_hess = [[0.0; 2]; 2];

    {
        let g_par_term = tensor1(|i| g_par[0][0][i] / g[0][0]);
        let g_par2_term = tensor2(|i, j| {
            g_par2[0][0][i][j] / g[0][0] - g_par[0][0][i] * g_par[0][0][j] / (g[0][0] * g[0][0])
        });

        // Decompose lam_r into a regular part and an Order(1/r) part.
        let lam_r = s + pos[0] * s_par[0] + 0.5 * g_par_term[0]; // + 1.0 / pos[0]
        let lam_z = pos[0] * s_par[1] + 0.5 * g_par_term[1];

        lam_reg_co[0] = lam_r;
        lam_reg_co[1] = lam_z;

        lam_reg_con[0] = g_inv[0][0] * lam_r + g_inv[0][1] * lam_z;
        lam_reg_con[1] = g_inv[1][0] * lam_r + g_inv[1][1] * lam_z;

        if !on_axis {
            lam_reg_co[0] += 1.0 / pos[0];

            lam_reg_con[0] += g_inv[0][0] / pos[0];
            lam_reg_con[1] += g_inv[1][0] / pos[0];
        } else {
            // TODO Find source of this term
            lam_reg_con[1] += -g_par[0][1][0] / g_det;
        }

        let mut gamma_regular = tensor2(|i, j| sum1(|m| lam_reg_con[m] * gamma[m][i][j]));

        if on_axis {
            gamma_regular[0][0] += 0.5 * g_par2[0][0][0][0] / g[0][0];
            gamma_regular[0][1] += 0.0; // + 0.5 * g_par[0][0][1] / (pos[0] * g[0][0])
            gamma_regular[1][1] += 0.5 * (2.0 * g_par2[0][1][0][1] - g_par2[1][1][0][0]) / g[0][0];
        }

        let lam_rr = {
            // Plus a -1/r^2 term that gets cancelled by lam_r * lam_r
            let term1 = 2.0 * s_par[0] + pos[0] * s_par2[0][0] + 0.5 * g_par2_term[0][0]; // -1.0 / pos[0].powi(2)
            let term2 = lam_r * lam_r; // + 1.0 / pos[0].powi(2)
            let term3 = if on_axis {
                // Use lhopital's rule to compute on axis lam_r / r and lam_z / r on axis.
                let lam_r_lhopital = 2.0 * s_par[0] + 0.5 * g_par2[0][0][0][0] / g[0][0];
                2.0 * lam_r_lhopital
            } else {
                2.0 * lam_r / pos[0]
            };

            term1 + term2 + term3 - gamma_regular[0][0]
        };

        let lam_rz = {
            let term1 = s_par[1] + pos[0] * s_par2[0][1] + 0.5 * g_par2_term[0][1];
            let term2 = lam_r * lam_z;
            let term3 = if on_axis {
                s_par[1] // + 0.5 * g_par[0][0][1] / (g[0][0] * pos[0]);
            } else {
                lam_z / pos[0]
            };
            term1 + term2 + term3 - gamma_regular[0][1]
        };

        let lam_zz =
            pos[0] * s_par2[1][1] + 0.5 * g_par2_term[1][1] + lam_z * lam_z - gamma_regular[1][1];

        lam_hess[0][0] = lam_rr;
        lam_hess[0][1] = lam_rz;
        lam_hess[1][0] = lam_rz;
        lam_hess[1][1] = lam_zz;
    }

    let k_grad = grad2(k, k_par, gamma_2nd);

    let k_trace = sum2(|i, j| k[i][j] * g_inv[i][j]);
    let k_trace_grad =
        tensor1(|i| sum2(|m, n| k_par[m][n][i] * g_inv[m][n] + k[m][n] * g_inv_par[m][n][i]));

    let k_con = tensor2(|i, j| sum2(|m, n| g_inv[i][m] * g_inv[j][n] * k[m][n]));

    let l_grad = l_par;

    let theta_grad = theta_par;
    let z_grad = grad1(z, z_par, gamma_2nd);

    let lapse_grad = lapse_par;
    let lapse_hess = hess0(lapse_par, lapse_par2, gamma_2nd);

    // *********************************
    // Scalar Field ********************
    // *********************************

    let phi_grad = phi_par;
    let phi_hess = hess0(phi_par, phi_par2, gamma_2nd);
    let phi_lap = sum2(|i, j| phi_hess[i][j] * g_inv[i][j]);

    let pi_grad = pi_par;

    let tau = -0.5 * M * M * (phi * phi);
    let sigma = pi * pi - tau;

    let s_a = tensor1(|i| -pi * phi_grad[i]);
    let s_ab = tensor2(|i, j| phi_grad[i] * phi_grad[j] + g[i][j] * tau);
    let s_trace = sum2(|m, n| s_ab[m][n] * g_inv[m][n]);

    // *********************************
    // Hamiltonian *********************
    // *********************************

    let hamiltonian = {
        let term1 = 0.5 * (ricci_trace + k_trace * k_trace) + k_trace * l;
        let term2 = -0.5 * sum2(|i, j| k[i][j] * k_con[i][j]);
        let term3 = -sum2(|i, j| lam_hess[i][j] * g_inv[i][j]);

        term1 + term2 + term3 - KAPPA * sigma
    };

    // **********************************
    // Momentum *************************
    // **********************************

    let momentum = tensor1(|i| {
        let term1 = -k_trace_grad[i] - l_grad[i];
        let term2 = sum2(|m, n| k_grad[i][m][n] * g_inv[m][n]);

        let mut regular = sum1(|m| lam_reg_con[m] * k[m][i]) - lam_reg_co[i] * l;

        if on_axis && i == 0 {
            regular += -y;
        } else if on_axis && i == 1 {
            regular += k_par[0][1][0] / g[0][0];
        }

        term1 + term2 + regular - KAPPA * s_a[i]
    });

    // ***********************************
    // Metric ****************************
    // ***********************************

    let g_t = {
        let term1 = tensor2(|i, j| -2.0 * lapse * k[i][j]);
        let term2 = lie2(g, g_par, shift, shift_par);
        tensor2(|i, j| term1[i][j] + term2[i][j])
    };

    // (partial_t lambda) / lambda
    let mut lam_lt = -lapse * l + sum1(|i| lam_reg_co[i] * shift[i]);
    if on_axis {
        lam_lt += shift_par[0][0];
    }

    // ***********************************
    // Extrinsic curvature ***************
    // ***********************************

    let k_lie_shift = lie2(k, k_par, shift, shift_par);
    let k_t = tensor2(|i, j| {
        let term1 = lapse * ricci[i][j] - lapse * lam_hess[i][j] - lapse_hess[i][j];
        let term2 = lapse * (k_trace + l) * k[i][j];
        let term3 = -2.0 * lapse * sum2(|m, n| k[i][m] * g_inv[m][n] * k[n][j]);
        let term4 = lapse * (z_grad[i][j] + z_grad[j][i] - 2.0 * k[i][j] * theta);
        let term5 = -KAPPA * (s_ab[i][j] + 0.5 * g[i][j] * (sigma - s_trace - tau));

        term1 + term2 + term3 + term4 + term5 + k_lie_shift[i][j]
    });

    let l_t = {
        let term1 = lapse * l * (k_trace + l - 2.0 * theta);
        let term2 = -lapse * sum2(|i, j| lam_hess[i][j] * g_inv[i][j]);
        let term3 = sum1(|i| shift[i] * l_par[i]);
        let term4 = -0.5 * (sigma - s_trace + tau);

        let mut regular = sum1(|m| lam_reg_con[m] * (2.0 * lapse * z[m] - lapse_grad[m]));
        if on_axis {
            regular += (2.0 * lapse * z_par[0][0] - lapse_par2[0][0]) / g[0][0];
        }

        term1 + term2 + term3 + term4 + regular
    };

    // ************************************
    // Constraint *************************
    // ************************************

    let theta_t = {
        let term1 = lapse * hamiltonian - lapse * (k_trace + l) * theta;
        let term2 = lapse * sum2(|i, j| z_grad[i][j] * g_inv[i][j]);
        let term3 = -sum2(|i, j| lapse_grad[i] * g_inv[i][j] * z[j]);
        let term4 = sum1(|i| theta_par[i] * shift[i]);

        let mut regular = lapse * sum1(|m| lam_reg_con[m] * z[m]);
        if on_axis {
            regular += lapse * z_par[0][0] / g[0][0];
        }

        term1 + term2 + term3 + term4 + regular
    };

    let z_lie_shift = lie1(z, z_par, shift, shift_par);
    let z_t = tensor1(|i| {
        let term1 = lapse * momentum[i];
        let term2 = -2.0 * lapse * sum2(|m, n| k[i][m] * g_inv[m][n] * z[n]);
        let term3 = lapse * theta_grad[i] - lapse_grad[i] * theta;
        term1 + term2 + term3 + z_lie_shift[i]
    });

    const F: f64 = 1.0;
    const A: f64 = 1.0;
    const MU: f64 = 1.0;
    const D: f64 = 1.0;
    const M: f64 = 2.0;

    let lapse_t = {
        let term1 = -lapse * lapse * F * (k_trace + l - M * theta);
        let term2 = sum1(|i| shift[i] * lapse_par[i]);
        term1 + term2
    };

    let shift_t = tensor1(|i| {
        let lamg_term = 0.5 / g_det * sum1(|m| g_inv[i][m] * g_det_par[m]) + lam_reg_con[i];

        let g_inv_term: f64 = sum1(|m| g_inv_par[i][m][m]);

        let term1 = -lapse * lapse * MU * g_inv_term;
        let term2 = lapse * lapse * 2.0 * MU * sum1(|m| g_inv[i][m] * z[m]);
        let term3 = -lapse * A * sum1(|m| g_inv[i][m] * lapse_par[m]);
        let term4 = sum1(|m| shift[m] * shift_par[i][m]);

        let mut regular = -lapse * lapse * (2.0 * MU - D) * lamg_term;
        if !on_axis && i == 0 {
            regular += lapse * lapse * (2.0 * MU - D) * g_inv[0][0] / pos[0];
        }

        term1 + term2 + term3 + term4 + regular
    });

    let mut y_t = 0.0;
    let mut s_t = 0.0;

    if !on_axis {
        y_t = (l_t - k_t[0][0] / g[0][0] + k[0][0] / g[0][0].powi(2) * g_t[0][0]) / pos[0];
        s_t = (lam_lt - 0.5 * g_t[0][0] / g[0][0]) / pos[0];
    }

    // *********************************
    // Energy Momentum *****************
    // *********************************

    let phi_t = {
        let term1 = lapse * pi;

        term1 + sum1(|i| shift[i] * phi_grad[i])
    };

    let pi_t = {
        let term1 = lapse * phi_lap + lapse * pi * (k_trace + l);
        let term2 = sum2(|i, j| g_inv[i][j] * phi_grad[i] * lapse_grad[j]);
        let mut term3 = lapse * sum1(|i| phi_grad[i] * lam_reg_con[i]);

        if on_axis {
            term3 += sys.phi_rr / sys.grr;
        }

        let term4 = -M * M * phi * lapse;

        term1 + term2 + term3 + term4 + sum1(|i| shift[i] * pi_grad[i])
    };

    HyperbolicDerivs {
        grr_t: g_t[0][0],
        grz_t: g_t[0][1],
        gzz_t: g_t[1][1],
        s_t,

        krr_t: k_t[0][0],
        krz_t: k_t[0][1],
        kzz_t: k_t[1][1],
        y_t,

        theta_t,
        zr_t: z_t[0],
        zz_t: z_t[1],

        lapse_t,
        shiftr_t: shift_t[0],
        shiftz_t: shift_t[1],

        phi_t,
        pi_t,
    }
}

pub fn geometric(sys: HyperbolicSystem, _pos: [f64; 2]) -> Geometric {
    // ******************************
    // Unpack variables
    // ******************************

    // Metric
    let g = sys.metric();
    let g_par = sys.metric_par();
    let g_par2 = sys.metric_par2();

    // **************************************
    // Geometry *****************************
    // **************************************

    let g_det = det2(g);
    let g_det_par = det_par2(g, g_par);

    let g_inv = [
        [sys.gzz / g_det, -sys.grz / g_det],
        [-sys.grz / g_det, sys.grr / g_det],
    ];
    let g_inv_par = {
        let grr_inv_par =
            tensor1(|i| g_par[1][1][i] / g_det - g[1][1] / (g_det * g_det) * g_det_par[i]);
        let grz_inv_par =
            tensor1(|i| -g_par[0][1][i] / g_det + g[0][1] / (g_det * g_det) * g_det_par[i]);
        let gzz_inv_par =
            tensor1(|i| g_par[0][0][i] / g_det - g[0][0] / (g_det * g_det) * g_det_par[i]);

        [[grr_inv_par, grz_inv_par], [grz_inv_par, gzz_inv_par]]
    };

    // Next compute Christoffel symbols
    let gamma = christoffel(g_par);
    let gamma_par = christoffel_par(g_par2);

    let gamma_2nd = christoffel_2nd(gamma, g_inv);
    let gamma_2nd_par = christoffel_2nd_par(gamma, gamma_par, g_inv, g_inv_par);

    // And now the Ricci tensor
    let ricci = ricci(gamma_2nd, gamma_2nd_par);

    Geometric {
        ricci_rr: ricci[0][0],
        ricci_rz: ricci[0][1],
        ricci_zz: ricci[1][1],
        gamma_rrr: gamma_2nd[0][0][0],
        gamma_rrz: gamma_2nd[0][0][1],
        gamma_rzz: gamma_2nd[0][1][1],
        gamma_zrr: gamma_2nd[1][0][0],
        gamma_zrz: gamma_2nd[1][0][1],
        gamma_zzz: gamma_2nd[1][1][1],
    }
}
