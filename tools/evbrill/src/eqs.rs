#![allow(unused_assignments)]

// A very useful function
use std::array::from_fn;

// *********************************
// Tensor Utils ********************
// *********************************

type Rank1 = [f64; 2];
type Rank2 = [[f64; 2]; 2];
type Rank3 = [[[f64; 2]; 2]; 2];
type Rank4 = [[[[f64; 2]; 2]; 2]; 2];

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
    pub fn metric(&self) -> Rank2 {
        [[self.grr, self.grz], [self.grz, self.gzz]]
    }

    pub fn metric_par(&self) -> Rank3 {
        let grr_par = [self.grr_r, self.grr_z];
        let grz_par = [self.grz_r, self.grz_z];
        let gzz_par = [self.gzz_r, self.gzz_z];

        [[grr_par, grz_par], [grz_par, gzz_par]]
    }

    pub fn metric_par2(&self) -> Rank4 {
        let grr_par2 = [[self.grr_rr, self.grr_rz], [self.grr_rz, self.grr_zz]];
        let grz_par2 = [[self.grz_rr, self.grz_rz], [self.grz_rz, self.grz_zz]];
        let gzz_par2 = [[self.gzz_rr, self.gzz_rz], [self.gzz_rz, self.gzz_zz]];

        [[grr_par2, grz_par2], [grz_par2, gzz_par2]]
    }

    pub fn seed(&self) -> f64 {
        self.s
    }

    pub fn seed_par(&self) -> Rank1 {
        [self.s_r, self.s_z]
    }

    pub fn seed_par2(&self) -> Rank2 {
        [[self.s_rr, self.s_rz], [self.s_rz, self.s_zz]]
    }

    pub fn extrinsic(&self) -> Rank2 {
        [[self.krr, self.krz], [self.krz, self.kzz]]
    }

    pub fn extrinsic_par(&self) -> Rank3 {
        let krr_par = [self.krr_r, self.krr_z];
        let krz_par = [self.krz_r, self.krz_z];
        let kzz_par = [self.kzz_r, self.kzz_z];

        [[krr_par, krz_par], [krz_par, kzz_par]]
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
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

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Geometric {
    pub ricci_rr: f64,
    pub ricci_rz: f64,
    pub ricci_zz: f64,

    pub gamma_rrr: f64,
    pub gamma_rrz: f64,
    pub gamma_rzr: f64,
    pub gamma_rzz: f64,

    pub gamma_zrr: f64,
    pub gamma_zrz: f64,
    pub gamma_zzr: f64,
    pub gamma_zzz: f64,

    pub gamma_rrr_r: f64,

    pub g_inv_rr_r: f64,

    pub g_det_r: f64,
    pub g_det_z: f64,
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
    // Hamiltonian *********************
    // *********************************

    let hamiltonian = {
        let term1 = 0.5 * (ricci_trace + k_trace * k_trace) + k_trace * l;
        let term2 = -0.5 * sum2(|i, j| k[i][j] * k_con[i][j]);
        let term3 = -sum2(|i, j| lam_hess[i][j] * g_inv[i][j]);

        term1 + term2 + term3
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

        term1 + term2 + regular
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

        term1 + term2 + term3 + term4 + k_lie_shift[i][j]
    });

    let l_t = {
        let term1 = lapse * l * (k_trace + l - 2.0 * theta);
        let term2 = -lapse * sum2(|i, j| lam_hess[i][j] * g_inv[i][j]);
        let term3 = sum1(|i| shift[i] * l_par[i]);

        let mut regular = sum1(|m| lam_reg_con[m] * (2.0 * lapse * z[m] - lapse_grad[m]));
        if on_axis {
            regular += (2.0 * lapse * z_par[0][0] - lapse_par2[0][0]) / g[0][0];
        }

        term1 + term2 + term3 + regular
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
    }
}

pub fn geometric(sys: HyperbolicSystem, pos: [f64; 2]) -> Geometric {
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
    // // As well as the Ricci scalar
    // let ricci_trace = sum2(|i, j| ricci[i][j] * g_inv[i][j]);

    Geometric {
        ricci_rr: ricci[0][0],
        ricci_rz: ricci[0][1],
        ricci_zz: ricci[1][1],
        gamma_rrr: gamma_2nd[0][0][0],
        gamma_rrz: gamma_2nd[0][0][1],
        gamma_rzr: gamma_2nd[0][1][0],
        gamma_rzz: gamma_2nd[0][1][1],
        gamma_zrr: gamma_2nd[1][0][0],
        gamma_zrz: gamma_2nd[1][0][1],
        gamma_zzr: gamma_2nd[1][1][0],
        gamma_zzz: gamma_2nd[1][1][1],

        gamma_rrr_r: gamma_2nd_par[0][0][0][0],

        g_inv_rr_r: g_inv_par[0][0][0],
        g_det_r: g_det_par[0],
        g_det_z: g_det_par[1],
    }
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use crate::eqs;
    use crate::symbolicc;

    use super::{tensor3, HyperbolicSystem};
    use aeon::{fd::NodeSpace, geometry::Rectangle};
    use rand::prelude::*;

    #[test]
    fn tensor_utils() {
        let tensor = tensor3(|_, _, k| k as f64);
        assert_eq!(tensor[0][1][0], 0.0);
        assert_eq!(tensor[1][1][1], 1.0);

        let tensor = tensor3(|_, j, _| j as f64);
        assert_eq!(tensor[0][1][0], 1.0);
        assert_eq!(tensor[1][1][1], 1.0);
    }

    #[test]
    fn minkowski() {
        let space = NodeSpace {
            size: [100, 100],
            ghost: 2,
            context: Rectangle {
                origin: [0.0, 0.0],
                size: [10.0, 10.0],
            },
        };

        for node in space.inner_window().iter() {
            let position = space.position(node);

            let system = HyperbolicSystem {
                grr: 1.0,
                grr_r: 0.0,
                grr_z: 0.0,
                grr_rr: 0.0,
                grr_rz: 0.0,
                grr_zz: 0.0,

                grz: 0.0,
                grz_r: 0.0,
                grz_z: 0.0,
                grz_rr: 0.0,
                grz_rz: 0.0,
                grz_zz: 0.0,

                gzz: 1.0,
                gzz_r: 0.0,
                gzz_z: 0.0,
                gzz_rr: 0.0,
                gzz_rz: 0.0,
                gzz_zz: 0.0,

                s: 0.0,
                s_r: 0.0,
                s_z: 0.0,
                s_rr: 0.0,
                s_rz: 0.0,
                s_zz: 0.0,

                krr: 0.0,
                krr_r: 0.0,
                krr_z: 0.0,
                krz: 0.0,
                krz_r: 0.0,
                krz_z: 0.0,
                kzz: 0.0,
                kzz_r: 0.0,
                kzz_z: 0.0,
                y: 0.0,
                y_r: 0.0,
                y_z: 0.0,

                theta: 0.0,
                theta_r: 0.0,
                theta_z: 0.0,

                zr: 0.0,
                zr_r: 0.0,
                zr_z: 0.0,

                zz: 0.0,
                zz_r: 0.0,
                zz_z: 0.0,

                lapse: 1.0,
                lapse_r: 0.0,
                lapse_z: 0.0,
                lapse_rr: 0.0,
                lapse_rz: 0.0,
                lapse_zz: 0.0,

                shiftr: 0.0,
                shiftr_r: 0.0,
                shiftr_z: 0.0,

                shiftz: 0.0,
                shiftz_r: 0.0,
                shiftz_z: 0.0,
            };

            let derivs = eqs::hyperbolic(system, position);

            assert_eq!(derivs.grr_t, 0.0);
            assert_eq!(derivs.grz_t, 0.0);
            assert_eq!(derivs.gzz_t, 0.0);
            assert_eq!(derivs.s_t, 0.0);

            assert_eq!(derivs.krr_t, 0.0);
            assert_eq!(derivs.krz_t, 0.0);
            assert_eq!(derivs.kzz_t, 0.0);
            assert_eq!(derivs.y_t, 0.0);

            assert_eq!(derivs.lapse_t, 0.0);
            assert_eq!(derivs.shiftr_t, 0.0);
            assert_eq!(derivs.shiftz_t, 0.0);

            assert_eq!(derivs.theta_t, 0.0);
            assert_eq!(derivs.zr_t, 0.0);
            assert_eq!(derivs.zz_t, 0.0);
        }
    }

    #[test]
    fn kasner() {
        let space = NodeSpace {
            size: [100, 100],
            ghost: 2,
            context: Rectangle {
                origin: [0.0, 1.0],
                size: [10.0, 10.0],
            },
        };

        for node in space.inner_window().iter() {
            let [rho, z] = space.position(node);

            let conformal = z.powi(4);
            let conformal_r = 0.0;
            let conformal_rr = 0.0;
            let conformal_z = 4.0 * z.powi(3);
            let conformal_zz = 12.0 * z.powi(2);
            let conformal_rz = 0.0;

            let lapse = z.powi(-1);
            let lapse_r = 0.0;
            let lapse_rr = 0.0;
            let lapse_z = -z.powi(-2);
            let lapse_zz = 2.0 * z.powi(-3);
            let lapse_rz = 0.0;

            let system = HyperbolicSystem {
                grr: conformal,
                grr_r: conformal_r,
                grr_z: conformal_z,
                grr_rr: conformal_rr,
                grr_rz: conformal_rz,
                grr_zz: conformal_zz,

                grz: 0.0,
                grz_r: 0.0,
                grz_z: 0.0,
                grz_rr: 0.0,
                grz_rz: 0.0,
                grz_zz: 0.0,

                gzz: conformal,
                gzz_r: conformal_r,
                gzz_z: conformal_z,
                gzz_rr: conformal_rr,
                gzz_rz: conformal_rz,
                gzz_zz: conformal_zz,

                s: 0.0,
                s_r: 0.0,
                s_z: 0.0,
                s_rr: 0.0,
                s_rz: 0.0,
                s_zz: 0.0,

                krr: 0.0,
                krr_r: 0.0,
                krr_z: 0.0,
                krz: 0.0,
                krz_r: 0.0,
                krz_z: 0.0,
                kzz: 0.0,
                kzz_r: 0.0,
                kzz_z: 0.0,
                y: 0.0,
                y_r: 0.0,
                y_z: 0.0,

                theta: 0.0,
                theta_r: 0.0,
                theta_z: 0.0,

                zr: 0.0,
                zr_r: 0.0,
                zr_z: 0.0,

                zz: 0.0,
                zz_r: 0.0,
                zz_z: 0.0,

                lapse,
                lapse_r,
                lapse_z,
                lapse_rr,
                lapse_rz,
                lapse_zz,

                shiftr: 0.0,
                shiftr_r: 0.0,
                shiftr_z: 0.0,

                shiftz: 0.0,
                shiftz_r: 0.0,
                shiftz_z: 0.0,
            };

            let derivs = eqs::hyperbolic(system, [rho, z]);
            // println!("{derivs:?}");

            // break;

            macro_rules! assert_almost_eq {
                ($val:expr, $target:expr) => {
                    assert!(($val - $target).abs() <= 10e-15)
                };
                ($val:expr, $target:expr, $extra:tt) => {
                    assert!(($val - $target).abs() <= 10e-15, $extra)
                };
            }

            assert_almost_eq!(derivs.grr_t, 0.0);
            assert_almost_eq!(derivs.grz_t, 0.0);
            assert_almost_eq!(derivs.gzz_t, 0.0);
            assert_almost_eq!(derivs.s_t, 0.0);

            assert_almost_eq!(derivs.krr_t, 0.0);
            assert_almost_eq!(derivs.krz_t, 0.0);
            assert_almost_eq!(derivs.kzz_t, 0.0);
            assert_almost_eq!(derivs.y_t, 0.0);

            assert_almost_eq!(derivs.lapse_t, 0.0);
            assert_almost_eq!(derivs.shiftr_t, 0.0);
            // assert_eq!(derivs.shiftz_t, -1.0);

            assert_almost_eq!(derivs.theta_t, 0.0);
            assert_almost_eq!(derivs.zr_t, 0.0);
            assert_almost_eq!(derivs.zz_t, 0.0);
        }
    }

    #[test]
    fn symbolicc() {
        const NEAR_ONE: Range<f64> = 0.2..1.8;
        const NEAR_ZERO: Range<f64> = -1.0..1.0;

        let mut rng = rand::thread_rng();

        macro_rules! assert_almost_eq {
            ($val:expr, $target:expr) => {
                assert!(($val - $target).abs() <= 10e-10 * $val.abs().max($target.abs()))
            };
            ($val:expr, $target:expr, $extra:tt) => {
                if ($val - $target).abs() > 10e-10 * $val.abs().max($target.abs()) {
                    panic!(
                        "{}, value {:e}, difference: {:e}",
                        $extra,
                        $val,
                        ($val - $target).abs()
                    );
                }
            };
        }

        for trial in 0..10000 {
            println!("Running trial {trial}");

            let system = HyperbolicSystem {
                grr: rng.gen_range(NEAR_ONE),
                grr_r: rng.gen_range(NEAR_ZERO),
                grr_z: rng.gen_range(NEAR_ZERO),
                grr_rr: rng.gen_range(NEAR_ZERO),
                grr_rz: rng.gen_range(NEAR_ZERO),
                grr_zz: rng.gen_range(NEAR_ZERO),
                grz: rng.gen_range(NEAR_ZERO),
                grz_r: rng.gen_range(NEAR_ZERO),
                grz_z: rng.gen_range(NEAR_ZERO),
                grz_rr: rng.gen_range(NEAR_ZERO),
                grz_rz: rng.gen_range(NEAR_ZERO),
                grz_zz: rng.gen_range(NEAR_ZERO),
                gzz: rng.gen_range(NEAR_ONE),
                gzz_r: rng.gen_range(NEAR_ZERO),
                gzz_z: rng.gen_range(NEAR_ZERO),
                gzz_rr: rng.gen_range(NEAR_ZERO),
                gzz_rz: rng.gen_range(NEAR_ZERO),
                gzz_zz: rng.gen_range(NEAR_ZERO),
                s: rng.gen_range(NEAR_ZERO),
                s_r: rng.gen_range(NEAR_ZERO),
                s_z: rng.gen_range(NEAR_ZERO),
                s_rr: rng.gen_range(NEAR_ZERO),
                s_rz: rng.gen_range(NEAR_ZERO),
                s_zz: rng.gen_range(NEAR_ZERO),
                // s: 0.0,
                // s_r: 0.0,
                // s_z: 0.0,
                // s_rr: 0.0,
                // s_rz: 0.0,
                // s_zz: 0.0,
                krr: rng.gen_range(NEAR_ZERO),
                krr_r: rng.gen_range(NEAR_ZERO),
                krr_z: rng.gen_range(NEAR_ZERO),
                krz: rng.gen_range(NEAR_ZERO),
                krz_r: rng.gen_range(NEAR_ZERO),
                krz_z: rng.gen_range(NEAR_ZERO),
                kzz: rng.gen_range(NEAR_ZERO),
                kzz_r: rng.gen_range(NEAR_ZERO),
                kzz_z: rng.gen_range(NEAR_ZERO),
                y: rng.gen_range(NEAR_ZERO),
                y_r: rng.gen_range(NEAR_ZERO),
                y_z: rng.gen_range(NEAR_ZERO),
                // krr: 0.0,
                // krr_r: 0.0,
                // krr_z: 0.0,
                // krz: 0.0,
                // krz_r: 0.0,
                // krz_z: 0.0,
                // kzz: 0.0,
                // kzz_r: 0.0,
                // kzz_z: 0.0,
                // y: 0.0,
                // y_r: 0.0,
                // y_z: 0.0,
                lapse: rng.gen_range(NEAR_ONE),
                lapse_r: rng.gen_range(NEAR_ZERO),
                lapse_z: rng.gen_range(NEAR_ZERO),
                lapse_rr: rng.gen_range(NEAR_ZERO),
                lapse_rz: rng.gen_range(NEAR_ZERO),
                lapse_zz: rng.gen_range(NEAR_ZERO),
                shiftr: rng.gen_range(NEAR_ZERO),
                shiftr_r: rng.gen_range(NEAR_ZERO),
                shiftr_z: rng.gen_range(NEAR_ZERO),
                shiftz: rng.gen_range(NEAR_ZERO),
                shiftz_r: rng.gen_range(NEAR_ZERO),
                shiftz_z: rng.gen_range(NEAR_ZERO),
                // lapse: 1.0,
                // lapse_r: 0.0,
                // lapse_z: 0.0,
                // lapse_rr: 0.0,
                // lapse_rz: 0.0,
                // lapse_zz: 0.0,
                // shiftr: 0.0,
                // shiftr_r: 0.0,
                // shiftr_z: 0.0,
                // shiftz: 0.0,
                // shiftz_r: 0.0,
                // shiftz_z: 0.0,
                theta: rng.gen_range(NEAR_ZERO),
                theta_r: rng.gen_range(NEAR_ZERO),
                theta_z: rng.gen_range(NEAR_ZERO),
                zr: rng.gen_range(NEAR_ZERO),
                zr_r: rng.gen_range(NEAR_ZERO),
                zr_z: rng.gen_range(NEAR_ZERO),
                zz: rng.gen_range(NEAR_ZERO),
                zz_r: rng.gen_range(NEAR_ZERO),
                zz_z: rng.gen_range(NEAR_ZERO),
                // theta: 0.0,
                // theta_r: 0.0,
                // theta_z: 0.0,
                // zr: 0.0,
                // zr_r: 0.0,
                // zr_z: 0.0,
                // zz: 0.0,
                // zz_r: 0.0,
                // zz_z: 0.0,
            };

            let rho = rng.gen_range(0.1..10.0);
            let z = rng.gen_range(0.0..10.0);

            let det = system.grr * system.gzz - system.grz * system.grz;

            if det.abs() <= 1e-3 {
                continue;
            }

            println!("Det {:e}", det);

            // println!("{}", system.grr * system.gzz - system.grz * system.grz);

            // let fmore = system.gzz_r / det
            //     - (system.gzz * system.grr_r - 2.0 * system.grz * system.grz_r
            //         + system.grr * system.gzz_r)
            //         * system.gzz
            //         / (det * det);

            // let fmore = (-system.grr_r * system.gzz.powi(2)
            //     + 2.0 * system.grz * system.grz_r * system.gzz
            //     - system.grz * system.grz * system.gzz_r)
            //     / (det * det);

            // dbg!(system.clone());

            // let more = (-system.grr_r * system.gzz.powi(2)
            //     + 2.0 * system.grz * system.grz_r * system.gzz
            //     - system.gzz_r * system.grz * system.grz)
            //     / (system.grr.powi(2) * system.gzz.powi(2) + system.grz.powi(4)
            //         - system.grr * system.gzz * 2.0 * system.grz.powi(2));

            let explicit = eqs::geometric(system.clone(), [rho, z]);
            let symbolic = symbolicc::geometric(system.clone(), rho, z);

            assert_almost_eq!(
                explicit.gamma_rrr,
                symbolic.gamma_rrr,
                "Gammarrr does not match"
            );
            assert_almost_eq!(
                explicit.gamma_rrz,
                symbolic.gamma_rrz,
                "Gammarrz does not match"
            );
            assert_almost_eq!(
                explicit.gamma_rzr,
                symbolic.gamma_rzr,
                "Gammarzr does not match"
            );
            assert_almost_eq!(
                explicit.gamma_rzz,
                symbolic.gamma_rzz,
                "Gammarzz does not match"
            );

            assert_almost_eq!(
                explicit.gamma_zrr,
                symbolic.gamma_zrr,
                "Gammazrr does not match"
            );
            assert_almost_eq!(
                explicit.gamma_zrz,
                symbolic.gamma_zrz,
                "Gammazrz does not match"
            );
            assert_almost_eq!(
                explicit.gamma_zzr,
                symbolic.gamma_zzr,
                "Gammazzr does not match"
            );
            assert_almost_eq!(
                explicit.gamma_zzz,
                symbolic.gamma_zzz,
                "Gammazzz does not match"
            );

            assert_almost_eq!(explicit.g_det_r, symbolic.g_det_r, "Gdet_r does not match");

            assert_almost_eq!(explicit.g_det_z, symbolic.g_det_z, "Gdet_z does not match");

            // assert_almost_eq!(fmore, explicit.g_inv_rr_r, "More vs Explicit");

            // assert_almost_eq!(fmore, symbolic.g_inv_rr_r, "More vs Symbolic");

            assert_almost_eq!(
                explicit.g_inv_rr_r,
                symbolic.g_inv_rr_r,
                "Ginvrr_r does not match"
            );

            assert_almost_eq!(
                explicit.gamma_rrr_r,
                symbolic.gamma_rrr_r,
                "Gammarrrr does not match"
            );

            assert_almost_eq!(explicit.ricci_rr, symbolic.ricci_rr, "Rrr does not match");
            assert_almost_eq!(explicit.ricci_rz, symbolic.ricci_rz, "Rrz does not match");
            assert_almost_eq!(explicit.ricci_zz, symbolic.ricci_zz, "Rzz does not match");

            let explicit = eqs::hyperbolic(system.clone(), [rho, z]);
            let symbolic = symbolicc::hyperbolic(system, rho, z);

            assert_almost_eq!(explicit.grr_t, symbolic.grr_t, "Grr does not match");
            assert_almost_eq!(explicit.grz_t, symbolic.grz_t, "Grz does not match");
            assert_almost_eq!(explicit.gzz_t, symbolic.gzz_t, "Gzz does not match");
            assert_almost_eq!(explicit.s_t, symbolic.s_t, "S does not match");

            assert_almost_eq!(explicit.krr_t, symbolic.krr_t, "Krr does not match");
            assert_almost_eq!(explicit.krz_t, symbolic.krz_t, "Krz does not match");
            assert_almost_eq!(explicit.kzz_t, symbolic.kzz_t, "Kzz does not match");
            assert_almost_eq!(explicit.y_t, symbolic.y_t, "Y does not match");

            assert_almost_eq!(explicit.lapse_t, symbolic.lapse_t, "Lapse does not match");
            assert_almost_eq!(
                explicit.shiftr_t,
                symbolic.shiftr_t,
                "Shiftr does not match"
            );
            assert_almost_eq!(
                explicit.shiftz_t,
                symbolic.shiftz_t,
                "Shiftz does not match"
            );

            assert_almost_eq!(explicit.theta_t, symbolic.theta_t, "Theta does not match");
            assert_almost_eq!(explicit.zr_t, symbolic.zr_t, "Zr does not match");
            assert_almost_eq!(explicit.zz_t, symbolic.zz_t, "Zz does not match");
        }
    }
}
