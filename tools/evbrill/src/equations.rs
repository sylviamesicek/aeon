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

#[inline]
pub fn sum3(mut f: impl FnMut(usize, usize, usize) -> f64) -> f64 {
    sum1(|i| sum1(|j| sum1(|k| f(i, j, k))))
}

#[inline]
pub fn sum4(mut f: impl FnMut(usize, usize, usize, usize) -> f64) -> f64 {
    sum1(|i| sum1(|j| sum1(|k| sum1(|l| f(i, j, k, l)))))
}

#[inline]
pub fn grad1(value: Rank1, pars: Rank2, connect: Rank3) -> Rank2 {
    tensor2(|i, j| pars[i][j] - sum1(|k| connect[k][i][j] * value[k]))
}

#[inline]
pub fn grad2(value: Rank2, pars: Rank3, connect: Rank3) -> Rank3 {
    tensor3(|i, j, k| {
        pars[i][j][k] - sum1(|l| connect[l][i][k] * value[l][j] + connect[l][j][k] * value[i][l])
    })
}

#[inline]
pub fn hess0(pars: Rank1, pars2: Rank2, connect: Rank3) -> Rank2 {
    tensor2(|i, j| pars2[i][j] - sum1(|k| connect[k][i][j] * pars[k]))
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

pub fn christoffel(g_inv: Rank2, g_par: Rank3) -> Rank3 {
    tensor3(|i, j, k| {
        sum1(|l| 0.5 * g_inv[i][l] * (g_par[l][j][k] + g_par[k][l][j] - g_par[j][k][l]))
    })
}

pub fn chirstofel_par(g_inv: Rank2, g_inv_par: Rank3, g_par: Rank3, g_par2: Rank4) -> Rank4 {
    tensor4(|i, j, k, l| {
        let term1 =
            sum1(|m| 0.5 * g_inv_par[i][m][l] * (g_par[m][j][k] + g_par[k][m][j] - g_par[j][k][m]));
        let term2 = sum1(|m| {
            0.5 * g_inv[i][m] * (g_par2[m][j][k][l] + g_par2[k][m][j][l] - g_par2[j][k][m][l])
        });
        term1 + term2
    })
}

pub fn ricci(gamma: Rank3, gamma_par: Rank4) -> Rank2 {
    tensor2(|j, k| {
        let term1 = sum1(|i| gamma_par[i][j][k][i] - gamma_par[i][k][i][j]);
        let term2 = sum2(|i, p| gamma[i][i][p] * gamma[p][j][k] - gamma[i][j][p] * gamma[p][i][k]);
        term1 + term2
    })
}

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

    pub debug1: f64,
    pub debug2: f64,
}

pub fn hyperbolic(sys: HyperbolicSystem, pos: [f64; 2]) -> HyperbolicDerivs {
    let on_axis = pos[0].abs() <= 10e-10;

    // ******************************
    // Unpack variables
    // ******************************

    // Metric
    let g: [[f64; 2]; 2] = [[sys.grr, sys.grz], [sys.grz, sys.gzz]];

    let g_par = {
        let grr_par = [sys.grr_r, sys.grr_z];
        let grz_par = [sys.grz_r, sys.grz_z];
        let gzz_par = [sys.gzz_r, sys.gzz_z];

        [[grr_par, grz_par], [grz_par, gzz_par]]
    };

    let g_par2 = {
        let grr_par2 = [[sys.grr_rr, sys.grr_rz], [sys.grr_rz, sys.grr_zz]];
        let grz_par2 = [[sys.grz_rr, sys.grz_rz], [sys.grz_rz, sys.grz_zz]];
        let gzz_par2 = [[sys.gzz_rr, sys.gzz_rz], [sys.gzz_rz, sys.gzz_zz]];

        [[grr_par2, grz_par2], [grz_par2, gzz_par2]]
    };

    let s = sys.s;
    let s_par = [sys.s_r, sys.s_z];
    let s_par2 = [[sys.s_rr, sys.s_rz], [sys.s_rz, sys.s_zz]];

    let lam = pos[0] * (pos[0] * s).exp() * g[0][0].sqrt();

    // Extrinsic curvature
    let k: [[f64; 2]; 2] = [[sys.krr, sys.krz], [sys.krz, sys.kzz]];
    let k_par = {
        let krr_par = [sys.krr_r, sys.krr_z];
        let krz_par = [sys.krz_r, sys.krz_z];
        let kzz_par = [sys.kzz_r, sys.kzz_z];

        [[krr_par, krz_par], [krz_par, kzz_par]]
    };

    let y = sys.y;
    let y_par = [sys.y_r, sys.y_z];

    let l = pos[0] * y + k[0][0] / g[0][0];
    let l_par = {
        // Apply product rule to definition of l.
        let mut result = tensor1(|i| {
            pos[0] * y_par[i] + k_par[0][0][i] / g[0][0]
                - k[0][0] / g[0][0].powi(2) * g_par[0][0][i]
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

    let g_det = g[0][0] * g[1][1] - g[0][1] * g[0][1];
    let g_det_par = tensor1(|i| {
        g_par[0][0][i] * g[1][1] + g[0][0] * g_par[1][1][i] - 2.0 * g[0][1] * g_par[0][1][i]
    });

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
    let gamma = christoffel(g_inv, g_par);
    let gamma_par = chirstofel_par(g_inv, g_inv_par, g_par, g_par2);

    // And now the Ricci tensor
    let ricci = ricci(gamma, gamma_par);
    // As well as the Ricci scalar
    let ricci_trace = sum2(|i, j| ricci[i][j] * g_inv[i][j]);

    // *********************************
    // Contractions and Derivatives ****
    // *********************************

    // Logrithmic derivatives of lambda

    // Λ is 𝓞(1/r) on axis, so we split into a regular part and a 1/r part in the r component.
    // Off axis, this is the full gradient of lambda, including that 1/r term, but on axis this
    // term is set to zero, and one must apply lhopital's rule on a case by case basis.
    let mut lam_lgrad = [0.0; 2];
    // The logrithmic hessian has no such issues, and is regular everywhere
    let mut lam_lhess = [[0.0; 2]; 2];

    {
        let g_par_term = tensor1(|i| g_par[0][0][i] / g[0][0]);
        let g_par2_term = tensor2(|i, j| {
            g_par2[0][0][i][j] / g[0][0] - g_par[0][0][i] * g_par[0][0][j] / (g[0][0] * g[0][0])
        });

        // Decompose lam_r into a regular part and a Order(1/r) part.
        let lam_r = s + pos[0] * s_par[0] + 0.5 * g_par_term[0];
        let lam_z = pos[0] * s_par[1] + 0.5 * g_par_term[1];

        // Use lhopital's rule to compute on axis lam_r / r and lam_z / r on axis.
        let lam_r_lhopital = 2.0 * s_par[0] + 0.5 * g_par2[0][0][0][0] / g[0][0];
        let lam_z_lhopital = s_par[1]; // plus an irregular term that gets canceled by gamma_regular

        lam_lgrad[0] = lam_r;
        lam_lgrad[1] = lam_z;

        if !on_axis {
            lam_lgrad[0] += 1.0 / pos[0];
        }

        let mut gamma_regular = tensor2(|i, j| sum1(|m| gamma[m][i][j] * lam_lgrad[m]));
        if on_axis {
            gamma_regular[0][0] += 0.5 * g_par2[0][0][0][0] / g[0][0];
            // gamma_regular[0][1] += <irregular term>
            gamma_regular[1][1] += -0.5 * g_par2[1][1][0][0] / g[0][0];
        }

        // The only irregular term is a 0.5 * g_rr,z / (r * g_rr) term in gamma_regular[0][1] which percisely cancels
        // an irregular term in lam_rz

        let lam_rr = {
            // Plus a -1/r^2 term that gets cancelled by lam_r * lam_r
            let term1 = 2.0 * s_par[0] + pos[0] * s_par2[0][0] + 0.5 * g_par2_term[0][0];
            let term2 = lam_r * lam_r;
            let term3 = if on_axis {
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
                lam_z_lhopital
            } else {
                lam_z / pos[0]
            };
            term1 + term2 + term3 - gamma_regular[0][1]
        };

        let lam_zz =
            pos[0] * s_par2[1][1] + 0.5 * g_par2_term[1][1] + lam_z * lam_z - gamma_regular[1][1];

        lam_lhess[0][0] = lam_rr;
        lam_lhess[0][1] = lam_rz;
        lam_lhess[1][0] = lam_rz;
        lam_lhess[1][1] = lam_zz;
    }

    let k_grad = grad2(k, k_par, gamma);

    let k_trace = sum2(|i, j| k[i][j] * g_inv[i][j]);
    let k_trace_grad =
        tensor1(|i| sum2(|m, n| k_par[m][n][i] * g_inv[m][n] + k[m][n] * g_inv_par[m][n][i]));

    let k_mat = tensor2(|i, j| sum1(|m| g_inv[i][m] * k[m][j]));
    let k_con = tensor2(|i, j| sum2(|m, n| g_inv[i][m] * g_inv[j][n] * k[m][n]));

    let l_grad = l_par;

    let theta_grad = theta_par;

    let z_grad = grad1(z, z_par, gamma);

    let lapse_grad = lapse_par;
    let lapse_hess = hess0(lapse_par, lapse_par2, gamma);

    // *********************************
    // Hamiltonian *********************
    // *********************************

    let hamiltonian = {
        let term1 = 0.5 * (ricci_trace + k_trace * k_trace) + k_trace * l;
        let term2 = -0.5 * sum2(|i, j| k[i][j] * k_con[i][j]);
        let term3 = -sum2(|i, j| lam_lhess[i][j] * g_inv[i][j]);

        term1 + term2 + term3
    };

    // **********************************
    // Momentum *************************
    // **********************************

    let momentum = tensor1(|i| {
        let term1 = -k_trace_grad[i] - l_grad[i];
        let term2 = sum2(|m, n| k_grad[i][m][n] * g_inv[m][n]);

        let mut regular = sum1(|m| lam_lgrad[m] * k_mat[m][i]) - lam_lgrad[i] * l;
        if on_axis && i == 0 {
            regular -= y
        } else if on_axis && i == 1 {
            regular += k_par[0][1][0] / g[0][0]
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

    let mut lam_t = -lapse * lam * l + sum1(|i| lam * lam_lgrad[i] * shift[i]);
    if on_axis {
        lam_t += lam * shift_par[0][0];
    }

    // ***********************************
    // Extrinsic curvature ***************
    // ***********************************

    let k_lie_shift = lie2(k, k_par, shift, shift_par);
    let k_t = tensor2(|i, j| {
        let term1 = lapse * ricci[i][j] - lapse * lam_lhess[i][j] - lapse_hess[i][j];
        let term2 = lapse * (k_trace + l) * k[i][j];
        let term3 = -2.0 * lapse * sum1(|m| k[i][m] * k_mat[m][j]);
        let term4 = lapse * (z_grad[i][j] + z_grad[j][i] - 2.0 * k[i][j] * theta);

        term1 + term2 + term3 + term4 + k_lie_shift[i][j]
    });

    let l_t = {
        let term1 = lapse * l * (k_trace + l - 2.0 * theta);
        let term2 = -lapse * sum2(|i, j| lam_lhess[i][j] * g_inv[i][j]);
        let term3 = sum1(|i| shift[i] * l_par[i]);

        let mut regular =
            sum2(|i, j| lam_lgrad[i] * g_inv[i][j] * (2.0 * lapse * z[j] - lapse_grad[j]));
        if on_axis {
            regular += g_inv[0][0] * (2.0 * lapse * z_par[0][0] - lapse_par2[0][0]);
        }

        term1 + term2 + term3 + regular
    };

    // ************************************
    // Constraint *************************
    // ************************************

    let theta_t = {
        let term1 = lapse * hamiltonian - lapse * (k_trace + l) * theta;
        let term2 =
            lapse * sum2(|i, j| z_grad[i][j] * g_inv[i][j] - lapse_grad[i] * g_inv[i][j] * z[j]);
        let term3 = sum1(|i| theta_par[i] * shift[i]);

        let mut regular = lapse * sum2(|i, j| lam_lgrad[i] * g_inv[i][j] * z[j]);
        if on_axis {
            regular += lapse * z_par[0][0] / g[0][0];
        }

        term1 + term2 + term3 + regular
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
        let lamg_term = sum1(|m| g_inv[i][m] * (lam_lgrad[m] + 0.5 * g_det_par[m] / g_det));
        let g_inv_term = sum1(|m| g_inv_par[i][m][m]);

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
        s_t = (lam_t / lam - 0.5 * g_t[0][0] / g[0][0]) / pos[0];
    }

    let mut lregular =
        sum2(|i, j| lam_lgrad[i] * g_inv[i][j] * (2.0 * lapse * z[j] - lapse_grad[j]));
    if on_axis {
        lregular += g_inv[0][0] * (2.0 * lapse * z_par[0][0] - lapse_par2[0][0]);
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

        debug1: -k_t[0][0] / g[0][0] + k[0][0] / g[0][0].powi(2) * g_t[0][0],
        debug2: l_t,
    }
}

#[cfg(test)]
mod tests {
    use super::{hyperbolic, tensor3, HyperbolicSystem};
    use aeon::{common::NodeSpace, geometry::Rectangle};

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
            bounds: Rectangle {
                origin: [0.0, 0.0],
                size: [10.0, 10.0],
            },
            size: [100, 100],
            ghost: 2,
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

            let derivs = hyperbolic(system, position);

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
}
