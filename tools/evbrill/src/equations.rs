// A very useful function
use std::array::from_fn;

// *********************************
// Tensor utils ********************
// *********************************

type Rank1 = [f64; 2];
type Rank2 = [[f64; 2]; 2];
type Rank3 = [[[f64; 2]; 2]; 2];
type Rank4 = [[[[f64; 2]; 2]; 2]; 2];

#[inline]
pub fn tensor1(f: impl FnMut(usize) -> f64) -> Rank1 {
    from_fn(f)
}

#[inline]
pub fn tensor2(mut f: impl FnMut(usize, usize) -> f64) -> Rank2 {
    from_fn(|i| from_fn(|j| f(i, j)))
}

#[inline]
pub fn tensor3(mut f: impl FnMut(usize, usize, usize) -> f64) -> Rank3 {
    from_fn(|i| from_fn(|j| from_fn(|k| f(i, j, k))))
}

#[inline]
pub fn tensor4(mut f: impl FnMut(usize, usize, usize, usize) -> f64) -> Rank4 {
    from_fn(|i| from_fn(|j| from_fn(|k| from_fn(|l| f(i, j, k, l)))))
}

#[inline]
pub fn sum1(mut f: impl FnMut(usize) -> f64) -> f64 {
    f(0) + f(1)
}

#[inline]
pub fn sum2(mut f: impl FnMut(usize, usize) -> f64) -> f64 {
    f(0, 0) + f(0, 1) + f(1, 0) + f(1, 1)
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

#[allow(non_snake_case)]
pub fn hyperbolic(sys: HyperbolicSystem, pos: [f64; 2]) {
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

    // Logrithmic derivatives of lambda
    let lam_lpar = {
        let lam_r = pos[0].powi(-1) + s + pos[0] * s_par[0] + 0.5 * g_par[0][0][0] / g[0][0];
        let lam_z = pos[0] * s_par[1] + 0.5 * g_par[0][0][1] / g[0][0];
        [lam_r, lam_z]
    };

    let lam_lpar2 = {
        let mut result = tensor2(|i, j| {
            lam_lpar[i] * lam_lpar[j] + pos[0] * s_par2[i][j] + 0.5 * g_par2[0][0][i][j] / g[0][0]
                - 0.5 * g_par[0][0][i] * g_par[0][0][j] / g[0][0].powi(2)
        });

        result[0][0] += -pos[0].powi(-2) + 2.0 * s_par[0];
        result[0][1] += s_par[1];
        result[1][0] += s_par[1];

        result
    };

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
    let mut l_par = tensor1(|i| {
        pos[0] * y_par[i] + k_par[0][0][i] / g[0][0] - k[0][0] / g[0][0].powi(2) * g_par[0][0][i]
    });
    l_par[0] += y;

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
    let gamma = tensor3(|i, j, k| {
        sum1(|l| 0.5 * g_inv[i][l] * (g_par[l][j][k] + g_par[k][l][j] - g_par[j][k][l]))
    });
    let gamma_par = tensor4(|i, j, k, l| {
        let term1 =
            sum1(|m| 0.5 * g_inv_par[i][m][l] * (g_par[m][j][k] + g_par[k][m][j] - g_par[j][k][m]));
        let term2 = sum1(|m| {
            0.5 * g_inv[i][m] * (g_par2[m][j][k][l] + g_par2[k][m][j][l] - g_par2[j][k][m][l])
        });
        term1 + term2
    });

    // And now the Ricci tensor
    let ricci = tensor2(|j, k| {
        let term1 = sum1(|i| gamma_par[i][j][k][i] - gamma_par[i][k][i][j]);
        let term2 = sum2(|i, p| gamma[i][i][p] * gamma[p][j][k] - gamma[i][j][p] * gamma[p][i][k]);
        term1 + term2
    });
    // As well as the Ricci scalar
    let ricci_trace = sum2(|i, j| ricci[i][j] * g_inv[i][j]);

    // *********************************
    // Contractions and Derivatives ****
    // *********************************

    let lam_lgrad = lam_lpar;
    let lam_lhess = hess0(lam_lpar, lam_lpar2, gamma);

    let k_grad = grad2(k, k_par, gamma);

    let k_trace = sum2(|i, j| k[i][j] * g_inv[i][j]);
    let k_trace_grad =
        tensor1(|i| sum2(|m, n| k_par[m][n][i] * g_inv[m][n] + k[m][n] * g_inv_par[m][n][i]));

    let k_mat = tensor2(|i, j| sum1(|m| g_inv[i][m] * k[m][j]));
    let k_con = tensor2(|i, j| sum2(|m, n| g_inv[i][m] * g_inv[j][n] * k[m][n]));

    let l_grad = l_par;

    let theta_grad = theta_par;

    let z_con = tensor1(|i| sum1(|j| z[j] * g_inv[i][j]));
    let z_grad = grad1(z, z_par, gamma);

    let lapse_grad = lapse_par;
    let lapse_hess = hess0(lapse_par, lapse_par2, gamma);

    // *********************************
    // Hamiltonian *********************
    // *********************************

    let hamiltonian = {
        let term1 = 0.5 * (ricci_trace + k_trace * k_trace) + k_trace * l;
        let term2 = -sum2(|i, j| k[i][j] * k_con[i][j]);
        let term3 = -sum2(|i, j| lam_lhess[i][j] * g_inv[i][j]);

        term1 + term2 + term3
    };

    // **********************************
    // Momentum *************************
    // **********************************

    let momentum = tensor1(|i| {
        let term1 = -k_trace_grad[i] - l_grad[i] - lam_lgrad[i] * l;
        let term2 = sum1(|m| lam_lgrad[m] * k_mat[m][i]);
        let term3 = sum2(|m, n| k_grad[i][m][n] * g_inv[m][n]);
        term1 + term2 + term3
    });

    // ***********************************
    // Metric ****************************
    // ***********************************

    let g_t = {
        let term1 = tensor2(|i, j| -2.0 * lapse * k[i][j]);
        let term2 = lie2(g, g_par, shift, shift_par);
        tensor2(|i, j| term1[i][j] + term2[i][j])
    };
    let lam_t = -lapse * lam * l + sum1(|i| lam * lam_lpar[i] * shift[i]);

    // ***********************************
    // Extrinsic curvature ***************
    // ***********************************

    let k_lie_shift = lie2(k, k_par, shift, shift_par);
    let k_t = tensor2(|i, j| {
        let term1 = lapse * ricci[i][j] - lapse * lam_lhess[i][j] - lapse_hess[i][j];
        let term2 = lapse * (k_trace + l) * k[i][j];
        let term3 = lapse * sum1(|m| -2.0 * k[i][m] * k_mat[m][j]);
        let term4 = lapse * (z_grad[i][j] + z_grad[j][i] - 2.0 * k[i][j] * theta);
        term1 + term2 + term3 + term4 + k_lie_shift[i][j]
    });

    let l_t = {
        let term1 = lapse * sum2(|i, j| lam_lhess[i][j] * g_inv[i][j]);
        let term2 = sum2(|i, j| lam_lgrad[i] * lapse_grad[j]);
        let term3 = lapse * l * (k_trace + l);
        let term4 = lapse * (sum1(|i| 2.0 * lam_lgrad[i] * z_con[i]) - 2.0 * l * theta);
        let term5 = sum1(|i| l_par[i] * shift[i]);
        term1 + term2 + term3 + term4 + term5
    };

    // ************************************
    // Constraint *************************
    // ************************************

    let theta_t = {
        let term1 = lapse * hamiltonian;
        let term2 = sum1(|i| (lapse * lam_lgrad[i] - lapse_grad[i]) * z_con[i]);
        let term3 = sum2(|i, j| lapse * z_grad[i][j] * g_inv[i][j]);
        let term4 = -lapse * (k_trace + l) * theta;
        let term5 = sum1(|i| theta_par[i] * shift[i]);
        term1 + term2 + term3 + term4 + term5
    };

    let z_lie_shift = lie1(z, z_par, shift, shift_par);
    let z_t = tensor1(|i| {
        let term1 = lapse * momentum[i];
        let term2 = -2.0 * lapse * sum1(|m| k[i][m] * z_con[m]);
        let term3 = -lapse_grad[i] * theta + lapse * theta_grad[i];
        term1 + term2 + term3 + z_lie_shift[i]
    });
}
