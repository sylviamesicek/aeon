use crate::run::config::{ScalarFieldProfile, Smooth};
use aeon::{
    kernel::{Interpolation, ScalarConditions},
    mesh::{Gaussian, TanH},
    prelude::*,
};
use core::f64;
use std::convert::Infallible;

const KAPPA: f64 = 8.0 * f64::consts::PI;

pub const PHI_CH: usize = 0;
pub const PI_CH: usize = 1;
pub const CONFORMAL_CH: usize = 2;
pub const LAPSE_CH: usize = 3;
pub const PSI_CH: usize = 4;
pub const NUM_CHANNELS: usize = 5;

pub fn save_image(checkpoint: &mut Checkpoint<1>, image: ImageRef) {
    checkpoint.save_field("Phi", image.channel(PHI_CH));
    checkpoint.save_field("Pi", image.channel(PI_CH));
    checkpoint.save_field("Conformal", image.channel(CONFORMAL_CH));
    checkpoint.save_field("Lapse", image.channel(LAPSE_CH));
    checkpoint.save_field("Psi", image.channel(PSI_CH));
}

/// Boundary conditions for a system of fields.
#[derive(Clone)]
pub struct FieldConditions;

impl SystemBoundaryConds<1> for FieldConditions {
    fn kind(&self, channel: usize, face: Face<1>) -> BoundaryKind {
        if face.side {
            BoundaryKind::Radiative
        } else {
            match channel {
                PHI_CH => BoundaryKind::AntiSymmetric,
                PI_CH | CONFORMAL_CH | LAPSE_CH | PSI_CH => BoundaryKind::Symmetric,
                _ => unimplemented!("Unimplemented channel"),
            }
        }
    }

    fn radiative(&self, _channel: usize, _position: [f64; 1]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}

/// Symmetric condition across inner boundary.
#[derive(Clone)]
pub struct SymCondition;

impl BoundaryConds<1> for SymCondition {
    fn kind(&self, face: Face<1>) -> BoundaryKind {
        if face.side {
            BoundaryKind::Radiative
        } else {
            BoundaryKind::Symmetric
        }
    }

    fn radiative(&self, _position: [f64; 1]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}

/// AntiSymmetric condition across inner boundary.
#[derive(Clone)]
pub struct AntiSymCondition;

impl BoundaryConds<1> for AntiSymCondition {
    fn kind(&self, face: Face<1>) -> BoundaryKind {
        if face.side {
            BoundaryKind::Radiative
        } else {
            BoundaryKind::AntiSymmetric
        }
    }

    fn radiative(&self, _position: [f64; 1]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}

/// Temporal derivative of the scalar system.
#[derive(Clone)]
pub struct TimeDerivs;

impl Function<1> for TimeDerivs {
    type Error = Infallible;

    fn preprocess(&mut self, mesh: &mut Mesh<1>, input: ImageMut) -> Result<(), Infallible> {
        solve_constraints(mesh, input);
        Ok(())
    }

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), Infallible> {
        let a = input.channel(CONFORMAL_CH);
        let alpha = input.channel(LAPSE_CH);
        let phi = input.channel(PHI_CH);
        let pi = input.channel(PI_CH);

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [r] = engine.position(vertex);
            let a_r = engine.derivative(a, 0, vertex);
            let alpha_r = engine.derivative(alpha, 0, vertex);
            let phi_r = engine.derivative(phi, 0, vertex);
            let pi_r = engine.derivative(pi, 0, vertex);
            let alpha = alpha[index];
            let a = a[index];
            let phi = phi[index];
            let pi = pi[index];

            let phi_t = pi_r * alpha / a + alpha_r / a * pi - alpha / (a * a) * pi * a_r;
            let mut pi_t = phi_r * alpha / a + alpha_r / a * phi - alpha / (a * a) * phi * a_r;
            if r < 1e-15 {
                pi_t += alpha / a * phi_r * 2.0;
            } else {
                pi_t += alpha / a * phi * 2.0 / r;
            }

            let psi_t = alpha / a * pi;

            output.channel_mut(PHI_CH)[index] = phi_t;
            output.channel_mut(PI_CH)[index] = pi_t;
            output.channel_mut(PSI_CH)[index] = psi_t;

            output.channel_mut(CONFORMAL_CH)[index] = 0.0;
            output.channel_mut(LAPSE_CH)[index] = 0.0;
        }

        Ok(())
    }
}

/// Solves for the metric and lapse given a set of scalar fields. This uses a simple outwards-inwards
/// ODE solver to compute solutions to the first order elliptic equations used for these two variables.
pub fn solve_constraints(mesh: &mut Mesh<1>, system: ImageMut) {
    let shared: ImageShared = system.into();
    // Unpack individual fields
    let phi = unsafe { shared.channel_mut(PHI_CH) };
    let pi = unsafe { shared.channel_mut(PI_CH) };

    mesh.fill_boundary(4, ScalarConditions(AntiSymCondition), phi.into());
    mesh.fill_boundary(4, ScalarConditions(SymCondition), pi.into());

    let conformal = unsafe { shared.channel_mut(CONFORMAL_CH) };
    let lapse = unsafe { shared.channel_mut(LAPSE_CH) };

    // Perform radial quadrature for conformal factor
    let mut conformal_prev = 1.0;

    for block in 0..mesh.num_blocks() {
        let space = mesh.block_space(BlockId(block));
        let nodes = mesh.block_nodes(BlockId(block));
        // let bounds = mesh.block_bounds(block);
        let spacing = mesh.block_spacing(BlockId(block));
        let cell_size = space.cell_size()[0];

        let phi = &phi[nodes.clone()];
        let pi = &pi[nodes.clone()];
        let conformal = &mut conformal[nodes.clone()];

        debug_assert!(phi.len() == space.num_nodes());

        let derivative = |r: f64, a: f64, phi: f64, pi: f64| {
            if r < 1e-15 || r.is_nan() || r.is_infinite() {
                return 0.0;
            }

            KAPPA / 4.0 * r * a * (phi * phi + pi * pi) - a * (a * a - 1.0) / (2.0 * r)
        };

        conformal[space.index_from_vertex([0])] = conformal_prev;

        for vertex in 0..cell_size {
            let index = space.index_from_vertex([vertex]);
            let [r] = space.position([vertex as isize]);
            // Intermediate step interpolation
            let r_half = r + spacing / 2.0;
            let phi_half = space.prolong(Interpolation::<4>, [(2 * vertex + 1) as isize], phi);
            let pi_half = space.prolong(Interpolation::<4>, [(2 * vertex + 1) as isize], pi);
            let r_next = r + spacing;
            let phi_next = phi[index + 1];
            let pi_next = pi[index + 1];
            let phi = phi[index];
            let pi = pi[index];
            let a = conformal[index];

            let k1 = derivative(r, a, phi, pi);
            let k2 = derivative(r_half, a + k1 * spacing / 2.0, phi_half, pi_half);
            let k3 = derivative(r_half, a + k2 * spacing / 2.0, phi_half, pi_half);
            let k4 = derivative(r_next, a + k3 * spacing, phi_next, pi_next);

            conformal[index + 1] =
                conformal[index] + spacing / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

        conformal_prev = conformal[space.index_from_vertex([cell_size])];
    }
    // Fill ghost nodes.
    mesh.fill_boundary(4, ScalarConditions(SymCondition), conformal.into());

    // Perform radial quadrature for lapse
    let mut lapse_prev = 1.0 / conformal_prev;

    for block in (0..mesh.num_blocks()).rev() {
        let space = mesh.block_space(BlockId(block));
        let nodes = mesh.block_nodes(BlockId(block));
        let spacing = mesh.block_spacing(BlockId(block));
        let cell_size = space.cell_size()[0];

        let phi = &phi[nodes.clone()];
        let pi = &pi[nodes.clone()];
        let conformal = &conformal[nodes.clone()];
        let lapse = &mut lapse[nodes.clone()];

        let derivative = |r: f64, alpha: f64, a: f64, phi: f64, pi: f64| {
            if r < 1e-15 || r.is_nan() || r.is_infinite() {
                return 0.0;
            }

            KAPPA / 4.0 * r * alpha * (phi * phi + pi * pi) + alpha * (a * a - 1.0) / (2.0 * r)
        };

        lapse[space.index_from_vertex([cell_size])] = lapse_prev;

        for vertex in (0..cell_size).rev().map(|i| i + 1) {
            let index = space.index_from_vertex([vertex]);
            let [r] = space.position([vertex as isize]);
            // Intermediate step interpolation
            let r_half = r - spacing / 2.0;
            let phi_half = space.prolong(Interpolation::<4>, [(2 * vertex - 1) as isize], phi);
            let pi_half = space.prolong(Interpolation::<4>, [(2 * vertex - 1) as isize], pi);
            let a_half = space.prolong(Interpolation::<4>, [(2 * vertex - 1) as isize], conformal);
            let r_next = r - spacing;
            let phi_next = phi[index - 1];
            let pi_next = pi[index - 1];
            let a_next = conformal[index - 1];
            let phi = phi[index];
            let pi = pi[index];
            let a = conformal[index];
            let alpha = lapse[index];

            let k1 = derivative(r, alpha, a, phi, pi);
            let k2 = derivative(
                r_half,
                alpha - k1 * spacing / 2.0,
                a_half,
                phi_half,
                pi_half,
            );
            let k3 = derivative(
                r_half,
                alpha - k2 * spacing / 2.0,
                a_half,
                phi_half,
                pi_half,
            );
            let k4 = derivative(r_next, alpha - k3 * spacing, a_next, phi_next, pi_next);

            lapse[index - 1] = lapse[index] - spacing / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }

        lapse_prev = lapse[space.index_from_vertex([0])];
    }

    // Fill lapse ghost nodes
    mesh.fill_boundary(4, ScalarConditions(SymCondition), lapse.into());
}

fn smooth(s: f64, n: f64, r: f64) -> f64 {
    if s.abs() <= 1e-15 {
        return 1.0;
    }

    let inner = s / r.powf(n);
    if inner.is_infinite() || inner.is_nan() {
        return 0.0;
    }

    1.0 - inner.tanh()
}

/// Gaussian initial data.
struct GaussGrad {
    amplitude: f64,
    sigma: f64,
    center: f64,

    sstrength: f64,
    spower: f64,
}

impl Projection<1> for GaussGrad {
    fn project(&self, [r]: [f64; 1]) -> f64 {
        let offset = (r - self.center) / self.sigma;

        let f = -2.0 * self.amplitude * offset / self.sigma * (-offset * offset).exp();

        f * smooth(self.sstrength, self.spower, r)
    }
}

/// Tanh initial data.
struct TanHGrad {
    amplitude: f64,
    sigma: f64,
    center: f64,

    sstrength: f64,
    spower: f64,
}

impl Projection<1> for TanHGrad {
    fn project(&self, [r]: [f64; 1]) -> f64 {
        let offset = (r - self.center) / self.sigma;
        let f = self.amplitude * offset.cosh().powi(-2) / self.sigma;

        f * smooth(self.sstrength, self.spower, r)
    }
}

/// Comverts a profile into a scalar field.
pub fn intial_data(
    mesh: &mut Mesh<1>,
    profile: &ScalarFieldProfile,
    smooth: &Smooth,
    mut output: ImageMut<'_>,
) {
    output.channel_mut(CONFORMAL_CH).fill(1.0);
    output.channel_mut(LAPSE_CH).fill(1.0);
    output.channel_mut(PI_CH).fill(0.0);

    match profile {
        ScalarFieldProfile::Gaussian {
            amplitude,
            sigma,
            center,
        } => {
            mesh.project(
                4,
                GaussGrad {
                    amplitude: amplitude.unwrap(),
                    sigma: sigma.unwrap(),
                    center: center.unwrap(),
                    spower: smooth.power,
                    sstrength: smooth.strength,
                },
                output.channel_mut(PHI_CH),
            );
            mesh.project(
                4,
                Gaussian {
                    amplitude: amplitude.unwrap(),
                    sigma: [sigma.unwrap()],
                    center: [center.unwrap()],
                },
                output.channel_mut(PSI_CH),
            );
        }
        ScalarFieldProfile::TanH {
            amplitude,
            sigma,
            center,
        } => {
            mesh.project(
                4,
                TanHGrad {
                    amplitude: amplitude.unwrap(),
                    sigma: sigma.unwrap(),
                    center: center.unwrap(),
                    spower: smooth.power,
                    sstrength: smooth.strength,
                },
                output.channel_mut(PHI_CH),
            );
            mesh.project(
                4,
                TanH {
                    amplitude: amplitude.unwrap(),
                    sigma: sigma.unwrap(),
                    center: [center.unwrap()],
                },
                output.channel_mut(PI_CH),
            );
        }
    }

    mesh.fill_boundary(4, FieldConditions, output);
}

pub fn find_mass(mesh: &Mesh<1>, system: ImageRef) -> f64 {
    let mut a_max = 0.0;
    let mut r_max = 0.0;

    for block in 0..mesh.num_blocks() {
        let space = mesh.block_space(BlockId(block));
        let nodes = mesh.block_nodes(BlockId(block));

        let vertex_size = space.vertex_size()[0];
        let a = &system.channel(CONFORMAL_CH)[nodes.clone()];

        for vertex in 0..vertex_size {
            let index = space.index_from_vertex([vertex]);
            let [r] = space.position([vertex as isize]);
            let a = a[index];

            if a > a_max {
                a_max = a;
                r_max = r;
            }
        }
    }

    r_max / 2.0
}

pub struct ConstraintRhs;

impl Function<1> for ConstraintRhs {
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), Self::Error> {
        let lapse = input.channel(LAPSE_CH);
        let phi = input.channel(PHI_CH);
        let pi = input.channel(PI_CH);

        let output = output.channel_mut(0);

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [r] = engine.position(vertex);

            output[index] = 4.0 * f64::consts::PI * r * lapse[index] * phi[index] * pi[index];
        }

        Ok(())
    }
}
