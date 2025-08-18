use crate::run::config::{ScalarFieldProfile, Smooth};
use aeon::{
    kernel::Interpolation,
    mesh::{Gaussian, TanH},
    prelude::*,
};
use core::f64;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;

const KAPPA: f64 = 8.0 * f64::consts::PI;

/// System for storing all fields necessary for axisymmetric evolution.
#[derive(Clone, Serialize, Deserialize)]
pub struct Fields;

impl System for Fields {
    const NAME: &'static str = "Fields";

    type Label = Field;

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        [
            Field::Phi,
            Field::Pi,
            Field::Conformal,
            Field::Lapse,
            Field::Psi,
        ]
        .into_iter()
    }

    fn count(&self) -> usize {
        4
    }

    fn label_from_index(&self, index: usize) -> Self::Label {
        [
            Field::Phi,
            Field::Pi,
            Field::Conformal,
            Field::Lapse,
            Field::Psi,
        ][index]
    }

    fn label_index(&self, label: Self::Label) -> usize {
        match label {
            Field::Phi => 0,
            Field::Pi => 1,
            Field::Conformal => 2,
            Field::Lapse => 3,
            Field::Psi => 4,
        }
    }

    fn label_name(&self, label: Self::Label) -> String {
        match label {
            Field::Phi => "Phi",
            Field::Pi => "Pi",
            Field::Conformal => "Conformal",
            Field::Lapse => "Lapse",
            Field::Psi => "Psi",
        }
        .to_string()
    }
}

/// Label for indexing fields in `Fields`.
#[derive(Clone, Copy)]
pub enum Field {
    Phi,
    Pi,
    Conformal,
    Lapse,
    Psi,
}

/// Boundary conditions for a system of fields.
#[derive(Clone)]
pub struct FieldConditions;

impl SystemBoundaryConds<1> for FieldConditions {
    type System = Fields;

    fn kind(&self, label: <Self::System as System>::Label, face: Face<1>) -> BoundaryKind {
        if face.side {
            BoundaryKind::Radiative
        } else {
            match label {
                Field::Phi => BoundaryKind::AntiSymmetric,
                Field::Pi | Field::Conformal | Field::Lapse | Field::Psi => BoundaryKind::Symmetric,
            }
        }
    }

    fn radiative(
        &self,
        _label: <Self::System as System>::Label,
        _position: [f64; 1],
    ) -> RadiativeParams {
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
    type Input = Fields;
    type Output = Fields;
    type Error = Infallible;

    fn preprocess(
        &mut self,
        mesh: &mut Mesh<1>,
        input: SystemSliceMut<Self::Input>,
    ) -> Result<(), Infallible> {
        solve_constraints(mesh, input);
        Ok(())
    }

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Infallible> {
        let a = input.field(Field::Conformal);
        let alpha = input.field(Field::Lapse);
        let phi = input.field(Field::Phi);
        let pi = input.field(Field::Pi);

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

            output.field_mut(Field::Phi)[index] = phi_t;
            output.field_mut(Field::Pi)[index] = pi_t;
            output.field_mut(Field::Psi)[index] = psi_t;

            output.field_mut(Field::Conformal)[index] = 0.0;
            output.field_mut(Field::Lapse)[index] = 0.0;
        }

        Ok(())
    }
}

/// Solves for the metric and lapse given a set of scalar fields. This uses a simple outwards-inwards
/// ODE solver to compute solutions to the first order elliptic equations used for these two variables.
pub fn solve_constraints(mesh: &mut Mesh<1>, system: SystemSliceMut<Fields>) {
    let shared = system.into_shared();
    // Unpack individual fields
    let phi = unsafe { shared.field_mut(Field::Phi) };
    let pi = unsafe { shared.field_mut(Field::Pi) };

    mesh.fill_boundary(Order::<4>, ScalarConditions(AntiSymCondition), phi.into());
    mesh.fill_boundary(Order::<4>, ScalarConditions(SymCondition), pi.into());

    let conformal = unsafe { shared.field_mut(Field::Conformal) };
    let lapse = unsafe { shared.field_mut(Field::Lapse) };

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
    mesh.fill_boundary(Order::<4>, ScalarConditions(SymCondition), conformal.into());

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
    mesh.fill_boundary(Order::<4>, ScalarConditions(SymCondition), lapse.into());
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
    mut output: SystemSliceMut<'_, Fields>,
) {
    output.field_mut(Field::Conformal).fill(1.0);
    output.field_mut(Field::Lapse).fill(1.0);
    output.field_mut(Field::Pi).fill(0.0);

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
                output.field_mut(Field::Phi),
            );
            mesh.project(
                4,
                Gaussian {
                    amplitude: amplitude.unwrap(),
                    sigma: [sigma.unwrap()],
                    center: [center.unwrap()],
                },
                output.field_mut(Field::Phi),
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
                output.field_mut(Field::Phi),
            );
            mesh.project(
                4,
                TanH {
                    amplitude: amplitude.unwrap(),
                    sigma: sigma.unwrap(),
                    center: [center.unwrap()],
                },
                output.field_mut(Field::Phi),
            );
        }
    }

    mesh.fill_boundary(Order::<4>, FieldConditions, output);
}

pub fn find_mass(mesh: &Mesh<1>, system: SystemSlice<Fields>) -> f64 {
    let mut a_max = 0.0;
    let mut r_max = 0.0;

    for block in 0..mesh.num_blocks() {
        let space = mesh.block_space(BlockId(block));
        let nodes = mesh.block_nodes(BlockId(block));

        let vertex_size = space.vertex_size()[0];
        let a = &system.field(Field::Conformal)[nodes.clone()];

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
    type Input = Fields;
    type Output = Scalar;
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Self::Error> {
        let lapse = input.field(Field::Lapse);
        let phi = input.field(Field::Phi);
        let pi = input.field(Field::Pi);

        let output = output.field_mut(());

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [r] = engine.position(vertex);

            output[index] = 4.0 * f64::consts::PI * r * lapse[index] * phi[index] * pi[index];
        }

        Ok(())
    }
}
