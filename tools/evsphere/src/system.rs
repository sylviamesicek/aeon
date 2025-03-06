use core::f64;

use aeon::{kernel::Interpolation, mesh::Gaussian, prelude::*};
use serde::{Deserialize, Serialize};

use crate::config::SIGMA;

const KAPPA: f64 = 8.0 * f64::consts::PI;
// const KAPPA: f64 = 1.0;

/// System for storing all fields necessary for axisymmetric evolution.
#[derive(Clone, Serialize, Deserialize)]
pub struct Fields;

impl System for Fields {
    const NAME: &'static str = "Fields";

    type Label = Field;

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        [Field::Phi, Field::Pi, Field::Conformal, Field::Lapse].into_iter()
    }

    fn count(&self) -> usize {
        4
    }

    fn label_from_index(&self, index: usize) -> Self::Label {
        [Field::Phi, Field::Pi, Field::Conformal, Field::Lapse][index]
    }

    fn label_index(&self, label: Self::Label) -> usize {
        match label {
            Field::Phi => 0,
            Field::Pi => 1,
            Field::Conformal => 2,
            Field::Lapse => 3,
        }
    }

    fn label_name(&self, label: Self::Label) -> String {
        match label {
            Field::Phi => "Phi",
            Field::Pi => "Pi",
            Field::Conformal => "Conformal",
            Field::Lapse => "Lapse",
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
}

#[derive(Clone)]
pub struct FieldConditions;

impl SystemBoundaryConds<1> for FieldConditions {
    type System = Fields;

    fn parity(&self, label: <Self::System as System>::Label, _face: Face<1>) -> bool {
        match label {
            Field::Phi => false,
            Field::Pi | Field::Conformal | Field::Lapse => true,
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

#[derive(Clone)]
pub struct SymCondition;

impl BoundaryCondition<1> for SymCondition {
    fn parity(&self, _face: Face<1>) -> bool {
        true
    }

    fn radiative(&self, _position: [f64; 1]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}

#[derive(Clone)]
pub struct AntiSymCondition;

impl BoundaryCondition<1> for AntiSymCondition {
    fn parity(&self, _face: Face<1>) -> bool {
        false
    }

    fn radiative(&self, _position: [f64; 1]) -> RadiativeParams {
        RadiativeParams {
            target: 0.0,
            speed: 1.0,
        }
    }
}

#[derive(Clone)]
pub struct InitialData;

impl Function<1> for InitialData {
    type Input = Scalar;
    type Output = Fields;

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        let scalar_field = input.into_scalar();

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);

            output.field_mut(Field::Conformal)[index] = 1.0;
            output.field_mut(Field::Lapse)[index] = 1.0;
            output.field_mut(Field::Phi)[index] = engine.derivative(scalar_field, 0, vertex);
            output.field_mut(Field::Pi)[index] = 0.0;
        }
    }
}

#[derive(Clone)]
pub struct TimeDerivs;

impl Function<1> for TimeDerivs {
    fn preprocess(&self, mesh: &mut Mesh<1>, input: SystemSliceMut<Self::Input>) {
        solve_constraints(mesh, input);
    }

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
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

            output.field_mut(Field::Phi)[index] = phi_t;
            output.field_mut(Field::Pi)[index] = pi_t;

            output.field_mut(Field::Conformal)[index] = 0.0;
            output.field_mut(Field::Lapse)[index] = 0.0;
        }
    }

    type Input = Fields;
    type Output = Fields;
}

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
        let space = mesh.block_space(block);
        let nodes = mesh.block_nodes(block);
        // let bounds = mesh.block_bounds(block);
        let spacing = mesh.block_spacing(block);
        let cell_size = space.cell_size()[0];

        let phi = &phi[nodes.clone()];
        let pi = &pi[nodes.clone()];
        let conformal = &mut conformal[nodes.clone()];

        debug_assert!(phi.len() == space.num_nodes());

        let derivative = |r: f64, a: f64, phi: f64, pi: f64| {
            if r < 10e-15 || r.is_nan() || r.is_infinite() {
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
        let space = mesh.block_space(block);
        let nodes = mesh.block_nodes(block);
        let spacing = mesh.block_spacing(block);
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

pub fn generate_initial_scalar_field(mesh: &mut Mesh<1>, amplitude: f64) -> Vec<f64> {
    let mut scalar_field = vec![0.0; mesh.num_nodes()];

    mesh.project(
        4,
        Gaussian {
            amplitude,
            sigma: [SIGMA],
            center: [0.0],
        },
        &mut scalar_field,
    );
    mesh.fill_boundary(
        Order::<4>,
        ScalarConditions(SymCondition),
        (&mut scalar_field).into(),
    );

    scalar_field
}

pub fn find_mass(mesh: &Mesh<1>, system: SystemSlice<Fields>) -> f64 {
    let mut a_max = 0.0;
    let mut r_max = 0.0;

    for block in 0..mesh.num_blocks() {
        let space = mesh.block_space(block);
        let nodes = mesh.block_nodes(block);

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
