use aeon::{
    basis::{Kernels, RadiativeParams},
    elliptic::HyperRelaxSolver,
    fd::{Gaussian, Mesh},
    prelude::*,
    system::System,
};

use reborrow::ReborrowMut;
use sharedaxi::{
    Constraint, Field, FieldConditions, Fields, Gauge, Metric, Quadrant, ScalarField, Solver,
    Source,
};

// *************************
// Garfinkle variables *****
// *************************

#[derive(Clone)]
pub struct ContextSystem {
    pub scalar_fields: Vec<f64>,
}

impl ContextSystem {
    pub fn num_scalar_fields(&self) -> usize {
        self.scalar_fields.len()
    }

    pub fn scalar_fields(&self) -> impl Iterator<Item = f64> + '_ {
        self.scalar_fields.iter().cloned()
    }
}

impl System for ContextSystem {
    const NAME: &'static str = "Context";

    type Label = Context;

    fn count(&self) -> usize {
        1 + self.num_scalar_fields()
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        std::iter::once(Context::Seed).chain((0..self.num_scalar_fields()).map(Context::Phi))
    }

    fn label_index(&self, label: Self::Label) -> usize {
        match label {
            Context::Seed => 0,
            Context::Phi(id) => id + 1,
        }
    }

    fn label_name(&self, label: Self::Label) -> String {
        match label {
            Context::Seed => "Seed".to_string(),
            Context::Phi(id) => format!("Phi{id}"),
        }
    }

    fn label_from_index(&self, mut index: usize) -> Self::Label {
        if index < 1 {
            return Context::Seed;
        }
        index -= 1;

        Context::Phi(index)
    }
}

#[derive(Clone, Copy)]
pub enum Context {
    Seed,
    Phi(usize),
}

// ***********************
// Boundary conditions ***
// ***********************

#[derive(Clone)]
pub struct PsiCondition;

impl Condition<2> for PsiCondition {
    fn parity(&self, face: Face<2>) -> bool {
        [true, true][face.axis]
    }

    fn radiative(&self, _position: [f64; 2], _spacing: f64) -> RadiativeParams {
        RadiativeParams::lightlike(1.0)
    }
}

/// Boundary Conditions for Garfinkle variables.
#[derive(Clone)]
pub struct ContextConditions;

impl Conditions<2> for ContextConditions {
    type System = ContextSystem;

    fn parity(&self, field: Context, face: Face<2>) -> bool {
        match field {
            Context::Seed => [false, true][face.axis],
            Context::Phi(_) => [true, true][face.axis],
        }
    }

    fn radiative(&self, _field: Context, _position: [f64; 2], _spacing: f64) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

#[derive(Clone)]
pub struct SeedProjection<'a>(&'a [Source]);

impl<'a> Projection<2> for SeedProjection<'a> {
    fn project(&self, [rho, z]: [f64; 2]) -> f64 {
        let rho2 = rho * rho;
        let z2 = z * z;

        let mut result = 0.0;

        for source in self.0 {
            match source {
                Source::Brill { amplitude, sigma } => {
                    let srho2 = sigma.0 * sigma.0;
                    let sz2 = sigma.1 * sigma.1;

                    result += rho * amplitude * (-rho2 / srho2 - z2 / sz2).exp()
                }
                _ => {}
            }
        }

        result
    }
}

#[derive(Clone)]
pub struct Hamiltonian<'a> {
    context: SystemSlice<'a, ContextSystem>,
}

impl<'a> Function<2> for Hamiltonian<'a> {
    type Input = Scalar;
    type Output = Scalar;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: SystemSlice<Self::Input>,
        output: SystemSliceMut<Self::Output>,
    ) {
        let context = self.context.slice(engine.node_range());
        let seed = context.field(Context::Seed);

        let psi = input.into_scalar();
        let dest = output.into_scalar();

        // Iterate over vertices in block.
        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);

            let [rho, _] = engine.position(vertex);

            let psi_r = engine.derivative(psi, 0, vertex);
            let psi_rr = engine.second_derivative(psi, 0, vertex);
            let psi_zz = engine.second_derivative(psi, 1, vertex);

            let seed_r = engine.derivative(seed, 0, vertex);
            let seed_rr = engine.second_derivative(seed, 0, vertex);
            let seed_zz = engine.second_derivative(seed, 1, vertex);

            let laplacian = if rho.abs() <= 10e-10 {
                2.0 * psi_rr + psi_zz
            } else {
                psi_rr + psi_r / rho + psi_zz
            };

            let mut source = 0.0;

            for (i, mass) in context.system().scalar_fields().enumerate() {
                let mass2 = mass * mass;
                let phi = context.field(Context::Phi(i));

                let phi2 = phi[index] * phi[index];
                let phi_r = engine.derivative(phi, 0, vertex);
                let phi_z = engine.derivative(phi, 1, vertex);

                let kinetic = 0.5 * (phi_r * phi_r + phi_z * phi_z);
                let potentional =
                    0.5 * (2.0 * rho * seed[index]).exp() * psi[index].powi(4) * mass2 * phi2;

                source += kinetic + potentional;
            }

            dest[index] = laplacian
                + psi[index] / 4.0 * (rho * seed_rr + 2.0 * seed_r + rho * seed_zz)
                + psi[index] / 4.0 * source;
        }
    }
}

#[derive(Clone)]
pub struct FieldsFromGarfinkle<'a> {
    psi: &'a [f64],
    context: SystemSlice<'a, ContextSystem>,
}

impl<'a> Function<2> for FieldsFromGarfinkle<'a> {
    type Input = Empty;
    type Output = Fields;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        _: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        let psi = &self.psi[engine.node_range()];
        let context = self.context.slice(engine.node_range());
        let seed = context.field(Context::Seed);

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [rho, _] = engine.position(vertex);

            let conformal = psi[index].powi(4) * (2.0 * rho * seed[index]).exp();

            output.field_mut(Field::Metric(Metric::Grr))[index] = conformal;
            output.field_mut(Field::Metric(Metric::Gzz))[index] = conformal;
            output.field_mut(Field::Metric(Metric::S))[index] = -seed[index];
        }
    }
}

pub fn solve_order<const ORDER: usize>(
    order: Order<ORDER>,
    mesh: &mut Mesh<2>,
    solver_con: &Solver,
    sources: &[Source],
    mut system: SystemSliceMut<Fields>,
) -> anyhow::Result<()>
where
    Order<ORDER>: Kernels,
{
    let num_scalar_field = system.system().num_scalar_fields();

    // Retrieve number of nodes in current version of mesh.
    let num_nodes = mesh.num_nodes();

    let mut psi = vec![0.0; num_nodes];
    let mut context = SystemVec::with_length(
        num_nodes,
        ContextSystem {
            scalar_fields: system.system().scalar_fields.clone(),
        },
    );

    // Compute seed values.
    mesh.project_scalar(
        order,
        Quadrant,
        SeedProjection(sources),
        context.field_mut(Context::Seed),
    );

    // Compute scalar field values.
    let mut scalar_field_index = 0;
    for source in sources {
        if let Source::ScalarField {
            amplitude, sigma, ..
        } = source
        {
            mesh.project_scalar(
                order,
                Quadrant,
                Gaussian {
                    amplitude: *amplitude,
                    sigma: [sigma.0, sigma.1],
                    center: [0.0; 2],
                },
                context.field_mut(Context::Phi(scalar_field_index)),
            );

            scalar_field_index += 1;
        }
    }

    // Fill boundary conditions for context fields.
    mesh.fill_boundary(order, Quadrant, ContextConditions, context.as_mut_slice());

    // Initial Guess for Psi
    psi.fill(1.0);

    log::trace!("Running Hyperbolic Relaxation. Nodes: {}", mesh.num_nodes());

    let mut solver = HyperRelaxSolver::new();
    solver.dampening = solver_con.dampening;
    solver.max_steps = solver_con.max_steps;
    solver.tolerance = solver_con.tolerance;
    solver.cfl = solver_con.cfl;
    solver.adaptive = true;

    solver.solve_scalar(
        mesh,
        order,
        Quadrant,
        PsiCondition,
        Hamiltonian {
            context: context.as_slice(),
        },
        &mut psi,
    )?;

    mesh.evaluate(
        order,
        Quadrant,
        FieldsFromGarfinkle {
            psi: &psi,
            context: context.as_slice(),
        },
        SystemSlice::empty(),
        system.rb_mut(),
    );

    // Copy scalar fields
    for i in 0..num_scalar_field {
        system
            .field_mut(Field::ScalarField(ScalarField::Phi, i))
            .copy_from_slice(context.field(Context::Phi(i)));
        system
            .field_mut(Field::ScalarField(ScalarField::Pi, i))
            .fill(0.0);
    }

    // Metric
    system.field_mut(Field::Metric(Metric::Grz)).fill(0.0);
    system.field_mut(Field::Metric(Metric::Krr)).fill(0.0);
    system.field_mut(Field::Metric(Metric::Kzz)).fill(0.0);
    system.field_mut(Field::Metric(Metric::Krz)).fill(0.0);
    system.field_mut(Field::Metric(Metric::Y)).fill(0.0);
    // Constraint
    system
        .field_mut(Field::Constraint(Constraint::Theta))
        .fill(0.0);
    system
        .field_mut(Field::Constraint(Constraint::Zr))
        .fill(0.0);
    system
        .field_mut(Field::Constraint(Constraint::Zz))
        .fill(0.0);
    // Gauge
    system.field_mut(Field::Gauge(Gauge::Lapse)).fill(1.0);
    system.field_mut(Field::Gauge(Gauge::Shiftr)).fill(0.0);
    system.field_mut(Field::Gauge(Gauge::Shiftz)).fill(0.0);

    mesh.fill_boundary(order, Quadrant, FieldConditions, system.rb_mut());

    Ok(())
}
