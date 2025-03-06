use std::path::{Path, PathBuf};

use aeon::{
    kernel::Kernels,
    mesh::{Gaussian, Mesh},
    prelude::*,
    solver::{HyperRelaxSolver, SolverCallback},
    system::System,
};

use reborrow::ReborrowMut;
use sharedaxi::{
    eqs, Constraint, Field, FieldConditions, Fields, Gauge, Metric, ScalarField, Solver, Source,
};

// *************************
// Garfinkle variables *****
// *************************

#[derive(Clone)]
pub struct ContextSystem {
    /// scalar field masses.
    pub scalar_fields: Vec<f64>,
}

impl ContextSystem {
    /// Returns the number of scalar fields.
    pub fn num_scalar_fields(&self) -> usize {
        self.scalar_fields.len()
    }

    /// Returns an interator over the scalar fields in this system.
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

/// Context system label.
#[derive(Clone, Copy)]
pub enum Context {
    Seed,
    Phi(usize),
}

// ***********************
// Boundary conditions ***
// ***********************

/// Boundary conditions for psi.
#[derive(Clone)]
pub struct PsiCondition;

impl BoundaryCondition<2> for PsiCondition {
    fn parity(&self, face: Face<2>) -> bool {
        [true, true][face.axis]
    }

    fn radiative(&self, _position: [f64; 2]) -> RadiativeParams {
        RadiativeParams::lightlike(1.0)
    }
}

/// Boundary Conditions for context variables.
#[derive(Clone)]
pub struct ContextConditions;

impl SystemBoundaryConds<2> for ContextConditions {
    type System = ContextSystem;

    fn parity(&self, field: Context, face: Face<2>) -> bool {
        match field {
            Context::Seed => [false, true][face.axis],
            Context::Phi(_) => [true, true][face.axis],
        }
    }

    fn radiative(&self, _field: Context, _position: [f64; 2]) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

/// Seed function projections.
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

/// Hamiltonian elliptic equation.
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
                let potential =
                    0.5 * (2.0 * rho * seed[index]).exp() * psi[index].powi(4) * mass2 * phi2;

                source += kinetic + potential;
            }

            dest[index] = laplacian
                + psi[index] / 4.0 * (rho * seed_rr + 2.0 * seed_r + rho * seed_zz)
                + psi[index] / 4.0 * source * eqs::KAPPA;
        }
    }
}

struct Callback<'a> {
    visualize: Option<VisualizeConfig<'a>>,
}

// Implement visualization for hamiltonian.
impl<'a> SolverCallback<2, Scalar> for Callback<'a> {
    fn callback(
        &self,
        mesh: &Mesh<2>,
        input: SystemSlice<Scalar>,
        output: SystemSlice<Scalar>,
        iteration: usize,
    ) {
        let Some(ref visualze) = self.visualize else {
            return;
        };

        if iteration % visualze.every != 0 {
            return;
        }

        let i = iteration / visualze.every;

        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_field("Solution", input.into_scalar());
        checkpoint.save_field("Derivative", output.into_scalar());

        mesh.export_vtu(
            PathBuf::from(visualze.path).join(format!(
                "{}_level_{}_iter_{}.vtu",
                visualze.name,
                mesh.max_level(),
                i
            )),
            &checkpoint,
            ExportVtuConfig {
                title: visualze.name.to_string(),
                ghost: false,
                stride: visualze.stride,
            },
        )
        .unwrap()
    }
}

/// Generate fields from Garfinkle variables.
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

#[derive(Clone)]
pub struct VisualizeConfig<'a> {
    /// Path to output relaxation data.
    pub path: &'a Path,
    /// Name of simulation.
    pub name: &'a str,
    /// Output vtu every `every` iterations.
    pub every: usize,
    /// Stride for outputting nodes on mesh.
    pub stride: usize,
}

pub fn solve_order<const ORDER: usize>(
    order: Order<ORDER>,
    mesh: &mut Mesh<2>,
    solver_con: &Solver,
    visualize: Option<VisualizeConfig>,
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
    mesh.project(
        ORDER,
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
            mesh.project(
                ORDER,
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
    mesh.fill_boundary(order, ContextConditions, context.as_mut_slice());

    // Initial Guess for Psi
    psi.fill(1.0);

    log::trace!(
        "Relaxing. Max Level {}, Nodes: {}",
        mesh.max_level(),
        mesh.num_nodes()
    );

    let mut solver = HyperRelaxSolver::new();
    solver.dampening = solver_con.dampening;
    solver.max_steps = solver_con.max_steps;
    solver.tolerance = solver_con.tolerance;
    solver.cfl = solver_con.cfl;
    solver.adaptive = true;

    solver.solve_with_callback(
        mesh,
        order,
        ScalarConditions(PsiCondition),
        Hamiltonian {
            context: context.as_slice(),
        },
        Callback { visualize },
        SystemSliceMut::from_scalar(&mut psi),
    )?;

    mesh.evaluate(
        ORDER,
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

    mesh.fill_boundary(order, FieldConditions, system.rb_mut());

    Ok(())
}
