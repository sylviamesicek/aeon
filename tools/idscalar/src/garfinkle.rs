use super::{Quadrant, Rinne, ORDER};
use aeon::{basis::RadiativeParams, elliptic::HyperRelaxSolver, fd::Mesh, prelude::*};

#[derive(Clone, SystemLabel)]
pub struct Psi;

#[derive(Clone)]
pub struct PsiConditions;

impl Conditions<2> for PsiConditions {
    type System = Psi;

    fn parity(&self, _: Self::System, face: Face<2>) -> bool {
        [true, true][face.axis]
    }

    fn radiative(
        &self,
        _field: Self::System,
        _position: [f64; 2],
        _spacing: f64,
    ) -> RadiativeParams {
        RadiativeParams::lightlike(1.0)
    }
}

#[derive(Clone, SystemLabel)]
pub enum Context {
    Seed,
    Phi,
}

#[derive(Clone)]
pub struct ContextConditions;

impl Conditions<2> for ContextConditions {
    type System = Context;

    fn parity(&self, field: Self::System, face: Face<2>) -> bool {
        match field {
            Context::Phi => [true, true][face.axis],
            Context::Seed => [false, true][face.axis],
        }
    }

    fn radiative(&self, _: Self::System, _position: [f64; 2], _spacing: f64) -> RadiativeParams {
        RadiativeParams::lightlike(0.0)
    }
}

#[derive(Clone)]
pub struct ContextProjection;

impl Projection<2> for ContextProjection {
    type Output = Context;

    fn project(&self, position: [f64; 2]) -> SystemValue<Self::Output> {
        let [rho, z] = position;

        let mut result = SystemValue::default();
        result.set_field(
            Context::Seed,
            rho * crate::SEED * (-(2.0 * rho * rho + z * z)).exp(),
        );
        result.set_field(
            Context::Phi,
            crate::SCALAR * (-(2.0 * rho * rho + z * z)).exp(),
        );

        result
    }
}

#[derive(Clone)]
pub struct PsiOperator;

impl Operator<2> for PsiOperator {
    type System = Psi;
    type Context = Context;

    fn apply(
        &self,
        engine: &impl Engine<2, Pair<Self::System, Self::Context>>,
    ) -> SystemValue<Self::System> {
        let [rho, _z] = engine.position();

        let psi = engine.value(Pair::Left(Psi));
        let psi_r = engine.derivative(Pair::Left(Psi), 0);

        let psi_rr = engine.second_derivative(Pair::Left(Psi), 0, 0);
        let psi_zz = engine.second_derivative(Pair::Left(Psi), 1, 1);

        let seed = engine.value(Pair::Right(Context::Seed));
        let seed_r = engine.derivative(Pair::Right(Context::Seed), 0);
        let seed_rr = engine.second_derivative(Pair::Right(Context::Seed), 0, 0);
        let seed_zz = engine.second_derivative(Pair::Right(Context::Seed), 1, 1);

        let laplacian = if rho.abs() <= 10e-10 {
            2.0 * psi_rr + psi_zz
        } else {
            psi_rr + psi_r / rho + psi_zz
        };

        let phi = engine.value(Pair::Right(Context::Phi));
        let phi_r = engine.derivative(Pair::Right(Context::Phi), 0);
        let phi_z = engine.derivative(Pair::Right(Context::Phi), 1);

        let source = 0.5 * (phi_r * phi_r + phi_z * phi_z)
            + 0.5 * (2.0 * rho * seed).exp() * psi.powi(4) * crate::MASS * crate::MASS * phi * phi;

        let result = laplacian
            + psi / 4.0 * (rho * seed_rr + 2.0 * seed_r + rho * seed_zz)
            + psi / 4.0 * source;

        SystemValue::new([result])
    }
}

#[derive(Clone)]
pub struct Hamiltonian;

impl Function<2> for Hamiltonian {
    type Input = Pair<Psi, Context>;
    type Output = Scalar;

    fn evaluate(&self, engine: &impl Engine<2, Self::Input>) -> SystemValue<Self::Output> {
        let [rho, _z] = engine.position();

        let psi = engine.value(Pair::Left(Psi));
        let psi_r = engine.derivative(Pair::Left(Psi), 0);

        let psi_rr = engine.second_derivative(Pair::Left(Psi), 0, 0);
        let psi_zz = engine.second_derivative(Pair::Left(Psi), 1, 1);

        let seed = engine.value(Pair::Right(Context::Seed));
        let seed_r = engine.derivative(Pair::Right(Context::Seed), 0);
        let seed_rr = engine.second_derivative(Pair::Right(Context::Seed), 0, 0);
        let seed_zz = engine.second_derivative(Pair::Right(Context::Seed), 1, 1);

        let laplacian = if rho.abs() <= 10e-10 {
            2.0 * psi_rr + psi_zz
        } else {
            psi_rr + psi_r / rho + psi_zz
        };

        let phi = engine.value(Pair::Right(Context::Phi));
        let phi_r = engine.derivative(Pair::Right(Context::Phi), 0);
        let phi_z = engine.derivative(Pair::Right(Context::Phi), 1);

        let source = 0.5 * (phi_r * phi_r + phi_z * phi_z)
            + 0.5 * (2.0 * rho * seed).exp() * psi.powi(4) * crate::MASS * crate::MASS * phi * phi;

        let result = laplacian
            + psi / 4.0 * (rho * seed_rr + 2.0 * seed_r + rho * seed_zz)
            + psi / 4.0 * source;

        SystemValue::new([result])
    }
}

#[derive(Clone)]
pub struct RinneFromGarfinkle;

impl Function<2> for RinneFromGarfinkle {
    type Input = Pair<Psi, Context>;
    type Output = Rinne;

    fn evaluate(&self, engine: &impl Engine<2, Self::Input>) -> SystemValue<Self::Output> {
        let [rho, _z] = engine.position();

        let psi = engine.value(Pair::Left(Psi));
        let seed = engine.value(Pair::Right(Context::Seed));
        let phi = engine.value(Pair::Right(Context::Phi));

        let mut result: SystemValue<_> = SystemValue::default();
        result.set_field(Rinne::Conformal, psi.powi(4) * (2.0 * rho * seed).exp());
        result.set_field(Rinne::Seed, -seed);
        result.set_field(Rinne::Phi, phi);

        result
    }
}

pub fn solve(
    mesh: &mut Mesh<2>,
    max_steps: usize,
    rinne: SystemSliceMut<Rinne>,
    hamiltonian: &mut [f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let num_nodes = mesh.num_nodes();

    log::info!("Solving Garfinkle with {} nodes", num_nodes);

    let mut psi = SystemVec::with_length(num_nodes);
    let mut context = SystemVec::with_length(num_nodes);

    // Compute seed values.
    log::info!("Filling Seed Function");
    mesh.project(ORDER, Quadrant, ContextProjection, context.as_mut_slice());
    mesh.fill_boundary(ORDER, Quadrant, ContextConditions, context.as_mut_slice());

    // Initial Guess for Psi
    psi.field_mut(Psi).fill(1.0);

    log::info!("Running Hyperbolic Relaxation");

    let mut solver = HyperRelaxSolver::new();
    solver.tolerance = 1e-6;
    solver.max_steps = max_steps;
    solver.cfl = 0.5;
    solver.dampening = 0.4;

    solver.solve(
        mesh,
        ORDER,
        Quadrant,
        PsiConditions,
        PsiOperator,
        context.as_slice(),
        psi.as_mut_slice(),
    );

    // Fill garkfinkle again.
    mesh.fill_boundary(ORDER, Quadrant, PsiConditions, psi.as_mut_slice());
    mesh.fill_boundary(ORDER, Quadrant, ContextConditions, context.as_mut_slice());

    let mut source = SystemVec::with_length(num_nodes);
    source
        .field_mut(Pair::Left(Psi))
        .copy_from_slice(psi.field(Psi));
    source
        .field_mut(Pair::Right(Context::Seed))
        .copy_from_slice(context.field(Context::Seed));
    source
        .field_mut(Pair::Right(Context::Phi))
        .copy_from_slice(context.field(Context::Phi));

    mesh.evaluate(
        ORDER,
        Quadrant,
        RinneFromGarfinkle,
        source.as_slice(),
        rinne,
    );

    mesh.evaluate(
        ORDER,
        Quadrant,
        Hamiltonian,
        source.as_slice(),
        SystemSliceMut::from_contiguous(hamiltonian),
    );

    Ok(())
}
