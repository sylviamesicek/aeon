use crate::{Quadrant, ORDER};

use super::Rinne;
use aeon::{
    elliptic::HyperRelaxSolver,
    fd::{Mesh, SystemCondition},
    prelude::*,
};

/// Initial data is Garfinkle's variables.
#[derive(Clone, SystemLabel)]
pub enum Choptuik {
    Psi,
    Seed,
}

/// Boundary Conditions for Choptuik variables.
#[derive(Clone)]
pub struct ChoptuikConditions;

impl Conditions<2> for ChoptuikConditions {
    type System = Choptuik;

    fn parity(&self, field: Self::System, face: Face<2>) -> bool {
        match field {
            Choptuik::Psi => [true, true][face.axis],
            Choptuik::Seed => [false, true][face.axis],
        }
    }

    fn radiative(&self, field: Self::System, _position: [f64; 2]) -> f64 {
        match field {
            Choptuik::Psi => 1.0,
            Choptuik::Seed => 0.0,
        }
    }
}

const PSI_CONDITIONS: ScalarConditions<SystemCondition<Choptuik, ChoptuikConditions>> =
    ScalarConditions::new(SystemCondition::new(Choptuik::Psi, ChoptuikConditions));

const SEED_CONDITIONS: ScalarConditions<SystemCondition<Choptuik, ChoptuikConditions>> =
    ScalarConditions::new(SystemCondition::new(Choptuik::Seed, ChoptuikConditions));

#[derive(Clone)]
pub struct SeedProjection(f64);

impl Projection<2> for SeedProjection {
    type Output = Scalar;

    fn project(&self, position: [f64; 2]) -> SystemValue<Self::Output> {
        let [rho, z] = position;
        SystemValue::new([-rho * self.0 * (-(rho * rho + z * z)).exp()])
    }
}

#[derive(Clone)]
pub struct PsiOperator;

impl Operator<2> for PsiOperator {
    type System = Scalar;
    type Context = Scalar;

    fn apply(
        &self,
        engine: &impl Engine<2, Pair<Self::System, Self::Context>>,
    ) -> SystemValue<Self::System> {
        let [rho, _z] = engine.position();
        let on_axis = rho.abs() <= 10e-10;

        let psi = engine.value(Pair::Left(Scalar));
        let psi_r = engine.derivative(Pair::Left(Scalar), 0);
        let psi_z = engine.derivative(Pair::Left(Scalar), 1);

        let psi_rr = engine.second_derivative(Pair::Left(Scalar), 0, 0);
        let psi_zz = engine.second_derivative(Pair::Left(Scalar), 1, 1);

        let seed = engine.value(Pair::Right(Scalar));
        let seed_r = engine.derivative(Pair::Right(Scalar), 0);
        let seed_z = engine.derivative(Pair::Right(Scalar), 1);
        let seed_rr = engine.second_derivative(Pair::Right(Scalar), 0, 0);
        let seed_zz = engine.second_derivative(Pair::Right(Scalar), 1, 1);

        let term1 = if on_axis {
            2.0 * psi_rr + psi_zz
        } else {
            psi_rr + psi_r / rho + psi_zz
        };

        let term2 = (seed + rho * seed_r) * psi_r + rho * seed_z * psi_z;

        let mut term3 = rho * seed_rr
            + 4.0 * seed_r
            + (seed + rho * seed_r).powi(2)
            + rho * seed_zz
            + (rho * seed_z).powi(2);

        if on_axis {
            term3 += 2.0 * seed_r;
        } else {
            term3 += 2.0 * seed / rho
        }

        let result = term1 + term2 + psi / 4.0 * term3;

        SystemValue::new([result])
    }
}

#[derive(Clone)]
pub struct Hamiltonian;

impl Function<2> for Hamiltonian {
    type Input = Choptuik;
    type Output = Scalar;

    fn evaluate(&self, engine: &impl Engine<2, Self::Input>) -> SystemValue<Self::Output> {
        let [rho, _z] = engine.position();
        let on_axis = rho.abs() <= 10e-10;

        let psi = engine.value(Choptuik::Psi);
        let psi_r = engine.derivative(Choptuik::Psi, 0);
        let psi_z = engine.derivative(Choptuik::Psi, 1);

        let psi_rr = engine.second_derivative(Choptuik::Psi, 0, 0);
        let psi_zz = engine.second_derivative(Choptuik::Psi, 1, 1);

        let seed = engine.value(Choptuik::Seed);
        let seed_r = engine.derivative(Choptuik::Seed, 0);
        let seed_z = engine.derivative(Choptuik::Seed, 1);
        let seed_rr = engine.second_derivative(Choptuik::Seed, 0, 0);
        let seed_zz = engine.second_derivative(Choptuik::Seed, 1, 1);

        let term1 = if on_axis {
            2.0 * psi_rr + psi_zz
        } else {
            psi_rr + psi_r / rho + psi_zz
        };

        let term2 = (seed + rho * seed_r) * psi_r + rho * seed_z * psi_z;

        let mut term3 = rho * seed_rr
            + 4.0 * seed_r
            + (seed + rho * seed_r).powi(2)
            + rho * seed_zz
            + (rho * seed_z).powi(2);

        if on_axis {
            term3 += 2.0 * seed_r;
        } else {
            term3 += 2.0 * seed / rho
        }

        let result = term1 + term2 + psi / 4.0 * term3;

        SystemValue::new([result])
    }
}

#[derive(Clone)]
pub struct RinneFromChoptuik;

impl Function<2> for RinneFromChoptuik {
    type Input = Choptuik;
    type Output = Rinne;

    fn evaluate(&self, engine: &impl Engine<2, Self::Input>) -> SystemValue<Self::Output> {
        let psi = engine.value(Choptuik::Psi);
        let seed = engine.value(Choptuik::Seed);

        let mut result: SystemValue<_> = SystemValue::default();
        result.set_field(Rinne::Conformal, psi.powi(4));
        result.set_field(Rinne::Seed, seed);
        result
    }
}

pub fn solve(
    mesh: &mut Mesh<2>,
    amplitude: f64,
    max_steps: usize,
    rinne: SystemSliceMut<Rinne>,
    hamiltonian: &mut [f64],
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Filling Seed Function");

    let num_nodes = mesh.num_nodes();

    let mut choptuik = vec![0.0; num_nodes * 2];
    let (psi, seed) = choptuik.split_at_mut(num_nodes);

    // Compute seed values.
    mesh.project(ORDER, Quadrant, SeedProjection(amplitude), seed.into());
    mesh.fill_boundary(ORDER, Quadrant, SEED_CONDITIONS, seed.into());

    // Initial Guess for Psi
    psi.fill(1.0);

    log::info!("Running Hyperbolic Relaxation");

    let mut solver = HyperRelaxSolver::new();
    solver.tolerance = 1e-9;
    solver.max_steps = max_steps;
    solver.cfl = 0.1;
    solver.dampening = 0.4;

    solver.solve(
        mesh,
        ORDER,
        Quadrant,
        PSI_CONDITIONS,
        PsiOperator,
        seed.into(),
        psi.into(),
    );

    // Fill garkfinkle again.
    mesh.fill_boundary(
        ORDER,
        Quadrant,
        ChoptuikConditions,
        SystemSliceMut::from_contiguous(&mut choptuik),
    );

    mesh.evaluate(
        ORDER,
        Quadrant,
        RinneFromChoptuik,
        SystemSlice::from_contiguous(&choptuik),
        rinne,
    );

    mesh.evaluate(
        ORDER,
        Quadrant,
        Hamiltonian,
        SystemSlice::from_contiguous(&choptuik),
        hamiltonian.into(),
    );

    Ok(())
}
