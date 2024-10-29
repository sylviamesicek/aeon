use super::{Quadrant, Rinne, ORDER};
use aeon::{
    elliptic::HyperRelaxSolver,
    fd::{Mesh, SystemCondition},
    prelude::*,
};

/// Initial data is Garfinkle's variables.
#[derive(Clone, SystemLabel)]
pub enum Garfinkle {
    Psi,
    Seed,
}

/// Boundary Conditions for Garfinkle variables.
#[derive(Clone)]
pub struct GarfinkleConditions;

impl Conditions<2> for GarfinkleConditions {
    type System = Garfinkle;

    fn parity(&self, field: Self::System, face: Face<2>) -> bool {
        match field {
            Garfinkle::Psi => [true, true][face.axis],
            Garfinkle::Seed => [false, true][face.axis],
        }
    }

    fn radiative(&self, field: Self::System, _position: [f64; 2]) -> f64 {
        match field {
            Garfinkle::Psi => 1.0,
            Garfinkle::Seed => 0.0,
        }
    }
}

const PSI_CONDITIONS: ScalarConditions<SystemCondition<Garfinkle, GarfinkleConditions>> =
    ScalarConditions::new(SystemCondition::new(Garfinkle::Psi, GarfinkleConditions));

const SEED_CONDITIONS: ScalarConditions<SystemCondition<Garfinkle, GarfinkleConditions>> =
    ScalarConditions::new(SystemCondition::new(Garfinkle::Seed, GarfinkleConditions));

#[derive(Clone)]
pub struct SeedProjection(f64);

impl Projection<2> for SeedProjection {
    type Output = Scalar;

    fn project(&self, position: [f64; 2]) -> SystemValue<Self::Output> {
        let [rho, z] = position;
        SystemValue::new([rho * self.0 * (-(rho * rho + z * z)).exp()])
    }
}

#[derive(Clone)]
pub struct PsiOperator;

impl Operator<2> for PsiOperator {
    type System = Scalar;
    type Context = Scalar;

    type SystemConditions = ScalarConditions<SystemCondition<Garfinkle, GarfinkleConditions>>;
    type ContextConditions = ScalarConditions<SystemCondition<Garfinkle, GarfinkleConditions>>;

    fn system_conditions(&self) -> Self::SystemConditions {
        PSI_CONDITIONS
    }

    fn context_conditions(&self) -> Self::ContextConditions {
        SEED_CONDITIONS
    }

    fn apply(
        &self,
        engine: &impl Engine<2, Pair<Self::System, Self::Context>>,
    ) -> SystemValue<Self::System> {
        let [rho, _z] = engine.position();

        let psi = engine.value(Pair::Left(Scalar));
        let psi_r = engine.derivative(Pair::Left(Scalar), 0);

        let psi_rr = engine.second_derivative(Pair::Left(Scalar), 0, 0);
        let psi_zz = engine.second_derivative(Pair::Left(Scalar), 1, 1);

        let seed_r = engine.derivative(Pair::Right(Scalar), 0);
        let seed_rr = engine.second_derivative(Pair::Right(Scalar), 0, 0);
        let seed_zz = engine.second_derivative(Pair::Right(Scalar), 1, 1);

        let laplacian = if rho.abs() <= 10e-10 {
            2.0 * psi_rr + psi_zz
        } else {
            psi_rr + psi_r / rho + psi_zz
        };

        let result = laplacian + psi / 4.0 * (rho * seed_rr + 2.0 * seed_r + rho * seed_zz);

        SystemValue::new([result])
    }

    // fn callback(
    //     &self,
    //     mesh: &mut Mesh<2>,
    //     system: SystemSlice<Self::System>,
    //     context: SystemSlice<Self::Context>,
    //     index: usize,
    // ) {
    //     if index % 25 == 0 {
    //         let mut garfinkle = SystemVec::with_length(mesh.num_nodes());
    //         garfinkle
    //             .field_mut(Garfinkle::Psi)
    //             .copy_from_slice(system.field(Scalar));
    //         garfinkle
    //             .field_mut(Garfinkle::Seed)
    //             .copy_from_slice(context.field(Scalar));

    //         let mut hamiltonian = vec![0.0; mesh.num_nodes()];

    //         mesh.fill_boundary(
    //             ORDER,
    //             Quadrant,
    //             GarfinkleConditions,
    //             garfinkle.as_mut_slice(),
    //         );

    //         mesh.evaluate(
    //             ORDER,
    //             Quadrant,
    //             Hamiltonian,
    //             garfinkle.as_slice(),
    //             hamiltonian.as_mut_slice().into(),
    //         );

    //         mesh.fill_boundary(
    //             ORDER,
    //             Quadrant,
    //             HAMILTONIAN_CONDITIONS,
    //             hamiltonian.as_mut_slice().into(),
    //         );

    //         // let mut systems = SystemCheckpoint::default();
    //         // systems.save_field("psi", garfinkle.field(Garfinkle::Psi));
    //         // systems.save_field("hamiltonian", &hamiltonian);

    //         // mesh.export_vtu(
    //         //     format!("output/garfinkle/iter{}.vtu", { index / 25 }),
    //         //     ExportVtuConfig {
    //         //         title: "garfinkle".to_string(),
    //         //         ghost: crate::GHOST,
    //         //         systems,
    //         //     },
    //         // )
    //         // .unwrap();
    //     }
    // }
}

#[derive(Clone)]
pub struct Hamiltonian;

impl Function<2> for Hamiltonian {
    type Input = Garfinkle;
    type Output = Scalar;

    fn evaluate(&self, engine: &impl Engine<2, Garfinkle>) -> SystemValue<Self::Output> {
        let [rho, _z] = engine.position();

        let psi = engine.value(Garfinkle::Psi);
        let psi_r = engine.derivative(Garfinkle::Psi, 0);

        let psi_rr = engine.second_derivative(Garfinkle::Psi, 0, 0);
        let psi_zz = engine.second_derivative(Garfinkle::Psi, 1, 1);

        let seed_r = engine.derivative(Garfinkle::Seed, 0);
        let seed_rr = engine.second_derivative(Garfinkle::Seed, 0, 0);
        let seed_zz = engine.second_derivative(Garfinkle::Seed, 1, 1);

        let laplacian = if rho.abs() <= 10e-10 {
            2.0 * psi_rr + psi_zz
        } else {
            psi_rr + psi_r / rho + psi_zz
        };

        let result = laplacian + psi / 4.0 * (rho * seed_rr + 2.0 * seed_r + rho * seed_zz);

        SystemValue::new([result])
    }
}

#[derive(Clone)]
pub struct RinneFromGarfinkle;

impl Function<2> for RinneFromGarfinkle {
    type Input = Garfinkle;
    type Output = Rinne;

    fn evaluate(&self, engine: &impl Engine<2, Self::Input>) -> SystemValue<Self::Output> {
        let [rho, _z] = engine.position();

        let psi = engine.value(Garfinkle::Psi);
        let seed = engine.value(Garfinkle::Seed);

        let mut result: SystemValue<_> = SystemValue::default();
        result.set_field(Rinne::Conformal, psi.powi(4) * (2.0 * rho * seed).exp());
        result.set_field(Rinne::Seed, -seed);
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
    let num_nodes = mesh.num_nodes();

    log::info!("Solving Garfinkle with {} nodes", num_nodes);

    let mut garfinkle = vec![0.0; num_nodes * 2];
    let (psi, seed) = garfinkle.split_at_mut(num_nodes);

    // Compute seed values.
    log::info!("Filling Seed Function");
    mesh.project(ORDER, Quadrant, SeedProjection(amplitude), seed.into());
    mesh.fill_boundary(ORDER, Quadrant, SEED_CONDITIONS, seed.into());

    // Initial Guess for Psi
    psi.fill(1.0);

    log::info!("Running Hyperbolic Relaxation");

    let mut solver = HyperRelaxSolver::new();
    solver.tolerance = 1e-6;
    solver.max_steps = max_steps;
    solver.cfl = 0.5;
    solver.dampening = 0.4;

    solver.solve(mesh, ORDER, Quadrant, PsiOperator, seed.into(), psi.into());

    // Fill garkfinkle again.
    mesh.fill_boundary(
        ORDER,
        Quadrant,
        GarfinkleConditions,
        SystemSliceMut::from_contiguous(&mut garfinkle),
    );

    mesh.evaluate(
        ORDER,
        Quadrant,
        RinneFromGarfinkle,
        SystemSlice::from_contiguous(&garfinkle),
        rinne,
    );

    mesh.evaluate(
        ORDER,
        Quadrant,
        Hamiltonian,
        SystemSlice::from_contiguous(&garfinkle),
        hamiltonian.into(),
    );

    Ok(())
}
