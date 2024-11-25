use aeon::{
    basis::{Kernels, RadiativeParams},
    elliptic::HyperRelaxSolver,
    fd::{Mesh, SystemCondition},
    prelude::*,
};

use crate::config::{Brill, Solver};

/// Quadrant upon which all simulations run.
#[derive(Clone)]
pub struct Quadrant;

impl Boundary<2> for Quadrant {
    fn kind(&self, face: Face<2>) -> BoundaryKind {
        if !face.side {
            BoundaryKind::Parity
        } else {
            BoundaryKind::Radiative
        }
    }
}

/// Initial data in Rinne's hyperbolic variables.
#[derive(Clone, SystemLabel)]
pub enum Rinne {
    Conformal,
    Seed,
}

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

    fn radiative(
        &self,
        field: Self::System,
        _position: [f64; 2],
        _spacing: f64,
    ) -> RadiativeParams {
        match field {
            Garfinkle::Psi => RadiativeParams::lightlike(1.0),
            Garfinkle::Seed => RadiativeParams::lightlike(0.0),
        }
    }
}

const PSI_CONDITIONS: ScalarConditions<SystemCondition<Garfinkle, GarfinkleConditions>> =
    ScalarConditions::new(SystemCondition::new(Garfinkle::Psi, GarfinkleConditions));

const SEED_CONDITIONS: ScalarConditions<SystemCondition<Garfinkle, GarfinkleConditions>> =
    ScalarConditions::new(SystemCondition::new(Garfinkle::Seed, GarfinkleConditions));

#[derive(Clone)]
pub struct SeedProjection {
    amplitude: f64,
    sigma: (f64, f64),
}

impl Projection<2> for SeedProjection {
    type Output = Scalar;

    fn project(&self, position: [f64; 2]) -> SystemValue<Self::Output> {
        let [rho, z] = position;

        let rho2 = rho * rho;
        let z2 = z * z;

        let srho2 = self.sigma.0 * self.sigma.0;
        let sz2 = self.sigma.1 * self.sigma.1;

        SystemValue::new([rho * self.amplitude * (-rho2 / srho2 - z2 / sz2).exp()])
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
    // if index % 25 == 0 {
    //     let max_level = mesh.max_level();

    //     let mut garfinkle = SystemVec::with_length(mesh.num_nodes());
    //     garfinkle
    //         .field_mut(Garfinkle::Psi)
    //         .copy_from_slice(system.field(Scalar));
    //     garfinkle
    //         .field_mut(Garfinkle::Seed)
    //         .copy_from_slice(context.field(Scalar));

    //     let mut hamiltonian = vec![0.0; mesh.num_nodes()];

    //     mesh.fill_boundary(
    //         ORDER,
    //         Quadrant,
    //         GarfinkleConditions,
    //         garfinkle.as_mut_slice(),
    //     );

    //     mesh.evaluate(
    //         ORDER,
    //         Quadrant,
    //         Hamiltonian,
    //         garfinkle.as_slice(),
    //         hamiltonian.as_mut_slice().into(),
    //     );

    //     mesh.fill_boundary(
    //         ORDER,
    //         Quadrant,
    //         HAMILTONIAN_CONDITIONS,
    //         hamiltonian.as_mut_slice().into(),
    //     );

    //     let mut systems = SystemCheckpoint::default();
    //     systems.save_field("psi", garfinkle.field(Garfinkle::Psi));
    //     systems.save_field("hamiltonian", &hamiltonian);

    //     std::fs::create_dir_all(format!("output/idbrill/level{}", max_level)).unwrap();

    //     mesh.export_vtu(
    //         format!("output/idbrill/level{}/relax{}.vtu", max_level, index / 25),
    //         &systems,
    //         ExportVtuConfig {
    //             title: "garfinkle".to_string(),
    //             ghost: crate::GHOST,
    //         },
    //     )
    //     .unwrap();
    // }
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

pub fn solve_wth_garfinkle<const ORDER: usize>(
    order: Order<ORDER>,
    mesh: &mut Mesh<2>,
    solver_con: &Solver,
    brill: &Brill,
) -> anyhow::Result<()>
where
    Order<ORDER>: Kernels,
{
    let num_nodes = mesh.num_nodes();

    // Back garfinkle variables into single variable.
    let mut garfinkle = vec![0.0; num_nodes * 2];
    let (psi, seed) = garfinkle.split_at_mut(num_nodes);

    // Compute seed values.
    log::trace!("Filling Seed Function");
    mesh.project(
        order,
        Quadrant,
        SeedProjection {
            amplitude: brill.amplitude,
            sigma: brill.sigma,
        },
        seed.into(),
    );
    mesh.fill_boundary(order, Quadrant, SEED_CONDITIONS, seed.into());

    // Initial Guess for Psi
    psi.fill(1.0);

    log::trace!("Running Hyperbolic Relaxation");

    let mut solver = HyperRelaxSolver::new();
    solver.dampening = solver_con.dampening;
    solver.max_steps = solver_con.max_steps;
    solver.tolerance = solver_con.tolerance;
    solver.cfl = solver_con.cfl;

    solver.solve(
        mesh,
        order,
        Quadrant,
        PSI_CONDITIONS,
        PsiOperator,
        seed.into(),
        psi.into(),
    );

    // Fill garkfinkle again.
    mesh.fill_boundary(
        order,
        Quadrant,
        GarfinkleConditions,
        SystemSliceMut::from_contiguous(&mut garfinkle),
    );

    let mut rinne = SystemVec::with_length(num_nodes);

    mesh.evaluate(
        order,
        Quadrant,
        RinneFromGarfinkle,
        SystemSlice::from_contiguous(&garfinkle),
        rinne.as_mut_slice(),
    );

    let mut hamiltonian = SystemVec::with_length(num_nodes);

    mesh.evaluate(
        order,
        Quadrant,
        Hamiltonian,
        SystemSlice::from_contiguous(&garfinkle),
        hamiltonian.as_mut_slice(),
    );

    Ok(())
}
