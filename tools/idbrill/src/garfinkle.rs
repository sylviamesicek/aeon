use std::path::PathBuf;

use super::Rinne;
use aeon::{
    elliptic::HyperRelaxSolver,
    fd::{Discretization, ExportVtkConfig},
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
pub struct BoundaryConditions;

impl Boundary<2> for BoundaryConditions {
    fn kind(&self, face: Face<2>) -> BoundaryKind {
        if face.side == false {
            BoundaryKind::Parity
        } else {
            BoundaryKind::Radiative
        }
    }
}

impl Conditions<2> for BoundaryConditions {
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

const PSI_COND: SystemBC<Garfinkle, BoundaryConditions> =
    SystemBC::new(Garfinkle::Psi, BoundaryConditions);

const SEED_COND: SystemBC<Garfinkle, BoundaryConditions> =
    SystemBC::new(Garfinkle::Seed, BoundaryConditions);

#[derive(Clone)]
pub struct SeedFunction(f64);

impl Function<2> for SeedFunction {
    type Output = Scalar;

    fn evaluate(&self, position: [f64; 2]) -> SystemValue<Self::Output> {
        let [rho, z] = position;
        SystemValue::new([rho * self.0 * (-(rho * rho + z * z)).exp()])
    }
}

#[derive(Clone)]
pub struct PsiOperator;

impl Operator<2> for PsiOperator {
    type System = Scalar;
    type Context = Scalar;

    fn apply(
        &self,
        engine: &impl Engine<2>,
        psi: SystemFields<'_, Self::System>,
        seed: SystemFields<'_, Self::Context>,
    ) -> SystemValue<Self::System> {
        let psi = psi.field(Scalar);
        let seed = seed.field(Scalar);

        let [rho, _z] = engine.position();

        let psi_val = engine.value(psi);
        let psi_grad = engine.gradient(PSI_COND, psi);
        let psi_hess = engine.hessian(PSI_COND, psi);

        let seed_grad = engine.gradient(SEED_COND, seed);
        let seed_hess = engine.hessian(SEED_COND, seed);

        let laplacian = if rho.abs() <= 10e-10 {
            2.0 * psi_hess[0][0] + psi_hess[1][1]
        } else {
            psi_hess[0][0] + psi_grad[0] / rho + psi_hess[1][1]
        };

        let result = laplacian
            + psi_val / 4.0 * (rho * seed_hess[0][0] + 2.0 * seed_grad[0] + rho * seed_hess[1][1]);

        SystemValue::new([result])
    }

    fn callback(
        &self,
        discrete: &mut Discretization<2>,
        system: SystemSlice<Self::System>,
        context: SystemSlice<Self::Context>,
        index: usize,
    ) {
        if index % 25 == 0 {
            let mut garfinkle = SystemVec::with_length(discrete.mesh().num_nodes());
            garfinkle
                .field_mut(Garfinkle::Psi)
                .copy_from_slice(system.field(Scalar));
            garfinkle
                .field_mut(Garfinkle::Seed)
                .copy_from_slice(context.field(Scalar));

            let mut hamiltonian = vec![0.0; discrete.mesh().num_nodes()];

            discrete.order::<4>().project(
                BoundaryConditions,
                Hamiltonian,
                garfinkle.as_slice(),
                hamiltonian.as_mut_slice().into(),
            );

            let mut model = Model::empty();
            model.set_mesh(discrete.mesh());
            model.write_field("psi", garfinkle.field(Garfinkle::Psi).to_vec());
            model.write_field("hamiltonian", hamiltonian);

            let path = PathBuf::from(format!("output/garfinkle/iter{}.vtu", { index / 25 }));
            let config = ExportVtkConfig {
                title: "garfinkle".to_string(),
                ghost: false,
            };

            model.export_vtk(path, config).unwrap();
        }
    }
}

#[derive(Clone)]
pub struct Hamiltonian;

impl Projection<2> for Hamiltonian {
    type Input = Garfinkle;
    type Output = Scalar;

    fn project(
        &self,
        engine: &impl Engine<2>,
        input: SystemFields<'_, Self::Input>,
    ) -> SystemValue<Self::Output> {
        let psi = input.field(Garfinkle::Psi);
        let seed = input.field(Garfinkle::Seed);

        let [rho, _z] = engine.position();

        let psi_val = engine.value(psi);
        let psi_grad = engine.gradient(PSI_COND, psi);
        let psi_hess = engine.hessian(PSI_COND, psi);

        let seed_grad = engine.gradient(SEED_COND, seed);
        let seed_hess = engine.hessian(SEED_COND, seed);

        let laplacian = if rho.abs() <= 10e-10 {
            2.0 * psi_hess[0][0] + psi_hess[1][1]
        } else {
            psi_hess[0][0] + psi_grad[0] / rho + psi_hess[1][1]
        };

        let result = laplacian
            + psi_val / 4.0 * (rho * seed_hess[0][0] + 2.0 * seed_grad[0] + rho * seed_hess[1][1]);

        SystemValue::new([result])
    }
}

#[derive(Clone)]
pub struct RinneFromGarfinkle;

impl Projection<2> for RinneFromGarfinkle {
    type Input = Garfinkle;
    type Output = Rinne;

    fn project(
        &self,
        engine: &impl Engine<2>,
        input: SystemFields<'_, Self::Input>,
    ) -> SystemValue<Self::Output> {
        let [rho, _z] = engine.position();

        let psi = engine.value(input.field(Garfinkle::Psi));
        let seed = engine.value(input.field(Garfinkle::Seed));

        let mut result: SystemValue<_> = SystemValue::default();
        result.set_field(Rinne::Conformal, psi.powi(4) * (2.0 * rho * seed).exp());
        result.set_field(Rinne::Seed, -seed);
        result
    }
}

pub fn solve(
    discrete: &mut Discretization<2>,
    amplitude: f64,
    rinne: SystemSliceMut<Rinne>,
    hamiltonian: &mut [f64],
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Filling Seed Function");

    let num_nodes = discrete.mesh().num_nodes();

    let mut garfinkle = vec![0.0; num_nodes * 2];
    let (psi, seed) = garfinkle.split_at_mut(num_nodes);

    // Compute seed values.
    discrete
        .order::<4>()
        .evaluate(SeedFunction(amplitude), seed.into());
    discrete.order::<4>().fill_boundary(SEED_COND, seed.into());

    // Initial Guess for Psi
    psi.fill(1.0);

    log::info!("Running Hyperbolic Relaxation");

    let mut solver = HyperRelaxSolver::new();
    solver.tolerance = 1e-9;
    solver.max_steps = 10000;
    solver.cfl = 0.1;
    solver.dampening = 0.4;

    solver.solve::<2, 4, _, _>(discrete, PSI_COND, PsiOperator, seed.into(), psi.into());

    // Fill garkfinkle again.
    discrete.order::<4>().fill_boundary(
        BoundaryConditions,
        SystemSliceMut::from_contiguous(&mut garfinkle),
    );

    discrete.order::<4>().project(
        BoundaryConditions,
        RinneFromGarfinkle,
        SystemSlice::from_contiguous(&garfinkle),
        rinne,
    );

    discrete.order::<4>().project(
        BoundaryConditions,
        Hamiltonian,
        SystemSlice::from_contiguous(&garfinkle),
        SystemSliceMut::from_contiguous(hamiltonian),
    );

    Ok(())
}
