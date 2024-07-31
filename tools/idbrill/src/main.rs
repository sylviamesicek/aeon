use aeon::array::Array;
use aeon::elliptic::HyperRelaxSolver;
use aeon::fd::{Boundary, BoundaryKind, Conditions, Engine, Function, Projection, SystemBC};
use aeon::prelude::*;
use aeon::system::SystemFields;

use reborrow::Reborrow;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

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
    type Output = ();

    fn evaluate(&self, position: [f64; 2]) -> SystemValue<Self::Output> {
        let [rho, z] = position;
        SystemValue::new([rho * self.0 * (-(rho * rho + z * z)).exp()])
    }
}

#[derive(Clone)]
pub struct SeedDRhoFunction(f64);

impl Function<2> for SeedDRhoFunction {
    type Output = ();

    fn evaluate(&self, position: [f64; 2]) -> SystemValue<Self::Output> {
        let [rho, z] = position;
        SystemValue::new([self.0 * (-(rho * rho + z * z)).exp()
            + rho * self.0 * (-(rho * rho + z * z)).exp() * (-2.0 * rho)])
    }
}

#[derive(Clone)]
pub struct PsiOperator;

impl Operator<2> for PsiOperator {
    type System = ();
    type Context = ();

    fn apply(
        &self,
        engine: &impl Engine<2>,
        psi: SystemFields<'_, Self::System>,
        seed: SystemFields<'_, Self::Context>,
    ) -> SystemValue<Self::System> {
        let psi = psi.field(());
        let seed = seed.field(());

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
pub struct Hamiltonian;

impl Projection<2> for Hamiltonian {
    type Input = Garfinkle;
    type Output = ();

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

/// Initial data in Rinne's hyperbolic variables.
#[derive(Clone, SystemLabel)]
pub enum Rinne {
    Conformal,
    Seed,
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

use clap::{Arg, Command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("idbrill")
        .about("A program for generating brill initial data using hyperbolic relaxation.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("v0.0.1")
        .arg(
            Arg::new("points")
                .num_args(1)
                .short('p')
                .long("points")
                .help("Number of grid points along each axis")
                .value_name("POINTS")
                .default_value("40"),
        )
        .arg(
            Arg::new("radius")
                .num_args(1)
                .short('r')
                .long("radius")
                .help("Size of grid along each axis")
                .value_name("RADIUS")
                .default_value("20.0"),
        )
        .get_matches();

    let points = matches
        .get_one::<String>("points")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(10);
    let radius = matches
        .get_one::<String>("radius")
        .map(|s| s.parse().unwrap())
        .unwrap_or(20.0);

    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    log::info!("Allocating Driver and Building Mesh Grid Size {points} Radius {radius}");

    let bounds = Rectangle {
        size: [radius, radius],
        origin: [0.0, 0.0],
    };

    let size = [40, 40];

    let mut mesh = Mesh::new(bounds, size, 2);
    mesh.refine(&[true, false, false, false]);
    mesh.refine(&[true, false, false, false, false, false, false]);

    std::fs::write("output/mesh.txt", mesh.write_debug()).unwrap();

    println!("Num Blocks: {}", mesh.num_blocks());
    println!("Num Cells: {}", mesh.num_cells());

    log::info!("Filling Seed Function");

    let num_nodes = mesh.num_nodes();

    let mut data = vec![0.0; num_nodes * 2];
    let (psi, seed) = data.split_at_mut(num_nodes);

    // Compute seed values.
    mesh.order::<4>().evaluate(SeedFunction(1.0), seed.into());
    mesh.order::<4>()
        .fill_boundary(UnitBC(SEED_COND), seed.into());

    // Initial Guess for Psi
    psi.fill(1.0);

    log::info!("Running Hyperbolic Relaxation");

    let mut solver = HyperRelaxSolver::new();
    solver.tolerance = 1e-9;
    solver.max_steps = 3000;
    solver.cfl = 0.1;
    solver.dampening = 0.4;

    solver.solve::<2, 4, _, _>(&mut mesh, PSI_COND, PsiOperator, seed.into(), psi.into());
    mesh.order::<4>().fill_boundary(PSI_COND, psi.into());

    let system = SystemSlice::from_contiguous(&data);

    let mut hamiltonian = vec![0.0; mesh.num_nodes()].into_boxed_slice();

    mesh.order::<4>().project(
        BoundaryConditions,
        Hamiltonian,
        system.rb(),
        SystemSliceMut::from_contiguous(&mut hamiltonian),
    );

    let mut model = Model::from_mesh(&mesh);
    model.attach_field(
        "psi",
        system
            .field(Garfinkle::Psi)
            .iter()
            .map(|&p| p - 1.0)
            .collect(),
    );
    model.attach_field("seed", system.field(Garfinkle::Seed).to_vec());
    model.attach_field("hamiltonian", hamiltonian.to_vec());

    model
        .export_vtk(
            format!("idbrill").as_str(),
            PathBuf::from(format!("output/idbrill.vtu")),
        )
        .unwrap();

    // Write model data to file
    {
        let mut rinne = SystemVec::with_length(mesh.num_nodes());

        mesh.order::<4>().project(
            BoundaryConditions,
            RinneFromGarfinkle,
            system.rb(),
            rinne.as_mut_slice(),
        );

        let mut model = Model::from_mesh(&mesh);
        model.attach_system(SystemSlice::<Garfinkle>::from_contiguous(&data));

        let mut file = File::create("output/idbrill.dat")?;
        file.write_all(ron::to_string(&model)?.as_bytes())?;
    }

    Ok(())
}
