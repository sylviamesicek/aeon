use aeon::array::Array;
use aeon::elliptic::HyperRelaxSolver;
use aeon::fd::{Boundary, BoundaryKind, Condition, Conditions, Engine, Projection};
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

#[derive(Clone, SystemLabel)]
pub struct Psi;

#[derive(Clone, SystemLabel)]
pub struct Seed;

/// Initial data in Rinne's hyperbolic variables.
#[derive(Clone, SystemLabel)]
pub enum Rinne {
    Conformal,
    Seed,
}

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
        let index = engine.index();

        let psi = input.field(Garfinkle::Psi)[index];
        let seed = input.field(Garfinkle::Seed)[index];

        let mut result: SystemValue<_> = SystemValue::default();
        result.set_field(Rinne::Conformal, psi.powi(4) * (2.0 * rho * seed).exp());
        result.set_field(Rinne::Seed, -seed);
        result
    }
}

/// Boundary for quadrant of domain.
#[derive(Clone)]
pub struct Quadrant;

impl Boundary for Quadrant {
    fn kind(&self, face: Face) -> aeon::fd::BoundaryKind {
        if face.side == false {
            BoundaryKind::Parity
        } else {
            BoundaryKind::Radiative
        }
    }
}

/// Boundary Conditions for Garfinkle variables.
pub struct GarfinkleConditions;

impl Conditions<2> for GarfinkleConditions {
    type Condition = GarfinkleCondition;
    type System = Garfinkle;

    fn field(&self, label: Self::System) -> Self::Condition {
        match label {
            Garfinkle::Psi => GarfinkleCondition([true, true], 1.0),
            Garfinkle::Seed => GarfinkleCondition([false, true], 0.0),
        }
    }
}

pub struct GarfinkleCondition([bool; 2], f64);

impl Condition<2> for GarfinkleCondition {
    fn parity(&self, face: Face) -> bool {
        self.0[face.axis]
    }

    fn radiative(&self, _position: [f64; 2]) -> f64 {
        self.1
    }
}

pub struct PsiConditions;

impl Conditions<2> for PsiConditions {
    type System = Psi;
    type Condition = GarfinkleCondition;

    fn field(&self, _label: Self::System) -> Self::Condition {
        GarfinkleConditions.field(Garfinkle::Psi)
    }
}

pub struct SeedConditions;

impl Conditions<2> for SeedConditions {
    type System = Seed;
    type Condition = GarfinkleCondition;

    fn field(&self, _label: Self::System) -> Self::Condition {
        GarfinkleConditions.field(Garfinkle::Seed)
    }
}

pub struct PsiOperator;

impl Operator<2> for PsiOperator {
    type System = Psi;
    type Context = Seed;

    fn evaluate(
        &self,
        engine: &impl Engine<2>,
        psi: SystemFields<'_, Self::System>,
        seed: SystemFields<'_, Self::Context>,
    ) -> SystemValue<Self::System> {
        let psi = psi.field(Psi);
        let seed = seed.field(Seed);

        let [rho, _z] = engine.position();
        let index = engine.index();

        let psi_val = psi[index];
        let psi_grad = engine.gradient(psi);
        let psi_hess = engine.hessian(psi);

        let seed_grad = engine.gradient(seed);
        let seed_hess = engine.hessian(seed);

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
        let index = engine.index();

        let psi_val = psi[index];
        let psi_grad = engine.gradient(psi);
        let psi_hess = engine.hessian(psi);

        let seed_grad = engine.gradient(seed);
        let seed_hess = engine.hessian(seed);

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

pub struct SeedProjection(f64);

impl Projection<2> for SeedProjection {
    type Input = Scalar;
    type Output = Scalar;

    fn project(
        &self,
        engine: &impl Engine<2>,
        _input: SystemFields<'_, Self::Input>,
    ) -> SystemValue<Self::Output> {
        let [rho, z] = engine.position();
        SystemValue::new([rho * self.0 * (-(rho * rho + z * z)).exp()])
    }
}

pub struct SeedDRhoProjection(f64);

impl Projection<2> for SeedDRhoProjection {
    type Input = Scalar;
    type Output = Scalar;

    fn project(
        &self,
        engine: &impl Engine<2>,
        _input: SystemFields<'_, Self::Input>,
    ) -> SystemValue<Self::Output> {
        let [rho, z] = engine.position();
        SystemValue::new([self.0 * (-(rho * rho + z * z)).exp()
            + rho * self.0 * (-(rho * rho + z * z)).exp() * (-2.0 * rho)])
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

    println!("Num Blocks: {}", mesh.num_blocks());
    println!("Num Cells: {}", mesh.num_cells());

    log::info!("Filling Seed Function");

    let num_nodes = mesh.num_nodes();

    let mut data = vec![0.0; num_nodes * 2];
    let (psi, seed) = data.split_at_mut(num_nodes);

    // Compute seed values.
    mesh.order::<4>().project(
        &Quadrant,
        &SeedProjection(1.0),
        SystemSlice::from_contiguous(&[]),
        SystemSliceMut::from_contiguous(seed),
    );
    mesh.order::<4>().fill_boundary(
        &Quadrant,
        &SeedConditions,
        SystemSliceMut::from_contiguous(seed),
    );

    // Initial Guess for Psi
    psi.fill(1.0);

    log::info!("Running Hyperbolic Relaxation");

    let mut solver = HyperRelaxSolver::new();
    solver.tolerance = 1e-9;
    solver.max_steps = 1000;
    solver.cfl = 0.1;
    solver.dampening = 0.4;

    solver.solve::<2, 4, _>(
        &mut mesh,
        &Quadrant,
        &PsiConditions,
        &PsiOperator,
        SystemSlice::from_contiguous(seed),
        SystemSliceMut::from_contiguous(psi),
    );
    mesh.order::<4>().fill_boundary(
        &Quadrant,
        &PsiConditions,
        SystemSliceMut::from_contiguous(psi),
    );

    let system = SystemSlice::from_contiguous(&data);

    let mut hamiltonian = vec![0.0; mesh.num_nodes()].into_boxed_slice();

    mesh.order::<4>().project(
        &Quadrant,
        &Hamiltonian,
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
            &Quadrant,
            &RinneFromGarfinkle,
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
