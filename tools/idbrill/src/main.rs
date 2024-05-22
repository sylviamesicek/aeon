use aeon::array::Array;
use aeon::elliptic::{HyperRelaxSolver, OutgoingOrder, OutgoingWave};
use aeon::prelude::*;

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

#[derive(Clone)]
pub enum InitialData {
    Conformal,
    Seed,
}

impl SystemLabel for InitialData {
    const NAME: &'static str = "InitialData";
    type FieldLike<T> = [T; 2];
    fn fields() -> Array<Self::FieldLike<Self>> {
        [InitialData::Conformal, InitialData::Seed].into()
    }

    fn field_index(&self) -> usize {
        match self {
            InitialData::Conformal => 0,
            InitialData::Seed => 1,
        }
    }

    fn field_name(&self) -> String {
        match self {
            InitialData::Conformal => "Conformal".to_string(),
            InitialData::Seed => "Seed".to_string(),
        }
    }
}

pub struct InitialDataProjection<'a> {
    seed: &'a [f64],
    psi: &'a [f64],
}

impl<'a> SystemProjection<2> for InitialDataProjection<'a> {
    type Label = InitialData;

    fn evaluate(
        &self,
        block: Block<2>,
        _pool: &MemPool,
        mut dest: SystemSliceMut<'_, Self::Label>,
    ) {
        let range = block.local_from_global();

        let psi = &self.psi[range.clone()];
        let seed = &self.seed[range.clone()];

        for vertex in block.iter() {
            let [rho, _z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            dest.field_mut(InitialData::Conformal)[index] =
                psi[index].powi(4) * (2.0 * rho * seed[index]).exp();
            dest.field_mut(InitialData::Seed)[index] = -seed[index];
        }
    }
}

pub struct OddBoundary;

impl Boundary for OddBoundary {
    fn face(&self, face: Face) -> BoundaryCondition {
        if face.axis == 0 && face.side == false {
            BoundaryCondition::Parity(false)
        } else if face.axis == 1 && face.side == false {
            BoundaryCondition::Parity(true)
        } else {
            BoundaryCondition::Free
        }
    }
}

pub struct EvenBoundary;

impl Boundary for EvenBoundary {
    fn face(&self, face: Face) -> BoundaryCondition {
        if face.axis == 0 && face.side == false {
            BoundaryCondition::Parity(true)
        } else if face.axis == 1 && face.side == false {
            BoundaryCondition::Parity(true)
        } else {
            BoundaryCondition::Free
        }
    }
}

pub struct PsiBoundary;

impl SystemBoundary for PsiBoundary {
    type Label = Scalar;
    type Boundary = EvenBoundary;

    fn field(&self, _: Self::Label) -> Self::Boundary {
        EvenBoundary
    }
}

pub struct PsiOperator<'a> {
    pub seed: &'a [f64],
}

impl<'a> SystemOperator<2> for PsiOperator<'a> {
    type Label = Scalar;

    fn apply(
        &self,
        block: Block<2>,
        pool: &MemPool,
        src: SystemSlice<'_, Self::Label>,
        mut dest: SystemSliceMut<'_, Self::Label>,
    ) {
        let node_count = block.node_count();
        let range = block.local_from_global();

        let psi = src.field(Scalar);
        let psi_r = pool.alloc_scalar(node_count);
        let psi_z = pool.alloc_scalar(node_count);
        let psi_rr = pool.alloc_scalar(node_count);
        let psi_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &EvenBoundary, psi, psi_r);
        block.derivative::<4>(1, &EvenBoundary, psi, psi_z);
        block.second_derivative::<4>(0, &EvenBoundary, psi, psi_rr);
        block.second_derivative::<4>(1, &EvenBoundary, psi, psi_zz);

        let dest = dest.field_mut(Scalar);

        let seed = &self.seed[range.clone()];
        let seed_r = pool.alloc_scalar(node_count);
        let seed_rr = pool.alloc_scalar(node_count);
        let seed_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &OddBoundary, seed, seed_r);
        block.second_derivative::<4>(0, &OddBoundary, seed, seed_rr);
        block.second_derivative::<4>(1, &OddBoundary, seed, seed_zz);

        for vertex in block.iter() {
            let [rho, _z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            let laplacian = if rho.abs() <= 10e-10 {
                2.0 * psi_rr[index] + psi_zz[index]
            } else {
                psi_rr[index] + psi_r[index] / rho + psi_zz[index]
            };

            dest[index] = laplacian
                + psi[index] / 4.0
                    * (rho * seed_rr[index] + 2.0 * seed_r[index] + rho * seed_zz[index]);
        }
    }
}

pub struct Hamiltonian<'a> {
    pub psi: &'a [f64],
    pub seed: &'a [f64],
}

impl<'a> Projection<2> for Hamiltonian<'a> {
    fn evaluate(&self, block: Block<2>, pool: &MemPool, dest: &mut [f64]) {
        let node_count = block.node_count();
        let range = block.local_from_global();

        let psi = &self.psi[range.clone()];
        let psi_r = pool.alloc_scalar(node_count);
        let psi_z = pool.alloc_scalar(node_count);
        let psi_rr = pool.alloc_scalar(node_count);
        let psi_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &EvenBoundary, psi, psi_r);
        block.derivative::<4>(1, &EvenBoundary, psi, psi_z);
        block.second_derivative::<4>(0, &EvenBoundary, psi, psi_rr);
        block.second_derivative::<4>(1, &EvenBoundary, psi, psi_zz);

        let seed = &self.seed[range.clone()];
        let seed_r = pool.alloc_scalar(node_count);
        let seed_rr = pool.alloc_scalar(node_count);
        let seed_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &OddBoundary, seed, seed_r);
        block.second_derivative::<4>(0, &OddBoundary, seed, seed_rr);
        block.second_derivative::<4>(1, &OddBoundary, seed, seed_zz);

        for vertex in block.iter() {
            let [rho, _z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            let laplacian = if rho.abs() <= 10e-10 {
                2.0 * psi_rr[index] + psi_zz[index]
            } else {
                psi_rr[index] + psi_r[index] / rho + psi_zz[index]
            };

            dest[index] = laplacian
                + psi[index] / 4.0
                    * (rho * seed_rr[index] + 2.0 * seed_r[index] + rho * seed_zz[index]);
        }
    }
}

pub struct SeedProjection(f64);

impl Projection<2> for SeedProjection {
    fn evaluate(&self, block: aeon::mesh::Block<2>, _pool: &MemPool, dest: &mut [f64]) {
        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            dest[index] = rho * self.0 * (-(rho * rho + z * z)).exp();
        }
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
                .short('s')
                .long("spatial")
                .help("Number of grid points along each axis")
                .value_name("POINTS")
                .default_value("80"),
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
        .get_one::<String>("spatial")
        .map(|s| s.parse().unwrap())
        .unwrap_or(80);
    let radius = matches
        .get_one::<String>("radius")
        .map(|s| s.parse().unwrap())
        .unwrap_or(20.0);

    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    log::info!("Allocating Driver and Building Mesh");

    let mut driver = Driver::new();

    let mesh = Mesh::new(
        Rectangle {
            size: [radius, radius],
            origin: [0.0, 0.0],
        },
        [points, points],
        3,
    );

    log::info!("Filling Seed Function");

    // Compute seed values.
    let mut seed = vec![0.0; mesh.node_count()].into_boxed_slice();
    driver.project(&mesh, &SeedProjection(1.0), &mut seed);
    driver.fill_boundary(&mesh, &OddBoundary, &mut seed);

    log::info!("Running Hyperbolic Relaxation");

    let mut psi = SystemVec::new(mesh.node_count());
    psi.field_mut(Scalar).fill(1.0);

    let mut solver = HyperRelaxSolver::new();
    solver.max_steps = 50000;
    solver.cfl = 0.1;
    solver.outgoing = OutgoingWave::Sommerfeld(1.0);
    solver.outgoing_order = OutgoingOrder::Fourth;
    solver.dampening = 0.4;

    solver.solve(
        &mut driver,
        &mesh,
        &PsiOperator { seed: &seed },
        &PsiBoundary,
        psi.as_mut_slice(),
    );
    driver.fill_boundary_system(&mesh, &PsiBoundary, psi.as_mut_slice());

    let mut hamiltonian = vec![0.0; mesh.node_count()].into_boxed_slice();

    driver.project(
        &mesh,
        &Hamiltonian {
            seed: &seed,
            psi: psi.field(Scalar),
        },
        &mut hamiltonian,
    );

    let mut model = Model::new(mesh.clone());
    model.attach_field("psi", psi.field(Scalar).iter().map(|&p| p - 1.0).collect());
    model.attach_field("seed", seed.to_vec());
    model.attach_field("hamiltonian", hamiltonian.to_vec());

    model
        .export_vtk(
            format!("idbrill").as_str(),
            PathBuf::from(format!("output/idbrill.vtu")),
        )
        .unwrap();

    // Write model data to file
    {
        let mut system = SystemVec::new(mesh.node_count());

        driver.project_system(
            &mesh,
            &InitialDataProjection {
                psi: psi.field(Scalar),
                seed: &seed,
            },
            system.as_mut_slice(),
        );

        let mut model = Model::new(mesh.clone());
        model.attach_system(system.as_slice());

        let mut file = File::create("output/idbrill.dat")?;
        file.write_all(ron::to_string(&model)?.as_bytes())?;
    }

    Ok(())
}
