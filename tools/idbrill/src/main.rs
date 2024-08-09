use aeon::{
    fd::{Discretization, ExportVtkConfig},
    prelude::*,
};

mod choptuik;
mod garfinkle;

pub const CHOPTUIK: bool = false;
pub const GHOST: bool = false;

/// Initial data in Rinne's hyperbolic variables.
#[derive(Clone, SystemLabel)]
pub enum Rinne {
    Conformal,
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
    type System = Rinne;

    fn parity(&self, field: Self::System, face: Face<2>) -> bool {
        match field {
            Rinne::Conformal => [true, true][face.axis],
            Rinne::Seed => [false, true][face.axis],
        }
    }

    fn radiative(&self, field: Self::System, _position: [f64; 2]) -> f64 {
        match field {
            Rinne::Conformal => 1.0,
            Rinne::Seed => 0.0,
        }
    }
}

pub const HAM_COND: SystemBC<Rinne, BoundaryConditions> =
    SystemBC::new(Rinne::Conformal, BoundaryConditions);

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
                .default_value("16"),
        )
        .arg(
            Arg::new("radius")
                .num_args(1)
                .short('r')
                .long("radius")
                .help("Size of grid along each axis")
                .value_name("RADIUS")
                .default_value("128.0"),
        )
        .get_matches();

    let points = matches
        .get_one::<String>("points")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(16);
    let radius = matches
        .get_one::<String>("radius")
        .map(|s| s.parse().unwrap())
        .unwrap_or(128.0);

    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    log::info!("Allocating Driver and Building Mesh Grid Size {points} Radius {radius}");

    let bounds = Rectangle {
        size: [radius, radius],
        origin: [0.0, 0.0],
    };

    let size = [16, 16];

    let mut mesh = Mesh::new(bounds, size, 3);

    for _ in 0..4 {
        let mut flags = vec![false; mesh.num_cells()];
        flags[0] = true;

        mesh.refine(&flags);
    }

    log::warn!("Min Spacing {}", mesh.min_spacing());

    if CHOPTUIK {
        std::fs::create_dir_all("output/choptuik")?;
    } else {
        std::fs::create_dir_all("output/garfinkle")?;
    }

    std::fs::write("output/mesh.txt", mesh.write_debug()).unwrap();

    println!("Num Blocks: {}", mesh.num_blocks());
    println!("Num Cells: {}", mesh.num_cells());

    let mut discrete = Discretization::new();
    discrete.set_mesh(&mesh);

    let mut rinne = SystemVec::with_length(mesh.num_nodes());
    let mut hamiltonian = vec![0.0; mesh.num_nodes()].into_boxed_slice();

    if CHOPTUIK {
        choptuik::solve(&mut discrete, 1.0, rinne.as_mut_slice(), &mut hamiltonian)?;
    } else {
        garfinkle::solve(&mut discrete, 1.0, rinne.as_mut_slice(), &mut hamiltonian)?;
    }

    discrete
        .order::<4>()
        .fill_boundary(BoundaryConditions, rinne.as_mut_slice());

    discrete
        .order::<4>()
        .fill_boundary(HAM_COND, hamiltonian.as_mut().into());

    let mut model = Model::empty();
    model.set_mesh(discrete.mesh());
    model.write_field("conformal", rinne.field(Rinne::Conformal).to_vec());
    model.write_field("seed", rinne.field(Rinne::Seed).to_vec());
    model.write_field("hamiltonian", hamiltonian.to_vec());

    if CHOPTUIK {
        model.export_vtk(
            format!("output/choptuik.vtu"),
            ExportVtkConfig {
                title: "idbrill".to_string(),
                ghost: crate::GHOST,
            },
        )?;
    } else {
        model.export_vtk(
            format!("output/garfinkle.vtu"),
            ExportVtkConfig {
                title: "idbrill".to_string(),
                ghost: crate::GHOST,
            },
        )?;
    }

    let mut model = Model::empty();
    model.set_mesh(discrete.mesh());
    model.write_system(rinne.as_slice());
    model.export_dat("output/idbrill.dat")?;

    Ok(())
}
