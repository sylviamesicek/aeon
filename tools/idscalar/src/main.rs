use aeon::{
    fd::{ExportVtuConfig, Mesh, SystemCondition},
    prelude::*,
};

mod garfinkle;

const GHOST: bool = false;

const ORDER: Order<4> = Order::<4>;

const SEED: f64 = 0.0;
const SCALAR: f64 = 1.0;
const MASS: f64 = 0.0;

/// Initial data in Rinne's hyperbolic variables.
#[derive(Clone, SystemLabel)]
pub enum Rinne {
    Conformal,
    Seed,
    Phi,
}

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

#[derive(Clone)]
pub struct RinneConditions;

impl Conditions<2> for RinneConditions {
    type System = Rinne;

    fn parity(&self, field: Self::System, face: Face<2>) -> bool {
        match field {
            Rinne::Conformal | Rinne::Phi => [true, true][face.axis],
            Rinne::Seed => [false, true][face.axis],
        }
    }

    fn radiative(&self, field: Self::System, _position: [f64; 2]) -> f64 {
        match field {
            Rinne::Conformal => 1.0,
            Rinne::Seed | Rinne::Phi => 0.0,
        }
    }
}

pub const HAMILTONIAN_CONDITIONS: ScalarConditions<SystemCondition<Rinne, RinneConditions>> =
    ScalarConditions::new(SystemCondition::new(Rinne::Conformal, RinneConditions));

use clap::{Arg, Command};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("idscalar")
        .about(
            "A program for generating initial data for scalar fields using hyperbolic relaxation.",
        )
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
                .default_value("16.0"),
        )
        .get_matches();

    let points = matches
        .get_one::<String>("points")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(16);
    let radius = matches
        .get_one::<String>("radius")
        .map(|s| s.parse().unwrap())
        .unwrap_or(16.0);

    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    log::info!("Allocating Driver and Building Mesh Grid Size {points} Radius {radius}");

    let bounds = Rectangle {
        size: [6.0 * 8.0; 2],
        origin: [0.0, 0.0],
    };

    let mut mesh = Mesh::new(bounds, 6, 3);

    for _ in 0..1 {
        mesh.refine_global();
    }

    std::fs::create_dir_all("output/idscalar")?;

    let mut transfer = SystemVec::new();

    for r in 0..12 {
        log::warn!("Min Spacing {}", mesh.min_spacing());
        log::warn!("Max Level {}", mesh.max_level());

        let mut debug = String::new();
        mesh.write_debug(&mut debug);

        std::fs::write("output/mesh.txt", debug).unwrap();

        log::info!("Num Blocks: {}", mesh.num_blocks());
        log::info!("Num Cells: {}", mesh.num_cells());
        log::info!("Num Nodes: {}", mesh.num_nodes());

        let mut rinne = SystemVec::with_length(mesh.num_nodes());
        let mut hamiltonian = vec![0.0; mesh.num_nodes()].into_boxed_slice();

        garfinkle::solve(
            &mut mesh,
            4000 * 2usize.pow(r as u32),
            rinne.as_mut_slice(),
            &mut hamiltonian,
        )?;

        mesh.fill_boundary(ORDER, Quadrant, RinneConditions, rinne.as_mut_slice());
        mesh.fill_boundary(
            ORDER,
            Quadrant,
            HAMILTONIAN_CONDITIONS,
            hamiltonian.as_mut().into(),
        );

        println!("Flagging wavelet");
        mesh.flag_wavelets(4, 0.0, 1e-6, Quadrant, rinne.as_slice());
        mesh.balance_flags();

        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_meta("MASS", &MASS.to_string());
        checkpoint.save_system(rinne.as_slice());
        checkpoint.save_field("hamiltonian", &hamiltonian);

        mesh.export_vtu(
            format!("output/idscalar/ellipticmasslesssec{r}.vtu"),
            &checkpoint,
            ExportVtuConfig {
                title: "idscalar".to_string(),
                ghost: crate::GHOST,
            },
        )?;

        mesh.export_dat(format!("output/idscalar/level{r}.dat"), &checkpoint)?;

        if mesh.requires_regridding() {
            transfer.resize(mesh.num_nodes());
            transfer
                .contigious_mut()
                .clone_from_slice(rinne.contigious());
            mesh.regrid();
            rinne.resize(mesh.num_nodes());
            mesh.transfer_system(ORDER, Quadrant, transfer.as_slice(), rinne.as_mut_slice());
        } else {
            log::info!("Sucessfully refined mesh to prescribed accuracy");
            mesh.export_dat("output/ellipticmasslesssec.dat", &checkpoint)?;
            break;
        }
    }

    Ok(())
}
