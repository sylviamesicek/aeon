use aeon::{
    basis::RadiativeParams,
    fd::{ExportVtuConfig, Mesh, SystemCondition},
    prelude::*,
};

mod choptuik;
mod garfinkle;

const CHOPTUIK: bool = false;
const GHOST: bool = false;

const ORDER: Order<4> = Order::<4>;
/// Initial data in Rinne's hyperbolic variables.
#[derive(Clone, SystemLabel)]
pub enum Rinne {
    Conformal,
    Seed,
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
            Rinne::Conformal => [true, true][face.axis],
            Rinne::Seed => [false, true][face.axis],
        }
    }

    fn radiative(
        &self,
        field: Self::System,
        _position: [f64; 2],
        _spacing: f64,
    ) -> RadiativeParams {
        match field {
            Rinne::Conformal => RadiativeParams::lightlike(1.0),
            Rinne::Seed => RadiativeParams::lightlike(0.0),
        }
    }
}

pub const HAMILTONIAN_CONDITIONS: ScalarConditions<SystemCondition<Rinne, RinneConditions>> =
    ScalarConditions::new(SystemCondition::new(Rinne::Conformal, RinneConditions));

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

    std::fs::create_dir_all("output/idbrill")?;

    let mut transfer = SystemVec::new();

    for r in 0..12 {
        let mut debug = String::new();
        mesh.write_debug(&mut debug);

        std::fs::write("output/mesh.txt", debug).unwrap();

        log::info!("Min Spacing {}", mesh.min_spacing());
        log::info!("Num Blocks: {}", mesh.num_blocks());
        log::info!("Num Cells: {}", mesh.num_cells());
        log::info!("Num Nodes: {}", mesh.num_nodes());

        let mut rinne = SystemVec::with_length(mesh.num_nodes());
        let mut hamiltonian = vec![0.0; mesh.num_nodes()].into_boxed_slice();

        if CHOPTUIK {
            choptuik::solve(
                &mut mesh,
                -4.0 / 2.0,
                10_000_000,
                rinne.as_mut_slice(),
                &mut hamiltonian,
            )?;
        } else {
            garfinkle::solve(
                &mut mesh,
                -4.0 / 2.0,
                10_000_000,
                rinne.as_mut_slice(),
                &mut hamiltonian,
            )?;
        }

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

        let mut interfaces = vec![0i64; mesh.num_nodes()];
        mesh.interface_index_debug(2, &mut interfaces);

        let mut interface_neighbors = vec![0i64; mesh.num_nodes()];
        mesh.interface_neighbor_debug(2, &mut interface_neighbors);

        let mut blocks = vec![0; mesh.num_nodes()];
        mesh.block_debug(&mut blocks);

        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_system(rinne.as_slice());
        checkpoint.save_field("hamiltonian", &hamiltonian);

        checkpoint.save_int_field("interface", &interfaces);
        checkpoint.save_int_field("interface_neighbors", &interface_neighbors);
        checkpoint.save_int_field("blocks", &blocks);

        if CHOPTUIK {
            mesh.export_vtu(
                format!("output/idbrill/choptuik{r}.vtu"),
                &checkpoint,
                ExportVtuConfig {
                    title: "idbrill".to_string(),
                    ghost: crate::GHOST,
                },
            )?;
        } else {
            mesh.export_vtu(
                format!("output/idbrill/garfinkle{r}.vtu"),
                &checkpoint,
                ExportVtuConfig {
                    title: "idbrill".to_string(),
                    ghost: crate::GHOST,
                },
            )?;
        }

        mesh.export_dat(format!("output/idbrill/level{r}.dat"), &checkpoint)?;

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
            mesh.export_dat("output/negcritical.dat", &checkpoint)?;
            break;
        }
    }

    Ok(())
}
