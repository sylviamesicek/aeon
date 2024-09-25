use aeon::{
    fd::{ExportVtkConfig, Mesh, SystemCondition},
    prelude::*,
};

mod choptuik;
mod garfinkle;

const CHOPTUIK: bool = false;
const GHOST: bool = true;

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
        if face.side == false {
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

    fn radiative(&self, field: Self::System, _position: [f64; 2]) -> f64 {
        match field {
            Rinne::Conformal => 1.0,
            Rinne::Seed => 0.0,
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
        size: [radius, radius],
        origin: [0.0, 0.0],
    };

    let mut mesh = Mesh::new(bounds, 4, 2);

    for _ in 0..1 {
        let flags = vec![true; mesh.num_cells()];
        mesh.refine(&flags);
    }

    // for _ in 0..4 {
    //     let mut flags = vec![false; mesh.num_cells()];
    //     flags[0] = true;
    //     mesh.refine(&flags);
    // }

    if CHOPTUIK {
        std::fs::create_dir_all("output/choptuik")?;
    } else {
        std::fs::create_dir_all("output/garfinkle")?;
    }

    for r in 0..3 {
        // let flags = vec![true; mesh.num_cells()];
        // mesh.refine(&flags);

        log::warn!("Min Spacing {}", mesh.min_spacing());

        let mut debug = String::new();
        mesh.write_debug(&mut debug);

        std::fs::write("output/mesh.txt", debug).unwrap();

        println!("Num Blocks: {}", mesh.num_blocks());
        println!("Num Cells: {}", mesh.num_cells());

        let mut rinne = SystemVec::with_length(mesh.num_nodes());
        let mut hamiltonian = vec![0.0; mesh.num_nodes()].into_boxed_slice();

        garfinkle::solve(
            &mut mesh,
            1.0,
            1000 * 2usize.pow(r as u32),
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

        let mut flags = vec![false; mesh.num_cells()];
        println!("Flagging wavelet");
        mesh.wavelet(Quadrant, 1e-5, rinne.as_slice(), &mut flags);
        mesh.tree().balance_refine_flags(&mut flags);

        println!("{flags:?}");

        let mut indices = vec![0i64; mesh.num_nodes()];
        mesh.interface_indices(2, &mut indices);

        let mut block_indices = vec![0i64; mesh.num_nodes()];
        mesh.block_interface_indices(2, &mut block_indices);

        let mut blocks = vec![0; mesh.num_nodes()];
        mesh.block_debug(&mut blocks);

        let mut systems = SystemCheckpoint::default();
        systems.save_system(rinne.as_slice());
        systems.save_field("hamiltonian", &hamiltonian);
        systems.save_int_field("interface", &indices);
        systems.save_int_field("block_interface", &block_indices);
        systems.save_int_field("blocks", &blocks);

        mesh.export_vtk(
            format!("output/garfinkle.vtu"),
            ExportVtkConfig {
                title: "idbrill".to_string(),
                ghost: crate::GHOST,
                systems: systems.clone(),
            },
        )?;

        mesh.refine(&flags);
    }

    // let mut debug = String::new();
    // mesh.write_debug(&mut debug);

    // std::fs::write("output/mesh.txt", debug).unwrap();

    // println!("Num Blocks: {}", mesh.num_blocks());
    // println!("Num Cells: {}", mesh.num_cells());

    // let mut rinne = SystemVec::with_length(mesh.num_nodes());
    // let mut hamiltonian = vec![0.0; mesh.num_nodes()].into_boxed_slice();

    // if CHOPTUIK {
    //     choptuik::solve(&mut mesh, 4.0, rinne.as_mut_slice(), &mut hamiltonian)?;
    // } else {
    //     garfinkle::solve(
    //         &mut mesh,
    //         4.0,
    //         50000,
    //         rinne.as_mut_slice(),
    //         &mut hamiltonian,
    //     )?;
    // }

    // mesh.fill_boundary(ORDER, Quadrant, RinneConditions, rinne.as_mut_slice());
    // mesh.fill_boundary(
    //     ORDER,
    //     Quadrant,
    //     HAMILTONIAN_CONDITIONS,
    //     hamiltonian.as_mut().into(),
    // );

    // let mut systems = SystemCheckpoint::default();
    // systems.save_system(rinne.as_slice());
    // systems.save_field("hamiltonian", &hamiltonian);

    // if CHOPTUIK {
    //     mesh.export_vtk(
    //         format!("output/choptuik.vtu"),
    //         ExportVtkConfig {
    //             title: "idbrill".to_string(),
    //             ghost: crate::GHOST,
    //             systems: systems.clone(),
    //         },
    //     )?;
    // } else {
    //     mesh.export_vtk(
    //         format!("output/garfinkle.vtu"),
    //         ExportVtkConfig {
    //             title: "idbrill".to_string(),
    //             ghost: crate::GHOST,
    //             systems: systems.clone(),
    //         },
    //     )?;
    // }

    // log::warn!("Min Spacing {}", mesh.min_spacing());
    // mesh.export_dat("output/idbrill_amr.dat", &systems)?;

    Ok(())
}
