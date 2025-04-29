//! An executable for creating general initial data for numerical relativity simulations in 2D.
#![allow(unused_assignments)]

use aeon::{
    prelude::*,
    solver::{Integrator, Method},
};
use clap::{Arg, Command};
use core::f64;
use eyre::{Context, Result, eyre};
use std::fmt::Write as _;
use std::{path::PathBuf, process::ExitCode};

mod config;
mod system;

use config::*;
use system::*;

#[derive(Clone)]
struct RunConfig {
    absolute: PathBuf,
    amplitude: f64,
    serial_id: Option<usize>,
    visualize: bool,
}

struct Snapshot {
    mass: f64,
    alpha: f64,
    phi: f64,
    level: usize,
    nodes: usize,
}

#[derive(Default)]
struct Diagnostics {
    times: Vec<f64>,
    data: Vec<Snapshot>,
}

impl Diagnostics {
    fn append(&mut self, time: f64, data: Snapshot) {
        self.times.push(time);
        self.data.push(data);
    }

    fn flush(&self, config: RunConfig) -> Result<()> {
        let Some(serial_id) = config.serial_id else {
            return Ok(());
        };

        let file1 = format!("Mass-{}-{}", 4, serial_id);
        let file2 = format!("Al-{}-{}", 4, serial_id);
        let file3 = format!("Level-{}-{}", 4, serial_id);
        let file4 = format!("Origin-{}.txt", serial_id);

        let mut data1 = String::new();
        let mut data2 = String::new();
        let mut data3 = String::new();
        let mut data4 = String::new();

        for (&time, data) in self.times.iter().zip(self.data.iter()) {
            writeln!(data1, "{} {}", time, data.mass)?;
            writeln!(data2, "{} {}", time, data.alpha)?;
            writeln!(data3, "{} {} {}", time, data.level, data.nodes)?;
            writeln!(
                data4,
                "{} {} {} {} {} {}",
                time, 0.0, data.alpha, data.alpha, data.phi, 0.0
            )?;
        }

        std::fs::write(file1, data1)?;
        std::fs::write(file2, data2)?;
        std::fs::write(file3, data3)?;
        std::fs::write(file4, data4)?;

        Ok(())
    }
}

fn run(config: RunConfig, diagnostics: &mut Diagnostics) -> Result<()> {
    // Unpack config
    let RunConfig {
        absolute,
        amplitude,
        serial_id,
        visualize,
    } = config;
    // Log Header data.
    log::info!(
        "Output Directory: {}",
        absolute
            .to_str()
            .ok_or(eyre!("Failed to find absolute output directory"))?
    );
    // As well as general information about the run
    if let Some(id) = serial_id {
        log::info!(
            "Amplitude {:.6}, sigma: {:.6} Serial {}",
            amplitude,
            SIGMA,
            id
        );
    } else {
        log::info!("Amplitude {:.6}, sigma: {:.6}", amplitude, SIGMA);
    }

    // Create output dir.
    std::fs::create_dir_all(&absolute)?;

    // Run brill simulation
    log::trace!(
        "Building Mesh with Radius {}, Cell Width {}, Ghost Nodes {}",
        RADIUS,
        CELL_WIDTH,
        GHOST
    );

    let mut mesh = Mesh::new(
        Rectangle {
            size: [RADIUS],
            origin: [0.0],
        },
        CELL_WIDTH,
        GHOST,
        FaceArray::from_sides([BoundaryClass::Ghost], [BoundaryClass::OneSided]),
    );

    log::trace!("Refining mesh globally {} times", REFINE_GLOBAL);

    for _ in 0..REFINE_GLOBAL {
        mesh.refine_global();
    }

    let mut system = SystemVec::new(Fields);

    loop {
        system.resize(mesh.num_nodes());

        // Set initial data for scalar field.
        let scalar_field = generate_initial_scalar_field(&mut mesh, amplitude);

        // Fill system using scalar field.
        mesh.evaluate(
            4,
            InitialData,
            (&scalar_field).into(),
            system.as_mut_slice(),
        );

        // Solve for conformal and lapse
        solve_constraints(&mut mesh, system.as_mut_slice());
        // Compute norm
        let l2_norm = mesh.l2_norm_system(system.as_slice());
        log::info!("Scalar Field Norm {}", l2_norm);

        // Save visualization
        if visualize {
            let mut checkpoint = SystemCheckpoint::default();
            checkpoint.save_system_ser(system.as_slice());
            mesh.export_vtu(
                absolute.join(format!("initial{}.vtu", mesh.max_level())),
                &checkpoint,
                ExportVtuConfig {
                    title: "Massless Scalar Field Initial Data".to_string(),
                    ghost: false,
                    stride: 1,
                },
            )?;
        }

        if mesh.num_nodes() >= MAX_NODES {
            log::error!(
                "Failed to solve initial data, level: {}, nodes: {}",
                mesh.max_level(),
                mesh.num_nodes()
            );
            return Err(eyre!("failed to refine within perscribed limits"));
        }

        mesh.flag_wavelets(4, 0.0, MAX_ERROR_TOLERANCE, system.as_slice());
        mesh.limit_level_range_flags(1, MAX_LEVELS);
        mesh.balance_flags();

        if !mesh.requires_regridding() {
            log::trace!(
                "Sucessfully refined mesh to give accuracy: {:.5e}",
                MAX_ERROR_TOLERANCE
            );
            break;
        } else {
            log::trace!(
                "Regridding mesh from level {} to {}",
                mesh.max_level(),
                mesh.max_level() + 1
            );

            mesh.regrid();
        }
    }

    if visualize {
        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_system_ser(system.as_slice());

        mesh.export_vtu(
            absolute.join("initial.vtu"),
            &checkpoint,
            ExportVtuConfig {
                title: "Massless Scalar Field Initial".to_string(),
                ghost: false,
                stride: 1,
            },
        )?;
    }

    // ****************************************
    // Run evolution

    let mut integrator = Integrator::new(Method::RK4KO6(DISSIPATION));
    let mut time = 0.0;
    let mut step = 0;

    let mut proper_time = 0.0;

    let mut save_step = 0;
    let mut steps_since_regrid = 0;
    let mut time_since_save = 0.0;

    diagnostics.append(
        proper_time,
        Snapshot {
            mass: find_mass(&mesh, system.as_slice()),
            alpha: mesh.bottom_left_value(system.field(Field::Lapse)),
            phi: mesh.bottom_left_value(system.field(Field::Phi)),
            level: mesh.max_level(),
            nodes: mesh.num_nodes(),
        },
    );

    while proper_time < MAX_PROPER_TIME {
        assert!(system.len() == mesh.num_nodes());
        mesh.fill_boundary(Order::<4>, FieldConditions, system.as_mut_slice());

        // Check Norm
        let norm = mesh.l2_norm_system(system.as_slice());

        if norm.is_nan() || norm >= 1e60 {
            log::trace!("Evolution collapses, norm: {}", norm);
            return Err(eyre!("exceded max allotted steps for evolution: {}", step));
        }

        if step >= MAX_TIME_STEPS {
            log::error!("Evolution exceded maximum allocated steps: {}", step);
            return Err(eyre!("exceded max allotted steps for evolution: {}", step));
        }

        if mesh.num_nodes() >= MAX_NODES {
            log::error!(
                "Evolution exceded maximum allocated nodes: {}",
                mesh.num_nodes()
            );
            return Err(eyre!(
                "exceded max allotted nodes for evolution: {}",
                mesh.num_nodes()
            ));
        }

        // if mesh.max_level() >= MAX_LEVELS {
        //     log::trace!(
        //         "Evolution collapses, Reached maximum allowed level of refinement: {}",
        //         mesh.max_level()
        //     );
        //     return Err(anyhow!(
        //         "reached maximum allowed level of refinement: {}",
        //         mesh.max_level()
        //     ));
        // }

        let h = mesh.min_spacing() * CFL;

        if steps_since_regrid > REGRID_FLAG_INTERVAL {
            steps_since_regrid = 0;

            mesh.flag_wavelets(
                4,
                MIN_ERROR_TOLERANCE,
                MAX_ERROR_TOLERANCE,
                system.as_slice(),
            );
            mesh.limit_level_range_flags(1, MAX_LEVELS);
            mesh.balance_flags();
            mesh.limit_level_range_flags(1, MAX_LEVELS);

            mesh.regrid();

            log::trace!(
                "Regrided Mesh at time: {proper_time:.5}, Max Level {}, Num Nodes {}, Step: {}",
                mesh.max_level(),
                mesh.num_nodes(),
                step,
            );

            // Copy system into tmp scratch space (provieded by dissipation).
            let scratch = integrator.scratch(system.contigious().len());
            scratch.copy_from_slice(system.contigious());
            system.resize(mesh.num_nodes());
            mesh.transfer_system(
                Order::<4>,
                SystemSlice::from_contiguous(&scratch, &Fields),
                system.as_mut_slice(),
            );

            continue;
        }

        if time_since_save >= SAVE_INTERVAL && visualize {
            time_since_save -= SAVE_INTERVAL;

            log::trace!(
                "Saving Checkpoint {save_step}, Time: {time:.5}, Dilated Time: {proper_time:.5}, Step: {step}, Norm: {norm:.5e}, Nodes: {}",
                mesh.num_nodes()
            );

            // Output current system to disk
            let mut systems = SystemCheckpoint::default();
            systems.save_system_ser(system.as_slice());

            mesh.export_vtu(
                absolute.join(format!("evolve_{save_step}.vtu")),
                &systems,
                ExportVtuConfig {
                    title: "Masslesss Scalar Field Evolution".to_string(),
                    ghost: false,
                    stride: 1,
                },
            )?;

            save_step += 1;
        }

        // Compute step
        integrator.step(
            &mut mesh,
            Order::<4>,
            FieldConditions,
            TimeDerivs,
            h,
            system.as_mut_slice(),
        );

        let alpha = mesh.bottom_left_value(system.field(Field::Lapse));
        if step % DIAGNOSTIC_STRIDE == 0 {
            diagnostics.append(
                proper_time,
                Snapshot {
                    mass: find_mass(&mesh, system.as_slice()),
                    alpha,
                    phi: mesh.bottom_left_value(system.field(Field::Phi)),
                    level: mesh.max_level(),
                    nodes: mesh.num_nodes(),
                },
            );
        }

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;

        proper_time += h * alpha;

        let norm = mesh.l2_norm_system(system.as_slice());

        if norm.is_nan() || norm >= 1e60 || alpha.is_nan() {
            log::trace!("Evolution collapses after step, norm: {}", norm);
            return Err(eyre!("exceded max allotted steps for evolution: {}", step));
        }
    }

    Ok(())
}

// Main function that can return an error
fn try_main() -> Result<()> {
    // Load configuration
    let matches = Command::new("evsphere")
        .about("A program for generating initial data for numerical relativity using hyperbolic relaxation.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .subcommand(Command::new("vis")
            .arg(
                Arg::new("amp")
                    .short('a')
                    .long("amp")
                    .default_value("1.0")
                    .value_name("FLOAT")
            )
            .arg(
                Arg::new("output")
                    .required(true)
                    .help("Output directory")
                    .value_name("DIR")
                )
            )
        .subcommand(
            Command::new("cole")
                .arg(
                    Arg::new("amp")
                        .value_name("FLOAT")
                        .required(true)
                        .help("Amplitude of massless scalar field to simulate")
                )
                .arg(
                    Arg::new("ser")
                        .value_name("INT")
                        .required(true)
                        .help("Serialization number for massless scalar field data")
                )
        )
        .version("0.1.0")
        .get_matches();

    let mut config = RunConfig {
        absolute: PathBuf::new(),
        amplitude: 0.0,
        serial_id: None,
        visualize: false,
    };

    if let Some(matches) = matches.subcommand_matches("cole") {
        config.amplitude = matches
            .get_one::<String>("amp")
            .ok_or(eyre!("Failed to find amplitude positional argument"))?
            .parse::<f64>()
            .map_err(|_| eyre!("Failed to parse amplitude as float"))?
            .clone();

        config.serial_id = Some(
            matches
                .get_one::<String>("ser")
                .ok_or(eyre!("Failed to find serial_id positional argument"))?
                .parse::<usize>()
                .map_err(|_| eyre!("Failed to parse serial_id as int"))?
                .clone(),
        );

        config.absolute =
            std::env::current_dir().context("Failed to find current working directory")?;
    } else if let Some(matches) = matches.subcommand_matches("vis") {
        config.amplitude = matches
            .get_one::<String>("amp")
            .ok_or(eyre!("Could not find amplitude argument"))?
            .parse::<f64>()
            .map_err(|_| eyre!("Failed parse amplitude argument"))?
            .clone();

        let output = PathBuf::from(
            matches
                .get_one::<String>("output")
                .ok_or(eyre!("Failed parse path argument"))?
                .clone(),
        );

        if output.is_absolute() {
            config.absolute = output
        } else {
            config.absolute = std::env::current_dir()
                .context("Failed to find current working directory")?
                .join(output);
        }

        config.visualize = true;
    }

    // Build enviornment logger.
    // env_logger::builder()
    //     .filter_level(log::LevelFilter::Trace)
    //     .init();

    // Diagnostic object
    let mut diagnostics = Diagnostics::default();
    // Run simulation
    let result = run(config.clone(), &mut diagnostics);
    // Write diagnostics to file
    diagnostics.flush(config.clone())?;

    match result {
        Ok(_) => eprintln!("A: {:.15} Disperses", config.amplitude),
        Err(_) => eprintln!("A: {:.15} Collapses", config.amplitude),
    }

    // Bubble up result
    result
}

fn main() -> ExitCode {
    match try_main() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            if log::log_enabled!(log::Level::Error) {
                log::error!("{:?}", err);
            } else {
                eprintln!("{:?}", err);
            }
            ExitCode::FAILURE
        }
    }
}
