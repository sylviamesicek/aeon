//! An executable for creating general initial data for numerical relativity simulations in 2D.
#![allow(unused_assignments)]

use aeon::{
    prelude::*,
    solver::{Integrator, Method},
};
use anyhow::{anyhow, Context, Result};
use clap::{Arg, Command};
use core::f64;
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

#[derive(Default)]
struct Diagnostics {
    times: Vec<f64>,
    masses: Vec<f64>,
    alphas: Vec<f64>,
}

impl Diagnostics {
    fn append(&mut self, time: f64, mass: f64, alpha: f64) {
        self.times.push(time);
        self.masses.push(mass);
        self.alphas.push(alpha);
    }

    fn flush(&self, config: RunConfig) -> Result<()> {
        let Some(serial_id) = config.serial_id else {
            return Ok(());
        };

        let file1 = format!("Mass-{}-{}", 4, serial_id);
        let file2 = format!("Al-{}-{}", 4, serial_id);
        let file3 = format!("Origin-{}.txt", serial_id);

        let mut data1 = String::new();
        let mut data2 = String::new();
        let mut data3 = String::new();

        for (&time, &mass) in self.times.iter().zip(self.masses.iter()) {
            writeln!(data1, "{} {}", time, mass)?;
        }

        for (&time, &alpha) in self.times.iter().zip(self.alphas.iter()) {
            writeln!(data2, "{} {}", time, alpha)?;
        }

        writeln!(data3, "TODO")?;

        std::fs::write(file1, data1)?;
        std::fs::write(file2, data2)?;
        std::fs::write(file3, data3)?;

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
            .ok_or(anyhow!("Failed to find absolute output directory"))?
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
    );
    mesh.set_face_boundary(Face::negative(0), BoundaryKind::Parity);
    mesh.set_face_boundary(Face::positive(0), BoundaryKind::Radiative);

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
        let l2_norm = mesh.l2_norm(system.as_slice());
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

        if mesh.max_level() >= MAX_LEVELS || mesh.num_nodes() >= MAX_NODES {
            log::error!(
                "Failed to solve initial data, level: {}, nodes: {}",
                mesh.max_level(),
                mesh.num_nodes()
            );
            break;
            // return Err(anyhow!("failed to refine within perscribed limits"));
        }

        mesh.flag_wavelets(4, 0.0, MAX_ERROR_TOLERANCE, system.as_slice());
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

    // mesh.export_dat(absolute.join(format!("{}.dat", config.name)), &checkpoint)?;

    // ****************************************
    // Run evolution

    let mut integrator = Integrator::new(Method::RK4KO6(DISSIPATION));
    let mut time = 0.0;
    let mut step = 0;

    let mut proper_time = 0.0;

    let mut save_step = 0;
    let mut steps_since_regrid = 0;
    let mut time_since_save = 0.0;

    let mass = find_mass(&mesh, system.as_slice());
    let alpha = mesh.bottom_left_value(system.field(Field::Lapse));
    diagnostics.append(time, mass, alpha);

    while proper_time < MAX_PROPER_TIME {
        assert!(system.len() == mesh.num_nodes());
        mesh.fill_boundary(Order::<4>, FieldConditions, system.as_mut_slice());

        // Check Norm
        let norm = mesh.l2_norm(system.as_slice());

        if norm.is_nan() || norm >= 1e60 {
            log::trace!("Evolution collapses, norm: {}", norm);
            return Err(anyhow!(
                "exceded max allotted steps for evolution: {}",
                step
            ));
        }

        if step >= MAX_TIME_STEPS {
            log::error!("Evolution exceded maximum allocated steps: {}", step);
            return Err(anyhow!(
                "exceded max allotted steps for evolution: {}",
                step
            ));
        }

        if mesh.num_nodes() >= MAX_NODES {
            log::error!(
                "Evolution exceded maximum allocated nodes: {}",
                mesh.num_nodes()
            );
            return Err(anyhow!(
                "exceded max allotted nodes for evolution: {}",
                mesh.num_nodes()
            ));
        }

        if mesh.max_level() >= MAX_LEVELS {
            log::trace!(
                "Evolution collapses, Reached maximum allowed level of refinement: {}",
                mesh.max_level()
            );
            return Err(anyhow!(
                "reached maximum allowed level of refinement: {}",
                mesh.max_level()
            ));
        }

        let h = mesh.min_spacing() * CFL;

        if steps_since_regrid > REGRID_FLAG_INTERVAL {
            steps_since_regrid = 0;

            mesh.flag_wavelets(
                4,
                MIN_ERROR_TOLERANCE,
                MAX_ERROR_TOLERANCE,
                system.as_slice(),
            );
            mesh.balance_flags();

            // let num_refine = mesh.num_refine_cells();
            // let num_coarsen = mesh.num_coarsen_cells();
            mesh.regrid();

            // log::trace!(
            //     "Regrided Mesh at time: {time:.5}, Max Level {}, {} R, {} C",
            //     mesh.max_level(),
            //     num_refine,
            //     num_coarsen,
            // );

            log::trace!(
                "Regrided Mesh at time: {time:.5}, Max Level {}, Num Nodes {}",
                mesh.max_level(),
                mesh.num_nodes(),
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

        let mass = find_mass(&mesh, system.as_slice());
        let alpha = mesh.bottom_left_value(system.field(Field::Lapse));
        diagnostics.append(time, mass, alpha);

        step += 1;
        steps_since_regrid += 1;

        time += h;
        time_since_save += h;

        proper_time += h * alpha;
    }

    Ok(())
}

// Main function that can return an error
fn try_main() -> Result<()> {
    // Load configuration
    let matches = Command::new("evsphere")
        .about("A program for generating initial data for numerical relativity using hyperbolic relaxation.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        // .arg(
        //     Arg::new("amplitude")
        //         .short('a')
        //         .long("amplitude")
        //         .default_value("1.0")
        //         .value_name("FLOAT")
        //         .global(false)
        // )
        // .arg(
        //     Arg::new("visualize")
        //         .short('v')
        //         .long("visualize")
        //         .num_args(0)
        //         .help("Output visualizations during evolution")
        //         .global(false)
        // )
        // .arg(
        //     Arg::new("output").required(true)
        //         .help("Output directory")
        //         .value_name("DIR")
        //         .global(false)
        //     )
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
            .ok_or(anyhow!("Failed to find amplitude positional argument"))?
            .parse::<f64>()
            .map_err(|_| anyhow!("Failed to parse amplitude as float"))?
            .clone();

        config.serial_id = Some(
            matches
                .get_one::<String>("ser")
                .ok_or(anyhow!("Failed to find serial_id positional argument"))?
                .parse::<usize>()
                .map_err(|_| anyhow!("Failed to parse serial_id as int"))?
                .clone(),
        );

        config.absolute =
            std::env::current_dir().context("Failed to find current working directory")?;
    } else {
        config.amplitude = matches
            .get_one::<String>("amplitude")
            .ok_or(anyhow!("Could not find amplitude argument"))?
            .parse::<f64>()
            .map_err(|_| anyhow!("Failed parse amplitude argument"))?
            .clone();

        let output = PathBuf::from(
            matches
                .get_one::<String>("output")
                .ok_or(anyhow!("Failed parse path argument"))?
                .clone(),
        );

        if output.is_absolute() {
            config.absolute = output
        } else {
            config.absolute = std::env::current_dir()
                .context("Failed to find current working directory")?
                .join(output);
        }

        config.visualize = matches.contains_id("visualize");
    }

    // Build enviornment logger.
    env_logger::builder()
        .filter_level(log::LevelFilter::Trace)
        .init();

    // Diagnostic object
    let mut diagnostics = Diagnostics::default();
    // Run simulation
    let result = run(config.clone(), &mut diagnostics);
    // Write diagnostics to file
    diagnostics.flush(config)?;
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
