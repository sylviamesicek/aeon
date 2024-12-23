//! An executable for creating general initial data for numerical relativity simulations in 2D.

use std::process::ExitCode;

use aeon::{basis::RadiativeParams, prelude::*};
use anyhow::{anyhow, Context, Result};
use clap::{Arg, Command};
use sharedaxi::{
    import_from_path_arg, Brill, Constraint, Field, Fields, Gauge, IDConfig, Metric, Quadrant,
    Source,
};

mod garfinkle;

#[derive(Clone, Copy, SystemLabel)]
pub enum Initial {
    Conformal,
    Seed,
}

#[derive(Clone)]
pub struct InitialConditions;

impl Conditions<2> for InitialConditions {
    type System = InitialSystem;

    fn parity(&self, field: Initial, face: Face<2>) -> bool {
        match field {
            Initial::Conformal => [true, true][face.axis],
            Initial::Seed => [false, true][face.axis],
        }
    }

    fn radiative(&self, field: Initial, _position: [f64; 2], _spacing: f64) -> RadiativeParams {
        match field {
            Initial::Conformal => RadiativeParams::lightlike(1.0),
            Initial::Seed => RadiativeParams::lightlike(0.0),
        }
    }
}

fn initial_data() -> Result<()> {
    // Load configuration
    let matches = Command::new("idaxi")
        .about("A program for generating initial data for numerical relativity using hyperbolic relaxation.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("0.0.1")
        .arg(
            Arg::new("path")
                .help("Path of config file for generating initial data")
                .value_name("FILE")
                .required(true),
        ).get_matches();

    let config = import_from_path_arg::<IDConfig>(&matches)?;

    // Load header data and defaults
    let output = config
        .output_dir
        .clone()
        .unwrap_or_else(|| format!("{}_output", &config.name));

    // Compute log filter level.
    let level = config.logging.filter();

    // Build enviornment logger.
    env_logger::builder().filter_level(level).init();
    // Find currect working directory
    let dir = std::env::current_dir().context("Failed to find current working directory")?;
    let absolute = dir.join(&output);

    // Log Header data.
    log::info!("Simulation name: {}", &config.name);
    log::info!("Logging Level: {} ", level);
    log::info!(
        "Output Directory: {}",
        absolute
            .to_str()
            .ok_or(anyhow!("Failed to find absolute output directory"))?
    );

    anyhow::ensure!(
        config.domain.radius > 0.0 && config.domain.height > 0.0,
        "Domain must have positive non-zero radius and height"
    );

    log::info!(
        "Domain is {:.5} by {:.5}",
        config.domain.radius,
        config.domain.height
    );

    anyhow::ensure!(
        config.domain.cell.subdivisions >= 2 * config.domain.cell.padding,
        "Domain cell nodes must be >= 2 * padding"
    );

    anyhow::ensure!(
        config.domain.mesh.refine_global <= config.domain.mesh.max_level,
        "Mesh global refinements must be <= mesh max_level"
    );

    // Create output dir.
    std::fs::create_dir_all(&absolute)?;

    let solver = &config.solver;
    let domain = &config.domain;
    let sources = config.source.as_slice();
    let order = config.order;

    for source in sources {
        match source {
            Source::Brill(Brill { amplitude, sigma }) => {
                log::info!(
                    "Running Instance: {}, Type: Brill Initial Data",
                    config.name
                );
                log::info!(
                    "A: {:.5e}, sigma_r: {:.5e}, sigma_z: {:.5e}",
                    amplitude,
                    sigma.0,
                    sigma.1
                );
            }
        }
    }

    // Run brill simulation
    log::trace!("Building Mesh {} by {}", domain.radius, domain.height);

    let mut mesh = Mesh::new(
        Rectangle {
            size: [domain.radius, domain.height],
            origin: [0.0, 0.0],
        },
        domain.cell.subdivisions,
        domain.cell.padding,
    );

    log::trace!("Refining mesh globally {} times", domain.mesh.refine_global);

    for _ in 0..domain.mesh.refine_global {
        mesh.refine_global();
    }

    let mut transfer = SystemVec::default();
    let mut system = SystemVec::default();
    system.resize(mesh.num_nodes());

    loop {
        match order {
            2 => {
                garfinkle::solve_order(
                    Order::<2>,
                    &mut mesh,
                    solver,
                    sources,
                    system.as_mut_slice(),
                )?;
                mesh.fill_boundary(
                    Order::<2>,
                    Quadrant,
                    InitialConditions,
                    system.as_mut_slice(),
                );
            }
            4 => {
                garfinkle::solve_order(
                    Order::<4>,
                    &mut mesh,
                    solver,
                    sources,
                    system.as_mut_slice(),
                )?;
                mesh.fill_boundary(
                    Order::<4>,
                    Quadrant,
                    InitialConditions,
                    system.as_mut_slice(),
                );
            }
            6 => {
                garfinkle::solve_order(
                    Order::<6>,
                    &mut mesh,
                    solver,
                    sources,
                    system.as_mut_slice(),
                )?;
                mesh.fill_boundary(
                    Order::<6>,
                    Quadrant,
                    InitialConditions,
                    system.as_mut_slice(),
                );
            }
            _ => return Err(anyhow!("Invalid initial data type and order")),
        };

        if config.visualize_levels {
            let mut checkpoint = SystemCheckpoint::default();
            checkpoint.save_system(system.as_slice());

            mesh.export_vtu(
                absolute.join(format!("{}_level{}.vtu", &config.name, mesh.max_level())),
                &checkpoint,
                ExportVtuConfig {
                    title: config.name.clone(),
                    ghost: false,
                },
            )?;
        }

        if mesh.max_level() >= config.max_level || mesh.num_nodes() >= config.max_nodes {
            log::error!(
                "Failed to solve initial data, level: {}, nodes: {}",
                mesh.max_level(),
                mesh.num_nodes()
            );
            return Err(anyhow!("failed to refine within perscribed limits"));
        }

        mesh.flag_wavelets(
            config.order,
            0.0,
            config.max_error,
            Quadrant,
            system.as_slice(),
        );
        mesh.balance_flags();

        if mesh.requires_regridding() {
            log::trace!(
                "Regridding mesh from level {} to {}",
                mesh.max_level(),
                mesh.max_level() + 1
            );

            transfer.resize(mesh.num_nodes());
            transfer
                .contigious_mut()
                .clone_from_slice(system.contigious());
            mesh.regrid();
            system.resize(mesh.num_nodes());

            match order {
                2 => mesh.transfer_system(
                    Order::<2>,
                    Quadrant,
                    transfer.as_slice(),
                    system.as_mut_slice(),
                ),
                4 => mesh.transfer_system(
                    Order::<4>,
                    Quadrant,
                    transfer.as_slice(),
                    system.as_mut_slice(),
                ),
                6 => mesh.transfer_system(
                    Order::<6>,
                    Quadrant,
                    transfer.as_slice(),
                    system.as_mut_slice(),
                ),
                _ => {}
            };
        } else {
            log::trace!(
                "Sucessfully refined mesh to give accuracy: {:.5e}",
                config.max_error
            );
            break;
        }
    }

    let mut fields = SystemVec::with_length(mesh.num_nodes(), Fields);

    // Metric
    fields
        .field_mut(Field::Metric(Metric::Grr))
        .copy_from_slice(system.field(Initial::Conformal));
    fields
        .field_mut(Field::Metric(Metric::Gzz))
        .copy_from_slice(system.field(Initial::Conformal));
    fields.field_mut(Field::Metric(Metric::Grz)).fill(0.0);
    fields
        .field_mut(Field::Metric(Metric::S))
        .copy_from_slice(system.field(Initial::Seed));
    fields.field_mut(Field::Metric(Metric::Krr)).fill(0.0);
    fields.field_mut(Field::Metric(Metric::Kzz)).fill(0.0);
    fields.field_mut(Field::Metric(Metric::Krz)).fill(0.0);
    fields.field_mut(Field::Metric(Metric::Y)).fill(0.0);
    // Constraint
    fields
        .field_mut(Field::Constraint(Constraint::Theta))
        .fill(0.0);
    fields
        .field_mut(Field::Constraint(Constraint::Zr))
        .fill(0.0);
    fields
        .field_mut(Field::Constraint(Constraint::Zz))
        .fill(0.0);
    // Gauge
    fields.field_mut(Field::Gauge(Gauge::Lapse)).fill(1.0);
    fields.field_mut(Field::Gauge(Gauge::Shiftr)).fill(0.0);
    fields.field_mut(Field::Gauge(Gauge::Shiftz)).fill(0.0);

    if config.visualize_result {
        let mut checkpoint = SystemCheckpoint::default();
        checkpoint.save_system(fields.as_slice());

        mesh.export_vtu(
            absolute.join(format!("{}.vtu", &config.name)),
            &checkpoint,
            ExportVtuConfig {
                title: config.name.clone(),
                ghost: false,
            },
        )?;
    }

    let mut checkpoint = SystemCheckpoint::default();
    checkpoint.save_system(fields.as_slice());

    mesh.export_dat(absolute.join(format!("{}.dat", config.name)), &checkpoint)?;

    Ok(())
}

fn main() -> ExitCode {
    match initial_data() {
        Ok(_) => ExitCode::SUCCESS,
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
