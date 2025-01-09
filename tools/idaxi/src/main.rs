//! An executable for creating general initial data for numerical relativity simulations in 2D.

use std::{path::PathBuf, process::ExitCode};

use aeon::prelude::*;
use anyhow::{anyhow, Context, Result};
use clap::{Arg, Command};
use garfinkle::VisualizeConfig;
use sharedaxi::{import_from_path_arg, Fields, IDConfig, Quadrant, Source};

mod garfinkle;

fn initial_data() -> Result<()> {
    // Load configuration
    let matches = Command::new("idaxi")
        .about("A program for generating initial data for numerical relativity using hyperbolic relaxation.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("0.1.0")
        .arg(
            Arg::new("path")
                .help("Path of config file for generating initial data")
                .value_name("FILE")
                .required(true),
        ).get_matches();

    let config = import_from_path_arg::<IDConfig>(&matches)?;

    // Load header data and defaults
    let output = PathBuf::from(
        config
            .output_dir
            .clone()
            .unwrap_or_else(|| format!("{}_output", &config.name)),
    );

    // Compute log filter level.
    let level = config.logging.filter();

    // Build enviornment logger.
    env_logger::builder().filter_level(level).init();
    // Find currect working directory
    let dir = std::env::current_dir().context("Failed to find current working directory")?;
    let absolute = if output.is_absolute() {
        output
    } else {
        dir.join(output)
    };

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
        config.domain.cell.subdivisions >= 2 * config.domain.cell.ghost,
        "Domain cell nodes must be >= 2 * padding"
    );

    anyhow::ensure!(
        config.refine_global <= config.max_level,
        "Mesh global refinements must be <= mesh max_level"
    );

    anyhow::ensure!(config.visualize_stride >= 1, "Stride must be >= 1");

    // Create output dir.
    std::fs::create_dir_all(&absolute)?;

    let solver = &config.solver;
    let domain = &config.domain;
    let sources = config.source.as_slice();
    let order = config.order;

    for source in sources {
        match source {
            Source::Brill { amplitude, sigma } => {
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
            Source::ScalarField {
                amplitude,
                sigma,
                mass,
            } => {
                log::info!(
                    "Running Instance: {}, Type: Scalar Field Initial Data",
                    config.name
                );
                log::info!(
                    "A: {:.5e}, sigma_r: {:.5e}, sigma_z: {:.5e}, Mass {:.5e}",
                    amplitude,
                    sigma.0,
                    sigma.1,
                    mass
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
        domain.cell.ghost,
    );

    log::trace!("Refining mesh globally {} times", config.refine_global);

    for _ in 0..config.refine_global {
        mesh.refine_global();
    }

    // Build fields from sources.
    let fields = Fields {
        scalar_fields: sources
            .iter()
            .flat_map(|source| {
                if let Source::ScalarField { mass, .. } = source {
                    Some(*mass)
                } else {
                    None
                }
            })
            .collect(),
    };

    let mut transfer = SystemVec::new(fields.clone());
    let mut system = SystemVec::new(fields.clone());
    system.resize(mesh.num_nodes());

    loop {
        let visualize = if config.visualize_relax {
            Some(VisualizeConfig {
                path: &absolute,
                name: &config.name,
                every: config.visualize_every,
                stride: config.visualize_stride,
            })
        } else {
            None
        };
        match order {
            2 => {
                garfinkle::solve_order(
                    Order::<2>,
                    &mut mesh,
                    solver,
                    visualize,
                    sources,
                    system.as_mut_slice(),
                )?;
            }
            4 => {
                garfinkle::solve_order(
                    Order::<4>,
                    &mut mesh,
                    solver,
                    visualize,
                    sources,
                    system.as_mut_slice(),
                )?;
            }
            6 => {
                garfinkle::solve_order(
                    Order::<6>,
                    &mut mesh,
                    solver,
                    visualize,
                    sources,
                    system.as_mut_slice(),
                )?;
            }
            _ => return Err(anyhow!("Invalid initial data type and order")),
        };

        if config.visualize_levels {
            let mut checkpoint = SystemCheckpoint::default();
            checkpoint.save_system_ser(system.as_slice());

            mesh.export_vtu(
                absolute.join(format!("{}_level{}.vtu", &config.name, mesh.max_level())),
                &checkpoint,
                ExportVtuConfig {
                    title: config.name.clone(),
                    ghost: false,
                    stride: config.visualize_stride,
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

    let mut checkpoint = SystemCheckpoint::default();
    checkpoint.save_system_ser(system.as_slice());

    if config.visualize_result {
        mesh.export_vtu(
            absolute.join(format!("{}.vtu", &config.name)),
            &checkpoint,
            ExportVtuConfig {
                title: config.name.clone(),
                ghost: false,
                stride: config.visualize_stride,
            },
        )?;
    }

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
