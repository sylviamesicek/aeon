//! An executable for creating general initial data for numerical relativity simulations in 2D.

use aeon::prelude::*;
use anyhow::{anyhow, Context, Result};
use brill::solve_wth_garfinkle;
use genparams::{Brill, Source};

mod brill;
mod config;

fn main() -> Result<()> {
    // Load configuration
    let config = config::configure()?;

    // Load header data and defaults
    let log_level = config.logging_level.unwrap_or(1);
    let output = config
        .output_dir
        .clone()
        .unwrap_or_else(|| format!("{}_output", &config.name));

    // Compute log filter level.
    let level = match log_level {
        0 => log::LevelFilter::Off,
        1 => log::LevelFilter::Warn,
        2 => log::LevelFilter::Info,
        3 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };

    // Build enviornment logger.
    env_logger::builder().filter_level(level).init();
    // Find currect working directory
    let dir = std::env::current_dir().context("Failed to find current working directory")?;
    let absolute = dir.join(&output);

    // Log Header data.
    log::info!("Simulation name: {}", &config.name);
    log::info!("Logging Level: {} ", log_level);
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

    anyhow::ensure!(config.source.len() == 1);

    let solver = &config.solver;
    let domain = &config.domain;
    let source = &config.source[0];
    let order = config.order;

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

    let result = match (source, order) {
        (Source::Brill(brill), 2) => solve_wth_garfinkle(Order::<2>, &mut mesh, solver, brill)?,
        (Source::Brill(brill), 4) => solve_wth_garfinkle(Order::<4>, &mut mesh, solver, brill)?,
        (Source::Brill(brill), 6) => solve_wth_garfinkle(Order::<6>, &mut mesh, solver, brill)?,
        _ => return Err(anyhow::anyhow!("Invalid initial data type and order")),
    };

    let mut checkpoint = SystemCheckpoint::default();
    checkpoint.save_system(result.as_slice());

    mesh.export_dat(absolute.join(format!("{}.dat", config.name)), &checkpoint)?;

    Ok(())
}
