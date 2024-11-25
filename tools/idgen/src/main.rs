use std::path::Path;

use anyhow::{anyhow, Context, Result};
use brill::solve_wth_garfinkle;
use config::{Brill, Domain, Instance, Solver};

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

    // Enumerate instances.
    for inst in config.instance.iter() {
        run_inst(
            &config.name,
            &absolute,
            config.order,
            &config.domain,
            &config.solver,
            inst,
        )?;
    }

    Ok(())
}

fn run_inst(
    name: &str,
    output_dir: &Path,
    order: usize,
    domain: &Domain,
    solver: &Solver,
    inst: &Instance,
) -> Result<()> {
    use aeon::prelude::*;

    let mut name = name.to_string();

    match inst {
        Instance::Brill(Brill {
            suffix,
            amplitude,
            sigma,
        }) => {
            // Find name with suffix.
            name.push_str(&suffix);

            log::info!("Running Instance: {}, Type: Brill Initial Data", name);
            log::info!(
                "A: {:.5e}, sigma_r: {:.5e}, sigma_z: {:.5e}",
                amplitude,
                sigma.0,
                sigma.1
            );
        }
    }

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

    match (inst, order) {
        (Instance::Brill(brill), 2) => {
            solve_wth_garfinkle(Order::<2>, &mut mesh, solver, brill)?;
        }
        (Instance::Brill(brill), 4) => {
            solve_wth_garfinkle(Order::<4>, &mut mesh, solver, brill)?;
        }
        (Instance::Brill(brill), 6) => {
            solve_wth_garfinkle(Order::<6>, &mut mesh, solver, brill)?;
        }
        _ => return Err(anyhow::anyhow!("Invalid initial data type and order")),
    }

    // while mesh.max_level() < domain.mesh.max_level {
    //     log::trace!("Adaptively refining mesh via wavelet criteria");
    // }

    Ok(())
}
