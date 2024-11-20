use anyhow::{anyhow, Context, Result};
use config::Instance;

mod config;
mod shared;

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
        config.cell.subdivisions >= 2 * config.cell.padding,
        "Domain cell nodes must be >= 2 * padding"
    );

    // Enumerate instances.
    for inst in config.instance.iter() {
        match inst {
            Instance::Brill {
                suffix,
                amplitude,
                sigma,
            } => {
                let mut name = config.name.clone();
                name.push_str(suffix);

                log::info!("Running Instance: {}, Type: Brill Initial Data", name);
                log::info!(
                    "A = {:.5e}, sigma_r = {:.5e}, sigma_z = {:.5e}",
                    amplitude,
                    sigma.0,
                    sigma.1
                );
            }
        }
    }

    Ok(())
}
