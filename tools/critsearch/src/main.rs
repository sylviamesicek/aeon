use std::process::Command;

use anyhow::{anyhow, Context, Result};
use genparams::{Brill, InitialDataConfig, Solver, Source};

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
    log::info!("Running Critical Search: {}", &config.name);
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

    anyhow::ensure!(config.start < config.end);
    std::fs::create_dir_all(&absolute)?;
    std::fs::create_dir(&absolute.join("config"))?;

    let range = config.start..config.end;

    // loop {
    let diff = range.end - range.start;

    let step = diff / (config.subsearches - 1) as f64;

    let subsearches = (0..config.subsearches)
        .map(|i| range.start + step * i as f64)
        .collect::<Vec<_>>();

    for amplitude in subsearches.iter() {
        let config = InitialDataConfig {
            name: format!("{}_{:?}", config.name, amplitude),
            output_dir: Some(format!("{}/initial", output)),
            logging_level: None,
            order: 4,

            _logging_dir: None,
            _visualize_levels: false,
            _visualize_result: false,

            domain: config.domain.clone(),
            source: vec![Source::Brill(Brill {
                amplitude: *amplitude,
                sigma: (1.0, 1.0),
            })],

            solver: Solver {
                max_steps: 100000,
                cfl: 0.1,
                tolerance: 1e-6,
                dampening: 0.4,
            },
        };

        let config_path = absolute
            .join("config")
            .join(format!("initial_{:?}.toml", amplitude));

        std::fs::write(config_path, toml::to_string_pretty(&config)?)?;
    }

    for amplitude in subsearches.iter() {
        let config_path = format!("{}/config/initial_{:?}.toml", output, amplitude);
        Command::new("idgen").arg(config_path).output()?;
    }

    Ok(())
}
