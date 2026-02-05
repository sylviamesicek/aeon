//! Submodule representing default `run` command (that is, `aaxi`) invoked without
//! any additional subcommands.

use crate::{CommandExt as _, parse_define_args, parse_invoke_arg};
use aeon_app::file;
use clap::{ArgMatches, Command};
use console::{Term, style};

mod config;
mod evolve;
mod history;
mod initial;
mod interval;
mod status;

pub use config::Config;
use eyre::Context;
pub use status::Status;

pub fn run(matches: &ArgMatches) -> eyre::Result<()> {
    let vars = parse_define_args(matches)?;
    let invoke = parse_invoke_arg(matches)?;
    // Find config file
    let config_run_file = std::env::current_dir()?.join(format!("{}.toml", invoke));
    // Load and apply variable transformation
    let config = file::import_toml::<Config>(&config_run_file)
        .with_context(|| format!("failed to find run config file: {:?}", config_run_file))?;
    let config = config.transform(&vars)?;
    // Run simulation
    let _ = run_simulation(&config);

    Ok(())
}

pub fn run_simulation(config: &Config) -> eyre::Result<Status> {
    let output_path = config.output_dir()?;
    // Ensure path exists.
    std::fs::create_dir_all(&output_path)?;

    // **********************************

    // Terminal output setup
    let term = Term::stdout();
    term.set_title("Axisymmetric Simulation");
    // Basic info dumping
    log::info!("Simulation: {}", style(&config.name).green());
    log::info!("Output Directory: {}", style(output_path.display()).green());
    log::info!(
        "Domain: {:.5} x {:.5}",
        config.domain.radius,
        config.domain.height
    );
    log::info!("Sources:");
    for source in &config.sources {
        log::info!("- {}", source.description());
    }

    // ***********************************
    // Validation

    config.validate()?;

    // ***********************************
    // Initial data

    let (mesh, system) = initial::initial_data(&config)?;

    // ************************************
    // Evolve data forwards

    let status = evolve::evolve_data(&config, mesh, system)?;

    Ok(status)
}

// ******************************
// Helpers **********************
// ******************************

/// Extension trait for defining helper methods on `clap::Command`.
pub trait CommandExt {
    fn run_cmd(self) -> Self;
}

impl CommandExt for Command {
    fn run_cmd(self) -> Self {
        self.subcommand(
            Command::new("run")
                .about("Runs an evolution of axisymmetric spacetime")
                .define_args()
                .invoke_arg(),
        )
    }
}

pub fn parse_run_cmd(matches: &ArgMatches) -> Option<&ArgMatches> {
    matches.subcommand_matches("run")
}
