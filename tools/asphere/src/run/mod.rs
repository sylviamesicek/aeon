use crate::{CommandExt as _, parse_define_args, parse_invoke_arg};
use aeon_app::config::Transform as _;
use aeon_app::file;
use clap::{ArgMatches, Command};
use console::style;
use indicatif::MultiProgress;
use serde::{Deserialize, Serialize};

pub mod config;
pub mod evolve;
mod initial;

pub use config::Config;

/// Status of an indivdual run.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub enum Status {
    Disperse,
    Collapse,
}

#[derive(Clone, Copy)]
pub struct SimulationInfo {
    pub status: Status,
    pub mass: f64,
}

pub fn run(matches: &ArgMatches) -> eyre::Result<()> {
    let vars = parse_define_args(matches)?;
    let invoke = parse_invoke_arg(matches)?;
    // Find config file
    let config_run_file = std::env::current_dir()?.join(format!("{}.toml", invoke));
    // Load and apply variable transformation
    let config = file::import_toml::<Config>(&config_run_file)?;
    let config = config.transform(&vars)?;
    // Run simulation
    let _ = run_simulation(&config);

    Ok(())
}

pub fn run_simulation(config: &Config) -> eyre::Result<SimulationInfo> {
    // Check that there is only one source term.
    eyre::ensure!(
        config.sources.len() == 1,
        "asphere currently only supports one source term"
    );

    let _source = config.sources[0].clone();

    // Basic info dumping
    println!("Simulation: {}", style(&config.name).green());
    println!(
        "Output Directory: {}",
        style(config.directory()?.display()).green()
    );
    println!("Domain: {:.5}", config.domain.radius);
    println!("Sources...");
    for source in &config.sources {
        source.println();
    }

    // Solve for initial data
    let (mesh, system) = initial::initial_data(config)?;

    // Run evolution
    evolve::evolve_data(config, mesh, system)
}

pub struct Subrun {
    pub multi: MultiProgress,
    pub parameter: f64,
}

pub fn subrun(config: &Config, s: &Subrun) -> eyre::Result<SimulationInfo> {
    // Check that there is only one source term.
    eyre::ensure!(
        config.sources.len() == 1,
        "asphere currently only supports one source term"
    );

    // Solve for initial data
    let (mesh, system) = initial::initial_data(config)?;

    // Run evolution
    evolve::evolve_data_full(config, mesh, system, Some(s))
}

/// Extension trait for defining helper methods on `clap::Command`.
pub trait CommandExt {
    fn run_cmd(self) -> Self;
}

impl CommandExt for Command {
    fn run_cmd(self) -> Self {
        self.subcommand(
            Command::new("run")
                .about("Runs an evolution of spherically symmetric spacetime")
                .define_args()
                .invoke_arg(),
        )
    }
}

pub fn parse_run_cmd(matches: &ArgMatches) -> Option<&ArgMatches> {
    matches.subcommand_matches("run")
}
