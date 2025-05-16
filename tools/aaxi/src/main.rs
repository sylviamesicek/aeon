use clap::{Arg, ArgMatches, Command, arg, value_parser};
use console::{Term, style};
use eyre::WrapErr;
use serde::de::DeserializeOwned;
use std::{
    collections::HashMap,
    num::ParseFloatError,
    path::{Path, PathBuf},
};

mod config;
mod misc;
mod rinne;
mod searches;
mod transform;

use config::*;
use rinne::*;

use transform::ConfigVars;

fn main() -> eyre::Result<()> {
    // Set up nice error handing.
    color_eyre::install()?;
    // Specify cli argument parsing.
    let command = Command::new("aaxi")
        .about("A program for running axisymmetric simulations using numerical relativity")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("0.1.0")
        .config_args();
    // Check matches
    let matches = command.get_matches();
    // Find currect working directory
    let dir = std::env::current_dir().context("Failed to find current working directory")?;

    // *********************************
    // Configuration

    let (config, vars) = parse_config(&dir, &matches)?;

    match config.execution {
        // Just perform normal evolution
        Execution::Run => run_simulation(&config.transform(&vars)?)?,
        Execution::Search { ref search } => {
            // Okay, we are doing a critical search instead

            // Apply positional arguments
            let search = search.clone().transform(&vars)?;
            // Get parameter key
            let param_key = search.parameter_key.clone();

            let start = search.start();
            let end = search.end();

            let midpoint = (start + end) / 2.0;

            // Helper function for running simulation
            let run_search = |amplitude: f64| -> eyre::Result<()> {
                let mut vars = vars.clone();
                // Set the parameter variable to the given amplitude
                vars.named.insert(param_key.clone(), amplitude);
                // Transform config appropriately
                let config = config.clone().transform(&vars)?;
                // Run simulation
                run_simulation(&config)
            };

            run_search(start)?;
            run_search(midpoint)?;
            run_search(end)?;
        }
    }

    Ok(())
}

fn run_simulation(config: &Config) -> eyre::Result<()> {
    // Find currect working directory
    let dir = std::env::current_dir().context("Failed to find current working directory")?;
    // Compute output directory path.
    let output_path = abs_or_relative(&dir, &PathBuf::from(&config.output));
    // Ensure path exists.
    std::fs::create_dir_all(&output_path)?;

    // *********************************
    // Arguments

    // if matches. && output_path.join("cache").exists() {
    //     std::fs::remove_dir_all(output_path.join("cache"))?;
    // }

    // **********************************

    // Terminal output setup
    let term = Term::stdout();
    term.set_title("Axisymmetric Simulation");
    // Basic info dumping
    println!("Simulation: {}", style(&config.name).green());
    println!("Output Directory: {}", style(output_path.display()).green());
    println!(
        "Domain: {:.5} x {:.5}",
        config.domain.radius, config.domain.height
    );
    println!("Sources...");
    for source in &config.source {
        source.println();
    }

    // ***********************************
    // Validation

    eyre::ensure!(
        config.domain.radius > 0.0 && config.domain.height > 0.0,
        "Domain must have positive non-zero radius and height"
    );

    eyre::ensure!(
        config.domain.cell_size >= 2 * config.domain.cell_ghost,
        "Domain cell nodes must be >= 2 * ghost"
    );

    eyre::ensure!(config.visualize.stride >= 1, "Stride must be >= 1");

    // ***********************************
    // Initial data

    let (mesh, system) = initial_data(&config, &output_path)?;

    // ************************************
    // Evolve data forwards

    evolve_data(&config, &output_path, mesh, system)?;

    Ok(())
}

// ******************************
// Helpers **********************
// ******************************

fn parse_config(dir: &Path, matches: &ArgMatches) -> eyre::Result<(Config, ConfigVars)> {
    // Compute config path.
    let config_path = matches
        .get_one::<PathBuf>("config")
        .cloned()
        .unwrap_or("template.toml".to_string().into());
    let config_path = abs_or_relative(&dir, &config_path);

    // Parse config file from toml.
    let config = import_from_toml::<Config>(config_path)?;

    // Read positional arguments
    let positional_args: Vec<&str> = matches
        .get_many::<String>("positional")
        .into_iter()
        // .ok_or(eyre::eyre!("Unable to parse positional arguments"))?
        .flat_map(|v| v.into_iter().map(|s| s.as_str()))
        .collect();

    let vars = ConfigVars {
        positional: positional_args
            .into_iter()
            .map(|sbuf| sbuf.parse::<f64>())
            .collect::<Result<Vec<f64>, ParseFloatError>>()?,
        named: HashMap::new(),
    };

    Ok((config, vars))
}

/// Extension trait for defining helper methods on `clap::Command`.
trait CommandExt {
    fn config_args(self) -> Self;
}

impl CommandExt for Command {
    fn config_args(self) -> Self {
        self.arg(
            arg!(-c --config <FILE> "Sets a custom config file")
                .required(false)
                .value_parser(value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("clear-cache")
                .long("clear-cache")
                .help("Clears cache directory before running initial data/evolution")
                .num_args(0)
                .required(false),
        )
        .arg(
            arg!(<positional> ... "positional arguments referenced in config file")
                .trailing_var_arg(true)
                .required(false),
        )
    }
}

fn abs_or_relative(cwd: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }

    cwd.join(path)
}

pub fn import_from_toml<T: DeserializeOwned>(path: impl AsRef<Path>) -> eyre::Result<T> {
    let string = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&string)?)
}
