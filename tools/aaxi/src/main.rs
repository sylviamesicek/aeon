use clap::{ArgMatches, Command, arg, value_parser};
use console::{Term, style};
use eyre::WrapErr;
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};

mod config;
mod misc;
mod rinne;

use config::*;
use rinne::*;

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

    let (config, sources) = parse_config(&dir, &matches)?;
    // Compute output directory path.
    let output_path = abs_or_relative(&dir, &PathBuf::from(&config.output));
    // Ensure path exists.
    std::fs::create_dir_all(&output_path)?;

    // **********************************

    // Terminal output setup
    let term = Term::stdout();
    term.set_title("Axisymmetric Simulation");
    // Basic info dumping
    println!("Running simulation: {}", style(&config.name).green());
    println!("Saving output to: {}", style(output_path.display()).green());
    println!(
        "Domain: {:.5} x {:.5}",
        config.domain.radius, config.domain.height
    );
    println!("Sources...");
    for source in &sources {
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

    let (_mesh, _system) = initial_data(&config, &sources, &output_path)?;

    Ok(())
}

// ******************************
// Helpers **********************
// ******************************

fn parse_config(dir: &Path, matches: &ArgMatches) -> eyre::Result<(Config, Vec<Source>)> {
    // Compute config path.
    let config_path = matches
        .get_one::<PathBuf>("config")
        .cloned()
        .unwrap_or("aaxi.toml".to_string().into());
    let config_path = abs_or_relative(&dir, &config_path);

    // Parse config file from toml.
    let config = import_from_toml::<Config>(config_path)?;

    let positional: Vec<&str> = matches
        .get_many::<String>("positional")
        .into_iter()
        // .ok_or(eyre::eyre!("Unable to parse positional arguments"))?
        .flat_map(|v| v.into_iter().map(|s| s.as_str()))
        .collect();

    // Collect transformed sources.
    let mut sources = Vec::new();
    for pos in &config.source {
        sources.push(Source::from_pos(pos, &positional)?);
    }

    Ok((config, sources))
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
