//! Submodule representing default `run` command (that is, `aaxi`) invoked without
//! any additional subcommands.
use std::{collections::HashMap, num::ParseFloatError, path::PathBuf};

use crate::{misc, run::status::Status};
use aeon_config::{ConfigVars, Transform as _};
use clap::{Arg, ArgMatches, Command, arg, value_parser};
use console::{Term, style};
use eyre::{Context, eyre};

mod config;
mod evolve;
mod history;
mod initial;
mod interval;
mod status;

use config::{Config, Execution};
use history::{RunHistory, SearchHistory};

pub fn run(matches: &ArgMatches) -> eyre::Result<()> {
    // *********************************
    // Configuration

    let (config, vars) = parse_config(&matches)?;

    match config.execution {
        // Just perform normal evolution
        Execution::Run => {
            let config = config.transform(&vars)?;
            let _ = run_simulation(&config, &mut RunHistory::empty())?;
        }
        // Okay, we are doing a critical search instead
        Execution::Search { ref search } => {
            // Apply positional arguments
            let search = search.clone().transform(&vars)?;
            // As well as search directory
            let search_dir = search.search_dir()?;
            std::fs::create_dir_all(&search_dir)?;

            // Setup range to search
            let mut start = search.start();
            let mut end = search.end();

            let history_file = search_dir.join("history.csv");
            let mut history =
                SearchHistory::load_csv(&history_file).unwrap_or_else(|_| SearchHistory::new());

            let start_status = run_search(&mut history, &config, &vars, start)?;
            let end_status = run_search(&mut history, &config, &vars, end)?;

            match (start_status, end_status) {
                (Status::Disperse, Status::Disperse) => {
                    return Err(eyre!("both sides of the parameter range disperse"));
                }
                (Status::Collapse, Status::Collapse) => {
                    return Err(eyre!("both sides of the parameter range collapse"));
                }
                _ => {}
            }

            let mut depth = 0;
            while depth < search.max_depth {
                // Have we reached minimum tolerance
                let tolerance = (end - start).abs();
                if tolerance <= search.min_error {
                    println!("Reached minimum critical parameter error {:.4e}", tolerance);
                    break;
                }

                println!("Searching range {} to {}", start, end);

                let midpoint = (start + end) / 2.0;
                let midpoint_status = run_search(&mut history, &config, &vars, start)?;

                match (start_status, midpoint_status, end_status) {
                    (Status::Disperse, Status::Disperse, Status::Collapse) => {
                        start = midpoint;
                    }
                    (Status::Disperse, Status::Collapse, Status::Collapse) => {
                        end = midpoint;
                    }
                    (Status::Collapse, Status::Disperse, Status::Disperse) => {
                        end = midpoint;
                    }
                    (Status::Collapse, Status::Collapse, Status::Disperse) => {
                        start = midpoint;
                    }
                    _ => unreachable!(),
                }

                // Cache history file.
                history.save_csv(&history_file)?;

                depth += 1;
                // Check maximum depth
                if depth == search.max_depth {
                    println!("Reached maximum critical search depth");
                }
            }
        }
    }

    Ok(())
}

/// Run a single iteration of a critical search
fn run_search(
    cache: &mut SearchHistory,
    config: &Config,
    vars: &ConfigVars,
    amplitude: f64,
) -> eyre::Result<Status> {
    if let Some(status) = cache.status(amplitude) {
        return Ok(status);
    }

    let search = config.search_config().unwrap();

    let mut vars = vars.clone();
    // Set the parameter variable to the given amplitude
    vars.named.insert(search.parameter.clone(), amplitude);
    // Transform config appropriately
    let config = config.clone().transform(&vars)?;
    // Setup history file
    let history_file = config
        .search_dir()?
        .join(format!("{}.csv", misc::encode_float(amplitude)));
    let mut history = RunHistory::output(&history_file)?;
    // Run simulation, keeping track of history
    let status = run_simulation(&config, &mut history)?;
    // Save to file
    history.flush()?;
    // Insert it into cache
    cache.insert(amplitude, status);
    return Ok(status);
}

fn run_simulation(config: &Config, history: &mut RunHistory) -> eyre::Result<Status> {
    let output_path = config.output_dir()?;
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
    for source in &config.sources {
        source.println();
    }

    // ***********************************
    // Validation

    config.validate()?;

    // ***********************************
    // Initial data

    let (mesh, system) = initial::initial_data(&config)?;

    // ************************************
    // Evolve data forwards

    let status = evolve::evolve_data(&config, history, mesh, system)?;

    Ok(status)
}

// ******************************
// Helpers **********************
// ******************************

fn parse_config(matches: &ArgMatches) -> eyre::Result<(Config, ConfigVars)> {
    // Compute config path.
    let config_path = matches
        .get_one::<PathBuf>("config")
        .cloned()
        .ok_or_else(|| eyre!("failed to specify config argument"))?;
    let config_path = misc::abs_or_relative(&config_path)?;

    // Parse config file from toml.
    let config =
        misc::import_from_toml::<Config>(&config_path).context("Failed to parse config file")?;

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
pub trait CommandExt {
    fn run_args(self) -> Self;
}

impl CommandExt for Command {
    fn run_args(self) -> Self {
        self.arg(
            arg!(-c --config <FILE> "Sets a custom config file")
                .required(true)
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
