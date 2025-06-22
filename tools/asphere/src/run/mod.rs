use crate::misc;
use aeon_config::{ConfigVars, Transform as _};
use clap::{ArgMatches, Command, arg, value_parser};
use console::style;
use eyre::{Context as _, eyre};
use std::{collections::HashMap, num::ParseFloatError, path::PathBuf};

pub mod config;
mod evolve;
mod history;
mod initial;

use config::*;
use evolve::{EvolveInfo, Status};
use history::{FillHistory, SearchHistory};

pub fn run(matches: &ArgMatches) -> eyre::Result<()> {
    // *********************************
    // Configuration

    let (config, vars) = parse_config(&matches)?;

    match config.execution {
        // Just perform normal evolution
        Execution::Run => {
            let config = config.transform(&vars)?;
            let _ = run_simulation(&config, None)?;
        }
        // Okay, we are doing a critical search instead
        Execution::Search { ref search } => {
            // Apply positional arguments
            let search = search.clone().transform(&vars)?;
            // As well as search directory
            let search_dir = search.search_dir()?;
            std::fs::create_dir_all(&search_dir)?;
            // Load history file
            let history_file = search_dir.join("history.csv");
            let mut history =
                SearchHistory::load_csv(&history_file).unwrap_or_else(|_| SearchHistory::new());

            // Setup range to search
            let mut start = search.start();
            let mut end = search.end();

            let mut start_status = run_search(&mut history, &config, &vars, start)?;
            let mut end_status = run_search(&mut history, &config, &vars, end)?;

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
            loop {
                // Have we reached minimum tolerance
                let tolerance = (end - start).abs();
                if tolerance <= search.min_error {
                    println!("Reached minimum critical parameter error {:.4e}", tolerance);
                    break;
                }

                println!(
                    "Searching range: {} to {}, diff: {:.4e}",
                    start, end, tolerance
                );

                let midpoint = (start + end) / 2.0;
                let midpoint_status = run_search(&mut history, &config, &vars, midpoint)?;

                match (start_status, midpoint_status, end_status) {
                    (Status::Disperse, Status::Disperse, Status::Collapse) => {
                        start = midpoint;
                        start_status = midpoint_status;
                    }
                    (Status::Disperse, Status::Collapse, Status::Collapse) => {
                        end = midpoint;
                        end_status = midpoint_status;
                    }
                    (Status::Collapse, Status::Disperse, Status::Disperse) => {
                        end = midpoint;
                        end_status = midpoint_status;
                    }
                    (Status::Collapse, Status::Collapse, Status::Disperse) => {
                        start = midpoint;
                        start_status = midpoint_status;
                    }
                    _ => unreachable!(),
                }

                // Cache history file.
                history.save_csv(&history_file)?;

                depth += 1;
                // Check maximum depth
                if depth >= search.max_depth {
                    println!("Reached maximum critical search depth");
                    break;
                }
            }

            println!(
                "Final search range: {} to {}, diff: {:.4e}",
                start,
                end,
                (end - start).abs()
            );
        }
        // Run a fill operation
        Execution::Fill { ref fill } => {
            // Apply positional arguments
            let fill = fill.clone().transform(&vars)?;
            // As well as search directory
            let fill_dir = fill.fill_dir()?;
            std::fs::create_dir_all(&fill_dir)?;
            // Load history file
            let history_file = fill_dir.join("history.csv");
            let mut history =
                FillHistory::load_csv(&history_file).unwrap_or_else(|_| FillHistory::new());

            fill.try_for_each(|amp| -> eyre::Result<()> {
                run_fill(&mut history, &config, &vars, amp)?;
                // Cache history
                history.save_csv(&history_file)?;

                Ok(())
            })?;
        }
    }

    Ok(())
}

/// Run a single iteration of a critical search
fn run_search(
    history: &mut SearchHistory,
    config: &Config,
    vars: &ConfigVars,
    amplitude: f64,
) -> eyre::Result<Status> {
    if let Some(status) = history.status(amplitude) {
        println!(
            "Using cached status: {:?} for amplitude: {}",
            status, amplitude
        );
        return Ok(status);
    }

    let search = config.search_config().unwrap();

    let mut vars = vars.clone();
    // Set the parameter variable to the given amplitude
    vars.named.insert(search.parameter.clone(), amplitude);
    // Transform config appropriately
    let config = config.transform(&vars)?;

    println!("Performing search for amplitude: {}", amplitude);

    let status = run_simulation(&config, None)?;

    // Insert it into cache
    history.insert(amplitude, status);
    return Ok(status);
}

/// Run a single iteration of a fill operation
fn run_fill(
    history: &mut FillHistory,
    config: &Config,
    vars: &ConfigVars,
    amplitude: f64,
) -> eyre::Result<()> {
    if let Some(mass) = history.mass(amplitude) {
        println!("Using cached mass: {} for amplitude: {}", mass, amplitude);
        return Ok(());
    }

    let fill = config.fill_config().unwrap();

    let mut vars = vars.clone();
    // Set the parameter variable to the given amplitude
    vars.named.insert(fill.parameter.clone(), amplitude);
    // Transform config appropriately
    let config = config.transform(&vars)?;

    println!("Performing fill for amplitude: {}", amplitude);

    let mut info = EvolveInfo::default();
    let status = run_simulation(&config, Some(&mut info))?;

    if status == Status::Disperse {
        println!("Simulation disperses during fill {}", amplitude);
        return Err(eyre!("dispersion during fill {}", amplitude));
    }

    // Insert it into history
    history.insert(amplitude, info.mass);
    Ok(())
}

fn run_simulation(config: &Config, info: Option<&mut EvolveInfo>) -> eyre::Result<Status> {
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
    evolve::evolve_data(config, mesh, system, info)
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
        self.subcommand_negates_reqs(true)
            .arg(
                arg!(-c --config <FILE> "Sets a custom config file")
                    .required(true)
                    .value_parser(value_parser!(PathBuf)),
            )
            .arg(
                arg!(<positional> ... "positional arguments referenced in config file")
                    .trailing_var_arg(true)
                    .required(false),
            )
    }
}
