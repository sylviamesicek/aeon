use aeon_config::{ConfigVars, Transform as _};
use clap::{Arg, ArgMatches, Command, arg, value_parser};
use console::{Term, style};
use eyre::{Context, eyre};
use history::{RunHistory, RunStatus, SearchHistory};
use std::{collections::HashMap, num::ParseFloatError, path::PathBuf};

mod config;
mod history;
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
    // Check argument matches
    let matches = command.get_matches();

    // *********************************
    // Configuration

    let (config, vars) = parse_config(&matches)?;

    match config.execution {
        // Just perform normal evolution
        Execution::Run => {
            let config = config.transform(&vars)?;
            run_simulation(&config, &mut RunHistory::empty())?
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

            let start_disperses = run_search(&mut history, &config, &vars, start)?.has_dispersed();
            let end_disperses = run_search(&mut history, &config, &vars, end)?.has_dispersed();

            if start_disperses == end_disperses {
                if start_disperses {
                    return Err(eyre!("both sides of the parameter range disperse"));
                } else {
                    return Err(eyre!("both sides of the parameter range collapse"));
                }
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

                let midpoint_disperses =
                    run_search(&mut history, &config, &vars, start)?.has_dispersed();

                if midpoint_disperses == start_disperses {
                    debug_assert!(midpoint_disperses != end_disperses);
                    start = midpoint;
                } else {
                    debug_assert!(midpoint_disperses == end_disperses);
                    end = midpoint;
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
) -> eyre::Result<RunStatus> {
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
    run_simulation(&config, &mut history)?;
    // Save to file
    history.flush()?;
    // Get status
    let status = history.status();
    // Insert it into cache
    cache.insert(amplitude, status);
    return Ok(status);
}

fn run_simulation(config: &Config, history: &mut RunHistory) -> eyre::Result<()> {
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

    eyre::ensure!(
        config.domain.radius > 0.0 && config.domain.height > 0.0,
        "Domain must have positive non-zero radius and height"
    );

    eyre::ensure!(
        config.domain.cell_size >= 2 * config.domain.cell_ghost,
        "Domain cell nodes must be >= 2 * ghost"
    );

    // ***********************************
    // Initial data

    let (mesh, system) = initial_data(&config)?;

    // ************************************
    // Evolve data forwards

    evolve_data(&config, history, mesh, system)?;

    Ok(())
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
trait CommandExt {
    fn config_args(self) -> Self;
}

impl CommandExt for Command {
    fn config_args(self) -> Self {
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
