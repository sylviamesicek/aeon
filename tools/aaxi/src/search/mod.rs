use crate::{
    CommandExt as _, parse_define_args, parse_invoke_arg,
    run::{self, RunHistory, Status},
};
use aeon_app::{
    config::{Transform, VarDefs},
    file,
};
use clap::{ArgMatches, Command};
use eyre::{Context as _, eyre};

mod config;
mod history;

use config::Config;
use history::SearchHistory;

pub fn search(matches: &ArgMatches) -> eyre::Result<()> {
    let vars = parse_define_args(matches)?;
    let invoke = parse_invoke_arg(matches)?;

    let config_run_file = std::env::current_dir()?.join(format!("{}.toml", invoke));
    let config_search_file = std::env::current_dir()?.join(format!("{}.search.toml", invoke));

    // Load search configuration
    let search_config = file::import_toml::<Config>(&config_search_file).with_context(|| {
        format!(
            "failed to find search config file: {:?}",
            config_search_file
        )
    })?;
    let search_config = search_config.transform(&vars)?;
    // Load run configuration
    let run_config = file::import_toml::<run::Config>(&config_run_file)
        .with_context(|| format!("failed to find run config file: {:?}", config_run_file))?;

    // Find search directory
    let search_dir = search_config.search_dir()?;
    std::fs::create_dir_all(&search_dir)?;
    // Load history file
    let history_file = search_dir.join("history.csv");
    let mut history =
        SearchHistory::load_csv(&history_file).unwrap_or_else(|_| SearchHistory::new());

    // Setup range to search
    let mut start = search_config.start();
    let mut end = search_config.end();
    // Name of parameter
    let parameter = search_config.parameter.clone();

    let mut start_status = search_iteration(&vars, &run_config, &parameter, start, &mut history)?;
    let mut end_status = search_iteration(&vars, &run_config, &parameter, end, &mut history)?;

    // Cache history file.
    history.save_csv(&history_file)?;

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
        if tolerance <= search_config.min_error {
            println!("Reached minimum critical parameter error {:.4e}", tolerance);
            break;
        }

        println!(
            "Searching range: {} to {}, diff: {:.4e}",
            start, end, tolerance
        );

        let midpoint = (start + end) / 2.0;
        let midpoint_status =
            search_iteration(&vars, &run_config, &parameter, midpoint, &mut history)?;

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
        if depth >= search_config.max_depth {
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

    Ok(())
}

/// Run a single iteration of a critical search
fn search_iteration(
    vars: &VarDefs,
    config: &run::Config,
    parameter: &str,
    amplitude: f64,
    history: &mut SearchHistory,
) -> eyre::Result<Status> {
    if let Some(status) = history.status(amplitude) {
        println!(
            "Using cached status: {:?} for amplitude: {}",
            status, amplitude
        );
        return Ok(status);
    }

    let mut vars = vars.clone();
    // Set the parameter variable to the given amplitude
    vars.defs
        .insert(parameter.to_string(), amplitude.to_string());
    // Transform config appropriately
    let config = config.transform(&vars)?;

    println!("Performing search for amplitude: {}", amplitude);
    // Setup history file
    let sim = run::run_simulation(&config, &mut RunHistory::empty())?;
    // Insert it into cache
    history.insert(amplitude, sim);
    return Ok(sim);
}

pub trait CommandExt {
    fn search_cmd(self) -> Self;
}

impl CommandExt for Command {
    fn search_cmd(self) -> Self {
        self.subcommand(
            Command::new("search")
                .about("Performs critical search")
                .define_args()
                .invoke_arg(),
        )
    }
}

pub fn parse_search_cmd(matches: &ArgMatches) -> Option<&ArgMatches> {
    matches.subcommand_matches("search")
}
