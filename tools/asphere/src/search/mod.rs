use crate::{
    CommandExt as _, parse_define_args, parse_invoke_arg,
    run::{self, SimulationInfo, Status, Subrun},
};
use aeon_app::{
    config::{Transform, VarDefs},
    file,
};
use clap::{ArgMatches, Command};
use console::style;
use eyre::eyre;

mod config;
mod history;

use config::Config;
use history::SearchHistory;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

pub fn search(matches: &ArgMatches) -> eyre::Result<()> {
    let vars = parse_define_args(matches)?;
    let invoke = parse_invoke_arg(matches)?;

    let config_run_file = std::env::current_dir()?.join(format!("{}.toml", invoke));
    let config_search_file = std::env::current_dir()?.join(format!("{}.search.toml", invoke));

    // Load search configuration
    let search_config = file::import_toml::<Config>(&config_search_file)?;
    let search_config = search_config.transform(&vars)?;
    // Load run configuration
    let run_config = file::import_toml::<run::Config>(&config_run_file)?;

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

    let multi = MultiProgress::new();

    let info = [start, end]
        .par_iter()
        .map(|value| search_iteration(&vars, &run_config, &parameter, *value, &history, &multi))
        .collect::<Result<Vec<_>, _>>()?;

    multi.clear()?;

    let start_info = info[0];
    let end_info = info[1];

    history.insert(start, start_info.status, start_info.mass);
    history.insert(end, end_info.status, end_info.mass);

    let start_status = start_info.status;
    let end_status = end_info.status;

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

        let amplitudes = (0..search_config.parallel)
            .map(|i| (i + 1) as f64 / (search_config.parallel + 1) as f64)
            .map(|w| (1.0 - w) * start + w * end)
            .collect::<Vec<_>>();

        let multi = MultiProgress::new();

        // Execute iterations independently
        let info = amplitudes
            .par_iter()
            .map(|&w| search_iteration(&vars, &run_config, &parameter, w, &history, &multi))
            .collect::<Result<Vec<SimulationInfo>, _>>()?;

        drop(multi);

        // Insert results into cache
        for (w, s) in amplitudes.iter().zip(info.iter()) {
            history.insert(*w, s.status, s.mass);
        }

        let status = info.iter().map(|info| info.status).collect::<Vec<_>>();

        println!("Status Results: {:?}", status);

        // Imagine vector of all statuses, including start and end

        let (sindex, eindex) = double_headed_search(start_status, &status, end_status);
        match (sindex, eindex) {
            (Some(idx), None) if idx == status.len() - 1 => {}
            (None, Some(0)) => {}
            (Some(start), Some(end)) if start + 1 == end => {}
            _ => return Err(eyre!("range contains multiple critical points")),
        }

        if let Some(sidx) = sindex {
            assert!(status[sidx] == start_status);
            start = amplitudes[sidx];
        }

        if let Some(eidx) = eindex {
            assert!(status[eidx] == end_status);
            end = amplitudes[eidx]
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
    history: &SearchHistory,
    multi: &MultiProgress,
) -> eyre::Result<SimulationInfo> {
    if let Some(s) = history.status(amplitude) {
        let bar = multi
            .add(ProgressBar::no_length().with_style(
                ProgressStyle::with_template("{prefix:.bold.dim} {wide_msg}").unwrap(),
            ));

        let status = if s.status == Status::Disperse {
            style("Disperses").green()
        } else {
            style("Collapses").red()
        };

        bar.set_prefix(format!("[P = {:.16}]", amplitude));
        bar.finish_with_message(format!("Cached - {}, Mass: {}", status, s.mass));

        return Ok(s);
    }

    let mut vars = vars.clone();
    // Set the parameter variable to the given amplitude
    vars.defs
        .insert(parameter.to_string(), amplitude.to_string());
    // Transform config appropriately
    let config = config.transform(&vars)?;

    let sim = run::subrun(
        &config,
        &Subrun {
            multi: multi.clone(),
            parameter: amplitude,
        },
    )?;
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

/// Returns the last index in values that matches start and the first index in values that matches end.
fn double_headed_search<T: PartialEq>(
    start: T,
    values: &[T],
    end: T,
) -> (Option<usize>, Option<usize>) {
    assert!(start != end);

    if values.len() == 0 {
        return (None, None);
    }

    if values.len() == 1 {
        if start == values[0] {
            return (Some(0), None);
        }

        if end == values[0] {
            return (None, Some(0));
        }

        panic!("Invalid search");
    }

    let sindex = if values[0] != start {
        None
    } else {
        let mut sindex = 0;
        while sindex < values.len() - 1 && values[sindex + 1] == start {
            sindex += 1;
        }
        Some(sindex)
    };

    let eindex = if values[values.len() - 1] != end {
        None
    } else {
        let mut eindex = values.len() - 1;
        while eindex > 0 && values[eindex - 1] == end {
            eindex -= 1;
        }
        Some(eindex)
    };

    (sindex, eindex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_headed() {
        assert_eq!(double_headed_search(false, &[true], true), (None, Some(0)));
        assert_eq!(double_headed_search(false, &[false], true), (Some(0), None));
        assert_eq!(
            double_headed_search(false, &[false, false, true, true], true),
            (Some(1), Some(2))
        );
        assert_eq!(
            double_headed_search(false, &[false, false, true, false, true, true], true),
            (Some(1), Some(4))
        );
    }
}
