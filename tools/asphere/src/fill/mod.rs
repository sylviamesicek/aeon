use crate::{
    CommandExt as _, parse_define_args, parse_invoke_arg,
    run::{self, Status},
};
use aeon_app::{
    config::{Transform as _, VarDefs},
    file,
};
use clap::{ArgMatches, Command};
use eyre::eyre;

mod config;
mod history;

use config::Config;
use history::FillHistory;

pub fn fill(matches: &ArgMatches) -> eyre::Result<()> {
    let vars = parse_define_args(matches)?;
    let invoke = parse_invoke_arg(matches)?;

    // Find configuration files
    let config_run_file = std::env::current_dir()?.join(format!("{}.toml", invoke));
    let config_fill_file = std::env::current_dir()?.join(format!("{}.fill.toml", invoke));

    // Load fill configuration
    let fill_config = file::import_toml::<Config>(&config_fill_file)?;
    let fill_config = fill_config.transform(&vars)?;
    // Load run configuration
    let run_config = file::import_toml::<run::Config>(&config_run_file)?;

    // Find fill directory
    let fill_dir = fill_config.fill_dir()?;
    std::fs::create_dir_all(&fill_dir)?;
    // Load history file
    let history_file = fill_dir.join("history.csv");
    let mut history = FillHistory::load_csv(&history_file).unwrap_or_else(|_| FillHistory::new());
    // Name of parameter
    let parameter = fill_config.parameter.clone();

    // Write info into fill directory
    #[derive(serde::Serialize)]
    struct FillInfo {
        pstar: f64,
    }
    let info_file = fill_dir.join("info.toml");
    file::export_toml(
        &info_file,
        &FillInfo {
            pstar: fill_config.pstar.unwrap(),
        },
    )?;
    // Run fill
    fill_config.try_for_each(|amp| -> eyre::Result<()> {
        fill_iteration(&vars, &run_config, &parameter, amp, &mut history)?;
        // Cache history
        history.save_csv(&history_file)?;

        Ok(())
    })?;

    Ok(())
}

/// Run a single iteration of a critical search
fn fill_iteration(
    vars: &VarDefs,
    config: &run::Config,
    parameter: &str,
    amplitude: f64,
    history: &mut FillHistory,
) -> eyre::Result<()> {
    if let Some(mass) = history.mass(amplitude) {
        println!("Using cached mass: {} for amplitude: {}", mass, amplitude);
        return Ok(());
    }

    let mut vars = vars.clone();
    // Set the parameter variable to the given amplitude
    vars.defs
        .insert(parameter.to_string(), amplitude.to_string());
    // Transform config appropriately
    let config = config.transform(&vars)?;

    println!("Performing fill for amplitude: {}", amplitude);

    let sim = run::run_simulation(&config)?;

    if sim.status == Status::Disperse {
        println!("Simulation disperses during fill {}", amplitude);
        return Err(eyre!("dispersion during fill {}", amplitude));
    }

    // Insert it into history
    history.insert(amplitude, sim.mass);
    Ok(())
}

pub trait CommandExt {
    fn fill_cmd(self) -> Self;
}

impl CommandExt for Command {
    fn fill_cmd(self) -> Self {
        self.subcommand(
            Command::new("fill")
                .about("Performs mass fill operation")
                .define_args()
                .invoke_arg(),
        )
    }
}

pub fn parse_fill_cmd(matches: &ArgMatches) -> Option<&ArgMatches> {
    matches.subcommand_matches("fill")
}
