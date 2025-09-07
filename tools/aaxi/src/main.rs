use aeon_app::config::{VarDef, VarDefs};
use clap::{Arg, ArgAction, ArgMatches, Command};
use eyre::eyre;

mod eqs;
mod horizon;
mod run;
mod schwarzschild;
mod search;
mod systems;

use run::CommandExt as _;
use schwarzschild::CommandExt as _;
use search::CommandExt as _;

fn main() -> eyre::Result<()> {
    // Set up nice error handing.
    color_eyre::install()?;
    // Specify cli argument parsing.
    let command = Command::new("aaxi")
        .about("A program for running axisymmetric simulations using numerical relativity")
        .author("Sylvia Mesicek, sylvia.mesicek@gmail.com")
        .version("0.3.0")
        .subcommand_negates_reqs(true)
        .schwarzschild_cmd()
        .run_cmd()
        .search_cmd();
    // Check argument matches
    let matches = command.get_matches();

    // Run schwarzschild subcommand
    if let Some(matches) = schwarzschild::parse_schwarzschild_cmd(&matches) {
        println!("Running Schwarzschild subcommand");
        return schwarzschild::schwarzschild(matches);
    }

    // Run search subcommand
    if let Some(matches) = search::parse_search_cmd(&matches) {
        println!("Running Search subcommand");
        return search::search(matches);
    }

    // Run default subcommand
    if let Some(matches) = run::parse_run_cmd(&matches) {
        println!("Running run subcommand");
        return run::run(matches);
    }

    // Run default subcommand
    println!("No subcommand provided");
    Ok(())
}

trait CommandExt {
    /// Implements default define args
    fn define_args(self) -> Self;
    /// Implements 0th positional argument for executing a specific subconfig
    fn invoke_arg(self) -> Self;
}

impl CommandExt for Command {
    fn define_args(self) -> Self {
        self.arg(
            Arg::new("define")
                .long("define")
                .short('D')
                .help("Define variable to be referenced in config files via ${} syntax")
                .required(false)
                .action(ArgAction::Append),
        )
    }

    fn invoke_arg(self) -> Self {
        self.arg(Arg::new("invoke").required(true))
    }
}

/// Loads a series of variable definitions from a set of argument matches
fn parse_define_args(matches: &ArgMatches) -> eyre::Result<VarDefs> {
    // Collection of cli invokation variable definitions.
    let mut vars = VarDefs::new();
    if let Some(defines) = matches.get_many::<String>("define") {
        for def in defines {
            vars.insert(VarDef::parse(def)?);
        }
    }
    Ok(vars)
}

/// Loads the invokation argument from a set of argument matches
fn parse_invoke_arg(matches: &ArgMatches) -> eyre::Result<String> {
    let Some(invoke) = matches.get_one::<String>("invoke") else {
        return Err(eyre!(
            "failed to find 0th positional argument in cli invokation"
        ));
    };

    Ok(invoke.clone())
}
