//! An executable for creating general initial data for numerical relativity simulations in 2D.

use aeon_app::config::{VarDef, VarDefs};
use clap::{Arg, ArgAction, ArgMatches, Command};
use eyre::eyre;

mod fill;
mod run;
mod search;
mod system;

use fill::CommandExt as _;
use run::CommandExt as _;
use search::CommandExt as _;

// Main function that can return an error
fn main() -> eyre::Result<()> {
    // Set up nice colored error handing.
    color_eyre::install()?;
    // Load configuration
    let command = Command::new("asphere")
        .about("A program for simulating GR in spherical symmetry.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("0.2.2")
        .subcommand_negates_reqs(true)
        .search_cmd()
        .fill_cmd()
        .run_cmd();
    // Find matches
    let matches = command.get_matches();

    // Try running search command
    if let Some(search_matches) = search::parse_search_cmd(&matches) {
        println!("Running search subcommand");
        return search::search(search_matches);
    }

    // Try running fill command
    if let Some(fill_matches) = fill::parse_fill_cmd(&matches) {
        println!("Running fill subcommand");
        return fill::fill(fill_matches);
    }

    // Try running run command
    if let Some(run_matches) = run::parse_run_cmd(&matches) {
        println!("Running run subcommand");
        return run::run(run_matches);
    }

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

    // fn config_arg(self) -> Self {
    //     self.arg(
    //         Arg::new("config")
    //             .long("config")
    //             .short('c')
    //             .value_hint(ValueHint::AnyPath)
    //             .help("asphere configuration file")
    //             .value_parser(value_parser!(PathBuf))
    //             .required(true),
    //     )
    // }
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

// /// Loads the invokation argument from a set of argument matches
// fn parse_config_arg(matches: &ArgMatches) -> eyre::Result<PathBuf> {
//     let Some(invoke) = matches.get_one::<PathBuf>("invoke") else {
//         return Err(eyre!(
//             "failed to find 0th positional argument in cli invokation"
//         ));
//     };

//     Ok(invoke.clone())
// }

/// Loads the invokation argument from a set of argument matches
fn parse_invoke_arg(matches: &ArgMatches) -> eyre::Result<String> {
    let Some(invoke) = matches.get_one::<String>("invoke") else {
        return Err(eyre!(
            "failed to find 0th positional argument in cli invokation"
        ));
    };

    Ok(invoke.clone())
}
