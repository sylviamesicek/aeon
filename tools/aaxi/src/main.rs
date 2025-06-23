use clap::Command;

mod eqs;
mod horizon;
mod run;
mod schwarzschild;
mod systems;

use run::CommandExt as _;
use schwarzschild::CommandExt as _;

fn main() -> eyre::Result<()> {
    // Set up nice error handing.
    color_eyre::install()?;
    // Specify cli argument parsing.
    let command = Command::new("aaxi")
        .about("A program for running axisymmetric simulations using numerical relativity")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("0.2.1")
        .subcommand_negates_reqs(true)
        .run_args()
        .schwarzschild_args();
    // Check argument matches
    let matches = command.get_matches();

    // Run schwarzschild subcommand
    if let Some(matches) = matches.subcommand_matches("schwarzschild") {
        println!("Running Schwarzschild subcommand");
        return schwarzschild::schwarzschild(matches);
    }

    // Run default subcommand
    println!("Running default subcommand");
    run::run(&matches)
}
