//! An executable for creating general initial data for numerical relativity simulations in 2D.
#![allow(unused_assignments)]

use clap::Command;

mod misc;
mod run;
mod system;

use run::CommandExt as _;

// Main function that can return an error
fn main() -> eyre::Result<()> {
    // Set up nice colored error handing.
    color_eyre::install()?;
    // Load configuration
    let command = Command::new("asphere")
        .about("A program for simulating GR in spherical symmetry.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("0.1.0")
        .subcommand_negates_reqs(true)
        .run_args();
    // Find matches
    let matches = command.get_matches();

    // Run default subcommand
    println!("Running default subcommand");
    run::run(&matches)
}
