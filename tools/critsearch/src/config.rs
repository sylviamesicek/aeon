use anyhow::{anyhow, Context, Result};
use clap::{Arg, Command};

use genparams::CritSearchConfig;

pub fn configure() -> Result<CritSearchConfig> {
    let matches = Command::new("idgen")
        .about("A program for searching for critical points in a range using idgen and evgen.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("v0.0.1")
        .arg(
            Arg::new("path")
                .help("Path of config file for searching for critical points")
                .value_name("PATH")
                .required(true),
        )
        .get_matches();

    // Get path argument
    let path = matches
        .get_one::<String>("path")
        .ok_or(anyhow!("Failed to specify path argument"))?
        .clone();

    // Read config file.
    let config_string =
        String::from_utf8(std::fs::read(&path).context(format!("Failed to find {} file", &path))?)
            .context("Config file must be UTF8 encoded")?;

    // Parse config file into structure.
    toml::from_str(&config_string).context("Failed to parse config file")
}
