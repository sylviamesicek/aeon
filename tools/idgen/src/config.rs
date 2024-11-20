use anyhow::{anyhow, Context, Result};
use clap::{Arg, Command};
use serde::Deserialize;

/// Configuration format for IdGen.
#[derive(Deserialize)]
pub struct Config {
    /// Name of process to be executed.
    pub name: String,
    /// Directory to store output.
    pub output_dir: Option<String>,
    /// Verbosity of logging.
    pub logging_level: Option<usize>,
    pub _logging_dir: Option<String>,
    /// Visualize level between each regrid?
    #[serde(default)]
    pub _visualize_levels: bool,
    /// Visualize each instance once complete?
    #[serde(default)]
    pub _visualize_result: bool,

    /// Specifies domain of problem
    #[serde(default)]
    pub domain: Domain,
    #[serde(default)]
    pub cell: Cell,

    /// Instances that should be executed.
    pub instance: Vec<Instance>,
}

/// Options defining the domain of the problem.
#[derive(Deserialize)]
pub struct Domain {
    pub radius: f64,
    pub height: f64,
}

impl Default for Domain {
    fn default() -> Self {
        Self {
            radius: 1.0,
            height: 1.0,
        }
    }
}

/// Options for size and order of cells on mesh.
#[derive(Deserialize)]
pub struct Cell {
    pub subdivisions: usize,
    pub padding: usize,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            subdivisions: 6,
            padding: 3,
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Instance {
    /// Instance generates Brill-type initial data with gunlach seed function.
    #[serde(rename = "brill")]
    Brill {
        #[serde(default)]
        suffix: String,
        amplitude: f64,
        sigma: (f64, f64),
    },
}

pub fn configure() -> Result<Config> {
    let matches = Command::new("idgen")
        .about("A program for generating initial data for numerical relativity using hyperbolic relaxation.")
        .author("Lukas Mesicek, lukas.m.mesicek@gmail.com")
        .version("v0.0.1")
        .arg(
            Arg::new("path")
                .help("Path of config file for generating initial data")
                .value_name("PATH")
                .required(true),
        ).get_matches();

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
