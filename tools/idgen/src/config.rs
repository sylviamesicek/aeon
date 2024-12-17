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
    /// Minimum order of stencils approximations.
    pub order: usize,
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

    /// Specifies paramteres for the initial data solver.
    #[serde(default)]
    pub solver: Solver,

    /// Instances that should be executed.
    pub instance: Vec<Instance>,
}

/// Options defining the domain of the problem.
#[derive(Deserialize)]
pub struct Domain {
    pub radius: f64,
    pub height: f64,

    #[serde(default)]
    pub cell: Cell,

    #[serde(default)]
    pub mesh: Mesh,
}

impl Default for Domain {
    fn default() -> Self {
        Self {
            radius: 1.0,
            height: 1.0,

            cell: Cell::default(),
            mesh: Mesh::default(),
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

/// Options for the initial shape of the mesh.
#[derive(Deserialize)]
pub struct Mesh {
    pub refine_global: usize,
    pub max_level: usize,
}

impl Default for Mesh {
    fn default() -> Self {
        Self {
            refine_global: 0,
            max_level: 10,
        }
    }
}

#[derive(Deserialize)]
pub struct Solver {
    pub max_steps: usize,
    pub cfl: f64,
    pub tolerance: f64,
    pub dampening: f64,
}

impl Default for Solver {
    fn default() -> Self {
        Self {
            max_steps: 100000,
            cfl: 0.5,
            tolerance: 1e-6,
            dampening: 0.4,
        }
    }
}

/// Describes initial data for a brill wave.
#[derive(Deserialize, Debug)]
pub struct BrillSource {
    pub amplitude: f64,
    pub sigma: (f64, f64),
}

pub struct ScalarField {
    pub mass: f64,
    pub amplitude: f64,
    pub sigma: (f64, f64),
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Instance {
    /// Instance generates Brill-type initial data with gunlach seed function.
    #[serde(rename = "brill")]
    Brill(Brill),
}

/// Describes Brill-type initial data parameters.
#[derive(Deserialize, Debug)]
pub struct Brill {
    #[serde(default)]
    pub suffix: String,
    pub amplitude: f64,
    pub sigma: (f64, f64),
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
