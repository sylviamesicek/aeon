//! This crate contains general configuration and paramter data types used by critsearch, idgen, and evgen.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct CritSearchConfig {
    pub name: String,

    pub start: f64,
    pub end: f64,

    #[serde(default = "default_subsearches")]
    pub subsearches: usize,

    /// Specifies domain of problem
    #[serde(default)]
    pub domain: Domain,

    /// Directory to store output.
    pub output_dir: Option<String>,
    /// Verbosity of logging.
    pub logging_level: Option<usize>,
}

fn default_subsearches() -> usize {
    3
}

/// Configuration format for IdGen.
#[derive(Serialize, Deserialize)]
pub struct InitialDataConfig {
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

    /// Sources used for simulation
    pub source: Vec<Source>,
}

/// Options defining the domain of the problem.
#[derive(Serialize, Deserialize, Clone)]
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
#[derive(Serialize, Deserialize, Clone)]
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
#[derive(Serialize, Deserialize, Clone)]
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

#[derive(Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
pub enum Source {
    /// Instance generates Brill-type initial data with gunlach seed function.
    #[serde(rename = "brill")]
    Brill(Brill),
}

/// Describes Brill-type initial data parameters.
#[derive(Serialize, Deserialize, Debug)]
pub struct Brill {
    pub amplitude: f64,
    pub sigma: (f64, f64),
}
