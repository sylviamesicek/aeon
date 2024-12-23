use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct CritConfig {
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
    #[serde(default)]
    pub logging: Logging,
}

fn default_subsearches() -> usize {
    3
}

/// Configuration format for IdGen.
#[derive(Serialize, Deserialize)]
pub struct IDConfig {
    /// Name of process to be executed.
    pub name: String,
    /// Directory to store output.
    pub output_dir: Option<String>,
    /// Logging configuration.
    #[serde(default)]
    pub logging: Logging,
    /// Minimum order of stencils approximations.
    pub order: usize,
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

#[derive(Serialize, Deserialize)]
pub struct EVConfig {
    /// Logging configuration.
    #[serde(default)]
    pub logging: Logging,
    /// Order of stencils approximations.
    #[serde(default = "default_order")]
    pub order: usize,
    /// Order of disspersion stencils.
    #[serde(default = "default_diss_order")]
    pub diss_order: usize,

    pub cfl: f64,

    pub max_time: f64,
    pub max_steps: usize,
    pub max_nodes: usize,

    /// Configuration for regridding.
    pub regrid: Regrid,
    /// Configuration for visualization (if any)
    #[serde(default)]
    pub visualize: Option<Visualize>,
}

fn default_order() -> usize {
    4
}

fn default_diss_order() -> usize {
    6
}

#[derive(Serialize, Deserialize)]
pub struct Logging {
    /// Verbosity of logging
    pub level: usize,
}

impl Logging {
    pub const OFF: usize = 0;
    pub const ERROR: usize = 1;
    pub const WARN: usize = 2;
    pub const INFO: usize = 3;
    pub const DEBUG: usize = 4;
    pub const TRACE: usize = 5;

    /// Converts a logging level to a `log::LevelFilter`.
    pub fn filter(&self) -> log::LevelFilter {
        match self.level {
            0 => log::LevelFilter::Off,
            1 => log::LevelFilter::Error,
            2 => log::LevelFilter::Warn,
            3 => log::LevelFilter::Info,
            4 => log::LevelFilter::Debug,
            _ => log::LevelFilter::Trace,
        }
    }
}

impl Default for Logging {
    fn default() -> Logging {
        Logging { level: 2 }
    }
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

#[derive(Serialize, Deserialize, Debug)]
pub struct Regrid {
    pub coarsen_limit: f64,
    pub refine_limit: f64,
    pub flag_interval: usize,
    pub max_levels: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Visualize {
    pub save_interval: f64,
}
