use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct CritConfig {
    /// Name of simulation.
    pub name: String,

    /// Lower bound of amplitudes to search.
    pub start: f64,
    /// Upper bound of amplitudes to search.
    pub end: f64,

    /// Should we cache initial data?
    #[serde(default)]
    pub cache_initial: bool,
    /// Should we cache evolution runs?
    #[serde(default)]
    pub cache_evolve: bool,
    /// Number of subsearches to perform.
    #[serde(default = "default_subsearches")]
    pub subsearches: usize,
    /// Depth of subsearches.
    pub bifurcations: usize,
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
    pub visualize_levels: bool,
    /// Visualize each instance once complete?
    #[serde(default)]
    pub visualize_result: bool,
    #[serde(default)]
    /// Visualize the relaxation process?
    pub visualize_relax: bool,
    /// Stride for vtu outputs.
    #[serde(default = "default_stride")]
    pub visualize_stride: usize,
    /// Produce visualization every certain number of iterations.
    pub visualize_every: usize,

    /// Maximum allowable level.
    pub max_level: usize,
    /// Maximum number of nodes.
    pub max_nodes: usize,
    /// Maximum error on any given cell.
    pub max_error: f64,

    /// Number of global refinements to perform before running the solver.
    pub refine_global: usize,

    /// Specifies domain of problem
    #[serde(default)]
    pub domain: Domain,

    /// Specifies paramteres for the initial data solver.
    #[serde(default)]
    pub solver: Solver,

    /// Sources used for simulation
    #[serde(default)]
    pub source: Vec<Source>,
}

fn default_stride() -> usize {
    1
}

#[derive(Serialize, Deserialize)]
pub struct EVConfig {
    /// Name of process to be executed.
    pub name: String,
    /// Directory to store output.
    pub output_dir: Option<String>,
    /// Logging configuration.
    #[serde(default)]
    pub logging: Logging,
    /// Order of stencils approximations.
    #[serde(default = "default_order")]
    pub order: usize,
    /// Order of disspersion stencils.
    #[serde(default = "default_diss_order")]
    pub diss_order: usize,

    /// CFL factor
    pub cfl: f64,
    /// Amount of Kriss-Olgier Dissipation to use
    pub dissipation: f64,

    /// Maximum amount of time to run.
    pub max_time: f64,
    /// Maximum number of steps to take before failing.
    pub max_steps: usize,
    /// Maximum number of nodes allowed for refinement before failing.
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
    /// Size of domain along the ρ axis.
    pub radius: f64,
    /// Size of domain along the z-axis.
    pub height: f64,
    /// Configuration of nodes per cell.
    #[serde(default)]
    pub cell: Cell,
}

impl Default for Domain {
    fn default() -> Self {
        Self {
            radius: 1.0,
            height: 1.0,

            cell: Cell::default(),
        }
    }
}

/// Options for size and order of cells on mesh.
#[derive(Serialize, Deserialize, Clone)]
pub struct Cell {
    pub subdivisions: usize,
    pub ghost: usize,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            subdivisions: 6,
            ghost: 3,
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
    Brill { amplitude: f64, sigma: (f64, f64) },
    #[serde(rename = "scalar_field")]
    ScalarField {
        amplitude: f64,
        sigma: (f64, f64),
        mass: f64,
    },
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Regrid {
    /// Any cell with error smaller than this will be coarsened.
    pub coarsen_tolerance: f64,
    /// Any cell with error larger than this will be refined.
    pub refine_tolerance: f64,
    /// How many steps do we take between checking?
    pub flag_interval: usize,
    /// Maximum number of levels allowed.
    pub max_levels: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Visualize {
    /// How often do we save a visualization?
    pub save_interval: f64,
    pub stride: usize,
}
