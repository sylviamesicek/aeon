use crate::misc;
use aeon_config::{ConfigVars, FloatVar, Transform};
use eyre::eyre;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Global configuration struct for simulation run.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "default_output")]
    pub output: String,
    /// Are we simply running simulations or doing a critical search?
    #[serde(default)]
    pub execution: Execution,
    /// Order of stencil used to approximate derivatives
    pub order: usize,
    /// Order of stencil used to approximate dissipation.
    pub diss_order: usize,

    pub domain: Domain,
    pub relax: Relax,
    pub evolve: Evolve,
    pub limits: Limits,
    pub regrid: Regrid,
    pub visualize: Visualize,
    #[serde(default)]
    pub cache: Cache,
    #[serde(default)]
    pub horizon: Horizon,

    #[serde(default)]
    pub sources: Vec<Source>,
}

impl Config {
    /// Applies variable transformation to `Config`.
    pub fn transform(self, vars: &ConfigVars) -> eyre::Result<Self> {
        Ok(Self {
            name: self.name.transform(vars)?,
            output: self.output.transform(vars)?,

            execution: self.execution.transform(vars)?,

            order: self.order,
            diss_order: self.diss_order,

            domain: self.domain,
            relax: self.relax,
            evolve: self.evolve,
            limits: self.limits,
            regrid: self.regrid,
            visualize: self.visualize,
            cache: self.cache,
            horizon: self.horizon,

            sources: self.sources.transform(vars)?,
        })
    }

    /// Retrieves output_director in absolution form.
    pub fn output_dir(&self) -> eyre::Result<PathBuf> {
        misc::abs_or_relative(Path::new(&self.output))
    }

    pub fn search_dir(&self) -> eyre::Result<PathBuf> {
        let Execution::Search { search } = &self.execution else {
            return Err(eyre!("program is not running in search mode"));
        };

        search.search_dir()
    }

    pub fn search_config(&self) -> Option<&Search> {
        match &self.execution {
            Execution::Search { search } => Some(search),
            _ => None,
        }
    }

    pub fn _is_search_mode(&self) -> bool {
        matches!(self.execution, Execution::Search { .. })
    }

    pub fn _is_run_mode(&self) -> bool {
        matches!(self.execution, Execution::Run)
    }
}

/// Settings deciding domain of the mesh.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Domain {
    /// Size of domain along the ρ axis.
    pub radius: f64,
    /// Size of domain along the z-axis.
    pub height: f64,
    /// Number of nodal subdivisions per cell axis.
    pub cell_size: usize,
    /// Number of ghost nodes on edge of cell.
    pub cell_ghost: usize,
}

/// Relaxation settings for solving initial data.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Relax {
    pub max_steps: usize,
    pub error_tolerance: f64,
    pub cfl: f64,
    pub dampening: f64,
}

/// Evolution settings for running evolution.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Evolve {
    /// CFL factor for evolution
    pub cfl: f64,
    /// Amount of Kriss-Olgier Dissipation to use
    pub dissipation: f64,
    /// Maximum amount of coordinate time to run (before assuming disspersion).
    pub max_time: f64,
    /// Maximum amount of proper time to run (before assuming disspersion).
    pub max_proper_time: f64,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Gauge condition to use when evolving data.
    pub gauge: GaugeCondition,
}

/// Limits before a given simulation crashes (or assumes collapse)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Limits {
    /// Maximum number of levels allowed during refinement.
    pub max_levels: usize,
    /// Maximum number of nodes allowed before program crashes.
    pub max_nodes: usize,
    /// Maximum amount of RAM avalable before the program crashes
    pub max_memory: usize,
}

/// Settings for regriding mesh.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Regrid {
    /// Any cell with error larger than this will be refined.
    pub refine_error: f64,
    /// Any cell with error smaller than this will be coarsened.
    pub coarsen_error: f64,
    /// Number of global regrids to do before solving.
    pub global: usize,
    /// How many steps do we take between regridding runs?
    pub flag_interval: usize,
}

/// Visualization settings for initial data and evolution output.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Visualize {
    /// Should we save evolution data?
    #[serde(default)]
    pub save_evolve: bool,
    /// How often do we save a visualization?
    #[serde(default = "default_onef")]
    pub save_evolve_interval: f64,
    /// Should we save relaxation iterations.
    #[serde(default)]
    pub save_relax: bool,
    /// How many iterations between each save?
    #[serde(default = "default_one")]
    pub save_relax_interval: usize,
    /// Should we save the final result for relaxing each leve?
    #[serde(default)]
    pub save_relax_levels: bool,
    /// Should we save the final result?
    #[serde(default)]
    pub save_relax_result: bool,
    /// How much data to poutput?
    pub stride: Stride,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum Stride {
    /// Output data for every vertex in the simulation
    #[serde(rename = "per_vertex")]
    PerVertex,
    /// Output data for each corner of a cell in the simulation
    /// This is significantly more compressed
    #[serde(rename = "per_cell")]
    PerCell,
}

impl Stride {
    pub fn into_int(self) -> usize {
        match self {
            Stride::PerVertex => 1,
            Stride::PerCell => 0,
        }
    }
}

/// Config struct describing how we cache data.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Cache {
    pub initial: bool,
    /// Should we cache evolution
    pub evolve: bool,
    #[serde(default = "default_one")]
    pub evolve_interval: usize,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            initial: false,
            evolve: false,
            evolve_interval: 1,
        }
    }
}

/// What subproduct should be executed.
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
#[serde(tag = "mode")]
pub enum Execution {
    #[serde(rename = "run")]
    #[default]
    Run,
    #[serde(rename = "search")]
    Search {
        #[serde(flatten)]
        search: Search,
    },
}

impl Execution {
    /// Performs variable transformation on `Execution`.
    pub fn transform(self, vars: &ConfigVars) -> eyre::Result<Self> {
        Ok(match self {
            Execution::Run => Self::Run,
            Execution::Search { search } => Execution::Search {
                search: search.transform(vars)?,
            },
        })
    }
}

/// Search subcommand.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Search {
    pub directory: String,
    pub parameter: String,
    /// Start of range to search
    start: FloatVar,
    /// End of range to search
    end: FloatVar,
    /// How many levels of binary search before we quit?
    pub max_depth: usize,
    /// Finish search when end-start < min_error
    pub min_error: f64,
}

impl Search {
    /// Gets start of search range.
    pub fn start(&self) -> f64 {
        let FloatVar::F64(start) = self.start else {
            panic!("Search has not been properly transformed")
        };
        start
    }

    /// Gets end of search range.
    pub fn end(&self) -> f64 {
        let FloatVar::F64(end) = self.end else {
            panic!("Search has not been properly transformed")
        };
        end
    }

    /// Finds absolute value of search directory as provided by the search.directory element
    /// in the toml fiile.
    pub fn search_dir(&self) -> eyre::Result<PathBuf> {
        misc::abs_or_relative(Path::new(&self.directory))
    }
}

impl Transform for Search {
    type Output = Self;

    fn transform(&self, vars: &ConfigVars) -> Result<Self::Output, aeon_config::TransformError> {
        Ok(Self {
            directory: self.directory.transform(vars)?,
            parameter: self.parameter.transform(vars)?,
            start: self.start.transform(&vars)?,
            end: self.end.transform(&vars)?,
            max_depth: self.max_depth,
            min_error: self.min_error,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Horizon {
    search: bool,
    search_interval: usize,
}

impl Default for Horizon {
    fn default() -> Self {
        Self {
            search: false,
            search_interval: 1,
        }
    }
}

/// Source term for the simulated system.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum Source {
    /// Instance generates Brill-type initial data with gunlach seed function.
    #[serde(rename = "brill")]
    Brill {
        amplitude: FloatVar,
        sigma: (FloatVar, FloatVar),
    },
    /// An axisymmetric scalar field with the given sigma.
    #[serde(rename = "scalar_field")]
    ScalarField {
        amplitude: FloatVar,
        sigma: (FloatVar, FloatVar),
        mass: FloatVar,
    },
}

impl Source {
    pub fn println(&self) {
        match self {
            Source::Brill { amplitude, sigma } => {
                println!(
                    "- Gunlach seed function: A = {}, σ = ({}, {})",
                    amplitude.unwrap(),
                    sigma.0.unwrap(),
                    sigma.1.unwrap()
                );
            }
            Source::ScalarField {
                amplitude,
                sigma,
                mass,
            } => {
                if mass.unwrap().abs() == 0.0 {
                    println!(
                        "- Massless Scalar Field: A = {}, σ = ({}, {})",
                        amplitude.unwrap(),
                        sigma.0.unwrap(),
                        sigma.1.unwrap()
                    );
                } else {
                    println!(
                        "- Massive Scalar Field: A = {}, σ = ({}, {}), m = {}",
                        amplitude.unwrap(),
                        sigma.0.unwrap(),
                        sigma.1.unwrap(),
                        mass.unwrap()
                    );
                }
            }
        }
    }
}

impl Transform for Source {
    type Output = Self;

    fn transform(&self, vars: &ConfigVars) -> Result<Self::Output, aeon_config::TransformError> {
        Ok(match self {
            Self::Brill { amplitude, sigma } => Self::Brill {
                amplitude: amplitude.transform(vars)?,
                sigma: (sigma.0.transform(vars)?, sigma.1.transform(vars)?),
            },
            Self::ScalarField {
                amplitude,
                sigma,
                mass,
            } => Self::ScalarField {
                amplitude: amplitude.transform(vars)?,
                sigma: (sigma.0.transform(vars)?, sigma.1.transform(vars)?),
                mass: mass.transform(vars)?,
            },
        })
    }
}

/// Gauge condition to use for evolution.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum GaugeCondition {
    /// Pure generalized harmonic gauge conditions.
    #[default]
    #[serde(rename = "harmonic")]
    Harmonic,
    /// Generalized harmonic gauge with no shift.
    #[serde(rename = "harmonic_zero_shift")]
    HarmonicZeroShift,
    /// Log + 1 slicing with no shift.
    #[serde(rename = "log_plus_one_zero_shift")]
    LogPlusOneZeroShift,
    /// Log + 1 slicing with harmonic shift.
    #[serde(rename = "log_plus_one")]
    LogPlusOne,
}

fn default_name() -> String {
    "axi_sim".to_string()
}

fn default_output() -> String {
    "output".to_string()
}

fn default_one() -> usize {
    1
}

fn default_onef() -> f64 {
    1.0
}
