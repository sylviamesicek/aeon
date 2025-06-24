use crate::eqs::GaugeCondition;
use crate::run::interval::Interval;
use crate::run::status::Strategy;
use aeon::mesh::ExportStride;
use aeon_app::config::{VarDefs, FloatVar, Transform};
use eyre::eyre;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

mod execute;
mod inline;
mod validate;

pub use execute::*;
pub use inline::*;

/// Global configuration struct for simulation run.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "default_output")]
    pub directory: String,
    /// Are we simply running simulations or doing a critical search?
    #[serde(default)]
    pub execution: Execution,
    /// Order of stencil used to approximate derivatives
    pub order: usize,
    /// Order of stencil used to approximate dissipation.
    pub diss_order: usize,

    pub domain: Domain,
    pub limits: Limits,
    pub initial: Initial,
    pub evolve: Evolve,

    // *******************
    // Default fields
    // *******************
    /// Details for saving visualization files.
    #[serde(default)]
    pub visualize: Visualize,
    /// Details for saving cache files.
    #[serde(default)]
    pub cache: Cache,
    /// Details for handling errors.
    #[serde(default)]
    pub error_handler: ErrorHandler,
    #[serde(default)]
    pub horizon: Horizon,

    #[serde(default)]
    pub sources: Vec<Source>,
}

impl Config {
    /// Applies variable transformation to `Config`.
    pub fn transform(self, vars: &VarDefs) -> eyre::Result<Self> {
        Ok(Self {
            name: self.name.transform(vars)?,
            directory: self.directory.transform(vars)?,

            execution: self.execution.transform(vars)?,

            order: self.order,
            diss_order: self.diss_order,

            domain: self.domain,
            initial: self.initial,
            evolve: self.evolve,
            limits: self.limits,
            visualize: self.visualize,
            cache: self.cache,
            error_handler: self.error_handler,
            horizon: self.horizon,

            sources: self.sources.transform(vars)?,
        })
    }

    /// Retrieves output_director in absolution form.
    pub fn output_dir(&self) -> eyre::Result<PathBuf> {
        Ok(aeon_app::file::abs_or_relative(Path::new(&self.directory))?)
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
    /// Number of global refinements to perform when creating mesh.
    pub global_refine: usize,
}

/// Global limits before simulation fails
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Limits {
    /// Maximum number of levels allowed during refinement.
    pub max_levels: usize,
    /// Maximum number of nodes allowed before program crashes.
    pub max_nodes: usize,
    /// Maximum amount of RAM avalable before the program crashes
    pub max_memory: usize,
}

/// Settings for initial data solving.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Initial {
    /// Relaxation settings for solving initial data.
    pub relax: Relax,
    /// Error threshold before triggering refinement.
    pub refine_error: f64,
    /// Error minimum before triggering coarsening.
    pub coarsen_error: f64,
}

/// Evolution settings for running evolution.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Evolve {
    /// CFL factor for evolution
    pub cfl: f64,
    /// Amount of Kriss-Olgier Dissipation to use
    pub dissipation: f64,
    /// Maximum amount of coordinate time to run (before assuming disspersion).
    pub max_coord_time: f64,
    /// Maximum amount of proper time to run (before assuming disspersion).
    pub max_proper_time: f64,
    /// Maximum number of steps of evolution
    pub max_steps: usize,
    /// Gauge condition to use when evolving data.
    pub gauge: GaugeCondition,
    /// Error threshold before triggering refinement.
    pub refine_error: f64,
    /// Error minimum before triggering coarsening.
    pub coarsen_error: f64,
    /// How often should we regrid the mesh?
    pub regrid_interval: Interval,
}

/// Visualization settings for initial data and evolution output.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Visualize {
    /// Should we save the final result for initial data?
    pub initial: bool,
    /// Should we save the final result for relaxing each leve?
    pub initial_levels: bool,
    /// Should we save relaxation iterations.
    pub initial_relax: bool,
    /// How many iterations between each save?
    pub initial_relax_interval: Interval,

    /// Should we save evolution data?
    pub evolve: bool,
    /// How often do we save a visualization?
    pub evolve_interval: Interval,

    /// Should we save horizon search data
    pub horizon_relax: bool,
    /// How often do we save a horizon visualization
    pub horizon_relax_interval: Interval,

    /// How much data to poutput?
    pub stride: ExportStride,
}

impl Default for Visualize {
    fn default() -> Self {
        Visualize {
            evolve: false,
            evolve_interval: Interval::default(),
            initial: false,
            initial_levels: false,
            initial_relax: false,
            initial_relax_interval: Interval::default(),
            horizon_relax: false,
            horizon_relax_interval: Interval::default(),
            stride: ExportStride::PerVertex,
        }
    }
}

/// Config struct describing how we cache data.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Cache {
    pub initial: bool,
    /// Should we cache evolution
    pub evolve: bool,
    /// How often do we cache evolution
    pub evolve_interval: Interval,
}

impl Default for Cache {
    fn default() -> Self {
        Self {
            initial: false,
            evolve: false,
            evolve_interval: Default::default(),
        }
    }
}

/// How should we handle possible errors in the code?
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct ErrorHandler {
    pub on_max_levels: Strategy,
    pub on_max_nodes: Strategy,
    pub on_max_memory: Strategy,
    pub on_max_initial_steps: Strategy,
    pub on_max_evolve_steps: Strategy,
    pub on_max_evolve_coord_time: Strategy,
    pub on_max_evolve_proper_time: Strategy,
    pub on_norm_diverge: Strategy,
}

impl Default for ErrorHandler {
    fn default() -> Self {
        Self {
            on_max_levels: Strategy::Ignore,
            on_max_nodes: Strategy::Collapse,
            on_max_memory: Strategy::Collapse,
            on_max_initial_steps: Strategy::Crash,
            on_max_evolve_steps: Strategy::Collapse,
            on_max_evolve_coord_time: Strategy::Disperse,
            on_max_evolve_proper_time: Strategy::Disperse,
            on_norm_diverge: Strategy::Collapse,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Horizon {
    /// Should we search for apparent horizons?
    pub search: bool,
    /// If so, how often?
    pub search_interval: Interval,
    /// What should our first guess for radius be?
    pub search_initial_radius: f64,
    /// Number of global refinements of surface mesh.
    pub global_refine: usize,
    /// Settings for search hyperbolic relaxtion.
    pub relax: Relax,

    pub on_max_search_steps: Strategy,
    pub on_search_not_contained: Strategy,
    pub on_search_diverged: Strategy,
    pub on_search_converged: Strategy,
    pub on_search_converged_to_zero: Strategy,
}

impl Default for Horizon {
    fn default() -> Self {
        Self {
            search: false,
            search_interval: Interval::default(),
            search_initial_radius: 1.0,
            global_refine: 0,
            relax: Relax::default(),

            on_max_search_steps: Strategy::Ignore,
            on_search_not_contained: Strategy::Ignore,
            on_search_diverged: Strategy::Ignore,
            on_search_converged: Strategy::Collapse,
            on_search_converged_to_zero: Strategy::Ignore,
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

    fn transform(
        &self,
        vars: &VarDefs,
    ) -> Result<Self::Output, aeon_app::config::TransformError> {
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

fn default_name() -> String {
    "axi_sim".to_string()
}

fn default_output() -> String {
    "output".to_string()
}
