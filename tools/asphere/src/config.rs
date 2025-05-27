use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "default_directory")]
    pub directory: String,

    pub domain: Domain,
    pub limits: Limits,
    pub evolve: Evolve,
    pub regrid: Regrid,
    pub visualize: Visualize,
    pub diagnostic: Diagnostic,

    pub sources: Vec<Source>,
}

impl Config {
    /// Retrieves output_director in absolution form.
    pub fn directory(&self) -> eyre::Result<PathBuf> {
        crate::misc::abs_or_relative(Path::new(&self.directory))
    }
}

/// Settings deciding domain of the mesh.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Domain {
    /// Size of domain along the r axis.
    pub radius: f64,
}

/// Evolution settings for running evolution.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Evolve {
    /// CFL factor for evolution
    pub cfl: f64,
    /// Amount of Kriss-Olgier Dissipation to use
    pub dissipation: f64,
    /// Maximum amount of proper time to run (before assuming disspersion).
    pub max_proper_time: f64,
    /// Maximum number of steps
    pub max_steps: usize,
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
    pub save_initial: bool,
    pub save_initial_levels: bool,
    /// Should we save evolution data?
    pub save_evolve: bool,
    /// How often do we save a visualization?
    #[serde(default = "default_onef")]
    pub save_evolve_interval: f64,
    /// Stride for saving visualizations.
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Diagnostic {
    pub save: bool,
    pub save_interval: usize,
    pub serial_id: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Source {
    pub amplitude: f64,
    pub sigma: f64,
    pub mass: f64,
}

impl Source {
    pub fn println(&self) {
        if self.mass.abs() == 0.0 {
            println!(
                "- Massless Scalar Field: A = {}, σ = {}",
                self.amplitude, self.sigma,
            );
        } else {
            println!(
                "- Massive Scalar Field: A = {}, σ = {}, m = {}",
                self.amplitude, self.sigma, self.mass,
            );
        }
    }
}

// *****************************
// Defaults for serialization

// Why serde doesn't have a better way to do this
// inline, idk.

fn default_name() -> String {
    "axi_sim".to_string()
}

fn default_directory() -> String {
    "output".to_string()
}

fn _default_one() -> usize {
    1
}

fn default_onef() -> f64 {
    1.0
}

// *****************************
