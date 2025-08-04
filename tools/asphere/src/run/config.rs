use aeon::mesh::ExportStride;
use aeon_app::config::{FloatVar, Transform, TransformError, VarDefs};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug)]
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

    #[serde(default)]
    pub diagnostic: Diagnostic,

    pub sources: Vec<Source>,
}

impl Config {
    /// Retrieves output_director in absolution form.
    pub fn directory(&self) -> eyre::Result<PathBuf> {
        let result = aeon_app::file::abs_or_relative(Path::new(&self.directory))?;
        Ok(result)
    }
}

impl Transform for Config {
    type Output = Self;

    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError> {
        Ok(Self {
            name: self.name.transform(vars)?,
            directory: self.directory.transform(vars)?,
            domain: self.domain.clone(),
            limits: self.limits.clone(),
            evolve: self.evolve.clone(),
            regrid: self.regrid.clone(),
            visualize: self.visualize.clone(),
            diagnostic: self.diagnostic.clone(),
            sources: self.sources.transform(vars)?,
        })
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
    /// Maximum violation of the momentum constraint before program assumes collapse.
    pub max_constraint: f64,
    /// Minimum lapse at origin before program assumes collapse.
    pub min_lapse: f64,
}

/// Limits before a given simulation crashes (or assumes collapse)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Limits {
    /// Maximum number of levels allowed during refinement.
    pub max_levels: usize,
    /// Maximum number of nodes allowed before program assumes collapse.
    pub max_nodes: usize,
    /// Maximum amount of RAM avalable before the program assumes collapse
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
    /// Fix the grid after a certain proper time?
    #[serde(default)]
    pub fix_grid: bool,
    /// At what proper time do we fix the grid?
    #[serde(default = "zero_f64")]
    pub fix_grid_time: f64,
    /// Within what radius do we fix the grid?
    #[serde(default = "zero_f64")]
    pub fix_grid_radius: f64,
    /// At what refinement level do we fix the grid?
    #[serde(default = "zero_usize")]
    pub fix_grid_level: usize,
}

fn zero_f64() -> f64 {
    0.0
}

fn zero_usize() -> usize {
    0
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
    pub stride: ExportStride,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Diagnostic {
    /// Should we save diagnostic info for evolution runs
    pub save_evolve: bool,
    /// How often do we save diagnostic info for evolution runs
    pub save_evolve_interval: usize,
}

impl Default for Diagnostic {
    fn default() -> Self {
        Self {
            save_evolve: false,
            save_evolve_interval: 1,
        }
    }
}

// #[derive(Serialize, Deserialize, Clone, Debug)]
// pub struct Source {
//     pub amplitude: FloatVar,
//     pub sigma: FloatVar,
//     pub mass: FloatVar,
// }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Source {
    pub mass: FloatVar,
    pub profile: ScalarFieldProfile,
    #[serde(default)]
    pub smooth: Smooth,
}

impl Source {
    pub fn println(&self) {
        match &self.profile {
            ScalarFieldProfile::Gaussian {
                amplitude,
                sigma,
                center,
            } => {
                if self.mass.unwrap().abs() == 0.0 {
                    println!(
                        "- Massless Scalar Field: Gaussian(A={}, σ={}, r₀={})",
                        amplitude.unwrap(),
                        sigma.unwrap(),
                        center.unwrap(),
                    );
                } else {
                    println!(
                        "- Massive Scalar Field: Gaussian(A={}, σ={}, r₀={}), m = {}",
                        amplitude.unwrap(),
                        sigma.unwrap(),
                        center.unwrap(),
                        self.mass.unwrap(),
                    );
                }
            }
            ScalarFieldProfile::TanH {
                amplitude,
                sigma,
                center,
            } => {
                if self.mass.unwrap().abs() == 0.0 {
                    println!(
                        "- Massless Scalar Field: Tanh(A={}, σ={}, r₀={})",
                        amplitude.unwrap(),
                        sigma.unwrap(),
                        center.unwrap(),
                    );
                } else {
                    println!(
                        "- Massive Scalar Field: Tanh(A={}, σ={}, r₀={}), m = {}",
                        amplitude.unwrap(),
                        sigma.unwrap(),
                        center.unwrap(),
                        self.mass.unwrap(),
                    );
                }
            }
        }
    }
}

impl Transform for Source {
    type Output = Self;

    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError> {
        Ok(Source {
            mass: self.mass.transform(vars)?,
            profile: self.profile.transform(vars)?,
            smooth: self.smooth.clone(),
        })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum ScalarFieldProfile {
    /// Initial profile of the form $ampl \exp{-(r - center)^2 / \sigma^2$.
    #[serde(rename = "gaussian")]
    Gaussian {
        amplitude: FloatVar,
        sigma: FloatVar,
        center: FloatVar,
    },
    /// Initial profile of the form $amp * \tanh{(r - center)/\sigma}$.
    #[serde(rename = "tanh")]
    TanH {
        amplitude: FloatVar,
        sigma: FloatVar,
        center: FloatVar,
    },
}

impl Transform for ScalarFieldProfile {
    type Output = Self;

    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError> {
        Ok(match self {
            ScalarFieldProfile::Gaussian {
                amplitude,
                sigma,
                center,
            } => Self::Gaussian {
                amplitude: amplitude.transform(vars)?,
                sigma: sigma.transform(vars)?,
                center: center.transform(vars)?,
            },
            ScalarFieldProfile::TanH {
                amplitude,
                sigma,
                center,
            } => Self::TanH {
                amplitude: amplitude.transform(vars)?,
                sigma: sigma.transform(vars)?,
                center: center.transform(vars)?,
            },
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Smooth {
    pub strength: f64,
    pub power: f64,
}

impl Default for Smooth {
    fn default() -> Self {
        Self {
            strength: 0.0,
            power: 1.0,
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
