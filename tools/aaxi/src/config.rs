use eyre::{Context, eyre};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_name")]
    pub name: String,
    #[serde(default = "default_output")]
    pub output: String,

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
    pub source: Vec<SourcePos>,
}

fn default_name() -> String {
    "axi_sim".to_string()
}

fn default_output() -> String {
    "output".to_string()
}

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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Relax {
    pub max_steps: usize,
    pub error_tolerance: f64,
    pub cfl: f64,
    pub dampening: f64,
}

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
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Limits {
    /// Maximum number of levels allowed during refinement.
    pub max_levels: usize,
    /// Maximum number of nodes allowed before program crashes.
    pub max_nodes: usize,
}

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

#[derive(Serialize, Deserialize, Debug)]
pub struct Visualize {
    /// How often do we save a visualization?
    #[serde(default)]
    pub save_evolve_interval: Option<f64>,
    #[serde(default)]
    pub save_relax_levels: bool,
    #[serde(default)]
    pub save_relax_interval: Option<usize>,
    #[serde(default)]
    pub save_relax_result: bool,
    /// Stride for saving visualizations.
    pub stride: usize,
}

/// A floating point argument that can either be provided via a configuration file
/// or as a positional argument in the cli.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum FPos {
    /// Fixed floating point input.
    F64(f64),
    /// Positional argument passed to cli.
    Pos(String),
}

impl FPos {
    pub fn parse(&self, args: &[&str]) -> eyre::Result<f64> {
        match self {
            FPos::F64(v) => Ok(*v),
            FPos::Pos(pos) => {
                if !pos.starts_with('$') {
                    return Err(eyre!("positional arg references must begin with $"));
                }

                let index = pos[1..]
                    .parse::<usize>()
                    .context("failed to parse positional argument index")?;

                if args.len() <= index {
                    return Err(eyre!(
                        "{}th positional argument referenced in file but {} positional arguments were provided",
                        index,
                        args.len()
                    ));
                }

                Ok(args[index]
                    .parse::<f64>()
                    .context("failed to parse positional argument as float")?)
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum SourcePos {
    /// Instance generates Brill-type initial data with gunlach seed function.
    #[serde(rename = "brill")]
    Brill {
        amplitude: FPos,
        sigma: (FPos, FPos),
    },
    /// An axisymmetric scalar field with the given sigma.
    #[serde(rename = "scalar_field")]
    ScalarField {
        amplitude: FPos,
        sigma: (FPos, FPos),
        mass: FPos,
    },
}

/// Transformed source data for aaxi.
#[derive(Clone, Debug)]
pub enum Source {
    /// Instance generates Brill-type initial data with gunlach seed function.
    Brill { amplitude: f64, sigma: (f64, f64) },
    /// An axisymmetric scalar field with the given sigma.
    ScalarField {
        amplitude: f64,
        sigma: (f64, f64),
        mass: f64,
    },
}

impl Source {
    pub fn from_pos(pos: &SourcePos, args: &[&str]) -> eyre::Result<Self> {
        Ok(match pos {
            SourcePos::Brill { amplitude, sigma } => Source::Brill {
                amplitude: amplitude.parse(args)?,
                sigma: (sigma.0.parse(args)?, sigma.1.parse(args)?),
            },
            SourcePos::ScalarField {
                amplitude,
                sigma,
                mass,
            } => Source::ScalarField {
                amplitude: amplitude.parse(args)?,
                sigma: (sigma.0.parse(args)?, sigma.1.parse(args)?),
                mass: mass.parse(args)?,
            },
        })
    }

    pub fn println(&self) {
        match self {
            Source::Brill { amplitude, sigma } => {
                println!(
                    "- Gunlach seed function: A = {}, σ = ({}, {})",
                    amplitude, sigma.0, sigma.1
                );
            }
            Source::ScalarField {
                amplitude,
                sigma,
                mass,
            } => {
                if mass.abs() == 0.0 {
                    println!(
                        "- Massless Scalar Field: A = {}, σ = ({}, {})",
                        amplitude, sigma.0, sigma.1
                    );
                } else {
                    println!(
                        "- Massive Scalar Field: A = {}, σ = ({}, {}), m = {}",
                        amplitude, sigma.0, sigma.1, mass
                    );
                }
            }
        }
    }
}

/// Gauge condition to use for evolution.
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum GaugeCondition {
    #[default]
    #[serde(rename = "harmonic")]
    Harmonic,
    #[serde(rename = "zero_shift")]
    ZeroShift,
}
