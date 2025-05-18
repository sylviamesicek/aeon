use crate::{
    misc,
    transform::{ConfigVars, transform},
};
use eyre::{Context, eyre};
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
    pub source: Vec<Source>,
}

impl Config {
    /// Applies variable transformation to `Config`.
    pub fn transform(self, vars: &ConfigVars) -> eyre::Result<Self> {
        Ok(Self {
            name: transform(&self.name, vars)?,
            output: transform(&self.output, vars)?,

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

            source: self
                .source
                .into_iter()
                .map(|source| source.transform(vars))
                .collect::<Result<_, _>>()?,
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
    /// Stride for saving visualizations.
    pub stride: usize,
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
    pub parameter_key: String,
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
    pub fn transform(self, vars: &ConfigVars) -> eyre::Result<Self> {
        Ok(Self {
            directory: transform(&self.directory, vars)?,
            parameter_key: transform(&self.parameter_key, vars)?,
            start: self.start.transform(&vars)?,
            end: self.end.transform(&vars)?,
            max_depth: self.max_depth,
            min_error: self.min_error,
        })
    }

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

/// A floating point argument that can either be provided via a configuration file
/// or as a positional argument in the cli.
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum FloatVar {
    /// Fixed floating point input.
    F64(f64),
    /// Script that will be parsed by the transformer
    Script(String),
}

impl FloatVar {
    fn transform(self, vars: &ConfigVars) -> eyre::Result<Self> {
        Ok(FloatVar::F64(match self {
            FloatVar::F64(v) => v,
            FloatVar::Script(pos) => transform(&pos, &vars)?
                .parse::<f64>()
                .context("failed to parse string as float")?,
        }))
    }

    /// Unwraps a float var into a float, assuming that it has already been transformed.
    pub fn unwrap(&self) -> f64 {
        let Self::F64(v) = self else {
            panic!("failed to unwrap FloatVar");
        };

        *v
    }
}

impl From<f64> for FloatVar {
    fn from(value: f64) -> Self {
        Self::F64(value)
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
    fn transform(self, vars: &ConfigVars) -> eyre::Result<Self> {
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
}

// fn template_str_apply_args(data: &str, args: &[&str]) -> eyre::Result<String> {
//     let mut result = String::new();
//     let mut current = String::new();
//     let mut searching = false;

//     let mut chars = data.char_indices();

//     while let Some((pos, ch)) = chars.next() {
//         if searching {
//             if data[pos..].starts_with('}') {
//                 searching = false;
//                 let index = current.parse::<usize>()?;

//                 if args.len() <= index as usize {
//                     return Err(eyre!(
//                         "{}th positional argument referenced in file but {} positional arguments were provided",
//                         index,
//                         args.len()
//                     ));
//                 }

//                 result.push_str(args[index as usize]);
//             }

//             current.push(ch);

//             continue;
//         }

//         if data[pos..].starts_with("${") {
//             // Reset current search
//             current.clear();

//             let next = chars.next();
//             debug_assert_eq!(next.map(|(_, c)| c), Some('{'));

//             searching = true;

//             continue;
//         }

//         if data[pos..].starts_with('$') {
//             let (_, index) = chars
//                 .next()
//                 .ok_or_else(|| eyre!("invalid argument string"))?;
//             let index: u32 = index
//                 .to_digit(10)
//                 .ok_or_else(|| eyre!("invalid argument string"))?;

//             if args.len() <= index as usize {
//                 return Err(eyre!(
//                     "{}th positional argument referenced in file but {} positional arguments were provided",
//                     index,
//                     args.len()
//                 ));
//             }

//             result.push_str(args[index as usize]);

//             continue;
//         }

//         result.push(ch);
//     }

//     Ok(result)
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

// #[test]
// fn template_str() {
//     let args = vec!["0.0", "hello world", "&&@"];

//     assert_eq!(
//         template_str_apply_args("test$0123", &args).unwrap(),
//         "test0.0123"
//     );
//     assert_eq!(
//         template_str_apply_args("test${1}123", &args).unwrap(),
//         "testhello world123"
//     );
//     assert_eq!(template_str_apply_args("$2tst", &args).unwrap(), "&&@tst");
// }
// }
