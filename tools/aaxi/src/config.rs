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
    pub source: Vec<Source>,
}

impl Config {
    pub fn apply_args(self, args: &[&str]) -> eyre::Result<Self> {
        let mut sources = Vec::with_capacity(self.source.len());
        for source in self.source {
            sources.push(source.apply_args(args)?);
        }

        Ok(Self {
            name: template_str_apply_args(&self.name, args)?,
            output: template_str_apply_args(&self.output, args)?,

            order: self.order,
            diss_order: self.diss_order,

            domain: self.domain,
            relax: self.relax,
            evolve: self.evolve,
            limits: self.limits,
            regrid: self.regrid,
            visualize: self.visualize,

            source: sources,
        })
    }
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
    fn apply_args(self, args: &[&str]) -> eyre::Result<Self> {
        Ok(FPos::F64(match self {
            FPos::F64(v) => v,
            FPos::Pos(pos) => {
                let trans = template_str_apply_args(&pos, args)?;
                trans
                    .parse::<f64>()
                    .context("failed to parse string as float")?
            }
        }))
    }

    pub fn as_f64(&self) -> f64 {
        let Self::F64(v) = self else {
            panic!("failed to unwrap FPos");
        };

        *v
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type")]
pub enum Source {
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

impl Source {
    fn apply_args(self, args: &[&str]) -> eyre::Result<Self> {
        Ok(match self {
            Self::Brill { amplitude, sigma } => Self::Brill {
                amplitude: amplitude.apply_args(args)?,
                sigma: (sigma.0.apply_args(args)?, sigma.1.apply_args(args)?),
            },
            Self::ScalarField {
                amplitude,
                sigma,
                mass,
            } => Self::ScalarField {
                amplitude: amplitude.apply_args(args)?,
                sigma: (sigma.0.apply_args(args)?, sigma.1.apply_args(args)?),
                mass: mass.apply_args(args)?,
            },
        })
    }

    pub fn println(&self) {
        match self {
            Source::Brill { amplitude, sigma } => {
                println!(
                    "- Gunlach seed function: A = {}, σ = ({}, {})",
                    amplitude.as_f64(),
                    sigma.0.as_f64(),
                    sigma.1.as_f64()
                );
            }
            Source::ScalarField {
                amplitude,
                sigma,
                mass,
            } => {
                if mass.as_f64().abs() == 0.0 {
                    println!(
                        "- Massless Scalar Field: A = {}, σ = ({}, {})",
                        amplitude.as_f64(),
                        sigma.0.as_f64(),
                        sigma.1.as_f64()
                    );
                } else {
                    println!(
                        "- Massive Scalar Field: A = {}, σ = ({}, {}), m = {}",
                        amplitude.as_f64(),
                        sigma.0.as_f64(),
                        sigma.1.as_f64(),
                        mass.as_f64()
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

fn template_str_apply_args(data: &str, args: &[&str]) -> eyre::Result<String> {
    let mut result = String::new();
    let mut current = String::new();
    let mut searching = false;

    let mut chars = data.char_indices();

    while let Some((pos, ch)) = chars.next() {
        if searching {
            if data[pos..].starts_with('}') {
                searching = false;
                let index = current.parse::<usize>()?;

                if args.len() <= index as usize {
                    return Err(eyre!(
                        "{}th positional argument referenced in file but {} positional arguments were provided",
                        index,
                        args.len()
                    ));
                }

                result.push_str(args[index as usize]);
            }

            current.push(ch);

            continue;
        }

        if data[pos..].starts_with("${") {
            // Reset current search
            current.clear();

            let next = chars.next();
            debug_assert_eq!(next.map(|(_, c)| c), Some('{'));

            searching = true;

            continue;
        }

        if data[pos..].starts_with('$') {
            let (_, index) = chars
                .next()
                .ok_or_else(|| eyre!("invalid argument string"))?;
            let index: u32 = index
                .to_digit(10)
                .ok_or_else(|| eyre!("invalid argument string"))?;

            if args.len() <= index as usize {
                return Err(eyre!(
                    "{}th positional argument referenced in file but {} positional arguments were provided",
                    index,
                    args.len()
                ));
            }

            result.push_str(args[index as usize]);

            continue;
        }

        result.push(ch);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_str() {
        let args = vec!["0.0", "hello world", "&&@"];

        assert_eq!(
            template_str_apply_args("test$0123", &args).unwrap(),
            "test0.0123"
        );
        assert_eq!(
            template_str_apply_args("test${1}123", &args).unwrap(),
            "testhello world123"
        );
        assert_eq!(template_str_apply_args("$2tst", &args).unwrap(), "&&@tst");
    }
}
