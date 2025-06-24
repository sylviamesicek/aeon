use aeon_app::{
    config::{FloatVar, Transform, TransformError, VarDefs},
    file, float,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub directory: String,
    pub parameter: String,
    pub pstar: FloatVar,
    pub start: FloatVar,
    pub end: FloatVar,
    pub samples: Samples,
}

impl Config {
    /// Finds absolute value of search directory as provided by the search.directory element
    /// in the toml fiile.
    pub fn fill_dir(&self) -> eyre::Result<PathBuf> {
        Ok(file::abs_or_relative(Path::new(&self.directory))?)
    }

    pub fn try_for_each<E, F: FnMut(f64) -> Result<(), E>>(&self, mut f: F) -> Result<(), E> {
        let pstar = self.pstar.unwrap();
        let start = self.start.unwrap();
        let end = self.end.unwrap();

        match self.samples {
            Samples::Log { log } => {
                assert!(start > 0.0 && end > 0.0);
                for amplitude in float::log_range(start, end, log) {
                    f(pstar + amplitude)?
                }
            }
            Samples::Linear { linear } => {
                for amplitude in float::lin_range(start, end, linear) {
                    f(pstar + amplitude)?
                }
            }
        }

        Ok(())
    }
}

impl Transform for Config {
    type Output = Self;

    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError> {
        Ok(Config {
            directory: self.directory.clone(),
            parameter: self.parameter.clone(),
            samples: self.samples.clone(),
            pstar: self.pstar.transform(vars)?,
            start: self.start.transform(vars)?,
            end: self.end.transform(vars)?,
        })
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(untagged)]
pub enum Samples {
    Log { log: usize },
    Linear { linear: usize },
}
