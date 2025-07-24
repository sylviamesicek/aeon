use aeon_app::{
    config::{FloatVar, Transform, TransformError, VarDefs},
    file, float,
};
use rayon::{
    ThreadPoolBuilder,
    iter::{IntoParallelIterator as _, ParallelBridge, ParallelIterator},
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Config {
    pub directory: String,
    pub parameter: String,
    // How many simulations to run in parallel at a given time.
    pub parallel: usize,
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

    pub fn try_for_each<E: Send, F: Fn(f64) -> Result<(), E> + Send + Sync>(
        &self,
        f: F,
    ) -> Result<(), E> {
        let pstar = self.pstar.unwrap();
        let start = self.start.unwrap();
        let end = self.end.unwrap();

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(self.parallel)
            .build()
            .unwrap();

        thread_pool.install(|| {
            match self.samples {
                Samples::Log { log } => {
                    assert!(start > 0.0 && end > 0.0);

                    float::log_range(start, end, log)
                        .par_bridge()
                        .into_par_iter()
                        .try_for_each(|amplitude| f(pstar + amplitude))?;
                }
                Samples::Linear { linear } => {
                    float::lin_range(start, end, linear)
                        .par_bridge()
                        .into_par_iter()
                        .try_for_each(|amplitude| f(pstar + amplitude))?;
                }
            }

            Ok(())
        })
    }
}

impl Transform for Config {
    type Output = Self;

    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError> {
        Ok(Config {
            directory: self.directory.clone(),
            parameter: self.parameter.clone(),
            parallel: self.parallel,
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
