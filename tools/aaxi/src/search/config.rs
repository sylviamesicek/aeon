use aeon_app::config::{FloatVar, Transform, TransformError, VarDefs};
use aeon_app::file;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Search subcommand.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    pub directory: String,
    pub parameter: String,
    /// How many simulations do we run/process in parallel?
    pub parallel: usize,
    /// Start of range to search
    pub start: FloatVar,
    /// End of range to search
    pub end: FloatVar,
    /// How many levels of binary search before we quit?
    pub max_depth: usize,
    /// Finish search when end-start < min_error
    pub min_error: f64,
}

impl Config {
    /// Gets start of search range.
    pub fn start(&self) -> f64 {
        let FloatVar::Inline(start) = self.start else {
            panic!("Search has not been properly transformed")
        };
        start
    }

    /// Gets end of search range.
    pub fn end(&self) -> f64 {
        let FloatVar::Inline(end) = self.end else {
            panic!("Search has not been properly transformed")
        };
        end
    }

    /// Finds absolute value of search directory as provided by the search.directory element
    /// in the toml fiile.
    pub fn search_dir(&self) -> eyre::Result<PathBuf> {
        let result = file::abs_or_relative(Path::new(&self.directory))?;
        Ok(result)
    }
}

impl Transform for Config {
    type Output = Self;

    fn transform(&self, vars: &VarDefs) -> Result<Self::Output, TransformError> {
        Ok(Self {
            directory: self.directory.transform(vars)?,
            parameter: self.parameter.transform(vars)?,
            parallel: self.parallel,
            start: self.start.transform(&vars)?,
            end: self.end.transform(&vars)?,
            max_depth: self.max_depth,
            min_error: self.min_error,
        })
    }
}
