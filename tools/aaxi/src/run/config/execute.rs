use std::path::{Path, PathBuf};

use aeon_config::{ConfigVars, FloatVar, Transform};
use serde::{Deserialize, Serialize};

use crate::misc;

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
