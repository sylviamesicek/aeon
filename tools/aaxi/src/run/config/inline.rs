//! Inline tables and types for use in config

use bincode::{Decode, Encode};
use serde::{Deserialize, Serialize};

/// Settings for hyperbolic relaxation.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, Encode, Decode)]
pub struct Relax {
    /// Ficticious cfl to use while relaxing initial data.
    pub cfl: f64,
    /// Ficticious dampening to stabilize relaxation.
    pub dampening: f64,
    /// Maximum steps before relaxation fails.
    pub max_steps: usize,
    /// Error threshold to reach before relaxation succeeds.
    pub tolerance: f64,
}

impl Default for Relax {
    fn default() -> Self {
        Self {
            cfl: 0.1,
            dampening: 0.4,
            max_steps: 10_000,
            tolerance: 1e-6,
        }
    }
}
