//! Helper command for supplying default config values compatible with
//! Cole's evolution searching code.

use crate::config::{
    Config, Diagnostic, Domain, Evolve, Limits, Regrid, Source, Stride, Visualize,
};

const MAX_LEVELS: usize = 21;
const MAX_STEPS: usize = 800_000;
const MAX_PROPER_TIME: f64 = 7.6;
const SIGMA: f64 = 5.35;
const RADIUS: f64 = 40.0;
const REFINE_GLOBAL: usize = 2;
const MAX_NODES: usize = 10_000_000;
const REFINE_ERROR: f64 = 1e-8;
const MAX_MEMORY: usize = 5_000_000_000;
const COARSEN_ERROR: f64 = 1e-10;
const DISSIPATION: f64 = 0.5;
const REGRID_FLAG_INTERVAL: usize = 20;
const CFL: f64 = 0.3;
const DIAGNOSTIC_INTERVAL: usize = 1;

// Generates a configuration compatible with cole's critical searching code.
pub fn cole_config(amplitude: f64, serial: usize) -> Config {
    Config {
        name: format!("cole-{}", serial),
        directory: ".".to_string(),
        domain: Domain { radius: RADIUS },
        limits: Limits {
            max_levels: MAX_LEVELS,
            max_nodes: MAX_NODES,
            max_memory: MAX_MEMORY,
        },
        evolve: Evolve {
            cfl: CFL,
            dissipation: DISSIPATION,
            max_proper_time: MAX_PROPER_TIME,
            max_steps: MAX_STEPS,
        },
        regrid: Regrid {
            refine_error: REFINE_ERROR,
            coarsen_error: COARSEN_ERROR,
            global: REFINE_GLOBAL,
            flag_interval: REGRID_FLAG_INTERVAL,
        },
        visualize: Visualize {
            save_initial: false,
            save_initial_levels: false,
            save_evolve: false,
            save_evolve_interval: 0.0,
            stride: Stride::PerVertex,
        },
        diagnostic: Diagnostic {
            save: true,
            save_interval: DIAGNOSTIC_INTERVAL.into(),
            serial_id: serial.into(),
        },
        sources: vec![Source {
            amplitude: amplitude.into(),
            sigma: SIGMA.into(),
            mass: 0.0.into(),
        }],
    }
}
