//! This crate contains general configuration and paramter data types used by critgen, idgen, and evgen.
//! These types are shared across crates, and thus moved here to prevent redundent definition.

mod eqs;
mod evolve;
mod horizon;
mod initial;
mod systems;

pub use evolve::evolve_data;
pub use initial::initial_data;
pub use systems::*;
