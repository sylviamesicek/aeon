//! Common utilities used by aeon-based applications.
//!
//! Includes utils for loading and unloading toml config files,
//! applying bash like transformations to strings and similar datatypes,
//! styles for progress bars (to keep styling consistent), and common
//! floating operations (encode, decode, log_range, lin_range).

pub mod config;
pub mod file;
pub mod float;
pub mod progress;
