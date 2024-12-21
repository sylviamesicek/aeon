//! This crate contains general configuration and paramter data types used by critgen, idgen, and evgen.
//! These types are shared across crates, and thus moved here to prevent redundent definition.

use std::path::Path;

use aeon::{macros::SystemLabel, system::System};
use anyhow::{anyhow, Result};
use clap::ArgMatches;
use serde::{de::DeserializeOwned, Serialize};

mod config;

pub use config::*;

#[derive(Clone, Default)]
pub struct Fields;

impl System for Fields {
    const NAME: &'static str = "Fields";

    type Label = Field;

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        MetricSystem
            .enumerate()
            .map(Field::Metric)
            .chain(GaugeSystem.enumerate().map(Field::Gauge))
            .chain(ConstraintSystem.enumerate().map(Field::Constraint))
    }

    fn count(&self) -> usize {
        MetricSystem.count() + GaugeSystem.count() + ConstraintSystem.count()
    }

    fn label_from_index(&self, index: usize) -> Self::Label {
        if index < MetricSystem.count() {
            Field::Metric(MetricSystem.label_from_index(index))
        } else if index < MetricSystem.count() + GaugeSystem.count() {
            Field::Gauge(GaugeSystem.label_from_index(index - MetricSystem.count()))
        } else {
            Field::Constraint(
                ConstraintSystem
                    .label_from_index(index - MetricSystem.count() - GaugeSystem.count()),
            )
        }
    }

    fn label_index(&self, label: Self::Label) -> usize {
        match label {
            Field::Metric(metric) => MetricSystem.label_index(metric),
            Field::Gauge(gauge) => GaugeSystem.label_index(gauge) + MetricSystem.count(),
            Field::Constraint(constraints) => {
                ConstraintSystem.label_index(constraints)
                    + MetricSystem.count()
                    + GaugeSystem.count()
            }
        }
    }

    fn label_name(&self, label: Self::Label) -> String {
        match label {
            Field::Metric(metric) => MetricSystem.label_name(metric),
            Field::Gauge(gauge) => GaugeSystem.label_name(gauge),
            Field::Constraint(constraints) => ConstraintSystem.label_name(constraints),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Field {
    Metric(Metric),
    Gauge(Gauge),
    Constraint(Constraint),
}

#[derive(Clone, Copy, SystemLabel)]
pub enum Metric {
    Grr,
    Grz,
    Gzz,
    S,
    Krr,
    Krz,
    Kzz,
    Y,
}

#[derive(Clone, Copy, SystemLabel)]
pub enum Gauge {
    Lapse,
    Shiftr,
    Shiftz,
}

#[derive(Clone, Copy, SystemLabel)]
pub enum Constraint {
    Theta,
    Zr,
    Zz,
}

/// Exports a config structure to a toml file.
pub fn export_to_toml<T: Serialize>(path: impl AsRef<Path>, config: T) -> Result<()> {
    let string = toml::to_string_pretty(&config)?;
    std::fs::write(path, string)?;

    Ok(())
}

/// Loads a config from a toml file given a path.
pub fn import_from_toml<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T> {
    let string = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&string)?)
}

/// Loads a config from the toml file pointed to by the 'path' argument in `ArgMatches`.
pub fn import_from_path_arg<T: DeserializeOwned>(matches: &ArgMatches) -> Result<T> {
    // Get path argument
    let path = matches
        .get_one::<String>("path")
        .ok_or(anyhow!("Failed to specify path argument"))?
        .clone();

    import_from_toml(path)
}
