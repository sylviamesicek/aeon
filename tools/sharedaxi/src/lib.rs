//! This crate contains general configuration and paramter data types used by critgen, idgen, and evgen.
//! These types are shared across crates, and thus moved here to prevent redundent definition.

use std::path::Path;

use aeon::{basis::RadiativeParams, macros::SystemLabel, prelude::*, system::System};
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

// pub enum Source {
//     ScalarField { amplitude: f64, sigma: (f64, f64) },
// }

#[derive(Clone)]
pub struct Quadrant;

impl Boundary<2> for Quadrant {
    fn kind(&self, face: Face<2>) -> BoundaryKind {
        match face.side {
            true => BoundaryKind::Radiative,
            false => BoundaryKind::Parity,
        }
    }
}

#[derive(Clone)]
pub struct FieldConditions;

impl Conditions<2> for FieldConditions {
    type System = Fields;

    fn parity(&self, field: Field, face: Face<2>) -> bool {
        let axes = match field {
            Field::Metric(Metric::Grr) | Field::Metric(Metric::Krr) => [true, true],
            Field::Metric(Metric::Grz) | Field::Metric(Metric::Krz) => [false, false],
            Field::Metric(Metric::Gzz) | Field::Metric(Metric::Kzz) => [true, true],
            Field::Metric(Metric::S) | Field::Metric(Metric::Y) => [false, true],

            Field::Constraint(Constraint::Theta) | Field::Gauge(Gauge::Lapse) => [true, true],
            Field::Constraint(Constraint::Zr) | Field::Gauge(Gauge::Shiftr) => [false, true],
            Field::Constraint(Constraint::Zz) | Field::Gauge(Gauge::Shiftz) => [true, false],
        };
        axes[face.axis]
    }

    fn radiative(&self, field: Field, _position: [f64; 2], _spacing: f64) -> RadiativeParams {
        match field {
            Field::Metric(Metric::Grr)
            | Field::Metric(Metric::Gzz)
            | Field::Gauge(Gauge::Lapse) => RadiativeParams::lightlike(1.0),
            _ => RadiativeParams::lightlike(0.0),
        }
    }
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
