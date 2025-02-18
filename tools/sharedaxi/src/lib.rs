//! This crate contains general configuration and paramter data types used by critgen, idgen, and evgen.
//! These types are shared across crates, and thus moved here to prevent redundent definition.

use std::path::Path;

use aeon::{macros::SystemLabel, prelude::*, system::System};
use anyhow::{anyhow, Result};
use clap::ArgMatches;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

mod config;
mod eqs;

pub use config::*;

/// System for storing all fields necessary for axisymmetric evolution.
#[derive(Clone, Serialize, Deserialize)]
pub struct Fields {
    /// Scalar field masses.
    pub scalar_fields: Vec<f64>,
}

impl Fields {
    pub fn num_scalar_fields(&self) -> usize {
        self.scalar_fields.len()
    }

    pub fn scalar_fields(&self) -> impl Iterator<Item = f64> + '_ {
        self.scalar_fields.iter().cloned()
    }
}

impl System for Fields {
    const NAME: &'static str = "Fields";

    type Label = Field;

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        MetricSystem
            .enumerate()
            .map(Field::Metric)
            .chain(GaugeSystem.enumerate().map(Field::Gauge))
            .chain(ConstraintSystem.enumerate().map(Field::Constraint))
            .chain((0..self.scalar_fields.len()).flat_map(|i| {
                [ScalarField::Phi, ScalarField::Pi]
                    .into_iter()
                    .map(move |scalar_field| Field::ScalarField(scalar_field, i))
            }))
    }

    fn count(&self) -> usize {
        MetricSystem.count()
            + GaugeSystem.count()
            + ConstraintSystem.count()
            + ScalarFields(self.scalar_fields.len()).count()
    }

    fn label_from_index(&self, mut index: usize) -> Self::Label {
        // ************************
        // Metric *****************

        if index < MetricSystem.count() {
            return Field::Metric(MetricSystem.label_from_index(index));
        }
        index -= MetricSystem.count();

        if index < GaugeSystem.count() {
            return Field::Gauge(GaugeSystem.label_from_index(index));
        }
        index -= GaugeSystem.count();

        if index < ConstraintSystem.count() {
            return Field::Constraint(ConstraintSystem.label_from_index(index));
        }
        index -= ConstraintSystem.count();

        // **************************
        // Sources ******************

        let (scalar_field, id) = ScalarFields(self.scalar_fields.len()).label_from_index(index);
        Field::ScalarField(scalar_field, id)
    }

    fn label_index(&self, label: Self::Label) -> usize {
        let mut offset = 0;

        if let Field::Metric(metric) = label {
            return MetricSystem.label_index(metric) + offset;
        }
        offset += MetricSystem.count();

        if let Field::Gauge(gauge) = label {
            return GaugeSystem.label_index(gauge) + offset;
        }
        offset += GaugeSystem.count();

        if let Field::Constraint(constraint) = label {
            return ConstraintSystem.label_index(constraint) + offset;
        }
        offset += ConstraintSystem.count();

        let Field::ScalarField(scalar_field, id) = label else {
            unreachable!()
        };

        ScalarFields(self.scalar_fields.len()).label_index((scalar_field, id)) + offset
    }

    fn label_name(&self, label: Self::Label) -> String {
        match label {
            Field::Metric(metric) => MetricSystem.label_name(metric),
            Field::Gauge(gauge) => GaugeSystem.label_name(gauge),
            Field::Constraint(constraints) => ConstraintSystem.label_name(constraints),
            Field::ScalarField(scalar_field, id) => {
                ScalarFields(self.scalar_fields.len()).label_name((scalar_field, id))
            }
        }
    }
}

/// Label for indexing fields in `Fields`.
#[derive(Clone, Copy)]
pub enum Field {
    /// Metric and extrinsic curvature.
    Metric(Metric),
    /// Lapse and shift.
    Gauge(Gauge),
    /// Constraint violation.
    Constraint(Constraint),
    /// Scalar field sources.
    ScalarField(ScalarField, usize),
}

/// Metric variables.
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

/// Gauge variables.
#[derive(Clone, Copy, SystemLabel)]
pub enum Gauge {
    Lapse,
    Shiftr,
    Shiftz,
}

/// Constraint violation variables.
#[derive(Clone, Copy, SystemLabel)]
pub enum Constraint {
    Theta,
    Zr,
    Zz,
}

/// Scalar field variables.
#[derive(Clone, Copy)]
pub struct ScalarFields(pub usize);

impl System for ScalarFields {
    const NAME: &'static str = "ScalarFields";

    type Label = (ScalarField, usize);

    fn count(&self) -> usize {
        self.0 * 2
    }

    fn enumerate(&self) -> impl Iterator<Item = Self::Label> {
        (0..self.0).flat_map(|i| {
            [ScalarField::Phi, ScalarField::Pi]
                .into_iter()
                .map(move |scalar_field| (scalar_field, i))
        })
    }

    fn label_index(&self, label: Self::Label) -> usize {
        match label {
            (ScalarField::Phi, i) => 2 * i,
            (ScalarField::Pi, i) => 2 * i + 1,
        }
    }

    fn label_from_index(&self, index: usize) -> Self::Label {
        if index % 2 == 0 {
            (ScalarField::Phi, index / 2)
        } else {
            (ScalarField::Pi, index / 2)
        }
    }

    fn label_name(&self, label: Self::Label) -> String {
        match label {
            (ScalarField::Phi, i) => format!("Phi{i}"),
            (ScalarField::Pi, i) => format!("Pi{i}"),
        }
    }
}

/// Variables for a single scalar field.
#[derive(Clone, Copy)]
pub enum ScalarField {
    /// Scalar field component
    Phi,
    /// Time derivative of `Phi`.
    Pi,
}

/// Boundary conditions for various fields.
#[derive(Clone)]
pub struct FieldConditions;

impl SystemConditions<2> for FieldConditions {
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

            Field::ScalarField(_, _) => [true, true],
        };
        axes[face.axis]
    }

    fn radiative(&self, field: Field, _position: [f64; 2]) -> RadiativeParams {
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
