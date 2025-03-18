use crate::geometry::{FaceArray, Rectangle, Tree};
use crate::kernel::BoundaryClass;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::collections::HashMap;
use std::str::FromStr;
use thiserror::Error;

use crate::mesh::Mesh;
use crate::system::{System, SystemSlice, SystemVec};

#[derive(Debug, Error)]
pub enum CheckpointParseError {
    #[error("Invalid key")]
    InvalidKey,
    #[error("Failed to parse {0}")]
    ParseFailed(String),
}

/// Represents all information nessessary to store and load meshes from
/// disk.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MeshCheckpoint<const N: usize> {
    tree: Tree<N>,
    width: usize,
    ghost: usize,
    boundary: FaceArray<N, BoundaryClass>,
}

impl<const N: usize> MeshCheckpoint<N> {
    pub fn save_mesh(&mut self, mesh: &Mesh<N>) {
        self.tree.clone_from(&mesh.tree);
        self.width = mesh.width;
        self.ghost = mesh.ghost;
        self.boundary = mesh.boundary;
    }

    pub fn load_mesh(&self, mesh: &mut Mesh<N>) {
        mesh.tree.clone_from(&self.tree);
        mesh.width = self.width;
        mesh.ghost = self.ghost;
        mesh.boundary = self.boundary;

        mesh.build();
    }
}

impl<const N: usize> Default for MeshCheckpoint<N> {
    fn default() -> Self {
        Self {
            tree: Tree::new(Rectangle::UNIT),
            width: 4,
            ghost: 1,
            boundary: FaceArray::from_fn(|_| BoundaryClass::OneSided),
        }
    }
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemCheckpoint {
    pub(crate) meta: HashMap<String, String>,
    pub(crate) systems: HashMap<String, SystemMeta>,
    pub(crate) fields: HashMap<String, Vec<f64>>,
    pub(crate) int_fields: HashMap<String, Vec<i64>>,
}

impl SystemCheckpoint {
    pub fn save_system_ser<S: System + Serialize>(&mut self, data: SystemSlice<S>) {
        assert!(!self.systems.contains_key(S::NAME));

        let count = data.len();
        let buffer = data
            .system()
            .enumerate()
            .flat_map(|label| data.field(label).iter().cloned())
            .collect();

        let fields = data
            .system()
            .enumerate()
            .map(|label| data.system().label_name(label))
            .collect();

        self.systems.insert(
            S::NAME.to_string(),
            SystemMeta {
                meta: ron::ser::to_string(data.system()).unwrap(),
                count,
                buffer,
                fields,
            },
        );
    }

    pub fn read_system_ser<S: System + DeserializeOwned>(&mut self) -> SystemVec<S> {
        let data = self.systems.get(S::NAME).unwrap();
        let system = ron::de::from_str::<S>(&data.meta).unwrap();

        SystemVec::from_contiguous(data.buffer.clone(), system)
    }

    pub fn save_system_default<S: System + Default>(&mut self, data: SystemSlice<S>) {
        assert!(!self.systems.contains_key(S::NAME));

        let count = data.len();
        let buffer = data
            .system()
            .enumerate()
            .flat_map(|label| data.field(label).iter().cloned())
            .collect();

        let fields = data
            .system()
            .enumerate()
            .map(|label| data.system().label_name(label))
            .collect();

        self.systems.insert(
            S::NAME.to_string(),
            SystemMeta {
                meta: String::new(),
                count,
                buffer,
                fields,
            },
        );
    }

    pub fn read_system_default<S: System + Default>(&mut self) -> SystemVec<S> {
        let data = self.systems.get(S::NAME).unwrap();
        SystemVec::from_contiguous(data.buffer.clone(), S::default())
    }

    /// Attaches a field for serialization in the model.
    pub fn save_field(&mut self, name: &str, data: &[f64]) {
        assert!(!self.fields.contains_key(name));
        self.fields.insert(name.to_string(), data.to_vec());
    }

    /// Reads a field from the model.
    pub fn load_field(&self, name: &str, data: &mut Vec<f64>) {
        data.clear();
        data.extend_from_slice(self.fields.get(name).unwrap());
    }

    /// Reads a field from the model.
    pub fn read_field(&self, name: &str) -> Vec<f64> {
        let mut result = Vec::new();
        self.load_field(name, &mut result);
        result
    }

    /// Attaches an integer field for serialization in the checkpoint.
    pub fn save_int_field(&mut self, name: &str, data: &[i64]) {
        assert!(!self.int_fields.contains_key(name));
        self.int_fields.insert(name.to_string(), data.to_vec());
    }

    /// Reads an integer field from the checkpoint.
    pub fn load_int_field(&self, name: &str, data: &mut Vec<i64>) {
        data.clear();
        data.extend_from_slice(self.int_fields.get(name).unwrap());
    }

    pub fn save_meta(&mut self, name: &str, data: &str) {
        let _ = self.meta.insert(name.to_string(), data.to_string());
    }

    pub fn load_meta(&self, name: &str, data: &mut String) {
        data.clone_from(self.meta.get(name).unwrap())
    }

    pub fn write_meta<T: ToString>(&mut self, name: &str, data: T) {
        self.save_meta(name, &data.to_string());
    }

    pub fn read_meta<T: FromStr>(&self, name: &str) -> Result<T, CheckpointParseError> {
        let data = self
            .meta
            .get(name)
            .ok_or(CheckpointParseError::InvalidKey)?;

        data.parse()
            .map_err(|_| CheckpointParseError::ParseFailed(data.clone()))
    }
}

/// Metadata required for storing a system.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemMeta {
    pub meta: String,
    pub count: usize,
    pub buffer: Vec<f64>,
    pub fields: Vec<String>,
}
