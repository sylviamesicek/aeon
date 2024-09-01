use aeon_geometry::{Rectangle, Tree};
use std::collections::HashMap;

use crate::fd::Mesh;
use crate::system::{SystemLabel, SystemSlice, SystemVec};

/// Represents all information nessessary to store and load meshes from
/// disk.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MeshCheckpoint<const N: usize> {
    tree: Tree<N>,
    #[serde(with = "aeon_array")]
    width: [usize; N],
    ghost: usize,
}

impl<const N: usize> MeshCheckpoint<N> {
    pub fn save_mesh(&mut self, mesh: &Mesh<N>) {
        self.tree.clone_from(&mesh.tree);
        self.width = mesh.width;
        self.ghost = mesh.ghost;
    }

    pub fn load_mesh(&self, mesh: &mut Mesh<N>) {
        mesh.tree.clone_from(&self.tree);
        mesh.width = self.width;
        mesh.ghost = self.ghost;

        mesh.build();
    }
}

impl<const N: usize> Default for MeshCheckpoint<N> {
    fn default() -> Self {
        Self {
            tree: Tree::new(Rectangle::UNIT),
            width: [4; N],
            ghost: 1,
        }
    }
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemCheckpoint {
    pub(crate) systems: HashMap<String, SystemMeta>,
    pub(crate) fields: HashMap<String, Vec<f64>>,
}

impl SystemCheckpoint {
    /// Attaches a system for serialization and deserialization
    pub fn save_system<Label: SystemLabel>(&mut self, system: SystemSlice<'_, Label>) {
        assert!(!self.systems.contains_key(Label::SYSTEM_NAME));

        let count = system.len();
        let data = system.to_vec().into_contiguous();

        let fields = Label::fields()
            .into_iter()
            .map(|label| label.name())
            .collect();

        let meta = SystemMeta {
            count,
            data,
            fields,
        };

        self.systems.insert(Label::SYSTEM_NAME.to_string(), meta);
    }

    /// Reads a system from the model.
    pub fn load_system<Label: SystemLabel>(&self, vec: &mut SystemVec<Label>) {
        let meta = self.systems.get(Label::SYSTEM_NAME).unwrap();
        let _ = std::mem::replace(vec, SystemSlice::from_contiguous(&meta.data).to_vec());
    }

    /// Attaches a field for serialization in the model.
    pub fn save_field(&mut self, name: &str, data: &[f64]) {
        assert!(!self.fields.contains_key(name));
        self.fields.insert(name.to_string(), data.to_vec());
    }

    /// Reads a field from the model.
    pub fn load_field(&self, name: &str, data: &mut Vec<f64>) {
        data.clone_from_slice(self.fields.get(name).unwrap());
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SystemMeta {
    pub count: usize,
    pub data: Vec<f64>,
    pub fields: Vec<String>,
}

// pub fn export_checkpoint<const N: usize>(
//     path: impl AsRef<Path>,
//     mesh: &MeshCheckpoint<N>,
//     systems: &SystemCheckpoint,
// ) -> io::Result<()> {
//     let data = ron::ser::to_string_pretty(&(mesh, systems), PrettyConfig::default())
//         .map_err(|err| io::Error::other(err))?;
//     let mut file = File::create(path)?;
//     file.write_all(data.as_bytes())
// }

// pub fn import_checkpoint<const N: usize>(
//     path: impl AsRef<Path>,
// ) -> io::Result<(MeshCheckpoint<N>, SystemCheckpoint)> {
//     let mut contents: String = String::new();
//     let mut file = File::open(path)?;
//     file.read_to_string(&mut contents)?;
//     ron::from_str(&contents).map_err(io::Error::other)
// }
