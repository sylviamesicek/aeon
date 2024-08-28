use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read as _, Write};
use std::path::Path;

use ron::ser::PrettyConfig;
use vtkio::{model::*, Vtk};

use crate::fd::Mesh;
use crate::geometry::{IndexSpace, Rectangle};
use crate::system::{SystemLabel, SystemSlice, SystemVec};

#[derive(Clone, Debug)]
pub struct ExportVtkConfig {
    pub title: String,
    pub ghost: bool,
}

impl Default for ExportVtkConfig {
    fn default() -> Self {
        Self {
            ghost: false,
            title: String::new(),
        }
    }
}

/// A model of numerical data which can be serialized and deserialized from the disk,
/// as well as converted to other visualization formats (such as VTK).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Model<const N: usize> {
    mesh: Mesh<N>,
    systems: HashMap<String, SystemMeta>,
    fields: HashMap<String, Vec<f64>>,
}

impl<const N: usize> Model<N> {
    /// Constructs an empty model, to which meshes
    pub fn empty() -> Model<N> {
        Self {
            mesh: Mesh::new(Rectangle::UNIT, [2; N], 0),
            systems: HashMap::new(),
            fields: HashMap::new(),
        }
    }

    pub fn load_mesh(&mut self, mesh: &Mesh<N>) {
        self.mesh.clone_from(mesh);
    }

    // Retrieves the mesh that this struct models.
    pub fn save_mesh(&self, mesh: &mut Mesh<N>) {
        mesh.clone_from(&self.mesh);
    }

    /// Attaches a system for serialization and deserialization
    pub fn write_system<Label: SystemLabel>(&mut self, system: SystemSlice<'_, Label>) {
        assert!(!self.systems.contains_key(Label::NAME));

        let node_count = system.len();
        let data = system.to_vec().into_contiguous();

        let fields = Label::fields()
            .into_iter()
            .map(|label| label.field_name())
            .collect();

        let meta = SystemMeta {
            node_count,
            data,
            fields,
        };

        self.systems.insert(Label::NAME.to_string(), meta);
    }

    /// Reads a system from the model.
    pub fn read_system<Label: SystemLabel>(&self) -> Option<SystemVec<Label>> {
        let meta = self.systems.get(Label::NAME)?;
        Some(SystemSlice::from_contiguous(&meta.data).to_vec())
    }

    /// Attaches a field for serialization in the model.
    pub fn write_field(&mut self, name: &str, data: Vec<f64>) {
        assert!(!self.fields.contains_key(name));
        self.fields.insert(name.to_string(), data);
    }

    /// Reads a field from the model.
    pub fn read_field(&self, name: &str) -> Option<Vec<f64>> {
        self.fields.get(name).cloned()
    }

    /// Exports a model to a dat file.
    pub fn export_dat(&self, path: impl AsRef<Path>) -> Result<(), io::Error> {
        let data = ron::ser::to_string_pretty(self, PrettyConfig::new())
            .map_err(|err| io::Error::other(err))?;
        let mut file = File::create(path)?;
        file.write_all(data.as_bytes())
    }

    /// Imports a model from a dat file.
    pub fn import_dat(path: impl AsRef<Path>) -> Result<Self, io::Error> {
        let mut contents: String = String::new();
        let mut file = File::open(path)?;
        file.read_to_string(&mut contents)?;

        Ok(ron::from_str(&contents).map_err(io::Error::other)?)
    }

    /// Exports the model to a vtk file.
    pub fn export_vtk(&self, path: impl AsRef<Path>, config: ExportVtkConfig) -> io::Result<()> {
        const {
            assert!(N > 0 && N <= 2, "Vtk Output only supported for 0 < N ≤ 2");
        }

        // Generate Cells
        let mut connectivity = Vec::new();
        let mut offsets = Vec::new();

        let mut vertex_total = 0;
        let mut cell_total = 0;

        for block in 0..self.mesh.num_blocks() {
            let space = self.mesh.block_space(block);

            let mut cell_size = space.cell_size();
            let mut vertex_size = space.inner_size();

            if config.ghost {
                for axis in 0..N {
                    cell_size[axis] += 2 * space.ghost;
                    vertex_size[axis] += 2 * space.ghost;
                }
            }

            let cell_space = IndexSpace::new(cell_size);
            let vertex_space = IndexSpace::new(vertex_size);

            for cell in cell_space.iter() {
                let mut vertex = [0; N];

                if N == 1 {
                    vertex[0] = cell[0];
                    let v1 = vertex_space.linear_from_cartesian(vertex);
                    vertex[0] = cell[0] + 1;
                    let v2 = vertex_space.linear_from_cartesian(vertex);

                    connectivity.push(vertex_total + v1 as u64);
                    connectivity.push(vertex_total + v2 as u64);
                } else if N == 2 {
                    vertex[0] = cell[0];
                    vertex[1] = cell[1];
                    let v1 = vertex_space.linear_from_cartesian(vertex);
                    vertex[0] = cell[0];
                    vertex[1] = cell[1] + 1;
                    let v2 = vertex_space.linear_from_cartesian(vertex);
                    vertex[0] = cell[0] + 1;
                    vertex[1] = cell[1] + 1;
                    let v3 = vertex_space.linear_from_cartesian(vertex);
                    vertex[0] = cell[0] + 1;
                    vertex[1] = cell[1];
                    let v4 = vertex_space.linear_from_cartesian(vertex);

                    connectivity.push(vertex_total + v1 as u64);
                    connectivity.push(vertex_total + v2 as u64);
                    connectivity.push(vertex_total + v3 as u64);
                    connectivity.push(vertex_total + v4 as u64);
                }

                offsets.push(connectivity.len() as u64);
            }

            cell_total += cell_space.index_count() as usize;
            vertex_total += vertex_space.index_count() as u64;
        }

        let cells = Cells {
            cell_verts: VertexNumbers::XML {
                connectivity,
                offsets,
            },
            types: vec![CellType::Quad; cell_total],
        };

        // Generate point data
        let mut vertices = Vec::new();

        for block in 0..self.mesh.num_blocks() {
            let bounds = self.mesh.block_bounds(block);
            let space = self.mesh.block_space(block);
            let window = if config.ghost {
                space.full_window()
            } else {
                space.inner_window()
            };

            for node in window {
                let position = space.position(node, bounds.clone());
                let mut vertex = [0.0; 3];
                vertex[..N].copy_from_slice(&position);
                vertices.extend(vertex);
            }
        }

        let points = IOBuffer::new(vertices);

        // Attributes
        let mut attributes = Attributes {
            point: Vec::new(),
            cell: Vec::new(),
        };

        for (name, system) in self.systems.iter() {
            for (idx, field) in system.fields.iter().enumerate() {
                let start = idx * system.node_count;
                let end = idx * system.node_count + system.node_count;

                let field_data = &system.data[start..end];

                let mut data = Vec::new();

                for block in 0..self.mesh.num_blocks() {
                    let space = self.mesh.block_space(block);
                    let nodes = self.mesh.block_nodes(block);
                    let window = if config.ghost {
                        space.full_window()
                    } else {
                        space.inner_window()
                    };

                    for node in window {
                        let value = space.value(node, &field_data[nodes.clone()]);
                        data.push(value);
                    }
                }

                attributes.point.push(Attribute::DataArray(DataArrayBase {
                    name: format!("{}::{}", name, field),
                    elem: ElementType::Scalars {
                        num_comp: 1,
                        lookup_table: None,
                    },
                    data: IOBuffer::new(data),
                }));
            }
        }

        for (name, system) in self.fields.iter() {
            let mut data = Vec::new();

            for block in 0..self.mesh.num_blocks() {
                let space = self.mesh.block_space(block);
                let nodes = self.mesh.block_nodes(block);

                let window = if config.ghost {
                    space.full_window()
                } else {
                    space.inner_window()
                };

                for node in window {
                    let value = space.value(node, &system[nodes.clone()]);
                    data.push(value);
                }
            }

            attributes.point.push(Attribute::DataArray(DataArrayBase {
                name: format!("Field::{}", name),
                elem: ElementType::Scalars {
                    num_comp: 1,
                    lookup_table: None,
                },
                data: IOBuffer::new(data),
            }));
        }

        let piece = UnstructuredGridPiece {
            points,
            cells,
            data: attributes,
        };

        let model = Vtk {
            version: (2, 2).into(),
            title: config.title,
            byte_order: ByteOrder::LittleEndian,
            data: DataSet::UnstructuredGrid {
                meta: None,
                pieces: vec![Piece::Inline(Box::new(piece))],
            },
            file_path: None,
        };

        model.export(path).map_err(|i| match i {
            vtkio::Error::IO(io) => io,
            _ => io::Error::from(io::ErrorKind::Other),
        })?;

        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SystemMeta {
    node_count: usize,
    data: Vec<f64>,
    fields: Vec<String>,
}
