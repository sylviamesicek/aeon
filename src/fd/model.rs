use std::collections::HashMap;
use std::io;
use std::path::Path;

use vtkio::{model::*, Vtk};

use crate::fd::Mesh;
use crate::prelude::IndexSpace;
use crate::system::{SystemLabel, SystemSlice, SystemVec};

/// A model of numerical data which can be serialized and deserialized from the disk,
/// as well as converted to other visualization formats (such as VTK).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Model<const N: usize> {
    mesh: Mesh<N>,
    systems: HashMap<String, SystemMeta>,
    fields: HashMap<String, Vec<f64>>,
}

impl<const N: usize> Model<N> {
    /// Constructs a new empty model of the mesh, to which systems and fields can be attached.
    pub fn from_mesh(mesh: &Mesh<N>) -> Self {
        Self {
            mesh: mesh.clone(),
            systems: HashMap::new(),
            fields: HashMap::new(),
        }
    }

    // Retrieves the mesh that this struct models.
    pub fn mesh(&self) -> &Mesh<N> {
        &self.mesh
    }

    /// Attaches a system for serialization and deserialization
    pub fn attach_system<Label: SystemLabel>(&mut self, system: SystemSlice<'_, Label>) {
        assert!(!self.systems.contains_key(Label::NAME));

        let node_count = system.len();
        let data = system.to_vec().to_contiguous();

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
    pub fn attach_field(&mut self, name: &str, data: Vec<f64>) {
        assert!(!self.fields.contains_key(name));
        self.fields.insert(name.to_string(), data);
    }

    /// Reads a field from the model.
    pub fn read_field(&self, name: &str) -> Option<Vec<f64>> {
        self.fields.get(name).cloned()
    }

    /// Saves the model as a vtk file on disk for visualization in an external program.
    pub fn export_vtk(&self, title: &str, path: impl AsRef<Path>) -> Result<(), io::Error> {
        let model = self.vtk_model(title, false);
        model.export(path).map_err(|i| match i {
            vtkio::Error::IO(io) => io,
            _ => io::Error::from(io::ErrorKind::Other),
        })
    }

    pub fn export_vtk_ghost(&self, title: &str, path: impl AsRef<Path>) -> Result<(), io::Error> {
        let model = self.vtk_model(title, true);
        model.export(path).map_err(|i| match i {
            vtkio::Error::IO(io) => io,
            _ => io::Error::from(io::ErrorKind::Other),
        })
    }

    fn vtk_model(&self, title: &str, ghost: bool) -> Vtk {
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
            let mut vertex_size = space.vertex_size();

            if ghost {
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
            let space = self.mesh.block_space(block).set_context(bounds);
            let window = if ghost {
                space.full_window()
            } else {
                space.inner_window()
            };

            for node in window {
                let position = space.position(node);
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
                    let window = if ghost {
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

                let window = if ghost {
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

        Vtk {
            version: (2, 2).into(),
            title: title.to_string(),
            byte_order: ByteOrder::LittleEndian,
            data: DataSet::UnstructuredGrid {
                meta: None,
                pieces: vec![Piece::Inline(Box::new(piece))],
            },
            file_path: None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SystemMeta {
    node_count: usize,
    data: Vec<f64>,
    fields: Vec<String>,
}
