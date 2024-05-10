use std::{collections::HashMap, io, path::Path};

use crate::common::node_from_vertex;
use crate::mesh::{Mesh, SystemLabel, SystemVec};
use vtkio::model::*;

/// A model of numerical data which can be serialized and deserialized from the disk,
/// as well as converted to other visualization formats (such as VTK).
#[derive(Debug, Clone)]
pub struct Model<const N: usize> {
    mesh: Mesh<N>,
    systems: HashMap<String, SystemMeta>,
    fields: HashMap<String, Vec<f64>>,
}

impl<const N: usize> Model<N> {
    /// Constructs a new empty model of the mesh, to which systems and fields can be attached.
    pub fn new(mesh: Mesh<N>) -> Self {
        Self {
            mesh,
            systems: HashMap::new(),
            fields: HashMap::new(),
        }
    }

    /// Retrieves the mesh that this struct models.
    pub fn mesh(&self) -> &Mesh<N> {
        &self.mesh
    }

    /// Attaches system to model.
    pub fn attach_system<Label: SystemLabel>(&mut self, system: &SystemVec<Label>) {
        assert!(self.systems.contains_key(Label::NAME) == false);

        let node_count = system.node_count();
        let data = system.into_contigious();

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

    pub fn read_system<Label: SystemLabel>(&self) -> Option<SystemVec<Label>> {
        let meta = self.systems.get(Label::NAME)?;
        Some(SystemVec::from_contigious(&meta.data))
    }

    pub fn attach_field(&mut self, name: &str, data: Vec<f64>) {
        assert!(self.fields.contains_key(name) == false);
        self.fields.insert(name.to_string(), data);
    }

    pub fn read_field(&self, name: &str) -> Option<Vec<f64>> {
        self.fields.get(name).cloned()
    }

    pub fn export_vtk(&self, title: &str, path: impl AsRef<Path>) -> Result<(), io::Error> {
        let model = self.vtk_model(title);
        model.export(path).map_err(|i| match i {
            vtkio::Error::IO(io) => io,
            _ => io::Error::from(io::ErrorKind::Other),
        })
    }

    fn vtk_model(&self, title: &str) -> Vtk {
        assert!(N > 0 && N <= 2, "Vtk Output only supported for 0 < N ≤ 2");

        let node_space = self.mesh.base_block().space;

        let cell_space = node_space.cell_space();
        let vertex_space = node_space.vertex_space();

        let cell_total = node_space.cell_space().len();

        // Generate Cells

        let mut connectivity = Vec::new();
        let mut offsets = Vec::new();

        for cell in cell_space.iter() {
            let mut vertex = [0; N];

            if N == 1 {
                vertex[0] = cell[0];
                let v1 = vertex_space.linear_from_cartesian(vertex);
                vertex[0] = cell[0] + 1;
                let v2 = vertex_space.linear_from_cartesian(vertex);

                connectivity.push(v1 as u64);
                connectivity.push(v2 as u64);
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

                connectivity.push(v1 as u64);
                connectivity.push(v2 as u64);
                connectivity.push(v3 as u64);
                connectivity.push(v4 as u64);
            }

            offsets.push(connectivity.len() as u64);
        }

        let cell_verts = VertexNumbers::XML {
            connectivity,
            offsets,
        };

        let cell_types = vec![CellType::Quad; cell_total];

        let cells = Cells {
            cell_verts,
            types: cell_types,
        };

        // Generate point data
        let mut vertices = Vec::new();

        for vertex in vertex_space.iter() {
            let position = node_space.position(node_from_vertex(vertex));
            let mut vertex = [0.0; 3];
            vertex[..N].copy_from_slice(&position);
            // Push to vertices
            vertices.extend(vertex);
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

                let mut vert_data = Vec::with_capacity(vertex_space.len());

                for vertex in vertex_space.iter() {
                    vert_data
                        .push(node_space.value(node_from_vertex(vertex), &system.data[start..end]));
                }

                attributes.point.push(Attribute::DataArray(DataArrayBase {
                    name: format!("{}::{}", name, field),
                    elem: ElementType::Scalars {
                        num_comp: 1,
                        lookup_table: None,
                    },
                    data: IOBuffer::new(vert_data),
                }));
            }
        }

        for (name, data) in self.fields.iter() {
            let mut vert_data = Vec::with_capacity(vertex_space.len());

            for vertex in vertex_space.iter() {
                vert_data.push(node_space.value(node_from_vertex(vertex), &data));
            }

            attributes.point.push(Attribute::DataArray(DataArrayBase {
                name: format!("Field::{}", name.clone()),
                elem: ElementType::Scalars {
                    num_comp: 1,
                    lookup_table: None,
                },
                data: IOBuffer::new(vert_data),
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
