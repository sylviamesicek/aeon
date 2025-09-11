use crate::image::{Image, ImageRef};
use crate::mesh::{Mesh, MeshSer};
use crate::prelude::IndexSpace;
use bincode::{Decode, Encode};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{Read as _, Write as _};
use std::path::Path;
use std::str::FromStr;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CheckpointParseError {
    #[error("Invalid key")]
    InvalidKey,
    #[error("Failed to parse {0}")]
    ParseFailed(String),
}

#[derive(Default, Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Checkpoint<const N: usize> {
    /// Meta data to be stored in checkpoint (useful for storing time, number of steps, ect.)
    meta: HashMap<String, String>,
    /// Mesh data to be attached to checkpoint,
    mesh: Option<MeshSer<N>>,
    /// Is this mesh embedded in a higher dimensional mesh?
    embedding: Option<Embedding>,
    /// Systems which are stored in the checkpoint.
    systems: HashMap<String, ImageMeta>,
    /// Fields which are stored in the checkpoint
    fields: HashMap<String, Vec<f64>>,
    /// Int fields (useful for debugging) which are stored in the checkpoint.
    int_fields: HashMap<String, Vec<i64>>,
}

impl<const N: usize> Checkpoint<N> {
    /// Attaches a mesh to the checkpoint
    pub fn attach_mesh(&mut self, mesh: &Mesh<N>) {
        self.mesh.replace(MeshSer {
            tree: mesh.tree.clone().into(),
            width: mesh.width,
            ghost: mesh.ghost,
            boundary: mesh.boundary,
        });
    }

    /// Sets the mesh as embedded in a higher dimensional space.
    pub fn set_embedding<const S: usize>(&mut self, positions: &[[f64; S]]) {
        self.embedding.replace(Embedding {
            dimension: S,
            positions: positions.iter().flatten().cloned().collect(),
        });
    }

    /// Clones the mesh attached to the checkpoint.
    pub fn read_mesh(&self) -> Mesh<N> {
        self.mesh.clone().unwrap().into()
    }

    pub fn save_image(&mut self, name: &str, data: ImageRef) {
        assert!(!self.systems.contains_key(name));

        let num_channels = data.num_channels();
        let buffer = data
            .channels()
            .flat_map(|label| data.channel(label).iter().cloned())
            .collect();

        // let fields = data
        //     .system()
        //     .enumerate()
        //     .map(|label| data.system().label_name(label))
        //     .collect();

        self.systems.insert(
            name.to_string(),
            ImageMeta {
                channels: num_channels,
                buffer,
            },
        );
    }

    pub fn read_image(&self, name: &str) -> Image {
        let data = self.systems.get(name).unwrap();
        // let system = ron::de::from_str::<S>(&data.meta).unwrap();

        Image::from_storage(data.buffer.clone(), data.channels)
    }

    // pub fn save_system_default<S: System + Default>(&mut self, data: SystemSlice<S>) {
    //     assert!(!self.systems.contains_key(S::NAME));

    //     let count = data.len();
    //     let buffer = data
    //         .system()
    //         .enumerate()
    //         .flat_map(|label| data.field(label).iter().cloned())
    //         .collect();

    //     let fields = data
    //         .system()
    //         .enumerate()
    //         .map(|label| data.system().label_name(label))
    //         .collect();

    //     self.systems.insert(
    //         S::NAME.to_string(),
    //         SystemMeta {
    //             meta: String::new(),
    //             count,
    //             buffer,
    //             fields,
    //         },
    //     );
    // }

    // pub fn read_system_default<S: System + Default>(&mut self) -> SystemVec<S> {
    //     let data = self.systems.get(S::NAME).unwrap();
    //     SystemVec::from_contiguous(data.buffer.clone(), S::default())
    // }

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

    /// Loads the mesh and any additional data from disk.
    pub fn import_dat(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut contents: String = String::new();
        let mut file = File::open(path)?;
        file.read_to_string(&mut contents)?;

        ron::from_str(&contents).map_err(std::io::Error::other)
    }

    pub fn export_dat(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let data = ron::ser::to_string_pretty::<Checkpoint<N>>(self, PrettyConfig::default())
            .map_err(std::io::Error::other)?;
        let mut file = File::create(path)?;
        file.write_all(data.as_bytes())
    }
}

/// Metadata required for storing a system.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImageMeta {
    pub channels: usize,
    pub buffer: Vec<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HyperSurface<const N: usize, const S: usize> {
    pub surface: MeshSer<S>,
    #[serde(with = "crate::array::vec")]
    pub position: Vec<[f64; N]>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Embedding {
    dimension: usize,
    positions: Vec<f64>,
}

use num_traits::ToPrimitive;
use vtkio::{
    IOBuffer, Vtk,
    model::{
        Attribute, Attributes, ByteOrder, CellType, Cells, DataArrayBase, DataSet, ElementType,
        Piece, UnstructuredGridPiece, Version, VertexNumbers,
    },
};

#[derive(Clone, Debug)]
pub struct ExportVtuConfig {
    pub title: String,
    pub ghost: bool,
    pub stride: ExportStride,
}

impl Default for ExportVtuConfig {
    fn default() -> Self {
        Self {
            title: "Title".to_string(),
            ghost: false,
            stride: ExportStride::PerVertex,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, Encode, Decode)]
pub enum ExportStride {
    /// Output data for every vertex in the simulation
    #[serde(rename = "per_vertex")]
    PerVertex,
    /// Output data for each corner of a cell in the simulation
    /// This is significantly more compressed
    #[serde(rename = "per_cell")]
    PerCell,
}

impl<const N: usize> Checkpoint<N> {
    pub fn export_csv(&self, path: impl AsRef<Path>, stride: ExportStride) -> std::io::Result<()> {
        let mesh: Mesh<N> = self.mesh.clone().unwrap().into();
        let stride = match stride {
            ExportStride::PerVertex => 1,
            ExportStride::PerCell => mesh.width,
        };

        let mut wtr = csv::Writer::from_path(path)?;
        let mut header = (0..N)
            .into_iter()
            .map(|i| format!("Coord{i}"))
            .collect::<Vec<String>>();

        for field in self.fields.keys() {
            header.push(field.clone());
        }

        wtr.write_record(header.iter())?;

        let mut buffer = String::new();

        for block in mesh.blocks.indices() {
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);
            let window = space.inner_window();

            'window: for node in window {
                for axis in 0..N {
                    if node[axis] % (stride as isize) != 0 {
                        continue 'window;
                    }
                }

                let index = space.index_from_node(node);
                let position = space.position(node);

                for i in 0..N {
                    buffer.clear();
                    write!(&mut buffer, "{}", position[i]).unwrap();
                    wtr.write_field(&buffer)?;
                }

                for (i, (name, data)) in self.fields.iter().enumerate() {
                    debug_assert_eq!(&header[i + N], name);

                    let value = data[nodes.clone()][index];
                    buffer.clear();
                    write!(&mut buffer, "{}", value).unwrap();
                    wtr.write_field(&buffer)?;
                }

                wtr.write_record::<&[String], &String>(&[])?;
            }
        }

        Ok(())
    }

    /// Checkpoint and additional field data to a .vtu file, for visualisation in applications like
    /// Paraview. This requires a mesh be attached to the checkpoint.
    pub fn export_vtu(
        &self,
        path: impl AsRef<Path>,
        config: ExportVtuConfig,
    ) -> std::io::Result<()> {
        const {
            assert!(N > 0 && N <= 2, "Vtu Output only supported for 0 < N ≤ 2");
        }
        assert!(self.mesh.is_some(), "Mesh must be attached to checkpoint");

        // Uncompress mesh.
        let mesh: Mesh<N> = self.mesh.clone().unwrap().into();

        let stride = match config.stride {
            ExportStride::PerVertex => 1,
            ExportStride::PerCell => mesh.width,
        };

        assert!(stride <= mesh.width, "Stride must be <= width");
        assert!(
            mesh.width % stride == 0,
            "Width must be evenly divided by stride"
        );
        assert!(
            !config.ghost || mesh.ghost % stride == 0,
            "Ghost must be evenly divided by stride"
        );

        // Generate Cells
        let cells = Self::mesh_cells(&mesh, config.ghost, stride);
        // Generate Point Data
        let points = match self.embedding {
            Some(ref embedding) => {
                Self::mesh_points_embedded(&mesh, embedding, config.ghost, stride)
            }
            None => Self::mesh_points(&mesh, config.ghost, stride),
        };
        // Attributes
        let mut attributes = Attributes {
            point: Vec::new(),
            cell: Vec::new(),
        };

        // for (name, system) in self.systems.iter() {
        //     for (idx, field) in system.fields.iter().enumerate() {
        //         let start = idx * system.count;
        //         let end = idx * system.count + system.count;

        //         attributes.point.push(Self::field_attribute(
        //             &mesh,
        //             format!("{}::{}", name, field),
        //             &system.buffer[start..end],
        //             config.ghost,
        //             stride,
        //         ));
        //     }
        // }

        for (name, system) in self.fields.iter() {
            attributes.point.push(Self::field_attribute(
                &mesh,
                format!("Field::{}", name),
                system,
                config.ghost,
                stride,
            ));
        }

        for (name, system) in self.int_fields.iter() {
            attributes.point.push(Self::field_attribute(
                &mesh,
                format!("IntField::{}", name),
                system,
                config.ghost,
                stride,
            ));
        }

        let mut pieces = Vec::new();
        // Primary piece
        pieces.push(Piece::Inline(Box::new(UnstructuredGridPiece {
            points,
            cells,
            data: attributes,
        })));

        let model = Vtk {
            version: Version::XML { major: 2, minor: 2 },
            title: config.title,
            byte_order: ByteOrder::LittleEndian,
            data: DataSet::UnstructuredGrid { meta: None, pieces },
            file_path: None,
        };

        model.export(path).map_err(|i| match i {
            vtkio::Error::IO(io) => io,
            v => {
                log::error!("Encountered error {:?} while exporting vtu", v);
                std::io::Error::from(std::io::ErrorKind::Other)
            }
        })?;

        Ok(())
    }

    fn mesh_cells(mesh: &Mesh<N>, ghost: bool, stride: usize) -> Cells {
        let mut connectivity = Vec::new();
        let mut offsets = Vec::new();

        let mut vertex_total = 0;
        let mut cell_total = 0;

        for block in mesh.blocks.indices() {
            let space = mesh.block_space(block);

            let mut cell_size = space.cell_size();
            let mut vertex_size = space.vertex_size();

            if ghost {
                for axis in 0..N {
                    cell_size[axis] += 2 * space.ghost();
                    vertex_size[axis] += 2 * space.ghost();
                }
            }

            for axis in 0..N {
                debug_assert!(cell_size[axis] % stride == 0);
                debug_assert!((vertex_size[axis] - 1) % stride == 0);

                cell_size[axis] /= stride;
                vertex_size[axis] = (vertex_size[axis] - 1) / stride + 1;
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

            cell_total += cell_space.index_count();
            vertex_total += vertex_space.index_count() as u64;
        }

        let cell_type = match N {
            1 => CellType::Line,
            2 => CellType::Quad,
            // 3 => CellType::Hexahedron,
            _ => panic!("Unsupported dimension"),
        };

        Cells {
            cell_verts: VertexNumbers::XML {
                connectivity,
                offsets,
            },
            types: vec![cell_type; cell_total],
        }
    }

    fn mesh_points(mesh: &Mesh<N>, ghost: bool, stride: usize) -> IOBuffer {
        // Generate point data
        let mut vertices = Vec::new();

        for block in mesh.blocks.indices() {
            let space = mesh.block_space(block);
            let window = if ghost {
                space.full_window()
            } else {
                space.inner_window()
            };

            'window: for node in window {
                for axis in 0..N {
                    if node[axis] % (stride as isize) != 0 {
                        continue 'window;
                    }
                }

                let position = space.position(node);
                let mut vertex = [0.0; 3];
                vertex[..N].copy_from_slice(&position);
                vertices.extend(vertex);
            }
        }

        IOBuffer::new(vertices)
    }

    fn mesh_points_embedded(
        mesh: &Mesh<N>,
        embedding: &Embedding,
        ghost: bool,
        stride: usize,
    ) -> IOBuffer {
        let dim = embedding.dimension;
        assert!(dim <= 3);

        // Generate point data
        let mut vertices = Vec::new();

        for block in mesh.blocks.indices() {
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);
            let window = if ghost {
                space.full_window()
            } else {
                space.inner_window()
            };

            let block_positions = &embedding.positions[nodes.start * dim..nodes.end * dim];

            'window: for node in window {
                let index = space.index_from_node(node);
                let position = &block_positions[index * dim..(index + 1) * dim];

                for axis in 0..N {
                    if node[axis] % (stride as isize) != 0 {
                        continue 'window;
                    }
                }

                let mut vertex = [0.0; 3];
                vertex[..dim].copy_from_slice(&position);
                vertices.extend(vertex);
            }
        }

        IOBuffer::new(vertices)
    }

    fn field_attribute<T: ToPrimitive + Copy + 'static>(
        mesh: &Mesh<N>,
        name: String,
        data: &[T],
        ghost: bool,
        stride: usize,
    ) -> Attribute {
        let mut buffer = Vec::new();

        for block in mesh.blocks.indices() {
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);
            let window = if ghost {
                space.full_window()
            } else {
                space.inner_window()
            };

            'window: for node in window {
                for axis in 0..N {
                    if node[axis] % (stride as isize) != 0 {
                        continue 'window;
                    }
                }

                let index = space.index_from_node(node);
                let value = data[nodes.clone()][index];
                buffer.push(value);
            }
        }

        Attribute::DataArray(DataArrayBase {
            name,
            elem: ElementType::Scalars {
                num_comp: 1,
                lookup_table: None,
            },
            data: IOBuffer::new(buffer),
        })
    }
}
