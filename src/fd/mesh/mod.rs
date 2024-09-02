use aeon_geometry::{
    FaceMask, IndexSpace, Rectangle, Tree, TreeBlocks, TreeDofs, TreeInterfaces, TreeNeighbors,
};
use ron::ser::PrettyConfig;
use std::{
    array,
    fmt::Write,
    fs::File,
    io::{self, Read as _, Write as _},
    ops::Range,
    path::Path,
};
use vtkio::{
    model::{
        Attribute, Attributes, ByteOrder, CellType, Cells, DataArrayBase, DataSet, ElementType,
        Piece, UnstructuredGridPiece, VertexNumbers,
    },
    IOBuffer, Vtk,
};

mod checkpoint;
mod order;

pub use checkpoint::{MeshCheckpoint, SystemCheckpoint};

use crate::system::{SystemLabel, SystemSlice};

use super::{BlockBoundary, NodeSpace};

#[derive(Debug, Clone)]
pub struct Mesh<const N: usize> {
    tree: Tree<N>,
    width: [usize; N],
    ghost: usize,

    blocks: TreeBlocks<N>,
    neighbors: TreeNeighbors<N>,

    dofs: TreeDofs<N>,
    interfaces: TreeInterfaces<N>,

    coarse_dofs: TreeDofs<N>,
    coarse_interfaces: TreeInterfaces<N>,
}

impl<const N: usize> Mesh<N> {
    pub fn new(bounds: Rectangle<N>, width: [usize; N], ghost: usize) -> Self {
        let tree = Tree::new(bounds);

        let mut result = Self {
            tree,
            width,
            ghost,

            blocks: TreeBlocks::default(),
            neighbors: TreeNeighbors::default(),

            dofs: TreeDofs::default(),
            interfaces: TreeInterfaces::default(),

            coarse_dofs: TreeDofs::default(),
            coarse_interfaces: TreeInterfaces::default(),
        };

        result.build();

        result
    }

    pub fn build(&mut self) {
        self.blocks.build(&self.tree);
        self.neighbors.build(&self.tree, &self.blocks);

        self.dofs.width = self.width;
        self.dofs.ghost = self.ghost;
        self.dofs.build(&self.blocks);
        self.interfaces
            .build(&self.tree, &self.blocks, &self.neighbors, &self.dofs);

        self.coarse_dofs.width = array::from_fn(|axis| self.width[axis] / 2);
        self.coarse_dofs.ghost = self.ghost;
        self.coarse_dofs.build(&self.blocks);
        self.coarse_interfaces
            .build(&self.tree, &self.blocks, &self.neighbors, &self.coarse_dofs);
    }

    pub fn refine(&mut self, flags: &[bool]) {
        self.tree.refine(flags);
        self.build();
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.num_blocks()
    }

    pub fn num_cells(&self) -> usize {
        self.tree.num_cells()
    }

    pub fn num_dofs(&self) -> usize {
        self.dofs.num_dofs()
    }

    pub fn block_dofs(&self, block: usize) -> Range<usize> {
        self.dofs.block_dofs(block)
    }

    /// Computes the nodespace corresponding to a block.
    pub fn block_space(&self, block: usize) -> NodeSpace<N> {
        let size = self.blocks.block_size(block);
        let cell_size = array::from_fn(|axis| size[axis] * self.dofs.width[axis]);

        NodeSpace::new(cell_size, self.dofs.ghost)
    }

    pub fn num_coarse_dofs(&self) -> usize {
        self.coarse_dofs.num_dofs()
    }

    pub fn block_coarse_dofs(&self, block: usize) -> Range<usize> {
        self.coarse_dofs.block_dofs(block)
    }

    pub fn block_coarse_space(&self, block: usize) -> NodeSpace<N> {
        let size = self.blocks.block_size(block);
        let cell_size = array::from_fn(|axis| size[axis] * self.coarse_dofs.width[axis]);

        NodeSpace::new(cell_size, self.coarse_dofs.ghost)
    }

    pub fn block_bounds(&self, block: usize) -> Rectangle<N> {
        self.blocks.block_bounds(block)
    }

    /// Computes flags indicating whether a particular face of a block borders a physical
    /// boundary.
    pub fn block_boundary_flags(&self, block: usize) -> FaceMask<N> {
        self.blocks.block_boundary_flags(block)
    }

    /// Produces a block boundary which correctly accounts for
    /// interior interfaces.
    pub fn block_boundary<B>(&self, block: usize, boundary: B) -> BlockBoundary<N, B> {
        BlockBoundary {
            flags: self.block_boundary_flags(block),
            inner: boundary,
        }
    }

    pub fn max_level(&self) -> usize {
        let mut level = 1;

        for block in 0..self.blocks.num_blocks() {
            let cell = self.blocks.block_cells(block)[0];
            level = level.max(self.tree.level(cell))
        }

        level
    }

    pub fn min_spacing(&self) -> f64 {
        let max_level = self.max_level();
        let domain = self.tree.domain();

        array::from_fn::<_, N, _>(|axis| {
            domain.size[axis] / self.dofs.width[axis] as f64 / 2_f64.powi(max_level as i32)
        })
        .iter()
        .min_by(|a, b| f64::total_cmp(a, b))
        .cloned()
        .unwrap_or(1.0)
    }

    /// Computes the maximum l2 norm of all fields in the system.
    pub fn norm<S: SystemLabel>(&mut self, source: SystemSlice<'_, S>) -> f64 {
        S::fields()
            .into_iter()
            .map(|label| self.norm_scalar(source.field(label)))
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    fn norm_scalar(&mut self, src: &[f64]) -> f64 {
        let mut result = 0.0;

        for block in 0..self.blocks.num_blocks() {
            let bounds = self.blocks.block_bounds(block);
            let space = self.block_space(block);
            let size = space.size();

            let data = &src[self.block_dofs(block)];

            let mut block_result = 0.0;

            for node in space.inner_window() {
                let index = space.index_from_node(node);
                let mut value = data[index] * data[index];

                for axis in 0..N {
                    if node[axis] == 0 || node[axis] == size[axis] as isize {
                        value *= 0.5;
                    }
                }

                block_result += value;
            }

            for spacing in space.spacing(bounds.clone()) {
                block_result *= spacing;
            }

            result += block_result;
        }

        result.sqrt()
    }

    pub fn write_debug(&self, mut result: impl Write) {
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Cells ****************").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "").unwrap();

        for cell in 0..self.tree.num_cells() {
            writeln!(result, "Cell {cell}").unwrap();
            writeln!(result, "    Bounds {:?}", self.tree.bounds(cell)).unwrap();
            writeln!(result, "    Block {}", self.blocks.cell_block(cell)).unwrap();
            writeln!(
                result,
                "    Block Position {:?}",
                self.blocks.cell_index(cell)
            )
            .unwrap();

            writeln!(result, "    Neighbors {:?}", self.tree.neighbor_slice(cell)).unwrap();
        }

        writeln!(result, "").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Blocks ***************").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "").unwrap();

        for block in 0..self.blocks.num_blocks() {
            writeln!(result, "Block {block}").unwrap();
            writeln!(result, "    Bounds {:?}", self.blocks.block_bounds(block)).unwrap();
            writeln!(result, "    Size {:?}", self.blocks.block_size(block)).unwrap();
            writeln!(result, "    Cells {:?}", self.blocks.block_cells(block)).unwrap();
            writeln!(
                result,
                "    Boundary {:?}",
                self.blocks.block_boundary_flags(block)
            )
            .unwrap();
        }

        writeln!(result, "").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Interfaces ***********").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "").unwrap();

        writeln!(result, "// Fine Interfaces").unwrap();
        writeln!(result, "").unwrap();

        for interface in self.interfaces.fine() {
            writeln!(
                result,
                "Fine Interface {} -> {}",
                interface.block, interface.neighbor
            )
            .unwrap();
            writeln!(
                result,
                "    Source {}: Origin {:?}, Size {:?}",
                interface.neighbor, interface.source, interface.size
            )
            .unwrap();
            writeln!(
                result,
                "    Dest {}: Origin {:?}, Size {:?}",
                interface.block, interface.dest, interface.size
            )
            .unwrap();
        }

        writeln!(result, "// Direct Interfaces").unwrap();
        writeln!(result, "").unwrap();

        for interface in self.interfaces.direct() {
            writeln!(
                result,
                "Direct Interface {} -> {}",
                interface.block, interface.neighbor
            )
            .unwrap();
            writeln!(
                result,
                "    Source {}: Origin {:?}, Size {:?}",
                interface.neighbor, interface.source, interface.size
            )
            .unwrap();
            writeln!(
                result,
                "    Dest {}: Origin {:?}, Size {:?}",
                interface.block, interface.dest, interface.size
            )
            .unwrap();
        }

        writeln!(result, "// Coarse Interfaces").unwrap();
        writeln!(result, "").unwrap();

        for interface in self.interfaces.coarse() {
            writeln!(
                result,
                "Coarse Interface {} -> {}",
                interface.block, interface.neighbor
            )
            .unwrap();
            writeln!(
                result,
                "    Source {}: Origin {:?}, Size {:?}",
                interface.neighbor, interface.source, interface.size
            )
            .unwrap();
            writeln!(
                result,
                "    Dest {}: Origin {:?}, Size {:?}",
                interface.block, interface.dest, interface.size
            )
            .unwrap();
        }
    }
}

impl<const N: usize> Default for Mesh<N> {
    fn default() -> Self {
        let mut result = Self {
            tree: Tree::new(Rectangle::UNIT),
            width: [4; N],
            ghost: 1,

            blocks: TreeBlocks::default(),
            neighbors: TreeNeighbors::default(),

            dofs: TreeDofs::default(),
            interfaces: TreeInterfaces::default(),

            coarse_dofs: TreeDofs::default(),
            coarse_interfaces: TreeInterfaces::default(),
        };

        result.build();

        result
    }
}

#[derive(Clone, Debug)]
pub struct ExportVtkConfig {
    pub title: String,
    pub ghost: bool,
    pub systems: SystemCheckpoint,
}

impl Default for ExportVtkConfig {
    fn default() -> Self {
        Self {
            ghost: false,
            title: String::new(),
            systems: SystemCheckpoint::default(),
        }
    }
}

impl<const N: usize> Mesh<N> {
    pub fn export_dat(&self, path: impl AsRef<Path>, systems: &SystemCheckpoint) -> io::Result<()> {
        let mut checkpoint = MeshCheckpoint::default();
        checkpoint.save_mesh(self);

        let data = ron::ser::to_string_pretty(&(checkpoint, systems), PrettyConfig::default())
            .map_err(|err| io::Error::other(err))?;
        let mut file = File::create(path)?;
        file.write_all(data.as_bytes())
    }

    pub fn import_dat(
        &mut self,
        path: impl AsRef<Path>,
        systems: &mut SystemCheckpoint,
    ) -> io::Result<()> {
        let mut contents: String = String::new();
        let mut file = File::create(path)?;
        file.read_to_string(&mut contents)?;

        let (mesh, system): (MeshCheckpoint<N>, SystemCheckpoint) =
            ron::from_str(&contents).map_err(io::Error::other)?;

        mesh.load_mesh(self);
        systems.clone_from(&system);

        Ok(())
    }

    pub fn export_vtk(&self, path: impl AsRef<Path>, config: ExportVtkConfig) -> io::Result<()> {
        const {
            assert!(N > 0 && N <= 2, "Vtk Output only supported for 0 < N ≤ 2");
        }

        // Generate Cells
        let mut connectivity = Vec::new();
        let mut offsets = Vec::new();

        let mut vertex_total = 0;
        let mut cell_total = 0;

        for block in 0..self.blocks.num_blocks() {
            let space = self.block_space(block);

            let mut cell_size = space.size();
            let mut vertex_size = space.inner_size();

            if config.ghost {
                for axis in 0..N {
                    cell_size[axis] += 2 * space.ghost();
                    vertex_size[axis] += 2 * space.ghost();
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

        for block in 0..self.blocks.num_blocks() {
            let bounds = self.blocks.block_bounds(block);
            let space = self.block_space(block);
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

        for (name, system) in config.systems.systems.iter() {
            for (idx, field) in system.fields.iter().enumerate() {
                let start = idx * system.count;
                let end = idx * system.count + system.count;

                let field_data = &system.data[start..end];

                let mut data = Vec::new();

                for block in 0..self.blocks.num_blocks() {
                    let space = self.block_space(block);
                    let nodes = self.block_dofs(block);
                    let window = if config.ghost {
                        space.full_window()
                    } else {
                        space.inner_window()
                    };

                    for node in window {
                        let index = space.index_from_node(node);
                        let value = field_data[nodes.start + index];
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

        for (name, system) in config.systems.fields.iter() {
            let mut data = Vec::new();

            for block in 0..self.blocks.num_blocks() {
                let space = self.block_space(block);
                let nodes = self.block_dofs(block);

                let window = if config.ghost {
                    space.full_window()
                } else {
                    space.inner_window()
                };

                for node in window {
                    let index = space.index_from_node(node);
                    let value = system[nodes.start + index];
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
