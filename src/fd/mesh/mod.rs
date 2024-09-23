use aeon_basis::{Boundary, BoundaryKind, NodeSpace, NodeWindow};
use aeon_geometry::{
    faces, FaceMask, IndexSpace, Rectangle, Tree, TreeBlocks, TreeInterfaces, TreeNeighbors,
    TreeNodes,
};
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use ron::ser::PrettyConfig;
use std::{
    array,
    cell::UnsafeCell,
    fmt::Write,
    fs::File,
    io::{self, Read as _, Write as _},
    ops::Range,
    path::Path,
};
use thread_local::ThreadLocal;
use vtkio::{
    model::{
        Attribute, Attributes, ByteOrder, CellType, Cells, DataArrayBase, DataSet, ElementType,
        Piece, UnstructuredGridPiece, VertexNumbers,
    },
    IOBuffer, Vtk,
};

mod checkpoint;
mod evaluate;
mod regrid;
mod store;
mod transfer;

pub use checkpoint::{MeshCheckpoint, SystemCheckpoint};
pub use store::MeshStore;

use crate::fd::BlockBoundary;
use crate::system::{SystemLabel, SystemSlice};

#[derive(Debug)]
pub struct Mesh<const N: usize> {
    tree: Tree<N>,
    width: usize,
    ghost: usize,

    blocks: TreeBlocks<N>,
    neighbors: TreeNeighbors<N>,

    nodes: TreeNodes<N>,
    interfaces: TreeInterfaces<N>,
    // refine_flags: Vec<bool>,
    stores: ThreadLocal<UnsafeCell<MeshStore>>,
}

impl<const N: usize> Mesh<N> {
    pub fn new(bounds: Rectangle<N>, width: usize, ghost: usize) -> Self {
        let tree = Tree::new(bounds);

        let mut result = Self {
            tree,
            width,
            ghost,

            blocks: TreeBlocks::default(),
            neighbors: TreeNeighbors::default(),

            nodes: TreeNodes::default(),
            interfaces: TreeInterfaces::default(),

            stores: ThreadLocal::new(),
            // refine_flags: Vec::new(),
        };

        result.build();

        result
    }

    pub fn build(&mut self) {
        self.blocks.build(&self.tree);
        self.neighbors.build(&self.tree, &self.blocks);

        self.nodes.width = [self.width; N];
        self.nodes.ghost = self.ghost;
        self.nodes.build(&self.blocks);
        self.interfaces
            .build(&self.tree, &self.blocks, &self.neighbors, &self.nodes);

        // self.refine_flags.clear();
        // self.refine_flags.resize(self.tree.num_cells(), false);
    }

    pub fn tree(&self) -> &Tree<N> {
        &self.tree
    }

    pub fn refine(&mut self, flags: &[bool]) {
        // self.tree.balance_refine_flags(&mut self.refine_flags);
        // self.tree.refine(&self.refine_flags);

        self.tree.refine(flags);
        self.build();
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn num_cells(&self) -> usize {
        self.tree.num_cells()
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn block_nodes(&self, block: usize) -> Range<usize> {
        self.nodes.range(block)
    }

    /// Returns the window of nodes in a block corresponding to
    pub fn element_window(&self, cell: usize) -> NodeWindow<N> {
        let position = self.blocks.cell_position(cell);

        let size = [2 * self.width + 1; N];
        let mut origin = [-(self.width as isize) / 2; N];

        for axis in 0..N {
            origin[axis] += (self.width * position[axis]) as isize
        }

        NodeWindow { origin, size }
    }

    pub fn element_coarse_window(&self, cell: usize) -> NodeWindow<N> {
        let position = self.blocks.cell_position(cell);

        let size = [self.width + 1; N];
        let mut origin = [0; N];

        for axis in 0..N {
            origin[axis] += (self.width * position[axis]) as isize
        }

        NodeWindow { origin, size }
    }

    /// Computes the nodespace corresponding to a block.
    pub fn block_space(&self, block: usize) -> NodeSpace<N> {
        let size = self.blocks.size(block);
        let cell_size = array::from_fn(|axis| size[axis] * self.nodes.width[axis]);

        NodeSpace::new(cell_size, self.nodes.ghost)
    }

    pub fn block_bounds(&self, block: usize) -> Rectangle<N> {
        self.blocks.bounds(block)
    }

    /// Computes flags indicating whether a particular face of a block borders a physical
    /// boundary.
    pub fn block_boundary_flags(&self, block: usize) -> FaceMask<N> {
        self.blocks.boundary_flags(block)
    }

    /// Produces a block boundary which correctly accounts for
    /// interior interfaces.
    pub fn block_boundary<B>(&self, block: usize, boundary: B) -> BlockBoundary<N, B> {
        BlockBoundary {
            flags: self.block_boundary_flags(block),
            inner: boundary,
        }
    }

    /// Returns true if the given cell is on a physical boundary.
    pub fn is_cell_on_boundary<B: Boundary<N>>(&self, cell: usize, boundary: B) -> bool {
        let block = self.blocks.cell_block(cell);
        let block_size = self.blocks.size(block);
        let block_flags = self.block_boundary_flags(block);

        let position = self.blocks.cell_position(cell);

        for face in faces::<N>() {
            let on_border = position[face.axis]
                == if face.side {
                    block_size[face.axis] - 1
                } else {
                    0
                };

            if matches!(
                boundary.kind(face),
                BoundaryKind::Free | BoundaryKind::Radiative
            ) && block_flags.is_set(face)
                && on_border
            {
                return true;
            }
        }

        false
    }

    pub fn max_level(&self) -> usize {
        let mut level = 1;

        for block in 0..self.blocks.len() {
            let cell = self.blocks.cells(block)[0];
            level = level.max(self.tree.level(cell))
        }

        level
    }

    pub fn min_spacing(&self) -> f64 {
        let max_level = self.max_level();
        let domain = self.tree.domain();

        array::from_fn::<_, N, _>(|axis| {
            domain.size[axis] / self.nodes.width[axis] as f64 / 2_f64.powi(max_level as i32)
        })
        .iter()
        .min_by(|a, b| f64::total_cmp(a, b))
        .cloned()
        .unwrap_or(1.0)
    }

    pub fn block_compute<F: Fn(&Self, &MeshStore, usize) + Sync>(&mut self, f: F) {
        (0..self.num_blocks())
            .par_bridge()
            .into_par_iter()
            .for_each(|block| {
                let store = unsafe { &mut *self.stores.get_or_default().get() };
                f(self, store, block);
                store.reset();
            });
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

        for block in 0..self.blocks.len() {
            let bounds = self.blocks.bounds(block);
            let space = self.block_space(block);
            let size = space.size();

            let data = &src[self.block_nodes(block)];

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
                self.blocks.cell_position(cell)
            )
            .unwrap();

            writeln!(result, "    Neighbors {:?}", self.tree.neighbor_slice(cell)).unwrap();
        }

        writeln!(result, "").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Blocks ***************").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "").unwrap();

        for block in 0..self.blocks.len() {
            writeln!(result, "Block {block}").unwrap();
            writeln!(result, "    Bounds {:?}", self.blocks.bounds(block)).unwrap();
            writeln!(result, "    Size {:?}", self.blocks.size(block)).unwrap();
            writeln!(result, "    Cells {:?}", self.blocks.cells(block)).unwrap();
            writeln!(
                result,
                "    Vertices {:?}",
                self.block_space(block).inner_size()
            )
            .unwrap();
            writeln!(
                result,
                "    Boundary {:?}",
                self.blocks.boundary_flags(block)
            )
            .unwrap();
        }

        writeln!(result, "").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Neighbors ************").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "").unwrap();

        writeln!(result, "// Fine Neighbors").unwrap();

        for neighbor in self.neighbors.fine() {
            writeln!(result, "Fine Neighbor").unwrap();

            writeln!(
                result,
                "    Block: {}, Neighbor: {}",
                neighbor.block, neighbor.neighbor,
            )
            .unwrap();
            writeln!(
                result,
                "    Lower: Cell {}, Neighbor {}, Region {}",
                neighbor.a.cell, neighbor.a.neighbor, neighbor.a.region,
            )
            .unwrap();
            writeln!(
                result,
                "    Upper: Cell {}, Neighbor {}, Region {}",
                neighbor.b.cell, neighbor.b.neighbor, neighbor.b.region,
            )
            .unwrap();
        }

        writeln!(result, "").unwrap();
        writeln!(result, "// Direct Neighbors").unwrap();

        for neighbor in self.neighbors.direct() {
            writeln!(result, "Direct Neighbor").unwrap();

            writeln!(
                result,
                "    Block: {}, Neighbor: {}",
                neighbor.block, neighbor.neighbor,
            )
            .unwrap();
            writeln!(
                result,
                "    Lower: Cell {}, Neighbor {}, Region {}",
                neighbor.a.cell, neighbor.a.neighbor, neighbor.a.region,
            )
            .unwrap();
            writeln!(
                result,
                "    Upper: Cell {}, Neighbor {}, Region {}",
                neighbor.b.cell, neighbor.b.neighbor, neighbor.b.region,
            )
            .unwrap();
        }

        writeln!(result, "").unwrap();
        writeln!(result, "// Coarse Neighbors").unwrap();

        for neighbor in self.neighbors.coarse() {
            writeln!(result, "Coarse Neighbor").unwrap();

            writeln!(
                result,
                "    Block: {}, Neighbor: {}",
                neighbor.block, neighbor.neighbor,
            )
            .unwrap();
            writeln!(
                result,
                "    Lower: Cell {}, Neighbor {}, Region {}",
                neighbor.a.cell, neighbor.a.neighbor, neighbor.a.region,
            )
            .unwrap();
            writeln!(
                result,
                "    Upper: Cell {}, Neighbor {}, Region {}",
                neighbor.b.cell, neighbor.b.neighbor, neighbor.b.region,
            )
            .unwrap();
        }

        writeln!(result, "").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Interfaces ***********").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "").unwrap();

        writeln!(result, "// Fine Interfaces").unwrap();

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

        writeln!(result, "").unwrap();
        writeln!(result, "// Direct Interfaces").unwrap();

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

        writeln!(result, "").unwrap();
        writeln!(result, "// Coarse Interfaces").unwrap();

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

impl<const N: usize> Clone for Mesh<N> {
    fn clone(&self) -> Self {
        Self {
            tree: self.tree.clone(),
            width: self.width,
            ghost: self.ghost,

            blocks: self.blocks.clone(),
            neighbors: self.neighbors.clone(),

            nodes: self.nodes.clone(),
            interfaces: self.interfaces.clone(),

            stores: ThreadLocal::new(),
        }
    }
}

impl<const N: usize> Default for Mesh<N> {
    fn default() -> Self {
        let mut result = Self {
            tree: Tree::new(Rectangle::UNIT),
            width: 4,
            ghost: 1,

            blocks: TreeBlocks::default(),
            neighbors: TreeNeighbors::default(),

            nodes: TreeNodes::default(),
            interfaces: TreeInterfaces::default(),
            // refine_flags: Vec::new(),
            stores: ThreadLocal::new(),
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

        let data = ron::ser::to_string_pretty::<(MeshCheckpoint<N>, SystemCheckpoint)>(
            &(checkpoint, systems.clone()),
            PrettyConfig::default(),
        )
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
        let mut file = File::open(path)?;
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

        for block in 0..self.blocks.len() {
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

        for block in 0..self.blocks.len() {
            let bounds = self.blocks.bounds(block);
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

                for block in 0..self.blocks.len() {
                    let space = self.block_space(block);
                    let nodes = self.block_nodes(block);
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

            for block in 0..self.blocks.len() {
                let space = self.block_space(block);
                let nodes = self.block_nodes(block);

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
