use aeon_basis::{Boundary, BoundaryKind, NodeSpace, NodeWindow};
use aeon_geometry::{faces, FaceMask, IndexSpace, Rectangle, Tree, TreeBlocks, TreeNeighbors};
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

use transfer::TreeInterface;

use crate::fd::BlockBoundary;
use crate::system::{SystemLabel, SystemSlice};

#[derive(Debug)]
pub struct Mesh<const N: usize> {
    /// Underlying Tree on which the mesh is built.
    tree: Tree<N>,
    /// The width of a cell on the mesh (i.e. how many subcells are in that cell).
    width: usize,
    /// The number of ghost cells used to facilitate inter-block communication.
    ghost: usize,

    /// Block structure induced by the tree.
    blocks: TreeBlocks<N>,
    /// Offsets linking each block to a range of nodes.
    block_node_offsets: Vec<usize>,

    /// Neighbors of each block.
    neighbors: TreeNeighbors<N>,

    /// Interfaces build from neighbors.
    interfaces: Vec<TreeInterface<N>>,
    /// Offsets linking each interface to a range of ghost/face nodes.
    interface_node_offsets: Vec<usize>,
    /// Mask that keeps different interfaces disjoint.sda
    interface_masks: Vec<bool>,

    /// Refinement flags for each cell on the mesh.
    refine_flags: Vec<bool>,
    /// Coarsening flags for each cell on the mesh.
    coarsen_flags: Vec<bool>,

    /// Map from cells before refinement to current cells.
    regrid_map: Vec<usize>,

    /// Blocks before most recent refinement.
    old_blocks: TreeBlocks<N>,
    /// Block node offsets from before most recent refinement.
    old_block_node_offsets: Vec<usize>,

    /// Thread-local stores used for allocation.
    stores: ThreadLocal<UnsafeCell<MeshStore>>,
}

impl<const N: usize> Mesh<N> {
    /// Constructs a new `Mesh` covering the domain, with a number of nodes
    /// defined by `width` and `ghost`.
    pub fn new(bounds: Rectangle<N>, width: usize, ghost: usize) -> Self {
        assert!(width % 2 == 0);
        assert!(ghost >= width / 2);

        let tree = Tree::new(bounds);

        let mut result = Self {
            tree,
            width,
            ghost,

            blocks: TreeBlocks::default(),
            block_node_offsets: Vec::default(),

            neighbors: TreeNeighbors::default(),

            interfaces: Vec::default(),
            interface_node_offsets: Vec::default(),
            interface_masks: Vec::default(),

            refine_flags: Vec::new(),
            coarsen_flags: Vec::new(),

            regrid_map: Vec::default(),
            old_blocks: TreeBlocks::default(),
            old_block_node_offsets: Vec::default(),

            stores: ThreadLocal::new(),
        };

        result.build();

        result
    }

    /// Rebuilds mesh from current tree.
    fn build(&mut self) {
        self.blocks.build(&self.tree);
        self.neighbors.build(&self.tree, &self.blocks);
        self.build_block_node_offests();
        self.build_interfaces();
        self.build_flags();
    }

    /// Computes block node offests, assuming blocks have already been built.
    fn build_block_node_offests(&mut self) {
        // Reset offsets
        self.block_node_offsets.clear();

        let mut cursor = 0;

        for block in 0..self.num_blocks() {
            self.block_node_offsets.push(cursor);

            let size = self.blocks.size(block);

            let nodes_in_block =
                NodeSpace::<N>::new(array::from_fn(|axis| size[axis] * self.width), self.ghost)
                    .num_nodes();

            cursor += nodes_in_block;
        }

        self.block_node_offsets.push(cursor);
    }

    /// Allocates requisite space for refinement and coarsening flags.
    fn build_flags(&mut self) {
        self.refine_flags.clear();
        self.coarsen_flags.clear();

        self.refine_flags.resize(self.num_cells(), false);
        self.coarsen_flags.resize(self.num_cells(), false);
    }

    /// Retrieves the Quadtree this mesh is built on top of.
    pub fn tree(&self) -> &Tree<N> {
        &self.tree
    }

    pub fn refine_flags(&self) -> &[bool] {
        self.refine_flags.as_slice()
    }

    pub fn balance_flags(&mut self) {
        // Propogate refinement flags outwards.
        self.tree.balance_refine_flags(&mut self.refine_flags);
        // Refinement has priority over coarsening. Ensure that there is never a cell marked
        // for refinement next to a equal or coarser cell marked for coarsening.
        for cell in 0..self.num_cells() {
            if self.refine_flags[cell] {
                let level = self.tree.level(cell);
                for neighbor in self.tree.neighborhood(cell) {
                    let nlevel = self.tree.level(neighbor);
                    if nlevel <= level {
                        self.coarsen_flags[neighbor] = false;
                    }
                }
            }
        }
        // Unmark coarsening flags as necessary.
        self.tree.balance_coarsen_flags(&mut self.coarsen_flags);
    }

    /// Refines the mesh using currently set flags.
    pub fn refine(&mut self) {
        self.regrid_map.clear();

        // Save old information
        self.old_blocks.clone_from(&self.blocks);
        self.old_block_node_offsets
            .clone_from(&self.block_node_offsets);

        // Perform regriding
        self.regrid_map.resize(self.tree.num_cells(), 0);

        let mut coarsen_map = vec![0; self.tree.num_cells()];
        self.tree
            .coarsen_index_map(&self.coarsen_flags, &mut coarsen_map);
        self.tree.coarsen(&self.coarsen_flags);

        let mut refine_map = vec![0; self.tree.num_cells()];
        let mut flags = vec![false; self.tree.num_cells()];

        for (old, &new) in coarsen_map.iter().enumerate() {
            flags[new] = self.refine_flags[old];
        }

        self.tree.refine_index_map(&flags, &mut refine_map);
        self.tree.refine(&flags);

        self.tree
            .refine_index_map(&self.refine_flags, &mut self.regrid_map);

        for i in 0..self.regrid_map.len() {
            self.regrid_map[i] = refine_map[coarsen_map[i]];
        }

        // Rebuild mesh
        self.build();
    }

    /// Flags every cell for refinement, then performs the operation.
    pub fn refine_global(&mut self) {
        self.refine_flags.fill(true);
        self.refine();
    }

    /// Returns the total number of blocks on the mesh.
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the total number of blocks on the mesh before the most recent refinement.
    pub(crate) fn num_old_blocks(&self) -> usize {
        self.old_blocks.len()
    }

    /// Returns the total number of cells on the mesh.
    pub fn num_cells(&self) -> usize {
        self.tree.num_cells()
    }

    /// Returns the total number of nodes on the mesh.
    pub fn num_nodes(&self) -> usize {
        *self.block_node_offsets.last().unwrap()
    }

    /// Returns the total number of nodes on the mesh before the most recent refinement.
    pub(crate) fn num_old_nodes(&self) -> usize {
        *self.old_block_node_offsets.last().unwrap_or(&0)
    }

    /// The range of nodes assigned to a given block.
    pub fn block_nodes(&self, block: usize) -> Range<usize> {
        self.block_node_offsets[block]..self.block_node_offsets[block + 1]
    }

    /// The range of nodes assigned to a given block on the mesh before the most recent refinement.
    pub(crate) fn old_block_nodes(&self, block: usize) -> Range<usize> {
        self.old_block_node_offsets[block]..self.old_block_node_offsets[block + 1]
    }

    /// Returns the window of nodes in a block corresponding to a given cell,
    /// including `self.width/2` of padding along each face.
    pub fn element_window(&self, cell: usize) -> NodeWindow<N> {
        let position = self.blocks.cell_position(cell);

        let size = [2 * self.width + 1; N];
        let mut origin = [-(self.width as isize) / 2; N];

        for axis in 0..N {
            origin[axis] += (self.width * position[axis]) as isize
        }

        NodeWindow { origin, size }
    }

    /// Returns the window of nodes in a block corresponding to a given cell, including
    /// no padding.
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
        let cell_size = array::from_fn(|axis| size[axis] * self.width);

        NodeSpace::new(cell_size, self.ghost)
    }

    /// Computes the nodespace corresponding to a block on the mesh before the most recent refinement.
    pub(crate) fn old_block_space(&self, block: usize) -> NodeSpace<N> {
        let size = self.old_blocks.size(block);
        let cell_size = array::from_fn(|axis| size[axis] * self.width);

        NodeSpace::new(cell_size, self.ghost)
    }

    /// The bounds of a block.
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

    /// Produces a block boundary for the mesh before its most recent refinement.
    pub(crate) fn old_block_boundary<B>(&self, block: usize, boundary: B) -> BlockBoundary<N, B> {
        BlockBoundary {
            flags: self.old_blocks.boundary_flags(block),
            inner: boundary,
        }
    }

    /// The level of a given block.
    pub fn block_level(&self, block: usize) -> usize {
        self.blocks.level(block)
    }

    /// The level of a block before the most recent refinement.
    pub(crate) fn old_block_level(&self, block: usize) -> usize {
        self.old_blocks.level(block)
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

    pub fn set_refine_flag(&mut self, cell: usize) {
        self.refine_flags[cell] = true
    }

    pub fn set_coarsen_flag(&mut self, cell: usize) {
        self.coarsen_flags[cell] = true
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
            domain.size[axis] / self.width as f64 / 2_f64.powi(max_level as i32)
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

    pub fn old_block_compute<F: Fn(&Self, &MeshStore, usize) + Sync>(&mut self, f: F) {
        (0..self.num_old_blocks())
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

        // for interface in self.interfaces.fine() {
        //     writeln!(
        //         result,
        //         "Fine Interface {} -> {}",
        //         interface.block, interface.neighbor
        //     )
        //     .unwrap();
        //     writeln!(
        //         result,
        //         "    Source {}: Origin {:?}, Size {:?}",
        //         interface.neighbor, interface.source, interface.size
        //     )
        //     .unwrap();
        //     writeln!(
        //         result,
        //         "    Dest {}: Origin {:?}, Size {:?}",
        //         interface.block, interface.dest, interface.size
        //     )
        //     .unwrap();
        // }

        writeln!(result, "").unwrap();
        writeln!(result, "// Direct Interfaces").unwrap();

        // for interface in self.interfaces.direct() {
        //     writeln!(
        //         result,
        //         "Direct Interface {} -> {}",
        //         interface.block, interface.neighbor
        //     )
        //     .unwrap();
        //     writeln!(
        //         result,
        //         "    Source {}: Origin {:?}, Size {:?}",
        //         interface.neighbor, interface.source, interface.size
        //     )
        //     .unwrap();
        //     writeln!(
        //         result,
        //         "    Dest {}: Origin {:?}, Size {:?}",
        //         interface.block, interface.dest, interface.size
        //     )
        //     .unwrap();
        // }

        writeln!(result, "").unwrap();
        writeln!(result, "// Coarse Interfaces").unwrap();

        // for interface in self.interfaces.coarse() {
        //     writeln!(
        //         result,
        //         "Coarse Interface {} -> {}",
        //         interface.block, interface.neighbor
        //     )
        //     .unwrap();
        //     writeln!(
        //         result,
        //         "    Source {}: Origin {:?}, Size {:?}",
        //         interface.neighbor, interface.source, interface.size
        //     )
        //     .unwrap();
        //     writeln!(
        //         result,
        //         "    Dest {}: Origin {:?}, Size {:?}",
        //         interface.block, interface.dest, interface.size
        //     )
        //     .unwrap();
        // }
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

            block_node_offsets: self.block_node_offsets.clone(),

            interfaces: self.interfaces.clone(),
            interface_node_offsets: self.interface_node_offsets.clone(),
            interface_masks: self.interface_masks.clone(),

            refine_flags: self.refine_flags.clone(),
            coarsen_flags: self.coarsen_flags.clone(),

            regrid_map: self.regrid_map.clone(),
            old_blocks: self.old_blocks.clone(),
            old_block_node_offsets: self.old_block_node_offsets.clone(),

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
            block_node_offsets: Vec::default(),
            neighbors: TreeNeighbors::default(),

            interfaces: Vec::default(),
            interface_node_offsets: Vec::default(),
            interface_masks: Vec::default(),

            refine_flags: Vec::default(),
            coarsen_flags: Vec::default(),

            regrid_map: Vec::default(),
            old_blocks: TreeBlocks::default(),
            old_block_node_offsets: Vec::default(),

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

        for (name, system) in config.systems.int_fields.iter() {
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
                name: format!("IntField::{}", name),
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
