//! Module containing the `Mesh` API, the main datastructure through which one
//! writes finite difference programs.
//!
//! `Mesh`s are the central driver of all finite difference codes, and provide many methods
//! for discretizing domains, approximating differential operators, applying boundary conditions,
//! filling interior interfaces, and adaptively regridding a domain based on various error heuristics.

use crate::geometry::{
    faces, ActiveCellId, AxisMask, BlockId, Face, FaceArray, FaceMask, IndexSpace, Rectangle, Tree,
    TreeBlocks, TreeInterfaces, TreeNeighbors, TreeNodes,
};
use crate::kernel::{BoundaryClass, DirichletParams, Element};
use crate::{
    kernel::{BoundaryKind, NodeSpace, NodeWindow},
    system::SystemBoundaryConds,
};
use num_traits::ToPrimitive;
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use ron::ser::PrettyConfig;
use std::collections::HashMap;
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
mod function;
mod regrid;
mod store;
mod transfer;

pub use checkpoint::{MeshCheckpoint, SystemCheckpoint};
pub use function::{Engine, Function, Gaussian, Projection};
pub use store::MeshStore;

use crate::system::{System, SystemSlice};

/// A discretization of a rectangular axis aligned grid into a collection of uniform grids of nodes
/// with different spacings. A `Mesh` is built on top of a Quadtree, allowing one to selectively
/// refine areas of interest without wasting computational power on smoother regions of the domain.
///
/// This abstraction also handles multithread dispatch and sharing nodes between threads in a
/// an effecient manner. This allows the user to write generic, sequential, and straighforward code
/// on the main thread, while still maximising performance and fully utilizing computational resources.
#[derive(Debug)]
pub struct Mesh<const N: usize> {
    /// Underlying Tree on which the mesh is built.
    tree: Tree<N>,
    /// The width of a cell on the mesh (i.e. how many subcells are in that cell).
    width: usize,
    /// The number of ghost cells used to facilitate inter-block communication.
    ghost: usize,
    /// `BoundaryClass` for each face. Restricts what kinds of boundary condition
    /// (encoded in `BoundaryKind`) may be enforced on that face.
    boundary: FaceArray<N, BoundaryClass>,

    /// Maximum level of cell on the mesh.
    max_level: usize,

    /// Block structure induced by the tree.
    blocks: TreeBlocks<N>,
    /// Nodes assigned to each block
    nodes: TreeNodes<N>,
    /// Neighbors of each block.
    neighbors: TreeNeighbors<N>,
    /// Neighbors translated into interfaces
    interfaces: TreeInterfaces<N>,

    /// Refinement flags for each cell on the mesh.
    refine_flags: Vec<bool>,
    /// Coarsening flags for each cell on the mesh.
    coarsen_flags: Vec<bool>,

    /// Map from cells before refinement to current cells.
    regrid_map: Vec<ActiveCellId>,

    /// Blocks before most recent refinement.
    old_blocks: TreeBlocks<N>,
    /// Block node offsets from before most recent refinement.
    old_nodes: TreeNodes<N>,
    /// Cell splits from before most recent refinement.
    ///
    /// May be temporary if I can find a more elegant solution.
    old_cell_splits: Vec<AxisMask<N>>,

    // ********************************
    // Caches *************************
    /// Thread-local stores used for allocation.
    stores: ThreadLocal<UnsafeCell<MeshStore>>,
    /// Cache for uniform elements
    elements: HashMap<(usize, usize), Element<N>>,
}

impl<const N: usize> Mesh<N> {
    /// Constructs a new `Mesh` covering the domain, with a number of nodes
    /// defined by `width` and `ghost`.
    pub fn new(
        bounds: Rectangle<N>,
        width: usize,
        ghost: usize,
        boundary: FaceArray<N, BoundaryClass>,
    ) -> Self {
        assert!(width % 2 == 0);
        assert!(ghost == width / 2);

        let mut tree = Tree::new(bounds);

        for axis in 0..N {
            let negative_periodic =
                matches!(boundary[Face::negative(axis)], BoundaryClass::Periodic);
            let positive_periodic =
                matches!(boundary[Face::positive(axis)], BoundaryClass::Periodic);

            assert_eq!(
                negative_periodic, positive_periodic,
                "Periodicity on a given axis must match"
            );

            tree.set_periodic(axis, negative_periodic && positive_periodic);
        }

        let mut result = Self {
            tree,
            width,
            ghost,
            boundary,

            max_level: 0,

            blocks: TreeBlocks::default(),
            nodes: TreeNodes::new([width; N], ghost),
            neighbors: TreeNeighbors::default(),
            interfaces: TreeInterfaces::default(),

            refine_flags: Vec::new(),
            coarsen_flags: Vec::new(),

            regrid_map: Vec::default(),
            old_blocks: TreeBlocks::default(),
            old_nodes: TreeNodes::new([width; N], ghost),
            old_cell_splits: Vec::default(),

            stores: ThreadLocal::new(),
            elements: HashMap::default(),
        };

        result.build();

        result
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn ghost(&self) -> usize {
        self.ghost
    }

    // pub fn set_boundary_class(&mut self, face: Face<N>, class: BoundaryClass) {
    //     self.boundary[face] = class;
    // }

    // pub fn set_boundary_classes(&mut self, class: BoundaryClass) {
    //     for face in faces() {
    //         self.set_boundary_class(face, class);
    //     }
    // }

    /// Rebuilds mesh from current tree.
    fn build(&mut self) {
        self.tree.build();
        self.blocks.build(&self.tree);

        self.max_level = 0;
        for block in self.blocks.indices() {
            self.max_level = self.max_level.max(self.blocks.level(block));
        }

        self.nodes.build(&self.blocks);
        self.neighbors.build(&self.tree, &self.blocks);
        self.interfaces
            .build(&self.tree, &self.blocks, &self.neighbors, &self.nodes);
        self.build_flags();
    }

    /// Allocates requisite space for refinement and coarsening flags.
    fn build_flags(&mut self) {
        self.refine_flags.clear();
        self.coarsen_flags.clear();

        self.refine_flags
            .resize(self.tree.num_active_cells(), false);
        self.coarsen_flags
            .resize(self.tree.num_active_cells(), false);
    }

    // *******************************
    // Global Info *******************

    /// Retrieves the Quadtree this mesh is built on top of.
    pub fn tree(&self) -> &Tree<N> {
        &self.tree
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
    pub fn num_active_cells(&self) -> usize {
        self.tree.num_active_cells()
    }

    /// Returns the total number of nodes on the mesh.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the total number of nodes on the mesh before the most recent refinement.
    pub(crate) fn num_old_nodes(&self) -> usize {
        self.old_nodes.len()
    }

    // *******************************
    // Data for each block ***********

    /// The range of nodes assigned to a given block.
    pub fn block_nodes(&self, block: BlockId) -> Range<usize> {
        self.nodes.range(block)
    }

    /// The range of nodes assigned to a given block on the mesh before the most recent refinement.
    pub(crate) fn old_block_nodes(&self, block: BlockId) -> Range<usize> {
        self.old_nodes.range(block)
    }

    /// Computes the nodespace corresponding to a block.
    pub fn block_space(&self, block: BlockId) -> NodeSpace<N> {
        let size = self.blocks.size(block);
        let cell_size = array::from_fn(|axis| size[axis] * self.width);

        NodeSpace {
            size: cell_size,
            ghost: self.ghost,
            boundary: self.block_boundary_classes(block),
            bounds: self.block_bounds(block),
        }
    }

    /// Computes the nodespace corresponding to a block on the mesh before the most recent refinement.
    pub(crate) fn old_block_space(&self, block: BlockId) -> NodeSpace<N> {
        let size = self.old_blocks.size(block);
        let cell_size = array::from_fn(|axis| size[axis] * self.width);

        NodeSpace {
            size: cell_size,
            ghost: self.ghost,
            bounds: Rectangle::UNIT,
            boundary: self.old_block_boundary_classes(block),
        }
    }

    /// The bounds of a block.
    pub fn block_bounds(&self, block: BlockId) -> Rectangle<N> {
        self.blocks.bounds(block)
    }

    /// Computes flags indicating whether a particular face of a block borders a physical
    /// boundary.
    pub fn block_physical_boundary_flags(&self, block: BlockId) -> FaceMask<N> {
        self.blocks.boundary_flags(block)
    }

    /// Indicates what class of boundary condition is enforced along each face of the block.
    pub fn block_boundary_classes(&self, block: BlockId) -> FaceArray<N, BoundaryClass> {
        let flag = self.block_physical_boundary_flags(block);

        FaceArray::from_fn(|face| {
            if flag.is_set(face) {
                self.boundary[face]
            } else {
                BoundaryClass::Ghost
            }
        })
    }

    /// Produces a block boundary which correctly accounts for
    /// interior interfaces.
    pub fn block_bcs<B: SystemBoundaryConds<N>>(
        &self,
        block: BlockId,
        bcs: B,
    ) -> BlockBoundaryConds<N, B> {
        BlockBoundaryConds {
            inner: bcs,
            physical_boundary_flags: self.block_physical_boundary_flags(block),
        }
    }

    /// Produces a block ghost flags for the mesh before its most recent refinement.
    pub(crate) fn old_block_boundary_classes(&self, block: BlockId) -> FaceArray<N, BoundaryClass> {
        let flag = self.old_blocks.boundary_flags(block);

        FaceArray::from_fn(|face| {
            if flag.is_set(face) {
                self.boundary[face]
            } else {
                BoundaryClass::Ghost
            }
        })
    }

    /// The level of a given block.
    pub fn block_level(&self, block: BlockId) -> usize {
        self.blocks.level(block)
    }

    /// The level of a block before the most recent refinement.
    pub(crate) fn old_block_level(&self, block: BlockId) -> usize {
        self.old_blocks.level(block)
    }

    // *******************************
    // Elements **********************

    /// Element associated with a given cell.
    pub fn element_window(&self, cell: ActiveCellId) -> NodeWindow<N> {
        let position = self.blocks.active_cell_position(cell);

        let size = [2 * self.width + 1; N];
        let mut origin = [(self.width as isize) / 2 - self.width as isize; N];

        for axis in 0..N {
            origin[axis] += (self.width * position[axis]) as isize
        }

        NodeWindow { origin, size }
    }

    /// Returns the window of nodes in a block corresponding to a given cell, including
    /// no padding.
    pub fn element_coarse_window(&self, cell: ActiveCellId) -> NodeWindow<N> {
        let position = self.blocks.active_cell_position(cell);

        let size = [self.width + 1; N];
        let mut origin = [0; N];

        for axis in 0..N {
            origin[axis] += (self.width * position[axis]) as isize
        }

        NodeWindow { origin, size }
    }

    /// Retrieves an element from the mesh's element cache.
    pub fn request_element(&mut self, width: usize, order: usize) -> Element<N> {
        self.elements
            .remove(&(width, order))
            .unwrap_or_else(|| Element::uniform(width, order))
    }

    /// Reinserts an element into the mesh's element cache.
    pub fn replace_element(&mut self, element: Element<N>) {
        _ = self
            .elements
            .insert((element.width(), element.order()), element)
    }

    /// Retrieves the number of nodes along each axis of a cell.
    /// This defaults to `[self.width; N]` but is increased by one
    /// if the cell lies along a block boundary for a given axis.
    pub fn cell_node_size(&self, cell: ActiveCellId) -> [usize; N] {
        let block = self.blocks.active_cell_block(cell);
        let size = self.blocks.size(block);
        let position = self.blocks.active_cell_position(cell);

        array::from_fn(|axis| {
            if position[axis] == size[axis] - 1 {
                self.width + 1
            } else {
                self.width
            }
        })
    }

    /// Returns the origin of a cell in its block's `NodeSpace<N>`.
    pub fn cell_node_origin(&self, cell: ActiveCellId) -> [usize; N] {
        let position = self.blocks.active_cell_position(cell);
        array::from_fn(|axis| position[axis] * self.width)
    }

    /// Returns true if the given cell is on a boundary that does not contain
    /// ghost nodes. If this is the case we must fall back to a lower order element
    /// error approximation.
    pub fn cell_needs_coarse_element(&self, cell: ActiveCellId) -> bool {
        let block = self.blocks.active_cell_block(cell);
        let block_size = self.blocks.size(block);
        let boundary = self.block_boundary_classes(block);
        let position = self.blocks.active_cell_position(cell);

        for face in faces::<N>() {
            let border = if face.side {
                block_size[face.axis] - 1
            } else {
                0
            };
            let on_border = position[face.axis] == border;

            if !boundary[face].has_ghost() && on_border {
                return true;
            }
        }

        false
    }

    /// Returns the maximum level of cell on the mesh.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Returns the minimum spatial distance between any
    /// two nodes on the mesh. Commonly used in conjunction
    /// with a CFL factor to determine time step.
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

    /// Computes the spacing on a particular block (albeit not accounting for coarse-fine interfaces).
    pub fn block_spacing(&self, block: BlockId) -> f64 {
        let space = self.block_space(block);

        space
            .spacing()
            .iter()
            .min_by(|a, b| f64::total_cmp(a, b))
            .cloned()
            .unwrap_or(1.0)
    }

    /// Runs a computation in parallel on every single block in the mesh, providing
    /// a `MeshStore` object for allocating scratch data.
    pub fn block_compute<F: Fn(&Self, &MeshStore, BlockId) + Sync>(&mut self, f: F) {
        self.blocks
            .indices()
            .par_bridge()
            .into_par_iter()
            .for_each(|block| {
                let store = unsafe { &mut *self.stores.get_or_default().get() };
                f(self, store, block);
                store.reset();
            });
    }

    /// Runs a computation in parallel on every single old block in the mesh, providing
    /// a `MeshStore` object for allocating scratch data.
    pub(crate) fn old_block_compute<F: Fn(&Self, &MeshStore, BlockId) + Sync>(&mut self, f: F) {
        (0..self.num_old_blocks())
            .map(BlockId)
            .par_bridge()
            .into_par_iter()
            .for_each(|block| {
                let store = unsafe { &mut *self.stores.get_or_default().get() };
                f(self, store, block);
                store.reset();
            });
    }

    /// Computes the maximum l2 norm of all fields in the system.
    pub fn l2_norm_system<S: System>(&mut self, source: SystemSlice<S>) -> f64 {
        source
            .system()
            .enumerate()
            .map(|label| self.l2_norm(source.field(label)))
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    /// Computes the maximum l-infinity norm of all fields in the system.
    pub fn max_norm_system<S: System>(&mut self, source: SystemSlice<S>) -> f64 {
        source
            .system()
            .enumerate()
            .map(|label| self.max_norm(source.field(label)))
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    /// Returns the value of a function at the bottom left corner of
    /// the mesh.
    pub fn bottom_left_value(&self, src: &[f64]) -> f64 {
        let space = self.block_space(BlockId(0));
        let nodes = self.block_nodes(BlockId(0));

        src[nodes][space.index_from_vertex([0; N])]
    }

    /// Computes the l2 norm of a field on the mesh.
    pub fn l2_norm(&mut self, src: &[f64]) -> f64 {
        let mut result = 0.0;

        for block in self.blocks.indices() {
            let space = self.block_space(block);
            let size = space.cell_size();

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

            for spacing in space.spacing() {
                block_result *= spacing;
            }

            result += block_result;
        }

        result.sqrt()
    }

    /// Computes the l-infinity norm of a field on a mesh.
    pub fn max_norm(&mut self, src: &[f64]) -> f64 {
        let mut result = 0.0f64;

        for block in self.blocks.indices() {
            let space = self.block_space(block);
            let data = &src[self.block_nodes(block)];

            let mut block_result = 0.0f64;

            for node in space.inner_window() {
                let index = space.index_from_node(node);
                block_result = block_result.max(data[index]);
            }

            result = result.max(block_result);
        }

        result
    }

    /// Writes a textual summary of the Mesh to a sink. This is pimrarily used to
    /// debug features of the mesh that can't be easily represented graphically (i.e in
    /// .vtu files).
    pub fn write_debug(&self, mut result: impl Write) {
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Cells ****************").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result).unwrap();

        for cell in self.tree.active_cell_indices() {
            writeln!(result, "Cell {}", cell.0).unwrap();
            writeln!(result, "    Bounds {:?}", self.tree.active_bounds(cell)).unwrap();
            writeln!(
                result,
                "    Block {:?}",
                self.blocks.active_cell_block(cell)
            )
            .unwrap();
            writeln!(
                result,
                "    Block Position {:?}",
                self.blocks.active_cell_position(cell)
            )
            .unwrap();
        }

        writeln!(result).unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Blocks ***************").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result).unwrap();

        for block in self.blocks.indices() {
            writeln!(result, "Block {block:?}").unwrap();
            writeln!(result, "    Bounds {:?}", self.blocks.bounds(block)).unwrap();
            writeln!(result, "    Size {:?}", self.blocks.size(block)).unwrap();
            writeln!(result, "    Cells {:?}", self.blocks.active_cells(block)).unwrap();
            writeln!(
                result,
                "    Vertices {:?}",
                self.block_space(block).cell_size()
            )
            .unwrap();
            writeln!(
                result,
                "    Boundary {:?}",
                self.blocks.boundary_flags(block)
            )
            .unwrap();
        }

        writeln!(result).unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Neighbors ************").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result).unwrap();

        writeln!(result, "// Fine Neighbors").unwrap();

        for neighbor in self.neighbors.fine() {
            writeln!(result, "Fine Neighbor").unwrap();

            writeln!(
                result,
                "    Block: {:?}, Neighbor: {:?}",
                neighbor.block, neighbor.neighbor,
            )
            .unwrap();
            writeln!(
                result,
                "    Lower: Cell {}, Neighbor {}, Region {}",
                neighbor.a.cell.0, neighbor.a.neighbor.0, neighbor.a.region,
            )
            .unwrap();
            writeln!(
                result,
                "    Upper: Cell {}, Neighbor {}, Region {}",
                neighbor.b.cell.0, neighbor.b.neighbor.0, neighbor.b.region,
            )
            .unwrap();
        }

        writeln!(result).unwrap();
        writeln!(result, "// Direct Neighbors").unwrap();

        for neighbor in self.neighbors.direct() {
            writeln!(result, "Direct Neighbor").unwrap();

            writeln!(
                result,
                "    Block: {:?}, Neighbor: {:?}",
                neighbor.block, neighbor.neighbor,
            )
            .unwrap();
            writeln!(
                result,
                "    Lower: Cell {}, Neighbor {}, Region {}",
                neighbor.a.cell.0, neighbor.a.neighbor.0, neighbor.a.region,
            )
            .unwrap();
            writeln!(
                result,
                "    Upper: Cell {}, Neighbor {}, Region {}",
                neighbor.b.cell.0, neighbor.b.neighbor.0, neighbor.b.region,
            )
            .unwrap();
        }

        writeln!(result).unwrap();
        writeln!(result, "// Coarse Neighbors").unwrap();

        for neighbor in self.neighbors.coarse() {
            writeln!(result, "Coarse Neighbor").unwrap();

            writeln!(
                result,
                "    Block: {:?}, Neighbor: {:?}",
                neighbor.block, neighbor.neighbor,
            )
            .unwrap();
            writeln!(
                result,
                "    Lower: Cell {}, Neighbor {}, Region {}",
                neighbor.a.cell.0, neighbor.a.neighbor.0, neighbor.a.region,
            )
            .unwrap();
            writeln!(
                result,
                "    Upper: Cell {}, Neighbor {}, Region {}",
                neighbor.b.cell.0, neighbor.b.neighbor.0, neighbor.b.region,
            )
            .unwrap();
        }

        writeln!(result).unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Interfaces ***********").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result).unwrap();
    }
}

impl<const N: usize> Clone for Mesh<N> {
    fn clone(&self) -> Self {
        Self {
            tree: self.tree.clone(),
            width: self.width,
            ghost: self.ghost,
            boundary: self.boundary.clone(),

            max_level: self.max_level,

            blocks: self.blocks.clone(),
            nodes: self.nodes.clone(),
            neighbors: self.neighbors.clone(),
            interfaces: self.interfaces.clone(),

            refine_flags: self.refine_flags.clone(),
            coarsen_flags: self.coarsen_flags.clone(),

            regrid_map: self.regrid_map.clone(),
            old_blocks: self.old_blocks.clone(),
            old_nodes: self.old_nodes.clone(),
            old_cell_splits: self.old_cell_splits.clone(),

            stores: ThreadLocal::new(),
            elements: HashMap::default(),
        }
    }
}

impl<const N: usize> Default for Mesh<N> {
    fn default() -> Self {
        let mut result = Self {
            tree: Tree::new(Rectangle::UNIT),
            width: 4,
            ghost: 1,
            boundary: FaceArray::default(),

            max_level: 0,

            blocks: TreeBlocks::default(),
            nodes: TreeNodes::new([4; N], 1),
            neighbors: TreeNeighbors::default(),
            interfaces: TreeInterfaces::default(),

            refine_flags: Vec::default(),
            coarsen_flags: Vec::default(),

            regrid_map: Vec::default(),
            old_blocks: TreeBlocks::default(),
            old_nodes: TreeNodes::new([4; N], 1),
            old_cell_splits: Vec::default(),

            stores: ThreadLocal::new(),
            elements: HashMap::default(),
        };

        result.build();

        result
    }
}

#[derive(Clone, Debug)]
pub struct ExportVtuConfig {
    pub title: String,
    pub ghost: bool,
    pub stride: usize,
}

impl Default for ExportVtuConfig {
    fn default() -> Self {
        Self {
            title: "Title".to_string(),
            ghost: false,
            stride: 1,
        }
    }
}

impl<const N: usize> Mesh<N> {
    /// Exports a copy of the mesh and associated fields to disk, stored as a dat file.
    pub fn export_dat(
        &self,
        path: impl AsRef<Path>,
        checkpoint: &SystemCheckpoint,
    ) -> io::Result<()> {
        let mut grid = MeshCheckpoint::default();
        grid.save_mesh(self);

        let data = ron::ser::to_string_pretty::<(MeshCheckpoint<N>, SystemCheckpoint)>(
            &(grid, checkpoint.clone()),
            PrettyConfig::default(),
        )
        .map_err(io::Error::other)?;
        let mut file = File::create(path)?;
        file.write_all(data.as_bytes())
    }

    /// Loads the mesh and any additional data from disk.
    pub fn import_dat(
        &mut self,
        path: impl AsRef<Path>,
        systems: &mut SystemCheckpoint,
    ) -> io::Result<()> {
        let mut contents: String = String::new();
        let mut file = File::open(path)?;
        file.read_to_string(&mut contents)?;

        let (grid, checkpoint): (MeshCheckpoint<N>, SystemCheckpoint) =
            ron::from_str(&contents).map_err(io::Error::other)?;

        grid.load_mesh(self);
        systems.clone_from(&checkpoint);

        Ok(())
    }

    /// Exports the mesh and additional field data to a .vtu files, for visualisation in applications like
    /// Paraview.
    pub fn export_vtu(
        &self,
        path: impl AsRef<Path>,
        checkpoint: &SystemCheckpoint,
        config: ExportVtuConfig,
    ) -> io::Result<()> {
        const {
            assert!(N > 0 && N <= 2, "Vtu Output only supported for 0 < N ≤ 2");
        }

        let mut stride = config.stride;

        if config.stride == 0 {
            stride = self.width;
        }

        assert!(self.width % stride == 0);
        assert!(stride <= self.width);

        if config.ghost {
            assert!(self.ghost % stride == 0);
        }

        // Generate Cells
        let mut connectivity = Vec::new();
        let mut offsets = Vec::new();

        let mut vertex_total = 0;
        let mut cell_total = 0;

        for block in self.blocks.indices() {
            let space = self.block_space(block);

            let mut cell_size = space.cell_size();
            let mut vertex_size = space.vertex_size();

            if config.ghost {
                for axis in 0..N {
                    cell_size[axis] += 2 * space.ghost();
                    vertex_size[axis] += 2 * space.ghost();
                }
            }

            for axis in 0..N {
                debug_assert!(cell_size[axis] % stride == 0);
                debug_assert!((vertex_size[axis] - 1) % stride == 0);

                cell_size[axis] = cell_size[axis] / stride;
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

        let cells = Cells {
            cell_verts: VertexNumbers::XML {
                connectivity,
                offsets,
            },
            types: vec![CellType::Quad; cell_total],
        };

        // Generate point data
        let mut vertices = Vec::new();

        for block in self.blocks.indices() {
            let space = self.block_space(block);
            let window = if config.ghost {
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

        let points = IOBuffer::new(vertices);

        // Attributes
        let mut attributes = Attributes {
            point: Vec::new(),
            cell: Vec::new(),
        };

        for (name, system) in checkpoint.systems.iter() {
            for (idx, field) in system.fields.iter().enumerate() {
                let start = idx * system.count;
                let end = idx * system.count + system.count;

                attributes.point.push(self.field_attribute(
                    format!("{}::{}", name, field),
                    &system.buffer[start..end],
                    config.ghost,
                    stride,
                ));
            }
        }

        for (name, system) in checkpoint.fields.iter() {
            attributes.point.push(self.field_attribute(
                format!("Field::{}", name),
                system,
                config.ghost,
                stride,
            ));
        }

        for (name, system) in checkpoint.int_fields.iter() {
            attributes.point.push(self.field_attribute(
                format!("IntField::{}", name),
                system,
                config.ghost,
                stride,
            ));
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
            v => {
                log::error!("Encountered error {:?} while exporting vtu", v);
                io::Error::from(io::ErrorKind::Other)
            }
        })?;

        Ok(())
    }

    fn field_attribute<T: ToPrimitive + Copy + 'static>(
        &self,
        name: String,
        data: &[T],
        ghost: bool,
        stride: usize,
    ) -> Attribute {
        let mut buffer = Vec::new();

        for block in self.blocks.indices() {
            let space = self.block_space(block);
            let nodes = self.block_nodes(block);
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
                let value = data[nodes.start + index];
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

#[derive(Clone, Debug)]
pub struct BlockBoundaryConds<const N: usize, I> {
    inner: I,
    /// Physical boundary mask for various faces.
    physical_boundary_flags: FaceMask<N>,
}

impl<const N: usize, I: SystemBoundaryConds<N>> SystemBoundaryConds<N>
    for BlockBoundaryConds<N, I>
{
    type System = I::System;

    fn kind(&self, label: <Self::System as System>::Label, face: Face<N>) -> BoundaryKind {
        if self.physical_boundary_flags.is_set(face) {
            self.inner.kind(label, face)
        } else {
            BoundaryKind::Custom
        }
    }

    fn dirichlet(
        &self,
        label: <Self::System as System>::Label,
        position: [f64; N],
    ) -> DirichletParams {
        self.inner.dirichlet(label, position)
    }

    fn radiative(
        &self,
        label: <Self::System as System>::Label,
        position: [f64; N],
    ) -> crate::prelude::RadiativeParams {
        self.inner.radiative(label, position)
    }
}
