//! Module containing the `Mesh` API, the main datastructure through which one
//! writes finite difference programs.
//!
//! `Mesh`s are the central driver of all finite difference codes, and provide many methods
//! for discretizing domains, approximating differential operators, applying boundary conditions,
//! filling interior interfaces, and adaptively regridding a domain based on various error heuristics.

use crate::geometry::{
    BlockId, CellId, Face, FaceMask, HyperBox, IndexSpace, LeafId, Region, Side, Split, Tree,
    TreeBlocks, TreeInterfaces, TreeNeighbors, TreeSer,
};
use crate::image::{ImageMut, ImageRef};
use crate::kernel::{
    Boundary, BoundaryClass, BoundaryClasses, BoundaryKind, DirichletParams, Element, NodeSpace,
    NodeWindow, node_from_vertex,
};
use datasize::DataSize;
use rand::{
    Rng,
    distr::{Distribution, StandardUniform},
};
use reborrow::ReborrowMut as _;

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelBridge, ParallelIterator};

use std::collections::HashMap;
use std::{array, fmt::Write, ops::Range};

mod checkpoint;
mod evaluate;
mod function;
mod regrid;
mod store;
mod transfer;

pub use checkpoint::{Checkpoint, ExportStride, ExportVtuConfig};
pub use function::{Engine, Function, FunctionBorrowMut, Gaussian, Projection, TanH};
pub use store::{MeshStore, UnsafeThreadCache};

/// A discretization of a rectangular axis aligned grid into a collection of uniform grids of nodes
/// with different spacings. A `Mesh` is built on top of a Quadtree, allowing one to selectively
/// refine areas of interest without wasting computational power on smoother regions of the domain.
///
/// This abstraction also handles multithread dispatch and sharing nodes between threads in a
/// an effecient manner. This allows the user to write generic, sequential, and straighforward code
/// on the main thread, while still maximising performance and fully utilizing computational resources.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(from = "MeshSer<N>", into = "MeshSer<N>")]
pub struct Mesh<const N: usize> {
    /// Underlying Tree on which the mesh is built.
    tree: Tree<N>,
    /// The width of a cell on the mesh (i.e. how many subcells are in that cell).
    width: usize,
    /// The number of ghost cells used to facilitate inter-block communication.
    ghost: usize,
    /// `BoundaryClass` for each face. Restricts what kinds of boundary condition
    /// (encoded in `BoundaryKind`) may be enforced on that face.
    boundary: BoundaryClasses<N>,

    /// Block structure induced by the tree.
    blocks: TreeBlocks<N>,
    /// Neighbors of each block.
    neighbors: TreeNeighbors<N>,
    /// Neighbors translated into interfaces
    interfaces: TreeInterfaces<N>,

    /// Refinement flags for each cell on the mesh.
    refine_flags: Vec<bool>,
    /// Coarsening flags for each cell on the mesh.
    coarsen_flags: Vec<bool>,

    /// Map from leaves before refinement to current leaves.
    regrid_map: Vec<LeafId>,

    /// Blocks before most recent refinement.
    old_blocks: TreeBlocks<N>,
    /// Cell splits from before most recent refinement.
    ///
    /// May be temporary if I can find a more elegant solution.
    old_cell_splits: Vec<Split<N>>,

    // ********************************
    // Caches *************************
    /// Thread-local stores used for allocation.
    stores: UnsafeThreadCache<MeshStore>,
    /// Cache for uniform elements
    elements: HashMap<(usize, usize), Element<N>>,
}

impl<const N: usize> Mesh<N> {
    /// Constructs a new `Mesh` covering the domain, with a number of nodes
    /// defined by `width` and `ghost`.
    pub fn new(
        bounds: HyperBox<N>,
        width: usize,
        ghost: usize,
        boundary: BoundaryClasses<N>,
    ) -> Self {
        assert!(width >= 2);
        assert!(width % 2 == 0);
        assert!(ghost >= width / 2);

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

            blocks: TreeBlocks::new([width; N], ghost),
            neighbors: TreeNeighbors::default(),
            interfaces: TreeInterfaces::default(),

            refine_flags: Vec::new(),
            coarsen_flags: Vec::new(),

            regrid_map: Vec::default(),
            old_blocks: TreeBlocks::new([width; N], ghost),
            old_cell_splits: Vec::default(),

            stores: UnsafeThreadCache::new(),
            elements: HashMap::default(),
        };

        result.build();

        result
    }

    /// Retrieves width of individual cells in this mesh.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Retrieves the number of ghost nodes on each cell of a mesh.
    pub fn ghost(&self) -> usize {
        self.ghost
    }

    /// Rebuilds mesh from current tree.
    fn build(&mut self) {
        debug_assert_eq!(self.blocks.width(), [self.width; N]);
        debug_assert_eq!(self.blocks.ghost(), self.ghost());
        debug_assert_eq!(self.old_blocks.width(), [self.width; N]);
        debug_assert_eq!(self.old_blocks.ghost(), self.ghost());

        // Rebuild tree
        self.tree.build();
        // Rebuild blocks
        self.blocks.build(&self.tree);

        // Rebuild neighbors
        self.neighbors.build(&self.tree, &self.blocks);
        // Rebuild interfaces
        self.interfaces
            .build(&self.tree, &self.blocks, &self.neighbors);
        // Resize flags, clearing value to false.
        self.build_flags();
    }

    /// Allocates requisite space for refinement and coarsening flags.
    fn build_flags(&mut self) {
        self.refine_flags.clear();
        self.coarsen_flags.clear();

        self.refine_flags.resize(self.tree.num_leaves(), false);
        self.coarsen_flags.resize(self.tree.num_leaves(), false);
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
    pub fn num_leaves(&self) -> usize {
        self.tree.num_leaves()
    }

    /// Returns the total number of nodes on the mesh.
    pub fn num_nodes(&self) -> usize {
        self.blocks.num_nodes()
    }

    pub fn num_dofs(&self) -> usize {
        let mut result = 0;

        for block in 0..self.num_blocks() {
            let mut size = self.blocks.size(BlockId(block));

            for axis in 0..N {
                size[axis] *= self.width;

                if self
                    .blocks
                    .boundary_flags(BlockId(block))
                    .is_set(Face::positive(axis))
                {
                    size[axis] += 1;
                }
            }

            result += size.iter().product::<usize>();
        }

        result
    }

    pub fn num_nodes_per_cell(&self) -> usize {
        (self.width + 1).pow(N as u32)
    }

    /// Returns the total number of nodes on the mesh before the most recent refinement.
    pub(crate) fn num_old_nodes(&self) -> usize {
        self.old_blocks.num_nodes()
    }

    /// Returns the boundary classes associated with each boundary of the physical domain.
    pub fn boundary_classes(&self) -> BoundaryClasses<N> {
        self.boundary.clone()
    }

    // *******************************
    // Data for each block ***********

    /// Returns underlying `TreeBlocks<N>` object.
    pub fn blocks(&self) -> &TreeBlocks<N> {
        &self.blocks
    }

    /// The range of nodes assigned to a given block.
    pub fn block_nodes(&self, block: BlockId) -> Range<usize> {
        self.blocks.nodes(block)
    }

    /// The range of nodes assigned to a given block on the mesh before the most recent refinement.
    pub(crate) fn old_block_nodes(&self, block: BlockId) -> Range<usize> {
        self.old_blocks.nodes(block)
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
            bounds: HyperBox::UNIT,
            boundary: self.old_block_boundary_classes(block),
        }
    }

    /// Computes the bounds of a block.
    pub fn block_bounds(&self, block: BlockId) -> HyperBox<N> {
        self.blocks.bounds(block)
    }

    /// Computes flags indicating whether a particular face of a block borders a physical
    /// boundary.
    pub fn block_physical_boundary_flags(&self, block: BlockId) -> FaceMask<N> {
        self.blocks.boundary_flags(block)
    }

    /// Indicates what class of boundary condition is enforced along each face of the block.
    pub fn block_boundary_classes(&self, block: BlockId) -> BoundaryClasses<N> {
        let flag = self.block_physical_boundary_flags(block);

        BoundaryClasses::from_fn(|face| {
            if flag.is_set(face) {
                self.boundary[face]
            } else {
                BoundaryClass::Ghost
            }
        })
    }

    /// Produces a block boundary which correctly accounts for
    /// interior interfaces.
    pub fn block_bcs<B: Boundary<N>>(&self, block: BlockId, bcs: B) -> BlockBoundary<N, B> {
        BlockBoundary {
            inner: bcs,
            physical_boundary_flags: self.block_physical_boundary_flags(block),
        }
    }

    /// Produces a block ghost flags for the mesh before its most recent refinement.
    pub(crate) fn old_block_boundary_classes(&self, block: BlockId) -> BoundaryClasses<N> {
        let flag = self.old_blocks.boundary_flags(block);

        BoundaryClasses::from_fn(|face| {
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
    // Node windows ******************

    /// Finds bounds associated with a node window.
    pub fn window_bounds(&self, block: BlockId, window: NodeWindow<N>) -> HyperBox<N> {
        debug_assert!(self.block_space(block).contains_window(window));
        let block_size = self.blocks.node_size(block);
        let bounds = self.blocks.bounds(block);

        HyperBox {
            size: array::from_fn(|axis| {
                bounds.size[axis] * window.size[axis] as f64 / block_size[axis] as f64
            }),
            origin: array::from_fn(|axis| {
                bounds.size[axis] * window.origin[axis] as f64 / block_size[axis] as f64
            }),
        }
    }

    /// Retrieves the node window associated with a certain active cell on its block.
    pub fn leaf_window(&self, cell: LeafId) -> NodeWindow<N> {
        NodeWindow {
            origin: node_from_vertex(self.leaf_node_origin(cell)),
            size: [self.width + 1; N],
        }
    }

    /// Retrieves the node window that is optimal for interpolating values to the given position,
    /// lying within the given leaf.
    pub fn interpolate_window(&self, cell: LeafId, position: [f64; N]) -> NodeWindow<N> {
        let cell_offset = self.blocks.leaf_position(cell);
        let cell_bounds = self.tree().leaf_bounds(cell);
        let cell_origin: [_; N] = array::from_fn(|axis| (self.width * cell_offset[axis]) as isize);

        debug_assert!(cell_bounds.contains(position));

        let local = cell_bounds.global_to_local(position);
        let local_node: [_; N] =
            array::from_fn(|axis| (local[axis] * self.width as f64).round() as isize);
        let local_origin: [_; N] =
            array::from_fn(|axis| local_node[axis] - self.width as isize / 2);

        NodeWindow {
            origin: array::from_fn(|axis| cell_origin[axis] + local_origin[axis]),
            size: [self.width + 1; N],
        }
    }

    /// Element associated with a leaf.
    pub fn element_window(&self, cell: LeafId) -> NodeWindow<N> {
        let position = self.blocks.leaf_position(cell);
        // Round ghost to nearest even number to make sure diagonal coefficients of element
        // actually correspond with newly refined points.
        let buffer = 2 * (self.ghost / 2);
        debug_assert!(buffer <= self.ghost);

        let size = [self.width + 2 * buffer + 1; N];
        let mut origin = [-(buffer as isize); N];

        for axis in 0..N {
            origin[axis] += (self.width * position[axis]) as isize
        }

        NodeWindow { origin, size }
    }

    /// Returns the window of nodes in a block corresponding to a given cell, including
    /// no padding.
    pub fn element_coarse_window(&self, cell: LeafId) -> NodeWindow<N> {
        let position = self.blocks.leaf_position(cell);

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
    pub fn cell_node_size(&self, cell: LeafId) -> [usize; N] {
        let block = self.blocks.block_from_leaf(cell);
        let size = self.blocks.size(block);
        let position = self.blocks.leaf_position(cell);

        array::from_fn(|axis| {
            if position[axis] == size[axis] - 1 {
                self.width + 1
            } else {
                self.width
            }
        })
    }

    /// Returns the origin of an active cell in its block's `NodeSpace<N>`.
    pub fn leaf_node_origin(&self, cell: LeafId) -> [usize; N] {
        let position = self.blocks.leaf_position(cell);
        array::from_fn(|axis| position[axis] * self.width)
    }

    /// Returns true if the given cell is on a boundary that does not contain
    /// ghost nodes. If this is the case we must fall back to a lower order element
    /// error approximation.
    pub fn cell_needs_coarse_element(&self, cell: LeafId) -> bool {
        let block = self.blocks.block_from_leaf(cell);
        let block_size = self.blocks.size(block);
        let boundary = self.block_boundary_classes(block);
        let position = self.blocks.leaf_position(cell);

        for face in Face::iterate() {
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

    // ***********************************
    // Global info

    /// Returns number of levels on the mesh.
    pub fn num_levels(&self) -> usize {
        self.tree().num_levels()
    }

    /// Returns the minimum spatial distance between any
    /// two nodes on the mesh. Commonly used in conjunction
    /// with a CFL factor to determine time step.
    pub fn min_spacing(&self) -> f64 {
        let max_level = self.num_levels() - 1;
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
        #[cfg(feature = "parallel")]
        self.blocks
            .indices()
            .par_bridge()
            .into_par_iter()
            .for_each(|block| {
                let store = unsafe { self.stores.get_or_default() };
                f(self, store, block);
                store.reset();
            });

        #[cfg(not(feature = "parallel"))]
        self.blocks.indices().for_each(|block| {
            let store = unsafe { self.stores.get_or_default() };
            f(self, store, block);
            store.reset();
        });
    }

    /// Runs a (possibily failable) computation in parallel on every single block in the mesh.
    pub fn try_block_compute<E: Send, F: Fn(&Self, &MeshStore, BlockId) -> Result<(), E> + Sync>(
        &mut self,
        f: F,
    ) -> Result<(), E> {
        #[cfg(feature = "parallel")]
        return self
            .blocks
            .indices()
            .par_bridge()
            .into_par_iter()
            .try_for_each(|block| {
                let store = unsafe { self.stores.get_or_default() };
                let result = f(self, store, block);
                store.reset();
                result
            });

        #[cfg(not(feature = "parallel"))]
        return self.blocks.indices().try_for_each(|block| {
            let store = unsafe { self.stores.get_or_default() };
            let result = f(self, store, block);
            store.reset();
            result
        });
    }

    /// Runs a computation in parallel on every single old block in the mesh, providing
    /// a `MeshStore` object for allocating scratch data.
    pub(crate) fn old_block_compute<F: Fn(&Self, &MeshStore, BlockId) + Sync>(&mut self, f: F) {
        #[cfg(feature = "parallel")]
        (0..self.num_old_blocks())
            .map(BlockId)
            .par_bridge()
            .into_par_iter()
            .for_each(|block| {
                let store = unsafe { self.stores.get_or_default() };
                f(self, store, block);
                store.reset();
            });

        #[cfg(not(feature = "parallel"))]
        (0..self.num_old_blocks()).map(BlockId).for_each(|block| {
            let store = unsafe { self.stores.get_or_default() };
            f(self, store, block);
            store.reset();
        });
    }

    /// Computes the maximum l2 norm of all fields in the system.
    pub fn l2_norm_system(&mut self, source: ImageRef) -> f64 {
        source
            .channels()
            .map(|label| self.l2_norm(source.channel(label)))
            .max_by(f64::total_cmp)
            .unwrap()
    }

    /// Computes the maximum l-infinity norm of all fields in the system.
    pub fn max_norm_system(&mut self, source: ImageRef) -> f64 {
        source
            .channels()
            .map(|label| self.linf_norm(source.channel(label)))
            .max_by(f64::total_cmp)
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

    pub fn oscillation_heuristic(&mut self, src: &[f64]) -> f64 {
        let mut result: f64 = 0.0;

        for block in self.blocks.indices() {
            let space = self.block_space(block);
            let cell_size = self.blocks.size(block);

            let data = &src[self.block_nodes(block)];

            for cell in IndexSpace::new(cell_size).iter() {
                let node_offset: [usize; N] = array::from_fn(|i| self.width * cell[i]);
                let node_size = [self.width + 1; N];

                let mut maximum = f64::NEG_INFINITY;
                let mut minimum = f64::INFINITY;

                for local in IndexSpace::new(node_size).iter() {
                    let global: [_; N] = array::from_fn(|axis| node_offset[axis] + local[axis]);
                    let index = space.index_from_vertex(global);

                    minimum = minimum.min(data[index]);
                    maximum = maximum.max(data[index]);
                }

                let cell_result = (maximum - minimum).abs();

                result += cell_result;
            }
        }

        result
    }

    pub fn oscillation_heuristic_system(&mut self, source: ImageRef) -> f64 {
        source
            .channels()
            .map(|cidx| self.oscillation_heuristic(source.channel(cidx)))
            .max_by(f64::total_cmp)
            .unwrap()
    }

    /// Computes the l-infinity norm of a field on a mesh.
    pub fn linf_norm(&mut self, src: &[f64]) -> f64 {
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

        for cell in self.tree.leaves() {
            writeln!(result, "Cell {}", cell.0).unwrap();
            writeln!(result, "    Bounds {:?}", self.tree.leaf_bounds(cell)).unwrap();
            writeln!(result, "    Block {:?}", self.blocks.block_from_leaf(cell)).unwrap();
            writeln!(
                result,
                "    Block Position {:?}",
                self.blocks.leaf_position(cell)
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
            writeln!(result, "    Cells {:?}", self.blocks.leaves(block)).unwrap();
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

    /// Load base node values (corresponding to a (self.width + 1)^N grid subdividing this cell) into dest from source.
    pub fn load_nodes_for_cell(
        &self,
        cell: CellId,
        source: ImageRef,
        dest: ImageMut,
        edge: [bool; N],
    ) {
        debug_assert!(source.num_nodes() == self.num_nodes());
        debug_assert!(source.num_channels() == dest.num_channels());
        debug_assert!(dest.num_nodes() == (self.width + 1).pow(N as u32));
        debug_assert!(self.width.is_power_of_two());

        self.load_nodes_for_cell_recusive(cell, 0, source, dest, [0; N], edge);
    }

    /// Load node values (corresponding to a (self.width + 1)^N grid subdividing this leaf) into dest from source.
    pub fn load_nodes_for_leaf(
        &self,
        leaf: LeafId,
        source: ImageRef,
        mut dest: ImageMut,
        edge: [bool; N],
    ) {
        let dest_space = IndexSpace::new([self.width + 1; N]);
        let block = self.blocks.block_from_leaf(leaf);
        let block_space = self.block_space(block);
        let block_node_origin = self.leaf_node_origin(leaf);
        let block_node_index_offset = self.block_nodes(block).start;

        for dest_node in
            IndexSpace::<N>::new(array::from_fn(|axis| self.width + edge[axis] as usize))
        {
            let dest_index = dest_space.linear_from_cartesian(dest_node);

            let block_node = array::from_fn(|axis| block_node_origin[axis] + dest_node[axis]);
            let source_index = block_node_index_offset + block_space.index_from_vertex(block_node);

            for channel in dest.channels() {
                dest.channel_mut(channel)[dest_index] = source.channel(channel)[source_index]
            }
        }
    }

    fn load_nodes_for_cell_recusive(
        &self,
        cell: CellId,
        depth: usize,
        source: ImageRef,
        mut dest: ImageMut,
        dest_origin: [usize; N],
        edge: [bool; N],
    ) {
        // IndexSpace for converting dest vertices into linear indices into dest.
        let dest_space = IndexSpace::new([self.width + 1; N]);
        // Width of current cell in base vertices.
        let current_cell_width = self.width >> depth;

        // First check if this cell has children
        if let Some(children) = self.tree.child_offset(cell) {
            // let w = self.width;
            // println!("{current_cell_width} {w} {depth}");
            if current_cell_width >> 1 == 0 {
                // We are at the minimum cell width, just load bottom left node (and corners if necessary)
                debug_assert!(current_cell_width == 1);
                // Iterate over chosen corners
                for split in Split::<N>::enumerate().filter(|split| {
                    (0..N)
                        .into_iter()
                        .all(|axis| !split.is_set(axis) || edge[axis])
                }) {
                    let dest_index = dest_space.linear_from_cartesian(array::from_fn(|axis| {
                        dest_origin[axis] + split.is_set(axis) as usize
                    }));
                    let source_index = self.cell_corner_node_index(cell, split.unpack());

                    // Set dest values
                    for channel in dest.channels() {
                        dest.channel_mut(channel)[dest_index] =
                            source.channel(channel)[source_index]
                    }
                }
                return;
            }

            // Recurse to children, masking edges as necessary
            for split in Split::<N>::enumerate() {
                let child = CellId(children.0 + split.to_linear());

                self.load_nodes_for_cell_recusive(
                    child,
                    depth + 1,
                    source,
                    dest.rb_mut(),
                    array::from_fn(|i| {
                        dest_origin[i] + split.is_set(i) as usize * (current_cell_width >> 1)
                    }),
                    array::from_fn(|axis| edge[axis] && split.is_set(axis)),
                );
            }
            return;
        }

        // This is a leaf, load all necessary points into dest
        let leaf = self.tree.contained_leaves(cell).next().unwrap();
        let block = self.blocks.block_from_leaf(leaf);
        let block_space = self.block_space(block);
        let block_node_origin = self.leaf_node_origin(leaf);
        let block_node_index_offset = self.block_nodes(block).start;

        for dest_offset in IndexSpace::<N>::new(array::from_fn(|axis| {
            current_cell_width + edge[axis] as usize
        })) {
            let block_node =
                array::from_fn(|axis| block_node_origin[axis] + (dest_offset[axis] << depth));
            let source_index = block_node_index_offset + block_space.index_from_vertex(block_node);
            let dest_index = dest_space.linear_from_cartesian(array::from_fn(|axis| {
                dest_origin[axis] + dest_offset[axis]
            }));

            for channel in dest.channels() {
                dest.channel_mut(channel)[dest_index] = source.channel(channel)[source_index]
            }
        }
    }

    /// Returns the index of the bottom left node for the given cell.
    pub fn cell_bottom_left_node_indexz(&self, cell: CellId) -> usize {
        let leaf = self.tree.contained_leaves(cell).next().unwrap();
        let block = self.blocks.block_from_leaf(leaf);
        let node_origin = self.leaf_node_origin(leaf);
        let offset = self.block_space(block).index_from_vertex(node_origin);
        self.blocks.nodes(block).start + offset
    }

    /// Returns the index of a corner node for the given cell.
    pub fn cell_corner_node_index(&self, mut cell: CellId, corner: [bool; N]) -> usize {
        let region = Region::new(array::from_fn(|axis| {
            if corner[axis] {
                Side::Right
            } else {
                crate::geometry::Side::Middle
            }
        }));

        if let Some(neighbor) = self.tree.neighbor_region(cell, region) {
            self.cell_bottom_left_node_indexz(neighbor)
        } else {
            let split = Split::pack(corner);

            while let Some(child) = self.tree.child(cell, split) {
                cell = child;
            }

            let leaf = self.tree.contained_leaves(cell).next().unwrap();
            self.leaf_corner_node_index(leaf, corner)
        }
    }

    pub fn leaf_corner_node_index(&self, leaf: LeafId, corner: [bool; N]) -> usize {
        let block = self.blocks.block_from_leaf(leaf);
        let mut node_origin = self.leaf_node_origin(leaf);
        for axis in 0..N {
            node_origin[axis] += corner[axis] as usize * self.width
        }
        let offset = self.block_space(block).index_from_vertex(node_origin);
        self.blocks.nodes(block).start + offset
    }

    pub fn load_coarse_nodes_for_block(
        &self,
        block: BlockId,
        source: ImageRef,
        mut dest: ImageMut,
        mut scratch: ImageMut,
    ) {
        debug_assert_eq!(source.num_nodes(), self.num_nodes());
        debug_assert_eq!(scratch.num_nodes(), (self.width + 1).pow(N as u32));

        let block_leaves = self.blocks().leaves(block);
        let block_size_in_leaves = self.blocks().size(block);
        let block_space_in_leaves = IndexSpace::new(block_size_in_leaves);
        let block_boundary_flags = self.block_physical_boundary_flags(block);
        let block_level = self.blocks().level(block);
        let block_space = self.block_space(block);
        let block_source = source.slice(self.block_nodes(block));

        let block_coarse_space = NodeSpace {
            size: block_size_in_leaves.map(|size| size * self.width() / 2),
            ghost: self.width() / 2,
            bounds: self.block_bounds(block),
            boundary: self.block_boundary_classes(block),
        };
        let block_num_coarse_nodes = block_coarse_space.num_nodes();

        debug_assert_eq!(dest.num_nodes(), block_num_coarse_nodes);

        let cell_space = IndexSpace::new([self.width + 1; N]);

        let mut coarse_window = block_coarse_space.inner_window();

        for axis in 0..N {
            if !block_boundary_flags.is_set(Face::positive(axis)) {
                coarse_window.size[axis] -= 1;
            }
        }

        for coarse in coarse_window {
            let block_node_index = block_space.index_from_node(coarse.map(|i| 2 * i));
            let coarse_index = block_coarse_space.index_from_node(coarse);

            for channel in source.channels() {
                dest.channel_mut(channel)[coarse_index] =
                    block_source.channel(channel)[block_node_index];
            }
        }

        'regions: for region in Region::<N>::enumerate() {
            if region == Region::CENTRAL {
                continue 'regions;
            }

            // Skip if on physical boundary
            for face in region.adjacent_faces() {
                if block_boundary_flags.is_set(face) {
                    continue 'regions;
                }
            }

            for adjacent_position in block_space_in_leaves.region_adjacent_window(region) {
                let adjacent_leaf =
                    block_leaves[block_space_in_leaves.linear_from_cartesian(adjacent_position)];
                let adjacent_cell = self.tree().cell_from_leaf(adjacent_leaf);

                let neighbor_cell = self.tree().neighbor_region(adjacent_cell, region).unwrap();
                let neighbor_level = self.tree().level(neighbor_cell);

                let mut coarse_node_origin =
                    adjacent_position.map(|position| (position * self.width() / 2) as isize);

                for axis in 0..N {
                    match region.side(axis) {
                        Side::Left => coarse_node_origin[axis] -= self.width() as isize / 2,
                        Side::Right => coarse_node_origin[axis] += self.width() as isize / 2,
                        Side::Middle => {}
                    }
                }

                debug_assert!(neighbor_level <= block_level);

                let edge = std::array::from_fn(|axis| {
                    (region.side(axis) == Side::Right)
                        || block_boundary_flags.is_set(Face::positive(axis))
                });

                let coarse_space = IndexSpace::new([self.width() / 2 + 1; N]);
                let mut coarse_window = coarse_space.window();
                for axis in 0..N {
                    coarse_window.size[axis] -= (!edge[axis]) as usize;
                }

                if neighbor_level == block_level {
                    self.load_nodes_for_cell(neighbor_cell, source, scratch.rb_mut(), edge);

                    for offset in coarse_window {
                        let cell_node = cell_space.linear_from_cartesian(offset.map(|o| 2 * o));
                        let coarse_node =
                            block_coarse_space.index_from_node(array::from_fn(|axis| {
                                coarse_node_origin[axis] + offset[axis] as isize
                            }));

                        for channel in source.channels() {
                            dest.channel_mut(channel)[coarse_node] =
                                scratch.channel(channel)[cell_node];
                        }
                    }
                } else {
                    let neighbor_leaf = self.tree().leaf_from_cell(neighbor_cell).unwrap();
                    let neighbor_block = self.blocks().block_from_leaf(neighbor_leaf);
                    let neighbor_space = self.block_space(neighbor_block);
                    let neighbor_block_node_offset = self.block_nodes(neighbor_block).start;

                    let mut neighbor_split =
                        self.tree().most_recent_leaf_split(adjacent_leaf).unwrap();
                    for axis in 0..N {
                        if region.side(axis) != Side::Middle {
                            neighbor_split.toggle(axis);
                        }
                    }

                    let mut neighbor_node_origin = self.leaf_node_origin(neighbor_leaf);
                    for axis in 0..N {
                        neighbor_node_origin[axis] +=
                            neighbor_split.is_set(axis) as usize * self.width / 2;
                    }

                    // We now have the split of the neighbor that we want
                    for offset in coarse_window {
                        let neighbor_node = neighbor_block_node_offset
                            + neighbor_space.index_from_vertex(array::from_fn(|axis| {
                                neighbor_node_origin[axis] + offset[axis]
                            }));
                        let coarse_node =
                            block_coarse_space.index_from_node(array::from_fn(|axis| {
                                coarse_node_origin[axis] + offset[axis] as isize
                            }));

                        for channel in source.channels() {
                            dest.channel_mut(channel)[coarse_node] =
                                source.channel(channel)[neighbor_node];
                        }
                    }
                }
            }
        }
    }
}

impl<const N: usize> Clone for Mesh<N> {
    fn clone(&self) -> Self {
        Self {
            tree: self.tree.clone(),
            width: self.width,
            ghost: self.ghost,
            boundary: self.boundary,

            blocks: self.blocks.clone(),
            neighbors: self.neighbors.clone(),
            interfaces: self.interfaces.clone(),

            refine_flags: self.refine_flags.clone(),
            coarsen_flags: self.coarsen_flags.clone(),

            regrid_map: self.regrid_map.clone(),
            old_blocks: self.old_blocks.clone(),
            old_cell_splits: self.old_cell_splits.clone(),

            stores: UnsafeThreadCache::default(),
            elements: HashMap::default(),
        }
    }
}

impl<const N: usize> Default for Mesh<N> {
    fn default() -> Self {
        let mut result = Self {
            tree: Tree::new(HyperBox::UNIT),
            width: 4,
            ghost: 1,
            boundary: BoundaryClasses::default(),

            blocks: TreeBlocks::new([4; N], 1),
            neighbors: TreeNeighbors::default(),
            interfaces: TreeInterfaces::default(),

            refine_flags: Vec::default(),
            coarsen_flags: Vec::default(),

            regrid_map: Vec::default(),
            old_blocks: TreeBlocks::new([4; N], 1),
            old_cell_splits: Vec::default(),

            stores: UnsafeThreadCache::default(),
            elements: HashMap::default(),
        };

        result.build();

        result
    }
}

impl<const N: usize> DataSize for Mesh<N> {
    const IS_DYNAMIC: bool = false;
    const STATIC_HEAP_SIZE: usize = 0;

    fn estimate_heap_size(&self) -> usize {
        self.tree.estimate_heap_size()
            + self.blocks.estimate_heap_size()
            + self.neighbors.estimate_heap_size()
            + self.interfaces.estimate_heap_size()
            + self.refine_flags.estimate_heap_size()
            + self.coarsen_flags.estimate_heap_size()
            + self.regrid_map.estimate_heap_size()
            + self.old_blocks.estimate_heap_size()
            + self.old_cell_splits.estimate_heap_size()
    }
}

impl<const N: usize> Distribution<Mesh<N>> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Mesh<N> {
        let width = rng.random_range(1..=3);
        let axes: [_; N] = array::from_fn(|_| rng.random());

        Mesh::new(
            rng.random(),
            1 << width,
            1 << (width - 1),
            BoundaryClasses::from_fn(|face| axes[face.axis]),
        )
    }
}

/// Helper for serializing meshes using minimal data.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
struct MeshSer<const N: usize> {
    tree: TreeSer<N>,
    width: usize,
    ghost: usize,
    boundary: BoundaryClasses<N>,
}

impl<const N: usize> From<Mesh<N>> for MeshSer<N> {
    fn from(value: Mesh<N>) -> Self {
        MeshSer {
            tree: value.tree.into(),
            width: value.width,
            ghost: value.ghost,
            boundary: value.boundary,
        }
    }
}

impl<const N: usize> From<MeshSer<N>> for Mesh<N> {
    fn from(value: MeshSer<N>) -> Self {
        let mut result = Mesh {
            tree: value.tree.into(),
            width: value.width,
            ghost: value.ghost,
            boundary: value.boundary,
            blocks: TreeBlocks::new([value.width; N], value.ghost),
            old_blocks: TreeBlocks::new([value.width; N], value.ghost),

            ..Default::default()
        };
        result.build();
        result
    }
}

#[derive(Clone, Debug)]
pub struct BlockBoundary<const N: usize, I> {
    inner: I,
    /// Physical boundary mask for various faces.
    physical_boundary_flags: FaceMask<N>,
}

impl<const N: usize, I: Boundary<N>> Boundary<N> for BlockBoundary<N, I> {
    fn kind(&self, channel: usize, face: Face<N>) -> BoundaryKind {
        if self.physical_boundary_flags.is_set(face) {
            self.inner.kind(channel, face)
        } else {
            BoundaryKind::Custom
        }
    }

    fn dirichlet(&self, channel: usize, position: [f64; N]) -> DirichletParams {
        self.inner.dirichlet(channel, position)
    }

    fn radiative(&self, channel: usize, position: [f64; N]) -> crate::prelude::RadiativeParams {
        self.inner.radiative(channel, position)
    }
}

#[cfg(test)]
mod tests {
    use crate::image::Image;

    use super::*;
    use rand::{Rng, SeedableRng as _};
    #[test]
    fn fuzz_serialize() -> eyre::Result<()> {
        let mut mesh = Mesh::<2>::new(
            HyperBox::UNIT,
            6,
            3,
            [
                [BoundaryClass::Ghost, BoundaryClass::OneSided],
                [BoundaryClass::Ghost, BoundaryClass::OneSided],
            ]
            .into(),
        );
        mesh.refine_global();

        // Randomly refine mesh
        let mut rng = rand::rng();
        for _ in 0..4 {
            mesh.set_refine_flags_random(&mut rng);
            mesh.balance_flags();
            mesh.regrid();
        }

        // Serialize tree
        let data = ron::to_string(&mesh)?;
        let mesh2: Mesh<2> = ron::from_str(data.as_str())?;

        assert_eq!(mesh.tree, mesh2.tree);
        assert_eq!(mesh.width, mesh2.width);
        assert_eq!(mesh.ghost, mesh2.ghost);
        assert_eq!(mesh.boundary, mesh2.boundary);
        assert_eq!(mesh.blocks, mesh2.blocks);
        assert_eq!(mesh.neighbors, mesh2.neighbors);
        assert_eq!(mesh.interfaces, mesh2.interfaces);

        Ok(())
    }

    #[test]
    fn load_nodes_for_cell_vs_leaf() -> eyre::Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1984);
        for _ in 0..10 {
            let mut mesh: Mesh<2> = rng.random();
            let num_node_per_cell = (mesh.width() + 1).pow(2);

            mesh.refine_global();
            mesh.refine_global();

            for _ in 0..5 {
                mesh.set_refine_flags_random(&mut rng);
                mesh.balance_flags();
                mesh.regrid();
            }

            let mut source = Image::new(1, mesh.num_nodes());
            rng.fill(&mut source);

            let mut cell_dest = Image::new(1, num_node_per_cell);
            let mut leaf_dest = Image::new(1, num_node_per_cell);

            cell_dest.channel_mut(0).fill(0.0);
            leaf_dest.channel_mut(0).fill(0.0);

            for leaf in mesh.tree.leaves() {
                let edge = [rng.random(), rng.random()];

                let cell = mesh.tree.cell_from_leaf(leaf);
                mesh.load_nodes_for_cell(cell, source.as_ref(), cell_dest.as_mut(), edge);
                mesh.load_nodes_for_leaf(leaf, source.as_ref(), leaf_dest.as_mut(), edge);

                assert_eq!(cell_dest.channel(0), leaf_dest.channel(0));
            }
        }

        Ok(())
    }

    #[test]
    fn analytic_load_node_for_cells() -> eyre::Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
        let mut mesh: Mesh<2> = Mesh::new(
            HyperBox::UNIT,
            4,
            2,
            BoundaryClasses::splat(BoundaryClass::Ghost),
        );
        let num_node_per_cell = (mesh.width() + 1).pow(2);

        mesh.refine_global();
        mesh.refine_global();

        for _ in 0..5 {
            mesh.set_refine_flags_random(&mut rng);
            mesh.balance_flags();
            mesh.regrid();
        }

        let mut source = Image::new(1, mesh.num_nodes());
        rng.fill(&mut source);

        mesh.project(
            Gaussian {
                center: [0.5, 0.5],
                amplitude: 1.0,
                sigma: [1.0, 1.0],
            },
            source.as_mut(),
        );

        let mut base_image = Image::new(1, num_node_per_cell);
        base_image.channel_mut(0).fill(0.0);

        mesh.load_nodes_for_cell(
            CellId(0),
            source.as_ref(),
            base_image.as_mut(),
            [true, true],
        );

        let base_space = IndexSpace::new([mesh.width() + 1; 2]);

        for node in base_space {
            let index = base_space.linear_from_cartesian(node);
            let position: [_; 2] = array::from_fn(|axis| node[axis] as f64 / mesh.width() as f64);
            let value = (-(position[0] - 0.5).powi(2) - (position[1] - 0.5).powi(2)).exp();

            assert_eq!(value, base_image.channel(0)[index]);
        }

        Ok(())
    }

    #[test]
    fn analytic_load_coarse_nodes_for_blocks() -> eyre::Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(2024);
        let mut mesh: Mesh<2> = Mesh::new(HyperBox::UNIT, 4, 2, BoundaryClasses::GHOST);

        mesh.refine_global();
        mesh.refine_global();

        for _ in 0..5 {
            mesh.set_refine_flags_random(&mut rng);
            mesh.balance_flags();
            mesh.regrid();
        }

        let mut source = Image::new(1, mesh.num_nodes());
        mesh.project(
            Gaussian {
                center: [0.5, 0.5],
                amplitude: 1.0,
                sigma: [1.0, 1.0],
            },
            source.as_mut(),
        );

        let mut scratch = Image::new(1, mesh.num_nodes_per_cell());

        for block in mesh.blocks().indices() {
            let block_size_in_leaves = mesh.blocks().size(block);
            let block_boundary_flags = mesh.block_physical_boundary_flags(block);
            let block_coarse_space = NodeSpace {
                size: block_size_in_leaves.map(|size| size * mesh.width() / 2),
                ghost: mesh.width() / 2,
                bounds: mesh.block_bounds(block),
                boundary: mesh.block_boundary_classes(block),
            };

            let mut dest = Image::new(1, block_coarse_space.num_nodes());
            dest.channel_mut(0).fill(f64::NAN);
            mesh.load_coarse_nodes_for_block(
                block,
                source.as_ref(),
                dest.as_mut(),
                scratch.as_mut(),
            );

            let mut coarse_window = block_coarse_space.inner_window();

            for axis in 0..2 {
                if !block_boundary_flags.is_set(Face::negative(axis)) {
                    coarse_window.origin[axis] -= mesh.width() as isize / 2;
                    coarse_window.size[axis] += mesh.width() / 2;
                }
            }

            for axis in 0..2 {
                if !block_boundary_flags.is_set(Face::positive(axis)) {
                    coarse_window.size[axis] += mesh.width() / 2;
                }
            }

            println!("Block: {block:.?}");
            println!("{block_coarse_space:.?}");
            for face in Face::iterate() {
                println!("{face:.?} is {}", block_boundary_flags.is_set(face));
            }

            for coarse in coarse_window {
                let index = block_coarse_space.index_from_node(coarse);
                let position = block_coarse_space.position(coarse);

                let value = (-(position[0] - 0.5).powi(2) - (position[1] - 0.5).powi(2)).exp();
                println!("Position: {coarse:?}");
                assert_eq!(value, dest.channel(0)[index]);
            }
        }

        Ok(())
    }

    #[derive(Clone)]
    struct Custom;

    impl Boundary<2> for Custom {
        fn kind(&self, _: usize, _: Face<2>) -> BoundaryKind {
            BoundaryKind::Custom
        }
    }

    #[test]
    fn fuzz_transfer_vs_load_nodes_for_cell() -> eyre::Result<()> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1984);
        for _ in 0..10 {
            let mut mesh: Mesh<2> = Mesh::new(HyperBox::UNIT, 4, 2, BoundaryClasses::GHOST);
            let num_node_per_cell = (mesh.width() + 1).pow(2);

            mesh.refine_global();
            mesh.refine_global();

            for _ in 0..4 {
                mesh.set_refine_flags_random(&mut rng);
                mesh.balance_flags();
                mesh.regrid();
            }

            let mut source = Image::new(1, mesh.num_nodes());
            rng.fill(&mut source);

            mesh.fill_boundary(4, Custom, source.as_mut());

            println!("{}", mesh.num_levels());

            let base_level = rng.random_range(0..=2);
            // Find a cell on the base_level
            let base_cell = mesh
                .tree()
                .level_cells(base_level)
                .nth(rng.random_range(0..mesh.tree().num_cells_on_level(base_level)))
                .unwrap();
            // let base_cell = CellId(0);
            let mut base_image = Image::new(1, num_node_per_cell);
            base_image.channel_mut(0).fill(0.0);
            let base_space = IndexSpace::new([mesh.width() + 1; 2]);

            let edge = [rng.random(), rng.random()];
            mesh.load_nodes_for_cell(base_cell, source.as_ref(), base_image.as_mut(), edge);

            while mesh.num_levels() > base_level + 1 {
                // Coarsen global but don't coarsen base_cell
                mesh.coarsen_flags.fill(true);
                mesh.refine_flags.fill(false);
                if let Some(leaf) = mesh.tree().leaf_from_cell(base_cell) {
                    mesh.coarsen_flags[leaf.0] = false;
                }
                mesh.balance_flags();
                mesh.regrid();

                let mut tmp = Image::new(1, mesh.num_nodes());
                mesh.transfer(4, source.as_ref(), tmp.as_mut());
                source.clone_from(&tmp);
                mesh.fill_boundary(4, Custom, source.as_mut());
            }

            assert_eq!(mesh.num_levels(), base_level + 1);

            let base_leaf = mesh.tree().leaf_from_cell(base_cell).unwrap();
            let base_block = mesh.blocks().block_from_leaf(base_leaf);
            let base_block_space = mesh.block_space(base_block);
            let base_block_source = source.slice(mesh.block_nodes(base_block));
            let base_block_node_origin = mesh.leaf_node_origin(base_leaf);

            let mut window = base_space.window();
            for axis in 0..2 {
                window.size[axis] -= !edge[axis] as usize;
            }

            for cart in window {
                let base_index = base_space.linear_from_cartesian(cart);
                let block_node_index = base_block_space.index_from_vertex(array::from_fn(|axis| {
                    base_block_node_origin[axis] + cart[axis]
                }));

                assert_eq!(
                    base_image.channel(0)[base_index],
                    base_block_source.channel(0)[block_node_index]
                );
            }
        }

        Ok(())
    }
}
