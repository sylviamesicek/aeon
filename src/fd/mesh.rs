#![allow(dead_code)]

use std::{array::from_fn, fmt::Write, ops::Range};

use crate::fd::{BlockBC, Boundary, NodeSpace};
use crate::geometry::{
    AxisMask, BlockInterface, FaceMask, Rectangle, Tree, TreeBlocks, TreeInterfaces, TreeNodes,
};

/// Implementation of an axis aligned tree mesh using standard finite difference operators.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mesh<const N: usize> {
    tree: Tree<N>,
    blocks: TreeBlocks<N>,
    nodes: TreeNodes<N>,
    interfaces: TreeInterfaces<N>,
}

impl<const N: usize> Mesh<N> {
    /// Constructs a new tree mesh, covering the given physical domain. Each cell has the given number of subdivisions
    /// per axis, and each block extends out an extra `ghost_nodes` distance to facilitate inter-cell communication.
    pub fn new(bounds: Rectangle<N>, cell_width: [usize; N], ghost_nodes: usize) -> Self {
        let mut result = Self {
            tree: Tree::new(bounds),
            blocks: TreeBlocks::default(),
            interfaces: TreeInterfaces::default(),
            nodes: TreeNodes::new(cell_width, ghost_nodes),
        };

        result.build();

        result
    }

    /// Checks if the given refinement flags are 2:1 balanced.
    pub fn is_balanced(&self, flags: &[bool]) -> bool {
        self.tree.check_refine_flags(flags)
    }

    /// Balances the given refinement flags.
    pub fn balance(&self, flags: &mut [bool]) {
        self.tree.balance_refine_flags(flags)
    }

    pub fn refine(&mut self, flags: &[bool]) {
        self.tree.refine(flags);
        self.build();
    }

    /// Reconstructs interal structure of the TreeMesh, automatically called during refinement.
    pub fn build(&mut self) {
        self.blocks.build(&self.tree);
        self.nodes.build(&self.blocks);
        self.interfaces.build(&self.tree, &self.blocks, &self.nodes);
    }

    /// Number of cells in the mesh.
    pub fn num_cells(&self) -> usize {
        self.tree.num_cells()
    }

    /// Number of blocks in the block.
    pub fn num_blocks(&self) -> usize {
        self.blocks.num_blocks()
    }

    // Number of nodes in mesh.
    pub fn num_nodes(&self) -> usize {
        self.nodes.num_nodes()
    }

    /// Size of a given block, measured in cells.
    pub fn block_size(&self, block: usize) -> [usize; N] {
        self.blocks.block_size(block)
    }

    /// Map from cartesian indices within block to cell at the given positions
    pub fn block_cells(&self, block: usize) -> &[usize] {
        self.blocks.block_cells(block)
    }

    /// Computes the physical bounds of a block.
    pub fn block_bounds(&self, block: usize) -> Rectangle<N> {
        self.blocks.block_bounds(block)
    }

    /// The range of nodes that the block owns.
    pub fn block_nodes(&self, block: usize) -> Range<usize> {
        self.nodes.block_nodes(block)
    }

    /// Computes flags indicating whether a particular face of a block borders a physical
    /// boundary.
    pub fn block_boundary_flags(&self, block: usize) -> FaceMask<N> {
        self.blocks.block_boundary_flags(block)
    }

    /// Produces a block boundary which correctly accounts for
    /// interior interfaces.
    pub fn block_boundary<B: Boundary<N>>(&self, block: usize, boundary: B) -> BlockBC<N, B> {
        BlockBC {
            flags: self.block_boundary_flags(block),
            inner: boundary,
        }
    }

    /// Computes the nodespace corresponding to a block.
    pub fn block_space(&self, block: usize) -> NodeSpace<N, ()> {
        let size = self.blocks.block_size(block);
        let cell_size = from_fn(|axis| size[axis] * self.nodes.cell_width[axis]);

        NodeSpace {
            size: cell_size,
            ghost: self.nodes.ghost,
            context: (),
        }
    }

    pub fn fine_interfaces(&self) -> impl Iterator<Item = &BlockInterface<N>> + '_ {
        self.interfaces.fine()
    }

    pub fn direct_interfaces(&self) -> impl Iterator<Item = &BlockInterface<N>> + '_ {
        self.interfaces.direct()
    }

    pub fn coarse_interfaces(&self) -> impl Iterator<Item = &BlockInterface<N>> + '_ {
        self.interfaces.coarse()
    }

    pub fn write_debug(&self) -> String {
        let mut result = String::new();

        writeln!(result, "// **********************").unwrap();
        writeln!(result, "// Cells ****************").unwrap();
        writeln!(result, "// **********************").unwrap();
        writeln!(result, "").unwrap();

        for cell in 0..self.num_cells() {
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

        for block in 0..self.num_blocks() {
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

        result
    }

    /// The level of the mesh the cell resides on.
    pub fn cell_level(&self, cell: usize) -> usize {
        self.tree.level(cell)
    }

    /// Most recent subdivision that the cell underwent.
    pub fn cell_split(&self, cell: usize) -> AxisMask<N> {
        self.tree.split(cell)
    }

    /// Retrieves which block the cell belongs to.
    pub fn cell_block(&self, cell: usize) -> usize {
        self.blocks.cell_block(cell)
    }

    pub fn max_level(&self) -> usize {
        let mut level = 1;

        for block in 0..self.num_blocks() {
            let cell = self.block_cells(block)[0];
            level = level.max(self.cell_level(cell))
        }

        level
    }

    pub fn min_spacing(&self) -> f64 {
        let max_level = self.max_level();
        let domain = self.tree.domain();

        from_fn::<_, N, _>(|axis| {
            domain.size[axis]
                / (self.nodes.cell_width[axis] + 1) as f64
                / 2_f64.powi(max_level as i32)
        })
        .iter()
        .min_by(|a, b| f64::total_cmp(a, b))
        .cloned()
        .unwrap_or(1.0)
    }
}
