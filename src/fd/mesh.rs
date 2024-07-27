#![allow(dead_code)]

use reborrow::ReborrowMut;
use std::{array::from_fn, ops::Range};

use crate::fd::{
    node_from_vertex, Boundary, BoundaryKind, Condition, Conditions, Engine, FdEngine, FdIntEngine,
    NodeSpace, Operator, Projection,
};
use crate::geometry::{
    faces, AxisMask, Face, FaceMask, IndexSpace, Rectangle, Tree, TreeBlocks, TreeInterfaces,
    TreeNodes,
};
use crate::system::{SystemLabel, SystemSlice, SystemSliceMut};

use super::boundary::BlockBC;
use super::SystemBC;

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
            ghost: self.nodes.ghost_nodes,
            context: (),
        }
    }

    // pub fn block_fields<'a, Label: SystemLabel>(
    //     &self,
    //     block: usize,
    //     system: &'a SystemSlice<Label>,
    // ) -> SystemFields<'a, Label> {
    //     let nodes = self.block_nodes(block);
    //     system.slice(nodes).fields()
    // }

    // pub fn block_fields_mut<'a, Label: SystemLabel>(
    //     &self,
    //     block: usize,
    //     system: &'a mut SystemSliceMut<Label>,
    // ) -> SystemFieldsMut<'a, Label> {
    //     let nodes = self.block_nodes(block);
    //     system.slice_mut(nodes).fields_mut()
    // }

    // /// Retrieves the neighbors of the given cell for each region.
    // pub fn cell_neighbors(&self, cell: usize) -> &[usize] {
    //     self.neighbors.cell_neighbors(cell)
    // }

    // /// Retrieves the neighbors of the given cell for each region.
    // pub fn cell_neighbor(&self, cell: usize, region: Region<N>) -> usize {
    //     self.neighbors.cell_neighbor(cell, region)
    // }

    // pub fn cell_neighbor_after_refinement(
    //     &self,
    //     cell: usize,
    //     split: AxisMask<N>,
    //     region: Region<N>,
    // ) -> usize {
    //     let neighbor = self.neighbors.cell_neighbor(cell, region);

    //     if neighbor == NULL || self.cell_level(neighbor) <= self.cell_level(cell) {
    //         return neighbor;
    //     }

    //     let mut target = [Side::Middle; N];
    //     for axis in 0..N {
    //         if region.side(axis) == Side::Middle && split.is_set(axis) {
    //             target[axis] = Side::Right;
    //         }
    //     }

    //     self.cell_neighbor(neighbor, Region::new(target))
    // }

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

    pub fn order<const ORDER: usize>(&mut self) -> MeshOrder<N, ORDER> {
        const {
            assert!(ORDER % 2 == 0);
        }
        MeshOrder(self)
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

    pub(crate) fn cell_node_origin(&self, index: [usize; N]) -> [isize; N] {
        from_fn(|axis| (index[axis] * self.nodes.cell_width[axis]) as isize)
    }
}

// *****************************
// Node Routines ***************
// *****************************

pub struct MeshOrder<'a, const N: usize, const ORDER: usize>(&'a mut Mesh<N>);

impl<'a, const N: usize, const ORDER: usize> MeshOrder<'a, N, ORDER> {
    /// Enforces strong boundary conditions. This includes strong physical boundary conditions, as well
    /// as handling interior boundaries (same level, coarse-fine, or fine-coarse).
    pub fn fill_boundary<BC: Boundary<N> + Conditions<N>>(
        mut self,
        bc: BC,
        mut system: SystemSliceMut<'_, BC::System>,
    ) {
        self.fill_physical(&bc, &mut system);
        self.fill_direct(&mut system);
        self.fill_fine(&mut system);
        self.fill_prolong(&bc, &mut system);
    }

    fn fill_physical<BC: Boundary<N> + Conditions<N>>(
        &mut self,
        bc: &BC,
        system: &mut SystemSliceMut<'_, BC::System>,
    ) {
        let mesh = self.0.rb_mut();

        for block in 0..mesh.num_blocks() {
            // Fill Physical Boundary conditions
            let nodes = mesh.block_nodes(block);
            let space = mesh.block_space(block);
            let boundary = mesh.block_boundary(block, bc.clone());

            // Take slice of system and
            let mut block_fields = system.slice_mut(nodes.clone()).fields_mut();

            for field in BC::System::fields() {
                space
                    .set_context(SystemBC::new(field.clone(), boundary.clone()))
                    .fill_boundary(block_fields.field_mut(field));
            }
        }
    }

    fn fill_direct<System: SystemLabel>(&mut self, system: &mut SystemSliceMut<'_, System>) {
        let mesh = self.0.rb_mut();

        // Fill direct interfaces
        for interface in mesh.interfaces.direct() {
            let block_space = mesh.block_space(interface.block);
            let block_nodes = mesh.block_nodes(interface.block);
            let neighbor_space = mesh.block_space(interface.neighbor);
            let neighbor_nodes = mesh.block_nodes(interface.neighbor);

            for node in IndexSpace::new(interface.size).iter() {
                let block_node = from_fn(|axis| node[axis] as isize + interface.dest[axis]);
                let neighbor_node = from_fn(|axis| node[axis] as isize + interface.source[axis]);

                let block_index = block_space.index_from_node(block_node);
                let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                for field in System::fields() {
                    let value =
                        system.slice(neighbor_nodes.clone()).field(field.clone())[neighbor_index];
                    system
                        .slice_mut(block_nodes.clone())
                        .field_mut(field.clone())[block_index] = value;
                }
            }
        }
    }

    fn fill_fine<System: SystemLabel>(&mut self, system: &mut SystemSliceMut<'_, System>) {
        let mesh = self.0.rb_mut();

        for interface in mesh.interfaces.fine() {
            let block_space = mesh.block_space(interface.block);
            let block_nodes = mesh.block_nodes(interface.block);
            let neighbor_space = mesh.block_space(interface.neighbor);
            let neighbor_nodes = mesh.block_nodes(interface.neighbor);

            for node in IndexSpace::new(interface.size).iter() {
                let block_node = from_fn(|axis| node[axis] as isize + interface.dest[axis]);
                let neighbor_node =
                    from_fn(|axis| 2 * (node[axis] as isize + interface.source[axis]));

                let block_index = block_space.index_from_node(block_node);
                let neighbor_index = neighbor_space.index_from_node(neighbor_node);

                for field in System::fields() {
                    let value =
                        system.slice(neighbor_nodes.clone()).field(field.clone())[neighbor_index];
                    system
                        .slice_mut(block_nodes.clone())
                        .field_mut(field.clone())[block_index] = value;
                }
            }
        }
    }

    fn fill_prolong<BC: Boundary<N> + Conditions<N>>(
        &mut self,
        bc: &BC,
        system: &mut SystemSliceMut<'_, BC::System>,
    ) {
        let mesh = self.0.rb_mut();

        for interface in mesh.interfaces.coarse() {
            let block_nodes = mesh.block_nodes(interface.block);
            let block_space = mesh.block_space(interface.block);

            let neighbor_nodes = mesh.block_nodes(interface.neighbor);
            let neighbor_boundary = mesh.block_boundary(interface.neighbor, bc.clone());
            let neighbor_space = mesh.block_space(interface.neighbor);

            for node in IndexSpace::new(interface.size).iter() {
                let block_node = from_fn(|axis| node[axis] as isize + interface.dest[axis]);

                let neighbor_vertex =
                    from_fn(|axis| (node[axis] as isize + interface.source[axis]) as usize);

                let block_index = block_space.index_from_node(block_node);

                for field in BC::System::fields() {
                    let bc = SystemBC::new(field.clone(), neighbor_boundary.clone());

                    let value = neighbor_space.set_context(bc).prolong::<ORDER>(
                        neighbor_vertex,
                        system.slice(neighbor_nodes.clone()).field(field.clone()),
                    );

                    system
                        .slice_mut(block_nodes.clone())
                        .field_mut(field.clone())[block_index] = value;
                }
            }
        }
    }

    pub fn weak_boundary<BC: Boundary<N> + Conditions<N>>(
        self,
        bc: BC,
        field: SystemSlice<'_, BC::System>,
        mut deriv: SystemSliceMut<'_, BC::System>,
    ) {
        let mesh = self.0;

        for block in 0..mesh.num_blocks() {
            let boundary = mesh.block_boundary(block, bc.clone());
            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block).set_context(bounds);
            let nodes = mesh.block_nodes(block);
            let vertex_size = space.vertex_size();

            let block_fields = field.slice(nodes.clone()).fields();
            let mut block_deriv_fields = deriv.slice_mut(nodes.clone()).fields_mut();

            for face in faces::<N>() {
                if boundary.kind(face) != BoundaryKind::Radiative {
                    continue;
                }

                // Sommerfeld radiative boundary conditions.
                for vertex in IndexSpace::new(vertex_size).face(face).iter() {
                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        vertex,
                        boundary: boundary.clone(),
                    };
                    let position: [f64; N] = engine.position();
                    let r = position.iter().map(|&v| v * v).sum::<f64>().sqrt();

                    let index = space.index_from_vertex(vertex);

                    for label in BC::System::fields() {
                        let f = block_fields.field(label.clone());
                        let dfdt = block_deriv_fields.field_mut(label.clone());

                        let field_boundary = SystemBC::new(label.clone(), bc.clone());

                        let target = field_boundary.radiative(position);
                        let gradient = engine.gradient(field_boundary.clone(), f);
                        let mut advection = engine.value(f) - target;

                        for axis in 0..N {
                            advection += position[axis] * gradient[axis];
                        }

                        dfdt[index] = -advection / r;
                    }
                }
            }
        }
    }

    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn project<BC: Boundary<N>, P: Projection<N>>(
        self,
        boundary: BC,
        projection: P,
        source: SystemSlice<'_, P::Input>,
        mut dest: SystemSliceMut<'_, P::Output>,
    ) {
        let mesh = self.0;

        for block in 0..mesh.num_blocks() {
            let nodes = mesh.block_nodes(block);

            let input = source.slice(nodes.clone()).fields();
            let mut output = dest.slice_mut(nodes.clone()).fields_mut();

            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block).set_context(bounds);
            let vertex_size = space.vertex_size();

            let boundary = mesh.block_boundary(block, boundary.clone());

            for vertex in IndexSpace::new(vertex_size).iter() {
                let is_interior = Self::is_interior(&boundary, vertex_size, vertex);

                let result = if is_interior {
                    let engine = FdIntEngine::<N, ORDER> {
                        space: space.clone(),
                        vertex,
                    };

                    projection.project(&engine, input.as_fields())
                } else {
                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        vertex,
                        boundary: boundary.clone(),
                    };

                    projection.project(&engine, input.as_fields())
                };

                let index = space.index_from_vertex(vertex);

                for field in P::Output::fields() {
                    output.field_mut(field.clone())[index] = result.field(field.clone());
                }
            }
        }
    }

    /// Applies the given operator to `source`, storing the result in `dest`, and utilizing `context` to store
    /// extra fields.
    pub fn apply<BC: Boundary<N>, O: Operator<N>>(
        self,
        boundary: BC,
        operator: O,
        source: SystemSlice<'_, O::System>,
        context: SystemSlice<'_, O::Context>,
        mut dest: SystemSliceMut<'_, O::System>,
    ) {
        let mesh = self.0;

        for block in 0..mesh.num_blocks() {
            let nodes = mesh.block_nodes(block);

            let input = source.slice(nodes.clone()).fields();
            let context = context.slice(nodes.clone()).fields();
            let mut output = dest.slice_mut(nodes.clone());

            let boundary = mesh.block_boundary(block, boundary.clone());
            let bounds = mesh.block_bounds(block);
            let space = mesh.block_space(block).set_context(bounds);
            let vertex_size = space.vertex_size();

            for vertex in IndexSpace::new(vertex_size).iter() {
                let is_interior = Self::is_interior(&boundary, vertex_size, vertex);

                let result = if is_interior {
                    let engine = FdIntEngine::<N, ORDER> {
                        space: space.clone(),
                        vertex,
                    };

                    operator.evaluate(&engine, input.as_fields(), context.as_fields())
                } else {
                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        vertex,
                        boundary: boundary.clone(),
                    };

                    operator.evaluate(&engine, input.as_fields(), context.as_fields())
                };

                let index = space.index_from_node(node_from_vertex(vertex));

                for field in O::System::fields() {
                    output.field_mut(field.clone())[index] = result.field(field.clone());
                }
            }
        }
    }

    /// Determines if a vertex is not within `ORDER` of any weakly enforced boundary.
    fn is_interior(
        boundary: &impl Boundary<N>,
        vertex_size: [usize; N],
        vertex: [usize; N],
    ) -> bool {
        let mut result = true;

        for axis in 0..N {
            result &= boundary.kind(Face::negative(axis)).has_ghost() || vertex[axis] >= ORDER / 2;
            result &= boundary.kind(Face::positive(axis)).has_ghost()
                || vertex[axis] < vertex_size[axis] - ORDER / 2;
        }

        result

        // false
    }
}

impl<const N: usize> Mesh<N> {
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

        for block in 0..self.num_blocks() {
            let bounds = self.block_bounds(block);
            let space = self.block_space(block).set_context(bounds);
            let vertex_size = space.vertex_size();

            let data = &src[self.block_nodes(block)];

            let mut block_result = 0.0;

            for vertex in IndexSpace::new(vertex_size).iter() {
                let index = space.index_from_vertex(vertex);

                let mut value = data[index] * data[index];

                for axis in 0..N {
                    if vertex[axis] == 0 || vertex[axis] == vertex_size[axis] - 1 {
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
}
