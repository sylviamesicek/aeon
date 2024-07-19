#![allow(dead_code)]

use reborrow::ReborrowMut;
use std::{array::from_fn, ops::Range};

use crate::fd::{
    node_from_vertex, Boundary, BoundaryKind, Condition, Conditions, Engine, FdEngine, FdIntEngine,
    NodeSpace, Operator, Projection,
};
use crate::geometry::{
    faces, regions, AxisMask, Face, FaceMask, IndexSpace, Rectangle, Region, Side, Tree,
    TreeBlocks, TreeNeighbors, TreeNodes, NULL,
};
use crate::system::{SystemFields, SystemFieldsMut, SystemLabel, SystemSlice, SystemSliceMut};

/// Implementation of an axis aligned tree mesh using standard finite difference operators.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Mesh<const N: usize> {
    tree: Tree<N>,
    blocks: TreeBlocks<N>,
    neighbors: TreeNeighbors<N>,
    nodes: TreeNodes<N>,
}

impl<const N: usize> Mesh<N> {
    /// Constructs a new tree mesh, covering the given physical domain. Each cell has the given number of subdivisions
    /// per axis, and each block extends out an extra `ghost_nodes` distance to facilitate inter-cell communication.
    pub fn new(bounds: Rectangle<N>, cell_width: [usize; N], ghost_nodes: usize) -> Self {
        let mut result = Self {
            tree: Tree::new(bounds),
            blocks: TreeBlocks::default(),
            neighbors: TreeNeighbors::default(),
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
        self.neighbors.build(&self.tree);
        self.nodes.build(&self.blocks);
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
    pub fn block_boundary<B: Boundary>(&self, block: usize, boundary: B) -> BlockBoundary<N, B> {
        BlockBoundary {
            mask: self.block_boundary_flags(block),
            inner: boundary,
        }
    }

    /// Computes the nodespace corresponding to a block.
    pub fn block_space(&self, block: usize) -> NodeSpace<N> {
        let size = self.blocks.block_size(block);
        let cell_size = from_fn(|axis| size[axis] * self.nodes.cell_width[axis]);

        NodeSpace {
            size: cell_size,
            ghost: self.nodes.ghost_nodes,
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

    /// Retrieves the neighbors of the given cell for each region.
    pub fn cell_neighbors(&self, cell: usize) -> &[usize] {
        self.neighbors.cell_neighbors(cell)
    }

    /// Retrieves the neighbors of the given cell for each region.
    pub fn cell_neighbor(&self, cell: usize, region: Region<N>) -> usize {
        self.neighbors.cell_neighbor(cell, region)
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

// ******************************
// Blocks ***********************
// ******************************

#[derive(Debug, Clone)]
pub struct BlockBoundary<const N: usize, B> {
    inner: B,
    mask: FaceMask<N>,
}

impl<const N: usize, B: Boundary> Boundary for BlockBoundary<N, B> {
    fn kind(&self, face: Face) -> BoundaryKind {
        if self.mask.is_set(face) {
            self.inner.kind(face)
        } else {
            BoundaryKind::Custom
        }
    }
}

// *****************************
// Node Routines ***************
// *****************************

pub struct MeshOrder<'a, const N: usize, const ORDER: usize>(&'a mut Mesh<N>);

impl<'a, const N: usize, const ORDER: usize> MeshOrder<'a, N, ORDER> {
    /// Enforces strong boundary conditions. This includes strong physical boundary conditions, as well
    /// as handling interior boundaries (same level, coarse-fine, or fine-coarse).
    pub fn fill_boundary<B: Boundary, C: Conditions<N>>(
        mut self,
        boundary: &B,
        conditions: &C,
        mut system: SystemSliceMut<'_, C::System>,
    ) {
        self.fill_physical(boundary, conditions, &mut system);
        self.fill_direct(&mut system);
        self.fill_prolong(boundary, &mut system);
    }

    fn fill_physical<B: Boundary, C: Conditions<N>>(
        &mut self,
        boundary: &B,
        conditions: &C,
        system: &mut SystemSliceMut<'_, C::System>,
    ) {
        let mesh = self.0.rb_mut();

        for block in 0..mesh.num_blocks() {
            // Fill Physical Boundary conditions
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            let boundary = mesh.block_boundary(block, boundary.clone());

            // println!("Filling Strong Boundary Conditions for {block}");

            // Take slice of system and
            let mut block_fields = system.slice_mut(nodes.clone()).fields_mut();

            for field in C::System::fields() {
                let condition = conditions.field(field.clone());
                space.fill_boundary(&boundary, &condition, block_fields.field_mut(field));
            }
        }
    }

    fn fill_direct<System: SystemLabel>(&mut self, system: &mut SystemSliceMut<'_, System>) {
        let mesh = self.0.rb_mut();

        for block in 0..mesh.num_blocks() {
            // Fill Physical Boundary conditions
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            // Fill Injection boundary conditions
            let size = mesh.block_size(block);
            let cells = mesh.block_cells(block);
            let cell_space = IndexSpace::new(size);

            for region in regions::<N>() {
                let offset_dir = region.offset_dir();

                let mut region_size: [_; N] = from_fn(|axis| mesh.nodes.cell_width[axis]);
                let mut coarse_region_size: [_; N] =
                    from_fn(|axis| mesh.nodes.cell_width[axis] / 2);

                for axis in 0..N {
                    if region.side(axis) == Side::Right {
                        region_size[axis] += 1;
                        coarse_region_size[axis] += 1;
                    }
                }

                for cell_index in region.face_vertices(size) {
                    let cell = cells[cell_space.linear_from_cartesian(cell_index)];
                    let cell_origin: [isize; N] = mesh.cell_node_origin(cell_index);

                    let neighbor = mesh.cell_neighbor(cell, region.clone());

                    // If physical boundary we skip
                    if neighbor == NULL {
                        continue;
                    }

                    // If neighbor is coarser we skip
                    if mesh.cell_level(neighbor) < mesh.cell_level(cell) {
                        continue;
                    }

                    // If neighbor is more refined we use injection
                    if mesh.cell_level(neighbor) > mesh.cell_level(cell) {
                        for mask in region.adjacent_splits() {
                            let mut nmask = mask;

                            for axis in 0..N {
                                if region.side(axis) != Side::Middle {
                                    nmask.toggle(axis);
                                }
                            }

                            let neighbor = neighbor + nmask.to_linear();
                            let neighbor_index = mesh.blocks.cell_index(neighbor);
                            let neighbor_block = mesh.cell_block(neighbor);
                            let neighbor_origin = mesh.cell_node_origin(neighbor_index);

                            let neighbor_nodes = mesh.block_nodes(neighbor_block);

                            let neighbor_offset: [isize; N] = from_fn(|axis| {
                                neighbor_origin[axis]
                                    - offset_dir[axis] * mesh.nodes.cell_width[axis] as isize
                            });

                            let mut origin = cell_origin;

                            for axis in 0..N {
                                if mask.is_set(axis) {
                                    origin[axis] += (mesh.nodes.cell_width[axis] / 2) as isize;
                                }
                            }

                            for node in region
                                .nodes(mesh.nodes.ghost_nodes, coarse_region_size)
                                .chain(
                                    region
                                        .face_vertices(coarse_region_size)
                                        .map(node_from_vertex),
                                )
                            {
                                let source = from_fn(|axis| neighbor_offset[axis] + 2 * node[axis]);
                                let dest = from_fn(|axis| origin[axis] + node[axis]);

                                for field in System::fields() {
                                    let v = space.value(
                                        source,
                                        &system.field(field.clone())[neighbor_nodes.clone()],
                                    );
                                    space.set_value(
                                        dest,
                                        v,
                                        &mut system.field_mut(field)[nodes.clone()],
                                    )
                                }
                            }
                        }

                        continue;
                    }

                    // Store various information about neighbor
                    let neighbor_index = mesh.blocks.cell_index(neighbor);
                    let neighbor_block = mesh.blocks.cell_block(neighbor);
                    let neighbor_origin: [isize; N] = mesh.cell_node_origin(neighbor_index);

                    let neighbor_nodes = mesh.block_nodes(neighbor_block);

                    let neighbor_offset: [isize; N] = from_fn(|axis| {
                        neighbor_origin[axis]
                            - offset_dir[axis] * mesh.nodes.cell_width[axis] as isize
                    });

                    for node in region
                        .nodes(mesh.nodes.ghost_nodes, region_size)
                        .chain(region.face_vertices(region_size).map(node_from_vertex))
                    {
                        let source = from_fn(|axis| neighbor_offset[axis] + node[axis]);
                        let dest = from_fn(|axis| cell_origin[axis] + node[axis]);

                        for field in System::fields() {
                            let v = space.value(
                                source,
                                &system.field(field.clone())[neighbor_nodes.clone()],
                            );
                            space.set_value(dest, v, &mut system.field_mut(field)[nodes.clone()])
                        }
                    }
                }
            }
        }
    }

    fn fill_prolong<B: Boundary, System: SystemLabel>(
        &mut self,
        boundary: &B,
        system: &mut SystemSliceMut<'_, System>,
    ) {
        let mesh = self.0.rb_mut();

        for block in 0..mesh.num_blocks() {
            // Cache node space
            let space = mesh.block_space(block);
            let domain = mesh.block_boundary(block, boundary.clone());
            let nodes = mesh.block_nodes(block);
            // Fill Injection boundary conditions
            let size = mesh.block_size(block);
            let cells = mesh.block_cells(block);
            let cell_space = IndexSpace::new(size);

            for region in regions::<N>() {
                let offset_dir = region.offset_dir();

                let mut region_size: [_; N] = from_fn(|axis| mesh.nodes.cell_width[axis]);

                for axis in 0..N {
                    if region.side(axis) == Side::Right {
                        region_size[axis] += 1;
                    }
                }

                for cell_index in region.face_vertices(size) {
                    let cell = cells[cell_space.linear_from_cartesian(cell_index)];
                    let cell_origin = mesh.cell_node_origin(cell_index);
                    let cell_split = mesh.cell_split(cell);

                    let neighbor = mesh.cell_neighbor(cell, region.clone());

                    // If physical boundary we skip
                    if neighbor == NULL {
                        continue;
                    }

                    // We only consider this neighbor if it is coarser
                    if mesh.cell_level(neighbor) >= mesh.cell_level(cell) {
                        continue;
                    }

                    let neighbor_index = mesh.blocks.cell_index(neighbor);
                    let neighbor_block = mesh.blocks.cell_block(neighbor);
                    let neighbor_nodes = mesh.block_nodes(neighbor_block);

                    let mut neighbor_split = cell_split;
                    for axis in 0..N {
                        if region.side(axis) != Side::Middle {
                            neighbor_split.toggle(axis);
                        }
                    }

                    let mut neighbor_origin: [isize; N] = mesh.cell_node_origin(neighbor_index);
                    for axis in 0..N {
                        if neighbor_split.is_set(axis) {
                            neighbor_origin[axis] += mesh.nodes.cell_width[axis] as isize / 2;
                        }
                    }

                    let neighbor_offset: [isize; N] = from_fn(|axis| {
                        neighbor_origin[axis]
                            - offset_dir[axis] * mesh.nodes.cell_width[axis] as isize / 2
                    });

                    for node in region.nodes(mesh.nodes.ghost_nodes, region_size) {
                        let source =
                            from_fn(|axis| (2 * neighbor_offset[axis] + node[axis]) as usize);
                        let dest = from_fn(|axis| cell_origin[axis] + node[axis]);

                        for field in System::fields() {
                            let v = space.prolong::<ORDER>(
                                &domain,
                                source,
                                &system.field(field.clone())[neighbor_nodes.clone()],
                            );
                            space.set_value(
                                dest,
                                v,
                                &mut system.field_mut(field.clone())[nodes.clone()],
                            );
                        }
                    }
                }
            }
        }
    }

    pub fn weak_boundary<B: Boundary, C: Conditions<N>>(
        self,
        boundary: &B,
        conditions: &C,
        field: SystemSlice<'_, C::System>,
        mut deriv: SystemSliceMut<'_, C::System>,
    ) {
        let mesh = self.0;

        for block in 0..mesh.num_blocks() {
            let boundary = mesh.block_boundary(block, boundary.clone());
            let space = mesh.block_space(block);
            let bounds = mesh.block_bounds(block);
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
                        bounds: bounds.clone(),
                        boundary: boundary.clone(),
                        vertex,
                    };
                    let index = engine.index();
                    let position: [f64; N] = engine.position();
                    let r = position.iter().map(|&v| v * v).sum::<f64>().sqrt();

                    for label in C::System::fields() {
                        let f = block_fields.field(label.clone());
                        let dfdt = block_deriv_fields.field_mut(label.clone());

                        let target = conditions.field(label.clone()).radiative(position);
                        let gradient = engine.gradient(f);
                        let mut advection = f[index] - target;

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
    pub fn project<P: Projection<N>>(
        self,
        boundary: &impl Boundary,
        projection: &P,
        source: SystemSlice<'_, P::Input>,
        mut dest: SystemSliceMut<'_, P::Output>,
    ) {
        let mesh = self.0;

        for block in 0..mesh.num_blocks() {
            let nodes = mesh.block_nodes(block);

            let input = source.slice(nodes.clone()).fields();
            let mut output = dest.slice_mut(nodes.clone()).fields_mut();

            let space = mesh.block_space(block);
            let boundary = mesh.block_boundary(block, boundary.clone());
            let bounds = mesh.block_bounds(block);
            let vertex_size = space.vertex_size();

            for vertex in IndexSpace::new(vertex_size).iter() {
                let is_interior = Self::is_interior(&boundary, vertex_size, vertex);

                let result = if is_interior {
                    let engine = FdIntEngine::<N, ORDER> {
                        space: space.clone(),
                        bounds: bounds.clone(),
                        vertex,
                    };

                    projection.project(&engine, input.as_fields())
                } else {
                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        bounds: bounds.clone(),
                        boundary: boundary.clone(),
                        vertex,
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
    pub fn apply<O: Operator<N>>(
        self,
        boundary: &impl Boundary,
        operator: &O,
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

            let space = mesh.block_space(block);
            let boundary = mesh.block_boundary(block, boundary.clone());
            let bounds = mesh.block_bounds(block);
            let vertex_size = space.vertex_size();

            for vertex in IndexSpace::new(vertex_size).iter() {
                let is_interior = Self::is_interior(&boundary, vertex_size, vertex);

                let result = if is_interior {
                    let engine = FdIntEngine::<N, ORDER> {
                        space: space.clone(),
                        bounds: bounds.clone(),
                        vertex,
                    };

                    operator.evaluate(&engine, input.as_fields(), context.as_fields())
                } else {
                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        bounds: bounds.clone(),
                        boundary: boundary.clone(),
                        vertex,
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
    fn is_interior(boundary: &impl Boundary, vertex_size: [usize; N], vertex: [usize; N]) -> bool {
        // let mut result = true;

        // for axis in 0..N {
        //     result &= !(boundary.kind(Face::negative(axis)).is_weak() && vertex[axis] < ORDER / 2);
        //     result &= !(boundary.kind(Face::positive(axis)).is_weak()
        //         && vertex[axis] >= vertex_size[axis] - ORDER / 2);
        // }

        // result

        false
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
            let space = self.block_space(block);
            let bounds = self.block_bounds(block);
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

            for spacing in space.spacing(bounds) {
                block_result *= spacing;
            }

            result += block_result;
        }

        result.sqrt()
    }
}
