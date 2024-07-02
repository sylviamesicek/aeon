use crate::{
    fd::{node_from_vertex, Boundary, Mesh},
    geometry::{regions, IndexSpace, Side, NULL},
    prelude::{Face, SystemSlice},
    system::{SystemLabel, SystemSliceMut},
};
use std::array::from_fn;
use std::marker::PhantomData;

use super::{boundary::Conditions, engine::FdIntEngine, FdEngine, Operator, Projection};

/// Caches various data structured used to launch jobs (such as threadpools, memory pools, GPU interface, etc.).
pub struct Driver<const N: usize> {
    _marker: PhantomData<[usize; N]>,
}

impl<const N: usize> Driver<N> {
    /// Constructs a default and empty driver.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Enforces strong boundary conditions. This includes strong physical boundary conditions, as well
    /// as handling interior boundaries (same level, coarse-fine, or fine-coarse).
    pub fn fill_boundary<const ORDER: usize, B: Boundary, C: Conditions<N>>(
        &mut self,
        mesh: &Mesh<N>,
        boundary: &B,
        conditions: &C,
        mut system: SystemSliceMut<'_, C::System>,
    ) {
        self.fill_physical(mesh, boundary, conditions, &mut system);
        self.fill_direct(mesh, &mut system);
        self.fill_prolong::<ORDER, _, _>(mesh, boundary, &mut system);
    }

    fn fill_physical<B: Boundary, C: Conditions<N>>(
        &self,
        mesh: &Mesh<N>,
        boundary: &B,
        conditions: &C,
        system: &mut SystemSliceMut<'_, C::System>,
    ) {
        for block in 0..mesh.num_blocks() {
            // Fill Physical Boundary conditions
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            let boundary = mesh.block_boundary(block, boundary.clone());

            for field in C::System::fields() {
                let condition = conditions.field(field.clone());
                let data = &mut system.field_mut(field)[nodes.clone()];
                space.fill_boundary(&boundary, &condition, data);
            }
        }
    }

    fn fill_direct<System: SystemLabel>(
        &self,
        mesh: &Mesh<N>,
        system: &mut SystemSliceMut<'_, System>,
    ) {
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

                let mut region_size: [_; N] = from_fn(|axis| mesh.cell_width[axis]);
                let mut coarse_region_size: [_; N] = from_fn(|axis| mesh.cell_width[axis] / 2);

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
                                    - offset_dir[axis] * mesh.cell_width[axis] as isize
                            });

                            let mut origin = cell_origin;

                            for axis in 0..N {
                                if mask.is_set(axis) {
                                    origin[axis] += (mesh.cell_width[axis] / 2) as isize;
                                }
                            }

                            for node in region.nodes(mesh.ghost_nodes, coarse_region_size).chain(
                                region
                                    .face_vertices(coarse_region_size)
                                    .map(node_from_vertex),
                            ) {
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

                    let neighbor_nodes = mesh.blocks.nodes(neighbor_block);

                    let neighbor_offset: [isize; N] = from_fn(|axis| {
                        neighbor_origin[axis] - offset_dir[axis] * mesh.cell_width[axis] as isize
                    });

                    for node in region
                        .nodes(mesh.ghost_nodes, region_size)
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

    fn fill_prolong<const ORDER: usize, B: Boundary, System: SystemLabel>(
        &mut self,
        mesh: &Mesh<N>,
        boundary: &B,
        system: &mut SystemSliceMut<'_, System>,
    ) {
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

                let mut region_size: [_; N] = from_fn(|axis| mesh.cell_width[axis]);

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
                    let neighbor_nodes = mesh.blocks.nodes(neighbor_block);

                    let mut neighbor_split = cell_split;
                    for axis in 0..N {
                        if region.side(axis) != Side::Middle {
                            neighbor_split.toggle(axis);
                        }
                    }

                    let mut neighbor_origin: [isize; N] = mesh.cell_node_origin(neighbor_index);
                    for axis in 0..N {
                        if neighbor_split.is_set(axis) {
                            neighbor_origin[axis] += mesh.cell_width[axis] as isize / 2;
                        }
                    }

                    let neighbor_offset: [isize; N] = from_fn(|axis| {
                        neighbor_origin[axis]
                            - offset_dir[axis] * mesh.cell_width[axis] as isize / 2
                    });

                    for node in region.nodes(mesh.ghost_nodes, region_size) {
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

    /// Applies the projection to `source`, and stores the result in `dest`.
    pub fn project<const ORDER: usize, P: Projection<N>>(
        &self,
        mesh: &Mesh<N>,
        boundary: &impl Boundary,
        projection: &P,
        source: SystemSlice<'_, P::Input>,
        mut dest: SystemSliceMut<'_, P::Output>,
    ) {
        for block in 0..mesh.num_blocks() {
            let nodes = mesh.block_nodes(block);

            let input = source.slice(nodes.clone());
            let mut output = dest.slice_mut(nodes.clone());

            let space = mesh.block_space(block);
            let boundary = mesh.block_boundary(block, boundary.clone());
            let bounds = mesh.block_bounds(block);
            let vertex_size = space.vertex_size();

            for vertex in IndexSpace::new(vertex_size).iter() {
                let is_interior = Self::is_interior::<ORDER>(&boundary, vertex_size, vertex);

                let result = if is_interior {
                    let engine = FdIntEngine::<N, ORDER> {
                        space: space.clone(),
                        bounds: bounds.clone(),
                        vertex,
                    };

                    projection.project(&engine, input.clone())
                } else {
                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        bounds: bounds.clone(),
                        boundary: boundary.clone(),
                        vertex,
                    };

                    projection.project(&engine, input.clone())
                };

                let index = space.index_from_node(node_from_vertex(vertex));

                for field in P::Output::fields() {
                    output.field_mut(field.clone())[index] = result.field(field.clone());
                }
            }
        }
    }

    /// Applies the given operator to `source`, storing the result in `dest`, and utilizing `context` to store
    /// extra fields.
    pub fn apply<const ORDER: usize, O: Operator<N>>(
        &self,
        mesh: &Mesh<N>,
        boundary: &impl Boundary,
        operator: &O,
        source: SystemSlice<'_, O::System>,
        context: SystemSlice<'_, O::Context>,
        mut dest: SystemSliceMut<'_, O::System>,
    ) {
        for block in 0..mesh.num_blocks() {
            let nodes = mesh.block_nodes(block);

            let input = source.slice(nodes.clone());
            let context = context.slice(nodes.clone());
            let mut output = dest.slice_mut(nodes.clone());

            let space = mesh.block_space(block);
            let boundary = mesh.block_boundary(block, boundary.clone());
            let bounds = mesh.block_bounds(block);
            let vertex_size = space.vertex_size();

            for vertex in IndexSpace::new(vertex_size).iter() {
                let is_interior = Self::is_interior::<ORDER>(&boundary, vertex_size, vertex);

                let result = if is_interior {
                    let engine = FdIntEngine::<N, ORDER> {
                        space: space.clone(),
                        bounds: bounds.clone(),
                        vertex,
                    };

                    operator.evaluate(&engine, input.clone(), context.clone())
                } else {
                    let engine = FdEngine::<N, ORDER, _> {
                        space: space.clone(),
                        bounds: bounds.clone(),
                        boundary: boundary.clone(),
                        vertex,
                    };

                    operator.evaluate(&engine, input.clone(), context.clone())
                };

                let index = space.index_from_node(node_from_vertex(vertex));

                for field in O::System::fields() {
                    output.field_mut(field.clone())[index] = result.field(field.clone());
                }
            }
        }
    }

    /// Computes the maximum l2 norm of all fields in the system.
    pub fn norm<S: SystemLabel>(&mut self, mesh: &Mesh<N>, source: SystemSlice<'_, S>) -> f64 {
        S::fields()
            .into_iter()
            .map(|label| self.norm_scalar(mesh, source.field(label)))
            .max_by(|a, b| a.total_cmp(b))
            .unwrap()
    }

    fn norm_scalar(&mut self, mesh: &Mesh<N>, src: &[f64]) -> f64 {
        let mut result = 0.0;

        for block in 0..mesh.num_blocks() {
            let space = mesh.block_space(block);
            let bounds = mesh.block_bounds(block);
            let vertex_size = space.vertex_size();

            let data = &src[mesh.block_nodes(block)];

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

    /// Determines if a vertex is not within `ORDER` of any weakly enforced boundary.
    fn is_interior<const ORDER: usize>(
        boundary: &impl Boundary,
        vertex_size: [usize; N],
        vertex: [usize; N],
    ) -> bool {
        let mut result = true;

        for axis in 0..N {
            result &= !(boundary.kind(Face::negative(axis)).is_weak() && vertex[axis] < ORDER);
            result &= !(boundary.kind(Face::positive(axis)).is_weak()
                && vertex[axis] >= vertex_size[axis] - ORDER);
        }

        result
    }
}
