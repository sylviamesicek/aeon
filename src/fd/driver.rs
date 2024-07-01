use crate::{
    fd::{node_from_vertex, Boundary, Mesh},
    geometry::{regions, IndexSpace, Side, SPATIAL_BOUNDARY},
    prelude::SystemSlice,
    system::{SystemLabel, SystemSliceMut},
};
use std::array::from_fn;
use std::marker::PhantomData;

use super::Projection;

pub struct Driver<const N: usize> {
    _marker: PhantomData<[usize; N]>,
}

impl<const N: usize> Driver<N> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    /// Fills ghost nodes on the mesh. This includes applying physical boundary conditions, as well
    /// as handling interior boundaries (same level, coarse-fine, or fine-coarse).
    pub fn fill_boundary<const ORDER: usize, B: Boundary>(
        &mut self,
        mesh: &Mesh<N>,
        physical: &B,
        mut system: SystemSliceMut<'_, B::System>,
    ) {
        self.fill_physical(mesh, physical, &mut system);
        self.fill_direct(mesh, &mut system);
        self.fill_prolong::<ORDER, _>(mesh, physical, &mut system);
    }

    fn fill_physical<B: Boundary>(
        &self,
        mesh: &Mesh<N>,
        physical: &B,
        system: &mut SystemSliceMut<'_, B::System>,
    ) {
        for block in 0..mesh.num_blocks() {
            // Fill Physical Boundary conditions
            let space = mesh.block_space(block);
            let nodes = mesh.block_nodes(block);

            for field in B::System::fields() {
                let physical = physical.boundary(field.clone());
                let boundary = mesh.block_boundary(block).with_boundary(physical);
                let data = &mut system.field_mut(field)[nodes.clone()];
                space.with_boundary(boundary).fill_boundary(data);
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
                    if neighbor == SPATIAL_BOUNDARY {
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

    fn fill_prolong<const ORDER: usize, B: Boundary>(
        &mut self,
        mesh: &Mesh<N>,
        physical: &B,
        system: &mut SystemSliceMut<'_, B::System>,
    ) {
        for block in 0..mesh.num_blocks() {
            // Cache node space
            let space = mesh.block_space(block);
            let boundary = mesh.block_boundary(block);
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
                    if neighbor == SPATIAL_BOUNDARY {
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

                        for field in B::System::fields() {
                            let space = space.with_boundary(
                                boundary.with_boundary(physical.boundary(field.clone())),
                            );

                            let v = space.prolong::<ORDER>(
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

    pub fn project<const ORDER: usize, B, P>(
        &self,
        mesh: &Mesh<N>,
        projection: &P,
        physical: &B,
        source: SystemSlice<'_, P::Input>,
        mut dest: SystemSliceMut<'_, P::Output>,
    ) where
        P: Projection<N>,
        B: Boundary<System = P::Input>,
    {
        for block in 0..mesh.num_blocks() {
            let nodes = mesh.block_nodes(block);

            let input = source.slice(nodes.clone());
            let output = dest.slice_mut(nodes.clone());

            let space = mesh.block_space(block);
            let boundary = mesh.block_boundary(block);

            for vertex in IndexSpace::new(space.vertex_size()).iter() {}
        }
    }
}
