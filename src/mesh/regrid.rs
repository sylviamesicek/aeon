use std::array;

use rand::Rng;
use reborrow::ReborrowMut;

use crate::geometry::{Face, HyperBox, IndexSpace, LeafId, Region, Side};
use crate::image::{ImageMut, ImageRef};
use crate::kernel::NodeSpace;
use crate::{mesh::Mesh, shared::SharedSlice};

impl<const N: usize> Mesh<N> {
    /// Retrieves a vector representing all regridding flags for the mesh.
    pub fn refine_flags(&self) -> &[bool] {
        self.refine_flags.as_slice()
    }
    /// Retrievs a vector representing all coarsening flags for the mesh.
    pub fn coarsen_flags(&self) -> &[bool] {
        self.coarsen_flags.as_slice()
    }

    /// Refines the mesh using currently set flags.
    pub fn regrid(&mut self) {
        self.regrid_map.clear();

        // Save old information
        self.old_blocks.clone_from(&self.blocks);

        self.old_cell_splits.clear();
        self.old_cell_splits.extend(
            self.tree
                .leaves()
                .flat_map(|leaf| self.tree.most_recent_leaf_split(leaf)),
        );

        // Perform regriding
        self.regrid_map.resize(self.tree.num_leaves(), LeafId(0));

        let mut coarsen_map = vec![LeafId(0); self.tree.num_leaves()];

        let check = self.tree.check_coarsen_flags(&self.coarsen_flags);
        if !check {
            println!("{:?}", self.coarsen_flags);
            assert!(false);
        }

        debug_assert!(self.tree.check_coarsen_flags(&self.coarsen_flags));
        self.tree
            .coarsen_leaf_index_map(&self.coarsen_flags, &mut coarsen_map);
        self.tree.coarsen(&self.coarsen_flags);

        let mut refine_map = vec![LeafId(0); self.tree.num_leaves()];
        let mut flags = vec![false; self.tree.num_leaves()];

        for (old, &new) in coarsen_map.iter().enumerate() {
            flags[new.0] = self.refine_flags[old];
        }

        debug_assert!(self.tree.check_refine_flags(&flags));
        self.tree.refine_leaf_index_map(&flags, &mut refine_map);
        self.tree.refine(&flags);

        for i in 0..self.regrid_map.len() {
            self.regrid_map[i] = refine_map[coarsen_map[i].0];
        }

        // Rebuild mesh
        self.build();
    }

    /// Flags every cell for refinement, then regrids.
    pub fn refine_global(&mut self) {
        self.refine_flags.fill(true);
        self.coarsen_flags.fill(false);
        self.regrid();
    }

    /// Flags every cell for coarsening, then regrids.
    pub fn coarsen_global(&mut self) {
        self.coarsen_flags.fill(true);
        self.refine_flags.fill(false);
        self.balance_flags();
        self.regrid();
    }

    /// Refines innermost cell
    pub fn refine_innermost(&mut self) {
        self.refine_flags.fill(false);
        // Find the innermost cell and flag it for refinement
        let mut cell_inner_index = 0;
        let mut cell_inner_center = f64::MAX;
        for cell in self.tree().leaves() {
            let cell_bounds = self.tree().leaf_bounds(cell);
            let cell_center = cell_bounds.center()[0]; // get 1st element because we only have 1 dimension
            if cell_center < cell_inner_center {
                cell_inner_center = cell_center;
                cell_inner_index = cell.0;
            }
        }
        // Set flag and refine
        self.refine_flags[cell_inner_index] = true;
        self.balance_flags();
        self.regrid();
    }

    /// Coarsens innermost cell
    pub fn coarsen_innermost(&mut self) {
        self.coarsen_flags.fill(false);
        // Find the innermost cell and flag it for coarsening
        let mut cell_inner_index = 0;
        let mut cell_inner_center = f64::MAX;
        for cell in self.tree().leaves() {
            let cell_bounds = self.tree().leaf_bounds(cell);
            let cell_center = cell_bounds.center()[0]; // get 1st element because we only have 1 dimension
            if cell_center < cell_inner_center {
                cell_inner_center = cell_center;
                cell_inner_index = cell.0;
            }
        }
        // Set flag and refine
        self.coarsen_flags[cell_inner_index] = true;
        self.balance_flags();
        self.regrid();
    }

    /// Refines or coarsens cells one level (towards target level fgl) within a given radius
    pub fn regrid_in_radius(&mut self, radius: f64, fgl: usize) {
        // Loop through the active cells and flag any that need to be refined or coarsened
        self.refine_flags.fill(false);
        self.coarsen_flags.fill(false);
        let mut temp_rflags = vec![false; self.tree().num_leaves()];
        let mut temp_cflags = vec![false; self.tree().num_leaves()];
        for cell in self.tree().leaves() {
            // Get some information about the cell
            let cell_bounds = self.tree().leaf_bounds(cell);
            let cell_center = cell_bounds.center()[0]; // get 1st element because we only have 1 dimension
            let cell_level = self.tree().leaf_level(cell);
            let cell_index = cell.0;
            // Set flags if necessary
            if cell_center < radius {
                if cell_level < fgl {
                    temp_rflags[cell_index] = true;
                }
                if cell_level > fgl {
                    temp_cflags[cell_index] = true;
                }
            }
        }
        // Update the mesh's flags
        self.refine_flags = temp_rflags;
        self.coarsen_flags = temp_cflags;
        // Perform the regridding
        self.balance_flags();
        self.regrid();
    }

    /// Flags cells for refinement using a wavelet criterion. The system must have filled
    /// boundaries. This function tags any cell that is insufficiently refined to approximate
    /// operators of the given `order` within the range of error.
    pub fn flag_wavelets(&mut self, order: usize, lower: f64, upper: f64, data: ImageRef) {
        let buffer = 2 * (self.ghost / 2);
        let support = (self.width + 2 * buffer) / 2;

        assert!(order % 2 == 0);
        assert!(order <= support);
        // Example w = 6, g = 3, o = 6
        // -> o > (6 + 4) / 2 because the ghost offset is odd

        assert_eq!(data.num_nodes(), self.num_nodes());

        let element = self.request_element(support, order);
        let element_coarse = self.request_element(self.width / 2, order / 2);

        let support = element.support_refined();

        let mut rflags_buf = std::mem::take(&mut self.refine_flags);
        let mut cflags_buf = std::mem::take(&mut self.coarsen_flags);

        // Allows interior mutability.
        let rflags = SharedSlice::new(&mut rflags_buf);
        let cflags = SharedSlice::new(&mut cflags_buf);

        self.block_compute(|mesh, store, block| {
            let imsrc = store.scratch(support);
            let imdest = store.scratch(support);

            let nodes = mesh.block_nodes(block);
            let space = mesh.block_space(block);

            let block_system = data.slice(nodes.clone());

            for &cell in mesh.blocks.leaves(block) {
                let is_cell_on_boundary = mesh.cell_needs_coarse_element(cell);

                // Window of nodes on element.
                let window = if is_cell_on_boundary {
                    mesh.element_coarse_window(cell)
                } else {
                    mesh.element_window(cell)
                };

                let mut should_refine = false;
                let mut should_coarsen = true;

                for field in data.channels() {
                    // Unpack data to element
                    for (i, node) in window.iter().enumerate() {
                        imsrc[i] = block_system.channel(field)[space.index_from_node(node)];
                    }

                    if is_cell_on_boundary {
                        let src = &imsrc[..element_coarse.support_refined()];
                        let dst = &mut imdest[..element_coarse.support_refined()];

                        element_coarse.wavelet(src, dst);

                        for point in element_coarse.diagonal_points() {
                            should_refine = should_refine || dst[point].abs() >= upper;
                            should_coarsen = should_coarsen && dst[point].abs() <= lower;
                        }
                    } else {
                        element.wavelet(imsrc, imdest);

                        for point in element.diagonal_int_points(buffer) {
                            should_refine = should_refine || imdest[point].abs() >= upper;
                            should_coarsen = should_coarsen && imdest[point].abs() <= lower;
                        }
                    }

                    unsafe {
                        if should_refine {
                            *rflags.get_mut(cell.0) = true;
                        }
                    }

                    unsafe {
                        *cflags.get_mut(cell.0) = should_coarsen;
                    }
                }
            }
        });

        let _ = std::mem::replace(&mut self.refine_flags, rflags_buf);
        let _ = std::mem::replace(&mut self.coarsen_flags, cflags_buf);

        self.replace_element(element);
        self.replace_element(element_coarse);
    }

    pub fn flag_wavelets2_impl<const ORDER: usize>(
        &mut self,
        lower: f64,
        upper: f64,
        data: ImageRef,
    ) {
        let mut rflags_buf = std::mem::take(&mut self.refine_flags);
        let mut cflags_buf = std::mem::take(&mut self.coarsen_flags);

        // Allows interior mutability.
        let rflags = SharedSlice::new(&mut rflags_buf);
        let cflags = SharedSlice::new(&mut cflags_buf);

        self.block_compute(|mesh, store, block| {
            let block_leaves = mesh.blocks().leaves(block);
            let block_size_in_leaves = mesh.blocks().size(block);
            let block_space_in_leaves = IndexSpace::new(block_size_in_leaves);
            let block_boundary_flags = mesh.block_physical_boundary_flags(block);
            let block_level = mesh.blocks().level(block);
            let block_space = mesh.block_space(block);
            let block_data = data.slice(mesh.block_nodes(block));

            let block_coarse_space = NodeSpace {
                size: block_size_in_leaves.map(|size| (size + 2) * mesh.width() / 2),
                ghost: mesh.width() / 2,
                bounds: HyperBox::UNIT,
                boundary: mesh.block_boundary_classes(block),
            };
            let block_num_coarse_nodes = block_coarse_space.num_nodes();

            let block_coarse_buffer = store.scratch(block_num_coarse_nodes * data.num_channels());
            let mut block_coarse = ImageMut::from_storage(block_coarse_buffer, data.num_channels());

            let cell_data_buffer = store.scratch((mesh.width() + 1).pow(N as u32));
            let mut cell_data = ImageMut::from_storage(cell_data_buffer, data.num_channels());

            let cell_space = IndexSpace::new([mesh.width() + 1; N]);

            let mut coarse_window = block_coarse_space.inner_window();

            for axis in 0..N {
                coarse_window.size[axis] -=
                    (!block_boundary_flags.is_set(Face::positive(axis))) as usize;
            }

            for coarse in coarse_window {
                let block_node_index = block_space.index_from_node(coarse.map(|i| 2 * i));
                let coarse_index = block_coarse_space.index_from_node(coarse);

                for channel in data.channels() {
                    block_coarse.channel_mut(channel)[coarse_index] =
                        block_data.channel(channel)[block_node_index];
                }
            }

            for region in Region::<N>::enumerate() {
                if region == Region::CENTRAL {
                    continue;
                }

                // Skip if on physical boundary
                for face in region.adjacent_faces() {
                    if block_boundary_flags.is_set(face) {
                        continue;
                    }
                }

                for adjacent_position in block_space_in_leaves.region_adjacent_window(region) {
                    let adjacent_leaf = block_leaves
                        [block_space_in_leaves.linear_from_cartesian(adjacent_position)];
                    let adjacent_cell = mesh.tree().cell_from_leaf(adjacent_leaf);
                    let neighbor_cell = mesh.tree().neighbor_region(adjacent_cell, region).unwrap();
                    let neighbor_level = mesh.tree().level(neighbor_cell);

                    let mut coarse_node_origin =
                        adjacent_position.map(|position| position * mesh.width() / 2);

                    for axis in 0..N {
                        match region.side(axis) {
                            Side::Left => coarse_node_origin[axis] -= mesh.width() / 2,
                            Side::Right => coarse_node_origin[axis] += mesh.width() / 2,
                            Side::Middle => {}
                        }
                    }

                    debug_assert!(neighbor_level <= block_level);

                    let edge = std::array::from_fn(|axis| region.side(axis) == Side::Right);

                    let coarse_space = IndexSpace::new([mesh.width() / 2 + 1; N]);
                    let mut coarse_window = coarse_space.window();
                    for axis in 0..N {
                        coarse_window.size[axis] -= (!edge[axis]) as usize;
                    }

                    if neighbor_level == block_level {
                        mesh.load_nodes_for_cell(neighbor_cell, data, cell_data.rb_mut(), edge);

                        for offset in coarse_window {
                            let cell_node = cell_space.linear_from_cartesian(offset.map(|o| 2 * o));
                            let coarse_node =
                                block_coarse_space.index_from_vertex(array::from_fn(|axis| {
                                    coarse_node_origin[axis] + offset[axis]
                                }));

                            for channel in data.channels() {
                                block_coarse.channel_mut(channel)[coarse_node] =
                                    cell_data.channel(channel)[cell_node];
                            }
                        }
                    } else {
                        let neighbor_leaf = mesh.tree().leaf_from_cell(neighbor_cell).unwrap();
                        let neighbor_block = mesh.blocks().block_from_leaf(neighbor_leaf);
                        let neighbor_space = mesh.block_space(neighbor_block);
                        let neighbor_block_node_offset = mesh.block_nodes(neighbor_block).start;

                        let mut neighbor_split =
                            mesh.tree().most_recent_leaf_split(adjacent_leaf).unwrap();
                        for axis in 0..N {
                            if region.side(axis) != Side::Middle {
                                neighbor_split.toggle(axis);
                            }
                        }

                        let mut neighbor_node_origin = mesh.leaf_node_origin(neighbor_leaf);
                        for axis in 0..N {
                            neighbor_node_origin[axis] +=
                                neighbor_split.is_set(axis) as usize * mesh.width / 2;
                        }
                        // We now have the split of the neighbor that we want

                        for offset in coarse_window {
                            let neighbor_node = neighbor_block_node_offset
                                + neighbor_space.index_from_vertex(array::from_fn(|axis| {
                                    neighbor_node_origin[axis] + offset[axis]
                                }));
                            let coarse_node =
                                block_coarse_space.index_from_vertex(array::from_fn(|axis| {
                                    coarse_node_origin[axis] + offset[axis]
                                }));

                            for channel in data.channels() {
                                block_coarse.channel_mut(channel)[coarse_node] =
                                    data.channel(channel)[neighbor_node];
                            }
                        }
                    }
                }
            }
        });

        let _ = std::mem::replace(&mut self.refine_flags, rflags_buf);
        let _ = std::mem::replace(&mut self.coarsen_flags, cflags_buf);
    }

    /// Store flags for each cell in a debug buffer.
    pub fn flags_debug(&mut self, debug: &mut [i64]) {
        assert!(debug.len() == self.num_nodes());

        let debug = SharedSlice::new(debug);

        self.block_compute(|mesh, _, block| {
            let block_nodes = mesh.block_nodes(block);
            let block_space = mesh.block_space(block);
            let block_size = mesh.blocks.size(block);
            let cells = mesh.blocks.leaves(block);

            for (i, position) in IndexSpace::new(block_size).iter().enumerate() {
                let cell = cells[i];
                let origin: [_; N] = array::from_fn(|axis| (position[axis] * mesh.width) as isize);

                for offset in IndexSpace::new([mesh.width + 1; N]).iter() {
                    let node = array::from_fn(|axis| origin[axis] + offset[axis] as isize);

                    let idx = block_nodes.start + block_space.index_from_node(node);

                    unsafe {
                        *debug.get_mut(idx) = mesh.refine_flags[cell.0] as i64;
                    }
                }
            }
        });
    }

    /// Mark cells for refinement at random.
    pub fn set_refine_flags_random<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        rng.fill(self.refine_flags.as_mut_slice());
    }

    /// Manually marks a cell for refinement.
    pub fn set_refine_flag(&mut self, cell: usize) {
        self.refine_flags[cell] = true
    }

    /// Manually marks a cell for coarsening.
    pub fn set_coarsen_flag(&mut self, cell: usize) {
        self.coarsen_flags[cell] = true
    }

    // Mark `count` cells around each currently tagged cell for refinement.
    pub fn buffer_refine_flags(&mut self, count: usize) {
        for _ in 0..count {
            for cell in self.tree.leaves() {
                if !self.refine_flags[cell.0] {
                    continue;
                }

                for neighbor in self.tree.leaf_neighborhood(cell) {
                    self.refine_flags[neighbor.0] = true;
                }
            }
        }
    }

    /// Limits coarsening to cells with a `level > min_level`, and refinement to
    /// cells with a `level < max_level`.
    pub fn limit_level_range_flags(&mut self, min_level: usize, max_level: usize) {
        assert!(self.refine_flags.len() == self.num_leaves());
        assert!(self.coarsen_flags.len() == self.num_leaves());

        for cell in self.tree.leaves() {
            let level = self.tree.leaf_level(cell);
            if level >= max_level {
                self.refine_flags[cell.0] = false;
            }

            if level <= min_level {
                self.coarsen_flags[cell.0] = false;
            }
        }
    }

    /// After cells have been tagged, the refinement/coarsening flags
    /// must be balanced to ensure that the 2:1 balance across faces and vertices
    /// is still maintained.
    pub fn balance_flags(&mut self) {
        // Propogate refinement flags outwards.
        self.tree.balance_refine_flags(&mut self.refine_flags);
        // Refinement has priority over coarsening. Ensure that there is never a cell marked
        // for refinement next to a equal or coarser cell marked for coarsening.
        for cell in self.tree.leaves() {
            if self.refine_flags[cell.0] {
                let level = self.tree.leaf_level(cell);
                for neighbor in self.tree.leaf_neighborhood(cell) {
                    let nlevel = self.tree.leaf_level(neighbor);
                    if nlevel <= level {
                        self.coarsen_flags[neighbor.0] = false;
                    }
                }
            }
        }
        // Unmark coarsening flags as necessary.
        self.tree.balance_coarsen_flags(&mut self.coarsen_flags);
    }

    /// Returns true if the mesh requires regridding (i.e. any cells are tagged for either refinement
    /// or coarsening).
    pub fn requires_regridding(&self) -> bool {
        self.refine_flags.iter().any(|&b| b) || self.coarsen_flags.iter().any(|&b| b)
    }

    /// The number of cell that are marked for refinement.
    pub fn num_refine_cells(&self) -> usize {
        self.refine_flags.iter().filter(|&&b| b).count()
    }

    /// The number of cells that are marked for coarsening.
    pub fn num_coarsen_cells(&self) -> usize {
        self.coarsen_flags.iter().filter(|&&b| b).count()
    }
}

#[cfg(test)]
mod tests {
    use crate::geometry::{HyperBox, LeafId};
    use crate::kernel::{BoundaryClass, BoundaryClasses};
    use crate::mesh::Mesh;

    #[test]
    fn element_windows() {
        let mut mesh = Mesh::new(
            HyperBox::UNIT,
            4,
            2,
            BoundaryClasses::from_sides([BoundaryClass::Ghost; 2], [BoundaryClass::OneSided; 2]),
        );
        mesh.refine_global();

        mesh.set_refine_flag(0);
        mesh.regrid();

        assert!(!mesh.cell_needs_coarse_element(LeafId(0)));
        assert!(!mesh.cell_needs_coarse_element(LeafId(1)));
        assert!(!mesh.cell_needs_coarse_element(LeafId(2)));
        assert!(!mesh.cell_needs_coarse_element(LeafId(3)));
        assert!(mesh.cell_needs_coarse_element(LeafId(4)));
        assert!(mesh.cell_needs_coarse_element(LeafId(5)));
        assert!(mesh.cell_needs_coarse_element(LeafId(6)));
    }

    #[test]
    fn global_coarsening_and_refinement() {
        let mut mesh = Mesh::<2>::new(HyperBox::UNIT, 4, 2, BoundaryClasses::GHOST);
        mesh.coarsen_global();
        mesh.refine_global();
        mesh.coarsen_global();
        mesh.refine_global();
        mesh.refine_global();
        mesh.coarsen_global();
    }
}
