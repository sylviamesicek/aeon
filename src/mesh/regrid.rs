use std::array;

use crate::geometry::IndexSpace;
use crate::kernel::Element;

use crate::{
    mesh::Mesh,
    shared::SharedSlice,
    system::{System, SystemSlice},
};

impl<const N: usize> Mesh<N> {
    /// Flags cells for refinement using a wavelet criterion. The system must have filled
    /// boundaries. This function tags any cell that is insufficiently refined to approximate
    /// operators of the given `order` within the range of error.
    pub fn flag_wavelets<S: System + Sync>(
        &mut self,
        order: usize,
        lower: f64,
        upper: f64,
        result: SystemSlice<S>,
    ) {
        assert!(order % 2 == 0);
        assert!(order <= self.width);

        let element = Element::<N>::uniform(self.width, order);
        let element_coarse = Element::<N>::uniform(self.width / 2, order / 2);
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

            let block_system = result.slice(nodes.clone());

            for &cell in mesh.blocks.cells(block) {
                let is_cell_on_boundary = mesh.is_cell_on_boundary(cell);

                // Window of nodes on element.
                let window = if is_cell_on_boundary {
                    mesh.element_coarse_window(cell)
                } else {
                    mesh.element_window(cell)
                };

                let mut should_refine = false;
                let mut should_coarsen = true;

                for field in result.system().enumerate() {
                    // Unpack data to element
                    for (i, node) in window.iter().enumerate() {
                        imsrc[i] = block_system.field(field.clone())[space.index_from_node(node)];
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

                        for point in element.diagonal_int_points() {
                            should_refine = should_refine || imdest[point].abs() >= upper;
                            should_coarsen = should_coarsen && imdest[point].abs() <= lower;
                        }
                    }

                    unsafe {
                        if should_refine {
                            *rflags.get_mut(cell) = true;
                        }
                    }

                    unsafe {
                        *cflags.get_mut(cell) = should_coarsen;
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
            let cells = mesh.blocks.cells(block);

            for (i, position) in IndexSpace::new(block_size).iter().enumerate() {
                let cell = cells[i];
                let origin: [_; N] = array::from_fn(|axis| (position[axis] * mesh.width) as isize);

                for offset in IndexSpace::new([mesh.width + 1; N]).iter() {
                    let node = array::from_fn(|axis| origin[axis] + offset[axis] as isize);

                    let idx = block_nodes.start + block_space.index_from_node(node);

                    unsafe {
                        *debug.get_mut(idx) = mesh.refine_flags[cell] as i64;
                    }
                }
            }
        });
    }

    /// Manually marks a cell for refinement.
    pub fn set_refine_flag(&mut self, cell: usize) {
        self.refine_flags[cell] = true
    }

    /// Manually marks a cell for coarsening.
    pub fn set_coarsen_flag(&mut self, cell: usize) {
        self.coarsen_flags[cell] = true
    }

    /// Mark `count` cells around each currently tagged cell for refinement.
    // pub fn buffer_refine_flags(&mut self, count: usize) {
    //     for _ in 0..count {
    //         for cell in 0..self.num_cells() {
    //             if !self.refine_flags[cell] {
    //                 continue;
    //             }

    //             for neighbor in self.tree.neighborhood(cell) {
    //                 self.refine_flags[neighbor] = true;
    //             }
    //         }
    //     }
    // }

    /// Limits coarsening to cells with a `level > min_level`, and refinement to
    /// cells with a `level < max_level`.
    pub fn limit_level_range_flags(&mut self, min_level: usize, max_level: usize) {
        for cell in 0..self.num_cells() {
            let level = self.tree.level(cell);
            if level >= max_level {
                self.refine_flags[cell] = false;
            }

            if level <= min_level {
                self.coarsen_flags[cell] = false;
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
    use crate::geometry::{Face, Rectangle};
    use crate::mesh::Mesh;

    #[test]
    fn element_windows() {
        let mut mesh: Mesh<2> = Mesh::new(Rectangle::UNIT, 4, 2);
        mesh.set_boundary_ghost(Face::negative(0), true);
        mesh.set_boundary_ghost(Face::negative(1), true);
        mesh.set_boundary_ghost(Face::positive(0), false);
        mesh.set_boundary_ghost(Face::positive(1), false);

        mesh.set_refine_flag(0);
        mesh.regrid();

        assert!(!mesh.is_cell_on_boundary(0));
        assert!(!mesh.is_cell_on_boundary(1));
        assert!(!mesh.is_cell_on_boundary(2));
        assert!(!mesh.is_cell_on_boundary(3));
        assert!(mesh.is_cell_on_boundary(4));
        assert!(mesh.is_cell_on_boundary(5));
        assert!(mesh.is_cell_on_boundary(6));
    }
}
