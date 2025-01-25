use std::array;

use crate::kernel::Element;
use aeon_geometry::IndexSpace;

use crate::{
    fd::Mesh,
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
}

#[cfg(test)]
mod tests {
    use crate::{fd::Mesh, kernel::BoundaryKind};
    use aeon_geometry::{Face, Rectangle};

    #[test]
    fn element_windows() {
        let mut mesh: Mesh<2> = Mesh::new(Rectangle::UNIT, 4, 2);
        mesh.set_face_boundary(Face::negative(0), BoundaryKind::Parity);
        mesh.set_face_boundary(Face::negative(1), BoundaryKind::Parity);
        mesh.set_face_boundary(Face::positive(0), BoundaryKind::Free);
        mesh.set_face_boundary(Face::positive(1), BoundaryKind::Free);

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
