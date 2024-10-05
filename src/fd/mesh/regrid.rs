use std::array;

use aeon_basis::{Boundary, Element};
use aeon_geometry::IndexSpace;

use crate::{
    fd::Mesh,
    shared::SharedSlice,
    system::{SystemLabel, SystemSlice},
};

impl<const N: usize> Mesh<N> {
    /// Flags cells for refinement using a wavelet criterion. The system must have filled
    /// boundaries.
    pub fn flag_wavelets<S: SystemLabel, B: Boundary<N> + Sync>(
        &mut self,
        lower: f64,
        upper: f64,
        boundary: B,
        field: SystemSlice<S>,
    ) {
        let element = Element::<N>::uniform(self.width);
        let element_coarse = Element::<N>::uniform(self.width / 2);
        let support = element.support_refined();

        let field = field.as_range();

        let mut rflags_buf = std::mem::take(&mut self.refine_flags);
        let mut cflags_buf = std::mem::take(&mut self.coarsen_flags);

        // Allows interior mutability.
        let rflags = SharedSlice::new(&mut rflags_buf);
        let cflags = SharedSlice::new(&mut cflags_buf);

        self.block_compute(|mesh, store, block| {
            let boundary = mesh.block_boundary(block, boundary.clone());

            let imsrc = store.scratch(support);
            let mut imdest = store.scratch(support);

            let nodes = mesh.block_nodes(block);
            let space = mesh.block_space(block);

            let system = unsafe { field.slice(nodes.clone()).fields() };

            for &cell in mesh.blocks.cells(block) {
                let is_cell_on_boundary = mesh.is_cell_on_boundary(cell, boundary.clone());

                // Window of nodes on element.
                let window = if is_cell_on_boundary {
                    mesh.element_coarse_window(cell)
                } else {
                    mesh.element_window(cell)
                };

                'fields: for field in S::fields() {
                    // Unpack data to element
                    for (i, node) in window.iter().enumerate() {
                        imsrc[i] = system.field(field.clone())[space.index_from_node(node)];
                    }

                    if is_cell_on_boundary {
                        let src = &imsrc[..element_coarse.support_refined()];
                        let dst = &mut imdest[..element_coarse.support_refined()];

                        element_coarse.wavelet(src, dst);

                        for point in element_coarse.diagonal_points() {
                            if dst[point].abs() >= upper {
                                unsafe {
                                    *rflags.get_mut(cell) = true;
                                }
                                break 'fields;
                            }

                            if dst[point].abs() <= lower {
                                unsafe {
                                    *cflags.get_mut(cell) = true;
                                }
                                break 'fields;
                            }
                        }
                    } else {
                        element.wavelet(&imsrc, &mut imdest);

                        for point in element.diagonal_int_points() {
                            if imdest[point].abs() >= upper {
                                unsafe {
                                    *rflags.get_mut(cell) = true;
                                }
                                break 'fields;
                            }

                            if imdest[point].abs() <= lower {
                                unsafe {
                                    *cflags.get_mut(cell) = true;
                                }
                                break 'fields;
                            }
                        }
                    }
                }
            }
        });

        let _ = std::mem::replace(&mut self.refine_flags, rflags_buf);
        let _ = std::mem::replace(&mut self.coarsen_flags, cflags_buf);
    }

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
    use crate::fd::Mesh;
    use aeon_basis::{Boundary, BoundaryKind};
    use aeon_geometry::{Face, Rectangle};

    #[derive(Debug, Clone, Copy)]
    pub struct Quadrant;

    impl Boundary<2> for Quadrant {
        fn kind(&self, face: Face<2>) -> BoundaryKind {
            if face.side {
                BoundaryKind::Free
            } else {
                BoundaryKind::Parity
            }
        }
    }

    #[test]
    fn element_windows() {
        let mut mesh = Mesh::new(Rectangle::UNIT, 4, 2);
        mesh.set_refine_flag(0);
        mesh.regrid();

        assert_eq!(mesh.is_cell_on_boundary(0, Quadrant), false);
        assert_eq!(mesh.is_cell_on_boundary(1, Quadrant), false);
        assert_eq!(mesh.is_cell_on_boundary(2, Quadrant), false);
        assert_eq!(mesh.is_cell_on_boundary(3, Quadrant), false);
        assert_eq!(mesh.is_cell_on_boundary(4, Quadrant), true);
        assert_eq!(mesh.is_cell_on_boundary(5, Quadrant), true);
        assert_eq!(mesh.is_cell_on_boundary(6, Quadrant), true);
    }
}
