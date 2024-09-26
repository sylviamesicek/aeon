use aeon_basis::{Boundary, Element};

use crate::{
    fd::Mesh,
    shared::SharedSlice,
    system::{SystemLabel, SystemSlice},
};

impl<const N: usize> Mesh<N> {
    pub fn wavelet<S: SystemLabel, B: Boundary<N> + Sync>(
        &mut self,
        boundary: B,
        tolerance: f64,
        field: SystemSlice<S>,
        flags: &mut [bool],
    ) {
        let element = Element::<N>::uniform(self.width);
        let element_coarse = Element::<N>::uniform(self.width / 2);
        let support = element.support_refined();

        let field = field.as_range();

        // Allows interior mutability.
        let flags = SharedSlice::new(flags);

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
                            if dst[point].abs() > tolerance {
                                unsafe {
                                    *flags.get_mut(cell) = true;
                                }
                                break 'fields;
                            }
                        }
                    } else {
                        element.wavelet(&imsrc, &mut imdest);

                        for point in element.diagonal_int_points() {
                            if imdest[point].abs() > tolerance {
                                unsafe {
                                    *flags.get_mut(cell) = true;
                                }
                                break 'fields;
                            }
                        }
                    }
                }
            }
        });
    }
}

// #[repr(transparent)]
// pub struct SyncUnsafeCell<T: ?Sized> {
//     value: UnsafeCell<T>,
// }

// unsafe impl<T: ?Sized + Sync> Sync for SyncUnsafeCell<T> {}

// // Identical interface as UnsafeCell:
// impl<T: ?Sized> SyncUnsafeCell<T> {
//     pub const fn get(&self) -> *mut T {
//         self.value.get()
//     }
// }
