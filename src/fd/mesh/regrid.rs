use aeon_basis::{Boundary, Element};
use std::cell::UnsafeCell;

use crate::{
    fd::Mesh,
    system::{SystemLabel, SystemSlice},
};

impl<const N: usize> Mesh<N> {
    // pub fn regrid<S: SystemLabel>(
    //     &self,
    //     index_map: &mut [usize],
    //     dest: &Mesh<N>,
    //     ssource: SystemSlice<S>,
    //     mut sdest: SystemSliceMut<S>,
    // ) {
    //     let ssource = ssource.as_range();
    //     let sdest = sdest.as_range();

    //     (0..self.blocks.len()).par_bridge().for_each(|block| {
    //         let nodes = self.block_nodes(block);
    //         let space = self.block_space(block);

    //         for &cell in self.blocks.cells(block) {
    //             let position = self.blocks.cell_position(cell);

    //             let size = [self.width + 1; N];
    //             let mut origin = [0; N];

    //             for axis in 0..N {
    //                 origin[axis] += (self.width * position[axis]) as isize
    //             }

    //             let window = NodeWindow { size, origin };
    //         }
    //     })
    // }

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
        let flags = unsafe { &*(flags as *mut [bool] as *const [SyncUnsafeCell<bool>]) };

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
                        element_coarse.wavelet(&imsrc, &mut imdest);

                        for point in element_coarse.diagonal_points() {
                            if imdest[point].abs() > tolerance {
                                unsafe {
                                    *flags[cell].get() = true;
                                }
                                break 'fields;
                            }
                        }
                    } else {
                        element.wavelet(&imsrc, &mut imdest);

                        for point in element.diagonal_int_points() {
                            if imdest[point].abs() > tolerance {
                                unsafe {
                                    *flags[cell].get() = true;
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

#[repr(transparent)]
pub struct SyncUnsafeCell<T: ?Sized> {
    value: UnsafeCell<T>,
}

unsafe impl<T: ?Sized + Sync> Sync for SyncUnsafeCell<T> {}

// Identical interface as UnsafeCell:
impl<T: ?Sized> SyncUnsafeCell<T> {
    pub const fn get(&self) -> *mut T {
        self.value.get()
    }
}
