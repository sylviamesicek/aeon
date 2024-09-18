use std::cell::RefCell;

use aeon_basis::{Boundary, BoundaryKind, Element, NodeWindow};
use aeon_geometry::faces;
use rayon::iter::{ParallelBridge, ParallelIterator};
use thread_local::ThreadLocal;

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

        let tl = ThreadLocal::new();

        (0..self.blocks.len()).for_each(|block| {
            let boundary = self.block_boundary(block, boundary.clone());

            let store = tl.get_or(|| {
                (
                    RefCell::new(vec![0.0; support]),
                    RefCell::new(vec![0.0; support]),
                )
            });

            let mut imsrc = store.0.borrow_mut();
            let mut imdest = store.1.borrow_mut();

            let nodes = self.block_nodes(block);
            let space = self.block_space(block);

            let system = unsafe { field.slice(nodes.clone()).fields() };

            for &cell in self.blocks.cells(block) {
                let is_cell_on_boundary = self.is_cell_on_boundary(cell, boundary.clone());

                // Window of nodes on element.
                let window = if is_cell_on_boundary {
                    self.element_coarse_window(cell)
                } else {
                    self.element_window(cell)
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
                                flags[cell] = true;
                                break 'fields;
                            }
                        }
                    } else {
                        element.wavelet(&imsrc, &mut imdest);

                        for point in element.diagonal_int_points() {
                            if imdest[point].abs() > tolerance {
                                flags[cell] = true;
                                break 'fields;
                            }
                        }
                    }
                }
            }
        });
    }
}
