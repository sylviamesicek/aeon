#![allow(clippy::needless_range_loop)]

use crate::{faces, regions, AxisMask, Face, Rectangle, Region, Side};
use bitvec::prelude::*;
use std::iter::once;
use std::{array::from_fn, cmp::Ordering};

mod blocks;
mod interface;
mod nodes;

pub use blocks::TreeBlocks;
pub use interface::{TreeBlockNeighbor, TreeCellNeighbor, TreeNeighbors};
pub use nodes::TreeNodes;

/// Denotes that the cell neighbors the physical boundary of a spatial domain.
pub const NULL: usize = usize::MAX;

/// An implementation of a spatial quadtree in any number of dimensions.
/// Only leaves are stored by this tree, and its hierarchical structure is implied.
/// To enable various optimisations, and avoid certain checks, the tree always contains
/// at least one level of refinement.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tree<const N: usize> {
    /// Domain of the full tree
    domain: Rectangle<N>,
    /// Bounds of each cell in tree
    bounds: Vec<Rectangle<N>>,
    /// Stores neighbors along each face
    neighbors: Vec<usize>,
    /// Index within z-filling curve
    #[serde(with = "aeon_array")]
    indices: [BitVec<usize, Lsb0>; N],
    /// Offsets into indices,
    offsets: Vec<usize>,
}

impl<const N: usize> Tree<N> {
    /// Constructs a new tree consisting of a single root cell that has been
    /// subdivided once.
    pub fn new(domain: Rectangle<N>) -> Self {
        let bounds = AxisMask::enumerate()
            .map(|mask| domain.split(mask))
            .collect();
        let neighbors = AxisMask::<N>::enumerate()
            .flat_map(|mask| {
                faces::<N>().map(move |face| {
                    if mask.is_inner_face(face) {
                        mask.toggled(face.axis).to_linear()
                    } else {
                        NULL
                    }
                })
            })
            .collect();

        let mut indices = from_fn(|_| BitVec::new());

        for mask in AxisMask::<N>::enumerate() {
            for axis in 0..N {
                indices[axis].push(mask.is_set(axis));
            }
        }

        let offsets = (0..=AxisMask::<N>::COUNT).collect();

        Self {
            domain,
            bounds,
            neighbors,
            indices,
            offsets,
        }
    }

    /// Number of cells in tree
    pub fn num_cells(&self) -> usize {
        self.bounds.len()
    }

    /// Moves in a positive direction for each axis that mask is set. This is most
    /// commonly used to find sibling cells in neighbor searches.
    fn sibling(&self, mut cell: usize, sibling: AxisMask<N>) -> usize {
        debug_assert!(cell != NULL);

        let split = self.split(cell);
        let level = self.level(cell);

        (0..N)
            .filter(|&axis| split.is_set(axis) != sibling.is_set(axis))
            .for_each(|axis| {
                cell = self.neighbor(
                    cell,
                    Face {
                        side: sibling.is_set(axis),
                        axis,
                    },
                );

                debug_assert!(cell != NULL);
                debug_assert!(level == self.level(cell));
            });

        debug_assert!(self.split(cell) == sibling);

        cell
    }

    /// Returns the slice of neighbors for each face of the cell.
    pub fn neighbor_slice(&self, cell: usize) -> &[usize] {
        &self.neighbors[cell * 2 * N..(cell + 1) * 2 * N]
    }

    /// Returns the neighbor of the cell along the given face.
    pub fn neighbor(&self, cell: usize, face: Face<N>) -> usize {
        self.neighbors[cell * 2 * N + face.to_linear()]
    }

    /// Finds the outer neighbor along the face of the given subcell.
    pub fn neighbor_after_refinement(
        &self,
        cell: usize,
        subcell: AxisMask<N>,
        face: Face<N>,
    ) -> usize {
        debug_assert!(subcell.is_outer_face(face));

        let result = self.neighbor(cell, face);

        if result == NULL {
            return NULL;
        }

        if self.level(result) <= self.level(cell) {
            return result;
        }

        let mut sibling = subcell;
        sibling.toggle(face.axis);

        self.sibling(result, sibling)
    }

    /// Returns all neighbors on a given face
    pub fn neighbors_on_face(
        &self,
        cell: usize,
        face: Face<N>,
    ) -> impl Iterator<Item = usize> + '_ {
        let neighbor = self.neighbor(cell, face);

        let count = if neighbor == NULL {
            0
        } else if self.level(neighbor) > self.level(cell) {
            AxisMask::<N>::COUNT / 2
        } else {
            1
        };

        let mut splits = face.adjacent_splits();
        splits.next();

        once(neighbor)
            .chain(splits.map(move |mut split| {
                split.toggle(face.axis);
                self.sibling(neighbor, split)
            }))
            .take(count)
    }

    /// Returns the neighbor of the cell in the given region.
    pub fn neighbor_in_region(&self, cell: usize, region: Region<N>) -> usize {
        let split: AxisMask<N> = self.split(cell);
        let level = self.level(cell);

        // Start with self
        let mut neighbor = cell;

        // Flags indicating if we have crossed over a coarse or fine boundary
        let mut neighbor_fine: bool = false;
        let mut neighbor_coarse: bool = false;

        let mut subcell = region.adjacent_split();
        // Iterate over adjacent faces.
        for face in region.adjacent_faces() {
            // Make sure face is compatible with split.
            if neighbor_coarse && split.is_inner_face(face) {
                continue;
            }

            // Snap origin to be adjacent to face
            if neighbor_fine {
                if self.split(neighbor).is_inner_face(face) {
                    neighbor = self.neighbor(neighbor, face);
                }
                debug_assert!(self.split(neighbor).is_outer_face(face));
            }

            // Get neighbor of current cell
            let nneighbor = self.neighbor_after_refinement(neighbor, subcell, face);
            subcell.toggle(face.axis);

            // Short circut if we have encountered a boundary
            if nneighbor == NULL {
                return NULL;
            }

            // Get level of neighbor
            let nlevel = self.level(nneighbor);

            match nlevel.cmp(&level) {
                Ordering::Less => {
                    debug_assert!(nlevel == level - 1);
                    debug_assert!(!neighbor_fine);
                    neighbor_coarse = true;
                    neighbor_fine = false;
                }
                Ordering::Greater => {
                    if nlevel != level + 1 {
                        println!("Cell {cell}, {level}; Neighbor {nneighbor}, {nlevel}");
                    }

                    debug_assert!(nlevel == level + 1);
                    debug_assert!(!neighbor_coarse);
                    neighbor_fine = true;
                    neighbor_coarse = false;
                }
                Ordering::Equal => {
                    neighbor_coarse = false;
                    neighbor_fine = false;
                }
            }

            neighbor = nneighbor;
        }

        neighbor
    }

    /// Iterates over all neighbors in a given region.
    pub fn neighbors_in_region(
        &self,
        cell: usize,
        region: Region<N>,
    ) -> impl Iterator<Item = usize> + '_ {
        let neighbor = self.neighbor_in_region(cell, region);

        let count = if neighbor == NULL {
            0
        } else if self.level(neighbor) > self.level(cell) {
            AxisMask::<N>::COUNT / 2usize.pow(region.adjacency() as u32)
        } else {
            1
        };

        let mut splits = region.adjacent_splits();
        splits.next();

        once(neighbor)
            .chain(splits.map(move |mut split| {
                (0..N)
                    .filter(|&axis| region.side(axis) != Side::Middle)
                    .for_each(|axis| split.toggle(axis));

                self.sibling(neighbor, split)
            }))
            .take(count)
    }

    // pub fn neighborhood(&self, cell: usize, mask: FaceMask<N>) -> impl Iterator<Item = usize> + '_ {
    //     mask.adjacent_regions()
    //         .filter(|region| *region != Region::<N>::CENTRAL)
    //         .flat_map(move |region| self.neighbors_in_region(cell, region))
    // }

    /// Returns all cells that neighbor the given cell including the cell itself.
    pub fn neighborhood(&self, cell: usize) -> impl Iterator<Item = usize> + '_ {
        regions::<N>().flat_map(move |region| self.neighbors_in_region(cell, region))
    }

    /// Returns all coarse cells that neighbor the given cell, including on
    /// corners.
    pub fn neighborhood_coarse(&self, cell: usize) -> impl Iterator<Item = usize> + '_ {
        let split = self.split(cell);

        AxisMask::<N>::enumerate().skip(1).flat_map(move |dir| {
            let region = Region::<N>::new(from_fn(|axis| {
                match (dir.is_set(axis), split.is_set(axis)) {
                    (true, true) => Side::Right,
                    (true, false) => Side::Left,
                    (false, _) => Side::Middle,
                }
            }));

            let neighbor = self.neighbor_in_region(cell, region);

            if neighbor != NULL && self.level(neighbor) < self.level(cell) {
                Some(neighbor)
            } else {
                None
            }
        })
    }

    /// Computes the level of a cell.
    pub fn level(&self, cell: usize) -> usize {
        debug_assert!(cell != NULL);
        debug_assert!(cell < self.num_cells());
        self.offsets[cell + 1] - self.offsets[cell]
    }

    /// Returns the domain of the full quadtree.
    pub fn domain(&self) -> Rectangle<N> {
        self.domain
    }

    /// Returns the bounds of a cell
    pub fn bounds(&self, cell: usize) -> Rectangle<N> {
        self.bounds[cell]
    }

    /// If cell is not root, returns it most recent subdivision.
    pub fn split(&self, cell: usize) -> AxisMask<N> {
        AxisMask::pack(from_fn(|axis| self.indices[axis][self.offsets[cell]]))
    }

    /// Checks whether the given refinement flags are balanced.
    pub fn check_refine_flags(&self, flags: &[bool]) -> bool {
        assert!(flags.len() == self.num_cells());

        for block in 0..self.num_cells() {
            if !flags[block] {
                continue;
            }

            for coarse in self.neighborhood_coarse(block) {
                if !flags[coarse] {
                    return false;
                }
            }
        }

        true
    }

    /// Balances the given refinement flags, flagging additional cells
    /// for refinement to preserve the 2:1 fine coarse ratio between every
    /// two neighbors.
    pub fn balance_refine_flags(&self, flags: &mut [bool]) {
        assert!(flags.len() == self.num_cells());

        loop {
            let mut is_balanced = true;

            for cell in 0..self.num_cells() {
                if !flags[cell] {
                    continue;
                }

                for coarse in self.neighborhood_coarse(cell) {
                    if !flags[coarse] {
                        is_balanced = false;
                        flags[coarse] = true;
                    }
                }
            }

            if is_balanced {
                break;
            }
        }
    }

    /// Fills the map with updated indices after refinement is performed.
    /// If a cell is refined, this will point to the base cell in that new subdivision.
    pub fn refine_index_map(&self, flags: &[bool], map: &mut [usize]) {
        assert!(flags.len() == self.num_cells());
        assert!(map.len() == self.num_cells());

        let mut cursor = 0;

        for cell in 0..self.num_cells() {
            map[cell] = cursor;

            if flags[cell] {
                cursor += AxisMask::<N>::COUNT;
            } else {
                cursor += 1;
            }
        }
    }

    /// Refines the mesh using the given flags (temporary API).
    pub fn refine(&mut self, flags: &[bool]) {
        assert!(self.num_cells() == flags.len());
        assert!(self.check_refine_flags(flags));

        let num_flags = flags.iter().copied().filter(|&p| p).count();
        let total_blocks = self.num_cells() + (AxisMask::<N>::COUNT - 1) * num_flags;

        // ****************************************
        // Prepass to deteremine updated indices **

        // Maps current cell indices to new cell indices.
        let mut update_map = vec![0; self.num_cells()];
        self.refine_index_map(flags, &mut update_map);

        let mut bounds = Vec::with_capacity(total_blocks);
        let mut neighbors = Vec::with_capacity(total_blocks * 2 * N);
        let mut indices = from_fn(|_| BitVec::with_capacity(total_blocks));
        let mut offsets = Vec::with_capacity(total_blocks + 1);
        offsets.push(0);

        for cell in 0..self.num_cells() {
            let level = self.level(cell);

            if flags[cell] {
                let parent = bounds.len();
                // Loop over subdivisions
                for split in AxisMask::<N>::enumerate() {
                    // Set physical bounds of block
                    bounds.push(self.bounds(cell).split(split));
                    // Update neighbors
                    for face in faces::<N>() {
                        if split.is_inner_face(face) {
                            // We can just reflect across axis within this particular group
                            // because we are uniformly refining this cell.
                            let neighbor = parent + split.toggled(face.axis).to_linear();
                            neighbors.push(neighbor);
                        } else {
                            // Gets the neighbor across this face, after any refinement is performed
                            let neighbor = self.neighbor_after_refinement(cell, split, face);

                            if neighbor == NULL {
                                // Propagate boundary information
                                neighbors.push(NULL);
                                continue;
                            }

                            let neighbor_level = self.level(neighbor);

                            let neighbor =
                                match (flags[neighbor], level as isize - neighbor_level as isize) {
                                    (true, -1) => {
                                        // We refine both and the neighbor started finer
                                        update_map[neighbor]
                                            + face.reversed().adjacent_split().to_linear()
                                        // update_map[neighbor + mask.toggled(face.axis).to_linear()]
                                    }
                                    (false, -1) => {
                                        // Neighbor was more refined, but now they are on the same level
                                        update_map[neighbor]
                                    }
                                    (true, 0) => {
                                        // If we refine both and they started on the same level. simply flip
                                        // along axis.
                                        update_map[neighbor] + split.toggled(face.axis).to_linear()
                                    }
                                    (false, 0) => {
                                        // Started on same level, but neighbor was not refined
                                        update_map[neighbor]
                                    }
                                    (true, 1) => {
                                        // We refine both and this block starts finer than neighbor
                                        update_map[neighbor]
                                            + self.split(cell).toggled(face.axis).to_linear()
                                    }

                                    _ => panic!("Unbalanced quadtree."),
                                };

                            neighbors.push(neighbor);
                        }
                    }

                    // Compute index
                    for axis in 0..N {
                        // Multiply current index by 2 and set least significant bit
                        indices[axis].push(split.is_set(axis));
                        indices[axis].extend_from_bitslice(self.index_slice(cell, axis));
                    }
                    let previous = offsets[offsets.len() - 1];
                    offsets.push(previous + self.level(cell) + 1);
                }
            } else {
                // Set physical bounds of block
                bounds.push(self.bounds(cell));

                // Update neighbors
                for face in faces::<N>() {
                    let neighbor = self.neighbor(cell, face);
                    if neighbor == NULL {
                        neighbors.push(NULL);
                        continue;
                    }

                    let neighbor_level = self.level(neighbor);

                    // Is the neighbor along this face being refined?
                    if flags[neighbor] {
                        debug_assert!(neighbor_level <= level);

                        if neighbor_level < level {
                            let mut split = self.split(cell);
                            split.toggle(face.axis);

                            neighbors.push(update_map[neighbor] + split.to_linear());
                        } else {
                            // Find an adjacent split across face.
                            neighbors.push(
                                update_map[neighbor] + face.reversed().adjacent_split().to_linear(),
                            );
                        }
                    } else {
                        neighbors.push(update_map[neighbor])
                    }
                }

                for axis in 0..N {
                    indices[axis].extend_from_bitslice(self.index_slice(cell, axis));
                }

                let previous = offsets[offsets.len() - 1];
                offsets.push(previous + self.level(cell));
            }
        }

        self.bounds.clone_from(&bounds);
        self.neighbors.clone_from(&neighbors);
        self.indices.clone_from(&indices);
        self.offsets.clone_from(&offsets);
    }

    /// Checks that the given coarsening flags are balanced and valid.
    pub fn check_coarsen_flags(&self, flags: &[bool]) -> bool {
        assert!(flags.len() == self.num_cells());

        // Short circuit if this mesh only has two levels.
        if flags.len() == AxisMask::<N>::COUNT {
            return flags.iter().all(|&b| !b);
        }

        // First if any flagging would break 2:1 border, unmark it
        for cell in 0..self.num_cells() {
            if !flags[cell] {
                for neighbor in self.neighborhood_coarse(cell) {
                    // Set any coarser cells to not be coarsened further.
                    if flags[neighbor] {
                        return false;
                    }
                }
            }
        }

        // Make sure only cells that can be coarsened are coarsened. And that every single child of such a cell
        // is flagged.
        let mut cell = 0;

        while cell < self.num_cells() {
            if !flags[cell] {
                cell += 1;
                continue;
            }

            // if flags[cell] {
            let level = self.level(cell);
            let split = self.split(cell);

            if split != AxisMask::<N>::empty() {
                return false;
            }

            for offset in 0..AxisMask::<N>::COUNT {
                if self.level(cell + offset) != level {
                    return false;
                }
            }

            if !flags[cell..cell + AxisMask::<N>::COUNT].iter().all(|&b| b) {
                return false;
            }
            // Skip forwards. We have considered all cases.
            cell += AxisMask::<N>::COUNT;
        }

        true
    }

    /// Balances the given coarsening flags
    pub fn balance_coarsen_flags(&self, flags: &mut [bool]) {
        assert!(flags.len() == self.num_cells());

        // Short circuit if this mesh only has two levels.
        if flags.len() == AxisMask::<N>::COUNT {
            flags.fill(false);
        }

        loop {
            let mut is_balanced = true;

            // First if any flagging would break 2:1 border, unmark it
            for cell in 0..self.num_cells() {
                if !flags[cell] {
                    for neighbor in self.neighborhood_coarse(cell) {
                        // Set any coarser cells to not be coarsened further.
                        if flags[neighbor] {
                            is_balanced = false;
                        }
                        flags[neighbor] = false;
                    }
                }
            }

            // Make sure only cells that can be coarsened are coarsened. And that every single child of such a cell
            // is flagged.
            let mut cell = 0;

            while cell < self.num_cells() {
                if !flags[cell] {
                    cell += 1;
                    continue;
                }

                // if flags[cell] {
                let level = self.level(cell);
                let split = self.split(cell);

                if split != AxisMask::<N>::empty() {
                    flags[cell] = false;
                    is_balanced = false;
                    cell += 1;
                    continue;
                }

                for offset in 0..AxisMask::<N>::COUNT {
                    if self.level(cell + offset) != level {
                        flags[cell] = false;
                        is_balanced = false;
                        cell += 1;
                        continue;
                    }
                }

                if !flags[cell..cell + AxisMask::<N>::COUNT].iter().all(|&b| b) {
                    flags[cell..cell + AxisMask::<N>::COUNT].fill(false);
                    is_balanced = false;
                }
                // Skip forwards. We have considered all cases.
                cell += AxisMask::<N>::COUNT;
            }

            if is_balanced {
                break;
            }
        }
    }

    /// Maps current cells to indices after coarsening is performed.
    pub fn coarsen_index_map(&self, flags: &[bool], map: &mut [usize]) {
        assert!(flags.len() == self.num_cells());
        assert!(map.len() == self.num_cells());

        let mut cursor = 0;
        let mut cell = 0;

        while cell < self.num_cells() {
            if flags[cell] {
                map[cell..cell + AxisMask::<N>::COUNT].fill(cursor);
                cell += AxisMask::<N>::COUNT;
            } else {
                map[cell] = cursor;
                cell += 1;
            }

            cursor += 1;
        }
    }

    /// Coarsens the tree.
    pub fn coarsen(&mut self, flags: &[bool]) {
        assert!(flags.len() == self.num_cells());
        assert!(self.check_coarsen_flags(flags));

        // Compute number of cells after coarsening
        let num_flags = flags.iter().copied().filter(|&p| p).count();
        debug_assert!(num_flags % AxisMask::<N>::COUNT == 0);
        let total_cells = flags.len() - num_flags / AxisMask::<N>::COUNT;

        // Update index map
        let mut update_map = vec![0; self.num_cells()];
        self.coarsen_index_map(flags, &mut update_map);

        // New bounds and neighbors
        let mut bounds = Vec::with_capacity(total_cells);
        let mut neighbors = Vec::with_capacity(total_cells * 2 * N);
        let mut indices = from_fn(|_| BitVec::with_capacity(total_cells));
        let mut offsets = Vec::with_capacity(total_cells + 1);
        offsets.push(0);

        // Loop over cells
        let mut cell = 0;

        while cell < self.num_cells() {
            // This cell (and therefore the following `AxisMask::<N>::COUNT` cells) are to be refined.
            if flags[cell] {
                // Get level of the cell.
                let level = self.level(cell);

                debug_assert!(self.split(cell) == AxisMask::empty());

                // Compute new bounds
                bounds.push({
                    let mut result = self.bounds(cell);

                    for axis in 0..N {
                        result.size[axis] *= 2.0;
                    }

                    result
                });

                // Update neighbors
                for face in faces::<N>() {
                    let split = face.adjacent_split();

                    let neighbor = self.neighbor(cell + split.to_linear(), face);

                    if neighbor == NULL {
                        neighbors.push(NULL);
                    } else {
                        neighbors.push(update_map[neighbor]);
                    }
                }

                for axis in 0..N {
                    indices[axis].extend_from_bitslice(&self.index_slice(cell, axis)[1..]);
                }

                let previous = offsets[offsets.len() - 1];
                offsets.push(previous + level - 1);

                // Skip remaining cells that were refined in this action.
                cell += AxisMask::<N>::COUNT;
            } else {
                // Use same bounds on new tree.
                bounds.push(self.bounds(cell));

                // Update neighbors
                for face in faces::<N>() {
                    let neighbor = self.neighbor(cell, face);
                    if neighbor == NULL {
                        neighbors.push(NULL);
                        continue;
                    } else {
                        neighbors.push(update_map[neighbor]);
                    }
                }

                for axis in 0..N {
                    indices[axis].extend_from_bitslice(self.index_slice(cell, axis));
                }

                let previous = offsets[offsets.len() - 1];
                offsets.push(previous + self.level(cell));

                cell += 1;
            }
        }

        self.bounds.clone_from(&bounds);
        self.neighbors.clone_from(&neighbors);
        self.indices.clone_from(&indices);
        self.offsets.clone_from(&offsets);
    }

    /// Returns the z index for
    fn index_slice(&self, cell: usize, axis: usize) -> &BitSlice<usize, Lsb0> {
        &self.indices[axis][self.offsets[cell]..self.offsets[cell + 1]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{regions, FaceMask};

    #[test]
    fn refine_balancing() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        tree.refine(&[true, false, false, false]);

        // Produce unbalanced flags
        let mut flags = vec![false, false, false, true, false, false, false];
        assert!(!tree.check_refine_flags(&flags));

        // Perform rebalancing
        tree.balance_refine_flags(&mut flags);
        assert_eq!(flags, &[false, false, false, true, true, true, true]);

        assert!(tree.check_refine_flags(&flags));
    }

    #[test]
    fn refinement() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);

        assert_eq!(tree.num_cells(), 4);
        assert_eq!(tree.split(0).unpack(), [false, false]);
        assert_eq!(tree.split(1).unpack(), [true, false]);
        assert_eq!(tree.split(2).unpack(), [false, true]);
        assert_eq!(tree.split(3).unpack(), [true, true]);

        tree.refine(&[false, false, false, true]);

        assert_eq!(tree.num_cells(), 7);

        for block in 0..3 {
            assert_eq!(tree.level(block), 1);
        }
        for block in 3..7 {
            assert_eq!(tree.level(block), 2);
        }

        tree.refine(&[false, false, true, false, false, true, false]);

        assert_eq!(tree.num_cells(), 13);
        assert_eq!(tree.neighbor_slice(0), &[NULL, 1, NULL, 2]);
        assert_eq!(tree.neighbor_slice(2), &[NULL, 3, 0, 4]);
        assert_eq!(tree.neighbor_slice(5), &[4, 8, 3, NULL]);
        assert_eq!(tree.neighbor_slice(10), &[5, 11, 8, NULL]);
    }

    #[test]
    fn uniform() {
        let tree = Tree::new(Rectangle::<2>::UNIT);

        assert_eq!(tree.num_cells(), 4);
        assert_eq!(tree.neighbor_slice(0), &[NULL, 1, NULL, 2]);
        assert_eq!(tree.neighbor_slice(1), &[0, NULL, NULL, 3]);
        assert_eq!(tree.neighbor_slice(2), &[NULL, 3, 0, NULL]);
        assert_eq!(tree.neighbor_slice(3), &[2, NULL, 1, NULL]);

        assert_eq!(
            tree.bounds(0),
            Rectangle {
                origin: [0.0, 0.0],
                size: [0.5, 0.5]
            }
        );

        assert_eq!(
            tree.bounds(1),
            Rectangle {
                origin: [0.5, 0.0],
                size: [0.5, 0.5]
            }
        );

        assert_eq!(
            tree.bounds(2),
            Rectangle {
                origin: [0.0, 0.5],
                size: [0.5, 0.5]
            }
        );

        assert_eq!(
            tree.bounds(3),
            Rectangle {
                origin: [0.5, 0.5],
                size: [0.5, 0.5]
            }
        );

        let mut blocks = TreeBlocks::default();
        blocks.build(&tree);

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks.size(0), [2, 2]);
        assert_eq!(blocks.cells(0), &[0, 1, 2, 3]);

        assert_eq!(blocks.boundary_flags(0), FaceMask::pack([[true; 2]; 2]));
    }

    #[test]
    fn two_levels() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        tree.refine(&[true, false, false, false]);

        assert_eq!(tree.num_cells(), 7);

        let assert_eq_regions = |cell, values: [usize; 9]| {
            for region in regions::<2>() {
                assert_eq!(
                    tree.neighbor_in_region(cell, region),
                    values[region.to_linear()]
                );
            }
        };

        assert_eq_regions(0, [NULL, NULL, NULL, NULL, 0, 1, NULL, 2, 3]);
        assert_eq_regions(1, [NULL, NULL, NULL, 0, 1, 4, 2, 3, 4]);
        assert_eq_regions(2, [NULL, 0, 1, NULL, 2, 3, NULL, 5, 5]);
        assert_eq_regions(3, [0, 1, 4, 2, 3, 4, 5, 5, 6]);
        assert_eq_regions(4, [NULL, NULL, NULL, 1, 4, NULL, 5, 6, NULL]);
        assert_eq_regions(5, [NULL, 2, 4, NULL, 5, 6, NULL, NULL, NULL]);
        assert_eq_regions(6, [3, 4, NULL, 5, 6, NULL, NULL, NULL, NULL]);

        assert_eq!(
            tree.neighbor_after_refinement(5, AxisMask::pack([true, false]), Face::negative(1)),
            3
        );

        assert_eq!(
            tree.neighbor_after_refinement(6, AxisMask::pack([false, false]), Face::negative(0)),
            5
        );

        let mut blocks = TreeBlocks::default();
        blocks.build(&tree);

        assert_eq!(blocks.len(), 3);

        assert_eq!(blocks.size(0), [2, 2]);
        assert_eq!(blocks.cells(0), &[0, 1, 2, 3]);

        assert_eq!(blocks.size(1), [1, 2]);
        assert_eq!(blocks.cells(1), &[4, 6]);

        assert_eq!(blocks.size(2), [1, 1]);
        assert_eq!(blocks.cells(2), &[5]);
    }

    #[test]
    fn three_levels() {
        let mut tree = Tree::new(Rectangle::<2>::UNIT);
        tree.refine(&[true, false, false, false]);
        tree.refine(&[true, false, false, false, false, false, false]);

        assert_eq!(
            tree.neighbor_after_refinement(5, AxisMask::pack([true, false]), Face::negative(1)),
            3
        );

        let mut neighbors = tree.neighbors_on_face(5, Face::negative(1));
        assert_eq!(neighbors.next(), Some(2));
        assert_eq!(neighbors.next(), Some(3));
        assert_eq!(neighbors.next(), None);

        assert_eq!(
            tree.neighbor_in_region(5, Region::new([Side::Right, Side::Left])),
            4
        );

        let mut neighbors = tree.neighbors_in_region(5, Region::new([Side::Middle, Side::Left]));
        assert_eq!(neighbors.next(), Some(2));
        assert_eq!(neighbors.next(), Some(3));
        assert_eq!(neighbors.next(), None);

        let neighbors = [2, 3, 4, 5, 6, 8, 8];
        let offsets = [0, 0, 2, 3, 3, 4, 5, 5, 6, 7];

        for region in regions::<2>() {
            let linear = region.to_linear();
            let slice = &neighbors[offsets[linear]..offsets[linear + 1]];

            for (i, n) in tree.neighbors_in_region(5, region).enumerate() {
                assert_eq!(slice[i], n)
            }
        }

        let mut blocks = TreeBlocks::default();
        blocks.build(&tree);

        assert_eq!(blocks.len(), 5);

        assert_eq!(blocks.size(0), [2, 2]);
        assert_eq!(blocks.cells(0), &[0, 1, 2, 3]);

        assert_eq!(blocks.size(1), [1, 2]);
        assert_eq!(blocks.cells(1), &[4, 6]);

        assert_eq!(blocks.size(2), [1, 1]);
        assert_eq!(blocks.cells(2), &[5]);

        assert_eq!(blocks.size(3), [1, 2]);
        assert_eq!(blocks.cells(3), &[7, 9]);

        assert_eq!(blocks.size(4), [1, 1]);
        assert_eq!(blocks.cells(4), &[8]);
    }

    #[test]
    fn refinement_and_coarsening() {
        let mut tree = Tree::<2>::new(Rectangle::UNIT);
        // Make initially asymmetric.
        tree.refine(&[true, false, false, false]);

        for _ in 0..1 {
            let mut flags: Vec<bool> = vec![true; tree.num_cells()];
            tree.balance_refine_flags(&mut flags);
            tree.refine(&flags);
        }

        for _ in 0..2 {
            let mut flags = vec![true; tree.num_cells()];
            tree.balance_coarsen_flags(&mut flags);
            let mut coarsen_map = vec![0; tree.num_cells()];
            tree.coarsen_index_map(&flags, &mut coarsen_map);
            dbg!(&coarsen_map);
            dbg!(&flags);
            tree.coarsen(&flags);
        }

        assert_eq!(tree, Tree::<2>::new(Rectangle::UNIT));
    }
}
