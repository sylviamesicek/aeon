/// Groups cells of a `Tree` into uniform blocks, for more efficient inter-cell communication and multithreading.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeBlocks<const N: usize> {
    /// Stores each cell's position within its parent's block.
    #[serde(with = "crate::array::vec")]
    active_cell_positions: Vec<[usize; N]>,
    /// Maps cell to the block that contains it.
    active_cell_to_block: Vec<usize>,
    /// Stores the size of each block.
    #[serde(with = "crate::array::vec")]
    block_sizes: Vec<[usize; N]>,
    /// A flattened list of lists (for each block) that stores
    /// a local cell index to global cell index map.
    block_active_indices: Vec<ActiveCellIndex>,
    /// The offsets for the aforementioned flattened list of lists.
    block_active_offsets: Vec<usize>,
    /// The physical bounds of each block.
    block_bounds: Vec<Rectangle<N>>,
    /// The level of refinement of each block.
    block_levels: Vec<usize>,
    /// Stores whether block face is on physical boundary.
    boundaries: BitVec,
}

impl<const N: usize> TreeBlocks<N> {
    /// Rebuilds the tree block structure from existing geometric information. Performs greedy meshing
    /// to group cells into blocks.
    pub fn build(&mut self, tree: &Tree<N>) {
        self.build_blocks(tree);
        self.build_bounds(tree);
        self.build_boundaries(tree);
        self.build_levels(tree);
    }

    // Number of blocks in the mesh.
    pub fn len(&self) -> usize {
        self.block_sizes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the cells associated with the given block.
    pub fn active_cells(&self, block: usize) -> &[ActiveCellIndex] {
        &self.block_active_indices
            [self.block_active_offsets[block]..self.block_active_offsets[block + 1]]
    }

    /// Size of a given block, measured in cells.
    pub fn size(&self, block: usize) -> [usize; N] {
        self.block_sizes[block]
    }

    /// Returns the bounds of the given block.
    pub fn bounds(&self, block: usize) -> Rectangle<N> {
        self.block_bounds[block]
    }

    pub fn level(&self, block: usize) -> usize {
        self.block_levels[block]
    }

    /// Returns boundary flags for a block.
    pub fn boundary_flags(&self, block: usize) -> FaceMask<N> {
        let mut flags = [[false; 2]; N];

        for face in faces::<N>() {
            flags[face.axis][face.side as usize] =
                self.boundaries[block * 2 * N + face.to_linear()];
        }

        FaceMask::pack(flags)
    }

    /// Returns the position of the cell within the block.
    pub fn active_cell_position(&self, cell: ActiveCellIndex) -> [usize; N] {
        self.active_cell_positions[cell.0]
    }

    pub fn active_cell_block(&self, cell: ActiveCellIndex) -> usize {
        self.active_cell_to_block[cell.0]
    }

    fn build_blocks(&mut self, tree: &Tree<N>) {
        let num_active_cells = tree.num_active_cells();

        // Resize/reset various maps
        self.active_cell_positions.resize(num_active_cells, [0; N]);
        self.active_cell_positions.fill([0; N]);

        self.active_cell_to_block
            .resize(num_active_cells, usize::MAX);
        self.active_cell_to_block.fill(usize::MAX);

        self.block_sizes.clear();
        self.block_active_indices.clear();
        self.block_active_offsets.clear();

        // Loop over each cell in the tree
        for active in 0..num_active_cells {
            if self.active_cell_to_block[active] != usize::MAX {
                // This cell already belongs to a block, continue.
                continue;
            }

            // Get index of next block
            let block = self.block_sizes.len();

            self.active_cell_positions[active] = [0; N];
            self.active_cell_to_block[active] = block;

            self.block_sizes.push([1; N]);
            let block_cell_offset = self.block_active_indices.len();

            self.block_active_offsets.push(block_cell_offset);
            self.block_active_indices.push(ActiveCellIndex(active));

            // Try expanding the block along each axis.
            for axis in 0..N {
                // Perform greedy meshing.
                'expand: loop {
                    let face = Face::<N>::positive(axis);

                    let size = self.block_sizes[block];
                    let space = IndexSpace::new(size);

                    // Make sure every cell on face is suitable for expansion.
                    for index in space.face(Face::positive(axis)).iter() {
                        // Retrieves the cell on this face
                        let cell = tree.cell_from_active_index(
                            self.block_active_indices
                                [block_cell_offset + space.linear_from_cartesian(index)],
                        );
                        let level = tree.level(cell);
                        // We can only expand if:
                        // 1. We are not on a physical boundary.
                        // 2. We did not pass over a periodic boundary.
                        // 3. The neighbor is the same level of refinement.
                        // 4. The neighbor does not already belong to another block.
                        let Some(neighbor) = tree.neighbor(cell, face) else {
                            break 'expand;
                        };

                        if !tree.is_boundary_face(cell, face) {
                            break 'expand;
                        }

                        if level != tree.level(neighbor) {
                            break 'expand;
                        }

                        if self.active_cell_to_block
                            [tree.active_index_from_cell(neighbor).unwrap().0]
                            != usize::MAX
                        {
                            break 'expand;
                        }
                    }

                    // We may now expand along this axis
                    for index in space.face(Face::positive(axis)).iter() {
                        let active = self.block_active_indices
                            [block_cell_offset + space.linear_from_cartesian(index)];

                        let cell = tree.cell_from_active_index(active);
                        let cell_neighbor = tree.neighbor(cell, face).unwrap();
                        let active_neighbor = tree.active_index_from_cell(cell_neighbor).unwrap();

                        self.active_cell_positions[active_neighbor.0] = index;
                        self.active_cell_positions[active_neighbor.0][axis] += 1;
                        self.active_cell_to_block[active_neighbor.0] = block;

                        self.block_active_indices.push(active_neighbor);
                    }

                    self.block_sizes[block][axis] += 1;
                }
            }
        }

        self.block_active_offsets
            .push(self.block_active_indices.len());
    }

    fn build_bounds(&mut self, tree: &Tree<N>) {
        self.block_bounds.clear();

        for block in 0..self.len() {
            let size = self.block_sizes[block];
            let a = *self.active_cells(block).first().unwrap();

            let cell_bounds = tree.bounds(tree.cell_from_active_index(a));

            self.block_bounds.push(Rectangle {
                origin: cell_bounds.origin,
                size: array::from_fn(|axis| cell_bounds.size[axis] * size[axis] as f64),
            })
        }
    }

    fn build_boundaries(&mut self, tree: &Tree<N>) {
        self.boundaries.clear();

        for block in 0..self.len() {
            let a = 0;
            let b: usize = self.active_cells(block).len() - 1;

            for face in faces::<N>() {
                let active = if face.side {
                    self.active_cells(block)[b]
                } else {
                    self.active_cells(block)[a]
                };
                let cell = tree.cell_from_active_index(active);
                self.boundaries.push(tree.neighbor(cell, face).is_none());
            }
        }
    }

    fn build_levels(&mut self, tree: &Tree<N>) {
        self.block_levels.resize(self.len(), 0);
        for block in 0..self.len() {
            let active = self.active_cells(block)[0];
            let cell = tree.cell_from_active_index(active);
            self.block_levels[block] = tree.level(cell);
        }
    }
}
