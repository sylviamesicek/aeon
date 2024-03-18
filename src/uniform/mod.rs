use std::ops::Range;

use crate::arena::Arena;
use crate::common::{Block, NodeSpace, Operator, Projection};
use crate::geometry::Rectangle;

mod io;
mod multigrid;

pub use io::DataOut;
pub use multigrid::{UniformMultigrid, UniformMultigridConfig};

/// Represents a uniform mesh defined on a domain. Such a mesh consists
/// of a set of uniform levels over some rectangular bounds.
#[derive(Debug)]
pub struct UniformMesh<const N: usize> {
    /// Uniform bounds for mesh.
    bounds: Rectangle<N>,
    /// Number of cells on base block.
    size: [usize; N],
    /// Node offsets for each level.
    offsets: Vec<usize>,
}

impl<const N: usize> UniformMesh<N> {
    /// Constructs a new uniform mesh.
    pub fn new(bounds: Rectangle<N>, size: [usize; N], levels: usize) -> Self {
        for i in 0..N {
            assert!(size[i] % 2 == 0);
        }

        let mut offsets = vec![0; levels + 1];
        let mut level_size = size;

        offsets[0] = 0;

        for i in 0..levels {
            let mut total = 1;

            for i in 0..N {
                total *= level_size[i] + 1;
            }

            offsets[i + 1] = total + offsets[i];

            for i in 0..N {
                level_size[i] *= 2;
            }
        }

        Self {
            bounds,
            size,
            offsets,
        }
    }

    /// Physical bounds of the mesh.
    pub fn bounds(self: &Self) -> Rectangle<N> {
        self.bounds.clone()
    }

    /// Total number of nodes in the mesh.
    pub fn node_count(self: &Self) -> usize {
        *self.offsets.last().unwrap()
    }

    /// Number of nodes on base level of the mesh.
    pub fn base_node_count(self: &Self) -> usize {
        self.offsets[1]
    }

    /// Number of levels in mesh.
    pub fn level_count(self: &Self) -> usize {
        self.offsets.len() - 1
    }

    /// Number of cells along each axis for a given level.
    pub fn level_size(self: &Self, level: usize) -> [usize; N] {
        let mut result = self.size;

        for i in 0..N {
            result[i] *= 1 << level;
        }

        result
    }

    /// Constructs a node space for a level in the mesh.
    pub fn level_node_space(self: &Self, level: usize) -> NodeSpace<N> {
        NodeSpace {
            bounds: self.bounds.clone(),
            size: self.level_size(level),
        }
    }

    /// Offset into a node vector for the given level.
    pub fn level_node_offset(self: &Self, level: usize) -> usize {
        self.offsets[level]
    }

    /// Node range for a level.
    pub fn level_node_range(self: &Self, level: usize) -> Range<usize> {
        self.offsets[level]..self.offsets[level + 1]
    }

    /// Constructs a block for a level in the mesh.
    pub fn level_block(self: &Self, level: usize) -> Block<N> {
        let space = self.level_node_space(level);
        let offset = self.level_node_offset(level);
        let total = self.level_node_offset(level + 1) - offset;

        Block::new(space, offset, total)
    }

    /// Computes the spacing for the finest level of the mesh.
    pub fn min_spacing(self: &Self) -> f64 {
        let space = self.level_node_space(self.level_count() - 1);

        let mut result = f64::MAX;

        for i in 0..N {
            result = result.min(space.spacing(i));
        }

        result
    }

    // ***********************************
    // Node Operations *******************
    // ***********************************

    /// Injects values from the finest level to lower levels.
    pub fn restrict(self: &Self, field: &mut [f64]) {
        assert!(field.len() == self.node_count());

        for level in (1..self.level_count()).rev() {
            self.restrict_level(level, field);
        }
    }

    /// Injects values from `level -> level - 1`.
    pub fn restrict_level(self: &Self, level: usize, field: &mut [f64]) {
        assert!(field.len() == self.node_count());

        let offset = self.level_node_offset(level);
        let split = field.split_at_mut(offset);

        let space = self.level_node_space(level);

        let src = &split.1[0..self.level_node_offset(level + 1) - self.level_node_offset(level)];
        let dest = &mut split.0[self.level_node_offset(level - 1)..self.level_node_offset(level)];

        // Perform restriction
        space.restrict_inject(src, dest);
    }

    /// Fully restricts values from `level -> level - 1`.
    pub fn restrict_level_full(self: &Self, level: usize, field: &mut [f64]) {
        let range = self.level_node_range(level);
        let space = self.level_node_space(level);

        for i in 0..N {
            space.axis(i).restrict(&mut field[range.clone()]);
        }

        self.restrict_level(level, field);
    }

    pub fn prolong(self: &Self, field: &mut [f64]) {
        for level in 1..self.level_count() {
            self.prolong_level(level, field);
        }
    }

    pub fn prolong_level(self: &Self, level: usize, field: &mut [f64]) {
        assert!(field.len() == self.node_count());

        let offset = self.level_node_offset(level);
        let split = field.split_at_mut(offset);

        let space = self.level_node_space(level);

        let dest =
            &mut split.1[0..self.level_node_offset(level + 1) - self.level_node_offset(level)];
        let src = &split.0[self.level_node_offset(level - 1)..self.level_node_offset(level)];

        // Perform prolongation
        space.prolong_inject(src, dest);
    }

    pub fn prolong_level_full(self: &Self, level: usize, field: &mut [f64]) {
        let range = self.level_node_range(level);
        let space = self.level_node_space(level);

        self.prolong_level(level, field);

        for i in 0..N {
            space.axis(i).prolong(&mut field[range.clone()]);
        }
    }

    pub fn copy_level(self: &Self, level: usize, src: &[f64], dest: &mut [f64]) {
        let range = self.level_node_range(level);

        (&mut dest[range.clone()]).copy_from_slice(&src[range.clone()]);
    }

    pub fn project<P: Projection<N>>(
        self: &Self,
        arena: &mut Arena,
        projection: &P,
        dest: &mut [f64],
    ) {
        self.project_level(self.level_count() - 1, arena, projection, dest);
        self.restrict(dest);
    }

    pub fn project_level<P: Projection<N>>(
        self: &Self,
        level: usize,
        arena: &mut Arena,
        projection: &P,
        dest: &mut [f64],
    ) {
        let range = self.level_node_range(level);
        let block = self.level_block(level);

        projection.evaluate(arena, &block, &mut dest[range]);

        arena.reset();
    }

    pub fn apply<O: Operator<N>>(
        self: &Self,
        arena: &mut Arena,
        operator: &O,
        x: &[f64],
        dest: &mut [f64],
    ) {
        for i in 0..self.level_count() {
            self.apply_level(i, arena, operator, x, dest);
        }
    }

    pub fn apply_level<O: Operator<N>>(
        self: &Self,
        level: usize,
        arena: &mut Arena,
        operator: &O,
        x: &[f64],
        dest: &mut [f64],
    ) {
        let range = self.level_node_range(level);
        let block = self.level_block(level);

        operator.apply(arena, &block, &x[range.clone()], &mut dest[range.clone()]);
    }

    pub fn residual<O: Operator<N>>(
        self: &Self,
        arena: &mut Arena,
        b: &[f64],
        operator: &O,
        x: &[f64],
        dest: &mut [f64],
    ) {
        for i in 0..self.level_count() {
            self.residual_level(i, arena, b, operator, x, dest);
        }
    }

    pub fn residual_level<O: Operator<N>>(
        self: &Self,
        level: usize,
        arena: &mut Arena,
        b: &[f64],
        operator: &O,
        x: &[f64],
        dest: &mut [f64],
    ) {
        let range = self.level_node_range(level);
        let block = self.level_block(level);

        operator.apply(arena, &block, &x[range.clone()], &mut dest[range.clone()]);

        for i in range {
            dest[i] = b[i] - dest[i];
        }
    }

    pub fn norm(self: &Self, field: &[f64]) -> f64 {
        let mut result = 0.0;

        for &f in field {
            result += f * f;
        }

        result.sqrt()
    }
}
