use std::ops::Range;

use crate::arena::Arena;
use crate::common::{Block, NodeSpace, Operator, Projection};
use crate::geometry::Rectangle;

mod multigrid;

pub use multigrid::UniformMultigrid;

pub struct UniformMesh<const N: usize> {
    bounds: Rectangle<N>,
    size: [usize; N],
    offsets: Vec<usize>,
}

impl<const N: usize> UniformMesh<N> {
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

    pub fn bounds(self: &Self) -> Rectangle<N> {
        self.bounds.clone()
    }

    pub fn node_count(self: &Self) -> usize {
        *self.offsets.last().unwrap()
    }

    pub fn base_node_count(self: &Self) -> usize {
        self.offsets[1]
    }

    pub fn level_count(self: &Self) -> usize {
        self.offsets.len() - 1
    }

    pub fn level_size(self: &Self, level: usize) -> [usize; N] {
        let mut result = self.size;

        for i in 0..N {
            result[i] *= 1 << level;
        }

        result
    }

    pub fn level_node_space(self: &Self, level: usize) -> NodeSpace<N> {
        let mut size = self.level_size(level);

        for i in 0..N {
            size[i] += 1;
        }

        NodeSpace {
            bounds: self.bounds.clone(),
            size,
        }
    }

    pub fn level_node_offset(self: &Self, level: usize) -> usize {
        self.offsets[level]
    }

    pub fn level_node_range(self: &Self, level: usize) -> Range<usize> {
        self.offsets[level]..self.offsets[level + 1]
    }

    pub fn level_block(self: &Self, level: usize) -> Block<N> {
        let space = self.level_node_space(level);
        let offset = self.level_node_offset(level);
        let total = self.level_node_offset(level + 1) - offset;

        Block::new(space, offset, total)
    }

    pub fn restrict(self: &Self, field: &mut [f64]) {
        assert!(field.len() == self.node_count());

        for level in (1..self.level_count()).rev() {
            self.restrict_level(level, field);
        }
    }

    pub fn restrict_level(self: &Self, level: usize, field: &mut [f64]) {
        assert!(field.len() == self.node_count());

        let offset = self.level_node_offset(level);
        let split = field.split_at_mut(offset);

        let space = self.level_node_space(level);

        let src = &split.1[..self.level_node_offset(level + 1)];
        let dest = &mut split.0[self.level_node_offset(level - 1)..];

        // Perform restriction
        space.restrict_inject(src, dest);
    }

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

        let dest = &mut split.1[..self.level_node_offset(level + 1)];
        let src = &split.0[self.level_node_offset(level - 1)..];

        // Perform restriction
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

    pub fn residual<O: Operator<N>>(
        self: &Self,
        arena: &mut Arena,
        b: &[f64],
        operator: &mut O,
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
        operator: &mut O,
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
