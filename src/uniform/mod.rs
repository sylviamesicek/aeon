use crate::common::NodeSpace;
use crate::geometry::Rectangle;

pub struct Mesh<const N: usize> {
    bounds: Rectangle<N>,
    size: [usize; N],
    offsets: Vec<usize>,
}

impl<const N: usize> Mesh<N> {
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

    pub fn levels(self: &Self) -> usize {
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
}
