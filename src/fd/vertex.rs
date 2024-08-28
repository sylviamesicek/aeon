use std::array;

use crate::geometry::IndexSpace;

#[derive(Clone, Copy, Debug)]
pub struct StencilSpace<const N: usize> {
    size: [usize; N],
}

impl<const N: usize> StencilSpace<N> {
    pub fn new(size: [usize; N]) -> Self {
        Self { size }
    }

    pub fn size(self) -> [usize; N] {
        self.size
    }

    pub fn index_from_vertex(self, vertex: [usize; N]) -> usize {
        IndexSpace::new(self.size).linear_from_cartesian(vertex)
    }

    pub fn apply(self, corner: [usize; N], stencils: [&[f64]; N], field: &[f64]) -> f64 {
        let ssize: [_; N] = array::from_fn(|axis| stencils[axis].len());

        let mut result = 0.0;

        for offset in IndexSpace::new(ssize).iter() {
            let mut weight = 1.0;

            for axis in 0..N {
                weight *= stencils[axis][offset[axis]];
            }

            let vertex = array::from_fn(|axis| corner[axis] + offset[axis]);
            let index = self.index_from_vertex(vertex);
            result += field[index] * weight;
        }

        result
    }

    pub fn apply_axis(
        self,
        corner: [usize; N],
        stencil: &[f64],
        axis: usize,
        field: &[f64],
    ) -> f64 {
        let mut stencils: [&[f64]; N] = [&[1.0]; N];
        stencils[axis] = stencil;

        self.apply(corner, stencils, field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply() {
        let data = vec![
            0.0, 1.0, 2.0, 3.0, 4.0, //      0
            5.0, 6.0, 7.0, 8.0, 9.0, //      1
            10.0, 11.0, 12.0, 13.0, 14.0, // 2
            15.0, 16.0, 17.0, 18.0, 19.0, // 3
            20.0, 21.0, 22.0, 23.0, 24.0, // 4
        ];

        let space = StencilSpace::new([5, 5]);
        let stencils: [&[f64]; 2] = [&[-1.5, 0.0, -0.5], &[1.0, 2.0, 1.0]];

        let apply = |corner| space.apply(corner, stencils, &data);
        assert_eq!(apply([0, 0]), -1.0 + -15.0 - 7.0 - 15.0 - 6.0);
        assert_eq!(
            apply([1, 2]),
            -1.5 * 11.0 - 0.5 * 13.0 - 3.0 * 16.0 + -18.0 - 1.5 * 21.0 - 0.5 * 23.0
        );

        let apply = |corner| space.apply(corner, [&[0.0, 3.0, -4.0], &[1.0]], &data);
        assert_eq!(apply([0, 0]), -5.0);
        assert_eq!(apply([1, 2]), 36.0 - 52.0);
    }

    // #[test]
    // fn support() {
    //     let space = VertexSpace::new([7, 7]);
    //     let support = |vertex| [space.support(vertex, 2, 0), space.support(vertex, 2, 1)];

    //     assert_eq!(
    //         support([0, 0]),
    //         [Support::Negative(0), Support::Negative(0)]
    //     );
    //     assert_eq!(support([5, 3]), [Support::Positive(1), Support::Interior]);
    //     assert_eq!(support([2, 4]), [Support::Interior, Support::Interior]);
    // }
}
