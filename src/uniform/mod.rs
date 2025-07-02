use crate::{kernel::BoundaryClass, prelude::Rectangle};

const DISS_SIX_ORDER_0_7: [f64; 8] = [4.0, -27.0, 78.0, -125.0, 120.0, -69.0, 22.0, -3.0];
const DISS_SIX_ORDER_1_6: [f64; 8] = [3.0, -20.0, 57.0, -90.0, 85.0, -48.0, 15.0, -2.0];
const DISS_SIX_ORDER_2_5: [f64; 8] = [2.0, -13.0, 36.0, -55.0, 50.0, -27.0, 8.0, -1.0];
const DISS_SIX_ORDER_3_3: [f64; 7] = [1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0];
const DISS_SIX_ORDER_5_2: [f64; 8] = [-1.0, 8.0, -27.0, 50.0, -55.0, 36.0, -13.0, 2.0];
const DISS_SIX_ORDER_6_1: [f64; 8] = [-2.0, 15.0, -48.0, 85.0, -90.0, 57.0, -20.0, 3.0];

pub struct UniformGrid<const N: usize> {
    /// What physical bounds does this cover?
    bounds: Rectangle<N>,
    /// Number of cells along each axis
    size: [usize; N],
    /// Number of ghost nodes padding each face.
    ghost: usize,
    /// Boundary class along negative faces.
    negative_boundary: [BoundaryClass; N],
    /// Boundary class along positive faces.
    positive_boundary: [BoundaryClass; N],
}
