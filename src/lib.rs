#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

// Included so we can use custom derive macros from `aeon_macros` within this crate.
extern crate self as aeon;

pub mod kernel;
pub mod mesh;
pub mod shared;
pub mod solver;
pub mod system;

pub use aeon_geometry as geometry;
pub use aeon_macros as macros;

/// Provides common types used for most `aeon` applications.
pub mod prelude {
    pub use crate::kernel::{BoundaryKind, Condition, Order, RadiativeParams};
    pub use crate::mesh::{
        Engine, ExportVtuConfig, Function, Mesh, MeshCheckpoint, Projection, SystemCheckpoint,
    };
    pub use crate::system::{
        Empty, EmptyConditions, Pair, PairConditions, Scalar, ScalarConditions, System,
        SystemConditions, SystemSlice, SystemSliceMut, SystemVec,
    };
    pub use aeon_geometry::{Face, IndexSpace, Rectangle};
    pub use aeon_macros::SystemLabel;
}

mod test {
    use super::macros::tensor;

    fn hello() {
        let mut a = [[0.0; 2]; 2];

        const TENSOR_DIM: usize = 2;
        tensor!(i, j => a[i][j] = 0.0);

        let mut a_trace = 0.0;
        tensor!(;k => a_trace = a[k][k]);
    }
}
