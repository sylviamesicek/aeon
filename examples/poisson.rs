use aeon::{common::AntiSymmetricBoundary, prelude::*};
use bumpalo::Bump;

pub struct SecondDerivative;

impl<const N: usize> Convolution<N> for SecondDerivative {
    type Kernel = aeon::common::FDSecondDerivative<4>;

    type NegativeBoundary = AntiSymmetricBoundary<2>;
    type PositiveBoundary = AntiSymmetricBoundary<2>;

    fn negative(self: &Self, _: [f64; N]) -> Self::NegativeBoundary {
        AntiSymmetricBoundary::<2>
    }

    fn positive(self: &Self, _: [f64; N]) -> Self::PositiveBoundary {
        AntiSymmetricBoundary::<2>
    }
}

pub struct Laplacian {
    bump: Bump,
}

impl Operator<2> for Laplacian {
    fn apply(self: &Self, block: Block<2>, src: &[f64], dest: &mut [f64]) {
        let tmp = self.bump.alloc_slice_fill_default::<f64>(block.len());

        block.axis(0).evaluate(&SecondDerivative, src, tmp);

        for local in 0..block.len() {
            dest[local] += tmp[local];
        }

        block.axis(1).evaluate(&SecondDerivative, src, tmp);

        for local in 0..block.len() {
            dest[local] += tmp[local];
        }
    }

    fn apply_diag(self: &Self, block: Block<2>, src: &[f64], dest: &mut [f64]) {
        let tmp = self.bump.alloc_slice_fill_default::<f64>(block.len());

        block.axis(0).evaluate_diag(&SecondDerivative, src, tmp);

        for local in 0..block.len() {
            dest[local] += tmp[local];
        }

        block.axis(1).evaluate_diag(&SecondDerivative, src, tmp);

        for local in 0..block.len() {
            dest[local] += tmp[local];
        }
    }
}

pub fn main() {
    let mesh = UniformMesh::new(
        Rectangle {
            size: [1.0, 1.0],
            origin: [0.0, 0.0],
        },
        [8, 8],
        3,
    );

    // Allocate solution
    let mut solution = vec![0.0; mesh.node_count()];

    // Fill right hand side
    let mut rhs = vec![0.0; mesh.node_count()];

    let block = mesh.level_block(mesh.level_count() - 1);

    for node in block.iter() {
        let index = block.global_from_local(node);
        let position = block.position(node);

        rhs[index] = position[0] + position[1];
    }

    mesh.restrict(&mut rhs);

    // Run solver

    let mut multigrid: UniformMultigrid<'_, 2, BiCGStabSolver> = UniformMultigrid::new(
        &mesh,
        100,
        10e-12,
        5,
        5,
        &BiCGStabConfig {
            max_iterations: 1000,
            tolerance: 10e-12,
        },
    );

    multigrid.solve(&Laplacian { bump: Bump::new() }, &rhs, &mut solution);

    println!("Solved")
}
