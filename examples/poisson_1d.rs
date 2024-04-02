use aeon::{
    common::{BlockExt, FreeBoundary, Mixed, SymmetricBoundary},
    prelude::*,
};
use std::{f64::consts::PI, path::PathBuf};

type BoundarySet = Mixed<1, SymmetricBoundary<2>, FreeBoundary>;

const BOUNDARY_SET: BoundarySet = Mixed::new(SymmetricBoundary, FreeBoundary);

struct Solution;

impl Projection<1> for Solution {
    fn evaluate(&self, _: &Arena, block: &Block<1>, dest: &mut [f64]) {
        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let x = position[0];

            dest[i] = x * (PI * x).sin();
        }
    }
}

struct Rhs;

impl Projection<1> for Rhs {
    fn evaluate(&self, _: &Arena, block: &Block<1>, dest: &mut [f64]) {
        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let x = position[0];

            dest[i] = 2.0 * PI * (PI * x).cos() - PI * PI * x * (PI * x).sin();
        }
    }
}

pub struct Laplacian;

impl Operator<1> for Laplacian {
    fn apply(self: &Self, _: &Arena, block: &Block<1>, u: &[f64], dest: &mut [f64]) {
        block.second_derivative::<2>(0, &BOUNDARY_SET, u, dest);
    }

    fn apply_diag(self: &Self, _: &Arena, block: &Block<1>, dest: &mut [f64]) {
        block.second_derivative_diag::<2>(0, &BOUNDARY_SET, dest);
    }

    fn boundary(&self, mut callback: impl aeon::common::BoundaryCallback<1>) {
        callback.axis(0, &BOUNDARY_SET);
        callback.axis(1, &BOUNDARY_SET);
    }
}

fn write_vtk_output<const N: usize>(
    mesh: &UniformMesh<N>,
    exact: &[f64],
    solution: &[f64],
    rhs: &[f64],
    error: &[f64],
    residual: &[f64],
) {
    let title = format!("poisson_1d");
    let file_path = PathBuf::from(format!("output/{title}.vtu"));

    let mut output = DataOut::new(mesh);

    output.attrib_scalar("exact", exact);
    output.attrib_scalar("solution", solution);
    output.attrib_scalar("rhs", rhs);
    output.attrib_scalar("error", error);
    output.attrib_scalar("residual", residual);

    output.export_vtk(&title, file_path).unwrap();
}

fn main() {
    env_logger::builder()
        .format_timestamp(None)
        .filter_level(log::LevelFilter::max())
        .init();

    let mesh = UniformMesh::new(
        Rectangle {
            size: [1.0],
            origin: [0.0],
        },
        [8],
        3,
    );

    log::info!("Node Count: {}", mesh.node_count());
    log::info!("Level Size: {:?}", mesh.level_size(mesh.level_count() - 1));

    let mut arena = Arena::new();

    let mut solution = vec![0.0; mesh.node_count()];
    let mut rhs = vec![0.0; mesh.node_count()];
    let mut error = vec![0.0; mesh.node_count()];
    let mut residial = vec![0.0; mesh.node_count()];
    // let mut application = vec![0.0; mesh.node_count()];

    mesh.project(&mut arena, &Solution, &mut solution);
    mesh.project(&mut arena, &Rhs, &mut rhs);

    let mut approximation = vec![0.0; mesh.node_count()];

    let mut multigrid: UniformMultigrid<'_, 1, BiCGStabSolver> = UniformMultigrid::new(
        &mesh,
        &BiCGStabConfig {
            max_iterations: 10000,
            tolerance: 10e-12,
        },
        &UniformMultigridConfig {
            max_iterations: 100,
            tolerance: 10e-12,
            presmoothing: 5,
            postsmoothing: 5,
        },
    );

    multigrid.solve(&mut arena, &Laplacian, &rhs, &mut approximation);

    for i in 0..rhs.len() {
        error[i] = approximation[i] - solution[i];
    }

    mesh.residual(&mut arena, &rhs, &Laplacian, &approximation, &mut residial);

    write_vtk_output(&mesh, &solution, &approximation, &rhs, &error, &residial);
}
