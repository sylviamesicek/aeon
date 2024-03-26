use aeon::{
    common::{AntiSymmetricBoundary, Mixed, RobinBoundary, Simple},
    prelude::*,
};
use std::{f64::consts::PI, path::PathBuf};

type BoundarySet = Mixed<2, Simple<AntiSymmetricBoundary<4>>, Simple<RobinBoundary<4>>>;

const BOUNDARY_SET: BoundarySet = Mixed::new(
    Simple::new(AntiSymmetricBoundary),
    Simple::new(RobinBoundary::nuemann()),
);

struct Exact;

impl Projection<2> for Exact {
    fn evaluate(self: &Self, _: &Arena, block: &Block<2>, dest: &mut [f64]) {
        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);

            dest[i] = (position[0] * PI / 2.0).sin() * (position[1] * PI / 2.0).sin();
        }
    }
}

struct LaplacianOp;

impl Operator<2> for LaplacianOp {
    fn apply(self: &Self, arena: &Arena, block: &Block<2>, src: &[f64], dest: &mut [f64]) {
        let f_rr = arena.alloc(block.len());
        let f_zz = arena.alloc(block.len());

        block
            .axis::<4>(0)
            .second_derivative(&BOUNDARY_SET, src, f_rr);
        block
            .axis::<4>(1)
            .second_derivative(&BOUNDARY_SET, src, f_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = f_rr[i] + f_zz[i];
        }
    }

    fn apply_diag(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let f_rr = arena.alloc(block.len());
        let f_zz = arena.alloc(block.len());

        block
            .axis::<4>(0)
            .second_derivative_diag(&BOUNDARY_SET, f_rr);
        block
            .axis::<4>(1)
            .second_derivative_diag(&BOUNDARY_SET, f_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = f_rr[i] + f_zz[i];
        }
    }

    fn diritchlet(_axis: usize, face: bool) -> bool {
        face == false
    }
}

struct LaplacianRhs;

impl Projection<2> for LaplacianRhs {
    fn evaluate(self: &Self, _: &Arena, block: &Block<2>, dest: &mut [f64]) {
        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);

            dest[i] =
                -PI * PI / 2.0 * (position[0] * PI / 2.0).sin() * (position[1] * PI / 2.0).sin();
        }
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
    let title = format!("poisson");
    let file_path = PathBuf::from(format!("output/{title}.vtu"));

    let mut output = DataOut::new(mesh);

    output.attrib_scalar("exact", exact);
    output.attrib_scalar("solution", solution);
    output.attrib_scalar("rhs", rhs);
    output.attrib_scalar("error", error);
    output.attrib_scalar("residual", residual);

    output.export_vtk(&title, file_path).unwrap();
}

pub fn main() {
    env_logger::builder()
        .format_timestamp(None)
        .filter_level(log::LevelFilter::max())
        .init();

    log::info!("Running Poisson's Equation Multigrid Solver");

    let mut arena = Arena::new();

    let mesh = UniformMesh::new(
        Rectangle {
            size: [1.0, 1.0],
            origin: [0.0, 0.0],
        },
        [8, 8],
        3,
    );

    let mut exact = vec![0.0; mesh.node_count()];
    let mut solution = vec![0.0; mesh.node_count()];
    let mut rhs = vec![0.0; mesh.node_count()];
    let mut error = vec![0.0; mesh.node_count()];
    let mut residual = vec![0.0; mesh.node_count()];

    mesh.project(&mut arena, &Exact, &mut exact);
    mesh.project(&mut arena, &LaplacianRhs, &mut rhs);

    {
        solution.fill(0.0);
        let mut multigrid: UniformMultigrid<'_, 2, BiCGStabSolver> = UniformMultigrid::new(
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

        multigrid.solve(&mut arena, &LaplacianOp, &rhs, &mut solution);
    }

    for i in 0..rhs.len() {
        error[i] = exact[i] - solution[i];
    }

    mesh.residual(&mut arena, &rhs, &LaplacianOp, &solution, &mut residual);

    write_vtk_output(&mesh, &exact, &solution, &rhs, &error, &residual);
}
