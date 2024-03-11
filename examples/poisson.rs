use aeon::{
    common::{AntiSymmetricBoundary, Mixed, RobinBoundary, Simple, SymmetricBoundary},
    prelude::*,
};
use std::{f64::consts::PI, path::PathBuf};
use vtkio::model::*;

type BoundarySet = Mixed<2, Simple<AntiSymmetricBoundary<2>>, Simple<RobinBoundary<2>>>;
type BoundarySet2 = Mixed<2, Simple<AntiSymmetricBoundary<2>>, Simple<SymmetricBoundary<2>>>;

// const BOUNDARY_SET: BoundarySet = Mixed::new(
//     Simple::new(AntiSymmetricBoundary),
//     Simple::new(RobinBoundary::nuemann()),
// );

const BOUNDARY_SET: BoundarySet2 = Mixed::new(
    Simple::new(AntiSymmetricBoundary),
    Simple::new(SymmetricBoundary),
);

struct Field {}

impl Projection<2> for Field {
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
            .axis::<2>(0)
            .second_derivative(&BOUNDARY_SET, src, f_rr);
        block
            .axis::<2>(1)
            .second_derivative(&BOUNDARY_SET, src, f_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = f_rr[i] + f_zz[i];
        }
    }

    fn apply_diag(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
        let f_rr = arena.alloc(block.len());
        let f_zz = arena.alloc(block.len());

        block
            .axis::<2>(0)
            .second_derivative_diag(&BOUNDARY_SET, f_rr);
        block
            .axis::<2>(1)
            .second_derivative_diag(&BOUNDARY_SET, f_zz);

        for (i, _) in block.iter().enumerate() {
            dest[i] = f_rr[i] + f_zz[i];
        }
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

// struct Laplacian<'a> {
//     field: &'a [f64],
// }

// impl<'a> Projection<2> for Laplacian<'a> {
//     fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
//         let f_rr = arena.alloc(block.len());
//         let f_zz = arena.alloc(block.len());

//         let field = block.auxillary(self.field);

//         block
//             .axis::<4>(0)
//             .second_derivative(&BOUNDARY_SET, field, f_rr);
//         block
//             .axis::<2>(1)
//             .second_derivative(&BOUNDARY_SET, field, f_zz);

//         for (i, _) in block.iter().enumerate() {
//             dest[i] = f_rr[i] + f_zz[i];
//         }
//     }
// }

// struct LaplacianDiag {}

// impl Projection<2> for LaplacianDiag {
//     fn evaluate(self: &Self, arena: &Arena, block: &Block<2>, dest: &mut [f64]) {
//         let f_rr = arena.alloc(block.len());
//         let f_zz = arena.alloc(block.len());

//         block
//             .axis::<2>(0)
//             .second_derivative_diag(&BOUNDARY_SET, f_rr);
//         block
//             .axis::<2>(1)
//             .second_derivative_diag(&BOUNDARY_SET, f_zz);

//         for (i, _) in block.iter().enumerate() {
//             dest[i] = f_rr[i] + f_zz[i];
//         }
//     }
// }

fn write_vtk_output(mesh: &UniformMesh<2>, field: &[f64], solution: &[f64], rhs: &[f64]) {
    let title = "poisson".to_string();

    let range = mesh.level_node_range(mesh.level_count() - 1);

    let field = &field[range.clone()];
    let solution = &solution[range.clone()];
    let rhs = &rhs[range.clone()];

    let node_space = mesh.level_node_space(mesh.level_count() - 1);

    let cell_space = node_space.cell_space();
    let vertex_space = node_space.vertex_space();

    let cell_total = node_space.cell_space().len();

    // Generate Cells

    let mut connectivity = Vec::new();
    let mut offsets = Vec::new();

    for cell in cell_space.iter() {
        let v1 = vertex_space.linear_from_cartesian(cell);
        let v2 = vertex_space.linear_from_cartesian([cell[0], cell[1] + 1]);
        let v3 = vertex_space.linear_from_cartesian([cell[0] + 1, cell[1] + 1]);
        let v4 = vertex_space.linear_from_cartesian([cell[0] + 1, cell[1]]);

        connectivity.push(v1 as u64);
        connectivity.push(v2 as u64);
        connectivity.push(v3 as u64);
        connectivity.push(v4 as u64);

        offsets.push(connectivity.len() as u64);
    }

    let cell_verts = VertexNumbers::XML {
        connectivity,
        offsets,
    };

    let cell_types = vec![CellType::Quad; cell_total];

    let cells = Cells {
        cell_verts,
        types: cell_types,
    };

    // Generate points

    let mut vertices = Vec::new();

    for vertex in vertex_space.iter() {
        let position = node_space.position(vertex);
        vertices.extend([position[0], position[1], 0.0]);
    }

    let points = IOBuffer::new(vertices);

    // Attributes

    let field_attr = Attribute::DataArray(DataArrayBase {
        name: "field".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(field.to_vec()),
    });

    let laplacian_attr = Attribute::DataArray(DataArrayBase {
        name: "solution".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(solution.to_vec()),
    });

    let diag_attr = Attribute::DataArray(DataArrayBase {
        name: "rhs".to_string(),
        elem: ElementType::Scalars {
            num_comp: 1,
            lookup_table: None,
        },
        data: IOBuffer::new(rhs.to_vec()),
    });

    let attributes = Attributes {
        point: vec![field_attr, laplacian_attr, diag_attr],
        cell: Vec::new(),
    };

    let piece = UnstructuredGridPiece {
        points,
        cells,
        data: attributes,
    };

    let vtk = Vtk {
        version: (2, 2).into(),
        title: title.clone(),
        byte_order: ByteOrder::LittleEndian,
        data: DataSet::UnstructuredGrid {
            meta: None,
            pieces: vec![Piece::Inline(Box::new(piece))],
        },
        file_path: None,
    };

    // Write to output
    let file_path = PathBuf::from(format!("output/{title}.vtu"));
    vtk.export(&file_path).unwrap();
}

pub fn main() {
    let mut arena = Arena::new();

    let mesh = UniformMesh::new(
        Rectangle {
            size: [1.0, 1.0],
            origin: [0.0, 0.0],
        },
        [8, 8],
        3,
    );

    let mut field = vec![0.0; mesh.node_count()];
    let mut solution = vec![0.0; mesh.node_count()];
    let mut rhs = vec![0.0; mesh.node_count()];

    mesh.project(&mut arena, &Field {}, &mut field);
    // mesh.project(&mut arena, &LaplacianOp, &mut laplacian);
    solution.fill(0.0);
    mesh.project(&mut arena, &LaplacianRhs, &mut rhs);

    let mut multigrid: UniformMultigrid<'_, 2, BiCGStabSolver> = UniformMultigrid::new(
        &mesh,
        20,
        10e-12,
        5,
        5,
        &BiCGStabConfig {
            max_iterations: 10000,
            tolerance: 10e-12,
        },
    );

    multigrid.solve(&mut arena, &LaplacianOp, &rhs, &mut solution);

    for i in 0..rhs.len() {
        rhs[i] = solution[i] - field[i];
    }

    write_vtk_output(&mesh, &field, &solution, &rhs);
}
