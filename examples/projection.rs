use aeon_tk::prelude::*;

#[derive(Clone)]
struct MyFunction;

impl Projection<2> for MyFunction {
    fn project(&self, _: usize, [x, y]: [f64; 2]) -> f64 {
        x.sin() * y.exp()
    }
}

#[derive(Clone)]
struct MyBoundary;

impl Boundary<2> for MyBoundary {
    fn kind(&self, _channel: usize, _: Face<2>) -> BoundaryKind {
        BoundaryKind::StrongDirichlet
    }

    fn dirichlet(&self, _channel: usize, [x, y]: [f64; 2]) -> DirichletParams {
        DirichletParams {
            target: x.sin() * y.exp(),
            strength: 1.0,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut mesh = Mesh::new(
        HyperBox::<2>::UNIT, // domain
        4,                   // Width of each cell in nodes
        4,
        2,                                                  // Number of ghost nodes along each axis
        BoundaryClasses::from_fn(|_| BoundaryClass::Ghost), // Boundaries use centered stencils
    );
    mesh.refine_global();

    for i in 0..10 {
        let mut image = Image::new(1, mesh.num_nodes());
        mesh.project(MyFunction, image.as_mut());
        mesh.fill_boundary(MyBoundary, image.as_mut());

        mesh.flag_wavelets(1e-13, 1e-9, image.as_ref());
        mesh.balance_flags();

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_field("MyFunction", image.channel(0));

        checkpoint.export_vtu(
            format!("myfunction{i}.vtu"),
            ExportVtuConfig {
                title: "MyFunction".to_string(),
                ghost: false,
                stride: ExportStride::PerCell,
            },
        )?;

        if !mesh.requires_regridding() {
            break;
        }

        mesh.regrid();
    }

    Ok(())
}
