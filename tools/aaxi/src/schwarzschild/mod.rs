use std::{
    convert::Infallible,
    io,
    path::{Path, PathBuf},
};

use aeon::{prelude::*, solver::SolverCallback};
use clap::{Arg, ArgMatches, Command, arg, value_parser};

use crate::{
    horizon::{self, ApparentHorizonFinder},
    misc,
    systems::{Constraint, Field, FieldConditions, Fields, Gauge, Metric, ScalarField},
};

struct SchwarzschildData {
    mass: f64,
}

impl Function<2> for SchwarzschildData {
    type Input = Empty;
    type Output = Fields;
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        _: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Self::Error> {
        let mass = self.mass;

        assert!(output.len() == engine.num_nodes());

        output.field_mut(Field::Metric(Metric::Grz)).fill(0.0);
        output.field_mut(Field::Metric(Metric::S)).fill(0.0);
        output.field_mut(Field::Metric(Metric::Krr)).fill(0.0);
        output.field_mut(Field::Metric(Metric::Krz)).fill(0.0);
        output.field_mut(Field::Metric(Metric::Kzz)).fill(0.0);
        output.field_mut(Field::Metric(Metric::Y)).fill(0.0);
        output.field_mut(Field::Gauge(Gauge::Shiftr)).fill(0.0);
        output.field_mut(Field::Gauge(Gauge::Shiftz)).fill(0.0);
        output
            .field_mut(Field::Constraint(Constraint::Theta))
            .fill(0.0);
        output
            .field_mut(Field::Constraint(Constraint::Zr))
            .fill(0.0);
        output
            .field_mut(Field::Constraint(Constraint::Zz))
            .fill(0.0);

        for i in 0..output.system().num_scalar_fields() {
            output
                .field_mut(Field::ScalarField(ScalarField::Phi, i))
                .fill(0.0);
            output
                .field_mut(Field::ScalarField(ScalarField::Pi, i))
                .fill(0.0);
        }

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [rho, z] = engine.position(vertex);

            let mut radius = (rho * rho + z * z).sqrt();

            if radius <= mass / 10.0 {
                radius = mass / 10.0
            }

            let lapse = (mass - 2.0 * radius) / (mass + 2.0 * radius);
            let conformal = (1.0 + mass / (2.0 * radius)).powi(4);

            if conformal >= 1e4 {
                println!("What");
            }

            output.field_mut(Field::Metric(Metric::Grr))[index] = conformal;
            output.field_mut(Field::Metric(Metric::Gzz))[index] = conformal;
            output.field_mut(Field::Gauge(Gauge::Lapse))[index] = lapse;
        }

        Ok(())
    }
}

const REFINEMENTS: usize = 10;

struct HorizonCallback<'a> {
    output: &'a Path,
    positions: &'a mut Vec<[f64; 2]>,
}

impl<'a> SolverCallback<1, Scalar> for HorizonCallback<'a> {
    type Error = io::Error;

    fn callback(
        &mut self,
        surface: &Mesh<1>,
        input: SystemSlice<Scalar>,
        _output: SystemSlice<Scalar>,
        iteration: usize,
    ) -> Result<(), Self::Error> {
        if iteration % 100 != 0 {
            return Ok(());
        }

        let i = iteration / 100;
        let radius = input.field(());

        self.positions.resize(surface.num_nodes(), [0.0; 2]);
        horizon::compute_position_from_radius(surface, radius, &mut self.positions);

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&surface);
        checkpoint.set_embedding(&self.positions);
        checkpoint.save_field("radius", radius);
        checkpoint.export_vtu(
            self.output.join("horizons").join(format!("horizon{i}.vtu")),
            ExportVtuConfig {
                title: "schwarzschild".into(),
                ghost: false,
                stride: 1,
            },
        )?;

        Ok(())
    }
}

pub fn schwarzschild(matches: &ArgMatches) -> eyre::Result<()> {
    let output_directory = matches
        .get_one::<PathBuf>("directory")
        .cloned()
        .ok_or_else(|| eyre::eyre!("failed to specify directory argument"))?;

    let output = misc::abs_or_relative(&output_directory)?;

    let mass = matches
        .get_one::<f64>("mass")
        .cloned()
        .ok_or_else(|| eyre::eyre!("failed to specify mass argument"))?;

    // Create output directory
    std::fs::create_dir_all(&output)?;
    std::fs::create_dir_all(&output.join("initial"))?;
    std::fs::create_dir_all(&output.join("horizons"))?;

    let domain = Rectangle {
        origin: [0., 0.],
        size: [10.0, 10.0],
    };

    let mut mesh = Mesh::new(
        domain,
        6,
        3,
        FaceArray::from_fn(|face| match face.side {
            false => BoundaryClass::Ghost,
            true => BoundaryClass::OneSided,
        }),
    );
    mesh.refine_global();

    let mut system = SystemVec::new(Fields {
        scalar_fields: Vec::new(),
    });

    // Run 10 refinements to reach a suitable mesh.
    for i in 0..REFINEMENTS {
        system.resize(mesh.num_nodes());
        system.contigious_mut().fill(0.0);

        mesh.evaluate(
            4,
            SchwarzschildData { mass },
            SystemSlice::empty(),
            system.as_mut_slice(),
        )
        .unwrap();
        mesh.fill_boundary(Order::<4>, FieldConditions, system.as_mut_slice());

        // Perform amr
        mesh.flag_wavelets(4, 1e-13, 1e-6, system.as_slice());
        mesh.balance_flags();

        let mut flag_debug = vec![0; mesh.num_nodes()];
        let mut block_debug = vec![0; mesh.num_nodes()];
        let mut cell_debug = vec![0; mesh.num_nodes()];

        mesh.flags_debug(&mut flag_debug);
        mesh.block_debug(&mut block_debug);
        mesh.cell_debug(&mut cell_debug);

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        checkpoint.save_system(system.as_slice());
        checkpoint.save_int_field("Flags", &flag_debug);
        checkpoint.save_int_field("Blocks", &block_debug);
        checkpoint.save_int_field("Cell", &cell_debug);

        checkpoint.export_vtu(
            output
                .join("initial")
                .join(format!("schwarzschild{}.vtu", mesh.max_level())),
            ExportVtuConfig {
                title: "schwarzschild".into(),
                ghost: false,
                stride: 1,
            },
        )?;

        if !mesh.requires_regridding() {
            println!("Refined to requisite tolerance");
            break;
        }

        if i == REFINEMENTS - 1 {
            println!("Refined to max levels");
            break;
        }

        // Otherwise regrid and loop
        mesh.regrid();
    }

    let mut finder = ApparentHorizonFinder::new();
    finder.solver.max_steps = 20000;
    finder.solver.cfl = 0.3;
    finder.solver.dampening = 0.4;
    finder.solver.tolerance = 1e-3;
    finder.solver.adaptive = true;

    // Allocate horizon surface
    let mut surface = horizon::surface();
    for _ in 0..6 {
        surface.refine_global();
    }
    let mut radius = vec![mass; surface.num_nodes()];

    // Perform search
    let search = finder.search_with_callback(
        &mesh,
        system.as_slice(),
        Order::<4>,
        &mut surface,
        HorizonCallback {
            output: &output,
            positions: &mut Vec::new(),
        },
        &mut radius,
    );

    println!("Search Result {:?}", search);

    let mut checkpoint = Checkpoint::default();
    checkpoint.attach_mesh(&mesh);
    checkpoint.save_system(system.as_slice());
    checkpoint.export_vtu(
        output.join(format!("schwarzschild.vtu")),
        ExportVtuConfig {
            title: "schwarzschild".into(),
            ghost: false,
            stride: 1,
        },
    )?;

    Ok(())
}

pub trait CommandExt {
    fn schwarzschild_args(self) -> Self;
}

impl CommandExt for Command {
    fn schwarzschild_args(self) -> Self {
        self.subcommand(
            Command::new("schwarzschild")
                .name("schwarzschild")
                .about("Subcommand for simulating schwarzschild spacetime")
                .arg(
                    arg!(--directory <FILE> "Sets the output directory")
                        .required(true)
                        .value_parser(value_parser!(PathBuf)),
                )
                .arg(
                    Arg::new("mass")
                        .value_parser(value_parser!(f64))
                        .help("Mass of black hole to simulate"),
                ),
        )
    }
}
