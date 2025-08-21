use crate::{
    horizon::{self, ApparentHorizonFinder},
    systems::*,
};
use aeon::{prelude::*, solver::SolverCallback};
use aeon_app::file;
use clap::{Arg, ArgMatches, Command, arg, value_parser};
use std::{
    convert::Infallible,
    io,
    path::{Path, PathBuf},
};

struct SchwarzschildData {
    mass: f64,
}

impl Function<2> for SchwarzschildData {
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        _: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), Self::Error> {
        let mass = self.mass;

        assert!(output.num_nodes() == engine.num_nodes());

        output.channel_mut(GRZ_CH).fill(0.0);
        output.channel_mut(S_CH).fill(0.0);
        output.channel_mut(KRR_CH).fill(0.0);
        output.channel_mut(KRZ_CH).fill(0.0);
        output.channel_mut(KZZ_CH).fill(0.0);
        output.channel_mut(Y_CH).fill(0.0);
        output.channel_mut(SHIFTR_CH).fill(0.0);
        output.channel_mut(SHIFTZ_CH).fill(0.0);
        output.channel_mut(THETA_CH).fill(0.0);
        output.channel_mut(ZR_CH).fill(0.0);
        output.channel_mut(ZZ_CH).fill(0.0);

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

            output.channel_mut(GRR_CH)[index] = conformal;
            output.channel_mut(GZZ_CH)[index] = conformal;
            output.channel_mut(LAPSE_CH)[index] = lapse;
        }

        Ok(())
    }
}

const REFINEMENTS: usize = 10;

struct HorizonCallback<'a> {
    output: &'a Path,
    positions: &'a mut Vec<[f64; 2]>,
}

impl<'a> SolverCallback<1> for HorizonCallback<'a> {
    type Error = io::Error;

    fn callback(
        &mut self,
        surface: &Mesh<1>,
        input: ImageRef,
        _output: ImageRef,
        iteration: usize,
    ) -> Result<(), Self::Error> {
        if iteration % 100 != 0 {
            return Ok(());
        }

        let i = iteration / 100;
        let radius = input.channel(0);

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
                stride: ExportStride::PerVertex,
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

    let output = file::abs_or_relative(&output_directory)?;

    let mass = matches
        .get_one::<f64>("mass")
        .cloned()
        .ok_or_else(|| eyre::eyre!("failed to specify mass argument"))?;

    // Create output directory
    std::fs::create_dir_all(&output)?;
    std::fs::create_dir_all(&output.join("initial"))?;
    std::fs::create_dir_all(&output.join("horizons"))?;

    let domain = HyperBox {
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

    let mut system = Image::new(mesh.num_nodes(), num_channels(0));

    // Run 10 refinements to reach a suitable mesh.
    for i in 0..REFINEMENTS {
        system.resize(mesh.num_nodes());
        system.storage_mut().fill(0.0);

        mesh.evaluate(
            4,
            SchwarzschildData { mass },
            ImageRef::empty(),
            system.as_mut(),
        )
        .unwrap();
        mesh.fill_boundary(4, FieldConditions, system.as_mut());

        // Perform amr
        mesh.flag_wavelets(4, 1e-13, 1e-6, system.as_ref());
        mesh.balance_flags();

        let mut flag_debug = vec![0; mesh.num_nodes()];
        let mut block_debug = vec![0; mesh.num_nodes()];
        let mut cell_debug = vec![0; mesh.num_nodes()];

        mesh.flags_debug(&mut flag_debug);
        mesh.block_debug(&mut block_debug);
        mesh.cell_debug(&mut cell_debug);

        let mut checkpoint = Checkpoint::default();
        checkpoint.attach_mesh(&mesh);
        save_image(&mut checkpoint, system.as_ref());
        checkpoint.save_int_field("Flags", &flag_debug);
        checkpoint.save_int_field("Blocks", &block_debug);
        checkpoint.save_int_field("Cell", &cell_debug);

        checkpoint.export_vtu(
            output
                .join("initial")
                .join(format!("schwarzschild_{}.vtu", mesh.num_levels())),
            ExportVtuConfig {
                title: "schwarzschild".into(),
                ghost: false,
                stride: ExportStride::PerVertex,
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
        system.as_ref(),
        4,
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
    save_image(&mut checkpoint, system.as_ref());
    checkpoint.export_vtu(
        output.join(format!("schwarzschild.vtu")),
        ExportVtuConfig {
            title: "schwarzschild".into(),
            ghost: false,
            stride: ExportStride::PerVertex,
        },
    )?;

    Ok(())
}

pub trait CommandExt {
    fn schwarzschild_cmd(self) -> Self;
}

impl CommandExt for Command {
    fn schwarzschild_cmd(self) -> Self {
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
                        .required(true)
                        .index(1)
                        .value_parser(value_parser!(f64))
                        .value_name("FLOAT")
                        .help("Mass of black hole to simulate"),
                ),
        )
    }
}

pub fn parse_schwarzschild_cmd(matches: &ArgMatches) -> Option<&ArgMatches> {
    matches.subcommand_matches("schwarzschild")
}
