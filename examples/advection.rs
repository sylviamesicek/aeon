use aeon::{
    common::{FreeBoundary, Simple},
    prelude::*,
};
use std::path::PathBuf;

type BoundarySet = Simple<FreeBoundary>;
const BOUNDARY_SET: BoundarySet = Simple::new(FreeBoundary);

const DIMENSION: usize = 2;
const RADIUS: f64 = 10.0;
const CFL: f64 = 0.1;
const STEPS: usize = 1000;
const SIGMA: f64 = 1.0;
const DISS: f64 = 0.0;
const LEVELS: usize = 4;
const ORDER: usize = 2;
const OUTPUT: bool = true;

fn is_approximately_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < 10e-10
}

struct Exact {
    time: f64,
}

impl Projection<DIMENSION> for Exact {
    fn evaluate(self: &Self, _: &Arena, block: &Block<DIMENSION>, dest: &mut [f64]) {
        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);

            let mut offset = position;
            offset[0] -= self.time;

            let mut r2 = 0.0;

            for axis in 0..DIMENSION {
                r2 += offset[axis] * offset[axis];
            }

            dest[i] = (-r2 / (SIGMA * SIGMA)).exp();
        }
    }
}

struct Derivative<'a> {
    source: &'a [f64],
}

impl<'a> Projection<DIMENSION> for Derivative<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<DIMENSION>, dest: &mut [f64]) {
        let source_x = arena.alloc(block.len());
        let source = block.auxillary(self.source);

        block
            .axis::<ORDER>(0)
            .derivative(&BOUNDARY_SET, source, source_x);

        for (i, node) in block.iter().enumerate() {
            let position = block.position(node);
            let x = position[0];

            if is_approximately_equal(x, RADIUS) {
                dest[i] = -source_x[i];
            } else if is_approximately_equal(x, -RADIUS) {
                dest[i] = source_x[i];
            }

            dest[i] = -source_x[i];
        }
    }
}

pub struct Dissipation<'a> {
    source: &'a [f64],
}

impl<'a> Projection<DIMENSION> for Dissipation<'a> {
    fn evaluate(self: &Self, arena: &Arena, block: &Block<DIMENSION>, dest: &mut [f64]) {
        // Bizarre rust analyzer warning.
        let mut _f = dest;

        let diss_x = arena.alloc(block.len());
        // let diss_y = arena.alloc(block.len());
        let src = block.auxillary(self.source);

        block
            .axis::<ORDER>(0)
            .dissipation(&BOUNDARY_SET, src, diss_x);
        // block
        //     .axis::<ORDER>(1)
        //     .dissipation(&BOUNDARY_SET, src, diss_y);

        for (i, _) in block.iter().enumerate() {
            // _f[i] = DISS * (diss_x[i] + diss_y[i]);
            _f[i] = DISS * diss_x[i];
        }
    }
}

struct Integrator<'a> {
    mesh: &'a UniformMesh<DIMENSION>,

    system: Vec<f64>,

    scratch: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,

    time: f64,
}

impl<'a> Integrator<'a> {
    pub fn new(mesh: &'a UniformMesh<DIMENSION>, system: Vec<f64>) -> Self {
        let node_count = mesh.node_count();

        Self {
            mesh,

            system,

            scratch: vec![0.0; node_count],
            k1: vec![0.0; node_count],
            k2: vec![0.0; node_count],
            k3: vec![0.0; node_count],
            k4: vec![0.0; node_count],

            time: 0.0,
        }
    }

    pub fn step(self: &mut Self, arena: &mut Arena, k: f64) {
        // ************************
        // K1

        Self::compute_derivative(self.mesh, arena, &self.system, &mut self.k1);

        // *************************
        // K2

        for i in 0..self.system.len() {
            self.scratch[i] = self.system[i] + k / 2.0 * self.k1[i];
        }

        Self::compute_derivative(self.mesh, arena, &self.scratch, &mut self.k2);

        // *************************
        // K3

        for i in 0..self.system.len() {
            self.scratch[i] = self.system[i] + k / 2.0 * self.k2[i];
        }

        Self::compute_derivative(self.mesh, arena, &self.scratch, &mut self.k3);

        // **************************
        // K4

        for i in 0..self.system.len() {
            self.scratch[i] = self.system[i] + k * self.k3[i];
        }

        Self::compute_derivative(self.mesh, arena, &self.scratch, &mut self.k4);

        // Dissipation
        self.mesh.project(
            arena,
            &Dissipation {
                source: &self.system,
            },
            &mut self.scratch,
        );

        // Final step

        for i in 0..self.system.len() {
            self.system[i] +=
                k / 6.0 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i]);
        }

        for i in 0..self.system.len() {
            self.system[i] += self.scratch[i];
        }

        self.mesh.restrict(&mut self.system);

        // Increment time
        self.time += k;
    }

    fn compute_derivative(
        mesh: &'a UniformMesh<DIMENSION>,
        arena: &mut Arena,
        system: &[f64],
        dest: &mut [f64],
    ) {
        mesh.project(arena, &Derivative { source: system }, dest);
    }
}

fn write_vtk_output(
    step: usize,
    mesh: &UniformMesh<DIMENSION>,
    system: &[f64],
    exact: &[f64],
    residual: &[f64],
) {
    if !OUTPUT {
        return;
    }

    let title = format!("advection{step}");
    let file_path = PathBuf::from(format!("output/{title}.vtu"));

    let mut output = DataOut::new(mesh);

    output.attrib_scalar("system", system);
    output.attrib_scalar("exact", exact);
    output.attrib_scalar("error", residual);

    output.export_vtk(&title, file_path).unwrap()
}

fn main() {
    env_logger::builder()
        .format_timestamp(None)
        .filter_level(log::LevelFilter::max())
        .init();

    log::info!("Running Advection Equation Solver");

    let mut arena = Arena::new();

    let mesh = UniformMesh::new(
        Rectangle {
            size: [2.0 * RADIUS; DIMENSION],
            origin: [-RADIUS; DIMENSION],
        },
        [8; DIMENSION],
        LEVELS,
    );

    let mut system = vec![0.0; mesh.node_count()];
    mesh.project(&mut arena, &Exact { time: 0.0 }, &mut system);

    let mut exact = vec![0.0; mesh.node_count()];
    mesh.project(&mut arena, &Exact { time: 0.0 }, &mut exact);

    let mut residual = vec![0.0; mesh.node_count()];

    for i in 0..system.len() {
        residual[i] = exact[i] - system[i];
    }

    let mut integrator = Integrator::new(&mesh, system);

    // Write output
    write_vtk_output(0, &mesh, &integrator.system, &exact, &residual);

    let min_spacing = mesh.min_spacing();
    let k = CFL * min_spacing;

    log::info!("Step Size {k}");

    for i in 0..STEPS {
        log::info!(
            "Step {}, Residual {}, Time {}",
            i,
            mesh.norm(&residual) / (residual.len() as f64).sqrt(),
            k * i as f64
        );

        // Step
        integrator.step(&mut arena, k);

        // Error
        mesh.project(
            &mut arena,
            &Exact {
                time: integrator.time,
            },
            &mut exact,
        );

        for i in 0..integrator.system.len() {
            residual[i] = exact[i] - integrator.system[i];
        }
        // Write output
        write_vtk_output(i + 1, &mesh, &integrator.system, &exact, &residual);
    }
}
