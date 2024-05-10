use aeon::{
    array::Array,
    common::{Boundary, BoundaryKind},
    geometry::{Face, Rectangle},
    mesh::{Block, BlockExt, Driver, MemPool, Mesh, Model, Operator, Projection, SystemLabel},
    ode::{ForwardEuler, Ode},
};
use aeon_axisymmetry::InitialSystem;
use std::path::PathBuf;

pub struct InitialData;

impl SystemLabel for InitialData {
    const NAME: &'static str = "InitialData";
    type FieldLike<T> = [T; 1];
    fn fields() -> Array<Self::FieldLike<Self>> {
        [InitialData].into()
    }

    fn field_index(&self) -> usize {
        0
    }

    fn field_name(&self) -> String {
        "psi".to_string()
    }
}

pub struct OddBoundary;

impl Boundary<2> for OddBoundary {
    fn kind(&self, face: Face) -> BoundaryKind {
        if face.axis == 0 && face.side == false {
            BoundaryKind::Parity(false)
        } else if face.axis == 1 && face.side == false {
            BoundaryKind::Parity(true)
        } else {
            BoundaryKind::Free
        }
    }
}

pub struct EvenBoundary;

impl Boundary<2> for EvenBoundary {
    fn kind(&self, face: Face) -> BoundaryKind {
        if face.axis == 0 && face.side == false {
            BoundaryKind::Parity(true)
        } else if face.axis == 1 && face.side == false {
            BoundaryKind::Parity(true)
        } else {
            BoundaryKind::Free
        }
    }
}

pub struct InitialDataOp<'a> {
    pub seed: &'a [f64],
}

impl<'a> Operator<2> for InitialDataOp<'a> {
    fn apply(&self, block: Block<2>, pool: &MemPool, psi: &[f64], dest: &mut [f64]) {
        let node_count = block.node_count();
        let range = block.local_from_global();

        let psi_r = pool.alloc_scalar(node_count);
        let psi_z = pool.alloc_scalar(node_count);
        let psi_rr = pool.alloc_scalar(node_count);
        let psi_rz = pool.alloc_scalar(node_count);
        let psi_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &EvenBoundary, psi, psi_r);
        block.derivative::<4>(1, &EvenBoundary, psi, psi_z);
        block.second_derivative::<4>(0, &EvenBoundary, psi, psi_rr);
        block.second_derivative::<4>(1, &EvenBoundary, psi_r, psi_rz);
        block.second_derivative::<4>(1, &EvenBoundary, psi, psi_zz);

        let seed = &self.seed[range];

        let seed_r = pool.alloc_scalar(node_count);
        let seed_z = pool.alloc_scalar(node_count);
        let seed_rr = pool.alloc_scalar(node_count);
        let seed_rz = pool.alloc_scalar(node_count);
        let seed_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &OddBoundary, seed, seed_r);
        block.derivative::<4>(1, &OddBoundary, seed, seed_z);
        block.second_derivative::<4>(0, &OddBoundary, seed, seed_rr);
        block.second_derivative::<4>(1, &OddBoundary, seed_r, seed_rz);
        block.second_derivative::<4>(1, &OddBoundary, seed, seed_zz);

        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            let vars = InitialSystem {
                psi: psi[index],
                psi_r: psi_r[index],
                psi_z: psi_z[index],
                psi_rr: psi_rr[index],
                psi_rz: psi_rz[index],
                psi_zz: psi_zz[index],

                s: seed[index],
                s_r: seed_r[index],
                s_z: seed_z[index],
                s_rr: seed_rr[index],
                s_rz: seed_rz[index],
                s_zz: seed_zz[index],
            };

            let derivs = if rho.abs() <= 10e-10 {
                aeon_axisymmetry::initial_regular(vars, rho, z)
            } else {
                aeon_axisymmetry::initial(vars, rho, z)
            };

            dest[index] = derivs.psi_t;
        }
    }
}

pub struct InitialDataSolver {
    pub steps: usize,
    pub cfl: f64,
    integrator: ForwardEuler,
}

impl InitialDataSolver {
    pub fn new(steps: usize, cfl: f64) -> Self {
        Self {
            steps,
            cfl,
            integrator: ForwardEuler::new(0),
        }
    }

    pub fn solve(&mut self, mesh: &Mesh<2>, driver: &mut Driver, seed: &[f64], _psi: &mut [f64]) {
        let spacing = mesh.minimum_spacing();
        let step = spacing * self.cfl;

        self.integrator.reinit(mesh.node_count());
        self.integrator.system.fill(1.0);

        for i in 0..self.steps {
            println!("Step {i}");

            // if i % 1 == 0 {
            let sl = i;

            let mut model = Model::new(mesh.clone());
            // println!("{:?}", self.integrator.system.clone());
            model.attach_field("solution", self.integrator.system.clone());
            model.attach_field("seed", seed.to_vec());

            model
                .export_vtk(
                    format!("relax").as_str(),
                    PathBuf::from(format!("output/relax{sl}.vtu")),
                )
                .unwrap();
            // }

            let mut ode = RelaxationOde { seed, mesh, driver };
            self.integrator.step(&mut ode, step);
        }
    }
}

pub struct RelaxationOde<'a> {
    seed: &'a [f64],
    mesh: &'a Mesh<2>,
    driver: &'a mut Driver,
}

impl<'a> Ode for RelaxationOde<'a> {
    fn dim(&self) -> usize {
        self.seed.len()
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        self.driver.fill_boundary(self.mesh, &EvenBoundary, system);
    }

    fn derivative(&mut self, source: &[f64], dest: &mut [f64]) {
        self.driver
            .apply(self.mesh, &InitialDataOp { seed: &self.seed }, source, dest);
    }
}

pub struct SeedProjection {
    amplitude: f64,
}

pub struct Seed;

impl SystemLabel for Seed {
    const NAME: &'static str = "Seed";
    type FieldLike<T> = [T; 1];
    fn fields() -> Array<Self::FieldLike<Self>> {
        [Seed].into()
    }

    fn field_index(&self) -> usize {
        0
    }

    fn field_name(&self) -> String {
        "seed".to_string()
    }
}

impl Projection<2> for SeedProjection {
    fn evaluate(&self, block: aeon::mesh::Block<2>, _pool: &MemPool, dest: &mut [f64]) {
        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            dest[index] = -rho * self.amplitude * (-(rho * rho + z * z)).exp();
        }
    }
}

fn main() {
    let mut driver = Driver::new();

    let mesh = Mesh::new(
        Rectangle {
            size: [10.0, 10.0],
            origin: [0.0, 0.0],
        },
        [40, 40],
        2,
    );

    // Project seed values
    let mut seed = vec![0.0; mesh.node_count()];

    driver.project(&mesh, &SeedProjection { amplitude: 1.0 }, &mut seed);

    let mut psi = vec![1.0; mesh.node_count()].into_boxed_slice();

    let mut solver = InitialDataSolver::new(10000, 0.001);
    solver.solve(&mesh, &mut driver, &seed, &mut psi);
}
