use aeon::{
    array::Array,
    common::{Boundary, BoundaryKind},
    geometry::{Face, Rectangle},
    mesh::{Block, BlockExt, Driver, MemPool, Mesh, Model, Projection, SystemLabel},
    ode::{Ode, Rk4},
};
// use aeon_axisymmetry::InitialSystem;
use std::path::PathBuf;

const RADIUS: f64 = 10.0;
const MU: f64 = 1.0;
const CFL: f64 = 0.1;

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

pub struct VOuterBoundary<'a> {
    pub u: &'a [f64],
}

impl<'a> Projection<2> for VOuterBoundary<'a> {
    fn evaluate(&self, block: Block<2>, pool: &MemPool, dest: &mut [f64]) {
        let node_count = block.node_count();
        let range = block.local_from_global();

        let psi = &self.u[range.clone()];
        let psi_r = pool.alloc_scalar(node_count);
        let psi_z = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &EvenBoundary, psi, psi_r);
        block.derivative::<4>(1, &EvenBoundary, psi, psi_z);

        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            if (rho - RADIUS).abs() <= 10e-10 || (z - RADIUS).abs() <= 10e-10 {
                let r = (rho * rho + z * z).sqrt();

                let adv = (psi[index] - 1.0) + rho * psi_r[index] + z * psi_z[index];
                dest[index] = MU * psi[index] - adv / r;
            } else {
                continue;
            }
        }
    }
}

pub struct UProjection<'a> {
    pub u: &'a [f64],
    pub v: &'a [f64],
}

impl<'a> Projection<2> for UProjection<'a> {
    fn evaluate(&self, block: Block<2>, _pool: &MemPool, dest: &mut [f64]) {
        let range = block.local_from_global();

        let psi = &self.u[range.clone()];
        let inert = &self.v[range.clone()];

        for vertex in block.iter() {
            let index = block.index_from_vertex(vertex);

            dest[index] = inert[index] - MU * psi[index];
        }
    }
}

pub struct VProjection<'a> {
    pub seed: &'a [f64],
    pub psi: &'a [f64],
}

impl<'a> Projection<2> for VProjection<'a> {
    fn evaluate(&self, block: Block<2>, pool: &MemPool, dest: &mut [f64]) {
        let node_count = block.node_count();
        let range = block.local_from_global();

        let psi = &self.psi[range.clone()];
        let psi_r = pool.alloc_scalar(node_count);
        let psi_z = pool.alloc_scalar(node_count);
        let psi_rr = pool.alloc_scalar(node_count);
        let psi_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &EvenBoundary, psi, psi_r);
        block.derivative::<4>(1, &EvenBoundary, psi, psi_z);
        block.second_derivative::<4>(0, &EvenBoundary, psi, psi_rr);
        block.second_derivative::<4>(1, &EvenBoundary, psi, psi_zz);

        let seed = &self.seed[range.clone()];
        let seed_r = pool.alloc_scalar(node_count);
        let seed_rr = pool.alloc_scalar(node_count);
        let seed_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &OddBoundary, seed, seed_r);
        block.second_derivative::<4>(0, &OddBoundary, seed, seed_rr);
        block.second_derivative::<4>(1, &OddBoundary, seed, seed_zz);

        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            if (rho - RADIUS).abs() <= 10e-10 || (z - RADIUS).abs() <= 10e-10 {
                dest[index] = 0.0;

                continue;
            }

            let laplacian = if rho.abs() <= 10e-10 {
                2.0 * psi_rr[index] + psi_zz[index]
            } else {
                psi_rr[index] + psi_r[index] / rho + psi_zz[index]
            };

            dest[index] = laplacian
                + psi[index] / 4.0
                    * (rho * seed_rr[index] + 2.0 * seed_r[index] + rho * seed_zz[index]);
        }
    }
}

pub struct Relax<'a> {
    seed: &'a [f64],
    mesh: &'a Mesh<2>,
    driver: &'a mut Driver,
}

impl<'a> Ode for Relax<'a> {
    fn dim(&self) -> usize {
        2 * self.mesh.node_count()
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        let (u, v) = system.split_at_mut(self.mesh.node_count());

        // Currently the time derivative commutes with our boundary conditions, so this is allowed.
        self.driver.fill_boundary(self.mesh, &EvenBoundary, u);
        self.driver.fill_boundary(self.mesh, &EvenBoundary, v);

        // Fill v outer boundary conditions
        self.driver.project(self.mesh, &VOuterBoundary { u }, v);
    }

    fn derivative(&mut self, source: &[f64], dest: &mut [f64]) {
        let (usrc, vsrc) = source.split_at(self.mesh.node_count());
        let (udest, vdest) = dest.split_at_mut(self.mesh.node_count());

        self.driver
            .project(self.mesh, &UProjection { u: usrc, v: vsrc }, udest);
        self.driver.project(
            self.mesh,
            &VProjection {
                psi: usrc,
                seed: self.seed,
            },
            vdest,
        );
    }
}

pub struct SeedProjection(f64);

impl Projection<2> for SeedProjection {
    fn evaluate(&self, block: aeon::mesh::Block<2>, _pool: &MemPool, dest: &mut [f64]) {
        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            dest[index] = rho * self.0 * (-(rho * rho + z * z)).exp();
        }
    }
}

pub struct Gaussian(f64);

impl Projection<2> for Gaussian {
    fn evaluate(&self, block: aeon::mesh::Block<2>, _pool: &MemPool, dest: &mut [f64]) {
        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            dest[index] = self.0 * (-(rho * rho + z * z)).exp();
        }
    }
}

fn main() {
    const STEPS: usize = 10000;
    const SKIP_OUT: usize = 10;

    println!("Allocating Driver and Building Mesh");

    let mut driver = Driver::new();

    let mesh = Mesh::new(
        Rectangle {
            size: [RADIUS, RADIUS],
            origin: [0.0, 0.0],
        },
        [40, 40],
        2,
    );

    let h = mesh.minimum_spacing() * CFL;

    println!("Filling Seed Function");

    // Compute seed values.
    let mut seed = vec![0.0; mesh.node_count()].into_boxed_slice();
    driver.project(&mesh, &SeedProjection(1.0), &mut seed);
    driver.fill_boundary(&mesh, &OddBoundary, &mut seed);

    println!("Integrating Psi Values");

    // Compute initial psi values
    let mut integrator = Rk4::new(2 * mesh.node_count());

    {
        let (u, v) = integrator.system.split_at_mut(mesh.node_count());
        u.fill(1.0);
        v.fill(MU);
    }

    let mut derivs = vec![0.0; mesh.node_count()].into_boxed_slice();

    for i in 0..STEPS {
        driver.project(
            &mesh,
            &VProjection {
                seed: &seed,
                psi: &integrator.system[0..mesh.node_count()],
            },
            &mut derivs,
        );
        driver.fill_boundary(&mesh, &EvenBoundary, &mut derivs);

        let norm = derivs.iter().map(|f| f * f).sum::<f64>().sqrt();
        println!("Step {i} / {STEPS} Norm: {norm}");

        if i % SKIP_OUT == 0 {
            let si = i / SKIP_OUT;

            let (u, v) = integrator.system.split_at(mesh.node_count());

            let mut model = Model::new(mesh.clone());
            model.attach_field("psi", u.iter().map(|&p| p - 1.0).collect());
            model.attach_field("inert", v.to_vec());
            model.attach_field("seed", seed.to_vec());
            model.attach_field("derivs", derivs.to_vec());

            model
                .export_vtk(
                    format!("idbrill").as_str(),
                    PathBuf::from(format!("output/idbrill{si}.vtu")),
                )
                .unwrap();
        }

        integrator.step(
            &mut Relax {
                seed: &seed,
                mesh: &mesh,
                driver: &mut driver,
            },
            h,
        );
    }
}
