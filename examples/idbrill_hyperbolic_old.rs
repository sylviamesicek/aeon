use aeon::{
    array::Array,
    common::{GhostBoundary, GhostCondition},
    geometry::{Face, Rectangle},
    mesh::{
        Block, BlockExt, Driver, MemPool, Mesh, Model, Projection, Scalar, SystemLabel,
        SystemSliceMut,
    },
    ode::{Ode, Rk4},
};
// use aeon_axisymmetry::InitialSystem;
use std::path::PathBuf;

const RADIUS: f64 = 10.0;
const MU: f64 = 1.0;
const CFL: f64 = 0.1;

#[derive(Clone)]
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

impl GhostBoundary for OddBoundary {
    fn condition(&self, face: Face) -> GhostCondition {
        if face.axis == 0 && face.side == false {
            GhostCondition::Parity(false)
        } else if face.axis == 1 && face.side == false {
            GhostCondition::Parity(true)
        } else {
            GhostCondition::Free
        }
    }
}

pub struct EvenBoundary;

impl GhostBoundary for EvenBoundary {
    fn condition(&self, face: Face) -> GhostCondition {
        if face.axis == 0 && face.side == false {
            GhostCondition::Parity(true)
        } else if face.axis == 1 && face.side == false {
            GhostCondition::Parity(true)
        } else {
            GhostCondition::Free
        }
    }
}

pub struct UProjection<'a> {
    pub u: &'a [f64],
    pub v: &'a [f64],
}

impl<'a> Projection<2> for UProjection<'a> {
    type Label = Scalar;

    fn evaluate(&self, block: Block<2>, pool: &MemPool, mut dest: SystemSliceMut<'_, Scalar>) {
        let range = block.local_from_global();
        let node_count = block.node_count();

        let dest = dest.field_mut(Scalar);

        let psi = &self.u[range.clone()];
        let inert = &self.v[range.clone()];

        let psi_r = pool.alloc_scalar(node_count);
        let psi_z = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &EvenBoundary, psi, psi_r);
        block.derivative::<4>(1, &EvenBoundary, psi, psi_z);

        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            if (rho - RADIUS).abs() <= 10e-10 || (z - RADIUS).abs() <= 10e-10 {
                // dest[index] = 0.0;

                let r = (rho * rho + z * z).sqrt();
                let adv = (psi[index] - 1.0) + rho * psi_r[index] + z * psi_z[index];
                dest[index] = -adv / r;

                continue;
            }

            dest[index] = inert[index] - MU * psi[index];
        }
    }
}

pub struct VProjection<'a> {
    pub seed: &'a [f64],
    pub psi: &'a [f64],
    pub v: &'a [f64],
}

impl<'a> Projection<2> for VProjection<'a> {
    type Label = Scalar;

    fn evaluate(&self, block: Block<2>, pool: &MemPool, mut dest: SystemSliceMut<'_, Scalar>) {
        let node_count = block.node_count();
        let range = block.local_from_global();

        let dest = dest.field_mut(Scalar);

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

        let v = &self.v[range.clone()];
        let v_r = pool.alloc_scalar(node_count);
        let v_z = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &EvenBoundary, v, v_r);
        block.derivative::<4>(1, &EvenBoundary, v, v_z);

        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            if (rho - RADIUS).abs() <= 10e-10 || (z - RADIUS).abs() <= 10e-10 {
                // dest[index] = 0.0;

                let r = (rho * rho + z * z).sqrt();
                let adv = (v[index] - MU) + rho * v_r[index] + z * v_z[index];
                dest[index] = -adv / r;

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
        self.driver
            .fill_boundary_scalar(self.mesh, &EvenBoundary, u);
        self.driver
            .fill_boundary_scalar(self.mesh, &EvenBoundary, v);

        // Fill v outer boundary conditions
        // self.driver.project(self.mesh, &VOuterBoundary { u }, v);
    }

    fn derivative(&mut self, source: &[f64], dest: &mut [f64]) {
        let (usrc, vsrc) = source.split_at(self.mesh.node_count());
        let (udest, vdest) = dest.split_at_mut(self.mesh.node_count());

        self.driver.project(
            self.mesh,
            &UProjection { u: usrc, v: vsrc },
            SystemSliceMut::from_contiguous(udest),
        );
        self.driver.project(
            self.mesh,
            &VProjection {
                psi: usrc,
                seed: self.seed,
                v: vsrc,
            },
            SystemSliceMut::from_contiguous(vdest),
        );
    }
}

pub struct SeedProjection(f64);

impl Projection<2> for SeedProjection {
    type Label = Scalar;
    fn evaluate(
        &self,
        block: aeon::mesh::Block<2>,
        _pool: &MemPool,
        mut dest: SystemSliceMut<'_, Scalar>,
    ) {
        let dest = dest.field_mut(Scalar);

        for vertex in block.iter() {
            let [rho, z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

            dest[index] = rho * self.0 * (-(rho * rho + z * z)).exp();
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
    driver.project(
        &mesh,
        &SeedProjection(1.0),
        SystemSliceMut::from_contiguous(&mut seed),
    );
    driver.fill_boundary_scalar(&mesh, &OddBoundary, &mut seed);

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
                v: &integrator.system[mesh.node_count()..],
            },
            SystemSliceMut::from_contiguous(&mut derivs),
        );
        driver.fill_boundary_scalar(&mesh, &EvenBoundary, &mut derivs);

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
