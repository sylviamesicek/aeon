use aeon::{
    array::Array,
    common::{GhostBoundary, GhostCondition},
    elliptic::{HyperRelaxSolver, OutgoingOrder, OutgoingWave},
    geometry::{Face, Rectangle},
    mesh::{
        Block, BlockExt, Boundary, Driver, MemPool, Mesh, Model, Operator, Projection, Scalar,
        SystemLabel, SystemSlice, SystemSliceMut, SystemVec,
    },
};
// use aeon_axisymmetry::InitialSystem;
use std::path::PathBuf;

const RADIUS: f64 = 10.0;

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

pub struct PsiBoundary;

impl Boundary for PsiBoundary {
    type Label = Scalar;
    type Ghost = EvenBoundary;

    fn boundary(&self, _: Self::Label) -> Self::Ghost {
        EvenBoundary
    }
}

pub struct PsiOperator<'a> {
    pub seed: &'a [f64],
}

impl<'a> Operator<2> for PsiOperator<'a> {
    type Label = Scalar;

    fn apply(
        &self,
        block: Block<2>,
        pool: &MemPool,
        src: SystemSlice<'_, Self::Label>,
        mut dest: SystemSliceMut<'_, Self::Label>,
    ) {
        let node_count = block.node_count();
        let range = block.local_from_global();

        let psi = src.field(Scalar);
        let psi_r = pool.alloc_scalar(node_count);
        let psi_z = pool.alloc_scalar(node_count);
        let psi_rr = pool.alloc_scalar(node_count);
        let psi_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &EvenBoundary, psi, psi_r);
        block.derivative::<4>(1, &EvenBoundary, psi, psi_z);
        block.second_derivative::<4>(0, &EvenBoundary, psi, psi_rr);
        block.second_derivative::<4>(1, &EvenBoundary, psi, psi_zz);

        let dest = dest.field_mut(Scalar);

        let seed = &self.seed[range.clone()];
        let seed_r = pool.alloc_scalar(node_count);
        let seed_rr = pool.alloc_scalar(node_count);
        let seed_zz = pool.alloc_scalar(node_count);

        block.derivative::<4>(0, &OddBoundary, seed, seed_r);
        block.second_derivative::<4>(0, &OddBoundary, seed, seed_rr);
        block.second_derivative::<4>(1, &OddBoundary, seed, seed_zz);

        for vertex in block.iter() {
            let [rho, _z] = block.position(vertex);
            let index = block.index_from_vertex(vertex);

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

    let mut psi = SystemVec::new(mesh.node_count());
    psi.field_mut(Scalar).fill(1.0);

    let mut solver = HyperRelaxSolver::new();
    solver.max_steps = 10000;
    solver.cfl = 0.1;
    solver.outgoing = OutgoingWave::Sommerfeld(1.0);
    solver.outgoing_order = OutgoingOrder::Fourth;
    solver.dampening = 0.0;

    solver.solve(
        &mut driver,
        &mesh,
        &PsiOperator { seed: &seed },
        &PsiBoundary,
        psi.as_mut_slice(),
    );

    let mut model = Model::new(mesh.clone());
    model.attach_field("psi", psi.field(Scalar).iter().map(|&p| p - 1.0).collect());
    model.attach_field("seed", seed.to_vec());

    model
        .export_vtk(
            format!("idbrill").as_str(),
            PathBuf::from(format!("output/idbrill.vtu")),
        )
        .unwrap();
}
