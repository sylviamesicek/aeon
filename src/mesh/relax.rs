use crate::{
    array::Array,
    common::{Boundary, BoundaryKind},
    geometry::Face,
    mesh::{Block, BlockExt, Driver, MemPool, Mesh, Model, Operator},
    ode::{Ode, Rk4},
    system::{SystemLabel, SystemSlice, SystemSliceMut},
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
    type Output = InitialData;

    fn apply(
        &self,
        block: Block<2>,
        pool: &MemPool,
        src: SystemSlice<'_, Self::Output>,
        mut dest: SystemSliceMut<'_, Self::Output>,
    ) {
        let node_count = block.node_count();

        let psi = &src[InitialData];

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

        let seed = block.aux(&self.seed);

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

        let dest = &mut dest[InitialData];

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
    integrator: Rk4,
}

impl InitialDataSolver {
    pub fn new(steps: usize, cfl: f64) -> Self {
        Self {
            steps,
            cfl,
            integrator: Rk4::new(0),
        }
    }

    pub fn solve(&mut self, mesh: &Mesh<2>, driver: &mut Driver, seed: &[f64], psi: &mut [f64]) {
        let spacing = mesh.minimum_spacing()[0];
        let step = spacing * self.cfl;

        self.integrator.reinit(psi.len());
        self.integrator.system.copy_from_slice(&psi);

        for i in 0..self.steps {
            println!("Step {i}");

            let mut model = Model::new(mesh.clone());
            model.attach_debug_field("solution", self.integrator.system.clone());
            model
                .export_vtk(
                    format!("relax{i}").as_str(),
                    PathBuf::from(format!("output/relax{i}.vtu")),
                )
                .unwrap();

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
        self.mesh.fill_boundary(&EvenBoundary, system);
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        self.mesh.apply(
            &mut self.driver,
            &InitialDataOp { seed: &self.seed },
            SystemSlice::from_contiguous(system),
            SystemSliceMut::from_contiguous(result),
        );
    }
}
