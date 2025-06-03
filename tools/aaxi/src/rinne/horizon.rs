#![allow(mixed_script_confusables)]

use crate::rinne::eqs::{HorizonData, horizon};
use crate::rinne::{Field, Fields, Metric};
use aeon::solver::{HyperRelaxError, SolverCallback};
use aeon::{
    element::UniformInterpolate,
    kernel::{Kernels, NodeWindow, node_from_vertex},
    mesh::UnsafeThreadCache,
    prelude::*,
    solver::HyperRelaxSolver,
};
use aeon_tensor::Space;
use core::f64;
use reborrow::Reborrow;
use thiserror::Error;

pub enum HorizonStatus {
    CollapsedToZero,
    Converged,
}

#[derive(Debug, Error)]
pub enum HorizonError {
    #[error("surface point not contained in mesh: ${0:?}")]
    SurfaceNotContained([f64; 2]),
    #[error("surface doesn't sucessfully relax")]
    SurfaceDiverged,
    #[error("interpolation didn't converge")]
    InterpolateFailed,
}

pub struct ApparentHorizonFinder {
    /// Error tolerance (relaxation stops once error goes below this value).
    pub tolerance: f64,
    /// Maximum number of relaxation steps to perform
    pub max_steps: usize,
    /// Dampening term η.
    pub dampening: f64,
    /// CFL factor for ficticuous time step.
    pub cfl: f64,

    surface: Mesh<1>,
    surface_radius: SystemVec<Scalar>,

    surface_to_cell: Vec<CellId>,
    surface_position: Vec<[f64; 2]>,

    cache: UnsafeThreadCache<UniformInterpolate<2>>,
    solver: HyperRelaxSolver,
}

impl ApparentHorizonFinder {
    pub fn new() -> Self {
        let mut surface = Mesh::new(
            Rectangle::from_aabb([0.0], [f64::consts::PI / 2.0]),
            4,
            2,
            FaceArray::from_fn(|_| BoundaryClass::Periodic),
        );

        for _ in 0..3 {
            surface.refine_global();
        }

        let mut surface_radius = SystemVec::new(Scalar);
        surface_radius.resize(surface.num_nodes());
        surface_radius.field_mut(()).fill(1.0);

        let surface_to_cell = vec![CellId(0); surface.num_nodes()];
        let surface_position = vec![[0.0; 2]; surface.num_nodes()];

        Self {
            tolerance: 1e-5,
            max_steps: 100000,
            dampening: 1.0,
            cfl: 0.1,

            surface,
            surface_radius,

            surface_to_cell,
            surface_position,

            cache: UnsafeThreadCache::new(),

            solver: HyperRelaxSolver::default(),
        }
    }

    pub fn search<K: Kernels>(
        &mut self,
        mesh: &Mesh<2>,
        order: K,
        fields: SystemSlice<Fields>,
    ) -> Result<HorizonStatus, HorizonError>
    where
        K: Sync,
    {
        self.surface_radius.resize(self.surface.num_nodes());
        self.surface_radius.field_mut(()).fill(1.0);
        self.surface_to_cell
            .resize(self.surface.num_nodes(), CellId(0));
        self.surface_to_cell.fill(CellId(0));

        self.surface_position
            .resize(self.surface.num_nodes(), [0.0; 2]);

        self.solver.tolerance = self.tolerance;
        self.solver.max_steps = self.max_steps;
        self.solver.dampening = self.dampening;
        self.solver.cfl = self.cfl;
        self.solver.adaptive = true;
        // Run solver
        let result = self.solver.solve_with_callback(
            &mut self.surface,
            order,
            HorizonRadialBoundary,
            HorizonRadialDerivs::<K> {
                surface_to_cell: &mut self.surface_to_cell,
                surface_position: &mut self.surface_position,
                mesh,
                fields,
                _phantom: std::marker::PhantomData,
            },
            HorizonCallback { mesh },
            self.surface_radius.as_mut_slice(),
        );

        match result {
            Ok(()) => Ok(HorizonStatus::Converged),
            Err(HyperRelaxError::CallbackFailed(HorizonCallbackError::CollapsedToZero)) => {
                Ok(HorizonStatus::CollapsedToZero)
            }
            Err(HyperRelaxError::Diverged | HyperRelaxError::FailedToMeetTolerance) => {
                Err(HorizonError::SurfaceDiverged)
            }
            Err(HyperRelaxError::FunctionFailed(err)) => Err(err),
        }
    }
}

impl Clone for ApparentHorizonFinder {
    fn clone(&self) -> Self {
        Self {
            tolerance: self.tolerance,
            max_steps: self.max_steps,
            dampening: self.dampening,
            cfl: self.cfl,
            surface: self.surface.clone(),
            surface_radius: self.surface_radius.clone(),
            surface_to_cell: self.surface_to_cell.clone(),
            surface_position: self.surface_position.clone(),
            cache: UnsafeThreadCache::new(),
            solver: self.solver.clone(),
        }
    }
}

#[derive(Clone)]
struct HorizonRadialBoundary;

impl SystemBoundaryConds<1> for HorizonRadialBoundary {
    type System = Scalar;

    fn kind(&self, _label: <Self::System as System>::Label, _face: Face<1>) -> BoundaryKind {
        BoundaryKind::Symmetric
    }
}

struct HorizonRadialDerivs<'a, K> {
    surface_to_cell: &'a mut [CellId],
    surface_position: &'a mut [[f64; 2]],
    mesh: &'a Mesh<2>,
    fields: SystemSlice<'a, Fields>,
    _phantom: std::marker::PhantomData<K>,
}

impl<'a, K: Kernels> Function<1> for HorizonRadialDerivs<'a, K> {
    type Input = Scalar;
    type Output = Scalar;
    type Error = HorizonError;

    fn preprocess(
        &mut self,
        surface: &mut Mesh<1>,
        mut input: SystemSliceMut<Self::Input>,
    ) -> Result<(), Self::Error> {
        let radius = input.field_mut(());

        for block in 0..surface.num_blocks() {
            let nodes = surface.block_nodes(BlockId(block));
            let space = surface.block_space(BlockId(block));

            let surface_radius = &radius[nodes.clone()];
            let surface_position = &mut self.surface_position[nodes.clone()];
            let surface_to_cell = &mut self.surface_to_cell[nodes.clone()];

            for node in space.inner_window() {
                let index = space.index_from_node(node);
                let [theta] = space.position(node);
                let radius = surface_radius[index];

                let point = position(radius, theta);

                if self.mesh.tree().domain().contains(point) {
                    return Err(HorizonError::SurfaceNotContained(point));
                }

                let cell = self
                    .mesh
                    .tree()
                    .cell_from_point_cached(point, surface_to_cell[index]);

                surface_position[index] = point;
                surface_to_cell[index] = cell;
            }
        }

        Ok(())
    }

    fn evaluate(
        &self,
        engine: impl Engine<1>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Self::Error> {
        const S: Space<2> = Space::<2>;

        let mesh = self.mesh;
        let fields = self.fields.rb();

        let surface_radius = input.field(());
        let surface_deriv = output.field_mut(());

        let nodes = engine.node_range();

        let surface_positions = &self.surface_position[nodes.clone()];
        let surface_to_cell = &self.surface_to_cell[nodes.clone()];

        // Size of every cell along each axis
        let cell_size = [mesh.width() + 1; 2];
        // Number of support points per cell axis
        let cell_support = mesh.width() + 1;

        let num_nodes_per_cell = cell_size.iter().product();

        let scratch: &mut [f64] = engine.alloc(num_nodes_per_cell);

        let mut interpolate = UniformInterpolate::<2>::default();

        for [vertex] in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex([vertex]);
            let [theta] = engine.position([vertex]);
            let [r, z] = surface_positions[index];
            // Compute radius
            let radius = surface_radius[index];
            // Compute dr/dθ and d²r/dθ²
            let radius_deriv = engine.derivative(surface_radius, 0, [vertex]);
            let radius_second_deriv = engine.second_derivative(surface_radius, 0, [vertex]);

            // *********************************
            // Interpolate values from Mesh ****

            let mesh_cell = surface_to_cell[index];
            let mesh_active_cell = mesh.tree().active_index_from_cell(mesh_cell).unwrap();
            let mesh_block = mesh.blocks().active_cell_block(mesh_active_cell);

            let block_space = mesh.block_space(mesh_block);
            let cell_space = NodeWindow {
                origin: node_from_vertex(mesh.cell_node_origin(mesh_active_cell)),
                size: cell_size,
            };

            interpolate
                .build(cell_support, cell_support, [r, z])
                .map_err(|_err| HorizonError::InterpolateFailed)?;

            let block_fields = fields.slice(nodes.clone());

            let grr_f = block_fields.field(Field::Metric(Metric::Grr));
            let grz_f = block_fields.field(Field::Metric(Metric::Grz));
            let gzz_f = block_fields.field(Field::Metric(Metric::Gzz));
            let seed_f = block_fields.field(Field::Metric(Metric::S));

            let krr_f = block_fields.field(Field::Metric(Metric::Krr));
            let krz_f = block_fields.field(Field::Metric(Metric::Krz));
            let kzz_f = block_fields.field(Field::Metric(Metric::Kzz));
            let y_f = block_fields.field(Field::Metric(Metric::Y));

            // Handle metric values
            macro_rules! interpolate_value {
                ($output:ident, $field:ident) => {
                    for (i, node) in cell_space.iter().enumerate() {
                        scratch[i] = $field[block_space.index_from_node(node)];
                    }
                    let $output = interpolate.apply(&scratch);
                };
            }

            macro_rules! interpolate_derivative {
                ($output:ident, $field:ident, $axis:expr) => {
                    for (i, node) in cell_space.iter().enumerate() {
                        scratch[i] =
                            block_space.evaluate_axis(K::derivative(), node, $field, $axis);
                    }
                    let $output = interpolate.apply(&scratch);
                };
            }

            interpolate_value!(grr, grr_f);
            interpolate_value!(grz, grz_f);
            interpolate_value!(gzz, gzz_f);
            interpolate_derivative!(grr_r, grr_f, 0);
            interpolate_derivative!(grr_z, grr_f, 1);
            interpolate_derivative!(grz_r, grz_f, 0);
            interpolate_derivative!(grz_z, grz_f, 1);
            interpolate_derivative!(gzz_r, gzz_f, 0);
            interpolate_derivative!(gzz_z, gzz_f, 1);
            interpolate_value!(seed, seed_f);
            interpolate_derivative!(seed_r, seed_f, 0);
            interpolate_derivative!(seed_z, seed_f, 1);

            interpolate_value!(krr, krr_f);
            interpolate_value!(krz, krz_f);
            interpolate_value!(kzz, kzz_f);
            interpolate_value!(y, y_f);

            let system = HorizonData {
                grr,
                grz,
                gzz,
                grr_r,
                grr_z,
                grz_r,
                grz_z,
                gzz_r,
                gzz_z,
                s: seed,
                s_r: seed_r,
                s_z: seed_z,
                krr,
                krz,
                kzz,
                y,
                theta,
                radius,
                radius_deriv,
                radius_second_deriv,
            };

            surface_deriv[index] = horizon(system, [r, z]);
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
enum HorizonCallbackError {
    #[error("Coverged to zero")]
    CollapsedToZero,
}

struct HorizonCallback<'a> {
    mesh: &'a Mesh<2>,
}

impl<'a> SolverCallback<1, Scalar> for HorizonCallback<'a> {
    type Error = HorizonCallbackError;

    fn callback(
        &self,
        _surface: &Mesh<1>,
        input: SystemSlice<Scalar>,
        _output: SystemSlice<Scalar>,
        _iteration: usize,
    ) -> Result<(), Self::Error> {
        let radius = input.field(());

        let min_spacing = self.mesh.min_spacing();
        let collapsed = radius.iter().all(|r| r.abs() <= min_spacing);

        if collapsed {
            return Err(HorizonCallbackError::CollapsedToZero);
        }

        Ok(())
    }
}

fn position(radius: f64, theta: f64) -> [f64; 2] {
    let x = radius * theta.cos();
    let y = radius * theta.sin();

    [x.max(0.0), y.max(0.0)]
}
