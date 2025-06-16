use crate::eqs::{HorizonData, horizon};
use crate::systems::{Field, Fields, Metric};
use aeon::solver::{HyperRelaxError, SolverCallback};
use aeon::{
    element::UniformInterpolate, kernel::Kernels, mesh::UnsafeThreadCache, prelude::*,
    solver::HyperRelaxSolver,
};

use core::f64;
use reborrow::Reborrow;
use std::convert::Infallible;
use thiserror::Error;

// Build new horizon surface
pub fn surface() -> Mesh<1> {
    Mesh::new(HORIZON_DOMAIN, 4, 2, FaceArray::splat(BoundaryClass::Ghost))
}

pub fn compute_position_from_radius(surface: &Mesh<1>, radius: &[f64], output: &mut [[f64; 2]]) {
    assert_eq!(surface.tree().domain(), HORIZON_DOMAIN);
    assert_eq!(surface.num_nodes(), output.len());

    for block in surface.blocks().indices() {
        let space = surface.block_space(block);
        let nodes = surface.block_nodes(block);

        let block_radius = &radius[nodes.clone()];
        let block_result = &mut output[nodes.clone()];

        for node in space.full_window() {
            let index = space.index_from_node(node);
            let [theta] = space.position(node);
            let radius = block_radius[index];

            block_result[index] = polar_to_cartesian(radius, theta);
        }
    }
}

const HORIZON_DOMAIN: Rectangle<1> = Rectangle {
    size: [f64::consts::PI / 2.0],
    origin: [0.0],
};

#[derive(Debug, Clone)]
pub enum HorizonStatus {
    ConvergedToZero,
    Converged,
}

#[derive(Debug, Error, Clone)]
pub enum HorizonError<A> {
    #[error("surface point not contained in mesh: ${0:?}")]
    SurfaceNotContained([f64; 2]),
    #[error("surface radial norm diverged")]
    NormDiverged,
    #[error("surface failed to converge in allotted number of steps")]
    ReachedMaxSteps,
    #[error("interpolation didn't converge")]
    InterpolateFailed,
    #[error("callback failed")]
    CallbackFailed(#[from] A),
}

impl<A> HorizonError<A> {
    fn from_infaillable(other: HorizonError<Infallible>) -> Self {
        match other {
            HorizonError::SurfaceNotContained(pos) => Self::SurfaceNotContained(pos),
            HorizonError::NormDiverged => Self::NormDiverged,
            HorizonError::ReachedMaxSteps => Self::ReachedMaxSteps,
            HorizonError::InterpolateFailed => Self::InterpolateFailed,
            HorizonError::CallbackFailed(_) => unreachable!(),
        }
    }
}

/// Implements a horizon finding algorithm in the manner of Rinne 2007. Functions with a
/// similar interface to `aeon::solvers::*`, where the object serves as a public interface
/// for setting configuration variables and as a memory cache.
pub struct ApparentHorizonFinder {
    pub solver: HyperRelaxSolver,

    surface_to_cell: Vec<CellId>,
    surface_position: Vec<[f64; 2]>,

    cache: UnsafeThreadCache<UniformInterpolate<2>>,
}

impl ApparentHorizonFinder {
    /// Constructs a new horizon finder with default solver settings.
    pub fn new() -> Self {
        Self {
            solver: HyperRelaxSolver::default(),

            surface_to_cell: Vec::new(),
            surface_position: Vec::new(),

            cache: UnsafeThreadCache::new(),
        }
    }

    pub fn _search<K: Kernels>(
        &mut self,
        mesh: &Mesh<2>,
        fields: SystemSlice<Fields>,
        order: K,
        surface: &mut Mesh<1>,
        radius: &mut [f64],
    ) -> Result<HorizonStatus, HorizonError<Infallible>>
    where
        K: Sync,
    {
        self.search_with_callback(mesh, fields, order, surface, (), radius)
    }

    /// Performs a horizon search on the given mesh, using the 1d surface mesh and radius
    /// vector for the search. This passes the callback into the underlying solver to
    /// allow for visualization of iterations, etc.
    pub fn search_with_callback<K: Kernels, Call: SolverCallback<1, Scalar>>(
        &mut self,
        mesh: &Mesh<2>,
        fields: SystemSlice<Fields>,
        order: K,
        surface: &mut Mesh<1>,
        mut callback: Call,
        radius: &mut [f64],
    ) -> Result<HorizonStatus, HorizonError<Call::Error>>
    where
        K: Sync,
        Call::Error: Send,
    {
        assert_eq!(fields.len(), mesh.num_nodes());
        assert_eq!(radius.len(), surface.num_nodes());
        assert_eq!(surface.tree().domain(), HORIZON_DOMAIN);
        assert_eq!(
            surface.boundary_classes(),
            FaceArray::splat(BoundaryClass::Ghost)
        );

        self.surface_to_cell.resize(surface.num_nodes(), CellId(0));
        self.surface_to_cell.fill(CellId(0));

        self.surface_position.resize(surface.num_nodes(), [0.0; 2]);
        self.surface_position.fill([0.0; 2]);

        // Find cell at origin
        let origin_cell = mesh.tree().cell_from_point([0.0, 0.0]);
        let origin_bounds = mesh.tree().bounds(origin_cell);

        // Compute stopping radius from this cell.
        let mut min_radius = mesh.min_spacing();

        for axis in 0..2 {
            if origin_bounds.size[axis] > min_radius {
                min_radius = origin_bounds.size[axis];
            }
        }

        // Run solver
        let result = self.solver.solve_with_callback(
            surface,
            order,
            HorizonRadialBoundary,
            HorizonCallback {
                inner: &mut callback,
                min_radius,
            },
            HorizonNullExpansion::<K> {
                surface_to_cell: &mut self.surface_to_cell,
                surface_position: &mut self.surface_position,
                mesh,
                fields,
                cache: &mut self.cache,
                _phantom: std::marker::PhantomData,
            },
            SystemSliceMut::from_scalar(radius),
        );

        match result {
            Ok(()) => Ok(HorizonStatus::Converged),
            Err(HyperRelaxError::CallbackFailed(HorizonCallbackError::CollapsedToZero)) => {
                Ok(HorizonStatus::ConvergedToZero)
            }
            Err(HyperRelaxError::CallbackFailed(HorizonCallbackError::Inner(err))) => {
                Err(HorizonError::CallbackFailed(err))
            }
            Err(HyperRelaxError::NormDiverged) => Err(HorizonError::NormDiverged),
            Err(HyperRelaxError::ReachedMaxSteps) => Err(HorizonError::ReachedMaxSteps),
            Err(HyperRelaxError::FunctionFailed(err)) => Err(HorizonError::from_infaillable(err)),
        }
    }
}

impl Clone for ApparentHorizonFinder {
    fn clone(&self) -> Self {
        Self {
            surface_to_cell: self.surface_to_cell.clone(),
            surface_position: self.surface_position.clone(),
            cache: UnsafeThreadCache::new(),
            solver: self.solver.clone(),
        }
    }
}

/// Symmetric boundary for radius function on surface.
#[derive(Clone)]
struct HorizonRadialBoundary;

impl SystemBoundaryConds<1> for HorizonRadialBoundary {
    type System = Scalar;

    fn kind(&self, _label: <Self::System as System>::Label, _face: Face<1>) -> BoundaryKind {
        BoundaryKind::Symmetric
    }
}

/// Expansion of null paths
struct HorizonNullExpansion<'a, K> {
    surface_to_cell: &'a mut [CellId],
    surface_position: &'a mut [[f64; 2]],
    mesh: &'a Mesh<2>,
    fields: SystemSlice<'a, Fields>,
    cache: &'a mut UnsafeThreadCache<UniformInterpolate<2>>,
    _phantom: std::marker::PhantomData<K>,
}

impl<'a, K: Kernels> Function<1> for HorizonNullExpansion<'a, K> {
    type Input = Scalar;
    type Output = Scalar;
    type Error = HorizonError<Infallible>;

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

                let point = polar_to_cartesian(radius, theta);

                if !self.mesh.tree().domain().contains(point) {
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

        let interpolate = unsafe { self.cache.get_or_default() };

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
            let mesh_active = mesh.tree().active_index_from_cell(mesh_cell).unwrap();
            let mesh_block = mesh.blocks().active_cell_block(mesh_active);

            let block_space = mesh.block_space(mesh_block);
            let block_nodes = mesh.block_nodes(mesh_block);

            let active_window = mesh.active_window(mesh_active);
            let active_bounds = mesh.tree().active_bounds(mesh_active);

            interpolate
                .build(cell_support, cell_support, active_bounds, [r, z])
                .map_err(|_err| HorizonError::InterpolateFailed)?;

            let block_fields = fields.slice(block_nodes.clone());

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
                    for (i, node) in active_window.iter().enumerate() {
                        scratch[i] = $field[block_space.index_from_node(node)];
                    }
                    let $output = interpolate.apply(&scratch);
                };
            }

            macro_rules! interpolate_derivative {
                ($output:ident, $field:ident, $axis:expr) => {
                    for (i, node) in active_window.iter().enumerate() {
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

            surface_deriv[index] = -horizon(system, [r, z]);
        }

        Ok(())
    }
}

#[derive(Debug, Error)]
enum HorizonCallbackError<I> {
    #[error("coverged to zero")]
    CollapsedToZero,
    #[error("inner error")]
    Inner(#[from] I),
}

struct HorizonCallback<'a, I> {
    inner: &'a mut I,
    min_radius: f64,
}

impl<'a, I: SolverCallback<1, Scalar>> SolverCallback<1, Scalar> for HorizonCallback<'a, I> {
    type Error = HorizonCallbackError<I::Error>;

    fn callback(
        &mut self,
        surface: &Mesh<1>,
        input: SystemSlice<Scalar>,
        output: SystemSlice<Scalar>,
        iteration: usize,
    ) -> Result<(), Self::Error> {
        let radius = input.field(());

        let collapsed = radius.iter().all(|r| r.abs() <= self.min_radius);

        self.inner.callback(surface, input, output, iteration)?;

        if collapsed {
            return Err(HorizonCallbackError::CollapsedToZero);
        }

        Ok(())
    }
}

fn polar_to_cartesian(radius: f64, theta: f64) -> [f64; 2] {
    let x = radius * theta.cos();
    let y = radius * theta.sin();

    [x.max(0.0), y.max(0.0)]
}

pub struct HorizonProjection;

impl Function<2> for HorizonProjection {
    type Input = Fields;
    type Output = Scalar;
    type Error = Infallible;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) -> Result<(), Self::Error> {
        let grr_f = input.field(Field::Metric(Metric::Grr));
        let grz_f = input.field(Field::Metric(Metric::Grz));
        let gzz_f = input.field(Field::Metric(Metric::Gzz));
        let s_f = input.field(Field::Metric(Metric::S));

        let krr_f = input.field(Field::Metric(Metric::Krr));
        let krz_f = input.field(Field::Metric(Metric::Krz));
        let kzz_f = input.field(Field::Metric(Metric::Kzz));
        let y_f = input.field(Field::Metric(Metric::Y));

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let pos = engine.position(vertex);
            let index = engine.index_from_vertex(vertex);

            macro_rules! derivatives {
                ($field:ident, $value:ident, $dr:ident, $dz:ident) => {
                    let $value = $field[index];
                    let $dr = engine.derivative($field, 0, vertex);
                    let $dz = engine.derivative($field, 1, vertex);
                };
            }

            // Metric
            derivatives!(grr_f, grr, grr_r, grr_z);
            derivatives!(gzz_f, gzz, gzz_r, gzz_z);
            derivatives!(grz_f, grz, grz_r, grz_z);

            // S
            derivatives!(s_f, s, s_r, s_z);

            // K
            let krr = krr_f[index];
            let krz = krz_f[index];
            let kzz = kzz_f[index];

            // Y
            let y = y_f[index];

            let radius = (pos[0].powi(2) + pos[1].powi(1)).sqrt();
            let theta = pos[1].atan2(pos[0]);

            let horizon_system = HorizonData {
                grr,
                grz,
                gzz,
                grr_r,
                grr_z,
                grz_r,
                grz_z,
                gzz_r,
                gzz_z,
                s,
                s_r,
                s_z,
                krr,
                krz,
                kzz,
                y,
                theta,
                radius,
                radius_deriv: 0.0,
                radius_second_deriv: 0.0,
            };

            output.field_mut(())[index] = horizon(horizon_system, pos);
        }

        Ok(())
    }
}
