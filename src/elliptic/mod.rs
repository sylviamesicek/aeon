use aeon_basis::{Boundary, Condition, Kernels, RadiativeParams};
use aeon_geometry::IndexSpace;
use reborrow::{Reborrow, ReborrowMut};
use thiserror::Error;

use crate::{
    fd::{Conditions, Engine, Function, Mesh, ScalarConditions, SystemCheckpoint},
    ode::{Ode, Rk4},
    prelude::Face,
    system::{Empty, Scalar, System, SystemSlice, SystemSliceMut},
};

#[derive(Error, Debug)]
pub enum HyperRelaxError {
    #[error("failed to relax below tolerance in allotted number of steps")]
    FailedToMeetTolerance,
    #[error("norm diverged to NaN")]
    Diverged,
    #[error("failed to create and store visualizations of each iteration")]
    VisualizeFailed,
}

#[derive(Clone, Debug)]
pub struct HyperRelaxSolver {
    /// Error tolerance (relaxation stops once error goes below this value).
    pub tolerance: f64,
    /// Maximum number of relaxation steps to perform
    pub max_steps: usize,
    pub dampening: f64,
    pub cfl: f64,

    // pub visualize: Option<HyperRelaxVisualize>,
    integrator: Rk4,
}

impl Default for HyperRelaxSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperRelaxSolver {
    pub fn new() -> Self {
        Self {
            tolerance: 10e-5,
            max_steps: 100000,
            dampening: 1.0,
            cfl: 0.1,
            // visualize: None,
            integrator: Rk4::new(),
        }
    }

    pub fn solve<
        const N: usize,
        K: Kernels + Sync,
        B: Boundary<N> + Sync,
        C: Conditions<N> + Sync,
        F: Function<N, Input = C::System, Output = C::System> + Clone + Sync,
    >(
        &mut self,
        mesh: &mut Mesh<N>,
        order: K,
        boundary: B,
        conditions: C,
        deriv: F,
        mut result: SystemSliceMut<C::System>,
    ) -> Result<(), HyperRelaxError>
    where
        C::System: Default + Clone + Sync,
    {
        // Total number of degreees of freedom in the whole system
        let system = result.system().clone();
        let dimension = system.count() * mesh.num_nodes();

        // Allocate storage
        let mut data = vec![0.0; 2 * dimension].into_boxed_slice();
        let mut update = vec![0.0; 2 * dimension].into_boxed_slice();

        // Fill initial guess
        {
            let (u, v) = data.split_at_mut(dimension);
            let mut usys = SystemSliceMut::<C::System>::from_contiguous(u, &system);
            let mut vsys = SystemSliceMut::<C::System>::from_contiguous(v, &system);

            for field in system.enumerate() {
                usys.field_mut(field).copy_from_slice(result.field(field))
            }

            // Let us assume that du/dt is initially zero
            for field in system.enumerate() {
                vsys.field_mut(field).copy_from_slice(result.field(field));
                vsys.field_mut(field)
                    .iter_mut()
                    .for_each(|f| *f *= self.dampening);
            }
        }

        let min_spacing: f64 = mesh.min_spacing();
        let time_step = self.cfl * min_spacing;

        for index in 0..self.max_steps {
            {
                let mut u = SystemSliceMut::from_contiguous(&mut data[..dimension], &system);

                mesh.fill_boundary(order, boundary.clone(), conditions.clone(), u.rb_mut());
                mesh.evaluate(
                    order,
                    boundary.clone(),
                    deriv.clone(),
                    u.rb(),
                    result.rb_mut(),
                );

                mesh.weak_boundary(
                    order,
                    boundary.clone(),
                    UConditions {
                        conditions: conditions.clone(),
                    },
                    u.rb(),
                    result.rb_mut(),
                );

                let mut systems = SystemCheckpoint::default();
                systems.save_system(result.rb());

                if mesh
                    .export_vtu(
                        format!("output/debug/velax{index}.vtu"),
                        &systems,
                        crate::fd::ExportVtuConfig {
                            title: "debug".to_string(),
                            ghost: false,
                        },
                    )
                    .is_err()
                {
                    return Err(HyperRelaxError::VisualizeFailed);
                }

                // deriv.callback(
                //     mesh,
                //     SystemSlice::from_contiguous(&data[..dimension], system.system()),
                //     output,
                // );
            }

            let norm = mesh.l2_norm(result.rb());

            if !norm.is_finite() || norm >= 1e60 {
                return Err(HyperRelaxError::Diverged);
            }

            if index % 1000 == 0 {
                log::trace!("Relaxed {}k steps, norm: {:.5e}", index / 1000, norm);
            }

            if norm <= self.tolerance {
                log::trace!(
                    "Hyperbolic Relaxation converged with error {:.5e} in {} steps.",
                    self.tolerance,
                    index
                );
                break;
            }

            let mut ode = FictitiousOde {
                mesh,
                dimension,
                dampening: self.dampening,
                order,
                boundary: boundary.clone(),
                conditions: conditions.clone(),
                deriv: &deriv,
                system: system.clone(),
            };

            // Take step
            self.integrator
                .step(time_step, &mut ode, &data, &mut update);

            for i in 0..data.len() {
                data[i] += update[i];
            }

            if index == self.max_steps - 1 {
                log::error!(
                    "Hyperbolic relaxation failed to converge in {} steps.",
                    self.max_steps
                );
            }
        }

        // Copy solution back to system vector
        let (u, _v) = data.split_at_mut(dimension);
        let usys = SystemSlice::from_contiguous(u, &system);

        for field in system.enumerate() {
            result.field_mut(field).clone_from_slice(usys.field(field))
        }

        Ok(())
    }

    pub fn solve_scalar<
        const N: usize,
        K: Kernels + Sync,
        B: Boundary<N> + Sync,
        C: Condition<N> + Sync,
        F: Function<N, Input = Scalar, Output = Scalar> + Clone + Sync,
    >(
        &mut self,
        mesh: &mut Mesh<N>,
        order: K,
        boundary: B,
        condition: C,
        deriv: F,
        result: &mut [f64],
    ) -> Result<(), HyperRelaxError> {
        self.solve(
            mesh,
            order,
            boundary,
            ScalarConditions(condition),
            deriv,
            result.into(),
        )
    }
}

#[derive(Clone)]
struct UConditions<C> {
    conditions: C,
}

impl<const N: usize, I: Conditions<N>> Conditions<N> for UConditions<I> {
    type System = I::System;

    fn parity(&self, field: <Self::System as System>::Label, face: Face<N>) -> bool {
        self.conditions.parity(field, face)
    }

    fn radiative(
        &self,
        field: <Self::System as System>::Label,
        position: [f64; N],
        spacing: f64,
    ) -> RadiativeParams {
        self.conditions.radiative(field, position, spacing)
    }
}

/// Wraps the du/dt = v - u * η operation.
#[derive(Clone)]
struct UDerivs<'a, const N: usize, S> {
    dampening: f64,
    u: SystemSlice<'a, S>,
    v: SystemSlice<'a, S>,
}

impl<'a, const N: usize, S: System + Clone> Function<N> for UDerivs<'a, N, S> {
    type Input = Empty;
    type Output = S;

    fn evaluate(
        &self,
        engine: impl Engine<N>,
        _input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        let node_range = engine.node_range();
        let system = output.system().clone();

        for field in system.enumerate() {
            let u = &self.u.field(field)[node_range.clone()];
            let v = &self.v.field(field)[node_range.clone()];

            let dest = output.field_mut(field);

            for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                let index = engine.index_from_vertex(vertex);
                dest[index] = v[index] - u[index] * self.dampening
            }
        }
    }
}

/// Provides boundary conditions for v. All strong boundary condition commute with the time derivative
/// and radiative boundary conditions simply need to be multipled by the dampening.
#[derive(Clone)]
struct VConditions<C> {
    dampening: f64,
    conditions: C,
}

impl<const N: usize, I: Conditions<N>> Conditions<N> for VConditions<I> {
    type System = I::System;

    fn parity(&self, field: <Self::System as System>::Label, face: Face<N>) -> bool {
        self.conditions.parity(field, face)
    }

    fn radiative(
        &self,
        field: <Self::System as System>::Label,
        position: [f64; N],
        spacing: f64,
    ) -> RadiativeParams {
        let mut params = self.conditions.radiative(field, position, spacing);
        params.target *= self.dampening;
        params
    }
}

/// Wraps the c^2 * L{u(x)} operation.
#[derive(Clone)]
struct VDerivs<'a, const N: usize, F> {
    function: &'a F,
}

impl<'a, const N: usize, S: System, F: Function<N, Input = S, Output = S>> Function<N>
    for VDerivs<'a, N, F>
{
    type Input = S;
    type Output = S;

    fn evaluate(
        &self,
        engine: impl Engine<N>,
        input: SystemSlice<Self::Input>,
        output: SystemSliceMut<Self::Output>,
    ) {
        self.function.evaluate(engine, input, output);
    }
}

struct FictitiousOde<
    'a,
    const N: usize,
    K: Kernels,
    B: Boundary<N>,
    C: Conditions<N>,
    F: Function<N, Input = C::System, Output = C::System>,
> where
    C::System: Clone,
{
    mesh: &'a mut Mesh<N>,
    dimension: usize,
    dampening: f64,

    order: K,
    boundary: B,
    conditions: C,
    deriv: &'a F,

    system: C::System,
}

impl<
        'a,
        const N: usize,
        K: Kernels + Sync,
        B: Boundary<N> + Sync,
        C: Conditions<N> + Sync,
        F: Function<N, Input = C::System, Output = C::System> + Sync,
    > Ode for FictitiousOde<'a, N, K, B, C, F>
where
    C::System: Clone + Sync,
{
    fn dim(&self) -> usize {
        2 * self.dimension
    }

    fn preprocess(&mut self, system: &mut [f64]) {
        let (u, v) = system.split_at_mut(self.dimension);

        self.mesh.fill_boundary(
            self.order,
            self.boundary.clone(),
            UConditions {
                conditions: self.conditions.clone(),
            },
            SystemSliceMut::from_contiguous(u, &self.system),
        );

        self.mesh.fill_boundary(
            self.order,
            self.boundary.clone(),
            VConditions {
                conditions: self.conditions.clone(),
                dampening: self.dampening,
            },
            SystemSliceMut::from_contiguous(v, &self.system),
        );
    }

    fn derivative(&mut self, system: &[f64], result: &mut [f64]) {
        let (udata, vdata) = system.split_at(self.dimension);
        let u = SystemSlice::from_contiguous(udata, &self.system);
        let v = SystemSlice::from_contiguous(vdata, &self.system);

        let (udata, vdata) = result.split_at_mut(self.dimension);
        let mut dudt = SystemSliceMut::from_contiguous(udata, &self.system);
        let mut dvdt = SystemSliceMut::from_contiguous(vdata, &self.system);

        // Compute derivatives

        // Find du/dt from the definition v = du/dt + η u
        self.mesh.evaluate(
            self.order,
            self.boundary.clone(),
            UDerivs {
                dampening: self.dampening,
                u: u.rb(),
                v: v.rb(),
            },
            SystemSlice::empty(),
            dudt.rb_mut(),
        );

        // dv/dt = c^2 Lu
        self.mesh.evaluate(
            self.order,
            self.boundary.clone(),
            VDerivs {
                function: self.deriv,
            },
            u.rb(),
            dvdt.rb_mut(),
        );

        // Apply Outer boundary conditions
        self.mesh.weak_boundary(
            self.order,
            self.boundary.clone(),
            UConditions {
                conditions: self.conditions.clone(),
            },
            u,
            dudt,
        );

        self.mesh.weak_boundary(
            self.order,
            self.boundary.clone(),
            VConditions {
                dampening: self.dampening,
                conditions: self.conditions.clone(),
            },
            v,
            dvdt,
        );
    }
}
