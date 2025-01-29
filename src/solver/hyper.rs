use crate::kernel::{Kernels, RadiativeParams};
use aeon_geometry::{Face, IndexSpace};
use reborrow::{Reborrow, ReborrowMut};
use thiserror::Error;

use crate::{
    mesh::{Engine, Function, Mesh},
    solver::{Integrator, Method},
    system::{Pair, System, SystemConditions, SystemSlice, SystemSliceMut},
};

use super::SolverCallback;

// #[derive(Clone, Copy, Debug, Default)]
// pub enum Method {
//     #[default]
//     ForwardEuler,
//     RK4,
// }

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
    /// CFL factor for ficticuous time step.
    pub cfl: f64,
    /// If set, the relax solver uses larger time steps for
    /// vertices in less refined regions (subject to the CFL condition
    /// of course).
    pub adaptive: bool,

    integrator: Integrator,
}

impl Default for HyperRelaxSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperRelaxSolver {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-5,
            max_steps: 100000,
            dampening: 1.0,
            cfl: 0.1,
            adaptive: false,
            // visualize: None,
            integrator: Integrator::new(Method::RK4),
        }
    }

    pub fn solve<
        const N: usize,
        K: Kernels + Sync,
        C: SystemConditions<N> + Sync,
        F: Function<N, Input = C::System, Output = C::System> + SolverCallback<N> + Clone + Sync,
    >(
        &mut self,
        mesh: &mut Mesh<N>,
        order: K,
        conditions: C,
        deriv: F,
        mut result: SystemSliceMut<C::System>,
    ) -> Result<(), HyperRelaxError>
    where
        C::System: Default + Clone + Sync,
    {
        // Total number of degreees of freedom in the whole system
        let dimension = result.system().count() * mesh.num_nodes();

        assert!(result.len() == dimension);

        let system = (result.system().clone(), result.system().clone());

        // Allocate storage
        let mut data = vec![0.0; 2 * dimension].into_boxed_slice();
        // Compute minimum spacing and spacing per vertex.
        let min_spacing = mesh.min_spacing();

        let mut spacing_per_vertex = vec![min_spacing; mesh.num_nodes()];
        if self.adaptive {
            mesh.spacing_per_vertex(&mut spacing_per_vertex);
        }

        // Use CFL factor to compute time_step
        let time_step = self.cfl * min_spacing;

        // Fill initial guess
        {
            let (u, v) = data.split_at_mut(dimension);
            // u is initial guess
            mesh.copy_from_slice(
                SystemSliceMut::from_contiguous(u, result.system()),
                result.rb(),
            );
            // Let us assume that du/dt is initially zero
            mesh.copy_from_slice(
                SystemSliceMut::from_contiguous(v, result.system()),
                result.rb(),
            );
            for value in v.iter_mut() {
                *value *= self.dampening;
            }
        }

        for index in 0..self.max_steps {
            mesh.fill_boundary(
                order,
                FicticuousConditions {
                    dampening: self.dampening,
                    conditions: conditions.clone(),
                },
                SystemSliceMut::from_contiguous(&mut data, &system),
            );

            {
                let u = SystemSlice::from_contiguous(&mut data[..dimension], &system.0);

                mesh.copy_from_slice(result.rb_mut(), u.rb());

                mesh.apply(order, conditions.clone(), deriv.clone(), result.rb_mut());

                deriv.callback(mesh, u.rb(), result.rb(), index);
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

            self.integrator.step(
                mesh,
                order,
                FicticuousConditions {
                    dampening: self.dampening,
                    conditions: conditions.clone(),
                },
                FicticuousDerivs {
                    dampening: self.dampening,
                    function: &deriv,
                    spacing_per_vertex: &spacing_per_vertex,
                    min_spacing,
                },
                time_step,
                SystemSliceMut::from_contiguous(&mut data, &system),
            );

            if index == self.max_steps - 1 {
                log::error!(
                    "Hyperbolic relaxation failed to converge in {} steps.",
                    self.max_steps
                );
            }
        }

        // Copy solution back to system vector
        mesh.copy_from_slice(
            result.rb_mut(),
            SystemSlice::from_contiguous(&data[..dimension], &system.0),
        );
        mesh.fill_boundary(order, conditions, result.rb_mut());

        Ok(())
    }
}

#[derive(Clone)]
struct FicticuousConditions<C> {
    dampening: f64,
    conditions: C,
}

impl<const N: usize, C: SystemConditions<N>> SystemConditions<N> for FicticuousConditions<C> {
    type System = (C::System, C::System);

    fn parity(&self, label: <Self::System as System>::Label, face: Face<N>) -> bool {
        match label {
            Pair::First(label) => self.conditions.parity(label, face),
            Pair::Second(label) => self.conditions.parity(label, face),
        }
    }

    fn radiative(
        &self,
        label: <Self::System as System>::Label,
        position: [f64; N],
    ) -> RadiativeParams {
        match label {
            Pair::First(label) => self.conditions.radiative(label, position),
            Pair::Second(label) => {
                let mut result = self.conditions.radiative(label, position);
                result.target *= self.dampening;
                result
            }
        }
    }
}

#[derive(Clone)]
struct FicticuousDerivs<'a, const N: usize, F> {
    dampening: f64,
    function: &'a F,
    spacing_per_vertex: &'a [f64],
    min_spacing: f64,
}

impl<'a, const N: usize, S: System, F: Function<N, Input = S, Output = S>> Function<N>
    for FicticuousDerivs<'a, N, F>
{
    type Input = (S, S);
    type Output = (S, S);

    fn evaluate(
        &self,
        engine: impl Engine<N>,
        input: SystemSlice<Self::Input>,
        output: SystemSliceMut<Self::Output>,
    ) {
        let (uin, vin) = input.split_pair();
        let (mut uout, mut vout) = output.split_pair();

        // Find du/dt from the definition v = du/dt + η u
        for field in uin.system().enumerate() {
            let u = uin.field(field);
            let v = vin.field(field);

            let udest = uout.field_mut(field);

            for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                let index = engine.index_from_vertex(vertex);
                udest[index] = v[index] - u[index] * self.dampening;
            }
        }

        // dv/dt = c^2 Lu
        // TODO speed
        self.function.evaluate(&engine, uin, vout.rb_mut());

        // Use adaptive timestep
        let block_spacing = &self.spacing_per_vertex[engine.node_range()];

        for field in uout.system().enumerate() {
            let uout = uout.field_mut(field);
            for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                let index = engine.index_from_vertex(vertex);
                uout[index] *= block_spacing[index] / self.min_spacing;
            }
        }

        for field in vout.system().enumerate() {
            let vout = vout.field_mut(field);
            for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                let index = engine.index_from_vertex(vertex);
                vout[index] *= block_spacing[index] / self.min_spacing;
            }
        }
    }
}
