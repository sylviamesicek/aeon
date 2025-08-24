use std::convert::Infallible;

use crate::IRef;
use crate::geometry::{Face, IndexSpace};
use crate::image::{ImageMut, ImageRef};
use crate::kernel::{BoundaryKind, DirichletParams, RadiativeParams, SystemBoundaryConds};
use crate::mesh::FunctionBorrowMut;
use datasize::DataSize;
use reborrow::{Reborrow, ReborrowMut};
use thiserror::Error;

use crate::{
    mesh::{Engine, Function, Mesh},
    solver::{Integrator, Method},
};

use super::SolverCallback;

/// Error which may be thrown during hyperbolic relaxation.
#[derive(Error, Debug)]
pub enum HyperRelaxError<A, B> {
    #[error("failed to relax below tolerance in allotted number of steps")]
    ReachedMaxSteps,
    #[error("norm diverged to NaN")]
    NormDiverged,
    #[error("function error")]
    FunctionFailed(A),
    #[error("callback error")]
    CallbackFailed(B),
}

/// A solver which implements the algorithm described in NRPyElliptic. This transforms the elliptic equation
/// 𝓛{u} = p, into the hyperbolic equation ∂ₜ²u + η∂ₜu = c² (𝓛{u} - p), where c is the speed of the wave, and η is
/// a dampening term that speeds up convergence.
#[derive(Clone, Debug, DataSize)]
pub struct HyperRelaxSolver {
    /// Error tolerance (relaxation stops once error goes below this value).
    pub tolerance: f64,
    /// Maximum number of relaxation steps to perform
    pub max_steps: usize,
    /// Dampening term η.
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
    /// Constructs a new `HyperRelaxSolver` with default settings.
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

    /// Solves a given elliptic system
    pub fn solve<const N: usize, C: SystemBoundaryConds<N> + Sync, F: Function<N> + Sync>(
        &mut self,
        mesh: &mut Mesh<N>,
        order: usize,
        conditions: C,
        deriv: F,
        result: ImageMut,
    ) -> Result<(), HyperRelaxError<F::Error, Infallible>>
    where
        F::Error: Send,
    {
        self.solve_with_callback(mesh, order, conditions, (), deriv, result)
    }

    pub fn solve_with_callback<
        const N: usize,
        C: SystemBoundaryConds<N> + Sync,
        F: Function<N> + Sync,
        Call: SolverCallback<N>,
    >(
        &mut self,
        mesh: &mut Mesh<N>,
        order: usize,
        conditions: C,
        mut callback: Call,
        mut deriv: F,
        mut result: ImageMut,
    ) -> Result<(), HyperRelaxError<F::Error, Call::Error>>
    where
        F::Error: Send,
        Call::Error: Send,
    {
        assert_eq!(result.num_nodes(), mesh.num_nodes());
        // Total number of degreees of freedom in the whole system
        let dimension = result.num_channels() * mesh.num_nodes();
        let num_channels = result.num_channels();

        // assert!(result.len() == dimension);

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
            mesh.copy_from_slice(ImageMut::from_storage(u, num_channels), result.rb());
            // Let us assume that du/dt is initially zero
            mesh.copy_from_slice(ImageMut::from_storage(v, num_channels), result.rb());
            for value in v.iter_mut() {
                *value *= self.dampening;
            }
        }

        for index in 0..self.max_steps {
            mesh.fill_boundary(
                order,
                FicticuousBoundaryConds {
                    dampening: self.dampening,
                    conditions: conditions.clone(),
                    channels: num_channels,
                },
                ImageMut::from_storage(&mut data, 2 * num_channels),
            );

            {
                let u = ImageRef::from_storage(&data[..dimension], num_channels);
                mesh.copy_from_slice(result.rb_mut(), u.rb());
                mesh.apply(
                    order,
                    conditions.clone(),
                    FunctionBorrowMut(&mut deriv),
                    result.rb_mut(),
                )
                .map_err(|err| HyperRelaxError::FunctionFailed(err))?;
                callback
                    .callback(mesh, u.rb(), result.rb(), index)
                    .map_err(|err| HyperRelaxError::CallbackFailed(err))?;
            }

            let norm = mesh.l2_norm_system(result.rb());

            if !norm.is_finite() || norm >= 1e60 {
                return Err(HyperRelaxError::NormDiverged);
            }

            if index % 100 == 0 {
                log::trace!("Relaxed {}k steps, norm: {:.5e}", index / 100, norm);
            }

            if norm <= self.tolerance {
                log::trace!("Converged in {} steps.", index);

                // Copy solution back to system vector
                mesh.copy_from_slice(
                    result.rb_mut(),
                    ImageRef::from_storage(&data[..dimension], num_channels),
                );
                mesh.fill_boundary(order, conditions, result.rb_mut());

                return Ok(());
            }

            self.integrator
                .step(
                    mesh,
                    order,
                    FicticuousBoundaryConds {
                        dampening: self.dampening,
                        conditions: conditions.clone(),
                        channels: num_channels,
                    },
                    FicticuousDerivs {
                        dampening: self.dampening,
                        function: &deriv,
                        spacing_per_vertex: &spacing_per_vertex,
                        min_spacing,
                    },
                    time_step,
                    ImageMut::from_storage(&mut data, 2 * num_channels),
                )
                .map_err(|err| HyperRelaxError::FunctionFailed(err))?;

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
            ImageRef::from_storage(&data[..dimension], num_channels),
        );
        mesh.fill_boundary(order, conditions, result.rb_mut());

        Err(HyperRelaxError::ReachedMaxSteps)
    }
}

#[derive(Clone)]
struct FicticuousBoundaryConds<C> {
    dampening: f64,
    conditions: C,
    channels: usize,
}

impl<const N: usize, C: SystemBoundaryConds<N>> SystemBoundaryConds<N>
    for FicticuousBoundaryConds<C>
{
    fn kind(&self, channel: usize, face: Face<N>) -> BoundaryKind {
        let boundary_kind: BoundaryKind = self.conditions.kind(channel % self.channels, face);

        match boundary_kind {
            BoundaryKind::Symmetric => BoundaryKind::Symmetric,
            BoundaryKind::AntiSymmetric => BoundaryKind::AntiSymmetric,
            BoundaryKind::Custom => BoundaryKind::Custom,
            BoundaryKind::Radiative => BoundaryKind::Radiative,
            BoundaryKind::Free => BoundaryKind::Free,
            BoundaryKind::StrongDirichlet | BoundaryKind::WeakDirichlet => {
                BoundaryKind::WeakDirichlet
            }
        }
    }

    fn radiative(&self, channel: usize, position: [f64; N]) -> RadiativeParams {
        let mut result = self.conditions.radiative(channel % self.channels, position);
        if channel >= self.channels {
            result.target *= self.dampening;
        }
        result
    }

    fn dirichlet(&self, channel: usize, position: [f64; N]) -> DirichletParams {
        let mut result = self.conditions.dirichlet(channel % self.channels, position);
        if channel >= self.channels {
            result.target *= self.dampening;
        }
        result
    }
}

#[derive(Clone)]
struct FicticuousDerivs<'a, const N: usize, F> {
    dampening: f64,
    function: &'a F,
    spacing_per_vertex: &'a [f64],
    min_spacing: f64,
}

impl<const N: usize, F: Function<N>> Function<N> for FicticuousDerivs<'_, N, F> {
    type Error = F::Error;

    fn evaluate(
        &self,
        engine: impl Engine<N>,
        input: ImageRef,
        mut output: ImageMut,
    ) -> Result<(), F::Error> {
        assert_eq!(input.num_channels(), output.num_channels());
        let num_channels = output.num_channels();
        let (uin, vin) = input.split_channels(num_channels / 2);
        let (mut uout, mut vout) = output.rb_mut().split_channels(num_channels / 2);

        // Find du/dt from the definition v = du/dt + η u
        for field in uin.channels() {
            let u = uin.channel(field);
            let v = vin.channel(field);

            let udest = uout.channel_mut(field);

            for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                let index = engine.index_from_vertex(vertex);
                udest[index] = v[index] - u[index] * self.dampening;
            }
        }

        // dv/dt = c^2 Lu
        // TODO speed
        self.function.evaluate(IRef(&engine), uin, vout.rb_mut())?;

        // Use adaptive timestep
        let block_spacing = &self.spacing_per_vertex[engine.node_range()];

        for field in uout.channels() {
            let uout = uout.channel_mut(field);
            for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                let index = engine.index_from_vertex(vertex);
                uout[index] *= block_spacing[index] / self.min_spacing;
            }
        }

        for field in vout.channels() {
            let vout = vout.channel_mut(field);
            for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                let index = engine.index_from_vertex(vertex);
                vout[index] *= block_spacing[index] / self.min_spacing;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        geometry::{FaceArray, HyperBox},
        kernel::{BoundaryClass, DirichletParams},
    };

    use super::*;
    use crate::{kernel::BoundaryKind, mesh::Projection};
    use std::{convert::Infallible, f64::consts};

    #[derive(Clone)]
    struct _PoissonConditions;

    impl SystemBoundaryConds<2> for _PoissonConditions {
        fn kind(&self, _channel: usize, _face: Face<2>) -> BoundaryKind {
            BoundaryKind::StrongDirichlet
        }

        fn dirichlet(&self, _channel: usize, _position: [f64; 2]) -> DirichletParams {
            DirichletParams {
                target: 0.0,
                strength: 1.0,
            }
        }
    }

    #[derive(Clone)]
    pub struct PoissonSolution;

    impl Projection<2> for PoissonSolution {
        fn project(&self, [x, y]: [f64; 2]) -> f64 {
            (2.0 * consts::PI * x).sin() * (2.0 * consts::PI * y).sin()
        }
    }

    #[derive(Clone)]
    pub struct PoissonEquation;

    impl Function<2> for PoissonEquation {
        type Error = Infallible;

        fn evaluate(
            &self,
            engine: impl Engine<2>,
            input: ImageRef,
            mut output: ImageMut,
        ) -> Result<(), Infallible> {
            let input = input.channel(0);
            let output = output.channel_mut(0);

            for vertex in IndexSpace::new(engine.vertex_size()).iter() {
                let index = engine.index_from_vertex(vertex);
                let [x, y] = engine.position(vertex);

                let laplacian = engine.second_derivative(input, 0, vertex)
                    + engine.second_derivative(input, 1, vertex);
                let source = -8.0
                    * consts::PI
                    * consts::PI
                    * (2.0 * consts::PI * x).sin()
                    * (2.0 * consts::PI * y).sin();

                output[index] = laplacian - source;
            }

            Ok(())
        }
    }

    #[test]
    fn poisson() {
        let mut mesh = Mesh::new(
            HyperBox::from_aabb([0.0, 0.0], [1.0, 1.0]),
            4,
            2,
            FaceArray::splat(BoundaryClass::Ghost),
        );
        // Perform refinement
        mesh.refine_global();
        mesh.refine_global();

        // Write solution vector
        let mut solution = vec![0.0; mesh.num_nodes()];
        mesh.project(4, PoissonSolution, &mut solution);

        let mut solver = HyperRelaxSolver::new();
        solver.adaptive = true;
        solver.cfl = 0.5;
        solver.dampening = 0.4;
        solver.max_steps = 1_000_000;
        solver.tolerance = 1e-4;

        // loop {
        // if mesh.max_level() > 11 {
        //     panic!("Poisson mesh solver exceeded max levels");
        // }

        // let mut result = vec![1.0; mesh.num_nodes()];

        // solver
        //     .solve(
        //         &mut mesh,
        //         Order::<4>,
        //         PoissonConditions,
        //         PoissonEquation,
        //         (&mut result).into(),
        //     )
        //     .unwrap();

        // mesh.flag_wavelets::<Scalar>(4, 0.0, 1e-4, result.as_slice().into());
        // mesh.balance_flags();

        // if mesh.requires_regridding() {
        //     mesh.regrid();
        // } else {
        //     return;
        // }
        // }
    }
}
