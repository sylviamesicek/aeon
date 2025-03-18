use crate::geometry::{Face, IndexSpace};
use crate::kernel::{BoundaryKind, DirichletParams, Kernels, RadiativeParams};
use reborrow::{Reborrow, ReborrowMut};
use thiserror::Error;

use crate::{
    mesh::{Engine, Function, Mesh},
    solver::{Integrator, Method},
    system::{Pair, System, SystemBoundaryConds, SystemSlice, SystemSliceMut},
};

use super::SolverCallback;

/// Error which may be thrown during hyperbolic relaxation.
#[derive(Error, Debug)]
pub enum HyperRelaxError {
    #[error("failed to relax below tolerance in allotted number of steps")]
    FailedToMeetTolerance,
    #[error("norm diverged to NaN")]
    Diverged,
    #[error("failed to create and store visualizations of each iteration")]
    VisualizeFailed,
}

/// A solver which implements the algorithm described in NRPyElliptic. This transforms the elliptic equation
/// 𝓛{u} = p, into the hyperbolic equation ∂ₜ²u + η∂ₜu = c² (𝓛{u} - p), where c is the speed of the wave, and η is
/// a dampening term that speeds up convergence.
#[derive(Clone, Debug)]
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
    pub fn solve<
        const N: usize,
        K: Kernels + Sync,
        C: SystemBoundaryConds<N> + Sync,
        F: Function<N, Input = C::System, Output = C::System> + Clone + Sync,
    >(
        &mut self,
        mesh: &mut Mesh<N>,
        order: K,
        conditions: C,
        deriv: F,
        result: SystemSliceMut<C::System>,
    ) -> Result<(), HyperRelaxError>
    where
        C::System: Default + Clone + Sync,
    {
        self.solve_with_callback(mesh, order, conditions, deriv, (), result)
    }

    pub fn solve_with_callback<
        const N: usize,
        K: Kernels + Sync,
        C: SystemBoundaryConds<N> + Sync,
        F: Function<N, Input = C::System, Output = C::System> + Clone + Sync,
        Call: SolverCallback<N, C::System> + Sync,
    >(
        &mut self,
        mesh: &mut Mesh<N>,
        order: K,
        conditions: C,
        deriv: F,
        callback: Call,
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
                FicticuousBoundaryConds {
                    dampening: self.dampening,
                    conditions: conditions.clone(),
                },
                SystemSliceMut::from_contiguous(&mut data, &system),
            );

            {
                let u = SystemSlice::from_contiguous(&mut data[..dimension], &system.0);
                mesh.copy_from_slice(result.rb_mut(), u.rb());
                mesh.apply(order, conditions.clone(), deriv.clone(), result.rb_mut());
                callback.callback(mesh, u.rb(), result.rb(), index);
            }

            let norm = mesh.l2_norm(result.rb());

            if !norm.is_finite() || norm >= 1e60 {
                return Err(HyperRelaxError::Diverged);
            }

            if index % 1000 == 0 {
                log::trace!("Relaxed {}k steps, norm: {:.5e}", index / 1000, norm);
            }

            if norm <= self.tolerance {
                log::trace!("Converged in {} steps.", index);
                break;
            }

            self.integrator.step(
                mesh,
                order,
                FicticuousBoundaryConds {
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
struct FicticuousBoundaryConds<C> {
    dampening: f64,
    conditions: C,
}

impl<const N: usize, C: SystemBoundaryConds<N>> SystemBoundaryConds<N>
    for FicticuousBoundaryConds<C>
{
    type System = (C::System, C::System);

    fn kind(&self, label: <Self::System as System>::Label, face: Face<N>) -> BoundaryKind {
        let label = match label {
            Pair::First(label) => label,
            Pair::Second(label) => label,
        };

        let boundary_kind = self.conditions.kind(label, face);

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

    fn dirichlet(
        &self,
        label: <Self::System as System>::Label,
        position: [f64; N],
    ) -> DirichletParams {
        match label {
            Pair::First(label) => self.conditions.dirichlet(label, position),
            Pair::Second(label) => {
                let mut params = self.conditions.dirichlet(label, position);
                params.target *= self.dampening;
                params
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

#[cfg(test)]
mod tests {
    use crate::{
        geometry::Rectangle,
        kernel::{BoundaryClass, DirichletParams},
    };

    use super::*;
    use crate::{
        kernel::BoundaryKind,
        mesh::Projection,
        system::{Scalar, SystemBoundaryConds},
    };
    use std::f64::consts;

    #[derive(Clone)]
    struct PoissonConditions;

    impl SystemBoundaryConds<2> for PoissonConditions {
        type System = Scalar;

        fn kind(&self, _label: <Self::System as System>::Label, _face: Face<2>) -> BoundaryKind {
            BoundaryKind::StrongDirichlet
        }

        fn dirichlet(
            &self,
            _label: <Self::System as System>::Label,
            _position: [f64; 2],
        ) -> DirichletParams {
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
        type Input = Scalar;
        type Output = Scalar;

        fn evaluate(
            &self,
            engine: impl Engine<2>,
            input: SystemSlice<Self::Input>,
            output: SystemSliceMut<Self::Output>,
        ) {
            let input = input.into_scalar();
            let output = output.into_scalar();

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
        }
    }

    #[test]
    fn poisson() {
        let mut mesh = Mesh::new(Rectangle::from_aabb([0.0, 0.0], [1.0, 1.0]), 4, 2);
        // Set boundary ghost flags.
        mesh.set_boundary_classes(BoundaryClass::OneSided);
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
