use aeon::{
    basis::{Kernels, RadiativeParams},
    elliptic::HyperRelaxSolver,
    fd::{Mesh, SystemCondition},
    prelude::*,
};

use sharedaxi::{Solver, Source};

/// Quadrant upon which all simulations run.
#[derive(Clone)]
pub struct Quadrant;

impl Boundary<2> for Quadrant {
    fn kind(&self, face: Face<2>) -> BoundaryKind {
        if !face.side {
            BoundaryKind::Parity
        } else {
            BoundaryKind::Radiative
        }
    }
}

/// Initial data that Brill waves affects.
#[derive(Clone, Copy, SystemLabel)]
pub enum Rinne {
    Conformal,
    Seed,
}

/// Initial data is Garfinkle's variables.
#[derive(Clone, Copy, SystemLabel)]
pub enum Garfinkle {
    Psi,
    Seed,
}

/// Boundary Conditions for Garfinkle variables.
#[derive(Clone)]
pub struct GarfinkleConditions;

impl Conditions<2> for GarfinkleConditions {
    type System = GarfinkleSystem;

    fn parity(&self, field: Garfinkle, face: Face<2>) -> bool {
        match field {
            Garfinkle::Psi => [true, true][face.axis],
            Garfinkle::Seed => [false, true][face.axis],
        }
    }

    fn radiative(&self, field: Garfinkle, _position: [f64; 2], _spacing: f64) -> RadiativeParams {
        match field {
            Garfinkle::Psi => RadiativeParams::lightlike(1.0),
            Garfinkle::Seed => RadiativeParams::lightlike(0.0),
        }
    }
}

#[derive(Clone)]
pub struct SeedProjection<'a>(&'a [Source]);

impl<'a> Projection<2> for SeedProjection<'a> {
    fn project(&self, [rho, z]: [f64; 2]) -> f64 {
        let mut result = 0.0;
        let rho2 = rho * rho;
        let z2 = z * z;

        for source in self.0 {
            match source {
                Source::Brill(brill) => {
                    let srho2 = brill.sigma.0 * brill.sigma.0;
                    let sz2 = brill.sigma.1 * brill.sigma.1;

                    result += rho * brill.amplitude * (-rho2 / srho2 - z2 / sz2).exp()
                }
            }
        }

        result
    }
}

#[derive(Clone)]
pub struct Hamiltonian<'a> {
    seed: &'a [f64],
}

impl<'a> Function<2> for Hamiltonian<'a> {
    type Input = Scalar;
    type Output = Scalar;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: SystemSlice<Self::Input>,
        output: SystemSliceMut<Self::Output>,
    ) {
        let seed = &self.seed[engine.node_range()];
        let psi = input.into_scalar();

        let dest = output.into_scalar();

        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);

            let [rho, _] = engine.position(vertex);

            let psi_r = engine.derivative(psi, 0, vertex);
            let psi_rr = engine.second_derivative(psi, 0, vertex);
            let psi_zz = engine.second_derivative(psi, 1, vertex);

            let seed_r = engine.derivative(seed, 0, vertex);
            let seed_rr = engine.second_derivative(seed, 0, vertex);
            let seed_zz = engine.second_derivative(seed, 1, vertex);

            let laplacian = if rho.abs() <= 10e-10 {
                2.0 * psi_rr + psi_zz
            } else {
                psi_rr + psi_r / rho + psi_zz
            };

            dest[index] =
                laplacian + psi[index] / 4.0 * (rho * seed_rr + 2.0 * seed_r + rho * seed_zz);
        }
    }
}

#[derive(Clone)]
pub struct BrillFromGarfinkle;

impl Function<2> for BrillFromGarfinkle {
    type Input = GarfinkleSystem;
    type Output = RinneSystem;

    fn evaluate(
        &self,
        engine: impl Engine<2>,
        input: SystemSlice<Self::Input>,
        mut output: SystemSliceMut<Self::Output>,
    ) {
        for vertex in IndexSpace::new(engine.vertex_size()).iter() {
            let index = engine.index_from_vertex(vertex);
            let [rho, _] = engine.position(vertex);

            let psi = input.field(Garfinkle::Psi)[index];
            let seed = input.field(Garfinkle::Seed)[index];

            output.field_mut(Rinne::Conformal)[index] = psi.powi(4) * (2.0 * rho * seed).exp();
            output.field_mut(Rinne::Seed)[index] = -seed;
        }
    }
}

pub fn solve_order<const ORDER: usize>(
    order: Order<ORDER>,
    mesh: &mut Mesh<2>,
    solver_con: &Solver,
    sources: &[Source],
) -> anyhow::Result<SystemVec<RinneSystem>>
where
    Order<ORDER>: Kernels,
{
    // Retrieve number of nodes in current version of mesh.
    let num_nodes = mesh.num_nodes();

    // Back garfinkle variables into single variable.
    let mut garfinkle = vec![0.0; num_nodes * 2];
    let (psi, seed) = garfinkle.split_at_mut(num_nodes);

    // Compute seed values.
    log::trace!("Filling Seed Function");
    mesh.project_scalar(order, Quadrant, SeedProjection(&sources), seed);
    mesh.fill_boundary_scalar(
        order,
        Quadrant,
        SystemCondition::new(GarfinkleConditions, Garfinkle::Seed),
        seed,
    );

    // Initial Guess for Psi
    psi.fill(1.0);

    log::trace!("Running Hyperbolic Relaxation");

    let mut solver = HyperRelaxSolver::new();
    solver.dampening = solver_con.dampening;
    solver.max_steps = solver_con.max_steps;
    solver.tolerance = solver_con.tolerance;
    solver.cfl = solver_con.cfl;

    solver.solve_scalar(
        mesh,
        order,
        Quadrant,
        SystemCondition::new(GarfinkleConditions, Garfinkle::Psi),
        Hamiltonian { seed },
        psi,
    )?;

    mesh.fill_boundary_scalar(
        order,
        Quadrant,
        SystemCondition::new(GarfinkleConditions, Garfinkle::Psi),
        psi,
    );

    let mut rinne = SystemVec::with_length(num_nodes, RinneSystem);

    mesh.evaluate(
        order,
        Quadrant,
        BrillFromGarfinkle,
        SystemSlice::from_contiguous(&garfinkle, &GarfinkleSystem),
        rinne.as_mut_slice(),
    );

    // let mut hamiltonian = SystemVec::with_length(num_nodes, Scalar);
    // mesh.evaluate_scalar(order, Quadrant, Hamiltonian { seed }, &psi, hamiltonian);

    Ok(rinne)
}
