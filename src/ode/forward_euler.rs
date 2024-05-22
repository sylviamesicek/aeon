use super::Ode;

/// Standard forward Euler Integrator.
#[derive(Clone, Debug, Default)]
pub struct ForwardEuler {
    k1: Vec<f64>,
}

impl ForwardEuler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn step<Problem: Ode>(
        &mut self,
        h: f64,
        derivs: &mut Problem,
        system: &[f64],
        update: &mut [f64],
    ) {
        assert!(system.len() == update.len());

        let dim = system.len();
        self.k1.resize(dim, 0.0);
        // K1
        update.copy_from_slice(system);
        derivs.preprocess(update);
        derivs.derivative(update, &mut self.k1);

        // Compute total step
        for i in 0..dim {
            update[i] = h * self.k1[i];
        }
    }
}
