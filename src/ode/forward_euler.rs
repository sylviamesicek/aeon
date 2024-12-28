use super::Ode;

/// Standard forward Euler Integrator.
#[derive(Clone, Debug, Default)]
pub struct ForwardEuler {
    tmp: Vec<f64>,
}

impl ForwardEuler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn step<Problem: Ode>(&mut self, h: f64, derivs: &mut Problem, system: &mut [f64]) {
        let dim = system.len();
        assert!(system.len() == dim);

        self.tmp.resize(dim, 0.0);
        // K1
        derivs.copy_slice(&system, &mut self.tmp);
        derivs.derivative(&mut self.tmp);

        // Add update to system
        for i in 0..dim {
            system[i] += h * self.tmp[i];
        }
    }
}
