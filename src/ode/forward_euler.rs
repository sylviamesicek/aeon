use super::Ode;

/// Standard forward Euler Integrator.
#[derive(Clone, Debug)]
pub struct ForwardEuler {
    pub system: Vec<f64>,
    pub time: f64,

    k1: Vec<f64>,

    dim: usize,
}

impl ForwardEuler {
    pub fn new(dim: usize) -> Self {
        let system = vec![0.0; dim];
        let k1 = vec![0.0; dim];

        Self {
            system,
            time: 0.0,

            k1,

            dim,
        }
    }

    pub fn reinit(&mut self, dim: usize) {
        self.time = 0.0;
        self.dim = dim;
        self.system.resize(dim, 0.0);
        self.k1.resize(dim, 0.0);
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn step<Problem: Ode>(&mut self, problem: &mut Problem, h: f64) {
        assert!(problem.dim() == self.dim());
        // K1
        problem.preprocess(&mut self.system);
        problem.derivative(&self.system, &mut self.k1);

        // Compute total step
        for i in 0..self.dim {
            self.system[i] = self.system[i] + h * self.k1[i];
        }

        self.time += h;
    }
}
