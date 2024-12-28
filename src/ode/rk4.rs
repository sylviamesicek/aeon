use super::Ode;

/// RK4 Integrator.
#[derive(Clone, Debug, Default)]
pub struct Rk4 {
    tmp: Vec<f64>,
    update: Vec<f64>,
}

impl Rk4 {
    /// Constructs an empty, default Rk4 integrator.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn tmp(&mut self) -> &mut Vec<f64> {
        &mut self.tmp
    }

    /// Take one Rk4 step.
    pub fn step<Problem: Ode>(&mut self, h: f64, derivs: &mut Problem, system: &mut [f64]) {
        let dim = system.len();

        assert!(system.len() == dim);

        self.tmp.clear();
        self.update.clear();

        self.tmp.resize(dim, 0.0);
        self.update.resize(dim, 0.0);

        // K1
        self.tmp.copy_from_slice(&system);
        derivs.derivative(&mut self.tmp);
        for i in 0..dim {
            self.update[i] += 1.0 / 6.0 * self.tmp[i];
        }

        // K2
        for i in 0..dim {
            self.tmp[i] = system[i] + h / 2.0 * self.tmp[i];
        }
        derivs.derivative(&mut self.tmp);
        for i in 0..dim {
            self.update[i] += 1.0 / 3.0 * self.tmp[i];
        }

        // K3
        for i in 0..dim {
            self.tmp[i] = system[i] + h / 2.0 * self.tmp[i];
        }
        derivs.derivative(&mut self.tmp);
        for i in 0..dim {
            self.update[i] += 1.0 / 3.0 * self.tmp[i];
        }

        // K4
        for i in 0..dim {
            self.tmp[i] = system[i] + h * self.tmp[i];
        }
        derivs.derivative(&mut self.tmp);
        for i in 0..dim {
            self.update[i] += 1.0 / 6.0 * self.tmp[i];
        }

        // Total step
        for i in 0..dim {
            system[i] += h * self.update[i];
        }
    }
}
