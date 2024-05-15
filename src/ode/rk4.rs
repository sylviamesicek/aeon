use super::Ode;

/// RK4 Integrator.
#[derive(Clone, Debug)]
pub struct Rk4 {
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
}

impl Rk4 {
    pub fn new() -> Self {
        Self {
            k1: Vec::new(),
            k2: Vec::new(),
            k3: Vec::new(),
            k4: Vec::new(),
        }
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
        self.k2.resize(dim, 0.0);
        self.k3.resize(dim, 0.0);
        self.k4.resize(dim, 0.0);

        // K1
        for i in 0..dim {
            update[i] = system[i];
        }
        derivs.preprocess(update);
        derivs.derivative(update, &mut self.k1);

        // K2
        for i in 0..dim {
            update[i] = system[i] + h / 2.0 * self.k1[i];
        }
        derivs.preprocess(update);
        derivs.derivative(update, &mut self.k2);

        // K3
        for i in 0..dim {
            update[i] = system[i] + h / 2.0 * self.k2[i];
        }
        derivs.preprocess(update);
        derivs.derivative(update, &mut self.k3);

        // K4
        for i in 0..dim {
            update[i] = system[i] + h * self.k3[i];
        }
        derivs.preprocess(update);
        derivs.derivative(update, &mut self.k4);

        // Compute total step
        for i in 0..dim {
            update[i] = h / 6.0 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i])
        }
    }
}
