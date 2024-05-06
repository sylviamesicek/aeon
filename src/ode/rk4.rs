use super::Ode;

/// Global RK4 Integrator.
#[derive(Clone, Debug)]
pub struct Rk4 {
    pub system: Vec<f64>,
    pub time: f64,

    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    scr: Vec<f64>,

    dim: usize,
}

impl Rk4 {
    pub fn new(dim: usize) -> Self {
        let system = vec![0.0; dim];
        let k1 = vec![0.0; dim];
        let k2 = vec![0.0; dim];
        let k3 = vec![0.0; dim];
        let k4 = vec![0.0; dim];
        let scr = vec![0.0; dim];

        Self {
            system,
            time: 0.0,

            k1,
            k2,
            k3,
            k4,
            scr,

            dim,
        }
    }

    pub fn step<Problem: Ode>(&mut self, problem: &mut Problem, h: f64) {
        // K1
        problem.preprocess(&mut self.system);
        problem.derivative(&self.system, &mut self.k1);

        // K2
        for i in 0..self.dim {
            self.scr[i] = self.system[i] + h / 2.0 * self.k1[i];
        }

        problem.preprocess(&mut self.scr);
        problem.derivative(&self.scr, &mut self.k2);

        // K3
        for i in 0..self.dim {
            self.scr[i] = self.system[i] + h / 2.0 * self.k2[i];
        }
        problem.preprocess(&mut self.scr);
        problem.derivative(&self.scr, &mut self.k3);

        // K4
        for i in 0..self.dim {
            self.scr[i] = self.system[i] + h * self.k3[i];
        }
        problem.preprocess(&mut self.scr);
        problem.derivative(&self.scr, &mut self.k4);

        // Compute total step

        for i in 0..self.dim {
            self.system[i] +=
                h / 6.0 * (self.k1[i] + 2.0 * self.k2[i] + 2.0 * self.k3[i] + self.k4[i])
        }

        self.time += h;
    }
}
