use super::system::{System, SystemLabel};

pub trait DiffEq<Label: SystemLabel> {
    fn preprocess(&mut self, system: &mut System<Label>);
    fn derivative(&mut self, system: &System<Label>, result: &mut System<Label>);
}

#[derive(Clone, Debug)]
pub struct Rk4<Label: SystemLabel> {
    pub system: System<Label>,
    pub time: f64,

    k1: System<Label>,
    k2: System<Label>,
    k3: System<Label>,
    k4: System<Label>,
    scr: System<Label>,
}

impl<Label: SystemLabel> Rk4<Label> {
    pub fn new(len: usize) -> Self {
        let system = System::new(len);
        let k1 = System::new(len);
        let k2 = System::new(len);
        let k3 = System::new(len);
        let k4 = System::new(len);
        let scr = System::new(len);

        Self {
            system,
            time: 0.0,

            k1,
            k2,
            k3,
            k4,
            scr,
        }
    }

    pub fn step<ODE: DiffEq<Label>>(&mut self, eq: &mut ODE, h: f64) {
        // K1
        eq.preprocess(&mut self.system);
        eq.derivative(&self.system, &mut self.k1);

        // K2
        for field in 0..Label::FIELDS {
            let sys = self.system.field(field);
            let scr = self.scr.field_mut(field);
            let k1 = self.k1.field(field);

            for idx in 0..self.system.len() {
                scr[idx] = sys[idx] + h / 2.0 * k1[idx];
            }
        }
        eq.preprocess(&mut self.scr);
        eq.derivative(&self.scr, &mut self.k2);

        // K3
        for field in 0..Label::FIELDS {
            let sys = self.system.field(field);
            let scr = self.scr.field_mut(field);
            let k2 = self.k2.field(field);

            for idx in 0..self.system.len() {
                scr[idx] = sys[idx] + h / 2.0 * k2[idx];
            }
        }
        eq.preprocess(&mut self.scr);
        eq.derivative(&self.scr, &mut self.k3);

        // K4
        for field in 0..Label::FIELDS {
            let sys = self.system.field(field);
            let scr = self.scr.field_mut(field);
            let k3 = self.k3.field(field);

            for idx in 0..self.system.len() {
                scr[idx] = sys[idx] + h * k3[idx];
            }
        }
        eq.preprocess(&mut self.scr);
        eq.derivative(&self.scr, &mut self.k4);

        // Compute total step

        for field in 0..Label::FIELDS {
            let len = self.system.len();
            let sys = self.system.field_mut(field);
            let k1 = self.k1.field(field);
            let k2 = self.k2.field(field);
            let k3 = self.k3.field(field);
            let k4 = self.k4.field(field);

            for idx in 0..len {
                sys[idx] += h / 6.0 * (k1[idx] + 2.0 * k2[idx] + 2.0 * k3[idx] + k4[idx]);
            }
        }

        self.time += h;
    }
}
