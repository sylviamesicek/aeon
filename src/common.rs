pub trait Engine<const N: usize> {
    fn position(&self) -> [f64; N];

    fn value(&self, channel: usize) -> f64;
    fn gradient(&self, i: usize, channel: usize) -> f64;
    fn hessian(&self, i: usize, j: usize, channel: usize) -> f64;
    fn dissipation(&self, channel: usize) -> f64;
}
