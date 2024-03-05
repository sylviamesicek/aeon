pub trait Boundary {
    const IS_FREE: bool = false;

    const GHOST: usize = 0;

    fn negative(self: &Self, values: &[f64]) -> f64;
    fn positive(self: &Self, values: &[f64]) -> f64;
}
