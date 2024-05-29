pub struct Tensor<R: LayoutHelper>(R::Storage);

impl<R: LayoutHelper> Tensor<R> {
    pub fn new(value: R::Storage) -> Self {
        Self(value)
    }
}

pub struct Rank<const N: usize, const U: usize, const L: usize>;

impl<const N: usize> LayoutHelper for Rank<N, 0, 1> {
    const DIM: usize = N;
    type Storage = [f64; N];
    type UpperIndices = [usize; 0];
    type LowerIndices = [usize; 1];
}

impl<const N: usize> LayoutCovariant for Rank<N, 0, 1> {}

pub trait LayoutHelper {
    const DIM: usize;
    type Storage: AsRef<[f64]> + AsMut<[f64]>;
    type UpperIndices: AsRef<[usize]> + AsMut<[usize]>;
    type LowerIndices: AsRef<[usize]> + AsMut<[usize]>;
}

pub trait LayoutCovariant: LayoutHelper {}
pub trait LayoutContravariant: LayoutHelper {}
