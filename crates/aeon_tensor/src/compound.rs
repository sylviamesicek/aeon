use crate::indices::{Gen, IndexIterator, Sym, TensorIndex};
use std::marker::PhantomData;

/// Singleton for implementing `CompondSum`
pub struct CSum;

/// Seal CompoundSum to CSum.
mod private {
    pub trait Sealed {}
}

impl private::Sealed for CSum {}

pub trait CompoundSum<const T: usize, const L: usize, const R: usize>: private::Sealed {
    fn join(idx: ([usize; L], [usize; R])) -> [usize; T];
    fn split(idx: [usize; T]) -> ([usize; L], [usize; R]);
}

// DRY macro for Implementing Compound sum
macro_rules! impl_compound_sum {
    ($T:literal, $L:literal, $R:literal, [$($left:ident),+] [$($right:ident),+]) => {
        impl CompoundSum<$T, $L, $R> for CSum {
            fn join(([$($left),*], [$($right),*]): ([usize; $L], [usize; $R])) -> [usize; $T] {
                [$($left),* , $($right),*]
            }

            fn split([$($left),* , $($right),*]: [usize; $T]) -> ([usize; $L], [usize; $R]) {
                ([$($left),*], [$($right),*])
            }
        }
    };
}

impl_compound_sum!(3, 1, 2, [a][b, c]);
impl_compound_sum!(3, 2, 1, [a, b][c]);
impl_compound_sum!(4, 2, 2, [a, b][c, d]);
impl_compound_sum!(4, 1, 3, [a][b, c, d]);
impl_compound_sum!(4, 3, 1, [a, b, c][d]);

/// Represents compound combinations of fundemental indices.
pub struct Compound<const T: usize, const L: usize, const R: usize, Left, Right>
where
    CSum: CompoundSum<T, L, R>,
{
    _marker: PhantomData<(Left, Right)>,
}

impl<const N: usize, const T: usize, const L: usize, const R: usize, Left, Right> TensorIndex<N, T>
    for Compound<T, L, R, Left, Right>
where
    Left: TensorIndex<N, L>,
    Right: TensorIndex<N, R>,
    CSum: CompoundSum<T, L, R>,
{
    type Indices = CompoundIndices<N, T, L, R, Left::Indices, Right::Indices>;

    fn offset_from_index(index: [usize; T]) -> usize {
        let (a, b) = CSum::split(index);

        let stride = Right::count();
        let most_sig = Left::offset_from_index(a);
        let least_sig = Right::offset_from_index(b);

        most_sig * stride + least_sig
    }

    fn indices() -> Self::Indices {
        CompoundIndices::zero()
    }

    fn count() -> usize {
        Left::count() * Right::count()
    }

    fn for_each_index(mut f: impl FnMut([usize; T])) {
        Left::for_each_index(|a| {
            Right::for_each_index(|b| {
                let idx = CSum::join((a, b));
                f(idx)
            })
        });
    }
}

/// Indexes over compound indices.
#[derive(Default, Clone, Debug)]
pub struct CompoundIndices<
    const N: usize,
    const T: usize,
    const L: usize,
    const R: usize,
    Left,
    Right,
> {
    first: Left,
    first_cur: Option<Option<[usize; L]>>,
    second: Right,
}

impl<const N: usize, const T: usize, const L: usize, const R: usize, Left, Right> Iterator
    for CompoundIndices<N, T, L, R, Left, Right>
where
    Left: IndexIterator<L>,
    Right: IndexIterator<R>,
    CSum: CompoundSum<T, L, R>,
{
    type Item = [usize; T];

    fn next(&mut self) -> Option<Self::Item> {
        let b = match self.second.next() {
            Some(b) => b,
            None => {
                // Advance left iterator
                self.first_cur = Some(self.first.next());
                // Reset right-most iterator
                self.second = Right::zero();
                // Get current value of left-most iterator
                self.second.next()?
            }
        };

        self.first_cur
            .get_or_insert_with(|| self.first.next())
            .map(|a| CSum::join((a, b)))
    }
}

impl<const N: usize, const T: usize, const L: usize, const R: usize, Left, Right> IndexIterator<T>
    for CompoundIndices<N, T, L, R, Left, Right>
where
    Left: IndexIterator<L>,
    Right: IndexIterator<R>,
    CSum: CompoundSum<T, L, R>,
{
    fn zero() -> Self {
        Self {
            first: Left::zero(),
            first_cur: None,
            second: Right::zero(),
        }
    }
}

/// A tensor of the form Tᵢ₍ⱼₖ₎
pub type VecSym = Compound<3, 1, 2, Gen, Sym>;
pub type SymVec = Compound<3, 2, 1, Sym, Gen>;
pub type SymSym = Compound<4, 2, 2, Sym, Sym>;
pub type VecSymVec = Compound<4, 3, 1, VecSym, Gen>;
