use crate::{Static, Tensor, TensorIndex, TensorProd, TensorRank};

// ************************
// Fields *****************
// ************************

#[derive(Clone)]
pub struct TensorFieldC0<const N: usize, R: TensorRank<N>> {
    pub value: Tensor<N, R>,
}

impl<const N: usize, R: TensorRank<N>> From<TensorFieldC1<N, R>> for TensorFieldC0<N, R>
where
    R: TensorProd<N, Static<1>>,
{
    fn from(other: TensorFieldC1<N, R>) -> Self {
        Self { value: other.value }
    }
}

impl<const N: usize, R: TensorRank<N>> From<TensorFieldC2<N, R>> for TensorFieldC0<N, R>
where
    R: TensorProd<N, Static<1>>,
    <R as TensorProd<N, Static<1>>>::Result: TensorProd<N, Static<1>>,
{
    fn from(other: TensorFieldC2<N, R>) -> Self {
        Self { value: other.value }
    }
}

#[derive(Clone)]
pub struct TensorFieldC1<const N: usize, R: TensorRank<N>>
where
    R: TensorProd<N, Static<1>>,
{
    pub value: Tensor<N, R>,
    pub derivs: Tensor<N, R::Result>,
}

impl<const N: usize, R: TensorRank<N>> From<TensorFieldC2<N, R>> for TensorFieldC1<N, R>
where
    R: TensorProd<N, Static<1>>,
    <R as TensorProd<N, Static<1>>>::Result: TensorProd<N, Static<1>>,
{
    fn from(other: TensorFieldC2<N, R>) -> Self {
        Self {
            value: other.value,
            derivs: other.derivs,
        }
    }
}

#[derive(Clone)]
pub struct TensorFieldC2<const N: usize, R: TensorRank<N>>
where
    R: TensorProd<N, Static<1>>,
    <R as TensorProd<N, Static<1>>>::Result: TensorProd<N, Static<1>>,
{
    pub value: Tensor<N, R>,
    pub derivs: Tensor<N, <R as TensorProd<N, Static<1>>>::Result>,
    pub second_derivs:
        Tensor<N, <<R as TensorProd<N, Static<1>>>::Result as TensorProd<N, Static<1>>>::Result>,
}

/// Computes the lie derivative along a vector of a given tensor field.
pub fn lie_derivative<const N: usize, T>(
    direction: TensorFieldC1<N, Static<1>>,
    tensor: TensorFieldC1<N, T>,
) -> Tensor<N, T>
where
    T: TensorRank<N> + TensorProd<N, Static<1>>,
{
    let mut result = Tensor::zeros();

    for index in T::Idx::enumerate() {
        for m in 0..N {
            let findex = T::index_combine(index, [m]);
            result[index] += direction.value[[m]] * tensor.derivs[findex];
        }
    }

    if const { <T::Idx as TensorIndex<N>>::RANK == 0 } {
        return result;
    }

    for findex in T::Idx::enumerate() {
        // Loop over every index slot.
        for r in 0..T::Idx::RANK {
            // Retrieve 'r'th component of index
            let i = findex.as_ref()[r];

            // Sum
            for m in 0..N {
                // Build mutable version and set 'r'th component to m
                let mut index = findex;
                index.as_mut()[r] = m;

                // Add to result
                result[findex] += direction.derivs[[m, i]] * tensor.value[index];
            }
        }
    }

    result
}
