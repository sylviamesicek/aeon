pub trait TensorIndex<const N: usize, const R: usize> {
    /// Type of iterator over valid tensor indices.
    type Indices: IndexIterator<R>;
    /// Converts a valid index into a buffer offset for tensor storage.
    fn offset_from_index(index: [usize; R]) -> usize;
    /// Iterates over all (unique) indices in the tensor.
    /// This must index the tensor in the same order as `Self::offset_from_index`.
    fn indices() -> Self::Indices;
    /// Counts all unique indices used to store the tensor.
    fn count() -> usize {
        Self::indices().count()
    }
    /// Calls a function for each index in the tensor.
    /// Must index over the tensor in the same order as `Self::indices`.
    fn for_each_index(f: impl FnMut([usize; R])) {
        Self::indices().for_each(f);
    }
}

pub trait IndexIterator<const R: usize>: Iterator<Item = [usize; R]> {
    fn zero() -> Self;
}

/// General tensor implementation with no symmetries of the form Tᵢⱼₖ...
pub struct Gen;

impl<const N: usize, const R: usize> TensorIndex<N, R> for Gen {
    type Indices = GenIndices<N, R>;

    fn offset_from_index(index: [usize; R]) -> usize {
        let mut result = 0;
        let mut stride = 1;

        for i in (0..R).rev() {
            result += stride * index[i];
            stride *= N;
        }

        result
    }

    fn indices() -> Self::Indices {
        GenIndices { cursor: [0; R] }
    }

    fn for_each_index(mut f: impl FnMut([usize; R])) {
        if const { R == 0 } {
            f([0; R]);
            return;
        }

        let mut cursor = [0; R];

        f(cursor);

        'l: loop {
            for slot in (0..R).rev() {
                cursor[slot] += 1;

                if cursor[slot] < N {
                    f(cursor);
                    continue 'l;
                }

                cursor[slot] = 0;
            }

            break;
        }
    }

    fn count() -> usize {
        const {
            let mut result = 1;
            let mut i = 0;

            while i < R {
                result *= N;
                i += 1;
            }

            result
        }
    }
}

/// Iterate indices in row-major (or left-slot major more generally) order.
pub struct GenIndices<const N: usize, const R: usize> {
    cursor: [usize; R],
}

impl<const N: usize, const R: usize> Default for GenIndices<N, R> {
    fn default() -> Self {
        Self { cursor: [0; R] }
    }
}

impl<const N: usize, const R: usize> Iterator for GenIndices<N, R> {
    type Item = [usize; R];

    fn next(&mut self) -> Option<Self::Item> {
        if const { N == 0 } {
            // Short circuit if the dimension is zero.
            return None;
        }

        // Last index was incremented, iteration is complete
        if self.cursor[0] >= N {
            return None;
        }

        // Store current cursor value (this is what we will return)
        let result = self.cursor;

        for slot in (0..R).rev() {
            // If we need to increment this axis, we add to the cursor value
            self.cursor[slot] += 1;
            // If the cursor is equal to size, we wrap.
            // However, if we have reached the final axis,
            // this indicates we are at the end of iteration,
            // and will return None on the next call of next().
            if self.cursor[slot] == N && slot > 0 {
                self.cursor[slot] = 0;
                continue;
            }

            break;
        }

        Some(result)
    }
}

impl<const N: usize, const R: usize> IndexIterator<R> for GenIndices<N, R> {
    fn zero() -> Self {
        Self { cursor: [0; R] }
    }
}

/// A tensor of the form T₍ᵢⱼ₎
pub struct Sym;

impl<const N: usize> TensorIndex<N, 2> for Sym {
    type Indices = SymIndices<N>;

    fn offset_from_index([mut row, mut col]: [usize; 2]) -> usize {
        if const { N == 1 } {
            return 0;
        }

        if const { N == 2 } {
            return row + col;
        }

        // Make sure numbers are
        if col > row {
            // Swap col and row
            let tmp = col;
            col = row;
            row = tmp;
        }

        let row_offset = (row * (row + 1)) / 2; // Use gaussian addition to find row offset
        row_offset + col
    }

    fn count() -> usize {
        const { N * (N + 1) / 2 }
    }

    fn indices() -> Self::Indices {
        SymIndices::default()
    }

    fn for_each_index(mut f: impl FnMut([usize; 2])) {
        if const { N == 1 } {
            f([0, 0]);
            return;
        }

        if const { N == 2 } {
            f([0, 0]);
            f([1, 0]);
            f([1, 1]);
            return;
        }

        for row in 0..N {
            for col in 0..=row {
                f([row, col]);
            }
        }
    }
}

#[derive(Default)]
pub struct SymIndices<const N: usize> {
    cursor: [usize; 2],
}

impl<const N: usize> Iterator for SymIndices<N> {
    type Item = [usize; 2];

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor[0] >= N {
            return None;
        }

        let result = self.cursor;
        // Start at (i, j)
        self.cursor[1] += 1; // Now at (i, j + 1)
        let inc_row = self.cursor[1] / (self.cursor[0] + 1); // If j > i then 1 else 0
        self.cursor[1] %= self.cursor[0] + 1; // Make sure j = 0
        self.cursor[0] += inc_row;

        Some(result)
    }
}

impl<const N: usize> IndexIterator<2> for SymIndices<N> {
    fn zero() -> Self {
        Self { cursor: [0; 2] }
    }
}
