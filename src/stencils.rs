use num::{rational::Rational64 as Ratio, One, Zero};
use std::ops::Index;

#[derive(Debug, Clone, Copy)]
pub struct Stencil<const M: usize>([Ratio; M]);

impl<const M: usize> Stencil<M> {
    pub fn value_weights(self: Self, point: Ratio) -> [Ratio; M] {
        let mut weights = [Ratio::one(); M];

        for i in 0..M {
            for j in 0..M {
                if i != j {
                    weights[i] *= (point - self[j]) / (self[i] - self[j])
                }
            }
        }

        weights
    }

    pub fn derivative_weights(self: Self, point: Ratio) -> [Ratio; M] {
        let mut weights = [Ratio::zero(); M];

        for i in 0..M {
            for j in 0..M {
                if i != j {
                    let mut result = Ratio::one();

                    for k in 0..M {
                        if i != k && j != k {
                            result *= (point - self[k]) / (self[i] - self[k]);
                        }
                    }

                    weights[i] += result / (self[i] - self[j])
                }
            }
        }

        weights
    }

    pub fn second_derivative_weights(self: Self, point: Ratio) -> [Ratio; M] {
        let mut weights = [Ratio::zero(); M];

        for i in 0..M {
            for j in 0..M {
                if i != j {
                    let mut result1 = Ratio::zero();

                    for k in 0..M {
                        if i != k && j != k {
                            let mut result2 = Ratio::one();

                            for l in 0..M {
                                if l != i && l != j && l != k {
                                    result2 *= (point - self[l]) / (self[i] - self[l]);
                                }
                            }

                            result1 += result2 / (self[i] - self[k]);
                        }
                    }

                    weights[i] += result1 / (self[i] - self[j])
                }
            }
        }

        weights
    }
}

impl<const M: usize> Index<usize> for Stencil<M> {
    type Output = Ratio;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

#[cfg(test)]
mod tests {
    use num::rational::Ratio;

    use super::*;

    #[test]
    fn print_stencil() {
        let stencil = Stencil([
            Ratio::new(-2, 1),
            Ratio::new(-1, 1),
            Ratio::new(0, 1),
            Ratio::new(1, 1),
            Ratio::new(2, 1),
        ]);

        let edge = Stencil([
            Ratio::new(0, 1),
            Ratio::new(1, 1),
            Ratio::new(2, 1),
            Ratio::new(3, 1),
            Ratio::new(4, 1),
            Ratio::new(5, 1),
        ]);

        let forward0 = edge.second_derivative_weights(Ratio::new(0, 1));
        let forward1 = edge.second_derivative_weights(Ratio::new(1, 1));

        let centered = stencil.second_derivative_weights(Ratio::new(0, 1));

        println!("{:?}", centered);
        println!("{:?}", forward0);
        println!("{:?}", forward1);
    }

    // #[test]
    // fn stencil() {
    //     let stencil = Stencil([-2.0, -1.0, 0.0, 1.0, 2.0]);

    //     let extra1 = stencil.value_weights(3.0);
    //     let extra2 = stencil.value_weights(4.0);

    //     println!("{:?}", extra1);
    //     println!("{:?}", extra2);

    //     let centered = stencil.derivative_weights(0.0);
    //     let forward = stencil.derivative_weights(2.0);

    //     let mut result = [0.0; 5];
    //     result[2] = centered[0];
    //     result[3] = centered[1];
    //     result[4] = centered[2];

    //     for i in 0..3 {
    //         result[i] += centered[3] * extra1[i];
    //         result[i] += centered[4] * extra2[i];
    //     }

    //     println!("{:?}", result);
    //     println!("{:?}", forward)
    // }
}
