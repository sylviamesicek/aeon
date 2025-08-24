use std::f64;
use std::num::ParseIntError;

/// Encodes a float as a hexidecimal string
pub fn encode_float(value: f64) -> String {
    let bits: u64 = value.to_bits();
    format!("{:016x}", bits)
}

/// Decodes a float as a hexidecimal string
#[allow(dead_code)]
pub fn decode_float(value: &str) -> Result<f64, ParseIntError> {
    let bits = u64::from_str_radix(value, 16)?;
    Ok(f64::from_bits(bits))
}

pub fn log_range(start: f64, end: f64, n: usize) -> LogRange {
    assert!(n >= 2);
    assert!(start >= 0.0 && end >= 0.0);

    let loga = start.log2() / (n - 1) as f64;
    let logb = end.log2() / (n - 1) as f64;

    LogRange {
        index: 0,
        len: n,
        loga,
        logb,
    }
}

pub struct LogRange {
    len: usize,
    index: usize,
    loga: f64,
    logb: f64,
}

impl Iterator for LogRange {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<f64> {
        if self.index >= self.len {
            None
        } else {
            let i = self.index;
            self.index += 1;

            let logx = (self.len - 1 - i) as f64 * self.loga + i as f64 * self.logb;
            Some(logx.exp2())
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.index;

        (n, Some(n))
    }
}

pub fn lin_range(start: f64, end: f64, n: usize) -> LinRange {
    let step = if n > 1 {
        (end - start) / (n - 1) as f64
    } else {
        0.0
    };

    LinRange {
        start,
        step,
        index: 0,
        len: n,
    }
}

pub struct LinRange {
    start: f64,
    step: f64,
    len: usize,
    index: usize,
}

impl Iterator for LinRange {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<f64> {
        if self.index >= self.len {
            None
        } else {
            let i = self.index;
            self.index += 1;

            Some(self.start + self.step * i as f64)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.index;

        (n, Some(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    /// Make sure `decode_float(encode_float(v).as_str()) == v` for
    /// any $v: f64$.
    #[test]
    fn float_encode_decode() {
        let mut rng = rand::rng();

        for _ in 0..100 {
            let value = rng.random_range(-1e6..1e6);
            let encoded = encode_float(value);
            let decoded = decode_float(&encoded).unwrap();

            assert_eq!(value, decoded);
        }
    }

    #[test]
    fn ranges() {
        let mut space = log_range(2.0, 128.0, 7);
        macro_rules! assert_nearly_eq {
            ($term:expr, $value:expr) => {
                let tup = ($term, $value);
                if let (Some(t), Some(v)) = tup {
                    assert!((t - v as f64).abs() <= 1e-8);
                } else if let (None, None) = tup {
                } else {
                    panic!()
                }
            };
        }

        assert_nearly_eq!(space.next(), Some(2.0));
        assert_nearly_eq!(space.next(), Some(4.0));
        assert_nearly_eq!(space.next(), Some(8.0));
        assert_nearly_eq!(space.next(), Some(16.0));
        assert_nearly_eq!(space.next(), Some(32.0));
        assert_nearly_eq!(space.next(), Some(64.0));
        assert_nearly_eq!(space.next(), Some(128.0));
        assert_nearly_eq!(space.next(), None::<f64>);
    }
}
