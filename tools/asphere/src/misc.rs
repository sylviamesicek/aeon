use eyre::Context as _;
use indicatif::ProgressStyle;
use serde::de::DeserializeOwned;
use std::{
    num::ParseIntError,
    path::{Path, PathBuf},
};

/// Progress bar in the style
/// `<prefix> . <message>`
pub fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{prefix:.bold.dim} {spinner} {wide_msg}")
        .unwrap()
        .tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ")
}

/// Progress bar in the style
/// <prefix> ####.... <pos>/<len>, <percent>%
pub fn node_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold.dim} {bar:.cyan/blue} {human_pos}/{human_len}, {percent}%",
    )
    .unwrap()
}

pub fn byte_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold.dim} {bar:.cyan/blue} {binary_bytes}/{binary_total_bytes}, {percent}%",
    )
    .unwrap()
}

pub fn level_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{prefix:.bold.dim} {bar:.cyan/blue} {human_pos}/{human_len} levels",
    )
    .unwrap()
}

/// Returns the path if it is absolute, otherwise transform it into a
/// absolute path by appending it to the current working directory.
pub fn abs_or_relative(path: &Path) -> eyre::Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }

    Ok(std::env::current_dir()
        .context("Failed to find current working directory")?
        .join(path))
}

/// Deserialize data from toml file.
pub fn import_from_toml<T: DeserializeOwned>(path: &Path) -> eyre::Result<T> {
    let string = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&string)?)
}

/// Encodes a float as a hexidecimal string
pub fn encode_float(value: f64) -> String {
    let bits: u64 = unsafe { std::mem::transmute(value) };
    format!("{:016x}", bits)
}

/// Decodes a float as a hexidecimal string
#[allow(dead_code)]
pub fn decode_float(value: &str) -> Result<f64, ParseIntError> {
    let bits = u64::from_str_radix(value, 16)?;
    Ok(unsafe { std::mem::transmute(bits) })
}

use core::f64;

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
