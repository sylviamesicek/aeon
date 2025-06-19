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

pub fn logspace(base: f64, start: f64, end: f64, n: usize) -> Logspace {
    let step = if n > 1 {
        (end - start) / (n - 1) as f64
    } else {
        0.0
    };

    Logspace {
        sign: base.signum(),
        base: base.abs(),
        start,
        step,
        index: 0,
        len: n,
    }
}

pub struct Logspace {
    sign: f64,
    base: f64,
    start: f64,
    step: f64,
    index: usize,
    len: usize,
}

impl Iterator for Logspace {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<f64> {
        if self.index >= self.len {
            None
        } else {
            let i = self.index;
            self.index += 1;

            let exponent = self.start + self.step * i as f64;

            Some(self.sign * self.base.powf(exponent))
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len - self.index;

        (n, Some(n))
    }
}

pub fn linspace(start: f64, end: f64, n: usize) -> Linspace {
    let step = if n > 1 {
        (end - start) / (n - 1) as f64
    } else {
        0.0
    };

    Linspace {
        start,
        step,
        index: 0,
        len: n,
    }
}

pub struct Linspace {
    start: f64,
    step: f64,
    index: usize,
    len: usize,
}

impl Iterator for Linspace {
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
