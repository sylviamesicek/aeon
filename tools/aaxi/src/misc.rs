//! A module for misc functions and utils that are used elsewhere in `aaxi`.

use std::{
    num::ParseIntError,
    path::{Path, PathBuf},
};

use eyre::Context as _;
use indicatif::ProgressStyle;
use serde::de::DeserializeOwned;

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
}
