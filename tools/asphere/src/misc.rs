use eyre::Context as _;
use indicatif::ProgressStyle;
use serde::de::DeserializeOwned;
use std::path::{Path, PathBuf};

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
