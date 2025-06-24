use serde::{de::DeserializeOwned, Serialize};
use std::path::{Path, PathBuf};

/// Returns the path if it is absolute, otherwise transform it into a
/// absolute path by appending it to the current working directory.
pub fn abs_or_relative(path: &Path) -> std::io::Result<PathBuf> {
    abs_or_relative_to(&std::env::current_dir()?, path)
}

pub fn abs_or_relative_to(dir: &Path, path: &Path) -> std::io::Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }

    Ok(dir.join(path))
}

/// Deserialize data from toml file.
pub fn import_toml<T: DeserializeOwned>(path: &Path) -> std::io::Result<T> {
    let string = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&string).map_err(|err| std::io::Error::other(err))?)
}

/// Serialize data to toml file.
pub fn export_toml<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    let string = toml::to_string_pretty(value).map_err(|err| std::io::Error::other(err))?;
    std::fs::write(path, string)
}
