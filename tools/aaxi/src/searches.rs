use serde::{Deserialize, Serialize};
use serde_with::{KeyValueMap, serde_as};

#[derive(Serialize, Deserialize)]
pub struct Search {
    #[serde(rename = "$key$")]
    name: String,
    /// Start of range to search
    start: f64,
    /// End of range to search
    end: f64,
    /// Number of bifurcations to make
    bifurcations: usize,
}

#[serde_as]
#[derive(Serialize, Deserialize)]
pub struct Searches(#[serde_as(as = "KeyValueMap<_>")] Vec<Search>);
