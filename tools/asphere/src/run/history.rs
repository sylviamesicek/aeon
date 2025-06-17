use std::{collections::HashMap, path::Path};

use serde::{Deserialize, Serialize};

use crate::misc;

#[derive(Clone)]
pub struct SearchHistory {
    map: HashMap<u64, SearchRecord>,
}

impl SearchHistory {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn load_csv(path: &Path) -> eyre::Result<Self> {
        let mut map = HashMap::new();

        let mut reader = csv::Reader::from_path(path)?;
        for record in reader.deserialize::<SearchRecord>() {
            if let Ok(record) = record {
                let param = unsafe { std::mem::transmute(record.param) };

                map.insert(param, record);
            }
        }

        Ok(SearchHistory { map })
    }

    pub fn save_csv(&self, path: &Path) -> eyre::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;
        for record in self.map.values() {
            writer.serialize(record)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn insert(&mut self, key: f64, status: Status) {
        let bits: u64 = unsafe { std::mem::transmute(key) };
        self.map.insert(
            bits,
            SearchRecord {
                param: key,
                encode: misc::encode_float(key),
                status,
            },
        );
    }

    pub fn status(&mut self, key: f64) -> Option<Status> {
        let bits: u64 = unsafe { std::mem::transmute(key) };
        self.map.get(&bits).map(|v| v.status)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchRecord {
    param: f64,
    encode: String,
    status: Status,
}

/// Status of an indivdual run.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Disperse,
    Collapse,
}
