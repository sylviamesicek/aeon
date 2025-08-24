use crate::run::Status;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::Path};

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
                let param = record.param.to_bits();

                map.insert(param, record);
            }
        }

        Ok(SearchHistory { map })
    }

    pub fn save_csv(&self, path: &Path) -> eyre::Result<()> {
        let mut records = self.map.values().collect::<Vec<_>>();
        records.sort_unstable_by(|&a, &b| a.param.total_cmp(&b.param));

        let mut writer = csv::Writer::from_path(path)?;
        for record in records {
            writer.serialize(record)?;
        }
        writer.flush()?;
        Ok(())
    }

    pub fn insert(&mut self, key: f64, status: Status) {
        let bits: u64 = key.to_bits();
        self.map.insert(
            bits,
            SearchRecord {
                param: key,
                // encode: float::encode_float(key),
                status,
            },
        );
    }

    pub fn status(&mut self, key: f64) -> Option<Status> {
        let bits: u64 = key.to_bits();
        self.map.get(&bits).map(|v| v.status)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchRecord {
    param: f64,
    // encode: String,
    status: Status,
}
