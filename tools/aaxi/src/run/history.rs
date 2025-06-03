use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs::File, path::Path};

use crate::misc;

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Unknown,
    Dispersed,
    MaxNodesReached,
    MaxStepsReached,
    MaxMemoryReached,
    NormDiverged,
}

impl RunStatus {
    pub fn _has_collapsed(self) -> bool {
        !self.has_dispersed()
    }

    pub fn has_dispersed(self) -> bool {
        matches!(self, Self::Dispersed)
    }
}

#[derive(Debug)]
pub struct RunHistory {
    /// Csv file output (if any)
    writer: Option<csv::Writer<File>>,
    /// Status of the current run
    status: RunStatus,
}

impl RunHistory {
    /// Constructs a run history object that ignores record data.
    pub fn empty() -> Self {
        Self {
            writer: None,
            status: RunStatus::Unknown,
        }
    }

    /// Constructs a run history object that will periodically store record data in the given file.
    pub fn output(path: &Path) -> Result<Self, csv::Error> {
        Ok(Self {
            writer: Some(csv::Writer::from_path(path)?),
            status: RunStatus::Unknown,
        })
    }

    /// Retrieves
    pub fn status(&self) -> RunStatus {
        self.status
    }

    pub fn set_status(&mut self, status: RunStatus) {
        self.status = status
    }

    pub fn write_record(&mut self, record: RunRecord) -> Result<(), csv::Error> {
        let Some(ref mut writer) = self.writer else {
            return Ok(());
        };

        writer.serialize(record)
    }

    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        let Some(ref mut writer) = self.writer else {
            return Ok(());
        };

        writer.flush()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RunRecord {
    pub step: usize,
    pub time: f64,
    pub proper_time: f64,
    pub lapse: f64,
}

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

    pub fn insert(&mut self, key: f64, status: RunStatus) {
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

    pub fn status(&mut self, key: f64) -> Option<RunStatus> {
        let bits: u64 = unsafe { std::mem::transmute(key) };
        self.map.get(&bits).map(|v| v.status)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SearchRecord {
    param: f64,
    encode: String,
    status: RunStatus,
}
