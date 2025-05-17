use serde::{Deserialize, Serialize};
use std::{fs::File, path::Path};

pub struct RunHistory {
    writer: Option<csv::Writer<File>>,
}

impl RunHistory {
    pub fn empty() -> Self {
        Self { writer: None }
    }

    pub fn from_path(path: &Path) -> Result<Self, csv::Error> {
        Ok(Self {
            writer: Some(csv::Writer::from_path(path)?),
        })
    }
}

impl RunHistory {
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

#[derive(Serialize, Deserialize)]
pub struct RunRecord {
    pub step: usize,
    pub time: f64,
    pub proper_time: f64,
    pub lapse: f64,
}
