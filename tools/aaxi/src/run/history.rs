use serde::{Deserialize, Serialize};
use std::{fs::File, path::Path};

#[derive(Debug)]
pub struct RunHistory {
    /// Csv file output (if any)
    writer: Option<csv::Writer<File>>,
}

impl RunHistory {
    /// Constructs a run history object that ignores record data.
    pub fn empty() -> Self {
        Self { writer: None }
    }

    /// Constructs a run history object that will periodically store record data in the given file.
    pub fn _output(path: &Path) -> Result<Self, csv::Error> {
        Ok(Self {
            writer: Some(csv::Writer::from_path(path)?),
        })
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
