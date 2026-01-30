use std::{fs::File, path::Path};

#[derive(Clone, Debug, serde::Serialize)]
pub struct HistoryInfo {
    pub proper_time: f64,
    pub coord_time: f64,
    pub nodes: usize,
    pub dofs: usize,
    pub levels: usize,
    pub alpha: f64,
    pub grr: f64,
    pub grz: f64,
    pub gzz: f64,
    pub theta: f64,
}

#[derive(Debug)]
pub struct History {
    /// Csv file output (if any)
    writer: Option<csv::Writer<File>>,
}

impl History {
    /// Constructs a run history object that ignores record data.
    pub fn empty() -> Self {
        Self { writer: None }
    }

    /// Constructs a run history object that will periodically store record data in the given file.
    pub fn output(path: &Path) -> Result<Self, csv::Error> {
        Ok(Self {
            writer: Some(csv::Writer::from_path(path)?),
        })
    }

    pub fn write_record(&mut self, record: HistoryInfo) -> Result<(), csv::Error> {
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
