use std::{fmt::Write as _, fs::File, path::Path};

#[derive(Clone, Copy, Debug)]
pub struct ScalarFieldInfo {
    pub phi: f64,
    pub pi: f64,
}

#[derive(Clone, Debug)]
pub struct HistoryInfo {
    pub proper_time: f64,
    pub coord_time: f64,
    pub output_index: usize,
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

    row: Vec<String>,
}

impl History {
    /// Constructs a run history object that ignores record data.
    pub fn empty() -> Self {
        Self {
            writer: None,
            row: Vec::new(),
        }
    }

    /// Constructs a run history object that will periodically store record data in the given file.
    pub fn output(
        path: &Path,
        num_scalar_fields: usize,
        offset: Option<usize>,
    ) -> Result<Self, csv::Error> {
        let buffer = std::fs::read_to_string(path).ok();
        let history_ =
            offset.and_then(|_| Some(csv::Reader::from_reader(buffer.as_ref()?.as_bytes())));

        let mut header: Vec<_> = [
            "output_index",
            "proper_time",
            "coord_time",
            "nodes",
            "dofs",
            "levels",
            "alpha",
            "grr",
            "grz",
            "gzz",
            "theta",
        ]
        .iter()
        .map(|i| i.to_string())
        .collect();
        header.extend(
            (0..num_scalar_fields).flat_map(|i| [format!("phi{i}"), format!("pi{i}")].into_iter()),
        );

        let mut writer = csv::Writer::from_path(path)?;
        writer.write_record(&header)?;

        // If previous history exists write rows 0..offset

        if let (Some(mut history), Some(offset)) = (history_, offset) {
            let mut records = history.records();
            // _ = records.next();

            'records: while let Some(Ok(record)) = records.next() {
                let Some(output_index) =
                    record.get(0).and_then(|index| index.parse::<usize>().ok())
                else {
                    break 'records;
                };

                if output_index < offset {
                    writer.write_record(&record)?;
                }
            }
        }

        writer.flush()?;

        Ok(Self {
            writer: Some(writer),
            row: header,
        })
    }

    pub fn write_record(
        &mut self,
        history: HistoryInfo,
        fields: impl IntoIterator<Item = ScalarFieldInfo>,
    ) -> eyre::Result<()> {
        let Some(ref mut writer) = self.writer else {
            return Ok(());
        };

        for s in &mut self.row {
            s.clear();
        }

        write!(&mut self.row[0], "{}", history.output_index)?;
        write!(&mut self.row[1], "{}", history.proper_time)?;
        write!(&mut self.row[2], "{}", history.coord_time)?;
        write!(&mut self.row[3], "{}", history.nodes)?;
        write!(&mut self.row[4], "{}", history.dofs)?;
        write!(&mut self.row[5], "{}", history.levels)?;
        write!(&mut self.row[6], "{}", history.alpha)?;
        write!(&mut self.row[7], "{}", history.grr)?;
        write!(&mut self.row[8], "{}", history.grz)?;
        write!(&mut self.row[9], "{}", history.gzz)?;
        write!(&mut self.row[10], "{}", history.theta)?;

        for (i, field) in fields.into_iter().enumerate() {
            write!(&mut self.row[11 + 2 * i], "{}", field.phi)?;
            write!(&mut self.row[11 + 2 * i + 1], "{}", field.pi)?;
        }

        writer.write_record(&self.row)?;

        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        let Some(ref mut writer) = self.writer else {
            return Ok(());
        };

        writer.flush()
    }
}
