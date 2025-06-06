use serde::{Deserialize, Serialize};

/// What strategy should we employ on a possible error?
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    #[serde(rename = "crash")]
    Crash,
    #[serde(rename = "ignore")]
    Ignore,
    #[serde(rename = "disperse")]
    Disperse,
    #[serde(rename = "collapse")]
    Collapse,
}

impl Strategy {
    pub fn execute<E>(self, f: impl FnOnce() -> E) -> Result<Option<Status>, E> {
        if matches!(self, Self::Crash) {
            return Err(f());
        }

        Ok(match self {
            Strategy::Ignore => None,
            Strategy::Disperse => Some(Status::Disperse),
            Strategy::Collapse => Some(Status::Collapse),
            Strategy::Crash => unreachable!(),
        })
    }

    pub fn _ignore_or_crash<E>(self, f: impl FnOnce() -> E) -> Result<(), E> {
        assert!(
            !matches!(self, Self::Collapse | Self::Disperse),
            "collapse or disperse invalid, check that config was validated"
        );

        if matches!(self, Self::Crash) {
            return Err(f());
        }

        Ok(())
    }

    pub fn status_or_crash<E>(self, f: impl FnOnce() -> E) -> Result<Status, E> {
        assert!(
            !matches!(self, Self::Ignore),
            "ignore strategy invalid, check that config was validated"
        );

        if matches!(self, Self::Crash) {
            return Err(f());
        }

        match self {
            Strategy::Crash => Err(f()),
            Strategy::Ignore => unreachable!(),
            Strategy::Disperse => Ok(Status::Disperse),
            Strategy::Collapse => Ok(Status::Collapse),
        }
    }
}

/// Status of an indivdual run.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Disperse,
    Collapse,
}
