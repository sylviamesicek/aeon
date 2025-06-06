use serde::{Deserialize, Serialize};
use std::convert::Infallible;

// ***********************************
// Intervals

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
#[serde(untagged)]
pub enum Interval {
    ProperTime { proper_time: f64 },
    CoordTime { coord_time: f64 },
    Steps { steps: usize },
}

impl Interval {
    pub fn unwrap_steps(self) -> usize {
        let Self::Steps { steps } = self else {
            panic!("failed to unwarp interval into Interval::Steps")
        };
        steps
    }

    pub fn _unwrap_proper_time(self) -> f64 {
        let Self::ProperTime { proper_time } = self else {
            panic!("failed to unwarp interval into Interval::ProperTime")
        };
        proper_time
    }

    pub fn _unwrap_coord_time(self) -> f64 {
        let Self::CoordTime { coord_time } = self else {
            panic!("failed to unwarp interval into Interval::CoordTime")
        };
        coord_time
    }
}

impl Default for Interval {
    fn default() -> Self {
        Interval::Steps { steps: 1 }
    }
}

pub struct IntervalTracker {
    proper_time_elapsed: f64,
    coord_time_elapsed: f64,
    steps_elapsed: usize,
}

impl IntervalTracker {
    pub fn new() -> Self {
        Self {
            proper_time_elapsed: 0.0,
            coord_time_elapsed: 0.0,
            steps_elapsed: 0,
        }
    }

    pub fn every(&mut self, interval: Interval, f: impl FnOnce()) {
        self.try_every::<Infallible>(interval, || {
            f();
            Ok(())
        })
        .unwrap()
    }

    pub fn try_every<E>(
        &mut self,
        interval: Interval,
        f: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), E> {
        match interval {
            Interval::ProperTime { proper_time } => {
                if self.proper_time_elapsed < proper_time {
                    return Ok(());
                }

                self.proper_time_elapsed -= proper_time;

                f()
            }
            Interval::CoordTime { coord_time } => {
                if self.coord_time_elapsed < coord_time {
                    return Ok(());
                }

                self.coord_time_elapsed -= coord_time;

                f()
            }
            Interval::Steps { steps } => {
                if self.steps_elapsed < steps {
                    return Ok(());
                }

                self.steps_elapsed -= steps;

                f()
            }
        }
    }

    pub fn update(&mut self, proper: f64, coord: f64, steps: usize) {
        self.proper_time_elapsed += proper;
        self.coord_time_elapsed += coord;
        self.steps_elapsed += steps;
    }
}
