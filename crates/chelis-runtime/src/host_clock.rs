//! [05-OP-75] host clock reads, shared by every execution lane.
//!
//! The evaluator and compiled host code read the same clocks, normalize the
//! reading the same way, and fail with the same message, because both call
//! these definitions: the evaluator through its policy-checked system port,
//! compiled code through the `chelis_clock_*_read` C exports.

use std::fmt;
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// [05-OP-75]: the least and greatest admitted reading second, the unix
/// seconds whose civil reading at every UTC offset under one day lies in
/// years -9999 through 9999.
pub const CLOCK_SECONDS_MIN: i64 = -377_705_030_401;
pub const CLOCK_SECONDS_MAX: i64 = 253_402_214_400;

/// The two [05-OP-75] reads, named by their builtin identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockOperation {
    Wall,
    Monotonic,
}

impl ClockOperation {
    pub const fn builtin(self) -> &'static str {
        match self {
            Self::Wall => "clock_wall_read",
            Self::Monotonic => "clock_monotonic_read",
        }
    }
}

impl fmt::Display for ClockOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.builtin())
    }
}

/// One host clock reading as the host reports it: a distance on one side of
/// the clock's origin. [`ClockReading::checked_time`], not the reader,
/// normalizes and range checks it, so an injected clock is held to the same
/// contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockReading {
    AtOrAfterOrigin(Duration),
    BeforeOrigin(Duration),
}

/// A checked [05-OP-75] reading in Euclidean form: `nanoseconds` lies in
/// `0..1_000_000_000` and `seconds` in the admitted range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockTime {
    pub seconds: i64,
    pub nanoseconds: i64,
}

/// A reading, in Euclidean form, whose seconds lie outside
/// [`CLOCK_SECONDS_MIN`]..=[`CLOCK_SECONDS_MAX`], with its exact value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockOutOfRange {
    pub seconds: i128,
    pub nanoseconds: u32,
}

impl ClockReading {
    /// Euclidean normalization, then the range check. A reading outside the
    /// range is reported with its exact value, never clamped or wrapped.
    pub fn checked_time(self) -> Result<ClockTime, ClockOutOfRange> {
        let (seconds, nanoseconds) = match self {
            Self::AtOrAfterOrigin(distance) => {
                (i128::from(distance.as_secs()), distance.subsec_nanos())
            }
            Self::BeforeOrigin(distance) if distance.subsec_nanos() == 0 => {
                (-i128::from(distance.as_secs()), 0)
            }
            Self::BeforeOrigin(distance) => (
                -i128::from(distance.as_secs()) - 1,
                1_000_000_000 - distance.subsec_nanos(),
            ),
        };
        match i64::try_from(seconds) {
            Ok(admitted) if (CLOCK_SECONDS_MIN..=CLOCK_SECONDS_MAX).contains(&admitted) => {
                Ok(ClockTime {
                    seconds: admitted,
                    nanoseconds: i64::from(nanoseconds),
                })
            }
            _ => Err(ClockOutOfRange {
                seconds,
                nanoseconds,
            }),
        }
    }
}

/// One read of the host wall clock on the POSIX timescale.
pub fn read_wall_clock() -> std::io::Result<ClockReading> {
    // One host read; the sign split is the host's own representation.
    Ok(match SystemTime::now().duration_since(UNIX_EPOCH) {
        Ok(after) => ClockReading::AtOrAfterOrigin(after),
        Err(before) => ClockReading::BeforeOrigin(before.duration()),
    })
}

/// One read of a host clock that never runs backwards.
pub fn read_monotonic_clock() -> std::io::Result<ClockReading> {
    // `Instant` exposes no absolute value, so the origin is the first
    // reading this process takes. It is fixed before `now` is read, and
    // `Instant` never runs backwards, so the distance is exact.
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    let origin = *ORIGIN.get_or_init(Instant::now);
    let now = Instant::now();
    Ok(match now.checked_duration_since(origin) {
        Some(after) => ClockReading::AtOrAfterOrigin(after),
        None => ClockReading::BeforeOrigin(origin.duration_since(now)),
    })
}

/// [05-OP-75]'s failure text when the host clock supplies no reading.
pub fn clock_host_error_message(operation: ClockOperation, source: &std::io::Error) -> String {
    format!("{operation}: io: host clock error: {source}")
}

/// [05-OP-75]'s failure text for a reading outside the admitted range.
pub fn clock_out_of_range_message(operation: ClockOperation, range: ClockOutOfRange) -> String {
    let ClockOutOfRange {
        seconds,
        nanoseconds,
    } = range;
    format!(
        "{operation}: io: host reading (seconds {seconds}, nanoseconds {nanoseconds}) \
         is outside seconds {CLOCK_SECONDS_MIN}..{CLOCK_SECONDS_MAX}"
    )
}

/// The complete [05-OP-75] read for compiled host code: one reading,
/// normalized and range checked, or the operation's failure text.
pub fn checked_clock_read(operation: ClockOperation) -> Result<ClockTime, String> {
    let reading = match operation {
        ClockOperation::Wall => read_wall_clock(),
        ClockOperation::Monotonic => read_monotonic_clock(),
    };
    reading
        .map_err(|source| clock_host_error_message(operation, &source))?
        .checked_time()
        .map_err(|range| clock_out_of_range_message(operation, range))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn before_origin_readings_take_euclidean_form() {
        assert_eq!(
            ClockReading::BeforeOrigin(Duration::new(1, 250_000_000)).checked_time(),
            Ok(ClockTime {
                seconds: -2,
                nanoseconds: 750_000_000
            })
        );
        assert_eq!(
            ClockReading::BeforeOrigin(Duration::new(3, 0)).checked_time(),
            Ok(ClockTime {
                seconds: -3,
                nanoseconds: 0
            })
        );
    }

    #[test]
    fn readings_outside_the_range_are_reported_exactly() {
        let beyond = u64::try_from(CLOCK_SECONDS_MAX).unwrap() + 1;
        let range = ClockReading::AtOrAfterOrigin(Duration::new(beyond, 7))
            .checked_time()
            .unwrap_err();
        assert_eq!(
            clock_out_of_range_message(ClockOperation::Wall, range),
            format!(
                "clock_wall_read: io: host reading (seconds {beyond}, nanoseconds 7) \
                 is outside seconds {CLOCK_SECONDS_MIN}..{CLOCK_SECONDS_MAX}"
            )
        );
        let edge = u64::try_from(CLOCK_SECONDS_MAX).unwrap();
        assert!(
            ClockReading::AtOrAfterOrigin(Duration::new(edge, 999_999_999))
                .checked_time()
                .is_ok()
        );
    }

    #[test]
    fn monotonic_reads_never_decrease() {
        let first = checked_clock_read(ClockOperation::Monotonic).unwrap();
        let second = checked_clock_read(ClockOperation::Monotonic).unwrap();
        assert!(
            (first.seconds, first.nanoseconds) <= (second.seconds, second.nanoseconds),
            "{first:?} then {second:?}"
        );
    }
}
