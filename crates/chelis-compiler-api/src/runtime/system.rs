//! Policy-checked, evaluator-only filesystem, process, and clock access.
//!
//! Compiled artifacts and the runtime C ABI do not use this port. The policy
//! lives in the wrapper, not the default adapter or individual builtin arms.

use std::ffi::OsString;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvalSystemCapability {
    Filesystem,
    Process,
    Clock,
}

impl fmt::Display for EvalSystemCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Filesystem => "Filesystem",
            Self::Process => "Process",
            Self::Clock => "Clock",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvalSystemOperation {
    ReadFile,
    WriteFile,
    ReadLines,
    ReadBytes,
    FileExists,
    ListDir,
    MmapFile,
    ProcessRun,
    ClockWallRead,
    ClockMonotonicRead,
}

impl EvalSystemOperation {
    pub(crate) fn capability(self) -> EvalSystemCapability {
        match self {
            Self::ProcessRun => EvalSystemCapability::Process,
            Self::ClockWallRead | Self::ClockMonotonicRead => EvalSystemCapability::Clock,
            Self::ReadFile
            | Self::WriteFile
            | Self::ReadLines
            | Self::ReadBytes
            | Self::FileExists
            | Self::ListDir
            | Self::MmapFile => EvalSystemCapability::Filesystem,
        }
    }
}

impl fmt::Display for EvalSystemOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ReadFile => "read_file",
            Self::WriteFile => "write_file",
            Self::ReadLines => "read_lines",
            Self::ReadBytes => "read_bytes",
            Self::FileExists => "file_exists",
            Self::ListDir => "list_dir",
            Self::MmapFile => "mmap_file",
            Self::ProcessRun => "process_run",
            Self::ClockWallRead => "clock_wall_read",
            Self::ClockMonotonicRead => "clock_monotonic_read",
        })
    }
}

#[derive(Debug)]
pub(crate) enum EvalSystemError {
    Refused {
        operation: EvalSystemOperation,
        capability: EvalSystemCapability,
    },
    System {
        operation: EvalSystemOperation,
        path_or_program: String,
        source: std::io::Error,
    },
    /// [05-HOST-4] rejects the entire listing, identifying the first invalid
    /// host name after byte ordering; this is not an OS/read error.
    InvalidDirectoryName { directory: PathBuf, entry: OsString },
    /// [05-OP-75]: the host clock could not supply a reading.
    ClockHost {
        operation: EvalSystemOperation,
        source: std::io::Error,
    },
    /// [05-OP-75]: a reading, in Euclidean form, whose seconds lie outside
    /// [`CLOCK_SECONDS_MIN`]..=[`CLOCK_SECONDS_MAX`].
    ClockOutOfRange {
        operation: EvalSystemOperation,
        seconds: i128,
        nanoseconds: u32,
    },
}

impl fmt::Display for EvalSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // [05-OP-75] fixes every clock failure, a refusal included, as
            // `<operation>: io: <detail>`.
            Self::Refused {
                operation,
                capability: capability @ EvalSystemCapability::Clock,
            } => write!(
                f,
                "{operation}: io: {capability} capability is not permitted"
            ),
            Self::Refused {
                operation,
                capability,
            } => write!(
                f,
                "{operation} refused: {capability} capability is not permitted"
            ),
            Self::System {
                operation: EvalSystemOperation::ProcessRun,
                path_or_program,
                source,
            } => write!(
                f,
                "process_run failed to spawn `{path_or_program}`: {source}"
            ),
            Self::System {
                operation,
                path_or_program,
                source,
            } => write!(f, "{operation} failed for `{path_or_program}`: {source}"),
            Self::InvalidDirectoryName { directory, entry } => write!(
                f,
                "IO trap in list_dir: directory b\"{}\", entry b\"{}\": name is not valid UTF-8",
                directory.as_os_str().as_encoded_bytes().escape_ascii(),
                entry.as_encoded_bytes().escape_ascii(),
            ),
            Self::ClockHost { operation, source } => {
                write!(f, "{operation}: io: host clock error: {source}")
            }
            Self::ClockOutOfRange {
                operation,
                seconds,
                nanoseconds,
            } => write!(
                f,
                "{operation}: io: host reading (seconds {seconds}, nanoseconds {nanoseconds}) \
                 is outside seconds {CLOCK_SECONDS_MIN}..{CLOCK_SECONDS_MAX}"
            ),
        }
    }
}

impl std::error::Error for EvalSystemError {}

impl From<EvalSystemError> for String {
    fn from(error: EvalSystemError) -> Self {
        error.to_string()
    }
}

pub(crate) struct EvalProcessOutput {
    pub(crate) exit_status: Option<i32>,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
}

/// [05-OP-75]: the least and greatest admitted reading second, the unix
/// seconds whose civil reading at every UTC offset under one day lies in
/// years -9999 through 9999.
pub(crate) const CLOCK_SECONDS_MIN: i64 = -377_705_030_401;
pub(crate) const CLOCK_SECONDS_MAX: i64 = 253_402_214_400;

/// One host clock reading as the host reports it: a distance on one side of
/// the clock's origin. The boundary, not the adapter, normalizes and range
/// checks it, so an injected clock is held to the same contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvalClockReading {
    AtOrAfterOrigin(Duration),
    BeforeOrigin(Duration),
}

/// A checked [05-OP-75] reading in Euclidean form: `nanoseconds` lies in
/// `0..1_000_000_000` and `seconds` in the admitted range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EvalClockTime {
    pub(crate) seconds: i64,
    pub(crate) nanoseconds: i64,
}

impl EvalClockReading {
    /// Euclidean normalization, then the range check. A reading outside the
    /// range is reported with its exact value, never clamped or wrapped.
    fn checked_time(
        self,
        operation: EvalSystemOperation,
    ) -> Result<EvalClockTime, EvalSystemError> {
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
                Ok(EvalClockTime {
                    seconds: admitted,
                    nanoseconds: i64::from(nanoseconds),
                })
            }
            _ => Err(EvalSystemError::ClockOutOfRange {
                operation,
                seconds,
                nanoseconds,
            }),
        }
    }
}

/// Typed adapter results prevent an operation from receiving another
/// operation's payload. Pure line splitting and process decoding stay in the
/// evaluator; filename validation is part of the directory system boundary.
pub(crate) trait EvalSystem {
    fn read_file(&mut self, path: &Path) -> Result<String, EvalSystemError>;
    fn write_file(&mut self, path: &Path, contents: &str) -> Result<(), EvalSystemError>;
    fn read_lines_source(&mut self, path: &Path) -> Result<String, EvalSystemError>;
    fn read_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError>;
    fn file_exists(&mut self, path: &Path) -> Result<bool, EvalSystemError>;
    fn list_dir(&mut self, path: &Path) -> Result<Vec<String>, EvalSystemError>;
    fn load_mapped_file_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError>;
    fn run_process(
        &mut self,
        program: &str,
        args: &[String],
    ) -> Result<EvalProcessOutput, EvalSystemError>;
    fn read_wall_clock(&mut self) -> Result<EvalClockReading, EvalSystemError>;
    fn read_monotonic_clock(&mut self) -> Result<EvalClockReading, EvalSystemError>;
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EvalSystemPolicy {
    pub(crate) filesystem: bool,
    pub(crate) process: bool,
    pub(crate) clock: bool,
}

impl EvalSystemPolicy {
    pub(crate) const ALLOW_ALL: Self = Self {
        filesystem: true,
        process: true,
        clock: true,
    };
    pub(crate) const DENY_ALL: Self = Self {
        filesystem: false,
        process: false,
        clock: false,
    };

    fn permits(self, capability: EvalSystemCapability) -> bool {
        match capability {
            EvalSystemCapability::Filesystem => self.filesystem,
            EvalSystemCapability::Process => self.process,
            EvalSystemCapability::Clock => self.clock,
        }
    }
}

/// Program and invariant contexts own a real zero-sized adapter. Tests can
/// inject a fake without imposing a heap allocation on normal evaluation.
enum Adapter {
    Default(super::system_adapter::DefaultEvalSystem),
    #[cfg(test)]
    Injected(Box<dyn EvalSystem>),
}

/// The only system port a runtime evaluator can carry: policy checks happen
/// before every adapter call and cannot be omitted by a builtin arm.
pub(crate) struct EvalSystemBoundary {
    policy: EvalSystemPolicy,
    adapter: Adapter,
}

impl EvalSystemBoundary {
    pub(crate) fn permissive() -> Self {
        Self::with_policy(EvalSystemPolicy::ALLOW_ALL)
    }

    pub(crate) fn deny_all() -> Self {
        Self::with_policy(EvalSystemPolicy::DENY_ALL)
    }

    pub(crate) fn with_policy(policy: EvalSystemPolicy) -> Self {
        Self {
            policy,
            adapter: Adapter::Default(super::system_adapter::DefaultEvalSystem),
        }
    }

    #[cfg(test)]
    pub(crate) fn with_adapter(policy: EvalSystemPolicy, adapter: Box<dyn EvalSystem>) -> Self {
        Self {
            policy,
            adapter: Adapter::Injected(adapter),
        }
    }

    // The shipped adapter is concrete: besides avoiding dynamic dispatch on
    // every host operation, this keeps compiled String/Vec results attributable
    // to their actual implementation in the wire publication census.
    #[cfg(not(test))]
    fn adapter_mut(&mut self) -> &mut super::system_adapter::DefaultEvalSystem {
        let Adapter::Default(adapter) = &mut self.adapter;
        adapter
    }

    #[cfg(test)]
    fn adapter_mut(&mut self) -> &mut dyn EvalSystem {
        match &mut self.adapter {
            Adapter::Default(adapter) => adapter,
            Adapter::Injected(adapter) => adapter.as_mut(),
        }
    }

    fn check(&self, operation: EvalSystemOperation) -> Result<(), EvalSystemError> {
        let capability = operation.capability();
        if self.policy.permits(capability) {
            Ok(())
        } else {
            Err(EvalSystemError::Refused {
                operation,
                capability,
            })
        }
    }

    pub(crate) fn read_file(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        self.check(EvalSystemOperation::ReadFile)?;
        self.adapter_mut().read_file(path)
    }

    pub(crate) fn write_file(
        &mut self,
        path: &Path,
        contents: &str,
    ) -> Result<(), EvalSystemError> {
        self.check(EvalSystemOperation::WriteFile)?;
        self.adapter_mut().write_file(path, contents)
    }

    pub(crate) fn read_lines_source(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        self.check(EvalSystemOperation::ReadLines)?;
        self.adapter_mut().read_lines_source(path)
    }

    pub(crate) fn read_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError> {
        self.check(EvalSystemOperation::ReadBytes)?;
        self.adapter_mut().read_bytes(path)
    }

    pub(crate) fn file_exists(&mut self, path: &Path) -> Result<bool, EvalSystemError> {
        self.check(EvalSystemOperation::FileExists)?;
        self.adapter_mut().file_exists(path)
    }

    pub(crate) fn list_dir(&mut self, path: &Path) -> Result<Vec<String>, EvalSystemError> {
        self.check(EvalSystemOperation::ListDir)?;
        self.adapter_mut().list_dir(path)
    }

    pub(crate) fn load_mapped_file_bytes(
        &mut self,
        path: &Path,
    ) -> Result<Vec<u8>, EvalSystemError> {
        self.check(EvalSystemOperation::MmapFile)?;
        self.adapter_mut().load_mapped_file_bytes(path)
    }

    pub(crate) fn run_process(
        &mut self,
        program: &str,
        args: &[String],
    ) -> Result<EvalProcessOutput, EvalSystemError> {
        self.check(EvalSystemOperation::ProcessRun)?;
        self.adapter_mut().run_process(program, args)
    }

    pub(crate) fn clock_wall_read(&mut self) -> Result<EvalClockTime, EvalSystemError> {
        self.check(EvalSystemOperation::ClockWallRead)?;
        self.adapter_mut()
            .read_wall_clock()?
            .checked_time(EvalSystemOperation::ClockWallRead)
    }

    pub(crate) fn clock_monotonic_read(&mut self) -> Result<EvalClockTime, EvalSystemError> {
        self.check(EvalSystemOperation::ClockMonotonicRead)?;
        self.adapter_mut()
            .read_monotonic_clock()?
            .checked_time(EvalSystemOperation::ClockMonotonicRead)
    }
}
