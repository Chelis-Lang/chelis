//! Policy-checked, evaluator-only filesystem and process access.
//!
//! Compiled artifacts and the runtime C ABI do not use this port. The policy
//! lives in the wrapper, not the default adapter or individual builtin arms.

use std::ffi::OsString;
use std::fmt;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvalSystemCapability {
    Filesystem,
    Process,
}

impl fmt::Display for EvalSystemCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Filesystem => "Filesystem",
            Self::Process => "Process",
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
}

impl EvalSystemOperation {
    pub(crate) fn capability(self) -> EvalSystemCapability {
        match self {
            Self::ProcessRun => EvalSystemCapability::Process,
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
}

impl fmt::Display for EvalSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EvalSystemPolicy {
    pub(crate) filesystem: bool,
    pub(crate) process: bool,
}

impl EvalSystemPolicy {
    pub(crate) const ALLOW_ALL: Self = Self {
        filesystem: true,
        process: true,
    };
    pub(crate) const DENY_ALL: Self = Self {
        filesystem: false,
        process: false,
    };

    fn permits(self, capability: EvalSystemCapability) -> bool {
        match capability {
            EvalSystemCapability::Filesystem => self.filesystem,
            EvalSystemCapability::Process => self.process,
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
}
