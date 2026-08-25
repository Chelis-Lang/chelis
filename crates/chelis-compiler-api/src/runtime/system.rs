//! Evaluator-only system boundary (OpenSpec `add-eval-system-boundary`).
//!
//! `runtime/eval.rs` performs filesystem and subprocess operations for eight
//! covered builtins (`read_file`, `write_file`, `read_lines`, `read_bytes`,
//! `file_exists`, `list_dir`, `mmap_file`, `process_run`). Before this
//! module, those operations called `std::fs`, `std::path::Path::exists`,
//! and `std::process::Command` directly from evaluator dispatch, which made
//! deterministic fake-system tests impossible and left no single policy
//! boundary for evaluator host access.
//!
//! This module defines the closed capability vocabulary
//! ([`EvalSystemCapability`]), the closed operation identity
//! ([`EvalSystemOperation`]), the typed port ([`EvalSystem`]), and the
//! mandatory policy wrapper ([`EvalSystemBoundary`]) that every
//! `EvalContext` must carry. The wrapper checks a fixed capability before
//! delegating to the injected adapter, so a refused operation never reaches
//! the adapter and a new call site cannot forget its check (design D3).
//!
//! The **only** module permitted direct `std::fs` / `std::path::Path` /
//! `std::process::Command` access is `runtime/system_adapter.rs`
//! (`scripts/eval_system_guard.py` enforces this). This module and
//! `runtime/eval.rs` must not call those APIs directly.
//!
//! Scope (see `openspec/changes/add-eval-system-boundary/specs/eval-system-boundary/spec.md`):
//! this boundary governs only the compiler API evaluator. It does not
//! mediate generated C/HIP/Metal artifacts or the `chelis-runtime` C ABI.

use std::fmt;
use std::path::Path;

/// Closed evaluator capability classification. Every covered operation
/// belongs to exactly one of these two capabilities (design D1: no empty
/// future categories).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum EvalSystemCapability {
    Filesystem,
    Process,
}

impl EvalSystemCapability {
    fn as_str(self) -> &'static str {
        match self {
            EvalSystemCapability::Filesystem => "Filesystem",
            EvalSystemCapability::Process => "Process",
        }
    }
}

impl fmt::Display for EvalSystemCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Closed operation identity for the eight covered evaluator builtins
/// (spec "One evaluator system boundary" requirement).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    fn as_str(self) -> &'static str {
        match self {
            EvalSystemOperation::ReadFile => "read_file",
            EvalSystemOperation::WriteFile => "write_file",
            EvalSystemOperation::ReadLines => "read_lines",
            EvalSystemOperation::ReadBytes => "read_bytes",
            EvalSystemOperation::FileExists => "file_exists",
            EvalSystemOperation::ListDir => "list_dir",
            EvalSystemOperation::MmapFile => "mmap_file",
            EvalSystemOperation::ProcessRun => "process_run",
        }
    }

    fn capability(self) -> EvalSystemCapability {
        match self {
            EvalSystemOperation::ReadFile
            | EvalSystemOperation::WriteFile
            | EvalSystemOperation::ReadLines
            | EvalSystemOperation::ReadBytes
            | EvalSystemOperation::FileExists
            | EvalSystemOperation::ListDir
            | EvalSystemOperation::MmapFile => EvalSystemCapability::Filesystem,
            EvalSystemOperation::ProcessRun => EvalSystemCapability::Process,
        }
    }
}

impl fmt::Display for EvalSystemOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Closed identity for operations that can produce a system error.
///
/// `file_exists` is absent because its contract maps metadata errors to
/// `false`. This type makes that invalid error state unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum EvalSystemIoOperation {
    ReadFile,
    WriteFile,
    ReadLines,
    ReadBytes,
    ListDir,
    MmapFile,
    ProcessRun,
}

/// Typed raw output from [`EvalSystem::run_process`]. Exit-status
/// conversion (missing status -> `-1`) and output decoding (lossy UTF-8)
/// stay in pure evaluator code (`runtime/eval.rs`), not the adapter
/// (design D2/D4): the adapter hands back exactly what the operating
/// system returned.
#[derive(Debug, Clone)]
pub(crate) struct EvalProcessOutput {
    pub(crate) exit_status: Option<i32>,
    pub(crate) stdout: Vec<u8>,
    pub(crate) stderr: Vec<u8>,
}

/// The evaluator system boundary's error type (design D4). The operation
/// identity is closed, the refused-capability variant is distinct from a
/// real system error, and the system-error variant owns the underlying
/// `std::io::Error` without reducing it to `ErrorKind`.
#[derive(Debug)]
pub(crate) enum EvalSystemError {
    /// The policy refused this operation's capability before the adapter
    /// could run it.
    Refused {
        operation: EvalSystemOperation,
        capability: EvalSystemCapability,
    },
    /// The adapter ran the operation and the operating system reported a
    /// failure. `path_or_program` is the path for a filesystem operation or
    /// the program name for `process_run`, matching the exact templates in
    /// `specs/eval-system-boundary/spec.md`.
    System {
        operation: EvalSystemIoOperation,
        path_or_program: String,
        source: std::io::Error,
    },
}

impl fmt::Display for EvalSystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvalSystemError::Refused {
                operation,
                capability,
            } => write!(
                f,
                "{operation} refused: {capability} capability is not permitted"
            ),
            EvalSystemError::System {
                operation,
                path_or_program,
                source,
            } => match operation {
                EvalSystemIoOperation::ReadFile => {
                    write!(f, "read_file failed for `{path_or_program}`: {source}")
                }
                EvalSystemIoOperation::WriteFile => {
                    write!(f, "write_file failed for `{path_or_program}`: {source}")
                }
                EvalSystemIoOperation::ReadLines => {
                    write!(f, "read_lines failed for `{path_or_program}`: {source}")
                }
                EvalSystemIoOperation::ReadBytes => {
                    write!(f, "read_bytes failed for `{path_or_program}`: {source}")
                }
                EvalSystemIoOperation::ListDir => {
                    write!(f, "list_dir failed for `{path_or_program}`: {source}")
                }
                EvalSystemIoOperation::MmapFile => {
                    write!(f, "mmap_file failed for `{path_or_program}`: {source}")
                }
                EvalSystemIoOperation::ProcessRun => write!(
                    f,
                    "process_run failed to spawn `{path_or_program}`: {source}"
                ),
            },
        }
    }
}

impl std::error::Error for EvalSystemError {}

/// The one conversion from a typed boundary error to the current evaluator
/// error text (`Result<RuntimeValue, String>`). Every call site in
/// `runtime/eval.rs` uses `?`, which invokes this via `From`.
impl From<EvalSystemError> for String {
    fn from(error: EvalSystemError) -> String {
        error.to_string()
    }
}

/// The injected evaluator system port (design D2). One method per
/// operation class makes an invalid request/response pair unrepresentable;
/// pure transformations (line splitting, output decoding, exit-code
/// conversion, `RuntimeValue` construction) stay outside the adapter.
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

/// Typed allow/refuse decision for the two evaluator capabilities (design
/// D3). Construct with [`EvalSystemPolicy::permissive`] (program
/// evaluation) or [`EvalSystemPolicy::deny_all`] (invariant predicate
/// evaluation, `spec/04-type-system.md` §2.5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EvalSystemPolicy {
    filesystem: bool,
    process: bool,
}

impl EvalSystemPolicy {
    pub(crate) fn permissive() -> Self {
        Self {
            filesystem: true,
            process: true,
        }
    }

    pub(crate) fn deny_all() -> Self {
        Self {
            filesystem: false,
            process: false,
        }
    }

    fn permits(self, capability: EvalSystemCapability) -> bool {
        match capability {
            EvalSystemCapability::Filesystem => self.filesystem,
            EvalSystemCapability::Process => self.process,
        }
    }
}

/// The mandatory policy wrapper (design D3). Every `EvalContext` receives a
/// `&mut EvalSystemBoundary`; there is no construction path that skips the
/// capability check, because the wrapper — not evaluator dispatch — owns
/// both the policy and the adapter. Each method checks its fixed capability
/// before calling the adapter and returns `EvalSystemError::Refused`
/// without executing or substituting a value on refusal.
pub(crate) struct EvalSystemBoundary {
    policy: EvalSystemPolicy,
    adapter: Box<dyn EvalSystem>,
}

impl EvalSystemBoundary {
    /// Program evaluation's default: both capabilities permitted, backed by
    /// the real operating system adapter (`system_adapter::DefaultEvalSystem`).
    pub(crate) fn permissive() -> Self {
        Self {
            policy: EvalSystemPolicy::permissive(),
            adapter: Box::new(super::system_adapter::DefaultEvalSystem),
        }
    }

    /// Invariant predicate evaluation's boundary: both capabilities
    /// refused, enforcing `spec/04-type-system.md` §2.5.1's ban on effects
    /// in invariant predicates. Still backed by the real adapter (which the
    /// policy check keeps unreachable) rather than a special-cased no-op
    /// path, so refusal always goes through the same check.
    pub(crate) fn deny_all() -> Self {
        Self {
            policy: EvalSystemPolicy::deny_all(),
            adapter: Box::new(super::system_adapter::DefaultEvalSystem),
        }
    }

    /// Construct a test boundary over an arbitrary policy and adapter.
    #[cfg(test)]
    pub(super) fn with_adapter(policy: EvalSystemPolicy, adapter: Box<dyn EvalSystem>) -> Self {
        Self { policy, adapter }
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
        self.adapter.read_file(path)
    }

    pub(crate) fn write_file(
        &mut self,
        path: &Path,
        contents: &str,
    ) -> Result<(), EvalSystemError> {
        self.check(EvalSystemOperation::WriteFile)?;
        self.adapter.write_file(path, contents)
    }

    pub(crate) fn read_lines_source(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        self.check(EvalSystemOperation::ReadLines)?;
        self.adapter.read_lines_source(path)
    }

    pub(crate) fn read_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError> {
        self.check(EvalSystemOperation::ReadBytes)?;
        self.adapter.read_bytes(path)
    }

    pub(crate) fn file_exists(&mut self, path: &Path) -> Result<bool, EvalSystemError> {
        self.check(EvalSystemOperation::FileExists)?;
        self.adapter.file_exists(path)
    }

    pub(crate) fn list_dir(&mut self, path: &Path) -> Result<Vec<String>, EvalSystemError> {
        self.check(EvalSystemOperation::ListDir)?;
        self.adapter.list_dir(path)
    }

    pub(crate) fn load_mapped_file_bytes(
        &mut self,
        path: &Path,
    ) -> Result<Vec<u8>, EvalSystemError> {
        self.check(EvalSystemOperation::MmapFile)?;
        self.adapter.load_mapped_file_bytes(path)
    }

    pub(crate) fn run_process(
        &mut self,
        program: &str,
        args: &[String],
    ) -> Result<EvalProcessOutput, EvalSystemError> {
        self.check(EvalSystemOperation::ProcessRun)?;
        self.adapter.run_process(program, args)
    }
}
