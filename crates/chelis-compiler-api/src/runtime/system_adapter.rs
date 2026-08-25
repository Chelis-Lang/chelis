//! The default [`EvalSystem`](super::system::EvalSystem) adapter.
//!
//! This is the **only** module in the evaluator runtime permitted to call
//! `std::fs`, `std::path::Path::exists`, or `std::process::Command`
//! directly (`scripts/eval_system_guard.py` enforces this with a source
//! scan that excludes exactly this file). Every other evaluator runtime
//! module reaches the operating system only through
//! [`super::system::EvalSystemBoundary`].
//!
//! The adapter preserves the exact pre-boundary behavior (design D4):
//! `Path::exists` semantics including its `false` result for an
//! inaccessible-metadata path, `list_dir`'s unsorted `std::fs::read_dir`
//! iteration order with abort-on-first-error, and `process_run`'s
//! shell-free, direct-argv `Command` construction. Pure transformations
//! (line splitting, output decoding, exit-code conversion, `RuntimeValue`
//! construction) do not live here — they stay in `runtime/eval.rs` per D2.

use std::path::Path;

use super::system::{EvalProcessOutput, EvalSystem, EvalSystemError, EvalSystemIoOperation};

/// The real operating-system adapter. Constructed by
/// `EvalSystemBoundary::permissive`/`deny_all`; the policy check in the
/// boundary decides whether any of these methods ever run.
pub(super) struct DefaultEvalSystem;

impl EvalSystem for DefaultEvalSystem {
    fn read_file(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        std::fs::read_to_string(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemIoOperation::ReadFile,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn write_file(&mut self, path: &Path, contents: &str) -> Result<(), EvalSystemError> {
        std::fs::write(path, contents).map_err(|source| EvalSystemError::System {
            operation: EvalSystemIoOperation::WriteFile,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn read_lines_source(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        std::fs::read_to_string(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemIoOperation::ReadLines,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn read_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError> {
        std::fs::read(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemIoOperation::ReadBytes,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn file_exists(&mut self, path: &Path) -> Result<bool, EvalSystemError> {
        // `Path::exists` itself already folds a metadata access error (not
        // just "absent") into `false` (design D4); there is no error path
        // to convert here.
        Ok(path.exists())
    }

    fn list_dir(&mut self, path: &Path) -> Result<Vec<String>, EvalSystemError> {
        let entries = std::fs::read_dir(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemIoOperation::ListDir,
            path_or_program: path.display().to_string(),
            source,
        })?;
        let mut out = Vec::new();
        for entry in entries {
            // Abort on the first directory-entry error (D4), sharing the
            // same `list_dir failed for ...` template as a directory-open
            // error.
            let entry = entry.map_err(|source| EvalSystemError::System {
                operation: EvalSystemIoOperation::ListDir,
                path_or_program: path.display().to_string(),
                source,
            })?;
            // No sort: preserve `std::fs::read_dir`'s source iteration
            // order (D4).
            out.push(entry.file_name().to_string_lossy().into_owned());
        }
        Ok(out)
    }

    fn load_mapped_file_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError> {
        std::fs::read(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemIoOperation::MmapFile,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn run_process(
        &mut self,
        program: &str,
        args: &[String],
    ) -> Result<EvalProcessOutput, EvalSystemError> {
        // Direct argv via `Command::args` -- no shell, no glob expansion,
        // no `$VAR`/backtick interpolation (Hull Phase 0a note preserved
        // from the pre-boundary `process_run` implementation).
        let output = std::process::Command::new(program)
            .args(args)
            .output()
            .map_err(|source| EvalSystemError::System {
                operation: EvalSystemIoOperation::ProcessRun,
                path_or_program: program.to_string(),
                source,
            })?;
        Ok(EvalProcessOutput {
            exit_status: output.status.code(),
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}
