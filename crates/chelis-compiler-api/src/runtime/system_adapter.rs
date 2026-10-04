//! Default host adapter for the covered evaluator filesystem, process, and
//! clock operations. Their policy wrapper checks permission before calling
//! this adapter.

use std::ffi::OsString;
use std::path::Path;

use super::system::{
    EvalClockReading, EvalProcessOutput, EvalSystem, EvalSystemError, EvalSystemOperation,
};

pub(super) struct DefaultEvalSystem;

/// [05-HOST-4]: sort host names *before* converting; reject the whole call
/// on the first unrepresentable name, with reversible host-byte diagnostics.
/// Kept pure so invalid names can be tested on hosts whose filesystem cannot
/// create such entries.
pub(super) fn list_dir_names_to_strings(
    mut names: Vec<OsString>,
    path: &Path,
) -> Result<Vec<String>, EvalSystemError> {
    names.sort_by(|a, b| a.as_encoded_bytes().cmp(b.as_encoded_bytes()));
    names
        .into_iter()
        .map(|name| {
            name.into_string()
                .map_err(|entry| EvalSystemError::InvalidDirectoryName {
                    directory: path.to_path_buf(),
                    entry,
                })
        })
        .collect()
}

impl EvalSystem for DefaultEvalSystem {
    fn read_file(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        std::fs::read_to_string(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemOperation::ReadFile,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn write_file(&mut self, path: &Path, contents: &str) -> Result<(), EvalSystemError> {
        std::fs::write(path, contents).map_err(|source| EvalSystemError::System {
            operation: EvalSystemOperation::WriteFile,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn read_lines_source(&mut self, path: &Path) -> Result<String, EvalSystemError> {
        std::fs::read_to_string(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemOperation::ReadLines,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn read_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError> {
        std::fs::read(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemOperation::ReadBytes,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn file_exists(&mut self, path: &Path) -> Result<bool, EvalSystemError> {
        // Path::exists already maps metadata failures to false.
        Ok(path.exists())
    }

    fn list_dir(&mut self, path: &Path) -> Result<Vec<String>, EvalSystemError> {
        let entries = std::fs::read_dir(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemOperation::ListDir,
            path_or_program: path.display().to_string(),
            source,
        })?;
        let mut names = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|source| EvalSystemError::System {
                operation: EvalSystemOperation::ListDir,
                path_or_program: path.display().to_string(),
                source,
            })?;
            names.push(entry.file_name());
        }
        list_dir_names_to_strings(names, path)
    }

    fn load_mapped_file_bytes(&mut self, path: &Path) -> Result<Vec<u8>, EvalSystemError> {
        std::fs::read(path).map_err(|source| EvalSystemError::System {
            operation: EvalSystemOperation::MmapFile,
            path_or_program: path.display().to_string(),
            source,
        })
    }

    fn run_process(
        &mut self,
        program: &str,
        args: &[String],
    ) -> Result<EvalProcessOutput, EvalSystemError> {
        let output = std::process::Command::new(program)
            .args(args)
            .output()
            .map_err(|source| EvalSystemError::System {
                operation: EvalSystemOperation::ProcessRun,
                path_or_program: program.to_string(),
                source,
            })?;
        Ok(EvalProcessOutput {
            exit_status: output.status.code(),
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }

    fn read_wall_clock(&mut self) -> std::io::Result<EvalClockReading> {
        chelis_runtime::host_clock::read_wall_clock()
    }

    fn read_monotonic_clock(&mut self) -> std::io::Result<EvalClockReading> {
        chelis_runtime::host_clock::read_monotonic_clock()
    }
}
