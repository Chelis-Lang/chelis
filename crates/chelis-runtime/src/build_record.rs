//! The declared source inputs this runtime was compiled from
//! (`spec/08-backends.md` §2.1), recorded by `build.rs` when the runtime is
//! compiled.

/// One entry per line, with paths relative to the workspace root and `/` as
/// the separator:
///
/// - `dir <path>`: a declared directory, whose every file whose name does not
///   begin with `.` is recorded;
/// - `sha256 <hex digest> <path>`: a recorded file and the SHA-256 of its bytes;
/// - `unavailable <path>`: a required root the compilation could not find. A
///   record with such a line describes no checkout.
///
/// Directory lines come first, sorted, then file lines, sorted by path.
pub const SOURCES: &str = include_str!(concat!(env!("OUT_DIR"), "/build_record.txt"));
