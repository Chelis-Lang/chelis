//! The `chelis check <dir>` envelope (spec/04 § Directory mode, chelis#1678).
//!
//! Directory mode used to be assembled by `format!` in the CLI: each file's
//! report was rendered to a string and spliced into a hand-written wrapper, so
//! the envelope had two producers and no type. An empty directory printed a
//! different shape from a populated one, and a directory the walk could not
//! read printed nothing at all. This module is the envelope's one typed value
//! ([04-FIT-19]).
//!
//! The rules that are easy to get wrong by accident are enforced here, where
//! the value is built, rather than left to the caller:
//!
//! - an `empty_corpus` diagnostic appears exactly when the walk completed and
//!   found nothing, never beside a walk failure ([04-FIT-24]);
//! - an entry's `file` is a `/`-joined relative path or it is not an entry,
//!   never a lossy rendering ([04-FIT-21]);
//! - the exit rule reads every error list in the document ([04-FIT-25]).

use std::fmt;
use std::path::{Component, Path};

use chelis_vocab::DiagnosticKind;
use serde::{Deserialize, Serialize};

use super::numbers::UnitInterval;
use super::{CheckResult, Diagnostic, WireCheckResult, WireDiagnostic};

/// One checked file: its path relative to the directory target, and its report.
#[derive(Debug, Clone, Serialize)]
pub struct CheckDirectoryEntry {
    file: EntryPath,
    report: CheckResult,
}

impl CheckDirectoryEntry {
    pub fn new(file: EntryPath, report: CheckResult) -> Self {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        Self { file, report }
    }

    pub fn file(&self) -> &str {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        &self.file.0
    }

    pub fn report(&self) -> &CheckResult {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        &self.report
    }
}

/// An entry's `file`: a path relative to the target, `/`-joined, valid UTF-8.
///
/// Its own type so that a path is validated before its file is checked: a
/// path the envelope cannot write is a walk failure, and checking the file
/// first would spend a whole check on a report with nowhere to go.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct EntryPath(String);

impl EntryPath {
    /// The entry's `file` for `walked`, a path the walk reached below `target`.
    ///
    /// [04-FIT-21]: the components are joined by `/` on every host, and a path
    /// that is not valid UTF-8 is refused rather than written with replacement
    /// characters. The caller reports a refusal as a walk failure.
    ///
    /// Taking the walked path rather than an already-relative one keeps the
    /// stripping beside the validation, so a refusal names the same path every
    /// other walk failure names.
    pub fn relative_to(target: &Path, walked: &Path) -> Result<Self, UnrepresentablePath> {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        let Ok(relative) = walked.strip_prefix(target) else {
            // The walk only produces paths below its target, so this is a
            // defect in the caller. It is still reported, not panicked on:
            // the document must survive it ([04-FIT-12]).
            return Err(UnrepresentablePath::not_relative(walked));
        };
        let mut file = String::new();
        for component in relative.components() {
            let Component::Normal(name) = component else {
                return Err(UnrepresentablePath::not_relative(walked));
            };
            let Some(name) = name.to_str() else {
                return Err(UnrepresentablePath::not_utf8(walked));
            };
            if !file.is_empty() {
                file.push('/');
            }
            file.push_str(name);
        }
        if file.is_empty() {
            return Err(UnrepresentablePath::not_relative(walked));
        }
        Ok(Self(file))
    }

    pub fn as_str(&self) -> &str {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        &self.0
    }
}

/// A path rendered as [05-HOST-4] renders one in a diagnostic: reversible
/// escaped host bytes, delimited by `b"` and `"`.
///
/// A diagnostic that names a path must not substitute for the bytes it could
/// not represent, or two names the walk rejected become one unrecoverable
/// message. `Path::display()` substitutes U+FFFD, so it cannot be used here.
/// spec/05 [05-HOST-4] already fixed this rendering for `list_dir`'s own
/// conversion diagnostic, and spec/04 [04-FIT-23] adopts it.
///
/// `escape_ascii` is the same primitive `list_dir`'s renderer uses
/// (`runtime::eval::list_dir_names_to_strings`), so the two cannot drift into
/// spelling one byte differently.
pub fn escaped_path(path: &Path) -> String {
    let _fp_env = chelis_runtime::FpEnvGuard::enter();
    format!(
        "b\"{}\"",
        path.as_os_str().as_encoded_bytes().escape_ascii()
    )
}

/// A path [04-FIT-21] cannot write as an entry's `file`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnrepresentablePath {
    message: String,
}

impl UnrepresentablePath {
    fn not_utf8(relative: &Path) -> Self {
        Self {
            message: format!(
                "cannot report {}: its path is not valid UTF-8",
                escaped_path(relative)
            ),
        }
    }

    fn not_relative(relative: &Path) -> Self {
        Self {
            message: format!(
                "cannot report {}: it is not a path below the target",
                escaped_path(relative)
            ),
        }
    }
}

impl fmt::Display for UnrepresentablePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for UnrepresentablePath {}

/// The `chelis check <dir>` document.
///
/// `errors` is private, and so is every way to add to it: the only
/// diagnostics it can hold are the two kinds [`Self::from_walk`] and
/// [`EmptyWalk::into_report`] mint. A per-file failure belongs in that file's
/// report, not here.
#[derive(Debug, Clone, Serialize)]
pub struct CheckDirectoryReport {
    files: Vec<CheckDirectoryEntry>,
    errors: Vec<Diagnostic>,
}

impl CheckDirectoryReport {
    /// Assemble the envelope from a finished walk.
    ///
    /// `walk_failures` holds one message per failure of the walk itself: a
    /// directory it could not read, an entry it could not resolve, or a path
    /// it could not represent ([04-FIT-23]). Each becomes one
    /// `directory_walk_error` diagnostic.
    ///
    /// A walk that completed and found nothing is not an envelope yet: it is
    /// an [`EmptyWalk`], which only this function can produce, and which the
    /// caller turns into the `empty_corpus` failure with a description
    /// ([04-FIT-24]). So the empty-corpus diagnostic cannot appear beside a
    /// walk failure -- a failed walk has not established emptiness, and its
    /// own diagnostic already explains the result -- and the description,
    /// which costs a second walk, is computed only when it is needed.
    pub fn from_walk(
        files: Vec<CheckDirectoryEntry>,
        walk_failures: Vec<String>,
    ) -> Result<Self, EmptyWalk> {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        if files.is_empty() && walk_failures.is_empty() {
            return Err(EmptyWalk(()));
        }
        let errors = walk_failures
            .into_iter()
            .map(|message| Diagnostic::new(DiagnosticKind::DirectoryWalkError, message, severity()))
            .collect();
        Ok(Self { files, errors })
    }

    pub fn files(&self) -> &[CheckDirectoryEntry] {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        &self.files
    }

    pub fn errors(&self) -> &[Diagnostic] {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        &self.errors
    }

    /// Whether the document carries any error, at either level.
    ///
    /// [04-FIT-25]: directory mode exits `0` if and only if this is false.
    /// Reading only the envelope's own `errors` would pass a directory full of
    /// type errors; reading only the entries would pass an unreadable one.
    pub fn has_errors(&self) -> bool {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        !self.errors.is_empty()
            || self
                .files
                .iter()
                .any(|entry| !entry.report.errors.is_empty())
    }
}

/// Proof that a walk completed and found no checkable file ([04-FIT-24]).
///
/// Its field is private, so only [`CheckDirectoryReport::from_walk`] makes
/// one, and only for a walk with no entries and no failures.
#[derive(Debug)]
pub struct EmptyWalk(());

impl EmptyWalk {
    /// The envelope for an empty corpus: no entries, one `empty_corpus`
    /// diagnostic. `description` names the target and reports what the
    /// exclusions removed.
    pub fn into_report(self, description: String) -> CheckDirectoryReport {
        let _fp_env = chelis_runtime::FpEnvGuard::enter();
        CheckDirectoryReport {
            files: Vec::new(),
            errors: vec![Diagnostic::new(
                DiagnosticKind::EmptyCorpus,
                description,
                severity(),
            )],
        }
    }
}

/// Both directory failures mean the verdict covers less than the caller asked
/// for, so both carry the top of the scale. Severity orders; it never decides
/// presence or the exit status ([04-FIT-18]).
fn severity() -> UnitInterval {
    UnitInterval::new(1.0).expect("constant severity")
}

/// Consumer-side shape of the directory envelope.
///
/// Like [`WireCheckResult`], it has no conversion back into the producer
/// type: a decoded document cannot be re-emitted as compiler output.
#[derive(Debug, Clone, Deserialize)]
pub struct WireCheckDirectoryReport {
    pub files: Vec<WireCheckDirectoryEntry>,
    pub errors: Vec<WireDiagnostic>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct WireCheckDirectoryEntry {
    pub file: String,
    pub report: WireCheckResult,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FitnessComponents;
    use crate::schema::numbers::NonnegativeCount;
    use std::path::PathBuf;

    fn report(errors: Vec<Diagnostic>) -> CheckResult {
        let one = UnitInterval::new(1.0).unwrap();
        let zero = NonnegativeCount::try_from(0usize).unwrap();
        CheckResult {
            score: one,
            components: FitnessComponents {
                parse: one,
                structure: one,
                names: one,
                types: one,
            },
            typed_nodes: zero,
            untyped_nodes: zero,
            total_nodes: zero,
            unresolved_names: Vec::new(),
            inferred_signatures: None,
            errors,
        }
    }

    fn entry(path: &str, errors: Vec<Diagnostic>) -> CheckDirectoryEntry {
        CheckDirectoryEntry::new(
            EntryPath::relative_to(Path::new("/t"), &Path::new("/t").join(path))
                .expect("representable"),
            report(errors),
        )
    }

    fn kinds(envelope: &CheckDirectoryReport) -> Vec<DiagnosticKind> {
        envelope.errors().iter().map(Diagnostic::kind).collect()
    }

    #[test]
    fn a_completed_empty_walk_is_one_empty_corpus_error() {
        let empty = CheckDirectoryReport::from_walk(Vec::new(), Vec::new())
            .expect_err("a completed walk that found nothing is an empty corpus");
        let envelope = empty.into_report("nothing under x/".into());
        assert_eq!(kinds(&envelope), [DiagnosticKind::EmptyCorpus]);
        assert_eq!(envelope.errors()[0].message, "nothing under x/");
        assert!(envelope.files().is_empty());
        assert!(envelope.has_errors());
    }

    /// [04-FIT-24]: a walk failure already explains an empty result, and the
    /// walk never established emptiness, so there is no `EmptyWalk` to turn
    /// into a second diagnostic.
    #[test]
    fn a_failed_walk_never_also_reports_an_empty_corpus() {
        let envelope = CheckDirectoryReport::from_walk(Vec::new(), vec!["cannot read x/".into()])
            .expect("a failed walk is an envelope, not an empty corpus");
        assert_eq!(kinds(&envelope), [DiagnosticKind::DirectoryWalkError]);
    }

    #[test]
    fn a_populated_walk_has_no_envelope_error() {
        let envelope = CheckDirectoryReport::from_walk(vec![entry("a.ch", Vec::new())], Vec::new())
            .expect("populated");
        assert!(envelope.errors().is_empty());
        assert!(!envelope.has_errors());
    }

    /// One diagnostic per failure, in order, each alongside the entries the
    /// walk could still reach ([04-FIT-23]).
    #[test]
    fn every_walk_failure_is_its_own_diagnostic_beside_the_entries() {
        let envelope = CheckDirectoryReport::from_walk(
            vec![entry("a.ch", Vec::new())],
            vec!["first".into(), "second".into()],
        )
        .expect("populated");
        let messages: Vec<&str> = envelope
            .errors()
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        assert_eq!(messages, ["first", "second"]);
        assert_eq!(envelope.files().len(), 1);
        assert!(envelope.has_errors());
    }

    /// [04-FIT-25] reads the entries too, not only the envelope's own list.
    #[test]
    fn an_error_inside_one_report_fails_the_envelope() {
        let failing = Diagnostic::new(DiagnosticKind::TypeMismatch, "bad", severity());
        let envelope = CheckDirectoryReport::from_walk(
            vec![entry("a.ch", Vec::new()), entry("b.ch", vec![failing])],
            Vec::new(),
        )
        .expect("populated");
        assert!(envelope.errors().is_empty());
        assert!(envelope.has_errors());
    }

    #[test]
    fn a_nested_path_is_joined_with_forward_slashes() {
        let nested: PathBuf = ["/t", "one", "two", "three.ch"].iter().collect();
        assert_eq!(
            EntryPath::relative_to(Path::new("/t"), &nested)
                .unwrap()
                .as_str(),
            "one/two/three.ch"
        );
    }

    #[test]
    fn a_path_that_is_not_below_the_target_is_refused() {
        for path in ["/t", "/elsewhere/a.ch", "/t/../a.ch"] {
            assert!(
                EntryPath::relative_to(Path::new("/t"), Path::new(path)).is_err(),
                "{path:?} must not become an entry"
            );
        }
    }

    /// [05-HOST-4]'s rendering, which [04-FIT-23] adopts: the bytes come back
    /// out of the message, so two names the walk rejected stay distinct.
    #[test]
    fn a_path_renders_as_reversible_escaped_host_bytes() {
        assert_eq!(escaped_path(Path::new("plain.ch")), "b\"plain.ch\"");
        assert_eq!(
            escaped_path(Path::new("a\tb\nc\\d\"e'f")),
            "b\"a\\tb\\nc\\\\d\\\"e\\'f\""
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_non_utf8_path_is_refused_not_replaced() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let path = Path::new(OsStr::from_bytes(b"/t/dir/\xffname.ch"));
        let error = EntryPath::relative_to(Path::new("/t"), path).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("not valid UTF-8"), "{message}");
        // The offending byte survives as `\xff`, and nothing was substituted
        // for it: `Path::display()` would have written U+FFFD here.
        // The message names the walked path, as every walk failure does.
        assert!(message.contains("b\"/t/dir/\\xffname.ch\""), "{message}");
        assert!(!message.contains('\u{fffd}'), "{message}");
    }
}
