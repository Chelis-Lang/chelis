//! Expected-failure classification for `tests_neg/` and `tests_blocked/`.
//!
//! Contract §5/§6 require two expected-to-fail suites, each `<name>.ch` paired
//! with a `<name>.expect` sidecar whose **line 1 is the required diagnostic
//! substring** and **lines 2+ are the blocker citation**. School ships this as
//! two hand-copied Python runners (`run_negative_tests.py`,
//! `run_blocked_probes.py`); this module is the native, toolchain-shipped form,
//! driven by `chelis test --expect <neg|blocked>`.
//!
//! The logic is pure — it takes a file's reduced test outcome plus its parsed
//! sidecar and returns a [`Verdict`] — so it is fully unit-tested here without
//! the CLI. The CLI adapter reduces its per-file `TestRow`s to a [`FileOutcome`]
//! and prints the verdicts.

/// Which expected-failure suite semantics to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectMode {
    /// `tests_neg/`: the case MUST fail with the pinned diagnostic. A pass is a
    /// regression ("should have failed").
    Neg,
    /// `tests_blocked/`: a reproducer of an open upstream blocker that MUST
    /// still fail at the pin. A pass means upstream fixed it (FIX-detected —
    /// the wanted outcome at a bump, but the gate fails loudly so it is acted
    /// on); a different diagnostic means the failure mode moved (DRIFTED).
    Blocked,
}

impl ExpectMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            ExpectMode::Neg => "neg",
            ExpectMode::Blocked => "blocked",
        }
    }
}

/// A parsed `.expect` sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sidecar {
    /// Line 1: the required diagnostic substring (non-empty).
    pub substring: String,
    /// Lines 2+: non-empty citation/instruction lines.
    pub citations: Vec<String>,
}

impl Sidecar {
    /// Parse sidecar text. `Err` carries a human config-error reason (the caller
    /// turns it into a fail-closed [`Verdict::ConfigError`]).
    pub fn parse(text: &str) -> Result<Sidecar, String> {
        let mut lines = text.lines();
        let substring = lines
            .next()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                "empty .expect sidecar: line 1 must be the diagnostic substring".to_string()
            })?
            .to_string();
        let citations = lines
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        Ok(Sidecar {
            substring,
            citations,
        })
    }

    /// Does any citation line reference an upstream blocker in a
    /// mechanically-auditable form (contract §4: never a prose name)?
    pub fn has_blocker_citation(&self) -> bool {
        self.citations.iter().any(|c| is_blocker_citation(c))
    }
}

/// A citation is auditable iff it names `chelis#NNN`, a parked draft under
/// `docs/issue_drafts/`, or the `docs/UPSTREAM_BUGS.md` tracker.
pub fn is_blocker_citation(line: &str) -> bool {
    if line.contains("docs/issue_drafts/") || line.contains("docs/UPSTREAM_BUGS.md") {
        return true;
    }
    // `chelis#<digits>` anywhere in the line.
    if let Some(idx) = line.find("chelis#") {
        let rest = &line[idx + "chelis#".len()..];
        return rest.chars().next().is_some_and(|c| c.is_ascii_digit());
    }
    false
}

/// The reduced result of running one test `.ch` file (all its `test_*` records
/// collapsed to a single verdict-relevant outcome).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOutcome {
    /// The file compiled and every test in it passed.
    Pass,
    /// At least one test (or the compile itself) failed; carries the failure
    /// messages for substring matching.
    Fail { messages: Vec<String> },
    /// The file produced no test records at all (did not compile-fail and had no
    /// `test_*` functions) — a config problem for an expected-failure suite.
    Empty,
}

impl FileOutcome {
    /// Reduce per-test `(is_pass, message)` records to a file outcome. Any
    /// failure makes the file a [`FileOutcome::Fail`] collecting every failure
    /// message; all-pass is [`FileOutcome::Pass`]; no records is
    /// [`FileOutcome::Empty`].
    pub fn from_rows<'a>(rows: impl IntoIterator<Item = (bool, Option<&'a str>)>) -> FileOutcome {
        let mut any = false;
        let mut messages = Vec::new();
        for (is_pass, message) in rows {
            any = true;
            if !is_pass {
                messages.push(message.unwrap_or("").to_string());
            }
        }
        if !any {
            FileOutcome::Empty
        } else if messages.is_empty() {
            FileOutcome::Pass
        } else {
            FileOutcome::Fail { messages }
        }
    }
}

/// The verdict for one expected-failure file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Healthy: still failing with the pinned diagnostic.
    Ok,
    /// `neg`: the case passed but was required to fail (a regression).
    ShouldHaveFailed,
    /// `neg`: failed, but with a diagnostic that does not contain the required
    /// substring.
    WrongDiagnostic { expected: String, got: Vec<String> },
    /// `blocked`: the probe now passes — upstream fixed the blocker. Carries the
    /// sidecar's de-narrowing instructions (its citation/instruction lines).
    FixDetected { instructions: Vec<String> },
    /// `blocked`: still fails, but the failure mode moved (different diagnostic).
    Drifted { expected: String, got: Vec<String> },
    /// Fail-closed: missing/empty sidecar, no records, or (blocked) a sidecar
    /// with no auditable blocker citation.
    ConfigError { reason: String },
}

impl Verdict {
    /// Only [`Verdict::Ok`] passes the gate. FIX-detected is the *wanted*
    /// outcome at a pin bump but still fails loudly so it is acted on.
    pub fn is_ok(&self) -> bool {
        matches!(self, Verdict::Ok)
    }

    /// Short stable tag for machine output.
    pub fn tag(&self) -> &'static str {
        match self {
            Verdict::Ok => "ok",
            Verdict::ShouldHaveFailed => "should-have-failed",
            Verdict::WrongDiagnostic { .. } => "wrong-diagnostic",
            Verdict::FixDetected { .. } => "fix-detected",
            Verdict::Drifted { .. } => "drifted",
            Verdict::ConfigError { .. } => "config-error",
        }
    }
}

/// Classify one expected-failure file. `sidecar` is `None` when the `.expect`
/// file is absent (fail-closed).
pub fn classify(mode: ExpectMode, outcome: &FileOutcome, sidecar: Option<&Sidecar>) -> Verdict {
    let Some(sidecar) = sidecar else {
        return Verdict::ConfigError {
            reason: "missing .expect sidecar".to_string(),
        };
    };
    if let FileOutcome::Empty = outcome {
        return Verdict::ConfigError {
            reason: "no test records produced (nothing compiled-failed and no test_* functions)"
                .to_string(),
        };
    }
    // A blocked probe must cite its blocker in an auditable form; an uncited
    // probe is invisible to de-narrowing (contract §4).
    if mode == ExpectMode::Blocked && !sidecar.has_blocker_citation() {
        return Verdict::ConfigError {
            reason: "tests_blocked sidecar has no auditable blocker citation \
                     (chelis#NNN, docs/issue_drafts/, or docs/UPSTREAM_BUGS.md)"
                .to_string(),
        };
    }

    let matches_substring =
        |messages: &[String]| messages.iter().any(|m| m.contains(&sidecar.substring));

    match (mode, outcome) {
        (ExpectMode::Neg, FileOutcome::Pass) => Verdict::ShouldHaveFailed,
        (ExpectMode::Neg, FileOutcome::Fail { messages }) => {
            if matches_substring(messages) {
                Verdict::Ok
            } else {
                Verdict::WrongDiagnostic {
                    expected: sidecar.substring.clone(),
                    got: messages.clone(),
                }
            }
        }
        (ExpectMode::Blocked, FileOutcome::Pass) => Verdict::FixDetected {
            instructions: sidecar.citations.clone(),
        },
        (ExpectMode::Blocked, FileOutcome::Fail { messages }) => {
            if matches_substring(messages) {
                Verdict::Ok
            } else {
                Verdict::Drifted {
                    expected: sidecar.substring.clone(),
                    got: messages.clone(),
                }
            }
        }
        // Empty handled above.
        (_, FileOutcome::Empty) => unreachable!("Empty handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fail(msg: &str) -> FileOutcome {
        FileOutcome::Fail {
            messages: vec![msg.to_string()],
        }
    }

    fn blocked_sidecar() -> Sidecar {
        Sidecar {
            substring: "rank mismatch".to_string(),
            citations: vec!["chelis#345 fixed → promote to tests/".to_string()],
        }
    }

    fn neg_sidecar() -> Sidecar {
        Sidecar {
            substring: "expected i32".to_string(),
            citations: vec![],
        }
    }

    #[test]
    fn sidecar_parse_extracts_substring_and_citations() {
        let s = Sidecar::parse("rank mismatch\nchelis#345\n\ndocs/UPSTREAM_BUGS.md §Tracking\n")
            .unwrap();
        assert_eq!(s.substring, "rank mismatch");
        assert_eq!(
            s.citations,
            vec!["chelis#345", "docs/UPSTREAM_BUGS.md §Tracking"]
        );
        assert!(s.has_blocker_citation());
    }

    #[test]
    fn sidecar_parse_rejects_empty() {
        assert!(Sidecar::parse("").is_err());
        assert!(Sidecar::parse("\n\n").is_err());
    }

    #[test]
    fn blocker_citation_forms() {
        assert!(is_blocker_citation("blocked on chelis#293"));
        assert!(is_blocker_citation("see docs/issue_drafts/foo.md"));
        assert!(is_blocker_citation(
            "docs/UPSTREAM_BUGS.md §Actively blocking"
        ));
        assert!(!is_blocker_citation(
            "the generic-callback-unification limit"
        )); // prose name
        assert!(!is_blocker_citation("chelis#")); // no number
    }

    #[test]
    fn reduce_rows() {
        assert_eq!(
            FileOutcome::from_rows([(true, None), (true, Some("x"))]),
            FileOutcome::Pass
        );
        assert_eq!(
            FileOutcome::from_rows([] as [(bool, Option<&str>); 0]),
            FileOutcome::Empty
        );
        assert_eq!(
            FileOutcome::from_rows([(true, None), (false, Some("boom"))]),
            FileOutcome::Fail {
                messages: vec!["boom".to_string()]
            }
        );
    }

    // ---- neg direction ----
    #[test]
    fn neg_fail_with_substring_is_ok() {
        let v = classify(
            ExpectMode::Neg,
            &fail("error: expected i32, got f32"),
            Some(&neg_sidecar()),
        );
        assert_eq!(v, Verdict::Ok);
    }

    #[test]
    fn neg_pass_should_have_failed() {
        let v = classify(ExpectMode::Neg, &FileOutcome::Pass, Some(&neg_sidecar()));
        assert_eq!(v, Verdict::ShouldHaveFailed);
        assert!(!v.is_ok());
    }

    #[test]
    fn neg_fail_wrong_diagnostic() {
        let v = classify(
            ExpectMode::Neg,
            &fail("error: something else"),
            Some(&neg_sidecar()),
        );
        assert!(matches!(v, Verdict::WrongDiagnostic { .. }));
        assert!(!v.is_ok());
    }

    // ---- blocked direction ----
    #[test]
    fn blocked_fail_with_substring_is_ok() {
        let v = classify(
            ExpectMode::Blocked,
            &fail("error: rank mismatch in expand"),
            Some(&blocked_sidecar()),
        );
        assert_eq!(v, Verdict::Ok);
    }

    #[test]
    fn blocked_pass_is_fix_detected() {
        let v = classify(
            ExpectMode::Blocked,
            &FileOutcome::Pass,
            Some(&blocked_sidecar()),
        );
        assert!(matches!(v, Verdict::FixDetected { .. }));
        assert!(!v.is_ok(), "FIX-detected must fail the gate loudly");
        assert_eq!(v.tag(), "fix-detected");
    }

    #[test]
    fn blocked_fail_wrong_diagnostic_is_drifted() {
        let v = classify(
            ExpectMode::Blocked,
            &fail("error: totally different"),
            Some(&blocked_sidecar()),
        );
        assert!(matches!(v, Verdict::Drifted { .. }));
        assert!(!v.is_ok());
    }

    // ---- fail-closed config errors ----
    #[test]
    fn missing_sidecar_is_config_error() {
        assert!(matches!(
            classify(ExpectMode::Neg, &fail("x"), None),
            Verdict::ConfigError { .. }
        ));
    }

    #[test]
    fn empty_outcome_is_config_error() {
        assert!(matches!(
            classify(
                ExpectMode::Blocked,
                &FileOutcome::Empty,
                Some(&blocked_sidecar())
            ),
            Verdict::ConfigError { .. }
        ));
    }

    #[test]
    fn blocked_without_citation_is_config_error() {
        let uncited = Sidecar {
            substring: "rank mismatch".to_string(),
            citations: vec!["implementation convenience".to_string()],
        };
        assert!(matches!(
            classify(ExpectMode::Blocked, &fail("rank mismatch"), Some(&uncited)),
            Verdict::ConfigError { .. }
        ));
    }

    #[test]
    fn neg_without_citation_is_allowed() {
        // neg cases do not require a blocker citation.
        let v = classify(
            ExpectMode::Neg,
            &fail("error: expected i32"),
            Some(&neg_sidecar()),
        );
        assert_eq!(v, Verdict::Ok);
    }
}
