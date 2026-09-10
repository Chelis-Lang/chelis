//! chelis#886 [04-FIT-12]: every `chelis check` failure reaches the report.
//!
//! chelis#1672 gave the report one producer but left four Surf paths
//! bypassing it entirely: they returned `Err`, so the process emitted a
//! display string on stderr, nothing on stdout, and exit 1. §6.4 keeps its
//! "not fully implemented" caveat for exactly that reason.
//!
//! [04-FIT-12] is explicit that this is non-conforming -- "a consumer cannot
//! distinguish 'no diagnostics' from 'the diagnostics were not
//! transported'". These tests are the executable form of the atom.
//!
//! **The exit status moves 1 -> 2 on these paths, deliberately.** Exit is
//! derived from `cmd_check_one`'s `Ok`/`Err` discriminant, so routing a path
//! through the report converts it. `spec/04-type-system.md` § Gating pins
//! only "`0` iff empty, non-zero otherwise", so both values conform and the
//! choice was a maintainer call. It converges the Surf arm onto the Deep
//! arm, which already emitted a report and exit 2 for the identical failure:
//! before this change an unreadable `.ch` exited 1 with no document while an
//! unreadable `.dp` exited 2 with one, so "exit 1 means tooling broken" was
//! not a contract a consumer could rely on.
//!
//! # The boundary, stated so nobody reads these tests as more than they are
//!
//! This closes the PER-FILE bypasses. §6.4 keeps its "(Not fully
//! implemented; tracked by chelis#886.)" caveat, because one path above the
//! file level still emits a display string: `chelis check <dir>` on a
//! directory it cannot enumerate fails in `discover_check_files` and exits 1
//! with empty stdout. There is no per-file report to produce there -- no
//! file was reached -- so it belongs to the directory ENVELOPE, which
//! `spec/04` does not specify at all and which chelis#1678 owns.
//!
//! Removing the caveat is therefore chelis#1678's to do, once the envelope
//! has a normative shape to conform to. An earlier revision of chelis#1672
//! deleted it prematurely and a red-team pass caught that; this suite
//! deliberately does not repeat it.

use std::fs;
use std::io::Write;

use assert_cmd::Command;
use chelis_compiler_api::schema::WireCheckResult;
use tempfile::{TempDir, tempdir};

/// `chelis check` on a file, with the style gate LIVE.
///
/// Deliberately does not set `CHELIS_STYLE_GATE_DISABLE`, because one of the
/// paths under test IS the style gate.
///
/// Which tests that matters for, stated structurally rather than as a count.
/// The two gate-sensitive ones are `a_style_gate_rejection_...`, whose
/// subject IS the gate, and `a_style_violation_is_still_bypassable_...`,
/// which asserts the `--allow-style-violations` warning that the env var
/// suppresses. The read, non-UTF-8 and Deep fixtures fail in
/// `fs::read_to_string` before the gate is consulted, so the variable is
/// irrelevant to them, and the clean fixture passes either way.
///
/// Two earlier drafts of this comment each carried a number, and both were
/// wrong: the first claimed the sibling suite "cannot see any of this",
/// and the second said "exactly one of six" -- measured before the same
/// commit added the bypass assertion that made a second test
/// gate-sensitive. A count here depends on the suite's own contents and
/// goes stale the moment a test is added, so this names the mechanism
/// instead.
fn check(path: &std::path::Path) -> (Option<i32>, String, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check"])
        .arg(path)
        .output()
        .expect("run chelis check");
    (
        output.status.code(),
        String::from_utf8_lossy(&output.stdout).to_string(),
        String::from_utf8_lossy(&output.stderr).to_string(),
    )
}

/// Assert the shared [04-FIT-12] contract for one failing input.
fn assert_transported(label: &str, path: &std::path::Path) -> WireCheckResult {
    let (code, stdout, stderr) = check(path);
    assert!(
        !stdout.trim().is_empty(),
        "{label}: emitted no document at all; \
         a consumer cannot distinguish this from a clean check. stderr={stderr}"
    );
    let report: WireCheckResult = serde_json::from_str(&stdout).unwrap_or_else(|error| {
        panic!("{label}: stdout is not a check report ({error}); stdout={stdout}")
    });
    assert!(
        !report.errors.is_empty(),
        "{label}: transported a report with an EMPTY errors array, which \
         reads as success; stdout={stdout}"
    );
    for diagnostic in &report.errors {
        chelis_vocab::DiagnosticKind::decode(&diagnostic.kind).unwrap_or_else(|_| {
            panic!(
                "{label}: kind {:?} is outside the closed vocabulary",
                diagnostic.kind
            )
        });
        assert!(
            !diagnostic.message.trim().is_empty(),
            "{label}: a diagnostic with no message transports nothing"
        );
    }
    // chelis#207/#731: exit status mirrors the errors array. Converged onto
    // the Deep arm's existing value; see the module note.
    assert_eq!(
        code,
        Some(2),
        "{label}: a non-empty errors array must exit 2; stdout={stdout}"
    );
    report
}

fn write(dir: &TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let mut file = fs::File::create(&path).expect("create fixture");
    file.write_all(bytes).expect("write fixture");
    path
}

#[test]
fn an_unreadable_surf_file_reports_through_the_document() {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir().expect("tempdir");
        let path = write(&dir, "locked.ch", b"def f(x: f32) -> f32 = add(x, x)\n");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).expect("chmod");
        let report = assert_transported("unreadable .ch", &path);
        // Restore before the tempdir drops, or cleanup fails on some systems.
        let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o644));
        assert_eq!(report.score, 0.0, "an unread file scores zero");
    }
}

#[test]
fn a_non_utf8_surf_file_reports_through_the_document() {
    let dir = tempdir().expect("tempdir");
    // Lone continuation bytes: valid as bytes, never valid UTF-8.
    let path = write(&dir, "binary.ch", &[0x80, 0x81, 0xfe, 0xff, 0x0a]);
    assert_transported("non-UTF-8 .ch", &path);
}

#[test]
fn a_style_gate_rejection_reports_through_the_document() {
    let dir = tempdir().expect("tempdir");
    // Parses fine; not canonically formatted, so the gate rejects it before
    // the checker runs.
    let path = write(&dir, "ugly.ch", b"def  f(x:f32)->f32=add(x,x)\n");
    let report = assert_transported("style-gate rejection", &path);
    assert!(
        report
            .errors
            .iter()
            .any(|d| d.message.to_lowercase().contains("format")
                || d.message.to_lowercase().contains("lint")
                || d.message.to_lowercase().contains("style")),
        "the diagnostic should say what was rejected; got {:?}",
        report.errors.iter().map(|d| &d.message).collect::<Vec<_>>()
    );
}

#[test]
fn the_deep_arm_keeps_reporting_unreadable_input() {
    // Regression guard on the precedent this change converges toward. The
    // Deep arm already satisfied [04-FIT-12] here; converging the Surf arm
    // must not disturb it.
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir().expect("tempdir");
        let path = write(&dir, "locked.dp", b"(module {})\n");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).expect("chmod");
        assert_transported("unreadable .dp", &path);
        let _ = fs::set_permissions(&path, fs::Permissions::from_mode(0o644));
    }
}

#[test]
fn a_clean_file_is_unaffected() {
    // Negative parity: the paths above must not have made every input fail.
    let dir = tempdir().expect("tempdir");
    let path = write(&dir, "clean.ch", b"def f(x: f32) -> f32 = add(x, x)\n");
    let (code, stdout, stderr) = check(&path);
    let report: WireCheckResult =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("{e}; {stdout} {stderr}"));
    assert!(report.errors.is_empty(), "clean file, no errors: {stdout}");
    assert_eq!(code, Some(0), "clean file exits zero");
}

#[test]
fn a_style_violation_is_still_bypassable_by_the_documented_flag() {
    // `--allow-style-violations` must still bypass the gate rather than
    // being swallowed by the new reporting path.
    let dir = tempdir().expect("tempdir");
    let path = write(&dir, "ugly.ch", b"def  f(x:f32)->f32=add(x,x)\n");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", "--allow-style-violations"])
        .arg(&path)
        .output()
        .expect("run");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let report: WireCheckResult =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("{e}; {stdout}"));
    assert!(
        report.errors.is_empty(),
        "the flag bypasses the gate, so this checks clean: {stdout}"
    );
    assert_eq!(output.status.code(), Some(0));
    // Distinguishes the FLAG from `CHELIS_STYLE_GATE_DISABLE`, which reaches
    // the same stdout by a different route. Without this the test passes
    // vacuously under the env var and stops testing the flag at all.
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--allow-style-violations"),
        "the documented bypass warns on stderr; got {stderr:?}"
    );
}

#[test]
fn a_transported_message_is_a_machine_value_not_a_terminal_transcript() {
    // The defect this pins is a class, not a spelling. Routing the style
    // gate through the report first put its *stderr* text into the wire
    // `message` verbatim: embedded newlines, an "N issue(s)" plural
    // placeholder, the word "build" on a `check` surface, and the advice
    // "pass `--allow-style-violations` to bypass (CI must not)". All of
    // that is correct for a terminal and wrong for a machine-facing field
    // -- it is the shape chelis#886 exists to remove, so reintroducing it
    // while closing chelis#886 would be self-defeating.
    //
    // stderr keeps every bit of it. Only the report is constrained.
    let dir = tempdir().expect("tempdir");

    // A formatting difference AND a lint violation in one file -- the
    // genuinely mixed case. An earlier fixture here used an uppercase head
    // (`def BadName`), which is a Surf PARSE error, so `check_fmt` returned
    // `None` and the file only ever produced lint violations: the fmt branch
    // of the mapping was never exercised by the test that claimed to cover
    // the multi-issue case.
    let path = write(&dir, "both.ch", b"def  _internal(x:f32)->f32=add(x,x)\n");
    let (code, stdout, stderr) = check(&path);
    let report: WireCheckResult =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("{e}; {stdout}"));

    assert_eq!(
        report.errors.len(),
        2,
        "one diagnostic per issue, so the array length is the number of \
         problems found; got {stdout}"
    );
    assert_eq!(code, Some(2));

    for diagnostic in &report.errors {
        // NO control characters, not just no newline. A red-team pass got a
        // tab, a carriage return and ANSI SGR escapes past the newline-only
        // version of this check, which is a reminder that banning the one
        // character the last bug happened to use is not a class.
        if let Some(bad) = diagnostic.message.chars().find(|c| c.is_control()) {
            panic!(
                "a transported message carries no control characters; found \
                 {bad:?} (U+{:04X}) in {:?}",
                bad as u32, diagnostic.message
            );
        }
        for terminal_only in [
            "issue(s)",
            "--allow-style-violations",
            "blocked the build",
            "CI must not",
        ] {
            assert!(
                !diagnostic.message.contains(terminal_only),
                "{terminal_only:?} is terminal advice and belongs on stderr, \
                 not in a machine-facing message; got {:?}",
                diagnostic.message
            );
        }
    }

    // The negative half: stderr still carries the whole human report,
    // including exactly the strings the wire must not.
    assert!(
        stderr.contains("2 issue(s)") && stderr.contains("--allow-style-violations"),
        "the terminal keeps its summary and its advice; got {stderr:?}"
    );

    // And the single-issue case still produces exactly one.
    let single = write(&dir, "ugly.ch", b"def  f(x:f32)->f32=add(x,x)\n");
    let (_, stdout, _) = check(&single);
    let report: WireCheckResult =
        serde_json::from_str(&stdout).unwrap_or_else(|e| panic!("{e}; {stdout}"));
    assert_eq!(
        report.errors.len(),
        1,
        "one issue, one diagnostic: {stdout}"
    );
    assert!(!report.errors[0].message.chars().any(char::is_control));
}

#[test]
fn a_non_zero_exit_always_carries_at_least_one_diagnostic() {
    // The invariant a red-team pass falsified. The style-gate path ran the
    // gate TWICE, so a file rewritten between the two runs could have run A
    // reject -- setting the exit status -- while run B found nothing, so the
    // report shipped exit 2 with an EMPTY errors array. `spec/04` § Gating
    // says that state cannot exist, and it is precisely the "cannot
    // distinguish no diagnostics from diagnostics not transported" confusion
    // [04-FIT-12] exists to remove.
    //
    // The repair is structural: one gate run yields both the stderr report
    // and the per-issue messages, so the two cannot disagree. This asserts
    // the contract over every failing shape rather than probing a timing
    // window, which would be flaky and would only ever sample the bug.
    let dir = tempdir().expect("tempdir");
    let fixtures: Vec<(&str, Vec<u8>)> = vec![
        ("ugly.ch", b"def  f(x:f32)->f32=add(x,x)\n".to_vec()),
        ("both.ch", b"def  _internal(x:f32)->f32=add(x,x)\n".to_vec()),
        ("parse.ch", b"def f(x: f32) -> f32 = add(x,\n".to_vec()),
        ("empty.ch", Vec::new()),
        ("binary.ch", vec![0x80, 0x81, 0xfe, 0xff, 0x0a]),
        ("bad.dp", b"(this is not deep)\n".to_vec()),
        ("empty.dp", Vec::new()),
    ];
    for (name, bytes) in fixtures {
        let path = write(&dir, name, &bytes);
        let (code, stdout, stderr) = check(&path);
        assert_ne!(code, Some(0), "{name} must fail; stdout={stdout}");
        let report: WireCheckResult = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("{name}: no report ({e}); stderr={stderr}"));
        assert!(
            !report.errors.is_empty(),
            "{name}: exited {code:?} with an EMPTY errors array, which reads \
             as success to any consumer keying on the array; stdout={stdout}"
        );
    }
}
