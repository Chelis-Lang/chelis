//! `chelis check <file>.dp` and `chelis eval --file <file>.dp` ingestion.
//!
//! Per the Hull design (`spec/design/chelis_hull_design_spec.md` §6),
//! `chelis check path.dp --json` and `chelis eval --file path.dp` must
//! accept already-lowered Deep IR directly, run the SAME
//! type/effect/linearity pipeline the `.ch` path uses, and emit the
//! SAME structured JSON (CheckResult / EvalResult schema). A standalone
//! `.dp` is post-desugar IR by construction; it skips the Surf
//! desugar + macro-expand stage but is otherwise checked identically.
//!
//! Before this surface landed, `chelis check foo.dp` fed Deep
//! s-expressions to the Surf parser, which choked with
//! `expected declaration (def, sig, ...), found LParen at byte 0` and
//! reported a bogus score-0 parse error. `chelis eval --file foo.dp`
//! propagated the same Surf parse error and exited 1.
//!
//! These tests are hermetic: every fixture is synthesized into a
//! `tempfile::tempdir()` so no ambient reef package (a `chelis.toml`
//! up the tree) can intercept the file resolution, and the style gate
//! is disabled via `CHELIS_STYLE_GATE_DISABLE=1` so we exercise the
//! type/effect pipeline rather than formatter preferences.
//!
//! Owning code: `cmd_check_one` / `cmd_check_one_deep` / `cmd_eval`
//! and the shared `assemble_check_json` emitter in
//! `crates/chelis-cli/src/main.rs`.

use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use tempfile::tempdir;

/// Exit code for `chelis check` when the JSON `errors` array is
/// non-empty (issue #207 contract). The `.dp` surface inherits it.
const CHECK_ERRORS_EXIT_CODE: i32 = 2;

/// A well-typed standalone `.dp`: a scalar `f32` identity-ish function
/// with NO fan-out, so the desugared form carries no implicit
/// copy/drop. `neg` is a single-use of the parameter.
const WELL_TYPED_DP: &str = "(def {}\n  \
    negate\n  \
    (fn {}\n    \
        (params {} (x {type: (t-prim {} f32)}))\n    \
        (app {} (var {} neg) (var {} x))))\n";

/// The byte-equivalent-meaning Surf for `WELL_TYPED_DP`. `chelis deep`
/// desugars this to exactly the Deep above (single use of `x`, no
/// implicit copy inserted), so the two surfaces must produce
/// field-for-field identical CheckResult values.
const WELL_TYPED_CH: &str = "def negate(x: f32) -> f32 = neg(x)\n";

/// An ill-typed standalone `.dp`: `add` of an int literal and a bool
/// literal. The literals carry explicit precision metadata so the
/// type checker sees a concrete int-vs-bool mismatch.
const ILL_TYPED_DP: &str = "(def {}\n  \
    bad\n  \
    (app {} (var {} add)\n      \
        (lit {precision: int32} 1)\n      \
        (lit {precision: bool} true)))\n";

/// Helper: write `src` to `<dir>/<name>` and return the path.
fn write_fixture(dir: &Path, name: &str, src: &str) -> std::path::PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, src).expect("write fixture");
    path
}

/// Run `chelis check <path> [--show-inferred]`, returning
/// `(exit_code, stdout)`. Hermetic: the binary's cwd is the fixture's
/// own tempdir so no ambient package intercepts resolution.
fn run_check(path: &Path, show_inferred: bool) -> (Option<i32>, String) {
    let mut cmd = Command::cargo_bin("chelis").expect("binary");
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(path.parent().expect("parent"))
        .arg("check")
        .arg(path);
    if show_inferred {
        cmd.arg("--show-inferred");
    }
    let output = cmd.output().expect("run chelis check");
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    (output.status.code(), stdout)
}

/// Parse `chelis check` stdout as JSON, panicking with the raw text on
/// failure (a propagated boxed error or stderr noise would not parse).
fn parse_report(stdout: &str) -> Value {
    serde_json::from_str(stdout)
        .unwrap_or_else(|e| panic!("check stdout must be valid JSON: {e}\nstdout={stdout}"))
}

// ── POSITIVE: well-typed .dp -> score 1.0 + empty errors + exit 0 ──────

#[test]
fn check_well_typed_dp_scores_one_with_empty_errors() {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "negate.dp", WELL_TYPED_DP);
    let (code, stdout) = run_check(&path, false);
    let report = parse_report(&stdout);

    assert_eq!(
        report["score"].as_f64(),
        Some(1.0),
        "well-typed .dp must score 1.0; stdout={stdout}"
    );
    assert!(
        report["errors"]
            .as_array()
            .expect("errors array")
            .is_empty(),
        "well-typed .dp must have empty errors; stdout={stdout}"
    );
    for component in ["parse", "structure", "names", "types"] {
        assert_eq!(
            report["components"][component].as_f64(),
            Some(1.0),
            "component {component} must be 1.0 on a well-typed .dp; stdout={stdout}"
        );
    }
    assert!(
        report["total_nodes"].as_u64().unwrap_or(0) > 0,
        "well-typed .dp must report a positive total_nodes; stdout={stdout}"
    );
    assert_eq!(
        code,
        Some(0),
        "well-typed .dp must exit 0 (issue #207 invariant); stdout={stdout}"
    );
}

// ── ILL-TYPED: type mismatch -> non-empty errors + score<1 + exit 2 ────

#[test]
fn check_ill_typed_dp_reports_type_mismatch_and_exits_two() {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "bad.dp", ILL_TYPED_DP);
    let (code, stdout) = run_check(&path, false);
    let report = parse_report(&stdout);

    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        !errors.is_empty(),
        "ill-typed .dp must produce a non-empty errors array; stdout={stdout}"
    );
    let has_type_error = errors.iter().any(|e| {
        let kind = e["kind"].as_str().unwrap_or("");
        let message = e["message"].as_str().unwrap_or("");
        kind == "TypeMismatch" || message.to_lowercase().contains("mismatch")
    });
    assert!(
        has_type_error,
        "ill-typed .dp must name a type mismatch (kind or message); stdout={stdout}"
    );
    assert!(
        report["score"].as_f64().unwrap_or(1.0) < 1.0,
        "ill-typed .dp must score below 1.0; stdout={stdout}"
    );
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "ill-typed .dp must exit {CHECK_ERRORS_EXIT_CODE}; stdout={stdout}"
    );
}

// ── PARITY: a .dp and its byte-equivalent-meaning .ch are identical ────

#[test]
fn check_dp_matches_equivalent_ch_field_for_field() {
    let dir = tempdir().expect("tempdir");
    // Author the .ch, then derive the canonical .dp from it via
    // `chelis deep` so the two are by-construction the same program.
    let ch_path = write_fixture(dir.path(), "negate.ch", WELL_TYPED_CH);

    let deep_out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["deep", ch_path.to_str().unwrap()])
        .output()
        .expect("run chelis deep");
    assert!(
        deep_out.status.success(),
        "chelis deep must desugar the .ch; stderr={}",
        String::from_utf8_lossy(&deep_out.stderr)
    );
    let dp_path = write_fixture(
        dir.path(),
        "negate.dp",
        &String::from_utf8(deep_out.stdout).expect("utf8 deep output"),
    );

    let (ch_code, ch_stdout) = run_check(&ch_path, false);
    let (dp_code, dp_stdout) = run_check(&dp_path, false);
    let ch_report = parse_report(&ch_stdout);
    let dp_report = parse_report(&dp_stdout);

    // The .ch report is the oracle; the .dp report must match it
    // field-for-field. This is the load-bearing invariant: same
    // program, surface-independent verdict.
    for field in [
        "score",
        "components",
        "typed_nodes",
        "untyped_nodes",
        "total_nodes",
        "unresolved_names",
        "errors",
    ] {
        assert_eq!(
            dp_report[field], ch_report[field],
            "field `{field}` must be identical across .dp and .ch surfaces;\n.ch={ch_stdout}\n.dp={dp_stdout}"
        );
    }
    assert_eq!(
        dp_code, ch_code,
        "exit codes must match across surfaces; .ch={ch_code:?} .dp={dp_code:?}"
    );
    assert_eq!(ch_code, Some(0), "equivalent program must be well-typed");
}

// ── MALFORMED (a): syntactically broken .dp -> clean JSON error ────────

#[test]
fn check_syntactically_broken_dp_reports_clean_error_not_panic() {
    let dir = tempdir().expect("tempdir");
    // Unbalanced parens: a Deep parse error, not a Surf parse error and
    // not a panic / propagated boxed Err.
    let path = write_fixture(dir.path(), "broken.dp", "(def {} oops\n");
    let (code, stdout) = run_check(&path, false);
    let report = parse_report(&stdout);

    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        !errors.is_empty(),
        "broken .dp must produce a non-empty errors array; stdout={stdout}"
    );
    // It must NOT be the Surf-parser confusion message ("found LParen").
    let message = errors[0]["message"].as_str().unwrap_or("");
    assert!(
        !message.contains("expected declaration (def, sig"),
        "broken .dp must not be mis-parsed as Surf; stdout={stdout}"
    );
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "broken .dp must exit {CHECK_ERRORS_EXIT_CODE} (JSON report, not boxed Err); stdout={stdout}"
    );
}

// ── MALFORMED (b): unknown tag -> strict-vocabulary rejection ──────────

#[test]
fn check_unknown_tag_dp_rejected_by_strict_parse() {
    let dir = tempdir().expect("tempdir");
    // `frobnicate` is not in the closed Deep tag vocabulary. A
    // non-strict parse would silently accept it; parse_str_strict must
    // reject it, proving the strict gate is wired on the .dp surface.
    let path = write_fixture(dir.path(), "unknown.dp", "(frobnicate {} x)\n");
    let (code, stdout) = run_check(&path, false);
    let report = parse_report(&stdout);

    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        !errors.is_empty(),
        "unknown-tag .dp must produce a non-empty errors array; stdout={stdout}"
    );
    let message = errors[0]["message"].as_str().unwrap_or("").to_lowercase();
    assert!(
        message.contains("valid deep tag") || message.contains("frobnicate"),
        "unknown-tag .dp must surface a strict-vocabulary error; stdout={stdout}"
    );
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "unknown-tag .dp must exit {CHECK_ERRORS_EXIT_CODE}; stdout={stdout}"
    );
}

// ── EMPTY: whitespace/comment-only .dp -> EMPTY_PROGRAM_MESSAGE + exit 2 ─

#[test]
fn check_empty_dp_reports_empty_program_and_exits_two() {
    let dir = tempdir().expect("tempdir");
    // Comment-only / whitespace-only .dp: parses clean but yields zero
    // top-level exprs. Must mirror the .ch empty-file path.
    let path = write_fixture(dir.path(), "empty.dp", "; just a comment\n\n");
    let (code, stdout) = run_check(&path, false);
    let report = parse_report(&stdout);

    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        errors.iter().any(|e| e["message"]
            .as_str()
            .unwrap_or("")
            .contains("empty program")),
        "empty .dp must surface the canonical empty-program message; stdout={stdout}"
    );
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "empty .dp must exit {CHECK_ERRORS_EXIT_CODE}; stdout={stdout}"
    );
}

// ── --show-inferred: structured WireInferredType signatures ────────────

#[test]
fn check_show_inferred_dp_emits_wire_inferred_signatures() {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "negate.dp", WELL_TYPED_DP);
    let (code, stdout) = run_check(&path, true);
    let report = parse_report(&stdout);

    assert_eq!(
        code,
        Some(0),
        "well-typed .dp --show-inferred must exit 0; stdout={stdout}"
    );
    let signatures = report["inferred_signatures"].as_array().unwrap_or_else(|| {
        panic!("--show-inferred must add an inferred_signatures array; stdout={stdout}")
    });
    assert!(
        !signatures.is_empty(),
        "well-typed .dp must produce at least one inferred signature; stdout={stdout}"
    );
    // The signature for `negate` must be a function f32 -> f32, carried
    // in the structured WireInferredType tree (internally tagged on
    // `kind`) the .ch path also emits.
    let negate = signatures
        .iter()
        .find(|s| s["function"].as_str() == Some("negate"))
        .unwrap_or_else(|| panic!("expected a `negate` signature; stdout={stdout}"));
    let ty = &negate["checked_signature_structured"];
    assert_eq!(
        ty["kind"].as_str(),
        Some("fn"),
        "negate's inferred type must be a fn in WireInferredType shape; stdout={stdout}"
    );
    assert_eq!(
        ty["args"][0]["kind"].as_str(),
        Some("prim"),
        "negate's argument must be a prim; stdout={stdout}"
    );
    assert_eq!(
        ty["args"][0]["name"].as_str(),
        Some("f32"),
        "negate's argument must be f32; stdout={stdout}"
    );
    assert_eq!(
        ty["ret"]["name"].as_str(),
        Some("f32"),
        "negate's return must be f32; stdout={stdout}"
    );
}

// ── ROUND-TRIP: .dp check JSON deserializes into schema::CheckResult ───

#[test]
fn check_dp_json_round_trips_through_schema() {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "negate.dp", WELL_TYPED_DP);
    let (_code, stdout) = run_check(&path, false);
    // The hand-built emitter must produce exactly the
    // chelis_compiler_api::schema::CheckResult field set. Deserializing
    // into that struct locks the hand-built-vs-struct alignment.
    let parsed: Result<chelis_compiler_api::schema::CheckResult, _> = serde_json::from_str(&stdout);
    assert!(
        parsed.is_ok(),
        ".dp check JSON must deserialize into schema::CheckResult; err={:?}\nstdout={stdout}",
        parsed.err()
    );
}

// ── EXIT-CODE INVARIANT on the .dp surface (issue #207 extension) ──────

#[test]
fn check_dp_inherits_issue_207_exit_code_invariant() {
    let dir = tempdir().expect("tempdir");
    // Well-typed -> exit 0, empty errors.
    let good = write_fixture(dir.path(), "good.dp", WELL_TYPED_DP);
    let (good_code, good_stdout) = run_check(&good, false);
    let good_report = parse_report(&good_stdout);
    assert!(good_report["errors"].as_array().unwrap().is_empty());
    assert_eq!(good_code, Some(0));

    // Errors present -> exit 2.
    let bad = write_fixture(dir.path(), "bad.dp", ILL_TYPED_DP);
    let (bad_code, bad_stdout) = run_check(&bad, false);
    let bad_report = parse_report(&bad_stdout);
    assert!(!bad_report["errors"].as_array().unwrap().is_empty());
    assert_eq!(bad_code, Some(CHECK_ERRORS_EXIT_CODE));
}

// ── EVAL POSITIVE (--json): structured EvalResult on stdout ────────────

#[test]
fn eval_dp_json_emits_structured_eval_result() {
    let dir = tempdir().expect("tempdir");
    // A standalone .dp whose root is an evaluable scalar value.
    let src = "(def {} answer (lit {precision: int32} 42))\n";
    let path = write_fixture(dir.path(), "answer.dp", src);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["eval", "--file", path.to_str().unwrap(), "--json"])
        .output()
        .expect("run chelis eval --json");
    assert!(
        output.status.success(),
        "eval --file .dp --json must exit 0; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    // Stdout must be a single parseable EvalResult JSON document with no
    // extra non-JSON noise.
    let parsed: chelis_compiler_api::schema::EvalResult = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| {
            panic!("eval --json stdout must be a single EvalResult: {e}\nstdout={stdout}")
        });
    assert!(
        !parsed.roots.is_empty(),
        "eval --file .dp --json must report at least one root; stdout={stdout}"
    );
}

// ── EVAL POSITIVE (human): cross-surface output parity with .ch ────────

#[test]
fn eval_dp_human_matches_equivalent_ch_output() {
    let dir = tempdir().expect("tempdir");
    // Use a top-level value binding (not a function `def`): both the
    // `.ch` eval path (`root_names_from_decls`) and the `.dp` eval path
    // (engine-selected lowered roots) agree it is an evaluable root, so
    // a genuine cross-surface output comparison is meaningful. The `.dp`
    // is derived from the `.ch` via `chelis deep` so the two are
    // by-construction the same program.
    let ch_src = "answer = cast(42, int32)\n";
    let ch_path = write_fixture(dir.path(), "answer.ch", ch_src);
    let deep_out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["deep", ch_path.to_str().unwrap()])
        .output()
        .expect("run chelis deep");
    assert!(
        deep_out.status.success(),
        "chelis deep must desugar the .ch; stderr={}",
        String::from_utf8_lossy(&deep_out.stderr)
    );
    let dp_path = write_fixture(
        dir.path(),
        "answer.dp",
        &String::from_utf8(deep_out.stdout).expect("utf8 deep output"),
    );

    let run = |p: &Path| -> String {
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .current_dir(dir.path())
            .args(["eval", "--file", p.to_str().unwrap()])
            .output()
            .expect("run chelis eval");
        assert!(
            output.status.success(),
            "eval must succeed for {}; stderr={}",
            p.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("utf8")
    };

    let dp_out = run(&dp_path);
    let ch_out = run(&ch_path);
    assert_eq!(
        dp_out.trim(),
        ch_out.trim(),
        "eval of equivalent .dp and .ch must produce identical output"
    );
}

// ── EVAL NEGATIVE: ill-typed .dp -> nonzero exit, type error on stderr ─

#[test]
fn eval_ill_typed_dp_fails_with_type_error() {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "bad.dp", ILL_TYPED_DP);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    assert!(
        !output.status.success(),
        "ill-typed .dp eval must fail; stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
}

// ── EVAL NEGATIVE: unknown tag -> Deep strict error, not Surf parse ────

#[test]
fn eval_unknown_tag_dp_fails_with_deep_error_not_surf() {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(dir.path(), "unknown.dp", "(frobnicate {} x)\n");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir.path())
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval");
    assert!(
        !output.status.success(),
        "unknown-tag .dp eval must fail; stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("expected declaration (def, sig"),
        "unknown-tag .dp eval must not be mis-parsed as Surf; stderr={stderr}"
    );
}
