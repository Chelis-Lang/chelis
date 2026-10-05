//! chelis#353: a user `def` shadowing a builtin name (`def sum`) checked
//! clean but segfaulted on the C backend (rc=139) and mis-dispatched
//! under eval (`expected int arg at index 1, got None`) — three lanes,
//! three different answers.
//!
//! Fix: a check-time rejection (`CheckErrorKind::BuiltinShadowing`)
//! raised in the declaration-collection chokepoint every front-end lane
//! shares. This file pins the CLI-facing contract:
//!
//! 1. `chelis check` rejects the reproducer (exit 2, kind
//!    `BuiltinShadowing`, message naming the builtin + spec section);
//! 2. lane consistency: `check`, `eval --file`, and `build` ALL reject
//!    the same source with the same diagnostic — the chelis#353
//!    acceptance criterion;
//! 3. the rejection is semantic, not style: `--allow-style-violations`
//!    and `CHELIS_STYLE_GATE_DISABLE=1` do NOT unlock the broken path;
//! 4. near-miss names still check, eval (with correct numerics), and
//!    build;
//! 5. reef package modules keep working: package decls are
//!    internal-name-rewritten (`pkg__...`) before the checker runs, so a
//!    package-scoped `def sum` is allowed and dispatches to the user def
//!    (the stdlib's `Std.Test.fail`, which shares the builtin `fail`'s
//!    name, relies on exactly this).
//!
//! Unit-level coverage (full BUILTIN_NAMES sweep, defsig classes, dedupe,
//! params/locals scope pins) lives in
//! `crates/chelis-types/tests/issue_353_builtin_shadowing.rs`.

use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use tempfile::tempdir;

/// `chelis check` exit code when the JSON errors array is non-empty
/// (issue #207 invariant).
const CHECK_ERRORS_EXIT_CODE: i32 = 2;

const REPRO: &str = "module Repro\n\
                     def sum(x: &tensor[batch, f32]) -> tensor[batch, f32] = relu(x)\n\
                     out = sum(to_tensor([1.0, -2.0]))\n";

fn write_tempfile(prefix: &str, src: &str) -> tempfile::NamedTempFile {
    use std::io::Write;
    let mut tmp = tempfile::Builder::new()
        .prefix(prefix)
        .suffix(".ch")
        .tempfile()
        .expect("create tempfile");
    tmp.write_all(src.as_bytes()).expect("write tempfile");
    tmp.flush().expect("flush tempfile");
    tmp
}

fn run_check(path: &Path) -> (Option<i32>, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("path utf8")])
        .output()
        .expect("run chelis check");
    (
        output.status.code(),
        String::from_utf8(output.stdout).expect("utf8 stdout"),
    )
}

fn errors_array(stdout: &str) -> Vec<Value> {
    let json: Value = serde_json::from_str(stdout)
        .unwrap_or_else(|err| panic!("chelis check stdout must be valid JSON: {err}\n{stdout}"));
    json.get("errors")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_else(|| panic!("missing errors array; stdout={stdout}"))
}

fn has_builtin_shadowing(errors: &[Value], name: &str) -> bool {
    errors.iter().any(|e| {
        e.get("kind")
            .and_then(Value::as_str)
            .is_some_and(|k| k == "BuiltinShadowing")
            && e.get("message").and_then(Value::as_str).is_some_and(|m| {
                m.contains(&format!("`{name}`")) && m.contains("shadows the builtin")
            })
    })
}

/// The chelis#353 reproducer is rejected at check time with the new
/// diagnostic: kind `BuiltinShadowing`, message naming the builtin and
/// the declaration site, citing the spec section, exit 2.
#[test]
fn issue_353_def_sum_rejected_at_check() {
    let tmp = write_tempfile("issue353-repro-", REPRO);
    let (code, stdout) = run_check(tmp.path());
    let errors = errors_array(&stdout);
    assert!(
        has_builtin_shadowing(&errors, "sum"),
        "expected a BuiltinShadowing error naming `sum`; stdout={stdout}"
    );
    let message = errors
        .iter()
        .find_map(|e| {
            let kind = e.get("kind").and_then(Value::as_str)?;
            (kind == "BuiltinShadowing").then(|| e.get("message").and_then(Value::as_str))?
        })
        .unwrap_or_default()
        .to_string();
    assert!(
        message.contains("`def sum`") && message.contains("spec/04-type-system.md §8.6"),
        "diagnostic must name the def site and cite the spec; got: {message}"
    );
    assert_eq!(
        code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "shadowing must exit {CHECK_ERRORS_EXIT_CODE}; stdout={stdout}"
    );
}

/// A standalone `sig sum: ...` declaration is rejected the same way.
#[test]
fn issue_353_sig_only_rejected_at_check() {
    let src = "module SigOnly\n\
               sig sum: &tensor[batch, f32] -> tensor[batch, f32]\n\
               out = relu(to_tensor([1.0, -2.0]))\n";
    let tmp = write_tempfile("issue353-sig-", src);
    let (code, stdout) = run_check(tmp.path());
    let errors = errors_array(&stdout);
    assert!(
        has_builtin_shadowing(&errors, "sum"),
        "sig-only shadowing must be rejected; stdout={stdout}"
    );
    assert_eq!(code, Some(CHECK_ERRORS_EXIT_CODE), "stdout={stdout}");
}

/// Lane-consistency invariant (the chelis#353 acceptance criterion):
/// after the fix, `check`, `eval --file`, and `build` give the SAME
/// answer for the reproducer — the BuiltinShadowing rejection. Pre-fix
/// the three answers were: clean / builtin-arity error / SIGSEGV.
#[test]
fn issue_353_lane_consistency_check_eval_build_all_reject() {
    let tmp = write_tempfile("issue353-lanes-", REPRO);
    let path = tmp.path().to_str().expect("path utf8");

    // Lane 1: check.
    let (check_code, check_stdout) = run_check(tmp.path());
    assert_eq!(
        check_code,
        Some(CHECK_ERRORS_EXIT_CODE),
        "check lane must reject; stdout={check_stdout}"
    );
    assert!(
        has_builtin_shadowing(&errors_array(&check_stdout), "sum"),
        "check lane must surface BuiltinShadowing; stdout={check_stdout}"
    );

    // Lane 2: eval --file.
    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path])
        .output()
        .expect("run chelis eval");
    let eval_stderr = String::from_utf8_lossy(&eval.stderr).to_string();
    assert!(
        !eval.status.success(),
        "eval lane must reject; stderr={eval_stderr}"
    );
    assert!(
        eval_stderr.contains("shadows the builtin"),
        "eval lane must surface the shadowing diagnostic, not the builtin's \
         arity error; stderr={eval_stderr}"
    );
    assert!(
        !eval_stderr.contains("expected int arg"),
        "the pre-fix builtin mis-dispatch error must be gone; stderr={eval_stderr}"
    );

    // Lane 3: build (no artifacts must be emitted for a rejected program).
    let outdir = tempdir().expect("build outdir");
    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(outdir.path())
        .args(["build", "--emit-c", path])
        .output()
        .expect("run chelis build");
    let build_stderr = String::from_utf8_lossy(&build.stderr).to_string();
    assert!(
        !build.status.success(),
        "build lane must reject; stderr={build_stderr}"
    );
    assert!(
        build_stderr.contains("shadows the builtin"),
        "build lane must surface the shadowing diagnostic; stderr={build_stderr}"
    );
}

/// The rejection is a SEMANTIC front-end error (it prevents a backend
/// segfault), not a style-gate rule: neither `--allow-style-violations`
/// nor `CHELIS_STYLE_GATE_DISABLE=1` may unlock the broken path. The
/// lane tests above already run with `CHELIS_STYLE_GATE_DISABLE=1`; this
/// test pins the flag combination explicitly on both execution lanes.
#[test]
fn issue_353_bypass_flags_do_not_unlock_shadowing() {
    let tmp = write_tempfile("issue353-bypass-", REPRO);
    let path = tmp.path().to_str().expect("path utf8");

    let eval = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--allow-style-violations", "--file", path])
        .output()
        .expect("run chelis eval");
    assert!(
        !eval.status.success(),
        "--allow-style-violations must not bypass the semantic rejection (eval)"
    );
    assert!(
        String::from_utf8_lossy(&eval.stderr).contains("shadows the builtin"),
        "eval with bypass flags must still surface the shadowing diagnostic"
    );

    let outdir = tempdir().expect("build outdir");
    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(outdir.path())
        .args(["build", "--emit-c", "--allow-style-violations", path])
        .output()
        .expect("run chelis build");
    assert!(
        !build.status.success(),
        "--allow-style-violations must not bypass the semantic rejection (build)"
    );
    assert!(
        String::from_utf8_lossy(&build.stderr).contains("shadows the builtin"),
        "build with bypass flags must still surface the shadowing diagnostic"
    );
}

/// Positive parity: near-miss names check clean, eval with correct
/// numerics ([1.0, 0.0] — the relu the user wrote), and build.
#[test]
fn issue_353_near_miss_names_check_eval_build_clean() {
    for def_name in ["sum2", "my_sum"] {
        let src = format!(
            "module NearMiss\n\
             def {def_name}(x: &tensor[batch, f32]) -> tensor[batch, f32] = relu(x)\n\
             out = {def_name}(to_tensor([1.0, -2.0]))\n"
        );
        let tmp = write_tempfile(&format!("issue353-{def_name}-"), &src);
        let path = tmp.path().to_str().expect("path utf8");

        let (code, stdout) = run_check(tmp.path());
        assert_eq!(
            code,
            Some(0),
            "near-miss `{def_name}` must check clean; stdout={stdout}"
        );
        assert!(
            errors_array(&stdout).is_empty(),
            "near-miss `{def_name}` must have empty errors; stdout={stdout}"
        );

        let eval = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file", path])
            .output()
            .expect("run chelis eval");
        let eval_stdout = String::from_utf8_lossy(&eval.stdout).to_string();
        assert!(
            eval.status.success(),
            "near-miss `{def_name}` must eval; stderr={}",
            String::from_utf8_lossy(&eval.stderr)
        );
        assert!(
            eval_stdout.contains("[1.0, 0.0]"),
            "near-miss `{def_name}` must produce the user def's numerics; stdout={eval_stdout}"
        );

        let outdir = tempdir().expect("build outdir");
        let build = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .current_dir(outdir.path())
            .args(["build", "--emit-c", path])
            .output()
            .expect("run chelis build");
        assert!(
            build.status.success(),
            "near-miss `{def_name}` must build; stderr={}",
            String::from_utf8_lossy(&build.stderr)
        );
    }
}

/// Spec §8.6 lane inventory, executable: the rejection binds in every
/// lane that runs the type checker — `cost` included — while
/// `chelis validate` is a syntax-grammar lane that never runs the type
/// checker and therefore does not surface this (or any other) semantic
/// rejection. Pinning both directions keeps the spec's lane list honest:
/// if `validate` ever grows a checker pass, this test fails and §8.6
/// must be updated with it.
#[test]
fn issue_353_cost_rejects_and_validate_is_syntax_only() {
    let tmp = write_tempfile("issue353-lanes2-", REPRO);
    let path = tmp.path().to_str().expect("path utf8");

    let cost = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["cost", path])
        .output()
        .expect("run chelis cost");
    let cost_stderr = String::from_utf8_lossy(&cost.stderr).to_string();
    assert!(
        !cost.status.success(),
        "cost lane runs the checker and must reject; stderr={cost_stderr}"
    );
    assert!(
        cost_stderr.contains("shadows the builtin"),
        "cost lane must surface the shadowing diagnostic; stderr={cost_stderr}"
    );

    let validate = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["validate", "--surf", path])
        .output()
        .expect("run chelis validate");
    assert!(
        validate.status.success(),
        "validate is a syntax-grammar lane (no type checker) and must \
         accept the grammatically valid reproducer; stderr={}",
        String::from_utf8_lossy(&validate.stderr)
    );
}

/// Reef package carve-out: a package-scoped `def sum` stays accepted.
/// Reef rewrites package decl names to internal `pkg__...` names (and
/// rewrites their call sites with them) before the checker runs, so a
/// package def neither collides with the builtin table nor
/// mis-dispatches — the user def genuinely wins inside a package. The
/// stdlib's `Std.Test.fail` relies on this.
#[test]
fn issue_353_reef_package_def_sum_still_checks_clean() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    std::fs::create_dir_all(root.join("src")).expect("mkdir src");
    std::fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\n\
             name = \"shadowpkg\"\n\
             version = \"0.1.0\"\n\
             compiler = \"={}\"\n\
             module_prefix = \"Pkg\"\n",
            chelis_compiler_api::COMPILER_VERSION
        ),
    )
    .expect("write reef.toml");
    std::fs::write(
        root.join("src/app.ch"),
        "module Pkg.App\n\
         def sum(x: &tensor[2, f32]) -> tensor[2, f32] = relu(x)\n\
         def use_it() -> tensor[2, f32] = sum(to_tensor([1.0, -2.0]))\n",
    )
    .expect("write app.ch");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", root.to_str().expect("path utf8")])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    assert_eq!(
        output.status.code(),
        Some(0),
        "package-scoped def sum must stay accepted (reef internal-name \
         rewrite isolates it); stdout={stdout}"
    );
    assert!(
        !stdout.contains("BuiltinShadowing"),
        "no BuiltinShadowing error may fire inside a reef package; stdout={stdout}"
    );
}
