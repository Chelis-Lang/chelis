//! Integration tests for `redundant-linearity-call` autofix re-enablement.
//!
//! The autofix was disabled by 477bd0d because the source-only lint walker
//! could not prove that stripping an explicit `copy()` preserved semantics.
//! After Item 1 of the 0.7.6 toolchain hygiene workstream landed (PR #29),
//! implicit linearity handles both within-call fan-out and cross-statement
//! var-RHS aliasing, which is the broad class the lint targets.
//!
//! These fixtures pin the re-enablement scope:
//!
//! - F1 trivial strip: `realize(copy(w))` → `realize(w)`
//! - F2 within-call fan-out: `mul(copy(w), copy(w))` → `mul(w, w)`
//! - F3 cross-statement var-RHS aliasing:
//!   `let alias = copy(x); mul(x, alias)` → `let alias = x; mul(x, alias)`
//! - F4 architectural invariant: every flagged program survives the typed
//!   pipeline after autofix.
//!
//! Each fixture asserts:
//!   (a) `chelis lint --check` flags the `copy()` as redundant
//!   (b) `chelis lint --fix` strips it
//!   (c) the stripped output parses, type-checks, and evaluates identically
//!
//! Fixtures are `#[ignore]` until the autofix is re-enabled (Item 5).

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_and_format(path: &Path, source: &str) {
    fs::write(path, source).expect("write source");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
}

fn chelis_check_ok(path: &Path) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check");
    assert!(
        output.status.success(),
        "chelis check failed for {}: stdout={} stderr={}",
        path.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8(output.stdout).expect("utf8 stdout")
}

fn chelis_eval_ok(path: &Path) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval");
    assert!(
        output.status.success(),
        "chelis eval failed for {}: stdout={} stderr={}",
        path.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    String::from_utf8(output.stdout).expect("utf8 stdout")
}

fn chelis_lint_check_stdout(path: &Path) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["lint", "--check", path.to_str().unwrap()])
        .output()
        .expect("chelis lint");
    String::from_utf8(output.stdout).expect("utf8 stdout")
}

fn chelis_lint_fix(path: &Path) {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["lint", "--fix", path.to_str().unwrap()])
        .assert()
        .success();
}

/// Run the full assertion contract for a fixture: setup, lint flags copy,
/// autofix strips, post-fix source parses/type-checks/evals to the same
/// eval output as the input.
fn assert_autofix_strips_and_preserves(name: &str, source: &str) {
    let dir = tempdir().expect("tempdir");

    // Pre-fix file: copy the source for evaluation baseline.
    let pre_path = dir.path().join(format!("{name}_pre.ch"));
    write_and_format(&pre_path, source);
    let pre_eval = chelis_eval_ok(&pre_path);
    assert!(
        pre_eval.contains("tensor(") || !pre_eval.is_empty(),
        "{name}: pre-fix eval should produce non-empty output: {pre_eval}",
    );

    // Lint check should flag at least one redundant-linearity-call.
    let post_path = dir.path().join(format!("{name}_post.ch"));
    write_and_format(&post_path, source);
    let lint_stdout = chelis_lint_check_stdout(&post_path);
    assert!(
        lint_stdout.contains("redundant-linearity-call"),
        "{name}: lint --check should flag redundant-linearity-call; got: {lint_stdout}",
    );

    // Apply the autofix; the file must have no remaining `copy(` substring
    // for the simple let-binding shapes used in these fixtures.
    chelis_lint_fix(&post_path);
    let after = fs::read_to_string(&post_path).expect("read post-fix");
    assert!(
        !after.contains("copy("),
        "{name}: autofix should strip every flagged `copy(`; file still contains it:\n{after}",
    );

    // Post-fix source must still pass `chelis check` and `chelis eval`,
    // and produce identical eval output.
    chelis_check_ok(&post_path);
    let post_eval = chelis_eval_ok(&post_path);
    assert_eq!(
        pre_eval, post_eval,
        "{name}: eval output diverged after autofix\npre:\n{pre_eval}\npost:\n{post_eval}",
    );
}

#[test]
fn f1_trivial_strip() {
    let source = "\
def f(w: tensor[2, f32]) -> tensor[2, f32] = realize(copy(w))

result = f(to_tensor([1.0, 2.0]))
";
    assert_autofix_strips_and_preserves("f1_trivial_strip", source);
}

#[test]
fn f2_within_call_fan_out() {
    let source = "\
def f(w: tensor[2, f32]) -> tensor[2, f32] = mul(copy(w), copy(w))

result = f(to_tensor([1.0, 2.0]))
";
    assert_autofix_strips_and_preserves("f2_within_call_fan_out", source);
}

#[test]
fn f3_cross_statement_var_rhs_aliasing() {
    // After PR #29 (Item 1), implicit linearity allows reads through a
    // var-RHS let alias. The lint flags the explicit `copy()` on the RHS
    // of an alias binding because implicit linearity now inserts the
    // equivalent IR. Surf uses block binding syntax `x = expr` (not
    // `let x = expr;`) per `spec/02-surf-syntax.md` §P5. Avoid
    // tuple/record destructure (Linearity-F2 silent false-negative).
    let source = "\
def f(x: tensor[2, f32]) -> tensor[2, f32] = {
  alias = copy(x)
  mul(x, alias)
}

result = f(to_tensor([1.0, 2.0]))
";
    assert_autofix_strips_and_preserves("f3_cross_statement_var_rhs_aliasing", source);
}

/// Architectural invariant: enumerate a small adversarial corpus and assert
/// that, for every program in which (a) the pre-fix source type-checks and
/// evaluates, and (b) the lint flags any `copy()`, the autofix output
/// passes `chelis check` and evaluates identically.
///
/// Excludes tuple/record destructure programs per the Linearity-F2 silent
/// false-negative (see `docs/gap_synthesis.md` §5). Programs whose pre-fix
/// form does not type-check are skipped — they aren't a legitimate target
/// for the autofix invariant.
#[test]
fn f4_architectural_invariant_corpus() {
    let corpus: &[(&str, &str)] = &[
        (
            "branch_arm_copy",
            "\
def f(x: tensor[2, f32], flag: bool) -> tensor[2, f32] =
  if flag then realize(copy(x)) else realize(x)

result = f(to_tensor([1.0, 2.0]), true)
",
        ),
        (
            "nested_let_copy_chain",
            "\
def f(x: tensor[2, f32]) -> tensor[2, f32] = {
  a = copy(x)
  b = copy(x)
  add(a, b)
}

result = f(to_tensor([1.0, 2.0]))
",
        ),
        (
            "copy_then_consume",
            "\
def f(w: tensor[2, f32]) -> tensor[2, f32] = {
  z = realize(copy(w))
  add(z, w)
}

result = f(to_tensor([1.0, 2.0]))
",
        ),
        (
            "pipe_stage_copy",
            "\
def f(w: tensor[2, f32]) -> tensor[2, f32] = copy(w) |> realize

result = f(to_tensor([1.0, 2.0]))
",
        ),
        (
            "binary_op_with_copy",
            "\
def f(w: tensor[2, f32]) -> tensor[2, f32] = add(copy(w), w)

result = f(to_tensor([1.0, 2.0]))
",
        ),
        (
            "scalar_mul_with_copy",
            "\
def f(w: tensor[2, f32]) -> tensor[2, f32] = scalar_mul(copy(w), cast(2.0, f32))

result = f(to_tensor([1.0, 2.0]))
",
        ),
        (
            "let_aliased_copy_then_consume",
            "\
def f(x: tensor[2, f32]) -> tensor[2, f32] = {
  alias = copy(x)
  add(x, alias)
}

result = f(to_tensor([1.0, 2.0]))
",
        ),
    ];

    let mut covered = 0usize;
    for (name, source) in corpus {
        let dir = tempdir().expect("tempdir");
        let probe = dir.path().join(format!("{name}_probe.ch"));
        fs::write(&probe, source).expect("write fixture");
        let fmt = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .args(["fmt", "--inplace", probe.to_str().unwrap()])
            .output()
            .expect("fmt");
        if !fmt.status.success() {
            // Pre-fix source is not canonically formattable; skip — this
            // isn't a legitimate autofix target.
            continue;
        }
        // Pre-fix must type-check and evaluate; otherwise the invariant
        // does not apply to this program.
        let check = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .args(["check", probe.to_str().unwrap()])
            .output()
            .expect("check");
        if !check.status.success() {
            continue;
        }
        let eval = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .args(["eval", "--file", probe.to_str().unwrap()])
            .output()
            .expect("eval");
        if !eval.status.success() {
            continue;
        }

        let lint_stdout = chelis_lint_check_stdout(&probe);
        if !lint_stdout.contains("redundant-linearity-call") {
            continue;
        }
        // F4 invariant (architectural): post-fix file must still pass
        // `chelis check` and `chelis eval` with identical output. The
        // autofix may legitimately *keep* a flagged copy() when the
        // typed-pipeline gate rejects the strip (e.g., when stripping
        // would break a later consume site). What it MUST NOT do is
        // produce an output that fails the typed pipeline.
        assert_autofix_preserves_typed_pipeline(name, source);
        covered += 1;
    }
    assert!(
        covered >= 4,
        "f4 corpus should exercise at least 4 distinct shapes; only {covered} covered",
    );
}

/// Relaxed contract for F4: the autofix output must still parse,
/// type-check, and evaluate identically to the pre-fix source. The
/// autofix may leave some flagged `copy()` calls in place when stripping
/// them would break the typed pipeline (the CLI driver gates each
/// replacement independently). That's the safety bar working as intended.
fn assert_autofix_preserves_typed_pipeline(name: &str, source: &str) {
    let dir = tempdir().expect("tempdir");

    let pre_path = dir.path().join(format!("{name}_pre.ch"));
    write_and_format(&pre_path, source);
    let pre_eval = chelis_eval_ok(&pre_path);

    let post_path = dir.path().join(format!("{name}_post.ch"));
    write_and_format(&post_path, source);
    chelis_lint_fix(&post_path);

    // Post-fix must still pass check + eval with identical output.
    chelis_check_ok(&post_path);
    let post_eval = chelis_eval_ok(&post_path);
    assert_eq!(
        pre_eval, post_eval,
        "{name}: eval output diverged after autofix\npre:\n{pre_eval}\npost:\n{post_eval}",
    );
}
