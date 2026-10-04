//! `redundant-linearity-call` autofixes preserve the typed program's
//! behavior across direct calls, fan-out, aliases, and higher-order
//! operations. These fixtures cover:
//!
//! - F1 trivial strip: `realize(copy(w))` -> `realize(w)`
//! - F2 within-call fan-out: `mul(copy(w), copy(w))` -> `mul(w, w)`
//! - F3 cross-statement var-RHS aliasing:
//!   `let alias = copy(x); mul(x, alias)` -> `let alias = x; mul(x, alias)`
//! - F4 every flagged program survives the typed pipeline after autofix
//! - F5 Shape A borrow-to-owned at return position: `def f[a](x: &T) -> T = copy(x)`
//! - F6 Shape A with use-site driver
//! - F7 Shape B grad fan-out: `grad(f, wrt=p)(copy(args)...)` followed by
//!   a trailing borrow-read
//! - F8 Shape B four-arg mse fan-out with copies on every arg
//! - F9 Shape B vmap fan-out: `vmap(f)(copy(arg))`
//!
//! Each fixture asserts:
//!   (a) `chelis lint --check` flags the `copy()` as redundant
//!   (b) `chelis lint --fix` strips it
//!   (c) the stripped output parses, type-checks, and evaluates identically
//!

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
    // A var-RHS alias permits later borrows, so an explicit copy on
    // the binding RHS is redundant.
    let source = "\
def f(x: tensor[2, f32]) -> tensor[2, f32] = {
  alias = copy(x)
  mul(x, alias)
}

result = f(to_tensor([1.0, 2.0]))
";
    assert_autofix_strips_and_preserves("f3_cross_statement_var_rhs_aliasing", source);
}

/// For every checked, evaluable corpus program whose copy is flagged
/// as redundant, the autofix remains checked and evaluates identically.
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

/// F5: A return-position implicit copy permits a borrowed input to
/// satisfy an owned result after redundant source `copy()` is stripped.
#[test]
fn f5_shape_a_borrow_return_position() {
    let source = "\
def identity_dim[a](x: &tensor[a, f32]) -> tensor[a, f32] = copy(x)

input = to_tensor([1.0, 2.0])
result = identity_dim(&input)
";
    assert_autofix_strips_and_preserves("f5_shape_a_borrow_return_position", source);
}

/// F6: Shape A with a downstream consumer that exercises the inserted
/// return-position copy at a call site.
#[test]
fn f6_shape_a_with_use_site_driver() {
    let source = "\
def identity_dim[a](x: &tensor[a, f32]) -> tensor[a, f32] = copy(x)

def driver(x: tensor[3, f32]) -> tensor[3, f32] = {
  y = identity_dim(&x)
  add(y, y)
}

input = to_tensor([1.0, 2.0, 3.0])
result = driver(input)
";
    assert_autofix_strips_and_preserves("f6_shape_a_with_use_site_driver", source);
}

/// F7: Gradient applications borrow their arguments, so a trailing
/// borrow remains valid after redundant copies are stripped.
#[test]
fn f7_shape_b_grad_fanout() {
    let source = "\
def my_loss(w: tensor[3, f32], b: tensor[3, f32]) -> tensor[f32] = {
  d = sub(w, b)
  sq = mul(d, d)
  sum(sq, 0)
}

def step(w: tensor[3, f32], b: tensor[3, f32]) -> tensor[3, f32] = {
  dw = grad(my_loss, wrt=w)(copy(w), copy(b))
  trailing = sub(w, dw)
  add(trailing, b)
}

w = to_tensor([1.0, 2.0, 3.0])
b = to_tensor([0.5, 0.5, 0.5])
result = step(w, b)
";
    assert_autofix_strips_and_preserves("f7_shape_b_grad_fanout", source);
}

/// F8: Shape B four-arg mse-shape with copies on every arg. Mirrors the
/// hello-chelis linreg.ch sgd_step pattern at vector arity 3.
#[test]
fn f8_shape_b_grad_fanout_four_arg_mse() {
    let source = "\
def mse_loss(x: tensor[3, f32], y: tensor[3, f32], w: tensor[3, f32], b: tensor[3, f32]) -> tensor[f32] = {
  prod = mul(w, b)
  d = sub(prod, x)
  e = sub(d, y)
  sq = mul(e, e)
  sum(sq, 0)
}

def sgd_step(x: tensor[3, f32], y: tensor[3, f32], w: tensor[3, f32], b: tensor[3, f32]) -> tensor[3, f32] = {
  dw = grad(mse_loss, wrt=w)(copy(x), copy(y), copy(w), copy(b))
  db = grad(mse_loss, wrt=b)(copy(x), copy(y), copy(w), copy(b))
  new_w = sub(w, dw)
  new_b = sub(b, db)
  add(new_w, new_b)
}

x = to_tensor([1.0, 2.0, 3.0])
y = to_tensor([0.1, 0.2, 0.3])
w = to_tensor([0.5, 0.5, 0.5])
b = to_tensor([0.25, 0.25, 0.25])
result = sgd_step(x, y, w, b)
";
    assert_autofix_strips_and_preserves("f8_shape_b_grad_fanout_four_arg_mse", source);
}

/// F9: A vmap application borrows its input, so the surrounding
/// redundant copy can be stripped.
#[test]
fn f9_shape_b_vmap_observational() {
    let source = "\
def my_op(w: tensor[3, f32]) -> tensor[f32] = sum(w, 0)

def step(ws: tensor[5, 3, f32]) -> tensor[5, f32] = {
  out = vmap(my_op)(copy(ws))
  trailing = sub(out, out)
  trailing
}

ws = to_tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
result = step(ws)
";
    assert_autofix_strips_and_preserves("f9_shape_b_vmap_observational", source);
}

/// F10: No warning is emitted when stripping `copy()` would leave
/// a borrow-to-owned type mismatch. The typed-pipeline gate rejects
/// that rewrite, and the warning mirrors the gate.
#[test]
fn f10_warning_suppressed_when_typed_pipeline_rejects_strip_on_borrow() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("copy_on_borrow.ch");
    let source = "\
def consume_owned[n](x: tensor[n, f32]) -> tensor[n, f32] = realize(x)

def caller[n](y: &tensor[n, f32]) -> tensor[n, f32] = consume_owned(copy(y))

input = to_tensor([1.0, 2.0])
result = caller(&input)
";
    write_and_format(&path, source);

    // The pre-fix source type-checks and evaluates: the explicit copy
    // is structurally necessary because `consume_owned` takes an owned
    // tensor and `y` is a borrow.
    chelis_check_ok(&path);
    chelis_eval_ok(&path);

    // The warning must NOT fire: stripping `copy(y)` would leave the
    // call as `consume_owned(y)` which fails because `y: &tensor[n,
    // f32]` is not assignable to a parameter expecting an owned
    // tensor. The copy is structurally necessary, not migration-compat
    // noise.
    let lint_stdout = chelis_lint_check_stdout(&path);
    assert!(
        !lint_stdout.contains("redundant-linearity-call"),
        "f10: warning fired on structurally-necessary copy(borrow); stripping would fail the typed pipeline; got:\n{lint_stdout}",
    );
}

/// F11 (Item 1 negative control): the warning MUST still fire when
/// the strip is safe (typed pipeline accepts the post-strip source).
/// Pin both halves of the suppression gate so the fix does not
/// over-suppress.
#[test]
fn f11_warning_still_fires_when_strip_is_safe() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("strip_safe.ch");
    let source = "\
def f(w: tensor[2, f32]) -> tensor[2, f32] = realize(copy(w))

result = f(to_tensor([1.0, 2.0]))
";
    write_and_format(&path, source);

    let lint_stdout = chelis_lint_check_stdout(&path);
    assert!(
        lint_stdout.contains("redundant-linearity-call"),
        "f11: warning should still fire when the strip is safe; got:\n{lint_stdout}",
    );
    assert!(
        lint_stdout.contains("[fix]"),
        "f11: `[fix]` marker should be present when the strip is safe; got:\n{lint_stdout}",
    );
}

/// F12 (Lint-PreferPipeRedundantLinearityPair-F1, 0.7.9 cleanup), rewritten
/// by the [05-OP-54]/[05-OP-67] split. Coral reported that the pipe form of
/// the List slice put a literal single-argument call in the source text
/// while being semantically the two-argument slice, so the rule's regex
/// matched it and offered a strip that would fail the typed pipeline;
/// `check_mirrors_fix` suppressed the warning.
///
/// **The ambiguity the suppression existed for is gone**, and the fixture
/// has to change to compile at all: `xs |> drop(n)` is now an arity error,
/// and a fixture asserting lint silence over source the checker rejects
/// asserts nothing -- this rule is a text walker that never type-checks
/// its input.
///
/// What this now pins is the user-facing property, that the List slice in
/// pipe form draws no linearity warning, and TWO independent mechanisms
/// each suffice for it: the rule's `\b(copy|drop)\s*\(` regex does not
/// name `skip`, and `check_mirrors_fix` would suppress the warning anyway
/// because stripping `skip(n)` to `n` fails the typed pipeline. Measured:
/// adding `skip` to that regex does not make this test fire. So it is a
/// disposition lock over both paths rather than a regression test for
/// either one, and it cannot tell you which is doing the work.
///
/// Coral's own workaround, writing `skip(xs, one_i64())` directly, is now
/// simply the correct spelling.
#[test]
fn f12_linearity_warning_does_not_reach_the_list_slice_in_pipe_form() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pipe_list_skip.ch");
    let source = "\
def f(xs: List[i64], n: i64) -> List[i64] = xs |> skip(n)
";
    write_and_format(&path, source);

    let lint_stdout = chelis_lint_check_stdout(&path);
    assert!(
        !lint_stdout.contains("redundant-linearity-call"),
        "f12: the linearity rule fired on the List slice in pipe form; `skip` is [05-OP-54]'s container operation, not a linearity call; got:\n{lint_stdout}",
    );
}

/// F13 (Item 3 negative control): the legitimate single-arg
/// `drop(x)` redundancy warning must still fire (and offer a `[fix]`)
/// on a tensor argument where the strip is safe under implicit
/// linearity. Pins that the Item 3 fix does not over-suppress
/// legitimate linearity-primitive `drop()` flagging. The typed
/// pipeline accepts both the original and the stripped program.
#[test]
fn f13_warning_still_fires_on_legitimate_redundant_drop() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("redundant_drop.ch");
    let source = "\
def f(w: tensor[2, f32]) -> tensor[2, f32] = realize(w)
result = drop(f(to_tensor([1.0, 2.0])))
";
    write_and_format(&path, source);

    let lint_stdout = chelis_lint_check_stdout(&path);
    assert!(
        lint_stdout.contains("redundant-linearity-call") && lint_stdout.contains("[fix]"),
        "f13: warning should still fire with a fix on legitimate single-arg drop(tensor); got:\n{lint_stdout}",
    );
}

/// F13b (chelis#3108): a strip is offered only as a rewrite that
/// preserves semantics, which needs the typed pipeline to accept the
/// original too. `drop(realize(w))` returns `()` where the signature
/// promises a tensor, so the original is rejected; stripping `drop`
/// would turn a rejected program into an accepted one, which is not a
/// preserving rewrite. The warning is suppressed and `--fix` leaves
/// the source unchanged.
#[test]
fn f13b_no_strip_offered_when_the_original_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rejected_drop.ch");
    let source = "\
def f(w: tensor[2, f32]) -> tensor[2, f32] = drop(realize(w))
result = f(to_tensor([1.0, 2.0]))
";
    write_and_format(&path, source);
    let before = fs::read_to_string(&path).expect("read fixture");

    let lint_stdout = chelis_lint_check_stdout(&path);
    assert!(
        !lint_stdout.contains("redundant-linearity-call"),
        "f13b: no strip is offered for a rejected original; got:\n{lint_stdout}",
    );
    chelis_lint_fix(&path);
    assert_eq!(
        fs::read_to_string(&path).expect("read fixture"),
        before,
        "f13b: --fix must not rewrite a rejected original",
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
