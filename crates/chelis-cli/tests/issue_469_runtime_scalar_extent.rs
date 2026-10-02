//! chelis#469: an `expand`/`insert` size of any provenance executes, and
//! `chelis eval` agrees with compiled C.
//!
//! `spec/04-type-system.md` section 4.7.2 admits any `i64` size and forbids
//! rejecting an extent "because of its provenance"; section 4.7.4 says eval
//! and C "execute the same graph and checks". Each positive row runs one
//! program through `chelis check`, `chelis eval --file`, and `chelis build
//! --target c` plus the produced executable, and requires the two lanes to
//! print the same bytes. Before chelis#469 the checker rejected every
//! scalar-parameter spelling, and lowering rejected a bare name with no tensor
//! source even after the checker had accepted it (a top-level binding).
//!
//! The negative rows are the section 4.7.2 failures that must keep failing,
//! now reachable from these spellings: a runtime negative size traps `Domain`,
//! a runtime size disagreeing with a literal result claim traps `Domain`, a
//! claim the call's constant argument refutes is rejected when the activation
//! is lowered, and two different runtime extents meeting elementwise trap.
//! Each is required to render identically on both lanes.
//!
//! The checker-side rows are
//! `crates/chelis-types/tests/issue_469_runtime_scalar_extent_check.rs`.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::{Command as StdCommand, Output};
use tempfile::{TempDir, tempdir};

/// A runtime `i64` the compiler cannot fold: the sum of a tensor's elements.
/// `RUNTIME_FOUR` is 4, and every row that needs an extent the lowering cannot
/// see uses it, so a claim is checked at run time rather than refuted early.
const RUNTIME_FOUR: &str = "tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0))";

/// The operand and size spellings, once per operation. `expand` broadcasts an
/// existing unit axis, so its operand is `[1, 2]`; `insert` adds one, so its
/// operand is `[2]`. Both print `[3, 2]` for a size of 3.
const OPERATIONS: [(&str, &str); 2] = [
    ("expand", "to_tensor([[1i64, 2i64]])"),
    ("insert", "to_tensor([1i64, 2i64])"),
];

const THREE_BY_TWO: &str = "tensor(shape=[3, 2], data=[1, 2, 1, 2, 1, 2])";

fn cli() -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

fn combined(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn write_fixture(dir: &TempDir, stem: &str, source: &str) -> std::path::PathBuf {
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write fixture");
    path
}

fn check(path: &Path) -> Output {
    cli()
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check")
}

fn eval(path: &Path) -> Output {
    cli()
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval")
}

/// Build natively for C and run the executable. A build failure is returned
/// as the build's own output, so a row can tell a lowering rejection from a
/// run-time trap.
fn build_and_run(dir: &TempDir, stem: &str, path: &Path) -> Output {
    let out_dir = dir.path().join(format!("{stem}-out"));
    let build = cli()
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    if !build.status.success() {
        return build;
    }
    StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run the compiled executable")
}

/// Check, evaluate and build `source`; require both lanes to succeed, print
/// the same bytes, and print `expected` among them.
fn assert_lanes_agree(stem: &str, source: &str, expected: &str) {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(&dir, stem, source);
    let checked = check(&path);
    assert!(
        checked.status.success(),
        "{stem}: check must accept a runtime i64 size: {}",
        combined(&checked)
    );
    let evaluated = eval(&path);
    assert!(
        evaluated.status.success(),
        "{stem}: eval must execute: {}",
        combined(&evaluated)
    );
    let compiled = build_and_run(&dir, stem, &path);
    assert!(
        compiled.status.success(),
        "{stem}: the C build must build and run: {}",
        combined(&compiled)
    );
    let eval_out = String::from_utf8_lossy(&evaluated.stdout).to_string();
    let c_out = String::from_utf8_lossy(&compiled.stdout).to_string();
    assert_eq!(eval_out, c_out, "{stem}: eval and C print the same bytes");
    assert!(
        eval_out.contains(expected),
        "{stem}: expected `{expected}` in {eval_out}"
    );
}

/// Check accepts `source`; both lanes then fail at run time with the same
/// rendering, which contains every line of `lines`.
fn assert_lanes_trap_identically(stem: &str, source: &str, lines: &[&str]) {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(&dir, stem, source);
    let checked = check(&path);
    assert!(
        checked.status.success(),
        "{stem}: the failure is a run-time one, so check accepts: {}",
        combined(&checked)
    );
    let evaluated = eval(&path);
    let compiled = build_and_run(&dir, stem, &path);
    let eval_out = combined(&evaluated);
    let c_out = combined(&compiled);
    for (lane, output, text) in [("eval", &evaluated, &eval_out), ("c", &compiled, &c_out)] {
        assert!(!output.status.success(), "{stem}: {lane} must trap: {text}");
        assert!(
            !text.contains("out = "),
            "{stem}: {lane} must not print the trapped binding: {text}"
        );
        for line in lines {
            assert!(
                text.contains(line),
                "{stem}: {lane} must print `{line}`: {text}"
            );
        }
    }
    assert_eq!(
        eval_out.trim_end(),
        c_out.trim_end(),
        "{stem}: the two lanes render the trap identically"
    );
}

fn domain_trap_line(op: &str) -> String {
    format!("numeric trap: domain in {op} at i64")
}

#[test]
fn a_bare_i64_parameter_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_param"),
            &format!("def f(k: i64) = {op}({operand}, 0, k)\nout = f(3i64)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_cast_i32_parameter_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_cast"),
            &format!("def f(j: i32) = {op}({operand}, 0, cast(j, i64))\nout = f(3i32)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_let_bound_parameter_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_let"),
            &format!("def f(k: i64) = {{\n  m = k\n  {op}({operand}, 0, m)\n}}\nout = f(3i64)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn an_arithmetic_size_over_a_parameter_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_arith"),
            &format!("def f(k: i64) = {op}({operand}, 0, add(k, 1i64))\nout = f(2i64)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_user_function_result_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_userfn"),
            &format!(
                "def g(k: i64) -> i64 = add(k, 1i64)\n\
                 def f(k: i64) = {op}({operand}, 0, g(k))\n\
                 out = f(2i64)\n"
            ),
            THREE_BY_TWO,
        );
    }
}

/// A top-level binding whose value the checker folds. Before chelis#469 check
/// and eval accepted it and only the C build rejected it, because lowering
/// found no tensor source for the global's name.
#[test]
fn a_top_level_static_binding_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_toplevel_static"),
            &format!("h = add(1i64, 2i64)\nout = {op}({operand}, 0, h)\n"),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_top_level_runtime_binding_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_toplevel_runtime"),
            &format!(
                "def g(k: i64) -> i64 = add(k, 1i64)\n\
                 h = g(2i64)\n\
                 out = {op}({operand}, 0, h)\n"
            ),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_tuple_projection_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_tuple"),
            &format!("def f(t: (i64, i64)) = {op}({operand}, 0, t.0)\nout = f((3i64, 9i64))\n"),
            THREE_BY_TWO,
        );
    }
}

/// `expand.sourceless_size.pipe_position` and its lane twin in
/// `scripts/runtime_extent_oracle.py`: the size reaches `expand` through a
/// pipe stage, which chelis#1791 made the same tree as the direct spelling.
#[test]
fn a_pipe_position_runtime_size_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_pipe"),
            &format!(
                "def f(j: i32) = {{\n  m = j |> cast(i64)\n  {operand} |> {op}(0i32, m)\n}}\n\
                 out = f(3i32)\n"
            ),
            THREE_BY_TWO,
        );
    }
}

#[test]
fn a_zero_runtime_size_executes_on_both_lanes() {
    assert_lanes_agree(
        "insert_zero",
        "def f(k: i64) = insert(to_tensor([1i64, 2i64]), 0, k)\nout = f(0i64)\n",
        "tensor(shape=[0, 2], data=[])",
    );
}

/// The Std.Datetime shape that motivated the work: a table whose only extent
/// source is a scalar horizon, scanned and reduced.
#[test]
fn a_horizon_sized_table_executes_on_both_lanes() {
    assert_lanes_agree(
        "horizon_table",
        "def table(horizon: i64) = cumsum(expand(to_tensor([1i64]), 0, horizon), 0)\n\
         def total(horizon: i64) -> i64 = \
         tensor_to_scalar(sum(expand(to_tensor([1i64]), 0, horizon), 0))\n\
         t = table(5i64)\n\
         s = total(5i64)\n",
        "t = tensor(shape=[5], data=[1, 2, 3, 4, 5])\ns = 5",
    );
}

/// One runtime extent reaching two operands, then a callee generic over it.
#[test]
fn a_runtime_extent_composes_on_both_lanes() {
    assert_lanes_agree(
        "compose",
        "def g[n](t: &tensor[n, 2, i64]) -> tensor[n, 2, i64] = mul(t, t)\n\
         def f(k: i64) = g(add(insert(to_tensor([1i64, 2i64]), 0, k), \
         insert(to_tensor([1i64, 2i64]), 0, k)))\n\
         out = f(3i64)\n",
        "tensor(shape=[3, 2], data=[4, 16, 4, 16, 4, 16])",
    );
}

#[test]
fn a_runtime_negative_size_traps_domain_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_trap_identically(
            &format!("{op}_negative"),
            &format!("def f(k: i64) = {op}({operand}, 0, k)\nout = f(sub({RUNTIME_FOUR}, 5i64))\n"),
            &[
                "Domain: expansion axis or extent outside domain",
                &domain_trap_line(op),
            ],
        );
    }
}

#[test]
fn a_runtime_size_disagreeing_with_a_literal_claim_traps_domain_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_trap_identically(
            &format!("{op}_claim"),
            &format!(
                "def f(k: i64) -> tensor[3, 2, i64] = {op}({operand}, 0, k)\n\
                 out = f({RUNTIME_FOUR})\n"
            ),
            &["claimed = 3", &domain_trap_line(op)],
        );
    }
}

/// The agreeing control for the claim row: the same declaration over a runtime
/// size equal to its claim executes, so the trap above is the claim's.
#[test]
fn a_runtime_size_agreeing_with_a_literal_claim_executes_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        assert_lanes_agree(
            &format!("{op}_claim_ok"),
            &format!(
                "def f(k: i64) -> tensor[4, 2, i64] = {op}({operand}, 0, k)\n\
                 out = f({RUNTIME_FOUR})\n"
            ),
            "tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        );
    }
}

/// A constant argument makes the inlined body's extent a literal, so the
/// declared claim is refuted when the activation is lowered: section 4.7's
/// "A violation proven from literals is a type error", identical on both
/// lanes and carrying no run-time trap.
#[test]
fn a_claim_a_constant_argument_refutes_is_rejected_identically_on_both_lanes() {
    for (op, operand) in OPERATIONS {
        let dir = tempdir().expect("tempdir");
        let path = write_fixture(
            &dir,
            &format!("{op}_refuted"),
            &format!("def f(k: i64) -> tensor[3, 2, i64] = {op}({operand}, 0, k)\nout = f(4i64)\n"),
        );
        let evaluated = eval(&path);
        let compiled = build_and_run(&dir, &format!("{op}_refuted"), &path);
        let eval_out = combined(&evaluated);
        let c_out = combined(&compiled);
        for (lane, output, text) in [("eval", &evaluated, &eval_out), ("c", &compiled, &c_out)] {
            assert_eq!(output.status.code(), Some(1), "{op}: {lane}: {text}");
            assert!(
                text.contains(
                    "declares extent 3 at result axis 0, but the inlined body produces 4"
                ),
                "{op}: {lane}: {text}"
            );
            assert!(!text.contains("numeric trap"), "{op}: {lane}: {text}");
        }
        assert_eq!(eval_out, c_out, "{op}: one rendering on both lanes");
    }
}

#[test]
fn two_different_runtime_extents_meeting_elementwise_trap_on_both_lanes() {
    let dir = tempdir().expect("tempdir");
    let path = write_fixture(
        &dir,
        "mismatch",
        &format!(
            "def f(k: i64, j: i64) = add(insert(to_tensor([1i64, 2i64]), 0, k), \
             insert(to_tensor([1i64, 2i64]), 0, j))\n\
             out = f(3i64, {RUNTIME_FOUR})\n"
        ),
    );
    let checked = check(&path);
    assert!(
        checked.status.success(),
        "the disagreement is a run-time one, so check accepts: {}",
        combined(&checked)
    );
    let evaluated = eval(&path);
    let compiled = build_and_run(&dir, "mismatch", &path);
    for (lane, output) in [("eval", &evaluated), ("c", &compiled)] {
        let text = combined(output);
        assert!(!output.status.success(), "{lane} must trap: {text}");
        assert!(!text.contains("out = "), "{lane}: {text}");
    }
}

/// Section 4.7.2: a size name that denotes both a value binding and an
/// in-scope dimension is a type error. Here the local `n` shadows the binder
/// `n`; eval used to print the local's 3 while C printed the operand's 2.
/// Check rejects it, and so do eval and build, which check first. The
/// checker's rows for every spelling are in
/// `crates/chelis-types/tests/issue_469_runtime_scalar_extent_check.rs`.
#[test]
fn a_size_name_that_is_both_a_value_and_a_dimension_is_rejected_on_every_command() {
    let dir = tempdir().expect("tempdir");
    let stem = "ambiguous";
    let path = write_fixture(
        &dir,
        stem,
        "def f[n](x: tensor[n, i64], m: i64) -> i64 = {\n  n = m\n  y = insert(x, 0, n)\n  shape(y, 0)\n}\n\
         out = f(to_tensor([1i64, 2i64]), 3i64)\n",
    );
    let needle = "insert size names `n`, which is both a value of type i64 and a dimension \
                  binder of the enclosing definition, so the extent it denotes is ambiguous";
    for (command, output) in [
        ("check", check(&path)),
        ("eval", eval(&path)),
        ("build", build_and_run(&dir, stem, &path)),
    ] {
        let text = combined(&output);
        assert!(!output.status.success(), "{command} must reject: {text}");
        assert!(text.contains(needle), "{command}: {text}");
        assert!(!text.contains("out = "), "{command}: {text}");
    }
}

/// The renamed twins of the ambiguous spellings: once the value and the
/// dimension have different names, each one means what it says, on both
/// lanes.
#[test]
fn renamed_values_and_dimensions_execute_on_both_lanes() {
    let operand = "to_tensor([1i64, 2i64])";
    for (stem, source, expected) in [
        (
            "renamed_local",
            format!(
                "def f[n](x: tensor[n, i64], m: i64) -> i64 = {{\n  k = m\n  y = insert(x, 0, k)\n  shape(y, 0)\n}}\n\
                 out = f({operand}, 3i64)\n"
            ),
            "out = 3".to_string(),
        ),
        (
            "renamed_parameter",
            format!(
                "def f[n](x: tensor[n, i64], k: i64) = insert(x, 0, k)\nout = f({operand}, 3i64)\n"
            ),
            THREE_BY_TWO.to_string(),
        ),
        (
            "renamed_static_global",
            format!(
                "h = 5i64\n\
                 def f[n](x: tensor[n, i64]) -> i64 = shape(insert(x, 0, h), 0)\n\
                 out = f({operand})\n"
            ),
            "out = 5".to_string(),
        ),
        (
            "renamed_binder",
            format!(
                "n = 5i64\n\
                 def f[d](x: tensor[d, i64]) -> i64 = shape(insert(x, 0, n), 0)\n\
                 out = f({operand})\n"
            ),
            "out = 5".to_string(),
        ),
        (
            "renamed_runtime_global",
            format!(
                "def g(k: i64) -> i64 = add(k, 1i64)\n\
                 h = g(4i64)\n\
                 def f[n](x: tensor[n, i64]) = insert(x, 0, h)\n\
                 out = f({operand})\n"
            ),
            "tensor(shape=[5, 2], data=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2])".to_string(),
        ),
        (
            "renamed_computed_local",
            format!(
                "def f[n](x: tensor[n, i64]) -> i64 = {{\n  m = add(shape(x, 0), 1i64)\n  \
                 tensor_to_scalar(sum(sum(insert(x, 0, m), 0), 0))\n}}\n\
                 out = f({operand})\n"
            ),
            "out = 9".to_string(),
        ),
    ] {
        assert_lanes_agree(stem, &source, &expected);
    }
}

/// A size name that is neither a value nor a dimension has no extent to read.
/// Lowering keeps its fatal error for it on both lanes rather than reading the
/// name as an unbound runtime input.
#[test]
fn a_size_name_that_is_neither_a_value_nor_a_dimension_is_a_lowering_error_on_both_lanes() {
    let dir = tempdir().expect("tempdir");
    let stem = "unbound";
    let path = write_fixture(
        &dir,
        stem,
        "def f(x: tensor[2, i64]) = insert(x, 0, zz)\nout = f(to_tensor([1i64, 2i64]))\n",
    );
    let needle = "`insert` size resolves to `zz`, but no in-scope tensor axis supplies that extent";
    for (lane, output) in [
        ("eval", eval(&path)),
        ("c", build_and_run(&dir, stem, &path)),
    ] {
        let text = combined(&output);
        assert!(!output.status.success(), "{lane} must reject: {text}");
        assert!(text.contains(needle), "{lane}: {text}");
    }
}

/// A size name inside an inlined callee reads the callee's own scope. The
/// callee's value parameter `k` used to lose to the caller's binder `k`, which
/// stays visible while the callee is inlined: eval printed the parameter's
/// extent and C the caller's 2. The checker cannot see this, because inside
/// `g` the name `k` has one meaning.
#[test]
fn a_callee_value_spelled_like_a_caller_binder_sizes_by_the_value_on_both_lanes() {
    for (stem, source, expected) in [
        (
            "leak_runtime_four",
            format!(
                "def g(t: tensor[2, i64], k: i64) = insert(t, 0, k)\n\
                 def f[k](x: tensor[k, i64]) = g(to_tensor([1i64, 2i64]), {RUNTIME_FOUR})\n\
                 out = f(to_tensor([1i64, 2i64]))\n"
            ),
            "tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "leak_shape_arith",
            "def g(t: tensor[2, i64], k: i64) = insert(t, 0, k)\n\
             def f[k](x: tensor[k, i64]) = g(to_tensor([1i64, 2i64]), add(shape(x, 0), 1i64))\n\
             out = f(to_tensor([1i64, 2i64]))\n"
                .to_string(),
            THREE_BY_TWO,
        ),
        (
            "leak_expand",
            "def g(t: tensor[1, i64], k: i64) = expand(t, 0, k)\n\
             def f[k](x: tensor[k, i64]) = g(to_tensor([7i64]), add(shape(x, 0), 1i64))\n\
             out = f(to_tensor([1i64, 2i64]))\n"
                .to_string(),
            "tensor(shape=[3], data=[7, 7, 7])",
        ),
    ] {
        assert_lanes_agree(stem, &source, expected);
    }
}

/// Controls for the row above, and the adversarial direction: a callee's own
/// dimension binder `n` keeps reading its dimension when the caller has a value
/// `n`, and a caller's value still flows into a callee's dimension through the
/// argument's shape.
#[test]
fn callee_dimensions_and_renamed_values_keep_their_meaning_on_both_lanes() {
    for (stem, source, expected) in [
        (
            "renamed_callee_value",
            "def g(t: tensor[2, i64], q: i64) = insert(t, 0, q)\n\
             def f[k](x: tensor[k, i64]) = g(to_tensor([1i64, 2i64]), add(shape(x, 0), 1i64))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            THREE_BY_TWO,
        ),
        (
            "caller_value_into_callee_dimension",
            "def g[n](x: tensor[n, i64]) -> i64 = shape(insert(x, 0, n), 0)\n\
             def f(n: i64) -> i64 = g(expand(to_tensor([1i64]), 0, n))\n\
             out = f(3i64)\n",
            "out = 3",
        ),
        (
            "callee_value_from_caller_value",
            "def g(t: tensor[2, i64], k: i64) -> i64 = tensor_to_scalar(sum(sum(insert(t, 0, k), 0), 0))\n\
             def f[k](x: tensor[k, i64], j: i64) -> i64 = g(to_tensor([1i64, 2i64]), j)\n\
             out = f(to_tensor([1i64, 2i64]), 5i64)\n",
            "out = 15",
        ),
        (
            "callee_value_beside_caller_binder",
            "def g(t: tensor[2, i64], k: i64) = insert(t, 0, k)\n\
             def f[k](x: tensor[k, i64], j: i64) = g(to_tensor([1i64, 2i64]), j)\n\
             out = f(to_tensor([1i64, 2i64, 3i64, 4i64]), 3i64)\n",
            THREE_BY_TWO,
        ),
        (
            "callee_dimension_beside_caller_value",
            "def g[n](x: tensor[n, i64]) -> i64 = shape(insert(x, 0, n), 0)\n\
             def f(n: i64, y: tensor[2, i64]) -> i64 = g(y)\n\
             out = f(5i64, to_tensor([1i64, 2i64]))\n",
            "out = 2",
        ),
    ] {
        assert_lanes_agree(stem, source, expected);
    }
}

/// The size-path regression corpus. Each program names a size whose reading
/// a caller could capture: through inlining three deep, a function literal,
/// `map`, `vmap` and `grad`, a `cast` or arithmetic, a callee's own dimension
/// beside a caller's value, and a top-level value read by a callee whose caller
/// has a dimension of the same spelling. Lowering reads a size as a dimension
/// only where the checker stamped it as one, in the size's own definition, so
/// every row prints the same bytes on both lanes. The top-level rows used to
/// print the caller's extent in C (`w6b`), or trap on eval while C printed the
/// caller's extent (`w6`, `w6c`).
#[test]
fn size_names_read_in_their_own_definition_on_both_lanes() {
    for (stem, source, expected) in [
        (
            "size_dim_control",
            "def f[n](x: tensor[n, i64]) = insert(x, 0, n)\n\
             out = f(to_tensor([1i64, 2i64, 3i64]))\n",
            "out = tensor(shape=[3, 3], data=[1, 2, 3, 1, 2, 3, 1, 2, 3])",
        ),
        (
            "size_rename_control",
            "def f[n](x: tensor[n, i64], m: i64) -> i64 = {\n\
               k = m\n\
               shape(insert(x, 0, k), 0)\n\
             }\n\
             out = f(to_tensor([1i64, 2i64]), 3i64)\n",
            "out = 3",
        ),
        (
            "size_vmap",
            "def g(b: tensor[3, 2, i64], k: i64) = vmap(fn (r: tensor[2, i64]) -> insert(r, 0, k))(b)\n\
             def f[k](x: tensor[k, i64]) = g(to_tensor([[1i64, 2i64], [3i64, 4i64], [5i64, 6i64]]), tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = tensor(shape=[3, 4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4, 3, 4, 5, 6, 5, 6, 5, 6, 5, 6])",
        ),
        (
            "size_static_caller_leak",
            "def g[n](x: tensor[n, i64]) -> i64 = shape(insert(x, 0, n), 0)\n\
             def f() -> i64 = {\n\
               n = 5i64\n\
               g(to_tensor([1i64, 2i64]))\n\
             }\n\
             out = f()\n",
            "out = 2",
        ),
        (
            "size_runtime_caller_leak",
            "def g[n](x: tensor[n, i64]) = insert(x, 0, n)\n\
             def f(n: i64) = g(to_tensor([1i64, 2i64]))\n\
             out = f(tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n",
            "out = tensor(shape=[2, 2], data=[1, 2, 1, 2])",
        ),
        (
            "size_caller_param_leak_shape",
            "def g[n](x: tensor[n, i64]) -> i64 = shape(insert(x, 0, n), 0)\n\
             def f(n: i64) -> i64 = add(g(to_tensor([1i64, 2i64])), mul(n, 0i64))\n\
             out = f(tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n",
            "out = 2",
        ),
        (
            "size_three_deep",
            "def h(t: tensor[2, i64], k: i64) = insert(t, 0, k)\n\
             def g(t: tensor[2, i64], k: i64) = h(t, k)\n\
             def f[k](x: tensor[k, i64]) = g(to_tensor([1i64, 2i64]), tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "size_three_deep_middle_dim",
            "def h(t: tensor[2, i64], k: i64) = insert(t, 0, k)\n\
             def g[k](y: tensor[k, i64], j: i64) = h(to_tensor([1i64, 2i64]), j)\n\
             def f(x: tensor[3, i64]) = g(to_tensor([9i64]), tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n\
             out = f(to_tensor([1i64, 2i64, 3i64]))\n",
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "size_three_deep_expand",
            "def h(t: tensor[1, i64], k: i64) = expand(t, 0, k)\n\
             def g(t: tensor[1, i64], k: i64) = h(t, k)\n\
             def f[k](x: tensor[k, i64]) = g(to_tensor([7i64]), tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = tensor(shape=[4], data=[7, 7, 7, 7])",
        ),
        (
            "size_lambda_capture",
            "def g(k: i64) -> i64 = {\n\
               h = fn (t: tensor[2, i64]) -> shape(insert(t, 0, k), 0)\n\
               h(to_tensor([1i64, 2i64]))\n\
             }\n\
             def f[k](x: tensor[k, i64]) -> i64 = g(tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = 4",
        ),
        (
            "size_lambda_capture_tensor",
            "def g(k: i64) = {\n\
               h = fn (t: tensor[2, i64]) -> insert(t, 0, k)\n\
               h(to_tensor([1i64, 2i64]))\n\
             }\n\
             def f[k](x: tensor[k, i64]) = g(tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "size_lambda_map",
            "def g(k: i64) -> List[i64] = map(fn (i: i64) -> shape(insert(to_tensor([1i64, 2i64]), 0, k), 0), [1i64, 2i64])\n\
             def f[k](x: tensor[k, i64]) -> List[i64] = g(tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = [4, 4]",
        ),
        (
            "size_toplevel_after",
            "def g(t: tensor[2, i64]) = insert(t, 0, kk)\n\
             def f[kk](x: tensor[kk, i64]) = g(to_tensor([1i64, 2i64]))\n\
             kk = tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "size_toplevel_before",
            "kk = tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0))\n\
             def g(t: tensor[2, i64]) = insert(t, 0, kk)\n\
             def f[kk](x: tensor[kk, i64]) = g(to_tensor([1i64, 2i64]))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "size_toplevel_static_after",
            "def g(t: tensor[2, i64]) = insert(t, 0, kk)\n\
             def f[kk](x: tensor[kk, i64]) = g(to_tensor([1i64, 2i64]))\n\
             kk = 4i64\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "size_toplevel_no_caller_dim",
            "kk = tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0))\n\
             def g(t: tensor[2, i64]) = insert(t, 0, kk)\n\
             def f(x: tensor[3, i64]) = g(to_tensor([1i64, 2i64]))\n\
             out = f(to_tensor([1i64, 2i64, 3i64]))\n",
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "size_cast",
            "def g(t: tensor[2, i64], k: i32) = insert(t, 0, cast(k, i64))\n\
             def f[k](x: tensor[k, i64]) = g(to_tensor([1i64, 2i64]), cast(tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)), i32))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "size_arith",
            "def g(t: tensor[2, i64], k: i64) = insert(t, 0, add(k, 0i64))\n\
             def f[k](x: tensor[k, i64]) = g(to_tensor([1i64, 2i64]), tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n\
             out = f(to_tensor([1i64, 2i64]))\n",
            "out = tensor(shape=[4, 2], data=[1, 2, 1, 2, 1, 2, 1, 2])",
        ),
        (
            "size_grad",
            "def loss(x: tensor[2, f32], k: i64) -> f32 = tensor_to_scalar(sum(sum(insert(x, 0, k), 0), 0))\n\
             def loss4(x: tensor[2, f32]) -> f32 = loss(x, tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0)))\n\
             def f[k](y: tensor[k, f32]) -> tensor[2, f32] = grad(loss4)(to_tensor([1.0, 2.0]))\n\
             out = f(to_tensor([1.0, 2.0]))\n",
            "out = tensor(shape=[2], data=[4.0, 4.0])",
        ),
        (
            "size_grad_k",
            "def lossk(x: tensor[2, f32]) -> f32 = {\n\
               k = tensor_to_scalar(sum(to_tensor([2i64, 2i64]), 0))\n\
               tensor_to_scalar(sum(sum(insert(x, 0, k), 0), 0))\n\
             }\n\
             def f[k](y: tensor[k, f32]) -> tensor[2, f32] = grad(lossk)(to_tensor([1.0, 2.0]))\n\
             out = f(to_tensor([1.0, 2.0]))\n",
            "out = tensor(shape=[2], data=[4.0, 4.0])",
        ),
    ] {
        assert_lanes_agree(stem, source, expected);
    }
}

/// A pattern binder shadows an outer static value of the same name. Each row
/// binds `n` (or folds over it) under an outer `n = 3i64`, and the size reads
/// the pattern's 5. The checker used to keep the outer binding's folded 3
/// under the pattern's name: eval printed 5 while C guarded the axis against
/// 3 and trapped. The controls are a pattern with no outer `n`, a parameter
/// `n`, and a `fold` lambda parameter `n`; the claim row declares 3 and traps
/// identically on both lanes.
#[test]
fn a_pattern_binder_shadows_an_outer_static_size_on_both_lanes() {
    let some = "out = f(to_tensor([1i64, 2i64]), Some(5i64))\n";
    for (stem, source, expected) in [
        (
            "match_shadow_static",
            format!(
                "n = 3i64\n\
                 def f(x: tensor[2, i64], o: Option[i64]) -> i64 = match o with {{\n\
                   | Some(n) => shape(insert(x, 0, n), 0)\n\
                   | None => 0i64\n\
                 }}\n{some}"
            ),
            "out = 5",
        ),
        (
            "match_shadow_local",
            format!(
                "def f(x: tensor[2, i64], o: Option[i64]) -> i64 = {{\n\
                   n = 3i64\n\
                   match o with {{\n\
                     | Some(n) => shape(insert(x, 0, n), 0)\n\
                     | None => 0i64\n\
                   }}\n\
                 }}\n{some}"
            ),
            "out = 5",
        ),
        (
            "match_shadow_sum",
            format!(
                "n = 3i64\n\
                 def f(x: tensor[2, i64], o: Option[i64]) -> i64 = match o with {{\n\
                   | Some(n) => tensor_to_scalar(sum(sum(insert(x, 0, n), 0), 0))\n\
                   | None => 0i64\n\
                 }}\n{some}"
            ),
            "out = 15",
        ),
        (
            "match_shadow_tuple",
            "n = 3i64\n\
             def f(x: tensor[2, i64], p: (i64, i64)) -> i64 = match p with {\n\
               | (m, n) => add(m, shape(insert(x, 0, n), 0))\n\
             }\n\
             out = f(to_tensor([1i64, 2i64]), (1i64, 5i64))\n"
                .to_string(),
            "out = 6",
        ),
        (
            "match_shadow_arith",
            format!(
                "n = 3i64\n\
                 def f(x: tensor[2, i64], o: Option[i64]) -> i64 = match o with {{\n\
                   | Some(n) => shape(insert(x, 0, add(n, 0i64)), 0)\n\
                   | None => 0i64\n\
                 }}\n{some}"
            ),
            "out = 5",
        ),
        (
            "match_no_outer",
            format!(
                "def f(x: tensor[2, i64], o: Option[i64]) -> i64 = match o with {{\n\
                   | Some(n) => shape(insert(x, 0, n), 0)\n\
                   | None => 0i64\n\
                 }}\n{some}"
            ),
            "out = 5",
        ),
        (
            "match_shadow_param",
            "def f(x: tensor[2, i64], n: i64, o: Option[i64]) -> i64 = match o with {\n\
               | Some(n) => shape(insert(x, 0, n), 0)\n\
               | None => 0i64\n\
             }\n\
             out = f(to_tensor([1i64, 2i64]), 3i64, Some(5i64))\n"
                .to_string(),
            "out = 5",
        ),
        (
            "fold_param_shadow",
            "n = 3i64\n\
             def f(x: tensor[2, i64]) -> i64 = fold(fn (acc: i64, n: i64) -> add(acc, shape(insert(x, 0, n), 0)), 0i64, [1i64, 5i64])\n\
             out = f(to_tensor([1i64, 2i64]))\n"
                .to_string(),
            "out = 6",
        ),
    ] {
        assert_lanes_agree(stem, &source, expected);
    }
    assert_lanes_trap_identically(
        "match_shadow_claim",
        "n = 3i64\n\
         def f(x: tensor[2, i64], o: Option[i64]) -> tensor[3, 2, i64] = match o with {\n\
           | Some(n) => insert(x, 0, n)\n\
           | None => insert(x, 0, 3i64)\n\
         }\n\
         out = f(to_tensor([1i64, 2i64]), Some(5i64))\n",
        &[
            "claimed = 3, insert axis 0 = 5",
            &domain_trap_line("insert"),
        ],
    );
}

/// A callee's own dimension `n` keeps its meaning when a top-level value `n` is
/// declared after the callee, which [04-INF-4] makes invisible inside it. Host
/// inlining used to qualify the size `n` as a read of that later value before
/// lowering saw the checker's stamp: C then sized the axis by the value and
/// trapped, a shape mismatch for the static value and a claim of 2 against 5
/// for the runtime one, while eval printed 2.
#[test]
fn a_later_top_level_value_never_replaces_a_callee_dimension_on_both_lanes() {
    let callee = "def g[n](x: tensor[n, i64]) -> i64 = shape(insert(x, 0, n), 0)\n";
    for (stem, value) in [
        ("later_static_toplevel", "5i64"),
        (
            "later_runtime_toplevel",
            "tensor_to_scalar(sum(to_tensor([2i64, 3i64]), 0))",
        ),
    ] {
        assert_lanes_agree(
            stem,
            &format!("{callee}n = {value}\nout = g(to_tensor([1i64, 2i64]))\n"),
            "out = 2",
        );
    }
}
