//! chelis#1541 — a tensor whose dtype is a declared type parameter must be
//! able to flow through an `if`/`then`/`else` without aborting the process.
//!
//! Before the fix `chelis check` exited 101 with a developer-facing `BUG:`
//! message instead of returning a verdict. The abort came from a **routing
//! predicate**, not from emission: `if_expr_is_dag_lowerable` asks whether an
//! `if` can be lowered to the float DAG, and it read the node's precision
//! through a total reader that panicked on an unresolved `(t-var {} p)`
//! precision slot. A `bool`-returning pre-check has an answer for that input —
//! an unresolved dtype cannot be *established* as a float DAG join, so the
//! predicate answers `false` — and both of its callers then do something
//! sound: the host-runtime classifier routes to the general lane, and
//! `assert_ir_lowerable` raises a real lowering diagnostic.
//!
//! Spec authority: `spec/04-type-system.md` §5.8.1 scopes the
//! `TensorPrec::Concrete(_)` assertion to "**after** monomorphization", "at
//! lowering time", and a "`TensorPrec::Var(_)` **reaching a backend**". A
//! predicate deciding *whether* to use the DAG backend satisfies none of those
//! three, so the invariant never applied at that call site — the
//! implementation over-applied a correctly-scoped rule. §5.9 `[04-DTYPE-2]`
//! supplies the dtype-family bound (`[p: Float]`) that makes a generic numeric
//! def writable at all, which is the form real library code uses.
//!
//! Per the project's backend-numerics discipline these tests do not stop at
//! "it checks": the dtype-generic definition is compiled and run at two
//! different dtypes and the printed tensors are compared against the values
//! computed by hand. `pick(flag, x, y)` returns `2 * (flag ? x : y)`, so at
//! f32 with `flag = true` the answer is `2 * [1, 2] = [2, 4]`, and at f64 with
//! `flag = false` it is `2 * [10, 20] = [20, 40]`. Checking only that the
//! panic stopped would pass on a fix that silently dropped an arm.

use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;
use common::{
    authored_c_symbol, build_and_run, gcc_available, host_body_definition, parse_tensor_data,
    write_file,
};

/// The reproducer exactly as chelis#1541 reports it: an unbounded binder `p`
/// in the precision slot, joined through `if`/`then`/`else`.
const ISSUE_REPRODUCER: &str = "module DtypeGenericJoin
def go[n, p](x: tensor[n, p], y: tensor[n, p]) -> i32 = {
  w = if true then x else y
  1i32
}
";

/// The form real library code uses: the §5.9 dtype-family bound that makes
/// `cast(<literal>, p)` legal in the body (chelis#1558 rejects an unbounded
/// binder as a cast target, and prints declaring a bound as the remedy).
const BOUNDED_GENERIC_JOIN: &str = "module BoundedGenericJoin
export (pick)
def pick[n, p: Float](flag: bool, x: tensor[n, p], y: tensor[n, p]) -> tensor[n, p] = {
  w = if flag then x else y
  add(w, w)
}
";

/// The same generic `pick`, reached from two concrete call sites at different
/// dtypes. One definition, two monomorphizations.
const TWO_DTYPE_PROGRAM: &str = "module Issue1541TwoDtypes
export (pick, doubled_then_f32, doubled_else_f64)
def pick[n, p: Float](flag: bool, x: tensor[n, p], y: tensor[n, p]) -> tensor[n, p] = {
  w = if flag then x else y
  add(w, w)
}
def doubled_then_f32() -> tensor[2, f32] = {
  a = to_tensor([cast(1.0, f32), cast(2.0, f32)])
  b = to_tensor([cast(10.0, f32), cast(20.0, f32)])
  pick(true, a, b)
}
def doubled_else_f64() -> tensor[2, f64] = {
  a = to_tensor([cast(1.0, f64), cast(2.0, f64)])
  b = to_tensor([cast(10.0, f64), cast(20.0, f64)])
  pick(false, a, b)
}
";

/// The same shape at the `Int` family. `[p: Float]` was the only bound the
/// original change covered and §5.9 names three; `[p: Int]` aborted pre-fix for
/// the same reason and is repaired by the same line.
const INT_BOUND_PROGRAM: &str = "module Issue1541IntBound
export (pick_i, doubled_then_i32)
def pick_i[n, p: Int](flag: bool, x: tensor[n, p], y: tensor[n, p]) -> tensor[n, p] = {
  w = if flag then x else y
  add(w, w)
}
def doubled_then_i32() -> tensor[2, i32] = {
  a = to_tensor([cast(1, i32), cast(2, i32)])
  b = to_tensor([cast(10, i32), cast(20, i32)])
  pick_i(true, a, b)
}
";

/// Fully concrete joins at a float and a non-float dtype. Neither involves a
/// precision binder, so neither may change behaviour: the non-float one
/// already took the predicate's `false` branch that the fix now shares.
const CONCRETE_JOINS: &str = "module Issue1541Concrete
export (int_join_result, float_join_result)
def int_join(x: tensor[2, i32], y: tensor[2, i32]) -> tensor[2, i32] = {
  w = if true then x else y
  add(w, w)
}
def float_join(x: tensor[2, f32], y: tensor[2, f32]) -> tensor[2, f32] = {
  w = if false then x else y
  add(w, w)
}
def int_join_result() -> tensor[2, i32] = int_join(to_tensor([cast(1, i32), cast(2, i32)]), to_tensor([cast(3, i32), cast(4, i32)]))
def float_join_result() -> tensor[2, f32] = float_join(to_tensor([cast(1.0, f32), cast(2.0, f32)]), to_tensor([cast(3.0, f32), cast(4.0, f32)]))
";

fn check(dir: &Path, name: &str, source: &str) -> (Value, std::process::Output) {
    let path = dir.join(format!("{name}.ch"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("spawn chelis");
    let json = serde_json::from_slice(&output.stdout).unwrap_or(Value::Null);
    (json, output)
}

fn assert_no_abort(output: &std::process::Output, context: &str) {
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked") && !stderr.contains("BUG:"),
        "{context}: chelis aborted instead of returning a verdict; stderr:\n{stderr}"
    );
    assert_ne!(
        output.status.code(),
        Some(101),
        "{context}: exit 101 is the Rust panic code; stderr:\n{stderr}"
    );
}

fn assert_checks_clean(json: &Value, output: &std::process::Output, context: &str) {
    assert_no_abort(output, context);
    assert_eq!(
        json["score"], 1.0,
        "{context}: expected a clean verdict, got {json}"
    );
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    assert!(
        errors.is_empty(),
        "{context}: expected no errors, got {errors:?}"
    );
    assert!(
        output.status.success(),
        "{context}: expected exit 0, got {:?}",
        output.status.code()
    );
}

/// chelis#1541 positive: the reported reproducer returns a verdict.
#[test]
fn the_issue_reproducer_checks_clean_instead_of_aborting() {
    let dir = tempdir().expect("tempdir");
    let (json, output) = check(dir.path(), "issue_1541_repro", ISSUE_REPRODUCER);
    assert_checks_clean(&json, &output, "chelis#1541 reproducer");
}

/// chelis#1541 positive: the §5.9-bounded form real library code uses.
#[test]
fn a_bounded_dtype_generic_if_join_checks_clean() {
    let dir = tempdir().expect("tempdir");
    let (json, output) = check(dir.path(), "issue_1541_bounded", BOUNDED_GENERIC_JOIN);
    assert_checks_clean(&json, &output, "bounded dtype-generic if join");
}

/// chelis#1541 negative parity: concrete joins are untouched. The non-float
/// arm matters most — it already reached the predicate's `false` branch that
/// the fix now also returns for an unresolved precision, so if the new guard
/// over-triggered or the shared branch regressed, this is where it shows.
#[test]
fn concrete_if_joins_are_unaffected() {
    let dir = tempdir().expect("tempdir");
    let (json, output) = check(dir.path(), "issue_1541_concrete", CONCRETE_JOINS);
    assert_checks_clean(&json, &output, "concrete if joins");
}

/// chelis#1541 end-to-end: one dtype-generic definition, two dtypes, values
/// checked against hand-computed answers so a dropped arm cannot pass.
#[test]
fn a_dtype_generic_if_join_computes_correctly_at_f32_and_f64() {
    if !gcc_available() {
        eprintln!("skipping: no C compiler available");
        return;
    }
    let stdout = build_and_run(TWO_DTYPE_PROGRAM, "issue_1541_two_dtypes");

    let then_f32 = parse_tensor_data(&stdout, "doubled_then_f32");
    assert_eq!(
        then_f32,
        vec![2.0, 4.0],
        "f32 instantiation took the wrong arm or lost the doubling; stdout:\n{stdout}"
    );

    let else_f64 = parse_tensor_data(&stdout, "doubled_else_f64");
    assert_eq!(
        else_f64,
        vec![20.0, 40.0],
        "f64 instantiation took the wrong arm or lost the doubling; stdout:\n{stdout}"
    );
}

/// chelis#1541 negative parity: the fix must not trade the abort for silently
/// emitting a kernel for the polymorphic definition itself. A generic def is
/// monomorphized into each concrete caller and never emitted standalone, so
/// the generated C defines the two concrete entry points and no `pick`.
#[test]
fn the_polymorphic_definition_is_not_emitted_standalone() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("issue_1541_emission.ch");
    let out_dir = dir.path().join("issue_1541_emission-out");
    write_file(&path, TWO_DTYPE_PROGRAM);

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("spawn chelis");
    assert_no_abort(&output, "emitting a dtype-generic def");

    let emitted =
        std::fs::read_to_string(out_dir.join("issue_1541_emission.c")).expect("emitted C source");
    // The generic callee's public compiler-owned symbol must not appear in the
    // emitted translation unit. Its concrete monomorphizations use distinct
    // internal names derived from the same authored stem.
    let generic_symbol = authored_c_symbol("pick");
    assert!(
        !emitted.lines().any(|line| {
            let line = line.trim_start();
            line.contains(&format!("{generic_symbol}("))
                || line.contains(&format!("{generic_symbol}__chelis_owned_body("))
        }),
        "the polymorphic callee `pick` reached codegen; it must be monomorphized \
         into each concrete caller, not emitted standalone"
    );

    // And the concrete entries must be present as C *definitions*, not merely
    // as substrings of a printf format string.
    for entry in ["doubled_then_f32", "doubled_else_f64"] {
        let body = format!("{}__chelis_owned_body", authored_c_symbol(entry));
        let definition = host_body_definition(&emitted, &body);
        assert!(
            definition.contains("chelis_tensor"),
            "emitted C has no `chelis_tensor`-returning definition for `{entry}`"
        );
    }
}

/// chelis#1541 positive, §5.9 `Int` family: the repair is not float-specific.
#[test]
fn a_bounded_int_generic_if_join_computes_correctly() {
    let dir = tempdir().expect("tempdir");
    let (json, output) = check(dir.path(), "issue_1541_int_bound", INT_BOUND_PROGRAM);
    assert_checks_clean(&json, &output, "Int-bounded dtype-generic if join");

    if !gcc_available() {
        eprintln!("skipping execution: no C compiler available");
        return;
    }
    let stdout = build_and_run(INT_BOUND_PROGRAM, "issue_1541_int_bound_run");
    let got = parse_tensor_data(&stdout, "doubled_then_i32");
    assert_eq!(
        got,
        vec![2.0, 4.0],
        "i32 instantiation took the wrong arm or lost the doubling; stdout:\n{stdout}"
    );
}
