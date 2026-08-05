// Regression for the borrow-checker fix gating downstream School P1.5
// (BatchNorm) and P2.5 (`_step_tree`) work. Pre-fix:
//
//   InvalidBorrow: borrowed arguments must be tensor or tensor-carrying values
//
// was raised for `&p` at the call site when `p: SomeAdt[..]` was a
// record-style ADT whose fields contained `tensor[...]`. The linearity
// checker's `type_expr_contains_tensor` only inspected the ADT's type
// arguments (`t-adt {} Name <args>...`), so an ADT whose tensor was
// stored as a variant FIELD (not in the type-arg position) was
// treated as non-tensor-carrying and rejected from the borrow path.
//
// Fix in `crates/chelis-types/src/linearity.rs`: `Checker` now
// pre-computes a `tensor_carrying_adts: HashSet<String>` of every ADT
// whose definition (transitively) carries a tensor field. The
// `t-adt` arm of `type_expr_contains_tensor` consults this set so
// `&BatchNormParams { weight: tensor[..], ... }` is accepted as a
// borrow. Tensorless ADTs (`Counter { value: int64 }`) are still
// rejected with the same `InvalidBorrow` error.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

/// Run `chelis check <path>` and return the parsed JSON stdout.
/// Parsing the output (rather than substring-matching it) keeps these
/// regression tests insensitive to whitespace/key-order changes in
/// `chelis check`'s machine-facing output. Does not assert on the
/// process exit code: issue #207 made `chelis check` exit non-zero
/// when the JSON `errors` array is non-empty, but this helper is
/// shared between fixtures that expect a clean check and fixtures
/// that expect specific errors in the report.
fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn error_kinds(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect()
}

fn fmt_inplace(path: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn borrow_tensor_carrying_record_adt_is_accepted() {
    // The downstream School P1.5 blocker shape: `BatchNormParams` is a
    // record with two `tensor[n, f32]` fields. A function that takes
    // `&BatchNormParams[n]` should be callable on a `BatchNormParams[n]`
    // value via `borrow_params(&p)`. Pre-fix this errored at the call
    // site with `InvalidBorrow`.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("batchnorm_borrow.ch");
    write_file(
        &fixture,
        "module BatchNormShape\n\
         type BatchNormParams[n] =\n\
           | BatchNormParams { gamma: tensor[n, f32], beta: tensor[n, f32] }\n\
         sig borrow_params: &BatchNormParams[n] -> bool\n\
         def borrow_params(p) = true\n\
         def consume_params[n](p: BatchNormParams[n]) -> bool = borrow_params(&p)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "borrowing a tensor-carrying record ADT must produce no errors; got {kinds:?}"
    );
}

#[test]
fn borrow_tensorless_adt_is_still_rejected() {
    // Regression-guard: the original `InvalidBorrow` rule still fires
    // for ADTs whose fields are all non-tensor. The fix is precisely
    // scoped to ADTs carrying a tensor (transitively).
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("counter_borrow.ch");
    write_file(
        &fixture,
        "module TensorlessShape\n\
         type Counter =\n\
           | Counter { value: int64 }\n\
         sig borrow_counter: &Counter -> bool\n\
         def borrow_counter(c) = true\n\
         def consume_counter(c: Counter) -> bool = borrow_counter(&c)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "InvalidBorrow"),
        "tensorless ADT borrow must still produce InvalidBorrow; got {kinds:?}"
    );
}

#[test]
fn borrow_parametric_adt_at_tensor_arg_is_accepted() {
    // The OR's second branch: `Wrapper[a]` whose definition is just
    // `Wrapper { value: a }` is NOT a tensor-carrying ADT by name (its
    // field type is the abstract type variable `a`, not `tensor`). At
    // the call site, instantiation pins `a = tensor[..]`, so the
    // `t-adt {} Wrapper (t-tensor ...)` form *should* still be borrow-
    // eligible via the type-argument walk that the fix preserves with
    // an `OR`. This test guards that branch — if someone simplifies
    // the `t-adt` arm to "only consult `tensor_carrying_adts`", this
    // case starts failing.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("wrapper_borrow.ch");
    write_file(
        &fixture,
        "module WrapperShape\n\
         type Wrapper[a] =\n\
           | Wrapper { value: a }\n\
         sig borrow_wrapper: &Wrapper[tensor[4, f32]] -> bool\n\
         def borrow_wrapper(w) = true\n\
         def consume_wrapper(w: Wrapper[tensor[4, f32]]) -> bool = borrow_wrapper(&w)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        kinds.is_empty(),
        "parametric ADT at tensor arg must still borrow cleanly; got {kinds:?}"
    );
}

#[test]
fn destructured_generic_adt_field_borrow_is_accepted() {
    // Issue #181 (residual gap after #154): the linearity-side fix in
    // PR #153 handled `&adt_value` for tensor-carrying ADTs, but
    // `copy(&field)` against a *destructured* tensor field of a
    // generic ADT still failed because the destructured binding's
    // type stayed as a fresh type variable instead of being unified
    // with the ADT's concrete instantiation at the match site. With
    // the substitution fix, `match state with | FooState { x, y } =>
    // copy(&x)` resolves `x` to `tensor[n, f32]` from the scrutinee's
    // `FooState[tensor[n, f32]]` and the borrow checker accepts the
    // `&x` borrow.
    //
    // Downstream impact (per the issue body): unblocks the natural
    // shape for stateful optimizer `_step_tree` variants (RMSProp,
    // Lion, Adam) and Train-mode forward passes on stateful layers
    // (BatchNorm), removing the `to_list → to_tensor` workaround
    // adopted in School PR #36.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("destructured_field_borrow.ch");
    write_file(
        &fixture,
        "module DestructuredShape\n\
         type FooState[a] =\n\
           | FooState { x: a, y: a }\n\
         def use_foo[n](state: FooState[tensor[n, f32]]) -> tensor[n, f32] = match state with {\n\
             | FooState { x, y } => {\n\
               x_copy = copy(&x)\n\
               _ = y\n\
               x_copy\n\
             }\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "borrowing a destructured tensor field of a generic ADT must produce no errors; got {kinds:?}"
    );
}

#[test]
fn destructured_generic_adt_field_borrow_with_nested_adt() {
    // Transitive variant of #181: the field type is itself an ADT
    // (Inner[a]) parameterized by the outer ADT's type parameter.
    // After the substitution fix, the nested ADT's type argument also
    // pins to the scrutinee's instantiation, so `&inner` borrows
    // cleanly against the linearity carrier set.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("nested_destructured_field_borrow.ch");
    write_file(
        &fixture,
        "module NestedDestructuredShape\n\
         type Inner[a] =\n\
           | Inner { value: a }\n\
         type Outer[a] =\n\
           | Outer { inner: Inner[a] }\n\
         sig borrow_inner: &Inner[tensor[n, f32]] -> bool\n\
         def borrow_inner(i) = true\n\
         def use_outer[n](o: Outer[tensor[n, f32]]) -> bool = match o with {\n\
           | Outer { inner } => borrow_inner(&inner)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert_eq!(json["score"], 1, "perfect-score contract: {json}");
    assert!(
        kinds.is_empty(),
        "borrowing a destructured nested ADT field must produce no errors; got {kinds:?}"
    );
}

#[test]
fn destructured_generic_adt_nontensor_field_borrow_is_rejected() {
    // Negative parity for `destructured_generic_adt_field_borrow_is_accepted`
    // (CLAUDE.md "Negative Test Parity"): the same `match | FooState { x, y } => copy(&x)`
    // shape must STILL be rejected when the scrutinee's ADT instantiation pins
    // `a` to a non-tensor primitive. Without this guard, a future refactor that
    // broadens the destructured-field type lookup (e.g., defaulting to tensor
    // when ADT resolution loses the instantiation) could silently start
    // accepting borrows of `int64` fields.
    //
    // Expected behavior: type-check produces a TypeMismatch (or InvalidBorrow)
    // mentioning `int64` since `&int64` is not a valid borrow target.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("destructured_nontensor_borrow.ch");
    write_file(
        &fixture,
        "module DestructuredNontensorShape\n\
         type FooState[a] =\n\
           | FooState { x: a, y: a }\n\
         def use_foo(state: FooState[int64]) -> int64 = match state with {\n\
             | FooState { x, y } => {\n\
               _ = copy(&x)\n\
               y\n\
             }\n\
         }\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        kinds
            .iter()
            .any(|k| k == "TypeMismatch" || k == "InvalidBorrow"),
        "borrowing a destructured non-tensor field must be rejected; got {kinds:?}"
    );
    // The diagnostic must surface the offending type so the user can act.
    let messages: Vec<String> = json["errors"]
        .as_array()
        .expect("errors")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        messages.iter().any(|m| m.contains("int64")),
        "rejection diagnostic should mention `int64`; got {messages:?}"
    );
}

#[test]
fn borrow_nested_tensor_carrying_adt_is_accepted() {
    // Transitive tensor-carrying: ADT A wraps an ADT B that contains
    // a tensor field. Both A and B end up in `tensor_carrying_adts`
    // via the fixed-point pass.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("nested_borrow.ch");
    write_file(
        &fixture,
        "module NestedShape\n\
         type Inner[n] =\n\
           | Inner { values: tensor[n, f32] }\n\
         type Outer[n] =\n\
           | Outer { inner: Inner[n] }\n\
         sig borrow_outer: &Outer[n] -> bool\n\
         def borrow_outer(o) = true\n\
         def consume_outer[n](o: Outer[n]) -> bool = borrow_outer(&o)\n",
    );
    fmt_inplace(&fixture);

    let json = run_check(&fixture);
    let kinds = error_kinds(&json);
    assert!(
        kinds.is_empty(),
        "transitive tensor-carrying ADT must borrow cleanly; got {kinds:?}"
    );
}
