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
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn check_file(path: &Path) -> assert_cmd::assert::Assert {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
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
         def consume_params[n](p: BatchNormParams[n]) -> bool = {\n\
           borrow_params(&p)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    check_file(&fixture)
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"))
        .stdout(predicate::str::contains("InvalidBorrow").not());
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
         def consume_counter(c: Counter) -> bool = {\n\
           borrow_counter(&c)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    check_file(&fixture).stdout(predicate::str::contains("InvalidBorrow"));
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
         def consume_wrapper(w: Wrapper[tensor[4, f32]]) -> bool = {\n\
           borrow_wrapper(&w)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    check_file(&fixture)
        .stdout(predicate::str::contains("\"errors\": []"))
        .stdout(predicate::str::contains("InvalidBorrow").not());
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
         def consume_outer[n](o: Outer[n]) -> bool = {\n\
           borrow_outer(&o)\n\
         }\n",
    );
    fmt_inplace(&fixture);

    check_file(&fixture)
        .stdout(predicate::str::contains("\"errors\": []"))
        .stdout(predicate::str::contains("InvalidBorrow").not());
}
