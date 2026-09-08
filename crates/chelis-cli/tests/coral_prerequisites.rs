use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

#[test]
#[ignore = "manual gate: Coral prerequisite acceptance bundle exceeds the default inner-loop budget"]
fn coral_prerequisites() {
    let (_dir, reef_home, app_pkg) = make_app("coral-prereqs");

    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

type Column =
  | FloatCol(tensor[4, f32])
  | IntCol(tensor[4, int64])

def get_float(c: Column) -> tensor[4, f32] = match c with {
  | FloatCol(t) => t
  | IntCol(_) => (to_tensor([0.0, 0.0, 0.0, 0.0]) : tensor[4, f32])
}

value = get_float(FloatCol((to_tensor([1.0, 2.0, 3.0, 4.0]) : tensor[4, f32])))
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"));

    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

def make_dict() -> Dict[string, tensor[4, f32]] = {
  d = dict_of([] : List[(string, tensor[4, f32])])
  dict_insert(d, "price", (to_tensor([1.0, 1.0, 1.0, 1.0]) : tensor[4, f32]))
}

value = make_dict()
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"));

    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

def filter_bools(mask: tensor[4, bool], indices: tensor[2, int64]) -> tensor[2, bool] =
  gather(mask, indices, 0)
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            app_pkg.join("out-gather-bool").to_str().unwrap(),
        ])
        .assert()
        .success();

    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Construct (arange)

ar_0_4 = arange(cast(0, int32), cast(4, int32))
ar_2_6 = arange(cast(2, int32), cast(6, int32))
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "ar_0_4 = tensor(shape=[4], data=[0.0, 1.0, 2.0, 3.0])",
        ))
        .stdout(predicate::str::contains(
            "ar_2_6 = tensor(shape=[4], data=[2.0, 3.0, 4.0, 5.0])",
        ));
}

#[test]
#[ignore = "manual gate: Coral prerequisite and regression acceptance suite exceeds the default inner-loop budget"]
fn coral_where_indices_builds_and_matches_reference_values() {
    let (_dir, reef_home, app_pkg) = make_app("coral-where-indices");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Mask (where_indices)

mixed_mask = cmplt((to_tensor([0.0, 1.0, 0.0, 1.0, 1.0]) : tensor[5, f32]), (to_tensor([0.5, 0.5, 0.5, 0.5, 0.5]) : tensor[5, f32]))
all_true = cmplt((to_tensor([0.0, 0.0, 0.0]) : tensor[3, f32]), (to_tensor([1.0, 1.0, 1.0]) : tensor[3, f32]))

mixed_idx = where_indices(mixed_mask)
all_true_idx = where_indices(all_true)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "mixed_idx = tensor(shape=[2], data=[0.0, 2.0])",
        ))
        .stdout(predicate::str::contains(
            "all_true_idx = tensor(shape=[3], data=[0.0, 1.0, 2.0])",
        ));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            app_pkg.join("out").to_str().unwrap(),
        ])
        .assert()
        .success();
}

#[test]
#[ignore = "manual gate: Coral prerequisite and regression acceptance suite exceeds the default inner-loop budget"]
fn coral_where_indices_all_false_returns_empty_tensor() {
    // Previously this panicked the evaluator because `numel` floored zero-
    // length tensors to 1, tripping the length assertion in `from_vec`.
    // After the 3t cleanup fix, the evaluator honors the zero dimension and
    // the empty mask path produces a legitimate `tensor[0, int64]`.
    let (_dir, reef_home, app_pkg) = make_app("coral-where-indices-empty");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Tensor.Mask (where_indices)

all_false = cmplt((to_tensor([1.0, 1.0, 1.0]) : tensor[3, f32]), (to_tensor([0.0, 0.0, 0.0]) : tensor[3, f32]))
value = where_indices(all_false)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("shape=[0]"))
        .stdout(predicate::str::contains("data=[]"));
}

/// Coral upstream blocker (v0.2.5/v0.3.0): `to_tensor` previously rejected
/// `List[bool]` at type-check time with "to_tensor expects numeric List
/// elements, got bool". After this fix, bool lists produce `tensor[N, bool]`
/// at both `chelis check` and `chelis eval` time, which unblocks
/// `bool_list_to_tensor` and CSV/JSON bool round trips.
#[test]
#[ignore = "manual gate: Coral prerequisite and regression acceptance suite exceeds the default inner-loop budget"]
fn coral_to_tensor_accepts_bool_list() {
    let (_dir, reef_home, app_pkg) = make_app("coral-to-tensor-bool");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

mask = to_tensor([true, false, true])
roundtrip = to_list(mask)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        // Bool tensors are stored as 0.0/1.0 in the runtime f64 buffer.
        .stdout(predicate::str::contains(
            "mask = tensor(shape=[3], data=[1.0, 0.0, 1.0])",
        ))
        // Round-tripping through to_list recovers the original bool values.
        .stdout(predicate::str::contains("roundtrip = [true, false, true]"));
}

/// Negative parity for `to_tensor`: mixing bool and float in a single list
/// must still be rejected (the runtime requires homogeneous element kinds).
#[test]
#[ignore = "manual gate: Coral prerequisite and regression acceptance suite exceeds the default inner-loop budget"]
fn coral_to_tensor_rejects_mixed_bool_and_float_list() {
    let (_dir, reef_home, app_pkg) = make_app("coral-to-tensor-bool-mixed");
    // Force a heterogeneous List[bool|float] at the runtime by going through
    // `if`. The type checker rejects this at the surface, so we use a
    // surface that type-checks first and exercises the runtime guard via
    // `--expr` against an unchecked path.
    //
    // The simplest negative is the type-check level: a literal list with a
    // bool and a float must fail unification at the List[α] element type.
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

bad = to_tensor([true, 1.0])
"#,
    );
    // Issue #207: type errors now produce exit 2; assert on stdout
    // content only (the dedicated invariant test covers the exit
    // code).
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        // Type checker still flags the heterogeneous list.
        .stdout(
            predicate::str::contains("\"errors\":")
                .and(predicate::str::contains("\"score\": 1").not()),
        );
}

/// Coral upstream blocker (v0.2.5/v0.3.0): tensor-tensor comparison ops
/// (`eq`/`neq`/`lt`/`gt`) landed in v0.2.5 but tensor-scalar broadcast was
/// not extended at the same time, so `gt(tensor, scalar)` was rejected at
/// type-check time with `type mismatch: tensor[D, f32] vs f32`. This fix
/// extends the broadcast: when one arg is a tensor `T[D, p]` and the other
/// is a matching-precision scalar `Prim(p)`, the comparison broadcasts the
/// scalar element-wise and returns `tensor[D, bool]`. The same broadcast
/// applies to `eq`/`neq`/`lt`/`gt`/`lte`/`gte`/`cmplt` and is symmetric
/// (`gt(scalar, tensor)` works too).
#[test]
#[ignore = "manual gate: Coral prerequisite and regression acceptance suite exceeds the default inner-loop budget"]
fn coral_comparison_ops_broadcast_tensor_scalar() {
    let (_dir, reef_home, app_pkg) = make_app("coral-cmp-broadcast");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

def mk_xs() -> tensor[3, f32] = (to_tensor([1.0, 2.0, 3.0]) : tensor[3, f32])
def mk_ints() -> tensor[3, int64] = (to_tensor([cast(1, int64), cast(2, int64), cast(3, int64)]) : tensor[3, int64])
def mk_bools() -> tensor[3, bool] = to_tensor([true, false, true])

xs_print = mk_xs()
gt_mask = gt(mk_xs(), 1.5)
lt_mask = lt(mk_xs(), 2.5)
eq_mask = eq(mk_xs(), 2.0)
neq_mask = neq(mk_xs(), 2.0)
gte_mask = gte(mk_xs(), 2.0)
lte_mask = lte(mk_xs(), 2.0)
cmplt_mask = cmplt(mk_xs(), 2.0)
gt_left = gt(1.5, mk_xs())
gt_ints = gt(mk_ints(), cast(1, int64))
eq_bools = eq(mk_bools(), true)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success()
        // gt(xs, 1.5): [1.0>1.5, 2.0>1.5, 3.0>1.5] = [F, T, T]
        .stdout(predicate::str::contains(
            "gt_mask = tensor(shape=[3], data=[0.0, 1.0, 1.0])",
        ))
        // lt(xs, 2.5): [F, T, F]... wait: [1.0<2.5, 2.0<2.5, 3.0<2.5] = [T,T,F]
        .stdout(predicate::str::contains(
            "lt_mask = tensor(shape=[3], data=[1.0, 1.0, 0.0])",
        ))
        // eq(xs, 2.0): [F, T, F]
        .stdout(predicate::str::contains(
            "eq_mask = tensor(shape=[3], data=[0.0, 1.0, 0.0])",
        ))
        // neq(xs, 2.0): [T, F, T]
        .stdout(predicate::str::contains(
            "neq_mask = tensor(shape=[3], data=[1.0, 0.0, 1.0])",
        ))
        // gte(xs, 2.0): [F, T, T]
        .stdout(predicate::str::contains(
            "gte_mask = tensor(shape=[3], data=[0.0, 1.0, 1.0])",
        ))
        // lte(xs, 2.0): [T, T, F]
        .stdout(predicate::str::contains(
            "lte_mask = tensor(shape=[3], data=[1.0, 1.0, 0.0])",
        ))
        // cmplt(xs, 2.0): [T, F, F]
        .stdout(predicate::str::contains(
            "cmplt_mask = tensor(shape=[3], data=[1.0, 0.0, 0.0])",
        ))
        // gt(1.5, xs): [1.5>1.0, 1.5>2.0, 1.5>3.0] = [T, F, F]
        .stdout(predicate::str::contains(
            "gt_left = tensor(shape=[3], data=[1.0, 0.0, 0.0])",
        ))
        // gt(ints, 1): [F, T, T]
        .stdout(predicate::str::contains(
            "gt_ints = tensor(shape=[3], data=[0.0, 1.0, 1.0])",
        ))
        // eq(bools, true): bool tensor [1,0,1] eq true = [T, F, T]
        .stdout(predicate::str::contains(
            "eq_bools = tensor(shape=[3], data=[1.0, 0.0, 1.0])",
        ));
}

/// Negative parity for the comparison-op broadcast: mismatched precision
/// (e.g. `tensor[3, f32]` vs `int64` scalar) must still be rejected, so a
/// silent precision coercion can't sneak in. Ordered comparisons on bool
/// tensors must also still error (bool ordering has no defined meaning;
/// only `eq`/`neq` accept bool).
#[test]
#[ignore = "manual gate: Coral prerequisite and regression acceptance suite exceeds the default inner-loop budget"]
fn coral_comparison_ops_reject_mismatched_precision() {
    let (_dir, reef_home, app_pkg) = make_app("coral-cmp-mismatch");

    // f32 tensor vs int64 scalar must fail.
    // Issue #207: type errors now produce exit 2; assert on stdout
    // content only (the dedicated invariant test covers the exit code).
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

xs = (to_tensor([1.0, 2.0, 3.0]) : tensor[3, f32])
bad = gt(xs, cast(1, int64))
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .stdout(
            predicate::str::contains("\"errors\":")
                .and(predicate::str::contains("\"score\": 1").not()),
        );

    // Ordered comparison on bool tensor must fail (bool isn't ordered).
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

bools = to_tensor([true, false, true])
bad = gt(bools, true)
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .stdout(
            predicate::str::contains("\"errors\":")
                .and(predicate::str::contains("\"score\": 1").not()),
        );
}

/// Issue #5 regression: when a comparison op is used in `gt(scalar, tensor)`
/// form inside a `def name() -> tensor[D, bool] = ...` body, type inference
/// must produce `tensor[D, bool]` (not `bool`) so the declared signature
/// matches. v0.3.1's broadcast rewrite handled the unification, but the
/// comparison-op return-type override at the post-unify site only inspected
/// `arg_tys[0]`, so a leading scalar arg made the override return scalar
/// `Prim(Bool)` and discard the tensor shape. The user-facing symptom was
/// that a module containing both `gt(tensor, scalar)` and `gt(scalar, tensor)`
/// with declared `tensor[N, bool]` signatures rejected the second form. The
/// `gt(scalar, tensor)` form actually fails alone too with a declared
/// tensor signature; the existing v0.2.5 broadcast test just hides this
/// because it uses an untyped top-level binding (`gt_left = ...`) whose
/// inferred type is never checked against any signature.
/// chelis#1506 INVERTS this gate. The six programs above are the form
/// `[05-OP-36]` calls a type error, so the compiler now refuses them and names
/// the replacement. Coral has source that depends on the old acceptance; the
/// migration is the explicit spelling, and the second half of this test proves
/// the migration type checks so the pin bump has somewhere to go.
///
/// EVIDENTIARY STATUS: REGRESSION for the rejection half, which was a clean
/// `"score": 1` with `"errors": []` on the base `9b9e7bd56`; DISPOSITION LOCK
/// for the migration half, which checked clean before and after.
#[test]
#[ignore = "manual gate: Coral prerequisite and regression acceptance suite exceeds the default inner-loop budget"]
fn coral_comparison_ops_reject_a_scalar_operand_and_name_the_replacement() {
    let (_dir, reef_home, app_pkg) = make_app("coral-cmp-broadcast-issue5");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

def above() -> tensor[3, bool] = gt(to_tensor([1.0, 2.0, 3.0]), 1.5)
def below() -> tensor[3, bool] = gt(1.5, to_tensor([1.0, 2.0, 3.0]))
def lt_right() -> tensor[3, bool] = lt(to_tensor([1.0, 2.0, 3.0]), 2.5)
def lt_left() -> tensor[3, bool] = lt(2.5, to_tensor([1.0, 2.0, 3.0]))
def eq_right() -> tensor[3, bool] = eq(to_tensor([1.0, 2.0, 3.0]), 2.0)
def eq_left() -> tensor[3, bool] = eq(2.0, to_tensor([1.0, 2.0, 3.0]))
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains("[05-OP-36]"))
        .stdout(predicate::str::contains(
            "does not admit a scalar beside a tensor",
        ));

    // The migration Coral takes, in the same package, so the pin bump can
    // point at a spelling this compiler accepts.
    let (_migrated_dir, migrated_home, migrated_pkg) = make_app("coral-cmp-explicit-issue5");
    write_file(
        &migrated_pkg.join("src/main.ch"),
        r#"module Demo.Main

def above() -> tensor[3, bool] = gt(to_tensor([1.0, 2.0, 3.0]), expand(to_tensor([1.5]), 0i32, 3i64))
def below() -> tensor[3, bool] = gt(expand(to_tensor([1.5]), 0i32, 3i64), to_tensor([1.0, 2.0, 3.0]))
def lt_right() -> tensor[3, bool] = lt(to_tensor([1.0, 2.0, 3.0]), expand(to_tensor([2.5]), 0i32, 3i64))
def lt_left() -> tensor[3, bool] = lt(expand(to_tensor([2.5]), 0i32, 3i64), to_tensor([1.0, 2.0, 3.0]))
def eq_right() -> tensor[3, bool] = eq(to_tensor([1.0, 2.0, 3.0]), expand(to_tensor([2.0]), 0i32, 3i64))
def eq_left() -> tensor[3, bool] = eq(expand(to_tensor([2.0]), 0i32, 3i64), to_tensor([1.0, 2.0, 3.0]))
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &migrated_home)
        .current_dir(&migrated_pkg)
        .args(["check", migrated_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"))
        .stdout(predicate::str::contains("\"errors\": []"));
}
