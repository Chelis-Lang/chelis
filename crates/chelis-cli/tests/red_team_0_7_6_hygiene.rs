//! Red-team probes for the 0.7.6 toolchain hygiene workstream.
//!
//! Pins surface bugs discovered during the adversarial pass documented at
//! `docs/investigations/0_7_6_hygiene_redteam.md`. Tests marked
//! `#[ignore]` are the failing-on-shipped-behavior pins; flipping the
//! `#[ignore]` off should track the fix landing on `main`.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

/// Finding 1: `chelis eval --file` rejects `jit(...)` even though `check`
/// accepts it and the lowering pass treats jit as identity. Spec
/// `spec/06-transformations.md` §4 says `jit(f)(x) = f(x)` for all `x`.
#[test]
#[ignore = "0.7.6 hygiene red-team Finding 1: host runtime rejects `jit` despite PR #40"]
fn eval_jit_passthrough_matches_no_jit() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("jit_passthrough.ch");
    write_file(
        &path,
        "def compute(x: tensor[3, f32]) -> tensor[3, f32] = jit(mul(x, to_tensor([2.0, 2.0, 2.0])))\n\
         a = to_tensor([1.5, 2.7, -0.3])\n\
         result = compute(a)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("3.0"))
        .stdout(predicate::str::contains("5.4"))
        .stdout(predicate::str::contains("-0.6"));
}

/// Finding 2 (sub-case): `par` with tensor children is rejected by the
/// host runtime. Spec §2.3 / §5 say par is sequential (last expression
/// wins) — the type of the children shouldn't matter.
#[test]
#[ignore = "0.7.6 hygiene red-team Finding 2: host runtime rejects `par` with tensor children"]
fn eval_par_tensor_children_returns_last() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("par_tensor.ch");
    write_file(
        &path,
        "result = par { to_tensor([1.0, 1.0, 1.0]); to_tensor([2.0, 2.0, 2.0]); to_tensor([3.0, 3.0, 3.0]) }\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("3.0"));
}

/// Finding 2 (sub-case): `par` inside an f32-returning fn returns 0.0
/// instead of the last child's value. The compute() body lowers via the
/// IR path but `lower_par`'s skip-2 iteration produces no last value in
/// that lowering context.
#[test]
#[ignore = "0.7.6 hygiene red-team Finding 2: par-in-fn returns 0.0 instead of last"]
fn eval_par_in_fn_returns_last_not_zero() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("par_in_fn.ch");
    write_file(
        &path,
        "def compute() -> f32 = par { 1.0; 2.0; 3.0 }\nresult = compute()\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        // par returns last value (3.0), NOT the 0.0 fallback constant.
        .stdout(predicate::str::contains("3.0"))
        .stdout(predicate::str::contains("0.0").not());
}

/// Finding 3: `chelis fmt` is non-idempotent for 3+ stage pipes. The
/// formatter output of pass 1 (multi-line braced) is rejected by
/// `fmt --check` and pass 2 produces a different form.
#[test]
#[ignore = "0.7.6 hygiene red-team Finding 3: fmt non-idempotent for 3-stage pipes"]
fn fmt_three_stage_pipe_is_idempotent() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("three_pipe.ch");
    write_file(
        &path,
        "def f(x: tensor[3, f32]) -> tensor[3, f32] = neg(x)\n\
         def composed(x: tensor[3, f32]) -> tensor[3, f32] = x |> f |> f |> f\n",
    );

    // First fmt --check after authoring should already pass if the
    // formatter considers a 3-stage one-line pipe canonical, OR fmt
    // --inplace should converge to a canonical form in one pass.
    let out1 = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let pass1_path = dir.path().join("pass1.ch");
    write_file(&pass1_path, std::str::from_utf8(&out1).unwrap());

    let out2 = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", pass1_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    assert_eq!(
        out1, out2,
        "fmt must be idempotent: pass1 != pass2 violates spec §6 canonical form"
    );
}

/// Finding 3 (downstream): `lint --fix` produces output that fails
/// `fmt --check`.
#[test]
#[ignore = "0.7.6 hygiene red-team Finding 3: prefer-pipe-operator autofix breaks style gate"]
fn lint_fix_prefer_pipe_yields_canonical_output() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("autofix.ch");
    write_file(
        &path,
        "def f(x: tensor[3, f32]) -> tensor[3, f32] = neg(x)\n\
         def composed(x: tensor[3, f32]) -> tensor[3, f32] = f(f(f(x)))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["lint", "--fix", path.to_str().unwrap()])
        .assert()
        .success();

    // After lint --fix the file must still satisfy `fmt --check`.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--check", path.to_str().unwrap()])
        .assert()
        .success();
}

/// Finding 4: `x |> copy` fails type check even when the pipe input is
/// statically tensor-typed. The brief documents `copy` as a supported
/// bare-keyword pipe stage on par with `realize`.
#[test]
#[ignore = "0.7.6 hygiene red-team Finding 4: x |> copy fails type check"]
fn eval_bare_copy_pipe_stage_succeeds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("copy_pipe.ch");
    write_file(
        &path,
        "def main(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy\n\
         a = to_tensor([1.5, 2.7, -0.3])\n\
         result = main(a)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("1.5"))
        .stdout(predicate::str::contains("2.7"))
        .stdout(predicate::str::contains("-0.3"));
}

/// Finding 5: `chelis eval --file` on a Surf input containing only `def`
/// declarations exits 0 with no stdout and no stderr. The G7 fix exists
/// on `fix/cli-eval-warn-on-empty-roots` but was not merged.
#[test]
#[ignore = "0.7.6 hygiene red-team Finding 5: silent no-output on def-only programs (G7 unmerged)"]
fn eval_def_only_emits_warning_on_stderr() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("def_only.ch");
    write_file(
        &path,
        "def helper(x: f32) -> f32 = mul(x, 2.0)\n\
         def other(x: f32) -> f32 = add(x, 1.0)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains("nothing to evaluate"));
}

// Negative-result pins: confirmed-working surfaces. These run by default
// so regressions surface immediately.

#[test]
fn eval_var_rhs_alias_fanout_matches_xtx() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("alias.ch");
    write_file(
        &path,
        "def f(x: tensor[3, f32]) -> tensor[3, f32] = {\n  \
            alias = x\n  \
            mul(x, alias)\n\
         }\n\
         a = to_tensor([1.5, 2.7, -0.3])\n\
         result = f(a)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        // x*x for [1.5, 2.7, -0.3] = [2.25, 7.29, 0.09]
        .stdout(predicate::str::contains("2.25"))
        .stdout(predicate::str::contains("7.29"))
        .stdout(predicate::str::contains("0.09"));
}

#[test]
fn eval_grad_pipe_stage_matches_2x() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_pipe.ch");
    write_file(
        &path,
        "def loss(x: tensor[3, f32]) -> tensor[f32] = sum(mul(x, x), 0)\n\
         def gradient(x: tensor[3, f32]) -> tensor[3, f32] = x |> grad(loss)\n\
         a = to_tensor([1.5, 2.7, -0.3])\n\
         g = gradient(a)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        // d/dx sum(x*x) = 2x = [3.0, 5.4, -0.6]
        .stdout(predicate::str::contains("3.0"))
        .stdout(predicate::str::contains("5.4"))
        .stdout(predicate::str::contains("-0.6"));
}

#[test]
fn eval_realize_pipe_stage_is_identity() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("realize_pipe.ch");
    write_file(
        &path,
        "def main(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize\n\
         a = to_tensor([1.5, 2.7, -0.3])\n\
         result = main(a)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("1.5"))
        .stdout(predicate::str::contains("2.7"))
        .stdout(predicate::str::contains("-0.3"));
}

#[test]
fn eval_nested_fn_typed_param_application_is_two_levels() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("inlining_nested.ch");
    write_file(
        &path,
        "def doubler(x: tensor[3, f32]) -> tensor[3, f32] = mul(x, to_tensor([2.0, 2.0, 2.0]))\n\
         def outer(f: tensor[3, f32] -> tensor[3, f32], x: tensor[3, f32]) -> tensor[3, f32] = f(f(x))\n\
         a = to_tensor([1.5, 2.7, -0.3])\n\
         result = outer(doubler, a)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        // doubler(doubler(x)) = 4x = [6.0, 10.8, -1.2]
        .stdout(predicate::str::contains("6.0"))
        .stdout(predicate::str::contains("10.8"))
        .stdout(predicate::str::contains("-1.2"));
}
