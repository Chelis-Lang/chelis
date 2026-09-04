//! chelis#1544: `cast(<literal>, p)` binds the literal AT the binder, so every
//! instantiation gets the authored value rather than an f32 rounding of it.
//!
//! `spec/02-surf-syntax.md` §P10 names an explicit `cast(literal, p)` as one of
//! exactly three overrides of the f32/int32 literal default, and §P10b position
//! 4 states the rule with the binder spelling: "the first argument of an
//! explicit `cast(literal, p)` expression - the literals bind at `p`". Today
//! the value is materialized at f32 and then widened, so an f64 instantiation
//! silently returns `3.0 * f32(0.1)`.
//!
//! Every fixture builds a FRESH package directory and runs `chelis check`
//! before `chelis eval`, because the whole-library lowering that carries this
//! defect is selected by the presence of a `reef.lock`. A probe that skips the
//! check evaluates cleanly and proves nothing.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Output;
use tempfile::{TempDir, tempdir};

/// `3.0 * 0.1` computed entirely in f64. The wrong answer this suite exists to
/// catch is `0.30000000447034836`, which is `3.0 * f32(0.1)` widened to f64.
const EXACT_F64: &str = "0.30000000000000004";
const ROUNDED_VIA_F32: &str = "0.30000000447034836";

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent directory");
    }
    fs::write(path, contents).expect("write fixture");
}

fn make_package(name: &str, source: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join(name);
    fs::create_dir_all(root.join("src")).expect("create src");
    write_file(
        &root.join("reef.toml"),
        &format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n\
             compiler = \"={ver}\"\nmodule_prefix = \"Bind\"\n",
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(&root.join("src/main.ch"), source);
    (dir, root)
}

fn run(root: &Path, args: &[&str]) -> Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(args)
        .output()
        .expect("run chelis")
}

/// Run `check` (which writes the `reef.lock` the failing path needs) and then
/// `eval`, returning eval's stdout.
fn check_then_eval(root: &Path) -> String {
    let check = run(root, &["check", "src/main.ch"]);
    assert!(
        check.status.success(),
        "check must succeed: stdout={} stderr={}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(
        root.join("reef.lock").exists(),
        "check must write the reef.lock this fixture depends on"
    );
    let eval = run(root, &["eval", "--file", "src/main.ch"]);
    assert!(
        eval.status.success(),
        "eval must succeed: stdout={} stderr={}",
        String::from_utf8_lossy(&eval.stdout),
        String::from_utf8_lossy(&eval.stderr)
    );
    String::from_utf8_lossy(&eval.stdout).to_string()
}

const SCALAR_SCALE: &str = "module Bind.Main\n\
     export (main)\n\
     def scale[p: Float](x: p) -> p = mul(x, cast(0.1, p))\n\
     def at_f32() -> f32 = scale(3.0f32)\n\
     def at_f64() -> f64 = scale(3.0f64)\n\
     def main() -> f64 = at_f64()\n";

const TENSOR_SCALE: &str = "module Bind.Main\n\
     export (main)\n\
     def scale[p: Float](x: tensor[1, p]) -> tensor[1, p] = mul(x, cast(0.1, p))\n\
     def at_f32() -> tensor[1, f32] = scale(to_tensor([3.0f32]))\n\
     def at_f64() -> tensor[1, f64] = scale(to_tensor([3.0f64]))\n\
     def main() -> tensor[1, f64] = at_f64()\n";

/// Regression test. Both rows fail today: `at_f64` prints
/// `0.30000000447034836`.
#[test]
fn a_scalar_binder_cast_binds_the_literal_at_every_instantiation() {
    let (_dir, root) = make_package("scalar-scale", SCALAR_SCALE);
    let stdout = check_then_eval(&root);
    assert!(
        stdout.contains("at_f32 = 0.3\n"),
        "the f32 instantiation must be exact: {stdout}"
    );
    assert!(
        stdout.contains(&format!("at_f64 = {EXACT_F64}")),
        "the f64 instantiation must compute at f64, not round through f32 \
         (the wrong answer is {ROUNDED_VIA_F32}): {stdout}"
    );
}

/// Regression test, and the row that shows chelis#1544 is not scalar-specific:
/// a TENSOR-polymorphic declaration loses the same precision today. Measured
/// on `ed5698835` with no PP7 change applied, so this is not a regression the
/// binder-spelling repair introduces.
#[test]
fn a_tensor_binder_cast_binds_the_literal_at_every_instantiation() {
    let (_dir, root) = make_package("tensor-scale", TENSOR_SCALE);
    let stdout = check_then_eval(&root);
    assert!(
        stdout.contains("at_f32 = tensor(shape=[1], data=[0.3])"),
        "the f32 instantiation must be exact: {stdout}"
    );
    assert!(
        stdout.contains(&format!("at_f64 = tensor(shape=[1], data=[{EXACT_F64}])")),
        "the f64 instantiation must compute at f64, not round through f32 \
         (the wrong answer is {ROUNDED_VIA_F32}): {stdout}"
    );
}

/// Regression test. The C lane must agree with `eval` on the same program;
/// pinning only `eval` would let the two lanes drift on the value that is the
/// whole subject of this issue.
#[test]
fn the_build_lane_agrees_with_eval_on_a_binder_cast() {
    let (_dir, root) = make_package("scalar-scale-build", SCALAR_SCALE);
    let check = run(&root, &["check", "src/main.ch"]);
    assert!(check.status.success(), "check must succeed");
    let build = run(
        &root,
        &["build", "src/main.ch", "--target", "c", "--output", "out"],
    );
    assert!(
        build.status.success(),
        "build must succeed: stdout={} stderr={}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted = fs::read_to_string(root.join("out/main.c")).expect("emitted C");
    assert!(
        !emitted.contains("0.30000000447034836"),
        "the emitted C must not carry the f32-rounded constant"
    );
}

/// Disposition lock: a cast target that is NOT a declared binder is still an
/// unknown primitive name and both lanes must keep rejecting it. Without this
/// the binder repair could be satisfied by treating every unknown name as a
/// binder.
///
/// Measured, not assumed: the rejection is the CHECK-time chelis#756
/// diagnostic, not the lowering's chelis#744 / [04-DTYPE-1] one. A typo never
/// reaches lowering from Surf, so [04-DTYPE-1] is reachable only from
/// hand-written Deep. Asserting the lowering citation here would have pinned a
/// diagnostic this program cannot produce.
///
/// Green in both states.
#[test]
fn an_undeclared_cast_target_is_still_rejected_on_both_lanes() {
    let (_dir, root) = make_package(
        "typo-target",
        "module Bind.Main\n\
         export (main)\n\
         def typo(x: f32) -> f32 = cast(x, flt32)\n\
         def main() -> f32 = typo(1.0f32)\n",
    );
    for command in [
        vec!["eval", "--file", "src/main.ch"],
        vec!["build", "src/main.ch", "--target", "c", "--output", "out"],
    ] {
        let label = command[0];
        let output = run(&root, &command);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success()
                && stderr.contains("cast target `flt32` is not a recognized primitive type"),
            "{label} must reject an unknown cast target: {stderr}"
        );
    }
}

/// Regression test. The ascription spelling reaches the same defect through a
/// different node: it desugars to `(lit {type: (t-var {} p)} 0.1)` rather than
/// a `cast`, scores 1.0 at `check`, runs correctly at f32, and at f64 dies at
/// RUNTIME with `numeric kernel mul expects matching dtypes, got f64 and f32`.
///
/// Expected disposition: it computes at f64 exactly. The spec does not settle
/// compute-versus-reject. §P10's override list is closed and does not name
/// ascription, while `spec/04-type-system.md` §5.6 sanctions the ascription
/// metadata the desugarer already emits, so computing is the reading that
/// invents no rule; rejecting at check time would need a §P10 or §5.6
/// amendment. What is NOT defensible either way is the current behavior, a
/// perfect check score followed by a runtime dtype error.
#[test]
fn an_ascribed_literal_binds_at_the_binder_too() {
    let (_dir, root) = make_package(
        "ascribed",
        "module Bind.Main\n\
         export (main)\n\
         def scale[p: Float](x: p) -> p = mul(x, (0.1 : p))\n\
         def main() -> f64 = scale(3.0f64)\n",
    );
    let stdout = check_then_eval(&root);
    assert!(
        stdout.contains(&format!("main = {EXACT_F64}")),
        "an ascribed literal must bind at the binder: {stdout}"
    );
}
