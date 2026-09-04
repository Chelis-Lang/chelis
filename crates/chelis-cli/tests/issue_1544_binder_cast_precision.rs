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

/// Regression test. The C lane must materialize the literal at the
/// instantiated dtype.
///
/// Asserted on the emitted constant's exact bit pattern rather than on the
/// absence of a decimal substring: `0x3fb999999999999a` IS f64 `0.1`, and the
/// f32 rounding this issue is about would emit `0x3fb99999a0000000`. An
/// absence assertion would pass on any reformatting of the wrong value.
///
/// Verified by compiling and running the emitted C by hand:
/// `at_f64 = 0.30000000000000004`, which is what `chelis eval` prints for the
/// same program. The two lanes agree; the row above pins the eval half.
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
        emitted.contains("0x3fb999999999999a"),
        "the emitted C must carry f64 0.1 exactly, not an f32 rounding of it"
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

/// Round 1 F1, regression test. An unsuffixed FLOAT literal cast to an
/// UNBOUNDED binder is rejected on both lanes.
///
/// `spec/04-type-system.md` [04-DTYPE-2] makes an unbounded binder an
/// unconstrained type variable admitting every type, not a dtype, so §P10b
/// position 4's "the literals bind at `p`" has no meaning here: there is no
/// family to bind at. The literal therefore took the §P10 f32 default, and
/// casting that to an f64 instantiation is not the same value as binding at
/// f64. At the reviewed head `f124ce7f8` the program returned
/// `0.30000000447034836` from `chelis eval` while `chelis build` rejected the
/// same source.
///
/// This is a PARTIAL enforcement of [04-DTYPE-1], which constrains the cast
/// TARGET and not the source; the general enforcement is chelis#1558 and the
/// readings are recorded in chelis#1553.
///
/// This row has now been written three times, which is worth knowing before
/// changing it a fourth. It first asserted both-lane rejection, right against
/// base's measurement. It was restated to record a lane split, right once
/// three interpreter tests showed that premise false. Both were wrong against
/// the numbered spec, which settles the question the measurements could not.
#[test]
fn an_unbounded_binder_float_literal_cast_is_rejected_on_both_lanes() {
    let (_dir, root) = make_package(
        "unbounded-binder",
        "module Bind.Main\n\
         export (main)\n\
         def scale[p](x: p) -> p = mul(x, cast(0.1, p))\n\
         def main() -> f64 = scale(3.0f64)\n",
    );
    let check = run(&root, &["check", "src/main.ch"]);
    assert!(
        check.status.success(),
        "the program type-checks; the rejection is a lowering one until chelis#1558"
    );
    for command in [
        vec!["eval", "--file", "src/main.ch"],
        vec!["build", "src/main.ch", "--target", "c", "--output", "out"],
    ] {
        let label = command[0];
        let output = run(&root, &command);
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !output.status.success(),
            "{label} must reject it; stdout={stdout} stderr={stderr}"
        );
        assert!(
            !stdout.contains(ROUNDED_VIA_F32),
            "{label} must not return the f32-rounded value: {stdout}"
        );
    }
}

/// The positive control that pins the narrowness, so it is deliberate rather
/// than incidental. A VARIABLE source under an unbounded binder still
/// actualizes at the call site, exactly as it does today: there is no literal
/// taking a default, so there is no wrong value to prevent, and
/// `chelis-compiler-api runtime::tests::generic_cast_target_uses_*` pin that
/// capability directly. An INTEGER literal is left alone too, for the reason
/// the eval-lane comment gives: its default is exact at every integer width
/// that fits, and out of range it traps loudly under [04-NUM-14] rather than
/// returning a wrong value.
///
/// Disposition lock: green before this change and after it.
#[test]
fn an_unbounded_binder_keeps_variable_and_integer_sources() {
    let (_dir, root) = make_package(
        "unbounded-binder-controls",
        "module Bind.Main\n\
         export (main)\n\
         def recast[p](value: p) -> p = cast(value, p)\n\
         def addk[p](x: p) -> p = add(x, cast(1, p))\n\
         def at_i32() -> int32 = recast(41i32)\n\
         def at_i8() -> int8 = addk(recast(41i8))\n\
         def main() -> int32 = at_i32()\n",
    );
    let stdout = check_then_eval(&root);
    assert!(
        stdout.contains("at_i32 = 41") && stdout.contains("at_i8 = 42"),
        "a variable source and an integer literal must keep working: {stdout}"
    );
}

/// Round 1 F2, regression test./// Round 1 F2, regression test. An INTEGER literal adopting an `Int`-bounded
/// binder checks clean and computes at the instantiated integer width.
///
/// This is `arange_values` in `packages/chelis-std/src/tensor/construct.ch`,
/// verbatim in shape. At the reviewed head it failed `chelis check` with
/// `literal_source: integer requires a primitive float type`, a
/// compiler-internal diagnostic carrying no span and no action a Chelis author
/// could take, because the adopted literal was stamped with a marker that
/// records "an integer written where a float is wanted" and that the checker
/// requires to sit on a primitive float.
#[test]
fn an_integer_literal_adopts_an_int_bounded_binder() {
    let (_dir, root) = make_package(
        "int-binder",
        "module Bind.Main\n\
         export (main)\n\
         def addk[p: Int](x: p) -> p = add(x, cast(1, p))\n\
         def at_i32() -> int32 = addk(41i32)\n\
         def at_i64() -> int64 = addk(41i64)\n\
         def main() -> int64 = at_i64()\n",
    );
    let stdout = check_then_eval(&root);
    assert!(
        stdout.contains("at_i32 = 42") && stdout.contains("at_i64 = 42"),
        "an Int-bounded binder must compute at both integer widths: {stdout}"
    );
}

/// The float twin of the row above, so the repair cannot be "integers never
/// adopt". An integer literal under a `Float`-bounded binder adopts it and
/// computes at the instantiated float width.
///
/// Disposition lock at f32, regression at f64: the reviewed head failed both at
/// `chelis check` for the same marker reason.
#[test]
fn an_integer_literal_adopts_a_float_bounded_binder() {
    let (_dir, root) = make_package(
        "int-under-float-binder",
        "module Bind.Main\n\
         export (main)\n\
         def addk[p: Float](x: p) -> p = add(x, cast(7, p))\n\
         def at_f32() -> f32 = addk(1.0f32)\n\
         def at_f64() -> f64 = addk(1.0f64)\n\
         def main() -> f64 = at_f64()\n",
    );
    let stdout = check_then_eval(&root);
    assert!(
        stdout.contains("at_f32 = 8") && stdout.contains("at_f64 = 8"),
        "a Float-bounded binder must adopt an integer literal at both widths: {stdout}"
    );
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
