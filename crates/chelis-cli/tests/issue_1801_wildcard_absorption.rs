//! Inferred-signature rows for chelis#1801: a dimension variable an
//! application's instantiation minted, that unifies with a runtime extent
//! `*` and that no argument of the same application binds to a literal or
//! named dimension, denotes that runtime extent.
//!
//! `spec/04-type-system.md` section 3.2 Application states the rule. This
//! file is the checker-visible half: what each signature reads as. The
//! executable halves are `runtime_extent_slice_a.rs`'s
//! `a_root_that_keeps_a_dim_variable_is_still_dropped_on_both_lanes` and the
//! two `root.dim_variable.nested_helper` receipts in
//! `runtime_extent_slice_b.rs`.
//!
//! Every fixture here was measured on the base sha `ae9260727` before the
//! repair; each test's comment records that reading and says whether the
//! assertion is a regression test (the reading changed) or a disposition
//! lock (the reading must not change).

use assert_cmd::Command;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

/// The nested helper every fixture shares. `g` resolves its declared result
/// extent only at run time, so its checked result is `tensor[*, f32]`; `h`
/// claims one extent for its parameter and its result.
const NESTED_HELPERS: &str = "def g(x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
     def h(y: tensor[k, f32]) -> tensor[k, f32] = add(y, y)\n";

/// Check one source with `--show-inferred` and return `name -> display
/// signature`. Requires score 1.0 with no errors: every fixture in this file
/// is a well-typed program whose SIGNATURE is what is under test.
fn signatures(source: &str) -> BTreeMap<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("case.ch");
    fs::write(&path, source).expect("fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "check",
            "--show-inferred",
            "--allow-style-violations",
            path.to_str().unwrap(),
        ])
        .output()
        .expect("check");
    let report: Value = serde_json::from_slice(&output.stdout).expect("check json");
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    let errors = report["errors"].as_array().expect("errors array");
    assert!(errors.is_empty(), "{report}");
    report["inferred_signatures"]
        .as_array()
        .expect("inferred_signatures array")
        .iter()
        .map(|entry| {
            (
                entry["function"]
                    .as_str()
                    .expect("function name")
                    .to_string(),
                entry["display_signature"]
                    .as_str()
                    .expect("display signature")
                    .to_string(),
            )
        })
        .collect()
}

/// The full checker report, for the one row that asserts on more than the
/// signature map.
fn check(source: &str, path: &Path) -> Value {
    fs::write(path, source).expect("fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations", path.to_str().unwrap()])
        .output()
        .expect("check");
    serde_json::from_slice(&output.stdout).expect("check json")
}

fn eval_file(path: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            path.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("eval")
}

fn build_c(path: &Path, out_dir: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build")
}

/// Compile and run one emitted translation unit. A compile failure panics:
/// that is a defect in the emitter or the fixture, never the behaviour under
/// test.
fn compile_and_run_c(build_dir: &Path, stem: &str) -> std::process::Output {
    let binary = build_dir.join(format!("{stem}-bin"));
    let compile = StdCommand::new("cc")
        .args([
            "-std=c11",
            "-I",
            build_dir.to_str().unwrap(),
            build_dir.join(format!("{stem}.c")).to_str().unwrap(),
            build_dir.join("libchelis_runtime.a").to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-o",
            binary.to_str().unwrap(),
        ])
        .output()
        .expect("compile emitted C");
    assert!(
        compile.status.success(),
        "C compilation failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    StdCommand::new(binary).output().expect("run emitted C")
}

/// Regression test. On the base sha the issue's program checked as
/// `main :: () -> tensor[d0, f32]`: `h`'s instantiation variable met `g`'s
/// runtime extent, `unify_dim`'s permissive wildcard arm left it free, and
/// def-level generalization quantified it. The root then had no ABI and both
/// lanes dropped it (chelis#1801's user-visible half). The variable now
/// denotes the extent it met, so the root is concrete-enough to realize.
///
/// `g` and `h` keep their own signatures: the repair is at the application,
/// not at the generalization boundary, so a callee's declared polymorphism is
/// untouched.
#[test]
fn a_root_dim_variable_that_met_only_a_runtime_extent_denotes_it() {
    let signatures = signatures(&format!(
        "{NESTED_HELPERS}def main() = h(g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n"
    ));
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[*, f32]"),
        "{signatures:?}"
    );
    assert_eq!(
        signatures.get("g").map(String::as_str),
        Some("(tensor[d0, f32]) -> tensor[*, f32]"),
        "{signatures:?}"
    );
    assert_eq!(
        signatures.get("h").map(String::as_str),
        Some("(tensor[d0, f32]) -> tensor[d0, f32]"),
        "{signatures:?}"
    );
}

/// Regression test. The same absorption under a parameter rather than a
/// literal: on the base sha `p` read `(tensor[d0, f32]) -> tensor[d1, f32]`,
/// two unrelated variables where the result is in fact `g`'s runtime extent.
/// The parameter's own variable is NOT absorbed: nothing in this application
/// unified it with a wildcard.
#[test]
fn a_helper_result_absorbs_the_extent_without_absorbing_the_parameter() {
    let signatures = signatures(&format!("{NESTED_HELPERS}def p(x) = h(g(x))\n"));
    assert_eq!(
        signatures.get("p").map(String::as_str),
        Some("(tensor[d0, f32]) -> tensor[*, f32]"),
        "{signatures:?}"
    );
}

/// Disposition lock, and the control that rejects the repair inside
/// `unify_dim`. Binding the variable to `Dim::Wildcard` where it meets the
/// wildcard would freeze it, so a literal in a LATER argument of the same
/// call could no longer constrain it, and the reading would depend on
/// argument order. Absorbing after the whole call has unified makes the two
/// orders agree, and both read `tensor[2, f32]` here as they did on the base
/// sha.
#[test]
fn a_literal_argument_of_the_same_application_keeps_its_extent_in_either_order() {
    let signatures = signatures(&format!(
        "{NESTED_HELPERS}\
         def two(a: tensor[u, f32], b: tensor[u, f32]) -> tensor[u, f32] = add(a, b)\n\
         def wild_then_lit(x: tensor[m, f32]) = two(g(x), to_tensor([1.0f32, 2.0f32]))\n\
         def lit_then_wild(x: tensor[m, f32]) = two(to_tensor([1.0f32, 2.0f32]), g(x))\n"
    ));
    assert_eq!(
        signatures.get("wild_then_lit").map(String::as_str),
        Some("(tensor[d0, f32]) -> tensor[2, f32]"),
        "{signatures:?}"
    );
    assert_eq!(
        signatures.get("lit_then_wild").map(String::as_str),
        Some("(tensor[d0, f32]) -> tensor[2, f32]"),
        "{signatures:?}"
    );
}

/// Disposition lock. A named dimension another argument of the same
/// application supplies is a claim on the runtime extent, checked by a
/// section 4.7 guard, so the variable keeps that name rather than becoming
/// `*`. Read `(tensor[d0, f32], tensor[d0, f32]) -> tensor[d0, f32]` on the
/// base sha and unchanged here.
#[test]
fn a_named_dimension_another_argument_supplies_is_not_absorbed() {
    let signatures = signatures(&format!(
        "{NESTED_HELPERS}\
         def two(a: tensor[u, f32], b: tensor[u, f32]) -> tensor[u, f32] = add(a, b)\n\
         def keep(x: tensor[m, f32], y: tensor[m, f32]) = two(g(x), y)\n"
    ));
    assert_eq!(
        signatures.get("keep").map(String::as_str),
        Some("(tensor[d0, f32], tensor[d0, f32]) -> tensor[d0, f32]"),
        "{signatures:?}"
    );
}

/// Disposition lock. Absorption is scoped to the variables the application's
/// own instantiation minted, so the ENCLOSING definition's dimension variable
/// survives a wildcard it meets while unifying an unrelated call's arguments.
/// Without that scope `add`'s unification of `tensor[k]` against `h(g(y))`'s
/// `tensor[*]` would absorb `q`'s own `k` and publish
/// `(tensor[*], tensor[*]) -> tensor[*]`. Read `(tensor[d0, f32], tensor[d0,
/// f32]) -> tensor[d0, f32]` on the base sha and unchanged here.
#[test]
fn an_enclosing_definitions_own_dimension_variable_is_not_absorbed() {
    let signatures = signatures(&format!(
        "{NESTED_HELPERS}\
         def q(x: tensor[k, f32], y: tensor[k, f32]) -> tensor[k, f32] = add(x, h(g(y)))\n"
    ));
    assert_eq!(
        signatures.get("q").map(String::as_str),
        Some("(tensor[d0, f32], tensor[d0, f32]) -> tensor[d0, f32]"),
        "{signatures:?}"
    );
}

/// Disposition lock. A declared result still governs: the absorbed body type
/// `tensor[*, f32]` is admitted against `tensor[2, f32]` and the DECLARATION
/// is what the signature publishes, exactly as on the base sha. The extent
/// agreement itself is a section 4.7 runtime obligation, not a check-time
/// one; `runtime_extent_slice_b.rs` owns the refuted-claim receipt.
#[test]
fn a_declared_literal_result_still_governs_the_published_signature() {
    let signatures = signatures(&format!(
        "{NESTED_HELPERS}\
         def main() -> tensor[2, f32] = h(g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n"
    ));
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[2, f32]"),
        "{signatures:?}"
    );
}

/// Regression test for the block-bound spelling of the same root. On the base
/// sha this read `() -> tensor[d0, f32]` and was dropped like the direct
/// form: a block binding does not change which boundary quantified the
/// variable.
#[test]
fn a_block_bound_root_absorbs_the_extent_too() {
    let signatures = signatures(&format!(
        "{NESTED_HELPERS}\
         def main() = {{\n  \
         t = to_tensor([1.0f32, 2.0f32, 3.0f32])\n  \
         h(g(t))\n\
         }}\n"
    ));
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[*, f32]"),
        "{signatures:?}"
    );
}

/// Negative parity for the rows above: absorbing a runtime extent must not
/// absorb the CLAIM the callee's shared dimension makes about it. `both`
/// claims one extent `k` for two parameters and receives a 2-element and a
/// 3-element tensor, so the program is well typed and the disagreement is a
/// `spec/04-type-system.md` section 4.7 runtime obligation.
///
/// Regression test on both lanes, and the sharper half of this change. On the
/// base sha this program was accepted in silence: the root was dropped, so
/// eval ran nothing, the C link produced no entry point, and the guard that
/// owns the claim never got to fire. Making the root realizable is what puts
/// the guard back on the execution path, so a permissive absorption that
/// merely made programs run would fail here.
#[test]
fn a_refuted_claim_through_a_dim_variable_root_traps_on_both_lanes() {
    let source = "def g(x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
        def both(a: tensor[k, f32], b: tensor[k, f32]) -> tensor[k, f32] = add(a, b)\n\
        def main() = both(g(to_tensor([1.0f32, 2.0f32, 3.0f32])), \
        g(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])))\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("refuted_dim_variable_root.ch");
    let report = check(source, &path);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    assert!(
        report["errors"].as_array().is_some_and(Vec::is_empty),
        "{report}"
    );

    let eval = eval_file(&path);
    assert!(
        !eval.status.success(),
        "a refuted claim must not produce a value: {}",
        String::from_utf8_lossy(&eval.stdout)
    );
    let eval_stderr = String::from_utf8_lossy(&eval.stderr).to_string();
    assert!(
        eval_stderr.contains("numeric trap: domain in load at int64"),
        "[04-NUM-9]'s exact trap line: {eval_stderr}"
    );
    assert!(
        eval_stderr.contains("extent `k`: a axis 0 = 2, b axis 0 = 3"),
        "and the disagreeing sources, axis and observed values: {eval_stderr}"
    );

    let out_dir = dir.path().join("refuted-out");
    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted_path = out_dir.join("refuted_dim_variable_root.c");
    let emitted = fs::read_to_string(&emitted_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", emitted_path.display()));
    assert!(
        emitted.contains("int main("),
        "the realizable root owes a C entry point for the guard to run in:\n{emitted}"
    );
    let compiled = compile_and_run_c(&out_dir, "refuted_dim_variable_root");
    assert!(
        !compiled.status.success(),
        "the compiled binary must trap too: {}",
        String::from_utf8_lossy(&compiled.stdout)
    );
    let compiled_output = format!(
        "{}{}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert!(
        compiled_output.contains("numeric trap: domain in load at int64"),
        "C reports the same trap line as eval: {compiled_output}"
    );
    assert!(
        compiled_output.contains("extent `k`: a axis 0 = 2, b axis 0 = 3"),
        "and the same context: {compiled_output}"
    );
}
