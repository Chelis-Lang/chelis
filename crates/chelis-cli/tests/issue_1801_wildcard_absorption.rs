//! Inferred-signature rows for chelis#1801: a dimension variable an
//! application's instantiation minted, that unifies with a runtime extent
//! `*` and that no argument of the same application binds to a literal or
//! named dimension, denotes that runtime extent.
//!
//! `spec/04-type-system.md` section 3.2 Application states the rule. This
//! file is the checker-visible half: what each signature reads as. The
//! executable halves are `runtime_extent_slice_a.rs`'s
//! `a_root_that_keeps_a_dim_variable_is_sized_on_both_lanes` and the
//! `root.dim_variable.nested_helper`, `root.dim_variable.polymorphic_argument`
//! and `root.dim_variable.result_only_binder` receipts in
//! `runtime_extent_slice_b.rs`.
//!
//! Every fixture here was measured before the repair; each test's comment
//! records that reading, names the sha it was measured on (`ae9260727` for the
//! rows this file opened with, `0820ee28e` for the ones added over the rebase),
//! and says whether the assertion is a regression test (the reading changed)
//! or a disposition lock (the reading must not change).

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
const NESTED_HELPERS: &str = "def g[n, k](x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
     def h[k](y: tensor[k, f32]) -> tensor[k, f32] = add(y, y)\n";

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
         def two[u](a: tensor[u, f32], b: tensor[u, f32]) -> tensor[u, f32] = add(a, b)\n\
         def wild_then_lit[m](x: tensor[m, f32]) = two(g(x), to_tensor([1.0f32, 2.0f32]))\n\
         def lit_then_wild[m](x: tensor[m, f32]) = two(to_tensor([1.0f32, 2.0f32]), g(x))\n"
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
         def two[u](a: tensor[u, f32], b: tensor[u, f32]) -> tensor[u, f32] = add(a, b)\n\
         def keep[m](x: tensor[m, f32], y: tensor[m, f32]) = two(g(x), y)\n"
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
         def q[k](x: tensor[k, f32], y: tensor[k, f32]) -> tensor[k, f32] = add(x, h(g(y)))\n"
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
    let source = "def g[n, k](x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
        def both[k](a: tensor[k, f32], b: tensor[k, f32]) -> tensor[k, f32] = add(a, b)\n\
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
        eval_stderr.contains("numeric trap: domain in load at i64"),
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
        compiled_output.contains("numeric trap: domain in load at i64"),
        "C reports the same trap line as eval: {compiled_output}"
    );
    assert!(
        compiled_output.contains("extent `k`: a axis 0 = 2, b axis 0 = 3"),
        "and the same context: {compiled_output}"
    );
}

/// The nested helper reached through a POLYMORPHIC named def passed as an
/// argument. Regression test, and chelis#1925's round-1 P1.
///
/// `apply1`/`apply2` mint their own dimension variable for the shared extent,
/// and the polymorphic `h` mints one too while the ARGUMENT is inferred, which
/// is after the application has read back what its own instantiation minted.
/// Unification then identifies the two, so the variable the application owns
/// resolves to a root it does not own and the absorption has to reach the
/// alias class rather than the one variable.
///
/// On the base sha, and on this change's first head `f2238550d`, all three
/// spellings read `main :: () -> tensor[d0, f32]` with empty eval stdout and no
/// `int main(` in the emitted C. The lambda and monomorphic spellings below are
/// the controls that were already correct at `f2238550d`.
const POLY_HELPERS: &str = "def g[n, k](x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
     def h[k](y: tensor[k, f32]) -> tensor[k, f32] = add(y, y)\n";

#[test]
fn a_polymorphic_callee_argument_absorbs_through_its_alias_root() {
    // The function argument in SECOND position, and the intermediary's own
    // dimension spelled differently from the helper's.
    let signatures = signatures(&format!(
        "{POLY_HELPERS}\
         def apply2[p](v: tensor[p, f32], f: (tensor[p, f32]) -> tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
         def main() = apply2(g(to_tensor([1.0f32, 2.0f32, 3.0f32])), h)\n"
    ));
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[*, f32]"),
        "{signatures:?}"
    );
}

#[test]
fn a_polymorphic_callee_argument_absorbs_in_the_other_argument_order() {
    // The function argument FIRST, so unification reaches the alias before the
    // wildcard rather than after it. The touch is recorded on the other end of
    // the class in this order, which is why the absorbing site asks about both.
    let signatures = signatures(&format!(
        "{POLY_HELPERS}\
         def apply1[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
         def main() = apply1(h, g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n"
    ));
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[*, f32]"),
        "{signatures:?}"
    );
}

#[test]
fn a_polymorphic_callee_argument_absorbs_when_both_spell_the_same_dimension_name() {
    // Same as above with the intermediary's dimension spelled `k`, the helper's
    // own name. `unify_dim` compares semantic names before it reaches
    // `bind_dvar`, so the colliding spelling takes a different path into the
    // same alias class and must reach the same answer.
    let signatures = signatures(&format!(
        "{POLY_HELPERS}\
         def apply1[k](f: (tensor[k, f32]) -> tensor[k, f32], v: tensor[k, f32]) -> tensor[k, f32] = f(v)\n\
         def main() = apply1(h, g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n"
    ));
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[*, f32]"),
        "{signatures:?}"
    );
}

/// Disposition lock. A LAMBDA in the function position mints no scheme
/// dimension variable, so the intermediary's own variable is the alias root and
/// the absorption reached it at `f2238550d` already. Read `() -> tensor[*,
/// f32]` there and unchanged by the alias-root repair.
#[test]
fn a_lambda_in_the_function_position_still_absorbs() {
    let signatures = signatures(
        "def g[n, k](x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
         def apply1[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
         def main() = apply1(fn (w) -> add(w, w), g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n",
    );
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[*, f32]"),
        "{signatures:?}"
    );
}

/// Disposition lock, and the negative parity for the three rows above. A
/// MONOMORPHIC function argument binds the intermediary's variable to a
/// literal, so the alias root is a literal rather than a variable and there is
/// nothing to absorb: the extent stays `2` and the claim on it is a section 4.7
/// runtime obligation. Read `() -> tensor[2, f32]` on the base sha, at
/// `f2238550d`, and here.
#[test]
fn a_monomorphic_function_argument_keeps_its_literal_extent() {
    let signatures = signatures(
        "def g[n, k](x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
         def h3(y: tensor[2, f32]) -> tensor[2, f32] = add(y, y)\n\
         def apply1[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
         def main() = apply1(h3, g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n",
    );
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[2, f32]"),
        "{signatures:?}"
    );
}

/// Disposition lock. The absorbing site under `grad`, where the transform
/// rebuilds applications: a concrete result must stay concrete. Read
/// `() -> tensor[2, f32]` rendering `[0.0, 0.0]` on the base sha, at
/// `f2238550d`, and here.
#[test]
fn the_absorbing_site_under_grad_keeps_a_concrete_result() {
    let signatures = signatures(
        "def f[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n\
         def h(x: tensor[2, f32]) -> tensor[f32] = sum(f(x, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)\n\
         def main() = grad(h)(to_tensor([1.0f32, 2.0f32]))\n",
    );
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[2, f32]"),
        "{signatures:?}"
    );
}

/// Disposition lock, and the guard on the alias-root repair. The enclosing
/// definition's OWN named binder must not be absorbed even when the aliasing
/// call sits in its body: the root the class resolves to is not a variable any
/// instantiation minted, so this application has no authority over it. Read
/// `(tensor[seq, f32]) -> tensor[seq, f32]` on the base sha, at `f2238550d`,
/// and here. Without the mint-log guard this reads `tensor[*, f32]`.
#[test]
fn an_enclosing_binder_is_not_absorbed_through_an_aliasing_call() {
    let signatures = signatures(&format!(
        "{POLY_HELPERS}\
         def apply1[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
         def outer(s: tensor[seq, f32]) -> tensor[seq, f32] = apply1(h, s)\n"
    ));
    assert_eq!(
        signatures.get("outer").map(String::as_str),
        Some("(tensor[seq, f32]) -> tensor[seq, f32]"),
        "{signatures:?}"
    );
}

/// The sharper half of the guard above: the same enclosing binder with a
/// runtime extent actually IN the alias class, so the class does meet `*` and
/// only the mint-log guard keeps the binder free. Read `(tensor[seq, f32]) ->
/// tensor[seq, f32]` on the base sha, at `f2238550d`, and here.
#[test]
fn an_enclosing_binder_survives_a_wildcard_in_its_own_alias_class() {
    let signatures = signatures(&format!(
        "{POLY_HELPERS}\
         def apply1[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
         def outer2(s: tensor[seq, f32]) -> tensor[seq, f32] = apply1(h, g(s))\n"
    ));
    assert_eq!(
        signatures.get("outer2").map(String::as_str),
        Some("(tensor[seq, f32]) -> tensor[seq, f32]"),
        "{signatures:?}"
    );
}

/// A RESULT-ONLY binder, which chelis#1925's round-1 verification raised.
/// `outer`, `direct` and `bare` declare the same `-> tensor[seq, f32]` over
/// the same runtime extent and differ only in how many applications the
/// extent crosses on its way to the declaration.
///
/// Regression test for `direct` and `outer`, disposition lock for `bare`. On
/// `0820ee28e` `bare` already read `tensor[*, f32]` while `direct` and
/// `outer` read `tensor[seq, f32]`, so the published extent depended on the
/// number of intervening calls. `spec/04-type-system.md` section 4.7.3 says
/// "No syntactic form, binding scope, function boundary, or source-tensor
/// identity changes acceptance", and section 4.4.1 makes a dimension that
/// occurs only in the declared result output-inferred from what the body
/// produced rather than rigid. What the body produced here is a runtime
/// extent. The executable half of this reading, including the lane divergence
/// it repairs, is the `root.dim_variable.result_only_binder` pair in
/// `runtime_extent_slice_b.rs`.
///
/// The parameter-bound twin is
/// `an_enclosing_binder_is_not_absorbed_through_an_aliasing_call` above: a
/// binder a PARAMETER binds keeps its name, because the parameter type is in
/// scope while the body is inferred, so the alias class resolves to a name
/// and the absorption skips it.
#[test]
fn a_result_only_binder_is_absorbed_to_the_extent_it_met() {
    let signatures = signatures(&format!(
        "{POLY_HELPERS}\
         def apply1[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
         def outer(t: tensor[3, f32]) -> tensor[seq, f32] = apply1(h, g(t))\n\
         def direct(t: tensor[3, f32]) -> tensor[seq, f32] = h(g(t))\n\
         def bare(t: tensor[3, f32]) -> tensor[seq, f32] = g(t)\n"
    ));
    for name in ["outer", "direct", "bare"] {
        assert_eq!(
            signatures.get(name).map(String::as_str),
            Some("(tensor[3, f32]) -> tensor[*, f32]"),
            "{name}: {signatures:?}"
        );
    }
}

/// Negative parity for the row above: absorbing a result-only binder must not
/// make every symbolic-dimension program runnable. Here `seq` IS bound by a
/// parameter, so it stays a named dimension, the root has no value for it, and
/// eval refuses with the same diagnostic as on the base sha.
///
/// Disposition lock on both readings, measured on `0820ee28e` and here: the
/// signature stays `() -> tensor[seq, f32]` and eval exits non-zero with
/// `missing symbolic dimension binding \`seq\``. This row is eval-side only
/// because the program never executes on either lane.
#[test]
fn a_parameter_bound_binder_still_has_no_value_at_a_root() {
    let source = format!(
        "{POLY_HELPERS}\
         def apply1[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = f(v)\n\
         def outer(s: tensor[seq, f32]) -> tensor[seq, f32] = apply1(h, g(s))\n\
         def main() = outer(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    );
    let published = signatures(&source);
    assert_eq!(
        published.get("main").map(String::as_str),
        Some("() -> tensor[seq, f32]"),
        "{published:?}"
    );

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("parameter_bound_binder_root.ch");
    let report = check(&source, &path);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");

    let evaluated = eval_file(&path);
    assert!(
        !evaluated.status.success(),
        "a parameter-bound binder has no value here: {}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
    let stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
    assert!(
        stderr.contains("missing symbolic dimension binding `seq`"),
        "the unchanged diagnostic: {stderr}"
    );
}

/// The three-member alias class, chelis#1925's round-2 P1. `apply3` carries
/// two polymorphic function arguments beside the data one, so all three of its
/// instantiation's variables land in one class and the runtime-extent
/// argument's POSITION decides which member roots the class when the meeting
/// is recorded.
///
/// Regression test for the middle ordering, disposition lock for the other
/// two. On `0820ee28e` the middle ordering read `() -> tensor[d0, f32]` with
/// empty eval stdout and no `int main(`, while first and last read
/// `tensor[*, f32]` and executed. The assertion is written as an equality
/// ACROSS the orderings rather than three separate readings, because the
/// defect is the disagreement: `spec/04-type-system.md` section 4.7.3 forbids
/// a verdict that turns on the spelling, and an order-dependent absorption
/// could otherwise return with every individual row still green. The
/// executable halves are the `root.dim_variable.argument_order` corpus pair.
#[test]
fn a_three_member_alias_class_absorbs_in_every_argument_order() {
    let orderings = [
        (
            "first",
            "def apply3[p](v: tensor[p, f32], f: (tensor[p, f32]) -> tensor[p, f32], q: (tensor[p, f32]) -> tensor[p, f32]) -> tensor[p, f32] = q(f(v))\n\
             def main() = apply3(g(to_tensor([1.0f32, 2.0f32, 3.0f32])), h, h2)\n",
        ),
        (
            "middle",
            "def apply3[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32], q: (tensor[p, f32]) -> tensor[p, f32]) -> tensor[p, f32] = q(f(v))\n\
             def main() = apply3(h, g(to_tensor([1.0f32, 2.0f32, 3.0f32])), h2)\n",
        ),
        (
            "last",
            "def apply3[p](f: (tensor[p, f32]) -> tensor[p, f32], q: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32]) -> tensor[p, f32] = q(f(v))\n\
             def main() = apply3(h, h2, g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n",
        ),
    ];
    let mut published = Vec::new();
    for (name, body) in orderings {
        let signatures = signatures(&format!("{POLY_HELPERS}{THIRD_HELPER}{body}"));
        published.push((
            name,
            signatures
                .get("main")
                .cloned()
                .unwrap_or_else(|| format!("{signatures:?}")),
        ));
    }
    assert_eq!(published[0].1, "() -> tensor[*, f32]", "first");
    assert_eq!(
        published[1].1, published[0].1,
        "the middle ordering must read exactly as the first: {published:?}"
    );
    assert_eq!(
        published[2].1, published[0].1,
        "and so must the last: {published:?}"
    );
}

/// A second polymorphic helper, so an application can hold two function
/// arguments and put a third member in one alias class.
const THIRD_HELPER: &str = "def h2[k](z: tensor[k, f32]) -> tensor[k, f32] = add(z, z)\n";

/// Negative parity for the row above at the same class size: a binder a
/// PARAMETER binds still pins a four-member class, so the absorption reaching
/// further did not reach past the claim.
///
/// Disposition lock, measured on `0820ee28e` and here.
#[test]
fn a_four_member_binder_class_is_still_not_absorbed() {
    let signatures = signatures(&format!(
        "{POLY_HELPERS}{THIRD_HELPER}\
         def apply3[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32], q: (tensor[p, f32]) -> tensor[p, f32]) -> tensor[p, f32] = q(f(v))\n\
         def outer(s: tensor[seq, f32]) -> tensor[seq, f32] = apply3(h, g(s), h2)\n"
    ));
    assert_eq!(
        signatures.get("outer").map(String::as_str),
        Some("(tensor[seq, f32]) -> tensor[seq, f32]"),
        "{signatures:?}"
    );
}

/// The other half of the same exclusion: a literal another argument of the
/// same application supplies is a claim on the runtime extent, and it wins
/// over the absorption even when the class has four members.
///
/// Disposition lock, measured on `0820ee28e` and here: `tensor[2, f32]`,
/// which is the literal's extent and not the runtime one.
#[test]
fn a_literal_claim_still_wins_in_a_four_member_class() {
    let signatures = signatures(&format!(
        "{POLY_HELPERS}{THIRD_HELPER}\
         def apply4[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32], w: tensor[p, f32], q: (tensor[p, f32]) -> tensor[p, f32]) -> tensor[p, f32] = q(f(add(v, w)))\n\
         def main() = apply4(h, g(to_tensor([1.0f32, 2.0f32, 3.0f32])), to_tensor([1.0f32, 2.0f32]), h2)\n"
    ));
    assert_eq!(
        signatures.get("main").map(String::as_str),
        Some("() -> tensor[2, f32]"),
        "{signatures:?}"
    );
}

/// The result-only binder at three members, which regressed to its pre-fold
/// reading when the two-end query missed the middle member.
///
/// Regression test against `0820ee28e`, where this read
/// `outer :: (tensor[3, f32]) -> tensor[seq, f32]` and its root exited 1 on
/// eval with `missing symbolic dimension binding \`seq\`` while the C lane
/// emitted an entry point: the same lane divergence the two-member form
/// repairs, one alias member deeper.
#[test]
fn a_three_member_result_only_binder_is_absorbed_on_both_lanes() {
    let source = format!(
        "{POLY_HELPERS}{THIRD_HELPER}\
         def apply3[p](f: (tensor[p, f32]) -> tensor[p, f32], v: tensor[p, f32], q: (tensor[p, f32]) -> tensor[p, f32]) -> tensor[p, f32] = q(f(v))\n\
         def outer(t: tensor[3, f32]) -> tensor[seq, f32] = apply3(h, g(t), h2)\n\
         def main() = outer(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n"
    );
    let published = signatures(&source);
    assert_eq!(
        published.get("outer").map(String::as_str),
        Some("(tensor[3, f32]) -> tensor[*, f32]"),
        "{published:?}"
    );

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("three_member_result_only.ch");
    let report = check(&source, &path);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");

    let evaluated = eval_file(&path);
    let stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
    assert!(evaluated.status.success(), "{stderr}");
    assert!(
        !stderr.contains("missing symbolic dimension binding"),
        "a binder no parameter binds denotes the extent it met: {stderr}"
    );
    let rendering = String::from_utf8_lossy(&evaluated.stdout)
        .trim_end()
        .to_string();
    assert_eq!(rendering, "main = tensor(shape=[2], data=[8.0, 12.0])");

    let out_dir = dir.path().join("three-member-result-only-out");
    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted_path = out_dir.join("three_member_result_only.c");
    let emitted = fs::read_to_string(&emitted_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", emitted_path.display()));
    assert!(
        emitted.contains("int main("),
        "a realizable root owes a C entry point:\n{emitted}"
    );
    let compiled = compile_and_run_c(&out_dir, "three_member_result_only");
    assert!(
        compiled.status.success(),
        "the compiled binary must run: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&compiled.stdout).trim_end(),
        rendering,
        "C renders the root exactly as eval does"
    );
}

/// A computed callable has no C ABI representation. Eval remains supported;
/// C must refuse it instead of emitting the historical wrong answer.
#[test]
fn a_nested_application_root_is_fenced_on_c() {
    let source = format!(
        "{POLY_HELPERS}\
         def pick[p](f: tensor[p, f32] -> tensor[p, f32]) -> tensor[p, f32] -> tensor[p, f32] = f\n\
         def main() = (pick(h))(g(to_tensor([1.0f32, 2.0f32, 3.0f32])))\n"
    );
    let published = signatures(&source);
    assert_eq!(
        published.get("main").map(String::as_str),
        Some("() -> tensor[*, f32]"),
        "the absorption reaches this spelling: {published:?}"
    );

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested_application_root.ch");
    let report = check(&source, &path);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");

    let evaluated = eval_file(&path);
    assert!(
        evaluated.status.success(),
        "{}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&evaluated.stdout).trim_end(),
        "main = tensor(shape=[2], data=[4.0, 6.0])",
        "eval applies the picked function"
    );

    let out_dir = dir.path().join("nested-application-out");
    let build = build_c(&path, &out_dir);
    assert!(
        !build.status.success(),
        "C must refuse a computed callable rather than guess"
    );
    assert!(
        String::from_utf8_lossy(&build.stderr).contains("unresolved function value"),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
}

/// The fence applies even when the callable's tensor extent is concrete.
#[test]
fn a_nested_application_with_concrete_result_is_fenced_on_c() {
    let source = format!(
        "{POLY_HELPERS}\
         def pick[p](f: tensor[p, f32] -> tensor[p, f32]) -> tensor[p, f32] -> tensor[p, f32] = f\n\
         def main() = (pick(h))(to_tensor([2.0f32, 3.0f32]))\n"
    );
    let published = signatures(&source);
    assert_eq!(
        published.get("main").map(String::as_str),
        Some("() -> tensor[2, f32]"),
        "no runtime extent here, so nothing is absorbed: {published:?}"
    );

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested_application_concrete.ch");
    let report = check(&source, &path);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");

    let evaluated = eval_file(&path);
    assert_eq!(
        String::from_utf8_lossy(&evaluated.stdout).trim_end(),
        "main = tensor(shape=[2], data=[4.0, 6.0])",
        "eval applies the picked function"
    );

    let out_dir = dir.path().join("nested-application-concrete-out");
    let build = build_c(&path, &out_dir);
    assert!(
        !build.status.success(),
        "C must refuse a computed callable rather than guess"
    );
    assert!(
        String::from_utf8_lossy(&build.stderr).contains("unresolved function value"),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
}

/// Two disagreeing runtime extents reach one absorbed class through two
/// function-typed arguments. Eval rejects at host entry, naming the authored
/// binder and the first/later witnesses under section 4.7. #1788 retains that
/// same invocation boundary in C; the historical test identity stays stable.
#[test]
fn two_disagreeing_extents_in_one_class_are_refused_by_both_lanes_differently() {
    let source = format!(
        "{POLY_HELPERS}{THIRD_HELPER}\
         def g2[n, k](x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[2i64, shape(x, 0i32)]])\n\
         def apply4[p](f: tensor[p, f32] -> tensor[p, f32], v: tensor[p, f32], q2: tensor[p, f32] -> tensor[p, f32], w: tensor[p, f32]) -> tensor[p, f32] = add(f(v), q2(w))\n\
         def main() = apply4(h, g(to_tensor([1.0f32, 2.0f32, 3.0f32])), h2, g2(to_tensor([7.0f32, 8.0f32, 9.0f32])))\n"
    );
    let published = signatures(&source);
    assert_eq!(
        published.get("main").map(String::as_str),
        Some("() -> tensor[*, f32]"),
        "the class absorbs even though its two witnesses disagree: {published:?}"
    );

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("two_extents_differ.ch");
    let report = check(&source, &path);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");

    let evaluated = eval_file(&path);
    assert!(
        !evaluated.status.success(),
        "eval must refuse: {}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
    let eval_stderr = String::from_utf8_lossy(&evaluated.stderr).to_string();
    assert!(
        eval_stderr.contains("extent `p`: v axis 0 = 2, w axis 0 = 1"),
        "the entry guard retains the authored binder and witness order: {eval_stderr}"
    );

    assert!(
        eval_stderr
            .lines()
            .any(|line| line == "numeric trap: domain in load at i64"),
        "the canonical entry trap is a separate line: {eval_stderr}"
    );

    let out_dir = dir.path().join("two-extents-out");
    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let compiled = compile_and_run_c(&out_dir, "two_extents_differ");
    assert!(
        !compiled.status.success(),
        "C must refuse too: {}",
        String::from_utf8_lossy(&compiled.stdout)
    );
    let compiled_output = format!(
        "{}{}",
        String::from_utf8_lossy(&compiled.stdout),
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert!(
        compiled_output.contains("extent `p`: v axis 0 = 2, w axis 0 = 1"),
        "C retains the authored entry binder and witnesses: {compiled_output}"
    );
    assert!(
        compiled_output
            .lines()
            .any(|line| line == "numeric trap: domain in load at i64"),
        "and [04-NUM-9]'s exact entry trap line: {compiled_output}"
    );
}
