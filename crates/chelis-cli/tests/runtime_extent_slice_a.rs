//! Public CLI acceptance rows for runtime-extent Slice A (chelis#1277).

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn check(source: &str) -> Value {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("case.ch");
    fs::write(&path, source).expect("fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", "--allow-style-violations", path.to_str().unwrap()])
        .output()
        .expect("check");
    serde_json::from_slice(&output.stdout).expect("check json")
}

fn errors(report: &Value) -> Vec<&str> {
    report["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .filter_map(|error| error["message"].as_str())
        .collect()
}

fn rust_sources_below(path: &Path, files: &mut Vec<std::path::PathBuf>) {
    for entry in
        fs::read_dir(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
    {
        let entry = entry.expect("source entry");
        let path = entry.path();
        if path.is_dir() {
            rust_sources_below(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

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

#[test]
fn zero_extent_is_check_clean_and_evaluates_to_empty_tensor() {
    let source = "out = insert(scalar_to_tensor(1.0f32), 0, 0i64)\n";
    let report = check(source);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    assert!(errors(&report).is_empty(), "{report}");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("zero.ch");
    fs::write(&path, source).expect("fixture");
    let output = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            path.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("eval");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("shape=[0]") && stdout.contains("data=[]"),
        "{stdout}"
    );

    let build_dir = dir.path().join("zero-out");
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
            build_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let compiled = compile_and_run_c(&build_dir, "zero");
    assert!(
        compiled.status.success(),
        "compiled zero extent failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let compiled_stdout = String::from_utf8_lossy(&compiled.stdout);
    assert!(
        compiled_stdout.contains("shape=[0]") && compiled_stdout.contains("data=[]"),
        "{compiled_stdout}"
    );
}

#[test]
fn negative_extent_remains_a_static_type_error() {
    let report = check("out = insert(scalar_to_tensor(1.0f32), 0, -1i64)\n");
    let joined = errors(&report).join("\n");
    assert!(
        joined.contains("insert") && joined.contains("-1"),
        "{report}"
    );
}

/// A declared result one rank too high is refused, through the outer
/// unification rather than through the operation's own rank diagnostic.
///
/// `expand` has two legal result shapes, so it records a deferred obligation
/// and the ascription rejects it through the deferred-constraint path, whose
/// message names the callee. `insert` has one legal shape and nothing to
/// defer, so it builds that shape from the operand and axis alone and the
/// disagreement surfaces here as the generic let-binding mismatch. Same
/// verdict, same kind, same severity, less specific text; the pull request
/// body carries the row.
///
/// The exact text is the assertion because a looser `contains("rank")` needle
/// is satisfied by both messages and would not distinguish the two routes.
/// `insert`'s own rank diagnostic is reachable where a declared result is
/// threaded as an expected result, which
/// `insert_def_body_wrong_rank_names_the_callee` pins.
#[test]
fn shape_sourced_insert_rejects_wrong_rank_ascription() {
    let source = "def bad(g: &tensor[c, f32], x: &tensor[a, c, h, w, f32]) -> tensor[c, h, w, f32] = {\n\
        \x20 step1: tensor[c, h, w, f32] = insert(g, 1, shape(x, cast(2, int32)))\n\
        \x20 step1\n\
        }\n";
    let report = check(source);
    let joined = errors(&report).join("\n");
    assert!(
        joined.contains("tensor rank mismatch: 2 dims vs 3 dims"),
        "wrong-rank ascription must reject at check: {report}"
    );
}

/// Where the declared result reaches the call as an expected result, `insert`
/// rejects a wrong rank with its own diagnostic and names itself.
///
/// A `def` body is that route: `expr_function.rs` threads the declared return
/// type down to the application, and `app_post.rs` seeds it for both
/// spellings. Unseeded, `insert` would build its one shape and hand the
/// disagreement to the return-type unification, which names no operation, and
/// the `plus one` arm in `check_expand_signature` would be reachable from no
/// program at all.
#[test]
fn insert_def_body_wrong_rank_names_the_callee() {
    let source = "def bad(g: &tensor[c, f32], x: &tensor[a, c, h, w, f32]) -> tensor[c, h, w, f32] = insert(g, 1, shape(x, cast(2, int32)))\n";
    let report = check(source);
    let joined = errors(&report).join("\n");
    assert!(
        joined.contains("insert output rank 3 must equal input rank 1 plus one"),
        "a wrong-rank def body must reject with insert's own rank diagnostic: {report}"
    );
}

#[test]
fn bare_dimension_binder_executes_and_builds_without_symbolic_dim_ice() {
    let source = "def f(b: tensor[f32], c: tensor[k, f32]) -> tensor[k, f32] = insert(b, 0, k)\n\
        out = f(scalar_to_tensor(2.0f32), to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    let report = check(source);
    assert!(errors(&report).is_empty(), "{report}");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("binder.ch");
    fs::write(&path, source).expect("fixture");
    let eval = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            path.to_str().unwrap(),
            "--allow-style-violations",
        ])
        .output()
        .expect("eval");
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let stdout = String::from_utf8_lossy(&eval.stdout);
    assert!(
        stdout.contains("shape=[3]") && stdout.contains("[2.0, 2.0, 2.0]"),
        "{stdout}"
    );

    let out_dir = dir.path().join("out");
    let build = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().unwrap(),
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build");
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    assert!(
        !String::from_utf8_lossy(&build.stderr).contains("internal compiler error"),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let compiled = compile_and_run_c(&out_dir, "binder");
    assert!(
        compiled.status.success(),
        "compiled binder extent failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let compiled_stdout = String::from_utf8_lossy(&compiled.stdout);
    assert!(
        compiled_stdout.contains("shape=[3]") && compiled_stdout.contains("[2.0, 2.0, 2.0]"),
        "{compiled_stdout}"
    );
}

#[test]
fn stale_extent_guidance_is_removed_but_axis_guidance_stays_int32() {
    let extent = check("def bad(b: tensor[f32], k: int64) -> tensor[k, f32] = insert(b, 0, k)\n");
    let extent_errors = errors(&extent).join("\n");
    assert!(!extent_errors.contains("Form-3"), "{extent_errors}");
    assert!(!extent_errors.contains("cast(N, int32)"), "{extent_errors}");
    assert!(extent_errors.contains("int64"), "{extent_errors}");

    let axis = check(
        "def bad(x: tensor[2, f32], axis: int32) -> tensor[2, 2, f32] = insert(x, axis, 2i64)\n",
    );
    let axis_errors = errors(&axis).join("\n");
    assert!(axis_errors.contains("int32"), "{axis_errors}");

    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root");
    let mut sources = Vec::new();
    for relative in ["crates/chelis-ir/src", "crates/chelis-types/src"] {
        rust_sources_below(&repo.join(relative), &mut sources);
    }
    let obsolete = concat!("Form-", "3");
    let residues = sources
        .iter()
        .filter(|path| {
            fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("read {}: {error}", path.display()))
                .contains(obsolete)
        })
        .map(|path| {
            path.strip_prefix(repo)
                .unwrap_or(path)
                .display()
                .to_string()
        })
        .collect::<Vec<_>>();
    assert!(
        residues.is_empty(),
        "obsolete runtime-extent taxonomy remains in active compiler sources: {residues:?}"
    );
}

/// chelis#1378 red-team regression: vectorizing a shrink prepends the mapped
/// batch axis. A concrete batch must receive a concrete end bound; leaving the
/// internal `ToEnd` sentinel beside `tensor[2, ...]` makes the C backend panic
/// instead of emitting the program.
///
/// Evidentiary status per assertion. The emitted `chelis_tensor_shape(t0, 1)`
/// read is the original chelis#1378 regression assertion and predates this
/// change. The two value assertions are also regression assertions, but for
/// chelis#1397: until the root manifest kept a runtime-extent nullary root
/// this program emitted no `int main(` and evaluated to nothing, so the shifted
/// bound was only ever checked by reading the emitted source. Both lanes now
/// run it and must agree exactly.
///
/// The public checker receipt below owns the negative parity row. The IR-level
/// `vmap_rejects_element_derived_extent` remains defense in depth: a genuinely
/// batch-varying bound must still reject instead of being forced into this
/// shared-axis construction.
#[test]
fn vmap_shape_bound_with_concrete_batch_emits_c_without_to_end_ice() {
    let source = "def g(x: tensor[n, f32]) -> tensor[m, f32] = shrink(x, [[1i64, shape(x, 0)]])\n\
        def main() = vmap(g)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n";
    let report = check(source);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    assert!(errors(&report).is_empty(), "{report}");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("vmap_shape_bound.ch");
    fs::write(&path, source).expect("fixture");
    let out_dir = dir.path().join("out");
    let build = Command::cargo_bin("chelis")
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
        .expect("build");
    assert!(
        build.status.success(),
        "concrete mapped batch must emit C without an unresolved ToEnd ICE:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&build.stdout),
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted_path = out_dir.join("vmap_shape_bound.c");
    let emitted = fs::read_to_string(&emitted_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", emitted_path.display()));
    assert!(
        emitted.contains("chelis_tensor_shape(t0, 1)"),
        "the shared shape extent must still read the unbatched function's axis 0, shifted behind the mapped batch axis:\n{emitted}"
    );
    assert!(
        emitted.contains("int main("),
        "a runtime-extent nullary root owes a C entry point:\n{emitted}"
    );

    let eval = eval_file(&path);
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let evaluated = String::from_utf8_lossy(&eval.stdout);
    assert_eq!(
        evaluated.trim_end(),
        "main = tensor(shape=[2, 2], data=[2.0, 3.0, 5.0, 6.0])",
        "the mapped shrink keeps element 1 of each row"
    );
    let compiled = compile_and_run_c(&out_dir, "vmap_shape_bound");
    assert!(
        compiled.status.success(),
        "compiled mapped shrink failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&compiled.stdout).trim_end(),
        evaluated.trim_end(),
        "both lanes render the mapped shrink identically"
    );
}

/// Shared `chelis eval --file` invocation for the runtime-extent rows below.
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

/// Shared `chelis build --target c` invocation for the runtime-extent rows below.
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

/// chelis#1397 regression test. A nullary root whose result carries a runtime
/// extent (`() -> tensor[*, f32]`) is an owed root on both lanes. spec/04
/// §4.7: a `*` result axis is an unknown extent, not an unknown rank, so the
/// value exists and the ABI is the ordinary runtime tensor handle. On the base
/// sha the root manifest dropped it, and every assertion below fails: eval
/// prints nothing and warns "nothing to evaluate", and the emitted C carries no
/// `int main(`, so the program links to an object-only exit 0.
#[test]
fn a_runtime_extent_nullary_root_executes_identically_on_both_lanes() {
    let source = "def g(x: tensor[n, f32]) -> tensor[m, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
        def main() = g(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    let report = check(source);
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    assert!(errors(&report).is_empty(), "{report}");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wildcard_root.ch");
    fs::write(&path, source).expect("fixture");

    let eval = eval_file(&path);
    assert!(
        eval.status.success(),
        "{}",
        String::from_utf8_lossy(&eval.stderr)
    );
    let evaluated = String::from_utf8_lossy(&eval.stdout);
    assert_eq!(
        evaluated.trim_end(),
        "main = tensor(shape=[2], data=[2.0, 3.0])",
        "the realized extent sizes the rendered output"
    );
    assert!(
        !String::from_utf8_lossy(&eval.stderr).contains("nothing to evaluate"),
        "a runtime-extent root is not a bare declaration: {}",
        String::from_utf8_lossy(&eval.stderr)
    );

    let out_dir = dir.path().join("out");
    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted_path = out_dir.join("wildcard_root.c");
    let emitted = fs::read_to_string(&emitted_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", emitted_path.display()));
    assert!(
        emitted.contains("int main("),
        "a runtime-extent nullary root owes a C entry point:\n{emitted}"
    );
    let compiled = compile_and_run_c(&out_dir, "wildcard_root");
    assert!(
        compiled.status.success(),
        "compiled runtime-extent root failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&compiled.stdout).trim_end(),
        evaluated.trim_end(),
        "both lanes render the runtime-extent root identically"
    );
}

/// Negative parity for the row above, and a chelis#1397 regression test in its
/// own right. Executing a runtime-extent root must not execute one whose claim
/// is refuted: `bad` claims one extent `k` for both parameters and receives a
/// 2-element and a 3-element tensor. Both lanes trap with the same rendering.
///
/// On the base sha this program was accepted in silence on both lanes. The root
/// was dropped, so eval ran nothing and the C link produced no entry point, and
/// the guard that owns the claim never got to fire. That silence is the defect
/// this row closes, so this is a regression test rather than a disposition lock.
#[test]
fn a_refuted_claim_under_a_runtime_extent_root_traps_on_both_lanes() {
    let source = "def g(x: tensor[n, f32]) -> tensor[m, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
        def bad(p: tensor[k, f32], q: tensor[k, f32]) -> tensor[j, f32] = shrink(add(p, q), [[0i64, shape(p, 0i32)]])\n\
        def main() = bad(g(to_tensor([1.0f32, 2.0f32, 3.0f32])), to_tensor([1.0f32, 2.0f32, 3.0f32]))\n";
    let report = check(source);
    assert_eq!(
        report["score"].as_f64(),
        Some(1.0),
        "a refuted claim is a runtime trap, not a static error: {report}"
    );
    assert!(errors(&report).is_empty(), "{report}");

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("refuted_root.ch");
    fs::write(&path, source).expect("fixture");

    let expected = "extent `k`: p axis 0 = 2, q axis 0 = 3";
    let eval = eval_file(&path);
    assert!(
        !eval.status.success(),
        "a refuted claim must not evaluate: {}",
        String::from_utf8_lossy(&eval.stdout)
    );
    let eval_stderr = String::from_utf8_lossy(&eval.stderr);
    assert!(
        eval_stderr.contains(expected) && eval_stderr.contains("domain in load"),
        "{eval_stderr}"
    );

    let out_dir = dir.path().join("out");
    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "{}",
        String::from_utf8_lossy(&build.stderr)
    );
    let emitted_path = out_dir.join("refuted_root.c");
    let emitted = fs::read_to_string(&emitted_path)
        .unwrap_or_else(|error| panic!("read {}: {error}", emitted_path.display()));
    assert!(
        emitted.contains("int main("),
        "the refuted program still owes a C entry point, or the trap never runs:\n{emitted}"
    );
    let compiled = compile_and_run_c(&out_dir, "refuted_root");
    assert!(
        !compiled.status.success(),
        "the compiled refuted claim must trap: {}",
        String::from_utf8_lossy(&compiled.stdout)
    );
    let compiled_stderr = String::from_utf8_lossy(&compiled.stderr);
    assert!(
        compiled_stderr.contains(expected) && compiled_stderr.contains("domain in load"),
        "{compiled_stderr}"
    );
}

#[test]
fn vmap_rejects_element_derived_extent_at_public_checker() {
    let named = "def g(x: tensor[n, f32]) -> tensor[m, f32] = {\n\
        \x20 end = cast(tensor_to_scalar(sum(x, cast(0, int32))), int64)\n\
        \x20 shrink(x, [[0i64, end]])\n\
        }\n\
        out = vmap(g)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n";
    let inline = "out = vmap(fn (x: tensor[3, f32]) -> {\n\
        \x20 end = cast(tensor_to_scalar(sum(x, cast(0, int32))), int64)\n\
        \x20 shrink(x, [[0i64, end]])\n\
        })(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n";
    let stored_inline = "out = {\n\
        \x20 g = fn (x: tensor[3, f32]) -> {\n\
        \x20   end = cast(tensor_to_scalar(sum(x, cast(0, int32))), int64)\n\
        \x20   shrink(x, [[0i64, end]])\n\
        \x20 }\n\
        \x20 vmap(g)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n\
        }\n";
    for (form, source) in [
        ("named def", named),
        ("inline fn", inline),
        ("stored inline fn", stored_inline),
    ] {
        let report = check(source);
        let joined = errors(&report).join("\n");
        assert!(
            report["score"].as_f64().is_some_and(|score| score < 1.0),
            "a public checker rejection must lower the fitness score for {form}: {report}"
        );
        for required in [
            "batch_varying_extent",
            "shrink",
            "vmapped argument 'x'",
            "shape() or a scalar argument",
        ] {
            assert!(
                joined.contains(required),
                "missing {required:?} from the public diagnostic for {form}: {report}"
            );
        }
    }
}

#[test]
fn vmap_rejects_helper_result_derived_from_tensor_elements() {
    let source = "def data_end(x: tensor[n, f32]) -> int64 = \
        cast(tensor_to_scalar(sum(x, cast(0, int32))), int64)\n\
        def g(x: tensor[n, f32]) -> tensor[m, f32] = \
        shrink(x, [[0i64, data_end(x)]])\n\
        out = vmap(g)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n";
    let report = check(source);
    let joined = errors(&report).join("\n");
    assert!(
        joined.contains("batch_varying_extent")
            && joined.contains("shrink")
            && joined.contains("vmapped argument 'x'"),
        "a helper call must preserve the element-dependency proof: {report}"
    );
}

#[test]
fn vmap_accepts_shape_and_shared_scalar_extent_sources() {
    let shape_source = "def g(x: tensor[n, f32]) -> tensor[m, f32] = \
        shrink(x, [[0i64, shape(x, 0)]])\n\
        out = vmap(g)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n";
    let shape_report = check(shape_source);
    assert_eq!(shape_report["score"].as_f64(), Some(1.0), "{shape_report}");
    assert!(errors(&shape_report).is_empty(), "{shape_report}");

    let scalar_source = "def g(x: tensor[n, f32], end: int64) -> tensor[m, f32] = \
        shrink(x, [[0i64, end]])\n\
        out = vmap(g)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]), 2i64)\n";
    let scalar_report = check(scalar_source);
    assert_eq!(
        scalar_report["score"].as_f64(),
        Some(1.0),
        "{scalar_report}"
    );
    assert!(errors(&scalar_report).is_empty(), "{scalar_report}");

    let inline_shape_source = "out = vmap(fn (x: tensor[3, f32]) -> \
        shrink(x, [[0i64, shape(x, 0)]]))(\
        to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n";
    let inline_shape_report = check(inline_shape_source);
    assert_eq!(
        inline_shape_report["score"].as_f64(),
        Some(1.0),
        "{inline_shape_report}"
    );
    assert!(
        errors(&inline_shape_report).is_empty(),
        "{inline_shape_report}"
    );

    let inline_scalar_source = "out = vmap(fn (x: tensor[3, f32], end: int64) -> \
        shrink(x, [[0i64, end]]))(\
        to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]), 2i64)\n";
    let inline_scalar_report = check(inline_scalar_source);
    assert_eq!(
        inline_scalar_report["score"].as_f64(),
        Some(1.0),
        "{inline_scalar_report}"
    );
    assert!(
        errors(&inline_scalar_report).is_empty(),
        "{inline_scalar_report}"
    );

    let stored_shape_source = "out = {\n\
        \x20 g = fn (x: tensor[3, f32]) -> shrink(x, [[0i64, shape(x, 0)]])\n\
        \x20 vmap(g)(to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]]))\n\
        }\n";
    let stored_shape_report = check(stored_shape_source);
    assert_eq!(
        stored_shape_report["score"].as_f64(),
        Some(1.0),
        "{stored_shape_report}"
    );
    assert!(
        errors(&stored_shape_report).is_empty(),
        "{stored_shape_report}"
    );
}
