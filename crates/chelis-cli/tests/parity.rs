//! Phase 2 — three-runtime parity test harness for the chelis closure campaign.
//!
//! The closure campaign ratchets the IR evaluator, the C backend, and the type
//! checker onto a single primitive set. This file is the gate that bakes the
//! parity invariant into the corpus:
//!
//!     For every `.ch` file in `examples/`:
//!
//!       1. `chelis check <file>`              -> score == 1.0
//!       2. `chelis eval --file <file>`        -> stdout (IR evaluator lane)
//!       3. `chelis build --target c <file>`   -> writes <name>.c + runtime
//!       4. `gcc <name>.c -L. -lchelis_runtime -o <name>` -> binary
//!       5. `<name>`                           -> stdout (C backend lane)
//!
//!     Then assert the eval lane and the C lane agree: every line
//!     byte-equal, tensor lines included.
//!
//! The runtime-shaped dropout example retains a bounded per-file exception:
//! clean check, exact executable eval output, and the current typed C rejection
//! are all tested. The fixed-stream example runs the ordinary three-lane
//! driver and also pins its exact values, including the next draw after AD.
//! The annotated concat/softmax example likewise pins check, all eval values,
//! and its explicit C rejection; it is not a C parity claim.
//!
//! There is deliberately NO tolerant fallback for tensor lines. The old
//! mismatch path (re-parse both lines as `Vec<f64>`, compare under 1e-6)
//! was removed by chelis#729 Phase 0: it silently converted integer
//! divergences into passing float comparisons - exactly the path a real
//! bug takes (chelis#687). This mixed-operation example corpus has no one
//! operation identity, so it uses the exact branch of the shared Phase 3
//! comparator. Single-operation oracles may use the same comparator's
//! [05-OBS-3] table branch when the operation and arithmetic-width
//! preconditions are known.
//!
//! chelis#732 Phase 2 restored plain BYTE equality on every line: both
//! lanes now render through the one frozen [05-OBS] grammar (eval via
//! `format_element`, compiled C via the generated print helper and
//! `chelis_format_shortest`), so the Phase 1 interim value-equivalence
//! (`migrated_render_equivalent`) is deleted as that phase promised. Any
//! line diff is a real divergence and reports raw.
//!
//! Examples that compile to an object only (no `main`) — i.e. files that
//! define functions but never invoke them at top level — produce empty eval
//! output. We still check that both lanes parse and codegen cleanly, and
//! that both lanes produce empty stdout (vacuous parity). The C source is
//! compiled with `gcc -c` to confirm the object file is well-formed. There
//! is no executable to run, so the run+compare step is skipped for those
//! files (with a comment noting the reason).
//!
//! This harness *wraps* the existing executable corpus. It does not invent new
//! programs (per the closure plan: corpus expansion is its own task). If the
//! corpus is too narrow to give meaningful parity coverage, that is something
//! to surface, not paper over.
//!
//! Don't trust green: the inverse — a deliberate eval-runtime regression —
//! must make this harness fail. That was demonstrated manually during the
//! initial landing (forcing `tensor_trace_value` to return `sum + 1.0`
//! causes `tensor_structural_ops` parity to fail with a precise
//! line/element diff).
//!
//! # Independent result controls (chelis#1351)
//!
//! Equal output proves agreement on this corpus, not correctness. A defect
//! shared by both implementations can pass this comparison, as chelis#1349 did.
//! `issue_1351_structural_axis_independent_oracle` supplies independent
//! [05-OP-33] expectations for `diagonal` and `trace` on its declared cases.
//! It compares each implementation with those expectations separately.
//!
//! Source-mutation receipt, 2026-09-14, at `07590f629b5687f018e0814260818d59372a614f`:
//!
//! - Eval only: read zero for the second diagonal axis. Both tests failed.
//!   The independent test named the eval result.
//! - C only: apply the same edit in `chelis_tensor_diagonal`. Both tests failed.
//!   The independent test named the native result.
//! - Both implementations: apply both edits. `parity_tensor_structural_ops`
//!   passed, while the independent test rejected the shared wrong answer.
//! - Reference only: add one to each expected diagonal element. Three
//!   hand-computed controls and the execution test failed.
//!
//! The runtime came from each current Cargo artifact record, in a dedicated
//! directory. The changed runtime had a different recorded archive digest.
//! These probes rebuilt and executed actual code. They did not replace output
//! strings in the comparator. All temporary mutations were restored.
//!
//! The paired command selected `parity_tensor_structural_ops` and
//! `structural_axis_lanes_match_independent_op33_expectations` from the
//! `parity` and `issue_1351_structural_axis_independent_oracle` targets with
//! `cargo nextest run -p chelis-cli --no-fail-fast`. The reference-only probe
//! ran all seven tests in the independent target. Final positive results are
//! recorded in the PR receipt.
//!
//! This supplement covers its structural-operation cases only. It does not
//! prove arithmetic reduction order, result dtypes, or other operations.

use assert_cmd::Command;
use chelis_types::agreement::compare_exact_observations;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

#[path = "common/unsupported_wording.rs"]
mod unsupported_wording;

#[path = "common/parity_corpus.rs"]
mod parity_corpus;

// -----------------------------------------------------------------------------
// Corpus discovery
// -----------------------------------------------------------------------------

fn examples_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .canonicalize()
        .expect("examples directory should exist")
}

// -----------------------------------------------------------------------------
// Lane 1: chelis check
// -----------------------------------------------------------------------------

fn run_check(path: &Path) -> Value {
    let out = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&out).expect("check output should be JSON")
}

fn assert_check_clean(path: &Path) {
    let json = run_check(path);
    assert_eq!(
        json["score"].as_f64().unwrap_or(0.0),
        1.0,
        "{} did not pass `chelis check` cleanly: {}",
        path.display(),
        serde_json::to_string_pretty(&json).unwrap_or_default(),
    );
    assert_eq!(
        json["errors"].as_array().map(|a| a.len()).unwrap_or(0),
        0,
        "{} produced check errors",
        path.display(),
    );
    assert_eq!(
        json["unresolved_names"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or(0),
        0,
        "{} produced unresolved names",
        path.display(),
    );
}

// -----------------------------------------------------------------------------
// Lane 2: chelis eval --file
// -----------------------------------------------------------------------------

fn run_eval(path: &Path) -> Vec<u8> {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone()
}

// -----------------------------------------------------------------------------
// Lane 3: chelis build --target c, then gcc, then run
// -----------------------------------------------------------------------------

fn run_build_c(path: &Path, out_dir: &Path) {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
}

fn assert_exact_tagged_c_callers(stem: &str, source: &str) {
    for retired in [
        "chelis_option_i64",
        "chelis_option_f64",
        "chelis_parse_int64(",
        "chelis_parse_f64(",
        "chelis_dict_get_i64(",
        "chelis_dict_get_f64(",
    ] {
        assert!(
            !source.contains(retired),
            "[{stem}] generated C restored retired ABI spelling `{retired}`:\n{source}"
        );
    }

    let exact_call = |name: &str| {
        source
            .lines()
            .find(|line| line.contains(name))
            .unwrap_or_else(|| panic!("[{stem}] generated C did not call `{name}`:\n{source}"))
    };
    match stem {
        "dict_foundation" => {
            let call = exact_call("chelis_dict_get_scalar(");
            assert!(call.contains("CHELIS_DTYPE_I64"), "{call}");
        }
        "scalar_string_foundation" => {
            let calls = source
                .lines()
                .filter(|line| line.contains("chelis_parse_scalar("))
                .collect::<Vec<_>>();
            assert!(
                calls.iter().any(|line| line.contains("CHELIS_DTYPE_I64"))
                    && calls.iter().any(|line| line.contains("CHELIS_DTYPE_F64")),
                "[scalar_string_foundation] parse calls must carry both exact result dtypes:\n{source}"
            );
        }
        "tensor_structural_ops" => {
            let call = exact_call("chelis_tensor_einsum(");
            assert!(call.contains("CHELIS_DTYPE_F32"), "{call}");
        }
        _ => {}
    }
}

fn generated_source_needs_blas(out_dir: &Path, source: &str) -> bool {
    fs::read_to_string(out_dir.join(source))
        .map(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
        .unwrap_or(false)
}

fn cpu_toolchain(out_dir: &Path, source: &str) -> chelis_backend_c::toolchain::NativeToolchain {
    chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: generated_source_needs_blas(out_dir, source),
        },
    )
}

/// Try to link the generated C into a binary. Returns `Ok(binary_path)` on
/// success, or `Err(reason)` if linking fails (e.g. missing BLAS headers in
/// this environment, or there is no `main` because the program defines only
/// library functions).
fn try_link(out_dir: &Path, source: &str, binary: &str) -> Result<PathBuf, String> {
    let toolchain = cpu_toolchain(out_dir, source);
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    // Use -O0 for predictable parity (no FMA/reassoc reordering surprises).
    cmd.arg("-O0");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(source);
    cmd.args(["-L.", "-lchelis_runtime"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", binary]);
    let output = cmd
        .output()
        .map_err(|e| format!("gcc failed to spawn: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "gcc link failed (status {}):\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    Ok(out_dir.join(binary))
}

/// Compile the generated C as an object file only. Used for library-only
/// programs (no `main`) so we still prove the C backend produced
/// well-formed output.
fn try_compile_object(out_dir: &Path, source: &str) -> Result<(), String> {
    let toolchain = cpu_toolchain(out_dir, source);
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O0");
    cmd.args(&toolchain.compile_flags);
    cmd.args(["-I.", "-c", source]);
    let output = cmd
        .output()
        .map_err(|e| format!("gcc failed to spawn: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "gcc -c failed (status {}):\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr),
        ));
    }
    Ok(())
}

fn run_binary(binary: &Path) -> Vec<u8> {
    let out = StdCommand::new(binary)
        .current_dir(binary.parent().expect("binary has parent"))
        .output()
        .expect("compiled binary should run");
    assert!(
        out.status.success(),
        "compiled binary `{}` exited with status {}: stderr=\n{}",
        binary.display(),
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    out.stdout
}

// -----------------------------------------------------------------------------
// Shared Phase 3 comparison
// -----------------------------------------------------------------------------

/// Compare two stdout byte-streams under the parity invariant: same line
/// count, every line byte-equal.
///
/// The former mismatch fallback (re-parse both lines via a
/// `parse_tensor_line -> Vec<f64>` and compare under 1e-6 tolerance) is
/// deliberately gone (chelis#729 Phase 0, chelis#687): it engaged exactly
/// when a real divergence was present and re-read integer payloads as
/// floats, so an i64 corruption above 2^53 could never fail this
/// harness. A mismatch now REPORTS. chelis#732 Phase 2 is the release
/// valve for formatting differences by making both lanes canonical;
/// Phase 3's explicit per-op table is the only release valve for a genuine
/// value difference, and this mixed-operation corpus has no eligible row.
///
/// Returns `Ok(())` if parity holds, or `Err(reason)` on first divergence.
fn assert_parity(eval_out: &[u8], c_out: &[u8], label: &str) -> Result<(), String> {
    let eval_text = String::from_utf8_lossy(eval_out);
    let c_text = String::from_utf8_lossy(c_out);

    let eval_lines: Vec<&str> = eval_text.lines().collect();
    let c_lines: Vec<&str> = c_text.lines().collect();

    if eval_lines.len() != c_lines.len() {
        return Err(format!(
            "[{label}] line count mismatch: eval={} c={}\n--- eval ---\n{}\n--- c ---\n{}",
            eval_lines.len(),
            c_lines.len(),
            eval_text,
            c_text,
        ));
    }

    for (i, (e, c)) in eval_lines.iter().zip(c_lines.iter()).enumerate() {
        if let Err(error) = compare_exact_observations(&format!("{label} line {i}"), e, c) {
            return Err(format!(
                "[{label}] line {i} differs between lanes under the shared \
                 chelis#732 Phase 3 comparator (this mixed-op corpus is \
                 exact-only): {error}",
            ));
        }
    }
    Ok(())
}

// The chelis#732 Phase 1 interim line equivalence
// (`migrated_render_equivalent`) lived here while the compiled lane kept
// its pre-contract printf forms; Phase 2's generated printer restored
// byte equality on every line and the function is deleted as Phase 1
// promised. Its red-team-hardened blind-spot behaviors (exact-i64
// refusal above 2^53, second-tensor and shape divergences, the rank-0
// wrapper class) are now simply line diffs, which byte equality reports
// by construction.

// -----------------------------------------------------------------------------
// Shared per-file driver
// -----------------------------------------------------------------------------

/// Drive the three lanes for a single `.ch` file.
///
/// `expect_executable` selects between the two corpus shapes:
/// - `true`: the file produces top-level output and the C codegen yields an
///   executable. We link, run, and compare stdout.
/// - `false`: the file is library-only (no `main`). We confirm `chelis eval`
///   and `chelis build --target c` both succeed, the C source compiles to an
///   object file, and both lanes produce empty stdout (vacuous parity).
fn drive_parity(path: &Path, expect_executable: bool) {
    assert_check_clean(path);

    let eval_out = run_eval(path);

    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .expect("file stem is utf8")
        .to_string();
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("build");
    run_build_c(path, &out_dir);

    let c_source = format!("{stem}.c");
    let source = fs::read_to_string(out_dir.join(&c_source)).expect("generated C source");
    assert_exact_tagged_c_callers(&stem, &source);
    if !expect_executable {
        // Library-only path: prove the C source compiles cleanly and confirm
        // both lanes emit nothing visible.
        try_compile_object(&out_dir, &c_source)
            .unwrap_or_else(|e| panic!("[{stem}] gcc -c failed: {e}"));
        assert!(
            eval_out.is_empty(),
            "[{stem}] expected empty eval stdout for library-only program, got:\n{}",
            String::from_utf8_lossy(&eval_out),
        );
        return;
    }

    let binary = match try_link(&out_dir, &c_source, &stem) {
        Ok(p) => p,
        Err(reason) => panic!("[{stem}] gcc link failed: {reason}"),
    };
    let c_out = run_binary(&binary);

    if let Err(reason) = assert_parity(&eval_out, &c_out, &stem) {
        panic!("parity violation: {reason}");
    }
}

// -----------------------------------------------------------------------------
// Per-file tests
// -----------------------------------------------------------------------------
//
// One test per executable-corpus file so that a failure points straight at
// the file. Individual tests are
// preferable to a parametric loop that hides which file regressed.
//
// Files split into two shapes:
//   * "executable": top-level bindings produce stdout and the C target
//     produces a runnable binary
//   * "library":    only `def`s, no top-level work; both lanes emit nothing.
//                   We compile the C source as an object file to confirm the
//                   backend output is well-formed.
//
// The completeness check derives membership from these test inputs. A new
// example needs its per-file test, with no second filename list to update.

#[test]
fn parity_generic_value_roots() {
    drive_parity(&examples_root().join("generic_value_roots.ch"), true);
}

#[test]
fn parity_grad_wrt_order() {
    drive_parity(&examples_root().join("grad_wrt_order.ch"), true);
}

#[test]
fn parity_grad_disconnected() {
    let path = examples_root().join("grad_disconnected.ch");
    drive_parity(&path, true);
    assert_eq!(
        run_eval(&path),
        b"out = tensor(shape=[2, 3], data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])\n"
    );
}

#[test]
fn parity_grad_extent_claim() {
    let path = examples_root().join("grad_extent_claim.ch");
    drive_parity(&path, true);
    assert_eq!(
        run_eval(&path),
        b"out.0 = tensor(shape=[3], data=[0.0, 0.0, 0.0])\nout.1 = tensor(shape=[3], data=[7.0, 7.0, 7.0])\nindependent.0 = tensor(shape=[3], data=[7.0, 7.0, 7.0])\nindependent.1 = tensor(shape=[2], data=[7.0, 7.0])\n"
    );
}

#[test]
fn parity_grad_scalar_extent_claim() {
    let path = examples_root().join("grad_scalar_extent_claim.ch");
    drive_parity(&path, true);
    assert_eq!(
        run_eval(&path),
        b"out.0 = 21.0\nout.1 = tensor(shape=[3], data=[0.0, 0.0, 0.0])\n"
    );
}

#[test]
fn parity_grad_fused_zero() {
    drive_parity(&examples_root().join("grad_fused_zero.ch"), true);
}

#[test]
fn parity_vmap_shape_value() {
    let path = examples_root().join("vmap_shape_value.ch");
    drive_parity(&path, true);
    assert_eq!(
        run_eval(&path),
        b"out = tensor(shape=[2], data=[3.0, 3.0])\n"
    );
}

#[test]
fn parity_source_file_names() {
    drive_parity(&examples_root().join("source-file-names.ch"), true);
}

#[test]
fn parity_checked_reshape() {
    drive_parity(&examples_root().join("checked_reshape.ch"), true);
}

#[test]
fn declared_operation_bounds_example() {
    drive_parity(&examples_root().join("declared_operation_bounds.ch"), true);
}

#[test]
fn parity_dict_foundation() {
    drive_parity(&examples_root().join("dict_foundation.ch"), true);
}

#[test]
fn parity_dropout_entry_library_only() {
    drive_parity(&examples_root().join("dropout_entry.ch"), false);
}

#[test]
fn parity_dropout_fixed_stream() {
    let path = examples_root().join("dropout_fixed_stream.ch");
    assert_eq!(
        run_eval(&path),
        b"main.0 = tensor(shape=[4], data=[0.0, 2.0, 0.0, 0.0])\nmain.1 = tensor(shape=[4], data=[2.0, 0.0, 0.0, 0.0])\n",
    );
    drive_parity(&path, true);
}

#[test]
fn parity_dropout_static_rate() {
    let path = examples_root().join("dropout_static_rate.ch");
    assert_eq!(
        run_eval(&path),
        b"result.0 = tensor(shape=[4], data=[0.0, 2.0, 0.0, 0.0])\nresult.1 = tensor(shape=[4], data=[2.0, 0.0, 0.0, 0.0])\nresult.2 = tensor(shape=[4], data=[2.0, 2.0, 0.0, 0.0])\nresult.3 = tensor(shape=[4], data=[1.0, 1.0, 1.0, 1.0])\n",
    );
    drive_parity(&path, true);
}

#[test]
fn parity_dropout_staged_claim() {
    let path = examples_root().join("dropout_staged_claim.ch");
    assert_check_clean(&path);
    assert_eq!(
        run_eval(&path),
        b"main = tensor(shape=[2, 2], data=[2.0, 0.0, 0.0, 0.0])\n",
    );
    drive_parity(&path, true);
}

#[test]
fn parity_checked_window_geometry() {
    drive_parity(&examples_root().join("checked_window_geometry.ch"), true);
}

#[test]
fn parity_checked_sparse_axes() {
    drive_parity(&examples_root().join("checked_sparse_axes.ch"), true);
}

#[test]
fn parity_count_bool_axes() {
    drive_parity(&examples_root().join("count_bool_axes.ch"), true);
}

#[test]
fn parity_explicit_normalization() {
    drive_parity(&examples_root().join("explicit_normalization.ch"), true);
}

#[test]
fn parity_generic_explicit_shape() {
    drive_parity(&examples_root().join("generic_explicit_shape.ch"), true);
}

#[test]
fn parity_count_bool_device_entry_library_only() {
    drive_parity(&examples_root().join("count_bool_device_entry.ch"), false);
}

#[test]
fn parity_constraint_directed_risk_guards_library_only() {
    drive_parity(
        &examples_root().join("constraint_directed_risk_guards.ch"),
        false,
    );
}

#[test]
fn parity_iter_foundation() {
    drive_parity(&examples_root().join("iter_foundation.ch"), true);
}

#[test]
fn parity_list_foundation() {
    drive_parity(&examples_root().join("list_foundation.ch"), true);
}

#[test]
fn parity_checked_runtime_extents() {
    drive_parity(&examples_root().join("checked_runtime_extents.ch"), true);
}

#[test]
fn parity_mixed_signature_extents() {
    drive_parity(&examples_root().join("mixed_signature_extents.ch"), true);
}

#[test]
fn parity_ordered_extent_claims() {
    drive_parity(&examples_root().join("ordered_extent_claims.ch"), true);
}

#[test]
fn parity_pure_helper_result_claims() {
    drive_parity(&examples_root().join("pure_helper_result_claims.ch"), true);
}

#[test]
fn parity_literal_extent_claim() {
    drive_parity(&examples_root().join("literal_extent_claim.ch"), true);
}

#[test]
fn parity_integer_functions() {
    drive_parity(&examples_root().join("integer_functions.ch"), true);
}

/// chelis#1266: the record-projection broadcast a multi-input forward writes.
#[test]
fn parity_record_input_broadcast() {
    drive_parity(&examples_root().join("record_input_broadcast.ch"), true);
}

#[test]
fn parity_recursive_cast_targets() {
    drive_parity(&examples_root().join("recursive_cast_targets.ch"), true);
}

#[test]
fn parity_recursive_generic() {
    // chelis#1158: recursive generic host calls compile via bounded
    // memoized monomorphization; both lanes print the same value.
    drive_parity(&examples_root().join("recursive_generic.ch"), true);
}

#[test]
fn parity_scalar_string_foundation() {
    drive_parity(&examples_root().join("scalar_string_foundation.ch"), true);
}

#[test]
fn parity_unicode_string_foundation() {
    drive_parity(&examples_root().join("unicode_string_foundation.ch"), true);
}

#[test]
fn parity_signed_seed() {
    drive_parity(&examples_root().join("signed_seed.ch"), true);
}

#[test]
fn parity_tensor_structural_ops() {
    drive_parity(&examples_root().join("tensor_structural_ops.ch"), true);
}

// Programs with no owed [05-OBS-7] roots remain library-only: both lanes emit
// nothing and we build the C source as an object. Historical `_library_only`
// test names are frozen by the Phase 3 corpus oracle even where a newly owed
// pure-nullary or value root now makes the program executable.

#[test]
fn parity_hello_tensor_library_only() {
    // `main()` is a pure nullary declaration and therefore an owed root.
    drive_parity(&examples_root().join("hello_tensor.ch"), true);
}

#[test]
fn parity_hash_order_determinism() {
    drive_parity(&examples_root().join("hash_order_determinism.ch"), true);
}

#[test]
fn parity_induction_bond_library_only() {
    drive_parity(&examples_root().join("induction_bond.ch"), false);
}

#[test]
fn parity_beacon_scalar_range_library_only() {
    drive_parity(&examples_root().join("beacon_scalar_range.ch"), false);
}

#[test]
fn parity_linreg_library_only() {
    drive_parity(&examples_root().join("linreg.ch"), false);
}

#[test]
fn parity_mnist_library_only() {
    drive_parity(&examples_root().join("mnist.ch"), false);
}

// Same BLAS dependency as `parity_mnist_library_only` above. The transformer
// block is heavy on `matmul`, so the generated C drags in `cblas_sgemm()`
// and the `chelis_blas.h` shim. Un-ignore on a host with OpenBLAS dev
// headers (`openblas-devel` / `libopenblas-dev`).
#[test]
#[ignore = "requires system cblas.h (install openblas-devel / libopenblas-dev)"]
fn parity_transformer_block_library_only() {
    drive_parity(&examples_root().join("transformer_block.ch"), false);
}

#[test]
fn parity_vmap_relu_library_only() {
    drive_parity(&examples_root().join("vmap_relu.ch"), false);
}

#[test]
fn parity_vmap_tensor_capture() {
    let path = examples_root().join("vmap_tensor_capture.ch");
    assert_check_clean(&path);
    assert_eq!(
        run_eval(&path),
        b"weights = tensor(shape=[2], data=[10.0, 20.0])\nout = tensor(shape=[3], data=[30.0, 60.0, 20.0])\n",
    );
    drive_parity(&path, true);
}

// The opaque-invariants worked example (RFC `opaque_invariants_rfc.md`).
// Library-only: `@opaque`/`@invariant` declare the `Probability` type and
// `@property` desugars to a `bool` def, so there is no top-level work and both
// lanes emit nothing. The scalar unit-interval invariant lowers cleanly
// through the C backend.
#[test]
fn parity_opaque_invariants_library_only() {
    drive_parity(&examples_root().join("opaque_invariants.ch"), false);
}

// The `Simplex` tolerance-band variant has an owed top-level `eps` value root.
// Its invariant predicate is declaration metadata consumed only by `chelis
// prove`; it is never lowered to runtime IR, so the runtime IR audit skips it
// and the example lowers cleanly through the C backend. Promoted from
// `examples/illustrative/` once that audit stopped rejecting the declaration
// metadata.
#[test]
fn parity_opaque_invariants_simplex_library_only() {
    drive_parity(&examples_root().join("opaque_invariants_simplex.ch"), true);
}

#[test]
fn parity_kinded_nominal_dimensions() {
    drive_parity(&examples_root().join("kinded_nominal_dimensions.ch"), true);
}

#[test]
fn parity_rank_poly_borrow_library_only() {
    drive_parity(&examples_root().join("rank_poly_borrow.ch"), false);
}

// A callable tensor entry: check/evaluate the declaration and compile its C
// object. This is target admission coverage, not a claim of GPU execution.
#[test]
fn parity_resource_target_cpu_library_only() {
    drive_parity(&examples_root().join("resource_target_cpu.ch"), false);
}

#[test]
fn parity_wildcard_extents() {
    drive_parity(&examples_root().join("wildcard_extents.ch"), true);
}

#[test]
fn parity_annotated_concat_softmax_eval_and_c_rejection() {
    let path = examples_root().join("annotated_concat_softmax.ch");
    assert_check_clean(&path);
    assert_eq!(
        run_eval(&path),
        b"output = tensor(shape=[2, 4], data=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])\n",
    );
    let directory = tempdir().unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .args(["build", path.to_str().unwrap(), "--target", "c", "--output"])
        .arg(directory.path().join("out"))
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(1),
        "C host softmax rejects without a crash"
    );
    assert!(output.stdout.is_empty());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        unsupported_wording::stderr("annotated_concat_softmax_c")
    );
}

#[test]
fn parity_caller_actual_scope() {
    drive_parity(&examples_root().join("caller_actual_scope.ch"), true);
}

#[test]
fn parity_nested_tuple_projection() {
    let path = examples_root().join("nested_tuple_projection.ch");
    assert_check_clean(&path);
    assert_eq!(run_eval(&path), b"main = 2\n");
    drive_parity(&path, true);
}

// -----------------------------------------------------------------------------
// Corpus completeness guard
// -----------------------------------------------------------------------------

/// Catch the silent-shrinkage failure mode: if someone adds a new `.ch` file
/// to `examples/` without wiring it into the parity harness, this test
/// fails so the harness can't quietly stop covering the new file.
#[test]
fn parity_corpus_is_complete() {
    parity_corpus::validate(&examples_root(), include_str!("parity.rs"))
        .unwrap_or_else(|error| panic!("parity corpus: {error}"));
}

// -----------------------------------------------------------------------------
// Self-test for the byte-exact comparator
// -----------------------------------------------------------------------------

#[test]
fn parity_comparator_accepts_byte_identical_tensor_lines() {
    let a = b"contracted = tensor(shape=[2, 2], data=[19.0, 22.0, 43.0, 50.0])\n";
    let b = b"contracted = tensor(shape=[2, 2], data=[19.0, 22.0, 43.0, 50.0])\n";
    assert!(assert_parity(a, b, "self-test-ok").is_ok());
}

/// Under the removed 1e-6 fallback this pair PASSED - the chelis#687
/// blind spot in miniature. A sub-tolerance drift must now report.
#[test]
fn parity_comparator_reports_sub_tolerance_float_drift() {
    let a = b"contracted = tensor(shape=[2, 2], data=[19.0, 22.0, 43.0, 50.0])\n";
    let b = b"contracted = tensor(shape=[2, 2], data=[19.0000001, 22.0, 43.0, 50.0])\n";
    assert!(assert_parity(a, b, "self-test-drift").is_err());
}

#[test]
fn parity_comparator_rejects_value_divergence() {
    let a = b"contracted = tensor(shape=[2, 2], data=[19.0, 22.0, 43.0, 50.0])\n";
    let b = b"contracted = tensor(shape=[2, 2], data=[19.5, 22.0, 43.0, 50.0])\n";
    assert!(assert_parity(a, b, "self-test-fail").is_err());
}

#[test]
fn parity_comparator_byte_equal_for_non_tensor() {
    let a = b"len=4, items=4, shape=2x2\n[1, 2, 3]\n";
    let b = b"len=4, items=4, shape=2x2\n[1, 2, 3]\n";
    assert!(assert_parity(a, b, "byte-eq").is_ok());
}

#[test]
fn parity_comparator_rejects_non_tensor_diff() {
    let a = b"len=4, items=4, shape=2x2\n";
    let b = b"len=5, items=4, shape=2x2\n";
    assert!(assert_parity(a, b, "byte-diff").is_err());
}
