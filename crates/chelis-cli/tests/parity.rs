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
//! There is deliberately NO tolerant fallback for tensor lines. The old
//! mismatch path (re-parse both lines as `Vec<f64>`, compare under 1e-6)
//! was removed by chelis#729 Phase 0: it silently converted integer
//! divergences into passing float comparisons - exactly the path a real
//! bug takes (chelis#687). Ops with a legitimate cross-lane value
//! tolerance get it from the per-op tolerance table
//! (`spec/design/dtype_semantics.md` §C4.5, to be authored into spec/05
//! [05-OBS-3]) once it exists, never from a blanket re-parse. The corpus
//! passes byte-exact
//! today (123/123 lines) because it prints dyadic floats exclusively; a
//! new example printing a computed non-dyadic float will fail here for
//! formatting reasons until chelis#732 Phase 2 lands byte-identical
//! rendering - that is the release valve.
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

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

// -----------------------------------------------------------------------------
// Corpus discovery
// -----------------------------------------------------------------------------

fn examples_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples")
        .canonicalize()
        .expect("examples directory should exist")
}

/// Discover every `.ch` file directly under `examples/`. Skip the
/// `examples/illustrative/` subtree (those are syntax/design specimens, not
/// the executable Phase 0 corpus), and skip any other subtrees.
fn discover_executable_examples() -> Vec<PathBuf> {
    let root = examples_root();
    let mut paths = Vec::new();
    for entry in fs::read_dir(&root).expect("read examples dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("ch") {
            paths.push(path);
        }
    }
    paths.sort();
    paths
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
// Byte-exact comparison
// -----------------------------------------------------------------------------

/// Compare two stdout byte-streams under the parity invariant: same line
/// count, every line byte-equal.
///
/// The former mismatch fallback (re-parse both lines via a
/// `parse_tensor_line -> Vec<f64>` and compare under 1e-6 tolerance) is
/// deliberately gone (chelis#729 Phase 0, chelis#687): it engaged exactly
/// when a real divergence was present and re-read integer payloads as
/// floats, so an int64 corruption above 2^53 could never fail this
/// harness. A mismatch now REPORTS. chelis#732 Phase 2 is the release
/// valve for legitimate float-formatting differences.
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
        if e != c {
            return Err(format!(
                "[{label}] line {i} differs between lanes (byte-exact contract; \
                 no tolerant fallback exists - see the module docs):\n  eval: {e}\n  c:    {c}",
            ));
        }
    }
    Ok(())
}

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
// the file. The corpus is small (~10 files) so individual tests are
// preferable to a parametric loop that hides which file regressed.
//
// Files split into two shapes:
//   * "executable": top-level bindings produce stdout and the C target
//     produces a runnable binary
//   * "library":    only `def`s, no top-level work; both lanes emit nothing.
//                   We compile the C source as an object file to confirm the
//                   backend output is well-formed.
//
// If the corpus list changes, update `parity_corpus_is_complete` below so
// the harness fails loud rather than silently shrinking.

#[test]
fn parity_dict_foundation() {
    drive_parity(&examples_root().join("dict_foundation.ch"), true);
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
fn parity_scalar_string_foundation() {
    drive_parity(&examples_root().join("scalar_string_foundation.ch"), true);
}

#[test]
fn parity_tensor_structural_ops() {
    drive_parity(&examples_root().join("tensor_structural_ops.ch"), true);
}

// Library-only programs (no `main` / no top-level work). Both lanes emit
// nothing; we still build the C source as an object to prove the backend is
// happy.

#[test]
fn parity_hello_tensor_library_only() {
    drive_parity(&examples_root().join("hello_tensor.ch"), false);
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

// The opaque-invariants worked example (RFC `opaque_invariants_rfc.md`).
// Library-only: `@opaque`/`@invariant` declare the `Probability` type and
// `@property` desugars to a `bool` def, so there is no top-level work and both
// lanes emit nothing. The scalar unit-interval invariant lowers cleanly
// through the C backend.
#[test]
fn parity_opaque_invariants_library_only() {
    drive_parity(&examples_root().join("opaque_invariants.ch"), false);
}

// The `Simplex` tolerance-band variant: a tensor-field `sum(p.weights)`
// invariant. Like `Probability` it is library-only (only `@opaque`/
// `@invariant` declarations plus exported producers and a `@property`, so
// both lanes emit nothing). The invariant predicate is declaration metadata
// consumed only by `chelis prove`; it is never lowered to runtime IR, so the
// runtime IR audit now skips it and the example lowers cleanly through the C
// backend. Promoted from `examples/illustrative/` once that audit stopped
// rejecting the declaration metadata.
#[test]
fn parity_opaque_invariants_simplex_library_only() {
    drive_parity(&examples_root().join("opaque_invariants_simplex.ch"), false);
}

#[test]
fn parity_rank_poly_borrow_library_only() {
    drive_parity(&examples_root().join("rank_poly_borrow.ch"), false);
}

// -----------------------------------------------------------------------------
// Corpus completeness guard
// -----------------------------------------------------------------------------

/// Catch the silent-shrinkage failure mode: if someone adds a new `.ch` file
/// to `examples/` without wiring it into the parity harness, this test
/// fails so the harness can't quietly stop covering the new file.
#[test]
fn parity_corpus_is_complete() {
    let known: &[&str] = &[
        "dict_foundation.ch",
        "hello_tensor.ch",
        "iter_foundation.ch",
        "linreg.ch",
        "list_foundation.ch",
        "mnist.ch",
        "opaque_invariants.ch",
        "opaque_invariants_simplex.ch",
        "rank_poly_borrow.ch",
        "scalar_string_foundation.ch",
        "tensor_structural_ops.ch",
        "transformer_block.ch",
        "vmap_relu.ch",
    ];
    let actual: Vec<String> = discover_executable_examples()
        .iter()
        .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
        .collect();
    let missing: Vec<&str> = known
        .iter()
        .copied()
        .filter(|name| !actual.iter().any(|a| a == name))
        .collect();
    let extra: Vec<&String> = actual
        .iter()
        .filter(|name| !known.contains(&name.as_str()))
        .collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "parity harness corpus drifted from examples/.\n  known but missing on disk: {missing:?}\n  on disk but not wired into harness: {extra:?}\nUpdate parity.rs to add a per-file test for any new entries.",
    );
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
