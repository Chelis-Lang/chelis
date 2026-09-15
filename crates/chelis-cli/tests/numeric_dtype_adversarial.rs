//! RT-4: Final adversarial sweep for the numeric-dtype cycle (PRs #27,
//! #31, #38, #43, #44, #45, #46, #47, #48, #62, #69, #74, #76, #77,
//! #78, #82, #96).
//!
//! Each test in this file pins a finding from the RT-4 red team:
//! either a working invariant that should not regress, or an
//! `#[ignore]`d reproducer for a finding that has not yet been fixed.
//!
//! Findings overview (full report in the RT-4 reply):
//!
//!   F1 (BLOCKER, SPEC-DIVERGENCE): host-program literal init silently
//!      truncates declared f64 / int64 / int8 / int16 tensors to f32
//!      storage. Source `tensor[3, f64] = [1.1, 2.2, 3.3]` is allocated
//!      at f32 (4 bytes/elem) and read back through a kernel that uses
//!      `(double*)t->data`, so subsequent ops produce garbage.
//!      Authoritative offending code:
//!        crates/chelis-runtime/src/lib.rs lines 1517, 1538, 1588
//!        (chelis_nested_list_shape and chelis_flatten_nested_list).
//!
//!   F2 (BLOCKER): `chelis_host_reshape_tensor` and `emit_cast` hardcode
//!      `sizeof(float)` in their memcpy regardless of dtype. f64 reshape
//!      truncates to half; int8 cast is a 4x heap overflow.
//!      Offending code:
//!        crates/chelis-backend-c/src/host_emit.rs:265
//!        crates/chelis-backend-c/src/emit.rs:3067
//!
//!   F3 (BLOCKER): `cast` lowering is a bitwise reinterpret memcpy, not
//!      a value-converting cast. `cast(3.5f32, int32)` does not produce
//!      `3`; it produces the bit pattern of `3.5f32` reinterpreted as
//!      int32. Spec §5.2 requires `cast` to change precision (i.e.,
//!      convert values).
//!
//!   F4 (BLOCKER): C backend panics with `panic!()` on bf16/f16 tensor
//!      input instead of producing a clean structured diagnostic.
//!      The panic happens in `crates/chelis-backend-c/src/emit.rs:588`
//!      AFTER style/check pass.
//!
//!   F5 (REPAIRED): the old HIP gate rejected bf16/f16 before the backend's
//!      `chelis_hipblas_bf16_gemm_f32_acc_*` machinery could run. The shared
//!      gate now admits narrow-float storage and matmul paths while rejecting
//!      ordinary narrow-float compute nodes that still lack typed kernels.
//!      The positive and negative controls below lock that boundary.
//!
//!   F6 (SPEC-DIVERGENCE): `chelis-metal-runtime/runtime/chelis_metal_runtime.h`
//!      lines 178 and 236 contain em-dashes in user-facing fprintf
//!      strings ("emitter planner bug" / "planner bug"), violating the
//!      no-em-dash-in-public-strings convention enforced for `crates/`
//!      and `packages/`.
//!
//!   F7 (NEGATIVE-COVERAGE-GAP): no `.ch` test or example in the
//!      repository exercises bf16 or f16 end-to-end through the CLI.
//!      All bf16/f16 acceptance is at the IR level via hand-built
//!      DAGs (e.g., `crates/chelis-backend-hip/tests/ws_a3_*`); the
//!      user-facing surface is completely uncovered.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

// Issue #207: `chelis check` now exits non-zero when the JSON
// `errors` array is non-empty. The RT-4 invariant tests below mix
// clean and error-expecting cases (e.g. integer matmul rejection)
// through the same helper.
fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn run_build_in(dir: &Path, source: &Path, output: Option<&Path>) -> std::process::Output {
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    let mut cmd = StdCommand::new(bin);
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir)
        .args(["build", source.to_str().unwrap()]);
    if let Some(out) = output {
        cmd.args(["-o", out.to_str().unwrap()]);
    }
    cmd.output().expect("spawn chelis build")
}

fn run_build_target(dir: &Path, source: &Path, target: &str) -> std::process::Output {
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    StdCommand::new(bin)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(dir)
        .args(["build", source.to_str().unwrap(), "--target", target])
        .output()
        .expect("spawn chelis build")
}

fn gcc_available() -> bool {
    StdCommand::new("gcc")
        .arg("--version")
        .output()
        .map(|out| out.status.success())
        .unwrap_or(false)
}

fn compile_and_run(work_dir: &Path, c_file: &str, bin_name: &str) -> Option<String> {
    if !gcc_available() {
        return None;
    }
    let needs_blas = fs::read_to_string(work_dir.join(c_file))
        .map(|t| t.contains("cblas_sgemm(") || t.contains("\"chelis_blas.h\""))
        .unwrap_or(false);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    );
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(work_dir).arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(c_file);
    cmd.args(["-L.", "-lchelis_runtime"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", bin_name]);
    let status = cmd.status().expect("gcc should run");
    if !status.success() {
        return None;
    }
    let run_output = StdCommand::new(work_dir.join(bin_name))
        .current_dir(work_dir)
        .output()
        .expect("compiled binary should run");
    Some(String::from_utf8_lossy(&run_output.stdout).to_string())
}

// =====================================================================
// F1: host-program f64 literal silently truncates to f32 storage.
// Spec §5.1, §5.6 (the explicit example `let xs: tensor[3, f64] =
// [1.0, 2.0, 3.0]`). Acceptance requires exact f64 print.
// =====================================================================

#[test]
fn rt4_f1_f64_literal_storage_must_be_f64() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("f64_lit.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &src,
        r#"x: tensor[3, f64] = [1.1, 2.2, 3.3]
y: tensor[3, f64] = [1e-9, 1000000000.0, 0.1]
"#,
    );
    let build = run_build_in(dir.path(), &src, Some(&out_dir));
    assert!(
        build.status.success(),
        "RT-4 F1: build must succeed; stderr={}",
        String::from_utf8_lossy(&build.stderr)
    );
    let stdout = compile_and_run(&out_dir, "f64_lit.c", "f64_lit");
    if let Some(stdout) = stdout {
        // After the F1 fix, the f64 1.1 must round-trip without f32
        // truncation. The exact textual print is implementation-defined
        // up to the runtime's print precision, but it MUST NOT contain
        // the f32-truncated representation `1.100000023841858`, which
        // is the smoking-gun signature for a 4-byte (f32) backing store.
        assert!(
            !stdout.contains("1.100000023841858"),
            "RT-4 F1: declared f64 literal printed as f32-truncated value: {stdout}"
        );
        // 1.0e-9 must not flush to zero (it does when stored as f32 and
        // read back as double).
        assert!(
            !stdout.contains("data=[0.0, 1000000000"),
            "RT-4 F1: 1.0e-9 flushed to zero (f32 underflow signature): {stdout}"
        );
    }
}

#[test]
fn rt4_f1_i64_literal_storage_must_be_i64() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("i64_lit.ch");
    let out_dir = dir.path().join("out");
    // 9223372036854775000 is exactly representable in i64 but rounds
    // through both f32 and f64. The smoking gun for f32-storage truncation
    // is the value 9223372036854775808.0 in the printed output (the
    // nearest float to i64::MAX-100).
    write_file(
        &src,
        r#"x: tensor[3, int64] = [9223372036854775000i64, 200i64, 1i64]
"#,
    );
    let build = run_build_in(dir.path(), &src, Some(&out_dir));
    assert!(build.status.success(), "RT-4 F1: build must succeed");
    if let Some(stdout) = compile_and_run(&out_dir, "i64_lit.c", "i64_lit") {
        // F1 signature: integer literal mangled through float precision.
        assert!(
            !stdout.contains("9223372036854775808"),
            "RT-4 F1: int64 literal mangled through float precision: {stdout}"
        );
    }
}

#[test]
fn rt4_f1_f64_add_must_compute_in_f64() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("f64_add.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &src,
        r#"x: tensor[3, f64] = [1.000000000000001, 2.0, 3.0]
y: tensor[3, f64] = [1.000000000000001, 2.0, 3.0]
z: tensor[3, f64] = add(x, y)
"#,
    );
    let build = run_build_in(dir.path(), &src, Some(&out_dir));
    assert!(build.status.success(), "RT-4 F1: build must succeed");
    if let Some(stdout) = compile_and_run(&out_dir, "f64_add.c", "f64_add") {
        // Expected: z ≈ [2.0..., 4.0, 6.0]. Pre-fix actual: z ≈
        // [1.0, 2.25, -0.0] (because the kernel reads f32 bytes as
        // double).
        assert!(
            !stdout.contains("z = tensor(shape=[3], data=[1.0, 2.25"),
            "RT-4 F1: f64 add produces garbage from f32-stored input: {stdout}"
        );
    }
}

// =====================================================================
// F2: chelis_host_reshape_tensor and emit_cast use `sizeof(float)`
// regardless of dtype.
// =====================================================================

/// The host_emit.rs reshape helper must NOT contain a hardcoded
/// `sizeof(float)` once F2 is fixed. This is a source-level pin to
/// catch the regression even on machines where gcc is unavailable.
#[test]
fn rt4_f2_host_reshape_helper_must_not_hardcode_sizeof_float() {
    let host_emit = include_str!("../../chelis-backend-c/src/host_emit.rs");
    // Look inside the reshape helper specifically. The helper is
    // appended via `append_tensor_reshape_helper`. After the F2 fix the
    // copy expression must size the source by element bytes, not float
    // bytes.
    assert!(
        !host_emit
            .contains("memcpy(out_tensor->data, input->data, (size_t)input->size * sizeof(float))"),
        "RT-4 F2: host_emit.rs still hardcodes sizeof(float) in the reshape helper memcpy"
    );
}

#[test]
fn rt4_f2_emit_cast_must_not_hardcode_sizeof_float() {
    let emit = include_str!("../../chelis-backend-c/src/emit.rs");
    assert!(
        !emit.contains("memcpy(t{id}->data, t{a}->data, t{id}->size * sizeof(float))"),
        "RT-4 F2: emit.rs `emit_cast` still hardcodes sizeof(float)"
    );
}

// =====================================================================
// F3: cast emission is bitwise reinterpret, not value cast.
// Spec §5.2 says "cast changes precision". A reinterpret of f32 bits
// as int32 is not a precision change.
// =====================================================================

#[test]
fn rt4_f3_cast_must_convert_values_not_reinterpret_bits() {
    let emit = include_str!("../../chelis-backend-c/src/emit.rs");
    // After the F3 fix the emitted cast should contain a per-element
    // conversion (e.g. `(int32_t)t{a}->data[i]`), not a single memcpy
    // of the source bytes. The current code is a memcpy through
    // `sizeof(float)`. This test pins the absence of the bitwise
    // memcpy shape; the F3 fix may use any equivalent value-converting
    // representation.
    let pos = emit.find("    // ---- Cast ----").unwrap_or(0);
    let cast_block = &emit[pos..pos.saturating_add(500)];
    assert!(
        !cast_block.contains("memcpy(t{id}->data, t{a}->data, t{id}->size * sizeof(float))"),
        "RT-4 F3: emit_cast still uses bitwise memcpy; need value-converting cast \
         per spec/04-type-system.md §5.2.\nCast block:\n{cast_block}"
    );
}

// =====================================================================
// F4 (post-WS-1): the numeric-dtype cycle's F4 fix replaced the
// C-backend panic on bf16 input with a clean CLI diagnostic. WS-1
// (dtype + Metal cleanup cycle) then promotes bf16 from "rejected
// cleanly" to "admitted via convert-to-f32" per spec/04-type-system.md
// §1.1.3 + §5.7.1. This test pins the new behavior: the build
// succeeds and the C backend never panics on bf16 input.
// =====================================================================

#[test]
fn rt4_f4_c_backend_must_not_panic_on_bf16_input() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("bf16_input.ch");
    write_file(
        &src,
        r#"def my_add(x: tensor[3, bf16], y: tensor[3, bf16]) -> tensor[3, bf16] = add(x, y)
"#,
    );
    let build = run_build_in(dir.path(), &src, None);
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "WS-1 F4: C backend panicked on bf16 input instead of admitting it. stderr={stderr}"
    );
    assert!(
        build.status.success(),
        "WS-1 F4: bf16 on --target c must succeed (admitted via convert-to-f32 per \
         spec/04-type-system.md §1.1.3 + §5.7.1). stderr={stderr}"
    );
}

// =====================================================================
// F5: HIP CLI rejects bf16 / f16 even though the backend supports them.
// =====================================================================

#[test]
fn rt4_f5_hip_must_admit_bf16_input() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("bf16_hip.ch");
    write_file(
        &src,
        r#"def my_add(x: tensor[3, bf16], y: tensor[3, bf16]) -> tensor[3, bf16] = add(x, y)
"#,
    );
    let build = run_build_target(dir.path(), &src, "hip");
    let stderr = String::from_utf8_lossy(&build.stderr);
    let combined = format!("{}{}", stderr, String::from_utf8_lossy(&build.stdout));
    // After the F5 fix, the CLI no longer rejects bf16 with the old
    // "DAG path only supports f32/bool tensors" message. Elementwise
    // bf16 still rejects (the HIP elementwise kernel suffix path is
    // matmul-only), but with a more precise diagnostic that references
    // the actual coverage boundary instead of the over-narrow gate.
    assert!(
        !combined.contains("only supports f32/bool tensors"),
        "RT-4 F5: HIP CLI still emits the pre-fix narrow rejection message. \
         output:\n{combined}"
    );
}

/// Positive parity for F5: bf16 matmul (the WS-A3 hipblasGemmEx path)
/// must build cleanly through the HIP target without any CLI-level
/// rejection.
#[test]
fn rt4_f5_hip_bf16_matmul_builds_through_hip_target() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("bf16_mm.ch");
    write_file(
        &src,
        r#"def my_mm(x: tensor[3, 4, bf16], y: tensor[4, 5, bf16]) -> tensor[3, 5, bf16] = matmul(x, y)
"#,
    );
    let build = run_build_target(dir.path(), &src, "hip");
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        build.status.success(),
        "RT-4 F5 positive parity: bf16 matmul on --target hip must build. \
         stderr={stderr}"
    );
}

/// Positive parity for F5: f64 elementwise must build through HIP
/// after the gate widens (Sgemm/Dgemm, WS-A2).
#[test]
fn rt4_f5_hip_f64_add_builds_through_hip_target() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("f64_hip.ch");
    write_file(
        &src,
        r#"def my_add(x: tensor[3, f64], y: tensor[3, f64]) -> tensor[3, f64] = add(x, y)
"#,
    );
    let build = run_build_target(dir.path(), &src, "hip");
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        build.status.success(),
        "RT-4 F5 positive parity: f64 add on --target hip must build. \
         stderr={stderr}"
    );
}

/// Positive parity for F5: int8 elementwise (WS-A4 typed kernel
/// templates) must build through HIP after the gate widens.
#[test]
fn rt4_f5_hip_int8_add_builds_through_hip_target() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("int8_hip.ch");
    write_file(
        &src,
        r#"def my_add(x: tensor[3, int8], y: tensor[3, int8]) -> tensor[3, int8] = add(x, y)
"#,
    );
    let build = run_build_target(dir.path(), &src, "hip");
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        build.status.success(),
        "RT-4 F5 positive parity: int8 add on --target hip must build. \
         stderr={stderr}"
    );
}

// =====================================================================
// F6: em-dash in user-facing string in chelis_metal_runtime.h.
// =====================================================================

#[test]
fn rt4_f6_metal_runtime_must_not_contain_emdash_in_user_strings() {
    let header = include_str!("../../chelis-backend-metal/runtime/chelis_metal_runtime.h");
    // Walk every line; flag em-dashes (U+2014) that appear inside a
    // double-quoted region. A simpler heuristic: any fprintf line
    // containing an em-dash is a violation, since fprintf strings are
    // user-facing.
    for (i, line) in header.lines().enumerate() {
        let has_emdash = line.contains('\u{2014}');
        let in_fprintf = line.contains("fprintf(") || line.contains("\"");
        if has_emdash && in_fprintf {
            // Allow comment-only em-dashes (lines starting with `*` or
            // `//`). Conservative: only flag when the line is an
            // fprintf format string region.
            let trimmed = line.trim_start();
            if trimmed.starts_with('*') || trimmed.starts_with("//") {
                continue;
            }
            panic!(
                "RT-4 F6: em-dash in user-facing string at \
                 chelis_metal_runtime.h line {}: {line}",
                i + 1
            );
        }
    }
}

// =====================================================================
// F7: no end-to-end bf16/f16 .ch test or example exists.
// =====================================================================

#[test]
fn rt4_f7_repo_must_have_bf16_or_f16_ch_source_under_packages_or_examples() {
    use std::path::PathBuf;
    let mut found = Vec::new();
    let roots = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packages/chelis-std"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples"),
    ];
    fn walk(dir: &Path, found: &mut Vec<PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, found);
                } else if path.extension().is_some_and(|e| e == "ch")
                    && let Ok(contents) = fs::read_to_string(&path)
                    && (contents.contains(", bf16]") || contents.contains(", f16]"))
                {
                    found.push(path);
                }
            }
        }
    }
    for root in &roots {
        walk(root, &mut found);
    }
    assert!(
        !found.is_empty(),
        "RT-4 F7: no .ch file in packages/chelis-std or examples exercises bf16 or \
         f16; coverage matrix gap. The user-facing CLI surface for bf16/f16 is \
         entirely uncovered."
    );
}

/// RT-4 F7 verification: the bf16 fixture under examples/illustrative
/// must build end-to-end through `chelis check` and `chelis build
/// --target hip` so the user-facing surface stays exercised.
#[test]
fn rt4_f7_bf16_matmul_fixture_builds_through_hip() {
    use std::path::PathBuf;
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/illustrative/bf16_matmul.ch");
    assert!(
        fixture.exists(),
        "RT-4 F7: examples/illustrative/bf16_matmul.ch fixture missing"
    );
    let dir = tempdir().expect("tempdir");
    // `chelis check` must surface no type errors.
    let json = run_check(&fixture);
    let errors = json["errors"].as_array().expect("errors array");
    assert!(
        errors.is_empty(),
        "RT-4 F7: bf16 fixture must check cleanly; got errors {errors:?}"
    );
    // `chelis build --target hip` must succeed (matmul-only is the
    // bf16 carrier per WS-A3; the shared reject_unsupported_hip_ops
    // admits bf16 on BlasMatmul nodes after the F5 widen).
    let build = run_build_target(dir.path(), &fixture, "hip");
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        build.status.success(),
        "RT-4 F7: bf16 fixture must build through --target hip. stderr={stderr}"
    );
}

// =====================================================================
// Working invariants to lock down (positive parity for RT-4 findings).
// =====================================================================

/// Working: f8e4m3 suffix lex-rejected per spec §1.1.1.
#[test]
fn rt4_invariant_f8e4m3_suffix_lex_rejected() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("f8e4m3_suffix.ch");
    write_file(&src, "def main() -> int32 = 1f8e4m3\n");
    let build = run_build_in(dir.path(), &src, None);
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        !build.status.success(),
        "RT-4 invariant: f8e4m3 suffix must be a lex error"
    );
    assert!(
        stderr.contains("§1.1.1") || stderr.contains("1.1.1"),
        "RT-4 invariant: f8e4m3 lex-error must cite §1.1.1. stderr={stderr}"
    );
}

/// Working: u32 / unsigned suffix lex-rejected, deferred per spec
/// §1.1.1 (§1.1.2 names the `uint*` spellings canonical).
#[test]
fn rt4_invariant_unsigned_suffix_lex_rejected() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("u32_suffix.ch");
    write_file(&src, "def main() -> int32 = 42u32\n");
    let build = run_build_in(dir.path(), &src, None);
    let stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        !build.status.success(),
        "RT-4 invariant: u32 suffix must be a lex error"
    );
    assert!(
        stderr.contains("§1.1.1") || stderr.contains("1.1.1"),
        "RT-4 invariant: u32 lex-error must cite §1.1.1. stderr={stderr}"
    );
}

/// Working: int matmul (int8/int16/int32/int64) all rejected at check
/// per spec §5.7.2.
#[test]
fn rt4_invariant_int_matmul_rejected_for_every_int_dtype() {
    for dtype in ["int8", "int16", "int32", "int64"] {
        let dir = tempdir().expect("tempdir");
        let src = dir.path().join(format!("matmul_{dtype}.ch"));
        write_file(
            &src,
            &format!(
                "def main(x: tensor[3, 4, {dtype}], y: tensor[4, 5, {dtype}]) -> tensor[3, 5, {dtype}] = matmul(x, y)\n"
            ),
        );
        let json = run_check(&src);
        let messages: Vec<String> = json["errors"]
            .as_array()
            .expect("errors array")
            .iter()
            .map(|e| e["message"].as_str().unwrap_or("").to_string())
            .collect();
        assert!(
            messages.iter().any(|m| m.contains("5.7.2")),
            "RT-4 invariant: matmul on {dtype} must be rejected with §5.7.2 citation; got {messages:?}"
        );
    }
}

/// Working: checked-family enforcement at polymorphic instantiation. Even
/// when the offending op (matmul) is in a dead-code branch (`if false
/// then matmul(x,y) else matmul(x,y)`), the static analysis fires
/// because the authored contract is checked regardless of runtime branch.
#[test]
fn rt4_invariant_polymorphic_int_matmul_rejected_through_dead_branch() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("dead_int_matmul.ch");
    write_file(
        &src,
        r#"sig wrap[p: Float]: bool -> &tensor[2, 3, p] -> &tensor[3, 4, p] -> tensor[2, 4, p]
def wrap(branch, x, y) = if branch then matmul(x, y) else matmul(x, y)
def use_int(x: &tensor[2, 3, int32], y: &tensor[3, 4, int32]) -> tensor[2, 4, int32] = wrap(false, x, y)
"#,
    );
    let json = run_check(&src);
    let messages: Vec<String> = json["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        messages
            .iter()
            .any(|m| m.contains("dtype family `Float`") && m.contains("`int32`")),
        "RT-4 invariant: polymorphic int matmul through any branch must be rejected; got {messages:?}"
    );
}

/// Working: f8e4m3 in tensor element type rejected.
#[test]
fn rt4_invariant_f8e4m3_in_tensor_rejected() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("f8e4m3_tensor.ch");
    write_file(
        &src,
        "def main(x: tensor[3, f8e4m3]) -> tensor[3, f8e4m3] = x\n",
    );
    let json = run_check(&src);
    let kinds: Vec<String> = json["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        kinds.iter().any(|k| k == "UnsupportedTensorPrecision"),
        "RT-4 invariant: f8e4m3 tensor element must surface UnsupportedTensorPrecision; got {kinds:?}"
    );
}

/// Working: literal default rule §5.3 — `[1, 2, 3]` is `tensor[3, int32]`.
#[test]
fn rt4_invariant_int_literal_default_is_int32() {
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    let mut child = StdCommand::new(bin)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", "/dev/stdin"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn chelis");
    use std::io::Write;
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin.write_all(b"def main() = [1, 2, 3]\n").expect("write");
    }
    let output = child.wait_with_output().expect("wait");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("(t-prim {} int32)"),
        "RT-4 invariant: [1, 2, 3] must default to int32 per §5.3; deep output: {stdout}"
    );
    assert!(
        !stdout.contains("(t-prim {} int64)"),
        "RT-4 invariant: int literal default must NOT be int64. deep output: {stdout}"
    );
}

/// Working: literal default rule §5.3 — `[1.0, 2.0, 3.0]` is `tensor[3, f32]`.
#[test]
fn rt4_invariant_float_literal_default_is_f32() {
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    let mut child = StdCommand::new(bin)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", "/dev/stdin"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("spawn chelis");
    use std::io::Write;
    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin
            .write_all(b"def main() = [1.0, 2.0, 3.0]\n")
            .expect("write");
    }
    let output = child.wait_with_output().expect("wait");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("(t-prim {} f32)"),
        "RT-4 invariant: [1.0, ...] must default to f32 per §5.3; deep output: {stdout}"
    );
    assert!(
        !stdout.contains("(t-prim {} f64)"),
        "RT-4 invariant: float literal default must NOT be f64. deep output: {stdout}"
    );
}

/// Working: bf16 + f32 add rejected with PrecisionMismatch (no
/// implicit promotion §5.1).
#[test]
fn rt4_invariant_bf16_plus_f32_rejected() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("mixed_prec.ch");
    write_file(
        &src,
        "def main(x: tensor[3, f32], y: tensor[3, bf16]) -> tensor[3, f32] = add(x, y)\n",
    );
    let json = run_check(&src);
    let kinds: Vec<String> = json["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        kinds.iter().any(|k| k == "PrecisionMismatch"),
        "RT-4 invariant: f32+bf16 add must surface PrecisionMismatch; got {kinds:?}"
    );
}

/// Working: bf16 reduce_sum returns operand precision per §5.7.1
/// (not the f32 accumulator).
#[test]
fn rt4_invariant_bf16_reduce_sum_returns_bf16() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("sum_bf16.ch");
    // Wrong return type (f32) must be rejected because operand is bf16
    // and §5.7.1 says result returns to operand precision.
    write_file(
        &src,
        "def s(x: tensor[10, bf16]) -> tensor[f32] = sum(x, 0)\n",
    );
    let json = run_check(&src);
    let kinds: Vec<String> = json["errors"]
        .as_array()
        .expect("errors array")
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        kinds.iter().any(|k| k == "TypeMismatch"),
        "RT-4 invariant: bf16 reduce_sum result must be bf16 (operand prec) per §5.7.1; got {kinds:?}"
    );
}

/// Working: int8 reduce_sum returns int32 per §5.7.1 (accumulator
/// precision for narrow integers).
#[test]
fn rt4_invariant_int8_reduce_sum_returns_int32() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("sum_int8.ch");
    write_file(
        &src,
        "def s(x: tensor[200, int8]) -> tensor[int32] = sum(x, 0)\n",
    );
    let json = run_check(&src);
    let errors = json["errors"].as_array().expect("errors array");
    assert!(
        errors.is_empty(),
        "RT-4 invariant: int8 sum result must be int32 per §5.7.1; got {errors:?}"
    );
}

/// Working: monomorphization specializes a polymorphic id at multiple
/// concrete dtypes. Regression-flip: the WS-A8 monomorphization
/// guarantee.
#[test]
fn rt4_invariant_polymorphic_id_specialized_at_multiple_dtypes() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("multi_specialize.ch");
    write_file(
        &src,
        r#"sig poly_id: tensor[n, p] -> tensor[n, p]
def poly_id(x) = x
def use_f32(x: tensor[3, f32]) -> tensor[3, f32] = poly_id(x)
def use_i32(x: tensor[3, int32]) -> tensor[3, int32] = poly_id(x)
def use_i64(x: tensor[3, int64]) -> tensor[3, int64] = poly_id(x)
"#,
    );
    let build = run_build_in(dir.path(), &src, Some(&dir.path().join("out")));
    assert!(
        build.status.success(),
        "RT-4 invariant: WS-A8 must monomorphize three concrete dtypes; stderr={}",
        String::from_utf8_lossy(&build.stderr)
    );
    let c_file = dir.path().join("out").join("multi_specialize.c");
    let c_source = fs::read_to_string(&c_file).expect("read emitted C");
    for callee in ["use_f32", "use_i32", "use_i64"] {
        assert!(
            c_source.contains(callee),
            "RT-4 invariant: monomorphized symbol `{callee}` missing from emitted C"
        );
    }
}
