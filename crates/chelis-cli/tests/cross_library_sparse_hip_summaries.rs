//! Wave 6 / W6 Task B — HIP sparse-helper wrapper lock-tests.
//!
//! Sibling of `cross_library_sparse_summaries.rs` (which locks the
//! C-backend consumption of sparse helper summaries). This file locks
//! the analogous invariant on the HIP backend: user-`def` wrappers
//! around sparse builtins (`gather`, `scatter_replace`) are inlined
//! into the HIP entry DAG via `lower_named_tensor_entry_dag` before
//! HIP codegen, so the wrapper-form HIP source emits the same
//! per-op kernel name (`kernel_gather_i64`, `kernel_scatter_replace_i64`,
//! etc.) that a direct sparse-builtin call emits.
//!
//! ## Pinned framing — tests-only
//!
//! Per the W6 plan: HIP has zero references to `HostProgram` /
//! `HostFunction` / `HostTensorHelper`. `codegen_hip` only consumes a
//! `Dag`. `crates/chelis-compiler-api/src/compiler.rs:195-230`
//! selects HIP's entry DAG via `lower_named_tensor_entry_dag`, which
//! calls into `lower_plain_callable_app` and inlines user-`def`
//! bodies into the entry DAG before HIP codegen. So a wrapper-form
//! sparse helper produces the same in-line `RiscOp::Gather` /
//! `RiscOp::Scatter` in the entry DAG that a direct call would, and
//! HIP's emit dispatch picks the same kernel name.
//!
//! This file's tests assert that contract structurally. If any test
//! fails on first run, the W6 escalation trigger fires — do NOT fix
//! HIP codegen mid-Task-B; file the failing-test commit and surface
//! the gap.
//!
//! ## Coverage
//!
//! For each sparse op driveable through the Surf user-`def` surface
//! (`gather`, `scatter_replace`), two wrapper shapes:
//!
//!   1. single-wrapper: `def my_op(...) = sparse_builtin(...); def f(...) = my_op(...)`
//!   2. nested-wrapper: `def my_op(...) = sparse_builtin(...); def wrap(...) = my_op(...); def f(...) = wrap(...)`
//!
//! `scatter_add` has no Surf surface form (only the AD adjoint of
//! `gather` produces `RiscOp::ScatterAdd`). The host-lane
//! `scatter(..., "add")` pentaop is the generic dispatch path; we
//! still lock that the pentaop wrapper does NOT register a tensor-
//! lane `ScatterAdd` summary or emit `kernel_scatter_add_*` (the
//! generic runtime call is the contract).

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

/// Build a Chelis source via the CLI to a tempdir targeting `--target
/// hip` and return the generated `<name>_hip.cpp` contents. Mirrors
/// the idiom in `crates/chelis-cli/tests/cli.rs`'s
/// `build_hip_emits_sparse_gather_kernel`.
fn build_to_hip(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}_out"));
    fs::write(&src_path, source).expect("write source");

    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("chelis build should run");
    assert!(
        status.success(),
        "chelis build --target hip failed for {name}"
    );
    let cpp_path = out_dir.join(format!("{name}_hip.cpp"));
    fs::read_to_string(&cpp_path).expect("read generated hip source")
}

// =========================================================================
// Gather — single-wrapper and nested-wrapper forms
// =========================================================================

#[test]
fn hip_gather_single_user_def_wrapper_emits_kernel_gather_i64() {
    // Direct sparse-builtin call inside a user-`def` wrapper. HIP
    // codegen inlines `my_g`'s body into the entry DAG before
    // emitting, so the resulting HIP source MUST contain the same
    // `kernel_gather_i64` dispatch name a direct call would emit.
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = my_g(table, indices)\n";
    let hip = build_to_hip(source, "hip_gather_single_wrapper");
    assert!(
        hip.contains("kernel_gather_i64"),
        "single-wrapper gather MUST emit `kernel_gather_i64` (locks \
         wrapper-inlining-via-lower_named_tensor_entry_dag invariant). \
         HIP source:\n{hip}",
    );
    // Negative companion: must NOT emit the i32 variant or the
    // invalid-precision fallback.
    assert!(
        !hip.contains("kernel_gather_i32"),
        "single-wrapper gather (i64 indices) must NOT emit `kernel_gather_i32`",
    );
    assert!(
        !hip.contains("kernel_gather_invalid"),
        "single-wrapper gather must NOT emit `kernel_gather_invalid`",
    );
}

#[test]
fn hip_gather_single_user_def_wrapper_int32_indices_emits_kernel_gather_i32() {
    // Sibling sweep: i32 indices through a user-`def` wrapper must
    // dispatch to `kernel_gather_i32`. Locks the precision-aware
    // kernel-name dispatch through the inlining path.
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, i32]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, i32]) \
                  -> tensor[64, 128, f32] = my_g(table, indices)\n";
    let hip = build_to_hip(source, "hip_gather_int32_wrapper");
    assert!(
        hip.contains("kernel_gather_i32"),
        "single-wrapper gather with i32 indices MUST emit `kernel_gather_i32`. \
         HIP source:\n{hip}",
    );
    assert!(
        !hip.contains("kernel_gather_i64"),
        "single-wrapper gather (i32 indices) must NOT emit `kernel_gather_i64`",
    );
}

#[test]
fn hip_gather_nested_user_def_wrapper_emits_kernel_gather_i64() {
    // Two-level nested wrapper: `def my_g(...) = gather(...);
    // def wrap_g(...) = my_g(...); def f(...) = wrap_g(...)`. The
    // recursive inlining in `lower_plain_callable_app` should
    // produce the same in-line `RiscOp::Gather` in the entry DAG.
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def wrap_g(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = my_g(table, indices)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = wrap_g(table, indices)\n";
    let hip = build_to_hip(source, "hip_gather_nested_wrapper");
    assert!(
        hip.contains("kernel_gather_i64"),
        "nested wrapper gather MUST emit `kernel_gather_i64` (locks recursive \
         inlining through lower_plain_callable_app). HIP source:\n{hip}",
    );
}

// =========================================================================
// ScatterReplace — single-wrapper and nested-wrapper forms
// =========================================================================

#[test]
fn hip_scatter_replace_single_user_def_wrapper_emits_kernel_scatter_replace_i32() {
    // Direct `scatter_replace` builtin inside a user-`def`. The
    // wrapper inline should produce `RiscOp::Scatter` in the entry
    // DAG and HIP emits `kernel_scatter_replace_i32` (per
    // `emit.rs:714-721`: i32 indices route).
    let source = "def my_sr(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  scatter_replace(table, indices, updates, 0)\n\
                  def f(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  my_sr(table, indices, updates)\n";
    let hip = build_to_hip(source, "hip_scatter_replace_single_wrapper");
    assert!(
        hip.contains("kernel_scatter_replace_i32"),
        "single-wrapper scatter_replace MUST emit `kernel_scatter_replace_i32`. \
         HIP source:\n{hip}",
    );
    assert!(
        !hip.contains("kernel_scatter_replace_i64"),
        "single-wrapper scatter_replace (i32 indices) must NOT emit \
         `kernel_scatter_replace_i64`",
    );
    assert!(
        !hip.contains("kernel_scatter_replace_invalid"),
        "single-wrapper scatter_replace must NOT emit `kernel_scatter_replace_invalid`",
    );
    // Negative companion: scatter_replace must NOT emit the
    // `kernel_scatter_add_*` family (we are NOT in the
    // accumulating path).
    assert!(
        !hip.contains("kernel_scatter_add_i32"),
        "scatter_replace must NOT emit `kernel_scatter_add_i32`",
    );
    assert!(
        !hip.contains("kernel_scatter_add_i64"),
        "scatter_replace must NOT emit `kernel_scatter_add_i64`",
    );
}

#[test]
fn hip_scatter_replace_single_user_def_wrapper_int64_indices_emits_kernel_scatter_replace_i64() {
    // Sibling sweep: i64 indices through a user-`def` wrapper.
    let source = "def my_sr(table: tensor[3, 2, f32], indices: tensor[4, i64], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  scatter_replace(table, indices, updates, 0)\n\
                  def f(table: tensor[3, 2, f32], indices: tensor[4, i64], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  my_sr(table, indices, updates)\n";
    let hip = build_to_hip(source, "hip_scatter_replace_int64_wrapper");
    assert!(
        hip.contains("kernel_scatter_replace_i64"),
        "single-wrapper scatter_replace with i64 indices MUST emit \
         `kernel_scatter_replace_i64`. HIP source:\n{hip}",
    );
    assert!(
        !hip.contains("kernel_scatter_replace_i32"),
        "single-wrapper scatter_replace (i64 indices) must NOT emit \
         `kernel_scatter_replace_i32`",
    );
}

#[test]
fn hip_scatter_replace_nested_user_def_wrapper_emits_kernel_scatter_replace_i32() {
    // Two-level nested wrapper for scatter_replace.
    let source = "def my_sr(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  scatter_replace(table, indices, updates, 0)\n\
                  def wrap_sr(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  my_sr(table, indices, updates)\n\
                  def f(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  wrap_sr(table, indices, updates)\n";
    let hip = build_to_hip(source, "hip_scatter_replace_nested_wrapper");
    assert!(
        hip.contains("kernel_scatter_replace_i32"),
        "nested wrapper scatter_replace MUST emit `kernel_scatter_replace_i32` \
         (locks recursive inlining). HIP source:\n{hip}",
    );
}

// =========================================================================
// ScatterAdd — pentaop-only path
//
// `scatter_add` has no Surf surface form. The host-lane
// `scatter(base, ids, updates, axis, "add")` pentaop is the generic
// runtime dispatch (matches the C-backend invariant locked in
// `cross_library_sparse_summaries.rs::user_def_scatter_add_helper_emits_inline_sparse_scatter_add_loop`).
// We lock that no tensor-lane `RiscOp::ScatterAdd` kernel is emitted
// from a Surf user-`def` wrapper around the pentaop.
// =========================================================================

#[test]
fn hip_pentaop_scatter_add_wrapper_does_not_emit_kernel_scatter_add() {
    // The pentaop `scatter(..., "add")` is the host-lane generic
    // path. It must NOT register a tensor-lane summary that lowers
    // to `RiscOp::ScatterAdd` and consequently must NOT emit
    // `kernel_scatter_add_*`. Locks the invariant that the
    // host-lane pentaop stays on the generic runtime call path
    // even when wrapped through a user-def def — same contract the
    // C backend test surface asserts.
    //
    // Today, the pentaop path forces HIP to fall back to the C
    // host-program codegen route (the entry DAG ends up empty
    // because the program is host-heavy). The build still succeeds;
    // the produced source is C, not HIP. Either way, no
    // `kernel_scatter_add_*` should be present.
    let source = "def my_sa(base: tensor[10, 4, f32], bin_ids: tensor[64, i64], \
                  updates: tensor[64, 4, f32]) -> tensor[10, 4, f32] = \
                  scatter(base, bin_ids, updates, 0, \"add\")\n\
                  def f(base: tensor[10, 4, f32], bin_ids: tensor[64, i64], \
                  updates: tensor[64, 4, f32]) -> tensor[10, 4, f32] = \
                  my_sa(base, bin_ids, updates)\n";
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join("hip_scatter_add_pentaop.ch");
    let out_dir = dir.path().join("hip_scatter_add_pentaop_out");
    fs::write(&src_path, source).expect("write source");

    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("chelis build should run");
    assert!(
        status.success(),
        "chelis build --target hip must succeed for the pentaop-only program",
    );
    // The output may be HIP (`*_hip.cpp`) or the host-program C
    // fallback (per `compile_result_hip_host` route). Pick whichever
    // exists.
    let hip_path = out_dir.join("hip_scatter_add_pentaop_hip.cpp");
    let c_path = out_dir.join("hip_scatter_add_pentaop.c");
    let emitted = if hip_path.exists() {
        fs::read_to_string(&hip_path).expect("read hip cpp")
    } else if c_path.exists() {
        fs::read_to_string(&c_path).expect("read host-fallback c")
    } else {
        panic!(
            "expected either hip cpp or host-fallback c output; out_dir is {out_dir:?}",
            out_dir = out_dir,
        );
    };
    assert!(
        !emitted.contains("kernel_scatter_add_i32"),
        "pentaop scatter(..., \"add\") wrapper must NOT emit `kernel_scatter_add_i32`. \
         Source:\n{emitted}",
    );
    assert!(
        !emitted.contains("kernel_scatter_add_i64"),
        "pentaop scatter(..., \"add\") wrapper must NOT emit `kernel_scatter_add_i64`",
    );
}

// =========================================================================
// Direct (non-wrapped) baseline — sanity check
//
// Confirms the kernel-name dispatch fires for direct calls as well
// as wrapped calls. If a wrapped test above fails but this baseline
// passes, the regression is specifically in the user-def inlining
// path (not in HIP's per-op kernel dispatch).
// =========================================================================

#[test]
fn hip_gather_direct_call_baseline_emits_kernel_gather_i64() {
    let source = "def f(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n";
    let hip = build_to_hip(source, "hip_gather_direct_baseline");
    assert!(
        hip.contains("kernel_gather_i64"),
        "direct gather call MUST emit `kernel_gather_i64`. HIP source:\n{hip}",
    );
}

#[test]
fn hip_scatter_replace_direct_call_baseline_emits_kernel_scatter_replace_i32() {
    let source = "def f(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  scatter_replace(table, indices, updates, 0)\n";
    let hip = build_to_hip(source, "hip_scatter_replace_direct_baseline");
    assert!(
        hip.contains("kernel_scatter_replace_i32"),
        "direct scatter_replace call MUST emit `kernel_scatter_replace_i32`. \
         HIP source:\n{hip}",
    );
}
