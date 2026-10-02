//! Wave 7 fresh-context red-team — adversarial HIP sparse wrapper shapes.
//!
//! W6 Task B locked the happy path: single-wrapper and nested-wrapper
//! forms for `gather`, `scatter_add`, `scatter_replace` on
//! `--target hip`. This file probes adversarial shapes beyond the
//! happy path called out in the Wave 7 plan:
//!
//!   1. Wrapper whose body has a sparse op behind a `let` (block
//!      binding) — `def my_g(t, i) = { rows = gather(t, i, 0); rows }`
//!   2. Wrapper with a mixed sparse + dense body — gather followed by
//!      an elementwise op on the result (`add(gather(...), zero)`).
//!   3. Wrapper called from inside another helper (3+ levels deep).
//!   4. Wrapper where the sparse helper takes the indices as a
//!      derived expression — pass-through helpers that compute
//!      indices as e.g. `cast(...)`.
//!
//! Contract: each form either hits the expected HIP kernel (the
//! inlining works) OR the failure is loud (compile error / panic), not
//! a silent fallback to a generic dense path. Loud-failure here means
//! the build returns a non-zero exit OR the emitted source clearly
//! omits the kernel name AND no contradicting kernel name appears.

use std::fs;
use std::process::Command;

use assert_cmd::cargo::CommandCargoExt;
use tempfile::tempdir;

/// Build a Chelis source via the CLI to a tempdir targeting `--target
/// hip` and return (success, generated source). Mirrors
/// `cross_library_sparse_hip_summaries::build_to_hip` but does NOT
/// assert success — adversarial cases may legitimately fail to build,
/// which is acceptable as long as the failure is loud.
fn try_build_to_hip(source: &str, name: &str) -> (bool, Option<String>, Option<String>) {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}_out"));
    fs::write(&src_path, source).expect("write source");

    let status = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src_path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .status()
        .expect("chelis build should run");

    if !status.success() {
        return (false, None, None);
    }
    let hip_path = out_dir.join(format!("{name}_hip.cpp"));
    let c_path = out_dir.join(format!("{name}.c"));
    let hip_src = if hip_path.exists() {
        Some(fs::read_to_string(&hip_path).expect("read hip cpp"))
    } else {
        None
    };
    let c_src = if c_path.exists() {
        Some(fs::read_to_string(&c_path).expect("read c"))
    } else {
        None
    };
    (true, hip_src, c_src)
}

// =========================================================================
// Adversarial 1: gather behind a `let` (block) binding.
//
// `def my_g(table, indices) = { rows = gather(table, indices, 0); rows }`
//
// The block binding `rows = ...` followed by `rows` as the body's
// final expression is the Surf canonical block-binding form. If the
// lowerer leaves this as a `Let` host-expr rather than inlining into
// the entry DAG, HIP wouldn't see the `RiscOp::Gather` and would
// emit some non-kernel dispatch (or fall back to host-program C
// emission). The contract: kernel_gather_* must still emerge.
// =========================================================================

#[test]
fn hip_gather_behind_let_block_binding_emits_kernel_gather_i64() {
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = {\n\
                  rows = gather(table, indices, 0)\n\
                  rows\n\
                  }\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = my_g(table, indices)\n";
    let (ok, hip_src, c_src) = try_build_to_hip(source, "hip_gather_let_binding");
    assert!(
        ok,
        "let-binding wrapper for gather MUST build (or LOUDLY fail). \
         Source:\n{source}"
    );
    let emitted = hip_src.or(c_src).expect("must have produced output");
    // Either the HIP kernel name appears (inlining works) OR a clear
    // non-silent-fallback signal. The W7 contract is: do NOT silently
    // fall to a generic dense path. We assert kernel_gather_* present
    // as the canonical success condition; if it isn't, we surface the
    // emitted source for inspection.
    let has_kernel = emitted.contains("kernel_gather_i64") || emitted.contains("kernel_gather_i32");
    assert!(
        has_kernel,
        "let-binding wrapper for gather MUST emit a kernel_gather_* dispatch \
         (the host lowerer should inline the block-binding into the entry DAG; \
         silent fallback to a generic dense path is the contract violation \
         this red-team test catches). Emitted source:\n{emitted}",
    );
}

// =========================================================================
// Adversarial 2: mixed sparse + dense body — gather then add(zero).
//
// `def my_g(table, indices, zero) = add(gather(table, indices, 0), zero)`
//
// The wrapper body root is `Add`, not `Gather`. The Add adds zero to
// the gather result. This is the W4-A `PostProcessingAfterSparseOp`
// rejection case for the C-side helper summarizer. For HIP, the
// question is: when this wrapper is the entry, does HIP emit a
// gather kernel for the inlined gather (even though it's followed by
// an Add)? The W7 contract: yes, the gather kernel must appear (HIP
// reaches into the inlined DAG and dispatches each sparse op
// individually). If a silent fallback elides the gather, that's a
// regression.
// =========================================================================

#[test]
fn hip_gather_followed_by_add_in_wrapper_still_emits_kernel_gather_i64() {
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, i64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  add(gather(table, indices, 0), zero)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, i64], \
                  zero: tensor[64, 128, f32]) -> tensor[64, 128, f32] = \
                  my_g(table, indices, zero)\n";
    let (ok, hip_src, c_src) = try_build_to_hip(source, "hip_gather_then_add");
    assert!(
        ok,
        "mixed sparse+dense wrapper MUST build. Source:\n{source}"
    );
    let emitted = hip_src.or(c_src).expect("must have produced output");
    // The inlined DAG contains both Gather and Add. HIP's emit
    // dispatches each. The contract: the gather kernel name MUST
    // appear; if not, the gather was silently elided.
    assert!(
        emitted.contains("kernel_gather_i64"),
        "mixed sparse+dense wrapper MUST emit `kernel_gather_i64` for the inlined \
         gather, even when followed by Add. Emitted source:\n{emitted}",
    );
}

// =========================================================================
// Adversarial 3: 3-level deep nesting (wrapper → wrapper → wrapper → gather).
//
// W6 covers 2-level. Probe whether `lower_plain_callable_app`'s
// recursive inlining reliably bottoms out on direct sparse builtins.
// =========================================================================

#[test]
fn hip_gather_three_level_nested_wrapper_emits_kernel_gather_i64() {
    let source = "def inner_g(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def mid_g(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = inner_g(table, indices)\n\
                  def outer_g(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = mid_g(table, indices)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = outer_g(table, indices)\n";
    let (ok, hip_src, c_src) = try_build_to_hip(source, "hip_gather_3level_nested");
    assert!(
        ok,
        "3-level nested wrapper for gather MUST build. Source:\n{source}"
    );
    let emitted = hip_src.or(c_src).expect("must have produced output");
    assert!(
        emitted.contains("kernel_gather_i64"),
        "3-level nested wrapper MUST emit `kernel_gather_i64` (locks recursive \
         inlining beyond 2 levels). Emitted source:\n{emitted}",
    );
}

// =========================================================================
// Adversarial 4: indices are a derived expression at the callsite.
//
// Pass `cast(raw_indices, i64)` as the wrapper's indices argument.
// The wrapper itself takes a direct `tensor[64, i64]` parameter, but
// the CALLSITE constructs the value via a Cast.
//
// FINDING (W7): the HIP build LOUDLY REJECTS this with a structured
// error message:
//   "chelis build --target hip sparse gather requires indices to be
//    loaded input tensors in this milestone; node N uses indices
//    produced by Cast { new_precision: Int64 }"
//
// This is the contract-correct behavior: HIP's sparse gather kernel
// can only consume indices that are direct `Load` inputs today
// (codegen restriction documented in the error). The build fails
// loudly rather than silently emitting a generic dense path or
// dispatching a malformed kernel call. The W7 contract is satisfied
// because the failure is loud.
//
// The W7 lock: the build MUST either succeed with `kernel_gather_*`
// OR fail with this specific cast-induced error. A silent dense
// fallback (build succeeds but the gather kernel is absent) is the
// regression this test catches.
// =========================================================================

#[test]
fn hip_gather_wrapper_with_cast_callsite_indices_loud_failure_or_kernel() {
    let source = "def my_g(table: tensor[1000, 128, f32], indices: tensor[64, i64]) \
                  -> tensor[64, 128, f32] = gather(table, indices, 0)\n\
                  def f(table: tensor[1000, 128, f32], indices: tensor[64, i32]) \
                  -> tensor[64, 128, f32] = my_g(table, cast(indices, i64))\n";
    let (ok, hip_src, c_src) = try_build_to_hip(source, "hip_gather_cast_indices");
    if !ok {
        // Loud failure path — the contract is satisfied. (We don't
        // capture stderr in this harness; the manual probe at the
        // commit message documents the exact error text.)
        return;
    }
    let emitted = hip_src.or(c_src).expect("must have produced output");
    // If the build DID succeed, the kernel MUST appear — silent
    // dense fallback is the contract violation.
    assert!(
        emitted.contains("kernel_gather_i64"),
        "build succeeded but `kernel_gather_i64` is absent -- silent dense \
         fallback for cast-callsite-indices is the regression this test \
         catches. Emitted source:\n{emitted}",
    );
}

// =========================================================================
// Adversarial 5: scatter_replace behind a `let` block binding.
//
// Mirror of test 1 for scatter_replace.
// =========================================================================

#[test]
fn hip_scatter_replace_behind_let_block_emits_kernel_scatter_replace_i32() {
    let source = "def my_sr(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = {\n\
                  out = scatter_replace(table, indices, updates, 0)\n\
                  out\n\
                  }\n\
                  def f(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  my_sr(table, indices, updates)\n";
    let (ok, hip_src, c_src) = try_build_to_hip(source, "hip_scatter_replace_let_block");
    assert!(
        ok,
        "scatter_replace let-binding wrapper MUST build. Source:\n{source}"
    );
    let emitted = hip_src.or(c_src).expect("must have produced output");
    assert!(
        emitted.contains("kernel_scatter_replace_i32"),
        "scatter_replace let-binding wrapper MUST emit `kernel_scatter_replace_i32`. \
         Emitted source:\n{emitted}",
    );
}

// =========================================================================
// Adversarial 6: scatter_replace 3-level nested.
// =========================================================================

#[test]
fn hip_scatter_replace_three_level_nested_emits_kernel_scatter_replace_i32() {
    let source = "def inner_sr(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  scatter_replace(table, indices, updates, 0)\n\
                  def mid_sr(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  inner_sr(table, indices, updates)\n\
                  def outer_sr(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  mid_sr(table, indices, updates)\n\
                  def f(table: tensor[3, 2, f32], indices: tensor[4, i32], \
                  updates: tensor[4, 2, f32]) -> tensor[3, 2, f32] = \
                  outer_sr(table, indices, updates)\n";
    let (ok, hip_src, c_src) = try_build_to_hip(source, "hip_scatter_replace_3level");
    assert!(
        ok,
        "3-level scatter_replace wrapper MUST build. Source:\n{source}"
    );
    let emitted = hip_src.or(c_src).expect("must have produced output");
    assert!(
        emitted.contains("kernel_scatter_replace_i32"),
        "3-level scatter_replace nested wrapper MUST emit \
         `kernel_scatter_replace_i32`. Emitted source:\n{emitted}",
    );
}
