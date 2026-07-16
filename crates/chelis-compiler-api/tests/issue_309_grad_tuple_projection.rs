//! Issue #309 — C-backend grad-tuple projection emits a correctly-typed
//! receiver.
//!
//! A multi-`wrt` `grad` differentiates a scalar loss w.r.t. several
//! tensor parameters, so its result is a TUPLE of gradient tensors (one
//! per param), exactly as the IR/eval lane produces via
//! `LoweredValue::Tuple`. When such a result is projected with a tuple
//! index — `(grad(loss)(x, w)).0` — the host-lane C backend previously:
//!
//!   1. typed the grad-helper call as a single `chelis_tensor*`, then
//!      emitted `chelis_tuple_get(t, i)` over that tensor receiver
//!      (`chelis_tuple_get` expects a `chelis_tuple*`) — a mistyped
//!      receiver that crashes at runtime; and
//!   2. sized the helper output array to one slot and passed `n_out = 1`
//!      even though the emitted grad helper writes two outputs and its
//!      wrapper asserts `n_out == 2` — an arity-guard `abort()`.
//!
//! The fix types a multi-root tensor-helper call as a `Tuple` of the
//! per-root tensor types (root order), sizes the output array and
//! `n_out` from the helper's actual root count, and assembles a real
//! `chelis_tuple` from the boxed output tensors so the projection's
//! receiver is correctly typed.
//!
//! These tests pin the emitted-C structural invariants. Numerical
//! evaluator-vs-backend parity for `grad` is covered by the autodiff
//! parity suites; here we lock the shape of the host-lane emission.

use chelis_compiler_api::compiler::compile;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};

fn compile_c(source: &str, entry: &str) -> String {
    let result = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        target: CompileTarget::C,
        entry_name: Some(entry.to_string()),
    })
    .unwrap_or_else(|err| panic!("compile failed: {err:?}"));
    result
        .files
        .into_iter()
        .find(|file| file.path == format!("{entry}.c"))
        .unwrap_or_else(|| panic!("missing {entry}.c"))
        .contents
}

const MULTI_WRT_SOURCE: &str = r"module Repro.GradTupleProj
def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 =
  tensor_to_scalar(sum(mul(x, w), cast(0, int32)))
def dloss(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] = (grad(loss)(x, w)).0
";

/// Positive: the projection's `chelis_tuple_get` receiver must be a real
/// `chelis_tuple*` assembled from the helper outputs, NOT the raw
/// `chelis_tensor*` output slot.
#[test]
fn issue309_multi_wrt_grad_projection_reads_from_real_tuple() {
    let c = compile_c(MULTI_WRT_SOURCE, "repro");

    // A `chelis_tuple` is built from the helper's boxed output tensors.
    assert!(
        c.contains("chelis_tuple_from_values"),
        "multi-wrt grad projection must assemble a real tuple from helper \
         outputs, got:\n{c}"
    );
    assert!(
        c.contains("chelis_value_from_tensor"),
        "each grad-helper output tensor must be boxed into a chelis_value \
         before tuple assembly, got:\n{c}"
    );

    // The tuple-get receiver must be a chelis_value built from the tuple,
    // not the bare tensor output slot. We assert the projection reads via
    // chelis_tuple_get and that the result is unboxed back to a tensor.
    assert!(
        c.contains("chelis_tuple_get"),
        "the `.0` projection must still go through chelis_tuple_get, got:\n{c}"
    );
    assert!(
        c.contains("chelis_value_as_tensor"),
        "the projected tuple element must be unboxed to a tensor, got:\n{c}"
    );
}

/// Negative-parity: the grad helper produces two outputs, so its call
/// site must size the output array to two slots and pass `n_out = 2`.
/// The pre-fix bug hard-coded a one-slot array and `n_out = 1`, which
/// tripped the helper's `n_out == 2` arity guard.
#[test]
fn issue309_multi_wrt_grad_call_sizes_two_output_slots() {
    let c = compile_c(MULTI_WRT_SOURCE, "repro");

    // Two output slots, two-output call. (Whitespace-tolerant: assert the
    // `[2]` output array and a call ending in `, 2);`.)
    let has_two_slot_array = c.contains("chelis_tensor *__outputs")
        && c.lines()
            .any(|line| line.contains("__outputs") && line.contains("[2]"));
    assert!(
        has_two_slot_array,
        "multi-wrt grad helper call must size its output array to two \
         slots, got:\n{c}"
    );

    let calls_helper_with_two_outputs = c
        .lines()
        .any(|line| line.contains("__tensor_") && line.trim_end().ends_with(", 2);"));
    assert!(
        calls_helper_with_two_outputs,
        "multi-wrt grad helper must be called with n_out = 2, got:\n{c}"
    );

    // Regression guard: the projection must NOT pass a bare tensor output
    // slot directly into chelis_tuple_get (the pre-fix mistyped receiver).
    assert!(
        !c.contains("chelis_tuple_get(__outputs"),
        "chelis_tuple_get must never receive a raw helper output slot \
         (a chelis_tensor*), got:\n{c}"
    );
}

/// Control: a single-output grad whose body keeps it on the host lane
/// must still emit a plain single-tensor helper call (one output slot,
/// `n_out = 1`) and must NOT wrap the result in a tuple. This guards the
/// fix against over-firing on the common single-`wrt` case.
#[test]
fn issue309_single_wrt_grad_call_stays_single_tensor() {
    // `wrt=x` selects only `x`, so the grad result is a single tensor;
    // the function returns it directly (no projection).
    let source = r"module Repro.GradSingle
def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 =
  tensor_to_scalar(sum(mul(x, w), cast(0, int32)))
def dloss(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] = grad(loss, wrt=x)(x, w)
";
    let c = compile_c(source, "single");

    let calls_helper_with_one_output = c
        .lines()
        .any(|line| line.contains("__tensor_") && line.trim_end().ends_with(", 1);"));
    assert!(
        calls_helper_with_one_output,
        "single-wrt grad helper must be called with n_out = 1, got:\n{c}"
    );
    assert!(
        !c.contains("chelis_tuple_from_values"),
        "single-wrt grad must NOT assemble a tuple, got:\n{c}"
    );
}
