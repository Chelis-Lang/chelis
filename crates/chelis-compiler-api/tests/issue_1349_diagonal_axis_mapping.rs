//! chelis#1349: the host evaluator's `diagonal`/`trace` read the diagonal
//! coordinate at the SOURCE axis number instead of the diagonal's retained
//! OUTPUT slot, and the reconstruction walk never consumed that slot.
//! Exactly one ordered axis pair per rank was correct, `(rank-2, rank-1)`,
//! and every checked-in case used it; elsewhere the evaluator returned
//! wrong values, panicked on reversed pairs with `axis1 == rank - 1`, or
//! (`trace`) produced a shape the checker never declared, since
//! `infer_trace_result_type` removes both source axes while the runtime
//! reduced `min(axis1, axis2)` of the diagonal.
//!
//! These controls assert values derived independently from [05-OP-33]
//! (`spec/05-risc-primitives.md`): the output keeps source axis order with
//! the second axis removed, replaces the retained first axis extent with
//! the smaller selected extent, and reads equal coordinates on both axes.
//! `trace` is that diagonal reduced over the diagonal's own output axis,
//! removing both source axes. They deliberately never compare the two
//! execution lanes to each other: both lanes carried byte-identical copies
//! of this defect, so cross-lane parity stayed green throughout
//! (chelis#1351).

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{
    EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};

fn eval_surf(source: &str) -> EvalResult {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|err| panic!("eval failed for source:\n{source}\nerror: {err:?}"))
}

fn eval_is_err(source: &str) -> bool {
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .is_err()
}

fn root_tensor<'a>(result: &'a EvalResult, name: &str) -> &'a TensorValue {
    let root = result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("missing root {name} in {:?}", result.roots));
    match &root.value {
        ExecutionValue::Tensor { value } => value,
        other => panic!("expected tensor for {name}, got {other:?}"),
    }
}

/// Nested `to_tensor` literal for the ramp: element at row-major position
/// `flat` holds `flat` as an f32 literal, so every element is
/// distinguishable and each trace sum is an exact small integer.
fn ramp_literal(shape: &[usize]) -> String {
    fn nest(shape: &[usize], base: usize, stride: usize) -> String {
        match shape {
            [] => format!("{}.0", base),
            [extent, rest @ ..] => {
                let inner_stride = stride / extent;
                let items: Vec<String> = (0..*extent)
                    .map(|i| nest(rest, base + i * inner_stride, inner_stride))
                    .collect();
                format!("[{}]", items.join(", "))
            }
        }
    }
    let numel: usize = shape.iter().product();
    nest(shape, 0, numel)
}

/// Row-major coordinates of `linear` within `shape`.
fn unflatten(linear: usize, shape: &[usize]) -> Vec<usize> {
    let mut rem = linear;
    let mut coords = vec![0usize; shape.len()];
    for index in (0..shape.len()).rev() {
        coords[index] = rem % shape[index];
        rem /= shape[index];
    }
    coords
}

/// Independent [05-OP-33] reference: expected output shape and, for the
/// ramp input, the expected stored elements. Written directly from the
/// spec text (output position of source axis `i != axis2` is `i` minus one
/// when `i > axis2`), not from either implementation's cursor walk.
fn reference_diagonal(shape: &[usize], axis1: usize, axis2: usize) -> (Vec<usize>, Vec<f64>) {
    let diag = shape[axis1].min(shape[axis2]);
    let out_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis2)
        .map(|(index, &extent)| if index == axis1 { diag } else { extent })
        .collect();
    let out_pos = |index: usize| index - usize::from(index > axis2);
    let numel: usize = out_shape.iter().product();
    let mut values = Vec::with_capacity(numel);
    for linear in 0..numel {
        let coords = unflatten(linear, &out_shape);
        let diag_coord = coords[out_pos(axis1)];
        let mut flat = 0usize;
        for (index, &extent) in shape.iter().enumerate() {
            let coord = if index == axis1 || index == axis2 {
                diag_coord
            } else {
                coords[out_pos(index)]
            };
            flat = flat * extent + coord;
        }
        values.push(flat as f64);
    }
    (out_shape, values)
}

/// Independent trace reference: the [05-OP-33] diagonal reduced over the
/// diagonal's own output axis, removing both source axes. Exact for the
/// ramp because every partial sum is a small integer.
fn reference_trace(shape: &[usize], axis1: usize, axis2: usize) -> (Vec<usize>, Vec<f64>) {
    let (diag_shape, diag_values) = reference_diagonal(shape, axis1, axis2);
    let diag_axis = if axis2 < axis1 { axis1 - 1 } else { axis1 };
    let out_shape: Vec<usize> = diag_shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != diag_axis)
        .map(|(_, &extent)| extent)
        .collect();
    let numel: usize = out_shape.iter().product();
    let mut values = vec![0.0f64; numel];
    for (linear, value) in diag_values.iter().enumerate() {
        let coords = unflatten(linear, &diag_shape);
        let mut flat = 0usize;
        for (index, &extent) in diag_shape.iter().enumerate() {
            if index != diag_axis {
                flat = flat * extent + coords[index];
            }
        }
        values[flat] += value;
    }
    (out_shape, values)
}

/// Full ordered-pair sweep of one shape for one operation through the host
/// evaluator, collecting every mismatch so a failure reports the complete
/// wrong-pair set at once.
///
/// No zero-extent shape appears here: on this branch the host lane's
/// `tensor_numel` still floors at one element (chelis#1347, fixed
/// separately), so a zero-extent output divides by zero in
/// `linear_to_indices` independently of the chelis#1349 mapping repair.
/// The C-runtime control file carries the zero-extent case.
fn sweep(shape: &[usize], op: &str) {
    let literal = ramp_literal(shape);
    let mut failures = Vec::new();
    for axis1 in 0..shape.len() {
        for axis2 in 0..shape.len() {
            if axis1 == axis2 {
                continue;
            }
            let (want_shape, want_values) = if op == "trace" {
                reference_trace(shape, axis1, axis2)
            } else {
                reference_diagonal(shape, axis1, axis2)
            };
            let source = format!("x = to_tensor({literal})\nout = {op}(x, {axis1}, {axis2})\n");
            let result = eval_surf(&source);
            let out = root_tensor(&result, "out");
            let got_values = out.data.to_f64_lossy_vec();
            if out.shape != want_shape || got_values != want_values {
                failures.push(format!(
                    "axes ({axis1}, {axis2}): got shape {:?} values {got_values:?}, \
                     want shape {want_shape:?} values {want_values:?}",
                    out.shape
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{op} on shape {shape:?} diverges from the [05-OP-33] reference:\n{}",
        failures.join("\n")
    );
}

// Non-uniform extents throughout: a square tensor hides a wrong shape and
// makes distinct wrong picks collide with right ones.

#[test]
fn diagonal_rank2_every_ordered_pair_matches_op33() {
    sweep(&[3, 4], "diagonal");
}

#[test]
fn diagonal_rank3_every_ordered_pair_matches_op33() {
    sweep(&[2, 3, 4], "diagonal");
}

#[test]
fn diagonal_rank4_every_ordered_pair_matches_op33() {
    // Includes the non-adjacent reversed pair (2, 0) on distinct extents,
    // the smallest configuration where the wrong-shape defect is
    // observable in isolation.
    sweep(&[2, 3, 2, 5], "diagonal");
}

#[test]
fn trace_rank2_every_ordered_pair_matches_op33() {
    sweep(&[3, 4], "trace");
}

#[test]
fn trace_rank3_every_ordered_pair_matches_op33() {
    sweep(&[2, 3, 4], "trace");
}

#[test]
fn trace_rank4_every_ordered_pair_matches_op33() {
    sweep(&[2, 3, 2, 5], "trace");
}

/// Hand-pinned anchor for the reference itself, from the chelis#1349
/// reproduction: the rank-3 `[2, 2, 2]` ramp cube. The unfixed evaluator
/// returned `[0, 0, 7, 7]` and `[7, 7]` here.
#[test]
fn diagonal_and_trace_cube_axes_0_1_match_pinned_values() {
    let literal = ramp_literal(&[2, 2, 2]);
    let diag = eval_surf(&format!(
        "x = to_tensor({literal})\nout = diagonal(x, 0, 1)\n"
    ));
    let out = root_tensor(&diag, "out");
    assert_eq!(out.shape, vec![2, 2]);
    assert_eq!(out.data.to_f64_lossy_vec(), vec![0.0, 1.0, 6.0, 7.0]);
    let trace = eval_surf(&format!("x = to_tensor({literal})\nout = trace(x, 0, 1)\n"));
    let out = root_tensor(&trace, "out");
    assert_eq!(out.shape, vec![2]);
    assert_eq!(out.data.to_f64_lossy_vec(), vec![6.0, 8.0]);
}

/// Shape witness: `trace([2, 3, 2, 5], 2, 0)` must produce the
/// checker-declared `tensor[3, 5]` (`infer_trace_result_type` removes both
/// source axes). The unfixed runtime reduced `min(axis1, axis2)` of the
/// diagonal and produced `[2, 5]`, a checked type it did not honor.
#[test]
fn trace_non_adjacent_reversed_axes_honor_the_declared_shape() {
    let literal = ramp_literal(&[2, 3, 2, 5]);
    let result = eval_surf(&format!("x = to_tensor({literal})\nout = trace(x, 2, 0)\n"));
    let out = root_tensor(&result, "out");
    let (want_shape, want_values) = reference_trace(&[2, 3, 2, 5], 2, 0);
    assert_eq!(want_shape, vec![3, 5], "reference self-check");
    assert_eq!(out.shape, want_shape);
    assert_eq!(out.data.to_f64_lossy_vec(), want_values);
}

/// Negative axes denote from-the-end positions (chelis#522) and must land
/// on the same repaired mapping: `(-1, 0)` on rank 3 is `(2, 0)`, a
/// reversed pair with `axis1 == rank - 1` that panicked on the unfixed
/// evaluator.
#[test]
fn diagonal_negative_axes_normalize_onto_the_repaired_mapping() {
    let literal = ramp_literal(&[2, 3, 4]);
    let result = eval_surf(&format!(
        "x = to_tensor({literal})\nout = diagonal(x, -1, 0)\n"
    ));
    let out = root_tensor(&result, "out");
    let (want_shape, want_values) = reference_diagonal(&[2, 3, 4], 2, 0);
    assert_eq!(out.shape, want_shape);
    assert_eq!(out.data.to_f64_lossy_vec(), want_values);
}

// ---------------------------------------------------------------------
// Deliberate fail-closed negative controls: equal or out-of-range axes
// reject loud on both the unfixed and the repaired source (the checker
// rejects them before the host runtime runs), so these are not
// mutation-sensitive; they pin that the repair does not widen the
// accepted axis domain.
// ---------------------------------------------------------------------

#[test]
fn diagonal_equal_axes_reject() {
    assert!(
        eval_is_err("x = to_tensor([[1.0, 2.0], [3.0, 4.0]])\nout = diagonal(x, 1, 1)\n"),
        "diagonal with equal axes must reject"
    );
}

#[test]
fn diagonal_out_of_range_axis_rejects() {
    assert!(
        eval_is_err("x = to_tensor([[1.0, 2.0], [3.0, 4.0]])\nout = diagonal(x, 0, 2)\n"),
        "diagonal with an out-of-range axis must reject"
    );
}

#[test]
fn trace_equal_axes_reject() {
    assert!(
        eval_is_err("x = to_tensor([[1.0, 2.0], [3.0, 4.0]])\nout = trace(x, 0, 0)\n"),
        "trace with equal axes must reject"
    );
}

#[test]
fn trace_out_of_range_axis_rejects() {
    assert!(
        eval_is_err("x = to_tensor([[1.0, 2.0], [3.0, 4.0]])\nout = trace(x, 5, 0)\n"),
        "trace with an out-of-range axis must reject"
    );
}
