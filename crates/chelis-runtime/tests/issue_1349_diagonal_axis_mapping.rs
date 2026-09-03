//! chelis#1349: `diagonal`/`trace` read the diagonal coordinate at the
//! SOURCE axis number instead of the diagonal's retained OUTPUT slot, and
//! the reconstruction walk never consumed that slot. Exactly one ordered
//! axis pair per rank was correct, `(rank-2, rank-1)`, and every checked-in
//! case used it; elsewhere the C runtime returned out-of-bounds heap bytes
//! as ordinary data (exit 0, no diagnostic), wrong values, or a wrong
//! shape.
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

use std::ffi::c_int;
use std::process::Command;

use chelis_runtime::{
    chelis_alloc, chelis_tensor, chelis_tensor_begin_write, chelis_tensor_diagonal,
    chelis_tensor_end_write, chelis_tensor_numel, chelis_tensor_rank, chelis_tensor_read_view,
    chelis_tensor_release, chelis_tensor_shape, chelis_tensor_trace, chelis_tensor_write_view,
    CHELIS_DTYPE_F32,
};

const CHILD_CASE_ENV: &str = "CHELIS_ISSUE_1349_CHILD_CASE";

/// Ramp tensor: element at row-major position `flat` holds `flat as f32`,
/// so every element is distinguishable and each sum below is an exact
/// small integer in f32.
unsafe fn ramp_tensor_f32(shape: &[i64]) -> *mut chelis_tensor {
    unsafe {
        let tensor = chelis_alloc(shape.len() as c_int, shape.as_ptr(), CHELIS_DTYPE_F32);
        let guard = chelis_tensor_begin_write(tensor);
        let view = chelis_tensor_write_view(guard);
        let data = view.data.cast::<f32>();
        let numel: i64 = shape.iter().product();
        for flat in 0..numel {
            *data.add(flat as usize) = flat as f32;
        }
        chelis_tensor_end_write(guard);
        tensor
    }
}

/// Row-major coordinates of `linear` within `shape`.
fn unflatten(linear: i64, shape: &[i64]) -> Vec<i64> {
    let mut rem = linear;
    let mut coords = vec![0i64; shape.len()];
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
fn reference_diagonal(shape: &[i64], axis1: usize, axis2: usize) -> (Vec<i64>, Vec<f32>) {
    let diag = shape[axis1].min(shape[axis2]);
    let out_shape: Vec<i64> = shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis2)
        .map(|(index, &extent)| if index == axis1 { diag } else { extent })
        .collect();
    let out_pos = |index: usize| index - usize::from(index > axis2);
    let numel: i64 = out_shape.iter().product();
    let mut values = Vec::with_capacity(numel as usize);
    for linear in 0..numel {
        let coords = unflatten(linear, &out_shape);
        let diag_coord = coords[out_pos(axis1)];
        let mut flat = 0i64;
        for (index, &extent) in shape.iter().enumerate() {
            let coord = if index == axis1 || index == axis2 {
                diag_coord
            } else {
                coords[out_pos(index)]
            };
            flat = flat * extent + coord;
        }
        values.push(flat as f32);
    }
    (out_shape, values)
}

/// Independent trace reference: the [05-OP-33] diagonal reduced over the
/// diagonal's own output axis, removing both source axes. Exact for the
/// ramp because every partial sum is a small integer.
fn reference_trace(shape: &[i64], axis1: usize, axis2: usize) -> (Vec<i64>, Vec<f32>) {
    let (diag_shape, diag_values) = reference_diagonal(shape, axis1, axis2);
    let diag_axis = if axis2 < axis1 { axis1 - 1 } else { axis1 };
    let out_shape: Vec<i64> = diag_shape
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != diag_axis)
        .map(|(_, &extent)| extent)
        .collect();
    let numel: i64 = out_shape.iter().product();
    let mut values = vec![0.0f32; numel as usize];
    for (linear, value) in diag_values.iter().enumerate() {
        let coords = unflatten(linear as i64, &diag_shape);
        let mut flat = 0i64;
        for (index, &extent) in diag_shape.iter().enumerate() {
            if index != diag_axis {
                flat = flat * extent + coords[index];
            }
        }
        values[flat as usize] += value;
    }
    (out_shape, values)
}

unsafe fn observed_shape(tensor: *const chelis_tensor) -> Vec<i64> {
    // [05-OP-31]: rank zero has a null shape pointer, so read extents
    // through the pointer only for the declared rank.
    unsafe {
        (0..chelis_tensor_rank(tensor) as usize)
            .map(|axis| chelis_tensor_shape(tensor, axis as i32))
            .collect()
    }
}

unsafe fn observed_values(tensor: *mut chelis_tensor) -> Vec<f32> {
    unsafe {
        let view = chelis_tensor_read_view(tensor);
        let data = view.data.cast::<f32>();
        (0..view.count as usize)
            .map(|index| *data.add(index))
            .collect()
    }
}

/// Every ordered pair of distinct axes for `rank`.
fn ordered_axis_pairs(rank: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for axis1 in 0..rank {
        for axis2 in 0..rank {
            if axis1 != axis2 {
                pairs.push((axis1, axis2));
            }
        }
    }
    pairs
}

/// Full ordered-pair sweep of one shape for one operation, collecting every
/// mismatch so a failure reports the complete wrong-pair set at once.
fn sweep(shape: &[i64], trace: bool) {
    let mut failures = Vec::new();
    for (axis1, axis2) in ordered_axis_pairs(shape.len()) {
        let (want_shape, want_values) = if trace {
            reference_trace(shape, axis1, axis2)
        } else {
            reference_diagonal(shape, axis1, axis2)
        };
        unsafe {
            let input = ramp_tensor_f32(shape);
            let out = if trace {
                chelis_tensor_trace(input, axis1 as i32, axis2 as i32)
            } else {
                chelis_tensor_diagonal(input, axis1 as i32, axis2 as i32)
            };
            let got_shape = observed_shape(out);
            let got_values = observed_values(out);
            if got_shape != want_shape || got_values != want_values {
                failures.push(format!(
                    "axes ({axis1}, {axis2}): got shape {got_shape:?} values {got_values:?}, \
                     want shape {want_shape:?} values {want_values:?}"
                ));
            }
            chelis_tensor_release(out);
            chelis_tensor_release(input);
        }
    }
    let op = if trace { "trace" } else { "diagonal" };
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
    sweep(&[3, 4], false);
}

#[test]
fn diagonal_rank3_every_ordered_pair_matches_op33() {
    sweep(&[2, 3, 4], false);
}

#[test]
fn diagonal_rank4_every_ordered_pair_matches_op33() {
    // Includes the non-adjacent reversed pair (2, 0) on distinct extents,
    // the smallest configuration where the wrong-shape defect is
    // observable in isolation.
    sweep(&[2, 3, 2, 5], false);
}

#[test]
fn trace_rank2_every_ordered_pair_matches_op33() {
    sweep(&[3, 4], true);
}

#[test]
fn trace_rank3_every_ordered_pair_matches_op33() {
    sweep(&[2, 3, 4], true);
}

#[test]
fn trace_rank4_every_ordered_pair_matches_op33() {
    sweep(&[2, 3, 2, 5], true);
}

/// Hand-pinned anchor for the reference itself, from the chelis#1349
/// reproduction: the rank-3 `[2, 2, 2]` ramp cube. The unfixed runtime
/// returned `[0, 0, 7, 7]` and `[7, 7]` here.
#[test]
fn diagonal_and_trace_cube_axes_0_1_match_pinned_values() {
    unsafe {
        let cube = ramp_tensor_f32(&[2, 2, 2]);
        let diag = chelis_tensor_diagonal(cube, 0, 1);
        assert_eq!(observed_shape(diag), vec![2, 2]);
        assert_eq!(observed_values(diag), vec![0.0, 1.0, 6.0, 7.0]);
        let trace = chelis_tensor_trace(cube, 0, 1);
        assert_eq!(observed_shape(trace), vec![2]);
        assert_eq!(observed_values(trace), vec![6.0, 8.0]);
        chelis_tensor_release(trace);
        chelis_tensor_release(diag);
        chelis_tensor_release(cube);
    }
}

/// A zero extent on a retained axis flows through: shape `[3, 0, 4]` with
/// axes `(2, 0)` keeps axis 1's zero extent and produces an empty output
/// of the declared shape.
#[test]
fn diagonal_zero_extent_retained_axis_is_empty_with_declared_shape() {
    unsafe {
        let input = ramp_tensor_f32(&[3, 0, 4]);
        let out = chelis_tensor_diagonal(input, 2, 0);
        assert_eq!(observed_shape(out), vec![0, 3]);
        assert_eq!(chelis_tensor_numel(out), 0);
        chelis_tensor_release(out);
        chelis_tensor_release(input);
    }
}

/// Fail-closed child: `runtime_fail!` exits the process, so each rejection
/// case runs in a child process (the issue_980 pattern).
#[test]
fn diagonal_axis_rejection_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    unsafe {
        let input = ramp_tensor_f32(&[3, 4]);
        match case.as_str() {
            "diagonal_equal_axes" => chelis_tensor_diagonal(input, 1, 1),
            "diagonal_axis_oob" => chelis_tensor_diagonal(input, 0, 2),
            "trace_equal_axes" => chelis_tensor_trace(input, 0, 0),
            "trace_axis_oob" => chelis_tensor_trace(input, 0, 5),
            _ => panic!("unknown rejection child case `{case}`"),
        };
    }
    panic!("rejection child case `{case}` returned instead of failing");
}

/// Deliberate fail-closed negative controls (they reject identically before
/// and after the chelis#1349 repair, except `trace_axis_oob`, whose
/// diagnostic now names `trace` because trace normalizes its own axes
/// against the source rank before delegating).
#[test]
fn diagonal_and_trace_reject_equal_and_out_of_range_axes() {
    let test_binary = std::env::current_exe().expect("current test binary");
    for (case, expected) in [
        ("diagonal_equal_axes", "diagonal expects distinct axes"),
        ("diagonal_axis_oob", "diagonal axis 2 out of bounds"),
        ("trace_equal_axes", "diagonal expects distinct axes"),
        ("trace_axis_oob", "trace axis 5 out of bounds"),
    ] {
        let output = Command::new(&test_binary)
            .args(["--exact", "diagonal_axis_rejection_child", "--nocapture"])
            .env(CHILD_CASE_ENV, case)
            .output()
            .unwrap_or_else(|error| panic!("run rejection child `{case}`: {error}"));
        assert!(
            !output.status.success(),
            "rejection child `{case}` returned success instead of failing"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(expected),
            "rejection child `{case}` did not emit `{expected}`:\n{stderr}"
        );
    }
}
