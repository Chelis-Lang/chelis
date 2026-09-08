//! `diagonal` and `trace` must map output coordinates back to source axes.
//!
//! [05-OP-33] defines `diagonal` as "keeps source axis order with the second
//! axis removed, and replaces the retained first axis extent with the smaller
//! selected extent", and `trace` as "that diagonal followed by [05-OP-30]'s
//! canonical balanced tree ... and removes both axes".
//!
//! Two consequences of that wording are easy to get wrong, and both were:
//!
//! * The output holds one coordinate per RETAINED source axis, so the
//!   diagonal's own coordinate sits at axis1's position AFTER axis2 is
//!   removed, which is one slot earlier whenever axis2 comes first. Reading
//!   the output coordinate vector at the SOURCE axis number picked up a
//!   neighbouring axis's coordinate, and ran off the end of that vector when
//!   axis1 was the last source axis.
//! * Reconstructing the remaining source coordinates has to consume the slot
//!   the diagonal occupies. Without that, every source axis after the
//!   diagonal's output position read the diagonal's coordinate instead of its
//!   own.
//!
//! `trace` then has to reduce the axis the diagonal was written into.
//! `min(axis1, axis2)` names that slot only when the axes are adjacent and
//! axis1 is the later one.
//!
//! Every case below asserts exact stored elements rather than the absence of a
//! crash, because both defects produced values, not diagnostics: the rank-2
//! `(0, 1)` spelling that every other test in this repository uses is the one
//! case where all three mistakes cancel.

use chelis_runtime::{
    chelis_alloc, chelis_tensor, chelis_tensor_begin_write, chelis_tensor_diagonal,
    chelis_tensor_end_write, chelis_tensor_numel, chelis_tensor_rank, chelis_tensor_read_view,
    chelis_tensor_release, chelis_tensor_shape, chelis_tensor_trace, chelis_tensor_write_view,
    CHELIS_DTYPE_BOOL, CHELIS_DTYPE_F32, CHELIS_DTYPE_I64,
};
use std::env;
use std::process::Command;

const CHILD_ENV: &str = "CHELIS_OP33_DIAGONAL_CHILD";

unsafe fn tensor(dtype: u8, shape: &[i64]) -> *mut chelis_tensor {
    chelis_alloc(shape.len() as i32, shape.as_ptr(), dtype)
}

unsafe fn write_elements<T: Copy>(tensor: *mut chelis_tensor, values: &[T]) {
    let guard = chelis_tensor_begin_write(tensor);
    let view = chelis_tensor_write_view(guard);
    assert_eq!(view.count as usize, values.len());
    view.data
        .cast::<T>()
        .copy_from(values.as_ptr(), values.len());
    chelis_tensor_end_write(guard);
}

unsafe fn elements<T: Copy>(tensor: *const chelis_tensor) -> Vec<T> {
    let view = chelis_tensor_read_view(tensor);
    if view.count == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(view.data.cast::<T>(), view.count as usize).to_vec()
    }
}

/// `f32` tensor holding `0, 1, 2, ...` in row-major order, so every element
/// names its own flat index and a misrouted coordinate is legible.
unsafe fn ramp_f32(shape: &[i64]) -> *mut chelis_tensor {
    let out = tensor(CHELIS_DTYPE_F32, shape);
    let count = chelis_tensor_numel(out) as usize;
    let values = (0..count).map(|index| index as f32).collect::<Vec<_>>();
    write_elements(out, &values);
    out
}

unsafe fn f32_elements(tensor: *const chelis_tensor) -> Vec<f32> {
    // [05-OP-31]: a zero-size tensor has a null `data` pointer, which is not a
    // legal slice base even for a zero-length slice.
    elements(tensor)
}

unsafe fn extents(tensor: *const chelis_tensor) -> Vec<i64> {
    (0..chelis_tensor_rank(tensor) as usize)
        .map(|axis| chelis_tensor_shape(tensor, axis as i32))
        .collect()
}

/// One `diagonal` case: source shape, the two axes, the required output
/// extents, and the required stored elements.
struct DiagonalCase {
    name: &'static str,
    shape: &'static [i64],
    axis1: i32,
    axis2: i32,
    out_shape: &'static [i64],
    elements: &'static [f32],
}

/// `in[i][j][k] = 4i + 2j + k` for the rank-3 ramp, `8i + 4j + 2k + l` for the
/// rank-4 one; every expectation below is that closed form evaluated at the
/// coordinates [05-OP-33] names, not a recording of what the code emits.
const DIAGONAL_CASES: &[DiagonalCase] = &[
    // The universally covered spelling. Correct before and after, and kept so
    // the repair cannot regress it.
    DiagonalCase {
        name: "rank2-forward",
        shape: &[2, 2],
        axis1: 0,
        axis2: 1,
        out_shape: &[2],
        elements: &[0.0, 3.0],
    },
    // axis1 is the last source axis: the output coordinate vector has one
    // entry and the source axis number is 1.
    DiagonalCase {
        name: "rank2-reversed",
        shape: &[2, 2],
        axis1: 1,
        axis2: 0,
        out_shape: &[2],
        elements: &[0.0, 3.0],
    },
    // Same pair through §2.3's negative normalization.
    DiagonalCase {
        name: "rank2-negative-axes-reversed",
        shape: &[2, 2],
        axis1: -1,
        axis2: -2,
        out_shape: &[2],
        elements: &[0.0, 3.0],
    },
    // A retained axis sits AFTER the diagonal's output slot: out[d][k] =
    // in[d][d][k] = 6d + k.
    DiagonalCase {
        name: "rank3-leading-pair",
        shape: &[2, 2, 2],
        axis1: 0,
        axis2: 1,
        out_shape: &[2, 2],
        elements: &[0.0, 1.0, 6.0, 7.0],
    },
    // A retained axis sits BEFORE it: out[i][d] = in[i][d][d] = 4i + 3d. This
    // spelling was already correct, which is why it is here.
    DiagonalCase {
        name: "rank3-trailing-pair",
        shape: &[2, 2, 2],
        axis1: 1,
        axis2: 2,
        out_shape: &[2, 2],
        elements: &[0.0, 3.0, 4.0, 7.0],
    },
    // Straddling pair, axis1 first: out[d][j] = in[d][j][d] = 5d + 2j.
    DiagonalCase {
        name: "rank3-straddling-forward",
        shape: &[2, 2, 2],
        axis1: 0,
        axis2: 2,
        out_shape: &[2, 2],
        elements: &[0.0, 2.0, 5.0, 7.0],
    },
    // Straddling pair, axis2 first: out[j][d] = in[d][j][d] = 5d + 2j, with
    // the diagonal in the LAST output slot.
    DiagonalCase {
        name: "rank3-straddling-reversed",
        shape: &[2, 2, 2],
        axis1: 2,
        axis2: 0,
        out_shape: &[2, 2],
        elements: &[0.0, 5.0, 2.0, 7.0],
    },
    // Retained axes on both sides of the diagonal's slot, axis2 first:
    // out[j][d][l] = in[d][j][d][l] = 10d + 4j + l.
    DiagonalCase {
        name: "rank4-straddling-reversed",
        shape: &[2, 2, 2, 2],
        axis1: 2,
        axis2: 0,
        out_shape: &[2, 2, 2],
        elements: &[0.0, 1.0, 10.0, 11.0, 4.0, 5.0, 14.0, 15.0],
    },
    // Distinct extents everywhere, so a misrouted coordinate changes the
    // OUTPUT EXTENTS and not merely the values. `chelis-types`'
    // `infer_diagonal_result_type` derives the declared type the same way
    // ([3, 2, 5] here), so a disagreement is a checked type the runtime does
    // not honor. out[j][d][l] = in[d][j][d][l] = 35d + 10j + l.
    DiagonalCase {
        name: "rank4-distinct-extents-reversed",
        shape: &[2, 3, 2, 5],
        axis1: 2,
        axis2: 0,
        out_shape: &[3, 2, 5],
        elements: &[
            0.0, 1.0, 2.0, 3.0, 4.0, 35.0, 36.0, 37.0, 38.0, 39.0, 10.0, 11.0, 12.0, 13.0, 14.0,
            45.0, 46.0, 47.0, 48.0, 49.0, 20.0, 21.0, 22.0, 23.0, 24.0, 55.0, 56.0, 57.0, 58.0,
            59.0,
        ],
    },
    // The selected extent is the smaller of the two, in both axis orders.
    DiagonalCase {
        name: "non-square-forward",
        shape: &[3, 2],
        axis1: 0,
        axis2: 1,
        out_shape: &[2],
        elements: &[0.0, 3.0],
    },
    DiagonalCase {
        name: "non-square-reversed",
        shape: &[3, 2],
        axis1: 1,
        axis2: 0,
        out_shape: &[2],
        elements: &[0.0, 3.0],
    },
    // A zero selected extent yields an empty result, not one synthetic
    // element ([05-OP-33]).
    DiagonalCase {
        name: "zero-selected-extent",
        shape: &[0, 3],
        axis1: 0,
        axis2: 1,
        out_shape: &[0],
        elements: &[],
    },
    DiagonalCase {
        name: "zero-selected-extent-reversed",
        shape: &[0, 3],
        axis1: 1,
        axis2: 0,
        out_shape: &[0],
        elements: &[],
    },
];

struct TraceCase {
    name: &'static str,
    shape: &'static [i64],
    axis1: i32,
    axis2: i32,
    out_shape: &'static [i64],
    elements: &'static [f32],
}

/// Trace is symmetric in its two axes: the same unordered pair must produce
/// the same values whichever order it is spelled in, which is what the
/// forward/reversed siblings below assert.
const TRACE_CASES: &[TraceCase] = &[
    TraceCase {
        name: "rank2-forward",
        shape: &[2, 2],
        axis1: 0,
        axis2: 1,
        out_shape: &[],
        elements: &[3.0],
    },
    TraceCase {
        name: "rank2-reversed",
        shape: &[2, 2],
        axis1: 1,
        axis2: 0,
        out_shape: &[],
        elements: &[3.0],
    },
    // sum_d in[d][d][k] = (0 + 6, 1 + 7).
    TraceCase {
        name: "rank3-leading-pair",
        shape: &[2, 2, 2],
        axis1: 0,
        axis2: 1,
        out_shape: &[2],
        elements: &[6.0, 8.0],
    },
    TraceCase {
        name: "rank3-leading-pair-reversed",
        shape: &[2, 2, 2],
        axis1: 1,
        axis2: 0,
        out_shape: &[2],
        elements: &[6.0, 8.0],
    },
    // sum_d in[i][d][d] = (0 + 3, 4 + 7).
    TraceCase {
        name: "rank3-trailing-pair",
        shape: &[2, 2, 2],
        axis1: 1,
        axis2: 2,
        out_shape: &[2],
        elements: &[3.0, 11.0],
    },
    // sum_d in[d][j][d] = (0 + 5, 2 + 7).
    TraceCase {
        name: "rank3-straddling-forward",
        shape: &[2, 2, 2],
        axis1: 0,
        axis2: 2,
        out_shape: &[2],
        elements: &[5.0, 9.0],
    },
    TraceCase {
        name: "rank3-straddling-reversed",
        shape: &[2, 2, 2],
        axis1: 2,
        axis2: 0,
        out_shape: &[2],
        elements: &[5.0, 9.0],
    },
    // sum_d in[d][j][d][l] = 8j + 2l + 10.
    TraceCase {
        name: "rank4-straddling-reversed",
        shape: &[2, 2, 2, 2],
        axis1: 2,
        axis2: 0,
        out_shape: &[2, 2],
        elements: &[10.0, 12.0, 18.0, 20.0],
    },
    // Distinct extents, so reducing the wrong axis of the diagonal produces
    // the wrong RANK-BEARING extents too: `infer_trace_result_type` removes
    // both source axes and declares [3, 5], while reducing the diagonal's
    // axis 0 would yield [2, 5]. sum_d in[d][j][d][l] = 20j + 2l + 35.
    TraceCase {
        name: "rank4-distinct-extents-reversed",
        shape: &[2, 3, 2, 5],
        axis1: 2,
        axis2: 0,
        out_shape: &[3, 5],
        elements: &[
            35.0, 37.0, 39.0, 41.0, 43.0, 55.0, 57.0, 59.0, 61.0, 63.0, 75.0, 77.0, 79.0, 81.0,
            83.0,
        ],
    },
];

#[test]
fn diagonal_maps_every_axis_pair_to_the_declared_source_coordinates() {
    let mut failures = Vec::new();
    unsafe {
        for case in DIAGONAL_CASES {
            let input = ramp_f32(case.shape);
            let out = chelis_tensor_diagonal(input, case.axis1, case.axis2);
            let observed_shape = extents(out);
            let observed = f32_elements(out);
            if observed_shape != case.out_shape {
                failures.push(format!(
                    "diagonal `{}`: output extents {observed_shape:?}, required {:?}",
                    case.name, case.out_shape
                ));
            }
            if observed != case.elements {
                failures.push(format!(
                    "diagonal `{}`: elements {observed:?}, required {:?}",
                    case.name, case.elements
                ));
            }
            chelis_tensor_release(out);
            chelis_tensor_release(input);
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} diagonal axis-pair cases wrong:\n{}",
        failures.len(),
        DIAGONAL_CASES.len(),
        failures.join("\n")
    );
}

#[test]
fn trace_reduces_the_axis_the_diagonal_was_written_into() {
    let mut failures = Vec::new();
    unsafe {
        for case in TRACE_CASES {
            let input = ramp_f32(case.shape);
            let out = chelis_tensor_trace(input, case.axis1, case.axis2);
            let observed_shape = extents(out);
            let observed = f32_elements(out);
            if observed_shape != case.out_shape {
                failures.push(format!(
                    "trace `{}`: output extents {observed_shape:?}, required {:?}",
                    case.name, case.out_shape
                ));
            }
            if observed != case.elements {
                failures.push(format!(
                    "trace `{}`: elements {observed:?}, required {:?}",
                    case.name, case.elements
                ));
            }
            chelis_tensor_release(out);
            chelis_tensor_release(input);
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} trace axis-pair cases wrong:\n{}",
        failures.len(),
        TRACE_CASES.len(),
        failures.join("\n")
    );
}

/// [05-OP-33]: `diagonal` "admits every active dtype including bool" and
/// "preserves bits". The mapping repair must not be an f32-only property, and
/// bool exercises the one-byte storage lane the axis walk copies through.
#[test]
fn diagonal_preserves_stored_bits_for_bool_and_int64_in_both_axis_orders() {
    unsafe {
        let flags = tensor(CHELIS_DTYPE_BOOL, &[2, 2, 2]);
        write_elements(flags, &[1_u8, 0, 0, 1, 1, 1, 0, 0]);
        // out[d][k] = in[d][d][k]: (in[0][0][0], in[0][0][1], in[1][1][0], in[1][1][1]).
        for (axis1, axis2, required) in [(0_i32, 1_i32, [1_u8, 0, 0, 0]), (1, 0, [1, 0, 0, 0])] {
            let out = chelis_tensor_diagonal(flags, axis1, axis2);
            let observed = elements::<u8>(out);
            assert_eq!(
                observed, required,
                "bool diagonal at axes ({axis1}, {axis2}) lost stored bits"
            );
            chelis_tensor_release(out);
        }
        chelis_tensor_release(flags);

        let wide = tensor(CHELIS_DTYPE_I64, &[2, 2, 2]);
        write_elements(
            wide,
            &[
                i64::MIN,
                i64::MAX,
                -1,
                7,
                0,
                -9_007_199_254_740_993,
                123,
                i64::MIN + 1,
            ],
        );
        // out[j][d] = in[d][j][d] for axes (2, 0):
        // (in[0][0][0], in[1][0][1], in[0][1][0], in[1][1][1]).
        let out = chelis_tensor_diagonal(wide, 2, 0);
        let observed = elements::<i64>(out);
        assert_eq!(
            observed,
            [i64::MIN, -9_007_199_254_740_993, -1, i64::MIN + 1],
            "int64 diagonal at axes (2, 0) did not preserve exact stored values"
        );
        chelis_tensor_release(out);
        chelis_tensor_release(wide);
    }
}

/// Negative parity: the axis domain still fails closed. `runtime_fail!` exits
/// the process, so these run in a child.
fn run_invalid_case(case: &str) -> ! {
    unsafe {
        match case {
            "diagonal-equal-axes" => {
                chelis_tensor_diagonal(ramp_f32(&[2, 2, 2]), 1, 1);
            }
            "diagonal-equal-axes-normalized" => {
                // 2 and -1 normalize to the same axis of a rank-3 tensor.
                chelis_tensor_diagonal(ramp_f32(&[2, 2, 2]), 2, -1);
            }
            "diagonal-axis-out-of-range" => {
                chelis_tensor_diagonal(ramp_f32(&[2, 2]), 0, 2);
            }
            "diagonal-negative-axis-out-of-range" => {
                chelis_tensor_diagonal(ramp_f32(&[2, 2]), 0, -3);
            }
            "trace-equal-axes" => {
                chelis_tensor_trace(ramp_f32(&[2, 2, 2]), 0, 0);
            }
            "trace-axis-out-of-range" => {
                chelis_tensor_trace(ramp_f32(&[2, 2]), 5, 0);
            }
            "trace-bool-operand" => {
                let flags = tensor(CHELIS_DTYPE_BOOL, &[2, 2]);
                write_elements(flags, &[1_u8, 0, 0, 1]);
                chelis_tensor_trace(flags, 0, 1);
            }
            other => panic!("unknown invalid diagonal case: {other}"),
        }
    }
    unreachable!("invalid diagonal case must not return");
}

const INVALID_CASES: &[(&str, &str)] = &[
    ("diagonal-equal-axes", "distinct axes"),
    ("diagonal-equal-axes-normalized", "distinct axes"),
    ("diagonal-axis-out-of-range", "Domain:"),
    ("diagonal-negative-axis-out-of-range", "Domain:"),
    ("trace-equal-axes", "distinct axes"),
    ("trace-axis-out-of-range", "Domain:"),
    ("trace-bool-operand", "Domain:"),
];

#[test]
fn invalid_axis_pairs_and_dtypes_still_fail_closed() {
    if let Ok(case) = env::var(CHILD_ENV) {
        run_invalid_case(&case);
    }
    let test_binary = env::current_exe().expect("current test binary");
    for (case, expected) in INVALID_CASES {
        let output = Command::new(&test_binary)
            .args([
                "--exact",
                "invalid_axis_pairs_and_dtypes_still_fail_closed",
                "--nocapture",
            ])
            .env(CHILD_ENV, case)
            .output()
            .unwrap_or_else(|error| panic!("spawn invalid diagonal child `{case}`: {error}"));
        assert!(
            !output.status.success(),
            "invalid diagonal case `{case}` returned success"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(expected),
            "invalid diagonal case `{case}` did not report `{expected}`:\n{stderr}"
        );
    }
}
