//! chelis#1351: independent [05-OP-33] expectations for `diagonal` and
//! `trace`, checked against each lane separately.
//!
//! `crates/chelis-cli/tests/parity.rs` compares the IR evaluator lane
//! against the compiled C lane and asserts byte equality. That catches a
//! defect present in one lane. It cannot catch a defect present in both:
//! chelis#1349 mis-indexed the source coordinate identically in
//! `tensor_diagonal_value` and `chelis_tensor_diagonal`, so the two lanes
//! agreed bit-for-bit on the wrong answer and parity stayed green.
//!
//! This suite removes lane agreement from the evidence chain. The expected
//! extents and elements below are computed by [`diagonal_expected`] and
//! [`trace_expected`], which are written from the wording of [05-OP-33] in
//! `spec/05-risc-primitives.md`:
//!
//! > `diagonal` ... requires distinct axes, keeps source axis order with the
//! > second axis removed, and replaces the retained first axis extent with
//! > the smaller selected extent. It reads equal coordinates on both axes
//! > and preserves bits. ... `trace` is that diagonal followed by
//! > [05-OP-30]'s canonical balanced tree and default accumulator; it
//! > inherits sum's result dtype, empty identity, and overflow rule and
//! > removes both axes.
//!
//! The reference selects source cells whose two axis coordinates are equal
//! and places each one by its retained coordinates. It never reconstructs a
//! source index from an output index, which is the step both lanes got
//! wrong, and it is not derived from either lane's code. Each lane is then
//! required to agree with it. The ordinary lane-to-lane comparison is kept
//! as well, so a single-lane divergence still reports.
//!
//! Coverage is the part chelis#1349 showed to be load-bearing: all six
//! ordered rank-3 axis pairs, both rank-2 orders on a non-square matrix,
//! and a tensor whose elements alternate sign. At rank 2 the pair `(0, 1)`
//! is also `(rank-2, rank-1)`, the one configuration where the three
//! mistakes cancel, so a corpus pinned to it proves nothing about the
//! others. Element values are whole f32 numbers that name their own source
//! position, so a misrouted coordinate shows up as a wrong number rather
//! than as a crash.
//!
//! Bounds. This suite checks values and extents, not the result dtype. It does not
//! pin [05-OP-30]'s summation order: the traced values are exact in f32, so
//! any grouping gives the same result here, and the balanced-tree order is
//! locked by the runtime's own tree tests. It covers `diagonal` and `trace`
//! only. It adds no independent controls for other operations.

use std::fs;

use assert_cmd::Command;
use serde_json::Value;
use tempfile::tempdir;

mod common;

/// Program name for the generated `.ch` file, C translation unit, and binary.
const PROGRAM: &str = "issue_1351_structural_axis_oracle";

/// Number of printed roots the program owes. Pinned so a shrunken program,
/// an empty run, or a lane that prints nothing cannot pass by vacuity.
const EXPECTED_BINDINGS: usize = 21;

static RANK3_SHAPE: [usize; 3] = [2, 3, 4];
static RECT_SHAPE: [usize; 2] = [2, 3];

// -----------------------------------------------------------------------------
// Independent reference: [05-OP-33] read directly
// -----------------------------------------------------------------------------

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major coordinates of `flat` in `shape`.
fn unravel(flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut rest = flat;
    let mut coord = vec![0usize; shape.len()];
    for (slot, extent) in coord.iter_mut().zip(shape.iter()).rev() {
        *slot = rest % extent;
        rest /= extent;
    }
    coord
}

/// Row-major flat index of `coord` in `shape`.
fn ravel(coord: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    for (axis, extent) in shape.iter().enumerate() {
        flat = flat * extent + coord[axis];
    }
    flat
}

/// [05-OP-33] `diagonal`: source axis order with `axis2` removed, the
/// retained `axis1` extent replaced with the smaller selected extent, and
/// the elements that sit at equal coordinates on both axes.
///
/// Written as a selection over source cells. A source cell belongs to the
/// result when its two axis coordinates are equal and below the selected
/// extent, and its place in the result is given by its remaining
/// coordinates in source order. The result is complete when every output
/// cell is written exactly once, which the assertions below require.
fn diagonal_expected(shape: &[usize], data: &[f64], axis1: usize, axis2: usize) -> Observation {
    assert_ne!(axis1, axis2, "[05-OP-33] requires distinct axes");
    let selected = shape[axis1].min(shape[axis2]);
    let out_shape: Vec<usize> = (0..shape.len())
        .filter(|axis| *axis != axis2)
        .map(|axis| if axis == axis1 { selected } else { shape[axis] })
        .collect();

    let mut cells: Vec<Option<f64>> = vec![None; numel(&out_shape)];
    for (flat, value) in data.iter().enumerate() {
        let coord = unravel(flat, shape);
        if coord[axis1] != coord[axis2] || coord[axis1] >= selected {
            continue;
        }
        let retained: Vec<usize> = (0..shape.len())
            .filter(|axis| *axis != axis2)
            .map(|axis| coord[axis])
            .collect();
        let slot = ravel(&retained, &out_shape);
        assert!(
            cells[slot].is_none(),
            "reference fault: two source cells claim output slot {slot} \
             for diagonal({axis1}, {axis2}) of shape {shape:?}"
        );
        cells[slot] = Some(*value);
    }

    let values = cells
        .into_iter()
        .enumerate()
        .map(|(slot, value)| {
            value.unwrap_or_else(|| {
                panic!(
                    "reference fault: output slot {slot} unwritten for \
                     diagonal({axis1}, {axis2}) of shape {shape:?}"
                )
            })
        })
        .collect();
    Observation {
        shape: Some(out_shape),
        values,
    }
}

/// [05-OP-33] `trace`: that diagonal summed with the default accumulator,
/// with both axes removed.
///
/// Derived the same way, by summing the selected cells into the result
/// named by the coordinates that survive removing both axes. Every result
/// cell must receive exactly `selected` contributions.
fn trace_expected(shape: &[usize], data: &[f64], axis1: usize, axis2: usize) -> Observation {
    assert_ne!(axis1, axis2, "[05-OP-33] requires distinct axes");
    let selected = shape[axis1].min(shape[axis2]);
    let out_shape: Vec<usize> = (0..shape.len())
        .filter(|axis| *axis != axis1 && *axis != axis2)
        .map(|axis| shape[axis])
        .collect();

    let mut values = vec![0.0f64; numel(&out_shape)];
    let mut contributions = vec![0usize; values.len()];
    for (flat, value) in data.iter().enumerate() {
        let coord = unravel(flat, shape);
        if coord[axis1] != coord[axis2] || coord[axis1] >= selected {
            continue;
        }
        let retained: Vec<usize> = (0..shape.len())
            .filter(|axis| *axis != axis1 && *axis != axis2)
            .map(|axis| coord[axis])
            .collect();
        let slot = ravel(&retained, &out_shape);
        values[slot] += value;
        contributions[slot] += 1;
    }
    assert!(
        contributions.iter().all(|count| *count == selected),
        "reference fault: trace({axis1}, {axis2}) of shape {shape:?} summed \
         {contributions:?} cells, expected {selected} everywhere"
    );

    // Rank 0 renders bare, with no `tensor(shape=[], data=[..])` wrapper.
    let shape = if out_shape.is_empty() {
        None
    } else {
        Some(out_shape)
    };
    Observation { shape, values }
}

// -----------------------------------------------------------------------------
// Observations
// -----------------------------------------------------------------------------

/// One printed root: its extents (absent for a bare rank-0 scalar) and its
/// value tokens.
#[derive(Debug, Clone, PartialEq)]
struct Observation {
    shape: Option<Vec<usize>>,
    values: Vec<f64>,
}

impl Observation {
    /// The [05-OBS] value tokens this observation owes. Every value here is
    /// a whole f32 number well inside the exactly representable range, so
    /// the shortest round-trip token is the integer with one fractional
    /// digit.
    fn tokens(&self) -> Vec<String> {
        self.values
            .iter()
            .map(|value| {
                assert_eq!(
                    value.fract(),
                    0.0,
                    "oracle values are whole numbers so their rendering is unambiguous"
                );
                assert!(value.abs() < 16_777_216.0, "value must be exact in f32");
                format!("{value:.1}")
            })
            .collect()
    }
}

/// What a lane printed for one root.
#[derive(Debug, Clone, PartialEq)]
struct Printed {
    shape: Option<Vec<usize>>,
    tokens: Vec<String>,
}

/// Parse one printed root. Handles the `tensor(shape=[..], data=[..])` form
/// and the bare rank-0 scalar form. Deliberately not the parity
/// comparator: this suite must not inherit that comparator's assumptions.
fn parse_printed(line: &str) -> (String, Printed) {
    let (name, payload) = line
        .split_once(" = ")
        .unwrap_or_else(|| panic!("printed root is not a `name = value` line: {line:?}"));
    let payload = payload.trim();
    if let Some(rest) = payload.strip_prefix("tensor(shape=[") {
        let (shape_text, rest) = rest
            .split_once(']')
            .unwrap_or_else(|| panic!("unterminated shape in {line:?}"));
        let data_text = rest
            .split_once("data=[")
            .and_then(|(_, tail)| tail.rsplit_once("])"))
            .map(|(body, _)| body)
            .unwrap_or_else(|| panic!("unterminated data payload in {line:?}"));
        let shape: Vec<usize> = shape_text
            .split(',')
            .map(str::trim)
            .filter(|piece| !piece.is_empty())
            .map(|piece| {
                piece
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("non-integer extent in {line:?}"))
            })
            .collect();
        let tokens: Vec<String> = data_text
            .split(',')
            .map(|piece| piece.trim().to_string())
            .filter(|piece| !piece.is_empty())
            .collect();
        assert_eq!(
            tokens.len(),
            numel(&shape),
            "shape and element count disagree: {line}"
        );
        (
            name.trim().to_string(),
            Printed {
                shape: Some(shape),
                tokens,
            },
        )
    } else {
        (
            name.trim().to_string(),
            Printed {
                shape: None,
                tokens: vec![payload.to_string()],
            },
        )
    }
}

fn parse_lane(label: &str, stdout: &str) -> Vec<(String, Printed)> {
    let printed: Vec<(String, Printed)> = stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(parse_printed)
        .collect();
    assert_eq!(
        printed.len(),
        EXPECTED_BINDINGS,
        "[{label}] printed {} roots, expected {EXPECTED_BINDINGS}:\n{stdout}",
        printed.len(),
    );
    printed
}

/// Require one lane to agree with the independent expectations. Lane output
/// is never compared with the other lane here.
fn assert_lane_matches_expectations(label: &str, stdout: &str, expected: &[(String, Observation)]) {
    let printed = parse_lane(label, stdout);
    for (name, observation) in expected {
        let printed_value = printed
            .iter()
            .find(|(printed_name, _)| printed_name == name)
            .map(|(_, value)| value)
            .unwrap_or_else(|| panic!("[{label}] printed no root named `{name}`:\n{stdout}"));
        assert_eq!(
            printed_value.shape, observation.shape,
            "[{label}] `{name}` extents disagree with [05-OP-33]"
        );
        assert_eq!(
            printed_value.tokens,
            observation.tokens(),
            "[{label}] `{name}` elements disagree with [05-OP-33]"
        );
    }
}

// -----------------------------------------------------------------------------
// Program under test
// -----------------------------------------------------------------------------

/// `0, 1, 2, ...` in row-major order, so every element names its own source
/// position and a misrouted coordinate is legible.
fn ramp(shape: &[usize]) -> Vec<f64> {
    (0..numel(shape)).map(|index| index as f64).collect()
}

/// `+1, -2, +3, ...`: same magnitudes as a ramp, alternating sign, so a
/// transposed or shifted read changes the sign of the reported value.
fn alternating(shape: &[usize]) -> Vec<f64> {
    (0..numel(shape))
        .map(|index| {
            let magnitude = (index + 1) as f64;
            if index % 2 == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect()
}

fn literal_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|value| format!("{value:.1}f32"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn extent_list(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|extent| format!("{extent}i64"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn tensor_binding(name: &str, shape: &[usize], values: &[f64]) -> String {
    let annotation = shape
        .iter()
        .map(|extent| extent.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "{name}: tensor[{annotation}, f32] = reshape(to_tensor([{}]), [{}])\n",
        literal_list(values),
        extent_list(shape),
    )
}

/// The axis pairs under test, paired with the source they read.
///
/// All six ordered rank-3 pairs for both operations, because the retained
/// slot moves when `axis2 < axis1` and the selected extent changes with the
/// pair. Both rank-2 orders on a non-square matrix, because the selected
/// extent is the smaller one and a `max` would index past the shorter axis.
fn cases() -> Vec<(&'static str, &'static str, [usize; 2])> {
    vec![
        ("diagonal", "ramp", [0, 1]),
        ("diagonal", "ramp", [1, 0]),
        ("diagonal", "ramp", [0, 2]),
        ("diagonal", "ramp", [2, 0]),
        ("diagonal", "ramp", [1, 2]),
        ("diagonal", "ramp", [2, 1]),
        ("diagonal", "alt", [2, 0]),
        ("diagonal", "alt", [1, 2]),
        ("diagonal", "rect", [0, 1]),
        ("diagonal", "rect", [1, 0]),
        ("trace", "ramp", [0, 1]),
        ("trace", "ramp", [1, 0]),
        ("trace", "ramp", [0, 2]),
        ("trace", "ramp", [2, 0]),
        ("trace", "ramp", [1, 2]),
        ("trace", "ramp", [2, 1]),
        ("trace", "alt", [2, 1]),
        ("trace", "rect", [1, 0]),
    ]
}

fn source_of(source: &str) -> (&'static [usize], Vec<f64>) {
    match source {
        "ramp" => (&RANK3_SHAPE, ramp(&RANK3_SHAPE)),
        "alt" => (&RANK3_SHAPE, alternating(&RANK3_SHAPE)),
        "rect" => (&RECT_SHAPE, ramp(&RECT_SHAPE)),
        other => panic!("unknown source tensor `{other}`"),
    }
}

fn binding_name(op: &str, source: &str, axes: [usize; 2]) -> String {
    format!("{op}_{source}_{}{}", axes[0], axes[1])
}

/// The program both lanes run. Every root is a top-level binding, so both
/// lanes print it under [05-OBS-7].
fn program_source() -> String {
    let mut text = String::new();
    text.push_str(&tensor_binding("ramp", &RANK3_SHAPE, &ramp(&RANK3_SHAPE)));
    text.push_str(&tensor_binding(
        "alt",
        &RANK3_SHAPE,
        &alternating(&RANK3_SHAPE),
    ));
    text.push_str(&tensor_binding("rect", &RECT_SHAPE, &ramp(&RECT_SHAPE)));
    for (op, source, axes) in cases() {
        text.push_str(&format!(
            "{} = {op}({source}, {}, {})\n",
            binding_name(op, source, axes),
            axes[0],
            axes[1],
        ));
    }
    text
}

/// The expectations, in printed order: the three source tensors, then one
/// entry per case, each computed from [05-OP-33] alone.
fn independent_expectations() -> Vec<(String, Observation)> {
    let mut expected = vec![
        (
            "ramp".to_string(),
            Observation {
                shape: Some(RANK3_SHAPE.to_vec()),
                values: ramp(&RANK3_SHAPE),
            },
        ),
        (
            "alt".to_string(),
            Observation {
                shape: Some(RANK3_SHAPE.to_vec()),
                values: alternating(&RANK3_SHAPE),
            },
        ),
        (
            "rect".to_string(),
            Observation {
                shape: Some(RECT_SHAPE.to_vec()),
                values: ramp(&RECT_SHAPE),
            },
        ),
    ];
    for (op, source, axes) in cases() {
        let (shape, data) = source_of(source);
        let observation = match op {
            "diagonal" => diagonal_expected(shape, &data, axes[0], axes[1]),
            "trace" => trace_expected(shape, &data, axes[0], axes[1]),
            other => panic!("unknown operation `{other}`"),
        };
        expected.push((binding_name(op, source, axes), observation));
    }
    expected
}

// -----------------------------------------------------------------------------
// Lanes
// -----------------------------------------------------------------------------

fn assert_check_clean(program: &str) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{PROGRAM}.ch"));
    fs::write(&path, program).expect("write program");
    let stdout = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let json: Value = serde_json::from_slice(&stdout).expect("check output is JSON");
    assert_eq!(json["errors"], serde_json::json!([]), "{json}");
    assert_eq!(
        json["score"].as_f64().unwrap_or(0.0),
        1.0,
        "oracle program did not check cleanly: {}",
        String::from_utf8_lossy(&stdout),
    );
}

fn run_eval(program: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{PROGRAM}.ch"));
    fs::write(&path, program).expect("write program");
    let stdout = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    String::from_utf8(stdout).expect("utf-8 stdout")
}

// -----------------------------------------------------------------------------
// The authoritative test
// -----------------------------------------------------------------------------

/// Both lanes must agree with [05-OP-33], separately, and then with each
/// other.
///
/// The native lane really compiles and runs: `common::build_and_run` emits
/// C, links it against the runtime archive, and executes the binary. A
/// missing host compiler fails this test rather than skipping it, because
/// a skipped compile would leave the compiled lane unexercised while the
/// suite still reported success.
#[test]
fn structural_axis_lanes_match_independent_op33_expectations() {
    let expected = independent_expectations();
    assert_eq!(
        expected.len(),
        EXPECTED_BINDINGS,
        "expectation table shrank; the oracle must not pass by covering less"
    );

    let program = program_source();
    assert_check_clean(&program);

    let eval_stdout = run_eval(&program);
    assert!(
        common::gcc_available(),
        "the compiled lane is part of this oracle: install a host C compiler"
    );
    let native_stdout = common::build_and_run(&program, PROGRAM);

    assert_lane_matches_expectations("eval", &eval_stdout, &expected);
    assert_lane_matches_expectations("native-c", &native_stdout, &expected);

    // The ordinary parity comparison is preserved, not replaced: a
    // single-lane divergence still reports here even when both lanes have
    // already been checked against the atom.
    assert_eq!(
        eval_stdout, native_stdout,
        "eval and compiled lanes must remain byte-identical"
    );
}

// -----------------------------------------------------------------------------
// Controls for the reference itself
// -----------------------------------------------------------------------------
//
// The expectations are only worth their weight if they are wrong when the
// atom says something else. These cases are computed by hand from
// [05-OP-33] and compared with the reference, so a fault introduced in
// `diagonal_expected` or `trace_expected` fails here as well as in the lane
// test.

/// `ramp[i, j, k] = 12i + 4j + k` on shape `[2, 3, 4]`.
#[test]
fn reference_matches_hand_computed_rank3_diagonal() {
    let data = ramp(&RANK3_SHAPE);

    // axes (0, 2): axis 2 is removed, axis 0 keeps min(2, 4) = 2, so the
    // result is [2, 3] holding ramp[d, j, d] = 13d + 4j.
    assert_eq!(
        diagonal_expected(&RANK3_SHAPE, &data, 0, 2),
        Observation {
            shape: Some(vec![2, 3]),
            values: vec![0.0, 4.0, 8.0, 13.0, 17.0, 21.0],
        }
    );

    // axes (2, 0): axis 0 is removed, so the retained order is [1, 2] and
    // the diagonal sits last with extent min(4, 2) = 2. Same elements,
    // transposed against the pair above. A lane that ignores axis order
    // reports the case above here.
    assert_eq!(
        diagonal_expected(&RANK3_SHAPE, &data, 2, 0),
        Observation {
            shape: Some(vec![3, 2]),
            values: vec![0.0, 13.0, 4.0, 17.0, 8.0, 21.0],
        }
    );

    // axes (1, 2): axis 2 is removed, axis 1 keeps min(3, 4) = 3, so the
    // result is [2, 3] holding ramp[i, d, d] = 12i + 5d.
    assert_eq!(
        diagonal_expected(&RANK3_SHAPE, &data, 1, 2),
        Observation {
            shape: Some(vec![2, 3]),
            values: vec![0.0, 5.0, 10.0, 12.0, 17.0, 22.0],
        }
    );
}

#[test]
fn reference_matches_hand_computed_rank3_trace() {
    let data = ramp(&RANK3_SHAPE);

    // Both axes are removed, leaving [3]; the sum over d of 13d + 4j is
    // 13 + 8j.
    assert_eq!(
        trace_expected(&RANK3_SHAPE, &data, 0, 2),
        Observation {
            shape: Some(vec![3]),
            values: vec![13.0, 21.0, 29.0],
        }
    );
    // Removing both axes does not depend on their order.
    assert_eq!(
        trace_expected(&RANK3_SHAPE, &data, 2, 0),
        trace_expected(&RANK3_SHAPE, &data, 0, 2)
    );
}

/// The selected extent is the smaller of the two, on a non-square matrix
/// where a `max` would read past the shorter axis.
#[test]
fn reference_uses_the_smaller_selected_extent() {
    let data = ramp(&RECT_SHAPE);
    let expected = Observation {
        shape: Some(vec![2]),
        values: vec![0.0, 4.0],
    };
    assert_eq!(diagonal_expected(&RECT_SHAPE, &data, 0, 1), expected);
    assert_eq!(diagonal_expected(&RECT_SHAPE, &data, 1, 0), expected);

    // Rank 0 renders bare, so the expectation carries no extents.
    assert_eq!(
        trace_expected(&RECT_SHAPE, &data, 1, 0),
        Observation {
            shape: None,
            values: vec![4.0],
        }
    );
}

/// Sign is a value property, not an index property: the alternating source
/// makes a shifted or transposed read change the reported sign.
#[test]
fn reference_preserves_element_sign() {
    let data = alternating(&RANK3_SHAPE);
    let diagonal = diagonal_expected(&RANK3_SHAPE, &data, 2, 0);
    assert_eq!(diagonal.shape, Some(vec![3, 2]));
    // alt[flat] = (flat + 1) with odd flat negated; the picked cells are
    // flats 0, 13, 4, 17, 8, 21.
    assert_eq!(diagonal.values, vec![1.0, -14.0, 5.0, -18.0, 9.0, -22.0]);
}

#[test]
#[should_panic(expected = "shape and element count disagree")]
fn parser_rejects_an_incomplete_tensor_payload() {
    parse_printed("broken = tensor(shape=[2, 3], data=[0.0, 4.0])");
}

/// The printed-root parser reads both [05-OBS] forms.
#[test]
fn parser_reads_tensor_and_bare_scalar_roots() {
    let (name, printed) = parse_printed("diagonal_ramp_02 = tensor(shape=[2, 1], data=[0.0, 4.0])");
    assert_eq!(name, "diagonal_ramp_02");
    assert_eq!(printed.shape, Some(vec![2, 1]));
    assert_eq!(printed.tokens, vec!["0.0".to_string(), "4.0".to_string()]);

    let (name, printed) = parse_printed("trace_rect_10 = 4.0");
    assert_eq!(name, "trace_rect_10");
    assert_eq!(printed.shape, None);
    assert_eq!(printed.tokens, vec!["4.0".to_string()]);
}
