//! chelis#2971, chelis#2972, chelis#2973: `chelis eval` computes `softmax`,
//! `cumsum`, `clamp` and scatter-add at the declared dtype's arithmetic width
//! ([04-NUM-8]), with no f64 evaluation funnel ([05-OP-33], [05-OP-48]).
//!
//! Each `cumsum`, `clamp` and scatter-add row runs the same module through the
//! host evaluator and through the compiled C program, compares every element
//! at its stored width, and compares both with an exact reference: Rust
//! integers for integer rows, storage-width rounding at each accumulator step
//! for float rows. Trap rows require the same failure line, at the same
//! row-major position, in both lanes. Softmax rows compare eval bit for bit
//! with section 4.2's per-primitive graph composed here from the closed typed
//! kernels; the compiled lane's `exp` provenance is owned by chelis#2958 and
//! chelis#2963, so the softmax oracle does not depend on it.

mod ownership_support;

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_types::types::Prim;
use chelis_types::{
    FloatBinOp, FloatUnOp, ScalarValue, TensorStorage, float_binop, float_unop, scalar_from_f64,
};

/// 2^53 + 1, the smallest positive i64 an f64 cannot hold.
const ABOVE_F64: i64 = 9_007_199_254_740_993;

/// One stored element at its own width: integers exactly, floats by the bits
/// of their exact binary64 widening, and every NaN as one class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stored {
    Int(i64),
    Float(u64),
    Nan,
}

fn stored(value: ScalarValue) -> Stored {
    if value.prim().is_integer() {
        Stored::Int(value.as_i64_exact().expect("integer elements read exactly"))
    } else if value.as_f64_lossy().is_nan() {
        Stored::Nan
    } else {
        Stored::Float(value.as_f64_lossy().to_bits())
    }
}

fn float(prim: Prim, value: f64) -> ScalarValue {
    scalar_from_f64("test", prim, value).expect("finite reference value")
}

fn storage_elements(storage: &TensorStorage) -> Vec<Stored> {
    (0..storage.len())
        .map(|index| stored(storage.scalar_at(index)))
        .collect()
}

fn eval_source(source: &str) -> Result<BTreeMap<String, TensorValue>, String> {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
    .map_err(|err| format!("{err:?}"))?;
    Ok(result
        .roots
        .into_iter()
        .filter_map(|root| match (root.name, root.value) {
            (Some(name), ExecutionValue::Tensor { value }) => Some((name, value)),
            _ => None,
        })
        .collect())
}

fn eval_root(source: &str, name: &str) -> TensorValue {
    eval_source(source)
        .unwrap_or_else(|err| panic!("eval failed: {err}"))
        .remove(name)
        .unwrap_or_else(|| panic!("missing eval root {name}"))
}

/// Parse the compiled program's `name = tensor(shape=[..], data=[..])` line
/// for `name`, reading each element at `prim`'s width. The runtime prints
/// floats as their shortest round-trip spelling at their own width.
fn c_root(stdout: &str, name: &str, prim: Prim) -> Vec<Stored> {
    let prefix = format!("{name} = ");
    let line = stdout
        .lines()
        .find_map(|line| line.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("missing C root {name} in {stdout}"));
    let data = line
        .split_once("data=[")
        .and_then(|(_, rest)| rest.rsplit_once(']'))
        .map(|(data, _)| data)
        .unwrap_or_else(|| panic!("unexpected C tensor spelling {line}"));
    data.split(',')
        .map(str::trim)
        .filter(|word| !word.is_empty())
        .map(|word| {
            if prim.is_integer() {
                Stored::Int(word.parse().unwrap_or_else(|_| panic!("C integer {word}")))
            } else {
                let wide: f64 = word.parse().unwrap_or_else(|_| panic!("C float {word}"));
                if wide.is_nan() {
                    Stored::Nan
                } else {
                    stored(float(prim, wide))
                }
            }
        })
        .collect()
}

fn run_c(source: &str, entry: &str) -> String {
    let program = ownership_support::emit(source, entry);
    let (summary, stdout) = ownership_support::run_program(&program);
    ownership_support::balanced(&summary);
    stdout
}

/// Run `source` in both lanes and require `root` to equal `expected` in each.
fn assert_lanes(source: &str, entry: &str, root: &str, prim: Prim, expected: &[Stored]) {
    let evaluated = eval_root(source, root);
    assert_eq!(evaluated.data.prim(), prim, "{entry}: eval dtype");
    assert_eq!(
        storage_elements(&evaluated.data),
        expected,
        "{entry}: eval differs from the exact reference"
    );
    let stdout = run_c(source, entry);
    assert_eq!(
        c_root(&stdout, root, prim),
        expected,
        "{entry}: compiled C differs from the exact reference"
    );
}

/// Run `source` in both lanes and require both to fail with `line`.
fn assert_lanes_trap(source: &str, entry: &str, line: &str) {
    assert_lanes_trap_lines(source, entry, &[line]);
}

/// Run `source` in both lanes and require each to fail with every line of
/// `lines` as a whole line of its failure, in order.
fn assert_lanes_trap_lines(source: &str, entry: &str, lines: &[&str]) {
    let expected = lines.join("\n");
    let err = match eval_source(source) {
        Ok(roots) => panic!("{entry}: eval returned {roots:?} instead of failing with {expected}"),
        Err(err) => err,
    };
    // `eval_source` renders the failure with `{:?}`, which escapes newlines.
    let escaped = lines.join("\\n");
    assert!(
        err.contains(&escaped),
        "{entry}: eval failure {err} lacks `{expected}`"
    );
    let program = ownership_support::emit(source, entry);
    let stderr = ownership_support::run_failure_stderr(&program, "");
    // Each of `lines` must be a whole line of stderr, in order.
    assert!(
        stderr
            .lines()
            .collect::<Vec<_>>()
            .windows(lines.len())
            .any(|window| window == lines),
        "{entry}: compiled C failure {stderr} lacks `{expected}`"
    );
}

/// The [04-NUM-9] clamp domain trap at `dtype`, then its position detail.
fn clamp_trap(dtype: &str) -> String {
    format!("numeric trap: domain in clamp at {dtype}")
}

fn ints(values: &[i64]) -> Vec<Stored> {
    values.iter().map(|&value| Stored::Int(value)).collect()
}

fn floats(prim: Prim, values: &[f64]) -> Vec<Stored> {
    values
        .iter()
        .map(|&value| stored(float(prim, value)))
        .collect()
}

fn literal(prim: Prim, value: f64) -> String {
    match prim {
        Prim::F32 => format!("{value:?}f32"),
        Prim::F64 => format!("{value:?}f64"),
        Prim::F16 => format!("cast({value:?}, f16)"),
        Prim::Bf16 => format!("cast({value:?}, bf16)"),
        other => panic!("not a float row: {}", other.name()),
    }
}

fn float_tensor(prim: Prim, values: &[f64]) -> String {
    let items = values
        .iter()
        .map(|&value| literal(prim, value))
        .collect::<Vec<_>>()
        .join(", ");
    format!("to_tensor([{items}])")
}

fn int_tensor(suffix: &str, values: &[i64]) -> String {
    let items = values
        .iter()
        .map(|value| format!("{value}{suffix}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("to_tensor([{items}])")
}

/// The [05-OP-33] cumsum reference for a float row: an exact-zero f32 (or
/// f64) accumulator adds each element, and each prefix is stored at `prim`.
fn float_cumsum_reference(prim: Prim, values: &[f64]) -> Vec<Stored> {
    let accumulator = if prim == Prim::F64 {
        Prim::F64
    } else {
        Prim::F32
    };
    let mut running = float(accumulator, 0.0);
    values
        .iter()
        .map(|&value| {
            let leaf = float(accumulator, float(prim, value).as_f64_lossy());
            running = float_binop(FloatBinOp::Add, running, leaf).unwrap();
            stored(float(prim, running.as_f64_lossy()))
        })
        .collect()
}

// ---- cumsum (chelis#2972) ----

/// Evidentiary status: REGRESSION TEST. At 08939bc0e eval returns
/// `[2^53, 2^53, 2^53, 2^53 + 4]`: every prefix crossed binary64.
#[test]
fn i64_cumsum_above_two_to_the_53_is_exact_in_both_lanes() {
    let input = [ABOVE_F64, 1, 1, 3];
    let source = format!("out = cumsum({}, 0)\n", int_tensor("i64", &input));
    let expected = ints(&[ABOVE_F64, ABOVE_F64 + 1, ABOVE_F64 + 2, ABOVE_F64 + 5]);
    assert_lanes(&source, "cumsum-i64", "out", Prim::Int64, &expected);
}

#[test]
fn i32_cumsum_is_exact_in_both_lanes() {
    let input = [i64::from(i32::MAX) - 3, 1, 1, 1];
    let source = format!("out = cumsum({}, 0)\n", int_tensor("i32", &input));
    let expected = ints(&[
        i64::from(i32::MAX) - 3,
        i64::from(i32::MAX) - 2,
        i64::from(i32::MAX) - 1,
        i64::from(i32::MAX),
    ]);
    assert_lanes(&source, "cumsum-i32", "out", Prim::Int32, &expected);
}

/// Section 5.7.1: an i8 or i16 operand accumulates and is stored at the i32
/// default accumulator, `sum_result(i8, i32) = i32`. Evidentiary status:
/// REGRESSION TEST; at 08939bc0e eval narrows each prefix to i8 and traps
/// overflow at 128.
#[test]
fn narrow_integer_cumsum_is_stored_at_the_i32_accumulator_in_both_lanes() {
    for suffix in ["i8", "i16"] {
        let source = format!("out = cumsum({}, 0)\n", int_tensor(suffix, &[127, 1, 1]));
        let expected = ints(&[127, 128, 129]);
        assert_lanes(&source, suffix, "out", Prim::Int32, &expected);
    }
}

/// [04-NUM-3]: integer overflow is checked at each addition.
#[test]
fn i32_cumsum_overflow_traps_in_both_lanes() {
    let input = [i64::from(i32::MAX), 1];
    let source = format!("out = cumsum({}, 0)\n", int_tensor("i32", &input));
    assert_lanes_trap(
        &source,
        "cumsum-i32-overflow",
        "numeric trap: overflow in cumsum at i32",
    );
}

/// Evidentiary status: REGRESSION TEST. At 08939bc0e eval accumulates in
/// f64 and returns `2^24 + 2` at the third prefix where the f32 accumulator
/// keeps `2^24`.
#[test]
fn float_cumsum_accumulates_at_the_default_accumulator_in_both_lanes() {
    let rows: [(Prim, &[f64]); 4] = [
        (
            Prim::F32,
            &[16_777_216.0, 1.0, 1.0, 1.0, 0.1, 0.2, 0.3, 1e-8],
        ),
        (Prim::F64, &[9_007_199_254_740_992.0, 1.0, 1.0, 0.1, 0.2]),
        (Prim::F16, &[2048.0, 1.0, 1.0, 1.0, 0.1, 0.3]),
        (Prim::Bf16, &[256.0, 1.0, 0.5, 0.25, 1.0, 3.0]),
    ];
    for (prim, values) in rows {
        let source = format!("out = cumsum({}, 0)\n", float_tensor(prim, values));
        let expected = float_cumsum_reference(prim, values);
        assert_lanes(&source, prim.name(), "out", prim, &expected);
    }
}

#[test]
fn f32_cumsum_from_two_to_the_24_stays_at_two_to_the_24() {
    let source = format!(
        "out = cumsum({}, 0)\n",
        float_tensor(Prim::F32, &[16_777_216.0, 1.0, 1.0, 1.0])
    );
    let expected = floats(Prim::F32, &[16_777_216.0; 4]);
    assert_lanes(&source, "cumsum-f32-2^24", "out", Prim::F32, &expected);
}

// ---- clamp (chelis#2972, chelis#2973) ----

/// Evidentiary status: REGRESSION TEST. At 08939bc0e eval returns 2^53,
/// a value below the lower bound.
#[test]
fn i64_clamp_at_an_inclusive_bound_returns_the_exact_stored_input() {
    let x = [ABOVE_F64, ABOVE_F64 + 3, ABOVE_F64 - 2];
    let lo = [ABOVE_F64; 3];
    let hi = [ABOVE_F64 + 2; 3];
    let source = format!(
        "out = clamp({}, {}, {})\n",
        int_tensor("i64", &x),
        int_tensor("i64", &lo),
        int_tensor("i64", &hi)
    );
    let expected = ints(&[ABOVE_F64, ABOVE_F64 + 2, ABOVE_F64]);
    assert_lanes(&source, "clamp-i64", "out", Prim::Int64, &expected);
}

/// Negative control for the trap rows: valid bounds, both shapes, and a
/// signed zero at an inclusive bound keep the stored input.
#[test]
fn float_clamp_with_valid_bounds_selects_in_both_lanes() {
    for prim in [Prim::F32, Prim::F64, Prim::F16, Prim::Bf16] {
        let source = format!(
            "out = clamp({}, {}, {})\nscalar = clamp({}, sum({}, 0), sum({}, 0))\n",
            float_tensor(prim, &[-3.0, -0.0, 0.5, 9.0]),
            float_tensor(prim, &[-1.0, 0.0, 0.0, 0.0]),
            float_tensor(prim, &[1.0, 1.0, 1.0, 2.0]),
            float_tensor(prim, &[-3.0, 0.25, 9.0]),
            float_tensor(prim, &[-1.0]),
            float_tensor(prim, &[1.0]),
        );
        let expected = floats(prim, &[-1.0, -0.0, 0.5, 2.0]);
        assert_lanes(&source, prim.name(), "out", prim, &expected);
        let expected = floats(prim, &[-1.0, 0.25, 1.0]);
        assert_lanes(&source, prim.name(), "scalar", prim, &expected);
    }
}

/// Evidentiary status: REGRESSION TEST. At 08939bc0e eval panics in Rust
/// `f64::clamp` (exit 101) on every row below. chelis#3010: both lanes print
/// the position on its own line, then the [04-NUM-9] line at the operand
/// dtype; neither used to print the canonical line.
#[test]
fn clamp_lower_above_upper_traps_domain_at_its_position_in_both_lanes() {
    let detail = "clamp lower bound exceeds upper bound at row-major position 2";
    for prim in [Prim::F32, Prim::F64, Prim::F16, Prim::Bf16] {
        let source = format!(
            "out = clamp({}, {}, {})\n",
            float_tensor(prim, &[1.0, 2.0, 3.0, 4.0]),
            float_tensor(prim, &[0.0, 0.0, 5.0, 6.0]),
            float_tensor(prim, &[1.0, 1.0, 1.0, 1.0]),
        );
        assert_lanes_trap_lines(&source, prim.name(), &[detail, &clamp_trap(prim.name())]);
    }
    for suffix in ["i64", "i32"] {
        let source = format!(
            "out = clamp({}, {}, {})\n",
            int_tensor(suffix, &[1, 2, 3, 4]),
            int_tensor(suffix, &[0, 0, 5, 6]),
            int_tensor(suffix, &[1, 1, 1, 1]),
        );
        assert_lanes_trap_lines(&source, suffix, &[detail, &clamp_trap(suffix)]);
    }
}

#[test]
fn clamp_rank_zero_bounds_trap_at_the_first_position_in_both_lanes() {
    let source = format!(
        "out = clamp({}, sum({}, 0), sum({}, 0))\n",
        int_tensor("i64", &[1, 2]),
        int_tensor("i64", &[3]),
        int_tensor("i64", &[0]),
    );
    assert_lanes_trap_lines(
        &source,
        "clamp-rank-zero",
        &[
            "clamp lower bound exceeds upper bound at row-major position 0",
            &clamp_trap("i64"),
        ],
    );
}

#[test]
fn clamp_nan_bound_traps_domain_at_its_position_in_both_lanes() {
    for prim in [Prim::F32, Prim::F64, Prim::F16, Prim::Bf16] {
        let zeros = float_tensor(prim, &[0.0, 0.0, 0.0]);
        let source = format!(
            "nan_lo = div({zeros}, {})\nout = clamp({}, nan_lo, {})\n",
            float_tensor(prim, &[1.0, 0.0, 1.0]),
            float_tensor(prim, &[1.0, 2.0, 3.0]),
            float_tensor(prim, &[5.0, 5.0, 5.0]),
        );
        assert_lanes_trap_lines(
            &source,
            prim.name(),
            &[
                "clamp bound is NaN at row-major position 1",
                &clamp_trap(prim.name()),
            ],
        );
    }
}

// ---- scatter-add (chelis#2972) ----

/// Evidentiary status: REGRESSION TEST. At 08939bc0e eval returns 2^53.
#[test]
fn i64_scatter_add_above_two_to_the_53_is_exact_in_both_lanes() {
    let source = format!(
        "out = scatter({}, to_tensor([0i64, 0i64]), {}, 0i32, \"add\")\n",
        int_tensor("i64", &[ABOVE_F64, 0, 0]),
        int_tensor("i64", &[1, 1]),
    );
    let expected = ints(&[ABOVE_F64 + 2, 0, 0]);
    assert_lanes(&source, "scatter-i64", "out", Prim::Int64, &expected);
}

/// The base leaf and the targeting updates combine through the canonical
/// adjacent-pair tree at the operand width: `((2^24 + 1) + 1)` at f32 stays
/// `2^24`. Evidentiary status: REGRESSION TEST; at 08939bc0e eval returns
/// `2^24 + 2`.
#[test]
fn float_scatter_add_uses_the_balanced_tree_at_operand_width_in_both_lanes() {
    let source = format!(
        "out = scatter({}, to_tensor([0i64, 0i64]), {}, 0i32, \"add\")\n",
        float_tensor(Prim::F32, &[16_777_216.0, 0.0]),
        float_tensor(Prim::F32, &[1.0, 1.0]),
    );
    let expected = floats(Prim::F32, &[16_777_216.0, 0.0]);
    assert_lanes(&source, "scatter-f32", "out", Prim::F32, &expected);
}

/// chelis#2972's f16 and bf16 witnesses, as the issue's update states them:
/// a tie at the storage width followed by a small addend. A binary64 running
/// sum keeps the addend and rounds the last cumsum prefix up to 32800 (f16) or
/// 33024 (bf16); the scatter-add tree `(base + 1) + (1 + 1)` at the operand
/// width gives 2050 (f16) or 258 (bf16), where one binary64 sum rounds 2051
/// (259) to 2052 (260). Evidentiary status: REGRESSION TEST for eval's binary64
/// funnel at 08939bc0e.
#[test]
fn half_precision_cumsum_and_scatter_add_witnesses_match_in_both_lanes() {
    let tiny = 0.000_976_562_5;
    for (prim, addend) in [(Prim::F16, 16.0), (Prim::Bf16, 128.0)] {
        let values = [32_768.0, addend, tiny];
        let source = format!("out = cumsum({}, 0)\n", float_tensor(prim, &values));
        let expected = float_cumsum_reference(prim, &values);
        assert_eq!(
            expected,
            floats(prim, &[32_768.0; 3]),
            "{}: the witness's reference stays at 32768",
            prim.name()
        );
        assert_lanes(&source, prim.name(), "out", prim, &expected);
    }
    for (prim, base) in [(Prim::F16, 2048.0), (Prim::Bf16, 256.0)] {
        let source = format!(
            "out = scatter({}, to_tensor([0i64, 0i64, 0i64, 1i64]), {}, 0i32, \"add\")\n",
            float_tensor(prim, &[base, 0.5, 0.0]),
            float_tensor(prim, &[1.0, 1.0, 1.0, 0.1]),
        );
        let add = |left: ScalarValue, right: ScalarValue| {
            float_binop(FloatBinOp::Add, left, right).expect("finite sum")
        };
        let one = float(prim, 1.0);
        let head = add(add(float(prim, base), one), add(one, one));
        let tail = add(float(prim, 0.5), float(prim, 0.1));
        let expected = vec![stored(head), stored(tail), stored(float(prim, 0.0))];
        assert_eq!(
            expected[0],
            stored(float(prim, base + 2.0)),
            "{}: the witness's tree gives base + 2",
            prim.name()
        );
        assert_lanes(&source, prim.name(), "out", prim, &expected);
    }
}

#[test]
fn i32_scatter_add_overflow_traps_in_both_lanes() {
    let source = format!(
        "out = scatter({}, to_tensor([0i64]), {}, 0i32, \"add\")\n",
        int_tensor("i32", &[i64::from(i32::MAX), 0]),
        int_tensor("i32", &[1]),
    );
    assert_lanes_trap(
        &source,
        "scatter-i32-overflow",
        "numeric trap: overflow in scatter at i32",
    );
}

// ---- softmax (chelis#2971) ----

/// Section 4.2's graph at `prim`, composed from the closed typed kernels:
/// max, subtract, `exp`, the f32 (or f64) default-accumulator balanced sum
/// stored at `prim`, and the division, each finalized at `prim`.
fn softmax_reference(
    prim: Prim,
    values: &[f64],
    rows: usize,
    cols: usize,
    axis: usize,
) -> Vec<Stored> {
    let x: Vec<ScalarValue> = values.iter().map(|&value| float(prim, value)).collect();
    let accumulator = if prim == Prim::F64 {
        Prim::F64
    } else {
        Prim::F32
    };
    let (lanes, extent) = if axis == 0 {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let at = |lane: usize, k: usize| {
        if axis == 0 {
            k * cols + lane
        } else {
            lane * cols + k
        }
    };
    let mut out = vec![Stored::Nan; values.len()];
    for lane in 0..lanes {
        let mut max = x[at(lane, 0)];
        for k in 1..extent {
            max = float_binop(FloatBinOp::Max, max, x[at(lane, k)]).unwrap();
        }
        let exps: Vec<ScalarValue> = (0..extent)
            .map(|k| {
                let shifted = float_binop(FloatBinOp::Sub, x[at(lane, k)], max).unwrap();
                float_unop(FloatUnOp::Exp, shifted).unwrap()
            })
            .collect();
        let mut level: Vec<ScalarValue> = exps
            .iter()
            .map(|value| float(accumulator, value.as_f64_lossy()))
            .collect();
        while level.len() > 1 {
            level = level
                .chunks(2)
                .map(|pair| match pair {
                    [left, right] => float_binop(FloatBinOp::Add, *left, *right).unwrap(),
                    [last] => *last,
                    _ => unreachable!(),
                })
                .collect();
        }
        let sum = float(prim, level[0].as_f64_lossy());
        for k in 0..extent {
            out[at(lane, k)] = stored(float_binop(FloatBinOp::Div, exps[k], sum).unwrap());
        }
    }
    out
}

/// The chelis#2971 4x8 probe input.
const SOFTMAX_4X8: [f64; 32] = [
    0.1, 1.7, -2.3, 0.33, 3.9, -0.77, 2.2, 1.05, -5.5, 0.6, 4.1, -1.9, 2.75, 0.01, -0.4, 1.3, 7.25,
    -3.5, 0.125, 2.5, -0.06, 5.5, -9.0, 1.1, 0.0, -1.0, 3.3, 6.6, -2.2, 0.45, 1.9, -4.4,
];

/// Evidentiary status: REGRESSION TEST. At 08939bc0e eval computes in f64
/// and narrows once, differing from this graph by up to 13 f32 ULP.
#[test]
fn reduced_and_single_precision_softmax_executes_the_per_primitive_graph() {
    for prim in [Prim::F32, Prim::F16, Prim::Bf16, Prim::F64] {
        let rows = SOFTMAX_4X8
            .chunks(8)
            .map(|row| {
                let items = row
                    .iter()
                    .map(|&value| literal(prim, value))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("[{items}]")
            })
            .collect::<Vec<_>>()
            .join(", ");
        for axis in [0usize, 1] {
            let source = format!("x = to_tensor([{rows}])\nout = softmax(x, {axis})\n");
            let evaluated = eval_root(&source, "out");
            assert_eq!(evaluated.data.prim(), prim);
            assert_eq!(evaluated.shape, vec![4, 8]);
            assert_eq!(
                storage_elements(&evaluated.data),
                softmax_reference(prim, &SOFTMAX_4X8, 4, 8, axis),
                "{} softmax axis {axis}",
                prim.name()
            );
        }
    }
}
