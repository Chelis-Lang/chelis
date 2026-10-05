//! chelis#2076: `mod`, `bitand`, `bitor`, `bitxor`, `shl` and `shr` over two
//! signed-integer tensors run in both lanes.
//!
//! [05-OP-64] gives `mod` two same-shaped, same-dtype tensors, truncating with
//! the dividend's sign and trapping a zero divisor, and [05-OP-47] gives the
//! bitwise and shift operations the same surface at the declared width, with
//! a negative shift count trapping. Each case runs at i8, i16, i32 and i64,
//! through a definition (the typed graph lane) and over operands computed on
//! the host (the C host lane), and the compiled program must print what
//! `chelis eval` prints with every allocation finalized.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const OPS: [&str; 6] = ["mod", "bitand", "bitor", "bitxor", "shl", "shr"];
const INTS: [(&str, u32); 4] = [("i8", 8), ("i16", 16), ("i32", 32), ("i64", 64)];

/// One element spelled at `dtype`. The minimum is written as an expression,
/// since its magnitude is not a literal of the dtype.
fn element(value: i64, dtype: &str, width: u32) -> String {
    let minimum = if width == 64 {
        i64::MIN
    } else {
        -(1i64 << (width - 1))
    };
    if value == minimum {
        format!("sub({}{dtype}, 1{dtype})", value + 1)
    } else {
        format!("{value}{dtype}")
    }
}

fn list(values: &[i64], dtype: &str, width: u32) -> String {
    let items: Vec<String> = values.iter().map(|v| element(*v, dtype, width)).collect();
    format!("[{}]", items.join(", "))
}

/// Operands that reach each atom's boundary cases: signs, the minimum, the
/// maximum, and shift counts at and past the width.
fn operands(op: &str, width: u32) -> (Vec<i64>, Vec<i64>) {
    let max = if width == 64 {
        i64::MAX
    } else {
        (1i64 << (width - 1)) - 1
    };
    let min = -max - 1;
    let w = i64::from(width);
    match op {
        "mod" => (
            vec![7, -7, 7, -7, min, max, min, 0],
            vec![3, 3, -3, -3, -1, 7, max, 5],
        ),
        "shl" | "shr" => (
            vec![1, -1, max, min, 5, -5, 3, -8],
            vec![0, 1, 1, 1, w - 1, w, w.min(63), 2],
        ),
        _ => (
            vec![0, -1, max, min, 5, -6, 85, -86],
            vec![0, 1, min, max, 3, 3, 15, -16],
        ),
    }
}

fn program(op: &str, dtype: &str, width: u32, lhs: &[i64], rhs: &[i64]) -> String {
    let (lhs, rhs) = (list(lhs, dtype, width), list(rhs, dtype, width));
    format!(
        "module Demo.Main\n\
         def rebuilt[n](t: &tensor[n, {dtype}]) -> tensor[n, {dtype}] = to_tensor(to_list(t))\n\
         def apply[n](x: tensor[n, {dtype}], y: tensor[n, {dtype}]) -> tensor[n, {dtype}] = {op}(x, y)\n\
         graph = to_list(apply(to_tensor({lhs}), to_tensor({rhs})))\n\
         host = to_list({op}(rebuilt(to_tensor({lhs})), rebuilt(to_tensor({rhs}))))\n"
    )
}

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

fn evaluated(source: &str) -> String {
    let result = eval(request(source))
        .unwrap_or_else(|error| panic!("evaluator rejected the case: {error:?}"));
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}\n",
                root.name.as_deref().expect("named root"),
                root.display.as_deref().expect("rendered root")
            )
        })
        .collect()
}

fn refusal(source: &str) -> String {
    match eval(request(source)) {
        Ok(_) => String::new(),
        Err(refused) => refused
            .errors
            .iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

// REGRESSION TEST. On `4bb166024` the checker refused every one of these
// programs ("mod requires matching integer arguments, got tensor[..] and
// tensor[..]"), and with the checker repaired the evaluator's host lane
// refused the bitwise operations and `chelis build` refused all six over
// host-computed operands ("no tensor emission arm").
#[test]
fn integer_tensor_operations_compile_as_eval_runs_them() {
    let mut failures = Vec::new();
    for op in OPS {
        for (dtype, width) in INTS {
            let (lhs, rhs) = operands(op, width);
            let source = program(op, dtype, width, &lhs, &rhs);
            let name = format!("{op}_{dtype}");
            let outcome = catch_unwind(AssertUnwindSafe(|| {
                let expected = evaluated(&source);
                let generated = ownership_support::emit(&source, &name);
                let (summary, stdout) = ownership_support::run_program(&generated);
                ownership_support::balanced(&summary);
                assert_eq!(stdout, expected, "compiled output differs from eval");
            }));
            if let Err(payload) = outcome {
                let head: String = panic_message(payload).chars().take(600).collect();
                failures.push(format!("{name}:\n{source}  -> {head}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// The values [05-OP-64] and [05-OP-47] fix, at i8, so the agreement above is
/// not two lanes sharing one wrong answer.
#[test]
fn integer_tensor_operations_give_the_atom_values() {
    let cases = [
        ("mod", "[1, -1, 1, -1, 0, 1, -1, 0]"),
        ("bitand", "[0, 1, 0, 0, 1, 2, 5, -96]"),
        ("bitor", "[0, -1, -1, -1, 7, -5, 95, -6]"),
        ("bitxor", "[0, -2, -1, -1, 6, -7, 90, 90]"),
        ("shl", "[1, -2, -2, 0, -128, 0, 0, -32]"),
        ("shr", "[1, -1, 63, -64, 0, -1, 0, -2]"),
    ];
    for (op, expected) in cases {
        let (lhs, rhs) = operands(op, 8);
        let shown = evaluated(&program(op, "i8", 8, &lhs, &rhs));
        assert_eq!(
            shown,
            format!("graph = {expected}\nhost = {expected}\n"),
            "{op}"
        );
    }
}

#[test]
fn a_zero_divisor_and_a_negative_shift_trap_in_both_lanes() {
    for (op, expected) in [
        ("mod", "numeric trap: division by zero in mod at i32"),
        ("shl", "shift amount must be non-negative, got -1"),
        ("shr", "shift amount must be non-negative, got -1"),
    ] {
        let divisor = if op == "mod" { 0 } else { -1 };
        let source = program(op, "i32", 32, &[1, 2], &[1, divisor]);
        let refused = refusal(&source);
        assert!(refused.contains(expected), "{op}: eval reported {refused}");
        let generated = ownership_support::emit(&source, &format!("{op}_trap"));
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        assert!(
            stderr.contains(expected),
            "{op}: compiled C reported {stderr}"
        );
    }
}

#[test]
fn integer_tensors_of_different_lengths_fail_in_both_lanes() {
    // The lengths come from host lists, so only the run time can compare them.
    let source = "module Demo.Main\n\
                  def three(seed: i64) -> List[i32] = [1i32, 2i32, 3i32]\n\
                  def two(seed: i64) -> List[i32] = [1i32, 2i32]\n\
                  result = to_list(bitxor(to_tensor(three(0i64)), to_tensor(two(0i64))))\n";
    // spec/04 section 4.7: both lanes trap `Domain` in `bitxor` with the
    // same context line (chelis#3107).
    let failure = "bitxor operands disagree at axis 0: lhs [3] has 3, rhs [2] has 2\n\
                   numeric trap: domain in bitxor at i64";
    let refused = refusal(source);
    assert!(refused.contains(failure), "eval reported {refused}");
    let generated = ownership_support::emit(source, "lengths_differ");
    let stderr = ownership_support::run_failure_stderr(&generated, "");
    assert!(stderr.contains(failure), "{stderr}");
}
