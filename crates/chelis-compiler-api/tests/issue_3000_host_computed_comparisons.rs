//! chelis#3000: a tensor comparison whose operands are computed on the host
//! compiles to C.
//!
//! A comparison of two calls that each return a tensor, such as
//! `eq(sorted(t), sorted(t))`, reaches the C host lane with tensor operands,
//! and that lane had an element-wise arm for arithmetic and for `and` and
//! `or` but none for the six comparisons: `chelis build` refused a program
//! `chelis eval` ran. Each comparison now has an element-wise host arm that
//! applies the same operand agreement check as arithmetic and writes a bool
//! tensor, with [05-OP-36]'s float rule (signed zeros equal).
//!
//! Oracle: each program compiles, runs against the `ownership-ledger`
//! runtime with every allocation finalized, and prints what `chelis eval`
//! prints. The negative twin compares operands of different lengths; both
//! lanes must refuse it.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const PRELUDE: &str = "module Demo.Main\n\
def rebuilt[n](t: &tensor[n, i64]) -> tensor[n, i64] = to_tensor(to_list(t))\n\
def rebuilt_f64[n](t: &tensor[n, f64]) -> tensor[n, f64] = to_tensor(to_list(t))\n\
def left(seed: i64) -> List[i64] = [1i64, 2i64, 3i64, 2i64]\n\
def right(seed: i64) -> List[i64] = [3i64, 2i64, 1i64, 2i64]\n\
def nan(seed: i64) -> f64 = div(0.0f64, 0.0f64)\n\
def nans_left(seed: i64) -> List[f64] = [nan(0i64), 1.0f64, nan(0i64), -0.0f64, 2.0f64]\n\
def nans_right(seed: i64) -> List[f64] = [1.0f64, nan(0i64), nan(0i64), 0.0f64, 2.0f64]\n\
def rebuilt_i8[n](t: &tensor[n, i8]) -> tensor[n, i8] = to_tensor(to_list(t))\n\
def rebuilt_f16[n](t: &tensor[n, f16]) -> tensor[n, f16] = to_tensor(to_list(t))\n\
def checked[n](t: &tensor[n, i64], tag: string) -> tensor[n, i64] = if lt(index(to_list(t), 0i64), 0i64) then fail(tag) else rebuilt(t)\n";

const POSITIVE: &[Case] = &[
    Case {
        name: "lt_of_two_calls",
        body: "result = to_list(lt(rebuilt(to_tensor(left(0i64))), rebuilt(to_tensor(right(0i64)))))\n",
    },
    Case {
        name: "lte_of_two_calls",
        body: "result = to_list(lte(rebuilt(to_tensor(left(0i64))), rebuilt(to_tensor(right(0i64)))))\n",
    },
    Case {
        name: "gt_of_two_calls",
        body: "result = to_list(gt(rebuilt(to_tensor(left(0i64))), rebuilt(to_tensor(right(0i64)))))\n",
    },
    Case {
        name: "gte_of_two_calls",
        body: "result = to_list(gte(rebuilt(to_tensor(left(0i64))), rebuilt(to_tensor(right(0i64)))))\n",
    },
    Case {
        name: "eq_of_two_calls",
        body: "result = to_list(eq(rebuilt(to_tensor(left(0i64))), rebuilt(to_tensor(right(0i64)))))\n",
    },
    Case {
        name: "neq_of_two_calls",
        body: "result = to_list(neq(rebuilt(to_tensor(left(0i64))), rebuilt(to_tensor(right(0i64)))))\n",
    },
    // Inside a definition, as Std.Datetime.Business compares two counts.
    Case {
        name: "eq_inside_a_definition",
        body: "def same[n](t: &tensor[n, i64]) -> tensor[n, bool] = eq(rebuilt(t), rebuilt(t))\nresult = to_list(same(to_tensor(left(0i64))))\n",
    },
    // Equality of two host-computed bool tensors.
    Case {
        name: "eq_of_two_comparisons",
        body: "result = to_list(eq(lt(rebuilt(to_tensor(left(0i64))), rebuilt(to_tensor(right(0i64)))), gte(rebuilt(to_tensor(left(0i64))), rebuilt(to_tensor(right(0i64))))))\n",
    },
    // Floats: signed zeros compare equal.
    Case {
        name: "f64_signed_zeros",
        body: "result = to_list(eq(rebuilt_f64(to_tensor([0.0f64, -1.5f64, 2.5f64])), rebuilt_f64(to_tensor([-0.0f64, 2.5f64, -1.5f64]))))\nordered = to_list(lte(rebuilt_f64(to_tensor([0.0f64, -1.5f64, 2.5f64])), rebuilt_f64(to_tensor([-0.0f64, 3.5f64, -1.5f64]))))\n",
    },
    // A NaN on either side makes every comparison but `neq` false.
    Case {
        name: "lt_with_nans",
        body: "result = to_list(lt(rebuilt_f64(to_tensor(nans_left(0i64))), rebuilt_f64(to_tensor(nans_right(0i64)))))\n",
    },
    // A NaN on either side makes every comparison but `neq` false.
    Case {
        name: "lte_with_nans",
        body: "result = to_list(lte(rebuilt_f64(to_tensor(nans_left(0i64))), rebuilt_f64(to_tensor(nans_right(0i64)))))\n",
    },
    // A NaN on either side makes every comparison but `neq` false.
    Case {
        name: "gt_with_nans",
        body: "result = to_list(gt(rebuilt_f64(to_tensor(nans_left(0i64))), rebuilt_f64(to_tensor(nans_right(0i64)))))\n",
    },
    // A NaN on either side makes every comparison but `neq` false.
    Case {
        name: "gte_with_nans",
        body: "result = to_list(gte(rebuilt_f64(to_tensor(nans_left(0i64))), rebuilt_f64(to_tensor(nans_right(0i64)))))\n",
    },
    // A NaN on either side makes every comparison but `neq` false.
    Case {
        name: "eq_with_nans",
        body: "result = to_list(eq(rebuilt_f64(to_tensor(nans_left(0i64))), rebuilt_f64(to_tensor(nans_right(0i64)))))\n",
    },
    // A NaN on either side makes every comparison but `neq` false.
    Case {
        name: "neq_with_nans",
        body: "result = to_list(neq(rebuilt_f64(to_tensor(nans_left(0i64))), rebuilt_f64(to_tensor(nans_right(0i64)))))\n",
    },
    // Narrow dtypes compare in their own arms: i8 directly, f16 after the
    // exact widening to binary32.
    Case {
        name: "i8_operands",
        body: "result = to_list(lte(rebuilt_i8(to_tensor([1i8, -127i8, 127i8, 0i8])), rebuilt_i8(to_tensor([1i8, 127i8, -127i8, -1i8]))))\nsame = to_list(eq(rebuilt_i8(to_tensor([1i8, -127i8])), rebuilt_i8(to_tensor([1i8, 127i8]))))\n",
    },
    Case {
        name: "f16_operands",
        body: "result = to_list(lt(rebuilt_f16(to_tensor([1.5f16, -2.0f16, 0.0f16, 65504.0f16])), rebuilt_f16(to_tensor([1.5f16, -1.0f16, -0.0f16, 65504.0f16]))))\nsame = to_list(eq(rebuilt_f16(to_tensor([0.0f16, 0.5f16])), rebuilt_f16(to_tensor([-0.0f16, 0.25f16]))))\n",
    },
];

const LENGTHS_DIFFER: Case = Case {
    name: "operands_of_different_lengths",
    body: "result = to_list(eq(rebuilt(to_tensor(left(0i64))), rebuilt(to_tensor([1i64, 2i64]))))\n",
};

struct Case {
    name: &'static str,
    body: &'static str,
}

fn source(case: &Case) -> String {
    format!("{PRELUDE}{}", case.body)
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

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn check_positive(case: &Case) -> Result<(), String> {
    catch_unwind(AssertUnwindSafe(|| {
        let source = source(case);
        let expected = evaluated(&source);
        let generated = ownership_support::emit(&source, case.name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    }))
    .map_err(|payload| {
        let head: String = panic_message(payload).chars().take(800).collect();
        format!("{}:\n{}\n  -> {head}", case.name, source(case))
    })
}

// REGRESSION TEST. On `15ea7b927` the C build refused every positive program:
// `no tensor emission arm` for the ordered comparisons, and `the host scalar
// lane has no comparison for this operand pair` for `eq` and `neq`.
#[test]
fn host_computed_comparisons_compile_as_eval_runs_them() {
    let failures: Vec<String> = POSITIVE
        .iter()
        .filter_map(|case| check_positive(case).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn host_computed_comparisons_of_different_lengths_fail_in_both_lanes() {
    let refused = eval(request(&source(&LENGTHS_DIFFER))).expect_err("the evaluator must refuse");
    let refused = refused
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    // spec/04-type-system.md section 4.7: one `Domain` trap in `eq`, with
    // the same context line, in both lanes.
    let trap = "eq operands disagree at axis 0: lhs [4] has 4, rhs [2] has 2\n\
                numeric trap: domain in eq at i64";
    assert!(refused.contains(trap), "{refused}");
    let generated = ownership_support::emit(&source(&LENGTHS_DIFFER), LENGTHS_DIFFER.name);
    let stderr = ownership_support::run_failure_stderr(&generated, "");
    assert!(stderr.contains(trap), "{stderr}");
}

/// Operands that fail evaluate in source order: with both operands failing
/// the first one's failure is reported, and with only the second failing its
/// failure is.
const TRAPS: &[(&str, &str, &str)] = &[
    (
        "both_operands_fail",
        "result = to_list(lt(checked(to_tensor([-1i64, 2i64]), \"first operand\"), checked(to_tensor([-3i64, 4i64]), \"second operand\")))\n",
        "first operand",
    ),
    (
        "second_operand_fails",
        "result = to_list(eq(checked(to_tensor([1i64, 2i64]), \"first operand\"), checked(to_tensor([-3i64, 4i64]), \"second operand\")))\n",
        "second operand",
    ),
];

#[test]
fn failing_operands_fail_in_source_order_in_both_lanes() {
    let mut failures = Vec::new();
    for &(name, body, expected) in TRAPS {
        let case = Case { name, body };

        let refused = match eval(request(&source(&case))) {
            Ok(_) => {
                failures.push(format!("{name}: the evaluator returned a result"));
                continue;
            }
            Err(refused) => refused
                .errors
                .iter()
                .map(|diagnostic| diagnostic.message.clone())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        let generated = ownership_support::emit(&source(&case), name);
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        let other = if expected == "first operand" {
            "second operand"
        } else {
            "first operand"
        };
        if !refused.contains(expected) || refused.contains(other) {
            failures.push(format!("{name}: eval reported {refused}"));
        }
        if !stderr.contains(expected) || stderr.contains(other) {
            failures.push(format!("{name}: compiled C reported {stderr}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
