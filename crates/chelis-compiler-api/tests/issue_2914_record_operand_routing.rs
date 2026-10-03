//! chelis#2914: a tensor operation whose operand a definition computes from
//! a record value the host lane holds compiles to C.
//!
//! The tensor execution lane has no record carrier. Inlining an accessor such
//! as `col_days(a)` into a helper for `lt(col_days(a), col_days(b))` could not
//! read the record's field, the helper failed, and the whole comparison fell
//! back to host emission, which has no tensor arm: `chelis build` refused a
//! program `chelis eval` ran. Binding each operand to a name first already
//! compiled. The operand's typed producer fact is now Host, so it is
//! evaluated once, in source order, and enters the operation as a typed input
//! (computed_tensor_host_admission.md), the same program binding it by hand
//! compiles to.
//!
//! Oracle: each program compiles, runs against the `ownership-ledger` runtime
//! with every allocation finalized, and prints what `chelis eval` prints. The
//! negative twin makes the first operand's own validation fail; both lanes
//! must report that operand's failure, so hoisting it has kept its
//! evaluation, and its effect, in source order.
//!
//! Hoisting a later operand must not move it before an earlier one that does
//! work, so every such earlier operand is hoisted with it. The order twins
//! give an earlier operand that would otherwise stay in the operation's
//! helper a primitive trap, and the later record-reading operand a `fail` or
//! a `print`: both lanes report the trap, and the print never runs.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const PRELUDE: &str = "module Demo.Main\n\
type Col[n] =\n  | Col { days: tensor[n, i64] }\n\
def filled[n](like: &tensor[n, i64], value: i64) -> tensor[n, i64] = value |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))\n\
def first_false(flags: List[bool]) -> i64 = fold(fn (acc: (i64, i64), flag: bool) -> if gte(acc.1, 0i64) then acc else if flag then (add(acc.0, 1i64), -1i64) else (add(acc.0, 1i64), acc.0), (0i64, -1i64), flags).1\n\
def col[n](t: tensor[n, i64]) -> Col[n] = {\n  bad = first_false(to_list(gte(t, filled(t, -10i64))))\n  if gte(bad, 0i64) then fail(\"col: domain: out of range\") else Col { days: t }\n}\n\
def col_days[n](c: Col[n]) -> tensor[n, i64] = c.days\n";

struct Case {
    name: &'static str,
    body: &'static str,
}

const POSITIVE: &[Case] = &[
    // The issue's program.
    Case {
        name: "comparison_of_accessor_results",
        body: "def before[n](a: Col[n], b: Col[n]) -> tensor[n, bool] = lt(col_days(a), col_days(b))\ndef run(seed: i64) -> List[bool] = to_list(before(col(to_tensor([1i64, 5i64])), col(to_tensor([2i64, 3i64]))))\nresult = run(0i64)\n",
    },
    // The same refusal hit integer arithmetic over accessor results.
    Case {
        name: "floor_div_of_accessor_results",
        body: "def ratio[n](a: Col[n], b: Col[n]) -> tensor[n, i64] = floor_div(col_days(a), col_days(b))\ndef run(seed: i64) -> List[i64] = to_list(ratio(col(to_tensor([7i64, -7i64])), col(to_tensor([2i64, 2i64]))))\nresult = run(0i64)\n",
    },
    // One operand reads the record and the other is a plain tensor.
    Case {
        name: "one_accessor_operand",
        body: "def shifted[n](a: Col[n], t: &tensor[n, i64]) -> tensor[n, bool] = gt(col_days(a), t)\ndef run(seed: i64) -> List[bool] = to_list(shifted(col(to_tensor([1i64, 5i64])), to_tensor([2i64, 3i64])))\nresult = run(0i64)\n",
    },
    // A projection inside an inlined validating accessor, and a helper
    // whose argument is that projection.
    Case {
        name: "projections_inside_a_validating_accessor",
        body: "def before[n](a: Col[n], b: Col[n]) -> tensor[n, bool] = lt(checked_days(a), checked_days(b))\ndef run(seed: i64) -> List[bool] = to_list(before(col(to_tensor([1i64, 5i64])), col(to_tensor([2i64, 3i64]))))\nresult = run(0i64)\n",
    },
    // The control the issue names: the operands bound to names first.
    Case {
        name: "operands_bound_to_names",
        body: "def before[n](a: Col[n], b: Col[n]) -> tensor[n, bool] = {\n  x = col_days(a)\n  y = col_days(b)\n  lt(x, y)\n}\ndef run(seed: i64) -> List[bool] = to_list(before(col(to_tensor([1i64, 5i64])), col(to_tensor([2i64, 3i64]))))\nresult = run(0i64)\n",
    },
    // The operand's type, not its spelling, classifies it: projections of a
    // tuple of records, one bound to a name first, and an `if` operand.
    Case {
        name: "projections_of_a_tuple_of_records",
        body: "def before[n](p: (Col[n], Col[n])) -> tensor[n, bool] = lt(col_days(p.0), col_days(p.1))\ndef run(seed: i64) -> List[bool] = to_list(before((col(to_tensor([1i64, 5i64])), col(to_tensor([2i64, 3i64])))))\nresult = run(0i64)\n",
    },
    Case {
        name: "tuple_projection_bound_to_a_name",
        body: "def before[n](p: (Col[n], Col[n])) -> tensor[n, bool] = {\n  a = p.0\n  lt(col_days(a), col_days(p.1))\n}\ndef run(seed: i64) -> List[bool] = to_list(before((col(to_tensor([1i64, 5i64])), col(to_tensor([2i64, 3i64])))))\nresult = run(0i64)\n",
    },
    Case {
        name: "branch_operand",
        body: "def before[n](a: Col[n], t: &tensor[n, i64]) -> tensor[n, bool] = lt(if true then col_days(a) else col_days(a), t)\ndef run(seed: i64) -> List[bool] = to_list(before(col(to_tensor([1i64, 5i64])), to_tensor([2i64, 3i64])))\nresult = run(0i64)\n",
    },
];

/// Each program's first operand fails its own validation.
const FIRST_OPERAND_FAILS: &[Case] = &[
    Case {
        name: "first_operand_fails",
        body: "def before[n](a: Col[n], b: Col[n]) -> tensor[n, bool] = lt(checked_days(a), checked_days(b))\ndef run(seed: i64) -> List[bool] = to_list(before(col(to_tensor([1i64, -5i64])), col(to_tensor([2i64, 3i64]))))\nresult = run(0i64)\n",
    },
    Case {
        name: "first_tuple_projection_fails",
        body: "def before[n](p: (Col[n], Col[n])) -> tensor[n, bool] = lt(checked_days(p.0), checked_days(p.1))\ndef run(seed: i64) -> List[bool] = to_list(before((col(to_tensor([1i64, -5i64])), col(to_tensor([2i64, 3i64])))))\nresult = run(0i64)\n",
    },
    Case {
        name: "branch_operand_fails",
        body: "def before[n](a: Col[n], t: &tensor[n, i64]) -> tensor[n, bool] = lt(if true then checked_days(a) else col_days(a), t)\ndef run(seed: i64) -> List[bool] = to_list(before(col(to_tensor([1i64, -5i64])), to_tensor([2i64, 3i64])))\nresult = run(0i64)\n",
    },
];

/// An earlier operand whose addition overflows, before a record-reading
/// operand that also fails or prints. Only the earlier failure may be seen.
const CHECKED_DAYS: &str = "def checked_days[n](c: Col[n]) -> tensor[n, i64] = {\n  bad = first_false(to_list(gte(c.days, filled(c.days, 0i64))))\n  if gte(bad, 0i64) then fail(\"checked_days: domain: negative day\") else c.days\n}\n";
const NOISY_DAYS: &str = "def noisy_days[n](c: Col[n]) -> tensor[n, i64] = {\n  shown = print(to_list(c.days))\n  c.days\n}\n";
const ORDER_TWINS: &[(&str, &str, &str)] = &[
    (
        "earlier_trap_before_a_failing_operand",
        CHECKED_DAYS,
        "def before[n](b: Col[n], t: &tensor[n, i64]) -> tensor[n, bool] = lt(add(t, filled(t, 9223372036854775807i64)), checked_days(b))\ndef run(seed: i64) -> List[bool] = to_list(before(col(to_tensor([-1i64, 3i64])), to_tensor([1i64, 2i64])))\nresult = run(0i64)\n",
    ),
    (
        "earlier_trap_before_a_failing_branch_operand",
        CHECKED_DAYS,
        "def before[n](b: Col[n], t: &tensor[n, i64]) -> tensor[n, bool] = lt(add(t, filled(t, 9223372036854775807i64)), if true then checked_days(b) else col_days(b))\ndef run(seed: i64) -> List[bool] = to_list(before(col(to_tensor([-1i64, 3i64])), to_tensor([1i64, 2i64])))\nresult = run(0i64)\n",
    ),
    (
        "earlier_trap_before_a_printing_operand",
        NOISY_DAYS,
        "def before[n](b: Col[n], t: &tensor[n, i64]) -> tensor[n, bool] = lt(add(t, filled(t, 9223372036854775807i64)), noisy_days(b))\ndef run(seed: i64) -> List[bool] = to_list(before(col(to_tensor([7i64, 8i64])), to_tensor([1i64, 2i64])))\nresult = run(0i64)\n",
    ),
];
/// The same order through an operand the host lane keeps for its list map.
const CHECKED: &str =
    "def checked(v: i64) -> i64 = if lt(v, 0i64) then fail(\"checked: domain: negative\") else v\n";
const HOST_LANE_TWIN: (&str, &str, &str) = (
    "earlier_trap_before_a_host_lane_operand",
    CHECKED,
    "def run(xs: List[i64], ys: List[i64]) -> List[i64] = {\n  x = to_tensor(xs)\n  to_list(add(add(x, filled(x, 9223372036854775807i64)), to_tensor(map(checked, ys))))\n}\nresult = run([1i64, 2i64], [-1i64, 3i64])\n",
);
const EARLIER_TRAP: &str = "numeric trap: overflow in add at i64";

fn source(case: &Case) -> String {
    format!("{PRELUDE}{CHECKED_DAYS}{}", case.body)
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
    let source = source(case);
    catch_unwind(AssertUnwindSafe(|| {
        let expected = evaluated(&source);
        let generated = ownership_support::emit(&source, case.name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    }))
    .map_err(|payload| {
        let head: String = panic_message(payload).chars().take(800).collect();
        format!("{}:\n{source}\n  -> {head}", case.name)
    })
}

// REGRESSION TEST. On `4a53cc7bd` the C build refused the first four
// programs with `no tensor emission arm for this op`, and on `637c680cc`
// the last three.
#[test]
fn record_reading_operands_compile_as_eval_runs_them() {
    let failures: Vec<String> = POSITIVE
        .iter()
        .filter_map(|case| check_positive(case).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn a_failing_operand_fails_first_in_both_lanes() {
    let mut failures = Vec::new();
    for case in FIRST_OPERAND_FAILS {
        let source = source(case);
        let refused = eval(request(&source))
            .expect_err("the evaluator must refuse")
            .errors
            .iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect::<Vec<_>>()
            .join("\n");
        if !refused.contains("checked_days: domain: negative day") {
            failures.push(format!("{}: eval reported {refused}", case.name));
        }
        let generated = ownership_support::emit(&source, case.name);
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        if stderr.trim_end() != "checked_days: domain: negative day" {
            failures.push(format!("{}: C reported {stderr:?}", case.name));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// REGRESSION TEST. Before every earlier operand that does work was hoisted
// with the later one, compiled C reported `checked_days`'s failure, and
// printed `[7, 8]` before trapping, while `chelis eval` reported the
// addition's overflow. The host-lane twin reported `checked`'s failure.
#[test]
fn an_earlier_operand_runs_before_a_hoisted_later_one() {
    let mut failures = Vec::new();
    for (name, helper, body) in ORDER_TWINS.iter().chain([&HOST_LANE_TWIN]) {
        let source = format!("{PRELUDE}{helper}{body}");
        let refused = eval(request(&source))
            .expect_err("the evaluator must refuse")
            .errors
            .iter()
            .map(|diagnostic| diagnostic.message.clone())
            .collect::<Vec<_>>()
            .join("\n");
        if !refused.contains(EARLIER_TRAP) {
            failures.push(format!("{name}: eval reported {refused}"));
        }
        let generated = ownership_support::emit(&source, name);
        let (stdout, stderr) = ownership_support::run_failure_output(&generated, "");
        if stderr.trim_end() != EARLIER_TRAP || !stdout.is_empty() {
            failures.push(format!(
                "{name}: C printed {stdout:?} and reported {stderr:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
