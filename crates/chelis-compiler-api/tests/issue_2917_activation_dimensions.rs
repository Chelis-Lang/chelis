//! chelis#2917, chelis#2906, chelis#2893: an inlined generic definition's
//! dimension binders are instantiated per activation.
//!
//! The checker generalizes each definition over its own dimension binders
//! (`d55`, `d57`) and records them on every node of its body. Inlining the
//! body at a call, or lowering it as its own function, must instantiate them
//! with the axes this activation runs with, as it instantiates type and
//! precision variables (runtime_extents.md C2.2,
//! generic_tensor_actualization.md). Without that, two values one signature
//! ties to one binder kept two callee-private names, and two calls of one
//! helper shared one. Compiled C then refused a generic call whose arguments
//! came from inlined accessors (chelis#2917), panicked on a binder resolved
//! to two extents (chelis#2906), and refused `and` over comparisons of two
//! parameters sharing a binder (chelis#2893), while `chelis eval` ran each
//! program.
//!
//! Oracle for the positive corpus: the compiled program runs against the
//! `ownership-ledger` runtime, every allocation is finalized with no live
//! owner left, and stdout equals the evaluator's rendering of the same roots.
//! Each issue's program runs with static extents and again with extents only
//! known at run time.
//!
//! The negative twins give the run-time form two different lengths where the
//! signature declares one binder. Both lanes must still fail, with the same
//! entry guard where one observes the binder, so instantiation never turns a
//! claimed extent into a proved one.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const FILLED: &str = "def filled[n](like: &tensor[n, i64], value: i64) -> tensor[n, i64] = value |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))\n";

const FIRST_FALSE: &str = "def first_false(flags: List[bool]) -> i64 = fold(fn (acc: (i64, i64), flag: bool) -> if gte(acc.1, 0i64) then acc else if flag then (add(acc.0, 1i64), -1i64) else (add(acc.0, 1i64), acc.0), (0i64, -1i64), flags).1\n";

/// chelis#2917: a column's two fields read through inlined accessors, then
/// passed to a generic helper that ties both arguments to one binder.
const INSTANTS: &str = "type Inst[n] =\n  | Inst { secs: tensor[n, i64], nanos: tensor[n, i64] }\n\
def inst[n](s: tensor[n, i64], ns: tensor[n, i64]) -> Inst[n] = {\n  bad = first_false(to_list(gte(ns, filled(ns, 0i64))))\n  if gte(bad, 0i64) then fail(\"inst: domain: negative nanoseconds\") else Inst { secs: s, nanos: ns }\n}\n\
def secs_of[n](column: Inst[n]) -> tensor[n, i64] = column.secs\n\
def nanos_of[n](column: Inst[n]) -> tensor[n, i64] = column.nanos\n\
def parts[n](column: Inst[n]) -> (tensor[n, i64], tensor[n, i64]) = (secs_of(column), nanos_of(column))\n\
def combined[n](a: &tensor[n, i64], b: &tensor[n, i64], up: bool) -> (tensor[n, i64], tensor[n, bool]) = if up then (add(a, b), eq(a, a)) else (sub(a, b), eq(a, a))\n\
def summed[n](is: Inst[n]) -> tensor[n, i64] = {\n  (a, b) = parts(is)\n  (total, flags) = combined(a, b, true)\n  total\n}\n";

/// chelis#2906: two columns built by a validating constructor, read through
/// an accessor and combined with a helper result.
const COLUMNS: &str = "type Col[n] =\n  | Col { days: tensor[n, i64] }\n\
def col[n](t: tensor[n, i64]) -> Col[n] = {\n  bad = first_false(to_list(gte(t, filled(t, -10i64))))\n  if gte(bad, 0i64) then fail(\"col: domain: out of range\") else Col { days: t }\n}\n\
def col_days[n](c: Col[n]) -> tensor[n, i64] = c.days\n\
def gap[n](low_value: i64, a: Col[n], b: Col[n]) -> tensor[n, i64] = {\n  first = col_days(a)\n  last = col_days(b)\n  low = filled(first, low_value)\n  sub(last, low)\n}\n";

/// chelis#2893: `and` over comparisons of two parameters sharing a binder,
/// each with a helper result shaped like it.
const CHECK: &str = "def zeros_like[n](like: &tensor[n, i64]) -> tensor[n, i64] = 0i64 |> scalar_to_tensor |> insert(0i32, shape(like, 0i32))\n\
def check[n](a: &tensor[n, i64], b: &tensor[n, i64]) -> List[bool] = to_list(and(gte(a, zeros_like(a)), gte(b, zeros_like(b))))\n";

/// A helper's `where` over results shaped like one parameter, compared with
/// a sibling parameter of the same binder. The checker's caller-side label
/// for the helper result is the caller's `n`, which a callee's authored `n`
/// is not, however both are spelled.
const LENGTHS: &str = "def lengths[n](month: &tensor[n, i64]) -> tensor[n, i64] = where(eq(month, filled(month, 2i64)), filled(month, 28i64), filled(month, 31i64))\n\
def fits[n](day: &tensor[n, i64], month: &tensor[n, i64]) -> tensor[n, bool] = lte(day, lengths(month))\n";

/// A record accessor inlined into a validated update, whose binder occurs
/// only as the record's dimension argument and the update's tensor formal.
const UPDATE: &str = "type Ds[n] =\n  | Ds { days: tensor[n, i64] }\n\
def from_days[n](t: tensor[n, i64]) -> Ds[n] = {\n  bad = first_false(to_list(gte(t, filled(t, -1000i64))))\n  if gte(bad, 0i64) then fail(\"from_days: domain: out of range\") else Ds { days: t }\n}\n\
def epoch[n](ds: Ds[n]) -> tensor[n, i64] = ds.days\n\
def add_days[n](ds: Ds[n], days: &tensor[n, i64]) -> Ds[n] = {\n  start = epoch(ds)\n  fits = and(lte(days, sub(filled(start, 100i64), start)), gte(days, sub(filled(start, -100i64), start)))\n  bad = first_false(to_list(fits))\n  if gte(bad, 0i64) then fail(\"add_days: overflow: out of range\") else Ds { days: add(start, days) }\n}\n";

/// One kernel inlining the same helpers at several activations, as
/// `Std.Datetime.Columns` validates a date's fields.
const FIELDS: &str = "def euclid_mod[n](t: &tensor[n, i64], value: i64) -> tensor[n, i64] = sub(t, mul(floor_div(t, filled(t, value)), filled(t, value)))\n\
def divisible[n](t: &tensor[n, i64], value: i64) -> tensor[n, bool] = eq(euclid_mod(t, value), filled(t, 0i64))\n\
def within[n](t: &tensor[n, i64], low: i64, high: i64) -> tensor[n, bool] = and(gte(t, filled(t, low)), lte(t, filled(t, high)))\n\
def month_lengths[n](year: &tensor[n, i64], month: &tensor[n, i64]) -> tensor[n, i64] = {\n  leap = and(divisible(year, 4i64), or(not(divisible(year, 100i64)), divisible(year, 400i64)))\n  where(eq(month, filled(month, 2i64)), where(leap, filled(month, 29i64), filled(month, 28i64)), filled(month, 31i64))\n}\n\
def field_mask[n](year: &tensor[n, i64], month: &tensor[n, i64], day: &tensor[n, i64]) -> tensor[n, bool] = {\n  fields = and(within(year, -9999i64, 9999i64), within(month, 1i64, 12i64))\n  and(fields, and(gte(day, filled(day, 1i64)), lte(day, month_lengths(year, month))))\n}\n";

/// One generic body activated at sibling calls whose extents are known only
/// at run time and legitimately differ. Each activation's binder is its own,
/// so neither call's claim reaches the other.
const BUMP: &str = "def bump[n](t: &tensor[n, i64]) -> tensor[n, i64] = add(t, filled(t, 1i64))\n";

const BUMP_OWNED: &str =
    "def bump[n](t: tensor[n, i64]) -> tensor[n, i64] = add(t, filled(t, 1i64))\n";

/// Sibling activations of a body whose two parameters share one binder.
const PAIR_BUMP: &str = "def pair_bump[n](t: &tensor[n, i64], u: &tensor[n, i64]) -> tensor[n, i64] = add(add(t, u), filled(t, 1i64))\n";

const SIBLING_PAIRS: &str = "def run(xs: List[i64], ys: List[i64], zs: List[i64]) -> tensor[i64] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  c = to_tensor(zs)\n  add(sum(pair_bump(a, a), 0i32), sum(pair_bump(b, c), 0i32))\n}\n";

const SIBLINGS: &str = "def run(xs: List[i64], ys: List[i64]) -> tensor[i64] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  add(sum(bump(a), 0i32), sum(bump(b), 0i32))\n}\nresult = run([1i64, 2i64], [3i64, 4i64, 5i64])\n";

struct Case {
    name: &'static str,
    definitions: &'static [&'static str],
    run: &'static str,
}

impl Case {
    fn source(&self) -> String {
        let mut source = String::from("module Demo.Main\n");
        for definition in self.definitions {
            source.push_str(definition);
        }
        source.push_str(self.run);
        source
    }
}

const POSITIVE: &[Case] = &[
    Case {
        name: "chelis_2917_accessors_feed_a_generic_helper",
        definitions: &[FILLED, FIRST_FALSE, INSTANTS],
        run: "def run(seed: i64) -> List[i64] = to_list(summed(inst(to_tensor([1i64, 5i64]), to_tensor([2i64, 3i64]))))\nresult = run(0i64)\n",
    },
    Case {
        name: "chelis_2917_run_time_extents",
        definitions: &[FILLED, FIRST_FALSE, INSTANTS],
        run: "def run(xs: List[i64], ys: List[i64]) -> List[i64] = to_list(summed(inst(to_tensor(xs), to_tensor(ys))))\nresult = run([1i64, 5i64], [2i64, 3i64])\n",
    },
    Case {
        name: "chelis_2906_validated_columns",
        definitions: &[FILLED, FIRST_FALSE, COLUMNS],
        run: "def run(seed: i64) -> List[i64] = to_list(gap(1i64, col(to_tensor([1i64, 2i64])), col(to_tensor([5i64, 3i64]))))\nresult = run(0i64)\n",
    },
    Case {
        name: "chelis_2906_run_time_extents",
        definitions: &[FILLED, FIRST_FALSE, COLUMNS],
        run: "def run(xs: List[i64], ys: List[i64]) -> List[i64] = to_list(gap(1i64, col(to_tensor(xs)), col(to_tensor(ys))))\nresult = run([1i64, 2i64], [5i64, 3i64])\n",
    },
    Case {
        name: "chelis_2893_and_of_comparisons",
        definitions: &[CHECK],
        run: "def run(seed: i64) -> List[bool] = {\n  x = to_tensor([1i64, -2i64])\n  y = to_tensor([3i64, -4i64])\n  check(x, y)\n}\nresult = run(0i64)\n",
    },
    Case {
        name: "chelis_2893_run_time_extents",
        definitions: &[CHECK],
        run: "def run(xs: List[i64], ys: List[i64]) -> List[bool] = check(to_tensor(xs), to_tensor(ys))\nresult = run([1i64, -2i64], [3i64, -4i64])\n",
    },
    Case {
        name: "helper_result_compared_with_a_sibling_parameter",
        definitions: &[FILLED, LENGTHS],
        run: "def run(seed: i64) -> List[bool] = to_list(fits(to_tensor([1i64, 30i64]), to_tensor([2i64, 3i64])))\nresult = run(0i64)\n",
    },
    Case {
        name: "helper_result_compared_with_a_sibling_parameter_run_time",
        definitions: &[FILLED, LENGTHS],
        run: "def run(days: List[i64], months: List[i64]) -> List[bool] = to_list(fits(to_tensor(days), to_tensor(months)))\nresult = run([1i64, 30i64], [2i64, 3i64])\n",
    },
    Case {
        name: "record_accessor_in_a_validated_update",
        definitions: &[FILLED, FIRST_FALSE, UPDATE],
        run: "def run(seed: i64) -> List[i64] = to_list(epoch(add_days(from_days(to_tensor([0i64, 5i64, 7i64, 9i64])), to_tensor([1i64, -1i64, 0i64, 0i64]))))\nresult = run(0i64)\n",
    },
    Case {
        name: "helpers_inlined_at_several_activations",
        definitions: &[FILLED, FIELDS],
        run: "def run(years: List[i64], months: List[i64], days: List[i64]) -> List[bool] = to_list(field_mask(to_tensor(years), to_tensor(months), to_tensor(days)))\nresult = run([2024i64, 2023i64, 1900i64, 2000i64], [2i64, 2i64, 2i64, 13i64], [29i64, 29i64, 28i64, 1i64])\n",
    },
    Case {
        name: "sibling_activations_with_different_run_time_lengths",
        definitions: &[FILLED, BUMP],
        run: SIBLINGS,
    },
    Case {
        name: "sibling_activations_of_an_owned_parameter",
        definitions: &[FILLED, BUMP_OWNED],
        run: SIBLINGS,
    },
    Case {
        name: "three_sibling_activations",
        definitions: &[FILLED, BUMP],
        run: "def run(xs: List[i64], ys: List[i64], zs: List[i64]) -> tensor[i64] = {\n  a = to_tensor(xs)\n  b = to_tensor(ys)\n  c = to_tensor(zs)\n  add(add(sum(bump(a), 0i32), sum(bump(b), 0i32)), sum(bump(c), 0i32))\n}\nresult = run([1i64, 2i64], [3i64, 4i64, 5i64], [6i64])\n",
    },
    Case {
        name: "sibling_activations_of_a_shared_binder",
        definitions: &[FILLED, PAIR_BUMP, SIBLING_PAIRS],
        run: "result = run([1i64, 2i64], [3i64, 4i64, 5i64], [6i64, 7i64, 8i64])\n",
    },
];

/// Each run-time twin with one argument a different length, and the line
/// both lanes must report.
const NEGATIVE: &[(Case, &str)] = &[
    (
        Case {
            name: "chelis_2917_lengths_differ",
            definitions: &[FILLED, FIRST_FALSE, INSTANTS],
            run: "def run(xs: List[i64], ys: List[i64]) -> List[i64] = to_list(summed(inst(to_tensor(xs), to_tensor(ys))))\nresult = run([1i64, 5i64], [2i64, 3i64, 4i64])\n",
        },
        "extent `n`: s axis 0 = 2, ns axis 0 = 3",
    ),
    (
        Case {
            name: "chelis_2893_lengths_differ",
            definitions: &[CHECK],
            run: "def run(xs: List[i64], ys: List[i64]) -> List[bool] = check(to_tensor(xs), to_tensor(ys))\nresult = run([1i64, -2i64], [3i64, -4i64, 5i64])\n",
        },
        "extent `n`: a axis 0 = 2, b axis 0 = 3",
    ),
    (
        Case {
            name: "sibling_parameter_lengths_differ",
            definitions: &[FILLED, LENGTHS],
            run: "def run(days: List[i64], months: List[i64]) -> List[bool] = to_list(fits(to_tensor(days), to_tensor(months)))\nresult = run([1i64, 30i64], [2i64, 3i64, 4i64])\n",
        },
        "extent `n`: day axis 0 = 2, month axis 0 = 3",
    ),
    (
        Case {
            name: "several_activations_lengths_differ",
            definitions: &[FILLED, FIELDS],
            run: "def run(years: List[i64], months: List[i64], days: List[i64]) -> List[bool] = to_list(field_mask(to_tensor(years), to_tensor(months), to_tensor(days)))\nresult = run([2024i64, 2023i64, 1900i64, 2000i64], [2i64, 2i64, 2i64, 13i64], [29i64, 29i64, 28i64])\n",
        },
        "extent `n`: year axis 0 = 4, day axis 0 = 3",
    ),
    (
        Case {
            name: "sibling_activation_lengths_differ",
            definitions: &[FILLED, PAIR_BUMP, SIBLING_PAIRS],
            run: "result = run([1i64, 2i64], [3i64, 4i64, 5i64], [6i64, 7i64])\n",
        },
        "extent `n`: t axis 0 = 3, u axis 0 = 2",
    ),
];

/// chelis#2906's twin: the two columns' shared binder has no entry
/// observation (a record erases its dimension argument), so both lanes
/// fail at the first elementwise operation over the two lengths.
const COLUMN_LENGTHS_DIFFER: Case = Case {
    name: "chelis_2906_lengths_differ",
    definitions: &[FILLED, FIRST_FALSE, COLUMNS],
    run: "def run(xs: List[i64], ys: List[i64]) -> List[i64] = to_list(gap(1i64, col(to_tensor(xs)), col(to_tensor(ys))))\nresult = run([1i64, 2i64], [5i64, 3i64, 4i64])\n",
};

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

/// The evaluator's rendering of every root, in the compiled driver's format.
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

/// The evaluator's failure text for a program that must not run.
fn evaluator_failure(source: &str) -> String {
    let error = eval(request(source)).expect_err("the evaluator must refuse the case");
    error
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn outcome(case: &Case, check: impl FnOnce(&str)) -> Result<(), String> {
    let source = case.source();
    catch_unwind(AssertUnwindSafe(|| check(&source))).map_err(|payload| {
        let head: String = panic_message(payload).chars().take(800).collect();
        format!("{}:\n{source}\n  -> {head}", case.name)
    })
}

fn check_positive(case: &Case) -> Result<(), String> {
    outcome(case, |source| {
        let expected = evaluated(source);
        let generated = ownership_support::emit(source, case.name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    })
}

fn check_negative(case: &Case, evaluator: &str, compiled: &str) -> Result<(), String> {
    outcome(case, |source| {
        let refused = evaluator_failure(source);
        assert!(refused.contains(evaluator), "evaluator: {refused}");
        let generated = ownership_support::emit(source, case.name);
        let stderr = ownership_support::run_failure_stderr(&generated, "");
        assert!(stderr.contains(compiled), "compiled: {stderr}");
    })
}

// REGRESSION TEST. On `4a53cc7bd` the C build refused the chelis#2917 and
// chelis#2893 programs at ownership lowering, panicked on the chelis#2906
// program, and refused or panicked on the three neighbouring shapes. On
// `637c680cc` the compiled sibling activations trapped on one shared binder
// (`extent \`d44\`: claimed = 2, insert axis 0 = 3`).
#[test]
fn generic_bodies_compile_as_eval_runs_them() {
    let failures: Vec<String> = POSITIVE
        .iter()
        .filter_map(|case| check_positive(case).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

#[test]
fn a_declared_binder_with_two_lengths_still_fails_in_both_lanes() {
    let mut failures: Vec<String> = NEGATIVE
        .iter()
        .filter_map(|(case, line)| check_negative(case, line, line).err())
        .collect();
    failures.extend(
        check_negative(
            &COLUMN_LENGTHS_DIFFER,
            "tensor shapes must match for elementwise op, got [3] vs [2]",
            "elementwise operand shape mismatch",
        )
        .err(),
    );
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
