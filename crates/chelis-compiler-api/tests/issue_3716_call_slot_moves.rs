//! chelis#3716 and chelis#2206: a named heap value passed to a consuming call
//! is moved at its scheduled last use and copied everywhere else, whatever its
//! position and however many slots of one call name it.
//!
//! spec/04 makes a consuming call argument an ordinary consume, and "ordinary
//! consuming fan-out (a later use after an earlier ordinary consume) is handled
//! by inserted copies". The compiled lane used to decide per argument slot,
//! keyed on tail position: a value named in two owned slots of a tail call was
//! moved by both and rejected (`owner %N is not live`), while every last use
//! outside tail position paid a retain immediately cancelled by a release.
//!
//! Two halves. The corpus compiles each program against the
//! `ownership-ledger` runtime: every allocation must be finalized with no live
//! owner left, and stdout must equal the in-process evaluator's rendering of
//! the same roots. The receipts count the ledger's `retain` events and compare
//! two spellings of one computation, so an unrelated change to the driver or
//! the entry adapters moves both sides equally and cannot hide or fake a
//! difference.
//!
//! Evidentiary status: REGRESSION TESTS for the five corpus cases that name
//! one value in several slots of a call at the value's own scope
//! (`dup_list_parameter_tail`, `dup_string_parameter_tail`,
//! `dup_list_local_tail`, `dup_list_three_slots`, `dup_loop_result`), which
//! failed to build on the base of the fix, and for the three receipts, which
//! differed there by the retains the moves now save. The other corpus cases
//! built and balanced on the base and are LOCKS on the neighbouring lowering.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const PRELUDE: &str = "def f1(a: List[i64]) -> i64 = len(a)\n\
                       def f2(a: List[i64], b: List[i64]) -> i64 = add(len(a), len(b))\n\
                       def f3(a: List[i64], b: List[i64], c: List[i64]) -> i64 = add(add(len(a), len(b)), len(c))\n\
                       def fs(a: string, b: string) -> i64 = add(string_len(a), string_len(b))\n";

/// Programs whose values the two lanes must agree on, with a balanced ledger.
const CORPUS: &[(&str, &str)] = &[
    (
        "dup_list_parameter_tail",
        "def g(x: List[i64]) -> i64 = f2(x, x)\nout = g([1i64, 2i64, 3i64])\n",
    ),
    (
        "dup_string_parameter_tail",
        "def g(x: string) -> i64 = fs(x, x)\nout = g(\"abc\")\n",
    ),
    (
        "dup_list_local_tail",
        "def g(n: i64) -> i64 = {\n  xs = range(0i64, n)\n  f2(xs, xs)\n}\nout = g(3i64)\n",
    ),
    (
        "dup_list_three_slots",
        "def g(x: List[i64]) -> i64 = f3(x, x, x)\nout = g([1i64, 2i64, 3i64])\n",
    ),
    (
        "dup_list_in_one_branch",
        "def g(x: List[i64], b: bool) -> i64 = if b then f2(x, x) else 0i64\n\
         out = add(g([1i64, 2i64], true), g([1i64], false))\n",
    ),
    (
        "dup_loop_result",
        "def g(n: i64) -> i64 = {\n  hs = map(fn (v) -> add(v, 1i64), range(0i64, n))\n  f2(hs, hs)\n}\n\
         out = g(3i64)\n",
    ),
    (
        "dup_inside_a_loop_body",
        "def g(x: List[i64]) -> List[i64] = map(fn (j) -> add(f2(x, x), j), range(0i64, 3i64))\n\
         out = g([1i64, 2i64])\n",
    ),
    (
        "dup_then_later_read",
        "def g(x: List[i64]) -> i64 = {\n  r = f2(x, x)\n  add(r, len(x))\n}\nout = g([1i64, 2i64, 3i64])\n",
    ),
    (
        "let_bound_last_use",
        "def g(x: List[i64]) -> i64 = {\n  r = f1(x)\n  r\n}\nout = g([1i64, 2i64, 3i64])\n",
    ),
    (
        "calls_as_builtin_arguments",
        "def g(x: List[i64]) -> i64 = add(f1(x), f1(x))\nout = g([1i64, 2i64, 3i64])\n",
    ),
    (
        "borrowing_later_argument",
        "def h(a: List[i64], n: i64) -> i64 = add(len(a), n)\n\
         def g(x: List[i64]) -> i64 = h(x, len(x))\nout = g([1i64, 2i64, 3i64])\n",
    ),
    (
        "root_value_reused",
        "xs: List[i64] = [1i64, 2i64]\nout = f2(xs, xs)\n",
    ),
];

/// The evaluator's rendering of every root, in the compiled driver's format.
fn evaluated(source: &str) -> String {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    })
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

fn check(name: &str, body: &str) -> Result<(), String> {
    let source = format!("{PRELUDE}{body}");
    catch_unwind(AssertUnwindSafe(|| {
        let expected = evaluated(&source);
        let generated = ownership_support::emit(&source, name);
        let (summary, stdout) = ownership_support::run_program(&generated);
        ownership_support::balanced(&summary);
        assert_eq!(stdout, expected, "compiled output differs from eval");
    }))
    .map_err(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        let head: String = message.chars().take(600).collect();
        format!("{name}:\n{body}  -> {head}")
    })
}

/// Every case runs before the test reports, so one failing shape cannot hide
/// another.
#[test]
fn reused_and_last_use_arguments_agree_with_eval_and_balance_the_ledger() {
    let failures: Vec<String> = CORPUS
        .iter()
        .filter_map(|(name, body)| check(name, body).err())
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} cases failed:\n\n{}",
        failures.len(),
        CORPUS.len(),
        failures.join("\n\n")
    );
}

/// The ledger's `retain` events for one program, after checking that its
/// ledger balances and that it prints `expected`.
fn retains(body: &str, expected: &str) -> usize {
    let source = format!("{PRELUDE}{body}");
    let generated = ownership_support::emit(&source, "receipt");
    let (summary, stdout, events) = ownership_support::run_program_counting_events(&generated);
    ownership_support::balanced(&summary);
    assert_eq!(stdout, expected, "{body}");
    events.get("retain").copied().unwrap_or(0)
}

/// chelis#2206: binding a call's result costs no retain over returning it
/// directly. The value's last use moves it in both spellings.
#[test]
fn a_let_bound_last_use_retains_no_more_than_a_tail_use() {
    let tail = retains(
        "def g(x: List[i64]) -> i64 = f1(x)\nout = g([1i64, 2i64, 3i64])\n",
        "out = 3\n",
    );
    let bound = retains(
        "def g(x: List[i64]) -> i64 = {\n  r = f1(x)\n  r\n}\nout = g([1i64, 2i64, 3i64])\n",
        "out = 3\n",
    );
    assert_eq!(
        bound, tail,
        "a let-bound last use paid {bound} retains against {tail}"
    );
}

/// chelis#2206: N calls on one value cost N - 1 retains over a single call in
/// tail position; the last call moves the value instead of retaining it.
#[test]
fn n_calls_on_one_value_retain_n_minus_one_times() {
    let one = retains(
        "def g(x: List[i64]) -> i64 = f1(x)\nout = g([1i64, 2i64])\n",
        "out = 2\n",
    );
    let four = retains(
        "def g(x: List[i64]) -> i64 = {\n  r1 = f1(x)\n  r2 = f1(x)\n  r3 = f1(x)\n  r4 = f1(x)\n  \
         add(add(r1, r2), add(r3, r4))\n}\nout = g([1i64, 2i64])\n",
        "out = 8\n",
    );
    assert_eq!(
        four,
        one + 3,
        "four calls retained {four} times against {one} for one"
    );
}

/// chelis#3716: one value in two owned slots costs exactly the one inserted
/// copy spec/04 prescribes over two distinct values in those slots. Both
/// spellings take the same parameters, so any entry adaptation is equal.
#[test]
fn one_value_in_two_slots_retains_once_more_than_two_values() {
    let distinct = retains(
        "def g(x: List[i64], y: List[i64]) -> i64 = f2(x, y)\nout = g([1i64, 2i64], [3i64])\n",
        "out = 3\n",
    );
    let shared = retains(
        "def g(x: List[i64], y: List[i64]) -> i64 = f2(x, x)\nout = g([1i64, 2i64], [3i64])\n",
        "out = 4\n",
    );
    assert_eq!(
        shared,
        distinct + 1,
        "a shared value retained {shared} times against {distinct} for distinct values"
    );
}
