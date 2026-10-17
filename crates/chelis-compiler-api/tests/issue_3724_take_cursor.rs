//! chelis#3724: a List whose last use is `take` moves into the builtin and is
//! truncated in place instead of cloned, and never changes what a program
//! means.
//!
//! A cursor that trims its last element with `take(xs, len(xs) - 1)` cloned
//! the surviving prefix at every step, which is quadratic. `take` now has a
//! `CONTAINER_CONSUMERS` row like `skip`, so at the list's scheduled last use
//! the operand moves and `chelis_list_take_owned` truncates a uniquely owned
//! list in place.
//!
//! Two halves, in the shape chelis#2205 and chelis#2334 established. The
//! corpus compiles each program against the `ownership-ledger` runtime: every
//! allocation must be finalized with no live owner left, and stdout must equal
//! the in-process evaluator's rendering. Its alias cases are the shapes in
//! which a moved operand meets a strong-owner count above one (a seed walked
//! twice, a tuple-held alias, an element of a nested list) or an edge count
//! (an empty seed, a count above the length). The receipts count the ledger's
//! `allocate` events for one cursor at two lengths: a linear cursor allocates
//! the same number of objects at both, a cloning one allocates one list more
//! per extra step. A cursor that reads its list again inside a later argument
//! of the same call is the negative control: the list's last use is that
//! read, so the `take` keeps borrowing and keeps cloning.
//!
//! Evidentiary status: REGRESSION TEST for the linear receipt, which failed on
//! the base of the fix (the cursor allocated one list per step). The corpus
//! and the negative control are LOCKS: they pass on the base too.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const CURSOR: &str = "def trim_back(xs: List[i64], total: i64) -> i64 = \
                      if (len(xs) == 0i64) then total else trim_back(take(xs, (len(xs) - 1i64)), (total + 1i64))\n";

/// The cursor with the list read again inside a later argument of the
/// recursive call, after the `take`.
const READ_AGAIN_CURSOR: &str = "def trim_sum(xs: List[i64], total: i64) -> i64 = \
                                 if (len(xs) == 0i64) then total else trim_sum(take(xs, (len(xs) - 1i64)), (total + len(xs)))\n";

const CORPUS: &[(&str, &str)] = &[
    ("cursor", "out = trim_back(range(0i64, 8i64), 0i64)\n"),
    (
        "cursor_over_strings",
        "def trim_strings(xs: List[string], total: i64) -> i64 = \
         if (len(xs) == 0i64) then total else trim_strings(take(xs, (len(xs) - 1i64)), (total + string_len(index(xs, 0i64))))\n\
         out = trim_strings(map(fn (i) -> string_concat(\"s\", \"tt\"), range(0i64, 16i64)), 0i64)\n",
    ),
    (
        "seed_walked_twice",
        "def twice(xs: List[i64]) -> i64 = add(trim_back(xs, 0i64), trim_back(xs, 0i64))\n\
         out = twice(range(0i64, 8i64))\n",
    ),
    (
        "tuple_held_alias",
        "def held(xs: List[i64]) -> i64 = {\n  pair = (xs, 1i64)\n  ys = take(xs, 1i64)\n  \
         add(len(ys), len(pair.0))\n}\nout = held(range(0i64, 5i64))\n",
    ),
    (
        "element_of_a_nested_list",
        "def inner(xss: List[List[i64]]) -> i64 = {\n  first = take(index(xss, 0i64), 1i64)\n  \
         add(len(first), len(index(xss, 0i64)))\n}\nout = inner([[1i64, 2i64, 3i64], [4i64]])\n",
    ),
    (
        "empty_seed",
        "empty: List[i64] = []\ndef trim_empty(xs: List[i64]) -> i64 = len(take(xs, 5i64))\n\
         out = trim_empty(empty)\n",
    ),
    (
        "count_above_the_length",
        "def whole(xs: List[i64]) -> i64 = {\n  ys = take(xs, 10i64)\n  index(ys, 2i64)\n}\n\
         out = whole(range(0i64, 3i64))\n",
    ),
    (
        "read_again_cursor",
        "out = trim_sum(range(0i64, 8i64), 0i64)\n",
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
    let source = format!("{CURSOR}{READ_AGAIN_CURSOR}{body}");
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
fn take_cursors_and_alias_controls_agree_with_eval_and_balance_the_ledger() {
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

/// The ledger's `allocate` and `resize` events for the cursor `call` over `n`
/// elements, after checking that it balances and prints `expected`.
fn cursor_events(call: &str, n: i64, expected: i64) -> (usize, usize) {
    let source = format!("{CURSOR}{READ_AGAIN_CURSOR}out = {call}(range(0i64, {n}i64), 0i64)\n");
    let generated = ownership_support::emit(&source, "take_cursor");
    let (summary, stdout, events) = ownership_support::run_program_counting_events(&generated);
    ownership_support::balanced(&summary);
    assert_eq!(stdout, format!("out = {expected}\n"), "{call} over {n}");
    (
        events.get("allocate").copied().unwrap_or(0),
        events.get("resize").copied().unwrap_or(0),
    )
}

/// The receipt: doubling the cursor's length adds no allocation, because every
/// step truncates the one list in place, and every step records the in-place
/// arm's `resize`.
#[test]
fn a_take_cursor_allocates_the_same_at_any_length() {
    let (short, short_resizes) = cursor_events("trim_back", 32, 32);
    let (long, long_resizes) = cursor_events("trim_back", 64, 64);
    assert_eq!(
        long, short,
        "a 64-step cursor allocated {long} objects against {short} for 32 steps"
    );
    assert!(
        long_resizes >= short_resizes + 32,
        "every extra step must take the in-place arm: {short_resizes} resizes at 32 \
         steps, {long_resizes} at 64"
    );
}

/// The negative control: a cursor that reads its list after the `take` keeps
/// cloning, one list per step, so the receipt above is not satisfied by a
/// lane that never allocates at all.
#[test]
fn a_cursor_that_reads_its_list_again_keeps_cloning() {
    let (short, _) = cursor_events("trim_sum", 32, (1..=32).sum());
    let (long, _) = cursor_events("trim_sum", 64, (1..=64).sum());
    assert_eq!(
        long,
        short + 32,
        "the read-again cursor must clone one list per extra step"
    );
}
