//! chelis#2891: a record, list or tuple literal that stores a heap value in
//! one item and passes the same value to a consuming call in a later item
//! compiles to C that keeps the stored value alive.
//!
//! The ownership plan copies the shared value for the stored item, but the
//! copy was recorded at the literal's own site, which the C projection
//! renders after the whole literal. The stored item took the value without
//! its copy, the later call consumed and released the only reference, and
//! constructing the literal then read freed memory. The compiled program
//! aborted with `chelis_value_clone: heap kind mismatch` when the runtime's
//! handle check, itself a read of the freed header, caught it.
//!
//! Oracle: every program compiles, runs against the `ownership-ledger`
//! runtime with every allocation finalized and no live owner left, and
//! prints what `chelis eval` prints. The corpus crosses record, list and
//! tuple literals, both item orders, and ADT, list, string and tensor
//! operands. The twins share the value without a consuming call between the
//! store and the literal, or use it once, and must stay balanced. Every case
//! runs before the test reports, so one failing shape cannot hide another.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const PRELUDE: &str = "type Mask =\n  | Mask { a: bool, b: bool }\n\
type Cal =\n  | Cal { mask: Mask, hol: List[i64] }\n\
type Held =\n  | Held { xs: List[i64], n: i64 }\n\
type Pair =\n  | Pair { left: Mask, right: Mask }\n\
def count_true(m: Mask) -> i64 = if m.a then 1i64 else 0i64\n\
def flip(m: Mask) -> Mask = Mask { a: m.b, b: m.a }\n\
def total(xs: List[i64]) -> i64 = fold(fn (acc: i64, x: i64) -> add(acc, x), 0i64, xs)\n\
def width(s: string) -> i64 = string_len(s)\n\
def mass(t: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(t, 0))\n";

const MASK: &str = "Mask { a: true, b: false }";

struct Case {
    name: &'static str,
    body: String,
}

fn case(name: &'static str, body: &str) -> Case {
    Case {
        name,
        body: body.replace("{mask}", MASK),
    }
}

/// A later item's consuming call reads the value an earlier item stores.
fn shared() -> Vec<Case> {
    vec![
        // The issue's program.
        case(
            "record_stores_then_calls",
            "def norm(m: Mask) -> Cal = Cal { mask: m, hol: [count_true(m)] }\nr = norm({mask})\n",
        ),
        case(
            "record_literal_in_the_other_order",
            "def norm(m: Mask) -> Cal = Cal { hol: [count_true(m)], mask: m }\nr = norm({mask})\n",
        ),
        case(
            "list_stores_then_calls",
            "def both(m: Mask) -> List[Mask] = [m, flip(m)]\nr = both({mask})\n",
        ),
        case(
            "tuple_stores_then_calls",
            "def both(m: Mask) -> (Mask, i64) = (m, count_true(m))\nr = both({mask})\n",
        ),
        case(
            "nested_tuple_in_a_list",
            "def both(m: Mask) -> List[(Mask, i64)] = [(m, count_true(m))]\nr = both({mask})\n",
        ),
        case(
            "list_operand",
            "def keep(xs: List[i64]) -> Held = Held { xs, n: total(xs) }\nr = keep([4i64, 2i64, 3i64])\n",
        ),
        case(
            "string_operand",
            "def keep(s: string) -> (string, i64) = (s, width(s))\nr = keep(string_concat(\"ab\", \"c\"))\n",
        ),
        case(
            "tensor_operand",
            "def keep(t: tensor[3, f32]) -> (tensor[3, f32], f32) = (t, mass(t))\nr = keep(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n",
        ),
    ]
}

/// The same values without a consuming call between the store and the
/// literal: used once, bound before the literal, stored twice, consumed
/// before the item that stores it, or read through a field projection,
/// which borrows.
fn twins() -> Vec<Case> {
    vec![
        case(
            "list_calls_then_stores",
            "def both(m: Mask) -> List[Mask] = [flip(m), m]\nr = both({mask})\n",
        ),
        case(
            "record_reads_through_a_nested_block",
            "def run(seed: i64) -> List[i64] = {\n  c = Cal { mask: {mask}, hol: [seed] }\n  d = Cal { mask: c.mask, hol: [count_true(c.mask)] }\n  [len(d.hol), if d.mask.a then 1i64 else 0i64]\n}\nr = run(0i64)\n",
        ),
        case(
            "record_uses_the_value_once",
            "def norm(m: Mask) -> Cal = Cal { mask: m, hol: [1i64] }\nr = norm({mask})\n",
        ),
        case(
            "record_binds_the_call_first",
            "def norm(m: Mask) -> Cal = {\n  hol = [count_true(m)]\n  Cal { mask: m, hol }\n}\nr = norm({mask})\n",
        ),
        case(
            "record_stores_the_value_twice",
            "def twice(m: Mask) -> Pair = Pair { left: m, right: m }\nr = twice({mask})\n",
        ),
    ]
}

fn source(case: &Case) -> String {
    format!("{PRELUDE}{}", case.body)
}

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

fn check(case: &Case) -> Result<(), String> {
    let source = source(case);
    catch_unwind(AssertUnwindSafe(|| {
        let expected = evaluated(&source);
        let generated = ownership_support::emit(&source, case.name);
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
        format!("{}:\n{}\n  -> {head}", case.name, case.body)
    })
}

fn failures(cases: &[Case]) -> Vec<String> {
    cases.iter().filter_map(|case| check(case).err()).collect()
}

// REGRESSION TEST. On `1a772bea6` every case here aborted at the runtime's
// heap-handle check (heap kind mismatch, or no live strong owner).
#[test]
fn a_literal_keeps_a_value_a_later_item_consumes() {
    let cases = shared();
    let failed = failures(&cases);
    assert!(
        failed.is_empty(),
        "{} of {} cases failed:\n\n{}",
        failed.len(),
        cases.len(),
        failed.join("\n\n")
    );
}

#[test]
fn a_literal_without_a_later_consumer_stays_balanced() {
    let cases = twins();
    let failed = failures(&cases);
    assert!(
        failed.is_empty(),
        "{} of {} cases failed:\n\n{}",
        failed.len(),
        cases.len(),
        failed.join("\n\n")
    );
}
