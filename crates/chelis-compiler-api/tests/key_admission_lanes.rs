//! chelis#2413: the key allow-list (`chelis_types::key_admission`) is one
//! list for the checker and the lowered graph's key rules, so a program the
//! checker admits evaluates in the DAG evaluator and in compiled C, and a
//! key reaching an operation the list does not name is refused at the
//! checker, never by a later lane.
#![allow(deprecated)]
mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, EvalResult, SourceKind};
use std::collections::BTreeMap;

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    }
}

fn lines(result: &EvalResult) -> Vec<String> {
    result
        .roots
        .iter()
        .map(|root| {
            format!(
                "{} = {}",
                root.name.as_deref().expect("a named root"),
                root.display.as_deref().expect("an in-process display")
            )
        })
        .collect()
}

/// `drop` ([05-OP-67]) is on the allow-list, and so is the graph's `Drop`
/// that it lowers to: a function whose one use of its key is `drop` checks,
/// evaluates and runs in C with the ownership ledger balanced.
///
/// Evidentiary status: REGRESSION TEST. At `f4eeca363` the evaluator failed
/// `discard` with "the lowered program breaks the key rules: node 1 of `f`
/// produces a key ... key `k` of `f` reaches input 0 of node 1", while the
/// checker admitted it.
#[test]
fn a_dropped_key_evaluates_and_runs_in_c() {
    let source = "def discard(k: key) = drop(k)\n\ndef keep(k: key, x: tensor[2, f32]) -> tensor[2, \
                  f32] = {\n  _ = drop(k)\n  x\n}\n\ndef main() = (discard(key_from_seed(1i64)), \
                  keep(key_from_seed(2i64), to_tensor([1.0, 2.0], f32)))\n";
    let evaluated = eval(request(source)).unwrap_or_else(|error| panic!("eval: {error:?}"));
    assert_eq!(
        lines(&evaluated),
        ["main.0 = ()", "main.1 = tensor(shape=[2], data=[1.0, 2.0])"],
        "{source}"
    );
    let generated = ownership_support::emit(source, "dropped_key");
    let (summary, _) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
}

/// `realize` is not on the allow-list: a key reaching it is refused by the
/// checker, naming `realize`, before any lane lowers it.
///
/// Evidentiary status: REGRESSION TEST. At `f4eeca363` this program checked
/// at 1.0, then failed the evaluator's key rules and `chelis build`'s
/// ownership lowering ("ownership lowering invariant failed in `dag`").
#[test]
fn a_realized_key_is_refused_at_the_checker() {
    let source = "def main() = (realize(key_from_seed(3i64)), realize(split_keys(key_from_seed(4i64), \
                  2i64)))\n";
    let error = eval(request(source)).expect_err("a realized key is refused");
    let text = format!("{error:?}");
    assert!(
        text.contains("`realize` does not admit a key-carrying operand at argument 0")
            && !text.contains("breaks the key rules"),
        "{text}"
    );
}

/// A handler region's result moves out as a block's does ([04-LIN-9]): a
/// `with device("cpu")` region returning a derived key checks, and it
/// lowers to its body, so the DAG evaluator and compiled C print the same
/// keys, the keys its body gives without the region.
///
/// Evidentiary status: REGRESSION TEST. At `c18888f34` both regions were
/// refused with "`handle-effect` does not admit a key-carrying operand at
/// argument 1".
#[test]
fn a_handler_regions_key_result_moves_out_and_runs_in_c() {
    let source = "def folded(k: key) -> key = with device(\"cpu\") { fold_in(k, 1i64) }\n\ndef \
                  halves(k: key) -> (key, key) = with device(\"cpu\") { split_key(k) }\n\na = \
                  folded(key_from_seed(7i64))\nb = halves(key_from_seed(8i64))\nc = \
                  fold_in(key_from_seed(7i64), 1i64)\nd = split_key(key_from_seed(8i64))\n";
    let evaluated = eval(request(source)).unwrap_or_else(|error| panic!("eval: {error:?}"));
    let evaluated = lines(&evaluated);
    let value = |name: &str| {
        evaluated
            .iter()
            .find_map(|line| line.strip_prefix(&format!("{name} = ")))
            .unwrap_or_else(|| panic!("no root `{name}` in {evaluated:?}"))
            .to_string()
    };
    for (region, body) in [("a", "c"), ("b.0", "d.0"), ("b.1", "d.1")] {
        assert_eq!(value(region), value(body), "{evaluated:?}");
    }
    let generated = ownership_support::emit(source, "handler_region_key");
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    let printed = stdout.lines().map(str::to_string).collect::<Vec<_>>();
    assert_eq!(printed, evaluated, "{stdout}");
}

/// Negative parity: inside a handler region a key is still used at most
/// once, so a region that folds one key twice is refused as a key reuse,
/// and the region itself is not what refuses it.
#[test]
fn a_handler_region_that_uses_a_key_twice_is_refused() {
    let source = "def f(k: key) -> (key, key) = with device(\"cpu\") { (fold_in(k, 1i64), \
                  fold_in(k, 2i64)) }\n";
    let error = eval(request(source)).expect_err("a key used twice in a region");
    let text = format!("{error:?}");
    assert!(
        text.contains("key-carrying variable `k` was already consumed")
            && !text.contains("`handle-effect` does not admit"),
        "{text}"
    );
}
