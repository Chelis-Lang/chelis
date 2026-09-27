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
                  keep(key_from_seed(2i64), to_tensor([1.0, 2.0])))\n";
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
