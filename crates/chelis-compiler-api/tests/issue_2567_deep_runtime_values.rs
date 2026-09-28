//! chelis#2567: the host interpreter must build, pass, match, render and
//! release a data-type value nested far deeper than its native stack.
//!
//! A `fold` builds each chain, so no Chelis-level recursion is involved: the
//! depth is only the value's. Each read of the accumulator once deep-copied
//! the whole chain, so the interpreter overflowed between 2,000 and 5,000
//! links and copied quadratically. Shared container payloads make reads
//! constant-time; the compiled C lane releases these chains iteratively
//! (chelis#2522).
//!
//! `DEPTH` is far past both limits, and a test thread's stack is smaller than
//! the CLI's main thread, so a per-level native recursion in any walk the
//! evaluator makes over the value (copy, pass, match, render, release)
//! overflows here. The recursion runs through each container kind a data type
//! can nest through: a direct field, a list, a tuple and a dict value.

use std::collections::BTreeMap;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, EvalResult, SourceKind};

const DEPTH: i64 = 100_000;

const TYPES: &str = "\
type Chain =
  | End
  | Link(i64, Chain)
type Tree =
  | Leaf
  | Node(List[Tree])
type Pairs =
  | Done
  | More((i64, Pairs))
type Nest =
  | Stop
  | Deeper(Dict[string, Nest])
def chain(n: i64) -> Chain = fold(fn (acc: Chain, i: i64) -> Link(i, acc), End, range(0i64, n))
def tree(n: i64) -> Tree = fold(fn (acc: Tree, i: i64) -> Node([acc]), Leaf, range(0i64, n))
def pairs(n: i64) -> Pairs = fold(fn (acc: Pairs, i: i64) -> More((i, acc)), Done, range(0i64, n))
def nest(n: i64) -> Nest = fold(fn (acc: Nest, i: i64) -> Deeper(dict_of([(\"k\", acc)])), Stop, range(0i64, n))
";

fn evaluate(roots: &str) -> EvalResult {
    let source = format!("{TYPES}{roots}");
    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: BTreeMap::new(),
    })
    .unwrap_or_else(|error| panic!("the evaluator rejected:\n{roots}\n{error:?}"))
}

fn display(result: &EvalResult, name: &str) -> String {
    result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .and_then(|root| root.display.clone())
        .unwrap_or_else(|| panic!("no rendered root `{name}` in the result"))
}

/// The issue's witness: build, match the head, release the rest.
#[test]
fn a_deep_chain_is_built_matched_and_released() {
    let result = evaluate(&format!(
        "def head(n: i64) -> i64 = match chain(n) with {{\n  \
         | End => -1i64\n  \
         | Link(k, rest) => k\n\
         }}\n\
         a = head({DEPTH}i64)\n"
    ));
    assert_eq!(display(&result, "a"), (DEPTH - 1).to_string());
}

/// Passing the chain through a def with a declared data-type parameter and
/// result, then walking it with a second fold-free match on its tail.
#[test]
fn a_deep_chain_passes_through_a_declared_signature() {
    let result = evaluate(&format!(
        "def pass(c: Chain) -> Chain = c\n\
         def second(n: i64) -> i64 = match pass(chain(n)) with {{\n  \
         | End => -1i64\n  \
         | Link(k, rest) => match rest with {{\n    \
         | End => -1i64\n    \
         | Link(j, tail) => j\n  \
         }}\n\
         }}\n\
         a = second({DEPTH}i64)\n"
    ));
    assert_eq!(display(&result, "a"), (DEPTH - 2).to_string());
}

/// The single printed line of `print(<builder>(DEPTH))`. Printing renders
/// through the evaluator's one value renderer ([05-OBS-1]), the renderer a
/// root's display uses. A deep root's wire value (`ExecutionValue`) is a
/// separate surface: chelis#2601.
fn printed(builder: &str) -> String {
    let result = evaluate(&format!(
        "def show(n: i64) -> unit ! {{ IO }} = print({builder}(n))
         a = show({DEPTH}i64)
"
    ));
    let [line] = result.transcript.as_slice() else {
        panic!("one printed line expected, got {}", result.transcript.len());
    };
    line.clone()
}

#[test]
fn a_deep_chain_renders() {
    let text = printed("chain");
    let depth = usize::try_from(DEPTH).unwrap();
    assert!(text.starts_with(&format!("Link({}, Link({}, ", DEPTH - 1, DEPTH - 2)));
    assert!(text.ends_with(&format!("Link(0, End{}", ")".repeat(depth))));
    assert_eq!(text.matches("Link(").count(), depth);
}

#[test]
fn a_chain_through_lists_renders_and_releases() {
    let text = printed("tree");
    let depth = usize::try_from(DEPTH).unwrap();
    assert_eq!(text.matches("Node([").count(), depth);
    assert!(text.ends_with(&format!("Leaf{}", "])".repeat(depth))));
}

#[test]
fn a_chain_through_tuples_renders_and_releases() {
    let text = printed("pairs");
    let depth = usize::try_from(DEPTH).unwrap();
    assert_eq!(text.matches("More((").count(), depth);
    assert!(text.ends_with(&format!("Done{}", "))".repeat(depth))));
}

#[test]
fn a_chain_through_dict_values_renders_and_releases() {
    let text = printed("nest");
    let depth = usize::try_from(DEPTH).unwrap();
    assert_eq!(text.matches("Deeper(dict(k: ").count(), depth);
    assert!(text.ends_with(&format!("Stop{}", "))".repeat(depth))));
}

/// Negative parity: a shallow chain renders exactly as before.
#[test]
fn a_shallow_chain_renders_exactly() {
    let result = evaluate("a = chain(3i64)\nb = tree(2i64)\nc = pairs(2i64)\nd = nest(1i64)\n");
    assert_eq!(display(&result, "a"), "Link(2, Link(1, Link(0, End)))");
    assert_eq!(display(&result, "b"), "Node([Node([Leaf])])");
    assert_eq!(display(&result, "c"), "More((1, More((0, Done))))");
    assert_eq!(display(&result, "d"), "Deeper(dict(k: Stop))");
}
