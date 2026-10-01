//! chelis#2522: the C lane with data-type values.
//!
//! 1. Releasing a deep recursive data-type chain overflowed the native stack:
//!    the runtime finalized a child container by recursion. A chain 100,000
//!    links deep must now be built, passed to a function, and fully returned.
//! 2. Every private function body rebuilt its parameters' result origins by
//!    walking each carried value, `__chelis_host_result_origin_interface_*`,
//!    so recursion over a large carried value did work proportional to the
//!    value on every call. Interface values now carry one uniform `load`
//!    origin. The oracle counts the ledger's `retain` rows, which the walk
//!    produced once per element per call, rather than timing the program.
//! 3. chelis#2587: [05-OP-36]'s structural `eq` and `neq` walk the same deep
//!    chains, so the runtime compares them from an explicit stack too.
//!
//! Oracle: the compiled program runs against the `ownership-ledger` runtime,
//! every allocation must be finalized with no live owner left, and stdout must
//! equal what `chelis eval` prints for the same program.

mod ownership_support;

use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use std::collections::BTreeMap;

const CHAIN: &str = "type Chain =\n  | End\n  | Link(i64, Chain)\n\
def head(c: Chain) -> i64 =\n  match c with {\n    | End => 0i64\n    | Link(k, rest) => k\n  }\n\
def build(n: i64) -> i64 = head(fold(fn (acc: Chain, i: i64) -> Link(i, acc), End, range(0i64, n)))\n";

/// REGRESSION TEST. The runtime overflowed its stack releasing the chain
/// between depth 6,000 and 20,000 before the iterative finalizer.
#[test]
fn a_chain_one_hundred_thousand_links_deep_is_passed_and_released() {
    let source = format!("{CHAIN}a = build(100000i64)\n");
    let generated = ownership_support::emit(&source, "deep_chain");
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    // `chelis eval` overflows its own stack on this depth (chelis#2567); the
    // head of a fold-built chain is its last index.
    assert_eq!(stdout, "a = 99999\n");
    assert!(
        summary["allocations"].as_u64().unwrap() > 100_000,
        "{summary}"
    );
}

/// The same program at a depth `chelis eval` completes, compared with it.
#[test]
fn a_shallow_chain_agrees_with_eval() {
    let source = format!("{CHAIN}a = build(200i64)\n");
    let generated = ownership_support::emit(&source, "shallow_chain");
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    let evaluated = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: BTreeMap::new(),
    })
    .expect("eval");
    assert_eq!(evaluated.roots[0].display.as_deref(), Some("199"));
    assert_eq!(stdout, "a = 199\n");
}

fn walk_retains(n: usize) -> usize {
    // A literal list, so the count isolates the recursion from how the list
    // was built.
    let items = vec!["to_tensor([1i64, 2i64, 3i64])"; n].join(", ");
    let source = format!(
        "def walk(xs: List[tensor[3, i64]], i: i64, acc: i64) -> i64 =\n  \
         if (i >= len(xs)) then acc else walk(xs, i + 1i64, acc + 1i64)\n\
         def run(flag: bool) -> i64 = walk([{items}], 0i64, 0i64)\n\
         a = run(true)\n"
    );
    let generated = ownership_support::emit(&source, "walk");
    assert!(
        !generated.contains("__chelis_host_result_origin_interface_"),
        "no emitted code walks an interface value to build its origin"
    );
    let (summary, stdout, events) = ownership_support::run_program_counting_events(&generated);
    ownership_support::balanced(&summary);
    assert_eq!(stdout, format!("a = {n}\n"));
    events.get("retain").copied().unwrap_or(0)
}

/// REGRESSION TEST. Before the uniform interface origin, each of the `n`
/// recursive calls walked all `n` list elements, retaining each: quadrupling
/// `n` multiplied the retains by about sixteen. Linear work multiplies them
/// by about four.
#[test]
fn recursion_over_a_carried_list_does_linear_ownership_work() {
    let small = walk_retains(64);
    let large = walk_retains(256);
    assert!(
        large <= 5 * small,
        "retains grew from {small} at n = 64 to {large} at n = 256: superlinear"
    );
}

/// chelis#2587: structural equality over two chains 100,000 links deep runs in
/// bounded native stack and releases every owner; the compared values are
/// borrowed, so the ledger stays balanced. Two chains of different depths
/// compare unequal.
#[test]
fn chains_one_hundred_thousand_links_deep_compare_structurally() {
    let source = format!(
        "{CHAIN}def same(n: i64, m: i64) -> bool = eq(fold(fn (acc: Chain, i: i64) -> Link(i, \
         acc), End, range(0i64, n)), fold(fn (acc: Chain, i: i64) -> Link(i, acc), End, \
         range(0i64, m)))\n\
         a = same(100000i64, 100000i64)\n\
         b = same(3i64, 2i64)\n"
    );
    let generated = ownership_support::emit(&source, "deep_equality");
    let (summary, stdout) = ownership_support::run_program(&generated);
    ownership_support::balanced(&summary);
    assert_eq!(stdout, "a = true\nb = false\n");
}
