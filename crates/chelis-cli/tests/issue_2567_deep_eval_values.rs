//! chelis#2567: `chelis eval` must build, match, print and return a data-type
//! value nested deeper than its walks' native recursion reached.
//!
//! A `fold` builds each chain, so no Chelis-level recursion is involved. The
//! interpreter overflowed its stack between 2,000 and 5,000 links: reading the
//! accumulator deep-copied it, passing it into the
//! callback walked it to stamp its interface provenance, and printing or
//! returning it rendered and converted it, each one native frame group per
//! level. Those walks now run from worklists.
//!
//! `DEPTH` is past the old overflow. Shared container payloads make each
//! accumulator copy constant-time (chelis#2592).

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const DEPTH: usize = 5_000;

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

/// `chelis eval --file` stdout for `TYPES` followed by `roots`.
fn eval(roots: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("deep.ch");
    write_file(&path, &format!("{TYPES}{roots}"));
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval runs");
    assert!(
        output.status.success(),
        "eval failed on:\n{roots}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("utf-8 stdout")
}

/// The issue's witness: build, match the head, release the rest.
#[test]
fn a_deep_chain_is_built_matched_and_released() {
    let stdout = eval(&format!(
        "def head(n: i64) -> i64 = match chain(n) with {{\n  \
         | End => -1i64\n  \
         | Link(k, rest) => k\n\
         }}\n\
         a = head({DEPTH}i64)\n"
    ));
    assert_eq!(stdout, format!("a = {}\n", DEPTH - 1));
}

#[test]
fn a_deep_chain_prints_and_returns_as_a_root() {
    let stdout = eval(&format!(
        "def show(n: i64) -> unit ! {{ IO }} = print(chain(n))\n\
         a = show({DEPTH}i64)\n\
         b = chain({DEPTH}i64)\n"
    ));
    let rendered = format!(
        "{}End{}",
        (0..DEPTH)
            .rev()
            .map(|index| format!("Link({index}, "))
            .collect::<String>(),
        ")".repeat(DEPTH)
    );
    assert_eq!(stdout, format!("{rendered}\na = ()\nb = {rendered}\n"));
}

/// The same depth nested through a list, a tuple and a dict value.
#[test]
fn a_deep_chain_through_lists_returns_as_a_root() {
    let stdout = eval(&format!("a = tree({DEPTH}i64)\n"));
    let tree = format!("{}Leaf{}", "Node([".repeat(DEPTH), "])".repeat(DEPTH));
    assert_eq!(stdout, format!("a = {tree}\n"));
}

#[test]
fn a_deep_chain_through_tuples_returns_as_a_root() {
    let stdout = eval(&format!("a = pairs({DEPTH}i64)\n"));
    let pairs = format!(
        "{}Done{}",
        (0..DEPTH)
            .rev()
            .map(|index| format!("More(({index}, "))
            .collect::<String>(),
        "))".repeat(DEPTH)
    );
    assert_eq!(stdout, format!("a = {pairs}\n"));
}

#[test]
fn a_deep_chain_through_dict_values_returns_as_a_root() {
    let stdout = eval(&format!("a = nest({DEPTH}i64)\n"));
    let nest = format!(
        "{}Stop{}",
        "Deeper(dict(k: ".repeat(DEPTH),
        "))".repeat(DEPTH)
    );
    assert_eq!(stdout, format!("a = {nest}\n"));
}

/// Negative parity: a shallow value renders exactly as before.
#[test]
fn shallow_values_render_exactly() {
    let stdout = eval("a = chain(3i64)\nb = tree(2i64)\nc = pairs(2i64)\nd = nest(1i64)\n");
    assert_eq!(
        stdout,
        "a = Link(2, Link(1, Link(0, End)))\n\
         b = Node([Node([Leaf])])\n\
         c = More((1, More((0, Done))))\n\
         d = Deeper(dict(k: Stop))\n"
    );
}

/// chelis#2592: a fold-built Chain must no longer copy every
/// preceding link at each step. Four times the input should take under six
/// times as long, with room for process startup and a busy CI host. The
/// old owned-Vec payload took 0.39 s at 1,000 links and 4.22 s at 4,000
/// links on the merge head; both outputs were correct.
#[test]
fn fold_chain_growth_is_subquadratic() {
    let measure = |n: usize| {
        let roots = format!(
            "def head(n: i64) -> i64 = match chain(n) with {{\n  | End => 0i64\n  | Link(k, rest) => k\n}}\na = head({n}i64)\n"
        );
        let started = std::time::Instant::now();
        assert_eq!(eval(&roots), format!("a = {}\n", n - 1));
        started.elapsed()
    };
    let best_of_two = |n| std::cmp::min(measure(n), measure(n));
    let small = best_of_two(1_000);
    let large = best_of_two(4_000);
    assert!(
        large < small * 6,
        "fold of 4,000 links took {large:?} versus {small:?} for 1,000 links"
    );
}
