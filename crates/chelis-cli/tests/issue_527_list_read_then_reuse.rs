//! Chelis-Lang/chelis#527 — read-then-reuse of a `List[tensor]` must
//! compile and run.
//!
//! At 0.10.0 the idiomatic optimizer shape "read a list's length /
//! elements, then reuse the list" compiled; at 0.10.1 the #343
//! linearity fix (correctly typing `List[tensor]` parameters whose
//! name collides with a Deep tag, e.g. `params`) exposed that `len`
//! and `index` *consumed* their `List` argument, so the later reuse
//! became a use-after-consume with no non-consuming form to express
//! it. `len` / `index` now borrow their `List` argument by default
//! (the runtime `chelis_list_len` / `chelis_list_index` take a
//! `const chelis_list *` and never free it), restoring the pattern.
//!
//! This is the executable acceptance oracle: it builds a faithful
//! self-contained replica of School's `adamw_step_walk` /
//! `adamw_step_list` read-then-reuse shape — a recursive
//! `index(params, i)`-then-recurse walk under a `len(params)`-then-walk
//! driver, with the parameter literally named `params` (the
//! tag-colliding identifier #343 began consume-tracking) — compiles
//! the generated C, runs it, and checks the numeric output.

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run, gcc_available, parse_tensor_data};

/// Faithful replica of School's `adamw_step_list` + `adamw_step_walk`
/// read-then-reuse shape: `len(params)` then a recursive
/// `index(params, i)`-then-recurse walk that reuses `params` on every
/// frame. Reduces `[[1,2],[3,4],[5,6]]` to its elementwise sum
/// `[9, 12]`. Builds, runs, and checks the result.
#[test]
fn issue_527_len_index_then_reuse_list_builds_and_runs() {
    if !gcc_available() {
        eprintln!("skipping: c compiler not available");
        return;
    }
    let source = "\
def walk(params: List[tensor[2, f32]], i: i64, total: i64) -> tensor[2, f32] =\n\
  if gte(i, total) then to_tensor([0.0, 0.0]) else\n\
    {\n\
      p = index(params, i)\n\
      rest = walk(params, add(i, cast(1, i64)), total)\n\
      add(p, rest)\n\
    }\n\
def reduce_list(params: List[tensor[2, f32]]) -> tensor[2, f32] =\n\
  {\n\
    total = cast(len(params), i64)\n\
    walk(params, cast(0, i64), total)\n\
  }\n\
def run() -> tensor[2, f32] =\n\
  {\n\
    a = to_tensor([1.0, 2.0])\n\
    b = to_tensor([3.0, 4.0])\n\
    c = to_tensor([5.0, 6.0])\n\
    ps: List[tensor[2, f32]] = [a, b, c]\n\
    reduce_list(ps)\n\
  }\n\
out = run()\n";

    let stdout = build_and_run(source, "list_read_then_reuse");
    let actual = parse_tensor_data(&stdout, "out");
    let expected = [9.0, 12.0];
    assert_eq!(
        actual.len(),
        expected.len(),
        "unexpected output arity: {actual:?}"
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "element {i}: actual={a} expected={e}\nfull stdout:\n{stdout}"
        );
    }
}
