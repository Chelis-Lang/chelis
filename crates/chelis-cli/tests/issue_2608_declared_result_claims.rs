//! chelis#2608 and chelis#1900, spec/04 section 4.7: a declared result
//! extent is a claim the returned value must satisfy on every lane, whether
//! the claim is a literal or a binder a parameter witnesses. A named claim's
//! trap context names the binder and its declaring parameter axis, then the
//! producing primitive and its observed axis.
//!
//! On `7807ca4ff` every named host-lane claim below ran to completion on both
//! lanes and returned the wrong extent, and a literal claim on a block-bodied
//! pass-through of a wildcard parameter did the same.

mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

use result_claims::run;

/// What both lanes must observe.
enum Expect {
    /// The exact context and trap lines, in order, with nothing printed after.
    Trap(&'static str, &'static str),
    /// The exact line the program prints, with no trap.
    Value(&'static str),
}

struct Case {
    name: &'static str,
    source: &'static str,
    expect: Expect,
}

const CASES: &[Case] = &[
    // chelis#2608: a wildcard parameter returned under a binder another
    // parameter witnesses.
    Case {
        name: "named_identity",
        source: "def f[n](a: tensor[n, f32], t0: tensor[*, f32]) -> tensor[n, f32] = t0\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]))\n",
        expect: Expect::Trap(
            "extent `n`: a axis 0 = 2, load axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    Case {
        name: "named_identity_agrees",
        source: "def f[n](a: tensor[n, f32], t0: tensor[*, f32]) -> tensor[n, f32] = t0\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0]))\n",
        expect: Expect::Value("out = tensor(shape=[2], data=[4.0, 5.0])"),
    },
    Case {
        name: "named_taken_arm",
        source: "def f[n](a: tensor[n, f32], t0: tensor[*, f32], b: bool) -> tensor[n, f32] = if b then t0 else a\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]), true)\n",
        expect: Expect::Trap(
            "extent `n`: a axis 0 = 2, load axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    // Untaken-arm negative: the claim is checked only on the returned value.
    Case {
        name: "named_untaken_arm",
        source: "def f[n](a: tensor[n, f32], t0: tensor[*, f32], b: bool) -> tensor[n, f32] = if b then t0 else a\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]), false)\n",
        expect: Expect::Value("out = tensor(shape=[2], data=[1.0, 2.0])"),
    },
    Case {
        name: "named_builtin_in_taken_arm",
        source: "def f[n](a: tensor[n, f32], t0: tensor[*, f32], b: bool) -> tensor[n, f32] = if b then neg(t0) else a\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]), true)\n",
        expect: Expect::Trap(
            "extent `n`: a axis 0 = 2, neg axis 0 = 3",
            "numeric trap: domain in neg at i64",
        ),
    },
    Case {
        name: "named_builtin_in_untaken_arm",
        source: "def f[n](a: tensor[n, f32], t0: tensor[*, f32], b: bool) -> tensor[n, f32] = if b then neg(t0) else a\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]), false)\n",
        expect: Expect::Value("out = tensor(shape=[2], data=[1.0, 2.0])"),
    },
    Case {
        name: "named_second_axis",
        source: "def f[n](a: tensor[2, n, f32], t0: tensor[2, *, f32], b: bool) -> tensor[2, n, f32] = if b then t0 else a\n\
                 out = f(to_tensor([[1.0, 2.0], [3.0, 4.0]]), to_tensor([[4.0, 5.0, 6.0], [1.0, 1.0, 1.0]]), true)\n",
        expect: Expect::Trap(
            "extent `n`: a axis 1 = 2, load axis 1 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    // Two claimed axes: declared order decides which is reported.
    Case {
        name: "named_two_binders",
        source: "def f[n, m](a: tensor[n, m, f32], t0: tensor[*, *, f32], b: bool) -> tensor[n, m, f32] = if b then t0 else a\n\
                 out = f(to_tensor([[1.0, 2.0]]), to_tensor([[4.0, 5.0, 6.0], [1.0, 1.0, 1.0]]), true)\n",
        expect: Expect::Trap(
            "extent `n`: a axis 0 = 1, load axis 0 = 2",
            "numeric trap: domain in load at i64",
        ),
    },
    // The canonical source is the binder's first witness in signature order.
    Case {
        name: "named_repeated_binder_names_its_first_witness",
        source: "def f[n](a: tensor[n, f32], c: tensor[n, f32], t0: tensor[*, f32], b: bool) -> tensor[n, f32] = if b then t0 else a\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([7.0, 8.0]), to_tensor([4.0, 5.0, 6.0]), true)\n",
        expect: Expect::Trap(
            "extent `n`: a axis 0 = 2, load axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    // A same-sized witness of another binder is not the declaring source.
    Case {
        name: "named_distinct_same_sized_witness_is_not_the_source",
        source: "def f[k, m](u: tensor[k, f32], w: tensor[m, f32], t0: tensor[*, f32], b: bool) -> tensor[m, f32] = if b then t0 else w\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([7.0, 8.0]), to_tensor([4.0, 5.0, 6.0]), true)\n",
        expect: Expect::Trap(
            "extent `m`: w axis 0 = 2, load axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    // The generated C reads a parameter whose name is a C keyword through its
    // mapped identifier; the context keeps the authored name.
    Case {
        name: "named_source_spelled_as_a_c_keyword",
        source: "def f[n](int: tensor[n, f32], t0: tensor[*, f32]) -> tensor[n, f32] = t0\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]))\n",
        expect: Expect::Trap(
            "extent `n`: int axis 0 = 2, load axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    // Each recursive invocation builds its own frame from its own witness.
    Case {
        name: "named_through_recursion",
        source: "def f[n](a: tensor[n, f32], t0: tensor[*, f32], k: i64) -> tensor[n, f32] = if k > 0i64 then f(a, t0, k - 1i64) else t0\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]), 2i64)\n",
        expect: Expect::Trap(
            "extent `n`: a axis 0 = 2, load axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    // A caller's named claim inherited by a call to a tensor-kernel `def` is
    // checked at the kernel's producer with the same context on both lanes.
    Case {
        name: "named_claim_inherited_by_a_kernel_call",
        source: "def g(t: tensor[*, f32]) -> tensor[*, f32] = neg(t)\n\
                 def f[n](a: tensor[n, f32], t0: tensor[*, f32], b: bool) -> tensor[n, f32] = if b then g(t0) else a\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]), true)\n",
        expect: Expect::Trap(
            "extent `n`: a axis 0 = 2, neg axis 0 = 3",
            "numeric trap: domain in neg at i64",
        ),
    },
    Case {
        name: "named_claim_inherited_by_a_kernel_call_agrees",
        source: "def g(t: tensor[*, f32]) -> tensor[*, f32] = neg(t)\n\
                 def f[n](a: tensor[n, f32], t0: tensor[*, f32], b: bool) -> tensor[n, f32] = if b then g(t0) else a\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0]), true)\n",
        expect: Expect::Value("out = tensor(shape=[2], data=[-4.0, -5.0])"),
    },
    Case {
        name: "named_claim_inherited_by_an_untaken_kernel_call",
        source: "def g(t: tensor[*, f32]) -> tensor[*, f32] = neg(t)\n\
                 def f[n](a: tensor[n, f32], t0: tensor[*, f32], b: bool) -> tensor[n, f32] = if b then g(t0) else a\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]), false)\n",
        expect: Expect::Value("out = tensor(shape=[2], data=[1.0, 2.0])"),
    },
    // chelis#1900's original reproducer, its isolated forms and an agreeing
    // control. The body's `append` keeps `concat` on the host lane.
    Case {
        name: "issue_1900_original",
        source: "module HostNamed\n\
                 def probe[m, n](w: tensor[m, 3, f32], v: tensor[n, 3, f32]) -> tensor[m, 3, f32] = concat(append([v], v), 0i32)\n\
                 def main() -> tensor[1, 3, f32] = probe(to_tensor([[1.0, 2.0, 3.0]]), to_tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))\n",
        expect: Expect::Trap(
            "extent `m`: w axis 0 = 1, concat axis 0 = 8",
            "numeric trap: domain in concat at i64",
        ),
    },
    Case {
        name: "issue_1900_top_level_value",
        source: "def probe[m, n](w: tensor[m, 3, f32], v: tensor[n, 3, f32]) -> tensor[m, 3, f32] = concat(append([v], v), 0i32)\n\
                 out = probe(to_tensor([[1.0, 2.0, 3.0]]), to_tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))\n",
        expect: Expect::Trap(
            "extent `m`: w axis 0 = 1, concat axis 0 = 8",
            "numeric trap: domain in concat at i64",
        ),
    },
    Case {
        name: "issue_1900_undeclared_main",
        source: "def probe[m, n](w: tensor[m, 3, f32], v: tensor[n, 3, f32]) -> tensor[m, 3, f32] = concat(append([v], v), 0i32)\n\
                 def main() = probe(to_tensor([[1.0, 2.0, 3.0]]), to_tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))\n",
        expect: Expect::Trap(
            "extent `m`: w axis 0 = 1, concat axis 0 = 8",
            "numeric trap: domain in concat at i64",
        ),
    },
    Case {
        name: "issue_1900_agrees",
        source: "def probe[m, n](w: tensor[m, 3, f32], v: tensor[n, 3, f32]) -> tensor[m, 3, f32] = concat(append([v], v), 0i32)\n\
                 out = probe(to_tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]), to_tensor([[7.0, 8.0, 9.0]]))\n",
        expect: Expect::Value("out = tensor(shape=[2, 3], data=[7.0, 8.0, 9.0, 7.0, 8.0, 9.0])"),
    },
    // A literal claim keeps its literal rendering.
    Case {
        name: "literal_identity",
        source: "def f(a: tensor[2, f32], t0: tensor[*, f32]) -> tensor[2, f32] = t0\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]))\n",
        expect: Expect::Trap(
            "extent `2`: claimed = 2, load axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    // A literal claim on a block-bodied pass-through of a wildcard parameter:
    // the tensor-lane lowering had no owner for it and dropped it.
    Case {
        name: "literal_block_pass_through",
        source: "def f(t0: tensor[*, f32]) -> tensor[2, f32] = {\n  y = t0\n  y\n}\n\
                 out = f(to_tensor([4.0, 5.0, 6.0]))\n",
        expect: Expect::Trap(
            "extent `2`: claimed = 2, t0 axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    Case {
        name: "literal_block_pass_through_agrees",
        source: "def f(t0: tensor[*, f32]) -> tensor[2, f32] = {\n  y = t0\n  y\n}\n\
                 out = f(to_tensor([4.0, 5.0]))\n",
        expect: Expect::Value("out = tensor(shape=[2], data=[4.0, 5.0])"),
    },
    Case {
        name: "literal_pass_through_inlined_into_a_caller",
        source: "def f(t0: tensor[*, f32]) -> tensor[2, f32] = {\n  y = t0\n  y\n}\n\
                 def g(t: tensor[*, f32]) -> tensor[*, f32] = f(t)\n\
                 out = g(to_tensor([4.0, 5.0, 6.0]))\n",
        expect: Expect::Trap(
            "extent `2`: claimed = 2, t axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
    // Untaken-arm negative for that guard: the callee never runs.
    Case {
        name: "literal_pass_through_in_an_untaken_arm",
        source: "def f(t0: tensor[*, f32]) -> tensor[2, f32] = {\n  y = t0\n  y\n}\n\
                 def g(t: tensor[*, f32], b: bool) -> tensor[*, f32] = if b then f(t) else t\n\
                 out = g(to_tensor([4.0, 5.0, 6.0]), false)\n",
        expect: Expect::Value("out = tensor(shape=[3], data=[4.0, 5.0, 6.0])"),
    },
    // A named claim in a tensor-lane body already unifies the wildcard with
    // the binder and checks it at entry; the host frame adds no second check.
    Case {
        name: "named_block_body_checks_at_entry",
        source: "def f[n](a: tensor[n, f32], t0: tensor[*, f32]) -> tensor[n, f32] = {\n  y = t0\n  y\n}\n\
                 out = f(to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]))\n",
        expect: Expect::Trap(
            "extent `n`: a axis 0 = 2, t0 axis 0 = 3",
            "numeric trap: domain in load at i64",
        ),
    },
];

/// The context and trap lines a run printed, in order.
fn trap_lines(output: &str) -> Vec<&str> {
    output
        .lines()
        .map(|line| line.trim_start_matches("error: "))
        .filter(|line| line.starts_with("extent `") || line.starts_with("numeric trap:"))
        .collect()
}

/// Whether a program printed a top-level binding, which a trapping program
/// must not reach.
fn printed_a_binding(output: &str) -> bool {
    output
        .lines()
        .any(|line| line.starts_with("out =") || line.starts_with("a ="))
}

fn check(case: &Case) -> Result<(), String> {
    let (c_ok, compiled) = run(case.source, true);
    let (eval_ok, evaluated) = run(case.source, false);
    match case.expect {
        Expect::Trap(context, trap) => {
            for (lane, ok, output) in [("C", c_ok, &compiled), ("eval", eval_ok, &evaluated)] {
                if ok || trap_lines(output) != [context, trap] || printed_a_binding(output) {
                    return Err(format!(
                        "{}: {lane} must trap with `{context}` / `{trap}` and print nothing after\n{}\n{output}",
                        case.name, case.source
                    ));
                }
            }
        }
        Expect::Value(line) => {
            for (lane, ok, output) in [("C", c_ok, &compiled), ("eval", eval_ok, &evaluated)] {
                if !ok
                    || !output.lines().any(|printed| printed == line)
                    || output.contains("numeric trap:")
                {
                    return Err(format!(
                        "{}: {lane} must print `{line}`\n{}\n{output}",
                        case.name, case.source
                    ));
                }
            }
        }
    }
    Ok(())
}

// REGRESSION TEST. With the source reverted to `7807ca4ff`, 15 of these 25
// rows fail: every named trap row except `named_block_body_checks_at_entry` ran
// to completion (`issue_1900_original` instead trapped on `main`'s literal
// claim), and so did the two literal pass-through trap rows.
// `literal_identity`, the entry row and the value rows passed there; they lock
// the controls.
#[test]
fn declared_result_claims_are_checked_on_both_lanes() {
    let failures: Vec<String> = CASES.iter().filter_map(|case| check(case).err()).collect();
    assert!(
        failures.is_empty(),
        "{} of {} cases failed:\n\n{}",
        failures.len(),
        CASES.len(),
        failures.join("\n\n")
    );
}

/// chelis#2598, spec/04 section 4.7: a tensor a list combinator returns, or
/// nests in its result, has that combinator as its producer. Each shape
/// returns a three-element tensor, so `{n}` = 3 agrees and `{n}` = 2 traps.
struct Combinator {
    op: &'static str,
    source: &'static str,
    /// The value the agreeing program binds to `a`.
    value: &'static str,
}

const XS: &str = "[to_tensor([1.0, 2.0, 3.0]), to_tensor([4.0, 5.0, 6.0])]";

const COMBINATORS: &[Combinator] = &[
    Combinator {
        op: "map",
        source: "def f(xs: List[tensor[*, f32]]) -> tensor[{n}, f32] = index(map(fn (x: tensor[*, f32]) -> x, xs), 1i64)\n\
                 a = f({xs})\n",
        value: "tensor(shape=[3], data=[4.0, 5.0, 6.0])",
    },
    Combinator {
        op: "flat_map",
        source: "def f(xs: List[tensor[*, f32]]) -> tensor[{n}, f32] = index(flat_map(fn (x: tensor[*, f32]) -> [x, x], xs), 3i64)\n\
                 a = f({xs})\n",
        value: "tensor(shape=[3], data=[4.0, 5.0, 6.0])",
    },
    Combinator {
        op: "filter",
        source: "def f(xs: List[tensor[*, f32]]) -> tensor[{n}, f32] = index(filter(fn (x: tensor[*, f32]) -> true, xs), 0i64)\n\
                 a = f({xs})\n",
        value: "tensor(shape=[3], data=[1.0, 2.0, 3.0])",
    },
    Combinator {
        op: "scan",
        source: "def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] = index(scan(fn (acc: tensor[*, f32], x: tensor[*, f32]) -> (acc + x), t0, xs), 1i64)\n\
                 a = f({xs}, to_tensor([0.5, 0.5, 0.5]))\n",
        value: "tensor(shape=[3], data=[5.5, 7.5, 9.5])",
    },
    Combinator {
        op: "fold",
        source: "def f(xs: List[tensor[*, f32]]) -> tensor[{n}, f32] =\n  \
                 index(fold(fn (acc: List[tensor[*, f32]], x: tensor[*, f32]) -> append(acc, x), [index(xs, 0i64)], xs), 2i64)\n\
                 a = f({xs})\n",
        value: "tensor(shape=[3], data=[4.0, 5.0, 6.0])",
    },
    Combinator {
        op: "append",
        source: "def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[{n}, f32] = index(append(xs, t0), 0i64)\n\
                 a = f({xs}, to_tensor([7.0, 8.0, 9.0]))\n",
        value: "tensor(shape=[3], data=[1.0, 2.0, 3.0])",
    },
];

/// Projections that reach a combinator's tensor through a pattern rather
/// than `index`, a named claim, and an untaken arm.
const COMBINATOR_CASES: &[Case] = &[
    Case {
        name: "fold_tuple_accumulator_projected_by_a_pattern",
        source: "def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[2, f32] =\n  \
                 match fold(fn (acc: (tensor[*, f32], i64), x: tensor[*, f32]) -> match acc with {\n    \
                 | (t, n) => ((t + x), add(n, 1i64))\n  \
                 }, (t0, 0i64), xs) with {\n    \
                 | (t, n) => t\n  \
                 }\n\
                 a = f([to_tensor([1.0, 2.0, 3.0])], to_tensor([4.0, 5.0, 6.0]))\n",
        expect: Expect::Trap(
            "extent `2`: claimed = 2, fold axis 0 = 3",
            "numeric trap: domain in fold at i64",
        ),
    },
    Case {
        name: "fold_tuple_accumulator_agrees",
        source: "def f(xs: List[tensor[*, f32]], t0: tensor[*, f32]) -> tensor[3, f32] =\n  \
                 match fold(fn (acc: (tensor[*, f32], i64), x: tensor[*, f32]) -> match acc with {\n    \
                 | (t, n) => ((t + x), add(n, 1i64))\n  \
                 }, (t0, 0i64), xs) with {\n    \
                 | (t, n) => t\n  \
                 }\n\
                 a = f([to_tensor([1.0, 2.0, 3.0])], to_tensor([4.0, 5.0, 6.0]))\n",
        expect: Expect::Value("a = tensor(shape=[3], data=[5.0, 7.0, 9.0])"),
    },
    Case {
        name: "map_result_projected_by_a_cons_pattern",
        source: "def f(xs: List[tensor[*, f32]]) -> tensor[2, f32] = match map(fn (x: tensor[*, f32]) -> x, xs) with {\n    \
                 | Cons(h, t) => h\n    \
                 | Nil => to_tensor([0.0, 0.0])\n  \
                 }\n\
                 a = f([to_tensor([1.0, 2.0, 3.0]), to_tensor([4.0, 5.0, 6.0])])\n",
        expect: Expect::Trap(
            "extent `2`: claimed = 2, map axis 0 = 3",
            "numeric trap: domain in map at i64",
        ),
    },
    Case {
        name: "named_claim_on_a_map_result",
        source: "def f[n](w: tensor[n, f32], xs: List[tensor[*, f32]]) -> tensor[n, f32] = index(map(fn (x: tensor[*, f32]) -> x, xs), 1i64)\n\
                 a = f(to_tensor([1.0, 2.0]), [to_tensor([1.0, 2.0, 3.0]), to_tensor([4.0, 5.0, 6.0])])\n",
        expect: Expect::Trap(
            "extent `n`: w axis 0 = 2, map axis 0 = 3",
            "numeric trap: domain in map at i64",
        ),
    },
    Case {
        name: "map_result_in_an_untaken_arm",
        source: "def f(xs: List[tensor[*, f32]], b: bool, w: tensor[2, f32]) -> tensor[2, f32] = if b then index(map(fn (x: tensor[*, f32]) -> x, xs), 1i64) else w\n\
                 a = f([to_tensor([1.0, 2.0, 3.0]), to_tensor([4.0, 5.0, 6.0])], false, to_tensor([1.0, 1.0]))\n",
        expect: Expect::Value("a = tensor(shape=[2], data=[1.0, 1.0])"),
    },
];

fn check_combinator(shape: &Combinator) -> Result<(), String> {
    let failing = shape.source.replace("{n}", "2").replace("{xs}", XS);
    let context = format!("extent `2`: claimed = 2, {} axis 0 = 3", shape.op);
    let trap = format!("numeric trap: domain in {} at i64", shape.op);
    for (lane, native) in [("C", true), ("eval", false)] {
        let (ok, output) = run(&failing, native);
        if ok
            || trap_lines(&output) != [context.as_str(), trap.as_str()]
            || printed_a_binding(&output)
        {
            return Err(format!(
                "{}: {lane} must trap with `{context}` / `{trap}`\n{failing}\n{output}",
                shape.op
            ));
        }
    }
    // Negative parity: the true extent runs through the same combinator.
    let agreeing = shape.source.replace("{n}", "3").replace("{xs}", XS);
    let line = format!("a = {}", shape.value);
    for (lane, native) in [("C", true), ("eval", false)] {
        let (ok, output) = run(&agreeing, native);
        if !ok || !output.lines().any(|printed| printed == line) || output.contains("numeric trap:")
        {
            return Err(format!(
                "{}: {lane} must print `{line}`\n{agreeing}\n{output}",
                shape.op
            ));
        }
    }
    Ok(())
}

// REGRESSION TEST. With the source reverted to `7807ca4ff`, 10 of these 11
// fail: every mismatch ended in an internal provenance error, an abort or
// another primitive's name on at least one lane, and the agreeing tuple `fold`
// failed on eval. `map_result_in_an_untaken_arm` passed there; it locks the
// untaken-arm control.
#[test]
fn a_combinator_result_is_produced_by_its_combinator() {
    let failures: Vec<String> = COMBINATORS
        .iter()
        .filter_map(|shape| check_combinator(shape).err())
        .chain(COMBINATOR_CASES.iter().filter_map(|case| check(case).err()))
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} cases failed:\n\n{}",
        failures.len(),
        COMBINATORS.len() + COMBINATOR_CASES.len(),
        failures.join("\n\n")
    );
}
