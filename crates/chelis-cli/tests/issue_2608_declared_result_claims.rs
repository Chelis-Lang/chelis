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
        expect: Expect::Value(
            "out = tensor(shape=[2, 3], data=[7.0, 8.0, 9.0, 7.0, 8.0, 9.0])",
        ),
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

fn check(case: &Case) -> Result<(), String> {
    let (c_ok, compiled) = run(case.source, true);
    let (eval_ok, evaluated) = run(case.source, false);
    match case.expect {
        Expect::Trap(context, trap) => {
            for (lane, ok, output) in [("C", c_ok, &compiled), ("eval", eval_ok, &evaluated)] {
                if ok || trap_lines(output) != [context, trap] || output.contains("out =") {
                    return Err(format!(
                        "{}: {lane} must trap with `{context}` / `{trap}` and print nothing after\n{}\n{output}",
                        case.name, case.source
                    ));
                }
            }
        }
        Expect::Value(line) => {
            for (lane, ok, output) in [("C", c_ok, &compiled), ("eval", eval_ok, &evaluated)] {
                if !ok || !output.lines().any(|printed| printed == line) || output.contains("numeric trap:") {
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

// REGRESSION TEST. With the source reverted to `7807ca4ff`, 12 of these rows
// fail: every named trap row except `named_block_body_checks_at_entry` ran to
// completion (`issue_1900_original` instead trapped on `main`'s literal claim),
// and so did the two literal pass-through trap rows. `literal_identity`, the
// entry row and the value rows passed there; they lock the controls.
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
