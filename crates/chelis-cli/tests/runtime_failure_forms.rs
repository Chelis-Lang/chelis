//! Run-time language failures render in the form the spec fixes, identically
//! in `chelis eval` and in compiled C.
//!
//! A failing runtime extent or non-negativity guard is a `Domain` trap in the
//! operation that owns the guarded extent (spec/04-type-system.md section
//! 4.7). Its [04-NUM-9] line follows a context line naming the operation,
//! the axis or entry and the value observed, and every lane renders both
//! lines through `chelis_abi::failure`. Every failing program has a passing
//! twin that differs only in the guarded value, so each guard is shown to
//! fire on the bad value alone.

#[path = "common/host_effect_parity.rs"]
mod parity;

use assert_cmd::Command;
use tempfile::tempdir;

/// One guarded program: `{value}` in `source` is replaced by `failing` for the
/// trapping run and by `passing` for its twin.
struct Guard {
    name: &'static str,
    source: &'static str,
    failing: &'static str,
    passing: &'static str,
    context: &'static str,
    trap: &'static str,
    passing_out: &'static str,
}

/// Runs `guard` on both lanes, failing and passing, and checks the agreed
/// result of each.
fn assert_guard(guard: &Guard) {
    let failing = guard.source.replace("{value}", guard.failing);
    let run = parity::assert_lanes_agree(&failing, guard.name);
    assert_eq!(run.status, Some(1), "{}: {run:?}", guard.name);
    assert_eq!(run.context, guard.context, "{}: {run:?}", guard.name);
    assert_eq!(run.failure, guard.trap, "{}: {run:?}", guard.name);

    let passing = guard.source.replace("{value}", guard.passing);
    let twin = format!("{}_twin", guard.name);
    let run = parity::assert_lanes_agree(&passing, &twin);
    assert_eq!(run.status, Some(0), "{twin}: {run:?}");
    assert_eq!(run.stdout, guard.passing_out, "{twin}: {run:?}");
}

/// spec/04-type-system.md section 4.7: a negative runtime extent or bound of
/// a movement operation fails the non-negativity guard, a `Domain` trap in
/// the owning movement operation, never a message about the evaluator's
/// graph.
#[test]
fn negative_movement_bounds_trap_in_the_movement_operation() {
    for guard in [
        Guard {
            name: "expand",
            source: "def f(x: tensor[1, f32], k: i64) -> tensor[*, f32] = expand(x, 0i32, k)\n\
                     out = f(to_tensor([1.0f32]), {value})\n",
            failing: "-2i64",
            passing: "3i64",
            context: "expand target extent at axis 0 is negative: -2",
            trap: "numeric trap: domain in expand at i64",
            passing_out: "out = tensor(shape=[3], data=[1.0, 1.0, 1.0])\n",
        },
        Guard {
            name: "insert",
            source: "def f(x: tensor[*, f32], k: i64) -> tensor[*, *, f32] = insert(x, 0i32, k)\n\
                     out = f(to_tensor([1.0f32, 2.0f32]), {value})\n",
            failing: "-2i64",
            passing: "2i64",
            context: "insert target extent at axis 0 is negative: -2",
            trap: "numeric trap: domain in insert at i64",
            passing_out: "out = tensor(shape=[2, 2], data=[1.0, 2.0, 1.0, 2.0])\n",
        },
        Guard {
            name: "pad",
            source: "def f(x: tensor[*, f32], k: i64) -> tensor[*, f32] = \
                     pad(x, [[1i64, k]], 0.0f32)\n\
                     out = f(to_tensor([1.0f32, 2.0f32]), {value})\n",
            failing: "-1i64",
            passing: "1i64",
            context: "pad bound at axis 0 is negative: -1",
            trap: "numeric trap: domain in pad at i64",
            passing_out: "out = tensor(shape=[4], data=[0.0, 1.0, 2.0, 0.0])\n",
        },
        Guard {
            name: "shrink",
            source: "def f(x: tensor[*, f32], k: i64) -> tensor[*, f32] = shrink(x, [[k, 2i64]])\n\
                     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), {value})\n",
            failing: "-1i64",
            passing: "1i64",
            context: "shrink bound at axis 0 is negative: -1",
            trap: "numeric trap: domain in shrink at i64",
            passing_out: "out = tensor(shape=[1], data=[2.0])\n",
        },
    ] {
        assert_guard(&guard);
    }
}

/// [05-OP-53], [05-OP-57], [05-OP-62] and [05-HOST-1]: runtime sizes,
/// children and lengths that disagree with the extent they claim, or are
/// negative, are `Domain` traps in the operation, with the context line
/// before the trap line.
#[test]
fn runtime_extent_disagreements_trap_in_the_owning_operation() {
    for guard in [
        Guard {
            name: "concat",
            source: "def f(a: tensor[*, *, f32], b: tensor[*, *, f32], k: i64) -> \
                     tensor[*, *, f32] = concat([a, shrink(b, [[0i64, 1i64], [0i64, k]])], 0i32)\n\
                     out = f(to_tensor([[1.0f32, 2.0f32]]), to_tensor([[3.0f32, 4.0f32]]), {value})\n",
            failing: "1i64",
            passing: "2i64",
            context: "concat parts disagree at axis 1: part 0 has 2, part 1 has 1",
            trap: "numeric trap: domain in concat at i64",
            passing_out: "out = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])\n",
        },
        Guard {
            name: "to_tensor",
            source: "def f(k: i64) -> tensor[*, *, i64] = to_tensor([[1i64, 2i64], range(0i64, k)])\n\
                     out = f({value})\n",
            failing: "1i64",
            passing: "2i64",
            context: "to_tensor children disagree in shape: child 0 has [2], child 1 has [1]",
            trap: "numeric trap: domain in to_tensor at i64",
            passing_out: "out = tensor(shape=[2, 2], data=[1, 2, 0, 1])\n",
        },
        Guard {
            name: "split_negative",
            source: "def f(x: tensor[3, f32], k: i64) -> List[tensor[*, f32]] = \
                     split(x, 0i32, [k, 1i64])\n\
                     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), {value})\n",
            failing: "-1i64",
            passing: "2i64",
            context: "split list entry 0 is negative: -1",
            trap: "numeric trap: domain in split at i64",
            passing_out: "out = [tensor(shape=[2], data=[1.0, 2.0]), \
                          tensor(shape=[1], data=[3.0])]\n",
        },
        Guard {
            name: "split_sum",
            source: "def f(x: tensor[3, f32], k: i64) -> List[tensor[*, f32]] = \
                     split(x, 0i32, [k, 1i64])\n\
                     out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), {value})\n",
            failing: "1i64",
            passing: "2i64",
            context: "split sizes sum to 2 but axis 0 has extent 3",
            trap: "numeric trap: domain in split at i64",
            passing_out: "out = [tensor(shape=[2], data=[1.0, 2.0]), \
                          tensor(shape=[1], data=[3.0])]\n",
        },
        Guard {
            name: "tensor_scan",
            source: "def f(n: i64) -> tensor[*, f32] = \
                     tensor_scan(0.0f32, fn (previous: f32, i: i64) -> add(previous, 1.0f32), n)\n\
                     out = f({value})\n",
            failing: "-2i64",
            passing: "2i64",
            context: "tensor_scan length is negative: -2",
            trap: "numeric trap: domain in tensor_scan at i64",
            passing_out: "out = tensor(shape=[2], data=[1.0, 2.0])\n",
        },
    ] {
        assert_guard(&guard);
    }
}

/// [05-RWIN-1]: a runtime window wider than its input axis traps `Domain` in
/// the `reduce_window_*` builtin before any read. Compiled C refuses a
/// runtime window list at build time (chelis#1058), so only eval runs it.
#[test]
fn a_runtime_window_wider_than_its_axis_traps_in_reduce_window() {
    let source = |extra: &str| {
        format!(
            "def f(x: tensor[*, f32], e: i64) -> tensor[*, f32] = {{\n\
             w = add(shape(&x, 0i32), e)\n\
             reduce_window_max(x, [w], [1i64])\n\
             }}\n\
             out = f(to_tensor([1.0f32, 2.0f32, 3.0f32]), {extra})\n"
        )
    };
    let eval = |text: &str| {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("window.ch");
        std::fs::write(&path, text).expect("write source");
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file", path.to_str().unwrap()])
            .output()
            .expect("chelis eval runs")
    };
    let failing = eval(&source("1i64"));
    assert_eq!(failing.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&failing.stderr),
        "reduce_window_max window 4 at axis 0 exceeds the input extent 3\n\
         numeric trap: domain in reduce_window_max at i64\n"
    );
    let passing = eval(&source("-1i64"));
    assert!(passing.status.success(), "{passing:?}");
    assert_eq!(
        String::from_utf8_lossy(&passing.stdout),
        "out = tensor(shape=[2], data=[2.0, 3.0])\n"
    );
}

/// [05-OP-58] and [05-OP-33]: `char_code`, `char_from_code` and `clamp` trap
/// `Domain` in the operation, rendered in [04-NUM-9]'s form with no C ABI
/// function name in the text.
#[test]
fn value_domain_failures_trap_in_the_operation() {
    for guard in [
        Guard {
            name: "char_code",
            source: "def f(s: string) -> i64 = char_code(s)\nout = f({value})\n",
            failing: "\"ab\"",
            passing: "\"a\"",
            context: "char_code operand has 2 Unicode scalar values, expected exactly one",
            trap: "numeric trap: domain in char_code at i64",
            passing_out: "out = 97\n",
        },
        Guard {
            name: "char_from_code",
            source: "def f(c: i64) -> string = char_from_code(c)\nout = f({value})\n",
            failing: "55296i64",
            passing: "65i64",
            context: "char_from_code code 55296 is not a Unicode scalar value",
            trap: "numeric trap: domain in char_from_code at i64",
            passing_out: "out = A\n",
        },
        Guard {
            name: "clamp_inverted",
            source: "def f(x: tensor[2, f64], h: tensor[2, f64]) -> tensor[2, f64] = \
                     clamp(x, to_tensor([0.0f64, 3.0f64]), h)\n\
                     out = f(to_tensor([1.0f64, 2.0f64]), to_tensor([5.0f64, {value}]))\n",
            failing: "1.0f64",
            passing: "4.0f64",
            context: "clamp lower bound exceeds upper bound at row-major position 1",
            trap: "numeric trap: domain in clamp at f64",
            passing_out: "out = tensor(shape=[2], data=[1.0, 3.0])\n",
        },
        Guard {
            name: "clamp_shape",
            source: "def f(x: tensor[*, f32], k: i64) -> tensor[*, f32] = \
                     clamp(x, to_tensor(map(fn (i: i64) -> 0.0f32, range(0i64, k))), \
                     scalar_to_tensor(5.0f32))\n\
                     out = f(to_tensor([1.0f32, 2.0f32]), {value})\n",
            failing: "3i64",
            passing: "2i64",
            context: "clamp lower bound has shape [3] but the operand has [2]",
            trap: "numeric trap: domain in clamp at i64",
            passing_out: "out = tensor(shape=[2], data=[1.0, 2.0])\n",
        },
    ] {
        assert_guard(&guard);
    }
}
