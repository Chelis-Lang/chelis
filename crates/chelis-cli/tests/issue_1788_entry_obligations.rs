//! Spec/04 §4.7: every signature obligation executes at its invocation entry,
//! in parameter/axis order, including obligations split across tensor helpers.

mod common;

use assert_cmd::Command;
use std::fs;
use std::process::Output;

fn text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn run(source: &str, c: bool) -> Output {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("entry.ch");
    fs::write(&path, source).expect("source");
    if !c {
        return Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--allow-style-violations", "--file"])
            .arg(&path)
            .output()
            .expect("eval");
    }
    let out = dir.path().join("c");
    let built = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--allow-style-violations"])
        .arg(&path)
        .args(["--target", "c", "-o"])
        .arg(&out)
        .output()
        .expect("build C");
    assert!(built.status.success(), "{}", text(&built));
    assert!(common::link_generated(&out, "entry.c", "entry").success());
    std::process::Command::new(out.join("entry"))
        .output()
        .expect("run C")
}

fn check_both(source: &str, expected: &str, succeeds: bool) {
    check_both_context(source, expected, expected, succeeds);
}

fn check_both_context(source: &str, eval_expected: &str, c_expected: &str, succeeds: bool) {
    for c in [false, true] {
        let expected = if c { c_expected } else { eval_expected };
        let output = run(source, c);
        let rendered = text(&output);
        assert_eq!(output.status.success(), succeeds, "C={c}: {rendered}");
        assert!(
            rendered.contains(expected),
            "C={c}: expected {expected:?}: {rendered}"
        );
        if !succeeds {
            assert!(
                rendered
                    .lines()
                    .any(|line| line == "numeric trap: domain in load at int64"),
                "C={c}: {rendered}"
            );
            assert!(
                !rendered.contains("body-ran"),
                "C={c}: entry must precede body: {rendered}"
            );
        }
    }
}

fn mixed_source(first: &str, second: &str) -> String {
    format!(
        "def mixed(a: tensor[seq, f32], b: tensor[batch, seq, f32], x: tensor[width, f32], y: tensor[height, width, f32]) -> (tensor[seq, f32], tensor[width, f32]) ! {{ IO }} = {{\n _ = print(\"body-ran\")\n (neg(a), add(x, sum(y, 0i32)))\n}}\nout = mixed(to_tensor({first}), to_tensor([[1.0f32, 2.0f32]]), to_tensor({second}), to_tensor([[3.0f32, 4.0f32]]))\n"
    )
}

#[test]
fn mixed_helpers_keep_unused_signature_witnesses() {
    check_both(
        &mixed_source("[1.0f32, 2.0f32, 3.0f32]", "[1.0f32, 2.0f32]"),
        "extent `seq`: a axis 0 = 3, b axis 1 = 2",
        false,
    );
}

#[test]
fn mixed_helpers_fail_in_signature_order() {
    check_both(
        &mixed_source("[1.0f32, 2.0f32, 3.0f32]", "[1.0f32, 2.0f32, 3.0f32]"),
        "extent `seq`: a axis 0 = 3, b axis 1 = 2",
        false,
    );
}

#[test]
fn mixed_helpers_accept_independent_binders() {
    check_both(
        &mixed_source("[1.0f32, 2.0f32]", "[1.0f32, 2.0f32]"),
        "out.1 = tensor(shape=[2], data=[4.0, 6.0])",
        true,
    );
}

fn higher_order_source(second_skip: usize) -> String {
    format!(
        "def g(x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\ndef h(y: tensor[k, f32]) -> tensor[k, f32] = add(y, y)\ndef h2(z: tensor[k, f32]) -> tensor[k, f32] = add(z, z)\ndef g2(x: tensor[n, f32]) -> tensor[k, f32] = shrink(x, [[{second_skip}i64, shape(x, 0i32)]])\ndef apply4(f: tensor[p, f32] -> tensor[p, f32], v: tensor[p, f32], q2: tensor[p, f32] -> tensor[p, f32], w: tensor[p, f32]) -> tensor[p, f32] = add(f(v), q2(w))\ndef main() = apply4(h, g(to_tensor([1.0f32, 2.0f32, 3.0f32])), h2, g2(to_tensor([7.0f32, 8.0f32, 9.0f32])))\n"
    )
}

#[test]
fn higher_order_invocation_keeps_authored_entry_claim() {
    check_both(
        &higher_order_source(2),
        "extent `p`: v axis 0 = 2, w axis 0 = 1",
        false,
    );
}

#[test]
fn higher_order_entry_does_not_depend_on_shared_result_labels() {
    // Isolate invocation entry from #1991's independently tracked result-label
    // interaction: the two producers and callbacks have different labels.
    let source = higher_order_source(2)
        .replace(
            "def h2(z: tensor[k, f32]) -> tensor[k, f32]",
            "def h2(z: tensor[other, f32]) -> tensor[other, f32]",
        )
        .replace(
            "def g2(x: tensor[n, f32]) -> tensor[k, f32]",
            "def g2(x: tensor[n, f32]) -> tensor[other, f32]",
        );
    check_both(&source, "extent `p`: v axis 0 = 2, w axis 0 = 1", false);
}

#[test]
fn higher_order_invocation_accepts_agreeing_claim() {
    check_both(
        &higher_order_source(1),
        "main = tensor(shape=[2], data=[20.0, 24.0])",
        true,
    );
}

fn app_and_pipe_entry_source(pipe: bool, invalid_first: bool, second_skip: usize) -> String {
    let extra = usize::from(invalid_first);
    let first = format!("cut(to_tensor([1.0f32, 2.0f32, 3.0f32]), 1i64, {extra}i64)");
    let second = format!("cut(to_tensor([7.0f32, 8.0f32, 9.0f32]), {second_skip}i64, 0i64)");
    let call = if pipe {
        format!("{first} |> apply(twice, {second})")
    } else {
        format!("apply({first}, twice, {second})")
    };
    format!(
        "def cut(x: tensor[n, f32], lo: int64, extra: int64) -> tensor[*, f32] ! {{ IO }} = {{\n _ = print(\"argument-ran\")\n shrink(x, [[lo, add(shape(x, 0i32), extra)]])\n}}\ndef twice(x: tensor[q, f32]) -> tensor[q, f32] = add(x, x)\ndef apply(v: tensor[p, f32], f: tensor[p, f32] -> tensor[p, f32], w: tensor[p, f32]) -> tensor[p, f32] = add(f(v), w)\nout = {{\n result = {call}\n _ = print(\"following-ran\")\n result\n}}\n"
    )
}

#[test]
fn higher_order_app_and_pipe_preserve_actual_before_entry_failure() {
    for pipe in [false, true] {
        for (invalid_first, expected_op, argument_effects) in
            [(true, "shrink", 1), (false, "load", 2)]
        {
            let source = app_and_pipe_entry_source(pipe, invalid_first, 2);
            for c in [false, true] {
                let output = run(&source, c);
                let rendered = text(&output);
                assert!(!output.status.success(), "pipe={pipe}, C={c}: {rendered}");
                let expected_trap = if invalid_first && !c {
                    // Direct eval rejects intrinsic shrink bounds before the
                    // IR evaluator's numeric-trap formatter is reached.
                    "error: shrink axis 0 bound [1, 4] is out of range for input dim 3".into()
                } else {
                    format!("numeric trap: domain in {expected_op} at int64")
                };
                assert!(
                    rendered.lines().any(|line| line == expected_trap),
                    "pipe={pipe}, C={c}: {rendered}"
                );
                assert_eq!(
                    rendered.matches("argument-ran").count(),
                    argument_effects,
                    "pipe={pipe}, C={c}: {rendered}"
                );
                assert!(!rendered.contains("following-ran"), "{rendered}");
                if invalid_first {
                    assert!(!rendered.contains("extent `p`"), "{rendered}");
                    assert!(!rendered.contains("domain in load"), "{rendered}");
                } else {
                    assert!(
                        rendered.contains("extent `p`: v axis 0 = 2, w axis 0 = 1"),
                        "pipe={pipe}, C={c}: {rendered}"
                    );
                    assert!(!rendered.contains("domain in shrink"), "{rendered}");
                }
            }
        }
    }
}

#[test]
fn higher_order_app_and_pipe_accept_agreeing_entry_claim() {
    for pipe in [false, true] {
        let source = app_and_pipe_entry_source(pipe, false, 1);
        for c in [false, true] {
            let output = run(&source, c);
            let rendered = text(&output);
            assert!(output.status.success(), "pipe={pipe}, C={c}: {rendered}");
            assert_eq!(rendered.matches("argument-ran").count(), 2, "{rendered}");
            assert_eq!(rendered.matches("following-ran").count(), 1, "{rendered}");
            assert!(
                rendered.contains("out = tensor(shape=[2], data=[12.0, 15.0])"),
                "pipe={pipe}, C={c}: {rendered}"
            );
        }
    }
}

fn mixed_literal_source(literal_first: bool, repeated_bad: bool, literal_bad: bool) -> String {
    let fixed = "fixed: tensor[2, f32]";
    let pair = "a: tensor[seq, f32], b: tensor[batch, seq, f32]";
    let params = if literal_first {
        format!("{fixed}, {pair}")
    } else {
        format!("{pair}, {fixed}")
    };
    let literal = if literal_bad {
        "[1.0f32, 2.0f32, 3.0f32]"
    } else {
        "[1.0f32, 2.0f32]"
    };
    let a = if repeated_bad {
        "[1.0f32, 2.0f32, 3.0f32]"
    } else {
        "[1.0f32, 2.0f32]"
    };
    let fixed_arg = format!("opaque(to_tensor({literal}))");
    let pair_args = format!("to_tensor({a}), to_tensor([[1.0f32, 2.0f32]])");
    let args = if literal_first {
        format!("{fixed_arg}, {pair_args}")
    } else {
        format!("{pair_args}, {fixed_arg}")
    };
    format!(
        "def opaque(z: tensor[n, f32]) -> tensor[*, f32] = shrink(z, [[0i64, shape(z, 0i32)]])\ndef mixed({params}, x: tensor[width, f32], y: tensor[height, width, f32]) -> (tensor[seq, f32], tensor[width, f32]) ! {{ IO }} = {{\n _ = print(\"body-ran\")\n (neg(a), add(x, sum(y, 0i32)))\n}}\nout = mixed({args}, to_tensor([1.0f32, 2.0f32]), to_tensor([[3.0f32, 4.0f32]]))\n"
    )
}

#[test]
fn mixed_literal_and_binder_failures_follow_parameter_order() {
    check_both(
        &mixed_literal_source(false, true, true),
        "extent `seq`: a axis 0 = 3, b axis 1 = 2",
        false,
    );
    check_both_context(
        &mixed_literal_source(true, true, true),
        "fixed axis 0 = 3",
        "input `fixed` axis 0 expected 2, got 3",
        false,
    );
}

#[test]
fn mixed_literal_checks_survive_when_the_repeated_binder_agrees() {
    for first in [false, true] {
        check_both_context(
            &mixed_literal_source(first, false, true),
            "fixed axis 0 = 3",
            "input `fixed` axis 0 expected 2, got 3",
            false,
        );
        check_both(
            &mixed_literal_source(first, false, false),
            "out.1 = tensor(shape=[2], data=[4.0, 6.0])",
            true,
        );
    }
}

#[test]
fn every_unused_repeated_witness_is_checked() {
    let source = mixed_source("[1.0f32, 2.0f32]", "[1.0f32, 2.0f32]")
        .replace(
            "x: tensor[width",
            "unused: tensor[depth, height, seq, f32], x: tensor[width",
        )
        .replace(
            "to_tensor([[1.0f32, 2.0f32]]),",
            "to_tensor([[1.0f32, 2.0f32]]), to_tensor([[[1.0f32, 2.0f32, 3.0f32]]]),",
        );
    check_both(
        &source,
        "extent `seq`: a axis 0 = 2, unused axis 2 = 3",
        false,
    );
    check_both(
        &source.replace("[[[1.0f32, 2.0f32, 3.0f32]]]", "[[[1.0f32, 2.0f32]]]"),
        "out.1 = tensor(shape=[2], data=[4.0, 6.0])",
        true,
    );
}

#[test]
fn higher_order_entry_is_after_arguments_and_before_following_effects() {
    for (skip, succeeds) in [(1, true), (2, false)] {
        let source = higher_order_source(skip)
            .replace("def g2(x: tensor[n, f32]) -> tensor[k, f32] = shrink", "def g2(x: tensor[n, f32]) -> tensor[k, f32] ! { IO } = { _ = print(\"argument-ran\")\n shrink")
            .replace("def apply4", "}\ndef apply4")
            .replace("def main() = ", "def main() = {\n result = ")
            + " _ = print(\"body-ran\")\n result\n}\nout = main()\n";
        for c in [false, true] {
            let output = run(&source, c);
            let rendered = text(&output);
            assert_eq!(output.status.success(), succeeds, "C={c}: {rendered}");
            assert_eq!(
                rendered.matches("argument-ran").count(),
                1,
                "C={c}: {rendered}"
            );
            assert_eq!(
                rendered.matches("body-ran").count(),
                usize::from(succeeds),
                "C={c}: {rendered}"
            );
            if !succeeds {
                assert!(
                    rendered.contains("extent `p`: v axis 0 = 2, w axis 0 = 1"),
                    "C={c}: {rendered}"
                );
                assert!(
                    rendered
                        .lines()
                        .any(|line| line == "numeric trap: domain in load at int64"),
                    "C={c}: {rendered}"
                );
            }
        }
    }
}

#[test]
fn inline_callback_entry_precedes_body_effects_on_every_invocation() {
    let source = "def apply(xs: List[tensor[*, f32]]) -> List[tensor[2, f32]] ! { IO } = map(fn (x: tensor[2, f32]) -> { _ = print(\"callback-ran\")\n x }, xs)\nout = apply([to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32, 5.0f32])])\n";
    for unused in [false, true] {
        let source = if unused {
            source.replace("\n x }, xs)", "\n to_tensor([9.0f32, 10.0f32]) }, xs)")
        } else {
            source.into()
        };
        for c in [false, true] {
            let failed = run(&source, c);
            let rendered = text(&failed);
            assert!(!failed.status.success(), "C={c}: {rendered}");
            assert!(
                rendered
                    .lines()
                    .any(|line| line == "numeric trap: domain in load at int64"),
                "C={c}: {rendered}"
            );
            assert_eq!(
                rendered.matches("callback-ran").count(),
                1,
                "C={c}: {rendered}"
            );
            let agreed = run(&source.replace(", 5.0f32", ""), c);
            let rendered = text(&agreed);
            assert!(agreed.status.success(), "C={c}: {rendered}");
            assert_eq!(
                rendered.matches("callback-ran").count(),
                2,
                "C={c}: {rendered}"
            );
            let expected = if unused {
                "out = [tensor(shape=[2], data=[9.0, 10.0]), tensor(shape=[2], data=[9.0, 10.0])]"
            } else {
                "out = [tensor(shape=[2], data=[1.0, 2.0]), tensor(shape=[2], data=[3.0, 4.0])]"
            };
            assert!(rendered.contains(expected), "C={c}: {rendered}");
        }
    }
}

#[test]
fn successive_higher_order_invocations_have_independent_signature_instances() {
    let source = higher_order_source(1);
    let (definitions, call) = source.split_once("def main() = ").unwrap();
    let call = call.trim();
    let longer_first = call.replace(
        "[1.0f32, 2.0f32, 3.0f32]",
        "[1.0f32, 2.0f32, 3.0f32, 4.0f32]",
    );
    let longer_both = longer_first.replace(
        "[7.0f32, 8.0f32, 9.0f32]",
        "[7.0f32, 8.0f32, 9.0f32, 10.0f32]",
    );
    check_both(
        &format!("{definitions}def main() = ({call}, {longer_both})\n"),
        "main.1 = tensor(shape=[3], data=[20.0, 24.0, 28.0])",
        true,
    );
    check_both(
        &format!("{definitions}def main() = ({call}, {longer_first})\n"),
        "extent `p`: v axis 0 = 3, w axis 0 = 2",
        false,
    );
}
