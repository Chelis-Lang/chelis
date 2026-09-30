//! chelis#2751, spec/04 §4.7: a retained callable's named result uses
//! this invocation's formal-argument witness at the result producer.
//! These are result obligations, independent of List entry admission.

mod common;
#[allow(dead_code)]
#[path = "common/result_claims.rs"]
mod result_claims;

fn assert_result_trap(source: &str, witness: usize, actual: usize) {
    let observations = [false, true].map(|native| (native, result_claims::run(source, native)));
    let both = format!(
        "Eval:\n{}\nC:\n{}",
        observations[0].1.1, observations[1].1.1
    );
    for (native, (ok, output)) in observations {
        assert!(!ok, "native={native}\n{source}\n{both}");
        assert!(output.contains("extent `seq`"), "native={native}: {output}");
        assert!(
            output.contains(&format!("axis 0 = {witness}"))
                && output.contains(&format!("axis 0 = {actual}")),
            "native={native}: {output}"
        );
        assert!(
            output
                .lines()
                .any(|line| line == "numeric trap: domain in load at i64"),
            "native={native}: {output}"
        );
        assert!(!output.contains("out ="), "native={native}: {output}");
    }
}

fn assert_result_value(source: &str, values: &str) {
    for native in [false, true] {
        let (ok, output) = result_claims::run(source, native);
        assert!(ok, "native={native}\n{source}\n{output}");
        assert!(
            output
                .lines()
                .any(|line| line == format!("out = tensor(shape=[{values})")),
            "native={native}: {output}"
        );
        assert!(
            !output.contains("numeric trap:"),
            "native={native}: {output}"
        );
    }
}

const DIRECT: &str = "\
def broad(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] = y
def invoke(f: tensor[seq, f32] -> tensor[*, f32] -> tensor[seq, f32], x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] = f(x, y)
";

#[test]
fn direct_witness_guards_retained_result_on_eval_and_linked_c() {
    assert_result_value(
        &format!("{DIRECT}out = invoke(broad, to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0]))\n"),
        "2], data=[4.0, 5.0]",
    );
    assert_result_trap(
        &format!(
            "{DIRECT}out = invoke(broad, to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]))\n"
        ),
        2,
        3,
    );
}

const LIST: &str = "\
def broad(xs: List[tensor[*, f32]], y: tensor[*, f32]) -> tensor[*, f32] = y
def invoke(f: List[tensor[seq, f32]] -> tensor[*, f32] -> tensor[seq, f32], xs: List[tensor[*, f32]], y: tensor[*, f32]) -> tensor[*, f32] = f(xs, y)
";

#[test]
fn list_witness_guards_retained_result_when_list_entry_agrees() {
    assert_result_value(
        &format!("{LIST}out = invoke(broad, [to_tensor([1.0, 2.0])], to_tensor([4.0, 5.0]))\n"),
        "2], data=[4.0, 5.0]",
    );
    assert_result_trap(
        &format!(
            "{LIST}out = invoke(broad, [to_tensor([1.0, 2.0])], to_tensor([4.0, 5.0, 6.0]))\n"
        ),
        2,
        3,
    );
}

#[test]
fn nested_adapter_keeps_the_inner_callable_result_obligation() {
    let prefix = "\
def broad(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] = y
def adapter(f: tensor[seq, f32] -> tensor[*, f32] -> tensor[seq, f32], x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] = f(x, y)
def outer(f: tensor[seq, f32] -> tensor[*, f32] -> tensor[seq, f32], x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] = adapter(f, x, y)
";
    assert_result_value(
        &format!("{prefix}out = outer(broad, to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0]))\n"),
        "2], data=[4.0, 5.0]",
    );
    assert_result_trap(
        &format!("{prefix}out = outer(broad, to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]))\n"),
        2,
        3,
    );
}

#[test]
fn result_guard_runs_at_producer_before_later_effects() {
    let prefix = "\
def broad(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] ! { IO } = {
  _ = print(\"producer-before\")
  result = y
  _ = print(\"producer-after\")
  result
}
def invoke(f: tensor[seq, f32] -> tensor[*, f32] -> tensor[seq, f32], x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] ! { IO } = {
  _ = print(\"caller-before\")
  result = f(x, y)
  _ = print(\"caller-after\")
  result
}
";
    for agrees in [true, false] {
        let values = if agrees {
            "[4.0, 5.0]"
        } else {
            "[4.0, 5.0, 6.0]"
        };
        let source =
            format!("{prefix}out = invoke(broad, to_tensor([1.0, 2.0]), to_tensor({values}))\n");
        let observations =
            [false, true].map(|native| (native, result_claims::run(&source, native)));
        let both = format!(
            "Eval:\n{}\nC:\n{}",
            observations[0].1.1, observations[1].1.1
        );
        for (native, (ok, output)) in observations {
            assert_eq!(ok, agrees, "native={native}\n{source}\n{both}");
            for marker in ["caller-before", "producer-before"] {
                assert_eq!(
                    output.matches(marker).count(),
                    1,
                    "native={native}: {output}"
                );
            }
            assert!(
                output.find("caller-before") < output.find("producer-before"),
                "native={native}: {output}"
            );
            for marker in ["producer-after", "caller-after"] {
                assert_eq!(
                    output.matches(marker).count(),
                    usize::from(agrees),
                    "native={native}: {output}"
                );
            }
            if !agrees {
                assert!(
                    output
                        .lines()
                        .any(|line| line == "numeric trap: domain in load at i64"),
                    "native={native}: {output}"
                );
                assert!(!output.contains("out ="), "native={native}: {output}");
            }
        }
    }
}

#[test]
fn earlier_matching_invocation_cannot_supply_later_result_witness() {
    let source = format!(
        "{DIRECT}first = invoke(broad, to_tensor([1.0, 2.0, 3.0]), to_tensor([4.0, 5.0, 6.0]))\n\
         out = invoke(broad, to_tensor([1.0, 2.0]), to_tensor([7.0, 8.0, 9.0]))\n"
    );
    let observations = [false, true].map(|native| (native, result_claims::run(&source, native)));
    let both = format!(
        "Eval:\n{}\nC:\n{}",
        observations[0].1.1, observations[1].1.1
    );
    for (native, (ok, output)) in observations {
        assert!(!ok, "native={native}: {both}");
        assert!(
            output.contains("first = tensor(shape=[3], data=[4.0, 5.0, 6.0])"),
            "native={native}: {output}"
        );
        assert!(output.contains("extent `seq`"), "native={native}: {output}");
        assert!(
            output.contains("axis 0 = 2") && output.contains("axis 0 = 3"),
            "native={native}: {output}"
        );
        assert!(
            output
                .lines()
                .any(|line| line == "numeric trap: domain in load at i64"),
            "native={native}: {output}"
        );
        assert!(!output.contains("out ="), "native={native}: {output}");
    }
}

#[test]
fn later_list_element_fails_at_entry_before_body_or_result() {
    let prefix = "\
def broad(xs: List[tensor[*, f32]], y: tensor[*, f32]) -> tensor[*, f32] ! { IO } = {
  _ = print(\"body-ran\")
  y
}
def invoke(f: List[tensor[seq, f32]] -> tensor[*, f32] -> tensor[seq, f32], xs: List[tensor[*, f32]], y: tensor[*, f32]) -> tensor[*, f32] ! { IO } = f(xs, y)
";
    assert_result_value(
        &format!(
            "{prefix}out = invoke(broad, [to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0])], to_tensor([5.0, 6.0]))\n"
        ),
        "2], data=[5.0, 6.0]",
    );
    let source = format!(
        "{prefix}out = invoke(broad, [to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0, 5.0])], to_tensor([6.0, 7.0, 8.0]))\n"
    );
    for native in [false, true] {
        let (ok, output) = result_claims::run(&source, native);
        assert!(!ok, "native={native}\n{source}\n{output}");
        assert!(
            output.contains("extent `seq`: arg0[0] axis 0 = 2, arg0[1] axis 0 = 3"),
            "native={native}: {output}"
        );
        assert!(
            output
                .lines()
                .any(|line| line == "numeric trap: domain in load at i64"),
            "native={native}: {output}"
        );
        assert!(!output.contains("body-ran"), "native={native}: {output}");
        assert!(!output.contains("out ="), "native={native}: {output}");
    }
}

#[test]
fn aliased_formal_result_and_supplied_callable_own_claim_are_distinct() {
    let prefix = "\
type Action[n] = tensor[n, f32] -> tensor[*, f32] -> tensor[n, f32]
def own(x: tensor[*, f32], y: tensor[*, f32]) -> tensor[3, f32] = y
def invoke[n](f: Action[n], x: tensor[*, f32], y: tensor[*, f32]) -> tensor[*, f32] = f(x, y)
";
    assert_result_value(
        &format!(
            "{prefix}out = invoke(own, to_tensor([1.0, 2.0, 3.0]), to_tensor([4.0, 5.0, 6.0]))\n"
        ),
        "3], data=[4.0, 5.0, 6.0]",
    );
    let source =
        format!("{prefix}out = invoke(own, to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0]))\n");
    for native in [false, true] {
        let (ok, output) = result_claims::run(&source, native);
        assert!(!ok, "native={native}\n{source}\n{output}");
        assert!(
            output.contains("extent `3`: claimed = 3, load axis 0 = 2"),
            "native={native}: {output}"
        );
        assert!(
            output
                .lines()
                .any(|line| line == "numeric trap: domain in load at i64"),
            "native={native}: {output}"
        );
        assert!(!output.contains("out ="), "native={native}: {output}");
    }
    assert_result_trap(
        &format!("{prefix}out = invoke(own, to_tensor([1.0, 2.0]), to_tensor([4.0, 5.0, 6.0]))\n"),
        2,
        3,
    );
}

#[test]
fn selected_if_producer_waits_for_earlier_effect_and_guards_only_selected_arm() {
    let prefix = "\
def choose(x: tensor[*, f32], y: tensor[*, f32], z: tensor[*, f32], second: bool) -> tensor[*, f32] ! { IO } = {
  _ = print(\"before-choice\")
  selected = if second then {
    _ = print(\"selected-arm\")
    relu(y)
  } else {
    _ = print(\"other-arm\")
    neg(z)
  }
  _ = print(\"after-choice\")
  selected
}
def invoke(f: tensor[seq, f32] -> tensor[*, f32] -> tensor[*, f32] -> bool -> tensor[seq, f32], x: tensor[*, f32], y: tensor[*, f32], z: tensor[*, f32], second: bool) -> tensor[*, f32] ! { IO } = f(x, y, z, second)
";
    for second in [false, true] {
        let source = format!(
            "{prefix}out = invoke(choose, to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0, 5.0]), to_tensor([6.0, 7.0]), {second})\n"
        );
        let observations =
            [false, true].map(|native| (native, result_claims::run(&source, native)));
        let both = format!(
            "Eval:\n{}\nC:\n{}",
            observations[0].1.1, observations[1].1.1
        );
        for (native, (ok, output)) in observations {
            assert_eq!(ok, !second, "native={native}\n{source}\n{both}");
            assert_eq!(
                output.matches("before-choice").count(),
                1,
                "native={native}: {output}"
            );
            assert_eq!(
                output.matches("selected-arm").count(),
                usize::from(second),
                "native={native}: {output}"
            );
            assert_eq!(
                output.matches("other-arm").count(),
                usize::from(!second),
                "native={native}: {output}"
            );
            assert_eq!(
                output.matches("after-choice").count(),
                usize::from(!second),
                "native={native}: {output}"
            );
            if second {
                assert!(output.contains("extent `seq`"), "native={native}: {output}");
                assert!(
                    output.contains("axis 0 = 2") && output.contains("relu axis 0 = 3"),
                    "native={native}: {output}"
                );
                assert!(
                    output
                        .lines()
                        .any(|line| line == "numeric trap: domain in relu at i64"),
                    "native={native}: {output}"
                );
                assert!(!output.contains("out ="), "native={native}: {output}");
            } else {
                assert!(
                    output.contains("out = tensor(shape=[2], data=[-6.0, -7.0])"),
                    "native={native}: {output}"
                );
                assert!(
                    !output.contains("numeric trap:"),
                    "native={native}: {output}"
                );
            }
        }
    }
}
