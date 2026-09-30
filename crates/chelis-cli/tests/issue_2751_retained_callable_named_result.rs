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
