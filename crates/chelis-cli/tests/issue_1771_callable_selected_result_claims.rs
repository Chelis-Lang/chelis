//! Spec/04 section 4.7 and [04-NUM-9]: an inherited literal result
//! obligation reaches the selected producer through supported callable calls
//! and delayed selection.  The obligation is invocation-local and guards only
//! the selected value, at the first source position where that choice is known.

mod common;
#[path = "common/result_claims.rs"]
mod result_claims;

use assert_cmd::Command;
use result_claims::{assert_claim, run};
use std::fs;
use std::process::Command as StdCommand;

const TWO: &str = "[1.0f32, 2.0f32, 3.0f32]";
const THREE: &str = "[1.0f32, 2.0f32, 3.0f32, 4.0f32]";
const TWO_BY_FOUR: &str = "[[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]]";
const THREE_BY_FOUR: &str = "[[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32], [9.0f32, 10.0f32, 11.0f32, 12.0f32]]";

fn callable_source(values: &str, literal: bool) -> String {
    let call = if literal {
        format!(
            "{{\n local = fn (value: tensor[*, f32]) -> {{\n _ = print(\"producer-before\")\n produced = shrink(value, [[1i64, shape(value, 0i32)]])\n _ = print(\"producer-after\")\n produced\n}}\n claimed(local, to_tensor({values}))\n}}"
        )
    } else {
        format!("claimed(cut, to_tensor({values}))")
    };
    format!(
        "def cut[n](value: tensor[n, f32]) -> tensor[*, f32] ! {{ IO }} = {{\n _ = print(\"producer-before\")\n produced = shrink(value, [[1i64, shape(value, 0i32)]])\n _ = print(\"producer-after\")\n produced\n}}\n\
         def invoke(f: tensor[*, f32] -> tensor[*, f32], value: tensor[*, f32]) -> tensor[*, f32] ! {{ IO }} = f(value)\n\
         def claimed[n](f: tensor[*, f32] -> tensor[*, f32], value: tensor[n, f32]) -> tensor[3, f32] ! {{ IO }} = {{\n _ = print(\"caller-before\")\n result = invoke(f, value)\n _ = print(\"caller-after\")\n result\n}}\n\
         out = {call}\n"
    )
}

fn assert_callable_transport(native: bool) {
    for literal in [false, true] {
        for (values, agrees) in [(THREE, true), (TWO, false)] {
            let source = callable_source(values, literal);
            let (ok, output) = run(&source, native);
            assert_eq!(ok, agrees, "{source}\n{output}");
            assert_eq!(output.matches("caller-before").count(), 1, "{output}");
            assert_eq!(output.matches("producer-before").count(), 1, "{output}");
            assert_eq!(
                output.matches("producer-after").count(),
                usize::from(agrees),
                "{output}"
            );
            assert_eq!(
                output.matches("caller-after").count(),
                usize::from(agrees),
                "{output}"
            );
            if agrees {
                assert!(
                    output.contains("out = tensor(shape=[3], data=[2.0, 3.0, 4.0])"),
                    "{output}"
                );
                assert!(!output.contains("numeric trap:"), "{output}");
            } else {
                assert_claim(&output, "shrink", 2);
            }
        }
    }
}

#[test]
fn eval_inherited_claim_reaches_named_and_literal_callbacks() {
    assert_callable_transport(false);
}

#[test]
fn c_inherited_claim_reaches_named_and_literal_callbacks() {
    assert_callable_transport(true);
}

fn repeated_callable_source(last: &str) -> String {
    format!(
        "def cut[n](value: tensor[n, f32]) -> tensor[*, f32] ! {{ IO }} = {{\n _ = print(\"invocation-before\")\n result = shrink(value, [[1i64, shape(value, 0i32)]])\n _ = print(\"invocation-after\")\n result\n}}\n\
         def invoke(f: tensor[*, f32] -> tensor[*, f32], value: tensor[*, f32]) -> tensor[*, f32] ! {{ IO }} = f(value)\n\
         def two[n](f: tensor[*, f32] -> tensor[*, f32], value: tensor[n, f32]) -> tensor[2, f32] ! {{ IO }} = invoke(f, value)\n\
         def three[n](f: tensor[*, f32] -> tensor[*, f32], value: tensor[n, f32]) -> tensor[3, f32] ! {{ IO }} = invoke(f, value)\n\
         a = two(cut, to_tensor({TWO}))\n\
         b = three(cut, to_tensor({THREE}))\n\
         out = three(cut, to_tensor({last}))\n"
    )
}

fn assert_invocation_isolation(native: bool) {
    for (last, agrees) in [(THREE, true), (TWO, false)] {
        let source = repeated_callable_source(last);
        let (ok, output) = run(&source, native);
        assert_eq!(ok, agrees, "{source}\n{output}");
        assert_eq!(output.matches("invocation-before").count(), 3, "{output}");
        assert_eq!(
            output.matches("invocation-after").count(),
            if agrees { 3 } else { 2 },
            "{output}"
        );
        if agrees {
            assert!(
                output.contains("a = tensor(shape=[2], data=[2.0, 3.0])"),
                "{output}"
            );
            assert!(
                output.contains("b = tensor(shape=[3], data=[2.0, 3.0, 4.0])"),
                "{output}"
            );
            assert!(
                output.contains("out = tensor(shape=[3], data=[2.0, 3.0, 4.0])"),
                "{output}"
            );
        } else {
            assert_claim(&output, "shrink", 2);
        }
    }
}

#[test]
fn eval_shared_callback_claims_are_invocation_local() {
    assert_invocation_isolation(false);
}

#[test]
fn c_shared_callback_claims_are_invocation_local() {
    assert_invocation_isolation(true);
}

fn identity_callback_source(values: &str) -> String {
    format!(
        "def prepare[n](value: tensor[n, f32]) -> tensor[*, f32] ! {{ IO }} = {{\n  \
         _ = print(\"actual-before\")\n  \
         produced = shrink(value, [[1i64, shape(value, 0i32)]])\n  \
         _ = print(\"actual-after\")\n  \
         produced\n}}\n\
         def identity(value: tensor[*, f32]) -> tensor[*, f32] = value\n\
         def claimed[n](f: tensor[*, f32] -> tensor[*, f32], value: tensor[n, f32]) -> tensor[3, f32] ! {{ IO }} = {{\n  \
         _ = print(\"caller-before\")\n  \
         result = f(prepare(value))\n  \
         _ = print(\"caller-after\")\n  \
         result\n}}\n\
         out = claimed(identity, to_tensor({values}))\n"
    )
}

fn assert_identity_callback_checks_after_actual_preparation(native: bool) {
    for (values, agrees) in [(THREE, true), (TWO, false)] {
        let source = identity_callback_source(values);
        let (ok, output) = run(&source, native);
        assert_eq!(ok, agrees, "{source}\n{output}");
        assert_eq!(output.matches("caller-before").count(), 1, "{output}");
        assert_eq!(output.matches("actual-before").count(), 1, "{output}");
        assert_eq!(output.matches("actual-after").count(), 1, "{output}");
        assert_eq!(
            output.matches("caller-after").count(),
            usize::from(agrees),
            "{output}"
        );
        if agrees {
            assert!(
                output.contains("out = tensor(shape=[3], data=[2.0, 3.0, 4.0])"),
                "{output}"
            );
        } else {
            assert_claim(&output, "load", 2);
        }
    }
}

fn assert_markers_in_order(output: &str, markers: &[&str]) {
    let mut prior = None;
    for marker in markers {
        let position = output
            .find(marker)
            .unwrap_or_else(|| panic!("missing {marker}: {output}"));
        if let Some(prior) = prior {
            assert!(prior < position, "{marker} ran out of order: {output}");
        }
        prior = Some(position);
    }
}

fn polymorphic_identity_own_claim_source(agrees: bool) -> String {
    let values = if agrees {
        "[[1.0f64, 16777217.0f64], [3.0f64, 4.0f64], [5.0f64, 6.0f64]]"
    } else {
        "[[1.0f64, 16777217.0f64], [3.0f64, 4.0f64]]"
    };
    format!(
        "def identity[rest, p: Float](value: tensor[first, ..rest, p]) -> tensor[3, ..rest, p] = value\n\
         def invoke[p: Float](f: tensor[*, 2, p] -> tensor[*, 2, p], value: tensor[*, 2, p]) -> tensor[*, 2, p] ! {{ IO }} = {{\n\
         _ = print(\"caller-before\")\n\
         result = f({{\n\
         _ = print(\"actual-before\")\n\
         prepared = neg(value)\n\
         _ = print(\"actual-after\")\n\
         prepared\n\
         }})\n\
         _ = print(\"caller-after\")\n\
         result\n\
         }}\n\
         out = invoke(identity, to_tensor({values}))\n"
    )
}

fn polymorphic_identity_inherited_claim_source(agrees: bool) -> String {
    let values = if agrees {
        "[[1.0f64, 16777217.0f64], [3.0f64, 4.0f64], [5.0f64, 6.0f64]]"
    } else {
        "[[1.0f64, 16777217.0f64], [3.0f64, 4.0f64]]"
    };
    format!(
        "def identity[rest, p: Float](value: tensor[..rest, p], ignored: i64) -> tensor[..rest, p] = value\n\
         def invoke[p: Float](f: tensor[*, 2, p] -> i64 -> tensor[*, 2, p], value: tensor[*, 2, p], ignored: i64) -> tensor[3, 2, p] ! {{ IO }} = {{\n\
         _ = print(\"caller-before\")\n\
         result = f({{ _ = print(\"actual-first\")\n\
         value }}, {{ _ = print(\"actual-unused\")\n\
         ignored }})\n\
         _ = print(\"caller-after\")\n\
         result\n\
         }}\n\
         out = invoke(identity, to_tensor({values}), 7i64)\n"
    )
}

#[derive(Clone, Copy)]
enum PolymorphicEntryCase {
    Agrees,
    NamedMismatch,
    SpreadMismatch,
}

fn polymorphic_identity_entry_source(case: PolymorphicEntryCase) -> String {
    let second = match case {
        PolymorphicEntryCase::Agrees => "[[7.0f64, 8.0f64], [9.0f64, 10.0f64], [11.0f64, 12.0f64]]",
        PolymorphicEntryCase::NamedMismatch => "[[7.0f64, 8.0f64], [9.0f64, 10.0f64]]",
        PolymorphicEntryCase::SpreadMismatch => {
            "[[7.0f64, 8.0f64, 9.0f64], [10.0f64, 11.0f64, 12.0f64], [13.0f64, 14.0f64, 15.0f64]]"
        }
    };
    format!(
        "def first[rest, p: Float](value: tensor[extent, ..rest, p], ignored: tensor[extent, ..rest, p]) -> tensor[extent, ..rest, p] = value\n\
         def invoke[p: Float](f: tensor[*, *, p] -> tensor[*, *, p] -> tensor[*, *, p], value: tensor[*, *, p], ignored: tensor[*, *, p]) -> tensor[*, *, p] ! {{ IO }} = {{\n\
         _ = print(\"caller-before\")\n\
         result = f({{ _ = print(\"entry-first\")\n\
         value }}, {{ _ = print(\"entry-unused\")\n\
         ignored }})\n\
         _ = print(\"caller-after\")\n\
         result\n\
         }}\n\
         out = invoke(first, to_tensor([[1.0f64, 16777217.0f64], [3.0f64, 4.0f64], [5.0f64, 6.0f64]]), to_tensor({second}))\n"
    )
}

fn assert_polymorphic_identity_own_claim(native: bool) {
    for agrees in [true, false] {
        let source = polymorphic_identity_own_claim_source(agrees);
        let (ok, output) = run(&source, native);
        assert_eq!(ok, agrees, "{source}\n{output}");
        let mut expected_order = vec!["caller-before", "actual-before", "actual-after"];
        if agrees {
            expected_order.push("caller-after");
        }
        assert_markers_in_order(&output, &expected_order);
        for marker in ["caller-before", "actual-before", "actual-after"] {
            assert_eq!(output.matches(marker).count(), 1, "{marker}: {output}");
        }
        assert_eq!(output.matches("caller-after").count(), usize::from(agrees));
        if agrees {
            assert!(
                output.contains("out = tensor(shape=[3, 2]") && output.contains("16777217.0"),
                "{output}"
            );
        } else {
            assert!(
                output.contains("extent `3`: claimed = 3") && output.contains("axis 0 = 2"),
                "{output}"
            );
            assert!(
                output
                    .lines()
                    .any(|line| line == "numeric trap: domain in load at i64"),
                "{output}"
            );
            assert!(!output.contains("out ="), "{output}");
            assert!(!output.contains("domain in neg"), "{output}");
        }
    }
}

fn assert_polymorphic_identity_inherited_claim(native: bool) {
    for agrees in [true, false] {
        let source = polymorphic_identity_inherited_claim_source(agrees);
        let (ok, output) = run(&source, native);
        assert_eq!(ok, agrees, "{source}\n{output}");
        let mut expected_order = vec!["caller-before", "actual-first", "actual-unused"];
        if agrees {
            expected_order.push("caller-after");
        }
        assert_markers_in_order(&output, &expected_order);
        for marker in ["caller-before", "actual-first", "actual-unused"] {
            assert_eq!(output.matches(marker).count(), 1, "{marker}: {output}");
        }
        assert_eq!(output.matches("caller-after").count(), usize::from(agrees));
        if agrees {
            assert!(output.contains("out = tensor(shape=[3, 2]"), "{output}");
        } else {
            assert_claim(&output, "load", 2);
        }
    }
}

fn assert_polymorphic_identity_entry(native: bool) {
    for case in [
        PolymorphicEntryCase::Agrees,
        PolymorphicEntryCase::NamedMismatch,
        PolymorphicEntryCase::SpreadMismatch,
    ] {
        let agrees = matches!(case, PolymorphicEntryCase::Agrees);
        let source = polymorphic_identity_entry_source(case);
        let (ok, output) = run(&source, native);
        assert_eq!(ok, agrees, "{source}\n{output}");
        let mut expected_order = vec!["caller-before", "entry-first", "entry-unused"];
        if agrees {
            expected_order.push("caller-after");
        }
        assert_markers_in_order(&output, &expected_order);
        for marker in ["caller-before", "entry-first", "entry-unused"] {
            assert_eq!(output.matches(marker).count(), 1, "{marker}: {output}");
        }
        assert_eq!(output.matches("caller-after").count(), usize::from(agrees));
        if agrees {
            assert!(output.contains("out = tensor(shape=[3, 2]"), "{output}");
        } else {
            let (axis, first, later) = match case {
                PolymorphicEntryCase::NamedMismatch => (0, 3, 2),
                PolymorphicEntryCase::SpreadMismatch => (1, 2, 3),
                PolymorphicEntryCase::Agrees => unreachable!(),
            };
            assert!(
                output.contains(&format!("value axis {axis} = {first}"))
                    && output.contains(&format!("ignored axis {axis} = {later}")),
                "{output}"
            );
            assert!(
                output
                    .lines()
                    .any(|line| line == "numeric trap: domain in load at i64"),
                "{output}"
            );
            assert!(!output.contains("out ="), "{output}");
        }
    }
}

#[test]
fn eval_identity_callback_checks_claim_after_actual_preparation() {
    assert_identity_callback_checks_after_actual_preparation(false);
}

#[test]
fn c_identity_callback_checks_claim_after_actual_preparation() {
    assert_identity_callback_checks_after_actual_preparation(true);
}

#[test]
fn eval_polymorphic_identity_own_claim_preserves_formal_boundary() {
    assert_polymorphic_identity_own_claim(false);
}

#[test]
fn c_polymorphic_identity_own_claim_preserves_formal_boundary() {
    assert_polymorphic_identity_own_claim(true);
}

#[test]
fn eval_polymorphic_identity_inherited_claim_prepares_all_actuals() {
    assert_polymorphic_identity_inherited_claim(false);
}

#[test]
fn c_polymorphic_identity_inherited_claim_prepares_all_actuals() {
    assert_polymorphic_identity_inherited_claim(true);
}

#[test]
fn eval_polymorphic_identity_entry_precedes_body_and_result() {
    assert_polymorphic_identity_entry(false);
}

#[test]
fn c_polymorphic_identity_entry_precedes_body_and_result() {
    assert_polymorphic_identity_entry(true);
}

#[test]
fn rank_zero_identity_without_a_result_claim_executes_both_lanes() {
    let source = "def identity(value: tensor[f32]) -> tensor[f32] = value\n\
                  out = identity(scalar_to_tensor(7.0f32))\n";
    for native in [false, true] {
        let (ok, output) = run(source, native);
        assert!(ok, "{source}\n{output}");
        assert!(output.contains("out = 7"), "{output}");
        assert!(!output.contains("pending result claim"), "{output}");
    }
}

#[test]
fn c_rank_zero_host_identity_without_a_claim_preserves_value_and_effects() {
    let source = "def identity(value: tensor[f32]) -> tensor[f32] ! { IO } = {\n\
                  _ = print(\"rank-zero-before\")\n\
                  alias = value\n\
                  _ = print(\"rank-zero-after\")\n\
                  alias\n\
                  }\n\
                  out = identity(scalar_to_tensor(7.0f32))\n";
    let (ok, output) = run(source, true);
    assert!(ok, "{source}\n{output}");
    assert_eq!(output.matches("rank-zero-before").count(), 1, "{output}");
    assert_eq!(output.matches("rank-zero-after").count(), 1, "{output}");
    assert!(!output.contains("numeric trap:"), "{output}");
    let out = output
        .lines()
        .find_map(|line| line.strip_prefix("out = "))
        .unwrap_or_else(|| panic!("missing rank-zero result in:\n{output}"));
    assert!(
        !out.contains("tensor("),
        "rank-zero result was not scalar: {output}"
    );
    assert_eq!(out.parse::<f32>().unwrap(), 7.0, "{output}");
}

fn delayed_selection_source(form: &str, select_second: bool, selected_agrees: bool) -> String {
    let chosen = if selected_agrees {
        THREE_BY_FOUR
    } else {
        TWO_BY_FOUR
    };
    let unchosen = if selected_agrees {
        TWO_BY_FOUR
    } else {
        THREE_BY_FOUR
    };
    let (x, y) = if select_second {
        (unchosen, chosen)
    } else {
        (chosen, unchosen)
    };
    let selection = match form {
        "if" => {
            "if decide(second) then { _ = drop(first)\n second_value } else { _ = drop(second_value)\n first }"
        }
        "match" => {
            "match decide(second) with {\n | true => { _ = drop(first)\n second_value }\n | _ => { _ = drop(second_value)\n first }\n}"
        }
        _ => panic!("unknown delayed selection form"),
    };
    format!(
        "def decide(flag: bool) -> bool ! {{ IO }} = {{\n _ = print(\"selector-ran\")\n flag\n}}\n\
         def choose[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32], second: bool) -> tensor[3, f32] ! {{ IO }} = {{\n _ = print(\"diagonal-before\")\n first = diagonal(x, 0i32, 1i32)\n _ = print(\"diagonal-after\")\n _ = print(\"cumsum-before\")\n second_value = cumsum(diagonal(y, 0i32, 1i32), 0i32)\n _ = print(\"cumsum-after\")\n selected = {selection}\n alias = selected\n final_value = alias\n _ = print(\"selection-after\")\n final_value\n}}\n\
         out = choose(to_tensor({x}), to_tensor({y}), {select_second})\n"
    )
}

fn composed_delayed_selection_source(select_second: bool, selected_agrees: bool) -> String {
    let chosen = if selected_agrees {
        THREE_BY_FOUR
    } else {
        TWO_BY_FOUR
    };
    let unchosen = if selected_agrees {
        TWO_BY_FOUR
    } else {
        THREE_BY_FOUR
    };
    let (x, y) = if select_second {
        (unchosen, chosen)
    } else {
        (chosen, unchosen)
    };
    format!(
        "def decide(flag: bool) -> bool ! {{ IO }} = {{\n _ = print(\"selector-ran\")\n flag\n}}\n\
         def produce[n](x: tensor[n, 4, f32], summed: bool) -> tensor[*, f32] ! {{ IO }} = if summed then {{\n _ = print(\"cumsum-producer\")\n cumsum(diagonal(x, 0i32, 1i32), 0i32)\n }} else {{\n _ = print(\"diagonal-producer\")\n diagonal(x, 0i32, 1i32)\n }}\n\
         def choose[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32], second: bool) -> tensor[3, f32] ! {{ IO }} = {{\n first = produce(x, false)\n saved = first\n first = produce(y, true)\n selected = if decide(second) then {{ _ = drop(saved)\n first }} else {{ _ = drop(first)\n saved }}\n alias = selected\n _ = print(\"selection-after\")\n alias\n}}\n\
         out = choose(to_tensor({x}), to_tensor({y}), {select_second})\n"
    )
}

fn pure_helper_delayed_selection_source(select_second: bool, selected_agrees: bool) -> String {
    let chosen = if selected_agrees { THREE } else { TWO };
    let unchosen = if selected_agrees { TWO } else { THREE };
    let (x, y) = if select_second {
        (unchosen, chosen)
    } else {
        (chosen, unchosen)
    };
    format!(
        "def cut[n](x: tensor[n, f32]) -> tensor[*, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
         def doubled[n](x: tensor[n, f32]) -> tensor[*, f32] = {{\n value = shrink(x, [[1i64, shape(x, 0i32)]])\n add(value, value)\n}}\n\
         def choose[n, m](x: tensor[n, f32], y: tensor[m, f32], second: bool) -> tensor[3, f32] ! {{ IO }} = {{\n first = cut(x)\n saved = first\n first = doubled(y)\n _ = print(\"helpers-produced\")\n selected = if second then {{ _ = drop(saved)\n first }} else {{ _ = drop(first)\n saved }}\n alias = selected\n _ = print(\"selection-after\")\n alias\n}}\n\
         out = choose(to_tensor({x}), to_tensor({y}), {select_second})\n"
    )
}

fn identity_delayed_selection_source(select_identity: bool, selected_agrees: bool) -> String {
    let identity_selected = if selected_agrees { TWO } else { THREE };
    let identity_unselected = if selected_agrees { THREE } else { TWO };
    let cut_selected = if selected_agrees { THREE } else { TWO };
    let cut_unselected = if selected_agrees { TWO } else { THREE };
    let (identity_value, cut_value) = if select_identity {
        (identity_selected, cut_unselected)
    } else {
        (identity_unselected, cut_selected)
    };
    format!(
        "def forwarded[n](x: tensor[n, f32]) -> tensor[*, f32] = x\n\
         def cut[n](x: tensor[n, f32]) -> tensor[*, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
         def choose(x: tensor[*, f32], y: tensor[*, f32], identity: bool) -> tensor[3, f32] ! {{ IO }} = {{\n identity_value = forwarded(x)\n cut_value = cut(y)\n _ = print(\"identity-and-cut-produced\")\n selected = if identity then {{ _ = drop(cut_value)\n identity_value }} else {{ _ = drop(identity_value)\n cut_value }}\n _ = print(\"identity-selected\")\n selected\n}}\n\
         out = choose(to_tensor({identity_value}), to_tensor({cut_value}), {select_identity})\n"
    )
}

fn assert_delayed_selection(native: bool) {
    for form in ["if", "match"] {
        for select_second in [false, true] {
            for selected_agrees in [true, false] {
                let source = delayed_selection_source(form, select_second, selected_agrees);
                let (ok, output) = run(&source, native);
                assert_eq!(ok, selected_agrees, "{source}\n{output}");
                for effect in [
                    "diagonal-before",
                    "diagonal-after",
                    "cumsum-before",
                    "cumsum-after",
                    "selector-ran",
                ] {
                    assert_eq!(output.matches(effect).count(), 1, "{effect}: {output}");
                }
                assert_eq!(
                    output.matches("selection-after").count(),
                    usize::from(selected_agrees),
                    "{output}"
                );
                let selected_op = if select_second { "cumsum" } else { "diagonal" };
                let unselected_op = if select_second { "diagonal" } else { "cumsum" };
                if selected_agrees {
                    let values = if select_second {
                        "[1.0, 7.0, 18.0]"
                    } else {
                        "[1.0, 6.0, 11.0]"
                    };
                    assert!(
                        output.contains(&format!("out = tensor(shape=[3], data={values})")),
                        "{output}"
                    );
                    assert!(!output.contains("numeric trap:"), "{output}");
                } else {
                    assert_claim(&output, selected_op, 2);
                    assert!(
                        !output
                            .contains(&format!("numeric trap: domain in {unselected_op} at i64")),
                        "{output}"
                    );
                }
            }
        }
    }
    for select_second in [false, true] {
        for selected_agrees in [true, false] {
            let source = composed_delayed_selection_source(select_second, selected_agrees);
            let (ok, output) = run(&source, native);
            assert_eq!(ok, selected_agrees, "{source}\n{output}");
            for effect in ["diagonal-producer", "cumsum-producer", "selector-ran"] {
                assert_eq!(output.matches(effect).count(), 1, "{effect}: {output}");
            }
            assert_eq!(
                output.matches("selection-after").count(),
                usize::from(selected_agrees),
                "{output}"
            );
            let selected_op = if select_second { "cumsum" } else { "diagonal" };
            if selected_agrees {
                let values = if select_second {
                    "[1.0, 7.0, 18.0]"
                } else {
                    "[1.0, 6.0, 11.0]"
                };
                assert!(
                    output.contains(&format!("out = tensor(shape=[3], data={values})")),
                    "{output}"
                );
            } else {
                assert_claim(&output, selected_op, 2);
            }
        }
    }
    for select_second in [false, true] {
        for selected_agrees in [true, false] {
            let source = pure_helper_delayed_selection_source(select_second, selected_agrees);
            let (ok, output) = run(&source, native);
            assert_eq!(ok, selected_agrees, "{source}\n{output}");
            assert_eq!(output.matches("helpers-produced").count(), 1, "{output}");
            assert_eq!(
                output.matches("selection-after").count(),
                usize::from(selected_agrees),
                "{output}"
            );
            let selected_op = if select_second { "add" } else { "shrink" };
            if selected_agrees {
                let values = if select_second {
                    "[4.0, 6.0, 8.0]"
                } else {
                    "[2.0, 3.0, 4.0]"
                };
                assert!(
                    output.contains(&format!("out = tensor(shape=[3], data={values})")),
                    "{output}"
                );
            } else {
                assert_claim(&output, selected_op, 2);
            }
        }
    }
    for select_identity in [false, true] {
        for selected_agrees in [true, false] {
            let source = identity_delayed_selection_source(select_identity, selected_agrees);
            let (ok, output) = run(&source, native);
            assert_eq!(ok, selected_agrees, "{source}\n{output}");
            assert_eq!(
                output.matches("identity-and-cut-produced").count(),
                1,
                "{output}"
            );
            assert_eq!(
                output.matches("identity-selected").count(),
                usize::from(selected_agrees),
                "{output}"
            );
            if selected_agrees {
                let values = if select_identity {
                    "[1.0, 2.0, 3.0]"
                } else {
                    "[2.0, 3.0, 4.0]"
                };
                assert!(
                    output.contains(&format!("out = tensor(shape=[3], data={values})")),
                    "{output}"
                );
            } else {
                assert_claim(
                    &output,
                    if select_identity { "load" } else { "shrink" },
                    if select_identity { 4 } else { 2 },
                );
            }
        }
    }
}

fn callback_delayed_selection_source(
    literal: bool,
    select_second: bool,
    selected_agrees: bool,
) -> String {
    let chosen = if selected_agrees {
        THREE_BY_FOUR
    } else {
        TWO_BY_FOUR
    };
    let sibling = if selected_agrees {
        TWO_BY_FOUR
    } else {
        THREE_BY_FOUR
    };
    let (x, y) = if select_second {
        (sibling, chosen)
    } else {
        (chosen, sibling)
    };
    let callback = if literal {
        "fn (x: tensor[*, 4, f32], y: tensor[*, 4, f32], second: bool) -> {\n\
         _ = print(\"diagonal-before\")\n\
         first = diagonal(x, 0i32, 1i32)\n\
         _ = print(\"diagonal-after\")\n\
         _ = print(\"cumsum-before\")\n\
         second_value = cumsum(diagonal(y, 0i32, 1i32), 0i32)\n\
         _ = print(\"cumsum-after\")\n\
         selected = if decide(second) then second_value else first\n\
         _ = print(\"callback-after\")\n\
         selected\n\
        }"
        .to_string()
    } else {
        "choose".to_string()
    };
    format!(
        "def decide(flag: bool) -> bool ! {{ IO }} = {{\n\
         _ = print(\"selector-ran\")\n\
         flag\n\
        }}\n\
         def choose[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32], second: bool) -> tensor[*, f32] ! {{ IO }} = {{\n\
         _ = print(\"diagonal-before\")\n\
         first = diagonal(x, 0i32, 1i32)\n\
         _ = print(\"diagonal-after\")\n\
         _ = print(\"cumsum-before\")\n\
         second_value = cumsum(diagonal(y, 0i32, 1i32), 0i32)\n\
         _ = print(\"cumsum-after\")\n\
         selected = if decide(second) then second_value else first\n\
         _ = print(\"callback-after\")\n\
         selected\n\
        }}\n\
         def invoke(f: tensor[*, 4, f32] -> tensor[*, 4, f32] -> bool -> tensor[*, f32], x: tensor[*, 4, f32], y: tensor[*, 4, f32], second: bool) -> tensor[*, f32] ! {{ IO }} = f(x, y, second)\n\
         def claimed[n, m](f: tensor[*, 4, f32] -> tensor[*, 4, f32] -> bool -> tensor[*, f32], x: tensor[n, 4, f32], y: tensor[m, 4, f32], second: bool) -> tensor[3, f32] ! {{ IO }} = {{\n\
         _ = print(\"caller-before\")\n\
         selected = invoke(f, x, y, second)\n\
         _ = print(\"caller-after\")\n\
         selected\n\
        }}\n\
         out = claimed({callback}, to_tensor({x}), to_tensor({y}), {select_second})\n"
    )
}

fn assert_inherited_callback_delayed_selection(native: bool) {
    for literal in [false, true] {
        for select_second in [false, true] {
            for selected_agrees in [true, false] {
                let source =
                    callback_delayed_selection_source(literal, select_second, selected_agrees);
                let (ok, output) = run(&source, native);
                assert_eq!(ok, selected_agrees, "{source}\n{output}");
                for effect in [
                    "caller-before",
                    "diagonal-before",
                    "diagonal-after",
                    "cumsum-before",
                    "cumsum-after",
                    "selector-ran",
                ] {
                    assert_eq!(output.matches(effect).count(), 1, "{effect}: {output}");
                }
                let mut prior = None;
                for effect in [
                    "caller-before",
                    "diagonal-before",
                    "diagonal-after",
                    "cumsum-before",
                    "cumsum-after",
                    "selector-ran",
                ] {
                    let position = output
                        .find(effect)
                        .unwrap_or_else(|| panic!("missing {effect}: {output}"));
                    if let Some(prior) = prior {
                        assert!(prior < position, "{effect} ran out of order: {output}");
                    }
                    prior = Some(position);
                }
                assert_eq!(
                    output.matches("callback-after").count(),
                    usize::from(selected_agrees),
                    "{output}"
                );
                assert_eq!(
                    output.matches("caller-after").count(),
                    usize::from(selected_agrees),
                    "{output}"
                );
                if selected_agrees {
                    let callback_after = output.find("callback-after").expect("callback effect");
                    let caller_after = output.find("caller-after").expect("caller effect");
                    assert!(
                        prior.expect("pre-selection effects") < callback_after
                            && callback_after < caller_after,
                        "post-selection effects ran out of order: {output}"
                    );
                    let expected = if select_second {
                        "[1.0, 7.0, 18.0]"
                    } else {
                        "[1.0, 6.0, 11.0]"
                    };
                    assert!(
                        output.contains(&format!("out = tensor(shape=[3], data={expected})")),
                        "{output}"
                    );
                    assert!(!output.contains("numeric trap:"), "{output}");
                } else {
                    assert_claim(
                        &output,
                        if select_second { "cumsum" } else { "diagonal" },
                        2,
                    );
                }
            }
        }
    }
}

#[test]
fn eval_inherited_callback_claim_waits_for_body_selection() {
    assert_inherited_callback_delayed_selection(false);
}

#[test]
fn c_inherited_callback_claim_waits_for_body_selection() {
    assert_inherited_callback_delayed_selection(true);
}

#[test]
fn eval_delayed_selection_guards_only_the_selected_value() {
    assert_delayed_selection(false);
}

#[test]
fn c_delayed_selection_guards_only_the_selected_value() {
    assert_delayed_selection(true);
}

fn selection_before_production_source(select_second: bool, agrees: bool) -> String {
    let selected = if agrees { THREE_BY_FOUR } else { TWO_BY_FOUR };
    let unselected = if agrees { TWO_BY_FOUR } else { THREE_BY_FOUR };
    let (x, y) = if select_second {
        (unselected, selected)
    } else {
        (selected, unselected)
    };
    format!(
        "def decide(flag: bool) -> bool ! {{ IO }} = {{\n _ = print(\"selector-ran\")\n flag\n}}\n\
         def choose[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32], second: bool) -> tensor[3, f32] ! {{ IO }} = {{\n selected = if decide(second) then {{\n _ = print(\"cumsum-before\")\n value = cumsum(diagonal(y, 0i32, 1i32), 0i32)\n _ = print(\"cumsum-after\")\n value\n }} else {{\n _ = print(\"diagonal-before\")\n value = diagonal(x, 0i32, 1i32)\n _ = print(\"diagonal-after\")\n value\n }}\n _ = print(\"selection-after\")\n selected\n}}\n\
         out = choose(to_tensor({x}), to_tensor({y}), {select_second})\n"
    )
}

fn assert_selection_before_production(native: bool) {
    for select_second in [false, true] {
        for agrees in [true, false] {
            let source = selection_before_production_source(select_second, agrees);
            let (ok, output) = run(&source, native);
            assert_eq!(ok, agrees, "{source}\n{output}");
            assert_eq!(output.matches("selector-ran").count(), 1, "{output}");
            let (selected_op, unselected_op, expected) = if select_second {
                ("cumsum", "diagonal", "[1.0, 7.0, 18.0]")
            } else {
                ("diagonal", "cumsum", "[1.0, 6.0, 11.0]")
            };
            assert_eq!(
                output.matches(&format!("{selected_op}-before")).count(),
                1,
                "{output}"
            );
            assert!(
                !output.contains(&format!("{unselected_op}-before")),
                "{output}"
            );
            assert_eq!(
                output.matches(&format!("{selected_op}-after")).count(),
                usize::from(agrees),
                "{output}"
            );
            assert_eq!(
                output.matches("selection-after").count(),
                usize::from(agrees),
                "{output}"
            );
            if agrees {
                assert!(
                    output.contains(&format!("out = tensor(shape=[3], data={expected})")),
                    "{output}"
                );
            } else {
                assert_claim(&output, selected_op, 2);
            }
        }
    }
}

fn projected_tuple_source(select_second: bool, selected_agrees: bool) -> String {
    let selected = if selected_agrees {
        THREE_BY_FOUR
    } else {
        TWO_BY_FOUR
    };
    let sibling = if selected_agrees {
        TWO_BY_FOUR
    } else {
        THREE_BY_FOUR
    };
    let (x, y) = if select_second {
        (sibling, selected)
    } else {
        (selected, sibling)
    };
    format!(
        "def decide(flag: bool) -> bool ! {{ IO }} = {{\n\
         _ = print(\"tuple-selector\")\n\
         flag\n\
        }}\n\
        def choose[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32], second: bool) -> tensor[3, f32] ! {{ IO }} = {{\n\
         _ = print(\"tuple-before\")\n\
         pair = (diagonal(x, 0i32, 1i32), cumsum(diagonal(y, 0i32, 1i32), 0i32))\n\
         _ = print(\"tuple-after\")\n\
         selected = if decide(second) then pair.1 else pair.0\n\
         _ = print(\"projection-after\")\n\
         selected\n\
        }}\n\
        out = choose(to_tensor({x}), to_tensor({y}), {select_second})\n"
    )
}

fn assert_projected_tuple_provenance(native: bool) {
    for select_second in [false, true] {
        for selected_agrees in [true, false] {
            let source = projected_tuple_source(select_second, selected_agrees);
            let (ok, output) = run(&source, native);
            assert_eq!(ok, selected_agrees, "{source}\n{output}");
            assert_eq!(output.matches("tuple-before").count(), 1, "{output}");
            assert_eq!(output.matches("tuple-after").count(), 1, "{output}");
            assert_eq!(output.matches("tuple-selector").count(), 1, "{output}");
            assert_eq!(
                output.matches("projection-after").count(),
                usize::from(selected_agrees),
                "{output}"
            );
            let selected_op = if select_second { "cumsum" } else { "diagonal" };
            if selected_agrees {
                let expected = if select_second {
                    "[1.0, 7.0, 18.0]"
                } else {
                    "[1.0, 6.0, 11.0]"
                };
                assert!(
                    output.contains(&format!("out = tensor(shape=[3], data={expected})")),
                    "{output}"
                );
                assert!(!output.contains("numeric trap:"), "{output}");
            } else {
                assert_claim(&output, selected_op, 2);
            }
        }
    }
}

fn aggregate_interface_source(cached: bool, select_second: bool) -> String {
    if cached {
        return "pair: (tensor[*, f32], tensor[*, f32]) = (to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32]))\n\
                def select_cached() -> tensor[3, f32] ! { IO } = {\n\
                  _ = print(\"interface-before\")\n\
                  selected = pair.1\n\
                  _ = print(\"interface-after-selection\")\n\
                  selected\n\
                }\n\
                out = select_cached()\n"
            .to_string();
    }
    format!(
        "def select_pair(pair: (tensor[*, f32], tensor[*, f32]), second: bool) -> tensor[3, f32] ! {{ IO }} = {{\n\
         _ = print(\"interface-before\")\n\
         selected = if second then pair.1 else pair.0\n\
         _ = print(\"interface-after-selection\")\n\
         selected\n\
        }}\n\
        out = select_pair((to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32])), {select_second})\n"
    )
}

fn assert_aggregate_interface_ingress(native: bool) {
    for cached in [false, true] {
        let selections: &[bool] = if cached { &[true] } else { &[false, true] };
        for &select_second in selections {
            let source = aggregate_interface_source(cached, select_second);
            let (ok, output) = run(&source, native);
            assert_eq!(ok, select_second, "{source}\n{output}");
            assert_eq!(output.matches("interface-before").count(), 1, "{output}");
            assert_eq!(
                output.matches("interface-after-selection").count(),
                usize::from(select_second),
                "{output}"
            );
            if select_second {
                assert!(
                    output.contains("out = tensor(shape=[3], data=[1.0, 2.0, 3.0])"),
                    "{output}"
                );
                assert!(!output.contains("numeric trap:"), "{output}");
            } else {
                assert_claim(&output, "load", 2);
            }
        }
    }
    if !native {
        let static_mismatch = "pair = (to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32]))\n\
             def invalid_cached_projection() -> tensor[3, f32] = pair.0\n\
             out = invalid_cached_projection()\n";
        let (ok, output) = run(static_mismatch, false);
        assert!(!ok, "{static_mismatch}\n{output}");
        assert!(output.contains("DimensionMismatch"), "{output}");
        assert!(!output.contains("pending result claim"), "{output}");
    }
}

#[test]
fn eval_aggregate_interface_ingress_stamps_each_tensor_field_as_load() {
    assert_aggregate_interface_ingress(false);
}

#[test]
fn c_aggregate_interface_ingress_stamps_each_tensor_field_as_load() {
    assert_aggregate_interface_ingress(true);
}

fn aggregate_projection_source(kind: &str, select_second: bool, selected_agrees: bool) -> String {
    let selected = if selected_agrees {
        THREE_BY_FOUR
    } else {
        TWO_BY_FOUR
    };
    let sibling = if selected_agrees {
        TWO_BY_FOUR
    } else {
        THREE_BY_FOUR
    };
    let (x, y) = if select_second {
        (sibling, selected)
    } else {
        (selected, sibling)
    };
    let (types, constructed, projected) = match kind {
        "list" => (
            "",
            "values: List[tensor[*, f32]] = [first, second_value]",
            "if second then index(alias, 1i64) else index(alias, 0i64)",
        ),
        "record" => (
            "type Pair = | Pair { left: tensor[*, f32], right: tensor[*, f32] }\n",
            "values = Pair { right: second_value, left: first }",
            "if second then alias.right else alias.left",
        ),
        "pattern" => (
            "type Pair = | Pair { left: tensor[*, f32], right: tensor[*, f32] }\n",
            "values = Pair { right: second_value, left: first }",
            "match alias with { | Pair { left: first, right: second_value } => if second then second_value else first }",
        ),
        "nested-record" => (
            "type Pair = | Pair { left: tensor[*, f32], right: tensor[*, f32] }\n\
             type Outer = | Outer { tag: i64, pair: Pair }\n",
            "values = Outer { pair: Pair { right: second_value, left: first }, tag: 7i64 }",
            "if second then alias.pair.right else alias.pair.left",
        ),
        _ => panic!("unknown aggregate projection kind"),
    };
    format!(
        "{types}def choose[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32], second: bool) -> tensor[3, f32] ! {{ IO }} = {{\n\
         _ = print(\"aggregate-before\")\n\
         first = diagonal(x, 0i32, 1i32)\n\
         second_value = cumsum(diagonal(y, 0i32, 1i32), 0i32)\n\
         {constructed}\n\
         alias = values\n\
         selected = {projected}\n\
         _ = print(\"aggregate-after\")\n\
         selected\n\
        }}\n\
         out = choose(to_tensor({x}), to_tensor({y}), {select_second})\n"
    )
}

fn assert_aggregate_projection_provenance(native: bool) {
    for kind in ["list", "record", "pattern", "nested-record"] {
        for select_second in [false, true] {
            for selected_agrees in [true, false] {
                let source = aggregate_projection_source(kind, select_second, selected_agrees);
                let (ok, output) = run(&source, native);
                assert_eq!(ok, selected_agrees, "{kind}: {source}\n{output}");
                assert_eq!(output.matches("aggregate-before").count(), 1, "{output}");
                assert_eq!(
                    output.matches("aggregate-after").count(),
                    usize::from(selected_agrees),
                    "{output}"
                );
                if selected_agrees {
                    let expected = if select_second {
                        "[1.0, 7.0, 18.0]"
                    } else {
                        "[1.0, 6.0, 11.0]"
                    };
                    assert!(
                        output.contains(&format!("out = tensor(shape=[3], data={expected})")),
                        "{output}"
                    );
                    assert!(!output.contains("numeric trap:"), "{output}");
                } else {
                    assert_claim(
                        &output,
                        if select_second { "cumsum" } else { "diagonal" },
                        2,
                    );
                }
            }
        }
    }
}

fn nested_list_pattern_source(x: &str, y: &str) -> String {
    format!(
        "def select_second[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32]) -> tensor[3, f32] ! {{ IO }} = {{\n\
         _ = print(\"before-pattern\")\n\
         values: List[tensor[*, f32]] = [diagonal(x, 0i32, 1i32), cumsum(diagonal(y, 0i32, 1i32), 0i32)]\n\
         selected = match values with {{\n\
         | Cons(_, Cons(value, Nil)) => value\n\
         | _ => to_tensor([9.0f32, 8.0f32, 7.0f32])\n\
         }}\n\
         _ = print(\"after-pattern\")\n\
         selected\n\
         }}\n\
         out = select_second(to_tensor({x}), to_tensor({y}))\n"
    )
}

fn borrowed_list_tail_source(x: &str, y: &str) -> String {
    format!(
        "x = to_tensor({x})\n\
         y = to_tensor({y})\n\
         values: List[tensor[*, f32]] = [diagonal(x, 0i32, 1i32), cumsum(diagonal(y, 0i32, 1i32), 0i32)]\n\
         def select_second() -> tensor[3, f32] ! {{ IO }} = {{\n\
         _ = print(\"before-pattern\")\n\
         selected = match values with {{\n\
         | Cons(_, Cons(value, Nil)) => value\n\
         | _ => to_tensor([9.0f32, 8.0f32, 7.0f32])\n\
         }}\n\
         _ = print(\"after-pattern\")\n\
         selected\n\
         }}\n\
         out = select_second()\n"
    )
}

fn assert_nested_list_pattern_provenance(native: bool) {
    for (x, y, selected_agrees) in [
        (THREE_BY_FOUR, THREE_BY_FOUR, true),
        // A mismatching unselected neighbour must not claim the result.
        (TWO_BY_FOUR, THREE_BY_FOUR, true),
        (THREE_BY_FOUR, TWO_BY_FOUR, false),
    ] {
        for (path, source, producer) in [
            ("nested Cons", nested_list_pattern_source(x, y), "cumsum"),
            // A captured List is a genuine interface load and cannot be
            // consumed. Its nested pattern exercises cloning skip.
            ("captured Cons", borrowed_list_tail_source(x, y), "load"),
        ] {
            let (ok, output) = run(&source, native);
            assert_eq!(ok, selected_agrees, "{path}: {source}\n{output}");
            assert_eq!(output.matches("before-pattern").count(), 1, "{output}");
            assert_eq!(
                output.matches("after-pattern").count(),
                usize::from(selected_agrees),
                "{output}"
            );
            if selected_agrees {
                assert!(
                    output.contains("out = tensor(shape=[3], data=[1.0, 7.0, 18.0])"),
                    "{output}"
                );
                assert!(!output.contains("numeric trap:"), "{output}");
            } else {
                assert_claim(&output, producer, 2);
                assert!(!output.contains("pending result claim"), "{output}");
            }
        }
    }
}

fn assert_native_list_pattern_skip_routes() {
    // `Cons(_, Cons(value, Nil))` has three failure exits, so C lowering
    // decides the arm with a `bool` test and the selected body destructures
    // again ([04-PAT-2], chelis#2445). The test skips to both tails, to reach
    // the `Nil` it tests; the selected pass skips only to the tail that holds
    // `value`, and never to the `Nil` it no longer tests. In the owned fixture
    // the test's first skip borrows, because the body still reads the list; in
    // the borrowed fixture both first skips borrow the top-level list.
    for (name, source, expected_borrowed, expected_owned) in [
        (
            "owned_list_pattern",
            nested_list_pattern_source(TWO_BY_FOUR, THREE_BY_FOUR),
            1,
            2,
        ),
        (
            "borrowed_list_pattern",
            borrowed_list_tail_source(TWO_BY_FOUR, THREE_BY_FOUR),
            2,
            1,
        ),
    ] {
        let dir = tempfile::tempdir().expect("tempdir");
        let source_path = dir.path().join(format!("{name}.ch"));
        let output_dir = dir.path().join("c");
        fs::write(&source_path, &source).expect("fixture");
        let built = Command::cargo_bin("chelis")
            .expect("chelis")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["build", "--allow-style-violations"])
            .arg(&source_path)
            .args(["--target", "c", "-o"])
            .arg(&output_dir)
            .output()
            .expect("build");
        assert!(
            built.status.success(),
            "{source}\n{}{}",
            String::from_utf8_lossy(&built.stdout),
            String::from_utf8_lossy(&built.stderr)
        );
        let generated_name = format!("{name}.c");
        let generated = fs::read_to_string(output_dir.join(&generated_name)).expect("generated C");
        let borrowed = generated.matches(" = chelis_list_drop(").count();
        let owned = generated.matches(" = chelis_list_drop_owned(").count();
        assert_eq!(
            (borrowed, owned),
            (expected_borrowed, expected_owned),
            "{generated}"
        );
        let suffix_sites = generated
            .match_indices(" = __chelis_host_result_origin_list_suffix(")
            .map(|(offset, _)| offset)
            .collect::<Vec<_>>();
        let mut drop_sites = generated
            .match_indices(" = chelis_list_drop(")
            .chain(generated.match_indices(" = chelis_list_drop_owned("))
            .map(|(offset, _)| offset)
            .collect::<Vec<_>>();
        drop_sites.sort_unstable();
        assert_eq!(suffix_sites.len(), drop_sites.len(), "{generated}");
        assert!(
            suffix_sites
                .iter()
                .zip(&drop_sites)
                .all(|(suffix, drop)| suffix < drop),
            "origin suffixes must be built before either runtime drop:\n{generated}"
        );
        assert!(
            common::link_generated(&output_dir, &generated_name, name).success(),
            "link {name}"
        );
        let executed = StdCommand::new(output_dir.join(name))
            .output()
            .expect("execute native fixture");
        let output = format!(
            "{}{}",
            String::from_utf8_lossy(&executed.stdout),
            String::from_utf8_lossy(&executed.stderr)
        );
        assert!(executed.status.success(), "{source}\n{output}");
        assert!(
            output.contains("out = tensor(shape=[3], data=[1.0, 7.0, 18.0])"),
            "{output}"
        );
    }
}

fn direct_list_skip_source(x: &str, y: &str) -> String {
    format!(
        "def select_second[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32]) -> tensor[3, f32] ! {{ IO }} = {{\n\
         _ = print(\"before-skip\")\n\
         values: List[tensor[*, f32]] = [diagonal(x, 0i32, 1i32), cumsum(diagonal(y, 0i32, 1i32), 0i32)]\n\
         tail = skip(values, 1i64)\n\
         selected = index(tail, 0i64)\n\
         _ = print(\"after-skip\")\n\
         selected\n\
         }}\n\
         out = select_second(to_tensor({x}), to_tensor({y}))\n"
    )
}

fn assert_direct_list_skip_provenance(native: bool) {
    for (x, y, selected_agrees) in [
        (THREE_BY_FOUR, THREE_BY_FOUR, true),
        // The skipped producer must not claim the selected tail element.
        (TWO_BY_FOUR, THREE_BY_FOUR, true),
        (THREE_BY_FOUR, TWO_BY_FOUR, false),
    ] {
        let source = direct_list_skip_source(x, y);
        let (ok, output) = run(&source, native);
        assert_eq!(ok, selected_agrees, "{source}\n{output}");
        assert_eq!(output.matches("before-skip").count(), 1, "{output}");
        assert_eq!(
            output.matches("after-skip").count(),
            usize::from(selected_agrees),
            "{output}"
        );
        if selected_agrees {
            assert!(
                output.contains("out = tensor(shape=[3], data=[1.0, 7.0, 18.0])"),
                "{output}"
            );
            assert!(!output.contains("numeric trap:"), "{output}");
        } else {
            assert_claim(&output, "cumsum", 2);
            assert!(!output.contains("pending result claim"), "{output}");
        }
    }
}

fn direct_tail_list_source(select_second: bool, selected_agrees: bool) -> String {
    let selected = if selected_agrees {
        THREE_BY_FOUR
    } else {
        TWO_BY_FOUR
    };
    let sibling = if selected_agrees {
        TWO_BY_FOUR
    } else {
        THREE_BY_FOUR
    };
    let (x, y) = if select_second {
        (sibling, selected)
    } else {
        (selected, sibling)
    };
    format!(
        "def choose[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32], second: bool) -> tensor[3, f32] = {{\n\
         values: List[tensor[*, f32]] = [diagonal(x, 0i32, 1i32), cumsum(diagonal(y, 0i32, 1i32), 0i32)]\n\
         index(values, if second then 1i64 else 0i64)\n\
        }}\n\
         out = choose(to_tensor({x}), to_tensor({y}), {select_second})\n"
    )
}

fn assert_direct_tail_list_projection(native: bool) {
    for select_second in [false, true] {
        for selected_agrees in [true, false] {
            let source = direct_tail_list_source(select_second, selected_agrees);
            let (ok, output) = run(&source, native);
            assert_eq!(ok, selected_agrees, "{source}\n{output}");
            if selected_agrees {
                let expected = if select_second {
                    "[1.0, 7.0, 18.0]"
                } else {
                    "[1.0, 6.0, 11.0]"
                };
                assert!(
                    output.contains(&format!("out = tensor(shape=[3], data={expected})")),
                    "{output}"
                );
            } else {
                assert_claim(
                    &output,
                    if select_second { "cumsum" } else { "diagonal" },
                    2,
                );
            }
        }
    }
}

fn assert_projection_without_origin_fails_explicitly(native: bool) {
    let source = "def choose[n](x: tensor[n, 4, f32]) -> tensor[3, f32] = {\n\
                  empty: List[tensor[*, f32]] = []\n\
                  grown = append(empty, diagonal(x, 0i32, 1i32))\n\
                  index(grown, 0i64)\n\
                  }\n\
                  out = choose(to_tensor([[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]]))\n";
    let (ok, output) = run(source, native);
    assert!(!ok, "{source}\n{output}");
    assert!(
        output.contains(
            "host runtime: pending result claim reached a tensor without producer provenance"
        ),
        "{output}"
    );
    assert!(
        !output.contains("numeric trap: domain in index"),
        "{output}"
    );
}

fn assert_option_projection_provenance(native: bool) {
    for (values, agrees) in [(THREE_BY_FOUR, true), (TWO_BY_FOUR, false)] {
        let local = format!(
            "def local[n](x: tensor[n, 4, f32]) -> tensor[3, f32] = {{\n\
             wrapped: Option[tensor[*, f32]] = Some(diagonal(x, 0i32, 1i32))\n\
             match wrapped with {{\n\
             | Some(selected) => selected\n\
             | None => to_tensor([9.0f32, 8.0f32, 7.0f32])\n\
             }}\n\
            }}\n\
             out = local(to_tensor({values}))\n"
        );
        let (ok, output) = run(&local, native);
        assert_eq!(ok, agrees, "{local}\n{output}");
        if agrees {
            assert!(
                output.contains("out = tensor(shape=[3], data=[1.0, 6.0, 11.0])"),
                "{output}"
            );
        } else {
            assert_claim(&output, "diagonal", 2);
        }

        let formal = format!(
            "def select(value: Option[tensor[*, f32]]) -> tensor[3, f32] = match value with {{\n\
             | Some(selected) => selected\n\
             | None => to_tensor([9.0f32, 8.0f32, 7.0f32])\n\
            }}\n\
             out = select(Some(diagonal(to_tensor({values}), 0i32, 1i32)))\n"
        );
        let (ok, output) = run(&formal, native);
        assert_eq!(ok, agrees, "{formal}\n{output}");
        if agrees {
            assert!(
                output.contains("out = tensor(shape=[3], data=[1.0, 6.0, 11.0])"),
                "{output}"
            );
        } else {
            assert_claim(&output, "load", 2);
        }
    }

    let none = "def select(value: Option[tensor[*, f32]]) -> tensor[3, f32] = match value with {\n\
                | Some(selected) => selected\n\
                | None => to_tensor([9.0f32, 8.0f32, 7.0f32])\n\
                }\n\
                out = select(None)\n";
    let (ok, output) = run(none, native);
    assert!(ok, "{none}\n{output}");
    assert!(
        output.contains("out = tensor(shape=[3], data=[9.0, 8.0, 7.0])"),
        "{output}"
    );
}

fn assert_global_shadow_keeps_local_producer(native: bool) {
    for (values, agrees) in [(THREE_BY_FOUR, true), (TWO_BY_FOUR, false)] {
        let source = format!(
            "shadow = to_tensor([0.0f32, 0.0f32, 0.0f32])\n\
             def choose[n](x: tensor[n, 4, f32]) -> tensor[3, f32] = {{\n\
             shadow = diagonal(x, 0i32, 1i32)\n\
             shadow\n\
            }}\n\
             out = choose(to_tensor({values}))\n"
        );
        let (ok, output) = run(&source, native);
        assert_eq!(ok, agrees, "{source}\n{output}");
        if agrees {
            assert!(
                output.contains("out = tensor(shape=[3], data=[1.0, 6.0, 11.0])"),
                "{output}"
            );
        } else {
            assert_claim(&output, "diagonal", 2);
        }
    }
}

fn precision_only_callback_source(agrees: bool) -> String {
    let second = if agrees {
        "[1.0f32, 2.0f32, 3.0f32]"
    } else {
        "[1.0f32, 2.0f32]"
    };
    format!(
        "def candidate[p: Float](value: tensor[extent, p]) -> tensor[3, p] = relu(value)\n\
         def invoke[p: Float](f: tensor[*, p] -> tensor[*, p], value: tensor[*, p], label: string) -> tensor[*, p] ! {{ IO }} = {{\n\
         _ = print(label)\n\
         result = f(value)\n\
         _ = print(\"caller-after\")\n\
         result\n\
         }}\n\
         first = invoke(candidate, {{\n\
         _ = print(\"actual64-before\")\n\
         value = to_tensor([1.0f64, 16777217.0f64, 3.0f64])\n\
         _ = print(\"actual64-after\")\n\
         value\n\
         }}, \"caller64-before\")\n\
         out = invoke(candidate, {{\n\
         _ = print(\"actual32-before\")\n\
         value = to_tensor({second})\n\
         _ = print(\"actual32-after\")\n\
         value\n\
         }}, \"caller32-before\")\n"
    )
}

fn rank_only_callback_source(agrees: bool) -> String {
    let second = if agrees {
        "[1.0f64, 2.0f64, 3.0f64]"
    } else {
        "[1.0f64, 2.0f64]"
    };
    format!(
        "def candidate[rest](value: tensor[first, ..rest, f64]) -> tensor[3, ..rest, f64] = relu(value)\n\
         def invoke3(f: tensor[*, 2, 2, f64] -> tensor[*, 2, 2, f64], value: tensor[*, 2, 2, f64]) -> tensor[*, 2, 2, f64] ! {{ IO }} = {{\n\
         _ = print(\"caller3-before\")\n\
         result = f(value)\n\
         _ = print(\"caller3-after\")\n\
         result\n\
         }}\n\
         def invoke1(f: tensor[*, f64] -> tensor[*, f64], value: tensor[*, f64]) -> tensor[*, f64] ! {{ IO }} = {{\n\
         _ = print(\"caller1-before\")\n\
         result = f(value)\n\
         _ = print(\"caller1-after\")\n\
         result\n\
         }}\n\
         first = invoke3(candidate, {{\n\
         _ = print(\"actual3-before\")\n\
         value = to_tensor([[[1.0f64, 16777217.0f64], [3.0f64, 4.0f64]], [[5.0f64, 6.0f64], [7.0f64, 8.0f64]], [[9.0f64, 10.0f64], [11.0f64, 12.0f64]]])\n\
         _ = print(\"actual3-after\")\n\
         value\n\
         }})\n\
         out = invoke1(candidate, {{\n\
         _ = print(\"actual1-before\")\n\
         value = to_tensor({second})\n\
         _ = print(\"actual1-after\")\n\
         value\n\
         }})\n"
    )
}

fn assert_precision_only_callback_claims(native: bool) {
    for agrees in [true, false] {
        let source = precision_only_callback_source(agrees);
        let (ok, output) = run(&source, native);
        assert_eq!(ok, agrees, "{source}\n{output}");
        let mut markers = vec![
            "actual64-before",
            "actual64-after",
            "caller64-before",
            "caller-after",
            "actual32-before",
            "actual32-after",
            "caller32-before",
        ];
        if agrees {
            markers.push("out =");
        }
        assert_markers_in_order(&output, &markers);
        for marker in [
            "actual64-before",
            "actual64-after",
            "caller64-before",
            "actual32-before",
            "actual32-after",
            "caller32-before",
        ] {
            assert_eq!(output.matches(marker).count(), 1, "{marker}: {output}");
        }
        assert_eq!(
            output.matches("caller-after").count(),
            1 + usize::from(agrees)
        );
        if agrees {
            assert!(
                output.contains("first = tensor(shape=[3]") && output.contains("16777217.0"),
                "{output}"
            );
            assert!(output.contains("out = tensor(shape=[3]"), "{output}");
        } else {
            assert_claim(&output, "relu", 2);
            assert!(!output.contains("out ="), "{output}");
        }
    }
}

fn scalar_only_precision_source() -> &'static str {
    "def candidate[n, p: Float](witness: p, extent_source: tensor[n, i64]) -> tensor[3, p] = witness |> scalar_to_tensor |> insert(0i32, shape(extent_source, 0i32))\n\
     def invoke32(f: f32 -> tensor[*, i64] -> tensor[*, f32], witness: f32, extent_source: tensor[*, i64]) -> tensor[*, f32] = f(witness, extent_source)\n\
     def invoke64(f: f64 -> tensor[*, i64] -> tensor[*, f64], witness: f64, extent_source: tensor[*, i64]) -> tensor[*, f64] = f(witness, extent_source)\n\
     first = invoke32(candidate, 16777217.0f32, to_tensor([0i64, 0i64, 0i64]))\n\
     out = invoke64(candidate, 16777217.0f64, to_tensor([0i64, 0i64, 0i64]))\n"
}

fn assert_scalar_only_precision_evidence(native: bool) {
    let source = scalar_only_precision_source();
    let (ok, output) = run(source, native);
    assert!(ok, "{source}\n{output}");
    assert!(
        output.contains("first = tensor(shape=[3], data=[16777216.0, 16777216.0, 16777216.0])"),
        "{output}"
    );
    assert!(
        output.contains("out = tensor(shape=[3], data=[16777217.0, 16777217.0, 16777217.0])"),
        "{output}"
    );
    assert!(!output.contains("numeric trap:"), "{output}");
}

fn assert_rank_only_callback_claims(native: bool) {
    for agrees in [true, false] {
        let source = rank_only_callback_source(agrees);
        let (ok, output) = run(&source, native);
        assert_eq!(ok, agrees, "{source}\n{output}");
        let mut markers = vec![
            "actual3-before",
            "actual3-after",
            "caller3-before",
            "caller3-after",
            "actual1-before",
            "actual1-after",
            "caller1-before",
        ];
        if agrees {
            markers.push("caller1-after");
        }
        assert_markers_in_order(&output, &markers);
        for marker in [
            "actual3-before",
            "actual3-after",
            "caller3-before",
            "caller3-after",
            "actual1-before",
            "actual1-after",
            "caller1-before",
        ] {
            assert_eq!(output.matches(marker).count(), 1, "{marker}: {output}");
        }
        assert_eq!(output.matches("caller1-after").count(), usize::from(agrees));
        if agrees {
            assert!(
                output.contains("first = tensor(shape=[3, 2, 2]") && output.contains("16777217.0"),
                "{output}"
            );
            assert!(output.contains("out = tensor(shape=[3]"), "{output}");
        } else {
            assert_claim(&output, "relu", 2);
            assert!(!output.contains("out ="), "{output}");
        }
    }
}

fn polymorphic_callback_source(trailing_literal: bool, agrees: bool) -> String {
    let (
        callback,
        callback2,
        callback3,
        value2_ty,
        value3_ty,
        result2_ty,
        result3_ty,
        bounds2,
        bounds3,
        value2,
        value3,
    ) = if trailing_literal {
        (
            "def candidate[rest, p: Float](value: tensor[first, ..rest, extent, p]) -> tensor[first, ..rest, 3, p] = relu(value)",
            "tensor[4, extent, p] -> tensor[4, *, p]",
            "tensor[2, 2, extent, p] -> tensor[2, 2, *, p]",
            "tensor[4, extent, p]",
            "tensor[2, 2, extent, p]",
            "tensor[4, *, p]",
            "tensor[2, 2, *, p]",
            "[[0i64, shape(source, 0i32)], [1i64, shape(source, 1i32)]]",
            "[[0i64, shape(source, 0i32)], [0i64, shape(source, 1i32)], [1i64, shape(source, 2i32)]]",
            "[[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32], [9.0f32, 10.0f32, 11.0f32, 12.0f32], [13.0f32, 14.0f32, 15.0f32, 16.0f32]]",
            if agrees {
                "[[[1.0f64, 16777217.0f64, 3.0f64, 4.0f64], [5.0f64, 6.0f64, 7.0f64, 8.0f64]], [[9.0f64, 10.0f64, 11.0f64, 12.0f64], [13.0f64, 14.0f64, 15.0f64, 16.0f64]]]"
            } else {
                "[[[1.0f64, 16777217.0f64, 3.0f64], [4.0f64, 5.0f64, 6.0f64]], [[7.0f64, 8.0f64, 9.0f64], [10.0f64, 11.0f64, 12.0f64]]]"
            },
        )
    } else {
        (
            "def candidate[rest, p: Float](value: tensor[extent, first, ..rest, p]) -> tensor[3, first, ..rest, p] = relu(value)",
            "tensor[extent, 4, p] -> tensor[*, 4, p]",
            "tensor[extent, 2, 2, p] -> tensor[*, 2, 2, p]",
            "tensor[extent, 4, p]",
            "tensor[extent, 2, 2, p]",
            "tensor[*, 4, p]",
            "tensor[*, 2, 2, p]",
            "[[1i64, shape(source, 0i32)], [0i64, shape(source, 1i32)]]",
            "[[1i64, shape(source, 0i32)], [0i64, shape(source, 1i32)], [0i64, shape(source, 2i32)]]",
            "[[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32], [9.0f32, 10.0f32, 11.0f32, 12.0f32], [13.0f32, 14.0f32, 15.0f32, 16.0f32]]",
            if agrees {
                "[[[1.0f64, 2.0f64], [3.0f64, 4.0f64]], [[16777217.0f64, 6.0f64], [7.0f64, 8.0f64]], [[9.0f64, 10.0f64], [11.0f64, 12.0f64]], [[13.0f64, 14.0f64], [15.0f64, 16.0f64]]]"
            } else {
                "[[[1.0f64, 2.0f64], [3.0f64, 4.0f64]], [[16777217.0f64, 6.0f64], [7.0f64, 8.0f64]], [[9.0f64, 10.0f64], [11.0f64, 12.0f64]]]"
            },
        )
    };
    format!(
        "{callback}\n\
         def invoke2[p: Float](f: {callback2}, value: {value2_ty}) -> {result2_ty} ! {{ IO }} = {{\n\
         _ = print(\"caller2-before\")\n\
         result = f(value)\n\
         _ = print(\"caller2-after\")\n\
         result\n\
         }}\n\
         def invoke3[p: Float](f: {callback3}, value: {value3_ty}) -> {result3_ty} ! {{ IO }} = {{\n\
         _ = print(\"caller3-before\")\n\
         result = f(value)\n\
         _ = print(\"caller3-after\")\n\
         result\n\
         }}\n\
         first = invoke2(candidate, {{\n\
         _ = print(\"actual2-before\")\n\
         source = to_tensor({value2})\n\
         prepared = shrink(source, {bounds2})\n\
         _ = print(\"actual2-after\")\n\
         prepared\n\
         }})\n\
         out = invoke3(candidate, {{\n\
         _ = print(\"actual3-before\")\n\
         source = to_tensor({value3})\n\
         prepared = shrink(source, {bounds3})\n\
         _ = print(\"actual3-after\")\n\
         prepared\n\
         }})\n"
    )
}

fn assert_polymorphic_callback_claims(native: bool) {
    for trailing_literal in [false, true] {
        for agrees in [true, false] {
            let source = polymorphic_callback_source(trailing_literal, agrees);
            let (ok, output) = run(&source, native);
            assert_eq!(ok, agrees, "{source}\n{output}");
            for marker in [
                "actual2-before",
                "actual2-after",
                "caller2-before",
                "caller2-after",
                "actual3-before",
                "actual3-after",
                "caller3-before",
            ] {
                assert_eq!(output.matches(marker).count(), 1, "{output}");
            }
            assert_eq!(output.matches("caller3-after").count(), usize::from(agrees));
            let positions = [
                "actual2-before",
                "actual2-after",
                "caller2-before",
                "caller2-after",
                "actual3-before",
                "actual3-after",
                "caller3-before",
            ]
            .map(|marker| {
                output
                    .find(marker)
                    .unwrap_or_else(|| panic!("{marker}: {output}"))
            });
            assert!(
                positions.windows(2).all(|pair| pair[0] < pair[1]),
                "{output}"
            );
            if agrees {
                assert!(
                    output.contains("first = tensor(shape=[3, 4]")
                        || output.contains("first = tensor(shape=[4, 3]"),
                    "{output}"
                );
                let expected_shape = if trailing_literal {
                    "out = tensor(shape=[2, 2, 3]"
                } else {
                    "out = tensor(shape=[3, 2, 2]"
                };
                assert!(output.contains(expected_shape), "{output}");
                assert!(output.contains("16777217.0"), "{output}");
                assert!(!output.contains("numeric trap:"), "{output}");
            } else {
                let expected_axis = if trailing_literal { 2 } else { 0 };
                assert!(
                    output.contains(&format!(
                        "extent `3`: claimed = 3, relu axis {expected_axis} = 2"
                    )),
                    "{output}"
                );
                assert!(
                    output
                        .lines()
                        .any(|line| line == "numeric trap: domain in relu at i64"),
                    "{output}"
                );
                assert!(!output.contains("out ="), "{output}");
            }
        }
    }
}

#[test]
fn c_aggregate_origin_arena_is_fresh_for_repeated_public_calls() {
    if !common::gcc_available() {
        return;
    }
    let dir = tempfile::tempdir().expect("tempdir");
    let source_path = dir.path().join("arena_lifetime.ch");
    let output_dir = dir.path().join("c");
    fs::write(
        &source_path,
        "def make[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32]) -> (tensor[*, f32], tensor[*, f32]) =\n\
         (diagonal(x, 0i32, 1i32), cumsum(diagonal(y, 0i32, 1i32), 0i32))\n\
         def choose[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32], second: bool) -> tensor[3, f32] = {\n\
         pair = make(x, y)\n\
         if second then pair.1 else pair.0\n\
         }\n\
         def choose_tail[n, m](x: tensor[n, 4, f32], y: tensor[m, 4, f32]) -> tensor[3, f32] = {\n\
         values: List[tensor[*, f32]] = [diagonal(x, 0i32, 1i32), diagonal(x, 0i32, 1i32), cumsum(diagonal(y, 0i32, 1i32), 0i32)]\n\
         once = skip(values, 1i64)\n\
         twice = skip(once, 1i64)\n\
         index(twice, 0i64)\n\
         }\n",
    )
    .expect("fixture");
    let built = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--allow-style-violations"])
        .arg(&source_path)
        .args(["--target", "c", "--output"])
        .arg(&output_dir)
        .output()
        .expect("build");
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );

    let make = common::authored_c_symbol("make");
    let choose = common::authored_c_symbol("choose");
    let choose_tail = common::authored_c_symbol("choose_tail");
    let harness = format!(
        "#define main generated_main\n\
         #include \"arena_lifetime.c\"\n\
         #undef main\n\
         int main(void) {{\n\
         chelis_tensor *three = chelis_alloc(2, (int64_t[]){{3, 4}}, CHELIS_DTYPE_F32);\n\
         chelis_tensor *two = chelis_alloc(2, (int64_t[]){{2, 4}}, CHELIS_DTYPE_F32);\n\
         chelis_tuple *aggregate = {make}(three, two);\n\
         chelis_tuple_release(aggregate);\n\
         for (int i = 0; i < 64; ++i) {{\n\
             chelis_tensor *left = {choose}(three, two, false);\n\
             if (chelis_tensor_rank(left) != 1 || chelis_tensor_shape(left, 0) != 3) return 2;\n\
             chelis_tensor_release(left);\n\
             chelis_tensor *right = {choose}(two, three, true);\n\
             if (chelis_tensor_rank(right) != 1 || chelis_tensor_shape(right, 0) != 3) return 3;\n\
             chelis_tensor_release(right);\n\
             chelis_tensor *tail = {choose_tail}(two, three);\n\
             if (chelis_tensor_rank(tail) != 1 || chelis_tensor_shape(tail, 0) != 3) return 4;\n\
             chelis_tensor_release(tail);\n\
         }}\n\
         chelis_tensor_release(three);\n\
         chelis_tensor_release(two);\n\
         return 0;\n\
         }}\n"
    );
    fs::write(output_dir.join("harness.c"), harness).expect("harness");

    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: false,
        },
    );
    let linked = StdCommand::new(&toolchain.compiler)
        .current_dir(&output_dir)
        .args(["-O1", "-g", "-fsanitize=address", "-fno-omit-frame-pointer"])
        .args(&toolchain.compile_flags)
        .arg("harness.c")
        .args(["-L.", "-lchelis_runtime"])
        .args(&toolchain.link_flags)
        .args(["-fsanitize=address", "-o", "arena_lifetime"])
        .output()
        .expect("link ASan harness");
    assert!(
        linked.status.success(),
        "ASan harness link failed:\n{}{}",
        String::from_utf8_lossy(&linked.stdout),
        String::from_utf8_lossy(&linked.stderr)
    );
    let executed = StdCommand::new(output_dir.join("arena_lifetime"))
        .env("ASAN_OPTIONS", "detect_leaks=0:halt_on_error=1")
        .output()
        .expect("run ASan harness");
    assert!(
        executed.status.success(),
        "repeated public invocation failed:\n{}{}",
        String::from_utf8_lossy(&executed.stdout),
        String::from_utf8_lossy(&executed.stderr)
    );
}

#[test]
fn eval_list_and_adt_projection_retains_selected_producer() {
    assert_aggregate_projection_provenance(false);
}

#[test]
fn c_list_and_adt_projection_retains_selected_producer() {
    assert_aggregate_projection_provenance(true);
}

#[test]
fn eval_nested_list_pattern_retains_selected_tail_producer() {
    assert_nested_list_pattern_provenance(false);
}

#[test]
fn c_nested_list_pattern_retains_selected_tail_producer() {
    assert_nested_list_pattern_provenance(true);
    assert_native_list_pattern_skip_routes();
}

#[test]
fn eval_direct_list_skip_retains_selected_tail_producer() {
    assert_direct_list_skip_provenance(false);
}

#[test]
fn c_direct_list_skip_retains_selected_tail_producer() {
    assert_direct_list_skip_provenance(true);
}

#[test]
fn eval_direct_tail_list_projection_retains_selected_producer() {
    assert_direct_tail_list_projection(false);
}

#[test]
fn c_direct_tail_list_projection_retains_selected_producer() {
    assert_direct_tail_list_projection(true);
}

#[test]
fn eval_projection_without_origin_fails_instead_of_guessing_index() {
    assert_projection_without_origin_fails_explicitly(false);
}

#[test]
fn c_projection_without_origin_fails_instead_of_guessing_index() {
    assert_projection_without_origin_fails_explicitly(true);
}

#[test]
fn eval_option_projection_distinguishes_local_and_formal_origins() {
    assert_option_projection_provenance(false);
}

#[test]
fn c_option_projection_distinguishes_local_and_formal_origins() {
    assert_option_projection_provenance(true);
}

#[test]
fn eval_global_shadow_keeps_local_producer() {
    assert_global_shadow_keeps_local_producer(false);
}

#[test]
fn c_global_shadow_keeps_local_producer() {
    assert_global_shadow_keeps_local_producer(true);
}

#[test]
fn eval_precision_only_callback_claims_are_callsite_local() {
    assert_precision_only_callback_claims(false);
}

#[test]
fn c_precision_only_callback_claims_are_callsite_local() {
    assert_precision_only_callback_claims(true);
}

#[test]
fn eval_scalar_only_precision_evidence_specializes_each_callsite() {
    assert_scalar_only_precision_evidence(false);
}

#[test]
fn c_scalar_only_precision_evidence_specializes_each_callsite() {
    assert_scalar_only_precision_evidence(true);
}

#[test]
fn inadmissible_scalar_only_precision_evidence_is_rejected_by_checker() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("inadmissible_scalar_precision.ch");
    fs::write(
        &path,
        "def candidate[n, p: Float](witness: p, extent_source: tensor[n, i64]) -> tensor[3, p] = witness |> scalar_to_tensor |> insert(0i32, shape(extent_source, 0i32))\n\
         out = candidate(1i32, to_tensor([0i64, 0i64, 0i64]))\n",
    )
    .expect("fixture");
    let checked = Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .output()
        .expect("check");
    assert!(!checked.status.success(), "{checked:?}");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).expect("check report");
    assert!(report["score"].as_f64().is_some_and(|score| score < 1.0));
    assert!(
        report["errors"]
            .as_array()
            .is_some_and(|errors| errors.iter().any(|error| {
                error["kind"] == "PrecisionMismatch"
                    && error["message"]
                        .as_str()
                        .is_some_and(|message| message.contains("Float"))
            })),
        "{report}"
    );
}

#[test]
fn eval_rank_only_callback_claims_expand_empty_and_nonempty_spreads() {
    assert_rank_only_callback_claims(false);
}

#[test]
fn c_rank_only_callback_claims_expand_empty_and_nonempty_spreads() {
    assert_rank_only_callback_claims(true);
}

#[test]
fn eval_combined_precision_rank_callback_claims_actualize_each_callsite() {
    assert_polymorphic_callback_claims(false);
}

#[test]
fn c_combined_precision_rank_callback_claims_actualize_each_callsite() {
    assert_polymorphic_callback_claims(true);
}

#[test]
fn eval_tuple_projection_retains_only_the_selected_producer() {
    assert_projected_tuple_provenance(false);
}

#[test]
fn c_tuple_projection_retains_only_the_selected_producer() {
    assert_projected_tuple_provenance(true);
}

#[test]
fn eval_selection_before_production_forwards_only_to_the_selected_arm() {
    assert_selection_before_production(false);
}

#[test]
fn c_selection_before_production_forwards_only_to_the_selected_arm() {
    assert_selection_before_production(true);
}
