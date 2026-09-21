//! Spec/04 section 4.7 and [04-NUM-9]: an inherited literal result
//! obligation reaches the selected producer through supported callable calls
//! and delayed selection.  The obligation is invocation-local and guards only
//! the selected value, at the first source position where that choice is known.

mod common;
#[path = "common/result_claims.rs"]
mod result_claims;

use result_claims::{assert_claim, run};

const TWO: &str = "[1.0f32, 2.0f32, 3.0f32]";
const THREE: &str = "[1.0f32, 2.0f32, 3.0f32, 4.0f32]";
const TWO_BY_FOUR: &str = "[[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32]]";
const THREE_BY_FOUR: &str = "[[1.0f32, 2.0f32, 3.0f32, 4.0f32], [5.0f32, 6.0f32, 7.0f32, 8.0f32], [9.0f32, 10.0f32, 11.0f32, 12.0f32]]";

fn callable_source(values: &str, literal: bool) -> String {
    let (binding, callable) = if literal {
        (
            "local = fn (value: tensor[*, f32]) -> {\n _ = print(\"producer-before\")\n produced = shrink(value, [[1i64, shape(value, 0i32)]])\n _ = print(\"producer-after\")\n produced\n}\n",
            "local",
        )
    } else {
        ("", "cut")
    };
    format!(
        "def cut[n](value: tensor[n, f32]) -> tensor[*, f32] ! {{ IO }} = {{\n _ = print(\"producer-before\")\n produced = shrink(value, [[1i64, shape(value, 0i32)]])\n _ = print(\"producer-after\")\n produced\n}}\n\
         def invoke(f: tensor[*, f32] -> tensor[*, f32], value: tensor[*, f32]) -> tensor[*, f32] ! {{ IO }} = f(value)\n\
         def claimed[n](f: tensor[*, f32] -> tensor[*, f32], value: tensor[n, f32]) -> tensor[3, f32] ! {{ IO }} = {{\n _ = print(\"caller-before\")\n result = invoke(f, value)\n _ = print(\"caller-after\")\n result\n}}\n\
         out = {{\n {binding} claimed({callable}, to_tensor({values}))\n}}\n"
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
            "match decide(second) with {\n | true => { _ = drop(first)\n second_value }\n | false => { _ = drop(second_value)\n first }\n}"
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

#[test]
fn eval_selection_before_production_forwards_only_to_the_selected_arm() {
    assert_selection_before_production(false);
}

#[test]
fn c_selection_before_production_forwards_only_to_the_selected_arm() {
    assert_selection_before_production(true);
}
