//! Spec/04 section 4.7 and [04-NUM-9]: an inherited literal result
//! obligation reaches the selected producer through supported callable calls.
//! The obligation is invocation-local and guards at the producer.

mod common;
#[path = "common/result_claims.rs"]
mod result_claims;

use result_claims::{assert_claim, run};

const TWO: &str = "[1.0f32, 2.0f32, 3.0f32]";
const THREE: &str = "[1.0f32, 2.0f32, 3.0f32, 4.0f32]";

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

#[test]
fn eval_identity_callback_checks_claim_after_actual_preparation() {
    assert_identity_callback_checks_after_actual_preparation(false);
}

#[test]
fn c_identity_callback_checks_claim_after_actual_preparation() {
    assert_identity_callback_checks_after_actual_preparation(true);
}
