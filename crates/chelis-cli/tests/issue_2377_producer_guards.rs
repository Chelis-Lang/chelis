//! Spec/04 §4.7: a result extent belongs to its producing operation even
//! when its size is available from interface inputs.
mod common;
#[path = "common/result_claims.rs"]
mod result_claims;

use result_claims::{assert_claim, run};

fn candidate(prefix: &str, values: &str, effects: bool, monomorphic: bool) -> String {
    let (effect, before, after) = if effects {
        (
            " ! { IO }",
            "_ = print(\"before-producer\")",
            "_ = print(\"after-producer\")",
        )
    } else {
        ("", "", "")
    };
    let source = format!(
        "def candidate[n, p: Float](witness: p, extent_source: tensor[n, i64]) -> tensor[3, p]{effect} = {{\n {prefix}\n {before}\n result = witness |> scalar_to_tensor |> insert(0i32, shape(extent_source, 0i32))\n {after}\n result\n}}\nout = candidate(16777217.0f64, to_tensor({values}))\n"
    );
    if monomorphic {
        source
            .replace("[n, p: Float](witness: p,", "[n](")
            .replace("tensor[3, p]", "tensor[3, f64]")
            .replace(
                "witness |> scalar_to_tensor",
                "16777217.0f64 |> scalar_to_tensor",
            )
            .replace("candidate(16777217.0f64,", "candidate(")
    } else {
        source
    }
}

fn assert_order(native: bool) {
    for monomorphic in [false, true] {
        for values in ["[0i64, 0i64]", "[0i64, 0i64, 0i64]"] {
            for bound_tail in [false, true] {
                let mut source = candidate(
                    "_ = [-9223372036854775808i64] |> to_tensor |> neg",
                    values,
                    false,
                    monomorphic,
                );
                if !bound_tail {
                    // The issue's original return expression, without a binding.
                    source = source
                        .replace("result = ", "")
                        .replace("\n result\n}", "\n}");
                }
                let (ok, output) = run(&source, native);
                assert!(!ok, "{source}\n{output}");
                assert!(
                    output.contains("numeric trap: overflow in neg at i64"),
                    "{source}\n{output}"
                );
                assert!(!output.contains("domain in"), "{output}");
                assert_eq!(output.matches("numeric trap:").count(), 1, "{output}");
            }
        }
    }
}

fn assert_producer(native: bool) {
    for monomorphic in [false, true] {
        for effects in [false, true] {
            for (values, agrees) in [("[0i64, 0i64]", false), ("[0i64, 0i64, 0i64]", true)] {
                let (ok, output) = run(&candidate("", values, effects, monomorphic), native);
                assert_eq!(ok, agrees, "{output}");
                if effects {
                    assert_eq!(output.matches("before-producer").count(), 1, "{output}");
                    assert_eq!(
                        output.matches("after-producer").count(),
                        usize::from(agrees),
                        "{output}"
                    );
                }
                if agrees {
                    if effects {
                        assert!(
                            output.find("before-producer") < output.find("after-producer"),
                            "{output}"
                        );
                    }
                    assert!(
                        output.contains(
                            "out = tensor(shape=[3], data=[16777217.0, 16777217.0, 16777217.0])"
                        ),
                        "{output}"
                    );
                } else {
                    assert_claim(&output, "insert", 2);
                    assert_eq!(output.matches("numeric trap:").count(), 1, "{output}");
                }
            }
        }
    }
}

#[test]
fn eval_earlier_trap_precedes_insert_result_guard() {
    assert_order(false);
}
#[test]
fn c_earlier_trap_precedes_insert_result_guard() {
    assert_order(true);
}
#[test]
fn eval_insert_result_guard_owns_attribution_and_effect_order() {
    assert_producer(false);
}
#[test]
fn c_insert_result_guard_owns_attribution_and_effect_order() {
    assert_producer(true);
}

#[test]
fn input_axis_claim_still_precedes_body_on_both_lanes() {
    let source = "def opaque[n](x: tensor[n, i64]) -> tensor[*, i64] = shrink(x, [[0i64, shape(x, 0i32)]])\ndef candidate[n](a: tensor[n, i64], b: tensor[n, i64]) -> tensor[3, i64] ! { IO } = {\n _ = print(\"body-ran\")\n _ = [-9223372036854775808i64] |> to_tensor |> neg\n 0i64 |> scalar_to_tensor |> insert(0i32, shape(b, 0i32))\n}\nout = candidate(to_tensor([0i64, 0i64, 0i64]), opaque(to_tensor([0i64, 0i64])))\n";
    for native in [false, true] {
        let (ok, output) = run(source, native);
        assert!(!ok, "{output}");
        assert!(
            output.contains("numeric trap: domain in load at i64"),
            "{output}"
        );
        assert!(!output.contains("overflow in neg"), "{output}");
        assert!(!output.contains("body-ran"), "{output}");
    }
}
