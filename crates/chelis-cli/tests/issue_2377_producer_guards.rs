//! Spec/04 §4.7: a result extent belongs to its producing operation even
//! when its size is available from interface inputs.
mod common;
#[path = "common/result_claims.rs"]
mod result_claims;

use result_claims::{assert_claim, run};

fn candidate(values: &str, effects: bool, monomorphic: bool) -> String {
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
        "def candidate[n, p: Float](witness: p, extent_source: tensor[n, i64]) -> tensor[3, p]{effect} = {{\n {before}\n result = witness |> scalar_to_tensor |> insert(0i32, shape(extent_source, 0i32))\n {after}\n result\n}}\nout = candidate(16777217.0f64, to_tensor({values}))\n"
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

fn assert_producer(native: bool) {
    for monomorphic in [false, true] {
        for effects in [false, true] {
            for (values, agrees) in [("[0i64, 0i64]", false), ("[0i64, 0i64, 0i64]", true)] {
                let (ok, output) = run(&candidate(values, effects, monomorphic), native);
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

// A second insert forwards the first insert's axis. Both insertion positions
// exercise the output-to-operand axis relationship instead of assuming axis 0.
fn assert_forwarded_insert_axis(native: bool) {
    for monomorphic in [false, true] {
        for insert_axis in [0, 1] {
            for width in [2, 3] {
                let generics = if monomorphic {
                    "n, m"
                } else {
                    "n, m, p: Float"
                };
                let parameter = if monomorphic { "" } else { "witness: p, " };
                let dtype = if monomorphic { "f64" } else { "p" };
                let scalar = if monomorphic { "7.0f64" } else { "witness" };
                let actual = if monomorphic { "" } else { "7.0f64, " };
                let dims = if insert_axis == 0 { "4, 3" } else { "3, 4" };
                let values = vec!["0i64"; width].join(", ");
                let source = format!(
                    "def candidate[{generics}]({parameter}x: tensor[n, i64], y: tensor[m, i64]) -> tensor[{dims}, {dtype}] = {{\n a = {scalar} |> scalar_to_tensor |> insert(0i32, shape(y, 0i32))\n insert(a, {insert_axis}i32, shape(x, 0i32))\n}}\nout = candidate({actual}to_tensor([0i64, 0i64, 0i64, 0i64]), to_tensor([{values}]))\n"
                );
                let (ok, output) = run(&source, native);
                assert_eq!(ok, width == 3, "{source}\n{output}");
                if width == 2 {
                    let carried_axis = 1 - insert_axis;
                    assert!(
                        output.contains(&format!(
                            "extent `3`: claimed = 3, insert axis {carried_axis} = 2"
                        )),
                        "{output}"
                    );
                    assert!(
                        output
                            .lines()
                            .any(|line| line == "numeric trap: domain in insert at i64"),
                        "{output}"
                    );
                } else {
                    let values = ["7.0"; 12].join(", ");
                    assert!(
                        output.contains(&format!("out = tensor(shape=[{dims}], data=[{values}])")),
                        "{output}"
                    );
                }
                assert_eq!(
                    output.matches("numeric trap:").count(),
                    usize::from(!ok),
                    "{output}"
                );
            }
        }
    }
}

#[test]
fn eval_forwarded_insert_axis_keeps_result_ownership() {
    assert_forwarded_insert_axis(false);
}

#[test]
fn c_forwarded_insert_axis_keeps_result_ownership() {
    assert_forwarded_insert_axis(true);
}

// §4.7 orders claims ready at one producer by their declaration, including
// when one extent is inserted and another is forwarded from its operand.
fn assert_result_axis_declaration_order(native: bool) {
    for insert_axis in [0, 1] {
        for computed in [false, true] {
            for (new_bad, carried_bad) in
                [(false, false), (true, false), (false, true), (true, true)]
            {
                let new_extent = if new_bad { 5 } else { 4 };
                let carried_extent = if carried_bad { 2 } else { 3 };
                let x = vec!["0i64"; new_extent - usize::from(computed)].join(", ");
                let y = vec!["0i64"; carried_extent].join(", ");
                let size = if computed {
                    "add(shape(x, 0i32), 1i64)"
                } else {
                    "shape(x, 0i32)"
                };
                let dims = if insert_axis == 0 { "4, 3" } else { "3, 4" };
                let source = format!(
                    "def f[n, m](x: tensor[n, i64], y: tensor[m, i64]) -> tensor[{dims}, f64] = insert(insert(scalar_to_tensor(7.0f64), 0i32, shape(y, 0i32)), {insert_axis}i32, {size})\nout = f(to_tensor([{x}]), to_tensor([{y}]))\n"
                );
                let (ok, output) = run(&source, native);
                assert_eq!(ok, !new_bad && !carried_bad, "{source}\n{output}");
                if ok {
                    let values = ["7.0"; 12].join(", ");
                    assert!(
                        output.contains(&format!("out = tensor(shape=[{dims}], data=[{values}])")),
                        "{output}"
                    );
                } else {
                    let first_is_new = new_bad && (insert_axis == 0 || !carried_bad);
                    let (claim, axis, actual) = if first_is_new {
                        (4, insert_axis, new_extent)
                    } else {
                        (3, 1 - insert_axis, carried_extent)
                    };
                    assert!(
                        output.contains(&format!(
                            "extent `{claim}`: claimed = {claim}, insert axis {axis} = {actual}"
                        )),
                        "{source}\n{output}"
                    );
                    assert!(
                        output
                            .lines()
                            .any(|line| line == "numeric trap: domain in insert at i64"),
                        "{output}"
                    );
                }
                assert_eq!(
                    output.matches("numeric trap:").count(),
                    usize::from(!ok),
                    "{output}"
                );
            }
        }
    }
}

#[test]
fn eval_result_axes_follow_declaration_order() {
    assert_result_axis_declaration_order(false);
}

#[test]
fn c_result_axes_follow_declaration_order() {
    assert_result_axis_declaration_order(true);
}

// The old backend assertion placed the literal result at entry. Execute both
// obligations independently before migrating that structural expectation.
#[test]
fn mixed_named_entry_and_literal_result_keep_distinct_owners() {
    for native in [false, true] {
        for rows_agree in [false, true] {
            for cols in [2, 3] {
                let q = if rows_agree {
                    "[2.0f32, 2.0f32]"
                } else {
                    "[2.0f32]"
                };
                let a = vec!["0.0f32"; cols].join(", ");
                let source = format!(
                    "def opaque[n](x: tensor[n, f32]) -> tensor[*, f32] = shrink(x, [[0i64, shape(x, 0i32)]])\ndef f(z: tensor[rows, f32], a: tensor[cols, f32], q: tensor[rows, f32]) -> tensor[rows, 3, f32] ! {{ IO }} = {{\n _ = print(\"body-ran\")\n insert(add(z, q), 1i32, shape(a, 0i32))\n}}\nout = f(to_tensor([1.0f32, 1.0f32]), to_tensor([{a}]), opaque(to_tensor({q})))\n"
                );
                let (ok, output) = run(&source, native);
                assert_eq!(ok, rows_agree && cols == 3, "{source}\n{output}");
                if !rows_agree {
                    assert!(output.contains("domain in load at i64"), "{output}");
                    assert!(!output.contains("body-ran"), "{output}");
                } else if cols == 2 {
                    assert!(
                        output.contains("extent `3`: claimed = 3, insert axis 1 = 2"),
                        "{output}"
                    );
                    assert!(output.contains("domain in insert at i64"), "{output}");
                    assert!(output.contains("body-ran"), "{output}");
                } else {
                    assert!(
                        output.contains(
                            "out = tensor(shape=[2, 3], data=[3.0, 3.0, 3.0, 3.0, 3.0, 3.0])"
                        ),
                        "{output}"
                    );
                }
                assert_eq!(
                    output.matches("numeric trap:").count(),
                    usize::from(!ok),
                    "{output}"
                );
            }
        }
    }
}
