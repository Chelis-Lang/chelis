//! Spec/04 §4.7 and [04-NUM-9]: literal result placement and attribution.

mod common;
#[path = "common/result_claims.rs"]
mod result_claims;
use result_claims::{assert_claim, run};

fn pure_helper_claims(native: bool) {
    for (producer, body, result_prim) in [
        ("shrink", "shrink(x, [[1i64, shape(x, 0i32)]])", "f32"),
        (
            "add",
            "{\n shortened = shrink(x, [[1i64, shape(x, 0i32)]])\n add(shortened, shortened)\n}",
            "f32",
        ),
        (
            "cast",
            "cast(shrink(x, [[1i64, shape(x, 0i32)]]), f64)",
            "f64",
        ),
    ] {
        for agrees in [true, false] {
            let last = if agrees {
                "[1.0f32, 2.0f32, 3.0f32, 4.0f32]"
            } else {
                "[1.0f32, 2.0f32, 3.0f32]"
            };
            let source = format!(
                "def cut(x: tensor[n, f32]) -> tensor[*, {result_prim}] = {body}\n\
             def inner(x: tensor[n, f32]) -> tensor[*, {result_prim}] ! {{ IO }} = {{\n _ = print(\"producer-before\")\n value = cut(x)\n _ = print(\"producer-after\")\n value\n}}\n\
             def two(x: tensor[n, f32]) -> tensor[2, {result_prim}] ! {{ IO }} = inner(x)\n\
             def three(x: tensor[n, f32]) -> tensor[3, {result_prim}] ! {{ IO }} = inner(x)\n\
             a = two(to_tensor([1.0f32, 2.0f32, 3.0f32]))\n\
             b = three(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n\
             out = three(to_tensor({last}))\n"
            );
            let (ok, output) = run(&source, native);
            assert_eq!(ok, agrees, "{source}\n{output}");
            assert_eq!(output.matches("producer-before").count(), 3, "{output}");
            assert_eq!(
                output.matches("producer-after").count(),
                if agrees { 3 } else { 2 },
                "{output}"
            );
            if agrees {
                let expected = if producer == "add" {
                    "[4.0, 6.0, 8.0]"
                } else {
                    "[2.0, 3.0, 4.0]"
                };
                assert!(
                    output.contains(&format!("out = tensor(shape=[3], data={expected})")),
                    "{output}"
                );
            } else {
                assert_claim(&output, producer, 2);
            }
        }
    }
}
fn helper_owned_claims(native: bool) {
    for (producer, body) in [
        (
            "shrink",
            "shrink(x, [[1i64, shape(x, 0i32)], [1i64, shape(x, 1i32)]])",
        ),
        (
            "add",
            "{\n shortened = shrink(x, [[1i64, shape(x, 0i32)], [1i64, shape(x, 1i32)]])\n add(shortened, shortened)\n}",
        ),
    ] {
        for (rows, cols, expected) in [
            (3, 4, None),
            (3, 3, Some((3, 1, 2))),
            (4, 4, Some((2, 0, 3))),
            (4, 3, Some((2, 0, 3))),
        ] {
            let values = (0..rows)
                .map(|row| {
                    let values = (0..cols)
                        .map(|col| format!("{}.0f32", row * cols + col + 1))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("[{values}]")
                })
                .collect::<Vec<_>>()
                .join(", ");
            let source = format!(
                "def cut(x: tensor[n, m, f32]) -> tensor[2, *, f32] = {body}\n\
                 def erased(x: tensor[n, m, f32]) -> tensor[*, *, f32] ! {{ IO }} = {{\n _ = print(\"owned-before\")\n value = cut(x)\n _ = print(\"owned-after\")\n value\n}}\n\
                 def caller(x: tensor[n, m, f32]) -> tensor[*, 3, f32] ! {{ IO }} = erased(x)\n\
                 out = caller(to_tensor([{values}]))\n"
            );
            let (ok, output) = run(&source, native);
            assert_eq!(ok, expected.is_none(), "{source}\n{output}");
            assert_eq!(output.matches("owned-before").count(), 1, "{output}");
            assert_eq!(
                output.matches("owned-after").count(),
                usize::from(ok),
                "{output}"
            );
            if let Some((required, axis, observed)) = expected {
                assert!(output.contains(&format!("extent `{required}`: claimed = {required}, {producer} axis {axis} = {observed}")), "{output}");
                assert!(
                    output
                        .lines()
                        .any(|line| line == format!("numeric trap: domain in {producer} at int64")),
                    "{output}"
                );
            } else {
                let expected = if producer == "add" {
                    "[12.0, 14.0, 16.0, 20.0, 22.0, 24.0]"
                } else {
                    "[6.0, 7.0, 8.0, 10.0, 11.0, 12.0]"
                };
                assert!(
                    output.contains(&format!("out = tensor(shape=[2, 3], data={expected})")),
                    "{output}"
                );
            }
        }
    }
}

fn cast_element_and_extent_order(native: bool) {
    for (values, expected) in [
        ("[1.0f32, 2.0f32, 3.0f32, 4.0f32]", "success"),
        ("[1.0f32, 2.5f32, 3.5f32]", "extent"),
        ("[1.0f32, 2.5f32, 3.5f32, 4.5f32]", "element"),
    ] {
        let source = format!(
            "def convert(x: tensor[n, f32]) -> tensor[*, int64] = cast(shrink(x, [[1i64, shape(x, 0i32)]]), int64)\n\
             def claimed(x: tensor[n, f32]) -> tensor[3, int64] ! {{ IO }} = {{\n _ = print(\"cast-before\")\n value = convert(x)\n _ = print(\"cast-after\")\n value\n}}\n\
             out = claimed(to_tensor({values}))\n"
        );
        let (ok, output) = run(&source, native);
        assert_eq!(ok, expected == "success", "{source}\n{output}");
        assert!(output.contains("cast-before"), "{output}");
        assert_eq!(output.contains("cast-after"), ok, "{output}");
        match expected {
            "success" => assert!(
                output.contains("out = tensor(shape=[3], data=[2, 3, 4])"),
                "{output}"
            ),
            "extent" => {
                assert_claim(&output, "cast", 2);
                assert!(!output.contains("hint:"), "{output}");
            }
            _ => {
                assert!(
                    output.contains("numeric trap: domain in cast at int64"),
                    "{output}"
                );
                assert!(!output.contains("extent `"), "{output}");
                if !native {
                    assert!(
                        output.contains("fractional float-to-int conversion"),
                        "{output}"
                    );
                }
            }
        }
    }
}

#[test]
fn eval_inherited_result_claim_enters_pure_helpers() {
    pure_helper_claims(false);
    cast_element_and_extent_order(false);
    helper_owned_claims(false);
}

#[test]
fn c_inherited_result_claim_enters_pure_helpers() {
    pure_helper_claims(true);
    cast_element_and_extent_order(true);
    helper_owned_claims(true);
}
