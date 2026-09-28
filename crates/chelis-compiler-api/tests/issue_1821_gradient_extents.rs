//! spec/04 §4.7 and spec/06 §2: differentiation retains activation obligations.

use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
use std::collections::BTreeMap;

fn request(source: String) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source,
        bindings: BTreeMap::new(),
    }
}

fn multi_target_source(width: usize) -> String {
    let actual = if width == 2 {
        "[1.0f32, 2.0f32]"
    } else {
        "[1.0f32, 2.0f32, 3.0f32]"
    };
    format!(
        "def f[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n\
         def h(x: tensor[{width}, f32], z: tensor[2, f32]) -> tensor[f32] = sum(f(x, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)\n\
         def main() = grad(h, wrt=(x, z))(to_tensor({actual}), to_tensor([4.0f32, 5.0f32]))\n"
    )
}

fn assert_extent_failure(source: String) {
    let error = eval_selected(request(source), &["main".into()])
        .expect_err("a rejected forward activation has no gradient");
    let messages = error
        .errors
        .iter()
        .map(|error| error.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        messages.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
        "{messages}"
    );
    assert!(
        messages
            .lines()
            .any(|line| line == "numeric trap: domain in load at i64"),
        "{messages}"
    );
}

fn tensor_leaves(value: &ExecutionValue) -> Vec<(Vec<i64>, Vec<f64>)> {
    match value {
        ExecutionValue::Tensor { value } => vec![(
            value.shape.clone(),
            (0..value.data.len())
                .map(|index| value.data.element_f64_lossy(index))
                .collect(),
        )],
        ExecutionValue::Tuple { value } | ExecutionValue::List { value } => {
            value.iter().flat_map(tensor_leaves).collect()
        }
        ExecutionValue::Adt { fields, .. } => fields.iter().flat_map(tensor_leaves).collect(),
        ExecutionValue::Unit => Vec::new(),
        other => panic!("unexpected cotangent {other:?}"),
    }
}

#[test]
fn multi_target_gradient_rejects_the_same_activation_as_the_forward_call() {
    assert_extent_failure(multi_target_source(2));
}

const CLAIM_FUNCTION: &str = "def f[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = insert(scalar_to_tensor(7.0f32), 0i32, shape(y, 0i32))\n";

fn independent_activation_source(
    distinct_function: bool,
    distinct_binders: bool,
    second_width: usize,
    producing_widths: (usize, usize),
    zero_gradient: bool,
) -> String {
    let tensor = |width: usize| {
        format!(
            "to_tensor([{}])",
            (1..=width)
                .map(|value| format!("{value}.0f32"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let (second_function, declaration) = if distinct_function {
        let (input, output) = if distinct_binders {
            ("p", "q")
        } else {
            ("n", "m")
        };
        (
            "g",
            format!(
                "def g[{input}, {output}](x: tensor[{input}, f32], y: tensor[{output}, f32]) -> tensor[{input}, f32] = insert(scalar_to_tensor(11.0f32), 0i32, shape(y, 0i32))\n"
            ),
        )
    } else {
        ("f", String::new())
    };
    let left = format!("f(copy(a), {})", tensor(producing_widths.0));
    let right = format!("{second_function}(copy(b), {})", tensor(producing_widths.1));
    let (left, right) = if zero_gradient {
        (left, right)
    } else {
        (format!("mul({left}, a)"), format!("mul({right}, b)"))
    };
    format!(
        "{CLAIM_FUNCTION}{declaration}def h(a: tensor[2, f32], b: tensor[{second_width}, f32]) -> tensor[f32] = add(sum({left}, 0i32), sum({right}, 0i32))\ndef main() = grad(h, wrt=(b, a))({}, {})\n",
        tensor(2),
        tensor(second_width)
    )
}

#[test]
fn independent_gradient_activations_keep_declaring_extent_witnesses() {
    for (distinct_function, distinct_binders) in [(false, false), (true, false), (true, true)] {
        for second_width in [2, 3] {
            for zero_gradient in [false, true] {
                let source = independent_activation_source(
                    distinct_function,
                    distinct_binders,
                    second_width,
                    (2, second_width),
                    zero_gradient,
                );
                let result = eval_selected(request(source.clone()), &["main".into()])
                    .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
                let leaves = result
                    .roots
                    .iter()
                    .flat_map(|root| tensor_leaves(&root.value))
                    .collect::<Vec<_>>();
                let left = if zero_gradient { 0.0 } else { 7.0 };
                let right = if zero_gradient {
                    0.0
                } else if distinct_function {
                    11.0
                } else {
                    7.0
                };
                assert_eq!(
                    leaves,
                    vec![
                        (vec![second_width as i64], vec![right; second_width]),
                        (vec![2], vec![left; 2])
                    ],
                    "{source}"
                );
            }
        }
    }
}

#[test]
fn independent_gradient_activations_reject_only_their_own_claims_in_source_order() {
    for (distinct_function, distinct_binders) in [(false, false), (true, false), (true, true)] {
        for zero_gradient in [false, true] {
            for (producing_widths, label, claimed, actual) in [
                ((4, 5), "n", 2, 4),
                ((2, 5), if distinct_binders { "p" } else { "n" }, 3, 5),
            ] {
                let source = independent_activation_source(
                    distinct_function,
                    distinct_binders,
                    3,
                    producing_widths,
                    zero_gradient,
                );
                let error = eval_selected(request(source.clone()), &["main".into()])
                    .expect_err("each activation must discharge its own result claim");
                let messages = error
                    .errors
                    .iter()
                    .map(|error| error.message.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                assert!(
                    messages.contains(&format!(
                        "extent `{label}`: x axis 0 = {claimed}, y axis 0 = {actual}"
                    )),
                    "{source}\n{messages}"
                );
                assert!(
                    messages
                        .lines()
                        .any(|line| line == "numeric trap: domain in load at i64"),
                    "{messages}"
                );
            }
        }
    }
}

#[test]
fn independent_gradient_activations_preserve_computed_only_claims() {
    for (distinct_function, distinct_binders) in [(false, false), (true, false), (true, true)] {
        for zero_gradient in [false, true] {
            for mismatch in [false, true] {
                let source = independent_activation_source(
                    distinct_function,
                    distinct_binders,
                    3,
                    (2, if mismatch { 5 } else { 3 }),
                    zero_gradient,
                )
                .replace("shape(y, 0i32))", "add(shape(y, 0i32), 0i64))");
                let result = eval_selected(request(source.clone()), &["main".into()]);
                if mismatch {
                    let error = result.expect_err("a computed result has its own extent claim");
                    let messages = error
                        .errors
                        .iter()
                        .map(|error| error.message.as_str())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let label = if distinct_binders { "p" } else { "n" };
                    assert!(
                        messages
                            .contains(&format!("extent `{label}`: claimed = 3, insert axis 0 = 5")),
                        "{source}\n{messages}"
                    );
                    assert!(
                        messages
                            .lines()
                            .any(|line| line == "numeric trap: domain in insert at i64"),
                        "{messages}"
                    );
                } else {
                    let result = result.unwrap_or_else(|error| panic!("{source}\n{error:?}"));
                    let leaves = result
                        .roots
                        .iter()
                        .flat_map(|root| tensor_leaves(&root.value))
                        .collect::<Vec<_>>();
                    let left = if zero_gradient { 0.0 } else { 7.0 };
                    let right = if zero_gradient {
                        0.0
                    } else if distinct_function {
                        11.0
                    } else {
                        7.0
                    };
                    assert_eq!(
                        leaves,
                        vec![(vec![3], vec![right; 3]), (vec![2], vec![left; 2])],
                        "{source}"
                    );
                }
            }
        }
    }
}

#[test]
fn aggregate_gradient_preserves_the_named_claim_for_every_leaf_layout() {
    for (declaration, parameter_type, selected_leaf, actual) in [
        (
            "type Params =\n | Params { w: tensor[2, f32], b: tensor[2, f32] }\n",
            "Params",
            "p.w",
            "Params { w: to_tensor([1.0f32, 2.0f32]), b: to_tensor([3.0f32, 4.0f32]) }",
        ),
        (
            "",
            "(tensor[2, f32], tensor[2, f32])",
            "p.0",
            "(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32]))",
        ),
        (
            "type Params =\n | Params { w: tensor[2, f32] }\n",
            "Params",
            "p.w",
            "Params { w: to_tensor([1.0f32, 2.0f32]) }",
        ),
        (
            "type Inner =\n | Inner { w: tensor[2, f32] }\ntype Params =\n | Params { inner: Inner }\n",
            "Params",
            "p.inner.w",
            "Params { inner: Inner { w: to_tensor([1.0f32, 2.0f32]) } }",
        ),
        (
            "type Params =\n | Params { w: tensor[2, f32], count: i32 }\n",
            "Params",
            "p.w",
            "Params { w: to_tensor([1.0f32, 2.0f32]), count: 4i32 }",
        ),
    ] {
        for multiple in [false, true] {
            let extra = if multiple { ", z: tensor[2, f32]" } else { "" };
            let selection = if multiple {
                "grad(h, wrt=(p, z))"
            } else {
                "grad(h)"
            };
            let extra_actual = if multiple {
                ", to_tensor([5.0f32, 6.0f32])"
            } else {
                ""
            };
            assert_extent_failure(format!(
                "{declaration}{CLAIM_FUNCTION}def h(p: {parameter_type}{extra}) -> tensor[f32] = sum(f({selected_leaf}, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32)\ndef main() = {selection}({actual}{extra_actual})\n"
            ));
            let agreeing = format!(
                "{declaration}{CLAIM_FUNCTION}def h(p: {parameter_type}{extra}) -> tensor[f32] = sum(mul(f(copy({selected_leaf}), to_tensor([1.0f32, 2.0f32])), {selected_leaf}), 0i32)\ndef main() = {selection}({actual}{extra_actual})\n"
            );
            let result = eval_selected(request(agreeing), &["main".into()])
                .expect("an agreeing aggregate activation differentiates");
            let leaves = result
                .roots
                .iter()
                .flat_map(|root| tensor_leaves(&root.value))
                .collect::<Vec<_>>();
            let mut expected = vec![(vec![2], vec![7.0, 7.0])];
            if parameter_type.starts_with('(') || declaration.contains("b: tensor") {
                expected.push((vec![2], vec![0.0, 0.0]));
            }
            if multiple {
                expected.push((vec![2], vec![0.0, 0.0]));
            }
            assert_eq!(leaves, expected, "{parameter_type}, multiple={multiple}");
        }
    }
}

#[test]
fn primitive_scalar_gradient_preserves_the_named_claim_for_single_and_multiple_targets() {
    for selection in ["s", "(s, x)"] {
        assert_extent_failure(format!(
            "{CLAIM_FUNCTION}def h(s: f32, x: tensor[2, f32]) -> tensor[f32] = mul(sum(f(x, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32), scalar_to_tensor(s))\ndef main() = grad(h, wrt={selection})(3.0f32, to_tensor([1.0f32, 2.0f32]))\n"
        ));
        let agreeing = format!(
            "{CLAIM_FUNCTION}def h(s: f32, x: tensor[2, f32]) -> tensor[f32] = mul(sum(f(x, to_tensor([1.0f32, 2.0f32])), 0i32), scalar_to_tensor(s))\ndef main() = grad(h, wrt={selection})(3.0f32, to_tensor([1.0f32, 2.0f32]))\n"
        );
        let result = eval_selected(request(agreeing), &["main".into()]).unwrap();
        assert_eq!(result.roots[0].display.as_deref(), Some("14.0"));
        if selection == "s" {
            assert_eq!(result.roots.len(), 1);
        } else {
            assert_eq!(result.roots.len(), 2);
            assert_eq!(
                result.roots[1].display.as_deref(),
                Some("tensor(shape=[2], data=[0.0, 0.0])")
            );
        }
    }
}

#[test]
fn fixed_control_gradients_keep_the_same_authored_activation_claim() {
    for width in [2, 3] {
        let values = if width == 2 {
            "[1.0f32, 2.0f32]"
        } else {
            "[1.0f32, 2.0f32, 3.0f32]"
        };
        let source = format!(
            "{CLAIM_FUNCTION}def h(k: key, x: tensor[{width}, f32], z: tensor[2, f32]) -> tensor[f32] = dropout(k, sum(f(x, to_tensor([1.0f32, 2.0f32, 3.0f32])), 0i32), 0.0f32)\ndef main() = grad(h, wrt=(x, z))(key_from_seed(42i64), to_tensor({values}), to_tensor([4.0f32, 5.0f32]))\n"
        );
        if width == 2 {
            assert_extent_failure(source);
        } else {
            let result = eval_selected(request(source), &["main".into()]).unwrap();
            assert_eq!(result.roots.len(), 2);
            assert_eq!(
                result.roots[0].display.as_deref(),
                Some("tensor(shape=[3], data=[0.0, 0.0, 0.0])")
            );
            assert_eq!(
                result.roots[1].display.as_deref(),
                Some("tensor(shape=[2], data=[0.0, 0.0])")
            );
        }
    }
}

#[test]
fn subexpression_context_does_not_replace_an_authored_binder_with_inferred_spelling() {
    use chelis_ir::dag::{DimInfo, TensorType};
    use chelis_ir::eval::{TensorValue, eval_tensor_roots_with_strict};
    use chelis_ir::lower::SubexprLoweringContext;
    use chelis_types::types::Prim;
    use chelis_unord::UnordMap;
    let parse = |source: &str| chelis_deep::parser::parse_str(source).unwrap().remove(0);
    let signature = |first, second| {
        parse(&format!(
            "(t-fn {{}} (t-tensor {{}} (d-var {{}} {first}) (t-prim {{}} f32)) (t-tensor {{}} (d-var {{}} {second}) (t-prim {{}} f32)) (t-tensor {{}} (d-var {{}} {first}) (t-prim {{}} f32)))"
        ))
    };
    let definition = parse(
        "(fn {} (params {} (x {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}) (y {type: (t-tensor {} (d-var {} m) (t-prim {} f32))})) (app {} (var {} insert) (lit {type: (t-prim {} f32)} 7.0) (lit {type: (t-prim {} i32)} 0) (app {} (var {} shape) (var {} y) (lit {type: (t-prim {} i32)} 0))))",
    );
    let expression = parse("(app {} (var {} f) (var {} left) (var {} right))");
    let context = SubexprLoweringContext::new(
        UnordMap::from([("f".into(), signature("d43", "d44"))]),
        UnordMap::from([("f".into(), definition)]),
        UnordMap::from([("f".into(), signature("n", "m"))]),
    );
    let shape_type = |name: &str| TensorType {
        dims: vec![DimInfo::Named(name.into(), None)],
        precision: Prim::F32,
    };
    let dag = chelis_ir::lower::try_lower_subexpr_program_with_context(
        &expression,
        UnordMap::from([
            ("left".into(), shape_type("left_length")),
            ("right".into(), shape_type("right_length")),
        ]),
        &context,
    )
    .unwrap();
    for width in [2, 3] {
        let result = eval_tensor_roots_with_strict(&dag, dag.roots(), |name| {
            Some(TensorValue::from_vec(
                vec![if name == "left" { 2 } else { width }],
                vec![1.0; if name == "left" { 2 } else { width }],
            ))
        });
        if width == 3 {
            let error = result.unwrap_err();
            assert!(
                error.contains("extent `n`: x axis 0 = 2, y axis 0 = 3"),
                "{error}"
            );
            assert!(
                error
                    .lines()
                    .any(|line| line == "numeric trap: domain in load at i64"),
                "{error}"
            );
        } else {
            let values = result.unwrap();
            assert_eq!(values[&dag.roots()[0]].to_f64_lossy_vec(), [7.0, 7.0]);
        }
    }
}

#[test]
fn agreeing_multi_target_gradient_keeps_unused_zero_cotangent_slots() {
    let result = eval_selected(request(multi_target_source(3)), &["main".into()])
        .expect("an agreeing activation differentiates");
    assert_eq!(result.roots.len(), 2);
    for (root, width) in result.roots.iter().zip([3usize, 2]) {
        let ExecutionValue::Tensor { value } = &root.value else {
            panic!("{root:?}")
        };
        assert_eq!(value.shape, [width as i64]);
        assert_eq!(value.data.len(), width);
        for index in 0..width {
            assert_eq!(value.data.element_f64_lossy(index), 0.0);
        }
    }
    for (selection, expected) in [
        ("x", vec![vec![7.0, 7.0]]),
        ("(z, x)", vec![vec![2.0, 3.0], vec![7.0, 7.0]]),
    ] {
        let source = format!(
            "{CLAIM_FUNCTION}def h(x: tensor[2, f32], z: tensor[2, f32]) -> tensor[f32] = sum(add(mul(f(copy(x), to_tensor([1.0f32, 2.0f32])), x), mul(z, to_tensor([2.0f32, 3.0f32]))), 0i32)\ndef main() = grad(h, wrt={selection})(to_tensor([1.0f32, 2.0f32]), to_tensor([4.0f32, 5.0f32]))\n"
        );
        let result = eval_selected(request(source), &["main".into()]).unwrap();
        assert_eq!(
            result
                .roots
                .iter()
                .flat_map(|root| tensor_leaves(&root.value))
                .collect::<Vec<_>>(),
            expected
                .into_iter()
                .map(|data| (vec![2], data))
                .collect::<Vec<_>>()
        );
    }
}
