//! [05-OP-20..22]: float classification preserves surface and stored width.
mod common;
use assert_cmd::Command;
use common::build_and_run;

/// [04-INF-9]/[05-OP-20..22]: statically resolved aliases use the
/// direct operation's producer route even beneath an enclosing value.
#[test]
fn closed_predicate_routes_are_congruent_inside_option_payloads() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for predicate in ["is_nan", "is_finite", "is_infinite"] {
            let mask = match predicate {
                "is_nan" => "[false, true]",
                "is_finite" => "[true, false]",
                _ => "[false, false]",
            };
            for alias in [false, true] {
                let callee = if alias { "f" } else { predicate };
                let source = format!(
                    r#"
f = {predicate}
x = to_tensor([1.0{dtype},2.0{dtype},div(0.0{dtype},0.0{dtype})])
wrapped = Some({callee}(stride(x,2i64)))
def captured_option(z: tensor[3,{dtype}]) -> Option[tensor[*,bool]] = Some(f(stride(z,2i64)))
def local_option(z: tensor[3,{dtype}]) -> Option[tensor[*,bool]] = {{
  local = {predicate}
  Some(local(stride(z,2i64)))
}}
def borrowed_option[n,p: Float](h: (&tensor[n,p] -> tensor[n,bool]), z: tensor[n,p]) -> Option[tensor[n,bool]] = Some(h(z))
def producer_shadow({predicate}: (&tensor[3,{dtype}] -> tensor[3,bool]), z: tensor[3,{dtype}]) -> Option[tensor[*,bool]] = Some(f(stride(z,2i64)))
wrapped_captured = captured_option(x)
wrapped_local = local_option(x)
wrapped_borrowed = borrowed_option(f,stride(x,2i64))
wrapped_shadow = producer_shadow(is_nan,x)
inside = match wrapped with {{
  | Some(value) => value
  | None => to_tensor([false,false])
}}
"#
                );
                let evaluated = evaluate(&source);
                assert!(
                    evaluated.contains(&format!("inside = tensor(shape=[2], data={mask})")),
                    "{evaluated}"
                );
                let native = build_and_run(&source, &format!("option_{predicate}_{dtype}_{alias}"));
                assert_eq!(native, evaluated);
            }
        }
    }
}

/// Independent runtime producers remain valid at either branch choice;
/// binding one operand earlier cannot change alias executability.
#[test]
fn closed_predicate_routes_are_congruent_at_mixed_runtime_branch_joins() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for predicate in ["is_nan", "is_finite", "is_infinite"] {
            let mask = match predicate {
                "is_nan" => "[false, true]",
                "is_finite" => "[true, false]",
                _ => "[false, false]",
            };
            for alias in [false, true] {
                let callee = if alias { "f" } else { predicate };
                let source = format!(
                    r#"
f = {predicate}
x = to_tensor([1.0{dtype},2.0{dtype},div(0.0{dtype},0.0{dtype})])
bound = {callee}(stride(x,2i64))
def choose(flag: bool, z: tensor[3,{dtype}]) -> tensor[*,bool] = if flag then {callee}(stride(z,2i64)) else bound
left = choose(true,x)
right = choose(false,x)
"#
                );
                let evaluated = evaluate(&source);
                for root in ["left", "right"] {
                    assert!(
                        evaluated.contains(&format!("{root} = tensor(shape=[2], data={mask})")),
                        "{evaluated}"
                    );
                }
                assert_eq!(
                    build_and_run(&source, &format!("branch_{predicate}_{dtype}_{alias}")),
                    evaluated
                );
            }
        }
    }
}

#[test]
fn enclosing_predicate_values_reject_family_surface_and_known_shape_conflicts() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for predicate in ["is_nan", "is_finite", "is_infinite"] {
            for body in [
                "bad = Some(f(to_tensor([1i32,2i32])))\n".to_owned(),
                format!(
                    "def bad(z: tensor[2,{dtype}]) -> Option[tensor[3,bool]] = Some(f(z))\ny = bad(to_tensor([1.0{dtype},2.0{dtype}]))\n"
                ),
                format!(
                    "def bad(z: {dtype}) -> Option[tensor[bool]] = Some(f(z))\ny = bad(1.0{dtype})\n"
                ),
                format!(
                    "def bad(z: tensor[2,{dtype}]) -> tensor[2,{dtype}] = if true then f(z) else z\ny = bad(to_tensor([1.0{dtype},2.0{dtype}]))\n"
                ),
            ] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("bad_enclosure.ch");
                std::fs::write(&path, format!("f = {predicate}\n{body}")).unwrap();
                Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .args(["check", "--file"])
                    .arg(path)
                    .assert()
                    .failure();
            }
        }
    }
}

fn evaluate(source: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("classification.ch");
    std::fs::write(&path, source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

fn evaluate_json(source: &str) -> serde_json::Value {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("classification.ch");
    std::fs::write(&path, source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", "--file"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn scalar_and_tensor_classification_agree_in_eval_and_native_c() {
    for (dtype, subnormal) in [
        ("f16", "0.000000059604644775390625"),
        ("bf16", "9.183549615799121e-41"),
        ("f32", "1.401298464324817e-45"),
        ("f64", "5e-324"),
    ] {
        let source = format!(
            r#"
def nan_scalar(x: {dtype}) -> bool = is_nan(x)
def finite_scalar(x: {dtype}) -> bool = is_finite(x)
def infinite_scalar(x: {dtype}) -> bool = is_infinite(x)
def nan_tensor(x: tensor[8,{dtype}]) -> tensor[8,bool] = is_nan(x)
def finite_tensor(x: tensor[8,{dtype}]) -> tensor[8,bool] = is_finite(x)
def infinite_tensor(x: tensor[8,{dtype}]) -> tensor[8,bool] = is_infinite(x)
pos_inf = div(1.0{dtype},0.0{dtype})
neg_inf = div(-1.0{dtype},0.0{dtype})
nan = div(0.0{dtype},0.0{dtype})
x: tensor[8,{dtype}] = to_tensor([pos_inf,neg_inf,nan,0.0{dtype},-0.0{dtype},{subnormal}{dtype},1.0{dtype},-1.0{dtype}])
nan_values = nan_tensor(x)
finite_values = finite_tensor(x)
infinite_values = infinite_tensor(x)
nan_yes = nan_scalar(nan)
nan_no = nan_scalar(pos_inf)
finite_yes = finite_scalar({subnormal}{dtype})
finite_no = finite_scalar(neg_inf)
infinite_yes = infinite_scalar(neg_inf)
infinite_no = infinite_scalar(nan)
"#
        );
        for actual in [
            evaluate(&source),
            build_and_run(&source, &format!("classification_{dtype}")),
        ] {
            for expected in [
                "nan_values = tensor(shape=[8], data=[false, false, true, false, false, false, false, false])",
                "finite_values = tensor(shape=[8], data=[false, false, false, true, true, true, true, true])",
                "infinite_values = tensor(shape=[8], data=[true, true, false, false, false, false, false, false])",
                "nan_yes = true",
                "nan_no = false",
                "finite_yes = true",
                "finite_no = false",
                "infinite_yes = true",
                "infinite_no = false",
            ] {
                assert!(
                    actual.contains(expected),
                    "{dtype}: missing {expected}: {actual}"
                );
            }
        }
    }
}

#[test]
fn nonfloat_operands_wrong_shapes_and_wrong_arity_are_type_errors() {
    for operation in ["is_nan", "is_finite", "is_infinite"] {
        for source in [
            format!("result = {operation}(1i8)\n"),
            format!("result = {operation}(1i16)\n"),
            format!("result = {operation}(1i32)\n"),
            format!("result = {operation}(1i64)\n"),
            format!("f = {operation}\nresult = f(1i64)\n"),
            format!(
                "def bad(f: (&i32 -> bool), x: i32) -> bool = f(x)\nresult = bad({operation},1i32)\n"
            ),
            format!("result = {operation}(true)\n"),
            format!("result = {operation}(\"x\")\n"),
            format!("def f(x: tensor[2,f32]) -> tensor[3,bool] = {operation}(x)\n"),
            format!("def f(x: f32) -> f32 = {operation}(x)\n"),
            format!("def f(x: tensor[2,i64]) -> tensor[2,bool] = {operation}(x)\n"),
            format!("def f[p](x: p) -> bool = {operation}(x)\n"),
            format!("result = {operation}()\n"),
            format!("result = {operation}(1.0f32,2.0f32)\n"),
        ] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("rejected.ch");
            std::fs::write(&path, source).unwrap();
            Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(["check", "--file"])
                .arg(path)
                .assert()
                .failure();
        }
    }
}

#[test]
fn raw_scalar_payload_classification_agrees_in_eval_and_native_c() {
    for (dtype, width, sign, exponent, quiet) in [
        ("f16", 2, 0x8000u64, 0x7c00u64, 0x0200u64),
        ("bf16", 2, 0x8000, 0x7f80, 0x0040),
        ("f32", 4, 0x80000000, 0x7f800000, 0x00400000),
        (
            "f64",
            8,
            0x8000000000000000,
            0x7ff0000000000000,
            0x0008000000000000,
        ),
    ] {
        let patterns = [
            0,
            sign,
            1,
            sign | 1,
            exponent - 1,
            sign | (exponent - 1),
            exponent,
            sign | exponent,
            exponent | 1,
            sign | exponent | 1,
            exponent | quiet,
            sign | exponent | quiet | 3,
        ];
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("bits.bin");
        std::fs::write(
            &data,
            patterns
                .iter()
                .flat_map(|bits| bits.to_le_bytes()[..width].to_vec())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut source = format!(
            "def main() -> unit ! {{ IO }} = {{\n  m = mmap_file({:?})\n",
            data.to_str().unwrap()
        );
        let mut expected = String::new();
        for i in 0..patterns.len() {
            source.push_str(&format!("  x{i} = tensor_to_scalar(reshape(mmap_tensor(m, {}i64, 1i64, {dtype}), []))\n  _ = print((is_nan(x{i}), is_finite(x{i}), is_infinite(x{i})))\n", i * width));
            expected.push_str(&format!("({}, {}, {})\n", i >= 8, i <= 5, i == 6 || i == 7));
        }
        source.push_str("  ()\n}\nrun = main()\n");
        for actual in [
            evaluate(&source),
            build_and_run(&source, &format!("scalar_bits_{dtype}")),
        ] {
            assert!(
                actual.starts_with(&expected),
                "{dtype}: expected {expected}, got {actual}"
            );
        }
    }
}

#[test]
fn predicates_compose_through_float_generics_aliases_shapes_and_ad() {
    let source = r#"
def finite[p: Float](x: p) -> bool = is_finite(x)
def masked(x: tensor[3,f32]) -> tensor[3,f32] = where(is_finite(x), x, to_tensor([0.0f32,0.0f32,0.0f32]))
def loss(x: tensor[3,f32]) -> f32 = tensor_to_scalar(sum(masked(x),0))
f = is_finite
alias_result = f(1.0f16)
generic_result = finite(1.0bf16)
x: tensor[3,f32] = to_tensor([1.0f32,-2.0f32,0.0f32])
derivative = grad(loss)(x)
def symbolic[n,p: Float](x: tensor[n,p]) -> tensor[n,bool] = is_finite(x)
shape_result = symbolic(x)
"#;
    for actual in [
        evaluate(source),
        build_and_run(source, "classification_composition"),
    ] {
        for expected in [
            "alias_result = true",
            "generic_result = true",
            "derivative = tensor(shape=[3], data=[1.0, 1.0, 1.0])",
            "shape_result = tensor(shape=[3], data=[true, true, true])",
        ] {
            assert!(actual.contains(expected), "missing {expected}: {actual}");
        }
    }
}

#[test]
fn device_targets_report_the_open_classification_issue() {
    for target in ["hip", "metal"] {
        for dtype in ["f16", "bf16", "f32", "f64"] {
            for predicate in ["is_nan", "is_finite", "is_infinite"] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("classification.ch");
                std::fs::write(
                    &path,
                    format!("def main(x: tensor[2,{dtype}]) -> tensor[2,bool] = {predicate}(x)\n"),
                )
                .unwrap();
                let output = Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .args(["build", "--emit-c", "--target", target])
                    .arg(path)
                    .output()
                    .unwrap();
                assert!(!output.status.success());
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(
                    stderr.contains(if target == "metal" && dtype == "f64" {
                        "[04-TGT-1]"
                    } else {
                        "965"
                    }),
                    "{target} {dtype} {predicate}: {stderr}"
                );
            }
        }
    }
}

#[test]
fn predicate_values_select_exact_scalar_and_tensor_callback_signatures() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            r#"
def apply_scalar(f: (&{dtype} -> bool), x: {dtype}) -> bool = f(x)
def apply_tensor(f: (&tensor[2,{dtype}] -> tensor[2,bool]), x: tensor[2,{dtype}]) -> tensor[2,bool] = f(x)
x: tensor[2,{dtype}] = to_tensor([1.0{dtype},div(0.0{dtype},0.0{dtype})])
scalar_nan = apply_scalar(is_nan,div(0.0{dtype},0.0{dtype}))
scalar_finite = apply_scalar(is_finite,1.0{dtype})
scalar_infinite = apply_scalar(is_infinite,div(-1.0{dtype},0.0{dtype}))
tensor_nan = apply_tensor(is_nan,x)
tensor_finite = apply_tensor(is_finite,x)
tensor_infinite = apply_tensor(is_infinite,x)
"#
        );
        for actual in [
            evaluate(&source),
            build_and_run(&source, &format!("predicate_values_{dtype}")),
        ] {
            for expected in [
                "scalar_nan = true",
                "scalar_finite = true",
                "scalar_infinite = true",
                "tensor_nan = tensor(shape=[2], data=[false, true])",
                "tensor_finite = tensor(shape=[2], data=[true, false])",
                "tensor_infinite = tensor(shape=[2], data=[false, false])",
            ] {
                assert!(
                    actual.contains(expected),
                    "{dtype}: missing {expected}: {actual}"
                );
            }
        }
    }
}

#[test]
fn predicate_tensor_values_keep_their_surface_through_every_callable_route() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for (predicate, first, second) in [
            ("is_nan", false, true),
            ("is_finite", true, false),
            ("is_infinite", false, false),
        ] {
            let source = format!(
                r#"
f = {predicate}
alias = f
def local(x: tensor[2,{dtype}]) -> tensor[2,bool] = {{
  g = {predicate}
  g(x)
}}
def apply(g: (&tensor[2,{dtype}] -> tensor[2,bool]), x: tensor[2,{dtype}]) -> tensor[2,bool] = g(x)
x = to_tensor([1.0{dtype},div(0.0{dtype},0.0{dtype})])
direct = {predicate}(x)
inline_alias = f(to_tensor([1.0{dtype},div(0.0{dtype},0.0{dtype})]))
bound_alias = f(x)
chained_alias = alias(x)
local_alias = local(x)
borrowed_direct = apply({predicate},x)
borrowed_alias = apply(f,x)
scalar_alias = f(1.0{dtype})
def loss(x: tensor[2,{dtype}]) -> {dtype} = tensor_to_scalar(sum(where(f(x),to_tensor([0.0{dtype},0.0{dtype}]),x),0))
derivative = grad(loss)(to_tensor([1.0{dtype},2.0{dtype}]))
"#
            );
            for actual in [
                evaluate(&source),
                build_and_run(&source, &format!("tensor_routes_{predicate}_{dtype}")),
            ] {
                for root in [
                    "direct",
                    "inline_alias",
                    "bound_alias",
                    "chained_alias",
                    "local_alias",
                    "borrowed_direct",
                    "borrowed_alias",
                ] {
                    let expected = format!("{root} = tensor(shape=[2], data=[{first}, {second}])");
                    assert!(
                        actual.contains(&expected),
                        "{predicate} {dtype}: missing {expected}: {actual}"
                    );
                }
                assert!(
                    actual.contains(&format!("scalar_alias = {first}")),
                    "{actual}"
                );
                let derivative = if predicate == "is_finite" {
                    "0.0"
                } else {
                    "1.0"
                };
                assert!(
                    actual.contains(&format!(
                        "derivative = tensor(shape=[2], data=[{derivative}, {derivative}])"
                    )),
                    "{predicate} {dtype}: {actual}"
                );
            }
        }
    }
}

#[test]
fn predicate_rank_zero_values_and_mixed_alias_instantiations_keep_tensor_roots() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for (predicate, expected) in [
            ("is_nan", false),
            ("is_finite", true),
            ("is_infinite", false),
        ] {
            let source = format!(
                r#"
f = {predicate}
alias = f
def local(x: tensor[{dtype}]) -> tensor[bool] = {{
  g = {predicate}
  g(x)
}}
def apply(g: (&tensor[{dtype}] -> tensor[bool]), x: tensor[{dtype}]) -> tensor[bool] = g(x)
x = scalar_to_tensor(1.0{dtype})
direct = {predicate}(x)
inline_alias = f(scalar_to_tensor(1.0{dtype}))
bound_alias = f(x)
chained_alias = alias(x)
local_alias = local(x)
borrowed_direct = apply({predicate},x)
borrowed_alias = apply(f,x)
scalar_alias = f(1.0{dtype})
vector_alias = f(to_tensor([1.0{dtype},2.0{dtype}]))
rank_zero_list = to_list(reshape(chained_alias,[1i64]))
"#
            );
            let json = evaluate_json(&source);
            for root in [
                "direct",
                "inline_alias",
                "bound_alias",
                "chained_alias",
                "local_alias",
                "borrowed_direct",
                "borrowed_alias",
            ] {
                let value = &json["roots"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|value| value["name"] == root)
                    .unwrap()["value"];
                assert_eq!(value["type"], "tensor", "{predicate} {dtype} {root}");
                assert_eq!(value["value"]["shape"], serde_json::json!([]));
                assert_eq!(
                    value["value"]["data"],
                    serde_json::json!({"dtype":"bool", "values":[expected]})
                );
            }
            for actual in [
                evaluate(&source),
                build_and_run(&source, &format!("rank_zero_routes_{predicate}_{dtype}")),
            ] {
                for root in [
                    "direct",
                    "inline_alias",
                    "bound_alias",
                    "chained_alias",
                    "local_alias",
                    "borrowed_direct",
                    "borrowed_alias",
                ] {
                    let output = format!("{root} = {expected}");
                    assert!(
                        actual.contains(&output),
                        "{predicate} {dtype}: missing {output}: {actual}"
                    );
                }
                assert!(
                    actual.contains(&format!("scalar_alias = {expected}")),
                    "{actual}"
                );
                assert!(
                    actual.contains(&format!(
                        "vector_alias = tensor(shape=[2], data=[{expected}, {expected}])"
                    )),
                    "{actual}"
                );
                assert!(
                    actual.contains(&format!("rank_zero_list = [{expected}]")),
                    "{actual}"
                );
            }
        }
    }
}

#[test]
fn predicate_values_reject_mismatched_scalar_tensor_callback_surfaces() {
    for predicate in ["is_nan", "is_finite", "is_infinite"] {
        for dtype in ["f16", "bf16", "f32", "f64"] {
            for source in [
                format!(
                    "f = {predicate}\ndef bad(g: (&tensor[{dtype}] -> bool), x: tensor[{dtype}]) -> bool = g(x)\ny = bad(f,scalar_to_tensor(1.0{dtype}))\n"
                ),
                format!(
                    "f = {predicate}\ndef bad(g: (&{dtype} -> tensor[bool]), x: {dtype}) -> tensor[bool] = g(x)\ny = bad(f,1.0{dtype})\n"
                ),
                format!(
                    "f = {predicate}\ndef bad(g: (&tensor[2,{dtype}] -> tensor[3,bool]), x: tensor[2,{dtype}]) -> tensor[3,bool] = g(x)\ny = bad(f,to_tensor([1.0{dtype},2.0{dtype}]))\n"
                ),
            ] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("rejected_callable.ch");
                std::fs::write(&path, source).unwrap();
                Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .args(["check", "--file"])
                    .arg(path)
                    .assert()
                    .failure();
            }
        }
    }
}

// [04-INF-9], [05-OP-20..22]: declaration-qualified captures preserve
// checked operation identity, independently of the applying lexical scope.
#[test]
fn captured_global_predicate_alias_wrappers_preserve_checked_identity() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for (predicate, expected, nan_expected, other) in [
            ("is_nan", false, true, "is_finite"),
            ("is_finite", true, false, "is_nan"),
            ("is_infinite", false, false, "is_finite"),
        ] {
            let source = format!(
                r#"
predicate = {predicate}
chain = predicate
def scalar(x: {dtype}) -> bool = predicate(x)
def generic[p: Float](x: p) -> bool = chain(x)
def classify_tensor(x: tensor[2,{dtype}]) -> tensor[2,bool] = predicate(x)
def rank_zero(x: tensor[{dtype}]) -> tensor[bool] = chain(x)
def apply_scalar(g: (&{dtype} -> bool), x: {dtype}) -> bool = g(x)
def apply_tensor(g: (&tensor[2,{dtype}] -> tensor[2,bool]), x: tensor[2,{dtype}]) -> tensor[2,bool] = g(x)
def borrowed_scalar(x: {dtype}) -> bool = apply_scalar(predicate,x)
def borrowed_tensor(x: tensor[2,{dtype}]) -> tensor[2,bool] = apply_tensor(chain,x)
def canonical(x: {dtype}) -> bool = {predicate}(x)
def local(x: {dtype}) -> bool = {{
  predicate = {predicate}
  predicate(x)
}}
def shadow(predicate: (&{dtype} -> bool), x: {dtype}) -> bool = predicate(x)
def caller_shadow(x: {dtype}) -> bool = {{
  predicate = {other}
  scalar(x)
}}
def producer_shadow({predicate}: (&{dtype} -> bool), x: {dtype}) -> bool = chain(x)
def loss(x: tensor[2,{dtype}]) -> {dtype} = tensor_to_scalar(sum(where(classify_tensor(x),to_tensor([0.0{dtype},0.0{dtype}]),x),0))
x = to_tensor([1.0{dtype},div(0.0{dtype},0.0{dtype})])
scalar_result = scalar(1.0{dtype})
generic_result = generic(1.0{dtype})
tensor_result = classify_tensor(x)
rank_zero_result = to_list(reshape(rank_zero(scalar_to_tensor(1.0{dtype})),[1i64]))
borrowed_scalar_result = borrowed_scalar(1.0{dtype})
borrowed_tensor_result = borrowed_tensor(x)
canonical_result = canonical(1.0{dtype})
local_result = local(1.0{dtype})
shadow_result = shadow({other},1.0{dtype})
caller_shadow_result = caller_shadow(1.0{dtype})
producer_shadow_result = producer_shadow({other},1.0{dtype})
derivative = grad(loss)(to_tensor([1.0{dtype},2.0{dtype}]))
"#
            );
            for actual in [
                evaluate(&source),
                build_and_run(&source, &format!("captured_alias_{predicate}_{dtype}")),
            ] {
                for root in [
                    "scalar_result",
                    "generic_result",
                    "borrowed_scalar_result",
                    "canonical_result",
                    "local_result",
                    "caller_shadow_result",
                    "producer_shadow_result",
                ] {
                    assert!(
                        actual.contains(&format!("{root} = {expected}")),
                        "{predicate} {dtype}: {actual}"
                    );
                }
                for root in ["tensor_result", "borrowed_tensor_result"] {
                    assert!(
                        actual.contains(&format!(
                            "{root} = tensor(shape=[2], data=[{expected}, {nan_expected}])"
                        )),
                        "{predicate} {dtype}: {actual}"
                    );
                }
                assert!(
                    actual.contains(&format!("rank_zero_result = [{expected}]")),
                    "{actual}"
                );
                let derivative = if predicate == "is_finite" {
                    "0.0"
                } else {
                    "1.0"
                };
                assert!(
                    actual.contains(&format!(
                        "derivative = tensor(shape=[2], data=[{derivative}, {derivative}])"
                    )),
                    "{actual}"
                );
                assert!(
                    actual.contains(&format!("shadow_result = {}", !expected)),
                    "{actual}"
                );
            }
        }
    }
}

#[test]
fn captured_global_predicate_alias_wrappers_reject_invalid_contracts() {
    for predicate in ["is_nan", "is_finite", "is_infinite"] {
        for dtype in ["f16", "bf16", "f32", "f64"] {
            for body in [
                "def bad(x: i32) -> bool = predicate(x)\ny = bad(1i32)\n".to_string(),
                format!(
                    "def bad(x: {dtype}) -> tensor[bool] = predicate(x)\ny = bad(1.0{dtype})\n"
                ),
                format!(
                    "def apply(g: (&tensor[2,{dtype}] -> tensor[3,bool]), x: tensor[2,{dtype}]) -> tensor[3,bool] = g(x)\ndef bad(x: tensor[2,{dtype}]) -> tensor[3,bool] = apply(predicate,x)\ny = bad(to_tensor([1.0{dtype},2.0{dtype}]))\n"
                ),
            ] {
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("bad_capture.ch");
                std::fs::write(&path, format!("predicate = {predicate}\n{body}")).unwrap();
                Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .args(["check", "--file"])
                    .arg(path)
                    .assert()
                    .failure();
            }
        }
    }
}

/// [04-INF-9], [05-OP-20..22], §4.7: runtime producer axes survive
/// all admitted routes; naming the producer cannot change executability.
#[test]
fn predicate_route_producer_shape_matrix_executes_in_eval_and_native_c() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for predicate in ["is_nan", "is_finite", "is_infinite"] {
            let mut source = format!(
                r#"
f = {predicate}
g = f
def captured[n,m,p: Float](z: tensor[n,m,p]) -> tensor[n,m,bool] = f(z)
def local[n,m,p: Float](z: tensor[n,m,p]) -> tensor[n,m,bool] = {{
  h = {predicate}
  h(z)
}}
def borrowed[n,m,p: Float](h: (&tensor[n,m,p] -> tensor[n,m,bool]), z: tensor[n,m,p]) -> tensor[n,m,bool] = h(z)
def captured_vector[n,p: Float](z: tensor[n,p]) -> tensor[n,bool] = f(z)
def local_vector[n,p: Float](z: tensor[n,p]) -> tensor[n,bool] = {{
  h = {predicate}
  h(z)
}}
def borrowed_vector[n,p: Float](h: (&tensor[n,p] -> tensor[n,bool]), z: tensor[n,p]) -> tensor[n,bool] = h(z)
x = to_tensor([[1.0{dtype},2.0{dtype},div(0.0{dtype},0.0{dtype})],[4.0{dtype},5.0{dtype},6.0{dtype}]])
v = to_tensor([1.0{dtype},2.0{dtype},div(0.0{dtype},0.0{dtype})])
"#
            );
            let producers = [
                (
                    "stride_matrix",
                    "stride(x,1i64,2i64)".to_string(),
                    vec![2, 2],
                    vec![false, true, false, false],
                    false,
                ),
                (
                    "reshape_matrix",
                    "reshape(x,[2i64,3i64])".to_string(),
                    vec![2, 3],
                    vec![false, false, true, false, false, false],
                    false,
                ),
                (
                    "shrink_matrix",
                    "shrink(x,[[0i64,2i64],[1i64,3i64]])".to_string(),
                    vec![2, 2],
                    vec![false, true, false, false],
                    false,
                ),
                (
                    "pad_matrix",
                    format!("pad(x,[[0i64,0i64],[1i64,0i64]],0.0{dtype})"),
                    vec![2, 4],
                    vec![false, false, false, true, false, false, false, false],
                    false,
                ),
                (
                    "permute_matrix",
                    "permute(x,1i32,0i32)".to_string(),
                    vec![3, 2],
                    vec![false, false, false, false, true, false],
                    false,
                ),
                (
                    "empty_matrix",
                    "shrink(x,[[0i64,0i64],[0i64,3i64]])".to_string(),
                    vec![0, 3],
                    vec![],
                    false,
                ),
                (
                    "stride_vector",
                    "stride(v,2i64)".to_string(),
                    vec![2],
                    vec![false, true],
                    true,
                ),
            ];
            let mut expected = Vec::new();
            for (stem, producer, shape, nan, vector) in producers {
                let suffix = if vector { "_vector" } else { "" };
                source.push_str(&format!("{stem}_producer = {producer}\n"));
                let routes = [
                    ("direct", format!("{predicate}({producer})")),
                    ("alias", format!("f({producer})")),
                    ("chain", format!("g({producer})")),
                    ("bound", format!("f({stem}_producer)")),
                    ("local", format!("local{suffix}({producer})")),
                    ("captured", format!("captured{suffix}({producer})")),
                    ("borrowed", format!("borrowed{suffix}(f,{producer})")),
                ];
                let values = nan
                    .iter()
                    .map(|nan| match predicate {
                        "is_nan" => *nan,
                        "is_finite" => !nan,
                        "is_infinite" => false,
                        _ => unreachable!(),
                    })
                    .collect::<Vec<_>>();
                for (route, expression) in routes {
                    let name = format!("{stem}_{route}");
                    source.push_str(&format!("{name} = {expression}\n"));
                    expected.push(format!("{name} = tensor(shape={shape:?}, data={values:?})"));
                }
            }
            for actual in [
                evaluate(&source),
                build_and_run(&source, &format!("shape_routes_{predicate}_{dtype}")),
            ] {
                for expected in &expected {
                    assert!(
                        actual.contains(expected),
                        "{predicate} {dtype}: missing {expected}: {actual}"
                    );
                }
            }
        }
    }
}

/// An authored result requirement remains separate from the producer-derived
/// ABI type and still executes on the actual runtime result.
#[test]
fn predicate_runtime_axes_keep_authored_result_claims_in_eval_and_native_c() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for predicate in ["is_nan", "is_finite", "is_infinite"] {
            for (count, succeeds) in [(1, true), (3, false)] {
                let values = (1..=count)
                    .map(|n| format!("{n}.0{dtype}"))
                    .collect::<Vec<_>>()
                    .join(",");
                let source = format!(
                    "f = {predicate}\ndef claimed[n](x: tensor[n,{dtype}]) -> tensor[1,bool] = f(stride(x,2i64))\nresult = claimed(to_tensor([{values}]))\n"
                );
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("claims.ch");
                let out = dir.path().join("out");
                std::fs::write(&path, source).unwrap();
                let evaluated = Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .args(["eval", "--file"])
                    .arg(&path)
                    .output()
                    .unwrap();
                assert_eq!(
                    evaluated.status.success(),
                    succeeds,
                    "{predicate} {dtype}: {}",
                    String::from_utf8_lossy(&evaluated.stderr)
                );
                Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .arg("build")
                    .arg(&path)
                    .args(["--target", "c", "--output"])
                    .arg(&out)
                    .assert()
                    .success();
                let executed = std::process::Command::new(out.join("claims"))
                    .output()
                    .unwrap();
                assert_eq!(
                    executed.status.success(),
                    succeeds,
                    "{predicate} {dtype}: {}",
                    String::from_utf8_lossy(&executed.stderr)
                );
                if !succeeds {
                    assert!(String::from_utf8_lossy(&evaluated.stderr).contains("numeric trap"));
                    assert!(String::from_utf8_lossy(&executed.stderr).contains("numeric trap"));
                }
            }
        }
    }
}

/// [04-INF-9]: different checked function identities form a valid logical
/// callable choice. Its absent C value ABI is target-specific (chelis#879).
#[test]
fn runtime_selected_predicates_reach_the_target_callable_fence() {
    for tensor in [false, true] {
        for (choice, reverse) in [
            ("if flag then is_nan else is_finite", false),
            ("if flag then is_finite else is_nan", true),
            (
                "if flag then (if flag then is_nan else is_finite) else is_finite",
                false,
            ),
            (
                "match flag with { | true => is_nan | false => is_finite }",
                false,
            ),
        ] {
            for flag in [false, true] {
                let (input, output, value) = if tensor {
                    (
                        "tensor[2,f32]",
                        "tensor[2,bool]",
                        "to_tensor([1.0f32,2.0f32])",
                    )
                } else {
                    ("f32", "bool", "1.0f32")
                };
                let source = format!(
                    "def selected(flag: bool, x: {input}) -> {output} = {{\n  f = {choice}\n  f(x)\n}}\nresult = selected({flag},{value})\n"
                );
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("selected.ch");
                std::fs::write(&path, &source).unwrap();
                Command::cargo_bin("chelis")
                    .unwrap()
                    .args(["fmt", "--inplace"])
                    .arg(&path)
                    .assert()
                    .success();
                let source = std::fs::read_to_string(&path).unwrap();
                let checked = Command::cargo_bin("chelis")
                    .unwrap()
                    .arg("check")
                    .arg(&path)
                    .output()
                    .unwrap();
                assert!(
                    checked.status.success(),
                    "{}",
                    String::from_utf8_lossy(&checked.stderr)
                );
                let checked: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
                assert_eq!(checked["score"], 1.0);
                assert_eq!(checked["errors"], serde_json::json!([]));
                let expected = if tensor {
                    format!(
                        "result = tensor(shape=[2], data=[{0}, {0}])",
                        flag == reverse
                    )
                } else {
                    format!("result = {}", flag == reverse)
                };
                assert!(evaluate(&source).contains(&expected));
                let built = Command::cargo_bin("chelis")
                    .unwrap()
                    .args(["build", "--target", "c", "--emit-c"])
                    .arg(&path)
                    .arg("-o")
                    .arg(dir.path().join("selected.c"))
                    .output()
                    .unwrap();
                assert!(!built.status.success());
                let error = String::from_utf8_lossy(&built.stderr);
                assert!(error.contains("chelis#879"), "{error}");
                assert!(!error.contains("chelis#730"), "{error}");
                assert!(
                    !error.contains("identity") || !error.contains("disagrees"),
                    "{error}"
                );
                assert!(!dir.path().join("selected.c").exists());
            }
        }
    }
}

/// Collapsing repeated identities is static resolution, including borrowed
/// callback arguments; a logical join must not discard this admitted path.
#[test]
fn repeated_predicate_choices_keep_the_closed_callable_route() {
    for tensor in [false, true] {
        for flag in [false, true] {
            let (input, output, value, expected) = if tensor {
                (
                    "tensor[2,f32]",
                    "tensor[2,bool]",
                    "to_tensor([1.0f32,2.0f32])",
                    "tensor(shape=[2], data=[true, true])",
                )
            } else {
                ("f32", "bool", "1.0f32", "true")
            };
            let source = format!(
                "alias = is_finite\ndef borrowed(f: (&{input} -> {output}), x: {input}) -> {output} = f(x)\ndef selected(flag: bool, x: {input}) -> {output} = {{\n  f = if flag then alias else is_finite\n  f(x)\n}}\nresult = selected({flag},{value})\nborrowed_result = borrowed(alias,{value})\n"
            );
            let native = build_and_run(&source, &format!("repeated_{tensor}_{flag}"));
            assert!(native.contains(&format!("result = {expected}")), "{native}");
            assert!(
                native.contains(&format!("borrowed_result = {expected}")),
                "{native}"
            );
            assert_eq!(native, evaluate(&source));
        }
    }
}

/// [04-INF-9]/[05-OP-20..22]: runtime function selection preserves the
/// complete Bool result relation on each explicitly enumerated surface.
#[test]
fn runtime_selected_predicates_evaluate_every_float_surface() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for reverse in [false, true] {
            let choice = if reverse {
                "if flag then is_finite else is_nan"
            } else {
                "if flag then is_nan else is_finite"
            };
            for (shape, input, nan_mask, finite_mask) in [
                (
                    "",
                    format!("reshape(to_tensor([div(0.0{dtype},0.0{dtype})]), [])"),
                    "[true]",
                    "[false]",
                ),
                (
                    "5,",
                    format!(
                        "to_tensor([div(0.0{dtype},0.0{dtype}),div(1.0{dtype},0.0{dtype}),0.0{dtype},-0.0{dtype},1.0{dtype}])"
                    ),
                    "[true, false, false, false, false]",
                    "[false, false, true, true, true]",
                ),
                (
                    "1,5,",
                    format!(
                        "reshape(to_tensor([div(0.0{dtype},0.0{dtype}),div(1.0{dtype},0.0{dtype}),0.0{dtype},-0.0{dtype},1.0{dtype}]),[1i64,5i64])"
                    ),
                    "[true, false, false, false, false]",
                    "[false, false, true, true, true]",
                ),
                ("0,", format!("to_tensor([],{dtype})"), "[]", "[]"),
            ] {
                let source = format!(
                    "def selected(flag: bool, x: tensor[{shape}{dtype}]) -> tensor[{shape}bool] = {{\n f = {choice}\n g = f\n g(x)\n}}\nleft = selected(true,{input})\nright = selected(false,{input})\n"
                );
                let evaluated = evaluate_json(&source);
                assert_runtime_callable_c_fence(&source);
                for (root, flag) in [("left", true), ("right", false)] {
                    let mask = if flag != reverse {
                        nan_mask
                    } else {
                        finite_mask
                    };
                    let dimensions = match shape {
                        "" => vec![],
                        "5," => vec![5],
                        "1,5," => vec![1, 5],
                        _ => vec![0],
                    };
                    let value = &evaluated["roots"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|value| value["name"] == root)
                        .unwrap()["value"];
                    assert_eq!(value["type"], "tensor", "{source}\n{evaluated}");
                    assert_eq!(value["value"]["shape"], serde_json::json!(dimensions));
                    let mask: serde_json::Value = serde_json::from_str(mask).unwrap();
                    assert_eq!(
                        value["value"]["data"],
                        serde_json::json!({"dtype":"bool","values":mask})
                    );
                }
            }
            let scalar = format!(
                "def selected(flag: bool, x: {dtype}) -> bool = {{\n f = {choice}\n f(x)\n}}\nnan_left = selected(true,div(0.0{dtype},0.0{dtype}))\nnan_right = selected(false,div(0.0{dtype},0.0{dtype}))\nfinite_left = selected(true,-0.0{dtype})\nfinite_right = selected(false,-0.0{dtype})\n"
            );
            let evaluated = evaluate(&scalar);
            assert_runtime_callable_c_fence(&scalar);
            assert!(
                evaluated.contains(&format!("nan_left = {}", !reverse)),
                "{evaluated}"
            );
            assert!(
                evaluated.contains(&format!("nan_right = {reverse}")),
                "{evaluated}"
            );
            assert!(
                evaluated.contains(&format!("finite_left = {reverse}")),
                "{evaluated}"
            );
            assert!(
                evaluated.contains(&format!("finite_right = {}", !reverse)),
                "{evaluated}"
            );
        }
    }
}

#[test]
fn runtime_callable_eval_preserves_named_functions_aliases_and_borrowing() {
    let source = r#"
def left(x: tensor[2,f32]) -> tensor[2,f32] = abs(x)
def right(x: tensor[2,f32]) -> tensor[2,f32] = neg(x)
def selected(flag: bool, x: tensor[2,f32]) -> tensor[2,f32] = {
  f = if flag then left else right
  alias = copy(f)
  alias(x)
}
def borrowed(flag: bool, x: &tensor[2,f32]) -> tensor[2,bool] = {
  f = if flag then is_nan else is_finite
  alias = copy(f)
  alias(x)
}
left_result = selected(true,to_tensor([1.0f32,-2.0f32]))
right_result = selected(false,to_tensor([1.0f32,-2.0f32]))
borrowed_result = borrowed(false,to_tensor([1.0f32,div(0.0f32,0.0f32)]))
"#;
    let evaluated = evaluate(source);
    assert!(
        evaluated.contains("left_result = tensor(shape=[2], data=[1.0, 2.0])"),
        "{evaluated}"
    );
    assert!(
        evaluated.contains("right_result = tensor(shape=[2], data=[-1.0, 2.0])"),
        "{evaluated}"
    );
    assert!(
        evaluated.contains("borrowed_result = tensor(shape=[2], data=[true, false])"),
        "{evaluated}"
    );
}

#[test]
fn runtime_callable_rejects_invalid_contracts_before_execution() {
    for source in [
        "def bad(flag: bool, x: tensor[2,i32]) -> tensor[2,bool] = { f = if flag then is_nan else is_finite\n f(x) }\nresult = bad(true,to_tensor([1i32,2i32]))\n",
        "def bad(flag: i32, x: tensor[2,f32]) -> tensor[2,bool] = { f = if flag then is_nan else is_finite\n f(x) }\nresult = bad(1i32,to_tensor([1.0f32,2.0f32]))\n",
        "def bad(flag: bool, x: tensor[2,f32]) -> tensor[3,bool] = { f = if flag then is_nan else is_finite\n f(x) }\nresult = bad(true,to_tensor([1.0f32,2.0f32]))\n",
        "def bad(flag: bool, x: tensor[2,f32]) -> tensor[2,bool] = { f = if flag then is_nan else abs\n f(x) }\nresult = bad(true,to_tensor([1.0f32,2.0f32]))\n",
        "def bad(flag: bool, x: tensor[2,f32]) -> tensor[2,bool] = { is_nan = 1i32\n f = if flag then is_nan else is_finite\n f(x) }\nresult = bad(true,to_tensor([1.0f32,2.0f32]))\n",
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_callable.ch");
        std::fs::write(&path, source).unwrap();
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("check")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "{source}\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

#[test]
fn runtime_callable_selection_evaluates_actual_condition_and_body_once() {
    for flag in [false, true] {
        let source = format!(
            r#"
def condition(flag: bool) -> bool = {{
  _ = print("condition")
  flag
}}
def actual() -> tensor[2,f32] = {{
  _ = print("actual")
  to_tensor([1.0f32,div(0.0f32,0.0f32)])
}}
def left(x: tensor[2,f32]) -> tensor[2,bool] = {{
  _ = print("left")
  is_nan(x)
}}
def right(x: tensor[2,f32]) -> tensor[2,bool] = {{
  _ = print("right")
  is_finite(x)
}}
def selected(flag: bool, x: tensor[2,f32]) -> tensor[2,bool] = {{
  f = if condition(flag) then left else right
  alias = f
  alias(x)
}}
result = selected({flag},actual())
"#
        );
        let output = evaluate_json(&source);
        let transcript = output["transcript"].as_array().unwrap();
        assert_eq!(
            transcript,
            &vec![
                serde_json::json!("actual"),
                serde_json::json!("condition"),
                serde_json::json!(if flag { "left" } else { "right" })
            ],
            "{output}"
        );
    }
}

#[test]
fn runtime_callable_selection_does_not_retry_or_execute_the_untaken_body() {
    let source = r#"
def trap(x: tensor[2,f32]) -> tensor[2,bool] = {
  _ = print("trap_once")
  fail("selected_trap")
}
def safe(x: tensor[2,f32]) -> tensor[2,bool] = is_finite(x)
def selected(flag: bool, x: tensor[2,f32]) -> tensor[2,bool] = {
  f = if flag then trap else safe
  f(x)
}
"#;
    assert!(
        evaluate(&format!(
            "{source}\nresult = selected(false,to_tensor([1.0f32,2.0f32]))\n"
        ))
        .contains("result = tensor(shape=[2], data=[true, true])")
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trap.ch");
    std::fs::write(
        &path,
        format!("{source}\nresult = selected(true,to_tensor([1.0f32,2.0f32]))\n"),
    )
    .unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("selected_trap"));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout)
            .matches("trap_once")
            .count(),
        1
    );
}

fn assert_runtime_callable_c_fence(source: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("choice.ch");
    let artifact = dir.path().join("choice.c");
    std::fs::write(&path, source).unwrap();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--target", "c", "--emit-c"])
        .arg(path)
        .arg("-o")
        .arg(&artifact)
        .output()
        .unwrap();
    assert!(!output.status.success(), "{source}");
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains("chelis#879"), "{source}\n{error}");
    assert!(!error.contains("chelis#730"), "{source}\n{error}");
    assert!(!artifact.exists());
}

#[test]
fn repeated_identity_choices_execute_in_eval_and_native_at_every_float_width() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            r#"
predicate = is_finite
def selected(flag: bool, x: tensor[2,{dtype}]) -> tensor[2,bool] = {{
  f = if flag then predicate else is_finite
  alias = f
  alias(x)
}}
left = selected(true,to_tensor([1.0{dtype},div(0.0{dtype},0.0{dtype})]))
right = selected(false,to_tensor([1.0{dtype},div(0.0{dtype},0.0{dtype})]))
"#
        );
        let evaluated = evaluate(&source);
        for root in ["left", "right"] {
            assert!(
                evaluated.contains(&format!("{root} = tensor(shape=[2], data=[true, false])")),
                "{evaluated}"
            );
        }
        assert_eq!(
            build_and_run(&source, &format!("same_identity_{dtype}")),
            evaluated
        );
    }
}

/// spec/04 §4.7: a whole-choice callable ascription guards entry even when
/// the enclosing signature permits arbitrary extents. Aliasing cannot erase it.
#[test]
fn runtime_callable_whole_choice_contract_guards_before_later_effects() {
    for (binding, call) in [
        ("", "f"),
        ("alias = f", "alias"),
        ("alias = copy(f)", "alias"),
        ("alias = copy(copy(f))", "alias"),
        ("alias: &tensor[2,f64] -> tensor[2,bool] = copy(f)", "alias"),
        (
            "first: &tensor[2,f64] -> tensor[2,bool] = copy(f)\n alias: &tensor[2,f64] -> tensor[2,bool] = copy(first)",
            "alias",
        ),
    ] {
        for flag in [false, true] {
            for reverse in [false, true] {
                let (left, right) = if reverse {
                    ("is_finite", "is_nan")
                } else {
                    ("is_nan", "is_finite")
                };
                for width in [2, 3] {
                    let values = if width == 2 {
                        "1.0f64, 2.0f64"
                    } else {
                        "1.0f64, 2.0f64, 3.0f64"
                    };
                    let source = format!(
                        r#"
def run(flag: bool, x: tensor[*,f64]) -> tensor[*,bool] = {{
  f: &tensor[2,f64] -> tensor[2,bool] = if flag then {left} else {right}
  {binding}
  {call}(x)
}}
result = run({flag},to_tensor([{values}]))
after = print("after")
"#
                    );
                    let dir = tempfile::tempdir().unwrap();
                    let path = dir.path().join("whole_choice.ch");
                    std::fs::write(&path, &source).unwrap();
                    let check = Command::cargo_bin("chelis")
                        .unwrap()
                        .env("CHELIS_STYLE_GATE_DISABLE", "1")
                        .arg("check")
                        .arg(&path)
                        .output()
                        .unwrap();
                    assert!(check.status.success(), "{source}\n{:?}", check);
                    let output = Command::cargo_bin("chelis")
                        .unwrap()
                        .env("CHELIS_STYLE_GATE_DISABLE", "1")
                        .args(["eval", "--file"])
                        .arg(&path)
                        .output()
                        .unwrap();
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if width == 2 {
                        assert!(output.status.success(), "{source}\n{stderr}");
                        let expected = if flag != reverse {
                            "[false, false]"
                        } else {
                            "[true, true]"
                        };
                        assert!(
                            stdout.contains(&format!("data={expected}")),
                            "{source}\n{stdout}"
                        );
                        assert_eq!(
                            stdout.lines().filter(|line| *line == "after").count(),
                            1,
                            "{stdout}"
                        );
                    } else {
                        assert!(!output.status.success(), "{source}\n{stdout}");
                        assert!(stderr.contains("domain in load"), "{source}\n{stderr}");
                        assert!(!stdout.contains("after"), "{source}\n{stdout}");
                    }
                }
            }
        }
    }
}

/// Aliases normalize through the host session; selected branch obligations and
/// independently authored choice result obligations remain cumulative.
#[test]
fn runtime_callable_choice_retains_branch_and_result_contracts() {
    for mode in ["aliases", "branches", "result"] {
        for flag in [false, true] {
            for reverse in [false, true] {
                let (left, right) = if reverse {
                    ("is_finite", "is_nan")
                } else {
                    ("is_nan", "is_finite")
                };
                let (prefix, signature) = match mode {
                    "aliases" => (String::new(), "&Pair -> Mask"),
                    "branches" => (
                        format!(
                            "left: &Pair -> tensor[*,bool] = {left}\n right: &Pair -> tensor[*,bool] = {right}\n"
                        ),
                        "&tensor[*,f64] -> Mask",
                    ),
                    _ => (String::new(), "&tensor[*,f64] -> Mask"),
                };
                let choice = if mode == "branches" {
                    "if flag then left else right".to_string()
                } else {
                    format!("if flag then {left} else {right}")
                };
                for width in [2, 3] {
                    let values = if width == 2 {
                        "1.0f64, 2.0f64"
                    } else {
                        "1.0f64, 2.0f64, 3.0f64"
                    };
                    let source = format!(
                        r#"
type Pair = tensor[2,f64]
type Mask = tensor[2,bool]
def run(flag: bool, x: tensor[*,f64]) -> tensor[*,bool] = {{
  {prefix}
  f: {signature} = {choice}
  alias = copy(copy(f))
  alias(x)
}}
result = run({flag},to_tensor([{values}]))
after = print("after")
"#
                    );
                    if width == 2 {
                        let output = evaluate(&source);
                        assert_eq!(
                            output.lines().filter(|line| *line == "after").count(),
                            1,
                            "{output}"
                        );
                    } else {
                        let dir = tempfile::tempdir().unwrap();
                        let path = dir.path().join("branch_result.ch");
                        std::fs::write(&path, &source).unwrap();
                        let check = Command::cargo_bin("chelis")
                            .unwrap()
                            .env("CHELIS_STYLE_GATE_DISABLE", "1")
                            .arg("check")
                            .arg(&path)
                            .output()
                            .unwrap();
                        assert!(check.status.success(), "{source}\n{check:?}");
                        let output = Command::cargo_bin("chelis")
                            .unwrap()
                            .env("CHELIS_STYLE_GATE_DISABLE", "1")
                            .args(["eval", "--file"])
                            .arg(path)
                            .output()
                            .unwrap();
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        assert!(!output.status.success(), "{source}\n{output:?}");
                        let op = if mode == "result" {
                            if flag != reverse {
                                "is_nan"
                            } else {
                                "is_finite"
                            }
                        } else {
                            "load"
                        };
                        assert!(
                            stderr.contains(&format!("domain in {op}")),
                            "{source}\n{stderr}"
                        );
                        assert!(!String::from_utf8_lossy(&output.stdout).contains("after"));
                    }
                }
            }
        }
    }
}

/// [04-INF-9], Deep let/block values and [05-OP-20..21]: lexical return
/// composition cannot depend on redundant metadata or an unrelated IO effect.
#[test]
fn runtime_callable_block_results_preserve_selected_identity() {
    for explicit in [false, true] {
        for nested_copy in [false, true] {
            for reverse in [false, true] {
                for flag in [false, true] {
                    let signature = if explicit {
                        ": &tensor[2,f64] -> tensor[2,bool]"
                    } else {
                        ""
                    };
                    let tail = if nested_copy {
                        "copy(copy(local))"
                    } else {
                        "local"
                    };
                    let (left, right) = if reverse {
                        ("is_finite", "is_nan")
                    } else {
                        ("is_nan", "is_finite")
                    };
                    let source = format!(
                        r#"
def run(flag: bool, x: tensor[2,f64]) -> tensor[2,bool] = {{
  f{signature} = if flag then {{ local = {left}
    {tail} }} else {{ local = {right}
    {tail} }}
  alias = copy(f)
  alias(x)
}}
result = run({flag},to_tensor([div(0.0f64,0.0f64),-0.0f64]))
"#
                    );
                    let output = evaluate(&source);
                    let expected = if flag != reverse {
                        "[true, false]"
                    } else {
                        "[false, true]"
                    };
                    assert!(
                        output.contains(&format!("result = tensor(shape=[2], data={expected})")),
                        "{source}\n{output}"
                    );
                    assert_runtime_callable_c_fence(&source);
                }
            }
        }
    }
}

/// Run a checked runtime-shape fixture with an independent mask/trap oracle.
fn assert_callable_copy_contract(
    source: &str,
    trap: Option<&str>,
    mask: &str,
    entry: Option<&str>,
) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(if source.trim_start().starts_with('(') {
        "copy_contract.dp"
    } else {
        "copy_contract.ch"
    });
    std::fs::write(&path, source).unwrap();
    let check = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    assert!(check.status.success(), "{source}\n{check:?}");
    let report: serde_json::Value = serde_json::from_slice(&check.stdout).expect("check JSON");
    assert_eq!(report["score"].as_f64(), Some(1.0), "{source}\n{report}");
    assert_eq!(
        report["errors"],
        serde_json::json!([]),
        "{source}\n{report}"
    );
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if let Some(trap) = trap {
        assert!(!output.status.success(), "{source}\n{stdout}");
        assert!(stderr.contains(trap), "{source}\n{stderr}");
        assert!(!stdout.contains("after"), "{source}\n{stdout}");
        if trap == "domain in load" {
            assert!(
                !stdout.contains("_enter"),
                "input contract precedes body IO: {source}\n{stdout}"
            );
        }
    } else {
        assert!(output.status.success(), "{source}\n{stderr}");
        assert!(
            stdout.contains(mask),
            "independent Bool mask: {source}\n{stdout}"
        );
        assert_eq!(
            stdout.lines().filter(|line| *line == "after").count(),
            1,
            "{stdout}"
        );
        if let Some(entry) = entry {
            assert_eq!(
                stdout.lines().filter(|line| *line == entry).count(),
                1,
                "{stdout}"
            );
            assert_eq!(
                stdout
                    .lines()
                    .filter(|line| line.ends_with("_enter"))
                    .count(),
                1,
                "{stdout}"
            );
        }
    }
}

/// [04-INF-9]/4.7/5.6: a Copy's own input/result claim must survive;
/// no outer signature or earlier adapter may accidentally enforce it instead.
#[test]
fn runtime_callable_copy_independent_input_and_result_contracts() {
    for input_claim in [true, false] {
        for operand in ["f", "copy(f)", "copy(copy(f))"] {
            for flag in [false, true] {
                for reverse in [false, true] {
                    let (left, right) = if reverse {
                        ("is_finite", "is_nan")
                    } else {
                        ("is_nan", "is_finite")
                    };
                    for width in [2, 3] {
                        let (input, result) = if input_claim { ("2", "*") } else { ("*", "2") };
                        let values = if width == 2 {
                            "1.0f64,2.0f64"
                        } else {
                            "1.0f64,2.0f64,3.0f64"
                        };
                        let source = format!(
                            "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {{\n f = if flag then {left} else {right}\n g: &tensor[{input}, f64] -> tensor[{result}, bool] = {operand}\n wide: &tensor[*, f64] -> tensor[*, bool] = copy(copy(g))\n wide(x)\n}}\nresult = run({flag},to_tensor([{values}]))\nafter = print(\"after\")\n"
                        );
                        let predicate = if flag { left } else { right };
                        let trap = (width == 3).then(|| {
                            if input_claim {
                                "domain in load"
                            } else if predicate == "is_nan" {
                                "domain in is_nan"
                            } else {
                                "domain in is_finite"
                            }
                        });
                        let mask = if predicate == "is_nan" {
                            "data=[false, false]"
                        } else {
                            "data=[true, true]"
                        };
                        assert_callable_copy_contract(&source, trap, mask, None);
                    }
                }
            }
        }
    }
}

/// Two independently authored axes remain cumulative through widening copies.
#[test]
fn runtime_callable_copy_rank_two_cumulative_contracts() {
    for flag in [false, true] {
        for reverse in [false, true] {
            let (left, right) = if reverse {
                ("is_finite", "is_nan")
            } else {
                ("is_nan", "is_finite")
            };
            for (rows, columns) in [(2, 3), (2, 4), (3, 3), (3, 4)] {
                let row = format!("[{}]", vec!["1.0f64"; columns].join(","));
                let values = vec![row; rows].join(",");
                let source = format!(
                    "def run(flag: bool, x: tensor[*, *, f64]) -> tensor[*, *, bool] = {{\n f: &tensor[2, *, f64] -> tensor[*, *, bool] = if flag then {left} else {right}\n g: &tensor[*, 3, f64] -> tensor[*, *, bool] = copy(f)\n wide: &tensor[*, *, f64] -> tensor[*, *, bool] = copy(copy(g))\n wide(x)\n}}\nresult = run({flag},to_tensor([{values}]))\nafter = print(\"after\")\n"
                );
                let mask = if flag != reverse {
                    "shape=[2, 3], data=[false, false, false, false, false, false]"
                } else {
                    "shape=[2, 3], data=[true, true, true, true, true, true]"
                };
                assert_callable_copy_contract(
                    &source,
                    ((rows, columns) != (2, 3)).then_some("domain in load"),
                    mask,
                    None,
                );
            }
        }
    }
}

/// An outer copied input obligation precedes selected-body IO, even when an
/// inner result obligation would eventually trap at the classifier instead.
#[test]
fn runtime_callable_copy_input_guards_before_selected_body_effects() {
    for flag in [false, true] {
        for reverse in [false, true] {
            let (left, right) = if reverse {
                ("right", "left")
            } else {
                ("left", "right")
            };
            for width in [2, 3] {
                let values = if width == 2 {
                    "1.0f64,2.0f64"
                } else {
                    "1.0f64,2.0f64,3.0f64"
                };
                let source = format!(
                    "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {{\n left = fn (z: tensor[*, f64]) -> {{\n _ = print(\"left_enter\")\n is_nan(z)\n }}\n right = fn (z: tensor[*, f64]) -> {{\n _ = print(\"right_enter\")\n is_finite(z)\n }}\n f: tensor[*, f64] -> tensor[2, bool] = if flag then {left} else {right}\n wide: tensor[*, f64] -> tensor[*, bool] = copy(f)\n g: tensor[2, f64] -> tensor[*, bool] = copy(copy(wide))\n g(x)\n}}\nresult = run({flag},to_tensor([{values}]))\nafter = print(\"after\")\n"
                );
                let selected_left = flag != reverse;
                assert_callable_copy_contract(
                    &source,
                    (width == 3).then_some("domain in load"),
                    if selected_left {
                        "data=[false, false]"
                    } else {
                        "data=[true, true]"
                    },
                    Some(if selected_left {
                        "left_enter"
                    } else {
                        "right_enter"
                    }),
                );
            }
        }
    }
}

/// Unannotated copies still preserve static identity and supported native routes.
#[test]
fn runtime_callable_copy_unannotated_static_native_control() {
    let source = "f = is_nan\ng = copy(copy(f))\nresult = g(to_tensor([1.0f64,2.0f64]))\nafter = print(\"after\")\n";
    assert_callable_copy_contract(source, None, "data=[false, false]", None);
    let evaluated = evaluate(source);
    assert_eq!(build_and_run(source, "unannotated_copy_static"), evaluated);
}

/// Produce public Deep authoring with one binding's optional Surf marker
/// independently absent, explicit, or inferred. The type syntax is unchanged.
fn callable_deep_binding_origin(
    source: &str,
    origin: Option<chelis_deep::annotations::BindingTypeOrigin>,
) -> String {
    fn rewrite(
        expr: &chelis_deep::Expr,
        origin: Option<chelis_deep::annotations::BindingTypeOrigin>,
    ) -> chelis_deep::Expr {
        let chelis_deep::Expr::Node(node, span) = expr else {
            return expr.clone();
        };
        let mut children: Vec<_> = node
            .children_slice()
            .iter()
            .map(|child| rewrite(child, origin))
            .collect();
        if node.tag() == chelis_deep::DeepTag::Bind {
            for pair in children.as_chunks_mut::<2>().0 {
                if matches!(&pair[0], chelis_deep::Expr::Atom(chelis_deep::Atom::Name(name), _) if name == "g")
                {
                    let chelis_deep::Expr::Node(value, value_span) = &pair[1] else {
                        panic!("binding initializer")
                    };
                    let mut metadata = value.meta().clone();
                    metadata.remove(chelis_deep::annotations::MetadataKey::SurfBindingType);
                    if let Some(origin) = origin {
                        metadata.replace(chelis_deep::annotations::MetadataValue::SurfBindingType(
                            chelis_deep::annotations::Spanned::new(origin, *value_span),
                        ));
                    }
                    pair[1] = chelis_deep::Expr::node(
                        value.tag(),
                        metadata,
                        value.children_slice().to_vec(),
                        *value_span,
                    );
                }
            }
        }
        chelis_deep::Expr::node(node.tag(), node.meta().clone(), children, *span)
    }
    let deep = chelis_surf::desugar::desugar_program(
        &chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture parses"),
        &chelis_syntax_host::NativeStack,
    )
    .expect("desugars");
    chelis_deep::printer::print_canonical(
        &deep
            .iter()
            .map(|expr| rewrite(expr, origin))
            .collect::<Vec<_>>(),
    )
}

fn assert_deep_callable_binding_contract(input_claim: bool) {
    use chelis_deep::annotations::BindingTypeOrigin;
    for origin in [
        None,
        Some(BindingTypeOrigin::Explicit),
        Some(BindingTypeOrigin::Inferred),
    ] {
        for operand in ["f", "copy(copy(f))"] {
            for width in [2, 3] {
                let (input, result) = if input_claim { ("2", "*") } else { ("*", "2") };
                let values = if width == 2 {
                    "1.0f64,2.0f64"
                } else {
                    "1.0f64,2.0f64,3.0f64"
                };
                let source = format!(
                    "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {{\n f = if flag then is_nan else is_finite\n g: &tensor[{input}, f64] -> tensor[{result}, bool] = {operand}\n g(x)\n}}\nresult = run(true,to_tensor([{values}]))\nafter = print(\"after\")\n"
                );
                let deep = callable_deep_binding_origin(&source, origin);
                let trap = (width == 3 && origin != Some(BindingTypeOrigin::Inferred)).then_some(
                    if input_claim {
                        "domain in load"
                    } else {
                        "domain in is_nan"
                    },
                );
                let mask = if width == 2 {
                    "data=[false, false]"
                } else {
                    "data=[false, false, false]"
                };
                assert_callable_copy_contract(&deep, trap, mask, None);
            }
        }
    }
}

/// Deep author constraints have the same entry semantics without a Surf marker.
#[test]
fn runtime_callable_deep_origin_input_contract() {
    assert_deep_callable_binding_contract(true);
}

/// Result obligations remain independent of input guards and source spelling.
#[test]
fn runtime_callable_deep_origin_result_contract() {
    assert_deep_callable_binding_contract(false);
}

/// Retained adapters carry closed operation identity through ownership lowering;
/// both branch values agree, so this requires no general first-class C ABI.
#[test]
fn runtime_callable_ascribed_copy_same_identity_native() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for operand in ["copy(f)", "copy(copy(f))"] {
            let source = format!(
                "def run(flag: bool, x: tensor[2, {dtype}]) -> tensor[2, bool] = {{\n f = if flag then is_finite else is_finite\n g: &tensor[2, {dtype}] -> tensor[2, bool] = {operand}\n g(x)\n}}\nleft = run(true,to_tensor([1.0{dtype},div(0.0{dtype},0.0{dtype})]))\nright = run(false,to_tensor([1.0{dtype},div(0.0{dtype},0.0{dtype})]))\n"
            );
            let evaluated = evaluate(&source);
            for name in ["left", "right"] {
                assert!(
                    evaluated.contains(&format!("{name} = tensor(shape=[2], data=[true, false])")),
                    "{source}\n{evaluated}"
                );
            }
            assert_eq!(
                build_and_run(&source, &format!("ascribed_same_{dtype}")),
                evaluated
            );
        }
    }
}

/// Restoring static operation identity cannot discard the adapter's guard.
#[test]
fn runtime_callable_ascribed_copy_same_identity_native_false_input() {
    for operand in ["copy(f)", "copy(copy(f))"] {
        let source = format!(
            "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {{\n f = if flag then is_finite else is_finite\n g: &tensor[2, f64] -> tensor[*, bool] = {operand}\n g(x)\n}}\nresult = run(false,to_tensor([1.0f64,2.0f64,3.0f64]))\nafter = print(\"after\")\n"
        );
        assert_callable_copy_contract(&source, Some("domain in load"), "", None);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("same_false.ch");
        let output = dir.path().join("native");
        std::fs::write(&path, &source).unwrap();
        let built = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("build")
            .arg(&path)
            .args(["--target", "c", "--output"])
            .arg(&output)
            .output()
            .unwrap();
        assert!(built.status.success(), "{source}\n{built:?}");
        let run = std::process::Command::new(output.join("same_false"))
            .output()
            .unwrap();
        assert!(!run.status.success(), "{source}\n{run:?}");
        assert!(
            String::from_utf8_lossy(&run.stderr).contains("domain in load"),
            "{source}\n{run:?}"
        );
        assert!(
            !String::from_utf8_lossy(&run.stdout).contains("after"),
            "{source}\n{run:?}"
        );
    }
}

/// A copied selected value keeps one defining contract authority at a closed join.
#[test]
fn runtime_callable_copy_shared_authority_join_native() {
    for flag in [true, false] {
        let source = format!(
            "def select(flag: bool) -> bool = {{ _ = print(\"select\")\n flag }}\ndef run(flag: bool, x: tensor[2, f64]) -> tensor[2, bool] = {{\n f: &tensor[2, f64] -> tensor[2, bool] = is_finite\n g = if select(flag) then copy(f) else copy(f)\n g(x)\n}}\nresult = run({flag},to_tensor([1.0f64,div(0.0f64,0.0f64)]))\n"
        );
        let evaluated = evaluate(&source);
        assert_eq!(evaluated.matches("select").count(), 1, "{evaluated}");
        assert!(evaluated.contains("data=[true, false]"), "{evaluated}");
        assert_eq!(build_and_run(&source, "shared_authority"), evaluated);
    }
}

/// Equal authored syntax cannot choose between independently formed authorities.
#[test]
fn runtime_callable_copy_distinct_authority_join_is_typed_unsupported() {
    let source = "def run(flag: bool, x: tensor[2, f64]) -> tensor[2, bool] = {\n left: &tensor[2, f64] -> tensor[2, bool] = is_finite\n right: &tensor[2, f64] -> tensor[2, bool] = is_finite\n g = if flag then copy(left) else copy(right)\n g(x)\n}\nresult = run(false,to_tensor([1.0f64,2.0f64]))\n";
    assert!(evaluate(source).contains("data=[true, true]"));
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("distinct_authority.ch");
    std::fs::write(&path, source).unwrap();
    let built = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(dir.path().join("out"))
        .output()
        .unwrap();
    assert!(!built.status.success(), "{built:?}");
    let error = String::from_utf8_lossy(&built.stderr);
    assert!(error.contains("chelis#879"), "{error}");
    assert!(
        !error.contains("unknown callee") && !error.contains("chelis#730"),
        "{error}"
    );
}

/// Native lowering consumes the same binding-owned Deep contract as Eval.
#[test]
fn runtime_callable_copy_deep_owner_native_guards() {
    use chelis_deep::annotations::BindingTypeOrigin;
    for origin in [
        None,
        Some(BindingTypeOrigin::Explicit),
        Some(BindingTypeOrigin::Inferred),
    ] {
        for input_claim in [true, false] {
            for width in [2, 3] {
                let (input, result) = if input_claim { ("2", "*") } else { ("*", "2") };
                let values = if width == 2 {
                    "1.0f64,2.0f64"
                } else {
                    "1.0f64,2.0f64,3.0f64"
                };
                let source = format!(
                    "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {{\n f = if flag then is_finite else is_finite\n g: &tensor[{input}, f64] -> tensor[{result}, bool] = copy(copy(f))\n g(x)\n}}\nresult = run(false,to_tensor([{values}]))\nafter = print(\"after\")\n"
                );
                let deep = callable_deep_binding_origin(&source, origin);
                let trap = (width == 3 && origin != Some(BindingTypeOrigin::Inferred)).then_some(
                    if input_claim {
                        "domain in load"
                    } else {
                        "domain in is_finite"
                    },
                );
                let mask = if width == 2 {
                    "data=[true, true]"
                } else {
                    "data=[true, true, true]"
                };
                assert_callable_copy_contract(&deep, trap, mask, None);
                let dir = tempfile::tempdir().unwrap();
                let path = dir.path().join("deep_owner.dp");
                std::fs::write(&path, &deep).unwrap();
                let output = dir.path().join("out");
                let built = Command::cargo_bin("chelis")
                    .unwrap()
                    .env("CHELIS_STYLE_GATE_DISABLE", "1")
                    .arg("build")
                    .arg(&path)
                    .args(["--target", "c", "--output"])
                    .arg(&output)
                    .output()
                    .unwrap();
                assert!(
                    built.status.success(),
                    "{origin:?} input={input_claim} width={width}\n{built:?}"
                );
                let run = std::process::Command::new(output.join("deep_owner"))
                    .output()
                    .unwrap();
                let stdout = String::from_utf8_lossy(&run.stdout);
                let stderr = String::from_utf8_lossy(&run.stderr);
                if let Some(trap) = trap {
                    assert!(
                        !run.status.success(),
                        "{origin:?} input={input_claim} width={width}\n{run:?}"
                    );
                    assert!(stderr.contains(trap), "{stderr}");
                    assert!(!stdout.contains("after"), "{stdout}");
                } else {
                    assert!(run.status.success(), "{run:?}");
                    assert!(stdout.contains(mask), "{stdout}");
                    assert_eq!(
                        stdout.lines().filter(|line| *line == "after").count(),
                        1,
                        "{stdout}"
                    );
                }
            }
        }
    }
}

/// [04-INF-9]: numeric and host callable consumers share checked source ownership.
#[test]
fn numeric_callable_deep_bindings_preserve_entry_and_result_contracts() {
    use chelis_deep::annotations::BindingTypeOrigin;
    for origin in [
        None,
        Some(BindingTypeOrigin::Explicit),
        Some(BindingTypeOrigin::Inferred),
    ] {
        for input_claim in [true, false] {
            for producer in ["is_finite", "copy(copy(is_finite))"] {
                for width in [2, 3] {
                    let (input, result) = if input_claim { ("2", "*") } else { ("*", "2") };
                    let values = vec!["1.0f64"; width].join(",");
                    let source = format!(
                        "def run(x: tensor[*,f64]) -> tensor[*,bool] = {{\n g: &tensor[{input},f64] -> tensor[{result},bool] = {producer}\n g(x)\n}}\nresult = run(to_tensor([{values}]))\nafter = print(\"after\")\n"
                    );
                    let deep = callable_deep_binding_origin(&source, origin);
                    let trap = (width == 3 && origin != Some(BindingTypeOrigin::Inferred))
                        .then_some(if input_claim {
                            "domain in load"
                        } else {
                            "domain in is_finite"
                        });
                    let mask = if width == 2 {
                        "data=[true, true]"
                    } else {
                        "data=[true, true, true]"
                    };
                    assert_callable_copy_contract(&deep, trap, mask, None);
                    let dir = tempfile::tempdir().unwrap();
                    let path = dir.path().join("numeric_deep_owner.dp");
                    std::fs::write(&path, &deep).unwrap();
                    let output = dir.path().join("out");
                    let built = Command::cargo_bin("chelis")
                        .unwrap()
                        .env("CHELIS_STYLE_GATE_DISABLE", "1")
                        .arg("build")
                        .arg(&path)
                        .args(["--target", "c", "--output"])
                        .arg(&output)
                        .output()
                        .unwrap();
                    assert!(
                        built.status.success(),
                        "{origin:?}, {producer}, input={input_claim}, width={width}: {built:?}"
                    );
                    let native = std::process::Command::new(output.join("numeric_deep_owner"))
                        .output()
                        .unwrap();
                    let stdout = String::from_utf8_lossy(&native.stdout);
                    let stderr = String::from_utf8_lossy(&native.stderr);
                    if let Some(trap) = trap {
                        assert!(
                            !native.status.success(),
                            "{origin:?}, {producer}, input={input_claim}, width={width}: {native:?}"
                        );
                        assert!(stderr.contains(trap), "{stderr}");
                        assert!(!stdout.contains("after"), "{stdout}");
                    } else {
                        assert!(native.status.success(), "{native:?}");
                        assert!(stdout.contains(mask), "{stdout}");
                        assert_eq!(
                            stdout.lines().filter(|line| *line == "after").count(),
                            1,
                            "{stdout}"
                        );
                    }
                }
            }
        }
    }
}

fn assert_copied_host_contract_native(source: &str, trap: bool, expected: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("copied_host_contract.ch");
    std::fs::write(&path, source).unwrap();
    let checked = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .output()
        .unwrap();
    assert!(checked.status.success(), "{source}\n{checked:?}");
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"].as_f64(), Some(1.0), "{report}");
    assert_eq!(report["errors"], serde_json::json!([]), "{report}");
    let evaluated = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file"])
        .arg(&path)
        .output()
        .unwrap();
    let output = dir.path().join("out");
    let built = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&output)
        .output()
        .unwrap();
    assert!(built.status.success(), "{source}\n{built:?}");
    let native = std::process::Command::new(output.join("copied_host_contract"))
        .output()
        .unwrap();
    for result in [&evaluated, &native] {
        let stdout = String::from_utf8_lossy(&result.stdout);
        let stderr = String::from_utf8_lossy(&result.stderr);
        if trap {
            assert!(!result.status.success(), "{source}\n{result:?}");
            assert!(stderr.contains("numeric trap: domain"), "{stderr}");
            if stderr.contains("input `arg0[0]`") {
                assert!(
                    stderr.contains("input `arg0[0]` axis 0 expected 2, got 3"),
                    "{stderr}"
                );
                assert!(stderr.contains("domain in load at i64"), "{stderr}");
            } else {
                assert!(stderr.contains("extent `2`"), "{stderr}");
            }
            assert!(!stdout.contains("after-copied-host"), "{stdout}");
        } else {
            assert!(result.status.success(), "{source}\n{result:?}");
            assert!(stdout.contains(expected), "{stdout}");
            assert_eq!(
                stdout
                    .lines()
                    .filter(|line| *line == "after-copied-host")
                    .count(),
                1,
                "{stdout}"
            );
        }
    }
    if !trap {
        assert_eq!(native.stdout, evaluated.stdout);
    }
}

/// Host-only Copy inputs retain nested tensor entry obligations.
#[test]
fn runtime_callable_copy_host_nested_entry_native() {
    for operand in ["copy(f)", "copy(copy(f))"] {
        for width in [2, 3] {
            let elements = vec!["1.0f32"; width].join(",");
            let source = format!(
                "def run(x: List[tensor[*, f32]]) -> i64 = {{\n f: &List[tensor[2, f32]] -> i64 = len\n g = {operand}\n g(x)\n}}\nresult = run([to_tensor([{elements}])])\nafter = print(\"after-copied-host\")\n"
            );
            assert_copied_host_contract_native(&source, width == 3, "result = 1");
        }
    }
}

/// Copied projecting builtins retain direct and nested result obligations.
#[test]
fn runtime_callable_copy_host_nested_result_native() {
    for nested in [false, true] {
        let (input, result) = if nested {
            ("List[List[tensor[*, f32]]]", "List[tensor[2, f32]]")
        } else {
            ("List[tensor[*, f32]]", "tensor[2, f32]")
        };
        let outer_result = if nested {
            "List[tensor[*, f32]]"
        } else {
            "tensor[*, f32]"
        };
        for width in [2, 3] {
            let elements = vec!["1.0f32"; width].join(",");
            let value = if nested {
                format!("[[to_tensor([{elements}])]]")
            } else {
                format!("[to_tensor([{elements}])]")
            };
            let source = format!(
                "def run(x: {input}) -> {outer_result} = {{\n f: &{input} -> i64 -> {result} = index\n g = copy(copy(f))\n g(x, 0i64)\n}}\nresult = run({value})\nafter = print(\"after-copied-host\")\n"
            );
            assert_copied_host_contract_native(&source, width == 3, "shape=[2]");
        }
    }
}

/// Forming/copying/forwarding an uninvoked callback creates no observation.
#[test]
fn runtime_callable_copy_host_uninvoked_contracts_native() {
    for (operation, signature) in [
        ("len", "&List[tensor[2, f32]] -> i64"),
        ("index", "&List[tensor[*, f32]] -> i64 -> tensor[2, f32]"),
    ] {
        let source = format!(
            "def keep(h: ({signature}), x: List[tensor[*, f32]]) -> i64 = len(x)\ndef run(x: List[tensor[*, f32]]) -> i64 = {{\n f: {signature} = {operation}\n g = copy(copy(f))\n keep(g, x)\n}}\nresult = run([to_tensor([1.0f32, 2.0f32, 3.0f32])])\nafter = print(\"after-copied-host\")\n"
        );
        assert_copied_host_contract_native(&source, false, "result = 1");
    }
}
