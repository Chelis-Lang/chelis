//! [05-OP-20..22]: float classification preserves surface and stored width.
mod common;
use assert_cmd::Command;
use common::build_and_run;

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
