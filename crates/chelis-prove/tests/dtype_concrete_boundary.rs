//! Structural guards for the chelis#729 exact-typed prover boundary.

mod support;

fn function_slice<'a>(source: &'a str, start: &str, next: &str) -> &'a str {
    let start = source.find(start).expect("guarded function exists");
    let tail = &source[start..];
    let end = tail.find(next).unwrap_or(tail.len());
    &tail[..end]
}

#[test]
fn concrete_evaluator_and_tier_c_have_no_bare_f64_environment() {
    crate::support::isolate();
    let concrete = include_str!("../src/concrete_eval.rs");
    assert!(
        concrete.contains("pub type ConcreteEnv = UnordMap<String, ScalarValue>"),
        "the concrete evaluator environment must carry sealed dtype values"
    );
    assert!(
        !concrete.contains("UnordMap<String, f64>"),
        "a bare f64 environment reopens the int64-collapse class"
    );

    let tier_c = include_str!("../src/tier_c.rs");
    assert!(
        tier_c.contains("let env: ConcreteEnv"),
        "Tier C must construct the typed environment explicitly"
    );
    let integer_arm = function_slice(tier_c, "SmtSort::Int =>", "SmtSort::Real =>");
    assert!(
        integer_arm.contains("scalar_from_i64") && !integer_arm.contains("next_f64"),
        "integer fuzz variables must be sampled as integers"
    );
}

#[test]
fn produced_value_flatteners_have_no_lossy_numeric_fallback() {
    crate::support::isolate();
    let obligation = include_str!("../src/obligation_engine.rs");
    let flatten = function_slice(
        obligation,
        "fn flatten_field_value(",
        "fn validate_produced_env(",
    );
    for forbidden in [
        "as f64",
        "to_f64_lossy_vec",
        "element_as_f64_lossy",
        "element_f64_lossy",
    ] {
        assert!(
            !flatten.contains(forbidden),
            "obligation produced-value flattener contains lossy path `{forbidden}`"
        );
    }
    assert!(
        flatten.contains("tensor_element_scalar"),
        "tensor fields must cross through the exhaustive typed wire reader"
    );

    let opaque = include_str!("../src/opaque.rs");
    let read = function_slice(
        opaque,
        "fn read_produced_field(",
        "fn sample_raw_input_expr(",
    );
    for forbidden in [
        "as f64",
        "to_f64_lossy_vec",
        "element_as_f64_lossy",
        "element_f64_lossy",
    ] {
        assert!(
            !read.contains(forbidden),
            "opaque producer reader contains lossy path `{forbidden}`"
        );
    }
    assert!(
        read.contains("tensor_element_scalar"),
        "opaque tensor fields must cross through the exhaustive typed wire reader"
    );

    for variant in ["ExecutionValue::Scalar", "ExecutionValue::Bool"] {
        assert!(
            flatten.contains(variant),
            "obligation flattener must classify {variant}"
        );
        assert!(
            read.contains(variant),
            "opaque reader must classify {variant}"
        );
    }
    for reader in [flatten, read] {
        assert!(
            reader.contains("value.get().prim() == prim"),
            "scalar wire admission must require exact dtype equality"
        );
    }
    for forbidden in ["if prim.is_float()", "if prim.is_integer()"] {
        assert!(
            !flatten.contains(forbidden),
            "obligation flattener must reject tagged dtype substitution: `{forbidden}`"
        );
        assert!(
            !read.contains(forbidden),
            "opaque producer reader must reject tagged dtype substitution: `{forbidden}`"
        );
    }
}

#[test]
fn every_prover_fuzz_tensor_builder_uses_typed_values() {
    crate::support::isolate();
    let obligation = include_str!("../src/obligation_engine.rs");
    let tensor_arg = function_slice(
        obligation,
        "fn obligation_tensor_arg(",
        "fn tensor_values_from_json(",
    );
    assert!(tensor_arg.contains("&[ScalarValue]"));
    assert!(tensor_arg.contains("tensor_value_expr_typed"));
    assert!(!tensor_arg.contains("&[f64]"));

    let opaque = include_str!("../src/opaque.rs");
    let producer_sample = function_slice(
        opaque,
        "fn sample_raw_input_expr(",
        "fn classify_pred_shape(",
    );
    assert!(producer_sample.contains("scalar_from_i64"));
    assert!(producer_sample.contains("tensor_value_expr_typed"));
    assert!(!producer_sample.contains("Vec<f64>"));

    let injection = include_str!("../src/property_runner/injection.rs");
    let injection_loop = function_slice(
        injection,
        "pub(super) fn prove_with_injection(",
        "fn max_injection_attempts(",
    );
    assert!(injection_loop.contains("tensor_value_expr_typed"));
    assert!(!injection_loop.contains("Vec<f64>"));
    let injection_sample = function_slice(injection, "fn sample_scalar(", "fn rng_f64(");
    assert!(injection_sample.contains("-> ScalarValue"));
    assert!(injection_sample.contains("scalar_from_i64"));
}
