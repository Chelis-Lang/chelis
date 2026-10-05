mod support;

use chelis_prove::property_runner::{
    PropertyRunOptions, PropertyRunResult, PropertyStatus, PropertyTier,
    run_deep_source_properties, run_surf_source_properties,
};

#[test]
fn deep_explicit_beacon_is_terminal_and_never_samples() {
    crate::support::isolate();
    let source = r#"(module {} m
      (defsig {} always_true (t-fn {} (t-prim {} f32) (t-prim {} bool)))
      (def {chelis_role: "property", property_preconditions: (tuple {}),
        property_quantifiers: (params {} (x {type: (t-prim {} f32)})),
        property_source_kind: "user"} always_true
        (fn {} (params {} (x {type: (t-prim {} f32)}))
          (app {} (var {} gte) (var {} x) (var {} x)))))"#;
    let options = PropertyRunOptions {
        tier: "beacon-only".into(),
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let PropertyRunResult::Ran(outcomes) = run_deep_source_properties(source, &options).unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert_eq!(outcomes[0].proof_tier, PropertyTier::Beacon);
    assert_eq!(outcomes[0].samples, 0);
    assert!(outcomes[0].reason.as_deref().unwrap().contains("Deep"));
}

#[test]
fn beacon_tier_rejects_missing_box_bounds_without_smt_or_fuzz_fallback() {
    crate::support::isolate();
    let options = PropertyRunOptions {
        tier: "beacon-only".into(),
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let PropertyRunResult::Ran(outcomes) = run_surf_source_properties(
        "@property bounded forall(x: tensor[f64]) where tensor_to_scalar(x) <= 1.0f64:\n  tensor_to_scalar(x) <= 1.0f64\n", &options).unwrap();
    assert_eq!(outcomes.len(), 1);
    assert_eq!(outcomes[0].status, PropertyStatus::Unsupported);
    assert_eq!(outcomes[0].samples, 0);
    assert!(outcomes[0].reason.as_deref().unwrap().contains("bound"));
}

#[test]
fn beacon_scalar_bridge_requires_explicit_f64_bounds_and_unshadowed_identity() {
    crate::support::isolate();
    let options = PropertyRunOptions {
        tier: "beacon-only".into(),
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    for (prefix, parameter, upper, reason) in [
        ("", "x", "1", "explicit f64"),
        ("", "tensor_to_scalar", "1.0f64", "shadowed"),
        (
            "sig tensor_to_scalar: tensor[f64] -> f64\n",
            "x",
            "1.0f64",
            "shadowed",
        ),
        ("import Suspicious\n", "x", "1.0f64", "imports"),
    ] {
        let source = format!(
            "{prefix}@property bounded forall({parameter}: tensor[f64]) where tensor_to_scalar({parameter}) >= -1.0f64, tensor_to_scalar({parameter}) <= 1.0f64:\n  tensor_to_scalar({parameter}) <= {upper}\n"
        );
        let PropertyRunResult::Ran(outcomes) =
            run_surf_source_properties(&source, &options).unwrap();
        assert_eq!(
            outcomes[0].status,
            PropertyStatus::Unsupported,
            "{outcomes:?}"
        );
        assert!(
            outcomes[0].reason.as_deref().unwrap().contains(reason),
            "{outcomes:?}"
        );
        assert!(outcomes[0].engine_evidence.is_none());
    }
}

#[test]
#[ignore = "requires current released CHELIS_BEACON_BIN with scalar v11 and Arb"]
fn beacon_property_route_reaches_real_engine_and_preserves_all_outcomes() {
    crate::support::isolate();
    assert!(std::env::var_os("CHELIS_BEACON_BIN").is_some());
    for (upper, budget, expected) in [
        (
            2.0,
            std::time::Duration::from_secs(2),
            PropertyStatus::Passed,
        ),
        (
            -2.0,
            std::time::Duration::from_secs(2),
            PropertyStatus::Failed,
        ),
        (2.0, std::time::Duration::ZERO, PropertyStatus::Unsupported),
    ] {
        let source = format!(
            "def neuron(x: tensor[f64]) -> tensor[f64] = relu(x)\n@property bounded forall(x: tensor[f64]) where tensor_to_scalar(x) >= -1.0f64, tensor_to_scalar(x) <= 1.0f64:\n  tensor_to_scalar(neuron(x)) <= {upper:.1}f64\n"
        );
        let options = PropertyRunOptions {
            tier: "beacon-only".into(),
            beacon_budget: budget,
            ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
        };
        let PropertyRunResult::Ran(outcomes) =
            run_surf_source_properties(&source, &options).unwrap();
        let result = &outcomes[0];
        assert_eq!(result.status, expected, "{result:?}");
        let evidence = result.engine_evidence.as_ref().unwrap();
        assert!(evidence["beacon_evidence"]["tree"].is_array());
        assert!(
            evidence["folded_goal"]
                .as_str()
                .unwrap()
                .contains("neuron(x)")
        );
        assert_eq!(evidence["input_bindings"]["beacon_input_0"], "x");
        if expected == PropertyStatus::Passed {
            assert_eq!(
                result.composite_verdict.as_str(),
                "proven_modulo_real_arithmetic"
            );
        }
    }
}

#[test]
#[ignore = "requires released CHELIS_BEACON_BIN"]
fn expired_outer_deadline_returns_engine_unknown_after_compiler_preparation() {
    crate::support::isolate();
    let source = "@property bounded forall(x: tensor[f64]) where tensor_to_scalar(x) >= -1.0f64, tensor_to_scalar(x) <= 1.0f64:\n  tensor_to_scalar(x) <= 2.0f64\n";
    let options = PropertyRunOptions {
        tier: "beacon-only".into(),
        beacon_deadline: Some(std::time::Instant::now()),
        ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
    };
    let PropertyRunResult::Ran(outcomes) = run_surf_source_properties(source, &options).unwrap();
    assert_eq!(
        outcomes[0].status,
        PropertyStatus::Unsupported,
        "{outcomes:?}"
    );
    let evidence = outcomes[0].engine_evidence.as_ref().unwrap();
    assert_eq!(evidence["beacon_evidence"]["reason"], "budget_exhausted");
    assert!(evidence["beacon_evidence"]["tree"].is_array());
    assert_eq!(evidence["search_options"]["budget_ms"], 0);
}

#[test]
#[ignore = "requires released CHELIS_BEACON_BIN"]
fn named_boxes_keep_parameter_order_and_exclude_unrelated_function_loads() {
    crate::support::isolate();
    let source = "def unrelated(z: tensor[f64]) -> tensor[f64] = relu(z)\ndef combine(b: tensor[f64], a: tensor[f64]) -> tensor[f64] = b + scalar_to_tensor(2.0f64) * a\n@property bounded forall(b: tensor[f64], a: tensor[f64]) where tensor_to_scalar(b) >= 1.0f64, tensor_to_scalar(b) <= 2.0f64, tensor_to_scalar(a) >= -4.0f64, tensor_to_scalar(a) <= -3.0f64:\n  tensor_to_scalar(combine(b, a)) <= -3.0f64\n";
    for (source, expected) in [
        (source.to_string(), PropertyStatus::Passed),
        (
            source.replace("<= -3.0f64\n", "<= -8.0f64\n"),
            PropertyStatus::Failed,
        ),
    ] {
        let options = PropertyRunOptions {
            tier: "beacon-only".into(),
            ..PropertyRunOptions::new(&chelis_std_bundle::EMBEDDED_RUNTIME)
        };
        let PropertyRunResult::Ran(outcomes) =
            run_surf_source_properties(&source, &options).unwrap();
        let result = &outcomes[0];
        assert_eq!(result.status, expected, "{result:?}");
        let evidence = result.engine_evidence.as_ref().unwrap();
        assert_eq!(evidence["input_bindings"]["beacon_input_0"], "b");
        assert_eq!(evidence["input_bindings"]["beacon_input_1"], "a");
        assert_eq!(evidence["input_box"]["beacon_input_0"]["lo"], 1.0);
        assert_eq!(evidence["input_box"]["beacon_input_1"]["lo"], -4.0);
        assert_eq!(evidence["input_box"].as_object().unwrap().len(), 2);
    }
}
