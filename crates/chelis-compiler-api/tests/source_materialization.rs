//! spec/10 §3.3 and [03-META-1/2/3]: transport is not source admission.
//!
//! These tests deliberately use the same observation driver as the C6 probe;
//! the Python case builder separately owns its expected syntax and values.
#[path = "../examples/support/wire_materialization.rs"]
mod materialization;

use chelis_compiler_api::{
    compiler,
    schema::{CheckRequest, EvalRequest, SourceKind},
};
use chelis_types::errors::CheckErrorKind;
use chelis_vocab::DiagnosticKind;
use serde_json::json;

#[test]
fn source_i64_and_finite_f64_materialize_without_a_lossy_lexical_hop() {
    for (token, expected) in [
        (
            "9007199254740993i64",
            json!({"dtype":"int64","value":9007199254740993_i64}),
        ),
        ("-0.0f64", json!({"dtype":"f64","bits":"8000000000000000"})),
        (
            "1.0000000000000002f64",
            json!({"dtype":"f64","bits":"3ff0000000000001"}),
        ),
        (
            "4611686293305294849f32",
            json!({"dtype":"f32","bits":"5e800001"}),
        ),
    ] {
        let source = format!("(def {{}} value (lit {{}} {token}))");
        assert_eq!(
            materialization::observe("SourceProgram", "eval-deep", &source).unwrap(),
            json!({"roots":[{"name":"value","value":{"type":"scalar","value":expected}}]})
        );
    }
}

#[test]
fn deep_source_float_grammar_rejects_overflow_before_every_consumer() {
    // The same text goes through the real public APIs, not a DTO decoder.
    for token in [
        "1.7976931348623157e308f64",
        "-1.7976931348623157e308f64",
        "-0.0f64",
    ] {
        let source = format!("(def {{}} value (lit {{}} {token}))");
        for codec in ["parse-deep", "check-deep", "eval-deep"] {
            assert!(
                materialization::observe("SourceProgram", codec, &source).is_ok(),
                "{codec}: {source}"
            );
        }
    }
    for token in ["1e999", "-1e999", "1e999f64", "-1e999f32"] {
        let source = format!("(def {{}} value (lit {{}} {token}))");
        for codec in [
            "parse-deep",
            "check-deep",
            "eval-deep",
            "validate-deep",
            "decompile-deep",
        ] {
            assert!(
                materialization::observe("SourceProgram", codec, &source).is_err(),
                "{codec}: {source}"
            );
        }
    }
}

#[test]
fn execution_ieee_nonfinite_values_keep_their_runtime_transport() {
    for bits in ["7ff0000000000000", "fff0000000000000", "7ff8000000000001"] {
        let json = json!({"type":"scalar", "value":{"dtype":"f64", "bits":bits}});
        let value: chelis_compiler_api::schema::ExecutionValue =
            serde_json::from_value(json.clone()).unwrap();
        assert_eq!(serde_json::to_value(value).unwrap(), json);
    }
}

#[test]
fn dto_suffixes_do_not_authorize_source_or_contradictory_literal_origins() {
    let dto = r#"{"kind":"typed_float","value":1.0,"suffix":"future"}"#;
    assert!(materialization::observe("WireLiteral", "json", dto).is_ok());
    for (codec, source) in [
        ("parse-surf", "value = 1.0future"),
        (
            "check-deep",
            "(def {} value (lit {type: (t-prim {} f64)} 1))",
        ),
        (
            "check-deep",
            "(def {} value (lit {type: (t-prim {} i64), literal_source: integer} 1))",
        ),
        (
            "check-deep",
            "(def {} value (lit {type: (t-prim {} f64), literal_source: integer} 1.0))",
        ),
    ] {
        assert!(
            materialization::observe("SourceProgram", codec, source).is_err(),
            "{codec}: {source}"
        );
    }
}

#[test]
fn preserved_history_is_readmitted_before_becoming_a_live_annotation() {
    let good = "(def {source: (macro_name (lit {} 7))} value (lit {} 1))";
    assert!(materialization::observe("SourceHistory", "live-annotation", good).is_ok());
    for payload in [
        "bare_name",
        "(lit {type: false} 1)",
        "(lit {type: (t-prim {} i32), type: (t-prim {} i32)} 1)",
        "(var {surf_path: \"X\"} x)",
    ] {
        let source = format!("(def {{source: (macro_name {payload})}} value (lit {{}} 1))");
        assert!(
            materialization::observe("SourceHistory", "transport", &source).is_ok(),
            "{source}"
        );
        assert!(
            materialization::observe("SourceHistory", "live-annotation", &source).is_err(),
            "{source}"
        );
    }
}

#[test]
fn opaque_extension_data_never_stamps_as_a_runtime_expression() {
    for source in ["(lit {} 7)", "(future {type: false, type: 1} bare)"] {
        assert!(materialization::observe("ExtensionData", "data", source).is_ok());
        assert!(materialization::observe("ExtensionData", "runtime", source).is_err());
    }
    assert!(materialization::observe("ExtensionData", "data", "(unclosed").is_err());
    assert!(materialization::observe("RawSourceAdmission", "runtime", "(lit {} 7)").is_ok());
    assert!(materialization::observe("RawSourceAdmission", "runtime", "bare_name").is_err());
}

#[test]
fn live_annotation_shape_uniqueness_and_placement_reject_at_program_ingress() {
    let good = "(def {property_seed: (lit {} 1)} value (lit {} 7))";
    assert!(materialization::observe("SourceProgram", "parse-deep", good).is_ok());
    for source in [
        "(def {property_seed: bare_name} value (lit {} 7))",
        "(def {property_seed: (lit {span: 1} 1)} value (lit {} 7))",
        "(def {property_seed: (lit {} 1), property_seed: (lit {} 1)} value (lit {} 7))",
        "(def {custom: 1, custom: 2} value (lit {} 7))",
        "(def {} value (var {property_seed: (lit {} 1)} x))",
    ] {
        let error = materialization::observe("SourceProgram", "parse-deep", source).unwrap_err();
        assert!(error.contains("parse"), "{error}");
    }
}

#[test]
fn registered_def_expression_metadata_has_check_and_eval_admission_parity() {
    // spec/03 [03-META-2] assigns these three keys the expression role.
    // [03-ROLE-2] and spec/04 [04-TOT-1/4] therefore require the same
    // semantic admission as an ordinary runtime-expression child: a valid
    // expression reaches both consumers, while a present unknown form is
    // diagnosed rather than skipped with the enclosing def's body.
    for key in ["property_seed", "property_samples", "property_tolerance"] {
        let good = format!("(def {{{key}: (lit {{}} 1)}} value (lit {{}} 7))");
        assert_eq!(
            materialization::observe("SourceProgram", "check-deep", &good).unwrap(),
            json!({"admitted": true}),
            "check-deep must admit the registered {key} expression role"
        );
        assert_eq!(
            materialization::observe("SourceProgram", "eval-deep", &good).unwrap(),
            json!({"roots":[{"name":"value","value":{"type":"scalar","value":{"dtype":"int32","value":7}}}]}),
            "eval-deep must admit the registered {key} expression role"
        );

        let bad = format!("(def {{{key}: (future_form {{}} 1)}} value (lit {{}} 7))");
        for codec in ["check-deep", "eval-deep"] {
            let error = materialization::observe("SourceProgram", codec, &bad)
                .expect_err("an unknown form in expression metadata must be rejected");
            assert!(
                error.contains("future_form"),
                "{codec} must identify the rejected {key} expression: {error}"
            );
        }

        let nested =
            format!("(def {{{key}: (tuple {{}} (future_nested {{}} 1))}} value (lit {{}} 7))");
        let stamped = chelis_deep::parse_and_stamp_file(&nested)
            .expect("the lenient Deep parser must preserve the nested unknown form");
        let checker_errors = chelis_types::check_typed_program(&stamped)
            .expect_err("a nested unknown metadata expression must fail checking")
            .errors;
        assert!(
            checker_errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::UnknownForm)
                    && error.message.contains("future_nested")
            }),
            "the checker must classify nested {key} metadata as UnknownForm: {checker_errors:?}"
        );

        let checked = compiler::check(CheckRequest {
            source_kind: SourceKind::Deep,
            source: nested.clone(),
        })
        .expect("check must return its structured diagnostic report");
        assert!(
            checked.errors.iter().any(|error| {
                error.kind() == DiagnosticKind::UnknownForm
                    && error.message.contains("future_nested")
            }),
            "check must preserve the nested {key} UnknownForm diagnostic: {:?}",
            checked.errors
        );

        let eval_error = compiler::eval(EvalRequest {
            source_kind: SourceKind::Deep,
            source: nested,
            bindings: Default::default(),
        })
        .expect_err("eval must reject a nested unknown metadata expression");
        assert!(
            eval_error.errors.iter().any(|error| {
                error.kind() == DiagnosticKind::UnknownForm
                    && error.message.contains("future_nested")
            }),
            "eval must preserve the nested {key} UnknownForm diagnostic: {:?}",
            eval_error.errors
        );
    }
}

#[test]
fn preconditions_and_invariant_bodies_are_live_expression_positions() {
    for (good, bad) in [
        (
            "(def {property_preconditions: (tuple {} (lit {} true))} value (lit {} 7))",
            "(def {property_preconditions: (tuple {} bare_name)} value (lit {} 7))",
        ),
        (
            "(deftype {opaque: true, invariant: (fn {} (params {} x) (lit {} true))} T () (variant {} T))",
            "(deftype {opaque: true, invariant: (fn {} (params {} x) bare_name)} T () (variant {} T))",
        ),
    ] {
        assert!(materialization::observe("SourceProgram", "parse-deep", good).is_ok());
        assert!(materialization::observe("SourceProgram", "parse-deep", bad).is_err());
    }
}

#[test]
fn transported_source_indices_do_not_bypass_tuple_or_axis_admission() {
    assert!(materialization::observe("SourceProgram", "check-surf", "value = (7, 8).0").is_ok());
    assert!(materialization::observe("SourceProgram", "check-surf", "value = (7, 8).2").is_err());
    let function = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x\n";
    for axis in ["", ", axis=1"] {
        let source = format!("{function}value = vmap(f{axis})");
        assert!(materialization::observe("SourceProgram", "check-surf", &source).is_ok());
    }
    let wide = format!("{function}value = vmap(f, axis=9007199254740993)");
    assert!(materialization::observe("SourceProgram", "check-surf", &wide).is_err());
}
