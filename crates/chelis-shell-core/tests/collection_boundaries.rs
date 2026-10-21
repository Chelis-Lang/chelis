mod support;

use chelis_shell_core::{
    CollectionObligation, CollectionValidationError, ShellValidationError,
    validate_collection_obligation_ledger, validate_collection_obligations_for_type_repr,
    validate_shell,
};
use chelis_syntax_host::NativeStack;
use support::{
    FUNCTION, I64, SHAPED_FUNCTION, T0, T1, TENSOR, all_obligations, carried_types_mut, shell,
    symbol,
};

#[test]
fn all_collection_variants_preserve_builtin_names_carried_types_and_valid_membership() {
    let obligations = all_obligations();
    let expected = [
        ("append", vec![T0, T1, T0]),
        ("concat", vec![T0, T1, T0]),
        ("fold_in", vec![T0, I64, T1]),
        ("index", vec![T0, I64, T1]),
        ("is_finite", vec![T0, T1]),
        ("is_infinite", vec![T0, T1]),
        ("is_nan", vec![T0, T1]),
        ("key_from_seed", vec![T0, T1]),
        ("len", vec![T0, T1]),
        ("split_key", vec![T0, T1]),
        ("split_keys", vec![T0, I64, T1]),
    ];
    for (obligation, (builtin, types)) in obligations.iter().zip(expected) {
        assert_eq!(obligation.builtin(), builtin);
        assert_eq!(obligation.types(), types);
    }
    validate_collection_obligation_ledger(&obligations, &NativeStack)
        .expect("canonical full ledger");
    validate_collection_obligations_for_type_repr(Some(FUNCTION), &obligations, &NativeStack)
        .expect("all collection variants use declared variables");
    let mut package = shell();
    symbol(&mut package).collection_obligations = obligations;
    validate_shell(&package, &NativeStack).expect("full ledger in package");
}

#[test]
fn collection_variants_preserve_serde_tags_and_reject_unknown_fields() {
    for obligation in all_obligations() {
        let value = serde_json::to_value(&obligation).expect("serialize collection relation");
        let mut map = value
            .as_object()
            .expect("externally tagged relation")
            .clone();
        assert_eq!(map.len(), 1);
        let tag = map.keys().next().expect("relation tag").clone();
        let expected_tag = match obligation.builtin() {
            "is_finite" | "is_infinite" | "is_nan" => "float_classify",
            builtin => builtin,
        };
        assert_eq!(tag, expected_tag);
        let restored: CollectionObligation =
            serde_json::from_value(value).expect("restore relation");
        assert_eq!(restored, obligation);
        map.get_mut(&tag)
            .expect("relation payload")
            .as_object_mut()
            .expect("named fields")
            .insert("unknown".into(), serde_json::Value::Null);
        assert!(
            serde_json::from_value::<CollectionObligation>(serde_json::Value::Object(map)).is_err()
        );
    }
}

#[test]
fn empty_collection_ledgers_do_not_require_or_parse_a_function_type() {
    validate_collection_obligation_ledger(&[], &NativeStack).expect("empty ledger");
    for source in [None, Some(""), Some("not Deep"), Some("(t-prim {} f32)")] {
        validate_collection_obligations_for_type_repr(source, &[], &NativeStack)
            .expect("empty ledger has no function requirement");
    }
}

#[test]
fn collection_contracts_require_a_valid_function_root() {
    let obligations = vec![CollectionObligation::Len {
        operand: T0.into(),
        result: T1.into(),
    }];
    for source in [None, Some(T0), Some("(t-prim {} f32)")] {
        let error =
            validate_collection_obligations_for_type_repr(source, &obligations, &NativeStack)
                .expect_err("function root is mandatory");
        assert!(matches!(
            error,
            CollectionValidationError::MissingFunctionType
        ));
        assert_eq!(error.to_string(), "require a function type representation");
    }
    let error =
        validate_collection_obligations_for_type_repr(Some("(t-fn {}"), &obligations, &NativeStack)
            .expect_err("malformed function");
    assert!(matches!(
        error,
        CollectionValidationError::InvalidFunctionDeep { .. }
    ));
    assert!(
        error
            .to_string()
            .starts_with("require a function type representation that is valid Deep:")
    );
}

#[test]
fn every_collection_field_rejects_malformed_and_noncanonical_deep() {
    for original in all_obligations() {
        for slot in 0..original.types().len() {
            for malformed in ["not Deep", "(t-var {}", "(t-unit {}) (t-unit {})"] {
                let mut obligation = original.clone();
                *carried_types_mut(&mut obligation)[slot] = malformed.into();
                let error = validate_collection_obligation_ledger(&[obligation], &NativeStack)
                    .expect_err("malformed carried type");
                assert!(matches!(
                    error,
                    CollectionValidationError::InvalidCarriedDeep { .. }
                ));
                assert!(
                    error
                        .to_string()
                        .starts_with("contain a type representation that is not valid Deep:"),
                    "{} slot {slot}: {error}",
                    original.builtin()
                );
            }
            for noncanonical in [" (t-var {} t0)", "(t-var   {} t0)", "(t-var {} t0)\n"] {
                let mut obligation = original.clone();
                *carried_types_mut(&mut obligation)[slot] = noncanonical.into();
                let error = validate_collection_obligation_ledger(&[obligation], &NativeStack)
                    .expect_err("noncanonical carried type");
                assert!(
                    matches!(&error, CollectionValidationError::NoncanonicalType { supplied, canonical }
                    if supplied == noncanonical && canonical == T0)
                );
                assert_eq!(
                    error.to_string(),
                    format!(
                        "must use exact canonical Deep rendering; got `{noncanonical}`, canonical form is `{T0}`"
                    )
                );
            }
        }
    }
}

#[test]
fn collection_ledgers_reject_duplicates_reverse_builtin_order_and_reverse_type_order() {
    let obligations = all_obligations();
    for obligation in &obligations {
        let error = validate_collection_obligation_ledger(
            &[obligation.clone(), obligation.clone()],
            &NativeStack,
        )
        .expect_err("duplicate relation");
        assert!(matches!(error, CollectionValidationError::StrictOrder));
        assert_eq!(
            error.to_string(),
            "must be strictly sorted with no duplicates"
        );
    }
    for pair in obligations.windows(2) {
        let error = validate_collection_obligation_ledger(
            &[pair[1].clone(), pair[0].clone()],
            &NativeStack,
        )
        .expect_err("reverse builtin order");
        assert!(matches!(error, CollectionValidationError::StrictOrder));
        assert_eq!(
            error.to_string(),
            "must be strictly sorted with no duplicates"
        );
    }
    for original in obligations {
        for slot in 0..original.types().len() {
            let mut first = original.clone();
            let mut second = original.clone();
            *carried_types_mut(&mut first)[slot] = T0.into();
            *carried_types_mut(&mut second)[slot] = T1.into();
            validate_collection_obligation_ledger(&[first.clone(), second.clone()], &NativeStack)
                .expect("carried type tuple determines order");
            let error = validate_collection_obligation_ledger(&[second, first], &NativeStack)
                .expect_err("reverse carried type order");
            assert!(matches!(error, CollectionValidationError::StrictOrder));
            assert_eq!(
                error.to_string(),
                "must be strictly sorted with no duplicates"
            );
        }
    }
}

#[test]
fn every_collection_field_rejects_hidden_type_dimension_and_rank_variables() {
    for original in all_obligations() {
        for slot in 0..original.types().len() {
            for (carried, variable) in [
                ("(t-var {} t2)", "t2"),
                (
                    "(t-tensor {} (d-var {} d1) (d-rank {} r0) (t-prim {} f32))",
                    "d1",
                ),
                (
                    "(t-tensor {} (d-var {} d0) (d-rank {} r1) (t-prim {} f32))",
                    "r1",
                ),
            ] {
                let mut obligation = original.clone();
                *carried_types_mut(&mut obligation)[slot] = carried.into();
                validate_collection_obligation_ledger(&[obligation.clone()], &NativeStack)
                    .expect("ledger alone does not check function membership");
                let error = validate_collection_obligations_for_type_repr(
                    Some(SHAPED_FUNCTION),
                    &[obligation],
                    &NativeStack,
                )
                .expect_err("hidden contract variable");
                let expected_prefix = &variable[..1];
                let expected_index: u32 = variable[1..].parse().expect("fixture index");
                assert!(
                    matches!(&error, CollectionValidationError::MissingContractVariable { builtin, prefix, index }
                    if *builtin == original.builtin() && *prefix == expected_prefix && *index == expected_index)
                );
                assert_eq!(
                    error.to_string(),
                    format!(
                        "contain a `{}` contract variable `{variable}` absent from its type representation",
                        original.builtin()
                    )
                );
            }
        }
    }
}

#[test]
fn collection_carried_variables_allow_sparse_subsets_but_reject_noncanonical_spelling() {
    let source = "(t-fn {} (t-var {} t0) (t-fn {} (t-var {} t1) (t-var {} t2)))";
    let mut obligation = CollectionObligation::Len {
        operand: "(t-var {} t2)".into(),
        result: I64.into(),
    };
    validate_collection_obligations_for_type_repr(
        Some(source),
        &[obligation.clone()],
        &NativeStack,
    )
    .expect("carried variables need not begin at zero");
    for (carried, expected) in [
        (
            "(t-var {} t02)",
            "t-var identities must use canonical `t<index>` spelling",
        ),
        (
            "(t-tensor {} (d-var {} d00) (d-rank {} r0) (t-prim {} f32))",
            "d-var identities must use canonical `d<index>` spelling",
        ),
        (
            "(t-tensor {} (d-var {} d0) (d-rank {} r00) (t-prim {} f32))",
            "d-rank identities must use canonical `r<index>` spelling",
        ),
        (
            "(t-var {} t4294967296)",
            "t-var identity index exceeds the u32 range",
        ),
    ] {
        *carried_types_mut(&mut obligation)[0] = carried.into();
        let error = validate_collection_obligations_for_type_repr(
            Some(SHAPED_FUNCTION),
            &[obligation.clone()],
            &NativeStack,
        )
        .expect_err("invalid carried identity");
        assert!(matches!(
            error,
            CollectionValidationError::TypeRepresentation { .. }
        ));
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn collection_membership_ignores_variables_that_occur_only_in_metadata() {
    for (carried, variable) in [(T0, "t0"), (TENSOR, "d0")] {
        let source = "(t-fn {} (t-prim {type: (t-tensor {} (d-var {} d0) (d-rank {} r0) (t-var {} t0))} f32) (t-prim {} i64))";
        let obligation = CollectionObligation::Len {
            operand: carried.into(),
            result: I64.into(),
        };
        let error = validate_collection_obligations_for_type_repr(
            Some(source),
            &[obligation],
            &NativeStack,
        )
        .expect_err("metadata does not declare contract variables");
        assert_eq!(
            error.to_string(),
            format!(
                "contain a `len` contract variable `{variable}` absent from its type representation"
            )
        );
    }
}

#[test]
fn collection_contracts_accept_declared_shape_variables_in_every_carried_field() {
    for original in all_obligations() {
        for slot in 0..original.types().len() {
            let mut obligation = original.clone();
            *carried_types_mut(&mut obligation)[slot] = TENSOR.into();
            validate_collection_obligations_for_type_repr(
                Some(SHAPED_FUNCTION),
                &[obligation],
                &NativeStack,
            )
            .expect("declared shape variables in any relation field");
        }
    }
}

#[test]
fn collection_ledger_errors_precede_function_and_variable_errors() {
    let duplicate = CollectionObligation::Len {
        operand: T0.into(),
        result: T1.into(),
    };
    let error = validate_collection_obligations_for_type_repr(
        None,
        &[duplicate.clone(), duplicate.clone()],
        &NativeStack,
    )
    .expect_err("ledger order precedes absent function");
    assert_eq!(
        error.to_string(),
        "must be strictly sorted with no duplicates"
    );
    let mut malformed = duplicate.clone();
    *carried_types_mut(&mut malformed)[0] = "not Deep".into();
    let error =
        validate_collection_obligations_for_type_repr(None, &[duplicate, malformed], &NativeStack)
            .expect_err("carried syntax precedes ordering and absent function");
    assert!(
        error
            .to_string()
            .starts_with("contain a type representation that is not valid Deep:")
    );
}

#[test]
fn shell_collection_errors_keep_export_context_and_precede_missing_value_types() {
    let mut package = shell();
    let export = symbol(&mut package);
    export.type_repr = None;
    export.type_variable_restrictions.clear();
    export.collection_obligations = vec![CollectionObligation::Len {
        operand: T0.into(),
        result: T1.into(),
    }];
    let error = validate_shell(&package, &NativeStack)
        .expect_err("collection boundary precedes missing value type");
    assert!(
        matches!(&error, ShellValidationError::CollectionObligations { module, symbol, source }
        if module == "Boundary.Types" && symbol == "apply"
            && matches!(source, CollectionValidationError::MissingFunctionType))
    );
    assert_eq!(
        support::shell_message(&error),
        "collection obligations for export `apply` in module `Boundary.Types` require a function type representation"
    );
    symbol(&mut package).type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t0))".into());
    let error = validate_shell(&package, &NativeStack).expect_err("hidden variable with context");
    assert!(
        matches!(&error, ShellValidationError::CollectionObligations { module, symbol, source }
        if module == "Boundary.Types" && symbol == "apply"
            && matches!(source, CollectionValidationError::MissingContractVariable { builtin: "len", prefix: "t", index: 1 }))
    );
    assert_eq!(
        support::shell_message(&error),
        "collection obligations for export `apply` in module `Boundary.Types` contain a `len` contract variable `t1` absent from its type representation"
    );
}
