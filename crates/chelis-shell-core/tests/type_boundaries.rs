mod support;

use chelis_shell_core::{
    RestrictionIdentityError, ShellValidationError, SymbolKind, TypeRepresentationError,
    validate_shell,
};
use chelis_syntax_host::NativeStack;
use support::{restriction, shell, symbol};

#[test]
fn deep_types_reject_invalid_syntax_roots_and_nested_carriers_in_every_scheme_shape() {
    let malformed = [
        "",
        "(t-unit {}",
        "(t-unit {}) (t-unit {})",
        "(not-a-type {} f32)",
        "f32",
        "()",
        "{type: (t-prim {} f32)}",
        "^{:doc \"legacy\"} (t-prim {} f32)",
        "(lit {type: (t-var {} t0)} 1)",
        "(t-fn {} (lit {type: (t-var {} t0)} 1) (t-var {} t0))",
        "(t-tensor {} (t-var {} t0) (d-rank {} r0))",
    ];
    for has_body in [false, true] {
        for restricted in [false, true] {
            for source in malformed {
                let mut package = shell();
                let export = symbol(&mut package);
                export.has_body = has_body;
                export.type_repr = Some(source.into());
                if !restricted {
                    export.type_variable_restrictions.clear();
                }
                let error = validate_shell(&package, &NativeStack).expect_err("invalid Deep type");
                assert!(
                    matches!(
                        error,
                        ShellValidationError::TypeRepresentation {
                            source: TypeRepresentationError::InvalidDeep { .. }
                        }
                    ),
                    "source={source:?}, has_body={has_body}, restricted={restricted}: {error:?}"
                );
                assert!(
                    support::shell_message(&error)
                        .starts_with("type representation is not valid Deep:")
                );
            }
        }
    }
}

#[test]
fn canonical_variable_spelling_rejects_each_namespace_and_u32_overflow() {
    for (tag, prefix, type_template) in [
        ("t-var", "t", "(t-var {} ID)"),
        (
            "d-var",
            "d",
            "(t-tensor {} (d-var {} ID) (d-rank {} r0) (t-prim {} f32))",
        ),
        (
            "d-rank",
            "r",
            "(t-tensor {} (d-var {} d0) (d-rank {} ID) (t-prim {} f32))",
        ),
    ] {
        for identity in [
            prefix.to_string(),
            format!("{prefix}00"),
            format!("{prefix}01"),
            format!("{prefix}x"),
            format!("{prefix}4294967296"),
            "wrong0".into(),
        ] {
            let mut package = shell();
            let export = symbol(&mut package);
            export.type_variable_restrictions.clear();
            export.type_repr = Some(type_template.replace("ID", &identity));
            let error = validate_shell(&package, &NativeStack).expect_err("noncanonical variable");
            if identity.ends_with("4294967296") {
                assert!(matches!(&error, ShellValidationError::TypeRepresentation {
                    source: TypeRepresentationError::IdentityOverflow { tag: actual }
                } if *actual == tag));
            } else {
                assert!(matches!(&error, ShellValidationError::TypeRepresentation {
                    source: TypeRepresentationError::IdentitySpelling { tag: actual_tag, prefix: actual_prefix }
                } if *actual_tag == tag && *actual_prefix == prefix));
            }
            let expected = if identity.ends_with("4294967296") {
                format!("{tag} identity index exceeds the u32 range")
            } else {
                format!("{tag} identities must use canonical `{prefix}<index>` spelling")
            };
            assert_eq!(support::shell_message(&error), expected, "{identity}");
        }
    }
}

#[test]
fn canonical_variables_reject_gaps_in_each_independent_namespace() {
    for (namespace, prefix, source) in [
        ("type", "t", "(t-fn {} (t-var {} t0) (t-var {} t2))"),
        (
            "dimension",
            "d",
            "(t-tensor {} (d-var {} d1) (d-rank {} r0) (t-prim {} f32))",
        ),
        (
            "rank",
            "r",
            "(t-tensor {} (d-var {} d0) (d-rank {} r1) (t-prim {} f32))",
        ),
    ] {
        let mut package = shell();
        let export = symbol(&mut package);
        export.type_variable_restrictions.clear();
        export.type_repr = Some(source.into());
        let error = validate_shell(&package, &NativeStack).expect_err("namespace gap");
        assert!(matches!(&error, ShellValidationError::TypeRepresentation {
            source: TypeRepresentationError::Noncontiguous { namespace: actual_namespace, prefix: actual_prefix }
        } if *actual_namespace == namespace && *actual_prefix == prefix));
        assert_eq!(
            support::shell_message(&error),
            format!(
                "type representation {namespace} variables must be contiguous from `{prefix}0`"
            )
        );
    }
}

#[test]
fn canonical_variable_sets_accept_repetition_reverse_occurrence_and_independent_namespaces() {
    for source in [
        "(t-fn {} (t-var {} t1) (t-var {} t0))",
        "(t-fn {} (t-var {} t0) (t-var {} t0))",
        "(t-fn {} (t-tensor {} (d-var {} d1) (d-rank {} r1) (t-var {} t1)) (t-tensor {} (d-var {} d0) (d-rank {} r0) (t-var {} t0)))",
    ] {
        let mut package = shell();
        symbol(&mut package).type_repr = Some(source.into());
        validate_shell(&package, &NativeStack).expect("canonical variable set");
    }
}

#[test]
fn restriction_indices_use_numeric_order_instead_of_lexical_order() {
    let mut source = "(t-var {} t10)".to_string();
    for index in (0..10).rev() {
        source = format!("(t-fn {{}} (t-var {{}} t{index}) {source})");
    }
    let mut package = shell();
    let export = symbol(&mut package);
    export.type_repr = Some(source);
    export.type_variable_restrictions = vec![restriction("t2"), restriction("t10")];
    validate_shell(&package, &NativeStack).expect("numeric restriction order");
    symbol(&mut package).type_variable_restrictions.reverse();
    let error =
        validate_shell(&package, &NativeStack).expect_err("lexical order is not numeric order");
    assert!(
        matches!(&error, ShellValidationError::RestrictionOrder { module, symbol }
        if module == "Boundary.Types" && symbol == "apply")
    );
    assert_eq!(
        support::shell_message(&error),
        "type-variable restrictions for export `apply` in module `Boundary.Types` must be strictly sorted by canonical index with no duplicates"
    );
}

#[test]
fn restrictions_reject_duplicate_indices_and_noncanonical_identities() {
    let mut package = shell();
    symbol(&mut package).type_variable_restrictions = vec![restriction("t0"), restriction("t0")];
    assert_eq!(
        support::shell_message(
            &validate_shell(&package, &NativeStack).expect_err("duplicate restriction")
        ),
        "type-variable restrictions for export `apply` in module `Boundary.Types` must be strictly sorted by canonical index with no duplicates"
    );
    for variable in [
        "",
        "t",
        "t00",
        "t01",
        "d0",
        "r0",
        "t-1",
        "t+1",
        "t١",
        "t4294967296",
    ] {
        symbol(&mut package).type_variable_restrictions = vec![restriction(variable)];
        let error =
            validate_shell(&package, &NativeStack).expect_err("invalid restriction identity");
        let expected_source = if variable == "t4294967296" {
            RestrictionIdentityError::IndexOverflow
        } else {
            RestrictionIdentityError::Spelling
        };
        assert!(
            matches!(&error, ShellValidationError::RestrictionIdentity { source }
            if *source == expected_source)
        );
        let expected = if variable == "t4294967296" {
            "restricted type-variable identity index exceeds the u32 range"
        } else {
            "restricted type-variable identities must use canonical `t<index>` spelling"
        };
        assert_eq!(support::shell_message(&error), expected, "{variable:?}");
    }
}

#[test]
fn restrictions_reject_absent_types_and_detached_or_metadata_only_variables() {
    let mut package = shell();
    let export = symbol(&mut package);
    export.kind = SymbolKind::Type;
    export.type_repr = None;
    let error = validate_shell(&package, &NativeStack).expect_err("restriction without type");
    assert!(
        matches!(&error, ShellValidationError::RestrictionsWithoutType { module, symbol }
        if module == "Boundary.Types" && symbol == "apply")
    );
    assert_eq!(
        support::shell_message(&error),
        "export `apply` in module `Boundary.Types` carries type-variable restrictions without a type representation"
    );
    for source in ["(t-prim {} f32)", "(t-prim {type: (t-var {} t0)} f32)"] {
        symbol(&mut package).type_repr = Some(source.into());
        let error = validate_shell(&package, &NativeStack).expect_err("detached restriction");
        assert!(
            matches!(&error, ShellValidationError::RestrictionAbsentFromType { module, symbol, variable }
            if module == "Boundary.Types" && symbol == "apply" && variable == "t0")
        );
        assert_eq!(
            support::shell_message(&error),
            "export `apply` in module `Boundary.Types` restricts `t0` but that canonical variable is absent from its type representation"
        );
    }
}

#[test]
fn type_errors_precede_restrictions_and_restrictions_precede_effects() {
    let mut package = shell();
    let export = symbol(&mut package);
    export.type_repr = Some("(t-unit {}".into());
    export.type_variable_restrictions = vec![restriction("bad")];
    export.effects = vec![String::new()];
    let error = validate_shell(&package, &NativeStack).expect_err("type first");
    assert!(matches!(
        error,
        ShellValidationError::TypeRepresentation { .. }
    ));
    symbol(&mut package).type_repr = Some("(t-var {} t0)".into());
    assert_eq!(
        support::shell_message(
            &validate_shell(&package, &NativeStack).expect_err("restriction second")
        ),
        "restricted type-variable identities must use canonical `t<index>` spelling"
    );
    symbol(&mut package).type_variable_restrictions = vec![restriction("t0")];
    let error = validate_shell(&package, &NativeStack).expect_err("effect last");
    assert!(matches!(error, ShellValidationError::InvalidText { .. }));
    assert_eq!(
        support::shell_message(&error),
        "effect name must be non-empty without leading or trailing whitespace"
    );
}

#[test]
fn canonical_u32_max_indices_are_valid_identities_but_not_contiguous_schemes() {
    for (namespace, prefix, source) in [
        ("type", "t", "(t-var {} t4294967295)"),
        (
            "dimension",
            "d",
            "(t-tensor {} (d-var {} d4294967295) (d-rank {} r0) (t-prim {} f32))",
        ),
        (
            "rank",
            "r",
            "(t-tensor {} (d-var {} d0) (d-rank {} r4294967295) (t-prim {} f32))",
        ),
    ] {
        let mut package = shell();
        let export = symbol(&mut package);
        export.type_repr = Some(source.into());
        export.type_variable_restrictions.clear();
        let error =
            validate_shell(&package, &NativeStack).expect_err("valid index with a scheme gap");
        assert!(matches!(&error, ShellValidationError::TypeRepresentation {
            source: TypeRepresentationError::Noncontiguous { namespace: actual_namespace, prefix: actual_prefix }
        } if *actual_namespace == namespace && *actual_prefix == prefix));
    }
    let mut package = shell();
    symbol(&mut package).type_variable_restrictions = vec![restriction("t4294967295")];
    assert!(matches!(validate_shell(&package, &NativeStack),
        Err(ShellValidationError::RestrictionAbsentFromType { variable, .. })
        if variable == "t4294967295"));
}

#[test]
fn type_namespace_gaps_precede_dimension_gaps_and_rank_gaps() {
    let mut package = shell();
    symbol(&mut package).type_variable_restrictions.clear();
    for (source, namespace) in [
        (
            "(t-tensor {} (d-var {} d1) (d-rank {} r1) (t-var {} t1))",
            "type",
        ),
        (
            "(t-tensor {} (d-var {} d1) (d-rank {} r1) (t-var {} t0))",
            "dimension",
        ),
        (
            "(t-tensor {} (d-var {} d0) (d-rank {} r1) (t-var {} t0))",
            "rank",
        ),
    ] {
        symbol(&mut package).type_repr = Some(source.into());
        let error = validate_shell(&package, &NativeStack).expect_err("first namespace gap");
        assert!(matches!(&error, ShellValidationError::TypeRepresentation {
            source: TypeRepresentationError::Noncontiguous { namespace: actual, .. }
        } if *actual == namespace));
    }
}
