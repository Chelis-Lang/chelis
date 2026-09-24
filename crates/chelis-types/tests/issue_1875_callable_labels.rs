//! Persistent semantic-label transport, separate from the immutable prototype oracle.

/// Consumer families, not repeated witnesses of only one operation.
#[test]
fn dimension_observations_cover_arithmetic_comparison_and_propagation() {
    let cases = [
        (
            "diagonal cannot erase authored result identity",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_00.deep"),
        ),
        (
            "stride identity cannot erase authored result identity",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_01.deep"),
        ),
        (
            "pad identity cannot erase authored result identity",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_02.deep"),
        ),
        (
            "deferred gather retained name",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_03.deep"),
        ),
        (
            "deferred gather false result",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_04.deep"),
        ),
        (
            "deferred trace retained name",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_05.deep"),
        ),
        (
            "deferred trace false result",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_06.deep"),
        ),
        (
            "deferred scatter retained name",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_07.deep"),
        ),
        (
            "deferred scatter bad updates",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_08.deep"),
        ),
        (
            "deferred scatter_replace retained name",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_09.deep"),
        ),
        (
            "deferred scatter_replace bad updates",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_10.deep"),
        ),
        (
            "expand bare nonunit",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_11.deep"),
        ),
        (
            "reshape bare bad product",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_12.deep"),
        ),
        (
            "concat bare wrong sum",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_13.deep"),
        ),
        (
            "window unknown runtime claim",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_14.deep"),
        ),
        (
            "stride unknown runtime claim",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_15.deep"),
        ),
        (
            "pad unknown runtime claim",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_16.deep"),
        ),
        (
            "diagonal bound identity query",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_17.deep"),
        ),
        (
            "where bound contradiction",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_18.deep"),
        ),
        (
            "where bare literal not a name",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_19.deep"),
        ),
        (
            "clamp bound contradiction",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_20.deep"),
        ),
        (
            "gather retained name",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_21.deep"),
        ),
        (
            "trace retained name",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_22.deep"),
        ),
        (
            "gather declared identity wrong",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_23.deep"),
        ),
        (
            "trace declared identity wrong",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_24.deep"),
        ),
        (
            "unresolved gather lambda at binding rejects",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_25.deep"),
        ),
        (
            "unresolved trace lambda at binding rejects",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_26.deep"),
        ),
        (
            "scatter symbolic",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_27.deep"),
        ),
        (
            "scatter different name",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_28.deep"),
        ),
        (
            "scatter known fits",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_29.deep"),
        ),
        (
            "scatter known overshoot",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_30.deep"),
        ),
        (
            "where symbolic",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_31.deep"),
        ),
        (
            "where distinct name",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_32.deep"),
        ),
        (
            "clamp symbolic",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_33.deep"),
        ),
        (
            "clamp distinct name",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_34.deep"),
        ),
        (
            "layer norm positive",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_35.deep"),
        ),
        (
            "layer norm zero",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_36.deep"),
        ),
        (
            "expand unit",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_37.deep"),
        ),
        (
            "expand named nonunit runtime claim",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_38.deep"),
        ),
        (
            "matmul unit batch",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_39.deep"),
        ),
        (
            "matmul wrong batch",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_40.deep"),
        ),
        (
            "window extent",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_41.deep"),
        ),
        (
            "window oversize",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_42.deep"),
        ),
        (
            "window wrong extent",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_43.deep"),
        ),
        (
            "shrink fits",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_44.deep"),
        ),
        (
            "shrink overshoot",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_45.deep"),
        ),
        (
            "stride extent",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_46.deep"),
        ),
        (
            "stride wrong extent",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_47.deep"),
        ),
        (
            "stride identity query",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_48.deep"),
        ),
        (
            "stride bound identity query",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_49.deep"),
        ),
        (
            "pad extent",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_50.deep"),
        ),
        (
            "pad wrong extent",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_51.deep"),
        ),
        (
            "pad identity query",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_52.deep"),
        ),
        (
            "pad bound identity query",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_53.deep"),
        ),
        (
            "reshape product",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_54.deep"),
        ),
        (
            "reshape named input runtime product claim",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_55.deep"),
        ),
        (
            "reshape negative",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_56.deep"),
        ),
        (
            "concat sum",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_57.deep"),
        ),
        (
            "concat named input unknown sum runtime claim",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_58.deep"),
        ),
        (
            "diagonal selected name query",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_59.deep"),
        ),
        (
            "diagonal minimum",
            true,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_60.deep"),
        ),
        (
            "diagonal wrong minimum",
            false,
            include_str!("fixtures/core/tests/issue_1875_callable_labels/dimension_61.deep"),
        ),
    ];
    let mut failures = Vec::new();
    for reverse in [false, true] {
        let library = if reverse {
            include_str!(
                "fixtures/core/tests/issue_1875_callable_labels/labels_library_reverse.deep"
            )
        } else {
            include_str!("fixtures/core/tests/issue_1875_callable_labels/labels_library.deep")
        };
        let (live, _) =
            build_compiled_library_context(&parse(library), chelis_types::ProgramOrigin::Unlinked)
                .unwrap();
        let decoded: TypeEnv = bincode::deserialize(&bincode::serialize(&live).unwrap()).unwrap();
        let (layered, _) = build_compiled_library_context_with_base(
            &live,
            &parse(include_str!(
                "fixtures/core/tests/issue_1875_callable_labels/labels_extension.deep"
            )),
            chelis_types::ProgramOrigin::Unlinked,
        )
        .unwrap();
        for (name, expected, source) in cases {
            for route in 0..4 {
                let result = match route {
                    0 => check_ir_program(
                        &parse(&format!("{library}\n{source}")),
                        chelis_types::ProgramOrigin::Unlinked,
                    ),
                    1 => check_ir_with_context(
                        &live,
                        &parse(source),
                        chelis_types::ProgramOrigin::Unlinked,
                    ),
                    2 => check_ir_with_context(
                        &decoded,
                        &parse(source),
                        chelis_types::ProgramOrigin::Unlinked,
                    ),
                    _ => check_ir_with_context(
                        &layered,
                        &parse(
                            &source
                                .replace(" tag)", " forward)")
                                .replace(" tag2)", " forward2)")
                                .replace(" tag3)", " forward3)")
                                .replace(" square)", " forward_square)"),
                        ),
                        chelis_types::ProgramOrigin::Unlinked,
                    ),
                };
                let correct = if expected {
                    result.is_ok()
                } else {
                    result.as_ref().err().is_some_and(|r| {
                        r.errors.iter().any(|e| {
                            matches!(
                                e.kind,
                                CheckErrorKind::DimensionMismatch | CheckErrorKind::TypeMismatch
                            )
                        })
                    })
                };
                if !correct {
                    failures.push(format!(
                        "{name} reverse={reverse} route={route} expected={expected}: {:?}",
                        result.err().map(|r| r.errors)
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} consumer observations failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
use chelis_deep::parse_and_stamp_file;
use chelis_types::errors::CheckErrorKind;
use chelis_types::{
    TypeEnv, build_compiled_library_context, build_compiled_library_context_with_base,
    check_ir_program, check_ir_with_context,
};

fn parse(source: &str) -> Vec<chelis_deep::Expr> {
    parse_and_stamp_file(source).expect("core fixture parses")
}

#[test]
fn concrete_instantiations_and_value_aliases_keep_names_without_coupling_calls() {
    for reverse in [false, true] {
        let library = if reverse {
            include_str!(
                "fixtures/core/tests/issue_1875_callable_labels/aligned_library_reverse.deep"
            )
        } else {
            include_str!("fixtures/core/tests/issue_1875_callable_labels/aligned_library.deep")
        };
        let consumer = include_str!("fixtures/core/tests/issue_1875_callable_labels/aliases.deep");
        check_ir_program(
            &parse(&format!("{library}\n{consumer}")),
            chelis_types::ProgramOrigin::Unlinked,
        )
        .expect("whole: independent calls and aliases");
        let (live, _) =
            build_compiled_library_context(&parse(library), chelis_types::ProgramOrigin::Unlinked)
                .unwrap();
        let bytes = bincode::serialize(&live).unwrap();
        let decoded: TypeEnv = bincode::deserialize(&bytes).unwrap();
        for context in [&live, &decoded] {
            check_ir_with_context(
                context,
                &parse(consumer),
                chelis_types::ProgramOrigin::Unlinked,
            )
            .expect("context: independent calls and aliases");
        }
    }
}

/// §4.5.2 joins concrete ragged axes without erasing genuine binder/name
/// uniformity. A label-bearing ID may already have a concrete constraint.
#[test]
fn concrete_labelled_list_join_preserves_ragged_and_uniformity_boundaries() {
    let mut failures = Vec::new();
    for reverse_mul in [false, true] {
        let library = if reverse_mul {
            include_str!(
                "fixtures/core/tests/issue_1875_callable_labels/aligned_library_reverse.deep"
            )
        } else {
            include_str!("fixtures/core/tests/issue_1875_callable_labels/aligned_library.deep")
        };
        let (live, _) =
            build_compiled_library_context(&parse(library), chelis_types::ProgramOrigin::Unlinked)
                .unwrap();
        let decoded: TypeEnv = bincode::deserialize(&bincode::serialize(&live).unwrap()).unwrap();
        let extension =
            include_str!("fixtures/core/tests/issue_1875_callable_labels/aligned_extension.deep");
        let (layered, _) = build_compiled_library_context_with_base(
            &live,
            &parse(extension),
            chelis_types::ProgramOrigin::Unlinked,
        )
        .unwrap();
        for reverse_list in [false, true] {
            let literal2 =
                include_str!("fixtures/core/tests/issue_1875_callable_labels/literal2.deep.in");
            let literal3 =
                include_str!("fixtures/core/tests/issue_1875_callable_labels/literal3.deep.in");
            let labelled2 = call("aligned", &[literal2, literal2]);
            let labelled3 = call("aligned", &[literal3, literal3]);
            let list = |a: &str, b: &str| {
                if reverse_list { pair(b, a) } else { pair(a, b) }
            };
            let mixed = list(&labelled2, literal3);
            let ragged = list(&labelled2, &labelled3);
            let rigid_mixed = list(&call("aligned", &["(var {} x)", "(var {} gain)"]), literal3);
            let rigid_distinct = list(
                &call("aligned", &["(var {} x)", "(var {} gain)"]),
                &call("aligned", &["(var {} y)", "(var {} gain)"]),
            );
            let same_concrete = list(&labelled2, &labelled2);
            let cases = [
                (
                    "mixed concrete",
                    true,
                    body_template(
                        include_str!("fixtures/core/tests/issue_1875_callable_labels/bare.deep.in"),
                        &mixed,
                    ),
                ),
                (
                    "independent concrete calls",
                    true,
                    body_template(
                        include_str!("fixtures/core/tests/issue_1875_callable_labels/bare.deep.in"),
                        &ragged,
                    ),
                ),
                (
                    "explicit wildcard",
                    true,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/wildcard.deep.in"
                        ),
                        &ragged,
                    ),
                ),
                (
                    "named uniformity",
                    false,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/uniform.deep.in"
                        ),
                        &ragged,
                    ),
                ),
                (
                    "rigid plus literal",
                    false,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/rigid.deep.in"
                        ),
                        &rigid_mixed,
                    ),
                ),
                (
                    "distinct rigid",
                    false,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/distinct_rigid.deep.in"
                        ),
                        &rigid_distinct,
                    ),
                ),
                (
                    "same rigid",
                    true,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/rigid.deep.in"
                        ),
                        &pair(
                            &call("aligned", &["(var {} x)", "(var {} gain)"]),
                            &call("aligned", &["(var {} x)", "(var {} gain)"]),
                        ),
                    ),
                ),
                (
                    "equal labelled query",
                    true,
                    body_template(
                        include_str!("fixtures/core/tests/issue_1875_callable_labels/bare.deep.in"),
                        &query(&same_concrete),
                    ),
                ),
                (
                    "distinct named uniformity",
                    false,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/distinct_named.deep.in"
                        ),
                        &list("(var {} a)", "(var {} gain)"),
                    ),
                ),
                (
                    "rank mismatch",
                    false,
                    body_template(
                        include_str!("fixtures/core/tests/issue_1875_callable_labels/bare.deep.in"),
                        &list(
                            &labelled2,
                            include_str!(
                                "fixtures/core/tests/issue_1875_callable_labels/matrix.deep.in"
                            ),
                        ),
                    ),
                ),
            ];
            for (case, expected, source) in cases {
                for route in 0..4 {
                    let result = match route {
                        0 => check_ir_program(
                            &parse(&format!("{library}\n{source}")),
                            chelis_types::ProgramOrigin::Unlinked,
                        ),
                        1 => check_ir_with_context(
                            &live,
                            &parse(&source),
                            chelis_types::ProgramOrigin::Unlinked,
                        ),
                        2 => check_ir_with_context(
                            &decoded,
                            &parse(&source),
                            chelis_types::ProgramOrigin::Unlinked,
                        ),
                        _ => check_ir_with_context(
                            &layered,
                            &parse(&source.replace(" aligned)", " forwarding)")),
                            chelis_types::ProgramOrigin::Unlinked,
                        ),
                    };
                    let correct = if expected {
                        result.is_ok()
                    } else {
                        result.as_ref().err().is_some_and(|report| {
                            report.errors.iter().any(|error| {
                                matches!(error.kind, CheckErrorKind::DimensionMismatch)
                            })
                        })
                    };
                    if !correct {
                        failures.push(format!("{case}: reverse_mul={reverse_mul} reverse_list={reverse_list} route={route} expected={expected}: {:?}", result.err().map(|r| r.errors)));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} observations failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn list_axis_classification_crosses_extent_label_rigidity_and_wildcard() {
    let mut failures = Vec::new();
    let mut observations = 0;
    for reverse_mul in [false, true] {
        let library = if reverse_mul {
            include_str!(
                "fixtures/core/tests/issue_1875_callable_labels/classification_library_reverse.deep"
            )
        } else {
            include_str!(
                "fixtures/core/tests/issue_1875_callable_labels/classification_library.deep"
            )
        };
        let (live, _) =
            build_compiled_library_context(&parse(library), chelis_types::ProgramOrigin::Unlinked)
                .unwrap();
        let decoded: TypeEnv = bincode::deserialize(&bincode::serialize(&live).unwrap()).unwrap();
        let extension = include_str!(
            "fixtures/core/tests/issue_1875_callable_labels/classification_extension.deep"
        );
        let (layered, _) = build_compiled_library_context_with_base(
            &live,
            &parse(extension),
            chelis_types::ProgramOrigin::Unlinked,
        )
        .unwrap();
        let literal2 =
            include_str!("fixtures/core/tests/issue_1875_callable_labels/literal2.deep.in");
        let literal3 =
            include_str!("fixtures/core/tests/issue_1875_callable_labels/literal3.deep.in");

        for reverse_list in [false, true] {
            let list = |a: &str, b: &str| {
                if reverse_list { pair(b, a) } else { pair(a, b) }
            };
            let mut cases = Vec::new();
            for same_label in [false, true] {
                let name = if same_label { "fixed" } else { "other" };
                let callee = format!("{name}_gain");
                for equal_extent in [false, true] {
                    let second = if equal_extent { literal2 } else { literal3 };
                    let xs = list(
                        &call("fixed_gain", &[literal2, literal2]),
                        &call(&callee, &[second, second]),
                    );
                    let uniform = same_label && equal_extent;
                    for (claim, expected, body) in [
                        (
                            "bare",
                            true,
                            body_template(
                                include_str!(
                                    "fixtures/core/tests/issue_1875_callable_labels/bare.deep.in"
                                ),
                                &xs,
                            ),
                        ),
                        (
                            "wildcard",
                            true,
                            body_template(
                                include_str!(
                                    "fixtures/core/tests/issue_1875_callable_labels/wildcard.deep.in"
                                ),
                                &xs,
                            ),
                        ),
                        (
                            "uniform",
                            uniform,
                            body_template(
                                include_str!(
                                    "fixtures/core/tests/issue_1875_callable_labels/uniform.deep.in"
                                ),
                                &xs,
                            ),
                        ),
                        (
                            "query",
                            uniform,
                            body_template(
                                include_str!(
                                    "fixtures/core/tests/issue_1875_callable_labels/bare.deep.in"
                                ),
                                &query(&xs),
                            ),
                        ),
                    ] {
                        cases.push((
                            format!("literal label={same_label} extent={equal_extent} {claim}"),
                            expected,
                            body,
                        ));
                    }
                }
                // Unknown extents of concrete named parameters are not authored
                // generic binders, even after a helper installs a private ID.

                let xs = list(
                    &call("fixed_gain", &["(var {} x)", "(var {} gx)"]),
                    &call(&callee, &["(var {} y)", "(var {} gy)"]),
                );
                cases.push((
                    format!("named bare {name}"),
                    true,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/named_bare.deep.in"
                        ),
                        &xs,
                    )
                    .replace("fixture_label", name),
                ));
                cases.push((
                    format!("named uniform {name}"),
                    same_label,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/named_uniform.deep.in"
                        ),
                        &xs,
                    )
                    .replace("fixture_label", name),
                ));
                cases.push((
                    format!("named query {name}"),
                    same_label,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/named_bare.deep.in"
                        ),
                        &query(&xs),
                    )
                    .replace("fixture_label", name),
                ));
                cases.push((
                    format!("published captures {name}"),
                    true,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/named_bare.deep.in"
                        ),
                        &captures(name, &list(&call("a", &[]), &call("b", &[]))),
                    )
                    .replace("fixture_label", name),
                ));
                // Actual active authored binders must still unify, even with an
                // explicit wildcard result and even through returned closures.

                cases.push((
                    format!("distinct generic {name}"),
                    false,
                    body_template(include_str!("fixtures/core/tests/issue_1875_callable_labels/generic_wildcard.deep.in"), &xs).replace("fixture_label", name),
                ));
                cases.push((
                    format!("generic captures {name}"),
                    false,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/generic_bare.deep.in"
                        ),
                        &captures(name, &list(&call("a", &[]), &call("b", &[]))),
                    )
                    .replace("fixture_label", name),
                ));
            }
            for known_extent in [false, true] {
                let bare = if known_extent {
                    include_str!(
                        "fixtures/core/tests/issue_1875_callable_labels/wildcard_known_bare.deep.in"
                    )
                } else {
                    include_str!(
                        "fixtures/core/tests/issue_1875_callable_labels/wildcard_named_bare.deep.in"
                    )
                };
                let uniform = if known_extent {
                    include_str!(
                        "fixtures/core/tests/issue_1875_callable_labels/wildcard_known_uniform.deep.in"
                    )
                } else {
                    include_str!(
                        "fixtures/core/tests/issue_1875_callable_labels/wildcard_named_uniform.deep.in"
                    )
                };
                let named = if known_extent {
                    call("fixed_gain", &[literal2, literal2])
                } else {
                    call("fixed_gain", &["(var {} x)", "(var {} gain)"])
                };
                let xs = list(&named, "(var {} w)");
                cases.push((
                    format!("wildcard bare {known_extent}"),
                    true,
                    body_template(bare, &xs),
                ));
                cases.push((
                    format!("wildcard uniform {known_extent}"),
                    !reverse_list,
                    body_template(uniform, &xs),
                ));
                cases.push((
                    format!("wildcard query {known_extent}"),
                    !reverse_list,
                    body_template(bare, &query(&xs)),
                ));
            }
            for equal_extent in [false, true] {
                let plain = if equal_extent { literal2 } else { literal3 };
                let xs = list(&call("fixed_gain", &[literal2, literal2]), plain);
                cases.push((
                    format!("name/literal bare {equal_extent}"),
                    true,
                    body_template(
                        include_str!("fixtures/core/tests/issue_1875_callable_labels/bare.deep.in"),
                        &xs,
                    ),
                ));
                cases.push((
                    format!("name/literal uniform {equal_extent}"),
                    false,
                    body_template(
                        include_str!(
                            "fixtures/core/tests/issue_1875_callable_labels/uniform.deep.in"
                        ),
                        &xs,
                    ),
                ));
                cases.push((
                    format!("name/literal query {equal_extent}"),
                    false,
                    body_template(
                        include_str!("fixtures/core/tests/issue_1875_callable_labels/bare.deep.in"),
                        &query(&xs),
                    ),
                ));
            }
            for (case, expected, source) in cases {
                for route in 0..4 {
                    let result = match route {
                        0 => check_ir_program(
                            &parse(&format!("{library}\n{source}")),
                            chelis_types::ProgramOrigin::Unlinked,
                        ),
                        1 => check_ir_with_context(
                            &live,
                            &parse(&source),
                            chelis_types::ProgramOrigin::Unlinked,
                        ),
                        2 => check_ir_with_context(
                            &decoded,
                            &parse(&source),
                            chelis_types::ProgramOrigin::Unlinked,
                        ),
                        _ => check_ir_with_context(
                            &layered,
                            &parse(
                                &source
                                    .replace(" fixed_gain)", " forward_fixed)")
                                    .replace(" other_gain)", " forward_other)"),
                            ),
                            chelis_types::ProgramOrigin::Unlinked,
                        ),
                    };
                    observations += 1;
                    let correct = if expected {
                        result.is_ok()
                    } else {
                        result.as_ref().err().is_some_and(|r| {
                            r.errors
                                .iter()
                                .any(|e| matches!(e.kind, CheckErrorKind::DimensionMismatch))
                        })
                    };
                    if !correct {
                        failures.push(format!("{case} mul={reverse_mul} list={reverse_list} route={route} expected={expected}: {:?}", result.err().map(|r| r.errors)));
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} observations failed:\n{}",
        failures.len(),
        observations,
        failures.join("\n")
    );
}

// Templates are captured from the real frontend, then filled only with Deep expressions.
fn body_template(template: &str, body: &str) -> String {
    template.replace("@BODY@", body)
}

fn call(name: &str, args: &[&str]) -> String {
    format!("(app {{}} (var {{}} {name}) {})", args.join(" "))
}

fn pair(a: &str, b: &str) -> String {
    include_str!("fixtures/core/tests/issue_1875_callable_labels/pair.deep.in")
        .replace("@A@", a)
        .replace("@B@", b)
}

fn query(list: &str) -> String {
    body_template(
        include_str!("fixtures/core/tests/issue_1875_callable_labels/query.deep.in"),
        list,
    )
}

fn captures(label: &str, body: &str) -> String {
    body_template(
        include_str!("fixtures/core/tests/issue_1875_callable_labels/captures.deep.in"),
        body,
    )
    .replace("fixture_label", label)
}
