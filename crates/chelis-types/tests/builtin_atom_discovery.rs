//! #1294: independently defend declaration and checked-shape discovery.
use chelis_types::types::{Prim, TensorPrec, Type};
use chelis_types::{
    BUILTINS, BuiltinCapabilityDecl, BuiltinSemanticDomain as Domain, BuiltinSiblingCaseDecl,
    BuiltinSiblingCaseId as Case, builtin_decl,
};
use std::collections::BTreeSet;

#[test]
fn every_declared_capability_is_valid() {
    for builtin in BUILTINS {
        builtin.capability.validate().expect("valid declaration");
    }
}

#[test]
fn invalid_domain_case_declarations_fail_loudly() {
    for declaration in [
        BuiltinCapabilityDecl {
            domains: &[],
            sibling_cases: &[],
        },
        BuiltinCapabilityDecl {
            domains: &[Domain::Numeric, Domain::Numeric],
            sibling_cases: &[],
        },
        BuiltinCapabilityDecl {
            domains: &[Domain::Container],
            sibling_cases: &[],
        },
        BuiltinCapabilityDecl {
            domains: &[Domain::Boundary],
            sibling_cases: &[BuiltinSiblingCaseDecl {
                domain: Domain::Container,
                case: Case::LenList,
            }],
        },
        BuiltinCapabilityDecl {
            domains: &[Domain::Container],
            sibling_cases: &[
                BuiltinSiblingCaseDecl {
                    domain: Domain::Container,
                    case: Case::LenList,
                },
                BuiltinSiblingCaseDecl {
                    domain: Domain::Container,
                    case: Case::LenList,
                },
            ],
        },
        BuiltinCapabilityDecl {
            domains: &[Domain::Container, Domain::Boundary],
            sibling_cases: &[
                BuiltinSiblingCaseDecl {
                    domain: Domain::Container,
                    case: Case::LenList,
                },
                BuiltinSiblingCaseDecl {
                    domain: Domain::Boundary,
                    case: Case::LenList,
                },
            ],
        },
    ] {
        assert!(declaration.validate().is_err());
    }
}

#[test]
fn every_declared_identity_has_a_unique_case_selector_witness() {
    let list = Type::Adt("List".into(), vec![Type::Prim(Prim::F32)]);
    let tensor = Type::Tensor(vec![], TensorPrec::Concrete(Prim::F32));
    let dict = Type::Adt(
        "Dict".into(),
        vec![Type::Prim(Prim::String), Type::Prim(Prim::F32)],
    );
    let mut selected = BTreeSet::new();
    for builtin in BUILTINS {
        let witnesses = match builtin.name {
            "len" => vec![vec![list.clone()], vec![dict.clone()]],
            "concat" => vec![
                vec![list.clone(), list.clone()],
                vec![
                    Type::Adt("List".into(), vec![tensor.clone()]),
                    Type::Prim(Prim::Int32),
                ],
            ],
            "drop" => vec![
                vec![list.clone()],
                vec![list.clone(), Type::Prim(Prim::Int64)],
            ],
            "eq" | "neq" => vec![vec![Type::Prim(Prim::F32)], vec![list.clone()]],
            "to_string" => vec![
                Type::Prim(Prim::F32),
                tensor.clone(),
                list.clone(),
                dict.clone(),
                Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]),
                Type::Adt("User".into(), vec![]),
                Type::Tuple(vec![]),
                Type::Unit,
                Type::Fn(vec![], Box::new(Type::Unit)),
            ]
            .into_iter()
            .map(|t| vec![t])
            .collect(),
            _ => vec![vec![Type::Unit]],
        };
        for args in witnesses {
            let identity = builtin.semantic_case(&args).unwrap();
            assert!(
                selected.insert(identity),
                "overlapping selector for {}",
                builtin.name
            );
        }
    }
    let declared = chelis_types::builtin_discovery::builtin_semantic_identities().unwrap();
    assert_eq!(selected, declared.into_iter().collect());
    assert_eq!(
        BUILTINS.iter().map(|b| b.name).collect::<BTreeSet<_>>(),
        chelis_types::BUILTIN_NAMES.iter().copied().collect()
    );
    for removed in ["normalize", "jint", "jnum", "jget", "to_json", "parse_json"] {
        assert!(builtin_decl(removed).is_none(), "removed alias {removed}");
    }
}

#[test]
fn overloaded_applications_resolve_one_exact_case() {
    let list = Type::Adt("List".into(), vec![Type::Prim(Prim::F32)]);
    let dict = Type::Adt(
        "Dict".into(),
        vec![Type::Prim(Prim::String), Type::Prim(Prim::F32)],
    );
    for (name, args, expected) in [
        ("len", vec![list.clone()], "Container:len:LenList"),
        ("len", vec![dict.clone()], "Container:len:LenDict"),
        (
            "concat",
            vec![list.clone(), list.clone()],
            "Container:concat:ConcatList",
        ),
        (
            "concat",
            vec![list.clone(), Type::Prim(Prim::Int32)],
            "Container:concat:ConcatTensors",
        ),
        ("drop", vec![list.clone()], "Container:drop:DropValue"),
        (
            "drop",
            vec![list.clone(), Type::Prim(Prim::Int64)],
            "Container:drop:DropList",
        ),
        ("eq", vec![Type::Prim(Prim::F32)], "Numeric:eq:TableA"),
        ("eq", vec![list.clone()], "Container:eq:EqRecursive"),
        ("to_string", vec![list], "Boundary:to_string:ToStringList"),
        ("to_string", vec![dict], "Boundary:to_string:ToStringDict"),
        (
            "to_string",
            vec![Type::Tensor(vec![], TensorPrec::Concrete(Prim::F32))],
            "Boundary:to_string:ToStringTensor",
        ),
    ] {
        assert_eq!(
            builtin_decl(name).unwrap().semantic_case(&args).unwrap(),
            expected
        );
    }
    assert!(
        builtin_decl("len")
            .unwrap()
            .semantic_case(&[Type::Unit])
            .is_err()
    );
    assert!(
        builtin_decl("concat")
            .unwrap()
            .semantic_case(&[Type::Unit])
            .is_err()
    );
    assert!(builtin_decl("drop").unwrap().semantic_case(&[]).is_err());
}

#[test]
fn observation_rejections_have_explicit_cases_not_an_accepted_wildcard() {
    let decl = builtin_decl("to_string").unwrap();
    for (ty, case) in [
        (Type::Unit, "ToStringUnit"),
        (Type::Tuple(vec![]), "ToStringTuple"),
        (
            Type::Adt("Option".into(), vec![Type::Prim(Prim::F32)]),
            "ToStringOption",
        ),
        (Type::Adt("UserValue".into(), vec![]), "ToStringAdt"),
        (Type::Fn(vec![], Box::new(Type::Unit)), "ToStringFunction"),
    ] {
        assert_eq!(
            decl.semantic_case(&[ty]).unwrap(),
            format!("Boundary:to_string:{case}")
        );
    }
}

#[test]
fn checked_applications_and_lexical_shadows_remain_distinct() {
    for source in [
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_00.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_01.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_02.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_03.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_04.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_05.deep"),
    ] {
        let deep = chelis_deep::parse_and_stamp_file(source).expect("core fixture parses");
        let checked =
            chelis_types::check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked);
        assert!(checked.is_ok(), "{source}: {checked:?}");
    }
    for source in [
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_06.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_07.deep"),
    ] {
        let deep = chelis_deep::parse_and_stamp_file(source).expect("core fixture parses");
        assert!(
            chelis_types::check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked)
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn symbolic_observation_preserves_legal_applications_and_rejects_wrong_bounds() {
    for source in [
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_08.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_09.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_10.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_11.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_12.deep"),
    ] {
        let deep = chelis_deep::parse_and_stamp_file(source).expect("core fixture parses");
        assert!(
            chelis_types::check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked).is_ok(),
            "{source}"
        );
    }
    for source in [
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_13.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_14.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_15.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_16.deep"),
        include_str!("fixtures/core/tests/builtin_atom_discovery/application_17.deep"),
    ] {
        let deep = chelis_deep::parse_and_stamp_file(source).expect("core fixture parses");
        let errors =
            chelis_types::check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked)
                .unwrap_err();
        assert!(!errors.errors.is_empty(), "{source}");
        assert!(
            errors
                .errors
                .iter()
                .all(|e| !e.message.contains("owner-stamp")),
            "{source}: {errors:?}"
        );
    }
}

#[test]
fn symbolic_selection_resolves_each_instantiation_without_a_default_case() {
    use chelis_types::builtin_discovery::BuiltinCaseSelection;
    use chelis_types::types::TypeVar;
    use chelis_types::unify::Subst;
    let variable = TypeVar(9000);
    for name in ["eq", "neq", "to_string"] {
        let decl = builtin_decl(name).unwrap();
        let selection = decl
            .semantic_selection(&[Type::Var(variable)], &mut Subst::new())
            .unwrap();
        assert!(matches!(&selection, BuiltinCaseSelection::ByOperand(_)));
        assert!(decl.semantic_case(&[Type::Var(variable)]).is_err());
        for (ty, expected) in [
            (
                Type::Prim(Prim::F32),
                if name == "to_string" {
                    "Boundary:to_string:ToStringScalar".into()
                } else {
                    format!("Numeric:{name}:TableA")
                },
            ),
            (
                Type::Adt("List".into(), vec![Type::Prim(Prim::F32)]),
                match name {
                    "eq" => "Container:eq:EqRecursive",
                    "neq" => "Container:neq:NeqRecursive",
                    _ => "Boundary:to_string:ToStringList",
                }
                .into(),
            ),
        ] {
            let mut subst = Subst::new();
            subst.insert_type(variable, ty).unwrap();
            assert_eq!(
                selection.resolve(&mut subst).unwrap(),
                BuiltinCaseSelection::Resolved(expected)
            );
        }
    }
    // A symbolic len is an obligation, not permission to skip its selector.
    let decl = builtin_decl("len").unwrap();
    let selection = decl
        .semantic_selection(&[Type::Var(variable)], &mut Subst::new())
        .unwrap();
    let mut invalid = Subst::new();
    invalid
        .insert_type(variable, Type::Prim(Prim::F32))
        .unwrap();
    assert!(selection.resolve(&mut invalid).is_err());
}
