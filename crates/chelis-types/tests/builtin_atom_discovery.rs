//! #1294: independently defend declaration and checked-shape discovery.
use chelis_types::types::{Prim, TensorPrec, Type};
use chelis_types::{
    BUILTINS, BuiltinCapabilityDecl, BuiltinSemanticDomain as Domain, BuiltinSiblingCaseDecl,
    BuiltinSiblingCaseId as Case, builtin_decl,
};
use std::collections::BTreeSet;

#[test]
fn every_declared_case_is_a_live_enum_member_and_every_member_is_declared() {
    let source = syn::parse_file(include_str!("../src/builtins.rs")).expect("valid Rust");
    let domains: BTreeSet<_> = source
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Enum(e) if e.ident == "BuiltinSemanticDomain" => {
                Some(e.variants.iter().map(|v| v.ident.to_string()).collect())
            }
            _ => None,
        })
        .expect("closed semantic domains");
    assert_eq!(
        domains,
        ["Numeric", "Container", "Boundary"]
            .map(str::to_string)
            .into_iter()
            .collect()
    );
    let members: BTreeSet<_> = source
        .items
        .iter()
        .find_map(|item| match item {
            syn::Item::Enum(e) if e.ident == "BuiltinSiblingCaseId" => {
                Some(e.variants.iter().map(|v| v.ident.to_string()).collect())
            }
            _ => None,
        })
        .expect("closed case enum");
    let mut declared = BTreeSet::new();
    for builtin in BUILTINS {
        builtin.capability.validate().expect("valid declaration");
        for case in builtin.capability.sibling_cases {
            declared.insert(format!("{:?}", case.case));
        }
    }
    assert_eq!(members, declared, "stale or missing sibling case");
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
            "skip" => vec![vec![list.clone(), Type::Prim(Prim::Int64)]],
            "drop" => vec![vec![list.clone()]],
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
            "skip",
            vec![list.clone(), Type::Prim(Prim::Int64)],
            "Container:skip:SkipList",
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
    // The arity selector is gone with the [05-OP-54]/[05-OP-67] split:
    // `drop` and `skip` are separate declarations with one sibling case
    // each, so neither reads its operands to choose an identity. Before
    // the split, `drop` with no arguments was the third arity and had no
    // exact case.
    assert_eq!(
        builtin_decl("drop").unwrap().semantic_case(&[]).unwrap(),
        "Container:drop:DropValue"
    );
    assert_eq!(
        builtin_decl("skip").unwrap().semantic_case(&[]).unwrap(),
        "Container:skip:SkipList"
    );
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
    use chelis_surf::{desugar::desugar_program, parser::parse_str};
    for source in [
        "a = len([1, 2])",
        "a = concat([1, 2], [3])",
        "a = skip([1, 2], 1i64)",
        "a = to_string([1, 2])",
        "def apply(len: (f32 -> f32), x: f32) -> f32 = len(x)",
        "def apply(to_string: (f32 -> f32), x: f32) -> f32 = to_string(x)",
    ] {
        let deep = desugar_program(&parse_str(source).unwrap()).expect("Surf fixture must desugar");
        let checked = chelis_types::check_typed_program(&deep);
        assert!(checked.is_ok(), "{source}: {checked:?}");
    }
    for source in ["a = len(1)", "a = concat([1], 1.0)"] {
        let deep = desugar_program(&parse_str(source).unwrap()).expect("Surf fixture must desugar");
        assert!(
            chelis_types::check_typed_program(&deep).is_err(),
            "{source}"
        );
    }
}

#[test]
fn symbolic_observation_preserves_legal_applications_and_rejects_wrong_bounds() {
    use chelis_surf::{desugar::desugar_program, parser::parse_str};
    for source in [
        "result = map(fn(x) -> to_string(x), [1.0])",
        "sig render[p: Float]: p -> string\ndef render(x) = to_string(x)\nresult = render(1.0f32)",
        "sig render[p: Int]: p -> string\ndef render(x) = to_string(x)\nresult = render(1i64)",
        "sig render[p: Numeric]: p -> string\ndef render(x) = to_string(x)\nresult = render(1i64)",
        "def equal(x,y) = eq(x,y)\na = equal(1,1)\nb = equal([1],[1])",
    ] {
        let deep = desugar_program(&parse_str(source).unwrap()).expect("Surf fixture must desugar");
        assert!(chelis_types::check_typed_program(&deep).is_ok(), "{source}");
    }
    for source in [
        "sig render[p: Float]: p -> string\ndef render(x) = to_string(x)\nresult = render(1i64)",
        "sig render[p: Int]: p -> string\ndef render(x) = to_string(x)\nresult = render(1.0f32)",
        "result = map(fn(x) -> len(x), [1.0])",
        "sig size[p: Int]: p -> i64\ndef size(x) = len(x)\nresult = size(1i64)",
        "def equal(x,y) = eq(x,y)\nresult = equal(1,[1])",
    ] {
        let deep = desugar_program(&parse_str(source).unwrap()).expect("Surf fixture must desugar");
        let errors = chelis_types::check_typed_program(&deep).unwrap_err();
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
            .semantic_selection(&[Type::Var(variable)], &Subst::new())
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
                selection.resolve(&subst).unwrap(),
                BuiltinCaseSelection::Resolved(expected)
            );
        }
    }
    // A symbolic len is an obligation, not permission to skip its selector.
    let decl = builtin_decl("len").unwrap();
    let selection = decl
        .semantic_selection(&[Type::Var(variable)], &Subst::new())
        .unwrap();
    let mut invalid = Subst::new();
    invalid
        .insert_type(variable, Type::Prim(Prim::F32))
        .unwrap();
    assert!(selection.resolve(&invalid).is_err());
}
