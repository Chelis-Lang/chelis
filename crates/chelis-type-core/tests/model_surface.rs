use chelis_numeric_core::prim::Prim;
use chelis_type_core::*;

#[test]
fn model_inspection_does_not_need_evidence_traits() {
    struct Evidence;
    let ty = Type::<Evidence>::kinded_adt(
        "Sized".into(),
        vec![
            NominalArg::Type(Type::Unit),
            NominalArg::Dimension(Dim::Lit(3)),
        ],
    );
    assert_eq!(ty.to_string(), "Sized () 3");
    let argument = NominalArg::from(Type::<Evidence>::reference(Type::Unit));
    assert!(argument.as_type().is_some());
    assert!(argument.as_dimension().is_none());
}

#[test]
fn compound_models_preserve_nominal_argument_order_and_shared_children() {
    let ty = Type::<core::convert::Infallible>::function(
        vec![
            Type::reference(Type::Tensor(
                vec![Dim::Name("batch".into())],
                Prim::F32.into(),
            )),
            Type::kinded_adt(
                "Sized".into(),
                vec![
                    NominalArg::Dimension(Dim::Lit(4)),
                    NominalArg::Type(Type::Prim(Prim::Bool)),
                ],
            ),
        ],
        Type::tuple(vec![
            Type::Unit,
            Type::adt("Box".into(), vec![Type::Prim(Prim::Int64)]),
        ]),
    );
    let cloned = ty.clone();
    let (Type::Fn(a, x), Type::Fn(b, y)) = (&ty, &cloned) else {
        panic!("function")
    };
    assert!(std::sync::Arc::ptr_eq(a, b));
    assert!(std::sync::Arc::ptr_eq(x, y));
    assert_eq!(ty, cloned);
}

#[cfg(feature = "serde")]
#[test]
fn scheme_decode_rejects_missing_contract_and_result_origin_fields() {
    #[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
    enum Evidence {}
    let value = serde_json::to_value(Scheme::mono(Type::<Evidence>::Unit)).unwrap();
    for field in ["constraints", "result_origin"] {
        let mut malformed = value.clone();
        malformed.as_object_mut().unwrap().remove(field);
        assert!(
            serde_json::from_value::<Scheme<Evidence>>(malformed).is_err(),
            "{field}"
        );
    }
}

#[cfg(feature = "serde")]
#[test]
fn effect_set_serialization_keeps_canonical_order() {
    let effects: EffectSet = [
        Effect::Resource("gpu".into()),
        Effect::Test,
        Effect::Io,
        Effect::Accum,
    ]
    .into_iter()
    .collect();
    let expected = vec![
        Effect::Accum,
        Effect::Io,
        Effect::Test,
        Effect::Resource("gpu".into()),
    ];
    let bytes = postcard::to_allocvec(&effects).unwrap();
    assert_eq!(bytes, postcard::to_allocvec(&expected).unwrap());
    assert_eq!(postcard::from_bytes::<EffectSet>(&bytes).unwrap(), effects);
}

#[test]
fn every_primitive_retains_exact_precision_identity() {
    let primitives = [
        Prim::F32,
        Prim::F64,
        Prim::F16,
        Prim::Bf16,
        Prim::F8e4m3,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::Bool,
        Prim::String,
        Prim::Key,
    ];
    for first in primitives {
        for second in primitives {
            let mut subst = Subst::<core::convert::Infallible>::new();
            let result = unify(&Type::Prim(first), &Type::Prim(second), &mut subst);
            assert_eq!(result.is_ok(), first == second);
            if let Err(error) = result {
                assert!(matches!(error.kind, TypeErrorKind::PrecisionMismatch));
            }
        }
    }
}

#[test]
fn standalone_evidence_propagates_without_new_evidence() {
    let error = Type::<()>::Error(());
    let mut subst = Subst::<()>::new();
    unify(&error, &Type::Prim(Prim::F32), &mut subst).unwrap();
    unify(&Type::Prim(Prim::F32), &error, &mut subst).unwrap();
    assert_eq!(subst.apply(&error), error);
    assert_eq!(error.to_string(), "<error>");
}
