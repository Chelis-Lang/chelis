//! spec/10 §3.3: source tuple indices and vmap axes retain lexical int64
//! spelling. Transport does not normalize an axis or admit executable syntax.
use chelis_compiler_api::schema::WireSurfExpr;
use chelis_types::types::Prim;
use serde_json::json;

fn expression(kind: &str, field: &str, value: serde_json::Value) -> serde_json::Value {
    let mut source = json!({
        "kind":kind,
        "expr":{"kind":"var","name":"x","span":{"offset":0,"len":1}},
        "span":{"offset":0,"len":20}
    });
    source[field] = value;
    source
}

#[test]
fn raw_source_parameters_preserve_every_int64_before_source_admission() {
    for value in [i64::MIN, -1, 0, 9_007_199_254_740_993, i64::MAX] {
        for (kind, field) in [("tuple_get", "index"), ("vmap", "axis")] {
            let json = expression(kind, field, json!(value));
            let decoded: WireSurfExpr = serde_json::from_value(json.clone()).unwrap();
            let number = match &decoded {
                WireSurfExpr::TupleGet { index, .. } => *index,
                WireSurfExpr::Vmap {
                    axis: Some(axis), ..
                } => *axis,
                _ => panic!("source variant must remain distinct"),
            };
            assert_eq!(number.scalar().prim(), Prim::Int64);
            assert_eq!(number.get(), value);
            assert_eq!(serde_json::to_value(decoded).unwrap(), json);
        }
    }
    let json = expression("vmap", "axis", json!(null));
    let decoded: WireSurfExpr = serde_json::from_value(json.clone()).unwrap();
    assert!(matches!(decoded, WireSurfExpr::Vmap { axis: None, .. }));
    assert_eq!(serde_json::to_value(decoded).unwrap(), json);
}

#[test]
fn source_parameters_reject_alternate_numeric_encodings_and_overflow() {
    for value in [
        json!(1.0),
        json!(u64::MAX),
        json!("1"),
        json!(true),
        json!({"dtype":"int64","value":1}),
    ] {
        for (kind, field) in [("tuple_get", "index"), ("vmap", "axis")] {
            assert!(
                serde_json::from_value::<WireSurfExpr>(expression(kind, field, value.clone()))
                    .is_err()
            );
        }
    }
    assert!(
        serde_json::from_value::<WireSurfExpr>(expression("tuple_get", "index", json!(null)))
            .is_err()
    );
}
