//! spec/10 §3.4 and [04-NUM-11]: observed extents are nonnegative int64.
use chelis_compiler_api::{compiler::ExecutionDim, schema::WireInferredDim};
use serde_json::json;

#[test]
fn metadata_extents_preserve_zero_absence_and_int64_maximum() {
    for size in [0, i64::MAX] {
        let value = json!({"name":"batch","size":size});
        let dim: ExecutionDim = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(dim).unwrap(), value);
        let value = json!({"kind":"lit","size":size});
        let dim: WireInferredDim = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(dim).unwrap(), value);
    }
    let absent: ExecutionDim = serde_json::from_value(json!({"name":"batch"})).unwrap();
    assert!(absent.size.is_none());
}

#[test]
fn observed_extents_reject_negative_fractional_and_out_of_range_values() {
    for size in [json!(-1), json!(1.0), json!(u64::MAX), json!("1")] {
        assert!(serde_json::from_value::<ExecutionDim>(json!({"name":null,"size":size})).is_err());
        assert!(
            serde_json::from_value::<WireInferredDim>(json!({"kind":"lit","size":size})).is_err()
        );
    }
}

#[test]
fn extent_carrier_requires_exact_int64_and_checks_all_ingress() {
    use chelis_compiler_api::schema::numbers::NonnegativeExtent;
    use chelis_types::{scalar_from_f64, scalar_from_i64, types::Prim};
    for value in [0, 9_007_199_254_740_993, i64::MAX] {
        let extent = NonnegativeExtent::new(value).unwrap();
        assert_eq!(extent.scalar().prim(), Prim::Int64);
        assert_eq!(extent.get(), value);
        assert_eq!(serde_json::to_string(&extent).unwrap(), value.to_string());
        let decoded: NonnegativeExtent =
            bincode::deserialize(&bincode::serialize(&extent).unwrap()).unwrap();
        assert_eq!(decoded.get(), value);
    }
    assert!(NonnegativeExtent::new(-1).is_err());
    assert!(
        NonnegativeExtent::try_from(scalar_from_i64("fixture", Prim::Int64, -1).unwrap()).is_err()
    );
    for wrong in [
        scalar_from_i64("fixture", Prim::Int32, 1).unwrap(),
        scalar_from_f64("fixture", Prim::F64, 1.0).unwrap(),
    ] {
        assert!(NonnegativeExtent::try_from(wrong).is_err());
    }
    assert!(
        bincode::deserialize::<NonnegativeExtent>(&bincode::serialize(&-1_i64).unwrap()).is_err()
    );
    if usize::BITS == 64 {
        assert!(NonnegativeExtent::try_from(usize::MAX).is_err());
    }
}
