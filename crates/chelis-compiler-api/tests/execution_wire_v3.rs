//! spec/10 §3.2: one stored-value codec and checked tensor reconstruction.

use chelis_compiler_api::schema::{EvalResult, ExecutionValue, TensorValue};
use serde_json::json;

#[test]
fn execution_scalars_preserve_stored_bits_and_exact_integers() {
    for scalar in [
        json!({"dtype":"f64","bits":"fff0000000000001"}),
        json!({"dtype":"f32","bits":"80000000"}),
        json!({"dtype":"f16","bits":"7c01"}),
        json!({"dtype":"bf16","bits":"ff81"}),
        json!({"dtype":"int64","value":9007199254740993_i64}),
        json!({"dtype":"int32","value":i32::MIN}),
        json!({"dtype":"int16","value":i16::MIN}),
        json!({"dtype":"int8","value":i8::MIN}),
    ] {
        let input = json!({"type":"scalar","value":scalar});
        let decoded: ExecutionValue = serde_json::from_value(input.clone()).unwrap();
        let runtime = chelis_compiler_api::decode_adt_value(&[], &decoded).unwrap();
        assert_eq!(
            serde_json::to_value(runtime.to_execution_value().unwrap()).unwrap(),
            input
        );
        assert_eq!(serde_json::to_value(decoded).unwrap(), input);
    }
    let boolean = json!({"type":"bool","value":true});
    let decoded: ExecutionValue = serde_json::from_value(boolean.clone()).unwrap();
    assert_eq!(serde_json::to_value(decoded).unwrap(), boolean);
}

#[test]
fn execution_rejects_old_scalar_shapes_and_alternate_boolean_encoding() {
    for input in [
        json!({"type":"float64","value":1.0}),
        json!({"type":"int64","value":1}),
        json!({"type":"scalar","value":{"dtype":"bool","value":true}}),
        json!({"type":"scalar","value":{"dtype":"f16","bits":"7C01"}}),
        json!({"type":"scalar","value":{"dtype":"f32","value":1.0}}),
        json!({"type":"scalar","value":{"dtype":"int64","value":1.0}}),
        json!({"type":"scalar","value":{"dtype":"int8","value":128}}),
        json!({"type":"scalar","value":{"dtype":"f64","bits":"0000000000000000"},"extra":1}),
    ] {
        assert!(
            serde_json::from_value::<ExecutionValue>(input.clone()).is_err(),
            "accepted {input}"
        );
    }
}

#[test]
fn tensor_shape_checks_preserve_scalar_and_empty_shapes() {
    for input in [
        json!({"shape":[],"data":{"dtype":"f16","bits":["8000"]}}),
        json!({"shape":[2],"data":{"dtype":"int64","values":[i64::MIN,i64::MAX]}}),
        json!({"shape":[i64::MAX,0,i64::MAX],"data":{"dtype":"bool","values":[]}}),
        json!({"shape":[i64::MAX,i64::MAX,0],"data":{"dtype":"f32","bits":[]}}),
    ] {
        let decoded: TensorValue = serde_json::from_value(input.clone()).unwrap();
        assert_eq!(serde_json::to_value(decoded).unwrap(), input);
    }
}

#[test]
fn tensor_reconstruction_preserves_bits_without_numeric_finalization() {
    for data in [
        json!({"dtype":"f64","bits":["7ff0000000000001","8000000000000000"]}),
        json!({"dtype":"f32","bits":["ff800001","80000000"]}),
        json!({"dtype":"f16","bits":["7c01","8000"]}),
        json!({"dtype":"bf16","bits":["ff81","8000"]}),
        json!({"dtype":"int64","values":[i64::MIN,i64::MAX]}),
    ] {
        let input = json!({"type":"tensor","value":{"shape":[2],"data":data}});
        let decoded = serde_json::from_value(input.clone()).unwrap();
        let runtime = chelis_compiler_api::decode_adt_value(&[], &decoded).unwrap();
        assert_eq!(
            serde_json::to_value(runtime.to_execution_value().unwrap()).unwrap(),
            input
        );
    }
    let input = json!({"type":"tensor","value":{"shape":[i64::MAX,i64::MAX,0],"data":{"dtype":"f32","bits":[]}}});
    let decoded = serde_json::from_value(input.clone()).unwrap();
    let runtime = chelis_compiler_api::decode_adt_value(&[], &decoded).unwrap();
    assert_eq!(
        serde_json::to_value(runtime.to_execution_value().unwrap()).unwrap(),
        input
    );
}

#[test]
fn invalid_in_memory_tensor_cannot_encode_or_reconstruct() {
    let mut value: TensorValue =
        serde_json::from_value(json!({"shape":[0],"data":{"dtype":"int8","values":[]}})).unwrap();
    for shape in [vec![1], vec![-1], vec![i64::MAX, 2]] {
        value.shape = shape;
        assert!(serde_json::to_value(&value).is_err());
        assert!(
            chelis_compiler_api::decode_adt_value(
                &[],
                &ExecutionValue::Tensor {
                    value: value.clone()
                }
            )
            .is_err()
        );
    }
}

#[test]
fn execution_schema_excludes_boolean_scalar_and_bounds_shape_extents() {
    let schema = serde_json::to_value(schemars::schema_for!(ExecutionValue)).unwrap();
    let variants = schema["definitions"]["NumericScalar"]["oneOf"]
        .as_array()
        .unwrap();
    assert_eq!(variants.len(), 8);
    assert!(
        variants
            .iter()
            .all(|variant| variant["properties"]["dtype"]["enum"] != json!(["bool"]))
    );
    let shape = &schema["definitions"]["TensorValue"]["properties"]["shape"];
    assert_eq!(shape["maxItems"], json!(i32::MAX));
    assert_eq!(shape["items"]["minimum"], json!(0.0));
    assert_eq!(shape["items"]["format"], json!("int64"));
}

#[test]
fn tensor_shape_rejects_negative_overflow_and_inconsistent_counts() {
    for shape in [
        json!([-1]),
        json!([u64::MAX]),
        json!([1.0]),
        json!([i64::MAX, 2]),
        json!([]),
        json!([1]),
    ] {
        let input = json!({"shape":shape,"data":{"dtype":"int8","values":[]}});
        assert!(
            serde_json::from_value::<TensorValue>(input.clone()).is_err(),
            "accepted {input}"
        );
    }
    assert!(
        serde_json::from_value::<TensorValue>(
            json!({"shape":[1],"data":{"dtype":"f16","values":[1.0]}})
        )
        .is_err()
    );
}

#[test]
fn execution_envelope_accepts_only_version_three() {
    assert!(serde_json::from_value::<EvalResult>(json!({"schema_version":3,"roots":[]})).is_ok());
    for input in [
        json!({"roots":[]}),
        json!({"schema_version":2,"roots":[]}),
        json!({"schema_version":4,"roots":[]}),
    ] {
        assert!(serde_json::from_value::<EvalResult>(input).is_err());
    }
}
