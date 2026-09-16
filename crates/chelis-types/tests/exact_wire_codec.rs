//! spec/10 §3.2: transport preserves stored bits and admits exactly one grammar.

use chelis_types::{ElementRef, ScalarValue, TensorStorage};
use serde_json::{Value, json};

fn assert_scalar(wire: Value) {
    let scalar: ScalarValue = serde_json::from_value(wire.clone()).expect("canonical scalar");
    assert_eq!(
        scalar.prim().interchange_name(),
        wire["dtype"].as_str().unwrap()
    );
    assert_element(scalar.element_ref(), &wire["bits"], &wire["value"]);
    assert_eq!(serde_json::to_value(scalar).expect("encode scalar"), wire);
    let binary = bincode::serialize(&scalar).expect("cache encode");
    let decoded: ScalarValue = bincode::deserialize(&binary).expect("cache decode");
    assert_eq!(
        serde_json::to_value(decoded).expect("encode cached scalar"),
        wire
    );
}

fn assert_storage(wire: Value) {
    let storage: TensorStorage = serde_json::from_value(wire.clone()).expect("canonical storage");
    assert_eq!(
        storage.prim().interchange_name(),
        wire["dtype"].as_str().unwrap()
    );
    let expected = wire
        .get("bits")
        .unwrap_or(&wire["values"])
        .as_array()
        .unwrap();
    assert_eq!(storage.len(), expected.len());
    for (index, value) in expected.iter().enumerate() {
        assert_element(storage.element_ref(index), value, value);
    }
    assert_eq!(
        serde_json::to_value(&storage).expect("encode storage"),
        wire
    );
    let binary = bincode::serialize(&storage).expect("cache encode");
    let decoded: TensorStorage = bincode::deserialize(&binary).expect("cache decode");
    assert_eq!(
        serde_json::to_value(decoded).expect("encode cached storage"),
        wire
    );
}

fn assert_element(element: ElementRef, bits: &Value, value: &Value) {
    // Check the actual own-width storage independently of the encoder. A pair
    // of inverse codec bugs cannot hide behind a successful JSON round trip.
    let actual_bits = match element {
        ElementRef::F64(v) => Some(v.to_bits()),
        ElementRef::F32(v) => Some(u64::from(v.to_bits())),
        ElementRef::F16(v) => Some(u64::from(v.to_bits())),
        ElementRef::Bf16(v) => Some(u64::from(v.to_bits())),
        ElementRef::I64(v) => {
            assert_eq!(Some(v), value.as_i64());
            None
        }
        ElementRef::I32(v) => {
            assert_eq!(Some(i64::from(v)), value.as_i64());
            None
        }
        ElementRef::I16(v) => {
            assert_eq!(Some(i64::from(v)), value.as_i64());
            None
        }
        ElementRef::I8(v) => {
            assert_eq!(Some(i64::from(v)), value.as_i64());
            None
        }
        ElementRef::Bool(v) => {
            assert_eq!(Some(v), value.as_bool());
            None
        }
    };
    if let Some(actual) = actual_bits {
        assert_eq!(
            actual,
            u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap()
        );
    }
}

#[test]
fn positional_cache_decoder_checks_the_same_bit_grammar() {
    // An independently constructed discriminant/string pair for F64. The
    // versioned cache owns this positional layout; it cannot bypass HexBits.
    let good = bincode::serialize(&(0u32, "8000000000000000")).unwrap();
    let scalar: ScalarValue = bincode::deserialize(&good).unwrap();
    assert_element(
        scalar.element_ref(),
        &json!("8000000000000000"),
        &Value::Null,
    );
    for invalid in [
        "800000000000000",
        "80000000000000000",
        "FFF0000000000000",
        "0x00000000000000",
    ] {
        let bad = bincode::serialize(&(0u32, invalid)).unwrap();
        assert!(
            bincode::deserialize::<ScalarValue>(&bad).is_err(),
            "accepted {invalid}"
        );
    }
}

#[test]
fn published_schema_has_fixed_width_bits_and_closed_payload_shapes() {
    let scalar = serde_json::to_value(schemars::schema_for!(ScalarValue)).unwrap();
    let storage = serde_json::to_value(schemars::schema_for!(TensorStorage)).unwrap();
    for schema in [scalar, storage] {
        for digits in [4, 8, 16] {
            let definition = &schema["definitions"][format!("IeeeBits{digits}")];
            assert_eq!(definition["type"], "string");
            assert_eq!(definition["minLength"], digits);
            assert_eq!(definition["maxLength"], digits);
            assert_eq!(definition["pattern"], format!("^[0-9a-f]{{{digits}}}$"));
        }
        let variants = schema["oneOf"].as_array().unwrap();
        assert_eq!(variants.len(), 9);
        for variant in variants {
            assert_eq!(variant["additionalProperties"], false);
            assert_eq!(variant["required"].as_array().unwrap().len(), 2);
        }
    }
}

#[test]
fn scalar_and_storage_preserve_ieee_special_patterns_at_each_width() {
    for (dtype, patterns) in [
        (
            "f64",
            vec![
                "0000000000000000",
                "8000000000000000",
                "0000000000000001",
                "3ff0000000000000",
                "7ff0000000000000",
                "fff0000000000000",
                "7ff0000000000001",
                "7ff8000000001234",
                "fff0000000001234",
            ],
        ),
        (
            "f32",
            vec![
                "00000000", "80000000", "00000001", "3f800000", "7f800000", "ff800000", "7f800001",
                "7fc01234", "ff801234",
            ],
        ),
        (
            "f16",
            vec![
                "0000", "8000", "0001", "3c00", "7c00", "fc00", "7c01", "7e12", "fc12",
            ],
        ),
        (
            "bf16",
            vec![
                "0000", "8000", "0001", "3f80", "7f80", "ff80", "7f81", "7fc2", "ff92",
            ],
        ),
    ] {
        for bits in &patterns {
            assert_scalar(json!({"dtype": dtype, "bits": bits}));
        }
        assert_storage(json!({"dtype": dtype, "bits": patterns}));
        assert_storage(json!({"dtype": dtype, "bits": []}));
    }
}

#[test]
fn all_reduced_float_patterns_survive_without_nan_or_zero_normalization() {
    for dtype in ["f16", "bf16"] {
        let patterns: Vec<_> = (0u32..=65535).map(|bits| format!("{bits:04x}")).collect();
        assert_storage(json!({"dtype": dtype, "bits": patterns}));
    }
}

#[test]
fn exact_integer_widths_and_booleans_use_their_own_payload_members() {
    for (dtype, values) in [
        ("int8", vec![i8::MIN as i64, 0, i8::MAX as i64]),
        ("int16", vec![i16::MIN as i64, 0, i16::MAX as i64]),
        ("int32", vec![i32::MIN as i64, 0, i32::MAX as i64]),
        (
            "int64",
            vec![i64::MIN, -9007199254740993, 0, 9007199254740993, i64::MAX],
        ),
    ] {
        for value in &values {
            assert_scalar(json!({"dtype": dtype, "value": value}));
        }
        assert_storage(json!({"dtype": dtype, "values": values}));
    }
    assert_scalar(json!({"dtype": "bool", "value": true}));
    assert_scalar(json!({"dtype": "bool", "value": false}));
    assert_storage(json!({"dtype": "bool", "values": [false, true]}));
}

#[test]
fn scalar_rejects_alternate_encodings_unknown_dtypes_and_payload_members() {
    for wire in [
        json!({"dtype":"f64", "bits":"000000000000000"}),
        json!({"dtype":"f64", "bits":"00000000000000000"}),
        json!({"dtype":"f32", "bits":"3F800000"}),
        json!({"dtype":"f16", "bits":"0x00"}),
        json!({"dtype":"f16", "bits":" 000"}),
        json!({"dtype":"bf16", "bits":"-000"}),
        json!({"dtype":"f64", "bits":1.0}),
        json!({"dtype":"f64", "bits":null}),
        json!({"dtype":"f64", "value":1.0}),
        json!({"dtype":"f64", "bits":"0000000000000000", "value":0}),
        json!({"dtype":"f8e4m3", "bits":"00"}),
        json!({"dtype":0, "bits":"0000000000000000"}),
        json!({"dtype":"int64", "value":9007199254740992.0}),
        json!({"dtype":"int64", "value":"9007199254740993"}),
        json!({"dtype":"int64", "value":u64::MAX}),
        json!({"dtype":"int8", "value":128}),
        json!({"dtype":"int8", "value":-129}),
        json!({"dtype":"int16", "value":32768}),
        json!({"dtype":"int32", "value":2147483648i64}),
        json!({"dtype":"bool", "value":1}),
        json!({"dtype":"int64", "value":true}),
        json!({"F64": 1.0}),
    ] {
        assert!(
            serde_json::from_value::<ScalarValue>(wire.clone()).is_err(),
            "accepted {wire}"
        );
    }
    for wire in [
        r#"{"dtype":"int64","value":1,"value":2}"#,
        r#"{"dtype":"f64","dtype":"int64","value":2}"#,
    ] {
        assert!(
            serde_json::from_str::<ScalarValue>(wire).is_err(),
            "accepted duplicate {wire}"
        );
    }
}

#[test]
fn storage_rejects_mixed_elements_old_float_images_and_extra_fields() {
    for wire in [
        json!({"dtype":"f16", "bits":["3c00", 1.0]}),
        json!({"dtype":"f32", "bits":["00000000", "0000"]}),
        json!({"dtype":"f64", "values":[1.0]}),
        json!({"dtype":"f64", "bits":[], "values":[]}),
        json!({"dtype":"int64", "values":[1, 2.0]}),
        json!({"dtype":"int8", "values":[128]}),
        json!({"dtype":"bool", "values":[false, 1]}),
        json!({"dtype":"f8e4m3", "bits":[]}),
        json!({"F16":[1.0]}),
    ] {
        assert!(
            serde_json::from_value::<TensorStorage>(wire.clone()).is_err(),
            "accepted {wire}"
        );
    }
}
