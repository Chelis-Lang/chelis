//! Scalar transport also works without the alloc feature.
#![cfg(feature = "serde")]

use chelis_numeric_core::dtype_semantics::{KeyBits, RandomKey, ScalarValue, scalar_from_i64};
use chelis_numeric_core::named_cast::NamedCastMode;
use chelis_numeric_core::prim::Prim;

#[test]
fn scalar_wire_preserves_json_and_positional_bytes() {
    for text in [
        r#"{"dtype":"f64","bits":"8000000000000000"}"#,
        r#"{"dtype":"f32","bits":"7f800001"}"#,
        r#"{"dtype":"f16","bits":"fc12"}"#,
        r#"{"dtype":"bf16","bits":"7fc2"}"#,
        r#"{"dtype":"int64","value":-9007199254740993}"#,
        r#"{"dtype":"int32","value":-2147483648}"#,
        r#"{"dtype":"int16","value":32767}"#,
        r#"{"dtype":"int8","value":-128}"#,
        r#"{"dtype":"bool","value":true}"#,
    ] {
        let scalar: ScalarValue = serde_json::from_str(text).unwrap();
        assert_eq!(serde_json::to_string(&scalar).unwrap(), text);
        let bytes = bincode::serialize(&scalar).unwrap();
        let decoded: ScalarValue = bincode::deserialize(&bytes).unwrap();
        assert_eq!(serde_json::to_string(&decoded).unwrap(), text);
    }
    let bytes = bincode::serialize(&(0_u32, "8000000000000000")).unwrap();
    let scalar: ScalarValue = bincode::deserialize(&bytes).unwrap();
    assert_eq!(bincode::serialize(&scalar).unwrap(), bytes);
}

#[test]
fn scalar_wire_accepts_fields_before_dtype_and_rejects_alternate_payloads() {
    let scalar: ScalarValue = serde_json::from_str(r#"{"value":1,"dtype":"int8"}"#).unwrap();
    assert_eq!(scalar.as_i64_exact(), Some(1));
    for text in [
        r#"{"dtype":"f16","bits":"00000000"}"#,
        r#"{"dtype":"f64","bits":"FFF0000000000000"}"#,
        r#"{"dtype":"int8","value":128}"#,
        r#"{"dtype":"int64","value":1.0}"#,
        r#"{"dtype":"bool","value":1}"#,
        r#"{"dtype":"int8","value":1,"bits":"0000"}"#,
        r#"{"dtype":"int8","value":1,"value":2}"#,
        r#"{"dtype":"int8","dtype":"int16","value":1}"#,
        r#"{"dtype":"int8","value":1,"extra":0}"#,
    ] {
        assert!(serde_json::from_str::<ScalarValue>(text).is_err(), "{text}");
    }
}

#[test]
fn key_wire_keeps_exact_bits_without_a_numeric_literal() {
    let seed = scalar_from_i64("test", Prim::Int64, -1).unwrap();
    let key = RandomKey::from_seed(seed).unwrap();
    let bits = KeyBits::new(key);
    assert_eq!(
        serde_json::to_string(&bits).unwrap(),
        r#""ffffffffffffffff""#
    );
    let decoded: KeyBits = serde_json::from_str(r#""ffffffffffffffff""#).unwrap();
    assert_eq!(decoded.key(), key);
    assert!(serde_json::from_str::<KeyBits>(r#""FFFFFFFFFFFFFFFF""#).is_err());
    assert!(serde_json::to_string(&ScalarValue::from_key(key)).is_err());
}

#[test]
fn named_cast_wire_keeps_variant_order_and_names() {
    for (index, mode) in NamedCastMode::ALL.iter().enumerate() {
        assert_eq!(
            bincode::serialize(mode).unwrap(),
            (index as u32).to_le_bytes()
        );
        assert_eq!(
            serde_json::from_str::<NamedCastMode>(&serde_json::to_string(mode).unwrap()).unwrap(),
            *mode
        );
    }
    assert_eq!(
        serde_json::to_string(&NamedCastMode::Trunc).unwrap(),
        "\"Trunc\""
    );
    assert_eq!(
        serde_json::to_string(&NamedCastMode::Saturate).unwrap(),
        "\"Saturate\""
    );
    assert_eq!(
        serde_json::to_string(&NamedCastMode::Wrap).unwrap(),
        "\"Wrap\""
    );
}
