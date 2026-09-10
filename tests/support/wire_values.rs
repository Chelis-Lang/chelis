//! Exact spec/10 stored-value fixtures shared by Rust consumer tests.
#![allow(dead_code)]

// The including module imports ExecutionValue so these same fixtures work
// inside the API crate's unit tests and in external consumer test targets.
use super::ExecutionValue;
use chelis_types::{TensorStorage, types::Prim};
use serde_json::json;

pub fn scalar_f32(value: f32) -> ExecutionValue {
    serde_json::from_value(json!({"type":"scalar", "value":{
        "dtype":"f32", "bits":format!("{:08x}", value.to_bits())
    }}))
    .unwrap()
}

pub fn scalar_f64(value: f64) -> ExecutionValue {
    serde_json::from_value(json!({"type":"scalar", "value":{
        "dtype":"f64", "bits":format!("{:016x}", value.to_bits())
    }}))
    .unwrap()
}

pub fn scalar_integer(prim: Prim, value: i64) -> ExecutionValue {
    ExecutionValue::Scalar {
        value: chelis_types::scalar_from_i64("wire-test", prim, value)
            .unwrap()
            .try_into()
            .unwrap(),
    }
}

pub fn storage_f32(values: Vec<f32>) -> TensorStorage {
    serde_json::from_value(json!({"dtype":"f32", "bits":values.into_iter()
        .map(|value| format!("{:08x}", value.to_bits())).collect::<Vec<_>>() }))
    .unwrap()
}

pub fn storage_f64(values: Vec<f64>) -> TensorStorage {
    serde_json::from_value(json!({"dtype":"f64", "bits":values.into_iter()
        .map(|value| format!("{:016x}", value.to_bits())).collect::<Vec<_>>() }))
    .unwrap()
}

pub fn storage_i64(values: Vec<i64>) -> TensorStorage {
    serde_json::from_value(json!({"dtype":"int64", "values":values})).unwrap()
}
