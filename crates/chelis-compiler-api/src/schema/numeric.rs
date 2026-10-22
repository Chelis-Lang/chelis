//! Host schema projections for the canonical stored-value codec.
//!
//! These types describe spec/10 section 3.2. They provide no value codec.

use schemars::{JsonSchema, r#gen::SchemaGenerator, schema::Schema};

struct HexBits<const DIGITS: usize>;

impl<const DIGITS: usize> JsonSchema for HexBits<DIGITS> {
    fn schema_name() -> String {
        format!("IeeeBits{DIGITS}")
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        use schemars::schema::{InstanceType, SchemaObject, StringValidation};
        SchemaObject {
            instance_type: Some(InstanceType::String.into()),
            string: Some(Box::new(StringValidation {
                min_length: Some(DIGITS as u32),
                max_length: Some(DIGITS as u32),
                pattern: Some(format!("^[0-9a-f]{{{DIGITS}}}$")),
            })),
            ..Default::default()
        }
        .into()
    }
}

/// The exact hexadecimal schema for a key execution value.
pub enum KeyBitsSchema {}

impl JsonSchema for KeyBitsSchema {
    fn schema_name() -> String {
        "KeyBits".to_string()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        use schemars::schema::{InstanceType, SchemaObject, StringValidation};
        SchemaObject {
            instance_type: Some(InstanceType::String.into()),
            string: Some(Box::new(StringValidation {
                min_length: Some(16),
                max_length: Some(16),
                pattern: Some("^[0-9a-f]{16}$".to_string()),
            })),
            ..Default::default()
        }
        .into()
    }
}

// Schema fields describe the codec. No values use these private types.
#[allow(dead_code)]
#[derive(JsonSchema)]
#[schemars(tag = "dtype", deny_unknown_fields)]
enum ScalarWire {
    #[schemars(rename = "f64")]
    F64 { bits: HexBits<16> },
    #[schemars(rename = "f32")]
    F32 { bits: HexBits<8> },
    #[schemars(rename = "f16")]
    F16 { bits: HexBits<4> },
    #[schemars(rename = "bf16")]
    Bf16 { bits: HexBits<4> },
    #[schemars(rename = "int64")]
    I64 { value: i64 },
    #[schemars(rename = "int32")]
    I32 { value: i32 },
    #[schemars(rename = "int16")]
    I16 { value: i16 },
    #[schemars(rename = "int8")]
    I8 { value: i8 },
    #[schemars(rename = "bool")]
    Bool { value: bool },
}

#[allow(dead_code)]
#[derive(JsonSchema)]
#[schemars(tag = "dtype", deny_unknown_fields)]
enum StorageWire {
    #[schemars(rename = "f64")]
    F64 { bits: Vec<HexBits<16>> },
    #[schemars(rename = "f32")]
    F32 { bits: Vec<HexBits<8>> },
    #[schemars(rename = "f16")]
    F16 { bits: Vec<HexBits<4>> },
    #[schemars(rename = "bf16")]
    Bf16 { bits: Vec<HexBits<4>> },
    #[schemars(rename = "int64")]
    I64 { values: Vec<i64> },
    #[schemars(rename = "int32")]
    I32 { values: Vec<i32> },
    #[schemars(rename = "int16")]
    I16 { values: Vec<i16> },
    #[schemars(rename = "int8")]
    I8 { values: Vec<i8> },
    #[schemars(rename = "bool")]
    Bool { values: Vec<bool> },
    #[schemars(rename = "key")]
    Key { bits: Vec<KeyBitsSchema> },
}

/// The scalar codec schema, without a key literal variant.
pub enum ScalarValueSchema {}

impl JsonSchema for ScalarValueSchema {
    fn schema_name() -> String {
        "ScalarValue".to_string()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        ScalarWire::json_schema(generator)
    }
}

/// The graph storage schema, without execution-only key storage.
pub enum TensorStorageSchema {}

impl JsonSchema for TensorStorageSchema {
    fn schema_name() -> String {
        "TensorStorage".to_string()
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        let mut schema = StorageWire::json_schema(generator).into_object();
        if let Some(variants) = schema
            .subschemas
            .as_mut()
            .and_then(|subschemas| subschemas.one_of.as_mut())
        {
            variants.retain(|variant| !is_key_variant(variant));
        }
        schema.into()
    }
}

fn is_key_variant(variant: &Schema) -> bool {
    let Schema::Object(variant) = variant else {
        return false;
    };
    variant
        .object
        .as_ref()
        .and_then(|object| object.properties.get("dtype"))
        .is_some_and(|dtype| match dtype {
            Schema::Object(dtype) => dtype
                .enum_values
                .as_deref()
                .is_some_and(|values| matches!(values, [value] if value.as_str() == Some("key"))),
            Schema::Bool(_) => false,
        })
}

/// The execution storage schema, including key tensors.
pub fn execution_storage_schema(generator: &mut SchemaGenerator) -> Schema {
    StorageWire::json_schema(generator)
}

/// A field schema with the canonical key definition name.
pub fn key_bits_schema(generator: &mut SchemaGenerator) -> Schema {
    generator.subschema_for::<KeyBitsSchema>()
}
