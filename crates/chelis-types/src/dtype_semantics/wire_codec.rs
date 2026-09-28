//! Exact stored-value transport under spec/10 §3.2.
//!
//! JSON and positional caches share the same payload definitions. Cache serde
//! uses an external discriminant because bincode cannot decode internally tagged
//! maps. Both forms preserve bits and use the same strict payload admission.

use super::{Bits, Buf, RandomKey, ScalarValue, TensorStorage};
use serde::{Deserialize, Serialize};

const KEY_HAS_NO_LITERAL_CARRIER: &str =
    "a random key has no literal carrier: every key in a graph is a node's output";

#[derive(Clone, Serialize)]
#[serde(transparent)]
struct HexBits<const DIGITS: usize>(String);

impl<const DIGITS: usize> HexBits<DIGITS> {
    fn from_bits(bits: u64) -> Self {
        let text = format!("{bits:0DIGITS$x}");
        assert_eq!(
            text.len(),
            DIGITS,
            "stored bits must fit their declared width"
        );
        Self(text)
    }

    fn bits(self) -> u64 {
        // Constructors and Deserialize admit exactly the lowercase fixed-width
        // grammar. Accumulation is an exact bit move, without a float conversion.
        self.0.bytes().fold(0, |bits, digit| {
            (bits << 4)
                | u64::from(if digit <= b'9' {
                    digit - b'0'
                } else {
                    digit - b'a' + 10
                })
        })
    }
}

impl<'de, const DIGITS: usize> Deserialize<'de> for HexBits<DIGITS> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        if text.len() != DIGITS
            || !text
                .bytes()
                .all(|digit| digit.is_ascii_digit() || (b'a'..=b'f').contains(&digit))
        {
            return Err(serde::de::Error::custom(format!(
                "stored IEEE bits require exactly {DIGITS} lowercase hexadecimal digits"
            )));
        }
        Ok(Self(text))
    }
}

impl<const DIGITS: usize> schemars::JsonSchema for HexBits<DIGITS> {
    fn schema_name() -> String {
        format!("IeeeBits{DIGITS}")
    }

    fn json_schema(_: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
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

/// A key's 64 bits as spec/10 section 3.2's `h`: exactly 16 lowercase
/// hexadecimal digits, most significant first. Its own type, not
/// `HexBits<16>`, so a key never reads as IEEE bits.
#[derive(Clone, Serialize)]
#[serde(transparent)]
struct KeyHex(String);

impl KeyHex {
    fn from_key(key: RandomKey) -> Self {
        Self(format!("{:016x}", key.bits))
    }

    fn key(&self) -> RandomKey {
        // Deserialize admits exactly the lowercase 16-digit grammar.
        let bits = self.0.bytes().fold(0_u64, |bits, digit| {
            (bits << 4)
                | u64::from(if digit <= b'9' {
                    digit - b'0'
                } else {
                    digit - b'a' + 10
                })
        });
        RandomKey { bits }
    }
}

impl<'de> Deserialize<'de> for KeyHex {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        if text.len() != 16
            || !text
                .bytes()
                .all(|digit| digit.is_ascii_digit() || (b'a'..=b'f').contains(&digit))
        {
            return Err(serde::de::Error::custom(
                "a key's bits require exactly 16 lowercase hexadecimal digits",
            ));
        }
        Ok(Self(text))
    }
}

impl schemars::JsonSchema for KeyHex {
    fn schema_name() -> String {
        "KeyBits".to_string()
    }

    fn json_schema(_: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
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

/// spec/10 section 3.2's scalar key carrier, the `h` of the execution value
/// `{"type":"key","bits":h}`. A key is not a number, so this is not a
/// [`ScalarValue`] payload and has no scalar object of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyBits(RandomKey);

impl KeyBits {
    pub fn new(key: RandomKey) -> Self {
        Self(key)
    }

    pub fn key(self) -> RandomKey {
        self.0
    }
}

impl Serialize for KeyBits {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        KeyHex::from_key(self.0).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for KeyBits {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self(KeyHex::deserialize(deserializer)?.key()))
    }
}

impl schemars::JsonSchema for KeyBits {
    fn schema_name() -> String {
        KeyHex::schema_name()
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        KeyHex::json_schema(generator)
    }
}

// Author each dtype/payload relationship once. The cache mirror changes only
// the enclosing discriminant representation; it has no alternate numeric path.
macro_rules! wire_enum {
    ($wire:ident, $binary:ident { $($variant:ident ($dtype:literal) { $field:ident: $ty:ty }),* $(,)? }) => {
        #[derive(Serialize, Deserialize)]
        #[derive(schemars::JsonSchema)]
        #[serde(tag = "dtype", deny_unknown_fields)]
        enum $wire {
            $(#[serde(rename = $dtype)] $variant { $field: $ty }),*
        }

        #[derive(Serialize, Deserialize)]
        enum $binary {
            $($variant { $field: $ty }),*
        }

        impl $wire {
            fn encode<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                if serializer.is_human_readable() {
                    self.serialize(serializer)
                } else {
                    let value = match self {
                        $(Self::$variant { $field } => $binary::$variant { $field: $field.clone() }),*
                    };
                    value.serialize(serializer)
                }
            }

            fn decode<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                if deserializer.is_human_readable() {
                    Self::deserialize(deserializer)
                } else {
                    Ok(match $binary::deserialize(deserializer)? {
                        $($binary::$variant { $field } => Self::$variant { $field }),*
                    })
                }
            }
        }
    };
}

wire_enum!(ScalarWire, BinaryScalarWire {
    F64("f64") { bits: HexBits<16> },
    F32("f32") { bits: HexBits<8> },
    F16("f16") { bits: HexBits<4> },
    Bf16("bf16") { bits: HexBits<4> },
    I64("int64") { value: i64 },
    I32("int32") { value: i32 },
    I16("int16") { value: i16 },
    I8("int8") { value: i8 },
    Bool("bool") { value: bool },
});

wire_enum!(StorageWire, BinaryStorageWire {
    F64("f64") { bits: Vec<HexBits<16>> },
    F32("f32") { bits: Vec<HexBits<8>> },
    F16("f16") { bits: Vec<HexBits<4>> },
    Bf16("bf16") { bits: Vec<HexBits<4>> },
    I64("int64") { values: Vec<i64> },
    I32("int32") { values: Vec<i32> },
    I16("int16") { values: Vec<i16> },
    I8("int8") { values: Vec<i8> },
    Bool("bool") { values: Vec<bool> },
    // Execution values only: a graph holds no key literal, so
    // `TensorStorage`'s own codec refuses this variant.
    Key("key") { bits: Vec<KeyHex> },
});

impl schemars::JsonSchema for ScalarValue {
    fn schema_name() -> String {
        "ScalarValue".to_string()
    }
    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        ScalarWire::json_schema(generator)
    }
}

impl schemars::JsonSchema for TensorStorage {
    fn schema_name() -> String {
        "TensorStorage".to_string()
    }
    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
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

fn is_key_variant(variant: &schemars::schema::Schema) -> bool {
    let schemars::schema::Schema::Object(variant) = variant else {
        return false;
    };
    variant
        .object
        .as_ref()
        .and_then(|object| object.properties.get("dtype"))
        .is_some_and(|dtype| match dtype {
            schemars::schema::Schema::Object(dtype) => dtype
                .enum_values
                .as_deref()
                .is_some_and(|values| matches!(values, [value] if value.as_str() == Some("key"))),
            schemars::schema::Schema::Bool(_) => false,
        })
}

impl Serialize for ScalarValue {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.bits {
            Bits::F64(v) => ScalarWire::F64 {
                bits: HexBits::from_bits(v.to_bits()),
            },
            Bits::F32(v) => ScalarWire::F32 {
                bits: HexBits::from_bits(u64::from(v.to_bits())),
            },
            Bits::F16(v) => ScalarWire::F16 {
                bits: HexBits::from_bits(u64::from(v.to_bits())),
            },
            Bits::Bf16(v) => ScalarWire::Bf16 {
                bits: HexBits::from_bits(u64::from(v.to_bits())),
            },
            Bits::I64(value) => ScalarWire::I64 { value },
            Bits::I32(value) => ScalarWire::I32 { value },
            Bits::I16(value) => ScalarWire::I16 { value },
            Bits::I8(value) => ScalarWire::I8 { value },
            Bits::Bool(value) => ScalarWire::Bool { value },
            // spec/10 §3.2: a key has no literal carrier; every key in a
            // graph is a node's output.
            Bits::Key(_) => return Err(serde::ser::Error::custom(KEY_HAS_NO_LITERAL_CARRIER)),
        }
        .encode(serializer)
    }
}

impl<'de> Deserialize<'de> for ScalarValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bits = match ScalarWire::decode(deserializer)? {
            ScalarWire::F64 { bits } => Bits::F64(f64::from_bits(bits.bits())),
            ScalarWire::F32 { bits } => Bits::F32(f32::from_bits(bits.bits() as u32)),
            ScalarWire::F16 { bits } => Bits::F16(half::f16::from_bits(bits.bits() as u16)),
            ScalarWire::Bf16 { bits } => Bits::Bf16(half::bf16::from_bits(bits.bits() as u16)),
            ScalarWire::I64 { value } => Bits::I64(value),
            ScalarWire::I32 { value } => Bits::I32(value),
            ScalarWire::I16 { value } => Bits::I16(value),
            ScalarWire::I8 { value } => Bits::I8(value),
            ScalarWire::Bool { value } => Bits::Bool(value),
        };
        Ok(Self { bits })
    }
}

impl Serialize for TensorStorage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if matches!(self.buf, Buf::Key(_)) {
            return Err(serde::ser::Error::custom(KEY_HAS_NO_LITERAL_CARRIER));
        }
        storage_wire(self).encode(serializer)
    }
}

fn storage_wire(storage: &TensorStorage) -> StorageWire {
    {
        match &storage.buf {
            Buf::F64(v) => StorageWire::F64 {
                bits: v.iter().map(|v| HexBits::from_bits(v.to_bits())).collect(),
            },
            Buf::F32(v) => StorageWire::F32 {
                bits: v
                    .iter()
                    .map(|v| HexBits::from_bits(u64::from(v.to_bits())))
                    .collect(),
            },
            Buf::F16(v) => StorageWire::F16 {
                bits: v
                    .iter()
                    .map(|v| HexBits::from_bits(u64::from(v.to_bits())))
                    .collect(),
            },
            Buf::Bf16(v) => StorageWire::Bf16 {
                bits: v
                    .iter()
                    .map(|v| HexBits::from_bits(u64::from(v.to_bits())))
                    .collect(),
            },
            Buf::I64(v) => StorageWire::I64 { values: v.clone() },
            Buf::I32(v) => StorageWire::I32 { values: v.clone() },
            Buf::I16(v) => StorageWire::I16 { values: v.clone() },
            Buf::I8(v) => StorageWire::I8 { values: v.clone() },
            Buf::Bool(v) => StorageWire::Bool {
                values: v.iter().map(|v| *v != 0).collect(),
            },
            Buf::Key(v) => StorageWire::Key {
                bits: v.iter().map(|key| KeyHex::from_key(*key)).collect(),
            },
        }
    }
}

impl<'de> Deserialize<'de> for TensorStorage {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = StorageWire::decode(deserializer)?;
        if matches!(wire, StorageWire::Key { .. }) {
            return Err(serde::de::Error::custom(KEY_HAS_NO_LITERAL_CARRIER));
        }
        Ok(storage_from_wire(wire))
    }
}

fn storage_from_wire(wire: StorageWire) -> TensorStorage {
    {
        let buf = match wire {
            StorageWire::F64 { bits } => {
                Buf::F64(bits.into_iter().map(|b| f64::from_bits(b.bits())).collect())
            }
            StorageWire::F32 { bits } => Buf::F32(
                bits.into_iter()
                    .map(|b| f32::from_bits(b.bits() as u32))
                    .collect(),
            ),
            StorageWire::F16 { bits } => Buf::F16(
                bits.into_iter()
                    .map(|b| half::f16::from_bits(b.bits() as u16))
                    .collect(),
            ),
            StorageWire::Bf16 { bits } => Buf::Bf16(
                bits.into_iter()
                    .map(|b| half::bf16::from_bits(b.bits() as u16))
                    .collect(),
            ),
            StorageWire::I64 { values } => Buf::I64(values),
            StorageWire::I32 { values } => Buf::I32(values),
            StorageWire::I16 { values } => Buf::I16(values),
            StorageWire::I8 { values } => Buf::I8(values),
            StorageWire::Bool { values } => Buf::Bool(values.into_iter().map(u8::from).collect()),
            StorageWire::Key { bits } => Buf::Key(bits.iter().map(KeyHex::key).collect()),
        };
        TensorStorage { buf }
    }
}

/// spec/10 section 3.2's storage object inside a tensor execution value:
/// the graph storage grammar plus a key tensor's
/// `{"dtype":"key","bits":[h,...]}`. Use with `#[serde(with = ...)]` on an
/// execution carrier's storage field; a graph field keeps
/// [`TensorStorage`]'s own codec, which refuses a key.
pub mod execution_storage {
    use super::{StorageWire, TensorStorage, storage_from_wire, storage_wire};

    pub fn serialize<S: serde::Serializer>(
        storage: &TensorStorage,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        storage_wire(storage).encode(serializer)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> Result<TensorStorage, D::Error> {
        Ok(storage_from_wire(StorageWire::decode(deserializer)?))
    }

    pub fn json_schema(
        generator: &mut schemars::r#gen::SchemaGenerator,
    ) -> schemars::schema::Schema {
        <StorageWire as schemars::JsonSchema>::json_schema(generator)
    }
}
