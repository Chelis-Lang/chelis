//! Exact stored-bit transport under spec/10 section 3.2.

use super::{Bits, RandomKey, ScalarValue};
#[cfg(feature = "alloc")]
use super::{Buf, TensorStorage};
#[cfg(feature = "alloc")]
use alloc::vec::Vec;
use core::fmt;
use serde::{Deserialize, Serialize};

const KEY_HAS_NO_LITERAL_CARRIER: &str =
    "a random key has no literal carrier: every key in a graph is a node's output";

#[derive(Clone, Copy)]
struct HexBits<const DIGITS: usize>([u8; DIGITS]);

impl<const DIGITS: usize> HexBits<DIGITS> {
    fn from_bits(mut bits: u64) -> Self {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut text = [b'0'; DIGITS];
        for digit in text.iter_mut().rev() {
            *digit = HEX[(bits & 15) as usize];
            bits >>= 4;
        }
        assert_eq!(bits, 0, "stored bits must fit their declared width");
        Self(text)
    }

    fn bits(self) -> u64 {
        self.0.into_iter().fold(0, |bits, digit| {
            (bits << 4)
                | u64::from(if digit <= b'9' {
                    digit - b'0'
                } else {
                    digit - b'a' + 10
                })
        })
    }
}

impl<const DIGITS: usize> Serialize for HexBits<DIGITS> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(core::str::from_utf8(&self.0).expect("hexadecimal text is ASCII"))
    }
}

impl<'de, const DIGITS: usize> Deserialize<'de> for HexBits<DIGITS> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct HexVisitor<const N: usize>;
        impl<'de, const N: usize> serde::de::Visitor<'de> for HexVisitor<N> {
            type Value = HexBits<N>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_str<E: serde::de::Error>(self, text: &str) -> Result<Self::Value, E> {
                if text.len() != N
                    || !text
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err(E::custom(format_args!(
                        "stored IEEE bits require exactly {N} lowercase hexadecimal digits"
                    )));
                }
                let mut bytes = [0; N];
                bytes.copy_from_slice(text.as_bytes());
                Ok(HexBits(bytes))
            }
        }
        deserializer.deserialize_str(HexVisitor::<DIGITS>)
    }
}

#[derive(Clone, Serialize)]
#[serde(transparent)]
struct KeyHex(HexBits<16>);

impl KeyHex {
    fn from_key(key: RandomKey) -> Self {
        Self(HexBits::from_bits(key.bits))
    }
    fn key(&self) -> RandomKey {
        RandomKey {
            bits: self.0.bits(),
        }
    }
}

impl<'de> Deserialize<'de> for KeyHex {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = KeyHex;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_str<E: serde::de::Error>(self, text: &str) -> Result<Self::Value, E> {
                if text.len() != 16
                    || !text
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err(E::custom(
                        "a key's bits require exactly 16 lowercase hexadecimal digits",
                    ));
                }
                let mut bytes = [0; 16];
                bytes.copy_from_slice(text.as_bytes());
                Ok(KeyHex(HexBits(bytes)))
            }
        }
        deserializer.deserialize_str(Visitor)
    }
}

/// A key's exact 64-bit execution carrier.
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

// The alloc decoder retains host diagnostics and error precedence.
// The scalar-only decoder uses no tagged-value buffer.
#[cfg_attr(feature = "alloc", derive(Deserialize))]
#[cfg_attr(feature = "alloc", serde(tag = "dtype", deny_unknown_fields))]
enum ScalarWire {
    #[cfg_attr(feature = "alloc", serde(rename = "f64"))]
    F64 { bits: HexBits<16> },
    #[cfg_attr(feature = "alloc", serde(rename = "f32"))]
    F32 { bits: HexBits<8> },
    #[cfg_attr(feature = "alloc", serde(rename = "f16"))]
    F16 { bits: HexBits<4> },
    #[cfg_attr(feature = "alloc", serde(rename = "bf16"))]
    Bf16 { bits: HexBits<4> },
    #[cfg_attr(feature = "alloc", serde(rename = "int64"))]
    I64 { value: i64 },
    #[cfg_attr(feature = "alloc", serde(rename = "int32"))]
    I32 { value: i32 },
    #[cfg_attr(feature = "alloc", serde(rename = "int16"))]
    I16 { value: i16 },
    #[cfg_attr(feature = "alloc", serde(rename = "int8"))]
    I8 { value: i8 },
    #[cfg_attr(feature = "alloc", serde(rename = "bool"))]
    Bool { value: bool },
}

#[derive(Serialize, Deserialize)]
enum BinaryScalarWire {
    F64 { bits: HexBits<16> },
    F32 { bits: HexBits<8> },
    F16 { bits: HexBits<4> },
    Bf16 { bits: HexBits<4> },
    I64 { value: i64 },
    I32 { value: i32 },
    I16 { value: i16 },
    I8 { value: i8 },
    Bool { value: bool },
}

#[cfg(not(feature = "alloc"))]
#[derive(Deserialize)]
enum ScalarDType {
    #[serde(rename = "f64")]
    F64,
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "bf16")]
    Bf16,
    #[serde(rename = "int64")]
    I64,
    #[serde(rename = "int32")]
    I32,
    #[serde(rename = "int16")]
    I16,
    #[serde(rename = "int8")]
    I8,
    #[serde(rename = "bool")]
    Bool,
}

#[cfg(not(feature = "alloc"))]
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "lowercase")]
enum ScalarField {
    Dtype,
    Bits,
    Value,
}

#[cfg(not(feature = "alloc"))]
enum ScalarPayload {
    Integer(i64),
    Bool(bool),
}
#[cfg(not(feature = "alloc"))]
impl<'de> Deserialize<'de> for ScalarPayload {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ValueVisitor;
        impl<'de> serde::de::Visitor<'de> for ValueVisitor {
            type Value = ScalarPayload;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an exact signed integer or bool")
            }
            fn visit_i64<E: serde::de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(ScalarPayload::Integer(value))
            }
            fn visit_u64<E: serde::de::Error>(self, value: u64) -> Result<Self::Value, E> {
                i64::try_from(value)
                    .map(ScalarPayload::Integer)
                    .map_err(E::custom)
            }
            fn visit_bool<E: serde::de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(ScalarPayload::Bool(value))
            }
        }
        deserializer.deserialize_any(ValueVisitor)
    }
}

#[cfg(not(feature = "alloc"))]
struct ScalarHex {
    bytes: [u8; 16],
    len: usize,
    valid: bool,
}
#[cfg(not(feature = "alloc"))]
impl<'de> Deserialize<'de> for ScalarHex {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = ScalarHex;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_str<E: serde::de::Error>(self, text: &str) -> Result<Self::Value, E> {
                let valid = text
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
                let mut bytes = [0; 16];
                let count = text.len().min(bytes.len());
                bytes[..count].copy_from_slice(&text.as_bytes()[..count]);
                Ok(ScalarHex {
                    bytes,
                    len: text.len(),
                    valid,
                })
            }
        }
        deserializer.deserialize_str(Visitor)
    }
}

#[cfg(not(feature = "alloc"))]
impl ScalarHex {
    fn fixed<const N: usize, E: serde::de::Error>(self) -> Result<HexBits<N>, E> {
        if self.len != N || !self.valid {
            return Err(E::custom(format_args!(
                "stored IEEE bits require exactly {N} lowercase hexadecimal digits"
            )));
        }
        let mut bytes = [0; N];
        bytes.copy_from_slice(&self.bytes[..N]);
        Ok(HexBits(bytes))
    }
}

impl ScalarWire {
    fn encode<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if !serializer.is_human_readable() {
            return match self {
                Self::F64 { bits } => BinaryScalarWire::F64 { bits: *bits },
                Self::F32 { bits } => BinaryScalarWire::F32 { bits: *bits },
                Self::F16 { bits } => BinaryScalarWire::F16 { bits: *bits },
                Self::Bf16 { bits } => BinaryScalarWire::Bf16 { bits: *bits },
                Self::I64 { value } => BinaryScalarWire::I64 { value: *value },
                Self::I32 { value } => BinaryScalarWire::I32 { value: *value },
                Self::I16 { value } => BinaryScalarWire::I16 { value: *value },
                Self::I8 { value } => BinaryScalarWire::I8 { value: *value },
                Self::Bool { value } => BinaryScalarWire::Bool { value: *value },
            }
            .serialize(serializer);
        }
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("ScalarWire", 2)?;
        macro_rules! fields {
            ($dtype:literal, $name:literal, $value:expr) => {{
                state.serialize_field("dtype", $dtype)?;
                state.serialize_field($name, $value)?;
            }};
        }
        match self {
            Self::F64 { bits } => fields!("f64", "bits", bits),
            Self::F32 { bits } => fields!("f32", "bits", bits),
            Self::F16 { bits } => fields!("f16", "bits", bits),
            Self::Bf16 { bits } => fields!("bf16", "bits", bits),
            Self::I64 { value } => fields!("int64", "value", value),
            Self::I32 { value } => fields!("int32", "value", value),
            Self::I16 { value } => fields!("int16", "value", value),
            Self::I8 { value } => fields!("int8", "value", value),
            Self::Bool { value } => fields!("bool", "value", value),
        }
        state.end()
    }

    fn decode<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if !deserializer.is_human_readable() {
            return Ok(match BinaryScalarWire::deserialize(deserializer)? {
                BinaryScalarWire::F64 { bits } => Self::F64 { bits },
                BinaryScalarWire::F32 { bits } => Self::F32 { bits },
                BinaryScalarWire::F16 { bits } => Self::F16 { bits },
                BinaryScalarWire::Bf16 { bits } => Self::Bf16 { bits },
                BinaryScalarWire::I64 { value } => Self::I64 { value },
                BinaryScalarWire::I32 { value } => Self::I32 { value },
                BinaryScalarWire::I16 { value } => Self::I16 { value },
                BinaryScalarWire::I8 { value } => Self::I8 { value },
                BinaryScalarWire::Bool { value } => Self::Bool { value },
            });
        }
        #[cfg(feature = "alloc")]
        {
            Self::deserialize(deserializer)
        }
        #[cfg(not(feature = "alloc"))]
        {
            Self::decode_map(deserializer)
        }
    }

    #[cfg(not(feature = "alloc"))]
    fn decode_map<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = ScalarWire;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("an exact scalar transport object")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                use serde::de::Error;
                let (mut dtype, mut bits, mut value) = (None, None, None);
                while let Some(field) = map.next_key::<ScalarField>()? {
                    match field {
                        ScalarField::Dtype => {
                            if dtype.is_some() {
                                return Err(M::Error::duplicate_field("dtype"));
                            }
                            dtype = Some(map.next_value::<ScalarDType>()?);
                        }
                        ScalarField::Bits => {
                            if bits.is_some() {
                                return Err(M::Error::duplicate_field("bits"));
                            }
                            bits = Some(map.next_value::<ScalarHex>()?);
                        }
                        ScalarField::Value => {
                            if value.is_some() {
                                return Err(M::Error::duplicate_field("value"));
                            }
                            value = Some(map.next_value::<ScalarPayload>()?);
                        }
                    }
                }
                let dtype = dtype.ok_or_else(|| M::Error::missing_field("dtype"))?;
                macro_rules! integer {
                    ($variant:ident, $ty:ty) => {
                        match (bits, value) {
                            (None, Some(ScalarPayload::Integer(value))) => {
                                Ok(ScalarWire::$variant {
                                    value: <$ty>::try_from(value).map_err(M::Error::custom)?,
                                })
                            }
                            _ => Err(M::Error::custom(
                                "integer scalar needs only its exact value",
                            )),
                        }
                    };
                }
                macro_rules! float {
                    ($variant:ident) => {
                        match (bits, value) {
                            (Some(bits), None) => Ok(ScalarWire::$variant {
                                bits: bits.fixed()?,
                            }),
                            _ => Err(M::Error::custom("float scalar needs only its stored bits")),
                        }
                    };
                }
                match dtype {
                    ScalarDType::F64 => float!(F64),
                    ScalarDType::F32 => float!(F32),
                    ScalarDType::F16 => float!(F16),
                    ScalarDType::Bf16 => float!(Bf16),
                    ScalarDType::I64 => integer!(I64, i64),
                    ScalarDType::I32 => integer!(I32, i32),
                    ScalarDType::I16 => integer!(I16, i16),
                    ScalarDType::I8 => integer!(I8, i8),
                    ScalarDType::Bool => match (bits, value) {
                        (None, Some(ScalarPayload::Bool(value))) => Ok(ScalarWire::Bool { value }),
                        _ => Err(M::Error::custom("bool scalar needs only its bool value")),
                    },
                }
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

#[cfg(feature = "alloc")]
// Author each dtype/payload relationship once. The cache mirror changes only
// the enclosing discriminant representation; it has no alternate numeric path.
macro_rules! wire_enum {
    ($wire:ident, $binary:ident { $($variant:ident ($dtype:literal) { $field:ident: $ty:ty }),* $(,)? }) => {
        #[derive(Serialize, Deserialize)]
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

#[cfg(feature = "alloc")]
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

#[cfg(feature = "alloc")]
impl Serialize for TensorStorage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if matches!(self.buf, Buf::Key(_)) {
            return Err(serde::ser::Error::custom(KEY_HAS_NO_LITERAL_CARRIER));
        }
        storage_wire(self).encode(serializer)
    }
}
#[cfg(feature = "alloc")]
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
#[cfg(feature = "alloc")]
impl<'de> Deserialize<'de> for TensorStorage {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = StorageWire::decode(deserializer)?;
        if matches!(wire, StorageWire::Key { .. }) {
            return Err(serde::de::Error::custom(KEY_HAS_NO_LITERAL_CARRIER));
        }
        Ok(storage_from_wire(wire))
    }
}
#[cfg(feature = "alloc")]
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
#[cfg(feature = "alloc")]
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
}
