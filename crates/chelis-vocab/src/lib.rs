//! Dependency-bottom closed vocabularies shared across Chelis stages.
//!
//! This crate has no dependencies, no standard library, no allocation, and no
//! fallback vocabulary variants.  It is the single authority for identifiers
//! that cross compiler, runtime, and generated-code boundaries.
//!
//! Emitting source text for another language is shell work and lives with the
//! artifact it produces; see `chelis-runtime`'s `dtype_header` module.

#![no_std]
#![forbid(unsafe_code)]

use core::error::Error;
use core::fmt;

/// The structurally distinct ways a Deep effect-kind annotation can arrive at
/// the vocabulary decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectKindInput<'a> {
    Missing,
    Malformed,
    Symbol(&'a str),
}

/// A closed effect vocabulary shared by every semantic consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectKind {
    Random,
    Resource,
}

impl EffectKind {
    pub const ALL: [Self; 2] = [Self::Random, Self::Resource];

    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Resource => "resource",
        }
    }

    pub fn decode<'a>(input: EffectKindInput<'a>) -> Result<Self, EffectKindDecodeError<'a>> {
        match input {
            EffectKindInput::Missing => Err(EffectKindDecodeError::Missing),
            EffectKindInput::Malformed => Err(EffectKindDecodeError::Malformed),
            EffectKindInput::Symbol("random") => Ok(Self::Random),
            EffectKindInput::Symbol("resource") => Ok(Self::Resource),
            EffectKindInput::Symbol(symbol) => Err(EffectKindDecodeError::Unknown { symbol }),
        }
    }
}

/// Borrows the offending symbol from the decoded input rather than owning a
/// heap copy, so the decode path performs no allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectKindDecodeError<'a> {
    Missing,
    Malformed,
    Unknown { symbol: &'a str },
}

impl fmt::Display for EffectKindDecodeError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => f.write_str("missing effect kind"),
            Self::Malformed => f.write_str("malformed effect kind"),
            Self::Unknown { symbol } => write!(f, "unknown effect kind `{symbol}`"),
        }
    }
}

impl Error for EffectKindDecodeError<'_> {}

/// Dtypes that the C-compatible runtime ABI can store.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeDType {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    Bool = 3,
    I64 = 4,
    Bf16 = 5,
    F16 = 6,
    I8 = 7,
    I16 = 8,
}

impl RuntimeDType {
    pub const ALL: [Self; 9] = [
        Self::F32,
        Self::F64,
        Self::I32,
        Self::Bool,
        Self::I64,
        Self::Bf16,
        Self::F16,
        Self::I8,
        Self::I16,
    ];

    pub const fn id(self) -> i32 {
        self as i32
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I32 => "int32",
            Self::Bool => "bool",
            Self::I64 => "int64",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
            Self::I8 => "int8",
            Self::I16 => "int16",
        }
    }

    pub const fn c_macro(self) -> &'static str {
        match self {
            Self::F32 => "CHELIS_F32",
            Self::F64 => "CHELIS_F64",
            Self::I32 => "CHELIS_I32",
            Self::Bool => "CHELIS_BOOL",
            Self::I64 => "CHELIS_I64",
            Self::Bf16 => "CHELIS_BF16",
            Self::F16 => "CHELIS_F16",
            Self::I8 => "CHELIS_I8",
            Self::I16 => "CHELIS_I16",
        }
    }

    /// Fixed-width because this describes the runtime ABI, which does not vary
    /// with the host pointer width.
    pub const fn byte_width(self) -> u32 {
        match self {
            Self::F32 | Self::I32 | Self::Bool => 4,
            Self::F64 | Self::I64 => 8,
            Self::Bf16 | Self::F16 | Self::I16 => 2,
            Self::I8 => 1,
        }
    }

    pub const fn decode_id(id: i32) -> Result<Self, RuntimeDTypeDecodeError> {
        match id {
            0 => Ok(Self::F32),
            1 => Ok(Self::F64),
            2 => Ok(Self::I32),
            3 => Ok(Self::Bool),
            4 => Ok(Self::I64),
            5 => Ok(Self::Bf16),
            6 => Ok(Self::F16),
            7 => Ok(Self::I8),
            8 => Ok(Self::I16),
            id => Err(RuntimeDTypeDecodeError::InvalidId { id }),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDTypeDecodeError {
    InvalidId { id: i32 },
}

impl fmt::Display for RuntimeDTypeDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidId { id } => write!(f, "invalid Chelis runtime dtype id: {id}"),
        }
    }
}

impl Error for RuntimeDTypeDecodeError {}

/// The runtime ABI tag is a fixed-width `i32`; generated C declares it as
/// `int`.  A representation change must fail the build, not a test.
const _: () = assert!(core::mem::size_of::<RuntimeDType>() == 4);

/// Each discriminant is a shipped ABI value that generated C and compiled
/// artifacts already depend on.  Reordering the variants must fail the build.
const _: () = {
    assert!(RuntimeDType::F32.id() == 0);
    assert!(RuntimeDType::F64.id() == 1);
    assert!(RuntimeDType::I32.id() == 2);
    assert!(RuntimeDType::Bool.id() == 3);
    assert!(RuntimeDType::I64.id() == 4);
    assert!(RuntimeDType::Bf16.id() == 5);
    assert!(RuntimeDType::F16.id() == 6);
    assert!(RuntimeDType::I8.id() == 7);
    assert!(RuntimeDType::I16.id() == 8);
};
