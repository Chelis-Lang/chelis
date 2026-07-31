//! Dependency-bottom vocabularies that Chelis stages share.
//!
//! This crate has no dependencies, standard library, allocation, unsafe code,
//! or fallback vocabulary variants. It defines identifiers that cross compiler,
//! runtime, and generated-code boundaries.
//!
//! Source renderers belong to their artifact owners. The runtime dtype renderer
//! is in `chelis-runtime::dtype_header`.

#![no_std]
#![forbid(unsafe_code)]

use core::error::Error;
use core::fmt;

/// The input forms that the effect-kind decoder accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectKindInput<'a> {
    Missing,
    Malformed,
    Symbol(&'a str),
}

/// A closed effect vocabulary for all semantic consumers.
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

/// An effect-kind decode error.
///
/// The unknown-symbol case borrows the input symbol. The decode path does not
/// allocate an error string.
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

/// The physical encoding of one runtime element.
///
/// A byte width is a property of a representation, but width does not identify
/// the representation. Equal-width encodings can have different bit meanings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Repr {
    Ieee754Binary16,
    Ieee754Binary32,
    Ieee754Binary64,
    Bfloat16,
    TwosComplement8,
    TwosComplement16,
    TwosComplement32,
    TwosComplement64,
    /// A boolean in an IEEE binary32 payload.
    ///
    /// The payload uses `0.0` for false and `1.0` for true. This variant names
    /// the current four-byte ABI debt without approval of that design.
    BoolInBinary32,
}

impl Repr {
    pub const ALL: [Self; 9] = [
        Self::Ieee754Binary16,
        Self::Ieee754Binary32,
        Self::Ieee754Binary64,
        Self::Bfloat16,
        Self::TwosComplement8,
        Self::TwosComplement16,
        Self::TwosComplement32,
        Self::TwosComplement64,
        Self::BoolInBinary32,
    ];

    pub const fn byte_width(self) -> usize {
        match self {
            Self::TwosComplement8 => 1,
            Self::Ieee754Binary16 | Self::Bfloat16 | Self::TwosComplement16 => 2,
            Self::Ieee754Binary32 | Self::TwosComplement32 | Self::BoolInBinary32 => 4,
            Self::Ieee754Binary64 | Self::TwosComplement64 => 8,
        }
    }

    /// Reports whether the encoding carries a logical value in another
    /// representation.
    pub const fn is_payload_encoded(self) -> bool {
        match self {
            Self::BoolInBinary32 => true,
            Self::Ieee754Binary16
            | Self::Ieee754Binary32
            | Self::Ieee754Binary64
            | Self::Bfloat16
            | Self::TwosComplement8
            | Self::TwosComplement16
            | Self::TwosComplement32
            | Self::TwosComplement64 => false,
        }
    }
}

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

    /// Returns the physical encoding of one element.
    pub const fn repr(self) -> Repr {
        match self {
            Self::F32 => Repr::Ieee754Binary32,
            Self::F64 => Repr::Ieee754Binary64,
            Self::I32 => Repr::TwosComplement32,
            Self::Bool => Repr::BoolInBinary32,
            Self::I64 => Repr::TwosComplement64,
            Self::Bf16 => Repr::Bfloat16,
            Self::F16 => Repr::Ieee754Binary16,
            Self::I8 => Repr::TwosComplement8,
            Self::I16 => Repr::TwosComplement16,
        }
    }

    /// Returns the width from the physical representation.
    pub const fn byte_width(self) -> usize {
        self.repr().byte_width()
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

const _: () = assert!(core::mem::size_of::<RuntimeDType>() == 4);

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
