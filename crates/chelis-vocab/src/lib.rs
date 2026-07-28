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

/// How a dtype's elements are physically encoded in a buffer.
///
/// **This, not the byte width, is the ABI contract.** A width is a projection
/// of a representation, and that projection is not injective:
/// `Ieee754Binary32` and `TwosComplement32` are both four bytes and are not
/// interchangeable, as are `TwosComplement8` and `Bool8` at one byte. Two
/// lanes agree about a dtype iff their `Repr` values are equal; comparing
/// widths can pass while the encodings differ, which is silent corruption
/// rather than a stride mismatch.
///
/// That first pair is not hypothetical. Eight runtime accessors decoded native
/// int32 through an f32 view, and a width-based ABI probe could not see it
/// because both sides reported four bytes. The same probe did catch bool,
/// whose widths disagreed. Width found the defect it happened to be sensitive
/// to and was blind to the one next to it.
///
/// Naming is by encoding rather than by logical type on purpose: it is what
/// makes an encoding that differs from its logical type impossible to write
/// down without saying so.
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
    /// A boolean in one native byte: `0` for false, `1` for true.
    ///
    /// This replaced a `BoolInBinary32` variant, which carried a boolean in an
    /// IEEE binary32 payload (`0.0` / `1.0`) and was tracked as
    /// `CRuntime-BoolStorage-F1`. Naming that deviation as a representation is
    /// what made the migration mechanical: width is derived from this enum, so
    /// flipping the variant moved every `chelis_alloc` sizing decision at once.
    ///
    /// The same property is why the migration had to be atomic. A one-byte
    /// width with four-byte writers still live is a heap overflow, not a
    /// mismatch, so no intermediate state was shippable.
    Bool8,
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
        Self::Bool8,
    ];

    pub const fn byte_width(self) -> u32 {
        match self {
            Self::TwosComplement8 | Self::Bool8 => 1,
            Self::Ieee754Binary16 | Self::Bfloat16 | Self::TwosComplement16 => 2,
            Self::Ieee754Binary32 | Self::TwosComplement32 => 4,
            Self::Ieee754Binary64 | Self::TwosComplement64 => 8,
        }
    }

    /// Whether the encoding carries a logical type other than the one its
    /// bit pattern denotes. Such a representation cannot be memcpy'd to or
    /// from a lane holding the type natively; it needs a conversion.
    /// No representation is payload-encoded today. `BoolInBinary32` was the
    /// only one, and it is gone; every remaining variant's bit pattern denotes
    /// the type it claims.
    ///
    /// The method stays because the concept is what the bool defect taught:
    /// two representations can share a width and still not be interchangeable,
    /// so a width comparison is not a compatibility check. Keeping it as a
    /// total function over `Repr` means the next payload encoding declares
    /// itself here rather than being discovered by a lane that memcpy'd it.
    pub const fn is_payload_encoded(self) -> bool {
        match self {
            Self::Ieee754Binary16
            | Self::Ieee754Binary32
            | Self::Ieee754Binary64
            | Self::Bfloat16
            | Self::TwosComplement8
            | Self::TwosComplement16
            | Self::TwosComplement32
            | Self::TwosComplement64
            | Self::Bool8 => false,
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

    /// The physical encoding of one element.
    ///
    /// This is the primitive concept; [`Self::byte_width`] is derived from it.
    /// Prefer comparing representations over comparing widths when checking
    /// that two lanes agree — see [`Repr`].
    pub const fn repr(self) -> Repr {
        match self {
            Self::F32 => Repr::Ieee754Binary32,
            Self::F64 => Repr::Ieee754Binary64,
            Self::F16 => Repr::Ieee754Binary16,
            Self::Bf16 => Repr::Bfloat16,
            Self::I8 => Repr::TwosComplement8,
            Self::I16 => Repr::TwosComplement16,
            Self::I32 => Repr::TwosComplement32,
            Self::I64 => Repr::TwosComplement64,
            // Not `Bool8`. The runtime stores bool as an f32 payload; see the
            // variant's documentation.
            Self::Bool => Repr::Bool8,
        }
    }

    /// Derived from [`Self::repr`]. Fixed-width because this describes the
    /// runtime ABI, which does not vary with the host pointer width.
    ///
    /// Equal widths do **not** imply interchangeable buffers.
    pub const fn byte_width(self) -> u32 {
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
