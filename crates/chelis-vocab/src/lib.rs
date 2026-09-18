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

/// Stable machine-facing identities for compiler diagnostics.
///
/// The enum governs producer construction while [`Self::as_str`] preserves
/// the existing JSON spelling. It deliberately has no serde dependency: wire
/// producers render through `as_str`, and wire consumers decode explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticKind {
    SurfParseError,
    DeepParseError,
    MacroError,
    NameResolutionError,
    DeepDeclError,
    DuplicateName,
    PreimageMismatch,
    CascadeIncomplete,
    TypeError,
    EffectError,
    LinearityError,
    LowerError,
    ReefError,
    EvalError,
    Cancelled,
    GradError,
    ValidationError,
    UnknownName,
    UnknownSchemaVersion,
    HashError,
    InvalidRequest,
    CompileError,
    GeneralOther,
    UnsupportedFeature,
    TypeMismatch,
    PrecisionMismatch,
    DimensionMismatch,
    ArityMismatch,
    UnboundVariable,
    UnknownConstructor,
    NotAFunction,
    NonExhaustiveMatch,
    OccursCheck,
    CastNonTensor,
    TupleIndexOutOfBounds,
    UseAfterConsume,
    UnconsumedLinear,
    InvalidBorrow,
    CycleDetected,
    UnsupportedTensorPrecision,
    DuplicateDefinition,
    DuplicateModule,
    OpaqueTypeViolation,
    ReservedLinkerName,
    BuiltinShadowing,
    UnknownForm,
    MalformedForm,
    CheckOther,
    // chelis#886: the effect checker's kinds. The `chelis check` report has
    // always published these spellings; they become governed identities here
    // so the wire stops being spelled from `chelis_effects`' Rust variant
    // names.
    UnhandledEffect,
    InvalidHandler,
    BuildTargetMismatch,
    TypeTotality,
    // chelis#1678: directory mode's two failures that belong to no single
    // file (spec/04 [04-FIT-23] and [04-FIT-24]).
    DirectoryWalkError,
    EmptyCorpus,
    // chelis#1825: a completed ordinary `chelis test` run selected no
    // runnable test (spec/04 [04-TEST-1..3]).
    EmptyTestSelection,
}

impl DiagnosticKind {
    pub const ALL: [Self; 55] = [
        Self::SurfParseError,
        Self::DeepParseError,
        Self::MacroError,
        Self::NameResolutionError,
        Self::DeepDeclError,
        Self::DuplicateName,
        Self::PreimageMismatch,
        Self::CascadeIncomplete,
        Self::TypeError,
        Self::EffectError,
        Self::LinearityError,
        Self::LowerError,
        Self::ReefError,
        Self::EvalError,
        Self::Cancelled,
        Self::GradError,
        Self::ValidationError,
        Self::UnknownName,
        Self::UnknownSchemaVersion,
        Self::HashError,
        Self::InvalidRequest,
        Self::CompileError,
        Self::GeneralOther,
        Self::UnsupportedFeature,
        Self::TypeMismatch,
        Self::PrecisionMismatch,
        Self::DimensionMismatch,
        Self::ArityMismatch,
        Self::UnboundVariable,
        Self::UnknownConstructor,
        Self::NotAFunction,
        Self::NonExhaustiveMatch,
        Self::OccursCheck,
        Self::CastNonTensor,
        Self::TupleIndexOutOfBounds,
        Self::UseAfterConsume,
        Self::UnconsumedLinear,
        Self::InvalidBorrow,
        Self::CycleDetected,
        Self::UnsupportedTensorPrecision,
        Self::DuplicateDefinition,
        Self::DuplicateModule,
        Self::OpaqueTypeViolation,
        Self::ReservedLinkerName,
        Self::BuiltinShadowing,
        Self::UnknownForm,
        Self::MalformedForm,
        Self::CheckOther,
        Self::UnhandledEffect,
        Self::InvalidHandler,
        Self::BuildTargetMismatch,
        Self::TypeTotality,
        Self::DirectoryWalkError,
        Self::EmptyCorpus,
        Self::EmptyTestSelection,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SurfParseError => "surf_parse_error",
            Self::DeepParseError => "deep_parse_error",
            Self::MacroError => "macro_error",
            Self::NameResolutionError => "name_resolution_error",
            Self::DeepDeclError => "deep_decl_error",
            Self::DuplicateName => "duplicate_name",
            Self::PreimageMismatch => "preimage_mismatch",
            Self::CascadeIncomplete => "cascade_incomplete",
            Self::TypeError => "type_error",
            Self::EffectError => "effect_error",
            Self::LinearityError => "linearity_error",
            Self::LowerError => "lower_error",
            Self::ReefError => "reef_error",
            Self::EvalError => "eval_error",
            Self::Cancelled => "cancelled",
            Self::GradError => "grad_error",
            Self::ValidationError => "validation_error",
            Self::UnknownName => "unknown_name",
            Self::UnknownSchemaVersion => "unknown_schema_version",
            Self::HashError => "hash_error",
            Self::InvalidRequest => "invalid_request",
            Self::CompileError => "compile_error",
            Self::GeneralOther => "other",
            Self::UnsupportedFeature => "unsupported_feature",
            Self::TypeMismatch => "TypeMismatch",
            Self::PrecisionMismatch => "PrecisionMismatch",
            Self::DimensionMismatch => "DimensionMismatch",
            Self::ArityMismatch => "ArityMismatch",
            Self::UnboundVariable => "UnboundVariable",
            Self::UnknownConstructor => "UnknownConstructor",
            Self::NotAFunction => "NotAFunction",
            Self::NonExhaustiveMatch => "NonExhaustiveMatch",
            Self::OccursCheck => "OccursCheck",
            Self::CastNonTensor => "CastNonTensor",
            Self::TupleIndexOutOfBounds => "TupleIndexOutOfBounds",
            Self::UseAfterConsume => "UseAfterConsume",
            Self::UnconsumedLinear => "UnconsumedLinear",
            Self::InvalidBorrow => "InvalidBorrow",
            Self::CycleDetected => "CycleDetected",
            Self::UnsupportedTensorPrecision => "UnsupportedTensorPrecision",
            Self::DuplicateDefinition => "DuplicateDefinition",
            Self::DuplicateModule => "DuplicateModule",
            Self::OpaqueTypeViolation => "OpaqueTypeViolation",
            Self::ReservedLinkerName => "ReservedLinkerName",
            Self::BuiltinShadowing => "BuiltinShadowing",
            Self::UnknownForm => "UnknownForm",
            Self::MalformedForm => "MalformedForm",
            Self::CheckOther => "Other",
            Self::UnhandledEffect => "UnhandledEffect",
            Self::InvalidHandler => "InvalidHandler",
            Self::BuildTargetMismatch => "BuildTargetMismatch",
            Self::TypeTotality => "TypeTotality",
            Self::DirectoryWalkError => "directory_walk_error",
            Self::EmptyCorpus => "empty_corpus",
            Self::EmptyTestSelection => "empty_test_selection",
        }
    }

    pub fn decode(spelling: &str) -> Result<Self, DiagnosticKindDecodeError<'_>> {
        Self::ALL
            .into_iter()
            .find(|kind| kind.as_str() == spelling)
            .ok_or(DiagnosticKindDecodeError::Unknown { spelling })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticKindDecodeError<'a> {
    Unknown { spelling: &'a str },
}

impl fmt::Display for DiagnosticKindDecodeError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self::Unknown { spelling } = self;
        write!(f, "unknown diagnostic kind `{spelling}`")
    }
}

impl Error for DiagnosticKindDecodeError<'_> {}

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
    /// A canonical boolean in one byte: `0` is false and `1` is true.
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

    pub const fn byte_width(self) -> usize {
        match self {
            Self::TwosComplement8 | Self::Bool8 => 1,
            Self::Ieee754Binary16 | Self::Bfloat16 | Self::TwosComplement16 => 2,
            Self::Ieee754Binary32 | Self::TwosComplement32 => 4,
            Self::Ieee754Binary64 | Self::TwosComplement64 => 8,
        }
    }

    /// Reports whether the encoding carries a logical value in another
    /// representation.
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
///
/// Stored and arithmetic representation are selected together by [`Self::contract`].
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

/// The representation used while computing, before storage finalization
/// ([04-NUM-8]). This is identity, not operation or backend capability policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArithmeticRepr {
    Ieee754Binary32,
    Ieee754Binary64,
    ExactTwosComplement8,
    ExactTwosComplement16,
    ExactTwosComplement32,
    ExactTwosComplement64,
}

impl ArithmeticRepr {
    pub const ALL: [Self; 6] = [
        Self::Ieee754Binary32,
        Self::Ieee754Binary64,
        Self::ExactTwosComplement8,
        Self::ExactTwosComplement16,
        Self::ExactTwosComplement32,
        Self::ExactTwosComplement64,
    ];

    /// Exact encoding of an arithmetic intermediate, not the tensor's storage.
    pub const fn repr(self) -> Repr {
        match self {
            Self::Ieee754Binary32 => Repr::Ieee754Binary32,
            Self::Ieee754Binary64 => Repr::Ieee754Binary64,
            Self::ExactTwosComplement8 => Repr::TwosComplement8,
            Self::ExactTwosComplement16 => Repr::TwosComplement16,
            Self::ExactTwosComplement32 => Repr::TwosComplement32,
            Self::ExactTwosComplement64 => Repr::TwosComplement64,
        }
    }
}

/// A closed [04-NUM-8] registration. Only [`RuntimeDType::contract`] constructs
/// it; consumers cannot supply a width, swap an encoding, or grant bool arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DTypeContract {
    dtype: RuntimeDType,
    repr: Repr,
    arithmetic: Option<ArithmeticRepr>,
}

impl DTypeContract {
    pub const fn dtype(self) -> RuntimeDType {
        self.dtype
    }

    pub const fn repr(self) -> Repr {
        self.repr
    }

    pub const fn arithmetic(self) -> Option<ArithmeticRepr> {
        self.arithmetic
    }

    pub const fn byte_width(self) -> usize {
        self.repr.byte_width()
    }
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
            Self::F32 => "CHELIS_DTYPE_F32",
            Self::F64 => "CHELIS_DTYPE_F64",
            Self::I32 => "CHELIS_DTYPE_I32",
            Self::Bool => "CHELIS_DTYPE_BOOL",
            Self::I64 => "CHELIS_DTYPE_I64",
            Self::Bf16 => "CHELIS_DTYPE_BF16",
            Self::F16 => "CHELIS_DTYPE_F16",
            Self::I8 => "CHELIS_DTYPE_I8",
            Self::I16 => "CHELIS_DTYPE_I16",
        }
    }

    /// Selects stored and arithmetic representation together, exactly as
    /// [04-NUM-8] declares. Bool has no arithmetic representation ([04-NUM-4]).
    pub const fn contract(self) -> DTypeContract {
        use ArithmeticRepr as A;
        let (repr, arithmetic) = match self {
            Self::F32 => (Repr::Ieee754Binary32, Some(A::Ieee754Binary32)),
            Self::F64 => (Repr::Ieee754Binary64, Some(A::Ieee754Binary64)),
            Self::I32 => (Repr::TwosComplement32, Some(A::ExactTwosComplement32)),
            Self::Bool => (Repr::Bool8, None),
            Self::I64 => (Repr::TwosComplement64, Some(A::ExactTwosComplement64)),
            Self::Bf16 => (Repr::Bfloat16, Some(A::Ieee754Binary32)),
            Self::F16 => (Repr::Ieee754Binary16, Some(A::Ieee754Binary32)),
            Self::I8 => (Repr::TwosComplement8, Some(A::ExactTwosComplement8)),
            Self::I16 => (Repr::TwosComplement16, Some(A::ExactTwosComplement16)),
        };
        DTypeContract {
            dtype: self,
            repr,
            arithmetic,
        }
    }

    /// Returns the registered physical encoding of one element.
    pub const fn repr(self) -> Repr {
        self.contract().repr()
    }

    /// Returns the width from the physical representation.
    pub const fn byte_width(self) -> usize {
        self.contract().byte_width()
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
