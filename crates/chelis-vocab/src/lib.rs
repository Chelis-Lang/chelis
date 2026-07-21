//! Dependency-bottom closed vocabularies shared across Chelis stages.
//!
//! This crate deliberately has no dependencies and no fallback vocabulary
//! variants.  It is the single authority for identifiers that cross compiler,
//! runtime, and generated-code boundaries.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

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

    pub fn decode(input: EffectKindInput<'_>) -> Result<Self, EffectKindDecodeError> {
        match input {
            EffectKindInput::Missing => Err(EffectKindDecodeError::Missing),
            EffectKindInput::Malformed => Err(EffectKindDecodeError::Malformed),
            EffectKindInput::Symbol("random") => Ok(Self::Random),
            EffectKindInput::Symbol("resource") => Ok(Self::Resource),
            EffectKindInput::Symbol(symbol) => Err(EffectKindDecodeError::Unknown {
                symbol: symbol.to_owned(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectKindDecodeError {
    Missing,
    Malformed,
    Unknown { symbol: String },
}

impl fmt::Display for EffectKindDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => f.write_str("missing effect kind"),
            Self::Malformed => f.write_str("malformed effect kind"),
            Self::Unknown { symbol } => write!(f, "unknown effect kind `{symbol}`"),
        }
    }
}

impl Error for EffectKindDecodeError {}

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

    pub const fn byte_width(self) -> usize {
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

/// Render the checked-in C ABI fragment from the Rust vocabulary.
///
/// The C decoder terminates on invalid input.  C callers cannot accidentally
/// reinterpret an unknown ABI tag as `f32`.
pub fn render_runtime_dtype_c_header() -> String {
    let mut header = String::from(
        "#ifndef CHELIS_RUNTIME_DTYPE_H\n#define CHELIS_RUNTIME_DTYPE_H\n\n#include <stddef.h>\n#include <stdio.h>\n#include <stdlib.h>\n\n",
    );
    for dtype in RuntimeDType::ALL {
        header.push_str(&format!("#define {} {}\n", dtype.c_macro(), dtype.id()));
    }
    header.push_str(
        "\nstatic inline size_t chelis_runtime_dtype_size_checked(int dtype) {\n    switch (dtype) {\n",
    );
    for dtype in RuntimeDType::ALL {
        header.push_str(&format!(
            "        case {}: return {};\n",
            dtype.c_macro(),
            dtype.byte_width()
        ));
    }
    header.push_str(
        "        default:\n            fprintf(stderr, \"invalid Chelis runtime dtype id: %d\\n\", dtype);\n            abort();\n    }\n}\n\n#endif\n",
    );
    header
}
