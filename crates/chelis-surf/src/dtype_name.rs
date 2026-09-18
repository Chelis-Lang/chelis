//! Closed primitive and reserved dtype spellings shared by Surf ownership,
//! desugaring, formatting, and resugaring.

// `f8e4m3` is reserved, not active. `DEFERRED_DTYPE_NAMES` keeps it on the
// primitive-request path so the checker reports [04-DTYPE-1].
const PRIMITIVES: &[&str] = &[
    "f32", "f64", "f16", "bf16", "i8", "i16", "i32", "i64", "bool", "string", "unit",
];

const RETIRED_INTEGER_DTYPE_NAMES: &[&str] = &["int8", "int16", "int32", "int64"];

/// Unsigned dtype names, deferred per `spec/04-type-system.md` §1.1.1
/// (§1.1.2 names the `uint*` spellings canonical; the short `u*` spellings
/// are also reserved).
const UNSIGNED_DTYPE_NAMES: &[&str] = &[
    "u8", "u16", "u32", "u64", "uint8", "uint16", "uint32", "uint64",
];

/// Reserved non-unsigned dtype names from `spec/04-type-system.md` §1.1.1.
const DEFERRED_DTYPE_NAMES: &[&str] = &[
    "f8e4m3",
    "f8e5m2",
    "int4",
    "uint4",
    "complex64",
    "complex128",
    "decimal128",
    "decimal256",
];

/// The canonical primitive spelling for a type-position name, or `None` when
/// the name is not a primitive at all.
pub(crate) fn canonical_primitive_name(name: &str) -> Option<&'static str> {
    PRIMITIVES
        .iter()
        .copied()
        .find(|primitive| *primitive == name)
}

pub(crate) fn is_retired_integer_dtype_name(name: &str) -> bool {
    RETIRED_INTEGER_DTYPE_NAMES.contains(&name)
}

pub(crate) fn migrated_integer_dtype_name(name: &str) -> Option<&'static str> {
    match name {
        "int8" => Some("i8"),
        "int16" => Some("i16"),
        "int32" => Some("i32"),
        "int64" => Some("i64"),
        _ => None,
    }
}

/// True for every reserved, retired, or deferred dtype spelling.
pub(crate) fn is_reserved_dtype_name(name: &str) -> bool {
    UNSIGNED_DTYPE_NAMES.contains(&name)
        || DEFERRED_DTYPE_NAMES.contains(&name)
        || is_retired_integer_dtype_name(name)
}

/// A declaration binder cannot capture any active, reserved, retired, or
/// deferred dtype spelling. Unknown intentional names remain legal.
pub(crate) fn is_forbidden_binder_name(name: &str) -> bool {
    canonical_primitive_name(name).is_some() || is_reserved_dtype_name(name)
}
