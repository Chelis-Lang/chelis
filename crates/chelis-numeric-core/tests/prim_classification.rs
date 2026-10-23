//! Pin every `Prim` variant against the active-set classification rules
//! in `spec/04-type-system.md` §1.1 and §1.1.1. Acceptance test (a) of
//! WS-A0; protects against an active dtype silently sliding back into
//! the deferred set or vice versa.
use chelis_numeric_core::prim::Prim;
use chelis_vocab::RuntimeDType;

/// Closed enumeration of every `Prim` variant. If a variant is added
/// or removed and this list is not updated, every per-variant test
/// below fails — that is the intended invariant lock.
const ALL_PRIMS: &[Prim] = &[
    Prim::F32,
    Prim::F64,
    Prim::F16,
    Prim::Bf16,
    Prim::F8e4m3,
    Prim::Int8,
    Prim::Int16,
    Prim::Int32,
    Prim::Int64,
    Prim::Bool,
    Prim::String,
    Prim::Key,
];

#[test]
fn is_float_matches_active_float_set() {
    for prim in ALL_PRIMS {
        let expected = matches!(prim, Prim::F32 | Prim::F64 | Prim::F16 | Prim::Bf16);
        assert_eq!(
            prim.is_float(),
            expected,
            "is_float({prim:?}) disagrees with §1.1 active float set; \
             f8e4m3 is deferred per §1.1.1 and must NOT be a float here"
        );
    }
}

#[test]
fn is_integer_matches_active_integer_set() {
    for prim in ALL_PRIMS {
        let expected = matches!(prim, Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64);
        assert_eq!(
            prim.is_integer(),
            expected,
            "is_integer({prim:?}) disagrees with §1.1 active integer set"
        );
    }
}

#[test]
fn is_numeric_excludes_bool_and_string() {
    for prim in ALL_PRIMS {
        let expected = !matches!(prim, Prim::Bool | Prim::String | Prim::Key);
        assert_eq!(
            prim.is_numeric(),
            expected,
            "is_numeric({prim:?}) disagrees with §1.1 numeric set"
        );
    }
}

#[test]
fn is_valid_tensor_precision_matches_active_set() {
    // Active dtype set per §1.1, including f16/bf16 (active per WS-0
    // spec lock cc47e6d) and key. Excludes the deferred f8e4m3 (§1.1.1)
    // and String (no tensor element representation). The data element
    // set is the same minus key.
    for prim in ALL_PRIMS {
        let data = matches!(
            prim,
            Prim::F32
                | Prim::F64
                | Prim::Bf16
                | Prim::F16
                | Prim::Bool
                | Prim::Int8
                | Prim::Int16
                | Prim::Int32
                | Prim::Int64
        );
        assert_eq!(
            prim.is_data_element_dtype(),
            data,
            "is_data_element_dtype({prim:?}) disagrees with §1.1 \
             active data element set"
        );
        assert_eq!(
            prim.is_valid_tensor_precision(),
            data || *prim == Prim::Key,
            "is_valid_tensor_precision({prim:?}) disagrees with §1.1 \
             active tensor element set"
        );
    }
}

#[test]
fn is_valid_scalar_cast_target_matches_active_set() {
    // Per §1.1: scalar cast targets mirror the tensor element set in
    // this cycle (no host-only widths beyond the active dtype set).
    // Deferred f8e4m3 is rejected per §1.1.1.
    for prim in ALL_PRIMS {
        let expected = matches!(
            prim,
            Prim::F32
                | Prim::F64
                | Prim::Bf16
                | Prim::F16
                | Prim::Bool
                | Prim::Int8
                | Prim::Int16
                | Prim::Int32
                | Prim::Int64
        );
        assert_eq!(
            prim.is_valid_scalar_cast_target(),
            expected,
            "is_valid_scalar_cast_target({prim:?}) disagrees with §1.1 \
             active scalar cast set"
        );
    }
}

#[test]
fn is_admissible_active_excludes_only_f8e4m3() {
    for prim in ALL_PRIMS {
        let expected = !matches!(prim, Prim::F8e4m3);
        assert_eq!(
            prim.is_admissible_active(),
            expected,
            "is_admissible_active({prim:?}) disagrees with §1.1.1: only \
             f8e4m3 is deferred, every other variant is admitted"
        );
    }
}

#[test]
fn parse_name_round_trips_through_name() {
    for prim in ALL_PRIMS {
        let name = prim.name();
        assert_eq!(
            Prim::parse_name(name),
            Some(*prim),
            "parse_name({name:?}) must round-trip {prim:?}"
        );
    }
}

// chelis#2413: a random key is an active tensor element dtype with
// runtime storage and the source and Deep spelling `key` (spec/04
// section 1.1). It is not numeric, it is not one of the nine active data
// element dtypes, and it has no cast.
#[test]
fn a_random_key_has_one_spelling_in_source_deep_and_interchange() {
    assert_eq!(Prim::parse_name("key"), Some(Prim::Key));
    assert_eq!(Prim::parse_interchange_name("key"), Some(Prim::Key));
    assert_eq!(Prim::Key.interchange_name(), "key");
    assert!(!Prim::Key.is_numeric());
    assert!(!Prim::Key.is_float());
    assert!(Prim::Key.is_valid_tensor_precision());
    assert!(!Prim::Key.is_data_element_dtype());
    assert!(!Prim::Key.is_valid_scalar_cast_target());
    assert_eq!(Prim::Key.runtime_dtype(), Ok(RuntimeDType::Key));
    assert!(Prim::String.runtime_dtype().is_err());
}

#[test]
fn from_runtime_dtype_inverts_runtime_dtype() {
    for dtype in RuntimeDType::ALL {
        assert_eq!(Prim::from_runtime_dtype(dtype).runtime_dtype(), Ok(dtype));
    }
    for prim in ALL_PRIMS {
        if let Ok(dtype) = prim.runtime_dtype() {
            assert_eq!(Prim::from_runtime_dtype(dtype), *prim);
        }
    }
}

#[test]
fn int16_is_an_active_dtype() {
    // Pin the new variant explicitly so a refactor cannot silently
    // remove it without removing this assertion.
    assert!(Prim::Int16.is_admissible_active());
    assert!(Prim::Int16.is_integer());
    assert!(Prim::Int16.is_numeric());
    assert!(Prim::Int16.is_valid_tensor_precision());
    assert!(Prim::Int16.is_valid_scalar_cast_target());
    assert!(!Prim::Int16.is_float());
    assert_eq!(Prim::Int16.name(), "i16");
    assert_eq!(Prim::parse_name("i16"), Some(Prim::Int16));
}

#[test]
fn f8e4m3_is_deferred_not_active() {
    // §1.1.1 invariants: f8e4m3 still parses (so producers can be
    // diagnosed) but no admissibility predicate accepts it.
    assert_eq!(Prim::parse_name("f8e4m3"), Some(Prim::F8e4m3));
    assert!(!Prim::F8e4m3.is_admissible_active());
    assert!(!Prim::F8e4m3.is_float());
    assert!(!Prim::F8e4m3.is_integer());
    assert!(!Prim::F8e4m3.is_valid_tensor_precision());
    assert!(!Prim::F8e4m3.is_valid_scalar_cast_target());
}
