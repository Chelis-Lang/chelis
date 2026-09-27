use chelis_vocab::{
    DiagnosticKind, DiagnosticKindDecodeError, EffectKind, EffectKindDecodeError, EffectKindInput,
    Repr, RuntimeDType, RuntimeDTypeDecodeError,
};

#[test]
fn diagnostic_kind_wire_spellings_are_closed_and_stable() {
    let expected = [
        (DiagnosticKind::SurfParseError, "surf_parse_error"),
        (DiagnosticKind::DeepParseError, "deep_parse_error"),
        (DiagnosticKind::MacroError, "macro_error"),
        (DiagnosticKind::NameResolutionError, "name_resolution_error"),
        (DiagnosticKind::DeepDeclError, "deep_decl_error"),
        (DiagnosticKind::DuplicateName, "duplicate_name"),
        (DiagnosticKind::PreimageMismatch, "preimage_mismatch"),
        (DiagnosticKind::CascadeIncomplete, "cascade_incomplete"),
        (DiagnosticKind::TypeError, "type_error"),
        (DiagnosticKind::EffectError, "effect_error"),
        (DiagnosticKind::LinearityError, "linearity_error"),
        (DiagnosticKind::LowerError, "lower_error"),
        (DiagnosticKind::ReefError, "reef_error"),
        (DiagnosticKind::EvalError, "eval_error"),
        (DiagnosticKind::NumericTrap, "numeric_trap"),
        (DiagnosticKind::Cancelled, "cancelled"),
        (DiagnosticKind::GradError, "grad_error"),
        (DiagnosticKind::ValidationError, "validation_error"),
        (DiagnosticKind::UnknownName, "unknown_name"),
        (
            DiagnosticKind::UnknownSchemaVersion,
            "unknown_schema_version",
        ),
        (DiagnosticKind::HashError, "hash_error"),
        (DiagnosticKind::InvalidRequest, "invalid_request"),
        (DiagnosticKind::CompileError, "compile_error"),
        (DiagnosticKind::GeneralOther, "other"),
        (DiagnosticKind::UnsupportedFeature, "unsupported_feature"),
        (DiagnosticKind::TypeMismatch, "TypeMismatch"),
        (DiagnosticKind::PrecisionMismatch, "PrecisionMismatch"),
        (DiagnosticKind::DimensionMismatch, "DimensionMismatch"),
        (DiagnosticKind::ArityMismatch, "ArityMismatch"),
        (DiagnosticKind::UnboundVariable, "UnboundVariable"),
        (DiagnosticKind::UnknownConstructor, "UnknownConstructor"),
        (DiagnosticKind::NotAFunction, "NotAFunction"),
        (DiagnosticKind::NonExhaustiveMatch, "NonExhaustiveMatch"),
        (DiagnosticKind::OccursCheck, "OccursCheck"),
        (DiagnosticKind::CastNonTensor, "CastNonTensor"),
        (
            DiagnosticKind::TupleIndexOutOfBounds,
            "TupleIndexOutOfBounds",
        ),
        (DiagnosticKind::UseAfterConsume, "UseAfterConsume"),
        (DiagnosticKind::UnconsumedLinear, "UnconsumedLinear"),
        (DiagnosticKind::InvalidBorrow, "InvalidBorrow"),
        (DiagnosticKind::KeyReuse, "KeyReuse"),
        (DiagnosticKind::CycleDetected, "CycleDetected"),
        (
            DiagnosticKind::UnsupportedTensorPrecision,
            "UnsupportedTensorPrecision",
        ),
        (DiagnosticKind::DuplicateDefinition, "DuplicateDefinition"),
        (DiagnosticKind::DuplicateModule, "DuplicateModule"),
        (DiagnosticKind::OpaqueTypeViolation, "OpaqueTypeViolation"),
        (DiagnosticKind::ReservedLinkerName, "ReservedLinkerName"),
        (DiagnosticKind::BuiltinShadowing, "BuiltinShadowing"),
        (DiagnosticKind::UnknownForm, "UnknownForm"),
        (DiagnosticKind::MalformedForm, "MalformedForm"),
        (DiagnosticKind::CheckOther, "Other"),
        (DiagnosticKind::UnhandledEffect, "UnhandledEffect"),
        (DiagnosticKind::InvalidHandler, "InvalidHandler"),
        (DiagnosticKind::BuildTargetMismatch, "BuildTargetMismatch"),
        (DiagnosticKind::TypeTotality, "TypeTotality"),
        (DiagnosticKind::DirectoryWalkError, "directory_walk_error"),
        (DiagnosticKind::EmptyCorpus, "empty_corpus"),
        (DiagnosticKind::EmptyTestSelection, "empty_test_selection"),
    ];
    assert_eq!(DiagnosticKind::ALL, expected.map(|(kind, _)| kind));

    for (kind, spelling) in expected {
        assert_eq!(kind.as_str(), spelling);
        assert_eq!(DiagnosticKind::decode(spelling), Ok(kind));
    }
}

#[test]
fn diagnostic_kind_unknown_spelling_is_an_error_not_a_default() {
    let input = String::from("unsupported");
    let error = DiagnosticKind::decode(&input).expect_err("unknown diagnostic kind");
    assert_eq!(
        error,
        DiagnosticKindDecodeError::Unknown { spelling: &input }
    );
    let DiagnosticKindDecodeError::Unknown { spelling } = error;
    assert_eq!(
        spelling.as_ptr(),
        input.as_ptr(),
        "the error must borrow input"
    );
}

#[test]
fn diagnostic_kind_consumer_match_is_a_compile_time_ratchet() {
    fn classify(kind: DiagnosticKind) -> &'static str {
        match kind {
            DiagnosticKind::UnsupportedFeature => "unsupported",
            DiagnosticKind::SurfParseError
            | DiagnosticKind::DeepParseError
            | DiagnosticKind::MacroError
            | DiagnosticKind::NameResolutionError
            | DiagnosticKind::DeepDeclError
            | DiagnosticKind::DuplicateName
            | DiagnosticKind::PreimageMismatch
            | DiagnosticKind::CascadeIncomplete
            | DiagnosticKind::TypeError
            | DiagnosticKind::EffectError
            | DiagnosticKind::LinearityError
            | DiagnosticKind::LowerError
            | DiagnosticKind::ReefError
            | DiagnosticKind::EvalError
            | DiagnosticKind::NumericTrap
            | DiagnosticKind::Cancelled
            | DiagnosticKind::GradError
            | DiagnosticKind::ValidationError
            | DiagnosticKind::UnknownName
            | DiagnosticKind::UnknownSchemaVersion
            | DiagnosticKind::HashError
            | DiagnosticKind::InvalidRequest
            | DiagnosticKind::CompileError
            | DiagnosticKind::GeneralOther
            | DiagnosticKind::TypeMismatch
            | DiagnosticKind::PrecisionMismatch
            | DiagnosticKind::DimensionMismatch
            | DiagnosticKind::ArityMismatch
            | DiagnosticKind::UnboundVariable
            | DiagnosticKind::UnknownConstructor
            | DiagnosticKind::NotAFunction
            | DiagnosticKind::NonExhaustiveMatch
            | DiagnosticKind::OccursCheck
            | DiagnosticKind::CastNonTensor
            | DiagnosticKind::TupleIndexOutOfBounds
            | DiagnosticKind::UseAfterConsume
            | DiagnosticKind::UnconsumedLinear
            | DiagnosticKind::InvalidBorrow
            | DiagnosticKind::KeyReuse
            | DiagnosticKind::CycleDetected
            | DiagnosticKind::UnsupportedTensorPrecision
            | DiagnosticKind::DuplicateDefinition
            | DiagnosticKind::DuplicateModule
            | DiagnosticKind::OpaqueTypeViolation
            | DiagnosticKind::ReservedLinkerName
            | DiagnosticKind::BuiltinShadowing
            | DiagnosticKind::UnknownForm
            | DiagnosticKind::MalformedForm
            | DiagnosticKind::CheckOther
            | DiagnosticKind::UnhandledEffect
            | DiagnosticKind::InvalidHandler
            | DiagnosticKind::BuildTargetMismatch
            | DiagnosticKind::TypeTotality
            | DiagnosticKind::DirectoryWalkError
            | DiagnosticKind::EmptyCorpus
            | DiagnosticKind::EmptyTestSelection => "general",
        }
    }

    for kind in DiagnosticKind::ALL {
        assert!(!classify(kind).is_empty());
    }
}

#[test]
fn effect_kind_canonical_symbols_round_trip() {
    let expected = [(EffectKind::Resource, "resource")];
    assert_eq!(EffectKind::ALL, expected.map(|(kind, _)| kind));

    for (kind, symbol) in expected {
        assert_eq!(kind.symbol(), symbol);
        assert_eq!(
            EffectKind::decode(EffectKindInput::Symbol(symbol)),
            Ok(kind)
        );
    }
}

#[test]
fn effect_kind_missing_malformed_and_unknown_are_distinct_errors() {
    assert_eq!(
        EffectKind::decode(EffectKindInput::Missing),
        Err(EffectKindDecodeError::Missing)
    );
    assert_eq!(
        EffectKind::decode(EffectKindInput::Malformed),
        Err(EffectKindDecodeError::Malformed)
    );

    let input = String::from("teleport");
    let error = EffectKind::decode(EffectKindInput::Symbol(&input)).expect_err("unknown effect");
    let EffectKindDecodeError::Unknown { symbol } = error else {
        panic!("expected an unknown-symbol error")
    };
    assert_eq!(symbol, input);
    assert_eq!(
        symbol.as_ptr(),
        input.as_ptr(),
        "the error must borrow the input"
    );
}

#[test]
fn effect_kind_random_is_a_retired_spelling_that_points_at_keys() {
    // The `random` handler kind was retired with the counter stream (#2413):
    // randomness flows through explicit key values, so `random` decodes as
    // the typed retired spelling, which names the key operations.
    let error = EffectKind::decode(EffectKindInput::Symbol("random")).expect_err("retired kind");
    assert_eq!(error, EffectKindDecodeError::RetiredRandom);
    let text = error.to_string();
    assert!(
        text.contains("`random` is retired") && text.contains("key_from_seed"),
        "{text}"
    );
}

#[test]
fn effect_kind_decoder_never_returns_option_or_a_default_kind() {
    let decoded: Result<EffectKind, EffectKindDecodeError<'_>> =
        EffectKind::decode(EffectKindInput::Symbol("not-a-kind"));
    assert!(matches!(
        decoded,
        Err(EffectKindDecodeError::Unknown { .. })
    ));
}

#[test]
fn effect_kind_consumer_match_is_a_compile_time_ratchet() {
    fn semantic_decision(kind: EffectKind) -> &'static str {
        match kind {
            EffectKind::Resource => "device scope",
        }
    }

    for kind in EffectKind::ALL {
        assert!(!semantic_decision(kind).is_empty());
    }
}

#[test]
fn representation_variants_have_current_widths_and_payload_status() {
    let expected = [
        (Repr::Ieee754Binary16, 2, false),
        (Repr::Ieee754Binary32, 4, false),
        (Repr::Ieee754Binary64, 8, false),
        (Repr::Bfloat16, 2, false),
        (Repr::TwosComplement8, 1, false),
        (Repr::TwosComplement16, 2, false),
        (Repr::TwosComplement32, 4, false),
        (Repr::TwosComplement64, 8, false),
        (Repr::Bool8, 1, false),
        (Repr::Word64, 8, false),
    ];
    assert_eq!(Repr::ALL, expected.map(|(repr, ..)| repr));

    for (repr, byte_width, payload_encoded) in expected {
        assert_eq!(repr.byte_width(), byte_width);
        assert_eq!(repr.is_payload_encoded(), payload_encoded);
    }
}

#[test]
fn equal_width_representations_keep_distinct_identities() {
    assert_eq!(
        Repr::Ieee754Binary32.byte_width(),
        Repr::TwosComplement32.byte_width()
    );
    assert_ne!(Repr::Ieee754Binary32, Repr::TwosComplement32);
    assert_ne!(Repr::Bool8, Repr::TwosComplement8);
    assert_eq!(
        Repr::Word64.byte_width(),
        Repr::TwosComplement64.byte_width()
    );
    assert_ne!(Repr::Word64, Repr::TwosComplement64);
}

#[test]
fn representation_consumer_match_is_a_compile_time_ratchet() {
    fn encoding_name(repr: Repr) -> &'static str {
        match repr {
            Repr::Ieee754Binary16 => "ieee754-binary16",
            Repr::Ieee754Binary32 => "ieee754-binary32",
            Repr::Ieee754Binary64 => "ieee754-binary64",
            Repr::Bfloat16 => "bfloat16",
            Repr::TwosComplement8 => "twos-complement-8",
            Repr::TwosComplement16 => "twos-complement-16",
            Repr::TwosComplement32 => "twos-complement-32",
            Repr::TwosComplement64 => "twos-complement-64",
            Repr::Bool8 => "bool8",
            Repr::Word64 => "word64",
        }
    }

    for repr in Repr::ALL {
        assert!(!encoding_name(repr).is_empty());
    }
}

#[test]
fn runtime_dtype_ids_names_macros_representations_and_widths_round_trip() {
    let expected = [
        (
            RuntimeDType::F32,
            0,
            "f32",
            "CHELIS_DTYPE_F32",
            Repr::Ieee754Binary32,
            4,
        ),
        (
            RuntimeDType::F64,
            1,
            "f64",
            "CHELIS_DTYPE_F64",
            Repr::Ieee754Binary64,
            8,
        ),
        (
            RuntimeDType::I32,
            2,
            "int32",
            "CHELIS_DTYPE_I32",
            Repr::TwosComplement32,
            4,
        ),
        (
            RuntimeDType::Bool,
            3,
            "bool",
            "CHELIS_DTYPE_BOOL",
            Repr::Bool8,
            1,
        ),
        (
            RuntimeDType::I64,
            4,
            "int64",
            "CHELIS_DTYPE_I64",
            Repr::TwosComplement64,
            8,
        ),
        (
            RuntimeDType::Bf16,
            5,
            "bf16",
            "CHELIS_DTYPE_BF16",
            Repr::Bfloat16,
            2,
        ),
        (
            RuntimeDType::F16,
            6,
            "f16",
            "CHELIS_DTYPE_F16",
            Repr::Ieee754Binary16,
            2,
        ),
        (
            RuntimeDType::I8,
            7,
            "int8",
            "CHELIS_DTYPE_I8",
            Repr::TwosComplement8,
            1,
        ),
        (
            RuntimeDType::I16,
            8,
            "int16",
            "CHELIS_DTYPE_I16",
            Repr::TwosComplement16,
            2,
        ),
        (
            RuntimeDType::Key,
            9,
            "key",
            "CHELIS_DTYPE_KEY",
            Repr::Word64,
            8,
        ),
    ];
    assert_eq!(RuntimeDType::ALL, expected.map(|(dtype, ..)| dtype));

    for (dtype, id, name, c_macro, repr, byte_width) in expected {
        assert_eq!(dtype.id(), id);
        assert_eq!(dtype.name(), name);
        assert_eq!(dtype.c_macro(), c_macro);
        assert_eq!(dtype.repr(), repr);
        assert_eq!(dtype.byte_width(), byte_width);
        assert_eq!(dtype.byte_width(), dtype.repr().byte_width());
        assert_eq!(RuntimeDType::decode_id(id), Ok(dtype));
    }
}

#[test]
fn runtime_dtype_invalid_ids_are_errors_not_f32() {
    let first_unknown = RuntimeDType::ALL
        .iter()
        .map(|dtype| dtype.id())
        .max()
        .expect("non-empty runtime dtype vocabulary")
        + 1;
    for id in [-1, first_unknown, i32::MIN, i32::MAX] {
        assert_eq!(
            RuntimeDType::decode_id(id),
            Err(RuntimeDTypeDecodeError::InvalidId { id })
        );
    }
}

#[test]
fn runtime_dtype_ids_are_pairwise_distinct() {
    let all = RuntimeDType::ALL;
    for (index, left) in all.iter().enumerate() {
        for right in all.iter().skip(index.saturating_add(1)) {
            assert_ne!(
                left.id(),
                right.id(),
                "{left:?} and {right:?} share ABI tag {}",
                left.id()
            );
        }
    }
}

#[test]
fn runtime_dtype_consumer_match_is_a_compile_time_ratchet() {
    fn storage_family(dtype: RuntimeDType) -> &'static str {
        match dtype {
            RuntimeDType::F32 => "f32",
            RuntimeDType::F64 => "f64",
            RuntimeDType::I32 => "i32",
            RuntimeDType::Bool => "f32-bool",
            RuntimeDType::I64 => "i64",
            RuntimeDType::Bf16 => "u16-bf16",
            RuntimeDType::F16 => "u16-f16",
            RuntimeDType::I8 => "i8",
            RuntimeDType::I16 => "i16",
            RuntimeDType::Key => "u64-key",
        }
    }

    for dtype in RuntimeDType::ALL {
        assert!(!storage_family(dtype).is_empty());
    }
}
