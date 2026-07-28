use chelis_vocab::{
    EffectKind, EffectKindDecodeError, EffectKindInput, Repr, RuntimeDType, RuntimeDTypeDecodeError,
};

#[test]
fn effect_kind_canonical_symbols_round_trip() {
    let expected = [
        (EffectKind::Random, "random"),
        (EffectKind::Resource, "resource"),
    ];
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
    assert_eq!(
        EffectKind::decode(EffectKindInput::Symbol("teleport")),
        Err(EffectKindDecodeError::Unknown { symbol: "teleport" })
    );
}

#[test]
fn effect_kind_decoder_never_returns_option_or_a_default_kind() {
    let decoded: Result<EffectKind, EffectKindDecodeError> =
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
            EffectKind::Random => "seed scope",
            EffectKind::Resource => "device scope",
        }
    }

    for kind in EffectKind::ALL {
        assert!(!semantic_decision(kind).is_empty());
    }
}

#[test]
fn runtime_dtype_ids_names_c_macros_and_widths_round_trip() {
    let expected = [
        (RuntimeDType::F32, 0, "f32", "CHELIS_F32", 4),
        (RuntimeDType::F64, 1, "f64", "CHELIS_F64", 8),
        (RuntimeDType::I32, 2, "int32", "CHELIS_I32", 4),
        (RuntimeDType::Bool, 3, "bool", "CHELIS_BOOL", 4),
        (RuntimeDType::I64, 4, "int64", "CHELIS_I64", 8),
        (RuntimeDType::Bf16, 5, "bf16", "CHELIS_BF16", 2),
        (RuntimeDType::F16, 6, "f16", "CHELIS_F16", 2),
        (RuntimeDType::I8, 7, "int8", "CHELIS_I8", 1),
        (RuntimeDType::I16, 8, "int16", "CHELIS_I16", 2),
    ];
    assert_eq!(RuntimeDType::ALL, expected.map(|(dtype, ..)| dtype));

    for (dtype, id, name, c_macro, byte_width) in expected {
        assert_eq!(dtype.id(), id);
        assert_eq!(dtype.name(), name);
        assert_eq!(dtype.c_macro(), c_macro);
        assert_eq!(dtype.byte_width(), byte_width);
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

/// The point of naming representations: width does not determine encoding.
///
/// If this ever fails because no two distinct representations share a width,
/// the projection has become injective and comparing widths would be safe
/// again. Until then a width comparison can pass while two lanes hold
/// incompatible bytes, which is the failure mode `Repr` exists to express.
#[test]
fn byte_width_is_a_lossy_projection_of_representation() {
    let collision = Repr::ALL.iter().any(|left| {
        Repr::ALL
            .iter()
            .any(|right| left != right && left.byte_width() == right.byte_width())
    });
    assert!(
        collision,
        "distinct representations sharing a width is what makes width \
         comparison unsound; if none remain, revisit the Repr abstraction"
    );

    // The concrete instance that motivated this: both four bytes, and a
    // memcpy between them is silent corruption.
    assert_eq!(
        Repr::TwosComplement32.byte_width(),
        Repr::BoolInBinary32.byte_width()
    );
    assert_ne!(Repr::TwosComplement32, Repr::BoolInBinary32);
}

/// Width is derived, so a dtype and its representation cannot disagree.
#[test]
fn dtype_byte_width_agrees_with_its_representation() {
    for dtype in RuntimeDType::ALL {
        assert_eq!(dtype.byte_width(), dtype.repr().byte_width());
    }
}

/// Only bool is payload-encoded today. This is the tracked
/// `CRuntime-BoolStorage-F1` deviation; when it closes, this is the test that
/// should change, deliberately.
#[test]
fn bool_is_the_only_payload_encoded_dtype() {
    let payload: Vec<RuntimeDType> = RuntimeDType::ALL
        .into_iter()
        .filter(|d| d.repr().is_payload_encoded())
        .collect();
    assert_eq!(payload, vec![RuntimeDType::Bool]);
    assert_eq!(RuntimeDType::Bool.repr(), Repr::BoolInBinary32);
    // int32 is native two's complement, contradicting a stale comment in
    // `chelis_tensor_cmplt`. `read_scalar_as_f64` reads it as `*const i32`.
    assert_eq!(RuntimeDType::I32.repr(), Repr::TwosComplement32);
}

/// Injectivity. `exhaustive_tag_domain.rs` establishes that nothing outside
/// this set decodes; this establishes that nothing inside it collides.
#[test]
fn runtime_dtype_ids_are_pairwise_distinct() {
    let all = RuntimeDType::ALL;
    for (i, left) in all.iter().enumerate() {
        for right in all.iter().skip(i + 1) {
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
        }
    }

    for dtype in RuntimeDType::ALL {
        assert!(!storage_family(dtype).is_empty());
    }
}
