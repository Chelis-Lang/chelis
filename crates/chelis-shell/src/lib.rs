use bincode::Options;
#[cfg(test)]
use chelis_shell_core::{
    CollectionObligation, FloatClassifyKind, PackageId, ShellModule, ShellSymbol, SymbolKind,
    TypeVariableDomain, TypeVariableRestriction,
};
use chelis_shell_core::{SHELL_FORMAT_VERSION, ShellPackage};
use std::fs;
use std::path::Path;

pub const SHELL_MAGIC: &[u8; 8] = b"CHELCHB\0";

pub fn encode_shell(shell: &ShellPackage) -> Result<Vec<u8>, bincode::Error> {
    chelis_shell_core::validate_shell(shell, &chelis_syntax_host::NativeStack)
        .map_err(metadata_error)?;
    let payload = bincode::serialize(shell)?;
    let mut encoded = Vec::with_capacity(SHELL_MAGIC.len() + 4 + payload.len());
    encoded.extend_from_slice(SHELL_MAGIC);
    encoded.extend_from_slice(&SHELL_FORMAT_VERSION.to_le_bytes());
    encoded.extend_from_slice(&payload);
    Ok(encoded)
}

pub fn decode_shell(bytes: &[u8]) -> Result<ShellPackage, bincode::Error> {
    if !bytes.starts_with(SHELL_MAGIC) {
        return Err(validation_error("unsupported predecessor shell format"));
    }
    let version_bytes: [u8; 4] = bytes
        .get(SHELL_MAGIC.len()..SHELL_MAGIC.len() + 4)
        .ok_or_else(|| validation_error("truncated shell format version"))?
        .try_into()
        .expect("the checked slice has exactly four bytes");
    let version = u32::from_le_bytes(version_bytes);
    if version != SHELL_FORMAT_VERSION {
        return Err(validation_error(&format!(
            "shell format version {version} is unsupported; expected {SHELL_FORMAT_VERSION}; {}regenerate the shell package",
            if version == 6 {
                "Deep 0.20 removes pipe nodes; "
            } else {
                ""
            }
        )));
    }
    let payload = &bytes[SHELL_MAGIC.len() + 4..];
    let shell: ShellPackage = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize(payload)?;
    chelis_shell_core::validate_shell(&shell, &chelis_syntax_host::NativeStack)
        .map_err(metadata_error)?;
    let canonical = encode_shell(&shell)?;
    if canonical != bytes {
        return Err(validation_error(
            "shell bytes are not the canonical encoding of the decoded payload",
        ));
    }
    Ok(shell)
}

pub fn write_shell(path: &Path, shell: &ShellPackage) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = encode_shell(shell)?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn read_shell(path: &Path) -> Result<ShellPackage, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    Ok(decode_shell(&bytes)?)
}

fn validation_error(message: &str) -> bincode::Error {
    Box::new(bincode::ErrorKind::Custom(format!(
        "invalid shell envelope: {message}"
    )))
}

fn metadata_error(error: chelis_shell_core::ShellValidationError) -> bincode::Error {
    Box::new(bincode::ErrorKind::Custom(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_roundtrips() {
        let shell = fixture_shell();

        let bytes = encode_shell(&shell).expect("encode shell");
        let decoded = decode_shell(&bytes).expect("decode shell");
        assert_eq!(decoded, shell);
    }

    #[test]
    fn shell_decode_rejects_trailing_bytes() {
        let mut bytes = encode_shell(&fixture_shell()).expect("encode shell");
        bytes.extend_from_slice(b"junk");

        assert!(
            decode_shell(&bytes).is_err(),
            "CHB decoding must consume the complete input"
        );
    }

    #[test]
    fn shell_decode_rejects_every_truncation() {
        let bytes = encode_shell(&fixture_shell()).expect("encode shell");

        for end in 0..bytes.len() {
            assert!(
                decode_shell(&bytes[..end]).is_err(),
                "truncation at byte {end} unexpectedly decoded"
            );
        }
    }

    #[test]
    fn shell_encoding_has_an_explicit_v7_envelope() {
        let bytes = encode_shell(&fixture_shell()).expect("encode shell");

        assert_eq!(&bytes[..8], b"CHELCHB\0");
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            SHELL_FORMAT_VERSION
        );
        assert_eq!(SHELL_FORMAT_VERSION, 8);
    }

    #[test]
    fn shell_decode_rejects_the_unversioned_predecessor_layout() {
        let predecessor = bincode::serialize(&fixture_shell()).expect("encode old shell payload");

        let error = decode_shell(&predecessor).expect_err("unversioned CHB must be rejected");
        assert!(
            error
                .to_string()
                .contains("unsupported predecessor shell format"),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn shell_decode_rejects_unknown_format_versions() {
        let mut bytes = encode_shell(&fixture_shell()).expect("encode shell");
        bytes[8..12].copy_from_slice(&99_u32.to_le_bytes());

        let error = decode_shell(&bytes).expect_err("unknown CHB version must be rejected");
        assert_eq!(
            error.to_string(),
            "invalid shell envelope: shell format version 99 is unsupported; expected 8; regenerate the shell package"
        );
    }

    #[test]
    fn shell_decode_rejects_the_pre_collection_contract_version() {
        let mut bytes = encode_shell(&fixture_shell()).expect("encode shell");
        bytes[8..12].copy_from_slice(&4_u32.to_le_bytes());

        let error = decode_shell(&bytes).expect_err("CHB v4 must not decode without relations");
        assert_eq!(
            error.to_string(),
            "invalid shell envelope: shell format version 4 is unsupported; expected 8; regenerate the shell package"
        );
    }

    /// chelis#2443 took the envelope to 6, because `TypeVariableDomain` gained
    /// spec/04 §5.9's explicit dtype set and a v5 reader cannot decode it.
    /// A v5 shell must therefore be refused rather than read as if the domain
    /// vocabulary were unchanged.
    #[test]
    fn shell_decode_rejects_the_pre_dtype_set_domain_version() {
        let mut bytes = encode_shell(&fixture_shell()).expect("encode shell");
        bytes[8..12].copy_from_slice(&5_u32.to_le_bytes());

        let error = decode_shell(&bytes).expect_err("CHB v5 predates the dtype-set domain");
        assert_eq!(
            error.to_string(),
            "invalid shell envelope: shell format version 5 is unsupported; expected 8; regenerate the shell package"
        );
    }

    #[test]
    fn shell_decode_rejects_the_pre_pipe_normalization_version_before_payload_decode() {
        let mut bytes = b"CHELCHB\0".to_vec();
        bytes.extend_from_slice(&6_u32.to_le_bytes());
        let error =
            decode_shell(&bytes).expect_err("v6 must be refused before reading its payload");
        assert_eq!(
            error.to_string(),
            "invalid shell envelope: shell format version 6 is unsupported; expected 8; Deep 0.20 removes pipe nodes; regenerate the shell package"
        );
    }

    #[test]
    fn checked_collection_contracts_round_trip_and_change_identity() {
        let mut obligated = fixture_shell();
        let symbol = &mut obligated.modules[0].exports[0];
        symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
        symbol.collection_obligations = vec![CollectionObligation::Len {
            operand: "(t-var {} t0)".into(),
            result: "(t-var {} t1)".into(),
        }];

        let encoded = encode_shell(&obligated).expect("checked contract encodes");
        assert_eq!(decode_shell(&encoded).unwrap(), obligated);
        assert_ne!(encoded, encode_shell(&fixture_shell()).unwrap());
    }

    #[test]
    fn float_classification_contracts_round_trip_and_reject_previous_version() {
        for kind in [
            FloatClassifyKind::Nan,
            FloatClassifyKind::Finite,
            FloatClassifyKind::Infinite,
        ] {
            let mut shell = fixture_shell();
            let symbol = &mut shell.modules[0].exports[0];
            symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
            symbol.collection_obligations = vec![CollectionObligation::FloatClassify {
                kind,
                operand: "(t-var {} t0)".into(),
                result: "(t-var {} t1)".into(),
            }];
            let bytes = encode_shell(&shell).unwrap();
            assert_eq!(decode_shell(&bytes).unwrap(), shell);
        }
        let mut old = b"CHELCHB\0".to_vec();
        old.extend_from_slice(&7u32.to_le_bytes());
        assert!(
            decode_shell(&old)
                .unwrap_err()
                .to_string()
                .contains("expected 8")
        );
    }

    #[test]
    fn key_operation_contracts_round_trip_and_change_identity() {
        for obligation in [
            CollectionObligation::KeyFromSeed {
                operand: "(t-var {} t0)".into(),
                result: "(t-var {} t1)".into(),
            },
            CollectionObligation::SplitKey {
                operand: "(t-var {} t0)".into(),
                result: "(t-var {} t1)".into(),
            },
            CollectionObligation::SplitKeys {
                operand: "(t-var {} t0)".into(),
                count: "(t-prim {} i64)".into(),
                result: "(t-var {} t1)".into(),
            },
            CollectionObligation::FoldIn {
                operand: "(t-var {} t0)".into(),
                index: "(t-var {} t0)".into(),
                result: "(t-var {} t1)".into(),
            },
        ] {
            let mut obligated = fixture_shell();
            let symbol = &mut obligated.modules[0].exports[0];
            symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
            symbol.collection_obligations = vec![obligation];
            let encoded = encode_shell(&obligated).unwrap();
            assert_eq!(decode_shell(&encoded).unwrap(), obligated);
            assert_ne!(encoded, encode_shell(&fixture_shell()).unwrap());
        }
    }

    #[test]
    fn checked_collection_contract_ledger_is_strictly_canonical() {
        let append = CollectionObligation::Append {
            list: "(t-var {} t0)".into(),
            value: "(t-var {} t1)".into(),
            result: "(t-var {} t0)".into(),
        };
        let len = CollectionObligation::Len {
            operand: "(t-var {} t0)".into(),
            result: "(t-var {} t1)".into(),
        };

        let mut canonical = fixture_shell();
        let symbol = &mut canonical.modules[0].exports[0];
        symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
        symbol.collection_obligations = vec![append.clone(), len.clone()];
        let encoded = encode_shell(&canonical).expect("canonical relation ledger must encode");
        assert_eq!(
            decode_shell(&encoded).expect("canonical relation ledger must decode"),
            canonical
        );

        for (label, obligations) in [
            ("duplicate", vec![len.clone(), len.clone()]),
            ("noncanonical order", vec![len.clone(), append.clone()]),
        ] {
            let mut invalid = fixture_shell();
            let symbol = &mut invalid.modules[0].exports[0];
            symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
            symbol.collection_obligations = obligations;

            let error =
                chelis_shell_core::validate_shell(&invalid, &chelis_syntax_host::NativeStack)
                    .expect_err("noncanonical relation ledger must fail validation");
            assert!(
                error.to_string().contains("strictly sorted"),
                "unexpected {label} validation diagnostic: {error}"
            );
            assert!(
                encode_shell(&invalid).is_err(),
                "{label} relation ledger must not encode"
            );
            assert!(
                decode_shell(&encode_unvalidated(&invalid)).is_err(),
                "{label} relation ledger must not decode"
            );
        }

        for (label, mutate) in [
            ("operand whitespace", 0usize),
            ("result whitespace", 1usize),
        ] {
            let mut invalid = fixture_shell();
            let symbol = &mut invalid.modules[0].exports[0];
            symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
            let mut noncanonical = len.clone();
            if let CollectionObligation::Len { operand, result } = &mut noncanonical {
                if mutate == 0 {
                    *operand = "(t-var   {} t0)".into();
                } else {
                    *result = "(t-var {}   t1)".into();
                }
            }
            symbol.collection_obligations = vec![noncanonical];

            let error =
                chelis_shell_core::validate_shell(&invalid, &chelis_syntax_host::NativeStack)
                    .expect_err("noncanonical Deep must fail validation");
            assert!(
                error.to_string().contains("canonical Deep"),
                "unexpected {label} validation diagnostic: {error}"
            );
            assert!(
                encode_shell(&invalid).is_err(),
                "{label} relation ledger must not encode"
            );
            assert!(
                decode_shell(&encode_unvalidated(&invalid)).is_err(),
                "{label} relation ledger must not decode"
            );
        }
    }

    #[test]
    fn checked_collection_contracts_reject_hidden_or_malformed_variables() {
        let mut hidden = fixture_shell();
        hidden.modules[0].exports[0].collection_obligations = vec![CollectionObligation::Len {
            operand: "(t-var {} t0)".into(),
            result: "(t-prim {} i64)".into(),
        }];
        assert!(
            chelis_shell_core::validate_shell(&hidden, &chelis_syntax_host::NativeStack)
                .unwrap_err()
                .to_string()
                .contains("absent from its type representation")
        );

        let mut malformed = fixture_shell();
        malformed.modules[0].exports[0].collection_obligations = vec![CollectionObligation::Len {
            operand: "List[t0]".into(),
            result: "(t-prim {} i64)".into(),
        }];
        assert!(
            chelis_shell_core::validate_shell(&malformed, &chelis_syntax_host::NativeStack)
                .unwrap_err()
                .to_string()
                .contains("not valid Deep")
        );

        for (operand, variable) in [
            (
                "(t-tensor {} (d-var {} d1) (d-rank {} r0) (t-prim {} f32))",
                "`d1`",
            ),
            (
                "(t-tensor {} (d-var {} d0) (d-rank {} r1) (t-prim {} f32))",
                "`r1`",
            ),
        ] {
            let mut hidden_shape = fixture_shell();
            let symbol = &mut hidden_shape.modules[0].exports[0];
            symbol.type_repr = Some(
                "(t-fn {} (t-tensor {} (d-var {} d0) (d-rank {} r0) (t-prim {} f32)) (t-prim {} i64))"
                    .into(),
            );
            symbol.collection_obligations = vec![CollectionObligation::Len {
                operand: operand.into(),
                result: "(t-prim {} i64)".into(),
            }];
            let error =
                chelis_shell_core::validate_shell(&hidden_shape, &chelis_syntax_host::NativeStack)
                    .unwrap_err();
            assert!(
                error.to_string().contains(variable),
                "unexpected hidden-variable diagnostic: {error}"
            );
        }
    }

    #[test]
    fn operation_value_domains_round_trip_without_becoming_dtype_bounds() {
        for domain in [
            TypeVariableDomain::FloatValue,
            TypeVariableDomain::IntValue,
            TypeVariableDomain::NumericValue,
        ] {
            let mut shell = fixture_shell();
            let symbol = &mut shell.modules[0].exports[0];
            symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t0))".into());
            symbol.type_variable_restrictions = vec![TypeVariableRestriction {
                variable: "t0".into(),
                domain,
            }];
            let encoded = encode_shell(&shell).unwrap();
            assert_eq!(decode_shell(&encoded).unwrap(), shell);
            let mut old_version = encoded;
            old_version[8..12].copy_from_slice(&3_u32.to_le_bytes());
            assert!(decode_shell(&old_version).is_err());
        }
    }

    #[test]
    fn shell_validation_rejects_a_restriction_detached_from_its_type_variable() {
        let mut shell = fixture_shell();
        shell.modules[0].exports[0].type_variable_restrictions = vec![TypeVariableRestriction {
            variable: "t1".to_string(),
            domain: TypeVariableDomain::ActiveFloat,
        }];

        let error = chelis_shell_core::validate_shell(&shell, &chelis_syntax_host::NativeStack)
            .expect_err("detached restriction must be rejected");
        assert!(
            error.to_string().contains(
                "restricts `t1` but that canonical variable is absent from its type representation"
            ),
            "unexpected diagnostic: {error}"
        );

        let mut metadata_only = fixture_shell();
        let symbol = &mut metadata_only.modules[0].exports[0];
        symbol.type_repr = Some("(t-prim {type: (t-var {} t0)} f32)".to_string());
        symbol.type_variable_restrictions = vec![TypeVariableRestriction {
            variable: "t0".to_string(),
            domain: TypeVariableDomain::ActiveFloat,
        }];
        let error =
            chelis_shell_core::validate_shell(&metadata_only, &chelis_syntax_host::NativeStack)
                .expect_err("a metadata-only variable is not part of the quantified type");
        assert!(
            error.to_string().contains(
                "restricts `t0` but that canonical variable is absent from its type representation"
            ),
            "unexpected diagnostic: {error}"
        );
    }

    #[test]
    fn shell_validation_rejects_malformed_type_representations_for_every_scheme_shape() {
        let malformed = [
            ("invalid syntax", "(t-unit {}"),
            ("multiple expressions", "(t-unit {}) (t-unit {})"),
            ("unknown root", "(not-a-type {} f32)"),
            ("bare atom", "f32"),
            ("bare list", "()"),
            ("bare map", "{type: (t-prim {} f32)}"),
            (
                "legacy metadata-expression wrapper",
                "^{:doc \"legacy\"} (t-prim {} f32)",
            ),
            ("runtime root", "(lit {type: (t-var {} t0)} 1)"),
            (
                "nested runtime node",
                "(t-fn {} (lit {type: (t-var {} t0)} 1) (t-var {} t0))",
            ),
            (
                "tensor namespace swap",
                "(t-tensor {} (t-var {} t0) (d-rank {} r0))",
            ),
        ];

        for has_body in [false, true] {
            for restricted in [false, true] {
                for (label, source) in malformed {
                    let mut shell = fixture_shell();
                    let symbol = &mut shell.modules[0].exports[0];
                    symbol.has_body = has_body;
                    symbol.type_repr = Some(source.to_string());
                    symbol.type_variable_restrictions = restricted
                        .then(|| TypeVariableRestriction {
                            variable: "t0".to_string(),
                            domain: TypeVariableDomain::ActiveFloat,
                        })
                        .into_iter()
                        .collect();

                    let shape = format!("{label}, has_body={has_body}, restricted={restricted}");
                    assert!(
                        chelis_shell_core::validate_shell(&shell, &chelis_syntax_host::NativeStack)
                            .is_err(),
                        "validate_shell admitted {shape}"
                    );
                    assert!(
                        encode_shell(&shell).is_err(),
                        "encode_shell admitted {shape}"
                    );
                    assert!(
                        decode_shell(&encode_unvalidated(&shell)).is_err(),
                        "decode_shell admitted {shape}"
                    );
                }
            }
        }
    }

    #[test]
    fn shell_validation_requires_value_signatures_and_accepts_valid_authored_fallbacks() {
        for has_body in [false, true] {
            for restricted in [false, true] {
                let mut shell = fixture_shell();
                let symbol = &mut shell.modules[0].exports[0];
                symbol.has_body = has_body;
                symbol.type_repr = Some(if restricted {
                    "(t-fn {} (t-var {} t0) (t-var {} t0))".to_string()
                } else {
                    "(t-fn {} (t-prim {} f32) (t-prim {} f32))".to_string()
                });
                symbol.type_variable_restrictions = restricted
                    .then(|| TypeVariableRestriction {
                        variable: "t0".to_string(),
                        domain: TypeVariableDomain::ActiveFloat,
                    })
                    .into_iter()
                    .collect();

                chelis_shell_core::validate_shell(&shell, &chelis_syntax_host::NativeStack).unwrap_or_else(|error| {
                    panic!(
                        "valid signature rejected for has_body={has_body}, restricted={restricted}: {error}"
                    )
                });
                decode_shell(&encode_shell(&shell).expect("encode valid shell"))
                    .expect("decode valid shell");
            }
        }

        for has_body in [false, true] {
            let mut shell = fixture_shell();
            let symbol = &mut shell.modules[0].exports[0];
            symbol.has_body = has_body;
            symbol.type_repr = None;
            assert!(
                chelis_shell_core::validate_shell(&shell, &chelis_syntax_host::NativeStack)
                    .is_err(),
                "value export without a type representation was admitted (has_body={has_body})"
            );
        }

        for kind in [SymbolKind::Type, SymbolKind::Macro, SymbolKind::Dim] {
            let mut shell = fixture_shell();
            let symbol = &mut shell.modules[0].exports[0];
            symbol.kind = kind;
            symbol.type_repr = None;
            symbol.has_body = false;
            chelis_shell_core::validate_shell(&shell, &chelis_syntax_host::NativeStack)
                .unwrap_or_else(|error| {
                    panic!("{kind:?} export without a value type failed: {error}")
                });
        }
    }

    fn encode_unvalidated(shell: &ShellPackage) -> Vec<u8> {
        let payload = bincode::serialize(shell).expect("serialize malformed test payload");
        let mut bytes = Vec::with_capacity(SHELL_MAGIC.len() + 4 + payload.len());
        bytes.extend_from_slice(SHELL_MAGIC);
        bytes.extend_from_slice(&SHELL_FORMAT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn fixture_shell() -> ShellPackage {
        ShellPackage {
            format_version: SHELL_FORMAT_VERSION,
            package: PackageId {
                name: "chelis-std".to_string(),
                version: "0.1.0".to_string(),
            },
            compiler: "=0.1.21".to_string(),
            modules: vec![ShellModule {
                module: "App.Demo".to_string(),
                exports: vec![ShellSymbol {
                    name: "forward".to_string(),
                    kind: SymbolKind::Value,
                    type_repr: Some("(t-fn {} (t-prim {} f32) (t-prim {} f32))".to_string()),
                    type_variable_restrictions: Vec::new(),
                    collection_obligations: Vec::new(),
                    effects: Vec::new(),
                    has_body: true,
                }],
            }],
            dependencies: Vec::new(),
            archive_sha256: "d".repeat(64),
        }
    }
}
