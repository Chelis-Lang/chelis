mod common;

use chelis_shell::{SHELL_MAGIC, decode_shell, encode_shell, read_shell, write_shell};
use chelis_shell_core::{
    CollectionObligation, SHELL_FORMAT_VERSION, ShellPackage, ShellValidationError,
    TypeVariableDomain, TypeVariableRestriction, validate_shell,
};
use chelis_syntax_host::NativeStack;
use common::{baseline_bytes, sample, unique_offset, unvalidated};
use std::fs;
use std::io::ErrorKind;

#[test]
fn host_codec_matches_native_chb8_baseline_bytes() {
    let shell = sample();
    validate_shell(&shell, &NativeStack).expect("validate native sample");
    assert_eq!(SHELL_FORMAT_VERSION, 8);
    let expected = baseline_bytes();
    assert_eq!(&expected[..8], SHELL_MAGIC);
    assert_eq!(&expected[8..12], &8_u32.to_le_bytes());
    assert_eq!(
        encode_shell(&shell).expect("encode native sample"),
        expected
    );
    assert_eq!(
        decode_shell(&expected).expect("decode native baseline"),
        shell
    );
}

#[test]
fn host_codec_rejects_every_native_sample_truncation() {
    let bytes = baseline_bytes();
    for end in 0..bytes.len() {
        let error = decode_shell(&bytes[..end]).expect_err("truncated CHB must fail");
        if end < 8 {
            assert_eq!(
                error.to_string(),
                "invalid shell envelope: unsupported predecessor shell format"
            );
        } else if end < 12 {
            assert_eq!(
                error.to_string(),
                "invalid shell envelope: truncated shell format version"
            );
        }
    }
}

#[test]
fn host_codec_rejects_trailing_data() {
    for suffix in [vec![0], b"junk".to_vec(), baseline_bytes()] {
        let mut bytes = baseline_bytes();
        bytes.extend_from_slice(&suffix);
        let error = decode_shell(&bytes).expect_err("trailing data must fail");
        assert!(matches!(*error, bincode::ErrorKind::Custom(_)));
    }
}

#[test]
fn host_codec_rejects_malformed_magic_and_predecessor_payloads() {
    for index in 0..SHELL_MAGIC.len() {
        let mut bytes = baseline_bytes();
        bytes[index] ^= 0xff;
        assert_eq!(
            decode_shell(&bytes).unwrap_err().to_string(),
            "invalid shell envelope: unsupported predecessor shell format"
        );
    }
    let predecessor = bincode::serialize(&sample()).unwrap();
    assert_eq!(
        decode_shell(&predecessor).unwrap_err().to_string(),
        "invalid shell envelope: unsupported predecessor shell format"
    );
    for payload in [vec![], vec![0], vec![0xff; 32]] {
        let mut bytes = SHELL_MAGIC.to_vec();
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&payload);
        assert!(decode_shell(&bytes).is_err());
    }
}

#[test]
fn host_codec_rejects_envelope_versions_before_payload_decode() {
    for version in [0_u32, 1, 2, 3, 4, 5, 6, 7, 9, 99, u32::MAX] {
        let mut bytes = SHELL_MAGIC.to_vec();
        bytes.extend_from_slice(&version.to_le_bytes());
        let hint = if version == 6 {
            "Deep 0.20 removes pipe nodes; "
        } else {
            ""
        };
        assert_eq!(
            decode_shell(&bytes).unwrap_err().to_string(),
            format!(
                "invalid shell envelope: shell format version {version} is unsupported; expected 8; {hint}regenerate the shell package"
            )
        );
    }
}

#[test]
fn host_codec_rejects_payload_versions_on_encode_and_decode() {
    for version in [0_u32, 1, 2, 3, 4, 5, 6, 7, 9, 99, u32::MAX] {
        let mut shell = sample();
        shell.format_version = version;
        assert!(matches!(validate_shell(&shell, &NativeStack),
            Err(ShellValidationError::UnsupportedFormatVersion { version: actual }) if actual == version));
        let hint = if version == 6 {
            "Deep 0.20 removes pipe nodes; "
        } else {
            ""
        };
        let expected = format!(
            "invalid shell envelope: payload format version {version} is unsupported; expected 8; {hint}regenerate the shell package"
        );
        assert_eq!(encode_shell(&shell).unwrap_err().to_string(), expected);
        assert_eq!(
            decode_shell(&unvalidated(&shell)).unwrap_err().to_string(),
            expected
        );
    }
}

#[test]
fn host_codec_preserves_current_relation_and_restriction_fields() {
    let mut shell = sample();
    let symbol = &mut shell.modules[0].exports[0];
    symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
    symbol.type_variable_restrictions = vec![TypeVariableRestriction {
        variable: "t0".into(),
        domain: TypeVariableDomain::ActiveSet(vec!["f32".into(), "f64".into()]),
    }];
    symbol.collection_obligations = vec![CollectionObligation::Len {
        operand: "(t-var {} t0)".into(),
        result: "(t-var {} t1)".into(),
    }];
    validate_shell(&shell, &NativeStack).expect("validate complete metadata");
    let encoded = encode_shell(&shell).expect("encode complete metadata");
    assert_ne!(encoded, baseline_bytes());
    assert_eq!(
        decode_shell(&encoded).expect("decode complete metadata"),
        shell
    );
}

fn invalid_metadata_cases() -> Vec<(&'static str, ShellPackage)> {
    let mut cases = Vec::new();
    let mut shell = sample();
    shell.package.name = " demo".into();
    cases.push(("package name", shell));
    let mut shell = sample();
    shell.compiler = "^1.2.3".into();
    cases.push(("compiler prefix", shell));
    let mut shell = sample();
    shell.compiler = "=01.2.3".into();
    cases.push(("compiler version", shell));
    let mut shell = sample();
    shell.archive_sha256 = "D".repeat(64);
    cases.push(("archive digest", shell));
    let mut shell = sample();
    shell.modules.push(shell.modules[0].clone());
    cases.push(("duplicate modules", shell));
    let mut shell = sample();
    let duplicate = shell.modules[0].exports[0].clone();
    shell.modules[0].exports.push(duplicate);
    cases.push(("duplicate exports", shell));
    let mut shell = sample();
    shell.modules[0].exports[0].type_repr = Some("(t-unit {}".into());
    cases.push(("invalid Deep", shell));
    let mut shell = sample();
    shell.modules[0].exports[0].type_variable_restrictions = vec![TypeVariableRestriction {
        variable: "t0".into(),
        domain: TypeVariableDomain::ActiveFloat,
    }];
    cases.push(("detached restriction", shell));
    let mut shell = sample();
    shell.modules[0].exports[0].collection_obligations = vec![CollectionObligation::Len {
        operand: "(t-var {} t0)".into(),
        result: "(t-prim {} i64)".into(),
    }];
    cases.push(("hidden relation variable", shell));
    let mut shell = sample();
    shell.modules[0].exports[0].effects.push(" ".into());
    cases.push(("invalid effect", shell));
    let mut shell = sample();
    shell.dependencies.push(shell.dependencies[0].clone());
    cases.push(("duplicate dependencies", shell));
    let mut shell = sample();
    shell.dependencies[0].version = " 1.0.0".into();
    cases.push(("dependency version", shell));
    cases
}

#[test]
fn host_codec_enforces_encode_decode_validation_parity() {
    for (label, shell) in invalid_metadata_cases() {
        let core_error = validate_shell(&shell, &NativeStack).expect_err(label);
        let encoded_error = encode_shell(&shell).expect_err(label);
        let decoded_error = decode_shell(&unvalidated(&shell)).expect_err(label);
        assert_eq!(
            encoded_error.to_string(),
            decoded_error.to_string(),
            "{label}"
        );
        assert_eq!(encoded_error.to_string(), core_error.to_string(), "{label}");
        assert!(
            matches!(*encoded_error, bincode::ErrorKind::Custom(_)),
            "{label}"
        );
    }
}

#[test]
fn host_codec_rejects_invalid_utf8() {
    let shell = sample();
    let bytes = baseline_bytes();
    for field in [
        &shell.package.name,
        &shell.compiler,
        &shell.modules[0].module,
        &shell.modules[0].exports[0].name,
        &shell.archive_sha256,
    ] {
        let encoded_field = bincode::serialize(field).unwrap();
        let offset = unique_offset(&bytes, &encoded_field) + 8;
        let mut invalid = bytes.clone();
        invalid[offset] = 0xff;
        let error = decode_shell(&invalid).expect_err("invalid UTF-8 must fail");
        assert!(matches!(*error, bincode::ErrorKind::InvalidUtf8Encoding(_)));
    }
}

#[test]
fn host_codec_rejects_invalid_option_enum_and_bool_tags() {
    let shell = sample();
    let bytes = baseline_bytes();
    let symbol = &shell.modules[0].exports[0];
    let symbol_bytes = bincode::serialize(symbol).unwrap();
    let start = unique_offset(&bytes, &symbol_bytes);
    let kind_offset = start + bincode::serialized_size(&symbol.name).unwrap() as usize;
    let option_offset = kind_offset + 4;
    let bool_offset = start + symbol_bytes.len() - 1;

    let mut invalid = bytes.clone();
    invalid[kind_offset..kind_offset + 4].copy_from_slice(&99_u32.to_le_bytes());
    let error = decode_shell(&invalid).unwrap_err();
    assert!(matches!(*error, bincode::ErrorKind::Custom(_)));
    assert!(error.to_string().contains("variant index"));

    let mut invalid = bytes.clone();
    invalid[option_offset] = 2;
    assert!(matches!(
        *decode_shell(&invalid).unwrap_err(),
        bincode::ErrorKind::InvalidTagEncoding(2)
    ));

    let mut invalid = bytes;
    invalid[bool_offset] = 2;
    assert!(matches!(
        *decode_shell(&invalid).unwrap_err(),
        bincode::ErrorKind::InvalidBoolEncoding(2)
    ));
}

#[test]
fn host_codec_rejects_invalid_relation_and_domain_tags() {
    let mut shell = sample();
    let symbol = &mut shell.modules[0].exports[0];
    symbol.type_repr = Some("(t-fn {} (t-var {} t0) (t-var {} t1))".into());
    symbol.type_variable_restrictions = vec![TypeVariableRestriction {
        variable: "t0".into(),
        domain: TypeVariableDomain::ActiveFloat,
    }];
    symbol.collection_obligations = vec![CollectionObligation::FloatClassify {
        kind: chelis_shell_core::FloatClassifyKind::Finite,
        operand: "(t-var {} t0)".into(),
        result: "(t-var {} t1)".into(),
    }];
    let bytes = encode_shell(&shell).expect("encode relation fixture");
    let symbol = &shell.modules[0].exports[0];
    let restriction = &symbol.type_variable_restrictions[0];
    let restriction_offset = unique_offset(&bytes, &bincode::serialize(restriction).unwrap());
    let domain_offset =
        restriction_offset + bincode::serialized_size(&restriction.variable).unwrap() as usize;
    let relation_offset = unique_offset(
        &bytes,
        &bincode::serialize(&symbol.collection_obligations[0]).unwrap(),
    );
    for offset in [domain_offset, relation_offset, relation_offset + 4] {
        let mut invalid = bytes.clone();
        invalid[offset..offset + 4].copy_from_slice(&99_u32.to_le_bytes());
        let error = decode_shell(&invalid).expect_err("invalid enum tag must fail");
        assert!(matches!(*error, bincode::ErrorKind::Custom(_)));
        assert!(error.to_string().contains("variant index"));
    }
}

#[test]
fn host_codec_rejects_invalid_string_and_collection_lengths() {
    let shell = sample();
    let bytes = baseline_bytes();
    let symbol = &shell.modules[0].exports[0];
    let fields = [
        bincode::serialize(&shell.package.name).unwrap(),
        bincode::serialize(&shell.modules).unwrap(),
        bincode::serialize(&shell.modules[0].exports).unwrap(),
        bincode::serialize(&symbol.effects).unwrap(),
        bincode::serialize(&shell.dependencies).unwrap(),
    ];
    for field in fields {
        let offset = unique_offset(&bytes, &field);
        for length in [0_u64, 1024, u64::MAX] {
            let mut invalid = bytes.clone();
            invalid[offset..offset + 8].copy_from_slice(&length.to_le_bytes());
            assert!(
                decode_shell(&invalid).is_err(),
                "length {length} at byte {offset}"
            );
        }
    }
}

#[test]
fn host_filesystem_round_trip_retains_native_baseline_bytes() {
    let directory = tempfile::tempdir().expect("create test directory");
    let path = directory.path().join("sample.chb");
    write_shell(&path, &sample()).expect("write shell");
    assert_eq!(fs::read(&path).expect("read raw CHB"), baseline_bytes());
    assert_eq!(read_shell(&path).expect("read shell"), sample());
}

#[test]
fn host_filesystem_validation_failure_preserves_existing_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("existing.chb");
    let original = baseline_bytes();
    fs::write(&path, &original).unwrap();
    for (label, shell) in invalid_metadata_cases() {
        let error = write_shell(&path, &shell).expect_err(label);
        let codec_error = error.downcast_ref::<bincode::Error>().expect("codec error");
        assert_eq!(
            codec_error.to_string(),
            encode_shell(&shell).unwrap_err().to_string(),
            "{label}"
        );
        assert_eq!(fs::read(&path).unwrap(), original, "{label}");
        assert_eq!(read_shell(&path).unwrap(), sample(), "{label}");
    }
}

#[test]
fn host_filesystem_invalid_metadata_does_not_create_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("absent.chb");
    let mut shell = sample();
    shell.modules[0].exports[0].type_repr = None;
    assert!(write_shell(&path, &shell).is_err());
    assert!(!path.exists());
}

#[test]
fn host_filesystem_reports_read_and_write_io_errors() {
    let directory = tempfile::tempdir().unwrap();
    let missing = directory.path().join("missing").join("sample.chb");
    let error = read_shell(&missing).expect_err("missing file must fail");
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        ErrorKind::NotFound
    );
    let error = write_shell(&missing, &sample()).expect_err("missing parent must fail");
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        ErrorKind::NotFound
    );
    assert!(!missing.exists());
    let error = read_shell(directory.path()).expect_err("directory read must fail");
    assert!(error.downcast_ref::<std::io::Error>().is_some());
    let error = write_shell(directory.path(), &sample()).expect_err("directory write must fail");
    assert!(error.downcast_ref::<std::io::Error>().is_some());
    assert!(directory.path().is_dir());
}

#[test]
fn host_filesystem_io_failure_preserves_existing_parent_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("existing.chb");
    let original = baseline_bytes();
    fs::write(&path, &original).unwrap();
    let error = write_shell(&path.join("child.chb"), &sample())
        .expect_err("file cannot be a parent directory");
    assert!(error.downcast_ref::<std::io::Error>().is_some());
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_eq!(read_shell(&path).unwrap(), sample());
}

#[test]
fn host_filesystem_retains_decode_errors_for_corrupt_files() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("corrupt.chb");
    let mut trailing = baseline_bytes();
    trailing.push(0);
    let mut invalid = sample();
    invalid.modules[0].exports[0].type_repr = None;
    for bytes in [
        vec![],
        b"not a CHB".to_vec(),
        baseline_bytes()[..11].to_vec(),
        trailing,
        unvalidated(&invalid),
    ] {
        fs::write(&path, &bytes).unwrap();
        let expected = decode_shell(&bytes).unwrap_err().to_string();
        let error = read_shell(&path).expect_err("corrupt file must fail");
        assert!(error.downcast_ref::<bincode::Error>().is_some());
        assert_eq!(error.to_string(), expected);
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
}
