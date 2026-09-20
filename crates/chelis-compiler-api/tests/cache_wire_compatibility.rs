//! spec/10 exact numeric codec migration: old artifacts cannot reach new
//! positional payload decoders; current artifacts must still hit.
#[allow(dead_code)]
#[path = "../src/cache_envelope.rs"]
mod cache_envelope;

use chelis_compiler_api::{
    CacheError, CompiledContext, LibraryContext, StdLibContext, build_library_context,
    build_stdlib_context, library_cache_key, library_cache_key_input_bytes, stdlib_cache_key,
    stdlib_cache_key_input_bytes,
};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn fixtures() -> &'static Path {
    Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/cache_wire_v3"
    ))
}

fn manifest() -> serde_json::Value {
    serde_json::from_str(include_str!("fixtures/cache_wire_v3/producer.json")).unwrap()
}

fn migrated_v018(source: &str) -> String {
    chelis_surf::format::migrate_source_v018(source)
        .expect("historical source must migrate to the current canonical spelling")
}

fn migrate_historical_package_for_current_compiler(package: &Path) {
    let source_path = package.join("mylib/src/math.ch");
    let source = fs::read_to_string(&source_path).expect("read historical package source");
    fs::write(source_path, migrated_v018(&source)).expect("write migrated package source");
}

#[test]
fn historical_artifacts_match_the_recorded_actual_producer() {
    let evidence = manifest();
    assert_eq!(
        evidence["producer_commit"],
        "010354f91d1efd210037f4090f6e862aa7a68773"
    );
    assert_eq!(evidence["stdlib_version"], 15);
    assert_eq!(evidence["library_version"], 11);
    assert_eq!(evidence["old_literal_result_declaration"], "tensor[4, f32]");
    for (file, digest) in evidence["sha256"].as_object().unwrap() {
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(fs::read(fixtures().join(file)).unwrap())
            ),
            digest.as_str().unwrap()
        );
    }
}

#[test]
fn previous_compiled_context_is_rejected_before_live_source_lookup() {
    let absent = tempfile::tempdir().unwrap();
    let error = CompiledContext::load_if_fresh(
        &fixtures().join("context-v17.ctx"),
        absent.path(),
        &absent.path().join("no-package"),
    )
    .expect_err("old positional payload must never be admitted");
    assert!(
        matches!(error, CacheError::Corrupt(ref message) if message.contains("magic")),
        "{error}"
    );
}

#[test]
fn previous_worker_handoff_is_rejected_before_positional_payload_decode() {
    let evidence = manifest();
    assert_eq!(evidence["old_worker_producer_old_worker_consumer"], "pass");
    let bytes = fs::read(fixtures().join("worker-unversioned.bin")).expect("old worker bytes");
    let error = CompiledContext::decode(&bytes).expect_err("old worker format must reject");
    assert!(error.contains("magic"), "{error}");
}

#[test]
fn version_changes_alone_reject_old_subcontexts_before_payload_decode() {
    let evidence = manifest();
    let std_source = migrated_v018(evidence["stdlib_source"].as_str().unwrap());
    let dep_source = migrated_v018(evidence["dependency_source"].as_str().unwrap());
    let std_decls = chelis_surf::parser::parse_str(&std_source).unwrap();
    let dep_decls = chelis_surf::parser::parse_str(&dep_source).unwrap();
    let std_key = stdlib_cache_key(&std_decls, [0x5a; 32]);
    let lib_key = library_cache_key(&dep_decls, std_key);
    for (input_file, cache_file, prefix, expected_version, current_input, current_key) in [
        (
            "stdlib-v15-key-input.bin",
            "stdlib-v15.tc",
            b"chelis_std_typecheck_v".as_slice(),
            24_u32,
            stdlib_cache_key_input_bytes(&std_decls, [0x5a; 32]),
            std_key,
        ),
        (
            "library-v11-key-input.bin",
            "library-v11.tc",
            b"chelis_library_typecheck_v".as_slice(),
            18_u32,
            library_cache_key_input_bytes(&dep_decls, std_key),
            lib_key,
        ),
    ] {
        let mut old_inputs = fs::read(fixtures().join(input_file)).unwrap();
        assert!(old_inputs.starts_with(prefix));
        assert!(current_input.starts_with(prefix));
        let version = &current_input[prefix.len()..prefix.len() + 4];
        assert_eq!(
            version,
            expected_version.to_le_bytes(),
            "the actual writer must select the new format"
        );
        assert_ne!(&old_inputs[prefix.len()..prefix.len() + 4], version);
        old_inputs[prefix.len()..prefix.len() + 4].copy_from_slice(version);
        // Keep the old build, bundle and declaration bytes unchanged. Only
        // the current format selector distinguishes this key from the old one.
        let isolated_key: [u8; 32] = Sha256::digest(old_inputs).into();
        for key in [isolated_key, current_key] {
            let path = fixtures().join(cache_file);
            if cache_file.starts_with("stdlib-") {
                assert!(
                    cache_envelope::load::<StdLibContext>(&path, key)
                        .unwrap()
                        .is_none()
                );
            } else {
                assert!(
                    cache_envelope::load::<LibraryContext>(&path, key)
                        .unwrap()
                        .is_none()
                );
            }
        }
    }
}

#[test]
fn current_numeric_subcontexts_roundtrip_through_actual_cache_codecs() {
    let evidence = manifest();
    let std_source = migrated_v018(evidence["stdlib_source"].as_str().unwrap());
    let dep_source = migrated_v018(evidence["dependency_source"].as_str().unwrap());
    let std_decls = chelis_surf::parser::parse_str(&std_source).unwrap();
    let dep_decls = chelis_surf::parser::parse_str(&dep_source).unwrap();
    let std_context = build_stdlib_context(&std_decls).unwrap();
    let lib_context = build_library_context(&std_context, &dep_decls)
        .unwrap()
        .unwrap();
    let std_key = stdlib_cache_key(&std_decls, [0x5a; 32]);
    let lib_key = library_cache_key(&dep_decls, std_key);
    historical_producer::assert_literal_result(
        &serde_json::to_value(std_context.library_dag().unwrap().raw()).unwrap(),
    );
    let dir = tempfile::tempdir().unwrap();
    let std_path = dir.path().join("std.tc");
    let lib_path = dir.path().join("lib.tc");
    cache_envelope::save(&std_path, std_key, &std_context).unwrap();
    cache_envelope::save(&lib_path, lib_key, &lib_context).unwrap();
    let std_hit = cache_envelope::load::<StdLibContext>(&std_path, std_key)
        .unwrap()
        .unwrap();
    let lib_hit = cache_envelope::load::<LibraryContext>(&lib_path, lib_key)
        .unwrap()
        .unwrap();
    historical_producer::assert_literal_result(
        &serde_json::to_value(std_hit.library_dag().unwrap().raw()).unwrap(),
    );
    assert_eq!(
        bincode::serialize(&std_hit).unwrap(),
        bincode::serialize(&std_context).unwrap()
    );
    assert_eq!(
        bincode::serialize(&lib_hit).unwrap(),
        bincode::serialize(&lib_context).unwrap()
    );
}

#[allow(dead_code)]
#[path = "fixtures/cache_wire_v3/producer.rs"]
mod historical_producer;

// These expectations are constructed independently from the actual producer's
// observations. Neither cache roundtrip supplies its own expected numeric bits.
fn expected_numeric_payloads(current: bool) -> Vec<serde_json::Value> {
    use serde_json::json;
    let mut rows = vec![
        json!({"kind":"scalar","dtype":"int64","value":9007199254740993_i64}),
        json!({"kind":"scalar","dtype":"f64","bits":"0000000000000000"}),
        json!({"kind":"storage","dtype":"f64","bits":["8000000000000000","3ff0000000000001"]}),
        json!({"kind":"storage","dtype":"int64","values":[9007199254740993_i64,-9007199254740993_i64]}),
    ];
    for row in &mut rows {
        let is_integer = row["dtype"] == "int64";
        let storage = row["kind"] == "storage";
        let mut bytes = (if is_integer { 4_u32 } else { 0_u32 })
            .to_le_bytes()
            .to_vec();
        if storage {
            bytes.extend(2_u64.to_le_bytes());
        }
        if is_integer {
            bytes.extend(9007199254740993_i64.to_le_bytes());
            if storage {
                bytes.extend((-9007199254740993_i64).to_le_bytes());
            }
        } else {
            for bits in if storage {
                vec![0x8000000000000000_u64, 0x3ff0000000000001]
            } else {
                vec![0_u64]
            } {
                if current {
                    bytes.extend(16_u64.to_le_bytes());
                    bytes.extend(format!("{bits:016x}").as_bytes());
                } else {
                    bytes.extend(bits.to_le_bytes());
                }
            }
        }
        row["bincode"] = json!(
            bytes
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
    }
    rows.sort_by_key(|row| row.to_string());
    rows
}

fn unhex(hex: &str) -> Vec<u8> {
    hex.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn historical_caches_contain_actual_old_scalar_and_storage_positional_bytes() {
    let evidence = manifest();
    let expected = expected_numeric_payloads(false);
    assert_eq!(
        evidence["stdlib_numeric_payloads"],
        serde_json::json!(expected)
    );
    assert_eq!(
        evidence["context_numeric_payloads"],
        serde_json::json!(expected)
    );
    for file in ["stdlib-v15.tc", "context-v17.ctx", "worker-unversioned.bin"] {
        let bytes = fs::read(fixtures().join(file)).unwrap();
        for row in &expected {
            let payload = unhex(row["bincode"].as_str().unwrap());
            assert!(
                bytes.windows(payload.len()).any(|window| window == payload),
                "{file}: {row}"
            );
            if row["dtype"] == "f64" {
                if row["kind"] == "scalar" {
                    assert!(bincode::deserialize::<chelis_types::ScalarValue>(&payload).is_err());
                } else {
                    assert!(bincode::deserialize::<chelis_types::TensorStorage>(&payload).is_err());
                }
            }
        }
    }
}

#[test]
fn current_stdlib_cache_preserves_scalar_storage_bits_and_checked_reconstruction() {
    let source = migrated_v018(historical_producer::NUMERIC_SOURCE);
    let decls = chelis_surf::parser::parse_str(&source).unwrap();
    let context = build_stdlib_context(&decls).unwrap();
    let expected = expected_numeric_payloads(true);
    assert_eq!(
        historical_producer::numeric_payloads(context.library_dag().unwrap().raw()),
        expected
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("numeric.tc");
    let key = stdlib_cache_key(&decls, [0x5a; 32]);
    cache_envelope::save(&path, key, &context).unwrap();
    let restored = cache_envelope::load::<StdLibContext>(&path, key)
        .unwrap()
        .unwrap();
    assert_eq!(
        historical_producer::numeric_payloads(restored.library_dag().unwrap().raw()),
        expected
    );
    assert_eq!(
        restored.checked_library().program().exprs(),
        context.checked_library().program().exprs()
    );
    assert!(
        restored
            .type_env()
            .matches_checked_program(restored.library_checked())
    );
    assert_payload_bytes(&fs::read(path).unwrap(), &expected);
}

fn assert_payload_bytes(bytes: &[u8], expected: &[serde_json::Value]) {
    for row in expected {
        let payload = unhex(row["bincode"].as_str().unwrap());
        assert!(
            bytes.windows(payload.len()).any(|window| window == payload),
            "missing {row}"
        );
    }
}

#[test]
fn current_compiled_disk_and_worker_preserve_scalar_storage_bits_and_reconstruct() {
    let directory = tempfile::tempdir().unwrap();
    let package = directory.path().join("package");
    historical_producer::package_fixture(&package);
    migrate_historical_package_for_current_compiler(&package);
    let reef_home = directory.path().join("reef-home");
    let context = chelis_compiler_api::compile_reef_context(&reef_home, &package).unwrap();
    let expected = expected_numeric_payloads(true);
    assert_eq!(
        historical_producer::context_numeric_payloads(&context),
        expected
    );
    let path = directory.path().join("numeric.ctx");
    context.save(&path).unwrap();
    let bytes = context.encode().unwrap();
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_payload_bytes(&bytes, &expected);
    let disk = CompiledContext::load_if_fresh(&path, &reef_home, &package)
        .unwrap()
        .unwrap();
    let worker = CompiledContext::decode(&bytes).unwrap();
    historical_producer::assert_literal_result(
        &serde_json::to_value(&context).unwrap()["library_dag"],
    );
    for restored in [&disk, &worker] {
        historical_producer::assert_literal_result(
            &serde_json::to_value(restored).unwrap()["library_dag"],
        );
        assert_eq!(
            historical_producer::context_numeric_payloads(restored),
            expected
        );
        assert_eq!(
            bincode::serialize(restored).unwrap(),
            bincode::serialize(&context).unwrap()
        );
    }
}

fn swap_first_two_selector_formals(value: &mut serde_json::Value) {
    let selector_context = &mut value["type_env"]["inner"]["selector_callables"];
    let swap_in_origins = |origins: &mut serde_json::Map<String, serde_json::Value>| {
        origins
            .values_mut()
            .find_map(|origin| {
                let params = origin.get_mut("Known")?.get_mut("params")?.as_array_mut()?;
                (params.len() >= 2).then_some(params)
            })
            .map(|params| params.swap(0, 1))
            .is_some()
    };
    if swap_in_origins(
        selector_context["root"]
            .as_object_mut()
            .expect("root selector origins"),
    ) {
        return;
    }
    for origins in selector_context["modules"]
        .as_object_mut()
        .expect("module selector origins")
        .values_mut()
    {
        if swap_in_origins(origins.as_object_mut().expect("module origin map")) {
            return;
        }
    }
    panic!("fixture must contain a callable with two formals");
}

fn assert_selector_forgery_rejects<T>(context: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let mut encoded =
        serde_json::to_value(context).expect("nonempty cache context must be JSON-safe");
    serde_json::from_value::<T>(encoded.clone()).expect("unchanged context must round-trip");
    swap_first_two_selector_formals(&mut encoded);
    let error = match serde_json::from_value::<T>(encoded) {
        Ok(_) => panic!("forged selector metadata must fail cache reconstruction"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("the type environment does not match the checked library"),
        "unexpected selector-forgery rejection: {error}"
    );
}

#[test]
fn every_cached_library_decoder_rejects_forged_selector_callable_metadata() {
    let pair_source = "def pair(x: f32, w: f32) -> f32 = x * w\n";
    let pair_decls = chelis_surf::parser::parse_str(pair_source).unwrap();
    let std_context = build_stdlib_context(&pair_decls).unwrap();
    assert_selector_forgery_rejects(&std_context);

    let empty_stdlib = build_stdlib_context(&[]).unwrap();
    let library_context = build_library_context(&empty_stdlib, &pair_decls)
        .unwrap()
        .unwrap();
    assert_selector_forgery_rejects(&library_context);

    let directory = tempfile::tempdir().unwrap();
    let package = directory.path().join("package");
    historical_producer::package_fixture(&package);
    migrate_historical_package_for_current_compiler(&package);
    let reef_home = directory.path().join("reef-home");
    let compiled = chelis_compiler_api::compile_reef_context(&reef_home, &package).unwrap();
    assert_selector_forgery_rejects(&compiled);
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SharedFixtureEnvelope {
    version: u32,
    key: [u8; 32],
    payload_sha256: [u8; 32],
    payload: Vec<u8>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct ContextFixtureEnvelope {
    version: u32,
    source_hash: chelis_compiler_api::ContextHash,
    identity: chelis_compiler_api::CacheIdentity,
    payload_sha256: [u8; 32],
    payload: Vec<u8>,
}

fn change_numeric_bits(payload: &mut [u8], storage: bool) {
    let expected = expected_numeric_payloads(true);
    let row = expected
        .iter()
        .find(|row| row["dtype"] == "f64" && (row["kind"] == "storage") == storage)
        .unwrap();
    let old = unhex(row["bincode"].as_str().unwrap());
    let start = payload
        .windows(old.len())
        .position(|window| window == old)
        .expect("numeric payload exists");
    // Change only the sign bit of zero, retaining dtype, width, node shape and
    // checked-library identity. The fresh lowering still contains negative zero.
    payload[start + if storage { 20 } else { 12 }] = if storage { b'0' } else { b'8' };
}

#[test]
fn cache_reconstruction_rejects_changed_numeric_bits_after_checksum_recomputed() {
    let directory = tempfile::tempdir().unwrap();
    let source = migrated_v018(&format!(
        "{}{}",
        historical_producer::NUMERIC_SOURCE,
        historical_producer::WITNESS_SOURCE
    ));
    let decls = chelis_surf::parser::parse_str(&source).unwrap();
    let context = build_stdlib_context(&decls).unwrap();
    let key = stdlib_cache_key(&decls, [0x5a; 32]);
    let std_path = directory.path().join("numeric.tc");
    cache_envelope::save(&std_path, key, &context).unwrap();
    let std_bytes = fs::read(&std_path).unwrap();
    let package = directory.path().join("package");
    historical_producer::package_fixture(&package);
    migrate_historical_package_for_current_compiler(&package);
    let reef_home = directory.path().join("reef-home");
    let compiled = chelis_compiler_api::compile_reef_context(&reef_home, &package).unwrap();
    let (compiled_bytes, compiled_digest) = compiled.encode_for_handoff().unwrap();
    let compiled_path = directory.path().join("numeric.ctx");
    for storage in [false, true] {
        let std_magic = b"CHELIS_CACHE_ENV_V1\n";
        let mut envelope: SharedFixtureEnvelope =
            bincode::deserialize(std_bytes.strip_prefix(std_magic).unwrap()).unwrap();
        change_numeric_bits(&mut envelope.payload, storage);
        envelope.payload_sha256 = Sha256::digest(&envelope.payload).into();
        let mut changed = std_magic.to_vec();
        changed.extend(bincode::serialize(&envelope).unwrap());
        fs::write(&std_path, changed).unwrap();
        let error = cache_envelope::load::<StdLibContext>(&std_path, key).unwrap_err();
        assert!(
            matches!(error,cache_envelope::CacheError::Decode(ref message) if message.contains("lowered library payload")),
            "{error}"
        );
        let magic_len = compiled_bytes
            .iter()
            .position(|byte| *byte == b'\n')
            .unwrap()
            + 1;
        let mut envelope: ContextFixtureEnvelope =
            bincode::deserialize(&compiled_bytes[magic_len..]).unwrap();
        change_numeric_bits(&mut envelope.payload, storage);
        envelope.payload_sha256 = Sha256::digest(&envelope.payload).into();
        let mut changed = compiled_bytes[..magic_len].to_vec();
        changed.extend(bincode::serialize(&envelope).unwrap());
        fs::write(&compiled_path, &changed).unwrap();
        let error = CompiledContext::decode(&changed).unwrap_err();
        assert!(error.contains("lowered library payload"), "{error}");
        let error =
            CompiledContext::load_if_fresh(&compiled_path, &reef_home, &package).unwrap_err();
        assert!(
            matches!(error,CacheError::Decode(ref message) if message.contains("lowered library payload")),
            "{error}"
        );
        // chelis#2211: the same rewrite, against the authenticated route. That
        // route does not re-derive, so it has to catch this some other way, and
        // it does: the producer's digest names the original payload and never
        // travelled with these bytes, so resealing the envelope buys nothing.
        let error = CompiledContext::decode_authenticated(&changed, &compiled_digest).unwrap_err();
        assert!(error.contains("digest its producer delivered"), "{error}");
        // And the untouched bytes still authenticate, so the assertion above is
        // detecting the rewrite rather than rejecting everything.
        CompiledContext::decode_authenticated(&compiled_bytes, &compiled_digest)
            .expect("the producer's own bytes must authenticate");
    }
}
