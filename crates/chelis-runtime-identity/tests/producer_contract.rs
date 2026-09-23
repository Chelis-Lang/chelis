#[path = "support/producer.rs"]
mod producer;

use chelis_runtime_identity::{BuildProvenance, Descriptor, RecordError, compare, decode_archive};
use object::{Object, ObjectSection};
use producer::{Fixture, Product, copy_tree, dimensions};
use std::fs;

fn agrees(runtime: &Product, consumer: &Product) {
    compare(&runtime.descriptor, &consumer.descriptor).unwrap();
    assert_eq!(runtime.descriptor, consumer.descriptor);
}
fn differs(before: &Descriptor, after: &Descriptor, dimension: &str) {
    assert!(
        compare(before, after).is_err(),
        "changed {dimension} did not invalidate compatibility"
    );
    assert_ne!(
        dimensions(before)[dimension],
        dimensions(after)[dimension],
        "missing {dimension} discrimination"
    );
}
fn fails_for(output: std::process::Output, evidence: &str) {
    assert!(
        !output.status.success(),
        "required failure instead emitted a successful producer"
    );
    let diagnostic = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        diagnostic.to_lowercase().contains(&evidence.to_lowercase()),
        "failure was unrelated to {evidence}: {diagnostic}"
    );
}

fn retained_record_controls(runtime: &Product) {
    let archive = object::read::archive::ArchiveFile::parse(runtime.bytes.as_slice()).unwrap();
    let mut record = None;
    let mut object_bytes = None;
    let mut section_name_offset = None;
    for member in archive.members() {
        let member = member.unwrap();
        let data = member.data(runtime.bytes.as_slice()).unwrap();
        let Ok(file) = object::File::parse(data) else {
            continue;
        };
        for section in file.sections() {
            if matches!(section.name().unwrap(), ".chelis.runtime.id" | "__ch_rt_id") {
                assert!(record.is_none());
                let (offset, size) = section.file_range().unwrap();
                let member_offset = data.as_ptr() as usize - runtime.bytes.as_ptr() as usize;
                record = Some((member_offset + offset as usize, size as usize));
                section_name_offset = Some(
                    section.name_bytes().unwrap().as_ptr() as usize
                        - runtime.bytes.as_ptr() as usize,
                );
                object_bytes = Some(data.to_vec());
            }
        }
    }
    let (offset, length) = record.expect("real runtime archive retained its native section");
    assert_eq!(length, chelis_runtime_identity::RECORD_LENGTH);
    // Corrupt actual retained bytes, not a fabricated empty/stale runtime.
    for (byte, value) in [(0, b'X'), (16, 255), (18, 255), (19, 1), (20, 0)] {
        let mut corrupt = runtime.bytes.clone();
        corrupt[offset + byte] = value;
        let error = decode_archive(&corrupt).unwrap_err();
        if byte == 16 || byte == 18 {
            assert!(matches!(error, RecordError::Unsupported { .. }));
        } else {
            assert!(matches!(error, RecordError::Malformed { .. }));
        }
    }
    assert!(matches!(
        decode_archive(&runtime.bytes[..offset + length - 1]),
        Err(RecordError::Truncated { .. })
    ));
    let mut missing = runtime.bytes.clone();
    // Strip recognition through the parsed section-name location, leaving record
    // bytes in the image: marker scanning must not recover a removed record.
    missing[section_name_offset.unwrap()] = b'x';
    assert_eq!(decode_archive(&missing), Err(RecordError::Missing));
    let object = object_bytes.unwrap();
    let mut duplicate = runtime.bytes.clone();
    if duplicate.len() % 2 != 0 {
        duplicate.push(b'\n');
    }
    duplicate.extend_from_slice(
        format!(
            "{:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
            "duplicate.o/",
            0,
            0,
            0,
            "100644",
            object.len()
        )
        .as_bytes(),
    );
    duplicate.extend_from_slice(&object);
    if object.len() % 2 != 0 {
        duplicate.push(b'\n');
    }
    assert_eq!(decode_archive(&duplicate), Err(RecordError::Duplicate));
    assert_eq!(decode_archive(&runtime.bytes).unwrap(), runtime.descriptor);
}

// This is deliberately one serial scenario: every mutation reuses only this test's
// target/observation directories, and each assertion has an independently built
// positive control. Invoking the ignored target makes missing native prerequisites
// failures, never skips or fallback identities.
#[test]
#[ignore = "real runtime/CLI/Python builds; run explicitly on native Linux and macOS"]
fn real_producers_closure_independence_provenance_and_cached_units() {
    {
        // Build real consumers where no candidate static archive is produced.
        // The runtime remains a real Rust dependency with observed inputs, not
        // a mocked recipe or a relabeled archive from another target directory.
        let archive_free = Fixture::new();
        archive_free.edit_manifest("crates/chelis-runtime/Cargo.toml", |manifest| {
            manifest["lib"]["crate-type"] = toml::Value::Array(vec!["rlib".into()]);
        });
        let cli = archive_free.build("cli-without-runtime-archive", "chelis-cli", &[], &[]);
        let python =
            archive_free.build("python-without-runtime-archive", "chelis-python", &[], &[]);
        assert!(
            cli.runtime_archives.is_empty(),
            "CLI witness produced a runtime static archive"
        );
        assert!(
            python.runtime_archives.is_empty(),
            "Python witness produced a runtime static archive"
        );
        archive_free.assert_no_runtime_static_archive();
        agrees(&cli, &python);
    }
    let mut fixture = Fixture::new();
    let original = fixture.build("runtime-a", "chelis-runtime", &[], &[]);
    let cli = fixture.build("cli-a", "chelis-cli", &[], &[]);
    let python = fixture.build("python-a", "chelis-python", &[], &[]);
    agrees(&original, &cli);
    agrees(&original, &python);
    assert!(matches!(
        original.provenance,
        BuildProvenance::SourceWorktree { .. }
    ));
    retained_record_controls(&original);
    let required_header = fixture
        .source
        .join("crates/chelis-runtime/include/chelis_runtime.h");
    let retained_header = fs::read(&required_header).unwrap();
    fs::remove_file(&required_header).unwrap();
    fails_for(
        fixture.invoke("runtime-a", "chelis-runtime", &[], &[], true),
        "include/chelis_runtime.h",
    );
    fs::write(&required_header, retained_header).unwrap();
    agrees(
        &original,
        &fixture.build("runtime-a", "chelis-runtime", &[], &[]),
    );
    fails_for(
        fixture.invoke("unmanaged-producer", "chelis-runtime", &[], &[], false),
        "identity",
    );
    fails_for(
        fixture.invoke(
            "ambiguous-provenance",
            "chelis-cli",
            &[],
            &[("CHELIS_IDENTITY_PROVENANCE", "")],
            true,
        ),
        "provenance",
    );

    let saved = fixture.root.join("retained-A-renamed.a");
    fs::write(&saved, &original.bytes).unwrap();
    fs::write(
        saved.with_extension("identity.json"),
        b"{\"mode\":\"sealed-distribution\"}",
    )
    .unwrap();
    assert_eq!(
        decode_archive(&fs::read(&saved).unwrap()).unwrap(),
        original.descriptor
    );

    let sealed = fixture.build(
        "runtime-sealed",
        "chelis-runtime",
        &[],
        &[("CHELIS_IDENTITY_PROVENANCE", "sealed-distribution")],
    );
    agrees(&original, &sealed);
    assert!(matches!(
        sealed.provenance,
        BuildProvenance::SealedDistribution { .. }
    ));
    let sealed_cli = fixture.build(
        "cli-sealed",
        "chelis-cli",
        &[],
        &[("CHELIS_IDENTITY_PROVENANCE", "sealed-distribution")],
    );
    let sealed_python = fixture.build(
        "python-sealed",
        "chelis-python",
        &[],
        &[("CHELIS_IDENTITY_PROVENANCE", "sealed-distribution")],
    );
    agrees(&sealed, &sealed_cli);
    agrees(&sealed, &sealed_python);
    assert!(matches!(
        sealed_cli.provenance,
        BuildProvenance::SealedDistribution { .. }
    ));
    assert!(matches!(
        sealed_python.provenance,
        BuildProvenance::SealedDistribution { .. }
    ));

    let relocated = fixture.root.join("relocated-source");
    copy_tree(&fixture.source, &relocated);
    let original_root = std::mem::replace(&mut fixture.source, relocated);
    let moved = fixture.build("runtime-relocated", "chelis-runtime", &[], &[]);
    agrees(&original, &moved);
    agrees(
        &moved,
        &fixture.build("cli-relocated", "chelis-cli", &[], &[]),
    );
    agrees(
        &moved,
        &fixture.build("python-relocated", "chelis-python", &[], &[]),
    );
    match (&original.provenance, &moved.provenance) {
        (
            BuildProvenance::SourceWorktree {
                source_root: left, ..
            },
            BuildProvenance::SourceWorktree {
                source_root: right, ..
            },
        ) => assert_ne!(left, right),
        _ => panic!("relocation must retain explicit source provenance"),
    }
    fixture.source = original_root;

    fixture.append(
        "crates/chelis-runtime/src/lib.rs",
        "\nconst IDENTITY_DIRTY_BYTES: u32 = 2394;\n",
    );
    let dirty = fixture.build("runtime-a", "chelis-runtime", &[], &[]);
    differs(&original.descriptor, &dirty.descriptor, "source");
    let dirty_cli = fixture.build("cli-a", "chelis-cli", &[], &[]);
    let dirty_python = fixture.build("python-a", "chelis-python", &[], &[]);
    agrees(&dirty, &dirty_cli);
    agrees(&dirty, &dirty_python);
    assert!(compare(&original.descriptor, &dirty_cli.descriptor).is_err());
    assert!(compare(&original.descriptor, &dirty_python.descriptor).is_err());
    assert_eq!(
        decode_archive(&fs::read(&saved).unwrap()).unwrap(),
        original.descriptor,
        "current source/sidecar relabeled old archive"
    );

    let added = fixture
        .source
        .join("crates/chelis-runtime/src/identity_untracked.rs");
    fs::write(&added, "pub const UNTRACKED: u32 = 19;\n").unwrap();
    let membership = fixture.build("runtime-a", "chelis-runtime", &[], &[]);
    differs(&dirty.descriptor, &membership.descriptor, "source");
    fs::remove_file(added).unwrap();
    let removed = fixture.build("runtime-a", "chelis-runtime", &[], &[]);
    agrees(&dirty, &removed);

    let header = fixture
        .source
        .join("crates/chelis-runtime/include/identity_fixture.h");
    fs::write(&header, "#define CHELIS_IDENTITY_FIXTURE 1\n").unwrap();
    let header_added = fixture.build("runtime-a", "chelis-runtime", &[], &[]);
    differs(&removed.descriptor, &header_added.descriptor, "interface");
    fs::write(&header, "#define CHELIS_IDENTITY_FIXTURE 2\n").unwrap();
    let header_changed = fixture.build("runtime-a", "chelis-runtime", &[], &[]);
    differs(
        &header_added.descriptor,
        &header_changed.descriptor,
        "interface",
    );
    fs::remove_file(header).unwrap();
    agrees(
        &dirty,
        &fixture.build("runtime-a", "chelis-runtime", &[], &[]),
    );

    let ledger = fixture.build(
        "runtime-ledger",
        "chelis-runtime",
        &["--features", "chelis-runtime/ownership-ledger"],
        &[],
    );
    differs(&dirty.descriptor, &ledger.descriptor, "features");
    let configuration = fixture.build(
        "runtime-config",
        "chelis-runtime",
        &[],
        &[("RUSTFLAGS", "-C debug-assertions=no")],
    );
    differs(&dirty.descriptor, &configuration.descriptor, "compile");
    let target_cpu = fixture.build(
        "runtime-cpu",
        "chelis-runtime",
        &[],
        &[("RUSTFLAGS", "-C target-cpu=native")],
    );
    differs(&dirty.descriptor, &target_cpu.descriptor, "target");

    // A source root cannot silently become a sealed build or borrow a neighbor.
    let absent = fixture.root.join("missing-source");
    fails_for(
        fixture.invoke(
            "missing-root",
            "chelis-cli",
            &[],
            &[("CHELIS_IDENTITY_WORKSPACE", absent.to_str().unwrap())],
            true,
        ),
        "identity",
    );

    fixture.install_input_fixture();
    let plain = fixture.build("fixture-runtime", "chelis-runtime", &[], &[]);
    let unified = fixture.build_unified("fixture-cli", "chelis-cli", &[], &[]);
    differs(&plain.descriptor, &unified.descriptor, "features");
    let boosted = fixture.build_unified("fixture-boosted", "chelis-runtime", &[], &[]);
    agrees(&boosted, &unified);
    let consumer_feature = fixture.build_unified(
        "fixture-cli",
        "chelis-cli",
        &["--features", "chelis-cli/identity-consumer-only"],
        &[],
    );
    agrees(&boosted, &consumer_feature);
    // Unrelated CLI input/configuration is not part of the runtime closure.
    fixture.append(
        "crates/chelis-cli/src/main.rs",
        "\nconst IDENTITY_UNRELATED: u32 = 11;\n",
    );
    agrees(
        &boosted,
        &fixture.build_unified("fixture-cli", "chelis-cli", &[], &[]),
    );
    let cli_configuration = fixture.build_unified(
        "fixture-cli",
        "chelis-cli",
        &["--config", "profile.dev.package.chelis-cli.opt-level=2"],
        &[],
    );
    assert_eq!(cli_configuration.profile["opt_level"], "2");
    assert_ne!(
        cli_configuration.profile["opt_level"], unified.profile["opt_level"],
        "the positive isolation witness must actually reconfigure the CLI unit"
    );
    agrees(&boosted, &cli_configuration);
    let build_configuration = fixture.build_unified(
        "fixture-build-config",
        "chelis-runtime",
        &["--config", "profile.dev.build-override.opt-level=1"],
        &[],
    );
    differs(
        &boosted.descriptor,
        &build_configuration.descriptor,
        "compile",
    );
    if let BuildProvenance::SourceWorktree { recipe, .. } = &build_configuration.provenance {
        assert_eq!(
            recipe.cargo_profile,
            chelis_runtime_identity::ProfileClass::Debug
        );
        assert!(
            recipe.units.iter().any(|unit| unit.kind
                == chelis_runtime_identity::UnitKind::BuildScript
                && unit.package.name == "fixture-runtime-input"
                && unit.configuration.opt_level == "1"),
            "fixture must observe the actual host build-script configuration, not infer the runtime profile"
        );
    } else {
        panic!("host build-script witness lost source provenance");
    }

    // Stable Cargo's cached artifact events must not turn manifest-global feature
    // resolution into a made-up per-unit observation. Warm the dependency alone
    // using unmanaged Cargo, then build the actual consumer with the driver.
    let warm = fixture.invoke(
        "warm-cached",
        "fixture-runtime-input",
        &["--features", "boosted"],
        &[],
        false,
    );
    assert!(
        warm.status.success(),
        "mandatory cache warm failed: {}",
        String::from_utf8_lossy(&warm.stderr)
    );
    let fresh = fixture.invoke(
        "warm-cached",
        "fixture-runtime-input",
        &["--features", "boosted"],
        &[],
        false,
    );
    assert!(
        fresh.status.success(),
        "mandatory cached dependency probe failed: {}",
        String::from_utf8_lossy(&fresh.stderr)
    );
    let cached_inputs: Vec<serde_json::Value> = String::from_utf8(fresh.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .filter(|message| {
            message["reason"] == "compiler-artifact"
                && message["target"]["name"] == "fixture_runtime_input"
        })
        .collect();
    assert_eq!(cached_inputs.len(), 1);
    assert_eq!(
        cached_inputs[0]["fresh"], true,
        "fixture must actually begin with a cached dependency"
    );
    assert_eq!(cached_inputs[0]["features"], serde_json::json!(["boosted"]));
    let refused = fixture.invoke(
        "warm-cached",
        "chelis-cli",
        &[
            "-p",
            "chelis-runtime",
            "-p",
            "chelis-python",
            "--lib",
            "--features",
            "chelis-cli/identity-unify",
        ],
        &[],
        true,
    );
    let refused_events: Vec<serde_json::Value> = String::from_utf8_lossy(&refused.stdout)
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect();
    assert!(
        !refused_events
            .iter()
            .any(|message| message["reason"] == "compiler-artifact"
                && matches!(
                    message["target"]["name"].as_str(),
                    Some("chelis" | "chelis_runtime" | "chelis_python")
                )),
        "an incomplete cached recipe emitted a new runtime/CLI/Python native artifact"
    );
    fails_for(refused, "CHELIS_IDENTITY_MISSING_OBSERVATION");
    // The adapter refuses incomplete user-owned caches rather than deleting them.
    // Only this fixture may discard its own roots to establish the clean control.
    fs::remove_dir_all(fixture.root.join("warm-cached")).unwrap();
    fs::remove_dir_all(fixture.root.join("warm-cached-observations")).unwrap();
    let warmed = fixture.build_unified("warm-cached", "chelis-cli", &[], &[]);
    agrees(&boosted, &warmed);
    let cached_again = fixture.build_unified("warm-cached", "chelis-cli", &[], &[]);
    agrees(&boosted, &cached_again);
    for (product, expected) in [
        (&plain, Vec::<String>::new()),
        (&cached_again, vec!["boosted".to_owned()]),
    ] {
        let BuildProvenance::SourceWorktree { recipe, .. } = &product.provenance else {
            panic!("cache witness lost its captured recipe");
        };
        let observed: Vec<_> = recipe
            .units
            .iter()
            .filter(|unit| {
                unit.package.name == "fixture-runtime-input"
                    && unit.kind == chelis_runtime_identity::UnitKind::Library
            })
            .collect();
        assert_eq!(observed.len(), 1);
        assert_eq!(
            observed[0].features, expected,
            "cached effective dependency features were inferred incorrectly"
        );
    }

    fixture.append(
        "identity-fixtures/leaf/src/lib.rs",
        "\npub const DIRTY_TRANSITIVE: u32 = 99;\n",
    );
    let transitive = fixture.build_unified("fixture-boosted", "chelis-runtime", &[], &[]);
    differs(&boosted.descriptor, &transitive.descriptor, "source");
    let refreshed_consumer = fixture.build_unified("warm-cached", "chelis-cli", &[], &[]);
    agrees(&transitive, &refreshed_consumer);
    assert!(compare(&boosted.descriptor, &refreshed_consumer.descriptor).is_err());
    let extra = fixture
        .source
        .join("identity-fixtures/leaf/src/untracked.rs");
    fs::write(&extra, "pub const NEW_TRANSITIVE: u32 = 5;\n").unwrap();
    let transitive_added = fixture.build_unified("fixture-boosted", "chelis-runtime", &[], &[]);
    differs(
        &transitive.descriptor,
        &transitive_added.descriptor,
        "source",
    );
    fs::remove_file(extra).unwrap();
    agrees(
        &transitive,
        &fixture.build_unified("fixture-boosted", "chelis-runtime", &[], &[]),
    );

    fs::write(
        fixture
            .source
            .join("identity-fixtures/input/inputs/value.txt"),
        "8\n",
    )
    .unwrap();
    let generated = fixture.build_unified("fixture-boosted", "chelis-runtime", &[], &[]);
    differs(&transitive.descriptor, &generated.descriptor, "source");
    agrees(
        &generated,
        &fixture.build_unified("warm-cached", "chelis-cli", &[], &[]),
    );
    fs::remove_file(
        fixture
            .source
            .join("identity-fixtures/input/inputs/value.txt"),
    )
    .unwrap();
    fails_for(
        fixture.invoke(
            "fixture-boosted",
            "chelis-runtime",
            &["--features", "fixture-runtime-input/boosted"],
            &[],
            true,
        ),
        "value.txt",
    );
    fs::write(
        fixture
            .source
            .join("identity-fixtures/input/inputs/value.txt"),
        "8\n",
    )
    .unwrap();
    agrees(
        &generated,
        &fixture.build_unified("fixture-boosted", "chelis-runtime", &[], &[]),
    );

    fixture.edit_manifest("identity-fixtures/leaf/Cargo.toml", |manifest| {
        manifest["package"]["version"] = "0.1.1".into();
    });
    let resolution = fixture.build_unified("fixture-boosted", "chelis-runtime", &[], &[]);
    differs(&generated.descriptor, &resolution.descriptor, "source");
    agrees(
        &resolution,
        &fixture.build_unified("warm-cached", "chelis-cli", &[], &[]),
    );
}

#[test]
#[ignore = "requires two real supported native Rust compilers and native producer prerequisites"]
fn alternate_real_compiler_changes_the_toolchain_dimension() {
    let alternate = std::env::var("CHELIS_IDENTITY_TEST_ALTERNATE_RUSTC")
        .expect("mandatory toolchain witness: set CHELIS_IDENTITY_TEST_ALTERNATE_RUSTC to a different installed, supported real rustc");
    let current = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = |compiler: &str| {
        let output = std::process::Command::new(compiler)
            .args(["--version", "--verbose"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "mandatory real compiler probe failed for {compiler}"
        );
        output.stdout
    };
    assert_ne!(
        version(&current),
        version(&alternate),
        "toolchain witness must use genuinely different compiler observations"
    );
    let fixture = Fixture::new();
    let baseline = fixture.build("toolchain-a", "chelis-runtime", &[], &[]);
    let changed = fixture.build(
        "toolchain-b",
        "chelis-runtime",
        &[],
        &[("RUSTC", &alternate)],
    );
    differs(&baseline.descriptor, &changed.descriptor, "toolchain");
    let expectation = fixture.build(
        "toolchain-cli-b",
        "chelis-cli",
        &[],
        &[("RUSTC", &alternate)],
    );
    agrees(&changed, &expectation);
    assert!(compare(&baseline.descriptor, &expectation.descriptor).is_err());
}
