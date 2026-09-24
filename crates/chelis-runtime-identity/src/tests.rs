use super::*;

fn unit(name: &str, input: &str) -> CompileUnit {
    CompileUnit {
        package: PackageIdentity {
            name: name.into(),
            version: "1.0.0".into(),
            source: format!("path:{name}"),
            checksum: None,
        },
        target_name: name.into(),
        kind: UnitKind::Library,
        target: TargetObservation {
            triple: "aarch64-apple-darwin".into(),
            llvm_triple: "arm64-apple-macosx11.0.0".into(),
            data_layout: "captured arm64 layout".into(),
            cpu: "generic".into(),
            features: vec![],
            specification: None,
        },
        features: vec![],
        configuration: CompileConfiguration {
            opt_level: "0".into(),
            debuginfo: "2".into(),
            debug_assertions: true,
            panic: "unwind".into(),
            rustflags: vec![],
            cfg: vec![],
        },
        compiler: CompilerObservation {
            verbose_version: "rustc 1.98.0 (test)\nhost: aarch64-apple-darwin".into(),
            tools: vec![],
        },
        inputs: vec![input.into()],
        dependencies: vec![],
        build_script: None,
        build_environment: vec![],
        compiler_environment: vec![],
    }
}

fn fixture() -> (RuntimeRecipe, Vec<CapturedInput>) {
    let mut runtime = unit("runtime", "runtime/src/lib.rs");
    runtime.dependencies.push(DependencyBinding {
        name: "dep".into(),
        unit: 1,
    });
    let required_inputs = vec![
        RequiredInput {
            logical_path: "runtime/src/lib.rs".into(),
            class: InputClass::Source,
        },
        RequiredInput {
            logical_path: "dep/src/lib.rs".into(),
            class: InputClass::Source,
        },
        RequiredInput {
            logical_path: "runtime/include/runtime.h".into(),
            class: InputClass::Header,
        },
    ];
    let captured = required_inputs
        .iter()
        .map(|input| CapturedInput {
            logical_path: input.logical_path.clone(),
            digest: hash_bytes(input.logical_path.as_bytes()),
        })
        .collect();
    (
        RuntimeRecipe {
            schema_version: 1,
            runtime_unit: 0,
            units: vec![runtime, unit("dep", "dep/src/lib.rs")],
            required_inputs,
            cargo_profile: ProfileClass::Debug,
            public_abi: 2,
        },
        captured,
    )
}

#[test]
fn complete_observations_are_required() {
    let (recipe, mut captured) = fixture();
    captured.pop();
    assert!(matches!(
        derive_descriptor(&recipe, &captured),
        Err(InputError::MissingInput { .. })
    ));
    let (mut recipe, captured) = fixture();
    recipe.units[0].dependencies.push(DependencyBinding {
        name: "missing".into(),
        unit: 99,
    });
    assert_eq!(
        derive_descriptor(&recipe, &captured),
        Err(InputError::MissingUnit { unit: 99 })
    );
    let (mut recipe, captured) = fixture();
    recipe.units[1].dependencies.push(DependencyBinding {
        name: "runtime".into(),
        unit: 0,
    });
    assert!(matches!(
        derive_descriptor(&recipe, &captured),
        Err(InputError::Cycle { .. })
    ));
    let (mut recipe, captured) = fixture();
    recipe.units[1].compiler.verbose_version.clear();
    assert!(matches!(
        derive_descriptor(&recipe, &captured),
        Err(InputError::InvalidObservation { .. })
    ));
}

#[test]
fn build_script_and_compiler_environment_remain_distinct_inputs() {
    let (mut recipe, captured) = fixture();
    recipe.units[0]
        .build_environment
        .push(EnvironmentObservation {
            name: "CONFIG".into(),
            value: Some("script-value".into()),
        });
    recipe.units[0]
        .compiler_environment
        .push(EnvironmentObservation {
            name: "CONFIG".into(),
            value: Some("rustc-value".into()),
        });
    let expected = derive_descriptor(&recipe, &captured).unwrap();
    let mut changed = recipe.clone();
    changed.units[0].build_environment[0].value = Some("other-script-value".into());
    assert_ne!(
        expected.compile,
        derive_descriptor(&changed, &captured).unwrap().compile
    );
    changed = recipe.clone();
    changed.units[0].compiler_environment[0].value = Some("other-rustc-value".into());
    assert_ne!(
        expected.compile,
        derive_descriptor(&changed, &captured).unwrap().compile
    );
    let unit = &mut recipe.units[0];
    std::mem::swap(&mut unit.build_environment, &mut unit.compiler_environment);
    assert_ne!(
        expected.compile,
        derive_descriptor(&recipe, &captured).unwrap().compile
    );
}

#[test]
fn graph_indices_and_set_order_are_not_identity() {
    let (mut recipe, mut captured) = fixture();
    recipe.units[0].features = vec!["beta".into(), "alpha".into()];
    let expected = derive_descriptor(&recipe, &captured).unwrap();
    recipe.units[0].features.reverse();
    recipe.units.swap(0, 1);
    recipe.runtime_unit = 1;
    recipe.units[1].dependencies[0].unit = 0;
    recipe.required_inputs.reverse();
    captured.reverse();
    assert_eq!(expected, derive_descriptor(&recipe, &captured).unwrap());
    recipe.units.push(unit("unrelated-cli", "not/captured.rs"));
    assert_eq!(expected, derive_descriptor(&recipe, &captured).unwrap());
}

#[test]
fn dependency_facts_and_source_membership_discriminate() {
    let (recipe, captured) = fixture();
    let expected = derive_descriptor(&recipe, &captured).unwrap();
    let mut changed = recipe.clone();
    changed.units[1].features.push("instrumentation".into());
    let actual = derive_descriptor(&changed, &captured).unwrap();
    assert_ne!(expected.features, actual.features);
    assert_ne!(expected.recipe, actual.recipe);
    changed = recipe.clone();
    changed.units[1].configuration.opt_level = "3".into();
    assert_ne!(
        expected.compile,
        derive_descriptor(&changed, &captured).unwrap().compile
    );
    changed = recipe.clone();
    changed.required_inputs.push(RequiredInput {
        logical_path: "runtime/src/new.rs".into(),
        class: InputClass::Source,
    });
    let mut added = captured.clone();
    added.push(CapturedInput {
        logical_path: "runtime/src/new.rs".into(),
        digest: hash_bytes(b"new"),
    });
    assert_ne!(
        expected.source,
        derive_descriptor(&changed, &added).unwrap().source
    );
    let mut header = captured;
    header[2].digest = hash_bytes(b"changed header");
    assert_ne!(
        expected.interface,
        derive_descriptor(&recipe, &header).unwrap().interface
    );
}

#[test]
fn inventory_preserves_observed_unfamiliar_inputs() {
    let mut roots = vec![InventoryRoot {
        logical_prefix: "runtime".into(),
        files: vec![
            "src/lib.rs".into(),
            "tests/observed.blob".into(),
            "README.md".into(),
            "target/out.rs".into(),
            "include/api.h".into(),
        ],
        explicitly_required: vec!["tests/observed.blob".into()],
        class: InputClass::Source,
    }];
    let planned = plan_inputs(&roots).unwrap();
    assert!(
        planned
            .iter()
            .any(|input| input.logical_path == "runtime/tests/observed.blob")
    );
    assert!(
        planned
            .iter()
            .any(|input| input.logical_path == "runtime/include/api.h"
                && input.class == InputClass::Header)
    );
    assert!(
        !planned
            .iter()
            .any(|input| input.logical_path.ends_with("README.md")
                || input.logical_path.starts_with("runtime/target/"))
    );
    roots[0].explicitly_required.push("missing.blob".into());
    assert!(plan_inputs(&roots).is_err());
}

#[test]
fn mapping_is_boundary_aware_and_longest_first() {
    let mappings = vec![
        PathMapping {
            physical: "/checkout".into(),
            logical: "workspace".into(),
        },
        PathMapping {
            physical: "/checkout/dep".into(),
            logical: "dependency".into(),
        },
    ];
    assert_eq!(
        normalize_paths("--remap-path-prefix=/checkout/dep=stable", &mappings).unwrap(),
        "--remap-path-prefix=dependency=stable"
    );
    assert_eq!(
        normalize_paths("/checkout/src/lib.rs /checkout-other", &mappings).unwrap(),
        "workspace/src/lib.rs /checkout-other"
    );
}

#[test]
fn physical_package_sources_are_not_compatibility_inputs() {
    let (mut recipe, captured) = fixture();
    for source in [
        "path+file:///Users/alice/checkout/dep",
        "git+file:///Users/alice/checkout/repo#rev",
        "git+file:/Users/alice/checkout/repo#rev",
        "path:C:/checkout/dep",
        "C:\\checkout\\dep",
        "C:/checkout/dep",
    ] {
        recipe.units[1].package.source = source.into();
        assert!(
            matches!(
                derive_descriptor(&recipe, &captured),
                Err(InputError::InvalidObservation { .. })
            ),
            "accepted physical package source: {source}"
        );
    }
    recipe.units[1].package.source = "sparse+https://index.crates.io/".into();
    assert!(matches!(
        derive_descriptor(&recipe, &captured),
        Err(InputError::InvalidObservation { .. })
    ));
    recipe.units[1].package.checksum = Some(hash_bytes(b"registry archive"));
    assert!(derive_descriptor(&recipe, &captured).is_ok());
}

#[test]
fn mapping_covers_uri_paths_and_attached_compiler_flags() {
    let mappings = [PathMapping {
        physical: "/Users/alice/checkout".into(),
        logical: "workspace".into(),
    }];
    assert_eq!(
        normalize_paths(
            "path+file:///Users/alice/checkout/dep -I/Users/alice/checkout/include -L/Users/alice/checkout/lib /other/Users/alice/checkout/file",
            &mappings
        )
        .unwrap(),
        "path+file://workspace/dep -Iworkspace/include -Lworkspace/lib /other/Users/alice/checkout/file"
    );
}

#[test]
fn nested_source_directories_are_not_build_output_exclusions() {
    let roots = [InventoryRoot {
        logical_prefix: "runtime".into(),
        files: vec![
            "src/target/mod.rs".into(),
            "src/tests/feature.rs".into(),
            "target/generated.rs".into(),
        ],
        explicitly_required: vec![],
        class: InputClass::Source,
    }];
    let planned = plan_inputs(&roots).unwrap();
    assert_eq!(
        planned
            .iter()
            .map(|input| input.logical_path.as_str())
            .collect::<Vec<_>>(),
        ["runtime/src/target/mod.rs", "runtime/src/tests/feature.rs"]
    );
}

#[test]
fn toolchain_roots_keep_their_class_for_headers_and_manifests() {
    let roots = [InventoryRoot {
        logical_prefix: "toolchain".into(),
        files: vec!["include/stdint.h".into(), "rust-src/core/Cargo.toml".into()],
        explicitly_required: vec![],
        class: InputClass::Toolchain,
    }];
    let planned = plan_inputs(&roots).unwrap();
    assert_eq!(planned.len(), 2);
    assert!(
        planned
            .iter()
            .all(|input| input.class == InputClass::Toolchain)
    );
}

fn native(format: object::BinaryFormat, records: &[Vec<u8>]) -> Vec<u8> {
    let mut object = object::write::Object::new(
        format,
        object::Architecture::Aarch64,
        object::Endianness::Little,
    );
    for bytes in records {
        let (segment, name) = if format == object::BinaryFormat::MachO {
            (b"__DATA".to_vec(), b"__ch_rt_id".to_vec())
        } else {
            (vec![], b".chelis.runtime.id".to_vec())
        };
        let section = object.add_section(segment, name, object::SectionKind::ReadOnlyData);
        object.append_section_data(section, bytes, 1);
    }
    object.write().unwrap()
}

#[test]
fn native_sections_strictly_decode_in_both_formats() {
    let (recipe, captured) = fixture();
    let descriptor = derive_descriptor(&recipe, &captured).unwrap();
    let record = encode_record(&descriptor, RecordKind::Runtime)
        .unwrap()
        .to_vec();
    for format in [object::BinaryFormat::Elf, object::BinaryFormat::MachO] {
        assert_eq!(
            decode_image(
                &native(format, std::slice::from_ref(&record)),
                RecordKind::Runtime
            )
            .unwrap(),
            descriptor
        );
        assert!(matches!(
            decode_image(&native(format, &[]), RecordKind::Runtime),
            Err(RecordError::Missing)
        ));
        assert!(matches!(
            decode_image(
                &native(format, &[record.clone(), record.clone()]),
                RecordKind::Runtime
            ),
            Err(RecordError::Duplicate)
        ));
        assert!(matches!(
            decode_image(
                &native(format, &[record[..100].to_vec()]),
                RecordKind::Runtime
            ),
            Err(RecordError::Truncated { .. })
        ));
        let mut unsupported = record.clone();
        unsupported[16] = 2;
        assert!(matches!(
            decode_image(&native(format, &[unsupported]), RecordKind::Runtime),
            Err(RecordError::Unsupported { .. })
        ));
        let mut malformed = record.clone();
        malformed[19] = 1;
        assert!(matches!(
            decode_image(&native(format, &[malformed]), RecordKind::Runtime),
            Err(RecordError::Malformed { .. })
        ));
        let mut trailing = record.clone();
        trailing.push(0);
        assert!(matches!(
            decode_image(&native(format, &[trailing]), RecordKind::Runtime),
            Err(RecordError::Malformed { .. })
        ));
    }
}

#[test]
fn native_provenance_is_role_scoped_and_binds_the_retained_source() {
    let (recipe, captured) = fixture();
    let base = derive_descriptor(&recipe, &captured).unwrap();
    for format in [object::BinaryFormat::Elf, object::BinaryFormat::MachO] {
        let image = |corrupt_cli: bool| {
            let mut object = object::write::Object::new(
                format,
                object::Architecture::Aarch64,
                object::Endianness::Little,
            );
            for (kind, label, elf_id, macho_id, elf_provenance, macho_provenance) in [
                (
                    RecordKind::Runtime,
                    "runtime",
                    ".chelis.runtime.id",
                    "__ch_rt_id",
                    ".chelis.runtime.provenance",
                    "__ch_rt_prov",
                ),
                (
                    RecordKind::Cli,
                    "cli",
                    ".chelis.runtime.expect.cli",
                    "__ch_rt_cli",
                    ".chelis.runtime.provenance.cli",
                    "__ch_prv_cli",
                ),
                (
                    RecordKind::Python,
                    "python",
                    ".chelis.runtime.expect.python",
                    "__ch_rt_py",
                    ".chelis.runtime.provenance.python",
                    "__ch_prv_py",
                ),
            ] {
                let mut descriptor = base.clone();
                descriptor.source = hash_bytes(label.as_bytes());
                let source_closure = if corrupt_cli && kind == RecordKind::Cli {
                    base.source
                } else {
                    descriptor.source
                };
                let provenance =
                    encode_provenance(&BuildProvenance::SealedDistribution { source_closure })
                        .unwrap();
                for (elf, macho, bytes) in [
                    (
                        elf_id,
                        macho_id,
                        encode_record(&descriptor, kind).unwrap().to_vec(),
                    ),
                    (elf_provenance, macho_provenance, provenance),
                ] {
                    let (segment, name) = if format == object::BinaryFormat::MachO {
                        (b"__DATA".to_vec(), macho)
                    } else {
                        (vec![], elf)
                    };
                    let section = object.add_section(
                        segment,
                        name.as_bytes().to_vec(),
                        object::SectionKind::ReadOnlyData,
                    );
                    object.append_section_data(section, &bytes, 1);
                }
            }
            object.write().unwrap()
        };
        let valid = image(false);
        for (kind, label) in [
            (RecordKind::Runtime, "runtime"),
            (RecordKind::Cli, "cli"),
            (RecordKind::Python, "python"),
        ] {
            assert_eq!(
                decode_image_provenance(&valid, kind).unwrap(),
                BuildProvenance::SealedDistribution {
                    source_closure: hash_bytes(label.as_bytes())
                }
            );
        }
        assert_eq!(
            decode_archive_provenance(&archive(&[valid])).unwrap(),
            BuildProvenance::SealedDistribution {
                source_closure: hash_bytes(b"runtime")
            }
        );
        assert!(matches!(
            decode_image_provenance(&image(true), RecordKind::Cli),
            Err(RecordError::Malformed { .. })
        ));
        let missing = native(
            format,
            &[encode_record(&base, RecordKind::Runtime).unwrap().to_vec()],
        );
        assert_eq!(
            decode_image_provenance(&missing, RecordKind::Runtime),
            Err(RecordError::Missing)
        );
    }
}

#[test]
fn malformed_macho_command_after_a_valid_record_is_rejected() {
    let (recipe, captured) = fixture();
    let descriptor = derive_descriptor(&recipe, &captured).unwrap();
    let record = encode_record(&descriptor, RecordKind::Runtime)
        .unwrap()
        .to_vec();
    let mut image = native(object::BinaryFormat::MachO, &[record]);
    assert_eq!(
        decode_image(&image, RecordKind::Runtime).unwrap(),
        descriptor
    );
    let count = u32::from_le_bytes(image[16..20].try_into().unwrap());
    let mut offset = 32;
    for _ in 0..count {
        let command = u32::from_le_bytes(image[offset..offset + 4].try_into().unwrap());
        let size = u32::from_le_bytes(image[offset + 4..offset + 8].try_into().unwrap()) as usize;
        if command == object::macho::LC_SYMTAB {
            image[offset + 4..offset + 8].copy_from_slice(&0u32.to_le_bytes());
            assert!(matches!(
                decode_image(&image, RecordKind::Runtime),
                Err(RecordError::Malformed { .. })
            ));
            assert!(matches!(
                decode_archive(&archive(&[image])),
                Err(RecordError::Malformed { .. })
            ));
            return;
        }
        offset += size;
    }
    panic!("Mach-O fixture has no trailing symbol-table command");
}

#[test]
fn digest_json_and_provenance_are_strict() {
    let digest = hash_bytes(b"abc");
    assert_eq!(
        digest.to_string(),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        serde_json::from_str::<ContentDigest>(&serde_json::to_string(&digest).unwrap()).unwrap(),
        digest
    );
    assert!(
        "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
            .parse::<ContentDigest>()
            .is_err()
    );
    assert!(
        decode_provenance(br#"{"mode":"sealed_distribution","source_closure":"bad"}"#).is_err()
    );
    assert!(
        decode_provenance(
            format!(
                "{{\"mode\":\"sealed_distribution\",\"source_closure\":\"{digest}\",\"unknown\":1}}"
            )
            .as_bytes()
        )
        .is_err()
    );
}

fn archive(members: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = b"!<arch>\n".to_vec();
    for (index, member) in members.iter().enumerate() {
        let name = format!("member{index}.o/");
        bytes.extend_from_slice(
            format!(
                "{name:<16}{:<12}{:<6}{:<6}{:<8}{:<10}`\n",
                0,
                0,
                0,
                100644,
                member.len()
            )
            .as_bytes(),
        );
        bytes.extend_from_slice(member);
        if member.len() % 2 != 0 {
            bytes.push(b'\n');
        }
    }
    bytes
}

#[test]
fn archives_require_one_structured_native_record() {
    let (recipe, captured) = fixture();
    let descriptor = derive_descriptor(&recipe, &captured).unwrap();
    let record = encode_record(&descriptor, RecordKind::Runtime)
        .unwrap()
        .to_vec();
    for format in [object::BinaryFormat::Elf, object::BinaryFormat::MachO] {
        let member = native(format, std::slice::from_ref(&record));
        assert_eq!(
            decode_archive(&archive(std::slice::from_ref(&member))).unwrap(),
            descriptor
        );
        assert!(matches!(
            decode_archive(&archive(&[member.clone(), member.clone()])),
            Err(RecordError::Duplicate)
        ));
        let mut truncated = archive(std::slice::from_ref(&member));
        truncated.truncate(truncated.len() - 8);
        assert!(matches!(
            decode_archive(&truncated),
            Err(RecordError::Truncated { .. })
        ));
        let concatenated = [record.as_slice(), record.as_slice()].concat();
        assert!(matches!(
            decode_archive(&archive(&[native(format, &[concatenated])])),
            Err(RecordError::Duplicate)
        ));
        let mut malformed = record.clone();
        malformed[0] = b'?';
        assert!(matches!(
            decode_archive(&archive(&[native(format, &[malformed])])),
            Err(RecordError::Malformed { .. })
        ));
        assert!(matches!(
            decode_archive(&archive(&[native(format, &[])])),
            Err(RecordError::Missing)
        ));
    }
    assert!(matches!(
        decode_archive(b"!<thin>\n"),
        Err(RecordError::Unsupported { .. })
    ));
    assert!(matches!(
        decode_archive(&archive(&[record])),
        Err(RecordError::Malformed { .. })
    ));
}

#[test]
fn unsupported_native_members_and_invalid_archive_suffix_have_distinct_errors() {
    for member in [
        b"BC\xc0\xde".as_slice(),
        b"not an object".as_slice(),
        b"".as_slice(),
    ] {
        assert!(matches!(
            decode_archive(&archive(&[member.to_vec()])),
            Err(RecordError::Unsupported { .. })
        ));
    }
    for prefix in [
        b"\x7fELF\x02\x01\x01\0\0\0".as_slice(),
        b"\xcf\xfa\xed\xfe\x0c\0\0\x01".as_slice(),
    ] {
        assert!(matches!(
            decode_archive(&archive(&[prefix.to_vec()])),
            Err(RecordError::Truncated { .. })
        ));
    }
    let valid = native(object::BinaryFormat::Elf, &[]);
    let mut archive_with_suffix = archive(&[valid]);
    archive_with_suffix.push(b'\n');
    assert!(matches!(
        decode_archive(&archive_with_suffix),
        Err(RecordError::Malformed { .. })
    ));
}

#[test]
fn observed_flag_order_and_graph_topology_remain_significant() {
    let (mut recipe, captured) = fixture();
    recipe.units[1].configuration.rustflags = vec![
        "-C".into(),
        "opt-level=0".into(),
        "-C".into(),
        "opt-level=3".into(),
    ];
    let expected = derive_descriptor(&recipe, &captured).unwrap();
    recipe.units[1].configuration.rustflags.rotate_left(2);
    assert_ne!(
        expected.compile,
        derive_descriptor(&recipe, &captured).unwrap().compile
    );
    let (mut recipe, captured) = fixture();
    let expected = derive_descriptor(&recipe, &captured).unwrap();
    recipe.units.push(recipe.units[1].clone());
    recipe.units[0].dependencies.push(DependencyBinding {
        name: "other".into(),
        unit: 2,
    });
    assert_ne!(
        expected.recipe,
        derive_descriptor(&recipe, &captured).unwrap().recipe
    );
}

#[test]
fn swapping_extern_bindings_changes_compatibility_without_changing_the_unit_set() {
    let (mut recipe, captured) = fixture();
    let mut alternate = recipe.units[1].clone();
    alternate.features.push("different_implementation".into());
    recipe.units.push(alternate);
    recipe.units[0].dependencies.push(DependencyBinding {
        name: "alternate".into(),
        unit: 2,
    });
    let expected = derive_descriptor(&recipe, &captured).unwrap();
    recipe.units[0].dependencies[0].unit = 2;
    recipe.units[0].dependencies[1].unit = 1;
    assert!(compare(&expected, &derive_descriptor(&recipe, &captured).unwrap()).is_err());
    recipe.units[0].dependencies.swap(0, 1);
    recipe.units[0].dependencies[0].unit = 2;
    recipe.units[0].dependencies[1].unit = 1;
    assert_eq!(expected, derive_descriptor(&recipe, &captured).unwrap());
}

#[test]
fn schemas_abi_and_canonical_paths_fail_closed() {
    let (mut recipe, captured) = fixture();
    recipe.public_abi = 0;
    assert!(matches!(
        derive_descriptor(&recipe, &captured),
        Err(InputError::Unsupported { .. })
    ));
    recipe.public_abi = 2;
    recipe.schema_version = 2;
    assert!(matches!(
        derive_descriptor(&recipe, &captured),
        Err(InputError::Unsupported { .. })
    ));
    let (mut recipe, captured) = fixture();
    recipe.required_inputs[0].logical_path = "../outside.rs".into();
    assert!(matches!(
        derive_descriptor(&recipe, &captured),
        Err(InputError::InvalidPath { .. })
    ));
    let (recipe, mut captured) = fixture();
    captured.push(captured[0].clone());
    assert!(matches!(
        derive_descriptor(&recipe, &captured),
        Err(InputError::DuplicateInput { .. })
    ));
}
