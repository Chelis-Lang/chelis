mod support;

use chelis_shell_core::{
    SHELL_FORMAT_VERSION, ShellCollection, ShellPackage, ShellTextField, ShellValidationError,
    SymbolKind, TypeVariableDomain, validate_shell,
};
use chelis_syntax_host::NativeStack;
use support::{all_obligations, shell, symbol};

#[test]
fn full_metadata_accepts_all_symbol_and_domain_variants() {
    assert_eq!(SHELL_FORMAT_VERSION, 8);
    for domain in [
        TypeVariableDomain::ActiveFloat,
        TypeVariableDomain::ActiveInt,
        TypeVariableDomain::ActiveNumeric,
        TypeVariableDomain::FloatValue,
        TypeVariableDomain::IntValue,
        TypeVariableDomain::NumericValue,
        TypeVariableDomain::ActiveSet(vec!["f32".into(), "f64".into()]),
    ] {
        for kind in [
            SymbolKind::Value,
            SymbolKind::Type,
            SymbolKind::Macro,
            SymbolKind::Dim,
        ] {
            for has_body in [false, true] {
                let mut package = shell();
                let export = symbol(&mut package);
                export.kind = kind;
                export.has_body = has_body;
                export.type_variable_restrictions[0].domain = domain.clone();
                export.collection_obligations = all_obligations();
                validate_shell(&package, &NativeStack).expect("valid complete metadata");
                let json = serde_json::to_vec(&package).expect("serialize complete metadata");
                let restored: ShellPackage =
                    serde_json::from_slice(&json).expect("restore metadata");
                assert_eq!(restored, package);
            }
        }
    }
}

#[test]
fn format_versions_reject_predecessors_and_future_values_before_other_fields() {
    for version in [0, 1, 2, 3, 4, 5, 6, 7, 9, u32::MAX] {
        let mut package = shell();
        package.format_version = version;
        package.package.name.clear();
        let error = validate_shell(&package, &NativeStack).expect_err("unsupported format");
        assert!(
            matches!(error, ShellValidationError::UnsupportedFormatVersion { version: actual } if actual == version)
        );
        let note = if version == 6 {
            "Deep 0.20 removes pipe nodes; "
        } else {
            ""
        };
        assert_eq!(
            support::shell_message(&error),
            format!(
                "payload format version {version} is unsupported; expected 8; {note}regenerate the shell package"
            )
        );
    }
}

#[test]
fn text_fields_reject_empty_and_edge_whitespace_with_typed_errors() {
    type Mutate = fn(&mut ShellPackage, &str);
    let fields: [(&str, Mutate); 7] = [
        ("package.name", |p, v| p.package.name = v.into()),
        ("package.version", |p, v| p.package.version = v.into()),
        ("module name", |p, v| p.modules[0].module = v.into()),
        ("export name", |p, v| symbol(p).name = v.into()),
        ("effect name", |p, v| symbol(p).effects = vec![v.into()]),
        ("dependency name", |p, v| p.dependencies[0].name = v.into()),
        ("dependency `dependency` version", |p, v| {
            p.dependencies[0].version = v.into()
        }),
    ];
    for (field, mutate) in fields {
        for value in ["", " ", " edge", "edge\n", "\u{2003}edge"] {
            if field.ends_with("version") && field.starts_with("dependency") && value.is_empty() {
                continue;
            }
            let mut package = shell();
            mutate(&mut package, value);
            let error = validate_shell(&package, &NativeStack).expect_err("invalid text");
            if field.starts_with("dependency `") {
                assert!(
                    matches!(&error, ShellValidationError::DependencyVersionWhitespace { dependency }
                    if dependency == "dependency")
                );
            } else {
                let expected_field = match field {
                    "package.name" => ShellTextField::PackageName,
                    "package.version" => ShellTextField::PackageVersion,
                    "module name" => ShellTextField::ModuleName,
                    "export name" => ShellTextField::ExportName,
                    "effect name" => ShellTextField::EffectName,
                    "dependency name" => ShellTextField::DependencyName,
                    _ => unreachable!(),
                };
                assert!(
                    matches!(&error, ShellValidationError::InvalidText { field: actual }
                    if *actual == expected_field),
                    "{field}: {error:?}"
                );
            }
            let expected = if field.starts_with("dependency `") {
                format!("{field} must not have leading or trailing whitespace")
            } else {
                format!("{field} must be non-empty without leading or trailing whitespace")
            };
            assert_eq!(support::shell_message(&error), expected);
        }
    }
}

#[test]
fn compiler_pin_accepts_semver_release_prerelease_build_and_u64_boundaries() {
    for compiler in [
        "=0.0.0",
        "=1.2.3-alpha.0+build.01",
        "=1.2.3-0",
        "=1.2.3-alpha-beta+build-meta",
        "=18446744073709551615.18446744073709551615.18446744073709551615",
    ] {
        let mut package = shell();
        package.compiler = compiler.into();
        validate_shell(&package, &NativeStack).expect("valid exact semver pin");
    }
}

#[test]
fn compiler_pin_distinguishes_missing_prefix_from_invalid_semver() {
    for compiler in ["", "1.2.3", "^1.2.3", " =1.2.3", "v1.2.3"] {
        let mut package = shell();
        package.compiler = compiler.into();
        let error = validate_shell(&package, &NativeStack).expect_err("missing exact pin prefix");
        assert!(matches!(error, ShellValidationError::CompilerPinPrefix));
        assert_eq!(
            support::shell_message(&error),
            "compiler must be an exact pin beginning with `=`"
        );
    }
    for version in [
        "",
        "1",
        "1.2",
        "01.2.3",
        "1.02.3",
        "1.2.03",
        "1.2.3-01",
        "1.2.3-",
        "1.2.3+",
        "1.2.3 ",
        "1.2.3\n",
        "v1.2.3",
        "=1.2.3",
        "1.2.3, <2.0.0",
        "18446744073709551616.0.0",
    ] {
        let mut package = shell();
        package.compiler = format!("={version}");
        let error = validate_shell(&package, &NativeStack).expect_err("invalid semver pin");
        let ShellValidationError::CompilerPinVersion { source } = &error else {
            panic!("wrong error for {version:?}: {error:?}");
        };
        let expected = semver::Version::parse(version).expect_err("invalid semver fixture");
        assert_eq!(source.to_string(), expected.to_string());
        assert_eq!(
            support::shell_message(&error),
            format!("compiler must be an exact `=<semver>` pin: {expected}")
        );
    }
}

#[test]
fn archive_digest_rejects_wrong_length_case_and_non_hex_bytes() {
    for digest in [
        String::new(),
        "a".repeat(63),
        "a".repeat(65),
        "A".repeat(64),
        "g".repeat(64),
        format!("{}\n", "a".repeat(63)),
        "é".repeat(32),
    ] {
        let mut package = shell();
        package.archive_sha256 = digest;
        let error = validate_shell(&package, &NativeStack).expect_err("invalid archive digest");
        assert!(matches!(error, ShellValidationError::ArchiveSha256));
        assert_eq!(
            support::shell_message(&error),
            "archive_sha256 must be exactly 64 lowercase hexadecimal characters"
        );
    }
}

#[test]
fn modules_exports_and_dependencies_reject_duplicates_and_descending_order() {
    for family in ["modules", "exports", "dependencies"] {
        for duplicate in [false, true] {
            let mut package = shell();
            let label = match family {
                "modules" => {
                    let mut second = package.modules[0].clone();
                    second.module = if duplicate { "Boundary.Types" } else { "A" }.into();
                    package.modules.push(second);
                    "modules"
                }
                "exports" => {
                    let mut second = symbol(&mut package).clone();
                    second.name = if duplicate { "apply" } else { "a" }.into();
                    package.modules[0].exports.push(second);
                    "exports for module `Boundary.Types`"
                }
                "dependencies" => {
                    let mut second = package.dependencies[0].clone();
                    second.name = if duplicate { "dependency" } else { "a" }.into();
                    package.dependencies.push(second);
                    "dependencies"
                }
                _ => unreachable!(),
            };
            let error = validate_shell(&package, &NativeStack).expect_err("noncanonical order");
            let expected_collection = match family {
                "modules" => ShellCollection::Modules,
                "exports" => ShellCollection::Exports {
                    module: "Boundary.Types".into(),
                },
                "dependencies" => ShellCollection::Dependencies,
                _ => unreachable!(),
            };
            assert!(
                matches!(&error, ShellValidationError::StrictOrder { collection }
                if *collection == expected_collection)
            );
            assert_eq!(
                support::shell_message(&error),
                format!("{label} must be strictly sorted with no duplicates")
            );
        }
    }
}

#[test]
fn sorted_metadata_accepts_empty_collections_and_non_semver_package_versions() {
    let mut package = shell();
    package.package.version = "authored version".into();
    package.dependencies[0].version = "not a semver range".into();
    symbol(&mut package).effects = vec!["State".into(), "IO".into(), "IO".into()];
    validate_shell(&package, &NativeStack)
        .expect("text fields do not impose semver or effect ordering");
    let mut module = package.modules[0].clone();
    module.module = "Z".into();
    module.exports.clear();
    package.modules.push(module);
    let mut dependency = package.dependencies[0].clone();
    dependency.name = "z".into();
    dependency.version.clear();
    package.dependencies.push(dependency);
    validate_shell(&package, &NativeStack).expect("strictly sorted metadata");
    package.modules.clear();
    package.dependencies.clear();
    validate_shell(&package, &NativeStack).expect("empty metadata collections");
}

#[test]
fn value_signatures_are_required_for_both_body_states() {
    for has_body in [false, true] {
        let mut package = shell();
        let export = symbol(&mut package);
        export.type_repr = None;
        export.type_variable_restrictions.clear();
        export.has_body = has_body;
        let error = validate_shell(&package, &NativeStack).expect_err("missing value signature");
        assert!(
            matches!(&error, ShellValidationError::MissingValueType { module, symbol }
            if module == "Boundary.Types" && symbol == "apply")
        );
        assert_eq!(
            support::shell_message(&error),
            "value export `apply` in module `Boundary.Types` requires a type representation"
        );
    }
}

#[test]
fn non_value_exports_accept_absent_types_but_reject_invalid_present_types() {
    for kind in [SymbolKind::Type, SymbolKind::Macro, SymbolKind::Dim] {
        let mut package = shell();
        let export = symbol(&mut package);
        export.kind = kind;
        export.type_repr = None;
        export.type_variable_restrictions.clear();
        validate_shell(&package, &NativeStack).expect("non-value type is optional");
        for source in ["", "(t-unit {}"] {
            symbol(&mut package).type_repr = Some(source.into());
            let error = validate_shell(&package, &NativeStack).expect_err("invalid present type");
            assert!(matches!(
                error,
                ShellValidationError::TypeRepresentation { .. }
            ));
        }
    }
}

#[test]
fn envelope_errors_precede_modules_and_dependencies() {
    let mut package = shell();
    package.package.name.clear();
    package.package.version.clear();
    package.compiler.clear();
    package.archive_sha256.clear();
    package.modules[0].module.clear();
    package.dependencies[0].name.clear();
    let expected = [
        "package.name must be non-empty without leading or trailing whitespace",
        "package.version must be non-empty without leading or trailing whitespace",
        "compiler must be an exact pin beginning with `=`",
        "archive_sha256 must be exactly 64 lowercase hexadecimal characters",
        "module name must be non-empty without leading or trailing whitespace",
        "dependency name must be non-empty without leading or trailing whitespace",
    ];
    for (index, message) in expected.into_iter().enumerate() {
        assert_eq!(
            support::shell_message(
                &validate_shell(&package, &NativeStack).expect_err("first error")
            ),
            message
        );
        match index {
            0 => package.package.name = "fixed".into(),
            1 => package.package.version = "fixed".into(),
            2 => package.compiler = "=1.2.3".into(),
            3 => package.archive_sha256 = "a".repeat(64),
            4 => package.modules[0].module = "Fixed".into(),
            5 => package.dependencies[0].name = "fixed".into(),
            _ => unreachable!(),
        }
    }
    validate_shell(&package, &NativeStack).expect("all faults repaired");
}

#[test]
fn collection_order_precedes_invalid_member_text() {
    let mut package = shell();
    let mut second = package.modules[0].clone();
    second.module.clear();
    package.modules.push(second);
    let error = validate_shell(&package, &NativeStack).expect_err("order precedes module text");
    assert!(matches!(error, ShellValidationError::StrictOrder { .. }));
    assert_eq!(
        support::shell_message(&error),
        "modules must be strictly sorted with no duplicates"
    );
    package.modules.pop();
    let mut second = symbol(&mut package).clone();
    second.name.clear();
    package.modules[0].exports.push(second);
    assert_eq!(
        support::shell_message(
            &validate_shell(&package, &NativeStack).expect_err("order precedes export text")
        ),
        "exports for module `Boundary.Types` must be strictly sorted with no duplicates"
    );
    package.modules[0].exports.pop();
    let mut second = package.dependencies[0].clone();
    second.name.clear();
    package.dependencies.push(second);
    assert_eq!(
        support::shell_message(
            &validate_shell(&package, &NativeStack).expect_err("order precedes dependency text")
        ),
        "dependencies must be strictly sorted with no duplicates"
    );
}
