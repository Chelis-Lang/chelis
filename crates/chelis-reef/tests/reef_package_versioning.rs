use chelis_reef::package_versioning::{
    CandidateSource, DependencyRequirement, ExactCompilerVersion, LocalCandidate, LockAssessment,
    LockDependency, LockSnapshot, PackageName, PackageRequirement, PackageVersion,
    RequestedPackage, ResolvedPackageId, ResolverLimits, ResolverVersion, TypedDependency,
    VersioningError, assess_lock, resolve_local,
};
use chelis_reef::{
    BuildOptions, GitHubReleaseSpec, LocalRegistryIndex, ManifestSchemaVersion, RegistryVersion,
    UpgradeMode, build_package_with_options, canonicalize_local_registry_index, init_package,
    package_schema, publish_package, upgrade_documents,
};
use std::collections::BTreeMap;
use std::str::FromStr;
use std::sync::Mutex;
use tempfile::tempdir;

const ACTIVE_COMPILER_PIN: &str = concat!("=", env!("CARGO_PKG_VERSION"));
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn name(value: &str) -> PackageName {
    PackageName::from_str(value).unwrap()
}

fn version(value: &str) -> PackageVersion {
    PackageVersion::from_str(value).unwrap()
}

fn requirement(value: &str) -> PackageRequirement {
    PackageRequirement::from_str(value).unwrap()
}

fn requested(requester: &str, package: &str, value: &str) -> RequestedPackage {
    RequestedPackage::new(name(requester), name(package), requirement(value))
}

fn candidate(package: &str, value: &str, dependencies: &[(&str, &str)]) -> LocalCandidate {
    LocalCandidate {
        id: ResolvedPackageId::new(name(package), version(value)),
        source: CandidateSource::LocalRegistry {
            source_identity: format!("registry://{package}/{value}"),
        },
        content_identity: format!("sha256:{package}-{value}"),
        dependencies: dependencies
            .iter()
            .map(|(dependency, value)| requested(package, dependency, value))
            .collect(),
    }
}

#[test]
fn package_names_are_canonical_and_portable() {
    for valid in ["a", "nautilus-core", "x1", &"a".repeat(64)] {
        let parsed = PackageName::from_str(valid).unwrap();
        assert_eq!(parsed.as_str(), valid);
        assert_eq!(parsed.to_string(), valid);
    }
    for invalid in [
        "",
        "Nautilus",
        "nautilus_core",
        "../nautilus",
        "nautilus/core",
        "-nautilus",
        "nautilus-",
        "nautilus--core",
        "1nautilus",
        "con",
        "prn",
        "aux",
        "nul",
        "com1",
        "com9",
        "lpt1",
        "lpt9",
        &"a".repeat(65),
    ] {
        assert!(
            matches!(
                PackageName::from_str(invalid),
                Err(VersioningError::InvalidPackageName { .. })
            ),
            "{invalid:?}"
        );
    }
}

#[test]
fn package_versions_require_complete_semver_without_metadata() {
    for valid in ["0.0.0", "1.2.3", "1.2.3-alpha.1"] {
        let parsed = PackageVersion::from_str(valid).unwrap();
        assert_eq!(parsed.to_string(), valid);
    }
    for invalid in [
        "1.2",
        "1",
        "01.2.3",
        "1.02.3",
        "1.2.03",
        "1.2.3+portable",
        "18446744073709551616.0.0",
    ] {
        assert!(
            matches!(
                PackageVersion::from_str(invalid),
                Err(VersioningError::InvalidPackageVersion { .. })
            ),
            "{invalid:?}"
        );
    }
}

#[test]
fn typed_identity_deserialization_reuses_the_parse_boundaries() {
    assert!(serde_json::from_str::<PackageName>("\"Bad_Name\"").is_err());
    assert!(serde_json::from_str::<PackageVersion>("\"1.2.3+portable\"").is_err());
    assert!(serde_json::from_str::<ExactCompilerVersion>("\"0.18.4\"").is_err());
    assert!(serde_json::from_str::<PackageRequirement>("\"=1.2.3+portable\"").is_err());
}

#[test]
fn resolver_versions_keep_schema_one_exact_and_schema_two_cargo_compatible() {
    assert_eq!(
        ResolverVersion::from_str("1").unwrap(),
        ResolverVersion::One
    );
    assert_eq!(
        ResolverVersion::from_str("2").unwrap(),
        ResolverVersion::Two
    );
    assert!(matches!(
        ResolverVersion::from_str("3"),
        Err(VersioningError::UnsupportedResolver { .. })
    ));

    let exact = DependencyRequirement::parse(ResolverVersion::One, "1.2.3").unwrap();
    assert!(exact.matches(&version("1.2.3")));
    assert!(!exact.matches(&version("1.2.4")));
    assert!(DependencyRequirement::parse(ResolverVersion::One, "=1.2.3").is_err());

    let caret = DependencyRequirement::parse(ResolverVersion::Two, "1.2.3").unwrap();
    assert!(caret.matches(&version("1.9.0")));
    assert!(!caret.matches(&version("2.0.0")));
    assert!(!caret.matches(&version("1.3.0-alpha.1")));

    let zero_major = DependencyRequirement::parse(ResolverVersion::Two, "0.2.3").unwrap();
    assert!(zero_major.matches(&version("0.2.9")));
    assert!(!zero_major.matches(&version("0.3.0")));

    let exact_v2 = DependencyRequirement::parse(ResolverVersion::Two, "=1.2.3").unwrap();
    assert!(exact_v2.matches(&version("1.2.3")));
    assert!(!exact_v2.matches(&version("1.2.4")));

    let intersection = DependencyRequirement::parse(ResolverVersion::Two, ">=1.2, <2.0").unwrap();
    assert!(intersection.matches(&version("1.9.0")));
    assert!(!intersection.matches(&version("2.0.0")));
    assert!(DependencyRequirement::parse(ResolverVersion::Two, ">=1.0 <2.0").is_err());
}

#[test]
fn schema_two_shorthand_and_inline_dependencies_are_equivalent() {
    let shorthand = TypedDependency::registry(ResolverVersion::Two, "0.7").unwrap();
    let inline = TypedDependency::inline(ResolverVersion::Two, Some("0.7"), None).unwrap();
    assert_eq!(shorthand, inline);

    let path_only = TypedDependency::inline(ResolverVersion::Two, None, Some("../octant")).unwrap();
    assert!(path_only.requirement().is_none());
    assert_eq!(path_only.path(), Some("../octant"));

    let checked_path =
        TypedDependency::inline(ResolverVersion::Two, Some("0.4"), Some("../octant")).unwrap();
    assert!(
        checked_path
            .requirement()
            .unwrap()
            .matches(&version("0.4.7"))
    );
    assert!(
        !checked_path
            .requirement()
            .unwrap()
            .matches(&version("0.5.0"))
    );
}

#[test]
fn compiler_pins_are_exact_stable_releases() {
    let pin = ExactCompilerVersion::from_str("=0.18.4").unwrap();
    assert_eq!(pin.to_string(), "=0.18.4");
    for invalid in [
        "0.18.4",
        "^0.18.4",
        ">=0.18.4",
        "=0.18.4-alpha.1",
        "=0.18.4+portable",
    ] {
        assert!(matches!(
            ExactCompilerVersion::from_str(invalid),
            Err(VersioningError::InvalidCompilerVersion { .. })
        ));
    }
}

#[test]
fn bundled_runtime_is_one_exact_local_candidate() {
    let mut bundled = candidate("chelis-std", "0.4.0", &[]);
    bundled.source = CandidateSource::BundledRuntime {
        compiler_version: ExactCompilerVersion::from_str(ACTIVE_COMPILER_PIN).unwrap(),
    };
    let candidates = BTreeMap::from([(name("chelis-std"), vec![bundled])]);

    let resolution = resolve_local(
        &[requested("root", "chelis-std", "=0.4.0")],
        &candidates,
        ResolverLimits::default(),
    )
    .unwrap();
    assert_eq!(
        resolution.selected[&name("chelis-std")].id.version,
        version("0.4.0")
    );
    assert!(matches!(
        resolve_local(
            &[requested("root", "chelis-std", "=0.5.0")],
            &candidates,
            ResolverLimits::default()
        ),
        Err(VersioningError::IncompatibleRequirements { .. })
    ));
}

#[test]
fn local_resolution_uses_numeric_semver_order() {
    let mut candidates = BTreeMap::new();
    candidates.insert(
        name("dep"),
        vec![
            candidate("dep", "1.5.0", &[]),
            candidate("dep", "1.19.0", &[]),
        ],
    );
    let resolution = resolve_local(
        &[requested("root", "dep", "^1")],
        &candidates,
        ResolverLimits::default(),
    )
    .unwrap();
    assert_eq!(
        resolution.selected[&name("dep")].id.version.to_string(),
        "1.19.0"
    );
}

#[test]
fn compatible_locked_candidate_precedes_newer_unlocked_candidate() {
    let mut locked = candidate("dep", "1.2.3", &[]);
    locked.source = CandidateSource::Locked {
        source_identity: "registry://dep/1.2.3".into(),
    };
    let candidates = BTreeMap::from([(name("dep"), vec![candidate("dep", "1.9.0", &[]), locked])]);

    let resolution = resolve_local(
        &[requested("root", "dep", "^1")],
        &candidates,
        ResolverLimits::default(),
    )
    .unwrap();

    assert_eq!(
        resolution.selected[&name("dep")].id.version,
        version("1.2.3")
    );
}

#[test]
fn local_resolution_recovers_from_a_transitive_dead_end() {
    let mut candidates = BTreeMap::new();
    candidates.insert(
        name("a"),
        vec![
            candidate("a", "2.0.0", &[("b", "^2")]),
            candidate("a", "1.0.0", &[("b", "^1")]),
        ],
    );
    candidates.insert(name("b"), vec![candidate("b", "1.5.0", &[])]);
    let resolution = resolve_local(
        &[requested("root", "a", ">=1, <3")],
        &candidates,
        ResolverLimits::default(),
    )
    .unwrap();
    assert_eq!(
        resolution.selected[&name("a")].id.version.to_string(),
        "1.0.0"
    );
    assert_eq!(
        resolution.selected[&name("b")].id.version.to_string(),
        "1.5.0"
    );
}

#[test]
fn local_resolution_memoizes_repeated_failed_requirement_sets() {
    let candidates = BTreeMap::from([
        (
            name("a"),
            vec![
                candidate("a", "2.0.0", &[("missing", "^1")]),
                candidate("a", "1.0.0", &[("missing", "^1")]),
                candidate("a", "0.5.0", &[]),
            ],
        ),
        (name("missing"), Vec::new()),
    ]);

    let resolution = resolve_local(
        &[requested("root", "a", ">=0.5, <3")],
        &candidates,
        ResolverLimits::default(),
    )
    .unwrap();

    assert_eq!(resolution.selected[&name("a")].id.version, version("0.5.0"));
    assert_eq!(resolution.memoized_failures, 1);
    assert!(resolution.memo_hits >= 1, "{resolution:?}");
}

#[test]
fn local_resolution_reports_all_disjoint_requesters() {
    let mut candidates = BTreeMap::new();
    candidates.insert(
        name("shared"),
        vec![
            candidate("shared", "1.0.0", &[]),
            candidate("shared", "2.0.0", &[]),
        ],
    );
    let error = resolve_local(
        &[
            requested("left", "shared", "^1"),
            requested("right", "shared", "^2"),
        ],
        &candidates,
        ResolverLimits::default(),
    )
    .unwrap_err();
    let VersioningError::IncompatibleRequirements {
        package,
        requirements,
        candidates,
    } = error
    else {
        panic!("unexpected error: {error:?}");
    };
    assert_eq!(package, name("shared"));
    assert_eq!(requirements.len(), 2);
    assert!(
        requirements
            .iter()
            .any(|item| item.requester == name("left"))
    );
    assert!(
        requirements
            .iter()
            .any(|item| item.requester == name("right"))
    );
    assert_eq!(candidates, vec![version("2.0.0"), version("1.0.0")]);
}

#[test]
fn path_override_has_explicit_priority_and_two_paths_conflict() {
    let registry = candidate("dep", "1.9.0", &[]);
    let path = LocalCandidate {
        id: ResolvedPackageId::new(name("dep"), version("1.4.0")),
        source: CandidateSource::Path {
            canonical_path: "/workspace/dep".into(),
        },
        content_identity: "path-bytes".into(),
        dependencies: Vec::new(),
    };
    let mut candidates = BTreeMap::from([(name("dep"), vec![registry, path.clone()])]);
    let resolution = resolve_local(
        &[requested("root", "dep", "^1")],
        &candidates,
        ResolverLimits::default(),
    )
    .unwrap();
    assert!(matches!(
        resolution.selected[&name("dep")].source,
        CandidateSource::Path { .. }
    ));

    let mut incompatible_path = path.clone();
    incompatible_path.id.version = version("9.0.0");
    incompatible_path.source = CandidateSource::Path {
        canonical_path: "/inactive/dep".into(),
    };
    candidates
        .get_mut(&name("dep"))
        .unwrap()
        .push(incompatible_path);
    let resolution = resolve_local(
        &[requested("root", "dep", "^1")],
        &candidates,
        ResolverLimits::default(),
    )
    .unwrap();
    assert_eq!(
        resolution.selected[&name("dep")].id.version,
        version("1.4.0")
    );

    candidates
        .get_mut(&name("dep"))
        .unwrap()
        .push(LocalCandidate {
            source: CandidateSource::Path {
                canonical_path: "/other/dep".into(),
            },
            ..path
        });
    assert!(matches!(
        resolve_local(
            &[requested("root", "dep", "^1")],
            &candidates,
            ResolverLimits::default()
        ),
        Err(VersioningError::PathSourceConflict { .. })
    ));
}

#[test]
fn duplicate_non_path_identity_requires_equal_source_and_content() {
    let first = candidate("dep", "1.0.0", &[]);
    let mut duplicate = first.clone();
    duplicate.content_identity = "different".into();
    let candidates = BTreeMap::from([(name("dep"), vec![first, duplicate])]);
    assert!(matches!(
        resolve_local(
            &[requested("root", "dep", "^1")],
            &candidates,
            ResolverLimits::default()
        ),
        Err(VersioningError::SourceConflict { .. })
    ));

    let first = candidate("dep", "1.0.0", &[("left", "^1")]);
    let mut different_dependencies = first.clone();
    different_dependencies.dependencies = vec![requested("dep", "right", "^1")];
    let candidates = BTreeMap::from([(name("dep"), vec![first, different_dependencies])]);
    assert!(matches!(
        resolve_local(
            &[requested("root", "dep", "^1")],
            &candidates,
            ResolverLimits::default()
        ),
        Err(VersioningError::SourceConflict { .. })
    ));
}

#[test]
fn local_limits_fail_before_excess_work() {
    let too_many = (0..257)
        .map(|patch| candidate("dep", &format!("1.0.{patch}"), &[]))
        .collect::<Vec<_>>();
    assert!(matches!(
        resolve_local(
            &[requested("root", "dep", "^1")],
            &BTreeMap::from([(name("dep"), too_many)]),
            ResolverLimits::default()
        ),
        Err(VersioningError::LimitExceeded {
            dimension: "candidates per package name",
            limit: 256,
            ..
        })
    ));

    let limits = ResolverLimits {
        states: 1,
        ..ResolverLimits::default()
    };
    let candidates = BTreeMap::from([(
        name("dep"),
        vec![
            candidate("dep", "2.0.0", &[("missing", "^2")]),
            candidate("dep", "1.0.0", &[("missing", "^1")]),
        ],
    )]);
    assert!(matches!(
        resolve_local(&[requested("root", "dep", ">=1, <3")], &candidates, limits),
        Err(VersioningError::LimitExceeded {
            dimension: "explored resolver states",
            limit: 1,
            ..
        })
    ));

    let memo_insert = BTreeMap::from([
        (
            name("dep"),
            vec![candidate("dep", "1.0.0", &[("missing", "^1")])],
        ),
        (name("missing"), Vec::new()),
    ]);
    assert!(matches!(
        resolve_local(
            &[requested("root", "dep", "^1")],
            &memo_insert,
            ResolverLimits {
                states: 1,
                ..ResolverLimits::default()
            }
        ),
        Err(VersioningError::LimitExceeded {
            dimension: "explored resolver states",
            limit: 1,
            ..
        })
    ));
}

#[test]
fn dependency_depth_limit_is_exact() {
    let mut candidates = BTreeMap::new();
    for depth in 0..=128 {
        let package = format!("p{depth}");
        let dependencies = if depth == 128 {
            Vec::new()
        } else {
            vec![(format!("p{}", depth + 1), "^1".to_string())]
        };
        candidates.insert(
            name(&package),
            vec![candidate(
                &package,
                "1.0.0",
                &dependencies
                    .iter()
                    .map(|(name, req)| (name.as_str(), req.as_str()))
                    .collect::<Vec<_>>(),
            )],
        );
    }
    let error = resolve_local(
        &[requested("root", "p0", "^1")],
        &candidates,
        ResolverLimits::default(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        VersioningError::LimitExceeded {
            dimension: "dependency depth",
            limit: 128,
            attempted: 129,
            ..
        }
    ));
}

#[test]
fn resolution_is_deterministic_across_candidate_and_requirement_order() {
    let low = candidate("dep", "1.5.0", &[]);
    let high = candidate("dep", "1.19.0", &[]);
    let first = resolve_local(
        &[
            requested("z-parent", "dep", "^1"),
            requested("a-parent", "dep", ">=1.0, <2"),
        ],
        &BTreeMap::from([(name("dep"), vec![low.clone(), high.clone()])]),
        ResolverLimits::default(),
    )
    .unwrap();
    let second = resolve_local(
        &[
            requested("a-parent", "dep", ">=1.0, <2"),
            requested("z-parent", "dep", "^1"),
        ],
        &BTreeMap::from([(name("dep"), vec![high, low])]),
        ResolverLimits::default(),
    )
    .unwrap();
    assert_eq!(first, second);
    assert_eq!(first.canonical_lock_order(), second.canonical_lock_order());
}

#[test]
fn lock_assessment_distinguishes_reuse_staleness_and_corruption() {
    let root = ResolvedPackageId::new(name("root"), version("1.0.0"));
    let dependency = DependencyRequirement::parse(ResolverVersion::One, "1.2.3").unwrap();
    let declarations = BTreeMap::from([(
        name("dep"),
        TypedDependency::Registry {
            requirement: dependency,
        },
    )]);
    let reusable = LockSnapshot {
        root: root.clone(),
        dependencies: vec![LockDependency {
            id: ResolvedPackageId::new(name("dep"), version("1.2.3")),
            source: CandidateSource::LocalRegistry {
                source_identity: "registry://dep/1.2.3".into(),
            },
            archive_sha256: "archive".into(),
            shell_sha256: "shell".into(),
        }],
    };
    assert_eq!(
        assess_lock(&root, &declarations, &reusable).unwrap(),
        LockAssessment::Reusable
    );

    let mut stale = reusable.clone();
    stale.dependencies[0].id.version = version("1.2.4");
    assert!(matches!(
        assess_lock(&root, &declarations, &stale).unwrap(),
        LockAssessment::Stale { .. }
    ));

    let wrong_root = ResolvedPackageId::new(name("other-root"), version("1.0.0"));
    assert!(matches!(
        assess_lock(&wrong_root, &declarations, &reusable).unwrap(),
        LockAssessment::Stale { .. }
    ));

    let mut wrong_source = reusable.clone();
    wrong_source.dependencies[0].source = CandidateSource::Path {
        canonical_path: "/workspace/dep".into(),
    };
    assert!(matches!(
        assess_lock(&root, &declarations, &wrong_source).unwrap(),
        LockAssessment::Stale { .. }
    ));

    let mut corrupt = reusable;
    corrupt.dependencies[0].archive_sha256.clear();
    assert!(matches!(
        assess_lock(&root, &declarations, &corrupt),
        Err(VersioningError::InvalidLock { .. })
    ));
}

#[test]
fn schema_one_rejects_resolver_two_outside_the_package_table() {
    let directory = tempdir().unwrap();
    std::fs::write(
        directory.path().join("reef.toml"),
        format!(
            "schema = \"1\"\nresolver = \"2\"\n\n[package]\nname = \"inactive\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Inactive\"\n",
            env!("CARGO_PKG_VERSION")
        ),
    )
    .unwrap();
    let error = upgrade_documents(
        directory.path(),
        UpgradeMode::Check,
        Some(ManifestSchemaVersion::from_str("1").unwrap()),
        None,
    )
    .unwrap_err();
    assert!(error.to_string().contains("resolver"), "{error}");
}

fn write_manifest(root: &std::path::Path, name: &str, dependencies: &str) {
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("reef.toml"),
        format!(
            "schema = \"1\"\n\n[package]\nname = \"{name}\"\nversion = \"0.1.0\"\ncompiler = \"{ACTIVE_COMPILER_PIN}\"\nmodule_prefix = \"Fixture\"\n{dependencies}"
        ),
    )
    .unwrap();
    std::fs::write(
        root.join("src/main.ch"),
        "module Fixture.Main\ndef value() -> int32 = 1\n",
    )
    .unwrap();
}

#[test]
fn manifest_boundary_rejects_invalid_identity_and_compiler_values() {
    for (name, version, compiler, expected) in [
        ("Bad_Name", "0.1.0", ACTIVE_COMPILER_PIN, "package name"),
        ("valid-name", "0.1", ACTIVE_COMPILER_PIN, "package version"),
        (
            "valid-name",
            "0.1.0+portable",
            ACTIVE_COMPILER_PIN,
            "build metadata",
        ),
        ("valid-name", "0.1.0", "^0.18.4", "compiler"),
    ] {
        let directory = tempdir().unwrap();
        std::fs::create_dir_all(directory.path().join("src")).unwrap();
        std::fs::write(
            directory.path().join("reef.toml"),
            format!(
                "schema = \"1\"\n\n[package]\nname = \"{name}\"\nversion = \"{version}\"\ncompiler = \"{compiler}\"\nmodule_prefix = \"Fixture\"\n"
            ),
        )
        .unwrap();
        let error = package_schema(directory.path()).unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn checked_path_dependencies_validate_name_version_and_publication() {
    let directory = tempdir().unwrap();
    let dependency = directory.path().join("dependency");
    let root = directory.path().join("root");
    write_manifest(&dependency, "dependency", "");
    write_manifest(
        &root,
        "root",
        "\n[dependencies]\ndependency = { path = \"../dependency\", version = \"0.1.0\" }\n",
    );

    build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("matching checked path dependency");
    let prior_lock = std::fs::read(root.join("reef.lock")).unwrap();
    assert!(
        publish_package(&root)
            .unwrap_err()
            .contains("path dependencies")
    );

    write_manifest(
        &root,
        "root",
        "\n[dependencies]\ndependency = { path = \"../dependency\", version = \"0.2.0\" }\n",
    );
    let mismatch =
        build_package_with_options(&root, &BuildOptions { auto_fetch: false }).unwrap_err();
    assert!(mismatch.contains("0.2.0"), "{mismatch}");
    assert!(mismatch.contains("0.1.0"), "{mismatch}");
    assert_eq!(std::fs::read(root.join("reef.lock")).unwrap(), prior_lock);

    write_manifest(
        &root,
        "root",
        "\n[dependencies]\nother-name = { path = \"../dependency\" }\n",
    );
    let mismatch =
        build_package_with_options(&root, &BuildOptions { auto_fetch: false }).unwrap_err();
    assert!(mismatch.contains("other-name"), "{mismatch}");
    assert!(mismatch.contains("dependency"), "{mismatch}");

    write_manifest(
        &root,
        "root",
        "\n[dependencies]\ndependency = { path = \"../dependency\" }\n",
    );
    build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("path-only dependency remains valid");
}

#[test]
fn path_only_dependency_accepts_a_prerelease_package_version() {
    let directory = tempdir().unwrap();
    let dependency = directory.path().join("dependency");
    let root = directory.path().join("root");
    write_manifest(&dependency, "dependency", "");
    let prerelease = std::fs::read_to_string(dependency.join("reef.toml"))
        .unwrap()
        .replace("version = \"0.1.0\"", "version = \"1.0.0-rc.1\"");
    std::fs::write(dependency.join("reef.toml"), prerelease).unwrap();
    write_manifest(
        &root,
        "root",
        "\n[dependencies]\ndependency = { path = \"../dependency\" }\n",
    );

    build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("path-only dependency must not invent a stable-only requirement");
}

#[test]
fn transitive_lock_requirements_and_reachability_are_revalidated() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("root");
    let package_a = directory.path().join("package-a");
    let package_b = directory.path().join("package-b");
    write_manifest(&package_b, "package-b", "");
    write_manifest(
        &package_a,
        "package-a",
        "\n[dependencies]\npackage-b = { path = \"../package-b\", version = \"0.1.0\" }\n",
    );
    write_manifest(
        &root,
        "root",
        "\n[dependencies]\npackage-a = { path = \"../package-a\", version = \"0.1.0\" }\n",
    );
    build_package_with_options(&root, &BuildOptions { auto_fetch: false }).unwrap();
    let lock_path = root.join("reef.lock");
    let prior_lock = std::fs::read(&lock_path).unwrap();

    write_manifest(
        &package_a,
        "package-a",
        "\n[dependencies]\npackage-b = { path = \"../package-b\", version = \"0.2.0\" }\n",
    );
    let error = build_package_with_options(&root, &BuildOptions { auto_fetch: false }).unwrap_err();
    assert!(error.contains("0.2.0"), "{error}");
    assert_eq!(std::fs::read(&lock_path).unwrap(), prior_lock);

    write_manifest(&package_a, "package-a", "");
    build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("stale orphan is removed after complete local resolution");
    let replacement = std::fs::read_to_string(lock_path).unwrap();
    assert!(
        !replacement.contains("name = \"package-b\""),
        "{replacement}"
    );
}

#[test]
fn changed_locked_path_and_unreachable_registry_entry_do_not_block_resolution() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("root");
    let old_dependency = directory.path().join("old-dependency");
    let new_dependency = directory.path().join("new-dependency");
    write_manifest(&old_dependency, "dependency", "");
    write_manifest(
        &root,
        "root",
        "\n[dependencies]\ndependency = { path = \"../old-dependency\" }\n",
    );
    build_package_with_options(&root, &BuildOptions { auto_fetch: false }).unwrap();
    std::fs::rename(&old_dependency, &new_dependency).unwrap();
    write_manifest(
        &root,
        "root",
        "\n[dependencies]\ndependency = { path = \"../new-dependency\" }\n",
    );

    build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("a changed current path invalidates the unavailable locked path");
    let replaced = std::fs::read_to_string(root.join("reef.lock")).unwrap();
    assert!(
        replaced.contains("path = \"../new-dependency\""),
        "{replaced}"
    );

    write_manifest(&root, "root", "");
    std::fs::write(
        root.join("reef.lock"),
        format!(
            "schema = \"1\"\n\n[package]\nname = \"root\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"unused-registry\"\nversion = \"1.0.0\"\ncompiler = \"{ACTIVE_COMPILER_PIN}\"\narchive_sha256 = \"archive\"\nshell_sha256 = \"shell\"\n\n[dependencies.source]\nkind = \"local_registry\"\n"
        ),
    )
    .unwrap();
    build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("an unreachable lock entry must not require materialization");
    let replaced = std::fs::read_to_string(root.join("reef.lock")).unwrap();
    assert!(!replaced.contains("unused-registry"), "{replaced}");
}

fn registry_version(value: &str, archive: &str) -> RegistryVersion {
    RegistryVersion {
        version: value.to_string(),
        compiler: ACTIVE_COMPILER_PIN.to_string(),
        archive_sha256: archive.to_string(),
        shell_sha256: format!("shell-{archive}"),
        remote_origin: None,
    }
}

#[test]
fn local_index_parses_identities_sorts_numerically_and_rejects_conflicts() {
    let mut index = LocalRegistryIndex {
        packages: BTreeMap::from([(
            "dep".to_string(),
            vec![
                registry_version("1.19.0", "high"),
                registry_version("1.5.0", "low"),
            ],
        )]),
    };
    canonicalize_local_registry_index(&mut index).unwrap();
    assert_eq!(
        index.packages["dep"]
            .iter()
            .map(|entry| entry.version.as_str())
            .collect::<Vec<_>>(),
        vec!["1.5.0", "1.19.0"]
    );
    let wire = serde_json::to_value(&index).unwrap();
    assert_eq!(wire["packages"]["dep"][0]["version"], "1.5.0");
    assert!(wire["packages"]["dep"][0]["compiler"].is_string());
    assert!(wire["packages"]["dep"][0]["archive_sha256"].is_string());

    let mut invalid_name = LocalRegistryIndex {
        packages: BTreeMap::from([("Bad_Name".into(), vec![])]),
    };
    assert!(canonicalize_local_registry_index(&mut invalid_name).is_err());

    let mut invalid_version = LocalRegistryIndex {
        packages: BTreeMap::from([(
            "dep".into(),
            vec![registry_version("1.0.0+portable", "hash")],
        )]),
    };
    assert!(canonicalize_local_registry_index(&mut invalid_version).is_err());

    let mut too_many = LocalRegistryIndex {
        packages: BTreeMap::from([(
            "dep".into(),
            (0..257)
                .map(|patch| registry_version(&format!("1.0.{patch}"), "hash"))
                .collect(),
        )]),
    };
    assert!(
        canonicalize_local_registry_index(&mut too_many)
            .unwrap_err()
            .contains("limit 256")
    );

    let mut conflict = LocalRegistryIndex {
        packages: BTreeMap::from([(
            "dep".into(),
            vec![
                registry_version("1.0.0", "first"),
                registry_version("1.0.0", "second"),
            ],
        )]),
    };
    assert!(
        canonicalize_local_registry_index(&mut conflict)
            .unwrap_err()
            .contains("source conflict")
    );
}

#[test]
fn release_specs_reject_noncanonical_repositories_and_versions() {
    assert!(GitHubReleaseSpec::parse("Chelis-Lang/nautilus@v1.2.3").is_ok());
    for invalid in [
        "Chelis-Lang/Nautilus@v1.2.3",
        "../nautilus@v1.2.3",
        "Chelis-Lang/nautilus@latest",
        "Chelis-Lang/nautilus@v1.2.3+portable",
    ] {
        assert!(GitHubReleaseSpec::parse(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn external_format_locations_remain_exact_strings() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("format-demo");
    init_package(&root, "format-demo", "FormatDemo").unwrap();
    let artifacts = build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("build format fixture");

    let manifest: toml::Value =
        toml::from_str(&std::fs::read_to_string(root.join("reef.toml")).unwrap()).unwrap();
    assert_eq!(manifest["schema"].as_str(), Some("3"));
    assert_eq!(manifest["package"]["name"].as_str(), Some("format-demo"));
    assert_eq!(manifest["package"]["version"].as_str(), Some("0.1.0"));
    assert_eq!(manifest["package"]["resolver"].as_str(), Some("2"));

    let lock_path = root.join("reef.lock");
    let lock_bytes = std::fs::read(&lock_path).unwrap();
    #[cfg(unix)]
    let lock_inode = {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(&lock_path).unwrap().ino()
    };
    build_package_with_options(&root, &BuildOptions { auto_fetch: false })
        .expect("valid lock is the exact preferred graph");
    assert_eq!(std::fs::read(&lock_path).unwrap(), lock_bytes);
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(std::fs::metadata(&lock_path).unwrap().ino(), lock_inode);
    }
    package_schema(&root).expect("schema command reuses the exact lock");
    assert_eq!(std::fs::read(&lock_path).unwrap(), lock_bytes);
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(std::fs::metadata(&lock_path).unwrap().ino(), lock_inode);
    }

    let lock: toml::Value = toml::from_str(&std::fs::read_to_string(&lock_path).unwrap()).unwrap();
    assert_eq!(lock["schema"].as_str(), Some("1"));
    assert_eq!(lock["package"]["name"].as_str(), Some("format-demo"));
    assert_eq!(lock["package"]["version"].as_str(), Some("0.1.0"));
    assert_eq!(
        artifacts.shell_path.file_name().unwrap().to_str(),
        Some("format-demo-0.1.0.chb")
    );
    assert_eq!(
        artifacts.archive_path.file_name().unwrap().to_str(),
        Some("format-demo-0.1.0.tar.zst")
    );
    let shell = chelis_shell::read_shell(&artifacts.shell_path).unwrap();
    assert_eq!(shell.package.name, "format-demo");
    assert_eq!(shell.package.version, "0.1.0");
}

#[test]
fn bundled_lock_compiler_version_normalization_is_strict() {
    let bundled = chelis_reef::compiler_bundled_chelis_std_version();
    let compiler = env!("CARGO_PKG_VERSION");
    for source_compiler in [compiler.to_string(), format!("={compiler}")] {
        let directory = tempdir().unwrap();
        write_manifest(directory.path(), "root", "");
        std::fs::write(
            directory.path().join("reef.lock"),
            format!(
                "schema = \"1\"\n\n[package]\nname = \"root\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"chelis-std\"\nversion = \"{bundled}\"\ncompiler = \"{ACTIVE_COMPILER_PIN}\"\narchive_sha256 = \"{}\"\nshell_sha256 = \"{}\"\n\n[dependencies.source]\nkind = \"bundled\"\ncompiler_version = \"{source_compiler}\"\n",
                chelis_std_bundle::archive_sha256(),
                chelis_std_bundle::shell_sha256(),
            ),
        )
        .unwrap();
        build_package_with_options(directory.path(), &BuildOptions { auto_fetch: false })
            .expect("bare and single-equals bundled compiler versions are valid");
    }

    let directory = tempdir().unwrap();
    write_manifest(directory.path(), "root", "");
    std::fs::write(
        directory.path().join("reef.lock"),
        format!(
            "schema = \"1\"\n\n[package]\nname = \"root\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"chelis-std\"\nversion = \"{bundled}\"\ncompiler = \"{ACTIVE_COMPILER_PIN}\"\narchive_sha256 = \"archive\"\nshell_sha256 = \"shell\"\n\n[dependencies.source]\nkind = \"bundled\"\ncompiler_version = \"=={compiler}\"\n"
        ),
    )
    .unwrap();
    let error = build_package_with_options(directory.path(), &BuildOptions { auto_fetch: false })
        .unwrap_err();
    assert!(
        error.contains("invalid bundled compiler version"),
        "{error}"
    );

    let directory = tempdir().unwrap();
    write_manifest(directory.path(), "root", "");
    std::fs::write(
        directory.path().join("reef.lock"),
        format!(
            "schema = \"1\"\n\n[package]\nname = \"root\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"chelis-std\"\nversion = \"9.9.9\"\ncompiler = \"{ACTIVE_COMPILER_PIN}\"\narchive_sha256 = \"archive\"\nshell_sha256 = \"shell\"\n\n[dependencies.source]\nkind = \"bundled\"\ncompiler_version = \"9.9.9\"\n"
        ),
    )
    .unwrap();
    build_package_with_options(directory.path(), &BuildOptions { auto_fetch: false })
        .expect("a foreign bundled runtime lock is stale, not materialized");
    let repaired = std::fs::read_to_string(directory.path().join("reef.lock")).unwrap();
    assert!(
        repaired.contains(&format!("version = \"{bundled}\"")),
        "{repaired}"
    );
    assert!(!repaired.contains("version = \"9.9.9\""), "{repaired}");
}

#[test]
fn malformed_lock_identity_fails_at_the_lock_boundary() {
    let directory = tempdir().unwrap();
    write_manifest(directory.path(), "root", "");
    std::fs::write(
        directory.path().join("reef.lock"),
        "schema = \"1\"\ndependencies = []\n\n[package]\nname = \"Bad_Root\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    let error = package_schema(directory.path()).unwrap_err();

    assert!(error.contains("invalid package name"), "{error}");
}

#[test]
fn locked_hash_failure_is_hard_and_preserves_the_lock() {
    let directory = tempdir().unwrap();
    let root = directory.path().join("hash-root");
    write_manifest(&root, "hash-root", "");
    build_package_with_options(&root, &BuildOptions { auto_fetch: false }).unwrap();
    let lock_path = root.join("reef.lock");
    let lock = std::fs::read_to_string(&lock_path).unwrap();
    let archive_line = lock
        .lines()
        .find(|line| line.starts_with("archive_sha256 = "))
        .unwrap();
    let poisoned = lock.replacen(
        archive_line,
        &format!("archive_sha256 = \"{}\"", "0".repeat(64)),
        1,
    );
    std::fs::write(&lock_path, &poisoned).unwrap();

    let error = build_package_with_options(&root, &BuildOptions { auto_fetch: false }).unwrap_err();

    assert!(error.contains("locked archive hash mismatch"), "{error}");
    assert_eq!(std::fs::read_to_string(lock_path).unwrap(), poisoned);
}

#[test]
fn unavailable_unrecorded_locked_origin_is_a_hard_failure() {
    let _guard = ENV_LOCK.lock().unwrap();
    let directory = tempdir().unwrap();
    let root = directory.path().join("origin-root");
    write_manifest(
        &root,
        "origin-root",
        "\n[dependencies]\nmissing-origin = { version = \"1.0.0\" }\n",
    );
    std::fs::write(
        root.join("reef.lock"),
        format!(
            "schema = \"1\"\n\n[package]\nname = \"origin-root\"\nversion = \"0.1.0\"\n\n[[dependencies]]\nname = \"missing-origin\"\nversion = \"1.0.0\"\ncompiler = \"{ACTIVE_COMPILER_PIN}\"\narchive_sha256 = \"archive\"\nshell_sha256 = \"shell\"\n\n[dependencies.source]\nkind = \"local_registry\"\n"
        ),
    )
    .unwrap();
    let registry = directory.path().join("empty-registry");
    // SAFETY: this integration-test process serializes environment mutation
    // through ENV_LOCK and restores the prior value before it releases the lock.
    let prior = std::env::var_os("CHELIS_REEF_HOME");
    unsafe { std::env::set_var("CHELIS_REEF_HOME", &registry) };

    let result = package_schema(&root);

    match prior {
        Some(value) => unsafe { std::env::set_var("CHELIS_REEF_HOME", value) },
        None => unsafe { std::env::remove_var("CHELIS_REEF_HOME") },
    }
    let error = result.unwrap_err();
    assert!(error.contains("locked origin"), "{error}");
    assert!(error.contains("missing-origin"), "{error}");
}
