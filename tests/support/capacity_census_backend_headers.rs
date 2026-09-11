//! Whole backend header census obligations. The final registry and manifest
//! activate with the device carrier cutover; this module grants no exceptions.

use super::*;

const BACKEND_BASELINE_REL: &str = "spec/design/capacity_census_backend_headers.json";
const HIP_HEADERS: &[&str] = &[
    "chelis_device_descriptor.h",
    "chelis_device_owner.h",
    "chelis_hip_runtime.h",
];
const METAL_HEADERS: &[&str] = &["chelis_metal_runtime.h"];

#[derive(Debug, Serialize, Deserialize)]
struct BackendBaseline {
    version: u32,
    rows: Vec<Row>,
}

fn copy_published_headers(source: &Path, destination: &Path) {
    for name in published_headers_on_disk(source) {
        let target = destination.join(&name);
        fs::create_dir_all(target.parent().expect("header parent")).unwrap();
        fs::copy(source.join(&name), target).unwrap_or_else(|error| {
            panic!(
                "copy published header {} into backend census staging: {error}",
                source.join(name).display()
            )
        });
    }
}

fn live_backend_rows(root: &Path) -> Vec<Row> {
    let runtime = root.join(INCLUDE_DIR_REL);
    let mut rows = Vec::new();

    let hip = planted_include_dir("backend-live-hip", &[]);
    copy_published_headers(&runtime, &hip);
    copy_published_headers(&root.join("crates/chelis-backend-hip/runtime"), &hip);
    let hip_sdk = [root.join("crates/chelis-backend-hip/tests/fixtures/device_entry_sdk")];
    let hip_environment = c_preprocessor::Environment {
        compiler: "clang",
        language: "c++",
        include_dirs: &hip_sdk,
        hermetic: true,
        ..c_preprocessor::Environment::native_c()
    };
    rows.extend(
        scan(
            &hip,
            &["chelis_hip_runtime.h", "chelis_blas.h", "chelis_math.h"],
            &[hip_environment],
        )
        .into_iter()
        .filter(|row| HIP_HEADERS.iter().any(|name| row.id.starts_with(name))),
    );
    fs::remove_dir_all(hip).unwrap();

    let metal = planted_include_dir("backend-live-metal", &[]);
    copy_published_headers(&runtime, &metal);
    copy_published_headers(&root.join("crates/chelis-backend-metal/runtime"), &metal);
    let metal_sdk = [root.join("tests/fixtures/capacity_backend_sdk")];
    let metal_environment = c_preprocessor::Environment {
        compiler: "clang",
        language: "objective-c++",
        include_dirs: &metal_sdk,
        hermetic: true,
        ..c_preprocessor::Environment::native_c()
    };
    rows.extend(
        scan(
            &metal,
            &["chelis_metal_runtime.h", "chelis_blas.h", "chelis_math.h"],
            &[metal_environment],
        )
        .into_iter()
        .filter(|row| METAL_HEADERS.iter().any(|name| row.id.starts_with(name))),
    );
    fs::remove_dir_all(metal).unwrap();

    rows.sort_by(|left, right| (&left.kind, &left.id).cmp(&(&right.kind, &right.id)));
    assert!(
        rows.windows(2)
            .all(|pair| (&pair[0].kind, &pair[0].id) != (&pair[1].kind, &pair[1].id)),
        "duplicate backend runtime census identity"
    );
    rows
}

fn scan(
    include_dir: &Path,
    roots: &[&str],
    environments: &[c_preprocessor::Environment<'_>],
) -> Vec<Row> {
    assert!(
        !environments.is_empty(),
        "missing backend preprocessing lane"
    );
    assert!(!roots.is_empty(), "missing backend header root");
    let mut expected = None;
    for environment in environments {
        let files = preprocessed_headers_with_environment(include_dir, roots, environment);
        let mut typedefs = BTreeMap::new();
        for text in files.values() {
            typedefs.append(&mut collect_typedefs(text));
        }
        let mut rows: Vec<Row> = files
            .iter()
            .flat_map(|(name, text)| header_rows(name, text, &typedefs))
            .collect();
        rows.sort_by(|left, right| (&left.kind, &left.id).cmp(&(&right.kind, &right.id)));
        assert!(
            rows.windows(2)
                .all(|pair| (&pair[0].kind, &pair[0].id) != (&pair[1].kind, &pair[1].id)),
            "duplicate backend declaration identity"
        );
        if let Some(previous) = &expected {
            assert_eq!(
                previous, &rows,
                "CONTEXT-VARYING PUBLIC ABI between backend preprocessing lanes"
            );
        } else {
            expected = Some(rows);
        }
    }
    expected.expect("at least one explicit backend lane executed")
}

#[test]
fn nested_generated_headers_have_the_same_raw_and_preprocessed_identity() {
    let dir = planted_include_dir(
        "backend-relative-closure",
        &[
            ("root.h", "#include \"detail/entry.h\"\n"),
            (
                "detail/entry.h",
                "# include \"packet.h\"\n#include \"../common.h\"\n",
            ),
            (
                "detail/packet.h",
                "typedef struct { long long count; } chelis_packet;\n",
            ),
            ("common.h", "long long chelis_count(void);\n"),
        ],
    );
    let rows = scan(
        &dir,
        &["root.h"],
        &[c_preprocessor::Environment::native_c()],
    );
    fs::remove_dir_all(&dir).unwrap();
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert!(
        rows.iter()
            .any(|row| row.id.starts_with("detail/packet.h:"))
    );
    assert!(rows.iter().any(|row| row.id.starts_with("common.h:")));
}

#[test]
fn nested_includes_cannot_hide_conditional_abi_or_macro_taint() {
    for (label, root, entry, packet) in [
        (
            "conditional-grandchild",
            "#ifdef CHELIS_FEATURE\n#include \"detail/entry.h\"\n#endif\n",
            "#include \"packet.h\"\n",
            "long long chelis_hidden(void);\n",
        ),
        (
            "nested-taint",
            "#include \"detail/entry.h\"\nCHELIS_NUMBER chelis_hidden(void);\n",
            "#include \"packet.h\"\n",
            "#ifdef CHELIS_FEATURE\n#define CHELIS_NUMBER double\n#else\n#define CHELIS_NUMBER int\n#endif\n",
        ),
    ] {
        let dir = planted_include_dir(
            &format!("backend-{label}"),
            &[
                ("root.h", root),
                ("detail/entry.h", entry),
                ("detail/packet.h", packet),
            ],
        );
        let error = expect_census_panic(|| {
            scan(
                &dir,
                &["root.h"],
                &[c_preprocessor::Environment::native_c()],
            );
        });
        fs::remove_dir_all(&dir).unwrap();
        assert!(
            error.contains("CONTEXT-VARYING PUBLIC ABI"),
            "{label}: {error}"
        );
    }
}

#[test]
fn complete_backend_closure_keeps_macro_fields_and_external_declarations() {
    let dir = planted_include_dir(
        "backend-closure",
        &[
            ("root.h", "#include \"generated.h\"\n#include <helpers.h>\n"),
            (
                "generated.h",
                "#define CHELIS_COUNT long long\ntypedef struct { CHELIS_COUNT count; } chelis_device_packet;\n",
            ),
            (
                "helpers.h",
                "long long chelis_extent(void);\nstatic inline float helper(float x) { return x; }\n",
            ),
        ],
    );
    let rows = scan(
        &dir,
        &["root.h"],
        &[c_preprocessor::Environment::native_c()],
    );
    fs::remove_dir_all(&dir).unwrap();
    assert_eq!(rows.len(), 2, "{rows:?}");
    assert!(rows.iter().any(|row| row.kind == "header-struct"
        && row.id.starts_with("generated.h:")
        && row.id.contains("long long count")));
    assert!(rows.iter().any(|row| row.kind == "header-export"
        && row.id.contains("chelis_extent")
        && row.flags.contains(&"numeric-op".into())));
}

#[test]
fn backend_roots_cannot_omit_an_unreached_generated_header() {
    let dir = planted_include_dir(
        "backend-unreached",
        &[
            ("root.h", "int chelis_present(void);\n"),
            ("generated.h", "extern double chelis_hidden;\n"),
        ],
    );
    let error = expect_census_panic(|| {
        scan(
            &dir,
            &["root.h"],
            &[c_preprocessor::Environment::native_c()],
        );
    });
    fs::remove_dir_all(&dir).unwrap();
    assert!(error.contains("PUBLISHED HEADER NOT REACHED"), "{error}");
}

#[test]
fn backend_environments_cannot_change_a_public_numeric_signature() {
    let dir = planted_include_dir(
        "backend-context",
        &[("root.h", "CHELIS_SDK_NUMBER chelis_value(void);\n")],
    );
    let error = expect_census_panic(|| {
        scan(
            &dir,
            &["root.h"],
            &[
                c_preprocessor::Environment {
                    arguments: &["-DCHELIS_SDK_NUMBER=int"],
                    ..c_preprocessor::Environment::native_c()
                },
                c_preprocessor::Environment {
                    arguments: &["-DCHELIS_SDK_NUMBER=double"],
                    ..c_preprocessor::Environment::native_c()
                },
            ],
        );
    });
    fs::remove_dir_all(&dir).unwrap();
    assert!(error.contains("CONTEXT-VARYING PUBLIC ABI"), "{error}");
}

#[test]
fn backend_declarations_retain_primary_fail_closed_guards() {
    for (label, text, expected) in [
        (
            "line",
            "#line 1 \"/external/hidden.h\"\nint chelis_hidden(void);\n",
            "LINE-DIRECTIVE",
        ),
        (
            "conditional",
            "#ifdef NDEBUG\nint chelis_hidden(void);\n#endif\n",
            "CONTEXT-VARYING",
        ),
        (
            "unknown",
            "_Float16 chelis_hidden(void);\n",
            "UNKNOWN TYPE WORD",
        ),
    ] {
        let dir = planted_include_dir(&format!("backend-{label}"), &[("root.h", text)]);
        let error = expect_census_panic(|| {
            scan(
                &dir,
                &["root.h"],
                &[c_preprocessor::Environment::native_c()],
            );
        });
        fs::remove_dir_all(&dir).unwrap();
        assert!(error.contains(expected), "{label}: {error}");
    }
}

#[test]
fn backend_scan_requires_an_actual_preprocessing_lane() {
    let dir = planted_include_dir(
        "backend-no-lane",
        &[("root.h", "int chelis_value(void);\n")],
    );
    let error = expect_census_panic(|| {
        scan(&dir, &["root.h"], &[]);
    });
    fs::remove_dir_all(&dir).unwrap();
    assert!(
        error.contains("missing backend preprocessing lane"),
        "{error}"
    );
}

#[test]
fn backend_runtime_headers_match_the_reviewed_final_authority() {
    let root = repo_root();
    let baseline_path = root.join(BACKEND_BASELINE_REL);
    let current = live_backend_rows(&root);
    if std::env::var("CHELIS_CAPACITY_CENSUS_WRITE").as_deref() == Ok("1") {
        fs::write(
            &baseline_path,
            serde_json::to_string_pretty(&BackendBaseline {
                version: 1,
                rows: current.clone(),
            })
            .unwrap()
                + "\n",
        )
        .unwrap();
    }
    let baseline: BackendBaseline =
        serde_json::from_str(&fs::read_to_string(&baseline_path).unwrap_or_else(|error| {
            panic!(
                "Phase 2 requires a reviewed live backend-header baseline at {}: {error}",
                baseline_path.display()
            )
        }))
        .expect("parse backend-header census baseline");
    assert_eq!(
        baseline.version, 1,
        "unknown backend-header baseline version"
    );
    assert_eq!(
        current, baseline.rows,
        "backend runtime headers differ from the reviewed exact census"
    );
    let spec = fs::read_to_string(root.join(CONTROLLING_SPEC_REL)).unwrap();
    for row in &baseline.rows {
        assert!(
            row.citation.is_empty(),
            "backend final authority cannot retain a transition citation: {row:?}"
        );
        capacity_census_authority::classify_final_authority(
            &authority_surface(row),
            backend_authority_registries(),
            &spec,
        )
        .unwrap_or_else(|problem| panic!("unclassified backend runtime row {row:?}: {problem}"));
    }
}

#[test]
fn generated_device_descriptor_requires_exact_tagged_transport_authority() {
    let root = repo_root();
    let row = live_backend_rows(&root)
        .into_iter()
        .find(|row| row.id.starts_with("chelis_device_descriptor.h:"))
        .expect("generated device descriptor must be part of the live backend census");
    let spec = fs::read_to_string(root.join(CONTROLLING_SPEC_REL)).unwrap();
    assert_eq!(
        capacity_census_authority::classify_final_authority(
            &authority_surface(&row),
            backend_authority_registries(),
            &spec,
        ),
        Ok(capacity_census_authority::FinalAuthority::TaggedTransport)
    );
    let mut changed = authority_surface(&row);
    changed.id = changed.id.replace("int64_t count", "int32_t count");
    assert!(
        capacity_census_authority::classify_final_authority(
            &changed,
            backend_authority_registries(),
            &spec,
        )
        .is_err(),
        "a narrowed generated packet must not inherit tagged-transport authority"
    );
}

#[test]
fn device_owner_callables_require_exact_op33_authority() {
    let root = repo_root();
    let include_dir = planted_include_dir("device-owner-authority", &[]);
    let runtime = root.join(INCLUDE_DIR_REL);
    for name in published_headers_on_disk(&runtime) {
        let destination = include_dir.join(&name);
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::copy(runtime.join(name), destination).unwrap();
    }
    for name in ["chelis_device_owner.h", "chelis_device_descriptor.h"] {
        fs::copy(
            root.join("crates/chelis-backend-hip/runtime").join(name),
            include_dir.join(name),
        )
        .expect("the published device header must exist");
    }
    let mut roots = HEADER_ROOTS.to_vec();
    roots.push("chelis_device_owner.h");
    let rows: Vec<_> = scan(
        &include_dir,
        &roots,
        &[c_preprocessor::Environment::native_c()],
    )
    .into_iter()
    .filter(|row| row.kind == "header-export" && row.id.starts_with("chelis_device_owner.h:"))
    .collect();
    fs::remove_dir_all(include_dir).unwrap();
    assert_eq!(rows.len(), 9, "complete opaque device owner API: {rows:?}");
    let spec = fs::read_to_string(root.join(CONTROLLING_SPEC_REL)).unwrap();
    let registry = fs::read_to_string(root.join("spec/registry/c_tensor_runtime.md")).unwrap();
    let normative: BTreeSet<_> = registry
        .lines()
        .filter(|line| line.contains("chelis_device_tensor_"))
        .map(|line| {
            let signature = line.split('`').nth(1).expect("exact normative C signature");
            format!(
                "chelis_device_owner.h: {}",
                canonical_c_tokens(&format!("{signature};"))
            )
        })
        .collect();
    assert_eq!(
        rows.iter()
            .map(|row| row.id.clone())
            .collect::<BTreeSet<_>>(),
        normative
    );
    for row in rows {
        let surface = authority_surface(&row);
        assert_eq!(
            capacity_census_authority::classify_final_authority(
                &surface,
                backend_authority_registries(),
                &spec,
            ),
            Ok(capacity_census_authority::FinalAuthority::NumericOperation { atom: "[05-OP-33]" }),
        );
        let mut successor = surface;
        successor.id = successor
            .id
            .replace("chelis_device_tensor_", "chelis_unchecked_device_");
        assert!(
            capacity_census_authority::classify_final_authority(
                &successor,
                backend_authority_registries(),
                &spec,
            )
            .is_err(),
            "an unregistered renamed device operation must fail"
        );
    }
}
