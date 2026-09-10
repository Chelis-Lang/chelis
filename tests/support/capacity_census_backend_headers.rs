//! Whole backend header census obligations. The final registry and manifest
//! activate with the device carrier cutover; this module grants no exceptions.

use super::*;

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
