#[path = "support/boundary.rs"]
mod boundary;

use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

fn core() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn cargo() -> std::ffi::OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into())
}
fn successful(output: Output) -> Output {
    assert!(
        output.status.success(),
        "command failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
fn probe(root: &Path, source: &str) {
    fs::create_dir_all(root.join("src")).unwrap();
    let core_path = toml::Value::String(core().display().to_string());
    fs::write(root.join("Cargo.toml"), format!(
        "[package]\nname='identity-boundary-probe'\nversion='0.0.0'\nedition='2021'\n[workspace]\n[dependencies]\nchelis-runtime-identity={{path={core_path}}}\n"
    )).unwrap();
    fs::write(root.join("src/lib.rs"), source).unwrap();
}

#[test]
fn declared_and_resolved_production_dependencies_stay_inward() {
    boundary::declared(&fs::read_to_string(core().join("Cargo.toml")).unwrap()).unwrap();
    let temp = tempfile::tempdir().unwrap();
    probe(temp.path(), "");
    // Isolated downstream root excludes this suite's object/write and syntax-parser
    // dev features. Metadata is used only for architecture, never runtime identity.
    for target in ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"] {
        for mode in ["minimal", "default", "all"] {
            let path = temp.path().join("Cargo.toml");
            let mut manifest: toml::Value =
                toml::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
            let dependency = manifest["dependencies"]["chelis-runtime-identity"]
                .as_table_mut()
                .unwrap();
            dependency.insert("default-features".into(), (mode != "minimal").into());
            let production: toml::Value =
                toml::from_str(&fs::read_to_string(core().join("Cargo.toml")).unwrap()).unwrap();
            let features = if mode == "all" {
                production
                    .get("features")
                    .and_then(toml::Value::as_table)
                    .map(|table| {
                        table
                            .keys()
                            .map(|name| toml::Value::String(name.clone()))
                            .collect()
                    })
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            dependency.insert("features".into(), toml::Value::Array(features));
            fs::write(path, toml::to_string(&manifest).unwrap()).unwrap();
            let output = successful(
                Command::new(cargo())
                    .current_dir(temp.path())
                    .args([
                        "metadata",
                        "--format-version=1",
                        "--offline",
                        "--filter-platform",
                        target,
                    ])
                    .output()
                    .unwrap(),
            );
            boundary::resolved(&serde_json::from_slice(&output.stdout).unwrap()).unwrap();
        }
    }
}

fn graph(edges: &[(&str, &[&str])]) -> Value {
    json!({
        "packages": edges.iter().map(|(name, _)| json!({"id":name,"name":name})).collect::<Vec<_>>(),
        "resolve": {"nodes": edges.iter().map(|(name, deps)| json!({
            "id":name,"features":[],"deps":deps.iter().map(|dep| json!({"pkg":dep,"dep_kinds":[{"kind":null}]})).collect::<Vec<_>>()
        })).collect::<Vec<_>>()}
    })
}

#[test]
fn dependency_guard_rejects_alias_optional_target_and_transitive_escapes() {
    boundary::declared("[dependencies]\nserde='1'\n[dev-dependencies]\ntokio='1'\n").unwrap();
    for table in [
        "dependencies",
        "build-dependencies",
        "target.'cfg(unix)'.dependencies",
        "target.'cfg(windows)'.build-dependencies",
    ] {
        let manifest = format!(
            "[{table}]\ninnocent={{package='chelis-runtime-identity-build',version='1',optional=true}}\n"
        );
        assert!(
            boundary::declared(&manifest)
                .unwrap_err()
                .contains("chelis-runtime-identity-build")
        );
    }
    let allowed = graph(&[
        ("chelis-runtime-identity", &["sha2"]),
        ("sha2", &["digest"]),
        ("digest", &[]),
    ]);
    boundary::resolved(&allowed).unwrap();
    for denied in [
        "chelis-runtime-identity-build",
        "chelis-cli",
        "unknown-parser",
    ] {
        for direct in [true, false] {
            let value = if direct {
                graph(&[("chelis-runtime-identity", &[denied]), (denied, &[])])
            } else {
                graph(&[
                    ("chelis-runtime-identity", &["sha2"]),
                    ("sha2", &[denied]),
                    (denied, &[]),
                ])
            };
            assert!(boundary::resolved(&value).unwrap_err().contains(denied));
        }
    }
    let mut write_enabled = graph(&[("chelis-runtime-identity", &["object"]), ("object", &[])]);
    write_enabled["resolve"]["nodes"][1]["features"] = json!(["write"]);
    assert!(
        boundary::resolved(&write_enabled)
            .unwrap_err()
            .contains("write")
    );
}

fn check_sources(path: &Path) {
    let text = fs::read_to_string(path).unwrap();
    boundary::source(&text).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    fn modules(items: &[syn::Item], directory: &Path) {
        for item in items {
            let syn::Item::Mod(module) = item else {
                continue;
            };
            if module.attrs.iter().any(|attribute| attribute.path().is_ident("cfg")
                && matches!(attribute.parse_args::<syn::Meta>(), Ok(syn::Meta::Path(path)) if path.is_ident("test"))) { continue; }
            let name = module.ident.to_string();
            if let Some((_, items)) = &module.content {
                modules(items, &directory.join(name));
            } else {
                let explicit = module.attrs.iter().find_map(|attribute| {
                    if !attribute.path().is_ident("path") {
                        return None;
                    }
                    let syn::Meta::NameValue(meta) = &attribute.meta else {
                        panic!("invalid path attribute");
                    };
                    let syn::Expr::Lit(literal) = &meta.value else {
                        panic!("invalid module path");
                    };
                    let syn::Lit::Str(value) = &literal.lit else {
                        panic!("invalid module path");
                    };
                    Some(directory.join(value.value()))
                });
                let file = explicit.unwrap_or_else(|| {
                    let sibling = directory.join(format!("{name}.rs"));
                    if sibling.exists() {
                        sibling
                    } else {
                        directory.join(name).join("mod.rs")
                    }
                });
                check_sources(&file);
            }
        }
    }
    let directory = if matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("lib.rs" | "mod.rs")
    ) {
        path.parent().unwrap().to_owned()
    } else {
        path.with_extension("")
    };
    modules(&syn::parse_file(&text).unwrap().items, &directory);
}

#[test]
fn all_owned_core_source_is_data_only() {
    check_sources(&core().join("src/lib.rs"));
}

#[test]
fn syntax_guard_follows_aliases_and_local_helpers_without_text_false_positives() {
    for source in [
        "fn f() { let _ = std::fs::read(\"input\"); }",
        "use std::fs as innocent; fn f() { let _ = innocent::read(\"input\"); }",
        "use std::{fs::{read as capture}}; fn f() { let _ = capture(\"input\"); }",
        "fn helper() { let _ = std::env::var(\"INPUT\"); } pub fn f() { helper(); }",
        "use std as platform; use platform::process::Command as Data; fn f() { Data::new(\"x\"); }",
        "pub fn f<R: std::io::Read>(input: R) {}",
        "pub fn f(provider: impl Fn() -> Vec<u8>) {}",
        "pub fn f(provider: fn() -> Vec<u8>) {}",
        "pub fn f() { println!(\"side effect\"); }",
        "pub fn f() { let _ = vec![std::fs::read(\"input\")]; }",
        "pub fn f() { let _ = matches!(1, _ if std::env::var(\"X\").is_ok()); }",
        "static STATE: u32 = 1;",
    ] {
        assert!(
            boundary::source(source).is_err(),
            "effect escaped: {source}"
        );
    }
    boundary::source(
        r#"
        use std::collections::BTreeMap as Ordered;
        // std::fs::read and std::env::var are not calls here.
        fn helper(bytes: &[u8]) -> usize { bytes.len() }
        pub fn f(bytes: &[u8]) -> usize {
            let note = "std::process::Command";
            let data = Ordered::<String, usize>::new();
            helper(bytes) + note.len() + data.len()
        }
    "#,
    )
    .unwrap();
}

#[test]
fn data_ports_accept_values_and_reject_effect_providers_at_compile_time() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let positive = include_str!("../fixtures/ports/data.rs");
    probe(root, positive);
    let check = || {
        Command::new(cargo())
            .current_dir(root)
            .args([
                "check",
                "--offline",
                "--message-format=json",
                "--target-dir",
            ])
            .arg(root.join("target"))
            .env_remove("RUSTC_WRAPPER")
            .env_remove("RUSTC_WORKSPACE_WRAPPER")
            .output()
            .unwrap()
    };
    successful(check());
    for (name, source) in [
        ("file", include_str!("../fixtures/ports/file.rs")),
        ("reader", include_str!("../fixtures/ports/reader.rs")),
        ("callback", include_str!("../fixtures/ports/callback.rs")),
        ("provider", include_str!("../fixtures/ports/provider.rs")),
    ] {
        fs::write(root.join("src/lib.rs"), source).unwrap();
        let output = check();
        assert!(
            !output.status.success(),
            "{name} unexpectedly crossed the value-only boundary"
        );
        let diagnostics: Vec<Value> = String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .filter(|message: &Value| {
                message["reason"] == "compiler-message" && message["message"]["level"] == "error"
            })
            .collect();
        assert!(
            !diagnostics.is_empty(),
            "{name}: no compiler diagnostics: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            diagnostics
                .iter()
                .all(|message| message["message"]["code"]["code"] == "E0308"),
            "{name}: failed for an unrelated reason: {diagnostics:#?}"
        );
    }
    fs::write(root.join("src/lib.rs"), positive).unwrap();
    successful(check());
}

#[test]
fn captured_values_and_typed_failures_ignore_ambient_process_state() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    probe(root, "");
    fs::write(
        root.join("src/main.rs"),
        include_str!("../fixtures/ports/ambient.rs"),
    )
    .unwrap();
    let output = successful(
        Command::new(cargo())
            .current_dir(root)
            .args([
                "build",
                "--offline",
                "--message-format=json",
                "--target-dir",
            ])
            .arg(root.join("target"))
            .env_remove("RUSTC_WRAPPER")
            .env_remove("RUSTC_WORKSPACE_WRAPPER")
            .output()
            .unwrap(),
    );
    let executables: Vec<PathBuf> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|message| {
            message["reason"] == "compiler-artifact"
                && message["target"]["name"] == "identity-boundary-probe"
        })
        .filter_map(|message| message["executable"].as_str().map(PathBuf::from))
        .collect();
    assert_eq!(executables.len(), 1);
    let left = root.join("ambient-left");
    let right = root.join("ambient-right");
    fs::create_dir_all(&left).unwrap();
    fs::create_dir_all(&right).unwrap();
    fs::write(left.join("Cargo.toml"), "left ambient source").unwrap();
    fs::write(right.join("Cargo.toml"), "different ambient source").unwrap();
    let run = |directory: &Path, value: &str| {
        successful(
            Command::new(&executables[0])
                .current_dir(directory)
                .env("CARGO_MANIFEST_DIR", directory)
                .env("OUT_DIR", directory)
                .env("PROFILE", value)
                .env("RUSTFLAGS", value)
                .env("CHELIS_IDENTITY_WORKSPACE", directory)
                .env("CHELIS_IDENTITY_PROVENANCE", value)
                .output()
                .unwrap(),
        )
        .stdout
    };
    assert_eq!(run(&left, "left"), run(&right, "right"));
}
