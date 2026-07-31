use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependencyClass {
    Normal,
    Development,
    Build,
}

#[derive(Debug, PartialEq, Eq)]
enum PurityViolation {
    Dependency(DependencyClass),
    MissingNoStd,
    StandardLibrary,
    Allocation,
    UnsafeAllowed,
    SourceRenderer,
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn manifest() -> String {
    fs::read_to_string(crate_root().join("Cargo.toml")).expect("read chelis-vocab Cargo.toml")
}

fn rust_sources() -> String {
    fn collect(path: &Path, output: &mut String) {
        let mut entries = fs::read_dir(path)
            .unwrap_or_else(|error| panic!("read source directory {}: {error}", path.display()))
            .collect::<Result<Vec<_>, _>>()
            .expect("read source entries");
        entries.sort_by_key(|entry| entry.path());

        for entry in entries {
            let path = entry.path();
            if path.is_dir() {
                collect(&path, output);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                output.push_str(&fs::read_to_string(&path).unwrap_or_else(|error| {
                    panic!("read Rust source {}: {error}", path.display())
                }));
                output.push('\n');
            }
        }
    }

    let mut output = String::new();
    collect(&crate_root().join("src"), &mut output);
    output
}

fn normalize_toml_key(key: &str) -> String {
    key.chars()
        .filter(|character| !matches!(character, '\'' | '"'))
        .collect::<String>()
        .trim()
        .to_owned()
}

fn dependency_class(section: &str) -> Option<(DependencyClass, bool)> {
    let candidates = [
        ("dependencies", DependencyClass::Normal),
        ("dev-dependencies", DependencyClass::Development),
        ("build-dependencies", DependencyClass::Build),
    ];

    for (name, class) in candidates {
        if section == name {
            return Some((class, false));
        }
        if section.starts_with(&format!("{name}.")) {
            return Some((class, true));
        }
        if section.starts_with("target.") {
            let marker = format!(".{name}");
            if let Some((_, suffix)) = section.rsplit_once(&marker) {
                if suffix.is_empty() {
                    return Some((class, false));
                }
                if suffix.starts_with('.') {
                    return Some((class, true));
                }
            }
        }
    }

    None
}

fn check_manifest(source: &str) -> Result<(), PurityViolation> {
    let mut current_dependency = None;
    let mut at_root = true;

    for raw_line in source.lines() {
        let line = raw_line.split('#').next().unwrap_or_default().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(section) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            at_root = false;
            let section = normalize_toml_key(section);
            current_dependency = dependency_class(&section);
            if let Some((class, true)) = current_dependency {
                return Err(PurityViolation::Dependency(class));
            }
            continue;
        }

        if at_root
            && let Some((key, _)) = line.split_once('=')
            && let Some((class, _)) = dependency_class(&normalize_toml_key(key))
        {
            return Err(PurityViolation::Dependency(class));
        }

        if let Some((class, false)) = current_dependency {
            return Err(PurityViolation::Dependency(class));
        }
    }

    Ok(())
}

fn check_sources(source: &str) -> Result<(), PurityViolation> {
    if !source.contains("#![no_std]") {
        return Err(PurityViolation::MissingNoStd);
    }
    if source.contains("extern crate std") || source.contains("std::") {
        return Err(PurityViolation::StandardLibrary);
    }
    if source.contains("extern crate alloc") || source.contains("alloc::") {
        return Err(PurityViolation::Allocation);
    }
    if !source.contains("#![forbid(unsafe_code)]") {
        return Err(PurityViolation::UnsafeAllowed);
    }
    if source.contains("fn render") || source.contains("CHELIS_RUNTIME_DTYPE_H") {
        return Err(PurityViolation::SourceRenderer);
    }
    Ok(())
}

#[test]
fn manifest_declares_no_dependencies() {
    assert_eq!(check_manifest(&manifest()), Ok(()));
}

#[test]
fn source_is_a_no_std_allocation_free_safe_vocabulary() {
    assert_eq!(check_sources(&rust_sources()), Ok(()));
}

#[test]
fn purity_checks_reject_each_guarded_mutation() {
    let clean_manifest = manifest();
    let clean_source = rust_sources();

    let dependency_mutations = [
        (
            "normal dependency",
            "\n[dependencies]\nserde = \"1\"\n",
            PurityViolation::Dependency(DependencyClass::Normal),
        ),
        (
            "development dependency",
            "\n[dev-dependencies]\nserde = \"1\"\n",
            PurityViolation::Dependency(DependencyClass::Development),
        ),
        (
            "build dependency",
            "\n[build-dependencies]\ncc = \"1\"\n",
            PurityViolation::Dependency(DependencyClass::Build),
        ),
        (
            "target dependency table",
            "\n[target.'cfg(unix)'.dependencies.serde]\nversion = \"1\"\n",
            PurityViolation::Dependency(DependencyClass::Normal),
        ),
        (
            "quoted dependency table",
            "\n[\"dependencies\"] # valid TOML comment\nserde = \"1\"\n",
            PurityViolation::Dependency(DependencyClass::Normal),
        ),
        (
            "quoted target dependency table",
            "\n[\"target\".'cfg(unix)'.\"dependencies\".serde]\nversion = \"1\"\n",
            PurityViolation::Dependency(DependencyClass::Normal),
        ),
    ];
    for (name, mutation, expected) in dependency_mutations {
        let mutated = format!("{clean_manifest}{mutation}");
        assert_eq!(check_manifest(&mutated), Err(expected), "{name}");
    }

    let inline_dependency = format!("dependencies = {{ serde = \"1\" }}\n{clean_manifest}");
    assert_eq!(
        check_manifest(&inline_dependency),
        Err(PurityViolation::Dependency(DependencyClass::Normal)),
        "root inline dependency table"
    );

    let source_mutations = [
        (
            "no_std removal",
            clean_source.replacen("#![no_std]", "", 1),
            PurityViolation::MissingNoStd,
        ),
        (
            "standard library opt-in",
            format!("{clean_source}\nextern crate std;\n"),
            PurityViolation::StandardLibrary,
        ),
        (
            "allocation opt-in",
            format!("{clean_source}\nextern crate alloc;\n"),
            PurityViolation::Allocation,
        ),
        (
            "unsafe prohibition removal",
            clean_source.replacen("#![forbid(unsafe_code)]", "", 1),
            PurityViolation::UnsafeAllowed,
        ),
        (
            "source renderer restoration",
            format!("{clean_source}\nfn render_runtime_dtype_c_header() {{}}\n"),
            PurityViolation::SourceRenderer,
        ),
    ];
    for (name, mutation, expected) in source_mutations {
        assert_eq!(check_sources(&mutation), Err(expected), "{name}");
    }
}
