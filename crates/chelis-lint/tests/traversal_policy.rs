use chelis_lint::policy::{TraversalClass, TraversalPolicy};
use chelis_lint::{
    Context, Rule, Surface,
    rules::{
        doc_filename_convention::DocFilenameConvention,
        opaque_domain_construction::OpaqueDomainConstruction,
    },
};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::tempdir;

fn write_spec(root: &Path) {
    let spec = root.join("spec/01-nomenclature.md");
    fs::create_dir_all(spec.parent().unwrap()).unwrap();
    fs::write(
        spec,
        "# Nomenclature\n\n### 12.2 Lint Traversal Exclusions\n",
    )
    .unwrap();
}

fn write_policy(root: &Path, entries: &str) {
    write_spec(root);
    fs::write(
        root.join("chelis-lint.toml"),
        format!("version = 1\nspec = \"spec/01-nomenclature.md\"\n\n{entries}"),
    )
    .unwrap();
}

fn exclusion(pattern: &str, class: &str, cross_ref: &str) -> String {
    format!(
        "[[exclude]]\npattern = \"{pattern}\"\nclass = \"{class}\"\ncross_ref = \"{cross_ref}\"\n"
    )
}

fn rust_sources_below(directory: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        let metadata = fs::symlink_metadata(&path).unwrap();
        if metadata.is_dir() {
            rust_sources_below(&path, out);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

fn walked(root: &Path) -> Vec<String> {
    chelis_lint::walker::walk(root)
        .expect("walk")
        .into_iter()
        .map(|entry| entry.expect("entry"))
        .filter_map(|entry| {
            entry
                .path
                .strip_prefix(root)
                .ok()
                .map(|path| path.to_string_lossy().replace('\\', "/"))
        })
        .collect()
}

#[test]
fn valid_repository_policy_is_discovered_anchored_and_explainable() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_policy(root, &exclusion("generated/", "generated", "§12.2"));
    fs::create_dir_all(root.join("generated")).unwrap();
    fs::write(root.join("generated/hidden.ch"), "def hidden() = 1\n").unwrap();
    fs::write(root.join("visible.ch"), "def visible() = 1\n").unwrap();

    let policy = TraversalPolicy::load_for(&root.join("generated/hidden.ch")).unwrap();
    assert_eq!(policy.repository_root(), Some(root));
    let matched = policy
        .exclusion_for(&root.join("generated"), true)
        .expect("generated directory is explained");
    assert_eq!(matched.pattern, "generated/");
    assert_eq!(matched.class, TraversalClass::Generated);
    assert_eq!(matched.cross_ref, "§12.2");

    let names = walked(root);
    assert!(names.iter().any(|path| path == "visible.ch"));
    assert!(!names.iter().any(|path| path == "generated/hidden.ch"));
}

#[test]
fn nearest_policy_is_used_and_patterns_stay_anchored_to_its_root() {
    let temp = tempdir().unwrap();
    let outer = temp.path();
    write_policy(outer, &exclusion("outer-only/", "generated", "§12.2"));
    let inner = outer.join("workspace");
    fs::create_dir_all(&inner).unwrap();
    write_policy(&inner, &exclusion("/generated/", "generated", "§12.2"));
    fs::create_dir_all(inner.join("src/generated")).unwrap();
    fs::create_dir_all(inner.join("generated")).unwrap();
    fs::write(inner.join("src/generated/keep.ch"), "def keep() = 1\n").unwrap();
    fs::write(inner.join("generated/drop.ch"), "def drop() = 1\n").unwrap();

    let policy = TraversalPolicy::load_for(&inner.join("src")).unwrap();
    assert_eq!(policy.repository_root(), Some(inner.as_path()));
    let names = walked(&inner.join("src"));
    assert!(names.iter().any(|path| path == "generated/keep.ch"));
    assert!(
        !walked(&inner)
            .iter()
            .any(|path| path == "generated/drop.ch")
    );
}

#[test]
fn relative_subdirectory_target_uses_workspace_anchoring() {
    let temp = tempdir().unwrap();
    let root = temp.path().join("workspace");
    fs::create_dir_all(root.join("src/generated")).unwrap();
    write_policy(&root, &exclusion("/src/generated/", "generated", "§12.2"));
    fs::write(root.join("src/keep.ch"), "def keep() = 1\n").unwrap();
    fs::write(root.join("src/generated/drop.ch"), "def drop() = 1\n").unwrap();

    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "relative_subdirectory_child_probe",
            "--nocapture",
        ])
        .current_dir(&root)
        .env("CHELIS_LINT_RELATIVE_PROBE", "1")
        .status()
        .unwrap();
    assert!(status.success(), "relative-target child probe failed");
}

#[test]
fn relative_subdirectory_child_probe() {
    if std::env::var_os("CHELIS_LINT_RELATIVE_PROBE").is_none() {
        return;
    }
    let names = walked(Path::new("src"));
    assert!(names.iter().any(|path| path == "keep.ch"));
    assert!(!names.iter().any(|path| path == "generated/drop.ch"));

    let policy = TraversalPolicy::load_for(Path::new("src/keep.ch")).unwrap();
    assert!(policy.is_excluded_or_parent(Path::new("src/generated/Cargo.toml"), false));
    assert!(!policy.is_excluded_or_parent(Path::new("src/Cargo.toml"), false));
}

#[test]
fn malformed_policies_fail_loudly() {
    let cases = [
        (
            "version = 2\nspec = \"spec/01-nomenclature.md\"\n",
            "unsupported",
        ),
        (
            "version = 1\nspec = \"spec/01-nomenclature.md\"\nunknown = true\n",
            "unknown",
        ),
        ("version = 1\n", "`spec`"),
        (
            "version = 1\nspec = \"spec/01-nomenclature.md\"\n[[exclude]]\npattern = \"x/\"\nclass = \"generated\"\n",
            "cross_ref",
        ),
        (
            "version = 1\nspec = \"spec/01-nomenclature.md\"\n[[exclude]]\npattern = \"x/\"\nclass = \"temporary\"\ncross_ref = \"§12.2\"\n",
            "temporary",
        ),
        (
            "version = 1\nspec = \"missing.md\"\n[[exclude]]\npattern = \"x/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
            "missing.md",
        ),
        (
            "version = 1\nspec = \"../outside.md\"\n",
            "outside its policy root",
        ),
        (
            "version = 1\nspec = \"/tmp/outside.md\"\n",
            "outside its policy root",
        ),
        (
            "version = 1\nspec = \"spec/01-nomenclature.md\"\n[[exclude]]\npattern = \"[\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
            "pattern",
        ),
        (
            "version = 1\nspec = \"spec/01-nomenclature.md\"\n[[exclude]]\npattern = \"!generated/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
            "not comments or negations",
        ),
        (
            "version = 1\nspec = \"spec/01-nomenclature.md\"\n[[exclude]]\npattern = \"# generated/\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
            "not comments or negations",
        ),
        (
            "version = 1\nspec = \"spec/01-nomenclature.md\"\n[[exclude]]\npattern = \"\"\nclass = \"generated\"\ncross_ref = \"§12.2\"\n",
            "non-empty",
        ),
        (
            "version = 1\nspec = \"spec/01-nomenclature.md\"\n[[exclude]]\npattern = \"x/\"\nclass = \"generated\"\ncross_ref = \"§99\"\n",
            "§99",
        ),
    ];

    for (policy, expected) in cases {
        let temp = tempdir().unwrap();
        write_spec(temp.path());
        fs::write(temp.path().join("chelis-lint.toml"), policy).unwrap();
        let error = TraversalPolicy::load_for(temp.path()).unwrap_err();
        assert!(
            error.to_string().contains(expected),
            "expected {expected:?} in {error}"
        );
    }
}

#[cfg(unix)]
#[test]
fn external_policy_symlink_fails_closed() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let external = tempdir().unwrap();
    write_spec(repository.path());
    let external_policy = external.path().join("policy.toml");
    fs::write(
        &external_policy,
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    let policy_path = repository.path().join("chelis-lint.toml");
    symlink(&external_policy, &policy_path).unwrap();

    let error = TraversalPolicy::load_for(repository.path()).unwrap_err();
    assert!(error.to_string().contains("outside its policy root"));
    assert!(error.to_string().contains("chelis-lint.toml"));
}

#[cfg(unix)]
#[test]
fn internal_policy_symlink_is_allowed() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    write_spec(repository.path());
    fs::create_dir_all(repository.path().join("config")).unwrap();
    fs::write(
        repository.path().join("config/policy.toml"),
        "version = 1\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    symlink(
        "config/policy.toml",
        repository.path().join("chelis-lint.toml"),
    )
    .unwrap();

    TraversalPolicy::load_for(repository.path()).unwrap();
}

#[cfg(unix)]
#[test]
fn broken_policy_symlink_fails_closed() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let policy_path = repository.path().join("chelis-lint.toml");
    symlink(repository.path().join("missing-policy.toml"), &policy_path).unwrap();

    let error = TraversalPolicy::load_for(repository.path()).unwrap_err();
    assert!(error.to_string().contains("chelis-lint.toml"));
    assert!(
        error
            .to_string()
            .contains("failed to read traversal policy input")
    );
}

#[test]
fn repository_policy_directory_fails_closed() {
    let repository = tempdir().unwrap();
    let policy_path = repository.path().join("chelis-lint.toml");
    fs::create_dir(&policy_path).unwrap();

    let error = TraversalPolicy::load_for(repository.path()).unwrap_err();
    assert!(error.to_string().contains("chelis-lint.toml"));
    assert!(error.to_string().contains("must be a regular file"));
}

#[cfg(unix)]
#[test]
fn external_spec_symlink_fails_closed_but_internal_spec_symlink_is_allowed() {
    use std::os::unix::fs::symlink;

    let external = tempdir().unwrap();
    let external_spec = external.path().join("lint.md");
    fs::write(&external_spec, "# Lint\n\n### 12.2 Traversal exclusions\n").unwrap();

    let escaped_repository = tempdir().unwrap();
    fs::create_dir_all(escaped_repository.path().join("spec")).unwrap();
    fs::write(
        escaped_repository.path().join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/lint.md\"\n",
    )
    .unwrap();
    symlink(
        &external_spec,
        escaped_repository.path().join("spec/lint.md"),
    )
    .unwrap();
    let error = TraversalPolicy::load_for(escaped_repository.path()).unwrap_err();
    assert!(error.to_string().contains("outside its policy root"));
    assert!(error.to_string().contains("spec/lint.md"));

    let internal_repository = tempdir().unwrap();
    fs::create_dir_all(internal_repository.path().join("spec")).unwrap();
    fs::write(
        internal_repository.path().join("spec/actual.md"),
        "# Lint\n\n### 12.2 Traversal exclusions\n",
    )
    .unwrap();
    symlink("actual.md", internal_repository.path().join("spec/lint.md")).unwrap();
    fs::write(
        internal_repository.path().join("chelis-lint.toml"),
        "version = 1\nspec = \"spec/lint.md\"\n",
    )
    .unwrap();
    TraversalPolicy::load_for(internal_repository.path()).unwrap();
}

#[test]
fn git_and_hidden_ignore_sources_do_not_affect_lint_entries() {
    let temp = tempdir().unwrap();
    let parent = temp.path();
    fs::write(parent.join(".gitignore"), "workspace/parent-ignored/\n").unwrap();
    let root = parent.join("workspace");
    fs::create_dir_all(&root).unwrap();
    write_policy(&root, "");
    fs::write(root.join(".gitignore"), "git-ignored/\n").unwrap();
    fs::write(root.join(".ignore"), "ignore-ignored/\n").unwrap();
    fs::create_dir_all(root.join(".git/info")).unwrap();
    fs::write(root.join(".git/info/exclude"), "git-info-ignored/\n").unwrap();
    for directory in [
        "git-ignored",
        "ignore-ignored",
        "parent-ignored",
        "git-info-ignored",
    ] {
        fs::create_dir_all(root.join(directory)).unwrap();
        fs::write(
            root.join(directory).join(format!("{directory}.ch")),
            "def admitted() = 1\n",
        )
        .unwrap();
    }
    fs::create_dir_all(root.join(".github/workflows")).unwrap();
    fs::write(root.join(".github/workflows/ci.yml"), "name: ci\n").unwrap();

    let names = walked(&root);
    for expected in [
        "git-ignored/git-ignored.ch",
        "ignore-ignored/ignore-ignored.ch",
        "parent-ignored/parent-ignored.ch",
        "git-info-ignored/git-info-ignored.ch",
        ".github/workflows/ci.yml",
    ] {
        assert!(
            names.iter().any(|path| path == expected),
            "ambient ignore source hid {expected}: {names:?}"
        );
    }
}

#[test]
fn global_git_ignore_does_not_affect_lint_entries() {
    let temp = tempdir().unwrap();
    let root = temp.path().join("workspace");
    let home = temp.path().join("home");
    fs::create_dir_all(root.join("globally-ignored")).unwrap();
    fs::create_dir_all(&home).unwrap();
    fs::write(
        root.join("globally-ignored/value.ch"),
        "def admitted() = 1\n",
    )
    .unwrap();
    let excludes = home.join("global-ignore");
    fs::write(&excludes, "globally-ignored/\n").unwrap();
    fs::write(
        home.join(".gitconfig"),
        format!("[core]\n\texcludesFile = {}\n", excludes.display()),
    )
    .unwrap();

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "global_git_ignore_child_probe", "--nocapture"])
        .env("CHELIS_LINT_GLOBAL_IGNORE_ROOT", &root)
        .env("HOME", &home)
        .env("XDG_CONFIG_HOME", home.join("xdg"))
        .status()
        .unwrap();
    assert!(status.success(), "global-ignore child probe failed");
}

#[test]
fn global_git_ignore_child_probe() {
    let Some(root) = std::env::var_os("CHELIS_LINT_GLOBAL_IGNORE_ROOT") else {
        return;
    };
    let names = walked(Path::new(&root));
    assert!(
        names.iter().any(|path| path == "globally-ignored/value.ch"),
        "global Git ignore hid source: {names:?}"
    );
}

#[test]
fn explicit_excluded_roots_are_linted_but_nested_exclusions_still_prune() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_policy(
        root,
        &format!(
            "{}\n{}",
            exclusion("generated/", "generated", "§12.2"),
            exclusion("nested/", "generated", "§12.2")
        ),
    );
    let generated = root.join("generated");
    fs::create_dir_all(generated.join("nested")).unwrap();
    fs::write(generated.join("keep.ch"), "def keep() = 1\n").unwrap();
    fs::write(generated.join("nested/drop.ch"), "def drop() = 1\n").unwrap();

    assert!(
        !walked(root)
            .iter()
            .any(|path| path.starts_with("generated"))
    );
    let explicit_dir = walked(&generated);
    assert!(explicit_dir.iter().any(|path| path == "keep.ch"));
    assert!(!explicit_dir.iter().any(|path| path == "nested/drop.ch"));

    let explicit_file = walked(&generated.join("keep.ch"));
    assert_eq!(explicit_file, vec![String::new()]);
}

#[cfg(unix)]
#[test]
fn explicit_excluded_directory_preserves_internal_symlink_root_override_only() {
    use std::os::unix::fs::symlink;

    let temp = tempdir().unwrap();
    let root = temp.path();
    write_policy(
        root,
        &format!(
            "{}\n{}",
            exclusion("generated/", "generated", "§12.2"),
            exclusion("nested/", "generated", "§12.2")
        ),
    );
    let generated = root.join("generated");
    fs::create_dir_all(generated.join("shared")).unwrap();
    fs::create_dir_all(generated.join("nested")).unwrap();
    fs::write(generated.join("shared/admitted.txt"), "admitted\n").unwrap();
    fs::write(generated.join("nested/excluded.txt"), "excluded\n").unwrap();
    symlink("shared/admitted.txt", generated.join("admitted.ch")).unwrap();
    symlink("nested/excluded.txt", generated.join("excluded.ch")).unwrap();

    let names = walked(&generated);
    assert!(
        names.iter().any(|path| path == "admitted.ch"),
        "the explicit directory's own exclusion is overridden for an internal admitted target: {names:?}"
    );
    assert!(
        !names.iter().any(|path| path == "excluded.ch"),
        "a separately excluded descendant target must remain excluded: {names:?}"
    );
}

#[test]
fn exact_file_patterns_prune_nested_files_but_not_explicit_file_roots() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_policy(root, &exclusion("/blocked.ch", "immutable", "§12.2"));
    let blocked = root.join("blocked.ch");
    fs::write(&blocked, "def blocked() = 1\n").unwrap();
    fs::write(root.join("visible.ch"), "def visible() = 1\n").unwrap();

    let names = walked(root);
    assert!(names.iter().any(|path| path == "visible.ch"));
    assert!(!names.iter().any(|path| path == "blocked.ch"));

    let policy = TraversalPolicy::load_for(root).unwrap();
    assert!(policy.is_excluded(&blocked, false));
    assert_eq!(walked(&blocked), vec![String::new()]);
}

#[test]
fn combined_hot_path_and_explain_matchers_agree() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_policy(
        root,
        &format!(
            "{}\n{}",
            exclusion("/generated/", "generated", "§12.2"),
            exclusion("/blocked.ch", "immutable", "§12.2")
        ),
    );
    let policy = TraversalPolicy::load_for(root).unwrap();

    for (path, is_dir) in [
        (root.join("generated"), true),
        (root.join("blocked.ch"), false),
        (root.join("visible.ch"), false),
        (root.join("other"), true),
    ] {
        assert_eq!(
            policy.is_excluded(&path, is_dir),
            policy.exclusion_for(&path, is_dir).is_some(),
            "combined and explain matchers disagreed for {}",
            path.display()
        );
    }
}

#[test]
fn shipped_policy_migrates_every_previous_walker_exclusion() {
    TraversalPolicy::verify_shipped_cross_refs(include_str!("../../../spec/01-nomenclature.md"))
        .unwrap();
    let temp = tempdir().unwrap();
    let root = temp.path();
    let skipped = [
        "target",
        ".git",
        "node_modules",
        "__pycache__",
        ".venv-issue-740",
        "nested/.claude/worktrees/generated",
    ];
    for directory in skipped {
        fs::create_dir_all(root.join(directory)).unwrap();
        fs::write(root.join(directory).join("hidden.ch"), "def hidden() = 1\n").unwrap();
    }
    fs::write(root.join("visible.ch"), "def visible() = 1\n").unwrap();

    let names = walked(root);
    assert!(names.iter().any(|path| path == "visible.ch"));
    assert!(!names.iter().any(|path| path.ends_with("hidden.ch")));

    let walker_source =
        fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/walker.rs"))
            .unwrap();
    for forbidden in [
        "target/",
        ".git/",
        "node_modules",
        "__pycache__",
        ".venv",
        ".claude/worktrees",
    ] {
        assert!(
            !walker_source.contains(forbidden),
            "walker.rs must not hard-code exclusion value {forbidden}"
        );
    }
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let rules_dir = crate_root.join("src/rules");
    let mut rule_paths = Vec::new();
    rust_sources_below(&rules_dir, &mut rule_paths);
    for path in rule_paths {
        let rule_source = fs::read_to_string(&path).unwrap();
        for forbidden in [
            "WalkBuilder",
            "WalkDir",
            "WalkParallel",
            "ignore::Walk",
            "read_dir",
            "glob::",
            "globwalk",
            "jwalk",
            "wax::",
        ] {
            assert!(
                !rule_source.contains(forbidden),
                "lint rule {} must consume canonical entries or a centralized bounded-discovery API, not use independent discovery primitive {forbidden}",
                path.display()
            );
        }
    }

    let manifest = fs::read_to_string(crate_root.join("Cargo.toml")).unwrap();
    let manifest: toml::Value = toml::from_str(&manifest).unwrap();
    for dependency_section in ["dependencies", "dev-dependencies", "build-dependencies"] {
        let Some(dependencies) = manifest
            .get(dependency_section)
            .and_then(toml::Value::as_table)
        else {
            continue;
        };
        for (name, specification) in dependencies {
            let package = specification
                .get("package")
                .and_then(toml::Value::as_str)
                .unwrap_or(name);
            assert!(
                !matches!(package, "walkdir" | "glob" | "globwalk" | "jwalk" | "wax"),
                "chelis-lint must not add independent traversal dependency {package}"
            );
        }
    }
}

#[test]
fn source_tripwire_discovery_reaches_nested_rule_modules() {
    let temp = tempdir().unwrap();
    fs::create_dir_all(temp.path().join("nested/deeper")).unwrap();
    let top = temp.path().join("top.rs");
    let nested = temp.path().join("nested/deeper/rule.rs");
    fs::write(&top, "// top\n").unwrap();
    fs::write(&nested, "// nested\n").unwrap();
    fs::write(temp.path().join("nested/ignored.txt"), "not Rust\n").unwrap();

    let mut sources = Vec::new();
    rust_sources_below(temp.path(), &mut sources);
    sources.sort();
    assert_eq!(sources, vec![nested, top]);
}

#[test]
fn crate_guardrail_locks_rule_registration_and_canonical_traversal_protocol() {
    let crate_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let agents_path = crate_root.join("AGENTS.md");
    let claude_path = crate_root.join("CLAUDE.md");
    let guardrail = fs::read_to_string(&agents_path)
        .expect("crates/chelis-lint/AGENTS.md must provide edit-time rule guardrails");
    assert_eq!(
        fs::read_link(&claude_path).expect("crates/chelis-lint/CLAUDE.md must be a symlink"),
        PathBuf::from("AGENTS.md"),
        "AGENTS.md is canonical; CLAUDE.md must resolve to it"
    );
    for required in [
        "registry::all_rules",
        "positive and negative",
        "Rule::prepare_run",
        "walker::walk",
        "spec/01-nomenclature.md",
    ] {
        assert!(
            guardrail.contains(required),
            "chelis-lint guardrail must document `{required}`"
        );
    }
}

#[test]
fn malformed_policy_is_not_silently_ignored_by_direct_rule_check() {
    let temp = tempdir().unwrap();
    write_spec(temp.path());
    fs::write(
        temp.path().join("chelis-lint.toml"),
        "version = 2\nspec = \"spec/01-nomenclature.md\"\n",
    )
    .unwrap();
    let path = temp.path().join("agent.ch");
    let source = "module Agent.Strategy\ndef identity(x: f32) -> f32 = x\n";
    fs::write(&path, source).unwrap();
    let ctx = Context {
        root: temp.path(),
        path: &path,
        source: Some(source),
        surface: Surface::SurfSource,
    };

    let violations = OpaqueDomainConstruction.check(&ctx);
    assert_eq!(violations.len(), 1);
    assert!(violations[0].message.contains("traversal policy error"));
    assert!(violations[0].message.contains("unsupported"));
}

#[test]
fn configured_exclusion_controls_opaque_prepared_catalog() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_policy(root, &exclusion("generated/", "generated", "§12.2"));
    fs::create_dir_all(root.join("generated")).unwrap();
    fs::write(
        root.join("generated/opaque.ch"),
        "module Hidden.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
    )
    .unwrap();
    fs::write(
        root.join("agent.ch"),
        "module Agent.Strategy\ndef forge(x: f32) -> Secret = Secret { value: x }\n",
    )
    .unwrap();
    let rules: Vec<Box<dyn Rule>> = vec![Box::new(OpaqueDomainConstruction)];

    let violations = chelis_lint::lint(root, &rules).unwrap();
    assert!(
        violations.is_empty(),
        "excluded opaque declaration reached prepared catalog: {violations:?}"
    );

    fs::rename(root.join("generated/opaque.ch"), root.join("opaque.ch")).unwrap();
    let violations = chelis_lint::lint(root, &rules).unwrap();
    assert_eq!(
        violations.len(),
        1,
        "admitted declaration must reach catalog"
    );
}

#[cfg(unix)]
#[test]
fn excluded_or_external_surf_symlink_targets_cannot_influence_opaque_catalog() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let external = tempdir().unwrap();
    let root = repository.path();
    write_policy(root, &exclusion("generated/", "generated", "§12.2"));
    fs::create_dir_all(root.join("generated")).unwrap();
    fs::write(
        root.join("generated/opaque.ch"),
        "module Hidden.Types\n@opaque\ntype HiddenSecret = | HiddenSecret { value: f32 }\n",
    )
    .unwrap();
    fs::write(
        external.path().join("opaque.ch"),
        "module External.Types\n@opaque\ntype ExternalSecret = | ExternalSecret { value: f32 }\n",
    )
    .unwrap();
    symlink("generated/opaque.ch", root.join("linked_excluded.ch")).unwrap();
    symlink(
        external.path().join("opaque.ch"),
        root.join("linked_external.ch"),
    )
    .unwrap();
    fs::write(
        root.join("agent.ch"),
        "module Agent.Strategy\ndef forge_hidden(x: f32) -> HiddenSecret = HiddenSecret { value: x }\ndef forge_external(x: f32) -> ExternalSecret = ExternalSecret { value: x }\n",
    )
    .unwrap();
    let rules: Vec<Box<dyn Rule>> = vec![Box::new(OpaqueDomainConstruction)];

    let violations = chelis_lint::lint(root, &rules).unwrap();
    assert!(
        violations.is_empty(),
        "excluded or machine-local symlink targets must not enter the prepared catalog: {violations:?}"
    );
}

#[cfg(unix)]
#[test]
fn broken_or_entry_kind_changing_source_symlinks_are_omitted() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let root = repository.path();
    write_policy(root, "");
    fs::create_dir_all(root.join("directory-target")).unwrap();
    symlink("missing.ch", root.join("broken.ch")).unwrap();
    symlink("directory-target", root.join("directory-as-file.ch")).unwrap();
    fs::write(root.join("visible.ch"), "def visible() = 1\n").unwrap();

    let names = walked(root);
    assert!(names.iter().any(|path| path == "visible.ch"));
    assert!(!names.iter().any(|path| path == "broken.ch"));
    assert!(!names.iter().any(|path| path == "directory-as-file.ch"));
}

#[cfg(unix)]
#[test]
fn non_regular_source_entries_and_symlink_targets_are_omitted() {
    use std::os::unix::fs::symlink;
    use std::os::unix::net::UnixListener;

    let repository = tempdir().unwrap();
    let root = repository.path();
    write_policy(root, "");
    let _direct_socket = UnixListener::bind(root.join("direct.ch")).unwrap();
    let _target_socket = UnixListener::bind(root.join("socket-target")).unwrap();
    symlink("socket-target", root.join("linked.ch")).unwrap();
    fs::write(root.join("visible.ch"), "def visible() = 1\n").unwrap();

    let names = walked(root);
    assert!(names.iter().any(|path| path == "visible.ch"));
    assert!(
        !names.iter().any(|path| path == "direct.ch"),
        "a discovered source-shaped socket must not enter the lint corpus: {names:?}"
    );
    assert!(
        !names.iter().any(|path| path == "linked.ch"),
        "a source-shaped symlink must resolve to a regular file, not a socket: {names:?}"
    );
}

#[cfg(unix)]
#[test]
fn internal_admitted_surf_symlink_target_still_contributes_to_opaque_catalog() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let root = repository.path();
    write_policy(root, "");
    fs::create_dir_all(root.join("shared")).unwrap();
    fs::write(
        root.join("shared/opaque.txt"),
        "module Shared.Types\n@opaque\ntype Secret = | Secret { value: f32 }\n",
    )
    .unwrap();
    symlink("shared/opaque.txt", root.join("linked.ch")).unwrap();
    fs::write(
        root.join("agent.ch"),
        "module Agent.Strategy\ndef forge(x: f32) -> Secret = Secret { value: x }\n",
    )
    .unwrap();
    let rules: Vec<Box<dyn Rule>> = vec![Box::new(OpaqueDomainConstruction)];

    let violations = chelis_lint::lint(root, &rules).unwrap();
    assert_eq!(
        violations.len(),
        1,
        "an internal policy-admitted symlink target must retain existing source-link behavior"
    );
}

fn lint_doc_filenames(root: &Path) -> Vec<chelis_lint::Violation> {
    let rules: Vec<Box<dyn Rule>> = vec![Box::new(DocFilenameConvention)];
    chelis_lint::lint(root, &rules).expect("lint doc filenames")
}

#[test]
fn excluded_manifest_cannot_grant_package_name_exception_to_admitted_doc() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_policy(root, &exclusion("crates/generated/", "generated", "§12.2"));
    fs::create_dir_all(root.join("docs")).unwrap();
    let doc = root.join("docs/foo-bar.md");
    fs::write(&doc, "# Docs\n").unwrap();
    fs::create_dir_all(root.join("crates/generated")).unwrap();
    fs::write(
        root.join("crates/generated/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    let violations = lint_doc_filenames(root);
    assert_eq!(
        violations.len(),
        1,
        "a policy-excluded manifest must not affect an admitted doc: {violations:?}"
    );
    assert_eq!(violations[0].rule_id, "doc-filename-convention");

    let direct_ctx = Context {
        root,
        path: &doc,
        source: None,
        surface: Surface::DocFile,
    };
    assert_eq!(
        DocFilenameConvention.check(&direct_ctx).len(),
        1,
        "the direct Rule::check compatibility path must enforce the same ancillary-file policy"
    );
}

#[cfg(unix)]
#[test]
fn excluded_or_external_manifest_symlink_targets_cannot_grant_doc_exception() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let external = tempdir().unwrap();
    let root = repository.path();
    write_policy(root, &exclusion("generated/", "generated", "§12.2"));
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::write(root.join("docs/foo-bar.md"), "# Docs\n").unwrap();
    fs::write(root.join("docs/external-bar.md"), "# Docs\n").unwrap();
    fs::create_dir_all(root.join("generated")).unwrap();
    fs::write(
        root.join("generated/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::write(
        external.path().join("Cargo.toml"),
        "[package]\nname = \"external-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("crates/excluded-link")).unwrap();
    fs::create_dir_all(root.join("crates/external-link")).unwrap();
    symlink(
        "../../generated/Cargo.toml",
        root.join("crates/excluded-link/Cargo.toml"),
    )
    .unwrap();
    symlink(
        external.path().join("Cargo.toml"),
        root.join("crates/external-link/Cargo.toml"),
    )
    .unwrap();

    let violations = lint_doc_filenames(root);
    assert_eq!(
        violations.len(),
        2,
        "excluded and machine-local manifest contents must not suppress admitted doc violations: {violations:?}"
    );
}

#[cfg(unix)]
#[test]
fn internal_admitted_manifest_symlink_target_still_grants_doc_exception() {
    use std::os::unix::fs::symlink;

    let repository = tempdir().unwrap();
    let root = repository.path();
    write_policy(root, "");
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::write(root.join("docs/foo-bar.md"), "# Docs\n").unwrap();
    fs::create_dir_all(root.join("config")).unwrap();
    fs::write(
        root.join("config/package.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("crates/foo-bar")).unwrap();
    symlink(
        "../../config/package.toml",
        root.join("crates/foo-bar/Cargo.toml"),
    )
    .unwrap();

    assert!(
        lint_doc_filenames(root).is_empty(),
        "an internal policy-admitted manifest link must retain the §8.3 exception"
    );
}

#[test]
fn admitted_manifest_still_grants_package_name_exception_to_admitted_doc() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("docs")).unwrap();
    fs::write(root.join("docs/foo-bar.md"), "# Docs\n").unwrap();
    fs::create_dir_all(root.join("crates/foo-bar")).unwrap();
    fs::write(
        root.join("crates/foo-bar/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    assert!(
        lint_doc_filenames(root).is_empty(),
        "an admitted matching Cargo manifest must retain the §8.3 package-name exception"
    );
}

#[test]
fn subdirectory_and_explicit_doc_lint_retain_admitted_sibling_package_exception() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_policy(root, "");
    let docs = root.join("docs");
    fs::create_dir_all(&docs).unwrap();
    let doc = docs.join("foo-bar.md");
    fs::write(&doc, "# Docs\n").unwrap();
    fs::create_dir_all(root.join("crates/foo-bar")).unwrap();
    fs::write(
        root.join("crates/foo-bar/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    assert!(
        lint_doc_filenames(&docs).is_empty(),
        "subdirectory lint must retain an admitted sibling workspace package"
    );
    assert!(
        lint_doc_filenames(&doc).is_empty(),
        "explicit-file lint must retain an admitted sibling workspace package"
    );
    let direct_ctx = Context {
        root: &docs,
        path: &doc,
        source: None,
        surface: Surface::DocFile,
    };
    assert!(
        DocFilenameConvention.check(&direct_ctx).is_empty(),
        "direct Rule::check must use the policy-root workspace package catalog"
    );
}

#[test]
fn external_parent_manifest_cannot_grant_package_name_exception() {
    let temp = tempdir().unwrap();
    fs::write(
        temp.path().join("Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    let root = temp.path().join("repository");
    write_policy(&root, "");
    let docs = root.join("docs");
    fs::create_dir_all(&docs).unwrap();
    let doc = docs.join("foo-bar.md");
    fs::write(&doc, "# Docs\n").unwrap();

    for target in [docs.clone(), doc.clone()] {
        let violations = lint_doc_filenames(&target);
        assert_eq!(
            violations.len(),
            1,
            "a Cargo manifest above the policy root is machine-local, not a workspace package: {violations:?}"
        );
    }
    let direct_ctx = Context {
        root: &docs,
        path: &doc,
        source: None,
        surface: Surface::DocFile,
    };
    assert_eq!(
        DocFilenameConvention.check(&direct_ctx).len(),
        1,
        "direct Rule::check must ignore machine-local manifests above policy root"
    );
}

#[test]
fn explicit_doc_file_lint_preserves_ancestor_package_name_exception() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("docs")).unwrap();
    let doc = root.join("docs/foo-bar.md");
    fs::write(&doc, "# Docs\n").unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    assert!(
        lint_doc_filenames(&doc).is_empty(),
        "explicit-file lint must retain the ancestor Cargo-package exception"
    );
}

#[test]
fn explicit_doc_does_not_admit_manifest_from_its_excluded_parent_tree() {
    let temp = tempdir().unwrap();
    let root = temp.path();
    write_policy(root, &exclusion("generated/", "generated", "§12.2"));
    let generated_docs = root.join("generated/docs");
    fs::create_dir_all(&generated_docs).unwrap();
    let doc = generated_docs.join("foo-bar.md");
    fs::write(&doc, "# Docs\n").unwrap();
    fs::write(
        root.join("generated/Cargo.toml"),
        "[package]\nname = \"foo-bar\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();

    let violations = lint_doc_filenames(&doc);
    assert_eq!(
        violations.len(),
        1,
        "depth-zero admission applies only to the explicit doc, not an excluded ancestor manifest"
    );
}
