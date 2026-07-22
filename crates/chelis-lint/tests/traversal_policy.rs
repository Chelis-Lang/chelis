use chelis_lint::policy::{TraversalClass, TraversalPolicy};
use chelis_lint::{
    Context, Rule, Surface, rules::opaque_domain_construction::OpaqueDomainConstruction,
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
    let rules_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/rules");
    for entry in fs::read_dir(&rules_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
            continue;
        }
        let rule_source = fs::read_to_string(&path).unwrap();
        for forbidden in ["WalkBuilder", "WalkDir", "WalkParallel", "ignore::Walk"] {
            assert!(
                !rule_source.contains(forbidden),
                "lint rule {} must consume canonical entries, not create an independent walker with {forbidden}",
                path.display()
            );
        }
    }
}

#[test]
fn crate_guardrail_locks_rule_registration_and_canonical_traversal_protocol() {
    let guardrail = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("CLAUDE.md"))
        .expect("crates/chelis-lint/CLAUDE.md must provide edit-time rule guardrails");
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
