//! The shell's AGENTS.md carries the pinned Chelis agent contract in its managed
//! span, while shell-owned text and exact heading exclusions remain local.

use chelis_conformance::{audit, managed_block, scaffold};

const VER: &str = env!("CARGO_PKG_VERSION");

fn green_shell() -> (tempfile::TempDir, std::path::PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("shell");
    scaffold::scaffold(&root, "shell", "Shell", VER).expect("scaffold");
    assert!(audit::audit(&root).ok(), "scaffold must audit green");
    (tmp, root)
}

fn append(path: &std::path::Path, text: &str) {
    let mut body = std::fs::read_to_string(path).unwrap();
    body.push_str(text);
    std::fs::write(path, body).unwrap();
}

fn selector_block(selectors: &[&str]) -> String {
    let mut block = String::from("\n<!-- shell-local:exclude:begin -->\n");
    for selector in selectors {
        block.push_str(&format!("<!-- {selector} -->\n"));
    }
    block.push_str("<!-- shell-local:exclude:end -->\n");
    block
}

#[test]
fn scaffold_materializes_the_complete_pinned_agents_contract() {
    let (_tmp, root) = green_shell();
    let agents = std::fs::read_to_string(root.join("AGENTS.md")).unwrap();
    let block = managed_block::find(&agents, "agents-inheritance").unwrap();
    let upstream = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../AGENTS.md"),
    )
    .unwrap();

    assert_eq!(
        managed_block::normalize_body(&block.body),
        managed_block::normalize_body(&upstream)
    );
    for skill in chelis_conformance::skills::SHARED_SKILLS {
        assert!(
            block.body.contains(&format!("`{skill}`")),
            "materialized AGENTS.md omitted shared skill {skill:?}"
        );
    }
}

#[test]
fn shell_owned_heading_exclusion_filters_sync_and_removal_restores_upstream() {
    let (_tmp, root) = green_shell();
    let path = root.join("AGENTS.md");
    let local_text = "\n## Voyage Local Rule\nKeep experiment eras immutable.\n";
    append(&path, local_text);
    let selector = selector_block(&["### Numeric Surface Discipline"]);
    append(&path, &selector);

    assert!(
        !audit::audit(&root).ok(),
        "adding an unapplied selector must make the current managed body stale"
    );
    scaffold::sync_managed_blocks(&root, VER).expect("apply AGENTS exclusion");

    let filtered = std::fs::read_to_string(&path).unwrap();
    let block = managed_block::find(&filtered, "agents-inheritance").unwrap();
    assert!(
        !block
            .body
            .lines()
            .any(|line| line == "### Numeric Surface Discipline")
    );
    assert!(
        !block
            .body
            .contains("No numeric channel outside the tagged carrier")
    );
    assert!(filtered.contains(local_text.trim()));
    assert!(filtered.contains(selector.trim()));
    assert!(audit::audit(&root).ok(), "filtered body must audit green");

    std::fs::write(&path, filtered.replace(&selector, "")).unwrap();
    scaffold::sync_managed_blocks(&root, VER).expect("restore AGENTS section");
    let restored = std::fs::read_to_string(&path).unwrap();
    let block = managed_block::find(&restored, "agents-inheritance").unwrap();
    assert!(
        block
            .body
            .lines()
            .any(|line| line == "### Numeric Surface Discipline")
    );
    assert!(
        block
            .body
            .contains("No numeric channel outside the tagged carrier")
    );
    assert!(restored.contains(local_text.trim()));
    assert!(audit::audit(&root).ok(), "restored body must audit green");
}

#[test]
fn invalid_agents_selectors_fail_audit_and_sync_before_any_write() {
    for (name, selectors, expected) in [
        (
            "missing",
            vec!["## Not An Upstream Heading"],
            "matches no upstream heading",
        ),
        (
            "overlap",
            vec!["## Quality Standards", "### Spec-First Development"],
            "overlap",
        ),
        (
            "duplicate",
            vec!["## Quality Standards", "## Quality Standards"],
            "duplicated",
        ),
    ] {
        let (_tmp, root) = green_shell();
        let agents_path = root.join("AGENTS.md");
        let surface_path = root.join("docs/CHELIS_SURFACE.md");
        append(&agents_path, &selector_block(&selectors));
        let agents_before = std::fs::read_to_string(&agents_path).unwrap();
        let surface_before = std::fs::read_to_string(&surface_path).unwrap();

        let report = audit::audit(&root);
        let row = report
            .rows
            .iter()
            .find(|row| row.key == "agents-md")
            .unwrap();
        assert_eq!(row.verdict, audit::Verdict::Fail, "{name}");
        assert!(
            row.diagnostic.contains(expected),
            "{name}: {}",
            row.diagnostic
        );

        let error = scaffold::sync_managed_blocks(&root, VER).unwrap_err();
        assert!(error.contains(expected), "{name}: {error}");
        assert_eq!(
            std::fs::read_to_string(&agents_path).unwrap(),
            agents_before
        );
        assert_eq!(
            std::fs::read_to_string(&surface_path).unwrap(),
            surface_before
        );
    }
}

#[test]
fn shell_may_exclude_the_complete_inherited_agents_contract() {
    let (_tmp, root) = green_shell();
    let path = root.join("AGENTS.md");
    let local_text = "\n## Shell Contract\nKeep this shell's current local guidance.\n";
    append(&path, local_text);
    append(&path, &selector_block(&["# Chelis Agent Contract"]));

    scaffold::sync_managed_blocks(&root, VER).expect("exclude complete AGENTS contract");
    let agents = std::fs::read_to_string(&path).unwrap();
    let block = managed_block::find(&agents, "agents-inheritance").unwrap();
    assert!(block.body.trim().is_empty());
    assert!(agents.contains(local_text.trim()));
    assert!(audit::audit(&root).ok());
}

#[test]
fn malformed_agents_selector_span_fails_before_any_write() {
    let (_tmp, root) = green_shell();
    let agents_path = root.join("AGENTS.md");
    append(
        &agents_path,
        "\n<!-- shell-local:exclude:begin -->\n<!-- ## Quality Standards -->\n",
    );
    let before = std::fs::read_to_string(&agents_path).unwrap();

    let report = audit::audit(&root);
    let row = report
        .rows
        .iter()
        .find(|row| row.key == "agents-md")
        .unwrap();
    assert_eq!(row.verdict, audit::Verdict::Fail);
    assert!(row.diagnostic.contains("shell-local:exclude:end"));

    let error = scaffold::sync_managed_blocks(&root, VER).unwrap_err();
    assert!(error.contains("shell-local:exclude:end"), "{error}");
    assert_eq!(std::fs::read_to_string(&agents_path).unwrap(), before);
}

#[test]
fn quoted_agents_selector_spans_fail_before_any_write() {
    let quoted_spans = [
        (
            "fenced code",
            "\n```markdown\n<!-- shell-local:exclude:begin -->\n<!-- ### Numeric Surface Discipline -->\n<!-- shell-local:exclude:end -->\n```\n",
        ),
        (
            "HTML block",
            "\n<div>\n<!-- shell-local:exclude:begin -->\n<!-- ### Numeric Surface Discipline -->\n<!-- shell-local:exclude:end -->\n</div>\n",
        ),
    ];

    for (name, quoted_span) in quoted_spans {
        let (_tmp, root) = green_shell();
        let agents_path = root.join("AGENTS.md");
        let surface_path = root.join("docs/CHELIS_SURFACE.md");
        append(&agents_path, quoted_span);
        let agents_before = std::fs::read_to_string(&agents_path).unwrap();
        let surface_before = std::fs::read_to_string(&surface_path).unwrap();

        let report = audit::audit(&root);
        let row = report
            .rows
            .iter()
            .find(|row| row.key == "agents-md")
            .unwrap();
        assert_eq!(row.verdict, audit::Verdict::Fail, "{name}");
        assert!(
            row.diagnostic.contains("standalone Markdown comment"),
            "{name}: {}",
            row.diagnostic
        );

        let error = scaffold::sync_managed_blocks(&root, VER).unwrap_err();
        assert!(
            error.contains("standalone Markdown comment"),
            "{name}: {error}"
        );
        assert_eq!(
            std::fs::read_to_string(&agents_path).unwrap(),
            agents_before,
            "{name} changed AGENTS.md"
        );
        assert_eq!(
            std::fs::read_to_string(&surface_path).unwrap(),
            surface_before,
            "{name} changed the other managed document"
        );
    }
}
