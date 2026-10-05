//! The shell's `docs/CHELIS_SURFACE.md` carries the pinned toolchain's complete
//! capability surface guide in its `chelis-surface` managed span, while
//! shell-owned text and exact heading exclusions remain local (contract §3).
//! The `CLAUDE.md -> AGENTS.md` symlink is restored by the same sync (§1).

use chelis_conformance::{audit, canonical, managed_block, scaffold};

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

fn row<'a>(report: &'a audit::AuditReport, key: &str) -> &'a audit::RowResult {
    report.rows.iter().find(|r| r.key == key).unwrap()
}

/// The surface guide as sync writes it for a shell with every shared skill:
/// the embedded text with its repo-relative links pinned to `VER`.
fn pinned_guide() -> String {
    chelis_conformance::links::pin_links(
        canonical::body("chelis-surface").unwrap(),
        "docs/CHELIS_SURFACE.md",
        "docs/CHELIS_SURFACE.md",
        VER,
        &chelis_conformance::links::LocalTargets::for_shell(&[]),
    )
}

fn surface_block(root: &std::path::Path) -> managed_block::ManagedBlock {
    let text = std::fs::read_to_string(root.join("docs/CHELIS_SURFACE.md")).unwrap();
    managed_block::find(&text, "chelis-surface").expect("chelis-surface block")
}

/// The last `## ` section of the pinned guide and one line of its body, read
/// from the embedded text so the test follows the guide's own headings.
fn last_section() -> (String, String) {
    let body = canonical::body("chelis-surface").unwrap();
    let lines: Vec<&str> = body.lines().collect();
    let start = lines.iter().rposition(|l| l.starts_with("## ")).unwrap();
    let sentinel = lines[start + 1..]
        .iter()
        .find(|l| l.trim().len() > 20 && !l.starts_with('#') && !l.contains("]("))
        .expect("section has body text");
    (lines[start].to_string(), sentinel.to_string())
}

#[test]
fn scaffold_materializes_the_complete_pinned_surface_guide() {
    let (_tmp, root) = green_shell();
    let upstream = std::fs::read_to_string(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../docs/CHELIS_SURFACE.md"),
    )
    .unwrap();
    assert_eq!(upstream, canonical::body("chelis-surface").unwrap());
    assert_eq!(
        managed_block::normalize_body(&surface_block(&root).body),
        managed_block::normalize_body(&pinned_guide())
    );
}

/// The inherited guide is current by construction, so the audit no longer asks
/// the shell for hand-maintained `@pin`/`@upstream` markers. A body without them
/// is green; tampering inside the fences is not.
#[test]
fn audit_checks_the_inherited_body_not_hand_maintained_markers() {
    let (_tmp, root) = green_shell();
    let path = root.join("docs/CHELIS_SURFACE.md");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(
        !text.contains("@pin"),
        "fixture must not carry the old markers"
    );
    assert_eq!(
        row(&audit::audit(&root), "chelis-surface").verdict,
        audit::Verdict::Pass
    );

    // A self-consistent fork of the inherited text (body edited, stamp
    // recomputed) still fails: the body must equal the pinned guide.
    let block = surface_block(&root);
    let forged = managed_block::render(
        "chelis-surface",
        &block.version,
        &block.body.replacen("Chelis", "Forked", 1),
    );
    let forged_doc = format!("{}{forged}{}", &text[..block.span.0], &text[block.span.1..]);
    std::fs::write(&path, forged_doc).unwrap();
    let report = audit::audit(&root);
    let r = row(&report, "chelis-surface");
    assert_eq!(r.verdict, audit::Verdict::Fail);
    assert!(
        r.diagnostic.contains("differs from the canonical"),
        "{}",
        r.diagnostic
    );
    assert!(!report.ok());

    scaffold::sync_managed_blocks(&root, VER).expect("sync");
    assert!(
        audit::audit(&root).ok(),
        "sync must restore the pinned body"
    );
}

#[test]
fn shell_owned_heading_exclusion_filters_sync_and_removal_restores_upstream() {
    let (_tmp, root) = green_shell();
    let (heading, sentinel) = last_section();
    let path = root.join("docs/CHELIS_SURFACE.md");
    let local_text =
        "\n## Shell Domain Notes\nThis shell drives the surface through its pricing kernels.\n";
    append(&path, local_text);
    let selector = selector_block(&[&heading]);
    append(&path, &selector);

    assert!(
        !audit::audit(&root).ok(),
        "adding an unapplied selector must make the current managed body stale"
    );
    scaffold::sync_managed_blocks(&root, VER).expect("apply surface exclusion");

    let filtered = std::fs::read_to_string(&path).unwrap();
    let block = surface_block(&root);
    assert!(!block.body.lines().any(|line| line == heading));
    assert!(!block.body.contains(&sentinel));
    assert!(filtered.contains(local_text.trim()));
    assert!(filtered.contains(selector.trim()));
    assert!(audit::audit(&root).ok(), "filtered body must audit green");

    std::fs::write(&path, filtered.replace(&selector, "")).unwrap();
    scaffold::sync_managed_blocks(&root, VER).expect("restore surface section");
    let restored = std::fs::read_to_string(&path).unwrap();
    let block = surface_block(&root);
    assert!(block.body.lines().any(|line| line == heading));
    assert!(block.body.contains(&sentinel));
    assert!(restored.contains(local_text.trim()));
    assert!(audit::audit(&root).ok(), "restored body must audit green");
}

#[test]
fn shell_may_exclude_the_complete_inherited_surface_guide() {
    let (_tmp, root) = green_shell();
    let title = canonical::body("chelis-surface")
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    assert!(
        title.starts_with("# "),
        "guide starts with its title: {title:?}"
    );
    let path = root.join("docs/CHELIS_SURFACE.md");
    append(&path, &selector_block(&[&title]));

    scaffold::sync_managed_blocks(&root, VER).expect("exclude complete surface guide");
    assert!(surface_block(&root).body.trim().is_empty());
    assert!(audit::audit(&root).ok());
}

#[test]
fn invalid_surface_selectors_fail_audit_and_sync_before_any_write() {
    let (heading, _) = last_section();
    for (name, selectors, expected) in [
        (
            "missing",
            vec!["## Not An Upstream Heading".to_string()],
            "matches no upstream heading",
        ),
        (
            "duplicate",
            vec![heading.clone(), heading.clone()],
            "duplicated",
        ),
    ] {
        let (_tmp, root) = green_shell();
        let agents_path = root.join("AGENTS.md");
        let surface_path = root.join("docs/CHELIS_SURFACE.md");
        let refs: Vec<&str> = selectors.iter().map(String::as_str).collect();
        append(&surface_path, &selector_block(&refs));
        let agents_before = std::fs::read_to_string(&agents_path).unwrap();
        let surface_before = std::fs::read_to_string(&surface_path).unwrap();

        let report = audit::audit(&root);
        let r = row(&report, "chelis-surface");
        assert_eq!(r.verdict, audit::Verdict::Fail, "{name}");
        assert!(r.diagnostic.contains(expected), "{name}: {}", r.diagnostic);

        let error = scaffold::sync_managed_blocks(&root, VER).unwrap_err();
        assert!(error.contains(expected), "{name}: {error}");
        assert!(error.contains("docs/CHELIS_SURFACE.md"), "{name}: {error}");
        assert_eq!(
            std::fs::read_to_string(&agents_path).unwrap(),
            agents_before,
            "{name} changed AGENTS.md"
        );
        assert_eq!(
            std::fs::read_to_string(&surface_path).unwrap(),
            surface_before,
            "{name} changed the surface document"
        );
    }
}

/// A shell stamped by an older toolchain carries the short pointer header in a
/// `chelis-surface-header` block above hand-authored rows. Sync writes the
/// complete guide where that header stood and leaves the shell's rows outside
/// the fences alone; no superseded block survives with a stale stamp.
#[test]
fn sync_replaces_the_legacy_surface_header_in_place() {
    let (_tmp, root) = green_shell();
    let path = root.join("docs/CHELIS_SURFACE.md");
    let legacy = managed_block::render(
        canonical::LEGACY_SURFACE_HEADER,
        "0.18.0",
        "This file is a domain-scoped view of the canonical Chelis capability surface.",
    );
    let hand_rows = "## Capabilities\n\n| Capability | Status |\n|---|---|\n| grad | `@pin` |\n";
    std::fs::write(
        &path,
        format!("# Shell Capability Surface\n\n{legacy}\n{hand_rows}"),
    )
    .unwrap();
    let report = audit::audit(&root);
    assert_eq!(
        row(&report, "chelis-surface").verdict,
        audit::Verdict::Fail,
        "a legacy header alone is not the inherited guide"
    );

    scaffold::sync_managed_blocks(&root, VER).expect("sync");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(managed_block::find(&text, canonical::LEGACY_SURFACE_HEADER).is_none());
    assert_eq!(text.matches("BEGIN CHELIS MANAGED BLOCK").count(), 1);
    let block = surface_block(&root);
    assert!(text.starts_with("# Shell Capability Surface\n\n"));
    assert!(text[block.span.1..].contains(hand_rows.trim()));
    assert!(audit::audit(&root).ok());
}

/// §1: `CLAUDE.md` is a symlink to `AGENTS.md`, and sync restores it whether it
/// is missing, a regular file, or a link to somewhere else. The audit rejects
/// each broken state first, so the repair is observable.
#[test]
fn sync_restores_the_claude_symlink() {
    // `notice` is the kind of content sync must announce replacing; a link
    // carries no content of its own and a missing file has none to lose.
    for (name, breakage, notice) in [
        ("missing", None, None),
        ("regular file", Some("copy"), Some("a regular file")),
        ("directory", Some("dir"), Some("a directory")),
        ("misdirected link", Some("link"), None),
    ] {
        let (_tmp, root) = green_shell();
        let claude = root.join("CLAUDE.md");
        std::fs::remove_file(&claude).unwrap();
        match breakage {
            Some("copy") => {
                std::fs::write(&claude, "shell-specific claude notes\n").unwrap();
            }
            Some("dir") => {
                std::fs::create_dir(&claude).unwrap();
                std::fs::write(claude.join("notes.md"), "kept here\n").unwrap();
            }
            Some(_) => std::os::unix::fs::symlink("README.md", &claude).unwrap(),
            None => {}
        }
        let report = audit::audit(&root);
        let r = row(&report, "agents-md");
        assert_eq!(r.verdict, audit::Verdict::Fail, "{name}");
        assert!(
            r.diagnostic.contains("CLAUDE.md"),
            "{name}: {}",
            r.diagnostic
        );

        let notices = scaffold::sync_managed_blocks(&root, VER).expect("sync");
        match notice {
            Some(kind) => assert!(
                notices
                    .iter()
                    .any(|n| n.starts_with("CLAUDE.md: replaced") && n.contains(kind)),
                "{name}: replacing shell content must be announced: {notices:?}"
            ),
            None => assert!(notices.is_empty(), "{name}: {notices:?}"),
        }
        let meta = std::fs::symlink_metadata(&claude).unwrap();
        assert!(meta.file_type().is_symlink(), "{name}");
        assert_eq!(
            std::fs::read_link(&claude).unwrap(),
            std::path::Path::new("AGENTS.md"),
            "{name}"
        );
        assert!(audit::audit(&root).ok(), "{name}");
    }
}

/// A surface document sync creates is byte for byte the one `init` writes.
#[test]
fn a_created_surface_document_matches_the_scaffolded_bytes() {
    let (_tmp, root) = green_shell();
    let path = root.join("docs/CHELIS_SURFACE.md");
    let scaffolded = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    scaffold::sync_managed_blocks(&root, VER).expect("sync");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), scaffolded);
}

/// The surface document is fully generated, so sync creates it in a shell that
/// lacks it, `docs/` included, instead of refusing (chelis#2831).
#[test]
fn sync_creates_the_surface_document_when_it_is_missing() {
    let (_tmp, root) = green_shell();
    std::fs::remove_dir_all(root.join("docs")).unwrap();
    assert!(scaffold::preflight_restamp_targets(&root).is_ok());

    scaffold::sync_managed_blocks(&root, VER).expect("sync creates the surface document");
    assert_eq!(
        managed_block::normalize_body(&surface_block(&root).body),
        managed_block::normalize_body(&pinned_guide())
    );
    let report = audit::audit(&root);
    assert_eq!(row(&report, "chelis-surface").verdict, audit::Verdict::Pass);
}
