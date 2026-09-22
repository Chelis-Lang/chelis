//! Contract §8 skill-set customization: shell-owned additions and exact
//! embedded-skill exclusions survive `conform sync` without allowing a silent
//! fork of a retained shared skill.
//!
//! Enforcement runs in both directions: an undeclared prune fails and sync
//! restores it; a declared exclusion removes the directory and audits green;
//! removing the declaration restores current upstream bytes. Unknown exclusion
//! names fail before sync writes, and near-miss control names remain rejected.
//!
//! Repo-local additions and trailing shell-local override blocks provide the
//! positive and negative parity for the two other sanctioned states.

use std::path::{Path, PathBuf};

use chelis_conformance::audit::{self, Verdict};
use chelis_conformance::{scaffold, skills};

const VER: &str = env!("CARGO_PKG_VERSION");

fn stamp(dir: &Path, name: &str) -> PathBuf {
    let root = dir.join(name);
    scaffold::scaffold(&root, name, "Myshell", VER).expect("scaffold");
    root
}

fn row<'a>(report: &'a audit::AuditReport, key: &str) -> &'a audit::RowResult {
    report.rows.iter().find(|r| r.key == key).unwrap()
}

fn append_conform_table(root: &Path, body: &str) {
    let path = root.join("reef.toml");
    let mut text = std::fs::read_to_string(&path).unwrap();
    text.push_str(body);
    std::fs::write(&path, text).unwrap();
}

/// Write `body` at the TOP of reef.toml, before `[package]`. Header-less
/// spellings (`conform = { … }`, `conform.exclude = …`) belong to whichever
/// table precedes them, so only here are they the top-level `conform` value.
/// Appending them instead makes them `package.conform`, which is a different
/// declaration and is covered by its own test.
fn prepend_to_reef(root: &Path, body: &str) {
    let path = root.join("reef.toml");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, format!("{body}\n{text}")).unwrap();
}

/// Where a snippet goes in the fixture's reef.toml.
#[derive(Clone, Copy)]
enum At {
    /// Before `[package]`: the snippet's keys are top-level.
    Top,
    /// After everything: header-less keys land under the last table.
    End,
}

fn write_snippet(root: &Path, at: At, body: &str) {
    match at {
        At::Top => prepend_to_reef(root, body),
        At::End => append_conform_table(root, body),
    }
}

// ---------------------------------------------------------------- 1. no prune

#[test]
fn a_pruned_shared_skill_fails_the_audit() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "pruned");
    assert!(audit::audit(&root).ok(), "baseline green");

    // The c-note shape: a reasoned exclusion, carried out by deleting the dir.
    std::fs::remove_dir_all(root.join("agent-skills/cli-surface")).unwrap();

    let report = audit::audit(&root);
    let r = row(&report, "vendored-skills");
    assert_eq!(r.verdict, Verdict::Fail);
    assert!(
        r.diagnostic.contains("cli-surface"),
        "the diagnostic must name the pruned skill: {}",
        r.diagnostic
    );
    assert!(
        r.fix.contains("excluded_skills") && r.fix.contains("shell-local"),
        "the fix must name both sanctioned alternatives, not just \
         say `run sync`: {}",
        r.fix
    );
}

#[test]
fn sync_restores_a_pruned_shared_skill_and_says_so() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "restored");
    std::fs::remove_dir_all(root.join("agent-skills/cli-surface")).unwrap();

    let notices = scaffold::materialize_skills(&root).expect("materialize");
    assert!(
        root.join("agent-skills/cli-surface/SKILL.md").is_file(),
        "sync restores the pruned skill"
    );
    let notice = notices
        .iter()
        .find(|n| n.starts_with("cli-surface:"))
        .unwrap_or_else(|| panic!("sync must announce the restore; notices: {notices:?}"));
    assert!(
        notice.contains("excluded_skills"),
        "the notice must explain how to omit the restored skill: {notice}"
    );
    assert!(audit::audit(&root).ok(), "sync restores green");
}

#[test]
fn a_fresh_init_does_not_report_its_own_skills_as_restored() {
    // Negative parity for the notice: materializing a tree that has no
    // `agent-skills/` at all is first-time materialization, not a restoration,
    // and must stay quiet. (Sampling the directory inside the loop instead of
    // once would make every skill after the first report itself.)
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "fresh");
    let notices = scaffold::materialize_skills(&root).expect("materialize");
    assert!(
        notices.is_empty(),
        "a re-sync of an intact tree is silent: {notices:?}"
    );

    let bare = tmp.path().join("bare");
    std::fs::create_dir_all(&bare).unwrap();
    let notices = scaffold::materialize_skills(&bare).expect("materialize");
    assert!(
        notices.is_empty(),
        "first-time materialization is not a restore: {notices:?}"
    );
}

// ------------------------------------------- 3. declared shared-skill exclusions

#[test]
fn sync_removes_declared_shared_skills_and_audit_accepts_the_result() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "excluded");
    append_conform_table(
        &root,
        "\n[conform]\nexcluded_skills = [\"cli-surface\", \"backend-numerics\"]\n",
    );

    assert_eq!(
        row(&audit::audit(&root), "vendored-skills").verdict,
        Verdict::Fail,
        "declared exclusions must take effect before the tree is conformant"
    );
    scaffold::materialize_skills(&root).expect("materialize exclusions");
    assert!(
        !root.join("agent-skills/cli-surface").exists()
            && !root.join("agent-skills/backend-numerics").exists(),
        "sync must remove every declared shared skill"
    );
    assert!(
        root.join("agent-skills/spec-sync/SKILL.md").is_file(),
        "sync must retain shared skills that were not excluded"
    );
    let upstream = std::fs::read_to_string(root.join("agent-skills/UPSTREAM.toml")).unwrap();
    assert!(
        upstream.contains("spec-sync"),
        "pointer must list retained skills"
    );
    assert!(
        !upstream.contains("cli-surface") && !upstream.contains("backend-numerics"),
        "pointer must list the materialized subset: {upstream}"
    );
    assert!(
        audit::audit(&root).ok(),
        "the configured subset must audit green"
    );
}

#[test]
fn audit_rejects_any_materialized_directory_for_an_excluded_shared_skill() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "excluded-directory");
    append_conform_table(&root, "\n[conform]\nexcluded_skills = [\"cli-surface\"]\n");
    scaffold::materialize_skills(&root).expect("materialize exclusions");

    let excluded_dir = root.join("agent-skills/cli-surface");
    std::fs::create_dir_all(&excluded_dir).unwrap();
    std::fs::write(excluded_dir.join("README.md"), "leftover\n").unwrap();

    let report = audit::audit(&root);
    let result = row(&report, "vendored-skills");
    assert_eq!(result.verdict, Verdict::Fail);
    assert!(
        result.diagnostic.contains("cli-surface")
            && result.diagnostic.contains("present but declared"),
        "diagnostic: {}",
        result.diagnostic
    );
}

#[test]
fn removing_an_exclusion_restores_the_current_embedded_skill() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "restored-exclusion");
    append_conform_table(&root, "\n[conform]\nexcluded_skills = [\"cli-surface\"]\n");
    scaffold::materialize_skills(&root).expect("exclude");
    assert!(
        !root.join("agent-skills/cli-surface").exists(),
        "precondition: exclusion removed the skill"
    );

    let manifest = root.join("reef.toml");
    let text = std::fs::read_to_string(&manifest).unwrap();
    std::fs::write(
        &manifest,
        text.replace(
            "excluded_skills = [\"cli-surface\"]",
            "excluded_skills = []",
        ),
    )
    .unwrap();
    scaffold::materialize_skills(&root).expect("restore");

    let restored = std::fs::read_to_string(root.join("agent-skills/cli-surface/SKILL.md"))
        .expect("restored skill");
    assert_eq!(
        restored.trim_end(),
        skills::skill_body("cli-surface").unwrap().trim_end(),
        "removing the exclusion must restore the current embedded body"
    );
    assert!(audit::audit(&root).ok());
}

#[test]
fn an_unknown_excluded_skill_fails_audit_and_sync_before_writing() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "unknown-exclusion");
    append_conform_table(&root, "\n[conform]\nexcluded_skills = [\"cli-surfaec\"]\n");
    let upstream_before = std::fs::read_to_string(root.join("agent-skills/UPSTREAM.toml")).unwrap();

    let report = audit::audit(&root);
    let r = row(&report, "vendored-skills");
    assert_eq!(r.verdict, Verdict::Fail);
    assert!(
        r.diagnostic.contains("cli-surfaec"),
        "diag: {}",
        r.diagnostic
    );
    let err = scaffold::materialize_skills(&root).unwrap_err();
    assert!(err.contains("cli-surfaec"), "sync error: {err}");
    assert_eq!(
        std::fs::read_to_string(root.join("agent-skills/UPSTREAM.toml")).unwrap(),
        upstream_before,
        "sync must validate exclusion names before its first write"
    );
}

#[test]
fn exclusion_aliases_are_rejected_in_favor_of_the_one_documented_key() {
    for key in [
        "exclude",
        "exclude_skills",
        "skip_skills",
        "skills",
        "profile",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "k");
        append_conform_table(&root, &format!("\n[conform]\n{key} = [\"cli-surface\"]\n"));
        let report = audit::audit(&root);
        let r = row(&report, "vendored-skills");
        assert_eq!(r.verdict, Verdict::Fail, "[conform] {key} must be rejected");
        assert!(r.diagnostic.contains(key), "diag: {}", r.diagnostic);
        assert!(
            r.diagnostic.contains("excluded_skills"),
            "diag: {}",
            r.diagnostic
        );
        let err = scaffold::materialize_skills(&root).unwrap_err();
        assert!(err.contains(key), "sync error: {err}");
        assert!(err.contains("excluded_skills"), "sync error: {err}");
    }
}

/// Every TOML spelling of the control §8 denies, end to end through the audit.
///
/// Two review rounds produced six of these against a hand-rolled line scan, each
/// one closed by a patch that left the next one open. The list is kept whole and
/// run against the structural parse, where spelling-independence is a property
/// of the parse rather than of an enumeration. The last three are the round-2
/// finds: they never reached the old scan at all, because it only entered scope
/// on a `[conform]` table HEADER, and header-less forms are ordinary TOML idiom.
#[test]
fn ordinary_toml_spellings_of_unsupported_exclusion_aliases_are_rejected_too() {
    let cases: &[(&str, At, &str, &str)] = &[
        // --- round 1: reached the scan, got past it.
        (
            "sub-table header",
            At::End,
            "\n[conform.skills]\nexclude = [\"cli-surface\"]\n",
            "conform.skills.exclude",
        ),
        (
            "dotted key",
            At::End,
            "\n[conform]\nskills.exclude = [\"cli-surface\"]\n",
            "conform.skills.exclude",
        ),
        (
            "quoted key",
            At::End,
            "\n[conform]\n\"exclude\" = [\"cli-surface\"]\n",
            "conform.exclude",
        ),
        (
            "literal-quoted key",
            At::End,
            "\n[conform]\n'exclude' = [\"cli-surface\"]\n",
            "conform.exclude",
        ),
        // --- round 2 (a): never reached the scan, which only entered scope on a
        // table HEADER. These are ordinary TOML idiom, not exotic spellings.
        (
            "header-less inline table",
            At::Top,
            "conform = { exclude = [\"cli-surface\"] }\n",
            "conform.exclude",
        ),
        (
            "header-less dotted key",
            At::Top,
            "conform.exclude = [\"cli-surface\"]\n",
            "conform.exclude",
        ),
        (
            "header-less dotted sub-table",
            At::Top,
            "conform.skills.exclude = [\"cli-surface\"]\n",
            "conform.skills.exclude",
        ),
        (
            "inline table beside the recognized key",
            At::Top,
            "conform = { local_skills = [\"x\"], exclude = [\"cli-surface\"] }\n",
            "conform.exclude",
        ),
        // --- round 2 (b): a bracket inside a STRING desynchronized a raw depth
        // counter, so every later key was swallowed. One crafted prefix line
        // re-opened the round-1 finding verbatim.
        (
            "bracket in a string value, then exclude",
            At::End,
            "\n[conform]\nlocal_skills = [\"a[\"]\nexclude = [\"cli-surface\"]\n",
            "conform.exclude",
        ),
        (
            "hash in a string value, then exclude",
            At::End,
            "\n[conform]\nlocal_skills = [\"a#b\"]\nexclude = [\"cli-surface\"]\n",
            "conform.exclude",
        ),
        (
            "brace in a string value, then a sub-table",
            At::End,
            "\n[conform]\nlocal_skills = [\"a{\"]\n\n[conform.skills]\nexclude = [\"x\"]\n",
            "conform.skills.exclude",
        ),
        // --- other shapes worth locking.
        (
            "array-of-tables",
            At::End,
            "\n[[conform.x]]\nexclude = [\"cli-surface\"]\n",
            "conform.x",
        ),
        (
            "inline table value under a header",
            At::End,
            "\n[conform]\nskills = { exclude = [\"cli-surface\"] }\n",
            "conform.skills.exclude",
        ),
        (
            "spaced header",
            At::End,
            "\n[ conform ]\nexclude = [\"cli-surface\"]\n",
            "conform.exclude",
        ),
        (
            "quoted header",
            At::End,
            "\n[\"conform\"]\nexclude = [\"cli-surface\"]\n",
            "conform.exclude",
        ),
    ];
    for (label, at, table, expected) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "spelling");
        write_snippet(&root, *at, table);
        let report = audit::audit(&root);
        let r = row(&report, "vendored-skills");
        assert_eq!(
            r.verdict,
            Verdict::Fail,
            "the {label} spelling must be rejected"
        );
        assert!(
            r.diagnostic.contains(expected),
            "the {label} diagnostic must name {expected:?}: {}",
            r.diagnostic
        );
        assert!(!report.ok(), "the {label} spelling must gate the audit");
    }
}

/// A `conform` table BELOW the top level controls nothing, which is exactly what
/// `conform.exclude = [...]` becomes when written after a table header. Silence
/// there would be the same defect one level down, so it is reported.
#[test]
fn a_conform_table_under_another_table_is_reported() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "nested");
    // Appended AFTER [package], so TOML reads it as `package.conform`.
    append_conform_table(&root, "conform.exclude = [\"cli-surface\"]\n");
    let report = audit::audit(&root);
    let r = row(&report, "vendored-skills");
    assert_eq!(r.verdict, Verdict::Fail);
    assert!(
        r.diagnostic.contains("package.conform") && r.diagnostic.contains("controls nothing"),
        "the diagnostic must say where it landed and that it is inert: {}",
        r.diagnostic
    );
}

/// A manifest this tool cannot parse cannot be audited against §8, so the row
/// fails closed and names the parse error. Reading an unreadable file as
/// "declares nothing" would be a silent pass on a MUST row.
#[test]
fn an_unparseable_manifest_fails_the_row_loudly() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "broken");
    std::fs::write(root.join("reef.toml"), "[conform\nlocal_skills = [\n").unwrap();
    let report = audit::audit(&root);
    let r = row(&report, "vendored-skills");
    assert_eq!(r.verdict, Verdict::Fail);
    assert!(
        r.diagnostic.contains("does not parse as TOML"),
        "diag: {}",
        r.diagnostic
    );
    assert!(!report.ok());
}

/// The two parsers used to disagree: `'local_skills'` was RECOGNIZED by the key
/// scan (quote-stripped) but not HONORED by the allowlist scan (literal match),
/// so an author got a row-14 failure telling them to do what they had just done.
/// One structural parse cannot disagree with itself.
#[test]
fn a_non_canonical_spelling_of_local_skills_is_honored_not_just_tolerated() {
    for (label, at, table) in [
        (
            "literal-quoted",
            At::End,
            "\n[conform]\n'local_skills' = [\"domain\"]\n",
        ),
        (
            "basic-quoted",
            At::End,
            "\n[conform]\n\"local_skills\" = [\"domain\"]\n",
        ),
        (
            "header-less inline table",
            At::Top,
            "conform = { local_skills = [\"domain\"] }\n",
        ),
        (
            "header-less dotted",
            At::Top,
            "conform.local_skills = [\"domain\"]\n",
        ),
        (
            "multi-line array",
            At::End,
            "\n[conform]\nlocal_skills = [\n  \"domain\",\n]\n",
        ),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let root = stamp(tmp.path(), "honored");
        write_snippet(&root, at, table);
        // Plant the repo-local domain skill only `local_skills` can legitimize.
        std::fs::create_dir_all(root.join("agent-skills/domain")).unwrap();
        std::fs::write(root.join("agent-skills/domain/SKILL.md"), "# domain\n").unwrap();
        scaffold::materialize_skills(&root).expect("mirror declared local skill");

        let report = audit::audit(&root);
        assert_eq!(
            row(&report, "vendored-skills").verdict,
            Verdict::Pass,
            "the {label} spelling must be honored, not reported: {}",
            row(&report, "vendored-skills").diagnostic
        );

        // And `sync` must agree: the declared skill survives materialization.
        scaffold::materialize_skills(&root).expect("materialize");
        assert!(
            root.join("agent-skills/domain/SKILL.md").is_file(),
            "the {label} spelling must also be honored by sync"
        );
    }
}

/// A sub-table named after the one recognized key is still a shape the contract
/// does not define. With a structural parse the distinction is no longer
/// "sub-table vs inline key" (there are no spellings after parsing) but VALUE
/// TYPE: `conform.local_skills` is an array of strings, so a table there is
/// unrecognized and cannot launder its body past a name-only check.
#[test]
fn a_sub_table_named_after_the_recognized_key_is_still_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "subtable");
    append_conform_table(&root, "\n[conform.local_skills]\nexclude = [\"x\"]\n");
    let report = audit::audit(&root);
    let r = row(&report, "vendored-skills");
    assert_eq!(r.verdict, Verdict::Fail);
    assert!(
        r.diagnostic.contains("conform.local_skills")
            && r.diagnostic.contains("expected an array of strings"),
        "the diagnostic must name the key and the shape it required: {}",
        r.diagnostic
    );
}

#[test]
fn the_recognized_conform_key_still_passes() {
    // Negative parity for the unknown-key check: `local_skills` is contract
    // surface (chelis#651) and must keep working, including beside a real
    // repo-local skill dir.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "local");
    append_conform_table(&root, "\n[conform]\nlocal_skills = [\"chelis-std\"]\n");
    std::fs::create_dir_all(root.join("agent-skills/chelis-std")).unwrap();
    std::fs::write(
        root.join("agent-skills/chelis-std/SKILL.md"),
        "# chelis-std\n",
    )
    .unwrap();
    scaffold::materialize_skills(&root).expect("mirror declared local skill");

    let report = audit::audit(&root);
    assert_eq!(row(&report, "vendored-skills").verdict, Verdict::Pass);
    assert!(report.ok());
}

#[test]
fn a_commented_out_exclusion_is_not_a_declaration() {
    // A shell writing down the decision it did NOT take (or a reviewer leaving
    // the rejected shape in a comment) must not be failed for it.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "commented");
    append_conform_table(
        &root,
        "\n[conform]\n# exclude = [\"cli-surface\"]  # rejected: see contract §8\nlocal_skills = []\n",
    );
    let report = audit::audit(&root);
    assert_eq!(row(&report, "vendored-skills").verdict, Verdict::Pass);
}

// --------------------------------------------- the sanctioned alternative works

#[test]
fn recording_non_applicability_with_a_shell_local_block_passes_and_survives_sync() {
    // The route §8 points shells at instead of an exclusion. It must (a) audit
    // green, (b) survive a sync, and (c) still be there after the skill's
    // toolchain-owned body is regenerated underneath it.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "annotated");
    let skill = root.join("agent-skills/cli-surface/SKILL.md");
    let body = std::fs::read_to_string(&skill).unwrap();
    let note = "<!-- shell-local:begin -->\nNot applicable: this shell has no Chelis CLI surface.\n<!-- shell-local:end -->\n";
    std::fs::write(&skill, format!("{body}\n{note}")).unwrap();
    scaffold::materialize_skills(&root).expect("mirror shell-local skill edit");

    assert!(
        audit::audit(&root).ok(),
        "the sanctioned way to record non-applicability must audit green"
    );
    scaffold::materialize_skills(&root).expect("materialize");
    let after = std::fs::read_to_string(&skill).unwrap();
    assert!(
        after.contains("Not applicable: this shell has no Chelis CLI surface."),
        "the shell-local note must survive sync"
    );
    assert!(audit::audit(&root).ok(), "and stay green afterwards");
}

#[test]
fn the_uniform_set_is_exactly_the_pinned_set() {
    // The set the contract calls uniform is the toolchain's embedded set. Each
    // agent surface gets a real materialized copy so discovery does not depend
    // on whether that agent follows a directory symlink.
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "uniform");
    let mut expected: Vec<String> = skills::SHARED_SKILLS
        .iter()
        .map(|s| s.to_string())
        .collect();
    expected.sort();

    for rel in ["agent-skills", ".claude/skills", ".codex/skills"] {
        let surface = root.join(rel);
        assert!(surface.is_dir(), "{rel} must be a directory");
        assert!(!surface.is_symlink(), "{rel} must be a materialized copy");
        let mut present: Vec<String> = std::fs::read_dir(&surface)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        present.sort();
        assert_eq!(present, expected, "{rel}");
        for skill in skills::SHARED_SKILLS {
            let actual = std::fs::read_to_string(surface.join(skill).join("SKILL.md")).unwrap();
            let canonical =
                std::fs::read_to_string(root.join("agent-skills").join(skill).join("SKILL.md"))
                    .unwrap();
            assert_eq!(actual, canonical, "{rel}/{skill}/SKILL.md");
        }
    }
}

#[test]
fn sync_repairs_each_materialized_agent_skill_surface() {
    let tmp = tempfile::tempdir().unwrap();
    let root = stamp(tmp.path(), "mirrors");

    std::fs::write(
        root.join(".codex/skills/spec-sync/SKILL.md"),
        "forked codex copy\n",
    )
    .unwrap();
    std::fs::remove_dir_all(root.join(".claude/skills/cli-surface")).unwrap();
    std::fs::create_dir_all(root.join(".claude/skills/rogue")).unwrap();
    std::fs::write(root.join(".claude/skills/rogue/SKILL.md"), "rogue copy\n").unwrap();

    let report = audit::audit(&root);
    assert_eq!(
        report
            .rows
            .iter()
            .find(|row| row.key == "vendored-skills")
            .unwrap()
            .verdict,
        audit::Verdict::Fail
    );

    scaffold::materialize_skills(&root).expect("repair materialized skill surfaces");
    assert!(audit::audit(&root).ok());
    assert!(!root.join(".claude/skills/rogue").exists());
    assert_eq!(
        std::fs::read_to_string(root.join(".codex/skills/spec-sync/SKILL.md")).unwrap(),
        std::fs::read_to_string(root.join("agent-skills/spec-sync/SKILL.md")).unwrap()
    );
    assert_eq!(
        std::fs::read_to_string(root.join(".claude/skills/cli-surface/SKILL.md")).unwrap(),
        std::fs::read_to_string(root.join("agent-skills/cli-surface/SKILL.md")).unwrap()
    );
}
