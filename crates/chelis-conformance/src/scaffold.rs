//! Shell scaffolding: the core of `conform init` (stamp a new/retrofit shell)
//! and the skill-materialization + managed-block regeneration used by
//! `conform sync`.
//!
//! `scaffold` writes a minimal but fully-conformant shell tree: it is the
//! executable form of the §11 bootstrap order, and the audit's negative-parity
//! oracle stamps a shell with it, confirms it is green, then mutates it. Symlink
//! creation is unix-only (the supported dev/CI platforms).

use std::fs;
use std::path::Path;

use crate::{canonical, managed_block, skills};

/// Write a fully-conformant shell tree rooted at `root`, pinned to `version`.
/// Existing files are overwritten (idempotent re-stamp). `name` is the package
/// name; `module_prefix` its module namespace (PascalCase).
pub fn scaffold(root: &Path, name: &str, module_prefix: &str, version: &str) -> Result<(), String> {
    fs::create_dir_all(root).map_err(|e| format!("create {}: {e}", root.display()))?;

    write(root, "reef.toml", &reef_toml(name, module_prefix, version))?;
    write(root, "AGENTS.md", &agents_md(name, version))?;
    symlink_file(root, "AGENTS.md", "CLAUDE.md")?;

    write(root, "docs/CHELIS_SURFACE.md", &chelis_surface(version))?;
    write(root, "docs/UPSTREAM_BUGS.md", UPSTREAM_BUGS)?;
    write(root, "docs/issue_drafts/README.md", ISSUE_DRAFTS_README)?;

    write(
        root,
        "src/main.ch",
        &format!(
            "module {module_prefix}.Main\n\ndef noop() -> unit = test_assert(true, \"noop\")\n"
        ),
    )?;

    write(
        root,
        "tests_neg/example/rejects.ch",
        &format!(
            "module {module_prefix}.TestsNeg.Rejects\n\n\
             def test_neg_example() -> unit = test_assert(false, \"example negative case\")\n"
        ),
    )?;
    write(
        root,
        "tests_neg/example/rejects.expect",
        "example negative case\n",
    )?;

    write(root, "tests_blocked/README.md", TESTS_BLOCKED_README)?;

    write(root, ".github/workflows/ci.yml", &ci_yml(version))?;
    write(root, ".github/workflows/bump-pr.yml", BUMP_PR_YML)?;

    materialize_skills(root)?;

    Ok(())
}

/// Fence markers for a trailing shell-local override block inside a materialized
/// shared `SKILL.md` (chelis#653). Everything from [`SHELL_LOCAL_BEGIN`] to end
/// of file is shell-owned: `sync` preserves it verbatim and re-generates only the
/// toolchain-owned managed span above it, so upstream skill edits still propagate
/// while the shell's override survives.
pub(crate) const SHELL_LOCAL_BEGIN: &str = "<!-- shell-local:begin -->";
pub(crate) const SHELL_LOCAL_END: &str = "<!-- shell-local:end -->";

/// Split a materialized `SKILL.md` into its toolchain-owned managed span and an
/// optional trailing shell-local block (from the first [`SHELL_LOCAL_BEGIN`] to
/// end of file, markers included). §8 byte-checks only the managed span.
pub(crate) fn split_shell_local(content: &str) -> (&str, Option<&str>) {
    match content.find(SHELL_LOCAL_BEGIN) {
        Some(i) => (&content[..i], Some(&content[i..])),
        None => (content, None),
    }
}

/// Materialize `agent-skills/` from the embedded pinned skill set and wire the
/// `.claude`/`.codex` skill-dir symlinks. Shared by `init` and `sync`.
///
/// Preserves any trailing shell-local block in each shared `SKILL.md` (chelis#653)
/// and repo-local domain skills declared in `[conform] local_skills` (chelis#651).
/// Returns human-readable notices for the caller to surface: an upstream body that
/// changed underneath a shell-local override, and any un-materialized skill it
/// pruned.
pub fn materialize_skills(root: &Path) -> Result<Vec<String>, String> {
    // The `[conform] local_skills` allowlist, read from the PARSED manifest so
    // `sync` and `audit` cannot disagree about what the shell declared
    // (chelis#1262). An absent reef.toml declares nothing; a PRESENT one that
    // does not parse is an error, not an empty declaration, because silently
    // reading it as empty would prune a repo-local skill the shell did declare.
    let local_skills = match fs::read_to_string(root.join("reef.toml")) {
        Ok(text) => crate::conform::parse(&text)
            .map_err(|e| {
                format!("reef.toml does not parse as TOML, so `[conform] local_skills` cannot be read: {e}")
            })?
            .local_skills,
        Err(_) => Vec::new(),
    };
    let mut notices = Vec::new();
    // Sampled ONCE, before the loop: the first written skill creates
    // `agent-skills/`, so testing it inside the loop would report every skill
    // after the first as "restored" on a fresh `init`. A tree with no
    // `agent-skills/` at all is being materialized for the first time, which is
    // not a restoration and gets no notice.
    let skills_dir_existed = root.join("agent-skills").is_dir();

    for (skill_name, body) in skills::EMBEDDED_SKILLS {
        let rel = format!("agent-skills/{skill_name}/SKILL.md");
        let existing = fs::read_to_string(root.join(&rel)).ok();
        let block = existing
            .as_deref()
            .and_then(|c| split_shell_local(c).1.map(str::to_string));
        // The on-disk managed span IS the base the block was written against
        // (§8 keeps it byte-equal to the previous toolchain). If it diverges from
        // the new embedded body while a block is present, flag it so the agent
        // re-checks the override against the propagated upstream text.
        if let Some(prev) = &existing {
            let (prev_managed, prev_block) = split_shell_local(prev);
            if prev_block.is_some() && prev_managed.trim_end() != body.trim_end() {
                notices.push(format!(
                    "{skill_name}: upstream skill body changed and a shell-local override is present; re-check it"
                ));
            }
        }
        // A shared skill that was ABSENT is being (re)created. Say so. §8 makes
        // the set uniform with no exclusion control (chelis#1262), so a shell
        // that pruned a skill it judged inapplicable gets it back on the next
        // sync; restoring it silently is what let that decision degrade into
        // AGENTS.md lore the tree then contradicted. The notice names the
        // sanctioned alternative at the moment the author is looking.
        if existing.is_none() && skills_dir_existed {
            notices.push(format!(
                "{skill_name}: re-materialized (the shared set is uniform; §8 has no exclusion control). To record that it does not fit this shell, append a shell-local block to its SKILL.md rather than deleting it."
            ));
        }
        let content = match &block {
            Some(b) => format!(
                "{}\n{}",
                body.trim_end(),
                b.trim_start_matches(['\n', '\r'])
            ),
            None => (*body).to_string(),
        };
        write(root, &rel, &content)?;
    }

    // Record the pointer manifest so the shell commits a stamp, not opaque bytes.
    let mut manifest = String::from(
        "# Materialized from the pinned chelis toolchain's embedded skill set.\n\
         # Do not edit skill bodies here; run `chelis reef conform sync`.\n\n\
         skills = [\n",
    );
    for name in skills::SHARED_SKILLS {
        manifest.push_str(&format!("  \"{name}\",\n"));
    }
    manifest.push_str("]\n");
    write(root, "agent-skills/UPSTREAM.toml", &manifest)?;

    // agent-skills/ is toolchain-owned apart from the declared escape hatches, so
    // remove anything else the pinned set did not just write — a forked/renamed
    // skill dir, an extra file inside a shared skill dir, or a stray top-level
    // file. Repo-local domain skills in `local_skills` are spared.
    for name in prune_skill_drift(root, &local_skills)? {
        notices.push(format!(
            "removed un-materialized skill agent-skills/{name}/ (not a shared skill; add it to [conform] local_skills to keep it)"
        ));
    }

    symlink_dir(root, "../agent-skills", ".claude/skills")?;
    symlink_dir(root, "../agent-skills", ".codex/skills")?;
    Ok(notices)
}

/// Remove any content under `agent-skills/` that the pinned embedded set does
/// not own: shared-skill dirs outside [`skills::SHARED_SKILLS`], files other than
/// `SKILL.md` inside a pinned skill dir, and top-level files other than
/// `UPSTREAM.toml`. Repo-local domain skills named in `local_skills` (chelis#651)
/// are preserved. Returns the names of the *pruned skill dirs* only — an
/// author-added skill dir the caller should warn about before it vanishes
/// (chelis#651). Stray non-`SKILL.md` files and top-level files are still removed
/// but are not surfaced: the "add it to `[conform] local_skills`" hint applies
/// only to whole skill dirs, and materialize never creates such files, so their
/// presence is leftover/editor cruft, not author-authored content.
fn prune_skill_drift(root: &Path, local_skills: &[String]) -> Result<Vec<String>, String> {
    let dir = root.join("agent-skills");
    let mut pruned_dirs = Vec::new();
    let Ok(entries) = fs::read_dir(&dir) else {
        return Ok(pruned_dirs);
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let path = e.path();
        if is_dir {
            if skills::SHARED_SKILLS.contains(&name.as_str()) {
                if let Ok(inner) = fs::read_dir(&path) {
                    for f in inner.flatten() {
                        if f.file_name().to_string_lossy() != "SKILL.md" {
                            remove_path(&f.path())?;
                        }
                    }
                }
            } else if local_skills.iter().any(|s| s == &name) {
                // Repo-local domain skill: shell-owned, preserved (chelis#651).
            } else {
                fs::remove_dir_all(&path)
                    .map_err(|err| format!("remove {}: {err}", path.display()))?;
                pruned_dirs.push(name);
            }
        } else if name != "UPSTREAM.toml" {
            fs::remove_file(&path).map_err(|err| format!("remove {}: {err}", path.display()))?;
        }
    }
    Ok(pruned_dirs)
}

fn remove_path(path: &Path) -> Result<(), String> {
    let res = if path.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    res.map_err(|e| format!("remove {}: {e}", path.display()))
}

/// The documents `conform sync` / `conform bump` **restamp in place** and
/// therefore cannot create: each is read, edited, and written back. Distinct
/// from `agent-skills/`, which the write path *materializes* from the embedded
/// set, so its absence is normal work rather than a missing prerequisite.
///
/// `reef.toml` is listed because [`crate::bump::rewrite_pins`] reads its
/// `compiler` pin before rewriting anything, and because both verbs derive the
/// version they stamp from that pin.
const RESTAMPED_ARTIFACTS: &[(&str, &str)] = &[
    ("reef.toml", "the compiler pin `conform bump` rewrites"),
    (
        "AGENTS.md",
        "carries the `agents-inheritance` managed block sync restamps",
    ),
    (
        "docs/CHELIS_SURFACE.md",
        "carries the `chelis-surface-header` managed block sync restamps",
    ),
];

/// One prerequisite the write path cannot proceed without.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightGap {
    /// Repo-relative path of the artifact.
    pub rel: &'static str,
    /// Why the verb needs it, and what is wrong with it here.
    pub reason: String,
}

/// Preflight the write path on a shell root: every artifact `sync`/`bump`
/// restamps must already exist, and `reef.toml` must carry a pin they can read
/// (chelis#1263).
///
/// **Why this is a separate pass rather than better error handling.** `bump`'s
/// edit sequence is repin -> materialize skills -> restamp blocks, and each step
/// writes as it goes. On a never-conformed repo the sequence used to die
/// partway: measured on chelis 0.18.5, a tree missing `AGENTS.md` came back with
/// `reef.toml` and `ci.yml` already repinned to the new version and
/// `agent-skills/` fully materialized, and a tree missing
/// `docs/CHELIS_SURFACE.md` came back with all of that *plus* a restamped
/// `AGENTS.md`. Both then failed. A nonzero exit is the right verdict, but a
/// half-bumped tree is not a state any caller asked for, and the wave also
/// observed the same shapes reported as success. Checking every prerequisite
/// before the first write makes "nothing was written" a structural fact instead
/// of a claim, so there is nothing to roll back or enumerate.
///
/// **Existence is not enough for `reef.toml`.** A shell whose pin is unreadable
/// (a truncated file, a hand-edit that lost the `=`, a range instead of an exact
/// pin) has no version for `sync` to stamp. `sync` used to fall back to the
/// *toolchain's own* version there and stamp the managed blocks with it, exit 0,
/// which is the same defect class in a quieter form: the shell then carries
/// blocks claiming a version it never adopted. Requiring a parseable pin makes
/// the version the shell's, always.
pub fn preflight_restamp_targets(root: &Path) -> Result<(), Vec<PreflightGap>> {
    let mut gaps = Vec::new();
    for (rel, why) in RESTAMPED_ARTIFACTS.iter().copied() {
        if !root.join(rel).is_file() {
            gaps.push(PreflightGap {
                rel,
                reason: format!("missing: {why}"),
            });
        } else if rel == "reef.toml" {
            let text = fs::read_to_string(root.join(rel)).unwrap_or_default();
            if let Err(e) = crate::conform::parse(&text) {
                // A manifest this tool cannot parse is not a manifest it may
                // half-apply a bump to: `materialize_skills` reads the
                // `[conform] local_skills` allowlist out of it, so proceeding
                // would prune a repo-local skill the shell did declare.
                gaps.push(PreflightGap {
                    rel,
                    reason: format!("does not parse as TOML: {e}"),
                });
            } else if crate::audit::parse_compiler_pin(&text).is_none() {
                gaps.push(PreflightGap {
                    rel,
                    reason: "has no readable `compiler = \"=X.Y.Z\"` pin, so there is no version \
                             to stamp the managed blocks with"
                        .to_string(),
                });
            }
        }
    }
    if gaps.is_empty() { Ok(()) } else { Err(gaps) }
}

/// Render [`preflight_restamp_targets`]'s failure as the message the CLI prints
/// before exiting nonzero. Lives here so `sync` and `bump` cannot drift into two
/// different explanations of the same refusal.
pub fn preflight_failure_message(verb: &str, root: &Path, gaps: &[PreflightGap]) -> String {
    let mut out = format!(
        "conform {verb} refuses to run on {}: it restamps files in place, and these are missing or unusable:\n",
        root.display()
    );
    for gap in gaps {
        out.push_str(&format!("  {}  ({})\n", gap.rel, gap.reason));
    }
    out.push_str(
        "Nothing was written. This repo is not conformed yet. For a NEW shell, stamp it with \
         `chelis reef conform init <name> --module-prefix <Prefix> --output <path>`. On an \
         EXISTING repo, repair the files above by hand first, then re-run this command: \
         `conform init` writes the WHOLE scaffold from its templates and overwrites every file \
         it owns, including reef.toml, AGENTS.md, docs/, .github/workflows/, src/main.ch, \
         tests_neg/, and tests_blocked/, so pointing it at a repo with real source loses that \
         source.",
    );
    out
}

/// Regenerate every managed block in the shell's documents to `version` (the
/// pointer half of `conform sync`). Only fenced regions are touched.
pub fn sync_managed_blocks(root: &Path, version: &str) -> Result<(), String> {
    // AGENTS.md :: agents-inheritance
    resync_block(
        root,
        "AGENTS.md",
        "agents-inheritance",
        version,
        managed_block::Anchor::AfterHeading("## Repo Identity"),
    )?;
    // docs/CHELIS_SURFACE.md :: chelis-surface-header
    resync_block(
        root,
        "docs/CHELIS_SURFACE.md",
        "chelis-surface-header",
        version,
        managed_block::Anchor::Top,
    )?;
    Ok(())
}

fn resync_block(
    root: &Path,
    rel: &str,
    id: &str,
    version: &str,
    anchor: managed_block::Anchor,
) -> Result<(), String> {
    let path = root.join(rel);
    let existing =
        fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let body = canonical::body(id).ok_or_else(|| format!("no canonical body for block {id:?}"))?;
    let updated = managed_block::upsert(&existing, id, version, body, anchor);
    fs::write(&path, updated).map_err(|e| format!("write {}: {e}", path.display()))
}

// ---------------------------------------------------------------- templates

fn reef_toml(name: &str, module_prefix: &str, version: &str) -> String {
    format!(
        "[package]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         compiler = \"={version}\"\n\
         module_prefix = \"{module_prefix}\"\n"
    )
}

fn agents_md(name: &str, version: &str) -> String {
    let block = managed_block::render(
        "agents-inheritance",
        version,
        canonical::body("agents-inheritance").expect("embedded canonical body"),
    );
    format!(
        "# {name}, a Chelis shell\n\n\
         ## Repo Identity\n\n\
         State this shell's intent (what users can ultimately do with it) and link a\n\
         vision doc. A deliverables checklist is not an intent statement.\n\n\
         {block}\n\
         ## Toolchain Policy\n\n\
         Install the pinned toolchain via `chelisup`; never hand-symlink a machine-global\n\
         default. Python is uv-managed. See the managed block above for the upstream contract.\n\n\
         ## Pin Bump Checklist\n\n\
         A pin bump is a de-narrowing event. Bump only through a `chelis reef conform bump`\n\
         PR that runs the blocked-probe suite, the staleness/narrowing audit, and restamps\n\
         `docs/CHELIS_SURFACE.md`. Never edit the pin directly on `main`.\n\n\
         ## Scaffolding Drift Rule\n\n\
         All shells share one scaffolding shape. Mirror any structural change into the\n\
         sibling shells in the same change set, or record a per-repo divergence. Contract\n\
         changes land upstream first (`Chelis-Lang/chelis`) and propagate here via\n\
         `chelis reef conform sync`.\n"
    )
}

fn chelis_surface(version: &str) -> String {
    let block = managed_block::render(
        "chelis-surface-header",
        version,
        canonical::body("chelis-surface-header").expect("embedded canonical body"),
    );
    format!(
        "# Chelis Capability Surface (this shell)\n\n\
         {block}\n\
         ## Capabilities\n\n\
         | Capability | Status |\n\
         |---|---|\n\
         | (fill in the primitive/builtin families this shell's domain touches) | `@pin` |\n\
         | (capabilities landing next bump) | `@upstream` |\n"
    )
}

const UPSTREAM_BUGS: &str = "# Upstream Bugs\n\n\
    Track suspected chelis bugs and capability gaps here. File upstream and cite by\n\
    `chelis#NNN` (never a prose name). Re-probe at every pin bump.\n\n\
    ## Actively blocking\n\n(none yet)\n\n\
    ## Tracking\n\n(none yet)\n\n\
    ## Parked\n\n(none yet)\n\n\
    ## Archived\n\n(none yet)\n";

/// Shell-driven scheduled bump PR. Detects a newer chelis release, runs
/// `conform bump` on a branch, and opens a PR whose CI runs the full checklist —
/// so a bump that breaks the shell is a red PR, never a broken `main`. Uses a
/// dynamic version (no static `CHELIS_VERSION` env) so it does not trip the
/// pin-consistency audit. Requires branch protection on `main` to close the
/// direct-push door.
const BUMP_PR_YML: &str = r#"name: chelis pin bump
on:
  schedule:
    - cron: "0 8 * * 1"   # weekly; adjust to your cadence
  workflow_dispatch:

permissions:
  contents: write
  pull-requests: write

jobs:
  bump:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - name: Resolve latest chelis release
        id: latest
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          latest="$(gh release view --repo Chelis-Lang/chelis --json tagName -q .tagName | sed 's/^v//')"
          echo "version=${latest}" >> "$GITHUB_OUTPUT"
      - name: Install chelis toolchain
        run: chelisup install "${{ steps.latest.outputs.version }}"
      - name: Run the Pin Bump Checklist
        run: chelis reef conform bump "${{ steps.latest.outputs.version }}"
      - name: Open bump PR
        uses: peter-evans/create-pull-request@v6
        with:
          branch: chelis-bump/${{ steps.latest.outputs.version }}
          title: "chelis pin bump -> ${{ steps.latest.outputs.version }}"
          body: |
            Mechanical `chelis reef conform bump` change set. CI runs the full
            checklist (conform audit + blocked/negative probes); do not merge red.
          commit-message: "chore: bump chelis pin to ${{ steps.latest.outputs.version }}"
"#;

const ISSUE_DRAFTS_README: &str = "# Parked upstream issue drafts\n\n\
    Ready-to-file `chelis#` drafts with their filing condition. Cite the draft path\n\
    at the narrowing site until it is filed and gets a number.\n";

const TESTS_BLOCKED_README: &str = "# Blocked-probe suite\n\n\
    One expected-to-fail reproducer per open upstream blocker\n\
    (`<area>/<name>.ch` + `<name>.expect`; line 1 = pinned diagnostic substring,\n\
    lines 2+ = the `chelis#NNN` citation + on-pass de-narrowing instructions).\n\
    Run with `chelis test tests_blocked/ --expect blocked`. Blockers the harness\n\
    cannot express (check-context-only, cross-module) are listed here for manual\n\
    re-probe.\n";

fn ci_yml(version: &str) -> String {
    format!(
        r#"name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CHELIS_TAG: v{version}
  CHELIS_VERSION: {version}

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with:
          # Full history so `conform bump-check --base origin/main` can resolve
          # the base pin. The default shallow clone does not fetch origin/main,
          # which would make the pin-bump guard fail closed.
          fetch-depth: 0
      - name: Install chelis toolchain
        run: chelisup install {version}
      - name: Conformance audit
        run: chelis reef conform audit
      - name: Pin-bump guard
        run: chelis reef conform bump-check --base origin/main
      - name: Negative tests
        run: chelis test tests_neg/ --expect neg
"#
    )
}

// ---------------------------------------------------------------- fs helpers

fn write(root: &Path, rel: &str, contents: &str) -> Result<(), String> {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    fs::write(&path, contents).map_err(|e| format!("write {}: {e}", path.display()))
}

#[cfg(unix)]
fn symlink_file(root: &Path, target: &str, link_rel: &str) -> Result<(), String> {
    symlink_generic(root, target, link_rel)
}

#[cfg(unix)]
fn symlink_dir(root: &Path, target: &str, link_rel: &str) -> Result<(), String> {
    let link = root.join(link_rel);
    if let Some(parent) = link.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    symlink_generic(root, target, link_rel)
}

#[cfg(unix)]
fn symlink_generic(root: &Path, target: &str, link_rel: &str) -> Result<(), String> {
    let link = root.join(link_rel);
    // Idempotent: replace an existing symlink/file at the link path.
    let _ = fs::remove_file(&link);
    std::os::unix::fs::symlink(target, &link)
        .map_err(|e| format!("symlink {} -> {target}: {e}", link.display()))
}

#[cfg(not(unix))]
fn symlink_file(_root: &Path, _target: &str, _link_rel: &str) -> Result<(), String> {
    Err("conform init/sync requires a unix platform for symlinks".to_string())
}

#[cfg(not(unix))]
fn symlink_dir(_root: &Path, _target: &str, _link_rel: &str) -> Result<(), String> {
    Err("conform init/sync requires a unix platform for symlinks".to_string())
}
