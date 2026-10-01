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

use pulldown_cmark::{Event, Parser, Tag};

use crate::links::{LocalTargets, pin_links};
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

    materialize_skills(root, version)?;

    Ok(())
}

/// Fence markers for a trailing shell-local override block inside a materialized
/// shared `SKILL.md` (chelis#653). Everything from [`SHELL_LOCAL_BEGIN`] to end
/// of file is shell-owned: `sync` preserves it verbatim and re-generates only the
/// toolchain-owned managed span above it, so upstream skill edits still propagate
/// while the shell's override survives.
pub(crate) const SHELL_LOCAL_BEGIN: &str = "<!-- shell-local:begin -->";
pub(crate) const SHELL_LOCAL_END: &str = "<!-- shell-local:end -->";
pub(crate) const SHELL_LOCAL_EXCLUDE_BEGIN: &str = "<!-- shell-local:exclude:begin -->";
pub(crate) const SHELL_LOCAL_EXCLUDE_END: &str = "<!-- shell-local:exclude:end -->";

/// Split a materialized `SKILL.md` into its toolchain-owned managed span and an
/// optional trailing shell-local block (from the first [`SHELL_LOCAL_BEGIN`] to
/// end of file, markers included). §8 byte-checks only the managed span.
pub(crate) fn split_shell_local(content: &str) -> (&str, Option<&str>) {
    match content.find(SHELL_LOCAL_BEGIN) {
        Some(i) => (&content[..i], Some(&content[i..])),
        None => (content, None),
    }
}

/// Apply the subtractive part of a trailing shell-local block to an embedded
/// upstream skill body. Each selector is an exact ATX heading; its complete
/// section ends at the next heading of equal or shallower depth.
pub(crate) fn apply_shell_local_exclusions(
    upstream: &str,
    block: Option<&str>,
) -> Result<String, String> {
    let Some(block) = block else {
        return Ok(upstream.to_string());
    };
    let selectors = shell_local_exclusion_selectors(block)?;
    apply_heading_exclusions(upstream, selectors)
}

/// Apply standalone shell-owned heading exclusions to the canonical body of
/// the managed block `block_id` in `document` (the `agents-inheritance` block in
/// `AGENTS.md`, the `chelis-surface` block in `docs/CHELIS_SURFACE.md`).
/// Selectors live outside the managed block so sync can replace the entire
/// upstream document while preserving the shell's controls and prose.
pub(crate) fn apply_document_exclusions(
    upstream: &str,
    document: &str,
    block_id: &str,
) -> Result<String, String> {
    let shell_owned = match managed_block::find(document, block_id) {
        Some(block) => format!("{}{}", &document[..block.span.0], &document[block.span.1..]),
        None => document.to_string(),
    };
    let selectors = standalone_exclusion_selectors(&shell_owned)?;
    apply_heading_exclusions(upstream, selectors)
}

fn apply_heading_exclusions(upstream: &str, selectors: Vec<&str>) -> Result<String, String> {
    if selectors.is_empty() {
        return Ok(upstream.to_string());
    }

    #[derive(Clone)]
    struct Heading<'a> {
        start: usize,
        level: usize,
        line: &'a str,
    }

    // Derive heading structure from a CommonMark parser rather than scanning
    // line prefixes. That makes fenced code, HTML blocks, and other Markdown
    // containers structurally incapable of supplying a selector boundary.
    let headings: Vec<_> = Parser::new(upstream)
        .into_offset_iter()
        .filter_map(|(event, range)| {
            let Event::Start(Tag::Heading { level, .. }) = event else {
                return None;
            };
            let start = upstream[..range.start]
                .rfind('\n')
                .map_or(0, |newline| newline + 1);
            let end = upstream[range.start..]
                .find('\n')
                .map_or(upstream.len(), |newline| range.start + newline);
            let line = upstream[start..end].trim_end_matches('\r');
            let atx_level = atx_heading_level(line)?;
            (atx_level == level as usize).then_some(Heading {
                start,
                level: atx_level,
                line,
            })
        })
        .collect();

    let mut ranges = Vec::new();
    for selector in selectors {
        let matches: Vec<usize> = headings
            .iter()
            .enumerate()
            .filter_map(|(i, heading)| (heading.line == selector).then_some(i))
            .collect();
        match matches.as_slice() {
            [] => {
                return Err(format!(
                    "shell-local exclusion selector {selector:?} matches no upstream heading"
                ));
            }
            [index] => {
                let heading = &headings[*index];
                let end = headings[*index + 1..]
                    .iter()
                    .find(|next| next.level <= heading.level)
                    .map_or(upstream.len(), |next| next.start);
                ranges.push((heading.start, end, selector));
            }
            _ => {
                return Err(format!(
                    "shell-local exclusion selector {selector:?} matches multiple upstream headings"
                ));
            }
        }
    }
    ranges.sort_by_key(|(start, _, _)| *start);
    for pair in ranges.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(format!(
                "shell-local exclusion selectors {:?} and {:?} overlap",
                pair[0].2, pair[1].2
            ));
        }
    }

    let mut filtered = String::with_capacity(upstream.len());
    let mut cursor = 0;
    for (start, end, _) in ranges {
        filtered.push_str(&upstream[cursor..start]);
        cursor = end;
    }
    filtered.push_str(&upstream[cursor..]);
    Ok(filtered)
}

/// Parse the optional exclusion span used in a document's shell-owned text.
fn standalone_exclusion_selectors(document: &str) -> Result<Vec<&str>, String> {
    let begin_count = document.matches(SHELL_LOCAL_EXCLUDE_BEGIN).count();
    let end_count = document.matches(SHELL_LOCAL_EXCLUDE_END).count();
    if begin_count == 0 && end_count == 0 {
        return Ok(Vec::new());
    }
    if begin_count != 1 {
        return Err(format!(
            "malformed shell-local exclusion block (expected exactly one `{SHELL_LOCAL_EXCLUDE_BEGIN}`)"
        ));
    }
    if end_count != 1 {
        return Err(format!(
            "malformed shell-local exclusion block (expected exactly one `{SHELL_LOCAL_EXCLUDE_END}`)"
        ));
    }
    let begin = standalone_markdown_comment_offset(document, SHELL_LOCAL_EXCLUDE_BEGIN)?;
    let end = standalone_markdown_comment_offset(document, SHELL_LOCAL_EXCLUDE_END)?;
    if end <= begin + SHELL_LOCAL_EXCLUDE_BEGIN.len() {
        return Err("shell-local exclusion end marker must follow its begin marker".into());
    }
    parse_exclusion_selector_lines(
        &document[begin + SHELL_LOCAL_EXCLUDE_BEGIN.len()..end],
        true,
    )
}

/// Validate the outer shell-local suffix and return its exact section-heading
/// exclusion selectors.
fn shell_local_exclusion_selectors(block: &str) -> Result<Vec<&str>, String> {
    if block.matches(SHELL_LOCAL_BEGIN).count() != 1 || !block.starts_with(SHELL_LOCAL_BEGIN) {
        return Err(format!(
            "malformed shell-local block (expected exactly one leading `{SHELL_LOCAL_BEGIN}`)"
        ));
    }
    if block.matches(SHELL_LOCAL_END).count() != 1 {
        return Err(format!(
            "malformed shell-local block (expected exactly one `{SHELL_LOCAL_END}`)"
        ));
    }
    let outer_end = block.find(SHELL_LOCAL_END).expect("count checked");
    if !block[outer_end + SHELL_LOCAL_END.len()..].trim().is_empty() {
        return Err(format!(
            "content after `{SHELL_LOCAL_END}` (the shell-local block must be the file suffix)"
        ));
    }

    let begin_count = block.matches(SHELL_LOCAL_EXCLUDE_BEGIN).count();
    let end_count = block.matches(SHELL_LOCAL_EXCLUDE_END).count();
    if begin_count == 0 && end_count == 0 {
        return Ok(Vec::new());
    }
    if begin_count != 1 {
        return Err(format!(
            "malformed shell-local exclusion block (expected exactly one `{SHELL_LOCAL_EXCLUDE_BEGIN}`)"
        ));
    }
    if end_count != 1 {
        return Err(format!(
            "malformed shell-local exclusion block (expected exactly one `{SHELL_LOCAL_EXCLUDE_END}`)"
        ));
    }
    let begin = standalone_markdown_comment_offset(block, SHELL_LOCAL_EXCLUDE_BEGIN)?;
    let end = standalone_markdown_comment_offset(block, SHELL_LOCAL_EXCLUDE_END)?;
    if begin <= SHELL_LOCAL_BEGIN.len()
        || end <= begin + SHELL_LOCAL_EXCLUDE_BEGIN.len()
        || end >= outer_end
    {
        return Err(
            "shell-local exclusion block must be inside the outer shell-local block".into(),
        );
    }

    parse_exclusion_selector_lines(&block[begin + SHELL_LOCAL_EXCLUDE_BEGIN.len()..end], false)
}

/// Locate a control marker only when Markdown parses it as its own HTML
/// comment. Raw text search still owns missing/duplicate diagnostics; this
/// check prevents examples in code fences or enclosing HTML blocks from
/// becoming active configuration.
fn standalone_markdown_comment_offset(document: &str, marker: &str) -> Result<usize, String> {
    let offset = document.find(marker).expect("marker count checked");
    let marker_end = offset + marker.len();
    let line_start = document[..offset]
        .rfind('\n')
        .map_or(0, |newline| newline + 1);
    let line_end = document[marker_end..]
        .find('\n')
        .map_or(document.len(), |newline| marker_end + newline);
    let alone_on_line = document[line_start..line_end].trim_end_matches('\r').trim() == marker;
    let own_html_event = Parser::new(document)
        .into_offset_iter()
        .any(|(event, range)| {
            matches!(event, Event::Start(Tag::HtmlBlock))
                && range.start <= offset
                && marker_end <= range.end
                && document[range].trim() == marker
        });
    if !alone_on_line || !own_html_event {
        return Err(format!(
            "`{marker}` must be a standalone Markdown comment outside fenced code and enclosing HTML blocks"
        ));
    }
    Ok(offset)
}

fn parse_exclusion_selector_lines(
    body: &str,
    allow_document_title: bool,
) -> Result<Vec<&str>, String> {
    let mut selectors = Vec::new();
    for line in body.lines() {
        let directive = line.trim();
        if directive.is_empty() {
            continue;
        }
        let Some(selector) = directive
            .strip_prefix("<!-- ")
            .and_then(|line| line.strip_suffix(" -->"))
            .map(str::trim)
        else {
            return Err(format!(
                "shell-local exclusion selector {directive:?} must be an HTML-comment-wrapped ATX heading"
            ));
        };
        let Some(level) = atx_heading_level(selector) else {
            return Err(format!(
                "shell-local exclusion selector {selector:?} is not an exact ATX heading"
            ));
        };
        if level == 1 && !allow_document_title {
            return Err(format!(
                "shell-local exclusion selector {selector:?} may not remove the skill title"
            ));
        }
        if selectors.contains(&selector) {
            return Err(format!(
                "shell-local exclusion selector {selector:?} is duplicated"
            ));
        }
        selectors.push(selector);
    }
    if selectors.is_empty() {
        return Err("shell-local exclusion block has no heading selectors".into());
    }
    Ok(selectors)
}

fn atx_heading_level(line: &str) -> Option<usize> {
    let bytes = line.as_bytes();
    let level = bytes.iter().take_while(|b| **b == b'#').count();
    (level > 0 && level <= 6 && bytes.get(level) == Some(&b' ')).then_some(level)
}

/// Materialize `agent-skills/` from the embedded pinned skill set and point
/// `.claude/skills` and `.codex/skills` at that one tree. Shared by `init` and
/// `sync`. `version` is the shell's pin: each skill's repo-relative links are
/// rewritten for it (see [`crate::links`]).
///
/// Preserves any trailing shell-local block in each retained shared `SKILL.md`
/// (chelis#653), repo-local domain skills declared in `[conform] local_skills`
/// (chelis#651), and omissions declared in `[conform] excluded_skills`.
/// Returns human-readable notices for the caller to surface: an upstream body
/// that changed underneath a shell-local override, an explicitly removed shared
/// skill, and any undeclared local skill it pruned.
pub fn materialize_skills(root: &Path, version: &str) -> Result<Vec<String>, String> {
    // The `[conform]` skill-set controls, read from the PARSED manifest so
    // `sync` and `audit` cannot disagree about what the shell declared
    // (chelis#1262). An absent reef.toml declares nothing; a PRESENT one that
    // does not parse is an error, not an empty declaration, because silently
    // reading it as empty would prune a repo-local skill the shell did declare.
    let (local_skills, excluded_skills) = match fs::read_to_string(root.join("reef.toml")) {
        Ok(text) => {
            let decl = crate::conform::parse(&text).map_err(|e| {
                format!("reef.toml does not parse as TOML, so `[conform]` skill settings cannot be read: {e}")
            })?;
            if !decl.is_clean() {
                return Err(format!(
                    "reef.toml declares unsupported `[conform]` setting(s): {}. Use `local_skills` for additions and `excluded_skills` for embedded removals",
                    decl.findings().join(", ")
                ));
            }
            (decl.local_skills, decl.excluded_skills)
        }
        Err(_) => (Vec::new(), Vec::new()),
    };
    // Validate before the first write or deletion. Exact membership makes a
    // typo or upstream rename loud instead of silently omitting nothing.
    for excluded in &excluded_skills {
        if !skills::SHARED_SKILLS.contains(&excluded.as_str()) {
            return Err(format!(
                "[conform] excluded_skills entry {excluded:?} does not name a shared skill in this toolchain"
            ));
        }
    }
    let mut notices = Vec::new();
    // Sampled ONCE, before the loop: the first written skill creates
    // `agent-skills/`, so testing it inside the loop would report every skill
    // after the first as "restored" on a fresh `init`. A tree with no
    // `agent-skills/` at all is being materialized for the first time, which is
    // not a restoration and gets no notice.
    let skills_dir_existed = root.join("agent-skills").is_dir();
    // The skills the previous sync materialized, from its stamp. A retained
    // skill absent from disk but listed there was removed by hand and is
    // restored; one never listed is new to the shell (a skill this toolchain
    // added to the shared set) and is announced as an addition instead.
    let previously_materialized: Vec<String> =
        fs::read_to_string(root.join("agent-skills/UPSTREAM.toml"))
            .ok()
            .and_then(|text| toml::from_str::<toml::Value>(&text).ok())
            .and_then(|stamp| {
                stamp.get("skills")?.as_array().map(|names| {
                    names
                        .iter()
                        .filter_map(|n| n.as_str().map(str::to_string))
                        .collect()
                })
            })
            .unwrap_or_default();

    // Validate and render every retained skill before the first write. A stale
    // selector must not leave a partially synchronized skill tree.
    let local = LocalTargets::for_shell(&excluded_skills);
    let mut planned = Vec::new();
    for &(skill_name, body) in skills::EMBEDDED_SKILLS {
        if excluded_skills.iter().any(|s| s == skill_name) {
            continue;
        }
        let rel = format!("agent-skills/{skill_name}/SKILL.md");
        let existing = fs::read_to_string(root.join(&rel)).ok();
        let block = existing
            .as_deref()
            .and_then(|content| split_shell_local(content).1.map(str::to_string));
        let managed = skill_managed_span(skill_name, body, block.as_deref(), version, &local)
            .map_err(|why| format!("{skill_name}: {why}"))?;
        planned.push((skill_name, rel, existing, block, managed));
    }

    for &(skill_name, _) in skills::EMBEDDED_SKILLS {
        let skill_dir = root.join("agent-skills").join(skill_name);
        if excluded_skills.iter().any(|s| s == skill_name) && skill_dir.exists() {
            remove_path(&skill_dir)?;
            notices.push(format!(
                "{skill_name}: removed by [conform] excluded_skills"
            ));
        }
    }

    for (skill_name, rel, existing, block, managed) in planned {
        // The on-disk managed span IS the base the block was written against
        // (§8 keeps it byte-equal to the previous toolchain). If it diverges from
        // the new embedded body while a block is present, flag it so the agent
        // re-checks the override against the propagated upstream text.
        if let Some(prev) = &existing {
            let (prev_managed, prev_block) = split_shell_local(prev);
            if prev_block.is_some() && prev_managed.trim_end() != managed.trim_end() {
                notices.push(format!(
                    "{skill_name}: upstream skill body changed and a shell-local override is present; re-check it"
                ));
            }
        }
        // A retained shared skill that was absent is being (re)created. Say so
        // and point to the explicit omission control.
        if existing.is_none() && skills_dir_existed {
            notices.push(if previously_materialized.iter().any(|s| s == skill_name) {
                format!(
                    "{skill_name}: re-materialized; add it to [conform] excluded_skills to omit it on sync"
                )
            } else {
                format!(
                    "{skill_name}: added, a shared skill this shell did not have before; add it to [conform] excluded_skills to omit it on sync"
                )
            });
        }
        let content = match &block {
            Some(b) => format!(
                "{}\n{}",
                managed.trim_end(),
                b.trim_start_matches(['\n', '\r'])
            ),
            None => managed,
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
        if !excluded_skills.iter().any(|s| s == name) {
            manifest.push_str(&format!("  \"{name}\",\n"));
        }
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

    symlink_file(root, "../agent-skills", ".claude/skills")?;
    symlink_file(root, "../agent-skills", ".codex/skills")?;
    Ok(notices)
}

/// The toolchain-owned span of shared skill `name` in a shell: the embedded
/// `body` minus the selectors in the shell's trailing shell-local `block`, with
/// its repo-relative links pinned to `version` (see [`crate::links`]). `sync`
/// writes this span and `audit` compares against it, so the two cannot apply
/// different transforms.
pub fn skill_managed_span(
    name: &str,
    body: &str,
    block: Option<&str>,
    version: &str,
    local: &LocalTargets,
) -> Result<String, String> {
    let managed = apply_shell_local_exclusions(body, block)?;
    Ok(pin_links(
        &managed,
        &skills::source_path(name),
        &format!("agent-skills/{name}/SKILL.md"),
        version,
        local,
    ))
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
    let metadata =
        fs::symlink_metadata(path).map_err(|e| format!("inspect {}: {e}", path.display()))?;
    let res = if metadata.file_type().is_symlink() || metadata.is_file() {
        fs::remove_file(path)
    } else if metadata.is_dir() {
        fs::remove_dir_all(path)
    } else {
        fs::remove_file(path)
    };
    res.map_err(|e| format!("remove {}: {e}", path.display()))
}

/// The documents `conform sync` / `conform bump` **restamp in place** and
/// therefore cannot create: each carries shell-authored content the write path
/// edits around. Distinct from `agent-skills/` and `docs/CHELIS_SURFACE.md`,
/// which the write path *materializes* from the embedded set, so their absence
/// is normal work rather than a missing prerequisite.
///
/// `reef.toml` is listed because [`crate::bump::rewrite_pins`] reads its
/// `compiler` pin before rewriting anything, and because both verbs derive the
/// version they stamp from that pin. `AGENTS.md` is listed because its Repo
/// Identity is hand-authored and anchors the inherited block.
const RESTAMPED_ARTIFACTS: &[(&str, &str)] = &[
    ("reef.toml", "the compiler pin `conform bump` rewrites"),
    (
        "AGENTS.md",
        "carries the `agents-inheritance` managed block sync restamps",
    ),
];

/// Generated documents the write path creates when absent. Their absence is not
/// a gap, but something other than a regular file at the path is: the write
/// would fail after earlier steps had already written, which is the partial
/// application chelis#1263 closed.
const MATERIALIZED_DOCUMENTS: &[&str] = &["docs/CHELIS_SURFACE.md"];

/// One prerequisite the write path cannot proceed without.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreflightGap {
    /// Repo-relative path of the artifact.
    pub rel: &'static str,
    /// Why the verb needs it, and what is wrong with it here.
    pub reason: String,
}

/// Preflight the write path on a shell root: every artifact `sync`/`bump`
/// restamps must already exist, `reef.toml` must carry a pin they can read, and
/// no generated document's path may be occupied by a non-file (chelis#1263).
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
    for rel in MATERIALIZED_DOCUMENTS.iter().copied() {
        let path = root.join(rel);
        if fs::symlink_metadata(&path).is_ok() && !path.is_file() {
            gaps.push(PreflightGap {
                rel,
                reason: "exists but is not a regular file, so sync cannot write the generated \
                         document there"
                    .to_string(),
            });
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
/// document half of `conform sync`), and restore the `CLAUDE.md -> AGENTS.md`
/// symlink. Only fenced regions are touched, so shell-owned text and exclusion
/// selectors outside them survive. Returns notices for the caller to surface:
/// a `CLAUDE.md` that was a regular file or a directory is replaced by the
/// symlink, and its content would otherwise vanish without a word.
pub fn sync_managed_blocks(root: &Path, version: &str) -> Result<Vec<String>, String> {
    // The materialized skill set decides which inherited links stay local.
    let excluded = match fs::read_to_string(root.join("reef.toml")) {
        Ok(text) => {
            crate::conform::parse(&text)
                .map_err(|e| format!("reef.toml does not parse as TOML: {e}"))?
                .excluded_skills
        }
        Err(_) => Vec::new(),
    };
    // Render both documents before the first write. A malformed or stale
    // selector in either therefore cannot leave the other one restamped.
    let agents_path = root.join("AGENTS.md");
    let agents_existing = fs::read_to_string(&agents_path)
        .map_err(|e| format!("read {}: {e}", agents_path.display()))?;
    let agents_updated = render_inherited_document(
        &agents_existing,
        "agents-inheritance",
        version,
        &excluded,
        managed_block::Anchor::AfterHeading("## Repo Identity"),
    )
    .map_err(|why| format!("AGENTS.md: {why}"))?;

    // The surface document is fully generated, so a shell that lacks it gets
    // one; any other read failure stops the sync before the first write.
    let surface_path = root.join("docs/CHELIS_SURFACE.md");
    let surface_existing = match fs::read_to_string(&surface_path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("read {}: {e}", surface_path.display())),
    };
    let surface_updated = render_inherited_document(
        &replace_legacy_surface_header(&surface_existing),
        "chelis-surface",
        version,
        &excluded,
        managed_block::Anchor::Top,
    )
    .map_err(|why| format!("docs/CHELIS_SURFACE.md: {why}"))?;

    fs::write(&agents_path, agents_updated)
        .map_err(|e| format!("write {}: {e}", agents_path.display()))?;
    write(root, "docs/CHELIS_SURFACE.md", &surface_updated)?;
    let mut notices = Vec::new();
    if let Ok(meta) = fs::symlink_metadata(root.join("CLAUDE.md"))
        && !meta.file_type().is_symlink()
    {
        let kind = if meta.is_dir() {
            "a directory"
        } else {
            "a regular file"
        };
        notices.push(format!(
            "CLAUDE.md: replaced {kind} with the symlink to AGENTS.md (contract §1); recover any \
             shell-specific text from version control and keep it in AGENTS.md outside the \
             managed block"
        ));
    }
    symlink_file(root, "AGENTS.md", "CLAUDE.md")?;
    Ok(notices)
}

/// `document` with its `block_id` managed block regenerated from the embedded
/// canonical body (see [`inherited_block_body`]).
fn render_inherited_document(
    document: &str,
    block_id: &str,
    version: &str,
    excluded_skills: &[String],
    anchor: managed_block::Anchor,
) -> Result<String, String> {
    let body = inherited_block_body(block_id, document, version, excluded_skills)?;
    // A document sync creates holds the block alone, byte for byte what
    // `init` writes; upserting into empty text would add a blank line.
    if document.is_empty() {
        return Ok(managed_block::render(block_id, version, &body));
    }
    Ok(managed_block::upsert(
        document, block_id, version, &body, anchor,
    ))
}

/// The body of inherited managed block `block_id` for a shell document: the
/// embedded canonical text, minus the document's own exclusion selectors, with
/// its repo-relative links pinned to `version` (see [`crate::links`]). `sync`
/// writes this and `audit` compares against it, so the two cannot disagree.
pub(crate) fn inherited_block_body(
    block_id: &str,
    document: &str,
    version: &str,
    excluded_skills: &[String],
) -> Result<String, String> {
    let canonical = canonical::body(block_id)
        .ok_or_else(|| format!("no canonical body for block {block_id:?}"))?;
    let path = canonical::document_path(block_id)
        .ok_or_else(|| format!("no document path for block {block_id:?}"))?;
    let body = apply_document_exclusions(canonical, document, block_id)?;
    Ok(pin_links(
        &body,
        path,
        path,
        version,
        &LocalTargets::for_shell(excluded_skills),
    ))
}

/// Turn the superseded `chelis-surface-header` block into an empty
/// `chelis-surface` block at the same position, so the following upsert writes
/// the complete surface guide where the old header stood instead of leaving the
/// header behind with a stale stamp. A document that already carries a
/// `chelis-surface` block keeps it, and the legacy block is simply dropped.
fn replace_legacy_surface_header(document: &str) -> String {
    let Some(legacy) = managed_block::find(document, canonical::LEGACY_SURFACE_HEADER) else {
        return document.to_string();
    };
    let replacement = if managed_block::find(document, "chelis-surface").is_some() {
        String::new()
    } else {
        managed_block::render("chelis-surface", &legacy.version, "")
    };
    format!(
        "{}{replacement}{}",
        &document[..legacy.span.0],
        &document[legacy.span.1..]
    )
}

// ---------------------------------------------------------------- templates

fn reef_toml(name: &str, module_prefix: &str, version: &str) -> String {
    format!(
        "schema = \"3\"\n\n\
         [package]\n\
         name = \"{name}\"\n\
         version = \"0.1.0\"\n\
         compiler = \"={version}\"\n\
         module_prefix = \"{module_prefix}\"\n\
         resolver = \"2\"\n"
    )
}

fn agents_md(name: &str, version: &str) -> String {
    let block = managed_block::render(
        "agents-inheritance",
        version,
        &inherited_block_body("agents-inheritance", "", version, &[])
            .expect("embedded canonical body"),
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
    managed_block::render(
        "chelis-surface",
        version,
        &inherited_block_body("chelis-surface", "", version, &[]).expect("embedded canonical body"),
    )
}

const UPSTREAM_BUGS: &str = "# Upstream Bugs\n\n\
    Track suspected bugs and capability gaps in other Chelis-Lang repositories here.\n\
    File each as an issue in the repository where it originates and cite it by number\n\
    (`chelis#NNN`, or `<repo>#NNN` for a sibling shell), never by a prose name.\n\
    Re-probe at every pin bump.\n\n\
    ## Actively blocking\n\n(none yet)\n\n\
    ## Tracking\n\n(none yet)\n\n\
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
fn symlink_generic(root: &Path, target: &str, link_rel: &str) -> Result<(), String> {
    let link = root.join(link_rel);
    // Idempotent, including migration from the former materialized-directory
    // layout: replace any existing filesystem entry at the link path.
    if fs::symlink_metadata(&link).is_ok() {
        remove_path(&link)?;
    }
    if let Some(parent) = link.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    std::os::unix::fs::symlink(target, &link)
        .map_err(|e| format!("symlink {} -> {target}: {e}", link.display()))
}

#[cfg(not(unix))]
fn symlink_file(_root: &Path, _target: &str, _link_rel: &str) -> Result<(), String> {
    Err("conform init/sync requires a unix platform for symlinks".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(selectors: &str) -> String {
        let selectors = selectors
            .lines()
            .map(|line| format!("<!-- {line} -->"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            "{SHELL_LOCAL_BEGIN}\n{SHELL_LOCAL_EXCLUDE_BEGIN}\n{selectors}\n\
             {SHELL_LOCAL_EXCLUDE_END}\n{SHELL_LOCAL_END}\n"
        )
    }

    #[test]
    fn nested_section_selectors_are_rejected_as_overlapping() {
        let upstream = "# Skill\n\n## Parent\nparent\n\n### Child\nchild\n\n## Sibling\nkeep\n";
        let err = apply_shell_local_exclusions(upstream, Some(&block("## Parent\n### Child")))
            .unwrap_err();
        assert!(err.contains("overlap"), "{err}");
    }

    #[test]
    fn duplicate_section_selectors_are_rejected() {
        let upstream = "# Skill\n\n## Remove\ntext\n";
        let err = apply_shell_local_exclusions(upstream, Some(&block("## Remove\n## Remove")))
            .unwrap_err();
        assert!(err.contains("duplicated"), "{err}");
    }

    #[test]
    fn selector_matching_multiple_upstream_headings_is_rejected() {
        let upstream = "# Skill\n\n## Repeated\none\n\n## Repeated\ntwo\n";
        let err = apply_shell_local_exclusions(upstream, Some(&block("## Repeated"))).unwrap_err();
        assert!(err.contains("multiple upstream headings"), "{err}");
    }

    #[test]
    fn skill_title_selector_is_rejected() {
        let upstream = "# Skill\n\n## Keep\ntext\n";
        let err = apply_shell_local_exclusions(upstream, Some(&block("# Skill"))).unwrap_err();
        assert!(err.contains("may not remove the skill title"), "{err}");
    }

    #[test]
    fn sibling_section_selectors_remove_only_their_own_ranges() {
        let upstream = "# Skill\n\nintro\n\n## A\na\n\n### A child\nchild\n\n## B\nb\n\n## C\nc\n";
        let filtered = apply_shell_local_exclusions(upstream, Some(&block("## A\n## C"))).unwrap();
        assert_eq!(filtered, "# Skill\n\nintro\n\n## B\nb\n\n");
    }

    #[test]
    fn fenced_heading_text_does_not_end_the_selected_section() {
        let upstream = "# Skill\n\n## Remove\nbefore\n\n```markdown\n\
                        ## Example heading\n```\n\nafter\n\n## Keep\nkeep\n";
        let filtered = apply_shell_local_exclusions(upstream, Some(&block("## Remove"))).unwrap();
        assert_eq!(filtered, "# Skill\n\n## Keep\nkeep\n");
    }

    #[test]
    fn selector_does_not_match_heading_text_inside_a_fence() {
        for fence in ["```", "~~~"] {
            let upstream = format!("# Skill\n\n{fence}markdown\n## Example heading\n{fence}\n");
            let err = apply_shell_local_exclusions(&upstream, Some(&block("## Example heading")))
                .unwrap_err();
            assert!(
                err.contains("matches no upstream heading"),
                "{fence}: {err}"
            );
        }
    }

    #[test]
    fn html_comment_heading_text_is_not_a_markdown_heading() {
        let upstream = "# Skill\n\n## Remove\nbefore\n\n<!--\n## Commented heading\n-->\n\nafter\n\n## Keep\nkeep\n";
        let filtered = apply_shell_local_exclusions(upstream, Some(&block("## Remove"))).unwrap();
        assert_eq!(filtered, "# Skill\n\n## Keep\nkeep\n");

        let err = apply_shell_local_exclusions(upstream, Some(&block("## Commented heading")))
            .unwrap_err();
        assert!(err.contains("matches no upstream heading"), "{err}");
    }
}
