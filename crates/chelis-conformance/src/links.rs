//! Version-pinned links for inherited text (contract §1, §3, §8).
//!
//! The inherited documents are authored in the chelis repository, so their
//! repo-relative Markdown links name chelis files. In a shell most of those
//! targets do not exist, and a relative link to them is dead. When `sync`
//! materializes inherited text it therefore rewrites each repo-relative link:
//!
//! - a link whose target `sync` also materializes into the shell (the agent
//!   contract, `CLAUDE.md`, the capability surface, a retained shared skill)
//!   stays relative, recomputed from the file's location in the shell;
//! - any other repo-relative link becomes an absolute URL into the chelis
//!   release the shell pins, `https://github.com/Chelis-Lang/chelis/blob/vX.Y.Z/<path>`;
//! - absolute URLs, scheme links, and pure in-page anchors are untouched.
//!
//! Each link resolves against its source file's own directory in the chelis
//! repository, and anchors are preserved. The embedded assets stay the raw
//! authored bytes, shared by every pin; the pin is only known at sync time, so
//! the rewrite runs there, and `conform audit` applies the same transform to
//! derive the body it expects. The transform is a pure function of the text,
//! the two paths, the version, and the shell's skill declaration.

use std::collections::BTreeMap;

use pulldown_cmark::{Event, LinkType, Options, Parser, Tag};

use crate::skills;

/// The release tree a pinned link points into.
const RELEASE_BLOB_BASE: &str = "https://github.com/Chelis-Lang/chelis/blob";

/// The chelis-repository paths `sync` materializes into a shell, mapped to
/// their shell paths. Built from the shell's `[conform] excluded_skills`, since
/// an omitted skill has no local copy.
#[derive(Debug, Clone)]
pub struct LocalTargets {
    paths: BTreeMap<String, String>,
}

impl LocalTargets {
    /// The materialized set for a shell that omits `excluded_skills`.
    pub fn for_shell(excluded_skills: &[String]) -> LocalTargets {
        let mut paths = BTreeMap::new();
        for doc in ["AGENTS.md", "CLAUDE.md", "docs/CHELIS_SURFACE.md"] {
            paths.insert(doc.to_string(), doc.to_string());
        }
        paths.insert("agent-skills".to_string(), "agent-skills".to_string());
        for name in skills::SHARED_SKILLS {
            if excluded_skills.iter().any(|s| s == name) {
                continue;
            }
            let shell_dir = format!("agent-skills/{name}");
            let source = skills::source_path(name);
            paths.insert(source.clone(), format!("{shell_dir}/SKILL.md"));
            // A repo skill's directory is materialized too; a package skill's
            // directory is the package, which is not.
            if let Some(dir) = source.strip_suffix("/SKILL.md")
                && dir == shell_dir
            {
                paths.insert(dir.to_string(), shell_dir);
            }
        }
        LocalTargets { paths }
    }

    fn shell_path(&self, repo_path: &str) -> Option<&str> {
        self.paths.get(repo_path).map(String::as_str)
    }
}

/// Rewrite the repo-relative links of `text`, authored at `source` in the chelis
/// repository and materialized at `dest` in the shell, for a shell pinned to
/// `version` whose materialized set is `local`.
pub fn pin_links(
    text: &str,
    source: &str,
    dest: &str,
    version: &str,
    local: &LocalTargets,
) -> String {
    let source_dir = parent(source);
    let dest_dir = parent(dest);
    let rewrite = |target: &str| -> Option<String> {
        let (path, anchor) = match target.find('#') {
            Some(i) => (&target[..i], &target[i..]),
            None => (target, ""),
        };
        if path.is_empty() || has_scheme(path) || path.starts_with("//") {
            return None;
        }
        let resolved = if let Some(rooted) = path.strip_prefix('/') {
            normalize("", rooted)?
        } else {
            normalize(source_dir, path)?
        };
        let new = match local.shell_path(resolved.trim_end_matches('/')) {
            Some(shell) => format!("{}{anchor}", relative(dest_dir, shell)),
            None => format!("{RELEASE_BLOB_BASE}/v{version}/{resolved}{anchor}"),
        };
        (new != target).then_some(new)
    };

    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    let parser = Parser::new_ext(text, Options::empty()).into_offset_iter();
    let defs: Vec<(String, std::ops::Range<usize>)> = parser
        .reference_definitions()
        .iter()
        .map(|(_, def)| (def.dest.to_string(), def.span.clone()))
        .collect();
    for (event, range) in parser {
        let (Event::Start(Tag::Link {
            link_type,
            dest_url,
            ..
        })
        | Event::Start(Tag::Image {
            link_type,
            dest_url,
            ..
        })) = event
        else {
            continue;
        };
        if link_type != LinkType::Inline {
            continue;
        }
        let Some(start) = inline_destination(&text[range.clone()], &dest_url) else {
            continue;
        };
        if let Some(new) = rewrite(&dest_url) {
            let at = range.start + start;
            edits.push((at, at + dest_url.len(), new));
        }
    }
    for (dest, span_range) in &defs {
        let span = &text[span_range.clone()];
        let Some(colon) = span.find("]:") else {
            continue;
        };
        let after = &span[colon + 2..];
        let lead = after.len() - after.trim_start().len();
        let rest = &after[lead..];
        let skip = usize::from(rest.starts_with('<'));
        if rest[skip..].starts_with(dest.as_str())
            && let Some(new) = rewrite(dest)
        {
            let at = span_range.start + colon + 2 + lead + skip;
            edits.push((at, at + dest.len(), new));
        }
    }

    edits.sort_by_key(|(start, _, _)| *start);
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    for (start, end, new) in edits {
        out.push_str(&text[cursor..start]);
        out.push_str(&new);
        cursor = end;
    }
    out.push_str(&text[cursor..]);
    out
}

/// Offset of an inline link's destination inside the link's source text: the
/// destination is the last `(`-introduced occurrence of `dest`, optionally
/// behind whitespace and `<`. `None` when the source spells it differently
/// (escapes), which the link-coverage test rules out for the inherited texts.
fn inline_destination(link: &str, dest: &str) -> Option<usize> {
    if dest.is_empty() {
        return None;
    }
    link.match_indices(dest)
        .map(|(i, _)| i)
        .filter(|&i| {
            let before = link[..i].trim_end_matches('<').trim_end();
            before.ends_with('(')
        })
        .last()
}

fn has_scheme(path: &str) -> bool {
    let Some(colon) = path.find(':') else {
        return false;
    };
    let scheme = &path[..colon];
    scheme
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

fn parent(path: &str) -> &str {
    path.rfind('/').map_or("", |i| &path[..i])
}

/// `path` joined to `base` with `.` and `..` resolved, or `None` when it climbs
/// above the repository root (such a link is dead upstream and left alone).
fn normalize(base: &str, path: &str) -> Option<String> {
    let mut parts: Vec<&str> = base.split('/').filter(|p| !p.is_empty()).collect();
    let trailing = path.ends_with('/');
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            other => parts.push(other),
        }
    }
    let mut joined = parts.join("/");
    if trailing && !joined.is_empty() {
        joined.push('/');
    }
    Some(joined)
}

/// The relative path from directory `from` to `to`, both shell-relative.
fn relative(from: &str, to: &str) -> String {
    let from: Vec<&str> = from.split('/').filter(|p| !p.is_empty()).collect();
    let to_parts: Vec<&str> = to.split('/').filter(|p| !p.is_empty()).collect();
    let common = from
        .iter()
        .zip(&to_parts)
        .take_while(|(a, b)| a == b)
        .count();
    let mut out: Vec<&str> = vec![".."; from.len() - common];
    out.extend(&to_parts[common..]);
    out.join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pin(text: &str, source: &str, dest: &str) -> String {
        pin_links(text, source, dest, "0.18.12", &LocalTargets::for_shell(&[]))
    }

    #[test]
    fn a_link_without_a_local_copy_is_pinned_to_the_release() {
        assert_eq!(
            pin("see [gate](docs/local_gate.md).", "AGENTS.md", "AGENTS.md"),
            "see [gate](https://github.com/Chelis-Lang/chelis/blob/v0.18.12/docs/local_gate.md)."
        );
    }

    #[test]
    fn a_parent_link_resolves_against_the_source_directory() {
        assert_eq!(
            pin(
                "[types](../spec/04-type-system.md#dtypes)",
                "docs/CHELIS_SURFACE.md",
                "docs/CHELIS_SURFACE.md"
            ),
            "[types](https://github.com/Chelis-Lang/chelis/blob/v0.18.12/spec/04-type-system.md#dtypes)"
        );
        // A sibling-relative link resolves into the source's own directory.
        assert_eq!(
            pin(
                "[gates](manual_gates.md#a)",
                "docs/CHELIS_SURFACE.md",
                "docs/CHELIS_SURFACE.md"
            ),
            "[gates](https://github.com/Chelis-Lang/chelis/blob/v0.18.12/docs/manual_gates.md#a)"
        );
    }

    #[test]
    fn a_materialized_target_stays_relative_from_the_shell_location() {
        let text =
            "[skill](agent-skills/spec-sync/SKILL.md) and [surface](docs/CHELIS_SURFACE.md#x)";
        assert_eq!(pin(text, "AGENTS.md", "AGENTS.md"), text);
        assert_eq!(
            pin(
                "[contract](../AGENTS.md) [std](../packages/chelis-std/SKILL.md)",
                "docs/CHELIS_SURFACE.md",
                "docs/CHELIS_SURFACE.md"
            ),
            "[contract](../AGENTS.md) [std](../agent-skills/chelis-std/SKILL.md)"
        );
        // The package skill moves on materialization; its links follow it.
        assert_eq!(
            pin(
                "[surface](../../docs/CHELIS_SURFACE.md) [src](src/test.ch)",
                "packages/chelis-std/SKILL.md",
                "agent-skills/chelis-std/SKILL.md"
            ),
            "[surface](../../docs/CHELIS_SURFACE.md) \
             [src](https://github.com/Chelis-Lang/chelis/blob/v0.18.12/packages/chelis-std/src/test.ch)"
        );
    }

    #[test]
    fn an_excluded_skill_has_no_local_copy() {
        let text = "[cli](agent-skills/cli-surface/SKILL.md)";
        let local = LocalTargets::for_shell(&["cli-surface".to_string()]);
        assert_eq!(
            pin_links(text, "AGENTS.md", "AGENTS.md", "0.18.12", &local),
            "[cli](https://github.com/Chelis-Lang/chelis/blob/v0.18.12/agent-skills/cli-surface/SKILL.md)"
        );
    }

    #[test]
    fn absolute_urls_anchors_and_code_are_untouched() {
        for text in [
            "[r](https://github.com/Chelis-Lang/chelis/releases)",
            "[m](mailto:a@b.c)",
            "[s](#issue-tracking)",
            "`[x](docs/local_gate.md)`",
            "```markdown\n[x](docs/local_gate.md)\n```\n",
            "def f[n](x: tensor[n, f32]) = x",
        ] {
            assert_eq!(pin(text, "AGENTS.md", "AGENTS.md"), text, "{text:?}");
        }
    }

    #[test]
    fn reference_definitions_and_titles_are_rewritten_in_place() {
        assert_eq!(
            pin(
                "[gate][g] and ![img](docs/a.png \"title\")\n\n[g]: docs/local_gate.md\n",
                "AGENTS.md",
                "AGENTS.md"
            ),
            "[gate][g] and ![img](https://github.com/Chelis-Lang/chelis/blob/v0.18.12/docs/a.png \"title\")\n\n\
             [g]: https://github.com/Chelis-Lang/chelis/blob/v0.18.12/docs/local_gate.md\n"
        );
    }

    /// Rewriting already-rewritten text changes nothing for links into the
    /// chelis tree. This is not claimed for the moved package skill: its shell
    /// path (`agent-skills/chelis-std/`) is not a chelis path. Sync only ever
    /// rewrites the raw embedded text, so it never re-reads its own output.
    #[test]
    fn the_transform_is_idempotent() {
        let once = pin(
            "[a](docs/local_gate.md) [b](agent-skills/spec-sync/SKILL.md)",
            "AGENTS.md",
            "AGENTS.md",
        );
        assert_eq!(pin(&once, "AGENTS.md", "AGENTS.md"), once);
    }
}
