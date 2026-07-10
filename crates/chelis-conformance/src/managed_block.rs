//! Pointer "managed blocks" — the mechanism that replaces verbatim-copied
//! upstream text with a stamped, hash-guarded region.
//!
//! Downstream `AGENTS.md` / `docs/CHELIS_SURFACE.md` carry generated regions
//! fenced by HTML comments (invisible when rendered, greppable in source):
//!
//! ```text
//! <!-- BEGIN CHELIS MANAGED BLOCK: agents-inheritance chelis@0.14.0 (sha256:1a2b3c4d5e6f7081) -->
//! Upstream of truth is Chelis-Lang/chelis; ...
//! <!-- END CHELIS MANAGED BLOCK: agents-inheritance -->
//! ```
//!
//! The opening fence stamps the block id, the chelis version it was generated
//! for (`= the reef pin`), and a sha256 of the body. `conform audit` checks two
//! things (both MUST): the stamped version equals the shell's reef pin, and the
//! body hash matches both the fence's declared hash (no hand-edits) and the
//! embedded canonical body for that id+version (not stale-but-self-consistent).
//! `conform sync` regenerates the region in place, touching nothing outside the
//! fences — so a shell's own prose survives.

use sha2::{Digest, Sha256};

const BEGIN_PREFIX: &str = "<!-- BEGIN CHELIS MANAGED BLOCK: ";
const END_PREFIX: &str = "<!-- END CHELIS MANAGED BLOCK: ";
const FENCE_SUFFIX: &str = " -->";

/// A parsed managed block found in a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagedBlock {
    pub id: String,
    /// The stamped chelis version (`chelis@X.Y.Z`).
    pub version: String,
    /// The sha256 prefix declared in the opening fence.
    pub declared_hash: String,
    /// The body text between the fences (includes its trailing newline).
    pub body: String,
    /// Byte range of the whole block (both fences inclusive) in the source.
    pub span: (usize, usize),
}

impl ManagedBlock {
    /// The block's body hash recomputed from its (normalized) bytes.
    pub fn actual_hash(&self) -> String {
        body_hash(&normalize_body(&self.body))
    }

    /// Whether the fence's declared hash matches the body (no hand-edit inside
    /// the block).
    pub fn integrity_ok(&self) -> bool {
        self.declared_hash == self.actual_hash()
    }

    /// Whether this block's body equals the given canonical body, up to the
    /// same normalization `render` applies. This is the check that catches a
    /// *stale-but-self-consistent* block: one whose fence hash matches its own
    /// body (so [`integrity_ok`](Self::integrity_ok) passes) but whose body has
    /// drifted from — or was forged against — the embedded upstream text.
    pub fn matches_canonical(&self, canonical: &str) -> bool {
        normalize_body(&self.body) == normalize_body(canonical)
    }
}

/// Normalize a managed-block body the way [`render`] stamps it: line endings
/// unified to `\n` (so a CRLF checkout does not spuriously fail integrity) and
/// exactly one trailing newline. Hashing and canonical comparison both go
/// through this so the stamp is stable across platforms.
pub fn normalize_body(body: &str) -> String {
    let unified = body.replace("\r\n", "\n").replace('\r', "\n");
    if unified.ends_with('\n') {
        unified
    } else {
        format!("{unified}\n")
    }
}

/// sha256 of a body, as the first 16 lowercase hex chars (64 bits — ample to
/// catch edits, short enough to keep the fence readable).
pub fn body_hash(body: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(body.as_bytes());
    let digest = hasher.finalize();
    let mut out = String::with_capacity(16);
    for b in digest.iter().take(8) {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

/// Render a full managed block. The body is normalized to end with a single
/// newline so `render` and [`find`] round-trip.
pub fn render(id: &str, version: &str, body: &str) -> String {
    let body = normalize_body(body);
    let hash = body_hash(&body);
    format!(
        "{BEGIN_PREFIX}{id} chelis@{version} (sha256:{hash}){FENCE_SUFFIX}\n\
         {body}\
         {END_PREFIX}{id}{FENCE_SUFFIX}\n"
    )
}

/// Locate the managed block with `id`. Returns `None` if absent; a malformed
/// block (open fence without a matching close) also returns `None`.
pub fn find(text: &str, id: &str) -> Option<ManagedBlock> {
    let begin_line_start = find_begin(text, id)?;
    let begin_line_end = text[begin_line_start..]
        .find('\n')
        .map(|i| begin_line_start + i)?;
    let begin_line = &text[begin_line_start..begin_line_end];
    let (version, declared_hash) = parse_begin_line(begin_line, id)?;

    let body_start = begin_line_end + 1; // after the '\n'
    let end_marker = format!("{END_PREFIX}{id}{FENCE_SUFFIX}");
    let end_rel = text[body_start..].find(&end_marker)?;
    let end_marker_start = body_start + end_rel;
    let body = text[body_start..end_marker_start].to_string();

    // Span runs to the end of the END fence line (through its newline if present).
    let end_line_end = text[end_marker_start..]
        .find('\n')
        .map(|i| end_marker_start + i + 1)
        .unwrap_or(text.len());

    Some(ManagedBlock {
        id: id.to_string(),
        version,
        declared_hash,
        body,
        span: (begin_line_start, end_line_end),
    })
}

/// Find the byte offset where the BEGIN fence line for `id` starts. Only a
/// match at the start of a line counts, so the literal fence prefix quoted in a
/// shell's own prose (or a fenced code span) is not mistaken for a real fence.
fn find_begin(text: &str, id: &str) -> Option<usize> {
    let needle = format!("{BEGIN_PREFIX}{id} chelis@");
    text.match_indices(&needle)
        .find(|(i, _)| *i == 0 || text.as_bytes()[i - 1] == b'\n')
        .map(|(i, _)| i)
}

/// Parse `chelis@<ver> (sha256:<hex>) -->` out of the BEGIN fence line.
fn parse_begin_line(line: &str, id: &str) -> Option<(String, String)> {
    let prefix = format!("{BEGIN_PREFIX}{id} chelis@");
    let rest = line.strip_prefix(&prefix)?;
    // rest = "<ver> (sha256:<hex>) -->"
    let (version, rest) = rest.split_once(' ')?;
    let rest = rest.trim();
    let rest = rest.strip_prefix("(sha256:")?;
    let hash_end = rest.find(')')?;
    let hash = &rest[..hash_end];
    if version.is_empty() || hash.is_empty() {
        return None;
    }
    Some((version.to_string(), hash.to_string()))
}

/// Where to place a managed block that is not yet present.
#[derive(Debug, Clone)]
pub enum Anchor<'a> {
    /// Prepend to the very top of the document.
    Top,
    /// Insert immediately after the first line that equals `heading` (e.g. a
    /// `## Repo Identity` section header), else fall back to the bottom.
    AfterHeading(&'a str),
    /// Append to the end of the document.
    Bottom,
}

/// Replace the managed block `id` in place, or insert a freshly rendered one at
/// `anchor` if absent. Only the fenced region is touched; everything else is
/// preserved byte-for-byte.
pub fn upsert(text: &str, id: &str, version: &str, body: &str, anchor: Anchor) -> String {
    let rendered = render(id, version, body);
    if let Some(block) = find(text, id) {
        let (start, end) = block.span;
        let mut out = String::with_capacity(text.len() + rendered.len());
        out.push_str(&text[..start]);
        out.push_str(&rendered);
        out.push_str(&text[end..]);
        return out;
    }
    match anchor {
        Anchor::Top => format!("{rendered}\n{text}"),
        Anchor::Bottom => {
            if text.is_empty() || text.ends_with('\n') {
                format!("{text}\n{rendered}")
            } else {
                format!("{text}\n\n{rendered}")
            }
        }
        Anchor::AfterHeading(heading) => insert_after_heading(text, heading, &rendered)
            .unwrap_or_else(|| {
                if text.is_empty() || text.ends_with('\n') {
                    format!("{text}\n{rendered}")
                } else {
                    format!("{text}\n\n{rendered}")
                }
            }),
    }
}

/// Insert `rendered` immediately after the first line equal to `heading`
/// (trimmed of its newline). Returns `None` if the heading is absent.
fn insert_after_heading(text: &str, heading: &str, rendered: &str) -> Option<String> {
    let mut offset = 0usize;
    for line in text.split_inclusive('\n') {
        let trimmed = line.strip_suffix('\n').unwrap_or(line);
        if trimmed == heading {
            let insert_at = offset + line.len();
            let mut out = String::with_capacity(text.len() + rendered.len() + 1);
            out.push_str(&text[..insert_at]);
            out.push('\n');
            out.push_str(rendered);
            out.push_str(&text[insert_at..]);
            return Some(out);
        }
        offset += line.len();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_find_round_trip() {
        let block = render("agents-inheritance", "0.14.0", "line one\nline two");
        let found = find(&block, "agents-inheritance").expect("found");
        assert_eq!(found.version, "0.14.0");
        assert_eq!(found.body, "line one\nline two\n");
        assert!(found.integrity_ok());
        assert_eq!(found.declared_hash, body_hash("line one\nline two\n"));
    }

    #[test]
    fn find_absent_is_none() {
        assert!(find("no blocks here\n", "agents-inheritance").is_none());
    }

    #[test]
    fn hand_edit_breaks_integrity() {
        let mut block = render("x", "0.14.0", "canonical body");
        // Tamper with the body without updating the fence hash.
        block = block.replace("canonical body", "tampered body");
        let found = find(&block, "x").expect("found");
        assert!(!found.integrity_ok(), "tampered body must fail integrity");
    }

    #[test]
    fn upsert_replaces_in_place_preserving_surroundings() {
        let doc = format!(
            "# Title\n\nintro\n\n{}\n## After\ntrailing\n",
            render("x", "0.13.0", "old body")
        );
        let updated = upsert(&doc, "x", "0.14.0", "new body", Anchor::Bottom);
        assert!(updated.contains("# Title"));
        assert!(updated.contains("## After"));
        assert!(updated.contains("trailing"));
        let found = find(&updated, "x").unwrap();
        assert_eq!(found.version, "0.14.0");
        assert_eq!(found.body, "new body\n");
        // Only one block after upsert (no duplicate insert).
        assert_eq!(updated.matches(BEGIN_PREFIX).count(), 1);
    }

    #[test]
    fn upsert_inserts_when_absent() {
        let doc = "# Title\n\nbody\n";
        let updated = upsert(doc, "x", "0.14.0", "hello", Anchor::Bottom);
        assert!(find(&updated, "x").is_some());
        assert!(updated.starts_with("# Title"));
    }

    #[test]
    fn version_stamp_readable_from_fence() {
        let block = render("chelis-surface-header", "1.2.3", "surface");
        let found = find(&block, "chelis-surface-header").unwrap();
        assert_eq!(found.version, "1.2.3");
    }

    #[test]
    fn crlf_body_still_passes_integrity() {
        // A block rendered with LF, then rewritten to CRLF by a Windows editor
        // or `core.autocrlf`, must not spuriously fail integrity.
        let block = render("x", "0.14.0", "line one\nline two");
        let crlf = block.replace('\n', "\r\n");
        let found = find(&crlf, "x").expect("found");
        assert!(
            found.integrity_ok(),
            "CRLF re-encoding must not break the stamp"
        );
    }

    #[test]
    fn matches_canonical_detects_self_consistent_fork() {
        // The dangerous case: body edited AND the fence hash recomputed, so
        // integrity_ok passes but the body no longer matches upstream.
        let forged = render("x", "0.14.0", "forged body");
        let found = find(&forged, "x").expect("found");
        assert!(found.integrity_ok(), "forged block is self-consistent");
        assert!(
            !found.matches_canonical("canonical body"),
            "a self-consistent fork must not match canonical"
        );
        assert!(
            found.matches_canonical("forged body"),
            "trailing-newline normalization must round-trip"
        );
    }

    #[test]
    fn fence_prefix_in_prose_is_not_a_fence() {
        // A shell that quotes the fence marker mid-sentence must not be parsed
        // as owning a real managed block.
        let doc = "See the `<!-- BEGIN CHELIS MANAGED BLOCK: x chelis@0.1.0 -->` marker.\n";
        assert!(find(doc, "x").is_none());
    }
}
