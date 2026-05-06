//! Rule `doc-filename-convention` — `.md` filenames follow §8 conventions.
//!
//! Path-based dispatch:
//! - `chelis/spec/` top-level numbered specs: `NN-kebab-case.md` (§8.1)
//! - `chelis/spec/design/`: `snake_case.md` (§8.2)
//! - `chelis/docs/`, shell `docs/`: `snake_case.md` for narrative docs;
//!   `SCREAMING_SNAKE_CASE.md` allowed for status reports (§8.3)
//! - `chelis/docs/book/src/`: kebab-case (§8.5, deliberate mdBook exception)
//! - Other paths: no rule (out of scope here; caught by sibling rules
//!   if applicable).

use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::path::Path;
use std::sync::OnceLock;

static SPEC_TOP_RE: OnceLock<Regex> = OnceLock::new();
static SNAKE_RE: OnceLock<Regex> = OnceLock::new();
static SCREAMING_SNAKE_RE: OnceLock<Regex> = OnceLock::new();
static KEBAB_RE: OnceLock<Regex> = OnceLock::new();

fn spec_top_re() -> &'static Regex {
    SPEC_TOP_RE.get_or_init(|| Regex::new(r"^[0-9]{2}-[a-z][a-z0-9-]*\.md$").unwrap())
}
fn snake_re() -> &'static Regex {
    SNAKE_RE.get_or_init(|| Regex::new(r"^[a-z][a-z0-9_]*\.md$").unwrap())
}
fn screaming_snake_re() -> &'static Regex {
    SCREAMING_SNAKE_RE.get_or_init(|| Regex::new(r"^[A-Z][A-Z0-9_]*\.md$").unwrap())
}
fn kebab_re() -> &'static Regex {
    KEBAB_RE.get_or_init(|| Regex::new(r"^[a-z][a-z0-9-]*\.md$").unwrap())
}

pub struct DocFilenameConvention;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    SpecTop,    // chelis/spec/*.md (numeric-prefix + kebab)
    SpecDesign, // chelis/spec/design/*.md (snake)
    Docs,       // <repo>/docs/*.md (snake or SCREAMING_SNAKE)
    BookSrc,    // chelis/docs/book/src/*.md (kebab; mdBook)
    Other,      // not covered by §8
}

fn classify_doc(path: &Path) -> Slot {
    let s = path.to_string_lossy();
    if s.contains("/docs/book/src/") {
        return Slot::BookSrc;
    }
    if s.contains("/spec/design/") {
        return Slot::SpecDesign;
    }
    if let Some(idx) = s.find("/spec/") {
        let after = &s[idx + "/spec/".len()..];
        if !after.contains('/') {
            // Distinguish §8.1 (numbered language specs in chelis/spec/)
            // from §8.2 (snake_case design/phase notes) by the filename
            // itself: a leading `NN-` prefix indicates intent to be a
            // numbered spec. Other names follow §8.2. This shape-based
            // dispatch works for both absolute and relative paths.
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if numbered_spec_prefix(name) {
                return Slot::SpecTop;
            }
            return Slot::SpecDesign;
        }
    }
    if s.contains("/docs/") {
        return Slot::Docs;
    }
    Slot::Other
}

/// True if the filename starts with `NN-` (two digits + hyphen), suggesting
/// it intends to be a §8.1 numbered language spec.
fn numbered_spec_prefix(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() >= 3 && bytes[0].is_ascii_digit() && bytes[1].is_ascii_digit() && bytes[2] == b'-'
}

impl Rule for DocFilenameConvention {
    fn id(&self) -> &str {
        "doc-filename-convention"
    }

    fn spec_ref(&self) -> &str {
        "§8"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::DocFile]
    }

    fn summary(&self) -> &str {
        "documentation filename conventions per §8 (numeric+kebab spec/, snake spec/design/, snake docs/ with SCREAMING_SNAKE for status reports, kebab mdBook book chapters)"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(name) = ctx.path.file_name().and_then(|n| n.to_str()) else {
            return Vec::new();
        };
        let slot = classify_doc(ctx.path);
        let (allowed, hint, sub_ref) = match slot {
            Slot::SpecTop => (
                spec_top_re().is_match(name),
                "numeric-prefix + kebab-case (e.g., `02-surf-syntax.md`)",
                "§8.1",
            ),
            Slot::SpecDesign => (
                snake_re().is_match(name),
                "snake_case (e.g., `phase1a_kernel_codegen.md`)",
                "§8.2",
            ),
            Slot::Docs => (
                snake_re().is_match(name) || screaming_snake_re().is_match(name),
                "snake_case for narrative documents; SCREAMING_SNAKE_CASE for status reports (e.g., `STATUS.md`, `RELEASES.md`)",
                "§8.3",
            ),
            Slot::BookSrc => (
                // Tool-required mdBook special files (per §8.5
                // "Tool-required exceptions") are exempt from the
                // kebab-case rule.
                kebab_re().is_match(name) || name == "SUMMARY.md" || name == "README.md",
                "kebab-case (mdBook URL stability), with `SUMMARY.md` and `README.md` exempt as mdBook tool-required filenames",
                "§8.5",
            ),
            Slot::Other => return Vec::new(),
        };
        if allowed {
            Vec::new()
        } else {
            vec![Violation {
                rule_id: self.id().to_string(),
                spec_ref: sub_ref.to_string(),
                path: ctx.path.to_path_buf(),
                line: None,
                col: None,
                message: format!(
                    "filename `{name}` does not match the convention for this directory: {hint}"
                ),
            }]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(path: &str) -> Vec<Violation> {
        let p = Path::new(path);
        let ctx = Context {
            root: Path::new("/"),
            path: p,
            source: None,
            surface: Surface::DocFile,
        };
        DocFilenameConvention.check(&ctx)
    }

    // Spec top-level (§8.1)

    #[test]
    fn accepts_numeric_kebab_spec() {
        assert!(run("/repo/spec/02-surf-syntax.md").is_empty());
        assert!(run("/repo/spec/00-context.md").is_empty());
        assert!(run("/repo/spec/12-roadmap.md").is_empty());
    }

    #[test]
    fn shape_based_dispatch_snake_spec_is_design_form() {
        // A non-numbered file in spec/ classifies as SpecDesign (§8.2),
        // not SpecTop (§8.1). snake_case is allowed.
        let v = run("/repo/spec/surf_syntax.md");
        assert!(v.is_empty());
    }

    #[test]
    fn flags_kebab_in_spec_when_not_numbered() {
        // Non-numbered + kebab is a §8.2 snake_case violation.
        let v = run("/repo/spec/surf-syntax.md");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].spec_ref, "§8.2");
    }

    // Spec design (§8.2)

    #[test]
    fn accepts_snake_in_spec_design() {
        assert!(run("/repo/spec/design/phase1a_kernel_codegen.md").is_empty());
        assert!(run("/repo/spec/design/chelis_canonical_reference.md").is_empty());
    }

    #[test]
    fn flags_kebab_in_spec_design() {
        let v = run("/repo/spec/design/grad-eval-host-runtime.md");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].spec_ref, "§8.2");
    }

    // Docs narrative + status reports (§8.3)

    #[test]
    fn accepts_snake_narrative_doc() {
        assert!(run("/repo/docs/lin_rca_report.md").is_empty());
        assert!(run("/repo/docs/perf_baseline.md").is_empty());
    }

    #[test]
    fn accepts_screaming_snake_status_report() {
        assert!(run("/repo/docs/STATUS.md").is_empty());
        assert!(run("/repo/docs/RELEASES.md").is_empty());
        assert!(run("/repo/docs/UPSTREAM_BUGS.md").is_empty());
        assert!(run("/repo/docs/BENCHMARK_FINDINGS.md").is_empty());
    }

    #[test]
    fn flags_kebab_in_docs() {
        let v = run("/repo/docs/red-team-v0.2.0-final.md");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].spec_ref, "§8.3");
    }

    #[test]
    fn flags_pascal_in_docs() {
        // PascalCase isn't on either approved form for §8.3.
        let v = run("/repo/docs/MyDoc.md");
        assert_eq!(v.len(), 1);
    }

    // mdBook exception (§8.5)

    #[test]
    fn accepts_kebab_in_mdbook_chapter() {
        assert!(run("/repo/docs/book/src/first-program.md").is_empty());
        assert!(run("/repo/docs/book/src/cli.md").is_empty());
        assert!(run("/repo/docs/book/src/install.md").is_empty());
    }

    #[test]
    fn flags_snake_in_mdbook_chapter() {
        // §8.5 carve-out is for kebab; snake is the violation here.
        let v = run("/repo/docs/book/src/first_program.md");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].spec_ref, "§8.5");
    }

    #[test]
    fn accepts_mdbook_summary_and_readme_tool_required() {
        // SUMMARY.md and README.md are mdBook-required literal filenames
        // and are exempt from §8.5 per the tool-required-exceptions
        // subsection.
        assert!(run("/repo/docs/book/src/SUMMARY.md").is_empty());
        assert!(run("/repo/docs/book/src/README.md").is_empty());
    }

    #[test]
    fn still_flags_other_uppercase_in_book_src() {
        // Only SUMMARY.md and README.md are exempt; other uppercase files
        // in docs/book/src/ are still violations.
        let v = run("/repo/docs/book/src/STATUS.md");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].spec_ref, "§8.5");
    }

    // Off-path: no rule

    #[test]
    fn ignores_paths_outside_doc_dirs() {
        // README at the repo root: not covered by §8 here.
        assert!(run("/repo/README.md").is_empty());
        // CHANGELOG at repo root: ditto.
        assert!(run("/repo/CHANGELOG.md").is_empty());
    }

    // Shell repos' spec/ dirs use §8.2 snake_case, not §8.1 numeric-prefix.

    #[test]
    fn accepts_shell_spec_snake_case() {
        // shoals/spec/phase3l.md is snake — pass under §8.2.
        assert!(run("/home/jeff/Documents/scratch/shoals/spec/phase3l.md").is_empty());
        assert!(run("/home/jeff/Documents/scratch/nautilus/spec/phase3j.md").is_empty());
        assert!(run("/home/jeff/Documents/scratch/coral/spec/phase3k.md").is_empty());
    }

    #[test]
    fn flags_shell_spec_kebab() {
        // Shell spec dirs follow §8.2 (snake_case); kebab is a violation.
        let v = run("/home/jeff/Documents/scratch/shoals/spec/phase-3l.md");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].spec_ref, "§8.2");
    }

    #[test]
    fn flags_partially_numbered_spec_filename() {
        // A file named `99-bad_format.md` claims numbered form but
        // uses underscore instead of kebab — §8.1 violation.
        let v = run("/repo/spec/99-bad_format.md");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].spec_ref, "§8.1");
    }
}
