//! Rule `doc-filename-convention` — `.md` filenames follow §8 conventions.
//!
//! Path-based dispatch:
//! - `chelis/spec/` top-level numbered specs: `NN-kebab-case.md` (§8.1)
//! - `chelis/spec/design/`: `snake_case.md` (§8.2)
//! - `chelis/docs/`, shell `docs/`: `snake_case.md` for narrative docs;
//!   `SCREAMING_SNAKE_CASE.md` allowed for status reports (§8.3)
//! - any path that has a component literally named `book/`: kebab-case
//!   (§8.5, deliberate mdBook exception). The opt-in is the path
//!   itself, not ancestor-file inference, so adding or removing a
//!   `book.toml` cannot flip the verdict for any `docs/*.md`.
//! - filename matching a Cargo package name in the workspace also accepts
//!   kebab-case in narrative `docs/` (the `c-earchin` case).
//! - Other paths: no rule (out of scope here; caught by sibling rules
//!   if applicable).

use crate::policy::TraversalPolicy;
use crate::{Context, LintError, PreparedRuleState, Rule, Surface, Violation};
use regex::Regex;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
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

#[derive(Debug, Default)]
struct DocFilenameState {
    package_names: HashSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    SpecTop,    // chelis/spec/*.md (numeric-prefix + kebab)
    SpecDesign, // chelis/spec/design/*.md (snake)
    Docs,       // <repo>/docs/*.md (snake or SCREAMING_SNAKE)
    BookSrc,    // chelis/docs/book/src/*.md (kebab; mdBook)
    Other,      // not covered by §8
}

fn classify_doc(path: &Path) -> Slot {
    // §8.5: any path with a component literally named `book` is mdBook
    // content and uses kebab-case. The discriminator is path-based (issue
    // #190); ancestor `book.toml` files are not consulted.
    if is_inside_mdbook_tree(path) {
        return Slot::BookSrc;
    }
    let s = path.to_string_lossy();
    if s.contains("/spec/design/") {
        return Slot::SpecDesign;
    }
    if let Some(idx) = s.find("/spec/") {
        let after = &s[idx + "/spec/".len()..];
        if !after.contains('/') {
            // Distinguish §8.1 (numbered language specs in chelis/spec/)
            // from §8.2 (snake_case design and implementation notes) by the filename
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

/// Detect whether `path` sits inside an mdBook source tree.
///
/// The discriminator is path-based: any path with a component
/// literally named `book` is treated as mdBook content. This covers
/// the chelis layout (`<repo>/docs/book/...`) and the top-level
/// `<repo>/book/...` layout some shells use. The legacy chelis
/// `<repo>/docs/book/src/` location is naturally a subpath of
/// `docs/book/` and is also covered.
///
/// Path-based discrimination (as opposed to ancestor-`book.toml`
/// inference) was adopted in issue #190 to remove the retroactive-flip
/// footgun: previously, adding or removing a `book.toml` anywhere in
/// the ancestor chain would flip every `docs/*.md` between accepted
/// (§8.3 snake) and rejected (§8.5 kebab). With path-based dispatch,
/// the verdict for a file depends only on the file's own path.
///
/// `book.toml` is not read here; it is documented for mdBook users in
/// the spec but is not used as a lint discriminator.
fn is_inside_mdbook_tree(path: &Path) -> bool {
    path.components()
        .any(|c| c.as_os_str().to_str().is_some_and(|name| name == "book"))
}

/// True if the filename stem of `path` (without the `.md` extension)
/// matches the `name` of a Cargo package somewhere in the workspace selected
/// by repository policy, falling back to `root` when no policy is present.
/// This is the "Cargo-package-name exception" carve-out
/// to §8.3: when a narrative doc is named for a Cargo crate (e.g.,
/// `docs/shells/c-earchin.md` for the `c-earchin` crate), the kebab-case
/// filename is intentional and accepted.
///
/// Detection scans the policy-root `crates/` directory, then walks ancestor
/// dirs of `path` looking for an admitted `Cargo.toml`. The scan is bounded so
/// it is safe to call from a lint check, and policy-root admission prevents
/// machine-local ancestors from affecting repository lint.
fn filename_matches_cargo_package(root: &Path, path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return false;
    };
    // Cargo package names are constrained to a small charset; the stem
    // must look like one to even consider checking. This prevents
    // spurious filesystem walks for stems like `SUMMARY` or
    // `RELEASES`.
    if !looks_like_cargo_package_name(stem) {
        return false;
    }
    let Ok(policy) = TraversalPolicy::load_for(root) else {
        return false;
    };
    let is_admitted = |manifest: &Path| policy.is_admitted_ancillary(manifest, false);
    // Scan workspace crates first — this is the common case. Repository
    // policy defines the workspace boundary even when the lint target is a
    // subdirectory or one explicit doc. Direct Rule::check callers retain
    // this bounded compatibility path, but every directory and ancillary
    // manifest still passes traversal-policy admission through the
    // centralized non-recursive discovery API.
    let manifest_root = policy.repository_root().unwrap_or(root);
    for manifest in policy.admitted_workspace_cargo_manifests(manifest_root) {
        if let Some(name) = read_cargo_package_name(&manifest)
            && name == stem
        {
            return true;
        }
    }
    // Fallback: walk ancestors looking for any admitted Cargo.toml whose
    // package name matches. Useful for layouts where the doc file is
    // colocated with a crate.
    let mut cursor = path.parent();
    while let Some(dir) = cursor {
        let manifest = dir.join("Cargo.toml");
        if is_admitted(&manifest)
            && let Some(name) = read_cargo_package_name(&manifest)
            && name == stem
        {
            return true;
        }
        cursor = dir.parent();
    }
    false
}

/// Return `Some(name)` if `manifest` is a readable `Cargo.toml` whose
/// `[package].name` is parseable. Errors return `None`.
fn read_cargo_package_name(manifest: &Path) -> Option<String> {
    let contents = std::fs::read_to_string(manifest).ok()?;
    extract_package_name(&contents)
}

/// Build the Cargo-package exception catalog from the canonical lint entries.
///
/// Ancestor manifests preserve explicit-file and subdirectory lint behavior,
/// but are admitted only when traversal policy does not exclude them or one of
/// their parents. No rule-local recursive filesystem discovery is permitted.
/// `policy` is the invocation's shared traversal policy; this hook must not
/// reload it.
fn prepare_doc_filename_state(
    root: &Path,
    entries: &[crate::walker::Entry],
    policy: &TraversalPolicy,
) -> DocFilenameState {
    let mut package_names = HashSet::new();
    let mut visited_manifests = HashSet::<PathBuf>::new();

    for entry in entries {
        if entry.surface != Some(Surface::ManifestToml)
            || entry.path.file_name().and_then(|name| name.to_str()) != Some("Cargo.toml")
        {
            continue;
        }
        visited_manifests.insert(entry.path.clone());
        if let Some(name) = read_cargo_package_name(&entry.path) {
            package_names.insert(name);
        }
    }

    let manifest_root = policy.repository_root().unwrap_or(root);
    for manifest in policy.admitted_workspace_cargo_manifests(manifest_root) {
        if visited_manifests.insert(manifest.clone())
            && let Some(name) = read_cargo_package_name(&manifest)
        {
            package_names.insert(name);
        }
    }

    for entry in entries {
        if entry.surface != Some(Surface::DocFile) {
            continue;
        }
        let mut cursor = entry.path.parent();
        while let Some(directory) = cursor {
            let manifest = directory.join("Cargo.toml");
            if visited_manifests.insert(manifest.clone())
                && policy.is_admitted_ancillary(&manifest, false)
                && let Some(name) = read_cargo_package_name(&manifest)
            {
                package_names.insert(name);
            }
            cursor = directory.parent();
        }
    }

    DocFilenameState { package_names }
}

fn filename_matches_known_cargo_package(path: &Path, package_names: &HashSet<String>) -> bool {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| looks_like_cargo_package_name(stem) && package_names.contains(stem))
}

/// Lightweight `[package].name = "..."` extractor. Avoids pulling in a
/// full TOML parser for this single helper. Matches the conventional
/// shape `[package]\nname = "..."` allowing whitespace and Windows
/// line endings. Returns the first match.
fn extract_package_name(toml_src: &str) -> Option<String> {
    static PACKAGE_NAME_RE: OnceLock<Regex> = OnceLock::new();
    let re = PACKAGE_NAME_RE.get_or_init(|| {
        Regex::new(r#"(?s)\[package\][^\[]*?\bname\s*=\s*"([A-Za-z0-9._\-]+)""#).unwrap()
    });
    re.captures(toml_src)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
}

/// True if `s` looks like a valid Cargo package name (kebab/lowercase,
/// digits, underscore, hyphen — no spaces or capital letters). This is
/// a cheap pre-filter that prevents irrelevant filesystem probes.
fn looks_like_cargo_package_name(s: &str) -> bool {
    !s.is_empty()
        && s.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_' || b == b'.'
        })
        && s.bytes().next().is_some_and(|b| b.is_ascii_lowercase())
}

/// True if the filename starts with `NN-` (two digits + hyphen), suggesting
/// it intends to be a §8.1 numbered language spec.
fn numbered_spec_prefix(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() >= 3 && bytes[0].is_ascii_digit() && bytes[1].is_ascii_digit() && bytes[2] == b'-'
}

impl DocFilenameConvention {
    fn check_with_package_names(
        &self,
        ctx: &Context<'_>,
        package_names: Option<&HashSet<String>>,
    ) -> Vec<Violation> {
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
                "snake_case (e.g., `dtype_semantics.md`)",
                "§8.2",
            ),
            Slot::Docs => {
                // §8.3 narrative-docs base shape: snake_case (narrative)
                // or SCREAMING_SNAKE (status reports). Plus the
                // Cargo-package-name carve-out: a kebab-case filename
                // whose stem matches a Cargo package in the workspace
                // is accepted as documentation for that package.
                let base = snake_re().is_match(name) || screaming_snake_re().is_match(name);
                let cargo_exception = kebab_re().is_match(name)
                    && package_names.map_or_else(
                        || filename_matches_cargo_package(ctx.root, ctx.path),
                        |known| filename_matches_known_cargo_package(ctx.path, known),
                    );
                (
                    base || cargo_exception,
                    "snake_case for narrative documents; SCREAMING_SNAKE_CASE for status reports (e.g., `STATUS.md`, `RELEASES.md`); kebab-case is accepted only when the filename stem matches a Cargo package name in the workspace",
                    "§8.3",
                )
            }
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

    fn prepare_run(
        &self,
        root: &Path,
        entries: &[crate::walker::Entry],
        policy: &TraversalPolicy,
    ) -> Result<PreparedRuleState, LintError> {
        Ok(Box::new(prepare_doc_filename_state(root, entries, policy)))
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        self.check_with_package_names(ctx, None)
    }

    fn check_prepared(
        &self,
        ctx: &Context<'_>,
        prepared: &(dyn std::any::Any + Send + Sync),
    ) -> Vec<Violation> {
        let Some(state) = prepared.downcast_ref::<DocFilenameState>() else {
            debug_assert!(
                false,
                "doc filename rule received another rule's prepared state"
            );
            return self.check(ctx);
        };
        self.check_with_package_names(ctx, Some(&state.package_names))
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
        assert!(run("/repo/spec/design/dtype_semantics.md").is_empty());
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
        assert!(run("/repo/shells/shoals/spec/runtime_abi.md").is_empty());
        assert!(run("/repo/shells/nautilus/spec/model_contract.md").is_empty());
        assert!(run("/repo/shells/coral/spec/deployment_contract.md").is_empty());
    }

    #[test]
    fn flags_shell_spec_kebab() {
        // Shell spec dirs follow §8.2 (snake_case); kebab is a violation.
        let v = run("/repo/shells/shoals/spec/runtime-abi.md");
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

    // §8.5 mdBook detection is path-based, not `book.toml`-anchored
    // (issue #190). A `book.toml` sitting in `docs/` does not retroactively
    // promote `docs/getting-started.md` to §8.5 kebab-case; the file is
    // still narrative `docs/` content because its path has no `book/`
    // component. Mirror invariant tests live in
    // `tests/issue_190_doc_filename_path_based.rs`.
    #[test]
    fn book_toml_at_docs_root_does_not_flip_kebab_in_docs() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo = tmp.path();
        let docs = repo.join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        std::fs::write(docs.join("book.toml"), "[book]\ntitle = \"x\"\n").expect("write");
        let target = docs.join("getting-started.md");
        std::fs::write(&target, "").expect("write target");

        let ctx = Context {
            root: repo,
            path: &target,
            source: None,
            surface: Surface::DocFile,
        };
        let v = DocFilenameConvention.check(&ctx);
        assert_eq!(
            v.len(),
            1,
            "kebab in docs/ remains a §8.3 violation regardless of book.toml; got: {v:?}"
        );
        assert_eq!(v[0].spec_ref, "§8.3");
    }

    #[test]
    fn accepts_kebab_for_filename_matching_cargo_package_name() {
        // `c-earchin.md` matches the `c-earchin` cargo crate name in
        // hello-chelis. The narrative-docs rule should accept this
        // hyphenated filename when a matching Cargo package exists in
        // the workspace.
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo = tmp.path();
        let crate_dir = repo.join("crates").join("c-earchin");
        std::fs::create_dir_all(&crate_dir).expect("mkdir");
        std::fs::write(
            crate_dir.join("Cargo.toml"),
            "[package]\nname = \"c-earchin\"\nversion = \"0.1.0\"\n",
        )
        .expect("write");
        let docs_shells = repo.join("docs").join("shells");
        std::fs::create_dir_all(&docs_shells).expect("mkdir");
        let target = docs_shells.join("c-earchin.md");
        std::fs::write(&target, "").expect("write target");

        let ctx = Context {
            root: repo,
            path: &target,
            source: None,
            surface: Surface::DocFile,
        };
        let v = DocFilenameConvention.check(&ctx);
        assert!(
            v.is_empty(),
            "kebab filename matching Cargo package name should pass; got: {v:?}"
        );
    }

    #[test]
    fn negative_kebab_in_docs_without_package_match_still_flags() {
        // Sanity: a kebab-case file in narrative docs without a matching
        // Cargo package name still fires under §8.3.
        let tmp = tempfile::tempdir().expect("tempdir");
        let repo = tmp.path();
        let docs = repo.join("docs");
        std::fs::create_dir_all(&docs).expect("mkdir");
        let target = docs.join("some-random-doc.md");
        std::fs::write(&target, "").expect("write target");

        let ctx = Context {
            root: repo,
            path: &target,
            source: None,
            surface: Surface::DocFile,
        };
        let v = DocFilenameConvention.check(&ctx);
        assert_eq!(
            v.len(),
            1,
            "kebab outside book/ and not Cargo-package-matching should still flag"
        );
        assert_eq!(v[0].spec_ref, "§8.3");
    }
}
