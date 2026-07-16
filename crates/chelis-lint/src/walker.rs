//! Filesystem walker that classifies each entry by [`Surface`] and skips
//! noisy directories (`target/`, `.git/`, vendored deps, agent worktrees).

use crate::Surface;
use std::path::PathBuf;
use walkdir::WalkDir;

/// One entry returned by [`walk`]: the path plus its classified surface (if
/// any). Entries the linter doesn't care about are filtered out at the walker
/// level so individual rules don't have to care.
pub struct Entry {
    pub path: PathBuf,
    pub surface: Option<Surface>,
}

/// Walk `root`, skipping directories that the lint should never descend into,
/// and yielding [`Entry`]s for every remaining file plus every meaningful
/// directory.
///
/// Skipped directories:
/// - `target/`, `.git/`, `node_modules/`, `__pycache__/`
/// - `.venv*/` (any directory whose name starts with `.venv`)
/// - `.claude/worktrees/` (agent worktree clutter)
/// - `tests/corpus/opaque_invariants/programs/` (mechanically generated
///   intentional-violation fixtures; linting them is meaningless and they
///   deliberately trip `opaque-domain-construction`)
///
/// Directory entries are emitted with `Surface::Directory` so rules can lint
/// directory naming (e.g., Rust crate dirs must be kebab-case).
pub fn walk(root: &std::path::Path) -> Result<Vec<Result<Entry, walkdir::Error>>, walkdir::Error> {
    let mut out = Vec::new();
    let root_for_filter = root.to_path_buf();
    let walker = WalkDir::new(root)
        .into_iter()
        .filter_entry(move |e| !is_skip_dir(e, &root_for_filter));
    for result in walker {
        match result {
            Ok(entry) => {
                let path = entry.path().to_path_buf();
                let is_dir = entry.file_type().is_dir();
                // Skip the root itself for directory classification — it's
                // not meaningful to lint the repo root's name as a "Rust
                // crate dir", since it isn't one.
                if is_dir && path == root {
                    continue;
                }
                let surface = Surface::classify(&path, is_dir);
                out.push(Ok(Entry { path, surface }));
            }
            Err(e) => out.push(Err(e)),
        }
    }
    Ok(out)
}

fn is_skip_dir(entry: &walkdir::DirEntry, root: &std::path::Path) -> bool {
    if !entry.file_type().is_dir() {
        return false;
    }
    // Never skip the walk root itself: the user explicitly asked for
    // this directory. The substring filters below are for filtering
    // nested clutter inside the walked tree; if the user runs lint
    // from inside (e.g.) a worktree, that worktree IS their target
    // and must not be skipped just because its absolute path contains
    // a `/.claude/worktrees/` segment.
    if entry.depth() == 0 {
        return false;
    }
    let Some(name) = entry.file_name().to_str() else {
        return false;
    };
    if matches!(name, "target" | ".git" | "node_modules" | "__pycache__") {
        return true;
    }
    if name.starts_with(".venv") {
        return true;
    }
    // Skip the agent worktree tree wholesale; per
    // feedback_skip_worktrees_in_ecosystem_survey.md these are not
    // separate repos and shouldn't be re-linted. Match against the
    // path relative to the walk root rather than the full path, so
    // running lint from inside a worktree (where the absolute path
    // contains `/.claude/worktrees/` at the top) is not misclassified
    // as nested-worktree clutter to be skipped wholesale.
    let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
    let rel_str = rel.to_string_lossy();
    if rel_str.contains(".claude/worktrees/") || rel_str.starts_with(".claude/worktrees/") {
        return true;
    }
    // The opaque-invariants differential corpus (RFC D-CORPUS) holds
    // mechanically generated programs that DELIBERATELY violate opacity (the
    // six rejections, etc.) so the coverage gate can MEASURE that each
    // diagnostic fires. Linting them is meaningless and trips the blocking
    // `opaque-domain-construction` rule. Skip the generated-programs dir the
    // same way `target/` and worktree clutter are skipped. The Python runners
    // and README are not under `programs/`, so they stay linted.
    if rel_str.ends_with("tests/corpus/opaque_invariants/programs")
        || rel_str.contains("tests/corpus/opaque_invariants/programs/")
    {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    fn names(root: &std::path::Path) -> Vec<String> {
        walk(root)
            .unwrap()
            .into_iter()
            .filter_map(std::result::Result::ok)
            .filter_map(|e| {
                e.path
                    .strip_prefix(root)
                    .ok()
                    .map(|p| p.to_string_lossy().replace('\\', "/"))
            })
            .collect()
    }

    #[test]
    fn skips_the_opaque_corpus_programs_dir_but_not_its_runners() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        let programs = root.join("tests/corpus/opaque_invariants/programs");
        fs::create_dir_all(&programs).unwrap();
        fs::write(programs.join("check_rej_record.dp"), "(module {} m)\n").unwrap();
        // A sibling Python runner outside programs/ must still be walked.
        fs::write(
            root.join("tests/corpus/opaque_invariants/generate_corpus.py"),
            "x = 1\n",
        )
        .unwrap();
        let walked = names(root);
        assert!(
            walked.iter().any(|p| p.ends_with("generate_corpus.py")),
            "the runner must still be linted: {walked:?}"
        );
        assert!(
            !walked.iter().any(|p| p.ends_with("check_rej_record.dp")),
            "the intentional-violation fixture must be skipped: {walked:?}"
        );
    }

    #[test]
    fn still_skips_target_and_venv() {
        let dir = tempdir().unwrap();
        let root = dir.path();
        fs::create_dir_all(root.join("target")).unwrap();
        fs::write(root.join("target/x.ch"), "def f() = 1\n").unwrap();
        fs::create_dir_all(root.join(".venv")).unwrap();
        fs::write(root.join(".venv/y.ch"), "def g() = 1\n").unwrap();
        fs::write(root.join("real.ch"), "def h() = 1\n").unwrap();
        let walked = names(root);
        assert!(walked.iter().any(|p| p.ends_with("real.ch")));
        assert!(!walked.iter().any(|p| p.ends_with("x.ch")));
        assert!(!walked.iter().any(|p| p.ends_with("y.ch")));
    }
}
