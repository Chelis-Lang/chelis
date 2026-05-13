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
    false
}
