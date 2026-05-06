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
    let walker = WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !is_skip_dir(e));
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

fn is_skip_dir(entry: &walkdir::DirEntry) -> bool {
    if !entry.file_type().is_dir() {
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
    // separate repos and shouldn't be re-linted.
    let path_str = entry.path().to_string_lossy();
    if path_str.contains("/.claude/worktrees/") {
        return true;
    }
    false
}
