//! Filesystem walker that classifies entries by [`Surface`] and prunes only
//! paths admitted by Chelis traversal policy.

use crate::Surface;
use crate::policy::{TraversalPolicy, TraversalPolicyError};
use ignore::WalkBuilder;
use std::path::PathBuf;
use std::sync::Arc;

fn is_admitted(policy: &TraversalPolicy, entry: &ignore::DirEntry) -> bool {
    let file_type = entry.file_type();
    if file_type.is_some_and(|kind| !kind.is_dir() && !kind.is_file() && !kind.is_symlink()) {
        return false;
    }
    let is_dir = file_type.is_some_and(|kind| kind.is_dir());
    if entry.depth() == 0 {
        return policy.is_admitted_explicit_entry(entry.path(), is_dir);
    }
    let resolve_target = file_type.is_none_or(|kind| kind.is_symlink());
    policy.is_admitted_entry(entry.path(), is_dir, resolve_target)
}

/// One entry returned by [`walk`]: the path plus its classified surface (if
/// any). Entries the linter doesn't care about are filtered out at the walker
/// level so individual rules don't have to care.
pub struct Entry {
    pub path: PathBuf,
    pub surface: Option<Surface>,
}

/// Walk `root` with all ambient `ignore` filters disabled.
///
/// The shipped and nearest repository [`TraversalPolicy`] are the only
/// pruning inputs. A depth-zero target overrides exclusion matching because it
/// was named explicitly, but still must have an admitted entry kind and resolve
/// inside the repository policy boundary.
pub fn walk(
    root: &std::path::Path,
) -> Result<Vec<Result<Entry, ignore::Error>>, TraversalPolicyError> {
    let policy = Arc::new(TraversalPolicy::load_for(root)?);
    // `ignore` does not apply `filter_entry` to an explicitly supplied root
    // before deciding whether to descend. Validate it first so a symlinked
    // directory root cannot import an external tree and a source-shaped
    // special file never reaches source loading.
    if std::fs::symlink_metadata(root).is_ok()
        && !policy.is_admitted_explicit_entry(root, root.is_dir())
    {
        return Ok(Vec::new());
    }
    let policy_for_filter = Arc::clone(&policy);
    let mut builder = WalkBuilder::new(root);
    builder
        .standard_filters(false)
        .follow_links(false)
        .filter_entry(move |entry| is_admitted(&policy_for_filter, entry));

    let mut out = Vec::new();
    for result in builder.build() {
        match result {
            Ok(entry) => {
                // `ignore` may still yield an explicitly supplied root even
                // when its filter rejects that root. Recheck before exposing
                // an entry to source loading or prepared catalogs.
                if !is_admitted(&policy, &entry) {
                    continue;
                }
                let path = entry.path().to_path_buf();
                let is_dir = entry.file_type().is_some_and(|kind| kind.is_dir());
                // The walk root's directory name is not an entry in its own
                // lint scope, but explicit file roots remain classifiable.
                if is_dir && entry.depth() == 0 {
                    continue;
                }
                let surface = Surface::classify(&path, is_dir);
                out.push(Ok(Entry { path, surface }));
            }
            Err(error) => out.push(Err(error)),
        }
    }
    Ok(out)
}
