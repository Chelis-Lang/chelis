//! Version-keyed **source-crate store** for shells that link chelis
//! compiler crates (`chelis-ir`, `chelis-types`, …) as Cargo path
//! dependencies (the class-(c) dependency in
//! `spec/design/shell_repo_contract.md`).
//!
//! Layout, under `~/.local/share/chelis-src/` (override `$CHELIS_SRC_HOME`):
//!
//! ```text
//! mirror.git/   bare mirror of canonical Chelis-Lang/chelis (all refs/tags)
//! <version>/    git worktree checked out at that version's pinned commit
//! ```
//!
//! This is the class-(c) analogue of the toolchain store
//! `~/.local/share/chelis/<ver>/`: versions sit side by side, immutable
//! once synced, resolved per consuming repo. git worktrees share one
//! object store, so each pin is a cheap checkout. The CLI layer
//! (`chelis reef src`) later points each shell's `../chelis` slot at the
//! worktree matching its own `reef.toml` pin.
//!
//! The module is intentionally **self-contained**: every entry point takes
//! an explicit `store_root`, remote URL, optional auth token, version, and
//! optional commit. It never reads global env or resolves auth itself, so
//! it is exercised by fixture tests against a local bare repo with no
//! network and no credentials. The CLI layer supplies the production
//! defaults ([`CANONICAL_REMOTE`], `$CHELIS_SRC_HOME`, and a token from
//! `resolve_github_token`).

use std::path::{Path, PathBuf};
use std::process::Command;

/// Canonical source repository. The CLI layer passes this as `remote_url`
/// in production; tests pass a local path instead.
pub const CANONICAL_REMOTE: &str = "https://github.com/Chelis-Lang/chelis.git";

/// Directory name of the bare mirror within the store root.
const MIRROR_DIR: &str = "mirror.git";

/// Errors from source-store operations. Each names the step so a CLI
/// diagnostic can be actionable.
#[derive(Debug)]
pub enum ChelisSrcError {
    /// Neither `$CHELIS_SRC_HOME` nor `$HOME` is set — cannot locate the store.
    NoStoreRoot,
    /// A `git` invocation exited non-zero. `context` names the step.
    Git {
        context: String,
        status: String,
        stderr: String,
    },
    /// `git` could not be spawned (not on PATH, etc.).
    GitSpawn { context: String, source: String },
    /// Filesystem error while preparing the store.
    Io { context: String, source: String },
    /// The pinned commit does not exist in the mirror after fetch.
    CommitNotFound { commit: String },
    /// The `v<version>` release tag does not exist in the mirror.
    TagNotFound { tag: String },
    /// The store slot exists but is not a worktree we manage (a stray
    /// directory or a clone). We refuse to clobber it.
    SlotOccupied { path: PathBuf, reason: String },
    /// Wiring a shell's `../chelis` slot to the store worktree failed, or a
    /// drift check found the slot pointing off the pin.
    Wiring { slot: PathBuf, detail: String },
}

impl std::fmt::Display for ChelisSrcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoStoreRoot => write!(
                f,
                "cannot locate the source store: set $CHELIS_SRC_HOME or $HOME"
            ),
            Self::Git {
                context,
                status,
                stderr,
            } => write!(f, "git {context} failed ({status}): {stderr}"),
            Self::GitSpawn { context, source } => {
                write!(f, "could not run git for {context}: {source}")
            }
            Self::Io { context, source } => write!(f, "{context}: {source}"),
            Self::CommitNotFound { commit } => {
                write!(f, "pinned commit {commit} not found in canonical mirror")
            }
            Self::TagNotFound { tag } => {
                write!(f, "release tag {tag} not found in canonical mirror")
            }
            Self::SlotOccupied { path, reason } => {
                write!(f, "store slot {} is occupied: {reason}", path.display())
            }
            Self::Wiring { slot, detail } => {
                write!(f, "../chelis slot {}: {detail}", slot.display())
            }
        }
    }
}

impl std::error::Error for ChelisSrcError {}

/// What [`sync`] did, for the CLI to report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncOutcome {
    pub mirror: PathBuf,
    pub worktree: PathBuf,
    /// The full 40-hex commit the worktree is checked out at.
    pub commit: String,
}

/// Resolve the default store root: `$CHELIS_SRC_HOME`, else
/// `~/.local/share/chelis-src`. Parallels `registry_root` for the reef
/// registry. The store *operations* take an explicit root; this is only
/// the production default the CLI passes.
pub fn default_store_root() -> Result<PathBuf, ChelisSrcError> {
    if let Some(p) = std::env::var_os("CHELIS_SRC_HOME") {
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME").ok_or(ChelisSrcError::NoStoreRoot)?;
    Ok(PathBuf::from(home).join(".local/share/chelis-src"))
}

/// Build the `git -c http.extraheader=...` prefix that authenticates
/// against a private remote, or an empty prefix when no token is supplied.
/// The token is passed per-invocation and never written to the mirror's
/// stored config, so it does not persist to disk.
fn auth_prefix(token: Option<&str>) -> Vec<String> {
    match token {
        Some(t) if !t.is_empty() => vec![
            "-c".to_string(),
            format!("http.extraheader=AUTHORIZATION: bearer {t}"),
        ],
        _ => Vec::new(),
    }
}

/// Run a `git` command, returning trimmed stdout on success.
fn run_git<I, S>(args: I, cwd: Option<&Path>, context: &str) -> Result<String, ChelisSrcError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let mut cmd = Command::new("git");
    cmd.args(args);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let out = cmd.output().map_err(|e| ChelisSrcError::GitSpawn {
        context: context.to_string(),
        source: e.to_string(),
    })?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    } else {
        Err(ChelisSrcError::Git {
            context: context.to_string(),
            status: out.status.to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).trim().to_string(),
        })
    }
}

/// Ensure the bare mirror exists under `store_root` and is up to date. On
/// first use, `git clone --mirror`; afterward, `git fetch --prune`. Returns
/// the mirror path.
pub fn ensure_mirror(
    store_root: &Path,
    remote_url: &str,
    token: Option<&str>,
) -> Result<PathBuf, ChelisSrcError> {
    let mirror = store_root.join(MIRROR_DIR);
    if mirror.join("HEAD").exists() {
        let mut args = auth_prefix(token);
        args.extend(["fetch".into(), "--prune".into(), "origin".into()]);
        run_git(&args, Some(&mirror), "fetch canonical mirror")?;
    } else {
        std::fs::create_dir_all(store_root).map_err(|e| ChelisSrcError::Io {
            context: format!("create store root {}", store_root.display()),
            source: e.to_string(),
        })?;
        let mut args = auth_prefix(token);
        args.extend([
            "clone".into(),
            "--mirror".into(),
            remote_url.to_string(),
            mirror.to_string_lossy().into_owned(),
        ]);
        run_git(&args, None, "clone canonical mirror")?;
    }
    Ok(mirror)
}

/// Resolve the pinned commit **from the mirror** (canonical after fetch,
/// never a local tag). With an explicit commit, assert it exists; without,
/// dereference the `v<version>` release tag to its commit.
pub fn resolve_commit(
    mirror: &Path,
    version: &str,
    explicit: Option<&str>,
) -> Result<String, ChelisSrcError> {
    match explicit {
        Some(commit) => {
            let rev = format!("{commit}^{{commit}}");
            run_git(
                ["rev-parse", "--verify", "--quiet", &rev],
                Some(mirror),
                "verify pinned commit",
            )
            .map_err(|_| ChelisSrcError::CommitNotFound {
                commit: commit.to_string(),
            })
        }
        None => {
            let tag = format!("v{version}");
            let rev = format!("refs/tags/{tag}^{{commit}}");
            run_git(
                ["rev-parse", "--verify", "--quiet", &rev],
                Some(mirror),
                "resolve release tag",
            )
            .map_err(|_| ChelisSrcError::TagNotFound { tag })
        }
    }
}

/// Read the HEAD commit of the store worktree for `version`, or `None` if
/// there is no worktree there yet.
pub fn worktree_head(store_root: &Path, version: &str) -> Result<Option<String>, ChelisSrcError> {
    let wt = store_root.join(version);
    if !wt.join(".git").exists() {
        return Ok(None);
    }
    let head = run_git(["rev-parse", "HEAD"], Some(&wt), "read worktree HEAD")?;
    Ok(Some(head))
}

/// Ensure a worktree at `<store_root>/<version>` is checked out at `commit`.
/// Idempotent: creates it if absent, re-checks-out if it drifted, no-ops if
/// already correct. Refuses to clobber a slot that exists but is not a
/// worktree.
pub fn ensure_worktree(
    store_root: &Path,
    mirror: &Path,
    version: &str,
    commit: &str,
) -> Result<PathBuf, ChelisSrcError> {
    let wt = store_root.join(version);
    if wt.exists() {
        if !wt.join(".git").exists() {
            return Err(ChelisSrcError::SlotOccupied {
                path: wt,
                reason: "exists but is not a git worktree (refusing to overwrite)".to_string(),
            });
        }
        let head = run_git(["rev-parse", "HEAD"], Some(&wt), "read worktree HEAD")?;
        if head != commit {
            run_git(
                ["checkout", "--detach", commit],
                Some(&wt),
                "refresh worktree to pinned commit",
            )?;
        }
    } else {
        run_git(
            ["worktree", "add", "--detach", &wt.to_string_lossy(), commit],
            Some(mirror),
            "add worktree",
        )?;
    }
    Ok(wt)
}

/// End-to-end sync: ensure the mirror, resolve the commit, ensure the
/// worktree. Returns the resolved paths and commit.
pub fn sync(
    store_root: &Path,
    remote_url: &str,
    token: Option<&str>,
    version: &str,
    explicit_commit: Option<&str>,
) -> Result<SyncOutcome, ChelisSrcError> {
    let mirror = ensure_mirror(store_root, remote_url, token)?;
    let commit = resolve_commit(&mirror, version, explicit_commit)?;
    let worktree = ensure_worktree(store_root, &mirror, version, &commit)?;
    Ok(SyncOutcome {
        mirror,
        worktree,
        commit,
    })
}

/// The result of wiring a shell's `../chelis` slot to a store worktree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireOutcome {
    /// The slot did not exist; a fresh symlink was created.
    Created,
    /// The slot was already the correct symlink; nothing changed.
    AlreadyWired,
    /// The slot was a symlink pointing elsewhere; it was repointed.
    Repointed { previous: PathBuf },
}

/// Point a shell's `../chelis` slot at the store `worktree` via a symlink.
///
/// Idempotent and non-destructive to real trees: creates the symlink if the
/// slot is absent, repoints it if it is a symlink elsewhere, no-ops if it is
/// already correct, and **refuses** (never deletes) a real directory in the
/// slot — that is a developer's chelis clone, which must be relocated out of
/// the sibling slot (e.g. to `~/chelis-dev`) by hand.
#[cfg(unix)]
pub fn wire_slot(slot: &Path, worktree: &Path) -> Result<WireOutcome, ChelisSrcError> {
    use std::os::unix::fs::symlink;
    match std::fs::symlink_metadata(slot) {
        Ok(meta) if meta.file_type().is_symlink() => {
            let current = std::fs::read_link(slot).map_err(|e| ChelisSrcError::Wiring {
                slot: slot.to_path_buf(),
                detail: format!("could not read existing symlink: {e}"),
            })?;
            if current == worktree {
                Ok(WireOutcome::AlreadyWired)
            } else {
                std::fs::remove_file(slot).map_err(|e| ChelisSrcError::Wiring {
                    slot: slot.to_path_buf(),
                    detail: format!("could not repoint symlink: {e}"),
                })?;
                symlink(worktree, slot).map_err(|e| ChelisSrcError::Wiring {
                    slot: slot.to_path_buf(),
                    detail: format!("could not create symlink: {e}"),
                })?;
                Ok(WireOutcome::Repointed { previous: current })
            }
        }
        Ok(_) => Err(ChelisSrcError::Wiring {
            slot: slot.to_path_buf(),
            detail: "a real directory/file occupies the slot (likely a chelis dev clone); \
                     move it out of the sibling slot (e.g. to ~/chelis-dev) and re-run — \
                     refusing to delete it"
                .to_string(),
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if let Some(parent) = slot.parent() {
                std::fs::create_dir_all(parent).map_err(|e| ChelisSrcError::Io {
                    context: format!("create slot parent {}", parent.display()),
                    source: e.to_string(),
                })?;
            }
            symlink(worktree, slot).map_err(|e| ChelisSrcError::Wiring {
                slot: slot.to_path_buf(),
                detail: format!("could not create symlink: {e}"),
            })?;
            Ok(WireOutcome::Created)
        }
        Err(e) => Err(ChelisSrcError::Io {
            context: format!("stat slot {}", slot.display()),
            source: e.to_string(),
        }),
    }
}

/// Assert a shell's `../chelis` slot is the symlink into the store that the
/// pin expects. Loud, actionable failure otherwise — this is the local drift
/// guard's filesystem half (the CLI also checks the worktree HEAD and
/// `Cargo.lock`).
#[cfg(unix)]
pub fn check_slot(slot: &Path, worktree: &Path) -> Result<(), ChelisSrcError> {
    let meta = std::fs::symlink_metadata(slot).map_err(|e| ChelisSrcError::Wiring {
        slot: slot.to_path_buf(),
        detail: format!("missing or unreadable (run `chelis reef src sync`): {e}"),
    })?;
    if !meta.file_type().is_symlink() {
        return Err(ChelisSrcError::Wiring {
            slot: slot.to_path_buf(),
            detail: "is not a symlink into the source store (run `chelis reef src sync`)"
                .to_string(),
        });
    }
    let target = std::fs::read_link(slot).map_err(|e| ChelisSrcError::Wiring {
        slot: slot.to_path_buf(),
        detail: format!("could not read symlink: {e}"),
    })?;
    if target != worktree {
        return Err(ChelisSrcError::Wiring {
            slot: slot.to_path_buf(),
            detail: format!(
                "points at {} but the pin needs {} (run `chelis reef src sync`)",
                target.display(),
                worktree.display()
            ),
        });
    }
    Ok(())
}

#[cfg(not(unix))]
pub fn wire_slot(slot: &Path, _worktree: &Path) -> Result<WireOutcome, ChelisSrcError> {
    Err(ChelisSrcError::Wiring {
        slot: slot.to_path_buf(),
        detail: "source-crate slot wiring requires Unix symlinks; unsupported on this platform"
            .to_string(),
    })
}

#[cfg(not(unix))]
pub fn check_slot(slot: &Path, _worktree: &Path) -> Result<(), ChelisSrcError> {
    Err(ChelisSrcError::Wiring {
        slot: slot.to_path_buf(),
        detail: "source-crate slot wiring requires Unix symlinks; unsupported on this platform"
            .to_string(),
    })
}

/// Given the text of a `Cargo.lock` and the chelis source crates a shell
/// declares, return every one whose locked version is **not** `version`,
/// formatted `name=foundversion`. Empty ⇒ all declared crates are at the
/// pin (or absent from the lock).
///
/// This is the lockfile half of the drift guard: it catches the
/// `.cargo`-override failure mode the Phase 1.5 probe found — where the
/// build compiles the pinned source but `Cargo.lock` still records the
/// dev-clone version — and a plain stale `../chelis`.
pub fn cargo_lock_mismatches(
    lock_text: &str,
    crates: &[String],
    version: &str,
) -> Result<Vec<String>, ChelisSrcError> {
    let lock: toml::Value = toml::from_str(lock_text).map_err(|e| ChelisSrcError::Io {
        context: "parse Cargo.lock".to_string(),
        source: e.to_string(),
    })?;
    let packages = lock
        .get("package")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();
    let mut mismatched = Vec::new();
    for crate_name in crates {
        for pkg in &packages {
            if pkg.get("name").and_then(|n| n.as_str()) == Some(crate_name.as_str()) {
                let v = pkg.get("version").and_then(|v| v.as_str()).unwrap_or("?");
                if v != version {
                    mismatched.push(format!("{crate_name}={v}"));
                }
            }
        }
    }
    Ok(mismatched)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Run a git command in `dir`, panicking on failure (test helper).
    fn git_in(dir: &Path, args: &[&str]) -> String {
        run_git(args, Some(dir), "test fixture git").expect("test fixture git command")
    }

    /// Create a canonical-style repo at `dir` with `n` commits; tag the
    /// `tag_at`-th commit (1-based) `tag`. Returns the commit SHAs in
    /// order.
    fn init_canonical(dir: &Path, n: usize, tag: &str, tag_at: usize) -> Vec<String> {
        std::fs::create_dir_all(dir).unwrap();
        git_in(dir, &["init", "-q"]);
        git_in(dir, &["config", "user.email", "fixture@example.invalid"]);
        git_in(dir, &["config", "user.name", "Fixture"]);
        // Keep git from forking a background `gc --auto` / maintenance
        // daemon that would inherit the test's pipes and trip nextest's
        // leak detector.
        git_in(dir, &["config", "gc.auto", "0"]);
        git_in(dir, &["config", "maintenance.auto", "false"]);
        let mut commits = Vec::new();
        for i in 1..=n {
            std::fs::write(dir.join("VERSION"), format!("commit {i}\n")).unwrap();
            git_in(dir, &["add", "."]);
            git_in(dir, &["commit", "-q", "-m", &format!("commit {i}")]);
            let sha = git_in(dir, &["rev-parse", "HEAD"]);
            if i == tag_at {
                git_in(dir, &["tag", tag]);
            }
            commits.push(sha);
        }
        commits
    }

    #[test]
    fn sync_creates_mirror_and_worktree_at_tagged_commit() {
        let tmp = tempdir().unwrap();
        let canonical = tmp.path().join("canonical");
        let commits = init_canonical(&canonical, 1, "v0.0.1", 1);
        let store = tmp.path().join("store");

        let outcome =
            sync(&store, &canonical.to_string_lossy(), None, "0.0.1", None).expect("sync");

        assert!(
            store.join("mirror.git/HEAD").exists(),
            "bare mirror created"
        );
        assert!(
            store.join("0.0.1").join(".git").exists(),
            "worktree created"
        );
        assert_eq!(outcome.commit, commits[0], "resolved to the tagged commit");
        assert_eq!(
            worktree_head(&store, "0.0.1").unwrap().as_deref(),
            Some(commits[0].as_str()),
            "worktree HEAD is the pinned commit"
        );
    }

    #[test]
    fn sync_is_idempotent() {
        let tmp = tempdir().unwrap();
        let canonical = tmp.path().join("canonical");
        let commits = init_canonical(&canonical, 1, "v0.0.1", 1);
        let store = tmp.path().join("store");
        let url = canonical.to_string_lossy();

        let first = sync(&store, &url, None, "0.0.1", None).expect("first sync");
        let second = sync(&store, &url, None, "0.0.1", None).expect("second sync");
        assert_eq!(first, second, "re-sync is a no-op with identical outcome");
        assert_eq!(second.commit, commits[0]);
    }

    #[test]
    fn sync_with_explicit_commit_pins_that_commit_not_the_tag() {
        let tmp = tempdir().unwrap();
        let canonical = tmp.path().join("canonical");
        // Tag points at commit 1; we pin commit 2 explicitly.
        let commits = init_canonical(&canonical, 2, "v0.0.1", 1);
        let store = tmp.path().join("store");

        let outcome = sync(
            &store,
            &canonical.to_string_lossy(),
            None,
            "0.0.1",
            Some(&commits[1]),
        )
        .expect("sync with explicit commit");
        assert_eq!(
            outcome.commit, commits[1],
            "explicit pin_commit wins over the tag"
        );
        assert_ne!(outcome.commit, commits[0]);
    }

    #[test]
    fn sync_missing_tag_errors() {
        let tmp = tempdir().unwrap();
        let canonical = tmp.path().join("canonical");
        init_canonical(&canonical, 1, "v0.0.1", 1);
        let store = tmp.path().join("store");

        let err = sync(&store, &canonical.to_string_lossy(), None, "9.9.9", None)
            .expect_err("missing tag must error");
        assert!(
            matches!(err, ChelisSrcError::TagNotFound { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn sync_missing_explicit_commit_errors() {
        let tmp = tempdir().unwrap();
        let canonical = tmp.path().join("canonical");
        init_canonical(&canonical, 1, "v0.0.1", 1);
        let store = tmp.path().join("store");
        let absent = "0".repeat(40);

        let err = sync(
            &store,
            &canonical.to_string_lossy(),
            None,
            "0.0.1",
            Some(&absent),
        )
        .expect_err("absent commit must error");
        assert!(
            matches!(err, ChelisSrcError::CommitNotFound { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn ensure_worktree_refuses_to_clobber_a_non_worktree_slot() {
        let tmp = tempdir().unwrap();
        let canonical = tmp.path().join("canonical");
        let commits = init_canonical(&canonical, 1, "v0.0.1", 1);
        let store = tmp.path().join("store");
        let mirror = ensure_mirror(&store, &canonical.to_string_lossy(), None).unwrap();
        // A stray directory squats the slot (e.g. an old dev clone).
        let slot = store.join("0.0.1");
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::write(slot.join("keep.txt"), "not a worktree").unwrap();

        let err = ensure_worktree(&store, &mirror, "0.0.1", &commits[0])
            .expect_err("occupied slot must error");
        assert!(
            matches!(err, ChelisSrcError::SlotOccupied { .. }),
            "got {err:?}"
        );
        assert!(
            slot.join("keep.txt").exists(),
            "stray content is left intact"
        );
    }

    #[test]
    fn worktree_head_is_none_before_sync_and_detects_drift_after() {
        let tmp = tempdir().unwrap();
        let canonical = tmp.path().join("canonical");
        let commits = init_canonical(&canonical, 2, "v0.0.1", 2); // tag at commit 2
        let store = tmp.path().join("store");

        assert_eq!(
            worktree_head(&store, "0.0.1").unwrap(),
            None,
            "absent before sync"
        );

        sync(&store, &canonical.to_string_lossy(), None, "0.0.1", None).unwrap();
        assert_eq!(
            worktree_head(&store, "0.0.1").unwrap().as_deref(),
            Some(commits[1].as_str())
        );

        // Simulate a developer repointing the slot off the pin.
        let wt = store.join("0.0.1");
        git_in(&wt, &["checkout", "--detach", "--quiet", &commits[0]]);
        assert_eq!(
            worktree_head(&store, "0.0.1").unwrap().as_deref(),
            Some(commits[0].as_str()),
            "drift is observable via worktree_head"
        );

        // A re-sync heals the drift back to the pin.
        sync(&store, &canonical.to_string_lossy(), None, "0.0.1", None).unwrap();
        assert_eq!(
            worktree_head(&store, "0.0.1").unwrap().as_deref(),
            Some(commits[1].as_str()),
            "re-sync restores the pinned commit"
        );
    }

    #[cfg(unix)]
    #[test]
    fn wire_slot_creates_repoints_and_is_idempotent() {
        let tmp = tempdir().unwrap();
        let store = tmp.path().join("store");
        let wt_a = store.join("0.0.1");
        let wt_b = store.join("0.0.2");
        std::fs::create_dir_all(&wt_a).unwrap();
        std::fs::create_dir_all(&wt_b).unwrap();
        // The slot's parent does not exist yet — wire_slot must create it.
        let slot = tmp.path().join("shell-parent").join("chelis");

        assert_eq!(wire_slot(&slot, &wt_a).unwrap(), WireOutcome::Created);
        assert_eq!(std::fs::read_link(&slot).unwrap(), wt_a);
        assert_eq!(wire_slot(&slot, &wt_a).unwrap(), WireOutcome::AlreadyWired);
        match wire_slot(&slot, &wt_b).unwrap() {
            WireOutcome::Repointed { previous } => assert_eq!(previous, wt_a),
            other => panic!("expected Repointed, got {other:?}"),
        }
        assert_eq!(std::fs::read_link(&slot).unwrap(), wt_b);
    }

    #[cfg(unix)]
    #[test]
    fn wire_slot_refuses_a_real_directory_and_leaves_it_intact() {
        let tmp = tempdir().unwrap();
        let wt = tmp.path().join("store").join("0.0.1");
        std::fs::create_dir_all(&wt).unwrap();
        // A real chelis dev clone squats the slot.
        let slot = tmp.path().join("chelis");
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::write(slot.join("Cargo.toml"), "[workspace]\n").unwrap();

        let err = wire_slot(&slot, &wt).expect_err("a real dir must be refused");
        assert!(matches!(err, ChelisSrcError::Wiring { .. }), "got {err:?}");
        assert!(
            slot.join("Cargo.toml").exists(),
            "the dev clone is left intact"
        );
        assert!(
            !std::fs::symlink_metadata(&slot)
                .unwrap()
                .file_type()
                .is_symlink(),
            "the slot was not converted to a symlink"
        );
    }

    #[cfg(unix)]
    #[test]
    fn check_slot_passes_when_wired_and_fails_on_drift_or_missing() {
        let tmp = tempdir().unwrap();
        let store = tmp.path().join("store");
        let wt = store.join("0.0.1");
        let other = store.join("0.0.2");
        std::fs::create_dir_all(&wt).unwrap();
        std::fs::create_dir_all(&other).unwrap();
        let slot = tmp.path().join("chelis");

        assert!(check_slot(&slot, &wt).is_err(), "missing slot fails check");

        wire_slot(&slot, &wt).unwrap();
        check_slot(&slot, &wt).expect("a correctly wired slot passes");

        // Repointed off the pin: drift must be loud.
        wire_slot(&slot, &other).unwrap();
        let err = check_slot(&slot, &wt).expect_err("drift must fail check");
        assert!(matches!(err, ChelisSrcError::Wiring { .. }), "got {err:?}");

        // A real directory in the slot is not a symlink → fails check.
        std::fs::remove_file(&slot).unwrap();
        std::fs::create_dir_all(&slot).unwrap();
        assert!(
            check_slot(&slot, &wt).is_err(),
            "a non-symlink slot fails check"
        );
    }

    #[test]
    fn cargo_lock_mismatches_flags_only_off_pin_declared_crates() {
        let lock = r#"
[[package]]
name = "chelis-types"
version = "0.12.0"

[[package]]
name = "chelis-ir"
version = "0.8.0"

[[package]]
name = "serde"
version = "1.0.0"
"#;
        let crates = vec!["chelis-types".to_string(), "chelis-ir".to_string()];
        // Pin 0.8.0: chelis-types is off-pin (0.12.0); chelis-ir is on-pin;
        // serde is not a declared chelis crate and is ignored.
        assert_eq!(
            cargo_lock_mismatches(lock, &crates, "0.8.0").unwrap(),
            vec!["chelis-types=0.12.0".to_string()]
        );
        // Pin 0.12.0 flips which one drifts.
        assert_eq!(
            cargo_lock_mismatches(lock, &crates, "0.12.0").unwrap(),
            vec!["chelis-ir=0.8.0".to_string()]
        );
        // A declared crate absent from the lock is not a mismatch.
        assert!(
            cargo_lock_mismatches(lock, &["chelis-deep".to_string()], "0.8.0")
                .unwrap()
                .is_empty()
        );
    }
}
