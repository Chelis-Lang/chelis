//! Toolchain install + store management.
//!
//! `chelisup install <ver>` downloads the host-platform release tarball
//! (`chelis-vX.Y.Z-<slug>.tar.gz`) from `Chelis-Lang/chelis` release
//! `v<ver>`, unpacks it, checks its runtime files against the unpacked
//! compiler's `chelis runtime export` (see [`crate::runtime_check`]), moves it
//! to `<home>/toolchains/<ver>/`, installs/refreshes the `chelis` shim, and
//! seeds the default on the first install. A failed check leaves the store
//! untouched. It is idempotent: an already-installed version refreshes the
//! shim only.
//!
//! # Fetch seams
//!
//! - `CHELISUP_RELEASE_BASE` (offline / air-gapped / the test seam):
//!   when set, the asset is read from `<base>/<asset>`. A `file://`
//!   prefix or a plain path is read from disk (no network, no token); an
//!   `http(s)://` prefix is a plain unauthenticated GET. This mirrors how
//!   chelis-reef injects `CHELIS_REEF_GITHUB_BASE_API` for hermetic
//!   tests.
//! - default (real installs): the GitHub REST two-step fetch
//!   (metadata-by-tag, then asset-by-id with `Accept: octet-stream`),
//!   authenticated via `GITHUB_TOKEN` or `gh auth token`. This is the
//!   only form that serves **private**-repo asset bytes; the public
//!   `/releases/download/...` URL does not. The API base is overridable
//!   via `CHELISUP_GITHUB_BASE_API` (the wiremock seam), and the repo via
//!   `CHELISUP_REPO`.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::paths::Store;
use crate::runtime_check::{self, RuntimeCheck};
use crate::version::validate_install_version;

const DEFAULT_REPO: &str = "Chelis-Lang/chelis";
const DEFAULT_API_BASE: &str = "https://api.github.com";

/// What an install did. The CLI prints a different line for each.
#[derive(Debug, PartialEq, Eq)]
pub enum InstallOutcome {
    /// The toolchain was downloaded, its runtime files checked, and unpacked
    /// into the store.
    Installed {
        version: String,
        slug: String,
        runtime: RuntimeCheck,
    },
    /// The toolchain was already present; the shim was refreshed.
    AlreadyInstalled { version: String },
}

/// Detect the release-asset platform slug for the host. The slugs match
/// `install_chelis_toolchain.py` exactly.
pub fn detect_slug() -> Result<&'static str, String> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Ok("darwin-arm64"),
        ("macos", "x86_64") => Ok("darwin-x86_64"),
        ("linux", "x86_64") => Ok("linux-x86_64"),
        (os, arch) => Err(format!(
            "unsupported host platform {os}/{arch}; \
             chelis releases ship darwin-arm64, darwin-x86_64, and linux-x86_64"
        )),
    }
}

/// The release-asset file name for a version + slug.
pub fn asset_name(version: &str, slug: &str) -> String {
    format!("chelis-v{version}-{slug}.tar.gz")
}

/// Install `version`. Idempotent. Validates the version, fetches and
/// unpacks the tarball, installs the shim, and seeds the default on the
/// first install.
///
/// GUARD (shim-corruption trap): `pub(crate)`, deliberately NOT `pub`. This
/// installs the shim by copying `current_exe()` into
/// `<home>/bin/{chelis,chelisup}` (see [`ensure_shim_installed`]), which is
/// correct only when the running executable IS chelisup. Another crate
/// calling this in-process (notably `chelis-cli`, the `chelis` compiler)
/// would copy the wrong binary over the shim. External callers must
/// subprocess the real `chelisup` binary instead (see `chelis reef setup` in
/// `chelis-cli`). Do NOT widen this to `pub`; the visibility is the guard.
pub(crate) fn install(store: &Store, version: &str) -> Result<InstallOutcome, String> {
    validate_install_version(version)?;

    if store.is_installed(version) {
        ensure_shim_installed(store)?;
        ensure_default_seeded(store, version)?;
        return Ok(InstallOutcome::AlreadyInstalled {
            version: version.to_string(),
        });
    }

    let slug = detect_slug()?;
    let asset = asset_name(version, slug);

    // Stage under the store root so the final rename is same-filesystem.
    fs::create_dir_all(store.home())
        .map_err(|e| format!("could not create {}: {e}", store.home().display()))?;
    let scratch = tempfile::Builder::new()
        .prefix(".chelisup-install-")
        .tempdir_in(store.home())
        .map_err(|e| format!("could not create a staging directory: {e}"))?;

    let tarball = scratch.path().join(&asset);
    fetch_asset(version, &asset, &tarball)?;
    let unpacked = extract_tarball(&tarball, scratch.path())?;
    let runtime = runtime_check::check(version, &unpacked, scratch.path())?;
    install_into_store(store, version, &unpacked)?;

    ensure_shim_installed(store)?;
    ensure_default_seeded(store, version)?;

    Ok(InstallOutcome::Installed {
        version: version.to_string(),
        slug: slug.to_string(),
        runtime,
    })
}

/// Fetch the asset to `target`, choosing the seam from the environment.
fn fetch_asset(version: &str, asset: &str, target: &Path) -> Result<(), String> {
    if let Some(base) = std::env::var_os("CHELISUP_RELEASE_BASE") {
        let base = base.to_string_lossy().into_owned();
        return fetch_from_base(&base, asset, target);
    }
    fetch_from_github(version, asset, target)
}

/// `CHELISUP_RELEASE_BASE` fetch: local directory (or `file://`) read,
/// or a plain unauthenticated `http(s)://` GET.
fn fetch_from_base(base: &str, asset: &str, target: &Path) -> Result<(), String> {
    if let Some(rest) = base
        .strip_prefix("http://")
        .map(|_| base)
        .or_else(|| base.strip_prefix("https://").map(|_| base))
    {
        let url = format!("{}/{asset}", rest.trim_end_matches('/'));
        return http_get_to_file(&url, target, None);
    }
    let dir = base.strip_prefix("file://").unwrap_or(base);
    let src = Path::new(dir).join(asset);
    if !src.is_file() {
        return Err(format!(
            "release asset {asset} not found under CHELISUP_RELEASE_BASE at {}",
            src.display()
        ));
    }
    fs::copy(&src, target).map_err(|e| {
        format!(
            "could not copy {} to {}: {e}",
            src.display(),
            target.display()
        )
    })?;
    Ok(())
}

/// GitHub REST two-step fetch (the real, private-repo-capable path).
fn fetch_from_github(version: &str, asset: &str, target: &Path) -> Result<(), String> {
    let token = resolve_github_token()?;
    let api = github_api_base();
    let repo = github_repo();
    let tag = format!("v{version}");

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| format!("could not build the HTTP client: {e}"))?;

    // Step 1: release metadata -> asset id.
    let meta_url = format!(
        "{}/repos/{}/releases/tags/{}",
        api.trim_end_matches('/'),
        repo,
        tag
    );
    let resp = client
        .get(&meta_url)
        .header("Authorization", format!("token {token}"))
        .header("User-Agent", "chelisup")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .map_err(|e| format!("network error fetching {meta_url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(github_status_error(
            &meta_url,
            resp.status().as_u16(),
            &tag,
            &repo,
        ));
    }
    let body = resp
        .text()
        .map_err(|e| format!("could not read release metadata from {meta_url}: {e}"))?;
    let asset_id = find_asset_id(&body, asset, &meta_url)?;

    // Step 2: asset bytes by id.
    let asset_url = format!(
        "{}/repos/{}/releases/assets/{}",
        api.trim_end_matches('/'),
        repo,
        asset_id
    );
    http_get_to_file(
        &asset_url,
        target,
        Some(GithubAuth {
            client: &client,
            token: &token,
            tag: &tag,
            repo: &repo,
        }),
    )
}

/// Optional GitHub auth context for [`http_get_to_file`].
struct GithubAuth<'a> {
    client: &'a reqwest::blocking::Client,
    token: &'a str,
    tag: &'a str,
    repo: &'a str,
}

/// GET `url` to `target`. With `auth`, send GitHub asset headers and map
/// HTTP failures to the loud GitHub error; without it, do a plain GET
/// (the `CHELISUP_RELEASE_BASE` http branch).
fn http_get_to_file(url: &str, target: &Path, auth: Option<GithubAuth>) -> Result<(), String> {
    let owned_client;
    let (client, builder) = match &auth {
        Some(a) => (
            a.client,
            a.client
                .get(url)
                .header("Authorization", format!("token {}", a.token))
                .header("User-Agent", "chelisup")
                .header("Accept", "application/octet-stream")
                .header("X-GitHub-Api-Version", "2022-11-28"),
        ),
        None => {
            owned_client = reqwest::blocking::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .build()
                .map_err(|e| format!("could not build the HTTP client: {e}"))?;
            let b = owned_client.get(url).header("User-Agent", "chelisup");
            (&owned_client, b)
        }
    };
    let _ = client;

    let mut resp = builder
        .send()
        .map_err(|e| format!("network error fetching {url}: {e}"))?;
    if !resp.status().is_success() {
        return match &auth {
            Some(a) => Err(github_status_error(
                url,
                resp.status().as_u16(),
                a.tag,
                a.repo,
            )),
            None => Err(format!(
                "fetching {url} returned HTTP {}",
                resp.status().as_u16()
            )),
        };
    }
    let mut out = fs::File::create(target)
        .map_err(|e| format!("could not create {}: {e}", target.display()))?;
    resp.copy_to(&mut out)
        .map_err(|e| format!("could not download {url}: {e}"))?;
    Ok(())
}

/// Map a GitHub HTTP status to a loud, actionable message.
fn github_status_error(url: &str, status: u16, tag: &str, repo: &str) -> String {
    match status {
        401 | 403 => format!(
            "GitHub rejected the token (HTTP {status}) for {url}: \
             the token may be invalid or lack `contents: read` on {repo}"
        ),
        404 => format!(
            "release {tag} on {repo} returned HTTP 404 for {url}. This is ambiguous: \
             either the release does not exist, or your token (GITHUB_TOKEN / `gh auth token`) \
             cannot read this private repo (GitHub returns 404, not 403, in that case). \
             Verify with: gh release view {tag} --repo {repo}"
        ),
        429 => format!("GitHub rate limit (HTTP 429) for {url}; retry later"),
        500..=599 => format!("GitHub returned HTTP {status} for {url}; not retrying"),
        other => format!("unexpected HTTP {other} for {url}"),
    }
}

/// Find the named asset's numeric id in the release-metadata JSON.
fn find_asset_id(body: &str, asset: &str, url: &str) -> Result<u64, String> {
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| format!("release metadata at {url} is not valid JSON: {e}"))?;
    let assets = value
        .get("assets")
        .and_then(|a| a.as_array())
        .ok_or_else(|| format!("release metadata at {url} has no assets array"))?;
    for entry in assets {
        if entry.get("name").and_then(|n| n.as_str()) == Some(asset)
            && let Some(id) = entry.get("id").and_then(|i| i.as_u64())
        {
            return Ok(id);
        }
    }
    let present: Vec<&str> = assets
        .iter()
        .filter_map(|a| a.get("name").and_then(|n| n.as_str()))
        .collect();
    let listing = if present.is_empty() {
        "(release has no attached assets)".to_string()
    } else {
        format!("present assets: [{}]", present.join(", "))
    };
    Err(format!(
        "release asset {asset} not found at {url}; {listing}"
    ))
}

/// Resolve a GitHub token: `GITHUB_TOKEN` first, then `gh auth token`.
fn resolve_github_token() -> Result<String, String> {
    if let Ok(t) = std::env::var("GITHUB_TOKEN") {
        let trimmed = t.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    let out = std::process::Command::new("gh")
        .args(["auth", "token"])
        .output()
        .map_err(|e| {
            format!(
                "GITHUB_TOKEN is not set and `gh auth token` could not be invoked: {e}. \
                 Run `export GITHUB_TOKEN=$(gh auth token)` and retry"
            )
        })?;
    if out.status.success() {
        let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if token.is_empty() {
            return Err(
                "GITHUB_TOKEN is not set and `gh auth token` returned an empty string. \
                        Run `export GITHUB_TOKEN=$(gh auth token)` and retry"
                    .to_string(),
            );
        }
        Ok(token)
    } else {
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        Err(format!(
            "GITHUB_TOKEN is not set and `gh auth token` failed: {stderr}. \
             Run `export GITHUB_TOKEN=$(gh auth token)` and retry"
        ))
    }
}

fn github_api_base() -> String {
    std::env::var("CHELISUP_GITHUB_BASE_API")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_API_BASE.to_string())
}

fn github_repo() -> String {
    std::env::var("CHELISUP_REPO")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_REPO.to_string())
}

/// Unpack `tarball` (gzip) into `dest` and return the single
/// top-level `chelis-v*` directory, verified to contain `bin/chelis`.
fn extract_tarball(tarball: &Path, dest: &Path) -> Result<PathBuf, String> {
    let file = fs::File::open(tarball)
        .map_err(|e| format!("could not open {}: {e}", tarball.display()))?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(dest)
        .map_err(|e| format!("could not extract {}: {e}", tarball.display()))?;

    let mut candidates: Vec<PathBuf> = fs::read_dir(dest)
        .map_err(|e| format!("could not read {}: {e}", dest.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("chelis-v"))
        })
        .collect();
    if candidates.len() != 1 {
        let names: Vec<String> = candidates
            .iter()
            .map(|p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        return Err(format!(
            "expected exactly one chelis-v* directory in the tarball, found {names:?}"
        ));
    }
    let unpacked = candidates.remove(0);
    let chelis = unpacked.join("bin").join("chelis");
    if !chelis.is_file() {
        return Err(format!(
            "extracted {} does not contain bin/chelis",
            unpacked.display()
        ));
    }
    // Make sure the compiler binary is executable even if the archive or
    // the extraction dropped the mode bit.
    set_executable(&chelis)?;
    Ok(unpacked)
}

/// Move `unpacked` into `<home>/toolchains/<ver>/`, replacing any
/// existing directory there. Falls back to a recursive copy if the
/// rename crosses a filesystem boundary.
fn install_into_store(store: &Store, version: &str, unpacked: &Path) -> Result<(), String> {
    let target = store.toolchain_dir(version);
    if target.exists() {
        fs::remove_dir_all(&target)
            .map_err(|e| format!("could not replace {}: {e}", target.display()))?;
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("could not create {}: {e}", parent.display()))?;
    }
    if fs::rename(unpacked, &target).is_err() {
        copy_dir_all(unpacked, &target)?;
    }
    if !store.is_installed(version) {
        return Err(format!(
            "post-install check failed: {} is missing after install",
            store.toolchain_bin(version).display()
        ));
    }
    Ok(())
}

/// Install/refresh the `chelis` shim and the `chelisup` copy by copying
/// this running executable into `<home>/bin/`.
///
/// GUARD (shim-corruption trap): `pub(crate)`, deliberately NOT `pub`. It
/// copies `current_exe()`; from any process that is not chelisup that writes
/// the wrong binary as the shim. Do NOT widen this to `pub`.
pub(crate) fn ensure_shim_installed(store: &Store) -> Result<(), String> {
    fs::create_dir_all(store.bin_dir())
        .map_err(|e| format!("could not create {}: {e}", store.bin_dir().display()))?;
    let current = std::env::current_exe()
        .map_err(|e| format!("could not locate the chelisup executable: {e}"))?;

    install_executable_copy(&current, &store.shim_path())?;

    let chelisup_path = store.chelisup_path();
    if !same_path(&current, &chelisup_path) {
        install_executable_copy(&current, &chelisup_path)?;
    }
    Ok(())
}

/// Seed the default on the first install only. Never overwrites an
/// existing default (installing must not repoint the machine default).
fn ensure_default_seeded(store: &Store, version: &str) -> Result<(), String> {
    if store.read_default().is_none() {
        store.write_default(version)?;
    }
    Ok(())
}

/// Atomically copy an executable `src` to `dest` (temp sibling + rename),
/// with mode 0755 on unix.
fn install_executable_copy(src: &Path, dest: &Path) -> Result<(), String> {
    let dir = dest
        .parent()
        .ok_or_else(|| format!("{} has no parent", dest.display()))?;
    let bytes = fs::read(src).map_err(|e| format!("could not read {}: {e}", src.display()))?;
    let mut tmp = tempfile::Builder::new()
        .prefix(".chelisup-shim-")
        .tempfile_in(dir)
        .map_err(|e| format!("could not stage a shim in {}: {e}", dir.display()))?;
    tmp.write_all(&bytes)
        .map_err(|e| format!("could not write the shim: {e}"))?;
    tmp.flush()
        .map_err(|e| format!("could not flush the shim: {e}"))?;
    set_executable(tmp.path())?;
    tmp.persist(dest)
        .map_err(|e| format!("could not install {}: {e}", dest.display()))?;
    Ok(())
}

#[cfg(unix)]
fn set_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o755))
        .map_err(|e| format!("could not chmod {}: {e}", path.display()))
}

#[cfg(not(unix))]
fn set_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// True iff both paths canonicalize to the same file. If `b` does not
/// exist yet (the common case for the first shim install), this is
/// false, which is the safe answer (do the copy).
fn same_path(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(ca), Ok(cb)) => ca == cb,
        _ => false,
    }
}

/// Recursively copy `src` into `dst`, preserving file permissions. Used
/// only when the staging rename crosses a filesystem boundary.
fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| format!("could not create {}: {e}", dst.display()))?;
    for entry in fs::read_dir(src).map_err(|e| format!("could not read {}: {e}", src.display()))? {
        let entry = entry.map_err(|e| format!("could not read a directory entry: {e}"))?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        let ty = entry
            .file_type()
            .map_err(|e| format!("could not stat {}: {e}", from.display()))?;
        if ty.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            fs::copy(&from, &to).map_err(|e| {
                format!("could not copy {} to {}: {e}", from.display(), to.display())
            })?;
        }
    }
    Ok(())
}

/// Remove an installed toolchain. Errors if it is not installed. Returns
/// whether the removed version was the recorded default (so the caller
/// can warn that the default is now unset).
///
/// `pub(crate)`: chelisup owns the toolchain lifecycle. Other crates drive
/// it through the `chelisup` binary, never in-process (see [`install`]).
pub(crate) fn uninstall(store: &Store, version: &str) -> Result<bool, String> {
    if !store.is_installed(version) {
        return Err(format!(
            "toolchain {version} is not installed.\n  installed: {}",
            installed_or_none(store)
        ));
    }
    let dir = store.toolchain_dir(version);
    fs::remove_dir_all(&dir).map_err(|e| format!("could not remove {}: {e}", dir.display()))?;
    let was_default = store.read_default().as_deref() == Some(version);
    if was_default {
        let _ = fs::remove_file(store.default_file());
    }
    Ok(was_default)
}

/// Removes the shim, installer copy, and recorded default.
/// Installed toolchains and the reef/src stores remain intact. The function
/// returns the list of paths removed.
///
/// `pub(crate)`: chelisup owns the toolchain lifecycle. Other crates drive
/// it through the `chelisup` binary, never in-process (see [`install`]).
pub(crate) fn self_uninstall(store: &Store) -> Result<Vec<PathBuf>, String> {
    let mut removed = Vec::new();
    for path in [
        store.shim_path(),
        store.chelisup_path(),
        store.default_file(),
    ] {
        if path.exists() || path.is_symlink() {
            fs::remove_file(&path)
                .map_err(|e| format!("could not remove {}: {e}", path.display()))?;
            removed.push(path);
        }
    }
    Ok(removed)
}

fn installed_or_none(store: &Store) -> String {
    let v = store.installed_versions();
    if v.is_empty() {
        "(none)".to_string()
    } else {
        v.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn asset_name_matches_release_convention() {
        assert_eq!(
            asset_name("0.12.0", "darwin-arm64"),
            "chelis-v0.12.0-darwin-arm64.tar.gz"
        );
    }

    #[test]
    fn detect_slug_is_supported_here() {
        // CI runs on darwin-arm64 / linux-x86_64; both are supported.
        assert!(detect_slug().is_ok());
    }

    #[test]
    fn find_asset_id_picks_the_named_asset() {
        let body = r#"{"assets":[
            {"id":1,"name":"chelis-v0.1.0-linux-x86_64.tar.gz"},
            {"id":2,"name":"chelis-v0.1.0-darwin-arm64.tar.gz"}
        ]}"#;
        assert_eq!(
            find_asset_id(body, "chelis-v0.1.0-darwin-arm64.tar.gz", "u").unwrap(),
            2
        );
    }

    #[test]
    fn find_asset_id_lists_present_on_miss() {
        let body = r#"{"assets":[{"id":1,"name":"other.tar.gz"}]}"#;
        let err = find_asset_id(body, "wanted.tar.gz", "u").unwrap_err();
        assert!(err.contains("wanted.tar.gz"), "{err}");
        assert!(err.contains("other.tar.gz"), "{err}");
    }

    #[test]
    fn github_404_message_is_ambiguity_aware() {
        let msg = github_status_error("u", 404, "v0.1.0", "Chelis-Lang/chelis");
        assert!(msg.contains("gh release view v0.1.0"), "{msg}");
        assert!(msg.contains("private"), "{msg}");
    }

    #[test]
    fn install_into_store_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let store = Store::at(tmp.path().join("home"));
        // Fake an unpacked toolchain directory.
        let unpacked = tmp.path().join("chelis-v0.1.0-linux-x86_64");
        fs::create_dir_all(unpacked.join("bin")).unwrap();
        fs::write(unpacked.join("bin").join("chelis"), b"#!/bin/true\n").unwrap();
        install_into_store(&store, "0.1.0", &unpacked).unwrap();
        assert!(store.is_installed("0.1.0"));
    }
}
