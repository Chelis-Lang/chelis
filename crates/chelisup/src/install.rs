//! Toolchain install + store management.
//!
//! `chelisup install <ver>` downloads the host's release tarball
//! (`chelis-vX.Y.Z-<build>.tar.gz`, see [`release_builds`]) from
//! `Chelis-Lang/chelis` release `v<ver>`, unpacks it, checks its runtime files
//! against the unpacked compiler's `chelis runtime export` (see
//! [`crate::runtime_check`]), moves it to `<home>/toolchains/<ver>/`,
//! installs/refreshes the `chelis` shim, and seeds the default on the first
//! install. A failed check leaves the store untouched. It is idempotent: an
//! already-installed version refreshes the shim only.
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
//!   authenticated via `GITHUB_TOKEN` or `gh auth token` when available,
//!   otherwise anonymous. The API form serves **private**-repo asset bytes;
//!   the public
//!   `/releases/download/...` URL does not. The API base is overridable
//!   via `CHELISUP_GITHUB_BASE_API` (the wiremock seam), and the repo via
//!   `CHELISUP_REPO`.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::paths::Store;
use crate::runtime_check::{self, RuntimeCheck};
use crate::version::{release_triple, validate_install_version};

const DEFAULT_REPO: &str = "Chelis-Lang/chelis";
const DEFAULT_API_BASE: &str = "https://api.github.com";

/// What an install did. The CLI prints a different line for each.
#[derive(Debug, PartialEq, Eq)]
pub enum InstallOutcome {
    /// The toolchain was downloaded from release build `build` (the first of
    /// [`release_builds`] the release publishes), its runtime files checked,
    /// and unpacked into the store.
    Installed {
        version: String,
        build: String,
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

/// The first release from which every release publishes a Linux build made
/// against glibc 2.31 (`chelis-vX.Y.Z-linux-x86_64-glibc2.31.tar.gz`, #330).
/// The `linux-x86_64` build needs the glibc of the runner that built it (2.39
/// for recent releases), so Linux never installs it from this release on
/// (chelis#2686). Of the earlier releases only 0.7.18 publishes one; they all
/// install their `linux-x86_64` build.
const FIRST_GLIBC_231_RELEASE: &str = "0.7.24";

/// The release builds `install` accepts for `version` on the platform `slug`,
/// in preference order, on a host whose programs use musl when `musl` is set
/// (see [`host_is_musl`]). From [`FIRST_GLIBC_231_RELEASE`] on, Linux prefers the
/// static build (`chelis-vX.Y.Z-linux-x86_64-static.tar.gz`), whose `chelis`
/// names no program interpreter and so starts on any x86-64 Linux, including a
/// stock NixOS. A release that publishes no static build installs its
/// glibc-2.31 build. A musl host first takes the musl build
/// (`chelis-vX.Y.Z-linux-x86_64-musl.tar.gz`): the static build carries a glibc
/// runtime archive, which the host's C compiler cannot link
/// (`spec/08-backends.md` §2.1). `version` is a validated `X.Y.Z`.
pub fn release_builds(version: &str, slug: &str, musl: bool) -> Result<Vec<String>, String> {
    if slug == "linux-x86_64"
        && release_triple(version)? >= release_triple(FIRST_GLIBC_231_RELEASE)?
    {
        let musl = musl.then(|| format!("{slug}-musl"));
        return Ok(musl
            .into_iter()
            .chain([format!("{slug}-static"), format!("{slug}-glibc2.31")])
            .collect());
    }
    Ok(vec![slug.to_owned()])
}

/// The build `install` prefers for `version` on `slug`: the first of
/// [`release_builds`].
pub fn release_build(version: &str, slug: &str, musl: bool) -> Result<String, String> {
    Ok(release_builds(version, slug, musl)?.remove(0))
}

/// Whether this host's programs use musl as their C library: whether `/bin/sh`
/// names musl's dynamic loader (`ld-musl-<arch>.so.1`) as its program
/// interpreter. chelisup is a static executable, so it cannot ask its own C
/// library, and a glibc system can hold musl's loader too, for `musl-gcc`
/// (Debian's `musl` package installs it), so the loader's presence does not
/// tell. A static or unreadable `/bin/sh` counts as not musl.
pub fn host_is_musl() -> bool {
    cfg!(target_os = "linux")
        && fs::read("/bin/sh")
            .ok()
            .as_deref()
            .and_then(elf_interpreter)
            .is_some_and(names_musl_loader)
}

/// Whether the program interpreter `path` is musl's dynamic loader.
fn names_musl_loader(path: &str) -> bool {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("ld-musl-"))
}

/// The program interpreter (`PT_INTERP`) a 64-bit little-endian ELF file names.
fn elf_interpreter(elf: &[u8]) -> Option<&str> {
    const PT_INTERP: u32 = 3;
    fn field<const N: usize>(elf: &[u8], at: usize) -> Option<[u8; N]> {
        elf.get(at..at.checked_add(N)?)?.try_into().ok()
    }
    let word = |at: usize| {
        field::<8>(elf, at)
            .map(u64::from_le_bytes)
            .and_then(|value| usize::try_from(value).ok())
    };
    let half = |at: usize| field::<2>(elf, at).map(u16::from_le_bytes).map(usize::from);
    if elf.get(..6)? != b"\x7fELF\x02\x01" {
        return None;
    }
    let (table, entry_size, entries) = (word(0x20)?, half(0x36)?, half(0x38)?);
    (0..entries).find_map(|index| {
        let header = table.checked_add(index.checked_mul(entry_size)?)?;
        if field::<4>(elf, header).map(u32::from_le_bytes)? != PT_INTERP {
            return None;
        }
        let offset = word(header.checked_add(8)?)?;
        let size = word(header.checked_add(32)?)?;
        let path = elf.get(offset..offset.checked_add(size)?)?;
        std::str::from_utf8(path.split(|&byte| byte == 0).next()?).ok()
    })
}

/// The release-asset file name for a version + release build.
pub fn asset_name(version: &str, build: &str) -> String {
    format!("chelis-v{version}-{build}.tar.gz")
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

    let builds = release_builds(version, detect_slug()?, host_is_musl())?;

    // Stage under the store root so the final rename is same-filesystem.
    fs::create_dir_all(store.home())
        .map_err(|e| format!("could not create {}: {e}", store.home().display()))?;
    let scratch = tempfile::Builder::new()
        .prefix(".chelisup-install-")
        .tempdir_in(store.home())
        .map_err(|e| format!("could not create a staging directory: {e}"))?;

    let (build, tarball) = fetch_release(version, &builds, scratch.path())?;
    let unpacked = extract_tarball(&tarball, scratch.path())?;
    let runtime = runtime_check::check(version, &unpacked, scratch.path())?;
    install_into_store(store, version, &unpacked)?;

    ensure_shim_installed(store)?;
    ensure_default_seeded(store, version)?;

    Ok(InstallOutcome::Installed {
        version: version.to_string(),
        build,
        runtime,
    })
}

/// Fetch the first of `builds` that release `version` publishes into `dir`,
/// choosing the seam from the environment. Returns that build and its file.
fn fetch_release(
    version: &str,
    builds: &[String],
    dir: &Path,
) -> Result<(String, PathBuf), String> {
    let assets: Vec<String> = builds
        .iter()
        .map(|build| asset_name(version, build))
        .collect();
    let index = match std::env::var_os("CHELISUP_RELEASE_BASE") {
        Some(base) => fetch_from_base(&base.to_string_lossy(), &assets, dir)?,
        None => fetch_from_github(version, &assets, dir)?,
    };
    Ok((builds[index].clone(), dir.join(&assets[index])))
}

/// `CHELISUP_RELEASE_BASE` fetch of the first of `assets` present, into
/// `dir`: a local directory (or `file://`) read, or plain unauthenticated
/// `http(s)://` GETs that move past an HTTP 404 to the next asset.
fn fetch_from_base(base: &str, assets: &[String], dir: &Path) -> Result<usize, String> {
    if base.starts_with("http://") || base.starts_with("https://") {
        let mut missing = Vec::new();
        for (index, asset) in assets.iter().enumerate() {
            let url = format!("{}/{asset}", base.trim_end_matches('/'));
            match http_get_to_file(&url, &dir.join(asset), None) {
                Ok(()) => return Ok(index),
                Err(GetError::NotFound(_)) => missing.push(url),
                Err(GetError::Failed(message)) => return Err(message),
            }
        }
        return Err(format!(
            "release asset {} not found under CHELISUP_RELEASE_BASE: {} returned HTTP 404",
            assets.join(" or "),
            missing.join(" and ")
        ));
    }
    let root = Path::new(base.strip_prefix("file://").unwrap_or(base));
    let Some(index) = assets.iter().position(|asset| root.join(asset).is_file()) else {
        return Err(format!(
            "release asset {} not found under CHELISUP_RELEASE_BASE at {}",
            assets.join(" or "),
            root.display()
        ));
    };
    let src = root.join(&assets[index]);
    let target = dir.join(&assets[index]);
    fs::copy(&src, &target).map_err(|e| {
        format!(
            "could not copy {} to {}: {e}",
            src.display(),
            target.display()
        )
    })?;
    Ok(index)
}

/// GitHub REST two-step fetch (public or private) of the
/// first of `assets` the release lists, into `dir`.
fn fetch_from_github(version: &str, assets: &[String], dir: &Path) -> Result<usize, String> {
    let token = resolve_github_token().ok();
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
    let mut request = client
        .get(&meta_url)
        .header("User-Agent", "chelisup")
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28");
    if let Some(token) = &token {
        request = request.header("Authorization", format!("token {token}"));
    }
    let resp = request
        .send()
        .map_err(|e| format!("network error fetching {meta_url}: {e}"))?;
    if !resp.status().is_success() {
        return Err(github_status_error(
            &meta_url,
            resp.status().as_u16(),
            &tag,
            &repo,
            token.is_some(),
        ));
    }
    let body = resp
        .text()
        .map_err(|e| format!("could not read release metadata from {meta_url}: {e}"))?;
    let (index, asset_id) = find_asset_id(&body, assets, &meta_url)?;

    // Step 2: asset bytes by id.
    let asset_url = format!(
        "{}/repos/{}/releases/assets/{}",
        api.trim_end_matches('/'),
        repo,
        asset_id
    );
    http_get_to_file(
        &asset_url,
        &dir.join(&assets[index]),
        Some(GithubAuth {
            client: &client,
            token: token.as_deref(),
            tag: &tag,
            repo: &repo,
        }),
    )
    .map_err(GetError::into_message)?;
    Ok(index)
}

#[cfg(test)]
mod public_release_tests {
    use super::*;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn public_release_fetches_without_gh_and_private_release_reports_auth() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let server = rt.block_on(MockServer::start());
        let asset = "chelis-v0.19.0-test.tar.gz".to_string();
        let metadata = serde_json::json!({"assets": [{"id": 77, "name": asset}]}).to_string();
        rt.block_on(async {
            Mock::given(method("GET"))
                .and(path("/repos/Chelis-Lang/chelis/releases/tags/v0.19.0"))
                .respond_with(ResponseTemplate::new(200).set_body_string(metadata.clone()))
                .mount(&server)
                .await;
            Mock::given(method("GET"))
                .and(path("/repos/Chelis-Lang/chelis/releases/assets/77"))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(b"public bytes"))
                .mount(&server)
                .await;
            Mock::given(method("GET"))
                .and(path("/repos/Chelis-Lang/chelis/releases/tags/v0.19.1"))
                .respond_with(ResponseTemplate::new(401))
                .mount(&server)
                .await;
            Mock::given(method("GET"))
                .and(path("/repos/Chelis-Lang/chelis/releases/tags/v0.19.2"))
                .and(header("authorization", "token unit-test-token"))
                .respond_with(ResponseTemplate::new(200).set_body_string(metadata))
                .mount(&server)
                .await;
        });

        let old_base = std::env::var_os("CHELISUP_GITHUB_BASE_API");
        let old_token = std::env::var_os("GITHUB_TOKEN");
        let old_path = std::env::var_os("PATH");
        unsafe {
            std::env::set_var("CHELISUP_GITHUB_BASE_API", server.uri());
            std::env::remove_var("GITHUB_TOKEN");
            std::env::set_var("PATH", "");
        }
        let dir = tempfile::tempdir().unwrap();
        let result = fetch_from_github("0.19.0", std::slice::from_ref(&asset), dir.path());
        let private_error = fetch_from_github("0.19.1", std::slice::from_ref(&asset), dir.path())
            .expect_err("a private release must explain the missing token");
        unsafe { std::env::set_var("GITHUB_TOKEN", "unit-test-token") };
        let private_result = fetch_from_github("0.19.2", std::slice::from_ref(&asset), dir.path());
        unsafe {
            match old_base {
                Some(v) => std::env::set_var("CHELISUP_GITHUB_BASE_API", v),
                None => std::env::remove_var("CHELISUP_GITHUB_BASE_API"),
            }
            match old_token {
                Some(v) => std::env::set_var("GITHUB_TOKEN", v),
                None => std::env::remove_var("GITHUB_TOKEN"),
            }
            match old_path {
                Some(v) => std::env::set_var("PATH", v),
                None => std::env::remove_var("PATH"),
            }
        }
        assert_eq!(result.unwrap(), 0);
        assert_eq!(private_result.unwrap(), 0);
        assert_eq!(
            std::fs::read(dir.path().join(&asset)).unwrap(),
            b"public bytes"
        );
        assert!(private_error.contains("GITHUB_TOKEN"), "{private_error}");
        let requests = rt.block_on(server.received_requests()).unwrap();
        assert_eq!(requests.len(), 5);
        assert!(
            requests[..3]
                .iter()
                .all(|r| !r.headers.contains_key("authorization"))
        );
        assert!(
            requests[3..]
                .iter()
                .all(|r| r.headers.contains_key("authorization"))
        );
    }
}

/// Why a GET failed: the server has no such file (HTTP 404), or anything else.
enum GetError {
    NotFound(String),
    Failed(String),
}

impl GetError {
    fn into_message(self) -> String {
        match self {
            Self::NotFound(message) | Self::Failed(message) => message,
        }
    }
}

/// Optional GitHub auth context for [`http_get_to_file`].
struct GithubAuth<'a> {
    client: &'a reqwest::blocking::Client,
    token: Option<&'a str>,
    tag: &'a str,
    repo: &'a str,
}

/// GET `url` to `target`. With `auth`, send GitHub asset headers and map
/// HTTP failures to the loud GitHub error; without it, do a plain GET
/// (the `CHELISUP_RELEASE_BASE` http branch). HTTP 404 is
/// [`GetError::NotFound`].
fn http_get_to_file(url: &str, target: &Path, auth: Option<GithubAuth>) -> Result<(), GetError> {
    let owned_client;
    let (client, builder) = match &auth {
        Some(a) => {
            let mut builder = a
                .client
                .get(url)
                .header("User-Agent", "chelisup")
                .header("Accept", "application/octet-stream")
                .header("X-GitHub-Api-Version", "2022-11-28");
            if let Some(token) = a.token {
                builder = builder.header("Authorization", format!("token {token}"));
            }
            (a.client, builder)
        }
        None => {
            owned_client = reqwest::blocking::Client::builder()
                .connect_timeout(Duration::from_secs(15))
                .build()
                .map_err(|e| GetError::Failed(format!("could not build the HTTP client: {e}")))?;
            let b = owned_client.get(url).header("User-Agent", "chelisup");
            (&owned_client, b)
        }
    };
    let _ = client;

    let mut resp = builder
        .send()
        .map_err(|e| GetError::Failed(format!("network error fetching {url}: {e}")))?;
    let status = resp.status();
    if !status.is_success() {
        let message = match &auth {
            Some(a) => github_status_error(url, status.as_u16(), a.tag, a.repo, a.token.is_some()),
            None => format!("fetching {url} returned HTTP {}", status.as_u16()),
        };
        return Err(if status.as_u16() == 404 {
            GetError::NotFound(message)
        } else {
            GetError::Failed(message)
        });
    }
    let mut out = fs::File::create(target)
        .map_err(|e| GetError::Failed(format!("could not create {}: {e}", target.display())))?;
    resp.copy_to(&mut out)
        .map_err(|e| GetError::Failed(format!("could not download {url}: {e}")))?;
    Ok(())
}

/// Map a GitHub HTTP status to a loud, actionable message.
fn github_status_error(url: &str, status: u16, tag: &str, repo: &str, has_token: bool) -> String {
    match status {
        401 | 403 if !has_token => format!(
            "GitHub requires authentication (HTTP {status}) for {url}: \
             set GITHUB_TOKEN or run gh auth login to access a private release"
        ),
        401 | 403 => format!(
            "GitHub rejected the token (HTTP {status}) for {url}: \
             the token may be invalid or lack `contents: read` on {repo}"
        ),
        404 if !has_token => format!(
            "release {tag} on {repo} returned HTTP 404 for {url}. The release may be missing \
             or private; set GITHUB_TOKEN or run gh auth login if it is private"
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

/// The first of `assets` the release-metadata JSON lists: its position in
/// `assets` and its numeric id.
fn find_asset_id(body: &str, assets: &[String], url: &str) -> Result<(usize, u64), String> {
    let value: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| format!("release metadata at {url} is not valid JSON: {e}"))?;
    let listed = value
        .get("assets")
        .and_then(|a| a.as_array())
        .ok_or_else(|| format!("release metadata at {url} has no assets array"))?;
    for (index, asset) in assets.iter().enumerate() {
        for entry in listed {
            if entry.get("name").and_then(|n| n.as_str()) == Some(asset.as_str())
                && let Some(id) = entry.get("id").and_then(|i| i.as_u64())
            {
                return Ok((index, id));
            }
        }
    }
    let present: Vec<&str> = listed
        .iter()
        .filter_map(|a| a.get("name").and_then(|n| n.as_str()))
        .collect();
    let listing = if present.is_empty() {
        "(release has no attached assets)".to_string()
    } else {
        format!("present assets: [{}]", present.join(", "))
    };
    Err(format!(
        "release asset {} not found at {url}; {listing}",
        assets.join(" or ")
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
/// top-level `chelis-v*` directory, a real directory rather than a link,
/// verified to contain `bin/chelis`.
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
        // `DirEntry::file_type` does not follow links. Through a linked root,
        // the runtime check and the store entry would reach outside the
        // release.
        .filter(|e| {
            e.file_type().is_ok_and(|kind| kind.is_dir())
                && e.file_name()
                    .to_str()
                    .is_some_and(|n| n.starts_with("chelis-v"))
        })
        .map(|e| e.path())
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
            "expected exactly one chelis-v* directory (not a link) in the tarball, found {names:?}"
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
    fn linux_prefers_the_static_build_from_the_first_glibc_2_31_release() {
        let both = ["linux-x86_64-static", "linux-x86_64-glibc2.31"];
        let musl_first = [
            "linux-x86_64-musl",
            "linux-x86_64-static",
            "linux-x86_64-glibc2.31",
        ];
        for (version, slug, musl, builds) in [
            ("0.7.23", "linux-x86_64", false, &["linux-x86_64"][..]),
            ("0.7.24", "linux-x86_64", false, &both[..]),
            // Ordered as numbers: 0.10.0 is after 0.7.24.
            ("0.10.0", "linux-x86_64", false, &both[..]),
            ("0.19.0", "linux-x86_64", false, &both[..]),
            ("0.19.0", "darwin-arm64", false, &["darwin-arm64"][..]),
            // A musl host takes the musl build, then what a glibc host takes.
            ("0.7.23", "linux-x86_64", true, &["linux-x86_64"][..]),
            ("0.19.0", "linux-x86_64", true, &musl_first[..]),
        ] {
            assert_eq!(
                release_builds(version, slug, musl).unwrap(),
                builds,
                "{version} {slug} musl={musl}"
            );
            assert_eq!(release_build(version, slug, musl).unwrap(), builds[0]);
        }
    }

    /// A 64-bit little-endian ELF header and program header table: a `PT_LOAD`
    /// entry, then a `PT_INTERP` entry naming `interpreter` when one is given.
    fn elf_naming(interpreter: Option<&str>) -> Vec<u8> {
        const ENTRY: usize = 56;
        let entries = if interpreter.is_some() { 2 } else { 1 };
        let mut elf = vec![0u8; 64 + ENTRY * entries];
        elf[..6].copy_from_slice(b"\x7fELF\x02\x01");
        elf[0x20..0x28].copy_from_slice(&64u64.to_le_bytes());
        elf[0x36..0x38].copy_from_slice(&(ENTRY as u16).to_le_bytes());
        elf[0x38..0x3a].copy_from_slice(&(entries as u16).to_le_bytes());
        elf[64..68].copy_from_slice(&1u32.to_le_bytes());
        if let Some(path) = interpreter {
            let header = 64 + ENTRY;
            let offset = elf.len() as u64;
            elf[header..header + 4].copy_from_slice(&3u32.to_le_bytes());
            elf[header + 8..header + 16].copy_from_slice(&offset.to_le_bytes());
            elf[header + 32..header + 40].copy_from_slice(&(path.len() as u64 + 1).to_le_bytes());
            elf.extend_from_slice(path.as_bytes());
            elf.push(0);
        }
        elf
    }

    #[test]
    fn a_musl_host_is_one_whose_shell_names_the_musl_loader() {
        for (interpreter, musl) in [
            (Some("/lib/ld-musl-x86_64.so.1"), true),
            (Some("/lib64/ld-linux-x86-64.so.2"), false),
            (
                Some("/nix/store/0-glibc-2.40/lib/ld-linux-x86-64.so.2"),
                false,
            ),
            // A static shell.
            (None, false),
        ] {
            let elf = elf_naming(interpreter);
            assert_eq!(elf_interpreter(&elf), interpreter);
            assert_eq!(
                elf_interpreter(&elf).is_some_and(names_musl_loader),
                musl,
                "{interpreter:?}"
            );
        }
        assert_eq!(elf_interpreter(b"#!/bin/sh\n"), None);
        let truncated = elf_naming(Some("/lib/ld-musl-x86_64.so.1"));
        assert_eq!(elf_interpreter(&truncated[..100]), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn this_hosts_shell_names_a_program_interpreter() {
        let shell = fs::read("/bin/sh").expect("/bin/sh is readable");
        let interpreter = elf_interpreter(&shell).expect("/bin/sh names a program interpreter");
        assert!(interpreter.starts_with('/'), "{interpreter}");
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
        let wanted = ["chelis-v0.1.0-darwin-arm64.tar.gz".to_owned()];
        assert_eq!(find_asset_id(body, &wanted, "u").unwrap(), (0, 2));
    }

    #[test]
    fn find_asset_id_takes_the_first_preference_the_release_lists() {
        let preferences = [
            "chelis-v0.19.0-linux-x86_64-static.tar.gz".to_owned(),
            "chelis-v0.19.0-linux-x86_64-glibc2.31.tar.gz".to_owned(),
        ];
        let both = r#"{"assets":[
            {"id":1,"name":"chelis-v0.19.0-linux-x86_64-glibc2.31.tar.gz"},
            {"id":2,"name":"chelis-v0.19.0-linux-x86_64-static.tar.gz"}
        ]}"#;
        assert_eq!(find_asset_id(both, &preferences, "u").unwrap(), (0, 2));
        let fallback = r#"{"assets":[
            {"id":1,"name":"chelis-v0.19.0-linux-x86_64-glibc2.31.tar.gz"}
        ]}"#;
        assert_eq!(find_asset_id(fallback, &preferences, "u").unwrap(), (1, 1));
    }

    #[test]
    fn find_asset_id_lists_present_on_miss() {
        let body = r#"{"assets":[{"id":1,"name":"other.tar.gz"}]}"#;
        let err = find_asset_id(body, &["wanted.tar.gz".to_owned()], "u").unwrap_err();
        assert!(err.contains("wanted.tar.gz"), "{err}");
        assert!(err.contains("other.tar.gz"), "{err}");
    }

    #[test]
    fn github_404_message_is_ambiguity_aware() {
        let msg = github_status_error("u", 404, "v0.1.0", "Chelis-Lang/chelis", true);
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
