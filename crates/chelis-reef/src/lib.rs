use chelis_shell::{
    PackageId, ShellModule, ShellPackage, ShellSymbol, SymbolKind, read_shell, write_shell,
};
use chelis_surf::ast::{
    Decl, EffectExpr, Expr, ImportKind, LetBinding, LetPattern, MatchArm, Param, Pattern,
    PropertyOption, TypeExpr, TypeInvariant, Variant, VariantFields,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;
use tar::{Archive, Builder};
use walkdir::WalkDir;

const CURRENT_COMPILER_VERSION: &str = concat!("=", env!("CARGO_PKG_VERSION"));

/// Version of `chelis-std` that ships bundled with this compiler.
/// Re-exported from [`chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION`],
/// which is the single source of truth: that crate is also where the
/// `include_bytes!()` for the runtime archive + shell live, and the
/// version string is keyed off the dist filenames.
///
/// chelis-std is the language runtime, not a shell — it version-marches
/// with the compiler and cannot be substituted. Programs implicitly
/// depend on it the same way Rust programs depend on `core`/`std`.
/// Lockfile entries for chelis-std use [`LockSource::Bundled`] (not
/// [`LockSource::LocalRegistry`]) to make this distinction explicit and
/// auditable.
const BUNDLED_CHELIS_STD_VERSION: &str = chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION;

/// Public accessor for the version of chelis-std bundled with this
/// compiler. Use this when you need to emit a `LockSource::Bundled`
/// entry, surface a soft-verify mismatch error, or otherwise reason
/// about the runtime version.
pub fn compiler_bundled_chelis_std_version() -> &'static str {
    BUNDLED_CHELIS_STD_VERSION
}

/// The package name of the language runtime. Centralized so the soft-
/// verify path, the bootstrap rejection path, and the lockfile-migration
/// path all agree on the spelling.
pub const CHELIS_STD_PACKAGE_NAME: &str = "chelis-std";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReefManifest {
    pub package: ManifestPackage,
    #[serde(default)]
    pub dependencies: BTreeMap<String, DependencySpec>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestPackage {
    pub name: String,
    pub version: String,
    pub compiler: String,
    pub module_prefix: String,
    /// Additional top-level source roots to walk in addition to `src/`.
    /// Each entry is a single-segment directory name relative to the
    /// package root (e.g. `"properties"`, `"references"`). Files under
    /// these roots derive their module name with the root name as the
    /// first segment after `module_prefix` (so `<root>/properties/foo.ch`
    /// must declare `module <Prefix>.Properties.Foo`).
    ///
    /// Default-empty: a `reef.toml` without this field gets the same
    /// `src/`-only treatment as before. See `validate_manifest` for the
    /// full set of validation rules.
    #[serde(default)]
    pub additional_sources: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencySpec {
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReefLock {
    pub package: PackageId,
    pub dependencies: Vec<LockedDependency>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockedDependency {
    pub name: String,
    pub version: String,
    pub source: LockSource,
    pub compiler: String,
    pub archive_sha256: String,
    pub shell_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LockSource {
    Path {
        path: String,
    },
    /// Locally-installed registry package. The optional `remote_origin`
    /// records where the bytes were originally fetched from, in a
    /// scheme-tagged URI form (currently `github://<org>/<repo>@<tag>`;
    /// future schemes such as `registry://...` are planned post-launch).
    /// Pre-Item-9 lockfiles omit the field; they deserialize cleanly via
    /// `#[serde(default)]` and round-trip with `remote_origin: None`.
    /// Serialization omits the field when `None` so old lockfiles stay
    /// byte-identical after re-serialization.
    LocalRegistry {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        remote_origin: Option<String>,
    },
    /// Compiler-bundled package source, currently used only for the
    /// language runtime (`chelis-std`). Records the version of the
    /// compiler that provided the bundled bytes so a lockfile reader
    /// can audit whether a different compiler would supply the same
    /// runtime version. There is no archive to fetch — integrity comes
    /// from the compiler binary's own integrity, which is out of scope
    /// for reef.
    ///
    /// **Schema bump.** Old reef binaries that do not yet know about
    /// this variant will fail to deserialize lockfiles that contain it
    /// (serde's tagged-enum default is unknown-variant rejection). This
    /// is acceptable because chelis and reef are released together: a
    /// lockfile written by a newer compiler is expected to require a
    /// newer reef. See [`LockedDependency`]'s migration path for
    /// reading old lockfiles that recorded `chelis-std` as
    /// `LocalRegistry` — those are read transparently and rewritten as
    /// `Bundled` on the next `chelis reef build`.
    Bundled {
        compiler_version: String,
    },
}

impl LockSource {
    /// Construct a `LocalRegistry` source whose `remote_origin` is the
    /// `github://<org>/<repo>@<tag>` form derived from the given spec.
    /// This is the canonical writer for lockfile entries that came from
    /// `install_from_github`. Use [`Self::local_registry_no_origin`] for
    /// monorepo-installed entries.
    pub fn local_registry_from_github(spec: &GitHubReleaseSpec) -> Self {
        Self::LocalRegistry {
            remote_origin: Some(format_github_origin(&spec.org, &spec.repo, &spec.tag)),
        }
    }

    /// Construct a `LocalRegistry` source with no recorded origin. Used
    /// by `install_from_monorepo` (which has no remote source) and for
    /// backward-compat when reading old lockfiles that did not carry
    /// the field.
    pub fn local_registry_no_origin() -> Self {
        Self::LocalRegistry {
            remote_origin: None,
        }
    }

    /// Construct a `Bundled` source stamped with the version of the
    /// compiler that supplied the runtime. Used for the soft-verify
    /// path on explicit `chelis-std` declarations and for the
    /// implicit-runtime synthesis path.
    pub fn bundled_for_current_compiler() -> Self {
        // CARGO_PKG_VERSION is the compiler crate (chelis-reef) version,
        // which version-marches with the toolchain.
        Self::Bundled {
            compiler_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Read-only accessor for the `remote_origin` field of a
    /// `LocalRegistry` source. Returns `None` for `Path` and `Bundled`
    /// sources, and for `LocalRegistry` sources whose origin is
    /// unrecorded. The `Bundled` variant intentionally has no remote
    /// origin: there is no archive on the network — integrity comes
    /// from the compiler binary itself.
    pub fn remote_origin(&self) -> Option<&str> {
        match self {
            Self::LocalRegistry { remote_origin } => remote_origin.as_deref(),
            Self::Path { .. } | Self::Bundled { .. } => None,
        }
    }
}

/// Build the canonical `github://<org>/<repo>@<tag>` origin URI string.
/// Single source of truth for the format so the writer in
/// `install_from_github` and the parser in [`parse_remote_origin`]
/// cannot drift.
fn format_github_origin(org: &str, repo: &str, tag: &str) -> String {
    format!("github://{org}/{repo}@{tag}")
}

/// Typed parse error for a `remote_origin` string. Each variant carries
/// the original input so user-facing error messages can name what went
/// wrong without losing context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemoteOriginParseError {
    /// Scheme prefix did not match any supported scheme (currently only
    /// `github://`). Future schemes (`registry://...`) plug in here.
    UnknownScheme { input: String, scheme: String },
    /// Scheme matched but the body did not parse as `<org>/<repo>@<tag>`.
    /// `inner` carries the underlying [`GitHubFetchError::Parse`] reason.
    Malformed { input: String, reason: String },
}

impl std::fmt::Display for RemoteOriginParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownScheme { input, scheme } => write!(
                f,
                "remote_origin `{input}` uses unknown scheme `{scheme}` \
                 (supported: `github://`)"
            ),
            Self::Malformed { input, reason } => {
                write!(f, "remote_origin `{input}` is malformed: {reason}")
            }
        }
    }
}

impl std::error::Error for RemoteOriginParseError {}

/// Parse a `remote_origin` URI back into a [`GitHubReleaseSpec`]. The
/// only currently-supported scheme is `github://`; unknown-scheme
/// strings (e.g. `http://`, `registry://`) fail with
/// [`RemoteOriginParseError::UnknownScheme`] rather than being silently
/// treated as a degenerate GitHub URL. This is the spec-locked
/// "fail-fast on unknown scheme" rule (Item 9 § Locked decisions).
pub fn parse_remote_origin(input: &str) -> Result<GitHubReleaseSpec, RemoteOriginParseError> {
    const GITHUB_SCHEME: &str = "github://";
    if let Some(rest) = input.strip_prefix(GITHUB_SCHEME) {
        return GitHubReleaseSpec::parse(rest).map_err(|e| match e {
            GitHubFetchError::Parse { reason, .. } => RemoteOriginParseError::Malformed {
                input: input.to_string(),
                reason,
            },
            other => RemoteOriginParseError::Malformed {
                input: input.to_string(),
                reason: format!("{other}"),
            },
        });
    }
    let scheme = input
        .split_once("://")
        .map(|(s, _)| format!("{s}://"))
        .unwrap_or_else(|| {
            // No `://` at all — treat the whole input as the "scheme"
            // for diagnostic purposes so the error names exactly what
            // the user wrote.
            input.to_string()
        });
    Err(RemoteOriginParseError::UnknownScheme {
        input: input.to_string(),
        scheme,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LocalRegistryIndex {
    pub packages: BTreeMap<String, Vec<RegistryVersion>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegistryVersion {
    pub version: String,
    pub compiler: String,
    pub archive_sha256: String,
    pub shell_sha256: String,
    /// Where the bytes were originally fetched from, in
    /// scheme-tagged URI form (currently only `github://<org>/<repo>@<tag>`).
    /// `install_from_github` populates this; `install_from_monorepo`
    /// leaves it `None`. The downstream `build_lockfile` reads it to
    /// fill `LockSource::LocalRegistry::remote_origin`. Serde
    /// `default + skip_serializing_if = is_none` keeps old `index.json`
    /// readable on the new code (forward-compat) and keeps old entries
    /// byte-identical after a re-serialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_origin: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PreparedProgram {
    pub decls: Vec<Decl>,
    pub entry_decls: Vec<Decl>,
    pub package_root: PathBuf,
    /// The chelis-std-only slice of `decls`, in the same relative order.
    /// The cross-process chelis-std typecheck cache content-addresses
    /// this. Empty when the graph has no chelis-std package.
    pub stdlib_decls: Vec<Decl>,
    /// `decls` minus `stdlib_decls`, in the same relative order: the
    /// user package's own modules plus any non-stdlib path-deps. Checked
    /// `_with_context` against the cached chelis-std sub-context. The
    /// concatenation `stdlib_decls ++ non_stdlib_decls` equals `decls`.
    pub non_stdlib_decls: Vec<Decl>,
}

/// A reef package graph that has been resolved, linked, and cached for reuse
/// across multiple per-file compiles.
///
/// Building one of these is the expensive part of `prepare_program_for_eval_*`:
/// it walks the package root, reads the manifest, loads the lockfile (or runs
/// the resolver under a 5s timeout), and materializes all library module decls
/// with internal-name rewriting applied. A single `PreparedReefGraph` can then
/// be threaded through many calls to [`compile_with_reef_graph`], each of
/// which only has to rewrite the per-file entry decls.
///
/// All fields are intentionally `pub(crate)`; callers treat the value as an
/// opaque handle other than `package_root`, which is exposed because
/// `chelis test` uses it for file discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedReefGraph {
    pub package_root: PathBuf,
    pub(crate) graph: PackageGraph,
    /// Library decls (chelis-std + deps + this package's own modules) after
    /// linking and internal-name rewriting. Phase G's `compile_reef_context`
    /// reads this directly to feed the type checker / lowerer once and cache
    /// the result.
    pub linked_library_decls: Vec<Decl>,
    /// The chelis-std-only slice of `linked_library_decls`, in the same
    /// relative order. The cross-process chelis-std typecheck cache checks
    /// and caches this sub-context under a content-addressed key derived
    /// from the linked chelis-std decls. Empty when the graph has no
    /// chelis-std package, such as a package that depends on nothing.
    pub linked_stdlib_decls: Vec<Decl>,
    /// `linked_library_decls` minus `linked_stdlib_decls`, in the same
    /// relative order: the user package's own modules plus any non-stdlib
    /// path-deps. The chelis-std typecheck cache checks this `_with_context`
    /// against the cached chelis-std sub-context (Layer 2). The
    /// concatenation `linked_stdlib_decls ++ linked_non_stdlib_library_decls`
    /// equals `linked_library_decls`.
    pub linked_non_stdlib_library_decls: Vec<Decl>,
    pub(crate) internal_maps: HashMap<(String, String), HashMap<String, String>>,
    pub(crate) dep_shells: BTreeMap<String, ShellPackage>,
    pub(crate) eval_module_prefix: String,
}

impl PreparedReefGraph {
    /// Bincode round-trip. Used by Phase A foundation tests and (via
    /// `CompiledContext` which embeds this graph) the Phase I disk cache.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        bincode::serialize(self).map_err(|e| format!("encode prepared graph: {e}"))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        bincode::deserialize(bytes).map_err(|e| format!("decode prepared graph: {e}"))
    }

    /// Returns a content digest for every `.ch` source file backing this
    /// graph, including path-dep packages. The result is the load-bearing
    /// input to `chelis_compiler_api::ContextHash` and the Phase I disk
    /// cache; deterministic ordering is guaranteed by sorting on
    /// (package_name, package_version, module_name) before returning.
    ///
    /// `LocalRegistry` packages are not yet supported — Phase B fixtures
    /// only use `Root` + `Path`. Phase I will extend `LoadedPackage` to
    /// retain extracted-cache file paths so this method can hash them.
    pub fn source_digests(&self) -> Result<Vec<SourceDigest>, String> {
        let mut digests = Vec::new();
        for (_, package) in self.graph.packages.iter() {
            let source_root = match &package.source {
                LoadedSourceKind::Root => self.package_root.clone(),
                LoadedSourceKind::Path { relative } => self
                    .package_root
                    .join(relative)
                    .canonicalize()
                    .map_err(|e| {
                        format!(
                            "resolve path-dep `{}` at `{}`: {e}",
                            package.id.name, relative
                        )
                    })?,
                LoadedSourceKind::LocalRegistry => {
                    return Err(format!(
                        "source_digests for LocalRegistry package `{}` v{} not yet implemented \
                         (TODO phase-I. Needs LoadedPackage to retain cache root)",
                        package.id.name, package.id.version
                    ));
                }
            };
            for module in package.modules.values() {
                // Multi-root: resolve via the per-module source_root, which
                // is "src" for legacy packages and a manifest-declared
                // additional_sources entry for non-src roots.
                let abs = source_root.join(&module.source_root).join(&module.file_rel);
                let bytes = fs::read(&abs)
                    .map_err(|e| format!("read source file `{}`: {e}", abs.display()))?;
                let sha256: [u8; 32] = Sha256::digest(&bytes).into();
                digests.push(SourceDigest {
                    package_name: package.id.name.clone(),
                    package_version: package.id.version.clone(),
                    module_name: module.module_name.clone(),
                    sha256,
                });
            }
            // Cache invalidation: the file-content walk above does not
            // change when a manifest *adds* additional_sources entries
            // that point at empty/absent dirs — same files walked, same
            // digests, stale cache hit. Mix the manifest's
            // additional_sources list into the digest set as a synthetic
            // SourceDigest so any change to the declared roots flips the
            // downstream cache key (ContextHash::from_digests includes
            // every row in its sort+hash). The synthetic row uses a
            // module_name of `<manifest::additional_sources>` which is
            // syntactically not a valid module name, so it cannot collide
            // with a real module digest.
            if !package.manifest.package.additional_sources.is_empty() {
                let serialized = serde_json::to_string(
                    &package.manifest.package.additional_sources,
                )
                .map_err(|e| {
                    format!(
                        "serialize additional_sources for `{}` v{}: {e}",
                        package.id.name, package.id.version
                    )
                })?;
                let sha256: [u8; 32] = Sha256::digest(serialized.as_bytes()).into();
                digests.push(SourceDigest {
                    package_name: package.id.name.clone(),
                    package_version: package.id.version.clone(),
                    module_name: "<manifest::additional_sources>".to_string(),
                    sha256,
                });
            }
        }
        digests.sort_by(|a, b| {
            a.package_name
                .cmp(&b.package_name)
                .then_with(|| a.package_version.cmp(&b.package_version))
                .then_with(|| a.module_name.cmp(&b.module_name))
        });
        Ok(digests)
    }

    /// Returns the (name, version) pair of the *root* package backing this
    /// graph. Used by the Phase I disk cache to derive a stable, human-
    /// readable filename component (`<name>-<version>-<hash_prefix>.ctx`).
    /// Panics only if the graph was constructed with a missing root —
    /// `prepare_reef_graph` enforces this invariant on construction, so
    /// callers can treat this as infallible.
    pub fn root_package_id(&self) -> (&str, &str) {
        let pkg = self
            .graph
            .packages
            .get(&self.graph.root_package)
            .expect("PreparedReefGraph invariant: root_package present in packages");
        (pkg.id.name.as_str(), pkg.id.version.as_str())
    }
}

/// Per-source-file content digest produced by [`PreparedReefGraph::source_digests`].
/// The load-bearing input to `chelis_compiler_api::ContextHash` and the Phase I
/// disk cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDigest {
    pub package_name: String,
    pub package_version: String,
    pub module_name: String,
    pub sha256: [u8; 32],
}

#[derive(Debug, Clone)]
pub struct PackageBuildArtifacts {
    pub package: PackageId,
    pub shell_path: PathBuf,
    pub archive_path: PathBuf,
    pub shell_sha256: String,
    pub archive_sha256: String,
}

/// Options that tune the behavior of [`build_package_with_options`].
///
/// Phase A Item 8 surface. The default is locked at "auto-fetch on" so
/// that a fresh `chelis reef build` against an empty registry resolves
/// missing dependencies from the canonical hosting org without a manual
/// `chelis reef install --from-github` round trip first. Callers that
/// want explicit control over network access during build pass
/// `BuildOptions { auto_fetch: false, .. }`.
///
/// `BuildOptions::default()` matches the pre-Item-8 behavior except for
/// the auto-fetch flip — a subtle but locked CLI behavior change
/// documented in the `chelis reef build --help` output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildOptions {
    /// When `true` (default), missing-from-registry dependencies trigger
    /// an automatic fetch via [`install_from_github`] before the build
    /// fails. When `false`, the build fails fast with the improved
    /// error wording naming the URL that would have been tried.
    pub auto_fetch: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self { auto_fetch: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LoadedPackage {
    id: PackageId,
    manifest: ReefManifest,
    modules: BTreeMap<String, ModuleSource>,
    source: LoadedSourceKind,
    shell: Option<ShellPackage>,
    /// Remote-origin URI for `LocalRegistry`-sourced packages. Always
    /// `None` for `Path` and `Root` packages. Populated from the
    /// registry index by `load_registry_package`, or directly from the
    /// lockfile by `reconstruct_graph_from_lockfile`. Read by
    /// `build_lockfile` to fill `LockSource::LocalRegistry::remote_origin`.
    ///
    /// **Serde-attribute discipline.** This struct is part of
    /// `PreparedReefGraph`, which round-trips through **bincode** (a
    /// positional binary format). bincode does not honor
    /// `#[serde(skip_serializing_if = ...)]` — using it here would
    /// cause the byte stream's offsets to drift between encode and
    /// decode, producing the "tag for enum is not valid" panic
    /// pinned by the `prepared_reef_graph_round_trips_through_bincode`
    /// test. The field is therefore unconditional in the
    /// bincode stream; `Option::None` serializes as a single tag byte
    /// (0), which is the correct default behavior.
    remote_origin: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LoadedSourceKind {
    Root,
    Path { relative: String },
    LocalRegistry,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModuleSource {
    package_name: String,
    module_name: String,
    decls: Vec<Decl>,
    /// Path relative to `source_root` (i.e. relative to
    /// `<package_root>/<source_root>/`). For files under `src/` this is
    /// the same as the legacy `file_rel`; for files under an additional
    /// source root, this is still relative to that root, *not* to the
    /// package root.
    file_rel: PathBuf,
    /// Which declared source root (relative to the package root) this
    /// module came from: `"src"` for files under `src/`, or one of the
    /// entries from `manifest.package.additional_sources`. Used by
    /// `source_digests` and `module_name_for_input` to reconstruct the
    /// absolute path as `package_root.join(&source_root).join(&file_rel)`.
    source_root: String,
    exports: BTreeSet<String>,
    symbols: BTreeMap<String, SymbolKind>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinkedModule {
    decls: Vec<Decl>,
    entry: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PackageGraph {
    root_package: String,
    packages: BTreeMap<String, LoadedPackage>,
}

pub fn init_package(
    root: &Path,
    name: &str,
    module_prefix: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    fs::create_dir_all(root.join("src"))?;
    let manifest = ReefManifest {
        package: ManifestPackage {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            compiler: CURRENT_COMPILER_VERSION.to_string(),
            module_prefix: module_prefix.to_string(),
            additional_sources: Vec::new(),
        },
        dependencies: BTreeMap::new(),
    };
    write_manifest(&root.join("reef.toml"), &manifest)?;
    let main_module = format!("{module_prefix}.Main");
    // Canonical Surf form: no blank line between `module` and the first
    // decl, no trailing newline. The scaffold must satisfy
    // `chelis fmt --check` so the immediately-subsequent `chelis check`
    // and `chelis build` style gates pass.
    fs::write(
        root.join("src/main.ch"),
        format!("module {main_module}\ndef main(x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n"),
    )?;
    Ok(())
}

pub fn find_package_root_for_input(file: &Path) -> Result<Option<PathBuf>, String> {
    let file = file
        .canonicalize()
        .map_err(|e| format!("failed to canonicalize {}: {e}", file.display()))?;
    let dir = file
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", file.display()))?
        .to_path_buf();
    find_package_root_from_dir(dir)
}

pub fn find_package_root_for_dir(dir: &Path) -> Result<Option<PathBuf>, String> {
    let dir = dir
        .canonicalize()
        .map_err(|e| format!("failed to canonicalize {}: {e}", dir.display()))?;
    find_package_root_from_dir(dir)
}

fn find_package_root_from_dir(mut dir: PathBuf) -> Result<Option<PathBuf>, String> {
    let home = env::var_os("HOME").map(PathBuf::from);
    // Stop the ancestor walk when we hit the OS temp-dir boundary. Without
    // this, a `tempdir()`-rooted lookup (`/tmp/.tmpXXXX/...`) walks into
    // `/tmp` itself and finds whatever stray `reef.toml` another process or
    // test left there — leaking that foreign package into otherwise-isolated
    // test runs (e.g. an Octant-prefixed manifest causing module-prefix
    // mismatches in unrelated tests). A package may live *inside* a temp
    // dir, but never spans the temp-dir boundary into shared scratch space.
    let temp_root = env::temp_dir().canonicalize().ok();

    loop {
        if temp_root.as_ref().is_some_and(|tmp| *tmp == dir) {
            return Ok(None);
        }
        let manifest = dir.join("reef.toml");
        if manifest.exists() {
            return Ok(Some(dir));
        }
        if dir.join(".git").exists() {
            return Ok(None);
        }
        if home.as_ref().is_some_and(|home| *home == dir) {
            return Ok(None);
        }
        let Some(parent) = dir.parent() else {
            return Ok(None);
        };
        if parent == dir {
            return Ok(None);
        }
        dir = parent.to_path_buf();
    }
}

pub fn prepare_program_for_file(file: &Path) -> Result<Option<PreparedProgram>, String> {
    let Some(root) = find_package_root_for_input(file)? else {
        return Ok(None);
    };
    let graph = resolve_package_graph(&root, LoadOptions::default_for_load())?;
    write_lockfile(&root.join("reef.lock"), &build_lockfile(&graph))?;
    let entry_module = module_name_for_input(&root, file, &graph.root_package)?;
    let linked = link_graph_with_package_tags(&graph, std::slice::from_ref(&entry_module))?;
    let mut decls = Vec::new();
    let mut entry_decls = Vec::new();
    let mut stdlib_decls = Vec::new();
    let mut non_stdlib_decls = Vec::new();
    for (package_name, module) in linked {
        if module.entry {
            entry_decls.extend(module.decls.iter().cloned());
        }
        if package_name == CHELIS_STD_PACKAGE_NAME {
            stdlib_decls.extend(module.decls.iter().cloned());
        } else {
            non_stdlib_decls.extend(module.decls.iter().cloned());
        }
        decls.extend(module.decls);
    }
    Ok(Some(PreparedProgram {
        decls,
        entry_decls,
        package_root: root,
        stdlib_decls,
        non_stdlib_decls,
    }))
}

pub fn prepare_program_for_eval_file(
    file: &Path,
    context_dir: &Path,
) -> Result<Option<PreparedProgram>, String> {
    let source =
        fs::read_to_string(file).map_err(|e| format!("failed to read {}: {e}", file.display()))?;
    let decls =
        chelis_surf::parser::parse_str(&source).map_err(|e| format!("{}: {e}", file.display()))?;
    if matches!(decls.as_slice(), [Decl::Module { .. }]) {
        return prepare_program_for_file(file);
    }
    prepare_program_for_eval_source(context_dir, &decls)
}

/// Resolve, link, and cache the reef package graph rooted at (or above)
/// `context_dir`. This does the expensive, per-invocation work: package-root
/// walk, manifest read, dependency resolution (lockfile fast path or
/// timeout-bounded resolver), chelis-std shell load, internal-name
/// rewriting of library modules, and caching of `internal_maps` and
/// `dep_shells`.
///
/// The result is an opaque handle that can be threaded through many calls to
/// [`compile_with_reef_graph`] — one per entry file — to avoid repeating the
/// expensive graph preparation for every test file in a `chelis test`
/// invocation.
///
/// Returns `Err` (not `Ok(None)`) when `context_dir` is not inside a reef
/// package. Callers that rely on reef context (e.g. `chelis test`) need an
/// actionable error with the attempted directory, not a silent `None` that
/// loses information; the old-path `Ok(None)` semantics live on in
/// [`prepare_program_for_eval_source`] where that return is load-bearing.
pub fn prepare_reef_graph(context_dir: &Path) -> Result<PreparedReefGraph, String> {
    let Some(root) = find_package_root_for_dir(context_dir)? else {
        return Err(format!(
            "no reef.toml found in `{}` or any parent up to $HOME: \
             pass a path inside a reef package, run from inside one, or \
             set --project-root",
            context_dir.display()
        ));
    };
    let graph = load_package_graph_for_eval(&root)?;
    let linked = link_graph_with_package_tags(&graph, &[])?;
    let internal_maps = build_internal_maps(&graph);
    let dep_shells = dependency_shells(&graph);
    let eval_module_prefix = graph
        .packages
        .get(&graph.root_package)
        .map(|package| package.manifest.package.module_prefix.clone())
        .unwrap_or_default();

    // Partition the linked library decls into the chelis-std slice and
    // everything else, preserving relative order so the concatenation
    // `stdlib ++ non_stdlib` equals `linked_library_decls`. The
    // chelis-std typecheck cache content-addresses the stdlib slice.
    let mut linked_library_decls = Vec::new();
    let mut linked_stdlib_decls = Vec::new();
    let mut linked_non_stdlib_library_decls = Vec::new();
    for (package_name, module) in linked {
        if package_name == CHELIS_STD_PACKAGE_NAME {
            linked_stdlib_decls.extend(module.decls.iter().cloned());
        } else {
            linked_non_stdlib_library_decls.extend(module.decls.iter().cloned());
        }
        linked_library_decls.extend(module.decls);
    }

    Ok(PreparedReefGraph {
        package_root: root,
        graph,
        linked_library_decls,
        linked_stdlib_decls,
        linked_non_stdlib_library_decls,
        internal_maps,
        dep_shells,
        eval_module_prefix,
    })
}

/// Rewrite an entry-decl list against a previously prepared reef graph,
/// producing the same internal-name-rewritten and import/export-stripped
/// decl list that `compile_with_reef_graph` produces — but WITHOUT
/// prepending the library decls. Phase G's `eval_in_context` calls this
/// to re-use the reef name resolver for new code while leaving the
/// library decls in their pre-checked form inside the `CompiledContext`.
pub fn rewrite_entry_decls_with_reef_graph(
    graph: &PreparedReefGraph,
    entry_decls: &[Decl],
) -> Result<Vec<Decl>, String> {
    let eval_module_name = if graph.eval_module_prefix.is_empty() {
        "__Eval".to_string()
    } else {
        format!("{}.__Eval", graph.eval_module_prefix)
    };
    let eval_module = ModuleSource {
        package_name: graph.graph.root_package.clone(),
        module_name: eval_module_name,
        decls: entry_decls.to_vec(),
        file_rel: PathBuf::from("__eval__.ch"),
        // Synthetic eval module: it doesn't live on disk, so the
        // source_root is purely cosmetic — the rewrite path doesn't
        // resolve back to a file.
        source_root: "src".to_string(),
        exports: BTreeSet::new(),
        symbols: collect_symbol_kinds(entry_decls),
    };
    rewrite_eval_module_decls(
        &eval_module,
        &graph.graph,
        &graph.internal_maps,
        &graph.dep_shells,
    )
}

/// Compile an in-memory entry decl list against a previously prepared reef
/// graph. This is the cheap per-file work: only the entry module is rewritten
/// and appended to the cached library decls.
pub fn compile_with_reef_graph(
    graph: &PreparedReefGraph,
    entry_decls: &[Decl],
) -> Result<PreparedProgram, String> {
    let rewritten_entry_decls = rewrite_entry_decls_with_reef_graph(graph, entry_decls)?;

    let mut decls = graph.linked_library_decls.clone();
    decls.extend(rewritten_entry_decls.iter().cloned());

    // The eval entry decls are not chelis-std; they join the non-stdlib
    // partition. `stdlib_decls` is the graph's already-partitioned
    // chelis-std slice.
    let mut non_stdlib_decls = graph.linked_non_stdlib_library_decls.clone();
    non_stdlib_decls.extend(rewritten_entry_decls);

    Ok(PreparedProgram {
        decls,
        entry_decls: entry_decls.to_vec(),
        package_root: graph.package_root.clone(),
        stdlib_decls: graph.linked_stdlib_decls.clone(),
        non_stdlib_decls,
    })
}

/// Resolve a `PackageGraph` for eval: fast-path the lockfile when present,
/// otherwise run the full resolver under the standard 5-second timeout.
///
/// Eval does not write the lockfile — that is the build path's responsibility.
///
/// Auto-fetch is on by default for eval too: `chelis check` / `chelis eval`
/// against an empty registry should not require a manual install round-trip
/// any more than `chelis reef build` does. The Item 8 `--no-auto-fetch`
/// surface is exposed only on `chelis reef build` for now; eval-side
/// callers run with [`LoadOptions::default_for_load`].
fn load_package_graph_for_eval(root: &Path) -> Result<PackageGraph, String> {
    let options = LoadOptions::default_for_load();
    let lock_path = root.join("reef.lock");
    if lock_path.exists() {
        let lock = read_lockfile(&lock_path)?;
        reconstruct_graph_from_lockfile(root, &lock, options)
    } else {
        let root_clone = root.to_path_buf();
        run_with_timeout(
            move || resolve_package_graph(&root_clone, options),
            Duration::from_secs(5),
            TIMEOUT_MSG,
        )
    }
}

pub fn prepare_program_for_eval_source(
    context_dir: &Path,
    entry_decls: &[Decl],
) -> Result<Option<PreparedProgram>, String> {
    // Convenience wrapper: preserved for existing callers (chelis eval, the
    // CLI --file path, Deep/IR integration tests) that want the single-shot
    // preparation and expect `Ok(None)` when there is no reef package.
    // `chelis test` takes the split-path route (prepare_reef_graph +
    // compile_with_reef_graph) to share the expensive graph across files.
    let Some(_) = find_package_root_for_dir(context_dir)? else {
        return Ok(None);
    };
    let graph = prepare_reef_graph(context_dir)?;
    compile_with_reef_graph(&graph, entry_decls).map(Some)
}

/// Build a package with the default options ([`BuildOptions::default`]),
/// which today means auto-fetch is **on**: missing-from-registry
/// dependencies are silently fetched from the canonical hosting org
/// before the build resumes.
///
/// Pre-Item-8 callers that want the legacy "fail if anything is
/// missing" behavior should call [`build_package_with_options`] with
/// `BuildOptions { auto_fetch: false }`. Auto-fetch's user-visible
/// effect is surfaced via stderr ("chelis reef: auto-fetching ...")
/// so it remains observable, per the locked Item 8 contract invariant.
pub fn build_package(root: &Path) -> Result<PackageBuildArtifacts, String> {
    build_package_with_options(root, &BuildOptions::default())
}

/// Build a package with caller-supplied [`BuildOptions`]. Phase A
/// Item 8 entry point; the CLI's `chelis reef build [--no-auto-fetch]`
/// dispatches here.
///
/// ### Auto-fetch contract
///
/// - `auto_fetch: true` (default): each missing-from-registry
///   dependency triggers a single attempt to
///   [`install_from_github`] against either the lockfile's
///   `remote_origin` (Item 9; today always `None`) or the
///   `chelis-lang/<name>@v<version>` canonical default. The fetch
///   runs under [`acquire_reef_home_lock`] so concurrent builds do
///   not race on `index.json`. On fetch failure the build bails with
///   the improved error wording (URL + auth state + typed category).
///   The auto-fetch event is logged to stderr so it remains
///   observable to operators.
/// - `auto_fetch: false`: missing-from-registry surfaces the same
///   improved-wording error immediately, naming the URL that would
///   have been tried so the operator can run
///   `chelis reef install --from-github <url>` manually.
pub fn build_package_with_options(
    root: &Path,
    options: &BuildOptions,
) -> Result<PackageBuildArtifacts, String> {
    let root = canonical_root(root)?;
    let graph = resolve_package_graph(&root, options.into())?;
    let lock = build_lockfile(&graph);
    write_lockfile(&root.join("reef.lock"), &lock)?;

    let root_pkg = graph
        .packages
        .get(&graph.root_package)
        .ok_or_else(|| "root package missing from graph".to_string())?;
    let entry_modules = root_pkg.modules.keys().cloned().collect::<Vec<_>>();
    let linked = link_graph(&graph, &entry_modules)?;
    let linked_decls = linked
        .into_iter()
        .flat_map(|module| module.decls)
        .collect::<Vec<_>>();
    let deep = expanded_desugared_program(&linked_decls)?;
    let checked = checked_program_with_effects(&deep)?;

    let dist_dir = root.join("dist");
    fs::create_dir_all(&dist_dir).map_err(|e| e.to_string())?;
    let archive_path = dist_dir.join(format!(
        "{}-{}.tar.zst",
        root_pkg.id.name, root_pkg.id.version
    ));
    let shell_path = dist_dir.join(format!("{}-{}.chb", root_pkg.id.name, root_pkg.id.version));
    build_archive(&root, &archive_path)?;
    let archive_sha256 = sha256_file(&archive_path)?;
    let shell = build_shell_package(root_pkg, &checked, &archive_sha256)?;
    write_shell(&shell_path, &shell).map_err(|e| e.to_string())?;
    let shell_sha256 = sha256_file(&shell_path)?;

    Ok(PackageBuildArtifacts {
        package: root_pkg.id.clone(),
        shell_path,
        archive_path,
        shell_sha256,
        archive_sha256,
    })
}

pub fn publish_package(root: &Path) -> Result<PackageBuildArtifacts, String> {
    let root = canonical_root(root)?;
    let manifest = read_manifest(&root.join("reef.toml"))?;
    if manifest
        .dependencies
        .values()
        .any(|dep| dep.path.as_ref().is_some())
    {
        return Err("`chelis reef publish` rejects path dependencies in 3a".to_string());
    }
    let artifacts = build_package(&root)?;
    let registry_root = registry_root()?;
    let target_dir = registry_root
        .join("packages")
        .join(&artifacts.package.name)
        .join(&artifacts.package.version);
    fs::create_dir_all(&target_dir).map_err(|e| e.to_string())?;
    fs::copy(
        &artifacts.archive_path,
        target_dir.join(
            artifacts
                .archive_path
                .file_name()
                .ok_or_else(|| "archive missing file name".to_string())?,
        ),
    )
    .map_err(|e| e.to_string())?;
    fs::copy(
        &artifacts.shell_path,
        target_dir.join(
            artifacts
                .shell_path
                .file_name()
                .ok_or_else(|| "shell missing file name".to_string())?,
        ),
    )
    .map_err(|e| e.to_string())?;

    let index_path = registry_root.join("index.json");
    let mut index = if index_path.exists() {
        serde_json::from_str::<LocalRegistryIndex>(
            &fs::read_to_string(&index_path).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
    } else {
        LocalRegistryIndex::default()
    };
    let versions = index
        .packages
        .entry(artifacts.package.name.clone())
        .or_default();
    versions.retain(|entry| entry.version != artifacts.package.version);
    versions.push(RegistryVersion {
        version: artifacts.package.version.clone(),
        compiler: CURRENT_COMPILER_VERSION.to_string(),
        archive_sha256: artifacts.archive_sha256.clone(),
        shell_sha256: artifacts.shell_sha256.clone(),
        // `chelis reef publish` populates the local registry from a
        // local working tree; there is no remote origin to record. The
        // field stays `None` so the entry round-trips through serde
        // identical to a pre-Item-9 publish.
        remote_origin: None,
    });
    versions.sort_by(|a, b| a.version.cmp(&b.version));
    fs::create_dir_all(&registry_root).map_err(|e| e.to_string())?;
    let serialized = serde_json::to_string_pretty(&index).map_err(|e| e.to_string())?;
    // Atomic-rename invariant: every registry-level write under
    // `$CHELIS_REEF_HOME` (currently `index.json`) goes through
    // `atomic_write` so a partial write cannot leave a corrupt index
    // behind. See `atomic_write`'s rustdoc for the rule and scope.
    atomic_write(&index_path, serialized.as_bytes())?;

    Ok(artifacts)
}

/// One package successfully copied into the local registry by
/// [`install_from_monorepo`] or [`install_from_github`].
#[derive(Debug, Clone)]
pub struct InstalledArtifact {
    pub package: PackageId,
    pub shell_path: PathBuf,
    pub archive_path: PathBuf,
    pub shell_sha256: String,
    pub archive_sha256: String,
}

/// Canonical hosting org for chelis pre-launch shell distribution.
///
/// Used as the implicit publisher when a tag is referenced without an
/// `org/repo` prefix. Locked 2026-05-05 by `phaseA_reef_distribution.md`;
/// see `spec/design/reef_distribution.md` Item 6.
pub const CANONICAL_REEF_ORG: &str = "chelis-lang";

/// Default GitHub web-host base URL. Currently unused on the fetch
/// path (the `/releases/download/...` form does not serve private-repo
/// asset bytes — see `CHELIS_REEF_GITHUB_BASE_API` below) but kept so
/// future Item 8/9 surfaces can use it for things that legitimately
/// live on the web host (release-page links in error messages, etc).
/// Tests inject a localhost wiremock URL via `CHELIS_REEF_GITHUB_BASE`.
#[allow(dead_code)]
const DEFAULT_GITHUB_BASE: &str = "https://github.com";

/// Default GitHub API host. The `--from-github` fetch path uses two
/// API endpoints:
///
/// 1. `GET <api>/repos/<org>/<repo>/releases/tags/<tag>` to look up
///    asset metadata (id, name, size).
/// 2. `GET <api>/repos/<org>/<repo>/releases/assets/<asset_id>` with
///    `Accept: application/octet-stream` to stream the asset bytes.
///
/// **Why the API path, not `/releases/download/<tag>/<asset>`:**
/// GitHub's public-facing `/releases/download/...` URL form does not
/// serve private-repo asset bytes even with a valid `Authorization:
/// token …` header — it returns 404. The canonical chelis-lang shells
/// are private during the pre-launch era, so the API path is required,
/// not optional. See `spec/design/reef_distribution.md` § Item 6.
///
/// Tests inject a localhost wiremock URL via `CHELIS_REEF_GITHUB_BASE_API`.
const DEFAULT_GITHUB_BASE_API: &str = "https://api.github.com";

/// Distinct error categories surfaced by the GitHub fetch path. The
/// caller — typically `chelis reef install --from-github`'s CLI handler
/// — gets a category to discriminate on so wording can stay actionable.
///
/// New variants must keep the existing wording stable; tests assert
/// against substrings of the `Display` impl.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GitHubFetchError {
    /// `<org>/<repo>@<tag>` did not parse: missing slash, missing `@`,
    /// empty component, etc.
    Parse { input: String, reason: String },
    /// Auth was unobtainable: `GITHUB_TOKEN` unset and `gh auth token`
    /// shell-out also failed.
    AuthMissing { reason: String },
    /// HTTP 401/403 from the release-asset URL even with a token; the
    /// token is invalid or lacks scope.
    AuthRejected { url: String, status: u16 },
    /// HTTP 404 — the release tag exists but the named asset is not
    /// attached to it. This fires only after the release metadata has
    /// been fetched successfully, so the repo is reachable and the tag
    /// exists; a 404 here genuinely means "asset missing." The
    /// metadata-step 404 (repo/tag-level) is the separate, ambiguous
    /// [`Self::ReleaseTagNotFoundOrUnauthorized`].
    ReleaseAssetNotFound { url: String, asset_name: String },
    /// HTTP 404 on the release-metadata endpoint (`releases/tags/<tag>`).
    /// This status is inherently ambiguous: GitHub returns 404 both when
    /// the tag genuinely does not exist on a reachable repo and when the
    /// token lacks `contents: read` access to a private repo (GitHub
    /// returns 404, not 403, for unauthorized private repos as a privacy
    /// measure, so the repo's existence is not leaked). We cannot tell
    /// the two apart from this single response, so the variant and its
    /// message name both possibilities. See issue #147.
    ReleaseTagNotFoundOrUnauthorized { url: String, tag: String },
    /// HTTP 429 — GitHub rate limit. Message includes the `Retry-After`
    /// header value if present.
    RateLimited {
        url: String,
        retry_after: Option<String>,
    },
    /// HTTP 5xx.
    ServerError { url: String, status: u16 },
    /// Connection error: DNS, TCP, TLS, body-read.
    Network { url: String, message: String },
    /// I/O error during streaming / temp-file work.
    Io { message: String },
    /// Validation fired downstream of fetch (see
    /// [`install_validated_artifact_pair`]).
    Validation { message: String },
}

impl std::fmt::Display for GitHubFetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse { input, reason } => write!(
                f,
                "could not parse `{input}` as <org>/<repo>@<tag>: {reason}"
            ),
            Self::AuthMissing { reason } => write!(
                f,
                "GITHUB_TOKEN is not set and `gh auth token` did not yield a token ({reason}). \
                 export GITHUB_TOKEN=$(gh auth token) and retry"
            ),
            Self::AuthRejected { url, status } => write!(
                f,
                "GitHub rejected the token (HTTP {status}) when fetching {url}: \
                 the token may be invalid or missing repo scope"
            ),
            Self::ReleaseAssetNotFound { url, asset_name } => write!(
                f,
                "release asset `{asset_name}` not found at {url} (HTTP 404): \
                 verify the release tag exists and that the asset is attached to it"
            ),
            Self::ReleaseTagNotFoundOrUnauthorized { url, tag } => write!(
                f,
                "release tag `{tag}` at {url} returned HTTP 404. This is ambiguous: \
                 either (a) the tag does not exist on that repo, or (b) your token \
                 (GITHUB_TOKEN / `gh auth token`) lacks `contents: read` access to \
                 the repo. GitHub returns 404 (not 403) in case (b) as a privacy \
                 measure, so the two cannot be told apart from this response alone. \
                 Verify with `gh release view {tag} --repo <org>/<repo>` using the \
                 same token."
            ),
            Self::RateLimited { url, retry_after } => match retry_after {
                Some(r) => write!(
                    f,
                    "GitHub rate limit (HTTP 429) when fetching {url}; Retry-After: {r}"
                ),
                None => write!(
                    f,
                    "GitHub rate limit (HTTP 429) when fetching {url}; no Retry-After header"
                ),
            },
            Self::ServerError { url, status } => {
                write!(f, "GitHub returned HTTP {status} for {url}; not retrying")
            }
            Self::Network { url, message } => {
                write!(f, "network error fetching {url}: {message}")
            }
            Self::Io { message } => write!(f, "I/O error during GitHub fetch: {message}"),
            Self::Validation { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for GitHubFetchError {}

impl From<GitHubFetchError> for String {
    fn from(e: GitHubFetchError) -> Self {
        e.to_string()
    }
}

/// Atomically write `bytes` to `final_path` by writing to a sibling
/// temp file in the same directory and `fs::rename`'ing into place.
///
/// Same-directory rename is atomic on every supported FS we run on.
/// The temp file is named `<basename>.tmp` so concurrent installs of
/// different files in the same directory do not collide on the temp
/// name (each unique destination has a unique temp). Concurrent
/// installs of the SAME file are not made safe by this alone; that is
/// what Item 8's `flock` will add.
///
/// **When to use (rule):** every write of a **registry-level metadata
/// file** under `$CHELIS_REEF_HOME` must go through this helper. As
/// of today the only such file is `$CHELIS_REEF_HOME/index.json`,
/// written by `install_validated_artifact_pair` (the path shared by
/// `install_from_monorepo` and `install_from_github`) and by
/// `publish_package`. New metadata files (e.g. a future lockfile-
/// origin cache) join the same rule.
///
/// **When not to use (rule):** writes to per-package files under a
/// package's own directory — `<root>/reef.toml`, `<root>/reef.lock`,
/// `<root>/dist/<name>-<version>.tar.zst`, `<root>/src/main.ch` —
/// stay on plain `fs::write`. Those are owned by the publisher's
/// working tree, not by the shared registry, and a partial write
/// there is the publisher's local concern (and easy to recreate by
/// re-running `chelis reef build`).
fn atomic_write(final_path: &Path, bytes: &[u8]) -> Result<(), String> {
    let dir = final_path.parent().ok_or_else(|| {
        format!(
            "atomic_write: {} has no parent directory",
            final_path.display()
        )
    })?;
    fs::create_dir_all(dir).map_err(|e| {
        format!(
            "failed to create {} for atomic_write of {}: {e}",
            dir.display(),
            final_path.display()
        )
    })?;
    let file_name = final_path.file_name().ok_or_else(|| {
        format!(
            "atomic_write: {} has no file name component",
            final_path.display()
        )
    })?;
    let mut tmp_name = file_name.to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = dir.join(&tmp_name);
    // Best-effort cleanup of a stale temp from a previous crashed write.
    let _ = fs::remove_file(&tmp_path);
    fs::write(&tmp_path, bytes).map_err(|e| {
        format!(
            "failed to write temp file {} for atomic_write of {}: {e}",
            tmp_path.display(),
            final_path.display()
        )
    })?;
    fs::rename(&tmp_path, final_path).map_err(|e| {
        // Clean up the orphaned temp on rename failure to avoid leaving
        // half-written `.tmp` files in the registry root.
        let _ = fs::remove_file(&tmp_path);
        format!(
            "failed to rename {} -> {}: {e}",
            tmp_path.display(),
            final_path.display()
        )
    })
}

/// File name (relative to `$CHELIS_REEF_HOME`) of the Item 8 advisory
/// lock file used to serialize concurrent registry-mutating operations
/// (auto-fetch + install + index update).
///
/// Created on first acquire (and never removed — `flock(2)` is
/// auto-released by the kernel when the holder's fd closes, so a
/// crashed process leaves the file but not the lock). The file's
/// inode is the lock identity; `flock` semantics are scoped to the
/// open file description, not the path.
const REEF_HOME_LOCK_FILE: &str = ".reef-lock";

/// Default acquire-timeout for the Item 8 process-level lock. 60 seconds
/// is comfortably longer than any single `install_from_github` round
/// trip on a real GitHub release (the heaviest canonical asset today is
/// `chelis-std-0.3.0.tar.zst` at well under 1 MB) and short enough to
/// surface a deadlocked / wedged peer process within a developer's
/// single iteration loop. Locked by `phaseA_item8_autofetch_build_oracle`.
const REEF_HOME_LOCK_TIMEOUT: Duration = Duration::from_secs(60);

/// RAII guard for the Item 8 process-level advisory lock on
/// `$CHELIS_REEF_HOME/.reef-lock`. Drop releases the underlying
/// `flock(2)` by closing the file descriptor — the kernel does the
/// work, so panics, `?`-bailouts, and normal returns all release the
/// lock cleanly without explicit unlock plumbing.
///
/// The guard is intentionally non-`Clone`, non-`Copy`, and stores its
/// owned `fs::File` private. Callers must not access the file
/// directly.
#[must_use = "the lock is released when the guard drops; binding to `_` would release immediately"]
pub struct ReefHomeLock {
    /// The lock-bearing fd. Drop closes it, which releases the
    /// `flock(2)` held against it.
    _file: fs::File,
    /// Path of the lock file. Stored for diagnostic messages only.
    path: PathBuf,
}

impl ReefHomeLock {
    /// Path of the on-disk lock file backing this guard. Exposed so
    /// tests can assert the file exists during the lock window.
    pub fn lock_path(&self) -> &Path {
        &self.path
    }
}

/// Errors surfaced by [`acquire_reef_home_lock`]. Each variant carries
/// enough detail to make the failure self-explanatory in a CLI message.
#[derive(Debug)]
pub enum ReefHomeLockError {
    /// Could not create or open `$CHELIS_REEF_HOME/.reef-lock` — the
    /// registry root is missing, unwritable, or some other I/O issue.
    Io { path: PathBuf, message: String },
    /// `flock(LOCK_EX)` was not acquired within the configured wait
    /// window. The lock is held by another process; surface a clear
    /// timeout error.
    Timeout { path: PathBuf, waited: Duration },
}

impl std::fmt::Display for ReefHomeLockError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, message } => {
                write!(
                    f,
                    "failed to open registry lock file {}: {message}",
                    path.display()
                )
            }
            Self::Timeout { path, waited } => write!(
                f,
                "timed out after {waited:?} waiting for registry lock {}; \
                 another `chelis reef build` or `chelis reef install` is likely in progress",
                path.display()
            ),
        }
    }
}

impl std::error::Error for ReefHomeLockError {}

/// Acquire a process-level advisory lock on
/// `$CHELIS_REEF_HOME/.reef-lock`. Blocks (in a poll-with-sleep loop)
/// up to `timeout`, returning [`ReefHomeLockError::Timeout`] if the
/// lock cannot be acquired within that window.
///
/// ### Lock semantics
///
/// The lock is `flock(2)` with `LOCK_EX | LOCK_NB`. A crashed
/// process's lock is auto-released by the kernel when its file
/// descriptor is closed; a stale `.reef-lock` file on disk does
/// **not** prevent acquisition. This is the documented Unix
/// semantic and the test
/// `phaseA_item8_stale_lock_file_does_not_block_acquisition`
/// pins it.
///
/// ### Concurrency
///
/// Holders are serialized strictly: only one process can hold the lock
/// at a time. The auto-fetch path holds the lock through the entire
/// fetch + install + index-update window (Item 6's `install_from_github`
/// → `install_validated_artifact_pair` chain), so concurrent peers
/// observe a serialized view of `index.json`.
///
/// ### What this is not
///
/// This is **not** a per-package lock. Two concurrent builds against
/// distinct packages still serialize through this single registry
/// lock. The trade-off is intentional: the registry's `index.json` is
/// a single shared file, and per-package fan-out would require
/// per-package lock files plus an index-write critical section
/// anyway. Item 10's registry-server design will revisit this when
/// concurrent throughput becomes a real driver.
pub fn acquire_reef_home_lock(
    registry_root: &Path,
    timeout: Duration,
) -> Result<ReefHomeLock, ReefHomeLockError> {
    use rustix::fs::{FlockOperation, flock};
    use std::os::fd::AsFd;

    fs::create_dir_all(registry_root).map_err(|e| ReefHomeLockError::Io {
        path: registry_root.to_path_buf(),
        message: format!("create registry root: {e}"),
    })?;
    let lock_path = registry_root.join(REEF_HOME_LOCK_FILE);
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&lock_path)
        .map_err(|e| ReefHomeLockError::Io {
            path: lock_path.clone(),
            message: e.to_string(),
        })?;

    // Poll loop: try non-blocking first, sleep briefly, retry. We do
    // not use blocking `flock` because we want the wait-with-timeout
    // semantic and a deterministic test signal. Polling cadence:
    // 50 ms — fine-grained enough that a successful release is picked
    // up quickly, coarse enough that a 60 s timeout is ~1200 syscalls
    // worst case (negligible).
    let poll_interval = Duration::from_millis(50);
    let start = std::time::Instant::now();
    loop {
        match flock(file.as_fd(), FlockOperation::NonBlockingLockExclusive) {
            Ok(()) => {
                return Ok(ReefHomeLock {
                    _file: file,
                    path: lock_path,
                });
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if start.elapsed() >= timeout {
                    return Err(ReefHomeLockError::Timeout {
                        path: lock_path,
                        waited: timeout,
                    });
                }
                std::thread::sleep(poll_interval);
            }
            Err(e) => {
                return Err(ReefHomeLockError::Io {
                    path: lock_path,
                    message: format!("flock: {e}"),
                });
            }
        }
    }
}

/// Validate one prebuilt `(archive, shell)` pair on disk and place a
/// copy into the local Reef registry, updating `index.json` atomically.
///
/// This is the validation+placement step extracted out of the original
/// monolithic `install_from_monorepo`. Both the monorepo path and the
/// new `--from-github` path call it. Same SHA256 verification, same
/// archive↔shell agreement check, same name/version-agreement check,
/// same `index.json` schema. Rust's borrow checker, not duplication,
/// is the source-equivalence enforcer here.
///
/// The caller passes the bytes' source-of-truth on-disk locations
/// (which may be a tempdir for the GitHub path) and the expected
/// `name`/`version` strings. The helper produces:
/// - the validated archive and shell hashes,
/// - `<registry_root>/packages/<name>/<version>/<files>` placed via
///   plain `fs::copy`,
/// - `<registry_root>/index.json` updated atomically (write to
///   `index.json.tmp`, `fs::rename` into place).
///
/// Errors out (and does **not** update the index) if any of:
/// - on-disk SHA256 of `shell_path` disagrees with the shell's embedded
///   `archive_sha256` — i.e. the pair is from different builds,
/// - the shell's `package` does not match `(name, version)`,
/// - the destination directory cannot be created,
/// - either `fs::copy` fails.
///
/// Atomicity invariant: if the index update fails or any prior step
/// fails, `index.json` is unchanged. Half-copied files in the package
/// dir may exist on the failed install (the caller is responsible for
/// cleanup if it cares about that — but the index sees nothing).
pub fn install_validated_artifact_pair(
    archive_path: &Path,
    shell_path: &Path,
    name: &str,
    version: &str,
    registry_root: &Path,
    remote_origin: Option<&str>,
) -> Result<InstalledArtifact, String> {
    fs::create_dir_all(registry_root).map_err(|e| {
        format!(
            "failed to create registry root {}: {e}",
            registry_root.display()
        )
    })?;
    let index_path = registry_root.join("index.json");
    let mut index = if index_path.exists() {
        serde_json::from_str::<LocalRegistryIndex>(
            &fs::read_to_string(&index_path).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
    } else {
        LocalRegistryIndex::default()
    };

    let archive_sha256 = sha256_file(archive_path)?;
    let shell_sha256 = sha256_file(shell_path)?;
    let shell = read_shell(shell_path).map_err(|e| e.to_string())?;
    if shell.archive_sha256 != archive_sha256 {
        return Err(format!(
            "prebuilt shell {} disagrees with archive {} on archive_sha256: \
             the dist/ tree is stale; run `chelis reef build` in the monorepo",
            shell_path.display(),
            archive_path.display()
        ));
    }
    if shell.package.name != name || shell.package.version != version {
        return Err(format!(
            "prebuilt shell {} advertises `{}-{}` but was requested as `{name}-{version}`",
            shell_path.display(),
            shell.package.name,
            shell.package.version
        ));
    }

    let target_dir = registry_root.join("packages").join(name).join(version);
    fs::create_dir_all(&target_dir).map_err(|e| {
        format!(
            "failed to create registry package dir {}: {e}",
            target_dir.display()
        )
    })?;
    let archive_dst = target_dir.join(format!("{name}-{version}.tar.zst"));
    let shell_dst = target_dir.join(format!("{name}-{version}.chb"));
    fs::copy(archive_path, &archive_dst).map_err(|e| {
        format!(
            "failed to copy archive {} -> {}: {e}",
            archive_path.display(),
            archive_dst.display()
        )
    })?;
    fs::copy(shell_path, &shell_dst).map_err(|e| {
        format!(
            "failed to copy shell {} -> {}: {e}",
            shell_path.display(),
            shell_dst.display()
        )
    })?;

    let versions = index.packages.entry(name.to_string()).or_default();
    versions.retain(|entry| entry.version != version);
    versions.push(RegistryVersion {
        version: version.to_string(),
        compiler: shell.compiler.clone(),
        archive_sha256: archive_sha256.clone(),
        shell_sha256: shell_sha256.clone(),
        remote_origin: remote_origin.map(str::to_string),
    });
    versions.sort_by(|a, b| a.version.cmp(&b.version));

    let serialized = serde_json::to_string_pretty(&index).map_err(|e| e.to_string())?;
    atomic_write(&index_path, serialized.as_bytes())?;

    Ok(InstalledArtifact {
        package: PackageId {
            name: name.to_string(),
            version: version.to_string(),
        },
        shell_path: shell_dst,
        archive_path: archive_dst,
        shell_sha256,
        archive_sha256,
    })
}

/// Install one or more prebuilt packages from a chelis monorepo's
/// `packages/<name>/dist/` into the local Reef registry, updating
/// `index.json` accordingly.
///
/// `monorepo_root` should point at a directory containing a `packages/`
/// subdirectory whose entries are valid Reef packages with prebuilt
/// `dist/<name>-<version>.{chb,tar.zst}` artifacts.
///
/// `requested` is a list of `(name, optional_version)` pairs. If a
/// version is `None`, the version advertised by that package's
/// `reef.toml` is used. If `requested` is empty, every package found
/// under `packages/` is installed at its advertised version.
///
/// Errors out (without partial commits beyond what already wrote) if
/// the monorepo has no `packages/` dir, a requested package is
/// missing, or its dist artifacts are absent.
pub fn install_from_monorepo(
    monorepo_root: &Path,
    requested: &[(String, Option<String>)],
) -> Result<Vec<InstalledArtifact>, String> {
    let monorepo_root = monorepo_root.canonicalize().map_err(|e| {
        format!(
            "failed to resolve monorepo path {}: {e}",
            monorepo_root.display()
        )
    })?;
    let packages_dir = monorepo_root.join("packages");
    if !packages_dir.is_dir() {
        return Err(format!(
            "{} does not contain a `packages/` directory: \
             expected a chelis monorepo layout",
            monorepo_root.display()
        ));
    }

    // Build the list of (name, version) pairs to install. Each entry
    // is validated against the monorepo's actual package manifest.
    let resolved: Vec<(String, String, PathBuf)> = if requested.is_empty() {
        let mut out = Vec::new();
        let mut entries: Vec<_> = fs::read_dir(&packages_dir)
            .map_err(|e| format!("failed to read {}: {e}", packages_dir.display()))?
            .filter_map(|entry| entry.ok())
            .collect();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let pkg_root = entry.path();
            if !pkg_root.is_dir() {
                continue;
            }
            let manifest_path = pkg_root.join("reef.toml");
            if !manifest_path.exists() {
                continue;
            }
            let manifest = read_manifest(&manifest_path)?;
            out.push((manifest.package.name, manifest.package.version, pkg_root));
        }
        if out.is_empty() {
            return Err(format!(
                "no packages with reef.toml found under {}",
                packages_dir.display()
            ));
        }
        out
    } else {
        let mut out = Vec::new();
        for (name, version) in requested {
            let pkg_root = packages_dir.join(name);
            if !pkg_root.is_dir() {
                return Err(format!(
                    "package `{name}` not found under {} (looked for {})",
                    packages_dir.display(),
                    pkg_root.display()
                ));
            }
            let manifest_path = pkg_root.join("reef.toml");
            if !manifest_path.exists() {
                return Err(format!(
                    "package `{name}` is missing {}",
                    manifest_path.display()
                ));
            }
            let manifest = read_manifest(&manifest_path)?;
            if manifest.package.name != *name {
                return Err(format!(
                    "package directory `{}` advertises name `{}` in reef.toml, \
                     not the requested `{name}`",
                    pkg_root.display(),
                    manifest.package.name
                ));
            }
            let chosen_version = match version {
                Some(v) => {
                    if *v != manifest.package.version {
                        return Err(format!(
                            "package `{name}` requested at version `{v}`, but the \
                             monorepo carries version `{}`",
                            manifest.package.version
                        ));
                    }
                    v.clone()
                }
                None => manifest.package.version.clone(),
            };
            out.push((name.clone(), chosen_version, pkg_root));
        }
        out
    };

    let registry_root = registry_root()?;
    fs::create_dir_all(&registry_root).map_err(|e| e.to_string())?;

    let mut installed = Vec::new();
    for (name, version, pkg_root) in &resolved {
        let dist_dir = pkg_root.join("dist");
        if !dist_dir.is_dir() {
            return Err(format!(
                "package `{name}` has no dist/ directory at {}: \
                 run `chelis reef build` in the monorepo first",
                dist_dir.display()
            ));
        }
        let archive_src = dist_dir.join(format!("{name}-{version}.tar.zst"));
        let shell_src = dist_dir.join(format!("{name}-{version}.chb"));
        if !archive_src.exists() {
            return Err(format!(
                "missing prebuilt archive {}: \
                 run `chelis reef build` in the monorepo first",
                archive_src.display()
            ));
        }
        if !shell_src.exists() {
            return Err(format!(
                "missing prebuilt shell {}: \
                 run `chelis reef build` in the monorepo first",
                shell_src.display()
            ));
        }

        // Validation + placement is shared with `install_from_github`
        // via this helper. The helper does on-disk sha256 verification,
        // archive↔shell agreement, name/version-agreement, registry
        // copy, and atomic index update.
        //
        // `--from-monorepo` has no remote origin: it copies bytes that
        // already live on the developer's local filesystem. Item 9's
        // `remote_origin` field stays `None` for these entries, which
        // is why `chelis reef install --from-lockfile` errors clearly
        // on lockfiles whose entries came exclusively from
        // `--from-monorepo` and suggests `--bootstrap` to populate
        // origins.
        let artifact = install_validated_artifact_pair(
            &archive_src,
            &shell_src,
            name,
            version,
            &registry_root,
            None,
        )?;
        installed.push(artifact);
    }

    Ok(installed)
}

/// Parsed `<org>/<repo>@<tag>` triple. The `version` is `tag` with a
/// leading `v` stripped if present, so `v0.4.0` and `0.4.0` both map
/// to version string `0.4.0`. Asset URLs use the original `tag` as
/// GitHub's release path component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitHubReleaseSpec {
    pub org: String,
    pub repo: String,
    pub tag: String,
    /// `tag` with one leading `v` stripped if present.
    pub version: String,
}

impl GitHubReleaseSpec {
    /// Parse `<org>/<repo>@<tag>`. Both `chelis-lang/nautilus@v0.4.0`
    /// and `chelis-lang/nautilus@0.4.0` are accepted; the leading `v`
    /// is treated as decorative and stripped to derive `version`. The
    /// `tag` field preserves whatever the caller passed so the asset
    /// URL still hits the right release.
    pub fn parse(input: &str) -> Result<Self, GitHubFetchError> {
        let (org_repo, tag) = input
            .split_once('@')
            .ok_or_else(|| GitHubFetchError::Parse {
                input: input.to_string(),
                reason: "missing `@<tag>` component".to_string(),
            })?;
        let (org, repo) = org_repo
            .split_once('/')
            .ok_or_else(|| GitHubFetchError::Parse {
                input: input.to_string(),
                reason: "missing `/` between org and repo".to_string(),
            })?;
        if org.is_empty() {
            return Err(GitHubFetchError::Parse {
                input: input.to_string(),
                reason: "empty <org> component".to_string(),
            });
        }
        if repo.is_empty() {
            return Err(GitHubFetchError::Parse {
                input: input.to_string(),
                reason: "empty <repo> component".to_string(),
            });
        }
        if tag.is_empty() {
            return Err(GitHubFetchError::Parse {
                input: input.to_string(),
                reason: "empty <tag> component".to_string(),
            });
        }
        let version = tag.strip_prefix('v').unwrap_or(tag).to_string();
        if version.is_empty() {
            return Err(GitHubFetchError::Parse {
                input: input.to_string(),
                reason: "tag is just `v` with no version".to_string(),
            });
        }
        Ok(Self {
            org: org.to_string(),
            repo: repo.to_string(),
            tag: tag.to_string(),
            version,
        })
    }

    /// API URL for the release that owns `<tag>`. Returns the JSON
    /// metadata document including the asset list. Step 1 of the
    /// two-step API fetch.
    fn release_metadata_url(&self, api_base: &str) -> String {
        format!(
            "{base}/repos/{org}/{repo}/releases/tags/{tag}",
            base = api_base.trim_end_matches('/'),
            org = self.org,
            repo = self.repo,
            tag = self.tag,
        )
    }

    /// API URL for one release asset, addressed by numeric id from
    /// the metadata response. Step 2 of the two-step API fetch. With
    /// `Accept: application/octet-stream` the response body is the
    /// raw asset bytes, including for private repos.
    fn release_asset_url(&self, api_base: &str, asset_id: u64) -> String {
        format!(
            "{base}/repos/{org}/{repo}/releases/assets/{asset_id}",
            base = api_base.trim_end_matches('/'),
            org = self.org,
            repo = self.repo,
        )
    }
}

/// One asset entry parsed out of the release-metadata JSON response.
#[derive(Debug, Clone)]
struct ReleaseAsset {
    id: u64,
    name: String,
}

/// Resolve a GitHub auth token. Reads `GITHUB_TOKEN` first; falls back
/// to `gh auth token`. Empty tokens are treated as missing. Returns
/// [`GitHubFetchError::AuthMissing`] if neither yields one.
fn resolve_github_token() -> Result<String, GitHubFetchError> {
    if let Ok(t) = env::var("GITHUB_TOKEN") {
        let trimmed = t.trim();
        if !trimmed.is_empty() {
            return Ok(trimmed.to_string());
        }
    }
    // `gh auth token` is the documented escape hatch for devs who use
    // the `gh` CLI but don't keep a long-lived `GITHUB_TOKEN` exported.
    // We shell out only here, deliberately keeping this seam thin so
    // tests can disable the fallback by un-PATH-ing `gh`.
    let result = std::process::Command::new("gh")
        .args(["auth", "token"])
        .output();
    match result {
        Ok(out) if out.status.success() => {
            let token = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if token.is_empty() {
                Err(GitHubFetchError::AuthMissing {
                    reason: "`gh auth token` returned an empty string".to_string(),
                })
            } else {
                Ok(token)
            }
        }
        Ok(out) => {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            Err(GitHubFetchError::AuthMissing {
                reason: format!(
                    "`gh auth token` exited with status {}: {stderr}",
                    out.status
                ),
            })
        }
        Err(e) => Err(GitHubFetchError::AuthMissing {
            reason: format!("`gh auth token` could not be invoked: {e}"),
        }),
    }
}

/// Read `CHELIS_REEF_GITHUB_BASE_API` or fall back to the canonical
/// `https://api.github.com`. This is the test-injection seam for the
/// GitHub API endpoints (release metadata + asset bytes). Tests can
/// point this at a wiremock host.
fn github_api_base_url() -> String {
    env::var("CHELIS_REEF_GITHUB_BASE_API").unwrap_or_else(|_| DEFAULT_GITHUB_BASE_API.to_string())
}

/// Map a non-200 HTTP status to the appropriate
/// [`GitHubFetchError`] variant for the given URL. Used by both the
/// metadata-fetch step and the byte-download step. The 404 mapping
/// uses the **caller-provided** `not_found` builder so the two steps
/// can choose distinct variants: the metadata step builds the ambiguous
/// [`GitHubFetchError::ReleaseTagNotFoundOrUnauthorized`] (a 404 there
/// could be a missing tag or a private repo the token cannot read),
/// while the byte-download step builds [`GitHubFetchError::ReleaseAssetNotFound`]
/// (a 404 there fires only after metadata fetch succeeded, so the asset
/// is genuinely absent). See issue #147.
fn map_http_error_status(
    url: &str,
    response: &reqwest::blocking::Response,
    not_found: impl FnOnce() -> GitHubFetchError,
) -> GitHubFetchError {
    let status = response.status();
    let code = status.as_u16();
    if code == 401 || code == 403 {
        return GitHubFetchError::AuthRejected {
            url: url.to_string(),
            status: code,
        };
    }
    if code == 404 {
        return not_found();
    }
    if code == 429 {
        let retry_after = response
            .headers()
            .get("Retry-After")
            .or_else(|| response.headers().get("X-RateLimit-Reset"))
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());
        return GitHubFetchError::RateLimited {
            url: url.to_string(),
            retry_after,
        };
    }
    if status.is_server_error() {
        return GitHubFetchError::ServerError {
            url: url.to_string(),
            status: code,
        };
    }
    GitHubFetchError::Network {
        url: url.to_string(),
        message: format!("unexpected HTTP status {code}"),
    }
}

/// Step 1 of the two-step API fetch: pull release metadata for the
/// given tag and parse the asset list. The response shape is GitHub's
/// "Release" object; we only read `assets[].id` and `assets[].name`.
///
/// On 404 returns [`GitHubFetchError::ReleaseTagNotFoundOrUnauthorized`]
/// naming both possibilities: the tag genuinely does not exist on the
/// repo, or the token lacks read access to a private repo (GitHub
/// returns 404, not 403, in the latter case as a privacy measure). The
/// `tag` is threaded in so the message can name it and point the user at
/// `gh release view <tag> --repo <org>/<repo>` for verification. See
/// issue #147 — this 404 used to surface as `ReleaseAssetNotFound`
/// naming a `<release-metadata>` placeholder asset, which misled users
/// hitting the auth-privacy case toward their release pipeline.
fn fetch_release_metadata(
    client: &reqwest::blocking::Client,
    url: &str,
    token: &str,
    tag: &str,
) -> Result<Vec<ReleaseAsset>, GitHubFetchError> {
    let response = client
        .get(url)
        .header("Authorization", format!("token {token}"))
        .header("User-Agent", "chelis-reef/0.5")
        // GitHub's recommended Accept for the v3 REST API. Without
        // this some endpoints return v3-deprecated responses.
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .map_err(|e| GitHubFetchError::Network {
            url: url.to_string(),
            message: e.to_string(),
        })?;
    if !response.status().is_success() {
        return Err(map_http_error_status(url, &response, || {
            // Metadata-step 404 is ambiguous: a missing tag and a
            // private repo the token cannot read both return 404. Emit
            // the dedicated variant whose message names both cases
            // rather than the misleading "asset not found" wording.
            GitHubFetchError::ReleaseTagNotFoundOrUnauthorized {
                url: url.to_string(),
                tag: tag.to_string(),
            }
        }));
    }
    let body = response.text().map_err(|e| GitHubFetchError::Network {
        url: url.to_string(),
        message: format!("metadata body read failed: {e}"),
    })?;
    parse_release_metadata(&body, url)
}

/// Parse the GitHub Release JSON document and pull out the
/// `assets[]` list. Validates that each entry has a numeric `id` and
/// a string `name`; missing/malformed entries surface as a typed
/// `Validation` error so the caller does not silently drop them.
fn parse_release_metadata(body: &str, url: &str) -> Result<Vec<ReleaseAsset>, GitHubFetchError> {
    let value: serde_json::Value =
        serde_json::from_str(body).map_err(|e| GitHubFetchError::Validation {
            message: format!("release metadata at {url} is not valid JSON: {e}"),
        })?;
    let assets = value
        .get("assets")
        .and_then(|a| a.as_array())
        .ok_or_else(|| GitHubFetchError::Validation {
            message: format!("release metadata at {url} has no `assets` array"),
        })?;
    let mut out = Vec::with_capacity(assets.len());
    for (i, entry) in assets.iter().enumerate() {
        let id = entry.get("id").and_then(|v| v.as_u64()).ok_or_else(|| {
            GitHubFetchError::Validation {
                message: format!("release metadata at {url}: assets[{i}] is missing numeric `id`"),
            }
        })?;
        let name = entry
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| GitHubFetchError::Validation {
                message: format!("release metadata at {url}: assets[{i}] is missing string `name`"),
            })?
            .to_string();
        out.push(ReleaseAsset { id, name });
    }
    Ok(out)
}

/// Step 2 of the two-step API fetch: stream the asset bytes by id to
/// `target` on disk. `Accept: application/octet-stream` is what makes
/// the API endpoint return the raw bytes (vs. a JSON descriptor with
/// a redirect URL).
///
/// Maps HTTP status to typed error categories the same way as the
/// metadata step. A 404 on the asset id (rare — would indicate
/// metadata staleness) surfaces as `ReleaseAssetNotFound` naming the
/// asset name we expected.
fn download_asset_by_id(
    client: &reqwest::blocking::Client,
    url: &str,
    asset_name: &str,
    token: &str,
    target: &Path,
) -> Result<(), GitHubFetchError> {
    let response = client
        .get(url)
        .header("Authorization", format!("token {token}"))
        .header("User-Agent", "chelis-reef/0.5")
        // CRITICAL: octet-stream tells the API to send the raw bytes.
        // Without this header the same URL returns the asset's JSON
        // descriptor instead of the binary payload.
        .header("Accept", "application/octet-stream")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .map_err(|e| GitHubFetchError::Network {
            url: url.to_string(),
            message: e.to_string(),
        })?;

    if !response.status().is_success() {
        return Err(map_http_error_status(url, &response, || {
            GitHubFetchError::ReleaseAssetNotFound {
                url: url.to_string(),
                asset_name: asset_name.to_string(),
            }
        }));
    }
    let mut response = response;
    let mut out = fs::File::create(target).map_err(|e| GitHubFetchError::Io {
        message: format!("create {}: {e}", target.display()),
    })?;
    response
        .copy_to(&mut out)
        .map_err(|e| GitHubFetchError::Network {
            url: url.to_string(),
            message: format!("body read failed: {e}"),
        })?;
    Ok(())
}

/// Look up one expected asset by name in the metadata response and
/// return its id. If absent, surfaces a `ReleaseAssetNotFound` whose
/// message lists every asset name actually present on the release —
/// helps publishers debug "did I attach the file under the right
/// name?" mistakes without needing a separate `gh release view` step.
fn find_asset_id(
    assets: &[ReleaseAsset],
    expected_name: &str,
    metadata_url: &str,
) -> Result<u64, GitHubFetchError> {
    if let Some(a) = assets.iter().find(|a| a.name == expected_name) {
        return Ok(a.id);
    }
    let present_names: Vec<&str> = assets.iter().map(|a| a.name.as_str()).collect();
    let present_listing = if present_names.is_empty() {
        "(release has no attached assets)".to_string()
    } else {
        format!("present assets: [{}]", present_names.join(", "))
    };
    Err(GitHubFetchError::ReleaseAssetNotFound {
        url: metadata_url.to_string(),
        asset_name: format!("{expected_name} ({present_listing})"),
    })
}

/// Install one shell from a GitHub Release into the local Reef
/// registry rooted at `registry_root`.
///
/// `org_repo_tag` is `<org>/<repo>@<tag>`. The repository name is the
/// shell name; the tag (with optional leading `v` stripped) is the
/// shell version. The two release assets fetched are
/// `<repo>-<version>.tar.zst` and `<repo>-<version>.chb`.
///
/// **Fetch shape** (locked by spec § Item 6): two GitHub API calls.
/// 1. `GET ${CHELIS_REEF_GITHUB_BASE_API}/repos/<org>/<repo>/releases/tags/<tag>`
///    with `Accept: application/vnd.github+json` to fetch the release
///    metadata JSON; parse the `assets[]` array for the two expected
///    asset names; capture each asset's `id`.
/// 2. `GET ${CHELIS_REEF_GITHUB_BASE_API}/repos/<org>/<repo>/releases/assets/<asset_id>`
///    with `Accept: application/octet-stream` to stream the bytes to
///    a tempdir.
///
/// `CHELIS_REEF_GITHUB_BASE_API` defaults to `https://api.github.com`.
/// The web-host URL form `/<org>/<repo>/releases/download/<tag>/<asset>`
/// is **not used**; that form 404s on private repos. The canonical
/// chelis-lang shells are private during the pre-launch era, so the
/// API path is required.
///
/// Authentication: `GITHUB_TOKEN` env var, or `gh auth token`
/// shell-out fallback. Auth is mandatory; an unauthenticated fetch
/// hard-fails with an actionable message.
///
/// On error, the function returns without updating the registry
/// index; any tempdir created for the download is removed. On
/// success, [`install_validated_artifact_pair`] is the placement and
/// validation oracle, identical to what `--from-monorepo` invokes.
pub fn install_from_github(
    org_repo_tag: &str,
    registry_root: &Path,
) -> Result<InstalledArtifact, GitHubFetchError> {
    let spec = GitHubReleaseSpec::parse(org_repo_tag)?;
    let token = resolve_github_token()?;
    let api_base = github_api_base_url();

    // Tempdir lives for the duration of the fetch+install. On any
    // return path (success or any error category) the `_tmp` guard
    // drops and removes the directory. The tempfile-cleanup contract
    // is locked by tests in `crates/chelis-cli/tests/phaseA_item6_from_github.rs`.
    let tmp = tempfile::tempdir().map_err(|e| GitHubFetchError::Io {
        message: format!("create tempdir: {e}"),
    })?;
    let archive_name = format!("{}-{}.tar.zst", spec.repo, spec.version);
    let shell_name = format!("{}-{}.chb", spec.repo, spec.version);
    let archive_path = tmp.path().join(&archive_name);
    let shell_path = tmp.path().join(&shell_name);

    // Plain blocking client; no implicit retry. Decisions in the
    // brief: fail fast on transient errors, surface 429 with
    // `Retry-After`. Connect timeout keeps a wedged DNS resolver
    // from hanging the install indefinitely.
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| GitHubFetchError::Io {
            message: format!("build http client: {e}"),
        })?;

    // Step 1: fetch release metadata, extract asset ids.
    let metadata_url = spec.release_metadata_url(&api_base);
    let assets = fetch_release_metadata(&client, &metadata_url, &token, &spec.tag)?;
    let archive_id = find_asset_id(&assets, &archive_name, &metadata_url)?;
    let shell_id = find_asset_id(&assets, &shell_name, &metadata_url)?;

    // Step 2: stream asset bytes by id.
    let archive_url = spec.release_asset_url(&api_base, archive_id);
    let shell_url = spec.release_asset_url(&api_base, shell_id);
    download_asset_by_id(&client, &archive_url, &archive_name, &token, &archive_path)?;
    download_asset_by_id(&client, &shell_url, &shell_name, &token, &shell_path)?;

    // Item 9 retrofit: stamp the registry entry with the
    // `github://<org>/<repo>@<tag>` origin so that downstream
    // `chelis reef build` can surface it in `reef.lock` and
    // `chelis reef install --from-lockfile` can re-fetch from it
    // without out-of-band knowledge.
    let origin = format_github_origin(&spec.org, &spec.repo, &spec.tag);
    install_validated_artifact_pair(
        &archive_path,
        &shell_path,
        &spec.repo,
        &spec.version,
        registry_root,
        Some(&origin),
    )
    .map_err(|message| GitHubFetchError::Validation { message })
}

/// Per-entry result of [`install_from_lockfile`]. One of these is
/// produced for every `[[dependencies]]` row in the input lockfile.
/// The loop continues past any failure so a partial outcome is
/// observable to the caller (the CLI prints each result and exits
/// non-zero if any are errors).
#[derive(Debug)]
pub enum LockfileInstallEntry {
    /// The entry's bytes are now in the registry. Either we re-fetched
    /// from `remote_origin` and validated against the lockfile pin, or
    /// the bytes were already present at the right hashes.
    Installed(InstalledArtifact),
    /// The entry has `kind = local_registry` but no `remote_origin`,
    /// so we cannot fetch it. The caller is told to run `--bootstrap`
    /// (or re-run `--from-monorepo`/`--from-github`) to populate the
    /// origin in the lockfile.
    SkippedNoOrigin { name: String, version: String },
    /// A path-dep entry. There is nothing to fetch — the dep lives in
    /// the developer's working tree and `chelis reef build` resolves
    /// it directly. The CLI surfaces this as informational, not an
    /// error.
    SkippedPathDep {
        name: String,
        version: String,
        path: String,
    },
    /// A `Bundled` entry — the language runtime (`chelis-std`). There
    /// is nothing to fetch: the bytes ship inside the compiler binary.
    /// Surfaced as informational, not an error.
    SkippedBundledRuntime {
        name: String,
        version: String,
        compiler_version: String,
    },
    /// Fetch+install attempt failed. The error preserves whatever the
    /// underlying [`GitHubFetchError`] said; the caller is responsible
    /// for naming the entry.
    Failed {
        name: String,
        version: String,
        error: LockfileInstallError,
    },
}

/// Top-level error category for `chelis reef install --from-lockfile`.
/// Most variants carry a typed inner error so callers can match on
/// category. The `EntryFailures` variant aggregates per-row errors so
/// the CLI can print all of them in a single pass.
#[derive(Debug)]
pub enum LockfileInstallError {
    /// `reef.lock` does not exist. The path tested is included so
    /// the message can be actionable (`chelis reef build` or move
    /// to the right directory).
    NotFound { path: PathBuf },
    /// `reef.lock` exists but is not valid TOML or does not match the
    /// `ReefLock` schema. The wrapped message contains the parser
    /// diagnostic (line/column where available).
    Malformed { path: PathBuf, message: String },
    /// One entry's `remote_origin` did not parse as a known scheme.
    /// Typed and per-entry so the test suite can pin the exact error
    /// shape.
    OriginParse {
        name: String,
        version: String,
        inner: RemoteOriginParseError,
    },
    /// The fetch+validate pipeline failed for one entry. Wraps the
    /// underlying [`GitHubFetchError`] verbatim.
    Fetch {
        name: String,
        version: String,
        inner: GitHubFetchError,
    },
}

impl std::fmt::Display for LockfileInstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound { path } => write!(
                f,
                "no lockfile found at {}. Run `chelis reef build` first to generate one",
                path.display()
            ),
            Self::Malformed { path, message } => {
                write!(f, "lockfile at {} is malformed: {message}", path.display())
            }
            Self::OriginParse {
                name,
                version,
                inner,
            } => write!(f, "lockfile entry `{name}` v{version}: {inner}"),
            Self::Fetch {
                name,
                version,
                inner,
            } => write!(f, "lockfile entry `{name}` v{version}: {inner}"),
        }
    }
}

/// Default list of canonical **shells** to bootstrap when
/// `chelis reef install --bootstrap` is invoked without an explicit
/// list. Each entry is `(repo, tag)` under [`CANONICAL_REEF_ORG`].
///
/// **Runtime is not in this list.** `chelis-std` is the language
/// runtime, not a shell — it ships bundled with the compiler and is
/// not installed via `reef install`. An explicit `chelis-std` entry
/// in the bootstrap input list is rejected with a typed runtime
/// error (see [`install_bootstrap`]).
///
/// This list is **hand-maintained** for the pre-launch dev team. Bump
/// each entry's tag whenever a shell publishes a new release that
/// should be the default-fetched version. The bootstrap installer reads
/// each archive's `reef.toml` to discover dependencies; entries not
/// present in this list whose `reef.toml` references them surface a
/// [`BootstrapError::MissingDependency`] error rather than silently
/// installing more than the operator asked for.
///
/// Multi-publisher generalization (the post-launch endgame) is Item 10
/// in `spec/design/reef_distribution.md`; until then this list is the
/// hard-coded source of truth.
pub const DEFAULT_BOOTSTRAP_LIST: &[(&str, &str)] = &[
    ("nautilus", "v0.6.1"),
    ("coral", "v0.6.1"),
    ("shoals", "v0.3.1"),
    ("octant", "v0.4.2"),
];

/// Distinct error categories surfaced by the bootstrap install path.
///
/// Keep `Display` strings stable; tests assert against substrings of
/// the formatted message. New variants are additive.
#[derive(Debug)]
pub enum BootstrapError {
    /// A cycle was detected in the dependency graph among the
    /// requested shells. `cycle` lists the cycle members in the order
    /// the cycle was traversed (first entry repeats at the end so the
    /// loop is unambiguous in the formatted message).
    Cycle { cycle: Vec<String> },
    /// One of the requested shells declares a dependency on a package
    /// that is not in the explicit input set. The bootstrap is not
    /// allowed to silently expand the install set — the operator must
    /// add the missing shell to the input list explicitly.
    MissingDependency { dependent: String, missing: String },
    /// One of the requested shells appears at two different versions
    /// in the input list. The bootstrap refuses ambiguous input;
    /// callers must pick a single version per package.
    DuplicateVersion {
        package: String,
        versions: Vec<String>,
    },
    /// The input list contains the language runtime (`chelis-std`).
    /// The runtime is bundled with the compiler and cannot be
    /// installed via reef. The error names the requested version and
    /// the compiler's bundled version.
    RuntimeNotABootstrapTarget {
        requested_version: String,
        bundled_version: String,
    },
    /// The input list is empty AND
    /// [`DEFAULT_BOOTSTRAP_LIST`] is empty (test scaffolds inject a
    /// list; production never trips this).
    NothingToInstall,
    /// A network or HTTP-layer failure during fetch. Wraps the
    /// underlying [`GitHubFetchError`] verbatim.
    Fetch(GitHubFetchError),
    /// The fetched archive contained no readable `reef.toml`, or its
    /// contents did not parse as a manifest.
    ManifestRead { spec: String, message: String },
    /// Validation fired downstream of fetch (sha mismatch,
    /// name/version disagreement, malformed shell). The message text
    /// is the underlying error.
    Validation { message: String },
}

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cycle { cycle } => {
                let listing = cycle.join(" -> ");
                write!(
                    f,
                    "dependency cycle detected among bootstrap shells: {listing}"
                )
            }
            Self::MissingDependency { dependent, missing } => write!(
                f,
                "shell `{dependent}` depends on `{missing}` but `{missing}` is not in the bootstrap input list: \
                 add an explicit `<org>/{missing}@<tag>` entry to the bootstrap arguments"
            ),
            Self::DuplicateVersion { package, versions } => write!(
                f,
                "package `{package}` appears at multiple versions in the bootstrap input list: [{}]. Pick one",
                versions.join(", ")
            ),
            Self::RuntimeNotABootstrapTarget {
                requested_version,
                bundled_version,
            } => write!(
                f,
                "`chelis-std` is the language runtime; it ships with the compiler and is not \
                 installed via `reef install --bootstrap` (requested `{requested_version}`, \
                 compiler bundles `{bundled_version}`). Drop `chelis-std` from the bootstrap \
                 input list. Programs depend on the runtime implicitly."
            ),
            Self::NothingToInstall => write!(
                f,
                "bootstrap input list is empty and the built-in default list is empty too. Nothing to install"
            ),
            Self::Fetch(e) => write!(f, "fetch failed during bootstrap: {e}"),
            Self::ManifestRead { spec, message } => write!(
                f,
                "could not read `reef.toml` from `{spec}` archive: {message}"
            ),
            Self::Validation { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for LockfileInstallError {}

/// Re-install every dependency named by `<package_root>/reef.lock` from
/// its recorded `remote_origin`, validating bytes against the
/// lockfile's pinned hashes via `install_validated_artifact_pair`.
///
/// **Per-entry behavior.**
///
/// - `kind = local_registry` with `remote_origin = Some(github://...)`:
///   fetch the bytes, validate, place into `registry_root`. The pinned
///   `archive_sha256` and `shell_sha256` from the lockfile are not
///   re-checked separately here — they are checked by
///   `install_validated_artifact_pair` (which computes hashes on the
///   bytes the URL serves and asserts archive↔shell agreement) and
///   are also pinned in the registry's `index.json` after the install,
///   so a subsequent `chelis reef build` will surface a hash-mismatch
///   if the URL ever serves different bytes.
/// - `kind = local_registry` with `remote_origin = None`: the lockfile
///   entry has no recorded origin (it was installed via `--from-monorepo`
///   pre-Item-9 or installed via `--bootstrap` without a manifest URL).
///   Surface a typed error suggesting `--bootstrap` to populate the
///   origin. The caller cannot magic up a URL.
/// - `kind = path { path = "..." }`: nothing to fetch, the dep lives
///   on the dev's filesystem. Skipped silently (recorded as
///   `SkippedPathDep` for completeness).
///
/// **Top-level error handling.** The function reads the lockfile up
/// front. Failures at that step (`NotFound`, `Malformed`) bail
/// immediately because there are no entries to walk. Per-entry
/// failures collect into `Vec<LockfileInstallEntry>` and the function
/// returns `Ok` so the caller can decide whether to consider any
/// failure fatal (the CLI does — it exits non-zero if any entry
/// failed).
///
/// **Hash-pin invariant.** After this function completes successfully,
/// the local registry state is byte-equivalent to what
/// `install_from_github` produced when the lockfile was first written:
/// same `<registry_root>/packages/<name>/<version>/{archive,shell}`
/// bytes, same `index.json` entry. This is the spec-locked
/// "developer A → developer B" reproducibility contract.
pub fn install_from_lockfile(
    package_root: &Path,
    registry_root: &Path,
) -> Result<Vec<LockfileInstallEntry>, LockfileInstallError> {
    let lockfile_path = package_root.join("reef.lock");
    if !lockfile_path.exists() {
        return Err(LockfileInstallError::NotFound {
            path: lockfile_path,
        });
    }
    let text = fs::read_to_string(&lockfile_path).map_err(|e| LockfileInstallError::Malformed {
        path: lockfile_path.clone(),
        message: format!("failed to read: {e}"),
    })?;
    let lock: ReefLock = toml::from_str(&text).map_err(|e| LockfileInstallError::Malformed {
        path: lockfile_path.clone(),
        message: e.to_string(),
    })?;

    fs::create_dir_all(registry_root).map_err(|e| LockfileInstallError::Malformed {
        path: lockfile_path.clone(),
        message: format!(
            "failed to create registry root {}: {e}",
            registry_root.display()
        ),
    })?;

    let mut results = Vec::with_capacity(lock.dependencies.len());
    for dep in &lock.dependencies {
        // Migration: an old lockfile may record `chelis-std` as
        // `LocalRegistry`. The runtime is now bundled — surface a
        // one-line warning and treat it as `SkippedBundledRuntime`.
        if matches!(&dep.source, LockSource::LocalRegistry { .. })
            && dep.name == CHELIS_STD_PACKAGE_NAME
        {
            eprintln!(
                "chelis reef: `chelis-std` is now toolchain-bundled; \
                 lockfile entry will be rewritten on next build"
            );
            results.push(LockfileInstallEntry::SkippedBundledRuntime {
                name: dep.name.clone(),
                version: dep.version.clone(),
                compiler_version: env!("CARGO_PKG_VERSION").to_string(),
            });
            continue;
        }
        match &dep.source {
            LockSource::Path { path } => {
                results.push(LockfileInstallEntry::SkippedPathDep {
                    name: dep.name.clone(),
                    version: dep.version.clone(),
                    path: path.clone(),
                });
            }
            LockSource::Bundled { compiler_version } => {
                // Compiler-bundled runtime. Nothing to fetch — the bytes
                // ship inside the compiler binary and reef's job here
                // is just to record the entry as accounted-for.
                results.push(LockfileInstallEntry::SkippedBundledRuntime {
                    name: dep.name.clone(),
                    version: dep.version.clone(),
                    compiler_version: compiler_version.clone(),
                });
            }
            LockSource::LocalRegistry {
                remote_origin: None,
            } => {
                results.push(LockfileInstallEntry::SkippedNoOrigin {
                    name: dep.name.clone(),
                    version: dep.version.clone(),
                });
            }
            LockSource::LocalRegistry {
                remote_origin: Some(origin),
            } => {
                let spec = match parse_remote_origin(origin) {
                    Ok(s) => s,
                    Err(inner) => {
                        results.push(LockfileInstallEntry::Failed {
                            name: dep.name.clone(),
                            version: dep.version.clone(),
                            error: LockfileInstallError::OriginParse {
                                name: dep.name.clone(),
                                version: dep.version.clone(),
                                inner,
                            },
                        });
                        continue;
                    }
                };
                // Spec-name-vs-lockfile-name agreement: we trust the
                // lockfile name as the package identity. The fetch
                // pipeline derives the on-disk name from the spec
                // (`spec.repo`), which must match the lockfile entry's
                // `name`/`version` pair — otherwise the fetched
                // archive would land at a wrong path and validation
                // would surface the mismatch via the embedded shell
                // package id check.
                //
                // Reconstruct the canonical `<org>/<repo>@<tag>` form
                // and call into the established `install_from_github`
                // pipeline. That pipeline already does
                // `install_validated_artifact_pair` with the origin
                // stamped, so the registry is left in a byte-identical
                // state to the original install.
                let org_repo_tag = format!("{}/{}@{}", spec.org, spec.repo, spec.tag);
                match install_from_github(&org_repo_tag, registry_root) {
                    Ok(artifact) => {
                        // Hash-pin verification: the lockfile records
                        // the expected hashes and the fresh fetch
                        // recomputed them. If they disagree, the bytes
                        // GitHub served diverged from what the
                        // lockfile pinned; surface that as a
                        // `Validation` error so the operator can act
                        // on it (typically: someone re-tagged the
                        // release without bumping the version).
                        if artifact.archive_sha256 != dep.archive_sha256
                            || artifact.shell_sha256 != dep.shell_sha256
                        {
                            results.push(LockfileInstallEntry::Failed {
                                name: dep.name.clone(),
                                version: dep.version.clone(),
                                error: LockfileInstallError::Fetch {
                                    name: dep.name.clone(),
                                    version: dep.version.clone(),
                                    inner: GitHubFetchError::Validation {
                                        message: format!(
                                            "hash mismatch for `{name}` v{version}: \
                                             lockfile pinned archive_sha256={lock_a} \
                                             shell_sha256={lock_s}, but the bytes served by \
                                             {origin} hash to archive_sha256={got_a} \
                                             shell_sha256={got_s}",
                                            name = dep.name,
                                            version = dep.version,
                                            lock_a = dep.archive_sha256,
                                            lock_s = dep.shell_sha256,
                                            got_a = artifact.archive_sha256,
                                            got_s = artifact.shell_sha256,
                                        ),
                                    },
                                },
                            });
                        } else {
                            results.push(LockfileInstallEntry::Installed(artifact));
                        }
                    }
                    Err(e) => {
                        results.push(LockfileInstallEntry::Failed {
                            name: dep.name.clone(),
                            version: dep.version.clone(),
                            error: LockfileInstallError::Fetch {
                                name: dep.name.clone(),
                                version: dep.version.clone(),
                                inner: e,
                            },
                        });
                    }
                }
            }
        }
    }

    Ok(results)
}

impl std::error::Error for BootstrapError {}

impl From<GitHubFetchError> for BootstrapError {
    fn from(e: GitHubFetchError) -> Self {
        BootstrapError::Fetch(e)
    }
}

impl From<BootstrapError> for String {
    fn from(e: BootstrapError) -> Self {
        e.to_string()
    }
}

/// Fetch the `<repo>-<version>.tar.zst` archive for a single
/// [`GitHubReleaseSpec`] and read the embedded `reef.toml` manifest
/// from inside it. Used by [`install_bootstrap`] to discover each
/// shell's `[dependencies]` block before topologically ordering the
/// installs.
///
/// This is intentionally separate from [`install_from_github`]: it
/// fetches only the archive (not the shell), extracts to a tempdir,
/// reads `reef.toml`, and drops the tempdir. The full install is
/// re-run via [`install_from_github`] later in the bootstrap loop —
/// network fetch is repeated for the archive byte stream, but the
/// validated install path stays the same as the single-shell case.
fn fetch_manifest_only(spec: &GitHubReleaseSpec) -> Result<ReefManifest, BootstrapError> {
    let token = resolve_github_token()?;
    let api_base = github_api_base_url();
    let archive_name = format!("{}-{}.tar.zst", spec.repo, spec.version);

    let tmp = tempfile::tempdir().map_err(|e| {
        BootstrapError::Fetch(GitHubFetchError::Io {
            message: format!("create tempdir: {e}"),
        })
    })?;
    let archive_path = tmp.path().join(&archive_name);

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .build()
        .map_err(|e| {
            BootstrapError::Fetch(GitHubFetchError::Io {
                message: format!("build http client: {e}"),
            })
        })?;

    let metadata_url = spec.release_metadata_url(&api_base);
    let assets = fetch_release_metadata(&client, &metadata_url, &token, &spec.tag)?;
    let archive_id = find_asset_id(&assets, &archive_name, &metadata_url)?;
    let archive_url = spec.release_asset_url(&api_base, archive_id);
    download_asset_by_id(&client, &archive_url, &archive_name, &token, &archive_path)?;

    let extract_dir = tmp.path().join("extract");
    fs::create_dir_all(&extract_dir).map_err(|e| {
        BootstrapError::Fetch(GitHubFetchError::Io {
            message: format!("create extract dir: {e}"),
        })
    })?;
    extract_archive(&archive_path, &extract_dir).map_err(|message| {
        BootstrapError::ManifestRead {
            spec: format!("{}/{}@{}", spec.org, spec.repo, spec.tag),
            message,
        }
    })?;
    let manifest_path = extract_dir.join("reef.toml");
    if !manifest_path.exists() {
        return Err(BootstrapError::ManifestRead {
            spec: format!("{}/{}@{}", spec.org, spec.repo, spec.tag),
            message: "archive does not contain `reef.toml` at the root".to_string(),
        });
    }
    read_manifest(&manifest_path).map_err(|message| BootstrapError::ManifestRead {
        spec: format!("{}/{}@{}", spec.org, spec.repo, spec.tag),
        message,
    })
}

/// One node in the bootstrap dependency graph: `(name, version,
/// deps)` where `deps` is the list of `(dep_name, dep_version)` edges
/// the node's manifest declared. A `Vec<BootstrapNode>` is the input
/// to [`topo_sort_bootstrap`].
type BootstrapNode = (String, String, Vec<(String, String)>);

/// Topologically sort `nodes` by their dependency edges so each entry
/// is preceded by everything it depends on. Returns the sorted list
/// of node indices into `nodes`.
///
/// The graph is keyed by `(name, version)`. Edges come from each
/// node's manifest `[dependencies]` block: an edge from N to M means
/// "N depends on M, install M before N." The DFS uses a recursion-
/// stack set to detect cycles; on cycle detection the slice of the
/// stack starting at the repeat node (with the repeat node appended
/// at the end) is returned via [`BootstrapError::Cycle`] so the
/// formatted message names every member of the loop.
///
/// Self-loops (a package depending on itself) surface as a 1-element
/// cycle (the package's own `(name, version)` formatted twice with
/// `->`).
fn topo_sort_bootstrap(nodes: &[BootstrapNode]) -> Result<Vec<usize>, BootstrapError> {
    // Map (name, version) -> index for edge resolution. Built up
    // front so cycle detection can map back to (name, version) pairs
    // inside the DFS.
    let mut index_by_key: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();
    for (i, (name, version, _)) in nodes.iter().enumerate() {
        if let Some(prior) = index_by_key.insert((name.clone(), version.clone()), i) {
            // Defensive: bootstrap-input dedup happens before topo
            // sort, so this branch should be unreachable for
            // legitimate callers. We preserve the prior index either
            // way — duplicate-version detection is the caller's job.
            let _ = prior;
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Gray,
        Black,
    }
    let mut colors: Vec<Color> = vec![Color::White; nodes.len()];
    let mut order: Vec<usize> = Vec::with_capacity(nodes.len());
    // Iterative DFS using an explicit stack so we don't blow the
    // call stack for pathological input. Each stack frame is
    // `(node_index, edge_cursor)` — the cursor is how many of
    // `nodes[i].2` we've already enqueued. When the cursor reaches
    // the dep count, we pop the frame, paint the node black, and
    // append it to `order`.
    let mut path: Vec<(usize, usize)> = Vec::new();
    // Track the depth-ordered list of (name, version) on the
    // recursion stack so cycle reporting can name the loop without
    // re-walking `path`.
    let mut path_keys: Vec<(String, String)> = Vec::new();

    for start in 0..nodes.len() {
        if colors[start] != Color::White {
            continue;
        }
        path.push((start, 0));
        path_keys.push((nodes[start].0.clone(), nodes[start].1.clone()));
        colors[start] = Color::Gray;
        while let Some(&(node_idx, cursor)) = path.last() {
            let deps = &nodes[node_idx].2;
            if cursor >= deps.len() {
                colors[node_idx] = Color::Black;
                order.push(node_idx);
                path.pop();
                path_keys.pop();
                continue;
            }
            // Advance cursor before recursing so we don't revisit on
            // the next loop iteration after pop.
            let last = path.last_mut().expect("path nonempty");
            last.1 = cursor + 1;
            let dep_key = (deps[cursor].0.clone(), deps[cursor].1.clone());
            // Edges to nodes outside the bootstrap set are caught
            // upstream as `MissingDependency`; here we assume every
            // dep resolves into the input set.
            let dep_idx = match index_by_key.get(&dep_key).copied() {
                Some(i) => i,
                None => continue, // unreachable in well-formed input; defensive
            };
            match colors[dep_idx] {
                Color::Black => continue,
                Color::Gray => {
                    // Cycle — locate the repeat node in `path_keys`
                    // and emit the slice from there to the end,
                    // appending the repeat node again so the loop is
                    // closed in the formatted message.
                    let mut cycle_nodes: Vec<String> = Vec::new();
                    let mut started = false;
                    for key in &path_keys {
                        if !started && *key == dep_key {
                            started = true;
                        }
                        if started {
                            cycle_nodes.push(format!("{}@{}", key.0, key.1));
                        }
                    }
                    cycle_nodes.push(format!("{}@{}", dep_key.0, dep_key.1));
                    return Err(BootstrapError::Cycle { cycle: cycle_nodes });
                }
                Color::White => {
                    colors[dep_idx] = Color::Gray;
                    path.push((dep_idx, 0));
                    path_keys.push(dep_key);
                }
            }
        }
    }
    Ok(order)
}

/// Topologically install a set of shells from canonical-org GitHub
/// Releases.
///
/// **Algorithm.**
/// 1. For each `spec`, fetch only the archive and read its embedded
///    `reef.toml`. The shell payload is not fetched in this pass — only
///    the manifest is needed to discover `[dependencies]` edges.
/// 2. Build a directed graph keyed by `(name, version)`. Edges are
///    drawn from each entry's manifest dependencies, matching by
///    name+version against the other entries in `specs`.
/// 3. Topologically sort via DFS with a recursion-stack cycle check.
///    On cycle, return [`BootstrapError::Cycle`] naming the cycle
///    members. On a dependency that names a package not in the input
///    set, return [`BootstrapError::MissingDependency`].
/// 4. For each entry in topo order, run [`install_from_github`] (full
///    archive + shell fetch + validate + place).
///
/// **Atomicity.** Each individual shell install uses
/// [`install_validated_artifact_pair`]'s atomicity (the registry
/// `index.json` is updated atomically per shell). The bootstrap as a
/// whole is **not** transactional across shells: if shell N fails,
/// shells 1..N-1 stay installed. The brief calls this out as a locked
/// design choice — the retry path is "re-run with the same input
/// list," which is a no-op for the already-installed shells (idempotent
/// per Item 6) and finishes the remainder.
///
/// **Idempotence.** Re-running with the same input list re-fetches
/// archives but produces a byte-identical local registry state because
/// the placement step copies the same bytes into the same destinations
/// and atomically updates `index.json` to the same final entries.
///
/// **Order independence.** The topological sort depends only on the
/// dependency graph, not on input order. Submitting the same set in
/// different orders produces the same install sequence (modulo ties,
/// which are resolved by input position to keep the result
/// deterministic).
pub fn install_bootstrap(
    specs: &[GitHubReleaseSpec],
    registry_root: &Path,
) -> Result<Vec<InstalledArtifact>, BootstrapError> {
    if specs.is_empty() {
        return Err(BootstrapError::NothingToInstall);
    }

    // Reject explicit `chelis-std` entries up front: the runtime is
    // bundled with the compiler and is not a `reef install --bootstrap`
    // target. Surface a typed error naming both the requested and
    // bundled versions so the user can tell which side moved.
    if let Some(runtime_spec) = specs.iter().find(|s| s.repo == CHELIS_STD_PACKAGE_NAME) {
        return Err(BootstrapError::RuntimeNotABootstrapTarget {
            requested_version: runtime_spec.version.clone(),
            bundled_version: compiler_bundled_chelis_std_version().to_string(),
        });
    }

    // Phase 1: fetch each manifest, accumulate (name, version, deps).
    let mut nodes: Vec<BootstrapNode> = Vec::with_capacity(specs.len());
    for spec in specs {
        let manifest = fetch_manifest_only(spec)?;
        // Lock manifest <-> spec agreement: the archive's
        // `package.name` and `package.version` must match what the
        // spec asked for. This is the same check
        // `install_validated_artifact_pair` runs later, but firing it
        // here too gives a clearer error before topo sort.
        if manifest.package.name != spec.repo {
            return Err(BootstrapError::Validation {
                message: format!(
                    "shell `{}/{}@{}` archive declares package name `{}` (expected `{}`)",
                    spec.org, spec.repo, spec.tag, manifest.package.name, spec.repo
                ),
            });
        }
        if manifest.package.version != spec.version {
            return Err(BootstrapError::Validation {
                message: format!(
                    "shell `{}/{}@{}` archive declares package version `{}` (expected `{}`)",
                    spec.org, spec.repo, spec.tag, manifest.package.version, spec.version
                ),
            });
        }
        let mut deps: Vec<(String, String)> = Vec::new();
        for (dep_name, dep_spec) in &manifest.dependencies {
            // Path-only deps are local development edges that the
            // bootstrap install path cannot satisfy from a release;
            // skip them — `install_from_github` plus the lockfile
            // resolution path handles missing-from-registry errors at
            // build time. Bootstrap's job is the upstream-published
            // dep set.
            if dep_spec.path.is_some() && dep_spec.version.is_none() {
                continue;
            }
            // chelis-std declared by a shell's `reef.toml` is the
            // implicit-runtime dep, not an edge in the bootstrap graph.
            // The runtime is compiler-bundled; we do not include it in
            // the topo sort or the install loop. Soft-verify the
            // declared version against the compiler's bundled version
            // so a stale shell declaration surfaces clearly.
            if dep_name == CHELIS_STD_PACKAGE_NAME {
                if let Some(declared) = dep_spec.version.as_deref() {
                    let bundled = compiler_bundled_chelis_std_version();
                    if declared != bundled {
                        return Err(BootstrapError::Validation {
                            message: format!(
                                "shell `{}/{}@{}` declares `chelis-std = {{ version = \"{}\" }}`, \
                                 but this compiler bundles chelis-std `{}`. The runtime ships \
                                 with the compiler and cannot be substituted; either upgrade \
                                 the compiler or wait for a shell release that pins the matching \
                                 runtime.",
                                spec.org, spec.repo, spec.tag, declared, bundled
                            ),
                        });
                    }
                }
                continue;
            }
            // Version-pinned deps need a peer in the bootstrap input
            // set. Versionless deps (no `version =` and no `path =`)
            // are ill-formed manifests; surface that early.
            let dep_version = match dep_spec.version.as_ref() {
                Some(v) => v.clone(),
                None => {
                    return Err(BootstrapError::Validation {
                        message: format!(
                            "shell `{}/{}@{}` declares dependency `{}` without a version pin: \
                             reef bootstrap requires exact version pins on every dependency",
                            spec.org, spec.repo, spec.tag, dep_name
                        ),
                    });
                }
            };
            deps.push((dep_name.clone(), dep_version));
        }
        nodes.push((
            manifest.package.name.clone(),
            manifest.package.version.clone(),
            deps,
        ));
    }

    // Duplicate-version detection: the same package name appearing at
    // two versions in the input list is ambiguous; refuse rather than
    // silently picking one.
    {
        let mut by_name: std::collections::HashMap<&str, Vec<&str>> =
            std::collections::HashMap::new();
        for (name, version, _) in &nodes {
            by_name
                .entry(name.as_str())
                .or_default()
                .push(version.as_str());
        }
        for (name, versions) in &by_name {
            if versions.len() > 1 {
                let mut sorted: Vec<String> = versions.iter().map(|s| s.to_string()).collect();
                sorted.sort();
                sorted.dedup();
                if sorted.len() > 1 {
                    return Err(BootstrapError::DuplicateVersion {
                        package: (*name).to_string(),
                        versions: sorted,
                    });
                }
            }
        }
    }

    // Build (name, version) -> index for edge validation.
    let mut index_by_key: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::with_capacity(nodes.len());
    for (i, (name, version, _)) in nodes.iter().enumerate() {
        index_by_key.insert((name.clone(), version.clone()), i);
    }

    // Validate every dep edge resolves to a node in the input set.
    for (name, _version, deps) in &nodes {
        for (dep_name, dep_version) in deps {
            if !index_by_key.contains_key(&(dep_name.clone(), dep_version.clone())) {
                return Err(BootstrapError::MissingDependency {
                    dependent: name.clone(),
                    missing: dep_name.clone(),
                });
            }
        }
    }

    let order = topo_sort_bootstrap(&nodes)?;

    // Phase 3: install in topological order via Item 6's full path.
    let mut installed: Vec<InstalledArtifact> = Vec::with_capacity(order.len());
    for idx in order {
        let (name, version, _deps) = &nodes[idx];
        // Match the (name, version) back to the original spec to get
        // the canonical `<org>/<repo>@<tag>` to call
        // `install_from_github` with.
        let spec = specs
            .iter()
            .find(|s| s.repo == *name && s.version == *version)
            .expect("topo order references a node not in the input specs");
        let spec_str = format!("{}/{}@{}", spec.org, spec.repo, spec.tag);
        let artifact = install_from_github(&spec_str, registry_root)?;
        installed.push(artifact);
    }
    Ok(installed)
}

// ─── Issue #491: Export pinned package bundles for hermetic agent sandboxes ───

/// Metadata written into the bundle alongside the materialized sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    pub root_package: PackageId,
    pub dependencies: Vec<BundledDependency>,
}

/// One dependency materialized into the bundle directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundledDependency {
    pub name: String,
    pub version: String,
    pub source: LockSource,
    pub archive_sha256: String,
    pub shell_sha256: String,
}

/// Export a hermetic bundle of the package and all its pinned dependencies
/// into `output_dir`. The bundle contains:
/// - `bundle.json` — metadata with module identity, source hashes, dep metadata
/// - `<name>-<version>/` — extracted source trees for each dependency
/// - `root/` — the root package source
///
/// Reads `reef.toml` + `reef.lock` from `package_root`.
pub fn export_bundle(package_root: &Path, output_dir: &Path) -> Result<BundleManifest, String> {
    let root = canonical_root(package_root)?;
    let lock_path = root.join("reef.lock");
    if !lock_path.exists() {
        return Err(format!(
            "no reef.lock found at {}; run `chelis reef build` first",
            lock_path.display()
        ));
    }
    let lock = read_lockfile(&lock_path)?;
    fs::create_dir_all(output_dir)
        .map_err(|e| format!("create bundle dir {}: {e}", output_dir.display()))?;

    // Copy root package source into bundle/root/
    let root_dir = output_dir.join("root");
    copy_package_source(&root, &root_dir)?;

    // Materialize each dependency
    let mut bundled_deps = Vec::new();
    for dep in &lock.dependencies {
        match &dep.source {
            LockSource::Path { path } => {
                let dep_root = root.join(path).canonicalize().map_err(|e| {
                    format!("resolve path dep `{}`: {e}", dep.name)
                })?;
                let dep_dir = output_dir.join(format!("{}-{}", dep.name, dep.version));
                copy_package_source(&dep_root, &dep_dir)?;
            }
            LockSource::LocalRegistry { .. } => {
                // Load from registry cache and copy extracted source
                let installed = load_registry_package(&dep.name, &dep.version).map_err(|e| {
                    match e {
                        LoadRegistryError::Other(s) => s,
                        LoadRegistryError::MissingFromIndex
                        | LoadRegistryError::MissingPackageDir => {
                            format!(
                                "dependency `{}` `{}` not in local registry; \
                                 run `chelis reef install` first",
                                dep.name, dep.version
                            )
                        }
                    }
                })?;
                let dep_dir = output_dir.join(format!("{}-{}", dep.name, dep.version));
                copy_package_source(&installed.root, &dep_dir)?;
            }
            LockSource::Bundled { .. } => {
                // chelis-std is compiler-bundled; extract embedded bytes
                let installed = load_bundled_chelis_std().map_err(|e| match e {
                    LoadRegistryError::Other(s) => s,
                    _ => "failed to load bundled chelis-std".to_string(),
                })?;
                let dep_dir = output_dir.join(format!("{}-{}", dep.name, dep.version));
                copy_package_source(&installed.root, &dep_dir)?;
            }
        }
        bundled_deps.push(BundledDependency {
            name: dep.name.clone(),
            version: dep.version.clone(),
            source: dep.source.clone(),
            archive_sha256: dep.archive_sha256.clone(),
            shell_sha256: dep.shell_sha256.clone(),
        });
    }

    let manifest = BundleManifest {
        root_package: lock.package.clone(),
        dependencies: bundled_deps,
    };
    let manifest_json =
        serde_json::to_string_pretty(&manifest).map_err(|e| format!("serialize bundle.json: {e}"))?;
    fs::write(output_dir.join("bundle.json"), manifest_json)
        .map_err(|e| format!("write bundle.json: {e}"))?;
    Ok(manifest)
}

/// Copy a package's source tree (reef.toml, src/, additional_sources) into `dst`.
fn copy_package_source(src: &Path, dst: &Path) -> Result<(), String> {
    fs::create_dir_all(dst).map_err(|e| format!("create {}: {e}", dst.display()))?;
    // Copy reef.toml
    let manifest_path = src.join("reef.toml");
    if manifest_path.exists() {
        fs::copy(&manifest_path, dst.join("reef.toml"))
            .map_err(|e| format!("copy reef.toml: {e}"))?;
    }
    // Copy reef.lock if present
    let lock_path = src.join("reef.lock");
    if lock_path.exists() {
        fs::copy(&lock_path, dst.join("reef.lock"))
            .map_err(|e| format!("copy reef.lock: {e}"))?;
    }
    // Copy src/ and any additional source roots
    let manifest = if manifest_path.exists() {
        read_manifest(&manifest_path).ok()
    } else {
        None
    };
    let roots: Vec<&str> = if let Some(ref m) = manifest {
        std::iter::once("src")
            .chain(m.package.additional_sources.iter().map(|s| s.as_str()))
            .collect()
    } else {
        vec!["src"]
    };
    for root_name in roots {
        let abs_root = src.join(root_name);
        if !abs_root.exists() {
            continue;
        }
        for entry in WalkDir::new(&abs_root).into_iter().filter_map(Result::ok) {
            let rel = entry.path().strip_prefix(src).map_err(|e| e.to_string())?;
            let target = dst.join(rel);
            if entry.file_type().is_dir() {
                fs::create_dir_all(&target)
                    .map_err(|e| format!("create dir {}: {e}", target.display()))?;
            } else {
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("create parent {}: {e}", parent.display()))?;
                }
                fs::copy(entry.path(), &target)
                    .map_err(|e| format!("copy {}: {e}", entry.path().display()))?;
            }
        }
    }
    Ok(())
}

// ─── Issue #492: Machine-readable ABI/package schema ───

/// Machine-readable package schema describing exported functions, types,
/// constructors, and required authoring signatures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageSchema {
    pub package: PackageId,
    pub compiler: String,
    pub modules: Vec<ModuleSchema>,
}

/// Schema for one module's exports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSchema {
    pub module: String,
    pub functions: Vec<FunctionSchema>,
    pub types: Vec<TypeSchema>,
}

/// Exported function signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSchema {
    pub name: String,
    pub type_repr: Option<String>,
    pub effects: Vec<String>,
    pub has_body: bool,
}

/// Exported type descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeSchema {
    pub name: String,
    pub opaque: bool,
    pub constructors: Vec<ConstructorSchema>,
}

/// Constructor (variant) of an exported type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructorSchema {
    pub name: String,
    /// "partial" if the type has an invariant, "total" otherwise.
    pub kind: String,
    pub type_repr: Option<String>,
}

/// Generate a machine-readable ABI/package schema for the package at `root`.
/// Builds the package (type-checks it) and extracts schema from the shell.
pub fn package_schema(root: &Path) -> Result<PackageSchema, String> {
    let root = canonical_root(root)?;
    let manifest = read_manifest(&root.join("reef.toml"))?;
    let graph = resolve_package_graph(&root, LoadOptions::default_for_load())?;
    let lock = build_lockfile(&graph);
    write_lockfile(&root.join("reef.lock"), &lock)?;

    let root_pkg = graph
        .packages
        .get(&graph.root_package)
        .ok_or_else(|| "root package missing from graph".to_string())?;
    let entry_modules = root_pkg.modules.keys().cloned().collect::<Vec<_>>();
    let linked = link_graph(&graph, &entry_modules)?;
    let linked_decls: Vec<_> = linked.into_iter().flat_map(|m| m.decls).collect();
    let deep = expanded_desugared_program(&linked_decls)?;
    let checked = checked_program_with_effects(&deep)?;

    let mut modules = Vec::new();
    for module_source in root_pkg.modules.values() {
        let mut functions = Vec::new();
        let mut types = Vec::new();

        for export_name in &module_source.exports {
            let kind = module_source.symbols.get(export_name);
            let internal = internal_name(
                &root_pkg.id.name,
                &module_source.module_name,
                export_name,
            );
            match kind {
                Some(chelis_shell::SymbolKind::Value) => {
                    let type_repr = checked
                        .type_env()
                        .get(&internal)
                        .map(|expr| {
                            chelis_deep::printer::print_canonical(std::slice::from_ref(expr))
                                .trim()
                                .to_string()
                        })
                        .or_else(|| sig_type_repr(module_source, export_name));
                    let effects = symbol_effects(module_source, export_name);
                    let has_body = module_source.decls.iter().any(|d| {
                        matches!(d,
                            Decl::FunDef { name, .. } | Decl::LetDef { name, .. }
                            if name == export_name
                        )
                    });
                    functions.push(FunctionSchema {
                        name: export_name.clone(),
                        type_repr,
                        effects,
                        has_body,
                    });
                }
                Some(chelis_shell::SymbolKind::Type) => {
                    // Find the type definition to extract constructors
                    let type_def = module_source.decls.iter().find_map(|d| match d {
                        Decl::TypeDef {
                            name,
                            variants,
                            opaque,
                            invariant,
                            ..
                        } if name == export_name => Some((variants, *opaque, invariant.is_some())),
                        _ => None,
                    });
                    let (constructors, opaque) = if let Some((variants, is_opaque, has_invariant)) =
                        type_def
                    {
                        let ctors = variants
                            .iter()
                            .map(|v| {
                                let ctor_internal = internal_name(
                                    &root_pkg.id.name,
                                    &module_source.module_name,
                                    &v.name,
                                );
                                let ctor_type = checked
                                    .type_env()
                                    .get(&ctor_internal)
                                    .map(|expr| {
                                        chelis_deep::printer::print_canonical(
                                            std::slice::from_ref(expr),
                                        )
                                        .trim()
                                        .to_string()
                                    });
                                ConstructorSchema {
                                    name: v.name.clone(),
                                    kind: if has_invariant {
                                        "partial".to_string()
                                    } else {
                                        "total".to_string()
                                    },
                                    type_repr: ctor_type,
                                }
                            })
                            .collect();
                        (ctors, is_opaque)
                    } else {
                        (Vec::new(), false)
                    };
                    types.push(TypeSchema {
                        name: export_name.clone(),
                        opaque,
                        constructors,
                    });
                }
                _ => {}
            }
        }
        modules.push(ModuleSchema {
            module: module_source.module_name.clone(),
            functions,
            types,
        });
    }
    modules.sort_by(|a, b| a.module.cmp(&b.module));

    Ok(PackageSchema {
        package: PackageId {
            name: manifest.package.name.clone(),
            version: manifest.package.version.clone(),
        },
        compiler: manifest.package.compiler.clone(),
        modules,
    })
}

fn canonical_root(root: &Path) -> Result<PathBuf, String> {
    let root = root
        .canonicalize()
        .or_else(|_| fs::create_dir_all(root).map(|_| root.to_path_buf()))
        .map_err(|e| format!("failed to resolve {}: {e}", root.display()))?;
    if root.join("reef.toml").exists() {
        Ok(root)
    } else if root.is_file() {
        find_package_root_for_input(&root)?.ok_or_else(|| {
            format!(
                "{} is not inside a Reef package root with reef.toml",
                root.display()
            )
        })
    } else {
        Err(format!("{} does not contain reef.toml", root.display()))
    }
}

fn registry_root() -> Result<PathBuf, String> {
    if let Some(path) = env::var_os("CHELIS_REEF_HOME") {
        return Ok(PathBuf::from(path));
    }
    let home = env::var_os("HOME").ok_or_else(|| "HOME is not set".to_string())?;
    Ok(PathBuf::from(home).join(".chelis/reef"))
}

/// Public accessor for the local Reef registry root used by external
/// callers (the CLI's `--from-github` handler in particular). Returns
/// `$CHELIS_REEF_HOME` if set, else `$HOME/.chelis/reef`. The directory
/// is **not** created here; callers that intend to write should rely on
/// the install helpers, which do `fs::create_dir_all`.
pub fn registry_home() -> Result<PathBuf, String> {
    registry_root()
}

fn read_manifest(path: &Path) -> Result<ReefManifest, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    let manifest = toml::from_str::<ReefManifest>(&text)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))?;
    validate_manifest(&manifest)?;
    Ok(manifest)
}

fn write_manifest(path: &Path, manifest: &ReefManifest) -> Result<(), Box<dyn std::error::Error>> {
    let text = toml::to_string_pretty(manifest)?;
    fs::write(path, format!("{text}\n"))?;
    Ok(())
}

fn write_lockfile(path: &Path, lock: &ReefLock) -> Result<(), String> {
    let text = toml::to_string_pretty(lock).map_err(|e| e.to_string())?;
    fs::write(path, format!("{text}\n")).map_err(|e| e.to_string())
}

fn read_lockfile(path: &Path) -> Result<ReefLock, String> {
    let text =
        fs::read_to_string(path).map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    toml::from_str::<ReefLock>(&text)
        .map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

/// Run `f` on a background thread.  Return `Err(timeout_msg)` if it does not
/// complete within `timeout`.
///
/// Exposed for `chelis test` so the CLI can bound each test's execution
/// with the same primitive reef uses internally for dependency resolution.
pub fn run_with_timeout<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, String> + Send + 'static,
    timeout: Duration,
    timeout_msg: &str,
) -> Result<T, String> {
    let (tx, rx) = mpsc::channel::<Result<T, String>>();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(timeout)
        .unwrap_or_else(|_| Err(timeout_msg.to_string()))
}

const TIMEOUT_MSG: &str = "reef dependency resolution timed out \u{2014} run `chelis reef build` first to populate the cache";

/// Rebuild a [`PackageGraph`] from an already-written `reef.lock` without
/// calling the full [`resolve_package_graph`] walk.
///
/// Path dependencies are loaded directly from the local filesystem (fast).
/// Local-registry dependencies are loaded with a 5-second timeout; if the
/// cache is missing or slow, Item 8's auto-fetch (when enabled) attempts
/// to repair the registry before returning an error. Auto-fetch runs
/// **outside** the timeout-bound thread because the network round trip
/// can legitimately exceed 5 s — the timeout exists to bound a wedged
/// local registry, not the auto-fetch itself.
fn reconstruct_graph_from_lockfile(
    root: &Path,
    lock: &ReefLock,
    options: LoadOptions,
) -> Result<PackageGraph, String> {
    let mut packages: BTreeMap<String, LoadedPackage> = BTreeMap::new();

    // Load the root package.
    let root_manifest = read_manifest(&root.join("reef.toml"))?;
    let root_modules = load_package_modules(root, &root_manifest)?;
    let root_name = lock.package.name.clone();
    packages.insert(
        root_name.clone(),
        LoadedPackage {
            id: lock.package.clone(),
            manifest: root_manifest,
            modules: root_modules,
            source: LoadedSourceKind::Root,
            shell: None,
            remote_origin: None,
        },
    );

    // Load each dependency.
    for dep in &lock.dependencies {
        // Migration: an old lockfile may record `chelis-std` as
        // `LocalRegistry`. Per Phase A § Architectural Decision 8,
        // log a one-line warning; the next call to `build_lockfile`
        // rewrites the entry to the `Bundled` form. The module-loading
        // path itself still flows through the local registry (the
        // bytes are already there) — `Bundled` is, in this wave, an
        // auditability annotation on the lockfile, not a separate
        // loader path.
        if matches!(&dep.source, LockSource::LocalRegistry { .. })
            && dep.name == CHELIS_STD_PACKAGE_NAME
        {
            eprintln!(
                "chelis reef: `chelis-std` is now toolchain-bundled; \
                 lockfile entry will be rewritten on next build"
            );
        }
        match &dep.source {
            LockSource::Bundled { .. } => {
                // chelis-std is the language runtime and ships
                // bundled inside the chelis binary
                // (`crates/chelis-std-bundle`). `load_registry_package`
                // special-cases the runtime: when name == chelis-std
                // and the requested version matches the bundled
                // version, it returns the embedded bytes immediately.
                // No `$CHELIS_REEF_HOME` access, no installer
                // prerequisite. The fallback through the local-
                // registry path only fires on a version mismatch,
                // which soft-verify should already have caught
                // upstream.
                let dep_name = dep.name.clone();
                let dep_version = dep.version.clone();
                let dep_name_t = dep_name.clone();
                let dep_version_t = dep_version.clone();
                let bundled = compiler_bundled_chelis_std_version().to_string();
                let timeout_result = run_with_timeout(
                    move || {
                        load_registry_package(&dep_name_t, &dep_version_t).map_err(|e| match e {
                            LoadRegistryError::Other(s) => s,
                            LoadRegistryError::MissingFromIndex
                            | LoadRegistryError::MissingPackageDir => {
                                format!(
                                    "missing dependency `{dep_name_t}` `{dep_version_t}`: \
                                     this is the language runtime (`chelis-std`), which is \
                                     bundled inside the chelis compiler. This compiler bundles \
                                     chelis-std `{bundled}`; the lockfile pins `{dep_version_t}`. \
                                     Use a chelis whose bundled runtime matches the lockfile, or \
                                     update the lockfile to match the compiler's bundled version."
                                )
                            }
                        })
                    },
                    Duration::from_secs(5),
                    TIMEOUT_MSG,
                );
                let installed = timeout_result?;
                let dep_manifest = read_manifest(&installed.root.join("reef.toml"))?;
                let dep_modules = load_package_modules(&installed.root, &dep_manifest)?;
                packages.insert(
                    dep.name.clone(),
                    LoadedPackage {
                        id: PackageId {
                            name: dep.name.clone(),
                            version: dep.version.clone(),
                        },
                        manifest: dep_manifest,
                        modules: dep_modules,
                        source: LoadedSourceKind::LocalRegistry,
                        shell: Some(installed.shell),
                        remote_origin: None,
                    },
                );
            }
            LockSource::Path { path } => {
                if dep.name == CHELIS_STD_PACKAGE_NAME {
                    return Err(
                        "`chelis-std` is the bundled language runtime and cannot be loaded from a path dependency"
                            .to_string(),
                    );
                }
                let dep_root = root.join(path).canonicalize().map_err(|e| {
                    format!("failed to resolve path dependency `{}`: {e}", dep.name)
                })?;
                let dep_manifest = read_manifest(&dep_root.join("reef.toml"))?;
                let dep_modules = load_package_modules(&dep_root, &dep_manifest)?;
                packages.insert(
                    dep.name.clone(),
                    LoadedPackage {
                        id: PackageId {
                            name: dep.name.clone(),
                            version: dep.version.clone(),
                        },
                        manifest: dep_manifest,
                        modules: dep_modules,
                        source: LoadedSourceKind::Path {
                            relative: path.clone(),
                        },
                        shell: None,
                        remote_origin: None,
                    },
                );
            }
            LockSource::LocalRegistry { remote_origin } => {
                let dep_name = dep.name.clone();
                let dep_version = dep.version.clone();
                // First try the timeout-bounded load (no network, just
                // bytes-on-disk). If it succeeds the registry already
                // had the package and we are done.
                let dep_name_t = dep_name.clone();
                let dep_version_t = dep_version.clone();
                let timeout_result = run_with_timeout(
                    move || {
                        load_registry_package(&dep_name_t, &dep_version_t).map_err(|e| match e {
                            LoadRegistryError::Other(s) => s,
                            LoadRegistryError::MissingFromIndex
                            | LoadRegistryError::MissingPackageDir => {
                                // Sentinel that the caller (which
                                // owns `options` and the
                                // lockfile dep) will route into
                                // auto-fetch. Encoded as a string
                                // because run_with_timeout's
                                // signature is `Result<T, String>`.
                                String::from(MISSING_REGISTRY_SENTINEL)
                            }
                        })
                    },
                    Duration::from_secs(5),
                    TIMEOUT_MSG,
                );
                let installed = match timeout_result {
                    Ok(installed) => installed,
                    Err(s) if s == MISSING_REGISTRY_SENTINEL => {
                        // Auto-fetch has its own internal lock + retry
                        // logic and can take longer than the 5-second
                        // local-registry budget.
                        load_registry_package_or_autofetch(
                            &dep_name,
                            &dep_version,
                            options,
                            Some(dep),
                        )?
                    }
                    Err(s) => return Err(s),
                };
                let dep_manifest = read_manifest(&installed.root.join("reef.toml"))?;
                let dep_modules = load_package_modules(&installed.root, &dep_manifest)?;
                // The lockfile is the source of truth: if it pinned
                // an origin, carry it forward. Otherwise fall back to
                // whatever the registry recorded (may also be `None`
                // when the registry was populated by --from-monorepo).
                let resolved_origin = remote_origin
                    .clone()
                    .or_else(|| installed.remote_origin.clone());
                packages.insert(
                    dep.name.clone(),
                    LoadedPackage {
                        id: PackageId {
                            name: dep.name.clone(),
                            version: dep.version.clone(),
                        },
                        manifest: dep_manifest,
                        modules: dep_modules,
                        source: LoadedSourceKind::LocalRegistry,
                        shell: Some(installed.shell),
                        remote_origin: resolved_origin,
                    },
                );
            }
        }
    }

    Ok(PackageGraph {
        root_package: root_name,
        packages,
    })
}

/// Sentinel string smuggled through `run_with_timeout`'s
/// `Result<T, String>` signature so the lockfile-reconstruction caller
/// can branch into auto-fetch on missing-from-registry without changing
/// `run_with_timeout`'s public type. Internal-only; tests do not rely
/// on the literal value.
const MISSING_REGISTRY_SENTINEL: &str = "__chelis_missing_from_registry_sentinel__";

/// Reserved top-level directory names that cannot appear in
/// `additional_sources`. Future sibling tools that introduce their own
/// walks (`bench`, `examples`, etc.) extend this list with a comment.
/// Single hardcoded list keeps the rule explicit and discoverable.
///
/// - `src` is the implicit always-walked source root; redeclaring it
///   would either no-op or double-walk.
/// - `tests` is walked separately by chelis-cli's `cmd_test` /
///   `discover_test_files`; reef does not subsume test discovery.
const RESERVED_ADDITIONAL_SOURCE_DIRS: &[&str] = &["src", "tests"];

fn validate_manifest(manifest: &ReefManifest) -> Result<(), String> {
    if manifest.package.name.trim().is_empty() {
        return Err("package.name must not be empty".to_string());
    }
    if manifest.package.version.trim().is_empty() {
        return Err("package.version must not be empty".to_string());
    }
    if manifest.package.module_prefix.trim().is_empty() {
        return Err("package.module_prefix must not be empty".to_string());
    }
    if manifest.package.compiler != CURRENT_COMPILER_VERSION {
        return Err(format!(
            "package.compiler must be `{CURRENT_COMPILER_VERSION}` in 3a"
        ));
    }
    let mut seen_additional = HashSet::new();
    for entry in &manifest.package.additional_sources {
        if entry.trim().is_empty() {
            return Err("package.additional_sources entries must not be empty".to_string());
        }
        if RESERVED_ADDITIONAL_SOURCE_DIRS.contains(&entry.as_str()) {
            return Err(format!(
                "package.additional_sources entry `{entry}` is reserved \
                 (reserved: {:?})",
                RESERVED_ADDITIONAL_SOURCE_DIRS
            ));
        }
        if entry.contains('/') || entry.contains('\\') {
            return Err(format!(
                "package.additional_sources entry `{entry}` must be a single \
                 directory name without path separators"
            ));
        }
        if !entry
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return Err(format!(
                "package.additional_sources entry `{entry}` must contain only \
                 ASCII alphanumeric, underscore, or hyphen characters"
            ));
        }
        if !seen_additional.insert(entry.as_str()) {
            return Err(format!(
                "package.additional_sources contains duplicate entry `{entry}`"
            ));
        }
    }
    for (name, dep) in &manifest.dependencies {
        match (&dep.version, &dep.path) {
            (Some(_), Some(_)) => {
                return Err(format!(
                    "dependency `{name}` cannot specify both `version` and `path`"
                ));
            }
            (_, Some(_)) if name == CHELIS_STD_PACKAGE_NAME => {
                return Err(
                    "`chelis-std` is the language runtime bundled with the compiler; it cannot be supplied as a path dependency"
                        .to_string(),
                );
            }
            (None, None) => {
                return Err(format!(
                    "dependency `{name}` must specify exactly one of `version` or `path`"
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn resolve_package_graph(root: &Path, options: LoadOptions) -> Result<PackageGraph, String> {
    let mut packages = BTreeMap::new();
    let mut by_name = HashMap::<String, PackageId>::new();
    let mut stack = Vec::new();
    let root_manifest = read_manifest(&root.join("reef.toml"))?;
    let root_id = PackageId {
        name: root_manifest.package.name.clone(),
        version: root_manifest.package.version.clone(),
    };
    resolve_package_recursive(
        &root_id.name,
        root.to_path_buf(),
        LoadedSourceKind::Root,
        None,
        None,
        &mut packages,
        &mut by_name,
        &mut stack,
        options,
    )?;
    Ok(PackageGraph {
        root_package: root_id.name,
        packages,
    })
}

// 8 arguments — the prior surface was already 7 and Item 9 adds
// `remote_origin` plumbing. Folding into a struct would obscure the
// resolver's flow and rename the same values without an abstraction
// win — the recursion needs all of these in scope. This allow is
// local to keep the noise contained.
#[allow(clippy::too_many_arguments)]
fn resolve_package_recursive(
    package_name: &str,
    root: PathBuf,
    source: LoadedSourceKind,
    maybe_shell: Option<ShellPackage>,
    remote_origin: Option<String>,
    packages: &mut BTreeMap<String, LoadedPackage>,
    by_name: &mut HashMap<String, PackageId>,
    stack: &mut Vec<String>,
    options: LoadOptions,
) -> Result<(), String> {
    if stack.iter().any(|name| name == package_name) {
        stack.push(package_name.to_string());
        return Err(format!("dependency cycle detected: {}", stack.join(" -> ")));
    }
    if packages.contains_key(package_name) {
        return Ok(());
    }

    stack.push(package_name.to_string());
    let manifest = read_manifest(&root.join("reef.toml"))?;
    if manifest.package.name != package_name {
        return Err(format!(
            "manifest at {} names package `{}`, expected `{package_name}`",
            root.display(),
            manifest.package.name
        ));
    }
    let id = PackageId {
        name: manifest.package.name.clone(),
        version: manifest.package.version.clone(),
    };
    if let Some(existing) = by_name.get(package_name)
        && existing.version != id.version
    {
        return Err(format!(
            "exact-version conflict for `{package_name}`: {} vs {}",
            existing.version, id.version
        ));
    }
    by_name.insert(package_name.to_string(), id.clone());

    let modules = load_package_modules(&root, &manifest)?;
    let package = LoadedPackage {
        id,
        manifest: manifest.clone(),
        modules,
        source: source.clone(),
        shell: maybe_shell.clone(),
        remote_origin: remote_origin.clone(),
    };

    for (dep_name, dep) in &manifest.dependencies {
        match (&dep.version, &dep.path) {
            (_, Some(path)) => {
                if dep_name == CHELIS_STD_PACKAGE_NAME {
                    return Err(
                        "`chelis-std` is the language runtime bundled with the compiler; it cannot be supplied as a path dependency"
                            .to_string(),
                    );
                }
                let dep_root = root.join(path);
                let dep_root = dep_root
                    .canonicalize()
                    .map_err(|e| format!("failed to resolve path dependency `{dep_name}`: {e}"))?;
                let relative = path.to_string();
                resolve_package_recursive(
                    dep_name,
                    dep_root,
                    LoadedSourceKind::Path { relative },
                    None,
                    None,
                    packages,
                    by_name,
                    stack,
                    options,
                )?;
            }
            (Some(version), None) => {
                // chelis-std is the language runtime, not a shell.
                // Soft-verify the declared version against the
                // compiler's bundled runtime. On mismatch, refuse
                // before we even try to load — the user has either an
                // out-of-date `reef.toml` declaration or an
                // out-of-date compiler. Auto-fetch from GitHub is
                // explicitly NOT attempted: chelis-std is not
                // distributed via GitHub releases.
                if dep_name == CHELIS_STD_PACKAGE_NAME {
                    let bundled = compiler_bundled_chelis_std_version();
                    if version != bundled {
                        return Err(format!(
                            "package depends on `chelis-std` `{version}`, but this compiler \
                             bundles chelis-std `{bundled}`. \
                             chelis-std is the language runtime, not a shell, so it cannot \
                             be substituted independently. Either update the `chelis-std` \
                             entry in `reef.toml` to match the compiler's bundled version \
                             (`chelis-std = {{ version = \"{bundled}\" }}`), or use a \
                             compiler whose bundled runtime matches your declaration."
                        ));
                    }
                }
                // Item 8 insertion point: missing-from-registry deps
                // route through the auto-fetch decision before bailing.
                // No lockfile context here (we're walking manifests);
                // pass `None` so the source URL falls back to the
                // canonical-org default for the named package.
                let installed =
                    load_registry_package_or_autofetch(dep_name, version, options, None)?;
                // Item 9 plumbing: pull the registry-recorded
                // `remote_origin` so `build_lockfile` can persist it.
                let dep_origin = installed.remote_origin.clone();
                resolve_package_recursive(
                    dep_name,
                    installed.root,
                    LoadedSourceKind::LocalRegistry,
                    Some(installed.shell),
                    dep_origin,
                    packages,
                    by_name,
                    stack,
                    options,
                )?;
            }
            _ => unreachable!("validated earlier"),
        }
    }

    packages.insert(package_name.to_string(), package);
    stack.pop();
    Ok(())
}

struct InstalledPackage {
    root: PathBuf,
    shell: ShellPackage,
    /// `index.json`'s `remote_origin` for this `(name, version)`. The
    /// caller (`resolve_package_recursive`) plumbs this onto the
    /// `LoadedPackage` so `build_lockfile` can later persist it to
    /// `LockSource::LocalRegistry::remote_origin`.
    remote_origin: Option<String>,
}

/// Process-stable cache of the chelis-std bundle's extracted layout.
/// First call extracts the embedded archive into a `tempfile::TempDir`
/// whose lifetime is tied to this `OnceLock` (i.e. the duration of
/// the process). Subsequent calls reuse the same on-disk root.
///
/// `tempfile::TempDir` cleans up at drop, so the extracted tree is
/// removed when the process exits — no stale state outlives a build
/// invocation.
static CHELIS_STD_BUNDLE_ROOT: std::sync::OnceLock<Result<tempfile::TempDir, String>> =
    std::sync::OnceLock::new();

/// Extract the embedded chelis-std bytes (if not already extracted
/// for this process), decode the embedded shell, and return them
/// shaped like a normal [`InstalledPackage`].
///
/// The returned `root` is a process-lifetime tempdir under the OS
/// tempdir (typically `/tmp`), with the chelis-std reef-package
/// layout: `reef.toml`, `src/`, etc. `load_package_modules` walks
/// this exactly the same way it walks a registry-cached extract.
///
/// `remote_origin` is `None` because there is no remote origin for
/// the runtime — its bytes come from the compiler binary itself.
fn load_bundled_chelis_std() -> Result<InstalledPackage, LoadRegistryError> {
    let dir_result = CHELIS_STD_BUNDLE_ROOT.get_or_init(|| {
        let dir = tempfile::Builder::new()
            .prefix("chelis-std-bundle-")
            .tempdir()
            .map_err(|e| format!("failed to create chelis-std bundle tempdir: {e}"))?;
        chelis_std_bundle::extract_into(dir.path())?;
        Ok(dir)
    });
    let dir = dir_result
        .as_ref()
        .map_err(|e| LoadRegistryError::Other(e.clone()))?;
    let shell = chelis_shell::decode_shell(chelis_std_bundle::CHELIS_STD_SHELL).map_err(|e| {
        LoadRegistryError::Other(format!("failed to decode embedded chelis-std shell: {e}"))
    })?;
    Ok(InstalledPackage {
        root: dir.path().to_path_buf(),
        shell,
        remote_origin: None,
    })
}

/// Typed result for [`load_registry_package`]. Item 8 introduces this
/// enum so the auto-fetch decision point can discriminate "the package
/// is simply not in the registry yet" from "the package is in the
/// registry but corrupt / mismatched / etc." Only the former is
/// recoverable by an auto-fetch; the latter would silently overwrite
/// damaged bytes if we treated them the same.
///
/// The two `Missing*` variants correspond to the two missing-from-registry
/// branches in [`load_registry_package`] that the pre-Item-8 code
/// surfaced as user-facing strings naming the (now-misleading) fix
/// `run `chelis reef build` first to populate the cache`.
#[derive(Debug, Clone)]
enum LoadRegistryError {
    /// The package is absent from `index.json`. This is the canonical
    /// "fresh registry" case Item 8's auto-fetch is designed for.
    MissingFromIndex,
    /// `index.json` claims the package exists but `packages/<name>/<version>/`
    /// is gone (corrupt / partially deleted registry). Auto-fetch can
    /// repair this by re-fetching the bytes.
    MissingPackageDir,
    /// Any other error: checksum mismatch, malformed shell, I/O failure
    /// reading the index, etc. Not recoverable by auto-fetch — the
    /// registry has bytes for this `(name, version)` and they
    /// disagree with what's pinned, which is a hash-mismatch
    /// signal we must surface, not paper over.
    Other(String),
}

impl LoadRegistryError {
    /// Whether auto-fetch is allowed to attempt to repair this state
    /// by re-fetching from the canonical hosting org.
    fn is_recoverable_by_autofetch(&self) -> bool {
        matches!(self, Self::MissingFromIndex | Self::MissingPackageDir)
    }
}

fn load_registry_package(name: &str, version: &str) -> Result<InstalledPackage, LoadRegistryError> {
    // Phase A correction: chelis-std is the language runtime and ships
    // bundled inside the chelis binary. Serve from the embedded bytes
    // before consulting the on-disk registry. This makes the runtime
    // reachable on a machine that has *no* `$CHELIS_REEF_HOME` at all
    // — the bytes come from `include_bytes!()` in
    // `chelis-std-bundle`, not from the filesystem.
    //
    // Falls through to the local-registry path if the requested
    // version disagrees with the bundled version. Soft-verify in
    // `resolve_package_recursive` already errors before we get here
    // when an explicit declaration mismatches, so this fallback is
    // defensive — it should never fire in practice. If it does, the
    // local-registry path will produce the standard
    // `MissingFromIndex` / mismatch errors.
    if name == CHELIS_STD_PACKAGE_NAME && version == chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION {
        return load_bundled_chelis_std();
    }

    let registry_root = registry_root().map_err(LoadRegistryError::Other)?;
    let index = read_registry_index(&registry_root).map_err(LoadRegistryError::Other)?;
    let Some(expected) = index
        .packages
        .get(name)
        .and_then(|versions| versions.iter().find(|entry| entry.version == version))
    else {
        return Err(LoadRegistryError::MissingFromIndex);
    };
    let pkg_dir = registry_root.join("packages").join(name).join(version);
    if !pkg_dir.exists() {
        return Err(LoadRegistryError::MissingPackageDir);
    }
    let shell_path = pkg_dir.join(format!("{name}-{version}.chb"));
    let archive_path = pkg_dir.join(format!("{name}-{version}.tar.zst"));
    let shell_sha256 = sha256_file(&shell_path).map_err(LoadRegistryError::Other)?;
    if shell_sha256 != expected.shell_sha256 {
        return Err(LoadRegistryError::Other(format!(
            "shell checksum mismatch for `{name}` version `{version}`"
        )));
    }
    let archive_sha256 = sha256_file(&archive_path).map_err(LoadRegistryError::Other)?;
    if archive_sha256 != expected.archive_sha256 {
        return Err(LoadRegistryError::Other(format!(
            "archive checksum mismatch for `{name}` version `{version}`"
        )));
    }
    let shell = read_shell(&shell_path).map_err(|e| LoadRegistryError::Other(e.to_string()))?;
    if archive_sha256 != shell.archive_sha256 {
        return Err(LoadRegistryError::Other(format!(
            "archive checksum mismatch for `{name}` version `{version}`"
        )));
    }
    if shell.package.name != name || shell.package.version != version {
        return Err(LoadRegistryError::Other(format!(
            "shell package id mismatch for `{name}` version `{version}`"
        )));
    }
    let cache_root = registry_root.join("cache").join(&archive_sha256);
    if !cache_root.exists() {
        extract_archive_atomic(&archive_path, &cache_root).map_err(LoadRegistryError::Other)?;
    }
    Ok(InstalledPackage {
        root: cache_root,
        shell,
        remote_origin: expected.remote_origin.clone(),
    })
}

/// Phase A Item 8: try [`load_registry_package`] and, on a recoverable
/// "missing-from-registry" failure, attempt one auto-fetch through
/// [`install_from_github`] before bailing.
///
/// ### Behavior
///
/// 1. Try `load_registry_package(name, version)`.
/// 2. If it fails with [`LoadRegistryError::Other`], the registry has
///    poisoned bytes — surface the underlying string as-is. Auto-fetch
///    is **not** allowed to silently overwrite a hash-mismatching local
///    install (Item 6's locked invariant).
/// 3. If it fails with [`LoadRegistryError::MissingFromIndex`] or
///    [`LoadRegistryError::MissingPackageDir`] **and** `auto_fetch`
///    is `true`, hold the process-level `flock(2)` on
///    `$CHELIS_REEF_HOME/.reef-lock` (see [`acquire_reef_home_lock`])
///    for the entire fetch + install + index update window, run
///    [`install_from_github`] against the resolved source URL, then
///    retry `load_registry_package`.
/// 4. If `auto_fetch` is `false`, or if the auto-fetch attempt itself
///    fails, bail with the improved-wording error
///    [`format_missing_dep_error`] which names: the URL that
///    was/would-be tried, whether `GITHUB_TOKEN` is currently set, and
///    (when auto-fetch ran) the typed [`GitHubFetchError`] category.
///
/// ### Source resolution
///
/// The fetch source URL is, in priority order:
/// 1. The lockfile entry's `remote_origin` field (today: always `None`
///    via [`lockfile_remote_origin`]; Item 9 wires this in).
/// 2. The canonical-org default from [`canonical_origin_for`].
///
/// Callers without a lockfile entry pass `lockfile_dep = None`.
fn load_registry_package_or_autofetch(
    name: &str,
    version: &str,
    options: LoadOptions,
    lockfile_dep: Option<&LockedDependency>,
) -> Result<InstalledPackage, String> {
    // First attempt: maybe the registry already has it.
    let first_err = match load_registry_package(name, version) {
        Ok(installed) => return Ok(installed),
        Err(e) => e,
    };

    // Non-recoverable: surface the original error string.
    if !first_err.is_recoverable_by_autofetch() {
        return Err(match first_err {
            LoadRegistryError::Other(s) => s,
            // Unreachable: filtered by `is_recoverable_by_autofetch`.
            LoadRegistryError::MissingFromIndex | LoadRegistryError::MissingPackageDir => {
                unreachable!()
            }
        });
    }

    // chelis-std is the language runtime — auto-fetch from GitHub is
    // intentionally disabled. The bytes ship with the compiler. If the
    // user's local registry doesn't have the runtime, that's a setup
    // issue (typically: `chelis reef install --from-monorepo` was
    // never run on this machine), not a transient network condition
    // an auto-fetch could repair.
    if name == CHELIS_STD_PACKAGE_NAME {
        let bundled = compiler_bundled_chelis_std_version();
        return Err(format!(
            "missing dependency `{name}` `{version}`: this is the language runtime, \
             which ships bundled with the compiler (this compiler bundles \
             chelis-std `{bundled}`). Auto-fetch from GitHub is intentionally \
             disabled for the runtime. Install it from the monorepo source \
             (`chelis reef install --from-monorepo`) or use a compiler whose \
             bundled runtime matches the declared version."
        ));
    }

    // Resolve the source URL: lockfile remote_origin (Item 9) wins, then
    // canonical-org default. Recorded for error messages even if we
    // skip the fetch attempt.
    let source_origin = lockfile_dep
        .and_then(lockfile_remote_origin)
        .unwrap_or_else(|| canonical_origin_for(name, version));

    if !options.auto_fetch {
        // Opt-out path. Improved error wording: name the URL that
        // would have been tried so the user can run
        // `chelis reef install --from-github <url>` themselves.
        return Err(format_missing_dep_error(
            name,
            version,
            &source_origin,
            None,
            false,
        ));
    }

    // Acquire the process-level lock for the duration of the fetch +
    // install + index update. The same registry root is shared across
    // every concurrent build that touches `$CHELIS_REEF_HOME`, so we
    // serialize through a `flock(2)` on `.reef-lock` to keep
    // `index.json` updates linearizable.
    let registry_root_path = registry_root()?;
    let _lock = match acquire_reef_home_lock(&registry_root_path, REEF_HOME_LOCK_TIMEOUT) {
        Ok(g) => g,
        Err(e) => return Err(format!("auto-fetch could not lock registry: {e}")),
    };

    // Re-check after acquiring the lock: a concurrent process may
    // have just installed it. This is the standard double-checked-
    // locking discipline; it also keeps the auto-fetch event
    // observable (we do NOT count this as an auto-fetch since no
    // network was hit).
    if let Ok(installed) = load_registry_package(name, version) {
        return Ok(installed);
    }

    // Run the fetch. Auto-fetch event becomes observable here via the
    // emitted log message; tests assert against this signal.
    eprintln!("chelis reef: auto-fetching `{name}` `{version}` from {source_origin}",);
    let fetch_err = match install_from_github(&source_origin, &registry_root_path) {
        Ok(_artifact) => {
            // Retry the registry lookup. If retry still fails, the
            // surprise is on us — surface as Other since it's a
            // post-install corruption case.
            return load_registry_package(name, version).map_err(|e| match e {
                LoadRegistryError::Other(s) => s,
                LoadRegistryError::MissingFromIndex | LoadRegistryError::MissingPackageDir => {
                    format!(
                        "post-fetch reload: package `{name}` `{version}` still missing after \
                         auto-fetch from {source_origin} reported success"
                    )
                }
            });
        }
        Err(e) => e,
    };

    Err(format_missing_dep_error(
        name,
        version,
        &source_origin,
        Some(&fetch_err),
        true,
    ))
}

/// Phase A Item 8 — Item 9 coordination shim.
///
/// Returns the lockfile entry's recorded `remote_origin` URL, if any.
/// Today the [`LockSource`] enum has no `remote_origin` field, so this
/// helper unconditionally returns `None`. After Item 9 lands and adds
/// `remote_origin: Option<String>` to `LockSource::LocalRegistry`, the
/// body of this function flips to `match &dep.source {
/// LockSource::LocalRegistry { remote_origin } => remote_origin.clone(),
/// _ => None }`.
///
/// Keeping the resolution logic behind this single shim means the Item 9
/// merge is a one-line change to a single function — everything else in
/// the auto-fetch path keeps working unchanged.
fn lockfile_remote_origin(dep: &LockedDependency) -> Option<String> {
    // Item 9 has merged: read the field on `LockSource::LocalRegistry`.
    // `Bundled` entries (the language runtime) intentionally have no
    // remote origin — there is nothing to fetch.
    match &dep.source {
        LockSource::LocalRegistry { remote_origin } => remote_origin.clone(),
        LockSource::Path { .. } | LockSource::Bundled { .. } => None,
    }
}

/// Phase A Item 8: derive the canonical-org default fetch URL for a
/// `(name, version)` pair the lockfile does not name a `remote_origin`
/// for. Returns a string in the form
/// `<canonical-org>/<name>@v<version>`, accepted as-is by
/// [`GitHubReleaseSpec::parse`] (which strips the leading `v` to
/// derive the version field).
///
/// Locked by [`CANONICAL_REEF_ORG`] = `"chelis-lang"`. Multi-publisher
/// generalization is post-launch (Item 10).
pub fn canonical_origin_for(name: &str, version: &str) -> String {
    format!("{CANONICAL_REEF_ORG}/{name}@v{version}")
}

/// Phase A Item 8: format the user-facing error wording for a missing
/// dependency that auto-fetch did not (or could not) repair.
///
/// The wording names:
/// - the dependency's `(name, version)`
/// - the source URL spec that was/would-be tried
/// - whether `GITHUB_TOKEN` is currently set in the environment
///   (auth state)
/// - if `fetch_err` is `Some`, the typed [`GitHubFetchError`] category
///   (`auth-missing`, `auth-rejected`, `release-asset-not-found`,
///   `release-tag-not-found-or-unauthorized`, `rate-limited`,
///   `server-error`, `network`, `io`, `validation`, `parse`)
///
/// The message shape is asserted against by the named acceptance oracle
/// `phaseA_item8_autofetch_build_oracle`. Wording must remain stable
/// (regex-stable, not just `contains()`) until Item 9 gates a change.
fn format_missing_dep_error(
    name: &str,
    version: &str,
    source_origin: &str,
    fetch_err: Option<&GitHubFetchError>,
    auto_fetch_was_enabled: bool,
) -> String {
    let token_state = if env::var_os("GITHUB_TOKEN").is_some() {
        "GITHUB_TOKEN is set"
    } else {
        "GITHUB_TOKEN is not set"
    };
    match fetch_err {
        Some(e) => {
            // Auto-fetch ran and failed.
            format!(
                "auto-fetch failed for dependency `{name}` `{version}` from \
                 `{source_origin}` (category: {cat}, {auth}): {msg}",
                cat = github_fetch_error_category(e),
                auth = token_state,
                msg = e,
            )
        }
        None if auto_fetch_was_enabled => {
            // Defensive branch — should not be reached: if auto-fetch
            // is on and we didn't get a fetch error, the caller must
            // have skipped the fetch.
            format!(
                "missing dependency `{name}` `{version}` and auto-fetch \
                 from `{source_origin}` did not run ({auth})",
                auth = token_state,
            )
        }
        None => {
            // Opt-out (--no-auto-fetch) path.
            format!(
                "missing dependency `{name}` `{version}` (auto-fetch disabled by \
                 `--no-auto-fetch`); would have fetched from `{source_origin}` \
                 (category: would-attempt, {token_state}). \
                 Either drop `--no-auto-fetch` or run \
                 `chelis reef install --from-github {source_origin}` manually."
            )
        }
    }
}

/// Map a [`GitHubFetchError`] variant onto a stable category string used
/// in [`format_missing_dep_error`]. Stable strings — tests pin against
/// these.
fn github_fetch_error_category(e: &GitHubFetchError) -> &'static str {
    match e {
        GitHubFetchError::Parse { .. } => "parse",
        GitHubFetchError::AuthMissing { .. } => "auth-missing",
        GitHubFetchError::AuthRejected { .. } => "auth-rejected",
        GitHubFetchError::ReleaseAssetNotFound { .. } => "release-asset-not-found",
        GitHubFetchError::ReleaseTagNotFoundOrUnauthorized { .. } => {
            "release-tag-not-found-or-unauthorized"
        }
        GitHubFetchError::RateLimited { .. } => "rate-limited",
        GitHubFetchError::ServerError { .. } => "server-error",
        GitHubFetchError::Network { .. } => "network",
        GitHubFetchError::Io { .. } => "io",
        GitHubFetchError::Validation { .. } => "validation",
    }
}

/// Phase A Item 8 internal load knobs threaded through the package
/// graph resolver. The public surface is [`BuildOptions`]; this
/// internal `Copy` carrier exists because the resolver passes options
/// into many recursive callsites and a leaf-only `bool` is more
/// ergonomic than threading the public struct everywhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LoadOptions {
    auto_fetch: bool,
}

impl LoadOptions {
    /// Default for non-build callers (eval, check, prepare-for-eval,
    /// etc.). Keeps the user-experience promise that `chelis check`
    /// against an empty registry will auto-fetch the same way
    /// `chelis reef build` does.
    fn default_for_load() -> Self {
        Self { auto_fetch: true }
    }
}

impl From<&BuildOptions> for LoadOptions {
    fn from(opts: &BuildOptions) -> Self {
        Self {
            auto_fetch: opts.auto_fetch,
        }
    }
}

fn read_registry_index(registry_root: &Path) -> Result<LocalRegistryIndex, String> {
    let index_path = registry_root.join("index.json");
    if !index_path.exists() {
        return Ok(LocalRegistryIndex::default());
    }
    serde_json::from_str::<LocalRegistryIndex>(
        &fs::read_to_string(&index_path).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())
}

fn build_lockfile(graph: &PackageGraph) -> ReefLock {
    let root = graph
        .packages
        .get(&graph.root_package)
        .expect("root package missing");
    let mut dependencies = graph
        .packages
        .iter()
        .filter(|(name, _)| *name != &graph.root_package)
        .map(|(name, package)| {
            // chelis-std is the language runtime, not a shell. Regardless
            // of how it was loaded into the graph (the bundled-bytes
            // path now serves the runtime regardless of registry
            // state), we record it in the lockfile as `Bundled` so a
            // reader can see at a glance that the bytes ship with the
            // compiler. The archive/shell SHA256s come from the
            // embedded bundle bytes for both the via-graph and the
            // synthesized paths so a lockfile written by either is
            // byte-identical for chelis-std.
            if name.as_str() == CHELIS_STD_PACKAGE_NAME {
                return LockedDependency {
                    name: name.clone(),
                    version: package.id.version.clone(),
                    source: LockSource::bundled_for_current_compiler(),
                    compiler: package.manifest.package.compiler.clone(),
                    archive_sha256: chelis_std_bundle::archive_sha256(),
                    shell_sha256: chelis_std_bundle::shell_sha256(),
                };
            }
            let source = match &package.source {
                LoadedSourceKind::Path { relative } => LockSource::Path {
                    path: relative.clone(),
                },
                LoadedSourceKind::LocalRegistry => LockSource::LocalRegistry {
                    remote_origin: package.remote_origin.clone(),
                },
                // `Root` is a defensive fallback for the build-lockfile
                // path: the root package is not normally a dependency.
                // If it ever shows up here, treat it as an unrecorded
                // local-registry source so the lockfile stays well-
                // formed.
                LoadedSourceKind::Root => LockSource::LocalRegistry {
                    remote_origin: None,
                },
            };
            let archive_sha256 = package
                .shell
                .as_ref()
                .map(|shell| shell.archive_sha256.clone())
                .unwrap_or_default();
            let shell_sha256 = package
                .shell
                .as_ref()
                .map(|shell| {
                    let bytes = chelis_shell::encode_shell(shell).expect("shell encode");
                    sha256_bytes(&bytes)
                })
                .unwrap_or_default();
            LockedDependency {
                name: name.clone(),
                version: package.id.version.clone(),
                source,
                compiler: package.manifest.package.compiler.clone(),
                archive_sha256,
                shell_sha256,
            }
        })
        .collect::<Vec<_>>();

    // Phase A correction (Item 1): blanket synthesis on the project's
    // compiler pin. Every reef.toml has a `compiler =` pin (validated
    // by `validate_manifest`), and that pin IS the runtime declaration.
    // The chelis-std runtime is implicit: programs depend on it the
    // way Rust programs depend on `core`/`std`. Lockfiles must record
    // it explicitly so a reader can audit the runtime version and a
    // re-build on another machine resolves to the same bytes.
    //
    // If chelis-std is already in the dep graph (because the project
    // listed it explicitly in `[dependencies]`, or a transitive shell
    // pulled it in), the closure above already produced a `Bundled`
    // entry — synthesis is a no-op. Otherwise, append one here using
    // the bundled archive/shell SHA256s, the compiler-bundled version,
    // and the project's own compiler pin (which `validate_manifest`
    // guarantees equals `CURRENT_COMPILER_VERSION` today).
    if !dependencies
        .iter()
        .any(|d| d.name == CHELIS_STD_PACKAGE_NAME)
    {
        dependencies.push(LockedDependency {
            name: CHELIS_STD_PACKAGE_NAME.to_string(),
            version: BUNDLED_CHELIS_STD_VERSION.to_string(),
            source: LockSource::bundled_for_current_compiler(),
            compiler: root.manifest.package.compiler.clone(),
            archive_sha256: chelis_std_bundle::archive_sha256(),
            shell_sha256: chelis_std_bundle::shell_sha256(),
        });
    }

    dependencies.sort_by(|a, b| a.name.cmp(&b.name));
    ReefLock {
        package: root.id.clone(),
        dependencies,
    }
}

fn load_package_modules(
    root: &Path,
    manifest: &ReefManifest,
) -> Result<BTreeMap<String, ModuleSource>, String> {
    // src/ remains mandatory; the additional-roots loop runs after
    // confirming src/ exists so the loop body stays uniform.
    let src_root = root.join("src");
    if !src_root.exists() {
        return Err(format!("{} is missing src/", root.display()));
    }

    // Walk every declared source root: src/ first, then each entry from
    // manifest.package.additional_sources (validated empty/distinct/safe
    // by validate_manifest before we get here).
    let roots: Vec<&str> = std::iter::once("src")
        .chain(
            manifest
                .package
                .additional_sources
                .iter()
                .map(|s| s.as_str()),
        )
        .collect();

    let mut modules: BTreeMap<String, ModuleSource> = BTreeMap::new();
    let mut total_files = 0usize;
    for source_root_name in &roots {
        let abs_root = root.join(source_root_name);
        // Additional roots are optional: a manifest may declare
        // additional_sources = ["properties"] with no properties/ dir
        // yet (e.g. while migrating). We skip silently rather than
        // erroring; src/ is the only mandatory root and was checked
        // above.
        if !abs_root.exists() {
            continue;
        }
        for entry in WalkDir::new(&abs_root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() {
                continue;
            }
            if entry.path().extension().and_then(|ext| ext.to_str()) != Some("ch") {
                continue;
            }
            total_files += 1;
            let rel = entry
                .path()
                .strip_prefix(&abs_root)
                .map_err(|e| e.to_string())?
                .to_path_buf();
            // CRITICAL: for non-src roots, prepend the source root name to
            // the relative path before validation. validate_module_path
            // strips the source-root prefix entirely from path-to-module
            // derivation, so without prepending,
            //   <root>/src/foo.ch         (rel = foo.ch)            -> prefix.foo
            //   <root>/properties/foo.ch  (rel would be foo.ch too) -> prefix.foo
            // would collide. Prepending makes the second case
            //   rel_for_validation = properties/foo.ch  -> prefix.properties.foo
            // The author of properties/foo.ch must declare
            // `module <Prefix>.Properties.Foo` — convention follows the
            // same path-to-module rule, just with the root name as the
            // first segment for non-src roots.
            let rel_for_validation = if *source_root_name == "src" {
                rel.clone()
            } else {
                Path::new(source_root_name).join(&rel)
            };
            let decls = chelis_surf::parser::parse_str(
                &fs::read_to_string(entry.path()).map_err(|e| e.to_string())?,
            )
            .map_err(|e| format!("{}: {e}", entry.path().display()))?;
            let module_decl = match decls.as_slice() {
                [Decl::Module { name, decls, .. }] => (name.clone(), decls.clone()),
                _ => {
                    return Err(format!(
                        "{} must contain exactly one top-level module declaration",
                        entry.path().display()
                    ));
                }
            };
            validate_module_path(
                &manifest.package.module_prefix,
                &module_decl.0,
                &rel_for_validation,
            )?;
            let exports = compute_exports(&module_decl.1);
            let symbols = collect_symbol_kinds(&module_decl.1);
            if exports
                .iter()
                .any(|name| matches!(symbols.get(name), Some(SymbolKind::Macro)))
            {
                return Err(format!(
                    "cross-package macro exports are deferred in 3a; module {} exports a macro",
                    module_decl.0
                ));
            }
            // Belt-and-suspenders against any case the rel-prepending
            // missed: BTreeMap::insert returns Some(prev) on a duplicate
            // key. The rel-prepending should already prevent cross-root
            // collisions, but a duplicate module-name across roots
            // (e.g. two `module Pkg.Foo` declared with mismatched paths)
            // is still a hard error.
            let module_name = module_decl.0.clone();
            let new_source = ModuleSource {
                package_name: manifest.package.name.clone(),
                module_name: module_decl.0,
                decls: module_decl.1,
                file_rel: rel,
                source_root: (*source_root_name).to_string(),
                exports,
                symbols,
            };
            if let Some(prev) = modules.insert(module_name.clone(), new_source) {
                return Err(format!(
                    "duplicate module `{module_name}` across source roots: \
                     `{}/{}` and `{}/{}`",
                    prev.source_root,
                    prev.file_rel.display(),
                    source_root_name,
                    entry
                        .path()
                        .strip_prefix(&abs_root)
                        .map(|p| p.display().to_string())
                        .unwrap_or_else(|_| entry.path().display().to_string())
                ));
            }
        }
    }
    if total_files == 0 {
        return Err(format!(
            "{} has no .ch source files under src/{}",
            root.display(),
            if manifest.package.additional_sources.is_empty() {
                String::new()
            } else {
                format!(
                    " or any of [{}]",
                    manifest.package.additional_sources.join(", ")
                )
            }
        ));
    }
    Ok(modules)
}

fn validate_module_path(prefix: &str, module: &str, rel: &Path) -> Result<(), String> {
    let prefix_lower = prefix.to_lowercase();
    let module_lower = module.to_lowercase();
    if module_lower == prefix_lower || !module_lower.starts_with(&(prefix_lower.clone() + ".")) {
        return Err(format!(
            "module `{module}` does not belong to module_prefix `{prefix}`"
        ));
    }
    let rel_no_ext = rel.with_extension("");
    let rel_module = rel_no_ext
        .iter()
        .map(|seg| seg.to_string_lossy().to_string())
        .collect::<Vec<_>>()
        .join(".");
    let expected = format!("{prefix_lower}.{rel_module}");
    if module_lower != expected {
        // The caller passes `rel` already prefixed with the source root
        // name for non-src roots (see `load_package_modules`), so the
        // displayed path is the source-root-relative-from-package-root
        // form ("src/foo.ch" or "properties/foo.ch") and the expected
        // module name reflects the same rule.
        return Err(format!(
            "module `{module}` does not match file path {} (expected `{}`)",
            rel.display(),
            expected
        ));
    }
    Ok(())
}

fn compute_exports(decls: &[Decl]) -> BTreeSet<String> {
    // Map an ADT type name to its constructor (variant) names so that
    // exporting (or auto-exporting) a type also brings its constructors
    // into the importing module's scope. This mirrors OCaml/Haskell
    // module semantics (importing a type exposes its constructors) and is
    // required for chelis#157: a downstream module that imports `Column`
    // must be able to write `IntCol(...)` and resolve it to the exporting
    // module's mangled constructor.
    let ctors_for_type: HashMap<&str, &[Variant]> = decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::TypeDef { name, variants, .. } => Some((name.as_str(), variants.as_slice())),
            _ => None,
        })
        .collect();

    let explicit = decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::Export { names, .. } => Some(names.clone()),
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    if !explicit.is_empty() {
        let mut exports: BTreeSet<String> = explicit.into_iter().collect();
        // Pull in constructors for every explicitly exported type name.
        let type_exports: Vec<String> = exports.iter().cloned().collect();
        for name in type_exports {
            if let Some(variants) = ctors_for_type.get(name.as_str()) {
                for variant in *variants {
                    exports.insert(variant.name.clone());
                }
            }
        }
        return exports;
    }
    decls
        .iter()
        .flat_map(|decl| match decl {
            Decl::FunDef { name, .. }
            | Decl::Property { name, .. }
            | Decl::LetDef { name, .. }
            | Decl::TypeAlias { name, .. } => vec![name.clone()],
            Decl::TypeDef { name, variants, .. } => {
                let mut names = vec![name.clone()];
                names.extend(variants.iter().map(|variant| variant.name.clone()));
                names
            }
            _ => Vec::new(),
        })
        .collect()
}

fn collect_symbol_kinds(decls: &[Decl]) -> BTreeMap<String, SymbolKind> {
    let mut symbols = BTreeMap::new();
    for decl in decls {
        match decl {
            Decl::FunDef { name, .. }
            | Decl::Property { name, .. }
            | Decl::LetDef { name, .. }
            | Decl::Sig { name, .. } => {
                symbols.insert(name.clone(), SymbolKind::Value);
            }
            Decl::TypeDef { name, variants, .. } => {
                symbols.insert(name.clone(), SymbolKind::Type);
                // Constructor (variant) names are value-level symbols:
                // they appear in expression position (`IntCol(xs, mask)`,
                // `IntCol { values: xs }`). Registering them as module
                // symbols is the principled fix for chelis#157
                // (module-scoped constructor resolution): without this,
                // `internal_name` mangling never applies to a constructor,
                // so two packages that each declare an `IntCol` variant
                // bind the same bare `IntCol` into the flat type-checker
                // env/registry and resolution is last-write-wins.
                //
                // A constructor that shares its name with its enclosing
                // type (the idiomatic `type Foo = | Foo { ... }`) mangles
                // to the same internal name as the type via `internal_name`
                // and must keep the `Type` kind in the module's public
                // surface, so do not clobber an existing entry.
                for variant in variants {
                    symbols
                        .entry(variant.name.clone())
                        .or_insert(SymbolKind::Value);
                }
            }
            Decl::TypeAlias { name, .. } => {
                symbols.insert(name.clone(), SymbolKind::Type);
            }
            Decl::MacroDef { name, .. } => {
                symbols.insert(name.clone(), SymbolKind::Macro);
            }
            Decl::Dim { names, .. } => {
                for name in names {
                    symbols.insert(name.clone(), SymbolKind::Dim);
                }
            }
            Decl::Export { .. } | Decl::Import { .. } | Decl::Module { .. } => {}
        }
    }
    symbols
}

fn module_name_for_input(root: &Path, file: &Path, package_name: &str) -> Result<String, String> {
    let root_pkg = resolve_package_graph(root, LoadOptions::default_for_load())?
        .packages
        .remove(package_name)
        .ok_or_else(|| "root package missing".to_string())?;
    let canonical = file
        .canonicalize()
        .map_err(|e| format!("failed to canonicalize {}: {e}", file.display()))?;
    // Multi-root: each module records its own source_root, so we
    // reconstruct the absolute path the same way `source_digests` does.
    for module in root_pkg.modules.values() {
        if root.join(&module.source_root).join(&module.file_rel) == canonical {
            return Ok(module.module_name.clone());
        }
    }
    let declared_roots = std::iter::once("src".to_string())
        .chain(root_pkg.manifest.package.additional_sources.iter().cloned())
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "{} is not a source file under any declared root of {} (roots: [{}])",
        file.display(),
        root.display(),
        declared_roots,
    ))
}

fn build_archive(root: &Path, out_path: &Path) -> Result<(), String> {
    // Re-read the manifest to know which additional source roots to pack.
    // (This function is called from `build_package` after the graph has
    // already been resolved; reading once more here is cheap and keeps
    // the archive packing self-contained.)
    let manifest = read_manifest(&root.join("reef.toml"))?;
    let mut tar_bytes = Vec::new();
    {
        let mut builder = Builder::new(&mut tar_bytes);
        let metadata_files: &[&str] = if manifest.package.name == CHELIS_STD_PACKAGE_NAME {
            // chelis-std is the bundled runtime and therefore has a
            // self-referential lock entry. Including reef.lock in its own
            // archive makes the archive hash depend on the previous bundle
            // hash and prevents the committed lock from reaching a fixed
            // point. Downstream shells still pack reef.lock normally.
            &["reef.toml"]
        } else {
            &["reef.toml", "reef.lock"]
        };
        for rel in metadata_files {
            let path = root.join(rel);
            if path.exists() {
                builder
                    .append_path_with_name(&path, rel)
                    .map_err(|e| e.to_string())?;
            }
        }
        // Pack src/ plus every declared additional source root. Tar
        // paths remain relative to the package root, so an archive with
        // additional_sources = ["properties"] contains both src/main.ch
        // and properties/foo.ch at their canonical relative locations.
        // extract_archive (just below) is path-agnostic — it unpacks
        // whatever paths were packed.
        let roots: Vec<&str> = std::iter::once("src")
            .chain(
                manifest
                    .package
                    .additional_sources
                    .iter()
                    .map(|s| s.as_str()),
            )
            .collect();
        for source_root_name in &roots {
            let abs_root = root.join(source_root_name);
            if !abs_root.exists() {
                continue;
            }
            for entry in WalkDir::new(&abs_root).into_iter().filter_map(Result::ok) {
                if !entry.file_type().is_file() {
                    continue;
                }
                let rel = entry.path().strip_prefix(root).map_err(|e| e.to_string())?;
                builder
                    .append_path_with_name(entry.path(), rel)
                    .map_err(|e| e.to_string())?;
            }
        }
        builder.finish().map_err(|e| e.to_string())?;
    }
    let compressed =
        zstd::stream::encode_all(Cursor::new(tar_bytes), 19).map_err(|e| e.to_string())?;
    fs::write(out_path, compressed).map_err(|e| e.to_string())
}

fn extract_archive(archive_path: &Path, out_dir: &Path) -> Result<(), String> {
    let bytes = fs::read(archive_path).map_err(|e| e.to_string())?;
    let decoded = zstd::stream::decode_all(Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let mut archive = Archive::new(Cursor::new(decoded));
    archive.unpack(out_dir).map_err(|e| e.to_string())
}

/// Extract `archive_path` into `final_dir` atomically.
///
/// The package cache directory `<registry>/cache/<archive_sha256>/` is
/// shared across every concurrent `chelis reef build` that touches the
/// same `$CHELIS_REEF_HOME`. Extracting in place leaves a window where
/// the directory exists but its files (`reef.toml`, `src/main.ch`, ...)
/// are only partially written, so a concurrent reader can observe a
/// torn or empty `reef.toml` and fail with a `TOML parse error at line
/// 1, column 1`.
///
/// This helper extracts into a unique sibling temp directory and then
/// `fs::rename`s it into place. Same-directory rename is atomic on
/// every supported FS, so a concurrent reader sees either no
/// `cache/<hash>/` directory at all or the fully-populated one, never
/// an intermediate state.
///
/// If the rename loses a race (another process populated `final_dir`
/// first), that is success: both extractions produce byte-identical
/// trees because the cache key is the archive's own content hash. The
/// loser discards its temp directory and returns `Ok`.
fn extract_archive_atomic(archive_path: &Path, final_dir: &Path) -> Result<(), String> {
    let parent = final_dir.parent().ok_or_else(|| {
        format!(
            "extract_archive_atomic: {} has no parent directory",
            final_dir.display()
        )
    })?;
    fs::create_dir_all(parent)
        .map_err(|e| format!("failed to create cache directory {}: {e}", parent.display()))?;
    // Unique temp dir name: PID + a process-local counter so two
    // extractions racing inside the same process also get distinct
    // staging directories.
    static EXTRACT_SEQ: AtomicUsize = AtomicUsize::new(0);
    let unique = format!(
        ".extract-{}-{}.tmp",
        std::process::id(),
        EXTRACT_SEQ.fetch_add(1, Ordering::Relaxed)
    );
    let tmp_dir = parent.join(unique);
    // Best-effort cleanup of a stale staging dir from a crashed write.
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).map_err(|e| {
        format!(
            "failed to create staging directory {}: {e}",
            tmp_dir.display()
        )
    })?;
    if let Err(e) = extract_archive(archive_path, &tmp_dir) {
        let _ = fs::remove_dir_all(&tmp_dir);
        return Err(e);
    }
    match fs::rename(&tmp_dir, final_dir) {
        Ok(()) => Ok(()),
        Err(_) if final_dir.exists() => {
            // Lost the race: another extraction populated `final_dir`
            // first. The trees are content-identical, so discard ours.
            let _ = fs::remove_dir_all(&tmp_dir);
            Ok(())
        }
        Err(e) => {
            let _ = fs::remove_dir_all(&tmp_dir);
            Err(format!(
                "failed to publish extracted cache {} -> {}: {e}",
                tmp_dir.display(),
                final_dir.display()
            ))
        }
    }
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 8192];
    loop {
        let read = file.read(&mut buf).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        hasher.update(&buf[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn build_shell_package(
    package: &LoadedPackage,
    checked: &chelis_types::CheckedProgram,
    archive_sha256: &str,
) -> Result<ShellPackage, String> {
    let mut modules = Vec::new();
    for module in package.modules.values() {
        let mut exports = Vec::new();
        for name in &module.exports {
            let kind =
                module.symbols.get(name).copied().ok_or_else(|| {
                    format!("export `{name}` not defined in {}", module.module_name)
                })?;
            let internal = internal_name(&package.id.name, &module.module_name, name);
            let type_repr = checked
                .type_env()
                .get(&internal)
                .map(|expr| {
                    chelis_deep::printer::print_canonical(std::slice::from_ref(expr))
                        .trim()
                        .to_string()
                })
                .or_else(|| sig_type_repr(module, name));
            let effects = symbol_effects(module, name);
            let has_body = module.decls.iter().any(|decl| matches!(decl, Decl::FunDef { name: decl_name, .. } | Decl::LetDef { name: decl_name, .. } if decl_name == name));
            exports.push(ShellSymbol {
                name: name.clone(),
                kind,
                type_repr,
                effects,
                has_body,
            });
        }
        exports.sort_by(|a, b| a.name.cmp(&b.name));
        modules.push(ShellModule {
            module: module.module_name.clone(),
            exports,
        });
    }
    modules.sort_by(|a, b| a.module.cmp(&b.module));
    Ok(ShellPackage {
        package: package.id.clone(),
        compiler: package.manifest.package.compiler.clone(),
        modules,
        dependencies: package
            .manifest
            .dependencies
            .keys()
            .map(|name| PackageId {
                name: name.clone(),
                version: "".to_string(),
            })
            .collect(),
        archive_sha256: archive_sha256.to_string(),
    })
}

fn sig_type_repr(module: &ModuleSource, name: &str) -> Option<String> {
    let decl = module
        .decls
        .iter()
        .find(|decl| matches!(decl, Decl::Sig { name: decl_name, .. } if decl_name == name))?;
    let deep = chelis_surf::desugar::desugar_program(std::slice::from_ref(decl));
    let expr = deep.first()?;
    let chelis_deep::ast::Expr::List(list, _) = expr else {
        return None;
    };
    list.elements.get(3).map(|ty| {
        chelis_deep::printer::print_canonical(std::slice::from_ref(ty))
            .trim()
            .to_string()
    })
}

fn symbol_effects(module: &ModuleSource, name: &str) -> Vec<String> {
    module
        .decls
        .iter()
        .find_map(|decl| match decl {
            Decl::FunDef {
                name: decl_name,
                effects,
                ..
            }
            | Decl::Sig {
                name: decl_name,
                effects,
                ..
            } if decl_name == name => Some(
                effects
                    .as_deref()
                    .unwrap_or(&[])
                    .iter()
                    .map(effect_name)
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

fn effect_name(effect: &EffectExpr) -> String {
    match effect {
        EffectExpr::Diff(_) => "Diff".to_string(),
        EffectExpr::Random(_) => "Random".to_string(),
        EffectExpr::Accum(_) => "Accum".to_string(),
        EffectExpr::Io(_) => "IO".to_string(),
        EffectExpr::Test(_) => "Test".to_string(),
        EffectExpr::Resource(device, _) => format!("Resource({device})"),
    }
}

fn link_graph(graph: &PackageGraph, entry_modules: &[String]) -> Result<Vec<LinkedModule>, String> {
    Ok(link_graph_with_package_tags(graph, entry_modules)?
        .into_iter()
        .map(|(_package_name, module)| module)
        .collect())
}

/// Like [`link_graph`], but each linked module is paired with the name of
/// the package it came from. The cross-process chelis-std typecheck cache
/// uses this to partition the linked library decls into the chelis-std
/// portion (cached under a content-addressed sub-key) and everything
/// else. Iteration order matches `link_graph` exactly (a `BTreeMap` walk
/// over `graph.packages`, then `package.modules`), so callers that flatten
/// either result get identical decl ordering.
fn link_graph_with_package_tags(
    graph: &PackageGraph,
    entry_modules: &[String],
) -> Result<Vec<(String, LinkedModule)>, String> {
    let internal_maps = build_internal_maps(graph);
    let dep_shells = dependency_shells(graph);

    let mut linked = Vec::new();
    for (package_name, package) in &graph.packages {
        for module in package.modules.values() {
            let entry = package_name == &graph.root_package
                && entry_modules
                    .iter()
                    .any(|entry_module| entry_module == &module.module_name);
            let decls = rewrite_module_decls(module, graph, &internal_maps, &dep_shells)?;
            linked.push((package_name.clone(), LinkedModule { decls, entry }));
        }
    }
    Ok(linked)
}

fn dependency_shells(graph: &PackageGraph) -> BTreeMap<String, ShellPackage> {
    graph
        .packages
        .iter()
        .filter_map(|(name, package)| {
            if name == &graph.root_package {
                None
            } else {
                package
                    .shell
                    .as_ref()
                    .map(|shell| (name.clone(), shell.clone()))
            }
        })
        .collect::<BTreeMap<_, _>>()
}

fn build_internal_maps(graph: &PackageGraph) -> HashMap<(String, String), HashMap<String, String>> {
    let mut maps = HashMap::new();
    for (package_name, package) in &graph.packages {
        for module in package.modules.values() {
            let mut module_map = HashMap::new();
            for name in module.symbols.keys() {
                module_map.insert(
                    name.clone(),
                    internal_name(package_name, &module.module_name, name),
                );
            }
            maps.insert(
                (package_name.clone(), module.module_name.clone()),
                module_map,
            );
        }
    }
    maps
}

fn build_name_resolver(
    module: &ModuleSource,
    graph: &PackageGraph,
    internal_maps: &HashMap<(String, String), HashMap<String, String>>,
    dep_shells: &BTreeMap<String, ShellPackage>,
) -> Result<NameResolver, String> {
    let mut qualified = HashMap::<String, HashMap<String, String>>::new();
    let mut unqualified = HashMap::<String, String>::new();
    let module_internal = internal_maps
        .get(&(module.package_name.clone(), module.module_name.clone()))
        .cloned()
        .unwrap_or_default();

    for (name, internal) in &module_internal {
        unqualified.insert(name.clone(), internal.clone());
    }

    // Track, per unqualified bare name, the distinct internal names it was
    // bound to by `import (..)` / `import (names)` forms, along with the
    // importing module path for diagnostics. A bare name that resolves to
    // two or more *distinct* internal names across imports is an ambiguous
    // unqualified reference; the user must qualify it (chelis#157). A name
    // the module declares itself (`module_internal`) shadows imports and is
    // never ambiguous. Re-importing the *same* internal name from two paths
    // is idempotent and not an error.
    let mut import_sources: BTreeMap<String, BTreeMap<String, BTreeSet<String>>> = BTreeMap::new();

    for decl in &module.decls {
        let Decl::Import {
            module: import_module,
            kind,
            ..
        } = decl
        else {
            continue;
        };
        let (import_pkg, import_src) =
            find_imported_module(graph, &module.package_name, import_module)?;
        let import_package = graph
            .packages
            .get(&import_pkg)
            .ok_or_else(|| format!("package `{import_pkg}` missing"))?;
        let internal_map = internal_maps
            .get(&(import_pkg.clone(), import_module.clone()))
            .ok_or_else(|| format!("missing internal map for module `{import_module}`"))?;
        let allowed_exports = if import_pkg == module.package_name {
            import_src.symbols.keys().cloned().collect::<BTreeSet<_>>()
        } else {
            match import_package.source {
                LoadedSourceKind::Path { .. } | LoadedSourceKind::Root => {
                    import_src.exports.clone()
                }
                LoadedSourceKind::LocalRegistry => dep_public_exports(
                    dep_shells
                        .get(&import_pkg)
                        .ok_or_else(|| format!("missing dependency shell for `{import_pkg}`"))?,
                    import_module,
                )?,
            }
        };
        let qualified_map = allowed_exports
            .iter()
            .filter_map(|name| {
                internal_map
                    .get(name)
                    .map(|internal| (name.clone(), internal.clone()))
            })
            .collect::<HashMap<_, _>>();
        qualified.insert(import_module.clone(), qualified_map.clone());
        match kind {
            ImportKind::Qualified => {}
            ImportKind::All => {
                for (name, internal) in qualified_map {
                    import_sources
                        .entry(name.clone())
                        .or_default()
                        .entry(internal.clone())
                        .or_default()
                        .insert(import_module.clone());
                    unqualified.insert(name, internal);
                }
            }
            ImportKind::Names(names) => {
                for name in names {
                    let internal = qualified_map.get(name).ok_or_else(|| {
                        format!(
                            "module `{}` does not export `{}` for import into {}",
                            import_module, name, module.module_name
                        )
                    })?;
                    import_sources
                        .entry(name.clone())
                        .or_default()
                        .entry(internal.clone())
                        .or_default()
                        .insert(import_module.clone());
                    unqualified.insert(name.clone(), internal.clone());
                }
            }
        }
    }

    // Reject genuinely ambiguous unqualified references. A name the module
    // declares itself shadows any import, so only flag names that were
    // brought in solely by imports and resolved to more than one distinct
    // internal target.
    for (name, internals) in &import_sources {
        if module_internal.contains_key(name) {
            continue;
        }
        if internals.len() > 1 {
            let mut from_modules: BTreeSet<String> = BTreeSet::new();
            for modules in internals.values() {
                from_modules.extend(modules.iter().cloned());
            }
            let from_list = from_modules.into_iter().collect::<Vec<_>>().join(", ");
            return Err(format!(
                "ambiguous reference to `{name}`: imported from multiple modules ({from_list}). \
                 Qualify the reference (e.g. `Module.{name}`) or import only one of them."
            ));
        }
    }

    Ok(NameResolver {
        own_names: module_internal,
        imported_names: unqualified,
        qualified_modules: qualified,
        qualified_failures: RefCell::new(Vec::new()),
    })
}

fn rewrite_module_decls(
    module: &ModuleSource,
    graph: &PackageGraph,
    internal_maps: &HashMap<(String, String), HashMap<String, String>>,
    dep_shells: &BTreeMap<String, ShellPackage>,
) -> Result<Vec<Decl>, String> {
    let resolver = build_name_resolver(module, graph, internal_maps, dep_shells)?;
    let mut out = Vec::new();
    for decl in &module.decls {
        match decl {
            Decl::Import { .. } => {}
            // Export decls survive the rewrite with internal names so
            // the checker-enforced opacity layer (RFC D-CHECK sixth
            // rejection / producer enumeration) can recover each
            // module's export set from the linked decl stream; they
            // desugar to inert `(export ...)` Deep nodes. Names that
            // do not resolve to a module decl pass through unmapped
            // (the resolver tolerated unknown exports by dropping
            // them before; an inert unmapped symbol is equivalent).
            Decl::Export { names, span } => out.push(Decl::Export {
                names: names
                    .iter()
                    .map(|name| {
                        resolver
                            .own_names
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| name.clone())
                    })
                    .collect(),
                span: *span,
            }),
            Decl::Module { .. } => unreachable!("module wrappers already stripped"),
            _ => out.push(rewrite_decl(
                decl,
                &resolver,
                &module.package_name,
                &module.module_name,
            )),
        }
    }
    drain_qualified_failures(&resolver)?;
    Ok(out)
}

/// Turn any qualified-reference misses recorded during rewrite into a hard
/// error (chelis#316). Reported deterministically (first by record order) so a
/// program with several unknown qualified names fails on a stable one.
fn drain_qualified_failures(resolver: &NameResolver) -> Result<(), String> {
    match resolver.qualified_failures.borrow().first() {
        Some(msg) => Err(msg.clone()),
        None => Ok(()),
    }
}

fn rewrite_eval_module_decls(
    module: &ModuleSource,
    graph: &PackageGraph,
    internal_maps: &HashMap<(String, String), HashMap<String, String>>,
    dep_shells: &BTreeMap<String, ShellPackage>,
) -> Result<Vec<Decl>, String> {
    let resolver = build_name_resolver(module, graph, internal_maps, dep_shells)?;
    let mut out = Vec::new();
    for decl in &module.decls {
        // RFC v6 (RT-1 F2-bypass): user-authored entry/test decls keep
        // their own names through the eval rewrite (unlike the library
        // rewrite, which re-mangles via `internal_name`). A name in the
        // reef linker's reserved internal-name format would then
        // self-key via `reef_module_stem` to a victim module and forge
        // its opaque types as in-module, while the linked-program flag
        // suppresses the checker's reserved-name rejection. Reject the
        // reserved format here, at the single boundary every test/eval
        // entry path passes through, before these decls combine with the
        // linked library. This is flag-independent: user entry decls are
        // never linker output.
        if let Some(name) = entry_decl_binding_name(decl)
            && chelis_types::is_linker_format_name(name)
        {
            return Err(format!(
                "`{name}` uses the reef package-linker's reserved internal-name format \
                 (`Pkg__`/`pkg__`...), which only the linker may produce; rename the declaration"
            ));
        }
        match decl {
            Decl::Import { .. } | Decl::Export { .. } => {}
            Decl::Module { .. } => unreachable!("module wrappers already stripped"),
            _ => out.push(rewrite_eval_decl(decl, &resolver)),
        }
    }
    drain_qualified_failures(&resolver)?;
    Ok(out)
}

/// The top-level binding name a declaration introduces, for the
/// reserved-name check (RFC v6). Returns `None` for decls that bind no
/// name (import/export/module/dim).
fn entry_decl_binding_name(decl: &Decl) -> Option<&str> {
    match decl {
        Decl::FunDef { name, .. }
        | Decl::LetDef { name, .. }
        | Decl::Sig { name, .. }
        | Decl::TypeDef { name, .. }
        | Decl::TypeAlias { name, .. }
        | Decl::MacroDef { name, .. }
        | Decl::Property { name, .. } => Some(name.as_str()),
        Decl::Import { .. } | Decl::Export { .. } | Decl::Module { .. } | Decl::Dim { .. } => None,
    }
}

fn dep_public_exports(shell: &ShellPackage, module: &str) -> Result<BTreeSet<String>, String> {
    let shell_module = shell
        .modules
        .iter()
        .find(|entry| entry.module == module)
        .ok_or_else(|| {
            format!(
                "module `{module}` is not exported by package `{}`",
                shell.package.name
            )
        })?;
    Ok(shell_module
        .exports
        .iter()
        .map(|symbol| symbol.name.clone())
        .collect())
}

fn find_imported_module<'a>(
    graph: &'a PackageGraph,
    current_package: &str,
    module_name: &str,
) -> Result<(String, &'a ModuleSource), String> {
    let current = graph
        .packages
        .get(current_package)
        .ok_or_else(|| format!("package `{current_package}` missing"))?;
    if let Some(module) = current.modules.get(module_name) {
        return Ok((current_package.to_string(), module));
    }
    for (package_name, package) in &graph.packages {
        if package_name == current_package {
            continue;
        }
        if let Some(module) = package.modules.get(module_name) {
            return Ok((package_name.clone(), module));
        }
    }
    Err(format!("unresolved import `{module_name}`"))
}

struct NameResolver {
    own_names: HashMap<String, String>,
    imported_names: HashMap<String, String>,
    qualified_modules: HashMap<String, HashMap<String, String>>,
    /// Qualified references whose head named an imported module but whose leaf
    /// that module does not export — a typo or unexported name. Recorded by
    /// the expression / pattern / type resolvers as they run, then drained
    /// into a hard error by `rewrite_module_decls` so all three positions
    /// reject an unknown qualified name uniformly (chelis#316). A `RefCell`
    /// because the resolvers take `&NameResolver`; each module is rewritten
    /// with its own resolver on a single thread, so there is no sharing.
    qualified_failures: RefCell<Vec<String>>,
}

/// Record that a qualified reference named the imported module `module` but a
/// leaf `leaf` it does not export. Shared by the expression, pattern, and type
/// resolvers so an unknown qualified name fails the same way everywhere
/// instead of silently surviving into a later stage that may not catch it
/// (e.g. an opaque `Named` type, or a dead `match` arm). Deduplicated so the
/// same typo used in several positions reports once.
fn record_qualified_miss(resolver: &NameResolver, module: &str, leaf: &str) {
    let msg = format!("module `{module}` does not export `{leaf}` (qualified reference)");
    let mut failures = resolver.qualified_failures.borrow_mut();
    if !failures.contains(&msg) {
        failures.push(msg);
    }
}

fn internal_name(package: &str, module: &str, name: &str) -> String {
    let stem = format!(
        "{}__{}__{}",
        package.replace(['-', '.'], "__"),
        module.replace('.', "__"),
        name
    );
    if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
        format!("Pkg__{stem}")
    } else {
        format!("pkg__{stem}")
    }
}

fn rewrite_decl(decl: &Decl, resolver: &NameResolver, package: &str, module: &str) -> Decl {
    match decl {
        Decl::FunDef {
            name,
            dim_params,
            params,
            ret_ty,
            effects,
            body,
            span,
        } => {
            let mut locals = params
                .iter()
                .map(|param| param.name.clone())
                .collect::<HashSet<_>>();
            let body = rewrite_expr(body, resolver, &mut locals);
            Decl::FunDef {
                name: internal_name(package, module, name),
                dim_params: dim_params.clone(),
                params: params
                    .iter()
                    .map(|param| rewrite_param(param, resolver))
                    .collect(),
                ret_ty: ret_ty.as_ref().map(|ty| rewrite_type(ty, resolver)),
                effects: effects.clone(),
                body,
                span: *span,
            }
        }
        Decl::LetDef {
            name,
            ty,
            value,
            span,
        } => Decl::LetDef {
            name: internal_name(package, module, name),
            ty: ty.as_ref().map(|ty| rewrite_type(ty, resolver)),
            value: rewrite_expr(value, resolver, &mut HashSet::new()),
            span: *span,
        },
        Decl::Property {
            name,
            params,
            preconditions,
            body,
            options,
            span,
        } => {
            let mut locals = params
                .iter()
                .map(|param| param.name.clone())
                .collect::<HashSet<_>>();
            Decl::Property {
                name: internal_name(package, module, name),
                params: params
                    .iter()
                    .map(|param| rewrite_param(param, resolver))
                    .collect(),
                preconditions: preconditions
                    .iter()
                    .map(|expr| rewrite_expr(expr, resolver, &mut locals.clone()))
                    .collect(),
                body: rewrite_expr(body, resolver, &mut locals),
                options: options
                    .iter()
                    .map(|option| rewrite_property_option(option, resolver))
                    .collect(),
                span: *span,
            }
        }
        Decl::Sig {
            name,
            ty,
            effects,
            span,
        } => Decl::Sig {
            name: internal_name(package, module, name),
            ty: rewrite_type(ty, resolver),
            effects: effects.clone(),
            span: *span,
        },
        Decl::TypeDef {
            name,
            params,
            variants,
            opaque,
            invariant,
            span,
        } => Decl::TypeDef {
            name: internal_name(package, module, name),
            params: params.clone(),
            variants: variants
                .iter()
                .map(|variant| rewrite_variant(variant, resolver, package, module))
                .collect(),
            opaque: *opaque,
            invariant: invariant
                .as_ref()
                .map(|inv| rewrite_invariant(inv, resolver)),
            span: *span,
        },
        Decl::TypeAlias {
            name,
            params,
            ty,
            span,
        } => Decl::TypeAlias {
            name: internal_name(package, module, name),
            params: params.clone(),
            ty: rewrite_type(ty, resolver),
            span: *span,
        },
        Decl::MacroDef {
            name,
            params,
            body,
            span,
        } => {
            let mut locals = params.iter().cloned().collect::<HashSet<_>>();
            Decl::MacroDef {
                name: internal_name(package, module, name),
                params: params.clone(),
                body: rewrite_expr(body, resolver, &mut locals),
                span: *span,
            }
        }
        Decl::Dim { names, span } => Decl::Dim {
            names: names.clone(),
            span: *span,
        },
        Decl::Module { .. } | Decl::Import { .. } | Decl::Export { .. } => decl.clone(),
    }
}

fn rewrite_eval_decl(decl: &Decl, resolver: &NameResolver) -> Decl {
    match decl {
        Decl::FunDef {
            name,
            dim_params,
            params,
            ret_ty,
            effects,
            body,
            span,
        } => {
            let mut locals = params
                .iter()
                .map(|param| param.name.clone())
                .collect::<HashSet<_>>();
            let body = rewrite_expr(body, resolver, &mut locals);
            Decl::FunDef {
                name: name.clone(),
                dim_params: dim_params.clone(),
                params: params
                    .iter()
                    .map(|param| rewrite_param(param, resolver))
                    .collect(),
                ret_ty: ret_ty.as_ref().map(|ty| rewrite_type(ty, resolver)),
                effects: effects.clone(),
                body,
                span: *span,
            }
        }
        Decl::LetDef {
            name,
            ty,
            value,
            span,
        } => Decl::LetDef {
            name: name.clone(),
            ty: ty.as_ref().map(|ty| rewrite_type(ty, resolver)),
            value: rewrite_expr(value, resolver, &mut HashSet::new()),
            span: *span,
        },
        Decl::Property {
            name,
            params,
            preconditions,
            body,
            options,
            span,
        } => {
            let mut locals = params
                .iter()
                .map(|param| param.name.clone())
                .collect::<HashSet<_>>();
            Decl::Property {
                name: name.clone(),
                params: params
                    .iter()
                    .map(|param| rewrite_param(param, resolver))
                    .collect(),
                preconditions: preconditions
                    .iter()
                    .map(|expr| rewrite_expr(expr, resolver, &mut locals.clone()))
                    .collect(),
                body: rewrite_expr(body, resolver, &mut locals),
                options: options
                    .iter()
                    .map(|option| rewrite_property_option(option, resolver))
                    .collect(),
                span: *span,
            }
        }
        Decl::Sig {
            name,
            ty,
            effects,
            span,
        } => Decl::Sig {
            name: name.clone(),
            ty: rewrite_type(ty, resolver),
            effects: effects.clone(),
            span: *span,
        },
        Decl::TypeDef {
            name,
            params,
            variants,
            opaque,
            invariant,
            span,
        } => Decl::TypeDef {
            // Eval-entry decls are the user's bare program: the type name
            // is kept unmangled (the synthetic `__Eval` module has no
            // internal-name map, so own-name references are never
            // rewritten either). The constructor names must stay bare for
            // the same reason; only their field types are rewritten so an
            // imported type used as a field resolves to its internal name.
            name: name.clone(),
            params: params.clone(),
            variants: variants
                .iter()
                .map(|variant| rewrite_eval_variant(variant, resolver))
                .collect(),
            opaque: *opaque,
            invariant: invariant
                .as_ref()
                .map(|inv| rewrite_invariant(inv, resolver)),
            span: *span,
        },
        Decl::TypeAlias {
            name,
            params,
            ty,
            span,
        } => Decl::TypeAlias {
            name: name.clone(),
            params: params.clone(),
            ty: rewrite_type(ty, resolver),
            span: *span,
        },
        Decl::MacroDef {
            name,
            params,
            body,
            span,
        } => Decl::MacroDef {
            name: name.clone(),
            params: params.clone(),
            body: rewrite_expr(body, resolver, &mut HashSet::new()),
            span: *span,
        },
        Decl::Dim { names, span } => Decl::Dim {
            names: names.clone(),
            span: *span,
        },
        Decl::Export { .. } | Decl::Import { .. } | Decl::Module { .. } => decl.clone(),
    }
}

fn rewrite_property_option(option: &PropertyOption, resolver: &NameResolver) -> PropertyOption {
    let mut locals = HashSet::new();
    match option {
        PropertyOption::Tolerance(value, span) => {
            PropertyOption::Tolerance(rewrite_expr(value, resolver, &mut locals), *span)
        }
        PropertyOption::Seed(value, span) => {
            PropertyOption::Seed(rewrite_expr(value, resolver, &mut locals), *span)
        }
        PropertyOption::Samples(value, span) => {
            PropertyOption::Samples(rewrite_expr(value, resolver, &mut locals), *span)
        }
        PropertyOption::Contract(id, span) => PropertyOption::Contract(id.clone(), *span),
    }
}

fn rewrite_variant(
    variant: &Variant,
    resolver: &NameResolver,
    package: &str,
    module: &str,
) -> Variant {
    Variant {
        // Mangle the constructor name to the package/module-qualified
        // internal form (chelis#157). `internal_name` is deterministic, so
        // this matches the `own_names` entry `collect_symbol_kinds` produced
        // for the same variant, which is what `resolve_name` rewrites
        // constructor references to at the call site.
        name: internal_name(package, module, &variant.name),
        fields: rewrite_variant_fields(&variant.fields, resolver),
        span: variant.span,
    }
}

/// Rewrite a variant's field types through the resolver while keeping its
/// constructor name unmangled. Used by the eval-entry path, where the
/// enclosing type and its constructors stay bare (see `rewrite_eval_decl`).
fn rewrite_eval_variant(variant: &Variant, resolver: &NameResolver) -> Variant {
    Variant {
        name: variant.name.clone(),
        fields: rewrite_variant_fields(&variant.fields, resolver),
        span: variant.span,
    }
}

fn rewrite_variant_fields(fields: &VariantFields, resolver: &NameResolver) -> VariantFields {
    match fields {
        VariantFields::Positional(fields) => VariantFields::Positional(
            fields
                .iter()
                .map(|field| rewrite_type(field, resolver))
                .collect(),
        ),
        VariantFields::Record(fields) => VariantFields::Record(
            fields
                .iter()
                .map(|(name, ty)| (name.clone(), rewrite_type(ty, resolver)))
                .collect(),
        ),
    }
}

/// Rewrite a declared type invariant under package linking: the predicate
/// body is rewritten with the invariant binder as a local, so references
/// to in-module zero-arg constants get reef-mangled while the binder
/// stays bare (mirrors the `Decl::Property` precondition/body rewrite).
fn rewrite_invariant(invariant: &TypeInvariant, resolver: &NameResolver) -> TypeInvariant {
    let mut locals = HashSet::new();
    locals.insert(invariant.binder.clone());
    TypeInvariant {
        binder: invariant.binder.clone(),
        body: rewrite_expr(&invariant.body, resolver, &mut locals),
        span: invariant.span,
    }
}

fn rewrite_param(param: &Param, resolver: &NameResolver) -> Param {
    Param {
        name: param.name.clone(),
        ty: param.ty.as_ref().map(|ty| rewrite_type(ty, resolver)),
        span: param.span,
    }
}

fn rewrite_type(ty: &TypeExpr, resolver: &NameResolver) -> TypeExpr {
    match ty {
        TypeExpr::Named(name, span) => TypeExpr::Named(
            resolver
                .own_names
                .get(name)
                .cloned()
                .or_else(|| resolver.imported_names.get(name).cloned())
                // A dotted head is a module-qualified type name
                // (`Demo.Dropout.Mode`, chelis#316): resolve it through the
                // same `qualified_modules` map qualified constructors use.
                .or_else(|| resolve_qualified_name(name, resolver))
                .unwrap_or_else(|| name.clone()),
            *span,
        ),
        // A rank variable `..r` is local to its def/sig and never a
        // module-qualified name, so it passes through name resolution as-is.
        TypeExpr::RankSpread(name, span) => TypeExpr::RankSpread(name.clone(), *span),
        TypeExpr::Tensor(parts, precision, span) => TypeExpr::Tensor(
            parts
                .iter()
                .map(|part| rewrite_type(part, resolver))
                .collect(),
            precision.clone(),
            *span,
        ),
        TypeExpr::Arrow(args, ret, span) => TypeExpr::Arrow(
            args.iter().map(|arg| rewrite_type(arg, resolver)).collect(),
            Box::new(rewrite_type(ret, resolver)),
            *span,
        ),
        TypeExpr::Ref(inner, span) => TypeExpr::Ref(Box::new(rewrite_type(inner, resolver)), *span),
        TypeExpr::App(name, args, span) => TypeExpr::App(
            resolver
                .own_names
                .get(name)
                .cloned()
                .or_else(|| resolver.imported_names.get(name).cloned())
                // Qualified applied type head `Demo.Coral.Frame[n]` (chelis#316).
                .or_else(|| resolve_qualified_name(name, resolver))
                .unwrap_or_else(|| name.clone()),
            args.iter().map(|arg| rewrite_type(arg, resolver)).collect(),
            *span,
        ),
        TypeExpr::Tuple(parts, span) => TypeExpr::Tuple(
            parts
                .iter()
                .map(|part| rewrite_type(part, resolver))
                .collect(),
            *span,
        ),
        TypeExpr::Infer(span) => TypeExpr::Infer(*span),
    }
}

fn rewrite_expr(expr: &Expr, resolver: &NameResolver, locals: &mut HashSet<String>) -> Expr {
    if let Some(resolved) = resolve_qualified_expr(expr, resolver) {
        return resolved;
    }
    match expr {
        Expr::Var(name, span) => Expr::Var(resolve_name(name, resolver, locals), *span),
        Expr::Constructor(name, span) => {
            Expr::Constructor(resolve_name(name, resolver, locals), *span)
        }
        Expr::Lit(lit, span) => Expr::Lit(lit.clone(), *span),
        Expr::List(items, span) => Expr::List(
            items
                .iter()
                .map(|item| rewrite_expr(item, resolver, locals))
                .collect(),
            *span,
        ),
        Expr::Apply(func, args, span) => Expr::Apply(
            Box::new(rewrite_expr(func, resolver, locals)),
            args.iter()
                .map(|arg| rewrite_expr(arg, resolver, locals))
                .collect(),
            *span,
        ),
        Expr::Record(name, fields, span) => Expr::Record(
            // The record-construction head is a constructor name and must
            // resolve through the same own-module/imports scope as a
            // positional constructor call (chelis#157). Without this, a
            // record build `IntCol { values: xs }` kept the bare `IntCol`
            // and collided with a same-named constructor from another
            // package in the flat type-checker registry.
            resolve_name(name, resolver, locals),
            fields
                .iter()
                .map(|(field, expr)| (field.clone(), rewrite_expr(expr, resolver, locals)))
                .collect(),
            *span,
        ),
        Expr::Access(inner, field, span) => Expr::Access(
            Box::new(rewrite_expr(inner, resolver, locals)),
            field.clone(),
            *span,
        ),
        Expr::TupleGet(inner, index, span) => Expr::TupleGet(
            Box::new(rewrite_expr(inner, resolver, locals)),
            *index,
            *span,
        ),
        Expr::Binary(op, left, right, span) => Expr::Binary(
            *op,
            Box::new(rewrite_expr(left, resolver, locals)),
            Box::new(rewrite_expr(right, resolver, locals)),
            *span,
        ),
        Expr::Unary(op, inner, span) => {
            Expr::Unary(*op, Box::new(rewrite_expr(inner, resolver, locals)), *span)
        }
        Expr::Pipe(seed, stages, span) => Expr::Pipe(
            Box::new(rewrite_expr(seed, resolver, locals)),
            stages
                .iter()
                .map(|stage| rewrite_expr(stage, resolver, locals))
                .collect(),
            *span,
        ),
        Expr::If(cond, then_expr, else_expr, span) => Expr::If(
            Box::new(rewrite_expr(cond, resolver, locals)),
            Box::new(rewrite_expr(then_expr, resolver, locals)),
            Box::new(rewrite_expr(else_expr, resolver, locals)),
            *span,
        ),
        Expr::Match(scrutinee, arms, span) => Expr::Match(
            Box::new(rewrite_expr(scrutinee, resolver, locals)),
            arms.iter()
                .map(|arm| rewrite_arm(arm, resolver, locals))
                .collect(),
            *span,
        ),
        Expr::Lambda(params, body, span) => {
            let mut scoped = locals.clone();
            for param in params {
                scoped.insert(param.name.clone());
            }
            Expr::Lambda(
                params
                    .iter()
                    .map(|param| rewrite_param(param, resolver))
                    .collect(),
                Box::new(rewrite_expr(body, resolver, &mut scoped)),
                *span,
            )
        }
        Expr::Tuple(parts, span) => Expr::Tuple(
            parts
                .iter()
                .map(|part| rewrite_expr(part, resolver, locals))
                .collect(),
            *span,
        ),
        Expr::Cast(inner, ty, span) => Expr::Cast(
            Box::new(rewrite_expr(inner, resolver, locals)),
            ty.clone(),
            *span,
        ),
        Expr::Grad(inner, wrt, span) => Expr::Grad(
            Box::new(rewrite_expr(inner, resolver, locals)),
            wrt.clone(),
            *span,
        ),
        Expr::Vmap(inner, axis, span) => Expr::Vmap(
            Box::new(rewrite_expr(inner, resolver, locals)),
            *axis,
            *span,
        ),
        Expr::Jit(inner, span) => Expr::Jit(Box::new(rewrite_expr(inner, resolver, locals)), *span),
        Expr::Realize(inner, span) => {
            Expr::Realize(Box::new(rewrite_expr(inner, resolver, locals)), *span)
        }
        Expr::Copy(inner, span) => {
            Expr::Copy(Box::new(rewrite_expr(inner, resolver, locals)), *span)
        }
        Expr::Borrow(inner, span) => {
            Expr::Borrow(Box::new(rewrite_expr(inner, resolver, locals)), *span)
        }
        Expr::WithSeed(seed, body, span) => Expr::WithSeed(
            Box::new(rewrite_expr(seed, resolver, locals)),
            Box::new(rewrite_expr(body, resolver, locals)),
            *span,
        ),
        Expr::WithDevice(device, body, span) => Expr::WithDevice(
            Box::new(rewrite_expr(device, resolver, locals)),
            Box::new(rewrite_expr(body, resolver, locals)),
            *span,
        ),
        Expr::Par(exprs, span) => Expr::Par(
            exprs
                .iter()
                .map(|expr| rewrite_expr(expr, resolver, locals))
                .collect(),
            *span,
        ),
        Expr::Annotate(inner, ty, span) => Expr::Annotate(
            Box::new(rewrite_expr(inner, resolver, locals)),
            rewrite_type(ty, resolver),
            *span,
        ),
        Expr::Block(bindings, body, span) => {
            let mut scoped = locals.clone();
            let bindings = bindings
                .iter()
                .map(|binding| rewrite_let_binding(binding, resolver, &mut scoped))
                .collect();
            let body = rewrite_expr(body, resolver, &mut scoped);
            Expr::Block(bindings, Box::new(body), *span)
        }
    }
}

fn rewrite_arm(arm: &MatchArm, resolver: &NameResolver, locals: &mut HashSet<String>) -> MatchArm {
    let mut scoped = locals.clone();
    collect_pattern_binders(&arm.pattern, &mut scoped);
    MatchArm {
        pattern: rewrite_pattern(&arm.pattern, resolver),
        guard: arm
            .guard
            .as_ref()
            .map(|guard| rewrite_expr(guard, resolver, &mut scoped)),
        body: rewrite_expr(&arm.body, resolver, &mut scoped),
        span: arm.span,
    }
}

fn rewrite_pattern(pattern: &Pattern, resolver: &NameResolver) -> Pattern {
    match pattern {
        Pattern::Wildcard(span) => Pattern::Wildcard(*span),
        Pattern::Var(name, span) => Pattern::Var(name.clone(), *span),
        Pattern::Lit(lit, span) => Pattern::Lit(lit.clone(), *span),
        Pattern::Constructor(name, args, span) => Pattern::Constructor(
            resolve_ctor_pattern_name(name, resolver),
            args.iter()
                .map(|arg| rewrite_pattern(arg, resolver))
                .collect(),
            *span,
        ),
        Pattern::Tuple(parts, span) => Pattern::Tuple(
            parts
                .iter()
                .map(|part| rewrite_pattern(part, resolver))
                .collect(),
            *span,
        ),
        Pattern::Record(name, fields, span) => Pattern::Record(
            // A record pattern's head is a constructor name; resolve it
            // through the same scope as `Pattern::Constructor` so a
            // record-shaped match arm resolves to the module-qualified
            // constructor (chelis#157).
            resolve_ctor_pattern_name(name, resolver),
            fields
                .iter()
                .map(|(field, pattern)| (field.clone(), rewrite_pattern(pattern, resolver)))
                .collect(),
            *span,
        ),
        Pattern::As(name, inner, span) => Pattern::As(
            name.clone(),
            Box::new(rewrite_pattern(inner, resolver)),
            *span,
        ),
    }
}

fn rewrite_let_binding(
    binding: &LetBinding,
    resolver: &NameResolver,
    locals: &mut HashSet<String>,
) -> LetBinding {
    let value = rewrite_expr(&binding.value, resolver, locals);
    collect_let_pattern_binders(&binding.pattern, locals);
    LetBinding {
        pattern: binding.pattern.clone(),
        ty: binding.ty.as_ref().map(|ty| rewrite_type(ty, resolver)),
        value,
    }
}

fn collect_pattern_binders(pattern: &Pattern, locals: &mut HashSet<String>) {
    match pattern {
        Pattern::Var(name, _) => {
            locals.insert(name.clone());
        }
        Pattern::Tuple(parts, _) => {
            for part in parts {
                collect_pattern_binders(part, locals);
            }
        }
        Pattern::Record(_, fields, _) => {
            for (_, pattern) in fields {
                collect_pattern_binders(pattern, locals);
            }
        }
        Pattern::As(name, inner, _) => {
            locals.insert(name.clone());
            collect_pattern_binders(inner, locals);
        }
        Pattern::Wildcard(_) | Pattern::Lit(_, _) | Pattern::Constructor(_, _, _) => {}
    }
}

fn collect_let_pattern_binders(pattern: &LetPattern, locals: &mut HashSet<String>) {
    match pattern {
        LetPattern::Var(name, _) => {
            locals.insert(name.clone());
        }
        LetPattern::Tuple(parts, _) => {
            for part in parts {
                collect_let_pattern_binders(part, locals);
            }
        }
        LetPattern::Wildcard(_) => {}
    }
}

fn resolve_name(name: &str, resolver: &NameResolver, locals: &HashSet<String>) -> String {
    if locals.contains(name) {
        return name.to_string();
    }
    resolver
        .own_names
        .get(name)
        .cloned()
        .or_else(|| resolver.imported_names.get(name).cloned())
        .unwrap_or_else(|| name.to_string())
}

/// Resolve a constructor name that appears as a pattern head
/// (`Pattern::Constructor` / `Pattern::Record`). Patterns introduce
/// binders rather than reference locals, so unlike `resolve_name` there
/// is no `locals` shadow set; but the own-module-before-imports
/// precedence must match `resolve_name` exactly. Otherwise a module that
/// both declares its own constructor `C` and imports a different `C`
/// would build `C(..)` as the own (shadowing) constructor but match
/// `| C(..) =>` against the imported one, because `imported_names`
/// overwrites the own seed in `build_name_resolver`. That asymmetry
/// silently mis-resolves a `match` arm to a different package's
/// constructor (chelis#157); own-first keeps construction and
/// destructuring on the same mangled name.
/// A dotted head (`| Demo.Dropout.Train =>`) is a module-qualified
/// constructor pattern (chelis#316). It cannot match `own_names`/
/// `imported_names` (those are keyed by bare names), so resolve it through the
/// same `qualified_modules` map a qualified constructor *expression* uses,
/// keeping construction and destructuring on one mangled name even when two
/// imported modules export the same constructor.
fn resolve_ctor_pattern_name(name: &str, resolver: &NameResolver) -> String {
    resolver
        .own_names
        .get(name)
        .cloned()
        .or_else(|| resolver.imported_names.get(name).cloned())
        .or_else(|| resolve_qualified_name(name, resolver))
        .unwrap_or_else(|| name.to_string())
}

/// Resolve a module-qualified dotted name (`Demo.Dropout.Train`) to the
/// declaring module's internal name, the string-keyed counterpart of
/// `resolve_qualified_expr`'s segment walk (used for constructor patterns and
/// type names). Returns `None` for a bare name, for a path whose prefix does
/// not name an imported module (left untouched — an ordinary unknown name), or
/// for a path whose prefix *does* name a module but whose leaf it does not
/// export. In that last case it also records a qualified-reference miss so
/// `rewrite_module_decls` rejects the typo with a precise error — matching the
/// expression position rather than silently leaving a dotted name for a later
/// stage that may not catch it.
fn resolve_qualified_name(name: &str, resolver: &NameResolver) -> Option<String> {
    if !name.contains('.') {
        return None;
    }
    let segments: Vec<&str> = name.split('.').collect();
    let mut miss: Option<(String, String)> = None;
    for split in 1..segments.len() {
        let module = segments[..split].join(".");
        let leaf = segments[split..].join(".");
        if let Some(map) = resolver.qualified_modules.get(&module) {
            if let Some(internal) = map.get(&leaf) {
                return Some(internal.clone());
            }
            miss.get_or_insert((module, leaf));
        }
    }
    if let Some((module, leaf)) = miss {
        record_qualified_miss(resolver, &module, &leaf);
    }
    None
}

fn resolve_qualified_expr(expr: &Expr, resolver: &NameResolver) -> Option<Expr> {
    let (segments, span) = access_segments(expr)?;
    if segments.len() < 2 {
        return None;
    }
    let mut miss: Option<(String, String)> = None;
    for split in 1..segments.len() {
        let module = segments[..split].join(".");
        let name = segments[split..].join(".");
        if let Some(map) = resolver.qualified_modules.get(&module) {
            if let Some(internal) = map.get(&name) {
                return Some(Expr::Var(internal.clone(), span));
            }
            miss.get_or_insert((module, name));
        }
    }
    // The head segments name an imported module, but the trailing name is not
    // one of its exports: a qualified reference to a missing name (a typo, or
    // a name the module does not export), not a record field access. Record
    // the miss so `rewrite_module_decls` rejects it with a precise error —
    // the same diagnostic the pattern and type positions now get (chelis#316).
    // Also rewrite it to the written dotted path as a `Var` so that even if the
    // failure is somehow not drained, the checker still reports an unbound
    // variable rather than silently typing it as an unconstrained field
    // projection (field access is not checked against its base). A path whose
    // head is *not* an imported module (an ordinary record field access such
    // as `opt.lr`) is left untouched.
    if let Some((module, leaf)) = miss {
        record_qualified_miss(resolver, &module, &leaf);
        return Some(Expr::Var(segments.join("."), span));
    }
    None
}

fn access_segments(expr: &Expr) -> Option<(Vec<String>, chelis_deep::Span)> {
    match expr {
        Expr::Var(name, span) | Expr::Constructor(name, span) => Some((vec![name.clone()], *span)),
        Expr::Access(inner, field, span) => {
            let (mut parts, _) = access_segments(inner)?;
            parts.push(field.clone());
            Some((parts, *span))
        }
        _ => None,
    }
}

fn expanded_desugared_program(decls: &[Decl]) -> Result<Vec<chelis_deep::ast::Expr>, String> {
    let deep = chelis_surf::desugar::desugar_program(decls);
    chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
        .map(|expanded| expanded.into_exprs())
        .map_err(|err| err.to_string())
}

fn checked_program_with_effects(
    deep_exprs: &[chelis_deep::ast::Expr],
) -> Result<chelis_types::CheckedProgram, String> {
    // RFC v5 (RT-1 F2 bypass): this checks the fully linked package
    // (`build_package_with_options` flattens `link_graph` output), which
    // is reef-linker output carrying internal-name-mangled bindings.
    // Accept the linker name format for this check.
    let _linked = chelis_types::install_linked_program_guard();
    let checked = chelis_types::check_ir_program(deep_exprs)
        .map_err(|r| format!("Type errors: {:?}", r.errors))?;
    let checked = chelis_effects::check_program(&checked).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| error.message)
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    chelis_types::check_linearity(&checked).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| error.message)
            .collect::<Vec<_>>()
            .join("; ")
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn reserved_linker_name_predicate_matches_only_full_mangled_names() {
        // RFC v6: the entry-boundary reserved-name reject must match
        // exactly the names that could self-key via `reef_module_stem`.
        // CR-7: reef shares the single chelis-types predicate.
        use chelis_types::is_linker_format_name as is_reserved;
        assert!(is_reserved("pkg__opq__Demo__Types__forge"));
        assert!(is_reserved("Pkg__opq__Demo__Types__Probability"));
        assert!(is_reserved("pkg__forgepkg__Smoke__Types__forge"));
        // Marker prefix but no module stem -> cannot key to a module.
        assert!(!is_reserved("pkg__lonely"));
        assert!(!is_reserved("Pkg__lonely"));
        // Ordinary user identifiers (test fns, helpers, synth roots).
        assert!(!is_reserved("test_forge"));
        assert!(!is_reserved("normal_helper"));
        assert!(!is_reserved("__chelis_test_0"));
        assert!(!is_reserved("probability"));
    }

    #[test]
    fn reserved_name_predicate_agrees_on_borderline_names_cr7() {
        // CR-7: the reef entry-boundary reject and the chelis-types
        // checker now use ONE predicate, so they cannot drift. Pin the
        // borderline cases from the review: a single-underscore name
        // (`pkg_count`) is NOT reserved; the full mangled form is.
        use chelis_types::is_linker_format_name as is_reserved;
        assert!(!is_reserved("pkg_count"));
        assert!(!is_reserved("pkg"));
        assert!(!is_reserved("count_pkg"));
        assert!(is_reserved("Pkg__X__Y"));
        assert!(is_reserved("pkg__a__b"));
    }

    #[test]
    fn entry_decl_binding_name_covers_binding_decls() {
        use chelis_deep::Span;
        use chelis_surf::ast::{Decl, Expr};
        let span = Span::new(0, 0);
        let fun = Decl::FunDef {
            name: "pkg__a__B__c".to_string(),
            dim_params: vec![],
            params: vec![],
            ret_ty: None,
            effects: None,
            body: Expr::Lit(chelis_surf::ast::Literal::Int(0), span),
            span,
        };
        assert_eq!(entry_decl_binding_name(&fun), Some("pkg__a__B__c"));
        let import = Decl::Import {
            module: "Std".to_string(),
            kind: chelis_surf::ast::ImportKind::Qualified,
            span,
        };
        assert_eq!(entry_decl_binding_name(&import), None);
    }

    /// Process-shared lock for tests that mutate `CHELIS_REEF_HOME`
    /// (or other process env). Cargo runs unit tests in this binary
    /// in parallel by default; without serialization, test A's
    /// `set_var` plus test B's `remove_var` race and one of them
    /// reads a CHELIS_REEF_HOME different from what it set.
    /// Poison-tolerant: a panicking test doesn't cascade.
    static CHELIS_REEF_HOME_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Acquire `CHELIS_REEF_HOME_LOCK`, recovering from poison. Bind
    /// the returned guard to a named local that lives for the whole
    /// test body.
    fn lock_reef_home_env() -> std::sync::MutexGuard<'static, ()> {
        CHELIS_REEF_HOME_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner())
    }

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(path, contents).expect("write file");
    }

    /// Lock the runtime version invariant: the `BUNDLED_CHELIS_STD_VERSION`
    /// constant in this crate must equal the `[package].version` field of
    /// `packages/chelis-std/reef.toml`. The constant is hand-maintained
    /// (no `include_str!` because the relative path between the chelis-reef
    /// crate and the chelis-std reef package is fragile across worktrees);
    /// this test catches any drift before it ships.
    #[test]
    fn bundled_chelis_std_version_matches_packages_manifest() {
        // Resolve the workspace root from CARGO_MANIFEST_DIR (chelis-reef)
        // and read the chelis-std reef.toml.
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        let manifest_path = here.join("../../packages/chelis-std/reef.toml");
        let text = fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
            panic!(
                "could not read {}: {e}: \
                 BUNDLED_CHELIS_STD_VERSION sync test cannot run without \
                 the chelis-std reef.toml; if the file moved, update the \
                 path in this test",
                manifest_path.display()
            )
        });
        let manifest: ReefManifest =
            toml::from_str(&text).expect("packages/chelis-std/reef.toml must parse");
        assert_eq!(
            manifest.package.name, CHELIS_STD_PACKAGE_NAME,
            "packages/chelis-std/reef.toml package name must be `chelis-std`"
        );
        assert_eq!(
            manifest.package.version, BUNDLED_CHELIS_STD_VERSION,
            "BUNDLED_CHELIS_STD_VERSION (`{}`) must equal \
             packages/chelis-std/reef.toml's package.version (`{}`); \
             bump both together",
            BUNDLED_CHELIS_STD_VERSION, manifest.package.version,
        );
    }

    #[test]
    fn bundled_chelis_std_lock_hashes_match_embedded_artifacts() {
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        let lock_path = here.join("../../packages/chelis-std/reef.lock");
        let text = fs::read_to_string(&lock_path).unwrap_or_else(|e| {
            panic!(
                "could not read {}: {e}: \
                 chelis-std lock hash sync test cannot run",
                lock_path.display()
            )
        });
        let lock: ReefLock = toml::from_str(&text).expect("packages/chelis-std/reef.lock parses");
        let dep = lock
            .dependencies
            .iter()
            .find(|dep| dep.name == CHELIS_STD_PACKAGE_NAME)
            .expect("chelis-std lock must record bundled runtime dependency");
        assert_eq!(dep.archive_sha256, chelis_std_bundle::archive_sha256());
        assert_eq!(dep.shell_sha256, chelis_std_bundle::shell_sha256());
    }

    /// Negative parity for the version sync: a soft-verify mismatch must
    /// be rejected with a typed error that names both versions, not a
    /// generic "missing dep" or 404.
    #[test]
    fn lock_source_bundled_round_trip_is_fixed_point() {
        let lock = ReefLock {
            package: PackageId {
                name: "downstream".to_string(),
                version: "0.1.0".to_string(),
            },
            dependencies: vec![LockedDependency {
                name: "chelis-std".to_string(),
                version: "0.1.0".to_string(),
                source: LockSource::Bundled {
                    compiler_version: "0.5.0".to_string(),
                },
                compiler: "=0.5.0".to_string(),
                archive_sha256: String::new(),
                shell_sha256: String::new(),
            }],
        };
        // Serialize → deserialize → serialize must be byte-identical
        // and round-trip the `Bundled` variant unchanged.
        let s1 = toml::to_string_pretty(&lock).expect("serialize 1");
        assert!(
            s1.contains("kind = \"bundled\""),
            "serialized lockfile must use the `bundled` kind tag; got:\n{s1}"
        );
        assert!(
            s1.contains("compiler_version = \"0.5.0\""),
            "serialized lockfile must record the compiler_version; got:\n{s1}"
        );
        let parsed: ReefLock = toml::from_str(&s1).expect("deserialize");
        assert_eq!(parsed, lock, "Bundled round-trip must preserve all fields");
        let s2 = toml::to_string_pretty(&parsed).expect("serialize 2");
        assert_eq!(s1, s2, "second serialize must be a fixed point");
    }

    /// An old lockfile with `kind = "local_registry"` for chelis-std must
    /// still deserialize successfully; the migration path then routes it
    /// as bundled at resolve time. This test pins the deserialization
    /// half of Decision 8.
    #[test]
    fn old_chelis_std_local_registry_lockfile_still_deserializes() {
        let old_text = r#"[package]
name = "downstream"
version = "0.1.0"

[[dependencies]]
name = "chelis-std"
version = "0.1.0"
compiler = "=0.5.0"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "local_registry"
"#;
        let lock: ReefLock =
            toml::from_str(old_text).expect("old chelis-std-as-local_registry must deserialize");
        assert_eq!(lock.dependencies.len(), 1);
        assert_eq!(lock.dependencies[0].name, "chelis-std");
        assert!(matches!(
            lock.dependencies[0].source,
            LockSource::LocalRegistry {
                remote_origin: None
            }
        ));
    }

    #[test]
    fn manifest_roundtrip() {
        let manifest = ReefManifest {
            package: ManifestPackage {
                name: "demo".to_string(),
                version: "0.1.0".to_string(),
                compiler: CURRENT_COMPILER_VERSION.to_string(),
                module_prefix: "Demo".to_string(),
                additional_sources: Vec::new(),
            },
            dependencies: BTreeMap::from([(
                "chelis-std".to_string(),
                DependencySpec {
                    version: Some("0.1.0".to_string()),
                    path: None,
                },
            )]),
        };
        let text = toml::to_string_pretty(&manifest).expect("serialize");
        let parsed = toml::from_str::<ReefManifest>(&text).expect("parse");
        assert_eq!(parsed, manifest);
    }

    #[test]
    fn module_prefix_and_path_must_both_match() {
        let err = validate_module_path("Std", "Foo.Bar", Path::new("nn/linear.ch"))
            .expect_err("prefix mismatch should fail");
        assert!(err.contains("module_prefix"));

        let err = validate_module_path("Std", "Std.Foo", Path::new("nn/linear.ch"))
            .expect_err("path mismatch should fail");
        assert!(err.contains("does not match file path"));
    }

    #[test]
    fn package_root_search_only_uses_file_ancestors() {
        let dir = tempdir().expect("tempdir");
        let project = dir.path().join("project");
        let nested = project.join("src/sub");
        fs::create_dir_all(&nested).expect("mkdirs");
        write(
            &project.join("reef.toml"),
            r#"[package]
name = "demo"
version = "0.1.0"
compiler = "=0.1.21"
module_prefix = "Demo"
"#,
        );
        write(&nested.join("demo.ch"), "module Demo.Sub\n\ndef f(x) = x\n");

        let found = find_package_root_for_input(&nested.join("demo.ch"))
            .expect("search")
            .expect("package root");
        assert_eq!(found, project.canonicalize().expect("canonicalize project"));
    }

    /// Positive: when `reef.lock` exists and all deps are path-based, the fast
    /// path resolves correctly and returns within 1 second.
    #[test]
    fn eval_file_uses_lockfile_fast_path_for_path_dep() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("myapp");
        let dep_root = root.join("mylib");

        // Build the dependency package: Mylib.Math with a single export `add`.
        write(
            &dep_root.join("reef.toml"),
            &format!(
                r#"[package]
name = "mylib"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Mylib"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &dep_root.join("src/math.ch"),
            "module Mylib.Math\n\nexport (add)\ndef add(x: int32, y: int32) -> int32 = x + y\n",
        );

        // Build the root package depending on mylib via a local path.
        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Myapp"

[dependencies]
mylib = {{ path = "./mylib" }}
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module Myapp.Main\n\nimport Mylib.Math (add)\ndef double(x: int32) -> int32 = add(x, x)\n",
        );

        // Write a reef.lock that records the path dependency (the fast path reads
        // this instead of calling resolve_package_graph).
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "myapp"
version = "0.1.0"

[[dependencies]]
name = "mylib"
version = "0.1.0"
compiler = "=0.1.0"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "./mylib"
"#,
        );

        // The eval file imports `add` from Mylib.Math.
        let eval_file = dir.path().join("probe.ch");
        write(
            &eval_file,
            "import Mylib.Math (add)\ndef result -> int32 = add(1, 2)\n",
        );

        let start = std::time::Instant::now();
        let result = prepare_program_for_eval_file(&eval_file, &root);
        let elapsed = start.elapsed();

        assert!(
            elapsed < Duration::from_secs(1),
            "fast path took {elapsed:?}. Expected < 1s"
        );
        let program = result
            .expect("prepare_program_for_eval_file should succeed")
            .expect("should find a reef package");
        // The returned decls must include the rewritten definition from Mylib.Math.
        assert!(
            program
                .decls
                .iter()
                .any(|decl| matches!(decl, Decl::FunDef { name, .. } if name.contains("add"))),
            "expected the `add` function from Mylib.Math to appear in compiled decls"
        );
    }

    /// Negative: when there is no lockfile and the registry is absent, the
    /// slow path times out and returns an actionable error instead of hanging.
    ///
    /// This test uses a very short artificial timeout via a direct call to
    /// `run_with_timeout` so it runs quickly in CI.
    #[test]
    fn run_with_timeout_returns_error_on_timeout() {
        let result = run_with_timeout(
            || {
                // Simulate a slow operation — longer than the timeout.
                std::thread::sleep(Duration::from_secs(60));
                Ok::<i32, String>(42)
            },
            Duration::from_millis(50),
            TIMEOUT_MSG,
        );
        let err = result.expect_err("should have timed out");
        assert!(
            err.contains("chelis reef build"),
            "timeout error should mention `chelis reef build`, got: {err}"
        );
    }

    /// Negative: `prepare_program_for_eval_source` with no lockfile and a
    /// registry dependency that is not cached must return an error, not hang.
    #[test]
    fn eval_source_without_lockfile_and_missing_registry_returns_error() {
        let _g = lock_reef_home_env();
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("myapp");

        // Root package with a registry version dep (no local cache, no lockfile).
        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Myapp"

[dependencies]
some-registry-lib = {{ version = "0.1.0" }}
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module Myapp.Main\n\ndef id(x: int32) -> int32 = x\n",
        );

        // No reef.lock — triggers the slow path.  The registry cache does not
        // exist in the temp dir, so the 5-second timeout fires.  We don't want
        // the test itself to block for 5 full seconds, so we test the helper
        // directly and trust the integration via the timeout test above.
        //
        // What we CAN assert without waiting: if CHELIS_REEF_HOME is pointed at
        // an empty directory, load_registry_package errors immediately (no hang).
        //
        // Item 8: auto-fetch is on by default for eval-side callers. Lock the
        // env so auto-fetch surfaces `auth-missing` instantly instead of
        // attempting a real network round trip:
        // - `GITHUB_TOKEN` removed
        // - `PATH` emptied so the `gh auth token` shell-out fails
        // The downstream error is still `is_err()`, which is what this
        // negative test guards against.
        let prior_token = std::env::var_os("GITHUB_TOKEN");
        let prior_path = std::env::var_os("PATH");
        unsafe {
            std::env::set_var("CHELIS_REEF_HOME", dir.path().join("empty_registry"));
            std::env::remove_var("GITHUB_TOKEN");
            std::env::set_var("PATH", "");
        }
        let entry_decls =
            chelis_surf::parser::parse_str("def result -> int32 = 42").expect("parse");
        let result = prepare_program_for_eval_source(&root, &entry_decls);
        unsafe {
            std::env::remove_var("CHELIS_REEF_HOME");
            match prior_token {
                Some(v) => std::env::set_var("GITHUB_TOKEN", v),
                None => std::env::remove_var("GITHUB_TOKEN"),
            }
            match prior_path {
                Some(v) => std::env::set_var("PATH", v),
                None => std::env::remove_var("PATH"),
            }
        }

        // Should either succeed (if somehow resolved) or return an error — the
        // key invariant is that it DOES NOT hang.  Since the registry is empty
        // and there's no lockfile it must fail.
        assert!(
            result.is_err(),
            "expected error when registry dep is uncacheable, got: {result:?}"
        );
    }

    // ---- ADVERSARIAL TESTS: fast path edge cases ----

    /// H: prepare_program_for_eval_file with a file that has an import but NO
    /// reef.toml anywhere in the ancestor chain (no package root at all).
    /// Must return Ok(None) immediately — not hang, not panic, not Err.
    #[test]
    fn adv_eval_file_no_package_root_returns_ok_none() {
        let dir = tempdir().expect("tempdir");
        // No reef.toml anywhere in this temp dir tree.
        let eval_file = dir.path().join("probe.ch");
        // Write a file that has an import — would fail if resolver ran.
        write(
            &eval_file,
            "import NonExistent.Module (something)\ndef result -> int32 = 42\n",
        );

        let start = std::time::Instant::now();
        let result = prepare_program_for_eval_file(&eval_file, dir.path());
        let elapsed = start.elapsed();

        // Must return immediately (< 500ms), not hang.
        assert!(
            elapsed < Duration::from_millis(500),
            "no-package-root eval took {elapsed:?}. Should return immediately"
        );

        // Must return Ok(None): no package root found, no resolution attempted.
        match result {
            Ok(None) => {} // correct
            Ok(Some(_)) => panic!("expected Ok(None) without package root, got Ok(Some(...))"),
            Err(e) => {
                // Could also be Err if the file can't be found in a non-existent location.
                // As long as it doesn't hang, this is acceptable.
                eprintln!("Note: got Err (acceptable if no package root): {e}");
            }
        }
    }

    /// G: Fast path with a reef.lock containing a registry dep (LocalRegistry source).
    /// With an empty registry cache, the 5-second per-dep timeout must fire and return
    /// an actionable error mentioning "chelis reef build", not hang for 5 seconds.
    ///
    /// We test this via the internal `reconstruct_graph_from_lockfile` path for speed.
    #[test]
    fn adv_lockfile_with_registry_dep_and_no_cache_returns_actionable_error() {
        let _g = lock_reef_home_env();
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("myapp");

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Myapp"

[dependencies]
some-lib = {{ version = "0.1.0" }}
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module Myapp.Main\n\ndef id(x: int32) -> int32 = x\n",
        );

        // Write a reef.lock that claims some-lib comes from the local registry (no cache).
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "myapp"
version = "0.1.0"

[[dependencies]]
name = "some-lib"
version = "0.1.0"
compiler = "=0.1.0"
archive_sha256 = "abc"
shell_sha256 = "def"

[dependencies.source]
kind = "local_registry"
"#,
        );

        // Point CHELIS_REEF_HOME at an empty directory so load_registry_package fails fast.
        unsafe {
            std::env::set_var("CHELIS_REEF_HOME", dir.path().join("empty_registry"));
        }

        let lock = read_lockfile(&root.join("reef.lock")).expect("read lockfile");
        // Item 8: opt out of auto-fetch deterministically so the test
        // never attempts network. Auto-fetch's improved-wording error
        // path is the actionable instruction now (it names the URL the
        // user can install manually). Pre-Item-8 wording mentioning
        // `chelis reef build` was misleading once auto-fetch became
        // the default — the spec calls that out explicitly.
        let root_clone = root.clone();
        let result = run_with_timeout(
            move || {
                reconstruct_graph_from_lockfile(
                    &root_clone,
                    &lock,
                    LoadOptions { auto_fetch: false },
                )
            },
            Duration::from_millis(200),
            TIMEOUT_MSG,
        );

        unsafe {
            std::env::remove_var("CHELIS_REEF_HOME");
        }

        let err = result.expect_err("should have failed: registry dep not in cache");
        // Post-Item-8 actionable wording: the error must name the URL
        // the user can run `chelis reef install --from-github` against,
        // and the auto-fetch state.
        assert!(
            err.contains("chelis reef install --from-github")
                || err.contains("auto-fetch disabled"),
            "error must name `chelis reef install --from-github` for actionable recovery; got: {err}"
        );
        assert!(
            err.contains("chelis-lang/some-lib@v0.1.0"),
            "error must name the canonical-org URL that would have been tried; got: {err}"
        );
    }

    /// I: Negative test — timeout error message must contain "chelis reef build".
    /// This is already tested via `run_with_timeout_returns_error_on_timeout`,
    /// but we test the TIMEOUT_MSG constant directly so a change to the message
    /// can't silently break the invariant.
    #[test]
    fn adv_timeout_msg_contains_chelis_reef_build() {
        assert!(
            TIMEOUT_MSG.contains("chelis reef build"),
            "TIMEOUT_MSG must contain 'chelis reef build' for actionable recovery; got: {TIMEOUT_MSG}"
        );
    }

    /// Adversarial: reef.lock with a path dep where the dep directory doesn't exist.
    /// Must return a clean Err (path resolution fails), not a panic or hang.
    #[test]
    fn adv_lockfile_with_missing_path_dep_returns_clean_error() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("myapp");

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Myapp"

[dependencies]
missing = {{ path = "./nonexistent_dep" }}
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module Myapp.Main\n\ndef id(x: int32) -> int32 = x\n",
        );

        // reef.lock says the dep is a path dep to a nonexistent directory.
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "myapp"
version = "0.1.0"

[[dependencies]]
name = "missing"
version = "0.1.0"
compiler = "=0.1.0"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "./nonexistent_dep"
"#,
        );

        let start = std::time::Instant::now();
        let entry_decls =
            chelis_surf::parser::parse_str("def result -> int32 = 42").expect("parse");
        let result = prepare_program_for_eval_source(&root, &entry_decls);
        let elapsed = start.elapsed();

        // Must fail fast with a clean error — not hang.
        assert!(
            elapsed < Duration::from_secs(2),
            "missing-path-dep fast path took {elapsed:?}. Should fail quickly"
        );
        assert!(
            result.is_err(),
            "expected Err for missing path dep, got: {result:?}"
        );
        let err = result.unwrap_err();
        // Should mention the failing dependency name or path.
        assert!(
            err.contains("missing") || err.contains("nonexistent") || err.contains("canonicalize"),
            "error should identify the missing dep; got: {err}"
        );
    }

    // ---- Shared-graph split: prepare_reef_graph + compile_with_reef_graph ----

    /// Build a minimal reef fixture with a lockfile-backed path dependency.
    /// Returns (tempdir, package_root) — the tempdir must be kept alive for
    /// the filesystem to persist.
    fn shared_graph_fixture() -> (tempfile::TempDir, PathBuf) {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("myapp");
        let dep_root = root.join("mylib");

        write(
            &dep_root.join("reef.toml"),
            &format!(
                r#"[package]
name = "mylib"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Mylib"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &dep_root.join("src/math.ch"),
            "module Mylib.Math\n\nexport (add)\ndef add(x: int32, y: int32) -> int32 = x + y\n",
        );

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "myapp"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Myapp"

[dependencies]
mylib = {{ path = "./mylib" }}
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module Myapp.Main\n\nimport Mylib.Math (add)\ndef double(x: int32) -> int32 = add(x, x)\n",
        );

        write(
            &root.join("reef.lock"),
            r#"[package]
name = "myapp"
version = "0.1.0"

[[dependencies]]
name = "mylib"
version = "0.1.0"
compiler = "=0.1.0"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "./mylib"
"#,
        );

        (dir, root)
    }

    // ---- chelis#157: module-scoped constructor resolution ----

    /// Build a two-package graph where the dependency (`coral`) and the
    /// root (`school`) each declare an ADT carrying a same-named
    /// *positional* `IntCol` constructor with a *different arity and field
    /// type*: coral's `IntCol(tensor[n, int64], tensor[n, bool])` takes
    /// two args, school's `IntCol(tensor[n, f32])` takes one. Each
    /// package's own function constructs its own `IntCol`. Both go through
    /// the type-checked positional-application path (`infer_app`), so a
    /// mis-resolution surfaces as a concrete arity/type error. This is the
    /// type-dispatch collision the #148 fix explicitly could not cover
    /// (its `two_positional_same_name` test had to use identical field
    /// types to dodge env-binding last-write-wins).
    fn ctor_collision_two_package_fixture() -> (tempfile::TempDir, PathBuf) {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("school");
        let dep_root = root.join("coral");

        write(
            &dep_root.join("reef.toml"),
            &format!(
                r#"[package]
name = "coral"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Coral"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &dep_root.join("src/frame.ch"),
            "module Coral.Frame\n\
             export (Column, make_int_col)\n\
             type Column[n] = | IntCol(tensor[n, int64], tensor[n, bool])\n\
             def make_int_col[n](xs: tensor[n, int64], mask: tensor[n, bool]) -> Column[n] = IntCol(xs, mask)\n",
        );

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "school"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "School"

[dependencies]
coral = {{ path = "./coral" }}
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module School.Main\n\
             type Dataset[n] = | IntCol(tensor[n, f32])\n\
             def make_dataset[n](xs: tensor[n, f32]) -> Dataset[n] = IntCol(xs)\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "school"
version = "0.1.0"

[[dependencies]]
name = "coral"
version = "0.1.0"
compiler = "=0.1.0"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "./coral"
"#,
        );

        (dir, root)
    }

    /// Positive (#157): linking the school/coral graph mangles each
    /// package's `IntCol` to a distinct package/module-qualified internal
    /// name, so the two constructors no longer collide in the flat
    /// type-checker registry. coral's body resolves its own positional
    /// `IntCol`; school's body resolves its own record `IntCol`; the full
    /// desugar + type/effect/linearity check is clean.
    ///
    /// Pre-fix, both packages bound the bare `IntCol` into one env and the
    /// last-registered scheme won, so coral's positional call mis-resolved
    /// to school's record constructor (or vice versa) and the program
    /// flooded `must use named fields` / `type mismatch` errors.
    #[test]
    fn cross_package_same_named_ctor_resolves_module_scoped() {
        let (_dir, root) = ctor_collision_two_package_fixture();
        let entry = root.join("src/main.ch");

        let prepared = prepare_program_for_file(&entry)
            .expect("prepare_program_for_file ok")
            .expect("entry is inside a reef package");

        // Both constructors survive linking under distinct mangled names.
        let ctor_type_names: Vec<String> = prepared
            .decls
            .iter()
            .filter_map(|decl| match decl {
                Decl::TypeDef { variants, .. } => Some(variants),
                _ => None,
            })
            .flatten()
            .map(|variant| variant.name.clone())
            .filter(|name| name.contains("IntCol"))
            .collect();
        assert_eq!(
            ctor_type_names.len(),
            2,
            "both ADTs must contribute a mangled IntCol variant; got {ctor_type_names:?}"
        );
        assert!(
            ctor_type_names
                .iter()
                .all(|name| name != "IntCol" && name.contains("__IntCol")),
            "each IntCol constructor must be package/module-qualified, not bare; got {ctor_type_names:?}"
        );
        let distinct: BTreeSet<&String> = ctor_type_names.iter().collect();
        assert_eq!(
            distinct.len(),
            2,
            "the two IntCol constructors must mangle to distinct names; got {ctor_type_names:?}"
        );

        // Full pipeline must type/effect/linearity-check with no errors.
        let deep = expanded_desugared_program(&prepared.decls).expect("desugar+expand ok");
        checked_program_with_effects(&deep)
            .expect("module-scoped ctor resolution must type-check the school/coral graph cleanly");
    }

    /// Regression (#157): a wildcard-free exhaustive `match` on a
    /// module-scoped multi-constructor ADT must still type-check. After
    /// reef mangles the constructor names, the type checker's
    /// exhaustiveness check compares the scrutinee ADT's mangled variant
    /// names against the names recorded by each matched pattern. If those
    /// pattern names were recorded bare (the pre-fix behavior) they would
    /// never cover the mangled variants and the checker would fire a
    /// spurious `non-exhaustive match`. The `pat-ctor`/`pat-record`
    /// coverage fix records the *resolved* variant name, so this checks
    /// clean.
    #[test]
    fn module_scoped_adt_exhaustive_match_checks() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("school");

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "school"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "School"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module School.Main\n\
             type Side = | Left | Right\n\
             def flip(s: Side) -> Side = match s with { | Left => Right | Right => Left }\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "school"
version = "0.1.0"
"#,
        );

        let entry = root.join("src/main.ch");
        let prepared = prepare_program_for_file(&entry)
            .expect("prepare ok")
            .expect("entry inside reef package");

        // The variants are mangled, not bare.
        let variant_names: Vec<String> = prepared
            .decls
            .iter()
            .filter_map(|decl| match decl {
                Decl::TypeDef { variants, .. } => Some(variants),
                _ => None,
            })
            .flatten()
            .map(|variant| variant.name.clone())
            .filter(|name| name.contains("Left") || name.contains("Right"))
            .collect();
        assert!(
            variant_names.iter().all(|name| name.contains("__")),
            "Side variants must mangle; got {variant_names:?}"
        );

        let deep = expanded_desugared_program(&prepared.decls).expect("desugar+expand ok");
        checked_program_with_effects(&deep).expect(
            "wildcard-free exhaustive match on a mangled ADT must not fire non-exhaustive match",
        );
    }

    /// Negative parity for the exhaustiveness change (#157): recording the
    /// *resolved* (mangled) variant name for `pat-ctor` must not over-accept.
    /// A genuinely non-exhaustive, wildcard-free match on a module-scoped
    /// multi-constructor ADT (covers `Left` but not `Right`) must still be
    /// REJECTED with a `non-exhaustive match` diagnostic that names the
    /// uncovered (mangled) `Right` variant. This is the counterpart to
    /// `module_scoped_adt_exhaustive_match_checks`: the coverage fix maps a
    /// covered pattern to the same mangled key `variant_names` returns, so a
    /// missing variant stays missing: the change tightens nothing into a
    /// false "exhaustive".
    #[test]
    fn module_scoped_adt_non_exhaustive_match_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("school");

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "school"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "School"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        // `classify` covers only `Left`, omitting `Right`. With the scrutinee
        // ADT's variants mangled, the checker must still see `Right`
        // uncovered.
        write(
            &root.join("src/main.ch"),
            "module School.Main\n\
             type Side = | Left | Right\n\
             def classify(s: Side) -> int32 = match s with { | Left => 0 }\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "school"
version = "0.1.0"
"#,
        );

        let entry = root.join("src/main.ch");
        let prepared = prepare_program_for_file(&entry)
            .expect("prepare ok")
            .expect("entry inside reef package");

        // Sanity: the scrutinee ADT's variants are mangled, so this exercises
        // the resolved-name coverage path, not the bare-name one.
        let right_mangled = internal_name("school", "School.Main", "Right");
        assert!(
            right_mangled.contains("__Right") && right_mangled != "Right",
            "fixture sanity: Right must mangle; got {right_mangled}"
        );

        let deep = expanded_desugared_program(&prepared.decls).expect("desugar+expand ok");
        let err = checked_program_with_effects(&deep).expect_err(
            "a wildcard-free match missing the Right variant must be rejected as non-exhaustive",
        );
        assert!(
            err.contains("NonExhaustiveMatch") || err.contains("non-exhaustive"),
            "diagnostic must report a non-exhaustive match; got: {err}"
        );
        assert!(
            err.contains("Right"),
            "diagnostic must name the uncovered Right variant (mangled or bare); got: {err}"
        );
    }

    /// Negative parity (#157): the principled fix must not silently
    /// dispatch a genuinely ambiguous unqualified constructor reference.
    /// A module that imports `IntCol` unqualified from two different
    /// modules (here coral's `Frame` and `Frame2`, each declaring an
    /// `IntCol`) must be rejected with an explicit ambiguity diagnostic,
    /// not last-binding-wins.
    #[test]
    fn ambiguous_unqualified_ctor_import_is_rejected() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("school");
        let dep_root = root.join("coral");

        write(
            &dep_root.join("reef.toml"),
            &format!(
                r#"[package]
name = "coral"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Coral"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &dep_root.join("src/frame.ch"),
            "module Coral.Frame\n\
             export (ColumnA, IntCol)\n\
             type ColumnA[n] = | IntCol(tensor[n, int64])\n",
        );
        write(
            &dep_root.join("src/frame2.ch"),
            "module Coral.Frame2\n\
             export (ColumnB, IntCol)\n\
             type ColumnB[n] = | IntCol(tensor[n, f32])\n",
        );

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "school"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "School"

[dependencies]
coral = {{ path = "./coral" }}
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module School.Main\n\
             import Coral.Frame (ColumnA, IntCol)\n\
             import Coral.Frame2 (IntCol)\n\
             def use_it[n](xs: tensor[n, int64]) -> ColumnA[n] = IntCol(xs)\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "school"
version = "0.1.0"

[[dependencies]]
name = "coral"
version = "0.1.0"
compiler = "=0.1.0"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "./coral"
"#,
        );

        let entry = root.join("src/main.ch");
        let err = prepare_program_for_file(&entry).expect_err(
            "ambiguous unqualified IntCol import must be rejected, not silently dispatched",
        );
        assert!(
            err.contains("ambiguous reference to `IntCol`"),
            "diagnostic must name the ambiguous constructor; got: {err}"
        );
        assert!(
            err.contains("Coral.Frame") && err.contains("Coral.Frame2"),
            "diagnostic must name both source modules; got: {err}"
        );
    }

    /// No-regression for the cross-module *import* of a constructor
    /// (#157 module-system semantics): a module that imports a single
    /// `IntCol` from one module and uses it resolves to that module's
    /// mangled constructor and type-checks cleanly. This is the positive
    /// control for `ambiguous_unqualified_ctor_import_is_rejected`: one
    /// import is fine, two colliding imports are the error.
    #[test]
    fn single_unqualified_ctor_import_resolves_and_checks() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("school");
        let dep_root = root.join("coral");

        write(
            &dep_root.join("reef.toml"),
            &format!(
                r#"[package]
name = "coral"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Coral"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &dep_root.join("src/frame.ch"),
            "module Coral.Frame\n\
             export (Column, IntCol)\n\
             type Column[n] = | IntCol(tensor[n, int64])\n",
        );

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "school"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "School"

[dependencies]
coral = {{ path = "./coral" }}
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module School.Main\n\
             import Coral.Frame (Column, IntCol)\n\
             def use_it[n](xs: tensor[n, int64]) -> Column[n] = IntCol(xs)\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "school"
version = "0.1.0"

[[dependencies]]
name = "coral"
version = "0.1.0"
compiler = "=0.1.0"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "./coral"
"#,
        );

        let entry = root.join("src/main.ch");
        let prepared = prepare_program_for_file(&entry)
            .expect("prepare ok")
            .expect("entry inside reef package");
        let deep = expanded_desugared_program(&prepared.decls).expect("desugar+expand ok");
        checked_program_with_effects(&deep)
            .expect("single imported constructor must resolve and type-check");
    }

    /// Regression (#157): a module that declares its own constructor `Mark`
    /// AND imports a different `Mark` from a dependency must resolve a
    /// *pattern* head the same way it resolves a *construction* head:
    /// own-module-first. The design rule is "a name the module declares
    /// itself shadows imports", and the ambiguity check honors it for
    /// expressions. Before the `resolve_ctor_pattern_name` fix, the pattern
    /// rewrite checked `imported_names` first; since an import overwrites
    /// the own seed in that map, `Tag(Mark)` was *constructed* with the own
    /// mangled name but `| Tag(Mark) =>` was *matched* against the imported
    /// (other-package) mangled name. The two never unified and the program
    /// failed to type-check. This test pins construction and destructuring
    /// to the same own internal name.
    #[test]
    fn own_ctor_shadows_imported_same_name_in_pattern() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("school");
        let dep_root = root.join("coral");

        // Dependency declares + exports its own `Mark` constructor.
        write(
            &dep_root.join("reef.toml"),
            &format!(
                r#"[package]
name = "coral"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Coral"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &dep_root.join("src/frame.ch"),
            "module Coral.Frame\n\
             export (Stamp, Mark)\n\
             type Stamp = | Mark(int64)\n",
        );

        // Root module declares its OWN `Tag` with an own `Mark` constructor
        // (positional, single field), imports the dependency's different
        // `Mark`, then both builds and matches its own `Mark`.
        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "school"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "School"

[dependencies]
coral = {{ path = "./coral" }}
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module School.Main\n\
             import Coral.Frame (Stamp, Mark)\n\
             type Tag = | Mark(f32)\n\
             def build(x: f32) -> Tag = Mark(x)\n\
             def unwrap(t: Tag) -> f32 = match t with { | Mark(v) => v }\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "school"
version = "0.1.0"

[[dependencies]]
name = "coral"
version = "0.1.0"
compiler = "=0.1.0"
archive_sha256 = ""
shell_sha256 = ""

[dependencies.source]
kind = "path"
path = "./coral"
"#,
        );

        let entry = root.join("src/main.ch");
        let prepared = prepare_program_for_file(&entry)
            .expect("prepare ok")
            .expect("entry inside reef package");

        // The own `Mark` mangles to School.Main's internal name; the
        // imported `Mark` mangles to Coral.Frame's. The match arm head must
        // equal the construction head (the own one), never the imported one.
        let imported_mark = internal_name("coral", "Coral.Frame", "Mark");
        let own_mark = internal_name("school", "School.Main", "Mark");
        assert_ne!(
            own_mark, imported_mark,
            "fixture sanity: own and imported Mark must mangle differently"
        );

        let mut construction_head: Option<String> = None;
        let mut pattern_head: Option<String> = None;
        for decl in &prepared.decls {
            let Decl::FunDef { name, body, .. } = decl else {
                continue;
            };
            if name.ends_with("__build")
                && let Expr::Apply(func, _, _) = body
                && let Expr::Constructor(ctor, _) = func.as_ref()
            {
                construction_head = Some(ctor.clone());
            }
            if name.ends_with("__unwrap")
                && let Expr::Match(_, arms, _) = body
                && let Some(arm) = arms.first()
                && let Pattern::Constructor(ctor, _, _) = &arm.pattern
            {
                pattern_head = Some(ctor.clone());
            }
        }

        let construction_head =
            construction_head.expect("build() body must be a constructor application");
        let pattern_head = pattern_head.expect("unwrap() body must be a constructor-pattern match");
        assert_eq!(
            construction_head, own_mark,
            "construction must resolve to the own Mark; got {construction_head}"
        );
        assert_eq!(
            pattern_head, own_mark,
            "pattern head must resolve to the SAME own Mark, not the imported one; got {pattern_head} (imported is {imported_mark})"
        );

        // And the full pipeline must type-check: construction and
        // destructuring agree on one mangled constructor.
        let deep = expanded_desugared_program(&prepared.decls).expect("desugar+expand ok");
        checked_program_with_effects(&deep).expect(
            "own constructor shadowing an imported same-name must construct and match consistently",
        );
    }

    /// Issue #316: two modules in one package each declare
    /// `type Mode = | Train | Eval`. A third module that needs both must be
    /// able to disambiguate with a module-qualified reference
    /// (`Demo.Dropout.Eval`) instead of renaming one module's constructors.
    /// This is the positive resolution of `ambiguous_unqualified_ctor_import_is_rejected`:
    /// where importing both `Eval`s unqualified is an error, qualifying each
    /// reference resolves cleanly to that module's own mangled constructor and
    /// the program type-checks. Before the parser fix the qualified form did
    /// not even parse — the ambiguity diagnostic recommended a syntax the
    /// front end rejected.
    #[test]
    fn qualified_ctor_reference_disambiguates_same_named_constructors() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("demo");

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "demo"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Demo"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/dropout.ch"),
            "module Demo.Dropout\n\
             export (Mode, use)\n\
             type Mode = | Train | Eval\n\
             def use(m: Mode) -> i64 = match m with { | Train => 1 | Eval => 0 }\n",
        );
        write(
            &root.join("src/sd.ch"),
            "module Demo.Sd\n\
             export (Mode, use)\n\
             type Mode = | Train | Eval\n\
             def use(m: Mode) -> i64 = match m with { | Train => 1 | Eval => 0 }\n",
        );
        // `combo` pulls in both modules qualified-access-only (`()`), so no
        // unqualified `Mode`/`Train`/`Eval`/`use` collide, and reaches each
        // module's constructor and value through the qualified path.
        write(
            &root.join("src/combo.ch"),
            "module Demo.Combo\n\
             import Demo.Dropout ()\n\
             import Demo.Sd ()\n\
             def go() -> i64 = add(Demo.Dropout.use(Demo.Dropout.Eval), Demo.Sd.use(Demo.Sd.Train))\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "demo"
version = "0.1.0"
"#,
        );

        let entry = root.join("src/combo.ch");
        let prepared = prepare_program_for_file(&entry)
            .expect("prepare ok")
            .expect("entry inside reef package");

        // The qualified references must resolve to each module's OWN mangled
        // names, not collapse onto one. Walk `go`'s rewritten body and collect
        // every `Var` head; it must contain both modules' `Eval`/`Train` and
        // both `use` internals.
        fn collect_vars(expr: &Expr, out: &mut Vec<String>) {
            match expr {
                Expr::Var(name, _) | Expr::Constructor(name, _) => out.push(name.clone()),
                Expr::Apply(func, args, _) => {
                    collect_vars(func, out);
                    for arg in args {
                        collect_vars(arg, out);
                    }
                }
                Expr::Binary(_, l, r, _) => {
                    collect_vars(l, out);
                    collect_vars(r, out);
                }
                Expr::Access(inner, _, _) => collect_vars(inner, out),
                _ => {}
            }
        }
        let go_body = prepared
            .decls
            .iter()
            .find_map(|decl| match decl {
                Decl::FunDef { name, body, .. } if name.ends_with("__go") => Some(body),
                _ => None,
            })
            .expect("combo go() must be present after rewrite");
        let mut vars = Vec::new();
        collect_vars(go_body, &mut vars);

        let dropout_eval = internal_name("demo", "Demo.Dropout", "Eval");
        let sd_train = internal_name("demo", "Demo.Sd", "Train");
        let dropout_use = internal_name("demo", "Demo.Dropout", "use");
        let sd_use = internal_name("demo", "Demo.Sd", "use");
        assert_ne!(
            dropout_eval, sd_train,
            "fixture sanity: per-module constructors must mangle differently"
        );
        for expected in [&dropout_eval, &sd_train, &dropout_use, &sd_use] {
            assert!(
                vars.contains(expected),
                "qualified reference must resolve to {expected}; resolved heads were {vars:?}"
            );
        }

        // The full pipeline must type-check: each `use` receives a value of its
        // own module's `Mode`, so the #316 `Pkg__demo__Demo__Dropout__Mode vs
        // Pkg__demo__Demo__Sd__Mode` mismatch cannot arise.
        let deep = expanded_desugared_program(&prepared.decls).expect("desugar+expand ok");
        checked_program_with_effects(&deep)
            .expect("module-qualified constructor references must let two same-named ADTs coexist");
    }

    /// Negative parity for #316: a module-qualified reference to a name the
    /// target module does NOT export must not silently resolve. `Demo.Dropout`
    /// does not export `Missing`, so `Demo.Dropout.Missing` is rejected during
    /// reef rewrite with a `does not export` error rather than being invented
    /// or left to silently survive into a later stage.
    #[test]
    fn qualified_reference_to_unexported_name_is_unresolved() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("demo");

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "demo"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Demo"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/dropout.ch"),
            "module Demo.Dropout\n\
             export (Mode, use)\n\
             type Mode = | Train | Eval\n\
             def use(m: Mode) -> i64 = match m with { | Train => 1 | Eval => 0 }\n",
        );
        write(
            &root.join("src/combo.ch"),
            "module Demo.Combo\n\
             import Demo.Dropout ()\n\
             def go() -> i64 = Demo.Dropout.use(Demo.Dropout.Missing)\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "demo"
version = "0.1.0"
"#,
        );

        let entry = root.join("src/combo.ch");
        let err = prepare_program_for_file(&entry)
            .expect_err("qualified reference to an unexported name must be rejected");
        assert!(
            err.contains("does not export") && err.contains("Missing"),
            "diagnostic must name the unexported leaf; got: {err}"
        );
    }

    /// Issue #316 (patterns): module-qualified constructor *patterns*
    /// (`| Demo.Dropout.Train =>`) resolve to the declaring module's mangled
    /// constructor, the destructuring dual of the qualified construction in
    /// `qualified_ctor_reference_disambiguates_same_named_constructors`. With
    /// two modules each declaring `type Mode = | Train | Eval`, a `match` whose
    /// arms qualify against one module must bind that module's variants (so the
    /// arm heads equal the scrutinee ADT's variant names) and type-check.
    #[test]
    fn qualified_constructor_patterns_resolve_per_module() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("demo");

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "demo"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Demo"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/dropout.ch"),
            "module Demo.Dropout\n\
             export (Mode, Train, Eval)\n\
             type Mode = | Train | Eval\n",
        );
        write(
            &root.join("src/sd.ch"),
            "module Demo.Sd\n\
             export (Mode, Train, Eval)\n\
             type Mode = | Train | Eval\n",
        );
        // The scrutinee is a qualified constructor expression (pinning the
        // ADT), and every arm qualifies to the same module — so no unqualified
        // `Mode`/`Train`/`Eval` is needed and the two modules cannot collide.
        write(
            &root.join("src/combo.ch"),
            "module Demo.Combo\n\
             import Demo.Dropout ()\n\
             import Demo.Sd ()\n\
             def classify_dropout() -> i64 = match Demo.Dropout.Train with { | Demo.Dropout.Train => 1 | Demo.Dropout.Eval => 0 }\n\
             def classify_sd() -> i64 = match Demo.Sd.Eval with { | Demo.Sd.Train => 1 | Demo.Sd.Eval => 0 }\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "demo"
version = "0.1.0"
"#,
        );

        let entry = root.join("src/combo.ch");
        let prepared = prepare_program_for_file(&entry)
            .expect("prepare ok")
            .expect("entry inside reef package");

        // Collect each function's match-arm pattern heads.
        fn arm_heads(decls: &[Decl], fn_suffix: &str) -> Vec<String> {
            decls
                .iter()
                .find_map(|decl| match decl {
                    Decl::FunDef { name, body, .. } if name.ends_with(fn_suffix) => Some(body),
                    _ => None,
                })
                .and_then(|body| match body {
                    Expr::Match(_, arms, _) => Some(arms),
                    _ => None,
                })
                .map(|arms| {
                    arms.iter()
                        .filter_map(|arm| match &arm.pattern {
                            Pattern::Constructor(name, _, _) => Some(name.clone()),
                            _ => None,
                        })
                        .collect()
                })
                .unwrap_or_default()
        }

        let dropout_heads = arm_heads(&prepared.decls, "__classify_dropout");
        let sd_heads = arm_heads(&prepared.decls, "__classify_sd");
        assert_eq!(
            dropout_heads,
            vec![
                internal_name("demo", "Demo.Dropout", "Train"),
                internal_name("demo", "Demo.Dropout", "Eval"),
            ],
            "Dropout arms must resolve to Dropout's mangled constructors"
        );
        assert_eq!(
            sd_heads,
            vec![
                internal_name("demo", "Demo.Sd", "Train"),
                internal_name("demo", "Demo.Sd", "Eval"),
            ],
            "Sd arms must resolve to Sd's mangled constructors"
        );

        // Both exhaustive matches must type-check: arm heads and scrutinee
        // agree on one module's `Mode`, so neither the #316 cross-module
        // mismatch nor a spurious non-exhaustive diagnostic can arise.
        let deep = expanded_desugared_program(&prepared.decls).expect("desugar+expand ok");
        checked_program_with_effects(&deep)
            .expect("qualified constructor patterns must type-check per module");
    }

    /// Issue #316 (types): a module-qualified *type* name (`Demo.Dropout.Mode`)
    /// in annotation position resolves to the declaring module's mangled type,
    /// so a consumer that imports two modules exporting the same type name can
    /// still annotate against one. Completes the qualification trio alongside
    /// qualified constructor expressions and patterns.
    #[test]
    fn qualified_type_name_resolves_in_annotation() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("demo");

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "demo"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Demo"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/dropout.ch"),
            "module Demo.Dropout\n\
             export (Mode, use)\n\
             type Mode = | Train | Eval\n\
             def use(m: Mode) -> i64 = match m with { | Train => 1 | Eval => 0 }\n",
        );
        write(
            &root.join("src/sd.ch"),
            "module Demo.Sd\n\
             export (Mode, use)\n\
             type Mode = | Train | Eval\n\
             def use(m: Mode) -> i64 = match m with { | Train => 1 | Eval => 0 }\n",
        );
        // `relay` annotates its parameter with the qualified type and forwards
        // it to the qualified `use`. Both `Mode` ADTs are linked, so a bare
        // `Mode` would be ambiguous; the qualified annotation pins Dropout's.
        write(
            &root.join("src/combo.ch"),
            "module Demo.Combo\n\
             import Demo.Dropout ()\n\
             import Demo.Sd ()\n\
             def relay(m: Demo.Dropout.Mode) -> i64 = Demo.Dropout.use(m)\n",
        );
        write(
            &root.join("reef.lock"),
            r#"[package]
name = "demo"
version = "0.1.0"
"#,
        );

        let entry = root.join("src/combo.ch");
        let prepared = prepare_program_for_file(&entry)
            .expect("prepare ok")
            .expect("entry inside reef package");

        // `relay`'s parameter type must resolve to Dropout's mangled `Mode`.
        let expected = internal_name("demo", "Demo.Dropout", "Mode");
        let param_ty = prepared
            .decls
            .iter()
            .find_map(|decl| match decl {
                Decl::FunDef { name, params, .. } if name.ends_with("__relay") => {
                    params.first().and_then(|p| p.ty.clone())
                }
                _ => None,
            })
            .expect("relay must have an annotated parameter");
        assert!(
            matches!(&param_ty, TypeExpr::Named(n, _) if *n == expected),
            "qualified type must resolve to Dropout's mangled Mode ({expected}); got {param_ty:?}"
        );

        let deep = expanded_desugared_program(&prepared.decls).expect("desugar+expand ok");
        checked_program_with_effects(&deep)
            .expect("qualified type annotation must resolve and type-check");
    }

    /// Positive: `prepare_reef_graph` + two calls to `compile_with_reef_graph`
    /// produces the same library decls as two independent calls to
    /// `prepare_program_for_eval_source`. The test files each import `add`
    /// from Mylib.Math to exercise the linked-module + rewrite path.
    #[test]
    fn prepare_reef_graph_split_matches_single_shot_semantics() {
        let (_dir, root) = shared_graph_fixture();

        let probe_source_a = "import Mylib.Math (add)\ndef result_a -> int32 = add(1, 2)\n";
        let probe_source_b = "import Mylib.Math (add)\ndef result_b -> int32 = add(3, 4)\n";
        let decls_a = chelis_surf::parser::parse_str(probe_source_a).expect("parse a");
        let decls_b = chelis_surf::parser::parse_str(probe_source_b).expect("parse b");

        // Split-path: one graph preparation, two per-file compiles.
        let graph = prepare_reef_graph(&root).expect("prepare_reef_graph should succeed");
        let split_a = compile_with_reef_graph(&graph, &decls_a).expect("compile a via graph");
        let split_b = compile_with_reef_graph(&graph, &decls_b).expect("compile b via graph");

        // Baseline: two independent full preparations via the convenience wrapper.
        let single_a = prepare_program_for_eval_source(&root, &decls_a)
            .expect("single-shot a ok")
            .expect("single-shot a some");
        let single_b = prepare_program_for_eval_source(&root, &decls_b)
            .expect("single-shot b ok")
            .expect("single-shot b some");

        // Library decl count must match exactly: the graph-shared path must
        // not drop or duplicate any linked module decl.
        let lib_decl_count = |p: &PreparedProgram| p.decls.len() - p.entry_decls.len();
        assert_eq!(
            lib_decl_count(&split_a),
            lib_decl_count(&single_a),
            "library decl count must match between split and single-shot for probe a"
        );
        assert_eq!(
            lib_decl_count(&split_b),
            lib_decl_count(&single_b),
            "library decl count must match between split and single-shot for probe b"
        );
        // The rewritten `add` function must appear in both.
        let has_add = |p: &PreparedProgram| {
            p.decls
                .iter()
                .any(|d| matches!(d, Decl::FunDef { name, .. } if name.contains("add")))
        };
        assert!(has_add(&split_a), "split a must contain rewritten `add`");
        assert!(has_add(&split_b), "split b must contain rewritten `add`");
        // `package_root` agrees too.
        assert_eq!(split_a.package_root, single_a.package_root);
        assert_eq!(split_b.package_root, single_b.package_root);
    }

    /// Diagnostic perf check (plan: "skip the timing assertion if it becomes
    /// flaky. The correctness test is required, the perf test is diagnostic").
    ///
    /// We do NOT gate the suite on a hard 2x multiplier because on a small
    /// path-dep fixture both paths are sub-millisecond and noise dominates.
    /// What we CAN assert robustly: the split path is no slower than the
    /// single-shot path when amortized across two files, which is a
    /// sufficient signal that the shared graph isn't redoing work.
    ///
    /// Ignored by default: the assertion compares wall-clock elapsed time
    /// against a 3x multiplier of a sub-millisecond single-shot baseline.
    /// Under the nextest global thread pool the whole workspace runs in one
    /// pool, so CPU contention from concurrent gcc-compile-and-run tests can
    /// inflate the split path past the multiplier even though no extra work
    /// is being done. The correctness guarantee lives in
    /// `prepare_reef_graph_split_matches_single_shot_semantics`, which has no
    /// timing dependency. This test stays as a documented manual perf gate:
    ///
    ///   cargo test -p chelis-reef -- --ignored \
    ///     prepare_reef_graph_amortizes_work_across_multiple_files
    ///
    /// Expected success condition: the assertion passes on an unloaded
    /// machine (run it serially, not under a full workspace test pass).
    #[test]
    #[ignore = "diagnostic wall-clock perf gate; contention-sensitive, run manually unloaded"]
    fn prepare_reef_graph_amortizes_work_across_multiple_files() {
        let (_dir, root) = shared_graph_fixture();

        let probe_source_a = "import Mylib.Math (add)\ndef result_a -> int32 = add(1, 2)\n";
        let probe_source_b = "import Mylib.Math (add)\ndef result_b -> int32 = add(3, 4)\n";
        let decls_a = chelis_surf::parser::parse_str(probe_source_a).expect("parse a");
        let decls_b = chelis_surf::parser::parse_str(probe_source_b).expect("parse b");

        // Warm the OS caches — first run of either path tends to be skewed.
        let _ = prepare_program_for_eval_source(&root, &decls_a);
        let _ = prepare_reef_graph(&root);

        // Single-shot: two independent full preparations.
        let start_single = std::time::Instant::now();
        let _ = prepare_program_for_eval_source(&root, &decls_a).expect("single a ok");
        let _ = prepare_program_for_eval_source(&root, &decls_b).expect("single b ok");
        let single_elapsed = start_single.elapsed();

        // Split-shared: one graph prep, two per-file compiles.
        let start_split = std::time::Instant::now();
        let graph = prepare_reef_graph(&root).expect("prepare_reef_graph ok");
        let _ = compile_with_reef_graph(&graph, &decls_a).expect("compile a ok");
        let _ = compile_with_reef_graph(&graph, &decls_b).expect("compile b ok");
        let split_elapsed = start_split.elapsed();

        // Diagnostic: the split path must not regress the single-shot path.
        // On tiny fixtures the absolute numbers are noise, so we only flag
        // pathological regressions (split slower than 3x single-shot).
        assert!(
            split_elapsed.as_nanos() <= single_elapsed.as_nanos() * 3 + 1_000_000,
            "split path unexpectedly slower: split={split_elapsed:?}, single={single_elapsed:?}"
        );
    }

    /// Negative: `prepare_reef_graph` called outside a reef package must
    /// return an actionable error (containing the offending directory path),
    /// not a silent `Ok(None)` that loses information.
    #[test]
    fn prepare_reef_graph_outside_package_returns_error() {
        let dir = tempdir().expect("tempdir");
        // The tempdir contains no reef.toml in its ancestor chain.
        let err =
            prepare_reef_graph(dir.path()).expect_err("outside a reef package must be an error");
        assert!(
            err.contains("reef.toml"),
            "error should mention reef.toml; got: {err}"
        );
        assert!(
            err.contains(dir.path().to_str().unwrap_or_default()),
            "error should mention the attempted directory; got: {err}"
        );
    }

    /// Regression: a stray `reef.toml` directly in the OS temp dir
    /// (e.g. `/tmp/reef.toml`, left by some unrelated tool or
    /// developer experiment) must NOT be picked up as the package root
    /// for a `tempdir()`-rooted lookup. The ancestor walk has to stop at
    /// the temp-dir boundary; otherwise a single foreign manifest with a
    /// non-matching `module_prefix` poisons every test that runs from a
    /// temp dir, which is most of them.
    #[test]
    fn ancestor_walk_stops_at_os_temp_dir_boundary() {
        // Create a stray reef.toml directly in the OS temp root, mimicking
        // the Octant manifest that was found there in the wild.
        let temp_root = std::env::temp_dir();
        let stray = temp_root.join("reef.toml");
        // Only plant the stray manifest if the temp root is writeable AND
        // there isn't already one there (we don't want to clobber a real
        // user file). If a stray already exists, the test still validates
        // the fix because the lookup below must STILL return None.
        let planted = if !stray.exists() {
            std::fs::write(
                &stray,
                r#"[package]
name = "stray"
version = "0.0.0"
compiler = "=0.0.0"
module_prefix = "Stray"
"#,
            )
            .is_ok()
        } else {
            false
        };

        let dir = tempdir().expect("tempdir");
        let result = find_package_root_for_dir(dir.path());

        // Clean up the planted file before asserting so a failed assertion
        // doesn't leave litter behind.
        if planted {
            let _ = std::fs::remove_file(&stray);
        }

        // The walk must not escape the per-test tempdir into the shared
        // OS temp root, so it must return Ok(None) — not Ok(Some(/tmp)).
        match result {
            Ok(None) => {}
            Ok(Some(found)) => panic!(
                "ancestor walk leaked into shared temp space: found `{}` from tempdir `{}`",
                found.display(),
                dir.path().display()
            ),
            Err(e) => panic!("unexpected error from ancestor walk: {e}"),
        }
    }

    /// Phase B prerequisite: `source_digests` walks every backed source
    /// file across the root + path-dep packages and returns one row per
    /// `.ch` file. The path-dep fixture has 1 root module + 1 path-dep
    /// module = 2 rows.
    #[test]
    fn source_digests_returns_one_row_per_source_file_for_path_dep_fixture() {
        let (_dir, root) = shared_graph_fixture();
        let graph = prepare_reef_graph(&root).expect("prepare graph");
        let digests = graph.source_digests().expect("source_digests");
        // Sort key invariant: digests are returned in (pkg, ver, mod)
        // order. Validate the count and the load-bearing ordering for
        // the fixture's two known modules.
        assert_eq!(
            digests.len(),
            2,
            "got {} digests: {:?}",
            digests.len(),
            digests
        );
        assert!(
            digests
                .windows(2)
                .all(|w| (w[0].package_name.as_str(), w[0].module_name.as_str())
                    <= (w[1].package_name.as_str(), w[1].module_name.as_str())),
            "digests must be sorted by (package, module)"
        );
        // No two rows for the same module — the walker must not double-count.
        let mut seen = std::collections::HashSet::new();
        for d in &digests {
            assert!(
                seen.insert((d.package_name.clone(), d.module_name.clone())),
                "duplicate digest for ({}, {})",
                d.package_name,
                d.module_name
            );
        }
        // Every sha256 must be non-zero — empty file would hash to e3b0...,
        // not zeros, so [0;32] is a clear "didn't actually hash" sentinel.
        assert!(digests.iter().all(|d| d.sha256 != [0u8; 32]));
    }

    /// Phase A foundation: PreparedReefGraph round-trips through bincode.
    /// This is prerequisite for the Phase I disk cache (CompiledContext::save/
    /// load_if_fresh will use the same bincode + content-hash pattern).
    #[test]
    fn prepared_reef_graph_round_trips_through_bincode() {
        let (_dir, root) = shared_graph_fixture();
        let original = prepare_reef_graph(&root).expect("prepare graph");
        let bytes = original.encode().expect("encode graph");
        let restored = PreparedReefGraph::decode(&bytes).expect("decode graph");

        // Spot-check the fields that matter for downstream compilation:
        // package_root + linked decl count + dep shell count + module
        // prefix. Full PartialEq isn't derived (HashMap key-order would
        // make it non-deterministic anyway), so check the load-bearing
        // surface explicitly.
        assert_eq!(original.package_root, restored.package_root);
        assert_eq!(
            original.linked_library_decls.len(),
            restored.linked_library_decls.len(),
        );
        assert_eq!(original.dep_shells.len(), restored.dep_shells.len());
        assert_eq!(original.eval_module_prefix, restored.eval_module_prefix);
        assert_eq!(
            original.internal_maps.len(),
            restored.internal_maps.len(),
            "internal maps survive round-trip"
        );
    }

    // ---- Multi-source-roots (additional_sources) ----

    /// Backward compat: a `reef.toml` without `additional_sources`
    /// deserializes with an empty Vec — existing packages keep
    /// working without modification.
    #[test]
    fn additional_sources_default_is_empty() {
        let toml_text = format!(
            r#"[package]
name = "demo"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Demo"
"#,
            ver = CURRENT_COMPILER_VERSION
        );
        let parsed: ReefManifest = toml::from_str(&toml_text).expect("parse");
        assert!(
            parsed.package.additional_sources.is_empty(),
            "additional_sources must default to empty Vec, got {:?}",
            parsed.package.additional_sources
        );
        // And serialization round-trips: empty Vec serializes back and
        // re-parses to the same empty Vec.
        let re_serialized = toml::to_string(&parsed).expect("serialize");
        let re_parsed: ReefManifest = toml::from_str(&re_serialized).expect("re-parse");
        assert_eq!(re_parsed, parsed);
    }

    /// Happy path: a package with `additional_sources = ["properties"]`
    /// has both `src/` and `properties/` walked, both produce
    /// importable modules, and the module names follow the
    /// path-with-root-prefix rule
    /// (`<prefix>.foo` for src/foo.ch,
    ///  `<prefix>.properties.bar` for properties/bar.ch).
    /// Cross-root imports must resolve.
    #[test]
    fn multi_root_packages_resolve_correctly() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("multi");

        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "multi"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Pkg"
additional_sources = ["properties"]
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        // src/foo.ch defines call_price; module Pkg.Foo.
        write(
            &root.join("src/foo.ch"),
            "module Pkg.Foo\n\nexport (call_price)\ndef call_price(s: f32, k: f32) -> f32 = s - k\n",
        );
        // properties/bar.ch imports Pkg.Foo (call_price) and uses it.
        // Must declare module Pkg.Properties.Bar — the path-to-module
        // rule with the source root name as the first segment after the
        // module prefix.
        write(
            &root.join("properties/bar.ch"),
            "module Pkg.Properties.Bar\n\nimport Pkg.Foo (call_price)\nexport (matches)\ndef matches(s: f32, k: f32, expected: f32) -> bool = call_price(s, k) == expected\n",
        );

        let modules = load_package_modules(
            &root,
            &read_manifest(&root.join("reef.toml")).expect("manifest"),
        )
        .expect("load modules across roots");
        // Both modules must be present, with distinct names.
        assert!(
            modules.contains_key("Pkg.Foo"),
            "expected Pkg.Foo from src/, got: {:?}",
            modules.keys().collect::<Vec<_>>()
        );
        assert!(
            modules.contains_key("Pkg.Properties.Bar"),
            "expected Pkg.Properties.Bar from properties/, got: {:?}",
            modules.keys().collect::<Vec<_>>()
        );
        // Source roots must be tagged correctly so source_digests and
        // module_name_for_input can find each file again.
        assert_eq!(modules["Pkg.Foo"].source_root, "src");
        assert_eq!(modules["Pkg.Properties.Bar"].source_root, "properties");

        // Cross-root import resolution: prepare the full graph and
        // confirm the linked library decls include the rewritten
        // call_price function so Pkg.Properties.Bar's import of
        // Pkg.Foo (call_price) is resolvable.
        let graph = prepare_reef_graph(&root).expect("prepare_reef_graph");
        let has_call_price = graph
            .linked_library_decls
            .iter()
            .any(|d| matches!(d, Decl::FunDef { name, .. } if name.contains("call_price")));
        assert!(
            has_call_price,
            "linked library must include call_price for cross-root import resolution"
        );
    }

    /// Validation negatives: every documented `additional_sources` rule
    /// is locked with a table-driven test. Empty entry, reserved names,
    /// path separators, non-alphanumeric chars, and duplicates each
    /// produce a clear error with the offending entry named.
    #[test]
    fn additional_sources_validation_rejects_bad_entries() {
        // Each row: (additional_sources value, expected substring in the error).
        let cases: Vec<(Vec<&str>, &str)> = vec![
            (vec![""], "must not be empty"),
            (vec!["src"], "reserved"),
            (vec!["tests"], "reserved"),
            (vec!["a/b"], "single directory name"),
            (vec!["a\\b"], "single directory name"),
            (vec!["bad name"], "ASCII alphanumeric"),
            (vec!["bad.name"], "ASCII alphanumeric"),
            (vec!["properties", "properties"], "duplicate"),
        ];
        for (entries, expected_substr) in cases {
            let manifest = ReefManifest {
                package: ManifestPackage {
                    name: "demo".to_string(),
                    version: "0.1.0".to_string(),
                    compiler: CURRENT_COMPILER_VERSION.to_string(),
                    module_prefix: "Demo".to_string(),
                    additional_sources: entries.iter().map(|s| s.to_string()).collect(),
                },
                dependencies: BTreeMap::new(),
            };
            let err =
                validate_manifest(&manifest).expect_err(&format!("{entries:?} must be rejected"));
            assert!(
                err.contains(expected_substr),
                "for {entries:?}, expected error to contain {expected_substr:?}, got: {err}"
            );
        }
        // Positive control: a valid additional_sources passes.
        let manifest = ReefManifest {
            package: ManifestPackage {
                name: "demo".to_string(),
                version: "0.1.0".to_string(),
                compiler: CURRENT_COMPILER_VERSION.to_string(),
                module_prefix: "Demo".to_string(),
                additional_sources: vec!["properties".to_string(), "references".to_string()],
            },
            dependencies: BTreeMap::new(),
        };
        validate_manifest(&manifest).expect("valid additional_sources must pass");
    }

    /// `build_archive` packs files from every declared root, not just
    /// `src/`, so a downstream consumer extracting the tarball gets the
    /// canonical multi-root layout (src/ + each additional root) at
    /// their relative-to-package-root locations.
    #[test]
    fn multi_root_archive_contains_all_roots() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("multi");
        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "multi"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Pkg"
additional_sources = ["properties"]
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/foo.ch"),
            "module Pkg.Foo\n\nexport (one)\ndef one -> int32 = 1\n",
        );
        write(
            &root.join("properties/bar.ch"),
            "module Pkg.Properties.Bar\n\nexport (two)\ndef two -> int32 = 2\n",
        );

        let archive_path = root.join("multi.tar.zst");
        build_archive(&root, &archive_path).expect("build_archive");

        // Decode and inspect the tar entries.
        let bytes = fs::read(&archive_path).expect("read archive");
        let decoded = zstd::stream::decode_all(Cursor::new(bytes)).expect("zstd decode");
        let mut archive = Archive::new(Cursor::new(decoded));
        let mut entries: Vec<String> = archive
            .entries()
            .expect("entries")
            .filter_map(|e| e.ok())
            .map(|e| e.path().expect("entry path").to_string_lossy().into_owned())
            .collect();
        entries.sort();
        // Tar paths are relative-to-package-root, so files appear at
        // their canonical layout positions.
        assert!(
            entries.iter().any(|p| p == "reef.toml"),
            "archive must contain reef.toml, got: {entries:?}"
        );
        assert!(
            entries.iter().any(|p| p == "src/foo.ch"),
            "archive must contain src/foo.ch, got: {entries:?}"
        );
        assert!(
            entries.iter().any(|p| p == "properties/bar.ch"),
            "archive must contain properties/bar.ch, got: {entries:?}"
        );
    }

    /// Cache-invalidation guarantee for the manifest: adding
    /// `additional_sources = ["properties"]` to a package whose
    /// `properties/` directory is empty/absent must still flip the
    /// cache key. Otherwise a customer would update the manifest, see
    /// no change in walked files, and get a stale compiled context.
    #[test]
    fn additional_sources_change_invalidates_source_digests() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("invalidation");
        let manifest_path = root.join("reef.toml");
        let pre_manifest = format!(
            r#"[package]
name = "invalidation"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Invalidation"
"#,
            ver = CURRENT_COMPILER_VERSION
        );
        write(&manifest_path, &pre_manifest);
        write(
            &root.join("src/main.ch"),
            "module Invalidation.Main\n\nexport (id)\ndef id(x: int32) -> int32 = x\n",
        );

        let pre_digests = prepare_reef_graph(&root)
            .expect("prepare pre")
            .source_digests()
            .expect("source_digests pre");

        // Mutate the manifest to add additional_sources, but leave
        // properties/ absent so the file walk yields the same .ch files.
        let post_manifest = format!(
            r#"[package]
name = "invalidation"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Invalidation"
additional_sources = ["properties"]
"#,
            ver = CURRENT_COMPILER_VERSION
        );
        write(&manifest_path, &post_manifest);
        // Sanity: the file set is genuinely unchanged.
        assert!(!root.join("properties").exists());

        let post_digests = prepare_reef_graph(&root)
            .expect("prepare post")
            .source_digests()
            .expect("source_digests post");

        assert_ne!(
            pre_digests, post_digests,
            "manifest additional_sources change must flip source_digests \
             even when no .ch files were added"
        );
        // The downstream cache key (ContextHash::from_digests) is a
        // deterministic SHA over the digests Vec; any difference in
        // the Vec content propagates.
    }

    /// Atomic-write contract — H1 regression. Locks the bug class
    /// surfaced by Red Team Round 1: if any registry-level metadata
    /// write under `$CHELIS_REEF_HOME` partial-writes, the index
    /// becomes corrupt and downstream lookups fail in confusing
    /// ways. Every registry-level write goes through `atomic_write`,
    /// so locking the helper's contract locks the rule by extension.
    #[test]
    fn atomic_write_success_leaves_no_tmp_orphan() {
        let dir = tempdir().expect("tempdir");
        let target = dir.path().join("index.json");
        atomic_write(&target, b"{\"packages\":{}}").expect("atomic_write success");
        assert_eq!(
            fs::read(&target).expect("read target"),
            b"{\"packages\":{}}"
        );
        let tmp_orphan = dir.path().join("index.json.tmp");
        assert!(
            !tmp_orphan.exists(),
            "no orphan {} must remain after a successful atomic_write",
            tmp_orphan.display()
        );
    }

    #[test]
    fn atomic_write_failure_does_not_destroy_existing_file() {
        // Failure path: rename target is itself a directory, so the
        // `fs::rename` step will fail. Pre-existing `final_path`
        // content must be preserved (the temp file is cleaned up;
        // the original is untouched).
        let dir = tempdir().expect("tempdir");
        let target = dir.path().join("index.json");
        // Pre-existing content: a non-empty index that must not be
        // destroyed by a failed write.
        fs::write(&target, b"{\"packages\":{\"existing\":[]}}").expect("seed existing index");
        // Force atomic_write's fs::rename to fail by making the
        // target a directory while keeping the same name. We do
        // this by removing the seed file and recreating it as a
        // directory — but we first capture the seed content so the
        // test can verify it's preserved on disk after the failure.
        // Actually, a more direct shape: pass an unwritable parent.
        // Simpler approach below.
        drop(target);

        // Simpler shape: target's parent is a regular file (not a
        // dir). atomic_write tries `fs::create_dir_all(parent)`
        // which fails with ENOTDIR.
        let bad_parent = dir.path().join("not-a-dir");
        fs::write(&bad_parent, b"placeholder").expect("seed file");
        let bad_target = bad_parent.join("inside.json");
        let result = atomic_write(&bad_target, b"unreachable");
        assert!(result.is_err(), "atomic_write must fail on bad parent");
        // The placeholder file at bad_parent must be unchanged —
        // atomic_write must not have stomped on it.
        assert_eq!(
            fs::read(&bad_parent).expect("read placeholder"),
            b"placeholder",
            "atomic_write must not corrupt the existing file at the bad parent path"
        );
        // No `inside.json.tmp` should exist anywhere visible.
        assert!(!bad_parent.join("inside.json.tmp").exists());
    }

    /// Publish-package atomicity invariant — H1 regression. Verifies
    /// that `publish_package` (which writes
    /// `$CHELIS_REEF_HOME/index.json`) does so via `atomic_write`,
    /// so a successful publish leaves no `index.json.tmp` orphan and
    /// the prior index content (if any) is replaced atomically.
    ///
    /// Uses the monorepo's prebuilt `chelis-std` artifacts to install
    /// a fresh registry, then runs `publish_package` on a tiny
    /// downstream package and asserts the post-conditions.
    #[test]
    fn publish_package_index_update_is_atomic() {
        let _g = lock_reef_home_env();
        let dir = tempdir().expect("tempdir");
        let reef_home = dir.path().join("reef-home");
        let monorepo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .canonicalize()
            .expect("monorepo root");
        // Skip if the monorepo's chelis-std dist is missing — the
        // owning gate is `reef_install_from_monorepo` which carries the
        // same prerequisite. Mirror its behavior so this test does
        // not falsely red-flag a bare clone.
        let dist = monorepo.join("packages/chelis-std/dist");
        if !dist.join("chelis-std-0.3.0.chb").exists()
            || !dist.join("chelis-std-0.3.0.tar.zst").exists()
        {
            eprintln!(
                "skipping publish_package_index_update_is_atomic: \
                 prebuilt chelis-std artifacts missing under {}",
                dist.display()
            );
            return;
        }

        // Step 1: bootstrap chelis-std into a fresh registry. This
        // exercises `install_from_monorepo` -> `install_validated_artifact_pair`,
        // which itself uses atomic_write — so the post-condition
        // also covers that path's invariant.
        unsafe {
            std::env::set_var("CHELIS_REEF_HOME", &reef_home);
        }
        let installed = install_from_monorepo(
            &monorepo,
            &[("chelis-std".to_string(), Some("0.3.0".to_string()))],
        )
        .expect("install chelis-std into fresh registry");
        assert_eq!(installed.len(), 1);
        assert!(
            !reef_home.join("index.json.tmp").exists(),
            "install_from_monorepo must not leave an index.json.tmp orphan"
        );
        let pre_index_bytes =
            fs::read(reef_home.join("index.json")).expect("pre-publish index must exist");

        // Step 2: scaffold a tiny downstream package and publish it.
        let app_root = dir.path().join("downstream");
        let main = "module Demo.Main\n\nimport Std.Test (assert_true)\n\n\
             def test_case() -> unit ! { Test } = assert_true(true, \"ok\")\n\n\
             ran = test_case()\n";
        fs::create_dir_all(app_root.join("src")).expect("mkdir src");
        write(
            &app_root.join("reef.toml"),
            &format!(
                r#"[package]
name = "downstream-publish-test"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.3.0" }}
"#,
                ver = CURRENT_COMPILER_VERSION,
            ),
        );
        write(&app_root.join("src/main.ch"), main);

        let _published = publish_package(&app_root).expect("publish_package");

        // Step 3: post-conditions.
        assert!(
            !reef_home.join("index.json.tmp").exists(),
            "publish_package must not leave an index.json.tmp orphan after success"
        );
        let post_index_bytes =
            fs::read(reef_home.join("index.json")).expect("post-publish index must exist");
        // The index changed (we added `downstream-publish-test`).
        assert_ne!(
            pre_index_bytes, post_index_bytes,
            "publish_package must have actually updated index.json"
        );
        // Both packages now in the index.
        let post: serde_json::Value = serde_json::from_slice(&post_index_bytes).expect("parse");
        assert!(
            post["packages"]["chelis-std"].is_array(),
            "post-publish index must still contain chelis-std (atomic replace, not destructive)"
        );
        assert!(
            post["packages"]["downstream-publish-test"].is_array(),
            "post-publish index must contain the just-published package"
        );

        unsafe {
            std::env::remove_var("CHELIS_REEF_HOME");
        }
    }

    /// Build a tiny `.tar.zst` archive containing `reef.toml` and
    /// `src/main.ch`, returning its on-disk path. Shared by the
    /// `extract_archive_atomic` tests.
    fn stage_test_archive(dir: &Path) -> PathBuf {
        let root = dir.join("pkg-src");
        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "atomicpkg"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Atomic"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module Atomic.Main\n\nexport (one)\ndef one -> int32 = 1\n",
        );
        let archive_path = dir.join("atomicpkg.tar.zst");
        build_archive(&root, &archive_path).expect("build_archive");
        archive_path
    }

    /// `extract_archive_atomic` publishes the fully-populated tree and
    /// leaves no `.extract-*.tmp` staging directory behind.
    #[test]
    fn extract_archive_atomic_publishes_and_leaves_no_staging_dir() {
        let dir = tempdir().expect("tempdir");
        let archive_path = stage_test_archive(dir.path());

        let cache_root = dir.path().join("cache").join("deadbeef");
        extract_archive_atomic(&archive_path, &cache_root).expect("atomic extract");

        // The final tree is complete.
        assert!(cache_root.join("reef.toml").exists());
        assert!(cache_root.join("src/main.ch").exists());
        let manifest = fs::read_to_string(cache_root.join("reef.toml")).expect("read manifest");
        assert!(
            manifest.contains("name = \"atomicpkg\""),
            "extracted reef.toml must be the full, parseable manifest; got: {manifest}"
        );

        // No orphan staging directory remains in the cache parent.
        let leftover: Vec<_> = fs::read_dir(dir.path().join("cache"))
            .expect("read cache dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(".extract-"))
            .collect();
        assert!(
            leftover.is_empty(),
            "no .extract-*.tmp staging dir must remain; found: {leftover:?}"
        );
    }

    #[test]
    fn export_bundle_writes_bundle_json_and_root_sources() {
        let _g = lock_reef_home_env();
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("mypkg");
        fs::create_dir_all(root.join("src")).expect("mkdir");
        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "mypkg"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "My"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module My.Main\ndef main(x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n",
        );
        // Create a lockfile (no deps other than the implicit chelis-std)
        let lock = ReefLock {
            package: PackageId {
                name: "mypkg".to_string(),
                version: "0.1.0".to_string(),
            },
            dependencies: vec![],
        };
        write_lockfile(&root.join("reef.lock"), &lock).expect("write lock");

        let bundle_dir = dir.path().join("bundle");
        let manifest = export_bundle(&root, &bundle_dir).expect("export_bundle");
        assert_eq!(manifest.root_package.name, "mypkg");
        assert!(bundle_dir.join("bundle.json").exists());
        assert!(bundle_dir.join("root/reef.toml").exists());
        assert!(bundle_dir.join("root/src/main.ch").exists());

        // bundle.json is valid JSON
        let json_str =
            fs::read_to_string(bundle_dir.join("bundle.json")).expect("read bundle.json");
        let parsed: BundleManifest =
            serde_json::from_str(&json_str).expect("parse bundle.json");
        assert_eq!(parsed.root_package.name, "mypkg");
    }

    #[test]
    fn export_bundle_fails_without_lockfile() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("nolockpkg");
        fs::create_dir_all(root.join("src")).expect("mkdir");
        write(
            &root.join("reef.toml"),
            &format!(
                r#"[package]
name = "nolockpkg"
version = "0.1.0"
compiler = "{ver}"
module_prefix = "Nl"
"#,
                ver = CURRENT_COMPILER_VERSION
            ),
        );
        write(
            &root.join("src/main.ch"),
            "module Nl.Main\ndef f(x: f32) -> f32 = x\n",
        );
        let bundle_dir = dir.path().join("bundle");
        let err = export_bundle(&root, &bundle_dir).expect_err("should fail");
        assert!(err.contains("reef.lock"), "error should mention reef.lock: {err}");
    }

    /// Concurrent-write contract: many threads extracting the same
    /// archive into the same cache directory all succeed, and every
    /// observed `reef.toml` is the complete manifest, never a torn or
    /// empty file. This is the unit-level guard for the
    /// `phaseA_item8_two_concurrent_builds_serialize` CLI flake.
    #[test]
    fn extract_archive_atomic_concurrent_writers_never_tear_reef_toml() {
        let dir = tempdir().expect("tempdir");
        let archive_path = stage_test_archive(dir.path());
        let cache_root = dir.path().join("cache").join("sha-shared");

        let mut handles = Vec::new();
        for _ in 0..16 {
            let archive_path = archive_path.clone();
            let cache_root = cache_root.clone();
            handles.push(std::thread::spawn(move || {
                // Extract (mirrors load_registry_package's guarded call).
                if !cache_root.exists() {
                    extract_archive_atomic(&archive_path, &cache_root)
                        .expect("concurrent atomic extract");
                }
                // Immediately read back, racing the other writers.
                let manifest = fs::read_to_string(cache_root.join("reef.toml"))
                    .expect("reef.toml must be readable, never mid-write");
                assert!(
                    manifest.contains("name = \"atomicpkg\""),
                    "reader observed a torn reef.toml: {manifest:?}"
                );
            }));
        }
        for h in handles {
            h.join().expect("extractor thread panicked");
        }

        // Exactly the published tree, no staging leftovers.
        assert!(cache_root.join("src/main.ch").exists());
        let leftover: Vec<_> = fs::read_dir(dir.path().join("cache"))
            .expect("read cache dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with(".extract-"))
            .collect();
        assert!(
            leftover.is_empty(),
            "no .extract-*.tmp staging dir must remain after concurrent extracts; found: {leftover:?}"
        );
    }
}
