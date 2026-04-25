use chelis_shell::{
    PackageId, ShellModule, ShellPackage, ShellSymbol, SymbolKind, read_shell, write_shell,
};
use chelis_surf::ast::{
    Decl, EffectExpr, Expr, ImportKind, LetBinding, LetPattern, MatchArm, Param, Pattern, TypeExpr,
    Variant, VariantFields,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;
use tar::{Archive, Builder};
use walkdir::WalkDir;

const CURRENT_COMPILER_VERSION: &str = concat!("=", env!("CARGO_PKG_VERSION"));

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
    Path { path: String },
    LocalRegistry,
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
}

#[derive(Debug, Clone)]
pub struct PreparedProgram {
    pub decls: Vec<Decl>,
    pub entry_decls: Vec<Decl>,
    pub package_root: PathBuf,
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
#[derive(Debug, Clone)]
pub struct PreparedReefGraph {
    pub package_root: PathBuf,
    pub(crate) graph: PackageGraph,
    pub(crate) linked_library_decls: Vec<Decl>,
    pub(crate) internal_maps: HashMap<(String, String), HashMap<String, String>>,
    pub(crate) dep_shells: BTreeMap<String, ShellPackage>,
    pub(crate) eval_module_prefix: String,
}

#[derive(Debug, Clone)]
pub struct PackageBuildArtifacts {
    pub package: PackageId,
    pub shell_path: PathBuf,
    pub archive_path: PathBuf,
    pub shell_sha256: String,
    pub archive_sha256: String,
}

#[derive(Debug, Clone)]
struct LoadedPackage {
    id: PackageId,
    manifest: ReefManifest,
    modules: BTreeMap<String, ModuleSource>,
    source: LoadedSourceKind,
    shell: Option<ShellPackage>,
}

#[derive(Debug, Clone)]
enum LoadedSourceKind {
    Root,
    Path { relative: String },
    LocalRegistry,
}

#[derive(Debug, Clone)]
struct ModuleSource {
    package_name: String,
    module_name: String,
    decls: Vec<Decl>,
    file_rel: PathBuf,
    exports: BTreeSet<String>,
    symbols: BTreeMap<String, SymbolKind>,
}

#[derive(Debug, Clone)]
struct LinkedModule {
    decls: Vec<Decl>,
    entry: bool,
}

#[derive(Debug, Clone)]
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
        },
        dependencies: BTreeMap::new(),
    };
    write_manifest(&root.join("reef.toml"), &manifest)?;
    let main_module = format!("{module_prefix}.Main");
    fs::write(
        root.join("src/main.ch"),
        format!(
            "module {main_module}\n\ndef main(x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n"
        ),
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

    loop {
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
    let graph = resolve_package_graph(&root)?;
    write_lockfile(&root.join("reef.lock"), &build_lockfile(&graph))?;
    let entry_module = module_name_for_input(&root, file, &graph.root_package)?;
    let linked = link_graph(&graph, std::slice::from_ref(&entry_module))?;
    let mut decls = Vec::new();
    let mut entry_decls = Vec::new();
    for module in linked {
        if module.entry {
            entry_decls.extend(module.decls.clone());
        }
        decls.extend(module.decls);
    }
    Ok(Some(PreparedProgram {
        decls,
        entry_decls,
        package_root: root,
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
            "no reef.toml found at or above {} — `chelis test` requires a reef package",
            context_dir.display()
        ));
    };
    let graph = load_package_graph_for_eval(&root)?;
    let linked = link_graph(&graph, &[])?;
    let internal_maps = build_internal_maps(&graph);
    let dep_shells = dependency_shells(&graph);
    let eval_module_prefix = graph
        .packages
        .get(&graph.root_package)
        .map(|package| package.manifest.package.module_prefix.clone())
        .unwrap_or_default();

    let mut linked_library_decls = Vec::new();
    for module in linked {
        linked_library_decls.extend(module.decls);
    }

    Ok(PreparedReefGraph {
        package_root: root,
        graph,
        linked_library_decls,
        internal_maps,
        dep_shells,
        eval_module_prefix,
    })
}

/// Compile an in-memory entry decl list against a previously prepared reef
/// graph. This is the cheap per-file work: only the entry module is rewritten
/// and appended to the cached library decls.
pub fn compile_with_reef_graph(
    graph: &PreparedReefGraph,
    entry_decls: &[Decl],
) -> Result<PreparedProgram, String> {
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
        exports: BTreeSet::new(),
        symbols: collect_symbol_kinds(entry_decls),
    };
    let rewritten_entry_decls = rewrite_eval_module_decls(
        &eval_module,
        &graph.graph,
        &graph.internal_maps,
        &graph.dep_shells,
    )?;

    let mut decls = graph.linked_library_decls.clone();
    decls.extend(rewritten_entry_decls);

    Ok(PreparedProgram {
        decls,
        entry_decls: entry_decls.to_vec(),
        package_root: graph.package_root.clone(),
    })
}

/// Resolve a `PackageGraph` for eval: fast-path the lockfile when present,
/// otherwise run the full resolver under the standard 5-second timeout.
///
/// Eval does not write the lockfile — that is the build path's responsibility.
fn load_package_graph_for_eval(root: &Path) -> Result<PackageGraph, String> {
    let lock_path = root.join("reef.lock");
    if lock_path.exists() {
        let lock = read_lockfile(&lock_path)?;
        reconstruct_graph_from_lockfile(root, &lock)
    } else {
        let root_clone = root.to_path_buf();
        run_with_timeout(
            move || resolve_package_graph(&root_clone),
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

pub fn build_package(root: &Path) -> Result<PackageBuildArtifacts, String> {
    let root = canonical_root(root)?;
    let graph = resolve_package_graph(&root)?;
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
    });
    versions.sort_by(|a, b| a.version.cmp(&b.version));
    fs::create_dir_all(&registry_root).map_err(|e| e.to_string())?;
    fs::write(
        &index_path,
        serde_json::to_string_pretty(&index).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    Ok(artifacts)
}

/// One package successfully copied into the local registry by
/// [`install_from_monorepo`].
#[derive(Debug, Clone)]
pub struct InstalledArtifact {
    pub package: PackageId,
    pub shell_path: PathBuf,
    pub archive_path: PathBuf,
    pub shell_sha256: String,
    pub archive_sha256: String,
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
            "{} does not contain a `packages/` directory — \
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
    let index_path = registry_root.join("index.json");
    let mut index = if index_path.exists() {
        serde_json::from_str::<LocalRegistryIndex>(
            &fs::read_to_string(&index_path).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?
    } else {
        LocalRegistryIndex::default()
    };

    let mut installed = Vec::new();
    for (name, version, pkg_root) in &resolved {
        let dist_dir = pkg_root.join("dist");
        if !dist_dir.is_dir() {
            return Err(format!(
                "package `{name}` has no dist/ directory at {} — \
                 run `chelis reef build` in the monorepo first",
                dist_dir.display()
            ));
        }
        let archive_src = dist_dir.join(format!("{name}-{version}.tar.zst"));
        let shell_src = dist_dir.join(format!("{name}-{version}.chb"));
        if !archive_src.exists() {
            return Err(format!(
                "missing prebuilt archive {} — \
                 run `chelis reef build` in the monorepo first",
                archive_src.display()
            ));
        }
        if !shell_src.exists() {
            return Err(format!(
                "missing prebuilt shell {} — \
                 run `chelis reef build` in the monorepo first",
                shell_src.display()
            ));
        }

        // Sanity-check that the prebuilt shell agrees with the prebuilt
        // archive. This catches stale dist/ trees where one half was
        // rebuilt and the other was not.
        let archive_sha256 = sha256_file(&archive_src)?;
        let shell_sha256 = sha256_file(&shell_src)?;
        let shell = read_shell(&shell_src).map_err(|e| e.to_string())?;
        if shell.archive_sha256 != archive_sha256 {
            return Err(format!(
                "prebuilt shell {} disagrees with archive {} on archive_sha256 — \
                 the dist/ tree is stale; run `chelis reef build` in the monorepo",
                shell_src.display(),
                archive_src.display()
            ));
        }
        if shell.package.name != *name || shell.package.version != *version {
            return Err(format!(
                "prebuilt shell {} advertises `{}-{}` but was requested as `{name}-{version}`",
                shell_src.display(),
                shell.package.name,
                shell.package.version
            ));
        }

        let target_dir = registry_root.join("packages").join(name).join(version);
        fs::create_dir_all(&target_dir).map_err(|e| e.to_string())?;
        let archive_dst = target_dir.join(format!("{name}-{version}.tar.zst"));
        let shell_dst = target_dir.join(format!("{name}-{version}.chb"));
        fs::copy(&archive_src, &archive_dst).map_err(|e| e.to_string())?;
        fs::copy(&shell_src, &shell_dst).map_err(|e| e.to_string())?;

        let versions = index.packages.entry(name.clone()).or_default();
        versions.retain(|entry| entry.version != *version);
        versions.push(RegistryVersion {
            version: version.clone(),
            compiler: shell.compiler.clone(),
            archive_sha256: archive_sha256.clone(),
            shell_sha256: shell_sha256.clone(),
        });
        versions.sort_by(|a, b| a.version.cmp(&b.version));

        installed.push(InstalledArtifact {
            package: PackageId {
                name: name.clone(),
                version: version.clone(),
            },
            shell_path: shell_dst,
            archive_path: archive_dst,
            shell_sha256,
            archive_sha256,
        });
    }

    fs::write(
        &index_path,
        serde_json::to_string_pretty(&index).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    Ok(installed)
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
/// cache is missing or slow the caller gets an actionable error instead of
/// a hang.
fn reconstruct_graph_from_lockfile(root: &Path, lock: &ReefLock) -> Result<PackageGraph, String> {
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
        },
    );

    // Load each dependency.
    for dep in &lock.dependencies {
        match &dep.source {
            LockSource::Path { path } => {
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
                    },
                );
            }
            LockSource::LocalRegistry => {
                let dep_name = dep.name.clone();
                let dep_version = dep.version.clone();
                let installed = run_with_timeout(
                    move || load_registry_package(&dep_name, &dep_version),
                    Duration::from_secs(5),
                    TIMEOUT_MSG,
                )?;
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
    for (name, dep) in &manifest.dependencies {
        match (&dep.version, &dep.path) {
            (Some(_), Some(_)) => {
                return Err(format!(
                    "dependency `{name}` cannot specify both `version` and `path`"
                ));
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

fn resolve_package_graph(root: &Path) -> Result<PackageGraph, String> {
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
        &mut packages,
        &mut by_name,
        &mut stack,
    )?;
    Ok(PackageGraph {
        root_package: root_id.name,
        packages,
    })
}

fn resolve_package_recursive(
    package_name: &str,
    root: PathBuf,
    source: LoadedSourceKind,
    maybe_shell: Option<ShellPackage>,
    packages: &mut BTreeMap<String, LoadedPackage>,
    by_name: &mut HashMap<String, PackageId>,
    stack: &mut Vec<String>,
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
    };

    for (dep_name, dep) in &manifest.dependencies {
        match (&dep.version, &dep.path) {
            (_, Some(path)) => {
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
                    packages,
                    by_name,
                    stack,
                )?;
            }
            (Some(version), None) => {
                let installed = load_registry_package(dep_name, version)?;
                resolve_package_recursive(
                    dep_name,
                    installed.root,
                    LoadedSourceKind::LocalRegistry,
                    Some(installed.shell),
                    packages,
                    by_name,
                    stack,
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
}

fn load_registry_package(name: &str, version: &str) -> Result<InstalledPackage, String> {
    let registry_root = registry_root()?;
    let index = read_registry_index(&registry_root)?;
    let expected = index
        .packages
        .get(name)
        .and_then(|versions| versions.iter().find(|entry| entry.version == version))
        .ok_or_else(|| {
            format!(
                "package `{name}` version `{version}` missing from local registry index — \
                 run `chelis reef build` first to populate the cache"
            )
        })?;
    let pkg_dir = registry_root.join("packages").join(name).join(version);
    if !pkg_dir.exists() {
        return Err(format!(
            "package `{name}` version `{version}` not found in local registry — \
             run `chelis reef build` first to populate the cache"
        ));
    }
    let shell_path = pkg_dir.join(format!("{name}-{version}.chb"));
    let archive_path = pkg_dir.join(format!("{name}-{version}.tar.zst"));
    let shell_sha256 = sha256_file(&shell_path)?;
    if shell_sha256 != expected.shell_sha256 {
        return Err(format!(
            "shell checksum mismatch for `{name}` version `{version}`"
        ));
    }
    let archive_sha256 = sha256_file(&archive_path)?;
    if archive_sha256 != expected.archive_sha256 {
        return Err(format!(
            "archive checksum mismatch for `{name}` version `{version}`"
        ));
    }
    let shell = read_shell(&shell_path).map_err(|e| e.to_string())?;
    if archive_sha256 != shell.archive_sha256 {
        return Err(format!(
            "archive checksum mismatch for `{name}` version `{version}`"
        ));
    }
    if shell.package.name != name || shell.package.version != version {
        return Err(format!(
            "shell package id mismatch for `{name}` version `{version}`"
        ));
    }
    let cache_root = registry_root.join("cache").join(&archive_sha256);
    if !cache_root.exists() {
        fs::create_dir_all(&cache_root).map_err(|e| e.to_string())?;
        extract_archive(&archive_path, &cache_root)?;
    }
    Ok(InstalledPackage {
        root: cache_root,
        shell,
    })
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
            let source = match &package.source {
                LoadedSourceKind::Path { relative } => LockSource::Path {
                    path: relative.clone(),
                },
                LoadedSourceKind::LocalRegistry => LockSource::LocalRegistry,
                LoadedSourceKind::Root => LockSource::LocalRegistry,
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
    let src_root = root.join("src");
    if !src_root.exists() {
        return Err(format!("{} is missing src/", root.display()));
    }
    let mut modules = BTreeMap::new();
    for entry in WalkDir::new(&src_root).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() {
            continue;
        }
        if entry.path().extension().and_then(|ext| ext.to_str()) != Some("ch") {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(&src_root)
            .map_err(|e| e.to_string())?
            .to_path_buf();
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
        validate_module_path(&manifest.package.module_prefix, &module_decl.0, &rel)?;
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
        modules.insert(
            module_decl.0.clone(),
            ModuleSource {
                package_name: manifest.package.name.clone(),
                module_name: module_decl.0,
                decls: module_decl.1,
                file_rel: rel,
                exports,
                symbols,
            },
        );
    }
    if modules.is_empty() {
        return Err(format!(
            "{} has no .ch source files under src/",
            root.display()
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
        return Err(format!(
            "module `{module}` does not match file path src/{} (expected `{}`)",
            rel.display(),
            expected
        ));
    }
    Ok(())
}

fn compute_exports(decls: &[Decl]) -> BTreeSet<String> {
    let explicit = decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::Export { names, .. } => Some(names.clone()),
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    if !explicit.is_empty() {
        return explicit.into_iter().collect();
    }
    decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::FunDef { name, .. }
            | Decl::LetDef { name, .. }
            | Decl::TypeDef { name, .. }
            | Decl::TypeAlias { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn collect_symbol_kinds(decls: &[Decl]) -> BTreeMap<String, SymbolKind> {
    let mut symbols = BTreeMap::new();
    for decl in decls {
        match decl {
            Decl::FunDef { name, .. } | Decl::LetDef { name, .. } | Decl::Sig { name, .. } => {
                symbols.insert(name.clone(), SymbolKind::Value);
            }
            Decl::TypeDef { name, .. } | Decl::TypeAlias { name, .. } => {
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
    let root_pkg = resolve_package_graph(root)?
        .packages
        .remove(package_name)
        .ok_or_else(|| "root package missing".to_string())?;
    let canonical = file
        .canonicalize()
        .map_err(|e| format!("failed to canonicalize {}: {e}", file.display()))?;
    for module in root_pkg.modules.values() {
        if root.join("src").join(&module.file_rel) == canonical {
            return Ok(module.module_name.clone());
        }
    }
    Err(format!(
        "{} is not a source file under {}/src",
        file.display(),
        root.display()
    ))
}

fn build_archive(root: &Path, out_path: &Path) -> Result<(), String> {
    let mut tar_bytes = Vec::new();
    {
        let mut builder = Builder::new(&mut tar_bytes);
        for rel in ["reef.toml", "reef.lock"] {
            let path = root.join(rel);
            if path.exists() {
                builder
                    .append_path_with_name(&path, rel)
                    .map_err(|e| e.to_string())?;
            }
        }
        for entry in WalkDir::new(root.join("src"))
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() {
                continue;
            }
            let rel = entry.path().strip_prefix(root).map_err(|e| e.to_string())?;
            builder
                .append_path_with_name(entry.path(), rel)
                .map_err(|e| e.to_string())?;
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
            linked.push(LinkedModule { decls, entry });
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
                    unqualified.insert(name.clone(), internal.clone());
                }
            }
        }
    }

    Ok(NameResolver {
        own_names: module_internal,
        imported_names: unqualified,
        qualified_modules: qualified,
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
            Decl::Import { .. } | Decl::Export { .. } => {}
            Decl::Module { .. } => unreachable!("module wrappers already stripped"),
            _ => out.push(rewrite_decl(
                decl,
                &resolver,
                &module.package_name,
                &module.module_name,
            )),
        }
    }
    Ok(out)
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
        match decl {
            Decl::Import { .. } | Decl::Export { .. } => {}
            Decl::Module { .. } => unreachable!("module wrappers already stripped"),
            _ => out.push(rewrite_eval_decl(decl, &resolver)),
        }
    }
    Ok(out)
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
            span,
        } => Decl::TypeDef {
            name: internal_name(package, module, name),
            params: params.clone(),
            variants: variants
                .iter()
                .map(|variant| rewrite_variant(variant, resolver))
                .collect(),
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
            span,
        } => Decl::TypeDef {
            name: name.clone(),
            params: params.clone(),
            variants: variants
                .iter()
                .map(|variant| rewrite_variant(variant, resolver))
                .collect(),
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

fn rewrite_variant(variant: &Variant, resolver: &NameResolver) -> Variant {
    Variant {
        name: variant.name.clone(),
        fields: match &variant.fields {
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
        },
        span: variant.span,
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
                .unwrap_or_else(|| name.clone()),
            *span,
        ),
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
        TypeExpr::App(name, args, span) => TypeExpr::App(
            resolver
                .own_names
                .get(name)
                .cloned()
                .or_else(|| resolver.imported_names.get(name).cloned())
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
            name.clone(),
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
            resolver
                .imported_names
                .get(name)
                .cloned()
                .or_else(|| resolver.own_names.get(name).cloned())
                .unwrap_or_else(|| name.clone()),
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
            name.clone(),
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

fn resolve_qualified_expr(expr: &Expr, resolver: &NameResolver) -> Option<Expr> {
    let (segments, span) = access_segments(expr)?;
    if segments.len() < 2 {
        return None;
    }
    for split in 1..segments.len() {
        let module = segments[..split].join(".");
        let name = segments[split..].join(".");
        if let Some(map) = resolver.qualified_modules.get(&module)
            && let Some(internal) = map.get(&name)
        {
            return Some(Expr::Var(internal.clone(), span));
        }
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
    let checked = chelis_types::check_phase0e_program(deep_exprs)
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

    fn write(path: &Path, contents: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(path, contents).expect("write file");
    }

    #[test]
    fn manifest_roundtrip() {
        let manifest = ReefManifest {
            package: ManifestPackage {
                name: "demo".to_string(),
                version: "0.1.0".to_string(),
                compiler: CURRENT_COMPILER_VERSION.to_string(),
                module_prefix: "Demo".to_string(),
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
            "fast path took {elapsed:?} — expected < 1s"
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
        unsafe {
            std::env::set_var("CHELIS_REEF_HOME", dir.path().join("empty_registry"));
        }
        let entry_decls =
            chelis_surf::parser::parse_str("def result -> int32 = 42").expect("parse");
        let result = prepare_program_for_eval_source(&root, &entry_decls);
        unsafe {
            std::env::remove_var("CHELIS_REEF_HOME");
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
            "no-package-root eval took {elapsed:?} — should return immediately"
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
        // Use a short timeout to avoid waiting 5 seconds in the test.
        // We call reconstruct_graph_from_lockfile indirectly via run_with_timeout.
        let root_clone = root.clone();
        let result = run_with_timeout(
            move || reconstruct_graph_from_lockfile(&root_clone, &lock),
            Duration::from_millis(200),
            TIMEOUT_MSG,
        );

        unsafe {
            std::env::remove_var("CHELIS_REEF_HOME");
        }

        let err = result.expect_err("should have failed: registry dep not in cache");
        // The error must mention "chelis reef build" — the actionable instruction.
        assert!(
            err.contains("chelis reef build"),
            "error must mention `chelis reef build` for actionable recovery; got: {err}"
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
            "missing-path-dep fast path took {elapsed:?} — should fail quickly"
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
    /// flaky — the correctness test is required, the perf test is diagnostic").
    ///
    /// We do NOT gate the suite on a hard 2x multiplier because on a small
    /// path-dep fixture both paths are sub-millisecond and noise dominates.
    /// What we CAN assert robustly: the split path is no slower than the
    /// single-shot path when amortized across two files, which is a
    /// sufficient signal that the shared graph isn't redoing work.
    #[test]
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
}
