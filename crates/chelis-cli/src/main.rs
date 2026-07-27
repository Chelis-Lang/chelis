//! Chelis compiler CLI.

mod prove;
mod style_gate;

use chelis_compiler_api::schema::{
    EvalRequest, SourceKind, WireInferredDim, WireInferredEffect, WireInferredPrecision,
    WireInferredType,
};
use chelis_deep::DeepTag;
use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr};
use chelis_surf::ast::Decl;
use chelis_types::types::{Dim, Effect, EffectSet, TensorPrec, Type};
use clap::{ArgAction, ArgGroup, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{self, BufRead, Read, Write};
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

const RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime.h"
));
const RUNTIME_DTYPE_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime_dtype.h"
));
const BLAS_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_blas.h"
));
const SIMD_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_simd.h"
));
const MATH_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_math.h"
));
const HIP_RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-hip/runtime/chelis_hip_runtime.h"
));
const METAL_RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-metal/runtime/chelis_metal_runtime.h"
));

fn find_runtime_library() -> Result<PathBuf, Box<dyn std::error::Error>> {
    const LIB_NAME: &str = "libchelis_runtime.a";
    const LIB_PREFIX: &str = "libchelis_runtime";

    fn find_in_dir(dir: &Path) -> Option<PathBuf> {
        let mut hashed_matches = Vec::new();
        let exact = dir.join(LIB_NAME);
        let entries = fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if name.starts_with(LIB_PREFIX) && name.ends_with(".a") {
                if name == LIB_NAME {
                    continue;
                }
                hashed_matches.push(path);
            }
        }
        hashed_matches
            .into_iter()
            .max_by_key(|path| fs::metadata(path).and_then(|meta| meta.modified()).ok())
            .or_else(|| exact.exists().then_some(exact))
    }

    if let Ok(dir) = env::var("CHELIS_RUNTIME_DIR") {
        if let Some(candidate) = find_in_dir(&PathBuf::from(&dir)) {
            return Ok(candidate);
        }
        return Err(format!(
            "cannot find {LIB_NAME} in CHELIS_RUNTIME_DIR; set CHELIS_RUNTIME_DIR to the directory containing the chelis runtime static library"
        )
        .into());
    }

    let exe = env::current_exe()?;
    let exe_dir = exe
        .parent()
        .ok_or("cannot determine chelis executable directory")?;
    for candidate_dir in [
        exe_dir.join("deps"),
        exe_dir.to_path_buf(),
        exe_dir.join("lib"),
        exe_dir.parent().map(|p| p.join("deps")).unwrap_or_default(),
        exe_dir
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default(),
        exe_dir.parent().map(|p| p.join("lib")).unwrap_or_default(),
    ] {
        if !candidate_dir.as_os_str().is_empty()
            && let Some(found) = find_in_dir(&candidate_dir)
        {
            return Ok(found);
        }
    }

    Err(format!(
        "cannot find {LIB_NAME}; set CHELIS_RUNTIME_DIR or install chelis so {LIB_NAME} is available relative to the chelis executable"
    )
    .into())
}

#[derive(Clone, Copy, Default)]
struct ExtraRuntimeArtifacts {
    hip: bool,
    metal: bool,
}

fn make_existing_copy_destination_writable(path: &Path) -> std::io::Result<()> {
    let mut permissions = match fs::metadata(path) {
        Ok(metadata) => metadata.permissions(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if !permissions.readonly() {
        return Ok(());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(permissions.mode() | 0o200);
    }
    #[cfg(not(unix))]
    permissions.set_readonly(false);
    fs::set_permissions(path, permissions)
}

fn copy_runtime_artifacts(
    runtime_dir: &Path,
    extras: ExtraRuntimeArtifacts,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::write(runtime_dir.join("chelis_runtime.h"), RUNTIME_H)?;
    fs::write(runtime_dir.join("chelis_runtime_dtype.h"), RUNTIME_DTYPE_H)?;
    fs::write(runtime_dir.join("chelis_blas.h"), BLAS_H)?;
    fs::write(runtime_dir.join("chelis_simd.h"), SIMD_H)?;
    fs::write(runtime_dir.join("chelis_math.h"), MATH_H)?;
    if extras.hip {
        fs::write(runtime_dir.join("chelis_hip_runtime.h"), HIP_RUNTIME_H)?;
    }
    if extras.metal {
        fs::write(runtime_dir.join("chelis_metal_runtime.h"), METAL_RUNTIME_H)?;
    }
    let source = find_runtime_library()?;
    let dest = runtime_dir.join("libchelis_runtime.a");
    // Rust static-library artifacts are read-only on some hosts, and
    // `fs::copy` preserves that mode. Restore owner-write before replacing a
    // previous build's staged archive so rebuilding into one directory works.
    make_existing_copy_destination_writable(&dest)?;
    fs::copy(&source, &dest)?;
    Ok(dest)
}

#[derive(Parser)]
#[command(
    name = "chelis",
    version = env!("CARGO_PKG_VERSION"),
    about = "The Chelis programming language"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Desugar Surf (.ch) to canonical Deep s-expressions
    Deep {
        #[arg(long)]
        flat: bool,
        /// Run the typechecker and print Deep with `{type: ...}` metadata
        #[arg(long)]
        annotate: bool,
        file: PathBuf,
    },
    /// Decompile Deep (.dp) to Surf (best-effort)
    Surf {
        file: PathBuf,
        #[arg(long)]
        verbose: bool,
    },
    /// Format source code (canonical form)
    Fmt {
        file: PathBuf,
        #[arg(long)]
        inplace: bool,
        #[arg(long)]
        check: bool,
    },
    /// Evaluate an expression or file
    ///
    /// `chelis eval --file FILE` runs the formatter and lint gates on FILE
    /// before the eval pipeline. `chelis eval EXPR` is a one-line snippet
    /// path with no on-disk source; the gate does not apply.
    Eval {
        /// File to evaluate
        #[arg(long)]
        file: Option<PathBuf>,
        /// Inline expression
        expr: Option<String>,
        /// Emit the raw `EvalResult` as JSON on stdout instead of the
        /// human-readable rendering. Stdout carries JSON only; warnings
        /// and errors stay on stderr. Empty-roots inputs emit
        /// `{"roots":[]}`.
        #[arg(long, action = ArgAction::SetTrue)]
        json: bool,
        /// Bypass `chelis fmt --check` and `chelis lint --check` gates.
        /// Emergency use only; CI must not pass this flag.
        #[arg(long, action = ArgAction::SetTrue)]
        allow_style_violations: bool,
    },
    /// Run front-end checks and report fitness-oriented diagnostics
    ///
    /// Before the type/effect/linearity passes, `check` enforces the same
    /// style gate `build` does — `chelis fmt --check` on the file plus
    /// every `chelis lint` rule that applies to the file's surface — so
    /// non-canonical or style-violating source is rejected up front.
    /// Pass `--allow-style-violations` to bypass the gate (CI must not).
    Check {
        file: PathBuf,
        /// Include signature-inference metadata in the JSON report.
        #[arg(long, action = ArgAction::SetTrue)]
        show_inferred: bool,
        /// Bypass `chelis fmt --check` and `chelis lint --check` gates.
        /// Emergency use only; CI must not pass this flag.
        #[arg(long, action = ArgAction::SetTrue)]
        allow_style_violations: bool,
    },
    /// Report lowered IR copy cost
    Cost {
        file: PathBuf,
        /// Emit JSON instead of human-readable text
        #[arg(long, action = ArgAction::SetTrue)]
        json: bool,
    },
    /// Validate syntax against executable grammar tooling
    #[command(group(
        ArgGroup::new("mode")
            .required(true)
            .args(["surf", "deep", "desugar"])
    ))]
    Validate {
        #[arg(long, action = ArgAction::SetTrue, group = "mode")]
        surf: bool,
        #[arg(long, action = ArgAction::SetTrue, group = "mode")]
        deep: bool,
        #[arg(long, action = ArgAction::SetTrue, group = "mode")]
        desugar: bool,
        file: PathBuf,
        /// Bypass `chelis fmt --check` and `chelis lint --check` gates.
        /// Emergency use only; CI must not pass this flag.
        #[arg(long, action = ArgAction::SetTrue)]
        allow_style_violations: bool,
    },
    /// Compile to C (default) or HIP GPU code
    ///
    /// Auto-detects the input language from the file extension: `.dp`
    /// inputs are routed through the Deep ingestion path; everything else
    /// (`.ch`, no-extension, etc.) goes through the Surf path. Pass
    /// `--deep` to force the Deep path on a non-`.dp` file. There is no
    /// `--no-deep` flag — to force a `.dp` file through Surf, rename it
    /// or pipe through `chelis surf` first. Conflict resolution rules
    /// are documented in `spec/design/chelis_span_survival.md` §2.5.
    ///
    /// Before the front-end pipeline runs, `build` enforces the style
    /// gate: `chelis fmt --check` on the file plus every `chelis lint`
    /// rule that applies to the file's surface. Style violations fail
    /// the build by default; pass `--allow-style-violations` to bypass
    /// (CI must not).
    Build {
        file: PathBuf,
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Backend target: "c" (default), "hip" (AMD GPU), or "metal" (Apple GPU)
        #[arg(long, default_value = "c")]
        target: String,
        /// Force the Deep ingestion path. Auto-on for `.dp` inputs;
        /// override for non-`.dp` inputs that happen to be Deep source.
        #[arg(long, action = ArgAction::SetTrue)]
        deep: bool,
        /// Bypass `chelis fmt --check` and `chelis lint --check` gates.
        /// Emergency use only; CI must not pass this flag.
        #[arg(long, action = ArgAction::SetTrue)]
        allow_style_violations: bool,
    },
    /// Interactive REPL, HTTP API, and MCP server
    Tide {
        #[command(subcommand)]
        command: Option<TideCommand>,
    },
    /// Launch the Cove terminal UI
    Cove {
        #[arg(long)]
        file: Option<PathBuf>,
    },
    /// Local-first Reef package management
    Reef {
        #[command(subcommand)]
        command: ReefCommand,
    },
    /// Run Chelis-native tests discovered under a `tests/` directory
    Test {
        /// Path to tests directory or a single `.ch` test file
        path: Option<PathBuf>,
        /// Substring filter on `<file>::<test_fn>`
        #[clap(long)]
        filter: Option<String>,
        /// Emit newline-delimited JSON records instead of plain text
        #[clap(long)]
        json: bool,
        /// Per-test wall-clock timeout (seconds)
        #[clap(long, default_value = "30")]
        timeout: u64,
        /// Whole-suite wall-clock timeout, including setup and finalization (seconds)
        #[clap(long, default_value = "600")]
        suite_timeout: u64,
        /// Test-file workers to run concurrently (`auto` uses available CPUs)
        #[clap(long, default_value = "auto", value_name = "N|auto")]
        jobs: TestJobs,
        /// Suite batching mode: `auto` batches eligible files, `file` keeps per-file workers
        #[clap(long, default_value = "auto", value_name = "auto|file")]
        batch_mode: TestBatchMode,
        /// Expected-failure mode: treat each `.ch` as an expected-to-fail case
        /// paired with a `.expect` sidecar (line 1 = required diagnostic
        /// substring). `neg` drives `tests_neg/`; `blocked` drives
        /// `tests_blocked/` (a pass = FIX-detected, fails loudly). Runs per-file
        /// isolated regardless of `--batch-mode`.
        #[clap(long, value_name = "neg|blocked")]
        expect: Option<ExpectArg>,
    },
    /// Run L2 property checks discovered in Surf or Deep inputs
    Prove {
        /// Path to a package, directory, `.ch`, or `.dp` input
        path: Option<PathBuf>,
        /// Filter by property name, substring, or trailing-* prefix glob
        #[clap(long)]
        only: Option<String>,
        /// Samples per property
        #[clap(long)]
        samples: Option<usize>,
        /// Deterministic run seed
        #[clap(long)]
        seed: Option<u64>,
        /// Maximum generated attempts before precondition exhaustion
        #[clap(long)]
        max_attempts: Option<usize>,
        /// Emit newline-delimited JSON records instead of plain text
        #[clap(long)]
        json: bool,
        /// Override bridge span manifest for a single `.dp` input
        #[clap(long)]
        spans: Option<PathBuf>,
        /// Verification tier: auto (A→B→C), fuzz-only, smt-only, type-only
        #[clap(long, default_value = "auto")]
        tier: String,
        /// SMT solver timeout in milliseconds (default 5000)
        #[clap(long, default_value = "5000")]
        smt_timeout: u64,
        /// Floor for invariant rejection-sampling acceptance rate before
        /// the generator-starvation classifier fires (RFC D-STARVE). 0.0
        /// disables the classifier (legacy exhaustion => error path).
        #[clap(long, default_value = "0.01")]
        invariant_min_rate: f64,
        /// Resolve imports through the reef package rooted at this path.
        /// If not set, auto-detects by walking ancestor directories for
        /// reef.toml.
        #[clap(long)]
        package: Option<PathBuf>,
        /// Print machine-readable JSON describing prove capabilities
        /// (supported tiers, SMT availability, engine status) and exit.
        #[clap(long)]
        capabilities: bool,
    },
    /// Lint naming conventions per `spec/01-nomenclature.md`
    Lint {
        /// Paths to lint. Defaults to the current directory.
        paths: Vec<PathBuf>,
        /// Exit nonzero on any violation (CI use).
        #[arg(long)]
        check: bool,
        /// Apply all available non-overlapping fixes in-place.
        #[arg(long)]
        fix: bool,
        /// List registered rules and exit.
        #[arg(long)]
        list: bool,
        /// Run only the rule with this id.
        #[arg(long)]
        rule: Option<String>,
        /// Run only the comma-separated set of rules.
        #[arg(long)]
        rules: Option<String>,
    },
    /// Internal: run the tests in a single file and emit NDJSON on stdout.
    /// Invoked by `chelis test` as a subprocess per file so a crash in one
    /// test file (e.g., stack overflow) does not kill every other test file.
    #[command(hide = true, name = "__test_file")]
    InternalTestFile {
        /// Absolute path to the .ch test file.
        file: PathBuf,
        /// Relative path used in the `<file>::<test>` filter key.
        #[clap(long)]
        rel_display: String,
        /// Optional substring filter on `<file>::<test_fn>`.
        #[clap(long)]
        filter: Option<String>,
        /// Per-test timeout in seconds.
        #[clap(long, default_value = "30")]
        timeout: u64,
        /// Preserve a bare file-level compile/check diagnostic even when the
        /// file declares no `test_*` function. Internal adapter flag used only
        /// by `chelis test --expect`.
        #[clap(long)]
        expect_file_diagnostic: bool,
    },
    /// Internal: run a manifest of test files in a single batch worker.
    #[command(hide = true, name = "__test_batch")]
    InternalTestBatch {
        /// JSON manifest describing the files and tests in the batch.
        #[clap(long)]
        manifest: PathBuf,
        /// Per-test timeout in seconds.
        #[clap(long, default_value = "30")]
        timeout: u64,
    },
}

#[derive(Subcommand)]
enum TideCommand {
    /// Start the Tide HTTP/JSON API server
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
        #[arg(long, default_value_t = 8080)]
        port: u16,
    },
    /// Start the Tide MCP server over stdio
    Mcp,
    /// Start the Tide LSP server over stdio
    Lsp {
        #[arg(long, hide = true, action = ArgAction::SetTrue)]
        stdio: bool,
    },
}

#[derive(Subcommand)]
enum ReefCommand {
    /// Initialize a Reef package root
    Init {
        name: String,
        #[arg(long)]
        module_prefix: String,
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
    /// Build package artifacts (.chb + .tar.zst).
    ///
    /// By default, missing-from-registry dependencies are
    /// auto-fetched from the canonical hosting org's GitHub release
    /// tags before the build resumes; pass `--no-auto-fetch` to
    /// opt out. Phase A Item 8 introduced this default and the
    /// opt-out flag — see `spec/design/reef_distribution.md` § Item 8.
    Build {
        path: Option<PathBuf>,
        /// Disable Item 8's default-on auto-fetch of missing-from-
        /// registry dependencies. With this flag set, a missing
        /// dependency surfaces an error naming the URL that
        /// would have been auto-fetched, plus the recommended
        /// `chelis reef install --from-github <url>` recovery step.
        ///
        /// Use this when you want explicit control over when the
        /// build performs network access (e.g. air-gapped CI,
        /// reproducible-rebuild auditing).
        #[arg(long = "no-auto-fetch")]
        no_auto_fetch: bool,
    },
    /// Validate a standalone Reef CHB/archive pair without installing it.
    ///
    /// Strictly consumes the complete CHB envelope, rejects malformed or
    /// noncanonical metadata, and checks the archive bytes against the
    /// SHA-256 embedded in the CHB. This command is read-only and is
    /// suitable for downstream release gates.
    VerifyArtifact {
        /// Source archive paired with the CHB.
        #[arg(long, value_name = "PATH")]
        archive: PathBuf,
        /// Compiled shell metadata (`.chb`) to validate.
        #[arg(long, value_name = "PATH")]
        shell: PathBuf,
        /// Emit a stable JSON report. In this mode stdout is JSON only,
        /// stderr is empty, and `valid` is true iff `errors` is empty.
        #[arg(long, action = ArgAction::SetTrue)]
        json: bool,
    },
    /// Publish a package into the local Reef registry
    Publish { path: Option<PathBuf> },
    /// Install prebuilt packages into the local Reef registry
    ///
    /// Populates `~/.chelis/reef/packages/<name>/<version>/` and updates
    /// `~/.chelis/reef/index.json` from a known-good source.
    ///
    /// Four source forms are supported:
    /// * `--from-monorepo <PATH>` — copy prebuilt artifacts out of a
    ///   chelis monorepo's `packages/<name>/dist/` directory.
    /// * `--from-github <ORG>/<REPO>@<TAG>` — fetch the release assets
    ///   `<repo>-<version>.tar.zst` and `<repo>-<version>.chb` via
    ///   the GitHub REST API (the public `/releases/download/...` URL
    ///   form does not serve private-repo bytes; the canonical
    ///   chelis-lang shells are private during the pre-launch era).
    ///   Both assets are validated through the same on-disk
    ///   verification path as `--from-monorepo`. Authentication uses
    ///   `GITHUB_TOKEN`, falling back to `gh auth token`.
    /// * `--from-lockfile` — read the project's `reef.lock`, walk every
    ///   dependency, and re-fetch each one from the `remote_origin` it
    ///   recorded. Hashes are verified against the lockfile pins; any
    ///   mismatch is surfaced as a typed validation error. Entries
    ///   without a recorded `remote_origin` (older monorepo-only
    ///   installs) error with a suggestion to re-run `--bootstrap` to
    ///   populate origins.
    /// * `--bootstrap [<ORG>/<REPO>@<TAG>...]` — install canonical shell
    ///   releases in dependency order. With no explicit entries, the
    ///   built-in default list installs nautilus, coral, shoals, and
    ///   octant. `chelis-std` is compiler-bundled and rejected as an
    ///   explicit bootstrap target.
    ///
    /// The four sources are mutually exclusive — exactly one of them
    /// must be supplied per invocation.
    ///
    /// `chelis reef build` does NOT auto-install dependencies. This is
    /// the explicit population step.
    Install {
        /// Path to a chelis monorepo (the directory containing `packages/`).
        #[arg(
            long,
            value_name = "PATH",
            conflicts_with_all = ["from_github", "from_lockfile", "bootstrap"]
        )]
        from_monorepo: Option<PathBuf>,
        /// GitHub release reference: `<org>/<repo>@<tag>`. The tag may
        /// have an optional leading `v` (e.g. `v0.4.0` or `0.4.0`).
        /// Requires `GITHUB_TOKEN` (or a working `gh auth token`)
        /// because the canonical-org repos are private.
        #[arg(long, value_name = "ORG/REPO@TAG", conflicts_with_all = ["from_lockfile", "bootstrap"])]
        from_github: Option<String>,
        /// Re-install every dependency named by the project's
        /// `reef.lock`, fetching each from its recorded
        /// `remote_origin`. Hashes are verified against the lockfile
        /// pins. Useful for fresh checkouts to reproduce another
        /// developer's local registry state without out-of-band
        /// knowledge of which `--from-github` invocations populated
        /// it.
        #[arg(long, conflicts_with = "bootstrap")]
        from_lockfile: bool,
        /// Optional path to the package root for `--from-lockfile`.
        /// Defaults to the current directory. Has no effect with
        /// `--from-monorepo`, `--from-github`, or `--bootstrap`.
        #[arg(long, value_name = "PATH", requires = "from_lockfile")]
        package_root: Option<PathBuf>,
        /// Topologically-ordered install of multiple shells from
        /// canonical-org GitHub Releases.
        ///
        /// Pass zero or more `<org>/<repo>@<tag>` entries. With no
        /// entries, the built-in
        /// [`chelis_reef::DEFAULT_BOOTSTRAP_LIST`] is used (canonical
        /// shells: nautilus, coral, shoals, octant). The
        /// installer fetches each shell's manifest, builds a
        /// dependency graph, topologically sorts, and installs each
        /// shell via the same path as `--from-github`. Cycles and
        /// references to packages outside the input set are surfaced
        /// as typed errors.
        ///
        /// `chelis-std` is the language runtime and ships with the
        /// compiler — it is not a bootstrap target. Including it in
        /// the input list is rejected with a typed runtime-error.
        #[arg(long, value_name = "ORG/REPO@TAG", num_args = 0..)]
        bootstrap: Option<Vec<String>>,
        /// `<name>` or `<name>=<version>` selectors. If omitted with
        /// `--from-monorepo`, every package in the monorepo is installed.
        /// Ignored with `--from-github` (the spec is the selector).
        /// Rejected with `--bootstrap` (entries are passed to
        /// `--bootstrap` directly).
        #[arg(value_name = "NAME[=VERSION]")]
        packages: Vec<String>,
    },
    /// Export a hermetic bundle of pinned dependencies for agent sandboxes.
    ///
    /// Reads `reef.toml` + `reef.lock`, materializes all pinned dependency
    /// sources into a directory bundle with preserved hashes and metadata.
    /// The bundle contains `bundle.json` (metadata), `root/` (the root
    /// package source), and `<dep>-<version>/` directories for each
    /// dependency.
    ExportBundle {
        /// Package root (defaults to `.`).
        #[arg(long, short)]
        path: Option<PathBuf>,
        /// Output directory for the bundle.
        #[arg(long, short)]
        output: PathBuf,
    },
    /// Emit a machine-readable ABI/package schema (JSON).
    ///
    /// Describes exported functions (name, params, return type), types
    /// (variants, opaque status), and constructors (partial/total).
    /// Stable enough for authoring harnesses to validate authored modules
    /// without duplicating compiler facts.
    Schema {
        /// Package root (defaults to `.`).
        path: Option<PathBuf>,
    },
    /// Manage chelis compiler **source crates** (the class-(c) dependency)
    /// for shells that link `chelis-ir` / `chelis-types` … as Cargo path
    /// deps via a `[chelis-src]` section in `reef.toml`.
    ///
    /// Maintains a version-keyed source store under
    /// `~/.local/share/chelis-src/` (a bare mirror of canonical
    /// `Chelis-Lang/chelis` plus one git worktree per pinned commit) and
    /// points this shell's `../chelis` slot at the worktree matching its own
    /// pin — so several shells on different chelis versions build side by
    /// side without colliding on a single sibling clone.
    Src {
        #[command(subcommand)]
        command: ReefSrcCommand,
    },
    /// Report chelis dependency health for one or more shells across every
    /// class (toolchain binary, source crates, binary artifacts).
    ///
    /// Prints a machine-wide header (the chelis home, the shim, and the
    /// recorded default), then scans a root for shell repos (a directory
    /// with a `reef.toml` carrying a `compiler =` pin). For each it reports
    /// whether the pinned toolchain is installed in the chelisup store
    /// (`<chelis home>/toolchains/<ver>`, fixed with `chelisup install`);
    /// for crate-linking shells, whether the source store and `../chelis`
    /// slot are synced to the pin; and for shells declaring `[artifacts]`,
    /// whether each binary artifact is installed (Item 11 / chelis#468).
    /// Read-only; it never installs.
    Doctor {
        /// Directory to scan (defaults to `.`): the root itself and each
        /// immediate subdirectory carrying a `reef.toml` is reported.
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Print the resolved on-disk path of an installed binary artifact.
    ///
    /// Item 11 (chelis#468): binary artifacts declared in `[artifacts]`
    /// and installed via `chelis reef install --from-lockfile` are placed
    /// at `$CHELIS_HOME/bin/<name>` (default `~/.chelis/bin/<name>`).
    /// `chelis reef which <artifact>` prints that path so consumers point
    /// at the binary with no out-of-band knowledge. Exits non-zero with a
    /// message on stderr if the artifact is not installed.
    Which {
        /// Logical artifact name (the `[artifacts.<name>]` key).
        artifact: String,
    },
    /// Bring a freshly-cloned shell to its pins in one command (WS-C, §7).
    ///
    /// Reads the `reef.toml` compiler pin, then, in order:
    /// 1. ensures the pinned toolchain is installed, auto-installing it by
    ///    delegating to `chelisup` when it is missing;
    /// 2. `reef install --from-lockfile` — source packages + binary
    ///    artifacts (chelis#468) from `reef.lock`, when one is present;
    /// 3. `reef src sync` — chelis source crates (chelis#571), when the
    ///    manifest carries a `[chelis-src]` section;
    /// 4. prints the `reef doctor` health summary.
    ///
    /// This is the current-chelis entry point for the cross-version case
    /// (§5.4): it may itself install the pinned toolchain, so a
    /// clone-and-`setup` does the right thing without reaching for `+<ver>`.
    Setup {
        /// Shell package root (defaults to `.`).
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// Downstream shell-repo contract conformance (audit / stamp / sync).
    ///
    /// The machine-checked form of `spec/design/shell_repo_contract.md`,
    /// shipped in the toolchain so it cannot copy-drift. See
    /// [`ConformCommand`] for the verbs.
    Conform {
        #[command(subcommand)]
        command: ConformCommand,
    },
}

/// `chelis reef conform <verb>` — see `shell_repo_contract.md`.
#[derive(Subcommand)]
enum ConformCommand {
    /// Audit the current repo against the §11 conformance manifest. Read-only,
    /// offline, deterministic. Exits non-zero on any MUST-tier failure.
    Audit {
        /// Repo root to audit (defaults to `.`).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Emit one NDJSON record per contract row plus a summary line.
        #[arg(long)]
        json: bool,
        /// Print the per-finding evidence under each failing row: the citation
        /// site(s) and the per-candidate coverage reasoning behind the verdict.
        #[arg(long)]
        explain: bool,
    },
    /// Scaffold a new shell (or retrofit an existing one) from the embedded
    /// templates: reef.toml, AGENTS.md (+ CLAUDE.md symlink), the CHELIS_SURFACE
    /// / UPSTREAM_BUGS docs, tests_neg/tests_blocked, CI, and materialized skills.
    Init {
        /// Package name.
        name: String,
        /// Module namespace (PascalCase).
        #[arg(long)]
        module_prefix: String,
        /// Output directory (defaults to `./<name>`).
        #[arg(long, short)]
        output: Option<PathBuf>,
    },
    /// Regenerate the pointer managed blocks and re-materialize the skill set
    /// from the pinned toolchain, restamping to the reef pin. Touches only
    /// managed regions and `agent-skills/`.
    Sync {
        /// Shell package root (defaults to `.`).
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// Mechanize the Pin Bump Checklist: rewrite every pin location in lockstep,
    /// restamp the managed blocks + re-materialize skills, then run the offline
    /// audit and the blocked/negative suites. Produces the change set on the
    /// working tree (git-agnostic; the caller opens the PR). Exits non-zero only
    /// when the bump's OWN output is non-conformant (its pins/stamps/skills) or a
    /// suite fails; a clean bump that leaves only author-follow-up rows (CI
    /// wiring, pre-existing doc fixes) exits 0 and lists the remaining steps.
    Bump {
        /// Target chelis version (bare `X.Y.Z`).
        version: String,
        /// Shell package root (defaults to `.`).
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// Pin-change guard for shell CI: pass trivially if the reef pin is
    /// unchanged vs `--base`; if it changed, require a green conformance audit
    /// (fresh managed-block stamps, lockstep workflow pins, wired probes) — so a
    /// raw pin edit that skips the checklist fails the PR.
    BumpCheck {
        /// Git ref to diff the pin against (e.g. `origin/main`).
        #[arg(long)]
        base: String,
        /// Shell package root (defaults to `.`).
        #[arg(long)]
        path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ReefSrcCommand {
    /// Sync the pinned source store and point this shell's `../chelis` slot
    /// at it. Idempotent. Refuses (never deletes) a real dev clone in the
    /// slot — relocate it out of the sibling directory first.
    Sync {
        /// Shell package root (defaults to `.`).
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// Verify this shell's source crates have not drifted off the pin: the
    /// store worktree is at the pinned commit, `../chelis` is the symlink
    /// into it, and `Cargo.lock` records the pinned crate versions. Offline;
    /// loud, actionable failure on drift. Intended for a shell's local gate.
    Check {
        /// Shell package root (defaults to `.`).
        #[arg(long)]
        path: Option<PathBuf>,
    },
    /// Show the resolved source-store state for this shell (pin, store
    /// worktree HEAD, slot target). Read-only and non-failing.
    Status {
        /// Shell package root (defaults to `.`).
        #[arg(long)]
        path: Option<PathBuf>,
    },
}

fn main() {
    // Tier B process isolation: if this process was spawned as a prove worker,
    // run one cvc5 solve and exit before doing anything else; otherwise enable
    // isolation so every Tier B solve runs in a short-lived child whose crash
    // (cvc5 C++ abort, stack overflow, OOM, panic) becomes a clean Tier C
    // result instead of taking the `chelis` process down. smt-only -- without
    // the feature there is no cvc5 and nothing to isolate.
    #[cfg(feature = "smt")]
    {
        chelis_prove::run_worker_if_requested();
        chelis_prove::enable_isolation();
    }
    chelis_ir::lower::install_chelis_panic_hook();
    // §5.4: intercept clap's unrecognized-subcommand error to append the
    // cross-version hint; every other clap outcome is left untouched.
    let cli = Cli::try_parse().unwrap_or_else(|e| handle_parse_error(e));
    let result = match cli.command {
        Some(Command::Deep {
            file,
            flat,
            annotate,
        }) => cmd_deep(&file, flat, annotate),
        Some(Command::Surf { file, verbose }) => cmd_surf(&file, verbose),
        Some(Command::Fmt {
            file,
            inplace,
            check,
        }) => cmd_fmt(&file, inplace, check),
        Some(Command::Eval {
            file,
            expr,
            json,
            allow_style_violations,
        }) => cmd_eval(
            file.as_deref(),
            expr.as_deref(),
            json,
            allow_style_violations,
        ),
        Some(Command::Check {
            file,
            show_inferred,
            allow_style_violations,
        }) => match cmd_check(&file, show_inferred, allow_style_violations) {
            // Issue #207: exit code mirrors the JSON `errors` array
            // (0 iff empty, CHECK_ERRORS_EXIT_CODE otherwise).
            Ok(code) => std::process::exit(code),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(1);
            }
        },
        Some(Command::Cost { file, json }) => cmd_cost(&file, json),
        Some(Command::Validate {
            surf,
            deep,
            desugar,
            file,
            allow_style_violations,
        }) => cmd_validate(&file, surf, deep, desugar, allow_style_violations),
        Some(Command::Build {
            file,
            output,
            target,
            deep,
            allow_style_violations,
        }) => cmd_build_dispatch(
            &file,
            output.as_deref(),
            &target,
            deep,
            allow_style_violations,
        ),
        Some(Command::Reef { command }) => cmd_reef(command),
        Some(Command::Tide { command }) => run_tide(command),
        Some(Command::Cove { file }) => cmd_cove(file),
        Some(Command::InternalTestFile {
            file,
            rel_display,
            filter,
            timeout,
            expect_file_diagnostic,
        }) => match cmd_internal_test_file(
            &file,
            &rel_display,
            filter.as_deref(),
            Duration::from_secs(timeout.max(1)),
            expect_file_diagnostic,
        ) {
            Ok(code) => std::process::exit(code),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(2);
            }
        },
        Some(Command::InternalTestBatch { manifest, timeout }) => {
            match cmd_internal_test_batch(&manifest, Duration::from_secs(timeout.max(1))) {
                Ok(code) => std::process::exit(code),
                Err(err) => {
                    eprintln!("error: {err}");
                    std::process::exit(2);
                }
            }
        }
        Some(Command::Test {
            path,
            filter,
            json,
            timeout,
            suite_timeout,
            jobs,
            batch_mode,
            expect,
        }) => match cmd_test_supervised(
            path.as_deref(),
            filter.as_deref(),
            json,
            timeout,
            suite_timeout,
            jobs,
            batch_mode,
            expect,
        ) {
            Ok(code) => std::process::exit(code),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(2);
            }
        },
        Some(Command::Prove {
            path,
            only,
            samples,
            seed,
            max_attempts,
            json,
            spans,
            tier: _tier,
            smt_timeout: _smt_timeout,
            invariant_min_rate,
            package,
            capabilities,
        }) => {
            if capabilities {
                let caps = prove::prove_capabilities();
                println!("{}", serde_json::to_string_pretty(&caps).unwrap());
                std::process::exit(0);
            }
            match prove::cmd_prove(prove::ProveOptions {
                path: path.as_deref(),
                only: only.as_deref(),
                samples,
                seed,
                max_attempts,
                json,
                spans: spans.as_deref(),
                tier: &_tier,
                smt_timeout_ms: _smt_timeout,
                invariant_min_rate,
                package: package.as_deref(),
            }) {
                Ok(code) => std::process::exit(code),
                Err(err) => {
                    eprintln!("error: {err}");
                    std::process::exit(3);
                }
            }
        }
        Some(Command::Lint {
            paths,
            check,
            fix,
            list,
            rule,
            rules,
        }) => match cmd_lint(paths, check, fix, list, rule.as_deref(), rules.as_deref()) {
            Ok(code) => std::process::exit(code),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(2);
            }
        },
        None => {
            println!(
                "chelis {} -- use --help for commands",
                env!("CARGO_PKG_VERSION")
            );
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn cmd_deep(file: &Path, flat: bool, annotate: bool) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    let decls = chelis_surf::parser::parse_str(&source)?;
    let deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
    let deep_exprs = if annotate {
        match chelis_types::check_ir_program(&deep_exprs) {
            Ok(checked) => checked.annotated_exprs().to_vec(),
            Err(result) => {
                return Err(format!(
                    "`chelis deep --annotate` requires a well-typed program; type errors: {:?}",
                    result.errors
                )
                .into());
            }
        }
    } else {
        deep_exprs
    };
    let output = if flat {
        chelis_deep::printer::print_canonical_flat(&deep_exprs)
    } else {
        chelis_deep::printer::print_canonical(&deep_exprs)
    };
    print!("{output}");
    Ok(())
}

fn cmd_surf(file: &Path, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    let options = if verbose {
        chelis_surf::decompile::DecompileOptions::verbose()
    } else {
        chelis_surf::decompile::DecompileOptions::idiomatic()
    };
    let synthetic_name = file.file_stem().and_then(|stem| stem.to_str());
    if ext == "dp" {
        let deep_source = style_gate::strip_deep_lint_directive_lines(&source);
        let deep_exprs = chelis_deep::parser::parse_str_strict(&deep_source)?;
        let surf = chelis_surf::decompile::decompile_program_with_context(
            &deep_exprs,
            &options,
            synthetic_name,
        );
        let surf = if verbose {
            surf
        } else {
            canonicalize_decompiled_surf(&surf)?
        };
        print!("{surf}");
    } else {
        // For .ch files, round-trip through deep and back
        let decls = chelis_surf::parser::parse_str(&source)?;
        let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
        let surf = chelis_surf::decompile::decompile_program_with_context(
            &deep_exprs,
            &options,
            synthetic_name,
        );
        let surf = if verbose {
            surf
        } else {
            canonicalize_decompiled_surf(&surf)?
        };
        print!("{surf}");
    }
    Ok(())
}

fn canonicalize_decompiled_surf(surf: &str) -> Result<String, Box<dyn std::error::Error>> {
    let decls = chelis_surf::parser::parse_str(surf)
        .map_err(|err| format!("decompiler emitted Surf that the parser rejected: {err}"))?;
    Ok(chelis_surf::format::format_program(&decls))
}

fn cmd_fmt(file: &Path, inplace: bool, check: bool) -> Result<(), Box<dyn std::error::Error>> {
    if inplace && check {
        return Err("`chelis fmt` does not allow `--inplace` and `--check` together".into());
    }
    let source = fs::read_to_string(file)?;
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    let output = if ext == "dp" {
        let deep_exprs = chelis_deep::parser::parse_str_strict(&source)?;
        chelis_deep::printer::print_canonical(&deep_exprs)
    } else {
        // .ch: parse Surf -> pretty-print Surf while preserving surface
        // choices and source comments.
        chelis_surf::format::format_source(&source)?
    };
    if check {
        if output == source {
            return Ok(());
        }
        return Err(format!("{} is not canonically formatted", file.display()).into());
    }
    if inplace {
        fs::write(file, &output)?;
    } else {
        print!("{output}");
    }
    Ok(())
}

fn cmd_eval(
    file: Option<&std::path::Path>,
    expr: Option<&str>,
    json: bool,
    allow_style_violations: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // The style gate runs only on the `--file` form (a real on-disk
    // source). The `--expr` form is a synthetic one-line snippet
    // wrapped as `__eval_result = <expr>` and never lands on disk, so
    // there's nothing canonical to compare against.
    if let Some(path) = file
        && let Ok(source) = fs::read_to_string(path)
    {
        style_gate::enforce_style_gate(path, &source, allow_style_violations)?;
    }
    // Phase H, cmd_eval slice: when the user is evaluating a `--file` whose
    // reef package is detectable (either the file lives inside a package or
    // the current working directory does, matching the existing dispatch
    // inside `chelis_reef::prepare_program_for_eval_file`), build a
    // `CompiledContext` once and route the user's source through
    // `eval_in_context`. The library decls (chelis-std + reef deps + the
    // package's own modules) are already type-checked and lowered, so the
    // per-eval cost drops to just the user's source. For the no-reef case
    // (raw `--file foo.ch` outside any package), or for the `--expr` form,
    // there is no library context to amortize against; keep the existing
    // monolithic path. Output (formatted result, exit code, error messages)
    // is byte-identical to pre-refactor on the same input — see
    // `crates/chelis-cli/tests/eval_in_reef_context.rs` for the parity
    // probes and the Phase G acceptance suite
    // (`crates/chelis-compiler-api/tests/compiled_context.rs`)
    // for the underlying API parity guarantee.
    match (file, expr) {
        (Some(path), _) => {
            // Deep (`.dp`) ingestion: a standalone `.dp` is already-lowered
            // IR, not a Surf package, so the reef fast path and the legacy
            // `load_eval_decls` fallback (both of which run the Surf parser)
            // would mis-parse it. Route `.dp` straight through the eval
            // engine's `SourceKind::Deep` path before either is reached.
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext.eq_ignore_ascii_case("dp") {
                let source = fs::read_to_string(path)?;
                let deep_source = style_gate::strip_deep_lint_directive_lines(&source);
                // Strict-parse first so an unknown tag is a clean
                // strict-vocabulary error here, matching the `.dp`
                // surfaces of `chelis check`, `build`, `fmt`, and `cost`.
                // The engine's own `parse_deep` is non-strict; surfacing
                // the strict error in the CLI keeps every `.dp` CLI
                // surface on the same closed-vocabulary gate.
                chelis_deep::parser::parse_str_strict(&deep_source)
                    .map_err(|err| boxed_string_error(err.to_string()))?;
                return if json {
                    run_eval_json_emit(try_eval_result(SourceKind::Deep, &deep_source, None))
                } else {
                    run_eval_emit(try_eval(SourceKind::Deep, &deep_source, None))
                };
            }
            // RFC v5 (RT-1 F2 bypass): a `--file` resolving into a reef
            // package evaluates reef-linker output (mangled), through the
            // fast path AND the `load_eval_decls` fallback below (which
            // re-formats + re-evaluates the linked decls). Accept the
            // linker name format for the rest of this arm. The raw `.dp`
            // case returned above, so it keeps the flag FALSE and rejects
            // mangled names as a forge.
            let eval_package_root = detect_eval_package_root(path)?;
            let _linked_guard = eval_package_root
                .is_some()
                .then(chelis_types::install_linked_program_guard);
            if let Some(package_root) = &eval_package_root {
                let source = fs::read_to_string(path)?;
                match run_eval_in_context(package_root, &source, json) {
                    Ok(()) => return Ok(()),
                    Err(EvalInContextError::Compile(msg)) => return Err(msg.into()),
                }
            }
            // Raw `--file foo.ch` outside any reef package, or a reef
            // package whose graph the new context-builder can't yet
            // hash: fall back to the legacy `prepare_eval` path.
            let (decls, entry_decls) = load_eval_decls(path)?;
            let deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
            let checked = checked_program_with_effects(&deep_exprs).map_err(boxed_string_error)?;
            let source = chelis_surf::format::format_program(&decls);
            let selected_roots = root_names_from_decls(&entry_decls, checked.type_env());
            if json {
                run_eval_json_emit(try_eval_result(
                    SourceKind::Surf,
                    &source,
                    Some(&selected_roots),
                ))
            } else {
                run_eval_emit(try_eval(SourceKind::Surf, &source, Some(&selected_roots)))
            }
        }
        (None, Some(e)) => {
            // `--expr` is by construction a one-line snippet with no reef
            // resolution — keep the legacy path.
            let source = format!("__eval_result = {e}");
            if json {
                run_eval_json_emit(try_eval_result(SourceKind::Surf, &source, None))
            } else {
                run_eval_emit(try_eval(SourceKind::Surf, &source, None))
            }
        }
        (None, None) => Err("provide --file or an expression".into()),
    }
}

/// Detect whether `chelis eval --file <path>` should route through the
/// Phase H `compile_reef_context + eval_in_context` fast path. Mirrors the
/// dispatch baked into `chelis_reef::prepare_program_for_eval_file`:
/// - if `<path>` parses as a single `module Foo` decl, the package root is
///   discovered by walking up from `<path>` itself,
/// - otherwise (a loose snippet file), the package root is discovered by
///   walking up from the current working directory.
///
/// Returns `Ok(Some(root))` when a reef package applies (route through the
/// new path), `Ok(None)` when no reef package is in scope (caller falls
/// back to the legacy `prepare_eval` path), and `Err` only on
/// canonicalize/IO errors that would have surfaced during the legacy path
/// anyway.
fn detect_eval_package_root(file: &Path) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    // A parse failure here is non-fatal for routing: fall back to the
    // legacy path so the user sees the same parse error they would have
    // before the refactor.
    let Ok(decls) = chelis_surf::parser::parse_str(&source) else {
        return Ok(None);
    };
    if matches!(decls.as_slice(), [Decl::Module { .. }]) {
        chelis_reef::find_package_root_for_input(file).map_err(boxed_string_error)
    } else {
        let cwd = env::current_dir()?;
        chelis_reef::find_package_root_for_dir(&cwd).map_err(boxed_string_error)
    }
}

/// Outcome from the Phase H context path. Kept distinct from a generic boxed
/// error so compile/eval diagnostics retain their established rendering.
enum EvalInContextError {
    /// Any other failure: type error, effect error, eval error, etc.
    /// Propagate to the user with the same format the legacy path used.
    Compile(String),
}

/// Build a `CompiledContext` for `package_root`, then evaluate `source`
/// against it. `reef_home` is sourced from the `CHELIS_REEF_HOME` env var
/// if present (matching how `chelis test` plumbs it to workers); Phase K
/// uses it to key the disk cache so a warm `chelis eval --file` re-run
/// against unchanged sources skips the ~67s library compile entirely.
fn run_eval_in_context(
    package_root: &Path,
    source: &str,
    json: bool,
) -> Result<(), EvalInContextError> {
    let reef_home = env::var_os("CHELIS_REEF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(""));
    // Phase K: route through `load_or_compile_for_package` so the disk
    // cache amortizes cold-compile cost across invocations. On a hit
    // (source unchanged since last run), the library compile is skipped
    // entirely. On a miss, the helper runs the full compile and saves
    // the result for next time. `verbose=true` so an operator with a
    // corrupt cache file sees a stderr breadcrumb instead of a silent
    // recompile.
    let context =
        match chelis_compiler_api::load_or_compile_for_package(&reef_home, package_root, true) {
            Ok(ctx) => ctx,
            Err(err) => {
                let msg = err
                    .errors
                    .iter()
                    .map(|d| d.message.clone())
                    .collect::<Vec<_>>()
                    .join("; ");
                return Err(EvalInContextError::Compile(msg));
            }
        };
    let result = chelis_compiler_api::eval_in_context(&context, source).map_err(|err| {
        EvalInContextError::Compile(
            err.errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("; "),
        )
    })?;
    if json {
        // JSON mode: stdout carries the raw `EvalResult` serde JSON
        // only. Empty-roots inputs serialize to `{"roots":[]}` (valid
        // JSON); the stderr breadcrumb is suppressed so scripted
        // consumers get a single parseable document on stdout.
        let rendered = serde_json::to_string(&result)
            .map_err(|err| EvalInContextError::Compile(format!("eval JSON serialize: {err}")))?;
        println!("{rendered}");
        return Ok(());
    }
    let formatted = format_eval_result(&result);
    if formatted.is_empty() {
        warn_eval_no_roots();
        return Ok(());
    }
    println!("{formatted}");
    Ok(())
}

fn run_eval_emit(outcome: Result<String, String>) -> Result<(), Box<dyn std::error::Error>> {
    match outcome {
        Ok(result) => {
            if result.is_empty() {
                warn_eval_no_roots();
                return Ok(());
            }
            println!("{result}");
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

/// JSON counterpart to [`run_eval_emit`]. Stdout carries the raw
/// `EvalResult` serde JSON only; nothing else is written there. An
/// empty-roots program serializes to `{"roots":[]}` (valid JSON) rather
/// than emitting the human stderr breadcrumb, so a scripted consumer
/// always receives a single parseable document. Errors propagate as a
/// boxed error (stderr + nonzero exit), unchanged from the text path.
fn run_eval_json_emit(
    outcome: Result<chelis_compiler_api::schema::EvalResult, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    match outcome {
        Ok(result) => {
            let rendered = serde_json::to_string(&result)?;
            println!("{rendered}");
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

// G7 CLI sub-bug: when `chelis eval --file <foo.ch>` is handed a Surf
// input that contains only `def` declarations and no top-level
// evaluable expression, `format_eval_result` returns an empty string
// and both eval paths short-circuit with exit 0 and no output. That
// silent success is a footgun for interactive users. Emit a stderr
// warning at the short-circuit site so humans get a breadcrumb;
// preserve exit 0 so scripted consumers that pipe stdout downstream
// keep working. See `docs/investigations/cli_eval_empty_roots_diagnosis.md`.
fn warn_eval_no_roots() {
    eprintln!("warning: input contains only def declarations; nothing to evaluate");
}

fn cmd_cost(file: &Path, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    let summary = copy_cost_for_file(file)?;
    if json {
        println!("{}", copy_cost_json(file, &summary));
    } else {
        print!("{}", copy_cost_human(file, &summary));
    }
    Ok(())
}

fn copy_cost_for_file(
    file: &Path,
) -> Result<chelis_ir::analysis::CopyCostSummary, Box<dyn std::error::Error>> {
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "dp" {
        let source = fs::read_to_string(file)?;
        let deep_source = style_gate::strip_deep_lint_directive_lines(&source);
        let deep_exprs = chelis_deep::parser::parse_str_strict(&deep_source)
            .map_err(|err| format!("Deep parse error: {err}"))?;
        let checked =
            checked_program_with_effects(&deep_exprs).map_err(|e| format!("Check errors: {e}"))?;
        return copy_cost_for_checked(&checked, &deep_exprs, &deep_exprs);
    }

    let prepared = chelis_reef::prepare_program_for_file(file).map_err(boxed_string_error)?;
    let (decls, entry_decls, linked_program) = match prepared {
        Some(prepared) => (prepared.decls, prepared.entry_decls, true),
        None => {
            let source = fs::read_to_string(file)?;
            (
                chelis_surf::parser::parse_str(&source)
                    .map_err(|e| format!("{}: {e}", file.display()))?,
                Vec::new(),
                false,
            )
        }
    };
    let deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
    let entry_deep_exprs = expanded_desugared_program(&entry_decls).map_err(boxed_string_error)?;
    let check_deep_exprs = if linked_program {
        let live_deep_exprs =
            drop_unreachable_eval_only_defs(deep_exprs.clone(), &entry_deep_exprs);
        prune_build_program_to_reachable_defs(&live_deep_exprs, &entry_deep_exprs)
    } else {
        deep_exprs.clone()
    };
    // Reef package sources are linker-rewritten before they reach the Deep
    // checker. Mirror `check`/`build`: accept reserved linker names only for
    // that prepared package path, not for raw single-file inputs.
    let _linked_guard = linked_program.then(chelis_types::install_linked_program_guard);
    let checked = checked_program_with_effects(&check_deep_exprs)
        .map_err(|e| format!("Check errors: {e}"))?;
    copy_cost_for_checked(&checked, &check_deep_exprs, &entry_deep_exprs)
}

fn copy_cost_for_checked(
    checked: &chelis_types::CheckedProgram,
    program_exprs: &[DeepExpr],
    entry_exprs: &[DeepExpr],
) -> Result<chelis_ir::analysis::CopyCostSummary, Box<dyn std::error::Error>> {
    let mut entry_names = Vec::new();
    for expr in entry_exprs {
        if let Some(name) = deep_top_level_expr_name(expr) {
            entry_names.push(name.to_string());
        }
    }
    if entry_names.is_empty() {
        for expr in program_exprs {
            if let Some(name) = deep_top_level_expr_name(expr) {
                entry_names.push(name.to_string());
            }
        }
    }
    let mut functions = Vec::new();
    for name in &entry_names {
        if let Some(dag) = chelis_ir::host::lower_named_tensor_entry_dag(checked, name) {
            let roots = dag.roots().to_vec();
            if roots.is_empty() {
                continue;
            }
            let summary = chelis_ir::analysis::analyze_copy_costs_for_roots(
                &dag,
                &[(display_root_name(name), roots)],
            );
            functions.extend(summary.functions);
        }
    }

    if functions.is_empty() {
        let dag = chelis_ir::lower::try_lower_program(checked)
            .map_err(|diagnostic| boxed_string_error(diagnostic.to_string()))?;
        let all_names = lowered_root_names_from_exprs(program_exprs, checked.type_env());
        let selected_names =
            lowered_root_names_from_selected_exprs(entry_exprs, program_exprs, checked.type_env());
        let selected_names = if selected_names.is_empty() {
            all_names.clone()
        } else {
            selected_names
        };
        let roots = all_names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                selected_names
                    .iter()
                    .any(|selected| selected == name)
                    .then(|| {
                        dag.roots()
                            .get(index)
                            .copied()
                            .map(|root| (name.clone(), vec![root]))
                    })
                    .flatten()
            })
            .collect::<Vec<_>>();
        return Ok(chelis_ir::analysis::analyze_copy_costs_for_roots(
            &dag, &roots,
        ));
    }

    Ok(summarize_copy_cost_functions(functions))
}

fn summarize_copy_cost_functions(
    functions: Vec<chelis_ir::analysis::FunctionCopyCost>,
) -> chelis_ir::analysis::CopyCostSummary {
    let total_copy_count = functions.iter().map(|function| function.copy_count).sum();
    let total_bytes_copied = functions.iter().try_fold(0usize, |acc, function| {
        function.bytes_copied.map(|bytes| acc.saturating_add(bytes))
    });
    let formula_terms = functions
        .iter()
        .filter_map(|function| function.byte_formula.as_ref())
        .filter(|formula| !formula.is_empty())
        .cloned()
        .collect::<Vec<_>>();
    let total_byte_formula = (!formula_terms.is_empty()).then(|| formula_terms.join(" + "));
    chelis_ir::analysis::CopyCostSummary {
        functions,
        total_copy_count,
        total_bytes_copied,
        total_byte_formula,
    }
}

fn copy_cost_json(
    file: &Path,
    summary: &chelis_ir::analysis::CopyCostSummary,
) -> serde_json::Value {
    let functions = summary
        .functions
        .iter()
        .map(|function| {
            let mut object = serde_json::Map::new();
            object.insert("name".to_string(), serde_json::json!(function.name));
            object.insert(
                "copy_count".to_string(),
                serde_json::json!(function.copy_count),
            );
            if let Some(bytes) = function.bytes_copied {
                object.insert("bytes_copied".to_string(), serde_json::json!(bytes));
            }
            serde_json::Value::Object(object)
        })
        .collect::<Vec<_>>();
    let mut object = serde_json::Map::new();
    object.insert(
        "file".to_string(),
        serde_json::json!(file.display().to_string()),
    );
    object.insert("functions".to_string(), serde_json::Value::Array(functions));
    object.insert(
        "total_copy_count".to_string(),
        serde_json::json!(summary.total_copy_count),
    );
    if let Some(bytes) = summary.total_bytes_copied {
        object.insert("total_bytes_copied".to_string(), serde_json::json!(bytes));
    }
    serde_json::Value::Object(object)
}

fn copy_cost_human(file: &Path, summary: &chelis_ir::analysis::CopyCostSummary) -> String {
    let mut out = format!("file: {}\n", file.display());
    for function in &summary.functions {
        out.push_str(&format!(
            "{}: copy_count={}",
            function.name, function.copy_count
        ));
        if let Some(bytes) = function.bytes_copied {
            out.push_str(&format!(", bytes_copied={bytes}"));
        } else if let Some(formula) = function.byte_formula.as_ref() {
            out.push_str(&format!(", bytes_copied={formula}"));
        }
        out.push('\n');
    }
    out.push_str(&format!("total_copy_count={}", summary.total_copy_count));
    if let Some(bytes) = summary.total_bytes_copied {
        out.push_str(&format!(", total_bytes_copied={bytes}"));
    } else if let Some(formula) = summary.total_byte_formula.as_ref() {
        out.push_str(&format!(", total_bytes_copied={formula}"));
    }
    out.push('\n');
    out
}

/// Exit code returned by `cmd_check` (and the `main` dispatch) when
/// the JSON `errors` array is non-empty. Matches `chelis test`'s
/// exit `2` for "compile test context" failures, the surface that
/// surfaces the same type errors when this gate misses them.
const CHECK_ERRORS_EXIT_CODE: i32 = 2;

/// Canonical message used by both `chelis check` and `chelis build`
/// when the input file parses to zero declarations (empty or
/// whitespace-only `.ch`). Surfaced as a `CheckErrorKind::Other`
/// entry in `cmd_check_one`'s JSON `errors[]` and as the boxed
/// error string from `cmd_build`. Wave-1 red-team finding M2 pins
/// the two surfaces to this same message so `check` and `build`
/// agree on the verdict.
const EMPTY_PROGRAM_MESSAGE: &str = "empty program: no declarations found";

/// Build a synthetic `cmd_check_one` JSON report for an error that
/// short-circuits parsing or program preparation, so the
/// `chelis check` exit-code invariant (issue #207) holds even when
/// the per-file pipeline never reaches the fitness checker. The
/// `errors[]` array carries a single `Other`-kind entry with the
/// supplied message; the rest of the report shape mirrors a zero-
/// node program with score 0.
fn synthetic_check_report_with_error(message: &str) -> String {
    let message_json = serde_json::to_string(message).unwrap_or_else(|_| "\"\"".to_string());
    format!(
        concat!(
            "{{\n",
            "  \"score\": 0,\n",
            "  \"components\": {{\n",
            "    \"parse\": 0,\n",
            "    \"structure\": 0,\n",
            "    \"names\": 0,\n",
            "    \"types\": 0\n",
            "  }},\n",
            "  \"typed_nodes\": 0,\n",
            "  \"untyped_nodes\": 0,\n",
            "  \"total_nodes\": 0,\n",
            "  \"unresolved_names\": [],\n",
            "  \"errors\": [{{\"kind\":\"Other\",\"message\":{},\"severity\":0.5}}]\n",
            "}}"
        ),
        message_json,
    )
}

/// Exit-code contract (issue #207, supersedes RT-205 F7):
///
/// `chelis check` exits `0` iff the JSON `errors` array is empty.
/// Any non-empty `errors` array (type errors, dimension mismatches,
/// validator rejections, effect errors, linearity errors) produces
/// exit code [`CHECK_ERRORS_EXIT_CODE`] (currently `2`, matching
/// `chelis test`'s convention for "compile test context" failures
/// raised by the same kind of error on a different surface).
///
/// Before issue #207, the original RT-205 F7 contract intentionally
/// exited `0` even with errors in the JSON. That contract let
/// downstream CI gates that treat exit `0` as success silently miss
/// type errors that `chelis test` later caught as exit `2`. The
/// machine-facing JSON shape is unchanged; only the process exit
/// status now reflects the errors array.
///
/// Non-zero exit is ALSO produced when the check could not be RUN
/// to completion (style-gate violation, parser/reef failure, IO
/// error). Those propagate through `Result::Err` and pick up the
/// default exit `1` in `main`'s error arm; only the
/// errors-array-non-empty path uses [`CHECK_ERRORS_EXIT_CODE`].
fn cmd_check(
    target: &Path,
    show_inferred: bool,
    allow_style_violations: bool,
) -> Result<i32, Box<dyn std::error::Error>> {
    // Bucket 6b: when given a directory, walk it and run the per-file
    // check on every `.ch` file found. We use the same dot-prefix and
    // `target/` skip rules as `discover_test_files`, so editor tempfiles
    // and build artifacts don't poison the corpus.
    if target.is_dir() {
        let files = discover_check_files(target)?;
        if files.is_empty() {
            // Empty corpus is legitimate (e.g. a fresh `examples/` skeleton).
            // Match `chelis test` ergonomics and report an empty corpus
            // explicitly rather than silently exiting 0 with no output.
            println!("{{\"files\":[],\"errors\":[]}}");
            return Ok(0);
        }
        let mut had_error = false;
        let mut any_errors_in_report = false;
        let mut entries: Vec<String> = Vec::with_capacity(files.len());
        for file in &files {
            match cmd_check_one(file, show_inferred, allow_style_violations) {
                Ok((json, errors_in_report)) => {
                    if errors_in_report {
                        any_errors_in_report = true;
                    }
                    let rel = file.strip_prefix(target).unwrap_or(file).display();
                    entries.push(format!(
                        "{{\"file\":{},\"report\":{json}}}",
                        serde_json::to_string(&rel.to_string()).unwrap_or_default(),
                    ));
                }
                Err(e) => {
                    had_error = true;
                    let rel = file.strip_prefix(target).unwrap_or(file).display();
                    entries.push(format!(
                        "{{\"file\":{},\"error\":{}}}",
                        serde_json::to_string(&rel.to_string()).unwrap_or_default(),
                        serde_json::to_string(&e.to_string()).unwrap_or_default(),
                    ));
                }
            }
        }
        println!("{{\"files\":[{}]}}", entries.join(","));
        if had_error {
            return Err("one or more files failed to check".into());
        }
        // Issue #207 invariant: any file with a non-empty errors array
        // in its report triggers the same exit code as the single-file
        // path. Per-file processing failures (Err arm above) keep the
        // legacy "tooling broken" exit-1 path through `Err`.
        return Ok(if any_errors_in_report {
            CHECK_ERRORS_EXIT_CODE
        } else {
            0
        });
    }

    let (json, errors_in_report) = cmd_check_one(target, show_inferred, allow_style_violations)?;
    println!("{json}");
    Ok(if errors_in_report {
        CHECK_ERRORS_EXIT_CODE
    } else {
        0
    })
}

/// Walk a directory and collect all `.ch` files, mirroring
/// `discover_test_files` exclusion rules (skip dot-prefixed entries and
/// any `target/` directories that accumulate build artifacts).
///
/// The root entry (depth 0) is exempt from the dot-prefix filter so
/// callers can point the walker at e.g. `tempfile::tempdir()` paths
/// (`/tmp/.tmpXyZ/...`) without the entire walk getting filtered out
/// because the temp-dir name starts with a dot.
fn discover_check_files(target: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    let walker = walkdir::WalkDir::new(target)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|entry| {
            if entry.depth() == 0 {
                return true;
            }
            let name = entry.file_name().to_string_lossy();
            if name.starts_with('.') {
                return false;
            }
            if entry.file_type().is_dir() && name == "target" {
                return false;
            }
            true
        });
    for entry in walker {
        let entry = entry.map_err(|e| format!("failed to walk {}: {e}", target.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        // Collect both Surf (`.ch`) and Deep (`.dp`) sources so a mixed
        // directory checks both surfaces. The per-file `cmd_check_one`
        // routes `.dp` through the Deep ingestion helper; both surfaces
        // emit the same CheckResult JSON shape.
        let is_checkable = matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("ch") | Some("dp")
        );
        if !is_checkable {
            continue;
        }
        files.push(path.to_path_buf());
    }
    Ok(files)
}

/// Returns the JSON report plus a flag indicating whether the report's
/// `errors` array is non-empty. The flag is the source of truth for
/// the issue #207 exit-code invariant; callers must thread it back
/// through to the process exit status.
fn cmd_check_one(
    file: &Path,
    show_inferred: bool,
    allow_style_violations: bool,
) -> Result<(String, bool), Box<dyn std::error::Error>> {
    // WI-1 follow-up: run the WHOLE check operation on a grown native stack.
    // The chelis-types check entries grow the stack around their own recursion,
    // but the reef/deep loader, the linked-program `clone()`, the desugarer,
    // the fitness structure walk, `chelis_deep::validate`, and the implicit
    // drop of the deep `Expr` tree all run here, OUTSIDE those entries. A
    // deeply-nested but finite reef-linked program (the Shoals pricer) would
    // otherwise SIGSEGV in one of those derived-recursive passes even though
    // the type checker itself is now safe. One grow at this boundary covers
    // them all uniformly. See docs/investigations/wi1_infer_recursion_depth.md.
    chelis_types::run_on_grown_stack(|| {
        cmd_check_one_on_grown_stack(file, show_inferred, allow_style_violations)
    })
}

fn cmd_check_one_on_grown_stack(
    file: &Path,
    show_inferred: bool,
    allow_style_violations: bool,
) -> Result<(String, bool), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file).ok();
    if let Some(source) = &source {
        style_gate::enforce_style_gate(file, source, allow_style_violations)?;
        emit_advisory_lint_warnings_for_file(file);
    }
    // Deep (`.dp`) ingestion: a standalone `.dp` is already-lowered IR,
    // not a Surf package, so the reef loader below returns `Ok(None)`
    // for it and the monolithic else-arm would feed Deep s-expressions
    // to the Surf parser (which fails with a bogus parse error). Route
    // `.dp` through the dedicated Deep helper before the reef load,
    // mirroring the established `.dp` branches in `cmd_build_dispatch`,
    // `cmd_surf`, `cmd_fmt`, and `copy_cost_for_file`.
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("dp") {
        let source = match &source {
            Some(source) => source,
            None => {
                let json = synthetic_check_report_with_error(&format!(
                    "failed to read {}",
                    file.display()
                ));
                return Ok((json, true));
            }
        };
        return cmd_check_one_deep(source, show_inferred);
    }
    // Wave-1 red-team M1 (#207 follow-up): parse failures used to
    // short-circuit through `?` into the `Err(err)` arm in `main`,
    // emitting no JSON and exiting 1. Catch the parse error here,
    // route it through `synthetic_check_report_with_error`, and let
    // the caller map "errors non-empty" to exit 2 as documented.
    let prepared = match chelis_reef::prepare_program_for_file(file) {
        Ok(prepared) => prepared,
        Err(message) => {
            let json = synthetic_check_report_with_error(&message);
            return Ok((json, true));
        }
    };
    // Wave-1 red-team M2 (#207 follow-up): inside a reef package the
    // file's own contribution lives in `entry_decls` (chelis-std files
    // partition into `stdlib_decls`, but `entry_decls` always reflects
    // the source file the user pointed `check` at). Reject only when
    // the file itself yields zero declarations, not when the rest of
    // the reef graph is empty.
    if let Some(prepared_ref) = &prepared
        && prepared_ref.entry_decls.is_empty()
    {
        let json = synthetic_check_report_with_error(EMPTY_PROGRAM_MESSAGE);
        return Ok((json, true));
    }

    // RFC v5 (RT-1 F2 bypass): inside a reef package every decl checked
    // below (layered fast path AND the monolithic fallback on
    // `prepared.decls`) is reef-linker output, so accept the linker's
    // reserved internal-name format for the rest of this function. Raw
    // single-file `.ch` (prepared == None) and `.dp` (routed earlier
    // through the deep helper) keep the flag FALSE and reject mangled
    // names as forged module identity.
    let _linked_guard = prepared
        .is_some()
        .then(chelis_types::install_linked_program_guard);

    // Layered fast path: when the input resolves inside a reef package
    // (so the chelis-std / non-chelis-std partition is available) and the
    // chelis-std typecheck cache is not disabled, check the non-chelis-std
    // decls `_with_context` against the cached chelis-std sub-context. A
    // type/macro error in the non-chelis-std decls returns `Ok(None)` and
    // we fall through to the monolithic checker so the error-path report
    // stays byte-identical.
    let layered = if let Some(prepared) = &prepared {
        if chelis_compiler_api::cache_disabled() {
            None
        } else {
            chelis_compiler_api::check_layered(&prepared.stdlib_decls, &prepared.non_stdlib_decls)
                .map_err(|e| boxed_string_error(compiler_error_messages(&e)))?
        }
    } else {
        None
    };

    let (report, effect_errors, linearity_errors, inferred_signatures_json) =
        if let Some(layered) = layered {
            let inferred = if show_inferred {
                format_inferred_signatures_json(&layered.typed_program)
            } else {
                String::new()
            };
            (
                layered.fitness,
                layered.effect_errors,
                layered.linearity_errors,
                inferred,
            )
        } else {
            // Monolithic path: full inference over the whole merged
            // program. Used when the input is not inside a reef package,
            // when the cache is disabled, or when the non-chelis-std
            // decls do not type-check clean.
            let decls = match &prepared {
                Some(prepared) => prepared.decls.clone(),
                None => {
                    let source = fs::read_to_string(file)?;
                    // Wave-1 red-team M1 (#207 follow-up): same handling
                    // as the prepared-path parse error above, for the
                    // raw `parse_str` branch used when no reef context
                    // resolves.
                    match chelis_surf::parser::parse_str(&source) {
                        Ok(decls) => decls,
                        Err(err) => {
                            let json = synthetic_check_report_with_error(&err.to_string());
                            return Ok((json, true));
                        }
                    }
                }
            };
            // Wave-1 red-team M2 (#207 follow-up): a parse-clean file
            // with zero declarations (empty or whitespace-only `.ch`)
            // used to report `score=1, errors=[]` and exit 0; both
            // `chelis check` and `chelis build` now reject it with the
            // same canonical message so the two surfaces agree.
            if decls.is_empty() {
                let json = synthetic_check_report_with_error(EMPTY_PROGRAM_MESSAGE);
                return Ok((json, true));
            }
            let deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
            let report = chelis_types::check_ir_fitness(&deep_exprs);
            let typed_program = chelis_types::check_typed_program(&deep_exprs);
            let inferred = if show_inferred {
                typed_program
                    .as_ref()
                    .ok()
                    .map(format_inferred_signatures_json)
                    .unwrap_or_else(|| "[]".to_string())
            } else {
                String::new()
            };
            // Linearity-F3 PR 2: `check_linearity` now returns
            // module-wrapped violations as errors (the PR 1 warning
            // channel was removed once the in-repo corpus was confirmed
            // clean), so this is a flat Ok/Err dispatch with no separate
            // warnings vector.
            let (effect_errors, linearity_errors) = match &typed_program {
                Ok(checked) => match chelis_effects::check_program(checked) {
                    Ok(checked) => (
                        Vec::new(),
                        chelis_types::check_linearity(&checked)
                            .err()
                            .unwrap_or_default(),
                    ),
                    Err(errors) => (errors, Vec::new()),
                },
                Err(_) => (Vec::new(), Vec::new()),
            };
            (report, effect_errors, linearity_errors, inferred)
        };
    assemble_check_json(
        report,
        &effect_errors,
        &linearity_errors,
        &inferred_signatures_json,
        show_inferred,
    )
}

/// Assemble the hand-built `chelis check` JSON report and the issue
/// #207 non-empty-errors flag from the post-pipeline analysis outputs.
///
/// Shared verbatim between the `.ch` monolithic / layered arm of
/// [`cmd_check_one`] and the `.dp` helper [`cmd_check_one_deep`] so the
/// two surfaces emit byte-identical [`chelis_compiler_api::schema::CheckResult`]
/// shapes. The only thing that differs between surfaces is how the
/// `deep_exprs` feeding the fitness / type / effect / linearity checks
/// are produced; the score-adjust and JSON emission MUST NOT drift, so
/// both surfaces call exactly this function.
///
/// Returns `(json, errors_in_report)`; the caller maps a non-empty
/// errors array to [`CHECK_ERRORS_EXIT_CODE`].
fn assemble_check_json(
    mut report: chelis_types::FitnessReport,
    effect_errors: &[chelis_effects::EffectError],
    linearity_errors: &[chelis_types::errors::CheckError],
    inferred_signatures_json: &str,
    show_inferred: bool,
) -> Result<(String, bool), Box<dyn std::error::Error>> {
    if !effect_errors.is_empty() {
        report.score = (report.score - 0.2 * effect_errors.len() as f64).max(0.0);
    }
    if !linearity_errors.is_empty() {
        report.score = (report.score - 0.2 * linearity_errors.len() as f64).max(0.0);
    }
    // Format as JSON manually
    let mut errors_json: Vec<String> = report
        .errors
        .iter()
        .map(|e| {
            format!(
                "{{\"kind\":\"{:?}\",\"message\":{},\"severity\":{}{}{}{}{}}}",
                e.kind,
                serde_json::to_string(&e.message).unwrap_or_default(),
                e.severity,
                e.expected
                    .as_ref()
                    .map(|s| format!(
                        ",\"expected\":{}",
                        serde_json::to_string(s).unwrap_or_default()
                    ))
                    .unwrap_or_default(),
                e.got
                    .as_ref()
                    .map(|s| format!(",\"got\":{}", serde_json::to_string(s).unwrap_or_default()))
                    .unwrap_or_default(),
                e.span_offset
                    .map(|o| format!(",\"span_offset\":{o}"))
                    .unwrap_or_default(),
                e.span_id
                    .as_ref()
                    .map(|s| format!(
                        ",\"span_id\":{}",
                        serde_json::to_string(s).unwrap_or_default()
                    ))
                    .unwrap_or_default(),
            )
        })
        .collect();
    errors_json.extend(effect_errors.iter().map(|e| {
        format!(
            "{{\"kind\":\"{:?}\",\"message\":{},\"severity\":0.8}}",
            e.kind,
            serde_json::to_string(&e.message).unwrap_or_default(),
        )
    }));
    errors_json.extend(linearity_errors.iter().map(|e| {
        format!(
            "{{\"kind\":\"{:?}\",\"message\":{},\"severity\":{}{}{}}}",
            e.kind,
            serde_json::to_string(&e.message).unwrap_or_default(),
            e.severity,
            e.span_offset
                .map(|o| format!(",\"span_offset\":{o}"))
                .unwrap_or_default(),
            e.span_id
                .as_ref()
                .map(|s| format!(
                    ",\"span_id\":{}",
                    serde_json::to_string(s).unwrap_or_default()
                ))
                .unwrap_or_default(),
        )
    }));

    let json = format!(
        concat!(
            "{{\n",
            "  \"score\": {},\n",
            "  \"components\": {{\n",
            "    \"parse\": {},\n",
            "    \"structure\": {},\n",
            "    \"names\": {},\n",
            "    \"types\": {}\n",
            "  }},\n",
            "  \"typed_nodes\": {},\n",
            "  \"untyped_nodes\": {},\n",
            "  \"total_nodes\": {},\n",
            "  \"unresolved_names\": {}{}\n",
            "  \"errors\": [{}]\n",
            "}}"
        ),
        report.score,
        report.components.parse,
        report.components.structure,
        report.components.names,
        report.components.types,
        report.typed_nodes,
        report.untyped_nodes,
        report.total_nodes,
        serde_json::to_string(&report.unresolved_names)?,
        if show_inferred {
            format!(",\n  \"inferred_signatures\": {inferred_signatures_json},")
        } else {
            ",".to_string()
        },
        errors_json.join(","),
    );
    // Issue #207: surface the non-empty-errors flag so the caller can
    // map it to the process exit code. The fitness JSON shape is
    // unchanged; this is purely an out-of-band signal.
    let errors_in_report = !errors_json.is_empty();
    Ok((json, errors_in_report))
}

/// `chelis check` ingestion for a standalone Deep (`.dp`) file.
///
/// A `.dp` is already-lowered IR by construction, so this skips the
/// Surf desugar + macro-expand stage (`expanded_desugared_program`)
/// that the `.ch` arm of [`cmd_check_one`] runs and parses the file
/// directly through the strict Deep parser. Everything downstream of
/// the parse is byte-for-byte the same pipeline the `.ch` arm uses:
/// `check_ir_fitness` -> `check_typed_program` -> `check_program`
/// (effects) -> `check_linearity`, then [`assemble_check_json`].
///
/// `parse_str_strict` (not the non-strict `parse_str`) keeps the `.dp`
/// check surface on the same closed-vocabulary tag gate as
/// `chelis build`, `chelis fmt`, and `chelis cost`: an unknown tag is a
/// hard error rather than a silently-accepted node. Parse failures are
/// caught and routed through `synthetic_check_report_with_error` so a
/// malformed `.dp` produces the same JSON-report-plus-exit-2 shape the
/// `.ch` parse-error path produces, never a propagated boxed `Err`.
fn cmd_check_one_deep(
    source: &str,
    show_inferred: bool,
) -> Result<(String, bool), Box<dyn std::error::Error>> {
    let deep_source = style_gate::strip_deep_lint_directive_lines(source);
    let deep_exprs = match chelis_deep::parser::parse_str_strict(&deep_source) {
        Ok(deep_exprs) => deep_exprs,
        Err(err) => {
            let json = synthetic_check_report_with_error(&err.to_string());
            return Ok((json, true));
        }
    };
    // Parity with the `.ch` empty-file path (a parse-clean file with
    // zero top-level exprs): reject with the same canonical message so
    // the `.dp` and `.ch` surfaces agree.
    if deep_exprs.is_empty() {
        let json = synthetic_check_report_with_error(EMPTY_PROGRAM_MESSAGE);
        return Ok((json, true));
    }
    let report = chelis_types::check_ir_fitness(&deep_exprs);
    let typed_program = chelis_types::check_typed_program(&deep_exprs);
    let inferred = if show_inferred {
        typed_program
            .as_ref()
            .ok()
            .map(format_inferred_signatures_json)
            .unwrap_or_else(|| "[]".to_string())
    } else {
        String::new()
    };
    let (effect_errors, linearity_errors) = match &typed_program {
        Ok(checked) => match chelis_effects::check_program(checked) {
            Ok(checked) => (
                Vec::new(),
                chelis_types::check_linearity(&checked)
                    .err()
                    .unwrap_or_default(),
            ),
            Err(errors) => (errors, Vec::new()),
        },
        Err(_) => (Vec::new(), Vec::new()),
    };
    assemble_check_json(
        report,
        &effect_errors,
        &linearity_errors,
        &inferred,
        show_inferred,
    )
}

fn emit_advisory_lint_warnings_for_file(file: &Path) {
    if style_gate::disabled_by_env() {
        return;
    }
    let parent = file
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let rules = chelis_lint::registry::non_blocking_rules();
    let raw = match chelis_lint::lint(parent, &rules) {
        Ok(violations) => violations,
        Err(_) => return,
    };
    let target_file = fs::canonicalize(file).unwrap_or_else(|_| file.to_path_buf());
    let mine = raw
        .into_iter()
        .filter(|violation| {
            fs::canonicalize(&violation.path).unwrap_or_else(|_| violation.path.clone())
                == target_file
        })
        .collect::<Vec<_>>();
    let exceptions_list = style_gate::exceptions();
    // Anchor exception matching against the detected Cargo workspace
    // root, the same anchor `cmd_lint` uses. If the file is not inside
    // a Cargo workspace, detection fails and no workspace-rooted
    // exception glob can legitimately apply, so the violations pass
    // through unfiltered (correct: a file outside the workspace is not,
    // e.g., `crates/chelis-surf/tests/fixtures/*.ch`). This advisory
    // emit is a best-effort nicety, so a detection failure skips
    // exception filtering rather than aborting `chelis check`.
    let filtered = match style_gate::detect_lint_workspace_root(parent) {
        Ok(workspace_root) => {
            chelis_lint::exceptions::apply_exceptions(&mine, &exceptions_list, &workspace_root)
        }
        Err(_) => mine,
    };
    for violation in filtered {
        // Parity with `cmd_lint`'s emit path (LP-LEAK-A / LP-LEAK-B):
        // suppress warnings for rules that opt in to `check_mirrors_fix`
        // when the autofix would silently decline or be rejected by the
        // typed-pipeline gate. Without this filter `chelis check` floods
        // with the same non-actionable false positives that
        // `chelis lint --check` already suppresses, because the advisory
        // emit path applied only the path-glob exception filter. Both
        // code paths now run `should_suppress_unfixable_violation`, so
        // the two cannot drift again. `parent` is the lint walk root,
        // passed as `target` exactly as `cmd_lint` does.
        if should_suppress_unfixable_violation(parent, &rules, &violation) {
            continue;
        }
        eprintln!("warning: {violation}");
    }
}

fn format_inferred_signatures_json(checked: &chelis_types::CheckedProgram) -> String {
    // Per-def inferred effect rows, keyed by def name. Computed from the
    // same `CheckedProgram` so the structured effect-row a consumer
    // (Hull) reads is the exact row `chelis check` infers. Functions
    // with no effects map to an empty row (`[]`), which is distinct from
    // "effects unknown".
    let effect_rows = chelis_effects::def_effect_rows(checked);
    let entries: Vec<serde_json::Value> = checked
        .signature_inference()
        .functions
        .values()
        .map(|func| {
            let params: Vec<serde_json::Value> = func
                .params
                .iter()
                .map(|param| {
                    serde_json::json!({
                        "index": param.index,
                        "name": param.name,
                        "written": param.written,
                        "inferred_read_only": param.inferred_read_only,
                        // Human-facing display strings (unchanged).
                        "checked_type": format_cli_type(&param.checked_type),
                        "display_type": format_cli_type(&param.display_type),
                        // Structured, lossless type trees (new).
                        "checked_type_structured": wire_inferred_type(&param.checked_type),
                        "display_type_structured": wire_inferred_type(&param.display_type),
                    })
                })
                .collect();
            let effect_row = effect_rows.get(&func.name);
            serde_json::json!({
                "function": func.name,
                "recursive_cycle": func.recursive_cycle,
                // Human-facing display strings (unchanged).
                "checked_signature": format_cli_type(&func.checked_signature),
                "display_signature": format_cli_type(&func.display_signature),
                // Structured, lossless signature type trees (new).
                "checked_signature_structured": wire_inferred_type(&func.checked_signature),
                "display_signature_structured": wire_inferred_type(&func.display_signature),
                // Structured effect row (new). Always present; empty for
                // a pure function. Also expose the human Display spelling
                // (`Random`/`Accum`/`IO`/`Test`/`Resource("dev")`) for
                // parity with stderr diagnostics.
                "effect_row": wire_effect_row(effect_row),
                "effect_row_display": effect_row_display(effect_row),
                "params": params,
            })
        })
        .collect();
    serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string())
}

/// Convert a checker [`Type`] into the lossless, serde-friendly
/// [`WireInferredType`] tree emitted by `chelis check --show-inferred
/// --json`. This is the structured counterpart to [`format_cli_type`];
/// the two must stay in lockstep on every `Type` variant.
fn wire_inferred_type(ty: &Type) -> WireInferredType {
    match ty {
        Type::Prim(prim) => WireInferredType::Prim {
            name: prim.name().to_string(),
        },
        Type::Fn(args, ret) => WireInferredType::Fn {
            args: args.iter().map(wire_inferred_type).collect(),
            ret: Box::new(wire_inferred_type(ret)),
        },
        Type::Ref(inner) => WireInferredType::Ref {
            inner: Box::new(wire_inferred_type(inner)),
        },
        Type::Tensor(dims, prec) => WireInferredType::Tensor {
            dims: dims.iter().map(wire_inferred_dim).collect(),
            precision: wire_inferred_precision(prec),
        },
        Type::Adt(name, args) => WireInferredType::Adt {
            name: name.clone(),
            args: args.iter().map(wire_inferred_type).collect(),
        },
        Type::Var(var) => WireInferredType::Var { id: var.0 },
        Type::Tuple(types) => WireInferredType::Tuple {
            items: types.iter().map(wire_inferred_type).collect(),
        },
        Type::Unit => WireInferredType::Unit,
        Type::Error(_) => WireInferredType::Error,
    }
}

/// Convert a checker [`Dim`] into a [`WireInferredDim`]. Structured
/// counterpart to [`format_cli_dim`].
fn wire_inferred_dim(dim: &Dim) -> WireInferredDim {
    match dim {
        Dim::Name(name) => WireInferredDim::Name { name: name.clone() },
        Dim::Var(var) => WireInferredDim::Var { id: var.0 },
        Dim::Lit(value) => WireInferredDim::Lit { size: *value },
        Dim::Wildcard => WireInferredDim::Wildcard,
        Dim::Rank(rank) => WireInferredDim::Rank { id: rank.0 },
    }
}

/// Convert a checker [`TensorPrec`] into a [`WireInferredPrecision`].
fn wire_inferred_precision(prec: &TensorPrec) -> WireInferredPrecision {
    match prec {
        TensorPrec::Concrete(prim) => WireInferredPrecision::Concrete {
            name: prim.name().to_string(),
        },
        TensorPrec::Var(var) => WireInferredPrecision::Var { id: var.0 },
    }
}

/// Convert a single checker [`Effect`] into a [`WireInferredEffect`].
fn wire_inferred_effect(effect: &Effect) -> WireInferredEffect {
    match effect {
        Effect::Random => WireInferredEffect::Random,
        Effect::Accum => WireInferredEffect::Accum,
        Effect::Io => WireInferredEffect::Io,
        Effect::Test => WireInferredEffect::Test,
        Effect::Resource(device) => WireInferredEffect::Resource {
            device: device.clone(),
        },
    }
}

/// Render an effect row as the structured wire list. A `None` row
/// (function name absent from the effect map) and an empty row both
/// serialize to `[]`: a pure function has no effects either way.
fn wire_effect_row(row: Option<&EffectSet>) -> Vec<WireInferredEffect> {
    match row {
        Some(set) => set.iter().map(wire_inferred_effect).collect(),
        None => Vec::new(),
    }
}

/// Render an effect row as the human Display spellings
/// (`Random`/`Accum`/`IO`/`Test`/`Resource("dev")`), matching the
/// `chelis_types::types::Effect` Display impl used in stderr
/// diagnostics. Stable ordering follows `EffectSet`'s `BTreeSet`.
fn effect_row_display(row: Option<&EffectSet>) -> Vec<String> {
    match row {
        Some(set) => set.iter().map(ToString::to_string).collect(),
        None => Vec::new(),
    }
}

fn format_cli_type(ty: &Type) -> String {
    match ty {
        Type::Prim(prim) => prim.name().to_string(),
        Type::Fn(args, ret) => {
            let args = args.iter().map(format_cli_type).collect::<Vec<_>>();
            format!("({}) -> {}", args.join(", "), format_cli_type(ret))
        }
        Type::Ref(inner) => format!("&{}", format_cli_type(inner)),
        Type::Tensor(dims, prim) => {
            let mut parts = dims.iter().map(format_cli_dim).collect::<Vec<_>>();
            parts.push(prim.name().to_string());
            format!("tensor[{}]", parts.join(", "))
        }
        Type::Adt(name, args) if args.is_empty() => name.clone(),
        Type::Adt(name, args) => {
            let args = args.iter().map(format_cli_type).collect::<Vec<_>>();
            format!("{name}[{}]", args.join(", "))
        }
        Type::Var(var) => format!("?{}", var.0),
        Type::Tuple(types) => {
            let types = types.iter().map(format_cli_type).collect::<Vec<_>>();
            format!("({})", types.join(", "))
        }
        Type::Unit => "unit".to_string(),
        Type::Error(_) => "<error>".to_string(),
    }
}

fn format_cli_dim(dim: &Dim) -> String {
    match dim {
        Dim::Name(name) => name.clone(),
        Dim::Var(var) => format!("d{}", var.0),
        Dim::Lit(value) => value.to_string(),
        Dim::Wildcard => "*".to_string(),
        Dim::Rank(rank) => format!("..r{}", rank.0),
    }
}

/// Bucket-5 closure: `with seed(...)` no longer blocks `chelis build`.
/// Direct `uniform_like` DAG lowering can bake the handled seed into
/// `RiscOp::UniformLike { seed }`; generated C host code also preserves
/// nested handler scopes with runtime RNG state so stdlib/user helpers
/// that call `uniform_like` draw from the active seed. Seeded dropout
/// backend codegen remains outside this hook's shipped coverage.
///
/// This function is retained as a forward-compatibility hook for
/// future user-defined effect handlers that the backends genuinely
/// cannot lower yet. Today it is a no-op.
fn reject_with_seed_for_build_target(
    _decls: &[Decl],
    _target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    Ok(())
}

/// Dispatch between the Surf and Deep ingestion paths per the rules in
/// `spec/design/chelis_span_survival.md` §2.5.
///
/// | Invocation               | Path |
/// |--------------------------|------|
/// | `chelis build foo.dp`    | Deep (auto-detect) |
/// | `chelis build foo.dp --deep` | Deep (flag agrees with extension) |
/// | `chelis build foo.ch --deep` | Deep (flag overrides) |
/// | `chelis build foo.ch`    | Surf (today's behavior) |
///
/// `--no-deep` is intentionally absent. To force a `.dp` file through
/// Surf, rename it or pipe through `chelis surf`.
fn cmd_build_dispatch(
    file: &std::path::Path,
    output: Option<&std::path::Path>,
    target: &str,
    deep_flag: bool,
    allow_style_violations: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let extension_is_dp = file
        .extension()
        .and_then(|s| s.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("dp"));
    let route_through_deep = deep_flag || extension_is_dp;
    if route_through_deep {
        if deep_flag && !extension_is_dp {
            // Override case: warn if the file is clearly Surf source. The
            // strict-parse step below produces the authoritative error if
            // the contents really are not Deep; this is just a friendly
            // heads-up so a user who fat-fingered `--deep` on a `.ch` file
            // doesn't get a confusing parse error first.
            let preview = std::fs::read_to_string(file)
                .ok()
                .and_then(|s| s.lines().take(3).collect::<Vec<_>>().join("\n").into());
            if let Some(text) = preview
                && (text.starts_with("import ")
                    || text.starts_with("module ")
                    || text.starts_with("def ")
                    || text.starts_with("export "))
            {
                eprintln!(
                    "warning: `--deep` was passed but {} looks like Surf source \
                     (starts with `import`/`module`/`def`/`export`). Routing through \
                     the Deep ingestion path anyway; rename to `.dp` or drop `--deep` \
                     to silence this warning.",
                    file.display()
                );
            }
        }
        cmd_build_deep(file, output, target, allow_style_violations)
    } else {
        cmd_build(file, output, target, allow_style_violations)
    }
}

fn cmd_build(
    file: &std::path::Path,
    output: Option<&std::path::Path>,
    target: &str,
    allow_style_violations: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(source) = fs::read_to_string(file) {
        style_gate::enforce_style_gate(file, &source, allow_style_violations)?;
    }
    let prepared = chelis_reef::prepare_program_for_file(file).map_err(boxed_string_error)?;
    // RFC v5 (RT-1 F2 bypass): a reef-prepared build checks reef-linker
    // output; accept the linker's reserved internal-name format. Raw
    // `.ch`/`.dp` builds keep the flag FALSE and reject mangled names.
    let _linked_guard = prepared
        .is_some()
        .then(chelis_types::install_linked_program_guard);
    let (decls, entry_decls) = match &prepared {
        Some(prepared) => (prepared.decls.clone(), prepared.entry_decls.clone()),
        None => {
            let source = fs::read_to_string(file)?;
            let decls = chelis_surf::parser::parse_str(&source)?;
            (decls.clone(), decls)
        }
    };
    // Wave-1 red-team M2 (#207 follow-up): align with `chelis check`
    // and reject a zero-declaration program rather than emitting a
    // degenerate no-op C function. For reef-prepared programs the
    // user's contribution lives in `entry_decls`; outside a reef the
    // raw `decls` carry it directly.
    let user_decls_empty = match &prepared {
        Some(_) => entry_decls.is_empty(),
        None => decls.is_empty(),
    };
    if user_decls_empty {
        return Err(boxed_string_error(EMPTY_PROGRAM_MESSAGE.to_string()));
    }
    reject_with_seed_for_build_target(&decls, target)?;
    let full_deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
    let entry_deep_exprs = expanded_desugared_program(&entry_decls).map_err(boxed_string_error)?;
    // chelis#334: drop dead library defs that use an eval-only host builtin
    // (`process_run`) so an unused transitive dependency module — e.g.
    // chelis-std's `Std.Process` — cannot force them into the compiled
    // lowering target and trip the build gate. These defs can never appear
    // in a compiled artifact, so they are not part of the host-library
    // surface worth preserving. A *reachable* eval-only use is left in place
    // for the build gate to reject with a clean diagnostic.
    let full_deep_exprs = drop_unreachable_eval_only_defs(full_deep_exprs, &entry_deep_exprs);
    let pruned_deep_exprs =
        prune_build_program_to_reachable_defs(&full_deep_exprs, &entry_deep_exprs);

    // Layered build fast path: when the input resolves inside a reef
    // package, the chelis-std typecheck cache is enabled, AND build-time
    // pruning did not drop any decls (so the full program is the
    // lowering target), reuse the cached chelis-std sub-context for the
    // type-check stage instead of re-inferring chelis-std. When pruning
    // fires the layered whole-program `CheckedProgram` would not match
    // the pruned lowering target, so the monolithic path is used.
    // `check_layered_for_build` returns `Ok(None)` on any non-chelis-std
    // type/effect/linearity error, falling back to monolithic so the
    // error-path output stays byte-identical.
    let layered_full_checked: Option<chelis_types::CheckedProgram> = match &prepared {
        Some(prepared)
            if !chelis_compiler_api::cache_disabled()
                && pruned_deep_exprs.len() == full_deep_exprs.len() =>
        {
            chelis_compiler_api::check_layered_for_build(
                &prepared.stdlib_decls,
                &prepared.non_stdlib_decls,
            )
            .map_err(|e| boxed_string_error(compiler_error_messages(&e)))?
        }
        _ => None,
    };

    let preserve_host_library_surface = if prepared.is_none()
        && target == "c"
        && pruned_deep_exprs.len() != full_deep_exprs.len()
    {
        let full_checked = checked_program_with_effects(&full_deep_exprs)
            .map_err(|e| format!("Check errors: {e}"))?;
        reject_host_only_builtins_before_host_lowering(&full_checked, target)?;
        chelis_ir::host::try_lower_compiled_program(&full_checked)
            .map_err(|diagnostic| format!("Lowering error: {diagnostic}"))?
            .host
            .as_ref()
            .map(chelis_ir::host::host_program_requires_host_backend)
            .unwrap_or(false)
    } else {
        false
    };
    // Cross-module checks (e.g. the §opaque-encapsulation rule) reject a
    // reference to an unexported producer whose signature mentions an
    // opaque type. That producer is unreachable from the entry point, so
    // build-time pruning drops it; checking only the pruned program would
    // then report the bare reference as a plain unbound variable and mask
    // the `OpaqueTypeViolation`. Mirror `chelis check`: when pruning fired
    // for a reef-prepared package, run the cross-module check against the
    // full program first so the encapsulation diagnostic surfaces, then
    // fall through to the existing pruned-lowering path (which preserves
    // the reef pricer/layered-cache lowering target unchanged).
    if prepared.is_some() && pruned_deep_exprs.len() != full_deep_exprs.len() {
        checked_program_with_effects(&full_deep_exprs).map_err(|e| format!("Check errors: {e}"))?;
    }
    let deep_exprs = if preserve_host_library_surface {
        full_deep_exprs
    } else {
        pruned_deep_exprs
    };
    let symbolic_dims = collect_symbolic_dims_from_deep(&deep_exprs);
    // Use the layered whole-program `CheckedProgram` when it is available
    // (no-pruning case) and the deep-exprs being lowered are the full
    // program; otherwise check monolithically.
    let checked = match layered_full_checked {
        Some(checked) if deep_exprs.len() == checked.exprs().len() => checked,
        _ => checked_program_with_effects(&deep_exprs).map_err(|e| format!("Check errors: {e}"))?,
    };
    chelis_effects::validate_build_target(&checked, target)
        .map_err(|errors| format_effect_errors(&errors))?;
    reject_host_only_builtins_before_host_lowering(&checked, target)?;
    let mut compiled_program = chelis_ir::host::try_lower_compiled_program(&checked)
        .map_err(|diagnostic| format!("Lowering error: {diagnostic}"))?;
    emit_summary_rejections(compiled_program.host.as_ref());
    let mut dag = lower_checked_for_cli(&checked, compiled_program.host.as_ref())?;
    let all_root_names = lowered_root_names_from_exprs(&deep_exprs, checked.type_env());
    let entry_root_names =
        lowered_root_names_from_decls(&entry_decls, &deep_exprs, checked.type_env());
    let entry_display_root_names = root_names_from_decls(&entry_decls, checked.type_env())
        .into_iter()
        .map(|name| {
            name.rsplit_once("__")
                .map(|(_, tail)| tail.to_string())
                .unwrap_or(name)
        })
        .collect::<Vec<_>>();
    if let Some(host_program) = compiled_program.host.as_mut() {
        host_program.globals = host_program
            .globals
            .iter()
            .map(|binding| {
                let mut binding = binding.clone();
                binding.display_name = match binding.ty {
                    chelis_ir::ConcreteHostType::Function(_, _) => None,
                    _ => host_display_root_name(&binding.name, &entry_display_root_names).or_else(
                        || {
                            // Tuple-typed top-level bindings get their root
                            // name expanded into `name.0` / `name.1` entries
                            // by `extend_root_names_from_value` (matching
                            // eval-side behavior). Surface a synthetic
                            // tuple-prefix display name so the C emitter
                            // can render the per-field "name.i = ..." lines.
                            if matches!(&binding.ty, chelis_ir::ConcreteHostType::Tuple(_)) {
                                host_display_tuple_root_prefix(
                                    &binding.name,
                                    &entry_display_root_names,
                                )
                            } else {
                                None
                            }
                        },
                    ),
                };
                binding
            })
            .collect();
    }
    let selected = all_root_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            if entry_root_names.iter().any(|entry| entry == name) {
                dag.roots().get(index).copied()
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if !entry_root_names.is_empty() {
        dag.set_roots(selected);
    }
    dag = chelis_ir::optimize::dead_code_eliminate(&dag);
    let func_name = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("chelis_main");

    match target {
        "c" => {
            if let Some(host_program) = compiled_program.host.as_ref()
                && (chelis_ir::host::host_program_requires_host_backend(host_program)
                    || dag.roots().is_empty()
                    || !host_program.functions.is_empty())
            {
                // AD-transform UX only: the lowerer marks an AD transform
                // it could not resolve with the dedicated transform
                // marker, so the workaround text names exactly those
                // defs. A plain unresolved callable value falls through
                // to `codegen_host_program`, whose ABI projection rejects
                // its marker with the frozen `unsupported:` diagnostic,
                // regardless of whether an unrelated grad/vmap exists
                // elsewhere in the program (chelis#730, chelis#841).
                let unresolved =
                    chelis_ir::host::host_program_unresolved_transform_sites(host_program);
                if !unresolved.is_empty() {
                    return Err(format!(
                        "`chelis build --target c` can't lower these defs. Their body \
                         applies/binds `grad` (or `vmap`) in a position the host lane \
                         can't resolve (inline `grad(f)(x)` or `g = grad(f); g(x)`). \
                         Workaround that compiles today: make the function you want to \
                         differentiate a parameter of the enclosing def, then call \
                         `grad(local, wrt=(arg))(arg)` where `local` is a locally-bound \
                         fn that uses the parameter; and make sure that function uses \
                         only pure tensor ops (sum, add, mul, einsum, etc.): `grad` \
                         through host-lane `fold`/`map` is not currently supported, \
                         rewrite to `tensor_to_scalar(sum(mul(v, v), 0))` or `einsum`. \
                         See `build_c_tensor_grad_local_wrapper_over_function_param_builds` \
                         in crates/chelis-cli/tests/cli.rs for a compiling example. \
                         Affected defs: {}",
                        unresolved.join(", ")
                    )
                    .into());
                }
                reject_eval_only_builtins_host(host_program)?;
                reject_unsupported_c_precisions_host(host_program)?;
                reject_symbolic_windowed_reduce_host(host_program, "c")?;
                reject_unsupported_reduce_window_precision_host(host_program, "c")?;
                let result = chelis_backend_c::codegen_host_program(host_program, func_name)?;
                cmd_build_c_result(result, func_name, output, &symbolic_dims)
            } else {
                reject_unsupported_effect_ops(&dag, "c")?;
                reject_unsupported_c_precisions(&dag)?;
                reject_symbolic_windowed_reduce(&dag, "c")?;
                reject_unsupported_reduce_window_precision(&dag, "c")?;
                let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
                let fused = chelis_ir::fuse::fuse(&specialized);
                cmd_build_c(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        "hip" => {
            if let Some(host_program) = compiled_program.host.as_ref() {
                reject_eval_only_builtins_host(host_program)?;
            }
            let host_requires_host_backend = compiled_program
                .host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            // NOTE: for programs without a `main` and with multiple
            // sibling tensor-signature defs, the "preferred" entry falls
            // back to the last fn and silently drops the others. This is a
            // known HIP backend limitation — the backend is single-entry
            // by design. Tracked as a residual issue.
            let preferred_entry_dag = compiled_program
                .host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name)
                .and_then(|name| chelis_ir::host::lower_named_tensor_entry_dag(&checked, name));
            if dag.roots().is_empty()
                && preferred_entry_dag.is_none()
                && host_requires_host_backend
                && let Some(host_program) = compiled_program.host.as_ref()
            {
                let result = chelis_backend_c::codegen_host_program(host_program, func_name)?;
                cmd_build_hip_host(result, func_name, output)
            } else {
                let mut hip_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    lower_checked_for_cli(&checked, compiled_program.host.as_ref())?
                };
                hip_dag = chelis_ir::optimize::dead_code_eliminate(&hip_dag);
                reject_unsupported_effect_ops(&hip_dag, "hip")?;
                let specialized = chelis_ir::specialize::specialize_for_blas(&hip_dag);
                reject_unsupported_hip_ops(&specialized)?;
                let fused = chelis_ir::fuse::fuse(&specialized);
                cmd_build_hip(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        "metal" => {
            if let Some(host_program) = compiled_program.host.as_ref() {
                reject_eval_only_builtins_host(host_program)?;
            }
            let host_requires_host_backend = compiled_program
                .host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            // Same single-entry limitation as HIP: programs without a `main`
            // and with multiple sibling tensor-signature defs fall back to
            // the preferred entry; others are silently dropped. Tracked as
            // a residual issue mirroring HIP.
            let preferred_entry_dag = compiled_program
                .host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name)
                .and_then(|name| chelis_ir::host::lower_named_tensor_entry_dag(&checked, name));
            if dag.roots().is_empty()
                && preferred_entry_dag.is_none()
                && host_requires_host_backend
                && let Some(host_program) = compiled_program.host.as_ref()
            {
                // Host-only programs fall through to the C backend, exactly
                // like the HIP path. The metal path doesn't have a separate
                // host wrapper today; reuse cmd_build_hip_host for parity.
                let result = chelis_backend_c::codegen_host_program(host_program, func_name)?;
                cmd_build_hip_host(result, func_name, output)
            } else {
                let mut metal_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    lower_checked_for_cli(&checked, compiled_program.host.as_ref())?
                };
                metal_dag = chelis_ir::optimize::dead_code_eliminate(&metal_dag);
                reject_unsupported_effect_ops(&metal_dag, "metal")?;
                reject_unsupported_metal_ops(&metal_dag)?;
                // F4: IR validation pass for the Metal admissible-precision
                // matrix per spec/04-type-system.md §1.1.3. The spec names
                // three rejection surfaces; this is the second (the CLI
                // gate `reject_unsupported_metal_ops` above is the first;
                // `Emitter::require_metal_admissible` in the backend is
                // the third). All three share the same diagnostic text
                // so the user sees one voice regardless of which surface
                // catches the f64 first.
                chelis_ir::verify::validate_metal_admissible_precisions(&metal_dag)?;
                let fused = chelis_ir::fuse::fuse(&metal_dag);
                cmd_build_metal(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        other => Err(format!("unknown target '{other}': expected 'c', 'hip', or 'metal'").into()),
    }
}

/// Deep-source ingestion path for `chelis build`.
///
/// Mirrors the shape of `cmd_build` but reads Deep s-expression text
/// directly via `chelis_deep::parser::parse_str_strict` and skips the
/// Surf desugar / macro-expand phase (Deep is canonical post-expansion
/// per `spec/03-deep-syntax.md` §2). All metadata — including span IDs
/// — flows through the existing `chelis_types::check_ir_program`
/// → `chelis_ir::lower::lower_program` → optimization → backend
/// codegen path; this function is plumbing, not new semantics.
///
/// The S5 oracle: span-attributed Deep produces span-annotated
/// C/HIP/Metal output (the per-op `// span: <id>` comments emitted by
/// every backend in S4). Span-free Deep produces output equivalent to
/// the Surf path for the same logical program, modulo absent spans.
fn cmd_build_deep(
    file: &std::path::Path,
    output: Option<&std::path::Path>,
    target: &str,
    allow_style_violations: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    style_gate::enforce_style_gate(file, &source, allow_style_violations)?;
    let deep_source = style_gate::strip_deep_lint_directive_lines(&source);
    let deep_exprs = chelis_deep::parser::parse_str_strict(&deep_source)
        .map_err(|err| format!("Deep parse error: {err}"))?;

    // Deep ingestion has no separate "entry decls" concept — the whole
    // .dp file is the program. Treat every top-level def as an entry
    // candidate; the existing pruner (`prune_build_program_to_reachable_defs`)
    // will trim unreachable defs.
    let entry_deep_exprs = deep_exprs.clone();
    let pruned_deep_exprs = prune_build_program_to_reachable_defs(&deep_exprs, &entry_deep_exprs);
    let preserve_host_library_surface =
        if target == "c" && pruned_deep_exprs.len() != deep_exprs.len() {
            let full_checked = checked_program_with_effects(&deep_exprs)
                .map_err(|e| format!("Check errors: {e}"))?;
            reject_host_only_builtins_before_host_lowering(&full_checked, target)?;
            chelis_ir::host::try_lower_compiled_program(&full_checked)
                .map_err(|diagnostic| format!("Lowering error: {diagnostic}"))?
                .host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false)
        } else {
            false
        };
    let final_deep_exprs = if preserve_host_library_surface {
        deep_exprs.clone()
    } else {
        pruned_deep_exprs
    };
    let symbolic_dims = collect_symbolic_dims_from_deep(&final_deep_exprs);
    let checked = checked_program_with_effects(&final_deep_exprs)
        .map_err(|e| format!("Check errors: {e}"))?;
    chelis_effects::validate_build_target(&checked, target)
        .map_err(|errors| format_effect_errors(&errors))?;
    reject_host_only_builtins_before_host_lowering(&checked, target)?;
    let mut compiled_program = chelis_ir::host::try_lower_compiled_program(&checked)
        .map_err(|diagnostic| format!("Lowering error: {diagnostic}"))?;
    emit_summary_rejections(compiled_program.host.as_ref());
    let mut dag = lower_checked_for_cli(&checked, compiled_program.host.as_ref())?;
    let all_root_names = lowered_root_names_from_exprs(&final_deep_exprs, checked.type_env());
    let entry_root_names = lowered_root_names_from_exprs(&entry_deep_exprs, checked.type_env());
    let entry_display_root_names = root_names_from_exprs(&entry_deep_exprs, checked.type_env())
        .into_iter()
        .map(|name| {
            name.rsplit_once("__")
                .map(|(_, tail)| tail.to_string())
                .unwrap_or(name)
        })
        .collect::<Vec<_>>();
    if let Some(host_program) = compiled_program.host.as_mut() {
        host_program.globals = host_program
            .globals
            .iter()
            .map(|binding| {
                let mut binding = binding.clone();
                binding.display_name = match binding.ty {
                    chelis_ir::ConcreteHostType::Function(_, _) => None,
                    _ => host_display_root_name(&binding.name, &entry_display_root_names).or_else(
                        || {
                            if matches!(&binding.ty, chelis_ir::ConcreteHostType::Tuple(_)) {
                                host_display_tuple_root_prefix(
                                    &binding.name,
                                    &entry_display_root_names,
                                )
                            } else {
                                None
                            }
                        },
                    ),
                };
                binding
            })
            .collect();
    }
    let selected = all_root_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            if entry_root_names.iter().any(|entry| entry == name) {
                dag.roots().get(index).copied()
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if !entry_root_names.is_empty() {
        dag.set_roots(selected);
    }
    dag = chelis_ir::optimize::dead_code_eliminate(&dag);
    let func_name = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("chelis_main");

    match target {
        "c" => {
            if let Some(host_program) = compiled_program.host.as_ref()
                && (chelis_ir::host::host_program_requires_host_backend(host_program)
                    || dag.roots().is_empty()
                    || !host_program.functions.is_empty())
            {
                // Same split as the Surf lane: the transform marker earns
                // the grad/vmap workaround text; plain callable markers
                // reach ABI projection's frozen diagnostic instead
                // (chelis#841).
                let unresolved =
                    chelis_ir::host::host_program_unresolved_transform_sites(host_program);
                if !unresolved.is_empty() {
                    return Err(format!(
                        "`chelis build --deep --target c` can't lower these defs: \
                         they apply/bind `grad` (or `vmap`) in a position the host \
                         lane can't resolve. Affected defs: {}",
                        unresolved.join(", ")
                    )
                    .into());
                }
                reject_eval_only_builtins_host(host_program)?;
                reject_unsupported_c_precisions_host(host_program)?;
                reject_symbolic_windowed_reduce_host(host_program, "c")?;
                reject_unsupported_reduce_window_precision_host(host_program, "c")?;
                let result = chelis_backend_c::codegen_host_program(host_program, func_name)?;
                cmd_build_c_result(result, func_name, output, &symbolic_dims)
            } else {
                reject_unsupported_effect_ops(&dag, "c")?;
                reject_unsupported_c_precisions(&dag)?;
                reject_symbolic_windowed_reduce(&dag, "c")?;
                reject_unsupported_reduce_window_precision(&dag, "c")?;
                let specialized = chelis_ir::specialize::specialize_for_blas(&dag);
                let fused = chelis_ir::fuse::fuse(&specialized);
                cmd_build_c(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        "hip" => {
            if let Some(host_program) = compiled_program.host.as_ref() {
                reject_eval_only_builtins_host(host_program)?;
            }
            let host_requires_host_backend = compiled_program
                .host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            let preferred_entry_dag = compiled_program
                .host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name)
                .and_then(|name| chelis_ir::host::lower_named_tensor_entry_dag(&checked, name));
            if dag.roots().is_empty()
                && preferred_entry_dag.is_none()
                && host_requires_host_backend
                && let Some(host_program) = compiled_program.host.as_ref()
            {
                let result = chelis_backend_c::codegen_host_program(host_program, func_name)?;
                cmd_build_hip_host(result, func_name, output)
            } else {
                let mut hip_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    lower_checked_for_cli(&checked, compiled_program.host.as_ref())?
                };
                hip_dag = chelis_ir::optimize::dead_code_eliminate(&hip_dag);
                reject_unsupported_effect_ops(&hip_dag, "hip")?;
                let specialized = chelis_ir::specialize::specialize_for_blas(&hip_dag);
                reject_unsupported_hip_ops(&specialized)?;
                let fused = chelis_ir::fuse::fuse(&specialized);
                cmd_build_hip(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        "metal" => {
            if let Some(host_program) = compiled_program.host.as_ref() {
                reject_eval_only_builtins_host(host_program)?;
            }
            let host_requires_host_backend = compiled_program
                .host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            let preferred_entry_dag = compiled_program
                .host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name)
                .and_then(|name| chelis_ir::host::lower_named_tensor_entry_dag(&checked, name));
            if dag.roots().is_empty()
                && preferred_entry_dag.is_none()
                && host_requires_host_backend
                && let Some(host_program) = compiled_program.host.as_ref()
            {
                let result = chelis_backend_c::codegen_host_program(host_program, func_name)?;
                cmd_build_hip_host(result, func_name, output)
            } else {
                let mut metal_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    lower_checked_for_cli(&checked, compiled_program.host.as_ref())?
                };
                metal_dag = chelis_ir::optimize::dead_code_eliminate(&metal_dag);
                reject_unsupported_effect_ops(&metal_dag, "metal")?;
                reject_unsupported_metal_ops(&metal_dag)?;
                // F4: IR validation pass; see cmd_build for the full
                // rationale. This is the same surface from the Deep
                // ingestion path so symbolic-dim and span-attributed
                // Deep get the same f64 rejection behavior.
                chelis_ir::verify::validate_metal_admissible_precisions(&metal_dag)?;
                let fused = chelis_ir::fuse::fuse(&metal_dag);
                cmd_build_metal(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        other => Err(format!("unknown target '{other}': expected 'c', 'hip', or 'metal'").into()),
    }
}

/// Handle a clap parse failure. §5.4: an unrecognized subcommand gets
/// clap's own message plus the cross-version hint; every other clap outcome
/// (help, version, unknown flag, missing/duplicate arg) defers to clap so
/// exit codes and rendering stay byte-for-byte identical to `Cli::parse()`.
/// Never returns.
fn handle_parse_error(e: clap::Error) -> ! {
    if e.kind() == clap::error::ErrorKind::InvalidSubcommand {
        // clap's message names the offending token (and any "did you mean"
        // tip) on stderr; append the hint and exit with clap's own code (2).
        let _ = e.print();
        eprint_pin_hint();
        std::process::exit(2);
    }
    e.exit();
}

/// Print the §5.4 cross-version hint after an unrecognized-subcommand error:
/// name this chelis's own version and, when resolvable, the pin source that
/// routed here, then point at the `+<ver>` override. Deliberately
/// version-generic: a chelis only errors on verbs newer than itself,
/// exactly the set it cannot name.
fn eprint_pin_hint() {
    let version = env!("CARGO_PKG_VERSION");
    let pin_clause = pin_source_clause();
    eprintln!();
    eprintln!(
        "You are running chelis {version}{pin_clause}. If this is a newer command, \
         run it with a toolchain that has it: `chelis +<ver> ...` \
         (see `chelis --version` and `chelisup list-installed`)."
    );
}

/// Best-effort `, pinned by <source>` clause naming what routed this
/// invocation (via the same resolver the shim uses), or an empty string
/// when nothing resolves, the chelis home is unavailable, or the cwd is
/// unreadable.
fn pin_source_clause() -> String {
    let store = match chelisup::paths::Store::from_env() {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    // The shim has already stripped any leading `+<ver>`, so re-resolving
    // from the forwarded args names the directory/env/pin source that
    // selected this toolchain.
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    // An unreadable cwd (e.g. deleted under us) would degrade the resolver
    // walk to a bare relative path; omit the clause instead.
    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(_) => return String::new(),
    };
    let input = chelisup::resolve::ResolveInput {
        args: &args,
        env_toolchain: std::env::var("CHELIS_TOOLCHAIN").ok(),
        cwd: &cwd,
        store: &store,
    };
    match chelisup::resolve::resolve(&input) {
        Some(r) => format!(", pinned by {}", r.source),
        None => String::new(),
    }
}

fn cmd_reef(command: ReefCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ReefCommand::Init {
            name,
            module_prefix,
            output,
        } => {
            let root = output.unwrap_or_else(|| PathBuf::from("."));
            chelis_reef::init_package(&root, &name, &module_prefix)?;
            println!(
                "Initialized Reef package `{name}` at {}",
                root.canonicalize().unwrap_or(root).display()
            );
        }
        ReefCommand::Build {
            path,
            no_auto_fetch,
        } => {
            let root = path.unwrap_or_else(|| PathBuf::from("."));
            let options = chelis_reef::BuildOptions {
                auto_fetch: !no_auto_fetch,
            };
            let artifacts = chelis_reef::build_package_with_options(&root, &options)?;
            println!(
                "Built {} {}",
                artifacts.package.name, artifacts.package.version
            );
            println!("Shell: {}", artifacts.shell_path.display());
            println!("Archive: {}", artifacts.archive_path.display());
        }
        ReefCommand::VerifyArtifact {
            archive,
            shell,
            json,
        } => match chelis_reef::verify_artifact_pair(&archive, &shell) {
            Ok(verified) => {
                if json {
                    let report = serde_json::json!({
                        "valid": true,
                        "package": verified.package,
                        "compiler": verified.compiler,
                        "shell_sha256": verified.shell_sha256,
                        "archive_sha256": verified.archive_sha256,
                        "errors": [],
                    });
                    println!("{}", serde_json::to_string_pretty(&report)?);
                } else {
                    println!(
                        "Verified {} {}",
                        verified.package.name, verified.package.version
                    );
                    println!("Shell SHA-256: {}", verified.shell_sha256);
                    println!("Archive SHA-256: {}", verified.archive_sha256);
                }
            }
            Err(error) if json => {
                let report = serde_json::json!({
                    "valid": false,
                    "package": null,
                    "compiler": null,
                    "shell_sha256": null,
                    "archive_sha256": null,
                    "errors": [error],
                });
                println!("{}", serde_json::to_string_pretty(&report)?);
                std::process::exit(1);
            }
            Err(error) => return Err(error.into()),
        },
        ReefCommand::Publish { path } => {
            let root = path.unwrap_or_else(|| PathBuf::from("."));
            let artifacts = chelis_reef::publish_package(&root)?;
            println!(
                "Published {} {}",
                artifacts.package.name, artifacts.package.version
            );
            println!("Shell: {}", artifacts.shell_path.display());
            println!("Archive: {}", artifacts.archive_path.display());
        }
        ReefCommand::Install {
            from_monorepo,
            from_github,
            from_lockfile,
            package_root,
            bootstrap,
            packages,
        } => {
            // clap's `conflicts_with_all` already enforces mutual
            // exclusivity at parse time; the runtime checks below are
            // belt-and-suspenders for any future code path that
            // bypasses clap.
            match (from_monorepo, from_github, from_lockfile, bootstrap) {
                (Some(monorepo_root), None, false, None) => {
                    let mut requested: Vec<(String, Option<String>)> = Vec::new();
                    for spec in packages {
                        let (name, version) = match spec.split_once('=') {
                            Some((n, v)) => (n.to_string(), Some(v.to_string())),
                            None => (spec.clone(), None),
                        };
                        if name.is_empty() {
                            return Err(format!("invalid package selector `{spec}`").into());
                        }
                        requested.push((name, version));
                    }
                    let installed = chelis_reef::install_from_monorepo(&monorepo_root, &requested)?;
                    for artifact in &installed {
                        println!(
                            "Installed {} {}",
                            artifact.package.name, artifact.package.version
                        );
                        println!("Shell: {}", artifact.shell_path.display());
                        println!("Archive: {}", artifact.archive_path.display());
                    }
                    if installed.is_empty() {
                        println!("No packages installed.");
                    }
                }
                (None, Some(spec), false, None) => {
                    if !packages.is_empty() {
                        return Err("`--from-github` does not accept positional package \
                                 selectors; the <ORG>/<REPO>@<TAG> spec is the selector"
                            .into());
                    }
                    let registry_root = chelis_reef::registry_home()?;
                    let artifact = chelis_reef::install_from_github(&spec, &registry_root)?;
                    println!(
                        "Installed {} {}",
                        artifact.package.name, artifact.package.version
                    );
                    println!("Shell: {}", artifact.shell_path.display());
                    println!("Archive: {}", artifact.archive_path.display());
                }
                (None, None, true, None) => {
                    if !packages.is_empty() {
                        return Err("`--from-lockfile` does not accept positional package \
                                 selectors; the lockfile is the selector"
                            .into());
                    }
                    let pkg_root = package_root.unwrap_or_else(|| PathBuf::from("."));
                    run_install_from_lockfile(&pkg_root)?;
                }
                (None, None, false, Some(bootstrap_args)) => {
                    if !packages.is_empty() {
                        return Err("`--bootstrap` does not accept positional package \
                                 selectors; pass each `<ORG>/<REPO>@<TAG>` after `--bootstrap`"
                            .into());
                    }
                    // Empty list = use the built-in default. The list is
                    // hand-maintained for the pre-launch dev team; see
                    // `chelis_reef::DEFAULT_BOOTSTRAP_LIST` rustdoc.
                    let raw_specs: Vec<String> = if bootstrap_args.is_empty() {
                        chelis_reef::DEFAULT_BOOTSTRAP_LIST
                            .iter()
                            .map(|(repo, tag)| {
                                format!("{}/{}@{}", chelis_reef::CANONICAL_REEF_ORG, repo, tag)
                            })
                            .collect()
                    } else {
                        bootstrap_args
                    };
                    let mut parsed: Vec<chelis_reef::GitHubReleaseSpec> =
                        Vec::with_capacity(raw_specs.len());
                    for s in &raw_specs {
                        parsed.push(chelis_reef::GitHubReleaseSpec::parse(s)?);
                    }
                    let registry_root = chelis_reef::registry_home()?;
                    let installed = chelis_reef::install_bootstrap(&parsed, &registry_root)?;
                    for artifact in &installed {
                        println!(
                            "Installed {} {}",
                            artifact.package.name, artifact.package.version
                        );
                        println!("Shell: {}", artifact.shell_path.display());
                        println!("Archive: {}", artifact.archive_path.display());
                    }
                    if installed.is_empty() {
                        println!("No packages installed.");
                    }
                }
                (None, None, false, None) => {
                    return Err("`chelis reef install` requires a source. \
                         Pass `--from-monorepo <PATH>` pointing at a chelis monorepo, \
                         `--from-github <ORG>/<REPO>@<TAG>` to fetch from a GitHub release, \
                         `--from-lockfile` to re-fetch from the project's reef.lock, \
                         or `--bootstrap [<ORG>/<REPO>@<TAG>...]` to install a topo-ordered \
                         set of shells (no args = use the default canonical list)."
                        .into());
                }
                _ => {
                    // Defensive backstop: clap's `conflicts_with_all`
                    // should reject these combinations at parse time.
                    return Err(
                        "`--from-monorepo`, `--from-github`, `--from-lockfile`, and \
                         `--bootstrap` are mutually exclusive"
                            .into(),
                    );
                }
            }
        }
        ReefCommand::ExportBundle { path, output } => {
            let root = path.unwrap_or_else(|| PathBuf::from("."));
            let manifest = chelis_reef::export_bundle(&root, &output)?;
            println!(
                "Exported bundle for {} {} to {}",
                manifest.root_package.name,
                manifest.root_package.version,
                output.display()
            );
            println!("Dependencies: {}", manifest.dependencies.len());
        }
        ReefCommand::Schema { path } => {
            let root = path.unwrap_or_else(|| PathBuf::from("."));
            let schema = chelis_reef::package_schema(&root)?;
            let json = serde_json::to_string_pretty(&schema)
                .map_err(|e| format!("serialize schema: {e}"))?;
            println!("{json}");
        }
        ReefCommand::Src { command } => cmd_reef_src(command)?,
        ReefCommand::Doctor { root } => cmd_reef_doctor(root.as_deref())?,
        ReefCommand::Which { artifact } => {
            // Item 11 (chelis#468): resolve and print the installed binary
            // path, or fail with a clear message if it is not installed.
            let path = chelis_reef::which_artifact(&artifact)?;
            println!("{}", path.display());
        }
        ReefCommand::Setup { path } => cmd_reef_setup(path)?,
        ReefCommand::Conform { command } => cmd_reef_conform(command)?,
    }
    Ok(())
}

/// Dispatch `chelis reef conform <verb>`.
fn cmd_reef_conform(command: ConformCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ConformCommand::Audit {
            root,
            json,
            explain,
        } => {
            let root = match root {
                Some(p) => p,
                None => env::current_dir()?,
            };
            let report = chelis_conformance::audit::audit(&root);
            print_audit_report(&report, json, explain)?;
            if !report.ok() {
                std::process::exit(1);
            }
        }
        ConformCommand::Init {
            name,
            module_prefix,
            output,
        } => {
            let root = output.unwrap_or_else(|| PathBuf::from(&name));
            chelis_conformance::scaffold::scaffold(
                &root,
                &name,
                &module_prefix,
                chelis_compiler_api::COMPILER_VERSION,
            )?;
            println!(
                "scaffolded conformant shell `{name}` at {} (chelis {})",
                root.display(),
                chelis_compiler_api::COMPILER_VERSION
            );
        }
        ConformCommand::Sync { path } => {
            let root = match path {
                Some(p) => p,
                None => env::current_dir()?,
            };
            let version = chelis_conformance::audit::audit(&root)
                .reef_pin
                .map(|p| p.trim_start_matches('=').to_string())
                .unwrap_or_else(|| chelis_compiler_api::COMPILER_VERSION.to_string());
            for notice in chelis_conformance::scaffold::materialize_skills(&root)? {
                eprintln!("note: {notice}");
            }
            chelis_conformance::scaffold::sync_managed_blocks(&root, &version)?;
            println!(
                "synced managed blocks + skills to chelis {version} at {}",
                root.display()
            );
        }
        ConformCommand::Bump { version, path } => {
            let root = match path {
                Some(p) => p,
                None => env::current_dir()?,
            };
            let changed = chelis_conformance::bump::rewrite_pins(&root, &version)?;
            for p in &changed {
                println!("repinned {}", p.display());
            }
            for notice in chelis_conformance::scaffold::materialize_skills(&root)? {
                eprintln!("note: {notice}");
            }
            chelis_conformance::scaffold::sync_managed_blocks(&root, &version)?;
            println!("restamped managed blocks + skills to chelis {version}");

            // Offline gate, categorized (chelis#655). A failure on a row whose
            // artifact the bump itself writes (its pins/managed-block stamps/
            // skills) blocks and exits non-zero: the shell is not in a clean
            // bumped state, whether from a bump-tool defect or a shell edit the
            // bump could only preserve (e.g. a malformed shell-local block).
            // Rows that are author follow-up (CI wiring, pre-existing doc/citation
            // fixes, the prose Pin-Bump-Checklist/Scaffolding-Drift-Rule headings)
            // the bump cannot write are listed as remaining steps, not a failure.
            let report = chelis_conformance::audit::audit(&root);
            let failing: Vec<&chelis_conformance::audit::RowResult> = report
                .rows
                .iter()
                .filter(|r| r.verdict == chelis_conformance::audit::Verdict::Fail && r.tier.gates())
                .collect();
            let bump_owned: Vec<&chelis_conformance::audit::RowResult> = failing
                .iter()
                .copied()
                .filter(|r| chelis_conformance::bump::is_bump_owned(r.key))
                .collect();
            if !bump_owned.is_empty() {
                eprintln!(
                    "bump left a bump-owned artifact non-conformant (pins/managed blocks/skills); \
                     the shell is not in a clean bumped state:"
                );
                for r in &bump_owned {
                    eprintln!("  row {:>2} {} ({})", r.row, r.key, r.section);
                    if !r.diagnostic.is_empty() {
                        eprintln!("        {}", r.diagnostic);
                    }
                }
                std::process::exit(1);
            }
            let follow_up: Vec<String> = failing
                .iter()
                .copied()
                .filter(|r| !chelis_conformance::bump::is_bump_owned(r.key))
                .map(|r| {
                    format!(
                        "  row {:>2} {} ({})  {}",
                        r.row, r.key, r.section, r.diagnostic
                    )
                })
                .collect();

            // Executable gates: the blocked-probe + negative suites, run against
            // the bumped toolchain. A FIX-detected probe is the wanted outcome
            // but still exits non-zero so it is acted on (de-narrow + promote).
            let mut gate_failed = false;
            for (dir, mode) in [("tests_blocked", "blocked"), ("tests_neg", "neg")] {
                if dir_has_ch_files(&root.join(dir)) {
                    let ok = run_self_test_expect(&root, dir, mode)?;
                    if !ok {
                        gate_failed = true;
                    }
                }
            }
            if gate_failed {
                std::process::exit(1);
            }

            // The bump's own output is clean. Report any remaining author
            // follow-up steps, but exit 0 so an automated bump-pr flow sees the
            // mechanical bump succeed (chelis#655).
            println!("bump to chelis {version} is mechanically complete.");
            if !follow_up.is_empty() {
                println!(
                    "{} conformance step(s) remain (author follow-up; see the Pin Bump Checklist):",
                    follow_up.len()
                );
                for line in &follow_up {
                    println!("{line}");
                }
            }
            println!(
                "Remaining manual checklist:\n\
                 - re-probe every docs/UPSTREAM_BUGS.md §Tracking entry naming this release, per-verb\n\
                 Open a PR with this change set; do not push the pin directly to main."
            );
        }
        ConformCommand::BumpCheck { base, path } => {
            let root = match path {
                Some(p) => p,
                None => env::current_dir()?,
            };
            // Current pin from the working tree. A missing or unpinned local
            // reef.toml is a hard error (exit 2): the guard cannot vouch for a
            // shell it cannot read, and this is *not* a base-resolution problem.
            let current = match std::fs::read_to_string(root.join("reef.toml")) {
                Ok(text) => chelis_conformance::bump::bare_pin(&text),
                Err(e) => {
                    eprintln!("bump-check: cannot read {}/reef.toml: {e}", root.display());
                    std::process::exit(2);
                }
            };
            let Some(current) = current else {
                eprintln!(
                    "bump-check: {}/reef.toml has no usable `compiler = \"=X.Y.Z\"` pin",
                    root.display()
                );
                std::process::exit(2);
            };

            match git_show_reef_pin(&root, &base) {
                Some(base_pin) if base_pin == current => {
                    println!("reef pin unchanged vs {base} (={current}); bump-check passes");
                }
                Some(base_pin) => {
                    // The pin changed: require a green audit (fresh stamps,
                    // lockstep workflow pins, wired probes).
                    let report = chelis_conformance::audit::audit(&root);
                    if report.ok() {
                        println!("pin change {base_pin} -> {current}: conformance audit green");
                    } else {
                        eprintln!(
                            "pin changed {base_pin} -> {current} but the checklist did not run \
                             (audit not green). Bump via `chelis reef conform bump {current}`, not a raw edit:"
                        );
                        print_audit_report(&report, false, false)?;
                        std::process::exit(1);
                    }
                }
                None if git_ref_resolvable(&root, &base) => {
                    // The base commit exists but carries no readable pin: this
                    // change is *introducing* the pin, so there is nothing to
                    // cascade. Not a fail-open — the base was genuinely resolved.
                    println!(
                        "bump-check: no reef pin at {base} (introducing the pin); nothing to cascade"
                    );
                }
                None => {
                    // The base ref itself could not be resolved (bad ref, or a
                    // shallow clone that never fetched it). FAIL CLOSED: the
                    // guard exists to block a raw pin cascade, so it must never
                    // silently pass when it cannot see the base.
                    eprintln!(
                        "bump-check: could not resolve `{base}` (bad ref, or a shallow clone that \
                         did not fetch it). The guard fails closed rather than skip. Fetch the base \
                         (actions/checkout with fetch-depth: 0), or pass a resolvable --base."
                    );
                    std::process::exit(2);
                }
            }
        }
    }
    Ok(())
}

/// Whether `dir` contains any `.ch` file (recursively).
fn dir_has_ch_files(dir: &Path) -> bool {
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&d) {
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|s| s.to_str()) == Some("ch") {
                    return true;
                }
            }
        }
    }
    false
}

/// Re-exec this binary as `chelis test <dir>/ --expect <mode>` in `root`,
/// inheriting stdio. Returns whether it passed (exit 0).
fn run_self_test_expect(
    root: &Path,
    dir: &str,
    mode: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let exe = std::env::current_exe()?;
    let status = std::process::Command::new(exe)
        .current_dir(root)
        .args(["test", &format!("{dir}/"), "--expect", mode])
        .status()?;
    Ok(status.success())
}

/// Read the `compiler` pin from `reef.toml` at git ref `base` (bare `X.Y.Z`),
/// or `None` if git or the file is unavailable at that ref. The pathspec is
/// cwd-relative (`:./reef.toml`) so a shell nested inside a larger repo reads
/// its *own* reef.toml at the base, matching the working-tree read.
fn git_show_reef_pin(root: &Path, base: &str) -> Option<String> {
    let out = std::process::Command::new("git")
        .current_dir(root)
        .args(["show", &format!("{base}:./reef.toml")])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    chelis_conformance::bump::bare_pin(&text)
}

/// Whether `base` resolves to a commit in the repo at `root`. Used to tell a
/// base that genuinely predates the pin (resolvable ref, no `reef.toml`) from a
/// base that could not be fetched at all (the shallow-clone / bad-ref case that
/// must fail the guard closed).
fn git_ref_resolvable(root: &Path, base: &str) -> bool {
    std::process::Command::new("git")
        .current_dir(root)
        .args([
            "rev-parse",
            "--verify",
            "--quiet",
            &format!("{base}^{{commit}}"),
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Print a conformance audit report as plain text or NDJSON.
fn print_audit_report(
    report: &chelis_conformance::audit::AuditReport,
    json: bool,
    explain: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let stdout = io::stdout();
    let mut out = stdout.lock();
    if json {
        for r in &report.rows {
            let record = serde_json::json!({
                "row": r.row,
                "key": r.key,
                "section": r.section,
                "tier": r.tier.tag(),
                // Whether a Fail on this row gates the audit — lets a machine
                // consumer tell a gating MUST-fail from an advisory SHOULD-fail
                // without re-deriving tier semantics.
                "gating": r.tier.gates(),
                "verdict": r.verdict.tag(),
                "diagnostic": r.diagnostic,
                "fix": r.fix,
                // Per-finding site/coverage evidence (chelis#654); empty for rows
                // that carry no site-level detail. Always emitted in JSON so a
                // machine consumer never needs the `--explain` text mode.
                "evidence": r.evidence,
            });
            writeln!(out, "{record}")?;
        }
        let summary = serde_json::json!({
            "summary": {
                "must_failures": report.must_failures(),
                "ok": report.ok(),
                "pin": report.reef_pin,
            }
        });
        writeln!(out, "{summary}")?;
    } else {
        for r in &report.rows {
            let mark = r.verdict.tag().to_uppercase();
            writeln!(
                out,
                "row {:>2} {:<10} {} ({})",
                r.row, mark, r.key, r.section
            )?;
            if !r.diagnostic.is_empty() {
                writeln!(out, "        {}", r.diagnostic)?;
                if !r.fix.is_empty() {
                    writeln!(out, "        fix: {}", r.fix)?;
                }
                if explain {
                    for line in &r.evidence {
                        writeln!(out, "        {line}")?;
                    }
                }
            }
        }
        let n = report.must_failures();
        if n == 0 {
            writeln!(out, "\nconformant: no MUST failures")?;
        } else {
            writeln!(out, "\n{n} MUST failure(s)")?;
        }
    }
    Ok(())
}

/// Re-install every dependency named by a package's `reef.lock`, fetching
/// each from its recorded `remote_origin` and verifying hashes against the
/// pins. Prints one line per [`chelis_reef::LockfileInstallEntry`] and
/// returns an aggregated error if any entry failed or lacked an origin.
///
/// Extracted from the `reef install --from-lockfile` arm so `reef setup`
/// (WS-C) reuses the exact same reporting.
fn run_install_from_lockfile(pkg_root: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let registry_root = chelis_reef::registry_home()?;
    let results = chelis_reef::install_from_lockfile(pkg_root, &registry_root)?;
    let mut any_failure = false;
    let mut any_no_origin = false;
    for entry in &results {
        match entry {
            chelis_reef::LockfileInstallEntry::Installed(artifact) => {
                println!(
                    "Installed {} {}",
                    artifact.package.name, artifact.package.version
                );
                println!("Shell: {}", artifact.shell_path.display());
                println!("Archive: {}", artifact.archive_path.display());
            }
            chelis_reef::LockfileInstallEntry::SkippedPathDep {
                name,
                version,
                path,
            } => {
                println!(
                    "Skipped path dep {name} {version} (path = {path}): \
                     resolved at build time, not via remote fetch"
                );
            }
            chelis_reef::LockfileInstallEntry::SkippedBundledRuntime {
                name,
                version,
                compiler_version,
            } => {
                println!(
                    "Skipped bundled runtime {name} {version} \
                     (compiler version {compiler_version}): \
                     ships with the compiler, not fetched"
                );
            }
            chelis_reef::LockfileInstallEntry::SkippedNoOrigin { name, version } => {
                eprintln!(
                    "error: lockfile entry `{name}` v{version} has no \
                     `remote_origin` recorded; cannot fetch. \
                     Run `chelis reef install --bootstrap` (or re-run \
                     `--from-github`) to populate the origin."
                );
                any_no_origin = true;
            }
            chelis_reef::LockfileInstallEntry::InstalledBinary {
                name,
                version,
                path,
            } => {
                println!("Installed binary {name} {version}");
                println!("Binary: {}", path.display());
            }
            chelis_reef::LockfileInstallEntry::SkippedForeignPlatform {
                name,
                version,
                platform,
                host,
            } => {
                let host_label = match host {
                    Some(h) => h.clone(),
                    None => "unsupported".to_string(),
                };
                println!(
                    "Skipped binary {name} {version} (built for {platform}; \
                     host is {host_label}): not runnable on this host"
                );
            }
            chelis_reef::LockfileInstallEntry::Failed { error, .. } => {
                eprintln!("error: {error}");
                any_failure = true;
            }
        }
    }
    if any_failure || any_no_origin {
        let detail = if any_no_origin && any_failure {
            "one or more lockfile entries failed to install and one or more \
             have no remote_origin"
        } else if any_no_origin {
            "one or more lockfile entries have no remote_origin"
        } else {
            "one or more lockfile entries failed to install"
        };
        return Err(detail.into());
    }
    if results.is_empty() {
        println!("No dependencies in lockfile.");
    }
    Ok(())
}

/// Normalize a manifest `compiler = "=X.Y.Z"` pin to the bare `X.Y.Z`.
fn bare_compiler_pin(pin: &str) -> String {
    pin.trim_start_matches('=').to_string()
}

/// WS-C (§7): bring a freshly-cloned shell to its pins in one verb. Reads
/// the `reef.toml` compiler pin, then in order: ensures the pinned
/// toolchain (auto-installing via chelisup), installs source packages +
/// binary artifacts from `reef.lock` (WS-A), syncs source crates when
/// `[chelis-src]` is present (chelis#571), and prints the `reef doctor`
/// health summary. This is the "clone -> one command -> build" entry point.
fn cmd_reef_setup(path: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let root = path
        .unwrap_or_else(|| PathBuf::from("."))
        .canonicalize()
        .map_err(|e| format!("cannot resolve setup root: {e}"))?;
    // Parse-only manifest read (like `reef src`): deliberately skips the
    // same-compiler-version gate so a shell pinned to a *different* chelis
    // still sets up.
    let manifest = chelis_reef::read_manifest_for_src(&root)?;
    let version = bare_compiler_pin(&manifest.package.compiler);
    println!(
        "chelis reef setup: {} (pin {version})",
        manifest.package.name
    );

    // Step 1: ensure the pinned toolchain. Fail-fast: a missing or wrong
    // compiler blocks the downstream build, so there is no point continuing.
    ensure_pinned_toolchain(&version)?;

    // Step 2: source packages + binary artifacts from reef.lock (WS-A).
    if root.join("reef.lock").is_file() {
        run_install_from_lockfile(&root)?;
    } else {
        println!("  install:   no reef.lock; skipping source-package/binary install");
    }

    // Step 3: source crates (chelis#571), only for crate-linking shells.
    // The `is_some` guard avoids the "no [chelis-src]" error path in
    // `resolve_shell_src_context`.
    if manifest.chelis_src.is_some() {
        cmd_reef_src(ReefSrcCommand::Sync {
            path: Some(root.clone()),
        })?;
    } else {
        println!("  src:       no [chelis-src]; skipping source-crate sync");
    }

    // Step 4: read-only health summary across every dependency class.
    println!("--- doctor ---");
    cmd_reef_doctor(Some(&root))?;
    Ok(())
}

/// Step 1 of `reef setup`: ensure the `version` toolchain is installed in
/// the chelisup store, auto-installing it by delegating to the real
/// `chelisup` binary when it is missing.
///
/// IMPORTANT (shim-corruption guard): this MUST subprocess the installed
/// `chelisup` binary and MUST NOT call `chelisup::install::install(...)`
/// in-process. That library helper copies `current_exe()` into
/// `<home>/bin/{chelis,chelisup}`; called from *this* (the `chelis`
/// compiler) binary it would overwrite the shim and the installer with the
/// compiler. The subprocess runs the real chelisup, whose `current_exe()`
/// is chelisup itself. Do not "simplify" this into a library call.
///
/// This is enforced at compile time: chelisup's `install` and
/// `ensure_shim_installed` are `pub(crate)`, so referencing them from here is
/// an `E0603` build error caught by the normal clippy/build/test stages. The
/// guard would only regress if someone widened that visibility *and* added
/// the call.
fn ensure_pinned_toolchain(version: &str) -> Result<(), Box<dyn std::error::Error>> {
    let store = chelisup::paths::Store::from_env()?;
    if store.is_installed(version) {
        println!(
            "  toolchain: ok ({} installed)",
            store.toolchain_dir(version).display()
        );
        return Ok(());
    }
    let chelisup = chelisup_binary(&store);
    println!("  toolchain: {version} not installed; running `chelisup install {version}`...");
    let status = match std::process::Command::new(&chelisup)
        .args(["install", version])
        .status()
    {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!(
                "the pinned toolchain {version} is not installed and `chelisup` was not \
                 found (looked at {} and on PATH). Bootstrap chelisup \
                 (`curl -fsSL <host>/chelisup.sh | sh`) so `chelisup install {version}` \
                 can run, then re-run `chelis reef setup`.",
                store.chelisup_path().display()
            )
            .into());
        }
        Err(e) => return Err(format!("could not run {}: {e}", chelisup.display()).into()),
    };
    if !status.success() {
        let code = status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "signal".to_string());
        return Err(format!(
            "`chelisup install {version}` failed (exit {code}). \
             Install the toolchain manually and re-run `chelis reef setup`."
        )
        .into());
    }
    // Post-condition: a successful chelisup install must have placed it.
    if !store.is_installed(version) {
        return Err(format!(
            "`chelisup install {version}` reported success but {} is still absent",
            store.toolchain_dir(version).display()
        )
        .into());
    }
    println!(
        "  toolchain: ok ({} installed)",
        store.toolchain_dir(version).display()
    );
    Ok(())
}

/// Resolve the `chelisup` binary `reef setup` delegates installs to:
/// `$CHELISUP_BIN` (the test seam) if it names a non-empty path, else the
/// installed copy at `<home>/bin/chelisup`, else bare `chelisup` on PATH
/// (the spawn surfaces a clear NotFound if it is absent).
fn chelisup_binary(store: &chelisup::paths::Store) -> PathBuf {
    if let Some(bin) = std::env::var_os("CHELISUP_BIN")
        && !bin.is_empty()
    {
        return PathBuf::from(bin);
    }
    let stored = store.chelisup_path();
    if stored.exists() {
        return stored;
    }
    PathBuf::from("chelisup")
}

/// Canonical remote for the source store, overridable by `CHELIS_SRC_REMOTE`
/// (a test-injection seam paralleling `CHELIS_REEF_GITHUB_BASE_API`; also
/// usable to point at a mirror). Production default is the canonical repo.
fn chelis_src_remote() -> String {
    std::env::var("CHELIS_SRC_REMOTE")
        .unwrap_or_else(|_| chelis_reef::chelis_src::CANONICAL_REMOTE.to_string())
}

/// The resolved source-crate facts for one shell: where its `../chelis`
/// slot is, the bare semver of its compiler pin, its `[chelis-src]` spec,
/// the store root, and the store worktree path for its version.
struct ShellSrcContext {
    root: PathBuf,
    slot: PathBuf,
    version: String,
    spec: chelis_reef::ChelisSrcSpec,
    store_root: PathBuf,
    worktree: PathBuf,
}

/// Resolve a shell root (defaulting to `.`) into its source-crate context,
/// or a clear error if it carries no `[chelis-src]` section.
fn resolve_shell_src_context(
    path: Option<PathBuf>,
) -> Result<ShellSrcContext, Box<dyn std::error::Error>> {
    let root = path
        .unwrap_or_else(|| PathBuf::from("."))
        .canonicalize()
        .map_err(|e| format!("cannot resolve shell root: {e}"))?;
    let manifest = chelis_reef::read_manifest_for_src(&root)?;
    let spec = manifest.chelis_src.ok_or_else(|| {
        format!(
            "{} has no [chelis-src] section: `reef src` applies only to shells that link \
             chelis source crates (chelis-ir/chelis-types) as Cargo path deps",
            root.join("reef.toml").display()
        )
    })?;
    let version = bare_compiler_pin(&manifest.package.compiler);
    // The committed `path = "../chelis/..."` resolves relative to the
    // package root, so the slot is `<root>/../chelis` = `<parent>/chelis`.
    let slot = root
        .parent()
        .ok_or("shell root has no parent directory for the ../chelis slot")?
        .join("chelis");
    let store_root = chelis_reef::chelis_src::default_store_root()?;
    let worktree = store_root.join(&version);
    Ok(ShellSrcContext {
        root,
        slot,
        version,
        spec,
        store_root,
        worktree,
    })
}

fn cmd_reef_src(command: ReefSrcCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ReefSrcCommand::Sync { path } => {
            let ctx = resolve_shell_src_context(path)?;
            let token = chelis_reef::try_github_token();
            let outcome = chelis_reef::chelis_src::sync(
                &ctx.store_root,
                &chelis_src_remote(),
                token.as_deref(),
                &ctx.version,
                ctx.spec.pin_commit.as_deref(),
            )?;
            let wire = chelis_reef::chelis_src::wire_slot(&ctx.slot, &outcome.worktree)?;
            println!("synced chelis source crates for pin {}", ctx.version);
            println!("  commit:   {}", outcome.commit);
            println!("  store:    {}", outcome.worktree.display());
            match wire {
                chelis_reef::chelis_src::WireOutcome::Created => {
                    println!("  ../chelis: linked -> {}", outcome.worktree.display());
                }
                chelis_reef::chelis_src::WireOutcome::AlreadyWired => {
                    println!(
                        "  ../chelis: already linked -> {}",
                        outcome.worktree.display()
                    );
                }
                chelis_reef::chelis_src::WireOutcome::Repointed { previous } => {
                    println!(
                        "  ../chelis: repointed {} -> {}",
                        previous.display(),
                        outcome.worktree.display()
                    );
                }
            }
            println!("next: run `cargo build`; chelis crates resolve at the pin, no lock churn.");
            Ok(())
        }
        ReefSrcCommand::Check { path } => {
            let ctx = resolve_shell_src_context(path)?;
            let problems = collect_src_drift(&ctx);
            if problems.is_empty() {
                println!(
                    "ok: chelis source crates pinned at {} (commit {})",
                    ctx.version,
                    expected_commit(&ctx).unwrap_or_else(|| "unresolved".to_string())
                );
                Ok(())
            } else {
                for p in &problems {
                    eprintln!("drift: {p}");
                }
                eprintln!("fix: run `chelis reef src sync` in {}", ctx.root.display());
                Err(format!("{} source-crate drift issue(s)", problems.len()).into())
            }
        }
        ReefSrcCommand::Status { path } => {
            let ctx = resolve_shell_src_context(path)?;
            println!("shell:     {}", ctx.root.display());
            println!("pin:       {}", ctx.version);
            println!(
                "pin_commit: {}",
                ctx.spec
                    .pin_commit
                    .as_deref()
                    .unwrap_or("(from v<version> tag)")
            );
            if !ctx.spec.crates.is_empty() {
                println!("crates:    {}", ctx.spec.crates.join(", "));
            }
            match chelis_reef::chelis_src::worktree_head(&ctx.store_root, &ctx.version)? {
                Some(head) => println!("store:     {} @ {head}", ctx.worktree.display()),
                None => println!("store:     (not synced; run `chelis reef src sync`)"),
            }
            match std::fs::read_link(&ctx.slot) {
                Ok(target) => println!("../chelis: {} -> {}", ctx.slot.display(), target.display()),
                Err(_) => println!("../chelis: {} (not a symlink)", ctx.slot.display()),
            }
            Ok(())
        }
    }
}

/// The commit the shell expects: its explicit `pin_commit`, else the commit
/// `v<version>` resolves to in the local mirror (offline; `None` if neither
/// is available).
fn expected_commit(ctx: &ShellSrcContext) -> Option<String> {
    if let Some(c) = &ctx.spec.pin_commit {
        return Some(c.clone());
    }
    let mirror = ctx.store_root.join("mirror.git");
    chelis_reef::chelis_src::resolve_commit(&mirror, &ctx.version, None).ok()
}

/// Collect every way this shell's source crates have drifted off the pin:
/// store worktree missing/wrong commit, `../chelis` slot not the expected
/// symlink, or `Cargo.lock` recording a non-pin crate version. Empty ⇒ clean.
fn collect_src_drift(ctx: &ShellSrcContext) -> Vec<String> {
    let mut problems = Vec::new();
    let expected = expected_commit(ctx);

    match chelis_reef::chelis_src::worktree_head(&ctx.store_root, &ctx.version) {
        Ok(Some(head)) => {
            if let Some(exp) = &expected
                && &head != exp
            {
                problems.push(format!(
                    "store worktree {} is at {head}, expected {exp}",
                    ctx.worktree.display()
                ));
            }
        }
        Ok(None) => problems.push(format!(
            "no source store worktree for {} ({})",
            ctx.version,
            ctx.worktree.display()
        )),
        Err(e) => problems.push(format!("reading store worktree: {e}")),
    }

    if let Err(e) = chelis_reef::chelis_src::check_slot(&ctx.slot, &ctx.worktree) {
        problems.push(format!("{e}"));
    }

    if let Err(e) = check_cargo_lock_at_pin(&ctx.root, &ctx.spec.crates, &ctx.version) {
        problems.push(e);
    }

    problems
}

/// Verify that every declared chelis source crate appears in `<root>/
/// Cargo.lock` at the pinned version. This catches the `.cargo`-override
/// failure mode (build correct, lockfile churned off the pin) and a stale
/// sibling. No `Cargo.lock` or no declared crates ⇒ nothing to check.
fn check_cargo_lock_at_pin(root: &Path, crates: &[String], version: &str) -> Result<(), String> {
    if crates.is_empty() {
        return Ok(());
    }
    let lock_path = root.join("Cargo.lock");
    let text = match std::fs::read_to_string(&lock_path) {
        Ok(t) => t,
        Err(_) => return Ok(()), // no lockfile yet — nothing to assert
    };
    let mismatched = chelis_reef::chelis_src::cargo_lock_mismatches(&text, crates, version)
        .map_err(|e| format!("{}: {e}", lock_path.display()))?;
    if mismatched.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Cargo.lock records non-pin chelis crate versions (expected {version}): {} \
             (the ../chelis slot is off the pin)",
            mismatched.join(", ")
        ))
    }
}

fn cmd_reef_doctor(root: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    let scan_root = root
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .canonicalize()
        .map_err(|e| format!("cannot resolve --root: {e}"))?;

    // The scan root itself, then each immediate subdirectory, that carries a
    // reef.toml with a `compiler =` pin.
    let mut shells: Vec<PathBuf> = Vec::new();
    if scan_root.join("reef.toml").is_file() {
        shells.push(scan_root.clone());
    }
    if let Ok(entries) = std::fs::read_dir(&scan_root) {
        let mut subs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir() && p.join("reef.toml").is_file())
            .collect();
        subs.sort();
        shells.extend(subs);
    }

    if shells.is_empty() {
        println!(
            "no shell repos (reef.toml) found under {}",
            scan_root.display()
        );
        return Ok(());
    }

    // The consolidated chelisup store (`$CHELIS_HOME` else `~/.chelis`),
    // which owns toolchains (`toolchains/<ver>`), the shim, and the recorded
    // default. Kept as a `Result` so an unresolved home degrades the
    // per-shell toolchain line to "unknown" rather than aborting the scan.
    let store = chelisup::paths::Store::from_env();
    match &store {
        Ok(s) => {
            println!("chelis home: {}", s.home().display());
            let shim = s.shim_path();
            println!(
                "shim:        {} ({})",
                shim.display(),
                if shim.exists() { "present" } else { "absent" }
            );
            println!(
                "default:     {}",
                s.read_default().unwrap_or_else(|| "(unset)".to_string())
            );
        }
        Err(e) => println!("chelis home: unresolved ({e})"),
    }
    println!();

    for shell in &shells {
        let manifest = match chelis_reef::read_manifest_for_src(shell) {
            Ok(m) => m,
            Err(e) => {
                println!("{}: unreadable reef.toml ({e})", shell.display());
                continue;
            }
        };
        let version = bare_compiler_pin(&manifest.package.compiler);
        println!("{} (pin {})", manifest.package.name, version);

        // Class (a): is the pinned toolchain installed in the chelisup store?
        match &store {
            Ok(s) if s.is_installed(&version) => {
                println!(
                    "  toolchain: ok ({} installed)",
                    s.toolchain_dir(&version).display()
                );
            }
            Ok(s) => println!(
                "  toolchain: MISSING; run `chelisup install {version}` ({} absent)",
                s.toolchain_dir(&version).display()
            ),
            Err(_) => println!("  toolchain: unknown (chelis home unresolved)"),
        }

        // Class (c): source crates (only for crate-linking shells).
        match resolve_shell_src_context(Some(shell.clone())) {
            Ok(ctx) => {
                let problems = collect_src_drift(&ctx);
                if problems.is_empty() {
                    println!("  src:       ok (../chelis pinned at {})", ctx.version);
                } else {
                    for p in &problems {
                        println!("  src:       DRIFT: {p}");
                    }
                    println!(
                        "  src:       fix: `chelis reef src sync` in {}",
                        ctx.root.display()
                    );
                }
            }
            Err(_) => {
                println!("  src:       n/a (no [chelis-src]; pure-Chelis shell)");
            }
        }

        // Class (binary artifacts, Item 11 / WS-A): declared in `[artifacts]`
        // and installed via `chelis reef install --from-lockfile` to
        // `<chelis home>/bin/<name>`.
        if manifest.artifacts.is_empty() {
            println!("  artifacts: n/a (no [artifacts])");
        } else {
            for name in manifest.artifacts.keys() {
                match chelis_reef::which_artifact(name) {
                    Ok(path) => println!("  artifacts: ok ({name} -> {})", path.display()),
                    Err(_) => println!(
                        "  artifacts: MISSING: {name} \
                         (fix: `chelis reef install --from-lockfile`)"
                    ),
                }
            }
        }
    }
    Ok(())
}

/// RAII guard that owns the bincode-encoded `CompiledContext` tempfile
/// the parent passes to each `chelis test` worker via the
/// `CHELIS_TEST_COMPILED_CONTEXT` env var.
///
/// Drop deletes the file (best-effort): on success the parent has already
/// drained every worker by the time the guard goes out of scope; on
/// failure (panic, error return, signal) the OS keeps the tempfile around
/// no longer than the parent process. We use `tempfile::NamedTempFile`
/// internally because (a) it places the file under the system temp dir
/// (`$TMPDIR`/`/tmp`) where every reef worker can read it, (b) the path
/// is unique per parent so two `chelis test` invocations cannot clobber
/// each other, and (c) the Drop impl removes the file even on panic.
struct CompiledContextTempfile {
    #[allow(dead_code)]
    file: tempfile::NamedTempFile,
}

impl CompiledContextTempfile {
    #[allow(dead_code)]
    fn write(bytes: &[u8]) -> Result<Self, String> {
        // Build the tempfile with a recognizable prefix so a stray copy is
        // easy to attribute back to `chelis test` if it ever leaks.
        // RT-H: integration tests can set `CHELIS_TEST_COMPILED_CONTEXT_TMPDIR`
        // to a private directory so cleanup probes don't race against
        // concurrent test binaries' tempfiles in /tmp.
        let mut builder = tempfile::Builder::new();
        builder.prefix("chelis-compiled-context-").suffix(".bin");
        let mut file = match env::var_os("CHELIS_TEST_COMPILED_CONTEXT_TMPDIR") {
            Some(dir) => builder.tempfile_in(dir),
            None => builder.tempfile(),
        }
        .map_err(|e| format!("create compiled-context tempfile: {e}"))?;
        std::io::Write::write_all(file.as_file_mut(), bytes)
            .map_err(|e| format!("write compiled-context tempfile: {e}"))?;
        // Flush before any worker can `open(2)` the file — otherwise a
        // worker might race the parent and read a truncated snapshot.
        std::io::Write::flush(file.as_file_mut())
            .map_err(|e| format!("flush compiled-context tempfile: {e}"))?;
        file.as_file()
            .sync_all()
            .map_err(|e| format!("sync compiled-context tempfile: {e}"))?;
        Ok(Self { file })
    }

    #[allow(dead_code)]
    fn path(&self) -> &Path {
        self.file.path()
    }
}

// `tempfile::NamedTempFile` already removes the underlying file on Drop;
// no explicit impl needed. The wrapper exists so the parent has a single
// owner and so the tempfile path can be used by every spawned worker
// without leaking the file handle into worker subprocesses (workers
// reopen the path themselves).

#[derive(Debug, Clone, Copy)]
enum TestJobs {
    Auto,
    Count(usize),
}

impl TestJobs {
    fn resolve(self, test_file_count: usize) -> usize {
        let requested = match self {
            TestJobs::Auto => std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            TestJobs::Count(count) => count,
        };
        requested.max(1).min(test_file_count.max(1))
    }
}

impl FromStr for TestJobs {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(TestJobs::Auto);
        }
        let count = value
            .parse::<usize>()
            .map_err(|_| "`--jobs` must be `auto` or a positive integer".to_string())?;
        if count == 0 {
            return Err("`--jobs` must be greater than zero".to_string());
        }
        Ok(TestJobs::Count(count))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestBatchMode {
    Auto,
    File,
}

impl FromStr for TestBatchMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("auto") {
            return Ok(TestBatchMode::Auto);
        }
        if value.eq_ignore_ascii_case("file") {
            return Ok(TestBatchMode::File);
        }
        Err("`--batch-mode` must be `auto` or `file`".to_string())
    }
}

/// `--expect <neg|blocked>`: which expected-failure suite semantics to apply.
/// Thin CLI mirror of [`chelis_conformance::expect::ExpectMode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpectArg {
    Neg,
    Blocked,
}

impl FromStr for ExpectArg {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("neg") {
            return Ok(ExpectArg::Neg);
        }
        if value.eq_ignore_ascii_case("blocked") {
            return Ok(ExpectArg::Blocked);
        }
        Err("`--expect` must be `neg` or `blocked`".to_string())
    }
}

impl ExpectArg {
    fn mode(self) -> chelis_conformance::expect::ExpectMode {
        match self {
            ExpectArg::Neg => chelis_conformance::expect::ExpectMode::Neg,
            ExpectArg::Blocked => chelis_conformance::expect::ExpectMode::Blocked,
        }
    }

    fn cli_value(self) -> &'static str {
        match self {
            ExpectArg::Neg => "neg",
            ExpectArg::Blocked => "blocked",
        }
    }
}

fn testing_hook_enabled(name: &str) -> bool {
    env::var("CHELIS_TEST_INTERNAL_TESTING").as_deref() == Ok("1")
        && env::var(name).as_deref() == Ok("1")
}

fn hang_test_suite_if_requested(name: &str) {
    if testing_hook_enabled(name) {
        loop {
            thread::park();
        }
    }
}

fn emit_finalized_test_suite_if_requested(
    json: bool,
    expect: Option<ExpectArg>,
) -> Option<Result<i32, String>> {
    if !testing_hook_enabled("CHELIS_TEST_EMIT_FINALIZED_SUITE") {
        return None;
    }

    // Finalization-timeout tests need a child that has conclusively emitted
    // rows and a summary before it hangs. Keep that lifecycle oracle
    // independent of Reef compilation and worker spawning: under a saturated
    // workspace run, those unrelated prerequisites can exhaust process
    // resources and return runner-error 2 before the finalization hook.
    let result = (|| {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        if let Some(expect) = expect {
            if json {
                writeln!(
                    out,
                    "{}",
                    serde_json::json!({
                        "file": "tests/smoke.ch",
                        "expect": expect.cli_value(),
                        "verdict": "config-error",
                        "detail": "missing .expect sidecar",
                    })
                )
                .map_err(|e| e.to_string())?;
                writeln!(
                    out,
                    "{}",
                    serde_json::json!({
                        "summary": {
                            "ok": 0,
                            "failed": 1,
                            "mode": expect.cli_value(),
                        }
                    })
                )
                .map_err(|e| e.to_string())?;
            } else {
                writeln!(out, "CONFIG-ERROR       tests/smoke.ch").map_err(|e| e.to_string())?;
                writeln!(out, "    missing .expect sidecar").map_err(|e| e.to_string())?;
                writeln!(out, "\n0 ok, 1 failing ({} mode)", expect.cli_value())
                    .map_err(|e| e.to_string())?;
            }
            out.flush().map_err(|e| e.to_string())?;
            return Ok(1);
        }

        if json {
            writeln!(
                out,
                "{}",
                serde_json::json!({
                    "file": "tests/smoke.ch",
                    "test": "test_ok",
                    "status": "pass",
                })
            )
            .map_err(|e| e.to_string())?;
            writeln!(
                out,
                "{}",
                serde_json::json!({ "summary": { "passed": 1, "failed": 0 } })
            )
            .map_err(|e| e.to_string())?;
        } else {
            writeln!(out, "tests/smoke.ch").map_err(|e| e.to_string())?;
            writeln!(out, "  test_ok ....................... PASS").map_err(|e| e.to_string())?;
            writeln!(out, "\n1 passed, 0 failed").map_err(|e| e.to_string())?;
        }
        out.flush().map_err(|e| e.to_string())?;
        Ok(0)
    })();
    Some(result)
}

fn write_test_progress_rows_if_requested(path: &Path) {
    if env::var("CHELIS_TEST_INTERNAL_TESTING").as_deref() != Ok("1") {
        return;
    }
    let Ok(count) = env::var("CHELIS_TEST_PROGRESS_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .ok_or(())
    else {
        return;
    };
    let Ok(file) = fs::OpenOptions::new().append(true).open(path) else {
        return;
    };
    let mut out = io::BufWriter::new(file);
    for index in 0..count {
        if writeln!(
            out,
            "{{\"file\":\"tests/backpressure.ch\",\"test\":\"test_{index}\",\"status\":\"pass\"}}"
        )
        .is_err()
        {
            return;
        }
    }
    let _ = out.flush();
}

#[cfg(unix)]
fn create_cloexec_pipe() -> Result<(std::os::fd::OwnedFd, std::os::fd::OwnedFd), String> {
    use std::os::fd::FromRawFd;

    let mut fds = [-1; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(format!(
            "create test supervisor pipe: {}",
            std::io::Error::last_os_error()
        ));
    }
    for fd in fds {
        let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
        if flags < 0 || unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) } < 0 {
            unsafe {
                libc::close(fds[0]);
                libc::close(fds[1]);
            }
            return Err(format!(
                "protect test supervisor pipe: {}",
                std::io::Error::last_os_error()
            ));
        }
    }
    Ok(unsafe {
        (
            std::os::fd::OwnedFd::from_raw_fd(fds[0]),
            std::os::fd::OwnedFd::from_raw_fd(fds[1]),
        )
    })
}

#[cfg(unix)]
fn start_forked_suite_parent_watchdog(pipe: std::os::fd::OwnedFd, progress_path: PathBuf) {
    let mut pipe = std::fs::File::from(pipe);
    thread::spawn(move || {
        let mut byte = [0u8; 1];
        loop {
            match pipe.read(&mut byte) {
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                _ => {
                    let _ = fs::remove_file(&progress_path);
                    unsafe {
                        libc::kill(0, libc::SIGKILL);
                    }
                    break;
                }
            }
        }
    });
}

#[cfg(unix)]
fn ignore_test_suite_sigterm_if_requested() {
    if testing_hook_enabled("CHELIS_TEST_IGNORE_SIGTERM") {
        unsafe {
            libc::signal(libc::SIGTERM, libc::SIG_IGN);
        }
    }
}

#[cfg(not(unix))]
fn ignore_test_suite_sigterm_if_requested() {}

#[allow(clippy::too_many_arguments)]
fn cmd_test_supervised(
    path: Option<&Path>,
    filter: Option<&str>,
    json: bool,
    timeout_secs: u64,
    suite_timeout_secs: u64,
    jobs: TestJobs,
    batch_mode: TestBatchMode,
    expect: Option<ExpectArg>,
) -> Result<i32, String> {
    ensure_test_suite_supervision_supported()?;
    if suite_timeout_secs == 0 {
        return Err("`--suite-timeout` must be at least 1 second".to_string());
    }
    let progress_file = tempfile::Builder::new()
        .prefix("chelis-test-progress-")
        .suffix(".ndjson")
        .tempfile()
        .map_err(|e| format!("could not create suite progress file: {e}"))?;
    let timeout = Duration::from_secs(suite_timeout_secs);
    let suite_deadline = Instant::now().checked_add(timeout);
    let output = run_forked_test_suite(
        path,
        filter,
        json,
        timeout_secs,
        jobs,
        batch_mode,
        expect,
        progress_file.path(),
        timeout,
    )
    .map_err(|e| format!("could not supervise test suite: {e}"))?;
    let batch_progress = read_test_batch_progress(progress_file.path());
    if output.timed_out || output.leader_signaled {
        let reason = if output.timed_out {
            SuiteIncompleteReason::Timeout(suite_timeout_secs)
        } else {
            SuiteIncompleteReason::AbnormalLeaderExit
        };
        return render_incomplete_test_suite(&output.output, reason, json, expect, batch_progress);
    }

    let exit_code = output.output.status.code().unwrap_or(2);
    let output_forwarded = match suite_deadline {
        Some(deadline) => {
            let stderr_forwarded = write_stream_bounded(
                OutputStream::Stderr,
                output.output.stderr,
                deadline.saturating_duration_since(Instant::now()),
            );
            if !stderr_forwarded {
                let _ = write_stream_bounded(
                    OutputStream::Stdout,
                    output_forwarding_failure_report(json, expect, suite_timeout_secs),
                    Duration::from_secs(1),
                );
                return Ok(1);
            }
            write_stream_bounded(
                OutputStream::Stdout,
                output.output.stdout,
                deadline.saturating_duration_since(Instant::now()),
            )
        }
        None => {
            io::stderr()
                .write_all(&output.output.stderr)
                .map_err(|e| e.to_string())?;
            io::stdout()
                .write_all(&output.output.stdout)
                .map_err(|e| e.to_string())?;
            true
        }
    };
    if !output_forwarded {
        let _ = write_stream_bounded(
            OutputStream::Stderr,
            output_forwarding_failure_diagnostic(suite_timeout_secs),
            Duration::from_secs(1),
        );
        return Ok(1);
    }
    Ok(exit_code)
}

#[cfg(unix)]
fn ensure_test_suite_supervision_supported() -> Result<(), String> {
    Ok(())
}

#[cfg(not(unix))]
fn ensure_test_suite_supervision_supported() -> Result<(), String> {
    Err(
        "`chelis test` whole-suite supervision requires Unix process-group semantics; \
         this target is unsupported and the command is refusing to start without \
         descendant-cleanup guarantees"
            .to_string(),
    )
}

fn read_test_batch_progress(path: &Path) -> Vec<serde_json::Value> {
    let mut rows = Vec::new();
    let Ok(bytes) = fs::read(path) else {
        return rows;
    };
    for line in bytes.split(|byte| *byte == b'\n') {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(line)
            && test_row_from_json(&value).is_some()
        {
            rows.push(value);
        }
    }
    rows
}

#[derive(Clone, Copy)]
enum SuiteIncompleteReason {
    Timeout(u64),
    AbnormalLeaderExit,
}

impl SuiteIncompleteReason {
    fn message(self) -> String {
        match self {
            Self::Timeout(seconds) => {
                format!("suite timeout after {seconds}s; terminated suite process group")
            }
            Self::AbnormalLeaderExit => {
                "suite process exited abnormally; terminated remaining process group".to_string()
            }
        }
    }

    fn status(self) -> &'static str {
        match self {
            Self::Timeout(_) => "timeout",
            Self::AbnormalLeaderExit => "abnormal-exit",
        }
    }
}

fn render_incomplete_test_suite(
    output: &std::process::Output,
    reason: SuiteIncompleteReason,
    json: bool,
    expect: Option<ExpectArg>,
    batch_progress: Vec<serde_json::Value>,
) -> Result<i32, String> {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut out = Vec::new();
    let mut passed = 0usize;
    let mut failed = 0usize;
    let mut expect_ok = 0usize;
    let mut expect_failed = 0usize;
    let mut child_test_summary = None::<(usize, usize)>;
    let mut child_expect_summary = None::<(usize, usize)>;

    if json {
        let mut seen = HashSet::<String>::new();
        let records = batch_progress.into_iter().chain(
            stdout
                .lines()
                .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok()),
        );
        for value in records {
            let is_test_row = value.get("file").is_some()
                && value.get("test").is_some()
                && matches!(
                    value.get("status").and_then(|status| status.as_str()),
                    Some("pass" | "fail")
                );
            let is_expect_row = value.get("file").is_some()
                && value.get("expect").is_some()
                && value
                    .get("verdict")
                    .and_then(|verdict| verdict.as_str())
                    .is_some();
            if !is_test_row && !is_expect_row {
                continue;
            }
            let key = serde_json::to_string(&value).map_err(|e| e.to_string())?;
            if !seen.insert(key) {
                continue;
            }
            if is_test_row {
                match value.get("status").and_then(|status| status.as_str()) {
                    Some("pass") => passed += 1,
                    Some("fail") => failed += 1,
                    _ => unreachable!("test-row shape checked above"),
                }
            } else if value.get("verdict").and_then(|verdict| verdict.as_str()) == Some("ok") {
                expect_ok += 1;
            } else {
                expect_failed += 1;
            }
            writeln!(out, "{value}").map_err(|e| e.to_string())?;
        }
        let mut suite = serde_json::json!({
            "status": reason.status(),
            "incomplete": true,
            "message": reason.message(),
        });
        if let SuiteIncompleteReason::Timeout(seconds) = reason {
            suite["timeout_seconds"] = serde_json::json!(seconds);
        }
        writeln!(out, "{}", serde_json::json!({ "suite": suite })).map_err(|e| e.to_string())?;
        if let Some(expect) = expect {
            writeln!(
                out,
                "{}",
                serde_json::json!({
                    "summary": {
                        "ok": expect_ok,
                        "failed": expect_failed + 1,
                        "mode": expect.cli_value(),
                        "incomplete": true,
                    }
                })
            )
            .map_err(|e| e.to_string())?;
        } else {
            writeln!(
                out,
                "{}",
                serde_json::json!({
                    "summary": {
                        "passed": passed,
                        "failed": failed + 1,
                        "incomplete": true,
                    }
                })
            )
            .map_err(|e| e.to_string())?;
        }
    } else {
        if stdout.trim().is_empty() {
            let mut last_file = None::<String>;
            for value in batch_progress {
                let Some(row) = test_row_from_json(&value) else {
                    continue;
                };
                if last_file.as_deref() != Some(row.file.as_str()) {
                    writeln!(out, "{}", row.file).map_err(|e| e.to_string())?;
                    last_file = Some(row.file.clone());
                }
                writeln!(out, "  {}", row.render_plain()).map_err(|e| e.to_string())?;
                match row.status {
                    TestStatus::Pass => passed += 1,
                    TestStatus::Fail => failed += 1,
                }
            }
        }
        for line in stdout.lines() {
            if let Some(counts) = parse_plain_test_summary(line) {
                child_test_summary = Some(counts);
                continue;
            }
            if let Some(counts) = parse_plain_expect_summary(line, expect) {
                child_expect_summary = Some(counts);
                continue;
            }
            if expect.is_some() && line.starts_with("OK") {
                expect_ok += 1;
            } else if expect.is_some()
                && [
                    "SHOULD-HAVE-FAILED",
                    "WRONG-DIAGNOSTIC",
                    "FIX-DETECTED",
                    "DRIFTED",
                    "CONFIG-ERROR",
                ]
                .iter()
                .any(|label| line.starts_with(label))
            {
                expect_failed += 1;
            } else if expect.is_none()
                && let Some(status) = plain_test_row_status(line)
            {
                match status {
                    TestStatus::Pass => passed += 1,
                    TestStatus::Fail => failed += 1,
                }
            }
            writeln!(out, "{line}").map_err(|e| e.to_string())?;
        }
        if let Some((summary_passed, summary_failed)) = child_test_summary {
            passed = summary_passed;
            failed = summary_failed;
        }
        if let Some((summary_ok, summary_failed)) = child_expect_summary {
            expect_ok = summary_ok;
            expect_failed = summary_failed;
        }
        writeln!(out, "test-suite").map_err(|e| e.to_string())?;
        writeln!(
            out,
            "  <suite> ....................... FAIL ({})",
            reason.message()
        )
        .map_err(|e| e.to_string())?;
        if let Some(expect) = expect {
            writeln!(
                out,
                "\n{expect_ok} ok, {} failing ({} mode, suite incomplete)",
                expect_failed + 1,
                expect.cli_value()
            )
            .map_err(|e| e.to_string())?;
        } else {
            writeln!(
                out,
                "\n{passed} passed, {} failed (suite incomplete)",
                failed + 1
            )
            .map_err(|e| e.to_string())?;
        }
    }
    write_timeout_report_bounded(out, output.stderr.clone());
    Ok(1)
}

fn write_timeout_report_bounded(stdout: Vec<u8>, stderr: Vec<u8>) {
    const REPORT_GRACE: Duration = Duration::from_secs(1);
    const STDOUT_SHARE: Duration = Duration::from_millis(800);
    let deadline = Instant::now()
        .checked_add(REPORT_GRACE)
        .unwrap_or_else(Instant::now);
    // Reserve part of the reporting grace for stderr. If stdout's consumer
    // has stopped reading, consuming the entire grace there would suppress
    // the only remaining channel for an honest incomplete-suite diagnostic.
    let stdout_written = write_stream_bounded(OutputStream::Stdout, stdout, STDOUT_SHARE);
    let mut stderr = stderr;
    if !stdout_written {
        stderr.extend_from_slice(
            b"error: suite timeout report could not be written to stdout; suite incomplete\n",
        );
    }
    let _ = write_stream_bounded(
        OutputStream::Stderr,
        stderr,
        deadline.saturating_duration_since(Instant::now()),
    );
}

#[derive(Clone, Copy)]
enum OutputStream {
    Stdout,
    Stderr,
}

fn write_stream_bounded(stream: OutputStream, bytes: Vec<u8>, budget: Duration) -> bool {
    if bytes.is_empty() {
        return true;
    }
    let (done_tx, done_rx) = std::sync::mpsc::channel::<bool>();
    thread::spawn(move || {
        let written = match stream {
            OutputStream::Stdout => {
                let mut out = io::stdout().lock();
                out.write_all(&bytes).and_then(|_| out.flush()).is_ok()
            }
            OutputStream::Stderr => {
                let mut err = io::stderr().lock();
                err.write_all(&bytes).and_then(|_| err.flush()).is_ok()
            }
        };
        let _ = done_tx.send(written);
    });
    // The command dispatcher calls process::exit immediately after a failure
    // return, terminating a writer blocked by consumer backpressure.
    done_rx.recv_timeout(budget).unwrap_or(false)
}

fn output_forwarding_failure_diagnostic(timeout_secs: u64) -> Vec<u8> {
    format!(
        "error: suite output forwarding exceeded the {timeout_secs}s \
         whole-command deadline; suite incomplete\n"
    )
    .into_bytes()
}

fn output_forwarding_failure_report(
    json: bool,
    expect: Option<ExpectArg>,
    timeout_secs: u64,
) -> Vec<u8> {
    if json {
        let summary = if let Some(expect) = expect {
            serde_json::json!({
                "summary": {
                    "ok": 0,
                    "failed": 1,
                    "mode": expect.cli_value(),
                    "incomplete": true,
                }
            })
        } else {
            serde_json::json!({
                "summary": {
                    "passed": 0,
                    "failed": 1,
                    "incomplete": true,
                }
            })
        };
        return format!(
            "{}\n{summary}\n",
            serde_json::json!({
                "suite": {
                    "status": "timeout",
                    "incomplete": true,
                    "message": format!(
                        "suite output forwarding exceeded the {timeout_secs}s whole-command deadline"
                    ),
                }
            })
        )
        .into_bytes();
    }
    let summary = if let Some(expect) = expect {
        format!(
            "0 ok, 1 failing ({} mode, suite incomplete)",
            expect.cli_value()
        )
    } else {
        "0 passed, 1 failed (suite incomplete)".to_string()
    };
    format!(
        "test-suite\n  <suite> ....................... FAIL \
         (output forwarding exceeded the {timeout_secs}s whole-command deadline)\n\
         \n{summary}\n"
    )
    .into_bytes()
}

fn parse_plain_test_summary(line: &str) -> Option<(usize, usize)> {
    let (passed, failed) = line.split_once(" passed, ")?;
    let failed = failed.strip_suffix(" failed")?;
    Some((passed.parse().ok()?, failed.parse().ok()?))
}

fn parse_plain_expect_summary(line: &str, expect: Option<ExpectArg>) -> Option<(usize, usize)> {
    let expect = expect?;
    let (ok, failed) = line.split_once(" ok, ")?;
    let failed = failed.strip_suffix(&format!(" failing ({} mode)", expect.cli_value()))?;
    Some((ok.parse().ok()?, failed.parse().ok()?))
}

fn plain_test_row_status(line: &str) -> Option<TestStatus> {
    let row = line.strip_prefix("  ")?;
    if row.ends_with(" PASS") {
        return Some(TestStatus::Pass);
    }
    let (_, suffix) = row.rsplit_once(" FAIL")?;
    if suffix.is_empty() || (suffix.starts_with(" (") && suffix.ends_with(')')) {
        return Some(TestStatus::Fail);
    }
    None
}

fn test_row_from_json(value: &serde_json::Value) -> Option<TestRow> {
    let status = match value.get("status")?.as_str()? {
        "pass" => TestStatus::Pass,
        "fail" => TestStatus::Fail,
        _ => return None,
    };
    Some(TestRow {
        file: value.get("file")?.as_str()?.to_string(),
        test: value.get("test")?.as_str()?.to_string(),
        status,
        message: value
            .get("message")
            .and_then(|message| message.as_str())
            .map(str::to_string),
    })
}

#[derive(Clone)]
struct TestFileJob {
    index: usize,
    file: PathBuf,
    rel_display: String,
}

struct TestFileResult {
    index: usize,
    rel_display: String,
    rows: Vec<TestRow>,
}

/// Discover and execute Chelis-native tests.
///
/// Walks `.ch` files under `path` (default `tests/` in CWD), extracts nullary
/// `def test_*` functions, and evaluates each against the surrounding module
/// with a shared reef graph. Returns the process exit code:
///
/// * `0` — every selected test passed.
/// * `1` — at least one test failed.
/// * `2` — runner error (missing dir, missing reef package, or no test files parsed).
#[allow(clippy::too_many_arguments)]
fn cmd_test(
    path: Option<&Path>,
    filter: Option<&str>,
    json: bool,
    timeout_secs: u64,
    jobs: TestJobs,
    batch_mode: TestBatchMode,
    expect: Option<ExpectArg>,
    progress_file: Option<&Path>,
) -> Result<i32, String> {
    // `--expect` runs an expected-failure suite over every probe; a name filter
    // is both ignored by `run_expect` and a false-green risk (a no-match filter
    // would otherwise short-circuit to "0 passed" before expect classification).
    if expect.is_some() && filter.is_some() {
        return Err(
            "`--filter` cannot be combined with `--expect`; an expected-failure suite runs every probe"
                .to_string(),
        );
    }

    let raw_cwd = env::current_dir().map_err(|e| format!("failed to read cwd: {e}"))?;
    let target = match path {
        Some(p) => p.to_path_buf(),
        None => raw_cwd.join("tests"),
    };

    if !target.exists() {
        return Err(format!(
            "path `{}` does not exist. Pass a tests directory or a single .ch file",
            target.display()
        ));
    }

    // Bucket 6c: when the user runs `chelis test path/to/file.ch` from a
    // directory that is not itself inside a reef package, derive the
    // reef-package root from the target path instead of the raw cwd.
    // We start the lookup at the target's directory (or the target itself
    // if it is a directory) and walk upward; if no reef.toml is found we
    // fall back to the raw cwd so the existing "no reef.toml" error path
    // still fires with its actionable message.
    let target_dir_for_reef = if target.is_dir() {
        target.clone()
    } else {
        target
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| raw_cwd.clone())
    };
    let cwd = match chelis_reef::find_package_root_for_dir(&target_dir_for_reef) {
        Ok(Some(root)) => root,
        // No reef found from the target — try the raw cwd next; if that
        // fails too, fall through with raw_cwd and let `prepare_reef_graph`
        // emit its standard error.
        _ => match chelis_reef::find_package_root_for_dir(&raw_cwd) {
            Ok(Some(root)) => root,
            _ => raw_cwd.clone(),
        },
    };

    let test_files = discover_test_files(&target)?;
    if test_files.is_empty() {
        // An `--expect` suite that finds zero probes is a misconfiguration, not
        // a pass: a guard that runs nothing is silently green (the exact
        // failure mode where renamed/removed probes disable a CI gate). Fail
        // loudly instead of the `cargo test`/`pytest` "0 passed" ergonomics.
        if expect.is_some() {
            return Err(format!(
                "no .ch test files under `{}`, but --expect requires a non-empty suite \
                 (an expected-failure guard that runs zero probes is silently green)",
                target.display()
            ));
        }
        // Empty test dir is a legitimate CI state (no tests yet, or all filtered
        // out before discovery). Report 0/0 and exit 0 — matches `cargo test` and
        // `pytest` ergonomics. A truly missing `tests/` dir already errored above.
        let stdout = io::stdout();
        let mut out = stdout.lock();
        if json {
            writeln!(out, "{{\"summary\":{{\"passed\":0,\"failed\":0}}}}")
                .map_err(|e| e.to_string())?;
        } else {
            writeln!(out, "0 passed, 0 failed").map_err(|e| e.to_string())?;
        }
        return Ok(0);
    }

    // Phase G' — fast path: when `--filter <substring>` matches zero
    // tests across every discovered file, skip the (expensive)
    // `compile_reef_context` build entirely. The pre-G' code paid the
    // full ~65 s parent overhead even for `chelis test --filter __no_match__`,
    // a documented bottleneck in `docs/perf_baseline.md` and the archived
    // Phase J notes. Surf-parsing
    // each test file is ~10 ms; that's the ceiling we accept here.
    //
    // We do this BEFORE `prepare_reef_graph` runs so a no-match
    // invocation pays only the file-walk + per-file parse cost.
    if let Some(needle) = filter
        && !any_file_has_filter_match(&test_files, &cwd, needle)?
    {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        if json {
            writeln!(out, "{{\"summary\":{{\"passed\":0,\"failed\":0}}}}")
                .map_err(|e| e.to_string())?;
        } else {
            writeln!(out, "0 passed, 0 failed").map_err(|e| e.to_string())?;
        }
        return Ok(0);
    }

    // Phase H: build the `CompiledContext` ONCE in the parent and hand it
    // to each per-file worker via a bincode-encoded tempfile. The
    // context wraps the lockfile-resolved package graph, the linked
    // library decls, and the (currently unused-by-Phase-H) library
    // type-env / DAG snapshots. Workers rehydrate the context and
    // borrow its `PreparedReefGraph` to drive the legacy
    // `compile_with_reef_graph` + `prepare_eval` path — they SKIP the
    // per-file `prepare_reef_graph` walk that the pre-Phase-H worker
    // paid on every spawn. The full in-context evaluator
    // (`eval_in_context`) is wired up at the API boundary but not used
    // here because the Phase G/G' linearity + host-runtime semantics
    // do not yet match the monolithic evaluator on the chelis-std
    // corpus; that integration is the Phase H' / G' follow-up.
    //
    // The cwd doubles as the package_dir; `compile_reef_context` resolves
    // the lockfile starting from there. The `reef_home` argument is
    // reserved for the Phase I disk cache and currently unused by the
    // encode path; we still pass `$CHELIS_REEF_HOME` if set so the
    // future cache key matches today's runtime layout.
    // First validate the package: `prepare_reef_graph` is the
    // pre-Phase-H gate that fails fast on "no reef.toml" / lockfile
    // resolution errors with exit-2 ergonomics. Tests assert that
    // contract, so we keep it as the FIRST thing the parent does.
    // The result is intentionally dropped — `compile_reef_context`
    // below builds its own graph; we only invoke this for the
    // up-front "is this a reef package?" gate.
    let _ = chelis_reef::prepare_reef_graph(&cwd)?;

    // Phase G' (final) — with the linearity divergence root-caused
    // (annotate_ir_program now registers prelude ADTs, matching
    // the _with_context variants) and the worker re-wired through
    // prepare_eval_in_context, the parent re-enables the
    // `compile_reef_context` build. The encoded context is handed to
    // each per-file worker via a bincode tempfile + env var; workers
    // rehydrate the library snapshot ONCE per spawn instead of
    // re-running the full reef graph + compile pipeline per file.
    //
    // Phase K: route through `load_or_compile_for_package` so an
    // unchanged-source re-run (typical CI / dev-loop iteration on
    // tests) skips the ~67s library compile entirely. When
    // CHELIS_REEF_HOME is unset the helper falls through to a direct
    // context compile. This parent context build is intentionally
    // fail-fast: a shared dependency compile error is one real runner
    // error, not N repeated per-worker fallbacks.
    let reef_home_path = env::var("CHELIS_REEF_HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    let context =
        match chelis_compiler_api::load_or_compile_for_package(&reef_home_path, &cwd, true) {
            Ok(context) => context,
            Err(err) if is_local_registry_hash_unsupported(&err) => {
                chelis_compiler_api::compile_reef_context(&reef_home_path, &cwd).map_err(|err| {
                    format!("compile test context: {}", compiler_error_messages(&err))
                })?
            }
            Err(err) => {
                return Err(format!(
                    "compile test context: {}",
                    compiler_error_messages(&err)
                ));
            }
        };
    let context_bytes = context.encode()?;
    drop(context);
    let context_tempfile = CompiledContextTempfile::write(&context_bytes)?;
    drop(context_bytes);

    let mut passed: usize = 0;
    let mut failed: usize = 0;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let self_path =
        std::env::current_exe().map_err(|e| format!("could not locate chelis binary: {e}"))?;

    let test_jobs = test_files
        .iter()
        .enumerate()
        .map(|(index, file)| TestFileJob {
            index,
            file: file.clone(),
            rel_display: file
                .strip_prefix(&cwd)
                .unwrap_or(file.as_path())
                .display()
                .to_string(),
        })
        .collect::<Vec<_>>();

    // Expected-failure mode reinterprets each file's result against its
    // `.expect` sidecar and always runs per-file isolated (contract §5: a
    // pinned failure mode can poison a shared compile unit), so it bypasses the
    // streaming batch/summary path entirely.
    if let Some(expect) = expect {
        return run_expect(
            expect.mode(),
            &self_path,
            &cwd,
            &test_jobs,
            jobs,
            timeout_secs,
            context_tempfile.path(),
            json,
            &mut out,
        );
    }

    match batch_mode {
        TestBatchMode::File => {
            let worker_count = jobs.resolve(test_jobs.len());
            run_test_file_jobs(
                &self_path,
                &cwd,
                &test_jobs,
                worker_count,
                filter,
                timeout_secs,
                context_tempfile.path(),
                json,
                &mut out,
                &mut passed,
                &mut failed,
            )?;
        }
        TestBatchMode::Auto => {
            run_test_jobs_auto(
                &self_path,
                &cwd,
                &test_jobs,
                jobs,
                filter,
                timeout_secs,
                context_tempfile.path(),
                json,
                &mut out,
                &mut passed,
                &mut failed,
                progress_file,
            )?;
        }
    }

    if json {
        writeln!(
            out,
            "{{\"summary\":{{\"passed\":{passed},\"failed\":{failed}}}}}"
        )
        .map_err(|e| e.to_string())?;
    } else {
        writeln!(out, "\n{passed} passed, {failed} failed").map_err(|e| e.to_string())?;
    }

    Ok(if failed == 0 { 0 } else { 1 })
}

/// Run the expected-failure suites (`tests_neg/` / `tests_blocked/`) with
/// per-file isolated workers, classifying each file against its `.expect`
/// sidecar via [`chelis_conformance::expect`]. Exit code is `0` iff every file
/// is healthy (still fails with the pinned diagnostic); any regression,
/// FIX-detected, drift, or config error is non-zero so CI blocks on it.
#[allow(clippy::too_many_arguments)]
fn run_expect(
    mode: chelis_conformance::expect::ExpectMode,
    self_path: &Path,
    cwd: &Path,
    test_jobs: &[TestFileJob],
    jobs: TestJobs,
    timeout_secs: u64,
    context_path: &Path,
    json: bool,
    out: &mut impl Write,
) -> Result<i32, String> {
    use chelis_conformance::expect::{FileOutcome, Sidecar, classify};

    let worker_count = jobs.resolve(test_jobs.len());
    let results = collect_test_file_jobs(
        self_path,
        cwd,
        test_jobs,
        worker_count,
        None,
        timeout_secs,
        TestFileWorkerOptions {
            compiled_context_path: Some(context_path),
            expect_file_diagnostic: true,
        },
    )?;

    let mut ok = 0usize;
    let mut bad = 0usize;
    for job in test_jobs {
        let empty: Vec<TestRow> = Vec::new();
        let rows = results.get(&job.index).unwrap_or(&empty);
        let outcome = FileOutcome::from_rows(
            rows.iter()
                .map(|r| (matches!(r.status, TestStatus::Pass), r.message.as_deref())),
        );
        let sidecar_path = job.file.with_extension("expect");
        let verdict = match std::fs::read_to_string(&sidecar_path) {
            Ok(text) => match Sidecar::parse(&text) {
                Ok(sidecar) => classify(mode, &outcome, Some(&sidecar)),
                Err(reason) => chelis_conformance::expect::Verdict::ConfigError { reason },
            },
            // A present-but-unreadable sidecar (e.g. non-UTF8) is a config error
            // with an accurate message, not the "missing sidecar" that
            // `classify(.., None)` would report.
            Err(e) if sidecar_path.exists() => chelis_conformance::expect::Verdict::ConfigError {
                reason: format!("unreadable .expect sidecar {}: {e}", sidecar_path.display()),
            },
            Err(_) => classify(mode, &outcome, None),
        };
        if verdict.is_ok() {
            ok += 1;
        } else {
            bad += 1;
        }
        emit_expect(out, json, mode, &job.rel_display, &verdict)?;
    }

    if json {
        writeln!(
            out,
            "{{\"summary\":{{\"ok\":{ok},\"failed\":{bad},\"mode\":\"{}\"}}}}",
            mode.as_str()
        )
        .map_err(|e| e.to_string())?;
    } else {
        writeln!(out, "\n{ok} ok, {bad} failing ({} mode)", mode.as_str())
            .map_err(|e| e.to_string())?;
    }
    Ok(if bad == 0 { 0 } else { 1 })
}

/// Human-readable one-line detail for an expected-failure verdict.
fn expect_detail(verdict: &chelis_conformance::expect::Verdict) -> String {
    use chelis_conformance::expect::Verdict;
    match verdict {
        Verdict::Ok => String::new(),
        Verdict::ShouldHaveFailed => "case passed but was required to fail".to_string(),
        Verdict::WrongDiagnostic { expected, .. } => {
            format!("failed without the required diagnostic substring {expected:?}")
        }
        Verdict::FixDetected { .. } => {
            "probe passes: upstream fixed the blocker; de-narrow now".to_string()
        }
        Verdict::Drifted { expected, .. } => {
            format!("failed with a different diagnostic (expected {expected:?})")
        }
        Verdict::ConfigError { reason } => reason.clone(),
    }
}

/// Emit one file's expected-failure verdict (NDJSON record or a plain line +
/// indented detail, mirroring `chelis test`'s two output shapes).
fn emit_expect(
    out: &mut impl Write,
    json: bool,
    mode: chelis_conformance::expect::ExpectMode,
    file: &str,
    verdict: &chelis_conformance::expect::Verdict,
) -> Result<(), String> {
    use chelis_conformance::expect::Verdict;
    if json {
        let mut record = serde_json::json!({
            "file": file,
            "expect": mode.as_str(),
            "verdict": verdict.tag(),
            "detail": expect_detail(verdict),
        });
        if let Verdict::WrongDiagnostic { got, .. } | Verdict::Drifted { got, .. } = verdict {
            // Machine consumers need the real compiler diagnostic to triage
            // drift without re-running the probe in plain mode (chelis#967).
            record["got"] = serde_json::json!(got);
        }
        writeln!(out, "{record}").map_err(|e| e.to_string())?;
        return Ok(());
    }

    let label = match verdict {
        Verdict::Ok => "OK",
        Verdict::ShouldHaveFailed => "SHOULD-HAVE-FAILED",
        Verdict::WrongDiagnostic { .. } => "WRONG-DIAGNOSTIC",
        Verdict::FixDetected { .. } => "FIX-DETECTED",
        Verdict::Drifted { .. } => "DRIFTED",
        Verdict::ConfigError { .. } => "CONFIG-ERROR",
    };
    writeln!(out, "{label:<18} {file}").map_err(|e| e.to_string())?;
    match verdict {
        Verdict::WrongDiagnostic { expected, got } | Verdict::Drifted { expected, got } => {
            writeln!(out, "    expected diagnostic substring: {expected:?}")
                .map_err(|e| e.to_string())?;
            for g in got {
                writeln!(out, "    got: {g}").map_err(|e| e.to_string())?;
            }
        }
        Verdict::FixDetected { instructions } => {
            writeln!(
                out,
                "    upstream fixed this blocker; de-narrowing instructions:"
            )
            .map_err(|e| e.to_string())?;
            for line in instructions {
                writeln!(out, "      {line}").map_err(|e| e.to_string())?;
            }
        }
        Verdict::ConfigError { reason } => {
            writeln!(out, "    {reason}").map_err(|e| e.to_string())?;
        }
        Verdict::Ok | Verdict::ShouldHaveFailed => {}
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestBatchManifest {
    files: Vec<TestBatchManifestFile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestBatchManifestFile {
    index: usize,
    file: PathBuf,
    rel_display: String,
    tests: Vec<String>,
}

struct ClassifiedTestJobs {
    batch_jobs: Vec<TestBatchManifestFile>,
    file_jobs: Vec<TestFileJob>,
}

struct TestBatchManifestTempfile {
    #[allow(dead_code)]
    file: tempfile::NamedTempFile,
}

impl TestBatchManifestTempfile {
    fn write(manifest: &TestBatchManifest) -> Result<Self, String> {
        let mut builder = tempfile::Builder::new();
        builder.prefix("chelis-test-batch-").suffix(".json");
        let mut file = builder
            .tempfile()
            .map_err(|e| format!("create test-batch manifest tempfile: {e}"))?;
        let bytes = serde_json::to_vec(manifest)
            .map_err(|e| format!("serialize test-batch manifest: {e}"))?;
        std::io::Write::write_all(file.as_file_mut(), &bytes)
            .map_err(|e| format!("write test-batch manifest tempfile: {e}"))?;
        std::io::Write::flush(file.as_file_mut())
            .map_err(|e| format!("flush test-batch manifest tempfile: {e}"))?;
        Ok(Self { file })
    }

    fn path(&self) -> &Path {
        self.file.path()
    }
}

fn compiler_error_messages(err: &chelis_compiler_api::compiler::CompilerError) -> String {
    let messages = err
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>()
        .join("; ");
    if messages.is_empty() {
        err.stage.clone()
    } else {
        messages
    }
}

fn is_local_registry_hash_unsupported(err: &chelis_compiler_api::compiler::CompilerError) -> bool {
    err.errors.iter().any(|diagnostic| {
        diagnostic.kind == "hash_error" && diagnostic.message.contains("LocalRegistry")
    })
}

#[allow(clippy::too_many_arguments)]
fn run_test_jobs_auto(
    self_path: &Path,
    cwd: &Path,
    test_jobs: &[TestFileJob],
    jobs: TestJobs,
    filter: Option<&str>,
    timeout_secs: u64,
    compiled_context_path: &Path,
    json: bool,
    out: &mut impl Write,
    passed: &mut usize,
    failed: &mut usize,
    progress_file: Option<&Path>,
) -> Result<(), String> {
    let classified = classify_test_jobs_for_batch(test_jobs, filter);
    let mut rows_by_index = BTreeMap::<usize, Vec<TestRow>>::new();
    let mut file_fallback_jobs = classified.file_jobs;

    if !classified.batch_jobs.is_empty() {
        match run_test_batch_subprocess(
            self_path,
            cwd,
            &classified.batch_jobs,
            timeout_secs,
            compiled_context_path,
            progress_file,
        )? {
            BatchSubprocessOutcome::Rows(rows) => {
                if !group_batch_rows_by_file(&classified.batch_jobs, rows, &mut rows_by_index) {
                    file_fallback_jobs.extend(batch_jobs_as_file_jobs(&classified.batch_jobs));
                }
            }
            BatchSubprocessOutcome::Fallback => {
                file_fallback_jobs.extend(batch_jobs_as_file_jobs(&classified.batch_jobs));
            }
        }
    }

    if !file_fallback_jobs.is_empty() {
        let worker_count = jobs.resolve(file_fallback_jobs.len());
        rows_by_index.extend(collect_test_file_jobs(
            self_path,
            cwd,
            &file_fallback_jobs,
            worker_count,
            filter,
            timeout_secs,
            TestFileWorkerOptions {
                compiled_context_path: Some(compiled_context_path),
                expect_file_diagnostic: false,
            },
        )?);
    }

    for job in test_jobs {
        if let Some(rows) = rows_by_index.remove(&job.index) {
            emit_test_file_rows(out, json, &job.rel_display, &rows, passed, failed)?;
        }
    }

    Ok(())
}

fn batch_jobs_as_file_jobs(batch_jobs: &[TestBatchManifestFile]) -> Vec<TestFileJob> {
    batch_jobs
        .iter()
        .map(|job| TestFileJob {
            index: job.index,
            file: job.file.clone(),
            rel_display: job.rel_display.clone(),
        })
        .collect()
}

fn group_batch_rows_by_file(
    batch_jobs: &[TestBatchManifestFile],
    rows: Vec<TestRow>,
    rows_by_index: &mut BTreeMap<usize, Vec<TestRow>>,
) -> bool {
    let expected_rows: usize = batch_jobs.iter().map(|job| job.tests.len()).sum();
    if rows.len() != expected_rows {
        return false;
    }
    let index_by_file = batch_jobs
        .iter()
        .map(|job| (job.rel_display.clone(), job.index))
        .collect::<HashMap<_, _>>();
    for row in rows {
        let Some(index) = index_by_file.get(&row.file).copied() else {
            return false;
        };
        rows_by_index.entry(index).or_default().push(row);
    }
    true
}

fn classify_test_jobs_for_batch(
    test_jobs: &[TestFileJob],
    filter: Option<&str>,
) -> ClassifiedTestJobs {
    let mut batch_jobs = Vec::new();
    let mut file_jobs = Vec::new();
    let mut seen_top_level_names = HashSet::<String>::new();

    for job in test_jobs {
        let Ok(source) = fs::read_to_string(&job.file) else {
            file_jobs.push(job.clone());
            continue;
        };
        let Ok(parsed) = chelis_surf::parser::parse_str(&source) else {
            file_jobs.push(job.clone());
            continue;
        };
        let flat = flatten_module_decls(&parsed);
        let tests = match enumerate_test_fns(&flat, filter, &job.rel_display) {
            EnumerationOutcome::Tests(tests) => tests,
            EnumerationOutcome::Error(_) => {
                file_jobs.push(job.clone());
                continue;
            }
        };
        if tests.is_empty() {
            continue;
        }
        if flat.iter().any(|decl| matches!(decl, Decl::LetDef { .. })) {
            file_jobs.push(job.clone());
            continue;
        }
        let names = top_level_decl_names(&flat);
        if names.iter().any(|name| seen_top_level_names.contains(name)) {
            file_jobs.push(job.clone());
            continue;
        }
        seen_top_level_names.extend(names);
        batch_jobs.push(TestBatchManifestFile {
            index: job.index,
            file: job.file.clone(),
            rel_display: job.rel_display.clone(),
            tests: tests.into_iter().map(|test| test.name).collect(),
        });
    }

    ClassifiedTestJobs {
        batch_jobs,
        file_jobs,
    }
}

fn top_level_decl_names(decls: &[Decl]) -> Vec<String> {
    let mut out = Vec::new();
    for decl in decls {
        match decl {
            Decl::FunDef { name, .. }
            | Decl::Sig { name, .. }
            | Decl::TypeDef { name, .. }
            | Decl::TypeAlias { name, .. }
            | Decl::MacroDef { name, .. } => out.push(name.clone()),
            Decl::Dim { names, .. } => out.extend(names.iter().cloned()),
            _ => {}
        }
    }
    out
}

enum BatchSubprocessOutcome {
    Rows(Vec<TestRow>),
    Fallback,
}

fn run_test_batch_subprocess(
    self_path: &Path,
    cwd: &Path,
    batch_jobs: &[TestBatchManifestFile],
    timeout_secs: u64,
    compiled_context_path: &Path,
    progress_file: Option<&Path>,
) -> Result<BatchSubprocessOutcome, String> {
    let manifest = TestBatchManifest {
        files: batch_jobs.to_vec(),
    };
    let manifest_tempfile = TestBatchManifestTempfile::write(&manifest)?;

    let mut cmd = std::process::Command::new(self_path);
    cmd.arg("__test_batch")
        .arg("--manifest")
        .arg(manifest_tempfile.path())
        .arg("--timeout")
        .arg(timeout_secs.to_string())
        .current_dir(cwd)
        .env("CHELIS_TEST_COMPILED_CONTEXT", compiled_context_path);

    let mut progress = progress_file
        .map(|path| {
            fs::OpenOptions::new()
                .append(true)
                .open(path)
                .map_err(|e| format!("open suite progress file `{}`: {e}", path.display()))
        })
        .transpose()?;
    let output = match run_batch_worker_command_with_timeout(
        cmd,
        batch_worker_timeout(batch_jobs, timeout_secs),
        |line| {
            let line = line.strip_suffix(b"\n").unwrap_or(line);
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            let Ok(value) = serde_json::from_slice::<serde_json::Value>(line) else {
                return;
            };
            if test_row_from_json(&value).is_none() {
                return;
            }
            if let Some(progress) = progress.as_mut() {
                let _ = progress.write_all(line);
                let _ = progress.write_all(b"\n");
                let _ = progress.flush();
            }
        },
    ) {
        Ok(output) => output,
        Err(_) => return Ok(BatchSubprocessOutcome::Fallback),
    };
    if output.timed_out {
        return Ok(BatchSubprocessOutcome::Fallback);
    }

    let stdout = String::from_utf8_lossy(&output.output.stdout);
    let mut rows = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            return Ok(BatchSubprocessOutcome::Fallback);
        };
        let Some(file) = value.get("file").and_then(|v| v.as_str()) else {
            return Ok(BatchSubprocessOutcome::Fallback);
        };
        let Some(test) = value.get("test").and_then(|v| v.as_str()) else {
            return Ok(BatchSubprocessOutcome::Fallback);
        };
        let Some(status_s) = value.get("status").and_then(|v| v.as_str()) else {
            return Ok(BatchSubprocessOutcome::Fallback);
        };
        let status = match status_s {
            "pass" => TestStatus::Pass,
            "fail" => TestStatus::Fail,
            _ => return Ok(BatchSubprocessOutcome::Fallback),
        };
        let message = value
            .get("message")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        rows.push(TestRow {
            file: file.to_string(),
            test: test.to_string(),
            status,
            message,
        });
    }

    let should_fallback = match output.output.status.code() {
        None => true,
        Some(0) => false,
        Some(1) => rows.is_empty(),
        Some(_) => true,
    };
    if should_fallback {
        return Ok(BatchSubprocessOutcome::Fallback);
    }

    Ok(BatchSubprocessOutcome::Rows(rows))
}

fn batch_worker_timeout(batch_jobs: &[TestBatchManifestFile], timeout_secs: u64) -> Duration {
    let selected_count: usize = batch_jobs.iter().map(|job| job.tests.len()).sum();
    let per_test = timeout_secs.max(1);
    let test_budget = per_test
        .saturating_mul(selected_count.saturating_add(1) as u64)
        .saturating_add(10);
    Duration::from_secs(test_budget.max(60))
}

fn collect_test_file_jobs(
    self_path: &Path,
    cwd: &Path,
    test_jobs: &[TestFileJob],
    worker_count: usize,
    filter: Option<&str>,
    timeout_secs: u64,
    worker_options: TestFileWorkerOptions<'_>,
) -> Result<BTreeMap<usize, Vec<TestRow>>, String> {
    if worker_count <= 1 {
        let mut out = BTreeMap::new();
        for job in test_jobs {
            let rows = run_test_file_subprocess(
                self_path,
                cwd,
                &job.file,
                &job.rel_display,
                filter,
                timeout_secs,
                worker_options,
            );
            out.insert(job.index, rows);
        }
        return Ok(out);
    }

    let jobs = Arc::new(test_jobs.to_vec());
    let next_index = Arc::new(AtomicUsize::new(0));
    let (result_tx, result_rx) = std::sync::mpsc::channel::<TestFileResult>();
    let self_path = self_path.to_path_buf();
    let cwd = cwd.to_path_buf();
    let filter = filter.map(str::to_string);
    let compiled_context_path = worker_options.compiled_context_path.map(Path::to_path_buf);
    let expect_file_diagnostic = worker_options.expect_file_diagnostic;
    let mut handles = Vec::new();

    for _ in 0..worker_count {
        let jobs = Arc::clone(&jobs);
        let next_index = Arc::clone(&next_index);
        let result_tx = result_tx.clone();
        let self_path = self_path.clone();
        let cwd = cwd.clone();
        let filter = filter.clone();
        let compiled_context_path = compiled_context_path.clone();
        handles.push(thread::spawn(move || {
            loop {
                let index = next_index.fetch_add(1, Ordering::SeqCst);
                let Some(job) = jobs.get(index).cloned() else {
                    break;
                };
                let rows = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_test_file_subprocess(
                        &self_path,
                        &cwd,
                        &job.file,
                        &job.rel_display,
                        filter.as_deref(),
                        timeout_secs,
                        TestFileWorkerOptions {
                            compiled_context_path: compiled_context_path.as_deref(),
                            expect_file_diagnostic,
                        },
                    )
                })) {
                    Ok(rows) => rows,
                    Err(_) => vec![TestRow {
                        file: job.rel_display.clone(),
                        test: "<file>".to_string(),
                        status: TestStatus::Fail,
                        message: Some("parent worker thread panicked".to_string()),
                    }],
                };
                let _ = result_tx.send(TestFileResult {
                    index: job.index,
                    rel_display: job.rel_display,
                    rows,
                });
            }
        }));
    }
    drop(result_tx);

    let mut out = BTreeMap::new();
    let mut received = 0usize;
    while received < test_jobs.len() {
        let result = result_rx
            .recv()
            .map_err(|_| "test worker pool terminated before every file completed".to_string())?;
        received += 1;
        out.insert(result.index, result.rows);
    }

    for handle in handles {
        handle
            .join()
            .map_err(|_| "test worker pool thread panicked".to_string())?;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn run_test_file_jobs(
    self_path: &Path,
    cwd: &Path,
    test_jobs: &[TestFileJob],
    worker_count: usize,
    filter: Option<&str>,
    timeout_secs: u64,
    compiled_context_path: &Path,
    json: bool,
    out: &mut impl Write,
    passed: &mut usize,
    failed: &mut usize,
) -> Result<(), String> {
    if worker_count <= 1 {
        for job in test_jobs {
            let rows = run_test_file_subprocess(
                self_path,
                cwd,
                &job.file,
                &job.rel_display,
                filter,
                timeout_secs,
                TestFileWorkerOptions {
                    compiled_context_path: Some(compiled_context_path),
                    expect_file_diagnostic: false,
                },
            );
            emit_test_file_rows(out, json, &job.rel_display, &rows, passed, failed)?;
        }
        return Ok(());
    }

    let jobs = Arc::new(test_jobs.to_vec());
    let next_index = Arc::new(AtomicUsize::new(0));
    let (result_tx, result_rx) = std::sync::mpsc::channel::<TestFileResult>();
    let self_path = self_path.to_path_buf();
    let cwd = cwd.to_path_buf();
    let filter = filter.map(str::to_string);
    let compiled_context_path = compiled_context_path.to_path_buf();
    let mut handles = Vec::new();

    for _ in 0..worker_count {
        let jobs = Arc::clone(&jobs);
        let next_index = Arc::clone(&next_index);
        let result_tx = result_tx.clone();
        let self_path = self_path.clone();
        let cwd = cwd.clone();
        let filter = filter.clone();
        let compiled_context_path = compiled_context_path.clone();
        handles.push(thread::spawn(move || {
            loop {
                let index = next_index.fetch_add(1, Ordering::SeqCst);
                let Some(job) = jobs.get(index).cloned() else {
                    break;
                };
                let rows = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_test_file_subprocess(
                        &self_path,
                        &cwd,
                        &job.file,
                        &job.rel_display,
                        filter.as_deref(),
                        timeout_secs,
                        TestFileWorkerOptions {
                            compiled_context_path: Some(&compiled_context_path),
                            expect_file_diagnostic: false,
                        },
                    )
                })) {
                    Ok(rows) => rows,
                    Err(_) => vec![TestRow {
                        file: job.rel_display.clone(),
                        test: "<file>".to_string(),
                        status: TestStatus::Fail,
                        message: Some("parent worker thread panicked".to_string()),
                    }],
                };
                let _ = result_tx.send(TestFileResult {
                    index: job.index,
                    rel_display: job.rel_display,
                    rows,
                });
            }
        }));
    }
    drop(result_tx);

    let mut pending = BTreeMap::<usize, TestFileResult>::new();
    let mut next_emit = 0usize;
    let mut received = 0usize;
    while received < test_jobs.len() {
        let result = result_rx
            .recv()
            .map_err(|_| "test worker pool terminated before every file completed".to_string())?;
        received += 1;
        pending.insert(result.index, result);
        while let Some(result) = pending.remove(&next_emit) {
            emit_test_file_rows(out, json, &result.rel_display, &result.rows, passed, failed)?;
            next_emit += 1;
        }
    }

    for handle in handles {
        handle
            .join()
            .map_err(|_| "test worker pool thread panicked".to_string())?;
    }
    Ok(())
}

fn emit_test_file_rows(
    out: &mut impl Write,
    json: bool,
    rel_display: &str,
    rows: &[TestRow],
    passed: &mut usize,
    failed: &mut usize,
) -> Result<(), String> {
    if rows.is_empty() {
        // Nothing matched the filter in this file — skip silently so the
        // operator can narrow a run without seeing noise.
        return Ok(());
    }

    if json {
        for row in rows {
            writeln!(out, "{}", row.to_json()).map_err(|e| e.to_string())?;
            match row.status {
                TestStatus::Pass => *passed += 1,
                TestStatus::Fail => *failed += 1,
            }
        }
    } else {
        writeln!(out, "{rel_display}").map_err(|e| e.to_string())?;
        for row in rows {
            writeln!(out, "  {}", row.render_plain()).map_err(|e| e.to_string())?;
            match row.status {
                TestStatus::Pass => *passed += 1,
                TestStatus::Fail => *failed += 1,
            }
        }
    }
    Ok(())
}

/// Phase G' — pre-flight filter scan. Returns `true` if any test file
/// contains at least one `def test_*` whose `<rel_display>::<name>`
/// key contains the filter substring. Returns `false` when the filter
/// matches zero tests across every file. A parse error on any file
/// surfaces as `Err`; the caller treats that as "fall through to the
/// normal path" because suppressing it here would silently swallow a
/// real file-level failure.
///
/// This intentionally mirrors the per-file enumeration logic in
/// `enumerate_test_fns` so the pre-flight is consistent with the
/// per-worker filter check; it just runs with no library link, so it's
/// dramatically cheaper than the full `compile_reef_context` walk.
fn any_file_has_filter_match(
    test_files: &[PathBuf],
    cwd: &Path,
    filter_needle: &str,
) -> Result<bool, String> {
    for file in test_files {
        let rel_display = file
            .strip_prefix(cwd)
            .unwrap_or(file.as_path())
            .display()
            .to_string();
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            // Unreadable file: don't short-circuit; let the normal path
            // surface the read error as a per-file FAIL row.
            Err(_) => return Ok(true),
        };
        let parsed = match chelis_surf::parser::parse_str(&source) {
            Ok(p) => p,
            // Parse error: same — let the normal path surface the
            // file-level error.
            Err(_) => return Ok(true),
        };
        let flat = flatten_module_decls(&parsed);
        for decl in &flat {
            let Decl::FunDef {
                name,
                params,
                ret_ty,
                ..
            } = decl
            else {
                continue;
            };
            // Same gating as enumerate_test_fns: nullary, test_-prefixed,
            // unit-typed.
            if !name.starts_with("test_") || name == "test_" {
                continue;
            }
            if !params.is_empty() {
                continue;
            }
            if let Some(ty) = ret_ty
                && !is_unit_type(ty)
            {
                continue;
            }
            let key = format!("{rel_display}::{name}");
            if key.contains(filter_needle) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn discover_test_files(target: &Path) -> Result<Vec<PathBuf>, String> {
    if target.is_file() {
        if target.extension().and_then(|e| e.to_str()) != Some("ch") {
            return Err(format!(
                "`{}` is not a .ch file: `chelis test` only accepts Chelis source",
                target.display()
            ));
        }
        return Ok(vec![target.to_path_buf()]);
    }
    let mut files = Vec::new();
    let walker = walkdir::WalkDir::new(target)
        .sort_by_file_name()
        .into_iter()
        .filter_entry(|entry| {
            // Skip dot-prefixed files AND directories (editor temp files, build
            // dirs like `.git` or `target/.rustc_info.json`, hidden fixtures).
            // Also skip `target/` directories which accumulate build artifacts
            // and have bitten us in red-team testing.
            let name = entry.file_name().to_string_lossy();
            if name.starts_with('.') {
                return false;
            }
            if entry.file_type().is_dir() && name == "target" {
                return false;
            }
            true
        });
    for entry in walker {
        let entry = entry.map_err(|e| format!("failed to walk {}: {e}", target.display()))?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("ch") {
            continue;
        }
        files.push(path.to_path_buf());
    }
    Ok(files)
}

/// Render the signal that killed a worker (e.g. "SIGABRT (6)").
/// On unix we read it from `ExitStatusExt::signal()`; on other platforms we
/// fall back to "<unknown>" because no signal concept exists.
#[cfg(unix)]
fn worker_signal_str(status: &std::process::ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;
    match status.signal() {
        Some(n) => {
            let name = match n {
                1 => "SIGHUP",
                2 => "SIGINT",
                3 => "SIGQUIT",
                4 => "SIGILL",
                6 => "SIGABRT",
                7 => "SIGBUS",
                8 => "SIGFPE",
                9 => "SIGKILL",
                11 => "SIGSEGV",
                13 => "SIGPIPE",
                14 => "SIGALRM",
                15 => "SIGTERM",
                _ => "signal",
            };
            format!("{name} ({n})")
        }
        None => "<unknown>".to_string(),
    }
}

#[cfg(not(unix))]
fn worker_signal_str(_status: &std::process::ExitStatus) -> String {
    "<unknown>".to_string()
}

/// Marker prepended to the `message` of a synthetic `<file>` row when the
/// user invoked `chelis test --filter <substring>`.
///
/// Issue #44 design call (Option 2 — "emit but tag"): a worker that crashes
/// before any per-test row streams (parse error, top-level type error,
/// stack-overflow before the first test, signal kill) yields a synthetic
/// `<file>` FAIL row attributed to that file. Under `--filter`, that row is
/// **not** filter-matched — the failure is at the file level and we never got
/// far enough to know whether the file contained a matching test name. We
/// could:
///   * Option 1: suppress the row entirely under `--filter`. Rejected: silent
///     data loss is the dominant bug pattern in this repo, and "the file you
///     filtered for happens to live in a file that won't parse" is exactly
///     the case where a silent suppression would mislead the operator.
///   * Option 2 (chosen): always emit the row but visually tag it so a user
///     scanning filtered output can see at a glance "this row isn't filtered;
///     it's a file-level failure". JSON consumers see the same prefix in the
///     `message` field, so machine readers stay parseable.
///   * Option 3 / 4 rejected as noted in the design notes — pre-enumerating
///     by path heuristic or grepping for `def <name>` either gives the worst
///     of both worlds or false-positives on commented code.
///
/// The marker text is part of the contract: regression tests assert on it.
const FILTER_INACTIVE_MARKER: &str = "(filter inactive. File-level error) ";

/// Tag the message of a file-level synthetic row to make it clear, under
/// `--filter`, that the row isn't filter-matched. Idempotent: re-tagging an
/// already-tagged row is a no-op so the parent never double-prefixes a row
/// the worker already emitted with the marker.
fn tag_filter_inactive(row: &mut TestRow) {
    if row.test != "<file>" {
        return;
    }
    let existing = row.message.take().unwrap_or_default();
    if existing.starts_with(FILTER_INACTIVE_MARKER) {
        row.message = Some(existing);
        return;
    }
    row.message = Some(format!("{FILTER_INACTIVE_MARKER}{existing}"));
}

struct TestWorkerOutput {
    output: std::process::Output,
    timed_out: bool,
    leader_signaled: bool,
}

fn run_batch_worker_command_with_timeout<F>(
    mut cmd: std::process::Command,
    timeout: Duration,
    mut on_stdout_line: F,
) -> Result<TestWorkerOutput, std::io::Error>
where
    F: FnMut(&[u8]),
{
    cmd.stdout(std::process::Stdio::piped())
        // Batch stderr is operator output, not protocol. Inherit the suite
        // worker's stderr so bytes flow directly to the public supervisor
        // without buffering, UTF-8 conversion, or progress multiplexing.
        .stderr(std::process::Stdio::inherit());
    let mut child = cmd.spawn()?;
    let child_stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("batch stdout pipe missing"))?;
    let (line_tx, line_rx) = std::sync::mpsc::channel::<Vec<u8>>();
    let stdout_reader = thread::spawn(move || {
        let mut reader = io::BufReader::new(child_stdout);
        let mut all = Vec::new();
        loop {
            let mut line = Vec::new();
            let read = reader.read_until(b'\n', &mut line)?;
            if read == 0 {
                break;
            }
            all.extend_from_slice(&line);
            if line_tx.send(line).is_err() {
                break;
            }
        }
        Ok::<_, std::io::Error>(all)
    });
    let deadline = Instant::now().checked_add(timeout);
    let mut timed_out = false;
    loop {
        while let Ok(line) = line_rx.try_recv() {
            on_stdout_line(&line);
        }
        if child.try_wait()?.is_some() {
            break;
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            timed_out = true;
            terminate_worker_process(&mut child)?;
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let status = child.wait()?;
    let stdout = stdout_reader
        .join()
        .map_err(|_| std::io::Error::other("batch stdout reader panicked"))??;
    while let Ok(line) = line_rx.try_recv() {
        on_stdout_line(&line);
    }
    Ok(TestWorkerOutput {
        output: std::process::Output {
            status,
            stdout,
            stderr: Vec::new(),
        },
        timed_out,
        leader_signaled: false,
    })
}

fn run_worker_command_with_timeout(
    cmd: std::process::Command,
    timeout: Duration,
) -> Result<TestWorkerOutput, std::io::Error> {
    run_command_with_timeout(cmd, timeout)
}

#[cfg(unix)]
#[allow(clippy::too_many_arguments)]
fn run_forked_test_suite(
    path: Option<&Path>,
    filter: Option<&str>,
    json: bool,
    timeout_secs: u64,
    jobs: TestJobs,
    batch_mode: TestBatchMode,
    expect: Option<ExpectArg>,
    progress_path: &Path,
    timeout: Duration,
) -> Result<TestWorkerOutput, std::io::Error> {
    use std::os::fd::AsRawFd;

    if testing_hook_enabled("CHELIS_TEST_FORCE_SUITE_FORK_FAILURE") {
        return Err(std::io::Error::other("forced suite fork failure"));
    }
    let (stdout_read, stdout_write) = create_cloexec_pipe().map_err(std::io::Error::other)?;
    let (stderr_read, stderr_write) = create_cloexec_pipe().map_err(std::io::Error::other)?;
    let (supervisor_read, supervisor_write) =
        create_cloexec_pipe().map_err(std::io::Error::other)?;
    let progress_path = progress_path.to_path_buf();

    // This is deliberately the first process-lifecycle operation: no reader
    // or watchdog threads exist until after fork, so the child does not inherit
    // a multithreaded runtime and never needs an externally addressable worker
    // subcommand.
    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(std::io::Error::last_os_error());
    }
    if pid == 0 {
        drop(stdout_read);
        drop(stderr_read);
        drop(supervisor_write);
        let stdout_fd = stdout_write.as_raw_fd();
        let stderr_fd = stderr_write.as_raw_fd();
        let setup_ok = unsafe {
            libc::setpgid(0, 0) == 0
                && libc::dup2(stdout_fd, libc::STDOUT_FILENO) >= 0
                && libc::dup2(stderr_fd, libc::STDERR_FILENO) >= 0
        };
        drop(stdout_write);
        drop(stderr_write);
        if !setup_ok {
            let _ = fs::remove_file(&progress_path);
            unsafe {
                libc::_exit(2);
            }
        }

        start_forked_suite_parent_watchdog(supervisor_read, progress_path.clone());
        ignore_test_suite_sigterm_if_requested();
        write_test_progress_rows_if_requested(&progress_path);
        hang_test_suite_if_requested("CHELIS_TEST_HANG_BEFORE_SUITE");
        let suite_result =
            emit_finalized_test_suite_if_requested(json, expect).unwrap_or_else(|| {
                cmd_test(
                    path,
                    filter,
                    json,
                    timeout_secs,
                    jobs,
                    batch_mode,
                    expect,
                    Some(&progress_path),
                )
            });
        let code = match suite_result {
            Ok(code) => code,
            Err(err) => {
                eprintln!("error: {err}");
                2
            }
        };
        hang_test_suite_if_requested("CHELIS_TEST_HANG_AFTER_SUITE");
        let _ = io::stdout().flush();
        let _ = io::stderr().flush();
        unsafe {
            libc::_exit(code);
        }
    }

    drop(stdout_write);
    drop(stderr_write);
    drop(supervisor_read);
    let output = run_forked_suite_pid_with_timeout(pid, stdout_read, stderr_read, timeout);
    // Keep this write end live until the child has been reaped and its output
    // collected. If this public supervisor is killed, kernel closure wakes the
    // child watchdog, which unlinks progress before killing the process group.
    drop(supervisor_write);
    output
}

#[cfg(not(unix))]
#[allow(clippy::too_many_arguments)]
fn run_forked_test_suite(
    _path: Option<&Path>,
    _filter: Option<&str>,
    _json: bool,
    _timeout_secs: u64,
    _jobs: TestJobs,
    _batch_mode: TestBatchMode,
    _expect: Option<ExpectArg>,
    _progress_path: &Path,
    _timeout: Duration,
) -> Result<TestWorkerOutput, std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "whole-suite descendant cleanup requires Unix process-group semantics",
    ))
}

#[cfg(unix)]
fn run_forked_suite_pid_with_timeout(
    pid: libc::pid_t,
    stdout_read: std::os::fd::OwnedFd,
    stderr_read: std::os::fd::OwnedFd,
    timeout: Duration,
) -> Result<TestWorkerOutput, std::io::Error> {
    use std::os::unix::process::ExitStatusExt;

    let mut child_stdout = std::fs::File::from(stdout_read);
    let mut child_stderr = std::fs::File::from(stderr_read);
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        child_stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        child_stderr.read_to_end(&mut bytes).map(|_| bytes)
    });
    let started = Instant::now();
    let deadline = started.checked_add(timeout);
    let suite_term_at = deadline.map(|deadline| {
        deadline
            .checked_sub(Duration::from_millis(200))
            .unwrap_or(started)
    });
    let mut timed_out = false;
    let mut suite_term_sent = false;
    let wait_status = loop {
        if let Some(status) = waitpid_nonblocking(pid)? {
            // A leader may die while a worker still owns the captured pipes.
            // Always quiesce its process group before joining reader threads;
            // a legitimate leader has already reaped its workers, making this
            // an ESRCH no-op.
            send_suite_pid_signal(pid, libc::SIGKILL)?;
            break status;
        }
        let now = Instant::now();
        if !suite_term_sent && suite_term_at.is_some_and(|term_at| now >= term_at) {
            timed_out = true;
            suite_term_sent = true;
            send_suite_pid_signal(pid, libc::SIGTERM)?;
        }
        if deadline.is_some_and(|deadline| now >= deadline) {
            timed_out = true;
            send_suite_pid_signal(pid, libc::SIGKILL)?;
            break waitpid_blocking(pid)?;
        }
        thread::sleep(Duration::from_millis(20));
    };
    let stdout = stdout_reader
        .join()
        .map_err(|_| std::io::Error::other("child stdout reader panicked"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| std::io::Error::other("child stderr reader panicked"))??;
    Ok(TestWorkerOutput {
        output: std::process::Output {
            status: std::process::ExitStatus::from_raw(wait_status),
            stdout,
            stderr,
        },
        timed_out,
        leader_signaled: libc::WIFSIGNALED(wait_status),
    })
}

#[cfg(unix)]
fn waitpid_nonblocking(pid: libc::pid_t) -> Result<Option<libc::c_int>, std::io::Error> {
    loop {
        let mut status = 0;
        match unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) } {
            0 => return Ok(None),
            value if value == pid => return Ok(Some(status)),
            -1 if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted => {
                continue;
            }
            -1 => return Err(std::io::Error::last_os_error()),
            _ => {
                return Err(std::io::Error::other(
                    "waitpid returned an unexpected child",
                ));
            }
        }
    }
}

#[cfg(unix)]
fn waitpid_blocking(pid: libc::pid_t) -> Result<libc::c_int, std::io::Error> {
    loop {
        let mut status = 0;
        match unsafe { libc::waitpid(pid, &mut status, 0) } {
            value if value == pid => return Ok(status),
            -1 if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted => {
                continue;
            }
            -1 => return Err(std::io::Error::last_os_error()),
            _ => {
                return Err(std::io::Error::other(
                    "waitpid returned an unexpected child",
                ));
            }
        }
    }
}

fn run_command_with_timeout(
    mut cmd: std::process::Command,
    timeout: Duration,
) -> Result<TestWorkerOutput, std::io::Error> {
    cmd.stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut child = cmd.spawn()?;
    let mut child_stdout = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("child stdout pipe missing"))?;
    let mut child_stderr = child
        .stderr
        .take()
        .ok_or_else(|| std::io::Error::other("child stderr pipe missing"))?;
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        child_stdout.read_to_end(&mut bytes).map(|_| bytes)
    });
    let stderr_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        child_stderr.read_to_end(&mut bytes).map(|_| bytes)
    });
    let deadline = Instant::now().checked_add(timeout);
    let mut timed_out = false;
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            timed_out = true;
            terminate_worker_process(&mut child)?;
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let status = child.wait()?;
    let stdout = stdout_reader
        .join()
        .map_err(|_| std::io::Error::other("child stdout reader panicked"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| std::io::Error::other("child stderr reader panicked"))??;
    let output = std::process::Output {
        status,
        stdout,
        stderr,
    };
    Ok(TestWorkerOutput {
        output,
        timed_out,
        leader_signaled: false,
    })
}

fn terminate_worker_process(child: &mut std::process::Child) -> Result<(), std::io::Error> {
    send_worker_sigterm(child)?;
    let grace_deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        if Instant::now() >= grace_deadline {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    child.kill()
}

#[cfg(unix)]
fn send_suite_pid_signal(pid: libc::pid_t, signal: libc::c_int) -> Result<(), std::io::Error> {
    let rc = unsafe { libc::kill(-pid, signal) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::ESRCH) {
            return Err(err);
        }
    }
    Ok(())
}

#[cfg(unix)]
fn send_worker_sigterm(child: &mut std::process::Child) -> Result<(), std::io::Error> {
    let rc = unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(unix))]
fn send_worker_sigterm(child: &mut std::process::Child) -> Result<(), std::io::Error> {
    child.kill()
}

fn worker_process_timeout(
    file: &Path,
    filter: Option<&str>,
    rel_display: &str,
    timeout_secs: u64,
) -> Duration {
    let selected_count = estimate_selected_test_count(file, filter, rel_display).max(1);
    let per_test = timeout_secs.max(1);
    let test_budget = per_test
        .saturating_mul(selected_count.saturating_add(1) as u64)
        .saturating_add(10);
    Duration::from_secs(test_budget.max(60))
}

fn estimate_selected_test_count(file: &Path, filter: Option<&str>, rel_display: &str) -> usize {
    let Ok(source) = fs::read_to_string(file) else {
        return 1;
    };
    let Ok(parsed) = chelis_surf::parser::parse_str(&source) else {
        return 1;
    };
    let flat = flatten_module_decls(&parsed);
    match enumerate_test_fns(&flat, filter, rel_display) {
        EnumerationOutcome::Tests(tests) => tests.len(),
        EnumerationOutcome::Error(_) => 1,
    }
}

/// Spawn `chelis __test_file <file> --rel-display ... --filter ... --timeout N`
/// as a subprocess. Capture its NDJSON stdout and parse into TestRows. A
/// child crash (stack overflow, panic in the evaluator) only kills the child;
/// the parent attributes the loss as a file-level worker crash and moves on.
#[derive(Clone, Copy)]
struct TestFileWorkerOptions<'a> {
    compiled_context_path: Option<&'a Path>,
    expect_file_diagnostic: bool,
}

fn run_test_file_subprocess(
    self_path: &Path,
    cwd: &Path,
    file: &Path,
    rel_display: &str,
    filter: Option<&str>,
    timeout_secs: u64,
    worker_options: TestFileWorkerOptions<'_>,
) -> Vec<TestRow> {
    let mut cmd = std::process::Command::new(self_path);
    cmd.arg("__test_file")
        .arg(file)
        .arg("--rel-display")
        .arg(rel_display)
        .arg("--timeout")
        .arg(timeout_secs.to_string())
        .current_dir(cwd);
    if worker_options.expect_file_diagnostic {
        cmd.arg("--expect-file-diagnostic");
    }
    if let Some(path) = worker_options.compiled_context_path {
        // Phase H: hand the bincode-encoded `CompiledContext` to the
        // worker via env var so the worker can deserialize the library
        // snapshot instead of re-running `prepare_reef_graph` per file.
        // Absent on packages whose deps are LocalRegistry-resolved
        // (`compile_reef_context` cannot hash those yet); the worker's
        // own fallback then runs the legacy reef-graph path.
        cmd.env("CHELIS_TEST_COMPILED_CONTEXT", path);
    } else {
        // Belt-and-suspenders: never let an inherited env var from the
        // outer environment shadow our "no context available" decision.
        // If the parent could not build a context, the worker MUST take
        // the legacy path on its own.
        cmd.env_remove("CHELIS_TEST_COMPILED_CONTEXT");
    }
    if let Some(needle) = filter {
        cmd.arg("--filter").arg(needle);
    }
    // Inherit CHELIS_REEF_HOME and PATH; the child needs the same reef
    // registry the parent was configured with.
    let worker_timeout = worker_process_timeout(file, filter, rel_display, timeout_secs);
    let output = match run_worker_command_with_timeout(cmd, worker_timeout) {
        Ok(o) => o,
        Err(err) => {
            return vec![TestRow {
                file: rel_display.to_string(),
                test: "<file>".to_string(),
                status: TestStatus::Fail,
                message: Some(format!("could not spawn test worker: {err}")),
            }];
        }
    };

    // Parse NDJSON rows from stdout. The child emits one `{file,test,status,message?}`
    // record per line. A malformed or empty line is ignored; a completely empty
    // stdout combined with a non-zero exit means the child crashed before
    // running anything.
    let stdout = String::from_utf8_lossy(&output.output.stdout);
    let mut rows = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(test) = value.get("test").and_then(|v| v.as_str()) else {
            continue;
        };
        let Some(status_s) = value.get("status").and_then(|v| v.as_str()) else {
            continue;
        };
        let status = match status_s {
            "pass" => TestStatus::Pass,
            _ => TestStatus::Fail,
        };
        let message = value
            .get("message")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let file_str = value
            .get("file")
            .and_then(|v| v.as_str())
            .unwrap_or(rel_display)
            .to_string();
        rows.push(TestRow {
            file: file_str,
            test: test.to_string(),
            status,
            message,
        });
    }

    // Child exited abnormally. Emit a synthetic file-level crash row when:
    //   * killed by signal (exit code is None) — the worker died mid-file
    //     and any tests after the last streamed row are lost. This case
    //     covers RT-A3 D1 (streaming preserves the rows up to the crash;
    //     this row covers the rest).
    //   * exit > 1 — the worker hit an internal error path, not a normal
    //     test-failure exit.
    //   * rows is empty AND exit != 0 — even with a "normal" exit 1, no
    //     rows means the worker emitted nothing for some reason; surface
    //     the failure so it does not silently disappear.
    // Exit 1 with at least one row is the normal "some test failed" exit
    // and does not warrant a synthetic row.
    if output.timed_out {
        rows.push(TestRow {
            file: rel_display.to_string(),
            test: "<file>".to_string(),
            status: TestStatus::Fail,
            message: Some(format!(
                "worker timeout after {}s; sent SIGTERM, then SIGKILL after 5s if needed",
                worker_timeout.as_secs()
            )),
        });
    }

    let worker_crashed = !output.timed_out
        && match output.output.status.code() {
            None => true,
            Some(0) => false,
            Some(1) => rows.is_empty(),
            Some(_) => true,
        };
    if worker_crashed {
        let stderr = String::from_utf8_lossy(&output.output.stderr);
        let msg = if let Some(code) = output.output.status.code() {
            format!("worker exited {code}: {}", stderr.trim())
        } else {
            let signal_str = worker_signal_str(&output.output.status);
            format!("worker killed by signal {signal_str}: {}", stderr.trim())
        };
        rows.push(TestRow {
            file: rel_display.to_string(),
            test: "<file>".to_string(),
            status: TestStatus::Fail,
            message: Some(msg),
        });
    }

    // Issue #44: under `--filter`, prepend the "filter inactive" marker to
    // every synthetic `<file>` row so the operator can tell at a glance that
    // the row is a file-level error rather than a filter match. We tag here
    // so the rule covers BOTH worker-emitted file-level rows (parse error,
    // file-level compile error) and parent-synthesized worker-crash rows.
    if filter.is_some() {
        for row in &mut rows {
            tag_filter_inactive(row);
        }
    }

    rows
}

/// Hidden subcommand body: run every test in a single file and emit
/// per-test NDJSON on stdout. Called by the parent `chelis test` via
/// `run_test_file_subprocess` for per-file crash isolation.
fn cmd_internal_test_file(
    file: &Path,
    rel_display: &str,
    filter: Option<&str>,
    timeout: Duration,
    expect_file_diagnostic: bool,
) -> Result<i32, String> {
    // Hidden testing knob — gates the regression test for per-file
    // subprocess isolation in `crates/chelis-cli/tests/subprocess_isolation.rs`.
    // Both env vars must be set together so a production user cannot trip
    // this by accident with a single stray variable. The value of
    // CHELIS_TEST_FORCE_ABORT is treated as a substring filter on the
    // worker's `--rel-display`: only files whose displayed path contains
    // that substring abort. This lets a regression test crash one file
    // and let the sibling worker run normally even though both inherit
    // the same env. Aborts via std::process::abort() to mimic a hard
    // crash (stack overflow, SIGABRT, etc.).
    if env::var("CHELIS_TEST_INTERNAL_TESTING").as_deref() == Ok("1")
        && let Ok(needle) = env::var("CHELIS_TEST_FORCE_ABORT")
        && !needle.is_empty()
        && rel_display.contains(&needle)
    {
        std::process::abort();
    }

    let exec_context = load_test_execution_context()?;

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut failed = 0usize;
    let mut io_err: Option<String> = None;
    // RT-A3 D1 streaming-regression hatch: when this env var is set (and
    // CHELIS_TEST_INTERNAL_TESTING=1), the worker aborts immediately after
    // emitting the row for the test whose name contains the configured
    // substring, simulating a stack-overflow / abort partway through a
    // file. The streaming write+flush below means earlier rows survive.
    let abort_after_test_substring =
        if env::var("CHELIS_TEST_INTERNAL_TESTING").as_deref() == Ok("1") {
            env::var("CHELIS_TEST_ABORT_AFTER_TEST").ok()
        } else {
            None
        };
    // RT-A3 D1: write+flush per row so a mid-file worker crash preserves
    // the rows emitted before it. Previously the worker buffered every row
    // in a Vec and printed all of them on exit, so a stack-overflow on test
    // #90 silently dropped the 89 prior PASS rows.
    let file_result = run_test_file(
        &exec_context,
        file,
        filter,
        rel_display,
        timeout,
        expect_file_diagnostic,
        |row| {
            if io_err.is_some() {
                return;
            }
            if let Err(e) = writeln!(out, "{}", row.to_json()) {
                io_err = Some(e.to_string());
                return;
            }
            if let Err(e) = out.flush() {
                io_err = Some(e.to_string());
                return;
            }
            if row.status == TestStatus::Fail {
                failed += 1;
            }
            if let Some(needle) = abort_after_test_substring.as_deref()
                && !needle.is_empty()
                && row.test.contains(needle)
            {
                std::process::abort();
            }
        },
    );
    if let Some(e) = io_err {
        return Err(e);
    }
    if let Err(err) = file_result {
        // Read/parse failures surface as one synthetic file-level row, the
        // same shape the parent's worker-crash path emits, so the renderer
        // does not need a special case.
        let row = TestRow {
            file: rel_display.to_string(),
            test: "<file>".to_string(),
            status: TestStatus::Fail,
            message: Some(err),
        };
        writeln!(out, "{}", row.to_json()).map_err(|e| e.to_string())?;
        out.flush().map_err(|e| e.to_string())?;
        failed += 1;
    }
    Ok(if failed == 0 { 0 } else { 1 })
}

fn load_test_execution_context() -> Result<TestExecutionContext, String> {
    // Phase H: when the parent populates `CHELIS_TEST_COMPILED_CONTEXT`,
    // load the bincode-encoded `CompiledContext` from the path it points
    // at. Workers used to run `prepare_reef_graph` per file, paying the
    // chelis-std re-check + re-lower cost N times per `chelis test`
    // invocation; now the parent runs that pipeline ONCE and hands the
    // result through. If the env var is missing (e.g., the worker is
    // invoked directly without going through `chelis test`), we fall
    // back to the reef-graph path so the worker still works standalone.
    let compiled_context_env = env::var("CHELIS_TEST_COMPILED_CONTEXT").ok();
    match compiled_context_env.as_deref() {
        Some(path) if !path.is_empty() => {
            let bytes = fs::read(path)
                .map_err(|e| format!("read CHELIS_TEST_COMPILED_CONTEXT tempfile `{path}`: {e}"))?;
            let ctx = chelis_compiler_api::CompiledContext::decode(&bytes)
                .map_err(|e| format!("decode CHELIS_TEST_COMPILED_CONTEXT: {e}"))?;
            let cwd = env::current_dir().map_err(|e| format!("failed to read cwd: {e}"))?;
            let ctx_root = ctx
                .reef_state()
                .package_root
                .canonicalize()
                .unwrap_or_else(|_| ctx.reef_state().package_root.clone());
            let cwd_canon = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
            if ctx_root != cwd_canon {
                return Err(format!(
                    "CHELIS_TEST_COMPILED_CONTEXT package_root `{}` does not match worker cwd `{}`; \
                     refusing to run tests with a mismatched library context",
                    ctx_root.display(),
                    cwd_canon.display()
                ));
            }
            Ok(TestExecutionContext::Context(Box::new(ctx)))
        }
        _ => {
            let cwd = env::current_dir().map_err(|e| format!("failed to read cwd: {e}"))?;
            let graph = chelis_reef::prepare_reef_graph(&cwd)?;
            Ok(TestExecutionContext::ReefGraph(Box::new(graph)))
        }
    }
}

fn cmd_internal_test_batch(manifest_path: &Path, timeout: Duration) -> Result<i32, String> {
    if env::var("CHELIS_TEST_INTERNAL_TESTING").as_deref() == Ok("1") {
        let stderr = io::stderr();
        let mut err = stderr.lock();
        if let Ok(message) = env::var("CHELIS_TEST_BATCH_STDERR") {
            err.write_all(message.as_bytes())
                .and_then(|_| err.write_all(b"\n"))
                .and_then(|_| err.flush())
                .map_err(|e| format!("write batch stderr test hook: {e}"))?;
        }
        if let Ok(path) = env::var("CHELIS_TEST_BATCH_STDERR_FILE") {
            let bytes = fs::read(&path)
                .map_err(|e| format!("read batch stderr test hook file `{path}`: {e}"))?;
            err.write_all(&bytes)
                .and_then(|_| err.flush())
                .map_err(|e| format!("write batch stderr test hook file: {e}"))?;
        }
    }
    if env::var("CHELIS_TEST_INTERNAL_TESTING").as_deref() == Ok("1")
        && env::var("CHELIS_TEST_FORCE_BATCH_ABORT").as_deref() == Ok("1")
    {
        std::process::abort();
    }

    let manifest_text = fs::read_to_string(manifest_path).map_err(|e| {
        format!(
            "read test-batch manifest `{}`: {e}",
            manifest_path.display()
        )
    })?;
    let manifest: TestBatchManifest = serde_json::from_str(&manifest_text).map_err(|e| {
        format!(
            "parse test-batch manifest `{}`: {e}",
            manifest_path.display()
        )
    })?;
    if env::var("CHELIS_TEST_INTERNAL_TESTING").as_deref() == Ok("1")
        && let Ok(needle) = env::var("CHELIS_TEST_FORCE_ABORT")
        && !needle.is_empty()
        && manifest
            .files
            .iter()
            .any(|file| file.rel_display.contains(&needle))
    {
        std::process::abort();
    }
    let exec_context = load_test_execution_context()?;

    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut failed = 0usize;
    let mut io_err: Option<String> = None;
    let abort_after_test_substring =
        if env::var("CHELIS_TEST_INTERNAL_TESTING").as_deref() == Ok("1") {
            env::var("CHELIS_TEST_ABORT_AFTER_TEST").ok()
        } else {
            None
        };
    run_test_batch(&exec_context, &manifest.files, timeout, |row| {
        if io_err.is_some() {
            return;
        }
        if let Err(e) = writeln!(out, "{}", row.to_json()) {
            io_err = Some(e.to_string());
            return;
        }
        if let Err(e) = out.flush() {
            io_err = Some(e.to_string());
            return;
        }
        if row.status == TestStatus::Fail {
            failed += 1;
        }
        if let Some(needle) = abort_after_test_substring.as_deref()
            && !needle.is_empty()
            && row.test.contains(needle)
        {
            std::process::abort();
        }
    })?;
    if let Some(e) = io_err {
        return Err(e);
    }
    if testing_hook_enabled("CHELIS_TEST_HANG_AFTER_BATCH") {
        if let Ok(path) = env::var("CHELIS_TEST_HANG_PID_FILE") {
            fs::write(path, std::process::id().to_string())
                .map_err(|e| format!("write hung-batch pid file: {e}"))?;
        }
        loop {
            thread::park();
        }
    }
    Ok(if failed == 0 { 0 } else { 1 })
}

fn run_test_batch<F>(
    exec_context: &TestExecutionContext,
    files: &[TestBatchManifestFile],
    timeout: Duration,
    mut on_row: F,
) -> Result<(), String>
where
    F: FnMut(&TestRow),
{
    let mut combined_decls = Vec::new();
    let mut selected = Vec::<(String, String, chelis_deep::Span, String)>::new();
    let mut seen_names = HashSet::<String>::new();

    for file in files {
        let source = fs::read_to_string(&file.file)
            .map_err(|e| format!("read {}: {e}", file.file.display()))?;
        let parsed = chelis_surf::parser::parse_str(&source)
            .map_err(|e| format!("parse {}: {e}", file.file.display()))?;
        let flat = flatten_module_decls(&parsed);
        for name in top_level_decl_names(&flat) {
            if !seen_names.insert(name.clone()) {
                return Err(format!("duplicate top-level name `{name}` in test batch"));
            }
        }

        let tests = match enumerate_test_fns(&flat, None, &file.rel_display) {
            EnumerationOutcome::Tests(tests) => tests,
            EnumerationOutcome::Error(msg) => return Err(msg),
        };
        for test_name in &file.tests {
            let Some(test) = tests.iter().find(|candidate| candidate.name == *test_name) else {
                return Err(format!(
                    "selected test `{test_name}` was not found in {}",
                    file.rel_display
                ));
            };
            let synth_name = format!("__chelis_test_f{}_t{}", file.index, selected.len());
            selected.push((
                file.rel_display.clone(),
                test.name.clone(),
                test.span,
                synth_name,
            ));
        }
        combined_decls.extend(flat);
    }

    for (_, test_name, span, synth_name) in &selected {
        let call = chelis_surf::ast::Expr::Apply(
            Box::new(chelis_surf::ast::Expr::Var(test_name.clone(), *span)),
            Vec::new(),
            *span,
        );
        combined_decls.push(Decl::LetDef {
            name: synth_name.clone(),
            ty: None,
            value: call,
            span: *span,
        });
    }

    let prepared_eval = prepare_eval_in_exec_context(exec_context, &combined_decls)?;

    for (rel_display, test_name, _, synth_name) in selected {
        let root = synth_name.clone();
        let handle = prepared_eval.clone();
        let outcome = run_test_with_timeout(
            move || Ok(handle.eval_root(BTreeMap::new(), &root)),
            timeout,
            &format!("timeout after {}s", timeout.as_secs()),
        );

        let (status, message) = match outcome {
            Err(msg) => (TestStatus::Fail, Some(msg)),
            Ok(result) => match result {
                Ok(_) => (TestStatus::Pass, None),
                Err(err) => {
                    let message = err
                        .errors
                        .iter()
                        .map(|d| d.message.clone())
                        .collect::<Vec<_>>()
                        .join("; ");
                    (TestStatus::Fail, Some(message))
                }
            },
        };

        on_row(&TestRow {
            file: rel_display,
            test: test_name,
            status,
            message,
        });
    }

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestStatus {
    Pass,
    Fail,
}

impl TestStatus {
    fn label(&self) -> &'static str {
        match self {
            TestStatus::Pass => "PASS",
            TestStatus::Fail => "FAIL",
        }
    }
    fn json_label(&self) -> &'static str {
        match self {
            TestStatus::Pass => "pass",
            TestStatus::Fail => "fail",
        }
    }
}

#[derive(Debug, Clone)]
struct TestRow {
    file: String,
    test: String,
    status: TestStatus,
    message: Option<String>,
}

impl TestRow {
    fn render_plain(&self) -> String {
        // Right-pad the test name with dots so the status column lines up,
        // matching the plan's worked example (`test_name ......... PASS`).
        const LEADER_WIDTH: usize = 32;
        let test_len = self.test.chars().count();
        let dots = if test_len + 2 >= LEADER_WIDTH {
            " ".to_string()
        } else {
            " ".to_string() + &".".repeat(LEADER_WIDTH - test_len - 2) + " "
        };
        let msg = match (&self.message, self.status) {
            (Some(m), TestStatus::Fail) => format!(" ({m})"),
            _ => String::new(),
        };
        format!("{}{dots}{}{msg}", self.test, self.status.label())
    }

    fn to_json(&self) -> String {
        let mut out = format!(
            "{{\"file\":{},\"test\":{},\"status\":\"{}\"",
            json_string(&self.file),
            json_string(&self.test),
            self.status.json_label()
        );
        if let Some(msg) = &self.message {
            out.push_str(",\"message\":");
            out.push_str(&json_string(msg));
        }
        out.push('}');
        out
    }
}

fn json_string(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}

/// Per-worker library-side compile state. The Phase H refactor introduced
/// the `Context` variant: the parent compiles the reef graph + library
/// once into a `CompiledContext`, encodes it to a tempfile, and hands
/// the path to each worker via `CHELIS_TEST_COMPILED_CONTEXT`. Workers
/// rehydrate the context and reuse its embedded [`PreparedReefGraph`]
/// without paying for their own `prepare_reef_graph` walk.
///
/// `ReefGraph` is the legacy fallback used when the env var is absent
/// (worker invoked outside of `chelis test`, e.g., by hand for
/// debugging). Behavior parity with the pre-Phase-H path is preserved
/// end-to-end so a missing env var is not a silent regression.
///
/// Both variants currently route per-test eval through
/// `compile_with_reef_graph` + `prepare_eval`. The
/// `eval_in_context` / `prepare_eval_in_context` Phase G/G' APIs are
/// available but, on the chelis-std test corpus, do not yet match the
/// monolithic evaluator's linearity-checker semantics nor stay within
/// host-runtime stack budget on the heaviest tensor-reduce tests; the
/// integration is a follow-up phase. The Phase H acceptance probe is
/// unchanged behavior + the parent-serializes + worker-deserializes
/// bridge being load-bearing across every spawned worker.
enum TestExecutionContext {
    // Both `CompiledContext` and `PreparedReefGraph` are large structs
    // (TypeEnv + reef state + DAG carrier; the reef graph carries the
    // linked decl lists plus the chelis-std / non-chelis-std partition).
    // Box both arms so the enum's stack footprint stays compact
    // regardless of which one is active.
    Context(Box<chelis_compiler_api::CompiledContext>),
    ReefGraph(Box<chelis_reef::PreparedReefGraph>),
}

impl TestExecutionContext {
    /// Read-only borrow of the underlying reef graph. Currently only
    /// surfaced through the in-context fast paths via the explicit
    /// match arms; kept here for any caller that needs the raw graph
    /// (e.g. a future direct compile pass) without cloning the
    /// surrounding context.
    #[allow(dead_code)]
    fn reef_graph(&self) -> &chelis_reef::PreparedReefGraph {
        match self {
            TestExecutionContext::Context(ctx) => ctx.reef_state(),
            TestExecutionContext::ReefGraph(g) => g,
        }
    }
}

/// Execute every `test_*` function in `file` and return one `TestRow` per
/// selected test. Returns `Err` only when the file itself cannot be read
/// or parsed — compile/runtime failures surface as per-row `FAIL` entries.
fn run_test_file<F>(
    exec_context: &TestExecutionContext,
    file: &Path,
    filter: Option<&str>,
    rel_display: &str,
    timeout: Duration,
    expect_file_diagnostic: bool,
    mut on_row: F,
) -> Result<(), String>
where
    F: FnMut(&TestRow),
{
    let source = fs::read_to_string(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    let parsed = chelis_surf::parser::parse_str(&source)
        .map_err(|e| format!("parse {}: {e}", file.display()))?;

    // A test file may wrap its contents in `module Foo.Bar` — we need the
    // flat decl list so the lower stages treat it as an eval module.
    // Imports are preserved so the reef rewriter (whether monolithic or
    // in-context) can resolve references to `Std.*` or sibling modules.
    let flat_decls = flatten_module_decls(&parsed);

    let matched_tests = match enumerate_test_fns(&flat_decls, filter, rel_display) {
        EnumerationOutcome::Tests(tests) => tests,
        EnumerationOutcome::Error(msg) => {
            on_row(&TestRow {
                file: rel_display.to_string(),
                test: "<file>".to_string(),
                status: TestStatus::Fail,
                message: Some(msg),
            });
            return Ok(());
        }
    };

    if matched_tests.is_empty() && !expect_file_diagnostic {
        // Preserve the legacy ordinary-test contract: a testless file is a
        // zero-record file and does not pay compile/check preparation. Only
        // `--expect` opts into treating a bare file diagnostic as the probe
        // outcome (chelis#967).
        return Ok(());
    }

    // File-level compile pre-check (RT3 H2). If the whole module doesn't
    // type-check, emit ONE file-level failure row instead of cascading the
    // same compile error across every discovered test.
    if let Err(compile_err) = compile_check_in_exec_context(exec_context, &flat_decls) {
        on_row(&TestRow {
            file: rel_display.to_string(),
            test: "<file>".to_string(),
            status: TestStatus::Fail,
            message: Some(format!("compile: {compile_err}")),
        });
        return Ok(());
    }

    if matched_tests.is_empty() {
        // Expected-failure files may intentionally contain no `test_*`
        // declaration because the file-level checker diagnostic *is* the
        // expected outcome. The cheap reef precheck above only resolves names
        // and module shape, so run the full check/lowering preparation before
        // deciding this is a clean, recordless file. A real failure becomes
        // the same synthetic `<file>` row consumed by the expected-failure
        // adapter; a genuinely clean file still emits no row and classifies as
        // config-error under `--expect` (chelis#967).
        if let Err(check_err) = prepare_eval_in_exec_context(exec_context, &flat_decls) {
            on_row(&TestRow {
                file: rel_display.to_string(),
                test: "<file>".to_string(),
                status: TestStatus::Fail,
                message: Some(format!("compile: {check_err}")),
            });
        }
        return Ok(());
    }

    // Module-init pre-check (RT3 H1). Evaluate the flat module alone — any
    // top-level `let _ = assert_*(...)` failure surfaces here as its own
    // `module-init` row and cascades every discovered test to FAIL. This
    // matches the plan's worked example output.
    if let Some(init_err) = eval_module_init(exec_context, &flat_decls, timeout) {
        on_row(&TestRow {
            file: rel_display.to_string(),
            test: "module-init".to_string(),
            status: TestStatus::Fail,
            message: Some(init_err),
        });
        for test in &matched_tests {
            on_row(&TestRow {
                file: rel_display.to_string(),
                test: test.name.clone(),
                status: TestStatus::Fail,
                message: Some("module-init failed".to_string()),
            });
        }
        return Ok(());
    }

    // Per-test isolation via filtered host-program eval. Synthesize ONE
    // `__chelis_test_<n> = test_<n>()` binding per discovered test and
    // compile the whole module ONCE; then run a per-test eval against the
    // shared handle. The single shared compile eliminates the
    // ~2.3s-per-test overhead per-test recompilation paid.
    //
    // Timeout semantics: each per-test eval runs under its own worker
    // thread with a per-test budget. An infinite-looping test fires its
    // own budget and is reported as timed-out without poisoning siblings.
    let synth_test_names: Vec<String> = (0..matched_tests.len())
        .map(|i| format!("__chelis_test_{i}"))
        .collect();
    let mut synth_decls = flat_decls.clone();
    for (test, synth_name) in matched_tests.iter().zip(synth_test_names.iter()) {
        let call = chelis_surf::ast::Expr::Apply(
            Box::new(chelis_surf::ast::Expr::Var(test.name.clone(), test.span)),
            Vec::new(),
            test.span,
        );
        synth_decls.push(Decl::LetDef {
            name: synth_name.clone(),
            ty: None,
            value: call,
            span: test.span,
        });
    }

    let prepared_eval = match prepare_eval_in_exec_context(exec_context, &synth_decls) {
        Ok(p) => p,
        Err(err) => {
            on_row(&TestRow {
                file: rel_display.to_string(),
                test: "<file>".to_string(),
                status: TestStatus::Fail,
                message: Some(format!("compile: {err}")),
            });
            return Ok(());
        }
    };

    for (test, synth_name) in matched_tests.iter().zip(synth_test_names.iter()) {
        let root = synth_name.clone();
        let handle = prepared_eval.clone();
        let outcome = run_test_with_timeout(
            move || Ok(handle.eval_root(BTreeMap::new(), &root)),
            timeout,
            &format!("timeout after {}s", timeout.as_secs()),
        );

        let (status, message) = match outcome {
            Err(msg) => (TestStatus::Fail, Some(msg)),
            Ok(result) => match result {
                Ok(_) => (TestStatus::Pass, None),
                Err(err) => {
                    let message = err
                        .errors
                        .iter()
                        .map(|d| d.message.clone())
                        .collect::<Vec<_>>()
                        .join("; ");
                    (TestStatus::Fail, Some(message))
                }
            },
        };

        on_row(&TestRow {
            file: rel_display.to_string(),
            test: test.name.clone(),
            status,
            message,
        });
    }
    Ok(())
}

/// File-level compile pre-check used by `run_test_file`. Routes
/// through `compile_with_reef_graph` regardless of which
/// `TestExecutionContext` variant is active — this stage is a name-
/// resolution + module-shape gate, NOT a full pipeline check. The
/// full type/effect/linearity check happens later inside
/// `prepare_eval_in_exec_context` (in-context path) or
/// `prepare_eval` (legacy path); pre-G' the file-level gate also
/// went through `compile_with_reef_graph`, so keeping it here
/// preserves behavior. Routing the file-level gate through the
/// stricter `check_in_context` would surface `_with_context`
/// linearity-rejections that the monolithic checker accepts (e.g.
/// chelis-std's `test_linspace_endpoints` two-call-on-same-arg
/// pattern), making the in-context path strictly worse than legacy.
fn compile_check_in_exec_context(
    exec_context: &TestExecutionContext,
    flat_decls: &[Decl],
) -> Result<(), String> {
    chelis_reef::compile_with_reef_graph(exec_context.reef_graph(), flat_decls)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Compile the synth_decls once and return a `PreparedEval` handle so
/// per-test evals share the compile. Routes through the legacy
/// `compile_with_reef_graph` + `prepare_eval` path for the
/// `ReefGraph` variant, and `prepare_eval_in_context` for the
/// `Context` variant.
///
/// Phase G' (final) — with the linearity divergence root-caused (the
/// monolithic `annotate_ir_program` was masking real
/// use-after-consume violations because it built with an empty
/// `AdtRegistry`; see `docs/archive/rca/lin_rca_report.md`) and chelis-std + the
/// CLI test fixtures rewritten to use `&t` / `copy(t)` at the right
/// sites, the `Context` arm now goes through
/// `prepare_eval_in_context(ctx, source)`. Per-file work drops from
/// "full pipeline on ~50 modules" to "parse + check + lower the test
/// file's ~10 lines." The `ReefGraph` arm stays on the legacy path
/// for direct workers that were not handed a compiled context.
#[derive(Clone)]
enum PreparedTestEval {
    Legacy(chelis_compiler_api::compiler::PreparedEval),
    InContext(chelis_compiler_api::compiler::PreparedEvalInContext),
}

impl PreparedTestEval {
    fn eval_root(
        &self,
        bindings: BTreeMap<String, chelis_compiler_api::schema::TensorValue>,
        root: &str,
    ) -> Result<chelis_compiler_api::schema::EvalResult, chelis_compiler_api::compiler::CompilerError>
    {
        match self {
            PreparedTestEval::Legacy(p) => p.eval_root(bindings, root),
            PreparedTestEval::InContext(p) => p.eval_root(bindings, root),
        }
    }
}

fn prepare_eval_in_exec_context(
    exec_context: &TestExecutionContext,
    synth_decls: &[Decl],
) -> Result<PreparedTestEval, String> {
    // RFC v5 (RT-1 F2 bypass): both branches feed reef-linked decls to
    // the checker -- the Context branch through `prepare_eval_in_context`
    // (which rewrites + checks internally) and the ReefGraph branch by
    // formatting linked decls to Surf text and re-evaluating via
    // `prepare_eval`, which loses the in-process provenance across the
    // text round-trip. Re-assert the linked flag for both so the
    // linker's internal names are accepted.
    let _linked = chelis_types::install_linked_program_guard();
    match exec_context {
        TestExecutionContext::Context(ctx) => {
            // `prepare_eval_in_context` runs the reef rewriter
            // (rewrite_entry_decls_with_reef_graph) and the
            // _with_context type/effect/linearity stages internally
            // against the cached library snapshot. Per-file work is
            // just the test file's ~10 decls, not the full reef
            // graph's ~50 modules. Avoid the legacy
            // compile_with_reef_graph call here — it would re-check
            // the entire library and defeat the cache.
            let source_text = chelis_surf::format::format_program(synth_decls);
            chelis_compiler_api::compiler::prepare_eval_in_context(ctx, &source_text)
                .map(PreparedTestEval::InContext)
                .map_err(|err| {
                    err.errors
                        .iter()
                        .map(|d| d.message.clone())
                        .collect::<Vec<_>>()
                        .join("; ")
                })
        }
        TestExecutionContext::ReefGraph(_) => {
            let prepared =
                chelis_reef::compile_with_reef_graph(exec_context.reef_graph(), synth_decls)
                    .map_err(|e| e.to_string())?;
            let source_text = chelis_surf::format::format_program(&prepared.decls);
            // `prepare_eval` is deprecated externally but retained for a
            // directly-invoked worker that was not handed a compiled context.
            #[allow(deprecated)]
            let prepared_eval = chelis_compiler_api::compiler::prepare_eval(EvalRequest {
                source_kind: SourceKind::Surf,
                source: source_text,
                bindings: BTreeMap::new(),
            });
            prepared_eval.map(PreparedTestEval::Legacy).map_err(|err| {
                err.errors
                    .iter()
                    .map(|d| d.message.clone())
                    .collect::<Vec<_>>()
                    .join("; ")
            })
        }
    }
}

#[derive(Debug, Clone)]
struct DiscoveredTest {
    name: String,
    span: chelis_deep::Span,
}

/// Return value from `enumerate_test_fns` — either the list of runnable tests,
/// or a fatal enumeration error (duplicate names) attributed to the file.
enum EnumerationOutcome {
    Tests(Vec<DiscoveredTest>),
    Error(String),
}

fn enumerate_test_fns(
    decls: &[Decl],
    filter: Option<&str>,
    rel_display: &str,
) -> EnumerationOutcome {
    let mut out = Vec::new();
    let mut seen: HashMap<String, bool> = HashMap::new();
    for decl in decls {
        let Decl::FunDef {
            name,
            params,
            ret_ty,
            span,
            ..
        } = decl
        else {
            continue;
        };
        // Tests must be genuinely nullary functions with a `test_<name>` prefix
        // (not `test_` alone) and must return unit — either implicitly (no
        // annotation), via `-> unit`, or via `-> _`. Rejecting non-unit return
        // types is what keeps typed-value bindings like `def test_x : int64 = 42`
        // from being mis-enumerated as zero-arg tests and cascading compile
        // errors across every other test in the same file (RT3 H4).
        if !name.starts_with("test_") || name == "test_" {
            continue;
        }
        if !params.is_empty() {
            continue;
        }
        if let Some(ty) = ret_ty
            && !is_unit_type(ty)
        {
            continue;
        }
        // Duplicate `def test_foo()` in the same file silently shadows in
        // Chelis; surface it as a fatal enumeration error so the operator
        // can fix the file instead of guessing which body ran (RT3 H3).
        if seen.insert(name.clone(), true).is_some() {
            return EnumerationOutcome::Error(format!(
                "duplicate test definition `{name}`. Each `def test_*()` in a test file must have a unique name"
            ));
        }
        let key = format!("{rel_display}::{name}");
        if let Some(needle) = filter
            && !key.contains(needle)
        {
            continue;
        }
        out.push(DiscoveredTest {
            name: name.clone(),
            span: *span,
        });
    }
    EnumerationOutcome::Tests(out)
}

fn is_unit_type(ty: &chelis_surf::ast::TypeExpr) -> bool {
    use chelis_surf::ast::TypeExpr;
    match ty {
        TypeExpr::Named(name, _) => name == "unit",
        TypeExpr::Infer(_) => true,
        _ => false,
    }
}

/// Per-test evaluation wrapper that runs the closure on a worker thread with
/// a generous (32 MB) stack and catches non-stack panics via `catch_unwind`.
/// The default 2 MB thread stack is too tight for common test patterns that
/// recurse through the evaluator's AST walker; 32 MB is a pragmatic v1 upper
/// bound. Pathological recursion beyond that still aborts the whole process
/// because Rust's stack-overflow handler is `abort()`, not `panic()`. Full
/// subprocess isolation is tracked as a follow-up.
fn run_test_with_timeout<T: Send + 'static>(
    f: impl FnOnce() -> Result<T, String> + Send + std::panic::UnwindSafe + 'static,
    timeout: Duration,
    timeout_msg: &str,
) -> Result<T, String> {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel::<Result<T, String>>();
    let builder = std::thread::Builder::new()
        .name("chelis-test-worker".to_string())
        .stack_size(32 * 1024 * 1024);
    if builder
        .spawn(move || {
            let result = std::panic::catch_unwind(f);
            let payload = match result {
                Ok(inner) => inner,
                Err(panic_payload) => {
                    let msg = panic_payload
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic_payload.downcast_ref::<&str>().map(|s| s.to_string()))
                        .unwrap_or_else(|| "test panicked (no message)".to_string());
                    Err(format!("panic: {msg}"))
                }
            };
            let _ = tx.send(payload);
        })
        .is_err()
    {
        return Err("failed to spawn test worker thread".to_string());
    }
    rx.recv_timeout(timeout)
        .unwrap_or_else(|_| Err(timeout_msg.to_string()))
}

/// Evaluate the module without any synthesized test-caller binding. Returns
/// `Some(err)` if any top-level `let _ = assert_*(...)` or other module-init
/// computation fails; `None` if module init is clean. Wrapped in the same
/// per-test timeout because a pathological module-init loop should not hang
/// the runner.
fn eval_module_init(
    exec_context: &TestExecutionContext,
    flat_decls: &[Decl],
    timeout: Duration,
) -> Option<String> {
    // Only the test file's own top-level `let` bindings are module-init for
    // the suite. If the file has none, skip the eval entirely.
    // Using `compiler::eval` (no selected roots) evaluates EVERY top-level
    // non-fn binding in the concatenated program — including library modules
    // when the package depends on chelis-std — which fails under strict-load
    // semantics because library bodies reference yet-to-be-provided inputs.
    // `eval_selected` scopes the eval to the test file's own roots so
    // module-init means what it should: the test file's top level, not the
    // standard library's.
    let mut module_roots: Vec<String> = Vec::new();
    for decl in flat_decls {
        if let Decl::LetDef { name, .. } = decl {
            module_roots.push(name.clone());
        }
    }
    if module_roots.is_empty() {
        return None;
    }

    // Phase G' — module-init evaluation. Routes through legacy
    // `compile_with_reef_graph` + `eval_selected`. The in-context
    // path is unblocked for `chelis eval`/`check` consumers but not
    // for `chelis test` until the linearity divergence is fixed.
    let prepared = match chelis_reef::compile_with_reef_graph(exec_context.reef_graph(), flat_decls)
    {
        Ok(p) => p,
        Err(err) => return Some(format!("compile: {err}")),
    };
    let source_text = chelis_surf::format::format_program(&prepared.decls);
    let outcome = run_test_with_timeout(
        move || {
            // RFC v5 (RT-1 F2 bypass): `source_text` is the reef-linked
            // module formatted back to Surf (internal-name mangled);
            // `eval_selected` re-parses + re-checks it. This closure
            // runs on a SPAWNED worker thread, so the main-thread
            // linked-program guard does not apply -- install it here so
            // the linker's own names are accepted in this thread.
            let _linked = chelis_types::install_linked_program_guard();
            let request = EvalRequest {
                source_kind: SourceKind::Surf,
                source: source_text,
                bindings: BTreeMap::new(),
            };
            Ok(chelis_compiler_api::compiler::eval_selected(request, &module_roots).map(|_| ()))
        },
        timeout,
        &format!("module-init timeout after {}s", timeout.as_secs()),
    );

    match outcome {
        Err(msg) => Some(msg),
        Ok(Ok(_)) => None,
        Ok(Err(err)) => Some(
            err.errors
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("; "),
        ),
    }
}

fn flatten_module_decls(decls: &[Decl]) -> Vec<Decl> {
    let mut out = Vec::new();
    for decl in decls {
        match decl {
            Decl::Module { decls: inner, .. } => {
                out.extend(flatten_module_decls(inner));
            }
            other => out.push(other.clone()),
        }
    }
    out
}

fn load_eval_decls(file: &Path) -> Result<(Vec<Decl>, Vec<Decl>), Box<dyn std::error::Error>> {
    let current_dir = env::current_dir()?;
    if let Some(prepared) = chelis_reef::prepare_program_for_eval_file(file, &current_dir)
        .map_err(boxed_string_error)?
    {
        return Ok((prepared.decls, prepared.entry_decls));
    }
    let source = fs::read_to_string(file)?;
    let decls = chelis_surf::parser::parse_str(&source)?;
    Ok((decls.clone(), decls))
}

fn reject_unsupported_hip_ops(dag: &chelis_ir::dag::Dag) -> Result<(), Box<dyn std::error::Error>> {
    let sparse_index_nodes: HashSet<chelis_ir::dag::NodeId> = dag
        .nodes()
        .iter()
        .filter_map(|node| match node.op {
            chelis_ir::dag::RiscOp::Gather { .. }
            | chelis_ir::dag::RiscOp::ScatterAdd { .. }
            | chelis_ir::dag::RiscOp::Scatter { .. } => node.inputs.get(1).copied(),
            _ => None,
        })
        .collect();

    for node in dag.nodes() {
        match &node.op {
            // `pad` / `shrink` are implemented on the HIP backend (typed
            // per-output-element kernels). They fall through to codegen;
            // no reject arm here.

            // `reduce_window_*` HIP codegen is deferred (spec §2.3.1). Reject
            // cleanly here rather than reaching the launch-emit `todo!`, which
            // would abort the build with an `internal error` panic.
            chelis_ir::dag::RiscOp::ReduceWindow { .. } => {
                return Err(format!(
                    "`chelis build --target hip` does not yet support `reduce_window_*`; \
                     lowered node {} requires it. HIP windowed-reduction codegen is deferred \
                     (spec/05-risc-primitives.md §2.3.1); use `--target c`.",
                    node.id.0
                )
                .into());
            }
            chelis_ir::dag::RiscOp::ReduceWindowGrad { .. } => {
                return Err(format!(
                    "`chelis build --target hip` does not yet support the `reduce_window_*` \
                     adjoint; lowered node {} requires it. HIP windowed-reduction codegen is \
                     deferred (spec/05-risc-primitives.md §2.3.1); use `--target c`.",
                    node.id.0
                )
                .into());
            }
            chelis_ir::dag::RiscOp::OneHot { .. } => {
                return Err(format!(
                    "`chelis build --target hip` cannot compile internal one_hot node {}: \
                     the sparse gather recognizer must consume OneHot before backend emission",
                    node.id.0
                )
                .into());
            }
            chelis_ir::dag::RiscOp::Shape { .. } => {
                return Err(format!(
                    "`chelis build --target hip` does not yet support the runtime `shape` \
                     value read; lowered node {} requires it. The C backend is canonical \
                     for runtime-dim reads (chelis#513/#558); use `--target c`.",
                    node.id.0
                )
                .into());
            }
            // chelis#616: node-valued (runtime) movement bounds are C-only.
            chelis_ir::dag::RiscOp::Shrink { bounds }
                if bounds
                    .iter()
                    .any(|(s, e)| s.node_input().is_some() || e.node_input().is_some()) =>
            {
                return Err(format!(
                    "`chelis build --target hip` does not yet support a runtime (node-valued) \
                     `shrink` bound; lowered node {} requires it. Use `--target c` (chelis#616).",
                    node.id.0
                )
                .into());
            }
            chelis_ir::dag::RiscOp::Pad { padding, .. }
                if padding
                    .iter()
                    .any(|(s, e)| s.node_input().is_some() || e.node_input().is_some()) =>
            {
                return Err(format!(
                    "`chelis build --target hip` does not yet support a runtime (node-valued) \
                     `pad` bound; lowered node {} requires it. Use `--target c` (chelis#616).",
                    node.id.0
                )
                .into());
            }
            chelis_ir::dag::RiscOp::Stride { strides }
                if strides.iter().any(|s| s.node_input().is_some()) =>
            {
                return Err(format!(
                    "`chelis build --target hip` does not yet support a runtime (node-valued) \
                     `stride` step; lowered node {} requires it. Use `--target c` (chelis#616).",
                    node.id.0
                )
                .into());
            }
            chelis_ir::dag::RiscOp::Reshape { new_shape }
                if new_shape.iter().any(|d| d.node_input().is_some()) =>
            {
                return Err(format!(
                    "`chelis build --target hip` does not yet support a runtime (node-valued) \
                     `reshape` target extent; lowered node {} requires it. \
                     Use `--target c` (chelis#616).",
                    node.id.0
                )
                .into());
            }
            chelis_ir::dag::RiscOp::Gather { .. } => {
                let values = &dag.get(node.inputs[0]).unwrap().output_type;
                let index_node = dag.get(node.inputs[1]).unwrap();
                let indices = &index_node.output_type;
                if values.precision != chelis_types::types::Prim::F32
                    || node.output_type.precision != chelis_types::types::Prim::F32
                {
                    return Err(format!(
                        "`chelis build --target hip` sparse gather supports f32 payloads only; \
                         node {} carries payload precision `{}` and output precision `{}`",
                        node.id.0,
                        values.precision.name(),
                        node.output_type.precision.name()
                    )
                    .into());
                }
                if !matches!(
                    indices.precision,
                    chelis_types::types::Prim::Int32 | chelis_types::types::Prim::Int64
                ) {
                    return Err(format!(
                        "`chelis build --target hip` sparse gather requires int32/int64 indices; \
                         node {} uses `{}`",
                        node.id.0,
                        indices.precision.name()
                    )
                    .into());
                }
                if !matches!(index_node.op, chelis_ir::dag::RiscOp::Load { .. }) {
                    return Err(format!(
                        "`chelis build --target hip` sparse gather requires indices to be loaded input tensors in this milestone; \
                         node {} uses indices produced by {:?}. \
                         Non-load integer index producers need integer HIP codegen before they can feed sparse kernels safely.",
                        node.id.0,
                        index_node.op
                    )
                    .into());
                }
            }
            chelis_ir::dag::RiscOp::ScatterAdd { .. } | chelis_ir::dag::RiscOp::Scatter { .. } => {
                let (label, payload_blocker) = match &node.op {
                    chelis_ir::dag::RiscOp::ScatterAdd { .. } => (
                        "scatter_add",
                        "f64 scatter_add needs backend-specific atomic support and is not in this milestone.",
                    ),
                    chelis_ir::dag::RiscOp::Scatter { .. } => (
                        "scatter_replace",
                        "f64 scatter_replace requires a widened serial last-write-wins kernel and is not in this milestone.",
                    ),
                    _ => unreachable!(),
                };
                let target = &dag.get(node.inputs[0]).unwrap().output_type;
                let index_node = dag.get(node.inputs[1]).unwrap();
                let indices = &index_node.output_type;
                let updates = &dag.get(node.inputs[2]).unwrap().output_type;
                if target.precision != chelis_types::types::Prim::F32
                    || updates.precision != chelis_types::types::Prim::F32
                    || node.output_type.precision != chelis_types::types::Prim::F32
                {
                    return Err(format!(
                        "`chelis build --target hip` sparse {label} supports f32 payloads only; \
                         node {} carries target `{}`, updates `{}`, output `{}`. \
                         {payload_blocker}",
                        node.id.0,
                        target.precision.name(),
                        updates.precision.name(),
                        node.output_type.precision.name()
                    )
                    .into());
                }
                if !matches!(
                    indices.precision,
                    chelis_types::types::Prim::Int32 | chelis_types::types::Prim::Int64
                ) {
                    return Err(format!(
                        "`chelis build --target hip` sparse {label} requires int32/int64 indices; \
                         node {} uses `{}`",
                        node.id.0,
                        indices.precision.name()
                    )
                    .into());
                }
                if !matches!(index_node.op, chelis_ir::dag::RiscOp::Load { .. }) {
                    return Err(format!(
                        "`chelis build --target hip` sparse {label} requires indices to be loaded input tensors in this milestone; \
                         node {} uses indices produced by {:?}. \
                         Non-load integer index producers need integer HIP codegen before they can feed sparse kernels safely.",
                        node.id.0,
                        index_node.op
                    )
                    .into());
                }
            }
            _ => {}
        }
    }
    // RT-4 F5: HIP backend dtype coverage today (verified end-to-end
    // before admitting):
    //   * f32, bool — full elementwise + matmul + reductions.
    //   * f64       — full elementwise + dgemm matmul (WS-A2).
    //   * bf16, f16 — MATMUL ONLY via `hipblasGemmEx` (WS-A3). The
    //                 elementwise kernel suffix path
    //                 (`dtype_kernel_suffix`) panics on bf16/f16.
    //                 Admit only when every bf16/f16 node is an input
    //                 load, output store, or BLAS matmul / its operand
    //                 path; reject earlier otherwise so the user sees
    //                 a structured CLI diagnostic instead of a panic.
    //   * int8/int16/int32/int64 — typed elementwise kernels (WS-A4),
    //                 plus loaded sparse indices.
    //
    // The widened admit-list (was f32/bool only) closes the divergence
    // the RT-4 red team flagged: the HIP backend's bf16/f16 GEMM
    // machinery was unreachable from the CLI surface.
    // First pass: collect the directly-admitted bf16/f16 nodes (Load,
    // Store, BlasMatmul). Then reachability-extend through the operand
    // path of any BlasMatmul: a bf16/f16 helper that lowers to BlasMatmul
    // typically introduces intermediate Expand / Realize / Copy nodes
    // (the matmul lowering shape per spec/04-type-system.md §5.7) whose
    // bf16/f16 buffers are never actually read by an elementwise kernel
    // because the BlasMatmul subsumes them; admitting these intermediates
    // keeps the WS-A3 hipblasGemmEx path reachable from the Surf CLI.
    let mut bf16_or_f16_admissible_ops: HashSet<chelis_ir::dag::NodeId> = dag
        .nodes()
        .iter()
        .filter(|node| {
            matches!(
                node.output_type.precision,
                chelis_types::types::Prim::Bf16 | chelis_types::types::Prim::F16
            )
        })
        .filter_map(|node| match &node.op {
            // Loads / stores carrying bf16/f16 are admitted; the storage
            // is 2 bytes (per `chelis_gpu_dtype_size`) and the buffers
            // round-trip without per-element arithmetic.
            chelis_ir::dag::RiscOp::Load { .. } | chelis_ir::dag::RiscOp::Store { .. } => {
                Some(node.id)
            }
            // BLAS matmul on bf16/f16 dispatches to `hipblasGemmEx` with
            // an f32 accumulator (WS-A3). Its operands stay bf16/f16.
            chelis_ir::dag::RiscOp::BlasMatmul { .. } => Some(node.id),
            _ => None,
        })
        .collect();
    // Reachability pass: walk upward from each BlasMatmul through its
    // direct/transitive operand bf16/f16 nodes. Admit Expand / Realize /
    // Copy / Mul / Sum that are subsumed by the BLAS substitution at
    // emit time.
    let mut frontier: Vec<chelis_ir::dag::NodeId> = bf16_or_f16_admissible_ops
        .iter()
        .copied()
        .filter(|id| {
            matches!(
                dag.get(*id).map(|n| &n.op),
                Some(chelis_ir::dag::RiscOp::BlasMatmul { .. })
            )
        })
        .collect();
    while let Some(id) = frontier.pop() {
        if let Some(node) = dag.get(id) {
            for &input in &node.inputs {
                if let Some(inp_node) = dag.get(input)
                    && matches!(
                        inp_node.output_type.precision,
                        chelis_types::types::Prim::Bf16 | chelis_types::types::Prim::F16
                    )
                    && bf16_or_f16_admissible_ops.insert(input)
                {
                    frontier.push(input);
                }
            }
        }
    }

    for node in dag.nodes() {
        match node.output_type.precision {
            chelis_types::types::Prim::F32
            | chelis_types::types::Prim::F64
            | chelis_types::types::Prim::Bool
            | chelis_types::types::Prim::Int8
            | chelis_types::types::Prim::Int16 => {}
            chelis_types::types::Prim::Int32 | chelis_types::types::Prim::Int64 => {
                // Loaded int32/int64 sparse indices are still admitted
                // unconditionally; non-load int producers are admitted
                // via the WS-A4 typed kernel templates.
                let _ = sparse_index_nodes.contains(&node.id);
            }
            chelis_types::types::Prim::Bf16 | chelis_types::types::Prim::F16 => {
                // Admit only nodes that the HIP backend can actually
                // emit today: load/store buffers and BLAS matmul.
                // Other bf16/f16 producers (elementwise, reductions,
                // casts) would panic in the kernel-suffix path; reject
                // here with a citation so the user gets a useful
                // error and not a Rust stack trace.
                if !bf16_or_f16_admissible_ops.contains(&node.id) {
                    return Err(format!(
                        "unsupported: `chelis build --target hip` admits `{}` only on tensor \
                         load/store nodes and on `BlasMatmul` operands today \
                         (`hipblasGemmEx` with an f32 accumulator, WS-A3). \
                         Node {} carries op {:?} which has no bf16/f16 kernel \
                         template yet (chelis-backend-hip emit::dtype_kernel_suffix). \
                         See spec/04-type-system.md §5.7.1.",
                        node.output_type.precision.name(),
                        node.id.0,
                        node.op
                    )
                    .into());
                }
            }
            other => {
                return Err(format!(
                    "unsupported: `chelis build --target hip` DAG path does not support tensor \
                     precision `{}` (node {}). \
                     Supported: f32/f64/bool plus the integer family \
                     (int8/int16/int32/int64), with bf16/f16 admitted on matmul \
                     and load/store nodes. See spec/04-type-system.md §5.7.1.",
                    other.name(),
                    node.id.0,
                )
                .into());
            }
        }
    }
    let _ = sparse_index_nodes;
    Ok(())
}

fn reject_unsupported_metal_ops(
    dag: &chelis_ir::dag::Dag,
) -> Result<(), Box<dyn std::error::Error>> {
    // WS-M1: per spec/04-type-system.md §1.1.3, the Metal backend admits
    // every active dtype except f64 (Apple Silicon GPUs lack FP64 ALUs;
    // software emulation explicitly out of scope). bf16 additionally
    // requires Apple7+ (M3 or later) at runtime; the kernel template
    // gates `bfloat` behind `#if __METAL_VERSION__ >= 320` so the
    // emitted source artifact is valid on every toolchain.
    //
    // Mirrors `reject_unsupported_hip_ops` precision discipline: admit
    // a precise allow-list per spec, reject the rest with a structured
    // CLI diagnostic instead of a panic from the kernel templates.
    for node in dag.nodes() {
        // WS-8A: `pad` / `shrink` are implemented on the Metal backend
        // (typed per-output-element MSL kernels). They fall through to
        // codegen; no reject arm here. f64 and any out-of-matrix dtype are
        // still rejected by the precision gate below.
        //
        // chelis#616: node-valued (runtime) movement bounds and reshape
        // target extents are C-only. Without this arm a runtime `pad` /
        // `shrink` bound reaches `metal_bound_to_usize`'s defensive panic
        // instead of a clean CLI diagnostic.
        let node_valued = match &node.op {
            chelis_ir::dag::RiscOp::Shrink { bounds } => bounds
                .iter()
                .any(|(s, e)| s.node_input().is_some() || e.node_input().is_some()),
            chelis_ir::dag::RiscOp::Pad { padding, .. } => padding
                .iter()
                .any(|(s, e)| s.node_input().is_some() || e.node_input().is_some()),
            chelis_ir::dag::RiscOp::Stride { strides } => {
                strides.iter().any(|s| s.node_input().is_some())
            }
            chelis_ir::dag::RiscOp::Reshape { new_shape } => {
                new_shape.iter().any(|d| d.node_input().is_some())
            }
            _ => false,
        };
        if node_valued {
            return Err(format!(
                "`chelis build --target metal` does not support a runtime (node-valued) \
                 movement bound or reshape target extent; lowered node {} requires it. \
                 Use `--target c` (chelis#616).",
                node.id.0
            )
            .into());
        }
        match node.output_type.precision {
            chelis_types::types::Prim::F32
            | chelis_types::types::Prim::F16
            | chelis_types::types::Prim::Bf16
            | chelis_types::types::Prim::Int8
            | chelis_types::types::Prim::Int16
            | chelis_types::types::Prim::Int32
            | chelis_types::types::Prim::Int64
            | chelis_types::types::Prim::Bool => {}
            chelis_types::types::Prim::F64 => {
                // Hardware-rejected per spec/04-type-system.md §1.1.3.
                // Diagnostic text is the spec-pinned string; tests
                // assert exact-string match so this must not drift.
                return Err(format!(
                    "unsupported: `chelis build --target metal` rejects f64 (node {}): \
                     Apple Silicon GPUs lack FP64 ALUs; use `--target c` or \
                     `--target hip` for f64 workloads. \
                     See spec/04-type-system.md §1.1.3.",
                    node.id.0
                )
                .into());
            }
            other => {
                return Err(format!(
                    "`chelis build --target metal` DAG path does not support tensor \
                     precision `{}` (node {}). The Metal backend admits the \
                     active dtype set per spec/04-type-system.md §1.1.3 except \
                     f64; supported: f32/f16/bf16/int8/int16/int32/int64/bool.",
                    other.name(),
                    node.id.0
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Decide whether a tensor precision is supported by the C backend.
///
/// Mirrors the admit-list in `validate_supported_precisions` inside
/// `chelis-backend-c::emit`. WS-1 admits bf16 and f16 (storage as
/// `uint16_t`; arithmetic via `chelis_bf16_to_f32` / `chelis_f16_to_f32`
/// per spec/04-type-system.md §5.7.1; matmul via convert-then-`cblas_sgemm`
/// with f32 scratch buffers). The C backend now admits the full active
/// numeric dtype set; f8e4m3 remains deferred per §1.1.1.
fn c_backend_supports_precision(precision: chelis_types::types::Prim) -> bool {
    use chelis_types::types::Prim;
    matches!(
        precision,
        Prim::F32
            | Prim::F64
            | Prim::Bool
            | Prim::Bf16
            | Prim::F16
            | Prim::Int8
            | Prim::Int16
            | Prim::Int32
            | Prim::Int64
    )
}

/// Target-build builtins that must reject before host lowering examines an
/// argument with no standalone compiled representation.
const COMPILED_HOST_ONLY_BUILTINS: &[&str] = &["tensor_scan"];

fn reject_host_only_builtins_before_host_lowering(
    program: &chelis_types::CheckedProgram,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let target = match target {
        "c" => "c",
        "hip" => "hip",
        "metal" => "metal",
        _ => return Ok(()),
    };
    if let Some(name) =
        chelis_ir::host::find_direct_builtin_call(program, COMPILED_HOST_ONLY_BUILTINS)
    {
        return Err(
            chelis_types::unsupported::Unsupported::compiled_host_only_builtin(name, target)
                .to_string()
                .into(),
        );
    }
    Ok(())
}

/// Reject eval/test-only builtins that have no compiled-target lowering.
///
/// Hull Phase 0a: `process_run` runs a subprocess from the IR evaluator
/// (under `chelis eval` / `chelis test`) but is deliberately unsupported by
/// the C/HIP/Metal build backends -- a compiled artifact cannot reach the
/// host interpreter's `Command` exec path, and emitting C for it would
/// silently fall through to `/* unsupported builtin */ 0` (a wrong value,
/// not a diagnostic). This guard turns that into a clean build error.
fn reject_eval_only_builtins_host(
    program: &chelis_ir::host::ConcreteHostProgram,
) -> Result<(), Box<dyn std::error::Error>> {
    for builtin in EVAL_ONLY_HOST_BUILTINS {
        if chelis_ir::host::host_program_uses_builtin(program, builtin) {
            return Err(format!(
                "{builtin} is an eval/test-only builtin; not available in compiled \
                    targets. Run the program with `chelis eval` or `chelis test` instead, \
                    or remove the {builtin} call before building."
            )
            .into());
        }
    }
    Ok(())
}

/// Mirror of `reject_unsupported_c_precisions` for the host-program lane.
///
/// The C backend's `codegen_host_program` recursively invokes
/// `CEmitter::emit_dag_with_options` on every `tensor_helper`'s DAG, which
/// internally panics on unsupported precisions (`validate_supported_precisions`
/// at chelis-backend-c::emit). For the DAG-only path the CLI guards the
/// panic with `reject_unsupported_c_precisions`; this function does the
/// same for the host-program lane (RT-4 F4: `def f(x: tensor[3, bf16]) ...`
/// previously panicked with a Rust stack trace).
fn reject_unsupported_c_precisions_host(
    program: &chelis_ir::host::ConcreteHostProgram,
) -> Result<(), Box<dyn std::error::Error>> {
    fn check_host_type(
        ty: &chelis_ir::ConcreteHostType,
        context: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use chelis_ir::ConcreteHostType as HostType;
        match ty {
            HostType::Tensor(t) if !c_backend_supports_precision(t.precision) => {
                return Err(format!(
                    "`chelis build --target c` host-program lane does not yet support \
                     tensor precision `{}` ({}). The C backend admits \
                     f32/f64/bf16/f16/bool/int8/int16/int32/int64; f8e4m3 is \
                     deferred per spec/04-type-system.md §1.1.1.",
                    t.precision.name(),
                    context
                )
                .into());
            }
            HostType::Tensor(_) => {}
            HostType::List(inner) | HostType::Option(inner) => check_host_type(inner, context)?,
            HostType::Dict(k, v) => {
                check_host_type(k, context)?;
                check_host_type(v, context)?;
            }
            HostType::Tuple(items) => {
                for item in items {
                    check_host_type(item, context)?;
                }
            }
            HostType::Function(params, ret) => {
                for p in params {
                    check_host_type(p, context)?;
                }
                check_host_type(ret, context)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn check_helper(
        helper: &chelis_ir::host::HostTensorHelper,
        context: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for input in &helper.inputs {
            if !c_backend_supports_precision(input.ty.precision) {
                return Err(format!(
                    "`chelis build --target c` host-program lane does not yet support \
                     tensor precision `{}` (helper `{}` input `{}` in {}). \
                     The C backend admits f32/f64/bf16/f16/bool/int8/int16/int32/int64; \
                     f8e4m3 is deferred per spec/04-type-system.md §1.1.1.",
                    input.ty.precision.name(),
                    helper.name,
                    input.name,
                    context
                )
                .into());
            }
        }
        if !c_backend_supports_precision(helper.output.precision) {
            return Err(format!(
                "`chelis build --target c` host-program lane does not yet support \
                 tensor precision `{}` (helper `{}` output in {}). \
                 The C backend admits f32/f64/bf16/f16/bool/int8/int16/int32/int64; \
                 f8e4m3 is deferred per spec/04-type-system.md §1.1.1.",
                helper.output.precision.name(),
                helper.name,
                context
            )
            .into());
        }
        for node in helper.dag.nodes() {
            if !c_backend_supports_precision(node.output_type.precision) {
                return Err(format!(
                    "`chelis build --target c` host-program lane does not yet support \
                     tensor precision `{}` (helper `{}` node {} in {}). \
                     The C backend admits f32/f64/bf16/f16/bool/int8/int16/int32/int64; \
                     f8e4m3 is deferred per spec/04-type-system.md §1.1.1.",
                    node.output_type.precision.name(),
                    helper.name,
                    node.id.0,
                    context
                )
                .into());
            }
        }
        Ok(())
    }

    for binding in &program.globals {
        check_host_type(&binding.ty, &format!("global `{}`", binding.name))?;
    }
    for helper in &program.global_tensor_helpers {
        check_helper(helper, "global tensor helpers")?;
    }
    for function in &program.functions {
        let context = format!("function `{}`", function.name);
        for param in &function.params {
            check_host_type(&param.ty, &format!("{} param `{}`", context, param.name))?;
        }
        check_host_type(&function.ret_ty, &format!("{} return", context))?;
        for helper in &function.tensor_helpers {
            check_helper(helper, &context)?;
        }
    }
    Ok(())
}

/// Mirror of the per-node precision walk that the C backend's emitter
/// performs internally (`validate_supported_precisions` panics). Emits a
/// clean user-facing error BEFORE the backend panics, closing a
/// check-pass/build-panic gap for programs like `def f() -> f64 = cast(1.0, f64)`
/// that reach DAG lowering with a non-F32/Bool node.
///
/// Prefer reporting the node that first mismatches the user's declared
/// output type (usually a scalar literal whose declared type is int64/f64
/// vs an internal int32/f32 node) so the error line matches the source
/// intent rather than the internal lowering.
fn reject_unsupported_c_precisions(
    dag: &chelis_ir::dag::Dag,
) -> Result<(), Box<dyn std::error::Error>> {
    let sparse_index_nodes: HashSet<chelis_ir::dag::NodeId> = dag
        .nodes()
        .iter()
        .filter_map(|node| match node.op {
            chelis_ir::dag::RiscOp::Gather { .. }
            | chelis_ir::dag::RiscOp::ScatterAdd { .. }
            | chelis_ir::dag::RiscOp::Scatter { .. } => node.inputs.get(1).copied(),
            _ => None,
        })
        .collect();

    for node in dag.nodes() {
        match node.output_type.precision {
            chelis_types::types::Prim::F32 | chelis_types::types::Prim::Bool => {}
            // WS-1: bf16 / f16 are admitted on the C-backend DAG path
            // post-cycle. The emitter routes elementwise ops through
            // `chelis_<x>_to_f32` convert-load helpers and matmul
            // through `convert-then-cblas_sgemm` per spec §5.7.1.
            chelis_types::types::Prim::Bf16 | chelis_types::types::Prim::F16 => {}
            chelis_types::types::Prim::Int32 | chelis_types::types::Prim::Int64
                if sparse_index_nodes.contains(&node.id) => {}
            other => {
                return Err(format!(
                    "`chelis build --target c` DAG path only supports f32/bool/bf16/f16 tensors, \
                     plus int32/int64 tensors when they are consumed as sparse indices; \
                     node {} carries precision `{}`. \
                     Non-f32/bool/bf16/f16 tensors must flow through the host-lane wrapper \
                     (use `to_tensor([...])`/`pad_sequences` or declare a helper fn \
                     that the host lane can emit as a real C symbol).",
                    node.id.0,
                    other.name()
                )
                .into());
            }
        }
    }
    Ok(())
}

fn reject_unsupported_effect_ops(
    dag: &chelis_ir::dag::Dag,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for node in dag.nodes() {
        if let chelis_ir::dag::RiscOp::Dropout { .. } = &node.op {
            return Err(format!(
                "`chelis build --target {target}` does not yet codegen `dropout`; evaluate it under `with seed(...)` instead"
            )
            .into());
        }
    }
    Ok(())
}

/// `reduce_window_*` over a runtime-symbolic windowed axis cannot be
/// lowered to a correct static output shape on the build path: the
/// windowed output extent `floor((d - window) / stride) + 1` is strictly
/// smaller than the input extent `d` and is not representable as a
/// `DimExpr` (no subtraction / floor), so the backend's symbolic-dim
/// binding would tie the windowed output axis to the *input* extent —
/// silently mis-allocating the output and emitting an out-of-bounds
/// window read (build output then diverges from the IR evaluator / host
/// runtime). Reject per spec/05-risc-primitives.md §2.3.1.
///
/// Mirrors `chelis_compiler_api::compiler::reject_symbolic_windowed_reduce`;
/// the CLI build pipeline is independent of `compile_for_execution`, so
/// the guard is duplicated here. Only the windowed (trailing
/// `window_shape.len()`) axes are checked; leading pass-through axes may
/// remain symbolic and bind correctly.
fn reject_symbolic_windowed_reduce(
    dag: &chelis_ir::dag::Dag,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use chelis_ir::dag::{DimInfo, RiscOp};
    for node in dag.nodes() {
        let RiscOp::ReduceWindow { window_shape, .. } = &node.op else {
            continue;
        };
        let dims = &node.output_type.dims;
        let leading = dims.len().saturating_sub(window_shape.len());
        for (offset, dim) in dims.iter().enumerate().skip(leading) {
            if let DimInfo::Named(name, None) = dim {
                return Err(format!(
                    "`chelis build --target {target}` requires statically-known \
                     windowed-axis extents for `reduce_window_*`; node {} windowed axis \
                     {offset} has runtime-only symbolic dimension `{name}`. The windowed \
                     output extent floor((d - window) / stride) + 1 is not representable \
                     for a runtime-only input extent, so the build cannot allocate a \
                     correct output. Window over a statically-sized axis, or pad the \
                     input to a concrete extent first. See spec/05-risc-primitives.md §2.3.1.",
                    node.id.0
                )
                .into());
            }
        }
    }
    Ok(())
}

/// Apply [`reject_symbolic_windowed_reduce`] to every tensor-helper DAG
/// embedded in a host program. The host codegen path
/// (`codegen_host_program`) lowers `reduce_window_*` from these helper
/// DAGs, so the pure-DAG guard alone would miss the node.
fn reject_symbolic_windowed_reduce_host(
    program: &chelis_ir::host::ConcreteHostProgram,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for helper in &program.global_tensor_helpers {
        reject_symbolic_windowed_reduce(&helper.dag, target)?;
    }
    for function in &program.functions {
        for helper in &function.tensor_helpers {
            reject_symbolic_windowed_reduce(&helper.dag, target)?;
        }
    }
    Ok(())
}

/// `reduce_window_*` (and its adjoint) are f32-only in the C backend:
/// `emit_reduce_window{,_grad}` have no bf16/f16 convert-load path yet.
/// `reject_unsupported_c_precisions` admits bf16/f16 generally, so without
/// this guard a bf16/f16 windowed reduction reaches the emitter and aborts
/// with an `internal error` panic instead of a clean diagnostic. Reject it
/// at compile time; the emitter `panic!` stays as a defensive backstop.
///
/// Mirrors `chelis_compiler_api::compiler::reject_unsupported_reduce_window_precision`;
/// the CLI build pipeline is independent of `compile_for_execution`, so the
/// guard is duplicated here. See spec/05-risc-primitives.md §2.3.1.
fn reject_unsupported_reduce_window_precision(
    dag: &chelis_ir::dag::Dag,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use chelis_ir::dag::RiscOp;
    for node in dag.nodes() {
        let (op_label, reducer) = match &node.op {
            RiscOp::ReduceWindow { reducer, .. } => ("reduce_window_*", reducer),
            RiscOp::ReduceWindowGrad { reducer, .. } => ("reduce_window_* adjoint", reducer),
            _ => continue,
        };
        let prec = node.output_type.precision;
        if prec != chelis_types::types::Prim::F32 {
            return Err(format!(
                "`chelis build --target {target}` supports `{op_label}` (`{}`) on f32 \
                 tensors only; node {} carries precision `{}`. bf16/f16 windowed \
                 reductions are not yet lowered (no convert-load path); cast to f32 \
                 before the windowed reduction. See spec/05-risc-primitives.md §2.3.1.",
                reducer.surf_name(),
                node.id.0,
                prec.name(),
            )
            .into());
        }
    }
    Ok(())
}

/// Apply [`reject_unsupported_reduce_window_precision`] to every
/// tensor-helper DAG embedded in a host program, mirroring
/// [`reject_symbolic_windowed_reduce_host`].
fn reject_unsupported_reduce_window_precision_host(
    program: &chelis_ir::host::ConcreteHostProgram,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for helper in &program.global_tensor_helpers {
        reject_unsupported_reduce_window_precision(&helper.dag, target)?;
    }
    for function in &program.functions {
        for helper in &function.tensor_helpers {
            reject_unsupported_reduce_window_precision(&helper.dag, target)?;
        }
    }
    Ok(())
}

fn cmd_validate(
    file: &Path,
    surf: bool,
    deep: bool,
    desugar: bool,
    allow_style_violations: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    style_gate::enforce_style_gate(file, &source, allow_style_violations)?;
    let mode = if surf {
        "surf"
    } else if deep {
        "deep"
    } else if desugar {
        "desugar"
    } else {
        return Err("validation mode is required".into());
    };

    let result = match mode {
        "surf" => chelis_validate::validate_surf(&source),
        "deep" => {
            let deep_source = style_gate::strip_deep_lint_directive_lines(&source);
            chelis_validate::validate_deep(&deep_source)
        }
        "desugar" => chelis_validate::validate_desugared(&source),
        _ => unreachable!("validated above"),
    };

    match result {
        Ok(()) => {
            println!("validated {mode}: {}", file.display());
            Ok(())
        }
        Err(err) => Err(err.into()),
    }
}

fn cmd_build_c(
    dag: &chelis_ir::dag::Dag,
    func_name: &str,
    _file: &std::path::Path,
    output: Option<&std::path::Path>,
    symbolic_dims_hint: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let result = chelis_backend_c::codegen_with_options(
        dag,
        func_name,
        chelis_backend_c::CodegenOptions {
            use_blas: true,
            ..chelis_backend_c::CodegenOptions::default()
        },
    )?;
    let symbolic_dims = fallback_symbolic_dims(dag, &result.symbolic_dims, symbolic_dims_hint);
    cmd_build_c_result(result, func_name, output, &symbolic_dims)
}

fn cmd_build_c_result(
    result: chelis_backend_c::CodegenResult,
    func_name: &str,
    output: Option<&std::path::Path>,
    symbolic_dims: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let c_path = if out_dir.extension().and_then(|e| e.to_str()) == Some("c") {
        out_dir.clone()
    } else {
        out_dir.join(format!("{func_name}.c"))
    };
    let h_path = c_path.with_extension("h");
    if let Some(parent) = c_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&c_path, &result.c_source)?;
    fs::write(&h_path, &result.h_header)?;

    let runtime_dir = c_path.parent().unwrap_or(std::path::Path::new("."));
    copy_runtime_artifacts(runtime_dir, ExtraRuntimeArtifacts::default())?;

    println!("Wrote {} and {}", c_path.display(), h_path.display());
    println!(
        "Wrote {}, {}, and {}",
        runtime_dir.join("chelis_runtime.h").display(),
        runtime_dir.join("chelis_blas.h").display(),
        runtime_dir.join("libchelis_runtime.a").display()
    );
    if !symbolic_dims.is_empty() {
        println!("Symbolic dims: {}", symbolic_dims.join(", "));
    }
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.requirements);
    if result.c_source.contains("int main(") {
        println!(
            "Compile: {} -O2 {} {} -L{} -lchelis_runtime {} -o {}",
            toolchain.compiler,
            toolchain.compile_flags.join(" "),
            c_path.display(),
            runtime_dir.display(),
            toolchain.link_flags.join(" "),
            c_path.with_extension("").display()
        );
    } else {
        println!(
            "Compile object: {} -O2 {} -c {}",
            toolchain.compiler,
            toolchain.compile_flags.join(" "),
            c_path.display()
        );
    }
    Ok(())
}

fn cmd_build_hip_host(
    result: chelis_backend_c::CodegenResult,
    func_name: &str,
    output: Option<&std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let c_path = if matches!(
        out_dir.extension().and_then(|e| e.to_str()),
        Some("c") | Some("cc") | Some("cpp") | Some("cxx")
    ) {
        out_dir.clone()
    } else {
        out_dir.join(format!("{func_name}_hip.cpp"))
    };
    let h_path = c_path.with_extension("h");
    if let Some(parent) = c_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&c_path, &result.c_source)?;
    fs::write(&h_path, &result.h_header)?;

    let runtime_dir = c_path.parent().unwrap_or(std::path::Path::new("."));
    copy_runtime_artifacts(
        runtime_dir,
        ExtraRuntimeArtifacts {
            hip: true,
            metal: false,
        },
    )?;

    println!("Wrote {} and {}", c_path.display(), h_path.display());
    println!(
        "Wrote runtime: {}, {}, {}, {}",
        runtime_dir.join("chelis_runtime.h").display(),
        runtime_dir.join("chelis_blas.h").display(),
        runtime_dir.join("libchelis_runtime.a").display(),
        runtime_dir.join("chelis_hip_runtime.h").display()
    );

    let cpu_toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.requirements);
    let mut compile_flags = cpu_toolchain
        .compile_flags
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    compile_flags.retain(|flag| *flag != "-fopenmp");
    let mut link_flags = cpu_toolchain
        .link_flags
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    link_flags.retain(|flag| *flag != "-fopenmp");
    if result.c_source.contains("int main(") {
        println!(
            "Compile: hipcc {} {} -L{} -lchelis_runtime -lpthread -ldl {} -o {}",
            compile_flags.join(" "),
            c_path.display(),
            runtime_dir.display(),
            link_flags.join(" "),
            c_path.with_extension("").display()
        );
    } else {
        println!(
            "Compile object: hipcc {} -c {}",
            compile_flags.join(" "),
            c_path.display()
        );
    }
    Ok(())
}

fn cmd_build_hip(
    dag: &chelis_ir::dag::Dag,
    func_name: &str,
    _file: &std::path::Path,
    output: Option<&std::path::Path>,
    symbolic_dims_hint: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let result = chelis_backend_hip::codegen_hip(dag, func_name)?;

    let out_dir = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let c_path = if matches!(
        out_dir.extension().and_then(|e| e.to_str()),
        Some("c") | Some("cc") | Some("cpp") | Some("cxx")
    ) {
        out_dir.clone()
    } else {
        out_dir.join(format!("{func_name}_hip.cpp"))
    };
    let h_path = c_path.with_extension("h");
    if let Some(parent) = c_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&c_path, &result.c_source)?;
    fs::write(&h_path, &result.h_header)?;

    // HIP runtime includes the CPU runtime (for chelis_tensor host struct)
    let runtime_dir = c_path.parent().unwrap_or(std::path::Path::new("."));
    copy_runtime_artifacts(
        runtime_dir,
        ExtraRuntimeArtifacts {
            hip: true,
            metal: false,
        },
    )?;

    println!("Wrote {} and {}", c_path.display(), h_path.display());
    println!(
        "Wrote runtime: {}, {}, {}",
        runtime_dir.join("chelis_runtime.h").display(),
        runtime_dir.join("libchelis_runtime.a").display(),
        runtime_dir.join("chelis_hip_runtime.h").display()
    );
    let symbolic_dims = fallback_symbolic_dims(dag, &result.symbolic_dims, symbolic_dims_hint);
    if !symbolic_dims.is_empty() {
        println!("Symbolic dims: {}", symbolic_dims.join(", "));
    }
    println!(
        "Peak device memory formula: {}",
        result.peak_device_bytes_formula
    );
    if let Some(bytes) = result.peak_device_bytes_estimate {
        println!("Estimated peak device memory: {}", human_bytes(bytes));
    }
    let mut flags: Vec<&str> = result
        .compile_flags
        .iter()
        .chain(result.link_flags.iter())
        .map(|s| s.as_str())
        .collect();
    flags.sort();
    flags.dedup();
    println!(
        "Compile: hipcc {} {} -L{} -lchelis_runtime -lpthread -ldl -o {}",
        flags.join(" "),
        c_path.display(),
        runtime_dir.display(),
        c_path.with_extension("").display()
    );
    Ok(())
}

fn cmd_build_metal(
    dag: &chelis_ir::dag::Dag,
    func_name: &str,
    _file: &std::path::Path,
    output: Option<&std::path::Path>,
    symbolic_dims_hint: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let result = chelis_backend_metal::codegen_metal(dag, func_name);

    let out_dir = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let mm_path = if matches!(
        out_dir.extension().and_then(|e| e.to_str()),
        Some("mm") | Some("cpp") | Some("cxx") | Some("cc")
    ) {
        out_dir.clone()
    } else {
        out_dir.join(format!("{func_name}_metal.mm"))
    };
    let h_path = mm_path.with_extension("h");
    if let Some(parent) = mm_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&mm_path, &result.mm_source)?;
    fs::write(&h_path, &result.h_header)?;

    // Metal runtime header includes the CPU runtime (for chelis_tensor host struct)
    let runtime_dir = mm_path.parent().unwrap_or(std::path::Path::new("."));
    copy_runtime_artifacts(
        runtime_dir,
        ExtraRuntimeArtifacts {
            hip: false,
            metal: true,
        },
    )?;

    println!("Wrote {} and {}", mm_path.display(), h_path.display());
    println!(
        "Wrote runtime: {}, {}, {}",
        runtime_dir.join("chelis_runtime.h").display(),
        runtime_dir.join("libchelis_runtime.a").display(),
        runtime_dir.join("chelis_metal_runtime.h").display()
    );
    let symbolic_dims = fallback_symbolic_dims(dag, &result.symbolic_dims, symbolic_dims_hint);
    if !symbolic_dims.is_empty() {
        println!("Symbolic dims: {}", symbolic_dims.join(", "));
    }
    // The Metal backend embeds the unified-memory caveat in the formula
    // string itself (see chelis_backend_metal::codegen_metal), so the
    // print line stays short — duplicating the note here would make it
    // double up.
    println!("Peak device memory: {}", result.peak_device_bytes_formula);
    if let Some(bytes) = result.peak_device_bytes_estimate {
        println!("Estimated peak device memory: {}", human_bytes(bytes));
    }
    // Preserve the (-framework, NAME) pair ordering — sorting would split
    // them. Compile flags first, then link flags, in their declared order.
    let flags: Vec<&str> = result
        .compile_flags
        .iter()
        .chain(result.link_flags.iter())
        .map(|s| s.as_str())
        .collect();
    println!(
        "Compile: clang++ {} -O2 {} -L{} -lchelis_runtime -o {}",
        flags.join(" "),
        mm_path.display(),
        runtime_dir.display(),
        mm_path.with_extension("").display()
    );
    Ok(())
}

fn human_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;

    if bytes >= MIB as usize {
        format!("{:.2} MiB", bytes as f64 / MIB)
    } else if bytes >= KIB as usize {
        format!("{:.2} KiB", bytes as f64 / KIB)
    } else {
        format!("{bytes} B")
    }
}

fn run_tide(command: Option<TideCommand>) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        None => run_tide_repl(),
        Some(TideCommand::Serve { host, port }) => {
            let addr = SocketAddr::new(host, port);
            chelis_tide::http::serve_blocking(addr)?;
            Ok(())
        }
        Some(TideCommand::Mcp) => {
            chelis_tide::mcp::run_stdio_blocking()?;
            Ok(())
        }
        Some(TideCommand::Lsp { .. }) => {
            chelis_lsp::serve_stdio_blocking()?;
            Ok(())
        }
    }
}

fn cmd_cove(file: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    chelis_cove::run(chelis_cove::CoveOptions { file })?;
    Ok(())
}

fn run_tide_repl() -> Result<(), Box<dyn std::error::Error>> {
    println!("Chelis Tide v0.1 -- type expressions or definitions. Ctrl-D to exit.");
    let stdin = io::stdin();
    let mut accumulated_source = String::new();

    loop {
        print!("chelis> ");
        io::stdout().flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == ":quit" || trimmed == ":q" {
            break;
        }

        if trimmed.starts_with("def ")
            || trimmed.starts_with("macro ")
            || trimmed.starts_with("type ")
            || trimmed.contains('=')
        {
            accumulated_source.push_str(trimmed);
            accumulated_source.push('\n');
            println!("  defined.");
        } else {
            let eval_source = format!("{}\n__tide_result = {}", accumulated_source, trimmed);
            match try_eval(SourceKind::Surf, &eval_source, None) {
                Ok(result) => println!("= {result}"),
                Err(e) => eprintln!("error: {e}"),
            }
        }
    }
    Ok(())
}

/// Compile and evaluate `source`, returning the raw `EvalResult`. This
/// is the structured counterpart to [`try_eval`]: the latter formats
/// the same result into the human stdout rendering, while this returns
/// the result unchanged so the `--json` path can serialize it directly.
/// Error joining matches [`try_eval`] exactly, so JSON and text mode
/// surface identical error text on failure.
fn try_eval_result(
    source_kind: SourceKind,
    source: &str,
    selected_roots: Option<&[String]>,
) -> Result<chelis_compiler_api::schema::EvalResult, String> {
    let request = EvalRequest {
        source_kind,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    };
    if let Some(roots) = selected_roots {
        chelis_compiler_api::compiler::eval_selected(request, roots)
    } else {
        chelis_compiler_api::compiler::eval(request)
    }
    .map_err(|err| {
        err.errors
            .iter()
            .map(|diag| diag.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    })
}

fn try_eval(
    source_kind: SourceKind,
    source: &str,
    selected_roots: Option<&[String]>,
) -> Result<String, String> {
    let result = try_eval_result(source_kind, source, selected_roots)?;
    Ok(format_eval_result(&result))
}

/// Format an `EvalResult` into the shape `chelis eval --file` emits on
/// stdout: the transcript lines first, then either a single value (when
/// exactly one root) or `<name> = <value>` lines (when multiple). Shared
/// between the legacy `try_eval` path and the Phase H
/// `eval_in_context` path so the output is byte-identical for either
/// dispatch.
fn format_eval_result(result: &chelis_compiler_api::schema::EvalResult) -> String {
    // Each root carries its display text pre-rendered by the runtime's
    // single [05-OBS-1] renderer (compiler-api fills it where the dtype
    // tags still exist; the wire `ExecutionValue` cannot carry them). The
    // expect names that in-process contract: every EvalResult this CLI
    // formats comes straight from compiler-api, never from a deserialized
    // wire payload (chelis#732 Phase 1 - no second formatter may exist).
    let root_display = |root: &chelis_compiler_api::schema::EvaluatedRoot| -> String {
        root.display
            .clone()
            .expect("eval roots carry display text rendered in-process by compiler-api")
    };
    let mut lines = result.transcript.clone();
    if result.roots.len() == 1 {
        if let Some(root) = result.roots.first() {
            lines.push(root_display(root));
        }
        return lines.join("\n");
    }

    lines.extend(result.roots.iter().enumerate().map(|(index, root)| {
        let name = root.name.clone().unwrap_or_else(|| format!("_{index}"));
        format!("{} = {}", display_root_name(&name), root_display(root))
    }));
    lines.join("\n")
}

// NOTE (chelis#732 Phase 1): the CLI-side `format_execution_value` renderer
// was deleted here. It was a second hand-written formatting path for
// numeric payloads (faithful_observation.md section B2.4 forbids those) and
// it rendered from the wire `ExecutionValue`, which carries no dtype tags -
// so it could not be made [05-OBS-1]-faithful (bool/int tensor elements and
// scalar widths were unrecoverable). Labeled roots now consume the display
// text pre-rendered by the runtime's single renderer (see
// `format_eval_result` above and `EvaluatedRoot::display`).

fn checked_program_with_effects(
    deep_exprs: &[chelis_deep::ast::Expr],
) -> Result<chelis_types::CheckedProgram, String> {
    let checked = chelis_types::check_ir_program(deep_exprs)
        .map_err(|r| format!("Type errors: {:?}", r.errors))?;
    let checked =
        chelis_effects::check_program(&checked).map_err(|errors| format_effect_errors(&errors))?;
    chelis_types::check_linearity(&checked).map_err(|errors| format_type_errors(&errors))
}

/// Emit any sparse-helper summary rejections collected during host
/// lowering to stderr as advisory warnings. These do not fail the
/// build — the C/HIP backend already falls back to the helper
/// marshaling path for rejected callsites — but they tell the user
/// (and downstream tooling) which callsites missed the sparse-loop
/// inlining and why.
///
/// The W4-A acceptance oracle pattern-matches on the structured
/// `SummaryRejection` values directly (see
/// `crates/chelis-cli/tests/cross_library_semantic_gap_diagnostics.rs`);
/// this function is the human-readable rendering, not the matchable
/// contract surface.
fn emit_summary_rejections(host: Option<&chelis_ir::host::ConcreteHostProgram>) {
    let Some(host) = host else {
        return;
    };
    let rejections = chelis_ir::host::host_program_summary_rejections(host);
    if rejections.is_empty() {
        return;
    }
    for rejection in rejections {
        eprintln!("warning: {rejection}");
    }
}

fn expanded_desugared_program(
    decls: &[chelis_surf::ast::Decl],
) -> Result<Vec<chelis_deep::ast::Expr>, String> {
    let deep = chelis_surf::desugar::desugar_program(decls);
    chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
        .map(|expanded| expanded.into_exprs())
        .map_err(|err| err.to_string())
}

fn lower_checked_for_cli(
    checked: &chelis_types::CheckedProgram,
    host_program: Option<&chelis_ir::host::ConcreteHostProgram>,
) -> Result<chelis_ir::Dag, Box<dyn std::error::Error>> {
    match chelis_ir::lower::try_lower_program(checked) {
        Ok(dag) => Ok(dag),
        // Issue #197: a *fatal* lowering diagnostic (the AD-rejection
        // path for `grad` over a non-differentiable op) must propagate
        // even when the host program would otherwise be able to take
        // over. The host fallback emits an unresolved call to the
        // grad-function symbol; we must surface the AD-rejection text
        // instead so the user sees `floor is non-differentiable
        // (piecewise constant)` rather than a compile-clean build
        // that fails at gcc-link time.
        Err(diagnostic)
            if !diagnostic.fatal
                && host_program
                    .map(chelis_ir::host::host_program_requires_host_backend)
                    .unwrap_or(false) =>
        {
            Ok(chelis_ir::Dag::new())
        }
        Err(diagnostic) => Err(boxed_string_error(diagnostic.to_string())),
    }
}

fn boxed_string_error(message: String) -> Box<dyn std::error::Error> {
    message.into()
}

fn format_effect_errors(errors: &[chelis_effects::EffectError]) -> String {
    errors
        .iter()
        .map(|error| error.message.clone())
        .collect::<Vec<_>>()
        .join("; ")
}

fn format_type_errors(errors: &[chelis_types::errors::CheckError]) -> String {
    errors
        .iter()
        .map(|error| error.message.clone())
        .collect::<Vec<_>>()
        .join("; ")
}

fn fallback_symbolic_dims(
    dag: &chelis_ir::dag::Dag,
    preferred: &[String],
    hint: &[String],
) -> Vec<String> {
    if !preferred.is_empty() {
        preferred.to_vec()
    } else if !hint.is_empty() {
        hint.to_vec()
    } else {
        // chelis#616: only Load-bound dims are caller-suppliable;
        // op-declared dims are computed at run time by their owning op.
        chelis_ir::dag::symbolic_params(dag)
    }
}

fn lowered_root_names_from_decls(
    decls: &[Decl],
    program_exprs: &[DeepExpr],
    type_env: &HashMap<String, DeepExpr>,
) -> Vec<String> {
    let deep_exprs = chelis_surf::desugar::desugar_program(decls);
    lowered_root_names_from_selected_exprs(&deep_exprs, program_exprs, type_env)
}

fn root_names_from_decls(decls: &[Decl], type_env: &HashMap<String, DeepExpr>) -> Vec<String> {
    let deep_exprs = chelis_surf::desugar::desugar_program(decls);
    root_names_from_exprs(&deep_exprs, type_env)
}

fn lowered_root_names_from_exprs(
    exprs: &[DeepExpr],
    type_env: &HashMap<String, DeepExpr>,
) -> Vec<String> {
    lowered_root_names_from_selected_exprs(exprs, exprs, type_env)
}

fn lowered_root_names_from_selected_exprs(
    selected_exprs: &[DeepExpr],
    program_exprs: &[DeepExpr],
    type_env: &HashMap<String, DeepExpr>,
) -> Vec<String> {
    let mut out = Vec::new();
    for expr in selected_exprs {
        collect_lowered_root_names_from_expr(expr, program_exprs, type_env, &mut out);
    }
    out
}

fn root_names_from_exprs(exprs: &[DeepExpr], type_env: &HashMap<String, DeepExpr>) -> Vec<String> {
    let mut out = Vec::new();
    for expr in exprs {
        collect_root_names_from_expr(expr, type_env, &mut out);
    }
    out
}

fn collect_lowered_root_names_from_expr(
    expr: &DeepExpr,
    program_exprs: &[DeepExpr],
    type_env: &HashMap<String, DeepExpr>,
    out: &mut Vec<String>,
) {
    let DeepExpr::List(list, _) = expr else {
        return;
    };
    match list.tag() {
        Some(DeepTag::Module) => {
            for child in list.elements.iter().skip(3) {
                collect_lowered_root_names_from_expr(child, program_exprs, type_env, out);
            }
        }
        _ => {
            let Some(name) = deep_top_level_expr_name(expr) else {
                return;
            };
            if type_env.get(name).is_some_and(type_expr_is_function) {
                return;
            }
            if chelis_ir::lower::top_level_expr_is_lowered(expr, program_exprs, type_env) {
                extend_root_names_from_value(
                    name,
                    type_env.get(name),
                    top_level_def_body(expr),
                    out,
                );
            }
        }
    }
}

fn collect_root_names_from_expr(
    expr: &DeepExpr,
    type_env: &HashMap<String, DeepExpr>,
    out: &mut Vec<String>,
) {
    let DeepExpr::List(list, _) = expr else {
        return;
    };
    match list.tag() {
        Some(DeepTag::Module) => {
            for child in list.elements.iter().skip(3) {
                collect_root_names_from_expr(child, type_env, out);
            }
        }
        _ => {
            let Some(name) = deep_top_level_expr_name(expr) else {
                return;
            };
            if type_env.get(name).is_some_and(type_expr_is_function) {
                return;
            }
            extend_root_names_from_value(name, type_env.get(name), top_level_def_body(expr), out);
        }
    }
}

fn deep_top_level_expr_name(expr: &DeepExpr) -> Option<&str> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    match (list.tag(), list.elements.get(2)) {
        (Some(DeepTag::Def), Some(DeepExpr::Atom(DeepAtom::Symbol(name), _))) => {
            Some(name.as_str())
        }
        _ => None,
    }
}

fn deep_named_decl_name(expr: &DeepExpr) -> Option<&str> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    match (list.tag(), list.elements.get(2)) {
        (Some(DeepTag::Def | DeepTag::Defsig), Some(DeepExpr::Atom(DeepAtom::Symbol(name), _))) => {
            Some(name.as_str())
        }
        _ => None,
    }
}

/// Host builtins that the IR evaluator (`chelis eval` / `chelis test`)
/// supports but the compiled build backends deliberately do not. Kept in
/// one place so [`reject_eval_only_builtins_host`] and
/// [`drop_unreachable_eval_only_defs`] stay in agreement.
const EVAL_ONLY_HOST_BUILTINS: &[&str] = &[
    "process_run",
    // Host-lane JSON I/O (chelis#890): eval-only by scope — the compiled
    // backends have no Json ADT value representation and emitting the
    // host_emit catch-all for them would produce a silent wrong value
    // (the chelis#734 class), so the build gate rejects them loudly.
    "parse_json",
    "to_json",
    "json_f64",
    "json_str",
    "json_list",
    "json_f64s",
    "jnum",
    "jstr",
    "jlist",
    "jdict",
    "json_set",
    "round_to",
];

/// Drop top-level decls for any function whose body references an eval-only
/// host builtin ([`EVAL_ONLY_HOST_BUILTINS`]) and is not reachable from the
/// entry program. Such functions can never be lowered into a compiled
/// artifact, so an unused transitive dependency module (e.g. chelis-std's
/// `Std.Process`) must not drag them into the build's lowering target. Both
/// the `def` body and its sibling `defsig` are removed by name. A reachable
/// eval-only use is preserved so the build gate still rejects it. chelis#334.
fn drop_unreachable_eval_only_defs(
    exprs: Vec<DeepExpr>,
    entry_exprs: &[DeepExpr],
) -> Vec<DeepExpr> {
    use std::collections::HashSet;

    let reachable = prune_build_program_to_reachable_defs(&exprs, entry_exprs)
        .iter()
        .filter_map(|expr| deep_named_decl_name(expr).map(str::to_string))
        .collect::<HashSet<_>>();

    // Names of unreachable functions whose body uses an eval-only builtin.
    // Collected first so both the `def` and its `defsig` are dropped.
    let drop_names = exprs
        .iter()
        .filter_map(|expr| {
            let name = deep_named_decl_name(expr)?;
            if reachable.contains(name) {
                return None;
            }
            deep_referenced_vars(expr)
                .iter()
                .any(|var| EVAL_ONLY_HOST_BUILTINS.contains(var))
                .then(|| name.to_string())
        })
        .collect::<HashSet<_>>();

    exprs
        .into_iter()
        .filter(|expr| {
            deep_named_decl_name(expr)
                .map(|name| !drop_names.contains(name))
                .unwrap_or(true)
        })
        .collect()
}

fn prune_build_program_to_reachable_defs(
    exprs: &[DeepExpr],
    entry_exprs: &[DeepExpr],
) -> Vec<DeepExpr> {
    // Delegate to the shared reachable-defs pruner (single source of truth in
    // chelis-compiler-api, also used by the WI-3 graph-extraction producer).
    // The build path seeds reachability from the entry program's top-level
    // def names; an EMPTY seed set drops every named decl, which this path
    // relies on (see `prune_to_reachable_seeds`).
    let seeds = entry_exprs
        .iter()
        .filter_map(deep_top_level_expr_name)
        .map(str::to_string)
        .collect::<Vec<_>>();
    chelis_compiler_api::prune::prune_to_reachable_seeds(exprs.to_vec(), seeds)
}

/// Every `var` reference name in `expr`. Delegates to the shared traversal in
/// chelis-compiler-api so the build path and the WI-3 producer agree on what
/// "references" means.
fn deep_referenced_vars(expr: &DeepExpr) -> Vec<&str> {
    chelis_compiler_api::prune::deep_referenced_vars(expr)
}

fn host_display_root_name(full_name: &str, entry_root_names: &[String]) -> Option<String> {
    entry_root_names.iter().find_map(|entry| {
        (full_name == entry
            || full_name
                .rsplit_once("__")
                .is_some_and(|(_, tail)| tail == entry)
            || full_name
                .rsplit_once('.')
                .is_some_and(|(_, tail)| tail == entry))
        .then(|| entry.clone())
    })
}

/// For a tuple-typed binding, the eval root-name expander produces
/// `<name>.0`, `<name>.1`, … entries (one per tuple field). The C emit
/// path stores a single global per binding, so we surface a tuple-prefix
/// display name (e.g. `"buckets"`) when at least one expanded entry
/// references this binding's terminal name. The C emitter detects the
/// `Tuple(_)` host type on the global and renders one labeled line per
/// field, mirroring eval's output shape.
fn host_display_tuple_root_prefix(full_name: &str, entry_root_names: &[String]) -> Option<String> {
    let terminal = full_name
        .rsplit_once("__")
        .map(|(_, tail)| tail)
        .unwrap_or(full_name);
    let prefix_dot = format!("{terminal}.");
    entry_root_names
        .iter()
        .any(|entry| entry.starts_with(&prefix_dot))
        .then(|| terminal.to_string())
}

fn top_level_def_body(expr: &DeepExpr) -> Option<&DeepExpr> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    (list.tag() == Some(DeepTag::Def))
        .then(|| list.elements.get(3))
        .flatten()
}

fn extend_root_names_from_value(
    name: &str,
    ty: Option<&DeepExpr>,
    value: Option<&DeepExpr>,
    out: &mut Vec<String>,
) {
    if let Some(DeepExpr::List(list, _)) = ty
        && let Some(tag) = list.tag()
    {
        if tag == DeepTag::TFn {
            extend_root_names_from_value(name, list.elements.last(), None, out);
            return;
        }
        if tag == DeepTag::TTuple {
            for (index, child) in list.elements.iter().skip(2).enumerate() {
                extend_root_names_from_value(&format!("{name}.{index}"), Some(child), None, out);
            }
            return;
        }
    }
    if let Some(DeepExpr::List(list, _)) = value
        && (list.tag() == Some(DeepTag::Tuple))
    {
        for (index, child) in list.elements.iter().skip(2).enumerate() {
            extend_root_names_from_value(
                &format!("{name}.{index}"),
                expr_type_metadata(child),
                Some(child),
                out,
            );
        }
        return;
    }
    out.push(name.to_string());
}

fn expr_type_metadata(expr: &DeepExpr) -> Option<&DeepExpr> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    match list.elements.get(1) {
        Some(DeepExpr::Map(meta, _)) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        _ => None,
    }
}

fn display_root_name(name: &str) -> String {
    let (base, suffix) = if let Some((base, suffix)) = name.rsplit_once('.')
        && suffix.chars().all(|ch| ch.is_ascii_digit())
    {
        (base, Some(suffix))
    } else {
        (name, None)
    };
    let short = base
        .rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| base.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(base);
    match suffix {
        Some(suffix) => format!("{short}.{suffix}"),
        None => short.to_string(),
    }
}

fn type_expr_is_function(expr: &DeepExpr) -> bool {
    matches!(expr, DeepExpr::List(list, _) if (list.tag() == Some(DeepTag::TFn)))
}

fn collect_symbolic_dims_from_deep(exprs: &[chelis_deep::ast::Expr]) -> Vec<String> {
    let mut dims = Vec::<String>::new();
    for expr in exprs {
        collect_symbolic_dims_expr(expr, &mut dims);
    }
    dims.sort();
    dims.dedup();
    dims
}

fn collect_symbolic_dims_expr(expr: &chelis_deep::ast::Expr, dims: &mut Vec<String>) {
    match expr {
        chelis_deep::ast::Expr::List(list, _) => {
            if list.tag() == Some(DeepTag::DName)
                && let Some(chelis_deep::ast::Expr::Atom(chelis_deep::ast::Atom::Symbol(name), _)) =
                    list.elements.get(2)
                && name != "*"
            {
                dims.push(name.clone());
            }
            for child in &list.elements {
                collect_symbolic_dims_expr(child, dims);
            }
        }
        chelis_deep::ast::Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                collect_symbolic_dims_expr(value, dims);
            }
        }
        chelis_deep::ast::Expr::MetaExpr(meta, _) => {
            collect_symbolic_dims_expr(&meta.expr, dims);
            for (_, value) in &meta.entries {
                collect_symbolic_dims_expr(value, dims);
            }
        }
        chelis_deep::ast::Expr::Atom(_, _) => {}
    }
}

/// Run the typed pipeline against a Surf source string (parse, desugar,
/// macro-expand, type-check, effect-check, linearity-check) and return
/// `true` iff every stage accepts.
///
/// Used by the lint fix driver to gate auto-fixes from rules that opt in
/// via `Rule::fix_requires_typed_pipeline_check()` — currently only
/// `redundant-linearity-call`. The spec safety bar in
/// `spec/01-nomenclature.md` §12 names this exact pipeline: "the proof
/// requires the type and linearity pipeline, not source-text matching".
///
/// Architectural rationale in
/// `docs/investigations/redundant_linearity_autofix_architecture.md`
/// (Path 1B).
fn typed_pipeline_accepts_surf(source: &str) -> bool {
    let Ok(decls) = chelis_surf::parser::parse_str(source) else {
        return false;
    };
    let Ok(deep_exprs) = expanded_desugared_program(&decls) else {
        return false;
    };
    let report = chelis_types::check_ir_fitness(&deep_exprs);
    if !report.errors.is_empty() {
        return false;
    }
    let Ok(typed_program) = chelis_types::check_typed_program(&deep_exprs) else {
        return false;
    };
    let Ok(effect_checked) = chelis_effects::check_program(&typed_program) else {
        return false;
    };
    chelis_types::check_linearity(&effect_checked).is_ok()
}

/// `chelis lint` — naming-convention lint per `spec/01-nomenclature.md`.
///
/// Walks each path in `paths` (default: `.`), dispatches to every registered
/// rule, prints violations as `path:line:col: rule (§ref): message`, and
/// returns an exit code. With `--check`, exit nonzero on any violation
/// (CI gate); otherwise exit 0 even when violations exist (informational).
fn cmd_lint(
    paths: Vec<PathBuf>,
    check: bool,
    fix: bool,
    list: bool,
    rule_filter: Option<&str>,
    rules_filter: Option<&str>,
) -> Result<i32, Box<dyn std::error::Error>> {
    if rule_filter.is_some() && rules_filter.is_some() {
        return Err("use either --rule or --rules, not both".into());
    }
    let mut rules = chelis_lint::registry::selectable_rules();
    if list {
        for rule in &rules {
            println!(
                "{}\t{}\t{}\t{}",
                rule.id(),
                rule.severity().as_str(),
                rule.spec_ref(),
                rule.summary()
            );
        }
        return Ok(0);
    }
    if let Some(id) = rule_filter {
        rules.retain(|r| r.id() == id);
        if rules.is_empty() {
            return Err(format!("no rule with id '{id}'").into());
        }
    }
    if let Some(ids) = rules_filter {
        let selected: HashSet<&str> = ids
            .split(',')
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .collect();
        if selected.is_empty() {
            return Err("--rules requires at least one rule id".into());
        }
        rules.retain(|r| selected.contains(r.id()));
        let found: HashSet<&str> = rules.iter().map(|r| r.id()).collect();
        let missing: Vec<&str> = selected.difference(&found).copied().collect();
        if !missing.is_empty() {
            return Err(format!("no rule with id(s) '{}'", missing.join(",")).into());
        }
    }
    let raw_targets = if paths.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        paths
    };
    // Normalize each user-supplied target to an absolute path before
    // passing it to the lint walker. Several rules classify files by
    // matching substrings against the path (e.g.,
    // `doc-filename-convention` looks for `/docs/`, `/spec/`,
    // `/spec/design/` in `path.to_string_lossy()`). Without
    // normalization, the same file tree produces different violations
    // depending on whether the user typed `chelis lint --check .` or
    // `chelis lint --check docs/`: the walker prefixes yielded paths
    // with the literal target argument, so `.` yields `./docs/...`
    // (substring `/docs/` matches) while `docs/` yields `docs/...` (no
    // leading slash, no match). Absolutizing at the CLI boundary
    // unifies the two walks and forecloses the bug class for any future
    // rule that does path-segment dispatch.
    //
    // Non-link targets are then canonicalized so exception matching
    // strips the same real-path prefix `detect_lint_workspace_root`
    // reports (macOS tempdirs spell `/var/...` for `/private/var/...`).
    // A target whose final component is a symlink is deliberately NOT
    // canonicalized: resolving it would erase the link's identity before
    // the traversal policy's depth-zero boundary check can see it — an
    // explicitly named link escaping the repository policy root would
    // lint its resolved external tree as a loose target instead of
    // failing loudly (§12.2).
    let targets: Vec<PathBuf> = raw_targets
        .into_iter()
        .map(|p| {
            let absolute = match std::path::absolute(&p) {
                Ok(absolute) => absolute,
                Err(err) => {
                    eprintln!(
                        "warning: failed to absolutize {}: {err}; using as-is",
                        p.display()
                    );
                    p
                }
            };
            // A trailing separator makes POSIX `lstat` dereference a
            // final-component symlink (the slash asserts "directory",
            // forcing resolution), so `is_symlink()` would report false
            // for `link/` and the escaping-link rejection would be
            // bypassed by a one-character spelling. `components()` drops
            // the trailing separator; probe and walk the stripped form.
            let absolute: PathBuf = absolute.components().collect();
            match std::fs::symlink_metadata(&absolute) {
                Ok(metadata) if !metadata.file_type().is_symlink() => {
                    std::fs::canonicalize(&absolute).unwrap_or(absolute)
                }
                _ => absolute,
            }
        })
        .collect();
    // Exception patterns in `style_gate::exceptions()` are written
    // workspace-relative (e.g.,
    // `crates/chelis-surf/tests/fixtures/*.ch`,
    // `docs/book/src/*.md`). They must therefore be matched against
    // workspace-relative paths, not against paths relative to a
    // per-target walk root. When the CLI walks a sub-directory
    // (e.g., `chelis lint --check crates`), the walk root is a
    // sub-directory of the workspace root; using it for
    // prefix-stripping in `apply_exceptions` drops the leading
    // workspace-relative segments (here, `crates/`) and the exception
    // silently fails to match.
    //
    // The workspace root is detected with `cargo locate-project
    // --workspace` (`style_gate::detect_lint_workspace_root`), Cargo's
    // own canonical workspace-locating probe, which resolves the root
    // correctly from any subdirectory and from absolute-path
    // invocations issued outside the workspace. This replaces the
    // earlier `canonicalize(cwd)` shortcut, which silently broke
    // workspace-rooted exceptions whenever the CLI was not invoked
    // from the workspace root. Closes `Lint-ExceptionPathRoot-F1` and
    // `Lint-WorkspaceRootCwdAssumption-F1`.
    //
    // Probe from the first target's directory (not the process CWD):
    // the targets are already canonicalized absolute paths, so for a
    // directory target the target itself is the probe dir, and for a
    // file target the parent is. This makes `chelis lint --check
    // /abs/workspace` resolve correctly regardless of the invoking
    // shell's working directory.
    //
    // If detection fails the targets are not inside a Cargo workspace,
    // so no workspace-rooted exception glob can legitimately apply
    // (a path outside the workspace is not, e.g.,
    // `crates/chelis-surf/tests/fixtures/*.ch`). Linting a loose file
    // outside any workspace is a supported operation, so the
    // violations pass through unfiltered rather than aborting `lint`.
    // This is the same uniform policy the advisory-emit path and the
    // build-time style gate apply: one detection mechanism, and a
    // detection failure means "no workspace, no workspace-anchored
    // exceptions", not a second detection mechanism with different
    // semantics.
    let probe_dir = targets
        .first()
        .map(|first| {
            if first.is_dir() {
                first.clone()
            } else {
                first
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from("."))
            }
        })
        .unwrap_or_else(|| PathBuf::from("."));
    let workspace_root = style_gate::detect_lint_workspace_root(&probe_dir).ok();
    // The exception list is sourced from `style_gate::exceptions()` so
    // the standalone `chelis lint` subcommand and the build-time style
    // gate filter against one shared registry. Path-glob entries with
    // §-cross-refs go here.
    let exceptions: Vec<chelis_lint::Exception> = style_gate::exceptions();
    // Bucket the rendered violation lines by severity instead of
    // printing them inline as targets are walked. A workspace `chelis
    // lint --check .` can emit several hundred advisory lines (e.g.
    // `prefer-pipe-operator` across packages/chelis-std/tests/*); when
    // a handful of blocking ERROR lines are interleaved into that
    // stream they are effectively invisible, and CI failure debugging
    // misreads the cause. The em-dash §8.6 rule has been bitten by
    // exactly this twice. Buckets let the blocking errors be printed
    // last, under a distinct header, so they are the final thing in
    // the CI log. See
    // docs/investigations/test_toolchain_guards_design.md.
    let mut error_lines: Vec<String> = Vec::new();
    let mut warning_lines: Vec<String> = Vec::new();
    let mut advisory_lines: Vec<String> = Vec::new();
    for target in &targets {
        if fix {
            let applied = apply_lint_fixes(target, workspace_root.as_deref(), &rules, &exceptions)?;
            if applied > 0 {
                println!(
                    "fixed {} replacement(s) under {}",
                    applied,
                    target.display()
                );
            }
        }
        let raw_violations = chelis_lint::lint(target, &rules)?;
        let kept = apply_exceptions_opt(&raw_violations, &exceptions, workspace_root.as_deref());
        for v in &kept {
            // V2-F3 (PR #58): suppress warnings for rules that opt in
            // to `check_mirrors_fix` when the autofix would silently
            // decline (or be rejected by the typed-pipeline gate).
            // Without this filter, `chelis lint --fix` is
            // non-convergent for those rules: the warning fires, the
            // autofix declines, and the next `--check` run fires the
            // same warning again. Inline `keep` directives are
            // honored separately by `should_suppress_unfixable_violation`.
            if should_suppress_unfixable_violation(target, &rules, v) {
                continue;
            }
            let severity = rule_severity(&rules, &v.rule_id);
            let suffix = if fix_available_for_violation(target, &rules, v) {
                " [fix]"
            } else {
                ""
            };
            match severity {
                chelis_lint::Severity::Error => error_lines.push(format!("{v}{suffix}")),
                chelis_lint::Severity::Warning => {
                    warning_lines.push(format!("warning: {v}{suffix}"))
                }
                chelis_lint::Severity::Advisory => {
                    advisory_lines.push(format!("advisory: {v}{suffix}"))
                }
            }
        }
    }
    // Print advisory and warning lines first (the bulk, non-blocking
    // noise), then the blocking errors last under a delimited header
    // and a summary count. Severity::blocks_check() decides what is
    // blocking; today that is exactly Severity::Error.
    for line in &advisory_lines {
        println!("{line}");
    }
    for line in &warning_lines {
        println!("{line}");
    }
    let blocking_total = error_lines.len();
    if blocking_total > 0 {
        println!("=== {blocking_total} blocking lint error(s) ===");
        for line in &error_lines {
            println!("{line}");
        }
        println!(
            "lint --check failed: {blocking_total} blocking error(s) above \
             (advisory/warning lines, if any, are non-blocking)"
        );
    }
    if check && blocking_total > 0 {
        Ok(1)
    } else {
        Ok(0)
    }
}

fn rule_severity(rules: &[Box<dyn chelis_lint::Rule>], id: &str) -> chelis_lint::Severity {
    rules
        .iter()
        .find(|rule| rule.id() == id)
        .map(|rule| rule.severity())
        .unwrap_or(chelis_lint::Severity::Error)
}

/// Apply the workspace-rooted exception list when a workspace root was
/// detected; otherwise pass the raw violations through unchanged.
///
/// Exception globs in `style_gate::exceptions()` are authored
/// workspace-root relative, so they can only match when there is a
/// workspace root to anchor against. When
/// `style_gate::detect_lint_workspace_root` fails (the targets are not
/// inside a Cargo workspace), no workspace-rooted exception can
/// legitimately apply, so the raw violations are the correct result.
/// This keeps `cmd_lint`, the advisory-emit path, and the style gate
/// on one uniform policy without a second detection mechanism.
fn apply_exceptions_opt(
    violations: &[chelis_lint::Violation],
    exceptions: &[chelis_lint::Exception],
    workspace_root: Option<&Path>,
) -> Vec<chelis_lint::Violation> {
    match workspace_root {
        Some(root) => chelis_lint::exceptions::apply_exceptions(violations, exceptions, root),
        None => violations.to_vec(),
    }
}

fn apply_lint_fixes(
    target: &Path,
    workspace_root: Option<&Path>,
    rules: &[Box<dyn chelis_lint::Rule>],
    exceptions: &[chelis_lint::Exception],
) -> Result<usize, Box<dyn std::error::Error>> {
    let mut total = 0usize;
    for _ in 0..5 {
        let raw = chelis_lint::lint(target, rules)?;
        let kept = apply_exceptions_opt(&raw, exceptions, workspace_root);
        let mut by_path: BTreeMap<PathBuf, Vec<chelis_lint::Violation>> = BTreeMap::new();
        for violation in kept {
            by_path
                .entry(violation.path.clone())
                .or_default()
                .push(violation);
        }
        let mut pass_total = 0usize;
        for (path, violations) in by_path {
            let source = match fs::read_to_string(&path) {
                Ok(source) => source,
                Err(_) => continue,
            };
            let Some(surface) = chelis_lint::Surface::classify(&path, false) else {
                continue;
            };
            let ctx = chelis_lint::Context {
                root: target,
                path: &path,
                source: Some(&source),
                surface,
            };
            let mut replacements: Vec<(chelis_lint::Replacement, bool)> = Vec::new();
            for violation in &violations {
                let Some(rule) = rules.iter().find(|rule| rule.id() == violation.rule_id) else {
                    continue;
                };
                if violation.line.is_some_and(|line| {
                    chelis_lint::inline_keeps(&source, surface, line, violation.rule_id.as_str())
                }) {
                    continue;
                }
                if let Some(replacement) = rule.fix(&ctx, violation) {
                    replacements.push((replacement, rule.fix_requires_typed_pipeline_check()));
                }
            }
            replacements.sort_by_key(|(replacement, _)| {
                (
                    replacement.end.saturating_sub(replacement.start),
                    replacement.start,
                )
            });
            let mut filtered: Vec<(chelis_lint::Replacement, bool)> = Vec::new();
            for (replacement, needs_check) in replacements {
                if replacement.start > replacement.end
                    || filtered.iter().any(|(kept, _)| {
                        replacement.start < kept.end && kept.start < replacement.end
                    })
                {
                    continue;
                }
                filtered.push((replacement, needs_check));
            }
            if filtered.is_empty() {
                continue;
            }
            filtered.sort_by_key(|(replacement, _)| replacement.start);

            // Per-replacement typed-pipeline gate for rules that opted in
            // (Path 1B per
            // docs/investigations/redundant_linearity_autofix_architecture.md).
            // We test each verification-required replacement independently
            // by applying it to the original source and running the typed
            // pipeline. Independent verification preserves the maximum set
            // of safe rewrites: one unsafe strip does not block the others.
            let surface_eligible_for_typed_check =
                matches!(surface, chelis_lint::Surface::SurfSource);
            let mut accepted: Vec<chelis_lint::Replacement> = Vec::new();
            for (replacement, needs_check) in filtered {
                if needs_check && surface_eligible_for_typed_check {
                    let mut candidate = source.clone();
                    candidate.replace_range(replacement.start..replacement.end, &replacement.text);
                    if !typed_pipeline_accepts_surf(&candidate) {
                        // Drop this replacement silently; the lint warning
                        // remains so the user sees the still-flagged copy().
                        continue;
                    }
                }
                accepted.push(replacement);
            }
            if accepted.is_empty() {
                continue;
            }
            let mut edited = source;
            for replacement in accepted.iter().rev() {
                edited.replace_range(replacement.start..replacement.end, &replacement.text);
            }
            fs::write(&path, edited)?;
            pass_total += accepted.len();
        }
        total += pass_total;
        if pass_total == 0 {
            break;
        }
    }
    Ok(total)
}

fn fix_available_for_violation(
    target: &Path,
    rules: &[Box<dyn chelis_lint::Rule>],
    violation: &chelis_lint::Violation,
) -> bool {
    let Some(rule) = rules.iter().find(|rule| rule.id() == violation.rule_id) else {
        return false;
    };
    let Ok(source) = fs::read_to_string(&violation.path) else {
        return false;
    };
    let Some(surface) = chelis_lint::Surface::classify(&violation.path, false) else {
        return false;
    };
    if violation.line.is_some_and(|line| {
        chelis_lint::inline_keeps(&source, surface, line, violation.rule_id.as_str())
    }) {
        return false;
    }
    fix_would_apply_for_violation(target, rule.as_ref(), &source, surface, violation)
}

/// Does the rule propose a safe fix for `violation` against `source` —
/// ignoring any `keep` directive on the violation's line?
///
/// `fix_available_for_violation` returns `false` when an inline `keep`
/// directive suppresses the rewrite; that's the correct gate for
/// printing the `[fix]` marker (no marker on kept-on-purpose
/// violations) and for `apply_lint_fixes` (no rewrite on kept
/// violations). The CLI's `check_mirrors_fix` warning suppression
/// (V2-F3 / PR #58) needs to distinguish "fix unavailable because the
/// user said keep" (warning should still print) from "fix unavailable
/// because the rule declined or the typed-pipeline gate rejected"
/// (warning should be suppressed; `--fix` is non-convergent
/// otherwise). This helper exposes the latter predicate.
fn fix_would_apply_for_violation(
    target: &Path,
    rule: &dyn chelis_lint::Rule,
    source: &str,
    surface: chelis_lint::Surface,
    violation: &chelis_lint::Violation,
) -> bool {
    let ctx = chelis_lint::Context {
        root: target,
        path: &violation.path,
        source: Some(source),
        surface,
    };
    let Some(replacement) = rule.fix(&ctx, violation) else {
        return false;
    };
    if rule.fix_requires_typed_pipeline_check()
        && matches!(surface, chelis_lint::Surface::SurfSource)
    {
        let mut candidate = source.to_string();
        if replacement.start > candidate.len() || replacement.end > candidate.len() {
            return false;
        }
        candidate.replace_range(replacement.start..replacement.end, &replacement.text);
        return typed_pipeline_accepts_surf(&candidate);
    }
    true
}

/// Should the CLI suppress this violation's warning because the rule
/// opted in to `check_mirrors_fix` and the autofix would silently
/// decline (or be rejected by the typed-pipeline gate)?
///
/// V2-F3 (PR #58): a rule whose warning is only actionable when
/// paired with a safe rewrite should not surface the warning when no
/// rewrite is on offer; otherwise `chelis lint --fix` is
/// non-convergent for that rule. The CLI applies this filter at the
/// final warning-emit path. Explicit `keep` directives are honored:
/// the user has opted to preserve the source and see the warning, so
/// suppression does not apply.
fn should_suppress_unfixable_violation(
    target: &Path,
    rules: &[Box<dyn chelis_lint::Rule>],
    violation: &chelis_lint::Violation,
) -> bool {
    let Some(rule) = rules.iter().find(|rule| rule.id() == violation.rule_id) else {
        return false;
    };
    if !rule.check_mirrors_fix() {
        return false;
    }
    let Ok(source) = fs::read_to_string(&violation.path) else {
        return false;
    };
    let Some(surface) = chelis_lint::Surface::classify(&violation.path, false) else {
        return false;
    };
    if violation.line.is_some_and(|line| {
        chelis_lint::inline_keeps(&source, surface, line, violation.rule_id.as_str())
    }) {
        // User explicitly asked to keep this occurrence; the warning
        // continues to fire (without a `[fix]` marker) so the user
        // can see the diagnostic they pinned.
        return false;
    }
    !fix_would_apply_for_violation(target, rule.as_ref(), &source, surface, violation)
}

#[cfg(test)]
mod eval_only_pruning_tests {
    use super::{
        deep_named_decl_name, drop_unreachable_eval_only_defs, expanded_desugared_program,
    };

    fn desugar(src: &str) -> Vec<chelis_deep::ast::Expr> {
        let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
        expanded_desugared_program(&decls).expect("desugar")
    }

    /// chelis#334: a pure-tensor entry program plus an unused library def
    /// that uses the eval-only `process_run` builtin (the shape of
    /// chelis-std's `Std.Process`). The dead eval-only def must be dropped
    /// so the build gate does not reject a program that never reaches it.
    #[test]
    fn drops_unreachable_eval_only_def_but_keeps_entry() {
        let full = desugar(
            "def unused_runner(cmd: string, args: List[string]) -> (int64, string, string) = process_run(cmd, args)\n\
             def main(x: tensor[2, 2, f32], w: tensor[2, 2, f32]) -> tensor[2, 2, f32] = matmul(&x, &w)\n",
        );
        let entry = desugar(
            "def main(x: tensor[2, 2, f32], w: tensor[2, 2, f32]) -> tensor[2, 2, f32] = matmul(&x, &w)\n",
        );
        let kept = drop_unreachable_eval_only_defs(full, &entry);
        let names: Vec<&str> = kept.iter().filter_map(deep_named_decl_name).collect();
        assert!(
            names.contains(&"main"),
            "entry `main` must survive: {names:?}"
        );
        assert!(
            !names.contains(&"unused_runner"),
            "unreachable eval-only def must be dropped: {names:?}"
        );
    }

    /// Negative parity: a *reachable* eval-only use is preserved so the
    /// build gate still rejects it with a clean diagnostic instead of the
    /// program silently building with a missing function.
    #[test]
    fn keeps_reachable_eval_only_def_for_the_gate() {
        let exprs =
            desugar("def main() -> (int64, string, string) = process_run(\"echo\", [\"hi\"])\n");
        let entry = exprs.clone();
        let kept = drop_unreachable_eval_only_defs(exprs, &entry);
        let names: Vec<&str> = kept.iter().filter_map(deep_named_decl_name).collect();
        assert!(
            names.contains(&"main"),
            "reachable eval-only def must be preserved for the build gate: {names:?}"
        );
    }
}

/// chelis#616: the CLI reject seams must turn a runtime (node-valued)
/// movement bound / reshape target into a clean diagnostic on the non-C
/// targets, never a backend panic. Kept in lockstep with the compiler-api
/// seam (`chelis-compiler-api::compiler::reject_unsupported_hip_ops`).
#[cfg(test)]
mod runtime_dim_reject_tests {
    use super::{reject_unsupported_hip_ops, reject_unsupported_metal_ops};
    use chelis_ir::dag::{Dag, RiscOp, RtDim, TensorType};
    use chelis_types::types::Prim;

    fn ty(dims: Vec<chelis_ir::dag::DimInfo>, precision: Prim) -> TensorType {
        TensorType { dims, precision }
    }

    fn lit_dims(sizes: &[usize]) -> Vec<chelis_ir::dag::DimInfo> {
        sizes
            .iter()
            .map(|n| chelis_ir::dag::DimInfo::Lit(*n))
            .collect()
    }

    /// Load x: [4] f32 plus a rank-0 int32 Load scalar (not a `Shape` read;
    /// the HIP seam blanket-rejects `Shape` first and these tests must
    /// exercise the movement/reshape arms).
    fn dag_with_scalar() -> (Dag, chelis_ir::dag::NodeId, chelis_ir::dag::NodeId) {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(lit_dims(&[4]), Prim::F32),
            None,
        );
        let m = dag.add_node(
            RiscOp::Load { name: "m".into() },
            vec![],
            ty(lit_dims(&[]), Prim::Int32),
            None,
        );
        (dag, x, m)
    }

    #[test]
    fn hip_seam_rejects_node_valued_reshape_target() {
        let (mut dag, x, m) = dag_with_scalar();
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Node(1)],
            },
            vec![x, m],
            ty(lit_dims(&[4]), Prim::F32),
            None,
        );
        let err = reject_unsupported_hip_ops(&dag)
            .expect_err("HIP seam must reject a node-valued reshape target");
        let message = err.to_string();
        assert!(
            message.contains("reshape") && message.contains("--target c"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn metal_seam_rejects_node_valued_shrink_bound() {
        let (mut dag, x, m) = dag_with_scalar();
        dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::Node(1))],
            },
            vec![x, m],
            ty(lit_dims(&[4]), Prim::F32),
            None,
        );
        let err = reject_unsupported_metal_ops(&dag)
            .expect_err("Metal seam must reject a node-valued shrink bound, not panic later");
        let message = err.to_string();
        assert!(
            message.contains("--target c") && message.contains("chelis#616"),
            "unexpected message: {message}"
        );
    }

    #[test]
    fn metal_seam_rejects_node_valued_reshape_target() {
        let (mut dag, x, m) = dag_with_scalar();
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Node(1)],
            },
            vec![x, m],
            ty(lit_dims(&[4]), Prim::F32),
            None,
        );
        let err = reject_unsupported_metal_ops(&dag)
            .expect_err("Metal seam must reject a node-valued reshape target");
        let message = err.to_string();
        assert!(
            message.contains("--target c") && message.contains("chelis#616"),
            "unexpected message: {message}"
        );
    }

    /// Concrete (literal) bounds keep flowing through both seams — the
    /// rejection is scoped to node-valued dims only.
    #[test]
    fn seams_accept_literal_movement_and_reshape() {
        let (mut dag, x, _m) = dag_with_scalar();
        let shrunk = dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::Lit(2))],
            },
            vec![x],
            ty(lit_dims(&[2]), Prim::F32),
            None,
        );
        dag.add_node(
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(2), RtDim::Lit(1)],
            },
            vec![shrunk],
            ty(lit_dims(&[2, 1]), Prim::F32),
            None,
        );
        reject_unsupported_hip_ops(&dag).expect("literal bounds must pass the HIP seam");
        reject_unsupported_metal_ops(&dag).expect("literal bounds must pass the Metal seam");
    }
}
