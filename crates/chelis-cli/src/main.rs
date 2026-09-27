//! Chelis compiler CLI.

mod c_source_name;
mod eval_output;
mod eval_timeout;
mod prove;
mod style_gate;

use chelis_compiler_api::compiler::{BuildTarget, CompilerError};
use chelis_compiler_api::schema::{
    CheckDirectoryEntry, CheckDirectoryReport, CheckResult, Diagnostic, EntryPath, EvalRequest,
    SourceKind, WireInferredAdtArg, WireInferredDim, WireInferredDimensionArg, WireInferredEffect,
    WireInferredPrecision, WireInferredType, escaped_path,
};
use chelis_deep::DeepTag;
use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr, ExprCarrier as DeepExprCarrier};
use chelis_surf::ast::{Decl, ImportKind};
use chelis_types::types::{Dim, Effect, EffectSet, NominalArg, TensorPrec, Type};
use chelis_unord::{UnordMap, UnordSet};
use chelis_vocab::DiagnosticKind;
use clap::{ArgAction, ArgGroup, Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
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

const HIP_RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-hip/runtime/chelis_hip_runtime.h"
));
const DEVICE_OWNER_CPP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-hip/runtime/chelis_device_owner.cpp"
));
const DEVICE_OWNER_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-hip/runtime/chelis_device_owner.h"
));
const DEVICE_DESCRIPTOR_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-hip/runtime/chelis_device_descriptor.h"
));
const METAL_RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-metal/runtime/chelis_metal_runtime.h"
));

#[derive(Clone, Copy, Default)]
struct ExtraRuntimeArtifacts {
    hip: bool,
    metal: bool,
}

/// Stage the runtime this chelis build carries, plus the backend support files
/// `extras` requests, into `runtime_dir` (spec/08-backends.md §2.1).
fn stage_runtime_artifacts(
    runtime_dir: &Path,
    extras: ExtraRuntimeArtifacts,
) -> Result<chelis_runtime_bundle::StagedRuntime, Box<dyn std::error::Error>> {
    let staged = chelis_runtime_bundle::stage(runtime_dir)?;
    if extras.hip {
        fs::write(runtime_dir.join("chelis_hip_runtime.h"), HIP_RUNTIME_H)?;
        fs::write(
            runtime_dir.join("chelis_device_owner.cpp"),
            DEVICE_OWNER_CPP,
        )?;
        fs::write(runtime_dir.join("chelis_device_owner.h"), DEVICE_OWNER_H)?;
        fs::write(
            runtime_dir.join("chelis_device_descriptor.h"),
            DEVICE_DESCRIPTOR_H,
        )?;
    }
    if extras.metal {
        fs::write(runtime_dir.join("chelis_metal_runtime.h"), METAL_RUNTIME_H)?;
    }
    Ok(staged)
}

/// Report the staged runtime archive and its SHA-256 on stdout.
fn print_staged_runtime(staged: &chelis_runtime_bundle::StagedRuntime) {
    println!(
        "Staged runtime {} (sha256 {})",
        staged.archive.display(),
        staged.archive_sha256
    );
}

/// `chelis runtime export <dir>`: write the runtime this chelis build carries
/// so packaging ships exactly those bytes (spec/08-backends.md §2.1).
fn cmd_runtime(command: RuntimeCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        RuntimeCommand::Export { dir } => {
            chelis_runtime_bundle::preflight()?;
            fs::create_dir_all(&dir)?;
            let staged = chelis_runtime_bundle::stage(&dir)?;
            print_staged_runtime(&staged);
            Ok(())
        }
    }
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
    /// Resugar well-formed public Deep (.dp) to canonical Surf
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
    /// Explicit source migration commands; never part of normal parsing.
    Migrate {
        #[command(subcommand)]
        command: MigrateCommand,
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
        /// Target backend for realizability inference. Determines which
        /// capability set is used for lane assignment. Default: `eval`
        /// (full capability). Use `--target c` to manifest under C
        /// backend constraints — required for #763 cross-lane comparison.
        #[arg(long)]
        target: Option<String>,
        /// Abandon the evaluation after this many seconds and exit
        /// non-zero with `error: evaluation timed out after <N>s`.
        ///
        /// For unattended and scripted use: without it, a mis-sized or
        /// accidentally quadratic program is indistinguishable from one
        /// that is still making progress (chelis#914). Interactive Ctrl-C
        /// already works and needs no flag.
        #[arg(long, value_name = "SECS")]
        timeout: Option<u64>,
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
    /// Write the runtime this chelis build carries
    Runtime {
        #[command(subcommand)]
        command: RuntimeCommand,
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
        /// Verification tier: auto (A→B→C), fuzz-only, smt-only, induction-only, type-only, beacon-only
        #[clap(long, default_value = "auto")]
        tier: String,
        /// SMT solver timeout in milliseconds (default 5000)
        #[clap(long, default_value = "5000")]
        smt_timeout: u64,
        /// Beacon search budget in milliseconds, excluding compiler preparation.
        #[clap(long, default_value = "60000")]
        beacon_budget: u64,
        /// Optional wall budget in milliseconds including compiler preparation (maximum one day)
        #[arg(long, value_parser = clap::value_parser!(u64).range(..=86_400_000))]
        beacon_wall_budget: Option<u64>,
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
enum MigrateCommand {
    /// Rewrite the isolated Surf v0.18 grammar to canonical Surf v0.19.
    Surf {
        /// Source grammar version. The only supported legacy version is 0.18.
        #[arg(long)]
        from: String,
        /// Verify that every path is already migrated without writing.
        #[arg(long)]
        check: bool,
        /// Rewrite every path after the complete batch passes preflight.
        #[arg(long)]
        inplace: bool,
        #[arg(required = true)]
        paths: Vec<PathBuf>,
    },
    /// Rewrite canonical Deep v0.18 integer names to canonical Deep v0.19.
    Deep {
        /// Source grammar version. The only supported legacy version is 0.18.
        #[arg(long)]
        from: String,
        /// Verify that every path is already migrated without writing.
        #[arg(long)]
        check: bool,
        /// Rewrite every path after the complete batch passes preflight.
        #[arg(long)]
        inplace: bool,
        #[arg(required = true)]
        paths: Vec<PathBuf>,
    },
}

#[derive(Subcommand)]
enum RuntimeCommand {
    /// Write the carried runtime archive, public runtime headers, and staging
    /// receipt to a directory, creating it when missing
    Export {
        /// Output directory
        dir: PathBuf,
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
    /// Regenerate the managed blocks and re-materialize the skill set from the
    /// pinned toolchain, restamping to the reef pin. The AGENTS.md block receives
    /// the complete pinned root Chelis contract. Touches only managed regions,
    /// `agent-skills/`, `.claude/skills/`, and `.codex/skills/`. Refuses, before writing anything, on
    /// a repo missing an artifact it restamps in place; run `conform init` first.
    /// Shell-owned text outside managed regions is preserved. In `reef.toml`,
    /// `[conform] local_skills = [...]` preserves shell-owned skill additions and
    /// `excluded_skills = [...]` removes named embedded skills during sync.
    /// Within a retained skill's trailing `shell-local` block, comment-wrapped
    /// headings between `<!-- shell-local:exclude:begin -->` and
    /// `<!-- shell-local:exclude:end -->` remove inherited sections.
    /// The same standalone selector span outside AGENTS.md's managed block
    /// removes exact inherited AGENTS.md sections. Removing a selector restores
    /// the current upstream section on the next sync; selecting the root
    /// `# Chelis Agent Contract` heading omits the whole inherited body. Full
    /// inheritance is the default, and each shell should keep the inherited and
    /// shell-owned guidance that remains relevant and current. Selector control
    /// markers must be standalone Markdown comments, outside code fences and
    /// enclosing HTML blocks.
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
    /// On a repo that has not been conformed (missing an artifact the bump
    /// restamps in place) it refuses before writing anything and names the gap.
    /// Honors the same `reef.toml` skill controls as `conform sync`:
    /// `[conform] local_skills = [...]` for shell-owned additions and
    /// `excluded_skills = [...]` for named embedded removals.
    /// A retained skill's `shell-local:exclude` heading selectors are also
    /// reapplied while its local block is preserved. Standalone AGENTS.md
    /// selectors are applied to the complete pinned root contract in the same
    /// way. Selector control markers must be standalone Markdown comments,
    /// outside code fences and enclosing HTML blocks. `.claude/skills` and
    /// `.codex/skills` are restored as `../agent-skills` symlinks.
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
        Some(Command::Migrate { command }) => match command {
            MigrateCommand::Surf {
                from,
                check,
                inplace,
                paths,
            } => cmd_migrate_surf(&from, &paths, check, inplace),
            MigrateCommand::Deep {
                from,
                check,
                inplace,
                paths,
            } => cmd_migrate_deep(&from, &paths, check, inplace),
        },
        Some(Command::Eval {
            file,
            expr,
            json,
            allow_style_violations,
            target,
            timeout,
        }) => {
            // Parse target for realizability inference (issue #912).
            match parse_eval_target(target.as_deref()) {
                Ok(parsed_target) => cmd_eval(
                    file.as_deref(),
                    expr.as_deref(),
                    json,
                    allow_style_violations,
                    parsed_target,
                    target.is_some(), // whether user explicitly passed --target
                    timeout,
                ),
                Err(error) => Err(error.into()),
            }
        }
        Some(Command::Check {
            file,
            show_inferred,
            allow_style_violations,
        }) => {
            // Issue #207: exit code mirrors the document's error lists (0 iff
            // all are empty, CHECK_ERRORS_EXIT_CODE otherwise). `cmd_check` is
            // total, so there is no exit-1 arm: every failure is in the
            // document it printed (chelis#886, chelis#1678).
            std::process::exit(cmd_check(&file, show_inferred, allow_style_violations))
        }
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
        Some(Command::Runtime { command }) => cmd_runtime(command),
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
            beacon_budget,
            beacon_wall_budget,
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
                beacon_budget: std::time::Duration::from_millis(beacon_budget),
                beacon_deadline: beacon_wall_budget
                    .map(|ms| std::time::Instant::now() + std::time::Duration::from_millis(ms)),
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
        if e.is::<eval_output::NumericTrapCliError>() {
            eprintln!("{e}");
        } else {
            eprintln!("error: {e}");
        }
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
                // [04-FIT-26]: the shared projected rendering, never `{:?}`.
                const PREFIX: &str =
                    "`chelis deep --annotate` requires a well-typed program; type errors:";
                return Err(
                    match chelis_compiler_api::check_report::render_check_errors(&result.errors) {
                        Ok(lines) => format!("{PREFIX}\n{lines}"),
                        Err(reason) => format!("{PREFIX} {reason}"),
                    }
                    .into(),
                );
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
        let deep_exprs = chelis_deep::parse_and_stamp_file(&deep_source)?;
        let surf = chelis_surf::decompile::try_decompile_program_with_context(
            &deep_exprs,
            &options,
            synthetic_name,
        )?;
        let surf = if verbose {
            surf
        } else {
            canonicalize_decompiled_surf(&surf)?
        };
        print!("{surf}");
    } else {
        // For .ch files, round-trip through deep and back
        let decls = chelis_surf::parser::parse_str(&source)?;
        // Public Deep is post-expansion. Resugaring the pre-expansion
        // compiler-only `defmacro`/`macro-invoke` forms would invent a second
        // Surf dialect and makes even a valid macro program fail here.
        let deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
        let surf = chelis_surf::decompile::try_decompile_program_with_context(
            &deep_exprs,
            &options,
            synthetic_name,
        )?;
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
        let deep_exprs = chelis_deep::parser::parse_and_stamp_file(&source)?;
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

fn cmd_migrate_surf(
    from: &str,
    paths: &[PathBuf],
    check: bool,
    inplace: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if from != "0.18" {
        return Err(format!(
            "unsupported Surf migration source version `{from}`; expected `--from 0.18`"
        )
        .into());
    }
    if check && inplace {
        return Err(
            "`chelis migrate surf` does not allow `--check` and `--inplace` together".into(),
        );
    }
    if !check && !inplace && paths.len() != 1 {
        return Err(
            "printing a migration requires exactly one path; use `--check` or `--inplace` for a batch"
                .into(),
        );
    }

    // Preflight the complete batch before the first write, collecting every
    // file that blocks it so one run names them all instead of stopping at the
    // first. The batch stays all-or-nothing: any blocked file writes nothing.
    let mut migrations = Vec::with_capacity(paths.len());
    let mut blocked = Vec::new();
    for path in paths {
        match preflight_migration(path) {
            Ok(migration) => migrations.push(migration),
            Err(failure) => blocked.push(failure),
        }
    }
    if !blocked.is_empty() {
        return Err(describe_blocked_migrations("Surf", &blocked, paths.len(), inplace).into());
    }

    if check {
        let stale = migrations
            .iter()
            .filter(|(_, source, migrated)| source != migrated)
            .map(|(path, _, _)| path.display().to_string())
            .collect::<Vec<_>>();
        if stale.is_empty() {
            return Ok(());
        }
        return Err(format!("Surf v0.18 migration required: {}", stale.join(", ")).into());
    }

    if inplace {
        persist_migrations_atomically(&migrations)?;
    } else if let Some((_, _, migrated)) = migrations.into_iter().next() {
        print!("{migrated}");
    }
    Ok(())
}

fn cmd_migrate_deep(
    from: &str,
    paths: &[PathBuf],
    check: bool,
    inplace: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if from != "0.18" {
        return Err(format!(
            "unsupported Deep migration source version `{from}`; expected `--from 0.18`"
        )
        .into());
    }
    if check && inplace {
        return Err(
            "`chelis migrate deep` does not allow `--check` and `--inplace` together".into(),
        );
    }
    if !check && !inplace && paths.len() != 1 {
        return Err(
            "printing a migration requires exactly one path; use `--check` or `--inplace` for a batch"
                .into(),
        );
    }

    let mut migrations = Vec::with_capacity(paths.len());
    let mut blocked = Vec::new();
    for path in paths {
        match preflight_deep_migration(path) {
            Ok(migration) => migrations.push(migration),
            Err(failure) => blocked.push(failure),
        }
    }
    if !blocked.is_empty() {
        return Err(describe_blocked_migrations("Deep", &blocked, paths.len(), inplace).into());
    }

    if check {
        let stale = migrations
            .iter()
            .filter(|(_, source, migrated)| source != migrated)
            .map(|(path, _, _)| path.display().to_string())
            .collect::<Vec<_>>();
        if stale.is_empty() {
            return Ok(());
        }
        return Err(format!("Deep v0.18 migration required: {}", stale.join(", ")).into());
    }

    if inplace {
        persist_migrations_atomically(&migrations)?;
    } else if let Some((_, _, migrated)) = migrations.into_iter().next() {
        print!("{migrated}");
    }
    Ok(())
}

fn preflight_deep_migration(path: &Path) -> Result<(PathBuf, String, String), String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let migrated = chelis_deep::migration::migrate_source_v018(&source)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let parsed = chelis_deep::parser::parse_and_stamp_file(&migrated).map_err(|error| {
        format!(
            "{}: migrated output does not parse: {error}",
            path.display()
        )
    })?;
    let canonical = chelis_deep::printer::print_canonical(&parsed);
    if canonical != migrated {
        return Err(format!(
            "{}: migration output was not a canonical Deep printer fixed point",
            path.display()
        ));
    }
    Ok((path.to_path_buf(), source, migrated))
}

/// Migrate one file and prove the result canonical, returning the staged
/// `(path, original, migrated)` triple or the reason the file blocks the batch.
///
/// Besides canonical parsing, the file must satisfy the public
/// Surf -> Deep -> Surf -> Deep structural law after macro expansion; comments
/// are intentionally outside Deep.
fn preflight_migration(path: &Path) -> Result<(PathBuf, String, String), String> {
    let source =
        fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let migrated = chelis_surf::format::migrate_source_v018(&source)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    let canonical = chelis_surf::format::format_source(&migrated).map_err(|error| {
        format!(
            "{}: migrated output is not canonical: {error}",
            path.display()
        )
    })?;
    if canonical != migrated {
        let line = migrated
            .lines()
            .zip(canonical.lines())
            .position(|(migrated, canonical)| migrated != canonical)
            .unwrap_or_else(|| migrated.lines().count().min(canonical.lines().count()));
        let migrated_line = migrated.lines().nth(line).unwrap_or("<end of file>");
        let canonical_line = canonical.lines().nth(line).unwrap_or("<end of file>");
        return Err(format!(
            "{}: migration output was not a canonical formatter fixed point at line {}:\n  migration: {migrated_line:?}\n  formatter: {canonical_line:?}",
            path.display(),
            line + 1,
        ));
    }

    let declarations = chelis_surf::parser::parse_str(&migrated).map_err(|error| {
        format!(
            "{}: migrated output does not parse: {error}",
            path.display()
        )
    })?;
    let deep = expanded_desugared_program(&declarations)
        .map_err(|error| format!("{}: macro expansion failed: {error}", path.display()))?;
    let resugared = chelis_surf::resugar::resugar_program(&deep)
        .map_err(|error| format!("{}: Deep resugaring failed: {error}", path.display()))?;
    let redesugared = expanded_desugared_program(&resugared).map_err(|error| {
        format!(
            "{}: resugared macro expansion failed: {error}",
            path.display()
        )
    })?;
    let deep_canonical = chelis_deep::printer::print_canonical(
        &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&deep)
            .map_err(|error| format!("{}: Deep normalization failed: {error}", path.display()))?,
    );
    let redesugared_canonical = chelis_deep::printer::print_canonical(
        &chelis_surf::resugar::normalize_deep_for_surface_roundtrip(&redesugared)
            .map_err(|error| format!("{}: Deep normalization failed: {error}", path.display()))?,
    );
    if deep_canonical != redesugared_canonical {
        return Err(format!(
            "{}: migrated program failed the Surf -> Deep -> Surf -> Deep structural oracle\noriginal Deep:\n{deep_canonical}resugared Deep:\n{redesugared_canonical}",
            path.display(),
        ));
    }
    Ok((path.to_path_buf(), source, migrated))
}

/// Render every file that blocked the batch.
///
/// A lone failure keeps its bare per-file diagnostic, which is the whole
/// message when a caller migrates one file at a time. A batch gets the count
/// as well, so a reader can see the run listed more than the first name.
///
/// Only `--inplace` promises that nothing was written, so only `--inplace`
/// says so. Reporting an untaken write on a read-only run would invite the
/// reader to look for damage that was never possible.
fn describe_blocked_migrations(
    carrier: &str,
    blocked: &[String],
    total: usize,
    inplace: bool,
) -> String {
    if let [only] = blocked {
        return only.clone();
    }
    let consequence = if inplace {
        "; no file was modified"
    } else {
        ""
    };
    format!(
        "{} of {total} files blocked the {carrier} v0.18 migration{consequence}:\n  {}",
        blocked.len(),
        blocked.join("\n  "),
    )
}

struct PendingMigrationWrite {
    path: PathBuf,
    staged: Option<tempfile::NamedTempFile>,
    backup: Option<tempfile::NamedTempFile>,
}

/// Stage every changed file beside its destination before replacing any file.
/// Each per-file replacement is atomic; if a later replacement reports an
/// error, already-replaced files are restored from their staged backups before
/// the command returns the error.
fn persist_migrations_atomically(
    migrations: &[(PathBuf, String, String)],
) -> Result<(), Box<dyn std::error::Error>> {
    // Linked, non-ordinary, and unwritable inputs invalidate the whole batch,
    // even when a particular file is already canonical and needs no bytes
    // replaced. Preflight every requested path before staging any write.
    for (path, _, _) in migrations {
        preflight_migration_target(path)?;
    }

    let mut pending = migrations
        .iter()
        .filter(|(_, source, migrated)| source != migrated)
        .map(|(path, source, migrated)| prepare_migration_write(path, source, migrated))
        .collect::<Result<Vec<_>, _>>()?;

    for index in 0..pending.len() {
        let path = pending[index].path.clone();
        let Some(staged) = pending[index].staged.take() else {
            return Err(format!("{} had no prepared migration file", path.display()).into());
        };
        let replacement = if env::var("CHELIS_TEST_MIGRATION_FAIL_PERSIST_INDEX")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            == Some(index)
        {
            Err("injected migration persist failure".to_string())
        } else {
            staged
                .persist(&path)
                .map(|_| ())
                .map_err(|error| error.to_string())
        };
        if let Err(error) = replacement {
            let rollback_errors = rollback_committed_migrations(&mut pending, index);
            let rollback = if rollback_errors.is_empty() {
                String::new()
            } else {
                format!("; rollback failures: {}", rollback_errors.join("; "))
            };
            return Err(format!(
                "failed to atomically replace {}: {error}{rollback}",
                path.display()
            )
            .into());
        }
    }
    Ok(())
}

fn prepare_migration_write(
    path: &Path,
    source: &str,
    migrated: &str,
) -> Result<PendingMigrationWrite, Box<dyn std::error::Error>> {
    // Repeat the target validation immediately before staging to close the
    // gap between the batch-wide preflight and this file's preparation.
    let metadata = preflight_migration_target(path)?;
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent directory", path.display()))?;

    let staged = prepare_sibling_temp(parent, migrated.as_bytes(), metadata.permissions())?;
    let backup = prepare_sibling_temp(parent, source.as_bytes(), metadata.permissions())?;
    Ok(PendingMigrationWrite {
        path: path.to_path_buf(),
        staged: Some(staged),
        backup: Some(backup),
    })
}

fn preflight_migration_target(path: &Path) -> Result<fs::Metadata, Box<dyn std::error::Error>> {
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() {
        return Err(format!(
            "{} is a symbolic link; no migration files were changed",
            path.display()
        )
        .into());
    }
    if !metadata.file_type().is_file() {
        return Err(format!(
            "{} is not an ordinary file; no migration files were changed",
            path.display()
        )
        .into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        if metadata.nlink() != 1 {
            return Err(format!(
                "{} has multiple hard links; no migration files were changed",
                path.display()
            )
            .into());
        }
    }
    if metadata.permissions().readonly() {
        return Err(format!(
            "{} is read-only; no migration files were changed",
            path.display()
        )
        .into());
    }
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|error| format!("{} is not writable: {error}", path.display()))?;
    Ok(metadata)
}

fn prepare_sibling_temp(
    parent: &Path,
    contents: &[u8],
    permissions: fs::Permissions,
) -> Result<tempfile::NamedTempFile, Box<dyn std::error::Error>> {
    let mut file = tempfile::Builder::new()
        .prefix(".chelis-migrate-")
        .tempfile_in(parent)?;
    file.write_all(contents)?;
    file.flush()?;
    file.as_file().sync_all()?;
    file.as_file().set_permissions(permissions)?;
    Ok(file)
}

fn rollback_committed_migrations(
    pending: &mut [PendingMigrationWrite],
    committed: usize,
) -> Vec<String> {
    let mut errors = Vec::new();
    for migration in pending[..committed].iter_mut().rev() {
        let Some(backup) = migration.backup.take() else {
            errors.push(format!(
                "{} had no prepared backup",
                migration.path.display()
            ));
            continue;
        };
        if let Err(error) = backup.persist(&migration.path) {
            errors.push(format!("{}: {error}", migration.path.display()));
        }
    }
    errors
}

/// Parse the `--target` flag for `chelis eval` into a `Target` enum value.
/// Default (None) → `Target::Eval`. Recognized values: "eval", "c", "hip", "metal".
/// The value is rejected rather than substituted: target choice controls the
/// manifest's lane assignment and therefore cannot fall back silently.
fn parse_eval_target(target: Option<&str>) -> Result<chelis_types::types::Target, String> {
    use chelis_types::types::Target;
    match target {
        None | Some("eval") => Ok(Target::Eval),
        Some("c") => Ok(Target::C),
        Some("hip") => Ok(Target::Hip),
        Some("metal") => Ok(Target::Metal),
        Some(other) => Err(format!(
            "unknown eval target `{other}`; valid targets: eval, c, hip, metal"
        )),
    }
}

fn build_root_manifest(
    checked: &chelis_types::CheckedProgram,
    target: BuildTarget,
) -> chelis_types::manifest::RootManifest {
    let target = match target {
        BuildTarget::C => chelis_types::types::Target::C,
        BuildTarget::Hip => chelis_types::types::Target::Hip,
        BuildTarget::Metal => chelis_types::types::Target::Metal,
    };
    let realizability = chelis_effects::realizability::infer_realizability(
        checked,
        chelis_compiler_api::target_capability::tensor_capable_prims(target),
    );
    chelis_effects::realizability::compute_root_manifest(checked, &realizability)
}

fn verified_host_codegen_program(
    checked: &chelis_types::CheckedProgram,
    manifest: &chelis_types::manifest::RootManifest,
    target: BuildTarget,
    program: chelis_ir::host::ConcreteHostProgram,
) -> Result<chelis_ir::ownership::VerifiedHostProgram, Box<dyn std::error::Error>> {
    let target = match target {
        BuildTarget::C => chelis_types::types::Target::C,
        BuildTarget::Hip => chelis_types::types::Target::Hip,
        BuildTarget::Metal => chelis_types::types::Target::Metal,
    };
    let manifested =
        chelis_types::manifest::ManifestedProgram::new(checked.clone(), manifest.clone(), target);
    let selected = chelis_backend_c::prepare_host_program_for_codegen(program)?;
    let lowered = chelis_ir::ownership::lower_host_ownership(&manifested, selected)?;
    Ok(chelis_ir::ownership::verify_ownership(lowered)?)
}

fn verified_host_execution_codegen_program(
    checked: &chelis_types::CheckedProgram,
    manifest: &chelis_types::manifest::RootManifest,
    program: chelis_ir::host::HostExecutionPlan,
) -> Result<chelis_ir::ownership::VerifiedHostProgram, Box<dyn std::error::Error>> {
    let manifested = chelis_types::manifest::ManifestedProgram::new(
        checked.clone(),
        manifest.clone(),
        chelis_types::types::Target::C,
    );
    let selected = chelis_backend_c::prepare_host_execution_plan_for_codegen(program)?;
    let lowered = chelis_ir::ownership::lower_host_execution_ownership(&manifested, selected)?;
    Ok(chelis_ir::ownership::verify_ownership(lowered)?)
}

fn execution_host_requires_host_backend(
    checked: &chelis_types::CheckedProgram,
) -> Result<bool, Box<dyn std::error::Error>> {
    let manifest = build_root_manifest(checked, BuildTarget::C);
    let (_, host) = chelis_ir::host::try_lower_execution_program_with_manifest(checked, &manifest)
        .map_err(|diagnostic| format!("Lowering error: {diagnostic}"))?;
    Ok(host
        .as_ref()
        .map(|host| chelis_ir::host::host_program_requires_host_backend(host.program()))
        .unwrap_or(false))
}

type CliLoweredBuildProgram = (
    chelis_ir::Dag,
    Option<chelis_ir::host::ConcreteHostProgram>,
    Option<chelis_ir::host::HostExecutionPlan>,
);

fn compiled_host_lowering_error_for_cli(diagnostic: chelis_ir::lower::LowerDiagnostic) -> String {
    let is_cross_lane_nonliteral_window = diagnostic.unsupported().is_some_and(|unsupported| {
        let identity = unsupported.identity();
        identity.stage == chelis_types::unsupported::Stage::Lowering
            && identity.context == "the compiled-backend lowering of `reduce_window_*`"
            && matches!(
                identity.what,
                chelis_types::unsupported::UnsupportedKind::Construct(ref what)
                    if what == "a non-literal window list for `reduce_window_max`"
            )
            && identity
                .tracking_issue
                .is_some_and(|issue| issue.number() == 1058)
    });
    if is_cross_lane_nonliteral_window {
        diagnostic.to_string()
    } else {
        format!("Lowering error: {diagnostic}")
    }
}

fn lower_build_program_for_cli(
    checked: &chelis_compiler_api::pipeline::CheckedCompilation,
    manifest: &chelis_types::manifest::RootManifest,
    target: BuildTarget,
) -> Result<CliLoweredBuildProgram, Box<dyn std::error::Error>> {
    if target == BuildTarget::C {
        // This mode asks the shared pipeline to select the CLI's ordinary
        // host/strict policy from the actual collected host. Planned helpers
        // instead use its explicit host binding, with no standalone root zip.
        let (lowered, mut ordinary_host, plan) =
            chelis_compiler_api::pipeline::lower_checked_for_c_execution(
                checked.clone(),
                manifest,
                chelis_compiler_api::pipeline::LoweringMode::AllowHostBackend,
            )
            .map_err(|rejection| boxed_string_error(rejection.to_string()))?;
        let host = plan
            .as_ref()
            .map(chelis_ir::host::HostExecutionPlan::program)
            .or(ordinary_host.as_ref());
        emit_summary_rejections(host);
        if let Some(host) = ordinary_host.as_mut() {
            apply_manifest_display_roots(host, manifest, target)?;
        }
        let plan = plan
            .map(|plan| {
                plan.try_transform_globals(|globals, functions| {
                    apply_manifest_display_roots_to_globals(globals, functions, manifest, target)
                })
            })
            .transpose()?;
        Ok((lowered.into_dag(), ordinary_host, plan))
    } else {
        let mut compiled =
            chelis_ir::host::try_lower_compiled_program_with_manifest(checked.program(), manifest)
                .map_err(compiled_host_lowering_error_for_cli)?;
        emit_summary_rejections(compiled.host.as_ref());
        let lowered = lower_checked_compilation_for_cli(checked.clone(), compiled.host.as_ref())?;
        if let Some(host) = compiled.host.as_mut() {
            apply_manifest_display_roots(host, manifest, target)?;
        }
        Ok((lowered.into_dag(), compiled.host, None))
    }
}

fn verified_dag_codegen_program(
    dag: chelis_ir::dag::Dag,
) -> Result<chelis_ir::ownership::VerifiedDagProgram, Box<dyn std::error::Error>> {
    let lowered = chelis_ir::ownership::lower_dag_ownership(dag)?;
    Ok(chelis_ir::ownership::verify_ownership(lowered)?)
}

fn require_build_manifest_inputs(
    manifest: &chelis_types::manifest::RootManifest,
    target: BuildTarget,
) -> Result<(), chelis_types::unsupported::Unsupported> {
    if let Some(entry) = manifest
        .entries
        .iter()
        .find(|entry| !entry.required_inputs.is_empty())
    {
        return Err(build_unavailable_root_error(
            entry,
            target,
            format!(
                "`chelis build` has no runtime binding for required input(s) {}",
                entry
                    .required_inputs
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        ));
    }
    Ok(())
}

fn build_unavailable_root_error(
    entry: &chelis_types::manifest::RootEntry,
    target: BuildTarget,
    reason: impl Into<String>,
) -> chelis_types::unsupported::Unsupported {
    chelis_types::unsupported::Unsupported::new(
        chelis_types::unsupported::UnsupportedKind::Construct(format!(
            "[05-UNS-1] unavailable root `{}`",
            entry.name
        )),
        format!(
            "{:?} lane while building target `{}`: {}",
            entry.lane,
            target.as_str(),
            reason.into()
        ),
        chelis_types::unsupported::Stage::Codegen(target.as_str()),
        chelis_types::unimplemented_rejection!(
            912,
            "this is a root-realization defect; file a bug with the program and target"
        ),
    )
}

/// chelis#914: `--timeout` wrapper around [`cmd_eval_inner`].
///
/// The token is installed on THIS thread because the eval runs inline in the
/// CLI; a watchdog thread trips it after `secs`. The guard is held across the
/// whole inner call so every dispatch arm is covered, and the cancellation
/// sentinel is translated here into the user-facing timeout message — the
/// inner path stays unaware that a timeout exists.
fn cmd_eval(
    file: Option<&std::path::Path>,
    expr: Option<&str>,
    json: bool,
    allow_style_violations: bool,
    target: chelis_types::types::Target,
    explicit_target: bool,
    timeout: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let watchdog = timeout.map(|secs| eval_timeout::EvalTimeout::install(secs, json));
    let outcome = cmd_eval_inner(
        file,
        expr,
        json,
        allow_style_violations,
        target,
        explicit_target,
    );
    // All formatting, including diagnostics, precedes the terminal claim.
    // A stalled formatter still leaves completed effects with the watchdog.
    let mut output = match outcome {
        Ok(output) => output,
        Err(error) => eval_output::EvalOutput::failure(Vec::new(), error.to_string(), json),
    };
    if let (Some(secs), Some(error)) = (timeout, output.error.as_mut())
        && chelis_compiler_api::is_cancellation(error)
    {
        *error = format!("evaluation timed out after {secs}s (--timeout)");
    }
    if watchdog
        .as_ref()
        .is_some_and(|watchdog| !watchdog.claim_normal())
    {
        // The forced path owns emission and process exit. Returning here
        // would let main report success or emit a second diagnostic.
        loop {
            std::thread::park();
        }
    }
    drop(watchdog);
    output.emit()
}

fn cmd_eval_inner(
    file: Option<&std::path::Path>,
    expr: Option<&str>,
    json: bool,
    allow_style_violations: bool,
    target: chelis_types::types::Target,
    _explicit_target: bool,
) -> Result<eval_output::EvalOutput, Box<dyn std::error::Error>> {
    // The style gate runs only on the `--file` form (a real on-disk
    // source). The `--expr` form is a synthetic one-line snippet
    // wrapped as `eval_result = <expr>` and never lands on disk, so
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
                chelis_deep::parse_and_stamp_file(&deep_source)
                    .map_err(|err| boxed_string_error(err.to_string()))?;
                return if json {
                    prepare_eval_json(try_eval_result_for_target(
                        SourceKind::Deep,
                        &deep_source,
                        None,
                        target,
                    ))
                } else {
                    prepare_eval_text(try_eval_for_target(
                        SourceKind::Deep,
                        &deep_source,
                        None,
                        target,
                    ))
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
                match prepare_eval_in_context(package_root, &source, json, target) {
                    Ok(output) => return Ok(output),
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
            // [05-OBS-7..11]: selection consumes the same target-aware
            // manifest as evaluation and build. The former source-derived
            // list excluded pure nullary defs whenever the file also had a
            // value root, so eval and C disagreed about owed output. Filter
            // by entry-file declaration identity, but take names/topology
            // exclusively from the manifest.
            let selected_roots = manifest_root_names_from_decls(&entry_decls, &checked, target)?;
            if json {
                prepare_eval_json(try_eval_result_for_target(
                    SourceKind::Surf,
                    &source,
                    Some(&selected_roots),
                    target,
                ))
            } else {
                prepare_eval_text(try_eval_for_target(
                    SourceKind::Surf,
                    &source,
                    Some(&selected_roots),
                    target,
                ))
            }
        }
        (None, Some(e)) => {
            // `--expr` is by construction a one-line snippet with no reef
            // resolution — keep the legacy path. Its synthetic binding uses
            // the public observation label directly; de-mangling is reserved
            // for actual linker provenance, so a private `__` sentinel would
            // otherwise leak while lexical file bindings with that spelling
            // must remain untouched.
            let source = format!("eval_result = {e}");
            if json {
                prepare_eval_json(try_eval_result_for_target(
                    SourceKind::Surf,
                    &source,
                    None,
                    target,
                ))
            } else {
                prepare_eval_text(try_eval_for_target(SourceKind::Surf, &source, None, target))
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
fn prepare_eval_in_context(
    package_root: &Path,
    source: &str,
    json: bool,
    target: chelis_types::types::Target,
) -> Result<eval_output::EvalOutput, EvalInContextError> {
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
    let result =
        match chelis_compiler_api::compiler::eval_in_context_for_target(&context, source, target) {
            Ok(result) => result,
            Err(error) => {
                return Ok(prepare_eval_failure(error, json));
            }
        };
    if json {
        // JSON mode: stdout carries the raw `EvalResult` serde JSON
        // only. Empty-roots inputs serialize to `{"roots":[]}` (valid
        // JSON); the stderr breadcrumb is suppressed so scripted
        // consumers get a single parseable document on stdout.
        let rendered = serde_json::to_string(&result)
            .map_err(|err| EvalInContextError::Compile(format!("eval JSON serialize: {err}")))?;
        return Ok(eval_output::EvalOutput::json(rendered));
    }
    Ok(eval_output::EvalOutput::text(format_eval_result(&result)))
}

fn prepare_eval_text(
    outcome: Result<String, CompilerError>,
) -> Result<eval_output::EvalOutput, Box<dyn std::error::Error>> {
    Ok(match outcome {
        Ok(result) => eval_output::EvalOutput::text(result),
        Err(error) => prepare_eval_failure(error, false),
    })
}

fn prepare_eval_json(
    outcome: Result<chelis_compiler_api::schema::EvalResult, CompilerError>,
) -> Result<eval_output::EvalOutput, Box<dyn std::error::Error>> {
    Ok(match outcome {
        Ok(result) => eval_output::EvalOutput::json(serde_json::to_string(&result)?),
        Err(error) => prepare_eval_failure(error, true),
    })
}

fn prepare_eval_failure(error: CompilerError, json: bool) -> eval_output::EvalOutput {
    let numeric_trap = error
        .errors
        .iter()
        .any(|diagnostic| diagnostic.kind() == DiagnosticKind::NumericTrap);
    let transcript = error.transcript.clone();
    let rendered = join_eval_error(error);
    if numeric_trap {
        eval_output::EvalOutput::numeric_trap_failure(transcript, rendered, json)
    } else {
        eval_output::EvalOutput::failure(transcript, rendered, json)
    }
}

/// Keep JSON stdout free of partial results and flush effects before the
/// caller reports the diagnostic, including when stdout is a pipe.
fn emit_failed_eval_transcript(transcript: &[String], json: bool) -> io::Result<()> {
    let write_lines = |output: &mut dyn Write| -> io::Result<()> {
        for line in transcript {
            writeln!(output, "{line}")?;
        }
        output.flush()
    };
    if json {
        write_lines(&mut io::stderr().lock())
    } else {
        write_lines(&mut io::stdout().lock())
    }
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
        let deep_exprs = chelis_deep::parse_and_stamp_file(&deep_source)
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
        let entry_seeds = entry_seed_names(&entry_deep_exprs);
        let live_deep_exprs = drop_unreachable_eval_only_defs(deep_exprs.clone(), &entry_seeds);
        prune_build_program_to_reachable_defs(&live_deep_exprs, &entry_seeds)
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

/// Build a synthetic `cmd_check_one` JSON report for an unclassified error that
/// short-circuits parsing or program preparation, so the
/// `chelis check` exit-code invariant (issue #207) holds even when
/// the per-file pipeline never reaches the fitness checker. The
/// `errors[]` array carries a single `Other`-kind entry with the
/// supplied message; the rest of the report shape mirrors a zero-
/// node program with score 0.
fn synthetic_check_report_with_error(message: &str) -> CheckResult {
    synthetic_check_report_with_errors(std::slice::from_ref(&message.to_string()))
}

fn synthetic_check_report_with_typed_error(
    kind: chelis_types::errors::CheckErrorKind,
    message: String,
    span_offset: Option<usize>,
) -> CheckResult {
    synthetic_check_report_from_errors(vec![chelis_types::errors::CheckError {
        kind,
        message,
        severity: 0.5,
        expected: None,
        got: None,
        span_offset,
        span_id: None,
        suggestions: Vec::new(),
    }])
}

/// The same synthetic report, carrying one diagnostic per message.
///
/// The style gate is the caller that needs more than one: it reports a
/// formatting difference and each lint violation as separate issues, and
/// `spec/01` § Style Gate calls that "a one-issue-per-line diagnostic".
/// Collapsing them into a single `message` would put a multi-line terminal
/// transcript into a machine-facing field -- the shape chelis#886 exists to
/// remove -- and would make `errors.len()` disagree with the number of
/// problems found.
fn synthetic_check_report_with_errors(messages: &[String]) -> CheckResult {
    // chelis#886 [04-FIT-12]: the early-failure path is not a second
    // producer. It builds the same `CheckResult` the checker's path builds
    // and renders it through the same serializer, so a failure that
    // short-circuits the pipeline is transported by the report's type
    // rather than by a template that happens to agree with it.
    let errors: Vec<chelis_types::errors::CheckError> = messages
        .iter()
        .map(|message| chelis_types::errors::CheckError {
            kind: chelis_types::errors::CheckErrorKind::Other,
            message: message.clone(),
            severity: 0.5,
            expected: None,
            got: None,
            span_offset: None,
            span_id: None,
            suggestions: Vec::new(),
        })
        .collect();
    synthetic_check_report_from_errors(errors)
}

fn synthetic_check_report_from_errors(
    errors: Vec<chelis_types::errors::CheckError>,
) -> CheckResult {
    let report = chelis_types::FitnessReport {
        score: 0.0,
        components: chelis_types::fitness::FitnessComponents {
            parse: 0.0,
            structure: 0.0,
            names: 0.0,
            types: 0.0,
        },
        errors,
        typed_nodes: 0,
        untyped_nodes: 0,
        total_nodes: 0,
        unresolved_names: vec![],
    };
    // This `expect` is the boundary of the whole design, so it is justified
    // rather than hopeful. Every numeric field here is a literal constant --
    // score 0, four components 0, three counters 0, severity 0.5 -- so
    // `try_from_fitness`'s validation cannot reject them. The messages are
    // never inspected.
    //
    // If it could fail there would be nothing to emit, and inventing a
    // fallback document here would be the second producer [04-FIT-11]
    // forbids. Panicking is the honest response; returning a `Result` would
    // only push the same impossibility onto every caller and reopen the `?`
    // channel this function exists to close.
    CheckResult::try_from_fitness(&report).expect("fixed synthetic report has valid numeric fields")
}

/// Print a file's report and return the exit status `spec/04` § Gating
/// assigns it: `0` iff its `errors` list is empty.
///
/// The serializer sees only owned strings and numbers every producer has
/// already validated, so it has no failure mode left to report; the `expect`
/// states that rather than hoping it.
fn print_check_report(report: &CheckResult) -> i32 {
    println!(
        "{}",
        report
            .to_report_json()
            .expect("a validated report serializes")
    );
    if report.errors.is_empty() {
        0
    } else {
        CHECK_ERRORS_EXIT_CODE
    }
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
/// A failure that stops the check from RUNNING -- an unreadable or
/// non-UTF-8 file, a style-gate violation, a parse failure, a
/// preparation failure -- is not an exception to that rule
/// (chelis#886 [04-FIT-12]). It is reported as a diagnostic in the
/// same `errors` array and therefore also exits
/// [`CHECK_ERRORS_EXIT_CODE`]. The exit status is a function of the
/// errors array and nothing else.
///
/// This paragraph used to say the opposite -- that those paths
/// propagated through `Result::Err` and picked up exit `1`. That was
/// true until the report started transporting them. `spec/04` §
/// Gating pins only "`0` iff empty, non-zero otherwise", so both
/// values conformed; `2` was chosen because the Deep arm already
/// behaved that way for an unreadable file while the Surf arm did
/// not.
///
/// Directory mode follows the same rule over every list in its envelope
/// (spec/04 [04-FIT-25], chelis#1678). A failure of the walk itself -- a
/// directory that cannot be read, a path the envelope cannot write, an empty
/// corpus -- is a diagnostic in the envelope's own `errors`, so there is no
/// exit `1` left anywhere in `chelis check`.
fn cmd_check(target: &Path, show_inferred: bool, allow_style_violations: bool) -> i32 {
    if target.is_dir() {
        let envelope = check_directory(target, show_inferred, allow_style_violations);
        println!("{}", render_check_directory(&envelope));
        return if envelope.has_errors() {
            CHECK_ERRORS_EXIT_CODE
        } else {
            0
        };
    }
    print_check_report(&cmd_check_one(
        target,
        show_inferred,
        allow_style_violations,
    ))
}

/// The directory envelope as `chelis check <dir>` publishes it.
///
/// [04-FIT-19]: one serialization of one typed value, through the report's
/// own `ReportFormatter`, so each report inside is written exactly as a
/// file's report is -- the envelope is not rendered report-by-report and
/// spliced. Everything inside `files` sits inside an array, so the formatter
/// prints it compactly.
///
/// The serializer sees only owned strings and numbers every producer has
/// already validated, so it has no failure mode left to report; the
/// `expect`s state that rather than hoping it.
fn render_check_directory(envelope: &CheckDirectoryReport) -> String {
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(
        &mut bytes,
        chelis_compiler_api::check_report::ReportFormatter::default(),
    );
    envelope
        .serialize(&mut serializer)
        .expect("an envelope of validated reports serializes");
    String::from_utf8(bytes).expect("serde_json emits UTF-8")
}

/// `chelis check <dir>`: one envelope for the whole directory (spec/04
/// § Directory mode, chelis#1678).
///
/// This used to splice each file's rendered report into a `format!` wrapper,
/// print a differently shaped document for an empty directory, and print
/// nothing at all -- exit 1 -- when any subdirectory could not be read. Each
/// report is now a member of one typed value ([04-FIT-19]), and a failure of
/// the walk is a diagnostic in it that does not stop the walk ([04-FIT-23]).
fn check_directory(
    target: &Path,
    show_inferred: bool,
    allow_style_violations: bool,
) -> CheckDirectoryReport {
    let mut entries = Vec::new();
    let mut failures = Vec::new();
    // Walk order is the envelope's order: [04-FIT-22] for `files`, and
    // [04-FIT-23]'s own sentence for the diagnostics.
    for item in walk_sources(target, &CHECK_WALK) {
        let file = match item {
            WalkItem::Source(file) => file,
            WalkItem::Failure(message) => {
                failures.push(message);
                continue;
            }
        };
        match EntryPath::relative_to(target, &file) {
            Ok(path) => entries.push(CheckDirectoryEntry::new(
                path,
                cmd_check_one(&file, show_inferred, allow_style_violations),
            )),
            // [04-FIT-21]: a path the envelope cannot write is not an entry,
            // and the file is not checked for a report with nowhere to go.
            Err(unrepresentable) => failures.push(unrepresentable.to_string()),
        }
    }
    CheckDirectoryReport::from_walk(entries, failures)
        .unwrap_or_else(|empty| empty.into_report(describe_empty_corpus(target)))
}

/// The `empty_corpus` message: the target, and what the exclusions removed
/// ([04-FIT-24]).
///
/// An empty result has several causes a user would act on differently -- a
/// mistyped path landing on a sibling, a level whose only sources sit under
/// `target/`, a directory of Markdown -- so the message counts the sources the
/// exclusions hid. That takes a second walk with the exclusions off, which is
/// why it runs only here, after the corpus is known to be empty.
fn describe_empty_corpus(target: &Path) -> String {
    let unfiltered = walk_sources(
        target,
        &WalkRules {
            exclusions: false,
            ..CHECK_WALK
        },
    );
    let excluded = unfiltered
        .iter()
        .filter(|item| matches!(item, WalkItem::Source(_)))
        .count();
    // The first walk completed, so a failure here lies under an excluded
    // entry. It makes the count a lower bound, and the message says so rather
    // than presenting a partial count as exact.
    let bound = if unfiltered
        .iter()
        .any(|item| matches!(item, WalkItem::Failure(_)))
    {
        "at least "
    } else {
        ""
    };
    format!(
        "no .ch or .dp files to check under {}: {bound}{excluded} excluded under dot-prefixed \
         entries or `target` directories",
        escaped_path(target)
    )
}

/// One step of a directory walk, in walk order.
#[derive(Debug)]
enum WalkItem {
    /// A checkable file, under the path the walk reached it by.
    Source(PathBuf),
    /// A failure of the walk itself: a directory it could not read, or an
    /// entry it could not resolve.
    Failure(String),
}

/// What a walk excludes and what it collects.
struct WalkRules {
    /// Skip every dot-prefixed entry, and every directory named `target`.
    exclusions: bool,
    /// Apply the exclusions to the target itself as well.
    exclude_root: bool,
    /// Which file names count as sources.
    is_source: fn(&Path) -> bool,
}

/// `chelis check <dir>` (spec/04 [04-FIT-20]). The target itself is never
/// excluded, so a walk rooted at e.g. a `tempfile::tempdir()` path
/// (`/tmp/.tmpXyZ/...`) is not pruned whole for its name.
const CHECK_WALK: WalkRules = WalkRules {
    exclusions: true,
    exclude_root: false,
    is_source: is_check_source,
};

/// `chelis test <dir>`. Unlike `check` it has always applied its filter to the
/// target too, and still does.
const TEST_WALK: WalkRules = WalkRules {
    exclusions: true,
    exclude_root: true,
    is_source: is_test_source,
};

/// Surf (`.ch`) and Deep (`.dp`): `cmd_check_one` routes `.dp` through the
/// Deep ingestion helper, and both surfaces emit the same report.
fn is_check_source(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("ch") | Some("dp")
    )
}

fn is_test_source(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("ch")
}

/// Dot-prefixed, read from the name's bytes, so a name that is not UTF-8 is
/// classified exactly rather than through a lossy rendering.
fn is_hidden(name: &std::ffi::OsStr) -> bool {
    name.as_encoded_bytes().first() == Some(&b'.')
}

/// Walk `target` for sources, following symlinks (chelis#1678).
///
/// `chelis check <dir>` and `chelis test <dir>` both walk a directory for
/// sources, and both used to skip every symlink, so a symlinked source -- or
/// a symlinked directory of them -- was never checked or run, with a
/// successful exit. #1827 fixed that on top of `walkdir`, and a red-team pass
/// found what `walkdir` could not do: it resolves a followed link BEFORE its
/// entry filter sees the link's name, so a dot-named link that failed to
/// resolve aborted the walk; its loop check reported a failure with no path;
/// and nothing stopped one file being reached many times over, exponentially
/// many through links that double back. This walk is written out so that each
/// of those is decided rather than inherited:
///
/// - **the name filter runs before a link is resolved**, so an excluded link
///   contributes nothing, whatever its target;
/// - **every file and directory is visited once**, identified by its canonical
///   path, and the first path in walk order names it. That subsumes loop
///   detection, and it bounds the walk by the tree's real size;
/// - **an entry that cannot be resolved** is collected when its own name is a
///   source name, so its report carries the read failure, as naming it
///   directly would; contributes nothing when its referent does not exist;
///   and is otherwise a walk failure;
/// - **a directory that cannot be read** is a walk failure naming it, and the
///   walk continues past it.
///
/// Links are followed wherever they point, including out of the target.
///
/// Order is depth-first pre-order with each directory's entries sorted by the
/// bytes of their names ([04-FIT-22], the order [05-HOST-4] fixes for
/// `list_dir`), with an explicit stack so a deep tree cannot exhaust the
/// native one.
fn walk_sources(target: &Path, rules: &WalkRules) -> Vec<WalkItem> {
    let mut items = Vec::new();
    if rules.exclusions && rules.exclude_root {
        // As `walkdir` named the root: its file name, or the whole path when
        // it has none. `chelis test`'s discovery depends on it.
        let name = target.file_name().unwrap_or(target.as_os_str());
        if is_hidden(name) || (name == "target" && target.is_dir()) {
            return items;
        }
    }
    let mut visited = BTreeSet::new();
    match fs::canonicalize(target) {
        Ok(identity) => {
            visited.insert(identity);
        }
        Err(error) => {
            items.push(WalkItem::Failure(format!(
                "cannot read directory {}: {error}",
                escaped_path(target)
            )));
            return items;
        }
    }
    let mut stack = Vec::new();
    match sorted_entries(target) {
        Ok(entries) => stack.push(entries.into_iter()),
        Err(message) => items.push(WalkItem::Failure(message)),
    }
    while let Some(entries) = stack.last_mut() {
        let Some((name, path)) = entries.next() else {
            stack.pop();
            continue;
        };
        if rules.exclusions && is_hidden(&name) {
            continue;
        }
        let metadata = match fs::metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) => {
                if (rules.is_source)(&path) {
                    items.push(WalkItem::Source(path));
                } else if error.kind() != io::ErrorKind::NotFound {
                    items.push(WalkItem::Failure(format!(
                        "cannot resolve {}: {error}",
                        escaped_path(&path)
                    )));
                }
                continue;
            }
        };
        if metadata.is_dir() {
            if rules.exclusions && name == "target" {
                continue;
            }
            match fs::canonicalize(&path) {
                Ok(identity) => {
                    if !visited.insert(identity) {
                        continue;
                    }
                }
                Err(error) => {
                    items.push(WalkItem::Failure(format!(
                        "cannot resolve {}: {error}",
                        escaped_path(&path)
                    )));
                    continue;
                }
            }
            match sorted_entries(&path) {
                Ok(entries) => stack.push(entries.into_iter()),
                Err(message) => items.push(WalkItem::Failure(message)),
            }
        } else if metadata.is_file() && (rules.is_source)(&path) {
            match fs::canonicalize(&path) {
                Ok(identity) => {
                    if visited.insert(identity) {
                        items.push(WalkItem::Source(path));
                    }
                }
                // It resolved a moment ago. Collect it and let the read
                // report whatever changed, rather than dropping it here.
                Err(_) => items.push(WalkItem::Source(path)),
            }
        }
    }
    items
}

/// A directory's entries, sorted by the bytes of their names.
///
/// A directory that fails part-way through is reported whole: sorting and
/// walking a partial listing would present it as complete.
fn sorted_entries(dir: &Path) -> Result<Vec<(std::ffi::OsString, PathBuf)>, String> {
    let read = || -> io::Result<Vec<(std::ffi::OsString, PathBuf)>> {
        fs::read_dir(dir)?
            .map(|entry| entry.map(|entry| (entry.file_name(), entry.path())))
            .collect()
    };
    let mut entries =
        read().map_err(|error| format!("cannot read directory {}: {error}", escaped_path(dir)))?;
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(entries)
}

/// The file's report. Its `errors` list is the source of truth for the
/// issue #207 exit-code invariant; there is no separate flag that could
/// disagree with it.
fn cmd_check_one(file: &Path, show_inferred: bool, allow_style_violations: bool) -> CheckResult {
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
) -> CheckResult {
    // chelis#886 [04-FIT-12]: a failure before the checker is transported by
    // the report, not by a display string on stderr and an empty stdout.
    //
    // Both arms below keep writing the human diagnostic to stderr as well.
    // The atom requires the failure to reach the report; it does not ask for
    // the terminal message to be taken away, and `spec/01` §Style Gate makes
    // that stderr line part of the gate's own contract.
    //
    // These paths exit 2, like every non-empty errors array: the status is a
    // function of the report, and this function always returns one (it is
    // total). § Gating pins only "`0` iff empty, non-zero otherwise", so
    // exit 1 would also conform; 2 converged the Surf arm on the Deep arm,
    // which already reported an unreadable file this way.
    let source = match fs::read_to_string(file) {
        Ok(source) => source,
        Err(error) => {
            let message = format!("failed to read {}: {error}", file.display());
            eprintln!("error: {message}");
            return synthetic_check_report_with_error(&message);
        }
    };
    if let Err(rejection) =
        style_gate::enforce_style_gate_structured(file, &source, allow_style_violations)
    {
        // ONE gate run produces both renderings. It used to run the gate a
        // second time to get structure for the report, and the two runs
        // disagreed under a concurrent writer: run A rejected (so the process
        // exited 2) while run B found the file clean, shipping exit 2 with an
        // empty `errors` array -- the state § Gating says cannot happen, and
        // exactly the "cannot distinguish no diagnostics from diagnostics not
        // transported" confusion [04-FIT-12] exists to remove.
        //
        // stderr keeps the whole human report, advice line included. The
        // report gets one diagnostic per issue: that display string is a
        // terminal transcript -- embedded newlines, an "N issue(s)" plural
        // placeholder, and the word "build" on a `check` surface -- and
        // putting it in a machine-facing `message` is the shape chelis#886
        // removes, not one to reintroduce while closing it.
        eprintln!("error: {}", rejection.report);
        return synthetic_check_report_with_errors(&rejection.issues);
    }
    emit_advisory_lint_warnings_for_file(file);
    // Deep (`.dp`) ingestion: a standalone `.dp` is already-lowered IR,
    // not a Surf package, so the reef loader below returns `Ok(None)`
    // for it and the monolithic else-arm would feed Deep s-expressions
    // to the Surf parser (which fails with a bogus parse error). Route
    // `.dp` through the dedicated Deep helper before the reef load,
    // mirroring the established `.dp` branches in `cmd_build_dispatch`,
    // `cmd_surf`, `cmd_fmt`, and `copy_cost_for_file`.
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext.eq_ignore_ascii_case("dp") {
        // The unreadable-`.dp` arm that used to live here is gone: the read
        // above now transports that failure for every extension, which is
        // what made the Surf and Deep arms disagree in the first place.
        return cmd_check_one_deep(&source, show_inferred);
    }
    // Wave-1 red-team M1 (#207 follow-up): parse failures used to
    // short-circuit through `?` into the `Err(err)` arm in `main`,
    // emitting no JSON and exiting 1. Catch the parse error here,
    // route it through `synthetic_check_report_with_error`, and let
    // the caller map "errors non-empty" to exit 2 as documented.
    let prepared = match chelis_reef::prepare_program_for_file(file) {
        Ok(prepared) => prepared,
        Err(message) => {
            return synthetic_check_report_with_error(&message);
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
        return synthetic_check_report_with_error(EMPTY_PROGRAM_MESSAGE);
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
            // [04-FIT-12]: the layered checker's own failure is transported
            // too. It used to propagate, so a cache-path failure emitted no
            // document while the identical monolithic-path failure did.
            match chelis_compiler_api::check_layered(
                &prepared.stdlib_decls,
                prepared.stdlib_source_digest,
                &prepared.non_stdlib_decls,
            ) {
                Ok(layered) => layered,
                Err(error) => {
                    // stderr is kept alongside the report. The atom asks for
                    // the failure to REACH the report, not for the terminal
                    // line to be taken away, and this message is the only
                    // human-facing account of a layered-check failure.
                    eprintln!("error: {}", compiler_error_messages(&error));
                    // One diagnostic per compiler error, for the same reason
                    // the style gate emits one per issue: `compiler_error_messages`
                    // joins them with "; " for the terminal, and a joined
                    // string in one `message` makes `errors.len()` disagree
                    // with the number of problems.
                    return synthetic_check_report_with_errors(&compiler_error_list(&error));
                }
            }
        }
    } else {
        None
    };

    let (report, effect_errors, linearity_errors, inferred_signatures) =
        if let Some(layered) = layered {
            match layered {
                chelis_compiler_api::LayeredCheck::Clean {
                    fitness,
                    typed_program,
                } => {
                    let mut fitness = fitness;
                    let inferred = if show_inferred {
                        inferred_signatures_or_diagnostic(&typed_program, &mut fitness)
                    } else {
                        None
                    };
                    (fitness, Vec::new(), Vec::new(), inferred)
                }
                chelis_compiler_api::LayeredCheck::EffectRejected {
                    fitness,
                    effect_errors,
                    typed_program,
                } => {
                    let mut fitness = fitness;
                    let inferred = if show_inferred {
                        inferred_signatures_or_diagnostic(&typed_program, &mut fitness)
                    } else {
                        None
                    };
                    (fitness, effect_errors, Vec::new(), inferred)
                }
                chelis_compiler_api::LayeredCheck::LinearityRejected {
                    fitness,
                    linearity_errors,
                    typed_program,
                } => {
                    let mut fitness = fitness;
                    let inferred = if show_inferred {
                        inferred_signatures_or_diagnostic(&typed_program, &mut fitness)
                    } else {
                        None
                    };
                    (fitness, Vec::new(), linearity_errors, inferred)
                }
            }
        } else {
            // Monolithic path: full inference over the whole merged
            // program. Used when the input is not inside a reef package,
            // when the cache is disabled, or when the non-chelis-std
            // decls do not type-check clean.
            let decls = match &prepared {
                Some(prepared) => prepared.decls.clone(),
                None => {
                    // Reuses the source read at the top of this function. It
                    // used to re-read the file here with `?`, a second
                    // unreported failure path for the same file
                    // (chelis#886 [04-FIT-12]).
                    //
                    // Wave-1 red-team M1 (#207 follow-up): same handling
                    // as the prepared-path parse error above, for the
                    // raw `parse_str` branch used when no reef context
                    // resolves.
                    match chelis_surf::parser::parse_str(&source) {
                        Ok(decls) => decls,
                        Err(err) => {
                            return synthetic_check_report_with_error(&err.to_string());
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
                return synthetic_check_report_with_error(EMPTY_PROGRAM_MESSAGE);
            }
            // [04-FIT-12]: "any preparation failure occurring before type
            // checking begins" is the atom's own wording, so this is
            // transported rather than propagated.
            let prepared = match chelis_compiler_api::pipeline::prepare_surf_decls(&decls, None) {
                Ok(prepared) => prepared,
                Err(error) => {
                    // stderr kept, as above. A prelude-name collision is
                    // reported here, and it is the kind of message a person
                    // reads in a terminal rather than parses out of JSON.
                    let message = error.to_string();
                    eprintln!("error: {message}");
                    return match error {
                        chelis_compiler_api::pipeline::PreparationError::SurfDesugar(error) => {
                            synthetic_check_report_with_typed_error(
                                chelis_types::errors::CheckErrorKind::TypeMismatch,
                                message,
                                error.span().map(|span| span.offset),
                            )
                        }
                        _ => synthetic_check_report_with_error(&message),
                    };
                }
            };
            // chelis#1664 made this call fallible when inferred signatures
            // gained a typed carrier, and for a while it was a [04-FIT-12]
            // bypass with a `?`. It is total now: a failure to build the
            // rows becomes a diagnostic inside `check_prepared_for_cli`.
            check_prepared_for_cli(prepared, show_inferred)
        };
    assemble_check_report(
        report,
        &effect_errors,
        &linearity_errors,
        inferred_signatures,
    )
}

type PreparedCliReport = (
    chelis_types::FitnessReport,
    Vec<chelis_effects::EffectError>,
    Vec<chelis_types::errors::CheckError>,
    Option<Vec<chelis_compiler_api::schema::WireInferredSignature>>,
);

fn check_prepared_for_cli(
    prepared: chelis_compiler_api::pipeline::PreparedProgram,
    show_inferred: bool,
) -> PreparedCliReport {
    let analysis = match chelis_compiler_api::pipeline::analyze_prepared(prepared) {
        chelis_compiler_api::pipeline::PreparedTypeAnalysisOutcome::Rejected { fitness } => {
            return (
                fitness,
                Vec::new(),
                Vec::new(),
                show_inferred.then(Vec::new),
            );
        }
        chelis_compiler_api::pipeline::PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
    };

    let mut fitness = analysis.fitness().clone();
    let inferred = if show_inferred {
        inferred_signatures_or_diagnostic(analysis.program(), &mut fitness)
    } else {
        None
    };
    match chelis_compiler_api::pipeline::complete_checks(
        analysis,
        chelis_compiler_api::pipeline::SemanticContext::Isolated,
    ) {
        Ok(_) => (fitness, Vec::new(), Vec::new(), inferred),
        Err(chelis_compiler_api::pipeline::SemanticRejection::Effects { errors }) => {
            (fitness, errors, Vec::new(), inferred)
        }
        Err(chelis_compiler_api::pipeline::SemanticRejection::Linearity { errors }) => {
            (fitness, Vec::new(), errors, inferred)
        }
    }
}

/// Assemble the typed `chelis check` report from the post-pipeline analysis
/// outputs.
///
/// Shared verbatim between the `.ch` monolithic / layered arm of
/// [`cmd_check_one`] and the `.dp` helper [`cmd_check_one_deep`] so the
/// two surfaces emit byte-identical [`chelis_compiler_api::schema::CheckResult`]
/// shapes. The only thing that differs between surfaces is how the
/// `deep_exprs` feeding the fitness / type / effect / linearity checks
/// are produced; the score-adjust and JSON emission MUST NOT drift, so
/// both surfaces call exactly this function.
///
/// The caller maps a non-empty `errors` list to [`CHECK_ERRORS_EXIT_CODE`].
fn assemble_check_report(
    report: chelis_types::FitnessReport,
    effect_errors: &[chelis_effects::EffectError],
    linearity_errors: &[chelis_types::errors::CheckError],
    inferred_signatures: Option<Vec<chelis_compiler_api::schema::WireInferredSignature>>,
) -> CheckResult {
    // Infallible by construction (chelis#886 [04-FIT-12]). A validation
    // failure below means the CHECKER measured something outside its own
    // declared domain -- a NaN score, a severity outside [0, 1]. That is a
    // real defect, and the atom is explicit that it must still reach the
    // report: "a consumer cannot distinguish 'no diagnostics' from 'the
    // diagnostics were not transported'". Propagating it would produce
    // exactly the empty stdout the atom forbids.
    //
    // So the fallback re-enters the SAME producer with a different input.
    // It is not a second producer: `synthetic_check_report_with_errors` is
    // the one this file already uses for every pre-checker failure.
    match assemble_validated_check_report(
        report,
        effect_errors,
        linearity_errors,
        inferred_signatures,
    ) {
        Ok(result) => result,
        Err(message) => synthetic_check_report_with_errors(&[format!(
            "the checker produced a report outside its declared numeric domain: {message}"
        )]),
    }
}

/// The fallible half of [`assemble_check_report`], kept separate so the caller
/// above can be total. Its `Err` is a message, not a `Box<dyn Error>`,
/// because the only consumer turns it into a diagnostic.
fn assemble_validated_check_report(
    mut report: chelis_types::FitnessReport,
    effect_errors: &[chelis_effects::EffectError],
    linearity_errors: &[chelis_types::errors::CheckError],
    inferred_signatures: Option<Vec<chelis_compiler_api::schema::WireInferredSignature>>,
) -> Result<CheckResult, String> {
    use chelis_compiler_api::schema::numbers::UnitInterval;
    // Validate the measured report before applying the specified penalties;
    // otherwise max(0) could turn an invalid NaN score into a plausible zero.
    let mut wire = CheckResult::try_from_fitness(&report)?;
    if !effect_errors.is_empty() {
        report.score = (report.score - 0.2 * effect_errors.len() as f64).max(0.0);
    }
    if !linearity_errors.is_empty() {
        report.score = (report.score - 0.2 * linearity_errors.len() as f64).max(0.0);
    }
    wire.score = UnitInterval::new(report.score)?;
    wire.errors.extend(effect_errors.iter().map(|error| {
        Diagnostic::from_effect_error(
            error,
            UnitInterval::new(0.8).expect("constant effect severity"),
        )
    }));
    wire.errors.extend(
        linearity_errors
            .iter()
            .map(Diagnostic::try_from_check_error)
            .collect::<Result<Vec<_>, _>>()?,
    );
    wire.inferred_signatures = inferred_signatures;
    // Serialization re-checks the counter relationship `try_from_fitness`
    // already enforced, so a report returned from here always renders.
    wire.validate()?;
    Ok(wire)
}

/// `chelis check` ingestion for a standalone Deep (`.dp`) file.
///
/// A `.dp` is already-lowered IR by construction, so this skips the
/// Surf desugar + macro-expand stage (`expanded_desugared_program`)
/// that the `.ch` arm of [`cmd_check_one`] runs and parses the file
/// directly through the strict Deep parser. Everything after the parse
/// uses the same compiler-API pipeline as the `.ch` arm.
///
/// [`check_prepared_for_cli`] calls `analyze_prepared` and `complete_checks`.
/// It then sends their typed results to [`assemble_check_report`].
///
/// `parse_and_stamp_file` keeps the `.dp`
/// check surface on the same closed-vocabulary tag gate as
/// `chelis build`, `chelis fmt`, and `chelis cost`: an unknown tag is a
/// hard error rather than a silently-accepted node. Parse failures are
/// caught and routed through `synthetic_check_report_with_error` so a
/// malformed `.dp` produces the same JSON-report-plus-exit-2 shape the
/// `.ch` parse-error path produces, never a propagated boxed `Err`.
fn cmd_check_one_deep(source: &str, show_inferred: bool) -> CheckResult {
    let deep_source = style_gate::strip_deep_lint_directive_lines(source);
    let deep_exprs = match chelis_deep::parse_and_stamp_file(&deep_source) {
        Ok(deep_exprs) => deep_exprs,
        Err(err) => {
            return synthetic_check_report_with_error(&err.to_string());
        }
    };
    // Parity with the `.ch` empty-file path (a parse-clean file with
    // zero top-level exprs): reject with the same canonical message so
    // the `.dp` and `.ch` surfaces agree.
    if deep_exprs.is_empty() {
        return synthetic_check_report_with_error(EMPTY_PROGRAM_MESSAGE);
    }
    let prepared = chelis_compiler_api::pipeline::prepare_deep(deep_exprs, None);
    let (report, effect_errors, linearity_errors, inferred) =
        check_prepared_for_cli(prepared, show_inferred);
    assemble_check_report(report, &effect_errors, &linearity_errors, inferred)
}

fn advisory_lint_scope(file: &Path) -> &Path {
    file
}

fn emit_advisory_lint_warnings_for_file(file: &Path) {
    if style_gate::disabled_by_env() {
        return;
    }
    let parent = file
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let lint_scope = advisory_lint_scope(file);
    let rules = chelis_lint::registry::non_blocking_rules();
    let raw = match chelis_lint::lint(lint_scope, &rules) {
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
        // the two cannot drift again. The explicit file is both the
        // lint scope and the fixability-probe target; walking its parent
        // can make a temp fixture recursively lint all of `/tmp`.
        if should_suppress_unfixable_violation(lint_scope, &rules, &violation) {
            continue;
        }
        eprintln!("warning: {violation}");
    }
}

#[cfg(test)]
mod check_path_totality_tests {
    //! chelis#886 [04-FIT-12]: the per-file check path cannot fail.
    //!
    //! These are type tests first and value tests second. Each one binds a
    //! return value at its concrete type, so if any of these functions
    //! regains a `Result` the module stops COMPILING -- which is the whole
    //! point. A `?` cannot be reintroduced on this path without first
    //! widening a signature, and widening a signature breaks this file.
    //!
    //! That is deliberately stronger than the guard this replaced. An
    //! earlier attempt was a source scan for `?`, and a red-team pass got a
    //! second producer past its sibling scan twice -- once with a raw string
    //! literal, once with `concat!`. A grep can always be spelled around; a
    //! type cannot.

    use super::{
        EMPTY_PROGRAM_MESSAGE, assemble_check_report, check_directory, check_prepared_for_cli,
        cmd_check, cmd_check_one, cmd_check_one_deep, cmd_check_one_on_grown_stack,
        synthetic_check_report_with_error, synthetic_check_report_with_errors,
    };
    use chelis_compiler_api::schema::{CheckDirectoryReport, CheckResult};
    use std::path::Path;

    /// `assemble_check_report`'s signature, spelled out so the pin stays
    /// readable.
    type AssembleCheckReport = fn(
        chelis_types::FitnessReport,
        &[chelis_effects::EffectError],
        &[chelis_types::errors::CheckError],
        Option<Vec<chelis_compiler_api::schema::WireInferredSignature>>,
    ) -> CheckResult;

    /// `check_prepared_for_cli`'s signature, with its return tuple spelled
    /// out HERE rather than through the production alias
    /// `PreparedCliReport`.
    ///
    /// Naming the production alias made the pin dodgeable: a red-team
    /// mutant redefined `PreparedCliReport` itself as a `Result`, restored
    /// the original `?` inside the function, and built with every pin
    /// present, because the pin and the function changed together. A pin
    /// has to name the shape independently of the code it constrains.
    type CheckPreparedForCli = fn(
        chelis_compiler_api::pipeline::PreparedProgram,
        bool,
    ) -> (
        chelis_types::FitnessReport,
        Vec<chelis_effects::EffectError>,
        Vec<chelis_types::errors::CheckError>,
        Option<Vec<chelis_compiler_api::schema::WireInferredSignature>>,
    );

    /// Every function on the per-file check path, pinned at its signature.
    ///
    /// This is the test that actually carries the guarantee. A function
    /// pointer of an explicit type only accepts a function of exactly that
    /// type, so if any of these six regains a `Result`, THIS BINDING stops
    /// compiling -- whether or not anything calls it with `?` yet. Every
    /// type here is spelled out locally; none is borrowed from production,
    /// where it could be widened along with the function it describes.
    ///
    /// chelis#1678 added the top of the path: `cmd_check` itself and the
    /// directory envelope's producer. Directory mode was the last place a
    /// `?` could still discard every report (an unreadable subdirectory exited
    /// 1 with nothing on stdout), so it is pinned with the rest.
    ///
    /// An earlier revision of this module bound only the report producer. A
    /// red-team mutation re-widened `cmd_check_one`,
    /// `cmd_check_one_on_grown_stack` and `cmd_check_one_deep` back to
    /// `Result`, put the original `?` back on the layered arms, and the
    /// module stayed green: the five functions that had held all 21 `?`
    /// sites were not pinned at all. The type system enforces nothing about a
    /// signature that no binding names.
    #[test]
    fn every_check_path_signature_is_total() {
        let _: fn(&Path, bool, bool) -> i32 = cmd_check;
        let _: fn(&Path, bool, bool) -> CheckDirectoryReport = check_directory;
        let _: fn(&Path, bool, bool) -> CheckResult = cmd_check_one;
        let _: fn(&Path, bool, bool) -> CheckResult = cmd_check_one_on_grown_stack;
        let _: fn(&str, bool) -> CheckResult = cmd_check_one_deep;
        let _: CheckPreparedForCli = check_prepared_for_cli;
        let _: AssembleCheckReport = assemble_check_report;
        let _: fn(&[String]) -> CheckResult = synthetic_check_report_with_errors;
        let _: fn(&str) -> CheckResult = synthetic_check_report_with_error;
    }

    /// The keystone. If this binding ever needs `?` or `.unwrap()`, the
    /// report producer has regained a failure mode and every caller can
    /// propagate again.
    #[test]
    fn the_report_producer_is_total() {
        let report: CheckResult = synthetic_check_report_with_errors(&["probe".to_string()]);
        let json: String = report.to_report_json().expect("the report renders");
        let parsed: serde_json::Value =
            serde_json::from_str(&json).expect("the producer emits JSON");
        assert_eq!(parsed["errors"].as_array().map(Vec::len), Some(1));
    }

    /// The singular wrapper is total too, and agrees with the plural form.
    #[test]
    fn the_singular_wrapper_is_total_and_agrees() {
        let one = synthetic_check_report_with_error(EMPTY_PROGRAM_MESSAGE)
            .to_report_json()
            .expect("render");
        let plural = synthetic_check_report_with_errors(&[EMPTY_PROGRAM_MESSAGE.to_string()])
            .to_report_json()
            .expect("render");
        assert_eq!(one, plural, "the wrapper is the plural form at n = 1");
    }

    /// One diagnostic per message, at several counts. The count is the
    /// property a joined string would break, and it is what makes
    /// `errors.len()` mean "how many problems".
    #[test]
    fn every_message_becomes_its_own_diagnostic() {
        for count in [0usize, 1, 2, 17] {
            let messages: Vec<String> = (0..count).map(|i| format!("issue {i}")).collect();
            let json = synthetic_check_report_with_errors(&messages)
                .to_report_json()
                .expect("render");
            let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON");
            assert_eq!(
                parsed["errors"].as_array().map(Vec::len),
                Some(count),
                "{count} messages must be {count} diagnostics; got {json}"
            );
        }
    }

    /// The producer never inspects a message, so no message can make it
    /// fail. A red-team round confirmed this empirically against emoji, RTL
    /// overrides, WTF-8 surrogates and a 6000-character name; this pins the
    /// cheap end of that range so a future `expect` cannot start panicking
    /// on content.
    #[test]
    fn no_message_content_can_break_the_producer() {
        let messages = vec![
            String::new(),
            "\u{202e}rtl override".to_string(),
            "emoji \u{1f600}".to_string(),
            "x".repeat(6000),
            "quote \" backslash \\ brace }".to_string(),
        ];
        let json = synthetic_check_report_with_errors(&messages)
            .to_report_json()
            .expect("render");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("JSON");
        assert_eq!(
            parsed["errors"].as_array().map(Vec::len),
            Some(messages.len())
        );
    }
}

#[cfg(test)]
mod advisory_lint_scope_tests {
    use super::advisory_lint_scope;
    use std::path::Path;

    #[test]
    fn advisory_lint_scope_is_the_explicit_input_file() {
        let file = Path::new("/tmp/oracle_fixture.dp");
        assert_eq!(advisory_lint_scope(file), file);
        assert_ne!(advisory_lint_scope(file), file.parent().unwrap());
    }
}

/// The inferred-signature rows, with a failure transported as a diagnostic
/// on the report instead of propagated (chelis#886 [04-FIT-12]).
///
/// The check itself succeeded here; it is the REPORTING of inferred
/// signatures that failed. Propagating would discard a complete, valid
/// report over an optional member the caller merely asked to see -- and
/// would emit nothing at all, which is the shape the atom forbids. The
/// member is omitted and the reason is carried where a consumer can read it.
fn inferred_signatures_or_diagnostic(
    checked: &chelis_types::CheckedProgram,
    fitness: &mut chelis_types::FitnessReport,
) -> Option<Vec<chelis_compiler_api::schema::WireInferredSignature>> {
    match inferred_signatures_value(checked) {
        Ok(rows) => Some(rows),
        Err(error) => {
            let message = format!("inferred signatures unavailable: {error}");
            // stderr keeps a line, as on every other path this series
            // transports: the atom asks for the failure to REACH the report,
            // not for the terminal line to be taken away. When this failure
            // propagated, `main`'s error arm printed it; now nothing would.
            eprintln!("error: {message}");
            fitness.errors.push(chelis_types::errors::CheckError {
                kind: chelis_types::errors::CheckErrorKind::Other,
                message,
                severity: 0.5,
                expected: None,
                got: None,
                span_offset: None,
                span_id: None,
                suggestions: Vec::new(),
            });
            None
        }
    }
}

fn inferred_signatures_value(
    checked: &chelis_types::CheckedProgram,
) -> Result<Vec<chelis_compiler_api::schema::WireInferredSignature>, Box<dyn std::error::Error>> {
    use chelis_compiler_api::schema::{WireInferredParameter, WireInferredSignature};
    let effect_rows = chelis_effects::def_effect_rows(checked);
    let entries = checked
        .signature_inference()
        .functions
        .values()
        .map(|func| {
            let params = func
                .params
                .iter()
                .map(|param| {
                    Ok(WireInferredParameter {
                        index: u64::try_from(param.index).map_err(|_| {
                            boxed_string_error("parameter index exceeds uint64".into())
                        })?,
                        name: param.name.clone(),
                        written: param.written,
                        inferred_read_only: param.inferred_read_only,
                        checked_type: format_cli_type(&param.checked_type),
                        display_type: format_cli_type(&param.display_type),
                        checked_type_structured: wire_inferred_type(&param.checked_type)?,
                        display_type_structured: wire_inferred_type(&param.display_type)?,
                    })
                })
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            let effect_row = effect_rows.get(&func.name);
            Ok(WireInferredSignature {
                function: func.name.clone(),
                recursive_cycle: func.recursive_cycle,
                checked_signature: format_cli_type(&func.checked_signature),
                display_signature: format_cli_type(&func.display_signature),
                checked_signature_structured: wire_inferred_type(&func.checked_signature)?,
                display_signature_structured: wire_inferred_type(&func.display_signature)?,
                effect_row: wire_effect_row(effect_row),
                effect_row_display: effect_row_display(effect_row),
                params: params.try_into().map_err(boxed_string_error)?,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    Ok(entries)
}

/// Convert a checker [`Type`] into the lossless, serde-friendly
/// [`WireInferredType`] tree emitted by `chelis check --show-inferred
/// --json`. This is the structured counterpart to [`format_cli_type`];
/// the two must stay in lockstep on every `Type` variant.
fn wire_inferred_type(ty: &Type) -> Result<WireInferredType, String> {
    Ok(match ty {
        Type::Prim(prim) => WireInferredType::Prim {
            name: prim.name().to_string(),
        },
        Type::Fn(args, ret) => WireInferredType::Fn {
            args: args
                .iter()
                .map(wire_inferred_type)
                .collect::<Result<_, _>>()?,
            ret: Box::new(wire_inferred_type(ret)?),
        },
        Type::Ref(inner) => WireInferredType::Ref {
            inner: Box::new(wire_inferred_type(inner)?),
        },
        Type::Tensor(dims, prec) => WireInferredType::Tensor {
            dims: dims
                .iter()
                .map(wire_inferred_dim)
                .collect::<Result<_, _>>()?,
            precision: wire_inferred_precision(prec),
        },
        Type::Adt(name, args) => WireInferredType::Adt {
            name: name.clone(),
            args: args
                .iter()
                .map(|ty| wire_inferred_type(ty).map(WireInferredAdtArg::Type))
                .collect::<Result<_, _>>()?,
        },
        Type::KindedAdt(name, args) => WireInferredType::Adt {
            name: name.clone(),
            args: args
                .iter()
                .map(|argument| {
                    Ok(match argument {
                        NominalArg::Type(ty) => WireInferredAdtArg::Type(wire_inferred_type(ty)?),
                        NominalArg::Dimension(dim) => {
                            WireInferredAdtArg::Dimension(WireInferredDimensionArg::Dimension {
                                dim: wire_inferred_dim(dim)?,
                            })
                        }
                    })
                })
                .collect::<Result<_, String>>()?,
        },
        Type::Var(var) => WireInferredType::Var { id: var.0 },
        Type::Tuple(types) => WireInferredType::Tuple {
            items: types
                .iter()
                .map(wire_inferred_type)
                .collect::<Result<_, _>>()?,
        },
        Type::Unit => WireInferredType::Unit,
        Type::Error(_) => WireInferredType::Error,
    })
}

#[cfg(test)]
mod inferred_wire_extent_tests {
    use super::*;
    use chelis_types::types::Prim;

    #[test]
    fn inferred_metadata_keeps_exact_nonnegative_extents() {
        let ty = Type::Tensor(
            vec![Dim::Lit(0), Dim::Lit(i64::MAX)],
            TensorPrec::Concrete(Prim::F32),
        );
        let wire = wire_inferred_type(&ty).expect("valid metadata extents");
        let json = serde_json::to_value(wire).unwrap();
        assert_eq!(json["dims"][0]["size"], 0);
        assert_eq!(json["dims"][1]["size"], i64::MAX);
    }

    #[test]
    fn extents_are_checked_through_every_recursive_type_container() {
        for extent in [0, -1] {
            let invalid = Type::Tensor(vec![Dim::Lit(extent)], TensorPrec::Concrete(Prim::F32));
            for ty in [
                invalid.clone(),
                Type::Ref(Box::new(invalid.clone())),
                Type::Fn(vec![invalid.clone()], Box::new(Type::Unit)),
                Type::Fn(vec![], Box::new(invalid.clone())),
                Type::Tuple(vec![invalid.clone()]),
                Type::Adt("Holder".into(), vec![invalid.clone()]),
                Type::KindedAdt("Holder".into(), vec![NominalArg::Type(invalid)]),
                Type::KindedAdt(
                    "Sized".into(),
                    vec![NominalArg::Dimension(Dim::Lit(extent))],
                ),
            ] {
                let result = wire_inferred_type(&ty);
                if extent == 0 {
                    result.expect("zero extent survives every container");
                } else {
                    let error = result.expect_err("negative metadata extent");
                    assert!(error.contains("nonnegative int64"), "{error}");
                }
            }
        }
    }
}

/// Convert a checker [`Dim`] into a [`WireInferredDim`]. Structured
/// counterpart to [`format_cli_dim`].
fn wire_inferred_dim(dim: &Dim) -> Result<WireInferredDim, String> {
    use chelis_compiler_api::schema::numbers::NonnegativeExtent;
    Ok(match dim {
        Dim::Name(name) => WireInferredDim::Name { name: name.clone() },
        Dim::Var(var) => WireInferredDim::Var { id: var.0 },
        Dim::Lit(value) => WireInferredDim::Lit {
            size: NonnegativeExtent::new(*value)?,
        },
        Dim::Wildcard => WireInferredDim::Wildcard,
        Dim::Rank(rank) => WireInferredDim::Rank { id: rank.0 },
    })
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
        Type::KindedAdt(name, args) => {
            let args = args
                .iter()
                .map(|argument| match argument {
                    NominalArg::Type(ty) => format_cli_type(ty),
                    NominalArg::Dimension(dim) => format_cli_dim(dim),
                })
                .collect::<Vec<_>>();
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
    let target = BuildTarget::try_from(target).map_err(boxed_string_error)?;
    // Every build target stages the runtime this chelis carries. A runtime
    // location variable, or in a development build runtime sources changed
    // since this build, is an error reported before any output is written
    // (spec/08-backends.md §2.1).
    chelis_runtime_bundle::preflight()?;
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
    target: BuildTarget,
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
    // `entry_decls` is `Some` only when the entry program is a proper part of
    // the whole program: a reef-prepared build links the library declarations
    // ahead of the entry module's. A bare `.ch` build parses one file, and the
    // entry program IS the whole program. Recording that identity where it is
    // established, rather than rediscovering it by comparing two decl lists,
    // is what lets the entry seeds below reuse the one desugar pass
    // (chelis#2331).
    let (decls, entry_decls) = match &prepared {
        Some(prepared) => (prepared.decls.clone(), Some(prepared.entry_decls.clone())),
        None => {
            let source = fs::read_to_string(file)?;
            (chelis_surf::parser::parse_str(&source)?, None)
        }
    };
    // Wave-1 red-team M2 (#207 follow-up): align with `chelis check`
    // and reject a zero-declaration program rather than emitting a
    // degenerate no-op C function. For reef-prepared programs the
    // user's contribution lives in `entry_decls`; outside a reef the
    // raw `decls` carry it directly.
    let user_decls_empty = match &entry_decls {
        Some(entry) => entry.is_empty(),
        None => decls.is_empty(),
    };
    if user_decls_empty {
        return Err(boxed_string_error(EMPTY_PROGRAM_MESSAGE.to_string()));
    }
    let full_deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
    // Both consumers of the entry program below read nothing from it but its
    // top-level names, so derive those seeds once. When the entry program is
    // the whole program they come straight off `full_deep_exprs`, which is the
    // second desugar pass this removes (chelis#2331).
    let entry_seeds = match &entry_decls {
        Some(entry) => {
            entry_seed_names(&expanded_desugared_program(entry).map_err(boxed_string_error)?)
        }
        None => entry_seed_names(&full_deep_exprs),
    };
    // chelis#334: drop dead library defs that use an eval-only host builtin
    // (`process_run`) so an unused transitive dependency module — e.g.
    // chelis-std's `Std.Process` — cannot force them into the compiled
    // lowering target and trip the build gate. These defs can never appear
    // in a compiled artifact, so they are not part of the host-library
    // surface worth preserving. A *reachable* eval-only use is left in place
    // for the build gate to reject with a clean diagnostic.
    let full_deep_exprs = drop_unreachable_eval_only_defs(full_deep_exprs, &entry_seeds);
    let pruned_deep_exprs = prune_build_program_to_reachable_defs(&full_deep_exprs, &entry_seeds);
    // Whether build-time pruning dropped any decls. The single predicate the
    // layered-cache and cross-module-check decisions below all key off, rather
    // than re-deriving it from `.len()` comparisons across differently-sourced
    // decl lists (chelis#1176 review).
    let pruning_fired = pruned_deep_exprs.len() != full_deep_exprs.len();

    // Layered build check: when the input resolves inside a reef package and
    // the typecheck cache is enabled, reuse the cached chelis-std + dependency
    // sub-contexts (chelis#1168) instead of re-inferring the whole library.
    // This runs whether or not build-time pruning fires:
    //   - no pruning: the layered whole-program `CheckedCompilation` IS the
    //     lowering target (selected directly below);
    //   - pruning fires (any chelis-std/shell package): the layered check
    //     still covers the FULL program, so it subsumes the cross-module
    //     `checked_program_with_effects(&full_deep_exprs)` re-inference below —
    //     the ~full-library type-inference cost this cache exists to remove.
    //     (The pruned program is still re-checked for the lowering target.)
    // `check_layered_for_build` returns `Ok(None)` on ANY dependency/entry
    // type/effect/linearity error (including the opaque-encapsulation
    // violation), so the monolithic full-program check below still runs on the
    // fallback path and the error output stays byte-identical.
    let layered_full_checked: Option<chelis_compiler_api::pipeline::CheckedCompilation> =
        match &prepared {
            Some(prepared) if !chelis_compiler_api::cache_disabled() => {
                // chelis#1168: split the non-chelis-std decls into the
                // stable dependency prefix (Layer 2, cached) and the
                // volatile entry suffix (re-analyzed). The concatenation
                // equals `non_stdlib_decls`, so the composed whole program
                // is byte-identical to the pre-split two-layer path.
                let (dependency_decls, entry_layer_decls) = prepared.dependency_entry_partition();
                chelis_compiler_api::check_layered_for_build(
                    &prepared.stdlib_decls,
                    prepared.stdlib_source_digest,
                    dependency_decls,
                    entry_layer_decls,
                )
                .map_err(|e| boxed_string_error(compiler_error_messages(&e)))?
            }
            _ => None,
        };

    let preserve_host_library_surface =
        if prepared.is_none() && target == BuildTarget::C && pruning_fired {
            let full_checked = checked_program_with_effects(&full_deep_exprs)
                .map_err(|e| format!("Check errors: {e}"))?;
            shared_compiler_gate(
                chelis_compiler_api::compiler::reject_host_only_builtins_before_host_lowering(
                    &full_checked,
                    target,
                ),
            )?;
            execution_host_requires_host_backend(&full_checked)?
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
    // full program so the encapsulation diagnostic surfaces. When the layered
    // check above already covered the full program (`Some`), it performed this
    // exact whole-program check from the cached contexts, so skip the
    // redundant monolithic re-inference (chelis#1168) — the whole point of the
    // cache; only run it on the layered fallback path (`None`), where the
    // full-program error report must stay byte-identical.
    if prepared.is_some() && pruning_fired && layered_full_checked.is_none() {
        checked_program_with_effects(&full_deep_exprs).map_err(|e| format!("Check errors: {e}"))?;
    }
    let deep_exprs = if preserve_host_library_surface {
        full_deep_exprs
    } else {
        pruned_deep_exprs
    };
    let symbolic_dims = collect_symbolic_dims_from_deep(&deep_exprs);
    // Use the layered whole-program `CheckedProgram` as the lowering target
    // only when `deep_exprs` IS that same whole program — neither the eval-only
    // drop nor build pruning removed anything. The length compare against the
    // layered program's own expr count is the exact test, and it is
    // deliberately NOT `!pruning_fired`: `deep_exprs` also reflects
    // `drop_unreachable_eval_only_defs` (the layered check runs on the PRE-drop
    // decls), so when the drop shrank the program but pruning did not fire, the
    // layered program still carries the dropped eval-only defs and must not be
    // the codegen target. When either shrank it, re-check the actual (pruned,
    // post-drop) lowering target monolithically. The lengths are ordered
    // `pruned <= full(post-drop) <= layered(pre-drop)` by construction, so a
    // single equality is SUFFICIENT: it forces all three equal, and the guard
    // can never select a length-coincident-but-different program.
    let checked_compilation = match layered_full_checked {
        Some(checked) if deep_exprs.len() == checked.program().exprs().len() => checked,
        _ => checked_compilation_with_effects(&deep_exprs)
            .map_err(|e| format!("Check errors: {e}"))?,
    };
    let checked = checked_compilation.program();
    let root_manifest = build_root_manifest(checked, target);
    let requires_main = root_manifest.requires_main();
    require_build_manifest_inputs(&root_manifest, target)?;
    chelis_effects::validate_build_target(checked, target.as_str())
        .map_err(|errors| format_effect_errors(&errors))?;
    shared_compiler_gate(
        chelis_compiler_api::compiler::reject_host_only_builtins_before_host_lowering(
            checked, target,
        ),
    )?;
    let (mut dag, mut compiled_host, mut execution_host) =
        lower_build_program_for_cli(&checked_compilation, &root_manifest, target)?;
    let tensor_root_names = checked_compilation.root_metadata().tensor_names().clone();
    let entry_root_names = lowered_root_names_from_decls(
        entry_decls.as_deref().unwrap_or(decls.as_slice()),
        &deep_exprs,
        checked.type_env(),
    )?;
    let selected = tensor_root_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            if entry_root_names.iter().any(|entry| entry == name.as_str()) {
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
        BuildTarget::C => {
            let c_name = c_source_name::CSourceName::from_path(file);
            let func_name = c_name.symbol();
            if let Some(host_program) = execution_host
                .as_ref()
                .map(chelis_ir::host::HostExecutionPlan::program)
                .or(compiled_host.as_ref())
                && (requires_main
                    || chelis_ir::host::host_program_requires_host_backend(host_program)
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
                         `grad(local, wrt=arg)(arg)` where `local` is a locally-bound \
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
                apply_shared_host_builtin_gates(host_program, BuildTarget::C)?;
                shared_compiler_gate(match execution_host.as_ref() {
                    Some(plan) => chelis_compiler_api::compiler::reject_unsupported_effect_ops_in_host_execution_plan(
                        plan,
                        BuildTarget::C,
                    ),
                    None => chelis_compiler_api::compiler::reject_unsupported_effect_ops_in_host_program(
                        host_program,
                        BuildTarget::C,
                    ),
                })?;
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_windowed_reductions_in_host_program(
                        host_program,
                        BuildTarget::C,
                    ),
                )?;
                let verified = if let Some(plan) = execution_host.take() {
                    verified_host_execution_codegen_program(checked, &root_manifest, plan)?
                } else {
                    verified_host_codegen_program(
                        checked,
                        &root_manifest,
                        BuildTarget::C,
                        compiled_host.take().expect("ordinary C host selected"),
                    )?
                };
                let result = chelis_backend_c::codegen_host_program(&verified, func_name)?;
                cmd_build_c_result(result, &c_name, output, &symbolic_dims, requires_main)
            } else {
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops(
                        &dag,
                        BuildTarget::C,
                    ),
                )?;
                apply_shared_window_gates(&dag, BuildTarget::C)?;
                let specialized = chelis_ir::specialize::specialize_for_exact_arithmetic(&dag);
                let fused = chelis_ir::fuse::fuse(&specialized);
                cmd_build_c(fused, &c_name, output, &symbolic_dims, &root_manifest)
            }
        }
        BuildTarget::Hip => {
            if let Some(host_program) = compiled_host.as_ref() {
                apply_shared_host_builtin_gates(host_program, BuildTarget::Hip)?;
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops_in_host_program(
                        host_program,
                        BuildTarget::Hip,
                    ),
                )?;
            }
            let host_requires_host_backend = compiled_host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            // NOTE: for programs without a `main` and with multiple
            // sibling tensor-signature defs, the "preferred" entry falls
            // back to the last fn and silently drops the others. This is a
            // known HIP backend limitation — the backend is single-entry
            // by design. Tracked as a residual issue.
            let preferred_entry = compiled_host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name);
            let preferred_entry_is_host = match preferred_entry {
                Some(name) => {
                    chelis_compiler_api::target_capability::hip_entry_lane(checked, name)?
                        == chelis_types::types::Lane::Host
                }
                None => false,
            };
            let preferred_entry_dag = preferred_entry
                .and_then(|name| chelis_ir::host::lower_named_tensor_entry_dag(checked, name));
            let has_host_roots = root_manifest
                .entries
                .iter()
                .any(|entry| entry.lane == chelis_types::types::Lane::Host);
            // chelis#2575: the single DAG entry would drop a def whose
            // parameter it cannot carry; the host backend keeps it, as C does.
            let keeps_host_signature = compiled_host
                .as_ref()
                .is_some_and(chelis_ir::host::host_program_keeps_signature_outside_dag_entry);
            if (has_host_roots
                || preferred_entry_is_host
                || keeps_host_signature
                || (dag.roots().is_empty()
                    && preferred_entry_dag.is_none()
                    && host_requires_host_backend))
                && let Some(host_program) = compiled_host.as_mut()
            {
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_hip_ops_in_host_program(
                        host_program,
                    ),
                )?;
                let selected = std::mem::take(host_program);
                // The helper manifest is read before C payload selection so a
                // Count-bearing helper reaches the HIP backend as its source
                // DAG and is lowered exactly like a HIP tensor entry.
                let (helpers, selected) =
                    chelis_backend_c::host_tensor_helper_codegen(selected, func_name)?;
                let verified = verified_host_codegen_program(
                    checked,
                    &root_manifest,
                    BuildTarget::Hip,
                    selected,
                )?;
                let result =
                    chelis_backend_hip::codegen_hip_host_program(&verified, func_name, helpers)?;
                cmd_build_hip_host(result, func_name, output, requires_main)
            } else {
                let mut hip_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    lower_checked_for_cli(checked_compilation.clone(), compiled_host.as_ref())?
                };
                hip_dag = chelis_ir::optimize::dead_code_eliminate(&hip_dag);
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops(
                        &hip_dag,
                        BuildTarget::Hip,
                    ),
                )?;
                let specialized = chelis_ir::specialize::specialize_for_blas(&hip_dag);
                shared_compiler_gate(chelis_compiler_api::compiler::reject_unsupported_hip_ops(
                    &specialized,
                ))?;
                let fused = chelis_ir::fuse::fuse(&specialized);
                cmd_build_hip(
                    fused,
                    func_name,
                    file,
                    output,
                    &symbolic_dims,
                    &root_manifest,
                )
            }
        }
        BuildTarget::Metal => {
            if let Some(host_program) = compiled_host.as_ref() {
                apply_shared_host_builtin_gates(host_program, BuildTarget::Metal)?;
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops_in_host_program(
                        host_program,
                        BuildTarget::Metal,
                    ),
                )?;
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_metal_ops_in_host_program(
                        host_program,
                    ),
                )?;
            }
            let host_requires_host_backend = compiled_host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            // Metal has no host-value lane. Validate any required host form
            // through the C emitter's fallible ABI projection before choosing
            // a Metal DAG entry, so unsupported recursive function values are
            // rejected instead of being silently dropped (#879). A genuinely
            // host-only program keeps the existing fallback artifact path.
            let preferred_entry_dag = compiled_host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name)
                .and_then(|name| chelis_ir::host::lower_named_tensor_entry_dag(checked, name));
            // chelis#2575: as on HIP, a def whose parameter the DAG entry
            // cannot carry keeps its authored signature on the host backend.
            let keeps_host_signature = compiled_host
                .as_ref()
                .is_some_and(chelis_ir::host::host_program_keeps_signature_outside_dag_entry);
            let validated_host = if host_requires_host_backend {
                if let Some(selected) = compiled_host.take() {
                    // The helper manifest is read before C payload selection
                    // so a Count-bearing helper reaches the Metal backend as
                    // its source DAG and is lowered exactly like a Metal
                    // tensor entry.
                    let (helpers, selected) =
                        chelis_backend_c::host_tensor_helper_codegen(selected, func_name)?;
                    let verified = verified_host_codegen_program(
                        checked,
                        &root_manifest,
                        BuildTarget::Metal,
                        selected,
                    )?;
                    Some(chelis_backend_metal::codegen_metal_host_program(
                        &verified, func_name, helpers,
                    )?)
                } else {
                    None
                }
            } else {
                None
            };
            if (keeps_host_signature || (dag.roots().is_empty() && preferred_entry_dag.is_none()))
                && host_requires_host_backend
                && let Some(result) = validated_host
            {
                cmd_build_metal_host(result, func_name, output, requires_main)
            } else {
                let mut metal_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    lower_checked_for_cli(checked_compilation.clone(), compiled_host.as_ref())?
                };
                metal_dag = chelis_ir::optimize::dead_code_eliminate(&metal_dag);
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops(
                        &metal_dag,
                        BuildTarget::Metal,
                    ),
                )?;
                shared_compiler_gate(chelis_compiler_api::compiler::reject_unsupported_metal_ops(
                    &metal_dag,
                ))?;
                // F4: IR validation pass for the Metal admissible-precision
                // matrix per spec/04-type-system.md §1.1.3. The spec names
                // three rejection surfaces; this is the second (the CLI
                // gate `reject_unsupported_metal_ops` above is the first;
                // `Emitter::require_metal_admissible` in the backend is
                // the third). Each independently enforces the same target
                // boundary; this typed gate owns the public early diagnostic.
                chelis_ir::verify::validate_metal_admissible_precisions(&metal_dag)?;
                let fused = chelis_ir::fuse::fuse(&metal_dag);
                cmd_build_metal(fused, func_name, file, output, &symbolic_dims)
            }
        }
    }
}

/// Deep-source ingestion path for `chelis build`.
///
/// Mirrors the shape of `cmd_build` but reads Deep s-expression text
/// directly via `chelis_deep::parse_and_stamp_file` and skips the
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
    target: BuildTarget,
    allow_style_violations: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    style_gate::enforce_style_gate(file, &source, allow_style_violations)?;
    let deep_source = style_gate::strip_deep_lint_directive_lines(&source);
    let deep_exprs = chelis_deep::parse_and_stamp_file(&deep_source)
        .map_err(|err| format!("Deep parse error: {err}"))?;

    // Deep ingestion has no separate "entry decls" concept — the whole
    // .dp file is the program. Treat every top-level def as an entry
    // candidate; the existing pruner (`prune_build_program_to_reachable_defs`)
    // will trim unreachable defs.
    let entry_seeds = entry_seed_names(&deep_exprs);
    let pruned_deep_exprs = prune_build_program_to_reachable_defs(&deep_exprs, &entry_seeds);
    let preserve_host_library_surface =
        if target == BuildTarget::C && pruned_deep_exprs.len() != deep_exprs.len() {
            let full_checked = checked_program_with_effects(&deep_exprs)
                .map_err(|e| format!("Check errors: {e}"))?;
            shared_compiler_gate(
                chelis_compiler_api::compiler::reject_host_only_builtins_before_host_lowering(
                    &full_checked,
                    target,
                ),
            )?;
            execution_host_requires_host_backend(&full_checked)?
        } else {
            false
        };
    let final_deep_exprs = if preserve_host_library_surface {
        deep_exprs.clone()
    } else {
        pruned_deep_exprs
    };
    let symbolic_dims = collect_symbolic_dims_from_deep(&final_deep_exprs);
    let checked_compilation = checked_compilation_with_effects(&final_deep_exprs)
        .map_err(|e| format!("Check errors: {e}"))?;
    let checked = checked_compilation.program();
    let root_manifest = build_root_manifest(checked, target);
    let requires_main = root_manifest.requires_main();
    require_build_manifest_inputs(&root_manifest, target)?;
    chelis_effects::validate_build_target(checked, target.as_str())
        .map_err(|errors| format_effect_errors(&errors))?;
    shared_compiler_gate(
        chelis_compiler_api::compiler::reject_host_only_builtins_before_host_lowering(
            checked, target,
        ),
    )?;
    let (mut dag, mut compiled_host, mut execution_host) =
        lower_build_program_for_cli(&checked_compilation, &root_manifest, target)?;
    let tensor_root_names = checked_compilation.root_metadata().tensor_names().clone();
    let entry_root_names = lowered_root_names_from_exprs(&deep_exprs, checked.type_env());
    let selected = tensor_root_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            if entry_root_names.iter().any(|entry| entry == name.as_str()) {
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
        BuildTarget::C => {
            let c_name = c_source_name::CSourceName::from_path(file);
            let func_name = c_name.symbol();
            if let Some(host_program) = execution_host
                .as_ref()
                .map(chelis_ir::host::HostExecutionPlan::program)
                .or(compiled_host.as_ref())
                && (requires_main
                    || chelis_ir::host::host_program_requires_host_backend(host_program)
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
                apply_shared_host_builtin_gates(host_program, BuildTarget::C)?;
                shared_compiler_gate(match execution_host.as_ref() {
                    Some(plan) => chelis_compiler_api::compiler::reject_unsupported_effect_ops_in_host_execution_plan(
                        plan,
                        BuildTarget::C,
                    ),
                    None => chelis_compiler_api::compiler::reject_unsupported_effect_ops_in_host_program(
                        host_program,
                        BuildTarget::C,
                    ),
                })?;
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_windowed_reductions_in_host_program(
                        host_program,
                        BuildTarget::C,
                    ),
                )?;
                let verified = if let Some(plan) = execution_host.take() {
                    verified_host_execution_codegen_program(checked, &root_manifest, plan)?
                } else {
                    verified_host_codegen_program(
                        checked,
                        &root_manifest,
                        BuildTarget::C,
                        compiled_host.take().expect("ordinary C host selected"),
                    )?
                };
                let result = chelis_backend_c::codegen_host_program(&verified, func_name)?;
                cmd_build_c_result(result, &c_name, output, &symbolic_dims, requires_main)
            } else {
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops(
                        &dag,
                        BuildTarget::C,
                    ),
                )?;
                apply_shared_window_gates(&dag, BuildTarget::C)?;
                let specialized = chelis_ir::specialize::specialize_for_exact_arithmetic(&dag);
                let fused = chelis_ir::fuse::fuse(&specialized);
                cmd_build_c(fused, &c_name, output, &symbolic_dims, &root_manifest)
            }
        }
        BuildTarget::Hip => {
            if let Some(host_program) = compiled_host.as_ref() {
                apply_shared_host_builtin_gates(host_program, BuildTarget::Hip)?;
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops_in_host_program(
                        host_program,
                        BuildTarget::Hip,
                    ),
                )?;
            }
            let host_requires_host_backend = compiled_host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            let preferred_entry = compiled_host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name);
            let preferred_entry_is_host = match preferred_entry {
                Some(name) => {
                    chelis_compiler_api::target_capability::hip_entry_lane(checked, name)?
                        == chelis_types::types::Lane::Host
                }
                None => false,
            };
            let preferred_entry_dag = preferred_entry
                .and_then(|name| chelis_ir::host::lower_named_tensor_entry_dag(checked, name));
            let has_host_roots = root_manifest
                .entries
                .iter()
                .any(|entry| entry.lane == chelis_types::types::Lane::Host);
            // chelis#2575: the single DAG entry would drop a def whose
            // parameter it cannot carry; the host backend keeps it, as C does.
            let keeps_host_signature = compiled_host
                .as_ref()
                .is_some_and(chelis_ir::host::host_program_keeps_signature_outside_dag_entry);
            if (has_host_roots
                || preferred_entry_is_host
                || keeps_host_signature
                || (dag.roots().is_empty()
                    && preferred_entry_dag.is_none()
                    && host_requires_host_backend))
                && let Some(host_program) = compiled_host.as_mut()
            {
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_hip_ops_in_host_program(
                        host_program,
                    ),
                )?;
                let selected = std::mem::take(host_program);
                // The helper manifest is read before C payload selection so a
                // Count-bearing helper reaches the HIP backend as its source
                // DAG and is lowered exactly like a HIP tensor entry.
                let (helpers, selected) =
                    chelis_backend_c::host_tensor_helper_codegen(selected, func_name)?;
                let verified = verified_host_codegen_program(
                    checked,
                    &root_manifest,
                    BuildTarget::Hip,
                    selected,
                )?;
                let result =
                    chelis_backend_hip::codegen_hip_host_program(&verified, func_name, helpers)?;
                cmd_build_hip_host(result, func_name, output, requires_main)
            } else {
                let mut hip_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    lower_checked_for_cli(checked_compilation.clone(), compiled_host.as_ref())?
                };
                hip_dag = chelis_ir::optimize::dead_code_eliminate(&hip_dag);
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops(
                        &hip_dag,
                        BuildTarget::Hip,
                    ),
                )?;
                let specialized = chelis_ir::specialize::specialize_for_blas(&hip_dag);
                shared_compiler_gate(chelis_compiler_api::compiler::reject_unsupported_hip_ops(
                    &specialized,
                ))?;
                let fused = chelis_ir::fuse::fuse(&specialized);
                cmd_build_hip(
                    fused,
                    func_name,
                    file,
                    output,
                    &symbolic_dims,
                    &root_manifest,
                )
            }
        }
        BuildTarget::Metal => {
            if let Some(host_program) = compiled_host.as_ref() {
                apply_shared_host_builtin_gates(host_program, BuildTarget::Metal)?;
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops_in_host_program(
                        host_program,
                        BuildTarget::Metal,
                    ),
                )?;
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_metal_ops_in_host_program(
                        host_program,
                    ),
                )?;
            }
            let host_requires_host_backend = compiled_host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false);
            let preferred_entry_dag = compiled_host
                .as_ref()
                .and_then(chelis_ir::host::preferred_tensor_entry_name)
                .and_then(|name| chelis_ir::host::lower_named_tensor_entry_dag(checked, name));
            // chelis#2575: as on HIP, a def whose parameter the DAG entry
            // cannot carry keeps its authored signature on the host backend.
            let keeps_host_signature = compiled_host
                .as_ref()
                .is_some_and(chelis_ir::host::host_program_keeps_signature_outside_dag_entry);
            let validated_host = if host_requires_host_backend {
                if let Some(selected) = compiled_host.take() {
                    // The helper manifest is read before C payload selection
                    // so a Count-bearing helper reaches the Metal backend as
                    // its source DAG and is lowered exactly like a Metal
                    // tensor entry.
                    let (helpers, selected) =
                        chelis_backend_c::host_tensor_helper_codegen(selected, func_name)?;
                    let verified = verified_host_codegen_program(
                        checked,
                        &root_manifest,
                        BuildTarget::Metal,
                        selected,
                    )?;
                    Some(chelis_backend_metal::codegen_metal_host_program(
                        &verified, func_name, helpers,
                    )?)
                } else {
                    None
                }
            } else {
                None
            };
            if (keeps_host_signature || (dag.roots().is_empty() && preferred_entry_dag.is_none()))
                && host_requires_host_backend
                && let Some(result) = validated_host
            {
                cmd_build_metal_host(result, func_name, output, requires_main)
            } else {
                let mut metal_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    lower_checked_for_cli(checked_compilation.clone(), compiled_host.as_ref())?
                };
                metal_dag = chelis_ir::optimize::dead_code_eliminate(&metal_dag);
                shared_compiler_gate(
                    chelis_compiler_api::compiler::reject_unsupported_effect_ops(
                        &metal_dag,
                        BuildTarget::Metal,
                    ),
                )?;
                shared_compiler_gate(chelis_compiler_api::compiler::reject_unsupported_metal_ops(
                    &metal_dag,
                ))?;
                // F4: IR validation pass; see cmd_build for the full
                // rationale. This is the same surface from the Deep
                // ingestion path so symbolic-dim and span-attributed
                // Deep get the same f64 rejection behavior.
                chelis_ir::verify::validate_metal_admissible_precisions(&metal_dag)?;
                let fused = chelis_ir::fuse::fuse(&metal_dag);
                cmd_build_metal(fused, func_name, file, output, &symbolic_dims)
            }
        }
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
            conform_preflight(&root, "sync")?;
            // The preflight guarantees a readable pin, so the version stamped is
            // always the SHELL's, never the toolchain's (chelis#1263). Falling
            // back to `COMPILER_VERSION` here used to stamp managed blocks for a
            // version the shell had not adopted, and exit 0 doing it.
            let version = chelis_conformance::audit::audit(&root)
                .reef_pin
                .map(|p| p.trim_start_matches('=').to_string())
                .ok_or_else(|| {
                    format!(
                        "conform sync: {}/reef.toml has no readable compiler pin (the preflight \
                         should have caught this)",
                        root.display()
                    )
                })?;
            let mut written: Vec<String> = Vec::new();
            let notices = report_partial_writes(
                chelis_conformance::scaffold::materialize_skills(&root),
                &written,
                "agent-skills/, .claude/skills/, and .codex/skills/",
            )?;
            written
                .extend(["agent-skills/", ".claude/skills/", ".codex/skills/"].map(str::to_string));
            for notice in notices {
                eprintln!("note: {notice}");
            }
            report_partial_writes(
                chelis_conformance::scaffold::sync_managed_blocks(&root, &version),
                &written,
                "the managed blocks in AGENTS.md / docs/CHELIS_SURFACE.md",
            )?;
            println!(
                "synced managed blocks + skill links to chelis {version} at {}",
                root.display()
            );
        }
        ConformCommand::Bump { version, path } => {
            let root = match path {
                Some(p) => p,
                None => env::current_dir()?,
            };
            // Fail closed BEFORE the first write (chelis#1263). The bump's edit
            // sequence used to run until it hit the first missing artifact,
            // leaving a half-bumped tree behind whichever step died.
            conform_preflight(&root, "bump")?;
            let mut written: Vec<String> = Vec::new();
            // The pin rewrite is the one step the preflight cannot make
            // all-or-nothing (it edits several files in sequence), so its error
            // path carries what it had already written.
            let changed = match chelis_conformance::bump::rewrite_pins(&root, &version) {
                Ok(changed) => changed,
                Err(e) => {
                    let already: Vec<String> =
                        e.written.iter().map(|p| repo_relative(&root, p)).collect();
                    return Err(partial_write_error(
                        e.message,
                        &already,
                        "the pin locations (reef.toml and the workflow env pins)",
                    ));
                }
            };
            for p in &changed {
                let rel = repo_relative(&root, p);
                println!("repinned {rel}");
                written.push(rel);
            }
            let notices = report_partial_writes(
                chelis_conformance::scaffold::materialize_skills(&root),
                &written,
                "agent-skills/, .claude/skills/, and .codex/skills/",
            )?;
            written
                .extend(["agent-skills/", ".claude/skills/", ".codex/skills/"].map(str::to_string));
            for notice in notices {
                eprintln!("note: {notice}");
            }
            report_partial_writes(
                chelis_conformance::scaffold::sync_managed_blocks(&root, &version),
                &written,
                "the managed blocks in AGENTS.md / docs/CHELIS_SURFACE.md",
            )?;
            println!("restamped managed blocks + skill links to chelis {version}");

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

/// Refuse a `conform` write verb on a repo that has not been conformed
/// (chelis#1263). Runs before the verb's first write, so a refusal leaves the
/// tree exactly as it found it and the message can say so without qualification.
fn conform_preflight(root: &Path, verb: &str) -> Result<(), Box<dyn std::error::Error>> {
    match chelis_conformance::scaffold::preflight_restamp_targets(root) {
        Ok(()) => Ok(()),
        Err(gaps) => {
            Err(chelis_conformance::scaffold::preflight_failure_message(verb, root, &gaps).into())
        }
    }
}

/// `path` relative to the shell root, for report output. Every path a `conform`
/// verb prints is repo-relative, so a reader can act on it without first
/// mentally stripping whatever `--path` happened to be.
fn repo_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .display()
        .to_string()
}

/// Build the error for a write step that failed partway: what it was writing,
/// and what this run had already written (chelis#1263). A nonzero exit that
/// abandons an edit sequence has to say which edits landed; the caller cannot be
/// left to diff the tree.
fn partial_write_error(
    message: String,
    written: &[String],
    in_progress: &str,
) -> Box<dyn std::error::Error> {
    let mut msg = message;
    msg.push_str(&format!(
        "\nfailed while writing {in_progress}, which may be partially written"
    ));
    if !written.is_empty() {
        msg.push_str("\nalready written by this run: ");
        msg.push_str(&written.join(", "));
    }
    Box::<dyn std::error::Error>::from(msg)
}

/// [`partial_write_error`] applied to a step that returns `Result<T, String>`.
/// The preflight makes the common pre-conformance case unreachable here, so this
/// covers the residue an offline tool cannot preflight away (a read-only file, a
/// full disk, a concurrent edit).
fn report_partial_writes<T>(
    result: Result<T, String>,
    written: &[String],
    in_progress: &str,
) -> Result<T, Box<dyn std::error::Error>> {
    result.map_err(|e| partial_write_error(e, written, in_progress))
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
// no explicit impl needed. The wrapper exists so the guard is a value the
// handoff below can own -- shared through an `Arc`, so the file outlives
// every worker that could still open it -- and so the tempfile path can be
// used by every spawned worker without leaking the file handle into worker
// subprocesses (workers reopen the path themselves).

/// Env var naming the tempfile a `chelis test` parent wrote for its workers.
const COMPILED_CONTEXT_PATH_ENV: &str = "CHELIS_TEST_COMPILED_CONTEXT";

/// Env var carrying that tempfile's payload digest.
///
/// The digest is what makes the handoff authentic, and it is only evidence
/// because it arrives on a different channel from the bytes. A process that can
/// replace the tempfile between the parent's `sync_all` and the worker's `open`
/// -- which file mode `0600` does not prevent when `TMPDIR` names a directory
/// other users can write, the ordinary shape of a shared build machine --
/// cannot also reach into the environment of a child the parent already
/// spawned.
const COMPILED_CONTEXT_DIGEST_ENV: &str = "CHELIS_TEST_COMPILED_CONTEXT_SHA256";

/// The complete parent-to-worker handoff: the file holding the bytes, and the
/// digest that says they are still the bytes the parent wrote.
///
/// These two travel together in this process and apart between processes, which
/// is the whole point. Keeping them in one value means a spawn site cannot pass
/// the path and forget the digest.
///
/// It owns the tempfile guard through an `Arc` rather than borrowing its path.
/// Worker threads need `'static`, so a borrow could not have reached them, and
/// a bare `PathBuf` would leave "the file still exists when the worker opens
/// it" resting on the order two locals happen to be declared in. Every clone
/// handed to a thread keeps the guard alive, so the file outlives the last
/// worker that could read it by construction.
#[derive(Clone)]
struct CompiledContextHandoff {
    tempfile: Arc<CompiledContextTempfile>,
    digest_hex: String,
}

impl CompiledContextHandoff {
    fn new(tempfile: CompiledContextTempfile, digest: &chelis_compiler_api::HandoffDigest) -> Self {
        Self {
            tempfile: Arc::new(tempfile),
            digest_hex: digest.to_hex(),
        }
    }

    fn apply_to(&self, cmd: &mut std::process::Command) {
        cmd.env(COMPILED_CONTEXT_PATH_ENV, self.tempfile.path())
            .env(COMPILED_CONTEXT_DIGEST_ENV, &self.digest_hex);
    }

    /// Clear both variables when this parent has no context to hand over, so an
    /// inherited pair from the outer environment cannot stand in for one the
    /// parent did not produce.
    fn clear_from(cmd: &mut std::process::Command) {
        cmd.env_remove(COMPILED_CONTEXT_PATH_ENV)
            .env_remove(COMPILED_CONTEXT_DIGEST_ENV);
    }
}

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
            let deadline = effective_output_forwarding_deadline(
                deadline,
                testing_hook_enabled("CHELIS_TEST_EXPIRE_OUTPUT_FORWARDING_DEADLINE"),
            );
            let stderr_forwarded = write_stream_bounded(
                OutputStream::Stderr,
                output.output.stderr,
                deadline.saturating_duration_since(Instant::now()),
            );
            if !stderr_forwarded {
                const FAILURE_REPORT_GRACE: Duration = Duration::from_secs(1);
                const SAME_STREAM_PROBE: Duration = Duration::from_millis(100);
                let fallback_deadline = Instant::now()
                    .checked_add(FAILURE_REPORT_GRACE)
                    .unwrap_or_else(Instant::now);
                // Missing the worker deadline does not prove that stderr is
                // blocked: under scheduler pressure the worker may never run.
                // Probe stderr on the caller before switching the only honest
                // incomplete-output report to stdout.
                let stderr_reported = write_fallback_stream_bounded(
                    OutputStream::Stderr,
                    output_forwarding_failure_diagnostic(suite_timeout_secs),
                    SAME_STREAM_PROBE,
                );
                if !stderr_reported {
                    let _ = write_fallback_stream_bounded(
                        OutputStream::Stdout,
                        output_forwarding_failure_report(json, expect, suite_timeout_secs),
                        fallback_deadline.saturating_duration_since(Instant::now()),
                    );
                }
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
        let _ = write_fallback_stream_bounded(
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
        let mut seen = UnordSet::<String>::new();
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
    let _ = write_fallback_stream_bounded(
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

fn effective_output_forwarding_deadline(deadline: Instant, force_expired: bool) -> Instant {
    if force_expired {
        Instant::now()
    } else {
        deadline
    }
}

fn write_stream_bounded(stream: OutputStream, bytes: Vec<u8>, budget: Duration) -> bool {
    if bytes.is_empty() {
        return true;
    }
    let (done_tx, done_rx) = std::sync::mpsc::channel::<bool>();
    thread::spawn(move || {
        if testing_hook_enabled("CHELIS_TEST_FAIL_BOUNDED_WRITER") {
            let _ = done_tx.send(false);
            return;
        }
        if testing_hook_enabled("CHELIS_TEST_DELAY_BOUNDED_WRITER") {
            thread::sleep(Duration::from_secs(2));
        }
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

fn write_fallback_stream_bounded(stream: OutputStream, bytes: Vec<u8>, budget: Duration) -> bool {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;

        let fd = match stream {
            OutputStream::Stdout => io::stdout().as_raw_fd(),
            OutputStream::Stderr => io::stderr().as_raw_fd(),
        };
        write_fd_bounded(fd, &bytes, budget)
    }

    #[cfg(not(unix))]
    {
        write_stream_bounded(stream, bytes, budget)
    }
}

#[cfg(unix)]
fn write_fd_bounded(fd: std::os::fd::RawFd, bytes: &[u8], budget: Duration) -> bool {
    if bytes.is_empty() {
        return true;
    }

    let original_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
    if original_flags < 0 {
        return false;
    }
    if unsafe { libc::fcntl(fd, libc::F_SETFL, original_flags | libc::O_NONBLOCK) } < 0 {
        return false;
    }

    // This path already follows a primary writer timeout. It must not depend on
    // another newly spawned writer thread being scheduled before the command
    // dispatcher calls process::exit. Write on the calling thread instead,
    // with O_NONBLOCK keeping the fallback diagnostic bounded when this stream
    // is blocked too.
    let deadline = Instant::now().checked_add(budget);
    let wrote_all = (|| {
        let mut offset = 0;
        while offset < bytes.len() {
            let written = unsafe {
                libc::write(
                    fd,
                    bytes[offset..].as_ptr().cast::<libc::c_void>(),
                    bytes.len() - offset,
                )
            };
            if written > 0 {
                offset += written as usize;
                continue;
            }
            if written == 0 {
                return false;
            }

            let error = std::io::Error::last_os_error();
            match error.kind() {
                std::io::ErrorKind::Interrupted => continue,
                std::io::ErrorKind::WouldBlock => {
                    let Some(deadline) = deadline else {
                        return false;
                    };
                    let now = Instant::now();
                    if now >= deadline {
                        return false;
                    }
                    thread::sleep(
                        deadline
                            .saturating_duration_since(now)
                            .min(Duration::from_millis(1)),
                    );
                }
                _ => return false,
            }
        }
        true
    })();

    let restored = unsafe { libc::fcntl(fd, libc::F_SETFL, original_flags) } >= 0;
    wrote_all && restored
}

#[cfg(all(test, unix))]
mod bounded_stream_write_tests {
    use super::{effective_output_forwarding_deadline, write_fd_bounded};
    use std::io::{ErrorKind, Read, Write};
    use std::os::fd::AsRawFd;
    use std::os::unix::net::UnixStream;
    use std::time::{Duration, Instant};

    #[test]
    fn writable_descriptor_receives_the_complete_diagnostic() {
        let (mut reader, writer) = UnixStream::pair().expect("socket pair");
        let diagnostic = b"error: suite incomplete\n";

        assert!(write_fd_bounded(
            writer.as_raw_fd(),
            diagnostic,
            Duration::from_millis(50),
        ));

        let mut received = vec![0; diagnostic.len()];
        reader.read_exact(&mut received).expect("read diagnostic");
        assert_eq!(received, diagnostic);
    }

    #[test]
    fn full_descriptor_is_bounded_and_restores_blocking_mode() {
        let (_reader, mut writer) = UnixStream::pair().expect("socket pair");
        writer.set_nonblocking(true).expect("set nonblocking");
        let chunk = [b'x'; 4096];
        loop {
            match writer.write(&chunk) {
                Ok(0) => panic!("socket stopped accepting bytes without reporting backpressure"),
                Ok(_) => {}
                Err(err) if err.kind() == ErrorKind::Interrupted => continue,
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(err) => panic!("fill socket: {err}"),
            }
        }
        writer.set_nonblocking(false).expect("restore blocking");

        let fd = writer.as_raw_fd();
        let original_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert!(original_flags >= 0, "read descriptor flags");
        assert_eq!(original_flags & libc::O_NONBLOCK, 0);

        let started = Instant::now();
        assert!(!write_fd_bounded(
            fd,
            b"must not block",
            Duration::from_millis(25),
        ));
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "bounded descriptor write exceeded its budget"
        );

        let restored_flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
        assert_eq!(restored_flags, original_flags);
    }

    #[test]
    fn forced_output_forwarding_deadline_is_expired_without_shortening_the_control() {
        let future = Instant::now()
            .checked_add(Duration::from_secs(30))
            .expect("future deadline");

        assert_eq!(effective_output_forwarding_deadline(future, false), future);
        let forced = effective_output_forwarding_deadline(future, true);
        assert!(forced <= Instant::now());
    }
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

/// Appended to the plain-text summary line when `--batch-mode auto` abandoned a
/// batch, so a stdout-only capture can tell a degraded run from a clean one
/// (chelis#1261). A clean run's summary line is unchanged.
const PLAIN_BATCH_FALLBACK_MARKER: &str = " (batch abandoned: ran per-file)";

fn parse_plain_test_summary(line: &str) -> Option<(usize, usize)> {
    let line = line
        .strip_suffix(PLAIN_BATCH_FALLBACK_MARKER)
        .unwrap_or(line);
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
/// * `2` — runner error (missing dir, missing reef package, or zero selected tests).
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
        return emit_empty_test_selection(
            json,
            describe_empty_test_selection(
                &target,
                EmptyTestSelection::NoSourceFiles {
                    excluded: count_excluded_test_files(&target),
                },
            ),
        );
    }

    // [04-TEST-1..3]: ordinary mode must establish a non-empty runnable
    // selection before paying for Reef/context compilation. The scan is
    // conservative: unreadable, unparsable, or duplicate-bearing files fall
    // through to the normal file-failure path instead of being used as
    // evidence that the selected set is empty. `--expect` remains file-probe
    // mode and deliberately bypasses this runnable-test count.
    //
    // This also preserves Phase G's no-match fast path: a filter selecting
    // nothing still avoids the historically expensive context build, but now
    // exits 2 with a diagnostic rather than presenting a false green.
    if expect.is_none() {
        match preflight_test_selection(&test_files, &cwd, filter) {
            TestSelectionPreflight::NeedsExecution => {}
            TestSelectionPreflight::Empty(reason) => {
                return emit_empty_test_selection(
                    json,
                    describe_empty_test_selection(&target, reason),
                );
            }
        }
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
    let (context_bytes, context_digest) = context.encode_for_handoff()?;
    drop(context);
    let context_handoff = CompiledContextHandoff::new(
        CompiledContextTempfile::write(&context_bytes)?,
        &context_digest,
    );
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
            &context_handoff,
            json,
            &mut out,
        );
    }

    // `--batch-mode auto` may abandon an attempted batch and re-run its files
    // per-file. The run is still complete and its exit code still tracks test
    // outcomes only, but the report has to say the batched path was dropped:
    // a perfect-looking summary that hides a degraded execution mode is the
    // failure chelis#1261 reported.
    let mut batch_fallback = false;

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
                &context_handoff,
                json,
                &mut out,
                &mut passed,
                &mut failed,
            )?;
        }
        TestBatchMode::Auto => {
            batch_fallback = run_test_jobs_auto(
                &self_path,
                &cwd,
                &test_jobs,
                jobs,
                filter,
                timeout_secs,
                &context_handoff,
                json,
                &mut out,
                &mut passed,
                &mut failed,
                progress_file,
            )?;
        }
    }

    if json {
        // Additive: the field is absent unless a batch was abandoned, so the
        // summary record every existing consumer parses is byte-identical.
        let fallback_field = if batch_fallback {
            ",\"batch_fallback\":true"
        } else {
            ""
        };
        writeln!(
            out,
            "{{\"summary\":{{\"passed\":{passed},\"failed\":{failed}{fallback_field}}}}}"
        )
        .map_err(|e| e.to_string())?;
    } else {
        // The marker rides on the summary line itself. A CI job that captures
        // only stdout (the common shape) would otherwise read a degraded run as
        // identical to a clean one, which is chelis#1261's complaint one
        // channel over.
        let marker = if batch_fallback {
            PLAIN_BATCH_FALLBACK_MARKER
        } else {
            ""
        };
        writeln!(out, "\n{passed} passed, {failed} failed{marker}").map_err(|e| e.to_string())?;
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
    context_handoff: &CompiledContextHandoff,
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
            compiled_context_handoff: Some(context_handoff),
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

/// The same diagnostics as [`compiler_error_messages`], unjoined.
///
/// The joined form is the terminal rendering; a machine-facing carrier wants
/// one entry per diagnostic so the array length is the number of problems
/// (chelis#886). Falls back to the stage name for the same reason the joined
/// form does: an empty error list still has to say something.
fn compiler_error_list(err: &chelis_compiler_api::compiler::CompilerError) -> Vec<String> {
    let messages: Vec<String> = err
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.clone())
        .collect();
    if messages.is_empty() {
        vec![err.stage.clone()]
    } else {
        messages
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

fn shared_compiler_gate(
    result: Result<(), chelis_compiler_api::compiler::CompilerError>,
) -> Result<(), Box<dyn std::error::Error>> {
    result.map_err(|error| boxed_string_error(compiler_error_messages(&error)))
}

fn apply_shared_host_builtin_gates(
    program: &chelis_ir::host::ConcreteHostProgram,
    target: BuildTarget,
) -> Result<(), Box<dyn std::error::Error>> {
    shared_compiler_gate(chelis_compiler_api::compiler::reject_host_only_builtins(
        program, target,
    ))?;
    shared_compiler_gate(chelis_compiler_api::compiler::reject_eval_only_builtins(
        program, target,
    ))
}

fn apply_shared_window_gates(
    dag: &chelis_ir::Dag,
    target: BuildTarget,
) -> Result<(), Box<dyn std::error::Error>> {
    shared_compiler_gate(
        chelis_compiler_api::compiler::reject_symbolic_windowed_reduce(dag, target),
    )?;
    shared_compiler_gate(
        chelis_compiler_api::compiler::reject_unsupported_reduce_window_precision(dag, target),
    )
}

fn is_local_registry_hash_unsupported(err: &chelis_compiler_api::compiler::CompilerError) -> bool {
    err.errors.iter().any(|diagnostic| {
        diagnostic.kind() == DiagnosticKind::HashError
            && diagnostic.message.contains("LocalRegistry")
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
    compiled_context_handoff: &CompiledContextHandoff,
    json: bool,
    out: &mut impl Write,
    passed: &mut usize,
    failed: &mut usize,
    progress_file: Option<&Path>,
) -> Result<bool, String> {
    let classified = classify_test_jobs_for_batch(test_jobs, filter);
    let mut rows_by_index = BTreeMap::<usize, Vec<TestRow>>::new();
    let mut file_fallback_jobs = classified.file_jobs;
    let mut fallback_reason = None::<BatchFallbackReason>;

    if !classified.batch_jobs.is_empty() {
        match run_test_batch_subprocess(
            self_path,
            cwd,
            &classified.batch_jobs,
            timeout_secs,
            compiled_context_handoff,
            progress_file,
        )? {
            BatchSubprocessOutcome::Rows(rows) => {
                if let Some(reason) =
                    group_batch_rows_by_file(&classified.batch_jobs, rows, &mut rows_by_index)
                {
                    file_fallback_jobs.extend(batch_jobs_as_file_jobs(&classified.batch_jobs));
                    fallback_reason = Some(reason);
                }
            }
            BatchSubprocessOutcome::Fallback(reason) => {
                file_fallback_jobs.extend(batch_jobs_as_file_jobs(&classified.batch_jobs));
                fallback_reason = Some(reason);
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
                compiled_context_handoff: Some(compiled_context_handoff),
                expect_file_diagnostic: false,
            },
        )?);
    }

    // Reported before the rows: the batch was abandoned before any of these
    // rows existed, and a reader who stops at the first failing row still sees
    // that the run did not take the path it asked for.
    if let Some(reason) = &fallback_reason {
        emit_batch_fallback_note(out, json, &classified.batch_jobs, reason)?;
    }

    for job in test_jobs {
        if let Some(rows) = rows_by_index.remove(&job.index) {
            emit_test_file_rows(out, json, &job.rel_display, &rows, passed, failed)?;
        }
    }

    Ok(fallback_reason.is_some())
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

/// Distribute a completed batch's rows back to their owning files. Returns the
/// fallback reason when the rows cannot be attributed, `None` on success.
fn group_batch_rows_by_file(
    batch_jobs: &[TestBatchManifestFile],
    rows: Vec<TestRow>,
    rows_by_index: &mut BTreeMap<usize, Vec<TestRow>>,
) -> Option<BatchFallbackReason> {
    let expected_rows: usize = batch_jobs.iter().map(|job| job.tests.len()).sum();
    if rows.len() != expected_rows {
        return Some(BatchFallbackReason::IncompleteRows(format!(
            "expected {expected_rows} rows for the selected tests, got {}",
            rows.len()
        )));
    }
    let index_by_file = batch_jobs
        .iter()
        .map(|job| (job.rel_display.clone(), job.index))
        .collect::<UnordMap<_, _>>();
    for row in rows {
        let Some(index) = index_by_file.get(&row.file).copied() else {
            return Some(BatchFallbackReason::IncompleteRows(format!(
                "row named file `{}`, which is not in the batch",
                row.file
            )));
        };
        rows_by_index.entry(index).or_default().push(row);
    }
    None
}

fn classify_test_jobs_for_batch(
    test_jobs: &[TestFileJob],
    filter: Option<&str>,
) -> ClassifiedTestJobs {
    let mut batch_jobs = Vec::new();
    let mut file_jobs = Vec::new();
    let mut batch_scope = BatchScope::default();

    for job in test_jobs {
        let source = match fs::read_to_string(&job.file) {
            Ok(source) => source,
            Err(e) => {
                explain_batch_demotion(&job.rel_display, &format!("it could not be read: {e}"));
                file_jobs.push(job.clone());
                continue;
            }
        };
        let parsed = match chelis_surf::parser::parse_str(&source) {
            Ok(parsed) => parsed,
            Err(e) => {
                explain_batch_demotion(&job.rel_display, &format!("it could not be parsed: {e}"));
                file_jobs.push(job.clone());
                continue;
            }
        };
        let flat = flatten_module_decls(&parsed);
        let tests = match enumerate_test_fns(&flat, filter, &job.rel_display) {
            EnumerationOutcome::Tests(tests) => tests,
            EnumerationOutcome::Error(msg) => {
                explain_batch_demotion(
                    &job.rel_display,
                    &format!("its tests could not be enumerated: {msg}"),
                );
                file_jobs.push(job.clone());
                continue;
            }
        };
        if tests.is_empty() {
            continue;
        }
        if flat.iter().any(|decl| matches!(decl, Decl::LetDef { .. })) {
            explain_batch_demotion(
                &job.rel_display,
                "it has a top-level module-init binding, which a shared batch would run once \
                 for every file",
            );
            file_jobs.push(job.clone());
            continue;
        }
        let scope = test_file_scope_names(&flat);
        if let Some(collision) = batch_scope.admit(&job.rel_display, &scope) {
            explain_batch_demotion(&job.rel_display, &collision);
            file_jobs.push(job.clone());
            continue;
        }
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

/// Operator knob: when set to `1`, `--batch-mode auto` says on stderr why each
/// test file took the per-file worker path instead of the suite batch.
///
/// Demotion is by design and silent, because it produces the same rows and the
/// same exit code. The reason is computed regardless, though, and without a way
/// to read it a maintainer tuning a slow suite has to bisect the colliding
/// names by hand, which is what chelis#1261's reporter did across five of them.
const EXPLAIN_BATCHING_ENV: &str = "CHELIS_TEST_EXPLAIN_BATCHING";

fn explain_batch_demotion(rel_display: &str, reason: &str) {
    if env::var(EXPLAIN_BATCHING_ENV).as_deref() != Ok("1") {
        return;
    }
    eprintln!("note: {rel_display} is not in the suite batch: {reason}");
}

/// Top-level names one test file contributes to a shared compilation unit,
/// split by how each name entered scope.
///
/// `--batch-mode auto` merges every batched file's flattened declarations into
/// a single unit, so the batch has one top-level scope. A name that file A
/// imports and file B declares therefore resolves to B's declaration inside A
/// as well, which recompiles A against a binding it never asked for
/// (chelis#1261). Import-versus-declaration is a batch scope collision on the
/// same footing as declaration-versus-declaration, so the colliding file takes
/// the per-file worker path instead.
struct TestFileScopeNames {
    /// Names the file itself binds at top level, including ADT variant
    /// constructors (a merged unit has one constructor namespace).
    declared: Vec<String>,
    /// Explicitly imported `import M (a, b)` names, each paired with `M`.
    /// Two files importing the same name from the same module agree on what
    /// it means; two files importing it from different modules do not.
    imported: Vec<(String, String)>,
    /// Set by an `import M (..)`. This runner cannot enumerate a wildcard's
    /// name set without resolving the package graph, so it cannot prove that
    /// no sibling declaration captures one of those names.
    wildcard_import: bool,
}

fn test_file_scope_names(decls: &[Decl]) -> TestFileScopeNames {
    let mut declared = Vec::new();
    let mut imported = Vec::new();
    let mut wildcard_import = false;
    for decl in decls {
        match decl {
            Decl::FunDef { name, .. }
            | Decl::Sig { name, .. }
            | Decl::TypeAlias { name, .. }
            | Decl::MacroDef { name, .. }
            | Decl::Property { name, .. } => declared.push(name.clone()),
            Decl::TypeDef { name, variants, .. } => {
                declared.push(name.clone());
                declared.extend(variants.iter().map(|variant| variant.name.clone()));
            }
            Decl::Dim { names, .. } => declared.extend(names.iter().cloned()),
            Decl::Import { module, kind, .. } => match kind {
                ImportKind::Names(names) => {
                    imported.extend(names.iter().map(|name| (name.clone(), module.clone())));
                }
                ImportKind::All => wildcard_import = true,
                // A qualified import binds only `M.name`, which no unqualified
                // sibling declaration in the merged unit can capture.
                ImportKind::Qualified => {}
            },
            // Exhaustive on purpose. A future `Decl` variant that binds a
            // top-level name would silently contribute nothing here and reopen
            // exactly the blind spot chelis#1261 reported, so a new variant has
            // to stop this compiling until someone classifies it.
            //
            // `Module` is already flattened away before this runs. `LetDef`
            // makes a file ineligible for batching on its own (module-init
            // bindings), so its name never reaches a shared scope. `Export`
            // marks existing declarations visible and binds nothing.
            Decl::Module { .. } | Decl::LetDef { .. } | Decl::Export { .. } => {}
        }
    }
    TestFileScopeNames {
        declared,
        imported,
        wildcard_import,
    }
}

/// The accumulated top-level scope of a suite batch.
///
/// The parent's eligibility classifier and the batch worker's own guard both
/// admit files through this one type so the two can never disagree about what
/// "collision" means: a worker that rejected a file the parent had already
/// batched would turn every such suite into a silent per-file fallback.
#[derive(Default)]
struct BatchScope {
    /// Declared name to the file that declared it.
    declared: UnordMap<String, String>,
    /// Imported name to the module it came from and the file that imported it.
    imported: UnordMap<String, (String, String)>,
}

impl BatchScope {
    /// Admit `file` into the batch, or report the collision that keeps it out.
    ///
    /// Names are checked against the files already admitted before any of this
    /// file's own names are recorded, so a file that legitimately repeats a
    /// name internally (a `sig` beside its `def`) is not a collision with
    /// itself.
    fn admit(&mut self, file: &str, scope: &TestFileScopeNames) -> Option<String> {
        if scope.wildcard_import {
            return Some(format!(
                "{file} imports a whole module, and this runner cannot enumerate \
                 the names that brings into the shared batch scope"
            ));
        }
        for name in &scope.declared {
            if let Some(owner) = self.declared.get(name) {
                return Some(format!("`{name}` is declared by both {owner} and {file}"));
            }
            if let Some((module, owner)) = self.imported.get(name) {
                return Some(format!(
                    "`{name}` is declared by {file} and imported from `{module}` by {owner}"
                ));
            }
        }
        for (name, module) in &scope.imported {
            if let Some(owner) = self.declared.get(name) {
                return Some(format!(
                    "`{name}` is imported from `{module}` by {file} and declared by {owner}"
                ));
            }
            if let Some((seen_module, owner)) = self.imported.get(name)
                && seen_module != module
            {
                return Some(format!(
                    "`{name}` is imported from `{seen_module}` by {owner} \
                     and from `{module}` by {file}"
                ));
            }
        }
        for name in &scope.declared {
            self.declared.insert(name.clone(), file.to_string());
        }
        for (name, module) in &scope.imported {
            self.imported
                .entry(name.clone())
                .or_insert_with(|| (module.clone(), file.to_string()));
        }
        None
    }
}

/// Why `--batch-mode auto` gave up on an attempted suite batch and re-ran its
/// files through per-file workers.
///
/// A batch that is never attempted (no eligible files) is not a fallback: the
/// distinction is exactly what the reasonless predecessor could not express,
/// which is how an abandoned batch reached the user as one unattributed line
/// on the worker's stderr (chelis#1261).
#[derive(Debug, Clone)]
enum BatchFallbackReason {
    WorkerUnavailable(String),
    Timeout(u64),
    MalformedOutput(String),
    WorkerFailed(String),
    IncompleteRows(String),
}

impl BatchFallbackReason {
    fn status(&self) -> &'static str {
        match self {
            Self::WorkerUnavailable(_) => "worker-unavailable",
            Self::Timeout(_) => "timeout",
            Self::MalformedOutput(_) => "malformed-output",
            Self::WorkerFailed(_) => "worker-failed",
            Self::IncompleteRows(_) => "incomplete-rows",
        }
    }

    fn message(&self) -> String {
        match self {
            Self::WorkerUnavailable(detail) => {
                // Covers spawn failure and every later failure to drive the
                // process: the runner never got a verdict out of it.
                format!("batch worker could not be run: {detail}")
            }
            Self::Timeout(seconds) => {
                format!("batch worker exceeded its {seconds}s window and was terminated")
            }
            Self::MalformedOutput(detail) => {
                format!("batch worker emitted output this runner could not read: {detail}")
            }
            // Already self-describing at every construction site.
            Self::WorkerFailed(detail) => detail.clone(),
            Self::IncompleteRows(detail) => {
                format!("batch worker did not report a usable row set: {detail}")
            }
        }
    }
}

/// Report an abandoned suite batch on both channels it can reach.
///
/// The human note names the files and the reason, so the batch worker's own
/// stderr (which is inherited, and therefore already on the terminal by the
/// time this runs) stops being an unattributed line. The `--json` record is
/// additive: it is a new top-level record kind beside the existing `suite`
/// record, so a consumer that reads rows and the summary keeps parsing.
fn emit_batch_fallback_note(
    out: &mut impl Write,
    json: bool,
    batch_jobs: &[TestBatchManifestFile],
    reason: &BatchFallbackReason,
) -> Result<(), String> {
    let files = batch_jobs
        .iter()
        .map(|job| job.rel_display.clone())
        .collect::<Vec<_>>();
    let noun = if files.len() == 1 { "file" } else { "files" };
    let note = format!(
        "warning: suite batching was abandoned; {count} test {noun} re-ran through \
         per-file workers\n  reason: {reason}\n  files: {files}\n  \
         any diagnostic printed above this warning came from the abandoned batch worker\n  \
         pass `--batch-mode file` to run this suite per-file without the batch attempt\n",
        count = files.len(),
        reason = reason.message(),
        files = files.join(", "),
    );
    let stderr = io::stderr();
    let mut stderr = stderr.lock();
    stderr
        .write_all(note.as_bytes())
        .and_then(|()| stderr.flush())
        .map_err(|e| e.to_string())?;

    if json {
        writeln!(
            out,
            "{}",
            serde_json::json!({
                "batch_fallback": {
                    "status": reason.status(),
                    "message": reason.message(),
                    "files": files,
                }
            })
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

enum BatchSubprocessOutcome {
    Rows(Vec<TestRow>),
    Fallback(BatchFallbackReason),
}

fn run_test_batch_subprocess(
    self_path: &Path,
    cwd: &Path,
    batch_jobs: &[TestBatchManifestFile],
    timeout_secs: u64,
    compiled_context_handoff: &CompiledContextHandoff,
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
        .current_dir(cwd);
    compiled_context_handoff.apply_to(&mut cmd);

    let mut progress = progress_file
        .map(|path| {
            fs::OpenOptions::new()
                .append(true)
                .open(path)
                .map_err(|e| format!("open suite progress file `{}`: {e}", path.display()))
        })
        .transpose()?;
    let worker_timeout = batch_worker_timeout(batch_jobs, timeout_secs);
    let output = match run_batch_worker_command_with_timeout(cmd, worker_timeout, |line| {
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
    }) {
        Ok(output) => output,
        Err(e) => {
            return Ok(BatchSubprocessOutcome::Fallback(
                BatchFallbackReason::WorkerUnavailable(e.to_string()),
            ));
        }
    };
    if output.timed_out {
        return Ok(BatchSubprocessOutcome::Fallback(
            BatchFallbackReason::Timeout(worker_timeout.as_secs()),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.output.stdout);
    let mut rows = Vec::new();
    for (number, line) in stdout.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let number = number + 1;
        let malformed = |detail: String| {
            Ok(BatchSubprocessOutcome::Fallback(
                BatchFallbackReason::MalformedOutput(format!("stdout line {number} {detail}")),
            ))
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
            return malformed("is not JSON".to_string());
        };
        let Some(file) = value.get("file").and_then(|v| v.as_str()) else {
            return malformed("has no `file` field".to_string());
        };
        let Some(test) = value.get("test").and_then(|v| v.as_str()) else {
            return malformed("has no `test` field".to_string());
        };
        let Some(status_s) = value.get("status").and_then(|v| v.as_str()) else {
            return malformed("has no `status` field".to_string());
        };
        let status = match status_s {
            "pass" => TestStatus::Pass,
            "fail" => TestStatus::Fail,
            other => {
                return malformed(format!("has status `{other}`, not `pass` or `fail`"));
            }
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

    // Exit 1 is the worker's "some test failed" code, so it is only a fallback
    // when the worker produced no rows to attribute that failure to.
    let failure = match output.output.status.code() {
        None => Some("batch worker was terminated by a signal".to_string()),
        Some(0) => None,
        Some(1) if rows.is_empty() => {
            Some("batch worker exited with status 1 and emitted no test rows".to_string())
        }
        Some(1) => None,
        Some(code) => Some(format!("batch worker exited with status {code}")),
    };
    if let Some(detail) = failure {
        return Ok(BatchSubprocessOutcome::Fallback(
            BatchFallbackReason::WorkerFailed(detail),
        ));
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
    let compiled_context_handoff = worker_options.compiled_context_handoff.cloned();
    let expect_file_diagnostic = worker_options.expect_file_diagnostic;
    let mut handles = Vec::new();

    for _ in 0..worker_count {
        let jobs = Arc::clone(&jobs);
        let next_index = Arc::clone(&next_index);
        let result_tx = result_tx.clone();
        let self_path = self_path.clone();
        let cwd = cwd.clone();
        let filter = filter.clone();
        let compiled_context_handoff = compiled_context_handoff.clone();
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
                            compiled_context_handoff: compiled_context_handoff.as_ref(),
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
    compiled_context_handoff: &CompiledContextHandoff,
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
                    compiled_context_handoff: Some(compiled_context_handoff),
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
    let compiled_context_handoff = compiled_context_handoff.clone();
    let mut handles = Vec::new();

    for _ in 0..worker_count {
        let jobs = Arc::clone(&jobs);
        let next_index = Arc::clone(&next_index);
        let result_tx = result_tx.clone();
        let self_path = self_path.clone();
        let cwd = cwd.clone();
        let filter = filter.clone();
        let compiled_context_handoff = compiled_context_handoff.clone();
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
                            compiled_context_handoff: Some(&compiled_context_handoff),
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

enum EmptyTestSelection {
    NoSourceFiles {
        excluded: ExcludedTestFileCount,
    },
    NoRunnableTests {
        source_files: usize,
    },
    FilterNoMatch {
        filter: String,
        runnable_tests: usize,
    },
}

enum TestSelectionPreflight {
    NeedsExecution,
    Empty(EmptyTestSelection),
}

struct ExcludedTestFileCount {
    count: usize,
    lower_bound: bool,
}

/// Count source files hidden by the ordinary test walk's exclusions.
///
/// The selected walk has already completed, so a failure here can only occur
/// below an excluded entry. As with `chelis check`'s [04-FIT-24] diagnostic,
/// that makes the result a lower bound rather than a reason to add a second
/// walk error.
fn count_excluded_test_files(target: &Path) -> ExcludedTestFileCount {
    let unfiltered = walk_sources(
        target,
        &WalkRules {
            exclusions: false,
            ..TEST_WALK
        },
    );
    ExcludedTestFileCount {
        count: unfiltered
            .iter()
            .filter(|item| matches!(item, WalkItem::Source(_)))
            .count(),
        lower_bound: unfiltered
            .iter()
            .any(|item| matches!(item, WalkItem::Failure(_))),
    }
}

fn describe_empty_test_selection(target: &Path, reason: EmptyTestSelection) -> String {
    match reason {
        EmptyTestSelection::NoSourceFiles { excluded } => {
            let bound = if excluded.lower_bound {
                "at least "
            } else {
                ""
            };
            format!(
                "no .ch test files under {}: {bound}{} excluded under dot-prefixed entries or \
                 `target` directories",
                escaped_path(target),
                excluded.count
            )
        }
        EmptyTestSelection::NoRunnableTests { source_files } => {
            let noun = if source_files == 1 { "file" } else { "files" };
            format!(
                "found {source_files} .ch {noun} under {} but no runnable `test_*` functions",
                escaped_path(target)
            )
        }
        EmptyTestSelection::FilterNoMatch {
            filter,
            runnable_tests,
        } => {
            let noun = if runnable_tests == 1 {
                "test was"
            } else {
                "tests were"
            };
            format!(
                "no tests matched filter {filter:?} under {}: {runnable_tests} runnable {noun} \
                 available before filtering",
                escaped_path(target)
            )
        }
    }
}

fn emit_empty_test_selection(json: bool, message: String) -> Result<i32, String> {
    if json {
        let stdout = io::stdout();
        let mut out = stdout.lock();
        writeln!(
            out,
            "{}",
            serde_json::json!({
                "errors": [{
                    "kind": DiagnosticKind::EmptyTestSelection.as_str(),
                    "message": message,
                    "severity": 1.0,
                }]
            })
        )
        .map_err(|error| error.to_string())?;
    } else {
        let stderr = io::stderr();
        let mut out = stderr.lock();
        writeln!(out, "error: {message}").map_err(|error| error.to_string())?;
    }
    Ok(2)
}

/// Cheap conservative scan for [04-TEST-1]'s ordinary runnable selection.
///
/// A source failure or duplicate test definition returns `NeedsExecution`,
/// allowing the normal worker path to report the real file error. Only a
/// complete, clean scan may return `Empty`.
fn preflight_test_selection(
    test_files: &[PathBuf],
    cwd: &Path,
    filter: Option<&str>,
) -> TestSelectionPreflight {
    let mut runnable_tests = 0usize;
    let mut selected_tests = 0usize;
    for file in test_files {
        let rel_display = file
            .strip_prefix(cwd)
            .unwrap_or(file.as_path())
            .display()
            .to_string();
        let source = match fs::read_to_string(file) {
            Ok(s) => s,
            Err(_) => return TestSelectionPreflight::NeedsExecution,
        };
        let parsed = match chelis_surf::parser::parse_str(&source) {
            Ok(p) => p,
            Err(_) => return TestSelectionPreflight::NeedsExecution,
        };
        let flat = flatten_module_decls(&parsed);
        let tests = match enumerate_test_fns(&flat, None, &rel_display) {
            EnumerationOutcome::Tests(tests) => tests,
            EnumerationOutcome::Error(_) => return TestSelectionPreflight::NeedsExecution,
        };
        runnable_tests += tests.len();
        selected_tests += tests
            .iter()
            .filter(|test| {
                filter.is_none_or(|needle| format!("{rel_display}::{}", test.name).contains(needle))
            })
            .count();
        if selected_tests > 0 {
            return TestSelectionPreflight::NeedsExecution;
        }
    }

    if runnable_tests == 0 {
        TestSelectionPreflight::Empty(EmptyTestSelection::NoRunnableTests {
            source_files: test_files.len(),
        })
    } else {
        match filter {
            Some(filter) => TestSelectionPreflight::Empty(EmptyTestSelection::FilterNoMatch {
                filter: filter.to_string(),
                runnable_tests,
            }),
            None => TestSelectionPreflight::NeedsExecution,
        }
    }
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
    // Skip dot-prefixed files AND directories (editor temp files, build dirs
    // like `.git` or `target/.rustc_info.json`, hidden fixtures), and
    // `target/` directories, which accumulate build artifacts and have bitten
    // us in red-team testing.
    //
    // A walk failure still aborts `chelis test`, as it always has: its
    // directory-level reporting is chelis#1825's, not the check envelope's.
    let mut files = Vec::new();
    for item in walk_sources(target, &TEST_WALK) {
        match item {
            WalkItem::Source(file) => files.push(file),
            WalkItem::Failure(message) => {
                return Err(format!("failed to walk {}: {message}", target.display()));
            }
        }
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
    compiled_context_handoff: Option<&'a CompiledContextHandoff>,
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
    match worker_options.compiled_context_handoff {
        // Phase H: hand the bincode-encoded `CompiledContext` to the
        // worker via env var so the worker can deserialize the library
        // snapshot instead of re-running `prepare_reef_graph` per file.
        // Absent on packages whose deps are LocalRegistry-resolved
        // (`compile_reef_context` cannot hash those yet); the worker's
        // own fallback then runs the legacy reef-graph path.
        Some(handoff) => handoff.apply_to(&mut cmd),
        // Belt-and-suspenders: never let an inherited env var from the
        // outer environment shadow our "no context available" decision.
        // If the parent could not build a context, the worker MUST take
        // the legacy path on its own.
        None => CompiledContextHandoff::clear_from(&mut cmd),
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
    //
    // chelis#2211: the parent also passes the payload's digest, in a second
    // variable, so the worker can establish that the bytes on disk are still
    // the ones the parent wrote. A path with no digest beside it is refused
    // rather than decoded the slow way. The slow route does not prove
    // authenticity -- it proves the payload's parts agree with each other --
    // so falling back to it would accept an unauthenticated handoff while
    // looking careful.
    let compiled_context_env = env::var(COMPILED_CONTEXT_PATH_ENV).ok();
    match compiled_context_env.as_deref() {
        Some(path) if !path.is_empty() => {
            let bytes = fs::read(path)
                .map_err(|e| format!("read {COMPILED_CONTEXT_PATH_ENV} tempfile `{path}`: {e}"))?;
            let digest_hex = env::var(COMPILED_CONTEXT_DIGEST_ENV)
                .ok()
                .filter(|value| !value.is_empty())
                .ok_or_else(|| {
                    format!(
                        "{COMPILED_CONTEXT_PATH_ENV} is set but {COMPILED_CONTEXT_DIGEST_ENV} is \
                         not; refusing to decode a compiled context this worker cannot \
                         authenticate"
                    )
                })?;
            let digest =
                chelis_compiler_api::HandoffDigest::from_hex(&digest_hex).map_err(|e| {
                    format!("{COMPILED_CONTEXT_DIGEST_ENV} is not a usable digest: {e}")
                })?;
            let ctx = chelis_compiler_api::CompiledContext::decode_authenticated(&bytes, &digest)
                .map_err(|e| format!("decode {COMPILED_CONTEXT_PATH_ENV}: {e}"))?;
            let cwd = env::current_dir().map_err(|e| format!("failed to read cwd: {e}"))?;
            let ctx_root = ctx
                .reef_state()
                .package_root
                .canonicalize()
                .unwrap_or_else(|_| ctx.reef_state().package_root.clone());
            let cwd_canon = cwd.canonicalize().unwrap_or_else(|_| cwd.clone());
            if ctx_root != cwd_canon {
                return Err(format!(
                    "{COMPILED_CONTEXT_PATH_ENV} package_root `{}` does not match worker cwd \
                     `{}`; refusing to run tests with a mismatched library context",
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
    let mut entries = Vec::new();
    let mut selected = Vec::<(usize, String, String)>::new();
    // Same admission rule the parent classifier applied, so this guard can only
    // reject a manifest the parent should never have built. A stricter guard
    // here would reject legitimate batches and turn them into silent per-file
    // fallbacks, which is how the two used to disagree.
    let mut batch_scope = BatchScope::default();

    for file in files {
        let source = fs::read_to_string(&file.file)
            .map_err(|e| format!("read {}: {e}", file.file.display()))?;
        let parsed = chelis_surf::parser::parse_str(&source)
            .map_err(|e| format!("parse {}: {e}", file.file.display()))?;
        let flat = flatten_module_decls(&parsed);
        if let Some(collision) = batch_scope.admit(&file.rel_display, &test_file_scope_names(&flat))
        {
            return Err(format!("test batch scope collision: {collision}"));
        }

        let tests = match enumerate_test_fns(&flat, None, &file.rel_display) {
            EnumerationOutcome::Tests(tests) => tests,
            EnumerationOutcome::Error(msg) => return Err(msg),
        };
        let mut selected_roots = Vec::new();
        for test_name in &file.tests {
            let Some(test) = tests.iter().find(|candidate| candidate.name == *test_name) else {
                return Err(format!(
                    "selected test `{test_name}` was not found in {}",
                    file.rel_display
                ));
            };
            selected.push((file.index, file.rel_display.clone(), test.name.clone()));
            selected_roots.push(chelis_reef::SelectedEntryRoot {
                name: test.name.clone(),
                span: test.span,
            });
        }
        entries.push(chelis_reef::IsolatedEntryModule {
            manifest_index: file.index,
            declarations: flat,
            selected_roots,
        });
    }

    let rewritten = chelis_reef::rewrite_isolated_entry_modules_with_reef_graph(
        exec_context.reef_graph(),
        &entries,
    )?;
    let prepared_eval = prepare_rewritten_batch_in_exec_context(exec_context, &rewritten)?;

    for (manifest_index, rel_display, test_name) in selected {
        let root = rewritten
            .exact_root(manifest_index, &test_name)
            .ok_or_else(|| {
                format!("isolated test entry {manifest_index} lost selected root `{test_name}`")
            })?
            .to_string();
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

/// Consume one batch product that Reef has already rewritten as independent
/// modules. Both context variants use these exact declarations and roots;
/// neither routes the combined batch through the flat eval-entry resolver.
fn prepare_rewritten_batch_in_exec_context(
    exec_context: &TestExecutionContext,
    batch: &chelis_reef::RewrittenEntryBatch,
) -> Result<PreparedTestEval, String> {
    let _linked = chelis_types::install_linked_program_guard();
    match exec_context {
        TestExecutionContext::Context(ctx) => {
            chelis_compiler_api::compiler::prepare_rewritten_entry_batch_in_context(ctx, batch)
                .map(PreparedTestEval::InContext)
                .map_err(|err| {
                    err.errors
                        .iter()
                        .map(|diagnostic| diagnostic.message.clone())
                        .collect::<Vec<_>>()
                        .join("; ")
                })
        }
        TestExecutionContext::ReefGraph(graph) => {
            let prepared =
                chelis_reef::compile_rewritten_entry_batch_with_reef_graph(graph, batch)?;
            let source_text = chelis_surf::format::format_program(&prepared.decls);
            #[allow(deprecated)]
            let prepared_eval = chelis_compiler_api::compiler::prepare_eval(EvalRequest {
                source_kind: SourceKind::Surf,
                source: source_text,
                bindings: BTreeMap::new(),
            });
            prepared_eval.map(PreparedTestEval::Legacy).map_err(|err| {
                err.errors
                    .iter()
                    .map(|diagnostic| diagnostic.message.clone())
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
    let mut seen: UnordMap<String, bool> = UnordMap::new();
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
        // types is what keeps typed-value bindings like `def test_x : i64 = 42`
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
        TypeExpr::Named(name, _) if name == "unit" => true,
        TypeExpr::Tuple(items, _) => items.is_empty(),
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
    dag: chelis_ir::dag::Dag,
    c_name: &c_source_name::CSourceName,
    output: Option<&std::path::Path>,
    symbolic_dims_hint: &[String],
    root_manifest: &chelis_types::manifest::RootManifest,
) -> Result<(), Box<dyn std::error::Error>> {
    let func_name = c_name.symbol();
    let symbolic_dims = fallback_symbolic_dims(&dag, &[], symbolic_dims_hint);
    let options = chelis_backend_c::CodegenOptions {
        use_blas: true,
        ..chelis_backend_c::CodegenOptions::default()
    };
    let selected = chelis_backend_c::prepare_dag_for_codegen(dag, options);
    let verified = verified_dag_codegen_program(selected)?;
    let mut result = chelis_backend_c::codegen_with_options(verified, func_name, options)?;
    let symbolic_dims = if result.symbolic_dims.is_empty() {
        symbolic_dims
    } else {
        result.symbolic_dims.clone()
    };
    let root_names =
        tensor_manifest_root_names(&result.output_labels, root_manifest, BuildTarget::C, "C")?;
    let requires_main = root_manifest.requires_main();
    if requires_main {
        result
            .c_source
            .push_str(&tensor_manifest_observation_driver(func_name, &root_names));
        result.reseal_artifact(func_name)?;
    }
    cmd_build_c_result(result, c_name, output, &symbolic_dims, requires_main)
}

fn cmd_build_c_result(
    result: chelis_backend_c::CodegenResult,
    c_name: &c_source_name::CSourceName,
    output: Option<&std::path::Path>,
    symbolic_dims: &[String],
    requires_main: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let c_path = if out_dir.extension().and_then(|e| e.to_str()) == Some("c") {
        out_dir.clone()
    } else {
        out_dir.join(c_name.filename())
    };
    let h_path = c_path.with_extension("h");
    if let Some(parent) = c_path.parent() {
        fs::create_dir_all(parent)?;
    }

    fs::write(&c_path, &result.c_source)?;
    fs::write(&h_path, &result.h_header)?;

    let runtime_dir = c_path.parent().unwrap_or(std::path::Path::new("."));
    let staged = stage_runtime_artifacts(runtime_dir, ExtraRuntimeArtifacts::default())?;

    println!("Wrote {} and {}", c_path.display(), h_path.display());
    println!(
        "Wrote {}, {}, and {}",
        runtime_dir.join("chelis_runtime.h").display(),
        runtime_dir.join("chelis_blas.h").display(),
        staged.archive.display()
    );
    print_staged_runtime(&staged);
    if !symbolic_dims.is_empty() {
        println!("Symbolic dims: {}", symbolic_dims.join(", "));
    }
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.requirements);
    if requires_main {
        println!(
            "Compile: {} -O2 {} {} {} {} -o {}",
            toolchain.compiler,
            toolchain.compile_flags.join(" "),
            c_path.display(),
            staged.archive.display(),
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
    result: chelis_backend_hip::HipHostProgramCodegenResult,
    func_name: &str,
    output: Option<&std::path::Path>,
    requires_main: bool,
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

    fs::write(&c_path, &result.host.c_source)?;
    fs::write(&h_path, &result.host.h_header)?;

    let runtime_dir = c_path.parent().unwrap_or(std::path::Path::new("."));
    let mut helper_paths = Vec::with_capacity(result.device_helpers.len());
    for helper in &result.device_helpers {
        let helper_path = runtime_dir.join(format!("{}_hip.cpp", helper.name));
        fs::write(&helper_path, &helper.result.c_source)?;
        helper_paths.push(helper_path);
    }
    let staged = stage_runtime_artifacts(
        runtime_dir,
        ExtraRuntimeArtifacts {
            hip: true,
            metal: false,
        },
    )?;

    println!("Wrote {} and {}", c_path.display(), h_path.display());
    for helper_path in &helper_paths {
        println!("Wrote device helper {}", helper_path.display());
    }
    println!(
        "Wrote runtime: {}, {}, {}, {}",
        runtime_dir.join("chelis_runtime.h").display(),
        runtime_dir.join("chelis_blas.h").display(),
        staged.archive.display(),
        runtime_dir.join("chelis_hip_runtime.h").display()
    );
    print_staged_runtime(&staged);

    let cpu_toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.host.requirements);
    let mut compile_flags = cpu_toolchain.compile_flags;
    compile_flags.retain(|flag| flag != "-fopenmp");
    let mut link_flags = cpu_toolchain.link_flags;
    link_flags.retain(|flag| flag != "-fopenmp");
    for helper in &result.device_helpers {
        for flag in &helper.result.compile_flags {
            if !compile_flags.contains(flag) {
                compile_flags.push(flag.clone());
            }
        }
        for flag in &helper.result.link_flags {
            if !link_flags.contains(flag) {
                link_flags.push(flag.clone());
            }
        }
    }
    let mut support_sources = helper_paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    support_sources.push(
        runtime_dir
            .join("chelis_device_owner.cpp")
            .display()
            .to_string(),
    );
    let support_sources = support_sources.join(" ");
    if requires_main {
        println!(
            "Compile: hipcc {} {} {} {} -lpthread -ldl {} -o {}",
            compile_flags.join(" "),
            c_path.display(),
            support_sources,
            staged.archive.display(),
            link_flags.join(" "),
            c_path.with_extension("").display()
        );
    } else {
        println!(
            "Compile objects: hipcc {} -c {} {}",
            compile_flags.join(" "),
            c_path.display(),
            support_sources
        );
    }
    Ok(())
}

fn cmd_build_metal_host(
    result: chelis_backend_metal::MetalHostProgramCodegenResult,
    func_name: &str,
    output: Option<&std::path::Path>,
    requires_main: bool,
) -> Result<(), Box<dyn std::error::Error>> {
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
    fs::write(&mm_path, &result.host.c_source)?;
    fs::write(&h_path, &result.host.h_header)?;

    let runtime_dir = mm_path.parent().unwrap_or(std::path::Path::new("."));
    let mut helper_paths = Vec::with_capacity(result.device_helpers.len());
    for helper in &result.device_helpers {
        let helper_path = runtime_dir.join(format!("{}_metal.mm", helper.name));
        fs::write(&helper_path, &helper.result.mm_source)?;
        helper_paths.push(helper_path);
    }
    let staged = stage_runtime_artifacts(
        runtime_dir,
        ExtraRuntimeArtifacts {
            hip: false,
            metal: true,
        },
    )?;

    println!("Wrote {} and {}", mm_path.display(), h_path.display());
    for helper_path in &helper_paths {
        println!("Wrote device helper {}", helper_path.display());
    }
    println!(
        "Wrote runtime: {}, {}, {}",
        runtime_dir.join("chelis_runtime.h").display(),
        staged.archive.display(),
        runtime_dir.join("chelis_metal_runtime.h").display()
    );
    print_staged_runtime(&staged);

    let cpu_toolchain = chelis_backend_c::toolchain::runtime_toolchain(result.host.requirements);
    let mut compile_flags = cpu_toolchain.compile_flags;
    compile_flags.retain(|flag| flag != "-fopenmp");
    let mut link_flags = cpu_toolchain.link_flags;
    link_flags.retain(|flag| flag != "-fopenmp");
    for helper in &result.device_helpers {
        for flag in &helper.result.compile_flags {
            if !compile_flags.contains(flag) {
                compile_flags.push(flag.clone());
            }
        }
        // `-framework NAME` pairs must keep their order; append each
        // helper's pair list once rather than deduplicating single tokens.
        if helper
            .result
            .link_flags
            .iter()
            .any(|flag| !link_flags.contains(flag))
        {
            link_flags.extend(helper.result.link_flags.iter().cloned());
        }
    }
    let helper_sources = helper_paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    if requires_main {
        println!(
            "Compile: clang++ {} -O2 {} {} {} {} -o {}",
            compile_flags.join(" "),
            mm_path.display(),
            helper_sources,
            staged.archive.display(),
            link_flags.join(" "),
            mm_path.with_extension("").display()
        );
    } else {
        println!(
            "Compile objects: clang++ {} -O2 -c {} {}",
            compile_flags.join(" "),
            mm_path.display(),
            helper_sources
        );
    }
    Ok(())
}

fn tensor_manifest_observation_driver(func_name: &str, root_names: &[String]) -> String {
    let mut source = String::from(
        r#"

static void chelis_manifest_print_tensor_elem(const chelis_tensor *tensor, int64_t index) {
    chelis_read_view view = chelis_tensor_read_view(tensor);
    uint64_t bits = 0;
    int64_t width = chelis_dtype_size(view.dtype);
    memcpy(&bits, (const uint8_t *)view.data + index * width, (size_t)width);
    chelis_scalar scalar = chelis_scalar_from_bits(view.dtype, bits);
    chelis_string text = chelis_string_from_scalar(scalar);
    chelis_print_string(text);
    chelis_string_release(text);
}

static void chelis_manifest_print_tensor(const chelis_tensor *tensor) {
    if (tensor == NULL) {
        fputs("unsupported: [05-UNS-1] Tensor root returned no tensor\n", stderr);
        exit(1);
    }
    int32_t rank = chelis_tensor_rank(tensor);
    if (rank == 0) {
        chelis_manifest_print_tensor_elem(tensor, 0);
        return;
    }
    fputs("tensor(shape=[", stdout);
    for (int32_t dim = 0; dim < rank; ++dim) {
        if (dim > 0) fputs(", ", stdout);
        printf("%lld", (long long)chelis_tensor_shape(tensor, dim));
    }
    fputs("], data=[", stdout);
    int64_t size = chelis_tensor_numel(tensor);
    int64_t limit = size < 32 ? size : 32;
    for (int64_t index = 0; index < limit; ++index) {
        if (index > 0) fputs(", ", stdout);
        chelis_manifest_print_tensor_elem(tensor, index);
    }
    if (size > limit) fputs(", ...", stdout);
    fputs("])", stdout);
}
"#,
    );
    source.push_str(&format!(
        "\nint main(void) {{\n    chelis_tensor *outputs[{}] = {{0}};\n    {func_name}(NULL, 0, outputs, {});\n",
        root_names.len(),
        root_names.len()
    ));
    for (index, name) in root_names.iter().enumerate() {
        let label = chelis_ir::span_sanitize::sanitize_for_format_string(name);
        source.push_str(&format!(
            "    printf(\"{label} = \");\n    chelis_manifest_print_tensor(outputs[{index}]);\n    printf(\"\\n\");\n    chelis_tensor_release(outputs[{index}]);\n"
        ));
    }
    source.push_str("    return 0;\n}\n");
    source
}

#[cfg(test)]
mod exact_manifest_observation_driver_tests {
    use super::tensor_manifest_observation_driver;

    #[test]
    fn manifest_tensor_elements_render_through_exact_tagged_scalars() {
        let source = tensor_manifest_observation_driver("entry", &["root".to_string()]);
        for required in [
            "chelis_scalar_from_bits",
            "chelis_string_from_scalar",
            "chelis_print_string",
            "chelis_string_release",
            "chelis_dtype_size",
            "view.dtype",
            "chelis_tensor_rank(tensor)",
        ] {
            assert!(
                source.contains(required),
                "manifest observation driver is missing `{required}`:\n{source}"
            );
        }
        for retired in [
            "chelis_format_shortest",
            "CHELIS_F64",
            "CHELIS_BOOL",
            "chelis_string_data",
            "tensor->ndim",
        ] {
            assert!(
                !source.contains(retired),
                "manifest observation driver restored retired ABI spelling `{retired}`:\n{source}"
            );
        }
    }

    #[test]
    fn manifest_bool_elements_read_one_byte_storage() {
        let source = tensor_manifest_observation_driver("entry", &["root".to_string()]);
        assert!(source.contains("(const uint8_t *)view.data"), "{source}");
        assert!(
            !source.contains("(const float *)view.data)[index] != 0.0f"),
            "manifest Bool observation retained four-byte float storage:\n{source}"
        );
    }
}

fn tensor_manifest_root_names(
    output_labels: &[String],
    root_manifest: &chelis_types::manifest::RootManifest,
    target: BuildTarget,
    lowering_name: &str,
) -> Result<Vec<String>, chelis_types::unsupported::Unsupported> {
    if !root_manifest.requires_main() {
        return Ok(Vec::new());
    }
    let root_names = root_manifest
        .entries
        .iter()
        .map(|entry| entry.name.clone())
        .collect::<Vec<_>>();
    let labels_match = output_labels.len() == root_names.len()
        && output_labels
            .iter()
            .zip(&root_names)
            .enumerate()
            .all(|(index, (lowered, owed))| lowered == owed || lowered == &format!("root{index}"));
    if labels_match {
        return Ok(root_names);
    }

    let owed_index = root_names
        .iter()
        .enumerate()
        .find(|(index, name)| {
            output_labels
                .get(*index)
                .is_none_or(|lowered| lowered != *name && lowered != &format!("root{index}"))
        })
        .map(|(index, _)| index)
        .unwrap_or(0);
    Err(build_unavailable_root_error(
        &root_manifest.entries[owed_index],
        target,
        format!(
            "{lowering_name} lowering produced outputs {output_labels:?} instead of the manifest {root_names:?}"
        ),
    ))
}

fn cmd_build_hip(
    dag: chelis_ir::dag::Dag,
    func_name: &str,
    _file: &std::path::Path,
    output: Option<&std::path::Path>,
    symbolic_dims_hint: &[String],
    root_manifest: &chelis_types::manifest::RootManifest,
) -> Result<(), Box<dyn std::error::Error>> {
    let symbolic_dims = fallback_symbolic_dims(&dag, &[], symbolic_dims_hint);
    let selected = chelis_backend_hip::prepare_dag_for_codegen(dag);
    let verified = verified_dag_codegen_program(selected)?;
    let mut result = chelis_backend_hip::codegen_hip(verified, func_name)?;
    let requires_main = root_manifest.requires_main();
    if requires_main {
        let root_names = tensor_manifest_root_names(
            &result.output_labels,
            root_manifest,
            BuildTarget::Hip,
            "HIP",
        )?;
        result
            .c_source
            .push_str(&tensor_manifest_observation_driver(func_name, &root_names));
    }

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
    let staged = stage_runtime_artifacts(
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
        staged.archive.display(),
        runtime_dir.join("chelis_hip_runtime.h").display()
    );
    print_staged_runtime(&staged);
    let symbolic_dims = if result.symbolic_dims.is_empty() {
        symbolic_dims
    } else {
        result.symbolic_dims.clone()
    };
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
    if requires_main {
        println!(
            "Compile: hipcc {} {} {} {} -lpthread -ldl -o {}",
            flags.join(" "),
            c_path.display(),
            runtime_dir.join("chelis_device_owner.cpp").display(),
            staged.archive.display(),
            c_path.with_extension("").display()
        );
    } else {
        println!(
            "Compile object: hipcc {} -c {} {}",
            flags.join(" "),
            c_path.display(),
            runtime_dir.join("chelis_device_owner.cpp").display()
        );
    }
    Ok(())
}

fn cmd_build_metal(
    dag: chelis_ir::dag::Dag,
    func_name: &str,
    _file: &std::path::Path,
    output: Option<&std::path::Path>,
    symbolic_dims_hint: &[String],
) -> Result<(), Box<dyn std::error::Error>> {
    let symbolic_dims = fallback_symbolic_dims(&dag, &[], symbolic_dims_hint);
    let verified = verified_dag_codegen_program(dag)?;
    let plan = chelis_backend_metal::plan_metal(verified);
    let result = chelis_backend_metal::codegen_metal(plan, func_name)?;

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
    let staged = stage_runtime_artifacts(
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
        staged.archive.display(),
        runtime_dir.join("chelis_metal_runtime.h").display()
    );
    print_staged_runtime(&staged);
    let symbolic_dims = if result.symbolic_dims.is_empty() {
        symbolic_dims
    } else {
        result.symbolic_dims.clone()
    };
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
        "Compile: clang++ {} -O2 {} {} -o {}",
        flags.join(" "),
        mm_path.display(),
        staged.archive.display(),
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
                Err(error) => {
                    emit_failed_eval_transcript(&error.transcript, false)?;
                    let numeric_trap = error
                        .errors
                        .iter()
                        .any(|diagnostic| diagnostic.kind() == DiagnosticKind::NumericTrap);
                    let rendered = join_eval_error(error);
                    if numeric_trap {
                        eprintln!("{rendered}");
                    } else {
                        eprintln!("error: {rendered}");
                    }
                }
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
fn format_eval_diagnostic(diag: &chelis_compiler_api::schema::Diagnostic) -> String {
    render_eval_diagnostic(
        &diag.message,
        &diag.suggestions,
        diag.kind() == DiagnosticKind::NumericTrap,
    )
}

/// The rendering itself, over the two fields it reads. Separated from the
/// `Diagnostic` adapter above because chelis#959 seals diagnostic production
/// inside compiler-api: no crate outside it can build a `Diagnostic`, so the
/// rendering contract is exercised through this function instead.
fn render_eval_diagnostic(message: &str, suggestions: &[String], numeric_trap: bool) -> String {
    let mut rendered = message.to_string();
    for hint in suggestions {
        rendered.push_str(if numeric_trap {
            "\n  hint: "
        } else {
            "; hint: "
        });
        rendered.push_str(hint);
    }
    rendered
}

fn try_eval_result(
    source_kind: SourceKind,
    source: &str,
    selected_roots: Option<&[String]>,
) -> Result<chelis_compiler_api::schema::EvalResult, CompilerError> {
    try_eval_result_for_target(
        source_kind,
        source,
        selected_roots,
        chelis_types::types::Target::Eval,
    )
}

fn try_eval_result_for_target(
    source_kind: SourceKind,
    source: &str,
    selected_roots: Option<&[String]>,
    target: chelis_types::types::Target,
) -> Result<chelis_compiler_api::schema::EvalResult, CompilerError> {
    let request = EvalRequest {
        source_kind,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    };
    if let Some(roots) = selected_roots {
        chelis_compiler_api::compiler::eval_selected_for_target(request, roots, target)
    } else {
        chelis_compiler_api::compiler::eval_for_target(request, target)
    }
}

/// Flatten a `CompilerError` into the single string this CLI's error channel
/// carries.
///
/// Cancellation (chelis#914) is decided on the STRUCTURED
/// `CompilerError::is_cancellation`, never by inspecting message text, and is
/// then transported as the bare sentinel. `cmd_eval` recognizes it at the
/// `Box<dyn Error>` boundary, where the typed error is no longer available.
fn join_eval_error(err: chelis_compiler_api::compiler::CompilerError) -> String {
    if err.is_cancellation() {
        return chelis_compiler_api::EVAL_CANCELLED_MSG.to_string();
    }
    let separator = if err
        .errors
        .iter()
        .any(|diagnostic| diagnostic.kind() == DiagnosticKind::NumericTrap)
    {
        "\n"
    } else {
        "; "
    };
    err.errors
        .iter()
        .map(format_eval_diagnostic)
        .collect::<Vec<_>>()
        .join(separator)
}

#[cfg(test)]
mod eval_diagnostic_rendering_tests {
    use super::render_eval_diagnostic;

    fn hints(suggestions: &[&str]) -> Vec<String> {
        suggestions.iter().map(|hint| (*hint).to_string()).collect()
    }

    #[test]
    fn structured_suggestions_render_without_matching_diagnostic_text() {
        assert_eq!(
            render_eval_diagnostic(
                "renamed trap wording that contains no cast substring",
                &hints(&["first recovery action", "second recovery action"]),
                false,
            ),
            "renamed trap wording that contains no cast substring; hint: first recovery action; \
             hint: second recovery action"
        );
    }

    #[test]
    fn diagnostic_without_suggestions_keeps_its_exact_message() {
        assert_eq!(
            render_eval_diagnostic("plain failure", &hints(&[]), false),
            "plain failure"
        );
    }

    #[test]
    fn numeric_trap_suggestions_are_separate_lines() {
        assert_eq!(
            render_eval_diagnostic(
                "numeric trap: domain in cast at i64",
                &hints(&["use cast_trunc"]),
                true,
            ),
            "numeric trap: domain in cast at i64\n  hint: use cast_trunc"
        );
    }
}

fn try_eval(
    source_kind: SourceKind,
    source: &str,
    selected_roots: Option<&[String]>,
) -> Result<String, CompilerError> {
    let result = try_eval_result(source_kind, source, selected_roots)?;
    Ok(format_eval_result(&result))
}

fn try_eval_for_target(
    source_kind: SourceKind,
    source: &str,
    selected_roots: Option<&[String]>,
    target: chelis_types::types::Target,
) -> Result<String, CompilerError> {
    let result = try_eval_result_for_target(source_kind, source, selected_roots, target)?;
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
    // Issue #912 [05-OBS-6]: always label, in both lanes. The bare-when-single
    // form is removed — it cost cross-lane byte identity and line-count parity.
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

fn checked_compilation_with_effects(
    deep_exprs: &[chelis_deep::ast::Expr],
) -> Result<chelis_compiler_api::pipeline::CheckedCompilation, String> {
    let prepared = chelis_compiler_api::pipeline::prepare_deep(deep_exprs.to_vec(), None);
    let analysis = match chelis_compiler_api::pipeline::analyze_prepared(prepared) {
        chelis_compiler_api::pipeline::PreparedTypeAnalysisOutcome::Accepted(analysis) => *analysis,
        chelis_compiler_api::pipeline::PreparedTypeAnalysisOutcome::Rejected { fitness } => {
            // chelis#1853 [04-FIT-26]: the shared rejection rendering, one
            // projected line per diagnostic.
            return Err(
                chelis_compiler_api::pipeline::PipelineRejection::Type { fitness }.to_string(),
            );
        }
    };
    chelis_compiler_api::pipeline::complete_checks(
        analysis,
        chelis_compiler_api::pipeline::SemanticContext::Isolated,
    )
    .map_err(|rejection| rejection.to_string())
}

fn checked_program_with_effects(
    deep_exprs: &[chelis_deep::ast::Expr],
) -> Result<chelis_types::CheckedProgram, String> {
    let checked = checked_compilation_with_effects(deep_exprs)?;
    let (_, _, program, _) = checked.into_parts();
    Ok(program)
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
    chelis_compiler_api::pipeline::prepare_surf_decls(decls, None)
        .map(chelis_compiler_api::pipeline::PreparedProgram::into_expanded_deep)
        .map_err(|error| error.to_string())
}

fn lower_checked_for_cli(
    checked: chelis_compiler_api::pipeline::CheckedCompilation,
    host_program: Option<&chelis_ir::host::ConcreteHostProgram>,
) -> Result<chelis_ir::Dag, Box<dyn std::error::Error>> {
    lower_checked_compilation_for_cli(checked, host_program)
        .map(chelis_compiler_api::pipeline::LoweredCompilation::into_dag)
}

fn lower_checked_compilation_for_cli(
    checked: chelis_compiler_api::pipeline::CheckedCompilation,
    host_program: Option<&chelis_ir::host::ConcreteHostProgram>,
) -> Result<chelis_compiler_api::pipeline::LoweredCompilation, Box<dyn std::error::Error>> {
    let mode = if host_program
        .map(chelis_ir::host::host_program_requires_host_backend)
        .unwrap_or(false)
    {
        chelis_compiler_api::pipeline::LoweringMode::AllowHostBackend
    } else {
        chelis_compiler_api::pipeline::LoweringMode::Strict
    };
    chelis_compiler_api::pipeline::lower_checked(checked, mode)
        .map_err(|rejection| boxed_string_error(rejection.to_string()))
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
    type_env: &BTreeMap<String, DeepExpr>,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let deep_exprs =
        chelis_surf::desugar::desugar_program(decls).map_err(|error| error.to_string())?;
    Ok(lowered_root_names_from_selected_exprs(
        &deep_exprs,
        program_exprs,
        type_env,
    ))
}

fn manifest_root_names_from_decls(
    decls: &[Decl],
    checked: &chelis_types::CheckedProgram,
    target: chelis_types::types::Target,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut selected_defs = UnordSet::new();
    for expr in chelis_surf::desugar::desugar_program(decls).map_err(|error| error.to_string())? {
        collect_manifest_decl_names(&expr, &mut selected_defs);
    }
    let realizability = chelis_effects::realizability::infer_realizability(
        checked,
        chelis_compiler_api::target_capability::tensor_capable_prims(target),
    );
    Ok(
        chelis_effects::realizability::compute_root_manifest(checked, &realizability)
            .entries
            .into_iter()
            .filter(|entry| selected_defs.contains(entry.def_name.as_str()))
            .map(|entry| entry.name)
            .collect(),
    )
}

fn collect_manifest_decl_names(expr: &DeepExpr, names: &mut UnordSet<String>) {
    match expr.carrier() {
        DeepExprCarrier::DecodedNode(DeepTag::Module, _, children) => {
            for child in children.iter().skip(1) {
                collect_manifest_decl_names(child, names);
            }
        }
        DeepExprCarrier::DecodedNode(_, _, _) => {
            if let Some(name) = deep_top_level_expr_name(expr) {
                names.insert(name.to_string());
            }
        }
        DeepExprCarrier::StructuralList(_)
        | DeepExprCarrier::UndecodableHead(_, _, _)
        | DeepExprCarrier::Atom(_)
        | DeepExprCarrier::MetadataMap(_)
        | DeepExprCarrier::MetadataExpression(_) => {}
    }
}

fn lowered_root_names_from_exprs(
    exprs: &[DeepExpr],
    type_env: &BTreeMap<String, DeepExpr>,
) -> Vec<String> {
    lowered_root_names_from_selected_exprs(exprs, exprs, type_env)
}

fn lowered_root_names_from_selected_exprs(
    selected_exprs: &[DeepExpr],
    program_exprs: &[DeepExpr],
    type_env: &BTreeMap<String, DeepExpr>,
) -> Vec<String> {
    let mut out = Vec::new();
    for expr in selected_exprs {
        collect_lowered_root_names_from_expr(expr, program_exprs, type_env, &mut out);
    }
    out
}

fn collect_lowered_root_names_from_expr(
    expr: &DeepExpr,
    program_exprs: &[DeepExpr],
    type_env: &BTreeMap<String, DeepExpr>,
    out: &mut Vec<String>,
) {
    match expr.carrier() {
        DeepExprCarrier::DecodedNode(DeepTag::Module, _, children) => {
            for child in children.iter().skip(1) {
                collect_lowered_root_names_from_expr(child, program_exprs, type_env, out);
            }
        }
        DeepExprCarrier::DecodedNode(_, _, _) => {
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
        DeepExprCarrier::StructuralList(_)
        | DeepExprCarrier::UndecodableHead(_, _, _)
        | DeepExprCarrier::Atom(_)
        | DeepExprCarrier::MetadataMap(_)
        | DeepExprCarrier::MetadataExpression(_) => {}
    }
}

fn deep_top_level_expr_name(expr: &DeepExpr) -> Option<&str> {
    match expr.carrier() {
        DeepExprCarrier::DecodedNode(DeepTag::Def, _, children) => match children.first() {
            Some(DeepExpr::Atom(DeepAtom::Name(name), _)) => Some(name.as_str()),
            _ => None,
        },
        _ => None,
    }
}

fn deep_named_decl_name(expr: &DeepExpr) -> Option<&str> {
    match expr.carrier() {
        DeepExprCarrier::DecodedNode(DeepTag::Def | DeepTag::Defsig, _, children) => {
            match children.first() {
                Some(DeepExpr::Atom(DeepAtom::Name(name), _)) => Some(name.as_str()),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Host builtins that the IR evaluator (`chelis eval` / `chelis test`)
/// supports but the compiled build backends deliberately do not. Kept in
/// one place so the shared compiler gate and [`drop_unreachable_eval_only_defs`]
/// stay in agreement.
// The list itself lives in `chelis_ir::host` and is shared with the
// public compiler API's `compile_for_execution` gate, so the CLI build
// pipeline and the chelis-python path cannot drift (chelis#891 review
// finding 13).
const EVAL_ONLY_HOST_BUILTINS: &[&str] = chelis_ir::host::EVAL_ONLY_HOST_BUILTINS;

/// Drop top-level decls for any function that can never be lowered into a
/// compiled artifact and is not reachable from the entry program: one whose
/// body references an eval-only host builtin ([`EVAL_ONLY_HOST_BUILTINS`]), OR
/// one that (transitively) references such a dropped def. An unused transitive
/// dependency module (e.g. chelis-std's `Std.Process`) must not drag them into
/// the build's lowering target. Both the `def` body and its sibling `defsig`
/// are removed by name. A reachable eval-only use is preserved so the build
/// gate still rejects it. chelis#334.
///
/// The drop is a TRANSITIVE closure (chelis#1168): dropping only the DIRECT
/// eval-only users would leave an unreachable wrapper with a dangling reference
/// to a dropped def, which the monolithic full-program check in `cmd_build`
/// then rejects as an unbound variable — a spurious error `chelis check` never
/// raises, and one the layered cache path (which sees the intact pre-drop
/// decls) does not, so build accept/reject would flip on cache state.
///
/// Divergence note (tracked in chelis#1184; the direct case originated with the
/// closed chelis#334): because these unreachable defs are removed before the
/// build's type check, `chelis build` alone does NOT surface a real error (e.g.
/// a type error or non-termination) that lives inside an unreachable,
/// eval-only-tainted def — such a def can never reach a compiled artifact, and
/// the transitive closure widens this to arbitrary depth. `chelis check`
/// remains the gate for those.
fn drop_unreachable_eval_only_defs(exprs: Vec<DeepExpr>, entry_seeds: &[String]) -> Vec<DeepExpr> {
    use chelis_unord::{UnordMap, UnordSet};

    let reachable = prune_build_program_to_reachable_defs(&exprs, entry_seeds)
        .iter()
        .filter_map(|expr| deep_named_decl_name(expr).map(str::to_string))
        .collect::<UnordSet<_>>();

    // Reverse index over the UNREACHABLE named defs, borrowing from `exprs` (no
    // per-def String clones): referenced-name -> the unreachable defs that
    // reference it, plus the seed worklist of unreachable defs that directly use
    // an eval-only builtin. Reachable defs are never dropped (a reachable
    // eval-only use is preserved for the build gate), and reachability is
    // transitive, so a dropped (unreachable) def is only ever referenced by
    // another unreachable def — the closure stays within this set.
    let mut dependents: UnordMap<&str, Vec<&str>> = UnordMap::new();
    let mut worklist: Vec<&str> = Vec::new();
    for expr in &exprs {
        let Some(name) = deep_named_decl_name(expr) else {
            continue;
        };
        if reachable.contains(name) {
            continue;
        }
        let mut direct_eval_only = false;
        for var in deep_referenced_vars(expr) {
            if EVAL_ONLY_HOST_BUILTINS.contains(&var) {
                direct_eval_only = true;
            }
            dependents.entry(var).or_default().push(name);
        }
        if direct_eval_only {
            worklist.push(name);
        }
    }

    // Transitive closure via the reverse index (O(edges), single pass per node):
    // a dropped name pulls in every unreachable def that references it. Dropping
    // only the DIRECT eval-only users would leave an unreachable wrapper with a
    // dangling reference to a dropped def (see the doc comment).
    let mut drop_borrowed: UnordSet<&str> = UnordSet::new();
    while let Some(name) = worklist.pop() {
        if !drop_borrowed.insert(name) {
            continue;
        }
        if let Some(refs) = dependents.get(name) {
            worklist.extend(refs.iter().copied());
        }
    }

    // Materialize the (typically small) dropped set as owned strings so the
    // borrows into `exprs` end before the move below.
    let drop_names: UnordSet<String> = drop_borrowed
        .into_sorted()
        .into_iter()
        .map(String::from)
        .collect();
    drop(worklist);
    drop(dependents);

    exprs
        .into_iter()
        .filter(|expr| {
            deep_named_decl_name(expr)
                .map(|name| !drop_names.contains(name))
                .unwrap_or(true)
        })
        .collect()
}

/// The entry program's top-level names, which is the whole of what the build
/// pruners read from it.
///
/// Naming the seeds separately from the entry expressions lets a caller that
/// already holds the desugared whole program reuse it when the entry program
/// IS that program, instead of desugaring the same declarations twice
/// (chelis#2331). An EMPTY seed set drops every named decl, which the build
/// path relies on (see `prune_to_reachable_seeds`).
fn entry_seed_names(entry_exprs: &[DeepExpr]) -> Vec<String> {
    entry_exprs
        .iter()
        .filter_map(deep_top_level_expr_name)
        .map(str::to_string)
        .collect()
}

fn prune_build_program_to_reachable_defs(
    exprs: &[DeepExpr],
    entry_seeds: &[String],
) -> Vec<DeepExpr> {
    // Delegate to the shared reachable-defs pruner (single source of truth in
    // chelis-compiler-api, also used by the WI-3 graph-extraction producer).
    chelis_compiler_api::prune::prune_to_reachable_seeds(exprs.to_vec(), entry_seeds.to_vec())
}

/// Every `var` reference name in `expr`. Delegates to the shared traversal in
/// chelis-compiler-api so the build path and the WI-3 producer agree on what
/// "references" means.
fn deep_referenced_vars(expr: &DeepExpr) -> Vec<&str> {
    chelis_compiler_api::prune::deep_referenced_vars(expr)
}

fn apply_manifest_display_roots(
    program: &mut chelis_ir::host::ConcreteHostProgram,
    manifest: &chelis_types::manifest::RootManifest,
    target: BuildTarget,
) -> Result<(), chelis_types::unsupported::Unsupported> {
    apply_manifest_display_roots_to_globals(
        &mut program.globals,
        &program.functions,
        manifest,
        target,
    )
}

fn apply_manifest_display_roots_to_globals(
    globals: &mut Vec<chelis_ir::host::ConcreteHostBinding>,
    functions: &[chelis_ir::host::ConcreteHostFunction],
    manifest: &chelis_types::manifest::RootManifest,
    target: BuildTarget,
) -> Result<(), chelis_types::unsupported::Unsupported> {
    use chelis_ir::host::{HostBinding, HostExpr, HostExprKind};
    use chelis_types::types::Lane;

    let mut represented_defs = std::collections::BTreeSet::new();
    for binding in globals.iter_mut() {
        binding.display_name = None;
        binding.display_roots = manifest
            .entries
            .iter()
            .filter(|entry| entry.def_name == binding.name)
            .map(|entry| {
                represented_defs.insert(entry.def_name.clone());
                manifest_host_display_root(entry)
            })
            .collect();
    }

    // A pure nullary definition is an observation root even though host
    // lowering quite correctly represents it as a function and leaves
    // `program.globals` empty. Materialize the manifest-selected call as a
    // compiler-owned binding so the host emitter's ordinary, typed `main`
    // path evaluates it and renders every dotted leaf. This consumes the
    // manifest before emission; the backend never scans generated C to guess
    // whether an entry point is owed.
    let mut seen_host_defs = UnordSet::new();
    let host_defs = manifest
        .entries
        .iter()
        .filter(|entry| {
            entry.lane == Lane::Host
                && !represented_defs.contains(&entry.def_name)
                && seen_host_defs.insert(entry.def_name.as_str())
        })
        .map(|entry| entry.def_name.as_str())
        .collect::<Vec<_>>();

    let mut observation_defs = UnordMap::<String, String>::new();
    for (observation_index, def_name) in host_defs.into_iter().enumerate() {
        let entries = manifest
            .entries
            .iter()
            .filter(|entry| entry.lane == Lane::Host && entry.def_name == def_name)
            .collect::<Vec<_>>();
        let first = entries
            .first()
            .copied()
            .expect("host def came from one manifest entry");
        let Some(function) = functions.iter().find(|function| function.name == def_name) else {
            return Err(build_unavailable_root_error(
                first,
                target,
                format!(
                    "the Host lowering produced neither a value binding nor a callable `{def_name}`"
                ),
            ));
        };
        if !function.params.is_empty() {
            return Err(build_unavailable_root_error(
                first,
                target,
                format!(
                    "callable `{def_name}` still requires generated Host parameter(s) {}",
                    function
                        .params
                        .iter()
                        .map(|param| format!("`{}`", param.name))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }

        let function_name = function.name.clone();
        let ty = function.ret_ty.clone();
        let display_roots = entries
            .into_iter()
            .map(manifest_host_display_root)
            .collect();
        let mut binding_name = format!("__chelis_manifest_observation_{observation_index}");
        while globals.iter().any(|binding| binding.name == binding_name)
            || functions
                .iter()
                .any(|function| function.name == binding_name)
        {
            binding_name.push('_');
        }
        observation_defs.insert(binding_name.clone(), def_name.to_string());
        let binding = HostBinding {
            name: binding_name,
            display_name: None,
            display_roots,
            ty: ty.clone(),
            value: HostExpr::new(HostExprKind::Call {
                function: function_name,
                args: Vec::new(),
                arg_tys: Vec::new(),
                ty,
            }),
        };

        // Preserve the lowerer's existing dependency order exactly. Some
        // compiler-owned globals are inputs to a later manifested Host root
        // without themselves appearing in this Host manifest; globally
        // sorting by manifest position moves those inputs after their use.
        // A synthetic pure-nullary observation is effect-free, so inserting
        // it immediately before the next represented manifest root preserves
        // both [05-OBS-11] observation order and every existing dependency.
        let manifest_index = manifest
            .entries
            .iter()
            .position(|entry| entry.def_name == def_name)
            .expect("host def came from the manifest");
        let insertion_index = globals
            .iter()
            .position(|existing| {
                let existing_def = observation_defs
                    .get(existing.name.as_str())
                    .map(String::as_str)
                    .unwrap_or(existing.name.as_str());
                manifest
                    .entries
                    .iter()
                    .position(|entry| entry.def_name == existing_def)
                    .is_some_and(|index| index > manifest_index)
            })
            .unwrap_or(globals.len());
        globals.insert(insertion_index, binding);
    }

    Ok(())
}

fn manifest_host_display_root(
    entry: &chelis_types::manifest::RootEntry,
) -> chelis_ir::host::HostDisplayRoot {
    let short_def = display_root_name(&entry.def_name);
    let suffix = entry
        .name
        .strip_prefix(entry.def_name.as_str())
        .unwrap_or_default();
    chelis_ir::host::HostDisplayRoot {
        name: format!("{short_def}{suffix}"),
        path: entry.path.clone(),
    }
}

fn top_level_def_body(expr: &DeepExpr) -> Option<&DeepExpr> {
    match expr.carrier() {
        DeepExprCarrier::DecodedNode(DeepTag::Def, _, children) => children.get(1),
        _ => None,
    }
}

fn extend_root_names_from_value(
    name: &str,
    ty: Option<&DeepExpr>,
    value: Option<&DeepExpr>,
    out: &mut Vec<String>,
) {
    if let Some(ty) = ty
        && let DeepExprCarrier::DecodedNode(tag, _, children) = ty.carrier()
    {
        if tag == DeepTag::TFn {
            extend_root_names_from_value(name, children.last(), None, out);
            return;
        }
        if tag == DeepTag::TTuple {
            for (index, child) in children.iter().enumerate() {
                extend_root_names_from_value(&format!("{name}.{index}"), Some(child), None, out);
            }
            return;
        }
    }
    if let Some(value) = value
        && let DeepExprCarrier::DecodedNode(DeepTag::Tuple, _, children) = value.carrier()
    {
        for (index, child) in children.iter().enumerate() {
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
    match expr.carrier() {
        DeepExprCarrier::DecodedNode(_, metadata, _) => metadata.ty().map(|ty| ty.expression()),
        _ => None,
    }
}

fn display_root_name(name: &str) -> String {
    let (root, descendants) = if let Some((root, descendants)) = name.split_once('.') {
        (root, Some(descendants))
    } else {
        (name, None)
    };
    let root = if chelis_types::is_linker_format_name(root) {
        chelis_types::demangle_ident(root)
    } else {
        root.to_string()
    };
    match descendants {
        Some(descendants) => format!("{root}.{descendants}"),
        None => root,
    }
}

#[cfg(test)]
mod issue_1359_display_root_name_tests {
    use super::display_root_name;

    #[test]
    fn only_complete_reef_linker_qualification_is_removed() {
        assert_eq!(
            display_root_name("pkg__obs__labels__App__Main__record_root.field__name"),
            "record_root.field__name"
        );
        assert_eq!(
            display_root_name("Pkg__obs__labels__App__Main__RecordRoot.inner__value"),
            "RecordRoot.inner__value"
        );
        assert_eq!(display_root_name("pkg__lonely.field"), "pkg__lonely.field");
    }

    #[test]
    fn authored_repeated_underscores_survive_in_roots_and_descendants() {
        assert_eq!(
            display_root_name("record_root.field__name"),
            "record_root.field__name"
        );
        assert_eq!(
            display_root_name("record_root.inner.inner__value"),
            "record_root.inner.inner__value"
        );
        assert_eq!(display_root_name("root__tuple.1.0"), "root__tuple.1.0");
    }
}

fn type_expr_is_function(expr: &DeepExpr) -> bool {
    matches!(
        expr.carrier(),
        DeepExprCarrier::DecodedNode(DeepTag::TFn, _, _)
    )
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
    match expr.carrier() {
        DeepExprCarrier::DecodedNode(tag, metadata, children) => {
            if tag == DeepTag::DName
                && let Some(DeepExpr::Atom(DeepAtom::Name(name), _)) = children.first()
                && name != "*"
            {
                dims.push(name.clone());
            }
            metadata.visit_expressions(&mut |value, _| collect_symbolic_dims_expr(value, dims));
            for child in children {
                collect_symbolic_dims_expr(child, dims);
            }
        }
        DeepExprCarrier::MetadataMap(map) => {
            map.visit_expressions(&mut |value, _| collect_symbolic_dims_expr(value, dims));
        }
        DeepExprCarrier::MetadataExpression(meta) => {
            collect_symbolic_dims_expr(&meta.expr, dims);
            meta.metadata
                .visit_expressions(&mut |value, _| collect_symbolic_dims_expr(value, dims));
        }
        DeepExprCarrier::StructuralList(elems) => {
            for elem in elems {
                collect_symbolic_dims_expr(elem, dims);
            }
        }
        // An unknown form's metadata is not a dimension source; its children
        // are still walked.
        DeepExprCarrier::UndecodableHead(_, _, children) => {
            for child in children {
                collect_symbolic_dims_expr(child, dims);
            }
        }
        DeepExprCarrier::Atom(_) => {}
    }
}

#[cfg(test)]
mod issue_1125_e5e_outer_reader_tests {
    use super::*;
    use chelis_deep::Span;
    use chelis_deep::ast::{MetaExpr, Metadata, UnknownFormData};

    fn span() -> Span {
        Span::new(5, 13)
    }

    fn name(value: &str) -> DeepExpr {
        DeepExpr::Atom(DeepAtom::Name(value.to_string()), span())
    }

    fn dname(value: &str) -> DeepExpr {
        DeepExpr::node(
            DeepTag::DName,
            Metadata::default(),
            vec![name(value)],
            span(),
        )
    }

    #[test]
    fn manifest_reader_reads_a_decoded_module() {
        let successor = DeepExpr::node(
            DeepTag::Module,
            Metadata::default(),
            vec![
                name("fixture"),
                DeepExpr::node(
                    DeepTag::Def,
                    Metadata::default(),
                    vec![name("value"), dname("n")],
                    span(),
                ),
            ],
            span(),
        );
        let names = |expr: &DeepExpr| {
            let mut names = UnordSet::new();
            collect_manifest_decl_names(expr, &mut names);
            names.to_sorted().into_iter().cloned().collect::<Vec<_>>()
        };
        assert_eq!(names(&successor), vec!["value".to_string()]);
    }

    #[test]
    fn symbolic_dimension_reader_reads_a_decoded_dimension() {
        let successor = dname("shared");

        assert_eq!(
            collect_symbolic_dims_from_deep(&[successor]),
            vec!["shared".to_string()]
        );
        assert_eq!(
            collect_symbolic_dims_from_deep(&[dname("*")]),
            Vec::<String>::new(),
            "the wildcard remains a non-parameter on the successor carrier"
        );
    }

    #[test]
    fn symbolic_dimension_reader_explicitly_traverses_or_declines_every_carrier() {
        let exprs = vec![
            DeepExpr::BareList(vec![dname("structural_dim")], span()),
            DeepExpr::UnknownForm(Box::new(UnknownFormData {
                head: "future-form".to_string(),
                meta: Metadata::default(),
                children: vec![dname("unknown_child_dim")],
                span: span(),
            })),
            DeepExpr::MetaExpr(
                MetaExpr {
                    metadata: Metadata::default(),
                    expr: Box::new(dname("wrapped_dim")),
                },
                span(),
            ),
            name("atom"),
            DeepExpr::Map(Metadata::default(), span()),
        ];

        assert_eq!(
            collect_symbolic_dims_from_deep(&exprs),
            vec!["structural_dim", "unknown_child_dim", "wrapped_dim",]
        );
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
    matches!(
        chelis_compiler_api::pipeline::run_source(chelis_compiler_api::pipeline::PipelineRequest {
            source_kind: chelis_compiler_api::schema::SourceKind::Surf,
            source,
            entry: None,
            goal: chelis_compiler_api::pipeline::PipelineGoal::FullCheck,
        },),
        Ok(chelis_compiler_api::pipeline::PipelineOutcome::Checked(_))
    )
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
        let selected: BTreeSet<&str> = ids
            .split(',')
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .collect();
        if selected.is_empty() {
            return Err("--rules requires at least one rule id".into());
        }
        rules.retain(|r| selected.contains(r.id()));
        let found: BTreeSet<&str> = rules.iter().map(|r| r.id()).collect();
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
mod batch_fallback_reason_tests {
    use super::{BatchFallbackReason, PLAIN_BATCH_FALLBACK_MARKER, parse_plain_test_summary};

    /// The `status` strings are the published `--json` vocabulary
    /// (`spec/design/chelis_native_testing_plan.md`). Two of the five are
    /// defensive branches with no currently reachable trigger, so a CLI test
    /// cannot pin them; this exhaustive match is what stops a rename or a new
    /// variant from drifting away from the documented set. The match is written
    /// without a wildcard on purpose: a sixth variant must fail to compile here.
    #[test]
    fn every_fallback_status_matches_the_documented_vocabulary() {
        let cases = [
            BatchFallbackReason::WorkerUnavailable("spawn refused".to_string()),
            BatchFallbackReason::Timeout(90),
            BatchFallbackReason::MalformedOutput("stdout line 1 is not JSON".to_string()),
            BatchFallbackReason::WorkerFailed("batch worker exited with status 2".to_string()),
            BatchFallbackReason::IncompleteRows("expected 3 rows, got 2".to_string()),
        ];
        for case in &cases {
            let expected = match case {
                BatchFallbackReason::WorkerUnavailable(_) => "worker-unavailable",
                BatchFallbackReason::Timeout(_) => "timeout",
                BatchFallbackReason::MalformedOutput(_) => "malformed-output",
                BatchFallbackReason::WorkerFailed(_) => "worker-failed",
                BatchFallbackReason::IncompleteRows(_) => "incomplete-rows",
            };
            assert_eq!(case.status(), expected, "status drifted for {case:?}");
        }
        let statuses = cases.iter().map(|case| case.status()).collect::<Vec<_>>();
        assert_eq!(
            statuses,
            vec![
                "worker-unavailable",
                "timeout",
                "malformed-output",
                "worker-failed",
                "incomplete-rows",
            ]
        );
    }

    /// Every reason must render a non-empty sentence that names the batch
    /// worker, and must carry its detail through: an attributed report whose
    /// reason line says nothing is the failure chelis#1261 reported.
    #[test]
    fn every_fallback_message_names_the_worker_and_keeps_its_detail() {
        let detail = "the-detail-marker";
        let cases = [
            BatchFallbackReason::WorkerUnavailable(detail.to_string()),
            BatchFallbackReason::MalformedOutput(detail.to_string()),
            BatchFallbackReason::WorkerFailed(format!("batch worker hit {detail}")),
            BatchFallbackReason::IncompleteRows(detail.to_string()),
        ];
        for case in &cases {
            let message = case.message();
            assert!(
                message.contains("batch worker"),
                "unattributed message for {case:?}: {message}"
            );
            assert!(
                message.contains(detail),
                "detail was dropped for {case:?}: {message}"
            );
        }
        let timeout = BatchFallbackReason::Timeout(90).message();
        assert!(
            timeout.contains("batch worker") && timeout.contains("90s"),
            "timeout message lost its window: {timeout}"
        );
    }

    /// The plain summary marker rides on the same line the supervisor's
    /// incomplete-suite renderer parses counts from, so the parser has to see
    /// through it. A clean summary line must parse exactly as before.
    #[test]
    fn plain_summary_parses_with_and_without_the_fallback_marker() {
        assert_eq!(parse_plain_test_summary("2 passed, 1 failed"), Some((2, 1)));
        assert_eq!(
            parse_plain_test_summary(&format!("2 passed, 1 failed{PLAIN_BATCH_FALLBACK_MARKER}")),
            Some((2, 1))
        );
        assert_eq!(parse_plain_test_summary("not a summary"), None);
        assert_eq!(parse_plain_test_summary("x passed, 1 failed"), None);
    }
}

#[cfg(test)]
mod eval_only_pruning_tests {
    use super::{
        deep_named_decl_name, drop_unreachable_eval_only_defs, entry_seed_names,
        expanded_desugared_program,
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
            "def unused_runner(cmd: string, args: List[string]) -> (i64, string, string) = process_run(cmd, args)\n\
             def main(x: tensor[2, 2, f32], w: tensor[2, 2, f32]) -> tensor[2, 2, f32] = matmul(&x, &w)\n",
        );
        let entry = desugar(
            "def main(x: tensor[2, 2, f32], w: tensor[2, 2, f32]) -> tensor[2, 2, f32] = matmul(&x, &w)\n",
        );
        let kept = drop_unreachable_eval_only_defs(full, &entry_seed_names(&entry));
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
            desugar("def main() -> (i64, string, string) = process_run(\"echo\", [\"hi\"])\n");
        let entry = entry_seed_names(&exprs);
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
    use super::compiler_error_messages;
    use chelis_compiler_api::compiler::reject_unsupported_hip_ops;
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

    /// Load x: [4] f32 plus a rank-0 i32 Load scalar (not a `Shape` read;
    /// the HIP seam blanket-rejects `Shape` first and these tests must
    /// exercise the movement/reshape arms).
    fn dag_with_scalar() -> (Dag, chelis_ir::dag::NodeId, chelis_ir::dag::NodeId) {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(lit_dims(&[4]), Prim::F32),
            None,
        );
        let m = dag.add_node(
            decl,
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
        let decl = dag.nodes()[0].owner.decl;
        dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: vec![RtDim::Node(1)],
            },
            vec![x, m],
            ty(lit_dims(&[4]), Prim::F32),
            None,
        );
        let err = reject_unsupported_hip_ops(&dag)
            .expect_err("HIP seam must reject a node-valued reshape target");
        let message = compiler_error_messages(&err);
        assert!(
            message.contains("reshape") && message.contains("--target c"),
            "unexpected message: {message}"
        );
    }

    /// Concrete (literal) bounds keep flowing through the HIP seam; the
    /// rejection is scoped to node-valued dims only.
    #[test]
    fn hip_seam_accepts_literal_movement_and_reshape() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            ty(lit_dims(&[4]), Prim::F32),
            None,
        );
        let shrunk = dag.add_node(
            decl,
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(0), RtDim::Lit(2))],
            },
            vec![x],
            ty(lit_dims(&[2]), Prim::F32),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(2), RtDim::Lit(1)],
            },
            vec![shrunk],
            ty(lit_dims(&[2, 1]), Prim::F32),
            None,
        );
        reject_unsupported_hip_ops(&dag).expect("literal bounds must pass the HIP seam");
    }
}
