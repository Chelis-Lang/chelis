//! Chelis compiler CLI.

mod style_gate;

use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr};
use chelis_surf::ast::Decl;
use clap::{ArgAction, ArgGroup, Parser, Subcommand};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::io::{self, BufRead, Write};
use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::time::Duration;

const RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-runtime/include/chelis_runtime.h"
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

fn copy_runtime_artifacts(
    runtime_dir: &Path,
    extras: ExtraRuntimeArtifacts,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::write(runtime_dir.join("chelis_runtime.h"), RUNTIME_H)?;
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
        /// Bypass `chelis fmt --check` and `chelis lint --check` gates.
        /// Emergency use only; CI must not pass this flag.
        #[arg(long, action = ArgAction::SetTrue)]
        allow_style_violations: bool,
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
    },
    /// Lint naming conventions per `spec/01-nomenclature.md`
    Lint {
        /// Paths to lint. Defaults to the current directory.
        paths: Vec<PathBuf>,
        /// Exit nonzero on any violation (CI use).
        #[arg(long)]
        check: bool,
        /// Run only the rule with this id.
        #[arg(long)]
        rule: Option<String>,
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
}

fn main() {
    chelis_ir::lower::install_chelis_panic_hook();
    let cli = Cli::parse();
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
            allow_style_violations,
        }) => cmd_eval(file.as_deref(), expr.as_deref(), allow_style_violations),
        Some(Command::Check {
            file,
            allow_style_violations,
        }) => cmd_check(&file, allow_style_violations),
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
        }) => match cmd_internal_test_file(
            &file,
            &rel_display,
            filter.as_deref(),
            Duration::from_secs(timeout.max(1)),
        ) {
            Ok(code) => std::process::exit(code),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(2);
            }
        },
        Some(Command::Test {
            path,
            filter,
            json,
            timeout,
        }) => match cmd_test(path.as_deref(), filter.as_deref(), json, timeout) {
            Ok(code) => std::process::exit(code),
            Err(err) => {
                eprintln!("error: {err}");
                std::process::exit(2);
            }
        },
        Some(Command::Lint { paths, check, rule }) => match cmd_lint(paths, check, rule.as_deref())
        {
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
        match chelis_types::check_phase0e_program(&deep_exprs) {
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
        let deep_exprs = chelis_deep::parser::parse_str_strict(&source)?;
        let surf = chelis_surf::decompile::decompile_program_with_context(
            &deep_exprs,
            &options,
            synthetic_name,
        );
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
        print!("{surf}");
    }
    Ok(())
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
        // .ch: parse Surf -> pretty-print Surf while preserving surface choices
        let decls = chelis_surf::parser::parse_str(&source)?;
        chelis_surf::format::format_program(&decls)
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
    // `crates/chelis-cli/tests/phase_h_eval_in_context.rs` for the parity
    // probes and the Phase G acceptance suite
    // (`crates/chelis-compiler-api/tests/phase_g_compiled_context.rs`)
    // for the underlying API parity guarantee.
    match (file, expr) {
        (Some(path), _) => {
            if let Some(package_root) = detect_eval_package_root(path)? {
                let source = fs::read_to_string(path)?;
                match run_eval_in_context(&package_root, &source) {
                    Ok(()) => return Ok(()),
                    Err(EvalInContextError::HashUnsupported) => {
                        // The Phase G hash step does not yet cover
                        // `LocalRegistry` packages (chelis-std published
                        // via `chelis reef publish`). Phase I will
                        // extend `LoadedPackage` to retain the
                        // extracted cache root so `source_digests` can
                        // hash them. Until then, fall through to the
                        // legacy `prepare_eval` path so users on a
                        // local-registry-backed chelis-std setup keep
                        // the same eval behavior they had before the
                        // Phase H refactor — byte-identical output to
                        // pre-refactor on the same input.
                    }
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
            run_eval_emit(try_eval(SourceKind::Surf, &source, Some(&selected_roots)))
        }
        (None, Some(e)) => {
            // `--expr` is by construction a one-line snippet with no reef
            // resolution — keep the legacy path.
            let source = format!("__eval_result = {e}");
            run_eval_emit(try_eval(SourceKind::Surf, &source, None))
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

/// Outcomes from the Phase H new path. Distinct from a generic boxed
/// error so the caller can fall back to the legacy `prepare_eval` path
/// on a Phase-I-shaped hash gap (`LocalRegistry` packages aren't yet
/// hashable) without swallowing real compile / eval failures.
enum EvalInContextError {
    /// `compile_reef_context` couldn't hash the package graph because
    /// `source_digests` doesn't yet cover `LocalRegistry`. The CLI can
    /// fall back to the legacy path here without losing correctness —
    /// the legacy path doesn't compute that hash.
    HashUnsupported,
    /// Any other failure: type error, effect error, eval error, etc.
    /// Propagate to the user with the same format the legacy path used.
    Compile(String),
}

/// Build a `CompiledContext` for `package_root`, then evaluate `source`
/// against it. `reef_home` is sourced from the `CHELIS_REEF_HOME` env var
/// if present (matching how `chelis test` plumbs it to workers); Phase K
/// uses it to key the disk cache so a warm `chelis eval --file` re-run
/// against unchanged sources skips the ~67s library compile entirely.
fn run_eval_in_context(package_root: &Path, source: &str) -> Result<(), EvalInContextError> {
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
                // The hash step is the one place `compile_reef_context`
                // can fail today on a graph the legacy path handles fine
                // (LocalRegistry source_digests TODO). Detect that
                // specifically — anything else is a real error and must
                // not be silently swallowed.
                let is_hash_unsupported = err
                    .errors
                    .iter()
                    .any(|d| d.kind == "hash_error" && d.message.contains("LocalRegistry"));
                if is_hash_unsupported {
                    return Err(EvalInContextError::HashUnsupported);
                }
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
    let formatted = format_eval_result(&result);
    if formatted.is_empty() {
        return Ok(());
    }
    println!("{formatted}");
    Ok(())
}

fn run_eval_emit(outcome: Result<String, String>) -> Result<(), Box<dyn std::error::Error>> {
    match outcome {
        Ok(result) => {
            if result.is_empty() {
                return Ok(());
            }
            println!("{result}");
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

fn cmd_check(
    target: &Path,
    allow_style_violations: bool,
) -> Result<(), Box<dyn std::error::Error>> {
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
            return Ok(());
        }
        let mut had_error = false;
        let mut entries: Vec<String> = Vec::with_capacity(files.len());
        for file in &files {
            match cmd_check_one(file, allow_style_violations) {
                Ok(json) => {
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
        return Ok(());
    }

    let json = cmd_check_one(target, allow_style_violations)?;
    println!("{json}");
    Ok(())
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
        if path.extension().and_then(|e| e.to_str()) != Some("ch") {
            continue;
        }
        files.push(path.to_path_buf());
    }
    Ok(files)
}

fn cmd_check_one(
    file: &Path,
    allow_style_violations: bool,
) -> Result<String, Box<dyn std::error::Error>> {
    if let Ok(source) = fs::read_to_string(file) {
        style_gate::enforce_style_gate(file, &source, allow_style_violations)?;
    }
    let (decls, _) = load_check_build_decls(file)?;
    let deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
    let mut report = chelis_types::check_phase0e_fitness(&deep_exprs);
    let (effect_errors, linearity_errors) = match chelis_types::check_typed_program(&deep_exprs) {
        Ok(checked) => match chelis_effects::check_program(&checked) {
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
                "{{\"kind\":\"{:?}\",\"message\":{},\"severity\":{}{}{}}}",
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
            "{{\"kind\":\"{:?}\",\"message\":{},\"severity\":{}}}",
            e.kind,
            serde_json::to_string(&e.message).unwrap_or_default(),
            e.severity,
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
            "  \"unresolved_names\": {},\n",
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
        errors_json.join(","),
    );
    Ok(json)
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
    let (decls, entry_decls) = load_check_build_decls(file)?;
    reject_with_seed_for_build_target(&decls, target)?;
    let full_deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
    let entry_deep_exprs = expanded_desugared_program(&entry_decls).map_err(boxed_string_error)?;
    let pruned_deep_exprs =
        prune_build_program_to_reachable_defs(&full_deep_exprs, &entry_deep_exprs);
    let preserve_host_library_surface =
        if target == "c" && pruned_deep_exprs.len() != full_deep_exprs.len() {
            let full_checked = checked_program_with_effects(&full_deep_exprs)
                .map_err(|e| format!("Check errors: {e}"))?;
            chelis_ir::host::lower_compiled_program(&full_checked)
                .host
                .as_ref()
                .map(chelis_ir::host::host_program_requires_host_backend)
                .unwrap_or(false)
        } else {
            false
        };
    let deep_exprs = if preserve_host_library_surface {
        full_deep_exprs
    } else {
        pruned_deep_exprs
    };
    let symbolic_dims = collect_symbolic_dims_from_deep(&deep_exprs);
    let checked =
        checked_program_with_effects(&deep_exprs).map_err(|e| format!("Check errors: {e}"))?;
    chelis_effects::validate_build_target(&checked, target)
        .map_err(|errors| format_effect_errors(&errors))?;
    let mut compiled_program = chelis_ir::host::lower_compiled_program(&checked);
    let mut dag = chelis_ir::lower::lower_program(&checked);
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
                    chelis_ir::host::HostType::Fn(_, _) => None,
                    _ => host_display_root_name(&binding.name, &entry_display_root_names).or_else(
                        || {
                            // Tuple-typed top-level bindings get their root
                            // name expanded into `name.0` / `name.1` entries
                            // by `extend_root_names_from_value` (matching
                            // eval-side behavior). Surface a synthetic
                            // tuple-prefix display name so the C emitter
                            // can render the per-field "name.i = ..." lines.
                            if matches!(&binding.ty, chelis_ir::host::HostType::Tuple(_)) {
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
    dag.set_roots(selected);
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
                let unresolved = chelis_ir::host::host_program_unresolved_call_sites(host_program);
                if !unresolved.is_empty() {
                    return Err(format!(
                        "`chelis build --target c` can't lower these defs — their body \
                         applies/binds `grad` (or `vmap`) in a position the host lane \
                         can't resolve (inline `grad(f)(x)` or `g = grad(f); g(x)`). \
                         Workaround that compiles today: make the function you want to \
                         differentiate a parameter of the enclosing def, then call \
                         `grad(local, wrt=(arg))(arg)` where `local` is a locally-bound \
                         fn that uses the parameter; and make sure that function uses \
                         only pure tensor ops (sum, add, mul, einsum, etc.) — `grad` \
                         through host-lane `fold`/`map` is not currently supported, \
                         rewrite to `tensor_to_scalar(sum(mul(v, v), 0))` or `einsum`. \
                         See `build_c_tensor_grad_local_wrapper_over_function_param_builds` \
                         in crates/chelis-cli/tests/cli.rs for a compiling example. \
                         Affected defs: {}",
                        unresolved.join(", ")
                    )
                    .into());
                }
                let result = chelis_backend_c::codegen_host_program(host_program, func_name);
                cmd_build_c_result(result, func_name, output, &symbolic_dims)
            } else {
                reject_unsupported_effect_ops(&dag, "c")?;
                reject_unsupported_c_precisions(&dag)?;
                let fused = chelis_ir::fuse::fuse(&dag);
                cmd_build_c(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        "hip" => {
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
                let result = chelis_backend_c::codegen_host_program(host_program, func_name);
                cmd_build_hip_host(result, func_name, output)
            } else {
                let mut hip_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    chelis_ir::lower::lower_program(&checked)
                };
                hip_dag = chelis_ir::optimize::dead_code_eliminate(&hip_dag);
                reject_unsupported_effect_ops(&hip_dag, "hip")?;
                reject_unsupported_hip_ops(&hip_dag)?;
                let fused = chelis_ir::fuse::fuse(&hip_dag);
                cmd_build_hip(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        "metal" => {
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
                let result = chelis_backend_c::codegen_host_program(host_program, func_name);
                cmd_build_hip_host(result, func_name, output)
            } else {
                let mut metal_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    chelis_ir::lower::lower_program(&checked)
                };
                metal_dag = chelis_ir::optimize::dead_code_eliminate(&metal_dag);
                reject_unsupported_effect_ops(&metal_dag, "metal")?;
                reject_unsupported_metal_ops(&metal_dag)?;
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
/// — flows through the existing `chelis_types::check_phase0e_program`
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
    let deep_exprs = chelis_deep::parser::parse_str_strict(&source)
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
            chelis_ir::host::lower_compiled_program(&full_checked)
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
    let mut compiled_program = chelis_ir::host::lower_compiled_program(&checked);
    let mut dag = chelis_ir::lower::lower_program(&checked);
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
                    chelis_ir::host::HostType::Fn(_, _) => None,
                    _ => host_display_root_name(&binding.name, &entry_display_root_names).or_else(
                        || {
                            if matches!(&binding.ty, chelis_ir::host::HostType::Tuple(_)) {
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
    dag.set_roots(selected);
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
                let unresolved = chelis_ir::host::host_program_unresolved_call_sites(host_program);
                if !unresolved.is_empty() {
                    return Err(format!(
                        "`chelis build --deep --target c` can't lower these defs — \
                         they apply/bind `grad` (or `vmap`) in a position the host \
                         lane can't resolve. Affected defs: {}",
                        unresolved.join(", ")
                    )
                    .into());
                }
                let result = chelis_backend_c::codegen_host_program(host_program, func_name);
                cmd_build_c_result(result, func_name, output, &symbolic_dims)
            } else {
                reject_unsupported_effect_ops(&dag, "c")?;
                reject_unsupported_c_precisions(&dag)?;
                let fused = chelis_ir::fuse::fuse(&dag);
                cmd_build_c(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        "hip" => {
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
                let result = chelis_backend_c::codegen_host_program(host_program, func_name);
                cmd_build_hip_host(result, func_name, output)
            } else {
                let mut hip_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    chelis_ir::lower::lower_program(&checked)
                };
                hip_dag = chelis_ir::optimize::dead_code_eliminate(&hip_dag);
                reject_unsupported_effect_ops(&hip_dag, "hip")?;
                reject_unsupported_hip_ops(&hip_dag)?;
                let fused = chelis_ir::fuse::fuse(&hip_dag);
                cmd_build_hip(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        "metal" => {
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
                let result = chelis_backend_c::codegen_host_program(host_program, func_name);
                cmd_build_hip_host(result, func_name, output)
            } else {
                let mut metal_dag = if let Some(entry_dag) = preferred_entry_dag {
                    entry_dag
                } else if !dag.roots().is_empty() {
                    dag.clone()
                } else {
                    chelis_ir::lower::lower_program(&checked)
                };
                metal_dag = chelis_ir::optimize::dead_code_eliminate(&metal_dag);
                reject_unsupported_effect_ops(&metal_dag, "metal")?;
                reject_unsupported_metal_ops(&metal_dag)?;
                let fused = chelis_ir::fuse::fuse(&metal_dag);
                cmd_build_metal(&fused, func_name, file, output, &symbolic_dims)
            }
        }
        other => Err(format!("unknown target '{other}': expected 'c', 'hip', or 'metal'").into()),
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
                    let registry_root = chelis_reef::registry_home()?;
                    let results = chelis_reef::install_from_lockfile(&pkg_root, &registry_root)?;
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
                                    "Skipped path dep {name} {version} (path = {path}) — \
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
                                     (compiler version {compiler_version}) — \
                                     ships with the compiler, not fetched"
                                );
                            }
                            chelis_reef::LockfileInstallEntry::SkippedNoOrigin {
                                name,
                                version,
                            } => {
                                eprintln!(
                                    "error: lockfile entry `{name}` v{version} has no \
                                     `remote_origin` recorded; cannot fetch. \
                                     Run `chelis reef install --bootstrap` (or re-run \
                                     `--from-github`) to populate the origin."
                                );
                                any_no_origin = true;
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

/// Discover and execute Chelis-native tests.
///
/// Walks `.ch` files under `path` (default `tests/` in CWD), extracts nullary
/// `def test_*` functions, and evaluates each against the surrounding module
/// with a shared reef graph. Returns the process exit code:
///
/// * `0` — every selected test passed.
/// * `1` — at least one test failed.
/// * `2` — runner error (missing dir, missing reef package, or no test files parsed).
fn cmd_test(
    path: Option<&Path>,
    filter: Option<&str>,
    json: bool,
    timeout_secs: u64,
) -> Result<i32, String> {
    let raw_cwd = env::current_dir().map_err(|e| format!("failed to read cwd: {e}"))?;
    let target = match path {
        Some(p) => p.to_path_buf(),
        None => raw_cwd.join("tests"),
    };

    if !target.exists() {
        return Err(format!(
            "path `{}` does not exist — pass a tests directory or a single .ch file",
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
    // (annotate_phase0e_program now registers prelude ADTs, matching
    // the _with_context variants) and the worker re-wired through
    // prepare_eval_in_context, the parent re-enables the
    // `compile_reef_context` build. The encoded context is handed to
    // each per-file worker via a bincode tempfile + env var; workers
    // rehydrate the library snapshot ONCE per spawn instead of
    // re-running the full reef graph + compile pipeline per file.
    //
    // Best-effort: failures fall back to the legacy reef-graph path,
    // which is still correct (just slower). Today the most common
    // miss is LocalRegistry packages whose `source_digests` step
    // isn't yet implemented.
    //
    // Phase K: route through `load_or_compile_for_package` so an
    // unchanged-source re-run (typical CI / dev-loop iteration on
    // tests) skips the ~67s library compile entirely. When
    // CHELIS_REEF_HOME is unset we still fall back to the legacy
    // `compile_reef_context` path via the helper's empty-reef-home
    // guardrail (see `load_or_compile_for_package` doc comment) — same
    // wall-clock as pre-Phase-K, no leakage.
    let reef_home_path = env::var("CHELIS_REEF_HOME")
        .map(PathBuf::from)
        .unwrap_or_default();
    let context_tempfile_opt: Option<CompiledContextTempfile> =
        match chelis_compiler_api::load_or_compile_for_package(&reef_home_path, &cwd, true) {
            Ok(context) => {
                let context_bytes = context.encode()?;
                drop(context);
                let tempfile = CompiledContextTempfile::write(&context_bytes)?;
                drop(context_bytes);
                Some(tempfile)
            }
            Err(_err) => None,
        };

    let mut passed: usize = 0;
    let mut failed: usize = 0;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    let self_path =
        std::env::current_exe().map_err(|e| format!("could not locate chelis binary: {e}"))?;

    for file in &test_files {
        let rel_display = file
            .strip_prefix(&cwd)
            .unwrap_or(file.as_path())
            .display()
            .to_string();

        let rows = run_test_file_subprocess(
            &self_path,
            &cwd,
            file,
            &rel_display,
            filter,
            timeout_secs,
            context_tempfile_opt.as_ref().map(|t| t.path()),
        );

        if rows.is_empty() {
            // Nothing matched the filter in this file — skip silently so the
            // operator can narrow a run without seeing noise.
            continue;
        }

        if json {
            for row in &rows {
                writeln!(out, "{}", row.to_json()).map_err(|e| e.to_string())?;
                match row.status {
                    TestStatus::Pass => passed += 1,
                    TestStatus::Fail => failed += 1,
                }
            }
        } else {
            writeln!(out, "{rel_display}").map_err(|e| e.to_string())?;
            for row in &rows {
                writeln!(out, "  {}", row.render_plain()).map_err(|e| e.to_string())?;
                match row.status {
                    TestStatus::Pass => passed += 1,
                    TestStatus::Fail => failed += 1,
                }
            }
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
                "`{}` is not a .ch file — `chelis test` only accepts Chelis source",
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
const FILTER_INACTIVE_MARKER: &str = "(filter inactive — file-level error) ";

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

/// Spawn `chelis __test_file <file> --rel-display ... --filter ... --timeout N`
/// as a subprocess. Capture its NDJSON stdout and parse into TestRows. A
/// child crash (stack overflow, panic in the evaluator) only kills the child;
/// the parent attributes the loss as a file-level worker crash and moves on.
fn run_test_file_subprocess(
    self_path: &Path,
    cwd: &Path,
    file: &Path,
    rel_display: &str,
    filter: Option<&str>,
    timeout_secs: u64,
    compiled_context_path: Option<&Path>,
) -> Vec<TestRow> {
    let mut cmd = std::process::Command::new(self_path);
    cmd.arg("__test_file")
        .arg(file)
        .arg("--rel-display")
        .arg(rel_display)
        .arg("--timeout")
        .arg(timeout_secs.to_string())
        .current_dir(cwd);
    if let Some(path) = compiled_context_path {
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
    let output = match cmd.output() {
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
    let stdout = String::from_utf8_lossy(&output.stdout);
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
    let worker_crashed = match output.status.code() {
        None => true,
        Some(0) => false,
        Some(1) => rows.is_empty(),
        Some(_) => true,
    };
    if worker_crashed {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let msg = if let Some(code) = output.status.code() {
            format!("worker exited {code}: {}", stderr.trim())
        } else {
            let signal_str = worker_signal_str(&output.status);
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
) -> Result<i32, String> {
    // Hidden testing knob — gates the regression test for per-file
    // subprocess isolation in `crates/chelis-cli/tests/phase3t_subprocess_isolation.rs`.
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

    // Phase H: when the parent populates `CHELIS_TEST_COMPILED_CONTEXT`,
    // load the bincode-encoded `CompiledContext` from the path it points
    // at. Workers used to run `prepare_reef_graph` per file, paying the
    // chelis-std re-check + re-lower cost N times per `chelis test`
    // invocation; now the parent runs that pipeline ONCE and hands the
    // result through. If the env var is missing (e.g., the worker is
    // invoked directly without going through `chelis test`), we fall
    // back to the reef-graph path so the worker still works standalone.
    let compiled_context_env = env::var("CHELIS_TEST_COMPILED_CONTEXT").ok();
    let exec_context = match compiled_context_env.as_deref() {
        Some(path) if !path.is_empty() => {
            let bytes = fs::read(path)
                .map_err(|e| format!("read CHELIS_TEST_COMPILED_CONTEXT tempfile `{path}`: {e}"))?;
            let ctx = chelis_compiler_api::CompiledContext::decode(&bytes)
                .map_err(|e| format!("decode CHELIS_TEST_COMPILED_CONTEXT: {e}"))?;
            // RT-H H3a fix: verify the decoded context belongs to this
            // worker's cwd. Without this, a stale tempfile, racy
            // pre-set env var, or a malicious actor could substitute a
            // context for an unrelated package and the worker would
            // silently run tests against the wrong library state.
            // Cheap path-equality check: the parent set both cwd and
            // the env var, so a genuine pairing has matching roots.
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
            TestExecutionContext::Context(Box::new(ctx))
        }
        _ => {
            let cwd = env::current_dir().map_err(|e| format!("failed to read cwd: {e}"))?;
            let graph = chelis_reef::prepare_reef_graph(&cwd)?;
            TestExecutionContext::ReefGraph(graph)
        }
    };

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
    let file_result = run_test_file(&exec_context, file, filter, rel_display, timeout, |row| {
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
    });
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
    // `CompiledContext` is ~640 bytes (TypeEnv + reef state + DAG carrier);
    // `PreparedReefGraph` is ~192 bytes. Box the larger variant so the
    // enum's stack footprint stays compact regardless of which arm runs.
    Context(Box<chelis_compiler_api::CompiledContext>),
    ReefGraph(chelis_reef::PreparedReefGraph),
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
    if matched_tests.is_empty() {
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
/// monolithic `annotate_phase0e_program` was masking real
/// use-after-consume violations because it built with an empty
/// `AdtRegistry`; see `docs/archive/rca/lin_rca_report.md`) and chelis-std + the
/// CLI test fixtures rewritten to use `&t` / `copy(t)` at the right
/// sites, the `Context` arm now goes through
/// `prepare_eval_in_context(ctx, source)`. Per-file work drops from
/// "full pipeline on ~50 modules" to "parse + check + lower the test
/// file's ~10 lines." The `ReefGraph` arm stays on the legacy path
/// for `LocalRegistry` packages whose graph the new context-builder
/// can't yet hash.
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
            // Phase K: `prepare_eval` is deprecated externally but retained
            // as the LocalRegistry fallback path until source_digests grows
            // LocalRegistry support. Silence the deprecation here — this
            // is the canonical fallback.
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
                "duplicate test definition `{name}` — each `def test_*()` in a test file must have a unique name"
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

fn load_check_build_decls(
    file: &Path,
) -> Result<(Vec<Decl>, Vec<Decl>), Box<dyn std::error::Error>> {
    if let Some(prepared) =
        chelis_reef::prepare_program_for_file(file).map_err(boxed_string_error)?
    {
        return Ok((prepared.decls, prepared.entry_decls));
    }
    let source = fs::read_to_string(file)?;
    let decls = chelis_surf::parser::parse_str(&source)?;
    Ok((decls.clone(), decls))
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
    for node in dag.nodes() {
        match &node.op {
            chelis_ir::dag::RiscOp::Pad { .. } => {
                return Err(format!(
                    "`chelis build --target hip` does not yet support `pad`; lowered node {} requires it",
                    node.id.0
                )
                .into());
            }
            chelis_ir::dag::RiscOp::Shrink { .. } => {
                return Err(format!(
                    "`chelis build --target hip` does not yet support `shrink`; lowered node {} requires it",
                    node.id.0
                )
                .into());
            }
            _ => {}
        }
        match node.output_type.precision {
            chelis_types::types::Prim::F32 | chelis_types::types::Prim::Bool => {}
            other => {
                return Err(format!(
                    "`chelis build --target hip` DAG path only supports f32/bool tensors; \
                     node {} carries precision `{}`. \
                     The HIP backend is single-entry and doesn't route through a \
                     host-lane wrapper — rewrite the program to use f32 tensors or \
                     build it with `--target c` instead.",
                    node.id.0,
                    other.name()
                )
                .into());
            }
        }
    }
    Ok(())
}

fn reject_unsupported_metal_ops(
    dag: &chelis_ir::dag::Dag,
) -> Result<(), Box<dyn std::error::Error>> {
    for node in dag.nodes() {
        match &node.op {
            chelis_ir::dag::RiscOp::Pad { .. } => {
                return Err(format!(
                    "`chelis build --target metal` does not yet support `pad`; lowered node {} requires it",
                    node.id.0
                )
                .into());
            }
            chelis_ir::dag::RiscOp::Shrink { .. } => {
                return Err(format!(
                    "`chelis build --target metal` does not yet support `shrink`; lowered node {} requires it",
                    node.id.0
                )
                .into());
            }
            _ => {}
        }
        match node.output_type.precision {
            chelis_types::types::Prim::F32 | chelis_types::types::Prim::Bool => {}
            other => {
                return Err(format!(
                    "`chelis build --target metal` DAG path only supports f32/bool tensors; \
                     node {} carries precision `{}`. \
                     Apple Silicon GPU has limited f64 support; rewrite the program \
                     to use f32 tensors or build it with `--target c` instead.",
                    node.id.0,
                    other.name()
                )
                .into());
            }
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
    for node in dag.nodes() {
        match node.output_type.precision {
            chelis_types::types::Prim::F32 | chelis_types::types::Prim::Bool => {}
            other => {
                return Err(format!(
                    "`chelis build --target c` DAG path only supports f32/bool tensors; \
                     node {} carries precision `{}`. \
                     Non-f32/bool tensors must flow through the host-lane wrapper \
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
        "deep" => chelis_validate::validate_deep(&source),
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
    );
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
    let result = chelis_backend_hip::codegen_hip(dag, func_name);

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

fn try_eval(
    source_kind: SourceKind,
    source: &str,
    selected_roots: Option<&[String]>,
) -> Result<String, String> {
    let request = EvalRequest {
        source_kind,
        source: source.to_string(),
        bindings: BTreeMap::new(),
    };
    let result = if let Some(roots) = selected_roots {
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
    })?;

    Ok(format_eval_result(&result))
}

/// Format an `EvalResult` into the shape `chelis eval --file` emits on
/// stdout: the transcript lines first, then either a single value (when
/// exactly one root) or `<name> = <value>` lines (when multiple). Shared
/// between the legacy `try_eval` path and the Phase H
/// `eval_in_context` path so the output is byte-identical for either
/// dispatch.
fn format_eval_result(result: &chelis_compiler_api::schema::EvalResult) -> String {
    let mut lines = result.transcript.clone();
    if result.roots.len() == 1 {
        if let Some(root) = result.roots.first() {
            lines.push(format_execution_value(&root.value));
        }
        return lines.join("\n");
    }

    lines.extend(result.roots.iter().enumerate().map(|(index, root)| {
        let name = root.name.clone().unwrap_or_else(|| format!("_{index}"));
        format!(
            "{} = {}",
            display_root_name(&name),
            format_execution_value(&root.value)
        )
    }));
    lines.join("\n")
}

fn format_execution_value(value: &ExecutionValue) -> String {
    match value {
        ExecutionValue::Tensor { value } => {
            // Limit raised from 10 to 32 to match the build-target renderer
            // (red-team v0.2.6 MEDIUM: a 4x4 attention output rendered only
            // 10 of 16 elements with no truncation marker). When the tensor
            // exceeds the cap, append a trailing `...` so downstream parsers
            // can distinguish a truncated prefix from a complete render.
            const PRINT_LIMIT: usize = 32;
            let visible = value.data.len().min(PRINT_LIMIT);
            if visible < value.data.len() {
                format!(
                    "tensor(shape={:?}, data={:?} + ...)",
                    value.shape,
                    &value.data[..visible]
                )
            } else {
                format!("tensor(shape={:?}, data={:?})", value.shape, value.data)
            }
        }
        ExecutionValue::Int64 { value } => value.to_string(),
        ExecutionValue::Float64 { value } => value.to_string(),
        ExecutionValue::Bool { value } => value.to_string(),
        ExecutionValue::String { value } => value.clone(),
        ExecutionValue::List { value: items } => format!(
            "[{}]",
            items
                .iter()
                .map(format_execution_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Dict { entries } => format!(
            "dict({})",
            entries
                .iter()
                .map(|entry| format!(
                    "{}: {}",
                    format_execution_value(&entry.key),
                    format_execution_value(&entry.value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Tuple { value: items } => format!(
            "({})",
            items
                .iter()
                .map(format_execution_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Adt { ctor, fields } if fields.is_empty() => ctor.clone(),
        ExecutionValue::Adt { ctor, fields } => format!(
            "{}({})",
            ctor,
            fields
                .iter()
                .map(format_execution_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        ExecutionValue::Unit => "()".to_string(),
    }
}

fn checked_program_with_effects(
    deep_exprs: &[chelis_deep::ast::Expr],
) -> Result<chelis_types::CheckedProgram, String> {
    let checked = chelis_types::check_phase0e_program(deep_exprs)
        .map_err(|r| format!("Type errors: {:?}", r.errors))?;
    let checked =
        chelis_effects::check_program(&checked).map_err(|errors| format_effect_errors(&errors))?;
    chelis_types::check_linearity(&checked).map_err(|errors| format_type_errors(&errors))
}

fn expanded_desugared_program(
    decls: &[chelis_surf::ast::Decl],
) -> Result<Vec<chelis_deep::ast::Expr>, String> {
    let deep = chelis_surf::desugar::desugar_program(decls);
    chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
        .map(|expanded| expanded.into_exprs())
        .map_err(|err| err.to_string())
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
        chelis_ir::dag::symbolic_bindings(dag)
            .into_iter()
            .map(|binding| binding.name)
            .collect()
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
    match list.elements.first() {
        Some(DeepExpr::Atom(DeepAtom::Symbol(tag), _)) if tag == "module" => {
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
    match list.elements.first() {
        Some(DeepExpr::Atom(DeepAtom::Symbol(tag), _)) if tag == "module" => {
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
    match (list.elements.first(), list.elements.get(2)) {
        (
            Some(DeepExpr::Atom(DeepAtom::Symbol(tag), _)),
            Some(DeepExpr::Atom(DeepAtom::Symbol(name), _)),
        ) if tag == "def" => Some(name.as_str()),
        _ => None,
    }
}

fn deep_named_decl_name(expr: &DeepExpr) -> Option<&str> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    match (list.elements.first(), list.elements.get(2)) {
        (
            Some(DeepExpr::Atom(DeepAtom::Symbol(tag), _)),
            Some(DeepExpr::Atom(DeepAtom::Symbol(name), _)),
        ) if tag == "def" || tag == "defsig" => Some(name.as_str()),
        _ => None,
    }
}

fn prune_build_program_to_reachable_defs(
    exprs: &[DeepExpr],
    entry_exprs: &[DeepExpr],
) -> Vec<DeepExpr> {
    use std::collections::{HashMap, HashSet, VecDeque};

    let def_map = exprs
        .iter()
        .filter_map(|expr| deep_top_level_expr_name(expr).map(|name| (name.to_string(), expr)))
        .collect::<HashMap<_, _>>();
    let reachable_seed = entry_exprs
        .iter()
        .filter_map(deep_top_level_expr_name)
        .map(str::to_string)
        .collect::<Vec<_>>();

    let mut reachable = HashSet::<String>::new();
    let mut queue = VecDeque::from(reachable_seed);
    while let Some(name) = queue.pop_front() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        if let Some(expr) = def_map.get(&name) {
            for reference in deep_referenced_vars(expr) {
                if def_map.contains_key(reference) && !reachable.contains(reference) {
                    queue.push_back(reference.to_string());
                }
            }
        }
    }

    exprs
        .iter()
        .filter(|expr| {
            deep_named_decl_name(expr)
                .map(|name| reachable.contains(name))
                .unwrap_or(true)
        })
        .cloned()
        .collect()
}

fn deep_referenced_vars(expr: &DeepExpr) -> Vec<&str> {
    let mut out = Vec::new();
    collect_deep_referenced_vars(expr, &mut out);
    out
}

fn collect_deep_referenced_vars<'a>(expr: &'a DeepExpr, out: &mut Vec<&'a str>) {
    match expr {
        DeepExpr::Atom(_, _) => {}
        DeepExpr::MetaExpr(meta, _) => collect_deep_referenced_vars(&meta.expr, out),
        DeepExpr::Map(map, _) => {
            for (_, value) in &map.entries {
                collect_deep_referenced_vars(value, out);
            }
        }
        DeepExpr::List(list, _) => {
            if let (
                Some(DeepExpr::Atom(DeepAtom::Symbol(tag), _)),
                Some(DeepExpr::Atom(DeepAtom::Symbol(name), _)),
            ) = (list.elements.first(), list.elements.get(2))
                && tag == "var"
            {
                out.push(name.as_str());
            }
            for child in &list.elements {
                collect_deep_referenced_vars(child, out);
            }
        }
    }
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
    matches!(
        list.elements.first(),
        Some(DeepExpr::Atom(DeepAtom::Symbol(tag), _)) if tag == "def"
    )
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
        && let Some(DeepExpr::Atom(DeepAtom::Symbol(tag), _)) = list.elements.first()
    {
        if tag == "t-fn" {
            extend_root_names_from_value(name, list.elements.last(), None, out);
            return;
        }
        if tag == "t-tuple" {
            for (index, child) in list.elements.iter().skip(2).enumerate() {
                extend_root_names_from_value(&format!("{name}.{index}"), Some(child), None, out);
            }
            return;
        }
    }
    if let Some(DeepExpr::List(list, _)) = value
        && matches!(
            list.elements.first(),
            Some(DeepExpr::Atom(DeepAtom::Symbol(tag), _)) if tag == "tuple"
        )
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
    matches!(expr, DeepExpr::List(list, _) if matches!(list.elements.first(), Some(DeepExpr::Atom(DeepAtom::Symbol(tag), _)) if tag == "t-fn"))
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
            if let Some(chelis_deep::ast::Expr::Atom(chelis_deep::ast::Atom::Symbol(tag), _)) =
                list.elements.first()
                && tag == "d-name"
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

/// `chelis lint` — naming-convention lint per `spec/01-nomenclature.md`.
///
/// Walks each path in `paths` (default: `.`), dispatches to every registered
/// rule, prints violations as `path:line:col: rule (§ref): message`, and
/// returns an exit code. With `--check`, exit nonzero on any violation
/// (CI gate); otherwise exit 0 even when violations exist (informational).
fn cmd_lint(
    paths: Vec<PathBuf>,
    check: bool,
    rule_filter: Option<&str>,
) -> Result<i32, Box<dyn std::error::Error>> {
    let targets = if paths.is_empty() {
        vec![PathBuf::from(".")]
    } else {
        paths
    };
    let mut rules = chelis_lint::registry::all_rules();
    if let Some(id) = rule_filter {
        rules.retain(|r| r.id() == id);
        if rules.is_empty() {
            return Err(format!("no rule with id '{id}'").into());
        }
    }
    // The exception list is sourced from `style_gate::exceptions()` so
    // the standalone `chelis lint` subcommand and the build-time style
    // gate filter against one shared registry. Rule-internal allowlists
    // (e.g., `module_pascal_components::KNOWN_SINGLE_WORDS`) cover the
    // common naming carve-outs; path-glob entries with §-cross-refs go
    // here.
    let exceptions: Vec<chelis_lint::Exception> = style_gate::exceptions();
    let mut total = 0usize;
    for target in &targets {
        let raw_violations = chelis_lint::lint(target, &rules)?;
        let kept = chelis_lint::exceptions::apply_exceptions(&raw_violations, &exceptions, target);
        for v in &kept {
            println!("{v}");
        }
        total += kept.len();
    }
    if check && total > 0 { Ok(1) } else { Ok(0) }
}
