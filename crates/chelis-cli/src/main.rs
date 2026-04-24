//! Chelis compiler CLI.

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

fn copy_runtime_artifacts(
    runtime_dir: &Path,
    include_hip_runtime: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    fs::write(runtime_dir.join("chelis_runtime.h"), RUNTIME_H)?;
    fs::write(runtime_dir.join("chelis_blas.h"), BLAS_H)?;
    fs::write(runtime_dir.join("chelis_simd.h"), SIMD_H)?;
    fs::write(runtime_dir.join("chelis_math.h"), MATH_H)?;
    if include_hip_runtime {
        fs::write(runtime_dir.join("chelis_hip_runtime.h"), HIP_RUNTIME_H)?;
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
    Eval {
        /// File to evaluate
        #[arg(long)]
        file: Option<PathBuf>,
        /// Inline expression
        expr: Option<String>,
    },
    /// Run front-end checks and report fitness-oriented diagnostics
    Check { file: PathBuf },
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
    },
    /// Compile to C (default) or HIP GPU code
    Build {
        file: PathBuf,
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Backend target: "c" (default) or "hip" (GPU)
        #[arg(long, default_value = "c")]
        target: String,
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
    /// Build package artifacts (.chb + .tar.zst)
    Build { path: Option<PathBuf> },
    /// Publish a package into the local Reef registry
    Publish { path: Option<PathBuf> },
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
        Some(Command::Eval { file, expr }) => cmd_eval(file.as_deref(), expr.as_deref()),
        Some(Command::Check { file }) => cmd_check(&file),
        Some(Command::Validate {
            surf,
            deep,
            desugar,
            file,
        }) => cmd_validate(&file, surf, deep, desugar),
        Some(Command::Build {
            file,
            output,
            target,
        }) => cmd_build(&file, output.as_deref(), &target),
        Some(Command::Reef { command }) => cmd_reef(command),
        Some(Command::Tide { command }) => run_tide(command),
        Some(Command::Cove { file }) => cmd_cove(file),
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
) -> Result<(), Box<dyn std::error::Error>> {
    let (source_kind, source, selected_roots) = match (file, expr) {
        (Some(path), _) => {
            let (decls, entry_decls) = load_eval_decls(path)?;
            let deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
            let checked = checked_program_with_effects(&deep_exprs).map_err(boxed_string_error)?;
            (
                SourceKind::Surf,
                chelis_surf::format::format_program(&decls),
                Some(root_names_from_decls(&entry_decls, checked.type_env())),
            )
        }
        (None, Some(e)) => (SourceKind::Surf, format!("__eval_result = {e}"), None),
        (None, None) => {
            return Err("provide --file or an expression".into());
        }
    };
    match try_eval(source_kind, &source, selected_roots.as_deref()) {
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

fn cmd_check(file: &Path) -> Result<(), Box<dyn std::error::Error>> {
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
    println!("{json}");
    Ok(())
}

/// Recursively walks a Surf `Expr` looking for any `with seed(...) { ... }`
/// form. The C and HIP backends do not currently plumb the user-provided
/// seed into the emitted runtime, so any program containing `with seed`
/// must be rejected with a hard error rather than silently dropped.
fn expr_contains_with_seed(expr: &chelis_surf::ast::Expr) -> bool {
    use chelis_surf::ast::Expr;
    match expr {
        Expr::Lit(_, _) | Expr::Var(_, _) | Expr::Constructor(_, _) => false,
        Expr::Apply(f, args, _) => {
            expr_contains_with_seed(f) || args.iter().any(expr_contains_with_seed)
        }
        Expr::List(items, _) | Expr::Tuple(items, _) | Expr::Par(items, _) => {
            items.iter().any(expr_contains_with_seed)
        }
        Expr::Record(_, fields, _) => fields.iter().any(|(_, e)| expr_contains_with_seed(e)),
        Expr::Access(e, _, _)
        | Expr::TupleGet(e, _, _)
        | Expr::Unary(_, e, _)
        | Expr::Cast(e, _, _)
        | Expr::Grad(e, _, _)
        | Expr::Vmap(e, _, _)
        | Expr::Jit(e, _)
        | Expr::Realize(e, _)
        | Expr::Copy(e, _)
        | Expr::Borrow(e, _)
        | Expr::Annotate(e, _, _) => expr_contains_with_seed(e),
        Expr::Binary(_, l, r, _) => expr_contains_with_seed(l) || expr_contains_with_seed(r),
        Expr::Pipe(head, tail, _) => {
            expr_contains_with_seed(head) || tail.iter().any(expr_contains_with_seed)
        }
        Expr::If(c, t, e, _) => {
            expr_contains_with_seed(c) || expr_contains_with_seed(t) || expr_contains_with_seed(e)
        }
        Expr::Match(scrut, arms, _) => {
            expr_contains_with_seed(scrut)
                || arms.iter().any(|arm| expr_contains_with_seed(&arm.body))
        }
        Expr::Lambda(_, body, _) => expr_contains_with_seed(body),
        Expr::WithSeed(_, _, _) => true,
        Expr::WithDevice(_, body, _) => expr_contains_with_seed(body),
        Expr::Block(bindings, tail, _) => {
            bindings.iter().any(|b| expr_contains_with_seed(&b.value))
                || expr_contains_with_seed(tail)
        }
    }
}

fn decls_contain_with_seed(decls: &[Decl]) -> bool {
    decls.iter().any(|decl| match decl {
        Decl::Module { decls, .. } => decls_contain_with_seed(decls),
        Decl::FunDef { body, .. } => expr_contains_with_seed(body),
        Decl::LetDef { value, .. } => expr_contains_with_seed(value),
        Decl::MacroDef { body, .. } => expr_contains_with_seed(body),
        Decl::Import { .. }
        | Decl::Sig { .. }
        | Decl::Dim { .. }
        | Decl::TypeDef { .. }
        | Decl::TypeAlias { .. }
        | Decl::Export { .. } => false,
    })
}

fn reject_with_seed_for_build_target(
    decls: &[Decl],
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if decls_contain_with_seed(decls) {
        return Err(format!(
            "`chelis build --target {target}` does not yet plumb `with seed(...)` into the generated runtime; \
             rejecting rather than silently dropping the seed. Track at \
             spec/design/chelis_phase3_plan.md §3j-pre Acknowledged Limitations \
             (Batch 7b: C/HIP backends drop `with seed`). Run the seeded program through `chelis eval` instead."
        )
        .into());
    }
    Ok(())
}

fn cmd_build(
    file: &std::path::Path,
    output: Option<&std::path::Path>,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
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
                    _ => host_display_root_name(&binding.name, &entry_display_root_names),
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
        other => Err(format!("unknown target '{other}': expected 'c' or 'hip'").into()),
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
        ReefCommand::Build { path } => {
            let root = path.unwrap_or_else(|| PathBuf::from("."));
            let artifacts = chelis_reef::build_package(&root)?;
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
    }
    Ok(())
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
fn cmd_test(
    path: Option<&Path>,
    filter: Option<&str>,
    json: bool,
    timeout_secs: u64,
) -> Result<i32, String> {
    let cwd = env::current_dir().map_err(|e| format!("failed to read cwd: {e}"))?;
    let target = match path {
        Some(p) => p.to_path_buf(),
        None => cwd.join("tests"),
    };

    if !target.exists() {
        return Err(format!(
            "path `{}` does not exist — pass a tests directory or a single .ch file",
            target.display()
        ));
    }

    // Prepare the reef graph once — this is the expensive step we share
    // across every test file in a single invocation.
    let graph = chelis_reef::prepare_reef_graph(&cwd)?;

    let test_files = discover_test_files(&target)?;
    if test_files.is_empty() {
        return Err(format!(
            "no .ch files found under `{}` — `chelis test` requires at least one test file to run",
            target.display()
        ));
    }

    let timeout = Duration::from_secs(timeout_secs.max(1));
    let mut passed: usize = 0;
    let mut failed: usize = 0;
    let stdout = io::stdout();
    let mut out = stdout.lock();

    for file in &test_files {
        let rel_display = file
            .strip_prefix(&cwd)
            .unwrap_or(file.as_path())
            .display()
            .to_string();

        let file_result = run_test_file(&graph, file, filter, &rel_display, timeout);
        let rows = match file_result {
            Ok(rows) => rows,
            Err(err) => {
                // File-level failure (parse error or compile error with no
                // matching tests). Record as a synthetic row so the operator
                // sees it, and count as failed.
                vec![TestRow {
                    file: rel_display.clone(),
                    test: "<file>".to_string(),
                    status: TestStatus::Fail,
                    message: Some(err),
                }]
            }
        };

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
    for entry in walkdir::WalkDir::new(target).sort_by_file_name() {
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

/// Execute every `test_*` function in `file` and return one `TestRow` per
/// selected test. Returns `Err` only when the file itself cannot be read
/// or parsed — compile/runtime failures surface as per-row `FAIL` entries.
fn run_test_file(
    graph: &chelis_reef::PreparedReefGraph,
    file: &Path,
    filter: Option<&str>,
    rel_display: &str,
    timeout: Duration,
) -> Result<Vec<TestRow>, String> {
    let source = fs::read_to_string(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    let parsed = chelis_surf::parser::parse_str(&source)
        .map_err(|e| format!("parse {}: {e}", file.display()))?;

    // A test file may wrap its contents in `module Foo.Bar` — we need the
    // flat decl list so `compile_with_reef_graph` can treat it as an eval
    // module. Imports are preserved so the rewriter can resolve references
    // to `Std.*` or sibling modules.
    let flat_decls = flatten_module_decls(&parsed);

    let matched_tests = enumerate_test_fns(&flat_decls, filter, rel_display);
    if matched_tests.is_empty() {
        return Ok(Vec::new());
    }

    // Per-test isolation. We cannot bundle every test as a separate top-level
    // `let __chelis_test_N = test_N()` because `evaluate_host_program` walks
    // all non-fn top-level defs unconditionally — a single failing module
    // binding would then cascade into every test in the file. Instead we
    // rebuild the program once per selected test with a single caller
    // binding, share the reef graph, and wrap each evaluation in
    // `run_with_timeout` to honor `--timeout` per test.
    let mut rows = Vec::with_capacity(matched_tests.len());
    for test in &matched_tests {
        let synth_name = "__chelis_test_root".to_string();
        let call = chelis_surf::ast::Expr::Apply(
            Box::new(chelis_surf::ast::Expr::Var(test.name.clone(), test.span)),
            Vec::new(),
            test.span,
        );
        let mut synth_decls = flat_decls.clone();
        synth_decls.push(Decl::LetDef {
            name: synth_name.clone(),
            ty: None,
            value: call,
            span: test.span,
        });

        let prepared = match chelis_reef::compile_with_reef_graph(graph, &synth_decls) {
            Ok(p) => p,
            Err(err) => {
                rows.push(TestRow {
                    file: rel_display.to_string(),
                    test: test.name.clone(),
                    status: TestStatus::Fail,
                    message: Some(format!("compile: {err}")),
                });
                continue;
            }
        };

        let source_text = chelis_surf::format::format_program(&prepared.decls);
        let root = synth_name.clone();
        let outcome = chelis_reef::run_with_timeout(
            move || {
                let request = EvalRequest {
                    source_kind: SourceKind::Surf,
                    source: source_text,
                    bindings: BTreeMap::new(),
                };
                let results = chelis_compiler_api::compiler::eval_many(request, &[root]);
                Ok(results)
            },
            timeout,
            &format!("timeout after {}s", timeout.as_secs()),
        );

        let (status, message) = match outcome {
            Err(msg) => (TestStatus::Fail, Some(msg)),
            Ok(mut results) => {
                let (_name, result) = results.pop().ok_or_else(|| {
                    format!("internal: eval_many returned no rows for `{}`", test.name)
                })?;
                match result {
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
                }
            }
        };

        rows.push(TestRow {
            file: rel_display.to_string(),
            test: test.name.clone(),
            status,
            message,
        });
    }
    Ok(rows)
}

#[derive(Debug, Clone)]
struct DiscoveredTest {
    name: String,
    span: chelis_deep::Span,
}

fn enumerate_test_fns(
    decls: &[Decl],
    filter: Option<&str>,
    rel_display: &str,
) -> Vec<DiscoveredTest> {
    let mut out = Vec::new();
    for decl in decls {
        if let Decl::FunDef {
            name, params, span, ..
        } = decl
            && name.starts_with("test_")
            && params.is_empty()
        {
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
    }
    out
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
) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
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
    copy_runtime_artifacts(runtime_dir, false)?;

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
    copy_runtime_artifacts(runtime_dir, true)?;

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
    copy_runtime_artifacts(runtime_dir, true)?;

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

    let mut lines = result.transcript;
    if result.roots.len() == 1 {
        if let Some(root) = result.roots.first() {
            lines.push(format_execution_value(&root.value));
        }
        return Ok(lines.join("\n"));
    }

    lines.extend(result.roots.iter().enumerate().map(|(index, root)| {
        let name = root.name.clone().unwrap_or_else(|| format!("_{index}"));
        format!(
            "{} = {}",
            display_root_name(&name),
            format_execution_value(&root.value)
        )
    }));
    Ok(lines.join("\n"))
}

fn format_execution_value(value: &ExecutionValue) -> String {
    match value {
        ExecutionValue::Tensor { value } => format!(
            "tensor(shape={:?}, data={:?})",
            value.shape,
            &value.data[..value.data.len().min(10)]
        ),
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
