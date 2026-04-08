//! Chelis compiler CLI.

use clap::{ArgAction, ArgGroup, Parser, Subcommand};
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::net::{IpAddr, SocketAddr};
use std::path::PathBuf;

const RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-c/runtime/chelis_runtime.h"
));
const RUNTIME_C: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-c/runtime/chelis_runtime.c"
));
const HIP_RUNTIME_H: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../chelis-backend-hip/runtime/chelis_hip_runtime.h"
));

#[derive(Parser)]
#[command(
    name = "chelis",
    version = "0.1.0",
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

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Command::Deep { file, flat }) => cmd_deep(&file, flat),
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
        Some(Command::Tide { command }) => run_tide(command),
        Some(Command::Cove { file }) => cmd_cove(file),
        None => {
            println!("chelis 0.1.0 -- use --help for commands");
            Ok(())
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn cmd_deep(file: &PathBuf, flat: bool) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    let decls = chelis_surf::parser::parse_str(&source)?;
    let deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
    let output = if flat {
        chelis_deep::printer::print_canonical_flat(&deep_exprs)
    } else {
        chelis_deep::printer::print_canonical(&deep_exprs)
    };
    print!("{output}");
    Ok(())
}

fn cmd_surf(file: &PathBuf, verbose: bool) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    let options = if verbose {
        chelis_surf::decompile::DecompileOptions::verbose()
    } else {
        chelis_surf::decompile::DecompileOptions::idiomatic()
    };
    let synthetic_name = file.file_stem().and_then(|stem| stem.to_str());
    if ext == "dp" {
        let deep_exprs = chelis_deep::parser::parse_str(&source)?;
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

fn cmd_fmt(file: &PathBuf, inplace: bool, check: bool) -> Result<(), Box<dyn std::error::Error>> {
    if inplace && check {
        return Err("`chelis fmt` does not allow `--inplace` and `--check` together".into());
    }
    let source = fs::read_to_string(file)?;
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    let output = if ext == "dp" {
        let deep_exprs = chelis_deep::parser::parse_str(&source)?;
        chelis_deep::printer::print_canonical(&deep_exprs)
    } else {
        // .ch: parse Surf -> desugar -> decompile back to Surf (idempotent)
        let decls = chelis_surf::parser::parse_str(&source)?;
        let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
        chelis_surf::decompile::decompile_program_with_context(
            &deep_exprs,
            &chelis_surf::decompile::DecompileOptions::idiomatic(),
            file.file_stem().and_then(|stem| stem.to_str()),
        )
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
    let source = match (file, expr) {
        (Some(path), _) => fs::read_to_string(path)?,
        (None, Some(e)) => format!("let __eval_result = {e}"),
        (None, None) => {
            return Err("provide --file or an expression".into());
        }
    };
    match try_eval(&source) {
        Ok(result) => {
            println!("{result}");
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}

fn cmd_check(file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    let decls = chelis_surf::parser::parse_str(&source)?;
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

fn cmd_build(
    file: &std::path::Path,
    output: Option<&std::path::Path>,
    target: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    let decls = chelis_surf::parser::parse_str(&source)?;
    let deep_exprs = expanded_desugared_program(&decls).map_err(boxed_string_error)?;
    let symbolic_dims = collect_symbolic_dims_from_deep(&deep_exprs);
    let checked =
        checked_program_with_effects(&deep_exprs).map_err(|e| format!("Check errors: {e}"))?;
    chelis_effects::validate_build_target(&checked, target)
        .map_err(|errors| format_effect_errors(&errors))?;
    let dag = chelis_ir::lower::lower_program(&checked);
    let func_name = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("chelis_main");

    match target {
        "c" => {
            reject_unsupported_effect_ops(&dag, "c")?;
            let fused = chelis_ir::fuse::fuse(&dag);
            cmd_build_c(&fused, func_name, file, output, &symbolic_dims)
        }
        "hip" => {
            reject_unsupported_effect_ops(&dag, "hip")?;
            reject_unsupported_hip_ops(&dag)?;
            // Current `chelis build` path lowers a forward DAG and then fuses before HIP emission.
            // When grad participates in a GPU compilation pipeline, the intended ordering is:
            // lower -> optimize -> grad -> optimize -> fuse -> codegen.
            let fused = chelis_ir::fuse::fuse(&dag);
            cmd_build_hip(&fused, func_name, file, output, &symbolic_dims)
        }
        other => Err(format!("unknown target '{other}': expected 'c' or 'hip'").into()),
    }
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
    file: &PathBuf,
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
    let result = chelis_backend_c::codegen(dag, func_name);

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
    fs::write(runtime_dir.join("chelis_runtime.h"), RUNTIME_H)?;
    fs::write(runtime_dir.join("chelis_runtime.c"), RUNTIME_C)?;

    println!("Wrote {} and {}", c_path.display(), h_path.display());
    println!(
        "Wrote {} and {}",
        runtime_dir.join("chelis_runtime.h").display(),
        runtime_dir.join("chelis_runtime.c").display()
    );
    let symbolic_dims = fallback_symbolic_dims(dag, &result.symbolic_dims, symbolic_dims_hint);
    if !symbolic_dims.is_empty() {
        println!("Symbolic dims: {}", symbolic_dims.join(", "));
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
        "Compile: gcc -O2 {} {} chelis_runtime.c -o {}",
        flags.join(" "),
        c_path.display(),
        c_path.with_extension("").display()
    );
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
    fs::write(runtime_dir.join("chelis_runtime.h"), RUNTIME_H)?;
    fs::write(runtime_dir.join("chelis_runtime.c"), RUNTIME_C)?;
    fs::write(runtime_dir.join("chelis_hip_runtime.h"), HIP_RUNTIME_H)?;

    println!("Wrote {} and {}", c_path.display(), h_path.display());
    println!(
        "Wrote runtime: chelis_runtime.{{h,c}}, chelis_hip_runtime.h in {}",
        runtime_dir.display()
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
        "Compile: hipcc {} {} {} -o {}",
        flags.join(" "),
        c_path.display(),
        runtime_dir.join("chelis_runtime.c").display(),
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
            || trimmed.starts_with("let ")
        {
            accumulated_source.push_str(trimmed);
            accumulated_source.push('\n');
            println!("  defined.");
        } else {
            let eval_source = format!("{}\nlet __tide_result = {}", accumulated_source, trimmed);
            match try_eval(&eval_source) {
                Ok(result) => println!("= {result}"),
                Err(e) => eprintln!("error: {e}"),
            }
        }
    }
    Ok(())
}

fn try_eval(source: &str) -> Result<String, String> {
    let decls = chelis_surf::parser::parse_str(source).map_err(|e| format!("{e}"))?;
    let deep = expanded_desugared_program(&decls)?;
    let checked = checked_program_with_effects(&deep)?;
    let dag = chelis_ir::lower::lower_program(&checked);

    if dag.is_empty() {
        return Err("empty program".into());
    }

    let inputs: HashMap<String, chelis_ir::eval::TensorValue> = HashMap::new();
    let vals = chelis_ir::eval::eval_tensor_with_strict(&dag, |name| inputs.get(name).cloned())
        .map_err(|e| e.to_string())?;

    // Get the last root's value, or the last node's value
    let roots = dag.roots();
    let target_id = if roots.is_empty() {
        chelis_ir::dag::NodeId(dag.len() - 1)
    } else {
        *roots.last().unwrap()
    };
    match vals.get(&target_id) {
        Some(tv) => {
            if tv.shape.is_empty() || tv.data.len() == 1 {
                Ok(format!("{}", tv.data[0]))
            } else {
                Ok(format!(
                    "tensor(shape={:?}, data={:?})",
                    tv.shape,
                    &tv.data[..tv.data.len().min(10)]
                ))
            }
        }
        None => Err("no result".into()),
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
