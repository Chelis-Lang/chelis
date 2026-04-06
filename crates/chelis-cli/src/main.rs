//! Chelis compiler CLI.

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

const RUNTIME_H: &str = include_str!("../../chelis-backend-c/runtime/chelis_runtime.h");
const RUNTIME_C: &str = include_str!("../../chelis-backend-c/runtime/chelis_runtime.c");
const HIP_RUNTIME_H: &str = include_str!("../../chelis-backend-hip/runtime/chelis_hip_runtime.h");

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
    Deep { file: PathBuf },
    /// Decompile Deep (.dp) to Surf (best-effort)
    Surf { file: PathBuf },
    /// Format source code (canonical form)
    Fmt {
        file: PathBuf,
        #[arg(long)]
        inplace: bool,
    },
    /// Evaluate an expression or file
    Eval {
        /// File to evaluate
        #[arg(long)]
        file: Option<PathBuf>,
        /// Inline expression
        expr: Option<String>,
    },
    /// Type-check and report fitness score
    Check { file: PathBuf },
    /// Compile to C (default) or HIP GPU code
    Build {
        file: PathBuf,
        #[arg(long, short)]
        output: Option<PathBuf>,
        /// Backend target: "c" (default) or "hip" (GPU)
        #[arg(long, default_value = "c")]
        target: String,
    },
    /// Interactive REPL
    Tide,
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Some(Command::Deep { file }) => cmd_deep(&file),
        Some(Command::Surf { file }) => cmd_surf(&file),
        Some(Command::Fmt { file, inplace }) => cmd_fmt(&file, inplace),
        Some(Command::Eval { file, expr }) => cmd_eval(file.as_deref(), expr.as_deref()),
        Some(Command::Check { file }) => cmd_check(&file),
        Some(Command::Build {
            file,
            output,
            target,
        }) => cmd_build(&file, output.as_deref(), &target),
        Some(Command::Tide) => run_tide(),
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

fn cmd_deep(file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    let decls = chelis_surf::parser::parse_str(&source)?;
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let output = chelis_deep::printer::print_canonical(&deep_exprs);
    print!("{output}");
    Ok(())
}

fn cmd_surf(file: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    if ext == "dp" {
        let deep_exprs = chelis_deep::parser::parse_str(&source)?;
        let surf = chelis_surf::decompile::decompile_program(&deep_exprs);
        print!("{surf}");
    } else {
        // For .ch files, round-trip through deep and back
        let decls = chelis_surf::parser::parse_str(&source)?;
        let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
        let surf = chelis_surf::decompile::decompile_program(&deep_exprs);
        print!("{surf}");
    }
    Ok(())
}

fn cmd_fmt(file: &PathBuf, inplace: bool) -> Result<(), Box<dyn std::error::Error>> {
    let source = fs::read_to_string(file)?;
    let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
    let output = if ext == "dp" {
        let deep_exprs = chelis_deep::parser::parse_str(&source)?;
        chelis_deep::printer::print_canonical(&deep_exprs)
    } else {
        // .ch: parse Surf -> desugar -> decompile back to Surf (idempotent)
        let decls = chelis_surf::parser::parse_str(&source)?;
        let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
        chelis_surf::decompile::decompile_program(&deep_exprs)
    };
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
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let report = chelis_types::check_phase0e_fitness(&deep_exprs);
    // Format as JSON manually
    let errors_json: Vec<String> = report
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
    let deep_exprs = chelis_surf::desugar::desugar_program(&decls);
    let checked = chelis_types::check_phase0e_program(&deep_exprs)
        .map_err(|r| format!("Type errors: {:?}", r.errors))?;
    let dag = chelis_ir::lower::lower_program(&checked);
    let func_name = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("chelis_main");

    match target {
        "c" => cmd_build_c(&dag, func_name, file, output),
        "hip" => cmd_build_hip(&dag, func_name, file, output),
        other => Err(format!("unknown target '{other}': expected 'c' or 'hip'").into()),
    }
}

fn cmd_build_c(
    dag: &chelis_ir::dag::Dag,
    func_name: &str,
    _file: &std::path::Path,
    output: Option<&std::path::Path>,
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

fn run_tide() -> Result<(), Box<dyn std::error::Error>> {
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
    let deep = chelis_surf::desugar::desugar_program(&decls);
    let checked = chelis_types::check_phase0e_program(&deep)
        .map_err(|r| format!("Type errors: {:?}", r.errors))?;
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
