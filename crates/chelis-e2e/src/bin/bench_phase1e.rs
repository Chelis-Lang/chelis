use std::env;
use std::path::PathBuf;

use chelis_e2e::bench::{BenchmarkReport, run_phase1e};
use serde_json::to_string_pretty;

struct Config {
    models: Vec<chelis_e2e::bench::Model>,
    emit_json: Option<PathBuf>,
}

fn parse_args() -> Result<Config, String> {
    let mut model_arg = "all".to_string();
    let mut emit_json = None;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                model_arg = args
                    .next()
                    .ok_or_else(|| "missing value for `--model`".to_string())?;
            }
            "--emit-json" => {
                emit_json =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        "missing value for `--emit-json`".to_string()
                    })?));
            }
            other => return Err(format!("unknown arg `{other}`")),
        }
    }

    Ok(Config {
        models: chelis_e2e::bench::Model::parse(&model_arg)?,
        emit_json,
    })
}

fn main() -> Result<(), String> {
    let config = parse_args()?;
    let report = run_phase1e(&config.models, config.emit_json.as_deref())?;
    emit_report(&report, config.emit_json.as_deref())
}

fn emit_report(report: &BenchmarkReport, path: Option<&std::path::Path>) -> Result<(), String> {
    let json = to_string_pretty(report).map_err(|e| format!("serialize report failed: {e}"))?;
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create output directory failed: {e}"))?;
        }
        std::fs::write(path, &json).map_err(|e| format!("write report failed: {e}"))?;
    } else {
        println!("{json}");
    }
    Ok(())
}
