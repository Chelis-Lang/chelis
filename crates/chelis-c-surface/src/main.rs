use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

use chelis_c_surface::{
    CarrierUse, collect_c_aliases, scan_c_source_with_aliases, scan_rust_source,
};
use serde::Serialize;

#[derive(Serialize)]
struct LocatedCarrierUse {
    path: String,
    kind: String,
    owner: String,
    signature: String,
}

fn usage() -> ! {
    eprintln!("usage: chelis-c-surface --repo <repository-root>");
    std::process::exit(2);
}

fn parse_root() -> PathBuf {
    let mut arguments = env::args().skip(1);
    if arguments.next().as_deref() != Some("--repo") {
        usage();
    }
    let root = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| usage());
    if arguments.next().is_some() {
        usage();
    }
    root
}

fn validate_relative(path: &Path) -> Result<(), String> {
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "scan path must be a normalized relative path: {}",
            path.display()
        ));
    }
    Ok(())
}

fn located(path: &str, rows: Vec<CarrierUse>) -> impl Iterator<Item = LocatedCarrierUse> + '_ {
    rows.into_iter().map(move |row| LocatedCarrierUse {
        path: path.to_string(),
        kind: row.kind,
        owner: row.owner,
        signature: row.signature,
    })
}

fn run() -> Result<(), String> {
    let root = parse_root();
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("read path manifest: {error}"))?;
    let paths: Vec<String> =
        serde_json::from_str(&input).map_err(|error| format!("parse path manifest: {error}"))?;
    let mut sources = Vec::with_capacity(paths.len());
    for raw_path in paths {
        let relative = Path::new(&raw_path);
        validate_relative(relative)?;
        let source = fs::read_to_string(root.join(relative))
            .map_err(|error| format!("read `{raw_path}`: {error}"))?;
        sources.push((raw_path, source));
    }
    let c_prelude = sources
        .iter()
        .filter(|(path, _)| {
            Path::new(path)
                .extension()
                .is_none_or(|extension| extension != "rs")
        })
        .map(|(_, source)| source.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    let aliases = collect_c_aliases(&c_prelude);
    let mut output = Vec::new();
    for (raw_path, source) in sources {
        let relative = Path::new(&raw_path);
        let rows = if relative
            .extension()
            .is_some_and(|extension| extension == "rs")
        {
            scan_rust_source(&source)
        } else {
            scan_c_source_with_aliases(&source, "c-source", &aliases)
        }
        .map_err(|error| format!("{raw_path}: {error}"))?;
        output.extend(located(&raw_path, rows));
    }
    serde_json::to_writer(io::stdout(), &output)
        .map_err(|error| format!("write scan result: {error}"))?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("C SURFACE SCAN: FAIL: {error}");
        std::process::exit(1);
    }
}
