use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

use chelis_c_surface::{
    CarrierUse, collect_c_aliases, production_rust_source, scan_c_source_at_path, scan_rust_source,
};
use serde::Serialize;

#[derive(Serialize)]
struct LocatedCarrierUse {
    path: String,
    kind: String,
    owner: String,
    signature: String,
}

#[derive(Serialize)]
struct ScanOutput {
    rows: Vec<LocatedCarrierUse>,
    production_rust_sources: BTreeMap<String, String>,
}

const REQUIRED_LIBCLANG_MAJOR: u32 = 18;

fn libclang_major(version: &str) -> Option<u32> {
    version
        .split_once("clang version ")
        .and_then(|(_, release)| {
            release
                .chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
                .parse()
                .ok()
        })
}

fn validate_libclang_version(version: &str) -> Result<(), String> {
    match libclang_major(version) {
        Some(REQUIRED_LIBCLANG_MAJOR) => Ok(()),
        _ => Err(format!(
            "runtime-representation inventory requires libclang major \
             {REQUIRED_LIBCLANG_MAJOR}; loaded {version:?}"
        )),
    }
}

fn validate_loaded_libclang() -> Result<(), String> {
    let _clang = clang::Clang::new()
        .map_err(|error| format!("cannot load libclang for version check: {error}"))?;
    validate_libclang_version(&clang::get_version())
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
    validate_loaded_libclang()?;
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
    let mut production_rust_sources = BTreeMap::new();
    for (raw_path, source) in sources {
        let relative = Path::new(&raw_path);
        let rows = if relative
            .extension()
            .is_some_and(|extension| extension == "rs")
        {
            production_rust_sources.insert(
                raw_path.clone(),
                production_rust_source(&source).map_err(|error| format!("{raw_path}: {error}"))?,
            );
            scan_rust_source(&source)
        } else {
            scan_c_source_at_path(&source, &raw_path, relative, &aliases)
        }
        .map_err(|error| format!("{raw_path}: {error}"))?;
        output.extend(located(&raw_path, rows));
    }
    serde_json::to_writer(
        io::stdout(),
        &ScanOutput {
            rows: output,
            production_rust_sources,
        },
    )
    .map_err(|error| format!("write scan result: {error}"))?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("C SURFACE SCAN: FAIL: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn libclang_major_parser_uses_the_clang_release_not_other_numbers() {
        assert_eq!(
            libclang_major("Debian 12 clang version 18.1.8 (release build)"),
            Some(18)
        );
        assert_eq!(
            libclang_major("Apple clang version 21.0.0 (clang-2100.0.0.1)"),
            Some(21)
        );
        assert_eq!(libclang_major("LLVM version 18.1.8"), None);
    }

    #[test]
    fn runtime_inventory_requires_libclang_major_18() {
        assert!(validate_libclang_version("clang version 18.1.8").is_ok());
        let error = validate_libclang_version("Apple clang version 21.0.0")
            .expect_err("an unpinned parser must be rejected");
        assert!(error.contains("requires libclang major 18"), "{error}");
        assert!(error.contains("Apple clang version 21.0.0"), "{error}");
    }
}
