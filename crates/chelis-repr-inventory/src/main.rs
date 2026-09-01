//! Emit the chelis#893 Phase 0 seam inventory for a registered source list.
//!
//! The oracle owns the frozen source list and passes it on stdin as JSON; this
//! binary owns the structural derivation. Splitting it that way keeps one
//! parser per language: Rust is read with `syn`, and the four plain C headers
//! are read with the capacity census's own token vocabulary.

use std::env;
use std::fs;
use std::io::{self, Read};
use std::path::{Component, Path, PathBuf};

use chelis_repr_inventory::{SeamRow, scan_c_header, scan_rust_source};
use serde::Serialize;

#[derive(Serialize)]
struct LocatedSeamRow {
    path: String,
    kind: String,
    owner: String,
    sample: String,
}

#[derive(Serialize)]
struct ScanOutput {
    rows: Vec<LocatedSeamRow>,
}

fn usage() -> ! {
    eprintln!("usage: chelis-repr-inventory --repo <repository-root>");
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

/// A traversal or absolute path would let the manifest read outside the
/// repository, so it is rejected rather than normalized.
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

fn located(path: &str, rows: Vec<SeamRow>) -> impl Iterator<Item = LocatedSeamRow> + '_ {
    rows.into_iter().map(move |row| LocatedSeamRow {
        path: path.to_string(),
        kind: row.kind,
        owner: row.owner,
        sample: row.sample,
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
    let mut output = Vec::new();
    for raw_path in paths {
        let relative = Path::new(&raw_path);
        validate_relative(relative)?;
        let source = fs::read_to_string(root.join(relative))
            .map_err(|error| format!("read `{raw_path}`: {error}"))?;
        let rows = match relative
            .extension()
            .and_then(|extension| extension.to_str())
        {
            Some("rs") => scan_rust_source(&raw_path, &source),
            Some("h") => scan_c_header(&raw_path, &source),
            other => {
                return Err(format!(
                    "`{raw_path}` has unsupported extension {other:?}; the inventory reads \
                     Rust and plain C headers only"
                ));
            }
        }
        .map_err(|error| format!("{error}"))?;
        output.extend(located(&raw_path, rows));
    }
    serde_json::to_writer(io::stdout(), &ScanOutput { rows: output })
        .map_err(|error| format!("write scan result: {error}"))?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("REPR INVENTORY SCAN: FAIL: {error}");
        std::process::exit(1);
    }
}
