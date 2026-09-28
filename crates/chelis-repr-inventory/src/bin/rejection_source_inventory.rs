use std::env;
use std::io::{self, Read};
use std::path::PathBuf;

use chelis_repr_inventory::discover_production_rust_sources;

fn usage() -> ! {
    eprintln!("usage: rejection_source_inventory --repo <repository-root>");
    std::process::exit(2);
}

fn run() -> Result<(), String> {
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

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|error| format!("read production target roots: {error}"))?;
    let roots: Vec<String> = serde_json::from_str(&input)
        .map_err(|error| format!("parse production target roots: {error}"))?;
    let sources =
        discover_production_rust_sources(&root, &roots).map_err(|error| error.to_string())?;
    serde_json::to_writer(io::stdout(), &sources)
        .map_err(|error| format!("write production source inventory: {error}"))?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("REJECTION SOURCE INVENTORY: FAIL: {error}");
        std::process::exit(1);
    }
}
