use std::fs;
use std::path::Path;

const ROOT: &str = "__DAG_ROOT__";
const OWNERS: &[(&str, &str)] = &[
    ("dag", "Dag"),
    ("load_store_name", "LoadStoreName"),
    ("analysis", "analyze_copy_costs"),
    ("axis_sources", "claimed_producers"),
    ("verify", "verify"),
    ("tier2", "lower_relu"),
    ("tier2_ordered", "einsum_contraction_candidate"),
    ("grad", "GradResult"),
    ("fuse", "FuseResult"),
    ("optimize", "constant_fold"),
    ("span_merge", "append_span_to_node"),
];

#[derive(Debug, PartialEq, Eq)]
enum Rejection {
    MissingDestination(String),
    MissingDefinition(String),
    DuplicateOwner(String),
    InvalidSource(String),
}

struct Source<'a> {
    module: &'a str,
    definition: &'a str,
    old: Option<&'a str>,
    canonical: Option<&'a str>,
}

fn validate(sources: &[Source<'_>]) -> Result<(), Rejection> {
    for source in sources {
        let path = format!("crates/chelis-dag-core/src/{}.rs", source.module);
        let text = source.canonical.ok_or_else(|| Rejection::MissingDestination(path.clone()))?;
        let parsed = syn::parse_file(text).map_err(|_| Rejection::InvalidSource(path.clone()))?;
        let count = parsed.items.iter().filter(|item| match item {
            syn::Item::Struct(item) => item.ident == source.definition,
            syn::Item::Enum(item) => item.ident == source.definition,
            syn::Item::Fn(item) => item.sig.ident == source.definition,
            _ => false,
        }).count();
        if count != 1 {
            return Err(Rejection::MissingDefinition(path));
        }
        if source.old.is_some() {
            return Err(Rejection::DuplicateOwner(format!("crates/chelis-ir/src/{}.rs", source.module)));
        }
    }
    Ok(())
}

fn main() {
    let root = Path::new(ROOT);
    let storage = OWNERS.iter().map(|(module, _)| {
        let old = root.join(format!("crates/chelis-ir/src/{module}.rs"));
        let canonical = root.join(format!("crates/chelis-dag-core/src/{module}.rs"));
        let read = |path: &Path| match fs::read_to_string(path) {
            Ok(text) => Some(text),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => panic!("cannot read {}: {error}", path.display()),
        };
        (read(&old), read(&canonical))
    }).collect::<Vec<_>>();
    let mut sources = OWNERS.iter().zip(&storage).map(|((module, definition), (old, canonical))| Source {
        module, definition, old: old.as_deref(), canonical: canonical.as_deref(),
    }).collect::<Vec<_>>();
    validate(&sources).expect("one real canonical definition per relocated module");
    let mut missing = Vec::new();
    let mut empty = Vec::new();
    let mut duplicate = Vec::new();
    for index in 0..sources.len() {
        let module = sources[index].module;
        let path = format!("crates/chelis-dag-core/src/{module}.rs");
        let original = sources[index].canonical.take();
        assert_eq!(validate(&sources), Err(Rejection::MissingDestination(path.clone())));
        missing.push(module);
        sources[index].canonical = Some("");
        assert_eq!(validate(&sources), Err(Rejection::MissingDefinition(path)));
        empty.push(module);
        sources[index].canonical = original;
        sources[index].old = original;
        assert_eq!(validate(&sources), Err(Rejection::DuplicateOwner(format!("crates/chelis-ir/src/{module}.rs"))));
        duplicate.push(module);
        sources[index].old = None;
    }
    validate(&sources).unwrap();
    println!("DAG_SOURCE_OWNERS {}", serde_json::json!({
        "canonical": OWNERS.iter().map(|(module, _)| *module).collect::<Vec<_>>(),
        "missing": missing, "empty": empty, "duplicate": duplicate,
    }));
}
