use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr, List as DeepList, MetaMap};
#[cfg(not(feature = "chelis-prove"))]
use chelis_surf::ast::{BinOp, LetBinding, LetPattern};
use chelis_surf::ast::{Decl, Expr, Literal, Param, TypeExpr};
use serde_json::json;
use walkdir::WalkDir;

// Under `chelis-prove` the user-property running -- discovery, Tier B
// (SMT), Tier C (fuzz), and assumption injection -- is the shared
// `chelis_prove::property_runner` the tide MCP tool also drives (U4). The
// CLI's own Surf->SMT lowering and injection modules were retired in favour
// of that single runner; the no-`chelis-prove` build keeps only a local
// Tier-C-fuzz property path (no SMT, no injection -- that machinery lives
// behind the capability).
#[cfg(feature = "chelis-prove")]
mod obligation_run;
#[cfg(feature = "chelis-prove")]
mod property_run;

#[derive(Debug, Clone)]
pub struct ProveOptions<'a> {
    pub path: Option<&'a Path>,
    pub only: Option<&'a str>,
    pub samples: Option<usize>,
    pub seed: Option<u64>,
    pub max_attempts: Option<usize>,
    pub json: bool,
    pub spans: Option<&'a Path>,
    #[allow(dead_code)]
    pub tier: &'a str,
    #[allow(dead_code)]
    pub smt_timeout_ms: u64,
    /// Floor for invariant rejection-sampling acceptance rate before the
    /// starvation classifier fires (RFC D-STARVE). `0.0` disables the
    /// classifier and preserves the legacy exhaustion => Error path.
    #[allow(dead_code)]
    pub invariant_min_rate: f64,
    /// Explicit reef package root for import resolution.
    #[allow(dead_code)]
    pub package: Option<&'a Path>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    Passed,
    Failed,
    Unsupported,
    Error,
}

impl Status {
    fn exit_code(self) -> i32 {
        match self {
            Status::Passed => 0,
            Status::Failed => 1,
            Status::Unsupported => 2,
            Status::Error => 3,
        }
    }
}

impl ProveOptions<'_> {
    fn effective_seed(&self, property_seed: Option<u64>) -> u64 {
        self.seed.or(property_seed).unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
#[cfg(not(feature = "chelis-prove"))]
struct Property {
    name: String,
    source: PathBuf,
    params: Vec<Param>,
    preconditions: Vec<Expr>,
    body: Expr,
    samples: Option<usize>,
    seed: Option<u64>,
}

#[derive(Debug, Clone)]
struct Sample {
    values: Vec<SampleValue>,
}

#[derive(Debug, Clone)]
struct SampleValue {
    name: String,
    // Read only by the CLI-local Surf property runner (the no-`chelis-prove`
    // build); the shared runner under `chelis-prove` uses `deep_expr`.
    #[cfg_attr(feature = "chelis-prove", allow(dead_code))]
    surf_expr: Expr,
    deep_expr: DeepExpr,
    json: serde_json::Value,
    tensor_binding: Option<(String, TensorValue)>,
}

#[derive(Debug, Clone)]
struct DeepProperty {
    name: String,
    source: PathBuf,
    source_kind: String,
    source_id: Option<String>,
    params: Vec<Param>,
    preconditions: Vec<DeepExpr>,
    body: DeepExpr,
    samples: Option<usize>,
    seed: Option<u64>,
}

#[derive(Default)]
struct Summary {
    total: usize,
    passed: usize,
    failed: usize,
    unsupported: usize,
    errors: usize,
    /// Count of derived producer obligations run (RFC D-OBLIG: summary
    /// gains `obligations: N`). These are also counted in `total`.
    obligations: usize,
}

pub fn cmd_prove(options: ProveOptions<'_>) -> Result<i32, String> {
    let inputs = discover_inputs(options.path)?;
    if inputs.is_empty() {
        return Err("no .ch or .dp files selected for property discovery".to_string());
    }
    if options.spans.is_some() && !is_single_explicit_deep_input(options.path, &inputs) {
        return Err("--spans can only be used with a single explicit .dp input".to_string());
    }

    let mut totals = Summary::default();
    let mut worst = Status::Passed;
    let mut all_dependency_edges: Vec<serde_json::Value> = Vec::new();
    for input in &inputs {
        let status = match input.extension().and_then(|ext| ext.to_str()) {
            Some("ch") => prove_surf_file(input, &options, &mut totals)?,
            Some("dp") => prove_deep_file(input, &options, &mut totals)?,
            _ => Status::Passed,
        };
        worst = combine_status(worst, status);
        // chelis#490: collect dependency edges for each input.
        if options.json
            && let Ok(edges) = compute_dependency_edges(input)
        {
            for edge in edges {
                all_dependency_edges.push(json!({
                    "property": edge.0,
                    "references": edge.1,
                }));
            }
        }
    }

    if options.json {
        println!(
            "{}",
            json!({
                "kind": "summary",
                "total": totals.total,
                "passed": totals.passed,
                "failed": totals.failed,
                "unsupported": totals.unsupported,
                "errors": totals.errors,
                "obligations": totals.obligations,
                "dependency_edges": all_dependency_edges,
            })
        );
    } else {
        println!(
            "{} passed, {} failed, {} unsupported, {} errors",
            totals.passed, totals.failed, totals.unsupported, totals.errors
        );
    }
    Ok(worst.exit_code())
}

/// Compute property dependency edges for a single input file (chelis#490).
/// Returns (property_name, referenced_exports) pairs.
fn compute_dependency_edges(path: &Path) -> Result<Vec<(String, Vec<String>)>, String> {
    let source = fs::read_to_string(path).map_err(|e| e.to_string())?;
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("ch") => compute_surf_dependency_edges(&source),
        Some("dp") => compute_deep_dependency_edges(&source),
        _ => Ok(Vec::new()),
    }
}

fn compute_surf_dependency_edges(source: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    #[cfg(feature = "chelis-prove")]
    {
        let edges = chelis_prove::property_runner::property_dependency_edges(source)?;
        Ok(edges
            .into_iter()
            .map(|e| (e.property, e.references))
            .collect())
    }
    #[cfg(not(feature = "chelis-prove"))]
    {
        use std::collections::BTreeSet;
        let parsed = chelis_surf::parser::parse_str(source).map_err(|e| format!("parse: {e}"))?;
        let flat = flatten_module_decls(&parsed);
        let module_names: BTreeSet<String> = flat
            .iter()
            .filter_map(|d| match d {
                Decl::FunDef { name, .. } | Decl::LetDef { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        let mut edges = Vec::new();
        for decl in &flat {
            if let Decl::Property {
                name,
                params,
                preconditions,
                body,
                ..
            } = decl
            {
                let param_names: BTreeSet<&str> = params.iter().map(|p| p.name.as_str()).collect();
                let mut refs = BTreeSet::new();
                collect_surf_refs(body, &param_names, &module_names, &mut refs);
                for pre in preconditions {
                    collect_surf_refs(pre, &param_names, &module_names, &mut refs);
                }
                edges.push((name.clone(), refs.into_iter().collect()));
            }
        }
        Ok(edges)
    }
}

#[cfg(not(feature = "chelis-prove"))]
fn collect_surf_refs(
    expr: &Expr,
    params: &std::collections::BTreeSet<&str>,
    module_names: &std::collections::BTreeSet<String>,
    out: &mut std::collections::BTreeSet<String>,
) {
    match expr {
        Expr::Var(name, _) => {
            if !params.contains(name.as_str()) && module_names.contains(name) {
                out.insert(name.clone());
            }
        }
        Expr::Apply(callee, args, _) => {
            if let Expr::Var(name, _) = callee.as_ref() {
                if !params.contains(name.as_str()) && module_names.contains(name) {
                    out.insert(name.clone());
                }
            } else {
                collect_surf_refs(callee, params, module_names, out);
            }
            for arg in args {
                collect_surf_refs(arg, params, module_names, out);
            }
        }
        Expr::Binary(_, l, r, _) => {
            collect_surf_refs(l, params, module_names, out);
            collect_surf_refs(r, params, module_names, out);
        }
        Expr::Unary(_, e, _) => collect_surf_refs(e, params, module_names, out),
        Expr::If(c, t, f, _) => {
            collect_surf_refs(c, params, module_names, out);
            collect_surf_refs(t, params, module_names, out);
            collect_surf_refs(f, params, module_names, out);
        }
        Expr::Pipe(head, stages, _) => {
            collect_surf_refs(head, params, module_names, out);
            for s in stages {
                collect_surf_refs(s, params, module_names, out);
            }
        }
        Expr::Block(bindings, body, _) => {
            for b in bindings {
                collect_surf_refs(&b.value, params, module_names, out);
            }
            collect_surf_refs(body, params, module_names, out);
        }
        Expr::Lambda(_, body, _) => collect_surf_refs(body, params, module_names, out),
        Expr::Tuple(elems, _) | Expr::List(elems, _) => {
            for e in elems {
                collect_surf_refs(e, params, module_names, out);
            }
        }
        Expr::Access(e, _, _) | Expr::TupleGet(e, _, _) | Expr::Annotate(e, _, _) => {
            collect_surf_refs(e, params, module_names, out);
        }
        _ => {}
    }
}

fn compute_deep_dependency_edges(source: &str) -> Result<Vec<(String, Vec<String>)>, String> {
    use std::collections::BTreeSet;
    let exprs = chelis_deep::parser::parse_str(source).map_err(|e| format!("parse: {e}"))?;
    // Collect module-level def names (exports).
    let mut module_names = BTreeSet::new();
    for expr in &exprs {
        if list_tag(expr) == Some("def")
            && let DeepExpr::List(list, _) = expr
            && let Some(name) = list.elements.get(2).and_then(symbol_text)
        {
            module_names.insert(name.to_string());
        }
    }
    // Discover properties and extract their references.
    let properties =
        discover_deep_properties(Path::new("<dep-scan>"), &exprs, None).unwrap_or_default();
    let mut edges = Vec::new();
    for prop in &properties {
        let mut refs = BTreeSet::new();
        let param_names: BTreeSet<&str> = prop.params.iter().map(|p| p.name.as_str()).collect();
        collect_deep_refs(&prop.body, &param_names, &module_names, &mut refs);
        for pre in &prop.preconditions {
            collect_deep_refs(pre, &param_names, &module_names, &mut refs);
        }
        // Exclude the property's own name.
        refs.remove(&prop.name);
        edges.push((prop.name.clone(), refs.into_iter().collect()));
    }
    Ok(edges)
}

fn collect_deep_refs(
    expr: &DeepExpr,
    params: &std::collections::BTreeSet<&str>,
    module_names: &std::collections::BTreeSet<String>,
    out: &mut std::collections::BTreeSet<String>,
) {
    if let DeepExpr::List(list, _) = expr {
        let tag = list_tag_from_list(list);
        if tag == Some("var") {
            if let Some(name) = list.elements.get(2).and_then(symbol_text)
                && !params.contains(name)
                && module_names.contains(name)
            {
                out.insert(name.to_string());
            }
        } else if tag == Some("app") {
            // First child after tag+meta is the callee.
            for child in list.elements.iter().skip(2) {
                collect_deep_refs(child, params, module_names, out);
            }
        } else {
            for child in list.elements.iter().skip(2) {
                collect_deep_refs(child, params, module_names, out);
            }
        }
    }
}

fn is_single_explicit_deep_input(path: Option<&Path>, inputs: &[PathBuf]) -> bool {
    matches!(path, Some(path) if path.is_file())
        && inputs.len() == 1
        && inputs[0].extension().and_then(|ext| ext.to_str()) == Some("dp")
}

fn combine_status(lhs: Status, rhs: Status) -> Status {
    use Status::*;
    // Precedence, worst wins: Error > Failed > Unsupported > Passed. A
    // DISPROVED property (Failed, exit 1) outranks an Unsupported one
    // (exit 2), so a genuine falsification is never masked by a co-occurring
    // "the prover could not handle this" -- a CI gate keying on exit 1 ("a
    // property was disproved") sees the failure instead of a misleading
    // exit 2 (RT #9).
    match (lhs, rhs) {
        (Error, _) | (_, Error) => Error,
        (Failed, _) | (_, Failed) => Failed,
        (Unsupported, _) | (_, Unsupported) => Unsupported,
        _ => Passed,
    }
}

fn discover_inputs(path: Option<&Path>) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    match path {
        Some(path) if path.is_file() => {
            if is_supported_input(path) {
                out.push(path.to_path_buf());
            }
        }
        Some(path) if path.is_dir() => collect_inputs_in_dir(path, &mut out),
        Some(path) => return Err(format!("path `{}` does not exist", path.display())),
        None => {
            for dirname in ["properties", "src"] {
                let path = Path::new(dirname);
                if path.exists() {
                    collect_inputs_in_dir(path, &mut out);
                }
            }
        }
    }
    out.sort();
    out.dedup();
    Ok(out)
}

fn collect_inputs_in_dir(path: &Path, out: &mut Vec<PathBuf>) {
    for entry in WalkDir::new(path)
        .into_iter()
        .filter_entry(|entry| !is_skipped_dir(entry.path()))
        .flatten()
    {
        let path = entry.path();
        if path.is_file() && is_supported_input(path) {
            out.push(path.to_path_buf());
        }
    }
}

fn is_skipped_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name == "tests"
                || name == "target"
                || name == "dist"
                || name == "dependencies"
                || name.starts_with('.')
        })
}

fn is_supported_input(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("ch" | "dp")
    )
}

fn prove_surf_file(
    path: &Path,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Result<Status, String> {
    let source =
        fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let parsed = chelis_surf::parser::parse_str(&source)
        .map_err(|err| format!("parse {}: {err}", path.display()))?;
    let flat = flatten_module_decls(&parsed);
    let mut file_status = Status::Passed;
    // Default (no obligation engine) build: type-check the module up-front so a
    // type-broken module errors instead of silently passing. A capability
    // build gets this from the obligation path (`obligation_run::run_obligations`,
    // which desugars + checks); without it nothing else here type-checks the
    // module. Use the SAME bare desugar + check the engine uses
    // (`desugar_program` then `check_typed_program`) so the default and
    // capability builds agree on what is type-broken.
    #[cfg(not(feature = "chelis-prove"))]
    {
        let deep_exprs = chelis_surf::desugar::desugar_program(&parsed);
        if let Err(infer) = chelis_types::check_typed_program(&deep_exprs) {
            let messages = infer
                .errors
                .iter()
                .map(|err| err.message.clone())
                .collect::<Vec<_>>();
            emit_module_check_failure(options, &messages, totals);
            return Ok(Status::Error);
        }
    }
    // Under the `chelis-prove` capability the user @property declarations run
    // through the SHARED property runner (U4 / D-PARITY): the SAME discovery
    // + engine the tide MCP tool uses, so a CLI prove and a tide prove agree
    // on the same module. The CLI renders the outcomes as the NDJSON
    // `{kind:"property"}` records and folds them into the summary. Without the
    // capability, fall back to the CLI-local Tier-C-only property path.
    #[cfg(feature = "chelis-prove")]
    {
        let _ = (&flat, &parsed);
        let package_root = resolve_package_root(path, options.package);
        let linked_program = match &package_root {
            Some(root) => chelis_reef::prepare_program_for_eval_file(path, root),
            None => chelis_reef::prepare_program_for_file(path),
        };
        let prop_status = match &linked_program {
            Ok(Some(prepared)) => {
                let display_names = linked_property_display_names(&flat, &prepared.entry_decls);
                property_run::run_surf_linked_properties_shared(
                    path,
                    &prepared.decls,
                    &prepared.entry_decls,
                    &prepared.stdlib_decls,
                    &display_names,
                    options,
                    totals,
                )
            }
            Ok(None) => property_run::run_surf_properties_shared(path, &source, options, totals),
            Err(message) => {
                totals.errors += 1;
                if options.json {
                    println!(
                        "{}",
                        json!({
                            "kind": "error",
                            "stage": "property-discovery",
                            "reason": format!("import resolution failed; properties not verified: {message}"),
                            "source": json!({ "kind": "surf", "file": path.display().to_string() }),
                        })
                    );
                } else {
                    eprintln!(
                        "prove error: import resolution failed in {}; properties not verified: {message}",
                        path.display()
                    );
                }
                Status::Error
            }
        };
        file_status = combine_status(file_status, prop_status);

        let obligation_count = count_invariant_opaque_surf(&flat);
        let ob_status = match (&linked_program, obligation_count) {
            (Ok(Some(prepared)), 0) => {
                obligation_run::check_linked_decls(&prepared.decls, options, totals)
            }
            (Ok(None), 0) => obligation_run::run_obligations(&parsed, options, totals),
            (Err(_), 0) => Status::Passed,
            (Ok(_), _) => obligation_run::run_obligations(&parsed, options, totals),
            (Err(_), _) => Status::Passed,
        };
        file_status = combine_status(file_status, ob_status);
    }
    #[cfg(not(feature = "chelis-prove"))]
    {
        let properties = collect_surf_properties(path, &flat, options.only);
        for property in properties {
            let status = prove_surf_property(&flat, &parsed, &property, options, totals);
            file_status = combine_status(file_status, status);
        }
    }
    // CR2-5: warn whenever obligations were NOT SMT-verified, gated on the
    // actual capability (`smt`) rather than on the optional `chelis-prove`
    // dependency. A `chelis-prove`-without-`smt` build compiles the
    // obligation machinery and runs it via Tier C (fuzz) -- NOT cvc5 -- so a
    // clean run must still be flagged as not formally verified. The warning
    // is stderr-only and never touches the stdout NDJSON stream or the exit
    // code.
    #[cfg(not(feature = "smt"))]
    warn_obligations_skipped_without_smt(path, count_invariant_opaque_surf(&flat), options);
    Ok(file_status)
}

fn flatten_module_decls(decls: &[Decl]) -> Vec<Decl> {
    let mut out = Vec::new();
    for decl in decls {
        match decl {
            Decl::Module { decls, .. } => out.extend(flatten_module_decls(decls)),
            other => out.push(other.clone()),
        }
    }
    out
}

#[cfg(feature = "chelis-prove")]
fn linked_property_display_names(
    source_entry_decls: &[Decl],
    linked_entry_decls: &[Decl],
) -> BTreeMap<String, String> {
    property_names(linked_entry_decls)
        .into_iter()
        .zip(property_names(source_entry_decls))
        .collect()
}

#[cfg(feature = "chelis-prove")]
fn property_names(decls: &[Decl]) -> Vec<String> {
    let mut names = Vec::new();
    for decl in decls {
        match decl {
            Decl::Module { decls, .. } => names.extend(property_names(decls)),
            Decl::Property { name, .. } => names.push(name.clone()),
            _ => {}
        }
    }
    names
}

/// Emit a module type-check failure as a prove error record, for the default
/// (no obligation engine) build's up-front type-check. Mirrors the shape of
/// the capability path's check-failure record (`kind:"error", stage:"check"`)
/// so a type-broken module reports identically whether or not the obligation
/// engine is compiled in. The diagnostics go to the stdout NDJSON stream under
/// `--json` and to stderr otherwise; the caller returns `Status::Error`.
#[cfg(not(feature = "chelis-prove"))]
fn emit_module_check_failure(
    options: &ProveOptions<'_>,
    messages: &[String],
    totals: &mut Summary,
) {
    totals.errors += 1;
    let joined = messages.join("; ");
    if options.json {
        println!(
            "{}",
            json!({
                "kind": "error",
                "stage": "check",
                "reason": format!("module does not type-check: {joined}"),
                "diagnostics": messages,
            })
        );
    } else {
        eprintln!("prove error: module does not type-check:");
        for m in messages {
            eprintln!("  - {m}");
        }
    }
}

#[cfg(not(feature = "chelis-prove"))]
fn collect_surf_properties(path: &Path, decls: &[Decl], only: Option<&str>) -> Vec<Property> {
    decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::Property {
                name,
                params,
                preconditions,
                body,
                options,
                ..
            } if matches_filter(name, only) => Some(Property {
                name: name.clone(),
                source: path.to_path_buf(),
                params: params.clone(),
                preconditions: preconditions.clone(),
                body: body.clone(),
                samples: property_samples(options),
                seed: property_seed(options),
            }),
            _ => None,
        })
        .collect()
}

#[cfg(not(feature = "chelis-prove"))]
fn property_samples(options: &[chelis_surf::ast::PropertyOption]) -> Option<usize> {
    options.iter().find_map(|option| match option {
        chelis_surf::ast::PropertyOption::Samples(Expr::Lit(Literal::Int(value), _), _) => {
            (*value >= 0).then_some(*value as usize)
        }
        _ => None,
    })
}

#[cfg(not(feature = "chelis-prove"))]
fn property_seed(options: &[chelis_surf::ast::PropertyOption]) -> Option<u64> {
    options.iter().find_map(|option| match option {
        chelis_surf::ast::PropertyOption::Seed(Expr::Lit(Literal::Int(value), _), _) => {
            (*value >= 0).then_some(*value as u64)
        }
        _ => None,
    })
}

#[cfg(not(feature = "chelis-prove"))]
fn prove_surf_property(
    decls: &[Decl],
    module_decls: &[Decl],
    property: &Property,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    let _ = module_decls;
    // The no-`chelis-prove` build runs only Tier C (fuzz). Tier B (SMT) and
    // assumption injection require the capability and live in the shared
    // `chelis_prove::property_runner`, which this build does not reach.
    totals.total += 1;

    let samples_needed = options.samples.or(property.samples).unwrap_or(100);
    let max_attempts = options
        .max_attempts
        .unwrap_or_else(|| samples_needed.saturating_mul(100).max(samples_needed));
    let seed = options.effective_seed(property.seed);
    let mut rng = Lcg::new(seed);

    if let Some(reason) = unsupported_property(property) {
        totals.unsupported += 1;
        emit_record(options, property, "unsupported", 0, None, Some(reason), 0);
        return Status::Unsupported;
    }

    let mut accepted = 0usize;
    let mut attempts = 0usize;
    while accepted < samples_needed && attempts < max_attempts {
        attempts += 1;
        let sample = match sample_property(property, &mut rng) {
            Ok(sample) => sample,
            Err(reason) => {
                totals.unsupported += 1;
                emit_record(options, property, "unsupported", 0, None, Some(reason), 0);
                return Status::Unsupported;
            }
        };
        if !property.preconditions.is_empty() {
            match eval_surf_sample(decls, property, &sample, true) {
                Ok(false) => continue,
                Ok(true) => {}
                Err(err) => {
                    totals.errors += 1;
                    emit_error(options, property, &err);
                    return Status::Error;
                }
            }
        }
        accepted += 1;
        match eval_surf_sample(decls, property, &sample, false) {
            Ok(true) => {}
            Ok(false) => {
                let (shrunk, shrink_steps) = shrink_surf_counterexample(decls, property, sample);
                totals.failed += 1;
                emit_record(
                    options,
                    property,
                    "failed",
                    accepted,
                    Some(counterexample_json(&shrunk)),
                    None,
                    shrink_steps,
                );
                return Status::Failed;
            }
            Err(err) => {
                totals.errors += 1;
                emit_error(options, property, &err);
                return Status::Error;
            }
        }
    }

    if accepted < samples_needed {
        totals.errors += 1;
        emit_error(
            options,
            property,
            &format!(
                "generator exhausted after {attempts} attempts before collecting {samples_needed} valid samples"
            ),
        );
        return Status::Error;
    }

    totals.passed += 1;
    emit_record(options, property, "passed", accepted, None, None, 0);
    Status::Passed
}

#[cfg(not(feature = "chelis-prove"))]
fn unsupported_property(property: &Property) -> Option<String> {
    unsupported_property_params(&property.params)
}

fn unsupported_property_params(params: &[Param]) -> Option<String> {
    for param in params {
        if let Some(reason) = unsupported_type(param.ty.as_ref()?) {
            return Some(format!("{}: {reason}", param.name));
        }
    }
    None
}

/// Whether a primitive type name is a signed integer width, via the type
/// system's single-source recognizer (review 5): a name is an int width iff
/// it parses to a `Prim` the type system classifies as integer.
fn is_int_width(name: &str) -> bool {
    chelis_types::types::Prim::parse_name(name).is_some_and(|p| p.is_integer())
}

fn unsupported_type(ty: &TypeExpr) -> Option<String> {
    match ty {
        // bool / f32 / f64 / string, plus EVERY signed integer width
        // recognized through the type system (review 5), so the supported-
        // type gate and the sampler agree on the admissible integer widths.
        TypeExpr::Named(name, _)
            if matches!(name.as_str(), "bool" | "f32" | "f64" | "string") || is_int_width(name) =>
        {
            None
        }
        TypeExpr::Tensor(dims, precision, _) if matches!(precision.as_str(), "f32" | "f64") => {
            if dims.iter().all(
                |dim| matches!(dim, TypeExpr::Named(value, _) if value.parse::<usize>().is_ok()),
            ) {
                None
            } else {
                Some("symbolic tensor dimensions are not supported in L2 v1".to_string())
            }
        }
        TypeExpr::Tensor(_, precision, _) => Some(format!(
            "tensor element type `{precision}` is not supported in L2 v1"
        )),
        _ => Some("type is not supported in L2 v1".to_string()),
    }
}

#[cfg(not(feature = "chelis-prove"))]
fn sample_property(property: &Property, rng: &mut Lcg) -> Result<Sample, String> {
    let mut values = Vec::new();
    for param in &property.params {
        let ty = param
            .ty
            .as_ref()
            .ok_or_else(|| format!("{} is missing an explicit type", param.name))?;
        values.push(sample_value(&param.name, ty, rng)?);
    }
    Ok(Sample { values })
}

fn sample_value(name: &str, ty: &TypeExpr, rng: &mut Lcg) -> Result<SampleValue, String> {
    let sp = chelis_deep::Span::new(0, 0);
    match ty {
        TypeExpr::Named(type_name, _) if type_name == "bool" => {
            let value = rng.next_bool();
            Ok(scalar_sample(
                name,
                Expr::Lit(Literal::Bool(value), sp),
                deep_lit(deep_bool(value), "bool"),
                json!(value),
            ))
        }
        // Every signed integer width, recognized through the type system and
        // sampled within the width's representable range via the single
        // workspace source `Prim::integer_fuzz_bounds` (review 5). int32 is
        // the literal default; the other widths cast an int32 literal to the
        // target width so the value is well-typed.
        TypeExpr::Named(type_name, _) if is_int_width(type_name) => {
            let (lo, hi) = chelis_types::types::Prim::parse_name(type_name)
                .and_then(|p| p.integer_fuzz_bounds())
                .expect("is_int_width implies integer_fuzz_bounds");
            let value = rng.next_i64(lo, hi);
            let lit = Expr::Lit(Literal::Int(value), sp);
            if type_name == "int32" {
                Ok(scalar_sample(
                    name,
                    lit,
                    deep_lit(deep_int(value), "int32"),
                    json!(value),
                ))
            } else {
                Ok(scalar_sample(
                    name,
                    cast_expr(lit, type_name),
                    deep_lit(deep_int(value), type_name),
                    json!(value),
                ))
            }
        }
        TypeExpr::Named(type_name, _) if type_name == "f32" || type_name == "f64" => {
            let value = rng.next_f64(-10.0, 10.0);
            let lit = Expr::Lit(Literal::Float(value), sp);
            if type_name == "f64" {
                Ok(scalar_sample(
                    name,
                    cast_expr(lit, "f64"),
                    deep_lit(deep_float(value), "f64"),
                    json!(value),
                ))
            } else {
                Ok(scalar_sample(
                    name,
                    lit,
                    deep_lit(deep_float(value), "f32"),
                    json!(value),
                ))
            }
        }
        TypeExpr::Named(type_name, _) if type_name == "string" => {
            let value = format!("s{}", rng.next_u64() % 1000);
            Ok(scalar_sample(
                name,
                Expr::Lit(Literal::Str(value.clone()), sp),
                deep_lit(deep_string(&value), "string"),
                json!(value),
            ))
        }
        TypeExpr::Tensor(dims, precision, _) => sample_tensor_value(name, dims, precision, rng),
        _ => Err("type is not supported in L2 v1".to_string()),
    }
}

fn scalar_sample(
    name: &str,
    surf_expr: Expr,
    deep_expr: DeepExpr,
    json: serde_json::Value,
) -> SampleValue {
    SampleValue {
        name: name.to_string(),
        surf_expr,
        deep_expr,
        json,
        tensor_binding: None,
    }
}

fn sample_tensor_value(
    name: &str,
    dims: &[TypeExpr],
    precision: &str,
    rng: &mut Lcg,
) -> Result<SampleValue, String> {
    let shape = dims
        .iter()
        .map(|dim| match dim {
            TypeExpr::Named(value, _) => value
                .parse::<usize>()
                .map_err(|_| "symbolic tensor dimensions are not supported in L2 v1".to_string()),
            _ => Err("symbolic tensor dimensions are not supported in L2 v1".to_string()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape.is_empty() || shape.len() > 2 {
        return Err(
            "only rank-1 and rank-2 fixed-shape tensors are supported in L2 v1".to_string(),
        );
    }
    let count = shape.iter().product::<usize>();
    let values = (0..count)
        .map(|_| rng.next_f64(-10.0, 10.0))
        .collect::<Vec<_>>();
    let surf_expr = tensor_surf_expr(&shape, precision, &values);
    let deep_expr = tensor_deep_expr(&shape, precision, &values);
    Ok(SampleValue {
        name: name.to_string(),
        surf_expr,
        deep_expr,
        json: json!({ "shape": shape, "data": values }),
        tensor_binding: None,
    })
}

fn tensor_surf_expr(shape: &[usize], precision: &str, values: &[f64]) -> Expr {
    let sp = chelis_deep::Span::new(0, 0);
    let scalar = |value| {
        let lit = Expr::Lit(Literal::Float(value), sp);
        if precision == "f64" {
            cast_expr(lit, "f64")
        } else {
            lit
        }
    };
    if shape.len() == 1 {
        return Expr::Apply(
            Box::new(Expr::Var("to_tensor".to_string(), sp)),
            vec![Expr::List(values.iter().copied().map(scalar).collect(), sp)],
            sp,
        );
    }
    let cols = shape[1];
    let rows = values
        .chunks(cols)
        .map(|row| Expr::List(row.iter().copied().map(scalar).collect(), sp))
        .collect::<Vec<_>>();
    Expr::Apply(
        Box::new(Expr::Var("pad_sequences".to_string(), sp)),
        vec![Expr::List(rows, sp), scalar(0.0)],
        sp,
    )
}

fn tensor_deep_expr(shape: &[usize], precision: &str, values: &[f64]) -> DeepExpr {
    let scalar = |value| deep_numeric_tensor_scalar(value, precision);
    if shape.len() == 1 {
        return deep_node(
            "app",
            vec![
                deep_var("to_tensor"),
                deep_cons_list(values.iter().copied().map(scalar).collect()),
            ],
        );
    }
    let cols = shape[1];
    let rows = values
        .chunks(cols)
        .map(|row| deep_cons_list(row.iter().copied().map(scalar).collect()))
        .collect::<Vec<_>>();
    deep_node(
        "app",
        vec![
            deep_var("pad_sequences"),
            deep_cons_list(rows),
            deep_numeric_tensor_scalar(0.0, precision),
        ],
    )
}

fn deep_numeric_tensor_scalar(value: f64, precision: &str) -> DeepExpr {
    let lit = deep_lit(deep_float(value), "f32");
    if precision == "f64" {
        deep_node(
            "cast",
            vec![lit, deep_node("t-prim", vec![deep_symbol("f64")])],
        )
    } else {
        lit
    }
}

fn deep_cons_list(items: Vec<DeepExpr>) -> DeepExpr {
    items.into_iter().rev().fold(deep_var("Nil"), |tail, item| {
        deep_node("app", vec![deep_var("Cons"), item, tail])
    })
}

fn cast_expr(expr: Expr, ty: &str) -> Expr {
    Expr::Cast(Box::new(expr), ty.to_string(), chelis_deep::Span::new(0, 0))
}

fn deep_span() -> chelis_deep::Span {
    chelis_deep::Span::new(0, 0)
}

fn deep_symbol(value: &str) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Symbol(value.to_string()), deep_span())
}

fn deep_int(value: i64) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Int(value), deep_span())
}

fn deep_float(value: f64) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Float(value), deep_span())
}

fn deep_bool(value: bool) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Bool(value), deep_span())
}

fn deep_string(value: &str) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Str(value.to_string()), deep_span())
}

fn deep_map(entries: Vec<(String, DeepExpr)>) -> DeepExpr {
    DeepExpr::Map(MetaMap { entries }, deep_span())
}

fn deep_list(elements: Vec<DeepExpr>) -> DeepExpr {
    DeepExpr::List(DeepList { elements }, deep_span())
}

fn deep_node(tag: &str, children: Vec<DeepExpr>) -> DeepExpr {
    let mut elements = vec![deep_symbol(tag), deep_map(Vec::new())];
    elements.extend(children);
    deep_list(elements)
}

fn deep_node_meta(
    tag: &str,
    entries: Vec<(String, DeepExpr)>,
    children: Vec<DeepExpr>,
) -> DeepExpr {
    let mut elements = vec![deep_symbol(tag), deep_map(entries)];
    elements.extend(children);
    deep_list(elements)
}

fn deep_var(name: &str) -> DeepExpr {
    deep_node("var", vec![deep_symbol(name)])
}

/// The discharged proposition of a Deep property as a Deep term (chelis#436):
/// the bare `body` when unguarded, or the implication `(implies (/\ pre) body)`
/// when guarded, so a guarded property's rendered `goal` is exactly what was
/// discharged and never reads as an unconditional claim (MED-1). Mirrors the
/// shared runner's `deep_proposition` so the CLI-local Deep path (bridge
/// c-earchin + the non-smt deep-user path) and the shared runner agree.
fn deep_proposition(preconditions: &[DeepExpr], body: &DeepExpr) -> DeepExpr {
    if preconditions.is_empty() {
        return body.clone();
    }
    let combined = preconditions
        .iter()
        .cloned()
        .reduce(|left, right| deep_node("app", vec![deep_var("and"), left, right]))
        .unwrap_or_else(|| deep_lit(deep_bool(true), "bool"));
    deep_node("app", vec![deep_var("implies"), combined, body.clone()])
}

fn deep_lit(value: DeepExpr, ty_name: &str) -> DeepExpr {
    deep_node_meta(
        "lit",
        vec![(
            "type".to_string(),
            deep_node("t-prim", vec![deep_symbol(ty_name)]),
        )],
        vec![value],
    )
}

#[cfg(not(feature = "chelis-prove"))]
fn eval_surf_sample(
    decls: &[Decl],
    property: &Property,
    sample: &Sample,
    precondition: bool,
) -> Result<bool, String> {
    let root = if precondition {
        "__chelis_property_pre"
    } else {
        "__chelis_property_probe"
    };
    let mut source_decls = decls.to_vec();
    for value in &sample.values {
        if let Some((binding_name, tensor)) = &value.tensor_binding {
            source_decls.push(Decl::Sig {
                name: binding_name.clone(),
                ty: property
                    .params
                    .iter()
                    .find(|param| param.name == value.name)
                    .and_then(|param| param.ty.clone())
                    .expect("tensor property params are typed"),
                effects: None,
                span: chelis_deep::Span::new(0, 0),
            });
            debug_assert_eq!(tensor.data.len(), tensor.shape.iter().product::<usize>());
        }
    }
    source_decls.push(Decl::LetDef {
        name: root.to_string(),
        ty: None,
        value: sample_block_expr(property, sample, precondition),
        span: chelis_deep::Span::new(0, 0),
    });
    let source = chelis_surf::format::format_program(&source_decls);
    eval_bool_with_bindings(SourceKind::Surf, source, root, sample_bindings(sample))
}

#[cfg(not(feature = "chelis-prove"))]
fn sample_block_expr(property: &Property, sample: &Sample, precondition: bool) -> Expr {
    let sp = chelis_deep::Span::new(0, 0);
    let bindings = sample
        .values
        .iter()
        .map(|value| LetBinding {
            pattern: LetPattern::Var(value.name.clone(), sp),
            ty: None,
            value: value.surf_expr.clone(),
        })
        .collect::<Vec<_>>();
    let body = if precondition {
        combine_preconditions(&property.preconditions)
    } else {
        Expr::Apply(
            Box::new(Expr::Var(property.name.clone(), sp)),
            property
                .params
                .iter()
                .map(|param| Expr::Var(param.name.clone(), sp))
                .collect(),
            sp,
        )
    };
    Expr::Block(bindings, Box::new(body), sp)
}

#[cfg(not(feature = "chelis-prove"))]
fn combine_preconditions(preconditions: &[Expr]) -> Expr {
    let sp = chelis_deep::Span::new(0, 0);
    preconditions
        .iter()
        .cloned()
        .reduce(|left, right| Expr::Binary(BinOp::And, Box::new(left), Box::new(right), sp))
        .unwrap_or(Expr::Lit(Literal::Bool(true), sp))
}

fn eval_bool_with_bindings(
    source_kind: SourceKind,
    source: String,
    root: &str,
    bindings: BTreeMap<String, TensorValue>,
) -> Result<bool, String> {
    let result = chelis_compiler_api::compiler::eval_selected(
        EvalRequest {
            source_kind,
            source,
            bindings,
        },
        &[root.to_string()],
    )
    .map_err(|err| {
        err.errors
            .iter()
            .map(|diag| diag.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    match result.roots.as_slice() {
        [root] => match &root.value {
            ExecutionValue::Bool { value } => Ok(*value),
            ExecutionValue::Tensor { value } if value.shape.is_empty() && value.data.len() == 1 => {
                Ok(value.data[0] != 0.0)
            }
            other => Err(format!(
                "property root evaluated to non-bool value: {other:?}"
            )),
        },
        _ => Err("property evaluation did not return exactly one root".to_string()),
    }
}

fn sample_bindings(sample: &Sample) -> BTreeMap<String, TensorValue> {
    sample
        .values
        .iter()
        .filter_map(|value| value.tensor_binding.clone())
        .collect()
}

fn counterexample_json(sample: &Sample) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for value in &sample.values {
        map.insert(value.name.clone(), value.json.clone());
    }
    serde_json::Value::Object(map)
}

const MAX_SHRINK_STEPS: usize = 64;

#[cfg(not(feature = "chelis-prove"))]
fn shrink_surf_counterexample(
    decls: &[Decl],
    property: &Property,
    sample: Sample,
) -> (Sample, usize) {
    shrink_counterexample(sample, &property.params, |candidate| {
        sample_still_fails_surf(decls, property, candidate)
    })
}

fn shrink_deep_counterexample(
    exprs: &[DeepExpr],
    property: &DeepProperty,
    sample: Sample,
) -> (Sample, usize) {
    shrink_counterexample(sample, &property.params, |candidate| {
        sample_still_fails_deep(exprs, property, candidate)
    })
}

fn shrink_counterexample<F>(
    mut sample: Sample,
    params: &[Param],
    mut still_fails: F,
) -> (Sample, usize)
where
    F: FnMut(&Sample) -> bool,
{
    let mut steps = 0usize;
    while steps < MAX_SHRINK_STEPS {
        let mut changed = false;
        for index in 0..sample.values.len() {
            let Some(ty) = params
                .iter()
                .find(|param| param.name == sample.values[index].name)
                .and_then(|param| param.ty.as_ref())
            else {
                continue;
            };
            for candidate in shrink_candidates(&sample.values[index], ty) {
                if candidate.json == sample.values[index].json {
                    continue;
                }
                let mut trial = sample.clone();
                trial.values[index] = candidate;
                if still_fails(&trial) {
                    sample = trial;
                    steps += 1;
                    changed = true;
                    break;
                }
            }
            if changed || steps >= MAX_SHRINK_STEPS {
                break;
            }
        }
        if !changed {
            break;
        }
    }
    (sample, steps)
}

#[cfg(not(feature = "chelis-prove"))]
fn sample_still_fails_surf(decls: &[Decl], property: &Property, sample: &Sample) -> bool {
    if !property.preconditions.is_empty() {
        match eval_surf_sample(decls, property, sample, true) {
            Ok(true) => {}
            Ok(false) | Err(_) => return false,
        }
    }
    matches!(eval_surf_sample(decls, property, sample, false), Ok(false))
}

fn sample_still_fails_deep(exprs: &[DeepExpr], property: &DeepProperty, sample: &Sample) -> bool {
    if !property.preconditions.is_empty() {
        match eval_deep_sample(exprs, property, sample, true) {
            Ok(true) => {}
            Ok(false) | Err(_) => return false,
        }
    }
    matches!(eval_deep_sample(exprs, property, sample, false), Ok(false))
}

fn shrink_candidates(value: &SampleValue, ty: &TypeExpr) -> Vec<SampleValue> {
    match ty {
        TypeExpr::Named(type_name, _) if type_name == "bool" => value
            .json
            .as_bool()
            .and_then(|current| current.then(|| bool_sample(&value.name, false)))
            .into_iter()
            .collect(),
        TypeExpr::Named(type_name, _) if is_int_width(type_name) => value
            .json
            .as_i64()
            .map(|current| {
                let mut candidates = Vec::new();
                push_unique_int_candidate(&mut candidates, &value.name, type_name, current, 0);
                push_unique_int_candidate(
                    &mut candidates,
                    &value.name,
                    type_name,
                    current,
                    current / 2,
                );
                push_unique_int_candidate(
                    &mut candidates,
                    &value.name,
                    type_name,
                    current,
                    current.signum(),
                );
                candidates
            })
            .unwrap_or_default(),
        TypeExpr::Named(type_name, _) if type_name == "f32" || type_name == "f64" => value
            .json
            .as_f64()
            .map(|current| {
                let mut candidates = Vec::new();
                push_unique_float_candidate(&mut candidates, &value.name, type_name, current, 0.0);
                push_unique_float_candidate(
                    &mut candidates,
                    &value.name,
                    type_name,
                    current,
                    current / 2.0,
                );
                candidates
            })
            .unwrap_or_default(),
        TypeExpr::Named(type_name, _) if type_name == "string" => value
            .json
            .as_str()
            .map(|current| {
                let mut candidates = Vec::new();
                push_unique_string_candidate(&mut candidates, &value.name, current, "");
                if !current.is_empty() {
                    push_unique_string_candidate(
                        &mut candidates,
                        &value.name,
                        current,
                        &current[..current.len() / 2],
                    );
                }
                candidates
            })
            .unwrap_or_default(),
        TypeExpr::Tensor(_, precision, _) => tensor_shrink_candidates(value, precision),
        _ => Vec::new(),
    }
}

fn push_unique_int_candidate(
    candidates: &mut Vec<SampleValue>,
    name: &str,
    type_name: &str,
    current: i64,
    candidate: i64,
) {
    if candidate != current
        && !candidates
            .iter()
            .any(|sample| sample.json.as_i64() == Some(candidate))
    {
        candidates.push(int_sample(name, type_name, candidate));
    }
}

fn push_unique_float_candidate(
    candidates: &mut Vec<SampleValue>,
    name: &str,
    type_name: &str,
    current: f64,
    candidate: f64,
) {
    if (candidate - current).abs() > f64::EPSILON
        && !candidates.iter().any(|sample| {
            sample
                .json
                .as_f64()
                .is_some_and(|prior| (prior - candidate).abs() <= f64::EPSILON)
        })
    {
        candidates.push(float_sample(name, type_name, candidate));
    }
}

fn push_unique_string_candidate(
    candidates: &mut Vec<SampleValue>,
    name: &str,
    current: &str,
    candidate: &str,
) {
    if candidate != current
        && !candidates
            .iter()
            .any(|sample| sample.json.as_str() == Some(candidate))
    {
        candidates.push(string_sample(name, candidate));
    }
}

fn bool_sample(name: &str, value: bool) -> SampleValue {
    scalar_sample(
        name,
        Expr::Lit(Literal::Bool(value), chelis_deep::Span::new(0, 0)),
        deep_lit(deep_bool(value), "bool"),
        json!(value),
    )
}

fn int_sample(name: &str, type_name: &str, value: i64) -> SampleValue {
    let (lo, hi) = chelis_types::types::Prim::parse_name(type_name)
        .and_then(|p| p.integer_fuzz_bounds())
        .expect("int shrink only uses int widths");
    let value = value.clamp(lo, hi);
    let lit = Expr::Lit(Literal::Int(value), chelis_deep::Span::new(0, 0));
    let surf_expr = if type_name == "int32" {
        lit
    } else {
        cast_expr(lit, type_name)
    };
    scalar_sample(
        name,
        surf_expr,
        deep_lit(deep_int(value), type_name),
        json!(value),
    )
}

fn float_sample(name: &str, type_name: &str, value: f64) -> SampleValue {
    let lit = Expr::Lit(Literal::Float(value), chelis_deep::Span::new(0, 0));
    let surf_expr = if type_name == "f64" {
        cast_expr(lit, "f64")
    } else {
        lit
    };
    scalar_sample(
        name,
        surf_expr,
        deep_lit(deep_float(value), type_name),
        json!(value),
    )
}

fn string_sample(name: &str, value: &str) -> SampleValue {
    scalar_sample(
        name,
        Expr::Lit(
            Literal::Str(value.to_string()),
            chelis_deep::Span::new(0, 0),
        ),
        deep_lit(deep_string(value), "string"),
        json!(value),
    )
}

fn tensor_shrink_candidates(value: &SampleValue, precision: &str) -> Vec<SampleValue> {
    let Some(shape) = value
        .json
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_u64().map(|dim| dim as usize))
                .collect::<Vec<_>>()
        })
    else {
        return Vec::new();
    };
    let Some(data) = value
        .json
        .get("data")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_f64)
                .collect::<Vec<_>>()
        })
    else {
        return Vec::new();
    };
    if shape.iter().product::<usize>() != data.len() {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    let zeros = vec![0.0; data.len()];
    if data.iter().any(|value| value.abs() > f64::EPSILON) {
        candidates.push(tensor_sample(&value.name, &shape, precision, &zeros));
    }
    let halves = data.iter().map(|value| value / 2.0).collect::<Vec<_>>();
    if halves
        .iter()
        .zip(&data)
        .any(|(candidate, current)| (candidate - current).abs() > f64::EPSILON)
    {
        candidates.push(tensor_sample(&value.name, &shape, precision, &halves));
    }
    candidates
}

fn tensor_sample(name: &str, shape: &[usize], precision: &str, values: &[f64]) -> SampleValue {
    SampleValue {
        name: name.to_string(),
        surf_expr: tensor_surf_expr(shape, precision, values),
        deep_expr: tensor_deep_expr(shape, precision, values),
        json: json!({ "shape": shape, "data": values }),
        tensor_binding: None,
    }
}

fn prove_deep_file(
    path: &Path,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Result<Status, String> {
    let source =
        fs::read_to_string(path).map_err(|err| format!("read {}: {err}", path.display()))?;
    let exprs = chelis_deep::parser::parse_str(&source)
        .map_err(|err| format!("parse {}: {err}", path.display()))?;
    if let Err(err) = chelis_validate::validate_deep(&source) {
        return Err(format!("validate {}: {err}", path.display()));
    }
    // Default (no obligation engine) build: type-check the module up-front so a
    // type-broken `.dp` errors instead of silently passing. A capability build
    // gets this from the obligation path (`run_deep_obligations`); without it,
    // nothing else here type-checks the module. Uses the SAME bare
    // `check_typed_program` the engine uses, so the default and capability
    // builds agree on what is type-broken (the checker needs no solver).
    #[cfg(not(feature = "chelis-prove"))]
    if let Err(infer) = chelis_types::check_typed_program(&exprs) {
        let messages = infer
            .errors
            .iter()
            .map(|err| err.message.clone())
            .collect::<Vec<_>>();
        emit_module_check_failure(options, &messages, totals);
        return Ok(Status::Error);
    }
    let properties = discover_deep_properties(path, &exprs, options.only)?;
    let mut file_status = Status::Passed;

    // User `@property` declarations run through the SHARED property runner
    // (the SAME engine the tide MCP tool drives, U4 / F6), so a CLI prove and
    // a tide prove of the same `.dp` module agree. Run the shared runner once
    // for ALL user properties, then render each. The c-earchin bridge
    // properties keep the CLI-local path with their span/requirement
    // rendering (a CLI-only surface tide does not run).
    #[cfg(feature = "chelis-prove")]
    {
        let user_status = property_run::run_deep_properties_shared(path, &source, options, totals);
        file_status = combine_status(file_status, user_status);
    }

    for property in properties {
        // Under `chelis-prove` the user properties were already run above by
        // the shared runner; only bridge properties remain for the local
        // path. Without the capability, the local path runs everything.
        #[cfg(feature = "chelis-prove")]
        if property.source_kind == "user" {
            continue;
        }
        let status = prove_deep_property(&exprs, &property, options, totals);
        file_status = combine_status(file_status, status);
    }
    // Derived producer obligations (RFC D-OBLIG / D-PARITY) for the Deep
    // surface. The Surf path (`prove_surf_file`) folds the SAME obligation
    // engine in; without this the `.dp` surface runs ZERO producer-obligation
    // verification, so a `.dp` opaque type declaring an `@invariant` and an
    // UNSOUND producer would pass `chelis prove` silently. A `.dp` is already
    // Deep, so it feeds the shared engine's Deep-program entry directly (no
    // Surf parse/desugar) -- the SAME `run_module_obligations` the Surf source
    // entry reaches after desugaring.
    #[cfg(feature = "chelis-prove")]
    {
        let ob_status = run_deep_obligations(&exprs, options, totals);
        file_status = combine_status(file_status, ob_status);
    }
    #[cfg(not(feature = "smt"))]
    warn_obligations_skipped_without_smt(path, count_invariant_opaque_deep(&exprs), options);
    Ok(file_status)
}

/// Run every derived producer obligation discovered in a `.dp` module's Deep
/// program (RFC D-OBLIG / D-PARITY). The `.dp` is already Deep, so it is fed
/// to the shared `chelis_prove::obligation_engine` directly: the checker runs
/// for inferred return types (a type-broken module is surfaced as an Error,
/// never silent success -- RT3-F2 parity with the Surf path), then the SAME
/// `run_module_obligations` the Surf source entry calls verifies each
/// obligation. Outcomes are rendered as the additive NDJSON
/// `{kind:"obligation", ...}` records and folded into the prove summary,
/// byte-identically to the Surf path's `obligation_run::run_obligations`.
#[cfg(feature = "chelis-prove")]
fn run_deep_obligations(
    exprs: &[DeepExpr],
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    use chelis_prove::obligation_engine::{ObligationRunOptions, run_module_obligations};

    // Type-check the module unconditionally for inferred return types. A
    // `.dp` declaring no invariant has nothing to verify (the engine returns
    // an empty outcome set), but a type-check FAILURE must surface as an Error
    // regardless -- the Deep prove path has no other whole-module type-check
    // stage, so this is what keeps a type-broken `.dp` from silently passing
    // (RT3-F2 parity with the Surf path).
    let sigs = match chelis_types::check_typed_program(exprs) {
        Ok(checked) => checked
            .signature_inference()
            .functions
            .iter()
            .map(|(name, meta)| (name.clone(), meta.checked_signature.clone()))
            .collect::<BTreeMap<_, _>>(),
        // A type-broken module cannot have its obligations meaningfully
        // verified (a rejectable producer can hide behind an unrelated type
        // error): surface the check diagnostics and Error, never silent
        // success. This mirrors `obligation_engine::run_surf_source_obligations`
        // for the Surf surface.
        Err(infer) => {
            // A type-broken `.dp` module is surfaced as an Error here, exactly
            // as the Surf path's `obligation_run::run_obligations` does on any
            // CheckFailed (regardless of whether an opaque invariant is
            // declared). The Deep prove path has no other stage that
            // type-checks the whole module, so escalating here is the ONLY
            // thing that keeps `chelis prove foo.dp` from silently passing a
            // module that `chelis prove foo.ch` rejects (RT3-F2 parity; the
            // re-review caught the no-invariant case slipping through).
            let messages = infer
                .errors
                .iter()
                .map(|err| err.message.clone())
                .collect::<Vec<_>>();
            emit_obligation_check_failure(options, &messages, totals);
            return Status::Error;
        }
    };

    let run_opts = ObligationRunOptions {
        seed: options.seed.unwrap_or(0),
        samples: options.samples.unwrap_or(100),
        smt_timeout_ms: options.smt_timeout_ms,
        tier: options.tier.to_string(),
        only: options.only.map(str::to_string),
        invariant_min_rate: options.invariant_min_rate,
    };
    let outcomes = run_module_obligations(exprs, &sigs, &run_opts);

    let mut status = Status::Passed;
    for outcome in &outcomes {
        let s = render_obligation_outcome(outcome, options, totals);
        status = combine_status(status, s);
    }
    status
}

/// Emit a module type-check failure as a prove error record (RT3-F2 parity).
/// Mirrors `obligation_run::emit_check_failure`: the check diagnostics are
/// surfaced so the failure is visible, never hidden behind a silent pass.
#[cfg(feature = "chelis-prove")]
fn emit_obligation_check_failure(
    options: &ProveOptions<'_>,
    messages: &[String],
    totals: &mut Summary,
) {
    totals.errors += 1;
    let joined = messages.join("; ");
    if options.json {
        println!(
            "{}",
            json!({
                "kind": "error",
                "stage": "check",
                "reason": format!("module does not type-check; obligations not verified: {joined}"),
                "diagnostics": messages,
            })
        );
    } else {
        eprintln!("prove error: module does not type-check; obligations not verified:");
        for m in messages {
            eprintln!("  - {m}");
        }
    }
}

/// Fold one obligation outcome into the running totals and render it. Mirrors
/// `obligation_run::render_outcome` so the Deep surface produces the SAME
/// summary counts and exit-code contribution as the Surf surface.
#[cfg(feature = "chelis-prove")]
fn render_obligation_outcome(
    outcome: &chelis_prove::obligation_engine::ObligationOutcome,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    use chelis_prove::obligation_engine::ObligationStatus;
    match outcome.status {
        ObligationStatus::Error => {
            // A collection-time declaration error (covered-or-rejected /
            // signature rejection). Always surfaces; not counted as an
            // obligation (parity with the Surf renderer).
            totals.errors += 1;
            emit_obligation_record(outcome, options);
            Status::Error
        }
        ObligationStatus::Passed => {
            totals.total += 1;
            totals.obligations += 1;
            totals.passed += 1;
            emit_obligation_record(outcome, options);
            Status::Passed
        }
        ObligationStatus::Failed => {
            totals.total += 1;
            totals.obligations += 1;
            totals.failed += 1;
            emit_obligation_record(outcome, options);
            Status::Failed
        }
        ObligationStatus::Unsupported => {
            totals.total += 1;
            totals.obligations += 1;
            totals.unsupported += 1;
            emit_obligation_record(outcome, options);
            Status::Unsupported
        }
    }
}

/// Render one obligation outcome as the additive NDJSON `{kind:"obligation"}`
/// record (or human-readable line). Emits the same field set and ordering as
/// `obligation_run::emit` -- including the `qualifiers` caveat array
/// (chelis#422) -- so a `.dp` obligation record is shaped exactly like a `.ch`
/// one. The one remaining difference is the `status` field: this Deep renderer
/// derives it directly from `outcome.status`, matching its own dispatcher
/// `render_obligation_outcome`, whereas the Surf path derives status from the
/// composite verdict via `obligation_display_status`. The two agree for every
/// outcome where `outcome.status` matches the composite-verdict status, which
/// is the case for SMT-proved obligations.
#[cfg(feature = "chelis-prove")]
fn emit_obligation_record(
    outcome: &chelis_prove::obligation_engine::ObligationOutcome,
    options: &ProveOptions<'_>,
) {
    use chelis_prove::obligation_engine::{ObligationStatus, ObligationTier};
    let status = match outcome.status {
        ObligationStatus::Passed => "passed",
        ObligationStatus::Failed => "failed",
        ObligationStatus::Unsupported => "unsupported",
        ObligationStatus::Error => "error",
    };
    if options.json {
        if outcome.status == ObligationStatus::Error {
            // Declaration errors carry no producer/name; emit a minimal
            // record so the count of kind:"obligation" stays accurate while
            // the reason names the offending producer.
            println!(
                "{}",
                json!({
                    "kind": "obligation",
                    "obligation_kind": "invariant_producer",
                    "status": "error",
                    "composite_verdict": outcome.composite_verdict.as_str(),
                    "assumptions": &outcome.assumptions,
                    "reason": outcome.reason,
                })
            );
            return;
        }
        let mut value = json!({
            "kind": "obligation",
            "obligation_kind": outcome.meta.obligation_kind,
            "source_type": outcome.meta.source_type,
            "producer": outcome.meta.producer,
            "name": outcome.name,
            "status": status,
            "composite_verdict": outcome.composite_verdict.as_str(),
            // chelis#422 (D2): full disclosed caveat set alongside the weakest
            // `composite_verdict` token.
            "qualifiers": outcome.disclosed_qualifiers(),
            "assumptions": &outcome.assumptions,
            "proof_tier": outcome.proof_tier.as_str(),
            "samples": outcome.samples,
            "seed": outcome.seed,
        });
        // chelis#436: the discharged proposition (the invariant predicate) travels
        // with the record so a consumer displays exactly what was discharged.
        if let Some(goal) = &outcome.goal {
            value["goal"] = json!(goal);
        }
        if outcome.proof_tier == ObligationTier::Smt {
            value["arith_model"] = json!("real");
        }
        if let Some(cx) = &outcome.counterexample {
            value["counterexample"] = cx.clone();
            value["shrink_steps"] = json!(outcome.shrink_steps);
        }
        if let Some(r) = &outcome.reason {
            value["reason"] = json!(r);
        }
        println!("{value}");
    } else {
        match outcome.status {
            ObligationStatus::Passed => println!(
                "obligation: {} -- proved ({})",
                outcome.name,
                outcome.proof_tier.as_str()
            ),
            ObligationStatus::Failed => println!(
                "obligation failure: {} ({} counterexample)",
                outcome.name,
                outcome.proof_tier.as_str()
            ),
            ObligationStatus::Unsupported => println!(
                "obligation unsupported: {}: {}",
                outcome.name,
                outcome.reason.clone().unwrap_or_default()
            ),
            ObligationStatus::Error => println!(
                "obligation error: {}",
                outcome.reason.clone().unwrap_or_default()
            ),
        }
    }
}

/// Count opaque types that carry a declared invariant in flattened Surf decls.
/// The capability path uses this to skip a no-op obligation collection pass;
/// non-smt builds also use it to warn that obligations were not SMT-verified.
fn count_invariant_opaque_surf(decls: &[Decl]) -> usize {
    decls
        .iter()
        .filter(|decl| {
            matches!(
                decl,
                Decl::TypeDef {
                    opaque: true,
                    invariant: Some(_),
                    ..
                }
            )
        })
        .count()
}

/// Deep twin of `count_invariant_opaque_surf`: a `deftype` whose metadata
/// carries both `opaque: true` and an `invariant` entry.
#[cfg(not(feature = "smt"))]
fn count_invariant_opaque_deep(exprs: &[DeepExpr]) -> usize {
    fn scan(expr: &DeepExpr, acc: &mut usize) {
        if let DeepExpr::List(list, _) = expr {
            let tag = list.elements.first().and_then(|head| match head {
                DeepExpr::Atom(DeepAtom::Symbol(sym), _) => Some(sym.as_str()),
                _ => None,
            });
            if tag == Some("deftype")
                && let Some(DeepExpr::Map(meta, _)) = list.elements.get(1)
            {
                let opaque = meta.entries.iter().any(|(key, value)| {
                    key == "opaque" && matches!(value, DeepExpr::Atom(DeepAtom::Bool(true), _))
                });
                let has_invariant = meta.entries.iter().any(|(key, _)| key == "invariant");
                if opaque && has_invariant {
                    *acc += 1;
                }
            }
            for child in &list.elements {
                scan(child, acc);
            }
        }
    }
    let mut acc = 0;
    for expr in exprs {
        scan(expr, &mut acc);
    }
    acc
}

/// Emit a one-line stderr warning (never touching stdout or the exit code)
/// when a non-smt build proves a module declaring invariant-carrying opaque
/// types, so a clean run is not mistaken for formally verified producer
/// obligations. CR2-5: this fires in EVERY non-`smt` build -- including a
/// `chelis-prove`-without-`smt` build, where the obligation machinery runs
/// but only at Tier C (fuzz), not cvc5 -- because the SMT verification the
/// flag promises is unavailable.
#[cfg(not(feature = "smt"))]
fn warn_obligations_skipped_without_smt(path: &Path, count: usize, options: &ProveOptions<'_>) {
    if count == 0 {
        return;
    }
    // Machine-facing (review residual-risk): a clean stdout summary alone must
    // not read as a verified proof run. Under `--json`, emit a record so a
    // consumer sees that producer obligations were NOT verified in this
    // non-smt build -- not only the stderr warning below.
    if options.json {
        println!(
            "{}",
            json!({
                "kind": "warning",
                "stage": "obligations",
                "skipped": count,
                "reason": "producer obligation verification requires the smt-enabled build \
                           (--features smt); obligations were not SMT-verified in this build",
            })
        );
    }
    eprintln!(
        "warning: producer obligation verification requires the smt-enabled build; {count} \
         invariant-carrying opaque type(s) in {} did not have their obligations SMT-verified. \
         Rebuild with --features smt to verify them.",
        path.display()
    );
}

fn prove_deep_property(
    exprs: &[DeepExpr],
    property: &DeepProperty,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    totals.total += 1;
    let samples_needed = options.samples.or(property.samples).unwrap_or(100);
    let max_attempts = options
        .max_attempts
        .unwrap_or_else(|| samples_needed.saturating_mul(100).max(samples_needed));
    let seed = options.effective_seed(property.seed);
    let mut rng = Lcg::new(seed);

    if let Some(reason) = unsupported_property_params(&property.params) {
        totals.unsupported += 1;
        emit_deep_record(options, property, "unsupported", 0, None, Some(reason), 0);
        return Status::Unsupported;
    }

    let mut accepted = 0usize;
    let mut attempts = 0usize;
    while accepted < samples_needed && attempts < max_attempts {
        attempts += 1;
        let sample = match sample_deep_property(property, &mut rng) {
            Ok(sample) => sample,
            Err(reason) => {
                totals.unsupported += 1;
                emit_deep_record(options, property, "unsupported", 0, None, Some(reason), 0);
                return Status::Unsupported;
            }
        };
        if !property.preconditions.is_empty() {
            match eval_deep_sample(exprs, property, &sample, true) {
                Ok(false) => continue,
                Ok(true) => {}
                Err(err) => {
                    totals.errors += 1;
                    emit_deep_error(options, property, &err);
                    return Status::Error;
                }
            }
        }
        accepted += 1;
        match eval_deep_sample(exprs, property, &sample, false) {
            Ok(true) => {}
            Ok(false) => {
                let (shrunk, shrink_steps) = shrink_deep_counterexample(exprs, property, sample);
                totals.failed += 1;
                emit_deep_record(
                    options,
                    property,
                    "failed",
                    accepted,
                    Some(counterexample_json(&shrunk)),
                    None,
                    shrink_steps,
                );
                return Status::Failed;
            }
            Err(err) => {
                totals.errors += 1;
                emit_deep_error(options, property, &err);
                return Status::Error;
            }
        }
    }

    if accepted < samples_needed {
        totals.errors += 1;
        emit_deep_error(
            options,
            property,
            &format!(
                "generator exhausted after {attempts} attempts before collecting {samples_needed} valid samples"
            ),
        );
        return Status::Error;
    }

    totals.passed += 1;
    emit_deep_record(options, property, "passed", accepted, None, None, 0);
    Status::Passed
}

fn discover_deep_properties(
    path: &Path,
    exprs: &[DeepExpr],
    only: Option<&str>,
) -> Result<Vec<DeepProperty>, String> {
    let mut out = Vec::new();
    for expr in exprs {
        discover_deep_properties_expr(path, expr, only, &mut out)?;
    }
    Ok(out)
}

fn discover_deep_properties_expr(
    path: &Path,
    expr: &DeepExpr,
    only: Option<&str>,
    out: &mut Vec<DeepProperty>,
) -> Result<(), String> {
    let DeepExpr::List(list, _) = expr else {
        return Ok(());
    };
    if list_tag(expr) == Some("def")
        && let Some(name) = list.elements.get(2).and_then(symbol_text)
        && let Some(meta) = list.elements.get(1).and_then(meta_map)
        && let Some(source_kind) = property_source_kind(meta, name)?
    {
        let fn_expr = list
            .elements
            .get(3)
            .ok_or_else(|| format!("property `{name}` def is missing a fn body"))?;
        let fn_params = deep_fn_params(fn_expr)
            .ok_or_else(|| format!("property `{name}` def body must be a callable `fn`"))?;
        let params = if let Some(params) = deep_property_params(meta) {
            if !params_match(&params, &fn_params) {
                return Err(format!(
                    "property `{name}` property_quantifiers must match fn parameters"
                ));
            }
            params
        } else if has_chelis_property_role(meta) {
            return Err(format!(
                "property `{name}` metadata must include `property_quantifiers`"
            ));
        } else {
            fn_params
        };
        if matches_filter(name, only) {
            out.push(DeepProperty {
                name: name.to_string(),
                source: path.to_path_buf(),
                source_kind,
                source_id: deep_string_meta(meta, "property_source_id").map(ToString::to_string),
                params,
                preconditions: deep_property_preconditions(meta).unwrap_or_default(),
                body: deep_fn_body(fn_expr)
                    .cloned()
                    .ok_or_else(|| format!("property `{name}` def body must be a callable `fn`"))?,
                samples: deep_int_meta(meta, "property_samples"),
                seed: deep_int_meta(meta, "property_seed").map(|value| value as u64),
            });
        }
    }
    for child in &list.elements {
        discover_deep_properties_expr(path, child, only, out)?;
    }
    Ok(())
}

fn has_chelis_property_role(meta: &MetaMap) -> bool {
    meta.entries
        .iter()
        .any(|(key, value)| key == "chelis_role" && string_value(value) == Some("property"))
}

fn has_legacy_property_role(meta: &MetaMap) -> bool {
    meta.entries.iter().any(|(key, value)| {
        key == "c_earchin_role" && string_value(value) == Some("property_witness")
    })
}

fn property_source_kind(meta: &MetaMap, name: &str) -> Result<Option<String>, String> {
    let has_chelis = has_chelis_property_role(meta);
    let has_legacy = has_legacy_property_role(meta);
    if !has_chelis && !has_legacy {
        return Ok(None);
    }
    let Some(kind) = deep_meta_value(meta, "property_source_kind").and_then(string_value) else {
        if has_legacy {
            return Ok(Some("bridge:c-earchin".to_string()));
        }
        return Err(format!(
            "property `{name}` metadata must include string `property_source_kind`"
        ));
    };
    if !matches!(kind, "user" | "bridge:c-earchin") {
        return Err(format!(
            "property `{name}` metadata has invalid property_source_kind `{kind}`"
        ));
    }
    Ok(Some(kind.to_string()))
}

fn meta_map(expr: &DeepExpr) -> Option<&MetaMap> {
    match expr {
        DeepExpr::Map(map, _) => Some(map),
        _ => None,
    }
}

fn deep_meta_value<'a>(meta: &'a MetaMap, key: &str) -> Option<&'a DeepExpr> {
    meta.entries
        .iter()
        .find_map(|(entry_key, value)| (entry_key == key).then_some(value))
}

fn deep_int_meta(meta: &MetaMap, key: &str) -> Option<usize> {
    match deep_meta_value(meta, key).and_then(deep_int_value) {
        Some(value) if value >= 0 => Some(value as usize),
        _ => None,
    }
}

fn deep_string_meta<'a>(meta: &'a MetaMap, key: &str) -> Option<&'a str> {
    deep_meta_value(meta, key).and_then(string_value)
}

fn deep_int_value(expr: &DeepExpr) -> Option<i64> {
    match expr {
        DeepExpr::Atom(DeepAtom::Int(value), _) => Some(*value),
        DeepExpr::List(list, _) if list_tag_from_list(list) == Some("lit") => {
            match list.elements.get(2) {
                Some(DeepExpr::Atom(DeepAtom::Int(value), _)) => Some(*value),
                _ => None,
            }
        }
        _ => None,
    }
}

fn deep_property_params(meta: &MetaMap) -> Option<Vec<Param>> {
    let DeepExpr::List(list, _) = deep_meta_value(meta, "property_quantifiers")? else {
        return None;
    };
    if list_tag_from_list(list) != Some("params") {
        return None;
    }
    let mut params = Vec::new();
    for child in list.elements.iter().skip(2) {
        let DeepExpr::List(param_list, span) = child else {
            continue;
        };
        let Some(name) = param_list.elements.first().and_then(symbol_text) else {
            continue;
        };
        let ty = param_list
            .elements
            .get(1)
            .and_then(meta_map)
            .and_then(|meta| deep_meta_value(meta, "type"))
            .and_then(type_expr_from_deep);
        params.push(Param {
            name: name.to_string(),
            ty,
            span: *span,
        });
    }
    Some(params)
}

fn deep_property_preconditions(meta: &MetaMap) -> Option<Vec<DeepExpr>> {
    let DeepExpr::List(list, _) = deep_meta_value(meta, "property_preconditions")? else {
        return None;
    };
    if list_tag_from_list(list) != Some("tuple") {
        return None;
    }
    Some(list.elements.iter().skip(2).cloned().collect())
}

fn type_expr_from_deep(expr: &DeepExpr) -> Option<TypeExpr> {
    let DeepExpr::List(list, span) = expr else {
        return None;
    };
    match list_tag_from_list(list)? {
        "t-prim" => list
            .elements
            .get(2)
            .and_then(symbol_text)
            .map(|name| TypeExpr::Named(name.to_string(), *span)),
        "t-tensor" => {
            let children = list.elements.iter().skip(2).collect::<Vec<_>>();
            let precision = children.last().and_then(|expr| {
                let DeepExpr::List(prim, _) = expr else {
                    return None;
                };
                (list_tag_from_list(prim) == Some("t-prim"))
                    .then(|| prim.elements.get(2).and_then(symbol_text))
                    .flatten()
            })?;
            let dims = children
                .iter()
                .take(children.len().saturating_sub(1))
                .map(|dim| match dim {
                    DeepExpr::List(dim_list, dim_span)
                        if list_tag_from_list(dim_list) == Some("d-lit") =>
                    {
                        dim_list.elements.get(2).and_then(|value| match value {
                            DeepExpr::Atom(DeepAtom::Int(value), _) => {
                                Some(TypeExpr::Named(value.to_string(), *dim_span))
                            }
                            _ => None,
                        })
                    }
                    DeepExpr::List(dim_list, dim_span)
                        if list_tag_from_list(dim_list) == Some("d-name") =>
                    {
                        dim_list
                            .elements
                            .get(2)
                            .and_then(symbol_text)
                            .map(|name| TypeExpr::Named(name.to_string(), *dim_span))
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            Some(TypeExpr::Tensor(dims, precision.to_string(), *span))
        }
        _ => None,
    }
}

fn deep_fn_body(expr: &DeepExpr) -> Option<&DeepExpr> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    if list_tag_from_list(list) != Some("fn") {
        return None;
    }
    list.elements.get(3)
}

fn deep_fn_params(expr: &DeepExpr) -> Option<Vec<Param>> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    if list_tag_from_list(list) != Some("fn") {
        return None;
    }
    let DeepExpr::List(params, _) = list.elements.get(2)? else {
        return None;
    };
    if list_tag_from_list(params) != Some("params") {
        return None;
    }
    let mut out = Vec::new();
    for child in params.elements.iter().skip(2) {
        let DeepExpr::List(param_list, span) = child else {
            return None;
        };
        let name = param_list.elements.first().and_then(symbol_text)?;
        let ty = param_list
            .elements
            .get(1)
            .and_then(meta_map)
            .and_then(|meta| deep_meta_value(meta, "type"))
            .and_then(type_expr_from_deep);
        out.push(Param {
            name: name.to_string(),
            ty,
            span: *span,
        });
    }
    Some(out)
}

fn params_match(left: &[Param], right: &[Param]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.name == right.name
                && left.ty.as_ref().and_then(surf_type_to_deep)
                    == right.ty.as_ref().and_then(surf_type_to_deep)
        })
}

fn surf_type_to_deep(ty: &TypeExpr) -> Option<DeepExpr> {
    match ty {
        TypeExpr::Named(name, _) => Some(deep_node("t-prim", vec![deep_symbol(name)])),
        TypeExpr::Tensor(dims, precision, _) => {
            let mut children = dims
                .iter()
                .map(|dim| match dim {
                    TypeExpr::Named(value, _) => value
                        .parse::<i64>()
                        .ok()
                        .map(|dim| deep_node("d-lit", vec![deep_int(dim)]))
                        .or_else(|| Some(deep_node("d-name", vec![deep_symbol(value)]))),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            children.push(deep_node("t-prim", vec![deep_symbol(precision)]));
            Some(deep_node("t-tensor", children))
        }
        _ => None,
    }
}

fn sample_deep_property(property: &DeepProperty, rng: &mut Lcg) -> Result<Sample, String> {
    let mut values = Vec::new();
    for param in &property.params {
        let ty = param
            .ty
            .as_ref()
            .ok_or_else(|| format!("{} is missing an explicit type", param.name))?;
        values.push(sample_value(&param.name, ty, rng)?);
    }
    Ok(Sample { values })
}

fn eval_deep_sample(
    exprs: &[DeepExpr],
    property: &DeepProperty,
    sample: &Sample,
    precondition: bool,
) -> Result<bool, String> {
    let root = if precondition {
        "__chelis_property_pre"
    } else {
        "__chelis_property_probe"
    };
    let mut source_exprs = exprs.to_vec();
    for value in &sample.values {
        if let Some((binding_name, _)) = &value.tensor_binding
            && let Some(ty) = property
                .params
                .iter()
                .find(|param| param.name == value.name)
                .and_then(|param| param.ty.as_ref())
                .and_then(surf_type_to_deep)
        {
            source_exprs.push(deep_node("defsig", vec![deep_symbol(binding_name), ty]));
        }
    }
    source_exprs.push(deep_node(
        "def",
        vec![
            deep_symbol(root),
            deep_sample_block_expr(property, sample, precondition),
        ],
    ));
    let source = chelis_deep::printer::print_canonical(&source_exprs);
    eval_bool_with_bindings(SourceKind::Deep, source, root, sample_bindings(sample))
}

fn deep_sample_block_expr(
    property: &DeepProperty,
    sample: &Sample,
    precondition: bool,
) -> DeepExpr {
    let body = if precondition {
        combine_deep_preconditions(&property.preconditions)
    } else {
        property.body.clone()
    };
    if sample.values.is_empty() {
        return body;
    }
    let mut bind_children = Vec::new();
    for value in &sample.values {
        bind_children.push(deep_symbol(&value.name));
        bind_children.push(value.deep_expr.clone());
    }
    deep_node("let", vec![deep_node("bind", bind_children), body])
}

fn combine_deep_preconditions(preconditions: &[DeepExpr]) -> DeepExpr {
    preconditions
        .iter()
        .cloned()
        .reduce(|left, right| deep_node("app", vec![deep_var("and"), left, right]))
        .unwrap_or_else(|| deep_lit(deep_bool(true), "bool"))
}

fn list_tag(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::List(list, _) => list_tag_from_list(list),
        _ => None,
    }
}

fn list_tag_from_list(list: &DeepList) -> Option<&str> {
    list.elements.first().and_then(symbol_text)
}

fn symbol_text(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(DeepAtom::Symbol(value), _) => Some(value),
        _ => None,
    }
}

fn string_value(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(DeepAtom::Str(value), _) => Some(value),
        _ => None,
    }
}

fn matches_filter(name: &str, only: Option<&str>) -> bool {
    let Some(pattern) = only else {
        return true;
    };
    if pattern == name {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    name.contains(pattern)
}

/// The CLI-local (no-`chelis-prove`) property path runs Tier C (fuzz) only --
/// SMT is not compiled in -- so a green here is a fuzz-only BASE pass, never an
/// SMT proof. It renders `fuzz_validated`, NEVER a `proven_*` badge: a
/// fuzz-only pass is not proven (chelis#422). The proven-flavored
/// `proven_modulo_fuzz_validated_contract` badge is reserved for an exact base
/// discharged modulo a fuzz-validated contract, which this path cannot
/// produce. Mirrors `chelis_prove::property_runner::base_verdict` for the fuzz
/// tier so a CLI-local green and a shared-runner fuzz-base green agree on the
/// same badge string.
fn simple_composite_verdict(status: &str, samples: usize) -> &'static str {
    match status {
        "passed" if samples > 0 => "fuzz_validated",
        "passed" => "unsupported",
        "failed" => "failed",
        _ => "unsupported",
    }
}

/// The disclosed `qualifiers:[...]` array for the CLI-local fuzz path (D2). A
/// fuzz-only green base discloses `fuzz_base`; any non-green outcome discloses
/// none. Mirrors `PropertyOutcome::disclosed_qualifiers` for the fuzz tier so
/// the two prove-JSON surfaces agree.
fn simple_qualifiers(status: &str, samples: usize) -> Vec<&'static str> {
    if status == "passed" && samples > 0 {
        vec!["fuzz_base"]
    } else {
        Vec::new()
    }
}

/// Mirror of `chelis_prove::composition::FUZZ_TOLERANCE`. The CLI-local
/// Tier-C-fuzz path is compiled in the default (no-`chelis-prove`) build,
/// where the `chelis-prove` crate is not linked and its constant cannot be
/// imported, so the value is duplicated here. The two must stay equal: the
/// `cli_fuzz_tolerance_matches_chelis_prove` test below (compiled only under
/// `chelis-prove`, where both are reachable) asserts the equality so they
/// cannot silently drift.
const CLI_FUZZ_TOLERANCE: f64 = 1e-10;

/// The non-vacuity assumption records a green CLI-local fuzz verdict carries
/// (WI-7). This Tier-C-only path (non-capability user properties and bridge
/// c-earchin properties) renders a green by rejection-sampling: it only counts
/// the `accepted` samples that satisfied every precondition, so an
/// unsatisfiable precondition set EXHAUSTS into a generator-exhaustion error
/// and never reaches a green here. A green over a non-empty precondition set is
/// therefore non-vacuous by construction -- the accepted samples ARE the
/// satisfiability witness -- and the green now CARRIES that evidence as an
/// established non-vacuity record instead of an empty assumption list, so no
/// green-rendering path reports a pass without recording the non-vacuity it
/// established. The shape mirrors the shared runner's
/// `fuzz_precondition_assumptions` so a CLI-local green and a shared-runner
/// green agree on the same module. A property with no preconditions has nothing
/// that could be vacuous, so it carries no assumption record (an empty list).
fn precondition_non_vacuity_assumptions(
    property_name: &str,
    status: &str,
    precondition_count: usize,
    samples: usize,
    seed: u64,
) -> serde_json::Value {
    // Only a genuine green (a pass with at least one witnessing sample) over a
    // non-empty precondition set records established non-vacuity. A failed,
    // unsupported, or zero-sample outcome is not a green and carries none.
    if precondition_count == 0 || status != "passed" || samples == 0 {
        return json!([]);
    }
    let name = format!("preconditions:{property_name}");
    json!([
        {
            "name": name,
            "discharge": {
                "method": "fuzz",
                "evidence": {
                    "status": "validated",
                    "property": property_name,
                    "samples": samples,
                    "seed": seed,
                    "tolerance": CLI_FUZZ_TOLERANCE,
                },
            },
            "non_vacuity": {
                "status": "established",
                "evidence": {
                    "method": "fuzz",
                    "result": "sat",
                    "accepted_samples": samples,
                    "seed": seed,
                },
            },
            // WI-8: the prover-side discharge tier, shaped exactly like the
            // shared runner's fuzz_precondition_assumptions stamp so a
            // CLI-local green and a shared-runner green agree byte-for-byte.
            "discharge_tier": {
                "engine": "fuzz-sampler",
                "guarantee": "fuzz",
                "source": name,
            },
        }
    ])
}

#[cfg(not(feature = "chelis-prove"))]
fn emit_record(
    options: &ProveOptions<'_>,
    property: &Property,
    status: &str,
    samples: usize,
    counterexample: Option<serde_json::Value>,
    reason: Option<String>,
    shrink_steps: usize,
) {
    if options.json {
        let seed = options.effective_seed(property.seed);
        let mut value = json!({
            "kind": "property",
            "name": property.name,
            "status": status,
            "composite_verdict": simple_composite_verdict(status, samples),
            "qualifiers": simple_qualifiers(status, samples),
            // chelis#436: the discharged proposition travels with the record,
            // rendered through the canonical Surf formatter. A guarded property's
            // proposition is its full `where`-guarded form, not its bare body
            // (MED-1): the prover discharges `pre => body`.
            "goal": chelis_surf::format::format_proposition(
                &property.params,
                &property.preconditions,
                &property.body,
            ),
            "assumptions": precondition_non_vacuity_assumptions(
                &property.name,
                status,
                property.preconditions.len(),
                samples,
                seed,
            ),
            "samples": samples,
            "seed": seed,
            "source": source_json(property, options),
        });
        if let Some(ref cx) = counterexample {
            value["counterexample"] = cx.clone();
            value["shrink_steps"] = json!(shrink_steps);
        }
        if let Some(ref r) = reason {
            value["reason"] = json!(r);
        }
        // chelis#489: structured failure summary for non-passing properties.
        if status != "passed" {
            value["failure_summary"] = json!({
                "status": status,
                "actual_tier": "fuzz",
                "seed": seed,
                "samples": samples,
            });
            if let Some(cx) = counterexample {
                value["failure_summary"]["counterexample"] = cx;
            }
        }
        println!("{value}");
    } else {
        match status {
            "passed" => println!("property: {} -- {samples}/{samples} passed", property.name),
            "failed" => println!(
                "property failure: {}\n  --> {}",
                property.name,
                property.source.display()
            ),
            "unsupported" => println!(
                "property unsupported: {}: {}",
                property.name,
                reason.unwrap_or_else(|| "unsupported in L2 v1".to_string())
            ),
            _ => {}
        }
    }
}

#[cfg(not(feature = "chelis-prove"))]
fn emit_error(options: &ProveOptions<'_>, property: &Property, message: &str) {
    if options.json {
        println!(
            "{}",
            json!({
                "kind": "property",
                "name": property.name,
                "status": "error",
                "composite_verdict": "unsupported",
                "assumptions": [],
                "samples": 0,
                "seed": options.effective_seed(property.seed),
                "reason": message,
                "source": source_json(property, options),
            })
        );
    } else {
        println!("property error: {}: {message}", property.name);
    }
}

fn emit_deep_record(
    options: &ProveOptions<'_>,
    property: &DeepProperty,
    status: &str,
    samples: usize,
    counterexample: Option<serde_json::Value>,
    reason: Option<String>,
    shrink_steps: usize,
) {
    if options.json {
        let seed = options.effective_seed(property.seed);
        let mut value = json!({
            "kind": "property",
            "name": property.name,
            "status": status,
            "composite_verdict": simple_composite_verdict(status, samples),
            "qualifiers": simple_qualifiers(status, samples),
            // chelis#436: the discharged proposition travels with the record,
            // rendered through the canonical Deep printer (flat) with
            // lowering/producer metadata stripped so a consumer sees the bare
            // proposition. A guarded property's proposition is the implication
            // `pre => body`, not the bare body (MED-1).
            "goal": chelis_deep::printer::print_expr_flat(&chelis_deep::ast::strip_metadata(
                &deep_proposition(&property.preconditions, &property.body),
            )),
            "assumptions": precondition_non_vacuity_assumptions(
                &property.name,
                status,
                property.preconditions.len(),
                samples,
                seed,
            ),
            "samples": samples,
            "seed": seed,
            "source": source_json_deep(property, options),
        });
        if let Some(ref cx) = counterexample {
            value["counterexample"] = cx.clone();
            value["shrink_steps"] = json!(shrink_steps);
        }
        if let Some(ref r) = reason {
            value["reason"] = json!(r);
        }
        // chelis#489: structured failure summary for non-passing properties.
        if status != "passed" {
            value["failure_summary"] = json!({
                "status": status,
                "actual_tier": "fuzz",
                "seed": seed,
                "samples": samples,
            });
            if let Some(cx) = counterexample {
                value["failure_summary"]["counterexample"] = cx;
            }
        }
        println!("{value}");
    } else {
        match status {
            "passed" => println!("property: {} -- {samples}/{samples} passed", property.name),
            "failed" => {
                if let Some(source) = bridge_source_details(property, options) {
                    println!(
                        "property failure: {}\n  --> {}:{}:{} {}\n  | {}",
                        property.name,
                        source.file,
                        source.line,
                        source.column,
                        source.id,
                        source.text
                    );
                } else {
                    println!(
                        "property failure: {}\n  --> {}",
                        property.name,
                        property.source.display()
                    );
                }
            }
            "unsupported" => println!(
                "property unsupported: {}: {}",
                property.name,
                reason.unwrap_or_else(|| "unsupported in L2 v1".to_string())
            ),
            _ => {}
        }
    }
}

fn emit_deep_error(options: &ProveOptions<'_>, property: &DeepProperty, message: &str) {
    if options.json {
        println!(
            "{}",
            json!({
                "kind": "property",
                "name": property.name,
                "status": "error",
                "composite_verdict": "unsupported",
                "assumptions": [],
                "samples": 0,
                "seed": options.effective_seed(property.seed),
                "reason": message,
                "source": source_json_deep(property, options),
            })
        );
    } else {
        if let Some(source) = bridge_source_details(property, options) {
            println!(
                "property error: {}: {message}\n  --> {}:{}:{} {}\n  | {}",
                property.name, source.file, source.line, source.column, source.id, source.text
            );
        } else {
            println!("property error: {}: {message}", property.name);
        }
    }
}

#[cfg(not(feature = "chelis-prove"))]
fn source_json(property: &Property, options: &ProveOptions<'_>) -> serde_json::Value {
    if property.source.extension().and_then(|ext| ext.to_str()) == Some("dp") {
        let spans = options
            .spans
            .map(Path::to_path_buf)
            .or_else(|| sibling_spans_path(&property.source));
        json!({
            "kind": "bridge:c-earchin",
            "spans": spans.map(|path| path.display().to_string()),
        })
    } else {
        json!({
            "kind": "surf",
            "file": property.source.display().to_string(),
        })
    }
}

fn source_json_deep(property: &DeepProperty, options: &ProveOptions<'_>) -> serde_json::Value {
    if property.source_kind == "user" {
        return json!({
            "kind": "user",
            "file": property.source.display().to_string(),
        });
    }
    let spans = options
        .spans
        .map(Path::to_path_buf)
        .or_else(|| sibling_spans_path(&property.source));
    let mut value = json!({
        "kind": "bridge:c-earchin",
        "spans": spans.map(|path| path.display().to_string()),
    });
    if let Some(source) = bridge_source_details(property, options) {
        value["requirement"] = json!({
            "id": source.id,
            "file": source.file,
            "line": source.line,
            "column": source.column,
            "text": source.text,
        });
    }
    value
}

#[derive(Debug, Clone)]
struct BridgeSourceDetails {
    id: String,
    file: String,
    line: usize,
    column: usize,
    text: String,
}

fn bridge_source_details(
    property: &DeepProperty,
    options: &ProveOptions<'_>,
) -> Option<BridgeSourceDetails> {
    if property.source_kind != "bridge:c-earchin" {
        return None;
    }
    let spans_path = options
        .spans
        .map(Path::to_path_buf)
        .or_else(|| sibling_spans_path(&property.source))?;
    let source = fs::read_to_string(spans_path).ok()?;
    let manifest = serde_json::from_str::<serde_json::Value>(&source).ok()?;
    let entry = manifest.get("spans")?.as_array()?.iter().find(|entry| {
        let ears_id = entry.get("ears_id").and_then(|value| value.as_str());
        let deep_node_id = entry.get("deep_node_id").and_then(|value| value.as_str());
        property
            .source_id
            .as_deref()
            .is_some_and(|id| ears_id == Some(id))
            || deep_node_id == Some(property.name.as_str())
    })?;
    let ears = entry.get("ears")?;
    Some(BridgeSourceDetails {
        id: entry.get("ears_id")?.as_str()?.to_string(),
        file: entry.get("ears_file")?.as_str()?.to_string(),
        line: ears.get("start_line")?.as_u64()? as usize,
        column: ears.get("start_column")?.as_u64()? as usize,
        text: entry.get("ears_text")?.as_str()?.to_string(),
    })
}

fn sibling_spans_path(path: &Path) -> Option<PathBuf> {
    let stem = path.file_stem()?.to_str()?;
    let candidate = path.with_file_name(format!("{stem}.spans.json"));
    candidate.exists().then_some(candidate)
}

#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    fn next_i64(&mut self, min: i64, max: i64) -> i64 {
        let span = (max - min + 1) as u64;
        min + (self.next_u64() % span) as i64
    }

    fn next_f64(&mut self, min: f64, max: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        min + (max - min) * unit
    }
}

/// Resolve the effective package root: explicit `--package` wins, otherwise
/// auto-detect by walking ancestor directories of the input file.
#[cfg(feature = "chelis-prove")]
fn resolve_package_root(input: &Path, explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(root) = explicit {
        return Some(root.to_path_buf());
    }
    let start = if input.is_file() {
        input.parent().unwrap_or(input)
    } else {
        input
    };
    chelis_reef::find_package_root_for_dir(start).ok().flatten()
}

/// Emit JSON describing this binary's prove capabilities (#488).
pub fn prove_capabilities() -> serde_json::Value {
    let smt_available = cfg!(feature = "smt");
    let beacon_available = std::env::var("CHELIS_BEACON_BIN").is_ok();
    let dispatcher_available = cfg!(feature = "chelis-prove");
    let obligation_engine_available = cfg!(feature = "chelis-prove");
    json!({
        "schema_version": 1,
        "prove_json_schema_version": 1,
        "supported_tiers": ["type_system", "smt", "fuzz"],
        "smt_available": smt_available,
        "beacon_available": beacon_available,
        "dispatcher_available": dispatcher_available,
        "obligation_engine_available": obligation_engine_available,
        "supported_flags": ["--json", "--only", "--samples", "--seed", "--tier", "--smt-timeout", "--package"],
        "engine_registry": ["tier_a_type_system", "tier_b_smt", "tier_c_fuzz", "beacon_shim"]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combine_status_failed_outranks_unsupported() {
        // RT #9: a DISPROVED property (Failed, exit 1) must win over a
        // co-occurring Unsupported one (exit 2), so a CI gate keying on exit 1
        // sees the falsification instead of a masking exit 2.
        assert_eq!(
            combine_status(Status::Failed, Status::Unsupported),
            Status::Failed
        );
        assert_eq!(
            combine_status(Status::Unsupported, Status::Failed),
            Status::Failed
        );
        // The rest of the worst-wins ladder is unchanged: Error dominates all,
        // Unsupported still beats Passed, Passed is the identity.
        assert_eq!(combine_status(Status::Error, Status::Failed), Status::Error);
        assert_eq!(
            combine_status(Status::Error, Status::Unsupported),
            Status::Error
        );
        assert_eq!(
            combine_status(Status::Unsupported, Status::Passed),
            Status::Unsupported
        );
        assert_eq!(
            combine_status(Status::Failed, Status::Passed),
            Status::Failed
        );
        assert_eq!(
            combine_status(Status::Passed, Status::Passed),
            Status::Passed
        );
        // Per-status exit codes are unchanged; only the combine precedence moved.
        assert_eq!(Status::Failed.exit_code(), 1);
        assert_eq!(Status::Unsupported.exit_code(), 2);
    }

    // The CLI-local fuzz tolerance constant must equal the shared runner's
    // `chelis_prove::composition::FUZZ_TOLERANCE`. Compiled only under
    // `chelis-prove`, the single build where both are linked, so the mirror
    // cannot drift from the source of truth without this test failing.
    #[cfg(feature = "chelis-prove")]
    #[test]
    fn cli_fuzz_tolerance_matches_chelis_prove() {
        assert_eq!(
            CLI_FUZZ_TOLERANCE,
            chelis_prove::composition::FUZZ_TOLERANCE
        );
    }
}
