use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr, List as DeepList, MetaMap};
use chelis_surf::ast::{BinOp, Decl, Expr, LetBinding, LetPattern, Literal, Param, TypeExpr};
use serde_json::json;
use walkdir::WalkDir;

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
struct Property {
    name: String,
    source: PathBuf,
    params: Vec<Param>,
    preconditions: Vec<Expr>,
    #[allow(dead_code)]
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
    for input in inputs {
        let status = match input.extension().and_then(|ext| ext.to_str()) {
            Some("ch") => prove_surf_file(&input, &options, &mut totals)?,
            Some("dp") => prove_deep_file(&input, &options, &mut totals)?,
            _ => Status::Passed,
        };
        worst = combine_status(worst, status);
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

fn is_single_explicit_deep_input(path: Option<&Path>, inputs: &[PathBuf]) -> bool {
    matches!(path, Some(path) if path.is_file())
        && inputs.len() == 1
        && inputs[0].extension().and_then(|ext| ext.to_str()) == Some("dp")
}

fn combine_status(lhs: Status, rhs: Status) -> Status {
    use Status::*;
    match (lhs, rhs) {
        (Error, _) | (_, Error) => Error,
        (Unsupported, _) | (_, Unsupported) => Unsupported,
        (Failed, _) | (_, Failed) => Failed,
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
    let properties = collect_surf_properties(path, &flat, options.only);
    let mut file_status = Status::Passed;
    for property in properties {
        let status = prove_surf_property(&flat, &property, options, totals);
        file_status = combine_status(file_status, status);
    }
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

fn property_samples(options: &[chelis_surf::ast::PropertyOption]) -> Option<usize> {
    options.iter().find_map(|option| match option {
        chelis_surf::ast::PropertyOption::Samples(Expr::Lit(Literal::Int(value), _), _) => {
            (*value >= 0).then_some(*value as usize)
        }
        _ => None,
    })
}

fn property_seed(options: &[chelis_surf::ast::PropertyOption]) -> Option<u64> {
    options.iter().find_map(|option| match option {
        chelis_surf::ast::PropertyOption::Seed(Expr::Lit(Literal::Int(value), _), _) => {
            (*value >= 0).then_some(*value as u64)
        }
        _ => None,
    })
}

fn prove_surf_property(
    decls: &[Decl],
    property: &Property,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    totals.total += 1;

    // Tier B: attempt SMT proof when --tier auto
    #[cfg(feature = "chelis-prove")]
    if (options.tier == "auto" || options.tier == "smt-only")
        && let Some(postcondition) = surf_expr_to_smt(
            &property.body,
            &InlineCtx {
                decls,
                depth: 0,
                max_depth: 3,
                call_stack: vec![],
            },
        )
    {
        let variables: Vec<(String, chelis_prove::solver::SmtSort)> = property
            .params
            .iter()
            .filter_map(|p| {
                let sort = match p.ty.as_ref()? {
                    TypeExpr::Named(name, _) => match name.as_str() {
                        "f32" | "f64" => chelis_prove::solver::SmtSort::Real,
                        "int32" | "int64" => chelis_prove::solver::SmtSort::Int,
                        "bool" => chelis_prove::solver::SmtSort::Bool,
                        _ => return None,
                    },
                    _ => return None,
                };
                Some((p.name.clone(), sort))
            })
            .collect();
        if variables.len() == property.params.len() {
            let preconditions: Vec<chelis_prove::solver::SmtExpr> = property
                .preconditions
                .iter()
                .filter_map(|e| {
                    surf_expr_to_smt(
                        e,
                        &InlineCtx {
                            decls,
                            depth: 0,
                            max_depth: 3,
                            call_stack: vec![],
                        },
                    )
                })
                .collect();
            if preconditions.len() == property.preconditions.len() {
                let smt_prop = chelis_prove::tier_b::SmtProperty {
                    variables,
                    preconditions,
                    postcondition,
                };
                if let chelis_prove::Inlineability::Inlineable =
                    chelis_prove::classify_inlineability(&smt_prop.postcondition)
                {
                    match chelis_prove::solve_property(&smt_prop, options.smt_timeout_ms) {
                        chelis_prove::tier_b::TierBResult::Proved => {
                            totals.passed += 1;
                            if options.json {
                                println!(
                                    "{}",
                                    serde_json::json!({"kind":"property","name":property.name,"status":"passed","proof_tier":"smt","samples":0,"seed":options.effective_seed(property.seed)})
                                );
                            } else {
                                println!("property: {} -- proved (smt)", property.name);
                            }
                            return Status::Passed;
                        }
                        chelis_prove::tier_b::TierBResult::Disproved(_model) => {
                            totals.failed += 1;
                            if options.json {
                                println!(
                                    "{}",
                                    serde_json::json!({"kind":"property","name":property.name,"status":"failed","proof_tier":"smt","samples":0,"seed":options.effective_seed(property.seed)})
                                );
                            } else {
                                println!(
                                    "property failure: {} (smt counterexample)",
                                    property.name
                                );
                            }
                            return Status::Failed;
                        }
                        chelis_prove::tier_b::TierBResult::Timeout
                        | chelis_prove::tier_b::TierBResult::Unknown => {
                            // Fall through to fuzz (Tier C)
                        }
                    }
                }
            }
        }
    }

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
                totals.failed += 1;
                emit_record(
                    options,
                    property,
                    "failed",
                    accepted,
                    Some(counterexample_json(&sample)),
                    None,
                    0,
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

fn unsupported_type(ty: &TypeExpr) -> Option<String> {
    match ty {
        TypeExpr::Named(name, _)
            if matches!(
                name.as_str(),
                "bool" | "int32" | "int64" | "f32" | "f64" | "string"
            ) =>
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
        TypeExpr::Named(type_name, _) if type_name == "int32" || type_name == "int64" => {
            let value = rng.next_i64(-1000, 1000);
            let lit = Expr::Lit(Literal::Int(value), sp);
            if type_name == "int64" {
                Ok(scalar_sample(
                    name,
                    cast_expr(lit, "int64"),
                    deep_lit(deep_int(value), "int64"),
                    json!(value),
                ))
            } else {
                Ok(scalar_sample(
                    name,
                    lit,
                    deep_lit(deep_int(value), "int32"),
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
    let properties = discover_deep_properties(path, &exprs, options.only)?;
    let mut file_status = Status::Passed;
    for property in properties {
        let status = prove_deep_property(&exprs, &property, options, totals);
        file_status = combine_status(file_status, status);
    }
    Ok(file_status)
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
                totals.failed += 1;
                emit_deep_record(
                    options,
                    property,
                    "failed",
                    accepted,
                    Some(counterexample_json(&sample)),
                    None,
                    0,
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
        let mut value = json!({
            "kind": "property",
            "name": property.name,
            "status": status,
            "samples": samples,
            "seed": options.effective_seed(property.seed),
            "source": source_json(property, options),
        });
        if let Some(counterexample) = counterexample {
            value["counterexample"] = counterexample;
            value["shrink_steps"] = json!(shrink_steps);
        }
        if let Some(reason) = reason {
            value["reason"] = json!(reason);
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

fn emit_error(options: &ProveOptions<'_>, property: &Property, message: &str) {
    if options.json {
        println!(
            "{}",
            json!({
                "kind": "property",
                "name": property.name,
                "status": "error",
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
        let mut value = json!({
            "kind": "property",
            "name": property.name,
            "status": status,
            "samples": samples,
            "seed": options.effective_seed(property.seed),
            "source": source_json_deep(property, options),
        });
        if let Some(counterexample) = counterexample {
            value["counterexample"] = counterexample;
            value["shrink_steps"] = json!(shrink_steps);
        }
        if let Some(reason) = reason {
            value["reason"] = json!(reason);
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

#[cfg(feature = "chelis-prove")]
use chelis_prove::convert::{InlineCtx, surf_expr_to_smt};

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
