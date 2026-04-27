use std::collections::HashMap;
use std::fs;

use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::lower::top_level_lowering_map;
use chelis_types::{BUILTIN_NAMES, CheckedProgram, types::Prim};

use crate::schema::{DictEntryValue, ExecutionValue, TensorValue};

#[derive(Debug, Clone)]
pub(crate) struct RuntimeTensorValue {
    pub(crate) value: IrTensorValue,
    pub(crate) precision: Prim,
}

#[derive(Debug, Clone)]
pub(crate) enum RuntimeValue {
    Tensor(RuntimeTensorValue),
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<RuntimeValue>),
    Dict(Vec<(RuntimeValue, RuntimeValue)>),
    Tuple(Vec<RuntimeValue>),
    Adt {
        ctor: String,
        fields: Vec<RuntimeValue>,
        field_names: Option<Vec<String>>,
    },
    MappedFile(Vec<u8>),
    Closure {
        params: Vec<String>,
        body: Expr,
        env: HashMap<String, RuntimeValue>,
    },
    Unit,
}

#[derive(Debug, Default)]
pub(crate) struct RuntimeOutcome {
    pub(crate) host_bindings: HashMap<String, RuntimeValue>,
    pub(crate) transcript: Vec<String>,
}

#[cfg(test)]
pub(crate) fn evaluate_host_program(
    program: &CheckedProgram,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
) -> Result<RuntimeOutcome, String> {
    evaluate_host_program_filtered(program, tensor_bindings, None)
}

/// Evaluate top-level non-fn bindings. When `selected_roots` is `Some`, only
/// bindings whose names appear in the filter are *eagerly* evaluated. Other
/// top-level bindings stay registered in `top_level_defs` so the body of a
/// selected binding can lazily resolve references to them via
/// `resolve_top_level`. This is what lets `chelis test` share a single
/// compile across every test in a file: compile once with N synthesized
/// `__chelis_test_k = test_k()` bindings, then run N eval passes each
/// selecting one root — without each pass paying for the other N-1 tests
/// running as module init.
pub(crate) fn evaluate_host_program_filtered(
    program: &CheckedProgram,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
    selected_roots: Option<&[String]>,
) -> Result<RuntimeOutcome, String> {
    evaluate_host_program_with_library(program, &[], None, tensor_bindings, selected_roots)
}

/// Phase G' — host-runtime entry that seeds the `top_level_defs` table
/// with library defs in addition to the new-code program. This is the
/// host-side parity counterpart to `lower_program_with_context`: when
/// new code calls a library function (e.g. `Std.Time.is_leap_year`),
/// `eval_app` looks up that name through `lookup_top_level_def`, and
/// the function body must be reachable. Pre-Phase-G' the runtime only
/// saw `program.exprs()`, so library names errored as `unknown runtime
/// name`.
///
/// Library defs are registered FIRST, then new-code defs, so on a name
/// collision the new-code def shadows the library def — mirroring the
/// type-env stacking semantics in `check_phase0e_with_context`.
///
/// `library_lowered_names` is the optional library-side
/// lowered-vs-host classification, threaded through so a library def
/// that the lowering pass identifies as "lives in the tensor DAG, not
/// in the host runtime" stays out of the host runtime's eager-eval
/// list. The new code's lowering map (computed locally below) merges
/// on top.
pub(crate) fn evaluate_host_program_with_library(
    program: &CheckedProgram,
    library_exprs: &[Expr],
    library_lowered_names: Option<&HashMap<String, bool>>,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
    selected_roots: Option<&[String]>,
) -> Result<RuntimeOutcome, String> {
    // Lowered classification: start with library's (if provided), then
    // overlay the new-code program's. New-code wins on shadow.
    let new_lowered_names = top_level_lowering_map(program.exprs(), program.type_env());
    let mut lowered_names: HashMap<String, bool> = HashMap::new();
    if let Some(lib) = library_lowered_names {
        lowered_names.extend(lib.iter().map(|(k, v)| (k.clone(), *v)));
    }
    lowered_names.extend(new_lowered_names);

    // ADT field map covers both library and new-code constructors so a
    // record-pattern match on a library ADT in new code resolves field
    // names correctly.
    let mut adt_fields = collect_adt_ctor_fields(library_exprs);
    adt_fields.extend(collect_adt_ctor_fields(program.exprs()));

    let mut top_level_defs = HashMap::new();
    let mut top_level_order = Vec::new();

    // Register library defs FIRST. New-code defs will overwrite on
    // name collision below — matching the Phase C type-env shadow rule
    // (new code wins).
    register_top_level_defs(
        library_exprs,
        &lowered_names,
        selected_roots,
        &mut top_level_defs,
        &mut top_level_order,
        /* register_runtime_order = */ false,
    );
    // Register new-code defs. New-code is the only source of eager
    // module-init bindings in `top_level_order` — library was already
    // checked + lowered at context-build time and any side effects
    // would have happened then; re-running them on every per-test
    // worker is exactly the regression we're fixing.
    register_top_level_defs(
        program.exprs(),
        &lowered_names,
        selected_roots,
        &mut top_level_defs,
        &mut top_level_order,
        /* register_runtime_order = */ true,
    );

    let mut ctx = EvalContext {
        bindings: HashMap::new(),
        top_level_defs,
        adt_fields,
        tensor_bindings,
        transcript: Vec::new(),
        resolving_top_levels: Vec::new(),
        random_seed: None,
        random_counter: 0,
    };

    for name in top_level_order {
        let _ = ctx.resolve_top_level(&name)?;
    }

    Ok(RuntimeOutcome {
        host_bindings: ctx.bindings,
        transcript: ctx.transcript,
    })
}

fn register_top_level_defs(
    exprs: &[Expr],
    lowered_names: &HashMap<String, bool>,
    selected_roots: Option<&[String]>,
    top_level_defs: &mut HashMap<String, Expr>,
    top_level_order: &mut Vec<String>,
    register_runtime_order: bool,
) {
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("def") {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        top_level_defs.insert(name.to_string(), body.clone());
        if !register_runtime_order {
            continue;
        }
        let is_fn = matches!(body, Expr::List(body_list, _) if tag(body_list) == Some("fn"));
        if !is_fn && !lowered_names.get(name).copied().unwrap_or(false) {
            let selected = match selected_roots {
                None => true,
                Some(filter) => filter.iter().any(|s| s == name),
            };
            if selected {
                top_level_order.push(name.to_string());
            }
        }
    }
}

/// Compute a lowered-vs-host classification map for a slice of
/// library exprs, using its own type-env. Phase G' threads this from
/// the `CompiledContext`'s `library_checked` into the host runtime so
/// the new-code lowering map merges with library state instead of
/// re-deriving the wrong answer for library names that shadow
/// builtins.
pub(crate) fn library_lowered_names(
    library_exprs: &[Expr],
    library_type_env: &HashMap<String, Expr>,
) -> HashMap<String, bool> {
    top_level_lowering_map(library_exprs, library_type_env)
}

fn top_level_items(exprs: &[Expr]) -> Vec<&Expr> {
    let mut out = Vec::new();
    for expr in exprs {
        collect_top_level_items(expr, &mut out);
    }
    out
}

fn collect_top_level_items<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    if tag(list) == Some("module") {
        for child in list.elements.iter().skip(3) {
            collect_top_level_items(child, out);
        }
        return;
    }
    out.push(expr);
}

pub(crate) fn runtime_value_to_schema(value: &RuntimeValue) -> Result<ExecutionValue, String> {
    Ok(match value {
        RuntimeValue::Tensor(tensor) => ExecutionValue::Tensor {
            value: TensorValue {
                shape: tensor.value.shape.clone(),
                data: tensor.value.data.clone(),
            },
        },
        RuntimeValue::Int(value) => ExecutionValue::Int64 { value: *value },
        RuntimeValue::Float(value) => ExecutionValue::Float64 { value: *value },
        RuntimeValue::Bool(value) => ExecutionValue::Bool { value: *value },
        RuntimeValue::String(value) => ExecutionValue::String {
            value: value.clone(),
        },
        RuntimeValue::List(items) => ExecutionValue::List {
            value: items
                .iter()
                .map(runtime_value_to_schema)
                .collect::<Result<Vec<_>, _>>()?,
        },
        RuntimeValue::Dict(entries) => ExecutionValue::Dict {
            entries: entries
                .iter()
                .map(|(key, value)| {
                    Ok(DictEntryValue {
                        key: runtime_value_to_schema(key)?,
                        value: runtime_value_to_schema(value)?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?,
        },
        RuntimeValue::Tuple(items) => ExecutionValue::Tuple {
            value: items
                .iter()
                .map(runtime_value_to_schema)
                .collect::<Result<Vec<_>, _>>()?,
        },
        RuntimeValue::Adt { ctor, fields, .. } => ExecutionValue::Adt {
            ctor: ctor.clone(),
            fields: fields
                .iter()
                .map(runtime_value_to_schema)
                .collect::<Result<Vec<_>, _>>()?,
        },
        RuntimeValue::MappedFile(_) => {
            return Err(
                "MappedFile values are not serializable on machine-facing APIs".to_string(),
            );
        }
        RuntimeValue::Closure { .. } => ExecutionValue::String {
            value: "<closure>".to_string(),
        },
        RuntimeValue::Unit => ExecutionValue::Unit,
    })
}

pub(crate) fn lookup_runtime_value_for_root(
    name: &str,
    host_bindings: &HashMap<String, RuntimeValue>,
    tensor_bindings: &HashMap<String, RuntimeTensorValue>,
) -> Option<RuntimeValue> {
    if let Some(value) = tensor_bindings.get(name) {
        return Some(RuntimeValue::Tensor(value.clone()));
    }

    let mut parts = name.split('.');
    let head = parts.next()?;
    let mut value = host_bindings.get(head)?.clone();
    for part in parts {
        let index = part.parse::<usize>().ok()?;
        value = match value {
            RuntimeValue::Tuple(items) => items.get(index)?.clone(),
            _ => return None,
        };
    }
    Some(value)
}

struct EvalContext<'a> {
    bindings: HashMap<String, RuntimeValue>,
    top_level_defs: HashMap<String, Expr>,
    adt_fields: HashMap<String, Vec<String>>,
    tensor_bindings: &'a HashMap<String, RuntimeTensorValue>,
    transcript: Vec<String>,
    resolving_top_levels: Vec<String>,
    random_seed: Option<u64>,
    random_counter: u64,
}

impl<'a> EvalContext<'a> {
    fn resolve_top_level(&mut self, name: &str) -> Result<RuntimeValue, String> {
        if let Some(value) = self.bindings.get(name) {
            return Ok(value.clone());
        }
        let Some((resolved_name, expr)) = self.lookup_top_level_def(name) else {
            return Err(format!("unknown runtime name `{name}`"));
        };
        if self
            .resolving_top_levels
            .iter()
            .any(|existing| existing == &resolved_name)
        {
            return Err(format!("cyclic top-level runtime definition `{name}`"));
        }
        self.resolving_top_levels.push(resolved_name.clone());
        let value = self.eval_expr(&expr)?;
        self.resolving_top_levels.pop();
        self.bindings.insert(resolved_name.clone(), value.clone());
        if resolved_name != name {
            self.bindings.insert(name.to_string(), value.clone());
        }
        Ok(value)
    }

    fn lookup_top_level_def(&self, name: &str) -> Option<(String, Expr)> {
        self.top_level_defs
            .get(name)
            .cloned()
            .map(|expr| (name.to_string(), expr))
            .or_else(|| {
                let mut matches = self.top_level_defs.iter().filter_map(|(key, value)| {
                    terminal_name_matches(key, name).then_some((key, value))
                });
                let (key, value) = matches.next()?;
                matches
                    .next()
                    .is_none()
                    .then_some((key.clone(), value.clone()))
            })
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<RuntimeValue, String> {
        match expr {
            Expr::Atom(_, _) => Err("bare atom is not a runtime expression".to_string()),
            Expr::Map(_, _) => Ok(RuntimeValue::Unit),
            Expr::MetaExpr(meta, _) => self.eval_expr(&meta.expr),
            Expr::List(list, _) => self.eval_list(list),
        }
    }

    fn eval_list(&mut self, list: &List) -> Result<RuntimeValue, String> {
        match tag(list) {
            Some("lit") => self.eval_lit(list),
            Some("var") => self.eval_var(list),
            Some("app") => self.eval_app(list),
            Some("if") => self.eval_if(list),
            Some("let") => self.eval_let(list),
            Some("tuple") => Ok(RuntimeValue::Tuple(
                children(list)
                    .iter()
                    .map(|child| self.eval_expr(child))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            Some("copy") => {
                let value = self.eval_expr(
                    children(list)
                        .first()
                        .ok_or_else(|| "copy missing value".to_string())?,
                )?;
                match value {
                    RuntimeValue::Tensor(tensor) => Ok(RuntimeValue::Tensor(tensor)),
                    other => Err(format!("copy expects tensor input, got {other:?}")),
                }
            }
            Some("borrow") => {
                // The IR lower path treats `borrow` as identity
                // (chelis-ir/src/lower.rs::lower_identity); mirror that
                // here so `&t` syntax type-checks AND evaluates.
                self.eval_expr(
                    children(list)
                        .first()
                        .ok_or_else(|| "borrow missing value".to_string())?,
                )
            }
            Some("record") => self.eval_record(list),
            Some("access") => self.eval_access(list),
            Some("tuple-get") => self.eval_tuple_get(list),
            Some("match") => self.eval_match(list),
            Some("fn") => self.eval_fn(list),
            Some("pipe") => self.eval_pipe(list),
            Some("cast") => self.eval_cast(list),
            Some("handle-effect") => {
                let kids = children(list);
                let effect = get_meta(list)
                    .and_then(|meta| {
                        meta.entries
                            .iter()
                            .find(|(key, _)| key == "effect")
                            .and_then(|(_, value)| symbol_name(value))
                    })
                    .unwrap_or_default();
                if effect == "random" {
                    let seed = self.eval_expr(
                        kids.first()
                            .ok_or_else(|| "handle-effect missing seed".to_string())?,
                    )?;
                    let seed = match seed {
                        RuntimeValue::Int(value) => value as u64,
                        other => {
                            return Err(format!("with seed expects int seed, got {other:?}"));
                        }
                    };
                    let saved_seed = self.random_seed;
                    let saved_counter = self.random_counter;
                    self.random_seed = Some(seed);
                    self.random_counter = 0;
                    let value = self.eval_expr(
                        kids.get(1)
                            .ok_or_else(|| "handle-effect missing body".to_string())?,
                    );
                    self.random_seed = saved_seed;
                    self.random_counter = saved_counter;
                    value
                } else {
                    self.eval_expr(
                        kids.get(1)
                            .ok_or_else(|| "handle-effect missing body".to_string())?,
                    )
                }
            }
            other => Err(format!(
                "host runtime does not support `{}`",
                other.unwrap_or("?")
            )),
        }
    }

    fn eval_record(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let ctor = kids
            .first()
            .and_then(symbol_name)
            .ok_or_else(|| "record missing constructor name".to_string())?;
        let mut fields_by_name = HashMap::new();
        let mut source_order = Vec::new();
        for field in kids.iter().skip(1) {
            let Some(field_list) = as_list(field) else {
                continue;
            };
            if tag(field_list) != Some("kv") {
                continue;
            }
            let field_kids = children(field_list);
            let Some(name) = field_kids.first().and_then(symbol_name) else {
                continue;
            };
            let value = self.eval_expr(
                field_kids
                    .get(1)
                    .ok_or_else(|| "record field missing value".to_string())?,
            )?;
            source_order.push(name.to_string());
            fields_by_name.insert(name.to_string(), value);
        }
        let declared = self
            .adt_fields
            .get(ctor)
            .cloned()
            .unwrap_or_else(|| source_order.clone());
        let mut ordered = Vec::with_capacity(declared.len());
        for field_name in declared {
            let value = fields_by_name.remove(&field_name).ok_or_else(|| {
                format!("record `{ctor}` missing field `{field_name}` at runtime")
            })?;
            ordered.push(value);
        }
        if let Some(extra) = fields_by_name.keys().next() {
            return Err(format!(
                "record `{ctor}` has unknown field `{extra}` at runtime"
            ));
        }
        Ok(RuntimeValue::Adt {
            ctor: ctor.to_string(),
            fields: ordered,
            field_names: Some(source_order),
        })
    }

    fn eval_access(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let target = self.eval_expr(
            kids.first()
                .ok_or_else(|| "access missing target".to_string())?,
        )?;
        let field = kids
            .get(1)
            .and_then(symbol_name)
            .ok_or_else(|| "access missing field".to_string())?;
        match target {
            RuntimeValue::Adt {
                ctor,
                fields,
                field_names,
            } => {
                let declared = self
                    .adt_fields
                    .get(&ctor)
                    .cloned()
                    .or(field_names)
                    .ok_or_else(|| format!("unknown record constructor `{ctor}`"))?;
                let Some(index) = declared.iter().position(|name| name == field) else {
                    return Err(format!("record `{ctor}` has no field `{field}`"));
                };
                fields
                    .get(index)
                    .cloned()
                    .ok_or_else(|| format!("record `{ctor}` missing field `{field}`"))
            }
            other => Err(format!("field access expects record value, got {other:?}")),
        }
    }

    fn eval_lit(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let value = children(list)
            .first()
            .ok_or_else(|| "lit missing value".to_string())?;
        match value {
            Expr::Atom(Atom::Int(value), _) => Ok(RuntimeValue::Int(*value)),
            Expr::Atom(Atom::Float(value), _) => Ok(RuntimeValue::Float(*value)),
            Expr::Atom(Atom::Bool(value), _) => Ok(RuntimeValue::Bool(*value)),
            Expr::Atom(Atom::Str(value), _) => Ok(RuntimeValue::String(value.clone())),
            // Unit literal `()` desugars to `(lit {type: (t-unit {})} ())` where the
            // inner `()` is an empty bare list. Treat that as RuntimeValue::Unit so
            // `def test_noop() -> unit = ()` runs cleanly instead of dying with
            // "unsupported literal form".
            Expr::List(inner, _) if inner.elements.is_empty() => Ok(RuntimeValue::Unit),
            _ => Err("unsupported literal form".to_string()),
        }
    }

    fn eval_var(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let name = children(list)
            .first()
            .and_then(symbol_name)
            .ok_or_else(|| "var missing name".to_string())?;
        if let Some(value) = self.bindings.get(name) {
            return Ok(value.clone());
        }
        if let Some(value) = self.tensor_bindings.get(name) {
            return Ok(RuntimeValue::Tensor(value.clone()));
        }
        if self.lookup_top_level_def(name).is_some() {
            return self.resolve_top_level(name);
        }
        if name == "Nil" {
            return Ok(RuntimeValue::List(Vec::new()));
        }
        if name.chars().next().is_some_and(|ch| ch.is_uppercase()) {
            return Ok(RuntimeValue::Adt {
                ctor: name.to_string(),
                fields: Vec::new(),
                field_names: None,
            });
        }
        Err(format!("unknown runtime name `{name}`"))
    }

    fn eval_fn(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let params_list = kids
            .first()
            .and_then(as_list)
            .ok_or_else(|| "fn missing params".to_string())?;
        if tag(params_list) != Some("params") {
            return Err("fn params malformed".to_string());
        }
        let params = children(params_list)
            .iter()
            .filter_map(runtime_param_name)
            .map(str::to_string)
            .collect::<Vec<_>>();
        let body = kids
            .get(1)
            .ok_or_else(|| "fn missing body".to_string())?
            .clone();
        Ok(RuntimeValue::Closure {
            params,
            body,
            env: self
                .bindings
                .iter()
                .filter(|(name, _)| !self.top_level_defs.contains_key(*name))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        })
    }

    fn eval_app(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let func = kids
            .first()
            .ok_or_else(|| "app missing function".to_string())?;
        let args = kids[1..]
            .iter()
            .map(|arg| self.eval_expr(arg))
            .collect::<Result<Vec<_>, _>>()?;

        if let Some(name) = var_name(func)
            && name.chars().next().is_some_and(|ch| ch.is_uppercase())
        {
            if name == "Cons" {
                if args.len() != 2 {
                    return Err(format!("Cons expects 2 arguments, got {}", args.len()));
                }
                let mut items = match &args[1] {
                    RuntimeValue::List(items) => items.clone(),
                    other => {
                        return Err(format!("Cons tail must be a List, got {other:?}"));
                    }
                };
                items.insert(0, args[0].clone());
                return Ok(RuntimeValue::List(items));
            }
            return Ok(RuntimeValue::Adt {
                ctor: name.to_string(),
                fields: args,
                field_names: None,
            });
        }

        if let Some(name) = builtin_name(func) {
            return self.eval_builtin(name, &args);
        }

        match self.eval_expr(func)? {
            RuntimeValue::Closure { params, body, env } => {
                if params.len() != args.len() {
                    return Err(format!(
                        "closure expected {} args, got {}",
                        params.len(),
                        args.len()
                    ));
                }
                let saved = self.bindings.clone();
                self.bindings = env;
                for (param, arg) in params.into_iter().zip(args) {
                    self.bindings.insert(param, arg);
                }
                let value = self.eval_expr(&body);
                self.bindings = saved;
                value
            }
            other => Err(format!("cannot apply non-callable value {other:?}")),
        }
    }

    fn eval_if(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let cond = self.eval_expr(kids.first().ok_or_else(|| "if missing cond".to_string())?)?;
        match cond {
            RuntimeValue::Bool(true) => self.eval_expr(
                kids.get(1)
                    .ok_or_else(|| "if missing then branch".to_string())?,
            ),
            RuntimeValue::Bool(false) => self.eval_expr(
                kids.get(2)
                    .ok_or_else(|| "if missing else branch".to_string())?,
            ),
            other => Err(format!("if condition must be bool, got {other:?}")),
        }
    }

    fn eval_let(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let bind_list = kids
            .first()
            .and_then(as_list)
            .ok_or_else(|| "let missing bindings".to_string())?;
        if tag(bind_list) != Some("bind") {
            return Err("let bindings malformed".to_string());
        }
        let saved = self.bindings.clone();
        let bind_kids = children(bind_list);
        let mut index = 0;
        while index + 1 < bind_kids.len() {
            let name = symbol_name(&bind_kids[index])
                .ok_or_else(|| "let binding must bind a name".to_string())?;
            let value = self.eval_expr(&bind_kids[index + 1])?;
            self.bindings.insert(name.to_string(), value);
            index += 2;
        }
        let body = self.eval_expr(kids.get(1).ok_or_else(|| "let missing body".to_string())?)?;
        self.bindings = saved;
        Ok(body)
    }

    fn eval_tuple_get(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let tuple = self.eval_expr(
            kids.first()
                .ok_or_else(|| "tuple-get missing tuple".to_string())?,
        )?;
        let index = children(
            as_list(
                kids.get(1)
                    .ok_or_else(|| "tuple-get missing index".to_string())?,
            )
            .ok_or_else(|| "tuple-get index must be a literal".to_string())?,
        );
        let idx = index
            .first()
            .and_then(int_value)
            .ok_or_else(|| "tuple-get index must be an int literal".to_string())?
            as usize;
        match tuple {
            RuntimeValue::Tuple(items) => items
                .get(idx)
                .cloned()
                .ok_or_else(|| format!("tuple-get index {idx} out of bounds")),
            other => Err(format!("tuple-get expects tuple input, got {other:?}")),
        }
    }

    fn eval_match(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let scrutinee = self.eval_expr(
            kids.first()
                .ok_or_else(|| "match missing scrutinee".to_string())?,
        )?;
        for arm in kids.iter().skip(1) {
            let Some(arm_list) = as_list(arm) else {
                continue;
            };
            if tag(arm_list) != Some("arm") {
                continue;
            }
            let arm_kids = children(arm_list);
            if arm_kids.len() < 3 {
                continue;
            }
            let saved = self.bindings.clone();
            if pattern_matches(
                &scrutinee,
                &arm_kids[0],
                &mut self.bindings,
                &self.adt_fields,
            )? {
                let value = self.eval_expr(&arm_kids[2]);
                self.bindings = saved;
                return value;
            }
            self.bindings = saved;
        }
        Err("non-exhaustive runtime match".to_string())
    }

    fn eval_pipe(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let mut value = self.eval_expr(
            kids.first()
                .ok_or_else(|| "pipe missing head".to_string())?,
        )?;
        for stage in kids.iter().skip(1) {
            value = self.apply_callable(stage, vec![value])?;
        }
        Ok(value)
    }

    fn apply_callable(
        &mut self,
        stage: &Expr,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, String> {
        if let Some(name) = builtin_name(stage) {
            return self.eval_builtin(name, &args);
        }
        match self.eval_expr(stage)? {
            RuntimeValue::Closure { params, body, env } => {
                self.apply_resolved_callable(RuntimeValue::Closure { params, body, env }, args)
            }
            other => Err(format!("pipe stage is not callable: {other:?}")),
        }
    }

    fn apply_resolved_callable(
        &mut self,
        callable: RuntimeValue,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, String> {
        match callable {
            RuntimeValue::Closure { params, body, env } => {
                if params.len() != args.len() {
                    return Err(format!(
                        "closure expected {} args, got {}",
                        params.len(),
                        args.len()
                    ));
                }
                let saved = self.bindings.clone();
                self.bindings = env;
                for (param, arg) in params.into_iter().zip(args) {
                    self.bindings.insert(param, arg);
                }
                let value = self.eval_expr(&body);
                self.bindings = saved;
                value
            }
            other => Err(format!("value is not callable: {other:?}")),
        }
    }

    fn eval_cast(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let value = self.eval_expr(
            kids.first()
                .ok_or_else(|| "cast missing value".to_string())?,
        )?;
        let target = kids
            .get(1)
            .and_then(as_list)
            .and_then(|ty| children(ty).first())
            .and_then(symbol_name)
            .ok_or_else(|| "cast missing target type".to_string())?;
        match (value, target) {
            (RuntimeValue::Int(value), "int32" | "int64") => Ok(RuntimeValue::Int(value)),
            (RuntimeValue::Int(value), "f32" | "f64") => Ok(RuntimeValue::Float(value as f64)),
            (RuntimeValue::Float(value), "f32" | "f64") => Ok(RuntimeValue::Float(value)),
            (RuntimeValue::Float(value), "int32" | "int64") => Ok(RuntimeValue::Int(value as i64)),
            (RuntimeValue::Bool(value), "bool") => Ok(RuntimeValue::Bool(value)),
            (RuntimeValue::String(value), "string") => Ok(RuntimeValue::String(value)),
            (other, _) => Err(format!("unsupported cast from {other:?}")),
        }
    }

    fn eval_builtin(&mut self, name: &str, args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
        match name {
            "add" => numeric_binop(args, |lhs, rhs| lhs + rhs),
            "sub" => numeric_binop(args, |lhs, rhs| lhs - rhs),
            "mul" => numeric_binop(args, |lhs, rhs| lhs * rhs),
            "div" => numeric_binop(args, |lhs, rhs| lhs / rhs),
            "mod" => int_binop(args, |lhs, rhs| lhs % rhs),
            "neg" => numeric_unop(args, |value| -value),
            "exp" => float_unop(args, f64::exp),
            "log" => float_unop(args, f64::ln),
            "sin" => float_unop(args, f64::sin),
            "sqrt" => float_unop(args, f64::sqrt),
            "eq" => compare_eq(args),
            "neq" => compare_eq(args).map(|value| match value {
                RuntimeValue::Bool(value) => RuntimeValue::Bool(!value),
                RuntimeValue::Tensor(t) => RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::from_vec(
                        t.value.shape.clone(),
                        t.value
                            .data
                            .iter()
                            .map(|x| if *x == 0.0 { 1.0 } else { 0.0 })
                            .collect(),
                    ),
                    precision: Prim::Bool,
                }),
                other => other,
            }),
            "cmplt" => {
                if let (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) =
                    (args.first(), args.get(1))
                {
                    tensor_compare_value(lhs, rhs, |lhs, rhs| lhs < rhs).map(RuntimeValue::Tensor)
                } else {
                    ordered_compare(args, |lhs, rhs| lhs < rhs)
                }
            }
            "lt" => ordered_compare(args, |lhs, rhs| lhs < rhs),
            "gt" => ordered_compare(args, |lhs, rhs| lhs > rhs),
            "gte" => ordered_compare(args, |lhs, rhs| lhs >= rhs),
            "lte" => ordered_compare(args, |lhs, rhs| lhs <= rhs),
            "uniform_like" => {
                let template = expect_tensor_arg(args, 0)?;
                let low = expect_float_arg(args, 1)?;
                let high = expect_float_arg(args, 2)?;
                let seed = self.random_seed.unwrap_or(0);
                let counter = self.random_counter;
                self.random_counter = self.random_counter.saturating_add(1);
                Ok(RuntimeValue::Tensor(uniform_like_value(
                    &template,
                    low,
                    high,
                    seed ^ counter.wrapping_mul(0x9E37_79B9_7F4A_7C15),
                )))
            }
            "and" => bool_binop(args, |lhs, rhs| lhs && rhs),
            "or" => bool_binop(args, |lhs, rhs| lhs || rhs),
            "not" => bool_unop(args, |value| !value),
            "bitand" => int_binop(args, |lhs, rhs| lhs & rhs),
            "bitor" => int_binop(args, |lhs, rhs| lhs | rhs),
            "bitxor" => int_binop(args, |lhs, rhs| lhs ^ rhs),
            "shl" => int_shift_binop(args, |lhs, rhs| lhs << rhs),
            "shr" => int_shift_binop(args, |lhs, rhs| lhs >> rhs),
            "string_len" => {
                let value = expect_string_arg(args, 0)?;
                Ok(RuntimeValue::Int(value.chars().count() as i64))
            }
            "string_concat" => Ok(RuntimeValue::String(format!(
                "{}{}",
                expect_string_arg(args, 0)?,
                expect_string_arg(args, 1)?
            ))),
            "string_slice" => {
                let value = expect_string_arg(args, 0)?;
                let start = expect_int_arg(args, 1)?;
                let len = expect_int_arg(args, 2)?;
                if start < 0 || len < 0 {
                    return Err("string_slice requires non-negative start and length".to_string());
                }
                let chars = value.chars().collect::<Vec<_>>();
                let start = start as usize;
                let len = len as usize;
                if start >= chars.len() {
                    return Ok(RuntimeValue::String(String::new()));
                }
                let end = start.saturating_add(len).min(chars.len());
                Ok(RuntimeValue::String(chars[start..end].iter().collect()))
            }
            "string_contains" => Ok(RuntimeValue::Bool(
                expect_string_arg(args, 0)?.contains(&expect_string_arg(args, 1)?),
            )),
            "string_starts_with" => Ok(RuntimeValue::Bool(
                expect_string_arg(args, 0)?.starts_with(&expect_string_arg(args, 1)?),
            )),
            "string_ends_with" => Ok(RuntimeValue::Bool(
                expect_string_arg(args, 0)?.ends_with(&expect_string_arg(args, 1)?),
            )),
            "string_trim" => Ok(RuntimeValue::String(
                expect_string_arg(args, 0)?.trim().to_string(),
            )),
            "to_string" => Ok(RuntimeValue::String(render_value(
                args.first()
                    .ok_or_else(|| "to_string expects 1 argument".to_string())?,
            ))),
            "to_int" => {
                let value = expect_string_arg(args, 0)?;
                Ok(match value.trim().parse::<i64>() {
                    Ok(parsed) => RuntimeValue::Adt {
                        ctor: "Some".to_string(),
                        fields: vec![RuntimeValue::Int(parsed)],
                        field_names: None,
                    },
                    Err(_) => RuntimeValue::Adt {
                        ctor: "None".to_string(),
                        fields: Vec::new(),
                        field_names: None,
                    },
                })
            }
            "to_float" => {
                let value = expect_string_arg(args, 0)?;
                Ok(match value.trim().parse::<f64>() {
                    Ok(parsed) => RuntimeValue::Adt {
                        ctor: "Some".to_string(),
                        fields: vec![RuntimeValue::Float(parsed)],
                        field_names: None,
                    },
                    Err(_) => RuntimeValue::Adt {
                        ctor: "None".to_string(),
                        fields: Vec::new(),
                        field_names: None,
                    },
                })
            }
            "len" => match args.first() {
                Some(RuntimeValue::List(list)) => Ok(RuntimeValue::Int(list.len() as i64)),
                Some(RuntimeValue::Dict(entries)) => Ok(RuntimeValue::Int(entries.len() as i64)),
                other => Err(format!("len expects list or dict arg, got {other:?}")),
            },
            "index" => {
                let list = expect_list_arg(args, 0)?;
                let index = expect_int_arg(args, 1)?;
                if index < 0 {
                    return Err(format!("index requires non-negative index, got {index}"));
                }
                list.get(index as usize).cloned().ok_or_else(|| {
                    format!("index {index} out of bounds for list of len {}", list.len())
                })
            }
            "append" => {
                let mut list = expect_list_arg(args, 0)?;
                list.push(
                    args.get(1)
                        .cloned()
                        .ok_or_else(|| "append expects 2 arguments".to_string())?,
                );
                Ok(RuntimeValue::List(list))
            }
            "concat" => match (args.first(), args.get(1)) {
                (Some(RuntimeValue::List(parts)), Some(RuntimeValue::Int(axis)))
                    if parts
                        .iter()
                        .all(|item| matches!(item, RuntimeValue::Tensor(_))) =>
                {
                    tensor_concat_value(parts, *axis)
                }
                _ => {
                    let mut lhs = expect_list_arg(args, 0)?;
                    lhs.extend(expect_list_arg(args, 1)?);
                    Ok(RuntimeValue::List(lhs))
                }
            },
            "take" => {
                let list = expect_list_arg(args, 0)?;
                let count = expect_int_arg(args, 1)?;
                if count < 0 {
                    return Err(format!("take requires non-negative count, got {count}"));
                }
                Ok(RuntimeValue::List(
                    list.into_iter().take(count as usize).collect(),
                ))
            }
            "drop" => {
                let list = expect_list_arg(args, 0)?;
                let count = expect_int_arg(args, 1)?;
                if count < 0 {
                    return Err(format!("drop requires non-negative count, got {count}"));
                }
                Ok(RuntimeValue::List(
                    list.into_iter().skip(count as usize).collect(),
                ))
            }
            "chunk" => {
                let list = expect_list_arg(args, 0)?;
                let size = expect_int_arg(args, 1)?;
                if size <= 0 {
                    return Err(format!("chunk requires positive size, got {size}"));
                }
                let mut out = Vec::new();
                let size = size as usize;
                for chunk in list.chunks(size) {
                    out.push(RuntimeValue::List(chunk.to_vec()));
                }
                Ok(RuntimeValue::List(out))
            }
            "range" => {
                let start = expect_int_arg(args, 0)?;
                let end = expect_int_arg(args, 1)?;
                Ok(RuntimeValue::List(
                    (start..end).map(RuntimeValue::Int).collect(),
                ))
            }
            "map" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "map expects 2 arguments".to_string())?;
                let items = expect_list_arg(args, 1)?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    out.push(self.apply_resolved_callable(callback.clone(), vec![item])?);
                }
                Ok(RuntimeValue::List(out))
            }
            "filter" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "filter expects 2 arguments".to_string())?;
                let items = expect_list_arg(args, 1)?;
                let mut out = Vec::new();
                for item in items {
                    let keep =
                        self.apply_resolved_callable(callback.clone(), vec![item.clone()])?;
                    match keep {
                        RuntimeValue::Bool(true) => out.push(item),
                        RuntimeValue::Bool(false) => {}
                        other => {
                            return Err(format!("filter callback must return bool, got {other:?}"));
                        }
                    }
                }
                Ok(RuntimeValue::List(out))
            }
            "fold" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "fold expects 3 arguments".to_string())?;
                let mut acc = args
                    .get(1)
                    .cloned()
                    .ok_or_else(|| "fold expects 3 arguments".to_string())?;
                let items = expect_list_arg(args, 2)?;
                for item in items {
                    acc = self.apply_resolved_callable(callback.clone(), vec![acc, item])?;
                }
                Ok(acc)
            }
            "scan" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "scan expects 3 arguments".to_string())?;
                let mut acc = args
                    .get(1)
                    .cloned()
                    .ok_or_else(|| "scan expects 3 arguments".to_string())?;
                let items = expect_list_arg(args, 2)?;
                let mut out = Vec::with_capacity(items.len());
                for item in items {
                    acc = self.apply_resolved_callable(callback.clone(), vec![acc, item])?;
                    out.push(acc.clone());
                }
                Ok(RuntimeValue::List(out))
            }
            "partition" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "partition expects 2 arguments".to_string())?;
                let items = expect_list_arg(args, 1)?;
                let mut kept = Vec::new();
                let mut rejected = Vec::new();
                for item in items {
                    let keep =
                        self.apply_resolved_callable(callback.clone(), vec![item.clone()])?;
                    match keep {
                        RuntimeValue::Bool(true) => kept.push(item),
                        RuntimeValue::Bool(false) => rejected.push(item),
                        other => {
                            return Err(format!(
                                "partition callback must return bool, got {other:?}"
                            ));
                        }
                    }
                }
                Ok(RuntimeValue::Tuple(vec![
                    RuntimeValue::List(kept),
                    RuntimeValue::List(rejected),
                ]))
            }
            "flat_map" => {
                let callback = args
                    .first()
                    .cloned()
                    .ok_or_else(|| "flat_map expects 2 arguments".to_string())?;
                let items = expect_list_arg(args, 1)?;
                let mut out = Vec::new();
                for item in items {
                    let mapped =
                        self.apply_resolved_callable(callback.clone(), vec![item.clone()])?;
                    let RuntimeValue::List(inner) = mapped else {
                        return Err(format!(
                            "flat_map callback must return List, got {mapped:?}"
                        ));
                    };
                    out.extend(inner);
                }
                Ok(RuntimeValue::List(out))
            }
            "flatten" => {
                let lists = expect_list_arg(args, 0)?;
                let mut out = Vec::new();
                for item in lists {
                    let RuntimeValue::List(inner) = item else {
                        return Err(format!("flatten expects nested List input, got {item:?}"));
                    };
                    out.extend(inner);
                }
                Ok(RuntimeValue::List(out))
            }
            "zip" => {
                let lhs = expect_list_arg(args, 0)?;
                let rhs = expect_list_arg(args, 1)?;
                Ok(RuntimeValue::List(
                    lhs.into_iter()
                        .zip(rhs)
                        .map(|(lhs, rhs)| RuntimeValue::Tuple(vec![lhs, rhs]))
                        .collect(),
                ))
            }
            "enumerate" => {
                let items = expect_list_arg(args, 0)?;
                Ok(RuntimeValue::List(
                    items
                        .into_iter()
                        .enumerate()
                        .map(|(index, value)| {
                            RuntimeValue::Tuple(vec![RuntimeValue::Int(index as i64), value])
                        })
                        .collect(),
                ))
            }
            "dict_of" => {
                let entries = expect_list_arg(args, 0)?;
                let mut dict = Vec::with_capacity(entries.len());
                for entry in entries {
                    let RuntimeValue::Tuple(items) = entry else {
                        return Err("dict_of expects a List of 2-tuples".to_string());
                    };
                    if items.len() != 2 {
                        return Err("dict_of expects a List of 2-tuples".to_string());
                    }
                    ensure_dict_key_supported(&items[0])?;
                    upsert_dict_entry(&mut dict, items[0].clone(), items[1].clone());
                }
                Ok(RuntimeValue::Dict(dict))
            }
            "dict_get" => {
                let dict = expect_dict_arg(args, 0)?;
                let key = args
                    .get(1)
                    .ok_or_else(|| "dict_get expects 2 arguments".to_string())?;
                ensure_dict_key_supported(key)?;
                Ok(match dict_lookup(&dict, key) {
                    Some(value) => RuntimeValue::Adt {
                        ctor: "Some".to_string(),
                        fields: vec![value.clone()],
                        field_names: None,
                    },
                    None => RuntimeValue::Adt {
                        ctor: "None".to_string(),
                        fields: Vec::new(),
                        field_names: None,
                    },
                })
            }
            "dict_contains" => {
                let dict = expect_dict_arg(args, 0)?;
                let key = args
                    .get(1)
                    .ok_or_else(|| "dict_contains expects 2 arguments".to_string())?;
                ensure_dict_key_supported(key)?;
                Ok(RuntimeValue::Bool(dict_lookup(&dict, key).is_some()))
            }
            "dict_remove" => {
                let dict = expect_dict_arg(args, 0)?;
                let key = args
                    .get(1)
                    .ok_or_else(|| "dict_remove expects 2 arguments".to_string())?;
                ensure_dict_key_supported(key)?;
                Ok(RuntimeValue::Dict(
                    dict.into_iter()
                        .filter(|(existing_key, _)| !runtime_value_eq(existing_key, key))
                        .collect(),
                ))
            }
            "dict_insert" => {
                let mut dict = expect_dict_arg(args, 0)?;
                let key = args
                    .get(1)
                    .cloned()
                    .ok_or_else(|| "dict_insert expects 3 arguments".to_string())?;
                ensure_dict_key_supported(&key)?;
                let value = args
                    .get(2)
                    .cloned()
                    .ok_or_else(|| "dict_insert expects 3 arguments".to_string())?;
                upsert_dict_entry(&mut dict, key, value);
                Ok(RuntimeValue::Dict(dict))
            }
            "dict_merge" => {
                let mut lhs = expect_dict_arg(args, 0)?;
                let rhs = expect_dict_arg(args, 1)?;
                for (key, value) in rhs {
                    ensure_dict_key_supported(&key)?;
                    upsert_dict_entry(&mut lhs, key, value);
                }
                Ok(RuntimeValue::Dict(lhs))
            }
            "dict_keys" => {
                let dict = expect_dict_arg(args, 0)?;
                Ok(RuntimeValue::List(
                    dict.into_iter().map(|(key, _)| key).collect(),
                ))
            }
            "dict_values" => {
                let dict = expect_dict_arg(args, 0)?;
                Ok(RuntimeValue::List(
                    dict.into_iter().map(|(_, value)| value).collect(),
                ))
            }
            "dict_entries" => {
                let dict = expect_dict_arg(args, 0)?;
                Ok(RuntimeValue::List(
                    dict.into_iter()
                        .map(|(key, value)| RuntimeValue::Tuple(vec![key, value]))
                        .collect(),
                ))
            }
            "copy" => match args.first() {
                Some(RuntimeValue::Tensor(tensor)) => Ok(RuntimeValue::Tensor(tensor.clone())),
                other => Err(format!("copy expects tensor input, got {other:?}")),
            },
            "reshape" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let shape = expect_list_arg(args, 1)?;
                tensor_reshape_value(&tensor, &shape).map(RuntimeValue::Tensor)
            }
            "to_tensor" => {
                let values = expect_list_arg(args, 0)?;
                let (precision, data) = list_to_tensor_data(&values)?;
                Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::from_vec(vec![data.len()], data),
                    precision,
                }))
            }
            "to_list" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let values = tensor_to_list_values(&tensor)?;
                Ok(RuntimeValue::List(values))
            }
            "pad_sequences" => {
                let sequences = expect_list_arg(args, 0)?;
                let pad = args
                    .get(1)
                    .cloned()
                    .ok_or_else(|| "pad_sequences expects 2 arguments".to_string())?;
                let (precision, data, batch, width) = pad_sequences_value(&sequences, &pad)?;
                Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::from_vec(vec![batch, width], data),
                    precision,
                }))
            }
            "pad_sequences_to" => {
                let sequences = expect_list_arg(args, 0)?;
                let width = expect_int_arg(args, 1)?;
                let pad = args
                    .get(2)
                    .cloned()
                    .ok_or_else(|| "pad_sequences_to expects 3 arguments".to_string())?;
                let (precision, data, batch) = pad_sequences_to_value(&sequences, width, &pad)?;
                Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::from_vec(vec![batch, width.max(0) as usize], data),
                    precision,
                }))
            }
            "read_file" => {
                let path = expect_string_arg(args, 0)?;
                let text = fs::read_to_string(&path)
                    .map_err(|err| format!("read_file failed for `{path}`: {err}"))?;
                Ok(RuntimeValue::String(text))
            }
            "write_file" => {
                let path = expect_string_arg(args, 0)?;
                let contents = expect_string_arg(args, 1)?;
                fs::write(&path, contents)
                    .map_err(|err| format!("write_file failed for `{path}`: {err}"))?;
                Ok(RuntimeValue::Unit)
            }
            "read_lines" => {
                let path = expect_string_arg(args, 0)?;
                let text = fs::read_to_string(&path)
                    .map_err(|err| format!("read_lines failed for `{path}`: {err}"))?;
                Ok(RuntimeValue::List(
                    text.lines()
                        .map(|line| RuntimeValue::String(line.to_string()))
                        .collect(),
                ))
            }
            "read_bytes" => {
                let path = expect_string_arg(args, 0)?;
                let bytes = fs::read(&path)
                    .map_err(|err| format!("read_bytes failed for `{path}`: {err}"))?;
                Ok(RuntimeValue::List(
                    bytes
                        .into_iter()
                        .map(|byte| RuntimeValue::Int(i64::from(byte)))
                        .collect(),
                ))
            }
            "file_exists" => {
                let path = expect_string_arg(args, 0)?;
                Ok(RuntimeValue::Bool(std::path::Path::new(&path).exists()))
            }
            "list_dir" => {
                let path = expect_string_arg(args, 0)?;
                let entries = fs::read_dir(&path)
                    .map_err(|err| format!("list_dir failed for `{path}`: {err}"))?;
                let mut out = Vec::new();
                for entry in entries {
                    let entry =
                        entry.map_err(|err| format!("list_dir failed for `{path}`: {err}"))?;
                    out.push(RuntimeValue::String(
                        entry.file_name().to_string_lossy().into_owned(),
                    ));
                }
                Ok(RuntimeValue::List(out))
            }
            "mmap_file" => {
                let path = expect_string_arg(args, 0)?;
                let bytes = fs::read(&path)
                    .map_err(|err| format!("mmap_file failed for `{path}`: {err}"))?;
                Ok(RuntimeValue::MappedFile(bytes))
            }
            "mmap_read" => {
                let mapped = args
                    .first()
                    .ok_or_else(|| "mmap_read expects 3 arguments".to_string())?;
                let offset = expect_int_arg(args, 1)?;
                let len = expect_int_arg(args, 2)?;
                let RuntimeValue::MappedFile(bytes) = mapped else {
                    return Err(format!("mmap_read expects MappedFile, got {mapped:?}"));
                };
                if offset < 0 || len < 0 {
                    return Err("mmap_read requires non-negative offset and length".to_string());
                }
                let offset = offset as usize;
                let len = len as usize;
                if offset > bytes.len() {
                    return Err("mmap_read offset out of bounds".to_string());
                }
                let end = offset.saturating_add(len).min(bytes.len());
                Ok(RuntimeValue::List(
                    bytes[offset..end]
                        .iter()
                        .map(|byte| RuntimeValue::Int(i64::from(*byte)))
                        .collect(),
                ))
            }
            "mmap_len" => match args.first() {
                Some(RuntimeValue::MappedFile(bytes)) => Ok(RuntimeValue::Int(bytes.len() as i64)),
                other => Err(format!("mmap_len expects MappedFile, got {other:?}")),
            },
            "split" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                let sizes = expect_list_arg(args, 2)?;
                tensor_split_value(&tensor, axis, &sizes)
            }
            "gather" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let indices = expect_tensor_arg(args, 1)?;
                let axis = expect_int_arg(args, 2)?;
                tensor_gather_value(&tensor, &indices, axis).map(RuntimeValue::Tensor)
            }
            "scatter" => {
                let base = expect_tensor_arg(args, 0)?;
                let indices = expect_tensor_arg(args, 1)?;
                let updates = expect_tensor_arg(args, 2)?;
                let axis = expect_int_arg(args, 3)?;
                let mode = expect_string_arg(args, 4)?;
                tensor_scatter_value(&base, &indices, &updates, axis, &mode)
                    .map(RuntimeValue::Tensor)
            }
            "where" => {
                let cond = expect_tensor_arg(args, 0)?;
                let then_tensor = expect_tensor_arg(args, 1)?;
                let else_tensor = expect_tensor_arg(args, 2)?;
                tensor_where_value(&cond, &then_tensor, &else_tensor).map(RuntimeValue::Tensor)
            }
            "cumsum" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_cumsum_value(&tensor, axis).map(RuntimeValue::Tensor)
            }
            "sort" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_sort_value(&tensor, axis)
            }
            "diagonal" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis1 = expect_int_arg(args, 1)?;
                let axis2 = expect_int_arg(args, 2)?;
                tensor_diagonal_value(&tensor, axis1, axis2).map(RuntimeValue::Tensor)
            }
            "trace" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis1 = expect_int_arg(args, 1)?;
                let axis2 = expect_int_arg(args, 2)?;
                tensor_trace_value(&tensor, axis1, axis2).map(RuntimeValue::Tensor)
            }
            "clamp" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let lo = expect_tensor_arg(args, 1)?;
                let hi = expect_tensor_arg(args, 2)?;
                tensor_clamp_value(&tensor, &lo, &hi).map(RuntimeValue::Tensor)
            }
            "einsum" => {
                let equation = expect_string_arg(args, 0)?;
                let lhs = expect_tensor_arg(args, 1)?;
                let rhs = expect_tensor_arg(args, 2)?;
                tensor_einsum_value(&equation, &lhs, &rhs).map(RuntimeValue::Tensor)
            }
            "rank" => {
                let tensor = expect_tensor_arg(args, 0)?;
                Ok(RuntimeValue::Int(tensor.value.shape.len() as i64))
            }
            "shape" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                if axis < 0 {
                    return Err(format!("shape requires non-negative axis, got {axis}"));
                }
                let axis = axis as usize;
                let dim = tensor
                    .value
                    .shape
                    .get(axis)
                    .copied()
                    .ok_or_else(|| format!("shape axis {axis} out of bounds"))?;
                Ok(RuntimeValue::Int(dim as i64))
            }
            "numel" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let numel = tensor.value.shape.iter().product::<usize>().max(1);
                Ok(RuntimeValue::Int(numel as i64))
            }
            "tensor_to_scalar" => {
                let tensor = expect_tensor_arg(args, 0)?;
                if !tensor.value.shape.is_empty() {
                    return Err("tensor_to_scalar expects a rank-0 tensor".to_string());
                }
                let value = tensor.value.data.first().copied().unwrap_or(0.0);
                match tensor.precision {
                    Prim::Bool => Ok(RuntimeValue::Bool(value != 0.0)),
                    Prim::Int8 | Prim::Int32 | Prim::Int64 => Ok(RuntimeValue::Int(value as i64)),
                    _ => Ok(RuntimeValue::Float(value)),
                }
            }
            "scalar_to_tensor" => match args.first() {
                Some(RuntimeValue::Int(value)) => Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::scalar(*value as f64),
                    precision: Prim::Int64,
                })),
                Some(RuntimeValue::Float(value)) => Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::scalar(*value),
                    precision: Prim::F32,
                })),
                Some(RuntimeValue::Bool(value)) => Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::scalar(if *value { 1.0 } else { 0.0 }),
                    precision: Prim::Bool,
                })),
                other => Err(format!(
                    "scalar_to_tensor expects scalar input, got {other:?}"
                )),
            },
            "print" => {
                let value = args
                    .first()
                    .ok_or_else(|| "print expects 1 argument".to_string())?;
                self.transcript.push(render_value(value));
                Ok(RuntimeValue::Unit)
            }
            "fail" => {
                let message = expect_string_arg(args, 0)?;
                Err(message)
            }
            "test_assert" => {
                let cond = expect_bool_arg(args, 0)?;
                let label = expect_string_arg(args, 1)?;
                if cond {
                    Ok(RuntimeValue::Unit)
                } else {
                    Err(format!("assert failed: {label}"))
                }
            }
            "test_assert_eq_f32" => {
                let actual = expect_float_arg(args, 0)?;
                let expected = expect_float_arg(args, 1)?;
                let label = expect_string_arg(args, 2)?;
                if actual == expected {
                    Ok(RuntimeValue::Unit)
                } else {
                    Err(format!(
                        "assert_eq_f32 ({label}): expected {expected}, got {actual}"
                    ))
                }
            }
            "test_assert_eq_int" => {
                let actual = expect_int_arg(args, 0)?;
                let expected = expect_int_arg(args, 1)?;
                let label = expect_string_arg(args, 2)?;
                if actual == expected {
                    Ok(RuntimeValue::Unit)
                } else {
                    Err(format!(
                        "assert_eq_int ({label}): expected {expected}, got {actual}"
                    ))
                }
            }
            "test_assert_eq_bool" => {
                let actual = expect_bool_arg(args, 0)?;
                let expected = expect_bool_arg(args, 1)?;
                let label = expect_string_arg(args, 2)?;
                if actual == expected {
                    Ok(RuntimeValue::Unit)
                } else {
                    Err(format!(
                        "assert_eq_bool ({label}): expected {expected}, got {actual}"
                    ))
                }
            }
            "test_assert_eq_string" => {
                let actual = expect_string_arg(args, 0)?;
                let expected = expect_string_arg(args, 1)?;
                let label = expect_string_arg(args, 2)?;
                if actual == expected {
                    Ok(RuntimeValue::Unit)
                } else {
                    Err(format!(
                        "assert_eq_string ({label}): expected {expected:?}, got {actual:?}"
                    ))
                }
            }
            "test_assert_eq_tensor_int64" => {
                // Bit-exact tensor equality for int64 tensors. Std.Test
                // exposes this as `assert_eq_tensor_int64` because
                // `assert_close_tensor` types only on f32 tensors and is
                // tolerance-based — neither fits int64 reduction outputs
                // (e.g. `argmax`/`argmin` which return int64 indices).
                let actual = expect_tensor_arg(args, 0)?;
                let expected = expect_tensor_arg(args, 1)?;
                let label = expect_string_arg(args, 2)?;
                let actual_data = &actual.value.data;
                let expected_data = &expected.value.data;
                if actual_data.len() != expected_data.len() {
                    return Err(format!(
                        "assert_eq_tensor_int64 ({label}): length mismatch, expected {} elements, got {}",
                        expected_data.len(),
                        actual_data.len()
                    ));
                }
                for (i, (&a, &e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
                    if a != e {
                        return Err(format!(
                            "assert_eq_tensor_int64 ({label}): at index {i} expected {} got {}",
                            e as i64, a as i64
                        ));
                    }
                }
                Ok(RuntimeValue::Unit)
            }
            "test_assert_close_tensor" => {
                let actual = expect_tensor_arg(args, 0)?;
                let expected = expect_tensor_arg(args, 1)?;
                let tol = expect_float_arg(args, 2)?;
                let label = expect_string_arg(args, 3)?;
                if tol.is_nan() || tol < 0.0 {
                    return Err(format!(
                        "assert_close_tensor ({label}): invalid tolerance {tol} (must be finite and non-negative)"
                    ));
                }
                let actual_data = &actual.value.data;
                let expected_data = &expected.value.data;
                if actual_data.len() != expected_data.len() {
                    return Err(format!(
                        "assert_close_tensor ({label}): length mismatch, expected {} elements, got {}",
                        expected_data.len(),
                        actual_data.len()
                    ));
                }
                for (i, (&a, &e)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
                    if a.is_nan() || e.is_nan() {
                        return Err(format!(
                            "assert_close_tensor ({label}): at index {i} expected {e}, got {a}, tol {tol} (NaN is never close)"
                        ));
                    }
                    let diff = (a - e).abs();
                    let mismatch = if tol == 0.0 { a != e } else { diff > tol };
                    if mismatch {
                        return Err(format!(
                            "assert_close_tensor ({label}): at index {i} expected {e}, got {a}, tol {tol}"
                        ));
                    }
                }
                Ok(RuntimeValue::Unit)
            }
            "debug" => {
                let value = args
                    .first()
                    .ok_or_else(|| "debug expects 1 argument".to_string())?;
                self.transcript.push(render_value(value));
                Ok(value.clone())
            }
            "min_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Min).map(RuntimeValue::Tensor)
            }
            "prod_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Prod).map(RuntimeValue::Tensor)
            }
            "argmax_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Argmax).map(RuntimeValue::Tensor)
            }
            "argmin_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Argmin).map(RuntimeValue::Tensor)
            }
            "sum" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Sum).map(RuntimeValue::Tensor)
            }
            "matmul" => {
                let lhs = expect_tensor_arg(args, 0)?;
                let rhs = expect_tensor_arg(args, 1)?;
                tensor_matmul_host(&lhs, &rhs).map(RuntimeValue::Tensor)
            }
            "permute" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let rank = tensor.value.shape.len();
                if args.len() != rank + 1 {
                    return Err(format!(
                        "permute expects {} axis arguments for rank-{rank} tensor, got {}",
                        rank,
                        args.len().saturating_sub(1)
                    ));
                }
                let mut axes: Vec<usize> = Vec::with_capacity(rank);
                for i in 0..rank {
                    let raw = expect_int_arg(args, i + 1)?;
                    if raw < 0 {
                        return Err(format!("permute requires non-negative axis, got {raw}"));
                    }
                    axes.push(raw as usize);
                }
                tensor_permute_host(&tensor, &axes).map(RuntimeValue::Tensor)
            }
            "expand" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                let count = expect_int_arg(args, 2)?;
                if axis < 0 {
                    return Err(format!("expand requires non-negative axis, got {axis}"));
                }
                if count <= 0 {
                    return Err(format!("expand requires positive count, got {count}"));
                }
                tensor_expand_host(&tensor, axis as usize, count as usize).map(RuntimeValue::Tensor)
            }
            "softmax" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_softmax_host(&tensor, axis).map(RuntimeValue::Tensor)
            }
            other => Err(format!("unsupported builtin `{other}` in host runtime")),
        }
    }
}

fn pattern_matches(
    value: &RuntimeValue,
    pattern: &Expr,
    bindings: &mut HashMap<String, RuntimeValue>,
    adt_fields: &HashMap<String, Vec<String>>,
) -> Result<bool, String> {
    let Some(list) = as_list(pattern) else {
        return Ok(false);
    };
    match tag(list) {
        Some("pat-var") => {
            if let Some(name) = children(list).first().and_then(symbol_name) {
                bindings.insert(name.to_string(), value.clone());
                return Ok(true);
            }
            Ok(false)
        }
        Some("pat-wild") => Ok(true),
        Some("pat-lit") => {
            let lit = children(list)
                .first()
                .ok_or_else(|| "pat-lit missing value".to_string())?;
            Ok(match (value, lit) {
                (RuntimeValue::Int(lhs), Expr::Atom(Atom::Int(rhs), _)) => lhs == rhs,
                (RuntimeValue::Float(lhs), Expr::Atom(Atom::Float(rhs), _)) => lhs == rhs,
                (RuntimeValue::Bool(lhs), Expr::Atom(Atom::Bool(rhs), _)) => lhs == rhs,
                (RuntimeValue::String(lhs), Expr::Atom(Atom::Str(rhs), _)) => lhs == rhs,
                _ => false,
            })
        }
        Some("pat-ctor") => {
            let kids = children(list);
            let Some(ctor) = kids.first().and_then(symbol_name) else {
                return Ok(false);
            };
            let RuntimeValue::Adt {
                ctor: got, fields, ..
            } = value
            else {
                return Ok(false);
            };
            if ctor != got || kids.len().saturating_sub(1) != fields.len() {
                return Ok(false);
            }
            for (subpat, field) in kids.iter().skip(1).zip(fields) {
                if !pattern_matches(field, subpat, bindings, adt_fields)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Some("pat-record") => {
            let kids = children(list);
            let Some(ctor) = kids.first().and_then(symbol_name) else {
                return Ok(false);
            };
            let RuntimeValue::Adt {
                ctor: got,
                fields,
                field_names,
            } = value
            else {
                return Ok(false);
            };
            if ctor != got {
                return Ok(false);
            }
            let Some(declared_fields) = adt_fields.get(ctor).cloned().or(field_names.clone())
            else {
                return Ok(false);
            };
            for kv_expr in kids.iter().skip(1) {
                let Some(kv_list) = as_list(kv_expr) else {
                    continue;
                };
                if tag(kv_list) != Some("kv") {
                    continue;
                }
                let kv_kids = children(kv_list);
                let Some(field_name) = kv_kids.first().and_then(symbol_name) else {
                    continue;
                };
                let Some(pattern_expr) = kv_kids.get(1) else {
                    continue;
                };
                let Some(index) = declared_fields
                    .iter()
                    .position(|declared| declared == field_name)
                else {
                    return Ok(false);
                };
                let Some(field_value) = fields.get(index) else {
                    return Ok(false);
                };
                if !pattern_matches(field_value, pattern_expr, bindings, adt_fields)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        Some("pat-tuple") => {
            let RuntimeValue::Tuple(items) = value else {
                return Ok(false);
            };
            if items.len() != children(list).len() {
                return Ok(false);
            }
            for (subpat, item) in children(list).iter().zip(items) {
                if !pattern_matches(item, subpat, bindings, adt_fields)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn collect_adt_ctor_fields(exprs: &[Expr]) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("deftype") {
            continue;
        }
        let kids = children(list);
        for variant in kids.iter().skip(2) {
            let Some(variant_list) = as_list(variant) else {
                continue;
            };
            if tag(variant_list) != Some("variant") {
                continue;
            }
            let variant_kids = children(variant_list);
            let Some(ctor) = variant_kids.first().and_then(symbol_name) else {
                continue;
            };
            let mut fields = Vec::new();
            for field in variant_kids.iter().skip(1) {
                let Some(field_list) = as_list(field) else {
                    continue;
                };
                if tag(field_list) != Some("field") {
                    continue;
                }
                if let Some(name) = children(field_list).first().and_then(symbol_name) {
                    fields.push(name.to_string());
                }
            }
            if !fields.is_empty() {
                out.insert(ctor.to_string(), fields);
            }
        }
    }
    out
}

fn terminal_name_matches(full_name: &str, short_name: &str) -> bool {
    full_name == short_name || terminal_name(full_name) == terminal_name(short_name)
}

fn terminal_name(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

fn numeric_binop(
    args: &[RuntimeValue],
    op: impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
            tensor_numeric_binop(lhs, rhs, &op)
        }
        (Some(RuntimeValue::Tensor(lhs)), Some(rhs)) => tensor_scalar_binop(lhs, rhs, &op),
        (Some(lhs), Some(RuntimeValue::Tensor(rhs))) => scalar_tensor_binop(lhs, rhs, &op),
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) => {
            Ok(RuntimeValue::Int(op(*lhs as f64, *rhs as f64) as i64))
        }
        (Some(RuntimeValue::Float(lhs)), Some(RuntimeValue::Float(rhs))) => {
            Ok(RuntimeValue::Float(op(*lhs, *rhs)))
        }
        other => Err(format!(
            "numeric op expects matching int or float args, got {other:?}"
        )),
    }
}

fn numeric_unop(args: &[RuntimeValue], op: impl Fn(f64) -> f64) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Tensor(tensor)) => tensor_numeric_unop(tensor, &op),
        Some(RuntimeValue::Int(value)) => Ok(RuntimeValue::Int(op(*value as f64) as i64)),
        Some(RuntimeValue::Float(value)) => Ok(RuntimeValue::Float(op(*value))),
        other => Err(format!(
            "numeric op expects int or float arg, got {other:?}"
        )),
    }
}

fn tensor_numeric_binop(
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
    op: &impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    if lhs.value.shape != rhs.value.shape {
        return Err(format!(
            "tensor shapes must match for elementwise op, got {:?} vs {:?}",
            lhs.value.shape, rhs.value.shape
        ));
    }
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(
            lhs.value.shape.clone(),
            lhs.value
                .data
                .iter()
                .zip(&rhs.value.data)
                .map(|(l, r)| op(*l, *r))
                .collect(),
        ),
        precision: lhs.precision,
    }))
}

fn tensor_scalar_binop(
    tensor: &RuntimeTensorValue,
    scalar: &RuntimeValue,
    op: &impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    let scalar = runtime_scalar_as_f64(scalar)
        .ok_or_else(|| format!("numeric op expects scalar rhs, got {scalar:?}"))?;
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(
            tensor.value.shape.clone(),
            tensor
                .value
                .data
                .iter()
                .map(|value| op(*value, scalar))
                .collect(),
        ),
        precision: tensor.precision,
    }))
}

fn scalar_tensor_binop(
    scalar: &RuntimeValue,
    tensor: &RuntimeTensorValue,
    op: &impl Fn(f64, f64) -> f64,
) -> Result<RuntimeValue, String> {
    let scalar = runtime_scalar_as_f64(scalar)
        .ok_or_else(|| format!("numeric op expects scalar lhs, got {scalar:?}"))?;
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(
            tensor.value.shape.clone(),
            tensor
                .value
                .data
                .iter()
                .map(|value| op(scalar, *value))
                .collect(),
        ),
        precision: tensor.precision,
    }))
}

fn tensor_numeric_unop(
    tensor: &RuntimeTensorValue,
    op: &impl Fn(f64) -> f64,
) -> Result<RuntimeValue, String> {
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(
            tensor.value.shape.clone(),
            tensor.value.data.iter().map(|value| op(*value)).collect(),
        ),
        precision: tensor.precision,
    }))
}

fn runtime_scalar_as_f64(value: &RuntimeValue) -> Option<f64> {
    match value {
        RuntimeValue::Int(value) => Some(*value as f64),
        RuntimeValue::Float(value) => Some(*value),
        RuntimeValue::Bool(value) => Some(if *value { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn int_binop(args: &[RuntimeValue], op: impl Fn(i64, i64) -> i64) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) => {
            Ok(RuntimeValue::Int(op(*lhs, *rhs)))
        }
        other => Err(format!("integer op expects int args, got {other:?}")),
    }
}

fn int_shift_binop(
    args: &[RuntimeValue],
    op: impl Fn(i64, u32) -> i64,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) if *rhs >= 0 => {
            Ok(RuntimeValue::Int(op(*lhs, *rhs as u32)))
        }
        (Some(RuntimeValue::Int(_)), Some(RuntimeValue::Int(rhs))) => {
            Err(format!("shift amount must be non-negative, got {rhs}"))
        }
        other => Err(format!("shift op expects int args, got {other:?}")),
    }
}

fn float_unop(args: &[RuntimeValue], op: impl Fn(f64) -> f64) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Float(value)) => Ok(RuntimeValue::Float(op(*value))),
        other => Err(format!("float op expects float arg, got {other:?}")),
    }
}

fn compare_eq(args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        (Some(RuntimeValue::Float(lhs)), Some(RuntimeValue::Float(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        (Some(RuntimeValue::Bool(lhs)), Some(RuntimeValue::Bool(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        (Some(RuntimeValue::String(lhs)), Some(RuntimeValue::String(rhs))) => {
            Ok(RuntimeValue::Bool(lhs == rhs))
        }
        // Element-wise tensor-tensor equality. The build-target lane already
        // supports this; the host evaluator was returning an error, blocking
        // IntCol/BoolCol construction and tensor-level is_nan in chelis test.
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
            tensor_compare_value(lhs, rhs, |a, b| a == b).map(RuntimeValue::Tensor)
        }
        other => Err(format!("eq/neq expect matching scalar args, got {other:?}")),
    }
}

fn ordered_compare(
    args: &[RuntimeValue],
    cmp: impl Fn(f64, f64) -> bool,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Int(lhs)), Some(RuntimeValue::Int(rhs))) => {
            Ok(RuntimeValue::Bool(cmp(*lhs as f64, *rhs as f64)))
        }
        (Some(RuntimeValue::Float(lhs)), Some(RuntimeValue::Float(rhs))) => {
            Ok(RuntimeValue::Bool(cmp(*lhs, *rhs)))
        }
        // Element-wise tensor-tensor ordering. Mirrors the build-target lane
        // and unblocks the same downstream tensor-level boolean ops.
        (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
            tensor_compare_value(lhs, rhs, cmp).map(RuntimeValue::Tensor)
        }
        other => Err(format!(
            "ordered comparison expects matching numeric args, got {other:?}"
        )),
    }
}

fn tensor_compare_value(
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
    cmp: impl Fn(f64, f64) -> bool,
) -> Result<RuntimeTensorValue, String> {
    if lhs.precision != rhs.precision {
        return Err("tensor comparison expects matching tensor precision".to_string());
    }
    if lhs.value.shape != rhs.value.shape {
        return Err("tensor comparison expects matching tensor shape".to_string());
    }
    let data = lhs
        .value
        .data
        .iter()
        .zip(&rhs.value.data)
        .map(|(lhs, rhs)| if cmp(*lhs, *rhs) { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(lhs.value.shape.clone(), data),
        precision: Prim::Bool,
    })
}

fn bool_binop(
    args: &[RuntimeValue],
    op: impl Fn(bool, bool) -> bool,
) -> Result<RuntimeValue, String> {
    match (args.first(), args.get(1)) {
        (Some(RuntimeValue::Bool(lhs)), Some(RuntimeValue::Bool(rhs))) => {
            Ok(RuntimeValue::Bool(op(*lhs, *rhs)))
        }
        other => Err(format!("bool op expects bool args, got {other:?}")),
    }
}

fn bool_unop(args: &[RuntimeValue], op: impl Fn(bool) -> bool) -> Result<RuntimeValue, String> {
    match args.first() {
        Some(RuntimeValue::Bool(value)) => Ok(RuntimeValue::Bool(op(*value))),
        other => Err(format!("bool op expects bool arg, got {other:?}")),
    }
}

fn expect_tensor_arg(args: &[RuntimeValue], index: usize) -> Result<RuntimeTensorValue, String> {
    match args.get(index) {
        Some(RuntimeValue::Tensor(value)) => Ok(value.clone()),
        other => Err(format!(
            "expected tensor arg at index {index}, got {other:?}"
        )),
    }
}

fn expect_string_arg(args: &[RuntimeValue], index: usize) -> Result<String, String> {
    match args.get(index) {
        Some(RuntimeValue::String(value)) => Ok(value.clone()),
        other => Err(format!(
            "expected string arg at index {index}, got {other:?}"
        )),
    }
}

fn expect_list_arg(args: &[RuntimeValue], index: usize) -> Result<Vec<RuntimeValue>, String> {
    match args.get(index) {
        Some(RuntimeValue::List(items)) => Ok(items.clone()),
        other => Err(format!("expected list arg at index {index}, got {other:?}")),
    }
}

fn expect_dict_arg(
    args: &[RuntimeValue],
    index: usize,
) -> Result<Vec<(RuntimeValue, RuntimeValue)>, String> {
    match args.get(index) {
        Some(RuntimeValue::Dict(entries)) => Ok(entries.clone()),
        other => Err(format!("expected dict arg at index {index}, got {other:?}")),
    }
}

fn expect_int_arg(args: &[RuntimeValue], index: usize) -> Result<i64, String> {
    match args.get(index) {
        Some(RuntimeValue::Int(value)) => Ok(*value),
        other => Err(format!("expected int arg at index {index}, got {other:?}")),
    }
}

fn expect_bool_arg(args: &[RuntimeValue], index: usize) -> Result<bool, String> {
    match args.get(index) {
        Some(RuntimeValue::Bool(value)) => Ok(*value),
        other => Err(format!("expected bool arg at index {index}, got {other:?}")),
    }
}

fn expect_float_arg(args: &[RuntimeValue], index: usize) -> Result<f64, String> {
    match args.get(index) {
        Some(RuntimeValue::Float(value)) => Ok(*value),
        Some(RuntimeValue::Int(value)) => Ok(*value as f64),
        other => Err(format!(
            "expected float arg at index {index}, got {other:?}"
        )),
    }
}

fn ensure_dict_key_supported(value: &RuntimeValue) -> Result<(), String> {
    match value {
        RuntimeValue::Int(_) | RuntimeValue::String(_) => Ok(()),
        other => Err(format!(
            "dict keys must be int64 or string in 3d, got {other:?}"
        )),
    }
}

fn runtime_value_eq(lhs: &RuntimeValue, rhs: &RuntimeValue) -> bool {
    match (lhs, rhs) {
        (RuntimeValue::Int(lhs), RuntimeValue::Int(rhs)) => lhs == rhs,
        (RuntimeValue::Float(lhs), RuntimeValue::Float(rhs)) => lhs == rhs,
        (RuntimeValue::Bool(lhs), RuntimeValue::Bool(rhs)) => lhs == rhs,
        (RuntimeValue::String(lhs), RuntimeValue::String(rhs)) => lhs == rhs,
        (RuntimeValue::Tuple(lhs), RuntimeValue::Tuple(rhs)) => {
            lhs.len() == rhs.len()
                && lhs
                    .iter()
                    .zip(rhs)
                    .all(|(lhs, rhs)| runtime_value_eq(lhs, rhs))
        }
        _ => false,
    }
}

fn upsert_dict_entry(
    dict: &mut Vec<(RuntimeValue, RuntimeValue)>,
    key: RuntimeValue,
    value: RuntimeValue,
) {
    if let Some((_, existing)) = dict
        .iter_mut()
        .find(|(existing_key, _)| runtime_value_eq(existing_key, &key))
    {
        *existing = value;
    } else {
        dict.push((key, value));
    }
}

fn dict_lookup<'a>(
    dict: &'a [(RuntimeValue, RuntimeValue)],
    key: &RuntimeValue,
) -> Option<&'a RuntimeValue> {
    dict.iter()
        .find(|(existing_key, _)| runtime_value_eq(existing_key, key))
        .map(|(_, value)| value)
}

fn list_to_tensor_data(values: &[RuntimeValue]) -> Result<(Prim, Vec<f64>), String> {
    let mut precision = None;
    let mut data = Vec::with_capacity(values.len());
    for value in values {
        match value {
            RuntimeValue::Int(value) => {
                precision.get_or_insert(Prim::Int64);
                if precision != Some(Prim::Int64) {
                    return Err(
                        "to_tensor requires homogeneous numeric or bool list elements".to_string(),
                    );
                }
                data.push(*value as f64);
            }
            RuntimeValue::Float(value) => {
                precision.get_or_insert(Prim::F32);
                if precision != Some(Prim::F32) {
                    return Err(
                        "to_tensor requires homogeneous numeric or bool list elements".to_string(),
                    );
                }
                data.push(*value);
            }
            RuntimeValue::Bool(value) => {
                precision.get_or_insert(Prim::Bool);
                if precision != Some(Prim::Bool) {
                    return Err(
                        "to_tensor requires homogeneous numeric or bool list elements".to_string(),
                    );
                }
                data.push(if *value { 1.0 } else { 0.0 });
            }
            other => {
                return Err(format!(
                    "to_tensor expects numeric or bool list elements, got {other:?}"
                ));
            }
        }
    }
    Ok((precision.unwrap_or(Prim::F32), data))
}

fn tensor_to_list_values(tensor: &RuntimeTensorValue) -> Result<Vec<RuntimeValue>, String> {
    if tensor.value.shape.len() != 1 {
        return Err(format!(
            "to_list expects a rank-1 tensor, got rank {} tensor",
            tensor.value.shape.len()
        ));
    }
    let mut values = Vec::with_capacity(tensor.value.data.len());
    for value in &tensor.value.data {
        values.push(match tensor.precision {
            Prim::Bool => RuntimeValue::Bool(*value != 0.0),
            Prim::Int8 | Prim::Int32 | Prim::Int64 => RuntimeValue::Int(*value as i64),
            Prim::F16 | Prim::Bf16 | Prim::F32 | Prim::F64 | Prim::F8e4m3 => {
                RuntimeValue::Float(*value)
            }
            other => {
                return Err(format!(
                    "to_list expects numeric or bool tensor input, got {other:?}"
                ));
            }
        });
    }
    Ok(values)
}

fn pad_sequences_value(
    sequences: &[RuntimeValue],
    pad: &RuntimeValue,
) -> Result<(Prim, Vec<f64>, usize, usize), String> {
    let pad_precision = match pad {
        RuntimeValue::Int(_) => Prim::Int64,
        RuntimeValue::Float(_) => Prim::F32,
        other => {
            return Err(format!(
                "pad_sequences expects numeric pad value, got {other:?}"
            ));
        }
    };
    let mut rows = Vec::<Vec<f64>>::with_capacity(sequences.len());
    let mut width = 0usize;
    for sequence in sequences {
        let RuntimeValue::List(items) = sequence else {
            return Err(format!(
                "pad_sequences expects nested lists, got {sequence:?}"
            ));
        };
        let (row_precision, row) = list_to_tensor_data(items)?;
        if row_precision != pad_precision {
            return Err("pad_sequences requires homogeneous numeric nested lists".to_string());
        }
        width = width.max(row.len());
        rows.push(row);
    }
    let pad_value = match pad {
        RuntimeValue::Int(value) => *value as f64,
        RuntimeValue::Float(value) => *value,
        _ => unreachable!(),
    };
    let batch = rows.len();
    let mut data = Vec::with_capacity(batch * width);
    for row in rows {
        data.extend(row.iter().copied());
        data.extend(std::iter::repeat_n(
            pad_value,
            width.saturating_sub(row.len()),
        ));
    }
    Ok((pad_precision, data, batch, width))
}

fn pad_sequences_to_value(
    sequences: &[RuntimeValue],
    width: i64,
    pad: &RuntimeValue,
) -> Result<(Prim, Vec<f64>, usize), String> {
    if width < 0 {
        return Err(format!(
            "pad_sequences_to requires non-negative width, got {width}"
        ));
    }
    let pad_precision = match pad {
        RuntimeValue::Int(_) => Prim::Int64,
        RuntimeValue::Float(_) => Prim::F32,
        other => {
            return Err(format!(
                "pad_sequences_to expects numeric pad value, got {other:?}"
            ));
        }
    };
    let width = width as usize;
    let mut rows = Vec::<Vec<f64>>::with_capacity(sequences.len());
    for sequence in sequences {
        let RuntimeValue::List(items) = sequence else {
            return Err(format!(
                "pad_sequences_to expects nested lists, got {sequence:?}"
            ));
        };
        let (row_precision, row) = list_to_tensor_data(items)?;
        if row_precision != pad_precision {
            return Err("pad_sequences_to requires homogeneous numeric nested lists".to_string());
        }
        rows.push(row);
    }
    let pad_value = match pad {
        RuntimeValue::Int(value) => *value as f64,
        RuntimeValue::Float(value) => *value,
        _ => unreachable!(),
    };
    let batch = rows.len();
    let mut data = Vec::with_capacity(batch * width);
    for row in rows {
        let used = row.len().min(width);
        data.extend(row.into_iter().take(used));
        data.extend(std::iter::repeat_n(pad_value, width.saturating_sub(used)));
    }
    Ok((pad_precision, data, batch))
}

fn tensor_numel(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

fn linear_to_indices(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut indices = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        indices[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
    indices
}

fn indices_to_linear(indices: &[usize], shape: &[usize]) -> usize {
    let mut linear = 0usize;
    for (axis, value) in indices.iter().enumerate() {
        linear *= shape[axis];
        linear += value;
    }
    linear
}

fn normalize_axis(rank: usize, axis: i64, op: &str) -> Result<usize, String> {
    if axis < 0 {
        return Err(format!("{op} requires non-negative axis, got {axis}"));
    }
    let axis = axis as usize;
    if axis >= rank {
        return Err(format!("{op} axis {axis} out of bounds for rank {rank}"));
    }
    Ok(axis)
}

fn expect_int_list(values: &[RuntimeValue], op: &str) -> Result<Vec<usize>, String> {
    values
        .iter()
        .map(|value| match value {
            RuntimeValue::Int(value) if *value >= 0 => Ok(*value as usize),
            RuntimeValue::Int(value) => {
                Err(format!("{op} expects non-negative sizes, got {value}"))
            }
            other => Err(format!("{op} expects int64 sizes, got {other:?}")),
        })
        .collect()
}

#[derive(Clone, Copy)]
enum ReduceOp {
    Sum,
    Min,
    Prod,
    Argmax,
    Argmin,
}

fn tensor_reduce_host(
    tensor: &RuntimeTensorValue,
    axis: i64,
    op: ReduceOp,
) -> Result<RuntimeTensorValue, String> {
    let rank = tensor.value.shape.len();
    let axis = normalize_axis(rank, axis, "reduction")?;
    let mut out_shape: Vec<usize> = tensor.value.shape.clone();
    let axis_len = out_shape.remove(axis);
    if axis_len == 0 {
        return Err("reduction over empty axis is undefined".to_string());
    }
    let out_numel = tensor_numel(&out_shape);
    let mut out = vec![0.0_f64; out_numel];
    #[allow(clippy::needless_range_loop)]
    for out_linear in 0..out_numel {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        let mut best_value = match op {
            ReduceOp::Sum => 0.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Prod => 1.0,
            ReduceOp::Argmax => f64::NEG_INFINITY,
            ReduceOp::Argmin => f64::INFINITY,
        };
        let mut best_index: usize = 0;
        for k in 0..axis_len {
            let mut in_indices = Vec::with_capacity(rank);
            let mut oi = 0;
            for dim in 0..rank {
                if dim == axis {
                    in_indices.push(k);
                } else {
                    in_indices.push(out_indices[oi]);
                    oi += 1;
                }
            }
            let in_linear = indices_to_linear(&in_indices, &tensor.value.shape);
            let value = tensor.value.data[in_linear];
            match op {
                ReduceOp::Sum => {
                    best_value += value;
                }
                ReduceOp::Min => {
                    if value < best_value {
                        best_value = value;
                    }
                }
                ReduceOp::Prod => {
                    best_value *= value;
                }
                ReduceOp::Argmax => {
                    if value > best_value {
                        best_value = value;
                        best_index = k;
                    }
                }
                ReduceOp::Argmin => {
                    if value < best_value {
                        best_value = value;
                        best_index = k;
                    }
                }
            }
        }
        out[out_linear] = match op {
            ReduceOp::Sum | ReduceOp::Min | ReduceOp::Prod => best_value,
            // Argmax/Argmin: store integer indices as integer-valued F32 per
            // the Phase 3j-pre Batch 1 caveat (documented on RiscOp::Argmax
            // and adv_argmax_output_stores_integer_valued_floats).
            ReduceOp::Argmax | ReduceOp::Argmin => best_index as f64,
        };
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
}

/// Permute axes of a tensor, given an `axes` permutation. `axes[i]` is the
/// source axis for output axis `i`.
fn tensor_permute_host(
    tensor: &RuntimeTensorValue,
    axes: &[usize],
) -> Result<RuntimeTensorValue, String> {
    let rank = tensor.value.shape.len();
    if axes.len() != rank {
        return Err(format!(
            "permute expects {rank} axis arguments for rank-{rank} tensor, got {}",
            axes.len()
        ));
    }
    let mut seen = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(format!("permute axis {axis} out of bounds for rank {rank}"));
        }
        if seen[axis] {
            return Err(format!("permute axes contain duplicate axis {axis}"));
        }
        seen[axis] = true;
    }
    let in_shape = tensor.value.shape.clone();
    let out_shape: Vec<usize> = axes.iter().map(|&a| in_shape[a]).collect();
    let out_numel = tensor_numel(&out_shape);
    let mut out = vec![0.0_f64; out_numel];
    for in_linear in 0..tensor.value.data.len() {
        let in_indices = linear_to_indices(in_linear, &in_shape);
        let out_indices: Vec<usize> = axes.iter().map(|&a| in_indices[a]).collect();
        let out_linear = indices_to_linear(&out_indices, &out_shape);
        out[out_linear] = tensor.value.data[in_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
}

/// 2D matmul: lhs is [m, k], rhs is [k, n], output is [m, n].
fn tensor_matmul_host(
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    if lhs.value.shape.len() != 2 || rhs.value.shape.len() != 2 {
        return Err(format!(
            "matmul host runtime currently supports only rank-2 × rank-2; got ranks {} and {}",
            lhs.value.shape.len(),
            rhs.value.shape.len()
        ));
    }
    let m = lhs.value.shape[0];
    let k_lhs = lhs.value.shape[1];
    let k_rhs = rhs.value.shape[0];
    let n = rhs.value.shape[1];
    if k_lhs != k_rhs {
        return Err(format!(
            "matmul shared-axis mismatch: lhs has {k_lhs}, rhs has {k_rhs}"
        ));
    }
    let mut out = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for kk in 0..k_lhs {
                acc += lhs.value.data[i * k_lhs + kk] * rhs.value.data[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![m, n], out),
        precision: lhs.precision,
    })
}

/// Replicate a tensor along a new axis. Matches the IR's `expand` semantics
/// when the output rank is `input_rank + 1`: `expand(b, axis, count)` produces
/// a tensor of shape `[..., count, ...]` (with `count` inserted at `axis`)
/// where every "slice" along the new axis is a copy of `b`. Also handles the
/// same-rank variant where the input axis has size 1 and is replicated to
/// `count`.
fn tensor_expand_host(
    tensor: &RuntimeTensorValue,
    axis: usize,
    count: usize,
) -> Result<RuntimeTensorValue, String> {
    let in_shape = tensor.value.shape.clone();
    let in_rank = in_shape.len();
    if axis > in_rank {
        return Err(format!(
            "expand axis {axis} out of bounds for rank-{in_rank} tensor (insert position must be <= rank)"
        ));
    }

    // Determine the output shape and the index-mapping mode.
    //
    // Mode A (insert): if `axis == in_rank` OR the existing axis at `axis`
    // is not 1, we INSERT a new axis of size `count` at position `axis`.
    // Mode B (replicate-singleton): if `axis < in_rank` and the existing
    // axis at `axis` is 1, we REPLACE that axis with size `count`.
    let (out_shape, same_rank) = if axis < in_rank && in_shape[axis] == 1 {
        let mut out = in_shape.clone();
        out[axis] = count;
        (out, true)
    } else {
        let mut out = Vec::with_capacity(in_rank + 1);
        out.extend_from_slice(&in_shape[..axis]);
        out.push(count);
        out.extend_from_slice(&in_shape[axis..]);
        (out, false)
    };

    let out_numel = tensor_numel(&out_shape);
    let mut out = vec![0.0_f64; out_numel];
    for (out_linear, slot) in out.iter_mut().enumerate() {
        let out_indices = linear_to_indices(out_linear, &out_shape);
        let in_indices: Vec<usize> = if same_rank {
            // Replicate singleton: input axis stays 0; other axes pass through.
            let mut idx = out_indices.clone();
            idx[axis] = 0;
            idx
        } else {
            // Insert: drop the inserted axis to recover the input index.
            let mut idx = out_indices;
            idx.remove(axis);
            idx
        };
        let in_linear = indices_to_linear(&in_indices, &in_shape);
        *slot = tensor.value.data[in_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
}

/// Numerically stable softmax along a single axis:
/// `softmax(x, axis)[i] = exp(x[i] - max(x, axis)) / sum_j exp(x[j] - max(x, axis))`.
/// Matches the spec §4.2 lowering used by `tier2::lower_softmax`.
///
/// Negative axes are normalized to `rank + axis` (e.g. `-1` is the last axis).
fn tensor_softmax_host(
    tensor: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeTensorValue, String> {
    let rank = tensor.value.shape.len();
    if rank == 0 {
        return Err("softmax requires a tensor of rank >= 1".to_string());
    }
    let axis_usize = if axis < 0 {
        let neg = (-axis) as usize;
        if neg > rank {
            return Err(format!("softmax axis {axis} out of bounds for rank {rank}"));
        }
        rank - neg
    } else {
        let a = axis as usize;
        if a >= rank {
            return Err(format!("softmax axis {axis} out of bounds for rank {rank}"));
        }
        a
    };

    let in_shape = tensor.value.shape.clone();
    let axis_size = in_shape[axis_usize];
    if axis_size == 0 {
        return Err("softmax axis has size 0".to_string());
    }
    let numel = tensor_numel(&in_shape);
    let mut out = vec![0.0_f64; numel];

    // Iterate over each "slice" along the reduced axis: for every combination
    // of the other axes, compute max -> exp(x - max) -> sum -> divide.
    let mut reduced_shape = in_shape.clone();
    reduced_shape[axis_usize] = 1;
    let reduced_numel = tensor_numel(&reduced_shape);

    for slice_linear in 0..reduced_numel {
        let mut base_indices = linear_to_indices(slice_linear, &reduced_shape);
        // First pass: max over the axis. Track positive-Inf positions
        // separately — `exp(+Inf - +Inf) = exp(NaN) = NaN` would otherwise
        // silently corrupt mask-style attention usage where the ones-hot
        // position is set to +Inf (red-team v0.2.6 HIGH).
        let mut max_val = f64::NEG_INFINITY;
        let mut pos_inf_count = 0usize;
        for k in 0..axis_size {
            base_indices[axis_usize] = k;
            let in_linear = indices_to_linear(&base_indices, &in_shape);
            let v = tensor.value.data[in_linear];
            if v.is_nan() {
                // NaN propagates: write NaN across the whole slice and
                // continue. This matches IEEE behavior of every other
                // numerical library (PyTorch / NumPy / JAX).
                for kk in 0..axis_size {
                    base_indices[axis_usize] = kk;
                    let l = indices_to_linear(&base_indices, &in_shape);
                    out[l] = f64::NAN;
                }
                // Restart the outer slice loop's bookkeeping cleanly.
                max_val = f64::NAN;
                break;
            }
            if v == f64::INFINITY {
                pos_inf_count += 1;
            }
            if v > max_val {
                max_val = v;
            }
        }
        if max_val.is_nan() {
            // NaN propagation handled above; nothing else to do for this slice.
            continue;
        }
        if pos_inf_count > 0 {
            // Standard formula yields exp(+Inf - +Inf) = NaN. Define the
            // softmax of a slice containing K positive-Inf values as
            // 1/K at each +Inf position and 0 elsewhere — the natural
            // limit as the input approaches the multi-Inf configuration.
            let share = 1.0_f64 / (pos_inf_count as f64);
            for k in 0..axis_size {
                base_indices[axis_usize] = k;
                let in_linear = indices_to_linear(&base_indices, &in_shape);
                out[in_linear] = if tensor.value.data[in_linear] == f64::INFINITY {
                    share
                } else {
                    0.0
                };
            }
            continue;
        }
        if max_val == f64::NEG_INFINITY {
            // All entries were -Inf. The standard formula yields
            // exp(-Inf - -Inf) = exp(NaN) = NaN; define this case as
            // uniform 1/N over the slice (the natural limit).
            let share = 1.0_f64 / (axis_size as f64);
            for k in 0..axis_size {
                base_indices[axis_usize] = k;
                let in_linear = indices_to_linear(&base_indices, &in_shape);
                out[in_linear] = share;
            }
            continue;
        }
        // Second pass: sum of exp(x - max).
        let mut sum_exp = 0.0_f64;
        for k in 0..axis_size {
            base_indices[axis_usize] = k;
            let in_linear = indices_to_linear(&base_indices, &in_shape);
            sum_exp += (tensor.value.data[in_linear] - max_val).exp();
        }
        if sum_exp == 0.0 {
            return Err("softmax sum-of-exp is zero (numerical underflow)".to_string());
        }
        // Third pass: write exp(x - max) / sum.
        for k in 0..axis_size {
            base_indices[axis_usize] = k;
            let in_linear = indices_to_linear(&base_indices, &in_shape);
            let numer = (tensor.value.data[in_linear] - max_val).exp();
            out[in_linear] = numer / sum_exp;
        }
    }

    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(in_shape, out),
        precision: tensor.precision,
    })
}

fn tensor_concat_value(parts: &[RuntimeValue], axis: i64) -> Result<RuntimeValue, String> {
    let tensors = parts
        .iter()
        .map(|value| match value {
            RuntimeValue::Tensor(tensor) => Ok(tensor.clone()),
            other => Err(format!("concat expects tensor parts, got {other:?}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let first = tensors
        .first()
        .ok_or_else(|| "concat expects at least one tensor part".to_string())?;
    let axis = normalize_axis(first.value.shape.len(), axis, "concat")?;
    for tensor in &tensors[1..] {
        if tensor.precision != first.precision {
            return Err("concat expects matching tensor precision".to_string());
        }
        if tensor.value.shape.len() != first.value.shape.len() {
            return Err("concat expects matching tensor rank".to_string());
        }
        for dim in 0..tensor.value.shape.len() {
            if dim != axis && tensor.value.shape[dim] != first.value.shape[dim] {
                return Err(format!(
                    "concat expects matching non-concatenated axes; axis {dim} differed"
                ));
            }
        }
    }
    let mut out_shape = first.value.shape.clone();
    out_shape[axis] = tensors.iter().map(|tensor| tensor.value.shape[axis]).sum();
    let mut out = vec![0.0; tensor_numel(&out_shape)];
    let mut axis_offset = 0usize;
    for tensor in &tensors {
        for linear in 0..tensor.value.data.len() {
            let mut index = linear_to_indices(linear, &tensor.value.shape);
            index[axis] += axis_offset;
            let out_linear = indices_to_linear(&index, &out_shape);
            out[out_linear] = tensor.value.data[linear];
        }
        axis_offset += tensor.value.shape[axis];
    }
    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: first.precision,
    }))
}

fn tensor_reshape_value(
    tensor: &RuntimeTensorValue,
    shape: &[RuntimeValue],
) -> Result<RuntimeTensorValue, String> {
    let new_shape = expect_int_list(shape, "reshape")?;
    let expected = new_shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| "reshape target shape overflows usize".to_string())?;
    if expected != tensor.value.data.len() {
        return Err(format!(
            "reshape expects {} elements but tensor has {}",
            expected,
            tensor.value.data.len()
        ));
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(new_shape, tensor.value.data.clone()),
        precision: tensor.precision,
    })
}

fn tensor_split_value(
    tensor: &RuntimeTensorValue,
    axis: i64,
    sizes: &[RuntimeValue],
) -> Result<RuntimeValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "split")?;
    let sizes = expect_int_list(sizes, "split")?;
    let total: usize = sizes.iter().sum();
    if total != tensor.value.shape[axis] {
        return Err(format!(
            "split sizes sum to {total}, expected {}",
            tensor.value.shape[axis]
        ));
    }
    let mut parts = Vec::with_capacity(sizes.len());
    let mut offset = 0usize;
    for size in sizes {
        let mut shape = tensor.value.shape.clone();
        shape[axis] = size;
        let mut data = vec![0.0; tensor_numel(&shape)];
        for (linear, slot) in data.iter_mut().enumerate() {
            let mut index = linear_to_indices(linear, &shape);
            index[axis] += offset;
            let src = indices_to_linear(&index, &tensor.value.shape);
            *slot = tensor.value.data[src];
        }
        offset += size;
        parts.push(RuntimeValue::Tensor(RuntimeTensorValue {
            value: IrTensorValue::from_vec(shape, data),
            precision: tensor.precision,
        }));
    }
    Ok(RuntimeValue::List(parts))
}

fn tensor_gather_value(
    tensor: &RuntimeTensorValue,
    indices: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeTensorValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "gather")?;
    if !indices.precision.is_integer() {
        return Err("gather expects integer tensor indices".to_string());
    }
    let mut out_shape = tensor.value.shape[..axis].to_vec();
    out_shape.extend_from_slice(&indices.value.shape);
    out_shape.extend_from_slice(&tensor.value.shape[axis + 1..]);
    let mut out = vec![0.0; tensor_numel(&out_shape)];
    for (linear, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_indices(linear, &out_shape);
        let mut src_index = Vec::with_capacity(tensor.value.shape.len());
        src_index.extend_from_slice(&out_index[..axis]);
        let gathered_idx = &out_index[axis..axis + indices.value.shape.len()];
        let index_linear = indices_to_linear(gathered_idx, &indices.value.shape);
        let value = indices.value.data[index_linear] as i64;
        if value < 0 || value as usize >= tensor.value.shape[axis] {
            return Err(format!("gather index {value} out of bounds at axis {axis}"));
        }
        src_index.push(value as usize);
        src_index.extend_from_slice(&out_index[axis + indices.value.shape.len()..]);
        let src_linear = indices_to_linear(&src_index, &tensor.value.shape);
        *slot = tensor.value.data[src_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
}

fn tensor_scatter_value(
    base: &RuntimeTensorValue,
    indices: &RuntimeTensorValue,
    updates: &RuntimeTensorValue,
    axis: i64,
    mode: &str,
) -> Result<RuntimeTensorValue, String> {
    let axis = normalize_axis(base.value.shape.len(), axis, "scatter")?;
    if !indices.precision.is_integer() {
        return Err("scatter expects integer tensor indices".to_string());
    }
    let expected = tensor_gather_value(base, indices, axis as i64)?;
    if expected.value.shape != updates.value.shape || expected.precision != updates.precision {
        return Err("scatter updates must match gathered tensor shape and precision".to_string());
    }
    let mut out = base.value.data.clone();
    let mut seen = std::collections::HashSet::new();
    for linear in 0..updates.value.data.len() {
        let update_index = linear_to_indices(linear, &updates.value.shape);
        let mut out_index = Vec::with_capacity(base.value.shape.len());
        out_index.extend_from_slice(&update_index[..axis]);
        let gathered_idx = &update_index[axis..axis + indices.value.shape.len()];
        let index_linear = indices_to_linear(gathered_idx, &indices.value.shape);
        let value = indices.value.data[index_linear] as i64;
        if value < 0 || value as usize >= base.value.shape[axis] {
            return Err(format!(
                "scatter index {value} out of bounds at axis {axis}"
            ));
        }
        out_index.push(value as usize);
        out_index.extend_from_slice(&update_index[axis + indices.value.shape.len()..]);
        let out_linear = indices_to_linear(&out_index, &base.value.shape);
        match mode {
            "replace" => {
                if !seen.insert(out_linear) {
                    return Err(format!(
                        "scatter replace mode rejects duplicate target index {}",
                        out_linear
                    ));
                }
                out[out_linear] = updates.value.data[linear];
            }
            "add" => out[out_linear] += updates.value.data[linear],
            other => return Err(format!("scatter mode must be replace or add, got {other}")),
        }
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(base.value.shape.clone(), out),
        precision: base.precision,
    })
}

fn tensor_where_value(
    cond: &RuntimeTensorValue,
    then_tensor: &RuntimeTensorValue,
    else_tensor: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    if cond.precision != Prim::Bool {
        return Err("where expects bool tensor condition".to_string());
    }
    if cond.value.shape != then_tensor.value.shape
        || then_tensor.value.shape != else_tensor.value.shape
    {
        return Err(
            "where expects condition and both branches to have identical shape".to_string(),
        );
    }
    if then_tensor.precision != else_tensor.precision {
        return Err("where expects matching branch precision".to_string());
    }
    let data = cond
        .value
        .data
        .iter()
        .zip(&then_tensor.value.data)
        .zip(&else_tensor.value.data)
        .map(|((cond, then_value), else_value)| {
            if *cond != 0.0 {
                *then_value
            } else {
                *else_value
            }
        })
        .collect();
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(then_tensor.value.shape.clone(), data),
        precision: then_tensor.precision,
    })
}

fn tensor_cumsum_value(
    tensor: &RuntimeTensorValue,
    axis: i64,
) -> Result<RuntimeTensorValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "cumsum")?;
    let mut data = tensor.value.data.clone();
    let axis_size = tensor.value.shape[axis];
    let inner: usize = tensor.value.shape[axis + 1..]
        .iter()
        .product::<usize>()
        .max(1);
    let outer: usize = tensor.value.shape[..axis].iter().product::<usize>().max(1);
    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            let mut running = 0.0;
            for axis_idx in 0..axis_size {
                let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                running += data[linear];
                data[linear] = running;
            }
        }
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(tensor.value.shape.clone(), data),
        precision: tensor.precision,
    })
}

fn tensor_sort_value(tensor: &RuntimeTensorValue, axis: i64) -> Result<RuntimeValue, String> {
    let axis = normalize_axis(tensor.value.shape.len(), axis, "sort")?;
    let axis_size = tensor.value.shape[axis];
    let inner: usize = tensor.value.shape[axis + 1..]
        .iter()
        .product::<usize>()
        .max(1);
    let outer: usize = tensor.value.shape[..axis].iter().product::<usize>().max(1);
    let mut values = tensor.value.data.clone();
    let mut indices = vec![0.0; tensor.value.data.len()];
    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            let mut items = (0..axis_size)
                .map(|axis_idx| {
                    let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                    (axis_idx, values[linear])
                })
                .collect::<Vec<_>>();
            items.sort_by(|(lhs_idx, lhs_val), (rhs_idx, rhs_val)| {
                lhs_val
                    .partial_cmp(rhs_val)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(lhs_idx.cmp(rhs_idx))
            });
            for (sorted_idx, (original_idx, value)) in items.into_iter().enumerate() {
                let linear = (outer_idx * axis_size + sorted_idx) * inner + inner_idx;
                values[linear] = value;
                indices[linear] = original_idx as f64;
            }
        }
    }
    Ok(RuntimeValue::Tuple(vec![
        RuntimeValue::Tensor(RuntimeTensorValue {
            value: IrTensorValue::from_vec(tensor.value.shape.clone(), values),
            precision: tensor.precision,
        }),
        RuntimeValue::Tensor(RuntimeTensorValue {
            value: IrTensorValue::from_vec(tensor.value.shape.clone(), indices),
            precision: Prim::Int64,
        }),
    ]))
}

fn tensor_diagonal_value(
    tensor: &RuntimeTensorValue,
    axis1: i64,
    axis2: i64,
) -> Result<RuntimeTensorValue, String> {
    let axis1 = normalize_axis(tensor.value.shape.len(), axis1, "diagonal")?;
    let axis2 = normalize_axis(tensor.value.shape.len(), axis2, "diagonal")?;
    if axis1 == axis2 {
        return Err("diagonal expects distinct axes".to_string());
    }
    let diag = tensor.value.shape[axis1].min(tensor.value.shape[axis2]);
    let mut out_shape = Vec::with_capacity(tensor.value.shape.len() - 1);
    for (index, size) in tensor.value.shape.iter().enumerate() {
        if index == axis1 {
            out_shape.push(diag);
        } else if index != axis2 {
            out_shape.push(*size);
        }
    }
    let mut data = vec![0.0; tensor_numel(&out_shape)];
    for (linear, slot) in data.iter_mut().enumerate() {
        let out_index = linear_to_indices(linear, &out_shape);
        let mut src_index = Vec::with_capacity(tensor.value.shape.len());
        let mut out_pos = 0usize;
        let diag_idx = out_index[axis1];
        for index in 0..tensor.value.shape.len() {
            if index == axis1 || index == axis2 {
                src_index.push(diag_idx);
            } else {
                src_index.push(out_index[out_pos]);
                out_pos += 1;
            }
        }
        let src_linear = indices_to_linear(&src_index, &tensor.value.shape);
        *slot = tensor.value.data[src_linear];
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, data),
        precision: tensor.precision,
    })
}

fn tensor_trace_value(
    tensor: &RuntimeTensorValue,
    axis1: i64,
    axis2: i64,
) -> Result<RuntimeTensorValue, String> {
    let diagonal = tensor_diagonal_value(tensor, axis1, axis2)?;
    let rank = diagonal.value.shape.len();
    let axis = normalize_axis(rank, axis1.min(axis2), "trace").unwrap_or(rank.saturating_sub(1));
    let axis_size = diagonal.value.shape[axis];
    let inner: usize = diagonal.value.shape[axis + 1..]
        .iter()
        .product::<usize>()
        .max(1);
    let outer: usize = diagonal.value.shape[..axis]
        .iter()
        .product::<usize>()
        .max(1);
    let out_shape = diagonal
        .value
        .shape
        .iter()
        .enumerate()
        .filter_map(|(idx, size)| (idx != axis).then_some(*size))
        .collect::<Vec<_>>();
    let mut out = vec![0.0; tensor_numel(&out_shape)];
    for outer_idx in 0..outer {
        for inner_idx in 0..inner {
            let mut sum = 0.0;
            for axis_idx in 0..axis_size {
                let linear = (outer_idx * axis_size + axis_idx) * inner + inner_idx;
                sum += diagonal.value.data[linear];
            }
            let out_linear = outer_idx * inner + inner_idx;
            out[out_linear] = sum;
        }
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: tensor.precision,
    })
}

fn tensor_clamp_value(
    tensor: &RuntimeTensorValue,
    lo: &RuntimeTensorValue,
    hi: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    let scalar_or_match = |bound: &RuntimeTensorValue| {
        bound.value.shape.is_empty() || bound.value.shape == tensor.value.shape
    };
    if tensor.precision != lo.precision || tensor.precision != hi.precision {
        return Err("clamp expects matching tensor precision".to_string());
    }
    if !scalar_or_match(lo) || !scalar_or_match(hi) {
        return Err(
            "clamp expects scalar tensor bounds or matching-shape tensor bounds".to_string(),
        );
    }
    let mut out = Vec::with_capacity(tensor.value.data.len());
    for linear in 0..tensor.value.data.len() {
        let lo_value = if lo.value.shape.is_empty() {
            lo.value.data[0]
        } else {
            lo.value.data[linear]
        };
        let hi_value = if hi.value.shape.is_empty() {
            hi.value.data[0]
        } else {
            hi.value.data[linear]
        };
        out.push(tensor.value.data[linear].clamp(lo_value, hi_value));
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(tensor.value.shape.clone(), out),
        precision: tensor.precision,
    })
}

fn tensor_einsum_value(
    equation: &str,
    lhs: &RuntimeTensorValue,
    rhs: &RuntimeTensorValue,
) -> Result<RuntimeTensorValue, String> {
    if equation.contains("...") {
        return Err("einsum ellipsis support is deferred in 3h".to_string());
    }
    let (inputs, output) = equation
        .split_once("->")
        .ok_or_else(|| "einsum equation must contain explicit output".to_string())?;
    let operands = inputs.split(',').collect::<Vec<_>>();
    if operands.len() != 2 {
        return Err("einsum 3h currently supports exactly two operands".to_string());
    }
    let lhs_labels = operands[0].chars().collect::<Vec<_>>();
    let rhs_labels = operands[1].chars().collect::<Vec<_>>();
    let out_labels = output.chars().collect::<Vec<_>>();
    if lhs_labels.len() != lhs.value.shape.len() || rhs_labels.len() != rhs.value.shape.len() {
        return Err("einsum label count must match operand rank".to_string());
    }
    let mut dims = std::collections::BTreeMap::<char, usize>::new();
    for (label, size) in lhs_labels.iter().zip(&lhs.value.shape) {
        if let Some(prev) = dims.insert(*label, *size)
            && prev != *size
        {
            return Err(format!("einsum label `{label}` has inconsistent extents"));
        }
    }
    for (label, size) in rhs_labels.iter().zip(&rhs.value.shape) {
        if let Some(prev) = dims.insert(*label, *size)
            && prev != *size
        {
            return Err(format!("einsum label `{label}` has inconsistent extents"));
        }
    }
    let out_shape = out_labels
        .iter()
        .map(|label| {
            dims.get(label)
                .copied()
                .ok_or_else(|| format!("einsum output label `{label}` missing from inputs"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut reduction_labels = Vec::<char>::new();
    for label in lhs_labels.iter().chain(rhs_labels.iter()) {
        if !out_labels.contains(label) && !reduction_labels.contains(label) {
            reduction_labels.push(*label);
        }
    }
    let reduction_shape = reduction_labels
        .iter()
        .map(|label| dims.get(label).copied().unwrap_or(1))
        .collect::<Vec<_>>();
    let mut out = vec![0.0; tensor_numel(&out_shape)];
    for (out_linear, slot) in out.iter_mut().enumerate() {
        let out_index = linear_to_indices(out_linear, &out_shape);
        let mut label_values = std::collections::HashMap::<char, usize>::new();
        for (label, value) in out_labels.iter().zip(out_index.iter()) {
            label_values.insert(*label, *value);
        }
        let reduction_total = tensor_numel(&reduction_shape);
        let mut acc = 0.0;
        for reduction_linear in 0..reduction_total {
            let reduction_index = linear_to_indices(reduction_linear, &reduction_shape);
            for (label, value) in reduction_labels.iter().zip(reduction_index.iter()) {
                label_values.insert(*label, *value);
            }
            let lhs_index = lhs_labels
                .iter()
                .map(|label| label_values[label])
                .collect::<Vec<_>>();
            let rhs_index = rhs_labels
                .iter()
                .map(|label| label_values[label])
                .collect::<Vec<_>>();
            acc += lhs.value.data[indices_to_linear(&lhs_index, &lhs.value.shape)]
                * rhs.value.data[indices_to_linear(&rhs_index, &rhs.value.shape)];
        }
        *slot = acc;
    }
    Ok(RuntimeTensorValue {
        value: IrTensorValue::from_vec(out_shape, out),
        precision: lhs.precision,
    })
}

fn render_value(value: &RuntimeValue) -> String {
    match value {
        RuntimeValue::Tensor(tensor) => format!(
            "tensor(shape={:?}, data={:?})",
            tensor.value.shape, tensor.value.data
        ),
        RuntimeValue::Int(value) => value.to_string(),
        RuntimeValue::Float(value) => value.to_string(),
        RuntimeValue::Bool(value) => value.to_string(),
        RuntimeValue::String(value) => value.clone(),
        RuntimeValue::List(items) => format!(
            "[{}]",
            items
                .iter()
                .map(render_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        RuntimeValue::Dict(entries) => format!(
            "dict({})",
            entries
                .iter()
                .map(|(key, value)| format!("{}: {}", render_value(key), render_value(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        RuntimeValue::Tuple(items) => format!(
            "({})",
            items
                .iter()
                .map(render_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        RuntimeValue::Adt { ctor, fields, .. } if fields.is_empty() => ctor.clone(),
        RuntimeValue::Adt { ctor, fields, .. } => format!(
            "{}({})",
            ctor,
            fields
                .iter()
                .map(render_value)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        RuntimeValue::MappedFile(bytes) => format!("<mapped-file:{}>", bytes.len()),
        RuntimeValue::Closure { .. } => "<closure>".to_string(),
        RuntimeValue::Unit => "()".to_string(),
    }
}

fn builtin_name(expr: &Expr) -> Option<&str> {
    let name = var_name(expr)?;
    BUILTIN_NAMES.contains(&name).then_some(name)
}

fn dropout_sample(seed: u64, index: u64) -> f64 {
    let mut x = seed ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    ((x >> 11) as f64) / ((1u64 << 53) as f64)
}

fn uniform_like_value(
    template: &RuntimeTensorValue,
    low: f64,
    high: f64,
    seed: u64,
) -> RuntimeTensorValue {
    let span = high - low;
    let data = template
        .value
        .data
        .iter()
        .enumerate()
        .map(|(index, _)| low + span * dropout_sample(seed, index as u64))
        .collect::<Vec<_>>();
    RuntimeTensorValue {
        value: IrTensorValue::from_vec(template.value.shape.clone(), data),
        precision: template.precision,
    }
}

fn var_name(expr: &Expr) -> Option<&str> {
    let list = as_list(expr)?;
    if tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn runtime_param_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        Expr::MetaExpr(meta, _) => runtime_param_name(&meta.expr),
        Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .or_else(|| children(list).first().and_then(symbol_name)),
        _ => None,
    }
}

fn as_list(expr: &Expr) -> Option<&List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn tag(list: &List) -> Option<&str> {
    match list.elements.first() {
        Some(Expr::Atom(Atom::Symbol(tag), _)) => Some(tag.as_str()),
        _ => None,
    }
}

fn get_meta(list: &List) -> Option<&MetaMap> {
    match list.elements.get(1) {
        Some(Expr::Map(map, _)) => Some(map),
        _ => None,
    }
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn int_value(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(value), _) => Some(*value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checked_surf(source: &str) -> CheckedProgram {
        let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        chelis_types::check_phase0e_program(&exprs).expect("phase0e check")
    }

    #[test]
    fn with_seed_uniform_like_evaluates_body() {
        let checked = checked_surf(
            r#"
x = with seed(7) {
  tensor_to_scalar(
    uniform_like(
      trace(pad_sequences_to([[0.0]], cast(1, int64), cast(0.0, f32)), cast(0, int32), cast(1, int32)),
      0.0,
      1.0
    )
  )
}
"#,
        );

        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("seeded host program should evaluate");
        let value = outcome.host_bindings.get("x").expect("x binding");
        match value {
            RuntimeValue::Float(v) => assert!((*v >= 0.0) && (*v <= 1.0), "got {v}"),
            other => panic!("expected float result, got {other:?}"),
        }
    }

    // ----- Phase 3t.1: test_assert_* builtins -----

    #[test]
    fn test_assert_true_returns_unit() {
        let checked = checked_surf(r#"x = test_assert(true, "ok")"#);
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("test_assert(true, ...) should evaluate to Ok");
        let value = outcome.host_bindings.get("x").expect("x binding");
        assert!(matches!(value, RuntimeValue::Unit), "got {value:?}");
    }

    #[test]
    fn test_assert_false_returns_err_with_label() {
        let checked = checked_surf(r#"x = test_assert(false, "my-label")"#);
        let err = evaluate_host_program(&checked, &HashMap::new())
            .expect_err("test_assert(false, ...) should surface as host Err");
        assert!(
            err.contains("assert failed") && err.contains("my-label"),
            "expected 'assert failed' and 'my-label' in error, got: {err}"
        );
    }

    #[test]
    fn test_assert_eq_f32_mismatch_includes_actual_and_expected() {
        let checked = checked_surf(r#"x = test_assert_eq_f32(1.0, 2.0, "label")"#);
        let err = evaluate_host_program(&checked, &HashMap::new())
            .expect_err("mismatched f32 assert should surface as host Err");
        assert!(err.contains("1") && err.contains("2"), "got: {err}");
        assert!(err.contains("label"), "expected label in error, got: {err}");
    }

    #[test]
    fn test_assert_eq_f32_match_returns_unit() {
        let checked = checked_surf(r#"x = test_assert_eq_f32(1.5, 1.5, "same")"#);
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("matched f32 assert should evaluate");
        let value = outcome.host_bindings.get("x").expect("x binding");
        assert!(matches!(value, RuntimeValue::Unit), "got {value:?}");
    }

    #[test]
    fn test_assert_eq_int_match_and_mismatch() {
        let ok = checked_surf(r#"x = test_assert_eq_int(cast(3, int64), cast(3, int64), "i")"#);
        let outcome = evaluate_host_program(&ok, &HashMap::new()).expect("int match should eval");
        assert!(matches!(
            outcome.host_bindings.get("x"),
            Some(RuntimeValue::Unit)
        ));

        let bad = checked_surf(r#"x = test_assert_eq_int(cast(3, int64), cast(5, int64), "i")"#);
        let err = evaluate_host_program(&bad, &HashMap::new())
            .expect_err("int mismatch should surface Err");
        assert!(
            err.contains("3") && err.contains("5") && err.contains("i"),
            "got: {err}"
        );
    }

    #[test]
    fn test_assert_eq_bool_match_and_mismatch() {
        let ok = checked_surf(r#"x = test_assert_eq_bool(true, true, "b")"#);
        evaluate_host_program(&ok, &HashMap::new()).expect("bool match should eval");

        let bad = checked_surf(r#"x = test_assert_eq_bool(true, false, "b")"#);
        let err = evaluate_host_program(&bad, &HashMap::new())
            .expect_err("bool mismatch should surface Err");
        assert!(
            err.contains("true") && err.contains("false") && err.contains("b"),
            "got: {err}"
        );
    }

    #[test]
    fn test_assert_eq_string_match_and_mismatch() {
        let ok = checked_surf(r#"x = test_assert_eq_string("hi", "hi", "s")"#);
        evaluate_host_program(&ok, &HashMap::new()).expect("string match should eval");

        let bad = checked_surf(r#"x = test_assert_eq_string("foo", "bar", "s")"#);
        let err = evaluate_host_program(&bad, &HashMap::new())
            .expect_err("string mismatch should surface Err");
        assert!(
            err.contains("foo") && err.contains("bar") && err.contains("s"),
            "got: {err}"
        );
    }

    #[test]
    fn test_assert_close_tensor_reports_first_mismatch_index() {
        // Use 4-element tensors [1.0, 2.0, 3.0, 4.0] vs [1.0, 2.0, 99.0, 4.0]:
        // index 2 is the first mismatch.
        let checked = checked_surf(
            r#"
actual: tensor[4, f32] = to_tensor([1.0, 2.0, 3.0, 4.0])
expected: tensor[4, f32] = to_tensor([1.0, 2.0, 99.0, 4.0])
x = test_assert_close_tensor(actual, expected, 0.001, "close")
"#,
        );
        let err = evaluate_host_program(&checked, &HashMap::new())
            .expect_err("close-tensor mismatch should surface Err");
        assert!(
            err.contains("at index 2") && err.contains("close"),
            "expected 'at index 2' and label, got: {err}"
        );
        assert!(err.contains("99") && err.contains('3'), "got: {err}");
    }

    #[test]
    fn test_assert_close_tensor_match_returns_unit() {
        let checked = checked_surf(
            r#"
actual: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])
expected: tensor[3, f32] = to_tensor([1.001, 2.001, 3.001])
x = test_assert_close_tensor(actual, expected, 0.01, "close")
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("close-tensor within tol should evaluate to Unit");
        assert!(matches!(
            outcome.host_bindings.get("x"),
            Some(RuntimeValue::Unit)
        ));
    }

    #[test]
    fn test_assert_close_tensor_zero_tol_passes_bit_exact() {
        // Regression: tol = 0 with identical data must pass, not report a false mismatch.
        let checked = checked_surf(
            r#"
actual: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])
expected: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])
x = test_assert_close_tensor(actual, expected, 0.0, "bit-exact")
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("zero-tol bit-exact equality should pass");
        assert!(matches!(
            outcome.host_bindings.get("x"),
            Some(RuntimeValue::Unit)
        ));
    }

    #[test]
    fn test_assert_close_tensor_zero_tol_rejects_any_delta() {
        let checked = checked_surf(
            r#"
actual: tensor[2, f32] = to_tensor([1.0, 2.0])
expected: tensor[2, f32] = to_tensor([1.0, 2.00001])
x = test_assert_close_tensor(actual, expected, 0.0, "strict")
"#,
        );
        let err = evaluate_host_program(&checked, &HashMap::new())
            .expect_err("zero-tol with any delta must fail");
        assert!(
            err.contains("at index 1") && err.contains("strict"),
            "got: {err}"
        );
    }

    #[test]
    fn test_assert_close_tensor_nan_actual_fails() {
        // Regression: NaN in actual must fail. (NaN - x).abs() is NaN, which silently
        // passes the old `>= tol` check. sqrt(-1.0) produces NaN.
        let checked = checked_surf(
            r#"
nan_val: f32 = sqrt(-1.0)
actual: tensor[2, f32] = to_tensor([1.0, nan_val])
expected: tensor[2, f32] = to_tensor([1.0, 2.0])
x = test_assert_close_tensor(actual, expected, 0.01, "nan-actual")
"#,
        );
        let err =
            evaluate_host_program(&checked, &HashMap::new()).expect_err("NaN in actual must fail");
        assert!(
            err.contains("at index 1") && err.contains("NaN"),
            "expected NaN-aware diagnostic, got: {err}"
        );
    }

    #[test]
    fn test_assert_close_tensor_nan_expected_fails() {
        let checked = checked_surf(
            r#"
nan_val: f32 = sqrt(-1.0)
actual: tensor[2, f32] = to_tensor([1.0, 2.0])
expected: tensor[2, f32] = to_tensor([1.0, nan_val])
x = test_assert_close_tensor(actual, expected, 0.01, "nan-expected")
"#,
        );
        let err = evaluate_host_program(&checked, &HashMap::new())
            .expect_err("NaN in expected must fail");
        assert!(
            err.contains("at index 1") && err.contains("NaN"),
            "got: {err}"
        );
    }

    #[test]
    fn test_assert_close_tensor_negative_tol_rejected() {
        let checked = checked_surf(
            r#"
actual: tensor[2, f32] = to_tensor([1.0, 2.0])
expected: tensor[2, f32] = to_tensor([1.0, 2.0])
x = test_assert_close_tensor(actual, expected, -0.001, "neg-tol")
"#,
        );
        let err = evaluate_host_program(&checked, &HashMap::new())
            .expect_err("negative tol must be rejected");
        assert!(
            err.contains("invalid tolerance") && err.contains("neg-tol"),
            "got: {err}"
        );
    }

    #[test]
    fn test_assert_close_tensor_nan_tol_rejected() {
        let checked = checked_surf(
            r#"
nan_tol: f32 = sqrt(-1.0)
actual: tensor[2, f32] = to_tensor([1.0, 2.0])
expected: tensor[2, f32] = to_tensor([1.0, 4.0])
x = test_assert_close_tensor(actual, expected, nan_tol, "nan-tol")
"#,
        );
        let err =
            evaluate_host_program(&checked, &HashMap::new()).expect_err("NaN tol must be rejected");
        assert!(
            err.contains("invalid tolerance") && err.contains("nan-tol"),
            "got: {err}"
        );
    }

    // ----- N2 fix: matmul / permute / sum host evaluator coverage -----
    //
    // Pins the closure of the upstream-reported gap: native `chelis test`
    // erroring with `unsupported builtin 'matmul'` / `'permute'` / `'sum'`
    // when those primitives appear in a test's dependency graph.

    fn first_tensor_data(outcome: &RuntimeOutcome, name: &str) -> Vec<f64> {
        match outcome.host_bindings.get(name) {
            Some(RuntimeValue::Tensor(t)) => t.value.data.clone(),
            other => panic!("expected tensor binding {name}, got {other:?}"),
        }
    }

    fn first_tensor_shape(outcome: &RuntimeOutcome, name: &str) -> Vec<usize> {
        match outcome.host_bindings.get(name) {
            Some(RuntimeValue::Tensor(t)) => t.value.shape.clone(),
            other => panic!("expected tensor binding {name}, got {other:?}"),
        }
    }

    #[test]
    fn host_runtime_matmul_2x2_identity_passthrough() {
        let checked = checked_surf(
            r#"
a = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32))
b = pad_sequences_to([[cast(3.0, f32), cast(5.0, f32)], [cast(7.0, f32), cast(11.0, f32)]], cast(2, int64), cast(0.0, f32))
y = matmul(a, b)
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("matmul should evaluate under host runtime");
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
        assert_eq!(first_tensor_data(&outcome, "y"), vec![3.0, 5.0, 7.0, 11.0]);
    }

    #[test]
    fn host_runtime_matmul_2x3_3x2_basic() {
        let checked = checked_surf(
            r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
b = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)], [cast(1.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32))
y = matmul(a, b)
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("rectangular matmul should evaluate");
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
        // Row 0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
        // Row 1: [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
        assert_eq!(first_tensor_data(&outcome, "y"), vec![4.0, 5.0, 10.0, 11.0]);
    }

    #[test]
    fn host_runtime_permute_2x2_transpose_swaps_off_diagonal() {
        let checked = checked_surf(
            r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]], cast(2, int64), cast(0.0, f32))
y = permute(a, 1, 0)
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("permute should evaluate under host runtime");
        // Row-major: original [[1,2],[3,4]] -> transpose [[1,3],[2,4]]
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
        assert_eq!(first_tensor_data(&outcome, "y"), vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn host_runtime_permute_2x3_transpose_to_3x2() {
        let checked = checked_surf(
            r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
y = permute(a, 1, 0)
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("rectangular permute should evaluate");
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![3, 2]);
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        assert_eq!(
            first_tensor_data(&outcome, "y"),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn host_runtime_sum_axis1_reduces_2x3_to_2() {
        let checked = checked_surf(
            r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
y = sum(a, cast(1, int32))
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("sum should evaluate under host runtime");
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![2]);
        assert_eq!(first_tensor_data(&outcome, "y"), vec![6.0, 15.0]);
    }

    #[test]
    fn host_runtime_sum_axis0_reduces_2x3_to_3() {
        let checked = checked_surf(
            r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
y = sum(a, cast(0, int32))
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("sum on axis 0 should evaluate");
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![3]);
        assert_eq!(first_tensor_data(&outcome, "y"), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn host_runtime_matmul_shared_axis_mismatch_errors() {
        // Build a 2x3 and a 2x2 — shared axis is 3 vs 2, must fail.
        let checked = checked_surf(
            r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
b = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32))
y = matmul(a, b)
"#,
        );
        let err = evaluate_host_program(&checked, &HashMap::new())
            .expect_err("matmul shared-axis mismatch must fail");
        assert!(
            err.contains("matmul") && err.contains("mismatch"),
            "expected matmul shared-axis diagnostic, got: {err}"
        );
    }

    // ----- expand / softmax host evaluator coverage (#38 follow-up) -----
    //
    // Pins the closure of the second host-runtime gap from the N2 fix:
    // `chelis test` / `chelis eval` erroring with `unsupported builtin
    // 'expand'` / `'softmax'` when those primitives appear in a test's
    // dependency graph (Std.Nn.Linear, Std.Nn.Attention, Std.Loss.CrossEntropy).

    #[test]
    fn host_runtime_expand_inserts_new_leading_axis() {
        // Linear.forward calls `expand(b, 0, batch)` where `b` is a 1-D
        // bias [out_dim] and the output is [batch, out_dim]. Pin that.
        let checked = checked_surf(
            r#"
b = to_tensor([cast(10.0, f32), cast(100.0, f32)])
y = expand(b, cast(0, int32), cast(3, int32))
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("expand should evaluate under host runtime");
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![3, 2]);
        // Three replicas of [10, 100].
        assert_eq!(
            first_tensor_data(&outcome, "y"),
            vec![10.0, 100.0, 10.0, 100.0, 10.0, 100.0]
        );
    }

    #[test]
    fn host_runtime_expand_inserts_trailing_axis() {
        // axis == rank inserts a new last axis.
        let checked = checked_surf(
            r#"
b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
y = expand(b, cast(1, int32), cast(2, int32))
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("expand at trailing axis should evaluate");
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
        // [1, 2] expanded along new last axis with count 2 -> [[1,1],[2,2]].
        assert_eq!(first_tensor_data(&outcome, "y"), vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn host_runtime_expand_negative_count_errors() {
        let checked = checked_surf(
            r#"
b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
y = expand(b, cast(0, int32), cast(0, int32))
"#,
        );
        let err = evaluate_host_program(&checked, &HashMap::new())
            .expect_err("expand with non-positive count must fail");
        assert!(
            err.contains("expand") && err.contains("count"),
            "expected expand count diagnostic, got: {err}"
        );
    }

    #[test]
    fn host_runtime_softmax_uniform_input_is_uniform_output() {
        // softmax of all-zeros along axis 0 of length 3 is [1/3, 1/3, 1/3].
        let checked = checked_surf(
            r#"
x = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
y = softmax(x, cast(0, int32))
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("softmax should evaluate under host runtime");
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![3]);
        let data = first_tensor_data(&outcome, "y");
        for value in &data {
            assert!(
                (*value - 1.0 / 3.0).abs() < 1e-9,
                "uniform softmax element should be 1/3, got {value}"
            );
        }
    }

    #[test]
    fn host_runtime_softmax_two_class_matches_reference() {
        // softmax([1.0, 0.0], 0) = [exp(1)/(exp(1)+1), 1/(exp(1)+1)]
        //                         ≈ [0.7310585, 0.2689414]
        let checked = checked_surf(
            r#"
x = to_tensor([cast(1.0, f32), cast(0.0, f32)])
y = softmax(x, cast(0, int32))
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("softmax 2-class should evaluate");
        assert_eq!(first_tensor_shape(&outcome, "y"), vec![2]);
        let data = first_tensor_data(&outcome, "y");
        let e = 1.0_f64.exp();
        let expected = [e / (e + 1.0), 1.0 / (e + 1.0)];
        for (got, want) in data.iter().zip(expected.iter()) {
            assert!(
                (*got - *want).abs() < 1e-6,
                "softmax 2-class mismatch: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn host_runtime_softmax_numerical_stability_handles_large_inputs() {
        // Without the max-subtraction trick, exp(1000) would overflow to
        // inf and produce NaN. The stable lowering must still produce
        // a normalized distribution.
        let checked = checked_surf(
            r#"
x = to_tensor([cast(1000.0, f32), cast(1000.0, f32)])
y = softmax(x, cast(0, int32))
"#,
        );
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .expect("softmax with large inputs should remain numerically stable");
        let data = first_tensor_data(&outcome, "y");
        assert_eq!(data.len(), 2);
        for value in &data {
            assert!(
                (*value - 0.5).abs() < 1e-9,
                "softmax([1000,1000]) should be uniform 0.5, got {value}"
            );
        }
    }

    #[test]
    fn host_runtime_softmax_positive_infinity_promotes_to_one_hot() {
        // Red-team v0.2.6 HIGH: `softmax([+Inf, 0, 0])` previously returned
        // `[NaN, 0, 0]` because the standard max-shift formula computes
        // `exp(+Inf - +Inf) = exp(NaN) = NaN`. Mask-style users (who set
        // ones-hot positions to +Inf) silently corrupted to NaN downstream.
        // The fix detects +Inf in the slice and emits 1/K at +Inf positions.
        let inf = f64::INFINITY;
        let tensor = RuntimeTensorValue {
            value: IrTensorValue::from_vec(vec![3], vec![inf, 0.0, 0.0]),
            precision: Prim::F32,
        };
        let out =
            tensor_softmax_host(&tensor, 0).expect("+Inf softmax must not error in host runtime");
        assert_eq!(out.value.data.len(), 3);
        assert!(out.value.data.iter().all(|v| !v.is_nan()), "no NaN allowed");
        assert!((out.value.data[0] - 1.0).abs() < 1e-9);
        assert!(out.value.data[1].abs() < 1e-9);
        assert!(out.value.data[2].abs() < 1e-9);
    }

    #[test]
    fn host_runtime_softmax_two_positive_infinities_split_uniformly() {
        let inf = f64::INFINITY;
        let tensor = RuntimeTensorValue {
            value: IrTensorValue::from_vec(vec![3], vec![inf, inf, 0.0]),
            precision: Prim::F32,
        };
        let out = tensor_softmax_host(&tensor, 0).expect("two +Inf softmax must not error");
        assert!((out.value.data[0] - 0.5).abs() < 1e-9);
        assert!((out.value.data[1] - 0.5).abs() < 1e-9);
        assert!(out.value.data[2].abs() < 1e-9);
    }

    #[test]
    fn host_runtime_softmax_all_negative_infinity_yields_uniform() {
        // All -Inf collapses to NaN under the standard formula too. Define
        // the natural limit: uniform 1/N (same as if all values were equal).
        let neg_inf = f64::NEG_INFINITY;
        let tensor = RuntimeTensorValue {
            value: IrTensorValue::from_vec(vec![3], vec![neg_inf, neg_inf, neg_inf]),
            precision: Prim::F32,
        };
        let out = tensor_softmax_host(&tensor, 0).expect("all -Inf softmax must not error");
        let third = 1.0 / 3.0;
        assert!(out.value.data.iter().all(|v| (*v - third).abs() < 1e-9));
    }

    #[test]
    fn host_runtime_softmax_nan_input_propagates_nan() {
        // NaN is contagious by spec; matches PyTorch / NumPy / JAX behavior.
        // We pin this so a future refactor doesn't accidentally mask it.
        let nan = f64::NAN;
        let tensor = RuntimeTensorValue {
            value: IrTensorValue::from_vec(vec![3], vec![nan, 0.0, 0.0]),
            precision: Prim::F32,
        };
        let out = tensor_softmax_host(&tensor, 0).expect("NaN softmax does not error");
        assert!(
            out.value.data.iter().all(|v| v.is_nan()),
            "NaN must propagate to every output element; got {:?}",
            out.value.data
        );
    }

    #[test]
    fn host_runtime_softmax_axis_out_of_bounds_errors() {
        let checked = checked_surf(
            r#"
x = to_tensor([cast(1.0, f32), cast(2.0, f32)])
y = softmax(x, cast(5, int32))
"#,
        );
        let err = evaluate_host_program(&checked, &HashMap::new())
            .expect_err("softmax with out-of-bounds axis must fail");
        assert!(
            err.contains("softmax") && err.contains("out of bounds"),
            "expected softmax axis-bounds diagnostic, got: {err}"
        );
    }
}
