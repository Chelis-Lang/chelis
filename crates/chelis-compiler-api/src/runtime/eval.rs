use chelis_deep::DeepTag;
use std::collections::HashMap;
use std::fs;

use chelis_deep::ast::{Atom, Expr, List};
use chelis_deep::{Span, decode_effect_kind};
use chelis_ir::eval::TensorValue as IrTensorValue;
use chelis_ir::tier2;
use chelis_types::types::Prim;
use chelis_vocab::EffectKind;

use super::host_ops::*;
use super::named_axis::*;
use super::transforms::*;
use super::*;

impl<'a> EvalContext<'a> {
    pub(super) fn resolve_top_level(&mut self, name: &str) -> Result<RuntimeValue, String> {
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

    pub(super) fn lookup_top_level_def(&self, name: &str) -> Option<(String, Expr)> {
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

    pub(super) fn eval_expr(&mut self, expr: &Expr) -> Result<RuntimeValue, String> {
        // chelis#914: cooperative cancellation. Every node visit passes
        // through here — including each element of a fold/map, which reach
        // `eval_expr` via `apply_resolved_callable_with_arg_types` — so this
        // is the single point that bounds how long a cancelled evaluation
        // keeps running. `self.cancel` was captured once at context
        // construction, so the common (no token) case is an `Option`
        // discriminant test and the cancellable case adds one relaxed load.
        if let Some(cancel) = &self.cancel
            && cancel.is_cancelled()
        {
            return Err(chelis_types::EVAL_CANCELLED_MSG.to_string());
        }
        match expr {
            Expr::Atom(_, _) => Err("bare atom is not a runtime expression".to_string()),
            Expr::Map(_, _) => Ok(RuntimeValue::Unit),
            Expr::MetaExpr(meta, _) => self.eval_expr(&meta.expr),
            Expr::List(list, _) => self.eval_list(list),
        }
    }

    fn eval_list(&mut self, list: &List) -> Result<RuntimeValue, String> {
        match tag(list) {
            Some(DeepTag::Lit) => self.eval_lit(list),
            Some(DeepTag::Var) => self.eval_var(list),
            Some(DeepTag::App) => self.eval_app(list),
            Some(DeepTag::If) => self.eval_if(list),
            Some(DeepTag::Let) => self.eval_let(list),
            Some(DeepTag::Tuple) => Ok(RuntimeValue::Tuple(
                children(list)
                    .iter()
                    .map(|child| self.eval_expr(child))
                    .collect::<Result<Vec<_>, _>>()?,
            )),
            Some(DeepTag::Copy) => {
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
            Some(DeepTag::Borrow) => {
                // The IR lower path treats `borrow` as identity
                // (chelis-ir/src/lower.rs::lower_identity); mirror that
                // here so `&t` syntax type-checks AND evaluates.
                self.eval_expr(
                    children(list)
                        .first()
                        .ok_or_else(|| "borrow missing value".to_string())?,
                )
            }
            Some(DeepTag::Block) => {
                // chelis#859: sequenced expressions, value is the last
                // child's (spec/03 §2.3). Non-last children evaluate for
                // their effects (e.g. `print` transcript lines).
                let kids = children(list);
                let Some((last, init)) = kids.split_last() else {
                    return Err("a `block` node has no children".to_string());
                };
                for child in init {
                    let _ = self.eval_expr(child)?;
                }
                self.eval_expr(last)
            }
            Some(DeepTag::Record) => self.eval_record(list),
            Some(DeepTag::Access) => self.eval_access(list),
            Some(DeepTag::TupleGet) => self.eval_tuple_get(list),
            Some(DeepTag::Match) => self.eval_match(list),
            Some(DeepTag::Fn) => self.eval_fn(list),
            Some(DeepTag::Pipe) => self.eval_pipe(list),
            Some(DeepTag::Cast) => self.eval_cast(list),
            Some(DeepTag::Realize) => {
                // Bucket 1: `realize` is identity in the host runtime,
                // matching the C-backend `lower_realize` pass-through
                // (`crates/chelis-ir/src/host.rs::lower_host_expr`).
                self.eval_expr(
                    children(list)
                        .first()
                        .ok_or_else(|| "realize missing value".to_string())?,
                )
            }
            Some(DeepTag::Grad) => {
                // Bucket 1: capture the `(grad ...)` form so it can be
                // applied later. The application path
                // (`apply_resolved_callable` for a `Transform`) routes
                // through `lower_subexpr_program` + the forward DAG
                // evaluator — the same machinery that `chelis build
                // --target c` uses.
                Ok(RuntimeValue::Transform {
                    kind: TransformKind::Grad,
                    transform_expr: Expr::List(list.clone(), Span::new(0, 0)),
                    captured_env: self.bindings.clone(),
                })
            }
            Some(DeepTag::Vmap) => {
                // Bucket 1: same pattern as `grad` above, capture-and-apply.
                Ok(RuntimeValue::Transform {
                    kind: TransformKind::Vmap,
                    transform_expr: Expr::List(list.clone(), Span::new(0, 0)),
                    captured_env: self.bindings.clone(),
                })
            }
            Some(DeepTag::Jit) => {
                // `spec/03-deep-syntax.md` §2.7: `jit` is a compilation
                // trigger and a semantic no-op at evaluation. The host
                // runtime evaluates the inner expression and returns its
                // value, mirroring `lower_jit` in
                // `crates/chelis-ir/src/lower.rs` and the IR DAG behavior.
                self.eval_expr(
                    children(list)
                        .first()
                        .ok_or_else(|| "jit missing value".to_string())?,
                )
            }
            Some(DeepTag::Par) => {
                // `spec/03-deep-syntax.md` §2.3: `par` v1 is sequential
                // composition; evaluate each child in order and return the
                // value of the last child. Mirrors `lower_par` in
                // `crates/chelis-ir/src/lower.rs`. Intermediate children
                // are evaluated for their side effects (any
                // `handle-effect` / `realize` / IO primitive in a child
                // routes through its own host arm). If `par` has zero
                // children, the spec doesn't define a v1 value; we return
                // an error rather than synthesizing a zero default, since
                // the parser/check layers should not have admitted an
                // empty par body.
                let kids = children(list);
                let mut last: Option<RuntimeValue> = None;
                for child in kids {
                    last = Some(self.eval_expr(child)?);
                }
                last.ok_or_else(|| "par has no children to evaluate".to_string())
            }
            Some(DeepTag::HandleEffect) => {
                let kids = children(list);
                match decode_effect_kind(list)
                    .map_err(|error| format!("{error} in `handle-effect` evaluation"))?
                {
                    EffectKind::Random => {
                        let seed_expr = kids
                            .first()
                            .ok_or_else(|| "handle-effect missing seed".to_string())?;
                        // chelis#771: read a *literal* seed at full i64 width so
                        // the evaluator derives the same u64 seed as the compiled
                        // C lane. `literal_seed_i64` peels `(lit …)` to the raw
                        // `Atom::Int`, mirroring host lowering (host.rs reads the
                        // raw atom and ignores the int32 default meta). Routing the
                        // literal through `eval_expr` -> `eval_lit` instead narrows
                        // it to int32 (spec/04-type-system.md §5.3 default),
                        // truncating then sign-extending any seed >= 2^31 into an
                        // unrelated stream. The seed is designed int64
                        // (spec/design/checker_totality.md §C1.5 item 5).
                        //
                        // The `eval_expr` fallback below is defensive/forward-looking,
                        // not a live narrowing path: computed (non-literal) seeds are
                        // currently gate-REJECTED in both lanes by chelis-effects'
                        // `validate_handler_expr` (a `random` handler whose seed is
                        // not an int literal errors "with seed(...) currently requires
                        // an int literal seed" in check, eval, AND build). That gate's
                        // accept set is exactly this peel's accept set, so every seed
                        // that reaches here is a literal read at full width and the
                        // fallback never runs on a checked path. It would only narrow
                        // if #731 Phase 1 relaxes the gate to admit computed seeds.
                        let seed = match literal_seed_i64(seed_expr) {
                            Some(value) => value as u64,
                            None => {
                                let seed = self.eval_expr(seed_expr)?;
                                match seed.as_i64() {
                                    Some(value) => value as u64,
                                    None => {
                                        return Err(format!(
                                            "with seed expects int seed, got {seed:?}"
                                        ));
                                    }
                                }
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
                    }
                    EffectKind::Resource => self.eval_expr(
                        kids.get(1)
                            .ok_or_else(|| "handle-effect missing body".to_string())?,
                    ),
                }
            }
            other => Err(format!(
                "host runtime does not support `{}`",
                other.map(DeepTag::as_str).unwrap_or("?")
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
            if tag(field_list) != Some(DeepTag::Kv) {
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
        for field_name in &declared {
            let value = fields_by_name.remove(field_name).ok_or_else(|| {
                format!("record `{ctor}` missing field `{field_name}` at runtime")
            })?;
            ordered.push(value);
        }
        if let Some(extra) = fields_by_name.keys().next() {
            return Err(format!(
                "record `{ctor}` has unknown field `{extra}` at runtime"
            ));
        }
        // `field_names` must stay aligned with `fields`: both follow the
        // DECLARED field order used for the reordering above. The pre-#520
        // code stored the kv SOURCE order here (the desugarer sorts record
        // kvs alphabetically), so any record whose alphabetical order
        // differs from its declared order carried misaligned
        // `field_names[i]` metadata against `fields[i]`.
        Ok(RuntimeValue::Adt {
            ctor: ctor.to_string(),
            fields: ordered,
            field_names: Some(declared),
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
        // Per spec/04-type-system.md §5.3, the desugarer narrows
        // unsuffixed integer literals to int32 and unsuffixed float
        // literals to f32. The type checker writes the resolved
        // primitive into the lit's `type` meta as `(t-prim {} <name>)`.
        // Honor that meta where present so a context-typed literal
        // (e.g. `(lit {type: (t-prim {} int64)} 42)`) carries the
        // surrounding-position dtype, not just the bare default.
        let meta_dtype = get_meta(list).and_then(lit_meta_prim);
        match value {
            Expr::Atom(Atom::Int(value), _) => match meta_dtype {
                Some(dtype) if dtype.is_integer() => RuntimeValue::scalar_like_int(dtype, *value),
                Some(dtype) if dtype.is_float() => {
                    RuntimeValue::scalar_like_float(dtype, *value as f64)
                }
                _ => Ok(RuntimeValue::int_lit(*value)),
            },
            Expr::Atom(Atom::Float(value), _) => match meta_dtype {
                Some(dtype) if dtype.is_float() => RuntimeValue::scalar_like_float(dtype, *value),
                _ => Ok(RuntimeValue::float_lit(*value)),
            },
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
        if tag(params_list) != Some(DeepTag::Params) {
            return Err("fn params malformed".to_string());
        }
        let params = children(params_list)
            .iter()
            .filter_map(runtime_param_name)
            .map(str::to_string)
            .collect::<Vec<_>>();
        let param_types = children(params_list)
            .iter()
            .filter(|param| runtime_param_name(param).is_some())
            .map(|param| param_decl_type_expr(param).cloned())
            .collect::<Vec<_>>();
        let body = kids
            .get(1)
            .ok_or_else(|| "fn missing body".to_string())?
            .clone();
        Ok(RuntimeValue::Closure {
            params,
            param_types,
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

        // chelis#338 site A: a reduction whose axis argument is a bare
        // `(var name)` names a *dimension* of the operand, not a runtime
        // value (spec/04-type-system.md SS4.5.3); the checker admits only
        // int-literal or named axes, so a bare var here is always a named
        // axis. It must be intercepted BEFORE generic argument evaluation
        // (which would fail with `unknown runtime name`) and routed
        // through IR lowering, where name -> index resolution lives.
        if let Some(reduce_name) = builtin_name(func)
            && REDUCTION_BUILTIN_NAMES.contains(&reduce_name)
            && kids.len() >= 3
            && kids[2..].iter().any(|axis| var_name(axis).is_some())
        {
            return self.eval_named_axis_reduction_app(reduce_name, kids);
        }

        // chelis#339 site A twin: a named-axis EXPAND app — the axis slot
        // is a bare `(var name)` naming the *inserted* axis (and the
        // optional fourth arg names the anchor). Same interception, same
        // routing lane: IR lowering resolves the insertion point against
        // the operand's named dims. The positional form (integer axis,
        // possibly with a symbolic size) keeps the host path.
        if let Some(expand_name) = builtin_name(func)
            && expand_name == "expand"
            && kids.len() >= 4
            && var_name(&kids[2]).is_some()
        {
            return self.eval_named_axis_reduction_app(expand_name, kids);
        }

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

        // chelis#338 site B: a call to a top-level def whose body needs
        // named-axis routing (it reduces a named axis directly, or calls
        // a rank-polymorphic def that does). Route the call through IR
        // lowering at this boundary, where the callee's declared (named)
        // param types are available, exactly as `chelis build` calls a
        // signature-typed compiled function. Falls through to ordinary
        // interpretation when no routing strategy applies; the body's own
        // reduction then hits site A with frame-typed bindings.
        if let Some(callee) = var_name(func)
            && !self.bindings.contains_key(callee)
            && !self.tensor_bindings.contains_key(callee)
            && let Some((resolved, def_expr)) = self.lookup_top_level_def(callee)
            && matches!(&def_expr, Expr::List(def_list, _) if tag(def_list) == Some(DeepTag::Fn))
            && self.def_requires_named_axis_routing(&resolved)
            && let Some(routed) = self.try_named_axis_def_call(&resolved, &def_expr, kids, &args)?
        {
            return Ok(routed);
        }

        let arg_type_exprs = kids[1..]
            .iter()
            .map(|arg| self.static_type_expr_of(arg))
            .collect::<Vec<_>>();
        // chelis#721: when the callee names a `(fn …)`-bodied top-level def and
        // is NOT a local binding, resolve it directly to its Closure. A nullary
        // (or otherwise DAG-lowerable) def folds to a constant that lands in
        // `tensor_bindings`; `eval_var`'s precedence returns that Tensor BEFORE
        // `resolve_top_level` (eval.rs eval_var), so `eval_expr(func)` here would
        // hand back the folded Tensor and `apply_*` would reject it as "value is
        // not callable". Going through `resolve_top_level` bypasses only the
        // tensor_bindings shadow — a local binding (checked here) still wins, and
        // a bare non-applied `(var f)` keeps today's eval_var behavior.
        let callable = if let Some(callee) = var_name(func)
            && !self.bindings.contains_key(callee)
            && let Some((resolved, def_expr)) = self.lookup_top_level_def(callee)
            && matches!(&def_expr, Expr::List(def_list, _) if tag(def_list) == Some(DeepTag::Fn))
        {
            self.resolve_top_level(&resolved)?
        } else {
            self.eval_expr(func)?
        };
        self.apply_resolved_callable_with_arg_types(callable, args, &arg_type_exprs)
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
        if tag(bind_list) != Some(DeepTag::Bind) {
            return Err("let bindings malformed".to_string());
        }
        let saved = self.bindings.clone();
        let saved_types = self.binding_types.clone();
        // Restore both frame maps on every exit path (including bind or
        // body evaluation errors) so a caught-and-continued error can
        // never leak partial binds or stale binding types.
        let result = (|| {
            let bind_kids = children(bind_list);
            let mut index = 0;
            while index + 1 < bind_kids.len() {
                let name = symbol_name(&bind_kids[index])
                    .ok_or_else(|| "let binding must bind a name".to_string())?;
                // Record the bound expr's static type (checker-annotated
                // `{type: ...}` on the value expr, or the source binding's
                // known type for a bare var) so chelis#338 named-axis routing
                // can recover named dims for let-bound tensors. The explicit
                // `None` insert masks any same-named top-level type.
                let static_ty = self.static_type_expr_of(&bind_kids[index + 1]);
                let value = self.eval_expr(&bind_kids[index + 1])?;
                self.binding_types.insert(name.to_string(), static_ty);
                self.bindings.insert(name.to_string(), value);
                index += 2;
            }
            self.eval_expr(kids.get(1).ok_or_else(|| "let missing body".to_string())?)
        })();
        self.bindings = saved;
        self.binding_types = saved_types;
        result
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
            if tag(arm_list) != Some(DeepTag::Arm) {
                continue;
            }
            let arm_kids = children(arm_list);
            if arm_kids.len() < 3 {
                continue;
            }
            let saved = self.bindings.clone();
            let saved_types = self.binding_types.clone();
            // Pattern-bound names carry no declared types; insert `None`
            // markers afterwards so they shadow rather than leak an outer
            // same-named binding's type into the arm body (chelis#338).
            if pattern_matches(
                &scrutinee,
                &arm_kids[0],
                &mut self.bindings,
                &self.adt_fields,
            )? {
                for name in self.bindings.keys() {
                    if !saved.contains_key(name) {
                        self.binding_types.insert(name.clone(), None);
                    }
                }
                let value = self.eval_expr(&arm_kids[2]);
                self.bindings = saved;
                self.binding_types = saved_types;
                return value;
            }
            self.bindings = saved;
            self.binding_types = saved_types;
        }
        Err("non-exhaustive runtime match".to_string())
    }

    fn eval_pipe(&mut self, list: &List) -> Result<RuntimeValue, String> {
        let kids = children(list);
        let head = kids
            .first()
            .ok_or_else(|| "pipe missing head".to_string())?;
        // Thread the piped value's static type into each stage's
        // synthesized `__chelis_pipe` param (which carries no annotation
        // of its own; the annotated AST leaves pipe lambdas as
        // unresolved type vars) so a named-axis reduction in any stage
        // can recover the operand's named dims (chelis#338). The type
        // starts from the head expression and propagates through
        // shape-preserving (Identity) builtin stages and def stages
        // with concrete declared return types; any other stage drops it.
        let mut value_ty = self.static_type_expr_of(head);
        let mut value = self.eval_expr(head)?;
        for stage in kids.iter().skip(1) {
            let next_ty = self.pipe_stage_output_type(stage, value_ty.as_ref());
            value = self.apply_callable(stage, vec![value], &[value_ty])?;
            value_ty = next_ty;
        }
        Ok(value)
    }

    /// Static output type of a pipe stage, for threading the piped
    /// value's type across stages (chelis#338): the stage body's own
    /// concrete checker annotation when present, else the incoming
    /// type when the stage applies a shape-preserving (Identity-class)
    /// builtin, else a called def's declared return type. Anything
    /// else is unknown and drops the thread.
    fn pipe_stage_output_type(&self, stage: &Expr, input_ty: Option<&Expr>) -> Option<Expr> {
        // A no-extra-arg stage stays a bare `(var f)`; a stage with
        // bound args is synthesized as
        // `(fn {..} (params {} __chelis_pipe) (app {..} (var f) args...))`.
        if let Some(callee) = var_name(stage) {
            return self.callee_output_type(callee, input_ty);
        }
        let Expr::List(stage_list, _) = stage else {
            return None;
        };
        if tag(stage_list) != Some(DeepTag::Fn) {
            return None;
        }
        let body = children(stage_list).get(1)?;
        // The body's own checker annotation wins when it is concrete
        // (pipe lambdas are typically left as unresolved `t-var`s).
        if let Some(ty) = self.static_type_expr_of(body)
            && !matches!(&ty, Expr::List(ty_list, _) if tag(ty_list) == Some(DeepTag::TVar))
        {
            return Some(ty);
        }
        let Expr::List(body_list, _) = body else {
            return None;
        };
        if tag(body_list) != Some(DeepTag::App) {
            return None;
        }
        let callee = children(body_list).first().and_then(var_name)?;
        self.callee_output_type(callee, input_ty)
    }

    /// Output type of applying `callee` to a value of type `input_ty`:
    /// the input type for shape-preserving (Identity-class) builtins,
    /// or a top-level def's declared return type.
    fn callee_output_type(&self, callee: &str, input_ty: Option<&Expr>) -> Option<Expr> {
        if chelis_types::shape_class(callee) == chelis_types::ShapeClass::Identity {
            return input_ty.cloned();
        }
        let (resolved, _) = self.lookup_top_level_def(callee)?;
        let sig = self.type_env.get(&resolved)?;
        let Expr::List(sig_list, _) = sig else {
            return None;
        };
        if tag(sig_list) != Some(DeepTag::TFn) {
            return None;
        }
        children(sig_list).last().cloned()
    }

    fn apply_callable(
        &mut self,
        stage: &Expr,
        args: Vec<RuntimeValue>,
        arg_type_exprs: &[Option<Expr>],
    ) -> Result<RuntimeValue, String> {
        if let Some(name) = builtin_name(stage) {
            return self.eval_builtin(name, &args);
        }
        match self.eval_expr(stage)? {
            value @ (RuntimeValue::Closure { .. } | RuntimeValue::Transform { .. }) => {
                self.apply_resolved_callable_with_arg_types(value, args, arg_type_exprs)
            }
            other => Err(format!("pipe stage is not callable: {other:?}")),
        }
    }

    pub(super) fn apply_resolved_callable(
        &mut self,
        callable: RuntimeValue,
        args: Vec<RuntimeValue>,
    ) -> Result<RuntimeValue, String> {
        self.apply_resolved_callable_with_arg_types(callable, args, &[])
    }

    /// Like [`Self::apply_resolved_callable`], but additionally records a
    /// static Deep type expression per argument into the callee frame's
    /// `binding_types` (chelis#338 named-axis routing). The closure's own
    /// declared param type wins; `arg_type_exprs` fills the gap for
    /// synthesized params with no annotation (e.g. `__chelis_pipe`).
    fn apply_resolved_callable_with_arg_types(
        &mut self,
        callable: RuntimeValue,
        args: Vec<RuntimeValue>,
        arg_type_exprs: &[Option<Expr>],
    ) -> Result<RuntimeValue, String> {
        match callable {
            RuntimeValue::Closure {
                params,
                param_types,
                body,
                env,
            } => {
                if params.len() != args.len() {
                    return Err(format!(
                        "closure expected {} args, got {}",
                        params.len(),
                        args.len()
                    ));
                }
                let saved = self.bindings.clone();
                let saved_types = std::mem::take(&mut self.binding_types);
                self.bindings = env;
                for (index, (param, arg)) in params.into_iter().zip(args).enumerate() {
                    let declared = param_types
                        .get(index)
                        .cloned()
                        .flatten()
                        .or_else(|| arg_type_exprs.get(index).cloned().flatten());
                    self.binding_types.insert(param.clone(), declared);
                    self.bindings.insert(param, arg);
                }
                let value = self.eval_expr(&body);
                self.bindings = saved;
                self.binding_types = saved_types;
                value
            }
            RuntimeValue::Transform {
                kind,
                transform_expr,
                captured_env,
            } => self.apply_transform(kind, &transform_expr, captured_env, args),
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
        // Resolve the textual target into a Prim using the canonical
        // active dtype map. The type checker has already rejected
        // f8e4m3 (spec/04-type-system.md §1.1.1) at this point so the
        // host eval lane just needs to pick the right re-pack.
        let target_prim = prim_from_name(target)
            .ok_or_else(|| format!("cast target `{target}` is not a recognized primitive type"))?;
        match (value, target_prim) {
            (RuntimeValue::Bool(value), Prim::Bool) => Ok(RuntimeValue::Bool(value)),
            (RuntimeValue::String(value), Prim::String) => Ok(RuntimeValue::String(value)),
            (RuntimeValue::Scalar(payload), dst_dtype) if dst_dtype.is_integer() => {
                RuntimeValue::scalar_like_int(dst_dtype, payload.bits().as_i64())
            }
            (RuntimeValue::Scalar(payload), dst_dtype) if dst_dtype.is_float() => {
                RuntimeValue::scalar_like_float(dst_dtype, payload.bits().as_f64())
            }
            (RuntimeValue::Bool(value), dst_dtype) if dst_dtype.is_integer() => {
                RuntimeValue::scalar_like_int(dst_dtype, if value { 1 } else { 0 })
            }
            (RuntimeValue::Bool(value), dst_dtype) if dst_dtype.is_float() => {
                RuntimeValue::scalar_like_float(dst_dtype, if value { 1.0 } else { 0.0 })
            }
            (RuntimeValue::Tensor(tensor), _) => cast_tensor_value(tensor, target),
            (other, _) => Err(format!(
                "unsupported cast from {other:?} to {}",
                target_prim.name()
            )),
        }
    }

    fn eval_builtin(&mut self, name: &str, args: &[RuntimeValue]) -> Result<RuntimeValue, String> {
        match name {
            "add" => numeric_binop(args, |lhs, rhs| lhs + rhs),
            "sub" => numeric_binop(args, |lhs, rhs| lhs - rhs),
            "mul" => numeric_binop(args, |lhs, rhs| lhs * rhs),
            // #387: integer `div`/`mod` trap on a zero divisor with one
            // shared diagnostic instead of returning a silently-wrong value
            // (`f64` div round-trip yielded `i64::MAX`/`-1`); float `div`
            // keeps IEEE-754 (`1.0 / 0.0 == inf`). The C backend follows the
            // platform SIGFPE for the same integer operands.
            "div" => eval_div(args),
            // chelis#178: integer-division primitives. `floor_div` rounds
            // the quotient toward -inf (ints and floats); `trunc_div`
            // rounds toward zero (integer-only). Both trap on an integer
            // zero divisor with the shared diagnostic.
            "floor_div" => eval_floor_div(args),
            "trunc_div" => eval_trunc_div(args),
            // Tier-1 `max_elem` and Tier-2 `min_elem` are element-wise
            // binary ops. The IR evaluator emits
            // `binary_map(.., f64::max)` for `RiscOp::MaxElem` and
            // `lower_min_elem` (`crates/chelis-ir/src/tier2.rs:364`)
            // synthesizes `neg(max_elem(neg a, neg b))`; the host-runtime
            // closure form fuses that into a direct `f64::min` for the
            // same observable result. Wired for issue
            // Chelis-Lang/chelis#185.
            "max_elem" => numeric_binop(args, f64::max),
            "min_elem" => numeric_binop(args, f64::min),
            "mod" => eval_mod(args),
            "neg" => numeric_unop(args, |value| -value),
            "recip" => numeric_unop(args, |value| 1.0 / value),
            "exp" => float_unop_with_tensor(args, f64::exp, f32::exp),
            "log" => float_unop_with_tensor(args, f64::ln, f32::ln),
            "sin" => float_unop_with_tensor(args, f64::sin, f32::sin),
            "sqrt" => float_unop_with_tensor(args, f64::sqrt, f32::sqrt),
            // Tier 1 unary primitives wired for issue Chelis-Lang/chelis#185.
            // Each delegates to the same `float_unop_with_tensor` /
            // `numeric_unop` helper used by the already-wired siblings; the
            // tensor lane runs through `f32` to mirror the C backend's libm
            // emit (`cosf`/`tanf`/`floorf`/`ceilf`/`atanf`), which is the
            // canonical-evaluator equivalent (per
            // `feedback_evaluator_byte_identical_gate`).
            "cos" => float_unop_with_tensor(args, f64::cos, f32::cos),
            "tan" => float_unop_with_tensor(args, f64::tan, f32::tan),
            "atan" => float_unop_with_tensor(args, f64::atan, f32::atan),
            "floor" => float_unop_with_tensor(args, f64::floor, f32::floor),
            "ceil" => float_unop_with_tensor(args, f64::ceil, f32::ceil),
            // Round-half-to-even (banker's rounding), matching the DAG
            // evaluator and the C backend's `rintf`. NOT `round`, which
            // is ties-away-from-zero.
            "round" => float_unop_with_tensor(args, f64::round_ties_even, f32::round_ties_even),
            // `abs` accepts ints and floats and is sign-flipping for both;
            // route through `numeric_unop` so scalar Int64/Int32/F32/F64
            // inputs all keep their dtype.
            "abs" => numeric_unop(args, f64::abs),
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
            // Logical ops dispatch on the actual argument shape: scalar
            // bool args (already wired) keep the `bool_binop` /
            // `bool_unop` path; tensor-bool args route through the
            // dedicated `tensor_bool_*` helpers wired for issue
            // Chelis-Lang/chelis#185. Per the brief's pinned decision,
            // the tensor lane is NOT a transparent extension of the
            // scalar lane — it pins input precision to `Bool` and
            // requires matching shapes, which scalar broadcasting
            // would hide.
            "and" => match (args.first(), args.get(1)) {
                (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
                    tensor_bool_binop(lhs, rhs, |a, b| a && b).map(RuntimeValue::Tensor)
                }
                _ => bool_binop(args, |lhs, rhs| lhs && rhs),
            },
            "or" => match (args.first(), args.get(1)) {
                (Some(RuntimeValue::Tensor(lhs)), Some(RuntimeValue::Tensor(rhs))) => {
                    tensor_bool_binop(lhs, rhs, |a, b| a || b).map(RuntimeValue::Tensor)
                }
                _ => bool_binop(args, |lhs, rhs| lhs || rhs),
            },
            "not" => match args.first() {
                Some(RuntimeValue::Tensor(tensor)) => {
                    tensor_bool_unop(tensor, |value| !value).map(RuntimeValue::Tensor)
                }
                _ => bool_unop(args, |value| !value),
            },
            "bitand" => int_binop(args, |lhs, rhs| lhs & rhs),
            "bitor" => int_binop(args, |lhs, rhs| lhs | rhs),
            "bitxor" => int_binop(args, |lhs, rhs| lhs ^ rhs),
            "shl" => int_shift_binop(args, IntShiftOp::Left),
            "shr" => int_shift_binop(args, IntShiftOp::Right),
            "string_len" => {
                let value = expect_string_arg(args, 0)?;
                Ok(RuntimeValue::int64(value.chars().count() as i64))
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
                        fields: vec![RuntimeValue::int64(parsed)],
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
                        fields: vec![RuntimeValue::float64(parsed)],
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
                Some(RuntimeValue::List(list)) => Ok(RuntimeValue::int64(list.len() as i64)),
                Some(RuntimeValue::Dict(entries)) => Ok(RuntimeValue::int64(entries.len() as i64)),
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
            "concat" => match (args.first(), args.get(1).and_then(RuntimeValue::as_i64)) {
                (Some(RuntimeValue::List(parts)), Some(axis))
                    if parts
                        .iter()
                        .all(|item| matches!(item, RuntimeValue::Tensor(_))) =>
                {
                    tensor_concat_value(parts, axis)
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
            "drop" => match args.len() {
                1 => Ok(RuntimeValue::Unit),
                2 => {
                    let list = expect_list_arg(args, 0)?;
                    let count = expect_int_arg(args, 1)?;
                    if count < 0 {
                        return Err(format!("drop requires non-negative count, got {count}"));
                    }
                    Ok(RuntimeValue::List(
                        list.into_iter().skip(count as usize).collect(),
                    ))
                }
                n => Err(format!("drop expects 1 or 2 arguments, got {n}")),
            },
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
                    (start..end).map(RuntimeValue::int64).collect(),
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
            // Issue #257: iterative scan that produces a rank-1 tensor
            // directly, bypassing the right-recursive Surf list build that
            // overflows the host worker stack at ~10k elements. The arg
            // shape is `(initial: T, fn: (T, int64) -> T, n: int64)` and
            // the loop runs `n` times on the host with no Surf-level
            // recursion. The output precision is taken from the initial
            // value's scalar dtype.
            "tensor_scan" => {
                if args.len() != 3 {
                    return Err(format!(
                        "tensor_scan expects 3 arguments (initial, fn, n), got {}",
                        args.len()
                    ));
                }
                let initial = args[0].clone();
                let callback = args[1].clone();
                let n = expect_int_arg(args, 2)?;
                if n < 0 {
                    return Err(format!(
                        "tensor_scan requires a non-negative length, got {n}"
                    ));
                }
                let precision = match &initial {
                    RuntimeValue::Scalar(payload) => payload.dtype(),
                    RuntimeValue::Bool(_) => Prim::Bool,
                    other => {
                        return Err(format!(
                            "tensor_scan expects a scalar initial value (numeric or bool), got {other:?}"
                        ));
                    }
                };
                // Reject non-callable callback up front so the error message
                // points at the second argument instead of failing inside the
                // first apply.
                if !matches!(
                    &callback,
                    RuntimeValue::Closure { .. } | RuntimeValue::Transform { .. }
                ) {
                    return Err(format!(
                        "tensor_scan expects a callable second argument, got {callback:?}"
                    ));
                }
                let n = n as usize;
                let mut data = Vec::with_capacity(n);
                let mut acc = initial;
                for i in 0..n {
                    let index = RuntimeValue::int64(i as i64);
                    acc = self.apply_resolved_callable(callback.clone(), vec![acc, index])?;
                    // Validate per-step that the accumulator stayed the same
                    // scalar precision; this catches a misbehaving callback
                    // that returns a different dtype before it corrupts the
                    // output tensor buffer.
                    let value = match &acc {
                        RuntimeValue::Scalar(payload) => {
                            if payload.dtype() != precision {
                                return Err(format!(
                                    "tensor_scan callback returned a {} scalar but the initial \
                                     value's dtype is {}",
                                    payload.dtype().name(),
                                    precision.name()
                                ));
                            }
                            payload.bits().as_f64()
                        }
                        RuntimeValue::Bool(b) => {
                            if precision != Prim::Bool {
                                return Err(format!(
                                    "tensor_scan callback returned a bool but the initial \
                                     value's dtype is {}",
                                    precision.name()
                                ));
                            }
                            if *b { 1.0 } else { 0.0 }
                        }
                        other => {
                            return Err(format!(
                                "tensor_scan callback must return a scalar, got {other:?}"
                            ));
                        }
                    };
                    data.push(value);
                }
                Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::from_vec(vec![n], data),
                    precision,
                }))
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
                            RuntimeValue::Tuple(vec![RuntimeValue::int64(index as i64), value])
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
                // Bucket 4b: support nested numeric/bool lists. The outer
                // list contributes the leading dim; if its elements are
                // themselves uniformly-shaped numeric/bool lists, those
                // contribute additional inner dims (and so on
                // recursively).
                let (precision, shape, data) = nested_list_to_tensor_data(&values)?;
                Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::from_vec(shape, data),
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
            // Hull Phase 0a: `process_run(cmd, args) -> (exit_code, stdout, stderr)`.
            //
            // Eval/test-only subprocess exec. Arguments are passed straight to
            // the OS as argv via `Command::args` -- there is no shell, no glob
            // expansion, and no `$VAR`/backtick interpolation, so a hostile
            // `cmd` or `args` value cannot inject extra shell commands. The C
            // and HIP build backends deliberately reject this builtin (see
            // `reject_eval_only_builtins_host`) rather than emit a silent `0`.
            "process_run" => {
                let cmd = expect_string_arg(args, 0)?;
                let raw_args = expect_list_arg(args, 1)?;
                let mut argv = Vec::with_capacity(raw_args.len());
                for (index, value) in raw_args.iter().enumerate() {
                    match value {
                        RuntimeValue::String(text) => argv.push(text.clone()),
                        other => {
                            return Err(format!(
                                "process_run expects List[String] args, got {other:?} at index {index}"
                            ));
                        }
                    }
                }
                let output = std::process::Command::new(&cmd)
                    .args(&argv)
                    .output()
                    .map_err(|err| format!("process_run failed to spawn `{cmd}`: {err}"))?;
                // A process killed by a signal has no exit code; report -1 so
                // callers can distinguish it from a clean exit 0.
                let exit_code = output.status.code().map_or(-1_i64, i64::from);
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                Ok(RuntimeValue::Tuple(vec![
                    RuntimeValue::int64(exit_code),
                    RuntimeValue::String(stdout),
                    RuntimeValue::String(stderr),
                ]))
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
                        .map(|byte| RuntimeValue::int64(i64::from(byte)))
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
                        .map(|byte| RuntimeValue::int64(i64::from(*byte)))
                        .collect(),
                ))
            }
            "mmap_len" => match args.first() {
                Some(RuntimeValue::MappedFile(bytes)) => {
                    Ok(RuntimeValue::int64(bytes.len() as i64))
                }
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
            "scatter_elements" => {
                let data = expect_tensor_arg(args, 0)?;
                let indices = expect_tensor_arg(args, 1)?;
                let updates = expect_tensor_arg(args, 2)?;
                let axis = expect_int_arg(args, 3)?;
                tensor_scatter_elements_value(&data, &indices, &updates, axis)
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
                Ok(RuntimeValue::int64(tensor.value.shape.len() as i64))
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
                Ok(RuntimeValue::int64(dim as i64))
            }
            "numel" => {
                let tensor = expect_tensor_arg(args, 0)?;
                // Empty-product identity handles the scalar (shape `[]`) case
                // correctly: `[].iter().product::<usize>() == 1`. For rank-1+
                // tensors with any zero dimension the product is 0, which is
                // the genuine element count and must not be clamped. The
                // historical `.max(1)` clamp here was the root cause of
                // Runtime-EmptyTensorNumel-F1: `numel(to_tensor([]))`
                // returning 1 instead of 0.
                let numel = tensor.value.shape.iter().product::<usize>();
                Ok(RuntimeValue::int64(numel as i64))
            }
            "tensor_to_scalar" => {
                let tensor = expect_tensor_arg(args, 0)?;
                if !tensor.value.shape.is_empty() {
                    return Err("tensor_to_scalar expects a rank-0 tensor".to_string());
                }
                let value = tensor.value.data.first().copied().unwrap_or(0.0);
                match tensor.precision {
                    Prim::Bool => Ok(RuntimeValue::Bool(value != 0.0)),
                    p if p.is_integer() => RuntimeValue::scalar_like_int(p, value as i64),
                    p if p.is_float() => RuntimeValue::scalar_like_float(p, value),
                    other => Err(format!(
                        "tensor_to_scalar: unsupported tensor element dtype `{}`",
                        other.name()
                    )),
                }
            }
            "scalar_to_tensor" => match args.first() {
                Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_integer() => {
                    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                        value: IrTensorValue::scalar(payload.bits().as_i64() as f64),
                        precision: payload.dtype(),
                    }))
                }
                Some(RuntimeValue::Scalar(payload)) if payload.dtype().is_float() => {
                    Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                        value: IrTensorValue::scalar(payload.bits().as_f64()),
                        precision: payload.dtype(),
                    }))
                }
                Some(RuntimeValue::Bool(value)) => Ok(RuntimeValue::Tensor(RuntimeTensorValue {
                    value: IrTensorValue::scalar(if *value { 1.0 } else { 0.0 }),
                    precision: Prim::Bool,
                })),
                // #381: a top-level scalar binding (e.g. `c = cast(1.1, f64)`)
                // captured by a def body is pre-evaluated through the DAG lane
                // and arrives here as an already rank-0 (0-d) tensor, not a
                // `Scalar`. `scalar_to_tensor` of a scalar produces a rank-0
                // tensor, so applying it to a rank-0 tensor is the identity;
                // accept it and pass the value through (preserving precision).
                // This matches the C backend, where the captured binding stays
                // a scalar C value and `scalar_to_tensor` materializes the same
                // rank-0 tensor, and the DAG lowering, where `scalar_to_tensor`
                // is a pass-through on its input node.
                Some(RuntimeValue::Tensor(tensor)) if tensor.value.shape.is_empty() => {
                    Ok(RuntimeValue::Tensor(tensor.clone()))
                }
                other => Err(format!(
                    "scalar_to_tensor expects a scalar or rank-0 tensor input, got {other:?}"
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
            // Mirror of `min_reduce` for issue Chelis-Lang/chelis#185. The
            // typer's `tensor_reduce_to_out` signature gives both ops the
            // same shape; the IR evaluator's `RiscOp::MaxReduce` uses
            // `reduce(.., f64::NEG_INFINITY, f64::max)` which the
            // `ReduceOp::Max` variant of `tensor_reduce_host` mirrors.
            "max_reduce" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                tensor_reduce_host(&tensor, axis, ReduceOp::Max).map(RuntimeValue::Tensor)
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
            // Composed Tier-2 ops wired for issue Chelis-Lang/chelis#185.
            // Each delegates to the canonical IR decomposition in
            // `crates/chelis-ir/src/tier2.rs` (the same path the C
            // backend takes) and forward-evaluates the resulting small
            // DAG through `chelis_ir::eval` — the canonical numerical
            // oracle per `feedback_evaluator_byte_identical_gate`. We do
            // not reimplement the math here; that's the path that drifts
            // when downstream tier2 updates land.
            "mean" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let axis = expect_int_arg(args, 1)?;
                let axis = normalize_axis(tensor.value.shape.len(), axis, "mean")?;
                eval_composed_unary(&tensor, |dag, x, ty| {
                    tier2::lower_mean(dag, x, axis, ty, None)
                })
                .map(RuntimeValue::Tensor)
            }
            "layer_norm" => {
                let x = expect_tensor_arg(args, 0)?;
                let gamma = expect_tensor_arg(args, 1)?;
                let beta = expect_tensor_arg(args, 2)?;
                eval_composed_triop(&x, &gamma, &beta, |dag, x_id, gamma_id, beta_id, tys| {
                    tier2::lower_layer_norm(
                        dag, x_id, gamma_id, beta_id, tys.0, tys.1, tys.2, 1e-5, None,
                    )
                })
                .map(RuntimeValue::Tensor)
            }
            "conv2d" => {
                let input = expect_tensor_arg(args, 0)?;
                let kernel = expect_tensor_arg(args, 1)?;
                let stride = expect_int_arg(args, 2)?;
                let padding = expect_int_arg(args, 3)?;
                if stride < 1 {
                    return Err(format!("conv2d stride must be >= 1, got {stride}"));
                }
                if padding < 0 {
                    return Err(format!("conv2d padding must be >= 0, got {padding}"));
                }
                let stride = stride as usize;
                let padding = padding as usize;
                conv2d_host(&input, &kernel, stride, padding).map(RuntimeValue::Tensor)
            }
            // Movement primitives that take parameterized window args. Both
            // delegate to the same arithmetic the IR evaluator at
            // `crates/chelis-ir/src/eval.rs` uses, so eval-in-context output
            // is byte-identical to a freshly-lowered DAG run -- per the
            // evaluator-vs-backend agreement gate. Issue Chelis-Lang/chelis#187.
            "shrink" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let raw = expect_list_arg(args, 1)?;
                let bounds = extract_bounds_pair_list(&raw, "shrink")?;
                tensor_shrink_host(&tensor, &bounds).map(RuntimeValue::Tensor)
            }
            "reduce_window_max" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let window_raw = expect_list_arg(args, 1)?;
                let strides_raw = expect_list_arg(args, 2)?;
                let window = expect_int_list(&window_raw, "reduce_window_max")?;
                let strides = expect_int_list(&strides_raw, "reduce_window_max")?;
                tensor_reduce_window_host(
                    &tensor,
                    &window,
                    &strides,
                    ReduceWindowOp::Max,
                    "reduce_window_max",
                )
                .map(RuntimeValue::Tensor)
            }
            "reduce_window_min" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let window_raw = expect_list_arg(args, 1)?;
                let strides_raw = expect_list_arg(args, 2)?;
                let window = expect_int_list(&window_raw, "reduce_window_min")?;
                let strides = expect_int_list(&strides_raw, "reduce_window_min")?;
                tensor_reduce_window_host(
                    &tensor,
                    &window,
                    &strides,
                    ReduceWindowOp::Min,
                    "reduce_window_min",
                )
                .map(RuntimeValue::Tensor)
            }
            "reduce_window_sum" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let window_raw = expect_list_arg(args, 1)?;
                let strides_raw = expect_list_arg(args, 2)?;
                let window = expect_int_list(&window_raw, "reduce_window_sum")?;
                let strides = expect_int_list(&strides_raw, "reduce_window_sum")?;
                tensor_reduce_window_host(
                    &tensor,
                    &window,
                    &strides,
                    ReduceWindowOp::Sum,
                    "reduce_window_sum",
                )
                .map(RuntimeValue::Tensor)
            }
            "reduce_window_mean" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let window_raw = expect_list_arg(args, 1)?;
                let strides_raw = expect_list_arg(args, 2)?;
                let window = expect_int_list(&window_raw, "reduce_window_mean")?;
                let strides = expect_int_list(&strides_raw, "reduce_window_mean")?;
                tensor_reduce_window_host(
                    &tensor,
                    &window,
                    &strides,
                    ReduceWindowOp::Mean,
                    "reduce_window_mean",
                )
                .map(RuntimeValue::Tensor)
            }
            "pad" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let raw = expect_list_arg(args, 1)?;
                let padding = extract_bounds_pair_list(&raw, "pad")?;
                let fill = expect_float_arg(args, 2)?;
                tensor_pad_host(&tensor, &padding, fill).map(RuntimeValue::Tensor)
            }
            "stride" => {
                let tensor = expect_tensor_arg(args, 0)?;
                let strides = expect_int_list(&args[1..], "stride")?;
                for (axis, step) in strides.iter().enumerate() {
                    if *step == 0 {
                        return Err(format!(
                            "stride axis {axis} step 0 is not allowed (must be positive)"
                        ));
                    }
                }
                tensor_stride_host(&tensor, &strides).map(RuntimeValue::Tensor)
            }
            // Activation primitives (Bucket 3).
            //
            // Each activation must produce values byte-identical (to documented
            // float tolerance) to the C backend's `chelis_host_*_f32` helpers
            // emitted from `crates/chelis-backend-c/src/host_emit.rs`. Those
            // helpers run all math through `float` (single precision); we
            // therefore route every transcendental through `f32` here too —
            // widening only happens at the very end when we re-store as
            // `f64`-shaped tensor data. The closures themselves accept and
            // return `f64` so `tensor_float_unop_f32` can cast at the
            // boundary, which means `(x as f32).exp() as f64` and never
            // `f64::exp(x)`.
            "relu" => {
                let tensor = expect_tensor_arg(args, 0)?;
                Ok(RuntimeValue::Tensor(tensor_float_unop_f32(
                    &tensor,
                    activation_relu_f32,
                )))
            }
            "sigmoid" => {
                let tensor = expect_tensor_arg(args, 0)?;
                Ok(RuntimeValue::Tensor(tensor_float_unop_f32(
                    &tensor,
                    activation_sigmoid_f32,
                )))
            }
            "tanh" => {
                let tensor = expect_tensor_arg(args, 0)?;
                Ok(RuntimeValue::Tensor(tensor_float_unop_f32(
                    &tensor,
                    activation_tanh_f32,
                )))
            }
            "silu" => {
                let tensor = expect_tensor_arg(args, 0)?;
                Ok(RuntimeValue::Tensor(tensor_float_unop_f32(
                    &tensor,
                    activation_silu_f32,
                )))
            }
            "gelu" => {
                let tensor = expect_tensor_arg(args, 0)?;
                Ok(RuntimeValue::Tensor(tensor_float_unop_f32(
                    &tensor,
                    activation_gelu_f32,
                )))
            }
            other => Err(format!("unsupported builtin `{other}` in host runtime")),
        }
    }
}
