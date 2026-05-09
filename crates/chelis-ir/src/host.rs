use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use chelis_deep::ast::{Atom, Expr, List};
use chelis_types::{BUILTIN_NAMES, CheckedProgram};

use crate::dag::TensorType;
use crate::lower::{lower_program, top_level_lowering_map};

thread_local! {
    // Tracks top-level callee names currently being inlined by
    // `inline_top_level_host_call`. Prevents infinite specialization for
    // recursive/mutually recursive definitions — the specialized body would
    // re-encounter the same call and inline forever.
    static INLINING_STACK: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
}

fn is_inlining(name: &str) -> bool {
    INLINING_STACK.with(|stack| stack.borrow().contains(name))
}

fn push_inlining(name: &str) -> bool {
    INLINING_STACK.with(|stack| stack.borrow_mut().insert(name.to_string()))
}

fn pop_inlining(name: &str) {
    INLINING_STACK.with(|stack| {
        stack.borrow_mut().remove(name);
    });
}

#[derive(Debug, Clone)]
pub struct CompiledProgram {
    pub dag: Option<crate::Dag>,
    pub host: Option<HostProgram>,
}

#[derive(Debug, Clone, Default)]
pub struct HostProgram {
    pub globals: Vec<HostBinding>,
    pub global_tensor_helpers: Vec<HostTensorHelper>,
    pub functions: Vec<HostFunction>,
}

#[derive(Debug, Clone)]
pub struct HostBinding {
    pub name: String,
    pub display_name: Option<String>,
    pub ty: HostType,
    pub value: HostExpr,
}

#[derive(Debug, Clone)]
pub struct HostFunction {
    pub name: String,
    pub params: Vec<HostParam>,
    pub ret_ty: HostType,
    pub body: HostExpr,
    pub tensor_helpers: Vec<HostTensorHelper>,
}

#[derive(Debug, Clone)]
pub struct HostParam {
    pub name: String,
    pub ty: HostType,
}

#[derive(Debug, Clone)]
pub struct HostTensorHelper {
    pub name: String,
    pub dag: crate::Dag,
    pub inputs: Vec<HostTensorInput>,
    pub output: TensorType,
}

#[derive(Debug, Clone)]
pub struct HostTensorInput {
    pub name: String,
    pub ty: TensorType,
}

#[derive(Debug, Clone)]
pub struct HostCallback {
    pub kind: HostCallbackKind,
    pub ret_ty: HostType,
}

#[derive(Debug, Clone)]
pub struct HostMatchArm {
    pub ctor: String,
    pub bindings: Vec<HostPatternBinding>,
    pub expr: HostExpr,
}

#[derive(Debug, Clone)]
pub struct HostPatternBinding {
    pub name: String,
    pub ty: HostType,
    pub field_index: usize,
}

#[derive(Debug, Clone)]
pub struct HostAdtField {
    pub name: Option<String>,
    pub ty: HostType,
}

#[derive(Debug, Clone)]
pub enum HostCallbackKind {
    Named {
        function: String,
        params: Vec<HostParam>,
    },
    Inline {
        params: Vec<HostParam>,
        body: Box<HostExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum HostType {
    Int64,
    Float64,
    Bool,
    String,
    Fn(Vec<HostType>, Box<HostType>),
    Adt(String, Vec<HostType>),
    List(Box<HostType>),
    Dict(Box<HostType>, Box<HostType>),
    Tuple(Vec<HostType>),
    Tensor(TensorType),
    Option(Box<HostType>),
    MappedFile,
    Unit,
    Unknown,
}

/// Host-lane expression with span-survival metadata.
///
/// `HostExpr` is a struct wrapper around `HostExprKind` carrying the
/// canonical span ID and any spans accumulated through N→1 merges, mirroring
/// the `DagNode` schema documented in `spec/design/chelis_span_survival.md`
/// §2.2 / §2.3 (S6 host-side rule table).
///
/// **Schema is locked.** Both fields stay even though the host-side passes
/// shipped today don't all produce N→1 merges; the schema mirrors `DagNode`
/// for orchestrator-tooling uniformity (one audit consumer reads both
/// schemas) and so future host-side optimizations have the field they need.
/// Reject simplification proposals on the same grounds the spec rejects
/// collapsing `DagNode.merged_spans` back to a single `Option<String>`.
#[derive(Debug, Clone)]
pub struct HostExpr {
    pub kind: HostExprKind,
    /// Canonical span ID, populated by host-lane lowering from the Deep
    /// `Expr`'s `meta["span"]` value. Threaded through later host-side
    /// passes per the rules in `spec/design/chelis_span_survival.md` §2.3
    /// host-side table.
    ///
    /// `None` is the normal case for hand-written Chelis or for nodes
    /// synthesized in places where the source-region rule does not apply.
    pub span_id: Option<String>,
    /// Additional spans accumulated when N→1 host-side merge passes
    /// collapse multiple source nodes into a single result node.
    ///
    /// Backend host emission (S6 step 5) emits one `// span:` line per
    /// `span_id ∪ merged_spans` so the audit invariant holds: every span
    /// ID present on any input Deep node appears on at least one IR or
    /// HostExpr node.
    pub merged_spans: Vec<String>,
}

impl HostExpr {
    /// Construct a HostExpr from a kind with no span metadata. The host-side
    /// lowering layer (§2.3 host-side table, rule "Lowering") populates
    /// `span_id` from the enclosing Deep expr's `meta["span"]` via the
    /// `with_span` constructor; default constructions (e.g. tests) start
    /// span-free.
    pub fn new(kind: HostExprKind) -> Self {
        Self {
            kind,
            span_id: None,
            merged_spans: Vec::new(),
        }
    }

    /// Construct a HostExpr from a kind with an explicit span ID. Empty
    /// `merged_spans`. Used by the host-lane lowering pass (`lower_host_expr`)
    /// to attach the current Deep expr's span to every freshly-produced node.
    pub fn with_span(kind: HostExprKind, span_id: Option<String>) -> Self {
        Self {
            kind,
            span_id,
            merged_spans: Vec::new(),
        }
    }

    /// Append a single span to this node's `merged_spans`, lex-sorted and
    /// deduped, with the same no-op rules as `crate::span_merge::append_span_to_node`:
    ///   * `span` is `None` (passthrough),
    ///   * the node's `span_id` already equals `span`,
    ///   * `merged_spans` already contains `span`.
    ///
    /// Used by the host-side N→1 lowering collapse rule (§2.3 host-side
    /// table, rule "Lowering — body collapses to existing HostExpr"): when
    /// a parent Deep expr lowers to an already-constructed inner HostExpr
    /// (e.g. `(realize ...)`, `(handle-effect ... body)`, `(lit ...)` whose
    /// child is the canonical node), the parent's `span_id` appends here so
    /// the audit invariant ("every input span appears as `span_id` or in
    /// `merged_spans` on at least one node") still holds.
    pub fn append_merged_span(&mut self, span: Option<&str>) {
        let Some(span) = span else {
            return;
        };
        if self.span_id.as_deref() == Some(span) {
            return;
        }
        if self.merged_spans.iter().any(|s| s == span) {
            return;
        }
        self.merged_spans.push(span.to_owned());
        self.merged_spans.sort();
    }
}

#[derive(Debug, Clone)]
pub enum HostExprKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<HostExpr>, HostType),
    Tuple(Vec<HostExpr>, HostType),
    Var(String, HostType),
    Call {
        function: String,
        args: Vec<HostExpr>,
        arg_tys: Vec<HostType>,
        ty: HostType,
    },
    Builtin {
        name: String,
        args: Vec<HostExpr>,
        ty: HostType,
    },
    AdtConstruct {
        ctor: String,
        fields: Vec<HostExpr>,
        ty: HostType,
    },
    AdtFieldAccess {
        base: Box<HostExpr>,
        field_index: usize,
        ty: HostType,
    },
    If {
        cond: Box<HostExpr>,
        then_expr: Box<HostExpr>,
        else_expr: Box<HostExpr>,
        ty: HostType,
    },
    MatchOption {
        scrutinee: Box<HostExpr>,
        bind_name: String,
        some_expr: Box<HostExpr>,
        none_expr: Box<HostExpr>,
        ty: HostType,
    },
    MatchAdt {
        scrutinee: Box<HostExpr>,
        arms: Vec<HostMatchArm>,
        default_expr: Option<Box<HostExpr>>,
        ty: HostType,
    },
    Let {
        bindings: Vec<HostBinding>,
        body: Box<HostExpr>,
        ty: HostType,
    },
    Map {
        callback: HostCallback,
        list: Box<HostExpr>,
        ty: HostType,
    },
    Filter {
        callback: HostCallback,
        list: Box<HostExpr>,
        ty: HostType,
    },
    Fold {
        callback: HostCallback,
        init: Box<HostExpr>,
        list: Box<HostExpr>,
        ty: HostType,
    },
    Scan {
        callback: HostCallback,
        init: Box<HostExpr>,
        list: Box<HostExpr>,
        ty: HostType,
    },
    Partition {
        callback: HostCallback,
        list: Box<HostExpr>,
        ty: HostType,
    },
    FlatMap {
        callback: HostCallback,
        list: Box<HostExpr>,
        ty: HostType,
    },
    TensorCall {
        helper: usize,
        args: Vec<HostExpr>,
        ty: HostType,
    },
    Unit,
}

pub fn lower_compiled_program(program: &CheckedProgram) -> CompiledProgram {
    let lowered_names = top_level_lowering_map(program.exprs(), program.type_env());
    let dag = lower_program(program);
    let host = lower_host_program(program, &lowered_names);

    CompiledProgram {
        dag: (!dag.roots().is_empty()).then_some(dag),
        host: if host.globals.is_empty() && host.functions.is_empty() {
            None
        } else {
            Some(host)
        },
    }
}

pub fn host_program_requires_host_backend(program: &HostProgram) -> bool {
    if !program.globals.is_empty() {
        return true;
    }

    let tensor_only_functions = program
        .functions
        .iter()
        .filter(|function| {
            matches!(function.ret_ty, HostType::Tensor(_))
                && function
                    .params
                    .iter()
                    .all(|param| matches!(param.ty, HostType::Tensor(_)))
        })
        .map(|function| function.name.clone())
        .collect::<HashSet<_>>();

    program.functions.iter().any(|function| {
        !tensor_only_functions.contains(&function.name)
            || !host_expr_stays_on_tensor_path(&function.body, &tensor_only_functions)
    })
}

fn host_expr_stays_on_tensor_path(
    expr: &HostExpr,
    tensor_only_functions: &HashSet<String>,
) -> bool {
    match &expr.kind {
        HostExprKind::Var(_, HostType::Tensor(_)) => true,
        HostExprKind::TensorCall { .. } => true,
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => {
            matches!(ty, HostType::Tensor(_))
                && tensor_only_functions.contains(function)
                && arg_tys.iter().all(|ty| matches!(ty, HostType::Tensor(_)))
                && args
                    .iter()
                    .all(|arg| host_expr_stays_on_tensor_path(arg, tensor_only_functions))
        }
        HostExprKind::Let { bindings, body, ty } => {
            matches!(ty, HostType::Tensor(_))
                && bindings.iter().all(|binding| {
                    matches!(binding.ty, HostType::Tensor(_))
                        && host_expr_stays_on_tensor_path(&binding.value, tensor_only_functions)
                })
                && host_expr_stays_on_tensor_path(body, tensor_only_functions)
        }
        _ => false,
    }
}

pub fn preferred_tensor_entry_name(program: &HostProgram) -> Option<&str> {
    fn tensor_signature(function: &HostFunction) -> bool {
        matches!(function.ret_ty, HostType::Tensor(_))
            && function
                .params
                .iter()
                .all(|param| matches!(param.ty, HostType::Tensor(_)))
    }

    if let Some(function) = program
        .functions
        .iter()
        .find(|function| function.name == "main" && tensor_signature(function))
    {
        return Some(function.name.as_str());
    }

    program
        .functions
        .iter()
        .rev()
        .find(|function| tensor_signature(function))
        .map(|function| function.name.as_str())
}

pub fn lower_named_tensor_entry_dag(program: &CheckedProgram, name: &str) -> Option<crate::Dag> {
    let defs = collect_program_defs(program.exprs());
    let body = lookup_program_def(&defs, name)?.clone();
    let Expr::List(list, _) = &body else {
        return None;
    };
    if tag(list) != Some("fn") {
        return None;
    }

    let kids = children(list);
    let params_list = kids.first().and_then(as_list)?;
    if tag(params_list) != Some("params") {
        return None;
    }

    let declared_param_tys = lookup_declared_type_expr(program, name)
        .and_then(parse_fn_type_expr)
        .map(|(params, _)| params)
        .unwrap_or_default();
    let mut scope = HashMap::new();
    for (index, param) in children(params_list).iter().enumerate() {
        let pname = param_name(param)?;
        let pty = param_host_type(param)
            .or_else(|| declared_param_tys.get(index).cloned())
            .filter(|ty| *ty != HostType::Unknown)?;
        let HostType::Tensor(tensor_ty) = pty else {
            return None;
        };
        scope.insert(pname, tensor_ty);
    }

    let body_expr = kids.get(1)?;
    Some(crate::lower::lower_subexpr_program(
        body_expr,
        scope,
        program.type_env().clone(),
        defs,
    ))
}

fn lower_host_program(
    program: &CheckedProgram,
    lowered_names: &HashMap<String, bool>,
) -> HostProgram {
    let mut host = HostProgram::default();
    let mut global_scope = HashMap::new();
    // Count pure-tensor `fn`-body top-level defs in the program. When there
    // is more than one, the legacy DAG-only path would collapse them into a
    // single file-named entry point that drops all but one def's parameters
    // (Nautilus Bug 3c). In that case we emit a host wrapper per def so each
    // gets its own C symbol.
    let lowered_fn_def_count = top_level_items(program.exprs())
        .iter()
        .filter(|expr| {
            let Expr::List(list, _) = expr else {
                return false;
            };
            if tag(list) != Some("def") {
                return false;
            }
            let kids = children(list);
            let Some(def_name) = kids.first().and_then(symbol_name) else {
                return false;
            };
            let is_fn_body = matches!(kids.get(1), Some(Expr::List(body_list, _)) if tag(body_list) == Some("fn"));
            is_fn_body && lowered_names.get(def_name).copied().unwrap_or(false)
        })
        .count();
    for expr in top_level_items(program.exprs()) {
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
        let ty_expr = lookup_declared_type_expr(program, name);
        // Pure-tensor top-level function defs are normally lowered to the
        // DAG. But when the program also has host-lane bindings (i.e. some
        // def is NOT DAG-lowerable), downstream host-lane callers still
        // need a real C function symbol for the wrapper. In that case,
        // emit a HostFunction wrapper alongside the DAG lowering.
        //
        // A single pure-tensor function def can still use the legacy
        // DAG-only path (the file-named entry point wraps it 1:1 with the
        // correct signature). But when there are multiple pure-tensor
        // function defs in the same program, the DAG path would collapse
        // them into a single file-named entry that silently drops all but
        // one def's parameters and outputs (Nautilus Bug 3c). Emit a host
        // wrapper per def in that case so each gets its own C symbol.
        let is_fn_body = matches!(body, Expr::List(list, _) if tag(list) == Some("fn"));
        let has_any_host_lane_def = lowered_names.values().any(|lowered| !*lowered);
        let has_callable_params = lookup_declared_fn_type(program, name)
            .is_some_and(|(params, _)| params.iter().any(|ty| matches!(ty, HostType::Fn(..))));
        // Non-F32/Bool tensor precisions (e.g. int32, int64) aren't
        // representable in the Phase 0f DAG-only codegen path — it still
        // hard-asserts f32/bool. Force a host-lane wrapper for any fn whose
        // signature carries such a tensor so the program stays on the
        // host-lane code path instead of panicking in DAG emit.
        let has_non_dag_tensor =
            lookup_declared_fn_type(program, name).is_some_and(|(params, ret)| {
                fn ty_has_non_dag_tensor(ty: &HostType) -> bool {
                    match ty {
                        HostType::Tensor(tensor) => !matches!(
                            tensor.precision,
                            chelis_types::types::Prim::F32 | chelis_types::types::Prim::Bool
                        ),
                        HostType::Option(inner) | HostType::List(inner) => {
                            ty_has_non_dag_tensor(inner)
                        }
                        HostType::Tuple(items) => items.iter().any(ty_has_non_dag_tensor),
                        HostType::Dict(k, v) => {
                            ty_has_non_dag_tensor(k) || ty_has_non_dag_tensor(v)
                        }
                        HostType::Fn(params, ret) => {
                            params.iter().any(ty_has_non_dag_tensor) || ty_has_non_dag_tensor(ret)
                        }
                        _ => false,
                    }
                }
                params.iter().any(ty_has_non_dag_tensor) || ty_has_non_dag_tensor(&ret)
            });
        // `has_callable_params` blocks the wrapper for fn-taking-fn signatures
        // because the host emitter doesn't lower higher-order wrappers
        // cleanly in every shape (grad specialization etc.). But if the
        // signature ALSO contains a non-F32/Bool tensor, the DAG-only
        // fallback panics — so in that combination still force the wrapper.
        //
        // Known limitation: when a caller references a callable-param fn by
        // name and that fn's body doesn't lower cleanly through the host
        // wrapper, the def gets dropped from emission and the caller will
        // fail gcc with `implicit declaration`. Tracked as a residual HOF
        // emission issue (red-team A20.1).
        let needs_host_wrapper = is_fn_body
            && (has_non_dag_tensor
                || (!has_callable_params && (has_any_host_lane_def || lowered_fn_def_count > 1)));
        let skip_for_lowered =
            lowered_names.get(name).copied().unwrap_or(false) && !needs_host_wrapper;
        if skip_for_lowered {
            continue;
        }
        // The wrapper emitter doesn't know every pattern the DAG-path
        // specializer does (e.g. `grad(local_fn)(theta)`). When the host
        // lowering would degrade to a fallback `Builtin { name: "call" }`,
        // emitting the wrapper produces broken C (`__result = call(...)`).
        // In that case prefer the DAG path if it's available (lowered_names
        // says so); otherwise we have no good lowering and must drop the
        // def — at least the caller will get `implicit declaration` rather
        // than `call(…)` undefined-symbol.
        if let Some(mut function) = lower_host_function(name, body, ty_expr, program) {
            // N→1 lowering collapse per `spec/design/chelis_span_survival.md`
            // §2.3 host-side table, rule "Lowering — top-level def collapses
            // to fn body": when a `(def {span: a} name (fn ... body))` lowers
            // to a HostFunction whose `body` is the lowered fn body, the
            // def's `span_id` appends to the body node's `merged_spans` so
            // the def's source region surfaces in the audit chain. (The fn
            // node's own span, if any, is already on the body via the
            // body-collapse rule applied by `lower_host_expr`.)
            function.body.append_merged_span(expr.span_id());
            global_scope.insert(
                name.to_string(),
                HostType::Fn(
                    function
                        .params
                        .iter()
                        .map(|param| param.ty.clone())
                        .collect(),
                    Box::new(function.ret_ty.clone()),
                ),
            );
            host.functions.push(function);
        } else {
            let mut value = lower_host_expr(
                body,
                program,
                &global_scope,
                &mut host.global_tensor_helpers,
            );
            // N→1 lowering collapse per `spec/design/chelis_span_survival.md`
            // §2.3 host-side table, rule "Lowering — top-level def collapses
            // to body": when a `(def {span: a} name body)` lowers to a
            // HostBinding whose `value` is the body's HostExpr, the def's
            // own `span_id` (sourced from the def's `meta["span"]`) appends
            // to the value node's `merged_spans` so the def's source region
            // doesn't drop out of the audit chain.
            value.append_merged_span(expr.span_id());
            let ty = host_expr_type(&value);
            host.globals.push(HostBinding {
                name: name.to_string(),
                display_name: None,
                ty,
                value: value.clone(),
            });
            global_scope.insert(name.to_string(), host_expr_type(&value));
        }
    }
    loop {
        let mut changed = false;
        changed |= refine_host_function_signatures(&mut host.functions);
        changed |= refine_host_globals(&mut host.globals, &host.functions);
        changed |= propagate_named_callback_signatures(&mut host.functions, &host.globals);
        if !changed {
            break;
        }
    }
    host
}

/// Walk a HostExpr and report whether any node is the generic-fallback
/// `Builtin { name: "call", ... }` that `lower_app_host_expr` emits when
/// it doesn't recognize the callee. A wrapper containing this node would
/// emit broken C (`__result = call(...);`) downstream — preferring the
/// DAG path's inline specialization is safer than emitting that wrapper.
/// Scan a host program for any function whose body still contains the
/// `Builtin { name: "call" }` fallback — the lowerer emits this when
/// it can't recognize the callee (typically `grad(f)(theta)` where the
/// callee is itself an application). Emitting such a wrapper produces
/// `__result = call(...)` C code that doesn't link. The CLI uses this
/// to surface a clean error instead of shipping broken C.
pub fn host_program_unresolved_call_sites(program: &HostProgram) -> Vec<String> {
    let mut out = Vec::new();
    for function in &program.functions {
        if host_body_has_fallback_call(&function.body) {
            out.push(function.name.clone());
        }
    }
    for binding in &program.globals {
        if host_body_has_fallback_call(&binding.value) {
            out.push(binding.name.clone());
        }
    }
    out
}

/// Scan a host program for functions with `HostType::Unknown` params or
/// return types — this happens for polymorphic defs where the type
/// variable hasn't been resolved at lowering time (e.g. `hamt_put[a]`
/// with `value: a`). The emitter currently collapses `Unknown` to `int`
/// in C, which breaks links when callers pass concrete pointer types.
/// Returns a list of `(def_name, position)` pairs for reporting.
pub fn host_program_unknown_typed_params(program: &HostProgram) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for function in &program.functions {
        for param in &function.params {
            if host_type_contains_unknown(&param.ty) {
                out.push((
                    function.name.clone(),
                    format!("parameter `{}` has unresolved polymorphic type", param.name),
                ));
            }
        }
        if host_type_contains_unknown(&function.ret_ty) {
            out.push((
                function.name.clone(),
                "return type is unresolved polymorphic".to_string(),
            ));
        }
    }
    out
}

fn host_type_contains_unknown(ty: &HostType) -> bool {
    match ty {
        HostType::Unknown => true,
        HostType::Option(inner) | HostType::List(inner) => host_type_contains_unknown(inner),
        HostType::Tuple(items) => items.iter().any(host_type_contains_unknown),
        HostType::Dict(k, v) => host_type_contains_unknown(k) || host_type_contains_unknown(v),
        HostType::Fn(params, ret) => {
            params.iter().any(host_type_contains_unknown) || host_type_contains_unknown(ret)
        }
        HostType::Adt(_, args) => args.iter().any(host_type_contains_unknown),
        _ => false,
    }
}

fn host_body_has_fallback_call(expr: &HostExpr) -> bool {
    match &expr.kind {
        HostExprKind::Builtin { name, args, .. } => {
            name == "call"
                || name.starts_with("__unresolved_")
                || args.iter().any(host_body_has_fallback_call)
        }
        HostExprKind::Call { function, args, .. } => {
            function == "call" || args.iter().any(host_body_has_fallback_call)
        }
        HostExprKind::Let { bindings, body, .. } => {
            bindings
                .iter()
                .any(|b| host_body_has_fallback_call(&b.value))
                || host_body_has_fallback_call(body)
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            host_body_has_fallback_call(cond)
                || host_body_has_fallback_call(then_expr)
                || host_body_has_fallback_call(else_expr)
        }
        HostExprKind::Tuple(items, _) | HostExprKind::List(items, _) => {
            items.iter().any(host_body_has_fallback_call)
        }
        HostExprKind::Map { list, .. }
        | HostExprKind::Filter { list, .. }
        | HostExprKind::Fold { list, .. }
        | HostExprKind::Scan { list, .. }
        | HostExprKind::Partition { list, .. }
        | HostExprKind::FlatMap { list, .. } => host_body_has_fallback_call(list),
        HostExprKind::TensorCall { args, .. } => args.iter().any(host_body_has_fallback_call),
        HostExprKind::AdtFieldAccess { base, .. } => host_body_has_fallback_call(base),
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            host_body_has_fallback_call(scrutinee)
                || host_body_has_fallback_call(some_expr)
                || host_body_has_fallback_call(none_expr)
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            host_body_has_fallback_call(scrutinee)
                || arms
                    .iter()
                    .any(|arm| host_body_has_fallback_call(&arm.expr))
                || default_expr
                    .as_ref()
                    .is_some_and(|d| host_body_has_fallback_call(d))
        }
        HostExprKind::AdtConstruct { fields, .. } => fields.iter().any(host_body_has_fallback_call),
        _ => false,
    }
}

fn lower_host_function(
    name: &str,
    body: &Expr,
    ty_expr: Option<&Expr>,
    program: &CheckedProgram,
) -> Option<HostFunction> {
    let declared_fn_type_expr = ty_expr
        .cloned()
        .or_else(|| lookup_declared_type_expr(program, name).cloned());
    let fn_type_parts = declared_fn_type_expr
        .as_ref()
        .and_then(parse_fn_type_expr_parts);
    let (param_tys, ret_ty) = ty_expr
        .and_then(parse_fn_type_expr)
        .or_else(|| expr_fn_type(body))
        .or_else(|| lookup_declared_fn_type(program, name))
        .unwrap_or((Vec::new(), HostType::Unknown));

    let mut scope = HashMap::new();
    let mut params = Vec::new();
    let mut tensor_helpers = Vec::new();
    let body_expr = if let Expr::List(list, _) = body {
        if tag(list) == Some("fn") {
            let kids = children(list);
            let params_list = kids.first().and_then(as_list)?;
            if tag(params_list) != Some("params") {
                return None;
            }
            for (index, param) in children(params_list).iter().enumerate() {
                let Some(pname) = param_name(param) else {
                    continue;
                };
                let pty = param_host_type(param)
                    .filter(|ty| *ty != HostType::Unknown)
                    .or_else(|| {
                        param_tys
                            .get(index)
                            .cloned()
                            .filter(|ty| *ty != HostType::Unknown)
                    })
                    .unwrap_or(HostType::Unknown);
                scope.insert(pname.clone(), pty.clone());
                params.push(HostParam {
                    name: pname,
                    ty: pty,
                });
            }
            kids.get(1)?.clone()
        } else {
            if param_tys.is_empty() && ret_ty == HostType::Unknown {
                return None;
            }
            for (index, param_ty) in param_tys.iter().enumerate() {
                let pname = format!("arg{index}");
                scope.insert(pname.clone(), param_ty.clone());
                params.push(HostParam {
                    name: pname,
                    ty: param_ty.clone(),
                });
            }
            synthesize_callable_application(
                body,
                &params,
                fn_type_parts.as_ref().map(|(params, _)| params.as_slice()),
                fn_type_parts.as_ref().map(|(_, ret)| ret),
            )
        }
    } else {
        return None;
    };
    // If the declared return type is a tensor, the body must produce a
    // tensor even when downstream type-metadata annotations are missing
    // from the reef'd deep AST. Force the body through the tensor-helper
    // path in that case so that pure-tensor wrapper defs like
    // `Std.Tensor.Reduce.min` get a real C function symbol rather than a
    // fallthrough `HostExpr::new(HostExprKind::Builtin)` with an "unsupported builtin"
    // placeholder (Phase 3j-pre Batch 5b bug 4).
    //
    // But skip the tensor-helper path when any param is callable: the DAG
    // helper has no representation for fn-pointer inputs and would otherwise
    // coerce the callable into `chelis_scalar_tensor_from_f64`, emitting C
    // that gcc rejects. Go straight through host-lane lowering so the fn
    // application becomes a direct `f(x)` call.
    let any_callable_param = params
        .iter()
        .any(|param| matches!(param.ty, HostType::Fn(_, _)));
    let mut host_body = if let HostType::Tensor(expected) = ret_ty.clone()
        && !any_callable_param
        && !expr_needs_host_lane_tensor_lowering(&body_expr, program)
        && !should_keep_tensor_expr_in_host_lane(&body_expr)
    {
        try_lower_tensor_helper_call(&body_expr, program, &scope, &mut tensor_helpers, expected)
            .unwrap_or_else(|| lower_host_expr(&body_expr, program, &scope, &mut tensor_helpers))
    } else {
        lower_host_expr(&body_expr, program, &scope, &mut tensor_helpers)
    };
    // Per `spec/design/chelis_span_survival.md` §2.3 host-side table, the
    // "Tensor-helper extraction" and "Lowering — fn-body" rules: every
    // input Deep span must surface as `span_id` or in `merged_spans` on at
    // least one HostExpr node. Two paths above bypass the
    // `lower_host_expr` wrapper's region-corresponding stamping:
    //
    // (1) `try_lower_tensor_helper_call(&body_expr, …)` at the success
    //     branch returns a `TensorCall` constructed via `HostExpr::new(…)`
    //     directly — `body_expr.span_id()` (the fn-body inner expr's
    //     span) is dropped.
    // (2) When `body` is a `(fn {span: …} (params …) body_expr)` form,
    //     all three subpaths above lower `body_expr` (kids[1]) but
    //     never see `body` itself — so the `(fn …)` form's own
    //     `meta["span"]` is dropped on every path.
    //
    // Append both spans to `host_body.merged_spans`. The
    // `append_merged_span` helper handles None-noop, dedup, lex-sort,
    // and canonical-equal-noop, so paths that already have the span as
    // canonical (the wrapper-routed lowering) are a no-op.
    host_body.append_merged_span(body_expr.span_id());
    host_body.append_merged_span(body.span_id());
    refine_function_params_from_body(&mut params, &host_body);
    let ret_ty = if ret_ty == HostType::Unknown {
        host_expr_type(&host_body)
    } else {
        ret_ty
    };
    Some(HostFunction {
        name: name.to_string(),
        params,
        ret_ty,
        body: host_body,
        tensor_helpers,
    })
}

fn synthesize_callable_application(
    body: &Expr,
    params: &[HostParam],
    param_type_exprs: Option<&[Expr]>,
    ret_type_expr: Option<&Expr>,
) -> Expr {
    let span = body.span();
    let mut elements = vec![
        Expr::Atom(Atom::Symbol("app".to_string()), span),
        Expr::Map(
            chelis_deep::ast::MetaMap {
                entries: ret_type_expr
                    .cloned()
                    .map(|ret| vec![("type".to_string(), ret)])
                    .unwrap_or_default(),
            },
            span,
        ),
        body.clone(),
    ];
    for (index, param) in params.iter().enumerate() {
        let var_meta = chelis_deep::ast::MetaMap {
            entries: param_type_exprs
                .and_then(|tys| tys.get(index))
                .cloned()
                .map(|ty| vec![("type".to_string(), ty)])
                .unwrap_or_default(),
        };
        elements.push(Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Symbol("var".to_string()), span),
                    Expr::Map(var_meta, span),
                    Expr::Atom(Atom::Symbol(param.name.clone()), span),
                ],
            },
            span,
        ));
    }
    Expr::List(List { elements }, span)
}

/// Force-lower an expression through the tensor-helper path using an
/// explicit expected tensor type hint. This is used by
/// `lower_host_function` so that pure-tensor wrapper function bodies get a
/// real C function definition even when downstream type metadata is
/// missing on the reef'd deep AST's `app` nodes.
fn try_lower_tensor_helper_call(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
    expected: TensorType,
) -> Option<HostExpr> {
    if let Expr::List(list, _) = expr
        && tag(list) == Some("var")
        && let Some(name) = children(list).first().and_then(symbol_name)
    {
        return Some(HostExpr::new(HostExprKind::Var(
            name.to_string(),
            HostType::Tensor(expected),
        )));
    }
    // Mark this catch_unwind scope so the DAG lowerer's `lower_unrepresentable`
    // can abort quietly instead of printing a stderr panic-location trace.
    // The outer host-lane fallback handles the un-representable case cleanly;
    // the panic message itself was pure noise.
    let result = crate::lower::with_suppress_unrepresentable_panic(|| {
        catch_unwind(AssertUnwindSafe(|| {
            lower_tensor_helper_dag(expr, program, scope, &expected)
        }))
    });
    let dag = result.ok()?;
    // Reject DAGs whose inputs reference known builtin names: a `Load("fold")`
    // (or `einsum`, `map`, etc.) means the lowerer fell back to treating a
    // host-lane builtin as a free variable. Emitting this DAG would generate
    // C with a `__tensor_scalar0_0 = fold;` line — `fold` is not a C symbol.
    // Fall back to `lower_host_expr` which handles HOFs directly.
    for node in dag.nodes() {
        if let crate::dag::RiscOp::Load { name } = &node.op
            && BUILTIN_NAMES.contains(&name.as_str())
        {
            return None;
        }
    }
    Some(finish_tensor_helper_call(
        dag,
        scope,
        tensor_helpers,
        expected,
    ))
}

fn lower_tensor_helper_dag(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    expected: &TensorType,
) -> crate::Dag {
    let dag = crate::lower::lower_subexpr_program(
        expr,
        collect_tensor_scope(scope),
        program.type_env().clone(),
        collect_program_defs(program.exprs()),
    );
    remap_tensor_helper_dim_symbols(&dag, scope, expected)
}

fn finish_tensor_helper_call(
    dag: crate::Dag,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
    expected: TensorType,
) -> HostExpr {
    let helper_index = tensor_helpers.len();
    let helper_name = format!("__host_tensor_helper_{helper_index}");
    let inputs = tensor_helper_inputs(&dag);
    let output = dag
        .roots()
        .first()
        .and_then(|id| dag.get(*id))
        .map(|node| node.output_type.clone())
        .unwrap_or_else(|| expected.clone());
    let args = tensor_helper_args(&inputs, scope);
    tensor_helpers.push(HostTensorHelper {
        name: helper_name,
        dag,
        inputs,
        output,
    });
    HostExpr::new(HostExprKind::TensorCall {
        helper: helper_index,
        args,
        ty: HostType::Tensor(expected),
    })
}

fn lower_host_expr(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let mut result = lower_host_expr_kind(expr, program, scope, tensor_helpers);
    // Per `spec/design/chelis_span_survival.md` §2.3 host-side table,
    // rule "Lowering": every freshly-produced HostExpr inherits the
    // enclosing Deep `Expr`'s `meta["span"]` as its `span_id`. When the
    // result already carries a different `span_id` (because lowering
    // recursed into a child whose own span was attached first — the N→1
    // body-collapse case), the parent's span appends to `merged_spans`,
    // lex-sorted and deduped, so the audit chain doesn't drop it. The
    // helper handles the no-op cases (None, equal canonical, already
    // present).
    if let Some(span) = expr.span_id() {
        if result.span_id.is_none() {
            result.span_id = Some(span.to_owned());
        } else {
            result.append_merged_span(Some(span));
        }
    }
    result
}

fn lower_host_expr_kind(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let is_app_expr = matches!(expr, Expr::List(list, _) if tag(list) == Some("app"));
    if !is_app_expr
        && let Some(tensor_ty) = expr_tensor_type(expr, program, scope)
        && !should_keep_tensor_expr_in_host_lane(expr)
        && let Some(tensor_call) =
            try_lower_tensor_helper_call(expr, program, scope, tensor_helpers, tensor_ty.clone())
    {
        return tensor_call;
    }

    match expr {
        Expr::Atom(Atom::Int(value), _) => HostExpr::new(HostExprKind::Int(*value)),
        Expr::Atom(Atom::Float(value), _) => HostExpr::new(HostExprKind::Float(*value)),
        Expr::Atom(Atom::Bool(value), _) => HostExpr::new(HostExprKind::Bool(*value)),
        Expr::Atom(Atom::Str(value), _) => HostExpr::new(HostExprKind::String(value.clone())),
        Expr::List(list, _) if tag(list) == Some("tuple") => {
            let items = children(list)
                .iter()
                .map(|child| lower_host_expr(child, program, scope, tensor_helpers))
                .collect::<Vec<_>>();
            let ty = expr_host_type(expr, program, scope);
            let ty = if ty == HostType::Unknown {
                HostType::Tuple(items.iter().map(host_expr_type).collect())
            } else {
                ty
            };
            HostExpr::new(HostExprKind::Tuple(items, ty))
        }
        Expr::List(list, _) if tag(list) == Some("record") => {
            lower_record_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("lit") => lower_host_expr(
            children(list).first().unwrap_or(expr),
            program,
            scope,
            tensor_helpers,
        ),
        Expr::List(list, _) if tag(list) == Some("var") => {
            let name = children(list)
                .first()
                .and_then(symbol_name)
                .unwrap_or("_")
                .to_string();
            let ty = expr_type(expr)
                .filter(|ty| *ty != HostType::Unknown)
                .or_else(|| scope.get(&name).cloned())
                .or_else(|| lookup_declared_host_type(program, &name))
                .unwrap_or(HostType::Unknown);
            if name == "Nil" {
                return HostExpr::new(HostExprKind::List(
                    Vec::new(),
                    match ty {
                        HostType::List(_) => ty,
                        _ => HostType::List(Box::new(HostType::Unknown)),
                    },
                ));
            }
            if let Some((adt_name, fields)) = lookup_adt_ctor(program, &name)
                && fields.is_empty()
            {
                return HostExpr::new(HostExprKind::AdtConstruct {
                    ctor: name,
                    fields: Vec::new(),
                    ty: HostType::Adt(adt_name, Vec::new()),
                });
            }
            HostExpr::new(HostExprKind::Var(name, ty))
        }
        Expr::List(list, _) if tag(list) == Some("if") => {
            let kids = children(list);
            let then_expr = lower_host_expr(&kids[1], program, scope, tensor_helpers);
            let else_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
            let explicit_ty = expr_host_type(expr, program, scope);
            let ty = if explicit_ty == HostType::Unknown {
                let then_ty = host_expr_type(&then_expr);
                if then_ty == HostType::Unknown {
                    host_expr_type(&else_expr)
                } else {
                    then_ty
                }
            } else {
                explicit_ty
            };
            HostExpr::new(HostExprKind::If {
                cond: Box::new(lower_host_expr(&kids[0], program, scope, tensor_helpers)),
                then_expr: Box::new(then_expr),
                else_expr: Box::new(else_expr),
                ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some("match") => {
            lower_match_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("let") => {
            let kids = children(list);
            let mut scoped = scope.clone();
            let mut bindings = Vec::new();
            if let Some(bind_first) = kids.first()
                && let Some(bind_list) = as_list(bind_first)
                && tag(bind_list) == Some("bind")
            {
                // The `(bind {span: a} ...)` node carries its own span;
                // when its child value is lowered to an existing HostExpr
                // (the value HostExpr), the bind's span appends to the
                // value's `merged_spans` per §2.3 host-side rule (b)
                // (N→1 lowering collapse — bind wraps value).
                let bind_span = bind_first.span_id().map(|s| s.to_owned());
                let bind_children = children(bind_list);
                let mut index = 0;
                while index + 1 < bind_children.len() {
                    if let Some(name) = symbol_name(&bind_children[index]) {
                        let mut value = lower_host_expr(
                            &bind_children[index + 1],
                            program,
                            &scoped,
                            tensor_helpers,
                        );
                        value.append_merged_span(bind_span.as_deref());
                        let bind_ty = host_expr_type(&value);
                        bindings.push(HostBinding {
                            name: name.to_string(),
                            display_name: None,
                            ty: bind_ty.clone(),
                            value,
                        });
                        scoped.insert(name.to_string(), bind_ty);
                    }
                    index += 2;
                }
            }
            let body = kids
                .get(1)
                .map(|child| lower_host_expr(child, program, &scoped, tensor_helpers))
                .unwrap_or(HostExpr::new(HostExprKind::Unit));
            let explicit_ty = expr_host_type(expr, program, scope);
            HostExpr::new(HostExprKind::Let {
                bindings,
                body: Box::new(body.clone()),
                ty: if explicit_ty == HostType::Unknown {
                    host_expr_type(&body)
                } else {
                    explicit_ty
                },
            })
        }
        Expr::List(list, _) if tag(list) == Some("tuple-get") => {
            lower_tuple_get_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("access") => {
            lower_access_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("cast") => {
            let value = lower_host_expr(
                children(list).first().unwrap_or(expr),
                program,
                scope,
                tensor_helpers,
            );
            let inferred_ty = host_expr_type(&value);
            let ty = expr_host_type(expr, program, scope);
            HostExpr::new(HostExprKind::Builtin {
                name: "cast".to_string(),
                args: vec![value],
                ty: if ty == HostType::Unknown {
                    inferred_ty
                } else {
                    ty
                },
            })
        }
        Expr::List(list, _) if tag(list) == Some("app") => {
            lower_app_host_expr(list, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("handle-effect") => {
            // `with seed(...) { body }` and similar effect handlers are
            // pure-result from the host emitter's perspective — the seed
            // flows into random-op lowering at DAG-build time and the
            // visible value is just `body`. Without this arm, the whole
            // form fell through to `HostExpr::new(HostExprKind::Unit)`, which is why
            // `kaiming_uniform` showed up as `()` in compiled output.
            let kids = children(list);
            // children(list) skips tag and metadata map, so for
            // `(handle-effect {effect: random, ...} seed body)` kids[0] is
            // the seed expression and kids[1] is the body. Some forms may
            // omit the seed slot.
            let body = kids.get(1).or_else(|| kids.first());
            if let Some(body) = body {
                lower_host_expr(body, program, scope, tensor_helpers)
            } else {
                HostExpr::new(HostExprKind::Unit)
            }
        }
        Expr::MetaExpr(meta, _) => lower_host_expr(&meta.expr, program, scope, tensor_helpers),
        Expr::List(list, _) if matches!(tag(list), Some("grad" | "vmap" | "vmap-grad")) => {
            // Higher-order differentiation/vmap expressions aren't representable
            // as host-lane values. When one appears in host position (e.g.
            // `g = grad(f)` bound to a local), lowering silently fell through
            // to `HostExpr::new(HostExprKind::Unit)`, which later produced `int g = 0` and a
            // no-op `/* unsupported builtin g */` in the emitted C — a silent
            // wrong-answer. Produce a recognizable marker Builtin instead so
            // `host_program_unresolved_call_sites` can surface it as a clean
            // pre-codegen error.
            let tag_name = tag(list).unwrap_or("grad").to_string();
            HostExpr::new(HostExprKind::Builtin {
                name: format!("__unresolved_{tag_name}"),
                args: Vec::new(),
                ty: expr_host_type(expr, program, scope),
            })
        }
        Expr::List(list, _) if tag(list) == Some("copy") => {
            // `(copy {} x)` exists for linearity bookkeeping. In the host
            // lane, lower as a tagged Builtin whose C emission is a direct
            // tensor-copy or a pass-through for non-tensors. Without this
            // arm the `copy` tag silently fell through to `HostExpr::new(HostExprKind::Unit)`,
            // which caused tensor-if bodies in folds to collapse to
            // `int new_t; new_t = 0;` (Nautilus P2 tensor-if-in-fold).
            let inner = children(list)
                .first()
                .map(|child| lower_host_expr(child, program, scope, tensor_helpers))
                .unwrap_or(HostExpr::new(HostExprKind::Unit));
            let inner_ty = host_expr_type(&inner);
            let explicit = expr_host_type(expr, program, scope);
            let ty = if explicit == HostType::Unknown {
                inner_ty
            } else {
                explicit
            };
            HostExpr::new(HostExprKind::Builtin {
                name: "copy".to_string(),
                args: vec![inner],
                ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some("realize") => {
            // Passthrough for phase-0 semantics: realize is an identity in
            // host lane.
            children(list)
                .first()
                .map(|child| lower_host_expr(child, program, scope, tensor_helpers))
                .unwrap_or(HostExpr::new(HostExprKind::Unit))
        }
        _ => HostExpr::new(HostExprKind::Unit),
    }
}

fn refine_function_params_from_body(params: &mut [HostParam], body: &HostExpr) {
    let HostExprKind::Call { args, arg_tys, .. } = &body.kind else {
        return;
    };
    for (index, param) in params.iter_mut().enumerate() {
        if param.ty != HostType::Unknown {
            continue;
        }
        let Some(HostExpr {
            kind: HostExprKind::Var(name, _),
            ..
        }) = args.get(index)
        else {
            continue;
        };
        if name != &param.name {
            continue;
        }
        let Some(inferred) = arg_tys.get(index) else {
            continue;
        };
        if *inferred != HostType::Unknown {
            param.ty = inferred.clone();
        }
    }
}

fn refine_host_function_signatures(functions: &mut [HostFunction]) -> bool {
    let mut any_changed = false;
    loop {
        let mut changed = false;
        let signatures = functions
            .iter()
            .map(|function| {
                (
                    function.name.clone(),
                    (
                        function
                            .params
                            .iter()
                            .map(|param| param.ty.clone())
                            .collect::<Vec<_>>(),
                        function.ret_ty.clone(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();

        for function in functions.iter_mut() {
            let inferred_param_fns = infer_callable_param_types(&function.params, &function.body);
            for param in function.params.iter_mut() {
                if param.ty == HostType::Unknown
                    && let Some(inferred) = inferred_param_fns.get(&param.name)
                    && !host_type_has_unknown(inferred)
                    && param.ty != *inferred
                {
                    param.ty = inferred.clone();
                    changed = true;
                }
            }

            let mut scope = function
                .params
                .iter()
                .map(|param| (param.name.clone(), param.ty.clone()))
                .collect::<HashMap<_, _>>();
            if refine_host_expr_types(&mut function.body, &mut scope, &signatures) {
                changed = true;
            }
            if function.ret_ty == HostType::Unknown {
                let body_ty = host_expr_type(&function.body);
                if body_ty != HostType::Unknown {
                    function.ret_ty = body_ty;
                    changed = true;
                }
            }

            let HostExprKind::Call {
                function: callee,
                args,
                arg_tys,
                ty,
            } = &mut function.body.kind
            else {
                continue;
            };
            let Some((callee_params, callee_ret)) = signatures.get(callee) else {
                continue;
            };

            if function.ret_ty == HostType::Unknown && *callee_ret != HostType::Unknown {
                function.ret_ty = callee_ret.clone();
                if *ty == HostType::Unknown {
                    *ty = callee_ret.clone();
                }
                changed = true;
            }

            for (index, param) in function.params.iter_mut().enumerate() {
                if param.ty != HostType::Unknown {
                    continue;
                }
                let Some(HostExpr {
                    kind: HostExprKind::Var(name, arg_ty),
                    ..
                }) = args.get_mut(index)
                else {
                    continue;
                };
                if name != &param.name {
                    continue;
                }
                let Some(inferred) = callee_params.get(index) else {
                    continue;
                };
                if *inferred == HostType::Unknown {
                    continue;
                }
                param.ty = inferred.clone();
                *arg_ty = inferred.clone();
                if let Some(call_arg_ty) = arg_tys.get_mut(index) {
                    *call_arg_ty = inferred.clone();
                }
                changed = true;
            }
        }

        if !changed {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn refine_host_globals(globals: &mut [HostBinding], functions: &[HostFunction]) -> bool {
    let mut any_changed = false;
    loop {
        let signatures = functions
            .iter()
            .map(|function| {
                (
                    function.name.clone(),
                    (
                        function
                            .params
                            .iter()
                            .map(|param| param.ty.clone())
                            .collect::<Vec<_>>(),
                        function.ret_ty.clone(),
                    ),
                )
            })
            .collect::<HashMap<_, _>>();

        let mut changed = false;
        let mut scope = HashMap::new();
        for binding in globals.iter_mut() {
            if refine_host_expr_types(&mut binding.value, &mut scope, &signatures) {
                changed = true;
            }
            let inferred = host_expr_type(&binding.value);
            if binding.ty != inferred && inferred != HostType::Unknown {
                binding.ty = inferred.clone();
                changed = true;
            }
            scope.insert(binding.name.clone(), binding.ty.clone());
        }

        if !changed {
            break;
        }
        any_changed = true;
    }
    any_changed
}

fn propagate_named_callback_signatures(
    functions: &mut [HostFunction],
    globals: &[HostBinding],
) -> bool {
    let mut inferred = HashMap::<String, (Vec<HostType>, HostType)>::new();
    for function in functions.iter() {
        collect_named_callback_signatures(&function.body, &mut inferred);
    }
    for binding in globals {
        collect_named_callback_signatures(&binding.value, &mut inferred);
    }

    let mut changed = false;
    for function in functions.iter_mut() {
        let Some((param_tys, ret_ty)) = inferred.get(&function.name) else {
            continue;
        };
        for (param, inferred_ty) in function.params.iter_mut().zip(param_tys.iter()) {
            if host_type_has_unknown(&param.ty) && !host_type_has_unknown(inferred_ty) {
                param.ty = inferred_ty.clone();
                changed = true;
            }
        }
        if host_type_has_unknown(&function.ret_ty) && !host_type_has_unknown(ret_ty) {
            function.ret_ty = ret_ty.clone();
            changed = true;
        }
    }
    changed
}

fn collect_named_callback_signatures(
    expr: &HostExpr,
    out: &mut HashMap<String, (Vec<HostType>, HostType)>,
) {
    match &expr.kind {
        HostExprKind::Call { args, .. } | HostExprKind::Builtin { args, .. } => {
            for arg in args {
                collect_named_callback_signatures(arg, out);
            }
        }
        HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
            for item in items {
                collect_named_callback_signatures(item, out);
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields {
                collect_named_callback_signatures(field, out);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            collect_named_callback_signatures(base, out);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            collect_named_callback_signatures(cond, out);
            collect_named_callback_signatures(then_expr, out);
            collect_named_callback_signatures(else_expr, out);
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            collect_named_callback_signatures(scrutinee, out);
            collect_named_callback_signatures(some_expr, out);
            collect_named_callback_signatures(none_expr, out);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            collect_named_callback_signatures(scrutinee, out);
            for arm in arms {
                collect_named_callback_signatures(&arm.expr, out);
            }
            if let Some(default_expr) = default_expr {
                collect_named_callback_signatures(default_expr, out);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                collect_named_callback_signatures(&binding.value, out);
            }
            collect_named_callback_signatures(body, out);
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            merge_named_callback_signature(callback, out);
            collect_named_callback_signatures_in_callback(callback, out);
            collect_named_callback_signatures(list, out);
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            merge_named_callback_signature(callback, out);
            collect_named_callback_signatures_in_callback(callback, out);
            collect_named_callback_signatures(init, out);
            collect_named_callback_signatures(list, out);
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args {
                collect_named_callback_signatures(arg, out);
            }
        }
        HostExprKind::Var(_, _)
        | HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
}

fn collect_named_callback_signatures_in_callback(
    callback: &HostCallback,
    out: &mut HashMap<String, (Vec<HostType>, HostType)>,
) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        collect_named_callback_signatures(body, out);
    }
}

fn merge_named_callback_signature(
    callback: &HostCallback,
    out: &mut HashMap<String, (Vec<HostType>, HostType)>,
) {
    let HostCallbackKind::Named { function, params } = &callback.kind else {
        return;
    };
    let entry = out
        .entry(function.clone())
        .or_insert_with(|| (vec![HostType::Unknown; params.len()], HostType::Unknown));
    if entry.0.len() < params.len() {
        entry.0.resize(params.len(), HostType::Unknown);
    }
    for (index, param) in params.iter().enumerate() {
        if entry.0[index] == HostType::Unknown && param.ty != HostType::Unknown {
            entry.0[index] = param.ty.clone();
        }
    }
    if entry.1 == HostType::Unknown && callback.ret_ty != HostType::Unknown {
        entry.1 = callback.ret_ty.clone();
    }
}

fn infer_callable_param_types(params: &[HostParam], body: &HostExpr) -> HashMap<String, HostType> {
    let unknown = params
        .iter()
        .filter(|param| param.ty == HostType::Unknown)
        .map(|param| param.name.clone())
        .collect::<HashSet<_>>();
    let mut out = HashMap::new();
    infer_callable_param_types_in_expr(body, &unknown, &mut out);
    out
}

fn infer_callable_param_types_in_expr(
    expr: &HostExpr,
    unknown: &HashSet<String>,
    out: &mut HashMap<String, HostType>,
) {
    match &expr.kind {
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ..
        } => {
            if unknown.contains(function) {
                out.entry(function.clone()).or_insert_with(|| {
                    HostType::Fn(arg_tys.clone(), Box::new(HostType::Tuple(Vec::new())))
                });
            }
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
            for item in items {
                infer_callable_param_types_in_expr(item, unknown, out);
            }
        }
        HostExprKind::Builtin { args, .. } => {
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields {
                infer_callable_param_types_in_expr(field, unknown, out);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            infer_callable_param_types_in_expr(base, unknown, out);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(cond, unknown, out);
            infer_callable_param_types_in_expr(then_expr, unknown, out);
            infer_callable_param_types_in_expr(else_expr, unknown, out);
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(scrutinee, unknown, out);
            infer_callable_param_types_in_expr(some_expr, unknown, out);
            infer_callable_param_types_in_expr(none_expr, unknown, out);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            infer_callable_param_types_in_expr(scrutinee, unknown, out);
            for arm in arms {
                infer_callable_param_types_in_expr(&arm.expr, unknown, out);
            }
            if let Some(default_expr) = default_expr {
                infer_callable_param_types_in_expr(default_expr, unknown, out);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                infer_callable_param_types_in_expr(&binding.value, unknown, out);
            }
            infer_callable_param_types_in_expr(body, unknown, out);
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            infer_callable_param_types_in_callback(callback, unknown, out);
            infer_callable_param_types_in_expr(list, unknown, out);
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            infer_callable_param_types_in_callback(callback, unknown, out);
            infer_callable_param_types_in_expr(init, unknown, out);
            infer_callable_param_types_in_expr(list, unknown, out);
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args {
                infer_callable_param_types_in_expr(arg, unknown, out);
            }
        }
        HostExprKind::Var(_, _)
        | HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
}

fn infer_callable_param_types_in_callback(
    callback: &HostCallback,
    unknown: &HashSet<String>,
    out: &mut HashMap<String, HostType>,
) {
    if let HostCallbackKind::Inline { body, .. } = &callback.kind {
        infer_callable_param_types_in_expr(body, unknown, out);
    }
}

fn refine_host_expr_types(
    expr: &mut HostExpr,
    scope: &mut HashMap<String, HostType>,
    signatures: &HashMap<String, (Vec<HostType>, HostType)>,
) -> bool {
    let mut changed = false;
    match &mut expr.kind {
        HostExprKind::Var(name, ty) => {
            if host_type_has_unknown(ty)
                && let Some(inferred) = scope.get(name)
                && !host_type_has_unknown(inferred)
            {
                *ty = inferred.clone();
                changed = true;
            }
        }
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
            let inferred_sig = scope.get(function).and_then(|ty| match ty {
                HostType::Fn(params, ret) => Some((params.clone(), (**ret).clone())),
                _ => None,
            });
            let declared_sig = signatures.get(function).cloned();
            let sig = inferred_sig.or(declared_sig);
            if let Some((params, ret)) = sig {
                for (index, arg_ty) in arg_tys.iter_mut().enumerate() {
                    if let Some(inferred) = params.get(index)
                        && *inferred != HostType::Unknown
                        && *arg_ty != *inferred
                    {
                        *arg_ty = inferred.clone();
                        changed = true;
                    }
                }
                if *ty == HostType::Unknown && ret != HostType::Unknown {
                    *ty = ret;
                    changed = true;
                }
            }
        }
        HostExprKind::Builtin { name, args, ty } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
            if let Some(inferred) = infer_builtin_host_type(name, args)
                && !host_type_has_unknown(&inferred)
                && *ty != inferred
            {
                *ty = inferred;
                changed = true;
            }
        }
        HostExprKind::List(items, ty) => {
            for item in items.iter_mut() {
                changed |= refine_host_expr_types(item, scope, signatures);
            }
            let item_ty = items
                .iter()
                .map(host_expr_type)
                .find(|item_ty| !host_type_has_unknown(item_ty))
                .unwrap_or(HostType::Unknown);
            let inferred = HostType::List(Box::new(item_ty));
            if host_type_has_unknown(ty) && !host_type_has_unknown(&inferred) {
                *ty = inferred;
                changed = true;
            }
        }
        HostExprKind::Tuple(items, ty) => {
            for item in items.iter_mut() {
                changed |= refine_host_expr_types(item, scope, signatures);
            }
            let inferred = HostType::Tuple(items.iter().map(host_expr_type).collect());
            if host_type_has_unknown(ty) && !host_type_has_unknown(&inferred) {
                *ty = inferred;
                changed = true;
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields.iter_mut() {
                changed |= refine_host_expr_types(field, scope, signatures);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => {
            changed |= refine_host_expr_types(base, scope, signatures);
        }
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(cond, scope, signatures);
            changed |= refine_host_expr_types(then_expr, scope, signatures);
            changed |= refine_host_expr_types(else_expr, scope, signatures);
            if *ty == HostType::Unknown {
                let then_ty = host_expr_type(then_expr);
                let else_ty = host_expr_type(else_expr);
                let inferred = if then_ty != HostType::Unknown {
                    then_ty
                } else {
                    else_ty
                };
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(scrutinee, scope, signatures);
            let mut some_scope = scope.clone();
            let inner_ty = option_inner_type(scrutinee);
            if inner_ty != HostType::Unknown {
                some_scope.insert(bind_name.clone(), inner_ty);
            }
            changed |= refine_host_expr_types(some_expr, &mut some_scope, signatures);
            changed |= refine_host_expr_types(none_expr, scope, signatures);
            if *ty == HostType::Unknown {
                let some_ty = host_expr_type(some_expr);
                let none_ty = host_expr_type(none_expr);
                let inferred = if some_ty != HostType::Unknown {
                    some_ty
                } else {
                    none_ty
                };
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        } => {
            changed |= refine_host_expr_types(scrutinee, scope, signatures);
            for arm in arms.iter_mut() {
                let mut arm_scope = scope.clone();
                for binding in &arm.bindings {
                    arm_scope.insert(binding.name.clone(), binding.ty.clone());
                }
                changed |= refine_host_expr_types(&mut arm.expr, &mut arm_scope, signatures);
            }
            if let Some(default_expr) = default_expr {
                changed |= refine_host_expr_types(default_expr, scope, signatures);
            }
            if *ty == HostType::Unknown {
                let inferred = arms
                    .iter()
                    .map(|arm| host_expr_type(&arm.expr))
                    .find(|ty| *ty != HostType::Unknown)
                    .or_else(|| default_expr.as_ref().map(|expr| host_expr_type(expr)));
                if let Some(inferred) = inferred
                    && inferred != HostType::Unknown
                {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Let { bindings, body, ty } => {
            let mut local_scope = scope.clone();
            for binding in bindings.iter_mut() {
                changed |= refine_host_expr_types(&mut binding.value, &mut local_scope, signatures);
                if binding.ty == HostType::Unknown {
                    let inferred = host_expr_type(&binding.value);
                    if inferred != HostType::Unknown {
                        binding.ty = inferred.clone();
                        changed = true;
                    }
                }
                local_scope.insert(binding.name.clone(), binding.ty.clone());
            }
            changed |= refine_host_expr_types(body, &mut local_scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(body);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Map { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            let expected_ret = list_item_type(ty);
            changed |= specialize_host_callback_types(callback, &[item_ty], &expected_ret);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(list);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Filter { callback, list, ty }
        | HostExprKind::Partition { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            changed |= specialize_host_callback_types(callback, &[item_ty], &HostType::Bool);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(list);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::FlatMap { callback, list, ty } => {
            changed |= refine_host_expr_types(list, scope, signatures);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            let expected_ret = match ty {
                HostType::List(inner) => HostType::List(Box::new((**inner).clone())),
                _ => HostType::Unknown,
            };
            changed |= specialize_host_callback_types(callback, &[item_ty], &expected_ret);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(list);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        } => {
            changed |= refine_host_expr_types(init, scope, signatures);
            changed |= refine_host_expr_types(list, scope, signatures);
            let init_ty = host_expr_type(init);
            let list_ty = host_expr_type(list);
            let item_ty = list_item_type(&list_ty);
            changed |=
                specialize_host_callback_types(callback, &[init_ty.clone(), item_ty], &init_ty);
            changed |= refine_host_callback_types(callback, scope, signatures);
            if *ty == HostType::Unknown {
                let inferred = host_expr_type(init);
                if inferred != HostType::Unknown {
                    *ty = inferred;
                    changed = true;
                }
            }
        }
        HostExprKind::TensorCall { args, .. } => {
            for arg in args.iter_mut() {
                changed |= refine_host_expr_types(arg, scope, signatures);
            }
        }
        HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
    }
    changed
}

fn list_item_type(ty: &HostType) -> HostType {
    match ty {
        HostType::List(inner) => (**inner).clone(),
        _ => HostType::Unknown,
    }
}

fn callback_params_mut(callback: &mut HostCallback) -> &mut [HostParam] {
    match &mut callback.kind {
        HostCallbackKind::Named { params, .. } | HostCallbackKind::Inline { params, .. } => params,
    }
}

fn specialize_host_callback_types(
    callback: &mut HostCallback,
    param_tys: &[HostType],
    ret_ty: &HostType,
) -> bool {
    let mut changed = false;
    for (param, inferred) in callback_params_mut(callback)
        .iter_mut()
        .zip(param_tys.iter())
    {
        if host_type_has_unknown(&param.ty) && !host_type_has_unknown(inferred) {
            param.ty = inferred.clone();
            changed = true;
        }
    }
    if host_type_has_unknown(&callback.ret_ty) && !host_type_has_unknown(ret_ty) {
        callback.ret_ty = ret_ty.clone();
        changed = true;
    }
    changed
}

fn host_type_has_unknown(ty: &HostType) -> bool {
    match ty {
        HostType::Unknown => true,
        HostType::Fn(params, ret) => {
            params.iter().any(host_type_has_unknown) || host_type_has_unknown(ret)
        }
        HostType::List(inner) | HostType::Option(inner) => host_type_has_unknown(inner),
        HostType::Dict(key, value) => host_type_has_unknown(key) || host_type_has_unknown(value),
        HostType::Tuple(items) => items.iter().any(host_type_has_unknown),
        _ => false,
    }
}

fn refine_host_callback_types(
    callback: &mut HostCallback,
    scope: &mut HashMap<String, HostType>,
    signatures: &HashMap<String, (Vec<HostType>, HostType)>,
) -> bool {
    match &mut callback.kind {
        HostCallbackKind::Inline { params, body } => {
            let mut callback_scope = scope.clone();
            for param in params.iter() {
                callback_scope.insert(param.name.clone(), param.ty.clone());
            }
            let mut changed = refine_host_expr_types(body, &mut callback_scope, signatures);
            let inferred = host_expr_type(body);
            if !host_type_has_unknown(&inferred) && callback.ret_ty != inferred {
                callback.ret_ty = inferred;
                changed = true;
            }
            changed
        }
        HostCallbackKind::Named { .. } => false,
    }
}

fn should_keep_tensor_expr_in_host_lane(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    if matches!(tag(list), Some("tuple-get" | "if" | "let" | "match")) {
        return true;
    }
    if tag(list) != Some("app") {
        return false;
    }
    let Some(callee) = children(list).first().and_then(as_list) else {
        return false;
    };
    if tag(callee) != Some("var") {
        return false;
    }
    let name = children(callee).first().and_then(symbol_name);
    matches!(
        name,
        Some(
            "copy"
                | "reshape"
                | "to_tensor"
                | "scalar_to_tensor"
                | "pad_sequences"
                | "pad_sequences_to"
                | "concat"
                | "split"
                | "gather"
                | "scatter"
                | "where"
                | "cumsum"
                | "map"
                | "filter"
                | "fold"
                | "scan"
                | "partition"
                | "flat_map"
                | "sort"
                | "diagonal"
                | "trace"
                | "clamp"
                | "einsum"
        )
    )
}

fn lower_match_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let scrutinee = lower_host_expr(&kids[0], program, scope, tensor_helpers);
    let scrutinee_ty = host_expr_type(&scrutinee);
    if matches!(
        scrutinee_ty,
        HostType::Int64 | HostType::Float64 | HostType::Bool | HostType::String
    ) {
        return lower_literal_match_host_expr(
            list,
            program,
            scope,
            tensor_helpers,
            scrutinee,
            scrutinee_ty,
        );
    }
    let mut bind_name = "value".to_string();
    let mut some_expr = HostExpr::new(HostExprKind::Unit);
    let mut none_expr = HostExpr::new(HostExprKind::Unit);
    let mut generic_arms = Vec::new();
    let mut generic_default = None;

    for arm in kids.iter().skip(1) {
        let Some(arm_list) = as_list(arm) else {
            continue;
        };
        if tag(arm_list) != Some("arm") {
            continue;
        }
        let arm_kids = children(arm_list);
        let Some(pattern) = arm_kids.first().and_then(as_list) else {
            continue;
        };
        if tag(pattern) == Some("pat-wild") {
            generic_default = Some(Box::new(lower_host_expr(
                &arm_kids[2],
                program,
                scope,
                tensor_helpers,
            )));
            continue;
        }
        if let Some("pat-ctor" | "pat-record") = tag(pattern) {
            let ctor = children(pattern).first().and_then(symbol_name);
            match ctor {
                Some("Some") => {
                    if let Some(bound) = children(pattern).get(1).and_then(as_list)
                        && tag(bound) == Some("pat-var")
                        && let Some(name) = children(bound).first().and_then(symbol_name)
                    {
                        bind_name = name.to_string();
                    }
                    let mut scoped = scope.clone();
                    scoped.insert(bind_name.clone(), option_inner_type(&scrutinee));
                    some_expr = lower_host_expr(&arm_kids[2], program, &scoped, tensor_helpers);
                }
                Some("None") => {
                    none_expr = lower_host_expr(&arm_kids[2], program, scope, tensor_helpers);
                }
                Some(ctor_name) => {
                    let ctor_fields =
                        lookup_adt_ctor_details_for_type(program, ctor_name, Some(&scrutinee_ty))
                            .map(|(_, fields)| fields)
                            .or_else(|| {
                                program
                                    .type_env()
                                    .get(ctor_name)
                                    .and_then(parse_fn_type_expr)
                                    .map(|(args, _)| {
                                        args.into_iter()
                                            .map(|ty| HostAdtField { name: None, ty })
                                            .collect::<Vec<_>>()
                                    })
                            })
                            .unwrap_or_default();
                    let mut scoped = scope.clone();
                    let mut bindings = Vec::new();
                    for (field_index, subpat, field_ty) in
                        pattern_field_bindings(pattern, &ctor_fields)
                    {
                        let Some(subpat_list) = as_list(subpat) else {
                            continue;
                        };
                        if tag(subpat_list) != Some("pat-var") {
                            continue;
                        }
                        let Some(name) = children(subpat_list).first().and_then(symbol_name) else {
                            continue;
                        };
                        let ty = field_ty
                            .or_else(|| expr_type(subpat))
                            .unwrap_or(HostType::Unknown);
                        scoped.insert(name.to_string(), ty.clone());
                        bindings.push(HostPatternBinding {
                            name: name.to_string(),
                            ty,
                            field_index,
                        });
                    }
                    generic_arms.push(HostMatchArm {
                        ctor: ctor_name.to_string(),
                        bindings,
                        expr: lower_host_expr(&arm_kids[2], program, &scoped, tensor_helpers),
                    });
                }
                None => {}
            }
        }
    }

    let ty = {
        let explicit = expr_host_type(
            &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
            program,
            scope,
        );
        if explicit == HostType::Unknown {
            // For ADT matches with multiple arms, Some/None exprs aren't set —
            // the arm bodies live in `generic_arms` / `generic_default`. Fall
            // back through those first so the match's result type reflects the
            // arms' actual shape. Otherwise Unit propagates and the match
            // target is emitted as `int`, which is the wrong C type for any
            // pointer-valued arm (regression hit by Coral groupby's
            // `next = match agg_spec { ... }` ADT match).
            let arm_ty = generic_arms
                .iter()
                .map(|arm| host_expr_type(&arm.expr))
                .find(|ty| *ty != HostType::Unknown)
                .or_else(|| {
                    generic_default
                        .as_deref()
                        .map(host_expr_type)
                        .filter(|ty| *ty != HostType::Unknown)
                });
            if let Some(ty) = arm_ty {
                ty
            } else {
                let some_ty = host_expr_type(&some_expr);
                if some_ty == HostType::Unknown {
                    host_expr_type(&none_expr)
                } else {
                    some_ty
                }
            }
        } else {
            explicit
        }
    };

    if matches!(scrutinee_ty, HostType::Adt(_, _)) {
        return HostExpr::new(HostExprKind::MatchAdt {
            scrutinee: Box::new(scrutinee),
            arms: generic_arms,
            default_expr: generic_default,
            ty,
        });
    }

    HostExpr::new(HostExprKind::MatchOption {
        scrutinee: Box::new(scrutinee),
        bind_name,
        some_expr: Box::new(some_expr),
        none_expr: Box::new(none_expr),
        ty,
    })
}

fn lower_literal_match_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
    scrutinee: HostExpr,
    scrutinee_ty: HostType,
) -> HostExpr {
    let kids = children(list);
    let mut literal_arms = Vec::new();
    let mut default_expr = HostExpr::new(HostExprKind::Unit);
    for arm in kids.iter().skip(1) {
        let Some(arm_list) = as_list(arm) else {
            continue;
        };
        if tag(arm_list) != Some("arm") {
            continue;
        }
        let arm_kids = children(arm_list);
        let Some(pattern) = arm_kids.first().and_then(as_list) else {
            continue;
        };
        let body = lower_host_expr(&arm_kids[2], program, scope, tensor_helpers);
        match tag(pattern) {
            Some("pat-wild") => default_expr = body,
            Some("pat-lit") => {
                if let Some(lit) = children(pattern)
                    .first()
                    .and_then(|expr| host_literal_expr(expr, &scrutinee_ty))
                {
                    literal_arms.push((lit, body));
                }
            }
            _ => {}
        }
    }

    let explicit = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    let mut body = default_expr;
    let result_ty = if explicit == HostType::Unknown {
        host_expr_type(&body)
    } else {
        explicit
    };
    for (lit, arm_expr) in literal_arms.into_iter().rev() {
        body = HostExpr::new(HostExprKind::If {
            cond: Box::new(HostExpr::new(HostExprKind::Builtin {
                name: "eq".to_string(),
                args: vec![scrutinee.clone(), lit],
                ty: HostType::Bool,
            })),
            then_expr: Box::new(arm_expr),
            else_expr: Box::new(body),
            ty: result_ty.clone(),
        });
    }
    body
}

fn host_literal_expr(expr: &Expr, expected_ty: &HostType) -> Option<HostExpr> {
    match (expr, expected_ty) {
        (Expr::Atom(Atom::Int(value), _), HostType::Int64) => {
            Some(HostExpr::new(HostExprKind::Int(*value)))
        }
        (Expr::Atom(Atom::Float(value), _), HostType::Float64) => {
            Some(HostExpr::new(HostExprKind::Float(*value)))
        }
        (Expr::Atom(Atom::Bool(value), _), HostType::Bool) => {
            Some(HostExpr::new(HostExprKind::Bool(*value)))
        }
        (Expr::Atom(Atom::Str(value), _), HostType::String) => {
            Some(HostExpr::new(HostExprKind::String(value.clone())))
        }
        (Expr::Atom(Atom::Int(value), _), _) => Some(HostExpr::new(HostExprKind::Int(*value))),
        (Expr::Atom(Atom::Float(value), _), _) => Some(HostExpr::new(HostExprKind::Float(*value))),
        (Expr::Atom(Atom::Bool(value), _), _) => Some(HostExpr::new(HostExprKind::Bool(*value))),
        (Expr::Atom(Atom::Str(value), _), _) => {
            Some(HostExpr::new(HostExprKind::String(value.clone())))
        }
        _ => None,
    }
}

fn lower_record_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let ctor = kids
        .first()
        .and_then(symbol_name)
        .unwrap_or("_")
        .to_string();
    let ctor_info = lookup_adt_ctor_details(program, &ctor);
    let mut supplied = HashMap::new();
    for field in kids.iter().skip(1) {
        let Some(kv_list) = as_list(field) else {
            continue;
        };
        if tag(kv_list) != Some("kv") {
            continue;
        }
        let kv_kids = children(kv_list);
        let Some(name) = kv_kids.first().and_then(symbol_name) else {
            continue;
        };
        let value = kv_kids
            .get(1)
            .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers))
            .unwrap_or(HostExpr::new(HostExprKind::Unit));
        supplied.insert(name.to_string(), value);
    }
    let fields = ctor_info
        .as_ref()
        .map(|(_, declared)| {
            declared
                .iter()
                .map(|field| {
                    let value = field
                        .name
                        .as_ref()
                        .and_then(|name| supplied.remove(name))
                        .unwrap_or(HostExpr::new(HostExprKind::Unit));
                    if host_expr_type(&value) == HostType::Unknown {
                        force_host_expr_type(value, field.ty.clone())
                    } else {
                        value
                    }
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    HostExpr::new(HostExprKind::AdtConstruct {
        ctor,
        fields,
        ty: if explicit_ty != HostType::Unknown {
            explicit_ty
        } else {
            ctor_info
                .map(|(adt_name, _)| HostType::Adt(adt_name, Vec::new()))
                .unwrap_or(HostType::Unknown)
        },
    })
}

fn lower_access_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let base = kids
        .first()
        .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers))
        .unwrap_or(HostExpr::new(HostExprKind::Unit));
    let field_name = kids.get(1).and_then(symbol_name).unwrap_or("");
    let (field_index, field_ty) =
        lookup_access_field(program, &base, field_name).unwrap_or((0, HostType::Unknown));
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    HostExpr::new(HostExprKind::AdtFieldAccess {
        base: Box::new(base),
        field_index,
        ty: if explicit_ty != HostType::Unknown {
            explicit_ty
        } else {
            field_ty
        },
    })
}

fn pattern_field_bindings<'a>(
    pattern: &'a List,
    ctor_fields: &'a [HostAdtField],
) -> Vec<(usize, &'a Expr, Option<HostType>)> {
    match tag(pattern) {
        Some("pat-record") => children(pattern)
            .iter()
            .skip(1)
            .filter_map(|kv_expr| {
                let kv_list = as_list(kv_expr)?;
                if tag(kv_list) != Some("kv") {
                    return None;
                }
                let kv_kids = children(kv_list);
                let field_name = kv_kids.first().and_then(symbol_name)?;
                let field_index = ctor_fields
                    .iter()
                    .position(|field| field.name.as_deref() == Some(field_name))?;
                Some((
                    field_index,
                    kv_kids.get(1)?,
                    ctor_fields.get(field_index).map(|field| field.ty.clone()),
                ))
            })
            .collect(),
        _ => children(pattern)
            .iter()
            .skip(1)
            .enumerate()
            .map(|(field_index, subpat)| {
                (
                    field_index,
                    subpat,
                    ctor_fields.get(field_index).map(|field| field.ty.clone()),
                )
            })
            .collect(),
    }
}

fn lower_tuple_get_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let kids = children(list);
    let tuple_expr = kids
        .first()
        .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers));
    let index_expr = kids
        .get(1)
        .map(|expr| lower_host_expr(expr, program, scope, tensor_helpers));
    let args = tuple_expr.into_iter().chain(index_expr).collect::<Vec<_>>();
    let explicit_ty = expr_host_type(
        &Expr::List(list.clone(), chelis_deep::Span::new(0, 0)),
        program,
        scope,
    );
    let ty = if explicit_ty == HostType::Unknown {
        infer_builtin_host_type("tuple-get", &args).unwrap_or(HostType::Unknown)
    } else {
        explicit_ty
    };
    HostExpr::new(HostExprKind::Builtin {
        name: "tuple-get".to_string(),
        args,
        ty,
    })
}

fn lower_app_host_expr(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> HostExpr {
    let app_expr = Expr::List(list.clone(), chelis_deep::Span::new(0, 0));
    let kids = children(list);
    let name = kids
        .first()
        .and_then(as_list)
        .and_then(|inner| {
            if tag(inner) == Some("var") {
                children(inner).first().and_then(symbol_name)
            } else {
                None
            }
        })
        .unwrap_or("call")
        .to_string();
    let fn_sig = scope
        .get(&name)
        .and_then(host_fn_signature)
        .or_else(|| lookup_declared_fn_type(program, &name))
        .or_else(|| kids.first().and_then(expr_fn_type));
    let explicit_ty = expr_host_type(&app_expr, program, scope);
    let ctor_info = lookup_adt_ctor(program, &name);
    let inferred_ret_ty = fn_sig
        .as_ref()
        .map(|(_, ret_ty)| ret_ty.clone())
        .unwrap_or(HostType::Unknown);
    if name == "Cons" && kids.len() == 3 {
        let expr = Expr::List(list.clone(), chelis_deep::Span::new(0, 0));
        if let Some(items) = lower_list_literal_items(&expr, program, scope, tensor_helpers) {
            let ty = expr_host_type(&expr, program, scope);
            let ty = if ty == HostType::Unknown {
                HostType::List(Box::new(
                    items
                        .first()
                        .map(host_expr_type)
                        .unwrap_or(HostType::Unknown),
                ))
            } else {
                ty
            };
            return HostExpr::new(HostExprKind::List(items, ty));
        }
    }
    if name == "Some" && kids.len() == 2 {
        let arg = lower_host_expr(&kids[1], program, scope, tensor_helpers);
        return HostExpr::new(HostExprKind::Builtin {
            name,
            args: vec![arg.clone()],
            ty: if explicit_ty != HostType::Unknown {
                explicit_ty
            } else {
                HostType::Option(Box::new(host_expr_type(&arg)))
            },
        });
    }
    if name == "map"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            HostType::List(Box::new(callback.ret_ty.clone()))
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Map {
            callback,
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "filter"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            host_expr_type(&list_expr)
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Filter {
            callback,
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "fold"
        && kids.len() == 4
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let init_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let list_expr = lower_host_expr(&kids[3], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            host_expr_type(&init_expr)
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Fold {
            callback,
            init: Box::new(init_expr),
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "scan"
        && kids.len() == 4
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let init_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let list_expr = lower_host_expr(&kids[3], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            HostType::List(Box::new(host_expr_type(&init_expr)))
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Scan {
            callback,
            init: Box::new(init_expr),
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "partition"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            let list_ty = host_expr_type(&list_expr);
            HostType::Tuple(vec![list_ty.clone(), list_ty])
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::Partition {
            callback,
            list: Box::new(list_expr),
            ty,
        });
    }
    if name == "flat_map"
        && kids.len() == 3
        && let Some(callback) = lower_host_callback(&kids[1], program, scope, tensor_helpers)
    {
        let list_expr = lower_host_expr(&kids[2], program, scope, tensor_helpers);
        let ty = expr_host_type(&app_expr, program, scope);
        let ty = if ty == HostType::Unknown {
            match callback.ret_ty.clone() {
                HostType::List(inner) => HostType::List(inner),
                _ => HostType::Unknown,
            }
        } else {
            ty
        };
        return HostExpr::new(HostExprKind::FlatMap {
            callback,
            list: Box::new(list_expr),
            ty,
        });
    }
    let helper_tensor_ty = expr_tensor_type(&app_expr, program, scope)
        .or_else(|| {
            if let HostType::Tensor(tensor_ty) = &inferred_ret_ty {
                Some(tensor_ty.clone())
            } else if let HostType::Tensor(tensor_ty) = &explicit_ty {
                Some(tensor_ty.clone())
            } else {
                None
            }
        })
        .or_else(|| {
            // When the callee is a `(grad {} fn_arg)` node (not a plain `var`),
            // `infer_app_expr_host_type` returns None because it only handles `var`
            // callees — so `helper_tensor_ty` is None and the tensor-helper path is
            // skipped entirely.  For `grad(named_fn)(x)` the output shape equals the
            // shape of the first differentiable argument `x`, so infer it from there.
            // This lets `try_lower_tensor_helper_call` succeed even when the outer
            // `app` node carries no explicit type annotation.
            let callee = kids.first().and_then(as_list)?;
            if tag(callee) != Some("grad") {
                return None;
            }
            kids.get(1)
                .and_then(|first_arg| expr_tensor_type(first_arg, program, scope))
        });
    let (helper_expr, helper_scope, helper_bindings) =
        hoist_host_lane_tensor_bindings(&app_expr, program, scope, fn_sig.as_ref(), tensor_helpers);
    // Local callable params (e.g. `f` in `def apply(f: fn, x) = f(x)`) are
    // not representable in the tensor-helper DAG — the DAG path would box
    // the fn pointer into `chelis_scalar_tensor_from_f64` and emit C that
    // gcc rejects. Skip both tensor-helper branches and fall through to
    // the generic `HostExpr::new(HostExprKind::Call)` path so the wrapper emits `return f(x);`.
    let callee_is_local_callable = scope
        .get(&name)
        .is_some_and(|ty| matches!(ty, HostType::Fn(_, _)));
    if let Some(tensor_ty) = helper_tensor_ty.clone()
        && !callee_is_local_callable
        && !top_level_fn_needs_host_lane_tensor_lowering(program, &name)
        && !should_keep_tensor_expr_in_host_lane(&app_expr)
        && let Some(tensor_call) = try_lower_tensor_helper_call(
            &helper_expr,
            program,
            &helper_scope,
            tensor_helpers,
            tensor_ty.clone(),
        )
    {
        if helper_bindings.is_empty() {
            return tensor_call;
        }
        let ty = host_expr_type(&tensor_call);
        return HostExpr::new(HostExprKind::Let {
            bindings: helper_bindings,
            body: Box::new(tensor_call),
            ty,
        });
    }
    if let Some(tensor_ty) = helper_tensor_ty
        && !callee_is_local_callable
        && !top_level_fn_needs_host_lane_tensor_lowering(program, &name)
        && !should_keep_tensor_expr_in_host_lane(&app_expr)
        && let Some(specialized) = inline_top_level_host_call(&app_expr, program)
    {
        let pushed = push_inlining(&name);
        let lowered =
            try_lower_tensor_helper_call(&specialized, program, scope, tensor_helpers, tensor_ty);
        if pushed {
            pop_inlining(&name);
        }
        if let Some(tensor_call) = lowered {
            return tensor_call;
        }
    }
    let has_callable_params = fn_sig
        .as_ref()
        .is_some_and(|(params, _)| params.iter().any(|ty| matches!(ty, HostType::Fn(..))));
    let tensor_result = matches!(explicit_ty, HostType::Tensor(_))
        || matches!(inferred_ret_ty, HostType::Tensor(_));
    if has_callable_params
        && tensor_result
        && let Some(specialized) = inline_top_level_host_call(&app_expr, program)
    {
        let pushed = push_inlining(&name);
        let lowered = lower_host_expr(&specialized, program, scope, tensor_helpers);
        if pushed {
            pop_inlining(&name);
        }
        return lowered;
    }
    let args = kids[1..]
        .iter()
        .map(|arg| lower_host_expr(arg, program, scope, tensor_helpers))
        .collect::<Vec<_>>();
    let construct_ty = if let Some((adt_name, _)) = &ctor_info {
        if matches!(explicit_ty, HostType::Adt(_, _)) {
            explicit_ty.clone()
        } else {
            HostType::Adt(adt_name.clone(), Vec::new())
        }
    } else {
        inferred_ret_ty.clone()
    };
    if ctor_info.is_some() && !matches!(name.as_str(), "Some" | "None") {
        return HostExpr::new(HostExprKind::AdtConstruct {
            ctor: name,
            fields: args,
            ty: construct_ty,
        });
    }
    if !BUILTIN_NAMES.contains(&name.as_str())
        && name != "Some"
        && name != "None"
        && fn_sig.is_some()
    {
        // When the explicit metadata type is Unknown OR contains unresolved
        // type variables (decoded as inner `HostType::Unknown`), prefer
        // the inferred return type from the function's declared signature.
        // The metadata can decay to "Tuple([Unknown, Unknown])" when the
        // node-level annotator re-runs inference with a fresh subst that
        // doesn't share the outer pass's tvar bindings.
        let prefer_inferred = host_type_has_unknown(&explicit_ty);
        return HostExpr::new(HostExprKind::Call {
            function: name,
            args,
            arg_tys: fn_sig
                .as_ref()
                .map(|(param_tys, _)| param_tys.clone())
                .unwrap_or_default(),
            ty: if prefer_inferred {
                inferred_ret_ty
            } else {
                explicit_ty
            },
        });
    }
    let ty = if explicit_ty != HostType::Unknown {
        explicit_ty
    } else {
        infer_builtin_host_type(&name, &args).unwrap_or(HostType::Unknown)
    };
    HostExpr::new(HostExprKind::Builtin { name, args, ty })
}

fn inline_top_level_host_call(expr: &Expr, program: &CheckedProgram) -> Option<Expr> {
    let Expr::List(app_list, _span) = expr else {
        return None;
    };
    if tag(app_list) != Some("app") {
        return None;
    }
    let kids = children(app_list);
    let callee_name = kids
        .first()
        .and_then(as_list)
        .filter(|callee| tag(callee) == Some("var"))
        .and_then(|callee| children(callee).first().and_then(symbol_name))?;
    if is_inlining(callee_name) {
        return None;
    }
    let defs = collect_program_defs(program.exprs());
    let body = lookup_program_def(&defs, callee_name)?;
    let Expr::List(fn_list, _) = body else {
        return None;
    };
    if tag(fn_list) != Some("fn") {
        return None;
    }
    let fn_kids = children(fn_list);
    let params_list = fn_kids.first().and_then(as_list)?;
    if tag(params_list) != Some("params") {
        return None;
    }
    let args = kids.get(1..)?;
    if args.len() != children(params_list).len() {
        return None;
    }
    let substitutions = children(params_list)
        .iter()
        .zip(args.iter())
        .filter_map(|(param, arg)| param_name(param).map(|name| (name, arg.clone())))
        .collect::<HashMap<_, _>>();
    Some(inline_local_callable_lets(&substitute_expr(
        fn_kids.get(1)?,
        &substitutions,
        &HashSet::new(),
    )))
}

fn substitute_expr(
    expr: &Expr,
    substitutions: &HashMap<String, Expr>,
    shadowed: &HashSet<String>,
) -> Expr {
    match expr {
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                entries: meta.entries.clone(),
                expr: Box::new(substitute_expr(&meta.expr, substitutions, shadowed)),
            },
            *span,
        ),
        Expr::List(list, _span) if tag(list) == Some("var") => {
            if let Some(name) = children(list).first().and_then(symbol_name)
                && !shadowed.contains(name)
                && let Some(replacement) = substitutions.get(name)
            {
                return replacement.clone();
            }
            expr.clone()
        }
        Expr::List(list, span) if tag(list) == Some("fn") => {
            let kids = children(list);
            let mut next_shadowed = shadowed.clone();
            if let Some(params) = kids.first().and_then(as_list) {
                for param in children(params) {
                    if let Some(name) = param_name(param) {
                        next_shadowed.insert(name);
                    }
                }
            }
            let mut elements = Vec::with_capacity(list.elements.len());
            elements.push(list.elements[0].clone());
            elements.push(list.elements[1].clone());
            if let Some(params) = kids.first() {
                elements.push(params.clone());
            }
            if let Some(body) = kids.get(1) {
                elements.push(substitute_expr(body, substitutions, &next_shadowed));
            }
            Expr::List(List { elements }, *span)
        }
        Expr::List(list, span) if tag(list) == Some("let") => {
            let kids = children(list);
            let mut next_shadowed = shadowed.clone();
            if let Some(bind_list) = kids.first().and_then(as_list)
                && tag(bind_list) == Some("bind")
            {
                let bind_kids = children(bind_list);
                for index in (0..bind_kids.len()).step_by(2) {
                    if let Some(name) = bind_kids.get(index).and_then(symbol_name) {
                        next_shadowed.insert(name.to_string());
                    }
                }
            }
            let elements = list
                .elements
                .iter()
                .map(|child| substitute_expr(child, substitutions, shadowed))
                .collect();
            if kids.len() >= 2 {
                let mut rebuilt = list.elements.clone();
                rebuilt[2] = substitute_expr(&list.elements[2], substitutions, shadowed);
                rebuilt[3] = substitute_expr(&list.elements[3], substitutions, &next_shadowed);
                Expr::List(List { elements: rebuilt }, *span)
            } else {
                Expr::List(List { elements }, *span)
            }
        }
        Expr::List(list, span) => Expr::List(
            List {
                elements: list
                    .elements
                    .iter()
                    .map(|child| substitute_expr(child, substitutions, shadowed))
                    .collect(),
            },
            *span,
        ),
        _ => expr.clone(),
    }
}

fn inline_local_callable_lets(expr: &Expr) -> Expr {
    let Expr::List(list, span) = expr else {
        return expr.clone();
    };
    if tag(list) != Some("let") {
        return Expr::List(
            List {
                elements: list
                    .elements
                    .iter()
                    .map(inline_local_callable_lets)
                    .collect(),
            },
            *span,
        );
    }

    let kids = children(list);
    let Some(bind_list) = kids.first().and_then(as_list) else {
        return expr.clone();
    };
    if tag(bind_list) != Some("bind") {
        return expr.clone();
    }

    let bind_kids = children(bind_list);
    let mut rebuilt_pairs = Vec::<(String, Expr)>::new();
    let mut body = kids
        .get(1)
        .map(inline_local_callable_lets)
        .unwrap_or_else(|| {
            Expr::List(
                List {
                    elements: Vec::new(),
                },
                *span,
            )
        });

    for index in (0..bind_kids.len()).step_by(2).rev() {
        let Some(name) = bind_kids.get(index).and_then(symbol_name) else {
            continue;
        };
        let Some(value) = bind_kids.get(index + 1) else {
            continue;
        };
        let value = inline_local_callable_lets(value);
        if matches!(&value, Expr::List(inner, _) if tag(inner) == Some("fn")) {
            body = substitute_expr(
                &body,
                &HashMap::from([(name.to_string(), value)]),
                &HashSet::new(),
            );
        } else {
            rebuilt_pairs.push((name.to_string(), value));
        }
    }

    if rebuilt_pairs.is_empty() {
        return body;
    }

    rebuilt_pairs.reverse();
    let mut rebuilt_bind = vec![bind_list.elements[0].clone(), bind_list.elements[1].clone()];
    for (name, value) in rebuilt_pairs {
        rebuilt_bind.push(Expr::Atom(Atom::Symbol(name), *span));
        rebuilt_bind.push(value);
    }
    Expr::List(
        List {
            elements: vec![
                list.elements[0].clone(),
                list.elements[1].clone(),
                Expr::List(
                    List {
                        elements: rebuilt_bind,
                    },
                    *span,
                ),
                body,
            ],
        },
        *span,
    )
}

fn expr_needs_host_lane_tensor_lowering(expr: &Expr, program: &CheckedProgram) -> bool {
    let graph = top_level_fn_call_graph(program);
    let recursive = recursive_top_level_fn_names_from_graph(&graph);
    let fn_names = graph.keys().cloned().collect::<HashSet<_>>();
    collect_called_top_level_fns(expr, &fn_names)
        .into_iter()
        .any(|name| call_graph_reaches_any(&graph, &name, &recursive))
}

fn top_level_fn_needs_host_lane_tensor_lowering(program: &CheckedProgram, name: &str) -> bool {
    let graph = top_level_fn_call_graph(program);
    let recursive = recursive_top_level_fn_names_from_graph(&graph);
    call_graph_reaches_any(&graph, name, &recursive)
}

fn top_level_fn_call_graph(program: &CheckedProgram) -> HashMap<String, HashSet<String>> {
    let defs = collect_program_defs(program.exprs());
    let fn_names = defs
        .iter()
        .filter(|(_, body)| matches!(body, Expr::List(list, _) if tag(list) == Some("fn")))
        .map(|(name, _)| name.clone())
        .collect::<HashSet<_>>();

    fn_names
        .iter()
        .map(|name| {
            let callees = defs
                .get(name)
                .map(|body| collect_called_top_level_fns(body, &fn_names))
                .unwrap_or_default();
            (name.clone(), callees)
        })
        .collect()
}

fn recursive_top_level_fn_names_from_graph(
    graph: &HashMap<String, HashSet<String>>,
) -> HashSet<String> {
    graph
        .keys()
        .filter(|name| call_graph_reaches_any(graph, name, &HashSet::from([(*name).clone()])))
        .cloned()
        .collect()
}

fn collect_called_top_level_fns(expr: &Expr, fn_names: &HashSet<String>) -> HashSet<String> {
    let mut out = HashSet::new();
    let mut stack = vec![expr];
    while let Some(current) = stack.pop() {
        match current {
            Expr::MetaExpr(meta, _) => stack.push(&meta.expr),
            Expr::List(list, _) => {
                if tag(list) == Some("app")
                    && let Some(callee) = children(list).first().and_then(as_list)
                    && tag(callee) == Some("var")
                    && let Some(name) = children(callee).first().and_then(symbol_name)
                    && fn_names.contains(name)
                {
                    out.insert(name.to_string());
                }
                stack.extend(children(list).iter());
            }
            _ => {}
        }
    }
    out
}

fn call_graph_reaches_any(
    graph: &HashMap<String, HashSet<String>>,
    start: &str,
    targets: &HashSet<String>,
) -> bool {
    if targets.contains(start) {
        return true;
    }
    let mut visited = HashSet::new();
    let mut stack = graph
        .get(start)
        .into_iter()
        .flat_map(|callees| callees.iter().cloned())
        .collect::<Vec<_>>();
    while let Some(name) = stack.pop() {
        if targets.contains(&name) {
            return true;
        }
        if !visited.insert(name.clone()) {
            continue;
        }
        if let Some(next) = graph.get(&name) {
            stack.extend(next.iter().cloned());
        }
    }
    false
}

fn hoist_host_lane_tensor_bindings(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    fn_sig: Option<&(Vec<HostType>, HostType)>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> (Expr, HashMap<String, HostType>, Vec<HostBinding>) {
    let Expr::List(list, span) = expr else {
        return (expr.clone(), scope.clone(), Vec::new());
    };
    if tag(list) != Some("app") {
        return (expr.clone(), scope.clone(), Vec::new());
    }

    let kids = children(list);
    if kids.is_empty() {
        return (expr.clone(), scope.clone(), Vec::new());
    }

    let mut new_elements = vec![
        list.elements[0].clone(),
        list.elements[1].clone(),
        kids[0].clone(),
    ];
    let mut scoped = scope.clone();
    let mut bindings = Vec::new();

    for (index, arg) in kids.iter().enumerate().skip(1) {
        if should_keep_tensor_expr_in_host_lane(arg) {
            let value = lower_host_expr(arg, program, scope, tensor_helpers);
            let preferred_ty = fn_sig
                .and_then(|(param_tys, _)| param_tys.get(index - 1))
                .cloned()
                .or_else(|| expr_tensor_type(arg, program, scope).map(HostType::Tensor))
                .or_else(|| {
                    let ty = expr_host_type(arg, program, scope);
                    (ty != HostType::Unknown).then_some(ty)
                });
            let ty = preferred_ty.unwrap_or_else(|| host_expr_type(&value));
            let value = force_host_expr_type(value, ty.clone());
            let name = format!("__host_tensor_arg_{index}");
            bindings.push(HostBinding {
                name: name.clone(),
                display_name: None,
                ty: ty.clone(),
                value,
            });
            scoped.insert(name.clone(), ty);
            new_elements.push(Expr::List(
                List {
                    elements: vec![
                        Expr::Atom(Atom::Symbol("var".to_string()), *span),
                        Expr::Map(
                            chelis_deep::ast::MetaMap {
                                entries: Vec::new(),
                            },
                            *span,
                        ),
                        Expr::Atom(Atom::Symbol(name), *span),
                    ],
                },
                *span,
            ));
        } else {
            new_elements.push(arg.clone());
        }
    }

    (
        Expr::List(
            List {
                elements: new_elements,
            },
            *span,
        ),
        scoped,
        bindings,
    )
}

fn host_fn_signature(ty: &HostType) -> Option<(Vec<HostType>, HostType)> {
    match ty {
        HostType::Fn(params, ret) => Some((params.clone(), (**ret).clone())),
        _ => None,
    }
}

fn lower_host_callback(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> Option<HostCallback> {
    match expr {
        Expr::MetaExpr(meta, _) => lower_host_callback(&meta.expr, program, scope, tensor_helpers),
        Expr::List(list, _) if tag(list) == Some("fn") => {
            let kids = children(list);
            let params_list = kids.first().and_then(as_list)?;
            if tag(params_list) != Some("params") {
                return None;
            }
            let (param_tys, ret_ty) = expr_fn_type(expr).unwrap_or((Vec::new(), HostType::Unknown));
            let mut callback_scope = scope.clone();
            let mut params = Vec::new();
            for (index, param) in children(params_list).iter().enumerate() {
                let name = param_name(param)?;
                let ty = param_tys
                    .get(index)
                    .cloned()
                    .filter(|ty| *ty != HostType::Unknown)
                    .or_else(|| param_host_type(param))
                    .unwrap_or(HostType::Unknown);
                callback_scope.insert(name.clone(), ty.clone());
                params.push(HostParam { name, ty });
            }
            let body = lower_host_expr(kids.get(1)?, program, &callback_scope, tensor_helpers);
            let ret_ty = if ret_ty == HostType::Unknown {
                host_expr_type(&body)
            } else {
                ret_ty
            };
            Some(HostCallback {
                kind: HostCallbackKind::Inline {
                    params,
                    body: Box::new(body),
                },
                ret_ty,
            })
        }
        Expr::List(list, _) if tag(list) == Some("var") => {
            let name = children(list).first().and_then(symbol_name)?;
            let (param_tys, ret_ty) = scope
                .get(name)
                .and_then(host_fn_signature)
                .or_else(|| {
                    lookup_declared_host_type(program, name).and_then(|ty| host_fn_signature(&ty))
                })
                .or_else(|| lookup_declared_fn_type(program, name))
                .or_else(|| {
                    lookup_program_def(&collect_program_defs(program.exprs()), name)
                        .and_then(expr_fn_type)
                })?;
            let params = param_tys
                .into_iter()
                .enumerate()
                .map(|(index, ty)| HostParam {
                    name: format!("arg{index}"),
                    ty,
                })
                .collect();
            Some(HostCallback {
                kind: HostCallbackKind::Named {
                    function: name.to_string(),
                    params,
                },
                ret_ty,
            })
        }
        _ => None,
    }
}

fn lower_list_literal_items(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
    tensor_helpers: &mut Vec<HostTensorHelper>,
) -> Option<Vec<HostExpr>> {
    match expr {
        Expr::MetaExpr(meta, _) => {
            lower_list_literal_items(&meta.expr, program, scope, tensor_helpers)
        }
        Expr::List(list, _) if tag(list) == Some("var") => {
            (children(list).first().and_then(symbol_name) == Some("Nil")).then(Vec::new)
        }
        Expr::List(list, _) if tag(list) == Some("app") => {
            let kids = children(list);
            if kids.len() != 3 {
                return None;
            }
            let func_name = kids
                .first()
                .and_then(as_list)
                .and_then(|inner| (tag(inner) == Some("var")).then_some(inner))
                .and_then(|inner| children(inner).first().and_then(symbol_name));
            if func_name != Some("Cons") {
                return None;
            }
            let head = lower_host_expr(&kids[1], program, scope, tensor_helpers);
            let mut tail = lower_list_literal_items(&kids[2], program, scope, tensor_helpers)?;
            tail.insert(0, head);
            Some(tail)
        }
        _ => None,
    }
}

fn tensor_helper_args(
    inputs: &[HostTensorInput],
    scope: &HashMap<String, HostType>,
) -> Vec<HostExpr> {
    inputs
        .iter()
        .map(|input| {
            HostExpr::new(HostExprKind::Var(
                input.name.clone(),
                scope
                    .get(&input.name)
                    .cloned()
                    .unwrap_or_else(|| host_type_from_tensor_input(&input.ty)),
            ))
        })
        .collect()
}

fn tensor_helper_inputs(dag: &crate::Dag) -> Vec<HostTensorInput> {
    let mut seen = HashSet::new();
    dag.nodes()
        .iter()
        .filter_map(|node| match &node.op {
            crate::RiscOp::Load { name } if seen.insert(name.as_str().to_string()) => {
                Some(HostTensorInput {
                    name: name.as_str().to_string(),
                    ty: node.output_type.clone(),
                })
            }
            _ => None,
        })
        .collect()
}

fn remap_tensor_helper_dim_symbols(
    dag: &crate::Dag,
    scope: &HashMap<String, HostType>,
    expected_output: &TensorType,
) -> crate::Dag {
    fn tensor_type_has_synthetic_dims(tensor_ty: &TensorType) -> bool {
        tensor_ty.dims.iter().any(|dim| {
            matches!(dim, crate::dag::DimInfo::Named(name, None) if {
                let mut chars = name.chars();
                matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
            })
        })
    }

    let formal_inputs = tensor_helper_inputs(dag);
    let mut actual_inputs = formal_inputs
        .iter()
        .map(|input| match scope.get(&input.name) {
            Some(HostType::Tensor(actual)) => actual.clone(),
            _ => input.ty.clone(),
        })
        .collect::<Vec<_>>();
    let mut formal_params = formal_inputs
        .iter()
        .map(|input| input.ty.clone())
        .collect::<Vec<_>>();
    if let Some(root) = dag.roots().first().and_then(|id| dag.get(*id)) {
        formal_params.push(root.output_type.clone());
        let actual_output = if tensor_type_has_synthetic_dims(expected_output) {
            match root.op {
                crate::dag::RiscOp::Permute { ref axes } => root
                    .inputs
                    .first()
                    .and_then(|id| dag.get(*id))
                    .map(|node| {
                        let mut output = node.output_type.clone();
                        output.dims = axes
                            .iter()
                            .filter_map(|axis| node.output_type.dims.get(*axis).cloned())
                            .collect();
                        output
                    })
                    .unwrap_or_else(|| expected_output.clone()),
                crate::dag::RiscOp::UniformLike { .. } | crate::dag::RiscOp::Dropout { .. } => root
                    .inputs
                    .first()
                    .and_then(|id| dag.get(*id))
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| expected_output.clone()),
                _ => expected_output.clone(),
            }
        } else {
            expected_output.clone()
        };
        actual_inputs.push(actual_output);
    }
    let remapped = crate::lower::remap_tensor_dim_symbols(dag, &formal_params, &actual_inputs);
    actualize_tensor_helper_types(&remapped, scope)
}

fn actualize_tensor_helper_types(
    dag: &crate::Dag,
    scope: &HashMap<String, HostType>,
) -> crate::Dag {
    fn inferred_load_type(
        name: &str,
        scope: &HashMap<String, HostType>,
        fallback: &TensorType,
    ) -> TensorType {
        match scope.get(name) {
            Some(HostType::Tensor(actual)) => actual.clone(),
            _ => fallback.clone(),
        }
    }

    fn precision_like(input: &TensorType, precision: chelis_types::types::Prim) -> TensorType {
        TensorType {
            dims: input.dims.clone(),
            precision,
        }
    }

    fn synthetic_dims(tensor_ty: &TensorType) -> bool {
        tensor_ty.dims.iter().any(|dim| {
            matches!(dim, crate::dag::DimInfo::Named(name, None) if {
                let mut chars = name.chars();
                matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
            })
        })
    }

    let mut inferred = HashMap::<crate::dag::NodeId, TensorType>::new();
    let mut uses = HashMap::<crate::dag::NodeId, Vec<crate::dag::NodeId>>::new();
    for node in dag.nodes() {
        for input in &node.inputs {
            uses.entry(*input).or_default().push(node.id);
        }
    }
    for node in dag.nodes() {
        let actual = match &node.op {
            crate::dag::RiscOp::Load { name } => {
                Some(inferred_load_type(name.as_str(), scope, &node.output_type))
            }
            crate::dag::RiscOp::Add
            | crate::dag::RiscOp::Mul
            | crate::dag::RiscOp::CmpLt
            | crate::dag::RiscOp::MaxElem
            | crate::dag::RiscOp::Neg
            | crate::dag::RiscOp::Exp
            | crate::dag::RiscOp::Log
            | crate::dag::RiscOp::Sin
            | crate::dag::RiscOp::Sqrt
            | crate::dag::RiscOp::UniformLike { .. }
            | crate::dag::RiscOp::Dropout { .. }
            | crate::dag::RiscOp::Realize
            | crate::dag::RiscOp::Cast { .. } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| precision_like(input, node.output_type.precision)),
            crate::dag::RiscOp::Sum { axis }
            | crate::dag::RiscOp::MaxReduce { axis }
            | crate::dag::RiscOp::MinReduce { axis }
            | crate::dag::RiscOp::ProdReduce { axis }
            | crate::dag::RiscOp::Argmax { axis }
            | crate::dag::RiscOp::Argmin { axis } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| TensorType {
                    dims: input
                        .dims
                        .iter()
                        .enumerate()
                        .filter_map(|(index, dim)| (index != *axis).then_some(dim.clone()))
                        .collect(),
                    precision: node.output_type.precision,
                }),
            crate::dag::RiscOp::Permute { axes } => node
                .inputs
                .first()
                .and_then(|id| inferred.get(id))
                .map(|input| TensorType {
                    dims: axes
                        .iter()
                        .filter_map(|axis| input.dims.get(*axis).cloned())
                        .collect(),
                    precision: node.output_type.precision,
                }),
            _ => None,
        }
        .or_else(|| {
            node.reusable_input
                .and_then(|id| inferred.get(&id))
                .filter(|input| input.dims.len() == node.output_type.dims.len())
                .map(|input| precision_like(input, node.output_type.precision))
        });
        if let Some(actual) = actual {
            inferred.insert(node.id, actual);
        }
    }

    loop {
        let mut changed = false;
        for node in dag.nodes() {
            if !matches!(node.op, crate::dag::RiscOp::Expand { .. })
                || !synthetic_dims(&node.output_type)
            {
                continue;
            }
            let Some(actual) = uses.get(&node.id).and_then(|user_ids| {
                user_ids
                    .iter()
                    .filter_map(|user_id| inferred.get(user_id))
                    .find(|user_ty| user_ty.dims.len() == node.output_type.dims.len())
                    .cloned()
            }) else {
                continue;
            };
            let entry = inferred
                .entry(node.id)
                .or_insert_with(|| node.output_type.clone());
            if entry.dims != actual.dims {
                *entry = TensorType {
                    dims: actual.dims,
                    precision: node.output_type.precision,
                };
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    let mut actualized = dag.clone();
    let node_ids = actualized
        .nodes()
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    for id in node_ids {
        let Some(node) = actualized.get(id).cloned() else {
            continue;
        };
        let Some(actual) = inferred.get(&id) else {
            continue;
        };
        if !synthetic_dims(&node.output_type) || node.output_type.dims.len() != actual.dims.len() {
            continue;
        }
        actualized.replace_node(id, node.op, node.inputs, actual.clone());
        if let Some(reusable_input) = node.reusable_input {
            actualized.set_reusable_input(id, reusable_input);
        }
    }
    actualized
}

fn collect_program_defs(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut defs = HashMap::new();
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) == Some("def")
            && let (Some(name), Some(body)) = (
                children(list).first().and_then(symbol_name),
                children(list).get(1),
            )
        {
            defs.insert(name.to_string(), body.clone());
        }
    }
    defs
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

fn collect_tensor_scope(scope: &HashMap<String, HostType>) -> HashMap<String, TensorType> {
    scope
        .iter()
        .filter_map(|(name, ty)| {
            tensor_type_from_host_input(ty).map(|tensor| (name.clone(), tensor))
        })
        .collect()
}

fn tensor_type_from_host_input(ty: &HostType) -> Option<TensorType> {
    match ty {
        HostType::Tensor(tensor) => Some(tensor.clone()),
        HostType::Float64 => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::F32,
        }),
        HostType::Int64 => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Int64,
        }),
        HostType::Bool => Some(TensorType {
            dims: vec![],
            precision: chelis_types::types::Prim::Bool,
        }),
        _ => None,
    }
}

fn host_type_from_tensor_input(ty: &TensorType) -> HostType {
    if ty.dims.is_empty() {
        match ty.precision {
            chelis_types::types::Prim::Bool => HostType::Bool,
            chelis_types::types::Prim::Int8
            | chelis_types::types::Prim::Int32
            | chelis_types::types::Prim::Int64 => HostType::Int64,
            _ => HostType::Float64,
        }
    } else {
        HostType::Tensor(ty.clone())
    }
}

fn expr_host_type(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
) -> HostType {
    match expr {
        Expr::Atom(Atom::Int(_), _) => HostType::Int64,
        Expr::Atom(Atom::Float(_), _) => HostType::Float64,
        Expr::Atom(Atom::Bool(_), _) => HostType::Bool,
        Expr::Atom(Atom::Str(_), _) => HostType::String,
        Expr::List(list, _) if tag(list) == Some("var") => children(list)
            .first()
            .and_then(symbol_name)
            .and_then(|name| {
                expr_type(expr)
                    .filter(|ty| *ty != HostType::Unknown)
                    .or_else(|| {
                        scope
                            .get(name)
                            .cloned()
                            .or_else(|| lookup_declared_host_type(program, name))
                    })
            })
            .unwrap_or(HostType::Unknown),
        Expr::List(list, _) if tag(list) == Some("app") => {
            let explicit = expr_type(expr).unwrap_or(HostType::Unknown);
            if app_expr_needs_inferred_type(&explicit) {
                let inferred =
                    infer_app_expr_host_type(list, program, scope).unwrap_or(HostType::Unknown);
                if should_prefer_inferred_app_type(&explicit, &inferred) {
                    inferred
                } else if explicit != HostType::Unknown {
                    explicit
                } else {
                    inferred
                }
            } else {
                explicit
            }
        }
        _ => expr_type(expr).unwrap_or(HostType::Unknown),
    }
}

fn app_expr_needs_inferred_type(explicit: &HostType) -> bool {
    explicit == &HostType::Unknown || host_type_has_synthetic_tensor_dims(explicit)
}

fn infer_app_expr_host_type(
    list: &List,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
) -> Option<HostType> {
    let kids = children(list);
    let callee = kids.first().and_then(as_list)?;
    if tag(callee) != Some("var") {
        return None;
    }
    let name = children(callee).first().and_then(symbol_name)?;
    if !BUILTIN_NAMES.contains(&name) {
        return lookup_declared_fn_type(program, name).map(|(_, ret)| ret);
    }
    if name == "einsum" {
        let equation = match kids.get(1) {
            Some(Expr::Atom(Atom::Str(value), _)) => value.as_str(),
            _ => return Some(HostType::Unknown),
        };
        let tensors = kids[2..]
            .iter()
            .map(|arg| match expr_host_type(arg, program, scope) {
                HostType::Tensor(tensor_ty) => Some(tensor_ty),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?;
        return infer_einsum_tensor_type(equation, &tensors).map(HostType::Tensor);
    }
    let arg_tys = kids[1..]
        .iter()
        .map(|arg| expr_host_type(arg, program, scope))
        .collect::<Vec<_>>();
    infer_builtin_host_type_from_arg_tys(name, &arg_tys)
}

fn should_prefer_inferred_app_type(explicit: &HostType, inferred: &HostType) -> bool {
    explicit == &HostType::Unknown
        || matches!(
            (explicit, inferred),
            (HostType::Tensor(_), HostType::Tensor(_)) if host_type_has_synthetic_tensor_dims(explicit)
                && !host_type_has_synthetic_tensor_dims(inferred)
        )
}

fn host_type_has_synthetic_tensor_dims(ty: &HostType) -> bool {
    fn synthetic_dim_name(name: &str) -> bool {
        let mut chars = name.chars();
        matches!(chars.next(), Some('d')) && chars.all(|ch| ch.is_ascii_digit())
    }

    match ty {
        HostType::Tensor(tensor_ty) => tensor_ty.dims.iter().any(
            |dim| matches!(dim, crate::dag::DimInfo::Named(name, None) if synthetic_dim_name(name)),
        ),
        HostType::Tuple(items) => items.iter().any(host_type_has_synthetic_tensor_dims),
        HostType::List(inner) | HostType::Option(inner) => {
            host_type_has_synthetic_tensor_dims(inner)
        }
        _ => false,
    }
}

fn infer_einsum_tensor_type(equation: &str, tensors: &[TensorType]) -> Option<TensorType> {
    let (inputs, output) = equation.split_once("->")?;
    let input_specs = inputs.split(',').map(str::trim).collect::<Vec<_>>();
    if input_specs.len() != tensors.len() {
        return None;
    }

    let mut labels = HashMap::<char, crate::dag::DimInfo>::new();
    for (spec, tensor) in input_specs.iter().zip(tensors.iter()) {
        let axes = spec
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect::<Vec<_>>();
        if axes.len() != tensor.dims.len() {
            return None;
        }
        for (axis, dim) in axes.into_iter().zip(tensor.dims.iter().cloned()) {
            labels.entry(axis).or_insert(dim);
        }
    }

    let dims = output
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .map(|axis| labels.get(&axis).cloned())
        .collect::<Option<Vec<_>>>()?;
    let precision = tensors
        .first()
        .map(|tensor| tensor.precision)
        .unwrap_or(chelis_types::types::Prim::F32);
    Some(TensorType { dims, precision })
}

fn lookup_type_expr<'a>(type_env: &'a HashMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    type_env.get(name).or_else(|| {
        let mut matches = type_env
            .iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
}

fn lookup_declared_type_expr<'a>(program: &'a CheckedProgram, name: &str) -> Option<&'a Expr> {
    find_top_level_sig_expr(program.exprs(), name)
        .or_else(|| lookup_type_expr(program.type_env(), name))
}

fn lookup_declared_host_type(program: &CheckedProgram, name: &str) -> Option<HostType> {
    lookup_declared_type_expr(program, name).map(parse_host_type)
}

fn lookup_declared_fn_type(
    program: &CheckedProgram,
    name: &str,
) -> Option<(Vec<HostType>, HostType)> {
    lookup_declared_type_expr(program, name).and_then(parse_fn_type_expr)
}

fn lookup_program_def<'a>(defs: &'a HashMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    defs.get(name).or_else(|| {
        let mut matches = defs
            .iter()
            .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
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

fn find_top_level_sig_expr<'a>(exprs: &'a [Expr], name: &str) -> Option<&'a Expr> {
    for expr in top_level_items(exprs) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("defsig") {
            continue;
        }
        let kids = children(list);
        let Some(sig_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if terminal_name_matches(sig_name, name) {
            return kids.get(1);
        }
    }
    None
}

fn expr_tensor_type(
    expr: &Expr,
    program: &CheckedProgram,
    scope: &HashMap<String, HostType>,
) -> Option<TensorType> {
    match expr_host_type(expr, program, scope) {
        HostType::Tensor(ty) => Some(ty),
        _ => None,
    }
}

fn expr_type(expr: &Expr) -> Option<HostType> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let meta = match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta,
        _ => return None,
    };
    meta.entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| parse_host_type(value))
}

fn expr_fn_type(expr: &Expr) -> Option<(Vec<HostType>, HostType)> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let meta = match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta,
        _ => return None,
    };
    meta.entries
        .iter()
        .find(|(key, _)| key == "type")
        .and_then(|(_, value)| parse_fn_type_expr(value))
}

fn parse_fn_type_expr(expr: &Expr) -> Option<(Vec<HostType>, HostType)> {
    let (args, ret) = parse_fn_type_expr_parts(expr)?;
    Some((
        args.iter().map(parse_host_type).collect(),
        parse_host_type(&ret),
    ))
}

fn parse_fn_type_expr_parts(expr: &Expr) -> Option<(Vec<Expr>, Expr)> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if tag(list) != Some("t-fn") {
        return None;
    }
    let kids = children(list);
    let (ret, args) = kids.split_last()?;
    Some((args.to_vec(), ret.clone()))
}

fn parse_host_type(expr: &Expr) -> HostType {
    parse_host_type_with_subst(expr, &HashMap::new())
}

fn parse_host_type_with_subst(expr: &Expr, subst: &HashMap<String, HostType>) -> HostType {
    if let Expr::MetaExpr(meta, _) = expr {
        return parse_host_type_with_subst(&meta.expr, subst);
    }
    let Expr::List(list, _) = expr else {
        return HostType::Unknown;
    };
    match tag(list) {
        Some("t-prim") => match children(list).first().and_then(symbol_name) {
            Some("int64") | Some("int32") => HostType::Int64,
            Some("f64") | Some("f32") => HostType::Float64,
            Some("bool") => HostType::Bool,
            Some("string") => HostType::String,
            _ => HostType::Unknown,
        },
        Some("t-tensor") => HostType::Tensor(crate::lower::tensor_type_from_deep(expr)),
        Some("t-var") => children(list)
            .first()
            .and_then(symbol_name)
            .and_then(|name| subst.get(name).cloned())
            .unwrap_or(HostType::Unknown),
        Some("t-adt") => {
            let kids = children(list);
            match kids.first().and_then(symbol_name) {
                Some("Option") if kids.len() == 2 => {
                    HostType::Option(Box::new(parse_host_type_with_subst(&kids[1], subst)))
                }
                Some("List") if kids.len() == 2 => {
                    HostType::List(Box::new(parse_host_type_with_subst(&kids[1], subst)))
                }
                Some("Dict") if kids.len() == 3 => HostType::Dict(
                    Box::new(parse_host_type_with_subst(&kids[1], subst)),
                    Box::new(parse_host_type_with_subst(&kids[2], subst)),
                ),
                Some("MappedFile") if kids.len() == 1 => HostType::MappedFile,
                Some(name) => HostType::Adt(
                    name.to_string(),
                    kids.iter()
                        .skip(1)
                        .map(|kid| parse_host_type_with_subst(kid, subst))
                        .collect(),
                ),
                _ => HostType::Unknown,
            }
        }
        Some("t-tuple") => HostType::Tuple(
            children(list)
                .iter()
                .map(|kid| parse_host_type_with_subst(kid, subst))
                .collect(),
        ),
        Some("t-fn") => {
            let kids = children(list);
            match kids.split_last() {
                Some((ret, args)) => HostType::Fn(
                    args.iter()
                        .map(|kid| parse_host_type_with_subst(kid, subst))
                        .collect(),
                    Box::new(parse_host_type_with_subst(ret, subst)),
                ),
                None => HostType::Unknown,
            }
        }
        Some("t-unit") => HostType::Unit,
        _ => HostType::Unknown,
    }
}

fn option_inner_type(expr: &HostExpr) -> HostType {
    match host_expr_type(expr) {
        HostType::Option(inner) => (*inner).clone(),
        _ => HostType::Unknown,
    }
}

fn host_expr_type(expr: &HostExpr) -> HostType {
    match &expr.kind {
        HostExprKind::Int(_) => HostType::Int64,
        HostExprKind::Float(_) => HostType::Float64,
        HostExprKind::Bool(_) => HostType::Bool,
        HostExprKind::String(_) => HostType::String,
        HostExprKind::List(_, ty) => ty.clone(),
        HostExprKind::Tuple(_, ty) => ty.clone(),
        HostExprKind::Var(_, ty)
        | HostExprKind::Call { ty, .. }
        | HostExprKind::Builtin { ty, .. }
        | HostExprKind::AdtConstruct { ty, .. }
        | HostExprKind::AdtFieldAccess { ty, .. }
        | HostExprKind::If { ty, .. }
        | HostExprKind::MatchOption { ty, .. }
        | HostExprKind::MatchAdt { ty, .. }
        | HostExprKind::Let { ty, .. }
        | HostExprKind::Map { ty, .. }
        | HostExprKind::Filter { ty, .. }
        | HostExprKind::Fold { ty, .. }
        | HostExprKind::Scan { ty, .. }
        | HostExprKind::Partition { ty, .. }
        | HostExprKind::FlatMap { ty, .. }
        | HostExprKind::TensorCall { ty, .. } => ty.clone(),
        HostExprKind::Unit => HostType::Unit,
    }
}

fn force_host_expr_type(expr: HostExpr, ty: HostType) -> HostExpr {
    let HostExpr {
        kind,
        span_id,
        merged_spans,
    } = expr;
    let new_kind = match kind {
        HostExprKind::Var(name, _) => HostExprKind::Var(name, ty),
        HostExprKind::Call {
            function,
            args,
            arg_tys,
            ..
        } => HostExprKind::Call {
            function,
            args,
            arg_tys,
            ty,
        },
        HostExprKind::Builtin { name, args, .. } => HostExprKind::Builtin { name, args, ty },
        HostExprKind::AdtConstruct { ctor, fields, .. } => {
            HostExprKind::AdtConstruct { ctor, fields, ty }
        }
        HostExprKind::AdtFieldAccess {
            base, field_index, ..
        } => HostExprKind::AdtFieldAccess {
            base,
            field_index,
            ty,
        },
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ty,
        },
        HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ..
        } => HostExprKind::MatchOption {
            scrutinee,
            bind_name,
            some_expr,
            none_expr,
            ty,
        },
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ty,
        },
        HostExprKind::Let { bindings, body, .. } => HostExprKind::Let { bindings, body, ty },
        HostExprKind::Map { callback, list, .. } => HostExprKind::Map { callback, list, ty },
        HostExprKind::Filter { callback, list, .. } => HostExprKind::Filter { callback, list, ty },
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        } => HostExprKind::Fold {
            callback,
            init,
            list,
            ty,
        },
        HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => HostExprKind::Scan {
            callback,
            init,
            list,
            ty,
        },
        HostExprKind::Partition { callback, list, .. } => {
            HostExprKind::Partition { callback, list, ty }
        }
        HostExprKind::FlatMap { callback, list, .. } => {
            HostExprKind::FlatMap { callback, list, ty }
        }
        HostExprKind::TensorCall { helper, args, .. } => {
            HostExprKind::TensorCall { helper, args, ty }
        }
        other => other,
    };
    HostExpr {
        kind: new_kind,
        span_id,
        merged_spans,
    }
}

fn infer_builtin_host_type(name: &str, args: &[HostExpr]) -> Option<HostType> {
    let arg_tys = args.iter().map(host_expr_type).collect::<Vec<_>>();
    match name {
        "einsum" => {
            let equation = match args.first().map(|e| &e.kind) {
                Some(HostExprKind::String(value)) => value.as_str(),
                _ => return Some(HostType::Unknown),
            };
            let tensors = args[1..]
                .iter()
                .map(|arg| match host_expr_type(arg) {
                    HostType::Tensor(tensor_ty) => Some(tensor_ty),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            infer_einsum_tensor_type(equation, &tensors).map(HostType::Tensor)
        }
        "tuple-get" => match (arg_tys.first(), args.get(1).map(|e| &e.kind)) {
            (Some(HostType::Tuple(items)), Some(HostExprKind::Int(index))) => items
                .get(*index as usize)
                .cloned()
                .or(Some(HostType::Unknown)),
            _ => Some(HostType::Unknown),
        },
        _ => infer_builtin_host_type_from_arg_tys(name, &arg_tys),
    }
}

fn infer_builtin_host_type_from_arg_tys(name: &str, arg_tys: &[HostType]) -> Option<HostType> {
    let tensor_arg = arg_tys.iter().find_map(|ty| match ty {
        HostType::Tensor(tensor_ty) => Some(tensor_ty.clone()),
        _ => None,
    });
    match name {
        "add" | "sub" | "mul" | "div" | "neg" | "exp" | "log" | "sin" | "sqrt" | "relu"
        | "sigmoid" | "max_elem" | "min_elem" | "copy" | "uniform_like" | "dropout" => {
            if let Some(tensor_ty) = tensor_arg {
                Some(HostType::Tensor(tensor_ty))
            } else if arg_tys.iter().any(|ty| matches!(ty, HostType::Float64)) {
                Some(HostType::Float64)
            } else {
                Some(HostType::Int64)
            }
        }
        "mod" | "bitand" | "bitor" | "bitxor" | "shl" | "shr" | "string_len" | "rank" | "shape"
        | "numel" => Some(HostType::Int64),
        "cmplt" => match arg_tys.first() {
            Some(HostType::Tensor(tensor_ty)) => Some(HostType::Tensor(TensorType {
                dims: tensor_ty.dims.clone(),
                precision: chelis_types::types::Prim::Bool,
            })),
            _ => Some(HostType::Bool),
        },
        "reshape" => match arg_tys.first() {
            Some(HostType::Tensor(tensor_ty)) => Some(HostType::Tensor(tensor_ty.clone())),
            _ => Some(HostType::Unknown),
        },
        "lt" | "gt" | "gte" | "lte" | "eq" | "neq" | "and" | "or" | "not" | "string_contains"
        | "string_starts_with" | "string_ends_with" => Some(HostType::Bool),
        "string_concat" | "string_trim" | "string_slice" | "to_string" => Some(HostType::String),
        "to_int" => Some(HostType::Option(Box::new(HostType::Int64))),
        "to_float" => Some(HostType::Option(Box::new(HostType::Float64))),
        "len" => Some(HostType::Int64),
        "index" => match arg_tys.first() {
            Some(HostType::List(inner)) => Some((**inner).clone()),
            _ => Some(HostType::Unknown),
        },
        "append" => arg_tys.first().cloned(),
        "concat" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(HostType::List(inner)), Some(HostType::Int64))
                if matches!(inner.as_ref(), HostType::Tensor(_)) =>
            {
                match inner.as_ref() {
                    HostType::Tensor(tensor_ty) => Some(HostType::Tensor(tensor_ty.clone())),
                    _ => Some(HostType::Unknown),
                }
            }
            (Some(lhs), Some(_)) => Some(lhs.clone()),
            _ => Some(HostType::Unknown),
        },
        "split" => match arg_tys.first() {
            Some(HostType::Tensor(tensor_ty)) => Some(HostType::List(Box::new(HostType::Tensor(
                tensor_ty.clone(),
            )))),
            _ => Some(HostType::Unknown),
        },
        "gather" | "scatter" | "where" | "cumsum" | "diagonal" | "trace" | "clamp" => {
            arg_tys.first().cloned()
        }
        "sort" => match arg_tys.first() {
            Some(HostType::Tensor(tensor_ty)) => Some(HostType::Tuple(vec![
                HostType::Tensor(tensor_ty.clone()),
                HostType::Tensor(TensorType {
                    dims: tensor_ty.dims.clone(),
                    precision: chelis_types::types::Prim::Int64,
                }),
            ])),
            _ => Some(HostType::Unknown),
        },
        "tuple-get" => Some(HostType::Unknown),
        "drop" if arg_tys.len() == 1 => Some(HostType::Unit),
        "take" | "drop" => match arg_tys.first() {
            Some(HostType::List(inner)) => Some(HostType::List(Box::new((**inner).clone()))),
            _ => Some(HostType::Unknown),
        },
        "chunk" => match arg_tys.first() {
            Some(HostType::List(inner)) => Some(HostType::List(Box::new(HostType::List(
                Box::new((**inner).clone()),
            )))),
            _ => Some(HostType::Unknown),
        },
        "range" => Some(HostType::List(Box::new(HostType::Int64))),
        "map" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(_), Some(HostType::List(inner))) => {
                Some(HostType::List(Box::new((**inner).clone())))
            }
            _ => Some(HostType::Unknown),
        },
        "filter" => arg_tys.get(1).cloned(),
        "fold" => arg_tys.get(1).cloned(),
        "scan" => match arg_tys.get(1) {
            Some(init_ty) => Some(HostType::List(Box::new(init_ty.clone()))),
            None => Some(HostType::Unknown),
        },
        "partition" => match arg_tys.get(1) {
            Some(list_ty) => Some(HostType::Tuple(vec![list_ty.clone(), list_ty.clone()])),
            None => Some(HostType::Unknown),
        },
        "flat_map" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(_), Some(HostType::List(_))) => match arg_tys.first() {
                Some(HostType::Unknown) => Some(HostType::Unknown),
                Some(_) => None,
                None => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "flatten" => match arg_tys.first() {
            Some(HostType::List(inner)) => match inner.as_ref() {
                HostType::List(nested) => Some(HostType::List(Box::new((**nested).clone()))),
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "zip" => match (arg_tys.first(), arg_tys.get(1)) {
            (Some(HostType::List(lhs)), Some(HostType::List(rhs))) => {
                Some(HostType::List(Box::new(HostType::Tuple(vec![
                    (**lhs).clone(),
                    (**rhs).clone(),
                ]))))
            }
            _ => Some(HostType::Unknown),
        },
        "enumerate" => match arg_tys.first() {
            Some(HostType::List(inner)) => Some(HostType::List(Box::new(HostType::Tuple(vec![
                HostType::Int64,
                (**inner).clone(),
            ])))),
            _ => Some(HostType::Unknown),
        },
        "dict_of" => match arg_tys.first() {
            Some(HostType::List(inner)) => match &**inner {
                HostType::Tuple(parts) if parts.len() == 2 => Some(HostType::Dict(
                    Box::new(parts[0].clone()),
                    Box::new(parts[1].clone()),
                )),
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "dict_get" => match arg_tys.first() {
            Some(HostType::Dict(_, value)) => Some(HostType::Option(Box::new((**value).clone()))),
            _ => Some(HostType::Unknown),
        },
        "dict_contains" => Some(HostType::Bool),
        "dict_remove" => match arg_tys.first() {
            Some(HostType::Dict(key, value)) => Some(HostType::Dict(
                Box::new((**key).clone()),
                Box::new((**value).clone()),
            )),
            _ => Some(HostType::Unknown),
        },
        "dict_insert" => match arg_tys.first() {
            Some(HostType::Dict(key, value)) => Some(HostType::Dict(
                Box::new((**key).clone()),
                Box::new((**value).clone()),
            )),
            _ => Some(HostType::Unknown),
        },
        "dict_merge" => match (arg_tys.first(), arg_tys.get(1)) {
            (
                Some(HostType::Dict(lhs_key, lhs_value)),
                Some(HostType::Dict(rhs_key, rhs_value)),
            ) if **lhs_key == **rhs_key && **lhs_value == **rhs_value => Some(HostType::Dict(
                Box::new((**lhs_key).clone()),
                Box::new((**lhs_value).clone()),
            )),
            _ => Some(HostType::Unknown),
        },
        "dict_keys" => match arg_tys.first() {
            Some(HostType::Dict(key, _)) => Some(HostType::List(Box::new((**key).clone()))),
            _ => Some(HostType::Unknown),
        },
        "dict_values" => match arg_tys.first() {
            Some(HostType::Dict(_, value)) => Some(HostType::List(Box::new((**value).clone()))),
            _ => Some(HostType::Unknown),
        },
        "dict_entries" => match arg_tys.first() {
            Some(HostType::Dict(key, value)) => {
                Some(HostType::List(Box::new(HostType::Tuple(vec![
                    (**key).clone(),
                    (**value).clone(),
                ]))))
            }
            _ => Some(HostType::Unknown),
        },
        "print" => Some(HostType::Unit),
        "fail" => Some(HostType::Unknown),
        "debug" => arg_tys.first().cloned(),
        "tensor_to_scalar" => match arg_tys.first() {
            Some(HostType::Tensor(tensor)) => Some(match tensor.precision {
                chelis_types::types::Prim::Bool => HostType::Bool,
                chelis_types::types::Prim::Int8
                | chelis_types::types::Prim::Int32
                | chelis_types::types::Prim::Int64 => HostType::Int64,
                _ => HostType::Float64,
            }),
            _ => Some(HostType::Float64),
        },
        "scalar_to_tensor" => match arg_tys.first() {
            Some(HostType::Int64) => Some(HostType::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::Int64,
            })),
            Some(HostType::Bool) => Some(HostType::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::Bool,
            })),
            Some(HostType::Float64) => Some(HostType::Tensor(TensorType {
                dims: vec![],
                precision: chelis_types::types::Prim::F32,
            })),
            _ => None,
        },
        "to_tensor" => match arg_tys.first() {
            Some(HostType::List(inner)) => match &**inner {
                HostType::Int64 => Some(HostType::Tensor(TensorType {
                    dims: vec![crate::dag::DimInfo::Named("list".to_string(), None)],
                    precision: chelis_types::types::Prim::Int64,
                })),
                HostType::Float64 => Some(HostType::Tensor(TensorType {
                    dims: vec![crate::dag::DimInfo::Named("list".to_string(), None)],
                    precision: chelis_types::types::Prim::F32,
                })),
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "to_list" => match arg_tys.first() {
            Some(HostType::Tensor(tensor)) if tensor.dims.len() == 1 => {
                let element_ty = match tensor.precision {
                    chelis_types::types::Prim::Bool => HostType::Bool,
                    chelis_types::types::Prim::Int8
                    | chelis_types::types::Prim::Int32
                    | chelis_types::types::Prim::Int64 => HostType::Int64,
                    chelis_types::types::Prim::F16
                    | chelis_types::types::Prim::Bf16
                    | chelis_types::types::Prim::F32
                    | chelis_types::types::Prim::F64
                    | chelis_types::types::Prim::F8e4m3 => HostType::Float64,
                    _ => HostType::Unknown,
                };
                Some(HostType::List(Box::new(element_ty)))
            }
            Some(HostType::Tensor(_)) => Some(HostType::Unknown),
            _ => Some(HostType::Unknown),
        },
        "pad_sequences" => match arg_tys.first() {
            Some(HostType::List(inner)) => match &**inner {
                HostType::List(nested) => match &**nested {
                    HostType::Int64 => Some(HostType::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::Int64,
                    })),
                    HostType::Float64 => Some(HostType::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::F32,
                    })),
                    _ => Some(HostType::Unknown),
                },
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "pad_sequences_to" => match arg_tys.first() {
            Some(HostType::List(inner)) => match &**inner {
                HostType::List(nested) => match &**nested {
                    HostType::Int64 => Some(HostType::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::Int64,
                    })),
                    HostType::Float64 => Some(HostType::Tensor(TensorType {
                        dims: vec![
                            crate::dag::DimInfo::Named("batch".to_string(), None),
                            crate::dag::DimInfo::Named("seq".to_string(), None),
                        ],
                        precision: chelis_types::types::Prim::F32,
                    })),
                    _ => Some(HostType::Unknown),
                },
                _ => Some(HostType::Unknown),
            },
            _ => Some(HostType::Unknown),
        },
        "read_file" => Some(HostType::String),
        "write_file" => Some(HostType::Unit),
        "read_lines" => Some(HostType::List(Box::new(HostType::String))),
        "read_bytes" => Some(HostType::List(Box::new(HostType::Int64))),
        "file_exists" => Some(HostType::Bool),
        "list_dir" => Some(HostType::List(Box::new(HostType::String))),
        "mmap_file" => Some(HostType::MappedFile),
        "mmap_read" => Some(HostType::List(Box::new(HostType::Int64))),
        "mmap_len" => Some(HostType::Int64),
        _ => None,
    }
}

fn lookup_adt_ctor(program: &CheckedProgram, ctor_name: &str) -> Option<(String, Vec<HostType>)> {
    lookup_adt_ctor_details(program, ctor_name).map(|(adt_name, fields)| {
        (
            adt_name,
            fields.into_iter().map(|field| field.ty).collect::<Vec<_>>(),
        )
    })
}

fn lookup_adt_ctor_details(
    program: &CheckedProgram,
    ctor_name: &str,
) -> Option<(String, Vec<HostAdtField>)> {
    lookup_adt_ctor_details_for_type(program, ctor_name, None)
}

fn lookup_adt_ctor_details_for_type(
    program: &CheckedProgram,
    ctor_name: &str,
    instantiated_ty: Option<&HostType>,
) -> Option<(String, Vec<HostAdtField>)> {
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("deftype") {
            continue;
        }
        let kids = children(list);
        let Some(adt_name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let subst = adt_type_substitution(children(list).get(1), adt_name, instantiated_ty);
        for variant in kids.iter().skip(2) {
            let Some(variant_list) = as_list(variant) else {
                continue;
            };
            if tag(variant_list) != Some("variant") {
                continue;
            }
            let variant_kids = children(variant_list);
            let Some(name) = variant_kids.first().and_then(symbol_name) else {
                continue;
            };
            if !terminal_name_matches(name, ctor_name) {
                continue;
            }
            let mut fields = Vec::new();
            for field in variant_kids.iter().skip(1) {
                if let Some(field_list) = as_list(field)
                    && tag(field_list) == Some("field")
                {
                    let field_kids = children(field_list);
                    if let Some(ty_expr) = field_kids.get(1) {
                        fields.push(HostAdtField {
                            name: field_kids.first().and_then(symbol_name).map(str::to_string),
                            ty: parse_host_type_with_subst(ty_expr, &subst),
                        });
                    }
                } else {
                    fields.push(HostAdtField {
                        name: None,
                        ty: parse_host_type_with_subst(field, &subst),
                    });
                }
            }
            return Some((adt_name.to_string(), fields));
        }
    }
    None
}

fn lookup_access_field(
    program: &CheckedProgram,
    base: &HostExpr,
    field_name: &str,
) -> Option<(usize, HostType)> {
    match &base.kind {
        HostExprKind::AdtConstruct { ctor, .. } => {
            lookup_adt_ctor_details(program, ctor).and_then(|(_, fields)| {
                fields.iter().enumerate().find_map(|(index, field)| {
                    (field.name.as_deref() == Some(field_name)).then_some((index, field.ty.clone()))
                })
            })
        }
        _ => match host_expr_type(base) {
            HostType::Adt(adt_name, args) => {
                lookup_adt_field_on_type(program, &adt_name, &args, field_name)
            }
            _ => None,
        },
    }
}

fn lookup_adt_field_on_type(
    program: &CheckedProgram,
    adt_name: &str,
    args: &[HostType],
    field_name: &str,
) -> Option<(usize, HostType)> {
    let mut found = None;
    for expr in top_level_items(program.exprs()) {
        let Expr::List(list, _) = expr else {
            continue;
        };
        if tag(list) != Some("deftype") {
            continue;
        }
        let kids = children(list);
        if kids.first().and_then(symbol_name) != Some(adt_name) {
            continue;
        }
        let subst = adt_type_substitution(
            children(list).get(1),
            adt_name,
            Some(&HostType::Adt(adt_name.to_string(), args.to_vec())),
        );
        for variant in kids.iter().skip(2) {
            let Some(variant_list) = as_list(variant) else {
                continue;
            };
            if tag(variant_list) != Some("variant") {
                continue;
            }
            for (index, field) in children(variant_list).iter().skip(1).enumerate() {
                let Some(field_list) = as_list(field) else {
                    continue;
                };
                if tag(field_list) != Some("field") {
                    continue;
                }
                if children(field_list).first().and_then(symbol_name) == Some(field_name) {
                    let ty = children(field_list)
                        .get(1)
                        .map(|expr| parse_host_type_with_subst(expr, &subst))
                        .unwrap_or(HostType::Unknown);
                    if let Some(existing) = &found
                        && existing != &(index, ty.clone())
                    {
                        return None;
                    }
                    found = Some((index, ty));
                }
            }
        }
    }
    found
}

fn adt_type_substitution(
    params_expr: Option<&Expr>,
    adt_name: &str,
    instantiated_ty: Option<&HostType>,
) -> HashMap<String, HostType> {
    let params = params_expr
        .and_then(as_list)
        .map(|list| {
            list.elements
                .iter()
                .filter_map(symbol_name)
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let actuals = match instantiated_ty {
        Some(HostType::Adt(name, args)) if terminal_name_matches(name, adt_name) => args.clone(),
        _ => Vec::new(),
    };
    params.into_iter().zip(actuals).collect()
}

fn tag(list: &List) -> Option<&str> {
    list.elements.first().and_then(|expr| match expr {
        Expr::Atom(Atom::Symbol(tag), _) => Some(tag.as_str()),
        _ => None,
    })
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn as_list(expr: &Expr) -> Option<&List> {
    match expr {
        Expr::List(list, _) => Some(list),
        _ => None,
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn param_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
        Expr::MetaExpr(meta, _) => param_name(&meta.expr),
        Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .or_else(|| children(list).first().and_then(symbol_name))
            .map(str::to_string),
        _ => None,
    }
}

fn param_host_type(expr: &Expr) -> Option<HostType> {
    match expr {
        Expr::MetaExpr(meta, _) => expr_type(expr)
            .filter(|ty| *ty != HostType::Unknown)
            .or_else(|| param_host_type(&meta.expr)),
        Expr::List(_, _) => expr_type(expr).filter(|ty| *ty != HostType::Unknown),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_types::types::Prim;

    fn parse_and_check(src: &str) -> CheckedProgram {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let checked = chelis_types::check_phase0e_program(&exprs)
            .unwrap_or_else(|result| panic!("phase 0e check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"))
    }

    #[test]
    fn tensor_helper_hoists_host_lane_tensor_args_with_f32_type() {
        let checked = parse_and_check(
            r#"
                (defsig {}
                  jac_row
                  (t-fn {}
                    (t-fn {}
                      (t-tensor {} (d-var {} n) (t-prim {} f32))
                      (t-prim {} f32)
                      (t-prim {} f32)
                      (t-prim {} f32))
                    (t-tensor {} (d-var {} n) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-tensor {} (d-var {} n) (t-prim {} f32))))
                (def {}
                  jac_row
                  (fn {}
                    (params {}
                      (model {type: (t-fn {}
                                       (t-tensor {} (d-var {} n) (t-prim {} f32))
                                       (t-prim {} f32)
                                       (t-prim {} f32)
                                       (t-prim {} f32))})
                      (theta {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        target
                        (fn {}
                          (params {}
                            (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
                          (app {} (var {} model) (var {} theta_local) (var {} x) (var {} y))))
                      (app {}
                        (grad {wrt: (var {} theta_local)}
                          (var {} target)
                          (lit {type: (t-prim {} int32)} 0))
                        (var {} theta)))))
                (defsig {}
                  lm_model
                  (t-fn {}
                    (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-prim {} f32)))
                (def {}
                  lm_model
                  (fn {}
                    (params {}
                      (theta {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        y_hat
                        (if {}
                          (app {}
                            (var {} lt)
                            (var {} x)
                            (cast {} (lit {type: (t-prim {} f32)} 0.0) (t-prim {} f32)))
                          (app {}
                            (var {} tensor_to_scalar)
                            (app {}
                              (var {} sum)
                              (copy {} (var {} theta))
                              (lit {type: (t-prim {} int32)} 0)))
                          (app {}
                            (var {} add)
                            (app {}
                              (var {} tensor_to_scalar)
                              (app {}
                                (var {} sum)
                                (copy {} (var {} theta))
                                (lit {type: (t-prim {} int32)} 0)))
                            (var {} x))))
                      (app {} (var {} sub) (var {} y) (var {} y_hat)))))
                (def {}
                  out
                  (app {}
                    (var {} jac_row)
                    (var {} lm_model)
                    (app {}
                      (var {} to_tensor)
                      (app {}
                        (var {} Cons)
                        (lit {type: (t-prim {} f32)} 1.0)
                        (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 2.0) (var {} Nil))))
                    (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} f32))
                    (cast {} (lit {type: (t-prim {} f32)} 3.0) (t-prim {} f32))))
            "#,
        );
        let lowered = top_level_lowering_map(checked.exprs(), checked.type_env());
        let host = lower_host_program(&checked, &lowered);
        let out_binding = host
            .globals
            .iter()
            .find(|binding| binding.name == "out")
            .expect("out host binding");
        let (bindings, body) = match &out_binding.value.kind {
            HostExprKind::Let { bindings, body, .. } => (bindings, body.as_ref()),
            other => panic!("expected out to hoist host-lane tensor arg into let, got {other:?}"),
        };
        let theta_binding = bindings
            .iter()
            .find(|binding| binding.name.starts_with("__host_tensor_arg_"))
            .unwrap_or_else(|| {
                panic!(
                    "theta temp binding missing; hoisted bindings were {:?}",
                    bindings
                        .iter()
                        .map(|binding| (&binding.name, &binding.ty))
                        .collect::<Vec<_>>()
                )
            });
        match &theta_binding.ty {
            HostType::Tensor(tensor) => assert_eq!(tensor.precision, Prim::F32),
            other => panic!("expected hoisted theta binding to be tensor-typed, got {other:?}"),
        }
        let helper_index = match &body.kind {
            HostExprKind::TensorCall { helper, .. } => *helper,
            other => panic!("expected hoisted body to call tensor helper, got {other:?}"),
        };
        let helper = host
            .global_tensor_helpers
            .get(helper_index)
            .expect("helper index in range");
        assert!(
            helper
                .inputs
                .iter()
                .any(|input| input.name == theta_binding.name),
            "helper inputs should reference hoisted tensor temp: {:?}",
            helper
                .inputs
                .iter()
                .map(|input| (&input.name, &input.ty))
                .collect::<Vec<_>>()
        );
        assert!(
            helper.inputs.iter().all(|input| input.name != "to_tensor"),
            "helper inputs must not contain raw `to_tensor` load: {:?}",
            helper
                .inputs
                .iter()
                .map(|input| (&input.name, &input.ty))
                .collect::<Vec<_>>()
        );
    }
}
