//! Type inference engine for the Chelis type checker.
//!
//! Walks Deep AST nodes and assigns types using Hindley-Milner inference.

use std::collections::{HashMap, HashSet};

use chelis_deep::Span;
use chelis_deep::ast as deep;

use crate::adt::AdtRegistry;
use crate::builtins;
use crate::context::{TypeEnv, TypeEnvInner};
use crate::env::Env;
use crate::errors::*;
use crate::linearity::LinearityInfo;
use crate::types::*;
use crate::unify::*;

/// Result of running type inference on a program.
#[derive(Debug)]
pub struct InferResult {
    pub errors: Vec<CheckError>,
    pub typed_nodes: usize,
    pub total_nodes: usize,
}

#[derive(Debug, Clone)]
pub struct CheckedProgram {
    annotated_exprs: Vec<deep::Expr>,
    type_env: HashMap<String, deep::Expr>,
    linearity: LinearityInfo,
}

impl CheckedProgram {
    pub fn from_parts(
        annotated_exprs: Vec<deep::Expr>,
        type_env: HashMap<String, deep::Expr>,
    ) -> Self {
        Self {
            annotated_exprs,
            type_env,
            linearity: LinearityInfo::default(),
        }
    }

    pub fn exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn annotated_exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn type_env(&self) -> &HashMap<String, deep::Expr> {
        &self.type_env
    }

    pub fn linearity(&self) -> &LinearityInfo {
        &self.linearity
    }

    pub fn with_linearity(mut self, linearity: LinearityInfo) -> Self {
        self.linearity = linearity;
        self
    }
}

/// Run type inference on a list of top-level Deep expressions.
pub fn infer_program(exprs: &[deep::Expr]) -> InferResult {
    let (mut env, mut vg) = builtins::builtin_env();
    let mut subst = Subst::new();
    let mut adt_reg = AdtRegistry::new();
    builtins::register_prelude_adts(&mut env, &mut vg, &mut adt_reg);
    let mut errors = Vec::new();
    let mut typed_nodes = 0;
    let mut total_nodes = 0;

    // First pass: collect deftype and defsig declarations. Descend through
    // `(module {} name ...)` wrappers so declarations in every idiomatic
    // Surf source (every .ch starts with `module X`) get collected.
    for expr in top_level_decl_items(exprs) {
        collect_declarations(
            expr,
            &mut env,
            &mut vg,
            &mut subst,
            &mut adt_reg,
            &mut errors,
        );
    }

    // Second pass: infer def bodies. Same module-descent rationale as the
    // declaration pass — without it, the entire HM checker is a no-op on
    // module-wrapped programs.
    for expr in top_level_decl_items(exprs) {
        infer_top_level(
            expr,
            &mut env,
            &mut vg,
            &mut subst,
            &adt_reg,
            &mut errors,
            &mut typed_nodes,
            &mut total_nodes,
        );
    }

    // Third pass: reject tensor types whose element precision isn't supported
    // by the Phase 0f backend (f16/bf16/f8e4m3). These would silently get
    // downcast to f32 by the current build targets, violating the "no implicit
    // precision promotion" rule. f64 is supported as of v0.2.3.
    validate_tensor_precisions_in_program(exprs, &mut errors);

    InferResult {
        errors,
        typed_nodes,
        total_nodes,
    }
}

pub fn check_phase0e_program(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    // Compose: empty outer scope, then check exprs as new code against it.
    // This keeps a single source of truth for the Phase 0e check pipeline.
    check_phase0e_with_context(&TypeEnv::empty(), exprs)
}

/// Build a stacked outer-scope context from a library decl list. The
/// library is run through the full Phase 0e pipeline; if any errors are
/// found they are returned to the caller (the context cannot be built
/// from an unchecked library).
///
/// Once built, the returned [`TypeEnv`] can be re-used to type-check
/// many separate "new code" snippets via
/// [`check_phase0e_with_context`]. The library state is `Arc`-shared and
/// never mutated, so concurrent reads are cheap.
pub fn build_type_env_from_library(library_exprs: &[deep::Expr]) -> Result<TypeEnv, InferResult> {
    // Start from the empty (builtins + prelude ADTs) state.
    let empty = TypeEnv::empty();
    let mut state = empty.inner().clone();

    // Library Phase 0e declared-type lookup.
    let library_phase0e = build_phase0e_type_env(library_exprs);

    let mut result = infer_phase0e_program_with_state(
        library_exprs,
        &library_phase0e,
        &mut state,
        /* combined_phase0e_for_validate = */ &library_phase0e,
        /* run_validate_passes_on = */ None,
    );
    validate_phase0e_program(library_exprs, &library_phase0e, &mut result.errors);
    validate_tensor_precisions_in_program(library_exprs, &mut result.errors);
    suppress_unbound_for_cycle_members(library_exprs, &mut result.errors);
    if !result.errors.is_empty() {
        return Err(result);
    }

    // Capture library def names — needed by new-code cycle / unbound
    // suppression to distinguish library refs from new-code refs.
    let mut library_def_names = std::collections::HashSet::new();
    for expr in top_level_decl_items(library_exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            library_def_names.insert(name.to_string());
        }
    }

    // Drain accumulated errors back into the state's storage; they were
    // empty above so this is a no-op, but the call site is symmetric
    // with check_phase0e_with_context.
    let _ = result.errors.drain(..);

    // Build a richer `phase0e_types` by annotating library exprs against
    // the now-populated state and re-extracting type metadata. The raw
    // `library_phase0e` (built from un-annotated source) only catches defs
    // with explicit type annotations; for downstream callers that read
    // `CheckedProgram::type_env()` to resolve cross-context name refs
    // (Phase D effects, Phase E linearity, Phase F lower) we need every
    // library def's inferred function type, not just the explicitly-typed
    // ones. Mirrors the monolithic `check_phase0e_program` flow which
    // calls `annotate_phase0e_program` then `build_phase0e_type_env` on
    // the annotated result.
    let library_annotated: Vec<deep::Expr> = library_exprs
        .iter()
        .map(|e| {
            annotate_expr_with_scope(e, &state.env, &state.var_gen, &state.subst, &state.adt_reg)
        })
        .collect();
    let library_phase0e_annotated = build_phase0e_type_env(&library_annotated);

    Ok(TypeEnv::from_inner(TypeEnvInner {
        env: state.env,
        var_gen: state.var_gen,
        subst: state.subst,
        adt_reg: state.adt_reg,
        phase0e_types: library_phase0e_annotated,
        library_def_names,
    }))
}

/// Type-check `new_exprs` against an outer-scope `context`. New-code
/// bindings shadow but do not consume library bindings; library ADT
/// constructor sets remain visible to new-code `match` exhaustivity
/// checks. The returned [`CheckedProgram`] contains ONLY the new-code's
/// checked decls; library decls are not duplicated.
///
/// The `context` is `Arc`-shared and never mutated — repeated calls
/// against the same context see the same outer scope.
///
/// ## Downstream-caller contract
///
/// The returned `CheckedProgram` is asymmetric on purpose:
/// - `type_env()` is **unioned** — it carries library + new-code declared
///   types so callers like `chelis_ir::lower_program` and
///   `chelis_effects::check_program` can resolve `(var libname)` references
///   from new-code bodies. New-code types win on shadow.
/// - `annotated_exprs()` is **new-code only** — library bodies are NOT
///   present. Phase D / E / F (effects, linearity, lowering) callers MUST
///   use the corresponding `_with_context` variants, not the monolithic
///   `check_program` / `check_linearity` / `lower_program`. The monolithic
///   APIs need to walk library bodies and will silently mis-handle
///   library-effect propagation, library tensor consumption, and library
///   IR roots if fed only the new-code annotated decls.
pub fn check_phase0e_with_context(
    context: &TypeEnv,
    new_exprs: &[deep::Expr],
) -> Result<CheckedProgram, InferResult> {
    let mut state = context.inner().clone();

    // New-code declared types (Phase 0e) layered on top of library's.
    let new_phase0e = build_phase0e_type_env(new_exprs);
    let combined_phase0e: HashMap<String, deep::Expr> = state
        .phase0e_types
        .iter()
        .chain(new_phase0e.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // Library is already validated; only run validate / inference on
    // new exprs. The Phase 0e env passed to inference is the new-code's
    // own declared types (library schemes are already in state.env).
    let mut result = infer_phase0e_program_with_state(
        new_exprs,
        &new_phase0e,
        &mut state,
        &combined_phase0e,
        /* run_validate_passes_on = */ None,
    );
    // Run cycle / shape / precision validators on new_exprs only. The
    // combined Phase 0e env is supplied so `(var libfoo)` references
    // resolve to the library's declared type during shape validation.
    validate_phase0e_program(new_exprs, &combined_phase0e, &mut result.errors);
    validate_tensor_precisions_in_program(new_exprs, &mut result.errors);
    suppress_unbound_for_cycle_members_against_context(
        new_exprs,
        &context.inner().library_def_names,
        &mut result.errors,
    );
    if !result.errors.is_empty() {
        return Err(result);
    }

    // Annotate ONLY the new-code exprs, starting from the library
    // snapshot state so library names resolve during annotation.
    let annotated_exprs = annotate_phase0e_program_with_context(context, new_exprs);
    // Surface library declared types in the returned type_env so downstream
    // passes (lower, effects, linearity) can resolve `(var libname)` calls
    // from new-code without a separate library lookup. New-code types take
    // precedence on shadow.
    let mut annotated_type_env = build_phase0e_type_env(&annotated_exprs);
    for (name, ty) in &context.inner().phase0e_types {
        annotated_type_env
            .entry(name.clone())
            .or_insert_with(|| ty.clone());
    }
    Ok(CheckedProgram::from_parts(
        annotated_exprs,
        annotated_type_env,
    ))
}

pub fn check_typed_program(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    let result = infer_program(exprs);
    if result.errors.is_empty() {
        let annotated_exprs = annotate_phase0e_program(exprs);
        let annotated_type_env = build_phase0e_type_env(&annotated_exprs);
        Ok(CheckedProgram::from_parts(
            annotated_exprs,
            annotated_type_env,
        ))
    } else {
        Err(result)
    }
}

pub fn infer_phase0e_program(exprs: &[deep::Expr]) -> InferResult {
    let type_env = build_phase0e_type_env(exprs);
    let mut result = infer_phase0e_program_with_env(exprs, &type_env);
    validate_phase0e_program(exprs, &type_env, &mut result.errors);
    validate_tensor_precisions_in_program(exprs, &mut result.errors);
    suppress_unbound_for_cycle_members(exprs, &mut result.errors);
    result
}

/// When a binding cycle is detected, the inference pass that processed
/// the cycle in textual order often reports `UnboundVariable` for the
/// later cycle members (the lookup landed before the subsequent def was
/// elaborated). Those errors are spurious noise — the names ARE defined,
/// they're just circularly. Drop any `UnboundVariable` whose name matches
/// a top-level def.
fn suppress_unbound_for_cycle_members(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    let mut def_names: HashSet<String> = HashSet::new();
    for expr in top_level_decl_items(exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            def_names.insert(name.to_string());
        }
    }
    errors.retain(|err| {
        if !matches!(err.kind, CheckErrorKind::UnboundVariable) {
            return true;
        }
        let name_start = match err.message.find("unbound variable: ") {
            Some(start) => start + "unbound variable: ".len(),
            None => return true,
        };
        let name = err.message[name_start..]
            .split_whitespace()
            .next()
            .unwrap_or("");
        !def_names.contains(name)
    });
}

/// Stacked-context variant of `suppress_unbound_for_cycle_members`.
///
/// Drops `UnboundVariable` errors whose name matches either a new-code
/// def OR a library def — the latter is needed because a library def
/// referenced from the new code might temporarily look unbound during
/// inference if the inferred error path runs before the env scheme
/// lookup, but the name IS in the library context.
fn suppress_unbound_for_cycle_members_against_context(
    new_exprs: &[deep::Expr],
    library_def_names: &HashSet<String>,
    errors: &mut Vec<CheckError>,
) {
    let mut def_names: HashSet<String> = library_def_names.clone();
    for expr in top_level_decl_items(new_exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(symbol_name)
        {
            def_names.insert(name.to_string());
        }
    }
    errors.retain(|err| {
        if !matches!(err.kind, CheckErrorKind::UnboundVariable) {
            return true;
        }
        let name_start = match err.message.find("unbound variable: ") {
            Some(start) => start + "unbound variable: ".len(),
            None => return true,
        };
        let name = err.message[name_start..]
            .split_whitespace()
            .next()
            .unwrap_or("");
        !def_names.contains(name)
    });
}

fn infer_phase0e_program_with_env(exprs: &[deep::Expr], type_env: &Phase0eTypeEnv) -> InferResult {
    // Backwards-compat wrapper. Callers (like `infer_phase0e_program` and
    // `check_typed_program` callers) run `validate_phase0e_program`
    // separately, so we pass `None` here to skip the embedded validate.
    let empty_inner = crate::context::TypeEnv::empty();
    let mut state = empty_inner.inner().clone();
    infer_phase0e_program_with_state(
        exprs, type_env, &mut state, type_env, /* run_validate_passes_on = */ None,
    )
}

/// Run the inference / Phase 0e binding / shape-validation passes against
/// `state`, mutating it as it goes. Library state should be supplied by
/// pre-cloning a snapshot; pass `&[]`-derived state for the monolithic
/// path. `new_phase0e_types` are bound into `state.env` here; the
/// `combined_phase0e` is what `validate_phase0e_program` consults so
/// new-code shape validation can look up declared types of library
/// references.
fn infer_phase0e_program_with_state(
    exprs: &[deep::Expr],
    new_phase0e_types: &Phase0eTypeEnv,
    state: &mut TypeEnvInner,
    combined_phase0e: &Phase0eTypeEnv,
    run_validate_passes_on: Option<&[deep::Expr]>,
) -> InferResult {
    let mut errors = Vec::new();
    let mut typed_nodes = 0;
    let mut total_nodes = 0;

    // Descend through `(module {} name ...)` wrappers: every idiomatic
    // Surf source wraps its declarations in `module X`, and without
    // flattening none of the walkers below see any def/defsig/deftype.
    for expr in top_level_decl_items(exprs) {
        collect_declarations(
            expr,
            &mut state.env,
            &mut state.var_gen,
            &mut state.subst,
            &mut state.adt_reg,
            &mut errors,
        );
    }

    for (name, ty_expr) in new_phase0e_types {
        let ty = deep_type_to_resolved_type(
            ty_expr,
            &mut state.var_gen,
            &state.adt_reg,
            &mut HashMap::new(),
        );
        let scheme = state.env.generalize(&ty, &state.subst);
        state.env.bind(name.clone(), scheme);
    }

    for expr in top_level_decl_items(exprs) {
        infer_top_level(
            expr,
            &mut state.env,
            &mut state.var_gen,
            &mut state.subst,
            &state.adt_reg,
            &mut errors,
            &mut typed_nodes,
            &mut total_nodes,
        );
    }

    for warning in chelis_deep::validate::validate(exprs) {
        errors.push(CheckError::new(
            match warning.kind {
                chelis_deep::validate::WarningKind::Arity => CheckErrorKind::ArityMismatch,
                _ => CheckErrorKind::Other,
            },
            warning.message,
            vec!["Use canonical Deep 3-tuple forms from spec/03".to_string()],
        ));
    }

    if let Some(target_exprs) = run_validate_passes_on {
        validate_phase0e_program(target_exprs, combined_phase0e, &mut errors);
    }

    InferResult {
        errors,
        typed_nodes,
        total_nodes,
    }
}

type Phase0eTypeEnv = HashMap<String, deep::Expr>;

fn build_phase0e_type_env(exprs: &[deep::Expr]) -> Phase0eTypeEnv {
    let mut env = HashMap::new();
    for expr in top_level_decl_items(exprs) {
        collect_phase0e_types(expr, &mut env);
    }
    env
}

fn collect_phase0e_types(expr: &deep::Expr, env: &mut Phase0eTypeEnv) {
    let deep::Expr::List(list, _) = expr else {
        return;
    };
    if get_tag(list) != Some("def") {
        return;
    }
    let kids = children(list);
    if kids.len() < 2 {
        return;
    }
    if let Some(name) = symbol_name(&kids[0])
        && let Some(ty) = expr_type_expr(&kids[1], env)
    {
        env.insert(name.to_string(), ty);
    }
}

fn validate_phase0e_program(
    exprs: &[deep::Expr],
    type_env: &Phase0eTypeEnv,
    errors: &mut Vec<CheckError>,
) {
    detect_top_level_binding_cycles(exprs, errors);
    detect_trivial_non_terminating_fns(exprs, errors);
    let mut static_env = HashMap::new();
    for expr in top_level_decl_items(exprs) {
        validate_phase0e_expr(expr, type_env, &mut static_env, errors);
    }
}

/// Detect fn defs whose body is a direct self-call with no conditional
/// guard — e.g. `def a(x) = a(x)`. These are guaranteed non-terminating
/// when called and, because the DAG lowerer can't represent recursion,
/// get silently elided to an identity in the generated C (source/object
/// divergence). Flag at check time so the user sees a clear error
/// instead of shipping a program that means something else than written.
///
/// This only catches the most trivial shape — a body that is literally
/// `(app (var name) ...)` with the def's own name as the callee. Real
/// recursive fns with a base case inside `if`/`match` (e.g. `fact n = if
/// n <= 1 then 1 else mul(n, fact(n-1))`) are NOT flagged.
fn detect_trivial_non_terminating_fns(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    // Collect each def's "terminal callees" — the top-level fn names
    // reached at every tail position of the body. `Some(set)` means
    // every tail is a call; the set is who's called. `None` means the
    // body has at least one non-call tail (a base case exists).
    let mut terminal_callees: HashMap<String, Option<HashSet<String>>> = HashMap::new();
    let mut def_order: Vec<String> = Vec::new();
    for expr in top_level_decl_items(exprs) {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some("def") {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else { continue };
        // Check params for a name that shadows the def — a body that
        // terminal-calls a shadowed name is NOT self-recursion.
        let mut shadows: HashSet<String> = HashSet::new();
        if let deep::Expr::List(fn_list, _) = body
            && get_tag(fn_list) == Some("fn")
            && let Some(deep::Expr::List(params, _)) = children(fn_list).first()
            && get_tag(params) == Some("params")
        {
            for param in children(params) {
                if let Some(pname) = param_name_for_refs(param) {
                    shadows.insert(pname);
                }
            }
        }
        let fn_body = match body {
            deep::Expr::List(list, _) if get_tag(list) == Some("fn") => children(list).get(1),
            _ => None,
        };
        let entry = if let Some(fn_body) = fn_body {
            let mut callees: HashSet<String> = HashSet::new();
            if collect_terminal_callees(fn_body, &shadows, &mut callees) {
                Some(callees)
            } else {
                None
            }
        } else {
            None
        };
        def_order.push(name.to_string());
        terminal_callees.insert(name.to_string(), entry);
    }

    // Greatest-fixed-point: start with ALL defs whose every tail is a
    // call (no base case) and iteratively remove any def that calls out
    // to a base-case def (outside the candidate set). What survives is
    // a closed recursion group with no base case anywhere.
    let mut non_terminating: HashSet<String> = terminal_callees
        .iter()
        .filter_map(|(name, callees)| {
            callees.as_ref().and_then(|set| {
                if set.is_empty() {
                    None
                } else {
                    Some(name.clone())
                }
            })
        })
        .collect();
    loop {
        let mut changed = false;
        let snapshot: Vec<String> = non_terminating.iter().cloned().collect();
        for name in &snapshot {
            let Some(Some(callees)) = terminal_callees.get(name) else {
                non_terminating.remove(name);
                changed = true;
                continue;
            };
            // Every callee must either be `name` itself OR remain in the
            // non_terminating candidate set. If any callee has a known
            // base case (isn't in non_terminating), this def has an
            // escape route and isn't trivially non-terminating.
            let ok = callees
                .iter()
                .all(|c| c == name || non_terminating.contains(c));
            if !ok {
                non_terminating.remove(name);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    for name in &def_order {
        if non_terminating.contains(name) {
            errors.push(CheckError::new(
                CheckErrorKind::CycleDetected,
                format!(
                    "def `{name}` is trivially non-terminating — every tail position \
                     calls back into the same recursion group `{name}` with no base case; \
                     add an `if`/`match` exit that returns without recursing"
                ),
                vec![
                    "Trivial (self- or mutual-) recursion without a base case isn't \
                     representable in the Phase 0 DAG lowering and would compile to \
                     an infinite loop or silent identity."
                        .to_string(),
                ],
            ));
        }
    }
}

/// Walk `expr` and, for every terminal (tail) position, record the name
/// called (if the tail is `(app (var Y) ...)`). Returns `true` if EVERY
/// terminal is a call (no base-case leaf); `false` if any terminal is a
/// non-call (literal, var-read, tuple, etc.) — a base case exists.
fn collect_terminal_callees(
    expr: &deep::Expr,
    shadowed: &HashSet<String>,
    out: &mut HashSet<String>,
) -> bool {
    match expr {
        deep::Expr::MetaExpr(meta, _) => collect_terminal_callees(&meta.expr, shadowed, out),
        deep::Expr::List(list, _) => match get_tag(list) {
            Some("app") => {
                let kids = children(list);
                let Some(callee) = kids.first() else {
                    return false;
                };
                let deep::Expr::List(callee_list, _) = callee else {
                    return false;
                };
                if get_tag(callee_list) != Some("var") {
                    return false;
                }
                let Some(cname) = children(callee_list).first().and_then(symbol_name) else {
                    return false;
                };
                if shadowed.contains(cname) {
                    return false;
                }
                out.insert(cname.to_string());
                true
            }
            Some("let") => {
                let kids = children(list);
                kids.get(1)
                    .map(|body| collect_terminal_callees(body, shadowed, out))
                    .unwrap_or(false)
            }
            Some("if") => {
                let kids = children(list);
                if kids.len() < 3 {
                    return false;
                }
                let then_ok = collect_terminal_callees(&kids[1], shadowed, out);
                let else_ok = collect_terminal_callees(&kids[2], shadowed, out);
                then_ok && else_ok
            }
            Some("match") => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                kids.iter().skip(1).all(|arm| {
                    if let deep::Expr::List(arm_list, _) = arm
                        && get_tag(arm_list) == Some("arm")
                    {
                        children(arm_list)
                            .get(2)
                            .map(|body| collect_terminal_callees(body, shadowed, out))
                            .unwrap_or(false)
                    } else {
                        false
                    }
                })
            }
            _ => false,
        },
        _ => false,
    }
}

#[allow(dead_code)]
fn fn_body_is_direct_self_call(def_body: &deep::Expr, def_name: &str) -> bool {
    let fn_list = match def_body {
        deep::Expr::List(list, _) if get_tag(list) == Some("fn") => list,
        _ => return false,
    };
    // If any fn param shadows the def name, the callee reference inside
    // the body refers to the param (a callable HOF argument), not the def
    // itself. This is a legitimate HOF call, not recursion.
    if let Some(params_list) = children(fn_list).first()
        && let deep::Expr::List(params, _) = params_list
        && get_tag(params) == Some("params")
    {
        for param in children(params) {
            if param_name_for_refs(param).as_deref() == Some(def_name) {
                return false;
            }
        }
    }
    let Some(body) = children(fn_list).get(1) else {
        return false;
    };
    every_terminal_is_self_call(body, def_name)
}

/// True when every terminal (tail) position of `expr` is a direct call
/// to `def_name`. Walks through `let` bodies, both arms of `if`, and
/// every `match` arm body. Any non-self-call terminal (a literal, a
/// different fn call, a non-self var) makes this false — that terminal
/// is a potential base case and the recursion isn't trivial.
#[allow(dead_code)]
fn every_terminal_is_self_call(expr: &deep::Expr, def_name: &str) -> bool {
    match expr {
        deep::Expr::MetaExpr(meta, _) => every_terminal_is_self_call(&meta.expr, def_name),
        deep::Expr::List(list, _) => match get_tag(list) {
            Some("app") => {
                let kids = children(list);
                let Some(callee) = kids.first() else {
                    return false;
                };
                let deep::Expr::List(callee_list, _) = callee else {
                    return false;
                };
                if get_tag(callee_list) != Some("var") {
                    return false;
                }
                children(callee_list).first().and_then(symbol_name) == Some(def_name)
            }
            Some("let") => {
                let kids = children(list);
                kids.get(1)
                    .map(|body| every_terminal_is_self_call(body, def_name))
                    .unwrap_or(false)
            }
            Some("if") => {
                let kids = children(list);
                if kids.len() < 3 {
                    return false;
                }
                every_terminal_is_self_call(&kids[1], def_name)
                    && every_terminal_is_self_call(&kids[2], def_name)
            }
            Some("match") => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                kids.iter().skip(1).all(|arm| {
                    if let deep::Expr::List(arm_list, _) = arm
                        && get_tag(arm_list) == Some("arm")
                    {
                        children(arm_list)
                            .get(2)
                            .map(|body| every_terminal_is_self_call(body, def_name))
                            .unwrap_or(false)
                    } else {
                        false
                    }
                })
            }
            _ => false,
        },
        _ => false,
    }
}

/// Check whether a def body is literally a self-reference — the Nautilus
/// external-input pattern `x = x` or `x = (x : T)`. Only this structural
/// shape earns the self-loop carve-out; self-references reached via a fn
/// application, a tuple, an if, etc. are real cycles and must be reported.
fn body_is_literal_self_ref(body: &deep::Expr, name: &str) -> bool {
    let mut current = body;
    loop {
        match current {
            deep::Expr::MetaExpr(meta, _) => current = &meta.expr,
            deep::Expr::List(list, _) => match get_tag(list) {
                Some("var") => {
                    return children(list).first().and_then(symbol_name) == Some(name);
                }
                // Type ascription desugars into a `(cast ... )`-like node
                // in Deep: `(x : T)` keeps `x` as the first child. When
                // the underlying is a var with the self name, treat it as
                // the Nautilus pattern.
                Some("ascribe") | Some(":") => match children(list).first() {
                    Some(inner) => current = inner,
                    None => return false,
                },
                _ => return false,
            },
            _ => return false,
        }
    }
}

/// Yield each top-level declaration, flattening through a `(module {} name ...)`
/// wrapper if present. Deep sources produced by Surf `module X` desugaring
/// have every def/defsig/deftype inside this wrapper; without flattening,
/// top-level walkers see a single `(module ...)` and miss everything inside.
fn top_level_decl_items(exprs: &[deep::Expr]) -> Vec<&deep::Expr> {
    fn push<'a>(expr: &'a deep::Expr, out: &mut Vec<&'a deep::Expr>) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("module")
        {
            // `(module {} name children...)` — skip tag, meta, name.
            for child in list.elements.iter().skip(3) {
                push(child, out);
            }
            return;
        }
        out.push(expr);
    }
    let mut out = Vec::new();
    for expr in exprs {
        push(expr, &mut out);
    }
    out
}

/// Detect cycles among top-level `def` bindings.
///
/// The Nautilus external-input pattern `x = (x : tensor[...])` is permitted —
/// a self-loop (the def body references the same name it is binding, with no
/// intermediate hops) is treated as a declaration of an external input, not as
/// a cycle. Any cycle of length >= 2 (e.g. `a -> b -> a`, `a -> b -> c -> a`)
/// is a real binding cycle and is reported as a `CycleDetected` error.
fn detect_top_level_binding_cycles(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    let mut def_names: Vec<String> = Vec::new();
    let mut def_name_set: HashSet<String> = HashSet::new();
    let mut def_bodies: HashMap<String, &deep::Expr> = HashMap::new();
    // Descend through `(module {} name ...)` wrappers so this check works
    // on idiomatic Surf sources (every `.ch` file starts with `module X`,
    // which desugars to a single top-level `module` list wrapping every
    // declaration). Without this, the cycle check is a no-op in practice.
    for expr in top_level_decl_items(exprs) {
        if let deep::Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
        {
            let kids = children(list);
            let Some(name) = kids.first().and_then(symbol_name) else {
                continue;
            };
            let Some(body) = kids.get(1) else { continue };
            if !def_name_set.contains(name) {
                def_name_set.insert(name.to_string());
                def_names.push(name.to_string());
            }
            def_bodies.insert(name.to_string(), body);
        }
    }

    // Phase 1: per-def, collect both the vars referenced EAGERLY (outside fn
    // bodies) and the top-level fns APPLIED eagerly. Lazy refs inside fn
    // bodies are captured separately so we can chain them in on demand.
    let mut direct_refs: HashMap<String, HashSet<String>> = HashMap::new();
    let mut applied_fns: HashMap<String, HashSet<String>> = HashMap::new();
    let mut fn_body_refs: HashMap<String, (HashSet<String>, HashSet<String>)> = HashMap::new();
    for name in &def_names {
        let Some(body) = def_bodies.get(name) else {
            continue;
        };
        let mut refs: HashSet<String> = HashSet::new();
        let mut applied: HashSet<String> = HashSet::new();
        let mut bound: HashSet<String> = HashSet::new();
        collect_eager_refs(body, &mut bound, &mut refs, &mut applied);
        direct_refs.insert(name.clone(), refs);
        applied_fns.insert(name.clone(), applied);
        // If this def's body IS itself a fn, also collect what its body
        // references so callers of this def can chain.
        if let deep::Expr::List(list, _) = body
            && get_tag(list) == Some("fn")
            && let Some(fn_body) = children(list).get(1)
        {
            let mut inner_refs: HashSet<String> = HashSet::new();
            let mut inner_applied: HashSet<String> = HashSet::new();
            let mut inner_bound: HashSet<String> = HashSet::new();
            // Bind the fn's own params so they aren't flagged as refs.
            if let Some(params_list) = children(list).first()
                && let deep::Expr::List(params, _) = params_list
                && get_tag(params) == Some("params")
            {
                for param in children(params) {
                    if let Some(pname) = param_name_for_refs(param) {
                        inner_bound.insert(pname);
                    }
                }
            }
            collect_eager_refs(
                fn_body,
                &mut inner_bound,
                &mut inner_refs,
                &mut inner_applied,
            );
            fn_body_refs.insert(name.clone(), (inner_refs, inner_applied));
        }
    }

    // Phase 2: build two edge sets per def.
    //   - value_edges[d]: names read AS VALUES in d's body (i.e. `(var x)`
    //     where x is not a fn callee). Reading a value requires that value
    //     to already be bound — a cycle here is a real binding cycle.
    //   - call_edges[d]: names d CALLS (`(app (var f) ...)`). Calling a fn
    //     pushes its body into eager evaluation but does NOT require f's
    //     value — f is a callable, not a scalar. Recursive calls with base
    //     cases terminate and don't close a cycle.
    //
    // Cycle condition: DFS from each VALUE def X, traversing both edge
    // kinds transitively. Track the stack of VALUE defs we're currently
    // evaluating. If a value-edge lands on a stack member, that's a real
    // binding cycle. Fn names aren't pushed onto the stack — they are
    // intermediates in the path.
    let mut value_edges: HashMap<String, Vec<String>> = HashMap::new();
    let mut call_edges: HashMap<String, Vec<String>> = HashMap::new();
    for name in &def_names {
        let body = def_bodies.get(name).copied();
        let is_nautilus_literal_self = body.is_some_and(|b| body_is_literal_self_ref(b, name));
        let body_is_fn = matches!(
            body,
            Some(deep::Expr::List(list, _)) if get_tag(list) == Some("fn")
        );

        let (raw_refs, raw_applied) = if body_is_fn {
            let empty_refs: HashSet<String> = HashSet::new();
            let empty_applied: HashSet<String> = HashSet::new();
            fn_body_refs
                .get(name)
                .map(|(r, a)| (r.clone(), a.clone()))
                .unwrap_or((empty_refs, empty_applied))
        } else {
            (
                direct_refs.get(name).cloned().unwrap_or_default(),
                applied_fns.get(name).cloned().unwrap_or_default(),
            )
        };

        let mut value_out: Vec<String> = raw_refs
            .into_iter()
            .filter(|r| {
                if r == name && is_nautilus_literal_self {
                    return false;
                }
                def_name_set.contains(r)
            })
            .collect();
        let mut call_out: Vec<String> = raw_applied
            .into_iter()
            .filter(|r| def_name_set.contains(r))
            .collect();
        value_out.sort();
        call_out.sort();
        value_edges.insert(name.clone(), value_out);
        call_edges.insert(name.clone(), call_out);
    }

    let mut reported: HashSet<Vec<String>> = HashSet::new();

    // DFS from each value def. Track:
    //   - `value_stack`: the value defs we're "currently evaluating". A
    //     value_edge landing on a member of this stack is a cycle.
    //   - `visited`: nodes we've already fully explored from some starting
    //     value def. Avoids re-walking fn bodies we've cleared.
    //   - `path`: the traversal path for error reporting (includes both
    //     values and fns as intermediates).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Color {
        White,
        Gray,
        Black,
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        node: &str,
        is_value: &dyn Fn(&str) -> bool,
        value_edges: &HashMap<String, Vec<String>>,
        call_edges: &HashMap<String, Vec<String>>,
        color: &mut HashMap<String, Color>,
        value_stack: &mut Vec<String>,
        path: &mut Vec<String>,
        reported: &mut HashSet<Vec<String>>,
        errors: &mut Vec<CheckError>,
    ) {
        color.insert(node.to_string(), Color::Gray);
        let this_is_value = is_value(node);
        if this_is_value {
            value_stack.push(node.to_string());
        }
        path.push(node.to_string());

        if let Some(neighbors) = value_edges.get(node) {
            for next in neighbors {
                if let Some(start) = value_stack.iter().position(|s| s == next) {
                    // Value-edge landing on a value currently being
                    // evaluated → real binding cycle.
                    let value_seg_start = path.iter().position(|s| s == &value_stack[start]);
                    let cycle: Vec<String> = if let Some(s) = value_seg_start {
                        path[s..].to_vec()
                    } else {
                        value_stack[start..].to_vec()
                    };
                    let mut canon = cycle.clone();
                    if let Some((min_idx, _)) = canon.iter().enumerate().min_by(|a, b| a.1.cmp(b.1))
                    {
                        canon.rotate_left(min_idx);
                    }
                    if reported.insert(canon.clone()) {
                        let mut pathstr = canon.clone();
                        pathstr.push(canon[0].clone());
                        let message = format!("binding cycle: {}", pathstr.join(" -> "));
                        errors.push(CheckError::new(
                            CheckErrorKind::CycleDetected,
                            message,
                            vec![
                                "Break the cycle by removing one of the \
                                 self-referential definitions or replacing it \
                                 with a concrete value."
                                    .to_string(),
                            ],
                        ));
                    }
                } else if color.get(next).copied().unwrap_or(Color::White) == Color::White {
                    dfs(
                        next,
                        is_value,
                        value_edges,
                        call_edges,
                        color,
                        value_stack,
                        path,
                        reported,
                        errors,
                    );
                }
            }
        }

        if let Some(neighbors) = call_edges.get(node) {
            for next in neighbors {
                // Fn calls don't require the callee's VALUE — they just
                // push the callee's body into eager evaluation. Gray nodes
                // are mid-exploration (recursive reentry) — skip to avoid
                // infinite DFS.
                if color.get(next).copied().unwrap_or(Color::White) == Color::White {
                    dfs(
                        next,
                        is_value,
                        value_edges,
                        call_edges,
                        color,
                        value_stack,
                        path,
                        reported,
                        errors,
                    );
                }
            }
        }

        path.pop();
        if this_is_value {
            value_stack.pop();
        }
        color.insert(node.to_string(), Color::Black);
    }

    let is_value = |name: &str| -> bool {
        def_bodies
            .get(name)
            .map(|body| !matches!(body, deep::Expr::List(list, _) if get_tag(list) == Some("fn")))
            .unwrap_or(false)
    };
    let mut color: HashMap<String, Color> = def_names
        .iter()
        .map(|n| (n.clone(), Color::White))
        .collect();
    let mut value_stack: Vec<String> = Vec::new();
    let mut path: Vec<String> = Vec::new();
    for name in &def_names {
        if !is_value(name) {
            continue;
        }
        if color.get(name).copied() == Some(Color::White) {
            dfs(
                name,
                &is_value,
                &value_edges,
                &call_edges,
                &mut color,
                &mut value_stack,
                &mut path,
                &mut reported,
                errors,
            );
        }
    }
}

fn param_name_for_refs(param: &deep::Expr) -> Option<String> {
    match param {
        deep::Expr::Atom(deep::Atom::Symbol(name), _) => Some(name.clone()),
        deep::Expr::MetaExpr(meta, _) => param_name_for_refs(&meta.expr),
        // A Deep param is `(name {type: ...})` — a List with the name as
        // the FIRST element and the meta map as the second. `children()`
        // skips first two (tag + meta) and returns nothing for a 2-elem
        // list, so read elements[0] directly.
        deep::Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .map(str::to_string),
        _ => None,
    }
}

/// Collect both eager references and top-level fn applications.
///
/// `refs` gets free `(var name)` references that fire at definition time
/// (i.e., NOT inside an enclosing `fn` body).
///
/// `applied` gets names of fns called as `(app (var F) ...)` at definition
/// time (again, not inside a nested fn body). Callers use `applied` to
/// chain in the called fn's own eager refs for cycle detection — this is
/// what catches top-level value cycles that route through a fn call:
///
/// ```text
/// a = f()
/// b = g()
/// def f() = b
/// def g() = a
/// ```
///
/// Plain `collect_free_var_refs` (kept below for backward compatibility)
/// ignores fn bodies entirely, which correctly permits mutual recursion
/// between fn defs never called eagerly — but misses the cycle above.
fn collect_eager_refs(
    expr: &deep::Expr,
    bound: &mut HashSet<String>,
    refs: &mut HashSet<String>,
    applied: &mut HashSet<String>,
) {
    match expr {
        deep::Expr::List(list, _) => match get_tag(list) {
            Some("var") => {
                if let Some(name) = children(list).first().and_then(symbol_name)
                    && !bound.contains(name)
                {
                    refs.insert(name.to_string());
                }
            }
            Some("fn") => {
                // Skip fn body — only its application at this site (if any)
                // is eager; the body itself is deferred.
            }
            Some("app") => {
                let kids = children(list);
                if let Some(callee) = kids.first()
                    && let deep::Expr::List(clist, _) = callee
                    && get_tag(clist) == Some("var")
                    && let Some(fname) = children(clist).first().and_then(symbol_name)
                    && !bound.contains(fname)
                {
                    // Callee is in `applied` only — NOT in `refs`. For cycle
                    // detection, reading `g` as a value is different from
                    // calling `g()`: the former requires g's value now, the
                    // latter just pushes g's body into eager evaluation and
                    // may terminate at a base case.
                    applied.insert(fname.to_string());
                } else if let Some(callee) = kids.first() {
                    collect_eager_refs(callee, bound, refs, applied);
                }
                for arg in kids.iter().skip(1) {
                    collect_eager_refs(arg, bound, refs, applied);
                }
            }
            Some("let") => {
                let kids = children(list);
                let mut added: Vec<String> = Vec::new();
                if let Some(deep::Expr::List(bind_list, _)) = kids.first()
                    && get_tag(bind_list) == Some("bind")
                {
                    let bind_kids = children(bind_list);
                    let mut i = 0;
                    while i + 1 < bind_kids.len() {
                        collect_eager_refs(&bind_kids[i + 1], bound, refs, applied);
                        if let Some(name) = symbol_name(&bind_kids[i])
                            && bound.insert(name.to_string())
                        {
                            added.push(name.to_string());
                        }
                        i += 2;
                    }
                }
                if let Some(body) = kids.get(1) {
                    collect_eager_refs(body, bound, refs, applied);
                }
                for name in added {
                    bound.remove(&name);
                }
            }
            _ => {
                for elem in &list.elements {
                    collect_eager_refs(elem, bound, refs, applied);
                }
            }
        },
        deep::Expr::Map(map, _) => {
            for (_, v) in &map.entries {
                collect_eager_refs(v, bound, refs, applied);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            for (_, v) in &meta.entries {
                collect_eager_refs(v, bound, refs, applied);
            }
            collect_eager_refs(&meta.expr, bound, refs, applied);
        }
        deep::Expr::Atom(_, _) => {}
    }
}

/// Walk the program's Deep AST and reject any `(t-tensor ... (t-prim {} P))`
/// whose precision P is not supported by the Phase 0f tensor backend
/// (currently: f16, bf16, f64, f8e4m3, string).
///
/// This runs after HM inference so it catches user-written tensor type
/// ascriptions, defsig tensor types, parameter type annotations, literal
/// type metadata, and any cast target that produces a tensor with an
/// unsupported element precision.
fn validate_tensor_precisions_in_program(exprs: &[deep::Expr], errors: &mut Vec<CheckError>) {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    // Descend through `(module {} name ...)` wrappers so per-def dedup
    // keeps each def's tensor types in their own key space (otherwise
    // every def lives under def_context="" and errors collapse).
    for expr in top_level_decl_items(exprs) {
        let def_name = match expr {
            deep::Expr::List(list, _)
                if matches!(
                    get_tag(list),
                    Some("def") | Some("defsig") | Some("deftype") | Some("typealias")
                ) =>
            {
                children(list)
                    .first()
                    .and_then(symbol_name)
                    .unwrap_or("")
                    .to_string()
            }
            _ => String::new(),
        };
        walk_for_tensor_precision(expr, errors, &mut seen, &def_name);
    }
}

fn walk_for_tensor_precision(
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
    seen: &mut HashSet<(String, String)>,
    def_context: &str,
) {
    match expr {
        deep::Expr::List(list, _span) => {
            // Check t-tensor nodes at this level.
            if get_tag(list) == Some("t-tensor") {
                let kids = children(list);
                if let Some(last) = kids.last()
                    && let deep::Expr::List(prec_list, _) = last
                    && get_tag(prec_list) == Some("t-prim")
                    && let Some(name) = children(prec_list).first().and_then(symbol_name)
                    && let Some(prim) = Prim::parse_name(name)
                    && !prim.is_valid_tensor_precision()
                    && seen.insert((def_context.to_string(), name.to_string()))
                {
                    errors.push(CheckError::new(
                        CheckErrorKind::UnsupportedTensorPrecision,
                        format!(
                            "tensor element precision `{}` is not supported by the current \
                             Phase 0f backend (supported: f32, f64, bool, int8, int32, int64)",
                            name
                        ),
                        vec![format!(
                            "Use tensor[..., f32] and cast host scalars explicitly, or keep `{name}` as a host scalar",
                            name = name,
                        )],
                    ));
                }
            }

            // `cast` is inferred by infer_cast which already emits a clearer
            // site-local error for bad precisions. Skip the walker's recursion
            // inside a cast so we don't duplicate the diagnostic.
            if get_tag(list) == Some("cast") {
                return;
            }

            // Recurse into metadata map (element[1]), which may carry
            // `type:` ascriptions that also contain t-tensor types.
            if list.elements.len() >= 2
                && let deep::Expr::Map(map, _) = &list.elements[1]
            {
                for (_, v) in &map.entries {
                    walk_for_tensor_precision(v, errors, seen, def_context);
                }
            }

            // Recurse into children (elements after index 1).
            for child in children(list) {
                walk_for_tensor_precision(child, errors, seen, def_context);
            }
        }
        deep::Expr::Map(map, _) => {
            for (_, v) in &map.entries {
                walk_for_tensor_precision(v, errors, seen, def_context);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            for (_, v) in &meta.entries {
                walk_for_tensor_precision(v, errors, seen, def_context);
            }
            walk_for_tensor_precision(&meta.expr, errors, seen, def_context);
        }
        deep::Expr::Atom(_, _) => {}
    }
}

fn validate_phase0e_expr(
    expr: &deep::Expr,
    type_env: &Phase0eTypeEnv,
    static_env: &mut HashMap<String, StaticValue>,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    match expr {
        deep::Expr::List(list, _) => {
            if get_tag(list) == Some("module") {
                for elem in list.elements.iter().skip(3) {
                    validate_phase0e_expr(elem, type_env, static_env, errors);
                }
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some("def") {
                let kids = children(list);
                let Some(name) = kids.first().and_then(symbol_name) else {
                    return StaticValue::Unknown;
                };
                let Some(value_expr) = kids.get(1) else {
                    return StaticValue::Unknown;
                };
                let value = validate_phase0e_expr(value_expr, type_env, static_env, errors);
                static_env.insert(name.to_string(), value);
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some("fn") {
                let scoped_env = extend_phase0e_env_with_fn_params(list, type_env);
                let mut scoped_static_env = static_env.clone();
                bind_fn_params_unknown(list, &mut scoped_static_env);
                for elem in &list.elements {
                    validate_phase0e_expr(elem, &scoped_env, &mut scoped_static_env, errors);
                }
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some("let") {
                let kids = children(list);
                let mut scoped_static_env = static_env.clone();
                if let Some(deep::Expr::List(bind_list, _)) = kids.first()
                    && get_tag(bind_list) == Some("bind")
                {
                    let bind_children = children(bind_list);
                    let mut index = 0;
                    while index + 1 < bind_children.len() {
                        if let Some(name) = symbol_name(&bind_children[index]) {
                            let value = validate_phase0e_expr(
                                &bind_children[index + 1],
                                type_env,
                                &mut scoped_static_env,
                                errors,
                            );
                            scoped_static_env.insert(name.to_string(), value);
                        }
                        index += 2;
                    }
                }
                if let Some(body) = kids.get(1) {
                    return validate_phase0e_expr(body, type_env, &mut scoped_static_env, errors);
                }
                return StaticValue::Unknown;
            }
            if let Some(tag) = get_tag(list) {
                if matches!(tag, "par" | "jit") {
                    errors.push(CheckError::new(
                        CheckErrorKind::Other,
                        format!("`{tag}` is not supported by Phase 0e lowering"),
                        vec!["Remove this construct or defer it to a later phase".to_string()],
                    ));
                }

                if tag == "app"
                    && let Some(func_name) = phase0e_builtin_name(list)
                    && is_phase0e_shape_sensitive_builtin(func_name)
                {
                    validate_phase0e_builtin_symbolic_requirements(
                        list, func_name, type_env, errors,
                    );
                }
            }

            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                return if name == "Nil" {
                    StaticValue::List(Vec::new())
                } else {
                    static_env
                        .get(name)
                        .cloned()
                        .unwrap_or(StaticValue::Unknown)
                };
            }
            if get_tag(list) == Some("lit") {
                return literal_static_value(expr);
            }
            if get_tag(list) == Some("cast") {
                let kids = children(list);
                return kids
                    .first()
                    .map(|inner| validate_phase0e_expr(inner, type_env, static_env, errors))
                    .unwrap_or(StaticValue::Unknown);
            }
            if get_tag(list) == Some("app") {
                let kids = children(list);
                let func_name = kids.first().and_then(app_builtin_name);
                let arg_values = kids
                    .iter()
                    .skip(1)
                    .map(|arg| validate_phase0e_expr(arg, type_env, static_env, errors))
                    .collect::<Vec<_>>();
                if func_name == Some("Cons") && arg_values.len() == 2 {
                    if let StaticValue::List(mut tail) = arg_values[1].clone() {
                        tail.insert(0, arg_values[0].clone());
                        return StaticValue::List(tail);
                    }
                    return StaticValue::Unknown;
                }
                if let Some(name) = func_name {
                    return validate_static_builtin_application(name, &arg_values, expr, errors);
                }
                return StaticValue::Unknown;
            }

            for elem in &list.elements {
                validate_phase0e_expr(elem, type_env, static_env, errors);
            }
            StaticValue::Unknown
        }
        deep::Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                validate_phase0e_expr(value, type_env, static_env, errors);
            }
            StaticValue::Unknown
        }
        deep::Expr::MetaExpr(meta, _) => {
            for (_, value) in &meta.entries {
                validate_phase0e_expr(value, type_env, static_env, errors);
            }
            validate_phase0e_expr(&meta.expr, type_env, static_env, errors)
        }
        deep::Expr::Atom(_, _) => literal_static_value(expr),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum StaticValue {
    Unknown,
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    List(Vec<StaticValue>),
    Tensor(StaticTensor),
}

#[derive(Debug, Clone, PartialEq)]
struct StaticTensor {
    shape: Vec<usize>,
    int_values: Option<Vec<i64>>,
}

fn bind_fn_params_unknown(fn_list: &deep::List, env: &mut HashMap<String, StaticValue>) {
    let Some(params_expr) = children(fn_list).first() else {
        return;
    };
    let deep::Expr::List(params_list, _) = params_expr else {
        return;
    };
    if get_tag(params_list) != Some("params") {
        return;
    }
    for param in children(params_list) {
        match param {
            deep::Expr::Atom(deep::Atom::Symbol(name), _) => {
                env.insert(name.clone(), StaticValue::Unknown);
            }
            deep::Expr::List(param_list, _) => {
                if let Some(name) = param_list.elements.first().and_then(symbol_name) {
                    env.insert(name.to_string(), StaticValue::Unknown);
                }
            }
            _ => {}
        }
    }
}

fn literal_static_value(expr: &deep::Expr) -> StaticValue {
    match expr {
        deep::Expr::Atom(deep::Atom::Int(value), _) => StaticValue::Int(*value),
        deep::Expr::Atom(deep::Atom::Float(value), _) => StaticValue::Float(*value),
        deep::Expr::Atom(deep::Atom::Bool(value), _) => StaticValue::Bool(*value),
        deep::Expr::Atom(deep::Atom::Str(value), _) => StaticValue::String(value.clone()),
        deep::Expr::List(list, _) if get_tag(list) == Some("lit") => children(list)
            .first()
            .map(literal_static_value)
            .unwrap_or(StaticValue::Unknown),
        _ => StaticValue::Unknown,
    }
}

fn app_builtin_name(expr: &deep::Expr) -> Option<&str> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn validate_static_builtin_application(
    name: &str,
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    match name {
        "pad_sequences" => static_pad_sequences(args),
        "to_tensor" => static_to_tensor(args),
        "concat" => static_concat(args, expr, errors),
        "split" => static_split(args, expr, errors),
        "gather" => static_gather(args, expr, errors),
        "scatter" => static_scatter(args, expr, errors),
        "clamp" => static_clamp(args, expr, errors),
        "einsum" => static_einsum(args, expr, errors),
        _ => StaticValue::Unknown,
    }
}

fn static_pad_sequences(args: &[StaticValue]) -> StaticValue {
    let Some(StaticValue::List(rows)) = args.first() else {
        return StaticValue::Unknown;
    };
    let mut width = 0usize;
    for row in rows {
        let StaticValue::List(items) = row else {
            return StaticValue::Unknown;
        };
        width = width.max(items.len());
    }
    StaticValue::Tensor(StaticTensor {
        shape: vec![rows.len(), width],
        int_values: None,
    })
}

fn static_to_tensor(args: &[StaticValue]) -> StaticValue {
    let Some(StaticValue::List(items)) = args.first() else {
        return StaticValue::Unknown;
    };
    let mut ints = Vec::with_capacity(items.len());
    for item in items {
        match item {
            StaticValue::Int(value) => ints.push(*value),
            StaticValue::Float(_) | StaticValue::Bool(_) => {
                return StaticValue::Tensor(StaticTensor {
                    shape: vec![items.len()],
                    int_values: None,
                });
            }
            _ => return StaticValue::Unknown,
        }
    }
    StaticValue::Tensor(StaticTensor {
        shape: vec![items.len()],
        int_values: Some(ints),
    })
}

fn static_concat(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (Some(StaticValue::List(parts)), Some(StaticValue::Int(axis))) =
        (args.first(), args.get(1))
    else {
        return StaticValue::Unknown;
    };
    if *axis < 0 {
        return StaticValue::Unknown;
    }
    let tensors = parts
        .iter()
        .map(|value| match value {
            StaticValue::Tensor(tensor) => Some(tensor.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>();
    let Some(tensors) = tensors else {
        return StaticValue::Unknown;
    };
    let Some(first) = tensors.first() else {
        return StaticValue::Unknown;
    };
    let axis = *axis as usize;
    if axis >= first.shape.len() {
        return StaticValue::Unknown;
    }
    let mut shape = first.shape.clone();
    let mut axis_total = shape[axis];
    for tensor in tensors.iter().skip(1) {
        if tensor.shape.len() != shape.len() {
            push_static_runtime_error(
                expr,
                errors,
                "concat expects matching tensor rank".to_string(),
            );
            return StaticValue::Unknown;
        }
        for (dim, expected_extent) in shape.iter().enumerate() {
            if dim != axis && tensor.shape[dim] != *expected_extent {
                push_static_runtime_error(
                    expr,
                    errors,
                    "concat expects matching non-concatenated axes".to_string(),
                );
                return StaticValue::Unknown;
            }
        }
        axis_total += tensor.shape[axis];
    }
    shape[axis] = axis_total;
    StaticValue::Tensor(StaticTensor {
        shape,
        int_values: None,
    })
}

fn static_split(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(tensor)),
        Some(StaticValue::Int(axis)),
        Some(StaticValue::List(sizes)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    if *axis < 0 {
        return StaticValue::Unknown;
    }
    let axis = *axis as usize;
    if axis >= tensor.shape.len() {
        return StaticValue::Unknown;
    }
    let Some(size_values) = sizes
        .iter()
        .map(|value| match value {
            StaticValue::Int(size) if *size >= 0 => Some(*size as usize),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    else {
        return StaticValue::Unknown;
    };
    if size_values.iter().sum::<usize>() != tensor.shape[axis] {
        push_static_runtime_error(
            expr,
            errors,
            "split sizes must sum to the selected axis extent".to_string(),
        );
    }
    StaticValue::Unknown
}

fn static_gather(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(tensor)),
        Some(StaticValue::Tensor(indices)),
        Some(StaticValue::Int(axis)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    let Some(axis) = normalize_static_axis(tensor.shape.len(), *axis) else {
        return StaticValue::Unknown;
    };
    if let Some(index_values) = &indices.int_values {
        for value in index_values {
            if *value < 0 || *value >= tensor.shape[axis] as i64 {
                push_static_runtime_error(
                    expr,
                    errors,
                    format!("gather index {value} out of bounds"),
                );
                return StaticValue::Unknown;
            }
        }
    }
    StaticValue::Tensor(StaticTensor {
        shape: gather_result_shape(&tensor.shape, &indices.shape, axis),
        int_values: None,
    })
}

fn static_scatter(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(base)),
        Some(StaticValue::Tensor(indices)),
        Some(StaticValue::Tensor(updates)),
        Some(StaticValue::Int(axis)),
        Some(StaticValue::String(mode)),
    ) = (
        args.first(),
        args.get(1),
        args.get(2),
        args.get(3),
        args.get(4),
    )
    else {
        return StaticValue::Unknown;
    };
    let Some(axis) = normalize_static_axis(base.shape.len(), *axis) else {
        return StaticValue::Unknown;
    };
    let expected_updates = gather_result_shape(&base.shape, &indices.shape, axis);
    if updates.shape != expected_updates {
        push_static_runtime_error(
            expr,
            errors,
            "scatter updates must match gathered tensor shape and precision".to_string(),
        );
        return StaticValue::Unknown;
    }
    if let Some(index_values) = &indices.int_values {
        let mut seen = HashSet::new();
        for linear in 0..updates_shape_numel(&updates.shape) {
            let update_index = unravel_index(linear, &updates.shape);
            let gather_index = update_index[axis..axis + indices.shape.len()].to_vec();
            let gather_linear = ravel_index(&gather_index, &indices.shape);
            let gathered = index_values[gather_linear];
            if gathered < 0 || gathered >= base.shape[axis] as i64 {
                push_static_runtime_error(
                    expr,
                    errors,
                    format!("scatter index {gathered} out of bounds"),
                );
                return StaticValue::Unknown;
            }
            if mode == "replace" {
                let mut out_index = Vec::with_capacity(base.shape.len());
                out_index.extend_from_slice(&update_index[..axis]);
                out_index.push(gathered as usize);
                out_index.extend_from_slice(&update_index[axis + indices.shape.len()..]);
                let out_linear = ravel_index(&out_index, &base.shape);
                if !seen.insert(out_linear) {
                    push_static_runtime_error(
                        expr,
                        errors,
                        format!("scatter replace mode rejects duplicate target index {out_linear}"),
                    );
                    return StaticValue::Unknown;
                }
            }
        }
    }
    StaticValue::Tensor(base.clone())
}

fn static_clamp(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::Tensor(input)),
        Some(StaticValue::Tensor(low)),
        Some(StaticValue::Tensor(high)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    let low_ok = low.shape.is_empty() || low.shape == input.shape;
    let high_ok = high.shape.is_empty() || high.shape == input.shape;
    if !low_ok || !high_ok {
        push_static_runtime_error(
            expr,
            errors,
            "clamp expects scalar bounds or matching-shape tensor bounds".to_string(),
        );
        return StaticValue::Unknown;
    }
    StaticValue::Tensor(input.clone())
}

fn static_einsum(
    args: &[StaticValue],
    expr: &deep::Expr,
    errors: &mut Vec<CheckError>,
) -> StaticValue {
    let (
        Some(StaticValue::String(equation)),
        Some(StaticValue::Tensor(lhs)),
        Some(StaticValue::Tensor(rhs)),
    ) = (args.first(), args.get(1), args.get(2))
    else {
        return StaticValue::Unknown;
    };
    let Some((inputs, output)) = equation.split_once("->") else {
        return StaticValue::Unknown;
    };
    let mut input_groups = inputs.split(',');
    let lhs_labels = input_groups
        .next()
        .unwrap_or_default()
        .chars()
        .collect::<Vec<_>>();
    let rhs_labels = input_groups
        .next()
        .unwrap_or_default()
        .chars()
        .collect::<Vec<_>>();
    if input_groups.next().is_some()
        || lhs_labels.len() != lhs.shape.len()
        || rhs_labels.len() != rhs.shape.len()
    {
        return StaticValue::Unknown;
    }
    let mut extents = HashMap::<char, usize>::new();
    for (label, extent) in lhs_labels.iter().zip(&lhs.shape) {
        if let Some(existing) = extents.insert(*label, *extent)
            && existing != *extent
        {
            push_static_runtime_error(
                expr,
                errors,
                format!("einsum label `{label}` has inconsistent extents"),
            );
            return StaticValue::Unknown;
        }
    }
    for (label, extent) in rhs_labels.iter().zip(&rhs.shape) {
        if let Some(existing) = extents.insert(*label, *extent)
            && existing != *extent
        {
            push_static_runtime_error(
                expr,
                errors,
                format!("einsum label `{label}` has inconsistent extents"),
            );
            return StaticValue::Unknown;
        }
    }
    let mut out_shape = Vec::new();
    for label in output.chars() {
        let Some(extent) = extents.get(&label).copied() else {
            return StaticValue::Unknown;
        };
        out_shape.push(extent);
    }
    StaticValue::Tensor(StaticTensor {
        shape: out_shape,
        int_values: None,
    })
}

fn normalize_static_axis(rank: usize, axis: i64) -> Option<usize> {
    let rank = rank as i64;
    let axis = if axis < 0 { rank + axis } else { axis };
    (0..rank).contains(&axis).then_some(axis as usize)
}

fn gather_result_shape(base: &[usize], indices: &[usize], axis: usize) -> Vec<usize> {
    let mut shape = Vec::with_capacity(base.len().saturating_sub(1) + indices.len());
    shape.extend_from_slice(&base[..axis]);
    shape.extend_from_slice(indices);
    shape.extend_from_slice(&base[axis + 1..]);
    shape
}

fn updates_shape_numel(shape: &[usize]) -> usize {
    shape.iter().copied().product::<usize>().max(1)
}

fn unravel_index(mut linear: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0; shape.len()];
    for dim in (0..shape.len()).rev() {
        out[dim] = linear % shape[dim];
        linear /= shape[dim];
    }
    out
}

fn ravel_index(indices: &[usize], shape: &[usize]) -> usize {
    let mut flat = 0usize;
    let mut stride = 1usize;
    for (index, extent) in indices.iter().zip(shape.iter()).rev() {
        flat += index * stride;
        stride *= extent;
    }
    flat
}

fn push_static_runtime_error(expr: &deep::Expr, errors: &mut Vec<CheckError>, message: String) {
    errors.push(CheckError::new(
        CheckErrorKind::Other,
        with_macro_provenance(expr, message),
        vec![],
    ));
}

fn annotate_phase0e_program(exprs: &[deep::Expr]) -> Vec<deep::Expr> {
    // Preserve historical behavior: build a fresh annotation state from
    // `builtin_env()` + an EMPTY ADT registry (no prelude registration).
    // This is asymmetric with `infer_phase0e_program_with_env` (which DOES
    // register prelude ADTs), but downstream tooling — the host pipeline's
    // `expr_type` reader, eval-result root expansion via
    // `extend_root_names_from_value` — depends on this asymmetric type
    // metadata shape. Phase C does NOT touch this; the
    // `_with_context` variant accepts a non-empty context and uses
    // library state as-is.
    let (mut env, mut vg) = builtins::builtin_env();
    let mut subst = Subst::new();
    let mut adt_reg = AdtRegistry::new();
    let mut declaration_errors = Vec::new();

    for expr in exprs {
        collect_declarations(
            expr,
            &mut env,
            &mut vg,
            &mut subst,
            &mut adt_reg,
            &mut declaration_errors,
        );
    }

    let mut annotated = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let mut step_errors = Vec::new();
        let mut typed_nodes = 0;
        let mut total_nodes = 0;
        infer_top_level(
            expr,
            &mut env,
            &mut vg,
            &mut subst,
            &adt_reg,
            &mut step_errors,
            &mut typed_nodes,
            &mut total_nodes,
        );

        annotated.push(annotate_expr_with_scope(expr, &env, &vg, &subst, &adt_reg));
    }

    annotated
}

/// Annotate `exprs` with inferred types, using `context` as the outer
/// scope. Library schemes are visible during inference; only the
/// new-code exprs are returned in the annotated result.
///
/// When the context is empty (`library_def_count() == 0`) this falls
/// back to [`annotate_phase0e_program`] to preserve the legacy
/// "no-prelude" annotation shape that downstream tooling depends on.
/// Once a non-empty library context is supplied, the library's prelude
/// ADTs and decls are visible during annotation.
fn annotate_phase0e_program_with_context(
    context: &TypeEnv,
    exprs: &[deep::Expr],
) -> Vec<deep::Expr> {
    if context.library_def_count() == 0 {
        return annotate_phase0e_program(exprs);
    }
    let mut state = context.inner().clone();
    let mut declaration_errors = Vec::new();

    for expr in exprs {
        collect_declarations(
            expr,
            &mut state.env,
            &mut state.var_gen,
            &mut state.subst,
            &mut state.adt_reg,
            &mut declaration_errors,
        );
    }

    let mut annotated = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let mut step_errors = Vec::new();
        let mut typed_nodes = 0;
        let mut total_nodes = 0;
        infer_top_level(
            expr,
            &mut state.env,
            &mut state.var_gen,
            &mut state.subst,
            &state.adt_reg,
            &mut step_errors,
            &mut typed_nodes,
            &mut total_nodes,
        );

        annotated.push(annotate_expr_with_scope(
            expr,
            &state.env,
            &state.var_gen,
            &state.subst,
            &state.adt_reg,
        ));
    }

    annotated
}

fn annotate_expr_with_scope(
    expr: &deep::Expr,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> deep::Expr {
    match expr {
        deep::Expr::Atom(_, _) => expr.clone(),
        deep::Expr::Map(map, span) => deep::Expr::Map(
            deep::MetaMap {
                entries: map
                    .entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            annotate_expr_with_scope(value, env, vg, subst, adt_reg),
                        )
                    })
                    .collect(),
            },
            *span,
        ),
        deep::Expr::MetaExpr(meta, span) => deep::Expr::MetaExpr(
            deep::MetaExpr {
                expr: Box::new(annotate_expr_with_scope(
                    &meta.expr, env, vg, subst, adt_reg,
                )),
                entries: meta
                    .entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            annotate_expr_with_scope(value, env, vg, subst, adt_reg),
                        )
                    })
                    .collect(),
            },
            *span,
        ),
        deep::Expr::List(list, span) => {
            if !matches!(list.elements.get(1), Some(deep::Expr::Map(_, _))) {
                return deep::Expr::List(
                    deep::List {
                        elements: list
                            .elements
                            .iter()
                            .map(|element| {
                                annotate_expr_with_scope(element, env, vg, subst, adt_reg)
                            })
                            .collect(),
                    },
                    *span,
                );
            }

            let tag = get_tag(list);
            let (annotated_children, fn_ty_override) = match tag {
                Some("fn") => {
                    let (kids, fn_ty) = annotate_fn_children(list, env, vg, subst, adt_reg);
                    (kids, Some(fn_ty))
                }
                Some("let") => (annotate_let_children(list, env, vg, subst, adt_reg), None),
                Some("match") => (annotate_match_children(list, env, vg, subst, adt_reg), None),
                _ => (
                    children(list)
                        .iter()
                        .map(|child| annotate_expr_with_scope(child, env, vg, subst, adt_reg))
                        .collect(),
                    None,
                ),
            };

            let mut elements = vec![
                list.elements[0].clone(),
                annotated_meta_map_with_override(
                    list,
                    expr,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    fn_ty_override,
                ),
            ];
            elements.extend(annotated_children);
            deep::Expr::List(deep::List { elements }, *span)
        }
    }
}

fn annotate_fn_children(
    list: &deep::List,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> (Vec<deep::Expr>, Type) {
    let kids = children(list);
    if kids.is_empty() {
        return (vec![], Type::Error);
    }

    let fn_ty = infer_expr_in_scope(
        &deep::Expr::List(list.clone(), span_of_list(list)),
        env,
        vg,
        subst,
        adt_reg,
    );
    let resolved_fn_ty = subst.apply(&fn_ty);
    let param_types = match &resolved_fn_ty {
        Type::Fn(args, _) => args.clone(),
        _ => Vec::new(),
    };

    let mut param_vg = vg.clone();
    let raw_params = extract_params(&kids[0], &mut param_vg, adt_reg);
    let mut fn_env = env.clone();
    for (index, (name, maybe_ty)) in raw_params.iter().enumerate() {
        let ty = maybe_ty
            .clone()
            .or_else(|| param_types.get(index).cloned())
            .unwrap_or(Type::Error);
        fn_env.bind(name.clone(), Scheme::mono(ty));
    }

    let mut result = vec![annotate_expr_with_scope(&kids[0], env, vg, subst, adt_reg)];
    if let Some(body) = kids.get(1) {
        result.push(annotate_expr_with_scope(body, &fn_env, vg, subst, adt_reg));
    }
    (result, resolved_fn_ty)
}

fn annotate_let_children(
    list: &deep::List,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Vec<deep::Expr> {
    let kids = children(list);
    if kids.len() < 2 {
        return kids.to_vec();
    }

    let mut let_env = env.clone();
    let annotated_bind = if let deep::Expr::List(bind_list, bind_span) = &kids[0] {
        let bind_kids = children(bind_list);
        let mut bind_elements = vec![bind_list.elements[0].clone(), bind_list.elements[1].clone()];
        let mut i = 0;
        while i + 1 < bind_kids.len() {
            bind_elements.push(bind_kids[i].clone());
            let value = annotate_expr_with_scope(&bind_kids[i + 1], &let_env, vg, subst, adt_reg);
            let value_ty = infer_expr_in_scope(&bind_kids[i + 1], &let_env, vg, subst, adt_reg);
            if let Some(name) = symbol_name(&bind_kids[i]) {
                let_env.bind(name.to_string(), let_env.generalize(&value_ty, subst));
            }
            bind_elements.push(value);
            i += 2;
        }
        deep::Expr::List(
            deep::List {
                elements: bind_elements,
            },
            *bind_span,
        )
    } else {
        annotate_expr_with_scope(&kids[0], env, vg, subst, adt_reg)
    };

    vec![
        annotated_bind,
        annotate_expr_with_scope(&kids[1], &let_env, vg, subst, adt_reg),
    ]
}

fn annotate_match_children(
    list: &deep::List,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Vec<deep::Expr> {
    let kids = children(list);
    if kids.is_empty() {
        return vec![];
    }

    let scrutinee = annotate_expr_with_scope(&kids[0], env, vg, subst, adt_reg);
    let scrutinee_ty = infer_expr_in_scope(&kids[0], env, vg, subst, adt_reg);
    let mut result = vec![scrutinee];

    for arm in &kids[1..] {
        if let deep::Expr::List(arm_list, arm_span) = arm
            && get_tag(arm_list) == Some("arm")
        {
            let arm_kids = children(arm_list);
            let mut arm_env = env.clone();
            let mut pattern_vg = vg.clone();
            let mut pattern_subst = subst.clone();
            let mut pattern_errors = Vec::new();
            let mut covered = Vec::new();
            let mut wildcard = false;
            if let Some(pattern) = arm_kids.first() {
                pattern_bindings(
                    pattern,
                    &scrutinee_ty,
                    &mut arm_env,
                    &mut pattern_vg,
                    &mut pattern_subst,
                    adt_reg,
                    &mut pattern_errors,
                    &mut covered,
                    &mut wildcard,
                );
            }

            let mut elements = vec![arm_list.elements[0].clone(), arm_list.elements[1].clone()];
            if let Some(pattern) = arm_kids.first() {
                elements.push(annotate_expr_with_scope(pattern, env, vg, subst, adt_reg));
            }
            if let Some(guard) = arm_kids.get(1) {
                elements.push(annotate_expr_with_scope(
                    guard, &arm_env, vg, subst, adt_reg,
                ));
            }
            if let Some(body) = arm_kids.get(2) {
                elements.push(annotate_expr_with_scope(body, &arm_env, vg, subst, adt_reg));
            }
            result.push(deep::Expr::List(deep::List { elements }, *arm_span));
            continue;
        }
        result.push(annotate_expr_with_scope(arm, env, vg, subst, adt_reg));
    }

    result
}

fn annotated_meta_map_with_override(
    list: &deep::List,
    expr: &deep::Expr,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
    precomputed_ty: Option<Type>,
) -> deep::Expr {
    let meta_span = match list.elements.get(1) {
        Some(deep::Expr::Map(_, span)) => *span,
        _ => span_of_expr(expr),
    };
    let mut entries = get_meta(list)
        .map(|meta| meta.entries.clone())
        .unwrap_or_default();

    let ty_for_meta = if let Some(tag) = get_tag(list) {
        match (tag, precomputed_ty) {
            ("fn", Some(ty)) => Some(ty),
            (t, _) if should_attach_type_metadata(t) => {
                Some(infer_expr_in_scope(expr, env, vg, subst, adt_reg))
            }
            _ => None,
        }
    } else {
        None
    };

    if let Some(ty) = ty_for_meta
        && !matches!(ty, Type::Error)
    {
        let ty_expr = type_to_deep_expr(&ty);
        if let Some((_, existing)) = entries.iter_mut().find(|(key, _)| key == "type") {
            *existing = ty_expr;
        } else {
            entries.push(("type".to_string(), ty_expr));
        }
    }

    deep::Expr::Map(deep::MetaMap { entries }, meta_span)
}

fn infer_expr_in_scope(
    expr: &deep::Expr,
    env: &Env,
    vg: &VarGen,
    _subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Type {
    let mut env = env.clone();
    let mut vg = vg.clone();
    let mut subst = Subst::new();
    let mut errors = Vec::new();
    let mut typed_nodes = 0;
    let mut total_nodes = 0;
    let ty = infer_expr(
        expr,
        &mut env,
        &mut vg,
        &mut subst,
        adt_reg,
        &mut errors,
        &mut typed_nodes,
        &mut total_nodes,
    );
    subst.apply(&ty)
}

fn should_attach_type_metadata(tag: &str) -> bool {
    !matches!(
        tag,
        "module"
            | "import"
            | "import-all"
            | "export"
            | "let"
            | "fn"
            | "var"
            | "tuple"
            | "tuple-get"
            | "defsig"
            | "deftype"
            | "typealias"
            | "variant"
            | "field"
            | "defdim"
            | "params"
            | "bind"
            | "kv"
            | "arm"
            | "effects"
            | "resource"
            | "pat-var"
            | "pat-lit"
            | "pat-ctor"
            | "pat-tuple"
            | "pat-record"
            | "pat-wild"
            | "pat-as"
            | "t-prim"
            | "t-fn"
            | "t-tensor"
            | "t-adt"
            | "t-var"
            | "t-unit"
            | "t-tuple"
            | "d-name"
            | "d-var"
            | "d-lit"
    )
}

fn type_to_deep_expr(ty: &Type) -> deep::Expr {
    match ty {
        Type::Prim(prim) => node_expr("t-prim", vec![symbol_expr(prim.name())]),
        Type::Fn(args, ret) => {
            let mut children: Vec<deep::Expr> = args.iter().map(type_to_deep_expr).collect();
            children.push(type_to_deep_expr(ret));
            node_expr("t-fn", children)
        }
        Type::Tensor(dims, prim) => {
            let mut children: Vec<deep::Expr> = dims.iter().map(dim_to_deep_expr).collect();
            children.push(type_to_deep_expr(&Type::Prim(*prim)));
            node_expr("t-tensor", children)
        }
        Type::Adt(name, args) => {
            let mut children = vec![symbol_expr(name)];
            children.extend(args.iter().map(type_to_deep_expr));
            node_expr("t-adt", children)
        }
        Type::Var(var) => node_expr("t-var", vec![symbol_expr(&format!("t{}", var.0))]),
        Type::Tuple(types) => node_expr("t-tuple", types.iter().map(type_to_deep_expr).collect()),
        Type::Unit => node_expr("t-unit", vec![]),
        Type::Error => node_expr("t-var", vec![symbol_expr("_")]),
    }
}

fn dim_to_deep_expr(dim: &Dim) -> deep::Expr {
    match dim {
        Dim::Name(name) => node_expr("d-name", vec![symbol_expr(name)]),
        Dim::Var(var) => node_expr("d-var", vec![symbol_expr(&format!("d{}", var.0))]),
        Dim::Lit(value) => node_expr(
            "d-lit",
            vec![deep::Expr::Atom(deep::Atom::Int(*value), zero_span())],
        ),
        Dim::Wildcard => node_expr("d-name", vec![symbol_expr("*")]),
    }
}

fn node_expr(tag: &str, children: Vec<deep::Expr>) -> deep::Expr {
    let mut elements = vec![
        symbol_expr(tag),
        deep::Expr::Map(deep::MetaMap::default(), zero_span()),
    ];
    elements.extend(children);
    deep::Expr::List(deep::List { elements }, zero_span())
}

fn symbol_expr(name: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Symbol(name.to_string()), zero_span())
}

fn zero_span() -> Span {
    Span::new(0, 0)
}

fn span_of_expr(expr: &deep::Expr) -> Span {
    match expr {
        deep::Expr::Atom(_, span)
        | deep::Expr::List(_, span)
        | deep::Expr::Map(_, span)
        | deep::Expr::MetaExpr(_, span) => *span,
    }
}

fn span_of_list(list: &deep::List) -> Span {
    list.elements
        .first()
        .map(span_of_expr)
        .unwrap_or_else(zero_span)
}

fn phase0e_builtin_name(list: &deep::List) -> Option<&str> {
    let func_expr = list.elements.get(2)?;
    let func_list = match func_expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    match (func_list.elements.first(), func_list.elements.get(2)) {
        (
            Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)),
            Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)),
        ) if tag == "var" => Some(name.as_str()),
        _ => None,
    }
}

fn is_phase0e_shape_sensitive_builtin(name: &str) -> bool {
    matches!(
        name,
        "matmul"
            | "softmax"
            | "mean"
            | "layer_norm"
            | "conv2d"
            | "sum"
            | "max_reduce"
            | "min_reduce"
            | "prod_reduce"
            | "argmax_reduce"
            | "argmin_reduce"
            | "reshape"
            | "permute"
            | "expand"
            | "pad"
            | "shrink"
            | "stride"
    )
}

fn expr_type_expr(expr: &deep::Expr, type_env: &Phase0eTypeEnv) -> Option<deep::Expr> {
    match expr {
        deep::Expr::List(list, _) => {
            if let Some(meta) = get_meta(list)
                && let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type")
            {
                return Some(ty.clone());
            }
            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                return type_env.get(name).cloned();
            }
            None
        }
        deep::Expr::MetaExpr(meta, _) => expr_type_expr(&meta.expr, type_env),
        _ => None,
    }
}

fn extend_phase0e_env_with_fn_params(
    fn_list: &deep::List,
    type_env: &Phase0eTypeEnv,
) -> Phase0eTypeEnv {
    let mut scoped = type_env.clone();
    let Some(params_expr) = children(fn_list).first() else {
        return scoped;
    };
    let deep::Expr::List(params_list, _) = params_expr else {
        return scoped;
    };
    if get_tag(params_list) != Some("params") {
        return scoped;
    }
    for param in children(params_list) {
        let deep::Expr::List(param_list, _) = param else {
            continue;
        };
        let Some(name) = param_list.elements.first().and_then(symbol_name) else {
            continue;
        };
        let Some(meta) = get_meta(param_list) else {
            continue;
        };
        let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type") else {
            continue;
        };
        scoped.insert(name.to_string(), ty.clone());
    }
    scoped
}

fn expr_tensor_type_is_concrete(expr: &deep::Expr, type_env: &Phase0eTypeEnv) -> bool {
    expr_type_expr(expr, type_env)
        .map(|ty| type_expr_is_phase0e_concrete(&ty))
        .unwrap_or(false)
}

fn validate_phase0e_builtin_symbolic_requirements(
    list: &deep::List,
    func_name: &str,
    type_env: &Phase0eTypeEnv,
    errors: &mut Vec<CheckError>,
) {
    match func_name {
        "conv2d" => {
            if !app_result_type_is_concrete(list) {
                errors.push(CheckError::new(
                    CheckErrorKind::Other,
                    "Phase 0e builtin `conv2d` requires concrete output tensor dimensions"
                        .to_string(),
                    vec!["Use concrete d-lit dimensions for Phase 0e lowering".to_string()],
                ));
            }
            for arg in list.elements.iter().skip(3).take(2) {
                if !expr_tensor_type_is_concrete(arg, type_env) {
                    errors.push(CheckError::new(
                        CheckErrorKind::Other,
                        "Phase 0e builtin `conv2d` requires concrete tensor argument metadata"
                            .to_string(),
                        vec!["Use concrete d-lit dimensions for Phase 0e lowering".to_string()],
                    ));
                    break;
                }
            }
        }
        "mean"
            if phase0e_builtin_axis_dim(list, type_env, 0, 1) == Some(DeepDimKind::NonConcrete) =>
        {
            errors.push(CheckError::new(
                CheckErrorKind::Other,
                "Phase 0e builtin `mean` requires a concrete reduced axis extent".to_string(),
                vec!["Use a concrete d-lit dimension on the reduced axis".to_string()],
            ));
        }
        "layer_norm" => {
            let x_dims = list
                .elements
                .get(3)
                .and_then(|expr| expr_type_expr(expr, type_env))
                .and_then(|ty| tensor_dims_from_type_expr(&ty));
            if matches!(
                x_dims.as_ref().and_then(|dims| dims.last()),
                Some(DeepDimKind::NonConcrete)
            ) {
                errors.push(CheckError::new(
                    CheckErrorKind::Other,
                    "Phase 0e builtin `layer_norm` requires a concrete normalized axis extent"
                        .to_string(),
                    vec!["Use a concrete d-lit dimension for the final axis".to_string()],
                ));
            }
        }
        _ => {}
    }
}

fn phase0e_builtin_axis_dim(
    list: &deep::List,
    type_env: &Phase0eTypeEnv,
    tensor_arg_index: usize,
    axis_arg_index: usize,
) -> Option<DeepDimKind> {
    let tensor_dims = list
        .elements
        .get(3 + tensor_arg_index)
        .and_then(|expr| expr_type_expr(expr, type_env))
        .and_then(|ty| tensor_dims_from_type_expr(&ty))?;
    let axis = list
        .elements
        .get(3 + axis_arg_index)
        .and_then(extract_axis_literal)?;
    tensor_dims.get(axis).copied()
}

fn app_result_type_is_concrete(list: &deep::List) -> bool {
    get_meta(list)
        .and_then(|meta| meta.entries.iter().find(|(k, _)| k == "type"))
        .map(|(_, ty)| type_expr_is_phase0e_concrete(ty))
        .unwrap_or(false)
}

fn extract_axis_literal(expr: &deep::Expr) -> Option<usize> {
    extract_int_literal(expr).and_then(|axis| (axis >= 0).then_some(axis as usize))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepDimKind {
    Lit(i64),
    NonConcrete,
}

fn tensor_dims_from_type_expr(expr: &deep::Expr) -> Option<Vec<DeepDimKind>> {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(list) != Some("t-tensor") {
        return None;
    }
    let kids = children(list);
    if kids.is_empty() {
        return None;
    }
    let mut dims = Vec::new();
    for kid in &kids[..kids.len().saturating_sub(1)] {
        dims.push(match kid {
            deep::Expr::List(dim_list, _) if get_tag(dim_list) == Some("d-lit") => {
                match children(dim_list).first() {
                    Some(deep::Expr::Atom(deep::Atom::Int(n), _)) => DeepDimKind::Lit(*n),
                    _ => DeepDimKind::NonConcrete,
                }
            }
            _ => DeepDimKind::NonConcrete,
        });
    }
    Some(dims)
}

fn type_expr_is_phase0e_concrete(expr: &deep::Expr) -> bool {
    match expr {
        deep::Expr::List(list, _) if get_tag(list) == Some("t-prim") => true,
        _ => tensor_dims_from_type_expr(expr)
            .map(|dims| dims.iter().all(|d| matches!(d, DeepDimKind::Lit(_))))
            .unwrap_or(false),
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn get_tag(list: &deep::List) -> Option<&str> {
    if let Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)) = list.elements.first() {
        Some(tag.as_str())
    } else {
        None
    }
}

fn children(list: &deep::List) -> &[deep::Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn is_numeric_literal_expr(expr: &deep::Expr) -> bool {
    match expr {
        deep::Expr::Atom(deep::Atom::Float(_) | deep::Atom::Int(_), _) => true,
        deep::Expr::List(list, _) if get_tag(list) == Some("lit") => matches!(
            list.elements.get(2),
            Some(deep::Expr::Atom(
                deep::Atom::Float(_) | deep::Atom::Int(_),
                _
            ))
        ),
        _ => false,
    }
}

/// Get metadata map from element[1] of a list.
fn get_meta(list: &deep::List) -> Option<&deep::MetaMap> {
    if list.elements.len() > 1
        && let deep::Expr::Map(meta, _) = &list.elements[1]
    {
        return Some(meta);
    }
    None
}

/// Extract a symbol name from an Expr.
fn symbol_name(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Atom(deep::Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn with_macro_provenance(expr: &deep::Expr, message: String) -> String {
    let Some(source) = macro_source(expr) else {
        return message;
    };
    format!("{message} (in expansion of {source})")
}

fn check_error_kind_from_type_error_kind(kind: &TypeErrorKind) -> CheckErrorKind {
    match kind {
        TypeErrorKind::TypeMismatch => CheckErrorKind::TypeMismatch,
        TypeErrorKind::PrecisionMismatch => CheckErrorKind::PrecisionMismatch,
        TypeErrorKind::DimensionMismatch => CheckErrorKind::DimensionMismatch,
        TypeErrorKind::ArityMismatch => CheckErrorKind::ArityMismatch,
        TypeErrorKind::OccursCheck => CheckErrorKind::OccursCheck,
        TypeErrorKind::NotAFunction => CheckErrorKind::NotAFunction,
    }
}

fn collection_helper_type_error(
    expr: &deep::Expr,
    helper: &str,
    contract: &str,
    te: TypeError,
) -> CheckError {
    let kind = check_error_kind_from_type_error_kind(&te.kind);
    let suggestions = match kind {
        CheckErrorKind::PrecisionMismatch => vec!["Insert explicit cast".to_string()],
        _ => vec![],
    };
    CheckError::new(
        kind,
        with_macro_provenance(expr, format!("{helper} {contract}; {}", te.message)),
        suggestions,
    )
}

fn extract_string_literal(expr: &deep::Expr) -> Option<String> {
    match expr {
        deep::Expr::Atom(deep::Atom::Str(value), _) => Some(value.clone()),
        deep::Expr::List(list, _) if get_tag(list) == Some("lit") => {
            children(list).first().and_then(|child| match child {
                deep::Expr::Atom(deep::Atom::Str(value), _) => Some(value.clone()),
                _ => None,
            })
        }
        _ => None,
    }
}

/// Narrow `Dim::Wildcard` slots in `ty` against the matching positions in
/// `template`, replacing each Wildcard with the template's concrete dim
/// (`Dim::Lit` or `Dim::Name`) where one is available. Used after a defsig
/// unify to ensure the scheme registered for callers reflects the declared
/// concrete shape rather than the body's permissive wildcards (#39).
///
/// This is structural and conservative: it only walks shapes that match
/// (same rank for tensors, same arity for fn/tuple/adt), and only narrows
/// Wildcard → concrete. Other dim shapes (Var, existing Lit/Name) are
/// preserved. If shapes don't line up, the input is returned unchanged so
/// genuine type errors flagged by `unify` aren't masked.
fn narrow_wildcards_with(ty: &Type, template: &Type) -> Type {
    match (ty, template) {
        (Type::Tensor(dims, prec), Type::Tensor(tmpl_dims, _)) if dims.len() == tmpl_dims.len() => {
            let new_dims = dims
                .iter()
                .zip(tmpl_dims.iter())
                .map(|(d, t)| match (d, t) {
                    // Only narrow Wildcard → concrete literal. Substituting
                    // Dim::Name or Dim::Var into the generalized scheme can
                    // bind a function-parameter-bound name (`tensor[n, f32]`
                    // where n is the def's int64 param) into the type, which
                    // overflows the inference stack when the resulting
                    // scheme is later instantiated. Concrete literals are
                    // self-contained and safe.
                    (Dim::Wildcard, Dim::Lit(_)) => t.clone(),
                    _ => d.clone(),
                })
                .collect();
            Type::Tensor(new_dims, *prec)
        }
        (Type::Fn(args, ret), Type::Fn(t_args, t_ret)) if args.len() == t_args.len() => {
            let new_args = args
                .iter()
                .zip(t_args.iter())
                .map(|(a, t)| narrow_wildcards_with(a, t))
                .collect();
            let new_ret = Box::new(narrow_wildcards_with(ret, t_ret));
            Type::Fn(new_args, new_ret)
        }
        (Type::Tuple(ts), Type::Tuple(t_ts)) if ts.len() == t_ts.len() => {
            let new_ts = ts
                .iter()
                .zip(t_ts.iter())
                .map(|(t, tt)| narrow_wildcards_with(t, tt))
                .collect();
            Type::Tuple(new_ts)
        }
        (Type::Adt(n, args), Type::Adt(_, t_args)) if args.len() == t_args.len() => {
            let new_args = args
                .iter()
                .zip(t_args.iter())
                .map(|(a, t)| narrow_wildcards_with(a, t))
                .collect();
            Type::Adt(n.clone(), new_args)
        }
        _ => ty.clone(),
    }
}

fn tensor_concat_result_type(element_ty: &Type) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = element_ty else {
        return Err(format!(
            "concat expects List[tensor[...]] for tensor concatenation, got {element_ty}"
        ));
    };
    if dims.is_empty() {
        return Err("concat expects tensor inputs with at least one axis".to_string());
    }
    let mut out_dims = dims.clone();
    let last_axis = out_dims.len() - 1;
    out_dims[last_axis] = Dim::Wildcard;
    Ok(Type::Tensor(out_dims, *precision))
}

fn infer_gather_result_type(
    tensor_ty: &Type,
    indices_ty: &Type,
    axis: usize,
) -> Result<Type, String> {
    let Type::Tensor(tensor_dims, tensor_precision) = tensor_ty else {
        return Err(format!("gather expects tensor input, got {tensor_ty}"));
    };
    let Type::Tensor(index_dims, index_precision) = indices_ty else {
        return Err(format!(
            "gather expects integer tensor indices, got {indices_ty}"
        ));
    };
    if !index_precision.is_integer() {
        return Err(format!(
            "gather expects integer tensor indices, got tensor[..., {}]",
            index_precision.name()
        ));
    }
    if axis >= tensor_dims.len() {
        return Err(format!(
            "gather axis {axis} out of bounds for rank {}",
            tensor_dims.len()
        ));
    }
    let mut out_dims = tensor_dims[..axis].to_vec();
    out_dims.extend(index_dims.clone());
    out_dims.extend_from_slice(&tensor_dims[axis + 1..]);
    Ok(Type::Tensor(out_dims, *tensor_precision))
}

fn infer_trace_result_type(tensor_ty: &Type, axis1: usize, axis2: usize) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = tensor_ty else {
        return Err(format!("trace expects tensor input, got {tensor_ty}"));
    };
    if axis1 >= dims.len() || axis2 >= dims.len() || axis1 == axis2 {
        return Err(format!(
            "trace expects distinct in-bounds axes, got {axis1} and {axis2} for rank {}",
            dims.len()
        ));
    }
    let out_dims = dims
        .iter()
        .enumerate()
        .filter_map(|(index, dim)| ((index != axis1) && (index != axis2)).then_some(dim.clone()))
        .collect();
    Ok(Type::Tensor(out_dims, *precision))
}

fn infer_diagonal_result_type(
    tensor_ty: &Type,
    axis1: usize,
    axis2: usize,
) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = tensor_ty else {
        return Err(format!("diagonal expects tensor input, got {tensor_ty}"));
    };
    if axis1 >= dims.len() || axis2 >= dims.len() || axis1 == axis2 {
        return Err(format!(
            "diagonal expects distinct in-bounds axes, got {axis1} and {axis2} for rank {}",
            dims.len()
        ));
    }
    let diag_dim = match (&dims[axis1], &dims[axis2]) {
        (Dim::Lit(lhs), Dim::Lit(rhs)) if lhs == rhs => Dim::Lit(*lhs),
        _ => Dim::Wildcard,
    };
    let mut out_dims = Vec::with_capacity(dims.len() - 1);
    for (index, dim) in dims.iter().enumerate() {
        if index == axis1 {
            out_dims.push(diag_dim.clone());
        } else if index != axis2 {
            out_dims.push(dim.clone());
        }
    }
    Ok(Type::Tensor(out_dims, *precision))
}

fn macro_source(expr: &deep::Expr) -> Option<String> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    let meta = get_meta(list)?;
    let source = meta
        .entries
        .iter()
        .find(|(key, _)| key == "source")
        .map(|(_, value)| value)?;
    let rendered = chelis_deep::printer::print_canonical(std::slice::from_ref(source));
    Some(rendered.replace('\n', " ").trim().to_string())
}

fn resolve_type_aliases(ty: &Type, adt_reg: &AdtRegistry) -> Type {
    let mut seen = HashSet::new();
    resolve_type_aliases_inner(ty, adt_reg, &mut seen)
}

fn resolve_type_aliases_inner(
    ty: &Type,
    adt_reg: &AdtRegistry,
    seen: &mut HashSet<String>,
) -> Type {
    match ty {
        Type::Adt(name, args) => {
            let resolved_args: Vec<Type> = args
                .iter()
                .map(|arg| resolve_type_aliases_inner(arg, adt_reg, seen))
                .collect();

            if seen.contains(name) {
                return Type::Adt(name.clone(), resolved_args);
            }

            if let Some(expanded) = adt_reg.instantiate_alias(name, &resolved_args) {
                seen.insert(name.clone());
                let resolved = resolve_type_aliases_inner(&expanded, adt_reg, seen);
                seen.remove(name);
                resolved
            } else {
                Type::Adt(name.clone(), resolved_args)
            }
        }
        Type::Fn(args, ret) => Type::Fn(
            args.iter()
                .map(|a| resolve_type_aliases_inner(a, adt_reg, seen))
                .collect(),
            Box::new(resolve_type_aliases_inner(ret, adt_reg, seen)),
        ),
        Type::Tuple(ts) => Type::Tuple(
            ts.iter()
                .map(|t| resolve_type_aliases_inner(t, adt_reg, seen))
                .collect(),
        ),
        _ => ty.clone(),
    }
}

fn deep_type_to_resolved_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    tvar_map: &mut HashMap<String, TypeVar>,
) -> Type {
    let ty = deep_type_to_type(expr, vg, tvar_map);
    resolve_type_aliases(&ty, adt_reg)
}

// ── Declaration collection (first pass) ──────────────────────────

fn collect_declarations(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &mut AdtRegistry,
    _errors: &mut Vec<CheckError>,
) {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return,
    };

    let tag = match get_tag(list) {
        Some(t) => t,
        None => return,
    };

    let kids = children(list);

    match tag {
        "deftype" => {
            let ctors = adt_reg.register_deftype(kids, vg);
            for (name, scheme) in ctors {
                env.bind(name, scheme);
            }
        }
        "defsig" => {
            // (defsig {} name type_expr)
            if kids.len() >= 2
                && let Some(name) = symbol_name(&kids[0])
            {
                let ty = deep_type_to_resolved_type(&kids[1], vg, adt_reg, &mut HashMap::new());
                let scheme = env.generalize(&ty, subst);
                env.bind(name.to_string(), scheme);
            }
        }
        "typealias" => {
            // (typealias {} Name (params...) type_expr)
            if kids.len() >= 3
                && let Some(name) = symbol_name(&kids[0])
            {
                let params = match &kids[1] {
                    deep::Expr::List(list, _) => list
                        .elements
                        .iter()
                        .filter_map(symbol_name)
                        .map(str::to_string)
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };

                let mut tvar_map = HashMap::new();
                let mut param_vars = Vec::with_capacity(params.len());
                for param in &params {
                    let tv = vg.fresh_tvar();
                    tvar_map.insert(param.clone(), tv);
                    param_vars.push(tv);
                }

                let aliased_ty = deep_type_to_type(&kids[2], vg, &mut tvar_map);
                adt_reg.register_alias(name.to_string(), params, param_vars, aliased_ty);
            }
        }
        _ => {}
    }
}

// ── Top-level inference (second pass) ────────────────────────────

#[allow(clippy::too_many_arguments)]
fn infer_top_level(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return,
    };

    let tag = match get_tag(list) {
        Some(t) => t,
        None => return,
    };

    // Skip deftype/defsig/typealias (already processed in first pass)
    if tag == "deftype" || tag == "defsig" || tag == "typealias" {
        return;
    }

    let kids = children(list);

    if tag == "def" && kids.len() >= 2 {
        let name = match symbol_name(&kids[0]) {
            Some(n) => n.to_string(),
            None => return,
        };

        // Save declared type from defsig BEFORE inferring (it may get overwritten)
        let declared_ty = env.lookup(&name).map(|s| {
            let s = s.clone();
            env.instantiate(&s, vg)
        });

        let body_ty = infer_expr(
            &kids[1],
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );

        // Enforce defsig: body must match declared signature.
        //
        // After unify succeeds, the body's inferred type may still carry
        // `Dim::Wildcard` slots (e.g. from `pad_sequences_to`, `concat`,
        // `to_tensor`) because `unify_dim` treats Wildcard as a permissive
        // matches-anything sentinel. Generalizing the raw body would expose
        // those wildcards to callers, who would then silently accept any
        // concrete dim. Narrow the body's Wildcards against the declared
        // template so the scheme registered for callers reflects the
        // declared concrete shape (#39).
        let scheme_body = if let Some(decl_ty) = declared_ty {
            if let Err(_te) = unify(&body_ty, &decl_ty, subst) {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("def '{}' body doesn't match declared signature", name),
                    vec![],
                ));
            }
            let resolved_body = subst.apply(&body_ty);
            let resolved_decl = subst.apply(&decl_ty);
            narrow_wildcards_with(&resolved_body, &resolved_decl)
        } else {
            body_ty
        };

        let scheme = env.generalize(&scheme_body, subst);
        env.bind(name, scheme);
    } else {
        // Any other top-level expression
        let _ = infer_expr(
            expr,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }
}

// ── Core inference ───────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn infer_expr(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    *total_nodes += 1;

    let result = match expr {
        deep::Expr::Atom(atom, _) => infer_atom(atom),
        deep::Expr::List(list, _) => {
            let tag = get_tag(list);
            match tag {
                Some("var") => infer_var(list, env, vg, subst, errors),
                Some("lit") => infer_lit(list, vg, adt_reg),
                Some("app") => infer_app(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("fn") => infer_fn(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("let") => infer_let(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("if") => infer_if(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("match") => infer_match(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("pipe") => infer_pipe(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("tuple") => infer_tuple(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("tuple-get") => infer_tuple_get(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("cast") => infer_cast(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("grad") => infer_grad(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("vmap") => infer_vmap(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("def") => infer_def(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("defsig") => {
                    // Already handled in first pass
                    Type::Unit
                }
                Some("deftype") | Some("typealias") => {
                    // Already handled in first pass
                    Type::Unit
                }
                Some("par") => {
                    // par: evaluate all children, return type of last (v1: sequential)
                    let kids = children(list);
                    let mut last_ty = Type::Unit;
                    for kid in kids {
                        last_ty = infer_expr(
                            kid,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        );
                    }
                    last_ty
                }
                Some("realize") => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        infer_expr(
                            inner,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        )
                    } else {
                        Type::Error
                    }
                }
                Some("copy") => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        let inner_ty = infer_expr(
                            inner,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        );
                        let resolved = subst.apply(&inner_ty);
                        match resolved {
                            Type::Tensor(_, _) | Type::Error => inner_ty,
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    format!("copy requires tensor input, got {resolved}"),
                                    vec!["Wrap only tensor values in copy".to_string()],
                                ));
                                Type::Error
                            }
                        }
                    } else {
                        Type::Error
                    }
                }
                Some("borrow") => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        let inner_ty = infer_expr(
                            inner,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        );
                        let resolved = subst.apply(&inner_ty);
                        match resolved {
                            Type::Tensor(_, _) | Type::Adt(_, _) | Type::Tuple(_) | Type::Error => {
                                inner_ty
                            }
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    format!("borrow requires tensor or tensor-carrying input, got {resolved}"),
                                    vec!["Use `&x` only with tensor values".to_string()],
                                ));
                                Type::Error
                            }
                        }
                    } else {
                        Type::Error
                    }
                }
                _ => {
                    // Unknown tag -- try to infer children
                    Type::Error
                }
            }
        }
        deep::Expr::Map(_, _) => Type::Unit,
        deep::Expr::MetaExpr(meta, _) => infer_expr(
            &meta.expr,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        ),
    };

    if !matches!(result, Type::Error) {
        *typed_nodes += 1;
    }

    result
}

fn infer_atom(atom: &deep::Atom) -> Type {
    match atom {
        deep::Atom::Int(_) => Type::Prim(Prim::Int32),
        deep::Atom::Float(_) => Type::Prim(Prim::F32),
        deep::Atom::Bool(_) => Type::Prim(Prim::Bool),
        deep::Atom::Str(_) => Type::Prim(Prim::String),
        deep::Atom::Symbol(_) | deep::Atom::Keyword(_) => Type::Error,
    }
}

fn infer_var(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    let kids = children(list);
    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
        if let Some(scheme) = env
            .lookup(name)
            .or_else(|| env.lookup_terminal_unique(name))
        {
            let scheme = scheme.clone();
            let ty = env.instantiate(&scheme, vg);
            subst.apply(&ty)
        } else {
            errors.push(CheckError::new(
                CheckErrorKind::UnboundVariable,
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("unbound variable: {name}"),
                ),
                vec![format!("Check spelling of '{}'", name)],
            ));
            Type::Error
        }
    } else {
        Type::Error
    }
}

fn infer_lit(list: &deep::List, vg: &mut VarGen, adt_reg: &AdtRegistry) -> Type {
    let meta = get_meta(list);
    let kids = children(list);

    // Check metadata for type annotation
    if let Some(meta) = meta {
        for (key, val) in &meta.entries {
            if key == "type" {
                return deep_type_to_resolved_type(val, vg, adt_reg, &mut HashMap::new());
            }
        }
    }

    // Fall back to value-based defaults
    if let Some(val) = kids.first() {
        match val {
            deep::Expr::Atom(deep::Atom::Int(_), _) => Type::Prim(Prim::Int32),
            deep::Expr::Atom(deep::Atom::Float(_), _) => Type::Prim(Prim::F32),
            deep::Expr::Atom(deep::Atom::Bool(_), _) => Type::Prim(Prim::Bool),
            deep::Expr::Atom(deep::Atom::Str(_), _) => Type::Prim(Prim::String),
            _ => Type::Error,
        }
    } else {
        Type::Error
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    // Check if func is a comparison op (for special return type handling)
    let func_name = if let deep::Expr::List(flist, _) = &kids[0] {
        if get_tag(flist) == Some("var") {
            children(flist)
                .first()
                .and_then(|e| symbol_name(e))
                .map(|s| s.to_string())
        } else {
            None
        }
    } else {
        None
    };

    if matches!(func_name.as_deref(), Some("permute")) {
        return infer_permute_app(
            list,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    if matches!(func_name.as_deref(), Some("reshape")) {
        return infer_reshape_app(
            list,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }

    let ctor_lookup_name = func_name.as_ref().and_then(|fname| {
        adt_reg
            .lookup_variant(fname)
            .map(|_| fname.clone())
            .or_else(|| {
                adt_reg
                    .lookup_variant_terminal_unique(fname)
                    .map(|(_, variant)| variant.name.clone())
            })
    });

    if let Some(ref fname) = ctor_lookup_name
        && let Some((_adt_name, variant)) = adt_reg
            .lookup_variant(fname)
            .or_else(|| adt_reg.lookup_variant_terminal_unique(fname))
        && !variant.fields.is_empty()
        && variant
            .fields
            .iter()
            .all(|(field_name, _)| field_name.is_some())
    {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("{fname} is a record constructor and must use named fields: {fname} {{ ... }}"),
            vec![],
        ));
        return Type::Error;
    }

    let func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let arg_tys: Vec<Type> = kids[1..]
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            if matches!(func_name.as_deref(), Some("expand"))
                && index == 2
                && symbolic_dim_ref_name(arg).is_some()
            {
                Type::Prim(Prim::Int32)
            } else {
                infer_expr(
                    arg,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                )
            }
        })
        .collect();

    // If func or any arg is Error, propagate
    if matches!(func_ty, Type::Error) || arg_tys.iter().any(|t| matches!(t, Type::Error)) {
        return Type::Error;
    }

    let ret_tv = vg.fresh_type();
    let expected_fn = Type::Fn(arg_tys.clone(), Box::new(ret_tv.clone()));

    const TENSOR_OPS: &[&str] = &[
        "add",
        "mul",
        "sub",
        "div",
        "neg",
        "exp",
        "log",
        "sin",
        "sqrt",
        "relu",
        "sigmoid",
        "matmul",
        "layer_norm",
        "max_elem",
        "min_elem",
        "normalize",
        "cmplt",
        "eq",
        "neq",
        "lt",
        "gt",
        "lte",
        "gte",
        "and",
        "or",
        "not",
    ];

    const LOGICAL_OPS: &[&str] = &["and", "or", "not"];
    const INT_BINOPS: &[&str] = &["mod", "bitand", "bitor", "bitxor"];
    const INT_SHIFT_OPS: &[&str] = &["shl", "shr"];

    match unify(&func_ty, &expected_fn, subst) {
        Ok(()) => {
            let mut result_ty = subst.apply(&ret_tv);

            // Post-check: shared builtins can operate on either tensors or host scalars.
            if let Some(ref fname) = func_name
                && TENSOR_OPS.contains(&fname.as_str())
            {
                for arg_ty in &arg_tys {
                    let resolved = subst.apply(arg_ty);
                    let ok = match fname.as_str() {
                        "matmul" | "layer_norm" | "normalize" => {
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                        }
                        "add" | "mul" | "sub" | "div" | "max_elem" | "min_elem" | "neg" => {
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                                || matches!(resolved, Type::Prim(prec) if prec.is_numeric())
                        }
                        "exp" | "log" | "sin" | "sqrt" | "relu" | "sigmoid" => {
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                                || matches!(resolved, Type::Prim(prec) if prec.is_float())
                        }
                        "cmplt" | "lt" | "gt" | "lte" | "gte" => {
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                                || matches!(resolved, Type::Prim(prec) if prec.is_numeric())
                        }
                        "eq" | "neq" => {
                            matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error)
                                || matches!(resolved, Type::Prim(_))
                        }
                        "and" | "or" | "not" => {
                            matches!(
                                resolved,
                                Type::Tensor(_, Prim::Bool) | Type::Var(_) | Type::Error
                            ) || matches!(resolved, Type::Prim(Prim::Bool))
                        }
                        _ => matches!(resolved, Type::Tensor(_, _) | Type::Var(_) | Type::Error),
                    };
                    if !ok {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "{} does not accept argument type {} in this context",
                                    fname, resolved
                                ),
                            ),
                            vec![],
                        ));
                        return Type::Error;
                    }
                }
            }

            if let Some(ref fname) = func_name
                && matches!(
                    fname.as_str(),
                    "softmax"
                        | "mean"
                        | "sum"
                        | "max_reduce"
                        | "min_reduce"
                        | "prod_reduce"
                        | "argmax_reduce"
                        | "argmin_reduce"
                )
            {
                if let Some(first_arg) = arg_tys.first() {
                    let resolved = subst.apply(first_arg);
                    match &resolved {
                        Type::Tensor(_, _) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("{} expects tensor input, got {}", fname, resolved),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }

                if let Some(axis_arg) = arg_tys.get(1) {
                    let resolved = subst.apply(axis_arg);
                    match &resolved {
                        Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("{} expects int32 axis, got {}", fname, resolved),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }
            }

            if let Some(ref fname) = func_name
                && fname == "uniform_like"
            {
                if let Some(first_arg) = arg_tys.first() {
                    let resolved = subst.apply(first_arg);
                    match &resolved {
                        Type::Tensor(_, prim) if prim.is_float() => {}
                        Type::Tensor(_, _) => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "uniform_like expects a float tensor template, got {}",
                                        resolved
                                    ),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "uniform_like expects tensor template input, got {}",
                                        resolved
                                    ),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }

                for (index, arg_ty) in arg_tys.iter().enumerate().skip(1).take(2) {
                    let resolved = subst.apply(arg_ty);
                    match &resolved {
                        Type::Prim(Prim::F32) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "uniform_like expects f32 bounds for args 2-3, got {}",
                                        resolved
                                    ),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }

                    if let Some(expr) = kids.get(index + 1)
                        && !is_numeric_literal_expr(expr)
                    {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                "uniform_like currently requires literal low/high bounds"
                                    .to_string(),
                            ),
                            vec![],
                        ));
                        return Type::Error;
                    }
                }
            }

            if let Some(ref fname) = func_name
                && fname == "dropout"
            {
                if let Some(first_arg) = arg_tys.first() {
                    let resolved = subst.apply(first_arg);
                    match &resolved {
                        Type::Tensor(_, _) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("dropout expects tensor input, got {}", resolved),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }

                if let Some(rate_arg) = arg_tys.get(1) {
                    let resolved = subst.apply(rate_arg);
                    match &resolved {
                        Type::Prim(Prim::F32) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("dropout expects f32 rate, got {}", resolved),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }
            }

            if let Some(ref fname) = func_name
                && fname == "conv2d"
            {
                for (index, arg_ty) in arg_tys.iter().enumerate() {
                    let resolved = subst.apply(arg_ty);
                    if index < 2 {
                        match &resolved {
                            Type::Tensor(_, _) | Type::Var(_) | Type::Error => {}
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "conv2d expects tensor inputs for args 1-2, got {}",
                                            resolved
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    } else {
                        match &resolved {
                            Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "conv2d expects int32 stride/padding, got {}",
                                            resolved
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                }
            }

            if let Some(ref fname) = func_name
                && INT_BINOPS.contains(&fname.as_str())
            {
                let lhs = arg_tys
                    .first()
                    .map(|ty| subst.apply(ty))
                    .unwrap_or(Type::Error);
                let rhs = arg_tys
                    .get(1)
                    .map(|ty| subst.apply(ty))
                    .unwrap_or(Type::Error);
                match (&lhs, &rhs) {
                    (Type::Prim(lhs_prec), Type::Prim(rhs_prec))
                        if lhs_prec.is_integer()
                            && rhs_prec.is_integer()
                            && lhs_prec == rhs_prec =>
                    {
                        return Type::Prim(*lhs_prec);
                    }
                    (Type::Var(_), Type::Prim(rhs_prec)) if rhs_prec.is_integer() => {
                        return lhs;
                    }
                    (Type::Prim(lhs_prec), Type::Var(_)) if lhs_prec.is_integer() => {
                        return lhs;
                    }
                    (Type::Var(_), Type::Var(_)) | (Type::Error, _) | (_, Type::Error) => {
                        return lhs;
                    }
                    _ => {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_macro_provenance(
                                &deep::Expr::List(list.clone(), zero_span()),
                                format!(
                                    "{} requires matching integer arguments, got {} and {}",
                                    fname, lhs, rhs
                                ),
                            ),
                            vec![],
                        ));
                        return Type::Error;
                    }
                }
            }

            if let Some(ref fname) = func_name
                && INT_SHIFT_OPS.contains(&fname.as_str())
            {
                let lhs = arg_tys
                    .first()
                    .map(|ty| subst.apply(ty))
                    .unwrap_or(Type::Error);
                let rhs = arg_tys
                    .get(1)
                    .map(|ty| subst.apply(ty))
                    .unwrap_or(Type::Error);
                let lhs_ok = matches!(&lhs, Type::Prim(prec) if prec.is_integer())
                    || matches!(&lhs, Type::Var(_) | Type::Error);
                let rhs_ok = matches!(&rhs, Type::Prim(prec) if prec.is_integer())
                    || matches!(&rhs, Type::Var(_) | Type::Error);
                if lhs_ok && rhs_ok {
                    return lhs;
                }
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!(
                            "{} requires integer lhs and shift amount, got {} and {}",
                            fname, lhs, rhs
                        ),
                    ),
                    vec![],
                ));
                return Type::Error;
            }

            if let Some(ref fname) = func_name {
                match fname.as_str() {
                    "matmul" => {
                        result_ty = check_matmul_signature(&arg_tys, &result_ty, subst, errors);
                    }
                    "sum" | "max_reduce" | "min_reduce" | "prod_reduce" | "argmax_reduce"
                    | "argmin_reduce" | "mean" => {
                        result_ty = check_reduction_signature(
                            fname,
                            &kids[1..],
                            &arg_tys,
                            &result_ty,
                            subst,
                            errors,
                        );
                    }
                    "expand" => {
                        result_ty =
                            check_expand_signature(&kids[1..], &arg_tys, &result_ty, subst, errors);
                    }
                    "layer_norm" => {
                        result_ty =
                            check_layer_norm_signature(&arg_tys, &result_ty, vg, subst, errors);
                    }
                    "conv2d" => {
                        result_ty = check_conv2d_signature(&arg_tys, &result_ty, vg, subst, errors);
                    }
                    _ => {}
                }
            }

            // Post-check: logical ops require tensor[D, bool] arguments
            if let Some(ref fname) = func_name
                && LOGICAL_OPS.contains(&fname.as_str())
            {
                for arg_ty in &arg_tys {
                    let resolved = subst.apply(arg_ty);
                    match &resolved {
                        Type::Tensor(_, Prim::Bool)
                        | Type::Prim(Prim::Bool)
                        | Type::Var(_)
                        | Type::Error => {} // OK
                        Type::Tensor(_, prec) => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!(
                                        "{} requires tensor[D, bool] arguments, got tensor[D, {}]",
                                        fname,
                                        prec.name()
                                    ),
                                ),
                                vec!["Logical ops only work on bool tensors".to_string()],
                            ));
                            return Type::Error;
                        }
                        other => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("{} requires bool arguments, got {}", fname, other),
                                ),
                                vec!["Logical ops only work on bool values".to_string()],
                            ));
                            return Type::Error;
                        }
                    }
                }
            }

            // Special case: comparison ops return tensor[D, bool]
            if let Some(ref fname) = func_name
                && builtins::COMPARISON_OPS.contains(&fname.as_str())
            {
                // Try to extract dims from arg types
                if let Some(first_arg) = arg_tys.first() {
                    let resolved_arg = subst.apply(first_arg);
                    if let Type::Tensor(dims, _) = resolved_arg {
                        return Type::Tensor(dims, Prim::Bool);
                    }
                    if matches!(resolved_arg, Type::Prim(_)) {
                        return Type::Prim(Prim::Bool);
                    }
                }
            }

            if let Some(ref fname) = func_name {
                match fname.as_str() {
                    "print" => {
                        return Type::Unit;
                    }
                    "fail" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Var(vg.fresh_tvar());
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("fail expects string input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "debug" => {
                        if let Some(first_arg) = arg_tys.first() {
                            return subst.apply(first_arg);
                        }
                    }
                    "string_len" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::Int64);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("string_len expects string input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "string_concat" => {
                        for arg_ty in &arg_tys {
                            match subst.apply(arg_ty) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "string_concat expects string arguments, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        return Type::Prim(Prim::String);
                    }
                    "string_slice" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "string_slice expects string input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        for (index, arg_ty) in arg_tys.iter().enumerate().skip(1) {
                            match subst.apply(arg_ty) {
                                Type::Prim(precision) if precision.is_integer() => {}
                                Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "string_slice expects integer index arguments; arg {} was {other}",
                                                index + 1
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        return Type::Prim(Prim::String);
                    }
                    "string_contains" | "string_starts_with" | "string_ends_with" => {
                        for arg_ty in &arg_tys {
                            match subst.apply(arg_ty) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "{} expects string arguments, got {other}",
                                                fname
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        return Type::Prim(Prim::Bool);
                    }
                    "string_trim" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::String);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "string_trim expects string input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "to_string" => {
                        return Type::Prim(Prim::String);
                    }
                    "to_int" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Adt(
                                        "Option".to_string(),
                                        vec![Type::Prim(Prim::Int64)],
                                    );
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("to_int expects string input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "to_float" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(Prim::String) | Type::Var(_) | Type::Error => {
                                    return Type::Adt(
                                        "Option".to_string(),
                                        vec![Type::Prim(Prim::F64)],
                                    );
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("to_float expects string input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "rank" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Tensor(_, _) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::Int32);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("rank expects tensor input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "shape" => {
                        let input_dims = if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Tensor(dims, _) => Some(dims),
                                Type::Var(_) | Type::Error => None,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("shape expects tensor input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        } else {
                            None
                        };
                        if let Some(axis_expr) = kids.get(2)
                            && let Some(axis) = extract_int_literal(axis_expr)
                        {
                            if axis < 0 {
                                errors.push(CheckError::new(
                                    CheckErrorKind::DimensionMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("shape requires non-negative axis, got {axis}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                            if let Some(dims) = input_dims.as_ref() {
                                let axis = axis as usize;
                                if axis >= dims.len() {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::DimensionMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "shape axis {axis} is out of bounds for rank {} tensor",
                                                dims.len()
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        if let Some(axis_arg) = arg_tys.get(1) {
                            match subst.apply(axis_arg) {
                                Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::Int32);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("shape expects int32 axis, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "numel" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Tensor(_, _) | Type::Var(_) | Type::Error => {
                                    return Type::Prim(Prim::Int64);
                                }
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("numel expects tensor input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "tensor_to_scalar" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Tensor(dims, precision) => {
                                    if !dims.is_empty() {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                "tensor_to_scalar expects a rank-0 tensor"
                                                    .to_string(),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                    return Type::Prim(precision);
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "tensor_to_scalar expects tensor input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "scalar_to_tensor" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Prim(precision) if !matches!(precision, Prim::String) => {
                                    return Type::Tensor(vec![], precision);
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "scalar_to_tensor expects scalar numeric/bool input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "einsum" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let Some(equation) = kids.get(1).and_then(extract_string_literal) else {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    "einsum expects a string equation as its first argument"
                                        .to_string(),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        };
                        if equation.contains("...") {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    "einsum ellipsis support is deferred in 3h".to_string(),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        return result_ty;
                    }
                    "gather" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let axis = kids.get(3).and_then(extract_axis_literal).unwrap_or(0);
                        let tensor_ty = subst.apply(&arg_tys[0]);
                        let indices_ty = subst.apply(&arg_tys[1]);
                        match infer_gather_result_type(&tensor_ty, &indices_ty, axis) {
                            Ok(ty) => return ty,
                            Err(message) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        message,
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "where" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let cond_ty = subst.apply(&arg_tys[0]);
                        let then_ty = subst.apply(&arg_tys[1]);
                        let else_ty = subst.apply(&arg_tys[2]);
                        match (&cond_ty, &then_ty, &else_ty) {
                            (
                                Type::Tensor(cond_dims, Prim::Bool),
                                Type::Tensor(then_dims, then_prec),
                                Type::Tensor(else_dims, else_prec),
                            ) => {
                                if then_prec != else_prec
                                    || cond_dims != then_dims
                                    || then_dims != else_dims
                                {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "where expects cond/both branches to have matching tensor shapes and branch precision, got {cond_ty}, {then_ty}, and {else_ty}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                                return Type::Tensor(then_dims.clone(), *then_prec);
                            }
                            (Type::Var(_), _, _)
                            | (_, Type::Var(_), _)
                            | (_, _, Type::Var(_))
                            | (Type::Error, _, _)
                            | (_, Type::Error, _)
                            | (_, _, Type::Error) => return result_ty,
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "where expects a bool tensor condition and matching tensor branches, got {cond_ty}, {then_ty}, and {else_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "cumsum" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let axis = kids.get(2).and_then(extract_axis_literal).unwrap_or(0);
                        match subst.apply(&arg_tys[0]) {
                            Type::Tensor(dims, precision) => {
                                if axis >= dims.len() {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "cumsum axis {axis} out of bounds for rank {}",
                                                dims.len()
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                                return Type::Tensor(dims, precision);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("cumsum expects tensor input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "diagonal" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let axis1 = kids.get(2).and_then(extract_axis_literal).unwrap_or(0);
                        let axis2 = kids.get(3).and_then(extract_axis_literal).unwrap_or(1);
                        match infer_diagonal_result_type(&subst.apply(&arg_tys[0]), axis1, axis2) {
                            Ok(ty) => return ty,
                            Err(message) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        message,
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "trace" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let axis1 = kids.get(2).and_then(extract_axis_literal).unwrap_or(0);
                        let axis2 = kids.get(3).and_then(extract_axis_literal).unwrap_or(1);
                        match infer_trace_result_type(&subst.apply(&arg_tys[0]), axis1, axis2) {
                            Ok(ty) => return ty,
                            Err(message) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        message,
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "clamp" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let input_ty = subst.apply(&arg_tys[0]);
                        let low_ty = subst.apply(&arg_tys[1]);
                        let high_ty = subst.apply(&arg_tys[2]);
                        match (&input_ty, &low_ty, &high_ty) {
                            (
                                Type::Tensor(input_dims, input_prec),
                                Type::Tensor(low_dims, low_prec),
                                Type::Tensor(high_dims, high_prec),
                            ) => {
                                let low_ok = low_dims.is_empty() || low_dims == input_dims;
                                let high_ok = high_dims.is_empty() || high_dims == input_dims;
                                if low_ok
                                    && high_ok
                                    && low_prec == input_prec
                                    && high_prec == input_prec
                                {
                                    return input_ty;
                                }
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "clamp expects tensor input plus scalar-tensor or matching-shape tensor bounds of the same precision, got {input_ty}, {low_ty}, and {high_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                            (Type::Var(_), _, _)
                            | (_, Type::Var(_), _)
                            | (_, _, Type::Var(_))
                            | (Type::Error, _, _)
                            | (_, Type::Error, _)
                            | (_, _, Type::Error) => return result_ty,
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "clamp expects tensor input and tensor bounds, got {input_ty}, {low_ty}, and {high_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "sort" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let axis = kids.get(2).and_then(extract_axis_literal).unwrap_or(0);
                        match subst.apply(&arg_tys[0]) {
                            Type::Tensor(dims, precision) => {
                                if axis >= dims.len() {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "sort axis {axis} out of bounds for rank {}",
                                                dims.len()
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                                return Type::Tuple(vec![
                                    Type::Tensor(dims.clone(), precision),
                                    Type::Tensor(dims, Prim::Int64),
                                ]);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("sort expects tensor input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "scatter" => {
                        if arg_tys.len() != 5 {
                            return Type::Error;
                        }
                        let base_ty = subst.apply(&arg_tys[0]);
                        let indices_ty = subst.apply(&arg_tys[1]);
                        let updates_ty = subst.apply(&arg_tys[2]);
                        let axis = kids.get(4).and_then(extract_axis_literal).unwrap_or(0);
                        let mode = kids.get(5).and_then(extract_string_literal);
                        match mode.as_deref() {
                            Some("replace") | Some("add") => {}
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        "scatter mode must be \"replace\" or \"add\"".to_string(),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                        match infer_gather_result_type(&base_ty, &indices_ty, axis) {
                            Ok(expected_updates) => {
                                if let Err(te) = unify(&expected_updates, &updates_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return base_ty;
                            }
                            Err(message) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        message,
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "len" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, _) if name == "List" || name == "Dict" => {
                                    return Type::Prim(Prim::Int64);
                                }
                                Type::Var(_) | Type::Error => return Type::Prim(Prim::Int64),
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("len expects List or Dict input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "index" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let list_arg = subst.apply(&arg_tys[0]);
                        let index_arg = subst.apply(&arg_tys[1]);
                        if !matches!(index_arg, Type::Prim(prec) if prec.is_integer())
                            && !matches!(index_arg, Type::Var(_) | Type::Error)
                        {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("index expects integer index, got {index_arg}"),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        match list_arg {
                            Type::Adt(name, mut args) if name == "List" && args.len() == 1 => {
                                return args.remove(0);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("index expects List input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "append" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let list_arg = subst.apply(&arg_tys[0]);
                        let value_arg = subst.apply(&arg_tys[1]);
                        match list_arg {
                            Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                if let Err(te) = unify(&args[0], &value_arg, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt("List".to_string(), vec![subst.apply(&args[0])]);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("append expects List input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "concat" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let lhs = subst.apply(&arg_tys[0]);
                        let rhs = subst.apply(&arg_tys[1]);
                        match (lhs, rhs) {
                            (Type::Adt(lhs_name, lhs_args), Type::Prim(precision))
                                if lhs_name == "List"
                                    && lhs_args.len() == 1
                                    && precision.is_integer() =>
                            {
                                match tensor_concat_result_type(&lhs_args[0]) {
                                    Ok(ty) => return ty,
                                    Err(message) => {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                message,
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                }
                            }
                            (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                                if lhs_name == "List"
                                    && rhs_name == "List"
                                    && lhs_args.len() == 1
                                    && rhs_args.len() == 1 =>
                            {
                                if let Err(te) = unify(&lhs_args[0], &rhs_args[0], subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "List".to_string(),
                                    vec![subst.apply(&lhs_args[0])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (lhs, rhs) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "concat expects matching List inputs, got {lhs} and {rhs}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "split" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let tensor_ty = subst.apply(&arg_tys[0]);
                        let axis_ty = subst.apply(&arg_tys[1]);
                        let sizes_ty = subst.apply(&arg_tys[2]);
                        if !matches!(axis_ty, Type::Prim(prec) if prec.is_integer())
                            && !matches!(axis_ty, Type::Var(_) | Type::Error)
                        {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("split expects integer axis, got {axis_ty}"),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        match (tensor_ty, sizes_ty) {
                            (Type::Tensor(dims, precision), Type::Adt(name, args))
                                if name == "List" && args.len() == 1 =>
                            {
                                if !matches!(&args[0], Type::Prim(prec) if prec.is_integer())
                                    && !matches!(&args[0], Type::Var(_) | Type::Error)
                                {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            "split expects List[int] sizes".to_string(),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                                let axis = kids.get(2).and_then(extract_axis_literal).unwrap_or(0);
                                if axis >= dims.len() {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "split axis {axis} out of bounds for rank {}",
                                                dims.len()
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                                let mut piece_dims = dims.clone();
                                piece_dims[axis] = Dim::Wildcard;
                                return Type::Adt(
                                    "List".to_string(),
                                    vec![Type::Tensor(piece_dims, precision)],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (tensor_ty, sizes_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "split expects tensor input and List[int] sizes, got {tensor_ty} and {sizes_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "take" | "drop" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let op_name = func_name.as_deref().unwrap_or("collection helper");
                        let list_arg = subst.apply(&arg_tys[0]);
                        let count_arg = subst.apply(&arg_tys[1]);
                        if !matches!(count_arg, Type::Prim(prec) if prec.is_integer())
                            && !matches!(count_arg, Type::Var(_) | Type::Error)
                        {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("{op_name} expects integer count, got {count_arg}"),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        match list_arg {
                            Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                return Type::Adt("List".to_string(), vec![args[0].clone()]);
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("{op_name} expects List input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "chunk" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let list_arg = subst.apply(&arg_tys[0]);
                        let count_arg = subst.apply(&arg_tys[1]);
                        if !matches!(count_arg, Type::Prim(prec) if prec.is_integer())
                            && !matches!(count_arg, Type::Var(_) | Type::Error)
                        {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_macro_provenance(
                                    &deep::Expr::List(list.clone(), zero_span()),
                                    format!("chunk expects integer size, got {count_arg}"),
                                ),
                                vec![],
                            ));
                            return Type::Error;
                        }
                        match list_arg {
                            Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                return Type::Adt(
                                    "List".to_string(),
                                    vec![Type::Adt("List".to_string(), vec![args[0].clone()])],
                                );
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("chunk expects List input, got {other}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "range" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        for arg_ty in &arg_tys {
                            match subst.apply(arg_ty) {
                                Type::Prim(prec) if prec.is_integer() => {}
                                Type::Var(_) | Type::Error => {}
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("range expects integer arguments, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                    }
                    "map" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let elem_ty = vg.fresh_type();
                        let out_ty = vg.fresh_type();
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(vec![elem_ty.clone()], Box::new(out_ty.clone())),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[1]),
                            &Type::Adt("List".to_string(), vec![elem_ty]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&out_ty)]);
                    }
                    "filter" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let elem_ty = vg.fresh_type();
                        let list_expr = deep::Expr::List(list.clone(), zero_span());
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(vec![elem_ty.clone()], Box::new(Type::Prim(Prim::Bool))),
                            subst,
                        ) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "filter",
                                "expects a callback that returns bool",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[1]),
                            &Type::Adt("List".to_string(), vec![elem_ty.clone()]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&elem_ty)]);
                    }
                    "fold" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let acc_ty = vg.fresh_type();
                        let elem_ty = vg.fresh_type();
                        let list_expr = deep::Expr::List(list.clone(), zero_span());
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(
                                vec![acc_ty.clone(), elem_ty.clone()],
                                Box::new(acc_ty.clone()),
                            ),
                            subst,
                        ) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "fold",
                                "expects a callback whose accumulator/result type matches the initial accumulator",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(&subst.apply(&arg_tys[1]), &acc_ty.clone(), subst) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "fold",
                                "expects a callback whose accumulator/result type matches the initial accumulator",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[2]),
                            &Type::Adt("List".to_string(), vec![elem_ty]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return subst.apply(&acc_ty);
                    }
                    "scan" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let acc_ty = vg.fresh_type();
                        let elem_ty = vg.fresh_type();
                        let list_expr = deep::Expr::List(list.clone(), zero_span());
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(
                                vec![acc_ty.clone(), elem_ty.clone()],
                                Box::new(acc_ty.clone()),
                            ),
                            subst,
                        ) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "scan",
                                "expects a callback whose accumulator/result type matches the initial accumulator",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(&subst.apply(&arg_tys[1]), &acc_ty.clone(), subst) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "scan",
                                "expects a callback whose accumulator/result type matches the initial accumulator",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[2]),
                            &Type::Adt("List".to_string(), vec![elem_ty]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&acc_ty)]);
                    }
                    "partition" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let elem_ty = vg.fresh_type();
                        let list_expr = deep::Expr::List(list.clone(), zero_span());
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(vec![elem_ty.clone()], Box::new(Type::Prim(Prim::Bool))),
                            subst,
                        ) {
                            errors.push(collection_helper_type_error(
                                &list_expr,
                                "partition",
                                "expects a callback that returns bool",
                                te,
                            ));
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[1]),
                            &Type::Adt("List".to_string(), vec![elem_ty.clone()]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        let out_list = Type::Adt("List".to_string(), vec![subst.apply(&elem_ty)]);
                        return Type::Tuple(vec![out_list.clone(), out_list]);
                    }
                    "flat_map" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let elem_ty = vg.fresh_type();
                        let out_elem_ty = vg.fresh_type();
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[0]),
                            &Type::Fn(
                                vec![elem_ty.clone()],
                                Box::new(Type::Adt("List".to_string(), vec![out_elem_ty.clone()])),
                            ),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        if let Err(te) = unify(
                            &subst.apply(&arg_tys[1]),
                            &Type::Adt("List".to_string(), vec![elem_ty]),
                            subst,
                        ) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        return Type::Adt("List".to_string(), vec![subst.apply(&out_elem_ty)]);
                    }
                    "flatten" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(outer_name, outer_args)
                                    if outer_name == "List" && outer_args.len() == 1 =>
                                {
                                    match &outer_args[0] {
                                        Type::Adt(inner_name, inner_args)
                                            if inner_name == "List" && inner_args.len() == 1 =>
                                        {
                                            return Type::Adt(
                                                "List".to_string(),
                                                vec![inner_args[0].clone()],
                                            );
                                        }
                                        Type::Var(_) | Type::Error => return result_ty,
                                        other => {
                                            errors.push(CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_macro_provenance(
                                                    &deep::Expr::List(
                                                        list.clone(),
                                                        zero_span(),
                                                    ),
                                                    format!(
                                                        "flatten expects List[List[T]] input, got List[{other}]"
                                                    ),
                                                ),
                                                vec![],
                                            ));
                                            return Type::Error;
                                        }
                                    }
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!(
                                                "flatten expects List[List[T]] input, got {other}"
                                            ),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "zip" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let lhs = subst.apply(&arg_tys[0]);
                        let rhs = subst.apply(&arg_tys[1]);
                        match (lhs, rhs) {
                            (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                                if lhs_name == "List"
                                    && rhs_name == "List"
                                    && lhs_args.len() == 1
                                    && rhs_args.len() == 1 =>
                            {
                                return Type::Adt(
                                    "List".to_string(),
                                    vec![Type::Tuple(vec![
                                        lhs_args[0].clone(),
                                        rhs_args[0].clone(),
                                    ])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (lhs, rhs) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("zip expects List inputs, got {lhs} and {rhs}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "enumerate" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                    return Type::Adt(
                                        "List".to_string(),
                                        vec![Type::Tuple(vec![
                                            Type::Prim(Prim::Int64),
                                            args[0].clone(),
                                        ])],
                                    );
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("enumerate expects List input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "dict_of" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                    match &args[0] {
                                        Type::Tuple(items) if items.len() == 2 => {
                                            match &items[0] {
                                                Type::Prim(Prim::Int64)
                                                | Type::Prim(Prim::String) => {}
                                                Type::Var(_) | Type::Error => return result_ty,
                                                other => {
                                                    errors.push(CheckError::new(
                                                        CheckErrorKind::TypeMismatch,
                                                        with_macro_provenance(
                                                            &deep::Expr::List(list.clone(), zero_span()),
                                                            format!(
                                                                "dict_of keys must be int64 or string, got {other}"
                                                            ),
                                                        ),
                                                        vec![],
                                                    ));
                                                    return Type::Error;
                                                }
                                            }
                                            return Type::Adt(
                                                "Dict".to_string(),
                                                vec![items[0].clone(), items[1].clone()],
                                            );
                                        }
                                        Type::Var(_) | Type::Error => return result_ty,
                                        other => {
                                            errors.push(CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_macro_provenance(
                                                    &deep::Expr::List(list.clone(), zero_span()),
                                                    format!(
                                                        "dict_of expects List[(K, V)] input, got List[{other}]"
                                                    ),
                                                ),
                                                vec![],
                                            ));
                                            return Type::Error;
                                        }
                                    }
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("dict_of expects List input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "dict_get" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                            (Type::Adt(name, args), key_ty)
                                if name == "Dict" && args.len() == 2 =>
                            {
                                if let Err(te) = unify(&args[0], &key_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "Option".to_string(),
                                    vec![subst.apply(&args[1])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (dict_ty, key_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!("dict_get expects Dict[K, V] and K, got {dict_ty} and {key_ty}"),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_contains" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                            (Type::Adt(name, args), key_ty)
                                if name == "Dict" && args.len() == 2 =>
                            {
                                if let Err(te) = unify(&args[0], &key_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Prim(Prim::Bool);
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (dict_ty, key_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "dict_contains expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_remove" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                            (Type::Adt(name, args), key_ty)
                                if name == "Dict" && args.len() == 2 =>
                            {
                                if let Err(te) = unify(&args[0], &key_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "Dict".to_string(),
                                    vec![subst.apply(&args[0]), subst.apply(&args[1])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (dict_ty, key_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "dict_remove expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_insert" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        match (
                            subst.apply(&arg_tys[0]),
                            subst.apply(&arg_tys[1]),
                            subst.apply(&arg_tys[2]),
                        ) {
                            (Type::Adt(name, args), key_ty, value_ty)
                                if name == "Dict" && args.len() == 2 =>
                            {
                                if let Err(te) = unify(&args[0], &key_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                if let Err(te) = unify(&args[1], &value_ty, subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "Dict".to_string(),
                                    vec![subst.apply(&args[0]), subst.apply(&args[1])],
                                );
                            }
                            (Type::Var(_), _, _)
                            | (_, Type::Var(_), _)
                            | (_, _, Type::Var(_))
                            | (Type::Error, _, _)
                            | (_, Type::Error, _)
                            | (_, _, Type::Error) => {
                                return result_ty;
                            }
                            (dict_ty, key_ty, value_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "dict_insert expects Dict[K, V], K, and V, got {dict_ty}, {key_ty}, and {value_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_merge" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                            (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                                if lhs_name == "Dict"
                                    && rhs_name == "Dict"
                                    && lhs_args.len() == 2
                                    && rhs_args.len() == 2 =>
                            {
                                if let Err(te) = unify(&lhs_args[0], &rhs_args[0], subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                if let Err(te) = unify(&lhs_args[1], &rhs_args[1], subst) {
                                    errors.push(te.into());
                                    return Type::Error;
                                }
                                return Type::Adt(
                                    "Dict".to_string(),
                                    vec![subst.apply(&lhs_args[0]), subst.apply(&lhs_args[1])],
                                );
                            }
                            (Type::Var(_), _)
                            | (_, Type::Var(_))
                            | (Type::Error, _)
                            | (_, Type::Error) => {
                                return result_ty;
                            }
                            (lhs_ty, rhs_ty) => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "dict_merge expects matching Dict inputs, got {lhs_ty} and {rhs_ty}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "dict_keys" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                                    return Type::Adt("List".to_string(), vec![args[0].clone()]);
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("dict_keys expects Dict input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "dict_values" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                                    return Type::Adt("List".to_string(), vec![args[1].clone()]);
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("dict_values expects Dict input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "dict_entries" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                                    return Type::Adt(
                                        "List".to_string(),
                                        vec![Type::Tuple(vec![args[0].clone(), args[1].clone()])],
                                    );
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("dict_entries expects Dict input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "to_tensor" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                                    match &args[0] {
                                        Type::Prim(precision) if precision.is_numeric() => {
                                            return Type::Tensor(vec![Dim::Wildcard], *precision);
                                        }
                                        Type::Var(_) | Type::Error => return result_ty,
                                        other => {
                                            errors.push(CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_macro_provenance(
                                                    &deep::Expr::List(list.clone(), zero_span()),
                                                    format!(
                                                        "to_tensor expects numeric List elements, got {other}"
                                                    ),
                                                ),
                                                vec![],
                                            ));
                                            return Type::Error;
                                        }
                                    }
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("to_tensor expects List input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "to_list" => {
                        if let Some(first_arg) = arg_tys.first() {
                            match subst.apply(first_arg) {
                                Type::Tensor(dims, precision) => {
                                    if dims.len() != 1 {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                format!(
                                                    "to_list expects a rank-1 tensor, got rank {} tensor",
                                                    dims.len()
                                                ),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                    if !precision.is_numeric() && !matches!(precision, Prim::Bool) {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                format!(
                                                    "to_list expects numeric or bool tensor input, got {precision:?}"
                                                ),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                    return Type::Adt(
                                        "List".to_string(),
                                        vec![Type::Prim(precision)],
                                    );
                                }
                                Type::Var(_) | Type::Error => return result_ty,
                                other => {
                                    errors.push(CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_macro_provenance(
                                            &deep::Expr::List(list.clone(), zero_span()),
                                            format!("to_list expects Tensor input, got {other}"),
                                        ),
                                        vec![],
                                    ));
                                    return Type::Error;
                                }
                            }
                        }
                    }
                    "pad_sequences" => {
                        if arg_tys.len() != 2 {
                            return Type::Error;
                        }
                        let seqs_ty = subst.apply(&arg_tys[0]);
                        let pad_ty = subst.apply(&arg_tys[1]);
                        match seqs_ty {
                            Type::Adt(outer_name, outer_args)
                                if outer_name == "List" && outer_args.len() == 1 =>
                            {
                                match &outer_args[0] {
                                    Type::Adt(inner_name, inner_args)
                                        if inner_name == "List" && inner_args.len() == 1 =>
                                    {
                                        if let Err(te) = unify(&inner_args[0], &pad_ty, subst) {
                                            errors.push(te.into());
                                            return Type::Error;
                                        }
                                        match subst.apply(&inner_args[0]) {
                                            Type::Prim(precision) if precision.is_numeric() => {
                                                return Type::Tensor(
                                                    vec![Dim::Wildcard, Dim::Wildcard],
                                                    precision,
                                                );
                                            }
                                            Type::Var(_) | Type::Error => return result_ty,
                                            other => {
                                                errors.push(CheckError::new(
                                                    CheckErrorKind::TypeMismatch,
                                                    with_macro_provenance(
                                                        &deep::Expr::List(
                                                            list.clone(),
                                                            zero_span(),
                                                        ),
                                                        format!(
                                                            "pad_sequences expects numeric nested lists, got {other}"
                                                        ),
                                                    ),
                                                    vec![],
                                                ));
                                                return Type::Error;
                                            }
                                        }
                                    }
                                    other => {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                format!(
                                                    "pad_sequences expects List[List[T]], got List[{other}]"
                                                ),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                }
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "pad_sequences expects List[List[T]] input, got {other}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "pad_sequences_to" => {
                        if arg_tys.len() != 3 {
                            return Type::Error;
                        }
                        let seqs_ty = subst.apply(&arg_tys[0]);
                        let width_ty = subst.apply(&arg_tys[1]);
                        let pad_ty = subst.apply(&arg_tys[2]);
                        if let Err(te) = unify(&width_ty, &Type::Prim(Prim::Int64), subst) {
                            errors.push(te.into());
                            return Type::Error;
                        }
                        match seqs_ty {
                            Type::Adt(outer_name, outer_args)
                                if outer_name == "List" && outer_args.len() == 1 =>
                            {
                                match &outer_args[0] {
                                    Type::Adt(inner_name, inner_args)
                                        if inner_name == "List" && inner_args.len() == 1 =>
                                    {
                                        if let Err(te) = unify(&inner_args[0], &pad_ty, subst) {
                                            errors.push(te.into());
                                            return Type::Error;
                                        }
                                        match subst.apply(&inner_args[0]) {
                                            Type::Prim(precision) if precision.is_numeric() => {
                                                return Type::Tensor(
                                                    vec![Dim::Wildcard, Dim::Wildcard],
                                                    precision,
                                                );
                                            }
                                            Type::Var(_) | Type::Error => return result_ty,
                                            other => {
                                                errors.push(CheckError::new(
                                                    CheckErrorKind::TypeMismatch,
                                                    with_macro_provenance(
                                                        &deep::Expr::List(
                                                            list.clone(),
                                                            zero_span(),
                                                        ),
                                                        format!(
                                                            "pad_sequences_to expects numeric nested lists, got {other}"
                                                        ),
                                                    ),
                                                    vec![],
                                                ));
                                                return Type::Error;
                                            }
                                        }
                                    }
                                    other => {
                                        errors.push(CheckError::new(
                                            CheckErrorKind::TypeMismatch,
                                            with_macro_provenance(
                                                &deep::Expr::List(list.clone(), zero_span()),
                                                format!(
                                                    "pad_sequences_to expects List[List[T]], got List[{other}]"
                                                ),
                                            ),
                                            vec![],
                                        ));
                                        return Type::Error;
                                    }
                                }
                            }
                            Type::Var(_) | Type::Error => return result_ty,
                            other => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    with_macro_provenance(
                                        &deep::Expr::List(list.clone(), zero_span()),
                                        format!(
                                            "pad_sequences_to expects List[List[T]] input, got {other}"
                                        ),
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                    "read_file" => return Type::Prim(Prim::String),
                    "write_file" => return Type::Unit,
                    "read_lines" => {
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::String)]);
                    }
                    "read_bytes" => {
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                    }
                    "file_exists" => return Type::Prim(Prim::Bool),
                    "list_dir" => {
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::String)]);
                    }
                    "mmap_file" => return Type::Adt("MappedFile".to_string(), Vec::new()),
                    "mmap_read" => {
                        return Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                    }
                    "mmap_len" => return Type::Prim(Prim::Int64),
                    _ => {}
                }
            }

            result_ty
        }
        Err(te) => {
            errors.push(te.into());
            Type::Error
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_permute_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            "permute expects a tensor followed by one or more axis indices".to_string(),
            vec![],
        ));
        return Type::Error;
    }

    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let input_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let axis_tys: Vec<Type> = kids[2..]
        .iter()
        .map(|arg| {
            infer_expr(
                arg,
                env,
                vg,
                subst,
                adt_reg,
                errors,
                typed_nodes,
                total_nodes,
            )
        })
        .collect();

    if matches!(input_ty, Type::Error) || axis_tys.iter().any(|ty| matches!(ty, Type::Error)) {
        return Type::Error;
    }

    for axis_ty in &axis_tys {
        let resolved = subst.apply(axis_ty);
        match resolved {
            Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
            other => {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("permute expects int32 axis indices, got {other}"),
                    ),
                    vec![],
                ));
                return Type::Error;
            }
        }
    }

    let input_ty = subst.apply(&input_ty);
    let Type::Tensor(dims, prec) = input_ty else {
        if matches!(input_ty, Type::Var(_) | Type::Error) {
            return input_ty;
        }
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("permute expects tensor input, got {input_ty}"),
            vec![],
        ));
        return Type::Error;
    };

    let Some(axes) = kids[2..]
        .iter()
        .map(extract_int_literal)
        .collect::<Option<Vec<_>>>()
    else {
        return Type::Tensor(dims, prec);
    };

    if axes.len() != dims.len() {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            format!(
                "permute expects {} axis indices for rank {} tensor, got {}",
                dims.len(),
                dims.len(),
                axes.len()
            ),
            vec![],
        ));
        return Type::Error;
    }

    let mut seen = HashSet::new();
    let mut reordered = Vec::with_capacity(dims.len());
    for axis in axes {
        if axis < 0 || axis as usize >= dims.len() {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "permute axis {axis} is out of bounds for rank {} tensor",
                    dims.len()
                ),
                vec![],
            ));
            return Type::Error;
        }
        let axis = axis as usize;
        if !seen.insert(axis) {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("permute axis {axis} appears more than once"),
                vec![],
            ));
            return Type::Error;
        }
        reordered.push(dims[axis].clone());
    }

    Type::Tensor(reordered, prec)
}

#[allow(clippy::too_many_arguments)]
fn infer_reshape_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 || kids.len() > 3 {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            "reshape expects a tensor and an optional shape list".to_string(),
            vec![],
        ));
        return Type::Error;
    }

    let _func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let input_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    match subst.apply(&input_ty) {
        Type::Prim(precision) => {
            if let Some(shape_expr) = kids.get(2) {
                let shape_ty = infer_expr(
                    shape_expr,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
                let expected_shape_ty =
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                if let Err(te) = unify(&shape_ty, &expected_shape_ty, subst) {
                    errors.push(te.into());
                    return Type::Error;
                }
                let dims = list_literal_dims(shape_expr).unwrap_or_else(|| {
                    let rank = list_literal_len(shape_expr).unwrap_or(1);
                    vec![Dim::Wildcard; rank]
                });
                return Type::Tensor(dims, precision);
            }

            Type::Tensor(vec![Dim::Wildcard], precision)
        }
        Type::Tensor(_, precision) => {
            if let Some(shape_expr) = kids.get(2) {
                let shape_ty = infer_expr(
                    shape_expr,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
                let expected_shape_ty =
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                if let Err(te) = unify(&shape_ty, &expected_shape_ty, subst) {
                    errors.push(te.into());
                    return Type::Error;
                }
                let dims = list_literal_dims(shape_expr).unwrap_or_else(|| {
                    let rank = list_literal_len(shape_expr).unwrap_or(1);
                    vec![Dim::Wildcard; rank]
                });
                return Type::Tensor(dims, precision);
            }

            Type::Tensor(vec![Dim::Wildcard], precision)
        }
        Type::Var(_) | Type::Error => {
            if let Some(shape_expr) = kids.get(2) {
                let shape_ty = infer_expr(
                    shape_expr,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
                let expected_shape_ty =
                    Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]);
                if let Err(te) = unify(&shape_ty, &expected_shape_ty, subst) {
                    errors.push(te.into());
                    return Type::Error;
                }
            }
            input_ty
        }
        _ => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                "reshape expects tensor input".to_string(),
                vec![],
            ));
            Type::Error
        }
    }
}

fn list_literal_len(expr: &deep::Expr) -> Option<usize> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("list") {
        return None;
    }
    Some(children(list).len())
}

/// If `expr` is a list literal whose every element is a concrete int literal
/// (`5`, `lit 5`, or `cast(5, int64)` / `cast(5, int32)`), return the dim
/// vector with each element as `Dim::Lit(N)`. Handles both the `(list ...)`
/// tag form and the desugared Cons/Nil chain — surface list literals lower
/// to the chain form by the time reshape is type-checked.
///
/// Returns `None` if any element is not a concrete integer or the list is
/// not closed by a `Nil` — the caller falls back to `Dim::Wildcard`.
///
/// Without this, `reshape(t, [2, 1, 3])` infers as
/// `tensor[Wildcard, Wildcard, Wildcard, p]` and a function declared as
/// `-> tensor[2, 1, 3, f32]` reports a body/signature mismatch — RT-A1W1
/// CRIT root cause (#35).
fn list_literal_dims(expr: &deep::Expr) -> Option<Vec<Dim>> {
    if let deep::Expr::List(list, _) = expr
        && get_tag(list) == Some("list")
    {
        return children(list)
            .iter()
            .map(extract_int_for_dim)
            .map(|opt| opt.map(Dim::Lit))
            .collect();
    }
    cons_chain_int_dims(expr)
}

/// Walk a `Cons(head, Cons(head, ..., Nil))` chain and collect each head as
/// a `Dim::Lit`. Returns `None` if the chain isn't closed by `Nil` or any
/// head fails to extract as a concrete int.
fn cons_chain_int_dims(expr: &deep::Expr) -> Option<Vec<Dim>> {
    let mut dims = Vec::new();
    let mut cursor = expr;
    loop {
        let deep::Expr::List(list, _) = cursor else {
            return None;
        };
        match get_tag(list)? {
            "var" => {
                let name = children(list).first().and_then(symbol_name)?;
                if name == "Nil" {
                    return Some(dims);
                }
                return None;
            }
            "app" => {
                let app_children = children(list);
                let func = app_children.first()?;
                if !is_builtin_var(func, "Cons") {
                    return None;
                }
                let head = app_children.get(1)?;
                let tail = app_children.get(2)?;
                dims.push(Dim::Lit(extract_int_for_dim(head)?));
                cursor = tail;
            }
            _ => return None,
        }
    }
}

/// Extract an int literal from a Deep expr, looking through `cast(N, int64)`
/// and `cast(N, int32)` — both are common in Chelis dim lists since integer
/// literals default to int32 and require an explicit cast for int64 contexts.
/// `cast` may surface either as the `(cast {} ... ...)` tag or as an `app`
/// of the `cast` var, depending on how far desugaring has progressed.
fn extract_int_for_dim(expr: &deep::Expr) -> Option<i64> {
    if let Some(value) = extract_int_literal(expr) {
        return Some(value);
    }
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    match get_tag(list)? {
        "cast" => extract_int_literal(children(list).first()?),
        "app" => {
            let app_children = children(list);
            let func = app_children.first()?;
            if !is_builtin_var(func, "cast") {
                return None;
            }
            extract_int_literal(app_children.get(1)?)
        }
        _ => None,
    }
}

fn check_layer_norm_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    _vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 3 {
        return Type::Error;
    }

    let x_ty = subst.apply(&arg_tys[0]);
    let gamma_ty = subst.apply(&arg_tys[1]);
    let beta_ty = subst.apply(&arg_tys[2]);

    let (x_dims, x_prec) = match x_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("layer_norm expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let (gamma_dims, gamma_prec) = match gamma_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("layer_norm expects tensor gamma, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let (beta_dims, beta_prec) = match beta_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("layer_norm expects tensor beta, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if x_dims.is_empty() {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            "layer_norm expects rank >= 1 input tensor".to_string(),
            vec![],
        ));
        return Type::Error;
    }
    if gamma_dims.len() != 1 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "layer_norm expects rank-1 gamma, got rank {}",
                gamma_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if beta_dims.len() != 1 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "layer_norm expects rank-1 beta, got rank {}",
                beta_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if x_prec != gamma_prec || x_prec != beta_prec {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "layer_norm requires matching precisions, got {}, {}, {}",
                x_prec.name(),
                gamma_prec.name(),
                beta_prec.name()
            ),
            vec!["Insert explicit cast".to_string()],
        ));
        return Type::Error;
    }

    let hidden_dim = x_dims.last().cloned().expect("checked non-empty");
    if let Err(te) = unify_dim(&hidden_dim, &gamma_dims[0], subst) {
        errors.push(te.into());
        return Type::Error;
    }
    if let Err(te) = unify_dim(&hidden_dim, &beta_dims[0], subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let canonical = Type::Tensor(
        x_dims.into_iter().map(|d| subst.apply_dim(&d)).collect(),
        x_prec,
    );
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

fn check_conv2d_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() < 2 {
        return Type::Error;
    }

    let input_ty = subst.apply(&arg_tys[0]);
    let kernel_ty = subst.apply(&arg_tys[1]);

    let (input_dims, input_prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("conv2d expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };
    let (kernel_dims, kernel_prec) = match kernel_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("conv2d expects tensor kernel, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if input_dims.len() != 4 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "conv2d expects rank-4 input tensor, got rank {}",
                input_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if kernel_dims.len() != 4 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "conv2d expects rank-4 kernel tensor, got rank {}",
                kernel_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if input_prec != kernel_prec {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "conv2d requires matching input/kernel precision, got {} and {}",
                input_prec.name(),
                kernel_prec.name()
            ),
            vec!["Insert explicit cast".to_string()],
        ));
        return Type::Error;
    }
    if let Err(te) = unify_dim(&input_dims[1], &kernel_dims[1], subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let output_template = Type::Tensor(
        vec![
            subst.apply_dim(&input_dims[0]),
            subst.apply_dim(&kernel_dims[0]),
            Dim::Var(vg.fresh_dvar()),
            Dim::Var(vg.fresh_dvar()),
        ],
        input_prec,
    );
    if let Err(te) = unify(result_ty, &output_template, subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let resolved_output = subst.apply(&output_template);
    if let Type::Tensor(out_dims, out_prec) = &resolved_output {
        if out_dims.len() != 4 {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("conv2d result must be rank 4, got rank {}", out_dims.len()),
                vec![],
            ));
            return Type::Error;
        }
        if *out_prec != input_prec {
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "conv2d result precision must match input/kernel precision {}, got {}",
                    input_prec.name(),
                    out_prec.name()
                ),
                vec!["Insert explicit cast".to_string()],
            ));
            return Type::Error;
        }
        if let Err(te) = unify_dim(&out_dims[0], &input_dims[0], subst) {
            errors.push(te.into());
            return Type::Error;
        }
        if let Err(te) = unify_dim(&out_dims[1], &kernel_dims[0], subst) {
            errors.push(te.into());
            return Type::Error;
        }
    }

    subst.apply(&output_template)
}

fn check_matmul_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 2 {
        return Type::Error;
    }

    let lhs = subst.apply(&arg_tys[0]);
    let rhs = subst.apply(&arg_tys[1]);

    let (lhs_dims, lhs_prec) = match lhs {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("matmul expects tensor lhs, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };
    let (rhs_dims, rhs_prec) = match rhs {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("matmul expects tensor rhs, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if lhs_prec != rhs_prec {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "matmul requires matching precisions, got {} and {}",
                lhs_prec.name(),
                rhs_prec.name()
            ),
            vec!["Insert explicit cast".to_string()],
        ));
        return Type::Error;
    }
    if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "matmul expects rank-2 tensors, got rank {} and {}",
                lhs_dims.len(),
                rhs_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if let Err(te) = unify_dim(&lhs_dims[1], &rhs_dims[0], subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let canonical = Type::Tensor(
        vec![subst.apply_dim(&lhs_dims[0]), subst.apply_dim(&rhs_dims[1])],
        lhs_prec,
    );
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

fn check_reduction_signature(
    name: &str,
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 2 {
        return Type::Error;
    }

    let input_ty = subst.apply(&arg_tys[0]);
    let (dims, prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("{name} expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let axis = match arg_exprs.get(1).and_then(extract_int_literal) {
        Some(axis) if axis >= 0 => axis as usize,
        Some(axis) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("{name} requires non-negative axis, got {axis}"),
                vec![],
            ));
            return Type::Error;
        }
        None => return subst.apply(result_ty),
    };

    if axis >= dims.len() {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "{name} axis {axis} is out of bounds for rank {} tensor",
                dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }

    let mut out_dims = dims;
    out_dims.remove(axis);
    let canonical = Type::Tensor(out_dims, prec);
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

fn check_expand_signature(
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 3 {
        return Type::Error;
    }

    let input_ty = subst.apply(&arg_tys[0]);
    let (input_dims, input_prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expand expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let axis = match arg_exprs.get(1).and_then(extract_int_literal) {
        Some(axis) if axis >= 0 => axis as usize,
        Some(axis) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("expand requires non-negative axis, got {axis}"),
                vec![],
            ));
            return Type::Error;
        }
        None => return subst.apply(result_ty),
    };
    let size = match arg_exprs.get(2).and_then(extract_int_literal) {
        Some(size) if size > 0 => Dim::Lit(size),
        Some(size) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("expand requires positive size, got {size}"),
                vec![],
            ));
            return Type::Error;
        }
        None => match arg_exprs.get(2).and_then(symbolic_dim_ref_name) {
            Some(name) => Dim::Name(name.to_string()),
            None => return subst.apply(result_ty),
        },
    };

    let resolved_result = subst.apply(result_ty);
    let canonical = match resolved_result {
        Type::Tensor(out_dims, out_prec) => {
            if out_prec != input_prec {
                errors.push(CheckError::new(
                    CheckErrorKind::PrecisionMismatch,
                    format!(
                        "expand output precision {} does not match input precision {}",
                        out_prec.name(),
                        input_prec.name()
                    ),
                    vec![],
                ));
                return Type::Error;
            }
            if out_dims.len() == input_dims.len() + 1 {
                if axis > input_dims.len() {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "expand insert axis {axis} is out of bounds for rank {} tensor",
                            input_dims.len()
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                let mut expected = input_dims.clone();
                expected.insert(axis, size.clone());
                Type::Tensor(expected, input_prec)
            } else if out_dims.len() == input_dims.len() {
                if axis >= input_dims.len() {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "expand axis {axis} is out of bounds for rank {} tensor",
                            input_dims.len()
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                let mut expected = input_dims.clone();
                expected[axis] = size.clone();
                Type::Tensor(expected, input_prec)
            } else {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "expand output rank {} must equal input rank {} or {}",
                        out_dims.len(),
                        input_dims.len(),
                        input_dims.len() + 1
                    ),
                    vec![],
                ));
                return Type::Error;
            }
        }
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expand expects tensor output, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

fn extract_int_literal(expr: &deep::Expr) -> Option<i64> {
    match expr {
        deep::Expr::Atom(deep::Atom::Int(n), _) => Some(*n),
        deep::Expr::List(list, _) if get_tag(list) == Some("lit") => {
            children(list).first().and_then(|child| match child {
                deep::Expr::Atom(deep::Atom::Int(n), _) => Some(*n),
                _ => None,
            })
        }
        deep::Expr::List(list, _) if get_tag(list) == Some("app") => {
            let app_children = children(list);
            match (app_children.first(), app_children.get(1)) {
                (Some(func), Some(arg)) if is_builtin_var(func, "neg") => {
                    extract_int_literal(arg).map(|value| -value)
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn is_builtin_var(expr: &deep::Expr, expected: &str) -> bool {
    let deep::Expr::List(list, _) = expr else {
        return false;
    };
    get_tag(list) == Some("var") && children(list).first().and_then(symbol_name) == Some(expected)
}

fn symbolic_dim_ref_name(expr: &deep::Expr) -> Option<&str> {
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

#[allow(clippy::too_many_arguments)]
fn infer_fn(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    // kids[0] = (params {} x1 ... xn)
    // kids[1] = body
    let params = extract_params(&kids[0], vg, adt_reg);
    let mut param_types = Vec::new();
    let mut fn_env = env.clone();

    for (pname, ty_ann) in &params {
        let ty = ty_ann.clone().unwrap_or_else(|| vg.fresh_type());
        fn_env.bind(pname.clone(), Scheme::mono(ty.clone()));
        param_types.push(ty);
    }

    // Snapshot the dimension variables introduced by the declared parameter
    // signatures. If the body later forces any of them to a concrete literal,
    // the signature's polymorphism claim is self-contradictory (Nautilus Bug 2):
    // the user wrote `def f[m, n](x: tensor[m, n, f32])` but the body body
    // demands `tensor[2, 2, f32]`. We flag this post-body so legitimate
    // polymorphic uses (where the dvar stays unbound) still type-check.
    let mut declared_dvars: Vec<DimVar> = Vec::new();
    for t in &param_types {
        for dv in crate::env::free_dvars(t) {
            if !declared_dvars.contains(&dv) {
                declared_dvars.push(dv);
            }
        }
    }

    let body = if kids.len() > 1 {
        &kids[1]
    } else {
        return Type::Error;
    };
    let body_ty = infer_expr(
        body,
        &mut fn_env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    for dv in &declared_dvars {
        let resolved = subst.apply_dim(&Dim::Var(*dv));
        if let Dim::Lit(n) = resolved {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "polymorphic dim variable forced to concrete Lit({n}) by function body — \
                     declared dim parameters must remain polymorphic"
                ),
                vec![
                    "Replace the polymorphic dim with the concrete literal in the signature, or \
                     ensure the body does not pin the dim to a specific size"
                        .to_string(),
                ],
            ));
        }
    }

    let resolved_params: Vec<Type> = param_types.iter().map(|t| subst.apply(t)).collect();
    let resolved_body = subst.apply(&body_ty);

    Type::Fn(resolved_params, Box::new(resolved_body))
}

/// Extract parameter names (and optional type annotations) from (params {} x1 ... xn).
/// Each param can be a bare symbol, a metadata-annotated symbol, or a legacy
/// `(name {type: T})` helper pair.
fn extract_params(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
) -> Vec<(String, Option<Type>)> {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list);
            let elems = if tag == Some("params") {
                children(list)
            } else {
                &list.elements
            };
            elems
                .iter()
                .filter_map(|e| match e {
                    deep::Expr::Atom(deep::Atom::Symbol(s), _) => Some((s.to_string(), None)),
                    deep::Expr::MetaExpr(meta, _) => {
                        let deep::Expr::Atom(deep::Atom::Symbol(name), _) = meta.expr.as_ref()
                        else {
                            return None;
                        };
                        let ty_ann = meta.entries.iter().find_map(|(key, val)| {
                            (key == "type").then(|| {
                                deep_type_to_resolved_type(val, vg, adt_reg, &mut HashMap::new())
                            })
                        });
                        Some((name.to_string(), ty_ann))
                    }
                    deep::Expr::List(plist, _) => {
                        // Typed param: (name {type: T}) — elements[0] is the name symbol,
                        // elements[1] is the metadata map with type annotation
                        if let Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)) =
                            plist.elements.first()
                        {
                            let mut ty_ann = None;
                            if let Some(deep::Expr::Map(meta, _)) = plist.elements.get(1) {
                                for (key, val) in &meta.entries {
                                    if key == "type" {
                                        ty_ann = Some(deep_type_to_resolved_type(
                                            val,
                                            vg,
                                            adt_reg,
                                            &mut HashMap::new(),
                                        ));
                                    }
                                }
                            }
                            Some((name.to_string(), ty_ann))
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect()
        }
        _ => vec![],
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_let(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    // kids[0] = (bind {} x1 e1 x2 e2 ...)
    // kids[1] = body
    let mut let_env = env.clone();

    if let deep::Expr::List(bind_list, _) = &kids[0] {
        let bind_children = children(bind_list);
        // Process pairs: name, expr
        let mut i = 0;
        while i + 1 < bind_children.len() {
            if let Some(name) = symbol_name(&bind_children[i]) {
                let expr_ty = infer_expr(
                    &bind_children[i + 1],
                    &mut let_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
                let scheme = let_env.generalize(&expr_ty, subst);
                let_env.bind(name.to_string(), scheme);
            }
            i += 2;
        }
    }

    infer_expr(
        &kids[1],
        &mut let_env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    )
}

#[allow(clippy::too_many_arguments)]
fn infer_if(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 3 {
        return Type::Error;
    }

    let cond_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    // Condition should be bool (or tensor[D, bool])
    if let Err(_te) = unify(&cond_ty, &Type::Prim(Prim::Bool), subst) {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("if condition must be bool, got {}", subst.apply(&cond_ty)),
            vec![],
        ));
    }

    let then_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let else_ty = infer_expr(
        &kids[2],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    match unify(&then_ty, &else_ty, subst) {
        Ok(()) => subst.apply(&then_ty),
        Err(te) => {
            errors.push(te.into());
            subst.apply(&then_ty)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_match(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let scrutinee_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    let mut result_ty: Option<Type> = None;
    let mut covered_variants: Vec<String> = Vec::new();
    let mut has_wildcard = false;

    for arm_expr in &kids[1..] {
        if let deep::Expr::List(arm_list, _) = arm_expr
            && get_tag(arm_list) == Some("arm")
        {
            let arm_kids = children(arm_list);
            // arm_kids[0] = pattern, arm_kids[1] = guard (usually ()), arm_kids[2] = body
            if arm_kids.len() >= 3 {
                let mut arm_env = env.clone();
                let pat = &arm_kids[0];
                pattern_bindings(
                    pat,
                    &scrutinee_ty,
                    &mut arm_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    &mut covered_variants,
                    &mut has_wildcard,
                );

                let body_ty = infer_expr(
                    &arm_kids[2],
                    &mut arm_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );

                match &result_ty {
                    None => result_ty = Some(body_ty),
                    Some(prev) => {
                        if let Err(te) = unify(prev, &body_ty, subst) {
                            errors.push(te.into());
                        }
                        result_ty = Some(subst.apply(prev));
                    }
                }
            }
        }
    }

    // Exhaustiveness check (wildcard covers everything)
    if !has_wildcard {
        let resolved_scrutinee = subst.apply(&scrutinee_ty);
        if let Type::Adt(ref adt_name, _) = resolved_scrutinee
            && let Some(all_variants) = adt_reg.variant_names(adt_name)
        {
            let missing: Vec<&String> = all_variants
                .iter()
                .filter(|v| !covered_variants.contains(v))
                .collect();
            if !missing.is_empty() {
                let names: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                errors.push(CheckError::new(
                    CheckErrorKind::NonExhaustiveMatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("non-exhaustive match: missing variants {:?}", names),
                    ),
                    vec![],
                ));
            }
        }
    }

    result_ty.unwrap_or(Type::Error)
}

#[allow(clippy::too_many_arguments)]
fn pattern_bindings(
    pat: &deep::Expr,
    scrutinee_ty: &Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    covered_variants: &mut Vec<String>,
    has_wildcard: &mut bool,
) {
    if let deep::Expr::List(list, _) = pat {
        let tag = get_tag(list).unwrap_or("");
        let kids = children(list);
        match tag {
            "pat-var" => {
                if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                    let resolved = subst.apply(scrutinee_ty);
                    env.bind(name.to_string(), Scheme::mono(resolved));
                }
            }
            "pat-wild" => {
                // Wildcard covers everything
                *has_wildcard = true;
            }
            "pat-lit" => {
                // No bindings, but value should match scrutinee type
            }
            "pat-ctor" => {
                if let Some(ctor_name) = kids.first().and_then(|e| symbol_name(e)) {
                    covered_variants.push(ctor_name.to_string());

                    // Look up constructor in env and decompose
                    if let Some(scheme) = env
                        .lookup(ctor_name)
                        .or_else(|| env.lookup_terminal_unique(ctor_name))
                    {
                        let scheme = scheme.clone();
                        let ctor_ty = env.instantiate(&scheme, vg);
                        // Unify the result of the constructor with scrutinee type
                        match &ctor_ty {
                            Type::Fn(arg_types, ret) => {
                                let _ = unify(ret, scrutinee_ty, subst);
                                // Bind sub-patterns to argument types
                                for (i, sub_pat) in kids[1..].iter().enumerate() {
                                    if i < arg_types.len() {
                                        let resolved = subst.apply(&arg_types[i]);
                                        pattern_bindings(
                                            sub_pat,
                                            &resolved,
                                            env,
                                            vg,
                                            subst,
                                            adt_reg,
                                            errors,
                                            covered_variants,
                                            has_wildcard,
                                        );
                                    }
                                }
                            }
                            _ => {
                                // Nullary constructor
                                let _ = unify(&ctor_ty, scrutinee_ty, subst);
                            }
                        }
                    }
                }
            }
            "pat-as" => {
                // (pat-as {} name inner_pat): bind name to scrutinee type, recurse into inner_pat
                if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                    let resolved = subst.apply(scrutinee_ty);
                    env.bind(name.to_string(), Scheme::mono(resolved));
                }
                if kids.len() >= 2 {
                    pattern_bindings(
                        &kids[1],
                        scrutinee_ty,
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        covered_variants,
                        has_wildcard,
                    );
                }
            }
            "pat-record" => {
                // (pat-record {} TypeName (kv {} k1 p1) ...): validate against ADT registry
                // kids[0] = TypeName, kids[1..] = (kv {} key pat)
                if let Some(ctor_name) = kids.first().and_then(|e| symbol_name(e)) {
                    covered_variants.push(ctor_name.to_string());

                    // Look up variant in ADT registry to get field types
                    let variant_info = adt_reg
                        .lookup_variant(ctor_name)
                        .or_else(|| adt_reg.lookup_variant_terminal_unique(ctor_name));
                    let declared_fields: std::collections::HashMap<&str, &Type> = variant_info
                        .map(|(_, vi)| {
                            vi.fields
                                .iter()
                                .filter_map(|(name, ty)| name.as_deref().map(|n| (n, ty)))
                                .collect()
                        })
                        .unwrap_or_default();

                    for kv_expr in kids.iter().skip(1) {
                        if let deep::Expr::List(kv_list, _) = kv_expr
                            && get_tag(kv_list) == Some("kv")
                        {
                            let kv_kids = children(kv_list);
                            if kv_kids.len() >= 2 {
                                let field_name = symbol_name(&kv_kids[0]);
                                // Look up declared field type — reject unknown fields
                                let field_ty = match field_name {
                                    Some(n) => match declared_fields.get(n) {
                                        Some(ty) => (*ty).clone(),
                                        None if !declared_fields.is_empty() => {
                                            // Unknown field name — error
                                            errors.push(CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                format!(
                                                    "unknown record field '{}' in pattern for {}",
                                                    n, ctor_name
                                                ),
                                                vec![format!(
                                                    "known fields: {:?}",
                                                    declared_fields.keys().collect::<Vec<_>>()
                                                )],
                                            ));
                                            Type::Error
                                        }
                                        None => vg.fresh_type(), // no ADT info available
                                    },
                                    None => vg.fresh_type(),
                                };
                                pattern_bindings(
                                    &kv_kids[1],
                                    &field_ty,
                                    env,
                                    vg,
                                    subst,
                                    adt_reg,
                                    errors,
                                    covered_variants,
                                    has_wildcard,
                                );
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_pipe(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let mut current_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    for stage in &kids[1..] {
        let stage_ty = infer_expr(
            stage,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
        let ret_tv = vg.fresh_type();
        let expected = Type::Fn(vec![current_ty.clone()], Box::new(ret_tv.clone()));

        match unify(&stage_ty, &expected, subst) {
            Ok(()) => {
                current_ty = subst.apply(&ret_tv);
            }
            Err(te) => {
                errors.push(te.into());
                return Type::Error;
            }
        }
    }

    current_ty
}

#[allow(clippy::too_many_arguments)]
fn infer_tuple(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    let elems: Vec<Type> = kids
        .iter()
        .map(|e| infer_expr(e, env, vg, subst, adt_reg, errors, typed_nodes, total_nodes))
        .collect();
    Type::Tuple(elems)
}

#[allow(clippy::too_many_arguments)]
fn infer_tuple_get(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    let tuple_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&tuple_ty);

    let index = match &kids[1] {
        deep::Expr::Atom(deep::Atom::Int(n), _) => *n as usize,
        _ => {
            return Type::Error;
        }
    };

    match resolved {
        Type::Tuple(ref elems) => {
            if index < elems.len() {
                elems[index].clone()
            } else {
                errors.push(CheckError::new(
                    CheckErrorKind::TupleIndexOutOfBounds,
                    format!(
                        "tuple index {} out of bounds for tuple of size {}",
                        index,
                        elems.len()
                    ),
                    vec![],
                ));
                Type::Error
            }
        }
        Type::Error => Type::Error,
        _ => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expected tuple type, got {resolved}"),
                vec![],
            ));
            Type::Error
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_cast(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    let expr_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&expr_ty);

    // kids[1] = (t-prim {} new_precision)
    let new_prec = match deep_type_to_resolved_type(&kids[1], vg, adt_reg, &mut HashMap::new()) {
        Type::Prim(p) => p,
        _ => return Type::Error,
    };

    match resolved {
        Type::Tensor(dims, _) => {
            if !new_prec.is_valid_tensor_precision() {
                errors.push(CheckError::new(
                    CheckErrorKind::UnsupportedTensorPrecision,
                    format!(
                        "cannot cast tensor to unsupported element precision `{}` \
                         (supported: f32, f64, bool, int8, int32, int64)",
                        new_prec.name()
                    ),
                    vec!["Cast to f32 or an integer precision instead".to_string()],
                ));
                return Type::Error;
            }
            Type::Tensor(dims, new_prec)
        }
        Type::Prim(_) => {
            // Scalar cast targets: supported widths are the same as
            // valid tensor precisions plus host f64. Reject f16, bf16,
            // f8e4m3 which have no scalar representation either.
            if !new_prec.is_valid_scalar_cast_target() {
                errors.push(CheckError::new(
                    CheckErrorKind::UnsupportedTensorPrecision,
                    format!(
                        "cannot cast scalar to unsupported precision `{}` \
                         (supported: f32, f64, bool, int8, int32, int64)",
                        new_prec.name()
                    ),
                    vec!["Use a supported scalar precision".to_string()],
                ));
                return Type::Error;
            }
            Type::Prim(new_prec)
        }
        Type::Error => Type::Error,
        _ => {
            errors.push(CheckError::new(
                CheckErrorKind::CastNonTensor,
                format!("cast requires tensor or prim type, got {resolved}"),
                vec![],
            ));
            Type::Error
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_grad(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let f_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&f_ty);

    match resolved {
        Type::Fn(args, ret) => {
            let ret = *ret;
            if !grad_output_supported(&ret) {
                errors.push(CheckError::new(
                    CheckErrorKind::Other,
                    format!("grad requires a scalar floating output, got {}", ret),
                    vec!["Reduce the function result to a scalar before applying grad".to_string()],
                ));
                return Type::Error;
            }

            match grad_result_type(list, &args, errors) {
                Some(grad_ret) => Type::Fn(args, Box::new(grad_ret)),
                None => Type::Error,
            }
        }
        Type::Error => Type::Error,
        _ => {
            // Can't determine function structure, return fresh var
            vg.fresh_type()
        }
    }
}

fn grad_output_supported(ty: &Type) -> bool {
    match ty {
        Type::Prim(prim) => prim.is_float(),
        Type::Tensor(dims, prim) => dims.is_empty() && prim.is_float(),
        _ => false,
    }
}

fn grad_result_type(
    list: &deep::List,
    args: &[Type],
    errors: &mut Vec<CheckError>,
) -> Option<Type> {
    let targets = if let Some(indices) = grad_wrt_indices(list, errors)? {
        let mut selected = Vec::with_capacity(indices.len());
        for index in indices {
            let Some(arg) = args.get(index) else {
                errors.push(CheckError::new(
                    CheckErrorKind::ArityMismatch,
                    format!(
                        "grad `wrt` index {} is out of bounds for function with {} parameters",
                        index,
                        args.len()
                    ),
                    vec![],
                ));
                return None;
            };
            let Some(grad_ty) = grad_argument_type(arg) else {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!("grad `wrt` index {index} is not differentiable"),
                    vec!["Select floating scalar or tensor parameters in `wrt`".to_string()],
                ));
                return None;
            };
            selected.push(grad_ty);
        }
        selected
    } else {
        args.iter().filter_map(grad_argument_type).collect()
    };

    Some(match targets.as_slice() {
        [] => Type::Unit,
        [single] => single.clone(),
        _ => Type::Tuple(targets),
    })
}

fn grad_wrt_indices(list: &deep::List, errors: &mut Vec<CheckError>) -> Option<Option<Vec<usize>>> {
    let kids = children(list);
    let Some(wrt_expr) = kids.get(1) else {
        return Some(None);
    };

    match wrt_expr {
        deep::Expr::List(tuple, _) if get_tag(tuple) == Some("tuple") => {
            let mut indices = Vec::new();
            for item in children(tuple) {
                let Some(index) = extract_int_literal(item) else {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        "grad `wrt` tuple must contain integer parameter indices".to_string(),
                        vec![],
                    ));
                    return None;
                };
                if index < 0 {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!("grad `wrt` index must be non-negative, got {index}"),
                        vec![],
                    ));
                    return None;
                }
                indices.push(index as usize);
            }
            Some(Some(indices))
        }
        other => {
            let Some(index) = extract_int_literal(other) else {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    "grad `wrt` must be an integer parameter index or tuple of indices".to_string(),
                    vec![],
                ));
                return None;
            };
            if index < 0 {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!("grad `wrt` index must be non-negative, got {index}"),
                    vec![],
                ));
                return None;
            }
            Some(Some(vec![index as usize]))
        }
    }
}

fn grad_argument_type(arg: &Type) -> Option<Type> {
    match arg {
        Type::Prim(prim) if prim.is_float() => Some(Type::Prim(*prim)),
        Type::Tensor(dims, prim) if prim.is_float() => Some(Type::Tensor(dims.clone(), *prim)),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_vmap(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let axis = kids.get(1).and_then(extract_int_literal).unwrap_or(0);
    if axis < 0 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!("vmap axis must be non-negative, got {axis}"),
            vec!["Use `vmap(f)` or `vmap(f, axis=n)` with n >= 0".to_string()],
        ));
        return Type::Error;
    }
    let axis = axis as usize;

    let f_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&f_ty);

    match resolved {
        Type::Fn(args, ret) => {
            let batch_dim = Dim::Var(vg.fresh_dvar());
            let args = args
                .iter()
                .map(|arg| vmap_transform_param_type(arg, axis, &batch_dim))
                .collect::<Result<Vec<_>, _>>();
            let ret = vmap_transform_result_type(&ret, axis, &batch_dim);

            match (args, ret) {
                (Ok(args), Ok(ret)) => Type::Fn(args, Box::new(ret)),
                (Err(message), _) | (_, Err(message)) => {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        message,
                        vec![
                            "Choose an axis that is in bounds for every vmapped tensor".to_string(),
                        ],
                    ));
                    Type::Error
                }
            }
        }
        Type::Error => Type::Error,
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("vmap expects a function, got {other}"),
                vec!["Apply `vmap` to a named function or inline lambda".to_string()],
            ));
            Type::Error
        }
    }
}

fn vmap_transform_param_type(ty: &Type, axis: usize, batch_dim: &Dim) -> Result<Type, String> {
    match ty {
        Type::Tensor(dims, precision) => {
            if axis > dims.len() {
                return Err(format!(
                    "vmap axis {axis} is out of bounds for rank {} tensor",
                    dims.len()
                ));
            }
            let mut dims = dims.clone();
            dims.insert(axis, batch_dim.clone());
            Ok(Type::Tensor(dims, *precision))
        }
        Type::Tuple(elements) => Ok(Type::Tuple(
            elements
                .iter()
                .map(|element| vmap_transform_param_type(element, axis, batch_dim))
                .collect::<Result<_, _>>()?,
        )),
        other => Ok(other.clone()),
    }
}

fn vmap_transform_result_type(ty: &Type, axis: usize, batch_dim: &Dim) -> Result<Type, String> {
    match ty {
        Type::Prim(precision) => Ok(Type::Tensor(vec![batch_dim.clone()], *precision)),
        Type::Tensor(dims, precision) => {
            if axis > dims.len() {
                return Err(format!(
                    "vmap axis {axis} is out of bounds for rank {} tensor",
                    dims.len()
                ));
            }
            let mut dims = dims.clone();
            dims.insert(axis, batch_dim.clone());
            Ok(Type::Tensor(dims, *precision))
        }
        Type::Tuple(elements) => Ok(Type::Tuple(
            elements
                .iter()
                .map(|element| vmap_transform_result_type(element, axis, batch_dim))
                .collect::<Result<_, _>>()?,
        )),
        other => Ok(other.clone()),
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_def(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    let name = match symbol_name(&kids[0]) {
        Some(n) => n.to_string(),
        None => return Type::Error,
    };

    let body_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let scheme = env.generalize(&body_ty, subst);
    env.bind(name, scheme);
    body_ty
}

// ── Type conversion from Deep AST ───────────────────────────────

/// Convert a Deep type expression to an internal Type.
/// `tvar_map` maps type variable names to TypeVars (created on demand).
/// `dvar_map` maps dimension variable names to DimVars (created on demand).
fn deep_type_to_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    tvar_map: &mut HashMap<String, TypeVar>,
) -> Type {
    let mut dvar_map = HashMap::new();
    deep_type_to_type_inner(expr, vg, tvar_map, &mut dvar_map)
}

fn deep_type_to_type_inner(
    expr: &deep::Expr,
    vg: &mut VarGen,
    tvar_map: &mut HashMap<String, TypeVar>,
    dvar_map: &mut HashMap<String, DimVar>,
) -> Type {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list).unwrap_or("");
            let kids = children(list);
            match tag {
                "t-prim" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        Prim::parse_name(name)
                            .map(Type::Prim)
                            .unwrap_or(Type::Error)
                    } else {
                        Type::Error
                    }
                }
                "t-var" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        if name == "_" {
                            vg.fresh_type()
                        } else {
                            let tv = *tvar_map
                                .entry(name.to_string())
                                .or_insert_with(|| vg.fresh_tvar());
                            Type::Var(tv)
                        }
                    } else {
                        Type::Error
                    }
                }
                "t-fn" => {
                    if kids.is_empty() {
                        return Type::Error;
                    }
                    let args: Vec<Type> = kids[..kids.len() - 1]
                        .iter()
                        .map(|c| deep_type_to_type_inner(c, vg, tvar_map, dvar_map))
                        .collect();
                    let ret =
                        deep_type_to_type_inner(&kids[kids.len() - 1], vg, tvar_map, dvar_map);
                    Type::Fn(args, Box::new(ret))
                }
                "t-tensor" => {
                    if kids.is_empty() {
                        return Type::Error;
                    }
                    let prec_expr = &kids[kids.len() - 1];
                    let prec = match deep_type_to_type_inner(prec_expr, vg, tvar_map, dvar_map) {
                        Type::Prim(p) => p,
                        _ => return Type::Error,
                    };
                    let dims: Vec<Dim> = kids[..kids.len() - 1]
                        .iter()
                        .filter_map(|c| parse_dim(c, vg, dvar_map))
                        .collect();
                    Type::Tensor(dims, prec)
                }
                "t-adt" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        let args: Vec<Type> = kids[1..]
                            .iter()
                            .map(|c| deep_type_to_type_inner(c, vg, tvar_map, dvar_map))
                            .collect();
                        Type::Adt(name.to_string(), args)
                    } else {
                        Type::Error
                    }
                }
                "t-tuple" => {
                    let elems: Vec<Type> = kids
                        .iter()
                        .map(|c| deep_type_to_type_inner(c, vg, tvar_map, dvar_map))
                        .collect();
                    Type::Tuple(elems)
                }
                "t-unit" => Type::Unit,
                _ => Type::Error,
            }
        }
        _ => Type::Error,
    }
}

/// Parse a dimension expression from Deep AST, with support for dim variables.
fn parse_dim(
    expr: &deep::Expr,
    vg: &mut VarGen,
    dvar_map: &mut HashMap<String, DimVar>,
) -> Option<Dim> {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list).unwrap_or("");
            let kids = children(list);
            match tag {
                "d-name" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        if name == "*" {
                            Some(Dim::Wildcard)
                        } else {
                            Some(Dim::Name(name.to_string()))
                        }
                    } else {
                        None
                    }
                }
                "d-var" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        let dv = *dvar_map
                            .entry(name.to_string())
                            .or_insert_with(|| vg.fresh_dvar());
                        Some(Dim::Var(dv))
                    } else {
                        Some(vg.fresh_dim())
                    }
                }
                "d-lit" => {
                    if let Some(deep::Expr::Atom(deep::Atom::Int(n), _)) = kids.first() {
                        Some(Dim::Lit(*n))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(src: &str) -> InferResult {
        let exprs = chelis_deep::parser::parse_str(src).unwrap();
        infer_program(&exprs)
    }

    fn checked_surf(src: &str) -> CheckedProgram {
        let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        check_phase0e_program(&exprs).expect("phase 0e check")
    }

    fn missing_shape_sensitive_app(expr: &deep::Expr) -> Option<String> {
        match expr {
            deep::Expr::List(list, _) => {
                if get_tag(list) == Some("app")
                    && is_shape_sensitive_app(list)
                    && !list
                        .elements
                        .get(1)
                        .and_then(|expr| match expr {
                            deep::Expr::Map(meta, _) => Some(meta),
                            _ => None,
                        })
                        .is_some_and(|meta| meta.entries.iter().any(|(key, _)| key == "type"))
                {
                    return Some(
                        chelis_deep::printer::print_canonical(std::slice::from_ref(expr))
                            .replace('\n', " ")
                            .trim()
                            .to_string(),
                    );
                }
                for child in &list.elements {
                    if let Some(missing) = missing_shape_sensitive_app(child) {
                        return Some(missing);
                    }
                }
                None
            }
            deep::Expr::Map(map, _) => map
                .entries
                .iter()
                .find_map(|(_, value)| missing_shape_sensitive_app(value)),
            deep::Expr::MetaExpr(meta, _) => {
                missing_shape_sensitive_app(&meta.expr).or_else(|| {
                    meta.entries
                        .iter()
                        .find_map(|(_, value)| missing_shape_sensitive_app(value))
                })
            }
            deep::Expr::Atom(_, _) => None,
        }
    }

    fn is_shape_sensitive_app(list: &deep::List) -> bool {
        get_tag(list) == Some("app")
            && phase0e_builtin_name(list).is_some_and(super::is_phase0e_shape_sensitive_builtin)
    }

    fn check_ok(src: &str) {
        let result = check(src);
        assert!(
            result.errors.is_empty(),
            "expected no errors, got: {:?}",
            result.errors
        );
    }

    fn check_err(src: &str, expected_kind: CheckErrorKind) {
        let result = check(src);
        assert!(
            !result.errors.is_empty(),
            "expected error {expected_kind:?}, got none"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| std::mem::discriminant(&e.kind) == std::mem::discriminant(&expected_kind)),
            "expected {expected_kind:?}, got: {:?}",
            result.errors
        );
    }

    // ── Variable / literal tests ─────────────────────────────────

    #[test]
    fn lit_int32() {
        check_ok("(def {} x (lit {type: (t-prim {} int32)} 42))");
    }

    #[test]
    fn lit_f32() {
        check_ok("(def {} x (lit {type: (t-prim {} f32)} 3.0))");
    }

    #[test]
    fn lit_bool() {
        check_ok("(def {} x (lit {type: (t-prim {} bool)} true))");
    }

    #[test]
    fn lit_string() {
        check_ok(r#"(def {} x (lit {type: (t-prim {} string)} "hello"))"#);
    }

    #[test]
    fn scalar_add_is_allowed() {
        check_ok(
            "(def {} x
                (app {} (var {} add)
                    (lit {type: (t-prim {} int64)} 2)
                    (lit {type: (t-prim {} int64)} 3)))",
        );
    }

    #[test]
    fn integer_mod_and_bitwise_builtins_are_allowed() {
        check_ok(
            "(def {} bits
                (app {} (var {} bitxor)
                    (app {} (var {} bitand)
                        (lit {type: (t-prim {} int64)} 7)
                        (lit {type: (t-prim {} int64)} 3))
                    (app {} (var {} shl)
                        (lit {type: (t-prim {} int64)} 1)
                        (lit {type: (t-prim {} int64)} 2))))
             (def {} rem
                (app {} (var {} mod)
                    (lit {type: (t-prim {} int64)} 17)
                    (lit {type: (t-prim {} int64)} 5)))
             (def {} shrunk
                (app {} (var {} shr)
                    (lit {type: (t-prim {} int64)} 8)
                    (lit {type: (t-prim {} int64)} 1)))",
        );
    }

    #[test]
    fn string_len_builtin_is_allowed() {
        check_ok(
            r#"(def {} x
                (app {} (var {} string_len)
                    (lit {type: (t-prim {} string)} "hé")))"#,
        );
    }

    #[test]
    fn string_predicates_and_transforms_are_allowed() {
        check_ok(
            r#"(def {} ok
                (if {}
                    (app {} (var {} and)
                        (app {} (var {} string_contains)
                            (lit {type: (t-prim {} string)} "ckpt-7.safetensors")
                            (lit {type: (t-prim {} string)} "ckpt"))
                        (app {} (var {} string_ends_with)
                            (app {} (var {} string_slice)
                                (app {} (var {} string_trim)
                                    (lit {type: (t-prim {} string)} "  ckpt-7.safetensors  "))
                                (lit {type: (t-prim {} int64)} 7)
                                (lit {type: (t-prim {} int64)} 12))
                            (lit {type: (t-prim {} string)} ".safetensors")))
                    (lit {type: (t-prim {} bool)} true)
                    (lit {type: (t-prim {} bool)} false)))"#,
        );
    }

    #[test]
    fn to_int_builtin_uses_prelude_option_without_local_deftype() {
        check_ok(
            r#"(def {} parsed
                (match {}
                    (app {} (var {} to_int)
                        (lit {type: (t-prim {} string)} "42"))
                    (arm {} (pat-ctor {} Some (pat-var {} n)) () (var {} n))
                    (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} int64)} 0))))"#,
        );
    }

    #[test]
    fn to_float_builtin_uses_prelude_option_without_local_deftype() {
        check_ok(
            r#"(def {} parsed
                (match {}
                    (app {} (var {} to_float)
                        (lit {type: (t-prim {} string)} "0.125"))
                    (arm {} (pat-ctor {} Some (pat-var {} x)) () (var {} x))
                    (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} f64)} 1.0))))"#,
        );
    }

    #[test]
    fn to_int_builtin_rejects_non_string_input() {
        check_err(
            r#"(def {} parsed
                (app {} (var {} to_int)
                    (lit {type: (t-prim {} int32)} 7)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn mod_rejects_float_input() {
        check_err(
            r#"(def {} bad
                (app {} (var {} mod)
                    (lit {type: (t-prim {} f64)} 7.0)
                    (lit {type: (t-prim {} f64)} 3.0)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn bitand_rejects_mismatched_integer_widths() {
        check_err(
            r#"(def {} bad
                (app {} (var {} bitand)
                    (lit {type: (t-prim {} int32)} 7)
                    (lit {type: (t-prim {} int64)} 3)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn shl_rejects_non_integer_shift_amount() {
        check_err(
            r#"(def {} bad
                (app {} (var {} shl)
                    (lit {type: (t-prim {} int64)} 1)
                    (lit {type: (t-prim {} f64)} 2.0)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn string_slice_rejects_non_string_input() {
        check_err(
            r#"(def {} bad
                (app {} (var {} string_slice)
                    (lit {type: (t-prim {} int32)} 7)
                    (lit {type: (t-prim {} int64)} 0)
                    (lit {type: (t-prim {} int64)} 1)))"#,
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn rank_builtin_requires_tensor_input() {
        check_err(
            "(def {} x
                (app {} (var {} rank)
                    (lit {type: (t-prim {} int64)} 2)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn shape_builtin_accepts_tensor_input() {
        check_ok(
            r#"(def {} x
                (lit {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} 0))
               (def {} dim
                (app {} (var {} shape)
                    (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x)
                    (lit {type: (t-prim {} int32)} 1)))"#,
        );
    }

    #[test]
    fn shape_builtin_rejects_negative_axis_when_rank_is_known() {
        check_err(
            r#"(def {} x
                (lit {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} 0))
               (def {} dim
                (app {} (var {} shape)
                    (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x)
                    (lit {type: (t-prim {} int32)} -1)))"#,
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn lit_default_int() {
        check_ok("(def {} x (lit {} 42))");
    }

    #[test]
    fn lit_default_float() {
        check_ok("(def {} x (lit {} 3.14))");
    }

    #[test]
    fn unbound_variable() {
        check_err(
            "(def {} x (var {} unknown))",
            CheckErrorKind::UnboundVariable,
        );
    }

    #[test]
    fn var_lookup_defined() {
        check_ok(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (var {} x))",
        );
    }

    // ── Application tests ────────────────────────────────────────

    #[test]
    fn app_add_tensors() {
        check_ok(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn app_precision_mismatch() {
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bf16))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    #[test]
    fn app_not_a_function() {
        check_err(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (app {} (var {} x) (lit {type: (t-prim {} int32)} 1)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn app_dimension_mismatch() {
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} seq) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    // ── Lambda tests ─────────────────────────────────────────────

    #[test]
    fn fn_identity() {
        check_ok("(def {} id (fn {} (params {} x) (var {} x)))");
    }

    #[test]
    fn fn_two_params() {
        check_ok("(def {} f (fn {} (params {} x y) (var {} x)))");
    }

    #[test]
    fn fn_with_body_app() {
        check_ok("(def {} f (fn {} (params {} x y) (app {} (var {} add) (var {} x) (var {} y))))");
    }

    // ── Let tests ────────────────────────────────────────────────

    #[test]
    fn let_simple() {
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 42))
                 (var {} x)))",
        );
    }

    #[test]
    fn let_multiple_bindings() {
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 1) y (lit {type: (t-prim {} f32)} 2.0))
                 (var {} x)))",
        );
    }

    #[test]
    fn let_scoping() {
        // Variable defined in let should be usable in body
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 42))
                 (var {} x)))",
        );
    }

    // ── If tests ─────────────────────────────────────────────────

    #[test]
    fn if_correct() {
        check_ok(
            "(def {} x (lit {type: (t-prim {} bool)} true))
             (def {} a (lit {type: (t-prim {} int32)} 1))
             (def {} b (lit {type: (t-prim {} int32)} 2))
             (def {} c (if {} (var {} x) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn if_branch_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-prim {} bool)} true))
             (def {} a (lit {type: (t-prim {} int32)} 1))
             (def {} b (lit {type: (t-prim {} f32)} 2.0))
             (def {} c (if {} (var {} x) (var {} a) (var {} b)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // ── Pipe tests ───────────────────────────────────────────────

    #[test]
    fn pipe_simple() {
        check_ok(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))
             (def {} result (pipe {} (lit {type: (t-prim {} f32)} 1.0) (var {} f)))",
        );
    }

    #[test]
    fn pipe_chain() {
        check_ok(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))
             (def {} result (pipe {} (lit {type: (t-prim {} f32)} 1.0) (var {} f) (var {} f)))",
        );
    }

    // ── Tuple tests ──────────────────────────────────────────────

    #[test]
    fn tuple_creation() {
        check_ok(
            "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} f32)} 2.0)))",
        );
    }

    #[test]
    fn tuple_get_valid() {
        check_ok(
            "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} f32)} 2.0)))
             (def {} x (tuple-get {} (var {} t) 0))",
        );
    }

    #[test]
    fn tuple_get_out_of_bounds() {
        check_err(
            "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1)))
             (def {} x (tuple-get {} (var {} t) 5))",
            CheckErrorKind::TupleIndexOutOfBounds,
        );
    }

    // ── Cast tests ───────────────────────────────────────────────

    #[test]
    fn cast_tensor() {
        // Cast to a precision the Phase 0f tensor backend supports.
        // Reduced-float targets (bf16/f16/f64/f8e4m3) are rejected — see
        // `cast_tensor_rejects_unsupported_precision` below.
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (cast {} (var {} x) (t-prim {} int32)))",
        );
    }

    #[test]
    fn cast_prim() {
        check_ok(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (cast {} (var {} x) (t-prim {} f32)))",
        );
    }

    #[test]
    fn cast_accepts_alias_precision() {
        check_ok(
            "(typealias {} Floaty () (t-prim {} f32))
             (def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (cast {} (var {} x) (t-adt {} Floaty)))",
        );
    }

    // ── Unsupported tensor precision tests ──────────────────────

    #[test]
    fn tensor_ascription_accepts_f64() {
        // v0.2.3: f64 tensors are now a first-class precision. The checker
        // accepts tensor[..., f64]; the C backend emits `double` arrays.
        check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f64))} 0))");
    }

    #[test]
    fn tensor_ascription_rejects_f16() {
        check_err(
            "(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f16))} 0))",
            CheckErrorKind::UnsupportedTensorPrecision,
        );
    }

    #[test]
    fn tensor_ascription_rejects_bf16() {
        check_err(
            "(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} bf16))} 0))",
            CheckErrorKind::UnsupportedTensorPrecision,
        );
    }

    #[test]
    fn tensor_ascription_rejects_f8e4m3() {
        check_err(
            "(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f8e4m3))} 0))",
            CheckErrorKind::UnsupportedTensorPrecision,
        );
    }

    #[test]
    fn tensor_ascription_accepts_int64() {
        // Integer tensor precisions remain valid.
        check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} int64))} 0))");
    }

    #[test]
    fn tensor_ascription_accepts_bool() {
        check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} bool))} false))");
    }

    #[test]
    fn cast_tensor_accepts_f64() {
        // v0.2.3: tensor-f64 is a valid cast target. The checker accepts
        // cast(tensor_f32, f64); the backend emits float→double conversion.
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (cast {} (var {} x) (t-prim {} f64)))",
        );
    }

    #[test]
    fn cast_tensor_rejects_bf16() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (cast {} (var {} x) (t-prim {} bf16)))",
            CheckErrorKind::UnsupportedTensorPrecision,
        );
    }

    #[test]
    fn cast_prim_to_f64_is_allowed() {
        // Host scalar f64 is still valid — only tensor-precision f64 is banned.
        check_ok(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (cast {} (var {} x) (t-prim {} f64)))",
        );
    }

    #[test]
    fn surf_source_accepts_f64_tensor_ascription() {
        // v0.2.3: exercise the full surf → desugar → check pipeline to confirm
        // the user-facing syntax `(expr : tensor[4, f64])` is accepted.
        let decls = chelis_surf::parser::parse_str(
            "y = (to_tensor([1.0, 2.0, 3.0, 4.0]) : tensor[4, f64])",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let result = infer_program(&exprs);
        assert!(
            !result
                .errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::UnsupportedTensorPrecision)),
            "expected no UnsupportedTensorPrecision for f64 tensor ascription, got: {:?}",
            result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
        );
    }

    #[test]
    fn surf_source_accepts_cast_to_f64_tensor() {
        // v0.2.3: exercise the full pipeline for `cast(tensor, f64)`.
        let decls =
            chelis_surf::parser::parse_str("y = cast(to_tensor([1.5]), f64)").expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let result = infer_program(&exprs);
        assert!(
            !result
                .errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::UnsupportedTensorPrecision)),
            "expected no UnsupportedTensorPrecision for cast(tensor, f64), got: {:?}",
            result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
        );
    }

    #[test]
    fn copy_accepts_tensor() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (copy {} (var {} x)))",
        );
    }

    #[test]
    fn copy_rejects_scalar() {
        check_err(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (copy {} (var {} x)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn copy_rejects_unconstrained_generic() {
        check_err(
            "(def {} id (fn {} (params {} x) (copy {} (var {} x))))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // ── Grad tests ───────────────────────────────────────────────

    #[test]
    fn grad_function() {
        check_ok(
            "(defsig {} loss (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} loss (fn {} (params {} x) (var {} x)))
             (def {} g (grad {} (var {} loss)))",
        );
    }

    #[test]
    fn grad_non_float_param_is_ignored_by_default() {
        let exprs = chelis_deep::parser::parse_str(
            "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f)))",
        )
        .unwrap();
        let checked = check_phase0e_program(&exprs).expect("phase 0e check");
        let ty = checked.type_env().get("g").expect("g type");
        assert_eq!(
            chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
                .trim()
                .to_string(),
            "(t-fn {} (t-prim {} bool) (t-unit {}))"
        );
    }

    #[test]
    fn grad_rejects_non_scalar_output() {
        check_err(
            "(defsig {} f (t-fn {}
                (t-prim {} f32)
                (t-tensor {} (d-lit {} 2) (t-prim {} f32))))
             (def {} f
                (fn {} (params {} x)
                  (lit {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} 1.0)))
             (def {} g (grad {} (var {} f)))",
            CheckErrorKind::Other,
        );
    }

    #[test]
    fn grad_with_explicit_wrt_returns_selected_gradient_only() {
        let exprs = chelis_deep::parser::parse_str(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} hidden) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x w)
                    (lit {type: (t-prim {} f32)} 1.0)))
             (def {} dw
                (grad {} (var {} loss) (lit {type: (t-prim {} int32)} 1)))",
        )
        .unwrap();
        let checked = check_phase0e_program(&exprs).expect("phase 0e check");
        let ty = checked.type_env().get("dw").expect("dw type");
        assert_eq!(
            chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
                .trim()
                .to_string(),
            "(t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32)))"
        );
    }

    #[test]
    fn grad_with_multiple_wrt_returns_flat_tuple() {
        let exprs = chelis_deep::parser::parse_str(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} hidden) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x w)
                    (lit {type: (t-prim {} f32)} 1.0)))
             (def {} grads
                (grad {} (var {} loss)
                    (tuple {}
                        (lit {type: (t-prim {} int32)} 0)
                        (lit {type: (t-prim {} int32)} 1))))",
        )
        .unwrap();
        let checked = check_phase0e_program(&exprs).expect("phase 0e check");
        let ty = checked.type_env().get("grads").expect("grads type");
        assert_eq!(
            chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
                .trim()
                .to_string(),
            "(t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32)) (t-tuple {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32))))"
        );
    }

    #[test]
    fn grad_rejects_nondifferentiable_explicit_wrt_target() {
        check_err(
            "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f) (lit {type: (t-prim {} int32)} 0)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn vmap_function_inserts_axis_zero_batch_dim() {
        check_ok(
            "(defsig {} process
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} features) (t-prim {} f32))))
             (def {} process
                (fn {} (params {} x)
                    (var {} x)))
             (defsig {} batch_process
                (t-fn {}
                    (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
             (def {} batch_process
                (vmap {} (var {} process) (lit {type: (t-prim {} int32)} 0)))",
        );
    }

    #[test]
    fn vmap_non_function_is_rejected() {
        check_err(
            "(def {} x (lit {type: (t-prim {} f32)} 1.0))
             (def {} y (vmap {} (var {} x) (lit {type: (t-prim {} int32)} 0)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn vmap_axis_out_of_bounds_is_rejected() {
        check_err(
            "(defsig {} process
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} features) (t-prim {} f32))))
             (def {} process
                (fn {} (params {} x)
                    (var {} x)))
             (def {} batch_process
                (vmap {} (var {} process) (lit {type: (t-prim {} int32)} 2)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn vmap_grad_single_tensor_param_type_checks() {
        check_ok(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x)
                    (app {type: (t-prim {} f32)} (var {} sum) (var {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} x) (lit {type: (t-prim {} int32)} 0)))
             )
             (defsig {} per_example_grad
                (t-fn {}
                    (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
             (def {} per_example_grad
                (vmap {} (grad {} (var {} loss)) (lit {type: (t-prim {} int32)} 0)))",
        );
    }

    #[test]
    fn vmap_grad_multiple_params_type_checks_with_tuple_result() {
        check_ok(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x y)
                        (app {type: (t-prim {} f32)} (var {} sum)
                            (app {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} (var {} add)
                                (var {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} x)
                                (var {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} y))
                        (lit {type: (t-prim {} int32)} 0)))
             )
             (def {} per_example_grad
                (vmap {} (grad {} (var {} loss)) (lit {type: (t-prim {} int32)} 0)))",
        );
    }

    #[test]
    fn grad_of_vmap_is_rejected_for_non_scalar_output() {
        check_err(
            "(defsig {} loss
                (t-fn {}
                    (t-tensor {} (d-name {} features) (t-prim {} f32))
                    (t-prim {} f32)))
             (def {} loss
                (fn {} (params {} x)
                    (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g
                (grad {} (vmap {} (var {} loss) (lit {type: (t-prim {} int32)} 0))))",
            CheckErrorKind::Other,
        );
    }

    // ── ADT tests ────────────────────────────────────────────────

    #[test]
    fn adt_deftype_and_construct() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42)))",
        );
    }

    #[test]
    fn adt_nullary_constructor() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))
             (def {} x (var {} None))",
        );
    }

    #[test]
    fn adt_no_type_params() {
        check_ok(
            "(deftype {} Color () (variant {} Red) (variant {} Green) (variant {} Blue))
             (def {} c (var {} Red))",
        );
    }

    // ── Match tests ──────────────────────────────────────────────

    #[test]
    fn match_simple_adt() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42)))
             (def {} result
               (match {} (var {} x)
                 (arm {} (pat-ctor {} Some (pat-var {} v)) () (var {} v))
                 (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} int32)} 0))))",
        );
    }

    #[test]
    fn match_non_exhaustive() {
        check_err(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42)))
             (def {} result
               (match {} (var {} x)
                 (arm {} (pat-ctor {} Some (pat-var {} v)) () (var {} v))))",
            CheckErrorKind::NonExhaustiveMatch,
        );
    }

    // ── Defsig tests ─────────────────────────────────────────────

    #[test]
    fn defsig_fn_signature() {
        check_ok(
            "(defsig {} double (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} double (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn defsig_tensor_signature() {
        check_ok(
            "(defsig {} normalize_fn
               (t-fn {} (t-tensor {} (d-name {} batch) (t-prim {} f32))
                        (t-tensor {} (d-name {} batch) (t-prim {} f32))))
             (def {} normalize_fn (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn typealias_zero_param_resolves_in_defsig() {
        check_ok(
            "(typealias {} Scalar () (t-prim {} f32))
             (defsig {} id (t-fn {} (t-adt {} Scalar) (t-adt {} Scalar)))
             (def {} id (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn typealias_parameterized_resolves_in_defsig() {
        check_ok(
            "(typealias {} Boxed (a) (t-tuple {} (t-var {} a)))
             (defsig {} wrap (t-fn {} (t-adt {} Boxed (t-prim {} f32)) (t-adt {} Boxed (t-prim {} f32))))
             (def {} wrap (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn typealias_resolves_in_typed_param_metadata() {
        check_ok(
            "(typealias {} Scalar () (t-prim {} f32))
             (def {} id (fn {} (params {} (x {type: (t-adt {} Scalar)})) (var {} x)))",
        );
    }

    #[test]
    fn typealias_resolves_in_literal_metadata() {
        check_ok(
            "(typealias {} Scalar () (t-prim {} f32))
             (def {} x (lit {type: (t-adt {} Scalar)} 1.0))",
        );
    }

    // ── Partial inference tests ──────────────────────────────────

    #[test]
    fn partial_inference_continues_after_error() {
        // First def has an error, second should still be processed
        let result = check(
            "(def {} x (var {} nonexistent))
             (def {} y (lit {type: (t-prim {} int32)} 42))",
        );
        assert!(!result.errors.is_empty(), "expected at least one error");
        // y should still have been typed
        assert!(result.typed_nodes > 0, "expected some typed nodes");
    }

    #[test]
    fn multiple_errors_collected() {
        let result = check(
            "(def {} x (var {} unknown1))
             (def {} y (var {} unknown2))",
        );
        assert!(
            result.errors.len() >= 2,
            "expected at least 2 errors, got {}",
            result.errors.len()
        );
    }

    // ── Builtin operation tests ──────────────────────────────────

    #[test]
    fn builtin_neg() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} neg) (var {} x)))",
        );
    }

    #[test]
    fn builtin_exp() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} exp) (var {} x)))",
        );
    }

    #[test]
    fn builtin_matmul() {
        check_ok(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} hidden) (d-name {} classes) (t-prim {} f32))} 0))
             (def {} c (app {type: (t-tensor {} (d-name {} batch) (d-name {} classes) (t-prim {} f32))}
                 (var {} matmul) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn builtin_layer_norm() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
        );
    }

    #[test]
    fn builtin_layer_norm_rejects_rank2_gamma() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (d-name {} extra) (t-prim {} f32))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn builtin_layer_norm_rejects_precision_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} bf16))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    #[test]
    fn builtin_conv2d_accepts_int_stride_padding() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_c) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_c) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} conv2d) (var {} x) (var {} k) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 1)))",
        );
    }

    #[test]
    fn builtin_conv2d_rejects_channel_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_a) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_b) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} conv2d) (var {} x) (var {} k) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 1)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn builtin_conv2d_rejects_kernel_precision_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_c) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_c) (d-lit {} 3) (d-lit {} 3) (t-prim {} bf16))} 0))
             (def {} y (app {} (var {} conv2d) (var {} x) (var {} k) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 1)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    #[test]
    fn builtin_relu() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} relu) (var {} x)))",
        );
    }

    // ── Tensor type tests ────────────────────────────────────────

    #[test]
    fn tensor_2d() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))",
        );
    }

    #[test]
    fn tensor_literal_dim() {
        check_ok("(def {} x (lit {type: (t-tensor {} (d-lit {} 512) (t-prim {} f32))} 0))");
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn empty_program() {
        let result = check("");
        assert!(result.errors.is_empty());
        assert_eq!(result.typed_nodes, 0);
        assert_eq!(result.total_nodes, 0);
    }

    #[test]
    fn def_with_fn_body() {
        check_ok(
            "(def {} double (fn {} (params {} x) (app {} (var {} add) (var {} x) (var {} x))))",
        );
    }

    #[test]
    fn nested_let() {
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 1))
                 (let {} (bind {} y (var {} x))
                   (var {} y))))",
        );
    }

    #[test]
    fn if_with_tensor_branches() {
        check_ok(
            "(def {} cond (lit {type: (t-prim {} bool)} true))
             (def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (if {} (var {} cond) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn fn_applied_to_args() {
        check_ok(
            "(def {} f (fn {} (params {} x) (var {} x)))
             (def {} result (app {} (var {} f) (lit {type: (t-prim {} int32)} 42)))",
        );
    }

    #[test]
    fn adt_with_fields() {
        check_ok(
            "(deftype {} Pair ()
               (variant {} MkPair
                 (field {} fst (t-prim {} int32))
                 (field {} snd (t-prim {} f32))))
             (def {} p
               (record {}
                 (var {} MkPair)
                 (kv {} fst (lit {type: (t-prim {} int32)} 1))
                 (kv {} snd (lit {type: (t-prim {} f32)} 2.0))))",
        );
    }

    #[test]
    fn pipe_with_lambda() {
        check_ok(
            "(def {} result
               (pipe {} (lit {type: (t-prim {} f32)} 1.0)
                        (fn {} (params {} x) (var {} x))))",
        );
    }

    #[test]
    fn total_nodes_counted() {
        let result = check("(def {} x (lit {type: (t-prim {} int32)} 42))");
        assert!(result.total_nodes > 0, "expected some total nodes");
    }

    #[test]
    fn tuple_three_elems() {
        check_ok(
            "(def {} t (tuple {}
               (lit {type: (t-prim {} int32)} 1)
               (lit {type: (t-prim {} f32)} 2.0)
               (lit {type: (t-prim {} bool)} true)))",
        );
    }

    // ── Regression tests for bug fixes ──────────────────────────────

    // Fix 1: scalar arithmetic accepts matching numeric scalars but still rejects bad mixes
    #[test]
    fn fix1_tensor_op_rejects_non_tensor_args() {
        check_err(
            "(def {} r (app {} (var {} add) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} bool)} true)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // Fix 2: unsound generalization — fn x -> { y = x; (y 1, y true) } should fail
    #[test]
    fn fix2_unsound_generalization_rejected() {
        // x is a monomorphic param, y = x so y is also monomorphic.
        // Applying y to both int32 and bool should fail.
        check_err(
            "(def {} test \
               (fn {} (params {} x) \
                 (let {} (bind {} y (var {} x)) \
                   (tuple {} \
                     (app {} (var {} y) (lit {type: (t-prim {} int32)} 1)) \
                     (app {} (var {} y) (lit {type: (t-prim {} bool)} true))))))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // Fix 3: defsig not enforced — body must match declared signature
    #[test]
    fn fix3_defsig_enforced() {
        check_err(
            "(defsig {} f (t-fn {} (t-prim {} int32) (t-prim {} int32))) \
             (def {} f (lit {type: (t-prim {} bool)} true))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // Fix 4: d-var names shared within a type — same d-var name maps to same DimVar
    #[test]
    fn fix4_dvar_names_shared() {
        // Declare a function requiring same dim 'a' in both args.
        // Call with tensor[batch,f32] and tensor[seq,f32] — should fail.
        check_err(
            "(defsig {} myfn \
               (t-fn {} \
                 (t-tensor {} (d-var {} a) (t-prim {} f32)) \
                 (t-tensor {} (d-var {} a) (t-prim {} f32)) \
                 (t-tensor {} (d-var {} a) (t-prim {} f32)))) \
             (def {} myfn (fn {} (params {} x y) (var {} x))) \
             (def {} result (app {} (var {} myfn) \
               (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0) \
               (lit {type: (t-tensor {} (d-name {} seq) (t-prim {} f32))} 0)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    // Fix 5: if condition must be bool
    #[test]
    fn fix5_if_condition_must_be_bool() {
        check_err(
            "(if {} (lit {type: (t-prim {} int32)} 0) \
                    (lit {type: (t-prim {} int32)} 1) \
                    (lit {type: (t-prim {} int32)} 2))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // Fix 6a: wildcard satisfies exhaustiveness
    #[test]
    fn fix6a_wildcard_exhaustive() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None)) \
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42))) \
             (def {} result \
               (match {} (var {} x) \
                 (arm {} (pat-wild {}) () (lit {type: (t-prim {} int32)} 0))))",
        );
    }

    // Fix 6b: pat-as binds name
    #[test]
    fn fix6b_pat_as_binds_name() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None)) \
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42))) \
             (def {} result \
               (match {} (var {} x) \
                 (arm {} (pat-as {} whole (pat-wild {})) () (var {} whole))))",
        );
    }

    // Fix 7: fitness report includes unresolved names
    #[test]
    fn fix7_fitness_unresolved_names() {
        let exprs = chelis_deep::parser::parse_str("(def {} x (var {} unknown))").unwrap();
        let result = crate::infer::infer_program(&exprs);
        let report = crate::fitness::FitnessReport::from_infer_result(&result);
        assert!(
            report.unresolved_names.contains(&"unknown".to_string()),
            "expected 'unknown' in unresolved_names, got {:?}",
            report.unresolved_names
        );
        // Check severity is set
        assert!(report.errors.iter().all(|e| e.severity > 0.0));
    }

    // Fix 7b: suggestions populated for UnboundVariable
    #[test]
    fn fix7b_suggestions_for_unbound() {
        let result = check("(def {} x (var {} typo))");
        let unbound_err = result
            .errors
            .iter()
            .find(|e| matches!(e.kind, CheckErrorKind::UnboundVariable))
            .expect("expected UnboundVariable error");
        assert!(
            !unbound_err.suggestions.is_empty(),
            "expected suggestions for unbound variable"
        );
    }

    #[test]
    fn phase0e_literal_dimension_mismatch_surfaces_error() {
        let decls = chelis_surf::parser::parse_str(
            "def want_2x2(a: tensor[2, 2, f32]) -> f32 = trace(a, 0, 1)\n\
             def main(a: tensor[3, 3, f32]) -> f32 = want_2x2(a)\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);

        let result = infer_phase0e_program(&exprs);
        assert!(
            result
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
            "expected phase0e inference to preserve literal dimension mismatches, got {:?}",
            result.errors
        );
    }

    #[test]
    fn phase0e_rejects_polymorphic_dims_pinned_by_body() {
        let decls = chelis_surf::parser::parse_str(
            "def want_2x2(a: tensor[2, 2, f32]) -> f32 = trace(a, 0, 1)\n\
             def bad_consumer[m, n](a: tensor[m, n, f32]) -> f32 = want_2x2(a)\n\
             def main() -> f32 = cast(0.0, f32)\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);

        let result = infer_phase0e_program(&exprs);
        assert!(
            result
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
            "expected phase0e inference to reject polymorphic dims forced to literals by the body, got {:?}",
            result.errors
        );
    }

    #[test]
    fn phase0e_preserves_unresolved_name_errors() {
        let decls = chelis_surf::parser::parse_str(
            "def probe(x: f32) -> f32 = sub(x, frobnicate(x))\n\
             def main() -> f32 = probe(cast(1.0, f32))\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);

        let result = infer_phase0e_program(&exprs);
        assert!(
            result
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::UnboundVariable)),
            "expected phase0e inference to preserve unresolved-name errors, got {:?}",
            result.errors
        );

        let report = crate::fitness::check_phase0e_program(&exprs);
        assert!(
            report.unresolved_names.contains(&"frobnicate".to_string()),
            "expected phase0e fitness report to include unresolved frobnicate, got {:?}",
            report.unresolved_names
        );
    }

    #[test]
    fn phase0e_resolves_unique_terminal_constructor_names() {
        let decls = chelis_surf::parser::parse_str(
            "type KVCache[a] = | KVCache(List[a])\n\
             def keep_cache[p](cache: Option[KVCache[p]]) -> KVCache[p] =\n\
               match cache with {\n\
                 | Some(value) => value\n\
                 | None => Pkg__chelis__std__Std__Nn__Generate__KVCache([])\n\
               }\n",
        )
        .expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);

        let result = infer_phase0e_program(&exprs);
        assert!(
            !result
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::UnboundVariable)),
            "expected qualified constructor names to resolve by unique terminal match, got {:?}",
            result.errors
        );
    }

    // Fix 8: typed params in Deep
    #[test]
    fn fix8_typed_params() {
        // fn with typed param x: f32 — using x should give f32
        check_ok("(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))");
    }

    // Fix 8b: typed param enforces type
    #[test]
    fn fix8b_typed_param_enforced() {
        // Param x is f32, so mixing it with a bool in arithmetic must fail.
        check_err(
            "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) \
               (app {} (var {} add) (var {} x) (lit {type: (t-prim {} bool)} true))))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // ── Round 3 regression tests ──────────────────────────────────

    #[test]
    fn fix9_logical_ops_reject_non_bool_tensors() {
        // and(tensor[batch, f32], tensor[batch, f32]) should fail — requires bool
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
             (def {} r (app {} (var {} and) (var {} a) (var {} b)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix9b_logical_ops_reject_non_tensor() {
        // and(int32, int32) should fail — requires tensor
        check_err(
            "(def {} r (app {} (var {} and) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 2)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix9c_not_rejects_non_bool_tensor() {
        // not(tensor[batch, f32]) should fail — requires bool
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
             (def {} r (app {} (var {} not) (var {} a)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix9d_logical_ops_accept_bool_tensors() {
        // and(tensor[batch, bool], tensor[batch, bool]) should pass
        check_ok(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bool))} 0)) \
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bool))} 0)) \
             (def {} r (app {} (var {} and) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn fix10_fitness_has_untyped_nodes() {
        let result = check(
            "(def {} good (lit {type: (t-prim {} int32)} 42)) \
                            (def {} bad (var {} nope))",
        );
        let report = crate::fitness::FitnessReport::from_infer_result(&result);
        assert!(report.untyped_nodes > 0, "expected untyped_nodes > 0");
        // Errors should have severity set
        assert!(
            report.errors.iter().all(|e| e.severity > 0.0),
            "all errors should have severity > 0"
        );
    }

    // ── Round 4 regression tests ──────────────────────────────────

    #[test]
    fn fix11_pat_record_rejects_unknown_field() {
        // Define Adam with field lr, then match on nonexistent field 'nope'
        check_err(
            "(deftype {} Optimizer () \
               (variant {} Adam (field {} lr (t-prim {} f32)))) \
             (def {} x (lit {type: (t-adt {} Optimizer)} 0)) \
             (def {} r \
               (match {} (var {} x) \
                 (arm {} (pat-record {} Adam (kv {} nope (pat-var {} v))) () (var {} v))))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix11b_pat_record_accepts_valid_field() {
        // Match on actual field lr — should pass
        check_ok(
            "(deftype {} Optimizer () \
               (variant {} Adam (field {} lr (t-prim {} f32)))) \
             (def {} x (lit {type: (t-adt {} Optimizer)} 0)) \
             (def {} r \
               (match {} (var {} x) \
                 (arm {} (pat-record {} Adam (kv {} lr (pat-var {} v))) () (var {} v))))",
        );
    }

    #[test]
    fn fix12_structure_score_measured() {
        // check_program runs tag validator — structure should be 1.0 for valid programs
        let exprs = chelis_deep::parser::parse_str("(def {} x (lit {type: (t-prim {} int32)} 42))")
            .unwrap();
        let report = crate::fitness::check_program(&exprs);
        assert!(
            (report.components.structure - 1.0).abs() < 0.01,
            "valid program structure should be ~1.0, got {}",
            report.components.structure
        );
    }

    #[test]
    fn checked_program_annotates_fn_bodies() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
        )
        .unwrap();
        let checked = check_phase0e_program(&exprs).expect("checked program");
        let text = chelis_deep::printer::print_canonical(checked.annotated_exprs());
        assert!(
            text.contains("(fn {type: (t-fn {} (t-prim {} f32) (t-prim {} f32))}"),
            "expected typed fn metadata, got:\n{text}"
        );
    }

    #[test]
    fn checked_program_annotates_apps_and_updates_type_env() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
        )
        .unwrap();
        let checked = check_phase0e_program(&exprs).expect("checked program");
        let text = chelis_deep::printer::print_canonical(checked.annotated_exprs());
        assert!(
            text.contains("(app {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))}"),
            "expected typed app metadata, got:\n{text}"
        );
        assert!(checked.type_env().contains_key("c"));
    }

    #[test]
    fn phase0e_rejects_symbolic_normalized_axis_for_layer_norm() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
        )
        .unwrap();
        let err =
            check_phase0e_program(&exprs).expect_err("symbolic hidden axis should be rejected");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("concrete normalized axis extent")),
            "expected layer_norm symbolic normalized-axis error, got: {:?}",
            err.errors
        );
    }

    #[test]
    fn checked_program_annotates_symbolic_expand_apps_from_surf() {
        let checked = checked_surf(
            r#"
def predict(
  x: tensor[batch, 64, f32],
  w: tensor[64, 1, f32],
  b: tensor[1, f32]
) -> tensor[batch, 1, f32] =
  add(matmul(x, w), expand(b, 0, batch))
"#,
        );
        let missing = checked
            .annotated_exprs()
            .iter()
            .find_map(missing_shape_sensitive_app);
        assert!(
            missing.is_none(),
            "expected all shape-sensitive apps to be annotated, missing: {:?}",
            missing
        );
    }

    #[test]
    fn surf_permute_with_axis_arguments_type_checks() {
        let checked = checked_surf(
            r#"
def transpose(x: tensor[seq, hidden, f32]) -> tensor[hidden, seq, f32] =
  permute(x, 1, 0)
"#,
        );
        let missing = checked
            .annotated_exprs()
            .iter()
            .find_map(missing_shape_sensitive_app);
        assert!(
            missing.is_none(),
            "expected typed shape-sensitive apps after permute, missing: {:?}",
            missing
        );
    }

    #[test]
    fn surf_list_builtins_type_check() {
        let checked = checked_surf(
            r#"
xs: List[f32] = [1.0, 2.0]
ys = append(xs, 3.0)
total = tensor_to_scalar(sum(to_tensor(ys), 0))
roundtrip = to_list(to_tensor(ys))
"#,
        );
        assert!(checked.annotated_exprs().len() >= 4);
    }

    #[test]
    fn surf_pad_sequences_type_checks() {
        let checked = checked_surf(
            r#"
tokens: List[List[int64]] = [[cast(1, int64), cast(2, int64)], [cast(3, int64)]]
padded = pad_sequences(tokens, cast(0, int64))
"#,
        );
        assert!(checked.annotated_exprs().len() >= 2);
    }

    #[test]
    fn surf_3g_io_and_exact_padding_builtins_type_check() {
        let checked = checked_surf(
            r#"
contents = read_file("dataset.txt")
lines = read_lines("dataset.txt")
bytes = read_bytes("dataset.txt")
exists = file_exists("dataset.txt")
names = list_dir(".")
mapped = mmap_file("dataset.txt")
mapped_len = mmap_len(mapped)
prefix = mmap_read(mapped, cast(0, int64), cast(4, int64))
padded = pad_sequences_to([[cast(1, int64)], [cast(2, int64), cast(3, int64)]], cast(4, int64), cast(0, int64))
"#,
        );
        assert!(checked.annotated_exprs().len() >= 9);
    }

    #[test]
    fn surf_dict_and_iteration_builtins_type_check() {
        let checked = checked_surf(
            r#"
keys: List[string] = ["alpha", "beta"]
ids: List[int64] = [cast(1, int64), cast(2, int64)]
pairs = zip(keys, ids)
indexed = enumerate(keys)
vocab: Dict[string, int64] = dict_of(pairs)
found = dict_contains(vocab, "alpha")
id = dict_get(vocab, "beta")
only_keys = dict_keys(vocab)
only_values = dict_values(vocab)
roundtrip = dict_entries(vocab)
"#,
        );
        assert!(checked.annotated_exprs().len() >= 9);
    }

    #[test]
    fn surf_3h_tensor_numeric_builtins_type_check() {
        let checked = checked_surf(
            r#"
def projection(
  x: tensor[batch, seq, hidden, f32],
  w: tensor[hidden, out_dim, f32],
  token_ids: tensor[batch, seq, int64],
  mask: tensor[batch, seq, out_dim, bool],
  table: tensor[vocab, out_dim, f32]
) -> tensor[batch, seq, out_dim, f32] = {
  logits = einsum("bsh,ho->bso", x, w)
  embed = gather(table, token_ids, 0)
  clipped = clamp(add(logits, embed), scalar_to_tensor(0.0), scalar_to_tensor(6.0))
  running = cumsum(clipped, 1)
  where(mask, running, clipped)
}
"#,
        );
        assert!(!checked.annotated_exprs().is_empty());
    }

    #[test]
    fn surf_3h_structural_tensor_builtins_type_check() {
        let checked = checked_surf(
            r#"
def pack_heads(
  q: tensor[batch, seq, 2, f32],
  k: tensor[batch, seq, 2, f32]
) -> tensor[batch, seq, *, f32] = {
  packed = concat([q, k], 2)
  pieces = split(packed, 2, [2, 2])
  concat(pieces, 2)
}
"#,
        );
        assert!(!checked.annotated_exprs().is_empty());
    }

    #[test]
    fn surf_3h_sort_and_trace_type_check() {
        let checked = checked_surf(
            r#"
def summarize(x: tensor[batch, hidden, hidden, f32]) -> (tensor[batch, hidden, f32], tensor[batch, hidden, int64], tensor[batch, f32]) = {
  diag = diagonal(x, 1, 2)
  sorted = sort(diag, 1)
  values = sorted.0
  indices = sorted.1
  total = trace(x, 1, 2)
  (values, indices, total)
}
"#,
        );
        assert!(!checked.annotated_exprs().is_empty());
    }

    #[test]
    fn surf_einsum_rejects_ellipsis_in_3h() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(
  x: tensor[batch, seq, hidden, f32],
  w: tensor[hidden, out_dim, f32]
) -> tensor[batch, seq, out_dim, f32] =
  einsum("...h,ho->...o", x, w)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("ellipsis should be rejected in 3h einsum");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("einsum")
                    && (error.message.contains("ellipsis") || error.message.contains("..."))
            }),
            "expected einsum ellipsis rejection, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_where_rejects_non_bool_condition_tensor() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(
  cond: tensor[batch, hidden, f32],
  x: tensor[batch, hidden, f32],
  y: tensor[batch, hidden, f32]
) -> tensor[batch, hidden, f32] =
  where(cond, x, y)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("where should reject non-bool condition tensors");
        assert!(
            err.errors
                .iter()
                .any(|error| { error.message.contains("where") && error.message.contains("bool") }),
            "expected where bool mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_scatter_replace_rejects_unknown_mode() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(
  base: tensor[seq, hidden, f32],
  ids: tensor[seq, int64],
  updates: tensor[seq, hidden, f32]
) -> tensor[seq, hidden, f32] =
  scatter(base, ids, updates, 0, "last")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("scatter should reject unsupported mode");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("scatter")
                    && error.message.contains("replace")
                    && error.message.contains("add")
            }),
            "expected scatter mode rejection, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_einsum_rejects_static_extent_mismatch() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
a = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
b = pad_sequences([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], 0.0)
out = einsum("ij,jk->ik", a, b)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("static einsum extent mismatch should be rejected");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("einsum") && error.message.contains("inconsistent extents")
            }),
            "expected einsum extent mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_scatter_replace_rejects_static_duplicate_indices() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
base = pad_sequences([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0)
ids: List[int64] = [cast(1, int64), cast(1, int64)]
idx = to_tensor(ids)
updates = pad_sequences([[5.0, 5.0], [6.0, 6.0]], 0.0)
out = scatter(base, idx, updates, 0, "replace")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("static scatter duplicate indices should be rejected");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("scatter")
                    && error.message.contains("duplicate target index")
            }),
            "expected scatter duplicate-index rejection, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_map_filter_fold_type_check() {
        let checked = checked_surf(
            r#"
def inc(x: int64) -> int64 = add(x, cast(1, int64))
xs: List[int64] = [cast(1, int64), cast(2, int64), cast(3, int64)]
mapped = map(inc, xs)
filtered = filter(fn (x: int64) -> eq(mod(x, cast(2, int64)), cast(0, int64)), mapped)
total = fold(fn (acc: int64, x: int64) -> add(acc, x), cast(0, int64), filtered)
"#,
        );
        assert!(checked.annotated_exprs().len() >= 5);
    }

    #[test]
    fn surf_append_rejects_wrong_element_type() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = append(xs, "oops")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("append should reject mismatched element type");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("List") || error.message.contains("string")),
            "expected list element mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_filter_rejects_non_bool_callback() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = filter(fn (x: int64) -> add(x, cast(1, int64)), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("filter should reject non-bool callback");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("filter")
                    && error.message.contains("bool")
                    && error.message.contains("callback")),
            "expected bool callback mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_dict_entries_rejects_non_dict_input() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = dict_entries(xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("dict_entries should reject non-dict input");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("dict_entries")
                    || error.message.contains("Dict")),
            "expected dict input mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_to_list_rejects_rank2_tensor() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(x: tensor[2, 2, f32]) -> List[f32] = to_list(x)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("to_list should reject rank-2 tensor input");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("rank-1 tensor") || error.message.contains("to_list")
            }),
            "expected rank mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_fold_rejects_accumulator_mismatch() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = fold(fn (acc: string, x: int64) -> string_concat(acc, to_string(x)), cast(0, int64), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("fold should reject mismatched accumulator type");
        assert!(
            err.errors.iter().any(|error| error.message.contains("fold")
                && error.message.contains("accumulator")
                && error.message.contains("string")
                && error.message.contains("int64")),
            "expected accumulator mismatch, got {:?}",
            err.errors
        );
    }

    fn surf_tuple_fold_tensor_slot_program() -> Vec<deep::Expr> {
        chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def f[n](xs: tensor[n, f32]) -> (tensor[n, f32], int64) = {
  idxs = range(cast(0, int64), numel(copy(xs)))
  state0 = (to_tensor(map(fn (x: f32) -> cast(0.0, f32), to_list(copy(xs)))), cast(0, int64))
  step = fn (state, i) -> {
    acc = state.0
    total = state.1
    (acc, add(total, i))
  }
  fold(step, state0, idxs)
}

out = f(to_tensor([1.0, 2.0, 3.0]))
"#,
            )
            .expect("surf parse"),
        )
    }

    #[test]
    fn surf_polymorphic_tuple_fold_with_tensor_slot_infers_phase0e() {
        let result = infer_phase0e_program(&surf_tuple_fold_tensor_slot_program());
        assert!(
            result.errors.is_empty(),
            "phase0e inference should succeed without overflowing: {:?}",
            result.errors
        );
    }

    #[test]
    fn surf_polymorphic_tuple_fold_with_tensor_slot_annotates_phase0e() {
        let program = surf_tuple_fold_tensor_slot_program();
        let _ = annotate_phase0e_program(&program);
    }

    #[test]
    fn surf_polymorphic_tuple_fold_with_tensor_slot_type_checks() {
        let result = check_phase0e_program(&surf_tuple_fold_tensor_slot_program());
        result.expect("polymorphic tuple fold should type check without overflowing");
    }

    #[test]
    fn surf_collection_helper_builtins_type_check() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64), cast(2, int64), cast(3, int64)]
prefix = take(xs, cast(2, int64))
suffix = drop(xs, cast(1, int64))
groups = chunk(xs, cast(2, int64))
scanned = scan(fn (acc: int64, x: int64) -> add(acc, x), cast(0, int64), xs)
buckets = partition(fn (x: int64) -> gt(x, cast(1, int64)), xs)
exploded = flat_map(fn (x: int64) -> [x, add(x, cast(10, int64))], xs)
flattened = flatten([[cast(1, int64)], [cast(2, int64), cast(3, int64)]])
base: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
extended = dict_insert(base, "beta", cast(2, int64))
merged = dict_merge(extended, dict_of([("beta", cast(20, int64)), ("gamma", cast(3, int64))]))
trimmed = dict_remove(merged, "gamma")
"#,
            )
            .expect("surf parse"),
        ));
        result.expect("collection helper builtins should type check");
    }

    #[test]
    fn surf_take_rejects_non_integer_count() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64), cast(2, int64)]
bad = take(xs, "two")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("take should reject non-integer count");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("take") || error.message.contains("integer")),
            "expected integer count mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_dict_insert_rejects_value_type_mismatch() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
base: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
bad = dict_insert(base, "beta", "two")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("dict_insert should reject mismatched value type");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("dict_insert")
                    || error.message.contains("int64")
                    || error.message.contains("string")),
            "expected dict value mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_dict_merge_rejects_mismatched_dict_value_types() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
lhs: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
rhs: Dict[string, string] = dict_of([("beta", "two")])
bad = dict_merge(lhs, rhs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("dict_merge should reject mismatched dict value types");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("dict_merge")
                    || error.message.contains("Dict")
                    || error.message.contains("int64")
                    || error.message.contains("string")),
            "expected dict merge mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_scan_rejects_accumulator_mismatch() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = scan(fn (acc: string, x: int64) -> string_concat(acc, to_string(x)), cast(0, int64), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("scan should reject mismatched accumulator type");
        assert!(
            err.errors.iter().any(|error| error.message.contains("scan")
                && error.message.contains("accumulator")
                && error.message.contains("string")
                && error.message.contains("int64")),
            "expected scan accumulator mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_partition_rejects_non_bool_callback() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = partition(fn (x: int64) -> add(x, cast(1, int64)), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("partition should reject non-bool callback");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("partition")
                    && error.message.contains("bool")
                    && error.message.contains("callback")),
            "expected partition callback mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_flat_map_rejects_non_list_callback() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = flat_map(fn (x: int64) -> add(x, cast(1, int64)), xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("flat_map should reject non-list callback");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("List") || error.message.contains("flat_map")),
            "expected flat_map callback mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_flatten_rejects_non_nested_list_input() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = flatten(xs)
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("flatten should reject non-nested list input");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("flatten") || error.message.contains("List")),
            "expected flatten input mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_dict_remove_rejects_mismatched_key_type() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
base: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
bad = dict_remove(base, cast(7, int64))
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("dict_remove should reject mismatched key type");
        assert!(
            err.errors.iter().any(|error| {
                error.message.contains("dict_remove")
                    || (error.message.contains("string") && error.message.contains("int64"))
            }),
            "expected dict_remove key mismatch, got {:?}",
            err.errors
        );
    }

    #[test]
    fn surf_chunk_rejects_non_integer_size() {
        let result = check_phase0e_program(&chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[int64] = [cast(1, int64)]
bad = chunk(xs, "two")
"#,
            )
            .expect("surf parse"),
        ));
        let err = result.expect_err("chunk should reject non-integer size");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("chunk") || error.message.contains("integer")),
            "expected chunk size mismatch, got {:?}",
            err.errors
        );
    }

    // ------------------------------------------------------------------
    // Top-level binding cycle detection
    // ------------------------------------------------------------------

    fn phase0e_errors_from_surf(src: &str) -> Vec<CheckError> {
        let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        infer_phase0e_program(&exprs).errors
    }

    #[test]
    fn nautilus_self_reference_is_allowed() {
        // The single-hop identity `x = (x : tensor[...])` is a pinned Nautilus
        // external-input pattern and must NOT be flagged as a binding cycle.
        let errors = phase0e_errors_from_surf("x = (x : tensor[4, f32])\n");
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::CycleDetected)),
            "Nautilus self-reference should not be flagged; got {errors:?}"
        );
    }

    #[test]
    fn two_hop_binding_cycle_is_detected() {
        let errors = phase0e_errors_from_surf(
            "a = (b : tensor[4, f32])\n\
             b = (a : tensor[4, f32])\n",
        );
        let cycle_err = errors
            .iter()
            .find(|e| matches!(e.kind, CheckErrorKind::CycleDetected))
            .unwrap_or_else(|| {
                panic!("expected a CycleDetected error for a two-hop cycle; got {errors:?}")
            });
        assert!(
            cycle_err.message.contains("binding cycle"),
            "message should mention 'binding cycle'; got {:?}",
            cycle_err.message
        );
        assert!(
            cycle_err.message.contains("a -> b -> a"),
            "expected 'a -> b -> a' in message; got {:?}",
            cycle_err.message
        );
    }

    #[test]
    fn three_hop_binding_cycle_is_detected() {
        let errors = phase0e_errors_from_surf(
            "a = (b : tensor[4, f32])\n\
             b = (c : tensor[4, f32])\n\
             c = (a : tensor[4, f32])\n",
        );
        let cycle_err = errors
            .iter()
            .find(|e| matches!(e.kind, CheckErrorKind::CycleDetected))
            .unwrap_or_else(|| {
                panic!("expected a CycleDetected error for a three-hop cycle; got {errors:?}")
            });
        assert!(
            cycle_err.message.contains("a -> b -> c -> a"),
            "expected 'a -> b -> c -> a' in message; got {:?}",
            cycle_err.message
        );
    }

    #[test]
    fn unrelated_defs_do_not_trigger_cycle_false_positive() {
        // Sanity: multiple Nautilus self-references together should still pass.
        let errors = phase0e_errors_from_surf(
            "x = (x : tensor[4, f32])\n\
             y = (y : tensor[4, f32])\n",
        );
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e.kind, CheckErrorKind::CycleDetected)),
            "multiple independent Nautilus inputs must not trigger a cycle error; got {errors:?}"
        );
    }
}
