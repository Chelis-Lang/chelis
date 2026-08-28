//! IR validation and type conversion.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

pub(super) fn validate_ir_program(
    exprs: &[deep::Expr],
    type_env: &IrTypeEnv,
    errors: &mut DiagnosticSink<'_>,
) {
    detect_top_level_binding_cycles(exprs, errors);
    detect_trivial_non_terminating_fns(exprs, errors);
    let mut static_env = HashMap::new();
    // Names of let-bindings whose RHS validation already emitted a
    // diagnostic (so their derived output type is unknown). Downstream
    // shape-sensitive calls that consume such a name emit a redundant
    // cascade diagnostic; suppress it. See RT-205 round-2 F3.
    let mut failed_let_names: HashSet<String> = HashSet::new();
    // chelis#930: per-top-level-declaration cancellation, same grain as
    // inference. Without it this validator is one uninterruptible step whose
    // cost grows with the program, and interrupt latency is bounded by the
    // longest such step. The caller's `cancellation_gate` rejects the
    // truncated walk.
    let cancel = crate::cancel::current_cancel_token();
    for expr in top_level_decl_items(exprs) {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            break;
        }
        validate_ir_expr(
            expr,
            type_env,
            &mut static_env,
            &mut failed_let_names,
            errors,
        );
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
pub(super) fn detect_trivial_non_terminating_fns(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) {
    // Collect each def's "terminal callees" — the top-level fn names
    // reached at every tail position of the body. `Some(set)` means
    // every tail is a call; the set is who's called. `None` means the
    // body has at least one non-call tail (a base case exists).
    let mut terminal_callees: HashMap<String, Option<HashSet<String>>> = HashMap::new();
    let mut def_order: Vec<String> = Vec::new();
    for expr in top_level_decl_items(exprs) {
        // chelis#1107 amendment: carrier-preserving read.
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else { continue };
        // Check params for a name that shadows the def — a body that
        // terminal-calls a shadowed name is NOT self-recursion.
        let mut shadows: HashSet<String> = HashSet::new();
        if let deep::Expr::List(fn_list, _) = body
            && get_tag(fn_list) == Some(DeepTag::Fn)
            && let Some(deep::Expr::List(params, _)) = children(fn_list).first()
            && get_tag(params) == Some(DeepTag::Params)
        {
            for param in children(params) {
                if let Some(pname) = param_name_for_refs(param) {
                    shadows.insert(pname);
                }
            }
        }
        let fn_body = match body {
            deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::Fn) => {
                children(list).get(1)
            }
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
                    "def `{name}` is trivially non-terminating. Every tail position \
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
pub(super) fn collect_terminal_callees(
    expr: &deep::Expr,
    shadowed: &HashSet<String>,
    out: &mut HashSet<String>,
) -> bool {
    stack_guard!("collect_terminal_callees", expr, false);
    match expr {
        deep::Expr::MetaExpr(meta, _) => collect_terminal_callees(&meta.expr, shadowed, out),
        deep::Expr::List(list, _) => match get_tag(list) {
            Some(DeepTag::App) => {
                let kids = children(list);
                let Some(callee) = kids.first() else {
                    return false;
                };
                // chelis#1107 amendment: carrier-preserving read.
                let Some((DeepTag::Var, _, callee_kids)) = stamped_parts(callee) else {
                    return false;
                };
                let Some(cname) = callee_kids.first().and_then(symbol_name) else {
                    return false;
                };
                if shadowed.contains(cname) {
                    return false;
                }
                out.insert(cname.to_string());
                true
            }
            Some(DeepTag::Let) => {
                let kids = children(list);
                kids.get(1)
                    .map(|body| collect_terminal_callees(body, shadowed, out))
                    .unwrap_or(false)
            }
            Some(DeepTag::If) => {
                let kids = children(list);
                if kids.len() < 3 {
                    return false;
                }
                let then_ok = collect_terminal_callees(&kids[1], shadowed, out);
                let else_ok = collect_terminal_callees(&kids[2], shadowed, out);
                then_ok && else_ok
            }
            Some(DeepTag::Match) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                kids.iter().skip(1).all(|arm| {
                    if let deep::Expr::List(arm_list, _) = arm
                        && get_tag(arm_list) == Some(DeepTag::Arm)
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
pub(super) fn fn_body_is_direct_self_call(def_body: &deep::Expr, def_name: &str) -> bool {
    let fn_list = match def_body {
        deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::Fn) => list,
        _ => return false,
    };
    // If any fn param shadows the def name, the callee reference inside
    // the body refers to the param (a callable HOF argument), not the def
    // itself. This is a legitimate HOF call, not recursion.
    if let Some(params_list) = children(fn_list).first()
        && let deep::Expr::List(params, _) = params_list
        && get_tag(params) == Some(DeepTag::Params)
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
pub(super) fn every_terminal_is_self_call(expr: &deep::Expr, def_name: &str) -> bool {
    stack_guard!("every_terminal_is_self_call", expr, false);
    match expr {
        deep::Expr::MetaExpr(meta, _) => every_terminal_is_self_call(&meta.expr, def_name),
        deep::Expr::List(list, _) => match get_tag(list) {
            Some(DeepTag::App) => {
                let kids = children(list);
                let Some(callee) = kids.first() else {
                    return false;
                };
                // chelis#1107 amendment: carrier-preserving read.
                let Some((DeepTag::Var, _, callee_kids)) = stamped_parts(callee) else {
                    return false;
                };
                callee_kids.first().and_then(symbol_name) == Some(def_name)
            }
            Some(DeepTag::Let) => {
                let kids = children(list);
                kids.get(1)
                    .map(|body| every_terminal_is_self_call(body, def_name))
                    .unwrap_or(false)
            }
            Some(DeepTag::If) => {
                let kids = children(list);
                if kids.len() < 3 {
                    return false;
                }
                every_terminal_is_self_call(&kids[1], def_name)
                    && every_terminal_is_self_call(&kids[2], def_name)
            }
            Some(DeepTag::Match) => {
                let kids = children(list);
                if kids.len() < 2 {
                    return false;
                }
                kids.iter().skip(1).all(|arm| {
                    if let deep::Expr::List(arm_list, _) = arm
                        && get_tag(arm_list) == Some(DeepTag::Arm)
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
pub(super) fn body_is_literal_self_ref(body: &deep::Expr, name: &str) -> bool {
    let mut current = body;
    loop {
        match current {
            deep::Expr::MetaExpr(meta, _) => current = &meta.expr,
            deep::Expr::List(list, _) => {
                // Type ascription desugars into a `(cast ... )`-like node
                // in Deep: `(x : T)` keeps `x` as the first child. When
                // the underlying is a var with the self name, treat it as
                // the Nautilus pattern. These legacy spellings are outside
                // the closed vocabulary, so they stay symbol-headed and are
                // recognized at the raw-string boundary.
                if matches!(list.unknown_tag_symbol(), Some("ascribe" | ":")) {
                    match children(list).first() {
                        Some(inner) => current = inner,
                        None => return false,
                    }
                    continue;
                }
                match get_tag(list) {
                    Some(DeepTag::Var) => {
                        return children(list).first().and_then(symbol_name) == Some(name);
                    }
                    _ => return false,
                }
            }
            _ => return false,
        }
    }
}

/// Yield each top-level declaration, flattening through a `(module {} name ...)`
/// wrapper if present. Deep sources produced by Surf `module X` desugaring
/// have every def/defsig/deftype inside this wrapper; without flattening,
/// top-level walkers see a single `(module ...)` and miss everything inside.
/// Extract the name of a top-level decl (`def`, `defsig`, `deftype`,
/// `typealias`) for profile instrumentation. Returns `None` for shapes
/// that don't have a leading symbol.
/// Walk the program's Deep AST and reject any `(t-tensor ... (t-prim {} P))`
/// whose precision P is not supported by the Phase 0f tensor backend
/// (currently: f16, bf16, f64, f8e4m3, string).
///
/// This runs after HM inference so it catches user-written tensor type
/// ascriptions, defsig tensor types, parameter type annotations, literal
/// type metadata, and any cast target that produces a tensor with an
/// unsupported element precision.
pub(super) fn validate_tensor_precisions_in_program(
    exprs: &[deep::Expr],
    errors: &mut impl DiagnosticOutput,
) {
    let mut seen: HashSet<(String, String)> = HashSet::new();
    // Descend through `(module {} name ...)` wrappers so per-def dedup
    // keeps each def's tensor types in their own key space (otherwise
    // every def lives under def_context="" and errors collapse).
    for expr in top_level_decl_items(exprs) {
        let def_name = match expr {
            deep::Expr::List(list, _)
                if matches!(
                    get_tag(list),
                    Some(DeepTag::Def)
                        | Some(DeepTag::Defsig)
                        | Some(DeepTag::Deftype)
                        | Some(DeepTag::Typealias)
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

pub(super) fn walk_for_tensor_precision(
    expr: &deep::Expr,
    errors: &mut impl DiagnosticOutput,
    seen: &mut HashSet<(String, String)>,
    def_context: &str,
) {
    // Bail before this walker's own unbounded recursion exhausts the
    // native stack (gdb confirmed this is a real SIGSEGV site on deep `app`
    // trees, distinct from `infer_expr`). The bail records into
    // `STACK_EXHAUSTED`; the check entry boundary turns that into a single
    // located failure, so we just stop recursing here. See
    // `STACK_RED_ZONE_BYTES`.
    stack_guard!(
        "validate_tensor_precisions (walk_for_tensor_precision)",
        expr
    );
    match expr {
        deep::Expr::List(list, _span) => {
            // Check t-tensor nodes at this level.
            if get_tag(list) == Some(DeepTag::TTensor) {
                let kids = children(list);
                if let Some(last) = kids.last()
                    && let deep::Expr::List(prec_list, _) = last
                    && get_tag(prec_list) == Some(DeepTag::TPrim)
                    && let Some(name) = children(prec_list).first().and_then(symbol_name)
                {
                    let active_set = "f32, f64, bf16, f16, bool, int8, int16, int32, int64";
                    // A1 (WS-A0 RT-1 fixup): unsigned dtype names, plus
                    // the other reserved-but-deferred names of
                    // spec/04-type-system.md §1.1.1. Mirror the f8e4m3
                    // §1.1.1 rejection contract — these names never
                    // resolve through `Prim::parse_name`, so without
                    // this guard `tensor[..., u8]` (or `tensor[...,
                    // complex64]`) would silently fall through with no
                    // §1.1.1-citing diagnostic.
                    if is_unsigned_dtype_name(name)
                        && seen.insert((def_context.to_string(), name.to_string()))
                    {
                        if let Some(diag) =
                            unsigned_family_diagnostic(name, /* tensor = */ true)
                        {
                            errors.push(diag);
                        }
                    } else if is_deferred_dtype_name(name)
                        && seen.insert((def_context.to_string(), name.to_string()))
                    {
                        if let Some(diag) =
                            deferred_family_diagnostic(name, /* tensor = */ true)
                        {
                            errors.push(diag);
                        }
                    } else if let Some(prim) = Prim::parse_name(name)
                        && !prim.is_valid_tensor_precision()
                        && seen.insert((def_context.to_string(), name.to_string()))
                    {
                        if matches!(prim, Prim::F8e4m3) {
                            errors.push(CheckError::new(
                                CheckErrorKind::UnsupportedTensorPrecision,
                                format!(
                                    "tensor element precision `f8e4m3` is deferred per \
                                     spec/04-type-system.md §1.1.1 and is not part of the active \
                                     numeric primitive set ({active_set})",
                                ),
                                vec![format!(
                                    "f8e4m3 has no active backend in this cycle; pick one of \
                                     {active_set} or see spec/04-type-system.md §1.1.1 for the \
                                     deferral rationale",
                                )],
                            ));
                        } else {
                            errors.push(CheckError::new(
                                CheckErrorKind::UnsupportedTensorPrecision,
                                format!(
                                    "tensor element precision `{name}` is not supported by the \
                                     current backend set (supported: {active_set})",
                                ),
                                vec![format!(
                                    "Use tensor[..., f32] and cast host scalars explicitly, or \
                                     keep `{name}` as a host scalar",
                                )],
                            ));
                        }
                    } else if Prim::parse_name(name).is_none()
                        && !is_unsigned_dtype_name(name)
                        && !is_deferred_dtype_name(name)
                        && seen.insert((def_context.to_string(), name.to_string()))
                    {
                        // WS-A5 RT-3a F3: an identifier in a `t-prim`
                        // precision slot that is neither a known active
                        // primitive nor a §1.1.1 deferred dtype name
                        // (unsigned alias or reserved name) is an
                        // unbound name. Inside a sig the desugarer emits
                        // such an identifier as `t-var`, so reaching this
                        // arm with `t-prim` proves the name appears in a
                        // value-position annotation (let binding, def
                        // param without a surrounding sig that quantified
                        // it) where the closed primitive set must apply.
                        // Without this guard the name silently collapses
                        // to a witnessed resolution failure at the centralized
                        // Deep type boundary's
                        // `Prim::parse_name` fall-through and the
                        // permissive unify rule absorbs the mismatch.
                        errors.push(CheckError::new(
                            CheckErrorKind::UnsupportedTensorPrecision,
                            format!(
                                "tensor element precision `{name}` is not a recognized \
                                 primitive (active set: {active_set}); inside a sig an \
                                 unbound lowercase name introduces a precision tvar per \
                                 spec/04-type-system.md §5.8, but in this position the \
                                 closed primitive set applies",
                            ),
                            vec![format!(
                                "Use one of {active_set}, or move the annotation into a \
                                 `sig` declaration that quantifies `{name}` as a precision \
                                 type variable",
                            )],
                        ));
                    }
                }
            }

            // `cast` is inferred by infer_cast which already emits a clearer
            // site-local error for bad precisions. Skip the walker's recursion
            // inside a cast so we don't duplicate the diagnostic.
            if get_tag(list) == Some(DeepTag::Cast) {
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
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        deep::Expr::Node(node, span) => {
            let bridged = deep::Expr::List(node.to_list(*span), *span);
            walk_for_tensor_precision(&bridged, errors, seen, def_context);
        }
        deep::Expr::BareList(elems, _) => {
            for child in elems {
                walk_for_tensor_precision(child, errors, seen, def_context);
            }
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                walk_for_tensor_precision(child, errors, seen, def_context);
            }
        }
    }
}

/// WS-A8 cross-row enforcement: spec/04-type-system.md §5.4 (transcendentals
/// on float-only) and §5.7.2 (matmul not admitted on integer operands)
/// fire correctly at direct primitive call sites
/// (see `check_matmul_signature` and the `TENSOR_OPS` post-check in
/// `infer_app`), but were silent when the same restricted op was reached
/// through a polymorphic-precision sig instantiation.
///
/// Example: a stdlib `linear.forward` body uses `matmul(x, w)`. Its sig
/// is `&tensor[a, b, p] -> &tensor[b, c, p] -> tensor[a, c, p]`. Inside
/// the body, `matmul`'s operand precisions are both `Var(p)`; the §5.7.2
/// check (`lhs_prec.is_integer()`) returns `false` for a `Var`. At a
/// concrete call site `linear.forward(x: int32, w: int32)`, the call
/// site instantiates `p` to `int32` via unification — but the body's
/// already-checked `matmul(x, w)` doesn't get re-checked. The integer
/// rejection silently slipped through.
///
/// This pass closes the gap. It walks every `(app (var name) ...)` call
/// site post-inference and, when the callee is a top-level user-def
/// with a polymorphic-precision sig, builds a precision-tvar
/// substitution from the call site's arg types vs the callee's sig
/// parameter types, then re-checks the callee's body for restricted
/// ops with the substituted operand precisions.
pub(super) fn validate_polymorphic_op_constraints(
    exprs: &[deep::Expr],
    type_env: &IrTypeEnv,
    errors: &mut DiagnosticSink<'_>,
) {
    let defs_with_bodies = collect_def_bodies(exprs);
    let defsigs = collect_defsig_exprs(exprs);
    // Merge defsig sigs into the type-env view so polymorphic
    // signatures (which haven't been annotated onto def bodies yet) are
    // visible to the call-site lookup.
    let mut combined_env: IrTypeEnv = type_env.clone();
    for (name, sig_expr) in &defsigs {
        combined_env.entry(name.clone()).or_insert(sig_expr.clone());
    }
    // For each top-level def we walk into, build a local scope mapping
    // body-param names to their declared types (extracted from the
    // def's sig in `combined_env`). This lets us resolve `(var x)`
    // references inside the body without requiring the bodies to have
    // been annotated. Bodies of polymorphic defs intentionally have
    // their inner exprs untouched by the annotator at this stage.
    // chelis#930: per-top-level-declaration cancellation; see
    // `validate_ir_program`. Measured at ~0.85s over 1500 declarations, this
    // was the largest remaining uninterruptible step in the check phase.
    let cancel = crate::cancel::current_cancel_token();
    for expr in top_level_decl_items(exprs) {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            break;
        }
        let scope = build_def_param_scope(expr, &combined_env);
        walk_for_poly_op_constraint_violations(
            expr,
            &defs_with_bodies,
            &combined_env,
            &scope,
            errors,
        );
    }
}

/// Build a name → declared-type-expr map for a def's body params, by
/// pairing each param's name with the corresponding sig parameter slot.
pub(super) fn build_def_param_scope(
    expr: &deep::Expr,
    sigs: &IrTypeEnv,
) -> HashMap<String, deep::Expr> {
    let mut scope = HashMap::new();
    // chelis#1107: carrier-preserving read. A `List`-only destructure returned
    // an empty scope for every stamped `def`, which -- together with the two
    // collectors below and the callee read in
    // `check_app_for_poly_op_constraint` -- left the whole WS-A8 cross-row
    // pass inert on `check_typed_program`.
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return scope;
    };
    if tag != DeepTag::Def {
        return scope;
    }
    let Some(name) = kids.first().and_then(symbol_name) else {
        return scope;
    };
    let Some(body_expr) = kids.get(1) else {
        return scope;
    };
    let Some((param_names, _)) = extract_fn_params_and_body(body_expr) else {
        return scope;
    };
    // Prefer the def's surrounding sig if any; fall back to inline
    // param-type annotations on the params themselves (def shape:
    // `def f(x: tensor[3, f32]) = ...`).
    if let Some(sig) = sigs.get(name)
        && let Some((sig_params, _)) = parse_t_fn_parts(sig)
    {
        for (pname, sig_param) in param_names.iter().zip(sig_params.iter()) {
            scope.insert(pname.clone(), sig_param.clone());
        }
        return scope;
    }
    // Fall back: inline param-type annotations.
    let Some((_, _, fn_kids)) = stamped_parts(body_expr) else {
        return scope;
    };
    let Some(params_expr) = fn_kids.first() else {
        return scope;
    };
    let Some((params_tag, _, params)) = stamped_parts(params_expr) else {
        return scope;
    };
    if params_tag != DeepTag::Params {
        return scope;
    }
    for param in params {
        if let Some((pname, Some(ty_expr))) = param_name_and_inline_type(param) {
            scope.insert(pname, ty_expr);
        }
    }
    scope
}

/// Extract `(name {type: T} ...)` shape's name+type from a single param
/// expression. Returns `None` for plain `(name {})` shape (no inline
/// type) — the surrounding sig fills those in.
pub(super) fn param_name_and_inline_type(
    param: &deep::Expr,
) -> Option<(String, Option<deep::Expr>)> {
    match param {
        deep::Expr::Atom(deep::Atom::Name(name), _) => Some((name.clone(), None)),
        deep::Expr::List(list, _) => {
            let Some(deep::Expr::Atom(deep::Atom::Name(name), _)) = list.elements.first() else {
                return None;
            };
            let ty = match list.elements.get(1) {
                Some(deep::Expr::Map(meta, _)) => meta
                    .entries
                    .iter()
                    .find(|(k, _)| k == "type")
                    .map(|(_, v)| v.clone()),
                _ => None,
            };
            Some((name.clone(), ty))
        }
        deep::Expr::MetaExpr(meta, _) => {
            let deep::Expr::Atom(deep::Atom::Name(name), _) = meta.expr.as_ref() else {
                return None;
            };
            let ty = meta
                .entries
                .iter()
                .find(|(k, _)| k == "type")
                .map(|(_, v)| v.clone());
            Some((name.clone(), ty))
        }
        _ => None,
    }
}

/// Collect every `(defsig {} name sig_expr)` at the top level into a
/// name → sig-expr map. WS-A8 needs this so polymorphic-sig info reaches
/// the cross-row enforcement pass even when the def's body annotation
/// has not yet been populated by the annotator.
pub(super) fn collect_defsig_exprs(exprs: &[deep::Expr]) -> HashMap<String, deep::Expr> {
    let mut out = HashMap::new();
    for expr in top_level_decl_items(exprs) {
        // chelis#1107: carrier-preserving read; see `build_def_param_scope`.
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if tag != DeepTag::Defsig {
            continue;
        }
        if let Some(name) = kids.first().and_then(symbol_name)
            && let Some(sig) = kids.get(1)
        {
            out.insert(name.to_string(), sig.clone());
        }
    }
    out
}

/// Map from def name to (params: Vec<param-name>, body-expr).
pub(super) type DefBodyMap = HashMap<String, (Vec<String>, deep::Expr)>;

pub(super) fn collect_def_bodies(exprs: &[deep::Expr]) -> DefBodyMap {
    let mut out = HashMap::new();
    for expr in top_level_decl_items(exprs) {
        // chelis#1107: carrier-preserving read; see `build_def_param_scope`.
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if tag != DeepTag::Def {
            continue;
        }
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        let Some((params, body_expr)) = extract_fn_params_and_body(body) else {
            continue;
        };
        out.insert(name.to_string(), (params, body_expr));
    }
    out
}

pub(super) fn extract_fn_params_and_body(expr: &deep::Expr) -> Option<(Vec<String>, deep::Expr)> {
    let (DeepTag::Fn, _, kids) = stamped_parts(expr)? else {
        return None;
    };
    let params_expr = kids.first()?;
    let body_expr = kids.get(1)?;
    let params = match params_expr {
        deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => node.children_slice(),
        deep::Expr::List(params_list, _) if get_tag(params_list) == Some(DeepTag::Params) => {
            children(params_list)
        }
        deep::Expr::BareList(elements, _) => elements.as_slice(),
        _ => return None,
    };
    let mut names = Vec::new();
    for param in params {
        if let Some(name) = param_name_for_refs(param) {
            names.push(name);
        }
    }
    Some((names, body_expr.clone()))
}

pub(super) fn walk_for_poly_op_constraint_violations(
    expr: &deep::Expr,
    defs: &DefBodyMap,
    type_env: &IrTypeEnv,
    scope: &HashMap<String, deep::Expr>,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("walk_for_poly_op_constraint_violations", expr);
    match expr {
        deep::Expr::List(list, _span) => {
            // Check if this is `(app (var name) arg1 arg2 ...)` calling
            // a top-level user-def with a polymorphic precision sig.
            if get_tag(list) == Some(DeepTag::App) {
                check_app_for_poly_op_constraint(list, defs, type_env, scope, errors);
            }
            for child in &list.elements {
                walk_for_poly_op_constraint_violations(child, defs, type_env, scope, errors);
            }
        }
        deep::Expr::Map(map, _) => {
            for (_, v) in &map.entries {
                walk_for_poly_op_constraint_violations(v, defs, type_env, scope, errors);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            walk_for_poly_op_constraint_violations(&meta.expr, defs, type_env, scope, errors);
            for (_, v) in &meta.entries {
                walk_for_poly_op_constraint_violations(v, defs, type_env, scope, errors);
            }
        }
        deep::Expr::Atom(_, _) => {}
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        deep::Expr::Node(node, span) => {
            let bridged = deep::Expr::List(node.to_list(*span), *span);
            walk_for_poly_op_constraint_violations(&bridged, defs, type_env, scope, errors);
        }
        deep::Expr::BareList(elems, _) => {
            for child in elems {
                walk_for_poly_op_constraint_violations(child, defs, type_env, scope, errors);
            }
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                walk_for_poly_op_constraint_violations(child, defs, type_env, scope, errors);
            }
        }
    }
}

/// At an `(app (var callee_name) arg1 arg2 ...)` call site, if `callee_name`
/// is a top-level user-def with a polymorphic-precision sig, compute the
/// call-site precision substitution and re-check the body for restricted
/// ops with substituted precisions.
pub(super) fn check_app_for_poly_op_constraint(
    list: &deep::List,
    defs: &DefBodyMap,
    type_env: &IrTypeEnv,
    scope: &HashMap<String, deep::Expr>,
    errors: &mut DiagnosticSink<'_>,
) {
    let kids = children(list);
    let Some(callee_expr) = kids.first() else {
        return;
    };
    // chelis#1107: carrier-preserving read. The walker's Node bridge rebuilds
    // only the `app` node, so the callee arrives as `Expr::Node` and a
    // `List`-only destructure returned before any call site was ever checked.
    let Some((callee_tag, _, callee_kids)) = stamped_parts(callee_expr) else {
        return;
    };
    if callee_tag != DeepTag::Var {
        return;
    }
    let Some(callee_name) = callee_kids.first().and_then(symbol_name) else {
        return;
    };
    // Look up the callee's declared signature; only proceed if it carries
    // a precision tvar (otherwise there's nothing to monomorphize).
    let Some(sig_expr) = lookup_sig_in_type_env(type_env, callee_name) else {
        return;
    };
    let Some((sig_params, _sig_ret)) = parse_t_fn_parts(sig_expr) else {
        return;
    };
    if !sig_params.iter().any(type_expr_has_tensor_prec_var) {
        return;
    }
    // Look up the callee's body so we can scan it.
    let Some((body_params, body)) = defs.get(callee_name) else {
        return;
    };
    // Build the call-site precision substitution: for each sig parameter
    // whose precision slot is `(t-var {} q)`, resolve the corresponding
    // call-site argument's concrete precision. Try the arg's `type:`
    // annotation first; fall back to the enclosing-def `scope` when the
    // arg is `(var x)` for an unannotated body-level reference.
    let mut subst: HashMap<String, String> = HashMap::new();
    for (sig_param, arg_expr) in sig_params.iter().zip(kids.iter().skip(1)) {
        let Some(prec_var_name) = precision_var_name_in_type_expr(sig_param) else {
            continue;
        };
        let arg_ty =
            annotated_type_of_expr(arg_expr).or_else(|| resolve_var_type_in_scope(arg_expr, scope));
        let Some(arg_ty) = arg_ty else {
            continue;
        };
        let Some(prim_name) = precision_prim_name_in_type_expr(&arg_ty) else {
            continue;
        };
        // Last-write-wins is fine: if the same `q` appears in multiple
        // params, the call site's unification already enforced consistency
        // (otherwise `chelis check` would have surfaced a precision
        // mismatch earlier in the pipeline).
        subst.insert(prec_var_name, prim_name);
    }
    if subst.is_empty() {
        return;
    }
    // Map the body's parameter names to the sig's parameter precision-var
    // names, so we can resolve `(var x)` inside the body to a precision
    // variable. The body's params and the sig's params line up by
    // position.
    let mut param_to_prec: HashMap<String, String> = HashMap::new();
    for (body_param_name, sig_param) in body_params.iter().zip(sig_params.iter()) {
        if let Some(prec_var_name) = precision_var_name_in_type_expr(sig_param) {
            param_to_prec.insert(body_param_name.clone(), prec_var_name);
        }
    }
    // Walk the body looking for restricted ops applied to body parameters
    // whose precision tvar (after substitution) violates §5.4 / §5.7.2.
    walk_body_for_restricted_ops(body, &param_to_prec, &subst, callee_name, list, errors);
}

/// Top-level def signature lookup: scan `type_env` for the callee's
/// declared `(t-fn ...)` signature.
pub(super) fn lookup_sig_in_type_env<'a>(
    type_env: &'a IrTypeEnv,
    name: &str,
) -> Option<&'a deep::Expr> {
    type_env.get(name).or_else(|| {
        // Fall back to terminal-name match (mirrors `lookup_declared_type_expr`).
        let mut matches = type_env.iter().filter_map(|(key, value)| {
            let key_terminal = key
                .rsplit_once("__")
                .map(|(_, t)| t)
                .unwrap_or(key.as_str());
            let key_terminal = key_terminal
                .rsplit_once('.')
                .map(|(_, t)| t)
                .unwrap_or(key_terminal);
            (key_terminal == name).then_some(value)
        });
        let first = matches.next()?;
        matches.next().is_none().then_some(first)
    })
}

pub(super) fn parse_t_fn_parts(expr: &deep::Expr) -> Option<(Vec<deep::Expr>, deep::Expr)> {
    // chelis#1107: carrier-preserving read. Every reader in this WS-A8 support
    // cluster took the stamped carrier only after `collect_defsig_exprs` was
    // fixed to see it; a `List`-only destructure here would have left the pass
    // half-live.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag != DeepTag::TFn {
        return None;
    }
    let (ret, args) = kids.split_last()?;
    Some((args.iter().map(|e| (*e).clone()).collect(), (*ret).clone()))
}

/// Strip a leading `(t-ref {} ...)` wrapper for precision-var probing;
/// the borrow doesn't affect the precision slot.
pub(super) fn strip_t_ref(expr: &deep::Expr) -> &deep::Expr {
    // chelis#1107: carrier-preserving read.
    if let Some((DeepTag::TRef, _, kids)) = stamped_parts(expr)
        && let Some(inner) = kids.first()
    {
        return inner;
    }
    expr
}

pub(super) fn type_expr_has_tensor_prec_var(expr: &deep::Expr) -> bool {
    let stripped = strip_t_ref(expr);
    // chelis#1107: carrier-preserving read.
    let Some((tag, _, kids)) = stamped_parts(stripped) else {
        return false;
    };
    match tag {
        DeepTag::TTensor => precision_var_name_in_type_expr(stripped).is_some(),
        DeepTag::TFn | DeepTag::TTuple | DeepTag::TAdt => {
            kids.iter().any(type_expr_has_tensor_prec_var)
        }
        _ => false,
    }
}

/// Pull the precision-var name out of a tensor-type expression's
/// last child (the precision slot). Returns `Some(name)` when the
/// slot is `(t-var {} name)`; `None` when concrete or non-tensor.
pub(super) fn precision_var_name_in_type_expr(expr: &deep::Expr) -> Option<String> {
    let stripped = strip_t_ref(expr);
    // chelis#1107: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(stripped)?;
    if tag != DeepTag::TTensor {
        return None;
    }
    let last = kids.last()?;
    let (prec_tag, _, prec_kids) = stamped_parts(last)?;
    if prec_tag != DeepTag::TVar {
        return None;
    }
    prec_kids.first().and_then(symbol_name).map(String::from)
}

/// Pull the concrete `(t-prim {} name)` from a tensor-type expression's
/// precision slot. Returns `None` when the slot is a tvar.
pub(super) fn precision_prim_name_in_type_expr(expr: &deep::Expr) -> Option<String> {
    let stripped = strip_t_ref(expr);
    // chelis#1107: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(stripped)?;
    if tag != DeepTag::TTensor {
        return None;
    }
    let last = kids.last()?;
    let (prec_tag, _, prec_kids) = stamped_parts(last)?;
    if prec_tag != DeepTag::TPrim {
        return None;
    }
    prec_kids.first().and_then(symbol_name).map(String::from)
}

/// Look up the type of `(var name)` in the enclosing-def `scope` map
/// (built from the def's sig + inline param annotations). Returns
/// `None` when the expression is not a var or the name is not in
/// scope.
pub(super) fn resolve_var_type_in_scope(
    expr: &deep::Expr,
    scope: &HashMap<String, deep::Expr>,
) -> Option<deep::Expr> {
    // chelis#1107: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag != DeepTag::Var {
        return None;
    }
    let name = kids.first().and_then(symbol_name)?;
    scope.get(name).cloned()
}

/// Pull the `type:` meta entry off a Deep expression. Returns the inner
/// type expression when present.
pub(super) fn annotated_type_of_expr(expr: &deep::Expr) -> Option<deep::Expr> {
    // chelis#1107: carrier-preserving read.
    let (_, meta, _) = stamped_parts(expr)?;
    meta.entries
        .iter()
        .find(|(k, _)| k == "type")
        .map(|(_, v)| v.clone())
}

/// Walk a polymorphic def's body looking for `(app (var op) arg1 arg2 ...)`
/// where `op` is one of the §5.4 (transcendental, float-only) or §5.7.2
/// (matmul, integer-rejected) restricted ops, and `arg1`/`arg2` are
/// `(var name)` references to body parameters. Validate the substituted
/// precision against the spec rule.
pub(super) fn walk_body_for_restricted_ops(
    expr: &deep::Expr,
    param_to_prec: &HashMap<String, String>,
    subst: &HashMap<String, String>,
    callee_name: &str,
    call_site_list: &deep::List,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("walk_body_for_restricted_ops", expr);
    // chelis#1107: carrier-preserving read, so the walk descends through
    // stamped `Expr::Node` bodies instead of stopping at the first one.
    if let Some((tag, _, kids)) = stamped_parts(expr) {
        if tag == DeepTag::App
            && let Some(callee) = kids.first()
            && let Some((DeepTag::Var, _, callee_kids)) = stamped_parts(callee)
            && let Some(op_name) = callee_kids.first().and_then(symbol_name)
        {
            check_restricted_op_in_body(
                op_name,
                &kids[1..],
                param_to_prec,
                subst,
                callee_name,
                call_site_list,
                errors,
            );
        }
        for child in kids {
            walk_body_for_restricted_ops(
                child,
                param_to_prec,
                subst,
                callee_name,
                call_site_list,
                errors,
            );
        }
    } else if let deep::Expr::MetaExpr(meta, _) = expr {
        walk_body_for_restricted_ops(
            &meta.expr,
            param_to_prec,
            subst,
            callee_name,
            call_site_list,
            errors,
        );
    }
}

/// `op_name`: the name of the inner operation (e.g. `matmul`, `exp`).
/// `op_args`: the arg expressions of the `(app (var op_name) ...)` form.
pub(super) const TRANSCENDENTAL_FLOAT_ONLY_OPS: &[&str] = &[
    "exp",
    "log",
    "sin",
    "cos",
    "sqrt",
    "tan",
    "atan",
    "softmax",
    "sigmoid",
    "tanh",
    "silu",
    "gelu",
    "layer_norm",
    "normalize",
    // `recip` is float-only per spec/05-risc-primitives.md §2.2: an
    // integer reciprocal has no meaningful IEEE-754 interpretation
    // (would always be 0 for |x| > 1 and undefined for x = 0).
    // `div` is NOT here — it is float-only too (chelis#178) but carries
    // a §2.1 citation pointing at `floor_div` / `trunc_div`, so it is
    // handled by `FLOAT_ONLY_DIV_OPS` with a tailored diagnostic.
    "recip",
];

/// Float-only ops whose integer-operand rejection cites
/// spec/05-risc-primitives.md §2.1 and points at the integer-division
/// replacements (chelis#178). Kept separate from
/// `TRANSCENDENTAL_FLOAT_ONLY_OPS` so the diagnostic names the migration
/// ops rather than the generic transcendental §5.4 rule.
pub(super) const FLOAT_ONLY_DIV_OPS: &[&str] = &["div"];

/// Integer-only ops whose float-operand rejection cites
/// spec/05-risc-primitives.md §2.1 (chelis#178). `trunc_div` is the
/// C/Rust truncating quotient and is not defined on float operands.
pub(super) const INTEGER_ONLY_DIV_OPS: &[&str] = &["trunc_div"];

pub(super) const INTEGER_REJECTED_OPS: &[&str] = &["matmul"];

pub(super) fn check_restricted_op_in_body(
    op_name: &str,
    op_args: &[deep::Expr],
    param_to_prec: &HashMap<String, String>,
    subst: &HashMap<String, String>,
    callee_name: &str,
    call_site_list: &deep::List,
    errors: &mut DiagnosticSink<'_>,
) {
    if !INTEGER_REJECTED_OPS.contains(&op_name)
        && !TRANSCENDENTAL_FLOAT_ONLY_OPS.contains(&op_name)
        && !FLOAT_ONLY_DIV_OPS.contains(&op_name)
        && !INTEGER_ONLY_DIV_OPS.contains(&op_name)
        && !BOOL_REJECTED_ARITH_OPS.contains(&op_name)
        && op_name != "mean"
    {
        return;
    }
    // Resolve each arg's precision through the param-to-prec mapping and
    // the call-site substitution. The arg may be either a direct
    // `(var x)` reference to a body param OR a deeper expression — for
    // the latter we look at its annotated type's precision slot.
    for (arg_index, arg) in op_args.iter().enumerate() {
        let resolved_prim = resolve_arg_precision_through_subst(arg, param_to_prec, subst);
        let Some(prim_name) = resolved_prim else {
            continue;
        };
        let Some(prim) = Prim::parse_name(&prim_name) else {
            continue;
        };
        let _ = call_site_list; // span hint reserved for future plumbing
        let first_data_arg = arg_index == 0;
        let consult_shared_policy =
            BOOL_REJECTED_ARITH_OPS.contains(&op_name) || (op_name == "mean" && first_data_arg);
        if consult_shared_policy
            && let Some((kind, message, mut hints)) =
                operand_dtype_rejection(op_name, &Type::Prim(prim))
        {
            hints.push(format!(
                "Reached through the polymorphic sig for `{callee_name}` instantiated at \
                 `{prim_name}`; the same operand-dtype rule applies to every such \
                 instantiation, including via stdlib wrappers."
            ));
            errors.push(CheckError::new(kind, message, hints));
            return;
        }
        if INTEGER_REJECTED_OPS.contains(&op_name) && prim.is_integer() {
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "{op_name} on integer operand precision `{prim_name}` is not \
                     admitted in this cycle per spec/04-type-system.md \u{00a7}5.7.2: \
                     integer matmul not admitted (the spec deliberately defers the \
                     integer-matmul accumulator rule). Reached through the polymorphic \
                     sig for `{callee_name}` instantiated at integer precision; the \
                     restriction fires on every integer instantiation, including via \
                     stdlib wrappers."
                ),
                vec![format!(
                    "spec/04-type-system.md \u{00a7}5.7.2: there is no current backend \
                     that supports integer BLAS. Use reduce_sum over an explicit \
                     expand+mul lowering for integer inner products, or float \
                     instantiations of `{callee_name}`."
                )],
            ));
            return;
        }
        if TRANSCENDENTAL_FLOAT_ONLY_OPS.contains(&op_name) && !prim.is_float() {
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "{op_name} on operand precision `{prim_name}` is not admitted per \
                     spec/04-type-system.md \u{00a7}5.4: transcendental operations are \
                     restricted to f32, f64, bf16, f16 (not integer). Reached through \
                     the polymorphic sig for `{callee_name}` instantiated at \
                     `{prim_name}`; the restriction fires on every non-float \
                     instantiation, including via stdlib wrappers."
                ),
                vec![format!(
                    "spec/04-type-system.md \u{00a7}5.4: cast to a float precision \
                     before applying `{op_name}`, or pick a float instantiation of \
                     `{callee_name}`."
                )],
            ));
            return;
        }
        if FLOAT_ONLY_DIV_OPS.contains(&op_name) && prim.is_integer() {
            // chelis#178: integer `div` reached through a polymorphic
            // wrapper instantiated at an integer dtype.
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "div on integer operand precision `{prim_name}` is not admitted per \
                     spec/05-risc-primitives.md \u{00a7}2.1: `div` is float-only \
                     (IEEE-754). Use `floor_div` (round toward -inf) or `trunc_div` \
                     (round toward zero) for integers. Reached through the polymorphic \
                     sig for `{callee_name}` instantiated at `{prim_name}`; the \
                     restriction fires on every integer instantiation, including via \
                     stdlib wrappers."
                ),
                vec![format!(
                    "spec/05-risc-primitives.md \u{00a7}2.1: integer division uses \
                     `floor_div` or `trunc_div`; pick a float instantiation of \
                     `{callee_name}` for `div`."
                )],
            ));
            return;
        }
        if INTEGER_ONLY_DIV_OPS.contains(&op_name) && prim.is_float() {
            // chelis#178: `trunc_div` reached through a polymorphic
            // wrapper instantiated at a float dtype.
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "trunc_div on float operand precision `{prim_name}` is not admitted \
                     per spec/05-risc-primitives.md \u{00a7}2.1: `trunc_div` is \
                     integer-only. Use `div` for IEEE-754 float division, or `floor_div` \
                     for a floored float quotient. Reached through the polymorphic sig \
                     for `{callee_name}` instantiated at `{prim_name}`."
                ),
                vec![format!(
                    "spec/05-risc-primitives.md \u{00a7}2.1: `trunc_div` requires integer \
                     operands; pick an integer instantiation of `{callee_name}`."
                )],
            ));
            return;
        }
    }
}

/// Resolve a single arg's precision-tvar binding for restricted-op
/// checking. If the arg is `(var name)` and `name` is in the body's
/// param-to-prec map, look up the call-site substitution.  If the arg
/// is itself an `app` with an annotated tensor type whose precision
/// slot is concrete, use that. Returns `Some(prim_name)` if resolvable.
pub(super) fn resolve_arg_precision_through_subst(
    arg: &deep::Expr,
    param_to_prec: &HashMap<String, String>,
    subst: &HashMap<String, String>,
) -> Option<String> {
    // chelis#1107: carrier-preserving read -- the last link in the WS-A8
    // chain, so a stamped `(var {} x)` operand resolves its precision.
    if let Some((DeepTag::Var, _, kids)) = stamped_parts(arg)
        && let Some(name) = kids.first().and_then(symbol_name)
        && let Some(prec_var) = param_to_prec.get(name)
        && let Some(prim) = subst.get(prec_var)
    {
        return Some(prim.clone());
    }
    // Fall back to the arg's annotated type's concrete precision (when
    // the body did its own arithmetic, e.g. `wx = matmul(x, w);
    // softmax(wx)`).
    if let Some(ty) = annotated_type_of_expr(arg)
        && let Some(prim) = precision_prim_name_in_type_expr(&ty)
    {
        return Some(prim);
    }
    None
}

pub(super) fn validate_ir_expr(
    expr: &deep::Expr,
    type_env: &IrTypeEnv,
    static_env: &mut HashMap<String, StaticValue>,
    failed_let_names: &mut HashSet<String>,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    stack_guard!("validate_ir_expr", expr, StaticValue::Unknown);
    match expr {
        deep::Expr::List(list, _) => {
            if get_tag(list) == Some(DeepTag::Module) {
                for elem in list.elements.iter().skip(3) {
                    validate_ir_expr(elem, type_env, static_env, failed_let_names, errors);
                }
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some(DeepTag::Def) {
                let kids = children(list);
                let Some(name) = kids.first().and_then(symbol_name) else {
                    return StaticValue::Unknown;
                };
                let Some(value_expr) = kids.get(1) else {
                    return StaticValue::Unknown;
                };
                let value =
                    validate_ir_expr(value_expr, type_env, static_env, failed_let_names, errors);
                static_env.insert(name.to_string(), value);
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some(DeepTag::Fn) {
                let scoped_env = extend_ir_env_with_fn_params(list, type_env);
                let mut scoped_static_env = static_env.clone();
                bind_fn_params_unknown(list, &mut scoped_static_env);
                for elem in &list.elements {
                    validate_ir_expr(
                        elem,
                        &scoped_env,
                        &mut scoped_static_env,
                        failed_let_names,
                        errors,
                    );
                }
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some(DeepTag::Let) {
                let kids = children(list);
                let mut scoped_static_env = static_env.clone();
                // Clone the type env on let-scope entry so each binding's
                // derivable IR-shape-sensitive type (e.g. conv2d's output
                // dims) can extend the env visible to the let body. Without
                // this the validator cannot resolve `(var y)` for a let-
                // bound `y = conv2d(...)` and silently rejects the next
                // shape-sensitive call that consumes `y` (RT-205 F5).
                let mut scoped_type_env = type_env.clone();
                if let Some(deep::Expr::List(bind_list, _)) = kids.first()
                    && get_tag(bind_list) == Some(DeepTag::Bind)
                {
                    let bind_children = children(bind_list);
                    let mut index = 0;
                    while index + 1 < bind_children.len() {
                        if let Some(name) = symbol_name(&bind_children[index]) {
                            let value_expr = &bind_children[index + 1];
                            // Recurse into the RHS so its own validation
                            // can push diagnostics and suppress downstream
                            // cascade errors via `failed_let_names` (set
                            // below when the RHS is a recognized
                            // shape-sensitive form whose output type is
                            // non-derivable, RT-205 round-4 / issue #212).
                            let value = validate_ir_expr(
                                value_expr,
                                &scoped_type_env,
                                &mut scoped_static_env,
                                failed_let_names,
                                errors,
                            );
                            scoped_static_env.insert(name.to_string(), value);
                            // If the RHS is a shape-sensitive IR builtin
                            // whose output type is derivable from its args,
                            // register the derived type so downstream uses
                            // of `name` resolve correctly.
                            let derived =
                                derive_ir_builtin_output_type(value_expr, &scoped_type_env);
                            match derived {
                                Some(ty) => {
                                    scoped_type_env.insert(name.to_string(), ty);
                                }
                                None => {
                                    // Mark as failed-derivation when the
                                    // RHS is structurally a recognized
                                    // shape-sensitive form (a known
                                    // shape-sensitive builtin or a
                                    // unary/binary passthrough wrapper
                                    // around one, recursively) but its
                                    // output type could not be derived.
                                    // This catches `y = conv2d(bad)`
                                    // and the R3 F-A passthrough cases
                                    // like `y = relu(conv2d(bad))`.
                                    //
                                    // RT-205 round-4 / issue #212: the
                                    // previous guard checked
                                    // `errors.len() > errs_before` to
                                    // detect an errored RHS, which fails
                                    // for chains of length 3+ because
                                    // cascade suppression already
                                    // silences the level-2 RHS's
                                    // diagnostic, so the level-2 name is
                                    // never marked and the level-3 RHS
                                    // re-emits a phantom error. The
                                    // structural check
                                    // `let_rhs_is_recognized_shape_sensitive`
                                    // does not depend on diagnostic
                                    // count and propagates the failed
                                    // marker unboundedly down the chain.
                                    //
                                    // The recognition is intentionally
                                    // narrow: a clean RHS that is not
                                    // a recognized shape-sensitive form
                                    // (e.g. a user-defined fn call) still
                                    // does NOT cause suppression
                                    // downstream, so legitimate
                                    // "really wrong arg" cases still
                                    // surface their own diagnostic.
                                    if let deep::Expr::List(_, _) = value_expr
                                        && let_rhs_is_recognized_shape_sensitive(value_expr)
                                    {
                                        failed_let_names.insert(name.to_string());
                                    }
                                }
                            }
                        }
                        index += 2;
                    }
                }
                if let Some(body) = kids.get(1) {
                    return validate_ir_expr(
                        body,
                        &scoped_type_env,
                        &mut scoped_static_env,
                        failed_let_names,
                        errors,
                    );
                }
                return StaticValue::Unknown;
            }
            if let Some(tag) = get_tag(list) {
                // `par` (sequential v1, spec/03-deep-syntax.md §2.3) and `jit`
                // (compilation trigger, §2.7) are spec-blessed pass-through
                // forms at Phase 0 evaluation. The validator used to reject
                // both; the rejection is removed because lowering handles them
                // (see `lower_par` and the `jit` lowering arm).
                if tag == DeepTag::App
                    && let Some(func_name) = ir_builtin_name(list)
                    && is_ir_shape_sensitive_builtin(func_name)
                {
                    validate_ir_builtin_symbolic_requirements(
                        list,
                        func_name,
                        type_env,
                        failed_let_names,
                        errors,
                    );
                }
            }

            if get_tag(list) == Some(DeepTag::Var)
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
            if get_tag(list) == Some(DeepTag::Lit) {
                return literal_static_value(expr);
            }
            if get_tag(list) == Some(DeepTag::Cast) {
                let kids = children(list);
                return kids
                    .first()
                    .map(|inner| {
                        validate_ir_expr(inner, type_env, static_env, failed_let_names, errors)
                    })
                    .unwrap_or(StaticValue::Unknown);
            }
            if get_tag(list) == Some(DeepTag::App) {
                let kids = children(list);
                let func_name = kids.first().and_then(app_builtin_name);
                let arg_values = kids
                    .iter()
                    .skip(1)
                    .map(|arg| {
                        validate_ir_expr(arg, type_env, static_env, failed_let_names, errors)
                    })
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
                validate_ir_expr(elem, type_env, static_env, failed_let_names, errors);
            }
            StaticValue::Unknown
        }
        deep::Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                validate_ir_expr(value, type_env, static_env, failed_let_names, errors);
            }
            StaticValue::Unknown
        }
        deep::Expr::MetaExpr(meta, _) => {
            for (_, value) in &meta.entries {
                validate_ir_expr(value, type_env, static_env, failed_let_names, errors);
            }
            validate_ir_expr(&meta.expr, type_env, static_env, failed_let_names, errors)
        }
        deep::Expr::Atom(_, _) => literal_static_value(expr),
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        deep::Expr::Node(node, span) => {
            let bridged = deep::Expr::List(node.to_list(*span), *span);
            validate_ir_expr(&bridged, type_env, static_env, failed_let_names, errors)
        }
        deep::Expr::BareList(elems, _) => {
            let mut last = StaticValue::Unknown;
            for child in elems {
                last = validate_ir_expr(child, type_env, static_env, failed_let_names, errors);
            }
            last
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                validate_ir_expr(child, type_env, static_env, failed_let_names, errors);
            }
            StaticValue::Unknown
        }
    }
}

/// Encode a checker-owned type as canonical Deep type metadata.
///
/// Lowering consumers use this boundary when they need the checker's
/// alias-resolved ADT field types without reparsing authored declarations.
pub fn type_to_deep_expr(ty: &Type) -> deep::Expr {
    type_to_deep_expr_with(ty, stamped_node_expr)
}

pub(super) fn type_to_legacy_deep_expr(ty: &Type) -> deep::Expr {
    type_to_deep_expr_with(ty, node_expr)
}

type NodeBuilder = fn(DeepTag, Vec<deep::Expr>) -> deep::Expr;

fn type_to_deep_expr_with(ty: &Type, make_node: NodeBuilder) -> deep::Expr {
    match ty {
        Type::Prim(prim) => make_node(DeepTag::TPrim, vec![symbol_expr(prim.name())]),
        Type::Fn(args, ret) => {
            let mut children: Vec<deep::Expr> = args
                .iter()
                .map(|arg| type_to_deep_expr_with(arg, make_node))
                .collect();
            children.push(type_to_deep_expr_with(ret, make_node));
            make_node(DeepTag::TFn, children)
        }
        Type::Ref(inner) => make_node(
            DeepTag::TRef,
            vec![type_to_deep_expr_with(inner, make_node)],
        ),
        Type::Tensor(dims, prec) => {
            let mut children: Vec<deep::Expr> = dims
                .iter()
                .map(|dim| dim_to_deep_expr_with(dim, make_node))
                .collect();
            children.push(match prec {
                TensorPrec::Concrete(p) => type_to_deep_expr_with(&Type::Prim(*p), make_node),
                TensorPrec::Var(v) => {
                    make_node(DeepTag::TVar, vec![symbol_expr(&format!("t{}", v.0))])
                }
            });
            make_node(DeepTag::TTensor, children)
        }
        Type::Adt(name, args) => {
            let mut children = vec![symbol_expr(name)];
            children.extend(
                args.iter()
                    .map(|arg| type_to_deep_expr_with(arg, make_node)),
            );
            make_node(DeepTag::TAdt, children)
        }
        Type::Var(var) => make_node(DeepTag::TVar, vec![symbol_expr(&format!("t{}", var.0))]),
        Type::Tuple(types) => make_node(
            DeepTag::TTuple,
            types
                .iter()
                .map(|ty| type_to_deep_expr_with(ty, make_node))
                .collect(),
        ),
        Type::Unit => make_node(DeepTag::TUnit, vec![]),
        Type::Error(_) => make_node(DeepTag::TVar, vec![symbol_expr("_")]),
    }
}

fn dim_to_deep_expr_with(dim: &Dim, make_node: NodeBuilder) -> deep::Expr {
    match dim {
        Dim::Name(name) => make_node(DeepTag::DName, vec![symbol_expr(name)]),
        Dim::Var(var) => make_node(DeepTag::DVar, vec![symbol_expr(&format!("d{}", var.0))]),
        Dim::Lit(value) => make_node(
            DeepTag::DLit,
            vec![deep::Expr::Atom(deep::Atom::Int(*value), zero_span())],
        ),
        Dim::Wildcard => make_node(DeepTag::DName, vec![symbol_expr("*")]),
        Dim::Rank(rank) => make_node(DeepTag::DRank, vec![symbol_expr(&format!("r{}", rank.0))]),
    }
}

pub(super) fn node_expr(tag: DeepTag, children: Vec<deep::Expr>) -> deep::Expr {
    let mut elements = vec![
        deep::Expr::Atom(deep::Atom::Tag(tag), zero_span()),
        deep::Expr::Map(deep::MetaMap::default(), zero_span()),
    ];
    elements.extend(children);
    deep::Expr::List(deep::List { elements }, zero_span())
}

pub(super) fn stamped_node_expr(tag: DeepTag, children: Vec<deep::Expr>) -> deep::Expr {
    deep::Expr::node(tag, deep::MetaMap::default(), children, zero_span())
}

pub(super) fn symbol_expr(name: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Name(name.to_string()), zero_span())
}

pub(super) fn zero_span() -> Span {
    Span::new(0, 0)
}

pub(super) fn span_of_expr(expr: &deep::Expr) -> Span {
    match expr {
        deep::Expr::Atom(_, span)
        | deep::Expr::List(_, span)
        | deep::Expr::Map(_, span)
        | deep::Expr::MetaExpr(_, span)
        | deep::Expr::Node(_, span)
        | deep::Expr::BareList(_, span) => *span,
        deep::Expr::UnknownForm(data) => data.span,
    }
}

pub(super) fn span_of_list(list: &deep::List) -> Span {
    list.elements
        .first()
        .map(span_of_expr)
        .unwrap_or_else(zero_span)
}

pub(super) fn ir_builtin_name(list: &deep::List) -> Option<&str> {
    ir_builtin_name_of_expr(list.elements.get(2)?)
}

/// The builtin callee name of an `app`'s callee child, on either carrier.
///
/// chelis#1107 amendment: `validate_ir_expr` bridges a stamped `Expr::Node`
/// one level (`Node::to_list`), so the callee child it hands on is still an
/// `Expr::Node`. The previous `List`-only read returned `None` for every
/// stamped callee, which silently disabled the shape-sensitivity and
/// output-type derivation below on the stamped carrier.
pub(super) fn ir_builtin_name_of_expr(func_expr: &deep::Expr) -> Option<&str> {
    let (DeepTag::Var, _, kids) = stamped_parts(func_expr)? else {
        return None;
    };
    match kids.first() {
        Some(deep::Expr::Atom(deep::Atom::Name(name), _)) => Some(name.as_str()),
        _ => None,
    }
}

/// Borrow `expr` as a `deep::List`, materializing a one-level bridge for a
/// stamped `Expr::Node` into `storage`.
///
/// chelis#1107 amendment: the `derive_*` output-type family threads
/// `&deep::List` through several helpers. Rather than change all of their
/// signatures, the entry points bridge once here; every leaf reader they call
/// (`tensor_precision_expr`, `ir_builtin_name`, …) is carrier-agnostic, so one
/// level is enough.
pub(super) fn as_list<'a>(
    expr: &'a deep::Expr,
    storage: &'a mut Option<deep::List>,
) -> Option<&'a deep::List> {
    match expr {
        deep::Expr::List(list, _) => Some(list),
        deep::Expr::Node(node, span) => Some(storage.insert(node.to_list(*span))),
        _ => None,
    }
}

pub(super) fn is_ir_shape_sensitive_builtin(name: &str) -> bool {
    matches!(
        name,
        "matmul"
            | "softmax"
            | "mean"
            | "layer_norm"
            | "conv2d"
            | "sum"
            | "count"
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

/// Is `name` a unary shape-passthrough op for the purposes of let-RHS
/// recognition? Must match the unary arm of
/// `derive_ir_builtin_output_type` so the failed-marker insertion in
/// the let arm covers the same surface as the type-derivation
/// passthrough recognition (issue #212 / RT-205 round-4).
pub(super) fn is_ir_unary_shape_passthrough_builtin(name: &str) -> bool {
    matches!(
        name,
        "relu"
            | "tanh"
            | "sigmoid"
            | "gelu"
            | "silu"
            | "exp"
            | "log"
            | "neg"
            | "recip"
            | "sqrt"
            | "abs"
            | "sin"
            | "cos"
            | "tan"
            | "atan"
            | "floor"
            | "ceil"
            | "round"
            | "not"
            | "softmax"
    )
}

/// Is `name` a binary shape-passthrough op? Must match the binary arm
/// of `derive_ir_builtin_output_type` for the same reason as
/// `is_ir_unary_shape_passthrough_builtin` (issue #212 / RT-205
/// round-4).
pub(super) fn is_ir_binary_shape_passthrough_builtin(name: &str) -> bool {
    matches!(
        name,
        "add"
            | "sub"
            | "mul"
            | "div"
            | "max_elem"
            | "min_elem"
            | "cmplt"
            | "lt"
            | "gt"
            | "gte"
            | "lte"
            | "eq"
            | "neq"
            | "and"
            | "or"
    )
}

/// Recognise a let-RHS expression as a "shape-sensitive form" for the
/// purposes of cascade-suppression marker insertion: either a direct
/// recognised shape-sensitive IR builtin, or a unary/binary shape-
/// passthrough wrapper around one (recursively). Peeks through
/// borrow wrappers like the rest of the validator.
///
/// Returns true when, structurally, this RHS shape COULD have a
/// derivable output type via `derive_ir_builtin_output_type`; the
/// caller pairs this with `derived.is_none()` to detect the "should
/// have derived but didn't" failure mode (issue #212 / RT-205
/// round-4). The decoupled structural check means we no longer
/// depend on whether the RHS validation pushed a diagnostic at this
/// level: cascade-suppressed intermediate let-binders are still
/// marked failed so the suppression propagates unboundedly down the
/// chain.
pub(super) fn let_rhs_is_recognized_shape_sensitive(expr: &deep::Expr) -> bool {
    stack_guard!("let_rhs_is_recognized_shape_sensitive", expr, false);
    let inner = peel_borrow(expr);
    // chelis#1107 amendment: carrier-preserving read.
    let Some((DeepTag::App, _, kids)) = stamped_parts(inner) else {
        return false;
    };
    let Some(func_name) = kids.first().and_then(ir_builtin_name_of_expr) else {
        return false;
    };
    if is_ir_shape_sensitive_builtin(func_name) {
        return true;
    }
    if is_ir_unary_shape_passthrough_builtin(func_name)
        && let Some(arg) = kids.get(1)
    {
        return let_rhs_is_recognized_shape_sensitive(arg);
    }
    if is_ir_binary_shape_passthrough_builtin(func_name) {
        // Either operand being a recognised shape-sensitive form is
        // sufficient: the passthrough derivation uses the first
        // resolvable operand's type and falls through to the second,
        // so a failed inner shape-sensitive call on either side
        // means the whole RHS is structurally broken.
        if let Some(lhs) = kids.get(1)
            && let_rhs_is_recognized_shape_sensitive(lhs)
        {
            return true;
        }
        if let Some(rhs) = kids.get(2)
            && let_rhs_is_recognized_shape_sensitive(rhs)
        {
            return true;
        }
    }
    false
}

pub(super) fn expr_type_expr(expr: &deep::Expr, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    stack_guard!("expr_type_expr", expr, None);
    match expr {
        deep::Expr::Node(node, _) => {
            if let Some((_, ty)) = node.meta().entries.iter().find(|(key, _)| key == "type") {
                return Some(ty.clone());
            }
            if node.tag() == DeepTag::Var
                && let Some(name) = node.children_slice().first().and_then(symbol_name)
            {
                return type_env.get(name).cloned();
            }
            None
        }
        deep::Expr::List(list, _) => {
            if let Some(meta) = get_meta(list)
                && let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type")
            {
                return Some(ty.clone());
            }
            if get_tag(list) == Some(DeepTag::Var)
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

pub(super) fn extend_ir_env_with_fn_params(
    fn_list: &deep::List,
    type_env: &IrTypeEnv,
) -> IrTypeEnv {
    let mut scoped = type_env.clone();
    let Some(params_expr) = children(fn_list).first() else {
        return scoped;
    };
    // chelis#1107 amendment: carrier-preserving read. `validate_ir_expr`
    // bridges only the `fn` node, so `(params {} ...)` arrives as `Expr::Node`.
    let Some((DeepTag::Params, _, param_entries)) = stamped_parts(params_expr) else {
        return scoped;
    };
    for param in param_entries {
        // An inline-annotated entry `(x {type: T})` is symbol-headed, so the
        // stamp pass carries it as `Expr::BareList`, never a `Node` -- this
        // one needs its own arm rather than `stamped_parts`.
        // chelis#1107 amendment (justified-safe, not routed): this match
        // handles every carrier a params entry can take -- `List` (legacy) and
        // `BareList` (stamped, symbol-headed) -- so there is no fall-through.
        let (name, meta) = match param {
            deep::Expr::List(param_list, _) => {
                let Some(name) = param_list.elements.first().and_then(symbol_name) else {
                    continue;
                };
                let Some(meta) = get_meta(param_list) else {
                    continue;
                };
                (name, meta)
            }
            deep::Expr::BareList(elems, _) => {
                let Some(name) = elems.first().and_then(symbol_name) else {
                    continue;
                };
                let Some(deep::Expr::Map(meta, _)) = elems.get(1) else {
                    continue;
                };
                (name, meta)
            }
            _ => continue,
        };
        let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type") else {
            continue;
        };
        scoped.insert(name.to_string(), ty.clone());
    }
    scoped
}

pub(super) fn expr_tensor_type_is_concrete(expr: &deep::Expr, type_env: &IrTypeEnv) -> bool {
    // Peel `(borrow {} ...)` so the idiomatic Surf borrow form does
    // not silently bypass the dim-concreteness check.
    arg_tensor_type_expr(expr, type_env)
        .map(|ty| type_expr_is_ir_concrete(&ty))
        .unwrap_or(false)
}

/// Check whether a conv2d input tensor argument is concrete in every
/// dimension EXCEPT axis 0 (batch). Per spec/05-risc-primitives.md
/// §4.5 the canonical signature is `tensor[batch, in_c, h, w, p]`
/// and `batch` is named, so symbolic-batch programs are first-class
/// (RT-205 round-3 F-C). The spatial dims and `in_c` must remain
/// concrete because they appear in the im2col/matmul lowering.
///
/// Returns true when the type resolves to a rank-4 tensor whose
/// axes 1, 2, 3 are all `Dim::Lit`. Axis 0 may be `Dim::Lit` or
/// `Dim::NonConcrete`. Returns false on unresolvable type or any
/// non-concrete axis other than 0.
pub(super) fn conv2d_input_dims_concrete_modulo_batch(
    expr: Option<&deep::Expr>,
    type_env: &IrTypeEnv,
) -> bool {
    let Some(expr) = expr else {
        return false;
    };
    let Some(ty) = arg_tensor_type_expr(expr, type_env) else {
        return false;
    };
    let Some(dims) = tensor_dims_from_type_expr(&ty) else {
        // Not a tensor; fall back to scalar-prim check.
        return type_expr_is_ir_concrete(&ty);
    };
    if dims.len() != 4 {
        // Rank mismatch is reported separately; return true so the
        // rank-4 guard later in the validator can fire instead of
        // suppressing it with a metadata error.
        return true;
    }
    // axes 1, 2, 3 must be concrete; axis 0 (batch) may be symbolic.
    dims[1..].iter().all(|d| matches!(d, DeepDimKind::Lit(_)))
}

pub(super) fn validate_ir_builtin_symbolic_requirements(
    list: &deep::List,
    func_name: &str,
    type_env: &IrTypeEnv,
    failed_let_names: &HashSet<String>,
    errors: &mut DiagnosticSink<'_>,
) {
    match func_name {
        "conv2d" => validate_conv2d_symbolic_requirements(list, type_env, failed_let_names, errors),
        "mean" if ir_builtin_axis_dim(list, type_env, 0, 1) == Some(DeepDimKind::NonConcrete) => {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                "IR builtin `mean` requires a concrete reduced axis extent".to_string(),
                vec!["Use a concrete d-lit dimension on the reduced axis".to_string()],
            ));
        }
        "layer_norm" => {
            let x_dims = list
                .elements
                .get(3)
                .and_then(|expr| arg_tensor_type_expr(expr, type_env))
                .and_then(|ty| tensor_dims_from_type_expr(&ty));
            if matches!(
                x_dims.as_ref().and_then(|dims| dims.last()),
                Some(DeepDimKind::NonConcrete)
            ) {
                errors.push(validator_error(
                    CheckErrorKind::DimensionMismatch,
                    list,
                    "IR builtin `layer_norm` requires a concrete normalized axis extent"
                        .to_string(),
                    vec!["Use a concrete d-lit dimension for the final axis".to_string()],
                ));
            }
        }
        _ => {}
    }
}

/// Return `true` if any of `list`'s tensor arguments (positional 3, 4)
/// is a `(var <name>)` whose `name` is in `failed_let_names`. Used by
/// `validate_conv2d_symbolic_requirements` to suppress the cascade
/// diagnostic when a let-bound name's own derivation already emitted
/// the owning diagnostic (RT-205 round-2 F3).
pub(super) fn conv2d_input_is_failed_let_name(
    list: &deep::List,
    failed_let_names: &HashSet<String>,
) -> bool {
    if failed_let_names.is_empty() {
        return false;
    }
    for arg in list.elements.iter().skip(3).take(2) {
        let inner = peel_borrow(arg);
        if let deep::Expr::List(arg_list, _) = inner
            && get_tag(arg_list) == Some(DeepTag::Var)
            && let Some(name) = children(arg_list).first().and_then(symbol_name)
            && failed_let_names.contains(name)
        {
            return true;
        }
    }
    false
}

/// Build a `CheckError` for a validator-arm diagnostic that
/// references a specific call site. Appends the call site's `:span`
/// metadata identifier (if present) to the message so JSON consumers
/// can locate the offending expression in the source.
///
/// All shape-sensitive validator errors flow through this helper so
/// they uniformly get DimensionMismatch-grade severity and span
/// suffixes, matching the inference-layer DimensionMismatch surface
/// that JSON tooling already understands (RT-205 F6).
pub(super) fn validator_error(
    kind: CheckErrorKind,
    call_site: &deep::List,
    message: String,
    suggestions: Vec<String>,
) -> CheckError {
    let suffixed = match validator_span_suffix(call_site) {
        Some(span) => format!("{message} {span}"),
        None => message,
    };
    CheckError::new(kind, suffixed, suggestions)
}

/// Render the call site's source span as a parenthesized suffix
/// (e.g. ` (at surf:144..165)`). Returns `None` when the call site
/// carries no `:span` metadata so the unmodified message is used.
pub(super) fn validator_span_suffix(call_site: &deep::List) -> Option<String> {
    let meta = get_meta(call_site)?;
    for (key, value) in &meta.entries {
        if key == "span"
            && let deep::Expr::Atom(deep::Atom::Str(s), _) = value
        {
            return Some(format!("(at {s})"));
        }
    }
    None
}

/// Extract the `:span` metadata string from a list node, if present.
/// Used to propagate external span identifiers into check diagnostics.
pub(super) fn list_span_id(list: &deep::List) -> Option<&str> {
    let meta = get_meta(list)?;
    for (key, value) in &meta.entries {
        if key == "span"
            && let deep::Expr::Atom(deep::Atom::Str(s), _) = value
        {
            return Some(s.as_str());
        }
    }
    None
}

/// Parse the start byte offset from a span identifier string.
/// Handles the `"surf:<start>..<end>"` format emitted by the desugar step
/// and bare `"<start>..<end>"` ranges.
pub(super) fn parse_span_offset(span_id: &str) -> Option<usize> {
    // Format: "surf:10..25" or "10..25" or "octant:line:7" (opaque)
    let numeric_part = span_id
        .rfind(':')
        .map(|i| &span_id[i + 1..])
        .unwrap_or(span_id);
    // Try to parse "start..end"
    numeric_part
        .split_once("..")
        .and_then(|(start, _)| start.parse::<usize>().ok())
}

/// Validate the symbolic requirements of an IR-level `conv2d` call.
///
/// The previous implementation read `:type` from the app node's
/// metadata via `app_result_type_is_concrete` to decide whether the
/// output dims were concrete. Surf-desugared apps only carry `:span`
/// metadata; the annotation pass that would stamp inferred app types
/// back into Deep runs after `validate_ir_program`, so that check was
/// structurally always-false for any Surf source (see issue #186).
///
/// The replacement derives output concreteness from the arguments
/// (input tensor dims, kernel tensor dims, stride/padding literal
/// values), all of which are knowable at validation time. After
/// extracting the args this function evaluates the output spatial-
/// dim formula `floor((in + 2 * padding - kernel) / stride) + 1`
/// per axis (spec/05-risc-primitives.md §471-483) and rejects calls
/// whose evaluated output dim is non-positive. Also enforces rank-4
/// input/kernel and positive-stride / non-negative-padding.
pub(super) fn validate_conv2d_symbolic_requirements(
    list: &deep::List,
    type_env: &IrTypeEnv,
    failed_let_names: &HashSet<String>,
    errors: &mut DiagnosticSink<'_>,
) {
    // The inference pass owns builtin arity diagnostics. This validator only
    // owns the symbolic requirements of the canonical 4-argument call:
    // (app {} (var conv2d) input kernel stride padding). Returning here keeps
    // malformed calls total and prevents a secondary validator diagnostic.
    let [_, _, _, input, kernel, _, _] = list.elements.as_slice() else {
        return;
    };

    // RT-205 round-2 F3: if either tensor arg is a `(var <name>)`
    // whose `name` is in the failed-derivation set, the owning
    // diagnostic was already emitted for the let-binding's own RHS.
    // Suppress the cascade so the user sees one error per root cause,
    // not one per consumer.
    if conv2d_input_is_failed_let_name(list, failed_let_names) {
        return;
    }
    // Args at elements[3]..[6] for the canonical 4-arg call shape:
    // (app {} (var conv2d) input kernel stride padding).
    //
    // RT-205 round-3 F-C: input axis 0 (batch) is allowed to be
    // NonConcrete per spec/05 §4.5, since it does not enter the
    // spatial-dim formula and conv2d's IR lowering can carry a
    // symbolic batch through. All OTHER input axes (in_c, h, w) and
    // all kernel axes must remain concrete -- they appear in the
    // im2col/matmul lowering and must be statically knowable.
    if !conv2d_input_dims_concrete_modulo_batch(Some(peel_borrow(input)), type_env)
        || !expr_tensor_type_is_concrete(kernel, type_env)
    {
        errors.push(validator_error(
            CheckErrorKind::DimensionMismatch,
            list,
            "IR builtin `conv2d` requires concrete tensor argument metadata".to_string(),
            vec![
                "Use concrete d-lit dimensions for IR lowering (axis 0 / batch may be symbolic)"
                    .to_string(),
            ],
        ));
        return;
    }
    // Extract and range-check stride/padding. The IR lowering relies
    // on these being statically-knowable positive (stride) or
    // non-negative (padding) integers; the output spatial dim formula
    // `floor((in + 2p - k) / s) + 1` (spec/05-risc-primitives.md
    // §471-483) divides by stride, so `stride <= 0` is undefined and
    // a negative padding shrinks the effective input below zero.
    // Without these guards the validator silently accepts the
    // ill-formed call and the back-end ICEs at codegen time
    // (issue #186 RT findings F1, F2, F3).
    let stride = match extract_typed_scalar_literal(list, 5, "stride", errors) {
        Some(v) => v,
        None => return,
    };
    if stride <= 0 {
        errors.push(validator_error(
            CheckErrorKind::DimensionMismatch,
            list,
            format!("IR builtin `conv2d` requires a positive stride, got {stride}"),
            vec!["Stride must be >= 1; the output dim formula divides by stride".to_string()],
        ));
        return;
    }
    let padding = match extract_typed_scalar_literal(list, 6, "padding", errors) {
        Some(v) => v,
        None => return,
    };
    if padding < 0 {
        errors.push(validator_error(
            CheckErrorKind::DimensionMismatch,
            list,
            format!("IR builtin `conv2d` requires non-negative padding, got {padding}"),
            vec!["Padding must be >= 0".to_string()],
        ));
        return;
    }
    // Resolve input + kernel tensor dims so we can evaluate the
    // output spatial-dim formula. expr_tensor_type_is_concrete above
    // already established concreteness; the lookups below should both
    // succeed, but bail gracefully on the unexpected case rather than
    // unwrap-panicking.
    let Some(input_dims) = list
        .elements
        .get(3)
        .and_then(|expr| arg_tensor_type_expr(expr, type_env))
        .and_then(|ty| tensor_dims_from_type_expr(&ty))
    else {
        return;
    };
    let Some(kernel_dims) = list
        .elements
        .get(4)
        .and_then(|expr| arg_tensor_type_expr(expr, type_env))
        .and_then(|ty| tensor_dims_from_type_expr(&ty))
    else {
        return;
    };
    // Rank guard: the canonical conv2d shape is [N, C, H, W] x [F, C, kH, kW].
    // The HM signature check (check_conv2d_signature, infer.rs:9999+) also
    // catches rank errors and may have already emitted its diagnostic via
    // `check_conv2d_signature`. Dedupe so the user sees ONE rank error per
    // role (input/kernel), not two (RT-205 round-2 F4).
    if input_dims.len() != 4 {
        let rank = input_dims.len();
        let hm_emitted = errors.iter().any(|e| {
            e.message.contains(&format!(
                "conv2d expects rank-4 input tensor, got rank {rank}"
            ))
        });
        if !hm_emitted {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!("IR builtin `conv2d` requires a rank-4 input tensor, got rank {rank}"),
                vec!["Pass a [N, C, H, W] tensor as the first argument".to_string()],
            ));
        }
        return;
    }
    if kernel_dims.len() != 4 {
        let rank = kernel_dims.len();
        let hm_emitted = errors.iter().any(|e| {
            e.message.contains(&format!(
                "conv2d expects rank-4 kernel tensor, got rank {rank}"
            ))
        });
        if !hm_emitted {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!("IR builtin `conv2d` requires a rank-4 kernel tensor, got rank {rank}"),
                vec!["Pass a [F, C, kH, kW] tensor as the second argument".to_string()],
            ));
        }
        return;
    }
    // Output spatial-dim formula per spec/05-risc-primitives.md §471-483:
    //   out = floor((in + 2 * padding - kernel) / stride) + 1
    // for both H (axis 2) and W (axis 3). If either evaluates to <= 0
    // the call is ill-formed; without this guard the back-end emits a
    // less-actionable error after codegen begins.
    let in_h = match input_dims[2] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return,
    };
    let in_w = match input_dims[3] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return,
    };
    let k_h = match kernel_dims[2] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return,
    };
    let k_w = match kernel_dims[3] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return,
    };
    for (axis, name, in_extent, k_extent) in [("H", "height", in_h, k_h), ("W", "width", in_w, k_w)]
        .iter()
        .map(|(short, long, inp, kr)| (*short, *long, *inp, *kr))
    {
        let Some(val) = conv2d_output_extent(in_extent, k_extent, stride, padding) else {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!(
                    "IR builtin `conv2d` output {name} (axis {axis}) cannot be computed: input={in_extent}, kernel={k_extent}, stride={stride}, padding={padding} overflows i64 in the canonical formula"
                ),
                vec![
                    "Use input/kernel/stride/padding values whose intermediate `input + 2 * padding - kernel` and final `+ 1` fit in a signed 64-bit integer".to_string(),
                ],
            ));
            return;
        };
        if val <= 0 {
            // Reaching this branch implies `conv2d_output_extent`
            // returned `Some(val)`, which in turn means
            // `padding.checked_mul(2)` and
            // `in_extent.checked_add(2 * padding)` both succeeded
            // upstream. Plain arithmetic is safe here; the
            // saturating-mul + checked-add fallback that earlier
            // code carried is unreachable. (RT-205 round-3 F-D.)
            let padded_hint = in_extent + 2 * padding;
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!(
                    "IR builtin `conv2d` output {name} (axis {axis}) evaluates to {val} for input={in_extent}, kernel={k_extent}, stride={stride}, padding={padding}; output dims must be positive"
                ),
                vec![format!(
                    "Increase padding, decrease stride, or shrink the kernel so the padded input ({padded_hint}) is at least the kernel size ({k_extent})"
                )],
            ));
            return;
        }
    }
}

/// Compute the output spatial extent of a conv2d axis using the
/// canonical formula `floor((in + 2 * padding - kernel) / stride) + 1`
/// (spec/05-risc-primitives.md §471-483). Returns a signed value so
/// the validator can detect ill-formed configurations (output <= 0)
/// before they reach the back-end.
///
/// Uses `div_euclid` for floor division so a negative numerator (the
/// kernel does not fit the padded input) produces an informative
/// negative output value rather than truncating toward zero.
/// `stride` is required to be positive by the caller, which is what
/// makes `div_euclid` equivalent to mathematical floor here.
///
/// Returns `None` on integer overflow in any intermediate (RT-205
/// round-2 F1). Callers must treat `None` as "input parameters
/// outside the representable range" and emit a diagnostic; previously
/// a huge `padding` like `i64::MAX/2` triggered `attempt to multiply
/// with overflow` and panicked `chelis check`.
pub(super) fn conv2d_output_extent(
    input: i64,
    kernel: i64,
    stride: i64,
    padding: i64,
) -> Option<i64> {
    let two_p = padding.checked_mul(2)?;
    let padded = input.checked_add(two_p)?;
    let numerator = padded.checked_sub(kernel)?;
    numerator.checked_div_euclid(stride)?.checked_add(1)
}

/// If `expr` is a recognizable shape-sensitive IR builtin call whose
/// output tensor type can be derived from its argument types and
/// literal scalar args, return that type as a Deep `(t-tensor ...)`
/// expression. Used to extend the validator's per-let-scope type env
/// so downstream uses of a let-bound name resolve to a concrete
/// tensor type (RT-205 F5).
///
/// In addition to `conv2d` direct calls, this also handles
/// shape-PRESERVING unary and binary point-wise ops (relu, tanh,
/// add, mul, etc.) so the canonical CNN layer pattern
/// `y = relu(conv2d(...))` chains correctly into a downstream
/// `conv2d(&y, ...)` (RT-205 round-2 F2). Reductions and movement
/// ops are intentionally NOT handled here; they would need a
/// separate per-op derivation because they change rank or shape.
///
/// Returns `None` when the call shape is unrecognized, the args are
/// non-concrete, or the derived output would be ill-formed (in which
/// case the validator's own arm will report the diagnostic).
pub(super) fn derive_ir_builtin_output_type(
    expr: &deep::Expr,
    type_env: &IrTypeEnv,
) -> Option<deep::Expr> {
    // chelis#1107 amendment: carrier-preserving entry. The `derive_*` helpers
    // below take `&deep::List`, so bridge a stamped Node once here.
    let mut bridge = None;
    let list = as_list(expr, &mut bridge)?;
    if get_tag(list) != Some(DeepTag::App) {
        return None;
    }
    let func_name = ir_builtin_name(list)?;
    match func_name {
        "conv2d" => derive_conv2d_output_type(list, type_env),
        // Shape-preserving unary point-wise: output type == input type.
        // Each entry below is cross-verified against the lowerer's
        // accepted name set in `crates/chelis-ir/src/lower.rs` (the
        // canonical IR vocabulary) and spec/05-risc-primitives.md
        // §2.2 / §3.3 (RT-205 round-3 F-B audit).
        //
        // Reductions (sum, mean, max_reduce, argmax_reduce,
        // prod_reduce, min_reduce, argmin_reduce) and movement ops
        // (reshape, permute, gather, pad, shrink, stride, expand) are
        // EXCLUDED: they change rank or shape and need per-op
        // derivation.
        //
        // softmax takes a (tensor, axis) tuple but its output shape
        // equals the input tensor's shape, so it fits the unary
        // passthrough path (positional [3] is the tensor).
        "relu" | "tanh" | "sigmoid" | "gelu" | "silu" | "exp" | "log" | "neg" | "recip"
        | "sqrt" | "abs" | "sin" | "cos" | "tan" | "atan" | "floor" | "ceil" | "round" | "not"
        | "softmax" => derive_unary_shape_passthrough(list, type_env),
        // Shape-preserving binary point-wise: output type == first
        // operand's type. Broadcasting cases are caught by HM
        // elsewhere; here we fall through to None if the first
        // operand's type is not derivable and try the second.
        //
        // RT-205 round-3 F-B: `maximum` and `minimum` were the wrong
        // names. The canonical IR names per spec/05 §2.1 and §3.4 are
        // `max_elem` (Tier 1) and `min_elem` (Tier 2). The lowerer
        // accepts `max_elem`/`min_elem` (lower.rs:1329-1330);
        // `maximum`/`minimum` do not appear anywhere in the IR
        // vocabulary, so the old allowlist never matched.
        //
        // `lt` is an alias for `cmplt` accepted at lowerer.rs:3913
        // (kept). `gte`, `lte`, `neq` are Tier 2 comparison ops
        // (spec/05 §3.2) accepted by the lowerer (lower.rs:1349-1352)
        // and added here so passthrough recognizes them. `and`, `or`
        // are bool binaries (lower.rs:1353-1354).
        "add" | "sub" | "mul" | "div" | "max_elem" | "min_elem" | "cmplt" | "lt" | "gt" | "gte"
        | "lte" | "eq" | "neq" | "and" | "or" => derive_binary_shape_passthrough(list, type_env),
        _ => None,
    }
}

/// Derive the output tensor type of a shape-preserving unary
/// point-wise call: it equals the type of the single argument.
/// Recurses through nested apps so e.g. `relu(conv2d(...))`
/// resolves to conv2d's derived output type, peeking through any
/// borrow wrapper as usual (RT-205 round-2 F2).
pub(super) fn derive_unary_shape_passthrough(
    list: &deep::List,
    type_env: &IrTypeEnv,
) -> Option<deep::Expr> {
    let arg = list.elements.get(3)?;
    resolve_let_value_tensor_type(arg, type_env)
}

/// Derive the output tensor type of a shape-preserving binary
/// point-wise call: it equals the type of whichever operand is
/// concretely resolvable (typically the first). Broadcasting and
/// dtype-promotion cases are caught by HM elsewhere; this helper
/// only needs to surface a shape that the next validator arm can
/// inspect (RT-205 round-2 F2).
pub(super) fn derive_binary_shape_passthrough(
    list: &deep::List,
    type_env: &IrTypeEnv,
) -> Option<deep::Expr> {
    let lhs = list.elements.get(3)?;
    if let Some(ty) = resolve_let_value_tensor_type(lhs, type_env) {
        return Some(ty);
    }
    let rhs = list.elements.get(4)?;
    resolve_let_value_tensor_type(rhs, type_env)
}

/// Resolve the tensor type expression of a let-binding RHS or any
/// nested sub-expression: try the borrow-aware var/lit lookup first,
/// and if that fails recurse into the sub-expression as another
/// recognized shape-sensitive call. Used by the unary and binary
/// passthrough helpers (RT-205 round-2 F2).
pub(super) fn resolve_let_value_tensor_type(
    expr: &deep::Expr,
    type_env: &IrTypeEnv,
) -> Option<deep::Expr> {
    if let Some(ty) = arg_tensor_type_expr(expr, type_env) {
        return Some(ty);
    }
    // Peek through borrow before recursing in case a wrapper op
    // appears under an `&` borrow (uncommon but cheap).
    let inner = peel_borrow(expr);
    derive_ir_builtin_output_type(inner, type_env)
}

/// Derive a conv2d call's output tensor type (rank-4 `[N, F, outH, outW]`
/// with the input's precision) from its argument types and literal
/// stride/padding values. Returns `None` if any non-batch input dim
/// or any kernel dim is non-concrete, stride/padding are not int
/// literals, ranks are wrong, or the output dims would be non-positive.
///
/// RT-205 round-3 F-C: input axis 0 (batch) is allowed to be
/// `Dim::NonConcrete` per spec/05 §4.5. When the input batch is
/// symbolic, the synthesized output type preserves the input
/// tensor's raw batch-dim expression (e.g. `(d-name {} batch)`)
/// rather than forcing a `d-lit`. This lets downstream chained
/// conv2d calls resolve `&y` to the symbolic-batch type.
pub(super) fn derive_conv2d_output_type(
    list: &deep::List,
    type_env: &IrTypeEnv,
) -> Option<deep::Expr> {
    let input_ty = list
        .elements
        .get(3)
        .and_then(|e| arg_tensor_type_expr(e, type_env))?;
    let kernel_ty = list
        .elements
        .get(4)
        .and_then(|e| arg_tensor_type_expr(e, type_env))?;
    let input_dims = tensor_dims_from_type_expr(&input_ty)?;
    let kernel_dims = tensor_dims_from_type_expr(&kernel_ty)?;
    if input_dims.len() != 4 || kernel_dims.len() != 4 {
        return None;
    }
    // Issue #216: cast-aware so `conv2d(x, k, cast(1, int32), cast(0, int32))`
    // surfaces the same derived output type as the bare-literal form.
    let stride = extract_int_for_dim(list.elements.get(5)?)?;
    let padding = extract_int_for_dim(list.elements.get(6)?)?;
    if stride <= 0 || padding < 0 {
        return None;
    }
    // Capture the input tensor's raw batch-dim Expr (axis 0) so a
    // symbolic batch can pass through verbatim into the synthesized
    // output type. axes 1-3 must be concrete literals (RT-205 r3 F-C).
    let input_dim_exprs = tensor_dim_exprs_from_type_expr(&input_ty)?;
    if input_dim_exprs.len() != 4 {
        return None;
    }
    let batch_dim_expr = input_dim_exprs[0].clone();
    let f = match kernel_dims[0] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    let in_h = match input_dims[2] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    let in_w = match input_dims[3] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    let k_h = match kernel_dims[2] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    let k_w = match kernel_dims[3] {
        DeepDimKind::Lit(v) => v,
        DeepDimKind::NonConcrete => return None,
    };
    // `conv2d_output_extent` returns None on integer overflow (RT-205
    // round-2 F1); in that case there's no valid output tensor type
    // to register, so the caller falls back to no extension and the
    // validator's own arm will emit the overflow diagnostic.
    let out_h = conv2d_output_extent(in_h, k_h, stride, padding)?;
    let out_w = conv2d_output_extent(in_w, k_w, stride, padding)?;
    if out_h <= 0 || out_w <= 0 {
        return None;
    }
    // Build `(t-tensor {} <batch-expr> (d-lit {} f) (d-lit {} out_h)
    // (d-lit {} out_w) <precision-expr>)` from the input's precision
    // and the captured batch-dim expression (which may be a symbolic
    // `(d-name {} ...)` per RT-205 r3 F-C).
    let prec_expr = tensor_precision_expr(&input_ty)?;
    Some(build_tensor_type_expr_with_batch(
        batch_dim_expr,
        &[f, out_h, out_w],
        prec_expr,
    ))
}

/// Return the raw Deep `Expr` for each dimension in a `(t-tensor {} dim1
/// dim2 ... prec)`. Unlike `tensor_dims_from_type_expr`, which returns a
/// `DeepDimKind` flattening, this preserves the original
/// `(d-name {} batch)` / `(d-var {} ...)` / `(d-lit {} N)` sub-expression
/// so the caller can carry it forward verbatim when synthesizing a
/// derived tensor type (RT-205 round-3 F-C, symbolic batch propagation).
pub(super) fn tensor_dim_exprs_from_type_expr(expr: &deep::Expr) -> Option<Vec<deep::Expr>> {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(list) == Some(DeepTag::TRef) {
        return children(list)
            .first()
            .and_then(tensor_dim_exprs_from_type_expr);
    }
    if get_tag(list) != Some(DeepTag::TTensor) {
        return None;
    }
    let kids = children(list);
    if kids.is_empty() {
        return None;
    }
    Some(kids[..kids.len().saturating_sub(1)].to_vec())
}

/// Extract the precision sub-expression (last child) of a
/// `(t-tensor {} dim1 dim2 ... precision)` expression. Returns the
/// raw Deep `Expr` so it can be re-used unchanged when synthesizing
/// a derived tensor type.
pub(super) fn tensor_precision_expr(ty: &deep::Expr) -> Option<deep::Expr> {
    // chelis#1107 amendment: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(ty)?;
    if tag == DeepTag::TRef {
        return kids.first().and_then(tensor_precision_expr);
    }
    if tag != DeepTag::TTensor {
        return None;
    }
    kids.last().cloned()
}

/// Build a synthetic `(t-tensor {} <batch-dim-expr> (d-lit {} d1)
/// (d-lit {} d2) ... prec)`, placing a verbatim Deep expression at
/// axis 0 (the batch dim) and integer literals for the remaining
/// axes. Used to preserve symbolic batch (`(d-name {} batch)`) when
/// deriving a chained conv2d's output type (RT-205 round-3 F-C).
/// Spans are zeroed because the derived type is synthetic; downstream
/// lookups care only about the structural shape.
pub(super) fn build_tensor_type_expr_with_batch(
    batch_dim: deep::Expr,
    other_dims: &[i64],
    prec: deep::Expr,
) -> deep::Expr {
    let zero = zero_span();
    let empty_meta = || deep::MetaMap { entries: vec![] };
    let make_d_lit = |v: i64| {
        deep::Expr::List(
            deep::List {
                elements: vec![
                    deep::Expr::Atom(deep::Atom::Tag(DeepTag::DLit), zero),
                    deep::Expr::Map(empty_meta(), zero),
                    deep::Expr::Atom(deep::Atom::Int(v), zero),
                ],
            },
            zero,
        )
    };
    let mut elements = vec![
        deep::Expr::Atom(deep::Atom::Tag(DeepTag::TTensor), zero),
        deep::Expr::Map(empty_meta(), zero),
    ];
    elements.push(batch_dim);
    for &d in other_dims {
        elements.push(make_d_lit(d));
    }
    elements.push(prec);
    deep::Expr::List(deep::List { elements }, zero)
}

/// Look up positional arg `idx` of a `conv2d` call, attempt to
/// extract it as an integer literal, and emit a clear diagnostic if
/// the arg is missing or non-literal.
///
/// `label` names the role (`"stride"` / `"padding"`) for the error
/// message. Returns `Some(value)` on success and `None` when an error
/// was pushed (the caller should bail to avoid piling on cascading
/// diagnostics).
pub(super) fn extract_typed_scalar_literal(
    list: &deep::List,
    idx: usize,
    label: &str,
    errors: &mut DiagnosticSink<'_>,
) -> Option<i64> {
    let Some(arg) = list.elements.get(idx) else {
        // Arity mismatch is caught elsewhere; bail without piling on.
        return None;
    };
    // Issue #216: cast-aware so a cast-wrapped literal (e.g.
    // `conv2d(x, k, cast(0, int32), 0)`) lands the precise
    // positive-stride / non-negative-padding diagnostic instead of the
    // misleading "requires a literal integer stride" message that
    // pre-fix appeared whenever the literal was wrapped.
    match extract_int_for_dim(arg) {
        Some(v) => Some(v),
        None => {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!("IR builtin `conv2d` requires a literal integer {label}"),
                vec![format!(
                    "Pass `{label}` as a constant int literal, not a variable or expression"
                )],
            ));
            None
        }
    }
}

pub(super) fn ir_builtin_axis_dim(
    list: &deep::List,
    type_env: &IrTypeEnv,
    tensor_arg_index: usize,
    axis_arg_index: usize,
) -> Option<DeepDimKind> {
    let tensor_dims = list
        .elements
        .get(3 + tensor_arg_index)
        .and_then(|expr| arg_tensor_type_expr(expr, type_env))
        .and_then(|ty| tensor_dims_from_type_expr(&ty))?;
    // Negative axes index from the end; normalize against the operand
    // rank so this concrete-extent check inspects the same axis the op
    // actually reduces.
    // Issue #216: cast-aware so a `cast(N, int32)`-wrapped axis arg
    // still resolves through to the operand's concrete dim.
    let raw_axis = list
        .elements
        .get(3 + axis_arg_index)
        .and_then(extract_int_for_dim)?;
    let axis = normalize_static_axis(tensor_dims.len(), raw_axis)?;
    tensor_dims.get(axis).copied()
}

/// Resolve the tensor type expression of a callsite argument, peeking
/// through a `(borrow {} <inner>)` wrapper if present.
///
/// Surf source idiomatically passes tensors to shape-sensitive IR
/// builtins via borrows (e.g. the `School.Nn.Conv.conv2d_small` sig
/// requires `&tensor[...]`). The validator's lookup helpers need to
/// see through that wrapper to find the underlying tensor type in the
/// IR type environment; otherwise the dim-concreteness checks in the
/// `conv2d`, `mean`, and `layer_norm` arms silently no-op on borrowed
/// inputs (see issue #186).
pub(super) fn arg_tensor_type_expr(expr: &deep::Expr, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    let inner = peel_borrow(expr);
    expr_type_expr(inner, type_env)
}

pub(super) fn peel_borrow(expr: &deep::Expr) -> &deep::Expr {
    // Recurses only through nested `borrow` wrappers (shallow in practice),
    // but guarded for uniformity; bail value is the identity input.
    stack_guard!("peel_borrow", expr, expr);
    if let deep::Expr::List(list, _) = expr
        && get_tag(list) == Some(DeepTag::Borrow)
        && let Some(child) = children(list).first()
    {
        return peel_borrow(child);
    }
    expr
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeepDimKind {
    Lit(i64),
    NonConcrete,
}

pub(super) fn tensor_dims_from_type_expr(expr: &deep::Expr) -> Option<Vec<DeepDimKind>> {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(list) == Some(DeepTag::TRef) {
        return children(list).first().and_then(tensor_dims_from_type_expr);
    }
    if get_tag(list) != Some(DeepTag::TTensor) {
        return None;
    }
    let kids = children(list);
    if kids.is_empty() {
        return None;
    }
    let mut dims = Vec::new();
    for kid in &kids[..kids.len().saturating_sub(1)] {
        dims.push(match kid {
            deep::Expr::List(dim_list, _) if get_tag(dim_list) == Some(DeepTag::DLit) => {
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

pub(super) fn type_expr_is_ir_concrete(expr: &deep::Expr) -> bool {
    match expr {
        deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::TPrim) => true,
        _ => tensor_dims_from_type_expr(expr)
            .map(|dims| dims.iter().all(|d| matches!(d, DeepDimKind::Lit(_))))
            .unwrap_or(false),
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// chelis#731 Phase 2 ([04-TOT-2] / spec/design/checker_totality.md §C4.1):
/// the ALWAYS-ON totality invariant. After a check completes with an EMPTY
/// error vector, the typed result must contain no error-typed node. With the
/// §C3 `ErrorWitness` token a silent `Type::Error` is unconstructible (every
/// `Type::Error` is minted by `report`, which pushes a diagnostic, or
/// `propagate`, which is downstream of one), so `errors.is_empty()`
/// structurally implies no fresh error was produced. This pass stays on as
/// the standing tripwire that verifies the claim -- it is a pushed internal
/// error, never a panic (the checker is reachable-input territory).
///
/// The shared finalizer checks both authoritative surfaces: every runtime
/// expression/pattern/function node in annotated Deep must carry the stamp
/// produced by its owning inference epoch, and the signature-inference table
/// must contain no structural `Type::Error`. The signature table remains the
/// value-level backstop; the annotated tree now has no re-inference gap.
pub(super) fn annotated_totality_invariant_traces(exprs: &[deep::Expr]) -> Vec<String> {
    fn walk(expr: &deep::Expr, traces: &mut Vec<String>) {
        match expr {
            deep::Expr::Atom(_, _) | deep::Expr::Map(_, _) => {}
            deep::Expr::MetaExpr(meta, _) => walk(&meta.expr, traces),
            deep::Expr::List(list, _) => {
                let tag = get_tag(list);
                let requires_stamp = tag.is_some_and(|tag| {
                    tag == DeepTag::Fn
                        || matches!(tag, DeepTag::PatVar | DeepTag::PatAs)
                        || should_attach_type_metadata(tag)
                });
                if requires_stamp
                    && !get_meta(list)
                        .is_some_and(|meta| meta.entries.iter().any(|(key, _)| key == "type"))
                {
                    traces.push(format!(
                        "annotated `{}` node is missing its type stamp",
                        tag.map(DeepTag::as_str).unwrap_or("<untagged-list>")
                    ));
                }

                let kids = children(list);
                for (index, child) in kids.iter().enumerate() {
                    // Decode-once: `child_stamp_role` is total over
                    // `DeepTag`, so the version-skew arm is
                    // unrepresentable; untagged structural lists take the
                    // recursive walk.
                    match tag.map(|tag| child_stamp_role(tag, index, kids.len())) {
                        Some(
                            ChildStampRole::RuntimeExpr | ChildStampRole::ExplicitInferenceBypass,
                        )
                        | None => walk(child, traces),
                        Some(
                            ChildStampRole::Syntax
                            | ChildStampRole::Selector
                            | ChildStampRole::EffectHandler
                            | ChildStampRole::Binder
                            | ChildStampRole::Type,
                        ) => {}
                    }
                }
            }
            // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
            deep::Expr::Node(node, span) => {
                let bridged = deep::Expr::List(node.to_list(*span), *span);
                walk(&bridged, traces);
            }
            deep::Expr::BareList(elems, _) => {
                for child in elems {
                    walk(child, traces);
                }
            }
            deep::Expr::UnknownForm(data) => {
                for child in &data.children {
                    walk(child, traces);
                }
            }
        }
    }

    let mut traces = Vec::new();
    for expr in exprs {
        walk(expr, &mut traces);
    }
    traces
}

pub(super) fn totality_invariant_traces(sig: &SignatureInferenceMetadata) -> Vec<String> {
    let mut out = Vec::new();
    for (name, f) in &sig.functions {
        if type_carries_error(&f.checked_signature) {
            out.push(format!(
                "function `{name}` checked_signature carries Type::Error"
            ));
        }
        if type_carries_error(&f.display_signature) {
            out.push(format!(
                "function `{name}` display_signature carries Type::Error"
            ));
        }
        for param in &f.params {
            if type_carries_error(&param.checked_type) || type_carries_error(&param.display_type) {
                out.push(format!(
                    "function `{name}` parameter #{} carries Type::Error",
                    param.index
                ));
            }
        }
    }
    out
}

/// True if `ty` is, or structurally contains, a `Type::Error`.
pub(super) fn type_carries_error(ty: &Type) -> bool {
    match ty {
        Type::Error(_) => true,
        Type::Fn(args, ret) => args.iter().any(type_carries_error) || type_carries_error(ret),
        Type::Ref(inner) => type_carries_error(inner),
        Type::Adt(_, args) => args.iter().any(type_carries_error),
        Type::Tuple(elems) => elems.iter().any(type_carries_error),
        Type::Prim(_) | Type::Tensor(_, _) | Type::Var(_) | Type::Unit => false,
    }
}

/// Build the internal diagnostic for a totality-invariant violation
/// ([04-TOT-2]). Only reachable if the §C3 witness token were bypassed (a
/// deserialization mint or a genuine bug); it names the offending nodes so a
/// regression is localizable rather than a bare "internal error".
pub(super) fn totality_violation_error(traces: &[String]) -> CheckError {
    CheckError::new(
        CheckErrorKind::Other,
        format!(
            "internal: [04-TOT-2] totality invariant violated -- the check reported \
             success (empty error vector) but the typed result carries {} silent \
             Type::Error verdict(s): {} (spec/design/checker_totality.md \u{00a7}C4.1; \
             chelis#731 Phase 2)",
            traces.len(),
            traces.join("; ")
        ),
        vec![],
    )
}
