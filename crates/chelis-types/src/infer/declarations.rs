//! Top-level declaration items, signature metadata, and parameter ownership.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

pub(super) fn top_level_decl_name(expr: &deep::Expr) -> Option<&str> {
    let (tag, _, kids) = stamped_parts(expr)?;
    if !matches!(
        tag,
        DeepTag::Def | DeepTag::Defsig | DeepTag::Deftype | DeepTag::Typealias
    ) {
        return None;
    }
    kids.first().and_then(symbol_name)
}

pub(super) fn top_level_decl_items(exprs: &[deep::Expr]) -> Vec<&deep::Expr> {
    fn push<'a>(expr: &'a deep::Expr, out: &mut Vec<&'a deep::Expr>) {
        if let Some((DeepTag::Module, _, kids)) = stamped_parts(expr) {
            // `(module {} name children...)` — skip the name child.
            for child in kids.iter().skip(1) {
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

/// Like [`top_level_decl_items`] but pairs each flattened item with
/// its enclosing lexical module key: nested `(module ...)` names
/// joined with `.`, `None` for items outside any wrapper. This is the
/// module-identity source for checker-enforced opacity (RFC D-CHECK);
/// reef package-linked items carry no wrapper and key through their
/// internal-name stem instead (see `opacity::module_key_for_item`).
pub(super) fn top_level_decl_items_with_modules(
    exprs: &[deep::Expr],
) -> Vec<(Option<String>, &deep::Expr)> {
    fn push<'a>(
        expr: &'a deep::Expr,
        prefix: Option<&str>,
        out: &mut Vec<(Option<String>, &'a deep::Expr)>,
    ) {
        if let Some((DeepTag::Module, _, kids)) = stamped_parts(expr) {
            // `(module {} name children...)` — skip the name child.
            let name = kids.first().and_then(symbol_name);
            let key = match (prefix, name) {
                (Some(p), Some(n)) => Some(format!("{p}.{n}")),
                (None, Some(n)) => Some(n.to_string()),
                (p, None) => p.map(str::to_string),
            };
            for child in kids.iter().skip(1) {
                push(child, key.as_deref(), out);
            }
            return;
        }
        out.push((prefix.map(str::to_string), expr));
    }
    let mut out = Vec::new();
    for expr in exprs {
        push(expr, None, &mut out);
    }
    out
}

/// RFC v4b (RT-1 F2): a named module may be opened by at most one
/// `(module ...)` wrapper per check unit. Module identity is otherwise
/// a forgeable string -- a second wrapper of an opaque type's defining
/// module would construct and inspect the type as if it were inside.
/// Walks every wrapper (including nested ones, keyed by their full
/// `.`-joined path) and emits ONE `DuplicateModule` error per
/// re-opened name. Surf emits one module per file and reef strips
/// wrappers before inference, so this only fires on hand-written `.dp`
/// (the forge surface).
pub(super) fn detect_module_reopens(exprs: &[deep::Expr], errors: &mut DiagnosticSink<'_>) {
    fn walk(
        expr: &deep::Expr,
        prefix: Option<&str>,
        seen: &mut UnordSet<String>,
        reported: &mut UnordSet<String>,
        errors: &mut DiagnosticSink<'_>,
    ) {
        // chelis#1107: carrier-preserving read. A `List`-only destructure
        // returned on every stamped `module`, so the reopen check never ran
        // on `check_typed_program`.
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            return;
        };
        if tag != DeepTag::Module {
            return;
        }
        let name = kids.first().and_then(symbol_name);
        let key = match (prefix, name) {
            (Some(p), Some(n)) => Some(format!("{p}.{n}")),
            (None, Some(n)) => Some(n.to_string()),
            (p, None) => p.map(str::to_string),
        };
        if let Some(key) = &key
            && !seen.insert(key.clone())
            && reported.insert(key.clone())
        {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateModule,
                format!(
                    "module `{key}` is opened by more than one module wrapper in this \
                     check unit; a named module may be opened at most once"
                ),
                vec![format!(
                    "merge the `{key}` wrappers into one, or rename one of them"
                )],
            ));
        }
        for child in kids.iter().skip(1) {
            walk(child, key.as_deref(), seen, reported, errors);
        }
    }
    let mut seen = UnordSet::new();
    let mut reported = UnordSet::new();
    for expr in exprs {
        walk(expr, None, &mut seen, &mut reported, errors);
    }

    // RFC v5 belt-and-suspenders (RT-1 F2 bypass): a stem-derived
    // module identity (from a top-level mangled `deftype`/`def` name)
    // that collides with a lexical wrapper key in the same check unit
    // is also a `DuplicateModule` error. Genuine linker output has NO
    // lexical wrappers, so this never fires on it; it defends the
    // stem-plus-wrapper forge shapes even if the name-format check is
    // somehow bypassed. Runs unconditionally (structural).
    for (lexical, expr) in top_level_decl_items_with_modules(exprs) {
        // Only flat (non-wrapped) mangled declarations introduce a
        // stem-derived module identity; a name inside a lexical
        // wrapper keys to the wrapper, not its stem.
        if lexical.is_some() {
            continue;
        }
        // chelis#1107: carrier-preserving read, as in `walk` above.
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if !matches!(tag, DeepTag::Deftype | DeepTag::Def) {
            continue;
        }
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(stem_key) = crate::opacity::reef_module_stem(name) else {
            continue;
        };
        if seen.contains(&stem_key) && reported.insert(stem_key.clone()) {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateModule,
                format!(
                    "module `{stem_key}` is opened by both a lexical wrapper and a \
                     reef-stem mangled name in this check unit; a named module may be \
                     opened at most once"
                ),
                vec![format!(
                    "rename the mangled declaration; the `{stem_key}` lexical module \
                     already exists"
                )],
            ));
        }
    }
}

/// RFC v5 (RT-1 F2 bypass): the reef package-linker's internal-name
/// format (`Pkg__<pkg>__<Module>__<Name>` / lowercase twin) is the
/// linker's PRIVATE output. A program NOT produced by the linker
/// (raw `.ch` or raw `.dp`) that uses it forges module identity
/// through the reef-stem channel, so any top-level declaration whose
/// binding name matches the format is a declaration error. Skipped
/// entirely when the linked-program provenance flag is set (the
/// linker's own output is accepted). The linker also re-mangles every
/// user source name, so user code inside a real package cannot smuggle
/// a clean mangled name into linked output.
pub(super) fn detect_forged_linker_names(exprs: &[deep::Expr], errors: &mut DiagnosticSink<'_>) {
    if crate::opacity::linked_program() {
        return;
    }
    for expr in top_level_decl_items(exprs) {
        // chelis#1107: carrier-preserving read. A `List`-only destructure
        // skipped every stamped declaration, so this forgery guard ran only
        // on `check_ir_program`.
        //
        // `defmacro` is compiler-internal pre-expansion syntax outside the
        // vocabulary; it stays symbol-headed (raw-string boundary), so it
        // never decodes and is matched on its head string instead.
        let (kids, is_declaration) = match stamped_parts(expr) {
            Some((tag, _, kids)) => (
                kids,
                matches!(
                    tag,
                    DeepTag::Deftype | DeepTag::Def | DeepTag::Defsig | DeepTag::Typealias
                ),
            ),
            None => match expr {
                deep::Expr::UnknownForm(data) => {
                    (data.children.as_slice(), data.head == "defmacro")
                }
                _ => continue,
            },
        };
        if !is_declaration {
            continue;
        }
        if let Some(name) = kids.first().and_then(symbol_name)
            && crate::opacity::is_linker_format_name(name)
        {
            errors.push(crate::opacity::forged_linker_name_error(name));
        }
    }
}

pub(super) fn infer_signature_metadata_with_context_and_headers(
    exprs: &[deep::Expr],
    function_plan: &FunctionInferencePlan,
    type_env: &BTreeMap<String, deep::Expr>,
    signature_context: &SignatureInferenceMetadata,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> SignatureInferenceMetadata {
    let defsig_names = collect_defsig_names(exprs);
    let authored_signature_types = collect_authored_signature_types(exprs, type_headers, errors);
    let recursive_members = function_plan.recursive_member_names();
    let mut functions = BTreeMap::new();
    let mut defs_by_name = UnordMap::<String, VecDeque<&deep::Expr>>::new();
    for expr in top_level_decl_items(exprs) {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        if kids
            .get(1)
            .and_then(|body| tagged_children(body, DeepTag::Fn))
            .is_none()
        {
            continue;
        }
        defs_by_name
            .entry(name.to_string())
            .or_default()
            .push_back(expr);
    }
    let ordered_defs = function_plan
        .ordered_members()
        .filter_map(|member| defs_by_name.get_mut(&member.name)?.pop_front())
        .collect::<Vec<_>>();
    let passes = ordered_defs.len().max(1);
    let imported_signatures = signature_context
        .functions
        .iter()
        .map(|(name, inference)| (name.clone(), inference.display_signature.clone()))
        .collect::<UnordMap<_, _>>();

    // chelis#930: cooperative cancellation at declaration granularity. This
    // fixed point runs one full sweep of every def per def (`passes` is the
    // def count), which makes it the front end's other declaration-count-
    // scaling pass — and on a large program the single most expensive one.
    // Polling the inner loop rather than the outer sweep keeps the bound
    // independent of program size: one declaration, not one O(n) sweep.
    let cancel = crate::cancel::current_cancel_token();
    'fixed_point: for _ in 0..passes {
        functions.clear();
        let mut available_signatures = imported_signatures.clone();
        for expr in &ordered_defs {
            if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
                break 'fixed_point;
            }
            let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
                continue;
            };
            let Some(name) = kids.first().and_then(symbol_name) else {
                continue;
            };
            let Some(fn_kids) = kids
                .get(1)
                .and_then(|body| tagged_children(body, DeepTag::Fn))
            else {
                continue;
            };
            let Some(checked_signature) = type_env
                .get(name)
                .and_then(|expr| type_from_deep_expr(expr, type_headers, errors))
            else {
                continue;
            };
            let Type::Fn(checked_args, checked_ret) = checked_signature.clone() else {
                continue;
            };
            let Some(params_expr) = fn_kids.first() else {
                continue;
            };
            let Some(body) = fn_kids.get(1) else {
                continue;
            };
            let param_infos = param_source_infos(params_expr);
            let recursive_cycle = recursive_members.contains(name);
            let all_written_by_defsig =
                defsig_names.contains(name) && param_infos.iter().all(|(_, written)| !*written);
            let mut display_args = checked_args.clone();
            let mut params = Vec::new();

            for (index, (pname, param_written)) in param_infos.iter().enumerate() {
                let Some(checked_type) = checked_args.get(index).cloned() else {
                    continue;
                };
                let written = all_written_by_defsig || *param_written;
                // [04-LIN-9]: a key holder is never borrowed, so a parameter
                // that carries a key is never inferred read-only.
                let can_infer = !recursive_cycle
                    && !written
                    && type_contains_tensor(&checked_type)
                    && !type_mentions_key(&checked_type)
                    && !matches!(checked_type, Type::Ref(_));
                let inferred_read_only = can_infer
                    && !param_has_consuming_use_with_headers(
                        body,
                        pname,
                        &available_signatures,
                        type_env,
                        type_headers,
                        errors,
                    );
                let display_type = if inferred_read_only {
                    Type::Ref(Box::new(checked_type.clone()))
                } else {
                    checked_type.clone()
                };
                if let Some(slot) = display_args.get_mut(index) {
                    *slot = display_type.clone();
                }
                params.push(ParamSignatureInference {
                    index,
                    name: pname.clone(),
                    written,
                    inferred_read_only,
                    checked_type,
                    display_type,
                });
            }

            let display_signature = Type::Fn(display_args, checked_ret);
            available_signatures.insert(name.to_string(), display_signature.clone());
            functions.insert(
                name.to_string(),
                FunctionSignatureInference {
                    name: name.to_string(),
                    authored_signature: defsig_names.contains(name),
                    authored_signature_type: authored_signature_types.get(name).cloned(),
                    recursive_cycle,
                    checked_signature,
                    display_signature,
                    params,
                },
            );
        }
    }

    SignatureInferenceMetadata { functions }
}

pub(super) fn collect_defsig_names(exprs: &[deep::Expr]) -> UnordSet<String> {
    let mut names = UnordSet::new();
    for expr in top_level_decl_items(exprs) {
        if let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr)
            && let Some(name) = kids.first().and_then(symbol_name)
        {
            names.insert(name.to_string());
        }
    }
    names
}

pub(super) fn collect_authored_signature_types(
    exprs: &[deep::Expr],
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> UnordMap<String, Type> {
    let mut signatures = UnordMap::new();
    for expr in top_level_decl_items(exprs) {
        let Some((DeepTag::Defsig, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some((name_expr, binder_list, signature_expr)) = defsig_parts(kids) else {
            continue;
        };
        let Some(name) = symbol_name(name_expr) else {
            continue;
        };
        let Some(binders) = defsig_binder_names(binder_list, errors) else {
            continue;
        };
        let mut vg = VarGen::default();
        let signature = DeepTypeResolver::new(
            TypeUseSite::Defsig,
            BinderMode::ExplicitGeneric(&binders),
            type_headers,
            &mut vg,
            errors,
        )
        .resolve(signature_expr)
        .ok()
        .map(|resolved| resolved.into_type());
        if let Some(signature) = signature {
            signatures.insert(name.to_string(), signature);
        }
    }
    signatures
}

pub(crate) fn param_has_consuming_use(
    expr: &deep::Expr,
    param: &str,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
) -> Result<bool, InferResult> {
    crate::session::param_has_consuming_use(
        expr,
        param,
        available_signatures,
        type_env,
        type_headers,
    )
}

pub(crate) fn param_has_consuming_use_in_session(
    expr: &deep::Expr,
    param: &str,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    param_has_consuming_use_with_headers(
        expr,
        param,
        available_signatures,
        type_env,
        type_headers,
        errors,
    )
}

pub(super) fn param_has_consuming_use_with_headers(
    expr: &deep::Expr,
    param: &str,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let mut bound = Vec::new();
    param_has_consuming_use_inner(
        expr,
        param,
        &mut bound,
        available_signatures,
        type_env,
        type_headers,
        errors,
    )
}

pub(super) fn param_has_consuming_use_inner(
    expr: &deep::Expr,
    param: &str,
    bound: &mut Vec<UnordSet<String>>,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    stack_guard!("param_has_consuming_use_inner", expr, false);
    match expr {
        deep::Expr::Atom(_, _) => false,
        deep::Expr::Map(map, _) => map.any_syntax(&mut |value| {
            param_has_consuming_use_inner(
                value,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            )
        }),
        deep::Expr::MetaExpr(meta, _) => param_has_consuming_use_inner(
            &meta.expr,
            param,
            bound,
            available_signatures,
            type_env,
            type_headers,
            errors,
        ),
        deep::Expr::Node(node, _) => match node.tag() {
            DeepTag::Var => var_name_node(node) == Some(param) && !is_bound_name(param, bound),
            DeepTag::Borrow | DeepTag::Copy => node.children_slice().first().is_some_and(|child| {
                param_nested_consuming_use(
                    child,
                    param,
                    bound,
                    available_signatures,
                    type_env,
                    type_headers,
                    errors,
                )
            }),
            DeepTag::Realize => node
                .children_slice()
                .first()
                .is_some_and(|child| expr_mentions_unshadowed_name(child, param, bound)),
            DeepTag::App => app_consumes_param(
                node,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            ),
            DeepTag::Fn => {
                let kids = node.children_slice();
                if kids.len() < 2 {
                    return false;
                }
                if expr_mentions_unshadowed_name(&kids[1], param, bound) {
                    return true;
                }
                false
            }
            DeepTag::Let => {
                let kids = node.children_slice();
                if kids.len() < 2 {
                    return false;
                }
                let mut let_names = UnordSet::new();
                if let Some(bind_kids) = kids
                    .first()
                    .and_then(|bind| tagged_children(bind, DeepTag::Bind))
                {
                    let mut index = 0;
                    while index + 1 < bind_kids.len() {
                        if param_has_consuming_use_inner(
                            &bind_kids[index + 1],
                            param,
                            bound,
                            available_signatures,
                            type_env,
                            type_headers,
                            errors,
                        ) {
                            return true;
                        }
                        if let Some(name) = symbol_name(&bind_kids[index]) {
                            let_names.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                bound.push(let_names);
                let result = param_has_consuming_use_inner(
                    &kids[1],
                    param,
                    bound,
                    available_signatures,
                    type_env,
                    type_headers,
                    errors,
                );
                bound.pop();
                result
            }
            DeepTag::Match => {
                let kids = node.children_slice();
                if kids
                    .first()
                    .is_some_and(|scrutinee| expr_mentions_unshadowed_name(scrutinee, param, bound))
                {
                    return true;
                }
                for arm in kids.iter().skip(1) {
                    let Some(arm_kids) = tagged_children(arm, DeepTag::Arm) else {
                        continue;
                    };
                    if arm_kids.len() < 3 {
                        continue;
                    }
                    bound.push(pattern_names_for_signature(&arm_kids[0]));
                    let consumes = param_has_consuming_use_inner(
                        &arm_kids[1],
                        param,
                        bound,
                        available_signatures,
                        type_env,
                        type_headers,
                        errors,
                    ) || param_has_consuming_use_inner(
                        &arm_kids[2],
                        param,
                        bound,
                        available_signatures,
                        type_env,
                        type_headers,
                        errors,
                    );
                    bound.pop();
                    if consumes {
                        return true;
                    }
                }
                false
            }
            _ => node.children_slice().iter().any(|child| {
                param_has_consuming_use_inner(
                    child,
                    param,
                    bound,
                    available_signatures,
                    type_env,
                    type_headers,
                    errors,
                )
            }),
        },
        deep::Expr::BareList(elems, _) => elems.iter().any(|child| {
            param_has_consuming_use_inner(
                child,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            )
        }),
        deep::Expr::UnknownForm(data) => data.children.iter().any(|child| {
            param_has_consuming_use_inner(
                child,
                param,
                bound,
                available_signatures,
                type_env,
                type_headers,
                errors,
            )
        }),
    }
}

pub(super) fn param_nested_consuming_use(
    expr: &deep::Expr,
    param: &str,
    bound: &mut Vec<UnordSet<String>>,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    if is_direct_unshadowed_var(expr, param, bound) {
        return false;
    }
    param_has_consuming_use_inner(
        expr,
        param,
        bound,
        available_signatures,
        type_env,
        type_headers,
        errors,
    )
}

pub(super) fn app_consumes_param(
    node: &DeepNode,
    param: &str,
    bound: &mut Vec<UnordSet<String>>,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let kids = node.children_slice();
    let callee = kids.first().and_then(var_name_expr);
    if let Some(func) = kids.first()
        && !matches!(callee, Some(name) if name != param)
        && param_has_consuming_use_inner(
            func,
            param,
            bound,
            available_signatures,
            type_env,
            type_headers,
            errors,
        )
    {
        return true;
    }
    for (index, arg) in kids.iter().skip(1).enumerate() {
        if borrow_inner_for_signature(arg)
            .is_some_and(|inner| is_direct_unshadowed_var(inner, param, bound))
        {
            continue;
        }
        if is_direct_unshadowed_var(arg, param, bound) {
            if callee_arg_is_borrowed(
                callee,
                index,
                available_signatures,
                type_env,
                type_headers,
                errors,
            ) {
                continue;
            }
            return true;
        }
        if param_has_consuming_use_inner(
            arg,
            param,
            bound,
            available_signatures,
            type_env,
            type_headers,
            errors,
        ) {
            return true;
        }
    }
    false
}

pub(super) fn callee_arg_is_borrowed(
    callee: Option<&str>,
    index: usize,
    available_signatures: &UnordMap<String, Type>,
    type_env: &BTreeMap<String, deep::Expr>,
    type_headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let Some(callee) = callee else {
        return false;
    };
    if let Some(Type::Fn(args, _)) = available_signatures.get(callee)
        && args.get(index).is_some_and(|ty| matches!(ty, Type::Ref(_)))
    {
        return true;
    }
    if let Some(Type::Fn(args, _)) = type_env
        .get(callee)
        .and_then(|expr| type_from_deep_expr(expr, type_headers, errors))
        && args.get(index).is_some_and(|ty| matches!(ty, Type::Ref(_)))
    {
        return true;
    }
    builtin_arg_is_ref(callee, index)
}

pub(super) fn builtin_arg_is_ref(name: &str, index: usize) -> bool {
    let (env, _) = builtins::builtin_env();
    if let Some(Type::Fn(args, _)) = env.lookup(name).map(|scheme| &scheme.body) {
        return args.get(index).is_some_and(|ty| matches!(ty, Type::Ref(_)));
    }
    false
}

pub(super) fn expr_mentions_unshadowed_name(
    expr: &deep::Expr,
    name: &str,
    bound: &mut Vec<UnordSet<String>>,
) -> bool {
    stack_guard!("expr_mentions_unshadowed_name", expr, false);
    match expr {
        deep::Expr::Atom(_, _) => false,
        deep::Expr::Map(map, _) => {
            map.any_syntax(&mut |value| expr_mentions_unshadowed_name(value, name, bound))
        }
        deep::Expr::MetaExpr(meta, _) => expr_mentions_unshadowed_name(&meta.expr, name, bound),
        deep::Expr::Node(node, _) => match node.tag() {
            DeepTag::Var => var_name_node(node) == Some(name) && !is_bound_name(name, bound),
            DeepTag::Fn => {
                let kids = node.children_slice();
                if kids.len() < 2 {
                    return false;
                }
                bound.push(
                    param_source_infos(&kids[0])
                        .into_iter()
                        .map(|(n, _)| n)
                        .collect(),
                );
                let result = expr_mentions_unshadowed_name(&kids[1], name, bound);
                bound.pop();
                result
            }
            _ => node
                .children_slice()
                .iter()
                .any(|child| expr_mentions_unshadowed_name(child, name, bound)),
        },
        deep::Expr::BareList(elems, _) => elems
            .iter()
            .any(|child| expr_mentions_unshadowed_name(child, name, bound)),
        deep::Expr::UnknownForm(data) => data
            .children
            .iter()
            .any(|child| expr_mentions_unshadowed_name(child, name, bound)),
    }
}

pub(super) fn type_from_deep_expr(
    expr: &deep::Expr,
    headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type> {
    let mut vg = VarGen::default();
    DeepTypeResolver::new(
        TypeUseSite::CompilerMetadata,
        BinderMode::TrustedCompilerMetadata,
        headers,
        &mut vg,
        errors,
    )
    .resolve(expr)
    .ok()
    .map(|ty| ty.into_type())
}

pub(super) fn type_contains_tensor(ty: &Type) -> bool {
    match ty {
        Type::Tensor(_, _) => true,
        Type::Ref(inner) => type_contains_tensor(inner),
        Type::Adt(_, args) | Type::Tuple(args) => args.iter().any(type_contains_tensor),
        Type::KindedAdt(_, args) => args
            .iter()
            .any(|argument| argument.as_type().is_some_and(type_contains_tensor)),
        Type::Fn(_, _) | Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error(_) => false,
    }
}

/// [04-LIN-9]: does `ty` mention the `key` dtype, as a scalar, a tensor
/// element, or inside a tuple, reference, or data type argument?
pub(super) fn type_mentions_key(ty: &Type) -> bool {
    match ty {
        Type::Prim(prim) => *prim == Prim::Key,
        Type::Tensor(_, precision) => matches!(precision, TensorPrec::Concrete(Prim::Key)),
        Type::Ref(inner) => type_mentions_key(inner),
        Type::Adt(_, args) | Type::Tuple(args) => args.iter().any(type_mentions_key),
        Type::KindedAdt(_, args) => args
            .iter()
            .any(|argument| argument.as_type().is_some_and(type_mentions_key)),
        Type::Fn(_, _) | Type::Var(_) | Type::Unit | Type::Error(_) => false,
    }
}

/// Issue #256 round 3: does `ty` carry a tensor, consulting `carriers`
/// for the by-name ADT carry decision? This is the `Type`-level mirror
/// of linearity's `type_expr_contains_tensor`: an ADT carries iff its
/// name is in the precomputed carrier set (its definition has a
/// tensor-carrying field) OR one of its type arguments carries (e.g.
/// `Wrapper[tensor[..]]`). Bare `type_contains_tensor` cannot make the
/// by-name decision — it only sees the `Type::Adt` shell, not the
/// variant fields — which is exactly why the deferred-borrow gate must
/// be handed the carrier set rather than trust an args-only check.
pub(super) fn type_carries_tensor_with_carriers(ty: &Type, carriers: &UnordSet<String>) -> bool {
    match ty {
        Type::Tensor(_, _) => true,
        Type::Ref(inner) => type_carries_tensor_with_carriers(inner, carriers),
        Type::Tuple(args) => args
            .iter()
            .any(|a| type_carries_tensor_with_carriers(a, carriers)),
        Type::Adt(name, args) => {
            carriers.contains(name)
                || args
                    .iter()
                    .any(|a| type_carries_tensor_with_carriers(a, carriers))
        }
        Type::KindedAdt(name, args) => {
            carriers.contains(name)
                || args.iter().any(|argument| {
                    argument
                        .as_type()
                        .is_some_and(|ty| type_carries_tensor_with_carriers(ty, carriers))
                })
        }
        Type::Fn(_, _) | Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error(_) => false,
    }
}

/// Issue #256 round 3: compute the set of tensor-carrying ADT names from
/// the registry. This is the registry-backed mirror of linearity's
/// `compute_tensor_carrying_adts` (which works off stamped Deep exprs):
/// fixed-point iteration where an ADT joins the carrier set once any of
/// its variant fields carries a tensor against the in-progress set, so a
/// chain `A { f: B }, B { g: tensor }` resolves transitively. Bounded by
/// the ADT count. The two classifiers must agree: the gate uses this set
/// to reject a deferred borrow that resolved to a non-carrying ADT, and
/// linearity uses its own set to reject the concrete (non-deferred) form.
pub(super) fn adt_carrier_set(adt_reg: &AdtRegistry) -> UnordSet<String> {
    let mut carriers: UnordSet<String> = UnordSet::new();
    loop {
        let mut grew = false;
        for (name, def) in &adt_reg.defs {
            if carriers.contains(name) {
                continue;
            }
            let carries = def.variants.iter().any(|variant| {
                variant
                    .fields
                    .iter()
                    .any(|(_, field_ty)| type_carries_tensor_with_carriers(field_ty, &carriers))
            });
            if carries {
                carriers.insert(name.clone());
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    carriers
}

/// Issue #256 round 2 soundness gate. The `borrow` inference arm accepts a
/// borrow whose inner type is still an unresolved `Type::Var`, recording
/// the variable in the substitution's deferred-borrow ledger. That
/// deferral is sound only when the variable is *eventually* pinned to a
/// tensor or tensor-carrying type by a later unification (the surrounding
/// `&tensor[..]` / `&Carrier[..]` parameter). This pass drains the ledger
/// after a def body's inference completes and re-checks each recorded
/// variable against the now-complete substitution:
///
///   - `Tensor` / `Ref(Tensor)`: pinned to a tensor — sound, accept.
///   - `Adt` / `Tuple` / `Ref(Adt|Tuple)`: an aggregate that *may* carry a
///     tensor. Round 3 (#256 soundness): classify it here against the
///     registry-backed carrier set rather than blanket-accepting and
///     deferring to linearity. Deferring was unsound — round 1 loosened
///     linearity's `expr_is_owned_or_borrow_linear` to accept a stale
///     `(t-var ..)` stamp (so a tensor that resolved late is not
///     rejected), and a deferred borrow that resolves to a *non*-carrying
///     ADT/tuple keeps that same `(t-var ..)` stamp at the linearity
///     layer. Both gates would then wave it through. So the gate, which
///     already holds the final `Type`, must make the carry decision: a
///     tensor-carrying aggregate is accepted, a non-carrying one rejected.
///   - still `Var`: never pinned. A fully-polymorphic consumer (e.g.
///     `consume_any[a](t: a)`) unifies the parameter to `&a` without ever
///     forcing a tensor, so a genuinely-non-tensor value would slip past
///     every other gate. Reject.
///   - `Prim` / `Unit` / `Fn`: pinned to a concretely-non-tensor scalar
///     only after the borrow arm ran (so the arm's own `_ => TypeMismatch`
///     could not fire). Reject.
pub(super) fn validate_deferred_borrow_vars(
    subst: &Subst,
    adt_reg: &AdtRegistry,
    declared_type_names: &UnordMap<TypeVar, String>,
    errors: &mut DiagnosticSink<'_>,
) {
    let deferred = subst.take_deferred_borrow_vars();
    if deferred.is_empty() {
        return;
    }
    // Computed lazily: only programs that actually deferred a borrow pay
    // the fixed-point pass, and only once per drain.
    let carriers = adt_carrier_set(adt_reg);
    for tv in deferred {
        let resolved = subst.apply(&Type::Var(tv));
        // Peel every `Ref` layer: the recorded variable is the borrow
        // inner, but a later unification may have wrapped it in one or
        // more `&` layers (e.g. the parameter type was itself `&T`).
        let mut peeled = &resolved;
        while let Type::Ref(inner) = peeled {
            peeled = inner.as_ref();
        }
        let sound = match peeled {
            // Pinned to a tensor: always sound.
            Type::Tensor(_, _) => true,
            // Pinned to an aggregate: sound iff it actually carries a
            // tensor against the registry carrier set (round 3). A
            // non-carrying record/tuple resolved through the deferred
            // path is rejected here — linearity's loosened classifier
            // can no longer be relied on to catch it.
            Type::Adt(_, _) | Type::KindedAdt(_, _) | Type::Tuple(_) => {
                type_carries_tensor_with_carriers(peeled, &carriers)
            }
            // Don't double-report an inner that already failed inference.
            Type::Error(_) => true,
            // Never pinned, or pinned to a concretely-non-tensor value.
            Type::Var(_) | Type::Prim(_) | Type::Unit | Type::Fn(_, _) => false,
            // `Ref` is fully peeled above; treat as sound to avoid a
            // spurious reject if a future shape reaches here.
            Type::Ref(_) => true,
        };
        if !sound {
            // chelis#260 Site 2 / spec/04 [04-FIT-9]: name the source type
            // parameter when this signature declared one.
            //
            // The lookup is on the RESOLVED variable, not the deferred one.
            // Measured on `def go[t](x: t)`: the resolver mints `t` as ?343,
            // the instantiation for the body renames it to ?344, the borrow
            // site defers a later variable ?345, and ?345 resolves to ?344.
            // The recorded map is keyed by what the instantiation minted, so
            // ?344 is the key that carries the name -- and ?344 is also what
            // this diagnostic prints, which is the coincidence that makes the
            // rendering correct rather than merely adjacent.
            //
            // [04-FIT-10]: with no recorded name the internal identity still
            // renders, because a variable with no source provenance must not
            // be given an invented one.
            let subject = match peeled {
                Type::Var(resolved_var) => match declared_type_names.get(resolved_var) {
                    Some(name) => format!("`{name}`"),
                    None => format!("{peeled}"),
                },
                _ => format!("{peeled}"),
            };
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("borrow requires tensor or tensor-carrying input, got {subject}"),
                vec!["Use `&x` only with tensor values".to_string()],
            ));
        }
    }
}

/// RFC D-CHECK: drain the deferred-access ledger after a def body's
/// inference completes and re-check each recorded target variable
/// against the final substitution: a target pinned to an
/// out-of-module opaque ADT (e.g. an unannotated lambda parameter
/// pinned by a later call) is rejected with the same action text as
/// the typed path. Draining per def keeps attribution exact and
/// prevents one def's deferrals from leaking into the next,
/// mirroring `validate_deferred_borrow_vars`.
pub(super) fn validate_deferred_opaque_uses(
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    let mut any_opaque_in_scope: Option<bool> = None;
    for (tv, use_kind) in subst.take_deferred_opaque_uses() {
        let resolved = subst.apply(&Type::Var(tv));
        let mut peeled = &resolved;
        while let Type::Ref(inner) = peeled {
            peeled = inner.as_ref();
        }
        let action = match use_kind {
            crate::unify::DeferredOpaqueUse::Access => crate::opacity::OpaqueAction::FieldAccess,
            crate::unify::DeferredOpaqueUse::RecordUpdate => {
                crate::opacity::OpaqueAction::RecordUpdate
            }
        };
        match peeled {
            Type::Adt(adt_name, _) | Type::KindedAdt(adt_name, _) => {
                crate::opacity::check_opaque_use(action, adt_name, adt_reg, errors);
            }
            // Never pinned: let-generalization makes an unannotated
            // accessor polymorphic, so callers instantiate FRESH
            // variables and the recorded one stays unbound -- a
            // laundering channel for opaque values. Mirror the
            // deferred-borrow ledger's never-pinned rejection,
            // fail-closed, scoped to check units that declare any
            // opaque type so opaque-free programs keep the lenient
            // status quo.
            Type::Var(_) => {
                let opaque_in_scope = *any_opaque_in_scope
                    .get_or_insert_with(|| adt_reg.defs.values().any(|def| def.opaque));
                if opaque_in_scope
                    && let Some(error) = crate::opacity::with_context(|ctx| {
                        crate::opacity::unresolved_target_error(ctx, action)
                    })
                {
                    errors.push(error);
                }
            }
            _ => {}
        }
    }
}

pub(super) fn param_source_infos(expr: &deep::Expr) -> Vec<(String, bool)> {
    let params = match expr {
        deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => node.children_slice(),
        deep::Expr::BareList(elements, _) => elements.as_slice(),
        _ => return Vec::new(),
    };
    params
        .iter()
        .filter_map(|param| match param {
            deep::Expr::Atom(deep::Atom::Name(name), _) => Some((name.clone(), false)),
            deep::Expr::MetaExpr(meta, _) => {
                let deep::Expr::Atom(deep::Atom::Name(name), _) = meta.expr.as_ref() else {
                    return None;
                };
                Some((name.clone(), meta.metadata.ty().is_some()))
            }
            deep::Expr::BareList(elements, _) => {
                let name = elements.first().and_then(symbol_name)?;
                let written = elements.get(1).is_some_and(|expr| {
                    let deep::Expr::Map(meta, _) = expr else {
                        return false;
                    };
                    meta.ty().is_some()
                });
                Some((name.to_string(), written))
            }
            _ => None,
        })
        .collect()
}

pub(super) fn pattern_names_for_signature(expr: &deep::Expr) -> UnordSet<String> {
    let mut names = UnordSet::new();
    collect_pattern_names_for_signature(expr, &mut names);
    names
}

pub(super) fn collect_pattern_names_for_signature(expr: &deep::Expr, names: &mut UnordSet<String>) {
    // Bail before unbounded recursion exhausts the native stack on a
    // deeply-nested pattern. No error vector here; the guard records the bail
    // so the check entry boundary fails hard with a located diagnostic. See
    // `STACK_RED_ZONE_BYTES`.
    stack_guard!("collect_pattern_names_for_signature", expr);
    // chelis#1107: carrier-preserving read; a `List`-only destructure collected
    // no pattern binder at all from a stamped `match`.
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                names.insert(name.to_string());
            }
        }
        DeepTag::PatAs => {
            if let Some(name) = kids.first().and_then(symbol_name) {
                names.insert(name.to_string());
            }
            if let Some(inner) = kids.get(1) {
                collect_pattern_names_for_signature(inner, names);
            }
        }
        _ => {
            for child in kids {
                collect_pattern_names_for_signature(child, names);
            }
        }
    }
}

pub(super) fn tagged_children(expr: &deep::Expr, tag: DeepTag) -> Option<&[deep::Expr]> {
    stamped_parts(expr).and_then(|(found, _, children)| (found == tag).then_some(children))
}

pub(super) fn var_name_expr(expr: &deep::Expr) -> Option<&str> {
    tagged_children(expr, DeepTag::Var)?
        .first()
        .and_then(symbol_name)
}

pub(super) fn var_name_node(node: &DeepNode) -> Option<&str> {
    if node.tag() != DeepTag::Var {
        return None;
    }
    node.children_slice().first().and_then(symbol_name)
}

pub(super) fn borrow_inner_for_signature(expr: &deep::Expr) -> Option<&deep::Expr> {
    // chelis#1107 amendment: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag != DeepTag::Borrow {
        return None;
    }
    kids.first()
}

pub(super) fn is_direct_unshadowed_var(
    expr: &deep::Expr,
    name: &str,
    bound: &[UnordSet<String>],
) -> bool {
    var_name_expr(expr) == Some(name) && !is_bound_name(name, bound)
}

pub(super) fn is_bound_name(name: &str, bound: &[UnordSet<String>]) -> bool {
    bound.iter().rev().any(|scope| scope.contains(name))
}

/// Check whether a def body carries its own explicit type stamp and is a
/// literal self-reference, as produced by `x = (x : T)`. A declaration-level
/// `x: T = x` is recognized separately by the cycle detector because Surf
/// represents its type as a sibling `defsig`, not as body metadata.
pub(super) fn body_is_type_stamped_literal_self_ref(body: &deep::Expr, name: &str) -> bool {
    if expr_type_expr(body, &IrTypeEnv::new()).is_none() {
        return false;
    }
    body_is_literal_self_ref_shape(body, name)
}

/// Check the literal self-reference shape independently of its explicit type
/// owner. Callers must first prove either a body type stamp or the matching
/// declaration signature; bare `x = x` must never earn this carve-out.
pub(super) fn body_is_literal_self_ref_shape(body: &deep::Expr, name: &str) -> bool {
    let mut current = body;
    loop {
        match current {
            deep::Expr::MetaExpr(meta, _) => current = &meta.expr,
            deep::Expr::Node(node, _) => {
                return node.tag() == DeepTag::Var
                    && node.children_slice().first().and_then(symbol_name) == Some(name);
            }
            _ => return false,
        }
    }
}

pub(super) fn param_name_for_refs(param: &deep::Expr) -> Option<String> {
    stack_guard!("param_name_for_refs", param, None);
    match param {
        deep::Expr::Atom(deep::Atom::Name(name), _) => Some(name.clone()),
        deep::Expr::Atom(_, _) => None,
        deep::Expr::MetaExpr(meta, _) => param_name_for_refs(&meta.expr),
        // A Deep param `(name {type: ...})` is a structural list with the
        // name as the FIRST element and the meta map as the second.
        deep::Expr::BareList(elements, _) => {
            elements.first().and_then(symbol_name).map(str::to_string)
        }
        deep::Expr::Node(_, _) | deep::Expr::Map(_, _) | deep::Expr::UnknownForm(_) => None,
    }
}
