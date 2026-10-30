//! Checker-boundary validation for `grad` selector identity.

use super::*;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Deserialize, serde::Serialize)
)]
pub(crate) struct SelectorCallableId {
    namespace: [u8; 32],
    ordinal: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Deserialize, serde::Serialize)
)]
pub(crate) enum SelectorCallableOrigin {
    Known {
        identity: SelectorCallableId,
        params: Vec<String>,
    },
    Unknown,
    NonCallable,
    Tuple(Vec<SelectorCallableOrigin>),
    Constructor {
        name: String,
        payloads: Vec<SelectorCallableOrigin>,
    },
    Record {
        name: String,
        fields: BTreeMap<String, SelectorCallableOrigin>,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
#[cfg_attr(
    any(test, feature = "serde"),
    derive(serde::Deserialize, serde::Serialize)
)]
pub(crate) struct SelectorCallableContext {
    root: BTreeMap<String, SelectorCallableOrigin>,
    modules: BTreeMap<String, BTreeMap<String, SelectorCallableOrigin>>,
}

impl SelectorCallableContext {
    fn origins(&self, module: Option<&str>) -> Option<&BTreeMap<String, SelectorCallableOrigin>> {
        match module {
            Some(module) => self.modules.get(module),
            None => Some(&self.root),
        }
    }

    fn origins_mut(
        &mut self,
        module: Option<&str>,
    ) -> &mut BTreeMap<String, SelectorCallableOrigin> {
        match module {
            Some(module) => self.modules.entry(module.to_string()).or_default(),
            None => &mut self.root,
        }
    }
}

type SelectorIdentityTable = BTreeMap<usize, SelectorCallableId>;

struct SelectorResolution {
    context: SelectorCallableContext,
    identities: SelectorIdentityTable,
}

impl SelectorCallableOrigin {
    fn alternate(&self, other: &Self) -> Self {
        match (self, other) {
            (
                Self::Known {
                    identity: left_identity,
                    params: left_params,
                },
                Self::Known {
                    identity: right_identity,
                    params: right_params,
                },
            ) if left_identity == right_identity && left_params == right_params => self.clone(),
            (Self::Tuple(left), Self::Tuple(right)) if left.len() == right.len() => Self::Tuple(
                left.iter()
                    .zip(right)
                    .map(|(left, right)| left.alternate(right))
                    .collect(),
            ),
            (
                Self::Constructor {
                    name: left_name,
                    payloads: left,
                },
                Self::Constructor {
                    name: right_name,
                    payloads: right,
                },
            ) if left_name == right_name && left.len() == right.len() => Self::Constructor {
                name: left_name.clone(),
                payloads: left
                    .iter()
                    .zip(right)
                    .map(|(left, right)| left.alternate(right))
                    .collect(),
            },
            (
                Self::Record {
                    name: left_name,
                    fields: left,
                },
                Self::Record {
                    name: right_name,
                    fields: right,
                },
            ) if left_name == right_name && left.keys().eq(right.keys()) => Self::Record {
                name: left_name.clone(),
                fields: left
                    .iter()
                    .map(|(field, left)| {
                        (
                            field.clone(),
                            left.alternate(
                                right
                                    .get(field)
                                    .expect("equal record key sets contain every left key"),
                            ),
                        )
                    })
                    .collect(),
            },
            (Self::NonCallable, Self::NonCallable) => Self::NonCallable,
            _ => Self::Unknown,
        }
    }
}

/// Enforce [03-META-2] at the checker boundary.
///
/// Surf owns named-selector elaboration, but Deep is a public compiler input
/// and every checker entry accepts it directly. The checker therefore
/// reconstructs only the immutable callable provenance needed to compare the
/// authored `wrt` names with the operative integer selector. This runs in the
/// shared semantic protocol, so IR, typed, fitness, library, and contextual
/// entries cannot drift into separate acceptance rules.
pub(super) fn validate_grad_selector_identity(
    exprs: &[deep::Expr],
    inherited: &SelectorCallableContext,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) {
    let resolution = resolve_selector_callable_context(exprs, inherited, operation);
    let mut module_items: BTreeMap<Option<String>, Vec<&deep::Expr>> = BTreeMap::new();
    for (module, expr) in top_level_decl_items_with_modules(exprs) {
        module_items.entry(module).or_default().push(expr);
    }
    for (module, items) in module_items {
        let callables = resolution
            .context
            .origins(module.as_deref())
            .cloned()
            .unwrap_or_default();
        for expr in items {
            walk_grad_selector_identity(
                expr,
                &callables,
                &mut SelectorLocals::new(),
                &resolution.identities,
                errors,
                operation,
            );
        }
    }
}

pub(crate) fn extend_selector_callable_context(
    exprs: &[deep::Expr],
    inherited: &SelectorCallableContext,
    operation: &crate::resources::Operation<'_>,
) -> SelectorCallableContext {
    resolve_selector_callable_context(exprs, inherited, operation).context
}

fn resolve_selector_callable_context(
    exprs: &[deep::Expr],
    inherited: &SelectorCallableContext,
    operation: &crate::resources::Operation<'_>,
) -> SelectorResolution {
    let identities = selector_identity_table(exprs, inherited, operation);
    let mut context = inherited.clone();
    let mut module_items: BTreeMap<Option<String>, Vec<&deep::Expr>> = BTreeMap::new();
    for (module, expr) in top_level_decl_items_with_modules(exprs) {
        module_items.entry(module).or_default().push(expr);
    }
    for (module, items) in module_items {
        let callables = context.origins_mut(module.as_deref());
        preseed_selector_function_declarations(&items, callables, &identities, operation);
        collect_selector_callable_declarations(&items, callables, &identities, operation);
    }
    SelectorResolution {
        context,
        identities,
    }
}

fn selector_identity_table(
    exprs: &[deep::Expr],
    inherited: &SelectorCallableContext,
    operation: &crate::resources::Operation<'_>,
) -> SelectorIdentityTable {
    let namespace = selector_identity_namespace(exprs, inherited);
    let mut identities = BTreeMap::new();
    let mut next_ordinal = 0_u64;
    for expr in exprs {
        collect_selector_identities(
            expr,
            namespace,
            &mut next_ordinal,
            &mut identities,
            operation,
        );
    }
    identities
}

fn selector_identity_namespace(
    exprs: &[deep::Expr],
    inherited: &SelectorCallableContext,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"chelis-selector-callable-origin-v1");
    let canonical = chelis_deep::printer::print_canonical_flat(exprs);
    hash_selector_bytes(&mut hasher, canonical.as_bytes());
    hash_selector_context(&mut hasher, inherited);
    hasher.finalize().into()
}

pub(crate) fn selector_callable_context_digest(context: &SelectorCallableContext) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"chelis-selector-callable-context-v1");
    hash_selector_context(&mut hasher, context);
    hasher.finalize().into()
}

fn hash_selector_context(hasher: &mut Sha256, context: &SelectorCallableContext) {
    hash_selector_module(hasher, None, &context.root);
    hasher.update((context.modules.len() as u64).to_le_bytes());
    for (module, callables) in &context.modules {
        hash_selector_module(hasher, Some(module), callables);
    }
}

fn hash_selector_module(
    hasher: &mut Sha256,
    module: Option<&str>,
    callables: &BTreeMap<String, SelectorCallableOrigin>,
) {
    match module {
        Some(module) => {
            hasher.update([1]);
            hash_selector_bytes(hasher, module.as_bytes());
        }
        None => hasher.update([0]),
    }
    hasher.update((callables.len() as u64).to_le_bytes());
    for (name, origin) in callables {
        hash_selector_bytes(hasher, name.as_bytes());
        hash_selector_origin(hasher, origin);
    }
}

fn hash_selector_bytes(hasher: &mut Sha256, value: &[u8]) {
    hasher.update((value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn hash_selector_origin(hasher: &mut Sha256, origin: &SelectorCallableOrigin) {
    match origin {
        SelectorCallableOrigin::Known { identity, params } => {
            hasher.update([0]);
            hasher.update(identity.namespace);
            hasher.update(identity.ordinal.to_le_bytes());
            hasher.update((params.len() as u64).to_le_bytes());
            for param in params {
                hash_selector_bytes(hasher, param.as_bytes());
            }
        }
        SelectorCallableOrigin::Unknown => hasher.update([1]),
        SelectorCallableOrigin::NonCallable => hasher.update([2]),
        SelectorCallableOrigin::Tuple(values) => {
            hasher.update([3]);
            hasher.update((values.len() as u64).to_le_bytes());
            for value in values {
                hash_selector_origin(hasher, value);
            }
        }
        SelectorCallableOrigin::Constructor { name, payloads } => {
            hasher.update([4]);
            hash_selector_bytes(hasher, name.as_bytes());
            hasher.update((payloads.len() as u64).to_le_bytes());
            for payload in payloads {
                hash_selector_origin(hasher, payload);
            }
        }
        SelectorCallableOrigin::Record { name, fields } => {
            hasher.update([5]);
            hash_selector_bytes(hasher, name.as_bytes());
            hasher.update((fields.len() as u64).to_le_bytes());
            for (field, value) in fields {
                hash_selector_bytes(hasher, field.as_bytes());
                hash_selector_origin(hasher, value);
            }
        }
    }
}

fn collect_selector_identities(
    expr: &deep::Expr,
    namespace: [u8; 32],
    next_ordinal: &mut u64,
    identities: &mut SelectorIdentityTable,
    operation: &crate::resources::Operation<'_>,
) {
    stack_guard!(operation, "collect_selector_identities", expr);
    if let deep::Expr::MetaExpr(meta, _) = expr {
        collect_selector_identities(&meta.expr, namespace, next_ordinal, identities, operation);
        return;
    }
    if selector_callable_params(expr, operation).is_some() {
        let key = selector_callable_key(expr);
        identities.entry(key).or_insert_with(|| {
            let identity = SelectorCallableId {
                namespace,
                ordinal: *next_ordinal,
            };
            *next_ordinal += 1;
            identity
        });
    }
    if let Some((_, _, children)) = stamped_parts(expr) {
        for child in children {
            collect_selector_identities(child, namespace, next_ordinal, identities, operation);
        }
    }
}

fn preseed_selector_function_declarations(
    exprs: &[&deep::Expr],
    callables: &mut BTreeMap<String, SelectorCallableOrigin>,
    identities: &SelectorIdentityTable,
    operation: &crate::resources::Operation<'_>,
) {
    for expr in exprs {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(value)) = (children.first().and_then(symbol_name), children.get(1))
        else {
            continue;
        };
        let Some(params) = selector_callable_params(value, operation) else {
            continue;
        };
        let Some(identity) = selector_callable_identity(value, identities) else {
            continue;
        };
        callables.insert(
            name.to_string(),
            SelectorCallableOrigin::Known { identity, params },
        );
    }
}

fn collect_selector_callable_declarations(
    exprs: &[&deep::Expr],
    callables: &mut BTreeMap<String, SelectorCallableOrigin>,
    identities: &SelectorIdentityTable,
    operation: &crate::resources::Operation<'_>,
) {
    for expr in exprs {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(value)) = (children.first().and_then(symbol_name), children.get(1))
        else {
            continue;
        };
        let origin = selector_callable_origin(
            value,
            callables,
            &mut SelectorLocals::new(),
            identities,
            operation,
        );
        callables.insert(name.to_string(), origin);
    }
}

fn selector_callable_params(
    expr: &deep::Expr,
    operation: &crate::resources::Operation<'_>,
) -> Option<Vec<String>> {
    let expr = strip_selector_metadata(expr);
    let (DeepTag::Fn, _, children) = stamped_parts(expr)? else {
        return None;
    };
    let (DeepTag::Params, _, params) = children.first().and_then(stamped_parts)? else {
        return None;
    };
    params
        .iter()
        .map(|__checker_arg0| param_name_for_refs(__checker_arg0, operation))
        .collect()
}

fn selector_callable_key(expr: &deep::Expr) -> usize {
    strip_selector_metadata(expr) as *const deep::Expr as usize
}

fn selector_callable_identity(
    expr: &deep::Expr,
    identities: &SelectorIdentityTable,
) -> Option<SelectorCallableId> {
    identities.get(&selector_callable_key(expr)).copied()
}

fn strip_selector_metadata(mut expr: &deep::Expr) -> &deep::Expr {
    while let deep::Expr::MetaExpr(meta, _) = expr {
        expr = &meta.expr;
    }
    expr
}

fn selector_variable_name(expr: &deep::Expr) -> Option<&str> {
    let (DeepTag::Var, _, [name]) = stamped_parts(strip_selector_metadata(expr))? else {
        return None;
    };
    symbol_name(name)
}

fn selector_integer_literal(expr: &deep::Expr) -> Option<i64> {
    let (DeepTag::Lit, _, [value]) = stamped_parts(strip_selector_metadata(expr))? else {
        return None;
    };
    match value {
        deep::Expr::Atom(deep::Atom::Int(value), _) => Some(*value),
        _ => None,
    }
}

fn selector_indices(expr: &deep::Expr) -> Option<Vec<i64>> {
    if let Some(index) = selector_integer_literal(expr) {
        return Some(vec![index]);
    }
    let (DeepTag::Tuple, _, children) = stamped_parts(strip_selector_metadata(expr))? else {
        return None;
    };
    if children.is_empty() {
        return None;
    }
    children.iter().map(selector_integer_literal).collect()
}

/// The callable origins of the local names in scope. A binder's origins last
/// for one [`ScopedMap::scope`], so entering a binder copies nothing
/// (chelis#3740).
type SelectorLocals = ScopedMap<String, SelectorCallableOrigin>;

fn walk_grad_selector_identity(
    expr: &deep::Expr,
    callables: &BTreeMap<String, SelectorCallableOrigin>,
    locals: &mut SelectorLocals,
    identities: &SelectorIdentityTable,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) {
    stack_guard!(operation, "walk_grad_selector_identity", expr);
    if let deep::Expr::MetaExpr(meta, _) = expr {
        walk_grad_selector_identity(&meta.expr, callables, locals, identities, errors, operation);
        return;
    }
    let Some((tag, metadata, children)) = stamped_parts(expr) else {
        return;
    };

    match tag {
        DeepTag::Fn if children.len() == 2 => {
            locals.scope(|scoped| {
                if let Some((DeepTag::Params, _, params)) = children.first().and_then(stamped_parts)
                {
                    for param in params {
                        if let Some(name) = param_name_for_refs(param, operation) {
                            scoped.insert(name, SelectorCallableOrigin::Unknown);
                        }
                    }
                }
                walk_grad_selector_identity(
                    &children[1],
                    callables,
                    scoped,
                    identities,
                    errors,
                    operation,
                );
            });
            return;
        }
        DeepTag::Let if children.len() == 2 => {
            let Some((DeepTag::Bind, _, bindings)) = children.first().and_then(stamped_parts)
            else {
                return;
            };
            if !bindings.len().is_multiple_of(2) {
                return;
            }
            locals.scope(|scoped| {
                for pair in bindings.as_chunks::<2>().0 {
                    walk_grad_selector_identity(
                        &pair[1], callables, scoped, identities, errors, operation,
                    );
                    if let Some(name) = symbol_name(&pair[0]) {
                        let origin = selector_callable_origin(
                            &pair[1], callables, scoped, identities, operation,
                        );
                        scoped.insert(name.to_string(), origin);
                    }
                }
                walk_grad_selector_identity(
                    &children[1],
                    callables,
                    scoped,
                    identities,
                    errors,
                    operation,
                );
            });
            return;
        }
        DeepTag::Match if !children.is_empty() => {
            walk_grad_selector_identity(
                &children[0],
                callables,
                locals,
                identities,
                errors,
                operation,
            );
            let scrutinee =
                selector_callable_origin(&children[0], callables, locals, identities, operation);
            for arm in &children[1..] {
                let Some((DeepTag::Arm, _, arm_children)) = stamped_parts(arm) else {
                    continue;
                };
                if arm_children.len() != 3 {
                    continue;
                }
                locals.scope(|scoped| {
                    bind_selector_match_pattern(&arm_children[0], &scrutinee, scoped, operation);
                    walk_grad_selector_identity(
                        &arm_children[1],
                        callables,
                        scoped,
                        identities,
                        errors,
                        operation,
                    );
                    walk_grad_selector_identity(
                        &arm_children[2],
                        callables,
                        scoped,
                        identities,
                        errors,
                        operation,
                    );
                });
            }
            return;
        }
        _ => {}
    }

    if tag == DeepTag::Grad
        && let Some(wrt) = metadata.wrt()
    {
        validate_one_grad_selector(
            expr,
            children,
            wrt,
            SelectorScope {
                callables,
                locals,
                identities,
            },
            errors,
            operation,
        );
    }

    for child in children {
        walk_grad_selector_identity(child, callables, locals, identities, errors, operation);
    }
}

struct SelectorScope<'a> {
    callables: &'a BTreeMap<String, SelectorCallableOrigin>,
    locals: &'a mut SelectorLocals,
    identities: &'a SelectorIdentityTable,
}

fn validate_one_grad_selector(
    expr: &deep::Expr,
    children: &[deep::Expr],
    wrt: &chelis_deep::annotations::WrtTargets,
    SelectorScope {
        callables,
        locals,
        identities,
    }: SelectorScope<'_>,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) {
    let names = wrt
        .variables()
        .map(|variable| variable.name().value().clone())
        .collect::<Vec<_>>();
    let Some([target, selector]) = <&[deep::Expr; 2]>::try_from(children).ok() else {
        report_at_check_site(
            errors,
            CheckError::new(
                CheckErrorKind::MalformedForm,
                "Deep `grad` carrying `wrt` metadata requires exactly one callable target and one operative integer selector child (spec/03 [03-META-2])".to_string(),
                vec![],
            ),
            CheckSite::Expr(expr),
        );
        return;
    };
    let Some(indices) = selector_indices(selector) else {
        report_at_check_site(
            errors,
            CheckError::new(
                CheckErrorKind::MalformedForm,
                "Deep `grad` carrying `wrt` metadata requires an operative integer literal or nonempty tuple of integer literals (spec/03 [03-META-2])".to_string(),
                vec![],
            ),
            CheckSite::Expr(expr),
        );
        return;
    };
    let origin = selector_callable_origin(target, callables, locals, identities, operation);
    let SelectorCallableOrigin::Known { params, .. } = origin else {
        report_at_check_site(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "Deep `grad` selector target `{}` has no statically resolvable callable origin (spec/03 [03-META-2])",
                    selector_variable_name(target).unwrap_or("<dynamic expression>")
                ),
                vec![],
            ),
            CheckSite::Expr(expr),
        );
        return;
    };
    if names.len() != indices.len() {
        report_at_check_site(
            errors,
            CheckError::new(
                CheckErrorKind::ArityMismatch,
                format!(
                    "Deep `grad` selector metadata names {} parameters but its operative child selects {} indices (spec/03 [03-META-2])",
                    names.len(),
                    indices.len()
                ),
                vec![],
            ),
            CheckSite::Expr(expr),
        );
        return;
    }
    for (name, index) in names.into_iter().zip(indices) {
        let indexed = usize::try_from(index)
            .ok()
            .and_then(|index| params.get(index));
        match indexed {
            Some(indexed_parameter) if indexed_parameter != &name => {
                let expected = format!("parameter `{name}`");
                let got = format!("parameter `{indexed_parameter}`");
                report_at_check_site(
                    errors,
                    CheckError::with_types(
                        CheckErrorKind::TypeMismatch,
                        format!(
                            "Deep `grad` selector index {index}: expected metadata-named {expected}, got {got} (spec/03 [03-META-2])"
                        ),
                        expected,
                        got,
                        vec![],
                    ),
                    CheckSite::Expr(expr),
                );
                return;
            }
            None => {
                let expected = format!("index in 0..{}", params.len());
                let got = format!("index {index}");
                report_at_check_site(
                    errors,
                    CheckError::with_types(
                        CheckErrorKind::ArityMismatch,
                        format!(
                            "Deep `grad` selector for parameter `{name}`: expected {expected} for the callable's {} parameters, got {got} (spec/03 [03-META-2])",
                            params.len()
                        ),
                        expected,
                        got,
                        vec![],
                    ),
                    CheckSite::Expr(expr),
                );
                return;
            }
            Some(_) => {}
        }
    }
}

fn selector_match_origin(
    children: &[deep::Expr],
    callables: &BTreeMap<String, SelectorCallableOrigin>,
    locals: &mut SelectorLocals,
    identities: &SelectorIdentityTable,
    operation: &crate::resources::Operation<'_>,
) -> SelectorCallableOrigin {
    let Some(scrutinee_expr) = children.first() else {
        return SelectorCallableOrigin::Unknown;
    };
    let scrutinee =
        selector_callable_origin(scrutinee_expr, callables, locals, identities, operation);
    let mut result = None;
    for arm in &children[1..] {
        let Some((DeepTag::Arm, _, arm_children)) = stamped_parts(arm) else {
            return SelectorCallableOrigin::Unknown;
        };
        let [pattern, _, body] = arm_children else {
            return SelectorCallableOrigin::Unknown;
        };
        let origin = locals.scope(|scoped| {
            bind_selector_match_pattern(pattern, &scrutinee, scoped, operation);
            selector_callable_origin(body, callables, scoped, identities, operation)
        });
        result = Some(result.map_or_else(
            || origin.clone(),
            |prior: SelectorCallableOrigin| prior.alternate(&origin),
        ));
    }
    result.unwrap_or(SelectorCallableOrigin::Unknown)
}

fn bind_selector_match_pattern(
    pattern: &deep::Expr,
    value: &SelectorCallableOrigin,
    locals: &mut SelectorLocals,
    operation: &crate::resources::Operation<'_>,
) {
    stack_guard!(operation, "bind_selector_match_pattern", pattern);
    if let deep::Expr::MetaExpr(meta, _) = pattern {
        bind_selector_match_pattern(&meta.expr, value, locals, operation);
        return;
    }
    let Some((tag, _, children)) = stamped_parts(pattern) else {
        return;
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = children.first().and_then(symbol_name) {
                locals.insert(name.to_string(), value.clone());
            }
        }
        DeepTag::PatAs => {
            if let Some(name) = children.first().and_then(symbol_name) {
                locals.insert(name.to_string(), value.clone());
            }
            if let Some(nested) = children.get(1) {
                bind_selector_match_pattern(nested, value, locals, operation);
            }
        }
        DeepTag::PatTuple => {
            for (index, child) in children.iter().enumerate() {
                let child_value = match value {
                    SelectorCallableOrigin::Tuple(values) => values
                        .get(index)
                        .cloned()
                        .unwrap_or(SelectorCallableOrigin::Unknown),
                    _ => SelectorCallableOrigin::Unknown,
                };
                bind_selector_match_pattern(child, &child_value, locals, operation);
            }
        }
        DeepTag::PatCtor => {
            let pattern_name = children.first().and_then(symbol_name);
            for (index, child) in children.iter().skip(1).enumerate() {
                let child_value = match value {
                    SelectorCallableOrigin::Constructor { name, payloads }
                        if pattern_name.is_some_and(|pattern_name| pattern_name == name) =>
                    {
                        payloads
                            .get(index)
                            .cloned()
                            .unwrap_or(SelectorCallableOrigin::Unknown)
                    }
                    _ => SelectorCallableOrigin::Unknown,
                };
                bind_selector_match_pattern(child, &child_value, locals, operation);
            }
        }
        DeepTag::PatRecord => {
            let pattern_name = children.first().and_then(symbol_name);
            for field in children.iter().skip(1) {
                let Some((DeepTag::Kv, _, [field_name, value_pattern])) = stamped_parts(field)
                else {
                    continue;
                };
                let field_value = match value {
                    SelectorCallableOrigin::Record { name, fields }
                        if pattern_name.is_some_and(|pattern_name| pattern_name == name) =>
                    {
                        symbol_name(field_name)
                            .and_then(|field_name| fields.get(field_name))
                            .cloned()
                            .unwrap_or(SelectorCallableOrigin::Unknown)
                    }
                    _ => SelectorCallableOrigin::Unknown,
                };
                bind_selector_match_pattern(value_pattern, &field_value, locals, operation);
            }
        }
        _ => {}
    }
}

fn selector_callable_origin(
    expr: &deep::Expr,
    callables: &BTreeMap<String, SelectorCallableOrigin>,
    locals: &mut SelectorLocals,
    identities: &SelectorIdentityTable,
    operation: &crate::resources::Operation<'_>,
) -> SelectorCallableOrigin {
    stack_guard!(
        operation,
        "selector_callable_origin",
        expr,
        SelectorCallableOrigin::Unknown
    );
    let expr = strip_selector_metadata(expr);
    if let Some(name) = selector_variable_name(expr) {
        return locals
            .get(name)
            .cloned()
            .or_else(|| callables.get(name).cloned())
            .unwrap_or(SelectorCallableOrigin::Unknown);
    }
    if let Some(params) = selector_callable_params(expr, operation) {
        let Some(identity) = selector_callable_identity(expr, identities) else {
            return SelectorCallableOrigin::Unknown;
        };
        return SelectorCallableOrigin::Known { identity, params };
    }
    let Some((tag, _, children)) = stamped_parts(expr) else {
        return SelectorCallableOrigin::Unknown;
    };
    match tag {
        DeepTag::If if children.len() == 3 => {
            let consequence =
                selector_callable_origin(&children[1], callables, locals, identities, operation);
            let alternative =
                selector_callable_origin(&children[2], callables, locals, identities, operation);
            consequence.alternate(&alternative)
        }
        DeepTag::Match => selector_match_origin(children, callables, locals, identities, operation),
        DeepTag::Tuple => SelectorCallableOrigin::Tuple(
            children
                .iter()
                .map(|child| {
                    selector_callable_origin(child, callables, locals, identities, operation)
                })
                .collect(),
        ),
        DeepTag::App
            if children
                .first()
                .and_then(selector_variable_name)
                .is_some_and(is_constructor_name) =>
        {
            SelectorCallableOrigin::Constructor {
                name: selector_variable_name(&children[0])
                    .expect("constructor guard established the name")
                    .to_string(),
                payloads: children[1..]
                    .iter()
                    .map(|child| {
                        selector_callable_origin(child, callables, locals, identities, operation)
                    })
                    .collect(),
            }
        }
        DeepTag::Record if !children.is_empty() => {
            let Some(name) = children.first().and_then(symbol_name) else {
                return SelectorCallableOrigin::Unknown;
            };
            let mut fields = BTreeMap::new();
            for field in &children[1..] {
                let Some((DeepTag::Kv, _, [field_name, value])) = stamped_parts(field) else {
                    return SelectorCallableOrigin::Unknown;
                };
                let Some(field_name) = symbol_name(field_name) else {
                    return SelectorCallableOrigin::Unknown;
                };
                fields.insert(
                    field_name.to_string(),
                    selector_callable_origin(value, callables, locals, identities, operation),
                );
            }
            SelectorCallableOrigin::Record {
                name: name.to_string(),
                fields,
            }
        }
        DeepTag::TupleGet if children.len() == 2 => {
            let tuple =
                selector_callable_origin(&children[0], callables, locals, identities, operation);
            match tuple {
                SelectorCallableOrigin::Tuple(values) => selector_integer_literal(&children[1])
                    .and_then(|index| usize::try_from(index).ok())
                    .and_then(|index| values.get(index).cloned())
                    .unwrap_or(SelectorCallableOrigin::Unknown),
                _ => SelectorCallableOrigin::Unknown,
            }
        }
        DeepTag::Grad if !children.is_empty() => {
            // `grad` transforms a callable without changing its formal
            // parameter identity. Preserve the target origin so an outer
            // `grad(..., wrt=...)` validates its operative selector against
            // the same parameter list.
            selector_callable_origin(&children[0], callables, locals, identities, operation)
        }
        DeepTag::Block => children
            .last()
            .map(|item| selector_callable_origin(item, callables, locals, identities, operation))
            .unwrap_or(SelectorCallableOrigin::NonCallable),
        DeepTag::Let if children.len() == 2 => {
            let Some((DeepTag::Bind, _, bindings)) = children.first().and_then(stamped_parts)
            else {
                return SelectorCallableOrigin::Unknown;
            };
            if !bindings.len().is_multiple_of(2) {
                return SelectorCallableOrigin::Unknown;
            }
            locals.scope(|scoped| {
                for pair in bindings.as_chunks::<2>().0 {
                    let Some(name) = symbol_name(&pair[0]) else {
                        return SelectorCallableOrigin::Unknown;
                    };
                    let origin = selector_callable_origin(
                        &pair[1], callables, scoped, identities, operation,
                    );
                    scoped.insert(name.to_string(), origin);
                }
                selector_callable_origin(&children[1], callables, scoped, identities, operation)
            })
        }
        DeepTag::Lit
        | DeepTag::RecordUpdate
        | DeepTag::Par
        | DeepTag::PatLit
        | DeepTag::PatTuple
        | DeepTag::PatRecord
        | DeepTag::PatWild
        | DeepTag::PatCtor
        | DeepTag::PatAs
        | DeepTag::PatVar => SelectorCallableOrigin::NonCallable,
        _ => SelectorCallableOrigin::Unknown,
    }
}
