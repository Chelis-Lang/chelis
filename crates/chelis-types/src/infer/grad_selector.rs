//! Checker-boundary validation for `grad` selector identity.

use super::*;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub(crate) struct SelectorCallableId {
    namespace: [u8; 32],
    ordinal: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
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

#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
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
) {
    let resolution = resolve_selector_callable_context(exprs, inherited);
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
                &BTreeMap::new(),
                &resolution.identities,
                errors,
            );
        }
    }
}

pub(crate) fn extend_selector_callable_context(
    exprs: &[deep::Expr],
    inherited: &SelectorCallableContext,
) -> SelectorCallableContext {
    resolve_selector_callable_context(exprs, inherited).context
}

fn resolve_selector_callable_context(
    exprs: &[deep::Expr],
    inherited: &SelectorCallableContext,
) -> SelectorResolution {
    let identities = selector_identity_table(exprs, inherited);
    let mut context = inherited.clone();
    let mut module_items: BTreeMap<Option<String>, Vec<&deep::Expr>> = BTreeMap::new();
    for (module, expr) in top_level_decl_items_with_modules(exprs) {
        module_items.entry(module).or_default().push(expr);
    }
    for (module, items) in module_items {
        let callables = context.origins_mut(module.as_deref());
        preseed_selector_function_declarations(&items, callables, &identities);
        collect_selector_callable_declarations(&items, callables, &identities);
    }
    SelectorResolution {
        context,
        identities,
    }
}

fn selector_identity_table(
    exprs: &[deep::Expr],
    inherited: &SelectorCallableContext,
) -> SelectorIdentityTable {
    let namespace = selector_identity_namespace(exprs, inherited);
    let mut identities = BTreeMap::new();
    let mut next_ordinal = 0_u64;
    for expr in exprs {
        collect_selector_identities(expr, namespace, &mut next_ordinal, &mut identities);
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
) {
    stack_guard!("collect_selector_identities", expr);
    if let deep::Expr::MetaExpr(meta, _) = expr {
        collect_selector_identities(&meta.expr, namespace, next_ordinal, identities);
        return;
    }
    if selector_callable_params(expr).is_some() {
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
            collect_selector_identities(child, namespace, next_ordinal, identities);
        }
    }
}

fn preseed_selector_function_declarations(
    exprs: &[&deep::Expr],
    callables: &mut BTreeMap<String, SelectorCallableOrigin>,
    identities: &SelectorIdentityTable,
) {
    for expr in exprs {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(value)) = (children.first().and_then(symbol_name), children.get(1))
        else {
            continue;
        };
        let Some(params) = selector_callable_params(value) else {
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
) {
    for expr in exprs {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(value)) = (children.first().and_then(symbol_name), children.get(1))
        else {
            continue;
        };
        let origin = selector_callable_origin(value, callables, &BTreeMap::new(), identities);
        callables.insert(name.to_string(), origin);
    }
}

fn selector_callable_params(expr: &deep::Expr) -> Option<Vec<String>> {
    let expr = strip_selector_metadata(expr);
    let (DeepTag::Fn, _, children) = stamped_parts(expr)? else {
        return None;
    };
    let (DeepTag::Params, _, params) = children.first().and_then(stamped_parts)? else {
        return None;
    };
    params.iter().map(param_name_for_refs).collect()
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

fn walk_grad_selector_identity(
    expr: &deep::Expr,
    callables: &BTreeMap<String, SelectorCallableOrigin>,
    locals: &BTreeMap<String, SelectorCallableOrigin>,
    identities: &SelectorIdentityTable,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("walk_grad_selector_identity", expr);
    if let deep::Expr::MetaExpr(meta, _) = expr {
        walk_grad_selector_identity(&meta.expr, callables, locals, identities, errors);
        return;
    }
    let Some((tag, metadata, children)) = stamped_parts(expr) else {
        return;
    };

    match tag {
        DeepTag::Fn if children.len() == 2 => {
            let mut scoped = locals.clone();
            if let Some((DeepTag::Params, _, params)) = children.first().and_then(stamped_parts) {
                for param in params {
                    if let Some(name) = param_name_for_refs(param) {
                        scoped.insert(name, SelectorCallableOrigin::Unknown);
                    }
                }
            }
            walk_grad_selector_identity(&children[1], callables, &scoped, identities, errors);
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
            let mut scoped = locals.clone();
            for pair in bindings.as_chunks::<2>().0 {
                walk_grad_selector_identity(&pair[1], callables, &scoped, identities, errors);
                if let Some(name) = symbol_name(&pair[0]) {
                    let origin = selector_callable_origin(&pair[1], callables, &scoped, identities);
                    scoped.insert(name.to_string(), origin);
                }
            }
            walk_grad_selector_identity(&children[1], callables, &scoped, identities, errors);
            return;
        }
        DeepTag::Match if !children.is_empty() => {
            walk_grad_selector_identity(&children[0], callables, locals, identities, errors);
            let scrutinee = selector_callable_origin(&children[0], callables, locals, identities);
            for arm in &children[1..] {
                let Some((DeepTag::Arm, _, arm_children)) = stamped_parts(arm) else {
                    continue;
                };
                if arm_children.len() != 3 {
                    continue;
                }
                let mut scoped = locals.clone();
                bind_selector_match_pattern(&arm_children[0], &scrutinee, &mut scoped);
                walk_grad_selector_identity(
                    &arm_children[1],
                    callables,
                    &scoped,
                    identities,
                    errors,
                );
                walk_grad_selector_identity(
                    &arm_children[2],
                    callables,
                    &scoped,
                    identities,
                    errors,
                );
            }
            return;
        }
        _ => {}
    }

    if tag == DeepTag::Grad
        && let Some(wrt) = metadata.wrt()
    {
        validate_one_grad_selector(children, wrt, callables, locals, identities, errors);
    }

    for child in children {
        walk_grad_selector_identity(child, callables, locals, identities, errors);
    }
}

fn validate_one_grad_selector(
    children: &[deep::Expr],
    wrt: &chelis_deep::annotations::WrtTargets,
    callables: &BTreeMap<String, SelectorCallableOrigin>,
    locals: &BTreeMap<String, SelectorCallableOrigin>,
    identities: &SelectorIdentityTable,
    errors: &mut DiagnosticSink<'_>,
) {
    let names = wrt
        .variables()
        .map(|variable| variable.name().value().clone())
        .collect::<Vec<_>>();
    let Some([target, selector]) = <&[deep::Expr; 2]>::try_from(children).ok() else {
        errors.push(CheckError::new(
            CheckErrorKind::MalformedForm,
            "Deep `grad` carrying `wrt` metadata requires exactly one callable target and one operative integer selector child (spec/03 [03-META-2])".to_string(),
            vec![],
        ));
        return;
    };
    let Some(indices) = selector_indices(selector) else {
        errors.push(CheckError::new(
            CheckErrorKind::MalformedForm,
            "Deep `grad` carrying `wrt` metadata requires an operative integer literal or nonempty tuple of integer literals (spec/03 [03-META-2])".to_string(),
            vec![],
        ));
        return;
    };
    let origin = selector_callable_origin(target, callables, locals, identities);
    let SelectorCallableOrigin::Known { params, .. } = origin else {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!(
                "Deep `grad` selector target `{}` has no statically resolvable callable origin (spec/03 [03-META-2])",
                selector_variable_name(target).unwrap_or("<dynamic expression>")
            ),
            vec![],
        ));
        return;
    };
    if names.len() != indices.len() {
        errors.push(CheckError::new(
            CheckErrorKind::ArityMismatch,
            format!(
                "Deep `grad` selector metadata names {} parameter(s), but its index child selects {} (spec/03 [03-META-2])",
                names.len(),
                indices.len()
            ),
            vec![],
        ));
        return;
    }
    for (name, index) in names.into_iter().zip(indices) {
        let indexed = usize::try_from(index)
            .ok()
            .and_then(|index| params.get(index));
        match indexed {
            Some(indexed_parameter) if indexed_parameter != &name => {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "Deep `grad` selector metadata names parameter `{name}`, but index {index} selects parameter `{indexed_parameter}` (spec/03 [03-META-2])"
                    ),
                    vec![],
                ));
                return;
            }
            None => {
                errors.push(CheckError::new(
                    CheckErrorKind::ArityMismatch,
                    format!(
                        "Deep `grad` selector metadata names parameter `{name}`, but index {index} is outside the callable's {} parameter(s) (spec/03 [03-META-2])",
                        params.len()
                    ),
                    vec![],
                ));
                return;
            }
            Some(_) => {}
        }
    }
}

fn selector_match_origin(
    children: &[deep::Expr],
    callables: &BTreeMap<String, SelectorCallableOrigin>,
    locals: &BTreeMap<String, SelectorCallableOrigin>,
    identities: &SelectorIdentityTable,
) -> SelectorCallableOrigin {
    let Some(scrutinee_expr) = children.first() else {
        return SelectorCallableOrigin::Unknown;
    };
    let scrutinee = selector_callable_origin(scrutinee_expr, callables, locals, identities);
    let mut result = None;
    for arm in &children[1..] {
        let Some((DeepTag::Arm, _, arm_children)) = stamped_parts(arm) else {
            return SelectorCallableOrigin::Unknown;
        };
        let [pattern, _, body] = arm_children else {
            return SelectorCallableOrigin::Unknown;
        };
        let mut scoped = locals.clone();
        bind_selector_match_pattern(pattern, &scrutinee, &mut scoped);
        let origin = selector_callable_origin(body, callables, &scoped, identities);
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
    locals: &mut BTreeMap<String, SelectorCallableOrigin>,
) {
    stack_guard!("bind_selector_match_pattern", pattern);
    if let deep::Expr::MetaExpr(meta, _) = pattern {
        bind_selector_match_pattern(&meta.expr, value, locals);
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
                bind_selector_match_pattern(nested, value, locals);
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
                bind_selector_match_pattern(child, &child_value, locals);
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
                bind_selector_match_pattern(child, &child_value, locals);
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
                bind_selector_match_pattern(value_pattern, &field_value, locals);
            }
        }
        _ => {}
    }
}

fn selector_callable_origin(
    expr: &deep::Expr,
    callables: &BTreeMap<String, SelectorCallableOrigin>,
    locals: &BTreeMap<String, SelectorCallableOrigin>,
    identities: &SelectorIdentityTable,
) -> SelectorCallableOrigin {
    stack_guard!(
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
    if let Some(params) = selector_callable_params(expr) {
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
            let consequence = selector_callable_origin(&children[1], callables, locals, identities);
            let alternative = selector_callable_origin(&children[2], callables, locals, identities);
            consequence.alternate(&alternative)
        }
        DeepTag::Match => selector_match_origin(children, callables, locals, identities),
        DeepTag::Tuple => SelectorCallableOrigin::Tuple(
            children
                .iter()
                .map(|child| selector_callable_origin(child, callables, locals, identities))
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
                    .map(|child| selector_callable_origin(child, callables, locals, identities))
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
                    selector_callable_origin(value, callables, locals, identities),
                );
            }
            SelectorCallableOrigin::Record {
                name: name.to_string(),
                fields,
            }
        }
        DeepTag::TupleGet if children.len() == 2 => {
            let tuple = selector_callable_origin(&children[0], callables, locals, identities);
            match tuple {
                SelectorCallableOrigin::Tuple(values) => selector_integer_literal(&children[1])
                    .and_then(|index| usize::try_from(index).ok())
                    .and_then(|index| values.get(index).cloned())
                    .unwrap_or(SelectorCallableOrigin::Unknown),
                _ => SelectorCallableOrigin::Unknown,
            }
        }
        DeepTag::Block => children
            .last()
            .map(|item| selector_callable_origin(item, callables, locals, identities))
            .unwrap_or(SelectorCallableOrigin::NonCallable),
        DeepTag::Let if children.len() == 2 => {
            let Some((DeepTag::Bind, _, bindings)) = children.first().and_then(stamped_parts)
            else {
                return SelectorCallableOrigin::Unknown;
            };
            if !bindings.len().is_multiple_of(2) {
                return SelectorCallableOrigin::Unknown;
            }
            let mut scoped = locals.clone();
            for pair in bindings.as_chunks::<2>().0 {
                let Some(name) = symbol_name(&pair[0]) else {
                    return SelectorCallableOrigin::Unknown;
                };
                let origin = selector_callable_origin(&pair[1], callables, &scoped, identities);
                scoped.insert(name.to_string(), origin);
            }
            selector_callable_origin(&children[1], callables, &scoped, identities)
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
