//! Checker-boundary validation for `grad` selector identity.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SelectorCallableId(usize);

#[derive(Clone, Debug, Eq, PartialEq)]
enum SelectorCallableOrigin {
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
    errors: &mut DiagnosticSink<'_>,
) {
    let mut module_items: BTreeMap<Option<String>, Vec<&deep::Expr>> = BTreeMap::new();
    for (module, expr) in top_level_decl_items_with_modules(exprs) {
        module_items.entry(module).or_default().push(expr);
    }
    for items in module_items.into_values() {
        let mut callables = BTreeMap::new();
        preseed_selector_function_declarations(&items, &mut callables);
        collect_selector_callable_declarations(&items, &mut callables);
        for expr in items {
            walk_grad_selector_identity(expr, &callables, &BTreeMap::new(), errors);
        }
    }
}

fn preseed_selector_function_declarations(
    exprs: &[&deep::Expr],
    callables: &mut BTreeMap<String, SelectorCallableOrigin>,
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
        callables.insert(
            name.to_string(),
            SelectorCallableOrigin::Known {
                identity: selector_callable_identity(value),
                params,
            },
        );
    }
}

fn collect_selector_callable_declarations(
    exprs: &[&deep::Expr],
    callables: &mut BTreeMap<String, SelectorCallableOrigin>,
) {
    for expr in exprs {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(value)) = (children.first().and_then(symbol_name), children.get(1))
        else {
            continue;
        };
        let origin = selector_callable_origin(value, callables, &BTreeMap::new());
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

fn selector_callable_identity(expr: &deep::Expr) -> SelectorCallableId {
    SelectorCallableId(strip_selector_metadata(expr) as *const deep::Expr as usize)
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
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("walk_grad_selector_identity", expr);
    if let deep::Expr::MetaExpr(meta, _) = expr {
        walk_grad_selector_identity(&meta.expr, callables, locals, errors);
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
            walk_grad_selector_identity(&children[1], callables, &scoped, errors);
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
                walk_grad_selector_identity(&pair[1], callables, &scoped, errors);
                if let Some(name) = symbol_name(&pair[0]) {
                    let origin = selector_callable_origin(&pair[1], callables, &scoped);
                    scoped.insert(name.to_string(), origin);
                }
            }
            walk_grad_selector_identity(&children[1], callables, &scoped, errors);
            return;
        }
        DeepTag::Match if !children.is_empty() => {
            walk_grad_selector_identity(&children[0], callables, locals, errors);
            let scrutinee = selector_callable_origin(&children[0], callables, locals);
            for arm in &children[1..] {
                let Some((DeepTag::Arm, _, arm_children)) = stamped_parts(arm) else {
                    continue;
                };
                if arm_children.len() != 3 {
                    continue;
                }
                let mut scoped = locals.clone();
                bind_selector_match_pattern(&arm_children[0], &scrutinee, &mut scoped);
                walk_grad_selector_identity(&arm_children[1], callables, &scoped, errors);
                walk_grad_selector_identity(&arm_children[2], callables, &scoped, errors);
            }
            return;
        }
        _ => {}
    }

    if tag == DeepTag::Grad
        && let Some(wrt) = metadata.wrt()
    {
        validate_one_grad_selector(children, wrt, callables, locals, errors);
    }

    for child in children {
        walk_grad_selector_identity(child, callables, locals, errors);
    }
}

fn validate_one_grad_selector(
    children: &[deep::Expr],
    wrt: &chelis_deep::annotations::WrtTargets,
    callables: &BTreeMap<String, SelectorCallableOrigin>,
    locals: &BTreeMap<String, SelectorCallableOrigin>,
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
    let origin = selector_callable_origin(target, callables, locals);
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
) -> SelectorCallableOrigin {
    let Some(scrutinee_expr) = children.first() else {
        return SelectorCallableOrigin::Unknown;
    };
    let scrutinee = selector_callable_origin(scrutinee_expr, callables, locals);
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
        let origin = selector_callable_origin(body, callables, &scoped);
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
        return SelectorCallableOrigin::Known {
            identity: selector_callable_identity(expr),
            params,
        };
    }
    let Some((tag, _, children)) = stamped_parts(expr) else {
        return SelectorCallableOrigin::Unknown;
    };
    match tag {
        DeepTag::If if children.len() == 3 => {
            let consequence = selector_callable_origin(&children[1], callables, locals);
            let alternative = selector_callable_origin(&children[2], callables, locals);
            consequence.alternate(&alternative)
        }
        DeepTag::Match => selector_match_origin(children, callables, locals),
        DeepTag::Tuple => SelectorCallableOrigin::Tuple(
            children
                .iter()
                .map(|child| selector_callable_origin(child, callables, locals))
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
                    .map(|child| selector_callable_origin(child, callables, locals))
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
                    selector_callable_origin(value, callables, locals),
                );
            }
            SelectorCallableOrigin::Record {
                name: name.to_string(),
                fields,
            }
        }
        DeepTag::TupleGet if children.len() == 2 => {
            let tuple = selector_callable_origin(&children[0], callables, locals);
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
            .map(|item| selector_callable_origin(item, callables, locals))
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
                let origin = selector_callable_origin(&pair[1], callables, &scoped);
                scoped.insert(name.to_string(), origin);
            }
            selector_callable_origin(&children[1], callables, &scoped)
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
