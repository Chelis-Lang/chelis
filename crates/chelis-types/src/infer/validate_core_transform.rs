//! Core-transform fragment admission.
//!
//! One responsibility: the launch-core fence that rejects a `grad` or `vmap`
//! target which aliases or shadows a top-level function, tracked through the
//! lexical provenance of module and local values. Semantic validation calls
//! [`validate_core_transform_fragment`] once per program.

use super::*;

/// Launch-core transforms are deliberately a smaller acceptance surface than
/// the timeless transform language contract. A named target must not alias an
/// unshadowed top-level function declaration; local wrapper closures remain
/// on their separately tested path. `grad` keeps its existing
/// direct-inline-lambda path. `vmap`'s row typing is owned by its deferred
/// type derivation. This fence prevents an alias or shadowed binding from
/// selecting a different callable (#1952, #1954).
///
/// The walk is lexical rather than type-directed.  A local `loss` with the
/// same function type as a top-level `loss` is exactly the #1954 hazard, so
/// looking only at `Type::Fn` would recreate the silent global fallback.  The
/// module and local values retain only the structural provenance this fence
/// needs through transparent `let`, block result, tuple, tuple-projection,
/// match-pattern, and match-result flow. The target is classified by that same
/// representation, and each binder replaces the previous value so lexical
/// order and shadowing stay explicit. The normative transformation semantics
/// remain in spec/06; the release supported-fragment document records this
/// temporary admission fence.
pub(super) fn validate_core_transform_fragment(
    exprs: &[deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) {
    let mut module_items: UnordMap<Option<String>, Vec<&deep::Expr>> = UnordMap::new();
    for (module, expr) in top_level_decl_items_with_modules(exprs) {
        module_items.entry(module).or_default().push(expr);
    }
    for (_, items) in module_items.to_sorted() {
        let top_level_functions = collect_top_level_function_names(items);
        let module_values = collect_top_level_transform_values(items, &top_level_functions);
        let lexical_scope = CoreTransformScope::default();
        for expr in items {
            walk_core_transform_targets(
                expr,
                &top_level_functions,
                &module_values,
                &lexical_scope,
                errors,
            );
        }
    }
}

#[derive(Clone, Default)]
struct CoreTransformScope {
    local_values: UnordMap<String, CoreTransformValue>,
}

impl CoreTransformScope {
    fn bind_local(&mut self, name: String, value: CoreTransformValue) {
        self.local_values.insert(name, value);
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
enum CoreTransformValue {
    #[default]
    Ordinary,
    TopLevelFunctionAlias,
    Tuple(Vec<CoreTransformValue>),
}

impl CoreTransformValue {
    fn aliases_top_level_function(&self) -> bool {
        matches!(self, Self::TopLevelFunctionAlias)
    }

    /// Conservative union for alternate result paths. `Ordinary` carries no
    /// transform provenance and is the bottom value. Equal tuple structures
    /// join element by element so projection keeps sibling isolation.
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::TopLevelFunctionAlias, _) | (_, Self::TopLevelFunctionAlias) => {
                Self::TopLevelFunctionAlias
            }
            (Self::Ordinary, value) | (value, Self::Ordinary) => value.clone(),
            (Self::Tuple(left), Self::Tuple(right)) if left.len() == right.len() => Self::Tuple(
                left.iter()
                    .zip(right)
                    .map(|(left, right)| left.join(right))
                    .collect(),
            ),
            (Self::Tuple(_), Self::Tuple(_)) => Self::Ordinary,
        }
    }
}

/// Bind the transform-relevant part of a known scrutinee value through one
/// match pattern.
///
/// Direct and `as` binders receive the whole value. Tuple patterns transfer
/// only through an exact structural correspondence, recursively, so one
/// component cannot taint a sibling. Constructor and record structure is not
/// represented by `CoreTransformValue`; their binders therefore shadow with
/// `Ordinary`, as do malformed or otherwise non-matching shapes.
fn bind_core_transform_pattern(
    pattern: &deep::Expr,
    value: &CoreTransformValue,
    scope: &mut CoreTransformScope,
) {
    stack_guard!("bind_core_transform_pattern", pattern);
    if let deep::Expr::MetaExpr(meta, _) = pattern {
        bind_core_transform_pattern(&meta.expr, value, scope);
        return;
    }
    let Some((tag, _, children)) = stamped_parts(pattern) else {
        return;
    };
    match tag {
        DeepTag::PatVar => {
            if let Some(name) = children.first().and_then(symbol_name) {
                scope.bind_local(name.to_string(), value.clone());
            }
        }
        DeepTag::PatTuple => {
            if let CoreTransformValue::Tuple(elements) = value
                && elements.len() == children.len()
            {
                for (child, element) in children.iter().zip(elements) {
                    bind_core_transform_pattern(child, element, scope);
                }
            } else {
                for child in children {
                    bind_core_transform_pattern(child, &CoreTransformValue::Ordinary, scope);
                }
            }
        }
        DeepTag::PatAs => {
            let carried = children
                .get(1)
                .map_or(CoreTransformValue::Ordinary, |inner| {
                    if core_transform_value_matches_pattern(value, inner) {
                        value.clone()
                    } else {
                        CoreTransformValue::Ordinary
                    }
                });
            if let Some(name) = children.first().and_then(symbol_name) {
                scope.bind_local(name.to_string(), carried.clone());
            }
            if let Some(inner) = children.get(1) {
                bind_core_transform_pattern(inner, &carried, scope);
            }
        }
        DeepTag::PatCtor | DeepTag::PatRecord => {
            for child in children.iter().skip(1) {
                bind_core_transform_pattern(child, &CoreTransformValue::Ordinary, scope);
            }
        }
        DeepTag::PatWild | DeepTag::PatLit => {}
        _ => {
            for name in pattern_names_for_signature(pattern).to_sorted() {
                scope.bind_local(name.clone(), CoreTransformValue::Ordinary);
            }
        }
    }
}

/// Whether an `as` pattern's inner structure can correspond to the represented
/// value. `Ordinary` deliberately matches every pattern because it carries no
/// transform provenance; the answer only matters for preventing a known
/// callable or tuple from crossing incompatible pattern structure.
fn core_transform_value_matches_pattern(value: &CoreTransformValue, pattern: &deep::Expr) -> bool {
    stack_guard!("core_transform_value_matches_pattern", pattern, false);
    if matches!(value, CoreTransformValue::Ordinary) {
        return true;
    }
    if let deep::Expr::MetaExpr(meta, _) = pattern {
        return core_transform_value_matches_pattern(value, &meta.expr);
    }
    let Some((tag, _, children)) = stamped_parts(pattern) else {
        return false;
    };
    match tag {
        DeepTag::PatVar | DeepTag::PatWild => true,
        DeepTag::PatAs => children
            .get(1)
            .is_some_and(|inner| core_transform_value_matches_pattern(value, inner)),
        DeepTag::PatTuple => {
            let CoreTransformValue::Tuple(elements) = value else {
                return false;
            };
            elements.len() == children.len()
                && children
                    .iter()
                    .zip(elements)
                    .all(|(child, element)| core_transform_value_matches_pattern(element, child))
        }
        DeepTag::PatLit | DeepTag::PatCtor | DeepTag::PatRecord => false,
        _ => false,
    }
}

fn collect_top_level_function_names(exprs: &[&deep::Expr]) -> UnordSet<String> {
    let mut names = UnordSet::new();
    for expr in exprs {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = children.first().and_then(symbol_name) else {
            continue;
        };
        if children
            .get(1)
            .and_then(stamped_parts)
            .is_some_and(|(tag, _, _)| tag == DeepTag::Fn)
        {
            names.insert(name.to_string());
        }
    }
    names
}

fn collect_top_level_transform_values(
    exprs: &[&deep::Expr],
    top_level_functions: &UnordSet<String>,
) -> UnordMap<String, CoreTransformValue> {
    let mut module_values = UnordMap::new();
    let lexical_scope = CoreTransformScope::default();
    for expr in exprs {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(value)) = (children.first().and_then(symbol_name), children.get(1))
        else {
            continue;
        };
        let value = classify_core_transform_value(
            value,
            top_level_functions,
            &module_values,
            &lexical_scope,
        );
        module_values.insert(name.to_string(), value);
    }
    module_values
}

fn walk_core_transform_targets(
    expr: &deep::Expr,
    top_level_functions: &UnordSet<String>,
    module_values: &UnordMap<String, CoreTransformValue>,
    lexical_scope: &CoreTransformScope,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("walk_core_transform_targets", expr);
    if let deep::Expr::MetaExpr(meta, _) = expr {
        walk_core_transform_targets(
            &meta.expr,
            top_level_functions,
            module_values,
            lexical_scope,
            errors,
        );
        return;
    }
    let Some((tag, _, children)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::Def => {
            if let Some(body) = children.get(1) {
                walk_core_transform_targets(
                    body,
                    top_level_functions,
                    module_values,
                    lexical_scope,
                    errors,
                );
            }
        }
        DeepTag::Fn => {
            let mut scoped = lexical_scope.clone();
            if let Some(params) = children.first()
                && let Some((DeepTag::Params, _, params)) = stamped_parts(params)
            {
                for param in params {
                    if let Some(name) = param_name_for_refs(param) {
                        scoped.bind_local(name, CoreTransformValue::Ordinary);
                    }
                }
            }
            if let Some(body) = children.get(1) {
                walk_core_transform_targets(
                    body,
                    top_level_functions,
                    module_values,
                    &scoped,
                    errors,
                );
            }
        }
        DeepTag::Let => {
            let (Some(binding), Some(body)) = (children.first(), children.get(1)) else {
                return;
            };
            let Some((DeepTag::Bind, _, bind_children)) = stamped_parts(binding) else {
                for child in children {
                    walk_core_transform_targets(
                        child,
                        top_level_functions,
                        module_values,
                        lexical_scope,
                        errors,
                    );
                }
                return;
            };
            if !bind_children.len().is_multiple_of(2) {
                for child in children {
                    walk_core_transform_targets(
                        child,
                        top_level_functions,
                        module_values,
                        lexical_scope,
                        errors,
                    );
                }
                return;
            }
            let mut scoped = lexical_scope.clone();
            for pair in bind_children.as_chunks::<2>().0 {
                let value = &pair[1];
                walk_core_transform_targets(
                    value,
                    top_level_functions,
                    module_values,
                    &scoped,
                    errors,
                );
                if let Some(name) = symbol_name(&pair[0]) {
                    let value = classify_core_transform_value(
                        value,
                        top_level_functions,
                        module_values,
                        &scoped,
                    );
                    scoped.bind_local(name.to_string(), value);
                }
            }
            walk_core_transform_targets(body, top_level_functions, module_values, &scoped, errors);
        }
        DeepTag::Match => {
            let scrutinee_value =
                children
                    .first()
                    .map_or(CoreTransformValue::Ordinary, |scrutinee| {
                        classify_core_transform_value(
                            scrutinee,
                            top_level_functions,
                            module_values,
                            lexical_scope,
                        )
                    });
            if let Some(scrutinee) = children.first() {
                walk_core_transform_targets(
                    scrutinee,
                    top_level_functions,
                    module_values,
                    lexical_scope,
                    errors,
                );
            }
            for arm in children.iter().skip(1) {
                let Some((DeepTag::Arm, _, arm_children)) = stamped_parts(arm) else {
                    walk_core_transform_targets(
                        arm,
                        top_level_functions,
                        module_values,
                        lexical_scope,
                        errors,
                    );
                    continue;
                };
                let mut scoped = lexical_scope.clone();
                if let Some(pattern) = arm_children.first() {
                    bind_core_transform_pattern(pattern, &scrutinee_value, &mut scoped);
                }
                for child in arm_children.iter().skip(1) {
                    walk_core_transform_targets(
                        child,
                        top_level_functions,
                        module_values,
                        &scoped,
                        errors,
                    );
                }
            }
        }
        DeepTag::Grad | DeepTag::Vmap => {
            validate_core_transform_target(
                tag,
                children.first(),
                top_level_functions,
                module_values,
                lexical_scope,
                errors,
            );
            for child in children {
                walk_core_transform_targets(
                    child,
                    top_level_functions,
                    module_values,
                    lexical_scope,
                    errors,
                );
            }
        }
        _ => {
            for child in children {
                walk_core_transform_targets(
                    child,
                    top_level_functions,
                    module_values,
                    lexical_scope,
                    errors,
                );
            }
        }
    }
}

/// Preserve the transform-relevant identity of a module, local, result, or
/// direct-target value through the transparent forwarding forms admitted by
/// the release fragment.
///
/// This deliberately is not general value inference: applications, branches,
/// records, constructors, and arbitrary computation collapse to `Ordinary`.
/// The fence only retains a callable already known to be unsupported while it
/// flows through lexical aliases, blocks, match results, or statically selected
/// tuple components.
fn classify_core_transform_value(
    value: &deep::Expr,
    top_level_functions: &UnordSet<String>,
    module_values: &UnordMap<String, CoreTransformValue>,
    lexical_scope: &CoreTransformScope,
) -> CoreTransformValue {
    stack_guard!(
        "classify_core_transform_value",
        value,
        CoreTransformValue::Ordinary
    );
    if let deep::Expr::MetaExpr(meta, _) = value {
        return classify_core_transform_value(
            &meta.expr,
            top_level_functions,
            module_values,
            lexical_scope,
        );
    }
    let Some((tag, _, children)) = stamped_parts(value) else {
        return CoreTransformValue::Ordinary;
    };
    match tag {
        DeepTag::Fn => CoreTransformValue::Ordinary,
        DeepTag::Var => {
            let Some(name) = children.first().and_then(symbol_name) else {
                return CoreTransformValue::Ordinary;
            };
            lexical_scope
                .local_values
                .get(name)
                .cloned()
                .unwrap_or_else(|| {
                    if top_level_functions.contains(name) {
                        CoreTransformValue::TopLevelFunctionAlias
                    } else {
                        module_values.get(name).cloned().unwrap_or_default()
                    }
                })
        }
        DeepTag::Tuple => CoreTransformValue::Tuple(
            children
                .iter()
                .map(|child| {
                    classify_core_transform_value(
                        child,
                        top_level_functions,
                        module_values,
                        lexical_scope,
                    )
                })
                .collect(),
        ),
        DeepTag::TupleGet => {
            let (Some(tuple), Some(index)) =
                (children.first(), children.get(1).and_then(tuple_get_index))
            else {
                return CoreTransformValue::Ordinary;
            };
            let CoreTransformValue::Tuple(elements) = classify_core_transform_value(
                tuple,
                top_level_functions,
                module_values,
                lexical_scope,
            ) else {
                return CoreTransformValue::Ordinary;
            };
            elements.get(index).cloned().unwrap_or_default()
        }
        DeepTag::Let => {
            let (Some((DeepTag::Bind, _, bind_children)), Some(body)) =
                (children.first().and_then(stamped_parts), children.get(1))
            else {
                return CoreTransformValue::Ordinary;
            };
            if !bind_children.len().is_multiple_of(2) {
                return CoreTransformValue::Ordinary;
            }
            let mut scoped = lexical_scope.clone();
            for pair in bind_children.as_chunks::<2>().0 {
                let Some(name) = symbol_name(&pair[0]) else {
                    return CoreTransformValue::Ordinary;
                };
                let bound_value = classify_core_transform_value(
                    &pair[1],
                    top_level_functions,
                    module_values,
                    &scoped,
                );
                scoped.bind_local(name.to_string(), bound_value);
            }
            classify_core_transform_value(body, top_level_functions, module_values, &scoped)
        }
        DeepTag::Block => children
            .last()
            .map_or(CoreTransformValue::Ordinary, |result| {
                classify_core_transform_value(
                    result,
                    top_level_functions,
                    module_values,
                    lexical_scope,
                )
            }),
        DeepTag::Match => {
            let Some(scrutinee) = children.first() else {
                return CoreTransformValue::Ordinary;
            };
            let scrutinee_value = classify_core_transform_value(
                scrutinee,
                top_level_functions,
                module_values,
                lexical_scope,
            );
            let mut result: Option<CoreTransformValue> = None;
            for arm in children.iter().skip(1) {
                let Some((DeepTag::Arm, _, arm_children)) = stamped_parts(arm) else {
                    return CoreTransformValue::Ordinary;
                };
                let (Some(pattern), Some(body)) = (arm_children.first(), arm_children.get(2))
                else {
                    return CoreTransformValue::Ordinary;
                };
                let mut scoped = lexical_scope.clone();
                bind_core_transform_pattern(pattern, &scrutinee_value, &mut scoped);
                let arm_value = classify_core_transform_value(
                    body,
                    top_level_functions,
                    module_values,
                    &scoped,
                );
                result =
                    Some(result.map_or_else(|| arm_value.clone(), |known| known.join(&arm_value)));
            }
            result.unwrap_or_default()
        }
        _ => CoreTransformValue::Ordinary,
    }
}

fn validate_core_transform_target(
    tag: DeepTag,
    target: Option<&deep::Expr>,
    top_level_functions: &UnordSet<String>,
    module_values: &UnordMap<String, CoreTransformValue>,
    lexical_scope: &CoreTransformScope,
    errors: &mut DiagnosticSink<'_>,
) {
    let target_parts = target.and_then(stamped_parts);
    let name = target_parts.and_then(|(target_tag, _, children)| {
        (target_tag == DeepTag::Var)
            .then(|| children.first().and_then(symbol_name))
            .flatten()
    });
    let shadows_top_level = name.is_some_and(|name| {
        top_level_functions.contains(name) && lexical_scope.local_values.contains_key(name)
    });
    let direct_unshadowed_top_level = name.is_some_and(|name| {
        top_level_functions.contains(name) && !lexical_scope.local_values.contains_key(name)
    });
    let target_value = target.map_or(CoreTransformValue::Ordinary, |target| {
        classify_core_transform_value(target, top_level_functions, module_values, lexical_scope)
    });
    let aliases_top_level_function =
        !direct_unshadowed_top_level && target_value.aliases_top_level_function();

    let requires_fence = match tag {
        // Existing `grad(fn (...) -> ...)` execution is a distinct, covered
        // path. The P1 hazards are aliases and a local binder choosing a
        // same-named global declaration, both represented as `var`.
        DeepTag::Grad => shadows_top_level || aliases_top_level_function,
        // Row-parameter inference is owned by the vmap type derivation.
        // This provenance fence still owns top-level aliases and shadowing.
        DeepTag::Vmap => shadows_top_level || aliases_top_level_function,
        _ => unreachable!("only transform tags call this validator"),
    };
    if !requires_fence {
        return;
    }

    let transform = match tag {
        DeepTag::Grad => "grad",
        DeepTag::Vmap => "vmap",
        _ => unreachable!("only transform tags call this validator"),
    };
    errors.push(CheckError::new(
        CheckErrorKind::TypeMismatch,
        format!(
            "the core transform fragment rejects this `{transform}` target: local aliases \
             and shadowing bindings of top-level functions must be direct, unshadowed \
             top-level function declarations (chelis#1952, #1954)"
        ),
        vec![
            format!("Define a top-level function and write `{transform}(that_function)`."),
            "The rejected callable form is outside the Chelis 0.19 core fragment.".to_string(),
        ],
    ));
}

#[cfg(test)]
mod core_transform_fragment_tests {
    use super::*;

    fn deep_type(source: &str) -> deep::Expr {
        let mut parsed = chelis_deep::parser::parse_str(source).expect("canonical Deep type");
        assert_eq!(parsed.len(), 1, "one Deep type expression");
        parsed.remove(0)
    }

    fn pattern_scope(source: &str, value: CoreTransformValue) -> CoreTransformScope {
        let pattern = deep_type(source);
        let mut scope = CoreTransformScope::default();
        bind_core_transform_pattern(&pattern, &value, &mut scope);
        scope
    }

    fn binding<'a>(scope: &'a CoreTransformScope, name: &str) -> &'a CoreTransformValue {
        scope
            .local_values
            .get(name)
            .unwrap_or_else(|| panic!("missing pattern binding `{name}`"))
    }

    #[test]
    fn module_transform_values_use_structural_projection() {
        let source = "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
                      pair = (reduce, 0i32)\n\
                      mapped = pair.0\n";
        let declarations = chelis_surf::parser::parse_str(source).expect("module values parse");
        let program = chelis_surf::desugar::desugar_program(&declarations)
            .expect("Surf fixture must desugar");
        let items = program.iter().collect::<Vec<_>>();
        let top_level_functions = collect_top_level_function_names(&items);
        let module_values = collect_top_level_transform_values(&items, &top_level_functions);
        assert!(
            module_values
                .get("mapped")
                .is_some_and(CoreTransformValue::aliases_top_level_function),
            "module tuple projection must retain top-level alias provenance"
        );
    }

    #[test]
    fn core_transform_provenance_is_scoped_by_enclosing_module() {
        let source = "\
          (module {} Alpha \
            (def {} reduce \
              (fn {} (params {} (v {type: \
                (t-tensor {} (d-lit {} 4) (d-lit {} 3) (t-prim {} f32))})) \
                (var {} v)))) \
          (module {} Beta \
            (def {} probe \
              (fn {} (params {} t) \
                (let {} \
                  (bind {} reduce \
                    (fn {} (params {} (v {type: \
                      (t-tensor {} (d-lit {} 4) (d-lit {} 3) (t-prim {} f32))})) \
                      (var {} v))) \
                  (vmap {} (var {} reduce))))))";
        let program = chelis_deep::parser::parse_str(source).expect("canonical module program");
        let result = crate::infer_program(&program);
        assert!(
            result
                .errors
                .iter()
                .all(|error| !error.message.contains("core transform fragment")),
            "Alpha.reduce must not taint Beta's local reduce: {:?}",
            result.errors
        );
    }

    #[test]
    fn alternate_result_join_preserves_aliases_and_tuple_siblings() {
        let alias = CoreTransformValue::TopLevelFunctionAlias;
        assert_eq!(CoreTransformValue::Ordinary.join(&alias), alias);
        let left = CoreTransformValue::Tuple(vec![alias.clone(), CoreTransformValue::Ordinary]);
        let right = CoreTransformValue::Tuple(vec![CoreTransformValue::Ordinary, alias.clone()]);
        assert_eq!(
            left.join(&right),
            CoreTransformValue::Tuple(vec![alias.clone(), alias.clone()])
        );
        assert_eq!(
            CoreTransformValue::Tuple(vec![alias]).join(&CoreTransformValue::Tuple(vec![
                CoreTransformValue::Ordinary,
                CoreTransformValue::Ordinary,
            ])),
            CoreTransformValue::Ordinary,
            "mismatched result structure must not invent transferable provenance"
        );
    }

    #[test]
    fn match_pattern_provenance_transfer_is_structural_and_conservative() {
        let direct = pattern_scope(
            "(pat-var {} mapped)",
            CoreTransformValue::TopLevelFunctionAlias,
        );
        assert_eq!(
            binding(&direct, "mapped"),
            &CoreTransformValue::TopLevelFunctionAlias
        );

        let nested = pattern_scope(
            "(pat-tuple {} \
               (pat-var {} alias) \
               (pat-tuple {} (pat-wild {}) (pat-var {} mapped)))",
            CoreTransformValue::Tuple(vec![
                CoreTransformValue::TopLevelFunctionAlias,
                CoreTransformValue::Tuple(vec![
                    CoreTransformValue::Ordinary,
                    CoreTransformValue::TopLevelFunctionAlias,
                ]),
            ]),
        );
        assert_eq!(
            binding(&nested, "alias"),
            &CoreTransformValue::TopLevelFunctionAlias
        );
        assert_eq!(
            binding(&nested, "mapped"),
            &CoreTransformValue::TopLevelFunctionAlias
        );
        assert!(!nested.local_values.contains_key("_"));

        let siblings = pattern_scope(
            "(pat-tuple {} (pat-var {} left) (pat-var {} right))",
            CoreTransformValue::Tuple(vec![
                CoreTransformValue::TopLevelFunctionAlias,
                CoreTransformValue::Ordinary,
            ]),
        );
        assert_eq!(
            binding(&siblings, "left"),
            &CoreTransformValue::TopLevelFunctionAlias
        );
        assert_eq!(binding(&siblings, "right"), &CoreTransformValue::Ordinary);

        let as_pattern = pattern_scope(
            "(pat-as {} whole (pat-var {} mapped))",
            CoreTransformValue::TopLevelFunctionAlias,
        );
        assert_eq!(
            binding(&as_pattern, "whole"),
            &CoreTransformValue::TopLevelFunctionAlias
        );
        assert_eq!(
            binding(&as_pattern, "mapped"),
            &CoreTransformValue::TopLevelFunctionAlias
        );

        for pattern in [
            "(pat-tuple {} (pat-var {} left) (pat-var {} right))",
            "(pat-as {} whole (pat-tuple {} (pat-var {} mapped)))",
            "(pat-ctor {} Box (pat-var {} mapped))",
            "(pat-record {} Box (kv {} value (pat-var {} mapped)))",
        ] {
            let mismatched = pattern_scope(pattern, CoreTransformValue::TopLevelFunctionAlias);
            for (_, value) in mismatched.local_values.to_sorted() {
                assert_eq!(
                    value,
                    &CoreTransformValue::Ordinary,
                    "{pattern} must not transfer provenance through unknown or mismatched structure"
                );
            }
        }

        for pattern in ["(pat-wild {})", "(pat-lit {} 0)"] {
            assert!(
                pattern_scope(pattern, CoreTransformValue::TopLevelFunctionAlias)
                    .local_values
                    .is_empty(),
                "{pattern} binds no value"
            );
        }
    }

    #[test]
    fn surf_vmap_tensor_slot_holes_have_earlier_or_safe_boundaries() {
        let bare_type_variable = "out = vmap(fn (v: a) -> v)(to_tensor([[1.0f32]]))\n";
        let declarations =
            chelis_surf::parser::parse_str(bare_type_variable).expect("bare type variable parses");
        let program = chelis_surf::desugar::desugar_program(&declarations)
            .expect("Surf fixture must desugar");
        let result = crate::infer_program(&program);
        assert!(
            result.errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::TypeMismatch)
                    && error
                        .message
                        .contains("unknown primitive type `a` in type annotation")
                    && error.suggestions.iter().any(|suggestion| {
                        suggestion.contains("declare `a` in the signature binder list")
                    })
            }),
            "Surf bare type names remain owned by explicit-binder type resolution: {:?}",
            result.errors
        );
        assert!(
            result
                .errors
                .iter()
                .all(|error| !error.message.contains("core transform fragment")),
            "the vmap fence must not steal an undeclared type variable: {:?}",
            result.errors
        );

        let precision_hole = "out = vmap(fn (v: tensor[4, _]) -> v)(to_tensor([[1.0f32]]))\n";
        assert!(
            chelis_surf::parser::parse_str(precision_hole).is_err(),
            "Surf requires a named tensor precision"
        );

        let rank_hole = "out = vmap(fn (v: tensor[.._, f32]) -> v)(to_tensor([[1.0f32]]))\n";
        assert!(
            chelis_surf::parser::parse_str(rank_hole).is_err(),
            "Surf requires a named rank spread"
        );

        let dimension_hole = "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
             vmap(fn (v: tensor[_, 3, f32]) -> sum(v, 0i32))(t)\n";
        assert!(
            chelis_surf::parser::parse_str(dimension_hole).is_err(),
            "Surf dimension `_` must be rejected before desugaring"
        );
    }
}
