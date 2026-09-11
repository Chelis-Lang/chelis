//! Source-only static rate specialization. Admission and lowering share this
//! typed scalar grammar; neither executes a helper or substitutes its arguments.
use super::*;
use std::rc::Rc;

pub(super) fn type_prim(expr: &Expr, substitutions: &UnordMap<String, Prim>) -> Option<Prim> {
    if let Some(prim) = LowerCtx::try_extract_prim(expr) {
        return Some(prim);
    }
    if let Some(name) = formal_param_type_var_name(expr) {
        return substitutions.get(&name).copied();
    }
    let (tag, _, kids) = stamped_parts(expr)?;
    match tag {
        DeepTag::TTensor => type_prim(kids.last()?, substitutions),
        DeepTag::TRef => type_prim(kids.first()?, substitutions),
        _ => None,
    }
}

pub(super) fn scalar(
    expr: &Expr,
    values: &UnordMap<String, StagedScalar>,
    substitutions: &UnordMap<String, Prim>,
    neg_is_builtin: bool,
) -> Option<StagedScalar> {
    let Some((tag, metadata, kids)) = stamped_parts(expr) else {
        return extract_numeric_leaf(expr);
    };
    match tag {
        DeepTag::Var => values.get(symbol_name(kids.first()?)?).copied(),
        DeepTag::Lit => {
            let value = extract_numeric_leaf(expr)?;
            let Some(ty) = metadata.ty() else {
                return Some(value);
            };
            let target = type_prim(ty.expression(), substitutions)?;
            let target = match value {
                StagedScalar::Raw(raw) => {
                    binder_float_literal_source_default(metadata, raw, substitutions)
                        .unwrap_or(target)
                }
                StagedScalar::Typed(_) => target,
            };
            finalize(value, target).map(StagedScalar::Typed)
        }
        DeepTag::Cast => {
            let value = scalar(kids.first()?, values, substitutions, neg_is_builtin)?;
            let source_target = kids.get(1)?;
            // Match lower_cast: a checked result identity outranks a source
            // binder spelling that an outer caller may independently reuse.
            let target = LowerCtx::try_extract_prim(source_target)
                .or_else(|| {
                    formal_param_type_var_name(source_target)
                        .and_then(|_| type_prim(metadata.ty()?.expression(), substitutions))
                })
                .or_else(|| type_prim(source_target, substitutions))?;
            let value = match (chelis_deep::cast_mode_of(kids).ok()?, value) {
                (chelis_deep::CastMode::Checked, value) => finalize(value, target),
                (chelis_deep::CastMode::Trunc, StagedScalar::Raw(raw)) => {
                    chelis_types::cast_trunc_raw("cast_trunc", raw, target).ok()
                }
                (chelis_deep::CastMode::Trunc, StagedScalar::Typed(value)) => {
                    chelis_types::cast_trunc_scalar("cast_trunc", value, target).ok()
                }
            }?;
            Some(StagedScalar::Typed(value))
        }
        DeepTag::App if neg_is_builtin && kids.len() == 2 && expr_is_var_named(&kids[0], "neg") => {
            let value = scalar(&kids[1], values, substitutions, neg_is_builtin)?;
            Some(match value {
                StagedScalar::Typed(value) => StagedScalar::Typed(if value.prim().is_float() {
                    chelis_types::float_unop(chelis_types::FloatUnOp::Neg, value).ok()?
                } else {
                    chelis_types::int_unop(chelis_types::IntUnOp::Neg, value).ok()?
                }),
                StagedScalar::Raw(chelis_types::RawScalar::Int(value)) => {
                    StagedScalar::Raw(chelis_types::RawScalar::Int(value.checked_neg()?))
                }
                StagedScalar::Raw(chelis_types::RawScalar::Float(value)) => {
                    StagedScalar::Raw(chelis_types::RawScalar::Float(-value))
                }
            })
        }
        _ => None,
    }
}

fn finalize(value: StagedScalar, target: Prim) -> Option<chelis_types::ScalarValue> {
    match value {
        StagedScalar::Raw(raw) => chelis_types::cast_raw("cast", raw, target).ok(),
        StagedScalar::Typed(value) => chelis_types::cast_scalar("cast", value, target).ok(),
    }
}

#[derive(Clone, Default)]
struct Environment {
    values: UnordMap<String, StagedScalar>,
    precisions: UnordMap<String, Prim>,
    value_precisions: UnordMap<String, Prim>,
    bound: UnordSet<String>,
    callables: BTreeMap<String, Closure>,
    shadowed_neg: bool,
}

#[derive(Clone)]
struct Closure {
    function: Expr,
    environment: Rc<Environment>,
    name: Option<String>,
}

impl Environment {
    fn scalar(&self, expr: &Expr) -> Option<StagedScalar> {
        scalar(expr, &self.values, &self.precisions, !self.shadowed_neg)
    }

    fn precision(&self, expr: &Expr) -> Option<Prim> {
        bare_var_name(expr)
            .and_then(|name| self.value_precisions.get(&name).copied())
            .or_else(|| type_prim(expr_type_metadata(expr)?, &self.precisions))
    }

    fn bind(&mut self, name: &str, value: Option<StagedScalar>, precision: Option<Prim>) {
        self.shadowed_neg |= name == "neg";
        self.bound.insert(name.to_owned());
        self.values.remove(name);
        self.value_precisions.remove(name);
        self.callables.remove(name);
        if let Some(value) = value {
            self.values.insert(name.to_owned(), value);
        }
        if let Some(precision) = precision {
            self.value_precisions.insert(name.to_owned(), precision);
        }
    }
}

struct Profile<'a> {
    defs: &'a BTreeMap<String, Expr>,
    active: BTreeSet<String>,
    dropout: bool,
    reason: Option<crate::evaluation::LegacyEvaluationReason>,
    resource_policy: crate::evaluation::ResourcePolicy,
}

impl Profile<'_> {
    fn callable(&self, expr: &Expr, env: &Environment) -> Option<Closure> {
        if stamped_parts(expr).is_some_and(|(tag, _, _)| tag == DeepTag::Fn) {
            return Some(Closure {
                function: expr.clone(),
                environment: Rc::new(env.clone()),
                name: None,
            });
        }
        let name = bare_var_name(expr)?;
        if let Some(closure) = env.callables.get(&name) {
            return Some(closure.clone());
        }
        if env.bound.contains(&name) {
            return None;
        }
        let function = self.defs.get(&name)?;
        stamped_parts(function).filter(|(tag, _, _)| *tag == DeepTag::Fn)?;
        Some(Closure {
            function: function.clone(),
            environment: Rc::new(Environment {
                shadowed_neg: self.defs.contains_key("neg"),
                ..Environment::default()
            }),
            name: Some(name),
        })
    }

    fn apply(
        &mut self,
        closure: Closure,
        args: &[Expr],
        caller_depth: usize,
        body_depth: usize,
        caller: &Environment,
    ) {
        use crate::evaluation::LegacyEvaluationReason as Reason;
        for arg in args {
            self.visit(arg, caller_depth, caller);
        }
        if let Some(name) = &closure.name
            && !self.active.insert(name.clone())
        {
            self.reason.get_or_insert(Reason::RecursiveControl);
            return;
        }
        let (_, _, kids) = stamped_parts(&closure.function).expect("resolved function");
        let Some((DeepTag::Params, _, params)) = kids.first().and_then(stamped_parts) else {
            // No parameter environment can be inferred from a malformed
            // function; its owning Deep/lowering boundary diagnoses it.
            return;
        };
        let Some(body) = kids.get(1) else { return };
        let mut env = (*closure.environment).clone();
        // The lowerer has no general lexical closure carrier. A captured
        // scalar with a different caller binding is no longer proven static
        // for this invocation. Forget that fact, rather than rejecting an
        // unrelated closure. An actual rate read (also through nested helper
        // calls/aliases) then declines admission; literals and callee-local
        // bindings retain their own proof. Ordinary value evaluation is not
        // performed or changed by this source-only profile.
        env.values = env
            .values
            .to_sorted()
            .into_iter()
            .filter(|(name, value)| caller.values.get(*name) == Some(*value))
            .map(|(name, value)| (name.clone(), *value))
            .collect();
        let actual_types = args
            .iter()
            .map(|arg| {
                caller.precision(arg).map(|precision| TensorType {
                    dims: vec![],
                    precision,
                })
            })
            .collect::<Option<Vec<_>>>();
        if let Some(actual_types) = actual_types {
            let formals = (0..params.len())
                .map(|index| extract_param_type(&closure.function, index).cloned())
                .collect::<Vec<_>>();
            env.precisions
                .merge(tensor_prec_substitutions(&formals, &actual_types));
            env.precisions.merge(formal_precision_var_bindings(
                &closure.function,
                body,
                &actual_types,
            ));
        }
        for (index, param) in params.iter().enumerate() {
            let mut names = UnordSet::new();
            collect_param_bound_names(param, &mut names);
            for name in names.into_sorted() {
                env.bind(
                    &name,
                    args.get(index).and_then(|arg| caller.scalar(arg)),
                    args.get(index).and_then(|arg| caller.precision(arg)),
                );
            }
        }
        self.visit(body, body_depth, &env);
        if let Some(name) = closure.name {
            self.active.remove(&name);
        }
    }

    fn visit(&mut self, expr: &Expr, depth: usize, env: &Environment) {
        use crate::evaluation::LegacyEvaluationReason as Reason;
        let Some((tag, metadata, kids)) = stamped_parts(expr) else {
            return;
        };
        if tag == DeepTag::App
            && let Some(callee) = kids.first()
        {
            if let Some(closure) = self.callable(callee, env) {
                self.apply(closure, &kids[1..], depth, depth, env);
                return;
            }
            if let Some((DeepTag::Grad, _, grad)) = stamped_parts(callee)
                && let Some(closure) = grad
                    .first()
                    .and_then(|function| self.callable(function, env))
            {
                if depth != 0 {
                    self.reason.get_or_insert(Reason::HigherOrderAd);
                }
                self.apply(closure, &kids[1..], depth, depth + 1, env);
                return;
            }
        }
        if let Some(("dropout", args)) = app_var_name_and_args(expr)
            && !env.bound.contains("dropout")
            && !self.defs.contains_key("dropout")
        {
            self.dropout = true;
            if args.get(1).and_then(|rate| env.scalar(rate)).is_none() {
                self.reason.get_or_insert(Reason::RuntimeRate);
            }
        }
        if tag == DeepTag::Var {
            if let Some(closure) = self.callable(expr, env) {
                self.apply(closure, &[], depth, depth, env);
            } else if let Some(name) = bare_var_name(expr)
                && !env.bound.contains(&name)
                && let Some(body) = self.defs.get(&name)
                && bare_var_name(body).as_ref() != Some(&name)
            {
                // Preserve the original profile's traversal of referenced
                // effecting declarations, not only function definitions.
                if self.active.insert(name.clone()) {
                    self.visit(
                        body,
                        depth,
                        &Environment {
                            shadowed_neg: self.defs.contains_key("neg"),
                            ..Environment::default()
                        },
                    );
                    self.active.remove(&name);
                } else {
                    self.reason.get_or_insert(Reason::RecursiveControl);
                }
            }
            return;
        }
        if tag == DeepTag::Fn {
            let mut scoped = env.clone();
            if let Some((_, _, params)) = kids.first().and_then(stamped_parts) {
                for param in params {
                    let mut names = UnordSet::new();
                    collect_param_bound_names(param, &mut names);
                    for name in names.into_sorted() {
                        scoped.bind(&name, None, None);
                    }
                }
            }
            if let Some(body) = kids.get(1) {
                self.visit(body, depth, &scoped);
            }
            return;
        }
        if tag == DeepTag::Let {
            let mut scoped = env.clone();
            if let Some((DeepTag::Bind, _, bindings)) = kids.first().and_then(stamped_parts) {
                for pair in bindings.as_chunks::<2>().0 {
                    let callable = self.callable(&pair[1], &scoped);
                    if callable.is_none() {
                        self.visit(&pair[1], depth, &scoped);
                    }
                    let value = scoped.scalar(&pair[1]);
                    let precision = scoped.precision(&pair[1]);
                    if let Some(name) = symbol_name(&pair[0]) {
                        scoped.bind(name, value, precision);
                        if let Some(callable) = callable {
                            scoped.callables.insert(name.to_owned(), callable);
                        }
                    }
                }
            }
            if let Some(body) = kids.get(1) {
                self.visit(body, depth, &scoped);
            }
            return;
        }
        let next_depth = match tag {
            DeepTag::Grad => {
                if depth != 0 {
                    self.reason.get_or_insert(Reason::HigherOrderAd);
                }
                depth + 1
            }
            DeepTag::Vmap => {
                self.reason.get_or_insert(Reason::RandomVmap);
                depth
            }
            DeepTag::HandleEffect => {
                let resource = metadata
                    .effect()
                    .is_some_and(|effect| *effect.value() == EffectKind::Resource);
                if resource {
                    if self.resource_policy == crate::evaluation::ResourcePolicy::Legacy {
                        self.reason.get_or_insert(Reason::ResourceScope);
                    }
                } else if kids.first().and_then(extract_numeric_leaf).is_none() {
                    self.reason.get_or_insert(Reason::RuntimeSeed);
                }
                depth
            }
            DeepTag::If => {
                let branch = match kids.first().and_then(extract_numeric_leaf) {
                    Some(StagedScalar::Raw(chelis_types::RawScalar::Int(value))) => {
                        Some(value != 0)
                    }
                    Some(StagedScalar::Typed(value)) if value.prim() == Prim::Bool => {
                        value.as_i64_exact().map(|value| value != 0)
                    }
                    _ => None,
                };
                if let Some(branch) = branch {
                    if let Some(selected) = kids.get(if branch { 1 } else { 2 }) {
                        self.visit(selected, depth, env);
                    }
                    return;
                }
                self.reason.get_or_insert(Reason::DynamicControl);
                depth
            }
            DeepTag::Match => {
                self.reason.get_or_insert(Reason::DynamicControl);
                depth
            }
            _ => depth,
        };
        for child in kids {
            self.visit(child, next_depth, env);
        }
    }
}

pub(super) fn profile(
    expr: &Expr,
    defs: &BTreeMap<String, Expr>,
    inputs: &[(String, TensorType)],
    resource_policy: crate::evaluation::ResourcePolicy,
) -> crate::evaluation::EvaluationProfile {
    use crate::evaluation::{EvaluationProfile, LegacyEvaluationReason};
    let mut profile = Profile {
        defs,
        active: BTreeSet::new(),
        dropout: false,
        reason: None,
        resource_policy,
    };
    let mut env = Environment {
        shadowed_neg: defs.contains_key("neg"),
        ..Environment::default()
    };
    for (name, ty) in inputs {
        env.bind(name, None, Some(ty.precision));
    }
    profile.visit(expr, 0, &env);
    if let Some(reason) = profile.reason {
        EvaluationProfile::Legacy(reason)
    } else if profile.dropout {
        EvaluationProfile::FixedControl
    } else {
        EvaluationProfile::Legacy(LegacyEvaluationReason::NoDropout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_static_control_preserves_referenced_declaration_profile() {
        use crate::evaluation::{EvaluationProfile, LegacyEvaluationReason};
        let reference = chelis_deep::parser::parse_str("(var {} draw)")
            .unwrap()
            .pop()
            .unwrap();
        for (rate, expected) in [
            ("(lit {} 0.5)", EvaluationProfile::FixedControl),
            (
                "(var {} runtime)",
                EvaluationProfile::Legacy(LegacyEvaluationReason::RuntimeRate),
            ),
        ] {
            let body = chelis_deep::parser::parse_str(&format!(
                "(app {{}} (var {{}} dropout) (var {{}} x) {rate})"
            ))
            .unwrap()
            .pop()
            .unwrap();
            assert_eq!(
                profile(
                    &reference,
                    &BTreeMap::from([("draw".into(), body)]),
                    &[],
                    crate::evaluation::ResourcePolicy::Legacy,
                ),
                expected
            );
        }
    }

    #[test]
    fn typed_static_control_prefers_checked_identity_and_rejects_unresolved_target() {
        let expr = chelis_deep::parser::parse_str(
            "(cast {type: (t-var {} result)} (lit {type: (t-prim {} f32)} 0.1) (t-var {} p))",
        )
        .unwrap()
        .pop()
        .unwrap();
        let mut substitutions = UnordMap::new();
        substitutions.insert("p".into(), Prim::F32);
        substitutions.insert("result".into(), Prim::F64);
        let Some(StagedScalar::Typed(value)) =
            scalar(&expr, &UnordMap::new(), &substitutions, true)
        else {
            panic!("typed cast")
        };
        assert_eq!(value.prim(), Prim::F64);
        assert_eq!(value.as_f64_lossy(), f64::from(0.1f32));
        assert!(scalar(&expr, &UnordMap::new(), &UnordMap::new(), true).is_none());
    }

    #[test]
    fn typed_static_control_does_not_execute_a_shadowed_neg() {
        let expr = chelis_deep::parser::parse_str(
            "(app {} (var {} neg) (lit {type: (t-prim {} f32)} -0.5))",
        )
        .unwrap()
        .pop()
        .unwrap();
        assert!(scalar(&expr, &UnordMap::new(), &UnordMap::new(), true).is_some());
        assert!(scalar(&expr, &UnordMap::new(), &UnordMap::new(), false).is_none());
    }
}
