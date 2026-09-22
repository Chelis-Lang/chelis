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

/// What a named definition's walk read from its caller: the definition, each
/// actual's static scalar and precision, and whether the body sits under a
/// `grad` (only zero versus nonzero depth is observable, through
/// `HigherOrderAd`). A named definition starts from a fresh environment, so
/// with the fixed definition table and resource policy these are every input
/// its walk has (chelis#2391).
struct WalkKey {
    name: String,
    under_grad: bool,
    scalars: Vec<Option<StagedScalar>>,
    precisions: Vec<Option<Prim>>,
}

impl WalkKey {
    fn same_walk(&self, other: &Self) -> bool {
        self.name == other.name
            && self.under_grad == other.under_grad
            && self.precisions == other.precisions
            && self.scalars.len() == other.scalars.len()
            && self
                .scalars
                .iter()
                .zip(&other.scalars)
                .all(|(lhs, rhs)| match (lhs, rhs) {
                    (None, None) => true,
                    (Some(lhs), Some(rhs)) => identical_scalar(lhs, rhs),
                    _ => false,
                })
    }
}

/// Bit-level identity of two staged scalars. A NaN is never identical to
/// anything, so a NaN actual always re-walks: a missed replay costs time,
/// a wrong one would change a classification.
fn identical_scalar(lhs: &StagedScalar, rhs: &StagedScalar) -> bool {
    match (lhs, rhs) {
        (
            StagedScalar::Raw(chelis_types::RawScalar::Int(lhs)),
            StagedScalar::Raw(chelis_types::RawScalar::Int(rhs)),
        ) => lhs == rhs,
        (
            StagedScalar::Raw(chelis_types::RawScalar::Float(lhs)),
            StagedScalar::Raw(chelis_types::RawScalar::Float(rhs)),
        ) => !lhs.is_nan() && lhs.to_bits() == rhs.to_bits(),
        (StagedScalar::Typed(lhs), StagedScalar::Typed(rhs)) => {
            lhs.prim() == rhs.prim()
                && !lhs.as_f64_lossy().is_nan()
                && lhs.as_i64_exact() == rhs.as_i64_exact()
                && lhs.as_f64_lossy().to_bits() == rhs.as_f64_lossy().to_bits()
        }
        _ => false,
    }
}

struct Profile<'a> {
    defs: &'a BTreeMap<String, Expr>,
    /// Names the caller's bindings shadow. They read as absent from `defs`
    /// everywhere in the walk, callee bodies included, exactly as a copy of
    /// `defs` with these names removed would (chelis#2391 retired that copy).
    excluded: &'a UnordSet<String>,
    active: BTreeSet<String>,
    dropout: bool,
    reason: Option<crate::evaluation::LegacyEvaluationReason>,
    resource_policy: crate::evaluation::ResourcePolicy,
    /// Named-definition walks that finished without a reason, with the
    /// dropout they found. Once any reason is recorded the profile's answer
    /// is fixed and the walk stops, so a walk that finishes reason-free met
    /// no recursion edge (each one records `RecursiveControl`) and its result
    /// does not depend on the `active` set. Replaying it is therefore exact,
    /// and it turns the per-call-site re-walk of a shared callee, which
    /// multiplied along every call chain, into one walk per distinct key.
    completed: UnordMap<String, Vec<(WalkKey, bool)>>,
    /// Set once a walk leaves a name in `active` (see `poison_replay`).
    replay_poisoned: bool,
}

#[cfg(test)]
thread_local! {
    /// Classifier invocations (`profile_excluding` calls).
    static PROFILE_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Named-definition body walks actually performed (not replayed).
    static BODY_WALKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// Disable replay and the early stop, so tests can compare against the
    /// walk as it was before chelis#2391.
    static REFERENCE_WALK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn reference_walk() -> bool {
    REFERENCE_WALK.with(std::cell::Cell::get)
}

#[cfg(not(test))]
fn reference_walk() -> bool {
    false
}

impl<'a> Profile<'a> {
    fn def(&self, name: &str) -> Option<&'a Expr> {
        if self.excluded.contains(name) {
            return None;
        }
        self.defs.get(name)
    }

    fn has_def(&self, name: &str) -> bool {
        self.def(name).is_some()
    }

    fn fresh_environment(&self) -> Environment {
        Environment {
            shadowed_neg: self.has_def("neg"),
            ..Environment::default()
        }
    }

    fn replay(&mut self, key: &WalkKey) -> bool {
        #[cfg(test)]
        if REFERENCE_WALK.with(std::cell::Cell::get) {
            return false;
        }
        let Some(dropout) = self.completed.get(&key.name).and_then(|done| {
            done.iter()
                .find_map(|(done, dropout)| done.same_walk(key).then_some(*dropout))
        }) else {
            return false;
        };
        self.dropout |= dropout;
        true
    }

    fn poison_replay(&mut self) {
        self.completed.clear();
        self.replay_poisoned = true;
    }

    /// Walk a named definition's body at most once per key.
    fn walk_named(&mut self, key: WalkKey, walk: impl FnOnce(&mut Self)) {
        if self.replay(&key) {
            return;
        }
        #[cfg(test)]
        BODY_WALKS.with(|walks| walks.set(walks.get() + 1));
        let before = self.dropout;
        self.dropout = false;
        walk(self);
        let found = self.dropout;
        self.dropout |= before;
        if self.reason.is_none() && !self.replay_poisoned {
            self.completed
                .entry(key.name.clone())
                .or_default()
                .push((key, found));
        }
    }

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
        let function = self.def(&name)?;
        stamped_parts(function).filter(|(tag, _, _)| *tag == DeepTag::Fn)?;
        Some(Closure {
            function: function.clone(),
            environment: Rc::new(self.fresh_environment()),
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
        if self.reason.is_some() && !reference_walk() {
            return;
        }
        let Some(name) = closure.name.clone() else {
            self.apply_body(&closure, args, body_depth, caller);
            return;
        };
        if !self.active.insert(name.clone()) {
            self.reason.get_or_insert(Reason::RecursiveControl);
            return;
        }
        let key = WalkKey {
            name: name.clone(),
            under_grad: body_depth != 0,
            scalars: args.iter().map(|arg| caller.scalar(arg)).collect(),
            precisions: args.iter().map(|arg| caller.precision(arg)).collect(),
        };
        // A replayed walk was well formed when it was recorded.
        let mut walked = true;
        self.walk_named(key, |profile| {
            walked = profile.apply_body(&closure, args, body_depth, caller);
        });
        if walked {
            self.active.remove(&name);
        } else {
            // A malformed named function stays in the active set, so every
            // later reference to it reads as recursion (the profile has always
            // behaved this way). Walks recorded before that point assumed it was
            // inactive, so none of them may be replayed from here on.
            self.poison_replay();
        }
    }

    fn apply_body(
        &mut self,
        closure: &Closure,
        args: &[Expr],
        body_depth: usize,
        caller: &Environment,
    ) -> bool {
        let (_, _, kids) = stamped_parts(&closure.function).expect("resolved function");
        let Some((DeepTag::Params, _, params)) = kids.first().and_then(stamped_parts) else {
            // No parameter environment can be inferred from a malformed
            // function; its owning Deep/lowering boundary diagnoses it.
            return false;
        };
        let Some(body) = kids.get(1) else {
            return false;
        };
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
        true
    }

    fn visit(&mut self, expr: &Expr, depth: usize, env: &Environment) {
        use crate::evaluation::LegacyEvaluationReason as Reason;
        // The first recorded reason is the profile's answer (`profile` reads
        // it before `dropout`, and every write is `get_or_insert`), so nothing
        // the rest of the walk could find changes the result.
        if self.reason.is_some() && !reference_walk() {
            return;
        }
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
            && !self.has_def("dropout")
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
                && let Some(body) = self.def(&name)
                && bare_var_name(body).as_ref() != Some(&name)
            {
                // Preserve the original profile's traversal of referenced
                // effecting declarations, not only function definitions.
                if self.active.insert(name.clone()) {
                    let key = WalkKey {
                        name: name.clone(),
                        under_grad: depth != 0,
                        scalars: Vec::new(),
                        precisions: Vec::new(),
                    };
                    let environment = self.fresh_environment();
                    self.walk_named(key, |profile| profile.visit(body, depth, &environment));
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
    profile_excluding(expr, defs, &UnordSet::new(), inputs, resource_policy)
}

/// [`profile`] over `defs` with every name in `excluded` treated as absent
/// from the table, which is what a caller whose bindings shadow those
/// definitions needs, without copying the table per ask.
pub(super) fn profile_excluding(
    expr: &Expr,
    defs: &BTreeMap<String, Expr>,
    excluded: &UnordSet<String>,
    inputs: &[(String, TensorType)],
    resource_policy: crate::evaluation::ResourcePolicy,
) -> crate::evaluation::EvaluationProfile {
    use crate::evaluation::{EvaluationProfile, LegacyEvaluationReason};
    #[cfg(test)]
    PROFILE_CALLS.with(|calls| calls.set(calls.get() + 1));
    let mut profile = Profile {
        defs,
        excluded,
        active: BTreeSet::new(),
        dropout: false,
        reason: None,
        resource_policy,
        completed: UnordMap::new(),
        replay_poisoned: false,
    };
    let mut env = profile.fresh_environment();
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

    fn surf_program(source: &str) -> Option<Vec<Expr>> {
        let declarations = chelis_surf::parser::parse_str(source).ok()?;
        let desugared = chelis_surf::desugar::desugar_program(&declarations).ok()?;
        // A checked program carries the type metadata the precision facts
        // read; a file that does not check alone (an import, a library
        // module) still exercises the walk's control structure unchecked.
        Some(match chelis_types::check_typed_program(&desugared) {
            Ok(checked) => checked.exprs().to_vec(),
            Err(_) => desugared,
        })
    }

    /// A handcrafted fixture must check, so its actuals carry the precision
    /// facts a replay key compares.
    fn surf_defs(source: &str) -> BTreeMap<String, Expr> {
        let declarations = chelis_surf::parser::parse_str(source).expect("fixture parses");
        let desugared =
            chelis_surf::desugar::desugar_program(&declarations).expect("fixture desugars");
        let checked = chelis_types::check_typed_program(&desugared)
            .unwrap_or_else(|error| panic!("fixture checks: {error:?}"));
        collect_top_level_defs(checked.exprs())
    }

    fn as_reference<T>(run: impl FnOnce() -> T) -> T {
        REFERENCE_WALK.with(|flag| flag.set(true));
        let result = run();
        REFERENCE_WALK.with(|flag| flag.set(false));
        result
    }

    fn body_walks<T>(run: impl FnOnce() -> T) -> (T, usize) {
        BODY_WALKS.with(|walks| walks.set(0));
        let result = run();
        (result, BODY_WALKS.with(std::cell::Cell::get))
    }

    fn legacy_profile(
        expr: &Expr,
        defs: &BTreeMap<String, Expr>,
    ) -> crate::evaluation::EvaluationProfile {
        profile(expr, defs, &[], crate::evaluation::ResourcePolicy::Legacy)
    }

    fn collect_apps<'e>(expr: &'e Expr, out: &mut Vec<&'e Expr>) {
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            return;
        };
        if tag == DeepTag::App {
            out.push(expr);
        }
        for kid in kids {
            collect_apps(kid, out);
        }
    }

    /// Handcrafted programs aimed at each way a replayed walk could differ
    /// from a fresh one. Each one also runs through the corpus differential.
    /// Deep-only fixtures for shapes Surf cannot spell. `m` has no
    /// parameter list, so the walk leaves it in the active set; `n` is then
    /// recorded before the second call to `n` re-reaches `m`, which the
    /// original walk reports as recursion. Replay must not skip that second
    /// walk (the round-1 red team's reproduction for `poison_replay`).
    const DEEP_EXACTNESS_FIXTURES: &[&str] = &["(def {} m (fn {} (lit {} 1) (lit {} 2)))\n\
         (def {} n (fn {} (params {} (x {})) (app {} (var {} m) (var {} x))))\n\
         (def {} root (fn {} (params {} (x {})) (app {} (var {} add) (app {} (var {} n) (var {} x)) (app {} (var {} n) (var {} x)))))\n"];

    const EXACTNESS_FIXTURES: &[&str] = &[
        // The same helper with a static and then a runtime rate: the key
        // must tell the two actuals apart, in either order.
        "def thin(x: tensor[4, f32], r: f32) -> tensor[4, f32] ! { Random } = dropout(x, r)\n\
         def static_first(x: tensor[4, f32], y: f32) -> tensor[4, f32] ! { Random } = add(thin(x, 0.5f32), thin(x, y))\n\
         def runtime_first(x: tensor[4, f32], y: f32) -> tensor[4, f32] ! { Random } = add(thin(x, y), thin(x, 0.5f32))\n\
         def static_twice(x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = add(thin(x, 0.5f32), thin(x, 0.25f32))\n\
         def typed_runtime(x: tensor[4, f32], y: f32) -> tensor[4, f32] ! { Random } = add(thin(x, 0.5f32), thin(x, mul(y, 1.0f32)))\n\
         def narrow(x: tensor[4, f32], n: i32) -> tensor[4, f32] ! { Random } = dropout(x, cast(cast(n, i8), f32))\n\
         def fits_then_overflows(x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = add(narrow(x, 1i32), narrow(x, 1000i32))\n\
         def keep_generic[p: Float](x: tensor[4, p]) -> tensor[4, p] ! { Random } = dropout(x, cast(0.5, p))\n\
         def typed_then_untyped(x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = add(keep_generic(add(x, x)), keep_generic(x))\n",
        // Recursion reached after a sibling helper was walked and recorded.
        "def leaf(x: tensor[f32]) -> tensor[f32] = add(x, x)\n\
         def ping(x: tensor[f32]) -> tensor[f32] = add(leaf(x), pong(x))\n\
         def pong(x: tensor[f32]) -> tensor[f32] = ping(leaf(x))\n",
        // A helper holding a `grad`, reached first at depth zero and then
        // under an outer `grad`, where the same body is higher-order AD.
        "def inner(x: tensor[f32]) -> tensor[f32] = mul(x, x)\n\
         def uses_grad(x: tensor[f32]) -> tensor[f32] = grad(inner)(x)\n\
         def outer(x: tensor[f32]) -> tensor[f32] = add(uses_grad(x), grad(uses_grad)(x))\n",
        // Shadowing: a definition named like a caller binding, and `neg`.
        "def neg(x: f32) -> f32 = x\n\
         def rate() -> f32 = 0.5f32\n\
         def scaled(x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = dropout(x, neg(0.5f32))\n\
         def reads(x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = add(scaled(x), dropout(x, rate()))\n",
    ];

    /// chelis#2391: replay and the early stop are optimisations, so every
    /// classification must equal the walk they replaced, and the named
    /// exclusion must equal the filtered table copy it replaced.
    #[test]
    fn memoised_profile_matches_the_reference_walk_over_the_corpus() {
        use crate::evaluation::{EvaluationProfile, ResourcePolicy};
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut sources = EXACTNESS_FIXTURES
            .iter()
            .map(|source| (format!("fixture:{}", &source[..24]), (*source).to_owned()))
            .collect::<Vec<_>>();
        let mut stack = [
            "examples",
            "packages/chelis-std",
            "tests/corpus",
            "crates/chelis-cli/tests/fixtures",
        ]
        .map(|dir| root.join(dir))
        .to_vec();
        while let Some(dir) = stack.pop() {
            let mut entries = std::fs::read_dir(&dir)
                .unwrap_or_else(|error| panic!("{}: {error}", dir.display()))
                .map(|entry| entry.unwrap().path())
                .collect::<Vec<_>>();
            entries.sort();
            for path in entries {
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|ext| ext == "ch") {
                    sources.push((
                        path.display().to_string(),
                        std::fs::read_to_string(&path).unwrap(),
                    ));
                }
            }
        }
        let mut programs = DEEP_EXACTNESS_FIXTURES
            .iter()
            .map(|source| {
                let exprs = chelis_deep::parser::parse_str(source).expect("Deep fixture parses");
                (format!("deep fixture:{}", &source[..24]), exprs)
            })
            .collect::<Vec<_>>();
        programs.extend(
            sources
                .iter()
                .filter_map(|(origin, source)| Some((origin.clone(), surf_program(source)?))),
        );
        let mut compared = 0usize;
        let mut outcomes = BTreeMap::<String, usize>::new();
        for (origin, exprs) in &programs {
            let defs = collect_top_level_defs(exprs);
            let names = defs.keys().cloned().collect::<Vec<_>>();
            let mut shadowed = names.iter().step_by(3).cloned().collect::<UnordSet<_>>();
            shadowed.insert("neg".into());
            shadowed.insert("dropout".into());
            let filtered = defs
                .iter()
                .filter(|(name, _)| !shadowed.contains(*name))
                .map(|(name, body)| (name.clone(), body.clone()))
                .collect::<BTreeMap<_, _>>();
            let mut asks = defs.values().collect::<Vec<_>>();
            for body in defs.values() {
                collect_apps(body, &mut asks);
            }
            for expr in asks {
                for policy in [ResourcePolicy::Legacy, ResourcePolicy::RecordRequirements] {
                    let memoised = profile(expr, &defs, &[], policy);
                    let reference = as_reference(|| profile(expr, &defs, &[], policy));
                    assert_eq!(memoised, reference, "{origin}: {:?}", expr.span());
                    let excluded = profile_excluding(expr, &defs, &shadowed, &[], policy);
                    let copied = as_reference(|| profile(expr, &filtered, &[], policy));
                    assert_eq!(excluded, copied, "{origin} (shadowed): {:?}", expr.span());
                    compared += 1;
                    let label = match memoised {
                        EvaluationProfile::Legacy(reason) => format!("{reason:?}"),
                        other => format!("{other:?}"),
                    };
                    *outcomes.entry(label).or_default() += 1;
                }
            }
        }
        // A corpus that silently stopped parsing would compare nothing and
        // pass; require volume and every outcome the fixtures aim at.
        assert!(compared > 2_000, "compared only {compared} classifications");
        for expected in [
            "FixedControl",
            "RuntimeRate",
            "RecursiveControl",
            "HigherOrderAd",
            "NoDropout",
        ] {
            assert!(
                outcomes.get(expected).is_some_and(|count| *count > 0),
                "no {expected} classification in the corpus: {outcomes:?}"
            );
        }
    }

    #[test]
    fn replay_key_distinguishes_a_static_rate_from_a_runtime_one() {
        use crate::evaluation::{EvaluationProfile, LegacyEvaluationReason};
        let defs = surf_defs(EXACTNESS_FIXTURES[0]);
        for (name, expected) in [
            (
                "static_first",
                EvaluationProfile::Legacy(LegacyEvaluationReason::RuntimeRate),
            ),
            (
                "runtime_first",
                EvaluationProfile::Legacy(LegacyEvaluationReason::RuntimeRate),
            ),
            ("static_twice", EvaluationProfile::FixedControl),
            (
                "typed_runtime",
                EvaluationProfile::Legacy(LegacyEvaluationReason::RuntimeRate),
            ),
            (
                "fits_then_overflows",
                EvaluationProfile::Legacy(LegacyEvaluationReason::RuntimeRate),
            ),
            (
                "typed_then_untyped",
                EvaluationProfile::Legacy(LegacyEvaluationReason::RuntimeRate),
            ),
        ] {
            assert_eq!(legacy_profile(&defs[name], &defs), expected, "{name}");
        }
    }

    /// chelis#2391: a helper shared along a call chain is walked once per
    /// distinct key, not once per path. `h10` reaches `h0` along 2^10 paths.
    #[test]
    fn shared_callee_is_walked_once_per_key() {
        use crate::evaluation::{EvaluationProfile, LegacyEvaluationReason};
        let mut source = String::from("def h0(x: tensor[f32]) -> tensor[f32] = add(x, x)\n");
        for level in 1..=10 {
            let below = level - 1;
            source.push_str(&format!(
                "def h{level}(x: tensor[f32]) -> tensor[f32] = add(h{below}(x), h{below}(x))\n"
            ));
        }
        let defs = surf_defs(&source);
        let (memoised, walks) = body_walks(|| legacy_profile(&defs["h10"], &defs));
        let (reference, reference_walks) =
            body_walks(|| as_reference(|| legacy_profile(&defs["h10"], &defs)));
        assert_eq!(
            memoised,
            EvaluationProfile::Legacy(LegacyEvaluationReason::NoDropout)
        );
        assert_eq!(memoised, reference);
        assert_eq!(
            reference_walks,
            (1 << 11) - 2,
            "the unmemoised walk is exponential"
        );
        assert_eq!(walks, 10, "each of h0..h9 is walked once");
    }

    /// The first reason fixes the answer, so a recursive program stops at
    /// its first recursion edge instead of exhausting the call graph.
    #[test]
    fn walk_stops_at_the_first_reason() {
        use crate::evaluation::{EvaluationProfile, LegacyEvaluationReason};
        let mut source = String::from(
            "def spin(x: tensor[f32]) -> tensor[f32] = spin(x)\n\
             def h0(x: tensor[f32]) -> tensor[f32] = add(x, x)\n",
        );
        for level in 1..=10 {
            let below = level - 1;
            source.push_str(&format!(
                "def h{level}(x: tensor[f32]) -> tensor[f32] = add(h{below}(x), h{below}(x))\n"
            ));
        }
        source.push_str("def root(x: tensor[f32]) -> tensor[f32] = add(spin(x), h10(x))\n");
        let defs = surf_defs(&source);
        let (memoised, walks) = body_walks(|| legacy_profile(&defs["root"], &defs));
        assert_eq!(
            memoised,
            EvaluationProfile::Legacy(LegacyEvaluationReason::RecursiveControl)
        );
        assert_eq!(walks, 1, "only `spin` is entered");
    }

    /// chelis#2391: `def_is_lowered` and every call site of a definition ask
    /// for the same body profile; one lowering-map computation profiles each
    /// definition at most once. Before, every one of the `callers * calls`
    /// call sites re-profiled `shared` and walked its whole call graph.
    #[test]
    fn lowering_map_profiles_each_definition_once() {
        let callers = 20;
        let calls = 5;
        let mut source = String::from("def shared(x: tensor[f32]) -> tensor[f32] = add(x, x)\n");
        for caller in 0..callers {
            let body = (0..calls).fold("x".to_owned(), |acc, _| format!("shared({acc})"));
            source.push_str(&format!(
                "def caller{caller}(x: tensor[f32]) -> tensor[f32] = {body}\n"
            ));
        }
        let declarations = chelis_surf::parser::parse_str(&source).expect("fixture parses");
        let desugared =
            chelis_surf::desugar::desugar_program(&declarations).expect("fixture desugars");
        let checked = chelis_types::check_typed_program(&desugared).expect("fixture checks");
        PROFILE_CALLS.with(|count| count.set(0));
        let lowered = top_level_lowering_map(checked.exprs(), checked.type_env());
        let asked = PROFILE_CALLS.with(std::cell::Cell::get);
        assert_eq!(lowered.len(), callers + 1);
        assert!(
            asked <= callers + 1,
            "{asked} classifier calls for {} definitions",
            callers + 1
        );
        assert!(asked > 0, "the lowering map no longer asks the classifier");
    }

    /// A malformed named function stays in the active set for the rest of
    /// the walk, so a walk recorded before it was reached must not be
    /// replayed afterwards.
    #[test]
    fn malformed_named_function_disables_replay() {
        use crate::evaluation::{EvaluationProfile, LegacyEvaluationReason};
        let exprs = chelis_deep::parser::parse_str(DEEP_EXACTNESS_FIXTURES[0]).unwrap();
        let defs = collect_top_level_defs(&exprs);
        let memoised = legacy_profile(&defs["root"], &defs);
        let reference = as_reference(|| legacy_profile(&defs["root"], &defs));
        assert_eq!(
            reference,
            EvaluationProfile::Legacy(LegacyEvaluationReason::RecursiveControl)
        );
        assert_eq!(memoised, reference);
    }
}
