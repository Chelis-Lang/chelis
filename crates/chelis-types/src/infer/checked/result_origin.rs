//! Result equations and their origin across generalization and application.

use super::*;

impl InferenceProduct {
    /// An ascription may check a suspended derivation's output only after its
    /// rule has settled. Its dependency is the published type, not a
    /// list of parameter holes read before the producer's constructor is known.
    /// A closed first-order value annotation can settle ordinary element/dtype
    /// holes immediately. It cannot hide a callable parameter, and a pending
    /// semantic producer still prevents result-to-input inference.
    /// Projection and alias transport retain that dependency through ordinary
    /// type unification. Each replay reads it anew from the remaining producers.
    pub(in crate::infer) fn defer_result_type_constraint(
        &mut self,
        actual: &Type,
        declared: &Type,
        subst: &mut Subst,
    ) -> bool {
        // Reject an incompatible annotation through its owning syntax boundary,
        // before a deferred replay could replace its diagnostic.
        if unify(actual, declared, &mut subst.clone()).is_err() {
            return false;
        }
        let mut ready_parts = subst.clone();
        let Ok(ready) = self.unify_ready_result_parts(actual, declared, &mut ready_parts) else {
            // The annotation owner reports the failed equality against the
            // unchanged substitution, just as for the full compatibility check.
            return false;
        };
        // A speculative substitution deliberately clears transient refinement
        // receipts. Commit the checked equalities to the live substitution,
        // retaining its receipts for the surrounding expression/contract walk.
        self.unify_ready_result_parts(actual, declared, subst)
            .expect("ready annotation parts were checked against this substitution");
        if ready {
            return false;
        }
        self.result_type_constraints.push(ResultTypeConstraint {
            actual: actual.clone(),
            declared: declared.clone(),
        });
        true
    }

    /// Equality is ready per structural slot. A callable can have an ordinary
    /// annotated parameter beside a result whose semantic producer still waits.
    /// Its result equation must not turn those independent slots into one wait.
    fn unify_ready_result_parts(
        &self,
        actual: &Type,
        declared: &Type,
        subst: &mut Subst,
    ) -> Result<bool, TypeError> {
        let actual = resolved(actual, subst);
        let declared = resolved(declared, subst);
        if actual == declared {
            return Ok(true);
        }
        let pairs: Option<Vec<(&Type, &Type)>> = match (&actual, &declared) {
            (Type::Fn(a, ar), Type::Fn(b, br)) if a.len() == b.len() => Some(
                a.iter()
                    .zip(b)
                    .chain(std::iter::once((ar.as_ref(), br.as_ref())))
                    .collect(),
            ),
            (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => {
                Some(a.iter().zip(b).collect())
            }
            (Type::Adt(a, aa), Type::Adt(b, ba)) if a == b && aa.len() == ba.len() => {
                Some(aa.iter().zip(ba).collect())
            }
            (Type::Ref(a), Type::Ref(b)) => Some(vec![(a.as_ref(), b.as_ref())]),
            _ => None,
        };
        if let Some(pairs) = pairs {
            let mut ready = true;
            for (actual, declared) in pairs {
                ready &= self.unify_ready_result_parts(actual, declared, subst)?;
            }
            return Ok(ready);
        }
        if !self.result_has_pending_producer(&actual, subst)
            && !self.result_has_pending_producer(&declared, subst)
            && (crate::env::free_tvars(&actual).is_empty() || is_closed_first_order(&declared))
        {
            unify(&actual, &declared, subst)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn result_has_pending_producer(&self, actual: &Type, subst: &Subst) -> bool {
        self.result_has_pending_producer_with(actual, subst, false)
    }

    fn result_has_open_inputs(&self, ty: &Type, subst: &Subst) -> bool {
        !crate::env::free_tvars(&resolved(ty, subst)).is_empty()
    }

    pub(in crate::infer) fn import_result_constraints(&mut self, subst: &Subst) -> bool {
        let equations = subst.take_result_constraints();
        let imported = !equations.is_empty();
        for equation in equations {
            match equation {
                ResultConstraint::Annotation { actual, declared } => self
                    .result_type_constraints
                    .push(ResultTypeConstraint { actual, declared }),
                ResultConstraint::Join { inputs, result } => self.defer_shape_check(
                    DeferredShapeRule::ResultJoin {
                        last_propagated: None,
                    },
                    Vec::new(),
                    inputs,
                    result,
                ),
            }
        }
        imported
    }

    /// Keep the equations that determined a helper's inputs alongside its
    /// conventional solved signature. They are instantiated before each use,
    /// even when no semantic producer existed when the helper was inferred.
    pub(in crate::infer) fn generalize_result_origins(
        &mut self,
        ty: &Type,
        env: &Env,
        subst: &Subst,
        owned_contracts: Option<&[CollectionContractId]>,
        errors: &mut DiagnosticSink<'_>,
    ) -> Scheme {
        self.import_result_constraints(subst);
        let body = resolved(ty, subst);
        let equations = self.result_equations_for(&body, subst);
        let generalize = |ty: &Type| {
            env.generalize_with_result_constraints(ty, subst, owned_contracts, &equations)
        };
        if equations.is_empty() {
            return generalize(&body);
        }
        let carried = Type::Tuple(
            std::iter::once(body.clone())
                .chain(
                    equations
                        .iter()
                        .flat_map(|equation| equation.types())
                        .cloned(),
                )
                .collect(),
        );
        let mut scheme = generalize(&carried);
        let origin = ResultOrigin {
            body,
            tvars: scheme.tvars.clone(),
            tvar_restrictions: scheme.tvar_restrictions.clone(),
            dvars: scheme.dvars.clone(),
            rvars: scheme.rvars.clone(),
            equations,
        };
        let mut trial = subst.clone();
        for equation in &origin.equations {
            match equation {
                ResultConstraint::Annotation { actual, declared } => {
                    if let Err(error) = unify(actual, declared, &mut trial) {
                        errors.push(error.into());
                    }
                }
                ResultConstraint::Join { inputs, result } => {
                    for input in inputs {
                        if let Err(error) = unify(result, input, &mut trial) {
                            errors.push(error.into());
                        }
                    }
                }
            }
        }
        scheme.body = trial.apply(&origin.body);
        let mut normalized_contracts = Vec::new();
        scheme.constraints.retain(|constraint| {
            let normalized = constraint.map_types(|ty| trial.apply(ty));
            if normalized_contracts.contains(&normalized) {
                false
            } else {
                normalized_contracts.push(normalized);
                true
            }
        });
        let free = crate::env::free_tvars(&scheme.body);
        scheme.tvars = origin
            .tvars
            .iter()
            .copied()
            .filter(|variable| free.contains(variable))
            .collect();
        scheme.tvar_restrictions = scheme
            .tvars
            .iter()
            .filter_map(|variable| {
                trial
                    .tvar_restriction(*variable)
                    .map(|restriction| (*variable, restriction))
            })
            .collect();
        let closed_key_variables = crate::env::closed_key_relation_variables(&scheme);
        for variable in &scheme.tvars {
            if !closed_key_variables.contains(variable) {
                subst
                    .forbid_key_instantiation(*variable, crate::unify::GenericParameter::default());
            }
        }
        scheme.result_origin = Some(origin);
        scheme
    }

    pub(in crate::infer) fn origin_component(&self) -> Self {
        Self {
            next_deferred_shape_id: self.next_deferred_shape_id,
            deferred_shape_checks: self.deferred_shape_checks.clone(),
            result_type_constraints: self.result_type_constraints.clone(),
            closing_group_types: self.closing_group_types.clone(),
            ..Self::default()
        }
    }

    /// Callable alternatives denoted by a joined result. This transports a
    /// checked operation contract without identifying its result variables
    /// with an unresolved gradient's inputs.
    pub(in crate::infer) fn callable_result_alternatives(
        &self,
        ty: &Type,
        subst: &Subst,
    ) -> Vec<Type> {
        let equations = self.result_equations_for(&resolved(ty, subst), subst);
        let mut alternatives = vec![resolved(ty, subst)];
        loop {
            let before = alternatives.len();
            for equation in &equations {
                let mut candidates = Vec::new();
                for alternative in &alternatives {
                    match equation {
                        ResultConstraint::Join { inputs, result } => {
                            for input in inputs {
                                corresponding_types(alternative, result, input, &mut candidates);
                            }
                        }
                        ResultConstraint::Annotation { actual, declared } => {
                            corresponding_types(alternative, actual, declared, &mut candidates);
                            corresponding_types(alternative, declared, actual, &mut candidates);
                        }
                    }
                }
                for candidate in candidates {
                    if !alternatives.contains(&candidate) {
                        alternatives.push(candidate);
                    }
                }
            }
            if before == alternatives.len() {
                return alternatives;
            }
        }
    }

    pub(in crate::infer) fn result_equations_for(
        &self,
        body: &Type,
        subst: &Subst,
    ) -> Vec<ResultConstraint> {
        let mut available: Vec<ResultConstraint> = self
            .result_type_constraints
            .iter()
            .map(|constraint| ResultConstraint::Annotation {
                actual: resolved(&constraint.actual, subst),
                declared: resolved(&constraint.declared, subst),
            })
            .collect();
        available.extend(
            self.deferred_shape_checks
                .iter()
                .filter(|check| matches!(check.rule, DeferredShapeRule::ResultJoin { .. }))
                .map(|check| ResultConstraint::Join {
                    inputs: check.arg_tys.iter().map(|ty| resolved(ty, subst)).collect(),
                    result: resolved(&check.result_ty, subst),
                }),
        );
        let mut visible = crate::env::free_tvars(body);
        let mut equations = Vec::new();
        loop {
            let before = equations.len();
            available.retain(|equation| {
                let variables: Vec<_> = equation
                    .types()
                    .into_iter()
                    .flat_map(crate::env::free_tvars)
                    .collect();
                if variables.iter().any(|variable| visible.contains(variable)) {
                    visible.extend(variables);
                    equations.push(equation.clone());
                    false
                } else {
                    true
                }
            });
            if equations.len() == before {
                break;
            }
        }
        equations
    }

    fn result_has_pending_producer_with(
        &self,
        actual: &Type,
        subst: &Subst,
        binding_only: bool,
    ) -> bool {
        let mut variables = crate::env::free_tvars(&resolved(actual, subst));
        variables.sort_unstable();
        variables.dedup();
        // A transported operation's result is still produced at application.
        // Its signature may expose ordinary input annotations, but annotating
        // that result (especially with runtime extents) must not replace the
        // fresh result on which the eventual operation publishes its facts.
        let contract_results: Vec<_> = subst
            .pending_collection_contracts()
            .iter()
            .flat_map(|(_, _, constraint)| crate::env::free_tvars(constraint.result()))
            .collect();
        loop {
            let before = variables.len();
            if variables
                .iter()
                .any(|variable| contract_results.contains(variable))
            {
                return true;
            }
            for check in &self.deferred_shape_checks {
                if !crate::env::free_tvars(&resolved(&check.result_ty, subst))
                    .iter()
                    .any(|variable| variables.contains(variable))
                {
                    continue;
                }
                if matches!(check.rule, DeferredShapeRule::ResultJoin { .. }) {
                    // A raw value hole remains the owner of the callable
                    // fields exposed by a join. Resolving that hole to a fresh
                    // skeleton is not a parameter annotation or application.
                    if !binding_only
                        && !self.result_inputs_closed
                        && check
                            .arg_tys
                            .iter()
                            .any(|input| matches!(input, Type::Var(_)))
                        && !is_closed_first_order(&resolved(&check.result_ty, subst))
                    {
                        return true;
                    }
                    let published = resolved(&check.result_ty, subst);
                    let mut corresponding = Vec::new();
                    for variable in variables.clone() {
                        for input in &check.arg_tys {
                            walk_corresponding_types(
                                &Type::Var(variable),
                                &published,
                                &resolved(input, subst),
                                &mut corresponding,
                                true,
                            );
                        }
                    }
                    variables.extend(corresponding.iter().flat_map(crate::env::free_tvars));
                    variables.sort_unstable();
                    variables.dedup();
                    continue;
                }
                // These rules derive a type from an operand: their output is not
                // an input that a result annotation may choose. Ordinary call
                // inference retains its result constraints, e.g. to_tensor([]).
                if matches!(check.rule, DeferredShapeRule::Derivation(_))
                // [04-INF-2] lets a recursive group's bodies determine its
                // result constructors. A projection of that result is not a
                // lambda input awaiting the [04-INF-1] binding site. Grad's
                // differentiated parameters still require that binding site.
                && !(binding_only
                    && matches!(check.rule, DeferredShapeRule::Derivation(TypeDerivation::TupleProjection { .. } | TypeDerivation::RecordField { .. }))
                    && self.is_group_result_variable(&check.arg_tys[0], subst))
                {
                    return true;
                }
            }
            if variables.len() == before {
                return false;
            }
        }
    }

    pub(in crate::infer) fn close_result_inputs(&mut self) {
        self.result_inputs_closed = true;
    }

    fn is_group_result_variable(&self, ty: &Type, subst: &Subst) -> bool {
        let Type::Var(variable) = resolved(ty, subst) else {
            return false;
        };
        self.group_provisional_types
            .iter()
            .map(|(_, ty)| ty)
            .chain(self.group_member_types.iter().map(|(_, ty)| ty))
            .chain(self.group_references.iter().map(|reference| &reference.ty))
            .chain(self.sibling_links.iter().map(|link| &link.ty))
            .chain(self.closing_group_types.iter())
            .any(|ty| match resolved(ty, subst) {
                Type::Fn(_, result) => crate::env::free_tvars(&result).contains(&variable),
                _ => false,
            })
    }

    pub(in crate::infer) fn defer_result_join(
        &mut self,
        left: &Type,
        right: &Type,
        vg: &mut VarGen,
        subst: &Subst,
    ) -> Option<Type> {
        if !self.result_has_pending_producer_with(left, subst, true)
            && !self.result_has_pending_producer_with(right, subst, true)
            && ![left, right]
                .iter()
                .any(|ty| self.result_has_open_inputs(ty, subst))
        {
            return None;
        }
        let published =
            common_result_structure(&resolved(left, subst), &resolved(right, subst), vg);
        let published = independent_type_variables(&published, vg, subst);
        self.defer_shape_check(
            DeferredShapeRule::ResultJoin {
                last_propagated: None,
            },
            Vec::new(),
            vec![left.clone(), right.clone()],
            published.clone(),
        );
        Some(published)
    }

    /// A structural join can determine an ordinary parameter slot before its
    /// result producer settles. A raw value hole carrying a callable remains
    /// indivisible: exposing its shape must not erase its origin for later use.
    fn settle_ready_join_parts(
        &self,
        inputs: &[Type],
        published: &Type,
        subst: &mut Subst,
    ) -> Result<(), TypeError> {
        if inputs.iter().any(|input| {
            matches!(input, Type::Var(_))
                && !matches!(resolved(input, subst), Type::Var(_))
                && !is_closed_first_order(&resolved(input, subst))
        }) {
            return Ok(());
        }
        let inputs: Vec<_> = inputs.iter().map(|ty| resolved(ty, subst)).collect();
        let published = resolved(published, subst);
        let fields = |ty: &Type| -> Option<Vec<Type>> {
            match (&published, ty) {
                (Type::Fn(args, _), Type::Fn(other, ret)) if args.len() == other.len() => Some(
                    other
                        .iter()
                        .cloned()
                        .chain(std::iter::once((**ret).clone()))
                        .collect(),
                ),
                (Type::Tuple(args), Type::Tuple(other)) if args.len() == other.len() => {
                    Some(other.clone())
                }
                (Type::Adt(name, args), Type::Adt(other_name, other))
                    if name == other_name && args.len() == other.len() =>
                {
                    Some(other.clone())
                }
                (Type::Ref(_), Type::Ref(other)) => Some(vec![(**other).clone()]),
                _ => None,
            }
        };
        if let Some(output_fields) = fields(&published)
            && let Some(input_fields) = inputs.iter().map(fields).collect::<Option<Vec<_>>>()
        {
            for (index, field) in output_fields.iter().enumerate() {
                let inputs = input_fields
                    .iter()
                    .map(|fields| fields[index].clone())
                    .collect::<Vec<_>>();
                self.settle_ready_join_parts(&inputs, field, subst)?;
            }
        } else if inputs.iter().any(is_closed_first_order)
            && inputs
                .iter()
                .all(|input| !self.result_has_pending_producer_with(input, subst, true))
        {
            for input in &inputs {
                unify(&published, input, subst)?;
            }
        }
        Ok(())
    }

    pub(super) fn replay_result_joins(
        &mut self,
        vg: &mut VarGen,
        subst: &mut Subst,
        errors: &mut DiagnosticSink<'_>,
    ) -> bool {
        // Keep the complete producer graph visible while deciding readiness.
        let joins: Vec<_> = self
            .deferred_shape_checks
            .iter()
            .filter_map(|check| match &check.rule {
                DeferredShapeRule::ResultJoin { last_propagated } => Some((
                    check.id,
                    check.arg_tys.clone(),
                    check.result_ty.clone(),
                    last_propagated.clone(),
                )),
                _ => None,
            })
            .collect();
        let mut progressed = false;
        for (id, inputs, published, last_propagated) in joins {
            // A rule publishes its equation before knowing a result
            // constructor. Expose only structure common to every producer;
            // retain independent holes until binding evidence reaches it.
            if matches!(resolved(&published, subst), Type::Var(_))
                && let Some((first, rest)) = inputs.split_first()
            {
                let common = rest.iter().fold(resolved(first, subst), |common, input| {
                    common_result_structure(&common, &resolved(input, subst), vg)
                });
                if !matches!(common, Type::Var(_)) {
                    let structure = independent_type_variables(&common, vg, subst);
                    if let Err(error) = unify(&published, &structure, subst) {
                        errors.push(error.into());
                    }
                }
            }
            let before_parts = (
                resolved(&published, subst),
                inputs
                    .iter()
                    .map(|input| resolved(input, subst))
                    .collect::<Vec<_>>(),
            );
            if let Err(error) = self.settle_ready_join_parts(&inputs, &published, subst) {
                errors.push(error.into());
            }
            progressed |= before_parts
                != (
                    resolved(&published, subst),
                    inputs
                        .iter()
                        .map(|input| resolved(input, subst))
                        .collect::<Vec<_>>(),
                );
            let first_order = inputs
                .iter()
                .any(|ty| is_closed_first_order(&resolved(ty, subst)));
            let pending = !first_order
                && inputs.iter().any(|ty| {
                    self.result_has_pending_producer_with(ty, subst, true)
                        || (!self.result_inputs_closed && self.result_has_open_inputs(ty, subst))
                });
            let output = resolved(&published, subst);
            if pending && last_propagated.as_ref() == Some(&output) {
                continue;
            }
            progressed = true;
            for input in &inputs {
                // Until every branch is ready, only constraints arriving at
                // the published result (an application) may flow into a branch.
                // Fresh holes prevent a concrete sibling from feeding back
                // through the published result into another pending branch.
                let expected = if pending {
                    independent_type_variables(&output, vg, subst)
                } else {
                    published.clone()
                };
                if let Err(error) = unify(&expected, input, subst) {
                    errors.push(error.into());
                }
            }
            if !pending {
                self.deferred_shape_checks.retain(|check| check.id != id);
            } else if let Some(check) = self
                .deferred_shape_checks
                .iter_mut()
                .find(|check| check.id == id)
            {
                check.rule = DeferredShapeRule::ResultJoin {
                    last_propagated: Some(output),
                };
            }
        }
        progressed
    }

    pub(super) fn replay_result_type_constraints(
        &mut self,
        subst: &mut Subst,
        errors: &mut DiagnosticSink<'_>,
    ) {
        for constraint in std::mem::take(&mut self.result_type_constraints) {
            if self.result_has_pending_producer(&constraint.actual, subst)
                || (!self.result_inputs_closed
                    && self.result_has_open_inputs(&constraint.actual, subst)
                    && !is_closed_first_order(&resolved(&constraint.declared, subst)))
            {
                self.result_type_constraints.push(constraint);
            } else if let Err(error) = unify(&constraint.actual, &constraint.declared, subst) {
                errors.push(error.into());
            }
        }
    }
}

fn corresponding_types(needle: &Type, from: &Type, to: &Type, found: &mut Vec<Type>) {
    walk_corresponding_types(needle, from, to, found, false);
}

/// Callable alternatives require exact structural correspondence. Dependency
/// traversal also follows a whole-value hole that owns a projected field.
fn walk_corresponding_types(
    needle: &Type,
    from: &Type,
    to: &Type,
    found: &mut Vec<Type>,
    variable_dependencies: bool,
) {
    if from == needle {
        found.push(to.clone());
        return;
    }
    if variable_dependencies
        && matches!(needle, Type::Var(_))
        && matches!(to, Type::Var(_))
        && crate::env::free_tvars(needle)
            .iter()
            .any(|var| crate::env::free_tvars(from).contains(var))
    {
        found.push(to.clone());
        return;
    }
    match (from, to) {
        (Type::Fn(args, result), Type::Fn(other_args, other_result))
            if args.len() == other_args.len() =>
        {
            for (from, to) in args.iter().zip(other_args) {
                walk_corresponding_types(needle, from, to, found, variable_dependencies);
            }
            walk_corresponding_types(needle, result, other_result, found, variable_dependencies);
        }
        (Type::Tuple(items), Type::Tuple(other)) if items.len() == other.len() => {
            for (from, to) in items.iter().zip(other) {
                walk_corresponding_types(needle, from, to, found, variable_dependencies);
            }
        }
        (Type::Adt(name, items), Type::Adt(other_name, other))
            if name == other_name && items.len() == other.len() =>
        {
            for (from, to) in items.iter().zip(other) {
                walk_corresponding_types(needle, from, to, found, variable_dependencies);
            }
        }
        (Type::KindedAdt(name, items), Type::KindedAdt(other_name, other))
            if name == other_name && items.len() == other.len() =>
        {
            for (from, to) in items.iter().zip(other) {
                if let (NominalArg::Type(from), NominalArg::Type(to)) = (from, to) {
                    walk_corresponding_types(needle, from, to, found, variable_dependencies);
                }
            }
        }
        (Type::Tensor(_, TensorPrec::Var(from)), Type::Tensor(_, to)) if variable_dependencies => {
            let target = match to {
                TensorPrec::Var(var) => Type::Var(*var),
                TensorPrec::Concrete(prim) => Type::Prim(*prim),
            };
            walk_corresponding_types(
                needle,
                &Type::Var(*from),
                &target,
                found,
                variable_dependencies,
            );
        }
        (Type::Ref(from), Type::Ref(to)) => {
            walk_corresponding_types(needle, from, to, found, variable_dependencies)
        }
        _ => {}
    }
}

/// A branch join exposes only constructors both branches already have.
/// A hole on either side stays a fresh hole, even opposite a concrete type.
fn is_closed_first_order(ty: &Type) -> bool {
    match ty {
        Type::Prim(_) | Type::Unit | Type::Tensor(..) => true,
        Type::Tuple(items) | Type::Adt(_, items) => items.iter().all(is_closed_first_order),
        Type::Ref(inner) => is_closed_first_order(inner),
        _ => false,
    }
}

fn common_result_structure(left: &Type, right: &Type, vg: &mut VarGen) -> Type {
    match (left, right) {
        (Type::Var(_), _) | (_, Type::Var(_)) => vg.fresh_type(),
        (Type::Fn(a, ar), Type::Fn(b, br)) if a.len() == b.len() => Type::Fn(
            a.iter()
                .zip(b)
                .map(|(a, b)| common_result_structure(a, b, vg))
                .collect(),
            Box::new(common_result_structure(ar, br, vg)),
        ),
        (Type::Tuple(a), Type::Tuple(b)) if a.len() == b.len() => Type::Tuple(
            a.iter()
                .zip(b)
                .map(|(a, b)| common_result_structure(a, b, vg))
                .collect(),
        ),
        (Type::Ref(a), Type::Ref(b)) => Type::Ref(Box::new(common_result_structure(a, b, vg))),
        (Type::Adt(a, aa), Type::Adt(b, ba)) if a == b && aa.len() == ba.len() => Type::Adt(
            a.clone(),
            aa.iter()
                .zip(ba)
                .map(|(a, b)| common_result_structure(a, b, vg))
                .collect(),
        ),
        (Type::KindedAdt(a, aa), Type::KindedAdt(b, ba))
            if a == b
                && aa.len() == ba.len()
                && aa.iter().zip(ba).all(|(a, b)| {
                    matches!((a, b), (NominalArg::Type(_), NominalArg::Type(_))) || a == b
                }) =>
        {
            Type::KindedAdt(
                a.clone(),
                aa.iter()
                    .zip(ba)
                    .map(|(a, b)| match (a, b) {
                        (NominalArg::Type(a), NominalArg::Type(b)) => {
                            NominalArg::Type(common_result_structure(a, b, vg))
                        }
                        _ => a.clone(),
                    })
                    .collect(),
            )
        }
        _ if left == right => left.clone(),
        _ => vg.fresh_type(),
    }
}

fn independent_type_variables(ty: &Type, vg: &mut VarGen, subst: &Subst) -> Type {
    let mut scheme = Scheme::mono(ty.clone());
    scheme.tvars = crate::env::free_tvars(ty).into_iter().collect();
    scheme.tvars.sort();
    scheme.tvar_restrictions = scheme
        .tvars
        .iter()
        .filter_map(|var| {
            subst
                .tvar_restriction(*var)
                .map(|restriction| (*var, restriction))
        })
        .collect();
    Env::new().instantiate(&scheme, vg, subst)
}
