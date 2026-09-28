//! Result equations and their origin across generalization and application.

use super::*;

impl InferenceProduct {
    /// An ascription may check a suspended derivation's output only after its
    /// rule has settled. Its dependency is the published type, not a
    /// list of parameter holes read before the producer's constructor is known.
    /// Projection and alias transport retain that dependency through ordinary
    /// type unification. Each replay reads it anew from the remaining producers.
    pub(in crate::infer) fn defer_result_type_constraint(
        &mut self,
        actual: &Type,
        declared: &Type,
        subst: &Subst,
    ) -> bool {
        if resolved(actual, subst) == resolved(declared, subst)
            || (crate::env::free_tvars(&resolved(actual, subst)).is_empty()
                && !self.result_has_pending_producer(actual, subst))
        {
            return false;
        }
        self.result_type_constraints.push(ResultTypeConstraint {
            actual: actual.clone(),
            declared: declared.clone(),
        });
        true
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
        for variable in &scheme.tvars {
            subst.forbid_key_instantiation(*variable, crate::unify::GenericParameter::default());
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
        let mut visited = Vec::new();
        loop {
            let before = visited.len();
            for check in &self.deferred_shape_checks {
                if visited.contains(&check.id)
                    || !crate::env::free_tvars(&resolved(&check.result_ty, subst))
                        .iter()
                        .any(|variable| variables.contains(variable))
                {
                    continue;
                }
                visited.push(check.id);
                if matches!(check.rule, DeferredShapeRule::ResultJoin { .. }) {
                    variables.extend(
                        check
                            .arg_tys
                            .iter()
                            .flat_map(|ty| crate::env::free_tvars(&resolved(ty, subst))),
                    );
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
            if visited.len() == before {
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
                    && self.result_has_open_inputs(&constraint.actual, subst))
            {
                self.result_type_constraints.push(constraint);
            } else if let Err(error) = unify(&constraint.actual, &constraint.declared, subst) {
                errors.push(error.into());
            }
        }
    }
}

fn corresponding_types(needle: &Type, from: &Type, to: &Type, found: &mut Vec<Type>) {
    if from == needle {
        found.push(to.clone());
        return;
    }
    match (from, to) {
        (Type::Fn(args, result), Type::Fn(other_args, other_result))
            if args.len() == other_args.len() =>
        {
            for (from, to) in args.iter().zip(other_args) {
                corresponding_types(needle, from, to, found);
            }
            corresponding_types(needle, result, other_result, found);
        }
        (Type::Tuple(items), Type::Tuple(other)) if items.len() == other.len() => {
            for (from, to) in items.iter().zip(other) {
                corresponding_types(needle, from, to, found);
            }
        }
        (Type::Adt(name, items), Type::Adt(other_name, other))
            if name == other_name && items.len() == other.len() =>
        {
            for (from, to) in items.iter().zip(other) {
                corresponding_types(needle, from, to, found);
            }
        }
        (Type::KindedAdt(name, items), Type::KindedAdt(other_name, other))
            if name == other_name && items.len() == other.len() =>
        {
            for (from, to) in items.iter().zip(other) {
                if let (NominalArg::Type(from), NominalArg::Type(to)) = (from, to) {
                    corresponding_types(needle, from, to, found);
                }
            }
        }
        (Type::Ref(from), Type::Ref(to)) => corresponding_types(needle, from, to, found),
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
