//! Pure scheme operations with explicit inference data.

use crate::model::*;
use crate::unify::{CollectionContractId, Subst};
use alloc::{sync::Arc, vec, vec::Vec};
use chelis_unord::UnordSet;

#[derive(Clone)]
pub struct InstantiatedScheme<E> {
    pub ty: Type<E>,
    pub tvars: Vec<(TypeVar, Type<E>)>,
    pub dvars: Vec<(DimVar, DimVar)>,
    pub rvars: Vec<(RankVar, RankVar)>,
}

pub struct CollectionQuantification<E> {
    /// Obligations this scheme now owns, in ledger order.
    pub constraints: Vec<CollectionConstraint<E>>,
    /// Exact ledger entries either moved into the scheme or discarded with a
    /// function-bearing subvalue no longer visible in the generalized type.
    pub ledger_removals: Vec<crate::unify::CollectionContractId>,
    /// Body variables that must stay monomorphic because an obligation this
    /// scheme does NOT own still carries them.
    pub pinned: UnordSet<TypeVar>,
}
impl<E> Default for CollectionQuantification<E> {
    fn default() -> Self {
        Self {
            constraints: Vec::new(),
            ledger_removals: Vec::new(),
            pinned: UnordSet::new(),
        }
    }
}

pub fn instantiate<E: Clone + PartialEq>(
    scheme: &Scheme<E>,
    var_gen: &mut VarGen,
    inference_subst: &Subst<E>,
) -> Type<E> {
    instantiate_scheme(scheme, var_gen, inference_subst).ty
}

pub fn instantiate_with_tvar_mapping<E: Clone + PartialEq>(
    scheme: &Scheme<E>,
    var_gen: &mut VarGen,
    inference_subst: &Subst<E>,
) -> (Type<E>, Vec<(TypeVar, Type<E>)>) {
    let instantiated = instantiate_scheme(scheme, var_gen, inference_subst);
    (instantiated.ty, instantiated.tvars)
}

pub fn instantiate_scheme<E: Clone + PartialEq>(
    scheme: &Scheme<E>,
    var_gen: &mut VarGen,
    inference_subst: &Subst<E>,
) -> InstantiatedScheme<E> {
    reserve_scheme_ids(scheme, var_gen);
    let origin = scheme.result_origin.as_ref();
    let mut tvars = origin.map_or(&scheme.tvars, |origin| &origin.tvars).clone();
    let mut dvars = origin.map_or(&scheme.dvars, |origin| &origin.dvars).clone();
    let mut rvars = origin.map_or(&scheme.rvars, |origin| &origin.rvars).clone();
    // The raw graph owns shared identity. A recursive group's solved
    // signature may mention additional variables, but must not freshen a
    // variable the origin deliberately kept shared with its producer.
    let carried_origin = origin.map(|origin| {
        Type::Tuple(
            core::iter::once(origin.body.clone())
                .chain(
                    origin
                        .equations
                        .iter()
                        .flat_map(|equation| equation.types())
                        .cloned(),
                )
                .collect(),
        )
    });
    let shared_tvars = carried_origin.as_ref().map(free_tvars).unwrap_or_default();
    let shared_dvars = carried_origin.as_ref().map(free_dvars).unwrap_or_default();
    let shared_rvars = carried_origin.as_ref().map(free_rvars).unwrap_or_default();
    for var in &scheme.tvars {
        if !tvars.contains(var) && !shared_tvars.contains(var) {
            tvars.push(*var);
        }
    }
    for var in &scheme.dvars {
        if !dvars.contains(var) && !shared_dvars.contains(var) {
            dvars.push(*var);
        }
    }
    for var in &scheme.rvars {
        if !rvars.contains(var) && !shared_rvars.contains(var) {
            rvars.push(*var);
        }
    }
    let mut subst = Subst::new();
    let mut tvar_mapping = Vec::with_capacity(tvars.len());
    for &tv in &tvars {
        let fresh = var_gen.fresh_type();
        subst
            .insert_type(tv, fresh.clone())
            .expect("a fresh quantified type-variable renaming is valid");
        if let Type::Var(fresh_var) = fresh {
            // Raw origins and published signatures can use different
            // representatives. Rename both restriction ledgers with the
            // same quantifiers; neither view may erase the other's domain.
            for (_, restriction) in scheme
                .tvar_restrictions
                .iter()
                .chain(
                    origin
                        .into_iter()
                        .flat_map(|origin| &origin.tvar_restrictions),
                )
                .filter(|(restricted, _)| *restricted == tv)
            {
                inference_subst
                    .narrow_tvar_restriction(fresh_var, *restriction)
                    .expect("a checked scheme's restriction ledgers are compatible");
            }
        }
        // [04-LIN-10]: a generic's type parameter stays key-free at every
        // instantiation. The mark lives on the quantified variable, which
        // generalization marked, so a builtin or constructor scheme, never
        // generalized, carries none.
        if let Type::Var(fresh_var) = fresh
            && let Some(origin) = inference_subst.key_free_origin(tv)
        {
            inference_subst.forbid_key_instantiation(fresh_var, origin);
        }
        tvar_mapping.push((tv, fresh));
    }
    // chelis#1654: renaming the quantifiers is exactly what turns a
    // scheme-level obligation into one this USE owes, so it is done here,
    // in the one instantiation mechanism, for the reason stated above:
    // a second copy of this loop that dropped the obligations would
    // compile, and the only symptom would be a program that should have
    // been rejected type-checking.
    //
    // The renaming is applied AFTER the dimension and rank quantifiers are
    // inserted below, so a constraint whose carried type mentions one gets
    // that variable renamed too.
    let mut dvar_mapping = Vec::with_capacity(scheme.dvars.len());
    for &dv in &dvars {
        // Mint the variable directly rather than destructuring
        // `fresh_dim()`: that is `Dim::Var(fresh_dvar())` today, but a
        // pattern match would silently drop the mapping entry (and the
        // name with it) if it ever returned another shape.
        let fresh_dv = var_gen.fresh_dvar();
        dvar_mapping.push((dv, fresh_dv));
        subst.insert_dim(dv, Dim::Var(fresh_dv));
        if let Some(axis) = inference_subst.mapped_axis(dv) {
            // The renaming substitution must recognize the old identity
            // while rewriting the scheme body, and the active inference
            // substitution must recognize the fresh identity afterward.
            subst.mark_mapped_axis(dv, axis);
            inference_subst.mark_mapped_axis(fresh_dv, axis);
        } else {
            inference_subst.copy_dimension_label(dv, fresh_dv);
        }
    }
    let mut rvar_mapping = Vec::with_capacity(scheme.rvars.len());
    for &rv in &rvars {
        // Each rank var instantiates to a fresh sole-`Rank` shape so every
        // call site gets its own rank (Tier-2 rank polymorphism).
        let fresh_rv = var_gen.fresh_rvar();
        rvar_mapping.push((rv, fresh_rv));
        subst.insert_rank(rv, vec![Dim::Rank(fresh_rv)]);
    }
    for constraint in &scheme.constraints {
        let renamed = constraint.map_types(|ty| subst.apply(ty));
        inference_subst.record_collection_contract(renamed);
    }
    if let Some(origin) = origin {
        for equation in &origin.equations {
            inference_subst.record_result_constraint(equation.map_types(|ty| subst.apply(ty)));
        }
        // Origin controls when an equation may supply input evidence; it
        // never replaces the checked signature's required equalities.
        // Instantiate both views together and retain their compatibility
        // as a result constraint, so it cannot admit an unresolved Grad.
        if origin.body != scheme.body {
            inference_subst.record_result_constraint(crate::model::ResultConstraint::Annotation {
                actual: subst.apply(&origin.body),
                declared: subst.apply(&scheme.body),
            });
        }
    }
    InstantiatedScheme {
        ty: subst.apply(origin.map_or(&scheme.body, |origin| &origin.body)),
        tvars: tvar_mapping,
        dvars: dvar_mapping,
        rvars: rvar_mapping,
    }
}

pub fn free_tvars<E>(ty: &Type<E>) -> Vec<TypeVar> {
    let mut vars = Vec::new();
    collect_tvars(ty, &mut vars);
    vars.sort_by_key(|v| v.0);
    vars.dedup_by_key(|v| v.0);
    vars
}

fn collect_tvars<E>(ty: &Type<E>, vars: &mut Vec<TypeVar>) {
    match ty {
        Type::Var(v) => vars.push(*v),
        Type::Fn(args, ret) => {
            for a in args.iter() {
                collect_tvars(a, vars);
            }
            collect_tvars(ret, vars);
        }
        Type::Ref(inner) => collect_tvars(inner, vars),
        Type::Adt(_, args) => {
            for a in args.iter() {
                collect_tvars(a, vars);
            }
        }
        Type::KindedAdt(_, args) => {
            for argument in args.iter() {
                if let NominalArg::Type(ty) = argument {
                    collect_tvars(ty, vars);
                }
            }
        }
        Type::Tuple(ts) => {
            for t in ts.iter() {
                collect_tvars(t, vars);
            }
        }
        // WS-A5 (spec/04-type-system.md §5.8): the tensor precision
        // slot may carry a TypeVar (precision polymorphism). The slot
        // must participate in free-var collection so generalization
        // can pin precision-quantified vars in the resulting Scheme;
        // missing this means each call site reuses the SAME var
        // across instantiations and the second call's precision
        // collides with the first call's binding.
        Type::Tensor(_, prec) => {
            if let TensorPrec::Var(v) = prec {
                vars.push(*v);
            }
        }
        Type::Prim(_) | Type::Unit | Type::Error(_) => {}
    }
}

pub fn free_dvars<E>(ty: &Type<E>) -> Vec<DimVar> {
    let mut vars = Vec::new();
    collect_dvars(ty, &mut vars);
    vars.sort_by_key(|v| v.0);
    vars.dedup_by_key(|v| v.0);
    vars
}

fn collect_dvars<E>(ty: &Type<E>, vars: &mut Vec<DimVar>) {
    match ty {
        Type::Tensor(dims, _) => {
            for d in dims {
                if let Dim::Var(v) = d {
                    vars.push(*v);
                }
            }
        }
        Type::Fn(args, ret) => {
            for a in args.iter() {
                collect_dvars(a, vars);
            }
            collect_dvars(ret, vars);
        }
        Type::Ref(inner) => collect_dvars(inner, vars),
        Type::Adt(_, args) => {
            for a in args.iter() {
                collect_dvars(a, vars);
            }
        }
        Type::KindedAdt(_, args) => {
            for argument in args.iter() {
                match argument {
                    NominalArg::Type(ty) => collect_dvars(ty, vars),
                    NominalArg::Dimension(Dim::Var(var)) => vars.push(*var),
                    NominalArg::Dimension(_) => {}
                }
            }
        }
        Type::Tuple(ts) => {
            for t in ts.iter() {
                collect_dvars(t, vars);
            }
        }
        _ => {}
    }
}

pub fn collect_dims<E>(ty: &Type<E>, dims: &mut Vec<Dim>) {
    match ty {
        Type::Tensor(ds, _) => {
            dims.extend(ds.iter().cloned());
        }
        Type::Fn(args, ret) => {
            for a in args.iter() {
                collect_dims(a, dims);
            }
            collect_dims(ret, dims);
        }
        Type::Ref(inner) => collect_dims(inner, dims),
        Type::Adt(_, args) => {
            for a in args.iter() {
                collect_dims(a, dims);
            }
        }
        Type::KindedAdt(_, args) => {
            for argument in args.iter() {
                match argument {
                    NominalArg::Type(ty) => collect_dims(ty, dims),
                    NominalArg::Dimension(dim) => dims.push(dim.clone()),
                }
            }
        }
        Type::Tuple(ts) => {
            for t in ts.iter() {
                collect_dims(t, dims);
            }
        }
        _ => {}
    }
}

pub fn free_rvars<E>(ty: &Type<E>) -> Vec<RankVar> {
    let mut vars = Vec::new();
    collect_rvars(ty, &mut vars);
    vars.sort_by_key(|v| v.0);
    vars.dedup_by_key(|v| v.0);
    vars
}

fn collect_rvars<E>(ty: &Type<E>, vars: &mut Vec<RankVar>) {
    match ty {
        Type::Tensor(dims, _) => {
            for d in dims {
                if let Dim::Rank(r) = d {
                    vars.push(*r);
                }
            }
        }
        Type::Fn(args, ret) => {
            for a in args.iter() {
                collect_rvars(a, vars);
            }
            collect_rvars(ret, vars);
        }
        Type::Ref(inner) => collect_rvars(inner, vars),
        Type::Adt(_, args) => {
            for a in args.iter() {
                collect_rvars(a, vars);
            }
        }
        Type::KindedAdt(_, args) => {
            for argument in args.iter() {
                match argument {
                    NominalArg::Type(ty) => collect_rvars(ty, vars),
                    NominalArg::Dimension(Dim::Rank(var)) => vars.push(*var),
                    NominalArg::Dimension(_) => {}
                }
            }
        }
        Type::Tuple(ts) => {
            for t in ts.iter() {
                collect_rvars(t, vars);
            }
        }
        _ => {}
    }
}

pub fn closed_key_relation_variables<E: Clone + PartialEq>(
    scheme: &Scheme<E>,
) -> UnordSet<TypeVar> {
    fn outside_callable<E: PartialEq>(
        ty: &Type<E>,
        callable: &Type<E>,
        outside: &mut UnordSet<TypeVar>,
    ) -> bool {
        if ty == callable {
            return true;
        }
        match ty {
            Type::Fn(args, ret) => {
                let mut found = outside_callable(ret, callable, outside);
                for arg in args.iter() {
                    found |= outside_callable(arg, callable, outside);
                }
                found
            }
            Type::Ref(inner) => outside_callable(inner, callable, outside),
            Type::Adt(_, args) | Type::Tuple(args) => {
                let mut found = false;
                for arg in args.iter() {
                    found |= outside_callable(arg, callable, outside);
                }
                found
            }
            Type::KindedAdt(_, args) => {
                let mut found = false;
                for arg in args.iter() {
                    if let NominalArg::Type(arg) = arg {
                        found |= outside_callable(arg, callable, outside);
                    }
                }
                found
            }
            other => {
                outside.extend(free_tvars(other));
                false
            }
        }
    }

    let mut closed = UnordSet::new();
    for relation in &scheme.constraints {
        if !matches!(
            relation,
            CollectionConstraint::FloatClassify { .. }
                | CollectionConstraint::KeyFromSeed { .. }
                | CollectionConstraint::SplitKey { .. }
                | CollectionConstraint::SplitKeys { .. }
                | CollectionConstraint::FoldIn { .. }
        ) {
            continue;
        }
        let callable = crate::unify::collection_contract_callable_type(relation);
        let mut outside = UnordSet::new();
        if !outside_callable(&scheme.body, &callable, &mut outside) {
            continue;
        }
        // The scheme can own several checked collection relations whose
        // variables unify. A variable carried by another relation is not
        // selected solely by this key operation, even when that other
        // callable is absent from the scheme body's visible shape.
        for other in &scheme.constraints {
            if core::ptr::eq(other, relation) {
                continue;
            }
            for carried in other.carried_types() {
                outside.extend(free_tvars(carried));
            }
        }
        for var in free_tvars(&callable) {
            if scheme.tvars.contains(&var) && !outside.contains(&var) {
                closed.insert(var);
            }
        }
    }
    closed
}

pub fn collection_constraints_to_quantify<E: Clone + PartialEq>(
    body: &Type<E>,
    body_variables: &[TypeVar],
    body_candidates: &[TypeVar],
    subst: &Subst<E>,
    owned_contracts: Option<&[crate::unify::CollectionContractId]>,
) -> CollectionQuantification<E> {
    let mut split = CollectionQuantification::default();
    let pending = subst.pending_collection_contracts();
    if pending.is_empty() {
        return split;
    }
    let visible = body_variables.iter().copied().collect::<UnordSet<_>>();
    let candidates = body_candidates.iter().copied().collect::<UnordSet<_>>();
    let current_level = subst.current_level();
    for (id, level, constraint) in pending {
        let owned = match owned_contracts {
            Some(owned) => owned.contains(&id),
            None => level > current_level,
        };
        if !owned {
            continue;
        }
        let mut footprint = UnordSet::default();
        for carried in constraint.carried_types() {
            footprint.extend(free_tvars(carried));
        }
        let footprint = footprint.into_sorted();
        let owned = if footprint.is_empty() {
            crate::unify::collection_contract_visible_in_type(&constraint, body)
        } else {
            footprint.iter().all(|var| candidates.contains(var))
        };
        if owned {
            split.ledger_removals.push(id);
            if !split.constraints.contains(&constraint) {
                split.constraints.push(constraint);
            }
        } else if footprint.is_empty() || footprint.iter().all(|var| !visible.contains(var)) {
            // The expression discarded the function-bearing subvalue:
            // `(len, 1).1` and `ignore(len)` must not leave the detached
            // contract behind to reject an unrelated result.
            split.ledger_removals.push(id);
        } else {
            split
                .pinned
                .extend(footprint.into_iter().filter(|var| candidates.contains(var)));
        }
    }
    split
}

pub fn generalize_by_levels<E: Clone + PartialEq>(
    ty: &Type<E>,
    subst: &Subst<E>,
    excluded: &UnordSet<TypeVar>,
    owned_contracts: Option<&[crate::unify::CollectionContractId]>,
) -> (Scheme<E>, Vec<crate::unify::CollectionContractId>) {
    let ty = subst.apply(ty);
    let ty_tvars = free_tvars(&ty);
    let ty_dvars = free_dvars(&ty);
    let ty_rvars = free_rvars(&ty);
    let level = subst.current_level();
    // chelis#1489: a variable still tied to a pending operand gate, as its
    // operand or its result, stays monomorphic until the gate discharges;
    // see `Subst::pending_gate_vars`. Levels cannot see that tie -- it
    // lives in the gate ledger, not in any unification.
    let (pending_t, pending_d, pending_r) = subst.pending_gate_vars();
    // spec/04 §3.1.1: a variable minted for an in-group recursive
    // instantiation stays monomorphic while its group is inferred, so a
    // let-bound alias of a group member cannot smuggle in polymorphic
    // recursion.
    let generalizable = |v: TypeVar| {
        subst.level_of_tvar(v) > level && !excluded.contains(&v) && !pending_t.contains(&v)
    };
    let tvars = ty_tvars
        .iter()
        .copied()
        .filter(|v| generalizable(*v))
        .collect::<Vec<_>>();
    let dvars = ty_dvars
        .into_iter()
        .filter(|v| {
            subst.level_of_dvar(*v) > level
                && !pending_d.contains(v)
                && matches!(subst.constraint_dim(&Dim::Var(*v)), Dim::Var(_))
        })
        .collect();
    let rvars = ty_rvars
        .into_iter()
        .filter(|v| subst.level_of_rvar(*v) > level && !pending_r.contains(v))
        .collect();
    generalize_candidates(ty, subst, &ty_tvars, tvars, dvars, rvars, owned_contracts)
}

/// Generalize a checked value with explicit recursion and declaration data.
pub fn generalize<E: Clone + PartialEq>(
    ty: &Type<E>,
    subst: &Subst<E>,
    excluded: &UnordSet<TypeVar>,
    owned_contracts: Option<&[CollectionContractId]>,
    equations: &[ResultConstraint<E>],
) -> Scheme<E> {
    let (mut scheme, ledger_removals) = generalize_by_levels(ty, subst, excluded, owned_contracts);
    crate::result_scope::ResultScope::new(equations).retain_closed_quantifiers(&mut scheme);
    if !ledger_removals.is_empty() {
        subst.take_collection_contracts(&ledger_removals);
    }
    if equations.is_empty() {
        let closed = closed_key_relation_variables(&scheme);
        for var in &scheme.tvars {
            if !closed.contains(var) {
                subst.forbid_key_instantiation(*var, crate::unify::GenericParameter::default());
            }
        }
    }
    scheme
}

/// Build a scheme from variable candidates selected by explicit scope policy.
pub fn generalize_candidates<E: Clone + PartialEq>(
    ty: Type<E>,
    subst: &Subst<E>,
    body_variables: &[TypeVar],
    mut tvars: Vec<TypeVar>,
    dvars: Vec<DimVar>,
    rvars: Vec<RankVar>,
    owned_contracts: Option<&[CollectionContractId]>,
) -> (Scheme<E>, Vec<CollectionContractId>) {
    let split =
        collection_constraints_to_quantify(&ty, body_variables, &tvars, subst, owned_contracts);
    tvars.retain(|var| !split.pinned.contains(var));
    let tvar_restrictions = tvars
        .iter()
        .filter_map(|var| {
            subst
                .tvar_restriction(*var)
                .map(|restriction| (*var, restriction))
        })
        .collect();
    (
        Scheme {
            result_origin: None,
            tvars,
            tvar_restrictions,
            dvars,
            rvars,
            constraints: split.constraints,
            body: ty,
        },
        split.ledger_removals,
    )
}

/// Collect free variables from explicit environment schemes.
pub fn free_scheme_tvars<'a, E: Clone + PartialEq + 'a>(
    schemes: impl IntoIterator<Item = &'a Scheme<E>>,
    subst: &Subst<E>,
) -> UnordSet<TypeVar> {
    let mut result = UnordSet::new();
    for scheme in schemes {
        let ty = subst.apply_scheme(scheme);
        for var in free_tvars(&ty) {
            if !scheme.tvars.contains(&var) {
                result.insert(var);
            }
        }
    }
    result
}

/// Collect free variables from explicit environment schemes.
pub fn free_scheme_dvars<'a, E: Clone + PartialEq + 'a>(
    schemes: impl IntoIterator<Item = &'a Scheme<E>>,
    subst: &Subst<E>,
) -> UnordSet<DimVar> {
    let mut result = UnordSet::new();
    for scheme in schemes {
        let ty = subst.apply_scheme(scheme);
        for var in free_dvars(&ty) {
            if !scheme.dvars.contains(&var) {
                result.insert(var);
            }
        }
    }
    result
}

/// Collect free variables from explicit environment schemes.
pub fn free_scheme_rvars<'a, E: Clone + PartialEq + 'a>(
    schemes: impl IntoIterator<Item = &'a Scheme<E>>,
    subst: &Subst<E>,
) -> UnordSet<RankVar> {
    let mut result = UnordSet::new();
    for scheme in schemes {
        let ty = subst.apply_scheme(scheme);
        for var in free_rvars(&ty) {
            if !scheme.rvars.contains(&var) {
                result.insert(var);
            }
        }
    }
    result
}

/// Generalize with an explicit environment and explicit recursion exclusions.
pub fn generalize_over_environment<'a, E: Clone + PartialEq + 'a>(
    ty: &Type<E>,
    subst: &Subst<E>,
    schemes: impl IntoIterator<Item = &'a Scheme<E>>,
    excluded: &UnordSet<TypeVar>,
) -> Scheme<E> {
    let mut env_tvars = UnordSet::new();
    let mut env_dvars = UnordSet::new();
    let mut env_rvars = UnordSet::new();
    for scheme in schemes {
        let body = subst.apply_scheme(scheme);
        env_tvars.extend(
            free_tvars(&body)
                .into_iter()
                .filter(|var| !scheme.tvars.contains(var)),
        );
        env_dvars.extend(
            free_dvars(&body)
                .into_iter()
                .filter(|var| !scheme.dvars.contains(var)),
        );
        env_rvars.extend(
            free_rvars(&body)
                .into_iter()
                .filter(|var| !scheme.rvars.contains(var)),
        );
    }
    let ty = subst.apply(ty);
    let (pending_t, pending_d, pending_r) = subst.pending_gate_vars();
    let ty_tvars = free_tvars(&ty);
    let tvars = ty_tvars
        .iter()
        .copied()
        .filter(|var| {
            !env_tvars.contains(var) && !excluded.contains(var) && !pending_t.contains(var)
        })
        .collect::<Vec<_>>();
    let dvars = free_dvars(&ty)
        .into_iter()
        .filter(|var| {
            !env_dvars.contains(var)
                && !pending_d.contains(var)
                && matches!(subst.constraint_dim(&Dim::Var(*var)), Dim::Var(_))
        })
        .collect();
    let rvars = free_rvars(&ty)
        .into_iter()
        .filter(|var| !env_rvars.contains(var) && !pending_r.contains(var))
        .collect();
    let (scheme, ledger_removals) =
        generalize_candidates(ty, subst, &ty_tvars, tvars, dvars, rvars, None);
    if !ledger_removals.is_empty() {
        subst.take_collection_contracts(&ledger_removals);
    }
    let closed = closed_key_relation_variables(&scheme);
    for var in &scheme.tvars {
        if !closed.contains(var) {
            subst.forbid_key_instantiation(*var, crate::unify::GenericParameter::default());
        }
    }
    scheme
}

fn reserve_scheme_ids<E>(scheme: &Scheme<E>, var_gen: &mut VarGen) {
    // The scheme retains every allocation until this operation returns.
    // Use the same allocation addresses as the shared-child substitution caches.
    let mut visited = UnordSet::new();
    reserve_body_ids(&scheme.body, var_gen, &mut visited);
    for var in &scheme.tvars {
        var_gen.reserve_tvar(*var);
    }
    for (var, _) in &scheme.tvar_restrictions {
        var_gen.reserve_tvar(*var);
    }
    for var in &scheme.dvars {
        var_gen.reserve_dvar(*var);
    }
    for var in &scheme.rvars {
        var_gen.reserve_rvar(*var);
    }
    for constraint in &scheme.constraints {
        for ty in constraint.carried_types() {
            reserve_body_ids(ty, var_gen, &mut visited);
        }
    }
    if let Some(origin) = &scheme.result_origin {
        reserve_body_ids(&origin.body, var_gen, &mut visited);
        for var in &origin.tvars {
            var_gen.reserve_tvar(*var);
        }
        for (var, _) in &origin.tvar_restrictions {
            var_gen.reserve_tvar(*var);
        }
        for var in &origin.dvars {
            var_gen.reserve_dvar(*var);
        }
        for var in &origin.rvars {
            var_gen.reserve_rvar(*var);
        }
        for equation in &origin.equations {
            for ty in equation.types() {
                reserve_body_ids(ty, var_gen, &mut visited);
            }
        }
    }
}

fn reserve_body_ids<E>(ty: &Type<E>, var_gen: &mut VarGen, visited: &mut UnordSet<usize>) {
    match ty {
        Type::Var(var) => var_gen.reserve_tvar(*var),
        Type::Fn(args, ret) => {
            reserve_list_ids(args, var_gen, visited);
            reserve_child_ids(ret, var_gen, visited);
        }
        Type::Ref(inner) => reserve_child_ids(inner, var_gen, visited),
        Type::Adt(_, args) | Type::Tuple(args) => reserve_list_ids(args, var_gen, visited),
        Type::KindedAdt(_, args) => {
            if visited.insert(Arc::as_ptr(args).cast::<NominalArg<E>>().addr()) {
                for arg in args.iter() {
                    match arg {
                        NominalArg::Type(ty) => reserve_body_ids(ty, var_gen, visited),
                        NominalArg::Dimension(dim) => reserve_dimension_id(dim, var_gen),
                    }
                }
            }
        }
        Type::Tensor(dims, prec) => {
            for dim in dims {
                reserve_dimension_id(dim, var_gen);
            }
            if let TensorPrec::Var(var) = prec {
                var_gen.reserve_tvar(*var);
            }
        }
        Type::Prim(_) | Type::Unit | Type::Error(_) => {}
    }
}

fn reserve_child_ids<E>(child: &Arc<Type<E>>, var_gen: &mut VarGen, visited: &mut UnordSet<usize>) {
    if visited.insert(Arc::as_ptr(child).addr()) {
        reserve_body_ids(child, var_gen, visited);
    }
}

fn reserve_list_ids<E>(
    items: &Arc<[Type<E>]>,
    var_gen: &mut VarGen,
    visited: &mut UnordSet<usize>,
) {
    if visited.insert(Arc::as_ptr(items).cast::<Type<E>>().addr()) {
        for item in items.iter() {
            reserve_body_ids(item, var_gen, visited);
        }
    }
}

fn reserve_dimension_id(dim: &Dim, var_gen: &mut VarGen) {
    match dim {
        Dim::Var(var) => var_gen.reserve_dvar(*var),
        Dim::Rank(var) => var_gen.reserve_rvar(*var),
        Dim::Name(_) | Dim::Lit(_) | Dim::Wildcard => {}
    }
}
