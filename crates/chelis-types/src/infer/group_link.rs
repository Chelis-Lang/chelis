//! chelis#2590: in-group references to a recursive-group member whose
//! declared header omits a type, decided when the group completes.
//!
//! [04-INF-5] types an in-group reference "at the member's provisional
//! monomorphic type, as [04-INF-2] provides for a recursive call", and
//! [04-INF-2] separates the two kinds of signature variable. An authored
//! binder admits no substitute, so every in-group reference shares the
//! member's own binders (`Env::bind_holed_group_member`). An omitted type is
//! introduced by inference, and a reference may instantiate it at the
//! caller's own type or at a fully concrete type; [04-INF-3] rejects anything
//! else. The reference therefore takes a fresh instance of each hole, and
//! only the completed group can say which of the two it is: the type a hole
//! denotes is whatever the member's body determines, and a sibling inferred
//! later can still be determining it.
//!
//! [`link_group_references`] is that decision. Each hole's type is written in
//! the member's type parameters, the variables of its body-determined type
//! that are neither authored binders nor free outside the group. Each
//! reference is unified with a fresh instance of that type, one variable per
//! parameter, so the reference is typed at the body-determined type and never
//! narrows it ([04-INF-5]: "a body is never narrowed to satisfy a reference").
//! The instance of each parameter is then decided under [04-INF-2]. A fully
//! concrete instance is admitted and stays the reference's own. Any other
//! instance must be the parameter itself. An instance that is still a
//! variable is chosen as the parameter, which is how mutually recursive
//! members share a type, unless that would identify two of one member's
//! parameters: that would narrow a published signature to fit a reference.
//! Everything else is polymorphic recursion ([04-INF-3]).
//!
//! The body-determined types are known only once the group's references are
//! linked, and linking a reference can determine more of a member's type
//! (the member's result may be another member's result). The links are
//! therefore repeated until no member's type changes shape, with the
//! suspended checks that a link makes ready replayed between rounds. A group
//! whose types keep growing is polymorphic recursion, and a bound on the
//! rounds reports it.

use super::*;

/// One in-group reference to a member whose header omits a type.
pub(super) struct GroupReference {
    /// The declaration whose body holds the reference.
    pub(super) caller: Option<String>,
    pub(super) callee: String,
    /// The reference's instance of the callee's provisional scheme: the
    /// group's authored binders, and a fresh variable for each hole.
    pub(super) ty: Type,
    pub(super) span_id: Option<String>,
    pub(super) span_offset: Option<usize>,
}

/// A signature variable of any kind.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Variable {
    Type(TypeVar),
    Dim(DimVar),
    Rank(RankVar),
}

/// What a variable denotes under the substitution.
#[derive(PartialEq)]
enum Resolved {
    /// Still a variable: this one represents its class.
    Variable(Variable),
    /// A type containing no variable.
    Concrete,
    /// A constructed type with a variable inside it.
    Open,
}

fn resolve(variable: Variable, subst: &Subst) -> Resolved {
    match variable {
        Variable::Type(var) => {
            let ty = subst.apply(&Type::Var(var));
            match ty {
                Type::Var(representative) => Resolved::Variable(Variable::Type(representative)),
                ty if is_concrete(&ty) => Resolved::Concrete,
                _ => Resolved::Open,
            }
        }
        Variable::Dim(var) => match subst.constraint_dim(&Dim::Var(var)) {
            Dim::Var(representative) => Resolved::Variable(Variable::Dim(representative)),
            Dim::Rank(_) => Resolved::Open,
            Dim::Name(_) | Dim::Lit(_) | Dim::Wildcard => Resolved::Concrete,
        },
        Variable::Rank(var) => {
            let dims = subst.constraint_rank(var);
            match dims.as_slice() {
                [Dim::Rank(representative)] => Resolved::Variable(Variable::Rank(*representative)),
                dims if dims
                    .iter()
                    .all(|dim| !matches!(dim, Dim::Var(_) | Dim::Rank(_))) =>
                {
                    Resolved::Concrete
                }
                _ => Resolved::Open,
            }
        }
    }
}

/// [04-INF-2]'s "fully concrete type argument -- one containing no type
/// variable". An error witness is concrete: its cause is already reported.
fn is_concrete(ty: &Type) -> bool {
    crate::env::free_tvars(ty).is_empty()
        && crate::env::free_dvars(ty).is_empty()
        && crate::env::free_rvars(ty).is_empty()
}

/// The authored binders every holed member's references share, as they
/// resolve now, with their source names for diagnostics.
struct GroupBinders {
    names: Vec<(Variable, String)>,
}

impl GroupBinders {
    fn new(env: &Env) -> Self {
        let (tvars, dvars, rvars) = env.holed_group_binders();
        let names = tvars
            .into_iter()
            .map(|(name, var)| (Variable::Type(var), name))
            .chain(
                dvars
                    .into_iter()
                    .map(|(name, var)| (Variable::Dim(var), name)),
            )
            .chain(
                rvars
                    .into_iter()
                    .map(|(name, var)| (Variable::Rank(var), name)),
            )
            .collect();
        Self { names }
    }

    /// The variable classes the binders resolve to now.
    fn classes(&self, subst: &Subst) -> BTreeSet<Variable> {
        self.names
            .iter()
            .filter_map(|(binder, _)| match resolve(*binder, subst) {
                Resolved::Variable(class) => Some(class),
                _ => None,
            })
            .collect()
    }
}

/// The member's type parameters in its resolved type `ty`: the variables that
/// are free in no enclosing declaration (minted at the component's level or
/// deeper) and are not authored binders.
fn parameters(ty: &Type, binders: &BTreeSet<Variable>, subst: &Subst) -> Vec<Variable> {
    let level = subst.current_level();
    crate::env::free_tvars(ty)
        .into_iter()
        .filter(|var| subst.level_of_tvar(*var) >= level)
        .map(Variable::Type)
        .chain(
            crate::env::free_dvars(ty)
                .into_iter()
                .filter(|var| subst.level_of_dvar(*var) >= level)
                .map(Variable::Dim),
        )
        .chain(
            crate::env::free_rvars(ty)
                .into_iter()
                .filter(|var| subst.level_of_rvar(*var) >= level)
                .map(Variable::Rank),
        )
        .filter(|variable| !binders.contains(variable))
        .collect()
}

/// A fresh instance of `ty` over `parameters`: the type, and each parameter
/// paired with its instance.
fn instance(
    ty: &Type,
    parameters: &[Variable],
    env: &Env,
    vg: &mut VarGen,
    subst: &Subst,
) -> (Type, Vec<(Variable, Variable)>) {
    let mut scheme = Scheme::mono(ty.clone());
    for parameter in parameters {
        match parameter {
            Variable::Type(var) => scheme.tvars.push(*var),
            Variable::Dim(var) => scheme.dvars.push(*var),
            Variable::Rank(var) => scheme.rvars.push(*var),
        }
    }
    let instantiated = env.instantiate_scheme(&scheme, vg, subst);
    let pairs = instantiated
        .tvars
        .iter()
        .filter_map(|(from, to)| match to {
            Type::Var(fresh) => Some((Variable::Type(*from), Variable::Type(*fresh))),
            _ => None,
        })
        .chain(
            instantiated
                .dvars
                .iter()
                .map(|(from, to)| (Variable::Dim(*from), Variable::Dim(*to))),
        )
        .chain(
            instantiated
                .rvars
                .iter()
                .map(|(from, to)| (Variable::Rank(*from), Variable::Rank(*to))),
        )
        .collect();
    (instantiated.ty, pairs)
}

/// One reference's latest link.
struct Link {
    /// The callee's index among the group's members.
    callee: usize,
    /// The callee's type when the reference was linked, with its variables
    /// renamed in order of first occurrence, so a renaming alone is no change.
    shape: Type,
    /// Each of the callee's parameters, with the reference's instance of it.
    instances: Vec<(Variable, Variable)>,
}

/// Decide every in-group reference to a holed member of the completed
/// component (chelis#2590). Runs before the component's obligations are
/// decided, so a check waiting on a reference's type sees the linked type.
pub(super) fn link_group_references(
    product: &mut InferenceProduct,
    env: &Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    let (references, members) = product.take_group_links();
    if references.is_empty() {
        return;
    }
    let binders = GroupBinders::new(env);
    let member_index: BTreeMap<&str, usize> = members
        .iter()
        .enumerate()
        .map(|(index, (name, _))| (name.as_str(), index))
        .collect();
    let own_type = |name: &str| {
        member_index
            .get(name)
            .map(|index| (*index, &members[*index].1))
    };
    let mut links: Vec<Option<Link>> = references.iter().map(|_| None).collect();
    let mut settled = vec![false; references.len()];
    // A group whose types keep growing never settles. A settling group
    // settles within one round per member and reference: each round carries
    // every change at least one link further.
    let bound = references.len() + members.len() + 2;
    let mut round = 0;
    loop {
        product.replay_ready_shape_checks(vg, subst, adt_reg, errors);
        let binder_classes = binders.classes(subst);
        let order: Vec<usize> = if round % 2 == 0 {
            (0..references.len()).collect()
        } else {
            (0..references.len()).rev().collect()
        };
        let mut relinked = Vec::new();
        for index in order {
            if settled[index] {
                continue;
            }
            let reference = &references[index];
            let Some((callee, own)) = own_type(&reference.callee) else {
                // The callee's body was never inferred against its
                // provisional scheme (its signature was rejected), so there
                // is no body-determined type to decide the reference by.
                settled[index] = true;
                continue;
            };
            let own = subst.apply(own);
            let shape = renamed(&own, &mut Renaming::default());
            if links[index]
                .as_ref()
                .is_some_and(|link| link.shape == shape)
            {
                continue;
            }
            let (linked, instances) = instance(
                &own,
                &parameters(&own, &binder_classes, subst),
                env,
                vg,
                subst,
            );
            if let Err(error) = unify(&reference.ty, &linked, subst) {
                let witness = report_witness(
                    errors,
                    disagreement(reference, &own, &error, &binders, subst),
                );
                // chelis#731 §C3: the reference's variables that are no
                // member's parameter take the reported error's witness, so a
                // check waiting on one stays silent rather than reporting the
                // same mismatch again.
                let used = subst.apply(&reference.ty);
                let owned = members
                    .iter()
                    .flat_map(|(_, ty)| parameters(&subst.apply(ty), &binder_classes, subst))
                    .collect::<BTreeSet<_>>();
                for var in parameters(&used, &binder_classes, subst) {
                    if let (false, Variable::Type(var)) = (owned.contains(&var), var) {
                        let _ = unify(&Type::Var(var), &propagate(&witness), subst);
                    }
                }
                settled[index] = true;
            }
            links[index] = Some(Link {
                callee,
                shape,
                instances,
            });
            relinked.push(index);
        }
        if relinked.is_empty() {
            break;
        }
        round += 1;
        if round > bound {
            for index in relinked {
                errors.push(unsettled(&references[index]));
                settled[index] = true;
            }
            break;
        }
    }
    decide_instances(
        &references,
        &links,
        &settled,
        &members,
        &binders,
        subst,
        errors,
    );
}

/// [04-INF-2]: each parameter's instance is concrete or the parameter itself.
fn decide_instances(
    references: &[GroupReference],
    links: &[Option<Link>],
    settled: &[bool],
    members: &[(String, Type)],
    binders: &GroupBinders,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    // Which members have a parameter in each variable class, so choosing an
    // instance as its parameter can refuse to identify two of one member's
    // parameters.
    let binder_classes = binders.classes(subst);
    let mut owners: BTreeMap<Variable, BTreeSet<usize>> = BTreeMap::new();
    for (member, (_, ty)) in members.iter().enumerate() {
        for parameter in parameters(&subst.apply(ty), &binder_classes, subst) {
            if let Resolved::Variable(class) = resolve(parameter, subst) {
                owners.entry(class).or_default().insert(member);
            }
        }
    }
    for (index, reference) in references.iter().enumerate() {
        let (false, Some(link)) = (settled[index], &links[index]) else {
            continue;
        };
        let mut admitted = true;
        for (parameter, instance) in &link.instances {
            let own = resolve(*parameter, subst);
            let chosen = resolve(*instance, subst);
            if chosen == own || chosen == Resolved::Concrete {
                continue;
            }
            let (Resolved::Variable(own), Resolved::Variable(chosen)) = (own, chosen) else {
                admitted = false;
                continue;
            };
            // A class no member has a parameter in belongs to no member.
            let members_of = |class: &Variable| match owners.get(class) {
                Some(members) => members.clone(),
                None => BTreeSet::new(),
            };
            let own_members = members_of(&own);
            let chosen_members = members_of(&chosen);
            if own_members.intersection(&chosen_members).next().is_some()
                || identify(own, chosen, subst).is_err()
            {
                admitted = false;
                continue;
            }
            owners.remove(&own);
            owners.remove(&chosen);
            if let Resolved::Variable(class) = resolve(*parameter, subst) {
                owners.insert(class, own_members.union(&chosen_members).copied().collect());
            }
        }
        if !admitted {
            let own = subst.apply(&members[link.callee].1);
            errors.push(not_admitted(reference, &own, binders, subst));
        }
    }
}

/// Choose `chosen` as the parameter `own`: unify the two variables.
fn identify(own: Variable, chosen: Variable, subst: &mut Subst) -> Result<(), TypeError> {
    match (own, chosen) {
        (Variable::Type(own), Variable::Type(chosen)) => {
            unify(&Type::Var(own), &Type::Var(chosen), subst)
        }
        (Variable::Dim(own), Variable::Dim(chosen)) => {
            unify_dim(&Dim::Var(own), &Dim::Var(chosen), subst)
        }
        (Variable::Rank(own), Variable::Rank(chosen)) => unify(
            &Type::Tensor(vec![Dim::Rank(own)], TensorPrec::Concrete(Prim::F32)),
            &Type::Tensor(vec![Dim::Rank(chosen)], TensorPrec::Concrete(Prim::F32)),
            subst,
        ),
        _ => Err(TypeError {
            kind: TypeErrorKind::TypeMismatch,
            message: "a type variable and a dimension or rank variable".to_string(),
        }),
    }
}

/// Variables renamed in order of first occurrence. Authored binders keep their
/// source names when rendered for a diagnostic.
#[derive(Default)]
struct Renaming {
    types: Vec<TypeVar>,
    dims: Vec<DimVar>,
    ranks: Vec<RankVar>,
    binder_names: Vec<(Variable, String)>,
}

impl Renaming {
    fn for_diagnostic(binders: &GroupBinders, subst: &Subst) -> Self {
        Self {
            binder_names: binders
                .names
                .iter()
                .filter_map(|(binder, name)| match resolve(*binder, subst) {
                    Resolved::Variable(class) => Some((class, name.clone())),
                    _ => None,
                })
                .collect(),
            ..Self::default()
        }
    }

    fn binder_name(&self, variable: Variable) -> Option<&str> {
        self.binder_names
            .iter()
            .find_map(|(binder, name)| (*binder == variable).then_some(name.as_str()))
    }

    fn position<T: PartialEq + Copy>(seen: &mut Vec<T>, var: T) -> u32 {
        let index = seen
            .iter()
            .position(|known| *known == var)
            .unwrap_or_else(|| {
                seen.push(var);
                seen.len() - 1
            });
        u32::try_from(index).expect("a signature has fewer than 2^32 variables")
    }
}

fn renamed(ty: &Type, renaming: &mut Renaming) -> Type {
    match ty {
        Type::Var(var) => match renaming.binder_name(Variable::Type(*var)) {
            Some(name) => Type::Adt(name.to_string(), Vec::new()),
            None => Type::Var(TypeVar(Renaming::position(&mut renaming.types, *var))),
        },
        Type::Prim(_) | Type::Unit | Type::Error(_) => ty.clone(),
        Type::Fn(args, ret) => Type::Fn(
            args.iter().map(|arg| renamed(arg, renaming)).collect(),
            Box::new(renamed(ret, renaming)),
        ),
        Type::Ref(inner) => Type::Ref(Box::new(renamed(inner, renaming))),
        Type::Tensor(dims, precision) => Type::Tensor(
            dims.iter().map(|dim| renamed_dim(dim, renaming)).collect(),
            match precision {
                TensorPrec::Var(var) => {
                    TensorPrec::Var(TypeVar(Renaming::position(&mut renaming.types, *var)))
                }
                TensorPrec::Concrete(_) => precision.clone(),
            },
        ),
        Type::Adt(name, args) => Type::Adt(
            name.clone(),
            args.iter().map(|arg| renamed(arg, renaming)).collect(),
        ),
        Type::KindedAdt(name, args) => Type::KindedAdt(
            name.clone(),
            args.iter()
                .map(|arg| match arg {
                    NominalArg::Type(ty) => NominalArg::Type(renamed(ty, renaming)),
                    NominalArg::Dimension(dim) => NominalArg::Dimension(renamed_dim(dim, renaming)),
                })
                .collect(),
        ),
        Type::Tuple(elements) => Type::Tuple(
            elements
                .iter()
                .map(|element| renamed(element, renaming))
                .collect(),
        ),
    }
}

fn renamed_dim(dim: &Dim, renaming: &mut Renaming) -> Dim {
    match dim {
        Dim::Var(var) => match renaming.binder_name(Variable::Dim(*var)) {
            Some(name) => Dim::Name(name.to_string()),
            None => Dim::Var(DimVar(Renaming::position(&mut renaming.dims, *var))),
        },
        Dim::Rank(var) => Dim::Rank(RankVar(Renaming::position(&mut renaming.ranks, *var))),
        Dim::Name(_) | Dim::Lit(_) | Dim::Wildcard => dim.clone(),
    }
}

/// The callee's body-determined type and the reference's type, rendered with
/// one renaming so a variable they share reads the same in both.
fn rendered_pair(
    own: &Type,
    reference: &Type,
    binders: &GroupBinders,
    subst: &Subst,
) -> (String, String) {
    let mut renaming = Renaming::for_diagnostic(binders, subst);
    let own = renamed(own, &mut renaming).to_string();
    let reference = renamed(&subst.apply(reference), &mut renaming).to_string();
    (own, reference)
}

fn caller_name(reference: &GroupReference) -> &str {
    reference.caller.as_deref().unwrap_or("<top level>")
}

fn at_reference(mut error: CheckError, reference: &GroupReference) -> CheckError {
    error.span_id.clone_from(&reference.span_id);
    error.span_offset = reference.span_offset;
    error
}

/// [04-INF-5]: the reference's use disagrees with the body-determined type.
fn disagreement(
    reference: &GroupReference,
    own: &Type,
    cause: &TypeError,
    binders: &GroupBinders,
    subst: &Subst,
) -> CheckError {
    let (own, used) = rendered_pair(own, &reference.ty, binders, subst);
    let mut error = CheckError::new(
        CheckErrorKind::TypeMismatch,
        format!(
            "`{caller}` uses `{callee}` at `{used}`, which disagrees with the type \
             `{callee}`'s body determines, `{own}`: {cause} (spec/04-type-system.md \
             \u{a7}3.1.3 [04-INF-5])",
            caller = caller_name(reference),
            callee = reference.callee,
            cause = cause.message,
        ),
        vec![format!(
            "`{callee}` omits a type in its signature, so that slot is the type its body \
             determines; a reference is typed at it, and the body is never narrowed to fit \
             a reference",
            callee = reference.callee,
        )],
    );
    error.expected = Some(own);
    error.got = Some(used);
    at_reference(error, reference)
}

/// [04-INF-3]: an instance that is neither concrete nor the parameter itself.
fn not_admitted(
    reference: &GroupReference,
    own: &Type,
    binders: &GroupBinders,
    subst: &Subst,
) -> CheckError {
    let (own, used) = rendered_pair(own, &reference.ty, binders, subst);
    let mut error = CheckError::new(
        CheckErrorKind::TypeMismatch,
        format!(
            "polymorphic recursion: `{caller}` makes an in-group reference to `{callee}` at \
             `{used}`, but `{callee}`'s body determines `{own}`, and an in-group reference \
             may instantiate a type `{callee}` omits only at `{callee}`'s own type or at a \
             fully concrete type (spec/04-type-system.md \u{a7}3.1.1 [04-INF-2], [04-INF-3])",
            caller = caller_name(reference),
            callee = reference.callee,
        ),
        vec![
            format!(
                "make the reference use `{callee}`'s own type or a fully concrete type",
                callee = reference.callee,
            ),
            "hoist the changed-instantiation call into a separate non-recursive helper `def`"
                .to_string(),
        ],
    );
    error.expected = Some(own);
    error.got = Some(used);
    at_reference(error, reference)
}

/// [04-INF-3]: linking never settles, because each round instantiates the
/// omitted types at larger types.
fn unsettled(reference: &GroupReference) -> CheckError {
    at_reference(
        CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!(
                "polymorphic recursion: `{caller}` makes an in-group reference to `{callee}` \
                 whose instantiation of the types `{callee}` omits grows without bound \
                 (spec/04-type-system.md \u{a7}3.1.1 [04-INF-2], [04-INF-3])",
                caller = caller_name(reference),
                callee = reference.callee,
            ),
            vec![
                "hoist the changed-instantiation call into a separate non-recursive helper \
                 `def`"
                    .to_string(),
            ],
        ),
        reference,
    )
}
