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
//! The decision is a predicate on the group's final solution, never on a
//! state that some processing order reaches first, so it cannot depend on
//! the order of the group's declarations or references:
//!
//! 1. [`solve`] links every reference and solves the group's constraints to a
//!    fixed point, refusing nothing. A member's type parameters are the
//!    variables of its body-determined type that are neither authored binders
//!    nor free outside the group. Each reference is unified with a fresh
//!    instance of that type, one variable per parameter, so the reference is
//!    typed at the body-determined type and never narrows it ([04-INF-5]: "a
//!    body is never narrowed to satisfy a reference"). A link can determine
//!    more of a member's type (the member's result may be another member's
//!    result), so the links are repeated in rounds. Each round instantiates
//!    every reference whose callee changed shape, from the types the previous
//!    round left, before it unifies any of them, so each round's equations,
//!    and the solution they reach, are the same in every order. The fixed
//!    point is the group's most general solution, unique up to renaming. A
//!    group whose types keep growing never reaches one; that is polymorphic
//!    recursion, and a bound on the rounds reports it.
//! 2. [`decide`] applies [04-INF-2] to that solution. Every instance that is
//!    still a variable is chosen as its parameter, all in one batch. The group
//!    is then rejected at a reference whose instance is neither its parameter
//!    nor fully concrete, and at the references of a member two of whose own
//!    types the group made one type: two of its parameters' types, distinct
//!    before any reference was linked, or two of its type parameters and
//!    binders, distinct in the solution. An in-group call that swaps or
//!    merges a member's omitted types makes such a merge, which would narrow
//!    the member's published signature to fit a reference.

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
            let ty = resolved(&Type::Var(var), subst);
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
                dims if dims.iter().all(|dim| !is_variable_dim(dim)) => Resolved::Concrete,
                _ => Resolved::Open,
            }
        }
    }
}

/// `ty` with every bound variable resolved. `Subst::apply` resolves a
/// variable to its binding as the binding was recorded, and a binding
/// recorded before one of its own variables was bound still names that
/// variable, so the application is repeated until nothing changes.
pub(super) fn resolved(ty: &Type, subst: &Subst) -> Type {
    let mut current = subst.apply(ty);
    loop {
        let next = subst.apply(&current);
        if next == current {
            return current;
        }
        current = next;
    }
}

fn is_variable_dim(dim: &Dim) -> bool {
    matches!(dim, Dim::Var(_) | Dim::Rank(_))
}

/// Everything a variable denotes under the substitution: two variables of
/// one kind denote one type exactly when these are equal.
#[derive(PartialEq)]
enum Denotation {
    Type(Type),
    Dim(Dim),
    Rank(Vec<Dim>),
}

impl Denotation {
    fn of(variable: Variable, subst: &Subst) -> Self {
        match variable {
            Variable::Type(var) => Self::Type(resolved(&Type::Var(var), subst)),
            Variable::Dim(var) => Self::Dim(subst.constraint_dim(&Dim::Var(var))),
            Variable::Rank(var) => Self::Rank(subst.constraint_rank(var)),
        }
    }

    fn is_concrete(&self) -> bool {
        match self {
            Self::Type(ty) => is_concrete(ty),
            Self::Dim(dim) => !is_variable_dim(dim),
            Self::Rank(dims) => dims.iter().all(|dim| !is_variable_dim(dim)),
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

/// The authored binders every holed member's references share, with their
/// source names for diagnostics.
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

/// Whether `variable` is free in no enclosing declaration: minted at the
/// component's level or deeper.
fn in_group(variable: Variable, subst: &Subst) -> bool {
    let level = subst.current_level();
    match variable {
        Variable::Type(var) => subst.level_of_tvar(var) >= level,
        Variable::Dim(var) => subst.level_of_dvar(var) >= level,
        Variable::Rank(var) => subst.level_of_rvar(var) >= level,
    }
}

/// Every variable of the resolved type `ty`.
fn variables(ty: &Type) -> impl Iterator<Item = Variable> {
    crate::env::free_tvars(ty)
        .into_iter()
        .map(Variable::Type)
        .chain(crate::env::free_dvars(ty).into_iter().map(Variable::Dim))
        .chain(crate::env::free_rvars(ty).into_iter().map(Variable::Rank))
}

/// The group variables of the resolved type `ty`, each marked as an authored
/// binder or not.
fn own_variables(ty: &Type, binders: &BTreeSet<Variable>, subst: &Subst) -> Vec<(Variable, bool)> {
    variables(ty)
        .filter(|variable| in_group(*variable, subst))
        .map(|variable| (variable, binders.contains(&variable)))
        .collect()
}

/// The member's type parameters in its resolved type `ty`: its group
/// variables that are not authored binders.
fn parameters(ty: &Type, binders: &BTreeSet<Variable>, subst: &Subst) -> Vec<Variable> {
    own_variables(ty, binders, subst)
        .into_iter()
        .filter_map(|(variable, binder)| (!binder).then_some(variable))
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

/// A type up to renaming of its variables, each marked as a parameter or
/// not. Two types of one shape have one fresh instance up to renaming,
/// whichever variables happen to represent their classes.
#[derive(PartialEq)]
struct Shape {
    ty: Type,
    /// Kind by kind, in order of first occurrence, whether each variable is
    /// a parameter.
    parameters: [Vec<bool>; 3],
}

impl Shape {
    fn of(ty: &Type, parameters: &[Variable]) -> Self {
        let mut renaming = Renaming::default();
        let ty = renamed(ty, &mut renaming);
        let marks = |variables: &mut dyn Iterator<Item = Variable>| {
            variables
                .map(|variable| parameters.contains(&variable))
                .collect::<Vec<_>>()
        };
        Self {
            ty,
            parameters: [
                marks(&mut renaming.types.iter().map(|var| Variable::Type(*var))),
                marks(&mut renaming.dims.iter().map(|var| Variable::Dim(*var))),
                marks(&mut renaming.ranks.iter().map(|var| Variable::Rank(*var))),
            ],
        }
    }
}

/// A member's type as one round of linking reads it.
struct Snapshot {
    ty: Type,
    parameters: Vec<Variable>,
    shape: Shape,
    /// Every variable of `ty`, and whether it is a parameter. While each one
    /// still represents its own class in the same role, the member's type is
    /// still `ty`, and a round need not rebuild the snapshot.
    leaves: Vec<(Variable, bool)>,
}

impl Snapshot {
    fn new(ty: &Type, binders: &BTreeSet<Variable>, subst: &Subst) -> Self {
        let ty = resolved(ty, subst);
        let parameters = parameters(&ty, binders, subst);
        let leaves = variables(&ty)
            .map(|variable| (variable, parameters.contains(&variable)))
            .collect();
        let shape = Shape::of(&ty, &parameters);
        Self {
            ty,
            parameters,
            shape,
            leaves,
        }
    }

    fn unchanged(&self, binders: &BTreeSet<Variable>, subst: &Subst) -> bool {
        self.leaves.iter().all(|(variable, parameter)| {
            resolve(*variable, subst) == Resolved::Variable(*variable)
                && *parameter == (!binders.contains(variable) && in_group(*variable, subst))
        })
    }
}

/// One reference's latest link.
struct Link {
    /// The callee's index among the group's members.
    callee: usize,
    /// Each of the callee's parameters, with the reference's instance of it.
    instances: Vec<(Variable, Variable)>,
}

/// A member's own types before any reference is linked, as its body alone
/// determines them.
struct Local {
    /// The member's type then, for diagnostics.
    ty: Type,
    /// The group variables of its parameters' types, each marked as an
    /// authored binder or not.
    arguments: Vec<(Variable, bool)>,
}

impl Local {
    fn new(ty: &Type, binders: &BTreeSet<Variable>, subst: &Subst) -> Self {
        let ty = resolved(ty, subst);
        let arguments = match &ty {
            Type::Fn(arguments, _) => {
                own_variables(&Type::Tuple(arguments.clone()), binders, subst)
            }
            _ => Vec::new(),
        };
        Self { ty, arguments }
    }
}

/// The references and members of one completed group.
struct Group<'a> {
    references: &'a [GroupReference],
    members: &'a [(String, Type)],
    binders: GroupBinders,
    /// Each reference's callee among the members. A callee whose signature
    /// was rejected was never inferred against its provisional scheme, so it
    /// has no body-determined type to decide a reference by.
    callees: Vec<Option<usize>>,
}

impl<'a> Group<'a> {
    fn new(references: &'a [GroupReference], members: &'a [(String, Type)], env: &Env) -> Self {
        let member_index: BTreeMap<&str, usize> = members
            .iter()
            .enumerate()
            .map(|(index, (name, _))| (name.as_str(), index))
            .collect();
        let callees = references
            .iter()
            .map(|reference| member_index.get(reference.callee.as_str()).copied())
            .collect();
        Self {
            references,
            members,
            binders: GroupBinders::new(env),
            callees,
        }
    }
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
    let group = Group::new(&references, &members, env);
    product.replay_ready_shape_checks(vg, subst, adt_reg, errors);
    let binder_classes = group.binders.classes(subst);
    let locals = members
        .iter()
        .map(|(_, ty)| Local::new(ty, &binder_classes, subst))
        .collect::<Vec<_>>();
    if let Some(links) = solve(&group, product, env, vg, subst, adt_reg, errors) {
        decide(&group, &links, &locals, subst, errors);
    }
}

/// Link every reference until no member's type changes shape, refusing
/// nothing. Returns each reference's latest link, or `None` when a reference
/// could not be linked: the group is then rejected at that reference, and
/// there is no solution to make the [04-INF-2] decision on.
#[allow(clippy::too_many_arguments)]
fn solve(
    group: &Group<'_>,
    product: &mut InferenceProduct,
    env: &Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Vec<Option<Link>>> {
    let references = group.references;
    let mut links: Vec<Option<Link>> = references.iter().map(|_| None).collect();
    let mut failed = vec![false; references.len()];
    let mut snapshots: Vec<Option<Snapshot>> = group.members.iter().map(|_| None).collect();
    // A group whose types keep growing never settles. A settling group
    // settles within one round per member and reference: each round carries
    // every change at least one link further.
    let bound = references.len() + group.members.len() + 2;
    let mut round = 0;
    loop {
        product.replay_ready_shape_checks(vg, subst, adt_reg, errors);
        let binder_classes = group.binders.classes(subst);
        let mut changed = vec![false; group.members.len()];
        for (member, (_, ty)) in group.members.iter().enumerate() {
            if snapshots[member]
                .as_ref()
                .is_some_and(|snapshot| snapshot.unchanged(&binder_classes, subst))
            {
                continue;
            }
            let next = Snapshot::new(ty, &binder_classes, subst);
            changed[member] = snapshots[member]
                .as_ref()
                .is_none_or(|snapshot| snapshot.shape != next.shape);
            snapshots[member] = Some(next);
        }
        let stale = (0..references.len())
            .filter(|index| {
                !failed[*index] && group.callees[*index].is_some_and(|callee| changed[callee])
            })
            .collect::<Vec<_>>();
        if stale.is_empty() {
            break;
        }
        if round == bound {
            for index in stale {
                errors.push(unsettled(&references[index]));
            }
            return None;
        }
        round += 1;
        // Every stale reference is instantiated from the types the previous
        // round left before any of them is unified.
        let linked = stale
            .into_iter()
            .filter_map(|index| {
                let callee = group.callees[index]?;
                let snapshot = snapshots[callee].as_ref()?;
                let (ty, instances) = instance(&snapshot.ty, &snapshot.parameters, env, vg, subst);
                let link = Link { callee, instances };
                Some((index, snapshot.ty.clone(), ty, link))
            })
            .collect::<Vec<_>>();
        for (index, own, ty, link) in linked {
            let reference = &references[index];
            if let Err(error) = unify(&reference.ty, &ty, subst) {
                let witness = report_witness(
                    errors,
                    disagreement(reference, &own, &error, &group.binders, subst),
                );
                // chelis#731 §C3: the reference's variables that are no
                // member's parameter take the reported error's witness, so a
                // check waiting on one stays silent rather than reporting the
                // same mismatch again.
                let binder_classes = group.binders.classes(subst);
                let used = resolved(&reference.ty, subst);
                let owned = group
                    .members
                    .iter()
                    .flat_map(|(_, ty)| parameters(&resolved(ty, subst), &binder_classes, subst))
                    .collect::<BTreeSet<_>>();
                for var in parameters(&used, &binder_classes, subst) {
                    if let (false, Variable::Type(var)) = (owned.contains(&var), var) {
                        let _ = unify(&Type::Var(var), &propagate(&witness), subst);
                    }
                }
                failed[index] = true;
            }
            links[index] = Some(link);
        }
    }
    (!failed.contains(&true)).then_some(links)
}

/// [04-INF-2] on the solved group: choose every instance that is still a
/// variable as its parameter, then reject an instance that is neither its
/// parameter nor fully concrete, and a member two of whose own types the
/// group made one type.
fn decide(
    group: &Group<'_>,
    links: &[Option<Link>],
    locals: &[Local],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    let references = group.references;
    // Each member's own variables in the solution: its type parameters and
    // the authored binders its type mentions.
    let binder_classes = group.binders.classes(subst);
    let solved = group
        .members
        .iter()
        .map(|(_, ty)| own_variables(&resolved(ty, subst), &binder_classes, subst))
        .collect::<Vec<_>>();
    // An instance that is still a variable "remains unconstrained and is
    // thereby chosen as" its parameter. Every choice is read from the
    // solution before any is made.
    let mut refused = vec![false; references.len()];
    let mut choices = Vec::new();
    for (index, link) in links.iter().enumerate() {
        let Some(link) = link else {
            continue;
        };
        for (parameter, instance) in &link.instances {
            if let (Resolved::Variable(own), Resolved::Variable(chosen)) =
                (resolve(*parameter, subst), resolve(*instance, subst))
                && own != chosen
            {
                choices.push((index, own, chosen));
            }
        }
    }
    for (index, own, chosen) in choices {
        if identify(own, chosen, subst).is_err() {
            refused[index] = true;
        }
    }
    // Every instance must now be its parameter or fully concrete.
    for (index, link) in links.iter().enumerate() {
        let Some(link) = link else {
            continue;
        };
        refused[index] |= link.instances.iter().any(|(parameter, instance)| {
            let chosen = resolve(*instance, subst);
            chosen != Resolved::Concrete && chosen != resolve(*parameter, subst)
        });
    }
    // A member two of whose own types are now one type is reported at the
    // references its own body makes, the calls its author wrote, or at the
    // references to it when its body makes none.
    let mut merged_at: Vec<Option<usize>> = references.iter().map(|_| None).collect();
    for (member, (name, _)) in group.members.iter().enumerate() {
        if !merges(&locals[member].arguments, subst) && !merges(&solved[member], subst) {
            continue;
        }
        let linked = |index: &usize| links[*index].is_some();
        let made = (0..references.len())
            .filter(linked)
            .filter(|index| references[*index].caller.as_deref() == Some(name.as_str()))
            .collect::<Vec<_>>();
        let sites = if made.is_empty() {
            (0..references.len())
                .filter(linked)
                .filter(|index| group.callees[*index] == Some(member))
                .collect()
        } else {
            made
        };
        for index in sites {
            merged_at[index].get_or_insert(member);
        }
    }
    for index in 0..references.len() {
        let Some(link) = &links[index] else {
            continue;
        };
        let reference = &references[index];
        if refused[index] {
            let own = resolved(&group.members[link.callee].1, subst);
            errors.push(not_admitted(reference, &own, &group.binders, subst));
        } else if let Some(member) = merged_at[index] {
            errors.push(merged(
                reference,
                &group.members[member].0,
                &locals[member].ty,
                &resolved(&group.members[member].1, subst),
                &group.binders,
                subst,
            ));
        }
    }
}

/// Whether two of `variables`, distinct where they were collected, now
/// denote one type that is not fully concrete. Two authored binders made one
/// are [04-INF-6]'s to report, not this rule's.
fn merges(variables: &[(Variable, bool)], subst: &Subst) -> bool {
    let denotations = variables
        .iter()
        .map(|(variable, binder)| (Denotation::of(*variable, subst), *binder))
        .collect::<Vec<_>>();
    denotations
        .iter()
        .enumerate()
        .any(|(index, (first, first_binder))| {
            !first.is_concrete()
                && denotations[index + 1..]
                    .iter()
                    .any(|(second, second_binder)| {
                        !(*first_binder && *second_binder) && first == second
                    })
        })
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

/// `ty` rendered for a diagnostic, its variables renamed in order of first
/// occurrence.
fn rendered(ty: &Type, binders: &GroupBinders, subst: &Subst) -> String {
    renamed(ty, &mut Renaming::for_diagnostic(binders, subst)).to_string()
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
    let reference = renamed(&resolved(reference, subst), &mut renaming).to_string();
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

/// [04-INF-3]: an instance that is neither the parameter nor fully concrete.
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
        vec![format!(
            "make the reference use `{callee}`'s own type or a fully concrete type",
            callee = reference.callee,
        )],
    );
    error.expected = Some(own);
    error.got = Some(used);
    at_reference(error, reference)
}

/// [04-INF-3]: the group's in-group references make two of `member`'s own
/// types one type.
fn merged(
    reference: &GroupReference,
    member: &str,
    local: &Type,
    solved: &Type,
    binders: &GroupBinders,
    subst: &Subst,
) -> CheckError {
    let local = rendered(local, binders, subst);
    let solved = rendered(solved, binders, subst);
    let used = rendered(&resolved(&reference.ty, subst), binders, subst);
    let mut error = CheckError::new(
        CheckErrorKind::TypeMismatch,
        format!(
            "polymorphic recursion: `{caller}` makes an in-group reference to `{callee}` at \
             `{used}`, and the in-group references of its recursive group make two of the \
             types `{member}` omits one type: `{member}`'s body alone determines `{local}`, \
             and the group would make it `{solved}`. An in-group reference may instantiate a \
             type that a member omits only at the member's own type or at a fully concrete \
             type, so it never swaps or merges two of them (spec/04-type-system.md \
             \u{a7}3.1.1 [04-INF-2], [04-INF-3])",
            caller = caller_name(reference),
            callee = reference.callee,
        ),
        vec![format!(
            "write `{member}`'s signature with explicit type binders, for example \
             `def {member}[t, u](...)`, giving two of its types one binder only where they \
             are the same type (spec/04-type-system.md \u{a7}3.1.1 [04-INF-2])"
        )],
    );
    error.expected = Some(local);
    error.got = Some(solved);
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
            vec![format!(
                "make the reference use `{callee}`'s own type or a fully concrete type",
                callee = reference.callee,
            )],
        ),
        reference,
    )
}
