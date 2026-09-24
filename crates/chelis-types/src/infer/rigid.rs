//! [04-INF-6]: authored type, dimension, and rank binders are rigid in the body.
//!
//! `spec/04-type-system.md` §3.1.3 makes every explicitly authored binder
//! universally quantified and rigid within the declaration's body:
//! the body must type-check for every admissible instantiation. A body
//! constraint that identifies the binder with a concrete type, with another
//! authored binder of the same signature, or with a type/shape containing
//! either is a type error at the declaration, and the declaration's scheme
//! stays the declared signature.
//!
//! The type and rank checks here run beside §4.4's dimension rule, enforced by
//! [`super::app_shape_helpers::check_declared_dvars_rigid`], immediately after
//! post-body signature unification. In addition to concrete pins and collapsed
//! binders, the type check rejects a body that requires a narrower dtype family
//! than its authored contract permits.
//!
//! A call suspended on an operand that an authored binder denotes is decided
//! here too, at the declaration boundary, by replaying it at the binder's
//! instantiations ([`decide_at_rigid_binder_instantiations`], chelis#2216).
//!
//! An inference hole (`(t-var {} _)`) is NOT a binder and is not checked here.
//! [04-INF-5] gives the hole the type its body determines; it never reaches
//! the declaration-owned identity maps because the binder list never includes
//! `_`.

use chelis_unord::UnordMap;

use super::app_helpers::type_for_readonly_check;
use super::checked::{InferenceProduct, PostAppCall, PostAppReplay};
use super::expr_pattern::family_members;
use crate::adt::AdtRegistry;
use crate::env::{free_dvars, free_rvars, free_tvars};
use crate::errors::{CheckError, CheckErrorKind};
use crate::session::DiagnosticSink;
use crate::types::{Dim, Prim, RankVar, Type, TypeVar, TypeVarRestriction, VarGen};
use crate::unify::{Subst, unify};

/// Render an authored binder for a diagnostic, falling back to the internal id
/// rather than inventing a name (the `render_declared_dim` convention).
fn render_declared_binder(type_names: &UnordMap<TypeVar, String>, tv: TypeVar) -> String {
    match type_names.get(&tv) {
        Some(name) => format!("`{name}`"),
        None => format!("t{}", tv.0),
    }
}

fn render_declared_rank(rank_names: &UnordMap<RankVar, String>, rv: RankVar) -> String {
    match rank_names.get(&rv) {
        Some(name) => format!("`{name}`"),
        None => format!("r{}", rv.0),
    }
}

/// Post-body rigidity check for a declaration's authored type binders
/// ([04-INF-6]).
///
/// `type_names` maps the FRESH variables this declaration's signature was
/// instantiated at to the names the source wrote, and its key set is exactly
/// the declaration's authored binders. After the body has been unified with
/// the declared signature, each of those variables must still resolve to a
/// bare type variable, and no two of them may resolve to the same one:
///
/// - resolving to anything other than a `Type::Var` means the body identified
///   the binder with a concrete type, or with a type containing another
///   binder (`a := List[b]`), which the atom rejects in the same breath;
/// - two binders resolving to the same variable means the body unified them
///   (`def g[a, b](x: a, y: b) -> a = y`), so the declaration no longer holds
///   for every admissible pair.
///
/// A binder that resolves to a variable other than its own instance is the
/// legitimate polymorphic case: the body's own fresh variable and the
/// signature's instance are two names for one unconstrained type.
///
/// A dtype-family bound ([04-DTYPE-2]) narrows which primitives may
/// instantiate the binder without making it concrete, so a bounded binder is
/// checked exactly like an unbounded one: `mul(x, cast(0.0, p))` leaves `p` a
/// variable and passes, while `mul(x, 0.0)` pins it to `f32` and does not
/// (`spec/02-surf-syntax.md` §P10 gives the unsuffixed literal its default
/// primitive type).
pub(super) fn check_declared_tvars_rigid(
    declaration: &str,
    type_names: &UnordMap<TypeVar, String>,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    if type_names.is_empty() {
        return;
    }
    // Deterministic order: `UnordMap` offers no `iter` precisely so hash order
    // cannot reach a diagnostic (chelis#1444), and the collapse report names
    // whichever binder was seen first.
    let mut binders = type_names
        .to_sorted()
        .into_iter()
        .map(|(tv, _)| *tv)
        .collect::<Vec<_>>();
    binders.sort_by_key(|tv| tv.0);

    // First resolved variable seen -> the authored binder that produced it.
    let mut seen: Vec<(TypeVar, TypeVar)> = Vec::new();
    for binder in binders {
        let resolved = subst.apply(&Type::Var(binder));
        let Type::Var(resolved_var) = resolved else {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "declared type parameter {} of `{declaration}` was narrowed to \
                     `{resolved}` by the function body: an authored type binder is rigid and \
                     the body must type-check for every instantiation \
                     (spec/04-type-system.md §3.1.3 [04-INF-6])",
                    render_declared_binder(type_names, binder),
                ),
                vec![
                    "Write the concrete type in the signature, or keep the body polymorphic. An \
                     unsuffixed literal binds at its default primitive type \
                     (spec/02-surf-syntax.md §P10); use `cast(<literal>, <binder>)` to write it \
                     at the binder's type instead."
                        .to_string(),
                ],
            ));
            continue;
        };
        if let Some((_, previous)) = seen
            .iter()
            .find(|(candidate, _)| *candidate == resolved_var)
        {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "distinct declared type parameters {} and {} of `{declaration}` were unified \
                     by the function body: authored type binders are rigid and must remain \
                     distinct (spec/04-type-system.md §3.1.3 [04-INF-6])",
                    render_declared_binder(type_names, *previous),
                    render_declared_binder(type_names, binder)
                ),
                vec![
                    "Use the same binder on both sides if they are meant to be equal, or fix the \
                     body so each declared binder stays independent."
                        .to_string(),
                ],
            ));
        } else {
            seen.push((resolved_var, binder));
        }
    }
}

/// Post-body rigidity check for a declaration's authored rank binders
/// ([04-INF-6]).
///
/// A rank binder may remain an unbound rank variable or alias another
/// unconstrained rank identity during signature/body reconciliation. Binding
/// it to any concrete shape run fixes its rank and violates the universal
/// declaration. Two authored rank binders may not collapse to the same
/// surviving rank identity.
pub(super) fn check_declared_rvars_rigid(
    declaration: &str,
    rank_names: &UnordMap<RankVar, String>,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    let mut binders = rank_names
        .to_sorted()
        .into_iter()
        .map(|(rv, _)| *rv)
        .collect::<Vec<_>>();
    binders.sort_by_key(|rv| rv.0);

    let mut seen: Vec<(RankVar, RankVar)> = Vec::new();
    for binder in binders {
        let resolved = subst.constraint_rank(binder);
        let [Dim::Rank(resolved_var)] = resolved.as_slice() else {
            let shape = resolved
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "declared rank parameter {} of `{declaration}` was narrowed to concrete \
                     shape [{shape}] by the function body: an authored rank binder is rigid and \
                     the body must type-check for every rank \
                     (spec/04-type-system.md §3.1.3 [04-INF-6])",
                    render_declared_rank(rank_names, binder),
                ),
                vec![
                    "Write the concrete shape in the annotation, or keep the body polymorphic in \
                     the declared rank."
                        .to_string(),
                ],
            ));
            continue;
        };
        if let Some((_, previous)) = seen.iter().find(|(candidate, _)| candidate == resolved_var) {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "distinct declared rank parameters {} and {} of `{declaration}` were unified \
                     by the function body: authored rank binders are rigid and must remain \
                     distinct (spec/04-type-system.md §3.1.3 [04-INF-6])",
                    render_declared_rank(rank_names, *previous),
                    render_declared_rank(rank_names, binder),
                ),
                vec![
                    "Use the same binder on both sides if they are meant to denote one rank, or \
                     fix the body so each declared rank stays independent."
                        .to_string(),
                ],
            ));
        } else {
            seen.push((*resolved_var, binder));
        }
    }
}

/// The dtype-family part of the authored contract also applies at reserved
/// wrapper boundaries that project checked types instead of unifying shapes.
pub(super) fn check_declared_dtype_bounds(
    declaration: &str,
    type_names: &UnordMap<TypeVar, String>,
    declared_bounds: &UnordMap<TypeVar, Option<TypeVarRestriction>>,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    for (binder, _) in type_names.to_sorted() {
        let binder = *binder;
        let Type::Var(resolved_var) = subst.apply(&Type::Var(binder)) else {
            // Concrete pins are owned by the separate type-rigidity check.
            continue;
        };
        let declared_bound = declared_bounds.get(&binder).copied().flatten();
        if let Some(required) = subst.tvar_restriction(resolved_var)
            && declared_bound != Some(required)
        {
            let authored = declared_bound
                .map(|bound| format!("the declared `{}` family", bound.family_name()))
                .unwrap_or_else(|| "an unbounded authored variable".to_string());
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "declared type parameter {} of `{declaration}` requires dtype family `{}` in its body, but its signature admits {authored}; an authored generic contract must satisfy its operation requirements at the definition (spec/04-type-system.md §3.1, [04-DTYPE-2])",
                    render_declared_binder(type_names, binder), required.family_name(),
                ),
                vec![format!(
                    "Declare this binder with `{}: {}` in the signature's binder list.",
                    type_names.get(&binder).expect("an authored binder has a source name"),
                    required.family_name(),
                )],
            ));
        }
    }
}

/// chelis#2216: one authored binder that a suspended call's operand denotes,
/// and the instantiations the call is decided at.
struct RigidOperand<'a> {
    var: TypeVar,
    name: &'a str,
    bound: Option<TypeVarRestriction>,
    instantiations: Vec<RigidInstantiation>,
}

/// One instantiation a rigid binder's decision replays the call at.
#[derive(Clone, Copy)]
enum RigidInstantiation {
    /// A member of the binder's dtype family.
    Member(Prim),
    /// No particular type. The binder is left unbound, so the replay sees an
    /// operand whose type is unknown and satisfies nothing that not every type
    /// satisfies.
    Arbitrary,
}

/// The instantiations a rigid binder's decision replays the call at.
///
/// Every dtype of the family its restriction names, which is the declared
/// bound unless the body narrowed it. A narrowing is reported by
/// [`check_declared_dtype_bounds`] against the declared bound, so replaying
/// only what the narrowed family admits reports each defect once.
///
/// A binder with no restriction admits every type ([04-DTYPE-2]), and no one
/// type stands for them all: `()` does not, because `eq` admits unit but not a
/// function ([05-OP-36]). It is replayed at [`RigidInstantiation::Arbitrary`]
/// instead.
fn rigid_instantiations(var: TypeVar, subst: &Subst) -> Vec<RigidInstantiation> {
    match subst.tvar_restriction(var) {
        Some(restriction) => family_members(restriction.precision_family())
            .map(RigidInstantiation::Member)
            .collect(),
        None => vec![RigidInstantiation::Arbitrary],
    }
}

/// What one replay at one combination of instantiations concluded.
enum ReplayOutcome {
    /// The call type-checks at this combination.
    Admitted,
    /// The call's own diagnostics at this combination, withdrawn from the sink
    /// so the caller reports them once, naming the combination.
    Rejected(Vec<CheckError>),
    /// The call suspended again, on an operand whose type is still unknown.
    Suspended,
}

/// chelis#2216: decide a suspended call whose operand is an authored binder,
/// at the declaration boundary.
///
/// [04-INF-6] quantifies an authored binder over every instantiation its
/// declaration admits, and nothing ever binds it. A call suspended on such an
/// operand (`operand_deferral.rs`) therefore waited for an event that cannot
/// occur, and the unresolved-operand disposition at the boundary dropped it
/// unchecked: `numel(k)` with `k: p` and `p: Float` checked at score 1, then
/// trapped in `eval` and emitted C that does not compile.
///
/// The call is decided instead by replaying it at every combination of its
/// binders' instantiations ([`rigid_instantiations`]) against isolated solver
/// state and a scratch product. The replay is the call's own route
/// ([`InferenceProduct::replay_post_app`]), so the verdict at `f32` is the one
/// a concrete `f32` operand gets, and an operation that accepts every member
/// of the bound (`take(xs, n)` with `n: p` and `p: Int`) stays accepted. The
/// first failing combination is reported with the route's own kind and text,
/// naming the instantiation.
///
/// An unbounded binder is left unbound in its replay. A call that suspends on
/// it again cannot be decided without knowing the operand's type, so it does
/// not type-check at an arbitrary type and is rejected, naming the binder as
/// "an arbitrary type". A call whose route never looks at that operand's type
/// stays accepted.
///
/// Returns true when the call is fully decided: it was rejected, or every
/// operand it waited on is an authored binder and every combination passed.
/// An operand that is a flexible inference variable keeps its existing
/// disposition (chelis#1489). So does a call whose binders all appear among
/// `shape_rule_operands`, the operands an unresolved shape-rule entry waits
/// on: that entry rejects the declaration on the same operand at the
/// boundary, and deciding here too would report it twice.
#[allow(clippy::too_many_arguments)]
pub(super) fn decide_at_rigid_binder_instantiations(
    call: PostAppCall<'_>,
    arg_tys: &[Type],
    result_ty: &Type,
    shape_rule_operands: &[TypeVar],
    declaration: Option<&str>,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let mut binders: Vec<RigidOperand<'_>> = Vec::new();
    let mut flexible_operand = false;
    for arg_ty in arg_tys {
        let Type::Var(var) = type_for_readonly_check(arg_ty, subst) else {
            continue;
        };
        match call.env.authored_type_binder(var, subst) {
            Some(_) if binders.iter().any(|binder| binder.var == var) => {}
            Some((name, bound)) => binders.push(RigidOperand {
                var,
                name,
                bound,
                instantiations: rigid_instantiations(var, subst),
            }),
            None => flexible_operand = true,
        }
    }
    if binders.is_empty()
        || binders
            .iter()
            .all(|binder| shape_rule_operands.contains(&binder.var))
    {
        return false;
    }

    // An odometer over the binders' instantiation lists, first binder slowest,
    // so the reported combination is the same on every run.
    let mut cursor = vec![0usize; binders.len()];
    loop {
        let assignment: Vec<(&RigidOperand<'_>, RigidInstantiation)> = binders
            .iter()
            .zip(&cursor)
            .map(|(binder, index)| (binder, binder.instantiations[*index]))
            .collect();
        match replay_at_instantiation(
            call,
            arg_tys,
            result_ty,
            &assignment,
            vg,
            subst,
            adt_reg,
            errors,
        ) {
            ReplayOutcome::Rejected(rejections) => {
                for rejection in rejections {
                    errors.push(at_rigid_instantiation(rejection, declaration, &assignment));
                }
                return true;
            }
            // A flexible operand is still unresolved too, and it may be the one
            // the call suspended on, so only an arbitrary binder decides here.
            ReplayOutcome::Suspended
                if !flexible_operand
                    && assignment.iter().any(|(_, instantiation)| {
                        matches!(instantiation, RigidInstantiation::Arbitrary)
                    }) =>
            {
                let rejection = CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "`{}` admits only some operand types, so it cannot be applied to an \
                         operand of an arbitrary type",
                        call.func_name
                    ),
                    vec![],
                );
                errors.push(at_rigid_instantiation(rejection, declaration, &assignment));
                return true;
            }
            ReplayOutcome::Admitted | ReplayOutcome::Suspended => {}
        }
        let Some(position) = (0..binders.len())
            .rev()
            .find(|position| cursor[*position] + 1 < binders[*position].instantiations.len())
        else {
            return !flexible_operand;
        };
        cursor[position] += 1;
        for later in &mut cursor[position + 1..] {
            *later = 0;
        }
    }
}

/// Replay `call` with each binder in `assignment` at its instantiation, in
/// solver state isolated from `subst`, and withdraw what the replay reported
/// from `errors`. A combination the binders' own restrictions reject is not
/// one the declaration admits and decides nothing.
///
/// The isolated state is a clone of `subst`, except for a dtype-admissibility
/// replay whose instantiated operands are ground. Those validators
/// (`replay_dtype_admissibility`) read an operand's type through the
/// substitution, which is the identity on a ground type, and reach the
/// substitution or the environment for anything else only on an operand that
/// is a variable or a tensor at a variable precision. A ground replay
/// therefore reads nothing from `subst`, and gets an empty substitution
/// rather than a copy of the whole module's (chelis#2216 round 1: the copy
/// made `check` about 40% slower on `nautilus`). A route replay, and any
/// replay with a variable left in an operand, reads solver state the operand
/// types do not show, and keeps the clone.
#[allow(clippy::too_many_arguments)]
fn replay_at_instantiation(
    call: PostAppCall<'_>,
    arg_tys: &[Type],
    result_ty: &Type,
    assignment: &[(&RigidOperand<'_>, RigidInstantiation)],
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> ReplayOutcome {
    let instantiate = |target: &mut Subst| {
        assignment
            .iter()
            .all(|(binder, instantiation)| match instantiation {
                RigidInstantiation::Member(prim) => {
                    unify(&Type::Var(binder.var), &Type::Prim(*prim), target).is_ok()
                }
                RigidInstantiation::Arbitrary => true,
            })
    };
    let mut instantiation = Subst::default();
    if !instantiate(&mut instantiation) {
        return ReplayOutcome::Admitted;
    }
    let settled: Vec<Type> = arg_tys
        .iter()
        .map(|ty| instantiation.apply(&subst.apply(ty)))
        .collect();
    let ground = call.replay == PostAppReplay::DtypeAdmissibility
        && settled.iter().all(|ty| {
            free_tvars(ty).is_empty() && free_dvars(ty).is_empty() && free_rvars(ty).is_empty()
        });
    let mut trial_subst = if ground {
        Subst::default()
    } else {
        let mut trial_subst = subst.clone();
        if !instantiate(&mut trial_subst) {
            return ReplayOutcome::Admitted;
        }
        trial_subst
    };
    let mut trial_vg = vg.clone();
    let mut trial_product = InferenceProduct::default();
    let checkpoint = errors.checkpoint();
    trial_product.replay_post_app(
        call,
        settled,
        result_ty,
        &mut trial_vg,
        &mut trial_subst,
        adt_reg,
        errors,
        true,
    );
    let rejections: Vec<CheckError> = errors.iter_since(checkpoint).cloned().collect();
    errors.retain_since(checkpoint, |_| false);
    if !rejections.is_empty() {
        ReplayOutcome::Rejected(rejections)
    } else if trial_product.has_post_app_checks() {
        ReplayOutcome::Suspended
    } else {
        ReplayOutcome::Admitted
    }
}

/// Name the instantiation a replayed rejection was raised at, and the repair.
fn at_rigid_instantiation(
    mut error: CheckError,
    declaration: Option<&str>,
    assignment: &[(&RigidOperand<'_>, RigidInstantiation)],
) -> CheckError {
    let owner =
        declaration.map_or_else(|| "its declaration".to_string(), |name| format!("`{name}`"));
    let binders = assignment
        .iter()
        .map(|(binder, _)| match binder.bound {
            Some(bound) => format!(
                "`{}`, declared `{}: {}`",
                binder.name,
                binder.name,
                bound.family_name()
            ),
            None => format!("`{}`, declared with no dtype-family bound", binder.name),
        })
        .collect::<Vec<_>>()
        .join(" and ");
    let at = assignment
        .iter()
        .map(|(binder, instantiation)| match instantiation {
            RigidInstantiation::Member(prim) => format!("`{} := {}`", binder.name, prim.name()),
            RigidInstantiation::Arbitrary => format!("`{}` := an arbitrary type", binder.name),
        })
        .collect::<Vec<_>>()
        .join(", ");
    error.message = format!(
        "{}: an operand's type is the authored type binder {binders} of {owner}, so the call \
         must type-check at every instantiation {owner} admits, and it does not at {at} \
         (spec/04-type-system.md §3.1.3 [04-INF-6], §5.9 [04-DTYPE-2])",
        error.message
    );
    let repairs = assignment.iter().map(|(binder, _)| match binder.bound {
        Some(bound) => format!(
            "`{name}: {family}` denotes a scalar of that family at every instantiation, never a \
             tensor, collection, or string. Declare the operand with the type this operation \
             requires, such as `tensor[n, {name}]`, or narrow `{name}`'s bound to the dtypes the \
             operation admits.",
            name = binder.name,
            family = bound.family_name(),
        ),
        None => format!(
            "`{name}` declares no dtype-family bound, so it denotes every type. Declare the \
             operand with the type this operation requires, or bound `{name}` with a dtype \
             family the operation admits.",
            name = binder.name,
        ),
    });
    error.suggestions.splice(0..0, repairs);
    error
}
