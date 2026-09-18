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
//! An inference hole (`(t-var {} _)`) is NOT a binder and is not checked here.
//! [04-INF-5] gives the hole the type its body determines; it never reaches
//! the declaration-owned identity maps because the binder list never includes
//! `_`.

use chelis_unord::UnordMap;

use crate::errors::{CheckError, CheckErrorKind};
use crate::session::DiagnosticSink;
use crate::types::{Dim, RankVar, Type, TypeVar, TypeVarRestriction};
use crate::unify::Subst;

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
