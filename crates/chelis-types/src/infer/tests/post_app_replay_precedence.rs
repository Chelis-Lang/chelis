//! chelis#1512: the route and dtype suspensions share one ledger key, and the
//! route registration is the one that survives.
//!
//! Both kinds are `PostApp` entries so that one key, one report-once
//! cancellation and one declaration boundary cover them. That sharing owes a
//! precedence rule, because the dtype validators run at the head of
//! `finish_unified_app` and register FIRST: if a dtype entry made the later
//! route registration stand down, the route's own validation would be replayed
//! by something that never re-enters the route, which is chelis#1512's own
//! defect re-opened.
//!
//! No callee reaches both today. Every name a dtype validator governs returns
//! before the route dispatch or is absent from it, so nothing in the language
//! exercises this and no diagnostic can express it. The invariant is asserted
//! where it is decided instead.

use super::*;
use crate::infer::checked::PostAppReplay;
use crate::infer::operand_deferral::UnresolvedOperandSite;

fn probe() -> (deep::List, Vec<deep::Expr>, Env, InferenceProduct, Subst, Vec<Type>) {
    (
        deep::List { elements: vec![] },
        Vec::new(),
        Env::new(),
        InferenceProduct::default(),
        Subst::new(),
        vec![Type::Var(TypeVar(0))],
    )
}

/// REGRESSION TEST. Watched RED against a `register` that returns early on any
/// existing entry: the ledger then keeps the dtype replay and the route's own
/// arm is never re-entered.
#[test]
fn a_route_registration_replaces_a_dtype_suspension_for_the_same_call() {
    let (list, kids, env, mut product, subst, arg_tys) = probe();
    let site = UnresolvedOperandSite::new(&list, &kids, "probe", &env);

    site.register_dtype_admissibility(&arg_tys, &Type::Unit, &subst, &mut product);
    assert_eq!(
        product.post_app_replays_for(&list),
        vec![PostAppReplay::DtypeAdmissibility],
        "the dtype validator registers first, as it runs first"
    );

    site.register(&arg_tys, &Type::Unit, &mut product);
    assert_eq!(
        product.post_app_replays_for(&list),
        vec![PostAppReplay::Route],
        "the route replay re-enters `finish_unified_app`, which runs the dtype validators on \
         its way to the route, so it must REPLACE the narrower entry rather than be skipped"
    );
}

/// NEGATIVE TWIN. Precedence in the other direction is not symmetric: a dtype
/// registration after a route one is a no-op, because the route replay already
/// covers it. One entry per call either way, so neither validator runs twice.
#[test]
fn a_dtype_suspension_after_a_route_registration_is_a_no_op() {
    let (list, kids, env, mut product, subst, arg_tys) = probe();
    let site = UnresolvedOperandSite::new(&list, &kids, "probe", &env);

    site.register(&arg_tys, &Type::Unit, &mut product);
    site.register_dtype_admissibility(&arg_tys, &Type::Unit, &subst, &mut product);
    assert_eq!(
        product.post_app_replays_for(&list),
        vec![PostAppReplay::Route],
        "a second entry for one call would replay the route twice"
    );
}
