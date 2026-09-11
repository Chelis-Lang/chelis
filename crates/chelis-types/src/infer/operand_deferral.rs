//! chelis#1512: suspending a checked route's decision until its operand binds.
//!
//! One responsibility, and it owns no route's rule. A route family module
//! decides what its operands must be; this module only records that the
//! decision could not be made yet and which call has to be re-entered once it
//! can. The replay itself lives with the deferred shape ledger in
//! `checked.rs`, beside the six per-route rules chelis#1489 registered there.

use super::*;

/// chelis#1512: the call site of a checked route that is about to return
/// early because one of its operands is still an unresolved type variable.
///
/// Such an arm publishes a result the route never derived from the operand
/// and skips the check the resolved arm performs, and nothing revisits it
/// when the operand later settles. Registering the call on the deferred shape
/// ledger re-enters `finish_unified_app` at the binding, so the route reaches
/// its own arm against a settled operand and reports its own diagnostic.
/// Nothing is copied onto a deferred path: the deferred path IS the eager
/// path, which is why a relocated decision here cannot silently drop what the
/// route used to do (the hazard measured on chelis#1489's conversions).
///
/// A variable that never binds is rejected at the declaration boundary by
/// `finish_deferred_shape_checks`, the acceptance boundary the shape-computed
/// builtins have carried since chelis#1489. That is deliberate: a route whose
/// operand never acquires an outer constructor cannot be shown to be well
/// typed, and accepting it is how chelis#1512's witnesses reached the backend.
pub(super) struct UnresolvedOperandSite<'a> {
    list: &'a deep::List,
    kids: &'a [deep::Expr],
    fname: &'a str,
    env: &'a Env,
}

impl<'a> UnresolvedOperandSite<'a> {
    pub(super) fn new(
        list: &'a deep::List,
        kids: &'a [deep::Expr],
        fname: &'a str,
        env: &'a Env,
    ) -> Self {
        Self {
            list,
            kids,
            fname,
            env,
        }
    }

    /// Suspend this call's decision. The eager pass keeps publishing whatever
    /// the arm published before, so this repair adds a ledger entry and
    /// changes nothing else about the first pass.
    pub(super) fn register(
        &self,
        arg_tys: &[Type],
        result_ty: &Type,
        product: &mut InferenceProduct,
    ) {
        // A route that inspects several operands reaches such an arm once per
        // unresolved operand. One suspended entry per CALL is what the replay
        // needs; a second would re-enter the route and report twice.
        if product.has_post_app_check_for(self.list) {
            return;
        }
        product.defer_shape_check(
            DeferredShapeRule::PostApp {
                site: product.post_app_key(self.list),
                list: self.list.clone(),
                kids: self.kids.to_vec(),
                func_name: self.fname.to_string(),
                env: Box::new(self.env.clone()),
            },
            Vec::new(),
            arg_tys.to_vec(),
            result_ty.clone(),
        );
    }

    /// [`Self::register`] for an arm that returns: hands back `eager`, the
    /// value that arm published before this repair.
    pub(super) fn defer(
        &self,
        arg_tys: &[Type],
        result_ty: &Type,
        product: &mut InferenceProduct,
        eager: Type,
    ) -> Type {
        self.register(arg_tys, result_ty, product);
        eager
    }
}

/// chelis#1512: which `app_shape` route a suspended call belongs to.
///
/// These five reach inference through `app.rs` rather than
/// `finish_unified_app`, and each infers its own children, so replaying the
/// entry point would re-infer them. The entry point therefore hands its
/// already-inferred argument types to the rule, and the ledger replays the
/// rule alone.
#[derive(Clone)]
pub(super) enum ShapeRouteKind {
    Permute,
    Shrink,
    Stride,
    Pad,
    ReduceWindow { name: String },
    Reshape { input_var_name: Option<String> },
}

impl ShapeRouteKind {
    /// The builtin's name, for the declaration-boundary diagnostic.
    pub(super) fn builtin(&self) -> String {
        match self {
            Self::Permute => "permute".to_string(),
            Self::Shrink => "shrink".to_string(),
            Self::Stride => "stride".to_string(),
            Self::Pad => "pad".to_string(),
            Self::ReduceWindow { name } => name.clone(),
            Self::Reshape { .. } => "reshape".to_string(),
        }
    }
}

/// Run an `app_shape` route's rule, or suspend it when the operand is still
/// an unresolved type variable.
///
/// The suspended call publishes a FRESH variable rather than the operand's
/// own type. Publishing the operand's type was the second half of
/// chelis#1512 on these five routes: it accepted exactly the operand's shape
/// and rejected the shape the rule actually produces, so a correct declared
/// result was refused and an incorrect one accepted. A fresh variable commits
/// to nothing until the replay unifies the rule's own answer into it.
#[allow(clippy::too_many_arguments)]
pub(super) fn defer_or_check_shape_route(
    route: ShapeRouteKind,
    list: &deep::List,
    kids: &[deep::Expr],
    arg_tys: Vec<Type>,
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    // chelis#1512: ANY unresolved operand, not just the tensor. These routes
    // check their axis, stride, bounds and window arguments too, and an arm
    // that admitted a variable there let a non-int32 `permute` axis and a
    // non-int64 `stride` step through while the tensor operand was settled.
    if arg_tys
        .iter()
        .any(|ty| matches!(type_for_readonly_check(ty, subst), Type::Var(_)))
    {
        let result = Type::Var(vg.fresh_tvar());
        product.defer_shape_check(
            DeferredShapeRule::ShapeRoute {
                route,
                list: list.clone(),
                kids: kids.to_vec(),
            },
            Vec::new(),
            arg_tys,
            result.clone(),
        );
        return result;
    }
    check_shape_route_signature(&route, list, kids, &arg_tys, vg, subst, errors)
}

/// The one entry both the eager pass and the ledger replay call.
pub(super) fn check_shape_route_signature(
    route: &ShapeRouteKind,
    list: &deep::List,
    kids: &[deep::Expr],
    arg_tys: &[Type],
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    match route {
        ShapeRouteKind::Permute => check_permute_signature(list, kids, arg_tys, subst, errors),
        ShapeRouteKind::Shrink => check_shrink_signature(list, kids, arg_tys, subst, errors),
        ShapeRouteKind::Stride => check_stride_signature(list, kids, arg_tys, subst, errors),
        ShapeRouteKind::Pad => check_pad_signature(list, kids, arg_tys, vg, subst, errors),
        ShapeRouteKind::ReduceWindow { name } => {
            check_reduce_window_signature(list, kids, name, arg_tys, subst, errors)
        }
        ShapeRouteKind::Reshape { input_var_name } => check_reshape_signature(
            list,
            kids,
            input_var_name.as_deref(),
            arg_tys,
            subst,
            errors,
        ),
    }
}
