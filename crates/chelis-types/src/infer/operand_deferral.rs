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
        if product.has_route_check_for(self.list) {
            return;
        }
        // chelis#1512 follow-up: a dtype-admissibility suspension for this same
        // call is REPLACED rather than kept. The dtype validators run at the
        // head of `finish_unified_app`, so they register first, and this replay
        // re-enters that function and runs them again before it reaches the
        // route. The route replay therefore subsumes the narrower one, and
        // keeping both would run the validators twice.
        product.cancel_post_app_check_for(self.list);
        product.defer_shape_check(
            DeferredShapeRule::PostApp {
                replay: PostAppReplay::Route,
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

    /// chelis#1512: suspend one dtype-admissibility decision.
    ///
    /// Private to this module on purpose, and reached only through
    /// [`DtypeAdmissibilitySite`]. The census recognizes a suspension by the
    /// `site.register` idiom in the arm body, and `register_dtype_admissibility`
    /// begins with that same text, so a route arm that called this one by
    /// mistake would still be recorded `deferred` while its own validation was
    /// never replayed (round 1 P3-2). Handing the two kinds to two types makes
    /// that unrepresentable rather than detectable: a route arm holds an
    /// `UnresolvedOperandSite`, which has no dtype registration at all.
    ///
    /// The three validators in `app_numeric.rs` and `app_operand_dtype.rs` run
    /// at the head of `finish_unified_app`, ahead of the route dispatch that
    /// owns [`Self::register`]. They decide whether the operand's DTYPE is
    /// admitted, not what shape the result has, so the replay re-runs those
    /// three functions and nothing else. Re-entering `finish_unified_app`
    /// instead would re-run the route's shape rule too, and `sum`, `mean`,
    /// `matmul` and `layer_norm` already carry their own ledger entry for the
    /// same call, so the shape diagnostic would be reported twice.
    ///
    /// The entry shares [`InferenceProduct::post_app_key`] with the route
    /// registration above, so chelis#1774's key translation, its report-once
    /// cancellation and its silent declaration boundary all apply unchanged.
    fn register_dtype_admissibility(
        &self,
        arg_tys: &[Type],
        result_ty: &Type,
        subst: &Subst,
        product: &mut InferenceProduct,
        awaits_precision: Option<TypeVar>,
    ) {
        // chelis#731 cascade suppression. An error witness never binds, the
        // readiness predicate does not wait for it, and the replay would print
        // this route's own diagnostic on top of the upstream failure that
        // produced the witness. The eager pass already admits such an operand;
        // leaving the call unsuspended keeps that.
        if arg_tys
            .iter()
            .any(|ty| matches!(type_for_readonly_check(ty, subst), Type::Error(_)))
        {
            return;
        }
        // One entry per CALL, as above: a validator that reads several operands
        // reaches its unresolved arm once per operand.
        if product.has_post_app_check_for(self.list) {
            return;
        }
        let rule = DeferredShapeRule::PostApp {
            replay: PostAppReplay::DtypeAdmissibility,
            site: product.post_app_key(self.list),
            list: self.list.clone(),
            kids: self.kids.to_vec(),
            func_name: self.fname.to_string(),
            env: Box::new(self.env.clone()),
        };
        match awaits_precision {
            // chelis#1805: the operand's outer constructor is already a tensor,
            // so the readiness predicate answers ready for it. The entry waits
            // on the precision variable instead.
            Some(precision) => product.defer_shape_check_awaiting_precision(
                rule,
                arg_tys.to_vec(),
                result_ty.clone(),
                precision,
            ),
            None => {
                product.defer_shape_check(rule, Vec::new(), arg_tys.to_vec(), result_ty.clone())
            }
        }
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

/// chelis#1512: the capability to suspend ONE dtype-admissibility decision.
///
/// `finish_unified_app` mints it once, before the three validators run, and
/// hands each of them a borrow. It is the only route to
/// [`UnresolvedOperandSite::register_dtype_admissibility`], so no route arm can
/// register the narrower replay kind for a call whose own arm needs the full
/// one; see that method's own note.
pub(super) struct DtypeAdmissibilitySite<'a> {
    site: UnresolvedOperandSite<'a>,
}

impl<'a> DtypeAdmissibilitySite<'a> {
    pub(super) fn new(
        list: &'a deep::List,
        kids: &'a [deep::Expr],
        fname: &'a str,
        env: &'a Env,
    ) -> Self {
        Self {
            site: UnresolvedOperandSite::new(list, kids, fname, env),
        }
    }

    /// Suspend this call's dtype decision on an unresolved operand TYPE.
    pub(super) fn register(
        &self,
        arg_tys: &[Type],
        result_ty: &Type,
        subst: &Subst,
        product: &mut InferenceProduct,
    ) {
        self.site
            .register_dtype_admissibility(arg_tys, result_ty, subst, product, None);
    }

    /// chelis#1805: suspend it on an unresolved operand PRECISION instead.
    ///
    /// Separate from [`Self::register`] because the two wait on different
    /// things and the ledger has to know which: a call whose operand type is
    /// unknown resumes when the type binds, and one whose operand is a tensor
    /// at an unknown precision resumes when the precision does. Registering the
    /// second as the first would make the entry ready immediately, which is
    /// chelis#1805 itself.
    pub(super) fn register_awaiting_precision(
        &self,
        arg_tys: &[Type],
        result_ty: &Type,
        subst: &Subst,
        product: &mut InferenceProduct,
        precision: TypeVar,
    ) {
        self.site
            .register_dtype_admissibility(arg_tys, result_ty, subst, product, Some(precision));
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

/// chelis#1512: re-run the dtype-admissibility validators against an operand
/// that has settled since the call was inferred.
///
/// The deferred path IS the eager path. These are the same three functions
/// [`finish_unified_app`] calls at its head, in the same order and with the
/// same arguments, so a relocated decision cannot silently drop what the eager
/// one did.
///
/// It validates rather than produces, which is why nothing is reconciled
/// against the type the suspended call published. Two of the three return a
/// type only when they REJECT, and the arms of the third that suspend all
/// publish the left operand itself, which by replay time is the very type
/// `integer_binop_result_type` derives from it.
///
/// `None` for the suspension: a replay decides against a settled operand, so
/// no arm here can suspend the call a second time.
#[allow(clippy::too_many_arguments)]
pub(super) fn replay_dtype_admissibility(
    list: &deep::List,
    kids: &[deep::Expr],
    func_name: &str,
    env: &Env,
    arg_tys: &[Type],
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) {
    let owned_name = Some(func_name.to_string());
    let mut route_observed = false;
    let result_ty = Type::Unit;
    if validate_numeric_and_reduction_arguments(
        list,
        kids,
        &owned_name,
        arg_tys,
        env,
        subst,
        errors,
        &mut route_observed,
        None,
        &result_ty,
        product,
    )
    .is_some()
    {
        return;
    }
    if reject_inadmissible_operand_dtypes(
        list,
        kids,
        Some(func_name),
        arg_tys,
        env,
        subst,
        errors,
        &mut route_observed,
        None,
        &result_ty,
        product,
    )
    .is_some()
    {
        return;
    }
    let _ = integer_binop_result_type(
        list,
        Some(func_name),
        arg_tys,
        vg,
        subst,
        errors,
        None,
        &result_ty,
        product,
    );
}
