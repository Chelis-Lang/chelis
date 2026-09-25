//! chelis#731: the declaration boundary, as one ordered step.
//!
//! A declaration's inference leaves obligations open: calls suspended on an
//! operand whose type is still a variable, literal patterns on a flexible
//! scrutinee, operand gates on the substitution (chelis#1489), deferred
//! borrows and opaque uses, and the declaration's own authored-binder contract
//! ([04-INF-6], [04-DTYPE-2]). [`close_declaration`] is the one place that
//! consumes them, and the order is part of the contract: a step that can still
//! bind or narrow a variable runs before every step that decides on one.
//!
//! The authored-binder contract is the case that order exists for. Deciding
//! the replays of suspended calls can narrow an authored binder, as the
//! `index` replay behind a returned closure did in chelis#2537, where a
//! `cast_trunc` gate discharged by that replay narrowed `p: Numeric` to `Float`.
//! The contract was checked inside `infer_top_level`, before those replays,
//! so the narrowing was never reported. It is now recorded there and decided
//! here, after the last step that can narrow.

use super::*;

/// The authored-binder contract of one declaration, recorded by
/// `infer_top_level` once its body is unified with the declared signature and
/// decided by [`close_declaration`].
pub(super) struct AuthoredBinderContract {
    declaration: String,
    /// The rigidity half ([04-INF-6]), present when the body was unified with
    /// a declared signature.
    rigidity: Option<BinderRigidity>,
    type_names: UnordMap<TypeVar, String>,
    /// Each authored type binder's declared dtype-family bound, captured before
    /// the body could narrow it.
    dtype_bounds: UnordMap<TypeVar, Option<TypeVarRestriction>>,
}

struct BinderRigidity {
    decl_ty: Type,
    dim_names: UnordMap<DimVar, String>,
    rank_names: UnordMap<RankVar, String>,
}

impl AuthoredBinderContract {
    pub(super) fn new(
        declaration: String,
        type_names: UnordMap<TypeVar, String>,
        dtype_bounds: UnordMap<TypeVar, Option<TypeVarRestriction>>,
    ) -> Self {
        Self {
            declaration,
            rigidity: None,
            type_names,
            dtype_bounds,
        }
    }

    /// Add the rigidity half for a body unified with `decl_ty`.
    pub(super) fn with_rigidity(
        mut self,
        decl_ty: Type,
        dim_names: UnordMap<DimVar, String>,
        rank_names: UnordMap<RankVar, String>,
    ) -> Self {
        self.rigidity = Some(BinderRigidity {
            decl_ty,
            dim_names,
            rank_names,
        });
        self
    }

    fn decide(self, subst: &Subst, errors: &mut DiagnosticSink<'_>) {
        let name = self.declaration.as_str();
        if let Some(rigidity) = &self.rigidity {
            // Parameter and body-only dimension roles are rigid under
            // [04-INF-6]; result-only roles retain §4.4.1 output inference.
            check_authored_dvars_rigid(name, &rigidity.decl_ty, &rigidity.dim_names, subst, errors);
            check_declared_tvars_rigid(name, &self.type_names, subst, errors);
            check_declared_rvars_rigid(name, &rigidity.rank_names, subst, errors);
        }
        check_declared_dtype_bounds(name, &self.type_names, &self.dtype_bounds, subst, errors);
    }
}

/// Close one top-level declaration: decide or report every obligation its
/// inference left open, in an order where nothing decided can still change.
///
/// 1. Suspended calls: replayed once their operands bind, then decided at the
///    declaration boundary (`finish_deferred_shape_checks`). These replays
///    unify, so they can bind and narrow variables.
/// 2. Literal patterns on a flexible scrutinee ([04-PAT-1]).
/// 3. The authored-binder contract, after the last step that can narrow.
/// 4. The type-stamp epoch.
/// 5. The substitution's own ledgers, which only report: deferred borrows,
///    operand-gate failures and never-bound operands (chelis#1489), and
///    deferred opaque uses.
#[allow(clippy::too_many_arguments)]
pub(super) fn close_declaration(
    product: &mut InferenceProduct,
    declaration: Option<&str>,
    env: &Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    product.finish_deferred_shape_checks(declaration, env, vg, subst, adt_reg, errors);
    product.finish_deferred_literal_patterns(declaration, env, subst, adt_reg, errors);
    if let Some(contract) = product.take_authored_binder_contract() {
        contract.decide(subst, errors);
    }
    product.finish_root(subst, errors);
    // Issue #256 round 2: re-check each deferred borrow against the
    // now-complete substitution. Draining per declaration keeps error
    // attribution local and prevents one declaration's deferrals from leaking
    // into the next.
    validate_deferred_borrow_vars(subst, adt_reg, env.active_declared_type_names(), errors);
    // chelis#1489: see `validate_deferred_tensor_operands`.
    validate_deferred_tensor_operands(subst, env.active_declared_type_names(), errors);
    // D-CHECK: drain the per-declaration deferred-access ledger.
    validate_deferred_opaque_uses(subst, adt_reg, errors);
}
