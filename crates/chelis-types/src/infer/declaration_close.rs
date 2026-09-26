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
    /// The declaration omits a type and belongs to a recursive group, so every
    /// in-group call to it is typed at its own instantiation ([04-INF-5]).
    monomorphic_group_member: bool,
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
            monomorphic_group_member: false,
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

    /// Mark a member of a recursive group bound at its provisional monomorphic
    /// type, so a rigidity violation names the repair that allows polymorphic
    /// recursion.
    pub(super) fn in_monomorphic_group(mut self, member: bool) -> Self {
        self.monomorphic_group_member = member;
        self
    }

    fn decide(self, subst: &Subst, errors: &mut DiagnosticSink<'_>) {
        let name = self.declaration.as_str();
        let checkpoint = errors.checkpoint();
        if let Some(rigidity) = &self.rigidity {
            // Parameter and body-only dimension roles are rigid under
            // [04-INF-6]; result-only roles retain §4.4.1 output inference.
            check_authored_dvars_rigid(name, &rigidity.decl_ty, &rigidity.dim_names, subst, errors);
            check_declared_tvars_rigid(name, &self.type_names, subst, errors);
            check_declared_rvars_rigid(name, &rigidity.rank_names, subst, errors);
        }
        check_declared_dtype_bounds(name, &self.type_names, &self.dtype_bounds, subst, errors);
        if self.monomorphic_group_member {
            // chelis#2584: a call in its own recursive group that instantiates
            // the binders at other types is what identified them here.
            let raised: Vec<CheckError> = errors.iter_since(checkpoint).cloned().collect();
            errors.retain_since(checkpoint, |_| false);
            for mut error in raised {
                error.suggestions.push(format!(
                    "`{name}` omits a type in its signature, so every call to it inside its \
                     recursive group is typed at `{name}`'s own instantiation \
                     (spec/04-type-system.md §3.1.3 [04-INF-5], §3.1.1 [04-INF-2]). Write the \
                     omitted types to allow a call at another instantiation (polymorphic \
                     recursion)."
                ));
                errors.push(error);
            }
        }
    }
}

/// Where a declaration's close falls.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CloseScope {
    /// An ordinary declaration: its obligations are decided at its own close.
    Declaration,
    /// A member of a recursive group or other cyclic component. Its
    /// obligations stay on the ledgers, each entry naming its declaration, and
    /// [`close_component`] decides them when the group completes (chelis#2584).
    /// A sibling inferred later can still fill a type an obligation waits on:
    /// [04-INF-5] types an in-group reference at the member's provisional
    /// type, which only that member's body determines.
    ComponentMember,
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
///
/// A [`CloseScope::ComponentMember`] runs only the ready replay and step 4;
/// its component's [`close_component`] runs the rest for every member.
#[allow(clippy::too_many_arguments)]
pub(super) fn close_declaration(
    product: &mut InferenceProduct,
    scope: CloseScope,
    env: &Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    if scope == CloseScope::ComponentMember {
        product.replay_ready_shape_checks(vg, subst, adt_reg, errors);
        product.finish_root(subst, errors);
        return;
    }
    decide_open_obligations(product, env, vg, subst, adt_reg, errors);
    product.finish_root(subst, errors);
    report_substitution_ledgers(env, subst, adt_reg, errors);
}

/// Decide the obligations every member of a completed cyclic component left
/// open, once the whole group's bodies have been inferred (chelis#2584).
///
/// The members' authored binders are one scope here: a binder variable
/// belongs to exactly one declaration, so the union of their names lets the
/// boundary decision recognize each member's binders, and each diagnostic
/// names the declaration that owns its entry.
pub(super) fn close_component(
    product: &mut InferenceProduct,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    let mut names = UnordMap::new();
    let mut bounds = UnordMap::new();
    for contract in product.pending_authored_binder_contracts() {
        for (variable, name) in contract.type_names.to_sorted() {
            names.insert(*variable, name.clone());
        }
        for (variable, bound) in contract.dtype_bounds.to_sorted() {
            bounds.insert(*variable, *bound);
        }
    }
    env.set_active_declared_type_names(names);
    env.set_active_declared_type_bounds(bounds);
    decide_open_obligations(product, env, vg, subst, adt_reg, errors);
    report_substitution_ledgers(env, subst, adt_reg, errors);
}

/// Steps 1 to 3 of [`close_declaration`], over whatever the ledgers hold.
fn decide_open_obligations(
    product: &mut InferenceProduct,
    env: &Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    product.finish_deferred_shape_checks(env, vg, subst, adt_reg, errors);
    product.finish_deferred_literal_patterns(env, subst, adt_reg, errors);
    for contract in product.take_authored_binder_contracts() {
        contract.decide(subst, errors);
    }
}

/// Step 5 of [`close_declaration`].
fn report_substitution_ledgers(
    env: &Env,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
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
