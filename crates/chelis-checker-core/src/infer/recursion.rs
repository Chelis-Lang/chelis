//! Uniform recursive instantiation enforcement (spec/04-type-system.md
//! §3.1.1 [04-INF-2]/[04-INF-3]).
//!
//! While a recursive binding group's bodies are inferred, every in-group
//! reference records the fresh type variables its instantiation minted.
//! Those variables are pinned against `let`-generalization (so an alias of
//! a group member stays monomorphic in them, exactly as a monomorphic
//! recursive occurrence would in SML), and after the whole group's bodies
//! have been inferred each recorded instantiation is resolved under the
//! final substitution. An entry that resolved to anything other than a
//! bare type variable means the recursive call was typed at a different
//! instantiation than the caller's own — polymorphic recursion — and is
//! rejected with the atom-citing diagnostic [04-INF-3] requires.

use crate::errors::ErrorWitness;
use crate::prelude::*;
use chelis_unord::{UnordMap, UnordSet};
use core::cell::RefCell;

use crate::env::{DeclarationBinderIdentities, Env};
use crate::errors::{CheckError, CheckErrorKind};
use crate::session::DiagnosticSink;
use chelis_numeric_core::prim::Prim;
use chelis_type_core::InstantiatedScheme;
use chelis_type_core::Subst;
use chelis_type_core::{NominalArg, Scheme, Type, TypeVar, TypeVarRestriction, VarGen};

#[derive(Clone, Copy)]
pub(super) enum RecursiveExpected<'a> {
    Provisional(&'a Type<ErrorWitness>),
    Published(&'a Scheme<ErrorWitness>),
}

/// Compiler-owned expected type for one recursive member inference. A
/// published expectation is instantiated through the ordinary Env/Subst path
/// so its checked contracts receive fresh identities; a provisional first-pass
/// type remains borrowed and carries no scheme payload.
pub(super) struct PreparedRecursiveExpected<'a> {
    provisional: Option<&'a Type<ErrorWitness>>,
    published: Option<InstantiatedScheme<ErrorWitness>>,
}

pub(super) struct DeclaredMemberRequest<'a> {
    name: &'a str,
    binder_names: Option<&'a UnordSet<String>>,
    declared_scheme: Option<&'a Scheme<ErrorWitness>>,
    use_published_as_declared: bool,
}

impl<'a> DeclaredMemberRequest<'a> {
    pub(super) fn new(
        name: &'a str,
        binder_names: Option<&'a UnordSet<String>>,
        declared_scheme: Option<&'a Scheme<ErrorWitness>>,
        use_published_as_declared: bool,
    ) -> Self {
        Self {
            name,
            binder_names,
            declared_scheme,
            use_published_as_declared,
        }
    }
}

impl<'a> PreparedRecursiveExpected<'a> {
    pub(super) fn prepare(
        expected: Option<RecursiveExpected<'a>>,
        var_gen: &mut VarGen,
        subst: &mut Subst<ErrorWitness>,
    ) -> Self {
        match expected {
            Some(RecursiveExpected::Provisional(ty)) => Self {
                provisional: Some(ty),
                published: None,
            },
            Some(RecursiveExpected::Published(scheme)) => Self {
                provisional: None,
                published: Some(chelis_type_core::instantiate_scheme(scheme, var_gen, subst)),
            },
            None => Self {
                provisional: None,
                published: None,
            },
        }
    }

    pub(super) fn expected_type(&self) -> Option<&Type<ErrorWitness>> {
        self.published
            .as_ref()
            .map(|instantiated| &instantiated.ty)
            .or(self.provisional)
    }

    pub(super) fn is_published(&self) -> bool {
        self.published.is_some()
    }

    pub(super) fn begin_caller<'operation>(
        &self,
        name: &str,
        binder_names: Option<&UnordSet<String>>,
        operation: &'operation crate::resources::Operation<'_>,
    ) -> CallerGuard<'operation> {
        let mapping = self
            .published
            .as_ref()
            .map_or(&[][..], |instantiated| instantiated.tvars.as_slice());
        begin_caller(name, binder_names, mapping, operation)
    }

    /// Prepare the declaration-side type and recursive caller identity without
    /// making an unsigned published expectation look authored. A real
    /// declaration scheme (including #2041's private rejected-signature
    /// recovery) wins; otherwise a published expectation is declaration-like
    /// only when the caller confirms the member already owns a defsig.
    pub(super) fn prepare_declared_member<'operation>(
        &self,
        request: DeclaredMemberRequest<'_>,
        env: &Env,
        var_gen: &mut VarGen,
        subst: &mut Subst<ErrorWitness>,
        operation: &'operation crate::resources::Operation<'_>,
    ) -> DeclaredMemberSetup<'operation> {
        let instantiation = if let Some(scheme) = request.declared_scheme {
            Some(chelis_type_core::instantiate_scheme(scheme, var_gen, subst))
        } else if request.use_published_as_declared {
            self.published.clone()
        } else {
            None
        };
        let binder_identities =
            request
                .binder_names
                .map_or_else(DeclarationBinderIdentities::default, |binders| {
                    env.declared_binder_identities_for_body(
                        request.name,
                        binders,
                        instantiation.as_ref(),
                        var_gen,
                    )
                });
        let Some(instantiation) = instantiation else {
            let caller_guard = if group_member(request.name, operation) {
                self.begin_caller(request.name, None, operation)
            } else {
                CallerGuard::inactive()
            };
            return DeclaredMemberSetup {
                ty: None,
                binder_identities,
                caller_guard,
            };
        };
        let caller_guard = if group_member(request.name, operation) {
            begin_caller(
                request.name,
                request.binder_names,
                &instantiation.tvars,
                operation,
            )
        } else {
            CallerGuard::inactive()
        };
        DeclaredMemberSetup {
            ty: Some(instantiation.ty),
            binder_identities,
            caller_guard,
        }
    }
}

pub(super) struct DeclaredMemberSetup<'operation> {
    pub(super) ty: Option<Type<ErrorWitness>>,
    pub(super) binder_identities: DeclarationBinderIdentities,
    pub(super) caller_guard: CallerGuard<'operation>,
}

pub(crate) struct GroupCtx {
    /// Member name -> member facts at group start, used to make sure a
    /// recorded reference resolved the top-level member and not a local
    /// shadow of the same name, and to select the strict or relaxed rule.
    members: UnordMap<String, MemberSnapshot>,
    caller: Option<Caller>,
    occurrences: Vec<Occurrence>,
    /// Fresh tvars minted for in-group instantiations. Pinned against
    /// generalization while the group is being inferred.
    pinned: UnordSet<TypeVar>,
}

struct MemberSnapshot {
    tvars: Vec<TypeVar>,
    tvar_restrictions: Vec<(TypeVar, TypeVarRestriction)>,
    /// chelis#1654: part of the snapshot because it is part of the scheme's
    /// identity. A member that acquired a collection obligation while its
    /// group was inferred has NOT been left unchanged, and comparing only the
    /// quantifiers and the body would report that it had.
    constraints: Vec<chelis_type_core::CollectionConstraint<ErrorWitness>>,
    body: Type<ErrorWitness>,
    /// Whether the member's `def` authored explicit type binders (`[a]`).
    /// Authored parameters take the strict caller's-own-instantiation rule;
    /// inference-introduced variables additionally admit fully concrete
    /// arguments, which cannot grow the instantiation set (spec/04 §3.1.1).
    authored_generic: bool,
    /// The member's header omits a type, so its in-group references share its
    /// authored binders and instantiate only its holes, and the component's
    /// completion decides each reference against the body
    /// (`super::group_link`, chelis#2590). Its references are pinned here but
    /// not validated by [`finish_group`].
    holed: bool,
}

struct Caller {
    name: String,
    /// Sorted declared type-parameter names, when the caller has a declared
    /// signature (used for diagnostic rendering only).
    binder_names: Vec<String>,
    /// The fresh tvars instantiated for the caller's own body inference,
    /// in scheme quantifier order.
    own_tvars: Vec<TypeVar>,
}

struct Occurrence {
    caller: String,
    caller_binder_names: Vec<String>,
    caller_own_tvars: Vec<TypeVar>,
    callee: String,
    callee_authored_generic: bool,
    /// Fresh instantiation types per callee scheme tvar, in quantifier order.
    instantiation: Vec<Type<ErrorWitness>>,
    span_id: Option<String>,
    span_offset: Option<usize>,
}

/// Clear recursive state before a new schedule.
pub(super) fn reset(operation: &crate::resources::Operation<'_>) {
    (|ctx: &RefCell<Option<GroupCtx>>| *ctx.borrow_mut() = None)(&operation.recursion);
}

/// Abandon an in-flight recursive group without validating occurrences.
/// Cancellation and other structured-abort paths use this to discard both
/// the group record and every temporary generalization pin.
pub(super) fn abort_group(operation: &crate::resources::Operation<'_>) {
    (|ctx: &RefCell<Option<GroupCtx>>| {
        ctx.borrow_mut().take();
    })(&operation.recursion);
}

/// Install the context for one recursive binding group. Member schemes are
/// snapshotted after any monomorphic prebinding so occurrence recording can
/// verify it resolved the top-level member.
pub(super) fn begin_group<'a>(
    member_names: impl Iterator<Item = (&'a str, bool)>,
    env: &Env,
    operation: &crate::resources::Operation<'_>,
) {
    let mut members = UnordMap::new();
    for (name, authored_generic) in member_names {
        let Some(scheme) = env.lookup(name) else {
            continue;
        };
        members.insert(
            name.to_string(),
            MemberSnapshot {
                tvars: scheme.tvars.clone(),
                tvar_restrictions: scheme.tvar_restrictions.clone(),
                constraints: scheme.constraints.clone(),
                body: scheme.body.clone(),
                authored_generic,
                holed: env.is_holed_group_member(name),
            },
        );
    }
    (|ctx: &RefCell<Option<GroupCtx>>| {
        *ctx.borrow_mut() = Some(GroupCtx {
            members,
            caller: None,
            occurrences: Vec::new(),
            pinned: UnordSet::new(),
        });
    })(&operation.recursion);
}

/// Whether `name` is a member of the active recursive group.
pub(super) fn group_member(name: &str, operation: &crate::resources::Operation<'_>) -> bool {
    (|ctx: &RefCell<Option<GroupCtx>>| {
        ctx.borrow()
            .as_ref()
            .is_some_and(|c| c.members.contains_key(name))
    })(&operation.recursion)
}

/// RAII guard clearing the current caller when the member's inference ends.
pub(super) struct CallerGuard<'operation> {
    context: Option<&'operation RefCell<Option<GroupCtx>>>,
}

impl CallerGuard<'_> {
    pub(super) fn inactive() -> Self {
        CallerGuard { context: None }
    }
}

impl Drop for CallerGuard<'_> {
    fn drop(&mut self) {
        if let Some(context) = self.context {
            if let Some(group) = context.borrow_mut().as_mut() {
                group.caller = None;
            }
        }
    }
}

/// Mark `name` as the group member whose body is being inferred.
/// `own_instantiation` is the declared-signature instantiation minted for
/// the caller's own body (empty for a monomorphically prebound member).
pub(super) fn begin_caller<'operation>(
    name: &str,
    binder_names: Option<&UnordSet<String>>,
    own_instantiation: &[(TypeVar, Type<ErrorWitness>)],
    operation: &'operation crate::resources::Operation<'_>,
) -> CallerGuard<'operation> {
    let mut binders: Vec<String> = binder_names
        .map(|set| set.to_sorted().into_iter().cloned().collect())
        .unwrap_or_default();
    binders.sort();
    let own_tvars = own_instantiation
        .iter()
        .filter_map(|(_, ty)| match ty {
            Type::Var(v) => Some(*v),
            _ => None,
        })
        .collect();
    (|ctx: &RefCell<Option<GroupCtx>>| {
        if let Some(c) = ctx.borrow_mut().as_mut() {
            c.caller = Some(Caller {
                name: name.to_string(),
                binder_names: binders,
                own_tvars,
            });
        }
    })(&operation.recursion);
    CallerGuard {
        context: Some(&operation.recursion),
    }
}

/// Whether an occurrence of `name`, resolved to `scheme`, should be
/// recorded: a group is active, a member body is being inferred, `name` is
/// in-group, and the resolved scheme is the member's own (not a local
/// shadow re-using the name).
pub(super) fn should_record_occurrence(
    name: &str,
    scheme: &Scheme<ErrorWitness>,
    operation: &crate::resources::Operation<'_>,
) -> bool {
    (|ctx: &RefCell<Option<GroupCtx>>| {
        let borrow = ctx.borrow();
        let Some(c) = borrow.as_ref() else {
            return false;
        };
        if c.caller.is_none() {
            return false;
        }
        let Some(snapshot) = c.members.get(name) else {
            return false;
        };
        snapshot.tvars == scheme.tvars
            && snapshot.tvar_restrictions == scheme.tvar_restrictions
            && snapshot.constraints == scheme.constraints
            && snapshot.body == scheme.body
    })(&operation.recursion)
}

/// Record one in-group reference's freshly minted instantiation and pin its
/// tvars against generalization for the rest of the group.
pub(super) fn record_occurrence(
    callee: &str,
    mapping: &[(TypeVar, Type<ErrorWitness>)],
    span_id: Option<String>,
    span_offset: Option<usize>,
    operation: &crate::resources::Operation<'_>,
) {
    (|ctx: &RefCell<Option<GroupCtx>>| {
        let mut borrow = ctx.borrow_mut();
        let Some(c) = borrow.as_mut() else {
            return;
        };
        let Some(caller) = c.caller.as_ref() else {
            return;
        };
        for (_, ty) in mapping {
            if let Type::Var(v) = ty {
                c.pinned.insert(*v);
            }
        }
        let Some(member) = c.members.get(callee) else {
            return;
        };
        if member.holed {
            return;
        }
        let callee_authored_generic = member.authored_generic;
        c.occurrences.push(Occurrence {
            caller: caller.name.clone(),
            caller_binder_names: caller.binder_names.clone(),
            caller_own_tvars: caller.own_tvars.clone(),
            callee: callee.to_string(),
            callee_authored_generic,
            instantiation: mapping.iter().map(|(_, ty)| ty.clone()).collect(),
            span_id,
            span_offset,
        });
    })(&operation.recursion);
}

/// Pin `vars`, a sibling reference's copies of a member's variables
/// (`group_link::sibling_instance`), against generalization for the rest of
/// the group, as an in-group instantiation's variables are.
pub(super) fn pin(
    vars: impl IntoIterator<Item = TypeVar>,
    operation: &crate::resources::Operation<'_>,
) {
    (|ctx: &RefCell<Option<GroupCtx>>| {
        if let Some(c) = ctx.borrow_mut().as_mut() {
            c.pinned.extend(vars);
        }
    })(&operation.recursion);
}

/// Query pinned variables for the reference generalization oracle and group cleanup tests.
#[cfg(any(test, feature = "generalize-sweep-oracle"))]
pub(crate) fn tvar_pinned(v: TypeVar, operation: &crate::resources::Operation<'_>) -> bool {
    (|ctx: &RefCell<Option<GroupCtx>>| ctx.borrow().as_ref().is_some_and(|c| c.pinned.contains(&v)))(
        &operation.recursion,
    )
}

pub(crate) fn with_pinned_tvars<R>(
    consume: impl FnOnce(&UnordSet<TypeVar>) -> R,
    operation: &crate::resources::Operation<'_>,
) -> R {
    (|ctx: &RefCell<Option<GroupCtx>>| {
        let context = ctx.borrow();
        match context.as_ref() {
            Some(group) => consume(&group.pinned),
            None => consume(&UnordSet::new()),
        }
    })(&operation.recursion)
}

#[cfg(test)]
pub(super) fn group_state_counts(operation: &crate::resources::Operation<'_>) -> (usize, usize) {
    (|ctx: &RefCell<Option<GroupCtx>>| {
        ctx.borrow()
            .as_ref()
            .map_or((0, 0), |group| (group.members.len(), group.pinned.len()))
    })(&operation.recursion)
}

/// Validate every recorded in-group instantiation under the group's final
/// substitution and clear the context. [04-INF-2] admits an entry that is
/// still a bare type variable: it is either the caller's own type parameter
/// or unconstrained (and so satisfiable by it). For a callee without
/// authored type binders, a fully concrete (variable-free) entry is also
/// admitted — it cannot grow the instantiation set. An entry embedding a
/// type variable inside a constructed type is polymorphic recursion
/// ([04-INF-3]); for an authored generic, any non-variable entry is.
pub(super) fn finish_group(
    subst: &Subst<ErrorWitness>,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) {
    let Some(ctx) =
        (|ctx: &RefCell<Option<GroupCtx>>| ctx.borrow_mut().take())(&operation.recursion)
    else {
        return;
    };
    for occ in ctx.occurrences {
        let resolved: Vec<Type<ErrorWitness>> =
            occ.instantiation.iter().map(|ty| subst.apply(ty)).collect();
        let admissible = |ty: &Type<ErrorWitness>| -> bool {
            match ty {
                Type::Var(_) | Type::Error(_) => true,
                _ => !occ.callee_authored_generic && chelis_type_core::free_tvars(ty).is_empty(),
            }
        };
        if resolved.iter().all(admissible) {
            continue;
        }
        let names = binder_display_names(&occ, subst);
        // With exactly one declared binder, any bare variable left inside an
        // offending entry is either the caller's own parameter or an
        // unconstrained variable that parameter could satisfy, so rendering
        // it under the binder's name (`Box[a]` rather than `Box[?379]`)
        // states the violation in the caller's vocabulary.
        let fallback =
            (occ.caller_binder_names.len() == 1).then(|| occ.caller_binder_names[0].clone());
        let got = resolved
            .iter()
            .map(|ty| render_type(ty, &names, fallback.as_deref()))
            .collect::<Vec<_>>()
            .join(", ");
        let expected = if occ.caller_binder_names.is_empty() {
            occ.caller_own_tvars
                .iter()
                .map(|v| names.get(v).cloned().unwrap_or_else(|| format!("?{}", v.0)))
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            occ.caller_binder_names.join(", ")
        };
        // A caller with no type parameters cannot supply an instantiation
        // at all, so "reuse the caller's own parameters" would be
        // unactionable advice; name the actual constraint instead.
        let caller_is_monomorphic =
            occ.caller_binder_names.is_empty() && occ.caller_own_tvars.is_empty();
        let (message, suggestions) = if caller_is_monomorphic {
            (
                format!(
                    "polymorphic recursion: `{caller}` makes an in-group recursive call to \
                     `{callee}` instantiated at `[{got}]`, but `{caller}` declares no type \
                     parameters, so a call inside this recursive binding group cannot \
                     instantiate `{callee}`'s type parameters afresh \
                     (spec/04-type-system.md \u{a7}3.1.1 [04-INF-3])",
                    caller = occ.caller,
                    callee = occ.callee,
                ),
                // A helper that made the call would itself call into the
                // group and be called from it, so it would join the group:
                // no helper escapes the rule.
                vec![format!(
                    "give `{caller}` matching type parameters",
                    caller = occ.caller,
                )],
            )
        } else {
            (
                format!(
                    "polymorphic recursion: `{caller}` makes an in-group recursive call to \
                     `{callee}` instantiated at `[{got}]`, but every recursive call in a \
                     recursive binding group must be typed at the caller's own instantiation \
                     `[{expected}]` (spec/04-type-system.md \u{a7}3.1.1 [04-INF-3])",
                    caller = occ.caller,
                    callee = occ.callee,
                ),
                vec!["make the recursive call reuse the caller's own type parameters".to_string()],
            )
        };
        let mut err = CheckError::new(CheckErrorKind::TypeMismatch, message, suggestions);
        err.expected = Some(format!("[{expected}]"));
        err.got = Some(format!("[{got}]"));
        err.span_id = occ.span_id;
        err.span_offset = occ.span_offset;
        errors.push(err);
    }
}

/// Display names for the caller's own type parameters: when the pairing is
/// unambiguous (one declared binder, one own tvar), the resolved own tvar
/// renders under its declared name so the diagnostic reads `Box[a]` rather
/// than `Box[?17]`.
fn binder_display_names(
    occ: &Occurrence,
    subst: &Subst<ErrorWitness>,
) -> UnordMap<TypeVar, String> {
    let mut names = UnordMap::new();
    if occ.caller_binder_names.len() == 1 && occ.caller_own_tvars.len() == 1 {
        let resolved = subst.apply(&Type::Var(occ.caller_own_tvars[0]));
        if let Type::Var(v) = resolved {
            names.insert(v, occ.caller_binder_names[0].clone());
        }
    }
    names
}

fn render_type(
    ty: &Type<ErrorWitness>,
    names: &UnordMap<TypeVar, String>,
    fallback: Option<&str>,
) -> String {
    match ty {
        Type::Prim(p) => prim_name(*p),
        Type::Var(v) => names.get(v).cloned().unwrap_or_else(|| {
            fallback
                .map(str::to_string)
                .unwrap_or_else(|| format!("?{}", v.0))
        }),
        Type::Adt(name, args) => {
            if args.is_empty() {
                name.clone()
            } else {
                let rendered: Vec<String> = args
                    .iter()
                    .map(|arg| render_type(arg, names, fallback))
                    .collect();
                format!("{name}[{}]", rendered.join(", "))
            }
        }
        Type::KindedAdt(name, args) => {
            let rendered = args
                .iter()
                .map(|argument| match argument {
                    NominalArg::Type(ty) => render_type(ty, names, fallback),
                    NominalArg::Dimension(dim) => dim.to_string(),
                })
                .collect::<Vec<_>>();
            format!("{name}[{}]", rendered.join(", "))
        }
        Type::Fn(args, ret) => {
            let rendered: Vec<String> = args
                .iter()
                .map(|arg| render_type(arg, names, fallback))
                .collect();
            format!(
                "({}) -> {}",
                rendered.join(", "),
                render_type(ret, names, fallback)
            )
        }
        Type::Tuple(elems) => {
            let rendered: Vec<String> = elems
                .iter()
                .map(|elem| render_type(elem, names, fallback))
                .collect();
            format!("({})", rendered.join(", "))
        }
        Type::Ref(inner) => format!("&{}", render_type(inner, names, fallback)),
        Type::Unit => "unit".to_string(),
        Type::Tensor(..) | Type::Error(_) => ty.to_string(),
    }
}

fn prim_name(p: Prim) -> String {
    p.name().to_string()
}
