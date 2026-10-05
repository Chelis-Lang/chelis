//! Shared inference helpers.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

/// Observe a decoded vocabulary node's tag, metadata and children.
///
/// A vocabulary node has the single spelling `Expr::Node` whatever its
/// ingress (chelis#1125), so this is the `DecodedNode` arm of
/// [`deep::Expr::carrier`] as an `Option` for readers that decline every other
/// carrier alike.
pub(super) fn stamped_parts(
    expr: &deep::Expr,
) -> Option<(DeepTag, &deep::Metadata, &[deep::Expr])> {
    match expr.carrier() {
        deep::ExprCarrier::DecodedNode(tag, metadata, children) => Some((tag, metadata, children)),
        deep::ExprCarrier::StructuralList(_)
        | deep::ExprCarrier::UndecodableHead(_, _, _)
        | deep::ExprCarrier::Atom(_)
        | deep::ExprCarrier::MetadataMap(_)
        | deep::ExprCarrier::MetadataExpression(_) => None,
    }
}

/// chelis#710 / spec/04-type-system.md §10 [04-TOT-3]: push a `MalformedForm`
/// diagnostic for a Deep node whose arity or shape the checker cannot type,
/// and return the `Type::Error` sentinel. This replaces the silent
/// `return Type::Error` arity guards that used to let structurally malformed
/// `.dp` check clean (score 1 with an empty error list). `expected` names the
/// well-formed shape, and the message also names the FOUND shape (the node's
/// tagged-children count) per §C2 -- the calibration example is the Deep
/// parser's own "found unknown tag ..." message.
pub(super) fn malformed_form(
    node: &DeepNode,
    tag: &str,
    expected: &str,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    let found = node.child_count();
    report(
        errors,
        CheckError::new(
            CheckErrorKind::MalformedForm,
            format!(
                "malformed `{tag}`: expected {expected}, found a `{tag}` with {found} \
                 child element(s) (spec/03-deep-syntax.md; chelis#731 [04-TOT-3])"
            ),
            vec![],
        ),
    )
}

/// chelis#731 Phase 2 (spec/design/checker_totality.md §C3): cascade
/// suppression helper. When one of `tys` already typed as `Type::Error(w)`,
/// return `Some(propagate(w))` so the enclosing node re-types as `Type::Error`
/// WITHOUT re-reporting -- the pre-token permissive behavior where an error's
/// descendants unify freely so one root cause does not spray a dozen
/// secondary diagnostics. Returns `None` when no operand is an error, so the
/// caller falls through to its normal typing. The witness is threaded from
/// the real upstream `report`, keeping the cascade provably downstream of a
/// reported error.
pub(super) fn propagate_if_error<'a>(tys: impl IntoIterator<Item = &'a Type>) -> Option<Type> {
    tys.into_iter().find_map(|ty| match ty {
        Type::Error(w) => Some(propagate(w)),
        _ => None,
    })
}

/// Use the same source-location carrier as Deep type diagnostics. Native Deep
/// supplies a measured coordinate; Surf supplies the opaque `surf:` identity.
pub(super) fn at_check_site(site: &deep::Expr, error: CheckError) -> CheckError {
    match TypeDiagnosticLocation::from_expr(site) {
        Some(span) => span.attach(error),
        None => error,
    }
}

pub(super) fn at_check_node(node: Option<&DeepNode>, error: CheckError) -> CheckError {
    match node.and_then(TypeDiagnosticLocation::from_node) {
        Some(span) => span.attach(error),
        None => error,
    }
}

/// Borrow the application node on eager paths; only suspended checks retain
/// an owned location. A successful call never clones its source identity.
#[derive(Clone, Copy)]
pub(super) enum CheckSite<'a> {
    Expr(&'a deep::Expr),
    Node(&'a DeepNode),
    Deferred(Option<&'a TypeDiagnosticLocation>),
}

pub(super) fn report_at_check_site(
    errors: &mut DiagnosticSink<'_>,
    error: CheckError,
    site: CheckSite<'_>,
) -> Type {
    match site {
        CheckSite::Expr(expr) => report(errors, at_check_site(expr, error)),
        CheckSite::Node(node) => report(errors, at_check_node(Some(node), error)),
        CheckSite::Deferred(location) => report_at(errors, error, location),
    }
}

impl CheckSite<'_> {
    pub(super) fn owned_location(self) -> Option<TypeDiagnosticLocation> {
        match self {
            Self::Expr(expr) => TypeDiagnosticLocation::from_expr(expr),
            Self::Node(node) => TypeDiagnosticLocation::from_node(node),
            Self::Deferred(location) => location.cloned(),
        }
    }
}

/// chelis#731 Phase 2 (spec/design/checker_totality.md §C3): report a
/// wrong-arity call of a shape-polymorphic builtin. Before the witness-token
/// migration these arity guards returned a SILENT `Type::Error` -- an
/// `einsum(a, b)` (two args) scored a perfect 1.0, the chelis#709 defect on
/// the builtin lane. The guard now pushes an `ArityMismatch` naming the
/// builtin and the expected/found counts, so the malformed call is rejected
/// with a diagnostic instead of exempted from checking.
/// A source-located arity guard for signature-checked builtins whose accepted
/// argument counts are ranges or alternatives rather than one exact count.
pub(super) fn report_builtin_arity_range(
    errors: &mut DiagnosticSink<'_>,
    site: CheckSite<'_>,
    builtin: &str,
    expected: &str,
    got: usize,
) -> Type {
    report_at_check_site(
        errors,
        CheckError::with_types(
            CheckErrorKind::ArityMismatch,
            format!(
                "builtin `{builtin}` expects {expected}, got {got} argument(s) \
                 (chelis#731 [04-TOT-3])"
            ),
            expected.to_string(),
            format!("{got} argument(s)"),
            vec![],
        ),
        site,
    )
}

pub(super) fn report_builtin_arity(
    errors: &mut DiagnosticSink<'_>,
    node: &DeepNode,
    site: CheckSite<'_>,
    builtin: &str,
    expected: usize,
    got: usize,
) -> Type {
    report_at_check_site(
        errors,
        CheckError::with_types(
            CheckErrorKind::ArityMismatch,
            with_node_provenance(
                node,
                format!(
                    "builtin `{builtin}` expects {expected} argument(s), got {got} argument(s) \
                     (chelis#731 [04-TOT-3])"
                ),
            ),
            format!("{expected} argument(s)"),
            format!("{got} argument(s)"),
            vec![],
        ),
        site,
    )
}

/// True when a `deftype` node carries `opaque: true` metadata
/// (RFC D-META; the key is unprefixed language semantics).
pub(super) fn deftype_opaque_meta(meta: &deep::Metadata) -> bool {
    meta.opaque().is_some()
}

/// Whether `expr` is the finite, untyped `Cons`/`Nil` chain a bracket literal
/// desugars to (spec/02-surf-syntax.md §P10b).
pub(super) fn is_bracket_literal(expr: &deep::Expr) -> bool {
    let is_variable = |expr: &deep::Expr, name: &str| {
        matches!(stamped_parts(expr), Some((DeepTag::Var, meta, [atom]))
            if meta.ty().is_none() && symbol_name(atom) == Some(name))
    };
    let mut tail = expr;
    loop {
        if is_variable(tail, "Nil") {
            return true;
        }
        match stamped_parts(tail) {
            Some((DeepTag::App, meta, [cons, _, rest]))
                if meta.ty().is_none() && is_variable(cons, "Cons") =>
            {
                tail = rest;
            }
            _ => return false,
        }
    }
}

/// Extract a symbol name from an Expr.
pub(super) fn symbol_name(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Atom(deep::Atom::Name(s), _) => Some(s.as_str()),
        _ => None,
    }
}

pub(super) fn with_macro_provenance(expr: &deep::Expr, message: String) -> String {
    let Some(source) = macro_source(expr) else {
        return message;
    };
    format!("{message} (in expansion of {source})")
}

/// [`with_macro_provenance`] for a node the caller already holds: the
/// provenance is the node's own `source` metadata.
pub(super) fn with_node_provenance(node: &DeepNode, message: String) -> String {
    let Some(source) = node.meta().source() else {
        return message;
    };
    format!(
        "{message} (in expansion of {})",
        chelis_deep::printer::print_macro_source(source)
    )
}

// chelis#317 constructor-scope invariant (read before touching the helpers
// below). The reef name resolver guarantees that every constructor reference
// which is genuinely *in scope* — declared in the current module, imported by
// name, or module-qualified — reaches the type checker rewritten to its exact
// reef-mangled name (`Pkg__Mod__Ctor`), and the matching `deftype` registers
// that exact name in both the type env and the ADT registry. A bare,
// un-mangled constructor name therefore arrives at type-check ONLY when the
// importing module never brought it into scope (a type-only import, or no
// import at all). The `lookup_terminal_unique` / `lookup_variant_terminal_unique`
// fuzzy fallbacks are diagnostic-only: they exist so a partially-mangled or
// out-of-scope name can be *named* in an error, never to bind a reference to a
// scope. The guards below depend on this: an in-scope constructor is always
// exact-bound, so rejecting a name that resolves only through the fuzzy
// fallback (or not at all) cannot reject an in-scope constructor. A future
// half-mangled producer (mangled `deftype`, bare reference) would violate the
// invariant and be wrongly rejected here — which is the intended failure mode:
// a half-mangled program is a structural defect, not a valid reference. The
// `ir_resolves_consistently_mangled_constructor_names` /
// `ir_rejects_out_of_scope_terminal_constructor_name` unit tests pin both
// directions.

/// The terminal (last) segment of a possibly module-qualified or
/// reef-mangled name. Mirrors `env::terminal_name` / `adt::terminal_name`:
/// `Pkg__Demo__Adt__Alpha` and `Demo.Adt.Alpha` both have terminal `Alpha`.
pub(super) fn terminal_segment(name: &str) -> &str {
    name.rsplit_once("__")
        .map(|(_, tail)| tail)
        .or_else(|| name.rsplit_once('.').map(|(_, tail)| tail))
        .unwrap_or(name)
}

/// Whether `name` is an ADT constructor reference by Chelis nomenclature:
/// its terminal segment starts with an uppercase ASCII letter (§3.1, the
/// same rule the surf parser uses to classify a bare uppercase identifier
/// as `Expr::Constructor` / `Pattern::Constructor`). Type names are also
/// PascalCase, but they never reach value-position `var`/`pat-ctor`
/// resolution, so an uppercase terminal in those positions is a
/// constructor.
pub(super) fn is_constructor_name(name: &str) -> bool {
    terminal_segment(name)
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase())
}

/// A constructor-position reference is *in scope* only when its exact name is
/// present in `env`'s constructor authority — either a bare builtin constructor
/// (`Some`/`None`/`Cons`/`Nil`, registered bare by
/// `register_prelude_adts`) or a reef-mangled in-scope constructor
/// (chelis#157/#316 rewrite the reference to its mangled name when the
/// importing module declares it locally or imports it by name).
///
/// Returns `true` when `name` looks like a constructor (PascalCase terminal)
/// but is not bound exactly. Whether the package registry contains zero, one,
/// or several foreign same-terminal constructors is deliberately irrelevant:
/// consulting that global population would let unrelated modules change this
/// reference's diagnostic kind. A type-only import can therefore neither
/// fuzzy-bind to another module's mangled tag nor fall through as an ordinary
/// unbound value. Every non-exact constructor reference is rejected at
/// `check` as an unknown constructor.
pub(super) fn constructor_out_of_scope(name: &str, env: &Env) -> bool {
    is_constructor_name(name) && env.lookup_constructor(name).is_none()
}

/// Bare value references have one extra disambiguation step: an exact ordinary
/// binding wins before an uppercase spelling can be diagnosed as an unknown
/// constructor (spec/01 §3.2). Constructor application and pattern positions
/// must not use this helper because a same-named lexical value does not replace
/// structural constructor authority there.
pub(super) fn bare_constructor_out_of_scope(name: &str, env: &Env) -> bool {
    is_constructor_name(name) && env.lookup(name).is_none()
}

/// Resolve an exact in-scope constructor candidate for a call shape.
///
/// Mixed positional/record collisions select the matching shape regardless of
/// declaration order (chelis#148). Candidates come only from the environment's
/// structural constructor scope, so an unrelated registry entry cannot become
/// visible merely because another constructor with the same spelling is in
/// scope. When several candidates share a shape, the latest declaration keeps
/// the established active-owner precedence.
pub(super) fn constructor_for_shape<'env, 'adt>(
    name: &str,
    call_shape: CallShape,
    env: &'env Env,
    adt_reg: &'adt AdtRegistry,
) -> Option<(&'env str, &'env Scheme, &'adt crate::adt::VariantInfo)> {
    let want_named = matches!(call_shape, CallShape::Record);
    let mut fallback = None;
    for (owner, scheme) in env.lookup_constructors(name).rev() {
        let variant = adt_reg
            .lookup(owner)?
            .variants
            .iter()
            .find(|variant| variant.name == name)?;
        fallback.get_or_insert((owner, scheme, variant));
        let shape_matches = !variant.fields.is_empty()
            && variant
                .fields
                .iter()
                .all(|(field_name, _)| field_name.is_some() == want_named);
        if shape_matches {
            return Some((owner, scheme, variant));
        }
    }
    fallback
}

/// Resolve a pattern constructor against the nominal scrutinee owner already
/// established by inference. That owner is stronger than a flat same-named
/// registry lookup and preserves existing shape-collision semantics when two
/// in-scope ADTs deliberately share a constructor spelling (chelis#148). The
/// structural constructor map remains the scope gate and the fallback when
/// the scrutinee is not nominal yet.
pub(super) fn pattern_constructor_for_scrutinee<'adt>(
    name: &str,
    call_shape: CallShape,
    scrutinee_ty: &Type,
    env: &Env,
    adt_reg: &'adt AdtRegistry,
) -> Option<(
    &'adt str,
    &'adt crate::adt::AdtDef,
    &'adt crate::adt::VariantInfo,
)> {
    if constructor_out_of_scope(name, env) {
        return None;
    }

    if let Type::Adt(owner, _) | Type::KindedAdt(owner, _) = scrutinee_ty
        && let Some(definition) = adt_reg.lookup(owner)
        && let Some(variant) = definition
            .variants
            .iter()
            .find(|variant| variant.name == name)
    {
        return Some((definition.name.as_str(), definition, variant));
    }

    let (owner, _, variant) = constructor_for_shape(name, call_shape, env, adt_reg)?;
    let definition = adt_reg.lookup(owner)?;
    Some((definition.name.as_str(), definition, variant))
}

pub(super) fn check_error_kind_from_type_error_kind(kind: &TypeErrorKind) -> CheckErrorKind {
    match kind {
        TypeErrorKind::TypeMismatch => CheckErrorKind::TypeMismatch,
        TypeErrorKind::PrecisionMismatch | TypeErrorKind::DtypeFamilyMismatch => {
            CheckErrorKind::PrecisionMismatch
        }
        TypeErrorKind::KeyInstantiation { .. } => CheckErrorKind::KeyReuse,
        TypeErrorKind::DimensionMismatch => CheckErrorKind::DimensionMismatch,
        TypeErrorKind::ArityMismatch => CheckErrorKind::ArityMismatch,
        TypeErrorKind::OccursCheck => CheckErrorKind::OccursCheck,
        TypeErrorKind::NotAFunction => CheckErrorKind::NotAFunction,
    }
}

pub(super) fn extract_string_literal(expr: &deep::Expr) -> Option<String> {
    match expr {
        deep::Expr::Atom(deep::Atom::Str(value), _) => Some(value.clone()),
        deep::Expr::Node(node, _) if node.tag() == DeepTag::Lit => {
            node.children_slice().first().and_then(|child| match child {
                deep::Expr::Atom(deep::Atom::Str(value), _) => Some(value.clone()),
                _ => None,
            })
        }
        _ => None,
    }
}

/// Narrow `Dim::Wildcard` slots in `ty` against the matching positions in
/// `template`, replacing each Wildcard with the template's concrete dim
/// where one is available. Used after a defsig unify to ensure the scheme
/// registered for callers reflects the declared concrete shape rather than
/// the body's permissive wildcards (#39).
///
/// This is structural and conservative: it only walks shapes that match
/// (same rank for tensors, same arity for fn/tuple/adt), and only narrows
/// Wildcard → a safe template dim. If shapes don't line up, the input is
/// returned unchanged so genuine type errors flagged by `unify` aren't
/// masked.
///
/// Which template dims are "safe" to substitute:
///
/// - `Dim::Lit(_)` — always safe. Concrete literals are self-contained.
///
/// - `Dim::Var(v)` or `Dim::Name(v)` — safe ONLY when `v` is also bound by a *parameter*
///   tensor-dim position of the declared signature (`param_dims`). This
///   is the `const_col[n](spots: tensor[n, f32], ..) -> tensor[n, 1]`
///   family: the body's `to_tensor(map(..))` return is `tensor[*, 1]`,
///   but the declared return dim `n` is the same dim var as the `spots`
///   parameter's axis-0, so the caller binds `n` from its actual argument.
///   Narrowing `*` → `Var(n)` reconnects the return to the input dim,
///   which is what downstream lowering needs so a `vmap` lane kernel reads
///   a Load-declared batch dim instead of an undeclared `_anon_dim`
///   (chelis#405 / WS-3 build ICE).
///
///   A *return-only* dim var (one that appears in the declared return but
///   in NO parameter tensor position — e.g. `arange[n](start: i32,
///   stop: i32) -> tensor[n, i32]`, where the length comes from a
///   value parameter) is NOT in `param_dims` and is deliberately left as
///   `Wildcard`. Baking such an unbound var into the generalized scheme is
///   the red-team RT-39+44 soundness regression (commit 8067c9ce): it
///   leaks a free dim var into callers and breaks the chelis-std self-test
///   corpus. The `param_dims` gate is exactly the line between "the
///   caller supplies this dim" (safe) and "this dim is output-inferred /
///   value-parameter-derived" (unsafe).
///
/// - `Dim::Name`, `Dim::Rank`, existing `Var`/`Lit` in `ty` — preserved.
pub(super) fn narrow_wildcards_with(ty: &Type, template: &Type, param_dims: &[Dim]) -> Type {
    match (ty, template) {
        (Type::Tensor(dims, prec), Type::Tensor(tmpl_dims, _)) if dims.len() == tmpl_dims.len() => {
            let new_dims = dims
                .iter()
                .zip(tmpl_dims.iter())
                .map(|(d, t)| match (d, t) {
                    (Dim::Wildcard, Dim::Lit(_)) => t.clone(),
                    (Dim::Wildcard, Dim::Var(_) | Dim::Name(_)) if param_dims.contains(t) => {
                        t.clone()
                    }
                    _ => d.clone(),
                })
                .collect();
            Type::Tensor(new_dims, prec.clone())
        }
        (Type::Fn(args, ret), Type::Fn(t_args, t_ret)) if args.len() == t_args.len() => {
            let new_args = args
                .iter()
                .zip(t_args.iter())
                .map(|(a, t)| narrow_wildcards_with(a, t, param_dims))
                .collect();
            let new_ret = Box::new(narrow_wildcards_with(ret, t_ret, param_dims));
            Type::Fn(new_args, new_ret)
        }
        (Type::Tuple(ts), Type::Tuple(t_ts)) if ts.len() == t_ts.len() => {
            let new_ts = ts
                .iter()
                .zip(t_ts.iter())
                .map(|(t, tt)| narrow_wildcards_with(t, tt, param_dims))
                .collect();
            Type::Tuple(new_ts)
        }
        (Type::Adt(n, args), Type::Adt(_, t_args)) if args.len() == t_args.len() => {
            let new_args = args
                .iter()
                .zip(t_args.iter())
                .map(|(a, t)| narrow_wildcards_with(a, t, param_dims))
                .collect();
            Type::Adt(n.clone(), new_args)
        }
        (Type::KindedAdt(n, args), Type::KindedAdt(_, t_args)) if args.len() == t_args.len() => {
            let new_args = args
                .iter()
                .zip(t_args)
                .map(|(argument, template)| match (argument, template) {
                    (NominalArg::Type(ty), NominalArg::Type(template)) => {
                        NominalArg::Type(narrow_wildcards_with(ty, template, param_dims))
                    }
                    (
                        NominalArg::Dimension(Dim::Wildcard),
                        NominalArg::Dimension(template @ Dim::Lit(_)),
                    ) => NominalArg::Dimension(template.clone()),
                    (
                        NominalArg::Dimension(Dim::Wildcard),
                        NominalArg::Dimension(template @ (Dim::Var(_) | Dim::Name(_))),
                    ) if param_dims.contains(template) => NominalArg::Dimension(template.clone()),
                    _ => argument.clone(),
                })
                .collect();
            Type::KindedAdt(n.clone(), new_args)
        }
        _ => ty.clone(),
    }
}

/// Collect the dimension variables and names in a *parameter* (non-return)
/// tensor-dim position of a resolved declared `Fn` signature. These are
/// the dims a caller binds from its actual arguments; the
/// `narrow_wildcards_with` gate uses this set to decide when a body
/// wildcard may be safely narrowed to its declared binder. A non-`Fn`
/// type (or one whose params carry no dimension binders) yields the empty
/// set, so narrowing falls back to the literal-only behavior.
pub(super) fn param_bound_dims(decl_ty: &Type) -> Vec<Dim> {
    let mut out = Vec::new();
    let Type::Fn(params, _) = decl_ty else {
        return out;
    };
    let mut pending = params.iter().collect::<Vec<_>>();
    while let Some(ty) = pending.pop() {
        match ty {
            Type::Tensor(dims, _) => out.extend(
                dims.iter()
                    .filter(|dim| matches!(dim, Dim::Name(_) | Dim::Var(_)))
                    .cloned(),
            ),
            Type::Ref(inner) => pending.push(inner),
            Type::Tuple(items) | Type::Adt(_, items) => pending.extend(items),
            Type::KindedAdt(_, items) => {
                for item in items {
                    match item {
                        NominalArg::Type(ty) => pending.push(ty),
                        NominalArg::Dimension(dim @ (Dim::Name(_) | Dim::Var(_))) => {
                            out.push(dim.clone());
                        }
                        NominalArg::Dimension(_) => {}
                    }
                }
            }
            Type::Fn(args, result) => {
                pending.extend(args);
                pending.push(result);
            }
            Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error(_) => {}
        }
    }
    out
}

/// Decide a shape-computing call now, or suspend it if the operand is not
/// resolved yet (chelis#1489).
///
/// The ONE entry point every shape-route arm uses, so "decide now" and "decide
/// later" cannot drift apart: both end in [`shape_route_result`], and the only
/// difference is when.
///
/// Suspending returns a FRESH result variable rather than the operand's,
/// exactly as the `copy`/`cast` gates do, so the expression's type cannot
/// depend on when the operand resolved.
pub(crate) fn decide_shape_route(
    route: crate::unify::ShapeRoute,
    operand: &Type,
    node: &DeepNode,
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    if let Type::Var(tv) = operand {
        let result = vg.fresh_type();
        subst.record_deferred_tensor_operand(
            *tv,
            crate::unify::DeferredOperandGate::ShapeRoute {
                route,
                result: Box::new(result.clone()),
            },
        );
        return result;
    }
    match shape_route_result(
        &route,
        operand,
        SumResultSlot::Fresh(vg),
        TypeDiagnosticLocation::from_node(node),
        subst,
    ) {
        Ok((result, updates)) => {
            if let Some(expected_updates) = updates
                && let crate::unify::ShapeRoute::Gather {
                    updates: Some(actual),
                    ..
                } = &route
                && let Err(te) = unify(&expected_updates, actual.as_ref(), subst)
            {
                return report(errors, te.into());
            }
            result
        }
        Err(message) => report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                with_node_provenance(node, message),
                vec![],
            ),
        ),
    }
}

/// Decide a shape-computing call (chelis#1489).
///
/// THE decision function for `gather`, `scatter`, `scatter_replace` and
/// `trace`. The eager arm calls it with the operand it just resolved; a
/// suspended [`crate::unify::DeferredOperandGate::ShapeRoute`] calls it with
/// the operand the binding just settled. One implementation, so the two cannot
/// disagree -- an earlier revision hand-copied the arm's tail here instead and
/// silently lost `scatter`'s mode check and the borrow normalization.
///
/// Everything the call decides happens HERE, not in the arm: the operand is
/// normalized through one borrow where the route's arm does so, the axis is
/// normalized against the operand's own rank, and the mode string is
/// validated. The arm's remaining job is to report the `Err`, and to unify the
/// updates obligation this returns.
///
/// These routes only COPY the operand's dims into their result, so a dim that
/// is still a variable when this runs passes through and resolves later.
/// `concat` and `diagonal` compute a new extent from those dims and are NOT
/// decided here: see the deferred-operand ledger's doc in `unify.rs`.
///
/// Returns `(result type, updates obligation)`. The updates unification is the
/// caller's because discharge runs inside unification while the eager arm has
/// a `&mut Subst`; the DECISION of what updates must equal is here.
///
/// A `trace` whose result precision waits on an inference variable publishes
/// a type from `slot`, gated to the decided result ([`publish_sum_result`]).
pub(crate) fn shape_route_result(
    route: &crate::unify::ShapeRoute,
    operand: &Type,
    slot: SumResultSlot<'_>,
    location: Option<TypeDiagnosticLocation>,
    subst: &Subst,
) -> Result<(Type, Option<Type>), String> {
    use crate::unify::ShapeRoute;
    // Strip one borrow only for the routes whose arms do, so a deferred
    // `gather(&t, ..)` accepts what the eager one accepts WITHOUT widening
    // `scatter`/`scatter_replace`, which reject a borrowed operand.
    let operand = match operand {
        Type::Ref(inner) if route.normalizes_borrow() => inner.as_ref(),
        other => other,
    };
    match route {
        ShapeRoute::Gather {
            op,
            indices,
            raw_axis,
            updates,
            mode,
        } => {
            // Axis bounds first, then the mode string: the order the eager arm
            // had, so a call that is wrong in both ways reports the same one.
            let axis = settled_axis(op, operand, *raw_axis, 0)?;
            if let Some(mode) = mode
                && !matches!(mode.as_str(), "replace" | "add")
            {
                return Err(format!("{op} mode must be \"replace\" or \"add\""));
            }
            let result = infer_gather_result_type(op, operand, indices.as_ref(), axis)?;
            match updates {
                // `scatter` returns the BASE tensor; the gathered shape is what
                // its updates operand must equal.
                Some(_) => Ok((operand.clone(), Some(result))),
                None => Ok((result, None)),
            }
        }
        ShapeRoute::Trace {
            raw_axis1,
            raw_axis2,
        } => {
            let axis1 = settled_axis("trace", operand, *raw_axis1, 0)?;
            let axis2 = settled_axis("trace", operand, *raw_axis2, 1)?;
            Ok((
                infer_trace_result_type(operand, axis1, axis2, slot, location, subst)?,
                None,
            ))
        }
    }
}

/// Normalize a raw axis against the operand's rank (chelis#1489).
///
/// The eager resolvers, with the same message text. `default` is the axis the
/// call uses when none was written, matching their fallback.
///
/// A non-tensor operand yields `default` rather than an error, exactly as the
/// eager resolvers do: the route's own helper rejects it, and it does so AFTER
/// `scatter`'s mode check. Rejecting here instead made a call that is wrong in
/// both ways report the operand error where it had reported the mode error.
fn settled_axis(
    op: &str,
    operand: &Type,
    raw: Option<i64>,
    default: usize,
) -> Result<usize, String> {
    let Type::Tensor(dims, _) = operand else {
        return Ok(default);
    };
    match raw {
        Some(raw) => normalize_static_axis(dims.len(), raw)
            .ok_or_else(|| format!("{op} axis {raw} out of bounds for rank {}", dims.len())),
        None => Ok(default),
    }
}

pub(super) fn infer_gather_result_type(
    op: &str,
    tensor_ty: &Type,
    indices_ty: &Type,
    axis: usize,
) -> Result<Type, String> {
    let Type::Tensor(tensor_dims, tensor_precision) = tensor_ty else {
        return Err(format!("{op} expects tensor input, got {tensor_ty}"));
    };
    let Type::Tensor(index_dims, index_precision) = indices_ty else {
        return Err(format!(
            "{op} expects integer tensor indices, got {indices_ty}"
        ));
    };
    if !index_precision.is_integer() {
        return Err(format!(
            "{op} expects integer tensor indices, got tensor[..., {}]",
            index_precision.name()
        ));
    }
    if axis >= tensor_dims.len() {
        return Err(format!(
            "{op} axis {axis} out of bounds for rank {}",
            tensor_dims.len()
        ));
    }
    let mut out_dims = tensor_dims[..axis].to_vec();
    out_dims.extend(index_dims.clone());
    out_dims.extend_from_slice(&tensor_dims[axis + 1..]);
    Ok(Type::Tensor(out_dims, tensor_precision.clone()))
}

pub(super) fn infer_trace_result_type(
    tensor_ty: &Type,
    axis1: usize,
    axis2: usize,
    slot: SumResultSlot<'_>,
    location: Option<TypeDiagnosticLocation>,
    subst: &Subst,
) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = tensor_ty else {
        return Err(format!("trace expects tensor input, got {tensor_ty}"));
    };
    if axis1 >= dims.len() || axis2 >= dims.len() || axis1 == axis2 {
        return Err(format!(
            "trace expects distinct in-bounds axes, got {axis1} and {axis2} for rank {}",
            dims.len()
        ));
    }
    let out_dims = dims
        .iter()
        .enumerate()
        .filter_map(|(index, dim)| ((index != axis1) && (index != axis2)).then_some(dim.clone()))
        .collect();
    let precision = default_sum_result_precision("trace", precision, subst)?;
    Ok(publish_sum_result(
        "trace", out_dims, precision, slot, location, subst,
    ))
}

/// The result of `einsum(equation, left, right)` under [05-OP-33], for two
/// settled tensor operands whose precisions the caller has already unified:
/// each output label takes the extent of its first occurrence scanning left
/// then right, at `sum_result(p, default(p))`. An equation outside the
/// grammar, a label count that differs from an operand's rank, an output
/// label absent from both inputs or repeated, and a rank-spread operand,
/// whose rank no fixed equation can match at every instantiation, are each
/// rejected; there is no undecided outcome.
pub(super) fn infer_einsum_result_type(
    equation: &str,
    left: &Type,
    right: &Type,
    slot: SumResultSlot<'_>,
    location: Option<TypeDiagnosticLocation>,
    subst: &Subst,
) -> Result<Type, String> {
    let (Type::Tensor(left_dims, precision), Type::Tensor(right_dims, _)) = (left, right) else {
        return Err(format!(
            "einsum expects two tensor operands, got {left} and {right}"
        ));
    };
    let grammar = || {
        format!(
            "einsum equation `{equation}` must match `[a-z]*,[a-z]*->[a-z]*` \
             (spec/05-risc-primitives.md [05-OP-33])"
        )
    };
    let Some((inputs, output)) = equation.split_once("->") else {
        return Err(grammar());
    };
    let Some((left_labels, right_labels)) = inputs.split_once(',') else {
        return Err(grammar());
    };
    if [left_labels, right_labels, output]
        .iter()
        .any(|labels| !labels.chars().all(|label| label.is_ascii_lowercase()))
    {
        return Err(grammar());
    }
    for (side, labels, dims) in [
        ("left", left_labels, left_dims),
        ("right", right_labels, right_dims),
    ] {
        if dims.iter().any(|dim| matches!(dim, Dim::Rank(_))) {
            return Err(format!(
                "einsum equation `{equation}` fixes the {side} operand's rank at {}, but \
                 the operand has a rank spread, which denotes every rank \
                 ([05-OP-33], [04-INF-6])",
                labels.len()
            ));
        }
        if labels.len() != dims.len() {
            return Err(format!(
                "einsum equation `{equation}` gives the {side} operand {} labels, but it \
                 has rank {} ([05-OP-33])",
                labels.len(),
                dims.len()
            ));
        }
    }
    let mut dims = Vec::with_capacity(output.len());
    for (index, label) in output.chars().enumerate() {
        if output.chars().take(index).any(|earlier| earlier == label) {
            return Err(format!(
                "einsum output label `{label}` must occur exactly once in `{equation}` \
                 ([05-OP-33])"
            ));
        }
        let dim = left_labels
            .chars()
            .position(|candidate| candidate == label)
            .map(|position| &left_dims[position])
            .or_else(|| {
                right_labels
                    .chars()
                    .position(|candidate| candidate == label)
                    .map(|position| &right_dims[position])
            })
            .ok_or_else(|| {
                format!(
                    "einsum output label `{label}` must occur in an input of `{equation}` \
                     ([05-OP-33])"
                )
            })?;
        dims.push(dim.clone());
    }
    let precision = default_sum_result_precision("einsum", precision, subst)?;
    Ok(publish_sum_result(
        "einsum", dims, precision, slot, location, subst,
    ))
}

/// How a `sum`-family call's result precision stands when the call is checked.
pub(super) enum SumResultPrecision {
    /// The result precision is known now.
    Decided(TensorPrec),
    /// The operand's precision is an inference variable whose dtypes have no
    /// single result. The call publishes its own result type and a
    /// [`crate::unify::DeferredOperandGate::SumResult`] decides it once the
    /// variable binds, or at the declaration boundary.
    Pending(TypeVar),
}

/// `sum_result(p, default(p))`, spec/04 §5.7.1: the result dtype of an
/// operation that accumulates with the default sum accumulator (`sum`,
/// `cumsum`, `trace`, `einsum`). i8 and i16 widen to i32; a dtype with no
/// default accumulator, such as bool, is rejected.
///
/// Over a precision variable the result is decided now when one type covers
/// every dtype the variable admits ([`bound_sum_result_precision`]).
/// Otherwise it waits for the variable: inference may still bind it (a hole,
/// or a lambda parameter's precision), and an authored binder is decided at
/// the declaration boundary over every dtype its bound admits ([04-INF-6]).
pub(super) fn default_sum_result_precision(
    op: &str,
    precision: &TensorPrec,
    subst: &Subst,
) -> Result<SumResultPrecision, String> {
    match precision {
        TensorPrec::Concrete(prim) => settled_sum_result_precision(op, &Type::Prim(*prim))
            .map(|result| SumResultPrecision::Decided(TensorPrec::Concrete(result))),
        TensorPrec::Var(variable) => Ok(match bound_sum_result_precision(op, *variable, subst) {
            Ok(result) => SumResultPrecision::Decided(result),
            Err(_) => SumResultPrecision::Pending(*variable),
        }),
    }
}

/// Where a waiting `sum`-family result gets the type it publishes.
pub(crate) enum SumResultSlot<'a> {
    /// The checked call: publish `dims` at a fresh precision variable, so a
    /// consumer sees the result's shape before its precision is decided.
    Fresh(&'a mut VarGen),
    /// A deferred decision: publish the type the call already handed out.
    Existing(&'a Type),
}

/// The result tensor a `sum`-family call publishes: `dims` at the decided
/// precision, or, while the precision waits, a type from `slot` with a gate
/// that unifies `dims` at the decided precision into it later.
pub(super) fn publish_sum_result(
    op: &str,
    dims: Vec<Dim>,
    precision: SumResultPrecision,
    slot: SumResultSlot<'_>,
    location: Option<TypeDiagnosticLocation>,
    subst: &Subst,
) -> Type {
    match precision {
        SumResultPrecision::Decided(precision) => Type::Tensor(dims, precision),
        SumResultPrecision::Pending(variable) => {
            let published = match slot {
                SumResultSlot::Fresh(vg) => {
                    Type::Tensor(dims.clone(), TensorPrec::Var(vg.fresh_tvar()))
                }
                SumResultSlot::Existing(published) => published.clone(),
            };
            subst.record_deferred_tensor_operand(
                variable,
                crate::unify::DeferredOperandGate::SumResult {
                    op: op.to_string(),
                    dims,
                    result: Box::new(published.clone()),
                    location,
                },
            );
            published
        }
    }
}

/// `sum_result(p, default(p))` for a settled precision.
pub(crate) fn settled_sum_result_precision(op: &str, resolved: &Type) -> Result<Prim, String> {
    match resolved {
        Type::Prim(prim) => prim
            .default_reduce_sum_result_precision()
            .map_err(|message| {
                format!(
                    "{op} expects a tensor precision with a default sum accumulator, so that its \
                 result sum_result(p, default(p)) exists (spec/04-type-system.md §5.7.1), \
                 got {}; {message}",
                    prim.name()
                )
            }),
        other => Err(format!(
            "{op} expects a tensor precision with a default sum accumulator, so that its \
             result sum_result(p, default(p)) exists (spec/04-type-system.md §5.7.1), \
             got {other}"
        )),
    }
}

/// `sum_result(p, default(p))` over every dtype the precision variable admits
/// ([04-INF-6]): the variable itself when sum_result keeps each one (a
/// float, i32 or i64 bound), one concrete dtype when it maps them all there
/// (`{i8, i16}` gives i32), and otherwise no single type.
pub(crate) fn bound_sum_result_precision(
    op: &str,
    variable: TypeVar,
    subst: &Subst,
) -> Result<TensorPrec, String> {
    let restriction = subst.tvar_restriction(variable);
    let members = Prim::ACTIVE_FLOATS
        .into_iter()
        .chain(Prim::ACTIVE_INTEGERS)
        .chain([Prim::Bool])
        .filter(|prim| restriction.is_none_or(|bound| bound.admits(*prim)))
        .collect::<Vec<_>>();
    let results = members
        .iter()
        .map(|member| member.default_reduce_sum_result_precision().ok())
        .collect::<Vec<_>>();
    if members
        .iter()
        .zip(&results)
        .all(|(member, result)| *result == Some(*member))
    {
        return Ok(TensorPrec::Var(variable));
    }
    if let Some((Some(first), rest)) = results.split_first()
        && rest.iter().all(|result| *result == Some(*first))
    {
        return Ok(TensorPrec::Concrete(*first));
    }
    let bound = match restriction {
        Some(bound) => format!("its bound `{}`", bound.bound_spelling()),
        None => "an unbounded variable".to_string(),
    };
    Err(format!(
        "{op} over a tensor whose precision is a type variable has no single result dtype, \
         because spec/04 §5.7.1 sums an i8 or i16 operand in i32 and returns that i32 total \
         but returns i32, i64 and the floats at their own dtype, so sum_result(p, default(p)) \
         is not one dtype across the dtypes {bound} admits ([04-INF-6]); bound the variable to dtypes that \
         share one result (`Float`, `{{i32, i64}}`, or `{{i8, i16}}` with an i32 result), \
         or cast the operand to a concrete dtype and declare the widened result"
    ))
}

/// The spec/04 §5.7.1 repair note for a mismatch between a dtype `p` that
/// `sum`, `cumsum`, `trace` and `einsum` widen (i8, i16) and the dtype they
/// return for it (i32), in either order. `op` names the operation when the
/// mismatched expression is known to be one; otherwise the note names all
/// four conditionally. A function type compares its results, so a def whose
/// body mismatches its declared signature gets the note too.
pub(crate) fn sum_result_widening_note(
    op: Option<&str>,
    left: &Type,
    right: &Type,
) -> Option<String> {
    fn result_prim(ty: &Type) -> Option<Prim> {
        match ty {
            Type::Prim(prim) | Type::Tensor(_, TensorPrec::Concrete(prim)) => Some(*prim),
            Type::Fn(_, result) => result_prim(result),
            _ => None,
        }
    }
    let (left, right) = (result_prim(left)?, result_prim(right)?);
    let widened = |operand: Prim, result: Prim| {
        operand.default_reduce_sum_result_precision() == Ok(result) && operand != result
    };
    let (operand, result) = if widened(left, right) {
        (left.name(), right.name())
    } else if widened(right, left) {
        (right.name(), left.name())
    } else {
        return None;
    };
    Some(match op {
        Some(op @ ("sum" | "einsum")) => format!(
            "`{op}` over {operand} returns {result}, because spec/04 §5.7.1 sums {operand} in \
             {result} and returns that {result} total; declare the result as {result} (or pass \
             `accumulator=i64` and declare i64), or narrow it explicitly with \
             `cast(..., {operand})`"
        ),
        Some(op) => format!(
            "`{op}` over {operand} returns {result}, because spec/04 §5.7.1 sums {operand} in \
             {result} and returns that {result} total; declare the result as {result}, or \
             narrow it explicitly with `cast(..., {operand})`"
        ),
        None => format!(
            "if the {result} value is the result of `sum`, `cumsum`, `trace` or `einsum` over \
             {operand}, spec/04 §5.7.1 widened it, because those operations sum {operand} in \
             {result} and return that {result} total; declare the result as {result} (or, for \
             `sum` and `einsum`, pass `accumulator=i64` and declare i64), or narrow it \
             explicitly with `cast(..., {operand})`"
        ),
    })
}

/// spec/04 §5.7.1's permitted-pairs table for an explicit accumulator:
/// the result dtype of `operation` (`matmul`, `sum` or `einsum`) over
/// operand dtype `operand` accumulating in `accumulator`, or the diagnostic
/// for a pair the table omits. `sum` and `einsum` return
/// `sum_result(p, a)`; `matmul` returns its operand dtype.
pub(crate) fn explicit_accumulator_result(
    operation: &str,
    operand: Prim,
    accumulator: Prim,
) -> Result<Prim, String> {
    use Prim::*;
    let permitted: &[Prim] = match (operation, operand) {
        (_, Bf16 | F16 | F32) => &[F32, F64],
        (_, F64) => &[F64],
        ("matmul", _) => &[],
        (_, Int8 | Int16 | Int32) => &[Int32, Int64],
        (_, Int64) => &[Int64],
        _ => &[],
    };
    if permitted.contains(&accumulator) {
        return Ok(match (operation, operand) {
            ("matmul", _) | (_, Bf16 | F16) => operand,
            _ => accumulator,
        });
    }
    let names = permitted
        .iter()
        .map(|prim| format!("`accumulator={}`", prim.name()))
        .collect::<Vec<_>>();
    if names.is_empty() {
        return Err(format!(
            "`{operation}` over {} admits no accumulator (spec/04 §5.7.1)",
            operand.name()
        ));
    }
    let reason = if accumulator.is_integer() != operand.is_integer()
        || accumulator.is_float() != operand.is_float()
    {
        "an accumulator has its operand's numeric kind"
    } else {
        "an accumulator is at least as wide as its operand and its default"
    };
    Err(format!(
        "`{operation}` over {} does not admit `accumulator={}`, because {reason} \
         (spec/04 §5.7.1); omit the argument to accumulate in the default, or write {}",
        operand.name(),
        accumulator.name(),
        names.join(" or ")
    ))
}

/// The spec/04 §5.7.1 note for a binder bounded to admit i8 or i16 that a
/// body instantiated at `prim`, the i32 that `sum`, `cumsum`, `trace` and
/// `einsum` return for those operands.
pub(crate) fn sum_result_bound_note(admits_small_integer: bool, prim: Prim) -> Option<String> {
    (admits_small_integer && prim == Prim::Int32).then(|| {
        "if this i32 is the result of `sum`, `cumsum`, `trace` or `einsum` over i8 or i16, \
         spec/04 §5.7.1 widened it, because those operations sum i8 and i16 in i32 and return \
         that i32 total; declare that result as i32 (or, for `sum` and `einsum`, pass \
         `accumulator=i64` and declare i64), or bound the binder to dtypes that share one \
         sum result"
            .to_string()
    })
}

/// The `sum`-family operation `expr` evaluates to directly: a call to `sum`,
/// `cumsum`, `trace` or `einsum`, or a `let` or `fn` whose tail is one.
pub(crate) fn sum_family_tail_op(expr: &deep::Expr) -> Option<&'static str> {
    let mut tail = expr;
    loop {
        let (tag, _, kids) = stamped_parts(tail)?;
        match tag {
            DeepTag::App => {
                let name = ir_builtin_name_of_expr(kids.first()?)?;
                return ["sum", "cumsum", "trace", "einsum"]
                    .into_iter()
                    .find(|op| *op == name);
            }
            DeepTag::Let | DeepTag::Fn => tail = kids.last()?,
            _ => return None,
        }
    }
}

/// One selected axis of a `diagonal` pair, identified by its position in the
/// pair rather than by its index in the operand, so the decision function does
/// not have to carry the operand's axis numbering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DiagonalBound {
    /// `0` for the first selected axis, `1` for the second.
    pub(super) selected: usize,
    /// The literal extent on that axis. `min` can never exceed it.
    pub(super) extent: i64,
}

/// What `[05-OP-33]`'s "smaller selected extent" is, plus the upper bound a
/// literal selected axis imposes when the minimum itself is not statically
/// known.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DiagonalExtent {
    /// The dimension the retained axis carries in the result type.
    pub(super) dim: Dim,
    /// Present only when exactly one selected axis is literal. `min(x, k) <= k`
    /// for every runtime `x`, so a declared extent strictly greater than `k` is
    /// unreachable even though `dim` stays a wildcard (chelis#1739).
    pub(super) bound: Option<DiagonalBound>,
}

/// The single decision function for a `diagonal` axis pair.
///
/// Preserve chelis#1355's minimum/name policy while observing carrier
/// constraints: known extents declare their minimum, equal names retain the
/// selected identity when no known contradiction exists, and unknown pairs
/// keep the wildcard. `Dim` has no bounded variant for `min(n, 4)`.
///
/// `bound` is what chelis#1739 adds. A literal beside a non-literal extent is
/// an upper bound on the result: whatever `n` is at run time, `min(n, 4) <= 4`.
/// The checker rejects a declared extent strictly greater than the bound and
/// accepts at or below it, where only the runtime can decide which side of the
/// minimum wins. A rank spread is deliberately excluded: it stands for a run of
/// dimensions rather than one extent, so no minimum relation holds.
///
/// Callers that need the bound read it through [`diagonal_result_bound`], which
/// maps the pair position back onto the operand's axes. Both readers go through
/// this one function so the eager and deferred shape routes cannot drift.
pub(super) fn select_diagonal_extent(a: &Dim, b: &Dim, subst: &Subst) -> DiagonalExtent {
    let left = subst.observe_dim(a);
    let right = subst.observe_dim(b);
    let same_name = left.name().is_some() && left.name() == right.name();
    let dim = match (left.known_extent(), right.known_extent()) {
        // Known unequal extents must not be hidden by equal labels.
        (Some(lhs), Some(rhs)) if lhs != rhs || !same_name => Dim::Lit(lhs.min(rhs)),
        // Retain the selected identity, not a bare Name before result guards.
        _ if same_name => a.clone(),
        _ => Dim::Wildcard,
    };
    let bound = match (left.known_extent(), right.known_extent()) {
        (Some(extent), None) if right.rank().is_none() => Some(DiagonalBound {
            selected: 0,
            extent,
        }),
        (None, Some(extent)) if left.rank().is_none() => Some(DiagonalBound {
            selected: 1,
            extent,
        }),
        _ => None,
    };
    DiagonalExtent { dim, bound }
}

/// The upper bound a `diagonal` call imposes on its declared result, expressed
/// in the coordinates a caller can compare against a declared type.
///
/// Returns the result-type axis the bound applies to, the operand axis the
/// literal came from (for the diagnostic), and the bound itself. The result
/// axis is not the source axis: `diagonal` removes `axis2`, so a retained
/// `axis1` after it shifts down by one.
pub(super) fn diagonal_result_bound(
    tensor_ty: &Type,
    axis1: usize,
    axis2: usize,
    subst: &Subst,
) -> Option<(usize, usize, i64)> {
    let Type::Tensor(dims, _) = tensor_ty else {
        return None;
    };
    if axis1 >= dims.len() || axis2 >= dims.len() || axis1 == axis2 {
        return None;
    }
    let bound = select_diagonal_extent(&dims[axis1], &dims[axis2], subst).bound?;
    let source_axis = if bound.selected == 0 { axis1 } else { axis2 };
    let result_axis = if axis1 < axis2 { axis1 } else { axis1 - 1 };
    Some((result_axis, source_axis, bound.extent))
}

pub(super) fn infer_diagonal_result_type(
    tensor_ty: &Type,
    axis1: usize,
    axis2: usize,
    subst: &Subst,
) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = tensor_ty else {
        return Err(format!("diagonal expects tensor input, got {tensor_ty}"));
    };
    if axis1 >= dims.len() || axis2 >= dims.len() || axis1 == axis2 {
        return Err(format!(
            "diagonal expects distinct in-bounds axes, got {axis1} and {axis2} for rank {}",
            dims.len()
        ));
    }
    // chelis#1355 / [05-OP-33]: `diagonal` "replaces the retained first axis
    // extent with the smaller selected extent". Declare that minimum wherever it
    // is statically known; a wildcard admits every declared extent, which is how
    // a return type the runtime cannot produce was still type-checking.
    //
    // Two literal extents: the minimum is the smaller value, and the former
    // equal-literals case is subsumed by `min`.
    //
    // Two occurrences of one named extent: they denote a single runtime value
    // (`spec/04-type-system.md` §4.1, two `d-name` unify only when equal), so the
    // minimum is that name. `tensor[hidden, hidden]` diagonalises to
    // `tensor[hidden]`, not to a wildcard that also admits `tensor[width]` and
    // `tensor[3]`.
    //
    // Everything else keeps the wildcard - distinct names, a mixed
    // literal/symbolic pair, a dimension variable, a rank spread - because there
    // the minimum genuinely is not known at check time.
    let diag_dim = select_diagonal_extent(&dims[axis1], &dims[axis2], subst).dim;
    let mut out_dims = Vec::with_capacity(dims.len() - 1);
    for (index, dim) in dims.iter().enumerate() {
        if index == axis1 {
            out_dims.push(diag_dim.clone());
        } else if index != axis2 {
            out_dims.push(dim.clone());
        }
    }
    Ok(Type::Tensor(out_dims, precision.clone()))
}

pub(super) fn macro_source(expr: &deep::Expr) -> Option<String> {
    // chelis#1107 amendment: carrier-preserving read.
    let (_, meta, _) = stamped_parts(expr)?;
    Some(chelis_deep::printer::print_macro_source(meta.source()?))
}

pub(super) struct AliasExpansionSession<'a> {
    adt_reg: &'a AdtRegistry,
    vg: &'a mut VarGen,
    cache: Vec<(String, Vec<NominalArg>, Type)>,
}

impl<'a> AliasExpansionSession<'a> {
    pub(super) fn new(adt_reg: &'a AdtRegistry, vg: &'a mut VarGen) -> Self {
        Self {
            adt_reg,
            vg,
            cache: Vec::new(),
        }
    }

    pub(super) fn resolve(&mut self, ty: &Type) -> Type {
        let mut seen = UnordSet::new();
        self.resolve_inner(ty, &mut seen)
    }

    fn resolve_inner(&mut self, ty: &Type, seen: &mut UnordSet<String>) -> Type {
        match ty {
            Type::Adt(name, args) => {
                let resolved_args: Vec<Type> = args
                    .iter()
                    .map(|arg| self.resolve_inner(arg, seen))
                    .collect();
                let nominal_args = resolved_args
                    .iter()
                    .cloned()
                    .map(NominalArg::Type)
                    .collect::<Vec<_>>();

                if seen.contains(name) {
                    return Type::Adt(name.clone(), resolved_args);
                }
                if let Some((_, _, cached)) =
                    self.cache.iter().find(|(cached_name, cached_args, _)| {
                        cached_name == name && cached_args == &nominal_args
                    })
                {
                    return cached.clone();
                }

                let Some(alias) = self.adt_reg.resolve_alias(name) else {
                    return Type::Adt(name.clone(), resolved_args);
                };
                let Some((type_subst, dim_subst)) =
                    crate::adt::nominal_substitutions(&alias.param_args, &nominal_args)
                else {
                    return Type::Adt(name.clone(), resolved_args);
                };
                let substituted =
                    crate::adt::substitute_nominal_type(&alias.body, &type_subst, &dim_subst);

                // Alias-body variables that did not come from a supplied type
                // argument are quantified by the alias declaration. Freshen
                // them at each distinct alias application. The cache shares
                // one expansion for repeated occurrences with identical
                // arguments inside a signature, preserving named-parameter
                // equality without leaking registration-time IDs across
                // signatures or inference levels.
                let protected_tvars: UnordSet<_> = resolved_args
                    .iter()
                    .flat_map(crate::env::free_tvars)
                    .collect();
                let protected_dvars: UnordSet<_> = resolved_args
                    .iter()
                    .flat_map(crate::env::free_dvars)
                    .collect();
                let protected_rvars: UnordSet<_> = resolved_args
                    .iter()
                    .flat_map(crate::env::free_rvars)
                    .collect();
                let mut renaming = Subst::new();
                for var in crate::env::free_tvars(&substituted) {
                    if !protected_tvars.contains(&var) {
                        renaming
                            .insert_type(var, self.vg.fresh_type())
                            .expect("fresh alias-body type renaming is valid");
                    }
                }
                for var in crate::env::free_dvars(&substituted) {
                    if !protected_dvars.contains(&var) {
                        renaming.insert_dim(var, self.vg.fresh_dim());
                    }
                }
                for var in crate::env::free_rvars(&substituted) {
                    if !protected_rvars.contains(&var) {
                        renaming.insert_rank(var, vec![Dim::Rank(self.vg.fresh_rvar())]);
                    }
                }

                let expanded = renaming.apply(&substituted);
                seen.insert(name.clone());
                let resolved = self.resolve_inner(&expanded, seen);
                seen.remove(name);
                self.cache
                    .push((name.clone(), nominal_args, resolved.clone()));
                resolved
            }
            Type::KindedAdt(name, args) => {
                let resolved_args = args
                    .iter()
                    .map(|argument| match argument {
                        NominalArg::Type(ty) => NominalArg::Type(self.resolve_inner(ty, seen)),
                        NominalArg::Dimension(dim) => NominalArg::Dimension(dim.clone()),
                    })
                    .collect::<Vec<_>>();
                if seen.contains(name) {
                    return Type::KindedAdt(name.clone(), resolved_args);
                }
                if let Some((_, _, cached)) =
                    self.cache.iter().find(|(cached_name, cached_args, _)| {
                        cached_name == name && cached_args == &resolved_args
                    })
                {
                    return cached.clone();
                }
                let Some(alias) = self.adt_reg.resolve_alias(name) else {
                    return Type::KindedAdt(name.clone(), resolved_args);
                };
                let Some((type_subst, dim_subst)) =
                    crate::adt::nominal_substitutions(&alias.param_args, &resolved_args)
                else {
                    return Type::KindedAdt(name.clone(), resolved_args);
                };
                let substituted =
                    crate::adt::substitute_nominal_type(&alias.body, &type_subst, &dim_subst);

                let protected_tvars = resolved_args
                    .iter()
                    .filter_map(NominalArg::as_type)
                    .flat_map(crate::env::free_tvars)
                    .collect::<UnordSet<_>>();
                let protected_dvars = resolved_args
                    .iter()
                    .flat_map(|argument| match argument {
                        NominalArg::Type(ty) => crate::env::free_dvars(ty),
                        NominalArg::Dimension(Dim::Var(var)) => vec![*var],
                        NominalArg::Dimension(_) => Vec::new(),
                    })
                    .collect::<UnordSet<_>>();
                let protected_rvars = resolved_args
                    .iter()
                    .flat_map(|argument| match argument {
                        NominalArg::Type(ty) => crate::env::free_rvars(ty),
                        NominalArg::Dimension(Dim::Rank(var)) => vec![*var],
                        NominalArg::Dimension(_) => Vec::new(),
                    })
                    .collect::<UnordSet<_>>();
                let mut renaming = Subst::new();
                for var in crate::env::free_tvars(&substituted) {
                    if !protected_tvars.contains(&var) {
                        renaming
                            .insert_type(var, self.vg.fresh_type())
                            .expect("fresh alias-body type renaming is valid");
                    }
                }
                for var in crate::env::free_dvars(&substituted) {
                    if !protected_dvars.contains(&var) {
                        renaming.insert_dim(var, self.vg.fresh_dim());
                    }
                }
                for var in crate::env::free_rvars(&substituted) {
                    if !protected_rvars.contains(&var) {
                        renaming.insert_rank(var, vec![Dim::Rank(self.vg.fresh_rvar())]);
                    }
                }

                let expanded = renaming.apply(&substituted);
                seen.insert(name.clone());
                let resolved = self.resolve_inner(&expanded, seen);
                seen.remove(name);
                self.cache
                    .push((name.clone(), resolved_args, resolved.clone()));
                resolved
            }
            Type::Fn(args, ret) => Type::Fn(
                args.iter()
                    .map(|arg| self.resolve_inner(arg, seen))
                    .collect(),
                Box::new(self.resolve_inner(ret, seen)),
            ),
            Type::Ref(inner) => Type::Ref(Box::new(self.resolve_inner(inner, seen))),
            Type::Tuple(items) => Type::Tuple(
                items
                    .iter()
                    .map(|item| self.resolve_inner(item, seen))
                    .collect(),
            ),
            Type::Tensor(_, _) | Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error(_) => {
                ty.clone()
            }
        }
    }
}

pub(super) fn resolve_type_aliases(ty: &Type, adt_reg: &AdtRegistry, vg: &mut VarGen) -> Type {
    AliasExpansionSession::new(adt_reg, vg).resolve(ty)
}

/// [`report`], lifted into the `Option<Type>` early-return channel that the
/// post-unification application checks use.
///
/// Those checks answer "did this callee reject its arguments?", so they return
/// `Option<Type>`: `Some(ty)` means rejected, with `ty` recorded as the call's
/// type, and `None` means the check had nothing to say and the caller
/// continues to the next one.
pub(super) fn reject(errors: &mut DiagnosticSink<'_>, error: CheckError) -> Option<Type> {
    Some(report(errors, error))
}

// ── Tier-2 rank-polymorphism Body Discipline ─────────────────────

/// True if any tensor inside `ty` carries a `Dim::Rank` (a rank variable).
pub(super) fn type_contains_rank(ty: &Type) -> bool {
    match ty {
        Type::Tensor(dims, _) => dims.iter().any(|d| matches!(d, Dim::Rank(_))),
        Type::Fn(args, ret) => args.iter().any(type_contains_rank) || type_contains_rank(ret),
        Type::Ref(inner) => type_contains_rank(inner),
        Type::Adt(_, args) => args.iter().any(type_contains_rank),
        Type::KindedAdt(_, args) => args.iter().any(|argument| match argument {
            NominalArg::Type(ty) => type_contains_rank(ty),
            NominalArg::Dimension(Dim::Rank(_)) => true,
            NominalArg::Dimension(_) => false,
        }),
        Type::Tuple(ts) => ts.iter().any(type_contains_rank),
        Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error(_) => false,
    }
}

/// Exact [05-OP-35] stdlib definitions whose bodies intentionally cross the
/// current procedural rank/generic-intrinsic checker boundary.  These names
/// are linker-reserved and cannot be forged by entry source. Their exact
/// signatures are checked by the declaration pass and their authored graphs
/// are locked separately by the stdlib surface oracle.
fn is_exact_op35_wrapper(name: &str) -> bool {
    matches!(
        name,
        "pkg__chelis__std__Std__Sort__sort"
            | "pkg__chelis__std__Std__Tensor__Construct__arange"
            | "pkg__chelis__std__Std__Tensor__Construct__arange_values"
            | "pkg__chelis__std__Std__Tensor__Construct__linspace"
            | "pkg__chelis__std__Std__Tensor__Construct__linspace_values"
            | "pkg__chelis__std__Std__Tensor__Construct__shape_with_axis"
            | "pkg__chelis__std__Std__Tensor__Construct__shape_without_axis"
            | "pkg__chelis__std__Std__Tensor__Construct__squeeze"
            | "pkg__chelis__std__Std__Tensor__Construct__stack"
            | "pkg__chelis__std__Std__Tensor__Construct__unsqueeze"
            | "pkg__chelis__std__Std__Tensor__Mask__where_indices"
            | "pkg__chelis__std__Std__Test__assert_close_tensor"
            | "pkg__chelis__std__Std__Test__assert_eq_tensor"
            | "pkg__chelis__std__Std__Test__assert_shape"
            | "pkg__chelis__std__Std__Test__shape_matches"
    )
}

/// Install dependency contracts for exact standard-library tensor wrappers
/// whose shape relations are not yet expressible through ordinary inference.
fn install_exact_op35_dependency_contracts(
    name: &str,
    declared_ty: Option<&Type>,
    env: &mut Env,
    vg: &mut VarGen,
) {
    if is_exact_op35_wrapper(name) {
        env.set_exact_stdlib_expected_result(declared_ty.and_then(|ty| match ty {
            Type::Fn(_, result) => Some((**result).clone()),
            _ => None,
        }));
    }
    let shape_helper = match name {
        "pkg__chelis__std__Std__Tensor__Construct__squeeze" => {
            Some("pkg__chelis__std__Std__Tensor__Construct__shape_without_axis")
        }
        "pkg__chelis__std__Std__Tensor__Construct__unsqueeze" => {
            Some("pkg__chelis__std__Std__Tensor__Construct__shape_with_axis")
        }
        _ => None,
    };
    if let Some(helper) = shape_helper {
        let tensor = vg.fresh_tvar();
        env.bind(
            helper.to_string(),
            Scheme {
                result_origin: None,
                constraints: vec![],
                tvars: vec![tensor],
                tvar_restrictions: vec![],
                dvars: vec![],
                rvars: vec![],
                body: Type::Fn(
                    vec![
                        Type::Ref(Box::new(Type::Var(tensor))),
                        Type::Prim(Prim::Int32),
                        Type::Prim(Prim::Int32),
                        Type::Prim(Prim::Int32),
                        Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]),
                    ],
                    Box::new(Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)])),
                ),
            },
        );
    }
}

/// Extract the callee name from an `app`'s first child when it is `(var {} name)`.
pub(super) fn app_var_name(callee: &deep::Expr) -> Option<&str> {
    // chelis#1107 amendment: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(callee)?;
    if tag != DeepTag::Var {
        return None;
    }
    kids.first().and_then(symbol_name)
}

/// Names of every top-level `def` in the program (after module flattening),
/// so the Body-Discipline check can reject a call that resolves to a user
/// function shadowing an Identity builtin name (chelis#258 §4.2).
pub(super) fn collect_user_def_names(items: &[&deep::Expr]) -> UnordSet<String> {
    let mut out = UnordSet::new();
    for expr in items {
        if let Some((DeepTag::Def, _, kids)) = stamped_parts(expr)
            && let Some(name) = kids.first().and_then(symbol_name)
        {
            out.insert(name.to_string());
        }
    }
    out
}

/// Walk a rank-polymorphic def's body and reject any call whose output shape is
/// not *name-trackable* at symbolic rank. Admitted: shape-identity (elementwise)
/// builtins and named-axis reductions (the procedural arm verifies those drop a
/// named axis and carry the rest through), and checked ordered-prefix key
/// derivations. Rejected: positional shape-rewriting
/// builtins (`permute`/`reshape`/`matmul`/…), and any user/non-builtin/computed
/// callee not proven rank-safe — against a spread `..r` there are no named axes
/// left to catch an untracked transposition/reshape, so admitting one would
/// silently break §4.2 transposition safety (spec/design/rank_polymorphism.md
/// §Soundness Boundary).
pub(super) fn check_rank_body_discipline(
    def_name: &str,
    expr: &deep::Expr,
    user_def_names: &UnordSet<String>,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("check_rank_body_discipline", expr);
    // Only decoded nodes carry calls; structural lists, unknown forms,
    // metadata and atoms hold none this discipline reads.
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        // Function-taking transforms apply a *referenced* user function across
        // the opaque rank. That callee is not inlined here, so its body can
        // transpose/reshape undetected — reject outright (spec §4.2).
        // `jit`/`realize`/`cast`/`copy` wrap an *inline* expression that the
        // recursion below still checks, so they are not rejected here.
        t @ (DeepTag::Grad | DeepTag::Vmap) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "rank-polymorphic def `{def_name}` may not use `{}` in its body: it applies \
                     a function across the opaque rank `..r`, whose body cannot be proven \
                     shape-identity (spec/04-type-system.md \u{00a7}4.2).",
                    t.as_str()
                ),
                vec![],
            ));
        }
        DeepTag::App => match kids.first().and_then(app_var_name) {
            // A user-defined `def` of this name — possibly SHADOWING an
            // Identity builtin (`def relu(x) = permute(x,1,0)`). The call
            // resolves to the user def, whose body is not proven rank-safe, so
            // it must be rejected before the builtin-name classification below.
            Some(name) if user_def_names.contains(name) => {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "rank-polymorphic def `{def_name}` may not call user-defined `{name}`: \
                         only shape-identity builtins are proven rank-safe in a `..r` body, and a \
                         user `def` (even one shadowing a builtin name) is not (spec/04-type-system.md \
                         \u{00a7}4.2)."
                    ),
                    vec![],
                ));
            }
            // OrderedPrefix relations retain every operand axis and may only
            // append trailing axes, including in each tuple result component.
            // Identity (elementwise) or NameTracked (named-axis reduction /
            // named-axis expand) builtin — admissible. For a NameTracked op
            // the procedural inference arm (`check_reduction_signature` /
            // `check_expand_signature`) is the real gate: it verifies the
            // addressed axis is name-anchored against the operand and
            // computes a symbolic output that carries the surviving named axes
            // through, rejecting a positional index at symbolic rank or a
            // non-existent/ambiguous/duplicate axis name. So no untracked
            // transposition can slip past.
            Some(name)
                if builtins::BUILTIN_NAMES.contains(&name)
                    && matches!(
                        builtins::shape_class(name),
                        builtins::ShapeClass::Identity
                            | builtins::ShapeClass::NameTracked
                            | builtins::ShapeClass::OrderedPrefix
                    ) => {}
            // A named builtin that rewrites shape positionally (not name-tracked).
            Some(name) if builtins::BUILTIN_NAMES.contains(&name) => {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "rank-polymorphic def `{def_name}` may not call shape-rewriting builtin \
                         `{name}`: it is not name-trackable at symbolic rank, so against a spread \
                         `..r` there are no named axes left to catch a transposition or reshape \
                         (spec/04-type-system.md \u{00a7}4.2). A `..r` body may call shape-identity \
                         (elementwise) operations, named-axis reductions, and the named-axis \
                         `expand` and `insert` forms, and ordered-prefix key derivations only."
                    ),
                    vec![format!(
                        "remove the `{name}` call from the rank-polymorphic body, or use \
                         concrete-rank `def`s instead of a `..r` signature"
                    )],
                ));
            }
            // A named user-defined function — not proven rank-safe.
            Some(name) => {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "rank-polymorphic def `{def_name}` may not call `{name}`: only \
                         shape-identity builtins are proven rank-safe in a `..r` body \
                         (spec/04-type-system.md \u{00a7}4.2). Calling a user-defined function \
                         from a rank-polymorphic body is not supported."
                    ),
                    vec![],
                ));
            }
            // A computed callee (a transform result like `grad(f)(x)`, a
            // first-class function value, or an applied lambda's non-inline
            // form): cannot be proven rank-safe.
            None => {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "rank-polymorphic def `{def_name}` may not apply a computed or \
                         non-builtin callee in a `..r` body: only shape-identity builtins are \
                         proven rank-safe (spec/04-type-system.md \u{00a7}4.2)."
                    ),
                    vec![],
                ));
            }
        },
        _ => {}
    }
    // Recurse so nested calls (in let/if/match/lambda bodies, args) are checked.
    for child in kids {
        check_rank_body_discipline(def_name, child, user_def_names, errors);
    }
}

// ── Top-level inference (second pass) ────────────────────────────

/// A recursive group member's inferred type, held until the whole group is
/// inferred and generalized together.
pub(super) struct DeferredRecursiveBinding {
    pub(super) name: String,
    pub(super) ty: Type,
    pub(super) owned_contracts: Vec<crate::unify::CollectionContractId>,
    /// The authored binders of the member's declaration, so [04-LIN-10]'s
    /// diagnostic can name the parameter a key reached.
    pub(super) binder_names: UnordMap<TypeVar, String>,
}

/// Generalize a recursive group member and name its generic parameters.
pub(super) fn generalize_deferred_recursive_binding(
    binding: DeferredRecursiveBinding,
    env: &Env,
    subst: &Subst,
    product: &mut InferenceProduct,
) -> (String, Scheme) {
    let mut scheme =
        env.generalize_with_collection_contracts(&binding.ty, subst, &binding.owned_contracts);
    if let Some(raw) = product.group_result_origins.remove(&binding.name) {
        if raw.result_origin.is_some() {
            scheme.constraints = raw.constraints;
        }
        scheme.result_origin = raw.result_origin;
        // The origin owns restrictions on its raw representatives. The
        // freshly generalized public scheme already carries the restrictions
        // on its solved signature; mixing the ledgers exports hidden IDs.
    }
    subst.name_generic_parameters(&scheme, &binding.name, &binding.binder_names);
    (binding.name, scheme)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_top_level(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    prebound_type_failure: Option<&ErrorWitness>,
    recursive_expected: Option<recursion::RecursiveExpected<'_>>,
    defer_recursive_binding: bool,
    user_def_names: &UnordSet<String>,
    declared_signatures: &UnordMap<String, DeclaredSigMetadata>,
    declaration_diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
) -> Option<DeferredRecursiveBinding> {
    let Some((tag, declaration_meta, kids)) = stamped_parts(expr) else {
        // chelis#858 / [04-TOT-1]: a top-level list with no decoded tag
        // used to be silently skipped here, so a program like
        // `((var {} f) (var {} x))` was never type-checked while the
        // fitness clean path manufactured a vacuous 1.0. The disposition
        // is a loud rejection; the raw-string boundary names an unknown
        // symbol head when there is one.
        let named = match expr {
            deep::Expr::UnknownForm(data) => data.head.as_str(),
            _ => "<untagged-list>",
        };
        report(
            errors,
            CheckError::new(
                CheckErrorKind::UnknownForm,
                format!(
                    "top-level Deep list `{named}` is not a canonical (tag {{}} ...)                      declaration or expression (chelis#858; [04-TOT-1])"
                ),
                vec![],
            ),
        );
        return None;
    };

    // Skip deftype/defsig/typealias (already processed in first pass)
    if tag == DeepTag::Deftype || tag == DeepTag::Defsig || tag == DeepTag::Typealias {
        return None;
    }

    if tag == DeepTag::Def && kids.len() >= 2 {
        let name = symbol_name(&kids[0])?.to_string();
        // Recursive SCCs own one surrounding level in their driver. An
        // ordinary declaration owns this per-definition level and closes it
        // before its inferred scheme is generalized.
        let ordinary_level = (!defer_recursive_binding).then(|| subst.enter_level(vg));
        let collection_contract_mark = subst.collection_contract_mark();
        let recursive_expected =
            recursion::PreparedRecursiveExpected::prepare(recursive_expected, env, vg, subst);
        let provisional_recursive_type = recursive_expected.expected_type();

        // Save declared type from defsig BEFORE inferring (it may get overwritten)
        // spec/04 §3.1.1: when this def is a member of the active recursive
        // binding group, remember which fresh tvars its own body was
        // instantiated at, so in-group recursive calls can be validated
        // against the caller's own instantiation.
        // chelis#260: the names of this signature's declared dim parameters,
        // keyed by the FRESH variables the instantiation below mints. Empty
        // when the signature was never recorded, in which case the collapse
        // diagnostic falls back to the internal id.
        // [04-INF-6]: compose this declaration's authored type, dimension, and
        // rank identities. Outer-signature occurrences come from one scheme
        // instantiation; roles absent there are completed from the structural
        // binder list. An inference hole is never a member. The deferred-borrow
        // drain reports on the type projection long after this setup, so that
        // projection is parked on `Env` below.
        let binder_names = declared_signatures
            .get(&name)
            .map(|metadata| &metadata.binders);
        let rejected_signature = env.rejected_signature(&name).cloned();
        let replaying_authored_signature =
            recursive_expected.is_published() && declared_signatures.contains_key(&name);
        let declared_scheme = if let Some(rejected) = &rejected_signature {
            Some(&rejected.scheme)
        } else if provisional_recursive_type.is_none() {
            env.lookup(&name)
        } else {
            None
        };
        let recursion::DeclaredMemberSetup {
            ty: declared_ty,
            binder_identities: declared_binder_identities,
            caller_guard: _recursion_caller_guard,
        } = recursive_expected.prepare_declared_member(
            recursion::DeclaredMemberRequest::new(
                &name,
                binder_names,
                declared_scheme,
                replaying_authored_signature,
            ),
            env,
            vg,
            subst,
        );
        // chelis#2590: a member whose header omits a type is inferred against
        // its own instance of its provisional scheme, and its component's
        // completion decides each in-group reference against this one.
        if env.is_holed_group_member(&name)
            && let Some(ty) = &declared_ty
        {
            product.record_group_member_type(&name, ty.clone());
        }
        let declared_dim_names = declared_binder_identities.dim_names();
        let declared_type_names = declared_binder_identities.type_names();
        let declared_rank_names = declared_binder_identities.rank_names();
        // A declaration's signature owns the only named binders legal in its
        // nested source annotations. Infer against a lexical clone so the
        // scope follows nested env clones but cannot leak to the next `def`
        // or into the reusable top-level environment.
        // chelis#260 Site 2: park the composed map on the OUTER env, which
        // is what the per-def deferred-borrow drain reads. `body_env` below
        // is a clone, so parking there would not survive to the drain.
        // chelis#260 Site 2 parks the map here and chelis#1486 checks
        // rigidity from it after body inference, so the two consumers
        // each take their own copy. The rigidity check keeps the value
        // THIS declaration computed rather than reading the parked one
        // back, which body inference could have replaced.
        env.set_active_declared_type_names(declared_type_names.clone());
        let declared_dtype_bounds: UnordMap<TypeVar, Option<TypeVarRestriction>> =
            declared_type_names
                .to_sorted()
                .into_iter()
                .map(|(variable, _)| (*variable, env.declared_binder_bound(*variable, subst)))
                .collect();
        env.set_active_declared_type_bounds(declared_dtype_bounds.clone());
        let mut body_env = env.clone();
        body_env.set_type_resolution_scope(
            binder_names.map(|_| &declared_binder_identities),
            declaration_diagnostic_owner,
        );
        install_exact_op35_dependency_contracts(&name, declared_ty.as_ref(), &mut body_env, vg);

        let body_diagnostic_checkpoint = errors.checkpoint();
        if let Some(Type::Fn(params, _)) = &declared_ty {
            subst.protect_dimensions(params.iter().flat_map(crate::env::free_dvars));
        }
        // WS-A7: when the body is a bare-arg `(fn (params) body)` and the
        // declared signature gives concrete param types, seed the body's
        // params with the declared types BEFORE inferring the body. Without
        // this seeding, bare params get fresh, unconstrained type variables.
        // For callee schemes that share a tvar between an `&T` param and a
        // non-borrow return position (e.g. `add: (&t, &t) -> t`), the
        // call-site `auto_borrow_call_arg_types` wraps the actual param
        // tvar in `Ref(...)` and unifies it with the formal `Ref(α)`,
        // collapsing the param-side and return-side of the callee into
        // the same equivalence class. The post-body sig-unify then drives
        // the return position to `Ref(t)` instead of the declared `t`,
        // surfacing as a mismatch between inferred `(&t, &t) -> &t`
        // and declared `(&t, &t) -> t`.
        // Annotated params do not hit this because their concrete type
        // (`&tensor[..]`) flows through the call site directly. Seeding
        // bare params with the declared type here makes the bare-arg path
        // behave the same as the annotated path. See
        // `crates/chelis-cli/tests/bareref_return_inference.rs`.
        let body_ty = if let Some(witness) = prebound_type_failure {
            propagate(witness)
        } else if let Some(decl_ty) = &declared_ty {
            infer_declared_def_body(
                &kids[1],
                decl_ty,
                declaration_meta,
                declared_signatures.get(&name),
                &mut body_env,
                vg,
                subst,
                adt_reg,
                errors,
                product,
                declaration_diagnostic_owner,
            )
        } else {
            infer_expr_with_declaration_diagnostic_owner(
                &kids[1],
                &mut body_env,
                vg,
                subst,
                adt_reg,
                errors,
                product,
                declaration_diagnostic_owner,
            )
        };
        // Did the body's inference report any UnboundVariable diagnostic?
        // We use this to discriminate WS-A5 RT-3a F1's masked-by-Error
        // case (where Error is a downstream consequence of a reportable
        // cause that the user can act on) from cascades where Error
        // emerges from an internal type-checker limitation that has no
        // matching upstream diagnostic. Without this discriminator the
        // F1 detector double-reports on legitimate code that exercises
        // type-checker gaps (record construction, region effects) which
        // the permissive unify rule was implicitly tolerating.
        let body_has_unbound_diagnostic = errors
            .iter_since(body_diagnostic_checkpoint)
            .any(|e| matches!(e.kind, CheckErrorKind::UnboundVariable { .. }));

        // Enforce defsig: body must match declared signature.
        //
        // After unify succeeds, the body's inferred type may still carry
        // `Dim::Wildcard` slots (e.g. from `pad_sequences_to`, `concat`,
        // `to_tensor`) because `unify_dim` treats Wildcard as a permissive
        // matches-anything sentinel. Generalizing the raw body would expose
        // those wildcards to callers, who would then silently accept any
        // concrete dim. Narrow the body's Wildcards against the declared
        // template so the scheme registered for callers reflects the
        // declared concrete shape (#39).
        //
        // Implicit-copy fan-out v3 Shape A: if the initial unify fails
        // and the body's tail position resolves to one borrowed
        // parameter (after walking `let`/`if`/`match` wrappers via
        // `descend_to_tail_parameter`) whose declared return is owned
        // `T` while the body's inferred type returns `Ref(T)`, retry the
        // unify against the declared return relaxed into `Ref(T)`.
        // This mirrors `auto_borrow_call_arg_types`'s owned-to-borrow
        // coercion at argument positions: the caller already arranged
        // the borrow lifetime via the param itself, and the descent
        // confirms every reachable return path returns the same
        // parameter's own value, by binding identity rather than by
        // source spelling.  Heterogeneous returns and bodies whose tail
        // is an `app` or other non-var expression still fail with the
        // existing TypeMismatch.
        let mut binder_rigidity = None;
        let scheme_body = if is_exact_op35_wrapper(&name) {
            // These package-reserved wrappers have exact signatures whose
            // generic or runtime-axis relations ordinary inference cannot yet
            // establish. Inference still walks each complete body and owns
            // every child stamp, while the exact manifest signature remains
            // authoritative at this compiler-owned boundary. Preserve all
            // actionable diagnostics; only the currently unprovable type and
            // shape relations are deferred to their owning contracts. The
            // stdlib surface oracle locks each accepted graph structurally.
            errors.retain_since(body_diagnostic_checkpoint, |error| {
                !matches!(
                    error.kind,
                    CheckErrorKind::TypeMismatch
                        | CheckErrorKind::DimensionMismatch
                        | CheckErrorKind::CastNonTensor
                )
            });
            let declared_ty =
                declared_ty.expect("every exact OP-35 wrapper has a declared signature");
            if let Err(error) = subst.project_tvar_restrictions(&body_ty, &declared_ty) {
                errors.push(error.into());
            }
            declared_ty
        } else if let Some(decl_ty) = declared_ty {
            product.replay_ready_shape_checks(vg, subst, adt_reg, errors);
            // Omitted results and prior checked sweep signatures are inference
            // identities. Only authored annotations add result-only equations.
            let inferred_result = recursive_expected.is_published()
                || matches!(&decl_ty, Type::Fn(_, result)
                if matches!(result.as_ref(), Type::Var(var) if !declared_type_names.contains_key(var)));
            let unify_result = if !inferred_result
                && product.defer_result_type_constraint(&body_ty, &decl_ty, subst)
            {
                Ok(())
            } else {
                unify(&body_ty, &decl_ty, subst)
            };
            let resolved_body = subst.apply(&body_ty);
            let resolved_decl = subst.apply(&decl_ty);
            // Declared-dim rigidity check (TypeCheck-FreeDimVarUnification-F1
            // Path B), with its type and rank twins ([04-INF-6]). After the
            // post-body sig-unify above, two distinct declared binders must not
            // have collapsed into one another, and none may have been pinned to
            // a concrete literal or type. The collapse for an annotated-param
            // body happens in the sig-unify itself, not during body inference:
            // the body's tail returns the wrong declared dim (`def g[n, m](x:
            // tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = y`), and
            // the structural relaxed-retry guard does not see it because the
            // initial unify already succeeded by collapsing `n` and `m`.
            //
            // The contract is decided by `close_declaration`, not here: the
            // declaration boundary still replays suspended calls after this
            // point, and a replay can pin or narrow a binder (chelis#2537).
            binder_rigidity = Some((
                decl_ty.clone(),
                declared_dim_names.clone(),
                declared_rank_names.clone(),
            ));
            // Tier-2 rank-polymorphism Body Discipline
            // (spec/design/rank_polymorphism.md §Soundness Boundary, spec §4.2):
            // a def whose signature mentions a rank variable `..r` may call only
            // shape-identity (elementwise) builtins. Against an opaque rank there
            // are no named axes left to catch a transposition/reshape, so any
            // shape-rewriting op (or an unproven user call) is rejected here.
            if type_contains_rank(&decl_ty)
                && !is_exact_op35_wrapper(&name)
                && let Some((_, body_expr)) = extract_fn_params_and_body(&kids[1])
            {
                check_rank_body_discipline(&name, &body_expr, user_def_names, errors);
            }
            // chelis#272 list-uniformity check. A list literal of
            // tensors with *mismatched concrete* element axes joins to a
            // `Wildcard` along the differing axis (the deliberate #218
            // bare-`concat` ergonomic). That wildcard is a defensible
            // "I don't know the shape" result for an unannotated bare
            // list, but it must NOT silently satisfy a declared element
            // type that names a rigid/named dimension — `List[tensor[k]]`
            // promises every element has the *same* length `k`. A body
            // like `def make[k](a: tensor[2], b: tensor[3])
            //   -> List[tensor[k]] = [a, b]` produces
            // `List<tensor[Wildcard]>`; the wildcard unifies permissively
            // with the rigid `k` and leaves it unbound, so neither the
            // pin-to-literal nor the distinct-collapse arm of
            // `check_declared_dvars_rigid` fires. Flag that mismatch here
            // by comparing the declared return's list-element dims
            // against the resolved body's. (The `[k, m]` variant is
            // already caught above: the tightened Cons-join now unifies
            // the two rigid dims, and `check_declared_dvars_rigid`
            // reports the collapse.)
            check_list_elem_rigid_dim_vs_wildcard(
                &decl_ty,
                &resolved_body,
                &declared_dim_names,
                errors,
            );
            // Implicit-copy fan-out v3 Shape A relaxed retry: if the
            // initial unify fails and the body's tail position resolves
            // to a `(var x)` reference whose declared return is owned
            // `T` while the body's inferred type returns `Ref(T)`,
            // retry the unify against the declared return relaxed
            // into `Ref(T)`.
            let initial_failed = unify_result.is_err();
            let recovered_by_relaxed_retry = if initial_failed {
                let relaxed_decl = shape_a_relaxed_return(&kids[1], &resolved_body, &resolved_decl);
                relaxed_decl
                    .as_ref()
                    .is_some_and(|relaxed| unify(&body_ty, relaxed, subst).is_ok())
            } else {
                false
            };
            // WS-A5 RT-3a F1: the permissive `(Error, _)` unify rule lets
            // a body whose return position collapses to `Type::Error`
            // (e.g. `def use_mix(x: tensor[3, i32]) -> tensor[3, f32]
            // = poly_id(nonexistent_function(x))`, where the outer call
            // early-exits at `Type::Error` so the body's `fn` type is
            // `Fn([tensor[3, i32]], Type::Error)`) silently satisfy a
            // concrete declared signature, masking the precision/shape
            // mismatch the user would otherwise see. Surface the masked
            // mismatch here when the body collapses to `Type::Error` at
            // the top level OR at the return position of a function
            // type, and the corresponding declared slot is concrete.
            // We deliberately do NOT recurse through tuples/ADTs/Fn
            // arg positions: those would re-fire on cascade patterns
            // (e.g. tuple destructuring on an unannotated parameter
            // produces `Tuple([Error, Error, ...])` from a localized
            // inference gap that has already been surfaced upstream
            // and should not double-report the def-level diagnostic).
            let body_return_collapsed = match (&resolved_body, &resolved_decl) {
                (Type::Error(_), decl) => !matches!(decl, Type::Error(_)),
                (Type::Fn(_, ret), Type::Fn(_, decl_ret))
                    if matches!(**ret, Type::Error(_)) && !matches!(**decl_ret, Type::Error(_)) =>
                {
                    true
                }
                _ => false,
            };
            let masked_by_error =
                unify_result.is_ok() && body_return_collapsed && body_has_unbound_diagnostic;
            let unrecovered_initial_failure = initial_failed && !recovered_by_relaxed_retry;
            if unrecovered_initial_failure || masked_by_error {
                // Declared-signature mismatches remain the established
                // TypeMismatch contract except when nominal-dimension
                // enforcement proves a concrete extent disagreement.  The
                // latter is the one structurally distinct case introduced by
                // [04-ADT-4].
                let mismatch_kind = match unify_result.as_ref().err().map(|error| &error.kind) {
                    Some(TypeErrorKind::DimensionMismatch) => CheckErrorKind::DimensionMismatch,
                    Some(TypeErrorKind::DtypeFamilyMismatch) => CheckErrorKind::PrecisionMismatch,
                    _ => CheckErrorKind::TypeMismatch,
                };
                // A §5.7.1 widening (an i8 or i16 `sum`-family result is
                // i32) gets its repair note here rather than leaving the
                // reader to find the rule behind a whole-signature mismatch.
                let extra = if let Some(error) = unify_result
                    .as_ref()
                    .err()
                    .filter(|error| matches!(error.kind, TypeErrorKind::DtypeFamilyMismatch))
                {
                    format!(": {}", error.message)
                } else {
                    sum_result_widening_note(
                        sum_family_tail_op(&kids[1]),
                        &resolved_body,
                        &resolved_decl,
                    )
                    .map(|note| format!("; {note}"))
                    .unwrap_or_default()
                };
                let error = CheckError::with_types(
                    mismatch_kind,
                    format!(
                        "def '{name}' body doesn't match declared signature: \
                         expected `{resolved_decl}`, got `{resolved_body}`{extra}"
                    ),
                    resolved_decl.to_string(),
                    resolved_body.to_string(),
                    vec![],
                );
                let location = TypeDiagnosticLocation::from_expr(&kids[1])
                    .or_else(|| TypeDiagnosticLocation::from_expr(expr));
                errors.push(match location {
                    Some(location) => location.attach(error),
                    None => error,
                });
            }
            // #39 wildcard narrowing. The narrow may now substitute a
            // declared `Dim::Var` or `Dim::Name` into a body
            // wildcard, but ONLY for binders in a parameter tensor
            // position (`param_bound_dims`). That keeps the
            // `const_col[n](spots: tensor[n, ..]) -> tensor[n, 1]` family's
            // return dim tied to its input (chelis#405 / WS-3 build ICE)
            // while leaving return-only / value-parameter dims as
            // wildcards (the RT-39+44 soundness boundary, commit 8067c9ce).
            let param_dims = param_bound_dims(&resolved_decl);
            narrow_wildcards_with(&resolved_body, &resolved_decl, &param_dims)
        } else if let Some(provisional) = provisional_recursive_type {
            if let Err(error) = unify(&body_ty, provisional, subst) {
                errors.push(error.into());
            }
            subst.apply(provisional)
        } else {
            body_ty
        };

        let binder_contract = AuthoredBinderContract::new(
            name.clone(),
            declared_type_names.clone(),
            declared_dtype_bounds,
        )
        .in_holed_group(env.is_holed_group_member(&name));
        product.record_authored_binder_contract(match binder_rigidity {
            Some((decl_ty, dim_names, rank_names)) => {
                binder_contract.with_rigidity(decl_ty, dim_names, rank_names)
            }
            None => binder_contract,
        });

        // The private frame constrains this body, but never replaces the
        // rejected declaration with a callable public signature.
        let scheme_body = match rejected_signature {
            Some(rejected) => propagate(&rejected.witness),
            None => scheme_body,
        };

        if let Some(level) = ordinary_level {
            subst.leave_level(level, vg);
        }

        product.record_bypass(expr, scheme_body.clone(), "top-level declaration inference");

        // The binding's value facts (a static extent such as
        // `zero_count = sub(0i64, 0i64)`, or a list literal's length), read
        // against the pre-binding scope. A deferred recursive member is a
        // function, which carries none.
        let facts = rhs_binding_facts(env, &kids[1]);
        if defer_recursive_binding {
            product
                .group_result_origins
                .insert(name.clone(), Scheme::mono(scheme_body.clone()));
            Some(DeferredRecursiveBinding {
                name,
                ty: scheme_body,
                owned_contracts: subst.collection_contract_ids_since(collection_contract_mark),
                binder_names: declared_type_names,
            })
        } else {
            let owned_contracts = subst.collection_contract_ids_since(collection_contract_mark);
            let scheme = product.generalize_result_origins(
                &scheme_body,
                env,
                subst,
                Some(&owned_contracts),
                errors,
            );
            subst.name_generic_parameters(&scheme, &name, &declared_type_names);
            env.bind_with_facts(name, scheme, facts);
            None
        }
    } else {
        // Any other top-level expression
        let _ = infer_expr(expr, env, vg, subst, adt_reg, errors, product);
        None
    }
}

// ── Core inference ───────────────────────────────────────────────
