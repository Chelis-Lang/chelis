//! Shared inference helpers.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

pub(super) fn get_tag(list: &deep::List) -> Option<DeepTag> {
    list.tag()
}

pub(super) fn children(list: &deep::List) -> &[deep::Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

/// Observe a decoded vocabulary node without changing its physical carrier.
///
/// `Expr::List` remains available only for legacy/programmatic callers during
/// the #1023 migration. Stamped compiler ingress uses `Expr::Node`; readers at
/// semantic boundaries must preserve that carrier instead of rebuilding a
/// `List` through `Node::to_list`.
pub(super) fn stamped_parts(expr: &deep::Expr) -> Option<(DeepTag, &deep::MetaMap, &[deep::Expr])> {
    match expr {
        deep::Expr::Node(node, _) => Some((node.tag(), node.meta(), node.children_slice())),
        deep::Expr::List(list, _) => {
            let tag = get_tag(list)?;
            let deep::Expr::Map(meta, _) = list.elements.get(1)? else {
                return None;
            };
            Some((tag, meta, children(list)))
        }
        _ => None,
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
    list: &deep::List,
    tag: &str,
    expected: &str,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    let found = children(list).len();
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

/// chelis#731 Phase 2 (spec/design/checker_totality.md §C3): report a
/// wrong-arity call of a shape-polymorphic builtin. Before the witness-token
/// migration these arity guards returned a SILENT `Type::Error` -- an
/// `einsum(a, b)` (two args) scored a perfect 1.0, the chelis#709 defect on
/// the builtin lane. The guard now pushes an `ArityMismatch` naming the
/// builtin and the expected/found counts, so the malformed call is rejected
/// with a diagnostic instead of exempted from checking.
/// Span-free sibling of [`report_builtin_arity`] for the `check_*_signature`
/// helpers, which receive `arg_tys`/`errors` but not the `(list)` node. Same
/// chelis#731 §C3 role: a wrong-arity signature-checked builtin used to return
/// a silent `Type::Error`; it now reports an `ArityMismatch`. `expected` is a
/// prose description ("at least 2 arguments", "3 or 4 arguments") since these
/// guards range over exact and minimum arities.
pub(super) fn report_builtin_arity_bare(
    errors: &mut DiagnosticSink<'_>,
    builtin: &str,
    expected: &str,
    got: usize,
) -> Type {
    report(
        errors,
        CheckError::new(
            CheckErrorKind::ArityMismatch,
            format!(
                "builtin `{builtin}` expects {expected}, got {got} argument(s) \
                 (chelis#731 [04-TOT-3])"
            ),
            vec![],
        ),
    )
}

pub(super) fn report_builtin_arity(
    errors: &mut DiagnosticSink<'_>,
    list: &deep::List,
    builtin: &str,
    expected: usize,
    got: usize,
) -> Type {
    report(
        errors,
        CheckError::new(
            CheckErrorKind::ArityMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!(
                    "builtin `{builtin}` expects {expected} argument(s), got {got} \
                     (chelis#731 [04-TOT-3])"
                ),
            ),
            vec![],
        ),
    )
}

/// True when a `uniform_like` low/high bound is a statically-resolvable numeric
/// value the compiled-backend lowering can fold to a compile-time constant.
///
/// This MIRRORS `extract_f64_value` in `chelis-ir/src/lower.rs` (chelis#776) so
/// the checker's disposition and the backend agree on what a valid bound is: a
/// bare `Float`/`Int` atom, a `(lit ...)` wrapping one, a float-target `cast` of
/// a resolvable value, or a `neg` of a resolvable value. A genuinely-runtime
/// bound (a variable, an `add`, a `shape()` read) is not resolvable and is the
/// checker's reject case; the lowering's fatal error is then defense-in-depth.
///
/// chelis#731 Phase 1 context: before the `handle-effect` case landed, this
/// gate never fired where `uniform_like` actually lives (inside a `with seed`
/// body, unchecked per chelis#709), so the older `is_numeric_literal_expr`
/// form (literal-only) never had to agree with the backend's broader fold set.
/// Now that the gate fires inside handler bodies, it is aligned with the fold
/// set so that `uniform_like(t, -3.0, ...)` (a `neg` literal) and
/// `uniform_like(t, cast(2.0, f32), ...)` -- both of which the backend folds --
/// are accepted rather than spuriously rejected.
pub(super) fn is_static_numeric_bound(expr: &deep::Expr) -> bool {
    // Recurses on the `cast`/`neg` inner node; guard the native stack against a
    // pathologically deep wrapper chain, bailing conservatively to `false` (the
    // bound is then treated as non-resolvable and rejected loudly, never a
    // silent accept).
    stack_guard!("is_static_numeric_bound", expr, false);
    match expr {
        deep::Expr::Atom(deep::Atom::Float(_) | deep::Atom::Int(_), _) => true,
        deep::Expr::List(list, _) => match get_tag(list) {
            // A float-target cast of a resolvable value. An integer target
            // truncates and is left unresolved, matching the lowering.
            Some(DeepTag::Cast) => {
                let inner_resolvable = list.elements.get(2).is_some_and(is_static_numeric_bound);
                let target_is_float = list.elements.get(3).is_some_and(deep_prim_is_float);
                target_is_float && inner_resolvable
            }
            // neg(<inner>): unary minus desugars to `(app {} (var {} neg) <inner>)`.
            Some(DeepTag::App) if children(list).first().is_some_and(expr_is_neg_var) => {
                children(list).get(1).is_some_and(is_static_numeric_bound)
            }
            // Only a `lit`-tagged list carries a numeric atom AS ITS VALUE.
            // The lowering's `extract_f64_value` catch-all reads element 2 of
            // ANY list, which blesses a form whose value is NOT at element 2 --
            // e.g. `(par {} 2.0 3.0)`, whose value is its LAST child (3.0) per
            // spec/03-deep-syntax.md §2.3, while element 2 is the FIRST child
            // (2.0). Accepting that would let the checker bless a bound the
            // lowering folds from the wrong position (the chelis#703 silent-
            // substitution shape; chelis#731 red team). The checker's arm is
            // therefore NARROWER than the lowering's on purpose: a `par`-wrapped
            // (or otherwise non-literal, non-cast, non-neg) bound is rejected
            // here, so it never reaches the fold. The lowering-side over-broad
            // catch-all is filed separately.
            Some(DeepTag::Lit) => matches!(
                list.elements.get(2),
                Some(deep::Expr::Atom(
                    deep::Atom::Float(_) | deep::Atom::Int(_),
                    _
                ))
            ),
            _ => false,
        },
        _ => false,
    }
}

/// True when `target` is a `(t-prim {} <name>)` naming a float precision.
pub(super) fn deep_prim_is_float(target: &deep::Expr) -> bool {
    matches!(target, deep::Expr::List(list, _)
        if get_tag(list) == Some(DeepTag::TPrim)
            && children(list)
                .first()
                .and_then(symbol_name)
                .and_then(Prim::parse_name)
                .is_some_and(|prim| prim.is_float()))
}

/// True when `expr` is `(var {} neg)`, the callee of a desugared unary minus.
pub(super) fn expr_is_neg_var(expr: &deep::Expr) -> bool {
    matches!(expr, deep::Expr::List(list, _)
        if get_tag(list) == Some(DeepTag::Var)
            && children(list).first().and_then(symbol_name) == Some("neg"))
}

/// Get metadata map from element[1] of a list.
pub(super) fn get_meta(list: &deep::List) -> Option<&deep::MetaMap> {
    if list.elements.len() > 1
        && let deep::Expr::Map(meta, _) = &list.elements[1]
    {
        return Some(meta);
    }
    None
}

/// True when a `deftype` node carries `opaque: true` metadata
/// (RFC D-META; the key is unprefixed language semantics).
pub(super) fn deftype_opaque_meta(meta: &deep::MetaMap) -> bool {
    meta.entries.iter().any(|(key, value)| {
        key == "opaque" && matches!(value, deep::Expr::Atom(deep::Atom::Bool(true), _))
    })
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

/// A constructor reference is *in scope* only when its exact name is bound
/// in `env` — either a bare builtin constructor (`Some`/`None`/`Cons`/`Nil`,
/// registered bare by `register_prelude_adts`) or a reef-mangled in-scope
/// constructor (chelis#157/#316 rewrite the reference to its mangled name
/// when the importing module declares it locally or imports it by name).
///
/// Returns `true` when `name` looks like a constructor (PascalCase terminal)
/// but is *not* bound exactly and is *only* reachable through the registry's
/// fuzzy terminal-segment fallback (`lookup_terminal_unique`). That fallback
/// is exactly the silent cross-module mis-resolution chelis#317 reports: a
/// type-only import leaves the bare constructor un-rewritten, and the fuzzy
/// match binds it to another module's mangled tag, deferring the failure to
/// a runtime non-exhaustive match. Such a reference must be rejected at
/// `check` as an unknown constructor instead.
pub(super) fn constructor_out_of_scope(name: &str, env: &Env) -> bool {
    is_constructor_name(name)
        && env.lookup(name).is_none()
        && env.lookup_terminal_unique(name).is_some()
}

/// Pattern-position counterpart of [`constructor_out_of_scope`]. A constructor
/// **pattern** head (`| Alpha =>`, `| Alpha { .. } =>`) is in scope only when it
/// resolves through an *exact* binding — either the type env (`env.lookup`, for
/// builtins and reef-mangled in-scope constructors) or the ADT registry
/// (`adt_reg.lookup_variant`, the exact mangled variant key). The terminal-unique
/// fallbacks (`env.lookup_terminal_unique` / `lookup_variant_terminal_unique`)
/// are diagnostic-only fuzzy matches, never an in-scope binding.
///
/// Returns `true` when `name` is a PascalCase constructor that resolves through
/// *neither* exact path. This rejects two out-of-scope cases the bare
/// [`constructor_out_of_scope`] env check misses for patterns (chelis#317):
///
///   1. unique fuzzy — exactly one foreign same-terminal variant exists, so a
///      bare `| Alpha =>` would fuzzy-bind to it; and
///   2. non-unique / unresolvable — two foreign modules export a same-terminal
///      `Dup`, so `lookup_*_terminal_unique` returns `None` and the arm would
///      otherwise push the bare name into `covered_variants` with no scheme and
///      no diagnostic. Normally that surfaces as `NonExhaustiveMatch`, but a `_`
///      wildcard arm (`has_wildcard`) suppresses exhaustiveness and the bogus
///      out-of-scope arm is silently accepted. Rejecting here closes that hole.
///
/// Soundness depends on the reef rewriter guaranteeing every genuinely in-scope
/// constructor reaches type-check exact-bound under its mangled name (see the
/// module-level note on the terminal-unique fallback). A future half-mangled
/// producer (mangled deftype, bare reference) would be wrongly rejected here —
/// which is the intended failure mode: a half-mangled program is a structural
/// defect, not a valid in-scope reference.
pub(super) fn constructor_pattern_out_of_scope(
    name: &str,
    env: &Env,
    adt_reg: &AdtRegistry,
) -> bool {
    is_constructor_name(name)
        && env.lookup(name).is_none()
        && adt_reg.lookup_variant(name).is_none()
}

pub(super) fn check_error_kind_from_type_error_kind(kind: &TypeErrorKind) -> CheckErrorKind {
    match kind {
        TypeErrorKind::TypeMismatch => CheckErrorKind::TypeMismatch,
        TypeErrorKind::PrecisionMismatch => CheckErrorKind::PrecisionMismatch,
        TypeErrorKind::DimensionMismatch => CheckErrorKind::DimensionMismatch,
        TypeErrorKind::ArityMismatch => CheckErrorKind::ArityMismatch,
        TypeErrorKind::OccursCheck => CheckErrorKind::OccursCheck,
        TypeErrorKind::NotAFunction => CheckErrorKind::NotAFunction,
    }
}

pub(super) fn extract_string_literal(expr: &deep::Expr) -> Option<String> {
    match expr {
        deep::Expr::Atom(deep::Atom::Str(value), _) => Some(value.clone()),
        deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::Lit) => {
            children(list).first().and_then(|child| match child {
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
/// - `Dim::Var(v)` — safe ONLY when `v` is also bound by a *parameter*
///   tensor-dim position of the declared signature (`param_dvars`). This
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
///   in NO parameter tensor position — e.g. `arange[n](start: int32,
///   stop: int32) -> tensor[n, int32]`, where the length comes from a
///   value parameter) is NOT in `param_dvars` and is deliberately left as
///   `Wildcard`. Baking such an unbound var into the generalized scheme is
///   the red-team RT-39+44 soundness regression (commit 8067c9ce): it
///   leaks a free dim var into callers and breaks the chelis-std self-test
///   corpus. The `param_dvars` gate is exactly the line between "the
///   caller supplies this dim" (safe) and "this dim is output-inferred /
///   value-parameter-derived" (unsafe).
///
/// - `Dim::Name`, `Dim::Rank`, existing `Var`/`Lit` in `ty` — preserved.
pub(super) fn narrow_wildcards_with(
    ty: &Type,
    template: &Type,
    param_dvars: &HashSet<DimVar>,
) -> Type {
    match (ty, template) {
        (Type::Tensor(dims, prec), Type::Tensor(tmpl_dims, _)) if dims.len() == tmpl_dims.len() => {
            let new_dims = dims
                .iter()
                .zip(tmpl_dims.iter())
                .map(|(d, t)| match (d, t) {
                    (Dim::Wildcard, Dim::Lit(_)) => t.clone(),
                    (Dim::Wildcard, Dim::Var(v)) if param_dvars.contains(v) => t.clone(),
                    _ => d.clone(),
                })
                .collect();
            Type::Tensor(new_dims, prec.clone())
        }
        (Type::Fn(args, ret), Type::Fn(t_args, t_ret)) if args.len() == t_args.len() => {
            let new_args = args
                .iter()
                .zip(t_args.iter())
                .map(|(a, t)| narrow_wildcards_with(a, t, param_dvars))
                .collect();
            let new_ret = Box::new(narrow_wildcards_with(ret, t_ret, param_dvars));
            Type::Fn(new_args, new_ret)
        }
        (Type::Tuple(ts), Type::Tuple(t_ts)) if ts.len() == t_ts.len() => {
            let new_ts = ts
                .iter()
                .zip(t_ts.iter())
                .map(|(t, tt)| narrow_wildcards_with(t, tt, param_dvars))
                .collect();
            Type::Tuple(new_ts)
        }
        (Type::Adt(n, args), Type::Adt(_, t_args)) if args.len() == t_args.len() => {
            let new_args = args
                .iter()
                .zip(t_args.iter())
                .map(|(a, t)| narrow_wildcards_with(a, t, param_dvars))
                .collect();
            Type::Adt(n.clone(), new_args)
        }
        _ => ty.clone(),
    }
}

/// Collect the set of dim vars that occur in a *parameter* (non-return)
/// tensor-dim position of a resolved declared `Fn` signature. These are
/// the dims a caller binds from its actual arguments; the
/// `narrow_wildcards_with` gate uses this set to decide when a body
/// wildcard may be safely narrowed to a declared `Dim::Var`. A non-`Fn`
/// type (or one whose params carry no tensor dim vars) yields the empty
/// set, so narrowing falls back to the literal-only behavior.
pub(super) fn param_bound_dvars(decl_ty: &Type) -> HashSet<DimVar> {
    let mut out = HashSet::new();
    if let Type::Fn(params, _) = decl_ty {
        for param in params {
            for dv in crate::env::free_dvars(param) {
                out.insert(dv);
            }
        }
    }
    out
}

pub(super) fn infer_gather_result_type(
    tensor_ty: &Type,
    indices_ty: &Type,
    axis: usize,
) -> Result<Type, String> {
    let Type::Tensor(tensor_dims, tensor_precision) = tensor_ty else {
        return Err(format!("gather expects tensor input, got {tensor_ty}"));
    };
    let Type::Tensor(index_dims, index_precision) = indices_ty else {
        return Err(format!(
            "gather expects integer tensor indices, got {indices_ty}"
        ));
    };
    if !index_precision.is_integer() {
        return Err(format!(
            "gather expects integer tensor indices, got tensor[..., {}]",
            index_precision.name()
        ));
    }
    if axis >= tensor_dims.len() {
        return Err(format!(
            "gather axis {axis} out of bounds for rank {}",
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
    Ok(Type::Tensor(out_dims, precision.clone()))
}

pub(super) fn infer_diagonal_result_type(
    tensor_ty: &Type,
    axis1: usize,
    axis2: usize,
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
    let diag_dim = match (&dims[axis1], &dims[axis2]) {
        (Dim::Lit(lhs), Dim::Lit(rhs)) if lhs == rhs => Dim::Lit(*lhs),
        _ => Dim::Wildcard,
    };
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
    let deep::Expr::List(list, _) = expr else {
        return None;
    };
    let meta = get_meta(list)?;
    let source = meta
        .entries
        .iter()
        .find(|(key, _)| key == "source")
        .map(|(_, value)| value)?;
    let rendered = chelis_deep::printer::print_canonical(std::slice::from_ref(source));
    Some(rendered.replace('\n', " ").trim().to_string())
}

pub(super) fn resolve_type_aliases(ty: &Type, adt_reg: &AdtRegistry) -> Type {
    let mut seen = HashSet::new();
    resolve_type_aliases_inner(ty, adt_reg, &mut seen)
}

pub(super) fn resolve_type_aliases_inner(
    ty: &Type,
    adt_reg: &AdtRegistry,
    seen: &mut HashSet<String>,
) -> Type {
    match ty {
        Type::Adt(name, args) => {
            let resolved_args: Vec<Type> = args
                .iter()
                .map(|arg| resolve_type_aliases_inner(arg, adt_reg, seen))
                .collect();

            if seen.contains(name) {
                return Type::Adt(name.clone(), resolved_args);
            }

            if let Some(expanded) = adt_reg.instantiate_alias(name, &resolved_args) {
                seen.insert(name.clone());
                let resolved = resolve_type_aliases_inner(&expanded, adt_reg, seen);
                seen.remove(name);
                resolved
            } else {
                Type::Adt(name.clone(), resolved_args)
            }
        }
        Type::Fn(args, ret) => Type::Fn(
            args.iter()
                .map(|a| resolve_type_aliases_inner(a, adt_reg, seen))
                .collect(),
            Box::new(resolve_type_aliases_inner(ret, adt_reg, seen)),
        ),
        Type::Tuple(ts) => Type::Tuple(
            ts.iter()
                .map(|t| resolve_type_aliases_inner(t, adt_reg, seen))
                .collect(),
        ),
        _ => ty.clone(),
    }
}

pub(super) fn resolve_deep_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    use_site: TypeUseSite,
    binder_mode: BinderMode<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> Result<Type, ErrorWitness> {
    let mut resolver =
        DeepTypeResolver::new(use_site, binder_mode, adt_reg.resolution_env(), vg, errors);
    let ty = resolver.resolve(expr)?.into_type();
    Ok(resolve_type_aliases(&ty, adt_reg))
}

// ── Declaration collection (first pass) ──────────────────────────

/// Which declaration kinds a `collect_declarations` sub-pass should process.
///
/// `deftype` constructor schemes expand transparent type aliases in their
/// field types at registration (see `AdtRegistry::expand_aliases`), so every
/// `typealias` must be in the registry first. Running `Aliases` over all
/// top-level items before `Rest` guarantees that even for a forward reference
/// — an alias declared textually after the `deftype` that uses it, as in
/// `Hull.Ast` where `type EffectRow = List[Effect]` follows `type Type = ...
/// | TArrow(Type, Type, EffectRow) | ...`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeclPhase {
    /// Process only `typealias` declarations.
    Aliases,
    /// Process everything except `typealias` (`deftype`, `defsig`, ...).
    Rest,
}

/// Run the two-phase declaration collection over `items` (already flattened
/// past `module` wrappers, each paired with its lexical module key):
/// register all type aliases, then everything else.
pub(super) fn collect_all_declarations(
    items: &[(Option<String>, &deep::Expr)],
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &mut AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    // chelis#258 (main): duplicate-def / builtin-shadowing rejection runs
    // over the bare item list. Our `items` is paired with module keys, so
    // project to the `&deep::Expr` slice the reporters expect.
    let bare_items: Vec<&deep::Expr> = items.iter().map(|(_, expr)| *expr).collect();
    report_duplicate_defs(&bare_items, errors);
    report_duplicate_defsigs(&bare_items, errors);
    report_builtin_shadowing(&bare_items, errors);
    report_builtin_param_call_shadowing(&bare_items, errors);
    let resolution_env = precollect_type_resolution_env(items, adt_reg);
    // Install the provisional self/forward header scope explicitly in this
    // per-check registry clone. Declaration bodies resolve against it, while
    // only successful bodies enter the validated maps that survive serde.
    adt_reg.install_resolution_env(resolution_env.clone());
    // chelis#930: per-declaration cancellation. This runs BEFORE body
    // inference, so without it the first ~1.3 s of a 1500-declaration check
    // (measured, debug build) is uninterruptible and a cancellation arriving
    // in that window waits it out. An abandoned collection leaves later
    // declarations unbound; the check entry's `cancellation_gate` rejects the
    // unit before anything reads it.
    let cancel = crate::cancel::current_cancel_token();
    for (module, expr) in items {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            return;
        }
        collect_declarations(
            expr,
            module.as_deref(),
            env,
            vg,
            subst,
            adt_reg,
            &resolution_env,
            errors,
            DeclPhase::Aliases,
        );
    }
    for (module, expr) in items {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            return;
        }
        collect_declarations(
            expr,
            module.as_deref(),
            env,
            vg,
            subst,
            adt_reg,
            &resolution_env,
            errors,
            DeclPhase::Rest,
        );
    }
}

/// Collect nominal names and arities before resolving any declaration body.
/// This permits self and forward references without registering an unchecked
/// definition in the serde-backed ADT registry.
pub(super) fn precollect_type_resolution_env(
    items: &[(Option<String>, &deep::Expr)],
    adt_reg: &AdtRegistry,
) -> TypeResolutionEnv {
    let mut headers = TypeResolutionEnv::from_registry(adt_reg);
    for (_, expr) in items {
        let Some((tag, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        if !matches!(tag, DeepTag::Deftype | DeepTag::Typealias) {
            continue;
        }
        let (Some(name), Some(params)) = (kids.first().and_then(symbol_name), kids.get(1)) else {
            continue;
        };
        if stamped_parts(params).is_some_and(|(tag, _, _)| tag == DeepTag::Variant) {
            // Legacy Deep permits omitting the explicit empty parameter list.
            headers.insert(name, 0);
        } else {
            let params = match params {
                deep::Expr::List(list, _) => list.elements.as_slice(),
                deep::Expr::BareList(elements, _) => elements.as_slice(),
                _ => continue,
            };
            if params.iter().all(|param| symbol_name(param).is_some()) {
                headers.insert(name, params.len());
            }
        }
    }
    headers
}

/// Build the program-shape opacity metadata (RFC D-CHECK) from the
/// flattened `(module key, item)` list: per-module export sets, the
/// top-level binding -> module map, and the formatted "exported
/// producers with signatures" entries per opaque type used by the
/// violation error contract. Runs after declaration collection so the
/// registry already carries every `deftype`'s `opaque` flag and
/// defining module.
pub(super) fn build_opacity_meta(
    items: &[(Option<String>, &deep::Expr)],
    adt_reg: &AdtRegistry,
    env: &Env,
) -> crate::opacity::OpacityModuleMeta {
    let mut meta = crate::opacity::OpacityModuleMeta::default();
    // Declared signature types (from `defsig` nodes) for producer
    // display; keyed by binding name like `meta.bindings`.
    let mut declared_sigs: HashMap<String, Type> = HashMap::new();
    // Pass 1: export sets. Lexical `(export ...)` nodes attribute to
    // their module wrapper; package-linked exports arrive as
    // top-level nodes whose names carry the reef internal-name stem
    // (the reef rewrite emits them with internal names), so each
    // exported name self-attributes through its stem.
    for (module, item) in items {
        let deep::Expr::List(list, _) = item else {
            continue;
        };
        if get_tag(list) != Some(DeepTag::Export) {
            continue;
        }
        for child in children(list) {
            let Some(name) = symbol_name(child) else {
                continue;
            };
            let target = match module {
                Some(module) => Some(module.clone()),
                None => crate::opacity::reef_module_stem(name),
            };
            if let Some(target) = target {
                meta.exports
                    .entry(target)
                    .or_default()
                    .insert(name.to_string());
            }
        }
    }
    // Pass 2: binding -> module attribution and declared sigs.
    // Stem-attributed (package-linked) bindings are recorded only
    // when their module's export set is known: without it, the sixth
    // rejection's no-export-decl-means-sealed rule would reject
    // legitimately exported producers in pipelines that strip Export
    // decls (fail-open for unattributable names by design).
    for (module, item) in items {
        let deep::Expr::List(list, _) = item else {
            continue;
        };
        if !matches!(get_tag(list), Some(DeepTag::Def) | Some(DeepTag::Defsig)) {
            continue;
        }
        let kids = children(list);
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let target = match module {
            Some(module) => Some(module.clone()),
            None => crate::opacity::reef_module_stem(name)
                .filter(|stem| meta.exports.contains_key(stem)),
        };
        let Some(target) = target else {
            continue;
        };
        meta.bindings.insert(name.to_string(), target);
        if get_tag(list) == Some(DeepTag::Defsig)
            && let Some(scheme) = env.lookup(name)
        {
            declared_sigs.insert(name.to_string(), scheme.body.clone());
        }
    }
    // Producer enumeration per opaque type: exported bindings of the
    // defining module whose declared RESULT type mentions the type
    // (containment chased through named type definitions).
    for (adt_name, def) in &adt_reg.defs {
        if !def.opaque {
            continue;
        }
        let Some(module) = &def.defining_module else {
            continue;
        };
        let Some(export_set) = meta.exports.get(module) else {
            continue;
        };
        let mut entries: std::collections::BTreeSet<String> = Default::default();
        for name in export_set {
            if meta.bindings.get(name) != Some(module) {
                continue;
            }
            let Some(sig) = declared_sigs.get(name) else {
                continue;
            };
            let result = match sig {
                Type::Fn(_, ret) => ret.as_ref(),
                other => other,
            };
            if crate::opacity::type_mentions_adt(result, adt_name, adt_reg) {
                // RT-1 F3: store the producer entry de-mangled so the
                // reef surface renders `probability: (f32) -> Probability`
                // rather than the internal `pkg__...` names.
                entries.insert(format!(
                    "{}: {}",
                    crate::opacity::demangle_ident(name),
                    crate::opacity::demangle_type(sig)
                ));
            }
        }
        if !entries.is_empty() {
            meta.producer_entries
                .entry(adt_name.clone())
                .or_default()
                .extend(entries);
        }
    }
    meta
}

/// Reject two same-name `def` declarations in one program (chelis#258).
///
/// A def's value binding is silent last-write-wins (`env.bind` →
/// `HashMap::insert`, like the `defsig` arm of `collect_declarations`), and
/// Chelis does not dispatch same-name `def`s by argument arity or tensor
/// rank. So two `def f`s whose sigs differ only in rank leave just one arm
/// reachable: callers of the other rank fire a confusing `DimensionMismatch`
/// at the call site instead of a clear error at the redundant definition.
/// This mirrors the duplicate-`deftype` / duplicate-`typealias` rejection
/// already in `collect_declarations`, moving the diagnostic to the
/// definition site.
///
/// Scoped to `def` (not `defsig`): a `defsig` legitimately co-occurs with a
/// synthesized signature for the same name (an inline-annotated `def`
/// desugars to both a `defsig` and a `def`), so a same-name `defsig` is not
/// on its own a duplicate definition. `items` is already flattened past
/// `module` wrappers, and the prelude lives in the builtin env rather than as
/// `def` nodes here, so only genuine in-program user redefinitions match.
pub(super) fn report_duplicate_defs(items: &[&deep::Expr], errors: &mut DiagnosticSink<'_>) {
    let mut seen: HashSet<&str> = HashSet::new();
    for expr in items {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some(DeepTag::Def) {
            continue;
        }
        let Some(name) = children(list).first().and_then(symbol_name) else {
            continue;
        };
        if !seen.insert(name) {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateDefinition,
                format!("duplicate definition: `{name}` is defined more than once"),
                vec![format!(
                    "rename one of the `{name}` definitions: Chelis does not dispatch same-name `def`s by argument type or rank"
                )],
            ));
        }
    }
}

/// Reject two same-name `defsig` declarations in one program.
///
/// Chelis does not dispatch user functions by arity, type, or rank; the valid
/// same-name declaration pair is exactly one `defsig` plus one `def`. Multiple
/// `defsig`s for a name otherwise feed several last-write-wins maps
/// (`collect_declarations`, declared-param-type collection, signature metadata)
/// and make the enforced signature order-dependent.
pub(super) fn report_duplicate_defsigs(items: &[&deep::Expr], errors: &mut DiagnosticSink<'_>) {
    let mut seen: HashSet<&str> = HashSet::new();
    for expr in items {
        let deep::Expr::List(list, _) = expr else {
            continue;
        };
        if get_tag(list) != Some(DeepTag::Defsig) {
            continue;
        }
        let Some(name) = children(list).first().and_then(symbol_name) else {
            continue;
        };
        if !seen.insert(name) {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateDefinition,
                format!("duplicate signature: `{name}` has more than one `defsig`"),
                vec![format!(
                    "keep a single `defsig` for `{name}`: Chelis does not dispatch same-name functions by argument type, arity, or rank"
                )],
            ));
        }
    }
}

/// Reject a top-level `def` or `defsig` whose name appears in the closed
/// builtin vocabulary (chelis#353, spec/04-type-system.md §8.6).
///
/// Call sites are dispatched builtin-first by name in both the host
/// evaluator (`runtime/host_ops.rs::builtin_name`) and IR lowering
/// (`lower.rs::builtin_name`); both import `BUILTIN_NAMES`, the same
/// table consulted here, so the rejected set and the dispatched set
/// cannot drift. A user definition with a builtin name is therefore
/// unreachable by name: pre-fix, `def sum` checked clean, hit the
/// builtin's arity error under eval, and segfaulted on the C backend —
/// three lanes, three different answers. Rejecting the declaration here,
/// in the collection chokepoint every checker entry point shares, makes
/// all lanes agree on the same diagnostic.
///
/// Deliberately narrow scope:
/// - Reef package modules never reach this check with bare names: reef
///   rewrites package decls to internal `pkg__...` names (and rewrites
///   their call sites with them) before the checker runs, so a package
///   `def sum` is allowed and genuinely dispatches to the user def (the
///   stdlib's `Std.Decimal.normalize` / `Std.Test.fail` rely on this).
/// - Function parameters and block-locals may reuse builtin names: they
///   bind values, not call-site dispatch, and shadow harmlessly on every
///   lane.
///
/// An inline-annotated `def` desugars to a `defsig` AND a `def` with the
/// same name; report once per name, as the `def` (what the user wrote).
pub(super) fn report_builtin_shadowing(items: &[&deep::Expr], errors: &mut DiagnosticSink<'_>) {
    let decl_name = |expr: &deep::Expr, tag: DeepTag| -> Option<String> {
        let deep::Expr::List(list, _) = expr else {
            return None;
        };
        if get_tag(list) != Some(tag) {
            return None;
        }
        children(list)
            .first()
            .and_then(symbol_name)
            .filter(|name| builtins::BUILTIN_NAMES.contains(name))
            .map(str::to_string)
    };

    let def_names: HashSet<String> = items
        .iter()
        .filter_map(|expr| decl_name(expr, DeepTag::Def))
        .collect();

    let mut reported: HashSet<String> = HashSet::new();
    for expr in items {
        let Some(name) = decl_name(expr, DeepTag::Def).or_else(|| decl_name(expr, DeepTag::Defsig))
        else {
            continue;
        };
        if !reported.insert(name.clone()) {
            continue;
        }
        let decl_kw = if def_names.contains(&name) {
            "def"
        } else {
            "sig"
        };
        errors.push(CheckError::new(
            CheckErrorKind::BuiltinShadowing,
            format!(
                "`{decl_kw} {name}` shadows the builtin function `{name}`: user `def`/`sig` \
                 declarations may not reuse builtin names (spec/04-type-system.md \u{00a7}8.6). \
                 Calls to `{name}` always dispatch to the builtin under eval and lowering, so \
                 the shadowing declaration can never be reached by name."
            ),
            vec![format!(
                "rename `{name}` (e.g. `{name}2` or `my_{name}`); inside a reef package \
                 module the name is allowed because package declarations are \
                 internal-name-rewritten before checking"
            )],
        ));
    }
}

/// A function parameter MAY reuse a builtin name (the deliberate #353
/// carve-out -- it binds a value and shadows harmlessly in value position),
/// but a CALL through that name never reaches the parameter: the host
/// evaluator (`runtime/eval.rs::eval_app`) and IR lowering both dispatch
/// builtin-first by name. `def apply(round_to, x) = round_to(x, 0)`
/// therefore type-checked while silently invoking the BUILTIN (chelis#891
/// review finding 4). Reject exactly that shape -- a builtin-named
/// parameter applied by name inside its own scope -- and leave
/// value-position reuse intact (pinned by
/// `value_params_and_locals_may_reuse_builtin_names`).
///
/// Walks through [`stamped_parts`], so both the stamped `Expr::Node`
/// carrier (all compiler ingress since #908) and the legacy programmatic
/// `Expr::List` carrier are covered.
pub(super) fn report_builtin_param_call_shadowing(
    items: &[&deep::Expr],
    errors: &mut DiagnosticSink<'_>,
) {
    let mut reported: HashSet<String> = HashSet::new();
    for expr in items {
        let mut scope: Vec<String> = Vec::new();
        walk_builtin_param_calls(expr, &mut scope, &mut reported, errors);
    }
}

/// Extract a parameter's bound name from any of the shapes
/// [`extract_params`] accepts (bare `Name` atom, `MetaExpr`-wrapped name,
/// legacy `(name {type: ..})` list, or stamped `BareList` pair), without
/// resolving annotations.
fn builtin_shadow_param_name(param: &deep::Expr) -> Option<&str> {
    let mut current = param;
    loop {
        match current {
            deep::Expr::Atom(deep::Atom::Name(name), _) => return Some(name.as_str()),
            deep::Expr::MetaExpr(meta, _) => current = &meta.expr,
            deep::Expr::BareList(elements, _) => return elements.first().and_then(symbol_name),
            deep::Expr::List(list, _) => return list.elements.first().and_then(symbol_name),
            _ => return None,
        }
    }
}

/// The element slice of a `(params ...)` container, mirroring the carrier
/// shapes [`extract_params`] accepts.
fn builtin_shadow_param_elems(container: &deep::Expr) -> &[deep::Expr] {
    match container {
        deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => node.children_slice(),
        deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::Params) => children(list),
        deep::Expr::List(list, _) => list.elements.as_slice(),
        deep::Expr::BareList(elements, _) => elements.as_slice(),
        _ => &[],
    }
}

/// Extract the name of a `(var name)` callee, unwrapping `MetaExpr`
/// annotation layers, on either physical carrier.
fn builtin_shadow_callee_name(callee: &deep::Expr) -> Option<&str> {
    let mut current = callee;
    loop {
        if let deep::Expr::MetaExpr(meta, _) = current {
            current = &meta.expr;
            continue;
        }
        return match stamped_parts(current) {
            Some((DeepTag::Var, _, kids)) => kids.first().and_then(symbol_name),
            _ => None,
        };
    }
}

fn walk_builtin_param_calls(
    expr: &deep::Expr,
    scope: &mut Vec<String>,
    reported: &mut HashSet<String>,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("walk_builtin_param_calls", expr);
    match expr {
        deep::Expr::MetaExpr(meta, _) => {
            walk_builtin_param_calls(&meta.expr, scope, reported, errors);
        }
        deep::Expr::BareList(elements, _) => {
            for element in elements {
                walk_builtin_param_calls(element, scope, reported, errors);
            }
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                walk_builtin_param_calls(child, scope, reported, errors);
            }
        }
        _ => {
            let Some((tag, _, kids)) = stamped_parts(expr) else {
                return;
            };
            if tag == DeepTag::Fn {
                let mut added = 0usize;
                if let Some(container) = kids.first() {
                    for param in builtin_shadow_param_elems(container) {
                        if let Some(name) = builtin_shadow_param_name(param)
                            && builtins::BUILTIN_NAMES.contains(&name)
                        {
                            scope.push(name.to_string());
                            added += 1;
                        }
                    }
                }
                for kid in kids.iter().skip(1) {
                    walk_builtin_param_calls(kid, scope, reported, errors);
                }
                scope.truncate(scope.len() - added);
                return;
            }
            if tag == DeepTag::App
                && let Some(callee) = kids.first()
                && let Some(name) = builtin_shadow_callee_name(callee)
                && scope.iter().any(|param| param == name)
                && reported.insert(name.to_string())
            {
                errors.push(CheckError::new(
                    CheckErrorKind::BuiltinShadowing,
                    format!(
                        "parameter `{name}` shadows the builtin `{name}` and is called in \
                         this function body: calls dispatch builtin-first under eval and \
                         lowering (spec/04-type-system.md \u{00a7}8.6), so `{name}(...)` here \
                         always invokes the builtin; the parameter can never be reached \
                         by name."
                    ),
                    vec![format!(
                        "rename the parameter (e.g. `{name}_fn`); builtin-named parameters \
                         remain allowed in value position"
                    )],
                ));
            }
            for kid in kids {
                walk_builtin_param_calls(kid, scope, reported, errors);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn collect_declarations(
    expr: &deep::Expr,
    lexical_module: Option<&str>,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &mut AdtRegistry,
    headers: &TypeResolutionEnv,
    errors: &mut DiagnosticSink<'_>,
    phase: DeclPhase,
) {
    let Some((tag, meta, kids)) = stamped_parts(expr) else {
        return;
    };

    // Aliases register first so `deftype` field-type alias expansion sees a
    // fully-populated alias table; every other decl kind runs in the second
    // sub-pass.
    let in_phase = match phase {
        DeclPhase::Aliases => tag == DeepTag::Typealias,
        DeclPhase::Rest => tag != DeepTag::Typealias,
    };
    if !in_phase {
        return;
    }

    match tag {
        DeepTag::Deftype => {
            // Reject same-namespace collisions (another `deftype`, a
            // `typealias`, or a prelude ADT registered earlier in this
            // program). Without this check `AdtRegistry::defs` is
            // silently last-write-wins, which propagates wrong
            // constructor types and (per `compute_tensor_carrying_adts`
            // in linearity.rs) order-dependent borrow semantics.
            if let Some(name) = kids.first().and_then(symbol_name)
                && let Some(prior_kind) = adt_reg.existing_kind(name)
            {
                errors.push(CheckError::new(
                    CheckErrorKind::DuplicateDefinition,
                    format!(
                        "duplicate type definition: `{name}` was already declared as a {prior_kind}"
                    ),
                    vec![format!("rename one of the `{name}` declarations")],
                ));
                return;
            }
            // RFC D-CHECK: record opacity + module identity on the
            // registered AdtDef. The module key is the lexical
            // wrapper when present, else the reef internal-name stem
            // of the deftype's own (rewritten) name. `@opaque`
            // requires a named module (RT-0 M6): a top-level opaque
            // declaration has no module identity, which would make
            // the enforcement boundary collide across combined
            // sources.
            let opaque = deftype_opaque_meta(meta);
            let defining_module = crate::opacity::module_key_for_item(
                lexical_module,
                kids.first().and_then(symbol_name),
            );
            if opaque
                && defining_module.is_none()
                && let Some(name) = kids.first().and_then(symbol_name)
            {
                errors.push(crate::opacity::unmoduled_opaque_error(name));
            }
            if let Ok(ctors) =
                adt_reg.register_deftype(kids, vg, headers, errors, opaque, defining_module)
            {
                for (name, scheme) in ctors {
                    env.bind(name, scheme);
                }
            }
        }
        DeepTag::Defsig => {
            // (defsig {} name type_expr)
            if kids.len() >= 2
                && let Some(name) = symbol_name(&kids[0])
            {
                let mut resolver = DeepTypeResolver::new(
                    TypeUseSite::Defsig,
                    BinderMode::ImplicitGeneric,
                    headers,
                    vg,
                    errors,
                );
                if let Ok(ty) = resolver.resolve(&kids[1]) {
                    let ty = resolve_type_aliases(&ty.into_type(), adt_reg);
                    let scheme = env.generalize(&ty, subst);
                    env.bind(name.to_string(), scheme);
                }
            }
        }
        DeepTag::Typealias => {
            // (typealias {} Name (params...) type_expr)
            if kids.len() >= 3
                && let Some(name) = symbol_name(&kids[0])
            {
                if let Some(prior_kind) = adt_reg.existing_kind(name) {
                    errors.push(CheckError::new(
                        CheckErrorKind::DuplicateDefinition,
                        format!(
                            "duplicate type definition: `{name}` was already declared as a {prior_kind}"
                        ),
                        vec![format!("rename one of the `{name}` declarations")],
                    ));
                    return;
                }
                let params = match &kids[1] {
                    deep::Expr::List(list, _) => list
                        .elements
                        .iter()
                        .filter_map(symbol_name)
                        .map(str::to_string)
                        .collect::<Vec<_>>(),
                    deep::Expr::BareList(elements, _) => elements
                        .iter()
                        .filter_map(symbol_name)
                        .map(str::to_string)
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };

                let explicit_params: HashSet<String> = params.iter().cloned().collect();
                let mut resolver = DeepTypeResolver::new(
                    TypeUseSite::TypeAliasBody,
                    BinderMode::Explicit(&explicit_params),
                    headers,
                    vg,
                    errors,
                );
                if let Ok(aliased_ty) = resolver.resolve(&kids[2]) {
                    let param_vars = params
                        .iter()
                        .map(|param| {
                            resolver.type_var(param).expect(
                                "explicit alias params are pre-bound as nominal type arguments",
                            )
                        })
                        .collect();
                    adt_reg.register_alias(
                        name.to_string(),
                        params,
                        param_vars,
                        aliased_ty.into_type(),
                    );
                }
            }
        }
        _ => {}
    }
}

// ── Tier-2 rank-polymorphism Body Discipline ─────────────────────

/// True if any tensor inside `ty` carries a `Dim::Rank` (a rank variable).
pub(super) fn type_contains_rank(ty: &Type) -> bool {
    match ty {
        Type::Tensor(dims, _) => dims.iter().any(|d| matches!(d, Dim::Rank(_))),
        Type::Fn(args, ret) => args.iter().any(type_contains_rank) || type_contains_rank(ret),
        Type::Ref(inner) => type_contains_rank(inner),
        Type::Adt(_, args) => args.iter().any(type_contains_rank),
        Type::Tuple(ts) => ts.iter().any(type_contains_rank),
        Type::Prim(_) | Type::Var(_) | Type::Unit | Type::Error(_) => false,
    }
}

/// Extract the callee name from an `app`'s first child when it is `(var {} name)`.
pub(super) fn app_var_name(callee: &deep::Expr) -> Option<&str> {
    let deep::Expr::List(list, _) = callee else {
        return None;
    };
    if get_tag(list) != Some(DeepTag::Var) {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

/// Names of every top-level `def` in the program (after module flattening),
/// so the Body-Discipline check can reject a call that resolves to a user
/// function shadowing an Identity builtin name (chelis#258 §4.2).
pub(super) fn collect_user_def_names(items: &[&deep::Expr]) -> HashSet<String> {
    let mut out = HashSet::new();
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
/// named axis and carry the rest through). Rejected: positional shape-rewriting
/// builtins (`permute`/`reshape`/`matmul`/…), and any user/non-builtin/computed
/// callee not proven rank-safe — against a spread `..r` there are no named axes
/// left to catch an untracked transposition/reshape, so admitting one would
/// silently break §4.2 transposition safety (spec/design/rank_polymorphism.md
/// §Soundness Boundary).
pub(super) fn check_rank_body_discipline(
    def_name: &str,
    expr: &deep::Expr,
    user_def_names: &HashSet<String>,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("check_rank_body_discipline", expr);
    let deep::Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        // Function-taking transforms apply a *referenced* user function across
        // the opaque rank. That callee is not inlined here, so its body can
        // transpose/reshape undetected — reject outright (spec §4.2).
        // `jit`/`realize`/`cast`/`copy` wrap an *inline* expression that the
        // recursion below still checks, so they are not rejected here.
        Some(t @ (DeepTag::Grad | DeepTag::Vmap)) => {
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
        Some(DeepTag::App) => match children(list).first().and_then(app_var_name) {
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
                        builtins::ShapeClass::Identity | builtins::ShapeClass::NameTracked
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
                         (elementwise) operations, named-axis reductions, and named-axis expand \
                         only."
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
    for child in &list.elements {
        check_rank_body_discipline(def_name, child, user_def_names, errors);
    }
}

// ── Top-level inference (second pass) ────────────────────────────

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
    provisional_recursive_type: Option<&Type>,
    defer_recursive_binding: bool,
    user_def_names: &HashSet<String>,
    declared_signatures: &HashMap<String, DeclaredSigMetadata>,
) -> Option<(String, Type)> {
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        // chelis#858 / [04-TOT-1]: a top-level list with no decoded tag
        // used to be silently skipped here, so a program like
        // `((var {} f) (var {} x))` was never type-checked while the
        // fitness clean path manufactured a vacuous 1.0. The disposition
        // is a loud rejection; the raw-string boundary names an unknown
        // symbol head when there is one.
        let named = match expr {
            deep::Expr::List(list, _) => list.unknown_tag_symbol().unwrap_or("<untagged-list>"),
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

        // Save declared type from defsig BEFORE inferring (it may get overwritten)
        let declared_ty = if provisional_recursive_type.is_none() {
            env.lookup(&name).map(|s| {
                let s = s.clone();
                env.instantiate(&s, vg)
            })
        } else {
            None
        };
        // A declaration's signature owns the only named binders legal in its
        // nested source annotations. Infer against a lexical clone so the
        // scope follows nested env clones but cannot leak to the next `def`
        // or into the reusable top-level environment.
        let mut body_env = env.clone();
        body_env.set_type_resolution_binders(
            declared_signatures
                .get(&name)
                .map(|metadata| &metadata.binders),
        );

        let body_diagnostic_checkpoint = errors.checkpoint();
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
        // surfacing as `def 'tadd' body doesn't match declared signature:
        // body has type `(&t, &t) -> &t`, declared type is `(&t, &t) -> t``.
        // Annotated params do not hit this because their concrete type
        // (`&tensor[..]`) flows through the call site directly. Seeding
        // bare params with the declared type here makes the bare-arg path
        // behave the same as the annotated path. See
        // `crates/chelis-cli/tests/bareref_return_inference.rs`.
        let body_ty = if let Some(witness) = prebound_type_failure {
            propagate(witness)
        } else if let Some(decl_ty) = &declared_ty {
            let inferred = infer_def_body_with_sig(
                &kids[1],
                decl_ty,
                &mut body_env,
                vg,
                subst,
                adt_reg,
                errors,
                product,
            );
            product.record_bypass(
                &kids[1],
                inferred.clone(),
                "declared-signature function inference",
            );
            inferred
        } else {
            infer_expr(&kids[1], &mut body_env, vg, subst, adt_reg, errors, product)
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
            .any(|e| matches!(e.kind, CheckErrorKind::UnboundVariable));

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
        // and the body's tail position resolves to a `(var x)`
        // reference (after walking `let`/`if`/`match` wrappers via
        // `descend_to_tail_var`) whose declared return is owned `T`
        // while the body's inferred type returns `Ref(T)`, retry the
        // unify against the declared return relaxed into `Ref(T)`.
        // This mirrors `auto_borrow_call_arg_types`'s owned-to-borrow
        // coercion at argument positions: the caller already arranged
        // the borrow lifetime via the param itself, and the tail-var
        // descent confirms every reachable return path returns the
        // same parameter.  Heterogeneous returns and bodies whose tail
        // is an `app` or other non-var expression still fail with the
        // existing TypeMismatch.
        let scheme_body = if let Some(decl_ty) = declared_ty {
            let unify_result = unify(&body_ty, &decl_ty, subst);
            let resolved_body = subst.apply(&body_ty);
            let resolved_decl = subst.apply(&decl_ty);
            // Declared-dim rigidity check (TypeCheck-FreeDimVarUnification-F1
            // Path B). The declared signature's param positions introduce
            // the universally-quantified dim parameters; after the
            // post-body sig-unify above, two distinct declared dims must
            // not have collapsed into one another (and none may have been
            // pinned to a concrete literal). This runs here, not inside
            // `infer_def_body_with_sig`, because the collapse for an
            // annotated-param body happens in the sig-unify itself, not
            // during body inference. The body's tail returns the wrong
            // declared dim (`def g[n, m](x: tensor[n, f32],
            // y: tensor[m, f32]) -> tensor[n, f32] = y`), and the
            // structural relaxed-retry guard does not see it because the
            // initial unify already succeeded by collapsing `n` and `m`.
            let mut declared_dvars: Vec<DimVar> = Vec::new();
            if let Type::Fn(decl_params, _) = &decl_ty {
                for t in decl_params {
                    for dv in crate::env::free_dvars(t) {
                        if !declared_dvars.contains(&dv) {
                            declared_dvars.push(dv);
                        }
                    }
                }
            }
            check_declared_dvars_rigid(&declared_dvars, subst, errors);
            // chelis#273: the param-position guard above never sees a dim
            // parameter that occurs only in the return type, so a body
            // could silently pin a return-only rigid dim. Reject the
            // input-coupled pins/collapses while keeping the legitimate
            // output-inferred uses (hello_tensor-style) green.
            check_return_only_dvars_rigid(&decl_ty, &declared_dvars, subst, errors);
            // Tier-2 rank-polymorphism Body Discipline
            // (spec/design/rank_polymorphism.md §Soundness Boundary, spec §4.2):
            // a def whose signature mentions a rank variable `..r` may call only
            // shape-identity (elementwise) builtins. Against an opaque rank there
            // are no named axes left to catch a transposition/reshape, so any
            // shape-rewriting op (or an unproven user call) is rejected here.
            if type_contains_rank(&decl_ty)
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
            check_list_elem_rigid_dim_vs_wildcard(&decl_ty, &resolved_body, errors);
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
            // (e.g. `def use_mix(x: tensor[3, int32]) -> tensor[3, f32]
            // = poly_id(nonexistent_function(x))`, where the outer call
            // early-exits at `Type::Error` so the body's `fn` type is
            // `Fn([tensor[3, int32]], Type::Error)`) silently satisfy a
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
                // RT-2 fixup B1: when the mismatch is a tensor
                // precision mismatch (notably a `reduce_sum` body
                // whose result precision differs from the declared
                // one), include a §5.7.1 cite directly in the message
                // so the user sees the result-precision table rule
                // rather than a generic "doesn't match declared
                // signature".
                let extra = match (&resolved_body, &resolved_decl) {
                    (Type::Tensor(_, body_prec), Type::Tensor(_, decl_prec))
                        if body_prec != decl_prec =>
                    {
                        format!(
                            " (precision `{}` vs declared `{}`; if the body is a \
                             `reduce_sum`, see spec/04-type-system.md §5.7.1: \
                             narrow integer operands widen to int32 to prevent \
                             silent overflow; use `tensor[{}]` or omit the result \
                             type)",
                            body_prec.name(),
                            decl_prec.name(),
                            body_prec.name(),
                        )
                    }
                    _ => String::new(),
                };
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "def '{}' body doesn't match declared signature: \
                         body has type `{}`, declared type is `{}`{}",
                        name, resolved_body, resolved_decl, extra
                    ),
                    vec![],
                ));
            }
            // #39 wildcard narrowing. The narrow may now substitute a
            // declared `Dim::Var` (not just `Dim::Lit`) into a body
            // wildcard, but ONLY for dim vars bound by a parameter tensor
            // position (`param_bound_dvars`). That keeps the
            // `const_col[n](spots: tensor[n, ..]) -> tensor[n, 1]` family's
            // return dim tied to its input (chelis#405 / WS-3 build ICE)
            // while leaving return-only / value-parameter dims as
            // wildcards (the RT-39+44 soundness boundary, commit 8067c9ce).
            let param_dvars = param_bound_dvars(&resolved_decl);
            narrow_wildcards_with(&resolved_body, &resolved_decl, &param_dvars)
        } else if let Some(provisional) = provisional_recursive_type {
            if let Err(error) = unify(&body_ty, provisional, subst) {
                errors.push(error.into());
            }
            subst.apply(provisional)
        } else {
            body_ty
        };

        product.record_bypass(expr, scheme_body.clone(), "top-level declaration inference");

        // chelis#397/#469: record the size provenance of a top-level value
        // binding (e.g. `zero_count = sub(cast(0, int32), cast(0, int32))`)
        // BEFORE binding it, so a later `expand(b, 0, zero_count)` recovers
        // whether it is a materializable extent (static / shape-sourced) or a
        // sourceless runtime scalar. Classified against the pre-binding scope.
        // The `Sourceless`/`Unknown` arm CLEARS any stale provenance so a
        // re-bind to a sourceless RHS does not inherit an earlier entry.
        match classify_expand_size(&kids[1], env) {
            SizeClass::Static => {
                env.mark_size_provenance(&name, crate::env::SizeProvenance::Static);
            }
            SizeClass::ShapeSourced => {
                env.mark_size_provenance(&name, crate::env::SizeProvenance::ShapeSourced);
            }
            SizeClass::Sourceless | SizeClass::Unknown => env.clear_size_provenance(&name),
        }
        // chelis#631: same discipline for list-literal lengths.
        note_list_literal_binding(env, &name, &kids[1]);
        if defer_recursive_binding {
            Some((name, scheme_body))
        } else {
            let scheme = env.generalize(&scheme_body, subst);
            env.bind(name, scheme);
            None
        }
    } else {
        // Any other top-level expression
        let _ = infer_expr(expr, env, vg, subst, adt_reg, errors, product);
        None
    }
}

// ── Core inference ───────────────────────────────────────────────
