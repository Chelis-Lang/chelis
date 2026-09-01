//! Collection operation and constructor rules.
//!
//! These helpers preserve list callback diagnostics and concat shape rules.

use super::*;
use crate::adt::VariantInfo;

pub(super) fn collection_helper_type_error(
    expr: &deep::Expr,
    helper: &str,
    contract: &str,
    te: TypeError,
) -> CheckError {
    let kind = check_error_kind_from_type_error_kind(&te.kind);
    let suggestions = match kind {
        CheckErrorKind::PrecisionMismatch => vec!["Insert explicit cast".to_string()],
        _ => vec![],
    };
    CheckError::new(
        kind,
        with_macro_provenance(expr, format!("{helper} {contract}; {}", te.message)),
        suggestions,
    )
}

/// How the concat arm can see the list's elements (chelis#594).
pub(super) enum ConcatListInfo {
    /// A literal `Cons` chain at the call site: each element's full dim
    /// vector, in list order (read from the current inference epoch's
    /// already-recorded child types).
    /// Ragged literal extents SUM.
    Direct(Vec<Vec<Dim>>),
    /// A variable (or anything else): only a binding-carried literal
    /// LENGTH survives ([`Env::list_literal_len`]); the per-axis extents
    /// come from the §4.5.2 joined element type, so the sum is
    /// `joined extent x length` and requires uniform extents.
    BindingLen(Option<usize>),
}

/// Result type of a tensor `concat(list, axis)` (spec/04-type-system.md
/// §4.5.4, chelis#631/#594), computed from the joined element type
/// (§4.5.2), the concat-axis value, and the statically-visible elements:
///
/// - literal axis + DIRECT literal list whose every element carries a
///   literal extent on the concat axis → `Lit(sum of extents)` — ragged
///   lists included (chelis#594). Any non-literal element extent (a
///   name, a variable, a wildcard) makes the sum unknown → `Wildcard`.
///   Every other axis is the joined element type's axis unchanged.
/// - literal axis + BINDING-carried length `n >= 1` + joined element
///   extent `Lit(k)` → `Lit(k * n)` (uniform extents only: the join has
///   already widened ragged literals to `*`, and the head-biased
///   `(concrete, wildcard)` join boundary is inherited on this path —
///   the runtime dim guards keep any violation loud, never mis-sized).
/// - literal axis, extents or count unknown → `Wildcard` on the CONCAT
///   axis. (Pre-chelis#631 the LAST axis was wildcarded unconditionally
///   and the concat axis kept the element's dim — a wrong concrete
///   extent the host-program C lane baked into its tensor-helper
///   signatures, aborting guarded forward binaries at run time.)
/// - literal axis out of bounds after negative-axis normalization → Err.
/// - non-literal (runtime) axis → every axis `Wildcard` at the element
///   rank: the host runtime concatenates along a computed axis, so rank
///   is known (§4.5.1 rank uniformity) but no per-axis extent survives.
pub(super) fn tensor_concat_result_type(
    element_ty: &Type,
    raw_axis: Option<i64>,
    list_info: ConcatListInfo,
) -> Result<Type, String> {
    let Type::Tensor(dims, precision) = element_ty else {
        return Err(format!(
            "concat expects List[tensor[...]] for tensor concatenation, got {element_ty}"
        ));
    };
    if dims.is_empty() {
        return Err("concat expects tensor inputs with at least one axis".to_string());
    }
    let mut out_dims = dims.clone();
    let Some(raw) = raw_axis else {
        out_dims.fill(Dim::Wildcard);
        return Ok(Type::Tensor(out_dims, precision.clone()));
    };
    let rank = out_dims.len();
    let Some(axis) = normalize_static_axis(rank, raw) else {
        return Err(format!("concat axis {raw} out of bounds for rank {rank}"));
    };
    out_dims[axis] = match list_info {
        ConcatListInfo::Direct(elements) if !elements.is_empty() => elements
            .iter()
            .try_fold(0i64, |total, dims| match dims.get(axis) {
                Some(Dim::Lit(k)) if dims.len() == rank => total.checked_add(*k),
                _ => None,
            })
            .map(Dim::Lit)
            .unwrap_or(Dim::Wildcard),
        ConcatListInfo::BindingLen(Some(n)) if n >= 1 => match &out_dims[axis] {
            Dim::Lit(k) => match k.checked_mul(n as i64) {
                Some(total) => Dim::Lit(total),
                None => Dim::Wildcard,
            },
            _ => Dim::Wildcard,
        },
        _ => Dim::Wildcard,
    };
    Ok(Type::Tensor(out_dims, precision.clone()))
}

/// Statically-known element count of a list expression (chelis#631): a
/// literal `Cons`/`Nil` chain counts directly; a variable carries a
/// length only when it was bound to a list literal in an enclosing scope
/// ([`Env::list_literal_len`]). `None` for anything else — a function
/// result, a parameter, a `split` output.
pub(super) fn static_list_len(expr: Option<&deep::Expr>, env: &Env) -> Option<usize> {
    let expr = expr?;
    if let Some(elements) = collect_cons_chain_for_shape(expr) {
        return Some(elements.len());
    }
    // chelis#1107: carrier-preserving read; a `List`-only destructure lost the
    // recorded literal length of a stamped `(var {} xs)` on the typed ingress.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag == DeepTag::Var {
        let name = kids.first().and_then(|e| symbol_name(e))?;
        return env.list_literal_len(name);
    }
    None
}

/// Record (or clear) the statically-known list-literal length of a
/// binding so a later `concat(name, axis)` can count elements
/// (chelis#631). Add-symmetric like the size-provenance marking beside
/// it: a re-bind to a non-literal RHS must clear any stale entry. A
/// `(var other)` RHS propagates an existing entry transitively.
pub(super) fn note_list_literal_binding(env: &mut Env, name: &str, rhs: &deep::Expr) {
    match static_list_len(Some(rhs), env) {
        Some(len) => env.mark_list_literal_len(name, len),
        None => env.clear_list_literal_len(name),
    }
}

/// Resolve the exact constructor variant selected by declaration/import
/// scope. The ADT registry may contain several same-named variants; its
/// iteration or sort order is not a name-resolution authority.
fn active_constructor_variant<'env, 'adt>(
    name: &str,
    env: &'env Env,
    adt_reg: &'adt AdtRegistry,
) -> Option<(&'env str, &'adt VariantInfo)> {
    let (owner, _) = env.lookup_constructor(name)?;
    let variant = adt_reg
        .lookup(owner)?
        .variants
        .iter()
        .find(|variant| variant.name == name)?;
    Some((owner, variant))
}

/// Resolve an ADT constructor application and enforce its call shape.
pub(super) fn prepare_constructor_application(
    func_name: &Option<String>,
    env: &Env,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> Result<Option<String>, Type> {
    macro_rules! reject {
        ($($arg:tt)*) => {
            return Err(report($($arg)*))
        };
    }

    let ctor_lookup_name = func_name
        .as_ref()
        .and_then(|fname| active_constructor_variant(fname, env, adt_reg).map(|_| fname.clone()));

    // The call site uses positional `(app)` syntax here (named-field
    // record construction lowers through a different builder, not
    // through `infer_app`). When two ADTs in the dep graph define
    // same-named constructors with different shapes (chelis#148: e.g.
    // Coral.Frame.Column.IntCol is positional, School.Data.Dataset.IntCol
    // is record), prefer the positional variant for this call site so
    // the call dispatches to the matching ADT instead of erroring on
    // the colliding record variant. Only emit the "must use named
    // fields" error when EVERY same-named variant in scope is record-
    // shaped, which is the original single-package case the error was
    // written for.
    // RFC D-CHECK: positional application of an out-of-module opaque
    // constructor is rejected (one violation per call site; the
    // callee `var`'s constructor-reference check is suppressed below
    // so the application does not double-report). Inference continues
    // so the call still yields its true type.
    if let Some(ref fname) = ctor_lookup_name
        && let Some((adt_name, _)) = active_constructor_variant(fname, env, adt_reg)
    {
        let adt_name = adt_name.to_string();
        crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::CtorApplication,
            &adt_name,
            adt_reg,
            errors,
        );
    }

    // chelis#317: do not emit the record-shape diagnostic for an applied
    // constructor whose name is out of scope (a type-only import that calls
    // `Alpha(...)`). The shape check resolves through the same fuzzy
    // terminal fallback that mis-binds out-of-scope names, so firing it here
    // would mask the real defect with a confusing "must use named fields"
    // message. Let the head's `infer_var` report `unknown constructor`
    // instead.
    let ctor_call_out_of_scope = func_name
        .as_deref()
        .is_some_and(|fname| constructor_out_of_scope(fname, env));
    if !ctor_call_out_of_scope
        && let Some(ref fname) = ctor_lookup_name
        && let Some((_adt_name, variant)) = active_constructor_variant(fname, env, adt_reg)
        && !variant.fields.is_empty()
        && variant
            .fields
            .iter()
            .all(|(field_name, _)| field_name.is_some())
    {
        reject!(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "{fname} is a record constructor and must use named fields: {fname} {{ ... }}"
                ),
                vec![],
            ),
        );
    }

    Ok(ctor_lookup_name)
}
