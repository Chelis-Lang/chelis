//! Declared type resolution, explicit binder lists, and dtype bounds.
//!
//! These helpers own the boundary between declaration syntax and the
//! checker's resolved binder state. Keeping them together prevents ordinary
//! expression inference from becoming a second declaration owner.

use super::*;

/// A declaration's resolved type together with the three binder facts the
/// declaration carries: the dtype-family bounds its metadata declared
/// (chelis#1474), the source spelling of every dimension parameter it
/// introduced (chelis#260), and the source spelling of every authored TYPE
/// binder it introduced (chelis#260 Site 2 and chelis#1486, [04-INF-6]).
///
/// A struct rather than a tuple because this has now grown twice: chelis#1474
/// added the dtype-family bounds and chelis#260 added the dimension names,
/// each time making an unnamed tuple harder to read at the call sites.
///
/// `dim_names` and `type_names` are the same fact on the two binder kinds,
/// and both exist for the same reason: the names are in scope only while the
/// declaration's signature is being resolved, and the diagnostics that need
/// them, the borrow report and the [04-INF-6] rigidity check, both run after
/// instantiation, where only the internal ids survive.
pub(super) struct ResolvedDeclaredType {
    pub(super) ty: Type,
    pub(super) bounds: Vec<(TypeVar, TypeVarRestriction)>,
    pub(super) dim_names: UnordMap<DimVar, String>,
    pub(super) type_names: UnordMap<TypeVar, String>,
}

pub(super) struct RejectedDeclaredType {
    pub(super) recovery: crate::deep_type::RejectedSignatureType,
    pub(super) bounds: Vec<(TypeVar, TypeVarRestriction)>,
    pub(super) dim_names: UnordMap<DimVar, String>,
    pub(super) type_names: UnordMap<TypeVar, String>,
}

pub(super) fn resolve_deep_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    use_site: TypeUseSite,
    binder_mode: BinderMode<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> Result<Type, ErrorWitness> {
    let resolved = resolve_deep_type_with_bounds_and_dim_names(
        expr,
        vg,
        adt_reg,
        use_site,
        binder_mode,
        UnordMap::new(),
        None,
        errors,
    )
    .map_err(|rejected| rejected.recovery.witness)?;
    Ok(resolved.ty)
}

/// Resolve a declaration's type expression under declared dtype-family bounds
/// (`spec/04-type-system.md` §5.9), additionally returning the source name
/// bound to each dimension variable (chelis#260) and to each authored type
/// binder (chelis#1486) the resolution minted.
///
/// The bounds arrive from the declaration node's `dtype_bounds` metadata and
/// leave as `(variable, family)` pairs the caller installs on the
/// substitution, so generalization re-quantifies them onto the scheme. The
/// resolver holds `name -> DimVar` only for its own lifetime; every other
/// caller drops it, which is why a declared-dim collapse could report `d44`
/// and `d45` but never `n` and `m`.
///
/// One call returns both because one call site needs both: the `Defsig` arm
/// installs the bounds and records the names for the same declaration. Two
/// entry points that each ran the resolver would resolve the declaration
/// twice and leave two mechanisms to keep in step.
#[allow(clippy::too_many_arguments)]
pub(super) fn resolve_deep_type_with_bounds_and_dim_names(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    use_site: TypeUseSite,
    binder_mode: BinderMode<'_>,
    dtype_bounds: UnordMap<String, TypeVarRestriction>,
    diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
    errors: &mut DiagnosticSink<'_>,
) -> Result<ResolvedDeclaredType, Box<RejectedDeclaredType>> {
    let (resolution, bound_result, bounds, dim_names, type_names) = {
        let mut resolver =
            DeepTypeResolver::new(use_site, binder_mode, adt_reg.resolution_env(), vg, errors)
                .with_declaration_diagnostic_owner(diagnostic_owner)
                .with_dtype_bounds(dtype_bounds);
        let resolution = resolver.resolve_signature(expr);
        let dim_names = resolver.dim_var_names();
        let type_names = resolver.type_var_names();
        let bounds = resolver.resolved_dtype_bounds();
        let bound_result = resolver.finish_dtype_bounds();
        (resolution, bound_result, bounds, dim_names, type_names)
    };
    let recovery = match (resolution, bound_result) {
        (Ok(resolved), Ok(bounds)) => {
            return Ok(ResolvedDeclaredType {
                ty: resolve_type_aliases(&resolved.into_type(), adt_reg, vg),
                bounds,
                dim_names,
                type_names,
            });
        }
        (Ok(resolved), Err(witness)) => {
            crate::deep_type::RejectedSignatureType::from_resolved(resolved.into_type(), witness)
        }
        // Both boundaries already reported their independent failures. The
        // first witness marks the rejected declaration without another report.
        (Err(recovery), _) => recovery,
    };
    let mut recovery = recovery;
    recovery.ty = resolve_type_aliases(&recovery.ty, adt_reg, vg);
    Err(Box::new(RejectedDeclaredType {
        recovery,
        bounds,
        dim_names,
        type_names,
    }))
}

/// Split canonical Deep `defsig` children. Monomorphic declarations have
/// `(name, type)`; polymorphic declarations have `(name, binder-list, type)`.
pub(super) fn defsig_parts(
    children: &[deep::Expr],
) -> Option<(&deep::Expr, Option<&deep::Expr>, &deep::Expr)> {
    match children {
        [name, ty] => Some((name, None, ty)),
        [name, binders, ty] => Some((name, Some(binders), ty)),
        _ => None,
    }
}

fn binder_list_items(expr: &deep::Expr) -> Option<&[deep::Expr]> {
    match expr {
        deep::Expr::BareList(items, _) => Some(items),
        deep::Expr::List(list, _) if get_tag(list).is_none() => Some(&list.elements),
        _ => None,
    }
}

pub(super) fn valid_defsig_binder_names(
    binder_list: Option<&deep::Expr>,
) -> Option<UnordSet<String>> {
    let Some(binder_list) = binder_list else {
        return Some(UnordSet::new());
    };
    let items = binder_list_items(binder_list)?;
    if items.is_empty() {
        return None;
    }
    let mut names = UnordSet::new();
    for item in items {
        let name = symbol_name(item)?;
        if crate::deep_type::is_forbidden_dtype_binder_name(name) {
            return None;
        }
        if !names.insert(name.to_string()) {
            return None;
        }
    }
    Some(names)
}

/// Decode one `defsig`'s explicit unkinded binder list. Absence is the empty
/// set. Malformed or duplicate entries are rejected as one declaration error.
pub(super) fn defsig_binder_names(
    binder_list: Option<&deep::Expr>,
    errors: &mut DiagnosticSink<'_>,
) -> Option<UnordSet<String>> {
    let Some(binder_list) = binder_list else {
        return Some(UnordSet::new());
    };
    let Some(items) = binder_list_items(binder_list) else {
        errors.push(CheckError::new(
            CheckErrorKind::MalformedForm,
            "malformed `defsig` binder list: expected a nonempty structural list of distinct symbol names"
                .to_string(),
            vec![],
        ));
        return None;
    };
    if items.is_empty() {
        errors.push(CheckError::new(
            CheckErrorKind::MalformedForm,
            "malformed `defsig` binder list: an empty list must be omitted".to_string(),
            vec!["write the monomorphic two-child form `(defsig {} name type-expr)`".to_string()],
        ));
        return None;
    }
    let mut names = UnordSet::new();
    for item in items {
        let Some(name) = symbol_name(item) else {
            errors.push(CheckError::new(
                CheckErrorKind::MalformedForm,
                "malformed `defsig` binder list: expected a nonempty structural list of distinct symbol names"
                    .to_string(),
                vec![],
            ));
            return None;
        };
        if crate::deep_type::is_forbidden_dtype_binder_name(name) {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "dtype spelling `{name}` cannot be a `defsig` binder: active, reserved, retired, and deferred dtype vocabulary is not rebindable"
                ),
                vec![format!(
                    "remove `{name}` from the binder list and use its canonical dtype meaning, or choose an intentional non-dtype binder name"
                )],
            ));
            return None;
        }
        if !names.insert(name.to_string()) {
            errors.push(CheckError::new(
                CheckErrorKind::DuplicateDefinition,
                format!("duplicate `defsig` binder `{name}`"),
                vec![format!("list `{name}` exactly once")],
            ));
            return None;
        }
    }
    Some(names)
}

/// Decode a declaration node's `dtype_bounds` metadata into checker
/// restrictions, reporting a malformed bound rather than dropping it.
pub(super) fn declaration_dtype_bounds(
    meta: &deep::Metadata,
) -> UnordMap<String, TypeVarRestriction> {
    chelis_deep::decode_dtype_bounds(meta)
        .into_iter()
        .map(|(binder, family)| (binder, restriction_for_family(family)))
        .collect()
}

/// Attach a declaration's resolved bounds to the substitution so
/// generalization re-quantifies them onto the scheme.
pub(super) fn install_declared_bounds(
    bounds: &[(TypeVar, TypeVarRestriction)],
    subst: &mut Subst,
    declaration: &str,
    errors: &mut DiagnosticSink<'_>,
) -> Result<(), ErrorWitness> {
    for (variable, restriction) in bounds {
        if let Err(error) = subst.narrow_tvar_restriction(*variable, *restriction) {
            return Err(report_witness(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "`{declaration}` declares conflicting dtype bounds: {}",
                        error.message
                    ),
                    vec![],
                ),
            ));
        }
    }
    Ok(())
}

/// The checker restriction for a surface dtype family. Total over the closed
/// `spec/04-type-system.md` §5.9 family set.
pub(super) fn restriction_for_family(family: chelis_deep::DtypeFamily) -> TypeVarRestriction {
    match family {
        chelis_deep::DtypeFamily::Float => TypeVarRestriction::ActiveFloat,
        chelis_deep::DtypeFamily::Int => TypeVarRestriction::ActiveInt,
        chelis_deep::DtypeFamily::Numeric => TypeVarRestriction::ActiveNumeric,
    }
}
