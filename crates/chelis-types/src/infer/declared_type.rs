//! Declared type resolution, explicit binder lists, and dtype bounds.
//!
//! These helpers own the boundary between declaration syntax and the
//! checker's resolved binder state. Keeping them together prevents ordinary
//! expression inference from becoming a second declaration owner.

use super::*;

/// A declaration's resolved type together with its dtype-family bounds and one
/// role-complete binder-identity object.
///
/// A struct rather than a tuple because this has now grown twice: chelis#1474
/// added the dtype-family bounds and chelis#260 added the dimension names,
/// each time making an unnamed tuple harder to read at the call sites.
///
/// The identities are in scope only while the declaration's signature is
/// being resolved. The ordinary body annotation resolvers and [04-INF-6]
/// rigidity checks run after instantiation, so all three variable kinds must
/// cross that boundary together.
pub(super) struct ResolvedDeclaredType {
    pub(super) ty: Type,
    pub(super) bounds: Vec<(TypeVar, TypeVarRestriction)>,
    pub(super) binder_identities: DeclarationBinderIdentities,
}

pub(super) struct RejectedDeclaredType {
    pub(super) recovery: crate::deep_type::RejectedSignatureType,
    pub(super) bounds: Vec<(TypeVar, TypeVarRestriction)>,
    pub(super) binder_identities: DeclarationBinderIdentities,
}

pub(super) fn resolve_deep_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    use_site: TypeUseSite,
    binder_mode: BinderMode<'_>,
    errors: &mut DiagnosticSink<'_>,
) -> Result<Type, ErrorWitness> {
    resolve_deep_type_with_diagnostic_owner(expr, vg, adt_reg, use_site, binder_mode, None, errors)
}

pub(super) fn resolve_deep_type_with_diagnostic_owner(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    use_site: TypeUseSite,
    binder_mode: BinderMode<'_>,
    diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
    errors: &mut DiagnosticSink<'_>,
) -> Result<Type, ErrorWitness> {
    let resolved = resolve_deep_type_with_binder_identities(
        expr,
        vg,
        adt_reg,
        use_site,
        binder_mode,
        UnordMap::new(),
        diagnostic_owner,
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
pub(super) fn resolve_deep_type_with_binder_identities(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    use_site: TypeUseSite,
    binder_mode: BinderMode<'_>,
    dtype_bounds: UnordMap<String, TypeVarRestriction>,
    diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
    errors: &mut DiagnosticSink<'_>,
) -> Result<ResolvedDeclaredType, Box<RejectedDeclaredType>> {
    let (resolution, bound_result, bounds, binder_identities) = {
        let mut resolver =
            DeepTypeResolver::new(use_site, binder_mode, adt_reg.resolution_env(), vg, errors)
                .with_declaration_diagnostic_owner(diagnostic_owner)
                .with_dtype_bounds(dtype_bounds);
        let resolution = resolver.resolve_signature(expr);
        let binder_identities = resolver.binder_identities();
        let bounds = resolver.resolved_dtype_bounds();
        let bound_result = resolver.finish_dtype_bounds();
        (resolution, bound_result, bounds, binder_identities)
    };
    let recovery = match (resolution, bound_result) {
        (Ok(resolved), Ok(bounds)) => {
            return Ok(ResolvedDeclaredType {
                ty: resolve_type_aliases(&resolved.into_type(), adt_reg, vg),
                bounds,
                binder_identities,
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
        binder_identities,
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
        .map(|(binder, bound)| (binder, restriction_for_bound(&bound)))
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

/// The checker restriction for either `spec/04-type-system.md` §5.9 bound
/// form. A family keeps its §1.1-tracking predicate; a set becomes the
/// corresponding [`PrimSet`], so the checker never turns one into the other.
pub(super) fn restriction_for_bound(bound: &chelis_deep::DtypeBound) -> TypeVarRestriction {
    match bound {
        chelis_deep::DtypeBound::Family(family) => restriction_for_family(*family),
        chelis_deep::DtypeBound::Set(members) => TypeVarRestriction::ActiveSet(
            crate::types::PrimSet::from_members(members.iter().map(|dtype| match dtype {
                chelis_deep::BoundDtype::F32 => crate::types::Prim::F32,
                chelis_deep::BoundDtype::F64 => crate::types::Prim::F64,
                chelis_deep::BoundDtype::Bf16 => crate::types::Prim::Bf16,
                chelis_deep::BoundDtype::F16 => crate::types::Prim::F16,
                chelis_deep::BoundDtype::I8 => crate::types::Prim::Int8,
                chelis_deep::BoundDtype::I16 => crate::types::Prim::Int16,
                chelis_deep::BoundDtype::I32 => crate::types::Prim::Int32,
                chelis_deep::BoundDtype::I64 => crate::types::Prim::Int64,
            })),
        ),
    }
}
