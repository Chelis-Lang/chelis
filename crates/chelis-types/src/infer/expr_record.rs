//! Tuple, record, access, update, and cast inference.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;
use chelis_deep::CastMode;

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_tuple(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    let elems: Vec<Type> = kids
        .iter()
        .map(|e| infer_expr(e, env, vg, subst, adt_reg, errors, product))
        .collect();
    Type::Tuple(elems)
}

/// Extract a non-negative tuple-projection index from a Deep index
/// node. Accepts a bare `Int` atom (hand-written Deep) and a `lit`
/// node wrapping an `Int` atom (the Surf `.N` desugar, chelis#707).
/// Returns `None` for any other shape or a negative literal, which the
/// sole caller maps to `Type::Error`.
pub(super) fn tuple_get_index(expr: &deep::Expr) -> Option<usize> {
    match expr {
        deep::Expr::Atom(deep::Atom::Int(n), _) => usize::try_from(*n).ok(),
        deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::Lit) => {
            match children(list).first() {
                Some(deep::Expr::Atom(deep::Atom::Int(n), _)) => usize::try_from(*n).ok(),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Human description of a malformed tuple-projection index, for the
/// diagnostic the sole caller pushes when `tuple_get_index` returns
/// `None`. Peeks through a `lit` wrapper to the payload atom.
pub(super) fn describe_tuple_index(expr: &deep::Expr) -> String {
    let atom = match expr {
        deep::Expr::List(list, _) if get_tag(list) == Some(DeepTag::Lit) => children(list).first(),
        other => Some(other),
    };
    match atom {
        Some(deep::Expr::Atom(deep::Atom::Int(n), _)) => format!("integer literal {n}"),
        Some(deep::Expr::Atom(deep::Atom::Float(f), _)) => format!("float literal {f}"),
        Some(deep::Expr::Atom(deep::Atom::Bool(b), _)) => format!("bool literal {b}"),
        Some(deep::Expr::Atom(deep::Atom::Str(_), _)) => "a string literal".to_string(),
        Some(deep::Expr::Atom(deep::Atom::Name(s), _)) => format!("symbol `{s}`"),
        _ => "a non-literal expression".to_string(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_tuple_get(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return malformed_form(list, "tuple-get", "a tuple expression and an index", errors);
    }

    let tuple_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let resolved = subst.apply(&tuple_ty);

    // The Surf `.N` desugar emits the projection index as a `lit` node
    // `(lit {type: int32} <Int>)` (`desugar.rs`, `Expr::TupleGet`),
    // while hand-written Deep may carry it as a bare `Int` atom. Read
    // the index from either shape. Matching only the bare atom made
    // every Surf-level `.N` projection fall through to `Type::Error`,
    // silently erasing the element type — `Type::Error` then unifies
    // with anything, so a value derived from a tuple projection lost
    // its nominal type at every downstream boundary (chelis#707).
    let index = match tuple_get_index(&kids[1]) {
        Some(index) => index,
        None => {
            // A malformed index (negative, float, symbol, or any
            // non-literal) is not a valid projection. Diagnose it
            // rather than returning a silent `Type::Error`: a bare
            // negative `Int` used to blow up as `-1 as usize` into a
            // loud out-of-bounds error, and every other shape was
            // silently swallowed — both are undiagnosed `Type::Error`
            // under an empty error vector, the §04-TOT-2 hole this fix
            // otherwise closes (chelis#707, rt-707).
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TupleIndexOutOfBounds,
                    format!(
                        "invalid tuple index: expected a non-negative integer \
                     literal, found {}",
                        describe_tuple_index(&kids[1]),
                    ),
                    vec![],
                ),
            );
        }
    };
    match resolved {
        Type::Tuple(ref elems) => {
            if index < elems.len() {
                elems[index].clone()
            } else {
                report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TupleIndexOutOfBounds,
                        format!(
                            "tuple index {} out of bounds for tuple of size {}",
                            index,
                            elems.len()
                        ),
                        vec![],
                    ),
                )
            }
        }
        Type::Error(w) => propagate(&w),
        // The target is not yet known to be a tuple — e.g. an
        // unannotated closure parameter (a `fold` accumulator)
        // constrained only structurally by its own projections. This
        // type system has no open/row-polymorphic tuple, so we cannot
        // pin the arity from one projection; defer by handing back a
        // fresh element type instead of committing to "not a tuple".
        // Before chelis#707 this path was unreachable (the index never
        // parsed, so every projection returned `Type::Error`), so this
        // preserves the prior permissiveness for genuinely-unresolved
        // targets while the `Tuple` arm now carries the real element
        // type for concrete tuples.
        Type::Var(_) => {
            let projected = vg.fresh_type();
            product.derive_shape_lambda_type(&tuple_ty, &projected, subst);
            product.defer_tuple_projection(tuple_ty, index, projected.clone());
            projected
        }
        _ => report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expected tuple type, got {resolved}"),
                vec![],
            ),
        ),
    }
}

/// Resolve a record head name to its constructor. The head is a
/// variant name (`Probability { ... }`); when it names a transparent
/// type alias instead, resolve through the alias to the nominal ADT
/// and use that ADT's same-named variant (alias transparency,
/// spec/02; this is also what keeps alias laundering from bypassing
/// opacity, RFC D-CHECK). Returns the canonical constructor name.
pub(super) fn resolve_record_head<'a>(
    head: &'a str,
    adt_reg: &'a AdtRegistry,
) -> Option<(&'a str, &'a crate::adt::VariantInfo, String)> {
    if let Some((adt_name, variant)) = adt_reg
        .lookup_variant_preferring_shape(head, CallShape::Record)
        .or_else(|| adt_reg.lookup_variant_terminal_unique(head))
    {
        return Some((adt_name, variant, variant.name.clone()));
    }
    // Alias head: `type P2 = Probability` makes `P2 { ... }` mean
    // `Probability { ... }`.
    let alias = adt_reg.resolve_alias(head)?;
    if let Type::Adt(target, _) = &alias.body {
        let (adt_name, variant) = adt_reg.lookup_variant(target)?;
        return Some((adt_name, variant, variant.name.clone()));
    }
    None
}

/// Infer `(record {} Ctor (kv {} field value)...)` — named-field
/// record construction (RFC D-CHECK prerequisite inference; closes
/// the latent bogus-field hole: unknown fields are now TypeMismatch
/// errors instead of silently untyped).
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_record(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    let Some(head) = kids.first().and_then(symbol_name) else {
        return malformed_form(
            list,
            "record",
            "a symbol constructor head as its first child",
            errors,
        );
    };

    let Some((adt_name, variant, ctor_name)) = resolve_record_head(head, adt_reg) else {
        // Infer field values so nested errors still surface, then
        // reject the unknown constructor.
        for kv_expr in kids.iter().skip(1) {
            if let deep::Expr::List(kv_list, _) = kv_expr
                && get_tag(kv_list) == Some(DeepTag::Kv)
                && let Some(value) = children(kv_list).get(1)
            {
                infer_expr(value, env, vg, subst, adt_reg, errors, product);
            }
        }
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("unknown record constructor `{head}`"),
                vec![format!("declare `type {head} = | {head} {{ ... }}`")],
            ),
        );
    };
    // chelis#317: a registry-known but OUT-OF-SCOPE record constructor (a
    // type-only import constructing e.g. `AdamState { ... }`, whose ADT is in
    // the registry but whose constructor is not in scope at the use site) is
    // unknown here. Opaque types are handled by the opacity check below
    // instead, so only a non-opaque out-of-scope head is rejected here; this
    // keeps the #317 record-constructor guard while leaving opacity rejection
    // (D-CHECK) for opaque heads.
    let head_is_opaque = adt_reg.lookup(adt_name).is_some_and(|d| d.opaque);
    if !head_is_opaque && constructor_out_of_scope(head, env) {
        for kv_expr in kids.iter().skip(1) {
            if let deep::Expr::List(kv_list, _) = kv_expr
                && get_tag(kv_list) == Some(DeepTag::Kv)
                && let Some(value) = children(kv_list).get(1)
            {
                infer_expr(value, env, vg, subst, adt_reg, errors, product);
            }
        }
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::UnknownConstructor {
                    identifier: head.to_string(),
                },
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("unknown constructor: {head}"),
                ),
                vec![format!(
                    "Constructor '{head}' is not in scope. Declare it locally or \
                 add it to an import (e.g. `import Mod ({head})`)"
                )],
            ),
        );
    }
    // RFC D-CHECK: record construction of an out-of-module opaque
    // type is rejected; inference continues so the literal still
    // yields its true type (no cascades).
    crate::opacity::check_opaque_use(
        crate::opacity::OpaqueAction::RecordConstruction,
        adt_name,
        adt_reg,
        errors,
    );

    let declared_field_names: Vec<Option<String>> =
        variant.fields.iter().map(|(n, _)| n.clone()).collect();
    let known_field_set: HashSet<&str> = declared_field_names
        .iter()
        .filter_map(|n| n.as_deref())
        .collect();

    // Instantiate the resolved ADT's constructor directly from its
    // registry definition (the issue #181 pat-record intent, made
    // collision-proof): the name-keyed env holds ONE scheme per
    // constructor name, so same-named constructors from colliding
    // ADTs (chelis#148) would dispatch the field types to whichever
    // deftype registered last.
    let (instantiated_arg_types, instantiated_ret) = match adt_reg.lookup(adt_name) {
        Some(adt_def) => instantiate_variant_of(adt_def, variant, vg),
        None => (Vec::new(), vg.fresh_type()),
    };

    for kv_expr in kids.iter().skip(1) {
        // chelis#1107: read the `kv` through `stamped_parts`, which accepts
        // both carriers. `infer_expr`'s Node bridge rebuilds only the record
        // node itself (`Node::to_list` is shallow), so on the stamped ingress
        // every `kv` child arrives as `Expr::Node`. A `List`-only destructure
        // sent all of them to `continue`, and NO field of a record literal was
        // type-checked at all -- a bool into an f32 field checked clean while
        // `check_ir_program`, which normalizes Node to List first, rejected it.
        let Some((kv_tag, _, kv_kids)) = stamped_parts(kv_expr) else {
            continue;
        };
        if kv_tag != DeepTag::Kv {
            continue;
        }
        let (Some(field_name), Some(value)) =
            (kv_kids.first().and_then(symbol_name), kv_kids.get(1))
        else {
            continue;
        };
        let value_ty = infer_expr(value, env, vg, subst, adt_reg, errors, product);
        if known_field_set.contains(field_name) {
            let pos = declared_field_names
                .iter()
                .position(|n| n.as_deref() == Some(field_name));
            if let Some(field_ty) = pos.and_then(|i| instantiated_arg_types.get(i))
                && let Err(te) = unify(&value_ty, field_ty, subst)
            {
                errors.push(te.into());
            }
        } else {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("unknown record field '{field_name}' in construction of {ctor_name}"),
                vec![format!(
                    "known fields: {:?}",
                    declared_field_names
                        .iter()
                        .filter_map(|f| f.as_deref())
                        .collect::<Vec<_>>()
                )],
            ));
        }
    }

    subst.apply(&instantiated_ret)
}

/// The single record-shaped variant of an ADT, when it has exactly
/// one variant and every field is named — the representation idiom
/// `access`/`record-update` resolve against.
pub(super) fn single_record_variant<'a>(
    adt_reg: &'a AdtRegistry,
    adt_name: &str,
) -> Option<&'a crate::adt::VariantInfo> {
    let def = adt_reg.lookup(adt_name)?;
    if def.variants.len() != 1 {
        return None;
    }
    let variant = &def.variants[0];
    (!variant.fields.is_empty() && variant.fields.iter().all(|(n, _)| n.is_some()))
        .then_some(variant)
}

/// Instantiate `variant` of `adt_def` with fresh variables of every kind:
/// returns the per-field types and the ADT result type with the definition's
/// registration-time variables renamed fresh. Bypasses the name-keyed env so
/// same-named constructors from colliding ADTs (chelis#148) cannot cross-wire
/// field types.
///
/// Dimension, rank, and precision variables in a variant field are quantified
/// by the constructor scheme just like nominal type parameters. Reusing their
/// registration-time IDs here would let a constructor occurrence escape its
/// ordinary inference level and make an otherwise generic imported signature
/// monomorphic (chelis#1207, exposed by chelis#968's stacked-context oracle).
pub(super) fn instantiate_variant_of(
    adt_def: &crate::adt::AdtDef,
    variant: &crate::adt::VariantInfo,
    vg: &mut VarGen,
) -> (Vec<Type>, Type) {
    let mut type_vars = adt_def.param_vars.clone();
    let mut dim_vars = Vec::new();
    let mut rank_vars = Vec::new();
    for (_, field_type) in &variant.fields {
        for var in crate::env::free_tvars(field_type) {
            if !type_vars.contains(&var) {
                type_vars.push(var);
            }
        }
        for var in crate::env::free_dvars(field_type) {
            if !dim_vars.contains(&var) {
                dim_vars.push(var);
            }
        }
        for var in crate::env::free_rvars(field_type) {
            if !rank_vars.contains(&var) {
                rank_vars.push(var);
            }
        }
    }

    let mut renaming = Subst::new();
    for var in type_vars {
        renaming
            .insert_type(var, vg.fresh_type())
            .expect("fresh constructor-field type renaming is valid");
    }
    for var in dim_vars {
        renaming.insert_dim(var, vg.fresh_dim());
    }
    for var in rank_vars {
        renaming.insert_rank(var, vec![Dim::Rank(vg.fresh_rvar())]);
    }

    let args: Vec<Type> = variant
        .fields
        .iter()
        .map(|(_, field_type)| renaming.apply(field_type))
        .collect();
    let ret = Type::Adt(
        adt_def.name.clone(),
        adt_def
            .param_vars
            .iter()
            .map(|var| renaming.apply(&Type::Var(*var)))
            .collect(),
    );
    (args, ret)
}

/// Instantiate the single record variant of the ADT named by
/// `target_ty` and unify the instantiated result with the target,
/// returning the per-field types aligned with `variant.fields` so
/// they reflect the target's concrete type arguments.
pub(super) fn instantiated_field_types(
    adt_name: &str,
    variant: &crate::adt::VariantInfo,
    target_ty: &Type,
    adt_reg: &AdtRegistry,
    vg: &mut VarGen,
    subst: &mut Subst,
) -> Vec<Type> {
    let Some(adt_def) = adt_reg.lookup(adt_name) else {
        return Vec::new();
    };
    let (args, ret) = instantiate_variant_of(adt_def, variant, vg);
    let _ = unify(&ret, target_ty, subst);
    args
}

/// Infer `(access {} target field)` — record field access (RFC
/// D-CHECK prerequisite inference).
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_access(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return malformed_form(
            list,
            "access",
            "a target expression and a field name",
            errors,
        );
    }
    let target_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let Some(field_name) = symbol_name(&kids[1]) else {
        return malformed_form(
            list,
            "access",
            "a symbol field name as its second child",
            errors,
        );
    };
    // Peel borrow layers: an `&T` target reads through the borrow.
    let mut resolved = subst.apply(&target_ty);
    while let Type::Ref(inner) = resolved {
        resolved = *inner;
    }
    match resolved {
        Type::Adt(ref adt_name, _) => {
            // RFC D-CHECK: field access on an out-of-module opaque
            // type is rejected; inference continues so the access
            // still yields its true field type (no cascades).
            crate::opacity::check_opaque_use(
                crate::opacity::OpaqueAction::FieldAccess,
                adt_name,
                adt_reg,
                errors,
            );
            let Some(variant) = single_record_variant(adt_reg, adt_name) else {
                // chelis#755 (discovered-hole conversion, chelis#731 Phase 2):
                // field access on a multi-variant or positional-field ADT is
                // not defined. This used to return a SILENT `Type::Error`
                // ("conservative status quo") so `s.radius` on a two-variant
                // `Shape` scored a perfect 1.0 and only failed at runtime
                // (`unknown record constructor`). It is now an explicit
                // rejection with a diagnostic (spec/design/checker_totality.md
                // §C3; the sanctioned direction from chelis#755).
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!(
                            "field access `.{field_name}` is only defined on a \
                             single-record-variant type; `{adt_name}` is a \
                             multi-variant or positional-field type (chelis#755)"
                        ),
                        vec![
                            "pattern-match on the variants with `match` to read \
                             their fields"
                                .to_string(),
                        ],
                    ),
                );
            };
            let pos = variant
                .fields
                .iter()
                .position(|(n, _)| n.as_deref() == Some(field_name));
            let Some(pos) = pos else {
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!("unknown record field '{field_name}' on {adt_name}"),
                        vec![format!(
                            "known fields: {:?}",
                            variant
                                .fields
                                .iter()
                                .filter_map(|(n, _)| n.as_deref())
                                .collect::<Vec<_>>()
                        )],
                    ),
                );
            };
            let field_types =
                instantiated_field_types(adt_name, variant, &resolved, adt_reg, vg, subst);
            match field_types.get(pos) {
                Some(ty) => subst.apply(ty),
                // `pos` came from a validated `.position(...)` hit, so the
                // field-type list should always be at least that long; a
                // shorter list is an internal inconsistency between the
                // variant's field names and its instantiated field types.
                // Report it loudly rather than exempt the access silently
                // (chelis#731 §C3).
                None => report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!(
                            "internal: field `{field_name}` of `{adt_name}` resolved \
                             to position {pos} but no instantiated field type is \
                             available (chelis#731 [04-TOT-2])"
                        ),
                        vec![],
                    ),
                ),
            }
        }
        Type::Var(tv) => {
            // Target not yet pinned (e.g. unannotated lambda param):
            // register in the deferred-access ledger so a later pin to an
            // out-of-module opaque ADT is still rejected at def-level
            // resolution (D-CHECK). chelis#755 / chelis#731 Phase 2: the
            // result is now a FRESH type variable, not a silent `Type::Error`
            // -- the access type is genuinely "unknown until the target is
            // pinned", an explicit typed rule (the alternative chelis#755
            // sanctions) rather than an exemption. Unification narrows it, and
            // the deferred ledger still catches an out-of-module opaque pin.
            subst.record_deferred_opaque_use(tv, crate::unify::DeferredOpaqueUse::Access);
            vg.fresh_type()
        }
        // chelis#755 (discovered-hole conversion, chelis#731 Phase 2): field
        // access on a non-record value (a tensor, prim, tuple, function, ...)
        // used to return a SILENT `Type::Error` so `x.field` on an `f32`
        // scored 1.0 and only failed at runtime. It is now an explicit
        // rejection with a diagnostic.
        other => report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "field access `.{field_name}` expects a record value, got a \
                     value of type `{other}` (chelis#755)"
                ),
                vec![],
            ),
        ),
    }
}

/// Infer `(record-update {} target (kv {} field value)...)` — Deep
/// functional record update (RFC D-CHECK prerequisite inference).
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_record_update(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return malformed_form(
            list,
            "record-update",
            "a target expression to update",
            errors,
        );
    }
    let target_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    // Infer the update values regardless of target resolution so
    // nested errors surface exactly once.
    let mut kv_pairs: Vec<(&str, Type)> = Vec::new();
    for kv_expr in kids.iter().skip(1) {
        // chelis#1107: same carrier-preserving read as `infer_record` above --
        // a stamped `kv` arrives as `Expr::Node` and a `List`-only destructure
        // skipped every update field.
        let Some((kv_tag, _, kv_kids)) = stamped_parts(kv_expr) else {
            continue;
        };
        if kv_tag != DeepTag::Kv {
            continue;
        }
        let (Some(field_name), Some(value)) =
            (kv_kids.first().and_then(symbol_name), kv_kids.get(1))
        else {
            continue;
        };
        let value_ty = infer_expr(value, env, vg, subst, adt_reg, errors, product);
        kv_pairs.push((field_name, value_ty));
    }
    let mut resolved = subst.apply(&target_ty);
    while let Type::Ref(inner) = resolved {
        resolved = *inner;
    }
    match resolved {
        Type::Adt(ref adt_name, _) => {
            // RFC D-CHECK: record update of an out-of-module opaque
            // type is rejected; inference continues and returns the
            // target's true type (no cascades).
            crate::opacity::check_opaque_use(
                crate::opacity::OpaqueAction::RecordUpdate,
                adt_name,
                adt_reg,
                errors,
            );
            let Some(variant) = single_record_variant(adt_reg, adt_name) else {
                // chelis#755 sibling (record-update, chelis#731 Phase 2):
                // functional record update on a multi-variant / positional
                // ADT is not defined; this used to return a silent
                // `Type::Error`. Reject it with a diagnostic like the field
                // access path.
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!(
                            "record update is only defined on a single-record-variant \
                             type; `{adt_name}` is a multi-variant or positional-field \
                             type (chelis#755)"
                        ),
                        vec![],
                    ),
                );
            };
            let field_types =
                instantiated_field_types(adt_name, variant, &resolved, adt_reg, vg, subst);
            for (field_name, value_ty) in &kv_pairs {
                let pos = variant
                    .fields
                    .iter()
                    .position(|(n, _)| n.as_deref() == Some(*field_name));
                match pos {
                    Some(pos) => {
                        if let Some(field_ty) = field_types.get(pos)
                            && let Err(te) = unify(value_ty, field_ty, subst)
                        {
                            errors.push(te.into());
                        }
                    }
                    None => {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!("unknown record field '{field_name}' on {adt_name}"),
                            vec![format!(
                                "known fields: {:?}",
                                variant
                                    .fields
                                    .iter()
                                    .filter_map(|(n, _)| n.as_deref())
                                    .collect::<Vec<_>>()
                            )],
                        ));
                    }
                }
            }
            subst.apply(&resolved)
        }
        Type::Var(tv) => {
            subst.record_deferred_opaque_use(tv, crate::unify::DeferredOpaqueUse::RecordUpdate);
            // The update returns the target's (still-unresolved) type.
            Type::Var(tv)
        }
        Type::Error(w) => propagate(&w),
        other => report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("record-update requires a record-typed target, got {other}"),
                vec![],
            ),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_cast(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return malformed_form(list, "cast", "an expression and a target type", errors);
    }
    let mode = match deep::cast_mode_of(kids) {
        Ok(mode) => mode,
        Err(selector) => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::CastNonTensor,
                    format!("`{selector}` is not a recognized cast mode selector"),
                    vec![
                        "the only named cast rung is `trunc` (`cast_trunc`, \
                         [05-OP-6]); omit the selector for the checked default"
                            .to_string(),
                    ],
                ),
            );
        }
    };

    let expr_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);
    let resolved = match subst.apply(&expr_ty) {
        Type::Var(v) => match subst.materialize_deferred_expand_default(v) {
            Ok(Some(ty)) => ty,
            Ok(None) => Type::Var(v),
            Err(error) => return report(errors, error.into()),
        },
        other => other,
    };

    // Every target spelling first crosses the centralized resolver. Bare
    // primitive symbols are retained for historical compatibility; canonical
    // `t-prim` still goes through the resolver's metadata and exact-arity
    // checks before semantic cast classification.
    let cast_owner = deep::Expr::List(list.clone(), span_of_list(list));
    let (resolved_target, target_location) = {
        let mut resolver = DeepTypeResolver::new(
            TypeUseSite::CastTarget,
            annotation_binder_mode(env),
            adt_reg.resolution_env(),
            vg,
            errors,
        )
        .with_diagnostic_owner(&cast_owner);
        let target = match resolver.resolve_cast_target(&kids[1]) {
            Ok(target) => target,
            Err(witness) => return propagate(&witness),
        };
        (target, resolver.diagnostic_location())
    };
    let target_ty = match resolved_target {
        ResolvedCastTarget::PrimitiveSpelling { name, canonical } => {
            // A1 (WS-A0 RT-1 fixup): per spec/04-type-system.md §1.1.1,
            // the unsigned family and the other reserved-but-deferred
            // dtype names are rejected with the owning precision
            // diagnostic after syntax has validated.
            if let Some(diag) = unsigned_family_diagnostic(&name, /* tensor = */ false)
                .or_else(|| deferred_family_diagnostic(&name, /* tensor = */ false))
            {
                let diag = target_location
                    .as_ref()
                    .map_or(diag.clone(), |location| location.attach(diag));
                return report(errors, diag);
            }
            match Prim::parse_name(&name) {
                Some(prim) => Type::Prim(prim),
                // RFC D-CHECK's historical Deep compatibility spelling uses
                // `(t-prim {} Nominal)` for cast-into probes. It is eligible
                // only after exact canonical syntax validation and only for a
                // registered zero-arity ADT or transparent alias target.
                None if canonical => match cast_target_nominal_name(&name, adt_reg) {
                    Some(target) => Type::Adt(target, Vec::new()),
                    None => {
                        return report_unknown_cast_target(errors, &name, target_location.as_ref());
                    }
                },
                None => {
                    return report_unknown_cast_target(errors, &name, target_location.as_ref());
                }
            }
        }
        ResolvedCastTarget::Type(ty) => resolve_type_aliases(&ty.into_type(), adt_reg, vg),
    };

    // RFC D-CHECK cast gates operate on the same resolved target as ordinary
    // cast typing, including transparent alias expansion.
    if let Type::Adt(target_adt, _) = &target_ty
        && crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::CastInto,
            target_adt,
            adt_reg,
            errors,
        )
    {
        return target_ty;
    }
    let mut peeled = &resolved;
    while let Type::Ref(inner) = peeled {
        peeled = inner.as_ref();
    }
    if let Type::Adt(source_adt, _) = peeled
        && crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::CastOut,
            source_adt,
            adt_reg,
            errors,
        )
    {
        return target_ty;
    }

    let new_prec = match target_ty {
        Type::Var(target) => {
            // A cast inside a declaration may name one of that declaration's
            // quantified scalar type variables. Keep the target symbolic and
            // let the declared signature plus its numeric consumers select the
            // concrete active dtype. This is the source-level spelling needed
            // by [05-OP-35]'s same-p `linspace` and `arange` graphs; a closed
            // cast outside such a declaration still rejects the name in the
            // resolver above.
            return match resolved {
                Type::Prim(source) if source.is_numeric() => Type::Var(target),
                Type::Var(_) | Type::Error(_) => Type::Var(target),
                other => report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::CastNonTensor,
                        format!(
                            "cast to a quantified scalar dtype requires a numeric scalar, got {other}"
                        ),
                        vec![],
                    ),
                ),
            };
        }
        Type::Prim(p) => p,
        other => {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::CastNonTensor,
                    format!(
                        "cast target `{other}` is not a recognized primitive type (chelis#756)"
                    ),
                    vec![
                        "cast targets a scalar primitive: f32, f64, bf16, f16, bool, \
                     int8, int16, int32, int64"
                            .to_string(),
                    ],
                ),
            );
        }
    };

    match resolved {
        Type::Tensor(dims, src_prec) => {
            if !new_prec.is_valid_tensor_precision() {
                return push_unsupported_precision_error(
                    errors, new_prec, /* tensor = */ true,
                );
            }
            if mode == CastMode::Trunc {
                let source = match src_prec {
                    TensorPrec::Concrete(p) => Some(p),
                    TensorPrec::Var(_) => None,
                };
                if let Some(error) = trunc_pair_error(source, new_prec) {
                    return report(errors, error);
                }
            }
            Type::Tensor(dims, TensorPrec::Concrete(new_prec))
        }
        Type::Prim(src_prec) => {
            if !new_prec.is_valid_scalar_cast_target() {
                return push_unsupported_precision_error(
                    errors, new_prec, /* tensor = */ false,
                );
            }
            if mode == CastMode::Trunc
                && let Some(error) = trunc_pair_error(Some(src_prec), new_prec)
            {
                return report(errors, error);
            }
            Type::Prim(new_prec)
        }
        Type::Error(w) => propagate(&w),
        other @ (Type::Fn(_, _)
        | Type::Ref(_)
        | Type::Adt(_, _)
        | Type::Var(_)
        | Type::Tuple(_)
        | Type::Unit) => report(
            errors,
            CheckError::new(
                CheckErrorKind::CastNonTensor,
                format!("cast requires tensor or prim type, got {other}"),
                vec![],
            ),
        ),
    }
}

/// The [05-OP-6] source/target contract: `cast_trunc` is float-to-integer
/// ONLY. Every other pair is a check-time type error naming the checked
/// `cast` as the remedy, so no program reaches a lane that has no
/// truncating semantics for it.
///
/// `source == None` means the operand's tensor precision is still a
/// quantified variable; the pair is re-checked once unification binds it,
/// so accepting it here is not a hole.
pub(super) fn trunc_pair_error(source: Option<Prim>, target: Prim) -> Option<CheckError> {
    let hint = "`cast_trunc` truncates a float toward zero into an integer \
                width ([05-OP-6]); use `cast` for every other conversion"
        .to_string();
    if !target.is_integer() {
        return Some(CheckError::new(
            CheckErrorKind::CastNonTensor,
            format!(
                "`cast_trunc` target `{}` is not an integer dtype",
                target.name()
            ),
            vec![hint],
        ));
    }
    match source {
        Some(prim) if !prim.is_float() => Some(CheckError::new(
            CheckErrorKind::CastNonTensor,
            format!("`cast_trunc` source `{}` is not a float dtype", prim.name()),
            vec![hint],
        )),
        _ => None,
    }
}

pub(super) fn cast_target_nominal_name(name: &str, adt_reg: &AdtRegistry) -> Option<String> {
    if adt_reg
        .lookup(name)
        .is_some_and(|definition| definition.type_params.is_empty())
    {
        return Some(name.to_string());
    }
    adt_reg
        .resolve_alias(name)
        .and_then(|alias| match (&alias.params[..], &alias.body) {
            ([], Type::Adt(target, args)) if args.is_empty() => Some(target.clone()),
            _ => None,
        })
}

pub(super) fn report_unknown_cast_target(
    errors: &mut DiagnosticSink<'_>,
    name: &str,
    location: Option<&TypeDiagnosticLocation>,
) -> Type {
    let error = CheckError::new(
        CheckErrorKind::CastNonTensor,
        format!("cast target `{name}` is not a recognized primitive type (chelis#756)"),
        vec![
            "cast targets a scalar primitive: f32, f64, bf16, f16, bool, \
             int8, int16, int32, int64"
                .to_string(),
        ],
    );
    let error = location.map_or(error.clone(), |location| location.attach(error));
    report(errors, error)
}

/// True if `name` is one of the unsigned integer dtype names reserved
/// as deferred by `spec/04-type-system.md` §1.1.1 (§1.1.2 names the
/// `uint*` spellings canonical; the short `u*` spellings are not
/// reserved). Covers both the short form (`u8`/`u16`/`u32`/`u64`) and
/// the canonical `uint*` family that LLMs and cross-language users
/// tend to write.
pub(super) fn is_unsigned_dtype_name(name: &str) -> bool {
    matches!(
        name,
        "u8" | "u16" | "u32" | "u64" | "uint8" | "uint16" | "uint32" | "uint64"
    )
}

/// Build a §1.1.1 diagnostic for an unsigned dtype name appearing as a
/// cast target or a tensor element type. Returns `None` for non-unsigned
/// names so call sites can short-circuit with `&&`.
pub(super) fn unsigned_family_diagnostic(name: &str, tensor: bool) -> Option<CheckError> {
    if !is_unsigned_dtype_name(name) {
        return None;
    }
    let surface = if tensor { "tensor element" } else { "scalar" };
    let active_set = "f32, f64, bf16, f16, bool, int8, int16, int32, int64";
    Some(CheckError::new(
        CheckErrorKind::UnsupportedTensorPrecision,
        format!(
            "cannot use `{name}` as a {surface} dtype: unsigned integer types \
             are deferred per spec/04-type-system.md §1.1.1 (canonical \
             spelling uint8/uint16/uint32/uint64 per §1.1.2; active set: \
             {active_set})"
        ),
        vec![
            "spec/04-type-system.md §1.1.2 documents the workaround: cast to \
             int32 or int64 and reason at the wider signed precision; or use \
             a tensor of int8 / int16 / int32 / int64 if the bit-width matters"
                .to_string(),
        ],
    ))
}

/// True if `name` is one of the remaining reserved-but-deferred dtype
/// names of `spec/04-type-system.md` §1.1.1 (`f8e4m3` is absent because
/// it is a real `Prim` variant and takes the `Prim::parse_name` path;
/// the unsigned family has its own predicate above). These spellings
/// never resolve through `Prim::parse_name`, so without a dedicated arm
/// they would fall to the generic unknown-name rejections with no
/// §1.1.1 citation.
pub(super) fn is_deferred_dtype_name(name: &str) -> bool {
    matches!(
        name,
        "f8e5m2" | "int4" | "uint4" | "complex64" | "complex128" | "decimal128" | "decimal256"
    )
}

/// Build a §1.1.1 diagnostic for a reserved-but-deferred dtype name
/// appearing as a cast target or a tensor element type. Returns `None`
/// for other names so call sites can short-circuit.
pub(super) fn deferred_family_diagnostic(name: &str, tensor: bool) -> Option<CheckError> {
    if !is_deferred_dtype_name(name) {
        return None;
    }
    let surface = if tensor { "tensor element" } else { "scalar" };
    let active_set = "f32, f64, bf16, f16, bool, int8, int16, int32, int64";
    Some(CheckError::new(
        CheckErrorKind::UnsupportedTensorPrecision,
        format!(
            "cannot use `{name}` as a {surface} dtype: {name} is reserved \
             but deferred per spec/04-type-system.md §1.1.1 (active set: \
             {active_set})"
        ),
        vec![format!(
            "spec/04-type-system.md §1.1.1 records the deferral rationale \
             and {name}'s declared arithmetic width; pick one of \
             {active_set} until it activates"
        )],
    ))
}

/// Emit the canonical "unsupported precision" diagnostic for either a
/// tensor element or a scalar cast target. The deferred `f8e4m3` dtype
/// (`spec/04-type-system.md` §1.1.1) gets a specific diagnostic citing the
/// owning spec section so producers can resolve the deferral state without
/// guessing.
/// chelis#731 Phase 2 (§C3): report an unsupported cast precision and return
/// the witness-carrying `Type::Error`. Every caller does `return
/// push_unsupported_precision_error(...)`, so the push and the error return
/// are one expression.
pub(super) fn push_unsupported_precision_error(
    errors: &mut DiagnosticSink<'_>,
    new_prec: Prim,
    tensor: bool,
) -> Type {
    let surface = if tensor { "tensor element" } else { "scalar" };
    let active_set = "f32, f64, bf16, f16, bool, int8, int16, int32, int64";
    let err = if matches!(new_prec, Prim::F8e4m3) {
        CheckError::new(
            CheckErrorKind::UnsupportedTensorPrecision,
            format!(
                "cannot cast {surface} to `f8e4m3`: f8e4m3 is deferred per \
                 spec/04-type-system.md §1.1.1 and is not part of the active \
                 numeric primitive set ({active_set})"
            ),
            vec![format!(
                "f8e4m3 has no active backend in this cycle; cast to one of \
                 {active_set} instead, or follow spec/04-type-system.md §1.1.1 \
                 for the deferral rationale"
            )],
        )
    } else {
        CheckError::new(
            CheckErrorKind::UnsupportedTensorPrecision,
            format!(
                "cannot cast {surface} to unsupported precision `{}` \
                 (supported: {active_set})",
                new_prec.name()
            ),
            vec![format!("Use a supported {surface} precision")],
        )
    };
    report(errors, err)
}
