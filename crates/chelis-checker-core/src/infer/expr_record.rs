//! Tuple, record, access, update, and cast inference.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;
use alloc::sync::Arc;
use chelis_deep::CastMode;
use chelis_numeric_core::named_cast::NamedCastMode;
use chelis_type_core::decisions::cast::{
    binder_cast_result_from_settled_source, cast_mode_atom, cast_result_from_settled_source,
    key_cast_source_error, named_cast_source_family, named_cast_source_noun,
    read_through_tensor_borrow, require_cast_source_family,
};

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_tuple(
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    operation: &crate::resources::Operation<'_>,
) -> Type<ErrorWitness> {
    let kids = node.children_slice();
    let elems: Vec<Type<ErrorWitness>> = kids
        .iter()
        .map(|e| infer_expr(e, env, vg, subst, adt_reg, errors, product, operation))
        .collect();
    Type::tuple(elems)
}

/// Extract a non-negative tuple-projection index from a Deep index
/// node. Accepts a bare `Int` atom (hand-written Deep) and a `lit`
/// node wrapping an `Int` atom (the Surf `.N` desugar, chelis#707).
/// Returns `None` for any other shape or a negative literal, which the
/// sole caller maps to `Type::Error`.
pub(super) fn tuple_get_index(expr: &deep::Expr) -> Option<usize> {
    // chelis#1125 PP7 / [04-TOT-5]: read the `lit` wrapper through the
    // carrier-preserving `stamped_parts`. The `Expr::List`-only arm returned
    // `None` for a stamped index node, and the sole caller turned that into
    // `invalid tuple index: ... found a non-literal expression` -- so
    // `check_typed_program` REJECTED a well-formed projection that
    // `check_ir_program` accepted. This is the set's one fail-closed row.
    if let deep::Expr::Atom(deep::Atom::Int(n), _) = expr {
        return usize::try_from(*n).ok();
    }
    match stamped_parts(expr) {
        Some((DeepTag::Lit, _, lit_kids)) => match lit_kids.first() {
            Some(deep::Expr::Atom(deep::Atom::Int(n), _)) => usize::try_from(*n).ok(),
            _ => None,
        },
        _ => None,
    }
}

/// Human description of a malformed tuple-projection index, for the
/// diagnostic the sole caller pushes when `tuple_get_index` returns
/// `None`. Peeks through a `lit` wrapper to the payload atom.
pub(super) fn describe_tuple_index(expr: &deep::Expr) -> String {
    // chelis#1125 PP7 / [04-TOT-5]: peek through the `lit` wrapper on either
    // carrier. The `Expr::List`-only match made every stamped index -- valid
    // or not -- describe as "a non-literal expression", so the two ingresses
    // rejected a malformed index with DIFFERENT text. A verdict includes its
    // diagnostic, so an agreeing rejection with disagreeing reasons is still
    // a divergence.
    let atom = match stamped_parts(expr) {
        Some((DeepTag::Lit, _, lit_kids)) => lit_kids.first(),
        _ => Some(expr),
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
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    operation: &crate::resources::Operation<'_>,
) -> Type<ErrorWitness> {
    let kids = node.children_slice();
    if kids.len() < 2 {
        return malformed_form(node, "tuple-get", "a tuple expression and an index", errors);
    }

    let tuple_ty = infer_expr(
        &kids[0], env, vg, subst, adt_reg, errors, product, operation,
    );
    let resolved = subst.apply(&tuple_ty);

    // The Surf `.N` desugar emits the projection index as a `lit` node
    // `(lit {type: i32} <Int>)` (`desugar.rs`, `Expr::TupleGet`),
    // while hand-written Deep may carry it as a bare `Int` atom. Read
    // the index from either shape. Matching only the bare atom made
    // every Surf-level `.N` projection fall through to `Type::Error`,
    // silently erasing the element type — `Type::Error` then unifies
    // with anything, so a value derived from a tuple projection lost
    // its nominal type at every downstream boundary (chelis#707).
    // A malformed index (negative, float, symbol, or any non-literal) is not a
    // valid projection. Diagnosing it rather than returning a silent
    // `Type::Error` is chelis#707/rt-707: a bare negative `Int` used to blow up
    // as `-1 as usize` into a loud out-of-bounds error, and every other shape
    // was silently swallowed.
    //
    // chelis#874 Slice 2 moves it onto the shared seam. The KIND changes from
    // `TupleIndexOutOfBounds` to `MalformedForm`, which is the honest one: an
    // unreadable index is not out of bounds, and the genuine out-of-bounds
    // arm below keeps the kind so it means only what it says. The caller
    // detail keeps `describe_tuple_index`, which peels a `lit` wrapper to name
    // the payload atom -- chelis#1107's PP7 [04-TOT-5] row exists because the
    // two ingresses once disagreed on exactly that wording. It is suppressed
    // for a bare atom, which `describe_slot_child` already names identically.
    let index = match read_required_slot_detailed(
        kids,
        DeepTag::TupleGet,
        1,
        SlotShape::TupleIndex,
        tuple_get_index,
        |child| {
            // Suppressed for a bare atom: `describe_slot_child` already names
            // it identically, and "found integer literal -1 (integer literal
            // -1)" would be noise.
            if matches!(child, deep::Expr::Atom(..)) {
                None
            } else {
                Some(describe_tuple_index(child))
            }
        },
        errors,
    ) {
        Ok(index) => index,
        Err(witness) => return propagate(&witness),
    };
    // chelis#1603: the selector twin of `vmap`'s axis. The index child is
    // read by `tuple_get_index` and otherwise never visited, so `infer_lit`
    // -- the boundary that owns [04-LIT-1] -- never saw a `lit` index and a
    // bool-stamped `0` projected element 0 with no diagnostic. Visit it.
    infer_expr(
        &kids[1], env, vg, subst, adt_reg, errors, product, operation,
    );
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
    env: &'a Env,
    adt_reg: &'a AdtRegistry,
) -> Option<(&'a str, &'a crate::adt::VariantInfo, String)> {
    if let Some((adt_name, _, variant)) =
        constructor_for_shape(head, CallShape::Record, env, adt_reg)
    {
        return Some((adt_name, variant, variant.name.clone()));
    }
    if let Some((adt_name, variant)) = adt_reg
        .lookup_variant_preferring_shape(head, CallShape::Record)
        .or_else(|| adt_reg.lookup_variant_terminal_unique(head))
    {
        return Some((adt_name, variant, variant.name.clone()));
    }
    // Alias head: `type P2 = Probability` makes `P2 { ... }` mean
    // `Probability { ... }`.
    let alias = adt_reg.resolve_alias(head)?;
    if let Type::Adt(target, _) | Type::KindedAdt(target, _) = &alias.body {
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
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    operation: &crate::resources::Operation<'_>,
) -> Type<ErrorWitness> {
    let kids = node.children_slice();
    let Some(head) = kids.first().and_then(symbol_name) else {
        return malformed_form(
            node,
            "record",
            "a symbol constructor head as its first child",
            errors,
        );
    };

    let Some((adt_name, variant, ctor_name)) = resolve_record_head(head, env, adt_reg) else {
        // Infer field values so nested errors still surface, then
        // reject the unknown constructor.
        for kv_expr in kids.iter().skip(1) {
            if let Some((DeepTag::Kv, _, kv_children)) = stamped_parts(kv_expr)
                && let Some(value) = kv_children.get(1)
            {
                infer_expr(value, env, vg, subst, adt_reg, errors, product, operation);
            }
        }
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::UnknownConstructor {
                    identifier: head.to_string(),
                },
                format!("unknown constructor: {head}"),
                vec![format!(
                    "Constructor '{head}' is not in scope. Declare it locally or \
                     add it to an import (e.g. `import Mod ({head})`)"
                )],
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
    let head_is_alias = adt_reg.resolve_alias(head).is_some();
    if !head_is_opaque && !head_is_alias && constructor_out_of_scope(head, env) {
        for kv_expr in kids.iter().skip(1) {
            if let Some((DeepTag::Kv, _, kv_children)) = stamped_parts(kv_expr)
                && let Some(value) = kv_children.get(1)
            {
                infer_expr(value, env, vg, subst, adt_reg, errors, product, operation);
            }
        }
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::UnknownConstructor {
                    identifier: head.to_string(),
                },
                with_node_provenance(node, format!("unknown constructor: {head}")),
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
        operation,
    );

    let declared_field_names: Vec<Option<String>> =
        variant.fields.iter().map(|(n, _)| n.clone()).collect();
    let known_field_set: UnordSet<&str> = declared_field_names
        .iter()
        .filter_map(|n| n.as_deref())
        .collect();

    // Instantiate the resolved ADT's constructor directly from its
    // registry definition (the issue #181 pat-record intent, made
    // collision-proof): the name-keyed env holds ONE scheme per
    // constructor name, so same-named constructors from colliding
    // ADTs (chelis#148) would dispatch the field types to whichever
    // deftype registered last.
    let (mut instantiated_arg_types, instantiated_ret) = match adt_reg.lookup(adt_name) {
        Some(adt_def) => instantiate_variant_of(adt_def, variant, vg),
        None => (Vec::new(), vg.fresh_type()),
    };
    split_record_result_origin(
        &mut instantiated_arg_types,
        &instantiated_ret,
        vg,
        subst,
        product,
    );

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
        // chelis#874 R5 / [04-TOT-4]: this was
        // `let (Some(..), Some(..)) = (..) else { continue };`. The `continue`
        // skipped the `infer_expr(value, ..)` below, so an unreadable key left
        // the VALUE unvisited and unstamped, and section C4.1's owner-stamp
        // tripwire fired on that value as an `internal:` invariant violation
        // naming `lit`. The program was rejected, but for a node the author did
        // not write wrongly. Read the key at its own slot, and infer the value
        // either way so the walk still covers it.
        let field_name = read_required_slot(
            kv_kids,
            DeepTag::Kv,
            0,
            SlotShape::FieldName,
            symbol_name,
            errors,
        )
        .ok();
        // The ABSENT value child is not this class and is not claimed here.
        // `arity_contract(Kv)` is `Fixed(2)` and `Node::try_new` enforces it at
        // the stamp boundary, so `(kv {} r)` is rejected as
        // `wrong child count for 'kv': expected Fixed(2), got 1` before inference
        // ever runs, and no other spelling of a `kv` node exists; the let-else
        // only keeps this read total.
        let Some(value) = kv_kids.get(1) else {
            continue;
        };
        let value_ty = infer_expr(value, env, vg, subst, adt_reg, errors, product, operation);
        let Some(field_name) = field_name else {
            continue;
        };
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
) -> (Vec<Type<ErrorWitness>>, Type<ErrorWitness>) {
    let mut type_vars = adt_def.param_vars.clone();
    let mut dim_vars = adt_def
        .param_args
        .iter()
        .filter_map(|argument| match argument {
            NominalArg::Dimension(Dim::Var(var)) => Some(*var),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut rank_vars = Vec::new();
    for (_, field_type) in &variant.fields {
        for var in chelis_type_core::scheme::free_tvars(field_type) {
            if !type_vars.contains(&var) {
                type_vars.push(var);
            }
        }
        for var in chelis_type_core::scheme::free_dvars(field_type) {
            if !dim_vars.contains(&var) {
                dim_vars.push(var);
            }
        }
        for var in chelis_type_core::scheme::free_rvars(field_type) {
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

    let args: Vec<Type<ErrorWitness>> = variant
        .fields
        .iter()
        .map(|(_, field_type)| renaming.apply(field_type))
        .collect();
    let renamed_args = adt_def
        .param_args
        .iter()
        .map(|argument| match argument {
            NominalArg::Type(ty) => NominalArg::Type(renaming.apply(ty)),
            NominalArg::Dimension(dim) => NominalArg::Dimension(renaming.apply_dim(dim)),
        })
        .collect::<Vec<_>>();
    let ret = if adt_def.param_kinds.contains(&NominalParamKind::Dimension) {
        Type::kinded_adt(adt_def.name.clone(), renamed_args)
    } else {
        Type::Adt(
            adt_def.name.clone(),
            renamed_args
                .into_iter()
                .map(|argument| match argument {
                    NominalArg::Type(ty) => ty,
                    NominalArg::Dimension(_) => unreachable!("type-only nominal definition"),
                })
                .collect(),
        )
    };
    (args, ret)
}

/// Instantiate the single record variant of the ADT named by
/// `target_ty` and unify the instantiated result with the target,
/// returning the per-field types aligned with `variant.fields` so
/// they reflect the target's concrete type arguments.
pub(super) fn instantiated_field_types(
    adt_name: &str,
    variant: &crate::adt::VariantInfo,
    target_ty: &Type<ErrorWitness>,
    adt_reg: &AdtRegistry,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
) -> Vec<Type<ErrorWitness>> {
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
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    operation: &crate::resources::Operation<'_>,
) -> Type<ErrorWitness> {
    let kids = node.children_slice();
    if kids.len() < 2 {
        return malformed_form(
            node,
            "access",
            "a target expression and a field name",
            errors,
        );
    }
    let target_ty = infer_expr(
        &kids[0], env, vg, subst, adt_reg, errors, product, operation,
    );
    // chelis#874 Slice 2: this read was already total, and its failure branch
    // already pushed. Migrating it onto the shared seam is what makes ONE
    // mechanism own every role-slot read, which is chelis#874's condition. The
    // message improves in passing: `malformed_form` reported the node's CHILD
    // COUNT as what it "found", which says nothing about a two-child `access`
    // whose second child is the wrong shape; the seam names that child.
    let field_name = match read_required_slot(
        kids,
        DeepTag::Access,
        1,
        SlotShape::FieldName,
        symbol_name,
        errors,
    ) {
        Ok(name) => name,
        Err(witness) => return propagate(&witness),
    };
    // Peel borrow layers: an `&T` target reads through the borrow.
    let mut resolved = subst.apply(&target_ty);
    while let Type::Ref(inner) = resolved {
        resolved = Arc::unwrap_or_clone(inner);
    }
    match resolved {
        Type::Adt(ref adt_name, _) | Type::KindedAdt(ref adt_name, _) => {
            // RFC D-CHECK: field access on an out-of-module opaque
            // type is rejected; inference continues so the access
            // still yields its true field type (no cascades).
            crate::opacity::check_opaque_use(
                crate::opacity::OpaqueAction::FieldAccess,
                adt_name,
                adt_reg,
                errors,
                operation,
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
            subst
                .record_deferred_opaque_use(tv, chelis_type_core::unify::DeferredOpaqueUse::Access);
            // chelis#1836: the fresh variable is also TIED to the field the
            // target turns out to carry. The opacity ledger above revisits the
            // TARGET when it binds; it says nothing about the projected field
            // type, so a shape-computed route over `q.x` used to publish a
            // result the declaration could bind to any shape, and under the
            // widened readiness predicate it would instead suspend on an
            // operand nothing ever binds. The derivation ledger resolves the
            // projection by ADT field lookup at the same point tuple
            // projection is resolved.
            let projected = vg.fresh_type();
            product.defer_record_field(resolved.clone(), field_name.to_string(), projected.clone());
            projected
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

/// Constructor field occurrences are inputs; their equality is owned by
/// the resulting record. Construction and update use the same relation.
fn split_record_result_origin(
    fields: &mut Vec<Type<ErrorWitness>>,
    result: &Type<ErrorWitness>,
    vg: &mut VarGen,
    subst: &Subst<ErrorWitness>,
    product: &mut InferenceProduct,
) {
    let signature = Type::function(fields.clone(), result.clone());
    let variables = chelis_type_core::scheme::free_tvars(&signature)
        .into_iter()
        .collect::<Vec<_>>();
    if let Some(origin) = ResultOrigin::aggregate(&signature, &variables, &[], &[], vg) {
        for equation in origin.equations {
            subst.record_result_constraint(equation);
        }
        product.import_result_constraints(subst);
        if let Type::Fn(params, _) = origin.body {
            *fields = params.to_vec();
        }
    }
}

/// Infer `(record-update {} target (kv {} field value)...)` — Deep
/// functional record update (RFC D-CHECK prerequisite inference).
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_record_update(
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    operation: &crate::resources::Operation<'_>,
) -> Type<ErrorWitness> {
    let kids = node.children_slice();
    if kids.is_empty() {
        return malformed_form(
            node,
            "record-update",
            "a target expression to update",
            errors,
        );
    }
    let target_ty = infer_expr(
        &kids[0], env, vg, subst, adt_reg, errors, product, operation,
    );
    // Infer the update values regardless of target resolution so
    // nested errors surface exactly once.
    let mut kv_pairs: Vec<(&str, Type<ErrorWitness>)> = Vec::new();
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
        // chelis#874 / [04-TOT-4]: `infer_record`'s repair, applied to the
        // identical `else { continue }` here. This site produced the same
        // owner-stamp misattribution on the update value.
        let field_name = read_required_slot(
            kv_kids,
            DeepTag::Kv,
            0,
            SlotShape::FieldName,
            symbol_name,
            errors,
        )
        .ok();
        // The ABSENT value child is not this class and is not claimed here.
        // `arity_contract(Kv)` is `Fixed(2)` and `Node::try_new` enforces it at
        // the stamp boundary, so `(kv {} r)` is rejected as
        // `wrong child count for 'kv': expected Fixed(2), got 1` before inference
        // ever runs, and no other spelling of a `kv` node exists; the let-else
        // only keeps this read total.
        let Some(value) = kv_kids.get(1) else {
            continue;
        };
        let value_ty = infer_expr(value, env, vg, subst, adt_reg, errors, product, operation);
        let Some(field_name) = field_name else {
            continue;
        };
        kv_pairs.push((field_name, value_ty));
    }
    let mut resolved = subst.apply(&target_ty);
    while let Type::Ref(inner) = resolved {
        resolved = Arc::unwrap_or_clone(inner);
    }
    match resolved {
        Type::Adt(ref adt_name, _) | Type::KindedAdt(ref adt_name, _) => {
            // RFC D-CHECK: record update of an out-of-module opaque
            // type is rejected; inference continues and returns the
            // target's true type (no cascades).
            crate::opacity::check_opaque_use(
                crate::opacity::OpaqueAction::RecordUpdate,
                adt_name,
                adt_reg,
                errors,
                operation,
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
            let definition = adt_reg
                .lookup(adt_name)
                .expect("the record variant belongs to its registered definition");
            let (mut update_fields, updated) = instantiate_variant_of(definition, variant, vg);
            split_record_result_origin(&mut update_fields, &updated, vg, subst, product);
            for (position, (field_name, _)) in variant.fields.iter().enumerate() {
                if !kv_pairs
                    .iter()
                    .any(|(name, _)| field_name.as_deref() == Some(*name))
                    && let Err(error) =
                        unify(&field_types[position], &update_fields[position], subst)
                {
                    errors.push(error.into());
                }
            }
            for (field_name, value_ty) in &kv_pairs {
                let pos = variant
                    .fields
                    .iter()
                    .position(|(n, _)| n.as_deref() == Some(*field_name));
                match pos {
                    Some(pos) => {
                        if let Some(field_ty) = update_fields.get(pos)
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
            if !product.defer_result_type_constraint(&updated, &resolved, subst)
                && let Err(error) = unify(&updated, &resolved, subst)
            {
                errors.push(error.into());
            }
            subst.apply(&updated)
        }
        Type::Var(tv) => {
            subst.record_deferred_opaque_use(
                tv,
                chelis_type_core::unify::DeferredOpaqueUse::RecordUpdate,
            );
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
    expr: &deep::Expr,
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    operation: &crate::resources::Operation<'_>,
) -> Type<ErrorWitness> {
    let kids = node.children_slice();
    if kids.len() < 2 {
        return malformed_form(node, "cast", "an expression and a target type", errors);
    }
    // chelis#874 Slice 2: the optional [05-OP-6] mode selector at child 2.
    // `deep::cast_mode_of` handled absence internally and returned a `Result`,
    // which is the seam's shape in miniature; reading it through the seam makes
    // the absence-versus-unreadability split visible AT the call site and puts
    // the last `Selector` slot on one mechanism. The kind changes from
    // `CastNonTensor` to `MalformedForm`, which is again the honest one: an
    // unrecognized selector is a malformed form, not a non-tensor cast, and
    // `CastNonTensor` keeps its two real users (the cast target and the
    // operand).
    //
    // `deep::cast_mode_of` itself is deliberately untouched: `chelis-ir`'s
    // `lower.rs` and `host.rs` and `chelis-compiler-api`'s `eval.rs` each
    // format their own copy of the old message from it, and a CHECKED program
    // never reaches those arms.
    let mode = match read_optional_slot(
        kids,
        DeepTag::Cast,
        2,
        SlotShape::ModeSelector,
        |child| symbol_name(child).and_then(deep::CastMode::from_deep_selector),
        errors,
    ) {
        Ok(Some(mode)) => mode,
        Ok(None) => deep::CastMode::Checked,
        Err(witness) => return propagate(&witness),
    };

    let expr_ty = infer_expr(
        &kids[0], env, vg, subst, adt_reg, errors, product, operation,
    );
    // chelis#3101, [05-OP-63] and [05-OP-6]: a cast's tensor source is a
    // read-only `&tensor[D, p]` parameter (spec/05 section 1.3.1), so an
    // explicit borrow reads as the owned tensor. Only a tensor is borrowed;
    // any other `&` source keeps its refusal below.
    let resolved = read_through_tensor_borrow(subst.apply(&expr_ty), subst);

    // Every target spelling first crosses the centralized resolver. Bare
    // primitive symbols are retained for historical compatibility; canonical
    // `t-prim` still goes through the resolver's metadata and exact-arity
    // checks before semantic cast classification.
    let (resolved_target, target_location) = {
        let mut resolver = DeepTypeResolver::new(
            TypeUseSite::CastTarget,
            annotation_binder_mode(env),
            adt_reg.resolution_env(),
            vg,
            errors,
        )
        .with_diagnostic_owner(expr);
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
            if let Some(diag) = crate::deep_type::retired_integer_diagnostic(&name, "deep", false)
                .or_else(|| unsigned_family_diagnostic(&name, false))
                .or_else(|| deferred_family_diagnostic(&name, false))
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
                    Some(target) => Type::adt(target, Vec::new()),
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
    if let Type::Adt(target_adt, _) | Type::KindedAdt(target_adt, _) = &target_ty
        && crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::CastInto,
            target_adt,
            adt_reg,
            errors,
            operation,
        )
    {
        return target_ty;
    }
    let mut peeled = &resolved;
    while let Type::Ref(inner) = peeled {
        peeled = inner.as_ref();
    }
    if let Type::Adt(source_adt, _) | Type::KindedAdt(source_adt, _) = peeled
        && crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::CastOut,
            source_adt,
            adt_reg,
            errors,
            operation,
        )
    {
        return target_ty;
    }
    if let Some(error) = key_cast_source_error(&resolved) {
        return report(errors, error.into());
    }

    let new_prec = match target_ty {
        Type::Var(target) => {
            // A cast inside a declaration may name one of that declaration's
            // bounded dtype variables. Keep the target symbolic and
            // let the declared signature plus its numeric consumers select the
            // concrete active dtype. This is the source-level spelling needed
            // by [05-OP-35]'s same-p `linspace` and `arange` graphs; a closed
            // cast outside such a declaration still rejects the name in the
            // resolver above.
            //
            // chelis#2158: every arm applies [05-OP-6]'s `cast_trunc` pair
            // rule. A source dtype that is still a variable, an authored binder
            // among them, is constrained rather than inspected: reading its
            // restriction rejected an inference variable that a later binding
            // makes a float, and admitting it skipped a binder that nothing
            // ever binds.
            let named_pair_rejected = |source_admitted: bool| {
                mode.named().is_some()
                    && (!source_admitted
                        || !subst
                            .tvar_restriction(target)
                            .is_some_and(|restriction| restriction.admits_only_integers()))
            };
            let named_pair_rejection = |named: NamedCastMode| {
                CheckError::new(
                    CheckErrorKind::CastNonTensor,
                    format!(
                        "`{}` requires {} source and an integer target ({}); use `cast` for other conversions",
                        named.keyword(),
                        named_cast_source_noun(named),
                        named.atom(),
                    ),
                    vec![],
                )
            };
            let source_admitted = |prim: Prim| {
                mode.named()
                    .is_none_or(|named| named_cast_source_family(named).admits(prim))
            };
            return match resolved {
                // A source that is still a variable. A `cast_trunc` source must
                // be a float one. An authored binder is held to a family here:
                // every bound is numeric, and an unbounded binder also denotes
                // types no cast admits, so it is held to `Numeric`.
                Type::Var(source) => {
                    if let Some(named) = mode.named()
                        && named_pair_rejected(true)
                    {
                        return report(errors, named_pair_rejection(named));
                    }
                    let required = match mode.named() {
                        Some(named) => Some(named_cast_source_family(named)),
                        None => env
                            .authored_type_binder(source, subst)
                            .map(|_| TypeVarRestriction::ActiveNumeric),
                    };
                    match required {
                        Some(required) => {
                            if let Some(error) =
                                constrain_cast_source(source, required, mode, env, subst)
                            {
                                return report(errors, error);
                            }
                            Type::Var(target)
                        }
                        // chelis#2534: an inference variable is decided when it
                        // binds, by the rule a settled source gets below.
                        // Requiring `Numeric` of it refused a lambda parameter
                        // bound to `bool` (chelis#2158 round 1), and admitting
                        // it let a `string` or an unbounded binder through. The
                        // result is a fresh variable that discharge unifies
                        // with the settled answer, which is a tensor for a
                        // tensor source.
                        None => {
                            let result = vg.fresh_type();
                            subst.record_deferred_tensor_operand(
                                source,
                                DeferredOperandGate::CastToBinder {
                                    target,
                                    result: Box::new(result.clone()),
                                    location: TypeDiagnosticLocation::from_expr(expr)
                                        .map(TypeDiagnosticLocation::into_source_label),
                                },
                            );
                            result
                        }
                    }
                }
                // A checked `cast` on a settled source: the one decision the
                // deferred gate also calls.
                settled if mode == CastMode::Checked => {
                    match binder_cast_result_from_settled_source(settled, target, subst) {
                        Ok(result) => result,
                        Err(error) => report(errors, (*error).into()),
                    }
                }
                Type::Tensor(dims, source) if subst.tvar_restriction(target).is_some() => {
                    // [05-OP-63]: a dtype change preserves every dimension.
                    // The declaration's [04-DTYPE-2] bound is retained on the
                    // precision variable and checked at each instantiation.
                    let admitted = match source {
                        TensorPrec::Concrete(p) => source_admitted(p),
                        TensorPrec::Var(_) => true,
                    };
                    if let Some(named) = mode.named()
                        && named_pair_rejected(admitted)
                    {
                        return report(errors, named_pair_rejection(named));
                    }
                    if let Some(named) = mode.named()
                        && let TensorPrec::Var(p) = source
                        && let Some(error) = constrain_cast_source(
                            p,
                            named_cast_source_family(named),
                            mode,
                            env,
                            subst,
                        )
                    {
                        return report(errors, error);
                    }
                    Type::Tensor(dims, TensorPrec::Var(target))
                }
                // Each named rung admits only its own family's scalar sources.
                Type::Prim(source) if source.is_data_element_dtype() => {
                    if let Some(named) = mode.named()
                        && named_pair_rejected(source_admitted(source))
                    {
                        return report(errors, named_pair_rejection(named));
                    }
                    Type::Var(target)
                }
                Type::Error(_) => Type::Var(target),
                other => report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::CastNonTensor,
                        format!(
                            "cast to a quantified scalar dtype requires a numeric or bool scalar, got {other}"
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
                at_check_site(
                    expr,
                    CheckError::with_types(
                        CheckErrorKind::CastNonTensor,
                        format!(
                            "cast argument 2: expected primitive dtype, got {other} (chelis#756)"
                        ),
                        "primitive dtype".to_string(),
                        other.to_string(),
                        vec![
                            "cast targets a scalar primitive: f32, f64, bf16, f16, bool, \
                     i8, i16, i32, i64"
                                .to_string(),
                        ],
                    ),
                ),
            );
        }
    };

    // chelis#2158: an authored binder whose bound admits an integer is named at
    // the cast, whether it is a tensor source's precision or a scalar source.
    // The float requirement itself is recorded where the settled source is
    // decided, which discharge shares.
    if let Some(named) = mode.named()
        && let Some(error) = authored_cast_source_rejection(
            &resolved,
            named_cast_source_family(named),
            mode,
            env,
            subst,
        )
    {
        return report(errors, error);
    }
    cast_result_from_source(expr, resolved, new_prec, mode, subst, vg, errors)
}

pub(super) fn cast_result_from_source(
    site: &deep::Expr,
    resolved: Type<ErrorWitness>,
    new_prec: Prim,
    mode: CastMode,
    subst: &mut Subst<ErrorWitness>,
    vg: &mut VarGen,
    errors: &mut DiagnosticSink<'_>,
) -> Type<ErrorWitness> {
    match resolved {
        // chelis#1489: the source may simply not be resolved YET. Deciding
        // here bound the verdict to inference order rather than to the
        // program, and this gate alone was the largest single contributor to
        // the spurious rejections measured on 0.18.6.
        //
        // Suspend the decision on the source variable instead. Unification
        // discharges it at the instant that variable is bound, so the verdict
        // depends on what the program says and not on when inference got
        // there. The result is a fresh variable, and discharge unifies the
        // settled answer into it: returning an unconstrained variable with
        // nothing to settle it is how an earlier revision let an ill-typed
        // program reach codegen.
        //
        // chelis#2151: a source carrying a declared dtype-family bound suspends
        // here too. It is settled by `unify::discharge_bounded_scalar_casts`
        // at the first binding that follows. The result variable returned
        // below is always consumed by one, so no second, cast-time copy of that
        // decision is needed.
        Type::Var(source_var) => {
            let result = vg.fresh_type();
            subst.record_deferred_tensor_operand(
                source_var,
                DeferredOperandGate::Cast {
                    target: new_prec,
                    mode,
                    borrowed: false,
                    result: Box::new(result.clone()),
                    location: TypeDiagnosticLocation::from_expr(site)
                        .map(TypeDiagnosticLocation::into_source_label),
                },
            );
            result
        }
        // chelis#3101: `&v` over a still-unresolved `v` (a lambda parameter)
        // is no settled source either. It suspends on `v`, and discharge
        // decides the type the eager `borrow` arm gives `&settled` through the
        // same function the eager arm calls.
        Type::Ref(inner) if matches!(subst.apply(&inner), Type::Var(_)) => {
            let Type::Var(source_var) = subst.apply(&inner) else {
                unreachable!("guarded above");
            };
            let result = vg.fresh_type();
            subst.record_deferred_tensor_operand(
                source_var,
                DeferredOperandGate::Cast {
                    target: new_prec,
                    mode,
                    borrowed: true,
                    result: Box::new(result.clone()),
                    location: TypeDiagnosticLocation::from_expr(site)
                        .map(TypeDiagnosticLocation::into_source_label),
                },
            );
            result
        }
        // Every settled source, accepted or rejected, is decided by the one
        // function discharge also calls. There is no second copy of this
        // decision to disagree with.
        settled => {
            // A cast converts values and never makes a bracket literal a
            // tensor (chelis#3114), so a List operand spelled as one gets the
            // value-preserving spelling. The truncating rung takes no hint:
            // the suffixed spelling would drop its truncation.
            let bracket_literal = mode == CastMode::Checked
                && matches!(&settled, Type::Adt(name, _) if name == "List")
                && matches!(stamped_parts(site), Some((DeepTag::Cast, _, [operand, ..]))
                    if is_bracket_literal(operand));
            match cast_result_from_settled_source(settled, new_prec, mode, subst) {
                Ok(result) => result,
                Err(mut error) => {
                    if bracket_literal {
                        error
                            .suggestions
                            .push(crate::errors::list_to_tensor_hint(Some(new_prec.name())));
                    }
                    report(errors, at_check_site(site, (*error).into()))
                }
            }
        }
    }
}

/// chelis#2158: an authored binder standing for a cast's source dtype, the
/// precision of a tensor `source` or the type of a scalar one, whose declared
/// bound admits a dtype `required` does not.
///
/// Decided here, from the declaration, rather than by narrowing the binder:
/// [04-DTYPE-2] puts the bound in the binder list, so no call site can change
/// the verdict, and the scheme the declaration publishes stays its declared
/// signature ([04-INF-6]). The same shape as a family-policy operation's
/// verdict on a bounded precision variable (chelis#1805).
fn authored_cast_source_rejection(
    source: &Type<ErrorWitness>,
    required: TypeVarRestriction,
    mode: CastMode,
    env: &Env,
    subst: &Subst<ErrorWitness>,
) -> Option<CheckError> {
    let variable = match source {
        Type::Tensor(_, TensorPrec::Var(precision)) => *precision,
        Type::Var(variable) => *variable,
        _ => return None,
    };
    let Type::Var(variable) = subst.apply(&Type::Var(variable)) else {
        return None;
    };
    let (name, Some(bound)) = env.authored_type_binder(variable, subst)? else {
        return None;
    };
    if bound.intersect(required) == Some(bound) {
        return None;
    }
    let operation = mode.keyword();
    let atom = cast_mode_atom(mode);
    Some(CheckError::new(
        CheckErrorKind::PrecisionMismatch,
        format!(
            "`{operation}` requires a source of dtype family `{}` ({atom}), but its source \
             dtype is the declared type parameter `{name}`, whose `{}` bound ({}) admits dtypes \
             outside `{}`; an authored binder must satisfy the operation at every instantiation \
             its declaration admits (spec/04-type-system.md §3.1.3 [04-INF-6], §5.9 \
             [04-DTYPE-2])",
            required.bound_spelling(),
            bound.bound_spelling(),
            bound.membership_gloss(),
            required.bound_spelling(),
        ),
        vec![format!(
            "Declare `{name}: {}` in the binder list, or convert the source with `cast` first \
             ({atom})",
            required.bound_spelling()
        )],
    ))
}

/// [`authored_cast_source_rejection`], then [`require_cast_source_family`]:
/// the whole source-dtype decision where the declaration's binders are in
/// scope.
fn constrain_cast_source(
    variable: TypeVar,
    required: TypeVarRestriction,
    mode: CastMode,
    env: &Env,
    subst: &Subst<ErrorWitness>,
) -> Option<CheckError> {
    authored_cast_source_rejection(&Type::Var(variable), required, mode, env, subst)
        .or_else(|| require_cast_source_family(variable, required, mode, subst).map(Into::into))
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
) -> Type<ErrorWitness> {
    let error = CheckError::with_types(
        CheckErrorKind::CastNonTensor,
        format!(
            "cast argument 2: expected primitive dtype, got `{name}` \
             (not a recognized primitive type; chelis#756)"
        ),
        "primitive dtype".to_string(),
        name.to_string(),
        vec![
            "cast targets a scalar primitive: f32, f64, bf16, f16, bool, \
             i8, i16, i32, i64"
                .to_string(),
        ],
    );
    report(
        errors,
        match location {
            Some(location) => location.attach(error),
            None => error,
        },
    )
}
