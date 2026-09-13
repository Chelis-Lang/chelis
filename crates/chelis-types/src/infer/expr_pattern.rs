//! Match and pattern inference.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_match(
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
            "match",
            "a scrutinee expression and at least one arm",
            errors,
        );
    }

    let scrutinee_ty = infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product);

    // chelis#710 [04-TOT-3]: a match with a scrutinee but no arms is
    // malformed. Return early so the exhaustiveness check below cannot also
    // fire (which would double-report on an ADT scrutinee); a bare bool/int
    // scrutinee with zero arms used to reach `result_ty.unwrap_or(Type::Error)`
    // as a silent `Type::Error` (census-verified silent-through).
    if kids.len() < 2 {
        return malformed_form(
            list,
            "match",
            "at least one arm after the scrutinee",
            errors,
        );
    }

    let mut result_ty: Option<Type> = None;
    let mut covered_variants: Vec<String> = Vec::new();
    let mut has_wildcard = false;

    for arm_expr in &kids[1..] {
        if let Some((DeepTag::Arm, _, arm_kids)) = stamped_parts(arm_expr) {
            // arm_kids[0] = pattern, arm_kids[1] = guard (usually ()), arm_kids[2] = body
            if arm_kids.len() >= 3 {
                let mut arm_env = env.clone();
                let pat = &arm_kids[0];
                // RFC D-CHECK exhaustiveness fix (RT-0 verified false
                // positives): a TOP-LEVEL irrefutable arm covers the
                // match -- a bare `pat-var`, or a `pat-as` whose
                // inner pattern is irrefutable. Nested `pat-var`
                // keeps not-covering so exhaustiveness is not
                // weakened on ordinary ADTs.
                if top_level_arm_is_irrefutable(pat) {
                    has_wildcard = true;
                }
                pattern_bindings(
                    pat,
                    &scrutinee_ty,
                    &mut arm_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    product,
                    &mut covered_variants,
                    &mut has_wildcard,
                );

                let guard = &arm_kids[1];
                let empty_guard = matches!(guard, deep::Expr::List(guard_list, _) if guard_list.elements.is_empty())
                    || matches!(guard, deep::Expr::BareList(elements, _) if elements.is_empty());
                if !empty_guard {
                    let guard_ty =
                        infer_expr(guard, &mut arm_env, vg, subst, adt_reg, errors, product);
                    let resolved_guard = subst.apply(&guard_ty);
                    if !matches!(resolved_guard, Type::Prim(Prim::Bool) | Type::Error(_)) {
                        errors.push(CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!("match arm guard must be bool, got {resolved_guard}"),
                            vec![],
                        ));
                    }
                }

                let body_ty = infer_expr(
                    &arm_kids[2],
                    &mut arm_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    product,
                );

                match &result_ty {
                    None => result_ty = Some(body_ty),
                    Some(prev) => {
                        if let Err(te) = unify(prev, &body_ty, subst) {
                            errors.push(te.into());
                        }
                        result_ty = Some(subst.apply(prev));
                    }
                }
            }
        }
    }

    // Exhaustiveness check (wildcard covers everything)
    if !has_wildcard {
        let resolved_scrutinee = subst.apply(&scrutinee_ty);
        if let Type::Adt(ref adt_name, _) | Type::KindedAdt(ref adt_name, _) = resolved_scrutinee
            && let Some(all_variants) = adt_reg.variant_names(adt_name)
        {
            let missing: Vec<&String> = all_variants
                .iter()
                .filter(|v| !covered_variants.contains(v))
                .collect();
            if !missing.is_empty() {
                let names: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                errors.push(CheckError::new(
                    CheckErrorKind::NonExhaustiveMatch,
                    with_macro_provenance(
                        &deep::Expr::List(list.clone(), zero_span()),
                        format!("non-exhaustive match: missing variants {:?}", names),
                    ),
                    vec![],
                ));
            }
        }
    }

    // A match with no arms is screened by the `(match {})` malformed guard
    // upstream, so `result_ty` is `Some` for any well-formed match; the
    // fallback is a fresh var (never a silent `Type::Error`, chelis#731 §C3).
    result_ty.unwrap_or_else(|| vg.fresh_type())
}

/// Visit sub-patterns whose element types could not be derived, each against a
/// fresh type variable.
///
/// chelis#874 / [04-TOT-4]: the recovery path after a pattern's constructor
/// head is rejected as unreadable. The head's rejection says nothing about the
/// sub-patterns, which are still nodes the author submitted; abandoning them
/// would leave an unvisited subtree behind the diagnostic -- the same coverage
/// defect one level down -- and would let §C4.1's owner-stamp tripwire report
/// an unstamped `pat-var` instead of the head that was actually wrong.
///
/// A fresh-variable element type is the established shape for this: the
/// `pat-tuple` arm already hands each child one when the resolved scrutinee is
/// not a matching tuple.
#[allow(clippy::too_many_arguments)]
fn visit_sub_patterns_untyped(
    sub_pats: &[deep::Expr],
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    covered_variants: &mut Vec<String>,
    has_wildcard: &mut bool,
) {
    for sub_pat in sub_pats {
        let fresh = vg.fresh_type();
        pattern_bindings(
            sub_pat,
            &fresh,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            product,
            covered_variants,
            has_wildcard,
        );
    }
}

/// True for arm patterns that match every value of the scrutinee:
/// `pat-var`, `pat-wild`, and `pat-as` wrapping an irrefutable inner
/// pattern (`q @ x`). Applies at the ARM level only.
pub(super) fn top_level_arm_is_irrefutable(pat: &deep::Expr) -> bool {
    let Some((tag, _, kids)) = stamped_parts(pat) else {
        return false;
    };
    match tag {
        DeepTag::PatVar | DeepTag::PatWild => true,
        DeepTag::PatAs => kids.get(1).is_some_and(top_level_arm_is_irrefutable),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn pattern_bindings(
    pat: &deep::Expr,
    scrutinee_ty: &Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    covered_variants: &mut Vec<String>,
    has_wildcard: &mut bool,
) {
    stack_guard!("pattern_bindings", pat);
    if let Some((tag, _, kids)) = stamped_parts(pat) {
        match tag {
            DeepTag::PatVar => {
                // chelis#874 / [04-TOT-4]: `if let Some(name) = ..` with no
                // `else` bound nothing and rejected nothing when the name child
                // could not be read. The pattern node then went unstamped and
                // §C4.1's owner-stamp tripwire fired on it as an `internal:`
                // invariant violation naming `pat-var` -- a diagnostic that
                // blames the walk rather than the slot. Stamp the node either
                // way, so the reported defect is the one the author can fix.
                let resolved = subst.apply(scrutinee_ty);
                product.record_bypass(pat, resolved.clone(), "pattern binding traversal");
                if let Ok(name) = read_required_slot(
                    kids,
                    DeepTag::PatVar,
                    0,
                    SlotShape::BindingName,
                    symbol_name,
                    errors,
                ) {
                    env.bind_lexical(name.to_string(), Scheme::mono(resolved));
                }
            }
            DeepTag::PatWild => {
                // Wildcard covers everything
                *has_wildcard = true;
            }
            DeepTag::PatLit => {
                // A literal pattern binds nothing. [04-PAT-1]'s constraint on
                // the scrutinee type is its entire contribution, and it is
                // checked here so every nesting depth reached by this walk --
                // tuple, record, and constructor sub-patterns included -- gets
                // one decision at both checker ingresses.
                //
                // chelis#1525: the value child is read FIRST and structurally.
                // spec/03-deep-syntax.md section 6.3 settles its shape -- "patterns
                // do not contain expression nodes" -- so `(pat-lit {} (lit {} 1))`
                // is malformed Deep rather than a typing question, and the read
                // belongs here rather than inside `check_literal_pattern`, which
                // returns early on an unresolved or already-failed scrutinee.
                // Structural well-formedness must not depend on the scrutinee's
                // type. Before this, `literal_pattern_atom` returned `None` for
                // any non-atom child and the [04-PAT-1] check silently declined,
                // so a `lit`-wrapped pattern of the wrong family scored 1.0.
                if let Ok(atom) = read_required_slot(
                    kids,
                    DeepTag::PatLit,
                    0,
                    SlotShape::LiteralValue,
                    literal_pattern_atom,
                    errors,
                ) {
                    check_literal_pattern(pat, atom, scrutinee_ty, subst, adt_reg, errors);
                }
            }
            DeepTag::PatCtor => {
                // chelis#874 R2 / [04-TOT-4]: an unreadable head used to fall
                // off the `if let` binding, so the arm neither bound nor
                // rejected and the whole program scored 1.0. The sub-patterns
                // are still nodes the author submitted, so they are visited
                // with fresh element types rather than abandoned behind the
                // diagnostic; leaving them unvisited would be the same coverage
                // defect one level down, and would make §C4.1's owner-stamp
                // tripwire report the collateral node instead of the head.
                let Ok(ctor_name) = read_required_slot(
                    kids,
                    DeepTag::PatCtor,
                    0,
                    SlotShape::ConstructorName,
                    symbol_name,
                    errors,
                ) else {
                    visit_sub_patterns_untyped(
                        kids.get(1..).unwrap_or_default(),
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        product,
                        covered_variants,
                        has_wildcard,
                    );
                    return;
                };
                {
                    // chelis#317: an out-of-scope constructor pattern (a type-
                    // only import that names `| Alpha =>` without importing
                    // `Alpha`) must be rejected at `check` here, the same way
                    // the construction site is in `infer_var`. Without this the
                    // fuzzy terminal fallbacks could otherwise bind the arm to
                    // a foreign module's tag. Resolve through the structural
                    // constructor map and its exact registry owner instead.
                    let resolved_scrutinee = subst.apply(scrutinee_ty);
                    let Some((adt_name, adt_def, variant_info)) = pattern_constructor_for_scrutinee(
                        ctor_name,
                        CallShape::Positional,
                        &resolved_scrutinee,
                        env,
                        adt_reg,
                    ) else {
                        errors.push(CheckError::new(
                            CheckErrorKind::UnknownConstructor {
                                identifier: ctor_name.to_string(),
                            },
                            with_macro_provenance(pat, format!("unknown constructor: {ctor_name}")),
                            vec![format!(
                                "Constructor '{ctor_name}' is not in scope. Declare it \
                                 locally or add it to an import (e.g. \
                                 `import Mod ({ctor_name})`)"
                            )],
                        ));
                        return;
                    };
                    let adt_name = adt_name.to_string();
                    covered_variants.push(variant_info.name.clone());

                    // RFC D-CHECK: constructor pattern match on an
                    // out-of-module opaque type is rejected; binding
                    // inference continues so no error cascades.
                    crate::opacity::check_opaque_use(
                        crate::opacity::OpaqueAction::PatCtor,
                        &adt_name,
                        adt_reg,
                        errors,
                    );

                    // Instantiate the exact nominal variant rather than an
                    // ordinary lexical binding. A parameter with the same
                    // spelling is a value binding, not constructor authority
                    // for a pattern head (chelis#1076).
                    let (arg_types, ret) = instantiate_variant_of(adt_def, variant_info, vg);
                    let _ = unify(&ret, scrutinee_ty, subst);
                    for (i, sub_pat) in kids[1..].iter().enumerate() {
                        if i < arg_types.len() {
                            let resolved = subst.apply(&arg_types[i]);
                            pattern_bindings(
                                sub_pat,
                                &resolved,
                                env,
                                vg,
                                subst,
                                adt_reg,
                                errors,
                                product,
                                covered_variants,
                                has_wildcard,
                            );
                        }
                    }
                }
            }
            DeepTag::PatAs => {
                // (pat-as {} name inner_pat): bind name to scrutinee type, recurse into inner_pat
                //
                // chelis#874 / [04-TOT-4]: the `pat-var` treatment, for the
                // same reason. Stamp the node, reject an unreadable name at its
                // own slot, and still recurse into the inner pattern below.
                let resolved = subst.apply(scrutinee_ty);
                product.record_bypass(pat, resolved.clone(), "pattern binding traversal");
                if let Ok(name) = read_required_slot(
                    kids,
                    DeepTag::PatAs,
                    0,
                    SlotShape::BindingName,
                    symbol_name,
                    errors,
                ) {
                    env.bind_lexical(name.to_string(), Scheme::mono(resolved));
                }
                if kids.len() >= 2 {
                    pattern_bindings(
                        &kids[1],
                        scrutinee_ty,
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        product,
                        covered_variants,
                        has_wildcard,
                    );
                }
            }
            DeepTag::PatRecord => {
                // (pat-record {} TypeName (kv {} k1 p1) ...): validate against ADT registry
                // kids[0] = TypeName, kids[1..] = (kv {} key pat)
                //
                // chelis#874 R3 / [04-TOT-4]: the `pat-ctor` head's twin, with
                // the same unvisited-subtree recovery -- the field patterns live
                // one level down inside `kv` nodes, so the recovery reads each
                // `kv`'s value child rather than the direct children.
                let Ok(ctor_name) = read_required_slot(
                    kids,
                    DeepTag::PatRecord,
                    0,
                    SlotShape::ConstructorName,
                    symbol_name,
                    errors,
                ) else {
                    for kv_expr in kids.iter().skip(1) {
                        let Some((DeepTag::Kv, _, kv_kids)) = stamped_parts(kv_expr) else {
                            continue;
                        };
                        visit_sub_patterns_untyped(
                            kv_kids.get(1..).unwrap_or_default(),
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            product,
                            covered_variants,
                            has_wildcard,
                        );
                    }
                    return;
                };
                {
                    // chelis#317: same out-of-scope guard as the positional
                    // `pat-ctor` arm — a record-shaped match against a
                    // constructor that was never imported must be an `unknown
                    // constructor` error, not a fuzzy bind to a foreign tag.
                    // Structural resolution also rejects the non-unique /
                    // unresolvable case a `_` wildcard arm could otherwise
                    // silently accept.
                    let resolved_scrutinee = subst.apply(scrutinee_ty);
                    let Some((adt_name, adt_def, variant_info)) = pattern_constructor_for_scrutinee(
                        ctor_name,
                        CallShape::Record,
                        &resolved_scrutinee,
                        env,
                        adt_reg,
                    ) else {
                        errors.push(CheckError::new(
                            CheckErrorKind::UnknownConstructor {
                                identifier: ctor_name.to_string(),
                            },
                            with_macro_provenance(pat, format!("unknown constructor: {ctor_name}")),
                            vec![format!(
                                "Constructor '{ctor_name}' is not in scope. Declare it \
                                 locally or add it to an import (e.g. \
                                 `import Mod ({ctor_name})`)"
                            )],
                        ));
                        return;
                    };
                    let adt_name = adt_name.to_string();
                    // RFC D-CHECK: record pattern match on an
                    // out-of-module opaque type is rejected; binding
                    // inference continues so no error cascades.
                    crate::opacity::check_opaque_use(
                        crate::opacity::OpaqueAction::PatRecord,
                        &adt_name,
                        adt_reg,
                        errors,
                    );

                    covered_variants.push(variant_info.name.clone());
                    let declared_field_names: Vec<Option<String>> = variant_info
                        .fields
                        .iter()
                        .map(|(name, _)| name.clone())
                        .collect();
                    let known_field_set: chelis_unord::UnordSet<&str> = declared_field_names
                        .iter()
                        .filter_map(|n| n.as_deref())
                        .collect();

                    // Mirror the `pat-ctor` (positional) path: instantiate
                    // the constructor scheme and unify its return type
                    // with the scrutinee so the ADT's type parameters get
                    // pinned to the scrutinee's concrete instantiation
                    // (e.g. `FooState[a] -> FooState[tensor[n, f32]]`).
                    // The instantiated function's arg types are the
                    // properly substituted per-field types. Without this
                    // step, the declared field types still reference the
                    // ADT's abstract `a`, leaving record-pattern bindings
                    // stuck as fresh type variables and breaking
                    // downstream linearity/borrow checks. (closes #181)
                    let (instantiated_arg_types, instantiated_ret) =
                        instantiate_variant_of(adt_def, variant_info, vg);
                    let _ = unify(&instantiated_ret, scrutinee_ty, subst);

                    for kv_expr in kids.iter().skip(1) {
                        if let Some((DeepTag::Kv, _, kv_kids)) = stamped_parts(kv_expr)
                            && kv_kids.len() >= 2
                        {
                            // chelis#874 / [04-TOT-4]: the third `kv` key read
                            // in the checker, and the one the design's PP8 table
                            // missed. `symbol_name` returning `None` used to fall
                            // through to the `None => vg.fresh_type()` arm below,
                            // so an unreadable key was indistinguishable from a
                            // field whose type the registry could not supply and
                            // the program scored 1.0. The sub-pattern is still
                            // visited afterwards, with that fresh type.
                            let field_name = read_required_slot(
                                kv_kids,
                                DeepTag::Kv,
                                0,
                                SlotShape::FieldName,
                                symbol_name,
                                errors,
                            )
                            .ok();
                            // Look up declared field type — reject unknown fields
                            let field_ty = match field_name {
                                Some(n) => {
                                    if known_field_set.contains(n) {
                                        // Prefer the instantiated arg type from
                                        // the constructor scheme so the
                                        // scrutinee's concrete type arguments
                                        // are reflected in the pattern binding.
                                        let pos = declared_field_names
                                            .iter()
                                            .position(|nm| nm.as_deref() == Some(n));
                                        match pos.and_then(|i| instantiated_arg_types.get(i)) {
                                            Some(ty) => subst.apply(ty),
                                            None => {
                                                variant_info
                                                    .fields
                                                    .iter()
                                                    .find_map(|(name, ty)| {
                                                        (name.as_deref() == Some(n))
                                                            .then(|| ty.clone())
                                                    })
                                                    // Per the loop guard `known_field_set
                                                    // .contains(n)` and the fact that
                                                    // `known_field_set` is derived from
                                                    // `declared_field_names` whose
                                                    // `Some(_)` entries are exactly the
                                                    // named fields of `vi.fields`, the
                                                    // find_map above always returns Some
                                                    // here. The expect makes that explicit;
                                                    // if it ever fires, both data sources
                                                    // are themselves out of sync — a bug
                                                    // upstream of this site.
                                                    .expect(
                                                        "known_field_set is derived from \
                                                             vi.fields' named entries; mismatch \
                                                             indicates a corrupted AdtRegistry",
                                                    )
                                            }
                                        }
                                    } else if !known_field_set.is_empty() {
                                        // Unknown field name — error
                                        report(
                                            errors,
                                            CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                format!(
                                                    "unknown record field '{}' in pattern for {}",
                                                    n, ctor_name
                                                ),
                                                vec![format!(
                                                    "known fields: {:?}",
                                                    declared_field_names
                                                        .iter()
                                                        .filter_map(|f| f.as_deref())
                                                        .collect::<Vec<_>>()
                                                )],
                                            ),
                                        )
                                    } else {
                                        vg.fresh_type() // no ADT info available
                                    }
                                }
                                // The key was rejected above; the sub-pattern
                                // still gets a type so it is visited and bound.
                                None => vg.fresh_type(),
                            };
                            pattern_bindings(
                                &kv_kids[1],
                                &field_ty,
                                env,
                                vg,
                                subst,
                                adt_reg,
                                errors,
                                product,
                                covered_variants,
                                has_wildcard,
                            );
                        }
                    }
                }
            }
            DeepTag::PatTuple => {
                // (pat-tuple {} sub0 sub1 ...): every child is itself a
                // sub-pattern. Recurse into each so a nested
                // `pat-record` / `pat-ctor` reaches the RFC D-CHECK
                // opacity gate (and `pat-var` bindings get the right
                // element type) exactly as a top-level destructure does.
                // Without this recursion the catch-all below silently
                // dropped tuple-nested patterns, bypassing
                // `check_opaque_use` for out-of-module opaque types wrapped
                // in a tuple scrutinee.
                let resolved = subst.apply(scrutinee_ty);
                // Pair each child sub-pattern with the matching tuple
                // element type when the resolved scrutinee is a tuple of
                // equal arity; otherwise hand each child a fresh type
                // variable. The opacity check inside the nested
                // `pat-record` / `pat-ctor` arms keys off the pattern's
                // constructor name, not the scrutinee type, so the gate
                // still fires under a fresh-var element type.
                // chelis#1836: an UNRESOLVED scrutinee is tied to the
                // pattern's own shape here. The pattern fixes the arity, so
                // the scrutinee unifies with a tuple of one fresh element per
                // sub-pattern and each binding IS the corresponding element.
                // Handing every child a disconnected `vg.fresh_type()`
                // instead left `a` in `match q with { | (a, k) => ... }`
                // descending from `q` by name alone: it never bound when `q`
                // did, so a shape-computed route over it published a result
                // the later binding could not contradict, and a false
                // declared shape checked at score 1.
                //
                // A failure to unify is reported rather than dropped: the
                // scrutinee is a variable here, so the only way this can fail
                // is an occurs-check violation, which is a real defect in the
                // program rather than a shape this arm may ignore.
                let tied: Option<Vec<Type>> = match &resolved {
                    Type::Var(_) => {
                        let elems: Vec<Type> = kids.iter().map(|_| vg.fresh_type()).collect();
                        match unify(&resolved, &Type::Tuple(elems.clone()), subst) {
                            Ok(()) => Some(elems),
                            Err(error) => {
                                errors.push(error.into());
                                None
                            }
                        }
                    }
                    _ => None,
                };
                let elem_tys: Option<&[Type]> = match (&resolved, &tied) {
                    (_, Some(elems)) => Some(elems.as_slice()),
                    (Type::Tuple(ts), None) if ts.len() == kids.len() => Some(ts.as_slice()),
                    _ => None,
                };
                for (i, sub_pat) in kids.iter().enumerate() {
                    let elem_ty = match elem_tys {
                        Some(ts) => subst.apply(&ts[i]),
                        None => vg.fresh_type(),
                    };
                    pattern_bindings(
                        sub_pat,
                        &elem_ty,
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        product,
                        covered_variants,
                        has_wildcard,
                    );
                }
            }
            _ => {}
        }
    }
}

/// The scalar family a `pat-lit` value atom denotes, for [04-LIT-1]'s closed
/// atom-to-primitive pairing. A `pat-lit` carries only its raw value, so this
/// is the only type information a literal pattern has.
enum LiteralPatternAtom<'a> {
    Integer(i64),
    Float(f64),
    Bool(bool),
    Str(&'a str),
}

impl LiteralPatternAtom<'_> {
    /// The `[04-LIT-1]` family name, for the diagnostic.
    fn family(&self) -> &'static str {
        match self {
            LiteralPatternAtom::Integer(_) => "integer",
            LiteralPatternAtom::Float(_) => "floating-point",
            LiteralPatternAtom::Bool(_) => "boolean",
            LiteralPatternAtom::Str(_) => "string",
        }
    }

    /// The canonical spelling of the pattern, for the diagnostic.
    fn rendered(&self) -> String {
        match self {
            LiteralPatternAtom::Integer(value) => value.to_string(),
            LiteralPatternAtom::Float(value) => {
                let printed = value.to_string();
                if printed.contains(['.', 'e', 'E', 'n', 'i']) {
                    printed
                } else {
                    format!("{printed}.0")
                }
            }
            LiteralPatternAtom::Bool(value) => value.to_string(),
            LiteralPatternAtom::Str(value) => format!("{value:?}"),
        }
    }

    /// The primitive families this atom may denote, for the diagnostic.
    fn admissible_primitives(&self) -> &'static str {
        match self {
            LiteralPatternAtom::Integer(_) => "an integer primitive",
            LiteralPatternAtom::Float(_) => "a float primitive",
            LiteralPatternAtom::Bool(_) => "`bool`",
            LiteralPatternAtom::Str(_) => "`string`",
        }
    }

    /// The scrutinee type that would admit this pattern, for the repair hint.
    fn admissible_scrutinee(&self) -> &'static str {
        match self {
            LiteralPatternAtom::Integer(_) => "an integer dtype",
            LiteralPatternAtom::Float(_) => "a float dtype",
            LiteralPatternAtom::Bool(_) => "the `bool` type",
            LiteralPatternAtom::Str(_) => "the `string` type",
        }
    }
}

fn literal_pattern_atom(value: &deep::Expr) -> Option<LiteralPatternAtom<'_>> {
    match value {
        deep::Expr::Atom(deep::Atom::Int(value), _) => Some(LiteralPatternAtom::Integer(*value)),
        deep::Expr::Atom(deep::Atom::Float(value), _) => Some(LiteralPatternAtom::Float(*value)),
        deep::Expr::Atom(deep::Atom::Bool(value), _) => Some(LiteralPatternAtom::Bool(*value)),
        deep::Expr::Atom(deep::Atom::Str(value), _) => {
            Some(LiteralPatternAtom::Str(value.as_str()))
        }
        // A `pat-lit` whose child is not a scalar atom is a Deep
        // well-formedness question, not a typing one. Deciding a family for it
        // here would report [04-PAT-1] against a node that has no literal
        // value at all. chelis#1525: that shape now HAS an owner -- the
        // `DeepTag::PatLit` arm reads this slot through the seam and rejects a
        // non-atom child as malformed, so `None` here is the seam's signal
        // rather than a silent decline.
        _ => None,
    }
}

/// chelis#1494 / [04-PAT-1]: a literal pattern is a typing constraint on the
/// scrutinee, so a pattern whose value atom cannot denote the scrutinee's
/// primitive is a `TypeMismatch` rather than a silently dead arm.
///
/// Three ways to violate it, all located at the pattern:
///
/// 1. the scrutinee is not primitive at all (a tensor, nominal, tuple, record,
///    or function scrutinee admits no literal pattern);
/// 2. the atom's family disagrees with the scrutinee primitive's family under
///    [04-LIT-1]'s closed pairing;
/// 3. an integer pattern lies outside the scrutinee integer width's range,
///    under the same range rule spec/04 §5.3 and §5.6 apply to a literal bound
///    at that type. A float primitive has no such range: finalization at a
///    float width is total under [04-NUM-1].
///
/// A literal pattern selects no width, because a `pat-lit` has no precision
/// slot and admits no suffix (spec/02 §P10a), so an unsuffixed integer pattern
/// is admissible against every integer primitive. Unifying it with §5.3's
/// `int32` default instead would reject `match x_int64 with { | 1 => ... }` and
/// leave no spelling for an `int64` literal pattern.
fn check_literal_pattern(
    pat: &deep::Expr,
    atom: LiteralPatternAtom<'_>,
    scrutinee_ty: &Type,
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) {
    let resolved = adt_reg.expand_aliases(&subst.apply(scrutinee_ty));
    // An unresolved scrutinee decides nothing yet, and an already-failed one
    // owns its own diagnostic: reporting here would either invent a rejection
    // or cascade off a root cause reported upstream (chelis#731 section C3).
    //
    // chelis#1525: this early return is why the structural read of the value
    // child does NOT live here. Well-formedness of the `pat-lit` form cannot
    // depend on whether the scrutinee's type happens to be resolved, so the
    // caller reads the slot first and this function receives an atom it can
    // always decide about.
    if matches!(resolved, Type::Var(_) | Type::Error(_)) {
        return;
    }

    let Type::Prim(prim) = &resolved else {
        report_literal_pattern_error(
            pat,
            errors,
            format!(
                "{} literal pattern `{}` cannot match a scrutinee of type `{resolved}`: a \
                 literal pattern is admissible only against a primitive scrutinee \
                 (spec/04-type-system.md [04-PAT-1])",
                atom.family(),
                atom.rendered(),
            ),
            vec![
                "Destructure the scrutinee with a constructor, record, or tuple pattern, or \
                 match a primitive field of it, instead of comparing it to a literal"
                    .to_string(),
            ],
        );
        return;
    };

    let admissible = match &atom {
        LiteralPatternAtom::Integer(_) => prim.is_integer(),
        LiteralPatternAtom::Float(_) => prim.is_float(),
        LiteralPatternAtom::Bool(_) => *prim == Prim::Bool,
        LiteralPatternAtom::Str(_) => *prim == Prim::String,
    };
    if !admissible {
        report_literal_pattern_error(
            pat,
            errors,
            format!(
                // No indefinite article before the dtype: the correct one is
                // pronunciation-dependent rather than spelling-dependent
                // ("an f32", "a bf16", "a bool"), so the phrasing avoids the
                // choice instead of encoding a pronunciation table. It also
                // matches the non-primitive arm above.
                "{} literal pattern `{}` cannot match a scrutinee of type `{}`: a literal \
                 pattern denotes only {} (integer to integer, float to float, boolean to \
                 `bool`, string to `string`), so this arm could never match \
                 (spec/04-type-system.md [04-PAT-1], [04-LIT-1])",
                atom.family(),
                atom.rendered(),
                prim.name(),
                atom.admissible_primitives(),
            ),
            vec![format!(
                "Write a pattern in the scrutinee's own family, or give the scrutinee {}. \
                 A literal pattern carries no suffix and no cast, so no conversion is \
                 available here (spec/02-surf-syntax.md section P10a)",
                atom.admissible_scrutinee(),
            )],
        );
        return;
    }

    if let LiteralPatternAtom::Integer(literal) = &atom
        && let Some((low, high)) = prim.integer_range()
        && (*literal < low || *literal > high)
    {
        report_literal_pattern_error(
            pat,
            errors,
            format!(
                "integer literal pattern `{literal}` is outside the `{}` range \
                 [{low}, {high}], so this arm could never match \
                 (spec/04-type-system.md [04-PAT-1], section 5.3)",
                prim.name(),
            ),
            vec![format!(
                "Use a value the scrutinee's `{}` width can hold, or widen the scrutinee",
                prim.name(),
            )],
        );
    }
}

/// Push one [04-PAT-1] rejection, located at the pattern node so downstream
/// tooling can point at the arm that carries it.
fn report_literal_pattern_error(
    pat: &deep::Expr,
    errors: &mut DiagnosticSink<'_>,
    message: String,
    suggestions: Vec<String>,
) {
    let mut error = CheckError::new(
        CheckErrorKind::TypeMismatch,
        with_macro_provenance(pat, message),
        suggestions,
    );
    if let Some(id) = pat.span_id() {
        error.span_offset = parse_span_offset(id);
        error.span_id = Some(id.to_string());
    } else if pat.span().offset > 0 {
        error.span_offset = Some(pat.span().offset);
    }
    errors.push(error);
}

/// A `pipe` node reached inference.
///
/// It cannot, from any checker entry: `chelis_deep::pipe::fold_pipe` states
/// `spec/02-surf-syntax.md` section 0.1's sentence -- `x |> f(y)` MEANS
/// `f(x, y)` -- once, over every entry's input, so inference only ever sees
/// the application. The rule that used to live here typed a stage from the
/// callee's FUNCTION type instead of as that application, which lost every
/// rule keyed on an application's arguments: `to_tensor`'s literal shape,
/// `sum`'s axis, `expand`'s size (chelis#1923, chelis#1791).
///
/// So this arm exists to make the class impossible to reintroduce quietly
/// rather than to handle a case. A pipe arriving here means an entry was
/// added that does not fold, and saying so is worth more than typing it a
/// second way.
pub(super) fn pipe_reached_inference_unfolded(
    list: &deep::List,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    let stages = children(list).len().saturating_sub(1);
    report(
        errors,
        CheckError::new(
            CheckErrorKind::MalformedForm,
            format!(
                "a pipe reached inference unfolded ({stages} stage(s)): every checker entry \
                 folds a pipe into the application it denotes before inference \
                 (spec/02-surf-syntax.md section 0.1; chelis#1923)"
            ),
            vec![],
        ),
    )
}
