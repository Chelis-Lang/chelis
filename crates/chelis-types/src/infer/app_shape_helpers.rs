//! Static shape readers and shape helpers.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

/// Extract the static shape of a `to_tensor` argument when the
/// argument is a nested Cons-chain literal whose every leaf is a
/// numeric/bool atom (or a recognized `cast` / `neg` wrapper).
///
/// Returns `Some(dims)` with one `Dim::Lit(n)` per axis (outermost
/// first) when the structure is statically resolvable and every
/// axis is uniformly shaped. Returns `None` when:
///   * the argument is not a Cons-chain (e.g. a variable like
///     `to_tensor(items)`),
///   * the chain is malformed or not closed by `Nil`,
///   * a leaf is not a recognizable numeric atom,
///   * sibling axes have different lengths (ragged literal).
///
/// `expected_rank` is the rank inferred from peeling List wrappers
/// in the argument's TYPE; it is used as a sanity check, not as a
/// hard requirement. A mismatch returns `None` so the caller falls
/// back to the legacy wildcard-rank path.
///
/// This is the issue Chelis-Lang/chelis#218 R2 HIGH-A source fix:
/// by emitting concrete dims here, every downstream consumer
/// (reductions, elementwise activations, anything that reads the
/// to_tensor app's `type:` metadata) sees a sound shape instead of
/// a `Dim::Wildcard`.
pub(super) fn static_to_tensor_shape(arg: &deep::Expr, expected_rank: usize) -> Option<Vec<Dim>> {
    let dims = walk_static_cons_chain_shape(arg)?;
    if dims.len() != expected_rank {
        return None;
    }
    Some(dims.into_iter().map(|n| Dim::Lit(n as i64)).collect())
}

/// Recursive helper for `static_to_tensor_shape`. Returns
/// `Some(dims)` if `expr` is a Cons / Nil chain whose every leaf
/// reduces to a numeric atom (or recursively to another Cons chain
/// of uniform length). `dims` is the rank-N shape (outermost axis
/// first).
pub(super) fn walk_static_cons_chain_shape(expr: &deep::Expr) -> Option<Vec<usize>> {
    stack_guard!("walk_static_cons_chain_shape", expr, None);
    let elements = collect_cons_chain_for_shape(expr)?;
    if elements.is_empty() {
        // Empty list at the outermost level has rank 1, size 0.
        // For empty inner lists we still want a concrete dim list,
        // but ragged-but-empty cases can't be uniformly typed; the
        // top-level case is sufficient here.
        return Some(vec![0]);
    }
    // If every element is a numeric leaf, this is a rank-1 axis.
    if elements
        .iter()
        .all(|element| extract_numeric_leaf_for_shape(element).is_some())
    {
        return Some(vec![elements.len()]);
    }
    // Otherwise every element should recurse to a same-shape
    // sub-vector. The outermost axis is `elements.len()`; the inner
    // axes must agree.
    let nested: Vec<Vec<usize>> = elements
        .iter()
        .map(|element| walk_static_cons_chain_shape(element))
        .collect::<Option<_>>()?;
    let inner_shape = nested.first()?.clone();
    if nested.iter().any(|shape| shape != &inner_shape) {
        return None;
    }
    let mut out = vec![nested.len()];
    out.extend(inner_shape);
    Some(out)
}

/// Collect a `Cons(head, Cons(head, ..., Nil))` chain into a vector
/// of head expressions. Returns `None` if the chain isn't closed by
/// `Nil` or contains a non-Cons app. Local helper for
/// `walk_static_cons_chain_shape`; mirrors `cons_chain_int_dims`'s
/// chain-walking shape but returns the heads themselves so the
/// caller can recurse.
pub(super) fn collect_cons_chain_for_shape(expr: &deep::Expr) -> Option<Vec<&deep::Expr>> {
    let mut out = Vec::new();
    let mut cursor = expr;
    loop {
        // chelis#1107: carrier-preserving read. A `List`-only destructure gave
        // up on every stamped node, so no cons chain was ever recognized on
        // `check_typed_program`.
        let (tag, _, kids) = stamped_parts(cursor)?;
        match tag {
            DeepTag::Var => {
                let name = kids.first().and_then(symbol_name)?;
                if name == "Nil" {
                    return Some(out);
                }
                return None;
            }
            DeepTag::App => {
                let func = kids.first()?;
                if !is_builtin_var(func, "Cons") {
                    return None;
                }
                let head = kids.get(1)?;
                let tail = kids.get(2)?;
                out.push(head);
                cursor = tail;
            }
            _ => return None,
        }
    }
}

/// True iff `expr` is a numeric leaf (Int/Float/Bool atom, `lit` of
/// the same, `cast` of one, or `neg` of one). Mirrors the shape of
/// `extract_numeric_leaf` in `crates/chelis-ir/src/lower.rs` so the
/// type-check and IR-lowering passes agree on what counts as a
/// "static to_tensor leaf." We don't need the actual value here,
/// only the static-recognition predicate.
pub(super) fn extract_numeric_leaf_for_shape(expr: &deep::Expr) -> Option<()> {
    stack_guard!("extract_numeric_leaf_for_shape", expr, None);
    if matches!(
        expr,
        deep::Expr::Atom(
            deep::Atom::Int(_) | deep::Atom::Float(_) | deep::Atom::Bool(_),
            _
        )
    ) {
        return Some(());
    }
    // chelis#1107 round 3: this used to be an `Expr::List`-only match arm with
    // a non-erroring `_ => None` default, so a stamped `lit`/`cast`/`neg` leaf
    // fell to the default, `walk_static_cons_chain_shape` gave up, and
    // `to_tensor` produced a rank-1 WILDCARD instead of the real element
    // count. That masked a genuine count mismatch: `to_tensor([-1.0, -2.0])`
    // declared `tensor[3, f32]` was accepted by `check_typed_program` and
    // rejected by `check_ir_program`. It bites positive literals too -- the
    // `neg` recognizer below is only one of the three shapes that were lost.
    let (tag, _, kids) = stamped_parts(expr)?;
    match tag {
        DeepTag::Lit => match kids.first()? {
            deep::Expr::Atom(deep::Atom::Int(_), _)
            | deep::Expr::Atom(deep::Atom::Float(_), _)
            | deep::Expr::Atom(deep::Atom::Bool(_), _) => Some(()),
            _ => None,
        },
        DeepTag::Cast => extract_numeric_leaf_for_shape(kids.first()?),
        DeepTag::App => {
            // Issue #218 R1 HIGH-1 mirror: a surface negative
            // literal `-x` desugars to `(app (var neg) <inner>)`.
            // Recurse through the unary minus so the static
            // recognizer matches the IR lowering's analogous
            // recognizer in `crates/chelis-ir/src/lower.rs`.
            let callee = kids.first()?;
            if !is_builtin_var(callee, "neg") {
                return None;
            }
            extract_numeric_leaf_for_shape(kids.get(1)?)
        }
        _ => None,
    }
}

pub(super) fn symbolic_dim_ref_name(expr: &deep::Expr) -> Option<&str> {
    // chelis#1107: carrier-preserving read; a stamped named axis (`sum(x, seq)`)
    // was unrecognizable on the typed ingress.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag != DeepTag::Var {
        return None;
    }
    kids.first().and_then(symbol_name)
}

/// True when `name` is a dimension binder of the enclosing definition. A
/// listed name its signature uses only as a precision or rank binder is not
/// one.
fn definition_binds_dimension(env: &Env, name: &str) -> bool {
    env.type_resolution_binders()
        .is_some_and(|binders| binders.binds_dimension(name))
}

/// True when `name` is a dimension of the definition being checked: a binder
/// of the enclosing definition, or a dimension a tensor type in scope carries
/// (spec/04-type-system.md section 4.7.2, "an in-scope symbolic dimension may
/// preserve its name").
pub(super) fn definition_dimension(env: &Env, name: &str, subst: &Subst) -> bool {
    definition_binds_dimension(env, name) || env.tensor_carries_dim_with_subst(name, subst)
}

/// The type error for an `expand`/`insert` size that names something both a
/// value and a dimension (spec/04-type-system.md section 4.7.2).
///
/// A size expression reads values, and a bare name in it may also preserve a
/// dimension's identity. When one name occurrence denotes both a value binding
/// (a parameter, a local or a top-level binding) and an in-scope dimension (a
/// binder of the enclosing definition, or a dimension a tensor type in scope
/// carries), the two readings can give different extents, and nothing in the
/// program says which is meant. Every free name of the size counts, under
/// `cast` and arithmetic as well as bare, so no spelling of the size can pick
/// a reading silently (chelis#469).
pub(super) fn ambiguous_size_name_error(
    builtin: &str,
    size: &deep::Expr,
    env: &Env,
    subst: &Subst,
) -> Option<CheckError> {
    crate::linearity::free_runtime_variables(size)
        .into_iter()
        .find_map(|name| {
            let value = match env.top_level_value_visibility(&name) {
                TopLevelValueVisibility::Visible => env.lookup(&name),
                TopLevelValueVisibility::NotYetDeclared { shadowed } => shadowed,
            }?;
            let dimension = if definition_binds_dimension(env, &name) {
                "a dimension binder of the enclosing definition"
            } else if env.tensor_carries_dim_with_subst(&name, subst) {
                "a dimension carried by a tensor type in scope"
            } else {
                return None;
            };
            let value_ty = subst.apply(&value.body);
            Some(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "{builtin} size names `{name}`, which is both a value of type {value_ty} \
                     and {dimension}, so argument 3 (size) has an ambiguous extent \
                     (spec/04-type-system.md \u{00a7}4.7.2)"
                ),
                vec![format!(
                    "Rename the value or the dimension so `{name}` has one meaning; \
                     `shape(t, axis)` reads a tensor's extent explicitly."
                )],
            ))
        })
}

/// The facts a binding to `rhs` carries, read against the pre-binding scope.
///
/// A right-hand side that folds to a checked integer constant lets a later
/// `expand`/`insert` size naming the binding type a literal extent, and a
/// list literal records its element count for `concat` (chelis#631). The
/// facts travel on the binding entry ([`Env::bind_with_facts`]), so any later
/// binding of the name, by any binder, replaces them. This records VALUES
/// only: a size of any provenance is admissible (spec/04-type-system.md
/// section 4.7.2, chelis#469), so nothing here classifies where a
/// non-constant value came from.
pub(super) fn rhs_binding_facts(env: &Env, rhs: &deep::Expr) -> BindingFacts {
    BindingFacts {
        static_size: fold_static_int_expr(rhs, |bound| env.static_size_value(bound)),
        list_literal_len: static_list_len(Some(rhs), env),
    }
}

/// Describe a non-literal axis argument for the issue #259 diagnostic.
///
/// When the axis is a `(var name)` (the common case: a function-parameter
/// `i32` such as `mean(&x, ax)`), name it so the user can see which
/// binding is the runtime value. Otherwise fall back to a generic
/// "non-constant expression" phrasing. Kept deliberately small: this only
/// feeds a user-facing message, not a control-flow decision.
pub(super) fn describe_axis_arg(axis_expr: Option<&deep::Expr>) -> String {
    match axis_expr {
        Some(expr) => match symbolic_dim_ref_name(expr) {
            Some(name) => format!("a runtime value `{name}`"),
            None => "a non-constant expression".to_string(),
        },
        None => "a missing argument".to_string(),
    }
}

/// Render a declared dimension parameter's identity for a diagnostic
/// (chelis#260, spec/04 [04-FIT-9] and [04-FIT-10]).
///
/// The source spelling when the signature recorded one, quoted the way this
/// chapter's diagnostics quote authored names. Otherwise the internal
/// `DimVar` id, bare: [04-FIT-10] admits a synthesized identity only where
/// provenance genuinely does not exist, and requires it be distinguishable
/// from a spelling the user wrote. The absence of the quoting is that
/// distinction, so the two arms may not converge on one form.
///
/// Rendering is per variable, not per message. A signature that recorded
/// some of its parameters names those and falls back for the rest, rather
/// than suppressing the whole diagnostic or attaching a recorded name to a
/// variable it does not belong to.
pub(super) fn render_declared_dim(dim_names: &UnordMap<DimVar, String>, dv: DimVar) -> String {
    match dim_names.get(&dv) {
        Some(name) => format!("`{name}`"),
        None => format!("d{}", dv.0),
    }
}

/// Post-body rigidity check for a def's declared dimension parameters.
///
/// Classify and check every dimension identity owned by one declaration.
///
/// The structural binder list owns the complete identity set. Signature
/// parameter occurrences are rigid, result-only occurrences are
/// output-inferred under §4.4.1, and identities absent from the outer
/// signature but used by body annotations are rigid under [04-INF-6].
///
/// A collapse is accepted only when both identities are return-only. If
/// either identity is parameter-position or body-only, the authored
/// universal contract has been narrowed and the declaration is rejected.
pub(super) fn check_authored_dvars_rigid(
    declaration: &str,
    decl_ty: &Type,
    dim_names: &UnordMap<DimVar, String>,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    let Type::Fn(decl_params, decl_ret) = decl_ty else {
        // A non-function declaration has no parameter/result role split:
        // every authored dimension in its declared value type is rigid under
        // [04-INF-6]. In particular, §4.4.1's return-only output-inference
        // exception applies only to a function result.
        let mut rigid_dvars = dim_names
            .to_sorted()
            .into_iter()
            .map(|(dv, _)| *dv)
            .collect::<Vec<_>>();
        rigid_dvars.sort_by_key(|dv| dv.0);
        check_declared_dvars_rigid(Some(declaration), &rigid_dvars, dim_names, subst, errors);
        return;
    };

    let mut param_dvars = Vec::new();
    for ty in decl_params {
        for dv in crate::env::free_dvars(ty) {
            if !param_dvars.contains(&dv) {
                param_dvars.push(dv);
            }
        }
    }
    let mut return_only_dvars = Vec::new();
    for dv in crate::env::free_dvars(decl_ret) {
        if dim_names.contains_key(&dv)
            && !param_dvars.contains(&dv)
            && !return_only_dvars.contains(&dv)
        {
            return_only_dvars.push(dv);
        }
    }
    let rigid_dvars = dim_names
        .to_sorted()
        .into_iter()
        .map(|(dv, _)| *dv)
        .filter(|dv| !return_only_dvars.contains(dv))
        .collect::<Vec<_>>();

    check_declared_dvars_rigid(Some(declaration), &rigid_dvars, dim_names, subst, errors);

    // chelis#273 return-position rigidity guard. A return-only identity may
    // remain unbound, resolve to a body-internal concrete output, or collapse
    // with another return-only identity. It may not derive from a declared
    // parameter dimension or collapse with any rigid authored identity.
    //
    // Known §4.4.1 boundary: coupling through a named symbolic dim is not
    // flagged unless another authored dimension identity participates.
    let render = |dv: DimVar| render_declared_dim(dim_names, dv);
    let mut param_dims = Vec::new();
    for ty in decl_params {
        crate::env::collect_dims(ty, &mut param_dims);
    }
    let resolved_param_dims = param_dims
        .iter()
        .map(|dim| subst.constraint_dim(dim))
        .collect::<Vec<_>>();
    for dv in &return_only_dvars {
        let constraint = subst.constraint_dim(&Dim::Var(*dv));
        let resolved = if matches!(constraint, Dim::Lit(_)) {
            constraint
        } else {
            subst.semantic_dim(&Dim::Var(*dv))
        };
        if let Dim::Lit(n) = resolved {
            if resolved_param_dims.contains(&Dim::Lit(n)) {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "return-position dim parameter {} was pinned to concrete \
                         Lit({n}) flowing from a declared parameter dimension: a dim \
                         parameter that appears only in the return type promises an \
                         output dimension the body must not derive from the inputs \
                         (spec/04-type-system.md \u{00a7}4.4.1)",
                        render(*dv)
                    ),
                    vec![
                        "Name the input dimension in the return type (reuse the \
                         parameter's dim parameter or literal) if the output \
                         genuinely tracks an input dimension, or fix the body so the \
                         output dimension does not depend on the input dims"
                            .to_string(),
                    ],
                ));
            }
        } else if let Some(rigid_dv) = rigid_dvars
            .iter()
            .find(|rigid_dv| subst.semantic_dim(&Dim::Var(**rigid_dv)) == resolved)
        {
            if param_dvars.contains(rigid_dv) {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "return-position dim parameter {} was unified with the distinct \
                         declared dim parameter {} from a parameter position: declared \
                         dim parameters are rigid and must remain distinct \
                         (spec/04-type-system.md \u{00a7}4.4.1)",
                        render(*dv),
                        render(*rigid_dv)
                    ),
                    vec![
                        "Use the same dim parameter in both positions if the return \
                         dimension is meant to equal the input's, or fix the body so the \
                         declared dims stay independent"
                            .to_string(),
                    ],
                ));
            } else {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "return-position dim parameter {} and rigid body-only dim parameter {} \
                         of `{declaration}` were unified by the function body: a collapse \
                         involving an authored rigid dimension binder is a type error \
                         (spec/04-type-system.md §3.1.3 [04-INF-6])",
                        render(*dv),
                        render(*rigid_dv)
                    ),
                    vec![
                        "Use the same binder if the dimensions are meant to be equal, or fix the \
                         body so the body-only binder remains independent. Only two return-only \
                         binders may share one body-inferred output dimension."
                            .to_string(),
                    ],
                ));
            }
        }
    }
}

/// `declared_dvars` are the rigid dim variables introduced by a declaration
/// or function contract, snapshotted before body inference. After the body is
/// inferred, each must stay distinct: the body must type-check for *all*
/// instantiations of those dims.
///
/// Two failure modes are flagged here, both `DimensionMismatch`:
///
/// - `Dim::Var -> Dim::Lit`: the body forced a polymorphic dim
///   parameter to a concrete literal (Nautilus Bug 2). The signature's
///   polymorphism claim is self-contradictory.
/// - `Dim::Var -> Dim::Var` (or any shared resolution) collapse: two
///   *distinct* declared dim parameters resolved to the *same*
///   dimension after body inference. The body unified two
///   universally-quantified dim parameters that must stay distinct.
///   This is Path B of `TypeCheck-FreeDimVarUnification-F1` (SR-LEAK-A):
///   `def g[n, m](x: tensor[n, f32], y: tensor[m, f32]) ->
///   tensor[n, f32] = y` collapses `n` and `m` via free `unify_dim`
///   and never routes through the Shape A relaxed-retry guard.
///
/// A single declared dim parameter appearing in multiple param
/// positions (`def h[n](x: tensor[n], y: tensor[n])`) is one dvar and
/// never trips the collapse check.
pub(super) fn check_declared_dvars_rigid(
    declaration: Option<&str>,
    declared_dvars: &[DimVar],
    dim_names: &UnordMap<DimVar, String>,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    let render = |dv: DimVar| render_declared_dim(dim_names, dv);
    let owner = declaration.map_or_else(String::new, |name| format!(" in declaration `{name}`"));
    // First resolved dim seen -> the declared dvar that produced it.
    // A second declared dvar resolving to the same dim is a collapse.
    let mut seen: Vec<(Dim, DimVar)> = Vec::new();
    for dv in declared_dvars {
        let constraint = subst.constraint_dim(&Dim::Var(*dv));
        let resolved = if matches!(constraint, Dim::Lit(_)) {
            constraint
        } else {
            subst.semantic_dim(&Dim::Var(*dv))
        };
        if let Dim::Lit(n) = resolved {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "polymorphic dim parameter {} forced to concrete Lit({n}) by function \
                     body{owner}: an authored dimension binder is rigid and the body must \
                     type-check for every dimension \
                     (spec/04-type-system.md §3.1.3 [04-INF-6])",
                    render(*dv),
                ),
                vec![
                    "Replace the polymorphic dim with the concrete literal in the signature, or \
                     ensure the body does not pin the dim to a specific size"
                        .to_string(),
                ],
            ));
            continue;
        }
        // A declared dim parameter that resolves to itself (still
        // unbound) is the legitimate polymorphic case; it cannot
        // collide with another declared dvar's distinct identity.
        if let Some((_, prev)) = seen.iter().find(|(candidate, _)| candidate == &resolved) {
            if *prev != *dv {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "distinct declared dim parameters {} and {} were unified by the function \
                         body{owner}: authored dimension binders are rigid and must remain \
                         distinct (spec/04-type-system.md §3.1.3 [04-INF-6])",
                        render(*prev),
                        render(*dv)
                    ),
                    vec![
                        "The body returns or constrains a value whose dimension differs from \
                         the declared one. Use the same dim parameter on both sides if they \
                         are meant to be equal, or fix the body so each declared dim stays \
                         independent"
                            .to_string(),
                    ],
                ));
            }
        } else {
            seen.push((resolved, *dv));
        }
    }
}

/// chelis#272 list-uniformity guard.
///
/// A declared return type of the form `List[tensor[..., d, ...]]` whose
/// element dim `d` is a *rigid/named* dimension (`Dim::Var` for a
/// declared dim parameter, or `Dim::Name` for a named symbolic dim)
/// promises that every list element has the *same* length at that axis.
/// The #218 Cons-join widens a *mismatched-concrete* list-element axis
/// to `Dim::Wildcard`, and `unify_dim` lets that wildcard satisfy a
/// rigid `Var`/`Name` permissively *without binding it* — so neither the
/// pin-to-literal nor the distinct-collapse arm of
/// `check_declared_dvars_rigid` observes the violation.
///
/// This check closes that gap structurally: it walks the declared type
/// and the resolved body type in parallel and flags any list-element
/// tensor axis where the declaration names a rigid/named dim but the
/// body produced a `Wildcard`. It is deliberately scoped to *list
/// element* tensors (`List[tensor[...]]`), the surface where the #272
/// soundness gap lives; it does not touch bare `tensor[...]` returns
/// whose wildcard axes legitimately flow from `expand`/`reshape`/`shape`
/// (§4.7), where the declared return's named dim binds the result tvar
/// directly rather than being absorbed by a heterogeneous-list wildcard.
///
/// chelis#276: the descent through `List` wrappers is *recursive*, so a
/// wildcard tensor nested under `List[List[tensor[k, f32]]]` (or any
/// deeper nesting) is checked against the inner rigid `k` too. `List[L]`
/// is statically homogeneous in `L`, so the uniformity promise of an
/// inner `List[tensor[k]]` holds at every depth; a single-level check
/// left the same #272 soundness gap one `List` deeper.
pub(super) fn check_list_elem_rigid_dim_vs_wildcard(
    decl_ty: &Type,
    body_ty: &Type,
    dim_names: &UnordMap<DimVar, String>,
    errors: &mut DiagnosticSink<'_>,
) {
    match (decl_ty, body_ty) {
        // Descend through the function type to its return position.
        (Type::Fn(_, decl_ret), Type::Fn(_, body_ret)) => {
            check_list_elem_rigid_dim_vs_wildcard(decl_ret, body_ret, dim_names, errors);
        }
        // `List[T]`: check the element type. The list element is where
        // the uniformity promise lives. When the element is a tensor we
        // compare its declared vs body axes here; when it is itself a
        // `List` (or any further nesting), we recurse so the inner rigid
        // dim is protected at arbitrary depth (chelis#276).
        (Type::Adt(dn, dargs), Type::Adt(bn, bargs))
            if dn == "List" && bn == "List" && dargs.len() == 1 && bargs.len() == 1 =>
        {
            match (&dargs[0], &bargs[0]) {
                (Type::Tensor(ddims, _), Type::Tensor(bdims, _)) if ddims.len() == bdims.len() => {
                    for (dd, bd) in ddims.iter().zip(bdims.iter()) {
                        let rigid = matches!(dd, Dim::Var(_) | Dim::Name(_));
                        if rigid && matches!(bd, Dim::Wildcard) {
                            let promised = match dd {
                                Dim::Var(v) => {
                                    format!("dim parameter {}", render_declared_dim(dim_names, *v))
                                }
                                Dim::Name(n) => format!("named dimension `{n}`"),
                                _ => unreachable!(),
                            };
                            errors.push(CheckError::new(
                                CheckErrorKind::DimensionMismatch,
                                format!(
                                    "list element dimension is unknown (wildcard) in the function \
                                     body but the declared element type promises a uniform \
                                     {promised}: a heterogeneous list literal cannot satisfy a \
                                     declared List[tensor[..]] whose element dimension names a \
                                     rigid/named axis"
                                ),
                                vec![
                                    "Every element of a `List[tensor[k, ..]]` must share the same \
                                     length `k`. Either give the elements a uniform dimension, or \
                                     declare the element axis as a concrete literal / wildcard \
                                     (`tensor[*, ..]`) if the lengths genuinely differ"
                                        .to_string(),
                                ],
                            ));
                        }
                    }
                }
                // Nested `List[...]`: recurse into the element so an inner
                // rigid dim under `List[List[tensor[k]]]` is still checked
                // (chelis#276).
                (decl_elem, body_elem) => {
                    check_list_elem_rigid_dim_vs_wildcard(decl_elem, body_elem, dim_names, errors);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod source_identity_tests {
    use super::*;

    fn names(pairs: &[(DimVar, &str)]) -> UnordMap<DimVar, String> {
        pairs
            .iter()
            .map(|(dv, name)| (*dv, (*name).to_string()))
            .collect()
    }

    /// spec/04 [04-FIT-9]: where provenance exists, the source spelling is
    /// the identity. The inference id does not accompany it.
    #[test]
    fn a_recorded_parameter_renders_its_source_spelling() {
        let recorded = names(&[(DimVar(1), "n"), (DimVar(2), "m")]);
        assert_eq!(render_declared_dim(&recorded, DimVar(1)), "`n`");
        assert_eq!(render_declared_dim(&recorded, DimVar(2)), "`m`");
        assert!(
            !render_declared_dim(&recorded, DimVar(1)).contains('1'),
            "a known spelling carries no inference id"
        );
    }

    /// spec/04 [04-FIT-10]: with no provenance the inference id is rendered
    /// as synthesized, and no name is invented for it.
    ///
    /// Distinguishability is the requirement, so it is asserted as a relation
    /// against the authored rendering rather than against a hard-coded
    /// spelling: a future synthesized form stays correct as long as it cannot
    /// be read as something the user wrote.
    #[test]
    fn an_unrecorded_parameter_renders_a_distinguishable_synthesized_id() {
        let synthesized = render_declared_dim(&UnordMap::new(), DimVar(1));
        assert_eq!(synthesized, "d1", "the fallback renders the internal id");
        assert!(
            !synthesized.contains('`'),
            "the synthesized form must not borrow the authored form's quoting, or a \
             reader cannot tell an invented identity from a written one; got {synthesized}"
        );
        assert_ne!(
            synthesized,
            render_declared_dim(&names(&[(DimVar(1), "n")]), DimVar(1)),
            "a synthesized identity must not render identically to an authored one"
        );
    }

    /// Rendering is per variable. A signature that recorded only some of its
    /// parameters names those and falls back for the rest, rather than
    /// suppressing the message or misattributing a recorded name.
    #[test]
    fn a_partially_recorded_signature_renders_each_variable_from_its_own_provenance() {
        let partial = names(&[(DimVar(2), "m")]);
        assert_eq!(render_declared_dim(&partial, DimVar(1)), "d1");
        assert_eq!(render_declared_dim(&partial, DimVar(2)), "`m`");
    }
}
