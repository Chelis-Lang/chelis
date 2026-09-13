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

/// chelis#397/#469: the materializability class of a runtime `expand` size
/// argument, by PROVENANCE rather than surface spelling.
///
/// A runtime `expand` size has a backend representation only when its
/// extent is recoverable. The discriminator must be uniform across the
/// bare-`var`, `cast`-wrapped, `let`-bound, and arithmetic spellings (the
/// four that #397's red team found drifting): all reduce to one of these
/// classes via the same recursive walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SizeClass {
    /// Folds to a compile-time constant (literal/`cast(N,_)`/arithmetic
    /// over such values, or a `let` name marked `Static`). The host runtime
    /// and the evaluator can compute it; a materializable extent.
    Static,
    /// Provably derives from an in-scope tensor's `shape(t, axis)` read, or
    /// names an in-scope tensor dimension (§4.7.2 Form-2) — directly,
    /// through `cast`, through integer arithmetic, or transitively through a
    /// `let` name marked `ShapeSourced`. The backend reads the extent from
    /// the tensor's shape.
    ShapeSourced,
    /// A runtime value with no static value and no tensor source — a bare
    /// `int32`/`int64` parameter, a `cast`/arithmetic over one, or a `let`
    /// name bound to such. No backend representation; rejected at check
    /// (#469) so check↔build↔eval agree.
    Sourceless,
    /// Not a recognized int-valued size shape (e.g. the input tensor is
    /// still a type var, or the expr is something the walk does not model).
    /// The caller leaves the existing non-rejecting behavior in place.
    Unknown,
}

/// The §4.7.2 sourceless-size diagnostic (chelis#469), emitted by
/// `check_expand_signature` when an `expand` size resolves to `Sourceless`.
/// Factored out so the diagnostic text has a single source of truth.
/// `size_expr` is the size sub-expression (used only to name a symbolic
/// dimension when the size is a bare `var`).
pub(super) fn sourceless_expand_size_error(
    builtin: &'static str,
    size_expr: Option<&deep::Expr>,
) -> CheckError {
    let described = size_expr
        .and_then(symbolic_dim_ref_name)
        .map(|name| format!("the symbolic dimension `{name}`"))
        .unwrap_or_else(|| "a runtime scalar".to_string());
    CheckError::new(
        CheckErrorKind::DimensionMismatch,
        format!(
            "`{builtin}` size resolves to {described}, but no tensor in scope carries \
             it. Runtime extents use exact `int64`; source the value from an in-scope \
             tensor dimension or a `shape(tensor, int32-axis)` read. A bare runtime \
             scalar has no shape identity to attach to the result yet. Tracked by \
             Chelis-Lang/chelis#469 (spec/04-type-system.md \u{00a7}4.7.2)"
        ),
        vec![
            "Source the extent from a tensor in scope with \
             `shape(x, cast(axis, int32))`, bind that read to a `let`, or use an \
             exact `int64` literal extent."
                .to_string(),
        ],
    )
}

/// chelis#397/#469: classify a runtime `expand` size argument by
/// provenance. Walks `cast`, integer arithmetic (`add`/`sub`/`mul`/`div`),
/// bare `var` references (resolved against `env` — an in-scope tensor dim
/// is Form-2 `ShapeSourced`, a recorded `let` provenance is followed, a
/// bare value binding with neither is `Sourceless`), and `shape(t, axis)`
/// reads. The recursion mirrors the IR layer's
/// `extract_dim_expr_value`/`symbol_has_tensor_source`/`shape_dep` triad so
/// the check-time accept set matches what the backends can materialize.
pub(super) fn classify_expand_size(
    expr: &deep::Expr,
    env: &Env,
    adt_reg: &AdtRegistry,
    subst: &Subst,
) -> SizeClass {
    classify_size_in(
        expr,
        &SizeCtx {
            env,
            adt_reg,
            subst,
            piped: &[],
        },
    )
}

/// What `classify_expand_size` reads while it walks: the type environment,
/// the ADT registry a record projection resolves its field type against
/// (chelis#1266), and the pipe-stage parameters currently standing for an
/// upstream value (chelis#569).
struct SizeCtx<'a> {
    env: &'a Env,
    adt_reg: &'a AdtRegistry,
    subst: &'a Subst,
    piped: &'a [PipedParam],
}

/// One pipe stage's parameter while the walk is inside that stage's body
/// (chelis#569). `(pipe {} v s1 s2)` means `s2(s1(v))`, and each `s` is the
/// `(fn {} (params {} p) body)` node the Surf parser synthesizes for a
/// call-stage, so the walk classifies `body` with `p` standing for what the
/// stage receives instead of materializing the rewritten application.
struct PipedParam {
    name: String,
    class: SizeClass,
    /// The piped value's type, when the check layer can read it without
    /// inference. Only the pipe's own input expression has one: a stage
    /// RESULT is a computed value whose type this layer does not model, and
    /// `None` there keeps the shape-operand question fail-closed.
    ty: Option<Type>,
}

impl SizeCtx<'_> {
    fn piped(&self, name: &str) -> Option<&PipedParam> {
        // Innermost binding wins: a nested pipe reuses the synthesized
        // parameter name, so the search runs from the end.
        self.piped.iter().rev().find(|param| param.name == name)
    }
}

fn classify_size_in(expr: &deep::Expr, ctx: &SizeCtx<'_>) -> SizeClass {
    stack_guard!("classify_expand_size", expr, SizeClass::Unknown);
    let env = ctx.env;
    // The checker and lowerer share one checked static folder.
    if fold_static_int_expr(expr, |name| env.static_size_value(name)).is_some() {
        return SizeClass::Static;
    }
    // An inline `shape(t, axis)` read (possibly `cast`-wrapped) of an
    // in-scope tensor is the canonical shape source (`bias_broadcast`).
    if let Some(operand) = shape_read_operand(expr) {
        return if shape_operand_is_in_scope_tensor(operand, ctx) {
            SizeClass::ShapeSourced
        } else {
            // `shape(<non-tensor>, ...)` cannot supply an extent.
            SizeClass::Sourceless
        };
    }
    // chelis#1107 round 3: this outer match was `Expr::List`-only with a
    // non-erroring `_ => SizeClass::Unknown` default, and `Unknown` is the
    // ACCEPTING class in `check_expand_signature` (`Sourceless` is the
    // rejecting one). So a stamped `Expr::Node` size argument fell to the
    // fail-OPEN default: `expand(x, 0, k)` with a runtime scalar `k` was
    // accepted by `check_typed_program` and rejected by `check_ir_program`
    // as a §4.7.2 sourceless size -- the exact silent-miscompile class
    // chelis#469 exists to prevent. Reading both carriers here also restores
    // the deliberately fail-CLOSED `_ => Sourceless` default below.
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return SizeClass::Unknown;
    };
    match tag {
        // `cast(<inner>, ty)` — provenance is the inner expr's.
        DeepTag::Cast => kids
            .first()
            .map_or(SizeClass::Unknown, |inner| classify_size_in(inner, ctx)),
        DeepTag::Var => match symbolic_dim_ref_name(expr) {
            // Inside a pipe stage, the stage parameter stands for what the
            // stage receives, so it carries that value's class (chelis#569).
            Some(name) if ctx.piped(name).is_some() => ctx
                .piped(name)
                .map_or(SizeClass::Unknown, |param| param.class),
            // A name carried by an in-scope tensor's shape is a
            // Form-2 symbolic dim with a real source.
            Some(name) if env.tensor_carries_dim_with_subst(name, ctx.subst) => {
                SizeClass::ShapeSourced
            }
            // A recorded `let` provenance (shape-sourced or static).
            Some(name) => match env.size_provenance(name) {
                Some(crate::env::SizeProvenance::ShapeSourced) => SizeClass::ShapeSourced,
                Some(crate::env::SizeProvenance::Static) => SizeClass::Static,
                // A bare value binding (a runtime scalar parameter)
                // with no tensor source and no static provenance.
                None if env.lookup(name).is_some() => SizeClass::Sourceless,
                None => SizeClass::Unknown,
            },
            None => SizeClass::Unknown,
        },
        // Integer arithmetic: combine the operands' classes.
        DeepTag::App => classify_arith_app(kids, ctx),
        // chelis#569's `pipe` arm is gone: `chelis_deep::pipe::fold_pipe`
        // rewrites `v |> s1 |> s2` into `s2(s1(v))` over the checker's input,
        // so the canonical spelling of a shape read reaches this slot as the
        // `app` node above and takes the same route the direct spelling does.
        // The arm that used to sit here read only the pipe's input for a
        // type, so a `shape(...)` read in a LATER stage classified
        // `Sourceless` where the direct spelling classified `ShapeSourced`:
        // an under-approximation the fold makes unnecessary rather than
        // safer.
        // chelis#530: any other List-shaped size — a tuple
        // projection (`t.0`), an inline `match`/`if`, a record
        // `access`, etc. — has NO backend-materializable shape
        // source. It is `Sourceless`, NOT `Unknown`: returning
        // `Unknown` here let the inline `expand(b, 0, t.0)` form
        // (and its `cast`/arithmetic wrappers) reach the
        // non-rejecting `_ => subst.apply(result_ty)` accept arm of
        // `check_expand_signature` and silently miscompile in C to a
        // hardcoded extent-1 axis (eval `[3, 2]` vs C `[1, 2]`),
        // exactly the silent-miscompile class #469 exists to
        // prevent. The `shape(t, ..)`, `cast(..)`, literal,
        // bare-`var`, and arithmetic forms are all recognized BEFORE
        // this arm, so reaching here means the size is genuinely
        // sourceless at the check layer. Mirrors the `classify_arith_app`
        // non-arith-`app` fail-closed default below.
        _ => SizeClass::Sourceless,
    }
}

/// Combine the size classes of an integer-arithmetic application's
/// operands (chelis#397/#469/#1379). `combine_arith_classes` below states the
/// fold and owns the rule; this arm decides only whether the callee is one of
/// the arithmetic builtins.
///
/// A non-arithmetic `app` — any other function call, e.g. `ident(a_dim)` or
/// a user `def` — produces a runtime value with NO shape source the backend
/// can read the extent from, exactly like a bare runtime scalar. It is
/// `Sourceless`, NOT `Unknown`: returning `Unknown` here let the inline
/// `expand(b, 0, ident(a_dim))` form (and its `cast`/arith wrappers) reach
/// the non-rejecting `_` arm of `check_expand_signature` and silently
/// miscompile in C to a hardcoded extent-1 axis (chelis#397 BLOCKER A — the
/// same silent-miscompile class #469 exists to prevent). The `shape(t, ..)`,
/// `cast(..)`, literal, and bare-`var` forms are all recognized BEFORE this
/// arm, so reaching here means the call is genuinely sourceless at the check
/// layer (a user `def` wrapping `shape` is opaque here and would be rejected
/// at lowering too — no `shape_dep`).
/// chelis#1107 round 3: takes the children slice rather than a `&deep::List`,
/// so the classifier works on either carrier (a stamped `Expr::Node` has no
/// `&deep::List` to hand over).
fn classify_arith_app(kids: &[deep::Expr], ctx: &SizeCtx<'_>) -> SizeClass {
    let Some(callee) = kids.first() else {
        return SizeClass::Sourceless;
    };
    let is_int_arith = INT_ARITH.iter().any(|name| is_builtin_var(callee, name));
    if !is_int_arith {
        return SizeClass::Sourceless;
    }
    combine_arith_classes(kids[1..].iter().map(|arg| classify_size_in(arg, ctx)))
}

/// The integer-arithmetic builtins the provenance walk follows.
///
/// The same operator set the shared static folder walks
/// (`fold_static_int_expr`), because the two decide one question about one
/// category: whether an expression is checked integer arithmetic. They
/// disagreed on the two integer division primitives, and `spec/05` section 2.1
/// makes those the ONLY way to divide an int64 extent, `div` being float-only.
/// A static `floor_div(4i64, 2i64)` therefore folded and was admitted while a
/// runtime `floor_div(shape(x, 0), 2i64)` was rejected as sourceless: the same
/// operator on the same category, decided opposite ways by two enumerations
/// (chelis#1379).
const INT_ARITH: &[&str] = &[
    "add",
    "sub",
    "mul",
    "div",
    "floor_div",
    "trunc_div",
    "mod",
    "neg",
];

/// Fold operand classes, shared by the application and bare-pipe-stage
/// spellings of the same arithmetic.
///
/// A `ShapeSourced` operand makes the whole expression `ShapeSourced`, because
/// the extent is computable from that tensor and whatever it is combined with.
/// `Sourceless` decides only an expression with no admissible operand at all,
/// and all-`Static` operands stay `Static`. `Sourceless` used to be absorbing,
/// which rejected `add(shape(x, 0), k)` at check even though both lanes execute
/// it (chelis#1379). The pipe spelling reaches this through the same fold, so
/// the two spellings cannot answer differently.
fn combine_arith_classes(classes: impl Iterator<Item = SizeClass>) -> SizeClass {
    let operand_classes: Vec<SizeClass> = classes.collect();
    if operand_classes.contains(&SizeClass::ShapeSourced) {
        return SizeClass::ShapeSourced;
    }
    if operand_classes.contains(&SizeClass::Sourceless) {
        return SizeClass::Sourceless;
    }
    if operand_classes.contains(&SizeClass::Unknown) {
        return SizeClass::Unknown;
    }
    SizeClass::Static
}

/// Recognize a `shape(operand, axis)` application — possibly wrapped in one
/// or more `cast(..., int32)` layers — and return its `operand` expr
/// (chelis#397/#469). The check-layer analog of the IR layer's
/// `shape_app_operand_axis`. The axis is not validated here (the operand's
/// presence is what proves a tensor source); a runtime axis is fine.
pub(super) fn shape_read_operand(expr: &deep::Expr) -> Option<&deep::Expr> {
    stack_guard!("shape_read_operand", expr, None);
    // chelis#1107: carrier-preserving read; a `List`-only destructure never
    // recognized a stamped `shape(...)` read.
    let (tag, _, kids) = stamped_parts(expr)?;
    // Strip outer `cast(..., ty)` wrappers (tag form and app form).
    if tag == DeepTag::Cast {
        return kids.first().and_then(shape_read_operand);
    }
    let callee = kids.first()?;
    if tag == DeepTag::App && is_builtin_var(callee, "cast") {
        return kids.get(1).and_then(shape_read_operand);
    }
    if tag == DeepTag::App && is_builtin_var(callee, "shape") {
        // `(app {} (var shape) <operand> <axis>)`.
        return kids.get(1);
    }
    None
}

/// True when a `shape(...)` operand expression resolves to an in-scope
/// tensor (chelis#397/#469). The operand is a bare `var` (`shape(x, 0)`) or
/// a borrow of one (`shape(&x, 0)`); either way the named binding must have
/// a tensor type in `env`. A non-tensor operand cannot supply an extent.
fn shape_operand_is_in_scope_tensor(operand: &deep::Expr, ctx: &SizeCtx<'_>) -> bool {
    shape_operand_type(operand, ctx)
        .as_ref()
        .is_some_and(type_is_tensor)
}

/// The type a `shape(...)` operand denotes, when the check layer can read it
/// without inference (chelis#397/#469, chelis#1266).
///
/// The question this answers is "does this operand have a tensor type", not
/// "is this operand a named tensor binding". The two diverged at a record
/// field: `shape(inp.q, 0)` names no tensor binding, yet `inp.q` is an
/// in-scope tensor with exactly the extent the size reads (chelis#1266).
/// Asking the ADT base whether IT is tensor-carrying is the wrong question --
/// `Inputs` is a record, not a tensor -- so the projection resolves the
/// FIELD's declared type instead.
///
/// Recognized spellings: a bare `var`, a `borrow`/`&` wrapper in either the
/// tag or the builtin-application form, and a field projection (`access`) of
/// any of these, nested to any depth. A projection resolves only on a
/// single-record-variant ADT with named fields, which is the only shape
/// `infer_access` itself projects.
fn shape_operand_type(operand: &deep::Expr, ctx: &SizeCtx<'_>) -> Option<Type> {
    stack_guard!("shape_operand_type", operand, None);
    if let Some(name) = symbolic_dim_ref_name(operand) {
        // A pipe stage's parameter denotes the upstream value (chelis#569).
        if let Some(param) = ctx.piped(name) {
            return param.ty.clone();
        }
        return ctx.env.lookup(name).map(|scheme| scheme.body.clone());
    }
    let (tag, _, kids) = stamped_parts(operand)?;
    // `&x` desugars to the `(borrow {} (var x))` TAG form; `borrow(x)`
    // may also appear as the `(app {} (var borrow) (var x))` builtin form.
    if tag == DeepTag::Borrow {
        return kids
            .first()
            .and_then(|inner| shape_operand_type(inner, ctx));
    }
    if tag == DeepTag::Access {
        // `(access {} target field)`: the field's declared type, read off the
        // single record variant of the target's ADT.
        let target = kids.first()?;
        let field = kids.get(1).and_then(symbol_name)?;
        let mut resolved = shape_operand_type(target, ctx)?;
        while let Type::Ref(inner) = resolved {
            resolved = *inner;
        }
        let (Type::Adt(adt_name, _) | Type::KindedAdt(adt_name, _)) = resolved else {
            return None;
        };
        let variant = single_record_variant(ctx.adt_reg, &adt_name)?;
        return variant
            .fields
            .iter()
            .find(|(name, _)| name.as_deref() == Some(field))
            .map(|(_, ty)| ty.clone());
    }
    let callee = kids.first()?;
    if tag == DeepTag::App && is_builtin_var(callee, "borrow") {
        return kids.get(1).and_then(|inner| shape_operand_type(inner, ctx));
    }
    None
}

/// True when a type is (or contains, through `Ref`) a tensor type.
fn type_is_tensor(ty: &Type) -> bool {
    match ty {
        Type::Tensor(..) => true,
        Type::Ref(inner) => type_is_tensor(inner),
        _ => false,
    }
}

/// Describe a non-literal axis argument for the issue #259 diagnostic.
///
/// When the axis is a `(var name)` (the common case: a function-parameter
/// `int32` such as `mean(&x, ax)`), name it so the user can see which
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
/// `declared_dvars` are the dim variables introduced by the declared
/// parameter signatures, snapshotted before body inference. After the
/// body is inferred, each declared dim parameter is universally
/// quantified and must stay distinct: the body must type-check for
/// *all* instantiations of those dims.
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
    declared_dvars: &[DimVar],
    dim_names: &UnordMap<DimVar, String>,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    let render = |dv: DimVar| render_declared_dim(dim_names, dv);
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
                     body: declared dim parameters must remain polymorphic",
                    render(*dv)
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
                        "distinct declared dim parameters {} and {} were unified by the \
                         function body: declared dim parameters are rigid and must remain \
                         distinct",
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

/// chelis#273 return-position rigidity guard.
///
/// `check_declared_dvars_rigid` collects declared dim parameters from
/// the signature's *parameter* positions only, so a dim parameter
/// appearing **only in the return type** was never checked and the body
/// could silently pin it (`def f[k](a: tensor[2, f32]) ->
/// tensor[k, f32] = a` pinned `k := 2`).
///
/// A return-only dim parameter is not fully rigid, though: the body is
/// the only place the output dimension can come from (Chelis has no
/// explicit dim application; callers instantiate dims by unification
/// against *arguments*, which never mention a return-only dim). The
/// legitimate **output-inferred** uses must stay green
/// (spec/04-type-system.md §4.4.1):
///
/// - the body leaves the dim var unbound (a clean, generalizable dim,
///   e.g. variable-fed `to_tensor`), or
/// - the body resolves it to a **body-internal** concrete dim
///   (`examples/hello_tensor.ch`: `def main() -> tensor[n, f32]` whose
///   body builds a `tensor[3, f32]`); the registered scheme then
///   resolves to the produced dim.
///
/// What is rejected is **input coupling** — the body deriving the
/// promised-independent output dim from the caller-visible parameter
/// world:
///
/// - `Dim::Var -> Dim::Lit` pin where the literal equals the
///   post-unification resolution of a dimension occurring in a declared
///   parameter position, or
/// - collapse with a distinct *param-position* declared dim parameter
///   (either binding orientation).
///
/// Two return-only dim parameters collapsing with each other are
/// tolerated (both are output-inferred; no caller-visible coupling).
/// Known residual: coupling through a named symbolic dim
/// (`def f(x: tensor[batch, f32]) -> tensor[m, f32] = x` binds
/// `m := Name("batch")`) is not flagged — `Dim::Name` unifies
/// permissively by design (chelis#219) and no declared dim parameter
/// participates.
pub(super) fn check_return_only_dvars_rigid(
    decl_ty: &Type,
    param_dvars: &[DimVar],
    dim_names: &UnordMap<DimVar, String>,
    subst: &Subst,
    errors: &mut DiagnosticSink<'_>,
) {
    let render = |dv: DimVar| render_declared_dim(dim_names, dv);
    let Type::Fn(decl_params, decl_ret) = decl_ty else {
        return;
    };
    let ret_only: Vec<DimVar> = crate::env::free_dvars(decl_ret)
        .into_iter()
        .filter(|dv| !param_dvars.contains(dv))
        .collect();
    if ret_only.is_empty() {
        return;
    }
    // Every dimension occurring in a declared parameter position,
    // resolved through the post-body substitution.
    let mut param_dims: Vec<Dim> = Vec::new();
    for t in decl_params {
        crate::env::collect_dims(t, &mut param_dims);
    }
    let resolved_param_dims: Vec<Dim> =
        param_dims.iter().map(|d| subst.constraint_dim(d)).collect();
    for dv in &ret_only {
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
        } else if let Some(pdv) = param_dvars
            .iter()
            .find(|pdv| subst.semantic_dim(&Dim::Var(**pdv)) == resolved)
        {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "return-position dim parameter {} was unified with the distinct \
                     declared dim parameter {} from a parameter position: declared \
                     dim parameters are rigid and must remain distinct \
                     (spec/04-type-system.md \u{00a7}4.4.1)",
                    render(*dv),
                    render(*pdv)
                ),
                vec![
                    "Use the same dim parameter in both positions if the return \
                     dimension is meant to equal the input's, or fix the body so the \
                     declared dims stay independent"
                        .to_string(),
                ],
            ));
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
