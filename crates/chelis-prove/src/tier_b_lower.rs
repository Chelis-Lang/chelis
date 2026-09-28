//! Deep-expression Tier B lowering for producer obligations (RFC
//! D-TIERB, D-PARITY).
//!
//! This is the chelis-prove home of the obligation Surf->SMT lowering (the
//! lowering relocates here per D-TIERB so the CLI and tide paths share one
//! implementation). It lowers a producer's obligation to an
//! [`SmtProperty`] by:
//!
//! 1. inlining the producer body (depth-bounded, cycle-guarded);
//! 2. `if` => `ite` (already a Tier B primitive elsewhere; carried here);
//! 3. **case-of-known-constructor reduction**: a `match` over an inlined
//!    `if guard then Some(R) else None` reduces per branch — the `Some v`
//!    arm with `v := R`, the `None`/wildcard arm to its body;
//! 4. **record beta-reduction**: an `(access (record C (kv f e) ...) f)`
//!    substitutes the field expression `e`;
//! 5. applying the invariant predicate to the produced representation,
//!    with the binder bound to the produced record (so `p.value` over a
//!    `Probability { value: x }` reduces to `x`).
//!
//! The acceptance bar (RFC D-TIERB): a guarded-Option constructor's
//! derived obligation proves with `proof_tier:"smt"` for linear-arithmetic
//! invariants. A residual irreducible `match` falls to Tier C.

use chelis_deep::annotations::{MetadataValue as M, TypeSyntax};
use chelis_deep::{DeepTag, ExprCarrier};
use chelis_unord::UnordMap;

use chelis_deep::ast::{Atom, Expr};

use crate::obligations::{ObligationProperty, ProducedPosition};
use crate::opaque::OpaqueInvariant;
use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr};
use crate::tier_b::SmtProperty;

const MAX_INLINE_DEPTH: usize = 3;

// ===========================================================================
// Deep helpers
// ===========================================================================

// chelis#1125 PP7 / spec/04-type-system.md §10 [04-TOT-5]: these two are the
// whole module's view of a Deep node. When they read only the deleted list
// spelling, the CLI `chelis prove foo.dp` path (stamped exprs from
// `parse_and_stamp_file`) got nothing back, the obligation could not lower,
// and `run_one` fell through to Tier C, while the same module submitted as
// `.ch` proved at Tier B. A node now has one spelling on every ingress.

fn tag(expr: &Expr) -> Option<DeepTag> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, _) => Some(tag),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn children(expr: &Expr) -> &[Expr] {
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, _, children) => children,
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => &[],
    }
}

fn symbol_text(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn var_name(expr: &Expr) -> Option<&str> {
    (tag(expr) == Some(DeepTag::Var))
        .then(|| symbol_text(children(expr).first()?))
        .flatten()
}

fn app_parts(expr: &Expr) -> Option<(&str, &[Expr])> {
    if tag(expr) == Some(DeepTag::App) {
        let kids = children(expr);
        let callee = kids.first()?;
        let name = var_name(callee)?;
        return Some((name, &kids[1..]));
    }
    None
}

/// The binder of an inline-annotated param `(name {type: T})`, a structural
/// list with the name atom first.
fn inline_param_name(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        ExprCarrier::StructuralList(elements) => elements.first().and_then(symbol_text),
        ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

/// A producer def found in the program: param names and body.
struct ProducerBody<'a> {
    params: Vec<String>,
    body: &'a Expr,
}

fn lookup_producer<'a>(exprs: &'a [Expr], name: &str) -> Option<ProducerBody<'a>> {
    fn find<'a>(exprs: &'a [Expr], name: &str) -> Option<ProducerBody<'a>> {
        for expr in exprs {
            if tag(expr) == Some(DeepTag::Def)
                && let kids = children(expr)
                && kids.first().and_then(symbol_text) == Some(name)
                && let Some(fn_node) = kids.get(1)
                && tag(fn_node) == Some(DeepTag::Fn)
            {
                let fkids = children(fn_node);
                let params_node = fkids.first()?;
                let mut params = Vec::new();
                for p in children(params_node) {
                    // `(name {type: ...})` or bare symbol. chelis#1125 PP7
                    // finding 1: an inline-annotated param is a TAGLESS list,
                    // so it is an `Expr::BareList` with the name atom first.
                    if let Some(n) = symbol_text(p) {
                        params.push(n.to_string());
                    } else if let Some(name) = inline_param_name(p) {
                        params.push(name.to_string());
                    }
                }
                let body = fkids.get(1)?;
                return Some(ProducerBody { params, body });
            }
            // Descend into a `module` wrapper (or any other decoded form);
            // `children` excludes the tag and metadata.
            if let Some(found) = find(children(expr), name) {
                return Some(found);
            }
        }
        None
    }
    find(exprs, name)
}

// ===========================================================================
// Obligation -> SmtProperty
// ===========================================================================

/// The lowered obligation, ready for the Tier B solver.
pub struct LoweredObligation {
    pub property: SmtProperty,
}

/// Lower a producer obligation to an [`SmtProperty`] over the producer's
/// scalar params, or `None` if the obligation does not lower (residual
/// match, non-scalar param, unsupported container, over-cap tensor).
///
/// `exprs` is the desugared Deep program (for producer-body lookup),
/// `consts` resolves in-module predicate constants.
pub fn lower_obligation(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    producer_params: &[(String, ProducerParamType)],
    consts: &crate::opaque::ConstEnv,
) -> Option<LoweredObligation> {
    // Constant producers have no inputs; the obligation is inv(value).
    // Not lowered to Tier B in V1 (the value is a concrete record we can
    // validate concretely); return None -> Tier C.
    if ob.is_constant {
        return None;
    }

    // Tier B cap: the binder representation must flatten to <= 64 scalars.
    if inv.scalar_count() > crate::opaque::TIER_B_SCALAR_CAP {
        return None;
    }

    let prod = lookup_producer(exprs, &ob.producer)?;
    if prod.params.len() != producer_params.len() {
        return None;
    }

    // Solver variables + the input-invariant preconditions (D-SOUND
    // inductive step). A scalar param contributes one solver var; an
    // opaque input param contributes one solver var per representation
    // field (flattened) plus its invariant as a precondition.
    let mut variables = Vec::new();
    let mut preconditions = Vec::new();
    // The set of producer-param names that are opaque inputs, mapped to
    // their (binder-name -> flattened-prefix) so the body lowering
    // resolves `access(var p) field` to `Var("p.field")`.
    let mut opaque_params: UnordMap<String, OpaqueInvariant> = UnordMap::new();
    // Substitution: a scalar param maps to a free solver var; an opaque
    // input param is left as itself (its field projections lower directly
    // to flattened vars, so it must NOT be substituted by a single var).
    let mut subst: UnordMap<String, Expr> = UnordMap::new();

    for (pname, (vname, pty)) in prod.params.iter().zip(producer_params.iter()) {
        match pty {
            ProducerParamType::Scalar(s) => {
                // Single-source int-width -> sort decision (F3): a producer
                // param of ANY integer width is Int, matching the field sort
                // and the constant lowering.
                let sort = crate::opaque::prim_to_smt_sort(s);
                variables.push((vname.clone(), sort));
                subst.insert(pname.clone(), make_var(vname));
            }
            ProducerParamType::Opaque(input_inv) => {
                // Cap: the flattened input representation must fit.
                if input_inv.scalar_count() > crate::opaque::TIER_B_SCALAR_CAP {
                    return None;
                }
                // One solver var per representation field, named
                // `<param>.<field>` to match the predicate flattening.
                for (fname, fty) in &input_inv.fields {
                    let sort = fty.scalar_sort()?; // tensor fields: V1 cap-out
                    variables.push((format!("{pname}.{fname}"), sort));
                }
                // The input invariant as a precondition (the assumption),
                // flattened over `<param>`. Pass the defining program so a
                // module constant in the precondition lowers with its
                // declared numeric type (U2), matching the int-sorted field
                // vars declared just above.
                let pre =
                    crate::opaque::lower_predicate_flattened_in(input_inv, pname, consts, exprs)?;
                preconditions.push(pre);
                opaque_params.insert(pname.clone(), input_inv.clone());
                // No subst entry: the param stays `p`, and its field
                // projections lower to flattened vars in the body.
            }
            ProducerParamType::Other => return None,
        }
    }

    let postcondition =
        lower_obligation_body(prod.body, &subst, &opaque_params, inv, ob, exprs, consts, 0)?;

    Some(LoweredObligation {
        property: SmtProperty {
            variables,
            preconditions,
            postcondition,
        },
    })
}

/// Rewrite every `(access (var p) field)` where `p` is an opaque input
/// parameter into a synthetic `(var "p.field")` node, recursively. The
/// flattened var name matches the input-invariant precondition's
/// flattening, so the body and the assumption share one variable space.
fn rewrite_opaque_field_access(
    expr: &Expr,
    opaque_params: &UnordMap<String, OpaqueInvariant>,
) -> Expr {
    // `(access (var p) field)` -> `(var "p.field")` when p is opaque.
    if tag(expr) == Some(DeepTag::Access) {
        let kids = children(expr);
        if let (Some(target), Some(field_node)) = (kids.first(), kids.get(1))
            && let Some(pname) = var_name(target)
            && opaque_params.contains_key(pname)
            && let Some(field) = symbol_text(field_node)
        {
            return make_var(&format!("{pname}.{field}"));
        }
    }
    // chelis#1125 PP7: a recursion that read only the deleted list spelling
    // cloned a stamped `Expr::Node` whole, so no opaque field access inside
    // it was rewritten and the obligation could not lower.
    match expr {
        Expr::Node(node, span) => {
            let children = node
                .children_slice()
                .iter()
                .map(|e| rewrite_opaque_field_access(e, opaque_params))
                .collect();
            let mut rewritten = node.clone();
            // The rewrite replaces one runtime expression (`access`) with
            // another (`var`) and touches no binder or selector child, so the
            // node's role contract still holds. A failure here would be a
            // compiler bug, which is what `Node`'s own panicking constructor
            // is documented for.
            rewritten
                .try_replace_children(children)
                .expect("opaque field-access rewrite preserves every child role");
            Expr::Node(rewritten, *span)
        }
        Expr::BareList(elements, span) => Expr::BareList(
            elements
                .iter()
                .map(|element| rewrite_opaque_field_access(element, opaque_params))
                .collect(),
            *span,
        ),
        Expr::UnknownForm(_) | Expr::Map(_, _) | Expr::MetaExpr(_, _) | Expr::Atom(_, _) => {
            expr.clone()
        }
    }
}

/// A producer parameter's type, for solver-var sort selection.
#[derive(Debug, Clone)]
pub enum ProducerParamType {
    Scalar(String),
    /// An invariant-carrying opaque input parameter (the update-shaped
    /// inductive step, RFC D-SOUND): its representation is flattened to
    /// per-field solver vars and its invariant is asserted as a
    /// precondition (the input assumption). Carries the input type's
    /// invariant model.
    Opaque(OpaqueInvariant),
    Other,
}

fn make_var(name: &str) -> Expr {
    use chelis_deep::Span;
    Expr::node(
        DeepTag::Var,
        Default::default(),
        vec![Expr::Atom(Atom::Name(name.to_string()), Span::new(0, 0))],
        Span::new(0, 0),
    )
}

/// Lower the obligation body for a producer whose (inlined) result is
/// `result_expr`, applying the produced-position case analysis and the
/// invariant.
#[allow(clippy::too_many_arguments)]
fn lower_obligation_body(
    result_expr: &Expr,
    subst: &UnordMap<String, Expr>,
    opaque_params: &UnordMap<String, OpaqueInvariant>,
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    exprs: &[Expr],
    consts: &crate::opaque::ConstEnv,
    depth: usize,
) -> Option<SmtExpr> {
    match &ob.position {
        ProducedPosition::Direct => {
            // The result is the produced record directly: apply inv to it.
            // After reduction, rewrite any `access(var <opaque_param>)
            // field` (e.g. an inlined `prob_value(p)`) into the flattened
            // input var so it shares the precondition's variable space.
            let reduced = reduce(result_expr, subst, consts, exprs, depth)?;
            let reduced = rewrite_opaque_field_access(&reduced, opaque_params);
            apply_invariant(&reduced, inv, consts, exprs)
        }
        ProducedPosition::InsideOption(inner) => {
            // result is Option[..]; reduce to known constructor.
            let reduced = reduce(result_expr, subst, consts, exprs, depth)?;
            let reduced = rewrite_opaque_field_access(&reduced, opaque_params);
            lower_option_obligation(&reduced, inner, inv, ob, exprs, consts, depth)
        }
        // Tuple obligations lower to a conjunction; not in the flagship
        // path. Return None -> Tier C for V1 Tier B.
        ProducedPosition::TupleComponents(_) => None,
    }
}

/// Lower a `match result with { Some v => inv(v) | _ => true }` over a
/// result that has been reduced. Case-of-known-constructor: if the result
/// is `if g then Some(R) else None`, reduce to `if g then inv(R) else
/// true`; if it is directly `Some(R)` => `inv(R)`; if `None` => `true`.
#[allow(clippy::too_many_arguments)]
fn lower_option_obligation(
    result: &Expr,
    inner: &ProducedPosition,
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    exprs: &[Expr],
    consts: &crate::opaque::ConstEnv,
    depth: usize,
) -> Option<SmtExpr> {
    // if g then <then> else <else>: push the case analysis into branches.
    if tag(result) == Some(DeepTag::If) {
        let kids = children(result);
        let g = lower_field_bool(kids.first()?)?;
        let then_e = lower_option_obligation(kids.get(1)?, inner, inv, ob, exprs, consts, depth)?;
        let else_e = lower_option_obligation(kids.get(2)?, inner, inv, ob, exprs, consts, depth)?;
        return Some(SmtExpr::Ite(
            Box::new(g),
            Box::new(then_e),
            Box::new(else_e),
        ));
    }
    // Some(R): apply the inner obligation to R.
    if let Some((name, args)) = app_parts(result)
        && name == "Some"
        && let Some(r) = args.first()
    {
        return lower_inner_obligation(r, inner, inv, ob, exprs, consts, depth);
    }
    // None (a bare var): vacuously true.
    if var_name(result) == Some("None") {
        return Some(SmtExpr::BoolLit(true));
    }
    // Residual irreducible match shape: fall to Tier C.
    None
}

/// Apply the inner produced-position obligation to the unwrapped value
/// `r` (e.g. the `R` inside `Some(R)`).
#[allow(clippy::too_many_arguments)]
fn lower_inner_obligation(
    r: &Expr,
    inner: &ProducedPosition,
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    exprs: &[Expr],
    consts: &crate::opaque::ConstEnv,
    depth: usize,
) -> Option<SmtExpr> {
    match inner {
        ProducedPosition::Direct => apply_invariant(r, inv, consts, exprs),
        ProducedPosition::InsideOption(deeper) => {
            lower_option_obligation(r, deeper, inv, ob, exprs, consts, depth)
        }
        ProducedPosition::TupleComponents(_) => None,
    }
}

/// Apply the invariant predicate to a produced record value `r` by
/// binding the invariant binder to `r` and reducing the field
/// projections via record beta-reduction, then lowering to SmtExpr.
fn apply_invariant(
    r: &Expr,
    inv: &OpaqueInvariant,
    consts: &crate::opaque::ConstEnv,
    exprs: &[Expr],
) -> Option<SmtExpr> {
    // r must be a record `(record C (kv {} field expr) ...)`.
    let fields = record_fields(r)?;
    // Lower the predicate body, resolving `(access (var binder) field)`
    // to the record's field expression (record beta).
    let body = predicate_body(&inv.predicate)?;
    lower_pred_bool(body, &inv.binder, &fields, consts, exprs, &inv.fields)
}

fn predicate_body(fn_node: &Expr) -> Option<&Expr> {
    children(fn_node).get(1)
}

/// Extract the `(record C (kv {} field expr) ...)` field map.
fn record_fields(expr: &Expr) -> Option<UnordMap<String, Expr>> {
    if tag(expr) != Some(DeepTag::Record) {
        return None;
    }
    let kids = children(expr);
    // kids[0] is the constructor name; the rest are kv nodes.
    let mut map = UnordMap::new();
    for kv in &kids[1..] {
        if tag(kv) == Some(DeepTag::Kv) {
            let kkids = children(kv);
            let field = symbol_text(kkids.first()?)?.to_string();
            let val = kkids.get(1)?.clone();
            map.insert(field, val);
        }
    }
    Some(map)
}

// ===========================================================================
// Reduction (inlining + record-beta + if normalization)
// ===========================================================================

/// Reduce a Deep expression: substitute free vars from `subst`, resolve
/// in-module zero-arg constants from `consts` (CR-8), inline
/// producer/helper calls, and beta-reduce `(access (record ...) field)`.
fn reduce(
    expr: &Expr,
    subst: &UnordMap<String, Expr>,
    consts: &crate::opaque::ConstEnv,
    exprs: &[Expr],
    depth: usize,
) -> Option<Expr> {
    use chelis_deep::Span;
    // var: substitute if bound, else resolve an in-module constant to a
    // literal (CR-8: a bare `(var hi)` for a value-binding constant
    // `hi = 1.0` was previously returned unchanged, reaching the SMT
    // lowering as an undeclared variable).
    if let Some(name) = var_name(expr) {
        if let Some(replacement) = subst.get(name) {
            return Some(replacement.clone());
        }
        if let Some(value) = consts.get(name) {
            // CR2-4: preserve the constant's DECLARED numeric type. The
            // ConstEnv only carries an f64, so an int-typed constant
            // (`def n() -> i32 = 3`, or the value binding `n = 3`) would
            // otherwise be inlined as an f32 literal and silently lower to
            // SmtSort::Real. Read the const def's declared literal type from
            // the module and emit a matching typed literal node.
            return Some(const_lit_node(exprs, name, *value));
        }
        return Some(expr.clone());
    }
    // access over a reduced record => record beta.
    if tag(expr) == Some(DeepTag::Access) {
        let kids = children(expr);
        let target = reduce(kids.first()?, subst, consts, exprs, depth)?;
        let field = symbol_text(kids.get(1)?)?;
        if let Some(fields) = record_fields(&target)
            && let Some(val) = fields.get(field)
        {
            return Some(val.clone());
        }
        // Rebuild the access with the reduced target (keeps dotted-var
        // lowering working for binder projections).
        return Some(rebuild(DeepTag::Access, vec![target, kids.get(1)?.clone()]));
    }
    // app: inline producer/helper, else reduce args.
    if let Some((name, args)) = app_parts(expr) {
        let reduced_args = args
            .iter()
            .map(|a| reduce(a, subst, consts, exprs, depth))
            .collect::<Option<Vec<_>>>()?;
        // Constructors Some/None and record builders are not inlined.
        if name == "Some" || name == "None" {
            return Some(rebuild_app(name, reduced_args));
        }
        if depth < MAX_INLINE_DEPTH
            && let Some(prod) = lookup_producer(exprs, name)
            && prod.params.len() == reduced_args.len()
        {
            let inner_subst: UnordMap<String, Expr> =
                prod.params.iter().cloned().zip(reduced_args).collect();
            return reduce(prod.body, &inner_subst, consts, exprs, depth + 1);
        }
        return Some(rebuild_app(name, reduced_args));
    }
    // record: reduce field exprs.
    if tag(expr) == Some(DeepTag::Record) {
        let kids = children(expr);
        let mut new_children = vec![kids.first()?.clone()];
        for kv in &kids[1..] {
            if tag(kv) == Some(DeepTag::Kv) {
                let kkids = children(kv);
                let field = kkids.first()?.clone();
                let val = reduce(kkids.get(1)?, subst, consts, exprs, depth)?;
                new_children.push(Expr::node(
                    DeepTag::Kv,
                    Default::default(),
                    vec![field, val],
                    Span::new(0, 0),
                ));
            } else {
                new_children.push(kv.clone());
            }
        }
        return Some(rebuild(DeepTag::Record, new_children));
    }
    // if: reduce children (keep structure for case analysis).
    if tag(expr) == Some(DeepTag::If) {
        let kids = children(expr);
        let c = reduce(kids.first()?, subst, consts, exprs, depth)?;
        let t = reduce(kids.get(1)?, subst, consts, exprs, depth)?;
        let e = reduce(kids.get(2)?, subst, consts, exprs, depth)?;
        return Some(rebuild(DeepTag::If, vec![c, t, e]));
    }
    // Other nodes (lits): return as-is.
    Some(expr.clone())
}

/// A typed `(lit {type: (t-prim {} <ty>)} value)` Deep node for an inlined
/// constant, preserving the constant's DECLARED numeric type (CR2-4 / U2).
/// An integer-typed constant (i8/i16/i32/i64) inlines as an integer
/// literal so it lowers to `SmtSort::Int`; otherwise it inlines as an f32
/// literal (CR-8). The declared-type decision is the SINGLE shared resolver
/// in `crate::opaque` -- the same one the invariant/precondition path
/// consults via `crate::opaque::lower_const_ref` -- so the producer-body and
/// invariant paths can never disagree on a constant's sort (the cvc5
/// sort-mismatch abort, U2).
fn const_lit_node(exprs: &[Expr], name: &str, value: f64) -> Expr {
    match crate::opaque::const_declared_int_type(exprs, name) {
        Some(int_ty) => int_lit_node(value as i64, &int_ty),
        None => float_lit_node(value),
    }
}

/// A typed integer literal Deep node `(lit {type: (t-prim {} <ty>)} value)`
/// for an inlined int-typed constant (CR2-4).
fn int_lit_node(value: i64, int_ty: &str) -> Expr {
    typed_lit_node(Atom::Int(value), int_ty)
}

/// `(lit {type: (t-prim {} <prim>)} value)`.
fn typed_lit_node(value: Atom, prim: &str) -> Expr {
    use chelis_deep::Span;
    use chelis_deep::ast::Metadata;
    let span = Span::new(0, 0);
    let type_node = Expr::node(
        DeepTag::TPrim,
        Metadata::default(),
        vec![Expr::Atom(Atom::Name(prim.to_string()), span)],
        span,
    );
    let mut meta = Metadata::default();
    meta.replace(M::Type(
        TypeSyntax::try_new(type_node).expect("primitive type"),
    ));
    Expr::node(DeepTag::Lit, meta, vec![Expr::Atom(value, span)], span)
}

/// A typed f32 literal Deep node `(lit {type: (t-prim {} f32)} value)` for
/// an inlined constant value (CR-8).
fn float_lit_node(value: f64) -> Expr {
    typed_lit_node(Atom::Float(value), "f32")
}

/// Rebuild a reduced `access`, `record` or `if` node around new children.
///
/// The caller passes the tag it just matched. The metadata map is dropped, as
/// it always was, which is inert at all three call sites: nothing downstream
/// reads their metadata. Every child either kept its runtime position or
/// replaced a `var` at one, so no metadata bound to a binding or pipe position
/// moves and the node gate admits the rebuild.
fn rebuild(tag: DeepTag, new_children: Vec<Expr>) -> Expr {
    use chelis_deep::Span;
    Expr::node(tag, Default::default(), new_children, Span::new(0, 0))
}

fn rebuild_app(callee: &str, args: Vec<Expr>) -> Expr {
    use chelis_deep::Span;
    let mut children = vec![make_var(callee)];
    children.extend(args);
    Expr::node(DeepTag::App, Default::default(), children, Span::new(0, 0))
}

// ===========================================================================
// Lowering reduced Deep -> SmtExpr (guard + invariant predicate)
// ===========================================================================

/// Lower a reduced boolean guard expression to SmtExpr (the `if`
/// condition of a producer body).
/// Lower the invariant predicate body with the binder's field
/// projections substituted by the produced record's field exprs (already
/// reduced to free-var arithmetic). `fields` maps field name -> the
/// produced field expression.
#[allow(clippy::too_many_arguments)]
fn lower_pred_bool(
    expr: &Expr,
    binder: &str,
    fields: &UnordMap<String, Expr>,
    consts: &crate::opaque::ConstEnv,
    exprs: &[Expr],
    binder_fields: &[(String, crate::opaque::FieldType)],
) -> Option<SmtExpr> {
    if let Expr::Atom(Atom::Bool(b), _) = expr {
        return Some(SmtExpr::BoolLit(*b));
    }
    if tag(expr) == Some(DeepTag::If) {
        let kids = children(expr);
        let c = lower_pred_bool(kids.first()?, binder, fields, consts, exprs, binder_fields)?;
        let t = lower_pred_bool(kids.get(1)?, binder, fields, consts, exprs, binder_fields)?;
        let e = lower_pred_bool(kids.get(2)?, binder, fields, consts, exprs, binder_fields)?;
        return Some(SmtExpr::Ite(Box::new(c), Box::new(t), Box::new(e)));
    }
    let (name, args) = app_parts(expr)?;
    match name {
        "and" => Some(SmtExpr::Bool(
            BoolOp::And,
            args.iter()
                .map(|a| lower_pred_bool(a, binder, fields, consts, exprs, binder_fields))
                .collect::<Option<Vec<_>>>()?,
        )),
        "or" => Some(SmtExpr::Bool(
            BoolOp::Or,
            args.iter()
                .map(|a| lower_pred_bool(a, binder, fields, consts, exprs, binder_fields))
                .collect::<Option<Vec<_>>>()?,
        )),
        "not" => Some(SmtExpr::Not(Box::new(lower_pred_bool(
            args.first()?,
            binder,
            fields,
            consts,
            exprs,
            binder_fields,
        )?))),
        "eq" | "neq" | "cmplt" | "gt" | "lte" | "gte" => {
            let op = cmp_op(name)?;
            let lhs_arg = args.first()?;
            let rhs_arg = args.get(1)?;
            let mut l = lower_pred_arith(lhs_arg, binder, fields, consts, exprs)?;
            let mut r = lower_pred_arith(rhs_arg, binder, fields, consts, exprs)?;
            // F4: route the producer-body comparison through the SAME sort
            // reconciliation the flattened-predicate path uses, so an int
            // binder field compared against an integral constant lowers
            // consistently (no int-vs-real mismatch). Int-sortedness is read
            // from the original arg (an `(access (var binder) <int-field>)`
            // projection, or an integer literal).
            let l_int = pred_arg_is_int_sorted(lhs_arg, binder, binder_fields);
            let r_int = pred_arg_is_int_sorted(rhs_arg, binder, binder_fields);
            crate::opaque::reconcile_cmp_operands(&mut l, &mut r, l_int, r_int);
            Some(SmtExpr::Cmp(op, Box::new(l), Box::new(r)))
        }
        _ => None,
    }
}

/// Whether a predicate comparison argument is integer-sorted: an integer
/// literal, or an `(access (var binder) field)` projection onto an integer
/// scalar binder field (F4). Used to drive the shared comparison-operand
/// reconciliation on the producer-body path.
fn pred_arg_is_int_sorted(
    arg: &Expr,
    binder: &str,
    binder_fields: &[(String, crate::opaque::FieldType)],
) -> bool {
    // An integer literal (bare or `(lit {} <int>)`).
    if matches!(arg, Expr::Atom(Atom::Int(_), _)) {
        return true;
    }
    if tag(arg) == Some(DeepTag::Lit)
        && matches!(children(arg).first(), Some(Expr::Atom(Atom::Int(_), _)))
    {
        return true;
    }
    // A field projection `(access (var binder) field)` onto an int field.
    if tag(arg) == Some(DeepTag::Access) {
        let kids = children(arg);
        if let (Some(target), Some(field_node)) = (kids.first(), kids.get(1))
            && var_name(target) == Some(binder)
            && let Some(field) = symbol_text(field_node)
        {
            return binder_fields
                .iter()
                .find(|(n, _)| n == field)
                .map(|(_, f)| f.scalar_sort() == Some(crate::solver::SmtSort::Int))
                .unwrap_or(false);
        }
    }
    false
}

fn lower_pred_arith(
    expr: &Expr,
    binder: &str,
    fields: &UnordMap<String, Expr>,
    consts: &crate::opaque::ConstEnv,
    exprs: &[Expr],
) -> Option<SmtExpr> {
    match expr {
        Expr::Atom(Atom::Float(v), _) => return Some(SmtExpr::RealLit(*v)),
        Expr::Atom(Atom::Int(v), _) => return Some(SmtExpr::IntLit(*v)),
        _ => {}
    }
    if tag(expr) == Some(DeepTag::Lit) {
        return match children(expr).first() {
            Some(Expr::Atom(Atom::Float(v), _)) => Some(SmtExpr::RealLit(*v)),
            Some(Expr::Atom(Atom::Int(v), _)) => Some(SmtExpr::IntLit(*v)),
            _ => None,
        };
    }
    // Field projection `(access (var binder) field)`: record beta — look
    // up the produced field expr and lower it (params are free vars).
    if tag(expr) == Some(DeepTag::Access) {
        let kids = children(expr);
        let target = kids.first()?;
        let field = symbol_text(kids.get(1)?)?;
        if var_name(target) == Some(binder)
            && let Some(field_expr) = fields.get(field)
        {
            return lower_field_expr(field_expr);
        }
        return None;
    }
    // A bare var that is the binder is invalid in arith position; an
    // in-module constant resolves via consts and lowers through the SINGLE
    // type-aware constant lowering (U2), so an int-typed constant lowers as
    // IntLit here exactly as it does on the producer-body path -- never a
    // RealLit that would mismatch an Int-sorted field var and abort cvc5.
    if let Some(name) = var_name(expr) {
        if name == binder {
            return None;
        }
        return consts
            .get(name)
            .copied()
            .map(|value| crate::opaque::lower_const_ref(exprs, name, value));
    }
    let (name, args) = app_parts(expr)?;
    match name {
        "add" | "sub" | "mul" | "div" => {
            let op = arith_op(name)?;
            let l = lower_pred_arith(args.first()?, binder, fields, consts, exprs)?;
            let r = lower_pred_arith(args.get(1)?, binder, fields, consts, exprs)?;
            Some(SmtExpr::Arith(op, Box::new(l), Box::new(r)))
        }
        "neg" => Some(SmtExpr::Arith(
            ArithOp::Neg,
            Box::new(lower_pred_arith(
                args.first()?,
                binder,
                fields,
                consts,
                exprs,
            )?),
            Box::new(SmtExpr::RealLit(0.0)),
        )),
        "abs" | "min" | "max" | "sqrt" | "exp" | "log" | "sin" | "cos" => {
            let lowered = args
                .iter()
                .map(|a| lower_pred_arith(a, binder, fields, consts, exprs))
                .collect::<Option<Vec<_>>>()?;
            Some(SmtExpr::Apply(name.to_string(), lowered))
        }
        _ => None,
    }
}

/// Lower a produced field expression (the value bound to a record field)
/// to SmtExpr — it is already in free-var arithmetic form.
fn lower_field_expr(expr: &Expr) -> Option<SmtExpr> {
    match expr {
        Expr::Atom(Atom::Float(v), _) => return Some(SmtExpr::RealLit(*v)),
        Expr::Atom(Atom::Int(v), _) => return Some(SmtExpr::IntLit(*v)),
        _ => {}
    }
    if tag(expr) == Some(DeepTag::Lit) {
        return match children(expr).first() {
            Some(Expr::Atom(Atom::Float(v), _)) => Some(SmtExpr::RealLit(*v)),
            Some(Expr::Atom(Atom::Int(v), _)) => Some(SmtExpr::IntLit(*v)),
            _ => None,
        };
    }
    if let Some(name) = var_name(expr) {
        return Some(SmtExpr::Var(name.to_string()));
    }
    // A field expr may itself be an `if` (e.g. a clamp): if => ite.
    if tag(expr) == Some(DeepTag::If) {
        let kids = children(expr);
        let c = lower_field_bool(kids.first()?)?;
        let t = lower_field_expr(kids.get(1)?)?;
        let e = lower_field_expr(kids.get(2)?)?;
        return Some(SmtExpr::Ite(Box::new(c), Box::new(t), Box::new(e)));
    }
    let (name, args) = app_parts(expr)?;
    match name {
        "add" | "sub" | "mul" | "div" => {
            let op = arith_op(name)?;
            let l = lower_field_expr(args.first()?)?;
            let r = lower_field_expr(args.get(1)?)?;
            Some(SmtExpr::Arith(op, Box::new(l), Box::new(r)))
        }
        "neg" => Some(SmtExpr::Arith(
            ArithOp::Neg,
            Box::new(lower_field_expr(args.first()?)?),
            Box::new(SmtExpr::RealLit(0.0)),
        )),
        "abs" | "min" | "max" | "sqrt" | "exp" | "log" | "sin" | "cos" => {
            let lowered = args
                .iter()
                .map(lower_field_expr)
                .collect::<Option<Vec<_>>>()?;
            Some(SmtExpr::Apply(name.to_string(), lowered))
        }
        _ => None,
    }
}

/// Lower a boolean guard appearing inside a field expression (the `if`
/// condition of a clamp), over free-var arithmetic.
fn lower_field_bool(expr: &Expr) -> Option<SmtExpr> {
    if let Expr::Atom(Atom::Bool(b), _) = expr {
        return Some(SmtExpr::BoolLit(*b));
    }
    let (name, args) = app_parts(expr)?;
    match name {
        "and" => Some(SmtExpr::Bool(
            BoolOp::And,
            args.iter()
                .map(lower_field_bool)
                .collect::<Option<Vec<_>>>()?,
        )),
        "or" => Some(SmtExpr::Bool(
            BoolOp::Or,
            args.iter()
                .map(lower_field_bool)
                .collect::<Option<Vec<_>>>()?,
        )),
        "not" => Some(SmtExpr::Not(Box::new(lower_field_bool(args.first()?)?))),
        "eq" | "neq" | "cmplt" | "gt" | "lte" | "gte" => {
            let op = cmp_op(name)?;
            let l = lower_field_expr(args.first()?)?;
            let r = lower_field_expr(args.get(1)?)?;
            Some(SmtExpr::Cmp(op, Box::new(l), Box::new(r)))
        }
        _ => None,
    }
}

fn cmp_op(name: &str) -> Option<CmpOp> {
    Some(match name {
        "eq" => CmpOp::Eq,
        "neq" => CmpOp::Ne,
        "cmplt" => CmpOp::Lt,
        "gt" => CmpOp::Gt,
        "lte" => CmpOp::Le,
        "gte" => CmpOp::Ge,
        _ => return None,
    })
}

fn arith_op(name: &str) -> Option<ArithOp> {
    Some(match name {
        "add" => ArithOp::Add,
        "sub" => ArithOp::Sub,
        "mul" => ArithOp::Mul,
        "div" => ArithOp::Div,
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
