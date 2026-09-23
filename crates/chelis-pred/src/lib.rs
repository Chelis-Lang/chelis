//! Invariant-predicate grammar and amenability classifier for opaque
//! types (RFC `opaque_invariants_rfc.md` D-PRED / D-WF).
//!
//! This is a leaf crate: it depends only on `chelis-deep` so it can be
//! consumed by `chelis-surf` (desugar-time amenability recording),
//! `chelis-types` (declaration-time well-formedness re-verification),
//! and `chelis-prove` (consumption + `.dp` recomputation) without
//! introducing a dependency cycle.
//!
//! The unit of work is the *predicate fn node*: the Deep encoding the
//! desugarer embeds in `deftype` metadata under the `invariant` key,
//! shaped `(fn {} (params {} <binder>) <body>)`. Every public function
//! here operates on that node.
//!
//! Two surfaces are exported:
//!
//! - [`classify_predicate`] computes a [`PredAmenability`] (recorded as
//!   the `invariant_amenability` metadata string and consumed by the
//!   prove tier dispatcher). It classifies the *structure* of the body;
//!   it never rejects.
//! - [`predicate_in_grammar`] is the pure-grammar admission check
//!   (D-WF). It rejects out-of-grammar nodes. Totality is deliberately
//!   not guaranteed: `/`, `log`, and `sqrt` are partial, which D-WF
//!   pins as the partial-eval semantics for downstream consumers.
//!
//! `From<PredAmenability> for SmtAmenability` deliberately does NOT live
//! here; it lives in `chelis-prove` so this crate stays prove-independent
//! and the existing tide amenability surface stays stable (RFC D-PRED).

use chelis_deep::DeepTag;
use chelis_deep::{Atom, Expr, ExprCarrier};

/// Amenability of a predicate to SMT reasoning. Mirrors the existing
/// `SmtAmenability` vocabulary (`chelis-prove`); the canonical metadata
/// strings are the [`PredAmenability::as_str`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredAmenability {
    /// Comparisons / boolean connectives over affine (linear)
    /// arithmetic only.
    Linear,
    /// Contains a product of two non-constant subterms (degree >= 2)
    /// but no transcendental call.
    Polynomial,
    /// Contains at least one whitelisted transcendental call
    /// (`exp`/`log`/`sin`/`cos`/`sqrt`).
    Transcendental,
    /// Contains an out-of-grammar node; not classified into the SMT
    /// arithmetic fragments.
    Opaque,
}

impl PredAmenability {
    /// The canonical metadata string for this class.
    pub fn as_str(self) -> &'static str {
        match self {
            PredAmenability::Linear => "linear",
            PredAmenability::Polynomial => "polynomial",
            PredAmenability::Transcendental => "transcendental",
            PredAmenability::Opaque => "opaque",
        }
    }

    /// Parse a canonical metadata string back to a class. Returns `None`
    /// for any string outside the closed four-value vocabulary.
    ///
    /// Named `from_str` per the RFC D-PRED public-API contract. It is an
    /// inherent infallible-`Option` method, not the fallible
    /// `std::str::FromStr` trait, so the std-trait shadowing lint is
    /// suppressed deliberately.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<PredAmenability> {
        match s {
            "linear" => Some(PredAmenability::Linear),
            "polynomial" => Some(PredAmenability::Polynomial),
            "transcendental" => Some(PredAmenability::Transcendental),
            "opaque" => Some(PredAmenability::Opaque),
            _ => None,
        }
    }
}

/// The canonical whitelist of intrinsic/transcendental functions
/// admitted in invariant predicates (RFC D-WF, D-PRED). Exported as the
/// single source of truth: `chelis-prove`'s inlineability module
/// consumes this rather than keeping a second copy (RFC L5).
///
/// `abs`/`min`/`max` are total intrinsics; `sqrt`/`exp`/`log`/`sin`/`cos`
/// are the transcendentals that drive a [`PredAmenability::Transcendental`]
/// classification. Of these eight, the transcendental subset is
/// [`TRANSCENDENTAL_WHITELIST`].
pub const INTRINSIC_WHITELIST: &[&str] = &["abs", "min", "max", "sqrt", "exp", "log", "sin", "cos"];

/// The transcendental subset of [`INTRINSIC_WHITELIST`]: a call to any
/// of these forces a [`PredAmenability::Transcendental`] classification.
pub const TRANSCENDENTAL_WHITELIST: &[&str] = &["sqrt", "exp", "log", "sin", "cos"];

/// Arithmetic operators admitted in the predicate grammar (`+ - * /`,
/// plus unary negation). Names are the desugared Deep operator symbols.
const ARITH_OPS: &[&str] = &["add", "sub", "mul", "div", "neg"];

/// Comparison operators admitted in the predicate grammar. Names are the
/// desugared Deep operator symbols (`<` desugars to `cmplt`, `>` to `gt`
/// with authored operand order per chelis#1180, `<=`/`>=` to `lte`/`gte`).
/// `eq`/`neq` are the `==`/`!=` forms.
const COMPARISON_OPS: &[&str] = &["eq", "neq", "cmplt", "gt", "lte", "gte"];

/// Boolean connectives admitted in the predicate grammar.
const BOOL_OPS: &[&str] = &["and", "or", "not"];

/// A predicate node fell outside the declaration grammar (RFC D-WF).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PredGrammarError {
    /// The top-level node was not a `(fn {} (params {} <binder>) <body>)`.
    #[error("invariant predicate is not a well-formed fn node: {0}")]
    NotAPredicateFn(String),
    /// A call to a function not in the grammar (general application).
    #[error("invariant predicate calls a function outside the grammar: {0}")]
    DisallowedCall(String),
    /// A Deep tag with no admitted form (match, lambda, record, tensor
    /// op, effect, etc.).
    #[error("invariant predicate uses a node outside the grammar: {0}")]
    DisallowedNode(String),
    /// A `sum` whose argument is not a binder field projection.
    #[error("invariant predicate `sum` must be applied to a binder field projection")]
    BadSum,
}

// ===========================================================================
// Node helpers (structural; metadata-agnostic)
// ===========================================================================

/// The decoded tag of a list node, if it is a stamped 3-tuple node.
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

/// The children of a 3-tuple node `(tag {meta} children...)`, skipping
/// the tag and the metadata map at index 1.
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

/// If `expr` is `(var {} name)`, return `name`.
fn var_name(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Var, _, [Expr::Atom(Atom::Name(name), _)]) => {
            Some(name.as_str())
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

/// Preserve the legacy free-variable reader's first-child interpretation
/// without letting malformed `var` nodes become binders or callable names.
fn free_var_name(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Var, _, [Expr::Atom(Atom::Name(name), _), ..]) => {
            Some(name.as_str())
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

/// If `expr` is an application `(app {} callee args...)`, return
/// `(callee, args)`.
fn as_app(expr: &Expr) -> Option<(&Expr, &[Expr])> {
    if tag(expr) == Some(DeepTag::App) {
        let kids = children(expr);
        if let Some((callee, args)) = kids.split_first() {
            return Some((callee, args));
        }
    }
    None
}

/// The callee name of an application, when the callee is a bare `var`.
fn app_callee_name(expr: &Expr) -> Option<&str> {
    as_app(expr).and_then(|(callee, _)| var_name(callee))
}

/// Extract the `(params {} <binder>) <body>` of a predicate fn node.
/// Accepts the binder either as a bare `<binder>` symbol (the canonical
/// schema, RFC D-META) or wrapped in a `var`/typed-param list.
fn fn_parts(fn_node: &Expr) -> Option<(String, &Expr)> {
    if tag(fn_node) != Some(DeepTag::Fn) {
        return None;
    }
    let kids = children(fn_node);
    if kids.len() != 2 {
        return None;
    }
    let params = &kids[0];
    let body = &kids[1];
    if tag(params) != Some(DeepTag::Params) {
        return None;
    }
    let param_kids = children(params);
    if param_kids.len() != 1 {
        return None;
    }
    let binder = binder_name(&param_kids[0])?;
    Some((binder, body))
}

/// The name of the single binder in the `params` node. The binder may be
/// a bare symbol, an exact `(var {} name)`, or an exact annotated
/// parameter pair whose first element is the name symbol and whose
/// second element is its metadata map (a structural `Expr::BareList`).
/// An unknown form's head is syntax, not a binder.
fn binder_name(expr: &Expr) -> Option<String> {
    match expr.carrier() {
        ExprCarrier::Atom(Atom::Name(s)) => Some(s.clone()),
        ExprCarrier::DecodedNode(_, _, _) => var_name(expr).map(str::to_string),
        ExprCarrier::StructuralList([Expr::Atom(Atom::Name(name), _), Expr::Map(_, _)]) => {
            Some(name.clone())
        }
        ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

// ===========================================================================
// Free variables
// ===========================================================================

/// The free variables of a predicate fn node: every `var` reference in
/// the body that is not the binder. Field-projection names (the `value`
/// in `p.value`) are NOT variables — they are field selectors. The
/// returned list is deduplicated, in first-seen order.
pub fn predicate_free_vars(fn_node: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    let Some((binder, body)) = fn_parts(fn_node) else {
        return out;
    };
    collect_free_vars(body, &binder, &mut out);
    out
}

fn collect_free_vars(expr: &Expr, binder: &str, out: &mut Vec<String>) {
    // A bare `var` in a value position is a free reference (the binder
    // or an in-module constant). The caller validates scoping.
    if let Some(name) = free_var_name(expr) {
        if name != binder && !out.iter().any(|n| n == name) {
            out.push(name.to_string());
        }
        return;
    }
    // An application `(app {} <callee> args...)`: the callee is an
    // operator / function name (a value-level selector, not a free
    // variable). Recurse into the arguments only. (When the callee is
    // not a bare var, fall through and recurse over everything.)
    if let Some((callee, args)) = as_app(expr)
        && free_var_name(callee).is_some()
    {
        for arg in args {
            collect_free_vars(arg, binder, out);
        }
        return;
    }
    if tag(expr) == Some(DeepTag::Access) {
        // (access {} <target> <field-symbol>): recurse into the target
        // only; the field symbol is a selector, not a variable.
        let kids = children(expr);
        if let Some(target) = kids.first() {
            collect_free_vars(target, binder, out);
        }
        return;
    }
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, _, children) => {
            for child in children {
                collect_free_vars(child, binder, out);
            }
        }
        // An unknown form is outside the predicate grammar, which
        // `predicate_in_grammar` reports; its children are not read.
        ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => {}
    }
}

// ===========================================================================
// Grammar admission (D-WF)
// ===========================================================================

/// Check that a predicate fn node lies within the declaration grammar
/// (RFC D-WF). Returns `Ok(())` for in-grammar predicates, or the first
/// out-of-grammar node found.
///
/// Admitted: literals; the binder and its field projections; arithmetic
/// (`+ - * /` and unary `-`); comparisons (`== != < <= > >=`);
/// `and`/`or`/`not`; `if`; the eight whitelisted intrinsics
/// ([`INTRINSIC_WHITELIST`]); `sum` over a binder field projection; and
/// bare `var` references (in-module zero-arg constant defs — free-var
/// scoping is checked separately by the caller against the module's
/// constant set).
///
/// Note (D-WF): the grammar admits partial functions (`/`, `log`,
/// `sqrt`); totality is not guaranteed here. Downstream consumers apply
/// the D-WF partial-eval semantics (decode failure / sample rejection /
/// real semantics in Tier B).
pub fn predicate_in_grammar(fn_node: &Expr) -> Result<(), PredGrammarError> {
    let (_binder, body) =
        fn_parts(fn_node).ok_or_else(|| PredGrammarError::NotAPredicateFn(node_desc(fn_node)))?;
    check_in_grammar(body)
}

fn check_in_grammar(expr: &Expr) -> Result<(), PredGrammarError> {
    match expr.carrier() {
        // Literals: bare atoms and `(lit {type} value)`.
        ExprCarrier::Atom(_) => Ok(()),
        ExprCarrier::MetadataMap(_) | ExprCarrier::MetadataExpression(_) => {
            Err(PredGrammarError::DisallowedNode(node_desc(expr)))
        }
        ExprCarrier::DecodedNode(_, _, _) => check_decoded_in_grammar(expr),
        ExprCarrier::StructuralList(_) | ExprCarrier::UndecodableHead(_, _, _) => {
            Err(PredGrammarError::DisallowedNode(node_desc(expr)))
        }
    }
}

fn check_decoded_in_grammar(expr: &Expr) -> Result<(), PredGrammarError> {
    let t = tag(expr).ok_or_else(|| PredGrammarError::DisallowedNode(node_desc(expr)))?;
    match t {
        DeepTag::Lit => Ok(()),
        // The binder reference and in-module constant references both
        // surface as bare `var` nodes; scoping is the caller's job.
        DeepTag::Var if var_name(expr).is_some() => Ok(()),
        DeepTag::Var => Err(PredGrammarError::DisallowedNode(node_desc(expr))),
        // Field projection on the binder (or nested records).
        DeepTag::Access => {
            let kids = children(expr);
            if let Some(target) = kids.first() {
                check_in_grammar(target)
            } else {
                Err(PredGrammarError::DisallowedNode(node_desc(expr)))
            }
        }
        DeepTag::If => {
            for child in children(expr) {
                check_in_grammar(child)?;
            }
            Ok(())
        }
        DeepTag::App => check_app_in_grammar(expr),
        _ => Err(PredGrammarError::DisallowedNode(node_desc(expr))),
    }
}

fn check_app_in_grammar(expr: &Expr) -> Result<(), PredGrammarError> {
    let (callee, args) =
        as_app(expr).ok_or_else(|| PredGrammarError::DisallowedNode(node_desc(expr)))?;
    let name =
        var_name(callee).ok_or_else(|| PredGrammarError::DisallowedCall(node_desc(callee)))?;

    if name == "sum" {
        // `sum` over a binder field projection only. The argument must
        // be an `access` (well-formedness of the tensor shape is the
        // checker's job; chelis-pred admits the grammar shape).
        if args.len() == 1 && tag(&args[0]) == Some(DeepTag::Access) {
            return check_in_grammar(&args[0]);
        }
        return Err(PredGrammarError::BadSum);
    }

    let admitted = ARITH_OPS.contains(&name)
        || COMPARISON_OPS.contains(&name)
        || BOOL_OPS.contains(&name)
        || INTRINSIC_WHITELIST.contains(&name);
    if !admitted {
        return Err(PredGrammarError::DisallowedCall(name.to_string()));
    }
    for arg in args {
        check_in_grammar(arg)?;
    }
    Ok(())
}

fn node_desc(expr: &Expr) -> String {
    match expr.carrier() {
        ExprCarrier::Atom(Atom::Name(s)) => format!("symbol `{s}`"),
        ExprCarrier::Atom(_) => "literal".to_string(),
        ExprCarrier::MetadataMap(_) => "map".to_string(),
        ExprCarrier::MetadataExpression(_) => "meta-expr".to_string(),
        ExprCarrier::DecodedNode(tag, _, _) => format!("`{}` node", tag.as_str()),
        ExprCarrier::StructuralList(_) => "bare list".to_string(),
        // chelis#731 / [04-TOT-3]: preserve the undecodable head so the
        // checker diagnostic names the malformed tag instead of collapsing
        // every future form into one generic grammar error.
        ExprCarrier::UndecodableHead(head, _, _) => format!("unknown form `{head}`"),
    }
}

// ===========================================================================
// Amenability classification (D-PRED)
// ===========================================================================

/// Classify a predicate fn node into a [`PredAmenability`] (RFC D-PRED).
///
/// Rules:
/// - any out-of-grammar node anywhere ⇒ [`PredAmenability::Opaque`];
/// - any whitelisted transcendental call (`sqrt`/`exp`/`log`/`sin`/`cos`)
///   ⇒ [`PredAmenability::Transcendental`];
/// - a product (`mul`) of two non-constant subterms ⇒
///   [`PredAmenability::Polynomial`];
/// - otherwise (comparisons / boolean connectives over affine
///   arithmetic) ⇒ [`PredAmenability::Linear`].
///
/// Binder field projections classify as variables (non-constant).
/// `sum` over a literal-shape tensor field is treated as an affine
/// combination of its scalar terms (it does not by itself raise the
/// class above Linear).
pub fn classify_predicate(fn_node: &Expr) -> PredAmenability {
    let Some((binder, body)) = fn_parts(fn_node) else {
        return PredAmenability::Opaque;
    };
    if check_in_grammar(body).is_err() {
        return PredAmenability::Opaque;
    }
    classify_node(body, &binder)
}

fn classify_node(expr: &Expr, binder: &str) -> PredAmenability {
    // Transcendental dominates; then Polynomial; then Linear. Opaque is
    // only reachable from an out-of-grammar node, already screened in
    // `classify_predicate`.
    let mut worst = PredAmenability::Linear;

    if let Some(name) = app_callee_name(expr) {
        if TRANSCENDENTAL_WHITELIST.contains(&name) {
            return PredAmenability::Transcendental;
        }
        if name == "mul" {
            let args = as_app(expr).map(|(_, a)| a).unwrap_or(&[]);
            if args.len() == 2 && !is_constant(&args[0], binder) && !is_constant(&args[1], binder) {
                worst = combine(worst, PredAmenability::Polynomial);
            }
        }
    }

    // Recurse: a transcendental anywhere wins, a polynomial product
    // anywhere lifts to Polynomial.
    for child in subexprs(expr) {
        worst = combine(worst, classify_node(child, binder));
        if worst == PredAmenability::Transcendental {
            return worst;
        }
    }
    worst
}

/// The classification-relevant subexpressions of a node: for an `app`,
/// its arguments (not the callee var); for `if`/`access`, its children.
fn subexprs(expr: &Expr) -> Vec<&Expr> {
    if let Some((_, args)) = as_app(expr) {
        return args.iter().collect();
    }
    match tag(expr) {
        Some(DeepTag::If) | Some(DeepTag::Access) => children(expr).iter().collect(),
        _ => Vec::new(),
    }
}

/// Whether a subterm is a constant for amenability purposes: a literal,
/// or a `var` reference that is NOT the binder and NOT a binder field
/// projection. The binder and `binder.field`/`sum(binder.field)` are
/// non-constant (they range over representation values). An in-module
/// constant `var` reference is a constant.
fn is_constant(expr: &Expr, binder: &str) -> bool {
    match expr {
        Expr::Atom(Atom::Int(_) | Atom::Float(_) | Atom::Bool(_) | Atom::Str(_), _) => true,
        _ => match tag(expr) {
            Some(DeepTag::Lit) => true,
            Some(DeepTag::Var) => var_name(expr) != Some(binder),
            Some(DeepTag::Access) => false,
            Some(DeepTag::App) => {
                let name = app_callee_name(expr);
                if name == Some("sum") {
                    // sum over a binder field is non-constant.
                    return false;
                }
                // An arithmetic combination is constant only if all
                // operands are constant.
                as_app(expr)
                    .map(|(_, args)| args.iter().all(|a| is_constant(a, binder)))
                    .unwrap_or(false)
            }
            _ => false,
        },
    }
}

fn combine(a: PredAmenability, b: PredAmenability) -> PredAmenability {
    fn rank(x: PredAmenability) -> u8 {
        match x {
            PredAmenability::Linear => 0,
            PredAmenability::Polynomial => 1,
            PredAmenability::Transcendental => 2,
            PredAmenability::Opaque => 3,
        }
    }
    if rank(a) >= rank(b) { a } else { b }
}

#[cfg(test)]
mod tests;
