//! Opaque-type invariant model shared across the CLI prove path and the
//! chelis-tide MCP path (RFC D-PARITY).
//!
//! This module is the single home for: the invariant-carrying opaque-type
//! registry (which types are `@opaque @invariant(...)`, their record
//! representation, and the predicate), the predicate -> [`SmtExpr`]
//! lowering over a *flattened* binder (`p.value`, dotted for nesting),
//! and the value-class checks that gate which representations the prove
//! layer can sample and flatten (RFC D-INJECT, D-STARVE, D-TIERB).
//!
//! The collector walks a desugared Deep program (the form both surfaces
//! share) so a `.ch` source is parsed + desugared and a `.dp` source is
//! parsed directly, then both feed the same collector. Per RFC D-META the
//! `invariant_amenability` recorded on a `.dp` is NOT trusted: it is
//! recomputed from the predicate via `chelis_pred::classify_predicate`.

use chelis_deep::ast::{Atom, Expr};
use chelis_pred::PredAmenability;

use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr, SmtSort};

/// Maximum number of scalar solver variables a single opaque binder may
/// flatten to before Tier B falls back to Tier C (RFC D-TIERB tensor
/// flattening cap).
pub const TIER_B_SCALAR_CAP: usize = 64;

/// A field of an opaque type's single record variant, in the V1 value
/// class (RFC D-WF): a scalar prim, a fixed-shape numeric tensor, or a
/// nested single-variant record of those.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    /// A scalar primitive: `f32`/`f64`/`int32`/`int64`/`bool`.
    Scalar(String),
    /// A fixed-shape numeric tensor: literal dims + precision.
    Tensor { dims: Vec<usize>, precision: String },
    /// A nested single-variant record (recursively in the value class).
    Record(Vec<(String, FieldType)>),
}

impl FieldType {
    /// The number of scalar leaves this field flattens to (RFC D-TIERB
    /// cap accounting).
    pub fn scalar_count(&self) -> usize {
        match self {
            FieldType::Scalar(_) => 1,
            FieldType::Tensor { dims, .. } => dims.iter().product::<usize>().max(1),
            FieldType::Record(fields) => fields.iter().map(|(_, f)| f.scalar_count()).sum(),
        }
    }

    /// The SMT sort of a scalar field; `None` for non-scalar fields.
    pub fn scalar_sort(&self) -> Option<SmtSort> {
        match self {
            FieldType::Scalar(name) => Some(match name.as_str() {
                "int32" | "int64" => SmtSort::Int,
                "bool" => SmtSort::Bool,
                _ => SmtSort::Real,
            }),
            _ => None,
        }
    }
}

/// An invariant-carrying opaque type's full model: the record fields, the
/// predicate fn node (Deep `(fn {} (params {} binder) body)`), and the
/// recomputed amenability.
#[derive(Debug, Clone)]
pub struct OpaqueInvariant {
    /// The type name (e.g. `Probability`). Opacity is keyed by name +
    /// defining module; duplicate-deftype across modules is a check error
    /// (RFC D-CHECK), so the name is a sufficient discriminant within a
    /// check unit.
    pub type_name: String,
    /// The single record variant's constructor name.
    pub ctor_name: String,
    /// The record fields in declaration order.
    pub fields: Vec<(String, FieldType)>,
    /// The predicate fn node `(fn {} (params {} binder) body)`.
    pub predicate: Expr,
    /// The binder name in the predicate (e.g. `p`).
    pub binder: String,
    /// Amenability recomputed from the predicate (never trusted from
    /// recorded `.dp` metadata).
    pub amenability: PredAmenability,
}

impl OpaqueInvariant {
    /// Total scalar leaves across all fields (RFC D-TIERB cap).
    pub fn scalar_count(&self) -> usize {
        self.fields.iter().map(|(_, f)| f.scalar_count()).sum()
    }
}

// ===========================================================================
// Deep node helpers (mirrors chelis-pred's structural helpers)
// ===========================================================================

fn tag(expr: &Expr) -> Option<&str> {
    if let Expr::List(list, _) = expr
        && let Some(Expr::Atom(Atom::Symbol(s), _)) = list.elements.first()
    {
        Some(s.as_str())
    } else {
        None
    }
}

fn children(expr: &Expr) -> &[Expr] {
    if let Expr::List(list, _) = expr
        && list.elements.len() >= 2
    {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn symbol_text(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn meta_value<'a>(expr: &'a Expr, key: &str) -> Option<&'a Expr> {
    if let Expr::List(list, _) = expr
        && let Some(Expr::Map(map, _)) = list.elements.get(1)
    {
        return map
            .entries
            .iter()
            .find_map(|(k, v)| (k == key).then_some(v));
    }
    None
}

// ===========================================================================
// Collection
// ===========================================================================

/// Walk a desugared Deep program and collect every invariant-carrying
/// opaque type. Non-opaque types, and opaque types without an invariant,
/// are skipped (the latter is plain opacity, unaffected by injection;
/// RFC D-INJECT test-lock).
pub fn collect_opaque_invariants(exprs: &[Expr]) -> Vec<OpaqueInvariant> {
    let mut out = Vec::new();
    for expr in exprs {
        collect_in(expr, &mut out);
    }
    out
}

fn collect_in(expr: &Expr, out: &mut Vec<OpaqueInvariant>) {
    if tag(expr) == Some("deftype")
        && let Some(inv) = opaque_invariant_from_deftype(expr)
    {
        out.push(inv);
    }
    // Recurse into module wrappers and any nesting.
    if let Expr::List(list, _) = expr {
        for child in list.elements.iter().skip(2) {
            collect_in(child, out);
        }
    }
}

fn opaque_invariant_from_deftype(deftype: &Expr) -> Option<OpaqueInvariant> {
    // Require opaque: true and an invariant fn node in the metadata.
    let opaque = matches!(
        meta_value(deftype, "opaque"),
        Some(Expr::Atom(Atom::Bool(true), _))
    );
    if !opaque {
        return None;
    }
    let predicate = meta_value(deftype, "invariant")?.clone();
    if tag(&predicate) != Some("fn") {
        return None;
    }
    let binder = predicate_binder(&predicate)?;

    // children: type-name, (params...), variant...
    let kids = children(deftype);
    let type_name = symbol_text(kids.first()?)?.to_string();

    // Find the single record variant.
    let variant = kids.iter().find(|c| tag(c) == Some("variant"))?;
    let var_kids = children(variant);
    let ctor_name = symbol_text(var_kids.first()?)?.to_string();
    let mut fields = Vec::new();
    for field in var_kids.iter().skip(1) {
        if tag(field) != Some("field") {
            // A positional variant has no `field` children; not a record
            // representation. Such a type is outside the V1 value class.
            return None;
        }
        let fk = children(field);
        let fname = symbol_text(fk.first()?)?.to_string();
        let fty = field_type_from_deep(fk.get(1)?)?;
        fields.push((fname, fty));
    }
    if fields.is_empty() {
        return None;
    }

    // RFC D-META: recompute amenability from the predicate; do not trust
    // the recorded `invariant_amenability` string.
    let amenability = chelis_pred::classify_predicate(&predicate);

    Some(OpaqueInvariant {
        type_name,
        ctor_name,
        fields,
        predicate,
        binder,
        amenability,
    })
}

fn predicate_binder(fn_node: &Expr) -> Option<String> {
    let kids = children(fn_node);
    let params = kids.first()?;
    if tag(params) != Some("params") {
        return None;
    }
    let pkids = children(params);
    let first = pkids.first()?;
    // The desugarer emits a bare symbol binder `(params {} p)`.
    if let Some(name) = symbol_text(first) {
        return Some(name.to_string());
    }
    // A typed-param list `(p {type: ...})`: head symbol is the name.
    if let Expr::List(list, _) = first
        && let Some(Expr::Atom(Atom::Symbol(s), _)) = list.elements.first()
    {
        return Some(s.clone());
    }
    None
}

/// Parse a Deep type node into a [`FieldType`] in the V1 value class.
/// Returns `None` for anything outside the class (functions, ADTs, lists,
/// symbolic-dim or non-numeric tensors).
fn field_type_from_deep(ty: &Expr) -> Option<FieldType> {
    match tag(ty)? {
        "t-prim" => {
            let name = symbol_text(children(ty).first()?)?;
            matches!(name, "f32" | "f64" | "int32" | "int64" | "bool")
                .then(|| FieldType::Scalar(name.to_string()))
        }
        "t-tensor" => {
            let kids = children(ty);
            // Last child is the precision t-prim; preceding are dims.
            let precision = {
                let last = kids.last()?;
                (tag(last) == Some("t-prim"))
                    .then(|| symbol_text(children(last).first()?))
                    .flatten()?
            };
            if !matches!(precision, "f32" | "f64") {
                return None;
            }
            let mut dims = Vec::new();
            for dim in &kids[..kids.len().saturating_sub(1)] {
                // Literal dims only (`(d-lit {} N)`); symbolic dims reject.
                if tag(dim) == Some("d-lit")
                    && let Some(Expr::Atom(Atom::Int(n), _)) = children(dim).first()
                    && *n >= 0
                {
                    dims.push(*n as usize);
                } else {
                    return None;
                }
            }
            Some(FieldType::Tensor {
                dims,
                precision: precision.to_string(),
            })
        }
        "t-adt" => {
            // A nested record ADT reference: not resolvable from the type
            // node alone (it names another deftype). V1 nested records are
            // handled by name resolution at a higher level; here we reject
            // so the producer/binder is flagged covered-or-rejected rather
            // than silently mis-sampled.
            None
        }
        _ => None,
    }
}

// ===========================================================================
// Flattened-binder predicate lowering (Tier B + concrete validation)
// ===========================================================================

/// Lower an invariant predicate to an [`SmtExpr`] over a *flattened*
/// binder: a field projection `(access (var binder) field)` becomes
/// `SmtExpr::Var("<prefix>.field")`, dotted for nesting. Returns `None`
/// if any node is outside the lowerable fragment (the caller then falls
/// back to Tier C / concrete evaluation).
///
/// `prefix` is the dotted path of the binder value (e.g. `"p"` at top
/// level, `"p.inner"` for a nested record binder).
pub fn lower_predicate_flattened(inv: &OpaqueInvariant, prefix: &str) -> Option<SmtExpr> {
    let body = predicate_body(&inv.predicate)?;
    lower_bool(body, &inv.binder, prefix)
}

fn predicate_body(fn_node: &Expr) -> Option<&Expr> {
    children(fn_node).get(1)
}

fn app_parts(expr: &Expr) -> Option<(&str, &[Expr])> {
    if tag(expr) == Some("app") {
        let kids = children(expr);
        let callee = kids.first()?;
        let name = symbol_text(children(callee).first()?)?;
        return Some((name, &kids[1..]));
    }
    None
}

fn lower_bool(expr: &Expr, binder: &str, prefix: &str) -> Option<SmtExpr> {
    // Boolean literal.
    if let Expr::Atom(Atom::Bool(b), _) = expr {
        return Some(SmtExpr::BoolLit(*b));
    }
    if tag(expr) == Some("lit")
        && let Some(Expr::Atom(Atom::Bool(b), _)) = children(expr).first()
    {
        return Some(SmtExpr::BoolLit(*b));
    }
    if tag(expr) == Some("if") {
        let kids = children(expr);
        let c = lower_bool(kids.first()?, binder, prefix)?;
        let t = lower_bool(kids.get(1)?, binder, prefix)?;
        let e = lower_bool(kids.get(2)?, binder, prefix)?;
        return Some(SmtExpr::Ite(Box::new(c), Box::new(t), Box::new(e)));
    }
    let (name, args) = app_parts(expr)?;
    match name {
        "and" => Some(SmtExpr::Bool(
            BoolOp::And,
            args.iter()
                .map(|a| lower_bool(a, binder, prefix))
                .collect::<Option<Vec<_>>>()?,
        )),
        "or" => Some(SmtExpr::Bool(
            BoolOp::Or,
            args.iter()
                .map(|a| lower_bool(a, binder, prefix))
                .collect::<Option<Vec<_>>>()?,
        )),
        "not" => Some(SmtExpr::Not(Box::new(lower_bool(
            args.first()?,
            binder,
            prefix,
        )?))),
        "eq" | "neq" | "cmplt" | "lte" | "gte" => {
            let op = match name {
                "eq" => CmpOp::Eq,
                "neq" => CmpOp::Ne,
                "cmplt" => CmpOp::Lt,
                "lte" => CmpOp::Le,
                "gte" => CmpOp::Ge,
                _ => unreachable!(),
            };
            let l = lower_arith(args.first()?, binder, prefix)?;
            let r = lower_arith(args.get(1)?, binder, prefix)?;
            Some(SmtExpr::Cmp(op, Box::new(l), Box::new(r)))
        }
        // `>` / `<` desugar to gte/cmplt with swapped operands already, so
        // only the five comparison symbols appear. Anything else is not a
        // boolean-shaped node we can lower.
        _ => None,
    }
}

fn lower_arith(expr: &Expr, binder: &str, prefix: &str) -> Option<SmtExpr> {
    match expr {
        Expr::Atom(Atom::Float(v), _) => return Some(SmtExpr::RealLit(*v)),
        Expr::Atom(Atom::Int(v), _) => return Some(SmtExpr::IntLit(*v)),
        _ => {}
    }
    if tag(expr) == Some("lit") {
        return match children(expr).first() {
            Some(Expr::Atom(Atom::Float(v), _)) => Some(SmtExpr::RealLit(*v)),
            Some(Expr::Atom(Atom::Int(v), _)) => Some(SmtExpr::IntLit(*v)),
            _ => None,
        };
    }
    // Field projection: `(access (var binder|path) field)` -> dotted var.
    if tag(expr) == Some("access") {
        return access_path(expr, binder, prefix).map(SmtExpr::Var);
    }
    // A bare `(var name)`: an in-module constant reference. We do not
    // resolve constants here (the caller-level lowering handles those for
    // user properties); for invariant predicates referencing a constant,
    // return None so the property falls to Tier C where the concrete
    // evaluator can resolve it via the producer/eval path.
    if tag(expr) == Some("var") {
        return None;
    }
    if tag(expr) == Some("if") {
        let kids = children(expr);
        let c = lower_bool(kids.first()?, binder, prefix)?;
        let t = lower_arith(kids.get(1)?, binder, prefix)?;
        let e = lower_arith(kids.get(2)?, binder, prefix)?;
        return Some(SmtExpr::Ite(Box::new(c), Box::new(t), Box::new(e)));
    }
    let (name, args) = app_parts(expr)?;
    match name {
        "add" | "sub" | "mul" | "div" => {
            let op = match name {
                "add" => ArithOp::Add,
                "sub" => ArithOp::Sub,
                "mul" => ArithOp::Mul,
                "div" => ArithOp::Div,
                _ => unreachable!(),
            };
            let l = lower_arith(args.first()?, binder, prefix)?;
            let r = lower_arith(args.get(1)?, binder, prefix)?;
            Some(SmtExpr::Arith(op, Box::new(l), Box::new(r)))
        }
        "neg" => Some(SmtExpr::Arith(
            ArithOp::Neg,
            Box::new(lower_arith(args.first()?, binder, prefix)?),
            Box::new(SmtExpr::RealLit(0.0)),
        )),
        "abs" | "min" | "max" | "sqrt" | "exp" | "log" | "sin" | "cos" => {
            let lowered = args
                .iter()
                .map(|a| lower_arith(a, binder, prefix))
                .collect::<Option<Vec<_>>>()?;
            Some(SmtExpr::Apply(name.to_string(), lowered))
        }
        // `sum` over a tensor field would expand to a sum of per-element
        // vars; left to a later extension. Return None -> Tier C.
        _ => None,
    }
}

/// Resolve an `(access ... field)` chain rooted at the binder into a
/// dotted solver-variable name `<prefix>.f1.f2...`. Returns `None` if the
/// chain is not rooted at the binder var.
fn access_path(expr: &Expr, binder: &str, prefix: &str) -> Option<String> {
    let kids = children(expr);
    let target = kids.first()?;
    let field = symbol_text(kids.get(1)?)?;
    // Base: the binder var.
    if tag(target) == Some("var") && symbol_text(children(target).first()?) == Some(binder) {
        return Some(format!("{prefix}.{field}"));
    }
    // Nested access: recurse.
    if tag(target) == Some("access") {
        let base = access_path(target, binder, prefix)?;
        return Some(format!("{base}.{field}"));
    }
    None
}

#[cfg(test)]
mod tests;
