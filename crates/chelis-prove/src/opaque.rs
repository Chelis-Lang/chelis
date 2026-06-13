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

/// In-module zero-argument constant values referenced by a predicate
/// (`sum(p.weights) >= 1.0 - eps` references `eps`). Resolved to concrete
/// `f64` values by the caller (RFC D-WF: in-grammar constant defs whose
/// bodies are themselves in-grammar).
pub type ConstEnv = std::collections::HashMap<String, f64>;

/// Context threaded through predicate lowering: the binder name, the
/// dotted path prefix of the binder value, the binder's fields (so `sum`
/// over a tensor field expands to the right number of scalar terms), and
/// the resolved in-module constant environment.
struct LowerCtx<'a> {
    binder: &'a str,
    prefix: &'a str,
    fields: &'a [(String, FieldType)],
    consts: &'a ConstEnv,
}

/// Lower an invariant predicate to an [`SmtExpr`] over a *flattened*
/// binder: a field projection `(access (var binder) field)` becomes
/// `SmtExpr::Var("<prefix>.field")`, dotted for nesting; `sum` over a
/// literal-shape tensor field expands to a sum of its per-element vars
/// (`<prefix>.field.0 + ... + <prefix>.field.N-1`); in-module constants
/// resolve through `consts`. Returns `None` if any node is outside the
/// lowerable fragment (the caller then falls back to Tier C).
///
/// `prefix` is the dotted path of the binder value (e.g. `"p"` at top
/// level, `"p.inner"` for a nested record binder).
pub fn lower_predicate_flattened(
    inv: &OpaqueInvariant,
    prefix: &str,
    consts: &ConstEnv,
) -> Option<SmtExpr> {
    let body = predicate_body(&inv.predicate)?;
    let ctx = LowerCtx {
        binder: &inv.binder,
        prefix,
        fields: &inv.fields,
        consts,
    };
    lower_bool(body, &ctx)
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

fn lower_bool(expr: &Expr, ctx: &LowerCtx) -> Option<SmtExpr> {
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
        let c = lower_bool(kids.first()?, ctx)?;
        let t = lower_bool(kids.get(1)?, ctx)?;
        let e = lower_bool(kids.get(2)?, ctx)?;
        return Some(SmtExpr::Ite(Box::new(c), Box::new(t), Box::new(e)));
    }
    let (name, args) = app_parts(expr)?;
    match name {
        "and" => Some(SmtExpr::Bool(
            BoolOp::And,
            args.iter()
                .map(|a| lower_bool(a, ctx))
                .collect::<Option<Vec<_>>>()?,
        )),
        "or" => Some(SmtExpr::Bool(
            BoolOp::Or,
            args.iter()
                .map(|a| lower_bool(a, ctx))
                .collect::<Option<Vec<_>>>()?,
        )),
        "not" => Some(SmtExpr::Not(Box::new(lower_bool(args.first()?, ctx)?))),
        "eq" | "neq" | "cmplt" | "lte" | "gte" => {
            let op = match name {
                "eq" => CmpOp::Eq,
                "neq" => CmpOp::Ne,
                "cmplt" => CmpOp::Lt,
                "lte" => CmpOp::Le,
                "gte" => CmpOp::Ge,
                _ => unreachable!(),
            };
            let l = lower_arith(args.first()?, ctx)?;
            let r = lower_arith(args.get(1)?, ctx)?;
            Some(SmtExpr::Cmp(op, Box::new(l), Box::new(r)))
        }
        // `>` / `<` desugar to gte/cmplt with swapped operands already, so
        // only the five comparison symbols appear. Anything else is not a
        // boolean-shaped node we can lower.
        _ => None,
    }
}

fn lower_arith(expr: &Expr, ctx: &LowerCtx) -> Option<SmtExpr> {
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
        return access_path(expr, ctx.binder, ctx.prefix).map(SmtExpr::Var);
    }
    // A bare `(var name)`: an in-module constant reference. Resolve it
    // through the constant environment; an unknown constant means the
    // caller could not resolve it in-grammar, so the predicate is not
    // lowerable here and the property falls to Tier C.
    if tag(expr) == Some("var") {
        let name = symbol_text(children(expr).first()?)?;
        return ctx.consts.get(name).copied().map(SmtExpr::RealLit);
    }
    if tag(expr) == Some("if") {
        let kids = children(expr);
        let c = lower_bool(kids.first()?, ctx)?;
        let t = lower_arith(kids.get(1)?, ctx)?;
        let e = lower_arith(kids.get(2)?, ctx)?;
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
            let l = lower_arith(args.first()?, ctx)?;
            let r = lower_arith(args.get(1)?, ctx)?;
            Some(SmtExpr::Arith(op, Box::new(l), Box::new(r)))
        }
        "neg" => Some(SmtExpr::Arith(
            ArithOp::Neg,
            Box::new(lower_arith(args.first()?, ctx)?),
            Box::new(SmtExpr::RealLit(0.0)),
        )),
        "abs" | "min" | "max" | "sqrt" | "exp" | "log" | "sin" | "cos" => {
            let lowered = args
                .iter()
                .map(|a| lower_arith(a, ctx))
                .collect::<Option<Vec<_>>>()?;
            Some(SmtExpr::Apply(name.to_string(), lowered))
        }
        // `sum` over a literal-shape tensor binder field: expand to the
        // sum of its per-element flattened vars (RFC D-WF / D-TIERB).
        "sum" => {
            let path = access_path(args.first()?, ctx.binder, ctx.prefix)?;
            let count = sum_field_count(&path, ctx)?;
            if count == 0 {
                return Some(SmtExpr::RealLit(0.0));
            }
            let mut acc = SmtExpr::Var(format!("{path}.0"));
            for i in 1..count {
                acc = SmtExpr::Arith(
                    ArithOp::Add,
                    Box::new(acc),
                    Box::new(SmtExpr::Var(format!("{path}.{i}"))),
                );
            }
            Some(acc)
        }
        _ => None,
    }
}

/// The number of scalar elements a `sum`-target field flattens to. The
/// dotted path is `<prefix>.<field>` at the top level; resolve the field
/// against the binder fields and require it to be a tensor.
fn sum_field_count(path: &str, ctx: &LowerCtx) -> Option<usize> {
    let field = path.strip_prefix(ctx.prefix)?.strip_prefix('.')?;
    // Only top-level (non-nested) tensor fields are supported for `sum`.
    let fty = ctx
        .fields
        .iter()
        .find_map(|(n, f)| (n == field).then_some(f))?;
    match fty {
        FieldType::Tensor { dims, .. } => Some(dims.iter().product::<usize>().max(1)),
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

// ===========================================================================
// Tiered validated binder generation (RFC D-STARVE)
// ===========================================================================

use std::collections::BTreeMap;

/// How a sample was proposed. Generation methods are proposal
/// distributions only; every accepted sample is predicate-validated
/// regardless of method (RFC D-STARVE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenMethod {
    /// Independent-component rejection sampling.
    Rejection,
    /// Constructor-based: an exported producer evaluated on sampled raw
    /// inputs (a smarter proposal for measure-near-zero satisfying sets).
    Constructor,
}

/// A validated opaque-binder sample. `env` is the flattened field
/// assignment keyed by dotted path (`p.value`, `p.weights.0`, ...) — the
/// exact key shape [`lower_predicate_flattened`] reads — and `value_expr`
/// is the Deep `(record Ctor (kv field <lit>) ...)` the evaluator
/// materializes.
#[derive(Debug, Clone)]
pub struct GeneratedBinder {
    pub env: BTreeMap<String, f64>,
    pub value_expr: Expr,
    pub method: GenMethod,
}

/// The syntactic shape of an invariant predicate, recorded to explain a
/// starvation and to skip straight to constructor generation when obvious
/// (RFC D-STARVE).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredShape {
    /// Contains an exact `==` equality atom over a field (starves both
    /// tiers by design; only Tier B over reals verifies it).
    EqualityAtoms,
    /// A two-sided tolerance band; `width` is the band width when it can
    /// be read syntactically, else `None`.
    BandWidth(Option<f64>),
    /// Neither an equality atom nor a recognizable band.
    Other,
}

impl PredShape {
    pub fn as_str(self) -> &'static str {
        match self {
            PredShape::EqualityAtoms => "equality-atoms",
            PredShape::BandWidth(_) => "band-width",
            PredShape::Other => "other",
        }
    }
}

/// A generator-starvation diagnostic: both tiers failed to reach the
/// acceptance-rate floor (RFC D-STARVE). Carries the per-method
/// accepted/attempted counts, the floor, the predicate shape, and the
/// recommended verification route.
#[derive(Debug, Clone)]
pub struct StarvationDiagnostic {
    pub type_name: String,
    pub rejection_accepted: usize,
    pub rejection_attempted: usize,
    pub constructor_accepted: usize,
    pub constructor_attempted: usize,
    pub constructor_distinct: usize,
    pub floor: f64,
    pub shape: PredShape,
    pub recommended_route: String,
}

impl StarvationDiagnostic {
    /// The best acceptance rate achieved across both tiers.
    pub fn best_rate(&self) -> f64 {
        let rej = rate(self.rejection_accepted, self.rejection_attempted);
        let ctor = rate(self.constructor_accepted, self.constructor_attempted);
        rej.max(ctor)
    }

    /// A one-line human-facing diagnostic string.
    pub fn message(&self) -> String {
        format!(
            "generator starvation for opaque type `{}`: rejection sampling \
             accepted {}/{} ({:.4}), constructor-based generation accepted \
             {}/{} ({:.4}, {} distinct), both below the floor {:.4}; \
             predicate shape `{}`; recommended route: {}",
            self.type_name,
            self.rejection_accepted,
            self.rejection_attempted,
            rate(self.rejection_accepted, self.rejection_attempted),
            self.constructor_accepted,
            self.constructor_attempted,
            rate(self.constructor_accepted, self.constructor_attempted),
            self.constructor_distinct,
            self.floor,
            self.shape.as_str(),
            self.recommended_route,
        )
    }
}

fn rate(accepted: usize, attempted: usize) -> f64 {
    if attempted == 0 {
        0.0
    } else {
        accepted as f64 / attempted as f64
    }
}

/// A minimal deterministic LCG (matches the prove-path generators).
pub struct GenRng {
    state: u64,
}

impl GenRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    /// Advance the generator and return the next 64-bit state. Public so
    /// the obligation engine can seed a per-binder generator
    /// deterministically from its own RNG stream.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f64(&mut self, min: f64, max: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        min + (max - min) * unit
    }
    fn next_i64(&mut self, min: i64, max: i64) -> i64 {
        let span = (max - min + 1) as u64;
        min + (self.next_u64() % span) as i64
    }
    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
}

/// An exported producer usable for constructor-based generation: its
/// scalar/tensor input parameter kinds and whether it returns the opaque
/// type directly or wrapped in `Option`.
#[derive(Debug, Clone)]
pub struct GenProducer {
    pub name: String,
    pub param_names: Vec<String>,
    pub param_kinds: Vec<GenParamKind>,
    /// `true` when the producer returns `Option[T]` (failures are
    /// Option-unwrapped during generation).
    pub option_wrapped: bool,
}

/// The kind of a producer input parameter for raw-input sampling.
#[derive(Debug, Clone, PartialEq)]
pub enum GenParamKind {
    Scalar(String),
    Tensor { dims: Vec<usize>, precision: String },
}

/// Generate one validated binder value for `inv`. Tiered (RFC D-STARVE):
/// rejection sampling first, constructor-based generation on starvation,
/// every accepted sample predicate-validated. Returns the validated
/// sample, or a [`StarvationDiagnostic`] when both tiers starve below
/// `floor`.
///
/// `floor` is the acceptance-rate floor (`--invariant-min-rate`); a floor
/// of `0.0` disables the starvation classifier — the caller then treats a
/// failure-to-generate as legacy exhaustion (`Error`), not `Unsupported`.
/// `budget` is the per-tier attempt budget.
///
/// `module_source` is the canonical Deep of the defining module (for the
/// constructor tier's evaluator calls); `producers` are the exported
/// producers usable for constructor-based generation.
pub fn generate_binder(
    inv: &OpaqueInvariant,
    consts: &ConstEnv,
    module_source: &str,
    producers: &[GenProducer],
    rng: &mut GenRng,
    floor: f64,
    budget: usize,
) -> Result<GeneratedBinder, StarvationDiagnostic> {
    // Lower the predicate once over the binder name as prefix.
    let predicate = lower_predicate_flattened(inv, &inv.binder, consts);

    // Tier 1: rejection sampling. The accepted count is definitionally 0
    // at the starvation point (we return early on the first accept), so
    // the diagnostic records 0 accepted for a tier that produced no
    // sample; only the attempt counts vary.
    let mut rej_attempts = 0usize;
    let rej_accepts = 0usize;
    for _ in 0..budget {
        rej_attempts += 1;
        let env = sample_fields_flat(&inv.fields, &inv.binder, rng);
        if validate_env(&env, inv, &predicate, consts) {
            // Accepted: rejection sampling succeeded (the per-tier accept
            // counts in the diagnostic are only read when BOTH tiers
            // starve, i.e. when we never reach here).
            let value_expr = record_value_expr(inv, &env);
            return Ok(GeneratedBinder {
                env,
                value_expr,
                method: GenMethod::Rejection,
            });
        }
        // Floor short-circuit. `generate_binder` returns ONE sample per
        // call (we return on the first accept above), so the rejection
        // tier is "starving" only when it fails to find a single valid
        // sample within a margin of the floor's expected attempt count.
        // For a floor `r` the expected attempts to one success is `1/r`;
        // we wait `3/r` attempts (a comfortable margin against an unlucky
        // run on a satisfiable band) before declaring starvation, capped
        // at the budget. A `[0,1]` band (~10% acceptance) finds a sample
        // in ~10 attempts and never short-circuits; a measure-near-zero
        // band hits the threshold and falls to the constructor tier.
        if floor > 0.0 {
            let starvation_threshold = ((3.0 / floor).ceil() as usize).min(budget);
            if rej_attempts >= starvation_threshold {
                break;
            }
        }
    }

    // Tier 2: constructor-based generation. Each proposal is an evaluator
    // call (expensive), and a working producer lands a valid sample within
    // a few attempts, so the constructor tier is capped well below the
    // rejection budget — beyond the cap a starving constructor tier is
    // declared starved rather than spending thousands of evaluator calls.
    let ctor_budget = budget.min(200);
    let mut ctor_attempts = 0usize;
    let ctor_accepts = 0usize;
    if !producers.is_empty() {
        for _ in 0..ctor_budget {
            ctor_attempts += 1;
            let Some(env) = propose_via_constructor(inv, module_source, producers, rng) else {
                continue;
            };
            // STILL validate (a buggy producer costs efficiency, never
            // soundness; RFC D-STARVE M2).
            if validate_env(&env, inv, &predicate, consts) {
                let value_expr = record_value_expr(inv, &env);
                return Ok(GeneratedBinder {
                    env,
                    value_expr,
                    method: GenMethod::Constructor,
                });
            }
        }
    }

    // Both starved.
    let shape = classify_pred_shape(inv, consts);
    let recommended_route = match shape {
        PredShape::EqualityAtoms => {
            "Tier B (real semantics): exact float `==` starves both fuzz tiers by design; \
             use a tolerance band over a module constant"
                .to_string()
        }
        _ => "a richer exported producer set, or Tier B where the property lowers".to_string(),
    };
    Err(StarvationDiagnostic {
        type_name: inv.type_name.clone(),
        rejection_accepted: rej_accepts,
        rejection_attempted: rej_attempts,
        constructor_accepted: ctor_accepts,
        constructor_attempted: ctor_attempts,
        constructor_distinct: 0,
        floor,
        shape,
        recommended_route,
    })
}

/// Validate a flattened field env against the predicate. When the
/// predicate lowers, use the fast concrete evaluator over [`SmtExpr`];
/// the env keys are exactly the lowered var names so the two agree by
/// construction.
fn validate_env(
    env: &BTreeMap<String, f64>,
    _inv: &OpaqueInvariant,
    predicate: &Option<SmtExpr>,
    _consts: &ConstEnv,
) -> bool {
    match predicate {
        Some(smt) => {
            let hash: std::collections::HashMap<String, f64> =
                env.iter().map(|(k, v)| (k.clone(), *v)).collect();
            crate::concrete_eval::eval_bool(smt, &hash)
        }
        // A predicate that does not lower (e.g. references an unresolved
        // constant) cannot be validated here; treat as not-satisfied so
        // generation starves rather than silently accepting.
        None => false,
    }
}

/// Sample each field independently into a flattened dotted-path env.
fn sample_fields_flat(
    fields: &[(String, FieldType)],
    prefix: &str,
    rng: &mut GenRng,
) -> BTreeMap<String, f64> {
    let mut env = BTreeMap::new();
    for (name, fty) in fields {
        let path = format!("{prefix}.{name}");
        sample_field_into(&path, fty, rng, &mut env);
    }
    env
}

fn sample_field_into(
    path: &str,
    fty: &FieldType,
    rng: &mut GenRng,
    env: &mut BTreeMap<String, f64>,
) {
    match fty {
        FieldType::Scalar(name) => {
            let v = match name.as_str() {
                "int32" | "int64" => rng.next_i64(-1000, 1000) as f64,
                "bool" => {
                    if rng.next_bool() {
                        1.0
                    } else {
                        0.0
                    }
                }
                _ => rng.next_f64(-10.0, 10.0),
            };
            env.insert(path.to_string(), v);
        }
        FieldType::Tensor { dims, .. } => {
            let count = dims.iter().product::<usize>().max(1);
            for i in 0..count {
                env.insert(format!("{path}.{i}"), rng.next_f64(-10.0, 10.0));
            }
        }
        FieldType::Record(inner) => {
            for (n, f) in inner {
                sample_field_into(&format!("{path}.{n}"), f, rng, env);
            }
        }
    }
}

/// Build the Deep record value `(record Ctor (kv {} field <lit-or-tensor>)
/// ...)` from a flattened env.
fn record_value_expr(inv: &OpaqueInvariant, env: &BTreeMap<String, f64>) -> Expr {
    let mut children = vec![Expr::Atom(
        Atom::Symbol(inv.ctor_name.clone()),
        chelis_deep::Span::new(0, 0),
    )];
    for (name, fty) in &inv.fields {
        let path = format!("{}.{}", inv.binder, name);
        let value = field_value_expr(&path, fty, env);
        children.push(kv_node(name, value));
    }
    record_node(children)
}

fn field_value_expr(path: &str, fty: &FieldType, env: &BTreeMap<String, f64>) -> Expr {
    match fty {
        FieldType::Scalar(name) => match name.as_str() {
            "int32" | "int64" => int_lit(*env.get(path).unwrap_or(&0.0) as i64, name),
            "bool" => bool_lit(*env.get(path).unwrap_or(&0.0) != 0.0),
            _ => float_lit(*env.get(path).unwrap_or(&0.0), name),
        },
        FieldType::Tensor { dims, precision } => {
            let count = dims.iter().product::<usize>().max(1);
            let values: Vec<f64> = (0..count)
                .map(|i| *env.get(&format!("{path}.{i}")).unwrap_or(&0.0))
                .collect();
            tensor_value_expr(dims, precision, &values)
        }
        FieldType::Record(inner) => {
            // A nested record value: `(record InnerCtor ...)`. The inner
            // ctor name is not tracked here in V1 nested support; build a
            // bare record with field kvs (the evaluator resolves the
            // single-variant ctor by field set).
            let mut children = vec![Expr::Atom(
                Atom::Symbol("__nested".to_string()),
                chelis_deep::Span::new(0, 0),
            )];
            for (n, f) in inner {
                let v = field_value_expr(&format!("{path}.{n}"), f, env);
                children.push(kv_node(n, v));
            }
            record_node(children)
        }
    }
}

/// Constructor-based proposal: pick a producer, sample its raw inputs,
/// evaluate it, Option-unwrap failures, and read back the produced
/// record's field values into a flattened env. Returns `None` on producer
/// failure (None result) or an unreadable output.
fn propose_via_constructor(
    inv: &OpaqueInvariant,
    module_source: &str,
    producers: &[GenProducer],
    rng: &mut GenRng,
) -> Option<BTreeMap<String, f64>> {
    let idx = (rng.next_u64() as usize) % producers.len();
    let producer = &producers[idx];

    // Build the call `producer(<sampled raw inputs>)` and read each
    // representation field of the result via a probe that projects the
    // field (so we recover the flattened env exactly).
    let arg_exprs: Vec<Expr> = producer
        .param_kinds
        .iter()
        .map(|k| sample_raw_input_expr(k, rng))
        .collect();

    // Evaluate each representation field of the produced value through the
    // module, unwrapping Option if needed.
    let mut env = BTreeMap::new();
    for (fname, fty) in &inv.fields {
        let field_path = format!("{}.{}", inv.binder, fname);
        if !read_produced_field(
            module_source,
            producer,
            &arg_exprs,
            &inv.type_name,
            fname,
            fty,
            &field_path,
            &mut env,
        )? {
            return None;
        }
    }
    Some(env)
}

/// Read one representation field of a producer's result into `env`.
/// Returns `Some(true)` on success, `Some(false)` when the producer
/// returned `None` (failure to unwrap), `None` on evaluation error.
#[allow(clippy::too_many_arguments)]
fn read_produced_field(
    module_source: &str,
    producer: &GenProducer,
    arg_exprs: &[Expr],
    type_name: &str,
    field: &str,
    fty: &FieldType,
    field_path: &str,
    env: &mut BTreeMap<String, f64>,
) -> Option<bool> {
    // The probe binds `r = producer(args)` (Option-unwrapped to a fresh
    // var via match when wrapped), accesses `r.<field>`, and we read the
    // resulting tensor/scalar values. We synthesize a probe def INSIDE the
    // defining module so field access on the opaque type is legal.
    let probe = "__chelis_gen_probe";
    let call = {
        let mut app = vec![var_node(&producer.name)];
        app.extend(arg_exprs.iter().cloned());
        app_node(app)
    };
    // For Option-wrapped producers, project the field only on Some;
    // a None result yields a sentinel we detect as failure.
    let access = access_node(var_node("__r"), field);
    let body = if producer.option_wrapped {
        // match producer(args) { Some(__r) => __r.field | _ => <sentinel> }
        match_some_else(call, "__r", access, sentinel_for(fty))
    } else {
        // { __r = producer(args); __r.field }
        let_block("__r", call, access)
    };
    let probe_def = node_def(probe, body);
    let exprs = chelis_deep::parser::parse_str(module_source).ok()?;
    let program = inject_into_module_with_source(&exprs, type_name, probe_def);
    let source = chelis_deep::printer::print_canonical(&program);
    let result = chelis_compiler_api::compiler::eval_selected(
        chelis_compiler_api::schema::EvalRequest {
            source_kind: chelis_compiler_api::schema::SourceKind::Deep,
            source,
            bindings: Default::default(),
        },
        &[probe.to_string()],
    )
    .ok()?;
    let root = match result.roots.as_slice() {
        [r] => r,
        _ => return None,
    };
    use chelis_compiler_api::schema::ExecutionValue;
    match fty {
        FieldType::Tensor { dims, .. } => {
            // A tensor field access yields a Tensor value.
            let ExecutionValue::Tensor { value } = &root.value else {
                return None;
            };
            // A None result yields the NaN-filled sentinel: treat as failure.
            if value.data.iter().any(|v| v.is_nan()) {
                return Some(false);
            }
            let count = dims.iter().product::<usize>().max(1);
            if value.data.len() != count {
                return None;
            }
            for (i, v) in value.data.iter().enumerate() {
                env.insert(format!("{field_path}.{i}"), *v);
            }
        }
        _ => {
            // A scalar field access yields a scalar ExecutionValue (RT3-F3:
            // a rank-0 access returns Float64 / Int64 / Bool, not a
            // single-element Tensor). Extract the scalar; a NaN result is
            // the None-sentinel and counts as a producer failure.
            let v = match &root.value {
                ExecutionValue::Float64 { value } => *value,
                ExecutionValue::Int64 { value } => *value as f64,
                ExecutionValue::Bool { value } => {
                    if *value {
                        1.0
                    } else {
                        0.0
                    }
                }
                // A rank-0/single-element tensor scalar, defensively.
                ExecutionValue::Tensor { value }
                    if value.shape.iter().product::<usize>().max(1) == 1
                        && !value.data.is_empty() =>
                {
                    value.data[0]
                }
                _ => return None,
            };
            if v.is_nan() {
                return Some(false);
            }
            env.insert(field_path.to_string(), v);
        }
    }
    Some(true)
}

fn sample_raw_input_expr(kind: &GenParamKind, rng: &mut GenRng) -> Expr {
    match kind {
        GenParamKind::Scalar(name) => match name.as_str() {
            "int32" | "int64" => int_lit(rng.next_i64(-1000, 1000), name),
            "bool" => bool_lit(rng.next_bool()),
            _ => float_lit(rng.next_f64(-10.0, 10.0), name),
        },
        GenParamKind::Tensor { dims, precision } => {
            let count = dims.iter().product::<usize>().max(1);
            let values: Vec<f64> = (0..count).map(|_| rng.next_f64(-10.0, 10.0)).collect();
            tensor_value_expr(dims, precision, &values)
        }
    }
}

/// Classify the predicate's syntactic shape for the starvation diagnostic.
fn classify_pred_shape(inv: &OpaqueInvariant, consts: &ConstEnv) -> PredShape {
    let Some(body) = predicate_body(&inv.predicate) else {
        return PredShape::Other;
    };
    if contains_equality_atom(body) {
        return PredShape::EqualityAtoms;
    }
    // A two-sided `>=`/`<=` band: read the constants if both bounds are
    // literal/constant.
    if let Some(width) = band_width(body, &inv.binder, consts) {
        return PredShape::BandWidth(Some(width));
    }
    if is_two_sided_band(body) {
        return PredShape::BandWidth(None);
    }
    PredShape::Other
}

fn contains_equality_atom(expr: &Expr) -> bool {
    if let Some((name, _)) = app_parts(expr)
        && (name == "eq" || name == "neq")
    {
        return true;
    }
    if let Expr::List(list, _) = expr {
        return list.elements.iter().skip(2).any(contains_equality_atom);
    }
    false
}

fn is_two_sided_band(expr: &Expr) -> bool {
    // `and(gte(.., lo), lte(.., hi))` or the reverse.
    if let Some(("and", args)) = app_parts(expr)
        && args.len() == 2
    {
        let has_ge = args
            .iter()
            .any(|a| matches!(app_parts(a), Some(("gte", _))));
        let has_le = args
            .iter()
            .any(|a| matches!(app_parts(a), Some(("lte", _))));
        return has_ge && has_le;
    }
    false
}

/// Best-effort band-width read: `sum(..) >= lo && sum(..) <= hi` => hi-lo.
fn band_width(expr: &Expr, binder: &str, consts: &ConstEnv) -> Option<f64> {
    let ("and", args) = app_parts(expr)? else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    let mut lo = None;
    let mut hi = None;
    for a in args {
        if let Some(("gte", cmp)) = app_parts(a)
            && cmp.len() == 2
        {
            lo = eval_const_arith(&cmp[1], binder, consts);
        }
        if let Some(("lte", cmp)) = app_parts(a)
            && cmp.len() == 2
        {
            hi = eval_const_arith(&cmp[1], binder, consts);
        }
    }
    match (lo, hi) {
        (Some(l), Some(h)) => Some(h - l),
        _ => None,
    }
}

/// Evaluate a constant arithmetic subterm (literals, constants, +/-) for
/// the band-width read. Returns `None` if it references the binder.
fn eval_const_arith(expr: &Expr, binder: &str, consts: &ConstEnv) -> Option<f64> {
    match expr {
        Expr::Atom(Atom::Float(v), _) => Some(*v),
        Expr::Atom(Atom::Int(v), _) => Some(*v as f64),
        _ => {
            if tag(expr) == Some("lit") {
                return match children(expr).first() {
                    Some(Expr::Atom(Atom::Float(v), _)) => Some(*v),
                    Some(Expr::Atom(Atom::Int(v), _)) => Some(*v as f64),
                    _ => None,
                };
            }
            if tag(expr) == Some("var") {
                let name = symbol_text(children(expr).first()?)?;
                if name == binder {
                    return None;
                }
                return consts.get(name).copied();
            }
            if let Some((op, args)) = app_parts(expr)
                && args.len() == 2
            {
                let l = eval_const_arith(&args[0], binder, consts)?;
                let r = eval_const_arith(&args[1], binder, consts)?;
                return match op {
                    "add" => Some(l + r),
                    "sub" => Some(l - r),
                    "mul" => Some(l * r),
                    "div" => (r != 0.0).then_some(l / r),
                    _ => None,
                };
            }
            None
        }
    }
}

// --- Deep builders for generation ---

fn span0() -> chelis_deep::Span {
    chelis_deep::Span::new(0, 0)
}
fn sym(s: &str) -> Expr {
    Expr::Atom(Atom::Symbol(s.to_string()), span0())
}
fn node(tag: &str, kids: Vec<Expr>) -> Expr {
    let mut elements = vec![sym(tag), Expr::Map(Default::default(), span0())];
    elements.extend(kids);
    Expr::List(chelis_deep::ast::List { elements }, span0())
}
fn record_node(children_after_tag: Vec<Expr>) -> Expr {
    node("record", children_after_tag)
}
fn kv_node(field: &str, value: Expr) -> Expr {
    node("kv", vec![sym(field), value])
}
fn var_node(name: &str) -> Expr {
    node("var", vec![sym(name)])
}
fn app_node(items: Vec<Expr>) -> Expr {
    node("app", items)
}
fn access_node(target: Expr, field: &str) -> Expr {
    node("access", vec![target, sym(field)])
}
fn typed_lit(prim: &str, value: Expr) -> Expr {
    let mut entries = chelis_deep::ast::MetaMap::default();
    entries
        .entries
        .push(("type".to_string(), node("t-prim", vec![sym(prim)])));
    Expr::List(
        chelis_deep::ast::List {
            elements: vec![sym("lit"), Expr::Map(entries, span0()), value],
        },
        span0(),
    )
}
fn float_lit(v: f64, prim: &str) -> Expr {
    let lit = typed_lit("f32", Expr::Atom(Atom::Float(v), span0()));
    if prim == "f64" {
        node("cast", vec![lit, node("t-prim", vec![sym("f64")])])
    } else {
        lit
    }
}
fn int_lit(v: i64, prim: &str) -> Expr {
    let lit = typed_lit("int32", Expr::Atom(Atom::Int(v), span0()));
    if prim == "int64" {
        node("cast", vec![lit, node("t-prim", vec![sym("int64")])])
    } else {
        lit
    }
}
fn bool_lit(v: bool) -> Expr {
    typed_lit("bool", Expr::Atom(Atom::Bool(v), span0()))
}
fn deep_cons_list(items: Vec<Expr>) -> Expr {
    items.into_iter().rev().fold(var_node("Nil"), |tail, item| {
        app_node(vec![var_node("Cons"), item, tail])
    })
}
/// Public wrapper: build a fixed-shape tensor value Deep expr (a
/// `to_tensor`/`pad_sequences` of typed float literals) for a sampled
/// producer input. Used by the obligation engine's Tier C tensor-input
/// sampling.
pub fn tensor_value_expr_pub(dims: &[usize], precision: &str, values: &[f64]) -> Expr {
    tensor_value_expr(dims, precision, values)
}

fn tensor_value_expr(dims: &[usize], precision: &str, values: &[f64]) -> Expr {
    let scalar = |v: f64| float_lit(v, precision);
    if dims.len() <= 1 {
        app_node(vec![
            var_node("to_tensor"),
            deep_cons_list(values.iter().copied().map(scalar).collect()),
        ])
    } else {
        let cols = dims[1];
        let rows = values
            .chunks(cols)
            .map(|row| deep_cons_list(row.iter().copied().map(scalar).collect()))
            .collect::<Vec<_>>();
        app_node(vec![
            var_node("pad_sequences"),
            deep_cons_list(rows),
            scalar(0.0),
        ])
    }
}
fn node_def(name: &str, body: Expr) -> Expr {
    node("def", vec![sym(name), body])
}
fn let_block(bind: &str, value: Expr, body: Expr) -> Expr {
    node("let", vec![node("bind", vec![sym(bind), value]), body])
}
fn match_some_else(scrut: Expr, bind: &str, some_body: Expr, else_body: Expr) -> Expr {
    node(
        "match",
        vec![
            scrut,
            node(
                "arm",
                vec![
                    node(
                        "pat-ctor",
                        vec![sym("Some"), node("pat-var", vec![sym(bind)])],
                    ),
                    Expr::List(chelis_deep::ast::List { elements: vec![] }, span0()),
                    some_body,
                ],
            ),
            node(
                "arm",
                vec![
                    node("pat-wild", vec![]),
                    Expr::List(chelis_deep::ast::List { elements: vec![] }, span0()),
                    else_body,
                ],
            ),
        ],
    )
}
/// A NaN-filled sentinel for the `None` branch of a constructor proposal,
/// detected as failure by the reader.
fn sentinel_for(fty: &FieldType) -> Expr {
    match fty {
        FieldType::Tensor { dims, precision } => {
            let count = dims.iter().product::<usize>().max(1);
            let values: Vec<f64> = (0..count).map(|_| f64::NAN).collect();
            // NaN literals are not representable; use 0/0 via div.
            let _ = values;
            // Build a tensor of (0.0 / 0.0).
            let nan = node(
                "app",
                vec![
                    var_node("div"),
                    float_lit(0.0, precision),
                    float_lit(0.0, precision),
                ],
            );
            let elems: Vec<Expr> = (0..count).map(|_| nan.clone()).collect();
            if dims.len() <= 1 {
                app_node(vec![var_node("to_tensor"), deep_cons_list(elems)])
            } else {
                let cols = dims[1];
                let rows = elems
                    .chunks(cols)
                    .map(|row| deep_cons_list(row.to_vec()))
                    .collect::<Vec<_>>();
                app_node(vec![var_node("pad_sequences"), deep_cons_list(rows), nan])
            }
        }
        _ => node(
            "app",
            vec![
                var_node("div"),
                float_lit(0.0, "f32"),
                float_lit(0.0, "f32"),
            ],
        ),
    }
}

/// Insert a single def into the module wrapper that defines `type_name`,
/// stripping the `invariant` / `invariant_amenability` metadata from every
/// deftype on the way. The invariant predicate fn node embeds
/// shape-sensitive ops (`sum` over a tensor field) that cannot be
/// IR-lowered without type annotation, so leaving it in the evaluated
/// module breaks the probe eval. The generator validates samples through
/// `concrete_eval`, never through this evaluated module, so dropping the
/// invariant here is sound — `opaque: true` is kept so construction stays
/// in-module-legal.
fn inject_into_module_with_source(exprs: &[Expr], type_name: &str, def: Expr) -> Vec<Expr> {
    fn module_defines(expr: &Expr, type_name: &str) -> bool {
        if tag(expr) == Some("deftype")
            && children(expr).first().and_then(symbol_text) == Some(type_name)
        {
            return true;
        }
        if let Expr::List(l, _) = expr {
            return l.elements.iter().any(|c| module_defines(c, type_name));
        }
        false
    }
    let mut out = Vec::with_capacity(exprs.len());
    let mut injected = false;
    for expr in exprs {
        let stripped = strip_invariant_metadata(expr);
        if !injected
            && tag(&stripped) == Some("module")
            && module_defines(&stripped, type_name)
            && let Expr::List(l, span) = &stripped
        {
            let mut elements = l.elements.clone();
            elements.push(def.clone());
            out.push(Expr::List(chelis_deep::ast::List { elements }, *span));
            injected = true;
        } else {
            out.push(stripped);
        }
    }
    if !injected {
        out.push(def);
    }
    out
}

/// Drop the `invariant` and `invariant_amenability` metadata keys from
/// every deftype, recursively. Keeps `opaque: true` so opacity is intact.
fn strip_invariant_metadata(expr: &Expr) -> Expr {
    match expr {
        Expr::List(list, span) => {
            let mut elements: Vec<Expr> =
                list.elements.iter().map(strip_invariant_metadata).collect();
            if list.elements.first().and_then(symbol_text) == Some("deftype")
                && let Some(Expr::Map(map, mspan)) = elements.get(1)
            {
                let kept: Vec<(String, Expr)> = map
                    .entries
                    .iter()
                    .filter(|(k, _)| k != "invariant" && k != "invariant_amenability")
                    .cloned()
                    .collect();
                elements[1] = Expr::Map(chelis_deep::ast::MetaMap { entries: kept }, *mspan);
            }
            Expr::List(chelis_deep::ast::List { elements }, *span)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod generate_tests;
#[cfg(test)]
mod tests;
