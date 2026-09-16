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

use chelis_deep::DeepTag;
use chelis_deep::annotations::{MetadataKey as K, MetadataValue as M, TypeSyntax};
use chelis_unord::{UnordMap, UnordSet};

use chelis_deep::ast::{Atom, Expr};
use chelis_pred::PredAmenability;
use chelis_types::{ScalarValue, scalar_from_f64, scalar_from_i64, types::Prim};

use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr, SmtSort};

/// Maximum number of scalar solver variables a single opaque binder may
/// flatten to before Tier B falls back to Tier C (RFC D-TIERB tensor
/// flattening cap).
pub const TIER_B_SCALAR_CAP: usize = 64;

/// Whether a primitive type NAME is one of the signed integer widths (F3 /
/// review-5). This defers to the type system's own integer recognizer
/// ([`chelis_types::types::Prim::is_integer`]) so the int-width set has ONE
/// definition for the whole workspace: a name is an int width iff it parses
/// to a `Prim` the type system classifies as integer. Every prove-layer site
/// that decides "is this prim an integer" / "what SMT sort" / "how to sample
/// or build a literal" routes through this (and [`prim_to_smt_sort`] /
/// [`int_sample_bounds`]), so a new integer width added to the type system
/// cannot silently diverge across the prove paths.
pub fn is_int_width(prim: &str) -> bool {
    chelis_types::types::Prim::parse_name(prim).is_some_and(|p| p.is_integer())
}

/// The SMT sort a scalar primitive type lowers to (F3 single source). Every
/// integer width -> `Int`; `bool` -> `Bool`; everything else (f32/f64 and
/// any unrecognized name) -> `Real`. All four sort-deciding sites -- the
/// field [`FieldType::scalar_sort`], the producer-param sort in
/// `tier_b_lower`, the `@property` param sort in `property_runner`, and the
/// in-module constant lowering -- route through this one function.
pub fn prim_to_smt_sort(prim: &str) -> SmtSort {
    if is_int_width(prim) {
        SmtSort::Int
    } else if prim == "bool" {
        SmtSort::Bool
    } else {
        SmtSort::Real
    }
}

/// Fuzz-sampling bounds for an integer width (F3, sampling axis): the
/// `[-1000, 1000]` convenience range clamped to the width's representable
/// range, so an `i8` field samples in `[-128, 127]` (never an
/// unrepresentable value that would yield a spurious counterexample) while
/// wider widths keep the convenience range. Single source for every
/// opaque-field integer sampling site. Returns `None` for a non-integer
/// primitive.
pub fn int_sample_bounds(prim: &str) -> Option<(i64, i64)> {
    // The single workspace source for integer fuzz bounds is
    // `Prim::integer_fuzz_bounds` (the per-width representable range clamped
    // to the [-1000, 1000] convenience window); this is a thin name-keyed
    // wrapper the prove paths call.
    chelis_types::types::Prim::parse_name(prim)?.integer_fuzz_bounds()
}

/// A field of an opaque type's single record variant, in the V1 value
/// class (RFC D-WF): a scalar prim, a fixed-shape numeric tensor, or a
/// nested single-variant record of those.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldType {
    /// A scalar primitive: `f32`/`f64`/`i32`/`i64`/`bool`.
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

    /// The SMT sort of a scalar field; `None` for non-scalar fields. Routes
    /// through the single-source [`prim_to_smt_sort`] (F3) so an integer
    /// field of ANY width is `Int`, matching the constant lowering.
    pub fn scalar_sort(&self) -> Option<SmtSort> {
        match self {
            FieldType::Scalar(name) => Some(prim_to_smt_sort(name)),
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

    /// The discharged proposition this invariant stands for, in canonical Deep
    /// text (chelis#436): the predicate body bound to its binder, rendered flat
    /// (one line) by the canonical Deep printer. This is the proposition every
    /// produced value of the opaque type must satisfy, so an obligation record
    /// carries exactly what it discharged. Falls back to the full predicate fn
    /// node if the body cannot be isolated (a malformed predicate that never
    /// reaches a real obligation outcome anyway).
    pub fn goal_text(&self) -> String {
        let node = predicate_body(&self.predicate).unwrap_or(&self.predicate);
        // Strip lowering/producer metadata (spans, types) so the goal is the
        // bare proposition a consumer can display, not the lowering's internal
        // annotations (chelis#436).
        chelis_deep::printer::print_expr_flat(&chelis_deep::ast::strip_metadata(node))
    }
}

// ===========================================================================
// Deep node helpers (mirrors chelis-pred's structural helpers)
// ===========================================================================

fn tag(expr: &Expr) -> Option<DeepTag> {
    match expr {
        Expr::Node(node, _) => Some(node.tag()),
        Expr::List(list, _) => list.tag(),
        _ => None,
    }
}

fn children(expr: &Expr) -> &[Expr] {
    match expr {
        Expr::Node(node, _) => node.children_slice(),
        Expr::List(list, _) if list.elements.len() >= 2 => &list.elements[2..],
        _ => &[],
    }
}

fn symbol_text(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn annotations(expr: &Expr) -> Option<&chelis_deep::Metadata> {
    match expr {
        Expr::Node(node, _) => Some(node.meta()),
        Expr::List(list, _) => match list.elements.get(1) {
            Some(Expr::Map(meta, _)) => Some(meta),
            _ => None,
        },
        _ => None,
    }
}

// ===========================================================================
// Collection
// ===========================================================================

/// Walk a desugared Deep program and collect every invariant-carrying
/// opaque type. Non-opaque types, and opaque types without an invariant,
/// are skipped (the latter is plain opacity, unaffected by injection;
/// RFC D-INJECT test-lock).
pub fn collect_opaque_invariants(exprs: &[Expr]) -> Vec<OpaqueInvariant> {
    collect_opaque_invariants_and_rejections(exprs).0
}

/// An invariant-carrying opaque type whose representation the prover CANNOT
/// model, with a human reason. The obligation engine turns each into a
/// covered-or-rejected `Error` outcome so such a type is never silently
/// dropped -- dropping it would let a VIOLATING exported producer pass with
/// zero obligations (the exact "covered-or-rejected" hole a review found).
#[derive(Debug, Clone)]
pub struct OpaqueInvariantRejection {
    pub type_name: String,
    pub reason: String,
}

/// Collect the modelable invariants AND the rejections in one pass. Building
/// the program's `deftype` index once lets a nested-record field resolve the
/// record it references.
pub fn collect_opaque_invariants_and_rejections(
    exprs: &[Expr],
) -> (Vec<OpaqueInvariant>, Vec<OpaqueInvariantRejection>) {
    let mut deftypes: UnordMap<String, &Expr> = UnordMap::new();
    index_deftypes(exprs, &mut deftypes);
    let mut oks = Vec::new();
    let mut errs = Vec::new();
    for expr in exprs {
        collect_in(expr, &deftypes, &mut oks, &mut errs);
    }
    (oks, errs)
}

/// Collect ONLY the rejections (see
/// [`collect_opaque_invariants_and_rejections`]).
pub fn collect_opaque_invariant_rejections(exprs: &[Expr]) -> Vec<OpaqueInvariantRejection> {
    collect_opaque_invariants_and_rejections(exprs).1
}

/// Index every `deftype` in the program (recursing module wrappers) by its
/// type name, so a nested-record field type (`t-adt` naming another record)
/// resolves while the field model is built.
fn index_deftypes<'a>(exprs: &'a [Expr], out: &mut UnordMap<String, &'a Expr>) {
    for expr in exprs {
        if tag(expr) == Some(DeepTag::Deftype)
            && let Some(name) = children(expr).first().and_then(|n| symbol_text(n))
        {
            out.entry(name.to_string()).or_insert(expr);
        }
        for child in children(expr) {
            index_deftypes(std::slice::from_ref(child), out);
        }
    }
}

fn collect_in<'a>(
    expr: &'a Expr,
    deftypes: &UnordMap<String, &'a Expr>,
    oks: &mut Vec<OpaqueInvariant>,
    errs: &mut Vec<OpaqueInvariantRejection>,
) {
    if tag(expr) == Some(DeepTag::Deftype) {
        match opaque_invariant_from_deftype(expr, deftypes) {
            Some(Ok(inv)) => oks.push(inv),
            Some(Err(rej)) => errs.push(rej),
            None => {}
        }
    }
    // Recurse into module wrappers and any nesting.
    for child in children(expr) {
        collect_in(child, deftypes, oks, errs);
    }
}

/// Model a single `deftype`. Returns:
/// - `None` -- not an invariant-carrying opaque type (legitimately skipped).
/// - `Some(Err(_))` -- an invariant-carrying opaque type the prover cannot
///   model (covered-or-rejected; NEVER a silent skip).
/// - `Some(Ok(_))` -- a modelable invariant.
fn opaque_invariant_from_deftype(
    deftype: &Expr,
    deftypes: &UnordMap<String, &Expr>,
) -> Option<Result<OpaqueInvariant, OpaqueInvariantRejection>> {
    // Require opaque: true and an invariant fn node in the metadata. Absent
    // either, this is not an invariant-carrying opaque type -> skip.
    let metadata = annotations(deftype)?;
    metadata.opaque()?;
    let predicate =
        crate::deep_compat::normalize_nodes_to_lists(&[metadata.invariant()?.to_expression()])
            .remove(0);

    // From here it IS an invariant-carrying opaque type. ANY failure to model
    // its representation is a covered-or-rejected ERROR, never a silent skip.
    let kids = children(deftype);
    let type_name = kids
        .first()
        .and_then(|n| symbol_text(n))
        .map(str::to_string)
        .unwrap_or_else(|| "<anonymous>".to_string());
    let reject = |reason: String| {
        Some(Err(OpaqueInvariantRejection {
            type_name: type_name.clone(),
            reason,
        }))
    };

    let Some(binder) = predicate_binder(&predicate) else {
        return reject(format!(
            "invariant on opaque type `{type_name}` has a malformed binder"
        ));
    };
    let Some(variant) = kids.iter().find(|c| tag(c) == Some(DeepTag::Variant)) else {
        return reject(format!(
            "opaque type `{type_name}` carries an invariant but is not a single record variant"
        ));
    };
    let var_kids = children(variant);
    let Some(ctor_name) = var_kids
        .first()
        .and_then(|n| symbol_text(n))
        .map(str::to_string)
    else {
        return reject(format!(
            "opaque type `{type_name}` has a malformed record variant"
        ));
    };
    let mut fields = Vec::new();
    for field in var_kids.iter().skip(1) {
        if tag(field) != Some(DeepTag::Field) {
            return reject(format!(
                "opaque type `{type_name}` is a positional variant, which cannot carry a \
                 mechanically verifiable invariant"
            ));
        }
        let fk = children(field);
        let Some(fname) = fk.first().and_then(|n| symbol_text(n)).map(str::to_string) else {
            return reject(format!("opaque type `{type_name}` has a malformed field"));
        };
        let Some(fty_node) = fk.get(1) else {
            return reject(format!(
                "field `{fname}` of opaque type `{type_name}` has no type"
            ));
        };
        let mut visiting = UnordSet::new();
        let Some(fty) = field_type_from_deep(fty_node, deftypes, &mut visiting) else {
            return reject(format!(
                "field `{fname}` of opaque type `{type_name}` has a representation type the prover \
                 cannot model, so its invariant cannot be mechanically verified (V1 value class: \
                 numeric/bool scalars, f32/f64 tensors, or nested single-variant records of those)"
            ));
        };
        fields.push((fname, fty));
    }
    if fields.is_empty() {
        return reject(format!(
            "opaque type `{type_name}` has an empty record representation"
        ));
    }

    // RFC D-META: recompute amenability from the predicate; do not trust
    // the recorded `invariant_amenability` string.
    let amenability = chelis_pred::classify_predicate(&predicate);

    Some(Ok(OpaqueInvariant {
        type_name,
        ctor_name,
        fields,
        predicate,
        binder,
        amenability,
    }))
}

fn predicate_binder(fn_node: &Expr) -> Option<String> {
    let kids = children(fn_node);
    let params = kids.first()?;
    if tag(params) != Some(DeepTag::Params) {
        return None;
    }
    let pkids = children(params);
    let first = pkids.first()?;
    // The desugarer emits a bare symbol binder `(params {} p)`.
    if let Some(name) = symbol_text(first) {
        return Some(name.to_string());
    }
    // A typed-param list `(p {type: ...})`: head symbol is the name.
    let elements = match first {
        Expr::BareList(elements, _) => Some(elements.as_slice()),
        Expr::List(list, _) => Some(list.elements.as_slice()),
        _ => None,
    };
    if let Some(name) = elements
        .and_then(|elements| elements.first())
        .and_then(symbol_text)
    {
        return Some(name.to_string());
    }
    None
}

/// Parse a Deep type node into a [`FieldType`] in the V1 value class.
/// Returns `None` for anything outside the class (functions, lists,
/// symbolic-dim or non-numeric tensors, non-numeric scalar prims such as
/// `string`/`f8e4m3`, multi-variant or generic ADTs). `deftypes` resolves a
/// nested-record `t-adt` reference to its record; `visiting` breaks recursive
/// type cycles (a recursive ADT is not value-class). The scalar-prim class is
/// the SAME `chelis_types::invariants::invariant_value_class_prim` the D-WF
/// checker uses, so the checker and the prover agree on the value class.
fn field_type_from_deep(
    ty: &Expr,
    deftypes: &UnordMap<String, &Expr>,
    visiting: &mut UnordSet<String>,
) -> Option<FieldType> {
    match tag(ty)? {
        DeepTag::TPrim => {
            let name = symbol_text(children(ty).first()?)?;
            chelis_types::invariants::invariant_value_class_prim(name)
                .then(|| FieldType::Scalar(name.to_string()))
        }
        DeepTag::TTensor => {
            let kids = children(ty);
            // Last child is the precision t-prim; preceding are dims.
            let precision = {
                let last = kids.last()?;
                (tag(last) == Some(DeepTag::TPrim))
                    .then(|| symbol_text(children(last).first()?))
                    .flatten()?
            };
            if !matches!(precision, "f32" | "f64") {
                return None;
            }
            let mut dims = Vec::new();
            for dim in &kids[..kids.len().saturating_sub(1)] {
                // Literal dims only (`(d-lit {} N)`); symbolic dims reject.
                if tag(dim) == Some(DeepTag::DLit)
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
        DeepTag::TAdt => {
            // A nested single-variant record: resolve the referenced deftype
            // and model it as `FieldType::Record` (the V1 nested-record class,
            // spec/04 / RFC D-WF). A generic instantiation (a type argument) or
            // an unresolvable / recursive reference is out of the class.
            let adt_kids = children(ty);
            let name = symbol_text(adt_kids.first()?)?;
            if adt_kids.len() > 1 {
                return None; // parameterized record: out of V1 class
            }
            if !visiting.insert(name.to_string()) {
                return None; // cycle: recursive ADT, not value-class
            }
            let resolved = deftypes
                .get(name)
                .and_then(|referenced| record_field_type(referenced, deftypes, visiting));
            visiting.remove(name);
            resolved
        }
        _ => None,
    }
}

/// Model a referenced `deftype` as a nested-record [`FieldType::Record`]: it
/// must be a single record variant whose every field is itself in the value
/// class. Mirrors D-WF's `is_single_record_of_value_class` but BUILDS the
/// field model the prover flattens (`p.inner.value`).
fn record_field_type(
    deftype: &Expr,
    deftypes: &UnordMap<String, &Expr>,
    visiting: &mut UnordSet<String>,
) -> Option<FieldType> {
    let kids = children(deftype);
    let variants: Vec<_> = kids
        .iter()
        .filter(|c| tag(c) == Some(DeepTag::Variant))
        .collect();
    if variants.len() != 1 {
        return None;
    }
    let var_kids = children(variants[0]);
    let mut fields = Vec::new();
    for field in var_kids.iter().skip(1) {
        if tag(field) != Some(DeepTag::Field) {
            return None;
        }
        let fk = children(field);
        let fname = symbol_text(fk.first()?)?.to_string();
        let fty = field_type_from_deep(fk.get(1)?, deftypes, visiting)?;
        fields.push((fname, fty));
    }
    if fields.is_empty() {
        return None;
    }
    Some(FieldType::Record(fields))
}

// ===========================================================================
// Flattened-binder predicate lowering (Tier B + concrete validation)
// ===========================================================================

/// In-module zero-argument constant values referenced by a predicate
/// (`sum(p.weights) >= 1.0 - eps` references `eps`). Resolved to concrete
/// `f64` values by the caller (RFC D-WF: in-grammar constant defs whose
/// bodies are themselves in-grammar).
pub type ConstEnv = chelis_unord::UnordMap<String, f64>;

/// Context threaded through predicate lowering: the binder name, the
/// dotted path prefix of the binder value, the binder's fields (so `sum`
/// over a tensor field expands to the right number of scalar terms), the
/// resolved in-module constant environment, and the defining program (so a
/// module constant lowers with its DECLARED numeric type, U2). `exprs` is
/// empty when no module is available (the concrete-eval-only test callers),
/// in which case a constant defaults to `Real` -- harmless because concrete
/// evaluation is sort-agnostic.
struct LowerCtx<'a> {
    binder: &'a str,
    prefix: &'a str,
    fields: &'a [(String, FieldType)],
    consts: &'a ConstEnv,
    exprs: &'a [Expr],
}

/// The numeric sort class of an in-module constant for SMT lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConstNumericType {
    Int,
    Real,
}

/// The ONE type-aware constant lowering (U2 review-3 unification): lower an
/// in-module constant reference `name` (resolved to `value`) to an
/// [`SmtExpr`], preserving its declared numeric type. An integer-typed
/// constant (i8/i16/i32/i64) lowers to `IntLit`; an f32/f64 (or an
/// unresolvable declared type) lowers to `RealLit`. Both the producer-body
/// path and the invariant/precondition path call this, so the SAME constant
/// can never lower as `IntLit` on one and `RealLit` on the other (the cvc5
/// sort-mismatch abort).
pub(crate) fn lower_const_ref(exprs: &[Expr], name: &str, value: f64) -> SmtExpr {
    match const_declared_numeric_type(exprs, name) {
        ConstNumericType::Int => SmtExpr::IntLit(value as i64),
        ConstNumericType::Real => SmtExpr::RealLit(value),
    }
}

/// The numeric sort class (Int vs Real) of an in-module constant `name`.
/// `Int` for any declared integer width (i8/i16/i32/i64), `Real`
/// otherwise. The single source both [`lower_const_ref`] and the
/// producer-body `const_lit_node` consult.
pub(crate) fn const_declared_numeric_type(exprs: &[Expr], name: &str) -> ConstNumericType {
    match const_declared_int_type(exprs, name) {
        Some(_) => ConstNumericType::Int,
        None => ConstNumericType::Real,
    }
}

/// The declared integer primitive type (i8/i16/i32/i64) of an
/// in-module constant `name`, or `None` if it is not declared with an
/// integer type. Reads the DECLARED return type from a sibling `(defsig
/// name <type>)` (authoritative -- a typed `def a() -> i8 = 1` carries
/// i8 in the defsig but the default i32 on the body literal), then the
/// body literal's own type tag, following a const -> const reference chain
/// transitively (depth-bounded) to the literal that carries the type tag.
pub(crate) fn const_declared_int_type(exprs: &[Expr], name: &str) -> Option<String> {
    /// The declared numeric prim of a sibling `(defsig name <type>)`, where
    /// `<type>` is a bare `(t-prim {} P)` or a `(t-fn ... (t-prim {} P))`
    /// whose LAST element is the return type.
    fn defsig_prim(exprs: &[Expr], name: &str) -> Option<String> {
        fn prim_of_type(ty: &Expr) -> Option<String> {
            match tag(ty)? {
                DeepTag::TPrim => symbol_text(children(ty).first()?).map(str::to_string),
                DeepTag::TFn => prim_of_type(children(ty).last()?),
                _ => None,
            }
        }
        fn scan(exprs: &[Expr], name: &str) -> Option<String> {
            for expr in exprs {
                if tag(expr) == Some(DeepTag::Defsig) {
                    let kids = children(expr);
                    if kids.first().and_then(symbol_text) == Some(name)
                        && let Some(ty) = kids.get(1)
                        && let Some(prim) = prim_of_type(ty)
                    {
                        return Some(prim);
                    }
                }
                if let Some(found) = scan(children(expr), name) {
                    return Some(found);
                }
            }
            None
        }
        scan(exprs, name)
    }
    fn lit_type_prim(expr: &Expr) -> Option<String> {
        if tag(expr) != Some(DeepTag::Lit) {
            return None;
        }
        if let Some(ty) = annotations(expr)
            .and_then(|meta| meta.ty())
            .map(|ty| ty.expression())
            && tag(ty) == Some(DeepTag::TPrim)
        {
            return symbol_text(children(ty).first()?).map(str::to_string);
        }
        None
    }
    /// The constant a body references, if the body is a bare `(var other)`
    /// or a zero-arg `(app (var other))`.
    fn referenced_const(body: &Expr) -> Option<&str> {
        if tag(body) == Some(DeepTag::Var) {
            return symbol_text(children(body).first()?);
        }
        if tag(body) == Some(DeepTag::App) {
            let kids = children(body);
            if kids.len() == 1 {
                let callee = kids.first()?;
                if tag(callee) == Some(DeepTag::Var) {
                    return symbol_text(children(callee).first()?);
                }
            }
        }
        None
    }
    fn const_body<'a>(exprs: &'a [Expr], name: &str) -> Option<&'a Expr> {
        fn scan<'a>(exprs: &'a [Expr], name: &str) -> Option<&'a Expr> {
            for expr in exprs {
                if tag(expr) == Some(DeepTag::Def) {
                    let kids = children(expr);
                    if kids.first().and_then(symbol_text) == Some(name)
                        && let Some(body) = kids.get(1)
                    {
                        return if tag(body) == Some(DeepTag::Fn) {
                            children(body).get(1)
                        } else {
                            Some(body)
                        };
                    }
                }
                if let Some(found) = scan(children(expr), name) {
                    return Some(found);
                }
            }
            None
        }
        scan(exprs, name)
    }
    fn as_int_width(prim: String) -> Option<String> {
        is_int_width(&prim).then_some(prim)
    }
    // Follow the const -> const chain to the FULL depth with cycle detection
    // (F5): a magic depth cap silently dropped a long int alias chain to None
    // (Real), mis-sorting it. A visited-set terminates a cyclic / self-
    // referential chain instead. The declared (defsig) type is authoritative
    // over the body literal's own tag.
    let mut current = name.to_string();
    let mut visited: chelis_unord::UnordSet<String> = chelis_unord::UnordSet::new();
    loop {
        if !visited.insert(current.clone()) {
            // Re-entered a name already on the chain: a cycle. No declared
            // numeric type is reachable, so it is not an integer constant.
            return None;
        }
        if let Some(prim) = defsig_prim(exprs, &current) {
            return as_int_width(prim);
        }
        let body = const_body(exprs, &current)?;
        if let Some(prim) = lit_type_prim(body) {
            return as_int_width(prim);
        }
        current = referenced_const(body)?.to_string();
    }
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
    lower_predicate_flattened_in(inv, prefix, consts, &[])
}

/// Like [`lower_predicate_flattened`] but with the defining program `exprs`
/// available, so a module constant lowers with its DECLARED numeric type
/// (U2). The `exprs`-free entry point above defaults a constant to `Real`
/// (sound for the concrete-eval-only callers); the cvc5 precondition path
/// (`tier_b_lower::lower_obligation`) passes the real module so an int
/// constant against an int field is a sound integer comparison.
pub fn lower_predicate_flattened_in(
    inv: &OpaqueInvariant,
    prefix: &str,
    consts: &ConstEnv,
    exprs: &[Expr],
) -> Option<SmtExpr> {
    let body = predicate_body(&inv.predicate)?;
    let ctx = LowerCtx {
        binder: &inv.binder,
        prefix,
        fields: &inv.fields,
        consts,
        exprs,
    };
    lower_bool(body, &ctx)
}

fn predicate_body(fn_node: &Expr) -> Option<&Expr> {
    children(fn_node).get(1)
}

fn app_parts(expr: &Expr) -> Option<(&str, &[Expr])> {
    if tag(expr) == Some(DeepTag::App) {
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
    if tag(expr) == Some(DeepTag::Lit)
        && let Some(Expr::Atom(Atom::Bool(b), _)) = children(expr).first()
    {
        return Some(SmtExpr::BoolLit(*b));
    }
    if tag(expr) == Some(DeepTag::If) {
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
        "eq" | "neq" | "cmplt" | "gt" | "lte" | "gte" => {
            let op = match name {
                "eq" => CmpOp::Eq,
                "neq" => CmpOp::Ne,
                "cmplt" => CmpOp::Lt,
                "gt" => CmpOp::Gt,
                "lte" => CmpOp::Le,
                "gte" => CmpOp::Ge,
                _ => unreachable!(),
            };
            let mut l = lower_arith(args.first()?, ctx)?;
            let mut r = lower_arith(args.get(1)?, ctx)?;
            // U2 (precondition path): a bare module constant in a flattened
            // predicate is lowered as a RealLit by default (the ConstEnv
            // carries only an f64). When it is compared against an
            // integer-sorted field var, the precondition would otherwise mix
            // Int and Real sorts -- the same mismatch the obligation paths
            // fixed. Coerce an integral RealLit constant to an IntLit when the
            // other operand is integer-sorted, so an int-field input
            // invariant lowers consistently as a sound cvc5 precondition.
            coerce_cmp_operands(&mut l, &mut r, ctx);
            Some(SmtExpr::Cmp(op, Box::new(l), Box::new(r)))
        }
        // `<` desugars to `cmplt` and `>` to `gt` with authored operand
        // order (chelis#1180); `<=`/`>=` to `lte`/`gte`. Anything else is
        // not a boolean-shaped node we can lower.
        _ => None,
    }
}

/// Reconcile the operand sorts of a flattened-predicate comparison so an
/// integer-sorted field var and an integral module constant lower to the
/// same SMT sort (U2 precondition path). Delegates to the SHARED
/// [`reconcile_cmp_operands`] (F4) -- the SAME reconciliation the
/// cvc5-feeding producer-body path (`tier_b_lower::lower_pred_bool`) uses,
/// so the two comparison-lowering paths can never diverge.
fn coerce_cmp_operands(l: &mut SmtExpr, r: &mut SmtExpr, ctx: &LowerCtx) {
    let l_int = operand_is_int_sorted(l, ctx);
    let r_int = operand_is_int_sorted(r, ctx);
    reconcile_cmp_operands(l, r, l_int, r_int);
}

/// The ONE comparison-operand sort reconciliation (F4 review-4
/// unification): given whether each operand is integer-sorted, rewrite an
/// integral `RealLit` on the OTHER side to an `IntLit`, so an int operand
/// is never compared against a real literal (a sort mismatch cvc5 aborts
/// on). This is the only direction that can produce a sound int comparison
/// (a genuinely fractional constant against an int operand is a type error
/// the checker already rejects). Both the flattened-predicate path
/// (`coerce_cmp_operands`) and the producer-body path
/// (`tier_b_lower::lower_pred_bool`) call this, so no comparison-lowering
/// path lacks the reconciliation.
pub(crate) fn reconcile_cmp_operands(
    l: &mut SmtExpr,
    r: &mut SmtExpr,
    l_is_int: bool,
    r_is_int: bool,
) {
    if l_is_int {
        coerce_integral_real_to_int(r);
    }
    if r_is_int {
        coerce_integral_real_to_int(l);
    }
}

/// Whether an operand is integer-sorted: an `IntLit`, or a field `Var`
/// whose declared field type is an integer scalar.
fn operand_is_int_sorted(expr: &SmtExpr, ctx: &LowerCtx) -> bool {
    match expr {
        SmtExpr::IntLit(_) => true,
        SmtExpr::Var(path) => field_var_sort(path, ctx) == Some(SmtSort::Int),
        _ => false,
    }
}

/// The declared SMT sort of a flattened field var `<prefix>.<field>` (the
/// top-level field's `scalar_sort`), or `None` if it is not a top-level
/// scalar field of the binder.
fn field_var_sort(path: &str, ctx: &LowerCtx) -> Option<SmtSort> {
    let field = path.strip_prefix(ctx.prefix)?.strip_prefix('.')?;
    ctx.fields
        .iter()
        .find_map(|(n, f)| (n == field).then(|| f.scalar_sort()))
        .flatten()
}

/// Rewrite an integral `RealLit` to an `IntLit` (no-op for any other node).
pub(crate) fn coerce_integral_real_to_int(expr: &mut SmtExpr) {
    if let SmtExpr::RealLit(v) = expr
        && v.fract() == 0.0
        && v.is_finite()
    {
        *expr = SmtExpr::IntLit(*v as i64);
    }
}

fn lower_arith(expr: &Expr, ctx: &LowerCtx) -> Option<SmtExpr> {
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
    // Field projection: `(access (var binder|path) field)` -> dotted var.
    if tag(expr) == Some(DeepTag::Access) {
        return access_path(expr, ctx.binder, ctx.prefix).map(SmtExpr::Var);
    }
    // A bare `(var name)`: an in-module constant reference. Resolve it
    // through the constant environment AND the single type-aware constant
    // lowering (U2), so an int-typed constant lowers as `IntLit` here exactly
    // as it does on the producer-body path -- never a `RealLit` that would
    // mismatch an Int-sorted field var and abort cvc5 in a precondition. An
    // unknown constant means the caller could not resolve it in-grammar, so
    // the predicate is not lowerable here and the property falls to Tier C.
    if tag(expr) == Some(DeepTag::Var) {
        let name = symbol_text(children(expr).first()?)?;
        return ctx
            .consts
            .get(name)
            .copied()
            .map(|value| lower_const_ref(ctx.exprs, name, value));
    }
    if tag(expr) == Some(DeepTag::If) {
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
    if tag(target) == Some(DeepTag::Var) && symbol_text(children(target).first()?) == Some(binder) {
        return Some(format!("{prefix}.{field}"));
    }
    // Nested access: recurse.
    if tag(target) == Some(DeepTag::Access) {
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
    pub env: BTreeMap<String, ScalarValue>,
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

/// The single shared finiteness helper (U1 review-3 unification): true if
/// ANY value in the iterator is non-finite (NaN OR Inf). The produced-value
/// validation chokepoint (`obligation_engine::validate_produced_env`) and
/// the generator's `validate_env` both call this, so a non-finite
/// representation leaf can never slip through one path while the other
/// rejects it.
pub fn any_non_finite(values: impl IntoIterator<Item = ScalarValue>) -> bool {
    values
        .into_iter()
        .any(|value| value.prim().is_float() && !value.as_f64_lossy().is_finite())
}

/// Read an already sealed element without numeric conversion or finalization.
/// The storage carrier owns exhaustive dtype handling and preserves every bit.
pub(crate) fn tensor_element_scalar(
    elements: &chelis_compiler_api::schema::TensorElements,
    index: usize,
) -> Option<ScalarValue> {
    (index < elements.len()).then(|| elements.scalar_at(index))
}

fn scalar_json(value: ScalarValue) -> serde_json::Value {
    if let Some(value) = value.as_bool_exact() {
        serde_json::Value::from(value)
    } else if let Some(value) = value.as_i64_exact() {
        serde_json::Value::from(value)
    } else {
        serde_json::Value::from(value.as_f64_lossy())
    }
}

/// User-facing counterexample shape for a typed flattened environment.
pub fn generated_env_json(env: &BTreeMap<String, ScalarValue>) -> serde_json::Value {
    serde_json::Value::Object(
        env.iter()
            .map(|(name, value)| (name.clone(), scalar_json(*value)))
            .collect(),
    )
}

/// Validate a flattened field env against the predicate. When the
/// predicate lowers, use the fast concrete evaluator over [`SmtExpr`];
/// the env keys are exactly the lowered var names so the two agree by
/// construction.
fn validate_env(
    env: &BTreeMap<String, ScalarValue>,
    _inv: &OpaqueInvariant,
    predicate: &Option<SmtExpr>,
    _consts: &ConstEnv,
) -> bool {
    match predicate {
        Some(smt) => {
            // CR2-2 / U1 (fail-CLOSED on non-finite): a NaN/Inf field is
            // never a valid inhabitant, regardless of predicate shape.
            // `NaN != C` is true under strict IEEE, so a `!=`/negation-shaped
            // invariant would otherwise accept a non-finite sample fail-OPEN.
            // Reject any non-finite field before the predicate runs via the
            // SAME helper the obligation-engine chokepoint uses.
            if any_non_finite(env.values().copied()) {
                return false;
            }
            let hash = env.iter().map(|(k, v)| (k.clone(), *v)).collect();
            // STRICT validation (CR-2 / CR-5 / CR-10): invariant-sample
            // acceptance uses exact `==`/`!=`, never the fuzz `1e-10`
            // tolerance. An epsilon-validated sample would weaken exactly
            // the soundness that validation provides (RFC D-STARVE).
            crate::concrete_eval::eval_bool_strict(smt, &hash)
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
) -> BTreeMap<String, ScalarValue> {
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
    env: &mut BTreeMap<String, ScalarValue>,
) {
    match fty {
        FieldType::Scalar(name) => {
            let value = if let Some((lo, hi)) = int_sample_bounds(name) {
                scalar_from_i64(
                    "prove-opaque-sample",
                    Prim::parse_name(name).expect("integer width is a Prim"),
                    rng.next_i64(lo, hi),
                )
                .expect("integer sample bounds are representable")
            } else if name == "bool" {
                scalar_from_i64(
                    "prove-opaque-sample",
                    Prim::Bool,
                    i64::from(rng.next_bool()),
                )
                .expect("boolean samples are exactly zero or one")
            } else {
                scalar_from_f64(
                    "prove-opaque-sample",
                    Prim::parse_name(name).expect("opaque scalar dtype is classified"),
                    rng.next_f64(-10.0, 10.0),
                )
                .expect("sample is valid at its declared float width")
            };
            env.insert(path.to_string(), value);
        }
        FieldType::Tensor { dims, precision } => {
            let count = dims.iter().product::<usize>().max(1);
            let prim = Prim::parse_name(precision).expect("opaque tensor dtype is classified");
            for i in 0..count {
                let value = if let Some((lo, hi)) = int_sample_bounds(precision) {
                    scalar_from_i64("prove-opaque-sample", prim, rng.next_i64(lo, hi))
                        .expect("integer tensor sample bounds are representable")
                } else if precision == "bool" {
                    scalar_from_i64(
                        "prove-opaque-sample",
                        Prim::Bool,
                        i64::from(rng.next_bool()),
                    )
                    .expect("boolean tensor samples are exactly zero or one")
                } else {
                    scalar_from_f64("prove-opaque-sample", prim, rng.next_f64(-10.0, 10.0))
                        .expect("tensor sample is valid at its declared float width")
                };
                env.insert(format!("{path}.{i}"), value);
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
fn record_value_expr(inv: &OpaqueInvariant, env: &BTreeMap<String, ScalarValue>) -> Expr {
    let mut children = vec![Expr::Atom(
        Atom::Name(inv.ctor_name.clone()),
        chelis_deep::Span::new(0, 0),
    )];
    for (name, fty) in &inv.fields {
        let path = format!("{}.{}", inv.binder, name);
        let value = field_value_expr(&path, fty, env);
        children.push(kv_node(name, value));
    }
    record_node(children)
}

fn field_value_expr(path: &str, fty: &FieldType, env: &BTreeMap<String, ScalarValue>) -> Expr {
    match fty {
        FieldType::Scalar(name) => {
            if is_int_width(name) {
                int_lit(
                    env.get(path)
                        .and_then(ScalarValue::as_i64_exact)
                        .expect("generated integer field is present and exact"),
                    name,
                )
            } else if name == "bool" {
                bool_lit(
                    env.get(path)
                        .and_then(ScalarValue::as_bool_exact)
                        .expect("generated bool field is present and exact"),
                )
            } else {
                float_lit(
                    env.get(path)
                        .map(ScalarValue::as_f64_lossy)
                        .expect("generated float field is present"),
                    name,
                )
            }
        }
        FieldType::Tensor { dims, precision } => {
            let count = dims.iter().product::<usize>().max(1);
            let values: Vec<ScalarValue> = (0..count)
                .map(|i| {
                    env.get(&format!("{path}.{i}"))
                        .copied()
                        .unwrap_or_else(|| zero_scalar(precision))
                })
                .collect();
            tensor_value_expr_typed(dims, precision, &values)
        }
        FieldType::Record(inner) => {
            // A nested record value: `(record InnerCtor ...)`. The inner
            // ctor name is not tracked here in V1 nested support; build a
            // bare record with field kvs (the evaluator resolves the
            // single-variant ctor by field set).
            let mut children = vec![Expr::Atom(
                Atom::Name("__nested".to_string()),
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
/// evaluate it, Option-unwrap failures, and read back the produced record's
/// field values into the legacy f64 env. The conversion is deliberately
/// named lossy until chelis#688 / #729 Phase 2 replaces that env. Returns
/// `None` on producer failure (None result) or an unreadable output.
fn propose_via_constructor(
    inv: &OpaqueInvariant,
    module_source: &str,
    producers: &[GenProducer],
    rng: &mut GenRng,
) -> Option<BTreeMap<String, ScalarValue>> {
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

/// Lossily read one representation field of a producer's result into the
/// legacy f64 `env`.
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
    env: &mut BTreeMap<String, ScalarValue>,
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
    let exprs = chelis_deep::parser::parse_and_stamp_file(module_source).ok()?;
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
        FieldType::Tensor { dims, precision } => {
            // A tensor field access yields a Tensor value.
            let ExecutionValue::Tensor { value } = &root.value else {
                return None;
            };
            let count = dims.iter().product::<usize>().max(1);
            if value.data.len() != count {
                return None;
            }
            let expected = Prim::parse_name(precision)?;
            for i in 0..count {
                let value = tensor_element_scalar(&value.data, i)?;
                if value.prim() != expected {
                    return None;
                }
                // A None result yields the NaN-filled sentinel: treat as failure.
                if value.prim().is_float() && value.as_f64_lossy().is_nan() {
                    return Some(false);
                }
                env.insert(format!("{field_path}.{i}"), value);
            }
        }
        FieldType::Scalar(prim_name) => {
            // A scalar field access yields the exact-width scalar
            // ExecutionValue variant. Preserve that carrier identity rather
            // than widening through Float64 / Int64 or accepting a tagged
            // dtype substitution. A NaN result is the None-sentinel and
            // counts as a producer failure.
            let prim = Prim::parse_name(prim_name)?;
            let value = match &root.value {
                ExecutionValue::Scalar { value } if value.get().prim() == prim => value.get(),
                ExecutionValue::Bool { value } if prim == Prim::Bool => {
                    scalar_from_i64("prove-produced-field", Prim::Bool, i64::from(*value)).ok()?
                }
                // A rank-0/single-element tensor scalar, defensively.
                ExecutionValue::Tensor { value }
                    if value.validate().is_ok() && value.data.len() == 1 =>
                {
                    let value = tensor_element_scalar(&value.data, 0)?;
                    if value.prim() != prim {
                        return None;
                    }
                    value
                }
                _ => return None,
            };
            if value.prim().is_float() && value.as_f64_lossy().is_nan() {
                return Some(false);
            }
            env.insert(field_path.to_string(), value);
        }
        FieldType::Record(_) => return None,
    }
    Some(true)
}

fn sample_raw_input_expr(kind: &GenParamKind, rng: &mut GenRng) -> Expr {
    match kind {
        GenParamKind::Scalar(name) => {
            if let Some((lo, hi)) = int_sample_bounds(name) {
                int_lit(rng.next_i64(lo, hi), name)
            } else if name == "bool" {
                bool_lit(rng.next_bool())
            } else {
                float_lit(rng.next_f64(-10.0, 10.0), name)
            }
        }
        GenParamKind::Tensor { dims, precision } => {
            let count = dims.iter().product::<usize>().max(1);
            let prim = Prim::parse_name(precision).expect("producer tensor dtype is classified");
            let values = (0..count)
                .map(|_| {
                    if let Some((lo, hi)) = int_sample_bounds(precision) {
                        scalar_from_i64("prove-producer-sample", prim, rng.next_i64(lo, hi))
                            .expect("integer tensor sample bounds are representable")
                    } else if precision == "bool" {
                        scalar_from_i64(
                            "prove-producer-sample",
                            Prim::Bool,
                            i64::from(rng.next_bool()),
                        )
                        .expect("boolean tensor sample is exactly zero or one")
                    } else {
                        scalar_from_f64("prove-producer-sample", prim, rng.next_f64(-10.0, 10.0))
                            .expect("float tensor sample is valid at its declared width")
                    }
                })
                .collect::<Vec<_>>();
            tensor_value_expr_typed(dims, precision, &values)
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
    children(expr).iter().any(contains_equality_atom)
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
            if tag(expr) == Some(DeepTag::Lit) {
                return match children(expr).first() {
                    Some(Expr::Atom(Atom::Float(v), _)) => Some(*v),
                    Some(Expr::Atom(Atom::Int(v), _)) => Some(*v as f64),
                    _ => None,
                };
            }
            if tag(expr) == Some(DeepTag::Var) {
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
    Expr::Atom(Atom::Name(s.to_string()), span0())
}
fn node(tag: &str, kids: Vec<Expr>) -> Expr {
    let mut elements = vec![
        Expr::Atom(
            Atom::Tag(DeepTag::parse(tag).expect("vocabulary builder")),
            span0(),
        ),
        Expr::Map(Default::default(), span0()),
    ];
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
    let mut entries = chelis_deep::ast::Metadata::default();
    entries.replace(M::Type(
        TypeSyntax::try_new(node("t-prim", vec![sym(prim)])).expect("primitive type"),
    ));
    Expr::List(
        chelis_deep::ast::List {
            elements: vec![sym("lit"), Expr::Map(entries, span0()), value],
        },
        span0(),
    )
}
fn float_lit(v: f64, prim: &str) -> Expr {
    typed_lit(prim, Expr::Atom(Atom::Float(v), span0()))
}
/// A width-appropriate integer literal for an internal Deep value. This is
/// already below Surf's unsuffixed-literal defaulting boundary, so stamp the
/// declared dtype directly; routing an i64 payload through an i32 literal
/// would reject exact values outside the i32 range before the cast ran.
fn int_lit(v: i64, prim: &str) -> Expr {
    typed_lit(prim, Expr::Atom(Atom::Int(v), span0()))
}
fn bool_lit(v: bool) -> Expr {
    typed_lit("bool", Expr::Atom(Atom::Bool(v), span0()))
}
fn deep_cons_list(items: Vec<Expr>) -> Expr {
    items.into_iter().rev().fold(var_node("Nil"), |tail, item| {
        app_node(vec![var_node("Cons"), item, tail])
    })
}
pub(crate) fn tensor_value_expr_typed(
    dims: &[usize],
    precision: &str,
    values: &[ScalarValue],
) -> Expr {
    let expected = Prim::parse_name(precision).expect("sampled tensor dtype is classified");
    let scalar = |value: ScalarValue| {
        assert_eq!(
            value.prim(),
            expected,
            "sampled tensor element must carry its declared dtype"
        );
        if is_int_width(precision) {
            int_lit(
                value
                    .as_i64_exact()
                    .expect("integer tensor value has an exact integer payload"),
                precision,
            )
        } else if precision == "bool" {
            bool_lit(
                value
                    .as_bool_exact()
                    .expect("bool tensor value has an exact boolean payload"),
            )
        } else {
            float_lit(value.as_f64_lossy(), precision)
        }
    };
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
        let zero = if is_int_width(precision) {
            int_lit(0, precision)
        } else if precision == "bool" {
            bool_lit(false)
        } else {
            float_lit(0.0, precision)
        };
        app_node(vec![var_node("pad_sequences"), deep_cons_list(rows), zero])
    }
}

fn zero_scalar(precision: &str) -> ScalarValue {
    let prim = Prim::parse_name(precision).expect("tensor dtype is classified");
    if precision == "bool" || is_int_width(precision) {
        scalar_from_i64("prove-opaque-default", prim, 0)
            .expect("zero is representable at every integer/bool dtype")
    } else {
        scalar_from_f64("prove-opaque-default", prim, 0.0)
            .expect("zero is representable at every float dtype")
    }
}

pub(crate) fn scalar_values_json(values: &[ScalarValue]) -> serde_json::Value {
    serde_json::Value::Array(
        values
            .iter()
            .copied()
            .map(|value| {
                if let Some(value) = value.as_bool_exact() {
                    serde_json::Value::from(value)
                } else if let Some(value) = value.as_i64_exact() {
                    serde_json::Value::from(value)
                } else {
                    serde_json::Value::from(value.as_f64_lossy())
                }
            })
            .collect(),
    )
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
        if tag(expr) == Some(DeepTag::Deftype)
            && children(expr).first().and_then(symbol_text) == Some(type_name)
        {
            return true;
        }
        children(expr)
            .iter()
            .any(|child| module_defines(child, type_name))
    }
    let mut out = Vec::with_capacity(exprs.len());
    let mut injected = false;
    for expr in exprs {
        let stripped = strip_invariant_metadata(expr);
        if !injected
            && tag(&stripped) == Some(DeepTag::Module)
            && module_defines(&stripped, type_name)
        {
            match &stripped {
                // The probe `def` this module receives is built in the
                // deprecated legacy `List` carrier (see `node` above), so it
                // cannot be pushed under a stamped `Module` Node: that node
                // revalidates its whole subtree and rejects a raw
                // closed-vocabulary tag below the gate. Inject into the
                // module's canonical List form instead. The result is printed
                // and reparsed by `eval_selected` immediately below, so the
                // carrier is transient and the emitted text is unchanged.
                Expr::Node(node, span) => {
                    let mut elements = node.to_list(*span).elements;
                    elements.push(def.clone());
                    out.push(Expr::List(chelis_deep::ast::List { elements }, *span));
                    injected = true;
                    continue;
                }
                Expr::List(list, span) => {
                    let mut elements = list.elements.clone();
                    elements.push(def.clone());
                    out.push(Expr::List(chelis_deep::ast::List { elements }, *span));
                    injected = true;
                    continue;
                }
                _ => {}
            }
        }
        out.push(stripped);
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
        Expr::Node(node, span) => {
            let tag = node.tag();
            let mut meta = node.meta().clone();
            if tag == DeepTag::Deftype {
                meta.remove(K::Invariant);
                meta.remove(K::InvariantAmenability);
            }
            let children = node
                .children_slice()
                .iter()
                .map(strip_invariant_metadata)
                .collect();
            Expr::Node(
                Box::new(chelis_deep::node::Node::new(tag, meta, children)),
                *span,
            )
        }
        Expr::List(list, span) => {
            let mut elements: Vec<Expr> =
                list.elements.iter().map(strip_invariant_metadata).collect();
            if list.tag() == Some(DeepTag::Deftype)
                && let Some(Expr::Map(map, mspan)) = elements.get(1)
            {
                let mut metadata = map.clone();
                metadata.remove(K::Invariant);
                metadata.remove(K::InvariantAmenability);
                elements[1] = Expr::Map(metadata, *mspan);
            }
            Expr::List(chelis_deep::ast::List { elements }, *span)
        }
        Expr::BareList(elements, span) => Expr::BareList(
            elements.iter().map(strip_invariant_metadata).collect(),
            *span,
        ),
        Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(chelis_deep::ast::UnknownFormData {
            head: data.head.clone(),
            meta: data.meta.clone(),
            children: data.children.iter().map(strip_invariant_metadata).collect(),
            span: data.span,
        })),
        other => other.clone(),
    }
}

#[cfg(test)]
mod generate_tests;
#[cfg(test)]
mod tests;
