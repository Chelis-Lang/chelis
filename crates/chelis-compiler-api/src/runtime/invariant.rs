//! Decode revalidation (RFC D-DECODE, spec/10 "Invariant revalidation at
//! decode boundaries"). This is the runtime-side machinery: the invariant
//! table built from `deftype` metadata, the representation-sanity pre-check
//! (NaN/Inf rejection, RFC H1), and `revalidate_adt_value`, which evaluates
//! the declared predicate through the interpreter's OWN `eval_expr` (the
//! compiler-api crate does NOT depend on chelis-prove). The public decode
//! chokepoint lives in `crate::decode`; it calls into this machinery.

use chelis_unord::{UnordMap, UnordSet};

use chelis_deep::Span;
use chelis_deep::ast::Atom;
use chelis_deep::ast::Expr;
use chelis_types::types::Prim;

use super::host_ops::render_value;
use super::transforms::{extract_prim_from_type_expr, prim_from_name, var_name};
// Brings the parent module's runtime types (`EvalContext`, `RuntimeValue`,
// `RuntimeTensorValue`) and the private Deep-shape helpers (`tag`, `children`,
// `get_meta`, `symbol_name`, `top_level_items`) into scope, mirroring the
// `use super::*` idiom the other `runtime` submodules use.
use super::*;

/// A declared opaque-type invariant, recovered from `deftype` metadata.
///
/// Keyed (in [`collect_type_invariants`]) by the **variant constructor
/// name** because [`RuntimeValue::Adt`] carries the constructor, not the
/// type name, and the reef linker leaves variant names un-mangled
/// (survey §2). The `fn_node` is the desugared predicate fn the Surf
/// desugarer embeds verbatim into the metadata
/// (`(fn {} (params {} <binder>) <body>)`, RFC D-META); evaluating it
/// requires binding `binder` to the value being checked and running the
/// interpreter's `eval_expr` on the body.
#[derive(Debug, Clone)]
pub(crate) struct InvariantPredicate {
    /// The opaque type's name (for diagnostics). The ADT value is keyed by
    /// constructor, but the violation message names the *type*.
    pub(crate) type_name: String,
    /// The single predicate binder (e.g. `p`).
    pub(crate) binder: String,
    /// The desugared predicate body -- element 2 of the embedded
    /// `(fn {} (params {} <binder>) <body>)` metadata node.
    pub(crate) body: Expr,
}

/// One constructor's entry in the decode-time invariant table.
///
/// A deftype that DECLARES an `invariant` metadata entry always contributes
/// an entry for each of its record-variant constructors. The distinction is
/// whether the declared metadata parses into a usable predicate:
///
/// - [`InvariantEntry::Predicate`] -- the metadata is the well-formed
///   `(fn {} (params {} <binder>) <body>)` shape and revalidation evaluates
///   it (the everyday path).
/// - [`InvariantEntry::Malformed`] -- the deftype declares an `invariant`
///   but the metadata is NOT that shape (a bare literal, a fn missing its
///   params or body, etc.). The predicate cannot be evaluated, and a
///   predicate that cannot be evaluated is a decode FAILURE (spec/10 §4.1).
///   Recording the malformed entry rather than dropping it keeps the decode
///   chokepoint fail-CLOSED: a structurally-valid payload for that
///   constructor is rejected instead of materializing with zero invariant
///   check. D-WF rejects malformed metadata at declaration, so this is not
///   reachable through a chelis-compiled module today, but the chokepoint
///   is the contract for the next external/hand-built codec.
///
/// A deftype with NO `invariant` entry is simply ABSENT from the table:
/// there is no invariant to check, which is legitimate, not a failure.
#[derive(Debug, Clone)]
pub(crate) enum InvariantEntry {
    /// The declared invariant parsed into an evaluable predicate.
    Predicate(InvariantPredicate),
    /// The deftype declared an `invariant` whose metadata is malformed; the
    /// predicate cannot be evaluated, so the value cannot be safely
    /// materialized (fail-closed).
    Malformed { type_name: String },
}

/// A decode-time invariant failure. Distinct from a *structural* decode
/// error (wrong constructor, missing field, wrong field type), which the
/// chokepoint reports separately, because the two have different causes:
/// structural means "this payload is not even shaped like the type,"
/// invariant means "this payload is shaped correctly but its value is not
/// admissible." Never a repair -- RFC D-DECODE: "Decode of a violating
/// payload is a failure, never a repair."
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum InvariantViolation {
    /// A NaN or non-finite value appeared in a numeric representation
    /// field. Rejected BEFORE predicate evaluation (RFC H1): an
    /// in-grammar predicate such as `not (p.value > 1.0)` is *true* on
    /// NaN, so relying on the comparison to fail closed is unsound.
    NonFiniteRepresentation {
        type_name: String,
        field_path: String,
        detail: String,
    },
    /// The declared predicate evaluated to `false` on the value.
    PredicateFalse {
        type_name: String,
        invariant: String,
        value: String,
    },
    /// The declared predicate could not be evaluated to a boolean
    /// (partiality -- division by zero feeding a non-comparison position,
    /// a domain error in `log`/`sqrt`, or any interpreter error). Per
    /// RFC D-WF the predicate is pure-by-construction but NOT total, so
    /// an evaluation error fails the decode rather than being swallowed.
    PredicateError {
        type_name: String,
        invariant: String,
        reason: String,
    },
    /// The enclosing evaluation was cancelled while the predicate was
    /// running. NOT a property of the user's opaque type: attributing the
    /// cancellation sentinel to the type as a `PredicateError` would be a
    /// section C1.1 substituted response, so the sentinel passes through
    /// verbatim and stays matchable by `chelis_types::is_cancellation`
    /// (chelis#914 review).
    Cancelled { reason: String },
    /// The deftype declared an invariant whose metadata is malformed (not
    /// the `(fn {} (params {} <binder>) <body>)` shape). The predicate
    /// cannot be evaluated, so the value cannot be safely materialized
    /// (fail-closed, spec/10 §4.1: a predicate that cannot be evaluated is
    /// a decode failure). D-WF rejects this at declaration, so it is not
    /// reachable through a chelis-compiled module, but the decode chokepoint
    /// must still fail closed for a hand-built or external codec.
    MalformedInvariant { type_name: String },
}

impl std::fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InvariantViolation::NonFiniteRepresentation {
                type_name,
                field_path,
                detail,
            } => write!(
                f,
                "decode rejected for opaque type `{type_name}`: representation field `{field_path}` \
                 holds a non-finite value ({detail}); NaN and Inf are rejected before invariant \
                 evaluation (fail-closed, RFC D-DECODE H1)"
            ),
            InvariantViolation::PredicateFalse {
                type_name,
                invariant,
                value,
            } => write!(
                f,
                "decode rejected for opaque type `{type_name}`: value {value} violates the declared \
                 invariant `{invariant}`"
            ),
            InvariantViolation::PredicateError {
                type_name,
                invariant,
                reason,
            } => write!(
                f,
                "decode rejected for opaque type `{type_name}`: the declared invariant `{invariant}` \
                 could not be evaluated on the value ({reason})"
            ),
            // Verbatim: the sentinel must survive so downstream
            // `is_cancellation` checks and the CLI/pyo3 mappers see a
            // cancellation, not a decode failure of the user's type.
            InvariantViolation::Cancelled { reason } => write!(f, "{reason}"),
            InvariantViolation::MalformedInvariant { type_name } => write!(
                f,
                "decode rejected for opaque type `{type_name}`: the declared invariant metadata is \
                 malformed; the value cannot be safely materialized (fail-closed, RFC D-DECODE)"
            ),
        }
    }
}

/// Build the invariant table from a program's `deftype` declarations.
///
/// Scans every `deftype` carrying an `invariant` metadata entry and keys an
/// [`InvariantEntry`] by the type's single record-variant constructor name.
/// A deftype that DECLARES an invariant always contributes an entry per
/// record-variant ctor:
///
/// - When the metadata parses into the RFC D-META shape
///   `(fn {} (params {} <binder>) <body>)`, the entry is
///   [`InvariantEntry::Predicate`].
/// - When the metadata is present but MALFORMED (no fn, no binder, no body),
///   the entry is [`InvariantEntry::Malformed`]. It is NOT dropped:
///   dropping it would make the constructor look invariant-free, and a
///   structurally-valid payload would then decode with zero invariant check
///   (a fail-OPEN). spec/10 §4.1 requires a predicate that cannot be
///   evaluated to be a decode FAILURE, so the malformed entry is recorded
///   and revalidation rejects the value (fail-CLOSED). Declaration-time
///   well-formedness (RFC D-WF) rejects malformed metadata before it can
///   reach a chelis-compiled module, but the decode chokepoint is the
///   contract for the next external/hand-built codec and must fail closed.
/// - When the type-name child is NOT a readable symbol (chelis#1305), the
///   entry is [`InvariantEntry::Malformed`] under the
///   [`UNREADABLE_DEFTYPE_NAME`] placeholder. Dropping the whole deftype
///   was the same fail-open reached through a different door: the absent
///   entry was indistinguishable from "no declared invariant".
///
/// A deftype with NO `invariant` entry is simply absent from the table:
/// there is no invariant to check for its constructors, which is
/// legitimate, not a failure.
/// The placeholder type name recorded when an invariant-declaring `deftype`'s
/// type-name child is not a readable symbol (chelis#1305). It is diagnostic
/// wire data on a fail-closed path, not a [03-PROG-2] rejection
/// identification, so the angle-bracket spelling follows the
/// `UNKNOWN_FORM_NON_SYMBOL_HEAD` convention.
pub(crate) const UNREADABLE_DEFTYPE_NAME: &str = "<unreadable deftype name>";

pub(crate) fn collect_type_invariants(exprs: &[Expr]) -> UnordMap<String, InvariantEntry> {
    let mut out = UnordMap::new();
    for expr in top_level_items(exprs) {
        let Some((DeepTag::Deftype, meta, kids)) = expr_parts(expr) else {
            continue;
        };
        let Some(inv_value) = meta.invariant() else {
            // No declared invariant: this type contributes no table entry.
            continue;
        };
        // chelis#1305 (fail-closed): a deftype that DECLARES an invariant
        // but whose type-name child is not a readable symbol must not
        // contribute zero entries. `revalidate_adt_value` treats an absent
        // entry as "no declared invariant; nothing to check", so a
        // `continue` here was the same fail-open the malformed-metadata arm
        // below already closed, reached through a different door. The
        // unreadable-name declaration is malformed as a whole, so every
        // extractable constructor records `Malformed` under the placeholder
        // name, whatever the invariant metadata itself parses to.
        let readable_type_name = kids.first().and_then(symbol_name);
        // The deftype DECLARES an invariant, so it always contributes an
        // entry. Parse the metadata once; a malformed metadata becomes a
        // `Malformed` entry rather than being skipped (fail-closed).
        let parsed = match readable_type_name {
            Some(_) => parse_invariant_fn(&inv_value.to_expression()),
            None => None,
        };
        let type_name = readable_type_name.unwrap_or(UNREADABLE_DEFTYPE_NAME);
        // Key by every record-variant constructor of the type. The RFC's
        // single-record-variant representation means there is exactly one
        // in V1, but iterating keeps the table honest if that widens.
        for variant in kids.iter().skip(2) {
            let Some((DeepTag::Variant, _, variant_kids)) = expr_parts(variant) else {
                // Not a fail-open: a non-`variant` child declares no
                // constructor, so there is no key this table could record
                // (chelis#1305).
                continue;
            };
            let Some(ctor) = variant_kids.first().and_then(symbol_name) else {
                // Not a fail-open, but only because of the backstop: with no
                // readable ctor name there is no key to insert under, and
                // the same unreadable name is also absent from the
                // field-type table, so `decode_adt`'s unknown-constructor
                // guard rejects every payload claiming it (chelis#1305).
                continue;
            };
            let entry = match &parsed {
                Some((binder, body)) => InvariantEntry::Predicate(InvariantPredicate {
                    type_name: type_name.to_string(),
                    binder: binder.clone(),
                    body: body.clone(),
                }),
                None => InvariantEntry::Malformed {
                    type_name: type_name.to_string(),
                },
            };
            out.insert(ctor.to_string(), entry);
        }
    }
    out
}

/// Collect in-module zero-argument constant defs as `name -> value-body`
/// (CR-3, CR2-6). RFC D-WF permits an invariant predicate to reference
/// in-module zero-argument constant defs (e.g. `eps` in a tolerance band),
/// so decode revalidation must resolve those names when it evaluates the
/// predicate. The returned map keys each constant by name to its
/// VALUE-PRODUCING body; the decode eval context registers these as
/// `top_level_defs`, where `resolve_top_level` evaluates each to its
/// scalar/tensor value on first reference (transitive constant chains
/// resolve through the same map).
///
/// Two constant def shapes reach decode, both handled here:
///
/// - **Fn-wrapped form** `(def {} <name> (fn {} (params {}) <inner>))`. The
///   Surf desugarer wraps *every* def body in a `fn`, so both
///   `def eps() -> f32 = 0.001` and `def eps() = 0.001` desugar to this shape
///   with empty params. The constant's value is the evaluation of `<inner>`,
///   so the UNWRAPPED inner body is registered.
/// - **Bare value-binding form** `(def {} <name> <value>)` where `<value>`
///   is not a `fn` node (CR2-6). This is legal hand-authored Deep
///   (`validate --deep` accepts it) that the Surf desugarer never produces,
///   and the decode chokepoint consumes hand-authored Deep. The body IS the
///   value, so it is registered directly.
///
/// Fn defs with one or more parameters are deliberately excluded: a
/// well-formed predicate (RFC D-WF) does not call general functions, and a
/// bare reference to a multi-arg function in a value position is not in the
/// grammar. Registering only the zero-arg value bodies keeps the decode
/// context to exactly what D-WF admits, without re-running the checker.
///
/// A bare value-binding body is registered only when it is **constant
/// foldable** (review-3): a literal, an application of a predicate-grammar
/// arithmetic/intrinsic/comparison/boolean op over constant-foldable
/// arguments, or a reference to another genuine constant. A def whose body
/// is a `(var ...)` to a non-constant or unbound name (or any other
/// non-foldable expression) is NOT a constant and must not pollute the
/// table -- registering it would map the name to an unresolved or wrong
/// value at predicate evaluation time.
pub(crate) fn collect_zero_arg_constants(exprs: &[Expr]) -> UnordMap<String, Expr> {
    // Phase 1: collect candidate constant bodies keyed by name. A candidate
    // is the unwrapped value body of either the fn-wrapped zero-arg form or
    // the bare value-binding form. Parameterized fn defs are not candidates.
    let mut candidates: UnordMap<String, Expr> = UnordMap::new();
    for expr in top_level_items(exprs) {
        let Some((DeepTag::Def, _, kids)) = expr_parts(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some(body) = kids.get(1) else {
            continue;
        };
        let candidate_body = match expr_parts(body) {
            // Fn-wrapped form: keep only the empty-params (zero-arg) case,
            // and take the unwrapped inner value body.
            Some((DeepTag::Fn, _, fn_kids)) => {
                let Some((DeepTag::Params, _, params_kids)) = fn_kids.first().and_then(expr_parts)
                else {
                    continue;
                };
                if !params_kids.is_empty() {
                    // Non-empty params => a parameterized function, not a
                    // constant. Not in the predicate grammar as a value.
                    continue;
                }
                match fn_kids.get(1) {
                    Some(inner) => inner.clone(),
                    None => continue,
                }
            }
            // Bare value-binding form (CR2-6): the body is the value itself.
            _ => body.clone(),
        };
        candidates.insert(name.to_string(), candidate_body);
    }

    // Phase 2: keep only candidates whose body is constant foldable against
    // the candidate set (review-3). A `(var name)` body folds only when
    // `name` is itself a genuine constant; this transitively validates
    // constant aliases and chains while excluding references to
    // non-constant or unbound names.
    candidates
        .to_sorted()
        .into_iter()
        .filter(|(_, body)| {
            let mut visiting = UnordSet::new();
            is_constant_foldable(body, &candidates, &mut visiting)
        })
        .map(|(name, body)| (name.clone(), body.clone()))
        .collect()
}

/// True when `expr` is a constant-foldable predicate-grammar value
/// (review-3): a literal, an application of an admitted arithmetic /
/// intrinsic / comparison / boolean op over constant-foldable arguments, or
/// a `(var name)` reference to another constant in `candidates`. `visiting`
/// guards against cyclic constant references (e.g. `def a() = b; def b() = a`),
/// which are not foldable.
fn is_constant_foldable(
    expr: &Expr,
    candidates: &UnordMap<String, Expr>,
    visiting: &mut UnordSet<String>,
) -> bool {
    let Some((node_tag, _, kids)) = expr_parts(expr) else {
        // A bare atom (not wrapped in a Deep node) is not a value form.
        return false;
    };
    match node_tag {
        // A literal value is the base constant.
        DeepTag::Lit => true,
        // A reference folds only to another genuine constant; chase it.
        DeepTag::Var => {
            let Some(name) = kids.first().and_then(symbol_name) else {
                return false;
            };
            if visiting.contains(name) {
                // Cyclic constant reference: not foldable.
                return false;
            }
            let Some(referent) = candidates.get(name) else {
                // Reference to a non-constant or unbound name: not a constant.
                return false;
            };
            visiting.insert(name.to_string());
            let ok = is_constant_foldable(referent, candidates, visiting);
            visiting.remove(name);
            ok
        }
        // An application folds when the callee is an admitted predicate-
        // grammar op and every argument is itself constant foldable.
        DeepTag::App => {
            let Some(callee) = kids.first() else {
                return false;
            };
            let Some(op) = var_name(callee) else {
                return false;
            };
            if !is_constant_grammar_op(op) {
                return false;
            }
            kids[1..]
                .iter()
                .all(|arg| is_constant_foldable(arg, candidates, visiting))
        }
        // Any other node (match, fn, record, tensor op, effect, ...) is not
        // a constant value.
        _ => false,
    }
}

/// The closed set of operators a constant body may apply, mirroring the
/// predicate grammar (RFC D-WF, the `chelis_pred` grammar): arithmetic, the
/// whitelisted intrinsics, comparisons, and boolean connectives. `sum` is
/// intentionally excluded -- it ranges over a binder field projection,
/// which a constant (binder-free) body never has. The intrinsic list is
/// inlined rather than imported from `chelis_pred` to keep
/// `chelis-compiler-api`'s dependency graph unchanged; it is a closed,
/// stable D-WF set and any drift is locked by the constant-folding tests.
fn is_constant_grammar_op(op: &str) -> bool {
    const ARITH: &[&str] = &["add", "sub", "mul", "div", "neg"];
    const COMPARISON: &[&str] = &["eq", "neq", "cmplt", "gt", "lte", "gte"];
    const BOOL: &[&str] = &["and", "or", "not"];
    const INTRINSICS: &[&str] = &["abs", "min", "max", "sqrt", "exp", "log", "sin", "cos"];
    ARITH.contains(&op)
        || COMPARISON.contains(&op)
        || BOOL.contains(&op)
        || INTRINSICS.contains(&op)
}

/// The declared representation type of one record-variant field, used by
/// the decode chokepoint's STRUCTURAL check (RFC D-DECODE). Distinguishes
/// the V1 decodable value class (RFC D-WF): scalar prims, fixed-shape
/// numeric tensors, and nested single-variant records of those.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DecodeFieldType {
    /// A scalar primitive (`f32`, `int64`, `bool`, ...).
    Prim(Prim),
    /// A fixed-shape numeric tensor with the given element precision.
    Tensor(Prim),
    /// A nested ADT, referenced by name. Field decode recurses into the
    /// referenced constructor's field-type table.
    Adt(String),
}

/// One field of a record variant: its declared name and representation
/// type, in source order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DecodeField {
    pub(crate) name: String,
    pub(crate) ty: DecodeFieldType,
}

/// Build a per-constructor field-type table from a program's `deftype`
/// declarations. Keyed by variant constructor name (same key space as
/// [`collect_type_invariants`]). The decode chokepoint uses this for the
/// structural check (arity, names, scalar-vs-tensor-vs-adt field types)
/// before invariant revalidation.
pub(crate) fn collect_ctor_field_types(exprs: &[Expr]) -> UnordMap<String, Vec<DecodeField>> {
    let mut out = UnordMap::new();
    for expr in top_level_items(exprs) {
        let Some((DeepTag::Deftype, _, kids)) = expr_parts(expr) else {
            continue;
        };
        for variant in kids.iter().skip(2) {
            let Some((DeepTag::Variant, _, variant_kids)) = expr_parts(variant) else {
                continue;
            };
            let Some(ctor) = variant_kids.first().and_then(symbol_name) else {
                continue;
            };
            let mut fields = Vec::new();
            for field in variant_kids.iter().skip(1) {
                let Some((DeepTag::Field, _, field_kids)) = expr_parts(field) else {
                    continue;
                };
                let Some(name) = field_kids.first().and_then(symbol_name) else {
                    continue;
                };
                let Some(ty_expr) = field_kids.get(1) else {
                    continue;
                };
                let Some(ty) = decode_field_type(ty_expr) else {
                    continue;
                };
                fields.push(DecodeField {
                    name: name.to_string(),
                    ty,
                });
            }
            if !fields.is_empty() {
                out.insert(ctor.to_string(), fields);
            }
        }
    }
    out
}

/// Classify a Deep field-type expression into a [`DecodeFieldType`].
/// Returns `None` for type shapes outside the V1 decodable value class
/// (function types, generics, type variables) -- the chokepoint treats a
/// field with no classifiable type as outside the decodable surface.
fn decode_field_type(expr: &Expr) -> Option<DecodeFieldType> {
    let (tag, _, kids) = expr_parts(expr)?;
    match tag {
        DeepTag::TPrim => kids
            .first()
            .and_then(symbol_name)
            .and_then(prim_from_name)
            .map(DecodeFieldType::Prim),
        DeepTag::TTensor => kids
            .last()
            .and_then(extract_prim_from_type_expr)
            .map(DecodeFieldType::Tensor),
        DeepTag::TAdt => kids
            .first()
            .and_then(symbol_name)
            .map(|name| DecodeFieldType::Adt(name.to_string())),
        _ => None,
    }
}

/// Parse `(fn {} (params {} <binder>) <body>)` into `(binder, body)`.
fn parse_invariant_fn(expr: &Expr) -> Option<(String, Expr)> {
    let (tag, _, kids) = expr_parts(expr)?;
    if tag != DeepTag::Fn {
        return None;
    }
    let (params_tag, _, params_kids) = expr_parts(kids.first()?)?;
    if params_tag != DeepTag::Params {
        return None;
    }
    let binder = params_kids.first().and_then(symbol_name)?;
    let body = kids.get(1)?;
    Some((binder.to_string(), body.clone()))
}

/// Borrow the common stamped shape without reconstructing a legacy `List`.
fn expr_parts(expr: &Expr) -> Option<(DeepTag, &Metadata, &[Expr])> {
    match expr {
        Expr::List(list, _) => Some((tag(list)?, get_meta(list)?, children(list))),
        Expr::Node(node, _) => Some((node.tag(), node.meta(), node.children_slice())),
        _ => None,
    }
}

/// Render a [`InvariantPredicate`] body for a violation message. Uses the
/// canonical Deep printer (flat, single-line) so the invariant the user
/// wrote is reproduced in the diagnostic. `span` metadata keys are stripped
/// first so the message reads as the source predicate, not the
/// span-annotated desugar output, and stays stable across re-spanning.
fn render_invariant_body(pred: &InvariantPredicate) -> String {
    let mut body = pred.body.clone();
    strip_span_meta(&mut body);
    chelis_deep::printer::print_expr_flat(&body)
}

/// Recursively drop `span` entries from a Deep expression's metadata maps.
/// Diagnostics-only helper: never mutates a value that round-trips, only a
/// throwaway clone used to render the invariant text.
fn strip_annotation_spans(metadata: &mut chelis_deep::Metadata) {
    metadata.remove(chelis_deep::annotations::MetadataKey::Span);
    *metadata = metadata
        .try_map_expressions_with_annotations::<chelis_deep::metadata::MetadataError>(
            &mut |value, _| {
                let mut value = value.clone();
                strip_span_meta(&mut value);
                Ok(value)
            },
            &mut |mut metadata, _| {
                metadata.remove(chelis_deep::annotations::MetadataKey::Span);
                Ok(metadata)
            },
        )
        .expect("span removal preserves annotation payloads");
}

fn strip_span_meta(expr: &mut Expr) {
    match expr {
        Expr::List(list, _) => {
            for element in &mut list.elements {
                strip_span_meta(element);
            }
        }
        Expr::Map(map, _) => {
            strip_annotation_spans(map);
        }
        Expr::MetaExpr(meta, _) => {
            strip_annotation_spans(&mut meta.metadata);
            strip_span_meta(&mut meta.expr);
        }
        Expr::Atom(_, _) => {}
        Expr::Node(..) => {
            // Bridge: convert Node to List in place so mutable meta stripping works (#908).
            let placeholder = Expr::Atom(Atom::Bool(false), Span::new(0, 0));
            match std::mem::replace(expr, placeholder) {
                Expr::Node(node, span) => {
                    *expr = Expr::List(node.to_list(span), span);
                    strip_span_meta(expr);
                }
                _ => unreachable!(),
            }
        }
        // chelis#1087: span metadata inside either transitional variant
        // would otherwise leak into the rendered invariant text.
        Expr::BareList(elements, _) => {
            for element in elements {
                strip_span_meta(element);
            }
        }
        Expr::UnknownForm(data) => {
            strip_annotation_spans(&mut data.meta);
            for child in &mut data.children {
                strip_span_meta(child);
            }
        }
    }
}

/// Render a [`RuntimeValue`] compactly for a violation message.
fn render_value_for_diagnostic(value: &RuntimeValue) -> String {
    render_value(value)
}

/// Recursively reject any NaN or non-finite scalar/tensor element inside a
/// value (RFC H1 representation-sanity pre-check). `path` accumulates a
/// human-readable field path for the diagnostic.
fn check_representation_finite(
    value: &RuntimeValue,
    type_name: &str,
    ctor: &str,
    invariants: &UnordMap<String, InvariantEntry>,
    adt_fields: &UnordMap<String, Vec<String>>,
    path: &str,
) -> Result<(), InvariantViolation> {
    match value {
        RuntimeValue::Scalar(payload) if payload.dtype().is_float() => {
            let v = payload.as_f64_lossy();
            if !v.is_finite() {
                return Err(InvariantViolation::NonFiniteRepresentation {
                    type_name: type_name.to_string(),
                    field_path: path.to_string(),
                    detail: describe_non_finite(v),
                });
            }
            Ok(())
        }
        RuntimeValue::Tensor(tensor) => {
            for (index, elem) in tensor.value.to_f64_lossy_vec().into_iter().enumerate() {
                if !elem.is_finite() {
                    return Err(InvariantViolation::NonFiniteRepresentation {
                        type_name: type_name.to_string(),
                        field_path: format!("{path}[{index}]"),
                        detail: describe_non_finite(elem),
                    });
                }
            }
            Ok(())
        }
        RuntimeValue::Adt {
            ctor: inner_ctor,
            fields,
            field_names,
        } => {
            // The type/ctor naming the violation is the OUTERMOST opaque
            // type when we recursed from one; a nested record field that
            // is itself an invariant-carrying opaque type names ITSELF.
            // Both entry shapes carry the inner type name (a malformed
            // invariant still names its declaring type), so either one
            // re-targets the diagnostic to the inner type.
            let (name_for_field, ctor_for_field) = match invariants.get(inner_ctor) {
                Some(InvariantEntry::Predicate(inner)) => {
                    (inner.type_name.as_str(), inner_ctor.as_str())
                }
                Some(InvariantEntry::Malformed { type_name: inner }) => {
                    (inner.as_str(), inner_ctor.as_str())
                }
                None => (type_name, ctor),
            };
            let declared = adt_fields
                .get(inner_ctor)
                .cloned()
                .or_else(|| field_names.clone());
            for (index, field) in fields.iter().enumerate() {
                let field_label = declared
                    .as_ref()
                    .and_then(|names| names.get(index))
                    .cloned()
                    .unwrap_or_else(|| index.to_string());
                let next_path = if path.is_empty() {
                    field_label
                } else {
                    format!("{path}.{field_label}")
                };
                check_representation_finite(
                    field,
                    name_for_field,
                    ctor_for_field,
                    invariants,
                    adt_fields,
                    &next_path,
                )?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Spell a non-finite value for a diagnostic. Routes through
/// `format_element` so diagnostics share the one normative grammar
/// (`inf`/`-inf`/`NaN`, [05-OBS-2]); the pre-chelis#732 `+Inf`/`-Inf`
/// spellings migrated with the Phase 1 grammar freeze.
fn describe_non_finite(v: f64) -> String {
    chelis_types::format_element(
        chelis_types::types::Prim::F64,
        chelis_types::ElementRef::F64(v),
    )
}

/// Revalidate a (possibly nested) ADT value against the declared
/// invariants (RFC D-DECODE).
///
/// For every [`RuntimeValue::Adt`] whose constructor carries an invariant:
/// 1. **Representation sanity pre-check** (RFC H1): reject NaN/Inf in any
///    numeric representation field BEFORE predicate evaluation. This is
///    required for fail-closed behavior -- comparisons such as
///    `not (p.value > 1.0)` are *true* on NaN.
/// 2. **Predicate evaluation**: bind the predicate binder to the value and
///    run the interpreter's OWN `eval_expr` on the predicate body. A
///    non-boolean or `false` result fails the decode; an evaluation error
///    (partiality -- `/`, `log`, `sqrt`) also fails it (RFC D-WF).
///
/// The walk is recursive: nested record fields that are themselves
/// invariant-carrying opaque types are checked, and an inner violation is
/// reported naming the *inner* type. Never repairs.
///
/// `module_constants` carries the in-module zero-argument constant defs
/// (CR-3): RFC D-WF permits an invariant predicate to reference them, so
/// they are registered as `top_level_defs` in the predicate eval context.
/// See [`collect_zero_arg_constants`].
pub(crate) fn revalidate_adt_value(
    value: &RuntimeValue,
    invariants: &UnordMap<String, InvariantEntry>,
    adt_fields: &UnordMap<String, Vec<String>>,
    module_constants: &UnordMap<String, Expr>,
) -> Result<(), InvariantViolation> {
    let RuntimeValue::Adt { ctor, fields, .. } = value else {
        // Non-ADT values carry no opaque invariant; structural decode has
        // already vetted their shape. Recurse into containers so an opaque
        // value nested in a tuple/list is still checked.
        match value {
            RuntimeValue::List(items) | RuntimeValue::Tuple(items) => {
                for item in items {
                    revalidate_adt_value(item, invariants, adt_fields, module_constants)?;
                }
            }
            RuntimeValue::Dict(entries) => {
                for (key, val) in entries {
                    revalidate_adt_value(key, invariants, adt_fields, module_constants)?;
                    revalidate_adt_value(val, invariants, adt_fields, module_constants)?;
                }
            }
            _ => {}
        }
        return Ok(());
    };

    // Recurse into fields first so an inner opaque violation is reported
    // before the outer predicate (inner-most failure is the precise cause).
    for field in fields {
        revalidate_adt_value(field, invariants, adt_fields, module_constants)?;
    }

    // Match the constructor's table entry:
    // - absent  => no declared invariant; nothing to check (Ok).
    // - Malformed => the deftype declared an invariant whose metadata is
    //   malformed. The predicate cannot be evaluated, so the value cannot
    //   be safely materialized: fail CLOSED (spec/10 §4.1). Dropping this
    //   case (the old `continue`-skip in `collect_type_invariants`) made
    //   the constructor look invariant-free and decoded the payload with
    //   zero check -- a fail-OPEN. Reject before any representation walk.
    // - Predicate => evaluate it (the everyday path).
    let pred = match invariants.get(ctor) {
        None => return Ok(()),
        Some(InvariantEntry::Malformed { type_name }) => {
            return Err(InvariantViolation::MalformedInvariant {
                type_name: type_name.clone(),
            });
        }
        Some(InvariantEntry::Predicate(pred)) => pred,
    };

    // (1) Representation sanity pre-check (RFC H1). Walk the WHOLE value
    // (including nested record fields of this opaque type) and reject any
    // non-finite numeric component before evaluating the predicate.
    check_representation_finite(value, &pred.type_name, ctor, invariants, adt_fields, "")?;

    // (2) Predicate evaluation through the interpreter's own eval_expr.
    // In-module zero-argument constants (CR-3) are registered as
    // top_level_defs so a predicate referencing e.g. a tolerance `eps`
    // resolves it to its value instead of dying on "unknown runtime name".
    let empty_tensors: UnordMap<String, RuntimeTensorValue> = UnordMap::new();
    let mut ctx = EvalContext {
        bindings: UnordMap::new(),
        binding_types: UnordMap::new(),
        precision_bindings: UnordMap::new(),
        declaration_values: UnordMap::new(),
        named_axis_route_cache: UnordMap::new(),
        named_axis_route_visiting: UnordSet::new(),
        top_level_defs: module_constants.clone(),
        sorted_defs_snapshot: None,
        declared_signatures: UnordMap::new(),
        adt_registry: chelis_types::adt::AdtRegistry::default(),
        type_env: UnordMap::new(),
        adt_fields: adt_fields.clone(),
        tensor_bindings: &empty_tensors,
        session: None,
        def_kernels: UnordMap::new(),
        transcript: Vec::new(),
        transcript_capture: None,
        resolving_top_levels: Vec::new(),
        random_seed: None,
        random_counter: 0,
        execution_exclusion: None,
        // Invariant predicates run inside an enclosing evaluation, so they
        // honour whatever token that evaluation installed (chelis#914).
        cancel: chelis_types::current_cancel_token(),
    };
    ctx.bindings.insert(pred.binder.clone(), value.clone());

    let invariant_text = render_invariant_body(pred);
    match ctx.eval_expr(&pred.body) {
        Ok(RuntimeValue::Bool(true)) => Ok(()),
        Ok(RuntimeValue::Bool(false)) => Err(InvariantViolation::PredicateFalse {
            type_name: pred.type_name.clone(),
            invariant: invariant_text,
            value: render_value_for_diagnostic(value),
        }),
        Ok(other) => Err(InvariantViolation::PredicateError {
            type_name: pred.type_name.clone(),
            invariant: invariant_text,
            reason: format!("predicate did not evaluate to a boolean (got {other:?})"),
        }),
        // Cancellation is the enclosing evaluation's state, not a defect in
        // this predicate: pass the sentinel through before the attributing
        // arm below can claim it (chelis#914 review).
        Err(reason) if chelis_types::is_cancellation(&reason) => {
            Err(InvariantViolation::Cancelled { reason })
        }
        Err(reason) => Err(InvariantViolation::PredicateError {
            type_name: pred.type_name.clone(),
            invariant: invariant_text,
            reason,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::ast::{Metadata, UnknownFormData};

    fn declaration_scope_library() -> crate::pipeline::CheckedLibrary {
        let source = "module Bounds\np = 0.25f32\nceiling = add(1.0f32, p)\n@opaque\n@invariant(p) ceiling >= p.value && p.value >= 0.0f32\ntype Bounded = | Bounded { value: f32 }\n";
        let prepared =
            crate::pipeline::prepare_source(crate::schema::SourceKind::Surf, source, None)
                .expect("invariant source prepares");
        crate::pipeline::check_prepared_library(prepared)
            .expect("type, effect, linearity and invariant admission")
    }

    fn bounded(value: f64) -> RuntimeValue {
        RuntimeValue::Adt {
            ctor: "Bounded".into(),
            fields: vec![RuntimeValue::float_lit(value)],
            field_names: None,
        }
    }

    #[test]
    fn declaration_constant_initialization_does_not_capture_the_predicate_binder() {
        let library = declaration_scope_library();
        let exprs = library.program().exprs();
        let invariants = collect_type_invariants(exprs);
        let fields = collect_adt_ctor_fields(exprs);
        let constants = collect_zero_arg_constants(exprs);
        assert!(constants.contains_key("p") && constants.contains_key("ceiling"));
        // ceiling reads module p=0.25; subsequent field access must still
        // read the predicate's ADT p, not the newly initialized scalar.
        revalidate_adt_value(&bounded(1.125), &invariants, &fields, &constants).unwrap();
        assert!(matches!(
            revalidate_adt_value(&bounded(1.5), &invariants, &fields, &constants),
            Err(InvariantViolation::PredicateFalse { .. })
        ));
        revalidate_adt_value(&bounded(0.5), &invariants, &fields, &constants).unwrap();
    }

    #[test]
    fn declaration_invariant_cancellation_is_not_a_predicate_failure() {
        let library = declaration_scope_library();
        let exprs = library.program().exprs();
        let invariants = collect_type_invariants(exprs);
        let fields = collect_adt_ctor_fields(exprs);
        let constants = collect_zero_arg_constants(exprs);
        let token = chelis_types::CancelToken::new();
        {
            let _guard = chelis_types::install_cancel_token(token.clone());
            revalidate_adt_value(&bounded(1.125), &invariants, &fields, &constants).unwrap();
            token.cancel();
            // No evaluator/checker runs between cancellation and this call:
            // the sentinel must come from the predicate's own EvalContext.
            let error = revalidate_adt_value(&bounded(1.125), &invariants, &fields, &constants)
                .unwrap_err();
            assert!(
                matches!(&error, InvariantViolation::Cancelled { reason } if reason == chelis_types::EVAL_CANCELLED_MSG)
            );
            assert_eq!(error.to_string(), chelis_types::EVAL_CANCELLED_MSG);
        }
        revalidate_adt_value(&bounded(1.125), &invariants, &fields, &constants).unwrap();
    }

    fn sp() -> Span {
        Span::new(0, 0)
    }

    fn span_metadata() -> Metadata {
        Metadata::from(chelis_deep::annotations::MetadataValue::Span(
            chelis_deep::annotations::SpanId::try_new("surf:0..1".into(), sp()).unwrap(),
        ))
    }

    fn mentions_span_key(expr: &Expr) -> bool {
        fn metadata_has_span(metadata: &Metadata) -> bool {
            let mut found = metadata.span_id().is_some();
            metadata.visit_syntax(&mut |_, value| found |= mentions_span_key(value));
            found
        }
        match expr {
            Expr::Atom(_, _) => false,
            Expr::Map(map, _) => metadata_has_span(map),
            Expr::MetaExpr(meta, _) => {
                metadata_has_span(&meta.metadata) || mentions_span_key(&meta.expr)
            }
            Expr::List(list, _) => list.elements.iter().any(mentions_span_key),
            Expr::Node(node, _) => {
                metadata_has_span(node.meta())
                    || node.children_slice().iter().any(mentions_span_key)
            }
            Expr::BareList(elements, _) => elements.iter().any(mentions_span_key),
            Expr::UnknownForm(data) => {
                metadata_has_span(&data.meta) || data.children.iter().any(mentions_span_key)
            }
        }
    }

    /// chelis#1087: `strip_span_meta` must reach span metadata inside both
    /// transitional variants, or the rendered invariant text leaks
    /// span-annotated desugar output.
    #[test]
    fn strip_span_meta_reaches_bare_list_and_unknown_form() {
        let mut expr = Expr::BareList(
            vec![
                Expr::Map(span_metadata(), sp()),
                Expr::UnknownForm(Box::new(UnknownFormData {
                    head: "mystery".to_string(),
                    meta: span_metadata(),
                    children: vec![Expr::BareList(vec![Expr::Map(span_metadata(), sp())], sp())],
                    span: sp(),
                })),
            ],
            sp(),
        );
        assert!(mentions_span_key(&expr), "fixture carries span metadata");
        strip_span_meta(&mut expr);
        assert!(
            !mentions_span_key(&expr),
            "span metadata inside transitional variants must be stripped \
             (a span key survived under a BareList or UnknownForm child)"
        );
    }
}
