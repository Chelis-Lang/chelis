//! Producer-obligation collection and synthesis for invariant-carrying
//! opaque types (RFC D-PRODUCER, D-OBLIG, D-PARITY).
//!
//! This module lives in `chelis-prove` so BOTH the CLI prove path and the
//! chelis-tide MCP path consume one implementation (D-PARITY). It takes a
//! desugared Deep program, computes the producer set of each
//! invariant-carrying opaque type from **checker-inferred** return types
//! (D-PRODUCER: an unannotated exported def cannot escape the set), and
//! synthesizes one [`ObligationProperty`] per producer. The CLI / tide
//! layer turns each `ObligationProperty` into a property and runs it
//! through the same engine as user properties.
//!
//! Covered-or-rejected (D-PRODUCER): a producer whose return reaches the
//! opaque type through an unsupported container (record / list / function
//! / non-Option generic), or whose exported signature hands a
//! caller-receives occurrence of the type (the domain of a function-typed
//! parameter), is an ERROR naming the producer and channel — never a
//! silent skip, since one uncovered producer collapses D-SOUND.

use chelis_deep::DeepTag;
use std::collections::{BTreeMap, BTreeSet};

use chelis_deep::ast::{Atom, Expr};
use chelis_types::types::Type;

use crate::opaque::OpaqueInvariant;

/// Where the opaque type sits in a producer's inferred return type, after
/// recursive Option/tuple decomposition (RFC D-PRODUCER).
#[derive(Debug, Clone, PartialEq)]
pub enum ProducedPosition {
    /// The result is the opaque type directly.
    Direct,
    /// The result is `Option[<inner>]`; the inner position decomposes
    /// recursively.
    InsideOption(Box<ProducedPosition>),
    /// The result is a tuple; the listed component indices carry the
    /// opaque type, each at the given inner position. Components without
    /// the type are omitted.
    TupleComponents(Vec<(usize, ProducedPosition)>),
}

/// Obligation metadata threaded into the proof artifact / JSON (RFC
/// D-OBLIG). `obligation_kind` is always `"invariant_producer"` in V1.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ObligationMeta {
    pub obligation_kind: String,
    pub source_type: String,
    pub producer: String,
}

/// A synthesized producer obligation: enough for the CLI / tide layer to
/// build a property `for all (typed inputs) where the producer succeeds,
/// inv(every produced value)` and run it (RFC D-OBLIG).
#[derive(Debug, Clone)]
pub struct ObligationProperty {
    /// `invariant:<Type>:<producer>`.
    pub name: String,
    /// The opaque type whose invariant is being discharged.
    pub source_type: String,
    /// The producer (exported def or constant) under obligation.
    pub producer: String,
    /// `true` for a non-function binding (constant) producer; the
    /// obligation is then over the constant value, with no args.
    pub is_constant: bool,
    /// Where the type sits in the producer's return.
    pub position: ProducedPosition,
    /// Obligation metadata for the artifact / JSON.
    pub meta: ObligationMeta,
}

/// An error in producer-set computation (RFC D-PRODUCER covered-or-rejected
/// / signature rejection). Each names the offending producer/function and,
/// where relevant, the channel, with a user-facing message.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ObligationError {
    #[error(
        "opaque type `{type_name}`: exported producer `{producer}` returns the type through an unsupported container ({container}); decompose-or-reject (RFC D-PRODUCER)"
    )]
    UnsupportedContainer {
        type_name: String,
        producer: String,
        container: String,
    },
    #[error(
        "opaque type `{type_name}`: exported signature `{function}` has a caller-receives occurrence of the type in the {channel}; this would hand caller-supplied code unobligated values (RFC D-PRODUCER signature rejection)"
    )]
    CallerReceives {
        type_name: String,
        function: String,
        channel: String,
    },
    #[error(
        "opaque type `{type_name}`: sig-only def `{producer}` in the defining module returns the type but has no body to prove (RFC D-PRODUCER)"
    )]
    SigOnlyProducer { type_name: String, producer: String },
}

/// The result of collecting obligations for one Deep program: the
/// synthesized obligations plus any covered-or-rejected errors.
#[derive(Debug, Clone, Default)]
pub struct ObligationCollection {
    pub obligations: Vec<ObligationProperty>,
    pub errors: Vec<ObligationError>,
}

// ===========================================================================
// Deep helpers (module / export discovery)
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

/// The set of exported names across all `(export {} ...)` nodes in the
/// program (lexical-module wrappers; the prove path does not go through
/// reef link, so exports survive as lexical `export` decls). A program
/// with no `export` decl exports nothing (zero producers; the sixth
/// rejection makes the type fully sealed — D-PRODUCER).
pub fn collect_exports(exprs: &[Expr]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    collect_exports_in(exprs, &mut out);
    out
}

fn collect_exports_in(exprs: &[Expr], out: &mut BTreeSet<String>) {
    for expr in exprs {
        if tag(expr) == Some(DeepTag::Export) {
            for child in children(expr) {
                if let Some(name) = symbol_text(child) {
                    out.insert(name.to_string());
                }
            }
        }
        collect_exports_in(children(expr), out);
    }
}

/// Names of defs that have a body (`(def {} name (fn ...))` or
/// `(def {} name <value>)`), vs sig-only `(defsig {} name <type>)`.
fn collect_def_bodies(exprs: &[Expr], with_body: &mut BTreeSet<String>) {
    for expr in exprs {
        if tag(expr) == Some(DeepTag::Def)
            && let Some(name) = children(expr).first().and_then(symbol_text)
        {
            with_body.insert(name.to_string());
        }
        collect_def_bodies(children(expr), with_body);
    }
}

/// The DECLARED return type of each `(defsig {} name (t-fn ...))`,
/// parsed from the Deep type node. Used as the authoritative producer
/// position when the checker-inferred return loses the opaque type to an
/// unresolved variable (e.g. a `None`-only body infers `Option[?]` and
/// drops the `-> Option[Probability]` annotation). The producer set is
/// the union of inferred-carrying and declared-carrying returns — the
/// declared annotation is part of the exported contract (RFC D-PRODUCER:
/// inferred types ensure unannotated defs cannot escape; declared types
/// remain authoritative where present).
pub fn collect_declared_returns(exprs: &[Expr]) -> std::collections::BTreeMap<String, Type> {
    let aliases = collect_type_aliases(exprs);
    let mut out = std::collections::BTreeMap::new();
    collect_declared_returns_in(exprs, &aliases, &mut out);
    out
}

fn collect_declared_returns_in(
    exprs: &[Expr],
    aliases: &BTreeMap<String, Expr>,
    out: &mut std::collections::BTreeMap<String, Type>,
) {
    for expr in exprs {
        if tag(expr) == Some(DeepTag::Defsig)
            && let Some(name) = children(expr).first().and_then(symbol_text)
            && let Some(ty_node) = children(expr).get(1)
            && let Some(Type::Fn(_, ret)) = type_from_deep(ty_node, aliases)
        {
            out.insert(name.to_string(), *ret);
        }
        collect_declared_returns_in(children(expr), aliases, out);
    }
}

/// Parse a Deep type node into a [`Type`] for producer-set analysis. The
/// only purpose of the parsed `Type` is name-matching the opaque type, so
/// nodes that CANNOT contain an opaque type are mapped to the inert
/// `Type::Unit` placeholder (CR-14): `t-prim` (a scalar), `t-tensor` (a
/// numeric tensor of prims), `t-var` (a free type variable; an opaque type
/// is nominal, never a variable), and `t-unit`. `t-adt`/`t-tuple`/`t-fn`
/// recurse; `t-ref` (a borrow `&T`) recurses into its inner type so a
/// borrow of (or containing) the opaque type is NOT silently dropped.
///
/// (chelis#731 Phase 2: this placeholder was `Type::Error` before the
/// `ErrorWitness` token made `Type::Error` mintable only inside the checker's
/// diagnostics module. `Type::Unit` is a behavior-preserving substitute here
/// -- this analyzer only ever name-matches ADTs, never pattern-matches or
/// unifies the placeholder, so an inert non-ADT type is all it needs.)
///
/// `aliases` maps each `(typealias ...)` name to its target Deep type
/// node. A `(t-adt {} <name>)` whose name is an alias is resolved through
/// the chain (RT3-F1): the record-field path reads syntactically from
/// Deep, so a field spelled with a type alias (`inner: TA` for
/// `type TA = T`) must resolve to the underlying opaque type, exactly as
/// the checker-inferred signature path already does.
fn type_from_deep(ty: &Expr, aliases: &BTreeMap<String, Expr>) -> Option<Type> {
    type_from_deep_depth(ty, aliases, 0)
}

fn type_from_deep_depth(ty: &Expr, aliases: &BTreeMap<String, Expr>, depth: usize) -> Option<Type> {
    if depth > 32 {
        return Some(Type::Unit);
    }
    match tag(ty)? {
        DeepTag::TPrim => {
            // The producer-set logic only matches ADT names; map prims to
            // a placeholder that never matches an opaque type name.
            Some(Type::Unit)
        }
        DeepTag::TAdt => {
            let kids = children(ty);
            let name = symbol_text(kids.first()?)?.to_string();
            // Resolve a bare (no-arg) alias to its target type. Aliases of
            // generic instantiations (`type TG = Option[T]`) carry their
            // args in the target node, so resolving the whole node is
            // correct; an alias applied with explicit args is not produced
            // by the grammar (V1 aliases are nullary), so we only resolve
            // the no-arg spelling.
            if kids.len() == 1
                && let Some(target) = aliases.get(&name)
            {
                return type_from_deep_depth(target, aliases, depth + 1);
            }
            let args = kids[1..]
                .iter()
                .map(|a| type_from_deep_depth(a, aliases, depth + 1).unwrap_or(Type::Unit))
                .collect();
            Some(Type::Adt(name, args))
        }
        DeepTag::TFn => {
            let kids = children(ty);
            if kids.len() < 2 {
                return None;
            }
            let ret = type_from_deep_depth(kids.last()?, aliases, depth + 1)?;
            let args = kids[..kids.len() - 1]
                .iter()
                .map(|a| type_from_deep_depth(a, aliases, depth + 1).unwrap_or(Type::Unit))
                .collect();
            Some(Type::Fn(args, Box::new(ret)))
        }
        DeepTag::TTuple => {
            let items = children(ty)
                .iter()
                .map(|a| type_from_deep_depth(a, aliases, depth + 1).unwrap_or(Type::Unit))
                .collect();
            Some(Type::Tuple(items))
        }
        DeepTag::TRef => {
            // A borrow `&T`: recurse into the inner type so a borrow of (or
            // containing) the opaque type is reachable by type_contains
            // (CR-14). A record field `inner: &T` was otherwise mapped to
            // the inert placeholder and the T inside hidden.
            let inner = type_from_deep_depth(children(ty).first()?, aliases, depth + 1)?;
            Some(Type::Ref(Box::new(inner)))
        }
        // `t-prim` / `t-tensor` / `t-var` / `t-unit` (and any other leaf):
        // none can contain a nominal opaque type, so the inert Unit
        // placeholder is safe (it never name-matches an opaque type).
        _ => Some(Type::Unit),
    }
}

/// Collect the type-alias map from `(typealias {} <name> (<params>)
/// <target>)` nodes: alias name -> target Deep type node. Alias chains are
/// resolved lazily by `type_from_deep` (which re-enters this map), so we
/// store the raw target node here.
pub fn collect_type_aliases(exprs: &[Expr]) -> BTreeMap<String, Expr> {
    let mut out = BTreeMap::new();
    collect_type_aliases_in(exprs, &mut out);
    out
}

fn collect_type_aliases_in(exprs: &[Expr], out: &mut BTreeMap<String, Expr>) {
    for expr in exprs {
        if tag(expr) == Some(DeepTag::Typealias) {
            let kids = children(expr);
            // children: name, (params), target
            if let Some(name) = kids.first().and_then(symbol_text)
                && let Some(target) = kids.get(2)
            {
                out.insert(name.to_string(), target.clone());
            }
        }
        collect_type_aliases_in(children(expr), out);
    }
}

// ===========================================================================
// Return-type decomposition (over checker-inferred Type)
// ===========================================================================

/// Decompose an inferred return [`Type`] for the opaque type `type_name`:
/// returns the [`ProducedPosition`] if the type is reachable through a
/// supported container (Direct / Option / tuple, recursively), `Ok(None)`
/// if the type does not appear at all, or an `Err(container)` naming the
/// first unsupported container the type is reached through.
///
/// `records` maps each non-opaque record ADT name to its field types, so a
/// producer returning a record that *transitively* wraps the opaque type
/// (`Wrapper { inner: T }`) is covered-or-rejected rather than silently
/// missed (RT-2 CRITICAL; D-PRODUCER "record" container). Without it the
/// `Type::Adt("Wrapper", [])` representation hides the `T` inside.
pub fn decompose_return(
    ty: &Type,
    type_name: &str,
    records: &BTreeMap<String, Vec<Type>>,
) -> Result<Option<ProducedPosition>, String> {
    match ty {
        Type::Adt(name, _) if name == type_name => Ok(Some(ProducedPosition::Direct)),
        Type::Adt(name, args) if name == "Option" => {
            // Option[inner]: decompose the inner.
            let inner = args.first().ok_or_else(|| "Option".to_string())?;
            match decompose_return(inner, type_name, records)? {
                Some(pos) => Ok(Some(ProducedPosition::InsideOption(Box::new(pos)))),
                None => Ok(None),
            }
        }
        Type::Adt(name, _) => {
            // Any other ADT carrying the type -- through its type arguments
            // (a generic like `List[T]`) OR through the fields of a named
            // record (`Wrapper { inner: T }`) -- is an unsupported
            // container (covered-or-rejected). A type NOT carrying the type
            // returns None.
            if type_contains(ty, type_name, records) {
                let kind = if records.contains_key(name) {
                    format!("record `{name}`")
                } else {
                    format!("generic `{name}`")
                };
                Err(kind)
            } else {
                Ok(None)
            }
        }
        Type::Tuple(components) => {
            let mut positions = Vec::new();
            for (idx, comp) in components.iter().enumerate() {
                if let Some(pos) = decompose_return(comp, type_name, records)? {
                    positions.push((idx, pos));
                }
            }
            if positions.is_empty() {
                Ok(None)
            } else {
                Ok(Some(ProducedPosition::TupleComponents(positions)))
            }
        }
        // A bare reference, function, or tensor carrying the type is an
        // unsupported container if it actually contains the type.
        Type::Ref(_) if type_contains(ty, type_name, records) => Err("borrow".to_string()),
        Type::Fn(_, _) if type_contains(ty, type_name, records) => Err("function type".to_string()),
        _ => Ok(None),
    }
}

/// Whether a type structurally mentions the named opaque type anywhere,
/// chasing named record fields through `records` (so `Wrapper { inner: T }`
/// mentions `T`). Recursion is depth-bounded against cyclic record types.
pub fn type_contains(ty: &Type, type_name: &str, records: &BTreeMap<String, Vec<Type>>) -> bool {
    type_contains_depth(ty, type_name, records, 0)
}

fn type_contains_depth(
    ty: &Type,
    type_name: &str,
    records: &BTreeMap<String, Vec<Type>>,
    depth: usize,
) -> bool {
    if depth > 16 {
        return false;
    }
    match ty {
        Type::Adt(name, _) if name == type_name => true,
        Type::Adt(name, args) => {
            args.iter()
                .any(|a| type_contains_depth(a, type_name, records, depth + 1))
                // Chase the named record's field types.
                || records.get(name).is_some_and(|fields| {
                    fields
                        .iter()
                        .any(|f| type_contains_depth(f, type_name, records, depth + 1))
                })
        }
        Type::Tuple(items) => items
            .iter()
            .any(|t| type_contains_depth(t, type_name, records, depth + 1)),
        Type::Fn(args, ret) => {
            args.iter()
                .any(|a| type_contains_depth(a, type_name, records, depth + 1))
                || type_contains_depth(ret, type_name, records, depth + 1)
        }
        Type::Ref(inner) => type_contains_depth(inner, type_name, records, depth + 1),
        _ => false,
    }
}

/// Build the record-field-type registry from the Deep deftypes: each
/// non-opaque record ADT name maps to the field types of its single
/// variant. The opaque types themselves are intentionally NOT entries
/// here (their fields are the representation, walled off by opacity).
pub fn collect_record_fields(exprs: &[Expr]) -> BTreeMap<String, Vec<Type>> {
    let aliases = collect_type_aliases(exprs);
    let mut out = BTreeMap::new();
    collect_record_fields_in(exprs, &aliases, &mut out);
    out
}

fn collect_record_fields_in(
    exprs: &[Expr],
    aliases: &BTreeMap<String, Expr>,
    out: &mut BTreeMap<String, Vec<Type>>,
) {
    for expr in exprs {
        if tag(expr) == Some(DeepTag::Deftype) {
            // Skip opaque types: their fields are the sealed representation.
            let opaque = matches!(
                meta_value(expr, "opaque"),
                Some(Expr::Atom(Atom::Bool(true), _))
            );
            if !opaque && let Some(name) = children(expr).first().and_then(symbol_text) {
                // Collect EVERY variant's payload types, not just the first
                // (CR-15): a non-first variant wrapping the opaque type
                // would otherwise be invisible to type_contains. A variant
                // payload is either a named-record `field` node or a
                // positional type child.
                let mut field_types = Vec::new();
                for variant in children(expr)
                    .iter()
                    .filter(|c| tag(c) == Some(DeepTag::Variant))
                {
                    for payload in children(variant).iter().skip(1) {
                        let fty_node = if tag(payload) == Some(DeepTag::Field) {
                            children(payload).get(1)
                        } else {
                            // A positional payload: the child IS the type
                            // node (`(variant {} Some (t-adt {} T))`).
                            Some(payload)
                        };
                        if let Some(node) = fty_node
                            && let Some(fty) = type_from_deep(node, aliases)
                        {
                            field_types.push(fty);
                        }
                    }
                }
                out.insert(name.to_string(), field_types);
            }
        }
        collect_record_fields_in(children(expr), aliases, out);
    }
}

fn meta_value<'a>(expr: &'a Expr, key: &str) -> Option<&'a Expr> {
    let meta = match expr {
        Expr::Node(node, _) => node.meta(),
        Expr::List(list, _) => match list.elements.get(1) {
            Some(Expr::Map(map, _)) => map,
            _ => return None,
        },
        _ => return None,
    };
    meta.entries
        .iter()
        .find_map(|(k, v)| (k == key).then_some(v))
}

/// Whether a parameter type hands the opaque type to caller-supplied code
/// (RFC D-PRODUCER signature rejection): the DOMAIN of a function-typed
/// parameter, transitively. A plain type-T parameter, a record parameter
/// with a T field, or a caller-implemented function whose RETURN contains
/// T are module-receives positions and stay legal.
fn caller_receives_in_param(
    ty: &Type,
    type_name: &str,
    records: &BTreeMap<String, Vec<Type>>,
) -> bool {
    match ty {
        // A function-typed parameter: its argument domain is
        // caller-receives (the module calls back into caller code with a
        // value of the type). Its return is module-receives (legal).
        Type::Fn(args, ret) => {
            args.iter().any(|a| type_contains(a, type_name, records))
                || caller_receives_in_param(ret, type_name, records)
                || args
                    .iter()
                    .any(|a| caller_receives_in_param(a, type_name, records))
        }
        // Containers: recurse, but a bare T or T-in-record is NOT
        // caller-receives (module-receives, legal).
        Type::Tuple(items) => items
            .iter()
            .any(|t| caller_receives_in_param(t, type_name, records)),
        Type::Adt(_, args) => args
            .iter()
            .any(|a| caller_receives_in_param(a, type_name, records)),
        Type::Ref(inner) => caller_receives_in_param(inner, type_name, records),
        _ => false,
    }
}

// ===========================================================================
// Top-level collection
// ===========================================================================

/// Collect and synthesize producer obligations for every
/// invariant-carrying opaque type in a desugared Deep program.
///
/// `sigs` maps def-name -> the checker-inferred function/value type
/// (`FunctionSignatureInference::checked_signature`). The caller obtains
/// it from `chelis_types::check_typed_program(exprs)`.
pub fn collect_obligations(
    exprs: &[Expr],
    invariants: &[OpaqueInvariant],
    sigs: &std::collections::BTreeMap<String, Type>,
) -> ObligationCollection {
    let exports = collect_exports(exprs);
    let mut def_bodies = BTreeSet::new();
    collect_def_bodies(exprs, &mut def_bodies);
    let declared = collect_declared_returns(exprs);
    let records = collect_record_fields(exprs);

    let mut col = ObligationCollection::default();
    for inv in invariants {
        for name in &exports {
            let Some(ty) = sigs.get(name) else {
                // Exported name with no inferred signature: could be a
                // type/constructor export, not a producer. Skip — only
                // value/function defs carry obligations.
                continue;
            };
            collect_one(
                inv,
                name,
                ty,
                declared.get(name),
                &def_bodies,
                &records,
                &mut col,
            );
        }
    }
    col
}

#[allow(clippy::too_many_arguments)]
fn collect_one(
    inv: &OpaqueInvariant,
    name: &str,
    ty: &Type,
    declared_ret: Option<&Type>,
    def_bodies: &BTreeSet<String>,
    records: &BTreeMap<String, Vec<Type>>,
    col: &mut ObligationCollection,
) {
    let type_name = &inv.type_name;
    match ty {
        Type::Fn(params, ret) => {
            // Signature rejection: any function-typed parameter whose
            // domain receives the type (caller-receives) is a declaration
            // error (D-PRODUCER signature rejection).
            for (idx, p) in params.iter().enumerate() {
                if caller_receives_in_param(p, type_name, records) {
                    col.errors.push(ObligationError::CallerReceives {
                        type_name: type_name.clone(),
                        function: name.to_string(),
                        channel: format!("domain of parameter {idx}"),
                    });
                    return;
                }
            }
            // Produced positions: decompose-or-reject the return. Prefer
            // the inferred return; if it does not carry the type but the
            // DECLARED return does (e.g. a None-only body infers
            // Option[?] and drops the -> Option[T] annotation), use the
            // declared return — it is part of the exported contract.
            let inferred = decompose_return(ret, type_name, records);
            let resolved = match &inferred {
                Ok(None) => match declared_ret {
                    Some(d) => decompose_return(d, type_name, records),
                    None => inferred,
                },
                _ => inferred,
            };
            match resolved {
                Ok(Some(position)) => {
                    if !def_bodies.contains(name) {
                        col.errors.push(ObligationError::SigOnlyProducer {
                            type_name: type_name.clone(),
                            producer: name.to_string(),
                        });
                        return;
                    }
                    col.obligations
                        .push(make_obligation(type_name, name, position, false));
                }
                Ok(None) => {}
                Err(container) => col.errors.push(ObligationError::UnsupportedContainer {
                    type_name: type_name.clone(),
                    producer: name.to_string(),
                    container,
                }),
            }
        }
        // A non-function exported binding whose type contains the type is
        // a constant producer (RFC D-PRODUCER L4): the obligation is that
        // the invariant holds of the value.
        other => match decompose_return(other, type_name, records) {
            Ok(Some(position)) => {
                col.obligations
                    .push(make_obligation(type_name, name, position, true));
            }
            Ok(None) => {}
            Err(container) => col.errors.push(ObligationError::UnsupportedContainer {
                type_name: type_name.clone(),
                producer: name.to_string(),
                container,
            }),
        },
    }
}

fn make_obligation(
    type_name: &str,
    producer: &str,
    position: ProducedPosition,
    is_constant: bool,
) -> ObligationProperty {
    ObligationProperty {
        name: format!("invariant:{type_name}:{producer}"),
        source_type: type_name.to_string(),
        producer: producer.to_string(),
        is_constant,
        position,
        meta: ObligationMeta {
            obligation_kind: "invariant_producer".to_string(),
            source_type: type_name.to_string(),
            producer: producer.to_string(),
        },
    }
}

#[cfg(test)]
mod tests;
