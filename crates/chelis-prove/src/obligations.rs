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

mod traversal;
pub use traversal::TraversalError;
use traversal::{Budget, DecomposeError, ProofTypes, Source};

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
        "opaque type `{type_name}`: could not analyze exported producer `{producer}`: {source}"
    )]
    TypeTraversal {
        type_name: String,
        producer: String,
        source: TraversalError,
    },
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
/// `(def {} name <value>)`), vs sig-only
/// `(defsig {} name [<binders>] <type>)`.
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

/// Raw proof-type declarations retain alias and record identities. The proof
/// walker projects reachable nodes into one finite graph; it never expands an
/// alias recursively into an invented semantic type.
#[derive(Default)]
struct ProofDeclarations<'a> {
    aliases: BTreeMap<&'a str, &'a Expr>,
    records: BTreeMap<&'a str, Vec<Source<'a>>>,
    signatures: BTreeMap<&'a str, &'a Expr>,
}

fn proof_declarations(exprs: &[Expr]) -> ProofDeclarations<'_> {
    let mut result = ProofDeclarations::default();
    let mut work: Vec<_> = exprs.iter().collect();
    while let Some(expr) = work.pop() {
        let kids = children(expr);
        if let Some(name) = kids.first().and_then(symbol_text) {
            match tag(expr) {
                Some(DeepTag::Typealias) => {
                    if let Some(target) = kids.get(2) {
                        result.aliases.insert(name, target);
                    }
                }
                Some(DeepTag::Defsig) => {
                    if let Some(ty) = kids.last() {
                        result.signatures.insert(name, ty);
                    }
                }
                Some(DeepTag::Deftype) if !is_opaque(expr) => {
                    let mut fields = Vec::new();
                    for variant in kids
                        .iter()
                        .filter(|node| tag(node) == Some(DeepTag::Variant))
                    {
                        for payload in children(variant).iter().skip(1) {
                            let ty = if tag(payload) == Some(DeepTag::Field) {
                                children(payload).get(1)
                            } else {
                                Some(payload)
                            };
                            if let Some(ty) = ty {
                                fields.push(Source::Deep(ty));
                            }
                        }
                    }
                    result.records.insert(name, fields);
                }
                _ => {}
            }
        }
        work.extend(kids.iter().rev());
    }
    result
}

/// Decompose a checked return through the same total proof-type graph used
/// by collection. Unsupported containers and resource exhaustion are errors.
pub fn decompose_return(
    ty: &Type,
    type_name: &str,
    records: &BTreeMap<String, Vec<Type>>,
) -> Result<Option<ProducedPosition>, String> {
    let fields = records
        .iter()
        .map(|(name, fields)| (name.as_str(), fields.iter().map(Source::Checked).collect()))
        .collect();
    let mut graph = ProofTypes::new(BTreeMap::new(), fields, Budget::default());
    let root = graph
        .project(Source::Checked(ty))
        .map_err(|error| error.to_string())?;
    graph
        .decompose(root, type_name)
        .map_err(|error| match error {
            DecomposeError::Container(container) => container,
            DecomposeError::Traversal(error) => error.to_string(),
        })
}

fn is_opaque(expr: &Expr) -> bool {
    let meta = match expr {
        Expr::Node(node, _) => node.meta(),
        Expr::List(list, _) => match list.elements.get(1) {
            Some(Expr::Map(map, _)) => map,
            _ => return false,
        },
        _ => return false,
    };
    meta.opaque().is_some()
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
    sigs: &BTreeMap<String, Type>,
) -> ObligationCollection {
    collect_with_budget(exprs, invariants, sigs, traversal::DEFAULT_BUDGET)
}

fn collect_with_budget(
    exprs: &[Expr],
    invariants: &[OpaqueInvariant],
    sigs: &BTreeMap<String, Type>,
    limit: usize,
) -> ObligationCollection {
    let exports = collect_exports(exprs);
    let mut def_bodies = BTreeSet::new();
    collect_def_bodies(exprs, &mut def_bodies);
    let declarations = proof_declarations(exprs);
    let mut col = ObligationCollection::default();
    for inv in invariants {
        for name in &exports {
            let Some(ty) = sigs.get(name) else {
                continue;
            };
            // Each exported signature gets an independent budget: an unrelated
            // sibling cannot consume its allowance or erase its obligation.
            let mut graph = ProofTypes::new(
                declarations.aliases.clone(),
                declarations.records.clone(),
                Budget::new(limit),
            );
            let result = collect_one(
                inv,
                name,
                ty,
                declarations.signatures.get(name.as_str()).copied(),
                def_bodies.contains(name),
                &mut graph,
                &mut col,
            );
            if let Err(source) = result {
                col.errors.push(ObligationError::TypeTraversal {
                    type_name: inv.type_name.clone(),
                    producer: name.clone(),
                    source,
                });
            }
        }
    }
    col
}

fn collect_one<'a>(
    inv: &OpaqueInvariant,
    name: &str,
    ty: &'a Type,
    declared: Option<&'a Expr>,
    has_body: bool,
    graph: &mut ProofTypes<'a>,
    col: &mut ObligationCollection,
) -> Result<(), TraversalError> {
    let type_name = &inv.type_name;
    let root = graph.project(Source::Checked(ty))?;
    let function = graph.function(root)?;
    let is_constant = function.is_none();
    let result_root = if let Some((params, ret)) = function {
        for (idx, param) in params.iter().enumerate() {
            if graph.caller_receives(*param, type_name)? {
                col.errors.push(ObligationError::CallerReceives {
                    type_name: type_name.clone(),
                    function: name.to_string(),
                    channel: format!("domain of parameter {idx}"),
                });
                return Ok(());
            }
        }
        ret
    } else {
        root
    };
    let mut resolved = graph.decompose(result_root, type_name);
    // Declared returns remain authoritative when inference lost a nominal
    // occurrence (for example a None-only Option producer).
    if matches!(resolved, Ok(None))
        && !is_constant
        && let Some(declared) = declared
    {
        let declared = graph.project(Source::Deep(declared))?;
        if let Some((_, ret)) = graph.function(declared)? {
            resolved = graph.decompose(ret, type_name);
        }
    }
    match resolved {
        Ok(Some(_)) if !is_constant && !has_body => {
            col.errors.push(ObligationError::SigOnlyProducer {
                type_name: type_name.clone(),
                producer: name.to_string(),
            })
        }
        Ok(Some(position)) => {
            col.obligations
                .push(make_obligation(type_name, name, position, is_constant))
        }
        Ok(None) => {}
        Err(DecomposeError::Container(container)) => {
            col.errors.push(ObligationError::UnsupportedContainer {
                type_name: type_name.clone(),
                producer: name.to_string(),
                container,
            })
        }
        Err(DecomposeError::Traversal(error)) => return Err(error),
    }
    Ok(())
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
