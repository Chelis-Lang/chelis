//! Typed Deep-to-Surf AST resugaring.
//!
//! This is the structural boundary shared by Deep emitters and the canonical
//! Surf printer. It covers every public Deep declaration and expression tag;
//! contextual tags remain explicit errors when requested in the wrong Surf
//! AST position. This module never substitutes `()` or a comment for source
//! code.

use chelis_deep::ast::{Atom, Expr as DeepExpr, MetaMap};
use chelis_deep::{DeepTag, LiteralSuffix, Span};
use chelis_vocab::{EffectKind, EffectKindInput};
use std::collections::HashMap;
use thiserror::Error;

use crate::ast::{
    BinOp, Decl, EffectExpr, Expr, ImportKind, LetBinding, LetPattern, Literal, MatchArm, Param,
    Pattern, PropertyOption, TypeExpr, TypeInvariant, UnaryOp, Variant, VariantFields,
};

/// Failure to structurally resugar a Deep expression.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ResugarError {
    #[error("expected a canonical Deep node, found {found}")]
    ExpectedNode { found: String },

    #[error("Deep `{tag}` expects exactly {expected} children, found {actual}")]
    ExactArity {
        tag: &'static str,
        expected: usize,
        actual: usize,
    },

    #[error("Deep `{tag}` expects at least {minimum} child{plural}, found {actual}", plural = if *minimum == 1 { "" } else { "ren" })]
    MinimumArity {
        tag: &'static str,
        minimum: usize,
        actual: usize,
    },

    #[error("Deep `{tag}` child {index} must be {expected}")]
    InvalidChild {
        tag: &'static str,
        index: usize,
        expected: &'static str,
    },

    #[error("Deep `{tag}` does not have a Surf representation in {position} position")]
    InvalidSurfacePosition {
        tag: &'static str,
        position: &'static str,
    },

    #[error("unknown key `{key}` in the closed Deep `surf_*` metadata namespace")]
    UnknownSurfaceMetadata { key: String },

    #[error("Deep decompiler emitted invalid Surf: {reason}")]
    InvalidSurfaceProgram { reason: String },

    #[error("Deep contains a non-finite float literal, which has no canonical Surf literal")]
    NonFiniteFloat,
}

#[derive(Clone, Copy)]
struct NodeRef<'a> {
    tag: DeepTag,
    meta: &'a MetaMap,
    children: &'a [DeepExpr],
    span: Span,
}

/// Resugar one Deep expression to the shared Surf AST.
///
/// The caller prints the result with [`crate::format::format_expression`].
/// Keeping construction and rendering separate makes it impossible for this
/// path to invent a second spelling for an AST construct.
pub fn resugar_expression(expr: &DeepExpr) -> Result<Expr, ResugarError> {
    let node = node_ref(expr)?;
    let annotation = (node.tag != DeepTag::Lit)
        .then(|| meta_value(node.meta, "type").map(resugar_type))
        .flatten()
        .transpose()?;
    let span = node.span;
    let expression = resugar_node(node)?;
    Ok(match annotation {
        Some(ty) => Expr::Annotate(Box::new(expression), ty, span),
        None => expression,
    })
}

/// Resugar a public Deep program into the shared canonical Surf AST.
///
/// Adjacent matching `defsig`/`def` pairs are deliberately folded into one
/// typed Surf declaration. Derived Deep metadata is ignored; the narrow
/// validated `surf_*` metadata namespace is consulted only where Deep has
/// erased a canonical source distinction such as module-path casing or a
/// grouped `dim` declaration.
pub fn resugar_program(exprs: &[DeepExpr]) -> Result<Vec<Decl>, ResugarError> {
    resugar_declaration_sequence(exprs)
}

/// Remove Deep metadata that is explicitly derived from source location and
/// therefore outside the structural Surf/Deep round-trip law.
///
/// Semantic metadata (`type`, `eff`, `wrt`, property contracts, and the
/// validated `surf_*` namespace) is retained, except for exact `type` entries
/// on a `def`, its function value, and its parameters when the immediately
/// preceding matching `defsig` already carries the same types. Canonical Surf
/// deliberately folds that Deep pair into one typed declaration, so
/// desugaring necessarily recreates those redundant annotations. A
/// disagreement is retained and therefore still fails the oracle.
///
/// This narrow normalization is intentionally not
/// [`chelis_deep::ast::strip_metadata`], which would erase language-relevant
/// information and could make a false round trip look green.
pub fn normalize_deep_for_surface_roundtrip(exprs: &[DeepExpr]) -> Vec<DeepExpr> {
    let normalized = exprs.iter().map(normalize_roundtrip_expr).collect();
    normalize_declaration_sequence_roundtrip(normalized)
}

fn normalize_declaration_sequence_roundtrip(exprs: Vec<DeepExpr>) -> Vec<DeepExpr> {
    let exprs = exprs
        .into_iter()
        .map(normalize_nested_declaration_sequence)
        .collect::<Vec<_>>();
    let mut exprs = materialize_standalone_definition_signatures(exprs);

    for index in 0..exprs.len().saturating_sub(1) {
        let signature = exprs[index].clone();
        let definition = exprs[index + 1].clone();
        if let Some(normalized) = strip_redundant_definition_types(&signature, &definition) {
            exprs[index + 1] = normalized;
        }
    }
    exprs
}

fn materialize_standalone_definition_signatures(exprs: Vec<DeepExpr>) -> Vec<DeepExpr> {
    let mut normalized = Vec::with_capacity(exprs.len());
    for definition_expr in exprs {
        let Ok(definition) = node_ref(&definition_expr) else {
            normalized.push(definition_expr);
            continue;
        };
        let Some(definition_type) = (definition.tag == DeepTag::Def)
            .then(|| meta_value(definition.meta, "type"))
            .flatten()
        else {
            normalized.push(definition_expr);
            continue;
        };
        let Some(name) = definition.children.first().and_then(atom_name) else {
            normalized.push(definition_expr);
            continue;
        };
        let already_paired = normalized.last().is_some_and(|candidate| {
            node_ref(candidate).is_ok_and(|signature| {
                signature.tag == DeepTag::Defsig
                    && signature.children.first().and_then(atom_name) == Some(name)
            })
        });
        if already_paired {
            normalized.push(definition_expr);
            continue;
        }

        let signature = DeepExpr::Node(
            Box::new(chelis_deep::node::Node::new(
                DeepTag::Defsig,
                MetaMap::default(),
                vec![definition.children[0].clone(), definition_type.clone()],
            )),
            definition.span,
        );
        let definition = strip_redundant_definition_types(&signature, &definition_expr)
            .unwrap_or(definition_expr);
        normalized.push(signature);
        normalized.push(definition);
    }
    normalized
}

fn normalize_nested_declaration_sequence(expr: DeepExpr) -> DeepExpr {
    let Ok(module) = node_ref(&expr) else {
        return expr;
    };
    if module.tag != DeepTag::Module || module.children.is_empty() {
        return expr;
    }

    let mut children = vec![module.children[0].clone()];
    children.extend(normalize_declaration_sequence_roundtrip(
        module.children[1..].to_vec(),
    ));
    rebuild_node_like(
        &expr,
        module.tag,
        module.meta.clone(),
        children,
        module.span,
    )
}

fn strip_redundant_definition_types(
    signature_expr: &DeepExpr,
    definition_expr: &DeepExpr,
) -> Option<DeepExpr> {
    let signature = node_ref(signature_expr).ok()?;
    let definition = node_ref(definition_expr).ok()?;
    if signature.tag != DeepTag::Defsig
        || definition.tag != DeepTag::Def
        || signature.children.len() != 2
        || definition.children.len() != 2
        || atom_name(&signature.children[0]) != atom_name(&definition.children[0])
    {
        return None;
    }

    let signature_type = &signature.children[1];
    let (definition_meta, mut changed) =
        strip_matching_type_metadata(definition.meta, signature_type);

    let Ok(function_type) = node_ref(signature_type) else {
        return changed.then(|| {
            rebuild_node_like(
                definition_expr,
                definition.tag,
                definition_meta,
                definition.children.to_vec(),
                definition.span,
            )
        });
    };
    let Ok(function) = node_ref(&definition.children[1]) else {
        return changed.then(|| {
            rebuild_node_like(
                definition_expr,
                definition.tag,
                definition_meta,
                definition.children.to_vec(),
                definition.span,
            )
        });
    };
    if function_type.tag != DeepTag::TFn
        || function_type.children.is_empty()
        || function.tag != DeepTag::Fn
        || function.children.len() != 2
    {
        return changed.then(|| {
            rebuild_node_like(
                definition_expr,
                definition.tag,
                definition_meta,
                definition.children.to_vec(),
                definition.span,
            )
        });
    }
    let (function_meta, function_changed) =
        strip_matching_type_metadata(function.meta, signature_type);
    changed |= function_changed;
    let params = node_ref(&function.children[0]).ok()?;
    if params.tag != DeepTag::Params || params.children.len() + 1 != function_type.children.len() {
        return None;
    }

    let normalized_params = params
        .children
        .iter()
        .zip(&function_type.children[..params.children.len()])
        .map(|(param, expected_type)| {
            let normalized = strip_matching_param_type(param, expected_type);
            changed |= normalized != *param;
            normalized
        })
        .collect::<Vec<_>>();
    if !changed {
        return None;
    }

    let params_expr = rebuild_node_like(
        &function.children[0],
        params.tag,
        params.meta.clone(),
        normalized_params,
        params.span,
    );
    let function_expr = rebuild_node_like(
        &definition.children[1],
        function.tag,
        function_meta,
        vec![params_expr, function.children[1].clone()],
        function.span,
    );
    Some(rebuild_node_like(
        definition_expr,
        definition.tag,
        definition_meta,
        vec![definition.children[0].clone(), function_expr],
        definition.span,
    ))
}

fn strip_matching_type_metadata(meta: &MetaMap, expected_type: &DeepExpr) -> (MetaMap, bool) {
    let Some(actual_type) = meta_value(meta, "type") else {
        return (meta.clone(), false);
    };
    if !same_deep_shape(actual_type, expected_type) {
        return (meta.clone(), false);
    }
    (
        MetaMap {
            entries: meta
                .entries
                .iter()
                .filter(|(key, _)| key != "type")
                .cloned()
                .collect(),
        },
        true,
    )
}

fn strip_matching_param_type(param: &DeepExpr, expected_type: &DeepExpr) -> DeepExpr {
    match param {
        DeepExpr::MetaExpr(meta, span) => {
            let Some((_, actual_type)) = meta.entries.iter().find(|(key, _)| key == "type") else {
                return param.clone();
            };
            if !same_deep_shape(actual_type, expected_type) {
                return param.clone();
            }
            let entries = meta
                .entries
                .iter()
                .filter(|(key, _)| key != "type")
                .cloned()
                .collect::<Vec<_>>();
            if entries.is_empty() {
                (*meta.expr).clone()
            } else {
                DeepExpr::MetaExpr(
                    chelis_deep::ast::MetaExpr {
                        entries,
                        expr: meta.expr.clone(),
                    },
                    *span,
                )
            }
        }
        DeepExpr::List(list, span) if list.tag().is_none() => {
            strip_matching_param_type_items(&list.elements, *span, expected_type, false)
                .unwrap_or_else(|| param.clone())
        }
        DeepExpr::BareList(items, span) => {
            strip_matching_param_type_items(items, *span, expected_type, true)
                .unwrap_or_else(|| param.clone())
        }
        _ => param.clone(),
    }
}

fn strip_matching_param_type_items(
    items: &[DeepExpr],
    span: Span,
    expected_type: &DeepExpr,
    bare: bool,
) -> Option<DeepExpr> {
    let [name, DeepExpr::Map(meta, map_span)] = items else {
        return None;
    };
    let actual_type = meta_value(meta, "type")?;
    if !same_deep_shape(actual_type, expected_type) {
        return None;
    }
    let entries = meta
        .entries
        .iter()
        .filter(|(key, _)| key != "type")
        .cloned()
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return Some(name.clone());
    }
    let items = vec![name.clone(), DeepExpr::Map(MetaMap { entries }, *map_span)];
    Some(if bare {
        DeepExpr::BareList(items, span)
    } else {
        DeepExpr::List(chelis_deep::ast::List { elements: items }, span)
    })
}

fn same_deep_shape(left: &DeepExpr, right: &DeepExpr) -> bool {
    chelis_deep::printer::print_canonical(std::slice::from_ref(left))
        == chelis_deep::printer::print_canonical(std::slice::from_ref(right))
}

fn rebuild_node_like(
    original: &DeepExpr,
    tag: DeepTag,
    meta: MetaMap,
    children: Vec<DeepExpr>,
    span: Span,
) -> DeepExpr {
    match original {
        DeepExpr::Node(..) => DeepExpr::Node(
            Box::new(chelis_deep::node::Node::new(tag, meta, children)),
            span,
        ),
        _ => DeepExpr::node(tag, meta, children, span),
    }
}

fn normalize_roundtrip_expr(expr: &DeepExpr) -> DeepExpr {
    if let Ok(node) = node_ref(expr) {
        if node.tag == DeepTag::Lit
            && node.children.len() == 1
            && let Some(value) = node.children.first()
        {
            let suffix = literal_suffix(node.meta).ok().flatten();
            if let DeepExpr::Atom(Atom::Int(value), span) = value {
                if suffix.is_some_and(|suffix| suffix.is_float()) {
                    let converted = DeepExpr::Node(
                        Box::new(chelis_deep::node::Node::new(
                            DeepTag::Lit,
                            normalize_roundtrip_meta(node.meta),
                            vec![DeepExpr::Atom(Atom::Float(*value as f64), *span)],
                        )),
                        node.span,
                    );
                    return normalize_roundtrip_expr(&converted);
                }
                if *value < 0 {
                    return normalize_negative_integer_literal(node, *value, suffix);
                }
            }
            if let DeepExpr::Atom(Atom::Float(value), span) = value
                && value.is_sign_negative()
            {
                return normalized_unary_neg_literal(
                    node,
                    DeepExpr::Atom(Atom::Float(-*value), *span),
                );
            }
        }
        if node.tag == DeepTag::Tuple && node.children.is_empty() {
            let unit_type = DeepExpr::Node(
                Box::new(chelis_deep::node::Node::new(
                    DeepTag::TUnit,
                    MetaMap::default(),
                    Vec::new(),
                )),
                node.span,
            );
            let mut meta = normalize_roundtrip_meta(node.meta);
            meta.entries.retain(|(key, _)| key != "type");
            meta.entries.push(("type".to_string(), unit_type));
            return DeepExpr::Node(
                Box::new(chelis_deep::node::Node::new(
                    DeepTag::Lit,
                    meta,
                    vec![DeepExpr::BareList(Vec::new(), node.span)],
                )),
                node.span,
            );
        }
        if node.tag == DeepTag::TTuple && node.children.is_empty() {
            return DeepExpr::Node(
                Box::new(chelis_deep::node::Node::new(
                    DeepTag::TUnit,
                    normalize_roundtrip_meta(node.meta),
                    Vec::new(),
                )),
                node.span,
            );
        }
        if node.tag == DeepTag::App
            && node.children.len() == 1
            && variable_name(&node.children[0]).is_some_and(starts_uppercase)
        {
            let name = variable_name(&node.children[0]).expect("checked above");
            return DeepExpr::Node(
                Box::new(chelis_deep::node::Node::new(
                    DeepTag::Var,
                    normalize_roundtrip_meta(node.meta),
                    vec![DeepExpr::Atom(Atom::Name(name.to_string()), node.span)],
                )),
                node.span,
            );
        }
    }
    match expr {
        DeepExpr::Atom(..) => expr.clone(),
        DeepExpr::Node(node, span) => DeepExpr::Node(
            Box::new(chelis_deep::node::Node::new(
                node.tag(),
                normalize_roundtrip_meta(node.meta()),
                node.children_slice()
                    .iter()
                    .map(normalize_roundtrip_expr)
                    .collect(),
            )),
            *span,
        ),
        DeepExpr::List(list, span) => DeepExpr::List(
            chelis_deep::ast::List {
                elements: list.elements.iter().map(normalize_roundtrip_expr).collect(),
            },
            *span,
        ),
        DeepExpr::Map(meta, span) => DeepExpr::Map(normalize_roundtrip_meta(meta), *span),
        DeepExpr::MetaExpr(meta, span) => DeepExpr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                entries: meta
                    .entries
                    .iter()
                    .filter(|(key, _)| !is_roundtrip_derived_key(key))
                    .map(|(key, value)| (key.clone(), normalize_roundtrip_expr(value)))
                    .collect(),
                expr: Box::new(normalize_roundtrip_expr(&meta.expr)),
            },
            *span,
        ),
        DeepExpr::BareList(items, span) => {
            DeepExpr::BareList(items.iter().map(normalize_roundtrip_expr).collect(), *span)
        }
        DeepExpr::UnknownForm(data) => {
            DeepExpr::UnknownForm(Box::new(chelis_deep::ast::UnknownFormData {
                head: data.head.clone(),
                meta: normalize_roundtrip_meta(&data.meta),
                children: data.children.iter().map(normalize_roundtrip_expr).collect(),
                span: data.span,
            }))
        }
    }
}

fn normalize_negative_integer_literal(
    node: NodeRef<'_>,
    value: i64,
    suffix: Option<LiteralSuffix>,
) -> DeepExpr {
    let value_span = match &node.children[0] {
        DeepExpr::Atom(_, span) => *span,
        _ => node.span,
    };
    if integer_minimum(suffix) == Some(value) {
        let maximum = -(value + 1);
        let negative_maximum =
            normalized_unary_neg_literal(node, DeepExpr::Atom(Atom::Int(maximum), value_span));
        let one = normalized_literal_like(node, DeepExpr::Atom(Atom::Int(1), value_span));
        return normalized_application("sub", vec![negative_maximum, one], node.span);
    }
    normalized_unary_neg_literal(node, DeepExpr::Atom(Atom::Int(-value), value_span))
}

fn normalized_literal_like(node: NodeRef<'_>, value: DeepExpr) -> DeepExpr {
    DeepExpr::Node(
        Box::new(chelis_deep::node::Node::new(
            DeepTag::Lit,
            normalize_roundtrip_meta(node.meta),
            vec![value],
        )),
        node.span,
    )
}

fn normalized_unary_neg_literal(node: NodeRef<'_>, positive: DeepExpr) -> DeepExpr {
    normalized_application(
        "neg",
        vec![normalized_literal_like(node, positive)],
        node.span,
    )
}

fn normalized_application(name: &str, arguments: Vec<DeepExpr>, span: Span) -> DeepExpr {
    let function = DeepExpr::Node(
        Box::new(chelis_deep::node::Node::new(
            DeepTag::Var,
            MetaMap::default(),
            vec![DeepExpr::Atom(Atom::Name(name.to_string()), span)],
        )),
        span,
    );
    let mut children = Vec::with_capacity(arguments.len() + 1);
    children.push(function);
    children.extend(arguments);
    DeepExpr::Node(
        Box::new(chelis_deep::node::Node::new(
            DeepTag::App,
            MetaMap::default(),
            children,
        )),
        span,
    )
}

fn normalize_roundtrip_meta(meta: &MetaMap) -> MetaMap {
    MetaMap {
        entries: meta
            .entries
            .iter()
            .filter(|(key, _)| !is_roundtrip_derived_key(key))
            .map(|(key, value)| (key.clone(), normalize_roundtrip_expr(value)))
            .collect(),
    }
}

fn is_roundtrip_derived_key(key: &str) -> bool {
    matches!(
        key,
        "span" | "loc" | "source" | "effects" | "invariant_amenability"
    )
}

fn resugar_declaration_sequence(exprs: &[DeepExpr]) -> Result<Vec<Decl>, ResugarError> {
    let mut declarations = Vec::new();
    let mut index = 0;
    while index < exprs.len() {
        let node = node_ref(&exprs[index])?;
        match node.tag {
            DeepTag::Defsig => {
                exact(&node, 2)?;
                let name = name_child(&node, 0)?.to_string();
                if let Some(next) = exprs.get(index + 1)
                    && let Ok(definition) = node_ref(next)
                    && definition.tag == DeepTag::Def
                    && definition.children.first().and_then(atom_name) == Some(name.as_str())
                {
                    declarations.push(resugar_definition(Some(&node), definition)?);
                    index += 2;
                    continue;
                }
                declarations.push(Decl::Sig {
                    name,
                    ty: resugar_type(&node.children[1])?,
                    effects: resugar_effect_metadata(&node.children[1])?,
                    span: node.span,
                });
                index += 1;
            }
            DeepTag::Def => {
                declarations.push(resugar_definition(None, node)?);
                index += 1;
            }
            DeepTag::Module => {
                at_least(&node, 1)?;
                declarations.push(Decl::Module {
                    name: resugar_surface_path(&node, 0)?,
                    decls: resugar_declaration_sequence(&node.children[1..])?,
                    span: node.span,
                });
                index += 1;
            }
            DeepTag::Import | DeepTag::ImportAll => {
                declarations.push(resugar_import(node)?);
                index += 1;
            }
            DeepTag::Export => {
                declarations.push(Decl::Export {
                    names: node
                        .children
                        .iter()
                        .enumerate()
                        .map(|(child_index, child)| {
                            atom_name(child)
                                .map(str::to_string)
                                .ok_or(ResugarError::InvalidChild {
                                    tag: node.tag.as_str(),
                                    index: child_index,
                                    expected: "a name atom",
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    span: node.span,
                });
                index += 1;
            }
            DeepTag::Defdim => {
                exact(&node, 1)?;
                let group_size = surf_group_size(&node)?;
                let mut names = vec![name_child(&node, 0)?.to_string()];
                for offset in 1..group_size {
                    let member = exprs
                        .get(index + offset)
                        .ok_or(ResugarError::InvalidChild {
                            tag: node.tag.as_str(),
                            index: offset,
                            expected: "the remaining adjacent `defdim` group members",
                        })?;
                    let member = node_ref(member)?;
                    if member.tag != DeepTag::Defdim {
                        return Err(ResugarError::InvalidChild {
                            tag: node.tag.as_str(),
                            index: offset,
                            expected: "an adjacent `(defdim ...)` group member",
                        });
                    }
                    exact(&member, 1)?;
                    names.push(name_child(&member, 0)?.to_string());
                }
                declarations.push(Decl::Dim {
                    names,
                    span: node.span,
                });
                index += group_size;
            }
            DeepTag::Deftype => {
                declarations.push(resugar_type_definition(node)?);
                index += 1;
            }
            DeepTag::Typealias => {
                declarations.push(resugar_type_alias(node)?);
                index += 1;
            }
            _ => {
                return Err(ResugarError::InvalidSurfacePosition {
                    tag: node.tag.as_str(),
                    position: "declaration",
                });
            }
        }
    }
    Ok(declarations)
}

fn resugar_definition(
    signature: Option<&NodeRef<'_>>,
    definition: NodeRef<'_>,
) -> Result<Decl, ResugarError> {
    exact(&definition, 2)?;
    let name = name_child(&definition, 0)?.to_string();
    let signature_type = signature.map(|signature| &signature.children[1]);
    let definition_type = meta_value(definition.meta, "type");
    if let (Some(signature_type), Some(definition_type)) = (signature_type, definition_type)
        && !same_deep_shape(signature_type, definition_type)
    {
        return Err(ResugarError::InvalidChild {
            tag: definition.tag.as_str(),
            index: 1,
            expected: "`type` metadata matching the adjacent `defsig`",
        });
    }
    let declared_type = signature_type.or(definition_type);

    if meta_string(definition.meta, "chelis_role") == Some("property") {
        return resugar_property(signature, definition, name);
    }

    let value_node = node_ref(&definition.children[1]).ok();
    if value_node.is_some_and(|node| node.tag == DeepTag::Fn) {
        let function = value_node.expect("checked above");
        exact(&function, 2)?;
        let mut params = resugar_params(&function.children[0])?;
        let mut ret_ty = None;
        let mut effects = None;
        let mut dim_params = Vec::new();
        if let Some(declared_type) = declared_type {
            let type_node = node_ref(declared_type)?;
            if type_node.tag != DeepTag::TFn || type_node.children.is_empty() {
                return Err(ResugarError::InvalidChild {
                    tag: definition.tag.as_str(),
                    index: 1,
                    expected: "a non-empty `(t-fn ...)` type for a function definition",
                });
            }
            let param_count = params.len();
            if type_node.children.len() != param_count + 1 {
                return Err(ResugarError::InvalidChild {
                    tag: definition.tag.as_str(),
                    index: 1,
                    expected: "a function type matching the definition parameter count",
                });
            }
            for (param, ty) in params.iter_mut().zip(&type_node.children[..param_count]) {
                let ty = resugar_type(ty)?;
                param.ty = (!is_infer_type(&ty)).then_some(ty);
            }
            let result = resugar_type(type_node.children.last().expect("nonempty checked"))?;
            ret_ty = (!is_infer_type(&result)).then_some(result);
            effects = resugar_effect_metadata(declared_type)?;
            collect_quantified_variables(declared_type, &mut dim_params)?;
        }
        return Ok(Decl::FunDef {
            name,
            dim_params,
            params,
            ret_ty,
            effects,
            body: resugar_expression(&function.children[1])?,
            span: definition.span,
        });
    }

    Ok(Decl::LetDef {
        name,
        ty: declared_type.map(resugar_type).transpose()?,
        value: resugar_expression(&definition.children[1])?,
        span: definition.span,
    })
}

fn resugar_property(
    signature: Option<&NodeRef<'_>>,
    definition: NodeRef<'_>,
    name: String,
) -> Result<Decl, ResugarError> {
    let function = node_ref(&definition.children[1])?;
    if function.tag != DeepTag::Fn {
        return Err(ResugarError::InvalidChild {
            tag: definition.tag.as_str(),
            index: 1,
            expected: "a function-valued property definition",
        });
    }
    exact(&function, 2)?;

    let params_expr =
        meta_value(definition.meta, "property_quantifiers").unwrap_or(&function.children[0]);
    let params = resugar_params(params_expr)?;
    let preconditions = match meta_value(definition.meta, "property_preconditions") {
        None => Vec::new(),
        Some(value) => {
            let tuple = node_ref(value)?;
            if tuple.tag != DeepTag::Tuple {
                return Err(ResugarError::InvalidChild {
                    tag: definition.tag.as_str(),
                    index: 1,
                    expected: "tuple-valued `property_preconditions` metadata",
                });
            }
            tuple
                .children
                .iter()
                .map(resugar_expression)
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    let mut options = Vec::new();
    for (key, value) in &definition.meta.entries {
        let option = match key.as_str() {
            "property_tolerance" => Some(PropertyOption::Tolerance(
                resugar_expression(value)?,
                value.span(),
            )),
            "property_seed" => Some(PropertyOption::Seed(
                resugar_expression(value)?,
                value.span(),
            )),
            "property_samples" => Some(PropertyOption::Samples(
                resugar_expression(value)?,
                value.span(),
            )),
            "property_contracts" => {
                let contracts = node_ref(value)?;
                if contracts.tag != DeepTag::Tuple {
                    return Err(ResugarError::InvalidChild {
                        tag: definition.tag.as_str(),
                        index: 1,
                        expected: "tuple-valued `property_contracts` metadata",
                    });
                }
                for contract in contracts.children {
                    let DeepExpr::Atom(Atom::Str(identifier), span) = contract else {
                        return Err(ResugarError::InvalidChild {
                            tag: definition.tag.as_str(),
                            index: 1,
                            expected: "string property contract identifiers",
                        });
                    };
                    options.push(PropertyOption::Contract(identifier.clone(), *span));
                }
                None
            }
            _ => None,
        };
        if let Some(option) = option {
            options.push(option);
        }
    }

    if let Some(signature) = signature {
        let ty = node_ref(&signature.children[1])?;
        if ty.tag != DeepTag::TFn || ty.children.len() != params.len() + 1 {
            return Err(ResugarError::InvalidChild {
                tag: signature.tag.as_str(),
                index: 1,
                expected: "a property function type matching its quantifiers",
            });
        }
    }

    Ok(Decl::Property {
        name,
        params,
        preconditions,
        body: resugar_expression(&function.children[1])?,
        options,
        span: definition.span,
    })
}

fn resugar_import(node: NodeRef<'_>) -> Result<Decl, ResugarError> {
    let module = resugar_surface_path(&node, 0)?;
    let kind = match node.tag {
        DeepTag::ImportAll => {
            exact(&node, 1)?;
            ImportKind::All
        }
        DeepTag::Import => {
            exact(&node, 2)?;
            let names = structural_items(&node.children[1]).ok_or(ResugarError::InvalidChild {
                tag: node.tag.as_str(),
                index: 1,
                expected: "a structural import-name list",
            })?;
            if names.is_empty() {
                ImportKind::Qualified
            } else {
                ImportKind::Names(
                    names
                        .iter()
                        .map(|name| {
                            atom_name(name)
                                .map(str::to_string)
                                .ok_or(ResugarError::InvalidChild {
                                    tag: node.tag.as_str(),
                                    index: 1,
                                    expected: "a list of import names",
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
        }
        _ => unreachable!("caller restricts import tags"),
    };
    Ok(Decl::Import {
        module,
        kind,
        span: node.span,
    })
}

fn resugar_surface_path(node: &NodeRef<'_>, index: usize) -> Result<String, ResugarError> {
    if let Some(path) = meta_string(node.meta, "surf_path") {
        return Ok(path.to_string());
    }
    let lowered = name_child(node, index)?;
    Ok(lowered
        .split('.')
        .map(|component| {
            let mut chars = component.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join("."))
}

fn surf_group_size(node: &NodeRef<'_>) -> Result<usize, ResugarError> {
    let Some(value) = meta_value(node.meta, "surf_dim_group_size") else {
        return Ok(1);
    };
    let DeepExpr::Atom(Atom::Int(size), _) = value else {
        return Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 1,
            expected: "positive integer `surf_dim_group_size` metadata",
        });
    };
    usize::try_from(*size)
        .ok()
        .filter(|size| *size > 0)
        .ok_or(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 1,
            expected: "positive integer `surf_dim_group_size` metadata",
        })
}

fn resugar_type_alias(node: NodeRef<'_>) -> Result<Decl, ResugarError> {
    exact(&node, 3)?;
    Ok(Decl::TypeAlias {
        name: name_child(&node, 0)?.to_string(),
        params: structural_name_list(&node.children[1], node.tag)?,
        ty: resugar_type(&node.children[2])?,
        span: node.span,
    })
}

fn resugar_type_definition(node: NodeRef<'_>) -> Result<Decl, ResugarError> {
    if node.children.len() < 2 {
        return Err(ResugarError::MinimumArity {
            tag: node.tag.as_str(),
            minimum: 2,
            actual: node.children.len(),
        });
    }
    let variants = node.children[2..]
        .iter()
        .map(resugar_variant)
        .collect::<Result<Vec<_>, _>>()?;
    let opaque = match meta_value(node.meta, "opaque") {
        None => false,
        Some(DeepExpr::Atom(Atom::Bool(value), _)) => *value,
        Some(_) => {
            return Err(ResugarError::InvalidChild {
                tag: node.tag.as_str(),
                index: 1,
                expected: "boolean `opaque` metadata",
            });
        }
    };
    let invariant = meta_value(node.meta, "invariant")
        .map(resugar_invariant)
        .transpose()?;
    if invariant.is_some() && !opaque {
        return Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 1,
            expected: "an invariant only on an opaque type",
        });
    }
    Ok(Decl::TypeDef {
        name: name_child(&node, 0)?.to_string(),
        params: structural_name_list(&node.children[1], node.tag)?,
        variants,
        opaque,
        invariant,
        span: node.span,
    })
}

fn resugar_variant(expr: &DeepExpr) -> Result<Variant, ResugarError> {
    let node = node_ref(expr)?;
    if node.tag != DeepTag::Variant {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Deftype.as_str(),
            index: 2,
            expected: "a `(variant ...)` node",
        });
    }
    at_least(&node, 1)?;
    let has_record_fields = node.children[1..]
        .iter()
        .any(|child| child.tag() == Some(DeepTag::Field));
    let fields = if has_record_fields {
        VariantFields::Record(
            node.children[1..]
                .iter()
                .map(|field| {
                    let field = node_ref(field)?;
                    if field.tag != DeepTag::Field {
                        return Err(ResugarError::InvalidChild {
                            tag: node.tag.as_str(),
                            index: 1,
                            expected: "record fields without positional fields",
                        });
                    }
                    exact(&field, 2)?;
                    Ok((
                        name_child(&field, 0)?.to_string(),
                        resugar_type(&field.children[1])?,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?,
        )
    } else {
        VariantFields::Positional(
            node.children[1..]
                .iter()
                .map(resugar_type)
                .collect::<Result<Vec<_>, _>>()?,
        )
    };
    Ok(Variant {
        name: name_child(&node, 0)?.to_string(),
        fields,
        span: node.span,
    })
}

fn resugar_invariant(expr: &DeepExpr) -> Result<TypeInvariant, ResugarError> {
    let function = node_ref(expr)?;
    if function.tag != DeepTag::Fn {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Deftype.as_str(),
            index: 1,
            expected: "a function-valued invariant",
        });
    }
    exact(&function, 2)?;
    let params = resugar_params(&function.children[0])?;
    if params.len() != 1 || params[0].ty.is_some() {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Deftype.as_str(),
            index: 1,
            expected: "a one-binder untyped invariant function",
        });
    }
    Ok(TypeInvariant {
        binder: params[0].name.clone(),
        body: resugar_expression(&function.children[1])?,
        span: function.span,
    })
}

fn structural_name_list(expr: &DeepExpr, owner: DeepTag) -> Result<Vec<String>, ResugarError> {
    structural_items(expr)
        .ok_or(ResugarError::InvalidChild {
            tag: owner.as_str(),
            index: 1,
            expected: "a structural name list",
        })?
        .iter()
        .map(|value| {
            atom_name(value)
                .map(str::to_string)
                .ok_or(ResugarError::InvalidChild {
                    tag: owner.as_str(),
                    index: 1,
                    expected: "a structural name list",
                })
        })
        .collect()
}

fn node_ref(expr: &DeepExpr) -> Result<NodeRef<'_>, ResugarError> {
    let node = match expr {
        DeepExpr::Node(node, span) => NodeRef {
            tag: node.tag(),
            meta: node.meta(),
            children: node.children_slice(),
            span: *span,
        },
        DeepExpr::List(list, span) => {
            let Some(DeepExpr::Atom(Atom::Tag(tag), _)) = list.elements.first() else {
                return Err(ResugarError::ExpectedNode {
                    found: describe_deep(expr),
                });
            };
            let Some(DeepExpr::Map(meta, _)) = list.elements.get(1) else {
                return Err(ResugarError::InvalidChild {
                    tag: tag.as_str(),
                    index: 1,
                    expected: "a metadata map",
                });
            };
            NodeRef {
                tag: *tag,
                meta,
                children: &list.elements[2..],
                span: *span,
            }
        }
        _ => {
            return Err(ResugarError::ExpectedNode {
                found: describe_deep(expr),
            });
        }
    };
    if let Some((key, _)) = node.meta.entries.iter().find(|(key, _)| {
        key.starts_with("surf_")
            && !matches!(
                key.as_str(),
                "surf_path" | "surf_dim_group_size" | "surf_pipe_stage"
            )
    }) {
        return Err(ResugarError::UnknownSurfaceMetadata { key: key.clone() });
    }
    Ok(node)
}

fn resugar_node(node: NodeRef<'_>) -> Result<Expr, ResugarError> {
    use DeepTag as T;

    match node.tag {
        T::Var => {
            exact(&node, 1)?;
            let name = name_child(&node, 0)?;
            if starts_uppercase(name) {
                Ok(Expr::Constructor(name.to_string(), node.span))
            } else {
                Ok(Expr::Var(name.to_string(), node.span))
            }
        }
        T::Lit => resugar_literal(node),
        T::App => {
            at_least(&node, 1)?;
            if let Some(items) = resugar_finite_list(&node)? {
                return Ok(Expr::List(items, node.span));
            }
            if let Some(operator) = resugar_operator_application(&node)? {
                return Ok(operator);
            }
            let function = resugar_expression(&node.children[0])?;
            if node.children.len() == 1 && matches!(function, Expr::Constructor(..)) {
                return Ok(function);
            }
            let arguments = node.children[1..]
                .iter()
                .map(resugar_expression)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::Apply(Box::new(function), arguments, node.span))
        }
        T::Cast => {
            exact(&node, 2)?;
            let precision =
                primitive_type_name(&node.children[1]).ok_or(ResugarError::InvalidChild {
                    tag: node.tag.as_str(),
                    index: 1,
                    expected: "a `(t-prim {} precision)` node",
                })?;
            let operand = if let Ok(literal) = node_ref(&node.children[0])
                && literal.tag == DeepTag::Lit
                && default_literal_suffix_is_semantic_in_cast(&literal, precision)?
            {
                resugar_literal_with_default_suffix(literal)?
            } else {
                resugar_expression(&node.children[0])?
            };
            Ok(Expr::Cast(
                Box::new(operand),
                precision.to_string(),
                node.span,
            ))
        }
        T::Fn => {
            exact(&node, 2)?;
            Ok(Expr::Lambda(
                resugar_params(&node.children[0])?,
                Box::new(resugar_expression(&node.children[1])?),
                node.span,
            ))
        }
        T::Let => resugar_let(node),
        T::If => {
            exact(&node, 3)?;
            Ok(Expr::If(
                Box::new(resugar_expression(&node.children[0])?),
                Box::new(resugar_expression(&node.children[1])?),
                Box::new(resugar_expression(&node.children[2])?),
                node.span,
            ))
        }
        T::Match => {
            at_least(&node, 1)?;
            let scrutinee = resugar_expression(&node.children[0])?;
            let arms = node.children[1..]
                .iter()
                .map(resugar_arm)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::Match(Box::new(scrutinee), arms, node.span))
        }
        T::Record => {
            at_least(&node, 1)?;
            let name = name_child(&node, 0)?.to_string();
            let fields = node.children[1..]
                .iter()
                .map(resugar_kv_expression)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::Record(name, fields, node.span))
        }
        T::RecordUpdate => {
            at_least(&node, 1)?;
            let base = resugar_expression(&node.children[0])?;
            let fields = node.children[1..]
                .iter()
                .map(resugar_kv_expression)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::RecordUpdate(Box::new(base), fields, node.span))
        }
        T::Access => {
            exact(&node, 2)?;
            Ok(Expr::Access(
                Box::new(resugar_expression(&node.children[0])?),
                name_child(&node, 1)?.to_string(),
                node.span,
            ))
        }
        T::TupleGet => {
            exact(&node, 2)?;
            let index = integer_literal(&node.children[1]).ok_or(ResugarError::InvalidChild {
                tag: node.tag.as_str(),
                index: 1,
                expected: "an integer `(lit ...)` node",
            })?;
            Ok(Expr::TupleGet(
                Box::new(resugar_expression(&node.children[0])?),
                index,
                node.span,
            ))
        }
        T::Pipe => {
            at_least(&node, 2)?;
            Ok(Expr::Pipe(
                Box::new(resugar_expression(&node.children[0])?),
                node.children[1..]
                    .iter()
                    .map(resugar_pipe_stage)
                    .collect::<Result<Vec<_>, _>>()?,
                node.span,
            ))
        }
        T::Block => {
            at_least(&node, 1)?;
            Ok(Expr::Do(
                node.children
                    .iter()
                    .map(resugar_expression)
                    .collect::<Result<Vec<_>, _>>()?,
                node.span,
            ))
        }
        T::Tuple => Ok(Expr::Tuple(
            node.children
                .iter()
                .map(resugar_expression)
                .collect::<Result<Vec<_>, _>>()?,
            node.span,
        )),
        T::HandleEffect => {
            exact(&node, 2)?;
            let effect = decode_effect_kind(node)?;
            let argument = Box::new(resugar_expression(&node.children[0])?);
            let body = Box::new(resugar_expression(&node.children[1])?);
            match effect {
                EffectKind::Random => Ok(Expr::WithSeed(argument, body, node.span)),
                EffectKind::Resource => Ok(Expr::WithDevice(argument, body, node.span)),
            }
        }
        T::Borrow => unary_node(node, Expr::Borrow),
        T::Grad => resugar_grad(node),
        T::Vmap => {
            exact(&node, 2)?;
            let axis = integer_literal(&node.children[1]).ok_or(ResugarError::InvalidChild {
                tag: node.tag.as_str(),
                index: 1,
                expected: "an integer `(lit ...)` axis",
            })?;
            Ok(Expr::Vmap(
                Box::new(resugar_expression(&node.children[0])?),
                (axis != 0).then_some(axis),
                node.span,
            ))
        }
        T::Jit => unary_node(node, Expr::Jit),
        T::Realize => unary_node(node, Expr::Realize),
        T::Copy => unary_node(node, Expr::Copy),
        T::Quote => unary_node(node, Expr::Quote),
        T::Unquote => unary_node(node, Expr::Unquote),
        T::Splice => unary_node(node, Expr::Splice),
        T::Par => {
            at_least(&node, 1)?;
            Ok(Expr::Par(
                node.children
                    .iter()
                    .map(resugar_expression)
                    .collect::<Result<Vec<_>, _>>()?,
                node.span,
            ))
        }

        // Compile-time-total disposition over the closed 62-tag vocabulary.
        // Adding support to this boundary requires moving a tag out of this
        // arm together with its positive and negative tests.
        T::Module
        | T::Import
        | T::ImportAll
        | T::Export
        | T::Def
        | T::Defsig
        | T::Deftype
        | T::Typealias
        | T::Variant
        | T::Field
        | T::Defdim
        | T::Arm
        | T::PatVar
        | T::PatLit
        | T::PatCtor
        | T::PatTuple
        | T::PatRecord
        | T::PatWild
        | T::PatAs
        | T::TPrim
        | T::TFn
        | T::TTensor
        | T::TRef
        | T::TAdt
        | T::TVar
        | T::TUnit
        | T::TTuple
        | T::DName
        | T::DVar
        | T::DLit
        | T::DRank
        | T::Params
        | T::Bind
        | T::Kv
        | T::Effects
        | T::Resource => Err(ResugarError::InvalidSurfacePosition {
            tag: node.tag.as_str(),
            position: "expression",
        }),
    }
}

fn decode_effect_kind(node: NodeRef<'_>) -> Result<EffectKind, ResugarError> {
    let mut values = node
        .meta
        .entries
        .iter()
        .filter_map(|(key, value)| (key == "effect").then_some(value));
    let input = match (values.next(), values.next()) {
        (None, _) => EffectKindInput::Missing,
        (Some(_), Some(_)) => EffectKindInput::Malformed,
        (Some(DeepExpr::Atom(Atom::Name(symbol), _)), None) => EffectKindInput::Symbol(symbol),
        (Some(_), None) => EffectKindInput::Malformed,
    };
    EffectKind::decode(input).map_err(|_| ResugarError::InvalidChild {
        tag: node.tag.as_str(),
        index: 1,
        expected: "one canonical name-valued `effect` metadata entry",
    })
}

fn resugar_literal(node: NodeRef<'_>) -> Result<Expr, ResugarError> {
    resugar_literal_impl(node, false)
}

fn resugar_literal_with_default_suffix(node: NodeRef<'_>) -> Result<Expr, ResugarError> {
    resugar_literal_impl(node, true)
}

fn resugar_literal_impl(
    node: NodeRef<'_>,
    preserve_default_suffix: bool,
) -> Result<Expr, ResugarError> {
    exact(&node, 1)?;
    validate_literal_type(&node)?;
    reject_non_finite_float(&node.children[0])?;
    let suffix = literal_suffix(node.meta)?;
    if let DeepExpr::Atom(Atom::Int(value), _) = &node.children[0]
        && integer_minimum(suffix) == Some(*value)
    {
        let maximum = -(value + 1);
        let literal = |value| Expr::Lit(surface_integer_literal(value, suffix), node.span);
        return Ok(Expr::Binary(
            BinOp::Sub,
            Box::new(Expr::Unary(
                UnaryOp::Neg,
                Box::new(literal(maximum)),
                node.span,
            )),
            Box::new(literal(1)),
            node.span,
        ));
    }
    let literal = match (&node.children[0], suffix) {
        (DeepExpr::Atom(Atom::Int(value), _), Some(LiteralSuffix::I32))
            if preserve_default_suffix =>
        {
            Literal::TypedInt(*value, LiteralSuffix::I32)
        }
        (DeepExpr::Atom(Atom::Int(value), _), None | Some(LiteralSuffix::I32)) => {
            Literal::Int(*value)
        }
        (DeepExpr::Atom(Atom::Int(value), _), Some(suffix)) => Literal::TypedInt(*value, suffix),
        (DeepExpr::Atom(Atom::Float(value), _), Some(LiteralSuffix::F32))
            if preserve_default_suffix =>
        {
            Literal::TypedFloat(*value, LiteralSuffix::F32)
        }
        (DeepExpr::Atom(Atom::Float(value), _), None | Some(LiteralSuffix::F32)) => {
            Literal::Float(*value)
        }
        (DeepExpr::Atom(Atom::Float(value), _), Some(suffix)) if suffix.is_float() => {
            Literal::TypedFloat(*value, suffix)
        }
        (DeepExpr::Atom(Atom::Str(value), _), None) => Literal::Str(value.clone()),
        (DeepExpr::Atom(Atom::Bool(value), _), None) => Literal::Bool(*value),
        (DeepExpr::BareList(items, _), None) if items.is_empty() => {
            return Ok(Expr::Tuple(Vec::new(), node.span));
        }
        (DeepExpr::List(list, _), None) if list.elements.is_empty() => {
            return Ok(Expr::Tuple(Vec::new(), node.span));
        }
        _ => {
            return Err(ResugarError::InvalidChild {
                tag: node.tag.as_str(),
                index: 0,
                expected: "a literal compatible with its primitive `type` metadata",
            });
        }
    };
    Ok(Expr::Lit(literal, node.span))
}

fn surface_integer_literal(value: i64, suffix: Option<LiteralSuffix>) -> Literal {
    match suffix {
        None | Some(LiteralSuffix::I32) => Literal::Int(value),
        Some(suffix) => Literal::TypedInt(value, suffix),
    }
}

fn integer_minimum(suffix: Option<LiteralSuffix>) -> Option<i64> {
    match suffix {
        None | Some(LiteralSuffix::I32) => Some(i32::MIN as i64),
        Some(LiteralSuffix::I8) => Some(i8::MIN as i64),
        Some(LiteralSuffix::I16) => Some(i16::MIN as i64),
        Some(LiteralSuffix::I64) => Some(i64::MIN),
        Some(_) => None,
    }
}

fn default_literal_suffix_is_semantic_in_cast(
    node: &NodeRef<'_>,
    precision: &str,
) -> Result<bool, ResugarError> {
    let suffix = literal_suffix(node.meta)?;
    let numeric_target = matches!(
        precision,
        "f32" | "f64" | "bf16" | "f16" | "int8" | "int16" | "int32" | "int64"
    );
    Ok(match (&node.children[0], suffix) {
        (DeepExpr::Atom(Atom::Float(_), _), Some(LiteralSuffix::F32)) => {
            matches!(precision, "f64" | "bf16" | "f16")
        }
        (DeepExpr::Atom(Atom::Int(_), _), Some(LiteralSuffix::I32)) => {
            numeric_target && precision != "int32"
        }
        _ => false,
    })
}

fn validate_literal_type(node: &NodeRef<'_>) -> Result<(), ResugarError> {
    let Some(ty) = meta_value(node.meta, "type") else {
        return Ok(());
    };
    let ty_node = node_ref(ty).map_err(|_| ResugarError::InvalidChild {
        tag: node.tag.as_str(),
        index: 1,
        expected: "literal `type` metadata compatible with its value",
    })?;
    let compatible = match (&node.children[0], ty_node.tag) {
        (DeepExpr::Atom(Atom::Bool(_), _), DeepTag::TPrim) => {
            primitive_type_name(ty) == Some("bool")
        }
        (DeepExpr::Atom(Atom::Str(_), _), DeepTag::TPrim) => {
            primitive_type_name(ty) == Some("string")
        }
        (DeepExpr::Atom(Atom::Int(_), _), DeepTag::TPrim) => {
            primitive_type_name(ty).is_some_and(|name| {
                matches!(
                    name,
                    "int8" | "int16" | "int32" | "int64" | "f16" | "bf16" | "f32" | "f64"
                )
            })
        }
        (DeepExpr::Atom(Atom::Float(_), _), DeepTag::TPrim) => primitive_type_name(ty)
            .is_some_and(|name| matches!(name, "f16" | "bf16" | "f32" | "f64")),
        (DeepExpr::BareList(items, _), DeepTag::TUnit) => items.is_empty(),
        (DeepExpr::List(list, _), DeepTag::TUnit) => list.elements.is_empty(),
        _ => false,
    };
    if compatible {
        Ok(())
    } else {
        Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 1,
            expected: "literal `type` metadata compatible with its value",
        })
    }
}

fn unary_node(
    node: NodeRef<'_>,
    constructor: impl FnOnce(Box<Expr>, Span) -> Expr,
) -> Result<Expr, ResugarError> {
    exact(&node, 1)?;
    Ok(constructor(
        Box::new(resugar_expression(&node.children[0])?),
        node.span,
    ))
}

fn resugar_let(node: NodeRef<'_>) -> Result<Expr, ResugarError> {
    if let Some(destructuring) = try_resugar_destructuring_let(&node)? {
        return Ok(destructuring);
    }
    exact(&node, 2)?;
    let binding_node = node_ref(&node.children[0])?;
    if binding_node.tag != DeepTag::Bind {
        return Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 0,
            expected: "a `(bind ...)` node",
        });
    }
    at_least(&binding_node, 2)?;
    if !binding_node.children.len().is_multiple_of(2) {
        return Err(ResugarError::InvalidChild {
            tag: binding_node.tag.as_str(),
            index: binding_node.children.len() - 1,
            expected: "complete name/expression binding pairs",
        });
    }
    let mut bindings = Vec::with_capacity(binding_node.children.len() / 2);
    for pair in binding_node.children.chunks_exact(2) {
        bindings.push(LetBinding {
            pattern: LetPattern::Var(
                atom_name(&pair[0])
                    .ok_or(ResugarError::InvalidChild {
                        tag: binding_node.tag.as_str(),
                        index: bindings.len() * 2,
                        expected: "a binding name",
                    })?
                    .to_string(),
                binding_node.span,
            ),
            ty: type_metadata(&pair[1]).transpose()?,
            // A let binding's declared type is encoded on its value node. The
            // binding field above consumes that outer annotation; nested child
            // annotations still resugar normally.
            value: resugar_node(node_ref(&pair[1])?)?,
        });
    }
    let body = resugar_expression(&node.children[1])?;
    match body {
        Expr::Block(body_bindings, body, span) => {
            bindings.extend(body_bindings);
            Ok(Expr::Block(bindings, body, node.span.merge(span)))
        }
        body => Ok(Expr::Block(bindings, Box::new(body), node.span)),
    }
}

fn try_resugar_destructuring_let(node: &NodeRef<'_>) -> Result<Option<Expr>, ResugarError> {
    if node.tag != DeepTag::Let || node.children.len() != 2 {
        return Ok(None);
    }
    let root_bind = node_ref(&node.children[0])?;
    if root_bind.tag != DeepTag::Bind || meta_bool(root_bind.meta, "destructure") != Some(true) {
        return Ok(None);
    }
    exact(&root_bind, 2)?;
    let root_name = name_child(&root_bind, 0)?.to_string();
    let mut temp_paths = HashMap::from([(root_name, Vec::<usize>::new())]);
    let mut pattern_nodes = HashMap::<Vec<usize>, Option<String>>::from([(Vec::new(), None)]);
    let mut current = &node.children[1];

    while let Ok(let_node) = node_ref(current) {
        if let_node.tag != DeepTag::Let || let_node.children.len() != 2 {
            break;
        }
        let bind = node_ref(&let_node.children[0])?;
        if bind.tag != DeepTag::Bind || meta_bool(bind.meta, "destructure") != Some(true) {
            break;
        }
        exact(&bind, 2)?;
        let bound_name = name_child(&bind, 0)?.to_string();
        if let Some((parent, index)) = tuple_get_source(&bind.children[1]) {
            let Some(parent_path) = temp_paths.get(parent) else {
                return Err(ResugarError::InvalidChild {
                    tag: DeepTag::Bind.as_str(),
                    index: 1,
                    expected: "a tuple extraction from an earlier destructuring temporary",
                });
            };
            let index = usize::try_from(index).map_err(|_| ResugarError::InvalidChild {
                tag: DeepTag::Bind.as_str(),
                index: 1,
                expected: "a non-negative tuple destructuring index",
            })?;
            let mut path = parent_path.clone();
            path.push(index);
            if pattern_nodes.insert(path.clone(), None).is_some() {
                return Err(ResugarError::InvalidChild {
                    tag: DeepTag::Bind.as_str(),
                    index: 1,
                    expected: "one extraction per tuple destructuring path",
                });
            }
            temp_paths.insert(bound_name, path);
        } else if let Some(source) = variable_name(&bind.children[1]) {
            let Some(path) = temp_paths.get(source).cloned() else {
                return Err(ResugarError::InvalidChild {
                    tag: DeepTag::Bind.as_str(),
                    index: 1,
                    expected: "a variable bound from an earlier destructuring temporary",
                });
            };
            let slot = pattern_nodes
                .get_mut(&path)
                .ok_or(ResugarError::InvalidChild {
                    tag: DeepTag::Bind.as_str(),
                    index: 1,
                    expected: "an existing tuple destructuring path",
                })?;
            if slot.replace(bound_name).is_some() {
                return Err(ResugarError::InvalidChild {
                    tag: DeepTag::Bind.as_str(),
                    index: 1,
                    expected: "one binder per tuple destructuring path",
                });
            }
        } else {
            return Err(ResugarError::InvalidChild {
                tag: DeepTag::Bind.as_str(),
                index: 1,
                expected: "a tuple extraction or variable destructuring bind",
            });
        }
        current = &let_node.children[1];
    }

    let pattern = build_destructuring_pattern(&[], &pattern_nodes, root_bind.span)?;
    let binding = LetBinding {
        pattern,
        ty: None,
        value: resugar_expression(&root_bind.children[1])?,
    };
    let body = resugar_expression(current)?;
    Ok(Some(match body {
        Expr::Block(mut bindings, body, span) => {
            bindings.insert(0, binding);
            Expr::Block(bindings, body, node.span.merge(span))
        }
        body => Expr::Block(vec![binding], Box::new(body), node.span),
    }))
}

fn build_destructuring_pattern(
    path: &[usize],
    nodes: &HashMap<Vec<usize>, Option<String>>,
    span: Span,
) -> Result<LetPattern, ResugarError> {
    let name = nodes.get(path).ok_or(ResugarError::InvalidChild {
        tag: DeepTag::Bind.as_str(),
        index: 1,
        expected: "a complete tuple destructuring tree",
    })?;
    let mut child_indices = nodes
        .keys()
        .filter(|candidate| candidate.len() == path.len() + 1 && candidate.starts_with(path))
        .map(|candidate| *candidate.last().expect("one element longer"))
        .collect::<Vec<_>>();
    child_indices.sort_unstable();
    child_indices.dedup();

    if let Some(name) = name {
        if !child_indices.is_empty() {
            return Err(ResugarError::InvalidChild {
                tag: DeepTag::Bind.as_str(),
                index: 1,
                expected: "a destructuring leaf without child extractions",
            });
        }
        return Ok(LetPattern::Var(name.clone(), span));
    }
    if child_indices.is_empty() {
        return Ok(LetPattern::Wildcard(span));
    }
    if child_indices != (0..child_indices.len()).collect::<Vec<_>>() {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Bind.as_str(),
            index: 1,
            expected: "contiguous tuple destructuring indices starting at zero",
        });
    }
    let parts = child_indices
        .into_iter()
        .map(|index| {
            let mut child = path.to_vec();
            child.push(index);
            build_destructuring_pattern(&child, nodes, span)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(LetPattern::Tuple(parts, span))
}

fn tuple_get_source(expr: &DeepExpr) -> Option<(&str, i64)> {
    let node = node_ref(expr).ok()?;
    if node.tag != DeepTag::TupleGet || node.children.len() != 2 {
        return None;
    }
    Some((
        variable_name(&node.children[0])?,
        integer_literal(&node.children[1])?,
    ))
}

fn resugar_params(expr: &DeepExpr) -> Result<Vec<Param>, ResugarError> {
    let node = node_ref(expr)?;
    if node.tag != DeepTag::Params {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Fn.as_str(),
            index: 0,
            expected: "a `(params ...)` node",
        });
    }
    node.children.iter().map(resugar_param).collect()
}

fn resugar_param(expr: &DeepExpr) -> Result<Param, ResugarError> {
    if let Some(name) = atom_name(expr) {
        return Ok(Param {
            name: name.to_string(),
            ty: None,
            span: expr.span(),
        });
    }

    if let DeepExpr::MetaExpr(meta, span) = expr {
        let name = atom_name(&meta.expr).ok_or(ResugarError::InvalidChild {
            tag: DeepTag::Params.as_str(),
            index: 0,
            expected: "a parameter name",
        })?;
        let ty = meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| resugar_type(value))
            .transpose()?;
        return Ok(Param {
            name: name.to_string(),
            ty,
            span: *span,
        });
    }

    let items = structural_items(expr).ok_or(ResugarError::InvalidChild {
        tag: DeepTag::Params.as_str(),
        index: 0,
        expected: "a parameter name or `(name {type: ...})` structural list",
    })?;
    if items.len() != 2 {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Params.as_str(),
            index: 0,
            expected: "a two-item `(name {type: ...})` structural list",
        });
    }
    let name = atom_name(&items[0]).ok_or(ResugarError::InvalidChild {
        tag: DeepTag::Params.as_str(),
        index: 0,
        expected: "a parameter name",
    })?;
    let DeepExpr::Map(meta, _) = &items[1] else {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Params.as_str(),
            index: 0,
            expected: "parameter type metadata",
        });
    };
    let ty_expr = meta_value(meta, "type").ok_or(ResugarError::InvalidChild {
        tag: DeepTag::Params.as_str(),
        index: 0,
        expected: "parameter `type` metadata",
    })?;
    Ok(Param {
        name: name.to_string(),
        ty: Some(resugar_type(ty_expr)?),
        span: expr.span(),
    })
}

fn resugar_arm(expr: &DeepExpr) -> Result<MatchArm, ResugarError> {
    let node = node_ref(expr)?;
    if node.tag != DeepTag::Arm {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Match.as_str(),
            index: 1,
            expected: "an `(arm ...)` node",
        });
    }
    exact(&node, 3)?;
    let guard = if is_empty_structural_list(&node.children[1]) {
        None
    } else {
        Some(resugar_expression(&node.children[1])?)
    };
    Ok(MatchArm {
        pattern: resugar_pattern(&node.children[0])?,
        guard,
        body: resugar_expression(&node.children[2])?,
        span: node.span,
    })
}

fn resugar_pattern(expr: &DeepExpr) -> Result<Pattern, ResugarError> {
    let node = node_ref(expr)?;
    match node.tag {
        DeepTag::PatWild => {
            exact(&node, 0)?;
            Ok(Pattern::Wildcard(node.span))
        }
        DeepTag::PatVar => {
            exact(&node, 1)?;
            Ok(Pattern::Var(name_child(&node, 0)?.to_string(), node.span))
        }
        DeepTag::PatLit => {
            exact(&node, 1)?;
            reject_non_finite_float(&node.children[0])?;
            Ok(Pattern::Lit(
                literal_from_atom(&node.children[0]).ok_or(ResugarError::InvalidChild {
                    tag: node.tag.as_str(),
                    index: 0,
                    expected: "a literal atom",
                })?,
                node.span,
            ))
        }
        DeepTag::PatCtor => {
            at_least(&node, 1)?;
            Ok(Pattern::Constructor(
                name_child(&node, 0)?.to_string(),
                node.children[1..]
                    .iter()
                    .map(resugar_pattern)
                    .collect::<Result<Vec<_>, _>>()?,
                node.span,
            ))
        }
        DeepTag::PatTuple => Ok(Pattern::Tuple(
            node.children
                .iter()
                .map(resugar_pattern)
                .collect::<Result<Vec<_>, _>>()?,
            node.span,
        )),
        DeepTag::PatRecord => {
            at_least(&node, 1)?;
            Ok(Pattern::Record(
                name_child(&node, 0)?.to_string(),
                node.children[1..]
                    .iter()
                    .map(resugar_kv_pattern)
                    .collect::<Result<Vec<_>, _>>()?,
                node.span,
            ))
        }
        DeepTag::PatAs => {
            exact(&node, 2)?;
            Ok(Pattern::As(
                name_child(&node, 0)?.to_string(),
                Box::new(resugar_pattern(&node.children[1])?),
                node.span,
            ))
        }
        _ => Err(ResugarError::InvalidChild {
            tag: DeepTag::Arm.as_str(),
            index: 0,
            expected: "a pattern node",
        }),
    }
}

fn resugar_kv_expression(expr: &DeepExpr) -> Result<(String, Expr), ResugarError> {
    let node = node_ref(expr)?;
    if node.tag != DeepTag::Kv {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Record.as_str(),
            index: 1,
            expected: "a `(kv ...)` node",
        });
    }
    exact(&node, 2)?;
    Ok((
        name_child(&node, 0)?.to_string(),
        resugar_expression(&node.children[1])?,
    ))
}

fn resugar_kv_pattern(expr: &DeepExpr) -> Result<(String, Pattern), ResugarError> {
    let node = node_ref(expr)?;
    if node.tag != DeepTag::Kv {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::PatRecord.as_str(),
            index: 1,
            expected: "a `(kv ...)` node",
        });
    }
    exact(&node, 2)?;
    Ok((
        name_child(&node, 0)?.to_string(),
        resugar_pattern(&node.children[1])?,
    ))
}

fn resugar_pipe_stage(expr: &DeepExpr) -> Result<Expr, ResugarError> {
    let node = node_ref(expr)?;
    let Some(marker) = meta_value(node.meta, "surf_pipe_stage") else {
        return resugar_expression(expr);
    };
    let DeepExpr::Atom(Atom::Str(marker), _) = marker else {
        return Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 1,
            expected: "string `surf_pipe_stage` metadata",
        });
    };
    if marker != "call-first" || node.tag != DeepTag::Fn {
        return Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 1,
            expected: "validated `surf_pipe_stage: \"call-first\"` on a function stage",
        });
    }
    let lambda = resugar_expression(expr)?;
    let Expr::Lambda(params, body, span) = lambda else {
        unreachable!("validated fn resugars to lambda")
    };
    let [param] = params.as_slice() else {
        return Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 0,
            expected: "a one-parameter call-first stage",
        });
    };
    let is_param = |expr: &Expr| matches!(expr, Expr::Var(name, _) if name == &param.name);
    match *body {
        Expr::Apply(function, mut arguments, apply_span)
            if arguments.first().is_some_and(is_param) =>
        {
            arguments.remove(0);
            Ok(Expr::Apply(function, arguments, apply_span))
        }
        special @ (Expr::Realize(_, _) | Expr::Copy(_, _) | Expr::Cast(_, _, _)) => {
            let carries_first = match &special {
                Expr::Realize(argument, _)
                | Expr::Copy(argument, _)
                | Expr::Cast(argument, _, _) => is_param(argument),
                _ => unreachable!("outer pattern restricts variants"),
            };
            if !carries_first {
                return Err(ResugarError::InvalidChild {
                    tag: node.tag.as_str(),
                    index: 1,
                    expected: "a call-first stage body using its parameter",
                });
            }
            Ok(Expr::Lambda(params, Box::new(special), span))
        }
        _ => Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 1,
            expected: "a call-first stage body using its parameter as the first argument",
        }),
    }
}

fn resugar_grad(node: NodeRef<'_>) -> Result<Expr, ResugarError> {
    if !(node.children.len() == 1 || node.children.len() == 2) {
        return Err(ResugarError::ExactArity {
            tag: node.tag.as_str(),
            expected: 1,
            actual: node.children.len(),
        });
    }
    let wrt = meta_value(node.meta, "wrt")
        .map(resugar_wrt_names)
        .transpose()?;
    if node.children.len() == 2 && wrt.is_none() {
        return Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 1,
            expected: "`wrt` metadata when an index child is present",
        });
    }
    Ok(Expr::Grad(
        Box::new(resugar_expression(&node.children[0])?),
        wrt,
        node.span,
    ))
}

fn resugar_wrt_names(expr: &DeepExpr) -> Result<Vec<String>, ResugarError> {
    if let Ok(node) = node_ref(expr) {
        match node.tag {
            DeepTag::Var => return Ok(vec![name_child(&node, 0)?.to_string()]),
            DeepTag::Tuple => {
                return node
                    .children
                    .iter()
                    .map(|child| {
                        let var = node_ref(child)?;
                        if var.tag != DeepTag::Var {
                            return Err(ResugarError::InvalidChild {
                                tag: DeepTag::Grad.as_str(),
                                index: 1,
                                expected: "variable-valued `wrt` metadata",
                            });
                        }
                        Ok(name_child(&var, 0)?.to_string())
                    })
                    .collect();
            }
            _ => {}
        }
    }
    Err(ResugarError::InvalidChild {
        tag: DeepTag::Grad.as_str(),
        index: 1,
        expected: "a variable or tuple of variables in `wrt` metadata",
    })
}

fn resugar_finite_list(node: &NodeRef<'_>) -> Result<Option<Vec<Expr>>, ResugarError> {
    let Some("Cons") = variable_name(node.children.first().expect("app has a callee")) else {
        return Ok(None);
    };
    if node.children.len() != 3 {
        return Ok(None);
    }

    let mut items = vec![resugar_expression(&node.children[1])?];
    let mut tail = &node.children[2];
    loop {
        if variable_name(tail) == Some("Nil") {
            return Ok(Some(items));
        }
        let Ok(cons) = node_ref(tail) else {
            return Ok(None);
        };
        if cons.tag != DeepTag::App
            || cons.children.len() != 3
            || variable_name(&cons.children[0]) != Some("Cons")
        {
            return Ok(None);
        }
        items.push(resugar_expression(&cons.children[1])?);
        tail = &cons.children[2];
    }
}

fn resugar_operator_application(node: &NodeRef<'_>) -> Result<Option<Expr>, ResugarError> {
    let Some(name) = node.children.first().and_then(variable_name) else {
        return Ok(None);
    };
    if node.children.len() == 2 {
        let operator = match name {
            "neg" => UnaryOp::Neg,
            "not" => UnaryOp::Not,
            _ => return Ok(None),
        };
        return Ok(Some(Expr::Unary(
            operator,
            Box::new(resugar_expression(&node.children[1])?),
            node.span,
        )));
    }
    if node.children.len() != 3 {
        return Ok(None);
    }
    let operator = match name {
        "add" => BinOp::Add,
        "sub" => BinOp::Sub,
        "mul" => BinOp::Mul,
        "div" => BinOp::Div,
        "mod" => BinOp::Mod,
        "eq" => BinOp::Eq,
        "neq" => BinOp::Ne,
        "cmplt" => BinOp::Lt,
        "lte" => BinOp::Le,
        "gte" => BinOp::Ge,
        "and" => BinOp::And,
        "or" => BinOp::Or,
        _ => return Ok(None),
    };
    Ok(Some(Expr::Binary(
        operator,
        Box::new(resugar_expression(&node.children[1])?),
        Box::new(resugar_expression(&node.children[2])?),
        node.span,
    )))
}

fn variable_name(expr: &DeepExpr) -> Option<&str> {
    let node = node_ref(expr).ok()?;
    (node.tag == DeepTag::Var && node.children.len() == 1)
        .then(|| atom_name(&node.children[0]))
        .flatten()
}

fn integer_literal(expr: &DeepExpr) -> Option<i64> {
    let node = node_ref(expr).ok()?;
    if node.tag != DeepTag::Lit || node.children.len() != 1 {
        return None;
    }
    match &node.children[0] {
        DeepExpr::Atom(Atom::Int(value), _) => Some(*value),
        _ => None,
    }
}

fn literal_from_atom(expr: &DeepExpr) -> Option<Literal> {
    match expr {
        DeepExpr::Atom(Atom::Int(value), _) => Some(Literal::Int(*value)),
        DeepExpr::Atom(Atom::Float(value), _) => Some(Literal::Float(*value)),
        DeepExpr::Atom(Atom::Str(value), _) => Some(Literal::Str(value.clone())),
        DeepExpr::Atom(Atom::Bool(value), _) => Some(Literal::Bool(*value)),
        _ => None,
    }
}

fn reject_non_finite_float(expr: &DeepExpr) -> Result<(), ResugarError> {
    if matches!(expr, DeepExpr::Atom(Atom::Float(value), _) if !value.is_finite()) {
        Err(ResugarError::NonFiniteFloat)
    } else {
        Ok(())
    }
}

fn structural_items(expr: &DeepExpr) -> Option<&[DeepExpr]> {
    match expr {
        DeepExpr::BareList(items, _) => Some(items),
        DeepExpr::List(list, _) if list.tag().is_none() => Some(&list.elements),
        _ => None,
    }
}

fn is_empty_structural_list(expr: &DeepExpr) -> bool {
    structural_items(expr).is_some_and(<[DeepExpr]>::is_empty)
}

fn meta_value<'a>(meta: &'a MetaMap, key: &str) -> Option<&'a DeepExpr> {
    meta.entries
        .iter()
        .find(|(candidate, _)| candidate == key)
        .map(|(_, value)| value)
}

fn meta_string<'a>(meta: &'a MetaMap, key: &str) -> Option<&'a str> {
    match meta_value(meta, key) {
        Some(DeepExpr::Atom(Atom::Str(value), _)) => Some(value),
        _ => None,
    }
}

fn meta_bool(meta: &MetaMap, key: &str) -> Option<bool> {
    match meta_value(meta, key) {
        Some(DeepExpr::Atom(Atom::Bool(value), _)) => Some(*value),
        _ => None,
    }
}

fn type_metadata(expr: &DeepExpr) -> Option<Result<TypeExpr, ResugarError>> {
    let node = node_ref(expr).ok()?;
    meta_value(node.meta, "type").map(resugar_type)
}

fn resugar_type(expr: &DeepExpr) -> Result<TypeExpr, ResugarError> {
    let node = node_ref(expr)?;
    match node.tag {
        DeepTag::TPrim | DeepTag::TVar => {
            exact(&node, 1)?;
            let name = name_child(&node, 0)?;
            if node.tag == DeepTag::TVar && name == "_" {
                Ok(TypeExpr::Infer(node.span))
            } else {
                Ok(TypeExpr::Named(name.to_string(), node.span))
            }
        }
        DeepTag::TUnit => {
            exact(&node, 0)?;
            Ok(TypeExpr::Tuple(Vec::new(), node.span))
        }
        DeepTag::TAdt => {
            at_least(&node, 1)?;
            let name = name_child(&node, 0)?.to_string();
            let arguments = node.children[1..]
                .iter()
                .map(resugar_type)
                .collect::<Result<Vec<_>, _>>()?;
            if arguments.is_empty() {
                Ok(TypeExpr::Named(name, node.span))
            } else {
                Ok(TypeExpr::App(name, arguments, node.span))
            }
        }
        DeepTag::TFn => {
            at_least(&node, 1)?;
            let (return_type, arguments) = node.children.split_last().expect("nonempty checked");
            Ok(TypeExpr::Arrow(
                arguments
                    .iter()
                    .map(resugar_type)
                    .collect::<Result<Vec<_>, _>>()?,
                Box::new(resugar_type(return_type)?),
                node.span,
            ))
        }
        DeepTag::TTensor => {
            at_least(&node, 1)?;
            let (precision, dimensions) = node.children.split_last().expect("nonempty checked");
            let precision = type_name(precision).ok_or(ResugarError::InvalidChild {
                tag: node.tag.as_str(),
                index: node.children.len() - 1,
                expected: "a primitive or type-variable precision",
            })?;
            Ok(TypeExpr::Tensor(
                dimensions
                    .iter()
                    .map(resugar_dimension)
                    .collect::<Result<Vec<_>, _>>()?,
                precision.to_string(),
                node.span,
            ))
        }
        DeepTag::TRef => {
            exact(&node, 1)?;
            Ok(TypeExpr::Ref(
                Box::new(resugar_type(&node.children[0])?),
                node.span,
            ))
        }
        DeepTag::TTuple => Ok(TypeExpr::Tuple(
            node.children
                .iter()
                .map(resugar_type)
                .collect::<Result<Vec<_>, _>>()?,
            node.span,
        )),
        DeepTag::DName | DeepTag::DVar | DeepTag::DLit | DeepTag::DRank => resugar_dimension(expr),
        _ => Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 0,
            expected: "a Deep type node",
        }),
    }
}

fn resugar_dimension(expr: &DeepExpr) -> Result<TypeExpr, ResugarError> {
    let node = node_ref(expr)?;
    exact(&node, 1)?;
    match node.tag {
        DeepTag::DName | DeepTag::DVar => Ok(TypeExpr::Named(
            name_child(&node, 0)?.to_string(),
            node.span,
        )),
        DeepTag::DLit => match &node.children[0] {
            DeepExpr::Atom(Atom::Int(value), _) => {
                Ok(TypeExpr::Named(value.to_string(), node.span))
            }
            _ => Err(ResugarError::InvalidChild {
                tag: node.tag.as_str(),
                index: 0,
                expected: "an integer dimension",
            }),
        },
        DeepTag::DRank => Ok(TypeExpr::RankSpread(
            name_child(&node, 0)?.to_string(),
            node.span,
        )),
        _ => Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 0,
            expected: "a Deep dimension node",
        }),
    }
}

fn type_name(expr: &DeepExpr) -> Option<&str> {
    let node = node_ref(expr).ok()?;
    matches!(node.tag, DeepTag::TPrim | DeepTag::TVar)
        .then(|| name_child(&node, 0).ok())
        .flatten()
}

fn is_infer_type(ty: &TypeExpr) -> bool {
    matches!(ty, TypeExpr::Infer(_))
}

fn collect_quantified_variables(
    expr: &DeepExpr,
    names: &mut Vec<String>,
) -> Result<(), ResugarError> {
    if let Ok(node) = node_ref(expr) {
        if matches!(node.tag, DeepTag::DVar | DeepTag::TVar) {
            exact(&node, 1)?;
            let name = name_child(&node, 0)?.to_string();
            if name != "_" && !names.contains(&name) {
                names.push(name);
            }
        }
        for child in node.children {
            collect_quantified_variables(child, names)?;
        }
    }
    Ok(())
}

fn resugar_effect_metadata(expr: &DeepExpr) -> Result<Option<Vec<EffectExpr>>, ResugarError> {
    let node = node_ref(expr)?;
    let Some(effects) = meta_value(node.meta, "eff") else {
        return Ok(None);
    };
    let effects = node_ref(effects)?;
    if effects.tag != DeepTag::Effects {
        return Err(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index: 1,
            expected: "an `(effects ...)` value in `eff` metadata",
        });
    }
    effects
        .children
        .iter()
        .map(|effect| match effect {
            DeepExpr::Atom(Atom::Name(name), span) => match name.as_str() {
                "diff" => Ok(EffectExpr::Diff(*span)),
                "random" => Ok(EffectExpr::Random(*span)),
                "accum" => Ok(EffectExpr::Accum(*span)),
                "io" => Ok(EffectExpr::Io(*span)),
                "test" => Ok(EffectExpr::Test(*span)),
                _ => Err(ResugarError::InvalidChild {
                    tag: DeepTag::Effects.as_str(),
                    index: 0,
                    expected: "a canonical effect name or resource node",
                }),
            },
            _ => {
                let resource = node_ref(effect)?;
                if resource.tag != DeepTag::Resource || resource.children.len() != 1 {
                    return Err(ResugarError::InvalidChild {
                        tag: DeepTag::Effects.as_str(),
                        index: 0,
                        expected: "a canonical effect name or resource node",
                    });
                }
                let DeepExpr::Atom(Atom::Str(device), _) = &resource.children[0] else {
                    return Err(ResugarError::InvalidChild {
                        tag: DeepTag::Resource.as_str(),
                        index: 0,
                        expected: "a device string",
                    });
                };
                Ok(EffectExpr::Resource(device.clone(), resource.span))
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn literal_suffix(meta: &MetaMap) -> Result<Option<LiteralSuffix>, ResugarError> {
    let Some(value) = meta
        .entries
        .iter()
        .find(|(key, _)| key == "type")
        .map(|(_, value)| value)
    else {
        return Ok(None);
    };
    if node_ref(value).is_ok_and(|node| node.tag == DeepTag::TUnit) {
        return Ok(None);
    }
    let Some(name) = primitive_type_name(value) else {
        return Err(ResugarError::InvalidChild {
            tag: DeepTag::Lit.as_str(),
            index: 1,
            expected: "primitive `type` metadata",
        });
    };
    let suffix = match name {
        "bool" | "string" => return Ok(None),
        "f32" => LiteralSuffix::F32,
        "f64" => LiteralSuffix::F64,
        "bf16" => LiteralSuffix::Bf16,
        "f16" => LiteralSuffix::F16,
        "int8" => LiteralSuffix::I8,
        "int16" => LiteralSuffix::I16,
        "int32" => LiteralSuffix::I32,
        "int64" => LiteralSuffix::I64,
        _ => {
            return Err(ResugarError::InvalidChild {
                tag: DeepTag::Lit.as_str(),
                index: 1,
                expected: "a numeric primitive `type` metadata value",
            });
        }
    };
    Ok(Some(suffix))
}

fn primitive_type_name(expr: &DeepExpr) -> Option<&str> {
    let node = node_ref(expr).ok()?;
    (node.tag == DeepTag::TPrim && node.children.len() == 1)
        .then(|| atom_name(&node.children[0]))
        .flatten()
}

fn name_child<'a>(node: &NodeRef<'a>, index: usize) -> Result<&'a str, ResugarError> {
    node.children
        .get(index)
        .and_then(atom_name)
        .ok_or(ResugarError::InvalidChild {
            tag: node.tag.as_str(),
            index,
            expected: "a name atom",
        })
}

fn atom_name(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(Atom::Name(name), _) => Some(name),
        _ => None,
    }
}

fn exact(node: &NodeRef<'_>, expected: usize) -> Result<(), ResugarError> {
    if node.children.len() == expected {
        Ok(())
    } else {
        Err(ResugarError::ExactArity {
            tag: node.tag.as_str(),
            expected,
            actual: node.children.len(),
        })
    }
}

fn at_least(node: &NodeRef<'_>, minimum: usize) -> Result<(), ResugarError> {
    if node.children.len() >= minimum {
        Ok(())
    } else {
        Err(ResugarError::MinimumArity {
            tag: node.tag.as_str(),
            minimum,
            actual: node.children.len(),
        })
    }
}

fn starts_uppercase(name: &str) -> bool {
    name.chars().next().is_some_and(char::is_uppercase)
}

fn describe_deep(expr: &DeepExpr) -> String {
    match expr {
        DeepExpr::Atom(_, _) => "an atom".to_string(),
        DeepExpr::List(_, _) => "an untagged list".to_string(),
        DeepExpr::Map(_, _) => "a metadata map".to_string(),
        DeepExpr::MetaExpr(_, _) => "a legacy metadata wrapper".to_string(),
        DeepExpr::Node(_, _) => "a typed node".to_string(),
        DeepExpr::BareList(_, _) => "a structural bare list".to_string(),
        DeepExpr::UnknownForm(data) => format!("unknown form `{}`", data.head),
    }
}
