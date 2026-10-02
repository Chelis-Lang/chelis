//! Private Deep-text and legacy-serde codecs for typed annotation payloads.
use serde::{Deserialize, Serialize};

use crate::annotations::*;
use crate::{Atom, DeepTag, Expr, RawAtom, RawExpr, Span, metadata::MetadataError};

fn stamp_error(key: &str, error: crate::StampError) -> MetadataError {
    let mut result = invalid(key, error.span, "valid syntax for this metadata payload");
    result.detail = Some(error.to_string());
    result
}
fn runtime(key: &str, raw: RawExpr) -> Result<RuntimeExpression, MetadataError> {
    RuntimeExpression::try_for_key(
        crate::stamp_to_typed::stamp_runtime_expr(raw).map_err(|e| stamp_error(key, e))?,
        key,
    )
}
fn name(raw: RawExpr, key: &str) -> Result<Spanned<String>, MetadataError> {
    match raw {
        RawExpr::Atom(RawAtom::Symbol(name), span) => Ok(Spanned::new(name, span)),
        other => Err(invalid(key, other.span(), "a bare name")),
    }
}
fn string(raw: RawExpr, key: &str) -> Result<Spanned<String>, MetadataError> {
    match raw {
        RawExpr::Atom(RawAtom::Str(value), span) => Ok(Spanned::new(value, span)),
        other => Err(invalid(key, other.span(), "a string atom")),
    }
}
fn integer(raw: RawExpr, key: &str) -> Result<IntegerSyntax, MetadataError> {
    match raw {
        RawExpr::Atom(RawAtom::Int(n), span) => {
            IntegerSyntax::try_new(Expr::Atom(Atom::Int(n), span))
        }
        other => Err(invalid(key, other.span(), "an integer atom")),
    }
}
fn parts(
    raw: RawExpr,
    tag: DeepTag,
    key: &str,
) -> Result<(Container, Vec<RawExpr>), MetadataError> {
    let span = raw.span();
    let RawExpr::List(mut items, _) = raw else {
        return Err(invalid(key, span, "the declared structural node"));
    };
    if items.len() < 2
        || !matches!(&items[0], RawExpr::Atom(RawAtom::Symbol(s), _) if s == tag.as_str())
    {
        return Err(invalid(key, span, "the declared structural node"));
    }
    let children = items.split_off(2);
    let RawExpr::Map(entries, _) = items.pop().expect("two items") else {
        return Err(invalid(key, span, "a structural node with annotations"));
    };
    let metadata = decode_entries(entries)?;
    crate::metadata::validate_typed_container(tag, &metadata)?;
    Ok((Container { metadata, span }, children))
}
fn variable(raw: RawExpr) -> Result<VariableRef, MetadataError> {
    let (container, children) = parts(raw, DeepTag::Var, "wrt")?;
    let [child]: [RawExpr; 1] = children
        .try_into()
        .map_err(|_| invalid("wrt", container.span, "one variable name"))?;
    Ok(VariableRef {
        container,
        name: name(child, "wrt")?,
    })
}
fn binder(raw: RawExpr) -> Result<PropertyBinder, MetadataError> {
    let span = raw.span();
    match raw {
        RawExpr::Atom(RawAtom::Symbol(value), _) => Ok(PropertyBinder {
            name: Spanned::new(value, span),
            container: Container {
                metadata: Metadata::default(),
                span,
            },
            spelling: BinderSpelling::Name,
            metadata_span: span,
        }),
        RawExpr::List(items, _) => {
            let [n, m]: [RawExpr; 2] = items.try_into().map_err(|_| {
                invalid(
                    "property_quantifiers",
                    span,
                    "a binder name and annotation map",
                )
            })?;
            let RawExpr::Map(entries, metadata_span) = m else {
                return Err(invalid(
                    "property_quantifiers",
                    span,
                    "a binder annotation map",
                ));
            };
            let mut result = PropertyBinder::new(
                name(n, "property_quantifiers")?,
                decode_entries(entries)?,
                span,
            )?;
            result.spelling = BinderSpelling::Pair;
            result.metadata_span = metadata_span;
            Ok(result)
        }
        RawExpr::MetaExpr { entries, expr, .. } => {
            let mut result = PropertyBinder::new(
                name(*expr, "property_quantifiers")?,
                decode_entries(entries)?,
                span,
            )?;
            result.spelling = BinderSpelling::Prefix;
            Ok(result)
        }
        _ => Err(invalid(
            "property_quantifiers",
            span,
            "a binder name with optional annotations",
        )),
    }
}
fn effects(raw: RawExpr, key: &str) -> Result<EffectSet, MetadataError> {
    let (container, children) = parts(raw, DeepTag::Effects, key)?;
    let mut values = Vec::with_capacity(children.len());
    for child in children {
        if matches!(&child, RawExpr::Atom(RawAtom::Symbol(_), _)) {
            values.push(EffectMember::Name(name(child, key)?));
        } else {
            let (container, children) = parts(child, DeepTag::Resource, key)?;
            let [child]: [RawExpr; 1] = children
                .try_into()
                .map_err(|_| invalid(key, container.span, "one resource string"))?;
            values.push(EffectMember::Resource(ResourceEffect {
                container,
                name: string(child, key)?,
            }));
        }
    }
    Ok(EffectSet { container, values })
}

pub(crate) fn decode_entries(entries: Vec<(String, RawExpr)>) -> Result<Metadata, MetadataError> {
    let mut meta = Metadata::default();
    for (key, raw) in entries {
        if let Some(kind) = MetadataKey::decode(&key) {
            crate::metadata::validate_raw_payload(&key, &raw)?;
            meta.insert(decode_value(kind, raw)?)?;
        } else {
            let value = match raw {
                RawExpr::ExtensionData(data) => data,
                raw => {
                    let span = raw.span();
                    crate::ExtensionData::from_raw(&raw).map_err(|error| {
                        let mut invalid = invalid(&key, span, "well-formed extension data");
                        invalid.detail = Some(error.to_string());
                        invalid
                    })?
                }
            };
            meta.extensions_mut().insert(key, value)?;
        }
    }
    Ok(meta)
}

fn decode_value(key: MetadataKey, raw: RawExpr) -> Result<MetadataValue, MetadataError> {
    use MetadataKey as K;
    use MetadataValue as V;
    let span = raw.span();
    let spelling = key.spelling();
    macro_rules! choice {
        ($variant:ident, $ty:ty, $read:ident) => {{
            let s = $read(raw, spelling)?;
            V::$variant(Spanned::new(
                <$ty>::decode(s.value())
                    .ok_or_else(|| invalid(spelling, span, "a declared choice"))?,
                span,
            ))
        }};
    }
    Ok(match key {
        K::Type => V::Type(TypeSyntax::try_new(
            crate::stamp_to_typed::stamp_serialized_type(raw)
                .map_err(|e| stamp_error(spelling, e))?,
        )?),
        K::Loc => {
            let RawExpr::List(items, _) = raw else {
                return Err(invalid(spelling, span, "(loc file line column)"));
            };
            let [head, file, line, column]: [RawExpr; 4] = items
                .try_into()
                .map_err(|_| invalid(spelling, span, "(loc file line column)"))?;
            let mut location = SourceLocation::new(
                string(file, spelling)?,
                integer(line, spelling)?,
                integer(column, spelling)?,
                span,
            );
            location.head_span = head.span();
            V::Loc(location)
        }
        K::Eff => V::Eff(effects(raw, spelling)?),
        K::Effects => V::Effects(effects(raw, spelling)?),
        K::DtypeBounds => {
            let RawExpr::Map(entries, _) = raw else {
                return Err(invalid(spelling, span, "a dtype binder map"));
            };
            let mut bounds = Vec::with_capacity(entries.len());
            for (binder, bound) in entries {
                // spec/03 §2.2: a family atom, or a list of member spellings
                // for §5.9's explicit set form.
                let decoded = match &bound {
                    RawExpr::List(members, list_span) => {
                        let mut set = std::collections::BTreeSet::new();
                        if members.is_empty() {
                            return Err(invalid(spelling, *list_span, "a non-empty dtype set"));
                        }
                        for member in members {
                            let member = name(member.clone(), spelling)?;
                            let dtype =
                                crate::BoundDtype::from_name(member.value()).ok_or_else(|| {
                                    invalid(spelling, member.span(), "an active §1.1 dtype")
                                })?;
                            if !set.insert(dtype) {
                                return Err(invalid(
                                    spelling,
                                    member.span(),
                                    "a dtype set without repeated members",
                                ));
                            }
                        }
                        Spanned::new(crate::DtypeBound::Set(set), *list_span)
                    }
                    _ => {
                        let family = name(bound, spelling)?;
                        let kind = crate::DtypeFamily::from_deep_name(family.value()).ok_or_else(
                            || invalid(spelling, family.span(), "float, int, or numeric"),
                        )?;
                        Spanned::new(crate::DtypeBound::Family(kind), family.span())
                    }
                };
                bounds.push((binder, decoded));
            }
            V::DtypeBounds(DtypeBounds::try_new(bounds, span)?)
        }
        K::Source => {
            let RawExpr::List(mut items, _) = raw else {
                return Err(invalid(spelling, span, "a preserved macro invocation"));
            };
            if items.is_empty() {
                return Err(invalid(spelling, span, "a preserved macro name"));
            }
            let arguments = items.split_off(1);
            V::Source(MacroSource::new(
                name(items.pop().expect("one head"), spelling)?,
                arguments,
                span,
            ))
        }
        K::Wrt => {
            if matches!(&raw, RawExpr::List(items, _) if matches!(items.first(), Some(RawExpr::Atom(RawAtom::Symbol(s), _)) if s == "var"))
            {
                V::Wrt(WrtTargets::Variable(variable(raw)?))
            } else {
                let (container, children) = parts(raw, DeepTag::Tuple, spelling)?;
                let mut children = children.into_iter();
                let first =
                    variable(children.next().ok_or_else(|| {
                        invalid(spelling, span, "a nonempty tuple of variables")
                    })?)?;
                let rest = children.map(variable).collect::<Result<_, _>>()?;
                V::Wrt(WrtTargets::Tuple(VariableTuple {
                    container,
                    first,
                    rest,
                }))
            }
        }
        K::Span => V::Span(SpanId::try_new(string(raw, spelling)?.into_value(), span)?),
        K::ChelisRole => V::ChelisRole(string(raw, spelling)?),
        K::PropertySourceKind => choice!(PropertySourceKind, PropertySourceKind, string),
        K::PropertyQuantifiers => {
            let (container, children) = parts(raw, DeepTag::Params, spelling)?;
            V::PropertyQuantifiers(PropertyQuantifiers {
                container,
                values: children.into_iter().map(binder).collect::<Result<_, _>>()?,
            })
        }
        K::PropertyPreconditions => {
            let (container, children) = parts(raw, DeepTag::Tuple, spelling)?;
            V::PropertyPreconditions(PropertyPreconditions {
                container,
                values: children
                    .into_iter()
                    .map(|v| runtime(spelling, v))
                    .collect::<Result<_, _>>()?,
            })
        }
        K::PropertyContracts => {
            let (container, children) = parts(raw, DeepTag::Tuple, spelling)?;
            V::PropertyContracts(PropertyContracts {
                container,
                values: children
                    .into_iter()
                    .map(|v| string(v, spelling))
                    .collect::<Result<_, _>>()?,
            })
        }
        K::PropertySourceId => V::PropertySourceId(string(raw, spelling)?),
        K::PropertyTolerance => V::PropertyTolerance(runtime(spelling, raw)?),
        K::PropertySeed => V::PropertySeed(runtime(spelling, raw)?),
        K::PropertySamples => V::PropertySamples(runtime(spelling, raw)?),
        K::Opaque => V::Opaque(Present::new(span)),
        K::Destructure => V::Destructure(Present::new(span)),
        K::Invariant => {
            let (container, children) = parts(raw, DeepTag::Fn, spelling)?;
            let [params, body]: [RawExpr; 2] = children
                .try_into()
                .map_err(|_| invalid(spelling, span, "a one-binder function"))?;
            let (params, children) = parts(params, DeepTag::Params, spelling)?;
            let [parameter]: [RawExpr; 1] = children
                .try_into()
                .map_err(|_| invalid(spelling, span, "one invariant binder"))?;
            V::Invariant(InvariantPredicate {
                container,
                params,
                binder: binder(parameter)?,
                body: runtime(spelling, body)?,
            })
        }
        K::InvariantAmenability => choice!(InvariantAmenability, Amenability, string),
        K::SurfPath => V::SurfPath(string(raw, spelling)?),
        K::SurfDimGroupSize => V::SurfDimGroupSize(PositiveInteger::try_new(
            integer(raw, spelling)?.expression().clone(),
        )?),
        K::SurfPipeStage => choice!(SurfPipeStage, PipeStageOrigin, string),
        K::SurfLiteralStyle => choice!(SurfLiteralStyle, LiteralStyle, string),
        K::SurfBindingType => choice!(SurfBindingType, BindingTypeOrigin, string),
        K::Lin => choice!(Lin, Linearity, name),
        K::Doc => V::Doc(string(raw, spelling)?),
        K::Effect => {
            let symbol = name(raw, spelling)?;
            let kind = chelis_vocab::EffectKind::decode(chelis_vocab::EffectKindInput::Symbol(
                symbol.value(),
            ))
            .map_err(|_| invalid(spelling, span, "resource"))?;
            V::Effect(Spanned::new(kind, span))
        }
        K::LiteralSource => choice!(LiteralSource, LiteralOrigin, name),
    })
}

impl Expr {
    /// Export syntax at a serialization boundary. Raw output must pass stamping
    /// again before it can re-enter the typed AST.
    pub fn to_raw(&self) -> RawExpr {
        WireExpr::from_ast(self).into_raw()
    }
}

impl MacroSource {
    /// Capture a programmatic invocation as historical syntax data. The
    /// arguments are preserved without annotation admission or rewriting.
    pub fn try_from_expression(expr: &Expr) -> Result<Self, MetadataError> {
        let raw = WireExpr::from_ast(expr).into_raw();
        crate::metadata::validate_raw_payload("source", &raw)?;
        match decode_value(MetadataKey::Source, raw)? {
            MetadataValue::Source(source) => Ok(source),
            _ => unreachable!("source decoder has a dedicated result"),
        }
    }
}

// This shadow is the old public serde shape. It exists only while encoding or
// decoding: invalid annotation entries never become an Expr::Map.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) enum WireExpr {
    Atom(Atom, Span),
    Map(WireMetadata, Span),
    MetaExpr(WireMetaExpr, Span),
    Node(Box<WireNode>, Span),
    BareList(Vec<WireExpr>, Span),
    UnknownForm(Box<WireUnknown>),
    ExtensionData(crate::ExtensionData),
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct WireMetadata {
    pub(crate) entries: Vec<(String, WireExpr)>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct WireMetaExpr {
    pub(crate) entries: Vec<(String, WireExpr)>,
    pub(crate) expr: Box<WireExpr>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct WireNode {
    pub(crate) tag: DeepTag,
    pub(crate) meta: WireMetadata,
    pub(crate) children: Vec<WireExpr>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct WireUnknown {
    pub(crate) head: String,
    pub(crate) meta: WireMetadata,
    pub(crate) children: Vec<WireExpr>,
    pub(crate) span: Span,
}

impl WireExpr {
    fn into_raw(self) -> RawExpr {
        match self {
            Self::ExtensionData(data) => RawExpr::ExtensionData(data),
            Self::Atom(atom, span) => RawExpr::Atom(
                match atom {
                    Atom::Name(v) => RawAtom::Symbol(v),
                    Atom::Int(v) => RawAtom::Int(v),
                    Atom::Float(v) => RawAtom::Float(v),
                    Atom::Str(v) => RawAtom::Str(v),
                    Atom::Bool(v) => RawAtom::Bool(v),
                },
                span,
            ),
            Self::BareList(v, span) => {
                RawExpr::List(v.into_iter().map(Self::into_raw).collect(), span)
            }
            Self::Map(v, span) => RawExpr::Map(v.into_raw(), span),
            Self::MetaExpr(v, span) => RawExpr::MetaExpr {
                entries: WireMetadata { entries: v.entries }.into_raw(),
                expr: Box::new(v.expr.into_raw()),
                span,
            },
            Self::Node(v, span) => {
                let mut children = vec![
                    RawExpr::Atom(RawAtom::Symbol(v.tag.as_str().into()), span),
                    RawExpr::Map(v.meta.into_raw(), span),
                ];
                children.extend(v.children.into_iter().map(Self::into_raw));
                RawExpr::List(children, span)
            }
            Self::UnknownForm(v) => {
                let mut children = vec![
                    RawExpr::Atom(RawAtom::Symbol(v.head), v.span),
                    RawExpr::Map(v.meta.into_raw(), v.span),
                ];
                children.extend(v.children.into_iter().map(Self::into_raw));
                RawExpr::List(children, v.span)
            }
        }
    }
}
// Serde preserves the carrier of genuine expression leaves. It is not a
// parser: a node is rebuilt through `Node::try_new`, never restamped.
impl WireExpr {
    fn into_ast(self, key: &str) -> Result<Expr, MetadataError> {
        Ok(match self {
            Self::ExtensionData(data) => {
                return Err(invalid(
                    key,
                    data.span(),
                    "program syntax, not extension data",
                ));
            }
            Self::Atom(atom, span) => Expr::Atom(atom, span),
            Self::Map(map, span) => Expr::Map(map.into_metadata()?, span),
            Self::MetaExpr(meta, span) => Expr::MetaExpr(
                crate::MetaExpr {
                    metadata: WireMetadata {
                        entries: meta.entries,
                    }
                    .into_metadata()?,
                    expr: Box::new(meta.expr.into_ast(key)?),
                },
                span,
            ),
            Self::BareList(items, span) => Expr::BareList(
                items
                    .into_iter()
                    .map(|v| v.into_ast(key))
                    .collect::<Result<_, _>>()?,
                span,
            ),
            Self::Node(node, span) => {
                let metadata = node.meta.into_metadata()?;
                let children = node
                    .children
                    .into_iter()
                    .map(|v| v.into_ast(key))
                    .collect::<Result<_, _>>()?;
                let node =
                    crate::node::Node::try_new(node.tag, metadata, children).map_err(|error| {
                        let mut result = invalid(key, span, "a valid serialized annotation node");
                        result.detail = Some(error.to_string());
                        result
                    })?;
                Expr::Node(Box::new(node), span)
            }
            Self::UnknownForm(data) => Expr::UnknownForm(Box::new(crate::UnknownFormData {
                head: data.head,
                meta: data.meta.into_metadata()?,
                children: data
                    .children
                    .into_iter()
                    .map(|v| v.into_ast(key))
                    .collect::<Result<_, _>>()?,
                span: data.span,
            })),
        })
    }
}
impl WireMetadata {
    fn into_raw(self) -> Vec<(String, RawExpr)> {
        self.entries
            .into_iter()
            .map(|(key, value)| (key, value.into_raw()))
            .collect()
    }
    fn into_metadata(self) -> Result<Metadata, MetadataError> {
        let mut metadata = Metadata::default();
        for (key, value) in self.entries {
            if let Some(kind) = MetadataKey::decode(&key) {
                metadata.insert(decode_wire_value(kind, value)?)?;
            } else {
                let WireExpr::ExtensionData(data) = value else {
                    return Err(invalid(
                        &key,
                        Span::new(0, 0),
                        "the explicit opaque extension-data encoding; regenerate the serialized AST",
                    ));
                };
                metadata.extensions_mut().insert(key, data)?;
            }
        }
        Ok(metadata)
    }
}
fn decode_wire_value(key: MetadataKey, value: WireExpr) -> Result<MetadataValue, MetadataError> {
    use crate::annotations_transform::parts;
    use MetadataKey as K;
    use MetadataValue as V;
    let spelling = key.spelling();
    // These payloads are syntax data or scalars. Decode their declared shape
    // without constructing annotation-bearing AST maps for their contents.
    if !matches!(
        key,
        K::Type
            | K::PropertySeed
            | K::PropertySamples
            | K::PropertyTolerance
            | K::Eff
            | K::Effects
            | K::Wrt
            | K::PropertyQuantifiers
            | K::PropertyPreconditions
            | K::PropertyContracts
            | K::Invariant
    ) {
        let raw = value.into_raw();
        crate::metadata::validate_raw_payload(spelling, &raw)?;
        return decode_value(key, raw);
    }
    let expr = value.into_ast(spelling)?;
    let span = expr.span();
    Ok(match key {
        K::Type => V::Type(TypeSyntax::try_new(expr)?),
        K::PropertySeed => V::PropertySeed(RuntimeExpression::try_for_key(expr, spelling)?),
        K::PropertySamples => V::PropertySamples(RuntimeExpression::try_for_key(expr, spelling)?),
        K::PropertyTolerance => {
            V::PropertyTolerance(RuntimeExpression::try_for_key(expr, spelling)?)
        }
        K::Invariant => V::Invariant(InvariantPredicate::try_from_expression(expr)?),
        K::PropertyQuantifiers => {
            let (container, values) = parts(expr, DeepTag::Params, spelling)?;
            V::PropertyQuantifiers(PropertyQuantifiers {
                container,
                values: values
                    .into_iter()
                    .map(PropertyBinder::try_from_expression)
                    .collect::<Result<_, _>>()?,
            })
        }
        K::PropertyPreconditions => {
            let (container, values) = parts(expr, DeepTag::Tuple, spelling)?;
            V::PropertyPreconditions(PropertyPreconditions {
                container,
                values: values
                    .into_iter()
                    .map(|value| RuntimeExpression::try_for_key(value, spelling))
                    .collect::<Result<_, _>>()?,
            })
        }
        K::PropertyContracts => {
            let (container, values) = parts(expr, DeepTag::Tuple, spelling)?;
            let values = values
                .into_iter()
                .map(|value| match value {
                    Expr::Atom(Atom::Str(value), span) => Ok(Spanned::new(value, span)),
                    other => Err(invalid(
                        spelling,
                        other.span(),
                        "a string contract identifier",
                    )),
                })
                .collect::<Result<_, _>>()?;
            V::PropertyContracts(PropertyContracts { container, values })
        }
        K::Eff | K::Effects => {
            let (container, values) = parts(expr, DeepTag::Effects, spelling)?;
            let values = values
                .into_iter()
                .map(|value| {
                    Ok(match value {
                        Expr::Atom(Atom::Name(name), span) => {
                            EffectMember::Name(Spanned::new(name, span))
                        }
                        value => {
                            let (container, children) = parts(value, DeepTag::Resource, spelling)?;
                            let [Expr::Atom(Atom::Str(name), name_span)]: [Expr; 1] = children
                                .try_into()
                                .map_err(|_| invalid(spelling, span, "one resource string"))?
                            else {
                                return Err(invalid(spelling, span, "one resource string"));
                            };
                            EffectMember::Resource(ResourceEffect {
                                container,
                                name: Spanned::new(name, name_span),
                            })
                        }
                    })
                })
                .collect::<Result<_, MetadataError>>()?;
            let effects = EffectSet { container, values };
            if key == K::Eff {
                V::Eff(effects)
            } else {
                V::Effects(effects)
            }
        }
        K::Wrt if expr.tag() == Some(DeepTag::Var) => V::Wrt(WrtTargets::Variable(
            VariableRef::try_from_expression(expr)?,
        )),
        K::Wrt => {
            let (container, children) = parts(expr, DeepTag::Tuple, spelling)?;
            let mut variables = children.into_iter().map(VariableRef::try_from_expression);
            let first = variables
                .next()
                .ok_or_else(|| invalid(spelling, span, "a nonempty variable tuple"))??;
            V::Wrt(WrtTargets::Tuple(VariableTuple {
                container,
                first,
                rest: variables.collect::<Result<_, _>>()?,
            }))
        }
        _ => unreachable!("data and scalar variants decoded above"),
    })
}
impl<'de> Deserialize<'de> for Metadata {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        WireMetadata::deserialize(d)?
            .into_metadata()
            .map_err(serde::de::Error::custom)
    }
}
impl Serialize for crate::MetaExpr {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        #[derive(Serialize)]
        struct Prefix<'a> {
            entries: Vec<(String, WireExpr)>,
            expr: &'a Expr,
        }
        Prefix {
            entries: WireMetadata::from_metadata(&self.metadata).entries,
            expr: &self.expr,
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for crate::MetaExpr {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Prefix {
            entries: Vec<(String, WireExpr)>,
            expr: Box<Expr>,
        }
        let prefix = Prefix::deserialize(d)?;
        Ok(Self {
            metadata: WireMetadata {
                entries: prefix.entries,
            }
            .into_metadata()
            .map_err(serde::de::Error::custom)?,
            expr: prefix.expr,
        })
    }
}

fn atom(value: Atom, span: Span) -> WireExpr {
    WireExpr::Atom(value, span)
}
fn text(value: &Spanned<String>) -> WireExpr {
    atom(Atom::Str(value.value().clone()), value.span())
}
fn symbol(value: &Spanned<String>) -> WireExpr {
    atom(Atom::Name(value.value().clone()), value.span())
}
fn node(tag: DeepTag, container: &Container, children: Vec<WireExpr>) -> WireExpr {
    WireExpr::Node(
        Box::new(WireNode {
            tag,
            meta: WireMetadata::from_metadata(&container.metadata),
            children,
        }),
        container.span,
    )
}
fn variable_wire(v: &VariableRef) -> WireExpr {
    node(DeepTag::Var, &v.container, vec![symbol(&v.name)])
}
fn binder_wire(v: &PropertyBinder) -> WireExpr {
    match v.spelling {
        BinderSpelling::Name => symbol(&v.name),
        BinderSpelling::Pair => WireExpr::BareList(
            vec![
                symbol(&v.name),
                WireExpr::Map(WireMetadata::from_metadata(v.metadata()), v.metadata_span),
            ],
            v.span(),
        ),
        BinderSpelling::Prefix => WireExpr::MetaExpr(
            WireMetaExpr {
                entries: WireMetadata::from_metadata(v.metadata()).entries,
                expr: Box::new(symbol(&v.name)),
            },
            v.span(),
        ),
    }
}
fn effect_wire(v: &EffectMember) -> WireExpr {
    match v {
        EffectMember::Name(v) => symbol(v),
        EffectMember::Resource(v) => node(DeepTag::Resource, &v.container, vec![text(&v.name)]),
    }
}
impl WireMetadata {
    pub(crate) fn from_metadata(meta: &Metadata) -> Self {
        let mut entries: Vec<_> = meta
            .values()
            .map(|v| (v.key().spelling().to_string(), WireExpr::from_value(v)))
            .collect();
        entries.extend(
            meta.extensions()
                .iter()
                .map(|(k, v)| (k.to_string(), WireExpr::ExtensionData(v.clone()))),
        );
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));
        Self { entries }
    }
}
impl WireExpr {
    pub(crate) fn from_value(value: &MetadataValue) -> Self {
        use MetadataValue as V;
        let span = value.span();
        match value {
            V::Type(v) => Self::from_ast(v.expression()),
            V::PropertyTolerance(v) | V::PropertySeed(v) | V::PropertySamples(v) => {
                Self::from_ast(v.expression())
            }
            V::Loc(v) => Self::BareList(
                vec![
                    atom(Atom::Name("loc".into()), v.head_span),
                    text(&v.file),
                    Self::from_ast(v.line.expression()),
                    Self::from_ast(v.column.expression()),
                ],
                span,
            ),
            V::Eff(v) | V::Effects(v) => node(
                DeepTag::Effects,
                &v.container,
                v.values.iter().map(effect_wire).collect(),
            ),
            V::DtypeBounds(v) => Self::Map(
                WireMetadata {
                    entries: v
                        .bounds()
                        .map(|(k, v)| {
                            let encoded = match v.value() {
                                crate::DtypeBound::Family(family) => {
                                    atom(Atom::Name(family.deep_name().into()), v.span())
                                }
                                crate::DtypeBound::Set(members) => WireExpr::BareList(
                                    members
                                        .iter()
                                        .map(|dtype| {
                                            atom(Atom::Name(dtype.name().into()), v.span())
                                        })
                                        .collect(),
                                    v.span(),
                                ),
                            };
                            (k.to_string(), encoded)
                        })
                        .collect(),
                },
                span,
            ),
            V::Source(v) => {
                let mut items = vec![symbol(&v.name)];
                items.extend(v.arguments.iter().map(Self::from_raw_data));
                Self::BareList(items, span)
            }
            V::Wrt(WrtTargets::Variable(v)) => variable_wire(v),
            V::Wrt(WrtTargets::Tuple(v)) => node(
                DeepTag::Tuple,
                &v.container,
                v.variables().map(variable_wire).collect(),
            ),
            V::Span(v) => atom(Atom::Str(v.value().into()), span),
            V::ChelisRole(v) | V::PropertySourceId(v) | V::SurfPath(v) | V::Doc(v) => text(v),
            V::PropertySourceKind(v) => atom(Atom::Str(v.value().spelling().into()), span),
            V::PropertyQuantifiers(v) => node(
                DeepTag::Params,
                &v.container,
                v.values.iter().map(binder_wire).collect(),
            ),
            V::PropertyPreconditions(v) => node(
                DeepTag::Tuple,
                &v.container,
                v.values
                    .iter()
                    .map(|v| Self::from_ast(v.expression()))
                    .collect(),
            ),
            V::PropertyContracts(v) => node(
                DeepTag::Tuple,
                &v.container,
                v.values.iter().map(text).collect(),
            ),
            V::Opaque(_) | V::Destructure(_) => atom(Atom::Bool(true), span),
            V::Invariant(v) => node(
                DeepTag::Fn,
                &v.container,
                vec![
                    node(DeepTag::Params, &v.params, vec![binder_wire(&v.binder)]),
                    Self::from_ast(v.body.expression()),
                ],
            ),
            V::InvariantAmenability(v) => atom(Atom::Str(v.value().spelling().into()), span),
            V::SurfDimGroupSize(v) => Self::from_ast(v.expression()),
            V::SurfPipeStage(v) => atom(Atom::Str(v.value().spelling().into()), span),
            V::SurfLiteralStyle(v) => atom(Atom::Str(v.value().spelling().into()), span),
            V::SurfBindingType(v) => atom(Atom::Str(v.value().spelling().into()), span),
            V::Lin(v) => atom(Atom::Name(v.value().spelling().into()), span),
            V::Effect(v) => atom(Atom::Name(v.value().symbol().into()), span),
            V::LiteralSource(v) => atom(Atom::Name(v.value().spelling().into()), span),
        }
    }
    fn from_raw_data(raw: &RawExpr) -> Self {
        match raw {
            RawExpr::ExtensionData(data) => Self::ExtensionData(data.clone()),
            RawExpr::Atom(a, span) => atom(
                match a {
                    RawAtom::Symbol(v) => Atom::Name(v.clone()),
                    RawAtom::Int(v) => Atom::Int(*v),
                    RawAtom::Float(v) => Atom::Float(*v),
                    RawAtom::Str(v) => Atom::Str(v.clone()),
                    RawAtom::Bool(v) => Atom::Bool(*v),
                },
                *span,
            ),
            RawExpr::List(items, span) => {
                Self::BareList(items.iter().map(Self::from_raw_data).collect(), *span)
            }
            RawExpr::Map(entries, span) => Self::Map(
                WireMetadata {
                    entries: entries
                        .iter()
                        .map(|(k, v)| (k.clone(), Self::from_raw_data(v)))
                        .collect(),
                },
                *span,
            ),
            RawExpr::MetaExpr {
                entries,
                expr,
                span,
            } => Self::MetaExpr(
                WireMetaExpr {
                    entries: entries
                        .iter()
                        .map(|(k, v)| (k.clone(), Self::from_raw_data(v)))
                        .collect(),
                    expr: Box::new(Self::from_raw_data(expr)),
                },
                *span,
            ),
        }
    }
    /// Convert a Deep expression to its wire shadow without recursing once per
    /// nesting level: the spine (`MetaExpr` bodies, `Node`, `BareList` and
    /// `UnknownForm` children) is walked on a heap stack, so the checker's
    /// canonical printing cannot overflow the native stack on a deeply nested
    /// program (chelis#2424). Metadata values still convert through
    /// [`WireExpr::from_value`], whose recursion follows metadata nesting, not
    /// the depth of the tree.
    pub(crate) fn from_ast(root: &Expr) -> Self {
        /// A carrier whose children are being converted.
        enum Pending {
            MetaExpr(Vec<(String, WireExpr)>, Span),
            Node(DeepTag, WireMetadata, usize, Span),
            BareList(usize, Span),
            UnknownForm(String, WireMetadata, usize, Span),
        }
        enum Step<'a> {
            Enter(&'a Expr),
            Finish(Pending),
        }

        let mut steps = vec![Step::Enter(root)];
        let mut values: Vec<Self> = Vec::new();
        while let Some(step) = steps.pop() {
            match step {
                Step::Enter(expr) => match expr {
                    Expr::Atom(a, span) => values.push(Self::Atom(a.clone(), *span)),
                    Expr::Map(v, span) => {
                        values.push(Self::Map(WireMetadata::from_metadata(v), *span));
                    }
                    Expr::MetaExpr(v, span) => {
                        let entries = WireMetadata::from_metadata(&v.metadata).entries;
                        steps.push(Step::Finish(Pending::MetaExpr(entries, *span)));
                        steps.push(Step::Enter(&v.expr));
                    }
                    Expr::Node(v, span) => {
                        let children = v.children_slice();
                        let meta = WireMetadata::from_metadata(v.meta());
                        steps.push(Step::Finish(Pending::Node(
                            v.tag(),
                            meta,
                            children.len(),
                            *span,
                        )));
                        steps.extend(children.iter().rev().map(Step::Enter));
                    }
                    Expr::BareList(v, span) => {
                        steps.push(Step::Finish(Pending::BareList(v.len(), *span)));
                        steps.extend(v.iter().rev().map(Step::Enter));
                    }
                    Expr::UnknownForm(v) => {
                        let meta = WireMetadata::from_metadata(&v.meta);
                        steps.push(Step::Finish(Pending::UnknownForm(
                            v.head.clone(),
                            meta,
                            v.children.len(),
                            v.span,
                        )));
                        steps.extend(v.children.iter().rev().map(Step::Enter));
                    }
                },
                Step::Finish(pending) => {
                    let value = match pending {
                        Pending::MetaExpr(entries, span) => {
                            let expr = values.pop().expect("a MetaExpr body was converted");
                            Self::MetaExpr(
                                WireMetaExpr {
                                    entries,
                                    expr: Box::new(expr),
                                },
                                span,
                            )
                        }
                        Pending::Node(tag, meta, len, span) => {
                            let children = values.split_off(values.len() - len);
                            Self::Node(
                                Box::new(WireNode {
                                    tag,
                                    meta,
                                    children,
                                }),
                                span,
                            )
                        }
                        Pending::BareList(len, span) => {
                            Self::BareList(values.split_off(values.len() - len), span)
                        }
                        Pending::UnknownForm(head, meta, len, span) => {
                            let children = values.split_off(values.len() - len);
                            Self::UnknownForm(Box::new(WireUnknown {
                                head,
                                meta,
                                children,
                                span,
                            }))
                        }
                    };
                    values.push(value);
                }
            }
        }
        values
            .pop()
            .expect("from_ast converts exactly one expression")
    }

    /// Drop a wire tree without recursing once per nesting level, as the
    /// derived drop glue would. Every nested value is moved onto a heap stack
    /// before its parent is dropped, so each drop is shallow (chelis#2424).
    pub(crate) fn dispose(self) {
        let mut pending = vec![self];
        while let Some(expr) = pending.pop() {
            match expr {
                Self::Map(map, _) => pending.extend(map.entries.into_iter().map(|(_, v)| v)),
                Self::MetaExpr(meta, _) => {
                    pending.extend(meta.entries.into_iter().map(|(_, v)| v));
                    pending.push(*meta.expr);
                }
                Self::Node(node, _) => {
                    let WireNode { meta, children, .. } = *node;
                    pending.extend(meta.entries.into_iter().map(|(_, v)| v));
                    pending.extend(children);
                }
                Self::BareList(items, _) => pending.extend(items),
                Self::UnknownForm(form) => {
                    let WireUnknown { meta, children, .. } = *form;
                    pending.extend(meta.entries.into_iter().map(|(_, v)| v));
                    pending.extend(children);
                }
                Self::Atom(..) | Self::ExtensionData(_) => {}
            }
        }
    }
}
impl Serialize for Metadata {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        WireMetadata::from_metadata(self).serialize(s)
    }
}
