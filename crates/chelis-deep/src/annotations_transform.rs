//! Typed rebuilding of annotation payloads. Structural roots are never passed
//! to expression transformations; genuine variable references retain Var shape.
use crate::annotations::*;
use crate::{
    Atom, DeepTag, Expr, MetaExpr,
    metadata::{MetadataError, MetadataRole},
};

pub(crate) fn parts(
    expr: Expr,
    tag: DeepTag,
    key: &str,
) -> Result<(Container, Vec<Expr>), MetadataError> {
    let span = expr.span();
    let (actual, metadata, children) = match expr {
        Expr::Node(node, _) => node.into_parts(),
        _ => return Err(invalid(key, span, "the declared structural node")),
    };
    if actual != tag {
        return Err(invalid(key, span, "the declared structural node"));
    }
    crate::metadata::validate_typed_container(tag, &metadata)?;
    Ok((Container { metadata, span }, children))
}

impl VariableRef {
    pub fn to_expression(&self) -> Expr {
        Expr::node(
            DeepTag::Var,
            self.metadata().clone(),
            vec![Expr::Atom(
                Atom::Name(self.name().value().clone()),
                self.name().span(),
            )],
            self.span(),
        )
    }
    pub fn try_from_expression(expr: Expr) -> Result<Self, MetadataError> {
        let (container, children) = parts(expr, DeepTag::Var, "wrt")?;
        let [Expr::Atom(Atom::Name(name), name_span)]: [Expr; 1] = children
            .try_into()
            .map_err(|_| invalid("wrt", container.span, "one variable name"))?
        else {
            return Err(invalid("wrt", container.span, "one variable name"));
        };
        Ok(Self {
            container,
            name: Spanned::new(name, name_span),
        })
    }
}
impl PropertyBinder {
    pub fn to_expression(&self) -> Expr {
        let name = Expr::Atom(Atom::Name(self.name().value().clone()), self.name().span());
        match self.spelling {
            BinderSpelling::Name => name,
            BinderSpelling::Pair => Expr::BareList(
                vec![name, Expr::Map(self.metadata().clone(), self.metadata_span)],
                self.span(),
            ),
            BinderSpelling::Prefix => Expr::MetaExpr(
                MetaExpr {
                    metadata: self.metadata().clone(),
                    expr: Box::new(name),
                },
                self.span(),
            ),
        }
    }
    pub fn try_from_expression(expr: Expr) -> Result<Self, MetadataError> {
        let span = expr.span();
        let (name, metadata, spelling, metadata_span) = match expr {
            name @ Expr::Atom(Atom::Name(_), _) => {
                (name, Metadata::default(), BinderSpelling::Name, span)
            }
            Expr::MetaExpr(meta, _) => (*meta.expr, meta.metadata, BinderSpelling::Prefix, span),
            Expr::BareList(items, _) => {
                let [name, Expr::Map(metadata, metadata_span)]: [Expr; 2] =
                    items.try_into().map_err(|_| {
                        invalid("property_quantifiers", span, "a name and annotation map")
                    })?
                else {
                    return Err(invalid(
                        "property_quantifiers",
                        span,
                        "a name and annotation map",
                    ));
                };
                (name, metadata, BinderSpelling::Pair, metadata_span)
            }
            _ => return Err(invalid("property_quantifiers", span, "a binder")),
        };
        let Expr::Atom(Atom::Name(name), name_span) = name else {
            return Err(invalid("property_quantifiers", span, "a binder name"));
        };
        let mut result = Self::new(Spanned::new(name, name_span), metadata, span)?;
        result.spelling = spelling;
        result.metadata_span = metadata_span;
        Ok(result)
    }
}

impl InvariantPredicate {
    pub fn to_expression(&self) -> Expr {
        let params = Expr::node(
            DeepTag::Params,
            self.parameter_metadata().clone(),
            vec![self.binder.to_expression()],
            self.params.span,
        );
        Expr::node(
            DeepTag::Fn,
            self.metadata().clone(),
            vec![params, self.body.expression().clone()],
            self.span(),
        )
    }
    pub fn try_from_expression(expr: Expr) -> Result<Self, MetadataError> {
        let (container, children) = parts(expr, DeepTag::Fn, "invariant")?;
        let [params, body]: [Expr; 2] = children
            .try_into()
            .map_err(|_| invalid("invariant", container.span, "a one-binder function"))?;
        let (params, children) = parts(params, DeepTag::Params, "invariant")?;
        let [binder]: [Expr; 1] = children
            .try_into()
            .map_err(|_| invalid("invariant", params.span, "one binder"))?;
        Ok(Self {
            container,
            params,
            binder: PropertyBinder::try_from_expression(binder)?,
            body: RuntimeExpression::try_for_key(body, "invariant")?,
        })
    }
}

/// The owner of an expression visited by a syntax analysis. Registered names
/// derive from the typed key, while extension names belong to their producer.
#[derive(Debug, Clone, Copy)]
pub enum MetadataName {
    Core(MetadataKey),
}
impl MetadataName {
    pub fn spelling(&self) -> &str {
        match self {
            Self::Core(k) => k.spelling(),
        }
    }
}

impl Expr {
    /// Transfer an originating owner's opaque annotations to its replacement.
    /// Both inputs retain their original state if a key conflicts.
    pub fn try_inherit_extensions(self, owner: &Expr) -> Result<Self, MetadataError> {
        let extensions = match owner {
            Expr::Node(node, _) => node.meta().extensions(),
            Expr::Map(meta, _) => meta.extensions(),
            Expr::MetaExpr(meta, _) => meta.metadata.extensions(),
            Expr::UnknownForm(data) => data.meta.extensions(),
            Expr::BareList(items, _) => match items.as_slice() {
                [_, Expr::Map(meta, _)] => meta.extensions(),
                _ => return Ok(self),
            },
            Expr::Atom(..) => return Ok(self),
        };
        if extensions.is_empty() {
            return Ok(self);
        }
        let mut candidate = self;
        let meta = match &mut candidate {
            Expr::Node(node, _) => {
                let mut meta = node.meta().clone();
                meta.extensions_mut().try_merge(extensions)?;
                node.try_replace_meta(meta).map_err(|error| {
                    let mut detail =
                        invalid("extension", owner.span(), "a valid replacement owner");
                    detail.detail = Some(error.to_string());
                    detail
                })?;
                return Ok(candidate);
            }
            Expr::Map(meta, _) => Some(meta),
            Expr::MetaExpr(meta, _) => Some(&mut meta.metadata),
            Expr::UnknownForm(data) => Some(&mut data.meta),
            Expr::BareList(items, _) => match items.as_mut_slice() {
                [_, Expr::Map(meta, _)] => Some(meta),
                _ => None,
            },
            Expr::Atom(..) => None,
        };
        if let Some(meta) = meta {
            meta.extensions_mut().try_merge(extensions)?;
            Ok(candidate)
        } else {
            let mut metadata = Metadata::default();
            metadata.extensions_mut().try_merge(extensions)?;
            let span = candidate.span();
            Ok(Expr::MetaExpr(
                crate::MetaExpr {
                    metadata,
                    expr: Box::new(candidate),
                },
                span,
            ))
        }
    }
}

impl Metadata {
    pub fn any_syntax(&self, predicate: &mut impl FnMut(&Expr) -> bool) -> bool {
        let mut found = false;
        self.visit_syntax(&mut |_, value| {
            if !found {
                found = predicate(value);
            }
        });
        found
    }
    /// Read-only traversal for analyses that need binder scope and diagnostic
    /// paths. Structured snapshots cannot mutate the stored payload. Rewriters
    /// use `try_map_expressions`, which never exposes structural roots.
    pub fn visit_syntax(&self, f: &mut impl FnMut(MetadataName, &Expr)) {
        use MetadataValue as V;
        for value in self.values() {
            let key = MetadataName::Core(value.key());
            match value {
                V::Type(v) => f(key, v.expression()),
                V::Accumulator(v) => f(key, v.expression()),
                V::PropertyTolerance(v) | V::PropertySeed(v) | V::PropertySamples(v) => {
                    f(key, v.expression())
                }
                V::Wrt(WrtTargets::Variable(v)) => f(key, &v.to_expression()),
                V::Wrt(WrtTargets::Tuple(v)) => f(
                    key,
                    &Expr::node(
                        DeepTag::Tuple,
                        v.metadata().clone(),
                        v.variables().map(VariableRef::to_expression).collect(),
                        v.span(),
                    ),
                ),
                V::Invariant(v) => f(key, &v.to_expression()),
                V::PropertyQuantifiers(v) => f(
                    key,
                    &Expr::node(
                        DeepTag::Params,
                        v.metadata().clone(),
                        v.values.iter().map(PropertyBinder::to_expression).collect(),
                        v.span(),
                    ),
                ),
                V::PropertyPreconditions(v) => f(
                    key,
                    &Expr::node(
                        DeepTag::Tuple,
                        v.metadata().clone(),
                        v.values.iter().map(|v| v.expression().clone()).collect(),
                        v.span(),
                    ),
                ),
                V::PropertyContracts(v) => f(
                    key,
                    &Expr::node(
                        DeepTag::Tuple,
                        v.metadata().clone(),
                        v.values
                            .iter()
                            .map(|v| Expr::Atom(Atom::Str(v.value().clone()), v.span()))
                            .collect(),
                        v.span(),
                    ),
                ),
                V::Eff(v) | V::Effects(v) => {
                    let members = v
                        .values
                        .iter()
                        .map(|v| match v {
                            EffectMember::Name(v) => {
                                Expr::Atom(Atom::Name(v.value().clone()), v.span())
                            }
                            EffectMember::Resource(v) => Expr::node(
                                DeepTag::Resource,
                                v.metadata().clone(),
                                vec![Expr::Atom(
                                    Atom::Str(v.name().value().clone()),
                                    v.name().span(),
                                )],
                                v.span(),
                            ),
                        })
                        .collect();
                    f(
                        key,
                        &Expr::node(DeepTag::Effects, v.metadata().clone(), members, v.span()),
                    );
                }
                V::Source(_)
                | V::DtypeBounds(_)
                | V::Loc(_)
                | V::Span(_)
                | V::ChelisRole(_)
                | V::PropertySourceKind(_)
                | V::PropertySourceId(_)
                | V::Opaque(_)
                | V::InvariantAmenability(_)
                | V::SurfPath(_)
                | V::SurfDimGroupSize(_)
                | V::SurfLiteralStyle(_)
                | V::SurfBindingType(_)
                | V::Lin(_)
                | V::Doc(_)
                | V::Effect(_)
                | V::LiteralSource(_)
                | V::Destructure(_) => {}
            }
        }
    }
    /// Rebuild expression leaves with role information. Every changed payload
    /// is re-admitted before the candidate can replace its original metadata.
    pub fn try_map_expressions<E: From<MetadataError>>(
        &self,
        f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
    ) -> Result<Self, E> {
        self.try_map_payloads(
            &mut |value, role| Ok(f(value, role)?.try_inherit_extensions(value)?),
            true,
            &mut |metadata, _| Ok(metadata),
        )
    }

    /// Rebuild exactly the borrowed leaves yielded by `visit_expressions`, in
    /// the same order. This lets an iterative AST worklist own all traversal;
    /// binder and variable identities remain in their dedicated containers.
    pub fn try_map_leaves<E: From<MetadataError>>(
        &self,
        f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
    ) -> Result<Self, E> {
        self.try_map_payloads(
            &mut |value, role| Ok(f(value, role)?.try_inherit_extensions(value)?),
            false,
            &mut |metadata, _| Ok(metadata),
        )
    }
    /// Transform annotations attached to structural payloads separately from
    /// expressions. The callback receives the owning tag (or a detached binder)
    /// and the rebuilt annotations; placement is checked before publication.
    pub fn try_map_expressions_with_annotations<E: From<MetadataError>>(
        &self,
        f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
        annotations: &mut impl FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
    ) -> Result<Self, E> {
        self.try_map_payloads(
            &mut |value, role| Ok(f(value, role)?.try_inherit_extensions(value)?),
            true,
            annotations,
        )
    }

    fn try_map_payloads<E: From<MetadataError>>(
        &self,
        f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
        rewrite_bindings: bool,
        annotations: &mut impl FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
    ) -> Result<Self, E> {
        use MetadataRole as R;
        use MetadataValue as V;
        fn container<E: From<MetadataError>>(
            v: &Container,
            f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
            rewrite_bindings: bool,
            annotations: &mut impl FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
            owner: Option<DeepTag>,
        ) -> Result<Container, E> {
            let metadata = v
                .metadata
                .try_map_payloads(f, rewrite_bindings, annotations)?;
            let mut metadata = annotations(metadata, owner)?;
            metadata
                .extensions_mut()
                .try_merge(v.metadata.extensions())?;
            if let Some(owner) = owner {
                crate::metadata::validate_typed_container(owner, &metadata)?;
            } else {
                crate::metadata::validate_typed_detached(&metadata)?;
            }
            Ok(Container {
                metadata,
                span: v.span,
            })
        }
        fn variable<E: From<MetadataError>>(
            v: &VariableRef,
            f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
            rewrite_bindings: bool,
            annotations: &mut impl FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
        ) -> Result<VariableRef, E> {
            if !rewrite_bindings {
                return Ok(VariableRef {
                    container: container(&v.container, f, false, annotations, Some(DeepTag::Var))?,
                    name: v.name.clone(),
                });
            }
            Ok(VariableRef::try_from_expression(f(
                &v.to_expression(),
                R::Expression,
            )?)?)
        }
        fn binder<E: From<MetadataError>>(
            v: &PropertyBinder,
            f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
            rewrite_bindings: bool,
            annotations: &mut impl FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
        ) -> Result<PropertyBinder, E> {
            if !rewrite_bindings {
                return Ok(PropertyBinder {
                    container: container(&v.container, f, false, annotations, None)?,
                    name: v.name.clone(),
                    spelling: v.spelling,
                    metadata_span: v.metadata_span,
                });
            }
            Ok(PropertyBinder::try_from_expression(f(
                &v.to_expression(),
                R::Syntax,
            )?)?)
        }
        fn effects<E: From<MetadataError>>(
            v: &EffectSet,
            f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
            rewrite_bindings: bool,
            annotations: &mut impl FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
        ) -> Result<EffectSet, E> {
            Ok(EffectSet {
                container: container(
                    &v.container,
                    f,
                    rewrite_bindings,
                    annotations,
                    Some(DeepTag::Effects),
                )?,
                values: v
                    .values
                    .iter()
                    .map(|member| {
                        Ok(match member {
                            EffectMember::Name(name) => EffectMember::Name(name.clone()),
                            EffectMember::Resource(v) => EffectMember::Resource(ResourceEffect {
                                container: container(
                                    &v.container,
                                    f,
                                    rewrite_bindings,
                                    annotations,
                                    Some(DeepTag::Resource),
                                )?,
                                name: v.name.clone(),
                            }),
                        })
                    })
                    .collect::<Result<_, E>>()?,
            })
        }
        let mut result = Self::default();
        for value in self.values() {
            let rebuilt = match value {
                V::Type(v) => V::Type(TypeSyntax::try_new(f(v.expression(), R::Type)?)?),
                V::Accumulator(v) => {
                    V::Accumulator(AccumulatorSyntax::try_new(f(v.expression(), R::Type)?)?)
                }
                V::PropertyTolerance(v) => V::PropertyTolerance(RuntimeExpression::try_for_key(
                    f(v.expression(), R::Expression)?,
                    "property_tolerance",
                )?),
                V::PropertySeed(v) => V::PropertySeed(RuntimeExpression::try_for_key(
                    f(v.expression(), R::Expression)?,
                    "property_seed",
                )?),
                V::PropertySamples(v) => V::PropertySamples(RuntimeExpression::try_for_key(
                    f(v.expression(), R::Expression)?,
                    "property_samples",
                )?),
                V::Eff(v) => V::Eff(effects(v, f, rewrite_bindings, annotations)?),
                V::Effects(v) => V::Effects(effects(v, f, rewrite_bindings, annotations)?),
                V::Wrt(WrtTargets::Variable(v)) => V::Wrt(WrtTargets::Variable(variable(
                    v,
                    f,
                    rewrite_bindings,
                    annotations,
                )?)),
                V::Wrt(WrtTargets::Tuple(v)) => V::Wrt(WrtTargets::Tuple(VariableTuple {
                    container: container(
                        &v.container,
                        f,
                        rewrite_bindings,
                        annotations,
                        Some(DeepTag::Tuple),
                    )?,
                    first: variable(&v.first, f, rewrite_bindings, annotations)?,
                    rest: v
                        .rest
                        .iter()
                        .map(|v| variable(v, f, rewrite_bindings, annotations))
                        .collect::<Result<_, _>>()?,
                })),
                V::PropertyQuantifiers(v) => V::PropertyQuantifiers(PropertyQuantifiers {
                    container: container(
                        &v.container,
                        f,
                        rewrite_bindings,
                        annotations,
                        Some(DeepTag::Params),
                    )?,
                    values: v
                        .values
                        .iter()
                        .map(|v| binder(v, f, rewrite_bindings, annotations))
                        .collect::<Result<_, _>>()?,
                }),
                V::PropertyPreconditions(v) => V::PropertyPreconditions(PropertyPreconditions {
                    container: container(
                        &v.container,
                        f,
                        rewrite_bindings,
                        annotations,
                        Some(DeepTag::Tuple),
                    )?,
                    values: v
                        .values
                        .iter()
                        .map(|v| {
                            Ok(RuntimeExpression::try_for_key(
                                f(v.expression(), R::Expression)?,
                                "property_preconditions",
                            )?)
                        })
                        .collect::<Result<_, E>>()?,
                }),
                V::PropertyContracts(v) => V::PropertyContracts(PropertyContracts {
                    container: container(
                        &v.container,
                        f,
                        rewrite_bindings,
                        annotations,
                        Some(DeepTag::Tuple),
                    )?,
                    values: v.values.clone(),
                }),
                V::Invariant(v) if rewrite_bindings => V::Invariant(
                    InvariantPredicate::try_from_expression(f(&v.to_expression(), R::Expression)?)?,
                ),
                V::Invariant(v) => V::Invariant(InvariantPredicate {
                    container: container(&v.container, f, false, annotations, Some(DeepTag::Fn))?,
                    params: container(&v.params, f, false, annotations, Some(DeepTag::Params))?,
                    binder: binder(&v.binder, f, false, annotations)?,
                    body: RuntimeExpression::try_for_key(
                        f(v.body.expression(), R::Expression)?,
                        "invariant",
                    )?,
                }),
                V::Source(_)
                | V::DtypeBounds(_)
                | V::Loc(_)
                | V::Span(_)
                | V::ChelisRole(_)
                | V::PropertySourceKind(_)
                | V::PropertySourceId(_)
                | V::Opaque(_)
                | V::InvariantAmenability(_)
                | V::SurfPath(_)
                | V::SurfDimGroupSize(_)
                | V::SurfLiteralStyle(_)
                | V::SurfBindingType(_)
                | V::Lin(_)
                | V::Doc(_)
                | V::Effect(_)
                | V::LiteralSource(_)
                | V::Destructure(_) => value.clone(),
            };
            result.insert(rebuilt)?;
        }
        for (key, value) in self.extensions().iter() {
            result.extensions_mut().insert(key.into(), value.clone())?;
        }
        Ok(result)
    }

    pub fn map_expressions(
        &self,
        f: &mut impl FnMut(&Expr, MetadataRole) -> Expr,
    ) -> Result<Self, MetadataError> {
        self.try_map_expressions(&mut |v, role| Ok(f(v, role)))
    }
}
