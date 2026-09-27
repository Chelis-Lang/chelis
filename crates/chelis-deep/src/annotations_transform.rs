//! Typed rebuilding of annotation payloads. Structural roots are never passed
//! to expression transformations; genuine variable references retain Var shape.
use crate::annotations::*;
use crate::{
    Atom, DeepTag, Expr, List, MetaExpr,
    metadata::{MetadataError, MetadataRole},
};

/// Leaf protocol markers of [`LeafMapper`]: they keep the blanket impls for
/// fallible and infallible closures disjoint under coherence.
pub(crate) struct Fallible;
pub(crate) struct Infallible;

/// One leaf rewrite of `try_map_payloads`. Threading the same `&mut M`
/// through the rebuild (instead of wrapping each caller's callback in
/// adapter closures) keeps the mutable state of every caller at one level.
pub(crate) trait LeafMapper<E, K> {
    fn map_leaf(&mut self, value: &Expr, role: MetadataRole) -> Result<Expr, E>;
}

/// The callbacks of the `try_map_*` entry points.
impl<E, F: FnMut(&Expr, MetadataRole) -> Result<Expr, E>> LeafMapper<E, Fallible> for F {
    fn map_leaf(&mut self, value: &Expr, role: MetadataRole) -> Result<Expr, E> {
        self(value, role)
    }
}

/// The callback of `map_expressions`.
impl<F: FnMut(&Expr, MetadataRole) -> Expr> LeafMapper<MetadataError, Infallible> for F {
    fn map_leaf(&mut self, value: &Expr, role: MetadataRole) -> Result<Expr, MetadataError> {
        Ok(self(value, role))
    }
}

/// One mapped leaf, re-admitted: the replacement inherits the original
/// leaf's opaque annotations before it can enter the rebuilt metadata.
fn map_leaf_readmitted<E: From<MetadataError>, K, M: LeafMapper<E, K>>(
    f: &mut M,
    value: &Expr,
    role: MetadataRole,
) -> Result<Expr, E> {
    Ok(f.map_leaf(value, role)?.try_inherit_extensions(value)?)
}

/// The annotations callback of every entry point that does not transform
/// structural annotations.
fn keep_annotations<E>(metadata: Metadata, _owner: Option<DeepTag>) -> Result<Metadata, E> {
    Ok(metadata)
}

pub(crate) fn parts(
    expr: Expr,
    tag: DeepTag,
    key: &str,
) -> Result<(Container, Vec<Expr>), MetadataError> {
    let span = expr.span();
    let (actual, metadata, children) = match expr {
        Expr::Node(node, _) => node.into_parts(),
        Expr::List(mut list, _) => {
            if list.elements.len() < 2 {
                return Err(invalid(key, span, "the declared structural node"));
            }
            let children = list.elements.split_off(2);
            let Expr::Map(metadata, _) = list.elements.pop().expect("two elements") else {
                return Err(invalid(key, span, "structural annotations"));
            };
            let Expr::Atom(Atom::Tag(tag), _) = list.elements.pop().expect("one tag") else {
                return Err(invalid(key, span, "a decoded vocabulary tag"));
            };
            (tag, metadata, children)
        }
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
            Expr::BareList(items, _) | Expr::List(List { elements: items }, _) => {
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
            Expr::List(list, _) => match list.elements.get(1) {
                Some(Expr::Map(meta, _)) => meta.extensions(),
                _ => return Ok(self),
            },
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
            Expr::List(list, _) => match list.elements.get_mut(1) {
                Some(Expr::Map(meta, _)) => Some(meta),
                _ => None,
            },
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
                | V::SurfPipeStage(_)
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
        self.try_map_payloads::<E, Fallible, _, _>(f, true, &mut keep_annotations)
    }

    /// Rebuild exactly the borrowed leaves yielded by `visit_expressions`, in
    /// the same order. This lets an iterative AST worklist own all traversal;
    /// binder and variable identities remain in their dedicated containers.
    pub fn try_map_leaves<E: From<MetadataError>>(
        &self,
        f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
    ) -> Result<Self, E> {
        self.try_map_payloads::<E, Fallible, _, _>(f, false, &mut keep_annotations)
    }
    /// Transform annotations attached to structural payloads separately from
    /// expressions. The callback receives the owning tag (or a detached binder)
    /// and the rebuilt annotations; placement is checked before publication.
    pub fn try_map_expressions_with_annotations<E: From<MetadataError>>(
        &self,
        f: &mut impl FnMut(&Expr, MetadataRole) -> Result<Expr, E>,
        annotations: &mut impl FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
    ) -> Result<Self, E> {
        self.try_map_payloads::<E, Fallible, _, _>(f, true, annotations)
    }

    fn try_map_payloads<
        E: From<MetadataError>,
        K,
        M: LeafMapper<E, K>,
        A: FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
    >(
        &self,
        f: &mut M,
        rewrite_bindings: bool,
        annotations: &mut A,
    ) -> Result<Self, E> {
        use MetadataRole as R;
        use MetadataValue as V;
        fn container<
            E: From<MetadataError>,
            K,
            M: LeafMapper<E, K>,
            A: FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
        >(
            v: &Container,
            f: &mut M,
            rewrite_bindings: bool,
            annotations: &mut A,
            owner: Option<DeepTag>,
        ) -> Result<Container, E> {
            let metadata =
                v.metadata
                    .try_map_payloads::<E, K, M, A>(f, rewrite_bindings, annotations)?;
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
        fn variable<
            E: From<MetadataError>,
            K,
            M: LeafMapper<E, K>,
            A: FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
        >(
            v: &VariableRef,
            f: &mut M,
            rewrite_bindings: bool,
            annotations: &mut A,
        ) -> Result<VariableRef, E> {
            if !rewrite_bindings {
                return Ok(VariableRef {
                    container: container::<E, K, M, A>(
                        &v.container,
                        f,
                        false,
                        annotations,
                        Some(DeepTag::Var),
                    )?,
                    name: v.name.clone(),
                });
            }
            let value = map_leaf_readmitted::<E, K, M>(f, &v.to_expression(), R::Expression)?;
            Ok(VariableRef::try_from_expression(value)?)
        }
        fn variables<
            E: From<MetadataError>,
            K,
            M: LeafMapper<E, K>,
            A: FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
        >(
            vs: &[VariableRef],
            f: &mut M,
            rewrite_bindings: bool,
            annotations: &mut A,
        ) -> Result<Vec<VariableRef>, E> {
            let mut out = Vec::new();
            for v in vs.iter() {
                out.push(variable::<E, K, M, A>(v, f, rewrite_bindings, annotations)?);
            }
            Ok(out)
        }
        fn binder<
            E: From<MetadataError>,
            K,
            M: LeafMapper<E, K>,
            A: FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
        >(
            v: &PropertyBinder,
            f: &mut M,
            rewrite_bindings: bool,
            annotations: &mut A,
        ) -> Result<PropertyBinder, E> {
            if !rewrite_bindings {
                return Ok(PropertyBinder {
                    container: container::<E, K, M, A>(&v.container, f, false, annotations, None)?,
                    name: v.name.clone(),
                    spelling: v.spelling,
                    metadata_span: v.metadata_span,
                });
            }
            let value = map_leaf_readmitted::<E, K, M>(f, &v.to_expression(), R::Syntax)?;
            Ok(PropertyBinder::try_from_expression(value)?)
        }
        fn binders<
            E: From<MetadataError>,
            K,
            M: LeafMapper<E, K>,
            A: FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
        >(
            vs: &[PropertyBinder],
            f: &mut M,
            rewrite_bindings: bool,
            annotations: &mut A,
        ) -> Result<Vec<PropertyBinder>, E> {
            let mut out = Vec::new();
            for v in vs.iter() {
                out.push(binder::<E, K, M, A>(v, f, rewrite_bindings, annotations)?);
            }
            Ok(out)
        }
        fn effects<
            E: From<MetadataError>,
            K,
            M: LeafMapper<E, K>,
            A: FnMut(Metadata, Option<DeepTag>) -> Result<Metadata, E>,
        >(
            v: &EffectSet,
            f: &mut M,
            rewrite_bindings: bool,
            annotations: &mut A,
        ) -> Result<EffectSet, E> {
            let effects_container = container::<E, K, M, A>(
                &v.container,
                f,
                rewrite_bindings,
                annotations,
                Some(DeepTag::Effects),
            )?;
            let mut values = Vec::new();
            for member in v.values.iter() {
                values.push(match member {
                    EffectMember::Name(name) => EffectMember::Name(name.clone()),
                    EffectMember::Resource(v) => EffectMember::Resource(ResourceEffect {
                        container: container::<E, K, M, A>(
                            &v.container,
                            f,
                            rewrite_bindings,
                            annotations,
                            Some(DeepTag::Resource),
                        )?,
                        name: v.name.clone(),
                    }),
                });
            }
            Ok(EffectSet {
                container: effects_container,
                values,
            })
        }
        fn preconditions<E: From<MetadataError>, K, M: LeafMapper<E, K>>(
            vs: &[RuntimeExpression],
            f: &mut M,
        ) -> Result<Vec<RuntimeExpression>, E> {
            let mut out = Vec::new();
            for v in vs.iter() {
                out.push(RuntimeExpression::try_for_key(
                    map_leaf_readmitted::<E, K, M>(f, v.expression(), R::Expression)?,
                    "property_preconditions",
                )?);
            }
            Ok(out)
        }
        let mut result = Self::default();
        for value in self.values() {
            let rebuilt = match value {
                V::Type(v) => V::Type(TypeSyntax::try_new(map_leaf_readmitted::<E, K, M>(
                    f,
                    v.expression(),
                    R::Type,
                )?)?),
                V::PropertyTolerance(v) => V::PropertyTolerance(RuntimeExpression::try_for_key(
                    map_leaf_readmitted::<E, K, M>(f, v.expression(), R::Expression)?,
                    "property_tolerance",
                )?),
                V::PropertySeed(v) => V::PropertySeed(RuntimeExpression::try_for_key(
                    map_leaf_readmitted::<E, K, M>(f, v.expression(), R::Expression)?,
                    "property_seed",
                )?),
                V::PropertySamples(v) => V::PropertySamples(RuntimeExpression::try_for_key(
                    map_leaf_readmitted::<E, K, M>(f, v.expression(), R::Expression)?,
                    "property_samples",
                )?),
                V::Eff(v) => V::Eff(effects::<E, K, M, A>(v, f, rewrite_bindings, annotations)?),
                V::Effects(v) => {
                    V::Effects(effects::<E, K, M, A>(v, f, rewrite_bindings, annotations)?)
                }
                V::Wrt(WrtTargets::Variable(v)) => {
                    V::Wrt(WrtTargets::Variable(variable::<E, K, M, A>(
                        v,
                        f,
                        rewrite_bindings,
                        annotations,
                    )?))
                }
                V::Wrt(WrtTargets::Tuple(v)) => V::Wrt(WrtTargets::Tuple(VariableTuple {
                    container: container::<E, K, M, A>(
                        &v.container,
                        f,
                        rewrite_bindings,
                        annotations,
                        Some(DeepTag::Tuple),
                    )?,
                    first: variable::<E, K, M, A>(&v.first, f, rewrite_bindings, annotations)?,
                    rest: variables::<E, K, M, A>(&v.rest, f, rewrite_bindings, annotations)?,
                })),
                V::PropertyQuantifiers(v) => V::PropertyQuantifiers(PropertyQuantifiers {
                    container: container::<E, K, M, A>(
                        &v.container,
                        f,
                        rewrite_bindings,
                        annotations,
                        Some(DeepTag::Params),
                    )?,
                    values: binders::<E, K, M, A>(&v.values, f, rewrite_bindings, annotations)?,
                }),
                V::PropertyPreconditions(v) => V::PropertyPreconditions(PropertyPreconditions {
                    container: container::<E, K, M, A>(
                        &v.container,
                        f,
                        rewrite_bindings,
                        annotations,
                        Some(DeepTag::Tuple),
                    )?,
                    values: preconditions::<E, K, M>(&v.values, f)?,
                }),
                V::PropertyContracts(v) => V::PropertyContracts(PropertyContracts {
                    container: container::<E, K, M, A>(
                        &v.container,
                        f,
                        rewrite_bindings,
                        annotations,
                        Some(DeepTag::Tuple),
                    )?,
                    values: v.values.clone(),
                }),
                V::Invariant(v) if rewrite_bindings => {
                    V::Invariant(InvariantPredicate::try_from_expression(
                        map_leaf_readmitted::<E, K, M>(f, &v.to_expression(), R::Expression)?,
                    )?)
                }
                V::Invariant(v) => V::Invariant(InvariantPredicate {
                    container: container::<E, K, M, A>(
                        &v.container,
                        f,
                        false,
                        annotations,
                        Some(DeepTag::Fn),
                    )?,
                    params: container::<E, K, M, A>(
                        &v.params,
                        f,
                        false,
                        annotations,
                        Some(DeepTag::Params),
                    )?,
                    binder: binder::<E, K, M, A>(&v.binder, f, false, annotations)?,
                    body: RuntimeExpression::try_for_key(
                        map_leaf_readmitted::<E, K, M>(f, v.body.expression(), R::Expression)?,
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
                | V::SurfPipeStage(_)
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
        self.try_map_payloads::<MetadataError, Infallible, _, _>(f, true, &mut keep_annotations)
    }

    /// `map_expressions` over any crate-internal leaf mapper (authoring's
    /// metadata edits).
    pub(crate) fn map_expressions_with<K, M: LeafMapper<MetadataError, K>>(
        &self,
        f: &mut M,
    ) -> Result<Self, MetadataError> {
        self.try_map_payloads::<MetadataError, K, M, _>(f, true, &mut keep_annotations)
    }
}
