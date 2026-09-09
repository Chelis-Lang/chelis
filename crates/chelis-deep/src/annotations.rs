//! Compiler-owned AST annotations, distinct from producer extensions.
//!
//! Keys are derived from typed variants. Raw key/value pairs exist only at
//! ingress and serialization boundaries; consumers observe dedicated payloads.
use std::collections::BTreeMap;

pub use crate::ExtensionData;
pub use crate::annotations_transform::MetadataName;
use crate::{Atom, DtypeFamily, Expr, RawExpr, Span, metadata::MetadataError};

/// A value and its original syntax span.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    value: T,
    span: Span,
}
impl<T> Spanned<T> {
    pub fn new(value: T, span: Span) -> Self {
        Self { value, span }
    }
    pub fn value(&self) -> &T {
        &self.value
    }
    pub fn span(&self) -> Span {
        self.span
    }
    pub fn into_value(self) -> T {
        self.value
    }
}

pub(crate) fn invalid(key: &str, span: Span, expected: &'static str) -> MetadataError {
    MetadataError {
        key: key.into(),
        span,
        expected,
        detail: None,
        forbidden_span_char: None,
    }
}

/// An external producer's opaque ID. Construction rejects ASCII controls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpanId(Spanned<String>);
impl SpanId {
    pub fn try_new(value: String, span: Span) -> Result<Self, MetadataError> {
        if let Some((i, b)) = value
            .bytes()
            .enumerate()
            .find(|(_, b)| *b <= 0x1f || *b == 0x7f)
        {
            let mut error = invalid("span", span, "a string without ASCII control characters");
            error.forbidden_span_char = Some((i, b));
            return Err(error);
        }
        Ok(Self(Spanned::new(value, span)))
    }
    pub fn value(&self) -> &str {
        self.0.value()
    }
    pub fn span(&self) -> Span {
        self.0.span()
    }
}

macro_rules! expression_payload {
    ($name:ident, $key:literal) => {
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name(Expr);
        impl $name {
            pub fn try_new(value: Expr) -> Result<Self, MetadataError> {
                crate::metadata::validate_payload($key, &value)?;
                Ok(Self(value))
            }
            pub fn expression(&self) -> &Expr {
                &self.0
            }
            pub fn into_expression(self) -> Expr {
                self.0
            }
            pub fn span(&self) -> Span {
                self.0.span()
            }
        }
    };
}
expression_payload!(TypeSyntax, "type");
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeExpression(Expr);
impl RuntimeExpression {
    pub fn try_new(value: Expr) -> Result<Self, MetadataError> {
        Self::try_for_key(value, "property_seed")
    }
    pub(crate) fn try_for_key(value: Expr, key: &str) -> Result<Self, MetadataError> {
        crate::metadata::validate_runtime_payload(key, &value)?;
        Ok(Self(value))
    }
    pub fn expression(&self) -> &Expr {
        &self.0
    }
    pub fn into_expression(self) -> Expr {
        self.0
    }
    pub fn span(&self) -> Span {
        self.0.span()
    }
}

/// An exact integer syntax carrier; this does not introduce numeric storage.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegerSyntax(Expr);
impl IntegerSyntax {
    pub fn try_new(value: Expr) -> Result<Self, MetadataError> {
        if !matches!(&value, Expr::Atom(Atom::Int(_), _)) {
            return Err(invalid("loc", value.span(), "an integer atom"));
        }
        Ok(Self(value))
    }
    pub fn expression(&self) -> &Expr {
        &self.0
    }
    pub fn span(&self) -> Span {
        self.0.span()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PositiveInteger(IntegerSyntax);
impl PositiveInteger {
    pub fn try_new(value: Expr) -> Result<Self, MetadataError> {
        if !matches!(&value, Expr::Atom(Atom::Int(n), _) if *n > 0) {
            return Err(invalid(
                "surf_dim_group_size",
                value.span(),
                "a positive integer atom",
            ));
        }
        Ok(Self(IntegerSyntax(value)))
    }
    pub fn expression(&self) -> &Expr {
        self.0.expression()
    }
    pub fn span(&self) -> Span {
        self.0.span()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SourceLocation {
    pub(crate) file: Spanned<String>,
    pub(crate) line: IntegerSyntax,
    pub(crate) column: IntegerSyntax,
    pub(crate) head_span: Span,
    pub(crate) span: Span,
}
impl SourceLocation {
    pub fn new(
        file: Spanned<String>,
        line: IntegerSyntax,
        column: IntegerSyntax,
        span: Span,
    ) -> Self {
        Self {
            file,
            line,
            column,
            head_span: span,
            span,
        }
    }
    pub fn file(&self) -> &Spanned<String> {
        &self.file
    }
    pub fn line(&self) -> &IntegerSyntax {
        &self.line
    }
    pub fn column(&self) -> &IntegerSyntax {
        &self.column
    }
    pub fn span(&self) -> Span {
        self.span
    }
}

macro_rules! choices {
    ($name:ident { $($variant:ident => $spelling:literal),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $name { $($variant),+ }
        impl $name {
            pub fn spelling(self) -> &'static str { match self { $(Self::$variant => $spelling),+ } }
            pub(crate) fn decode(s: &str) -> Option<Self> { match s { $($spelling => Some(Self::$variant)),+, _ => None } }
        }
    };
}
choices!(Linearity { Once => "once", Borrow => "borrow", Unrestricted => "unrestricted" });
choices!(PropertySourceKind { User => "user", CEarchin => "bridge:c-earchin" });
choices!(Amenability { Linear => "linear", Polynomial => "polynomial", Transcendental => "transcendental", Opaque => "opaque" });
choices!(LiteralStyle { Unsuffixed => "unsuffixed", Explicit => "explicit" });
choices!(BindingTypeOrigin { Inferred => "inferred", Explicit => "explicit" });
choices!(PipeStageOrigin { CallFirst => "call-first" });
choices!(LiteralOrigin { Integer => "integer" });

/// A presence marker, with no representable false payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Present {
    span: Span,
}
impl Present {
    pub fn new(span: Span) -> Self {
        Self { span }
    }
    pub fn span(&self) -> Span {
        self.span
    }
}

/// An annotation-bearing structural root; its owner determines its tag.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Container {
    pub(crate) metadata: Metadata,
    pub(crate) span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableRef {
    pub(crate) container: Container,
    pub(crate) name: Spanned<String>,
}
impl VariableRef {
    pub fn new(
        name: Spanned<String>,
        metadata: Metadata,
        span: Span,
    ) -> Result<Self, MetadataError> {
        crate::metadata::validate_typed_container(crate::DeepTag::Var, &metadata)?;
        Ok(Self {
            name,
            container: Container { metadata, span },
        })
    }
    pub fn name(&self) -> &Spanned<String> {
        &self.name
    }
    pub fn metadata(&self) -> &Metadata {
        &self.container.metadata
    }
    pub fn span(&self) -> Span {
        self.container.span
    }
}

/// A tuple target stores its first member separately, making emptiness impossible.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableTuple {
    pub(crate) container: Container,
    pub(crate) first: VariableRef,
    pub(crate) rest: Vec<VariableRef>,
}
impl VariableTuple {
    pub fn new(
        metadata: Metadata,
        first: VariableRef,
        rest: Vec<VariableRef>,
        span: Span,
    ) -> Result<Self, MetadataError> {
        crate::metadata::validate_typed_container(crate::DeepTag::Tuple, &metadata)?;
        Ok(Self {
            container: Container { metadata, span },
            first,
            rest,
        })
    }
    pub fn variables(&self) -> impl Iterator<Item = &VariableRef> {
        std::iter::once(&self.first).chain(&self.rest)
    }
    pub fn metadata(&self) -> &Metadata {
        &self.container.metadata
    }
    pub fn span(&self) -> Span {
        self.container.span
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum WrtTargets {
    Variable(VariableRef),
    Tuple(VariableTuple),
}
impl WrtTargets {
    pub fn variables(&self) -> impl Iterator<Item = &VariableRef> {
        let (first, rest) = match self {
            Self::Variable(v) => (v, &[][..]),
            Self::Tuple(t) => (&t.first, t.rest.as_slice()),
        };
        std::iter::once(first).chain(rest)
    }
    pub fn span(&self) -> Span {
        match self {
            Self::Variable(v) => v.span(),
            Self::Tuple(v) => v.span(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinderSpelling {
    Name,
    Pair,
    Prefix,
}
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyBinder {
    pub(crate) name: Spanned<String>,
    pub(crate) container: Container,
    pub(crate) spelling: BinderSpelling,
    pub(crate) metadata_span: Span,
}
impl PropertyBinder {
    pub fn new(
        name: Spanned<String>,
        metadata: Metadata,
        span: Span,
    ) -> Result<Self, MetadataError> {
        crate::metadata::validate_typed_detached(&metadata)?;
        Ok(Self {
            name,
            container: Container { metadata, span },
            spelling: BinderSpelling::Pair,
            metadata_span: span,
        })
    }
    pub fn name(&self) -> &Spanned<String> {
        &self.name
    }
    pub fn metadata(&self) -> &Metadata {
        &self.container.metadata
    }
    pub fn span(&self) -> Span {
        self.container.span
    }
}

macro_rules! sequence_payload {
    ($name:ident, $element:ty, $tag:ident) => {
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name {
            pub(crate) container: Container,
            pub(crate) values: Vec<$element>,
        }
        impl $name {
            pub fn new(metadata: Metadata, values: Vec<$element>, span: Span) -> Self {
                Self::try_new(metadata, values, span)
                    .expect("compiler produced invalid container annotations")
            }
            pub fn try_new(
                metadata: Metadata,
                values: Vec<$element>,
                span: Span,
            ) -> Result<Self, MetadataError> {
                crate::metadata::validate_typed_container(crate::DeepTag::$tag, &metadata)?;
                Ok(Self {
                    container: Container { metadata, span },
                    values,
                })
            }
            pub fn values(&self) -> &[$element] {
                &self.values
            }
            pub fn metadata(&self) -> &Metadata {
                &self.container.metadata
            }
            pub fn span(&self) -> Span {
                self.container.span
            }
        }
    };
}
sequence_payload!(PropertyQuantifiers, PropertyBinder, Params);
sequence_payload!(PropertyPreconditions, RuntimeExpression, Tuple);
sequence_payload!(PropertyContracts, Spanned<String>, Tuple);
sequence_payload!(EffectSet, EffectMember, Effects);

#[derive(Debug, Clone, PartialEq)]
pub struct ResourceEffect {
    pub(crate) container: Container,
    pub(crate) name: Spanned<String>,
}
impl ResourceEffect {
    pub fn new(
        name: Spanned<String>,
        metadata: Metadata,
        span: Span,
    ) -> Result<Self, MetadataError> {
        crate::metadata::validate_typed_container(crate::DeepTag::Resource, &metadata)?;
        Ok(Self {
            name,
            container: Container { metadata, span },
        })
    }
    pub fn name(&self) -> &Spanned<String> {
        &self.name
    }
    pub fn metadata(&self) -> &Metadata {
        &self.container.metadata
    }
    pub fn span(&self) -> Span {
        self.container.span
    }
}
#[derive(Debug, Clone, PartialEq)]
pub enum EffectMember {
    Name(Spanned<String>),
    Resource(ResourceEffect),
}

#[derive(Debug, Clone, PartialEq)]
pub struct InvariantPredicate {
    pub(crate) container: Container,
    pub(crate) params: Container,
    pub(crate) binder: PropertyBinder,
    pub(crate) body: RuntimeExpression,
}
impl InvariantPredicate {
    pub fn new(
        metadata: Metadata,
        parameter_metadata: Metadata,
        binder: PropertyBinder,
        body: RuntimeExpression,
        span: Span,
        parameter_span: Span,
    ) -> Result<Self, MetadataError> {
        crate::metadata::validate_typed_container(crate::DeepTag::Fn, &metadata)?;
        crate::metadata::validate_typed_container(crate::DeepTag::Params, &parameter_metadata)?;
        Ok(Self {
            container: Container { metadata, span },
            params: Container {
                metadata: parameter_metadata,
                span: parameter_span,
            },
            binder,
            body,
        })
    }
    pub fn binder(&self) -> &PropertyBinder {
        &self.binder
    }
    pub fn body(&self) -> &RuntimeExpression {
        &self.body
    }
    pub fn metadata(&self) -> &Metadata {
        &self.container.metadata
    }
    pub fn parameter_metadata(&self) -> &Metadata {
        &self.params.metadata
    }
    pub fn span(&self) -> Span {
        self.container.span
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DtypeBounds {
    pub(crate) bounds: BTreeMap<String, Spanned<DtypeFamily>>,
    pub(crate) span: Span,
}
impl DtypeBounds {
    pub fn try_new(
        bounds: impl IntoIterator<Item = (String, Spanned<DtypeFamily>)>,
        span: Span,
    ) -> Result<Self, MetadataError> {
        let mut result = Self {
            bounds: BTreeMap::new(),
            span,
        };
        for (name, family) in bounds {
            if result.bounds.contains_key(&name) {
                return Err(invalid(
                    "dtype_bounds",
                    family.span(),
                    "distinct binder names",
                ));
            }
            result.bounds.insert(name, family);
        }
        Ok(result)
    }
    pub fn bounds(&self) -> impl Iterator<Item = (&str, &Spanned<DtypeFamily>)> {
        self.bounds.iter().map(|(k, v)| (k.as_str(), v))
    }
    pub fn span(&self) -> Span {
        self.span
    }
}

/// Historical arguments are raw data, never an annotation-bearing AST.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroSource {
    pub(crate) name: Spanned<String>,
    pub(crate) arguments: Vec<RawExpr>,
    pub(crate) span: Span,
}
impl MacroSource {
    pub fn new(name: Spanned<String>, arguments: Vec<RawExpr>, span: Span) -> Self {
        Self {
            name,
            arguments,
            span,
        }
    }
    pub fn name(&self) -> &Spanned<String> {
        &self.name
    }
    pub fn arguments(&self) -> &[RawExpr] {
        &self.arguments
    }
    pub fn span(&self) -> Span {
        self.span
    }
}

macro_rules! core_inventory {
    ($($variant:ident, $getter:ident, $payload:ty, $key:literal;)+) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        pub enum MetadataKey { $($variant),+ }
        impl MetadataKey {
            pub fn spelling(self) -> &'static str { match self { $(Self::$variant => $key),+ } }
            pub(crate) fn decode(key: &str) -> Option<Self> { match key { $($key => Some(Self::$variant)),+, _ => None } }
        }
        #[derive(Debug, Clone, PartialEq)]
        pub enum MetadataValue { $($variant($payload)),+ }
        impl MetadataValue {
            pub fn key(&self) -> MetadataKey { match self { $(Self::$variant(_) => MetadataKey::$variant),+ } }
            pub fn span(&self) -> Span { match self { $(Self::$variant(v) => v.span()),+ } }
        }
        impl Metadata {
            $(pub fn $getter(&self) -> Option<&$payload> {
                match self.storage.as_ref()?.core.get(&MetadataKey::$variant)? { MetadataValue::$variant(v) => Some(v), _ => unreachable!("variant determines its key") }
            })+
        }
    };
}
core_inventory! {
    Type, ty, TypeSyntax, "type";
    Loc, loc, SourceLocation, "loc";
    Eff, eff, EffectSet, "eff";
    DtypeBounds, dtype_bounds, DtypeBounds, "dtype_bounds";
    Effects, effects, EffectSet, "effects";
    Source, source, MacroSource, "source";
    Wrt, wrt, WrtTargets, "wrt";
    Span, span_id, SpanId, "span";
    ChelisRole, chelis_role, Spanned<String>, "chelis_role";
    PropertySourceKind, property_source_kind, Spanned<PropertySourceKind>, "property_source_kind";
    PropertyQuantifiers, property_quantifiers, PropertyQuantifiers, "property_quantifiers";
    PropertyPreconditions, property_preconditions, PropertyPreconditions, "property_preconditions";
    PropertySourceId, property_source_id, Spanned<String>, "property_source_id";
    PropertyTolerance, property_tolerance, RuntimeExpression, "property_tolerance";
    PropertySeed, property_seed, RuntimeExpression, "property_seed";
    PropertySamples, property_samples, RuntimeExpression, "property_samples";
    PropertyContracts, property_contracts, PropertyContracts, "property_contracts";
    Opaque, opaque, Present, "opaque";
    Invariant, invariant, InvariantPredicate, "invariant";
    InvariantAmenability, invariant_amenability, Spanned<Amenability>, "invariant_amenability";
    SurfPath, surf_path, Spanned<String>, "surf_path";
    SurfDimGroupSize, surf_dim_group_size, PositiveInteger, "surf_dim_group_size";
    SurfPipeStage, surf_pipe_stage, Spanned<PipeStageOrigin>, "surf_pipe_stage";
    SurfLiteralStyle, surf_literal_style, Spanned<LiteralStyle>, "surf_literal_style";
    SurfBindingType, surf_binding_type, Spanned<BindingTypeOrigin>, "surf_binding_type";
    Lin, lin, Spanned<Linearity>, "lin";
    Doc, doc, Spanned<String>, "doc";
    Effect, effect, Spanned<chelis_vocab::EffectKind>, "effect";
    LiteralSource, literal_source, Spanned<LiteralOrigin>, "literal_source";
    Destructure, destructure, Present, "destructure";
}

#[derive(Debug, Clone, PartialEq, Default)]
struct Storage {
    core: BTreeMap<MetadataKey, MetadataValue>,
    extensions: ExtensionMap,
}

/// Empty metadata occupies one pointer and allocates nothing.
///
/// ```
/// use chelis_deep::annotations::{Metadata, MetadataValue, Spanned};
/// use chelis_deep::Span;
/// let mut meta = Metadata::default();
/// meta.insert(MetadataValue::Doc(Spanned::new("text".into(), Span::new(0, 0)))).unwrap();
/// assert_eq!(meta.doc().unwrap().value(), "text");
/// ```
/// Raw mutation cannot bypass the payload types:
/// ```compile_fail,E0616
/// use chelis_deep::annotations::Metadata;
/// let mut meta = Metadata::default();
/// meta.storage = None;
/// ```
/// ```compile_fail,E0308
/// use chelis_deep::{annotations::MetadataValue, Atom, Expr, Span};
/// let _ = MetadataValue::Doc(Expr::Atom(Atom::Bool(false), Span::new(0, 0)));
/// ```
/// ```compile_fail,E0308
/// use chelis_deep::annotations::{MetadataValue, RuntimeExpression};
/// use chelis_deep::{Atom, Expr, Span};
/// let runtime = RuntimeExpression::try_new(Expr::Atom(Atom::Bool(true), Span::new(0, 0))).unwrap();
/// let _ = MetadataValue::PropertyPreconditions(runtime);
/// ```
/// Expression wrappers cannot be forged or mutated after validation:
/// ```compile_fail,E0423
/// use chelis_deep::{annotations::RuntimeExpression, Expr, Atom, Span};
/// let _ = RuntimeExpression(Expr::Atom(Atom::Name("unchecked".into()), Span::new(0, 0)));
/// ```
/// ```compile_fail,E0616
/// use chelis_deep::{annotations::RuntimeExpression, Expr, Atom, Span};
/// let mut value = RuntimeExpression::try_new(Expr::Atom(Atom::Bool(true), Span::new(0, 0))).unwrap();
/// value.0 = Expr::Atom(Atom::Name("unchecked".into()), Span::new(0, 0));
/// ```
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    storage: Option<Box<Storage>>,
}
impl PartialEq for Metadata {
    fn eq(&self, other: &Self) -> bool {
        self.values().eq(other.values()) && self.extensions() == other.extensions()
    }
}
impl From<MetadataValue> for Metadata {
    fn from(value: MetadataValue) -> Self {
        let mut metadata = Self::default();
        metadata
            .insert(value)
            .expect("empty metadata has no duplicate");
        metadata
    }
}
impl Metadata {
    pub fn try_from_values(
        values: impl IntoIterator<Item = MetadataValue>,
    ) -> Result<Self, MetadataError> {
        let mut metadata = Self::default();
        for value in values {
            metadata.insert(value)?;
        }
        Ok(metadata)
    }
    pub fn is_empty(&self) -> bool {
        self.storage
            .as_ref()
            .is_none_or(|s| s.core.is_empty() && s.extensions.is_empty())
    }
    pub fn insert(&mut self, value: MetadataValue) -> Result<(), MetadataError> {
        let storage = self.storage.get_or_insert_with(Default::default);
        if storage.core.contains_key(&value.key()) {
            return Err(invalid(
                value.key().spelling(),
                value.span(),
                "exactly one occurrence of this metadata key",
            ));
        }
        storage.core.insert(value.key(), value);
        Ok(())
    }
    pub fn replace(&mut self, value: MetadataValue) -> Option<MetadataValue> {
        self.storage
            .get_or_insert_with(Default::default)
            .core
            .insert(value.key(), value)
    }
    pub fn remove(&mut self, key: MetadataKey) -> Option<MetadataValue> {
        self.storage.as_mut()?.core.remove(&key)
    }
    pub fn values(&self) -> impl Iterator<Item = &MetadataValue> {
        self.storage.iter().flat_map(|s| s.core.values())
    }
    pub fn extensions(&self) -> &ExtensionMap {
        static EMPTY: ExtensionMap = ExtensionMap {
            values: BTreeMap::new(),
        };
        self.storage.as_ref().map_or(&EMPTY, |s| &s.extensions)
    }
    pub fn extensions_mut(&mut self) -> &mut ExtensionMap {
        &mut self.storage.get_or_insert_with(Default::default).extensions
    }

    /// Visit the live expression leaves and annotations of structural payloads.
    /// Container roots, dtype binder data, and historical source are excluded.
    /// Variable target identities are available separately through `wrt()`.
    /// Find a borrowed expression leaf, excluding data and structural roots.
    pub fn find_expression<'a, T>(&'a self, mut f: impl FnMut(&'a Expr) -> Option<T>) -> Option<T> {
        let mut result = None;
        self.visit_expressions(&mut |value, _| {
            if result.is_none() {
                result = f(value);
            }
        });
        result
    }
    pub fn any_expression(&self, mut f: impl FnMut(&Expr) -> bool) -> bool {
        self.find_expression(|value| f(value).then_some(()))
            .is_some()
    }
    pub fn visit_expressions<'a>(
        &'a self,
        visit: &mut impl FnMut(&'a Expr, crate::metadata::MetadataRole),
    ) {
        use crate::metadata::MetadataRole as R;
        for value in self.values() {
            match value {
                MetadataValue::Type(v) => visit(v.expression(), R::Type),
                MetadataValue::PropertyTolerance(v)
                | MetadataValue::PropertySeed(v)
                | MetadataValue::PropertySamples(v) => visit(v.expression(), R::Expression),
                MetadataValue::Eff(v) | MetadataValue::Effects(v) => {
                    v.metadata().visit_expressions(visit);
                    for member in v.values() {
                        if let EffectMember::Resource(v) = member {
                            v.metadata().visit_expressions(visit);
                        }
                    }
                }
                MetadataValue::Wrt(v) => {
                    if let WrtTargets::Tuple(v) = v {
                        v.metadata().visit_expressions(visit);
                    }
                    for v in v.variables() {
                        v.metadata().visit_expressions(visit);
                    }
                }
                MetadataValue::PropertyQuantifiers(v) => {
                    v.metadata().visit_expressions(visit);
                    for binder in v.values() {
                        binder.metadata().visit_expressions(visit);
                    }
                }
                MetadataValue::PropertyPreconditions(v) => {
                    v.metadata().visit_expressions(visit);
                    for v in v.values() {
                        visit(v.expression(), R::Expression);
                    }
                }
                MetadataValue::PropertyContracts(v) => v.metadata().visit_expressions(visit),
                MetadataValue::Invariant(v) => {
                    v.metadata().visit_expressions(visit);
                    v.parameter_metadata().visit_expressions(visit);
                    v.binder().metadata().visit_expressions(visit);
                    visit(v.body().expression(), R::Expression);
                }
                MetadataValue::Source(_)
                | MetadataValue::DtypeBounds(_)
                | MetadataValue::Loc(_)
                | MetadataValue::Span(_)
                | MetadataValue::ChelisRole(_)
                | MetadataValue::PropertySourceKind(_)
                | MetadataValue::PropertySourceId(_)
                | MetadataValue::Opaque(_)
                | MetadataValue::InvariantAmenability(_)
                | MetadataValue::SurfPath(_)
                | MetadataValue::SurfDimGroupSize(_)
                | MetadataValue::SurfPipeStage(_)
                | MetadataValue::SurfLiteralStyle(_)
                | MetadataValue::SurfBindingType(_)
                | MetadataValue::Lin(_)
                | MetadataValue::Doc(_)
                | MetadataValue::Effect(_)
                | MetadataValue::LiteralSource(_)
                | MetadataValue::Destructure(_) => {}
            }
        }
    }
}

/// Producer-owned opaque data. Reserved compiler keys cannot be inserted here.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExtensionMap {
    values: BTreeMap<String, ExtensionData>,
}
impl ExtensionMap {
    fn validate(key: &str, value: &ExtensionData) -> Result<(), MetadataError> {
        let mut bytes = key.bytes();
        if !bytes
            .next()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
            || !bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(invalid(
                key,
                value.span(),
                "an ASCII identifier metadata key",
            ));
        }
        if MetadataKey::decode(key).is_some() || key.starts_with("surf_") {
            return Err(invalid(
                key,
                value.span(),
                "a producer extension key outside compiler-owned metadata",
            ));
        }
        Ok(())
    }
    pub fn insert(&mut self, key: String, value: ExtensionData) -> Result<(), MetadataError> {
        Self::validate(&key, &value)?;
        if self.values.contains_key(&key) {
            return Err(invalid(
                &key,
                value.span(),
                "exactly one occurrence of this metadata key",
            ));
        }
        self.values.insert(key, value);
        Ok(())
    }
    pub fn replace(
        &mut self,
        key: String,
        value: ExtensionData,
    ) -> Result<Option<ExtensionData>, MetadataError> {
        Self::validate(&key, &value)?;
        Ok(self.values.insert(key, value))
    }
    pub fn get(&self, key: &str) -> Option<&ExtensionData> {
        self.values.get(key)
    }
    pub fn remove(&mut self, key: &str) -> Option<ExtensionData> {
        self.values.remove(key)
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = (&str, &ExtensionData)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Combine owners transactionally; different data under one key cannot be lost.
    pub fn try_merge(&mut self, other: &Self) -> Result<(), MetadataError> {
        for (key, value) in other.iter() {
            if self
                .get(key)
                .is_some_and(|prior| !prior.same_payload(value))
            {
                return Err(invalid(
                    key,
                    value.span(),
                    "identical extension data when combining owners",
                ));
            }
        }
        for (key, value) in other.iter() {
            self.values
                .entry(key.into())
                .or_insert_with(|| value.clone());
        }
        Ok(())
    }
}
