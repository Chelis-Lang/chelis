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
                match self.storage.as_ref()?.get(MetadataKey::$variant)? { MetadataValue::$variant(v) => Some(v), _ => unreachable!("variant determines its key") }
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

/// Core annotations, held as a key-sorted vector rather than a map.
///
/// The core inventory has thirty keys and a real node carries a handful: the
/// chelis#1604 metadata fixture has 72 maps, 41 of them empty and none above
/// three entries. `MetadataValue` is 184 bytes wide on a 64-bit target, and a
/// `BTreeMap` allocates one full eleven-slot leaf node whatever it holds, so
/// a one-entry map costs a measured 2048 bytes.
///
/// A vector does not allocate only what it stores. It takes the four-slot
/// minimum non-zero capacity, a measured 736 bytes, and holds one to four
/// entries in that one allocation; it reallocates crossing four and stays
/// under the leaf node's footprint up to eight. So the win is about a third
/// of the map, 736 against 2048, at every size this corpus contains, not the
/// arbitrary factor "only what it stores" would suggest. Empty metadata still
/// allocates nothing, because the handle itself is `None`.
///
/// Iteration keeps the key order every consumer already relies on, which
/// `core_key_order_lock` pins, and lookup is a binary search over at most
/// thirty elements.
#[derive(Debug, PartialEq, Default)]
struct Storage {
    core: Vec<MetadataValue>,
    extensions: ExtensionMap,
}

// Counted receipt for the shared-storage promise (chelis#2117).
//
// `Metadata::clone` must not copy annotation storage; only a write through
// `Arc::make_mut` may. A test build counts every deep copy so
// `storage_copies_are_independent_of_clone_count` can assert that promise as
// a ratio instead of a wall clock. Release builds compile the recorder away.
#[cfg(test)]
thread_local! {
    static STORAGE_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn record_storage_copy() {
    STORAGE_COPIES.with(|count| count.set(count.get() + 1));
}

#[cfg(not(test))]
#[inline(always)]
fn record_storage_copy() {}

impl Clone for Storage {
    fn clone(&self) -> Self {
        record_storage_copy();
        Self {
            core: self.core.clone(),
            extensions: self.extensions.clone(),
        }
    }
}

impl Storage {
    fn locate(&self, key: MetadataKey) -> Result<usize, usize> {
        self.core.binary_search_by(|value| value.key().cmp(&key))
    }
    fn get(&self, key: MetadataKey) -> Option<&MetadataValue> {
        self.locate(key).ok().map(|index| &self.core[index])
    }
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
    /// Shared behind a reference count. Cloning a Deep tree copies one
    /// pointer per node instead of every node's annotation storage; a write
    /// copies through `Arc::make_mut`, so sharing stays invisible.
    storage: Option<std::sync::Arc<Storage>>,
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
        // Decide before `Arc::make_mut`. A rejected duplicate writes nothing,
        // so it must not deep-copy shared storage on its way to an error.
        let index = match self
            .storage
            .as_ref()
            .map(|storage| storage.locate(value.key()))
        {
            Some(Ok(_)) => {
                return Err(invalid(
                    value.key().spelling(),
                    value.span(),
                    "exactly one occurrence of this metadata key",
                ));
            }
            Some(Err(index)) => index,
            None => 0,
        };
        let storage = std::sync::Arc::make_mut(self.storage.get_or_insert_with(Default::default));
        storage.core.insert(index, value);
        Ok(())
    }
    pub fn replace(&mut self, value: MetadataValue) -> Option<MetadataValue> {
        let storage = std::sync::Arc::make_mut(self.storage.get_or_insert_with(Default::default));
        match storage.locate(value.key()) {
            Ok(index) => Some(std::mem::replace(&mut storage.core[index], value)),
            Err(index) => {
                storage.core.insert(index, value);
                None
            }
        }
    }
    pub fn remove(&mut self, key: MetadataKey) -> Option<MetadataValue> {
        // Decide before `Arc::make_mut`, for the same reason `insert` does:
        // removing a key that is not there writes nothing.
        let index = self.storage.as_ref()?.locate(key).ok()?;
        let storage = std::sync::Arc::make_mut(
            self.storage
                .as_mut()
                .expect("storage was present when the index was located"),
        );
        Some(storage.core.remove(index))
    }
    pub fn values(&self) -> impl Iterator<Item = &MetadataValue> {
        self.storage.iter().flat_map(|s| s.core.iter())
    }
    pub fn extensions(&self) -> &ExtensionMap {
        static EMPTY: ExtensionMap = ExtensionMap {
            values: BTreeMap::new(),
        };
        self.storage.as_ref().map_or(&EMPTY, |s| &s.extensions)
    }
    pub fn extensions_mut(&mut self) -> &mut ExtensionMap {
        &mut std::sync::Arc::make_mut(self.storage.get_or_insert_with(Default::default)).extensions
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

/// Shared vocabulary for the annotation test modules below, so the
/// deep-copy counter and the key-order helpers have one definition each.
#[cfg(test)]
mod test_support {
    use super::{Metadata, MetadataKey, MetadataValue, STORAGE_COPIES, Spanned};
    use crate::Span;

    pub(super) fn span() -> Span {
        Span::new(0, 0)
    }

    pub(super) fn text(value: &str) -> Spanned<String> {
        Spanned::new(value.to_string(), span())
    }

    pub(super) fn observed(metadata: &Metadata) -> Vec<MetadataKey> {
        metadata.values().map(|value| value.key()).collect()
    }

    /// Strictly increasing, not merely equal to a sorted copy: a vector
    /// carrying one key twice passes the latter and fails this.
    pub(super) fn strictly_increasing(metadata: &Metadata) -> bool {
        observed(metadata).windows(2).all(|pair| pair[0] < pair[1])
    }

    pub(super) fn copies_now() -> usize {
        STORAGE_COPIES.with(|count| count.get())
    }

    pub(super) fn copies_since(start: usize) -> usize {
        copies_now() - start
    }

    pub(super) fn annotated() -> Metadata {
        let mut metadata = Metadata::default();
        metadata
            .insert(MetadataValue::Doc(Spanned::new("note".to_string(), span())))
            .expect("empty metadata accepts one doc annotation");
        metadata
    }
}

#[cfg(test)]
mod shared_storage_receipt {
    use super::test_support::{annotated, copies_now, copies_since, span};
    use super::{Metadata, MetadataKey, MetadataValue, Spanned};
    use crate::tag::DeepTag;
    use crate::{Atom, Expr};

    /// A nested `app` chain of `depth` annotated nodes over one annotated
    /// `var` leaf, so the tree carries exactly `depth + 1` non-empty
    /// metadata maps.
    fn annotated_chain(depth: usize) -> Expr {
        let mut expr = Expr::node(
            DeepTag::Var,
            annotated(),
            vec![Expr::Atom(Atom::Name("f".to_string()), span())],
            span(),
        );
        for _ in 0..depth {
            expr = Expr::node(
                DeepTag::App,
                annotated(),
                vec![expr, Expr::Atom(Atom::Int(1), span())],
                span(),
            );
        }
        expr
    }

    /// chelis#2117: cloning a Deep tree must not copy its annotation
    /// storage, so the deep-copy count is a property of how many writes
    /// the clone performs and not of how many annotated nodes it spans.
    /// Doubling the node count must leave that count unchanged.
    ///
    /// Proved failing first. Reverting only the storage handle to
    /// `Option<Box<Storage>>`, with this counter and both trees unchanged,
    /// records `8` deep copies for the 8-node tree and `16` for the
    /// 16-node tree: a ratio of 2.0 against the 1.0 this asserts, because
    /// a boxed map is copied once per annotated node on every tree clone.
    /// The twin below fails the same revert with `0` copies where it
    /// requires `1`, since an unshared box never needs one.
    #[test]
    fn storage_copies_are_independent_of_clone_count() {
        let small = annotated_chain(7);
        let large = annotated_chain(15);

        let start = copies_now();
        let small_clone = small.clone();
        let small_copies = copies_since(start);

        let start = copies_now();
        let large_clone = large.clone();
        let large_copies = copies_since(start);

        assert_eq!(
            small_copies, 0,
            "cloning an 8-node annotated tree copied annotation storage"
        );
        assert_eq!(
            large_copies, small_copies,
            "doubling the annotated node count changed the deep-copy count"
        );
        assert_eq!(small_clone, small);
        assert_eq!(large_clone, large);
    }

    /// Switching `Box` to `Arc` narrows auto traits: `Box<T>: Send` needs
    /// only `T: Send`, while `Arc<T>: Send` needs `T: Send + Sync`. If a
    /// payload ever stops being `Sync`, `Metadata` would silently stop being
    /// `Send` and every consumer that moves a Deep tree across a thread
    /// would fail to compile somewhere far from here. Landed from the #2210
    /// review round.
    #[test]
    fn metadata_and_expr_stay_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Metadata>();
        assert_send_sync::<Expr>();
        assert_send_sync::<MetadataValue>();
    }

    /// A no-op must not un-share. `insert` of a duplicate key and `remove`
    /// of an absent key both write nothing, so both decide before
    /// `Arc::make_mut`; an eager `make_mut` would deep-copy the storage and
    /// silently give up the sharing this change exists to buy. The proof
    /// that sharing survived is the last assertion: the next real write
    /// still costs exactly one copy, which only a still-shared storage does.
    ///
    /// Proved failing first, once per method, since the test stops at the
    /// first failure. Hoisting `Arc::make_mut` back above the decision in
    /// `insert`, which is how it was first written, fails with 1 copy
    /// against the required 0 on the rejected duplicate. Hoisting it in
    /// `remove` fails the same way on the absent key.
    #[test]
    fn a_rejected_or_absent_key_operation_does_not_unshare_storage() {
        let original = annotated();
        let mut copy = original.clone();

        let start = copies_now();
        assert!(
            copy.insert(MetadataValue::Doc(Spanned::new(
                "again".to_string(),
                span()
            )))
            .is_err(),
            "the clone already carries a doc annotation"
        );
        assert_eq!(
            copies_since(start),
            0,
            "a rejected duplicate insert must not copy shared storage"
        );

        let start = copies_now();
        assert!(copy.remove(MetadataKey::SurfPath).is_none());
        assert_eq!(
            copies_since(start),
            0,
            "removing an absent key must not copy shared storage"
        );

        let start = copies_now();
        copy.replace(MetadataValue::Doc(Spanned::new(
            "rewritten".to_string(),
            span(),
        )));
        assert_eq!(
            copies_since(start),
            1,
            "the no-ops must have left the storage shared"
        );
        assert_eq!(
            original.doc().expect("original is untouched").value(),
            "note"
        );
    }

    /// The negative twin: sharing must stay invisible. A write through a
    /// clone copies exactly the one storage it touches and leaves the
    /// original alone.
    #[test]
    fn writing_through_a_clone_copies_once_and_does_not_alias() {
        let original = annotated();
        let mut copy = original.clone();

        let start = copies_now();
        copy.replace(MetadataValue::Doc(Spanned::new(
            "rewritten".to_string(),
            span(),
        )));
        assert_eq!(
            copies_since(start),
            1,
            "a write through a shared clone must copy exactly one storage"
        );

        assert_eq!(
            original.doc().expect("original keeps its doc").value(),
            "note"
        );
        assert_eq!(
            copy.doc().expect("clone carries the write").value(),
            "rewritten"
        );

        let start = copies_now();
        copy.remove(MetadataKey::Doc);
        assert_eq!(
            copies_since(start),
            0,
            "a second write to unshared storage must not copy again"
        );
        assert!(copy.is_empty());
        assert!(!original.is_empty());
    }
}

#[cfg(test)]
mod core_key_order_lock {
    use super::test_support::{observed, strictly_increasing, text};
    use super::{Metadata, MetadataKey, MetadataValue};

    /// `values()` yields `MetadataKey` order, which the `BTreeMap` this
    /// storage replaced supplied for free and a key-sorted vector supplies
    /// only while every mutation keeps it sorted. Two consumers observe the
    /// order directly: `Metadata`'s `PartialEq` compares `values()`
    /// pairwise, so a reordering makes two equal annotation sets compare
    /// unequal, and `chelis_surf::resugar` builds a property declaration's
    /// option list in this order, so a reordering changes decompiled Surf
    /// text. Serialization is insulated, because
    /// `WireMetadata::from_metadata` re-sorts by key spelling.
    ///
    /// It is a correctness invariant and not only a formatting one, which
    /// the #2210 review round established by mutating the sorted insert to
    /// a `push` and measuring what followed. An unsorted vector makes the
    /// binary search miss keys that are present: with
    /// `[Doc, SurfPath, ChelisRole]` in the vector, `doc()` and
    /// `surf_path()` both answer `None` while both keys sit in `values()`,
    /// and `remove(Doc)` reports absent and removes nothing. `insert` then
    /// accepts a duplicate, because its duplicate check is that same
    /// search, and `values()` yields the key twice, so `PartialEq` and the
    /// resugarer see a doubled entry while a read returns whichever copy
    /// the search lands on. The `unreachable!("variant determines its
    /// key")` in the generated getters is not affected:
    /// `binary_search_by` returns `Ok` only at an index whose comparator
    /// answered `Equal`, so a located value always carries the key that
    /// was asked for.
    ///
    /// The three mutating paths are exercised in one sequence: scrambled
    /// `insert`s, a `replace` onto an occupied key, a `remove` from the
    /// middle, and a `replace` that re-inserts into that middle slot.
    ///
    /// Proved failing first. Replacing the sorted `storage.core.insert(index,
    /// value)` in `Metadata::insert` with `storage.core.push(value)`, and
    /// changing nothing else, leaves the first assertion reporting insertion
    /// order `[Doc, SurfPath, ChelisRole, PropertySourceId]` against the
    /// required `[ChelisRole, PropertySourceId, SurfPath, Doc]`.
    #[test]
    fn values_are_key_ordered_through_insert_replace_and_remove() {
        // Deliberately scrambled against the declared key order: Doc is the
        // twenty-seventh key, SurfPath the twenty-first, ChelisRole the
        // ninth, PropertySourceId the thirteenth.
        let mut metadata = Metadata::default();
        metadata.insert(MetadataValue::Doc(text("doc"))).unwrap();
        metadata
            .insert(MetadataValue::SurfPath(text("path")))
            .unwrap();
        metadata
            .insert(MetadataValue::ChelisRole(text("role")))
            .unwrap();
        metadata
            .insert(MetadataValue::PropertySourceId(text("source")))
            .unwrap();

        let expected = vec![
            MetadataKey::ChelisRole,
            MetadataKey::PropertySourceId,
            MetadataKey::SurfPath,
            MetadataKey::Doc,
        ];
        assert_eq!(
            observed(&metadata),
            expected,
            "scrambled inserts must still yield MetadataKey order"
        );
        assert!(strictly_increasing(&metadata));

        // `replace` onto an occupied key keeps its slot.
        let prior = metadata.replace(MetadataValue::Doc(text("rewritten")));
        assert!(matches!(prior, Some(MetadataValue::Doc(_))));
        assert_eq!(observed(&metadata), expected);
        assert_eq!(
            metadata.doc().expect("doc survives replace").value(),
            "rewritten"
        );

        // `remove` from the middle keeps the rest ordered.
        let removed = metadata.remove(MetadataKey::PropertySourceId);
        assert!(matches!(removed, Some(MetadataValue::PropertySourceId(_))));
        assert_eq!(
            observed(&metadata),
            vec![
                MetadataKey::ChelisRole,
                MetadataKey::SurfPath,
                MetadataKey::Doc
            ]
        );

        // `replace` on an absent key inserts at its sorted slot, not the end.
        assert!(
            metadata
                .replace(MetadataValue::PropertySourceId(text("again")))
                .is_none()
        );
        assert_eq!(observed(&metadata), expected);
        assert!(
            strictly_increasing(&metadata),
            "keys must be strictly increasing"
        );
    }
}

#[cfg(test)]
mod shared_storage_order {
    //! Coverage of the locate-here-apply-there seam. `insert`, `remove` and
    //! `replace` each binary-search against a shared borrow and then apply
    //! that index after `Arc::make_mut`, which hands back a different
    //! allocation whenever the storage is shared. Every other order test in
    //! this crate builds a fresh `Metadata` and mutates it, so `make_mut`
    //! never clones and that seam never runs; these hold a live co-owner
    //! across each write, and assert the deep-copy count so a probe that
    //! stops reaching the seam fails instead of passing vacuously.
    //!
    //! Landed from the #2210 review round (rt-2210-round1). The assertions
    //! and their messages are the reviewer's; only the helper imports were
    //! folded onto `test_support`, which already held byte-identical copies
    //! of `span` and `text`, and the reviewer's `keys` is `observed` here.
    //!
    //! Order mutations, each run on this tree, killing the module's own
    //! `core_key_order_lock` as well. The five add shared-path coverage
    //! rather than catching a defect the suite misses today.
    //!
    //! | mutation | also killed here |
    //! |---|---|
    //! | `insert`'s `core.insert(index, value)` to `core.push(value)` | insert, remove, no-op |
    //! | `remove`'s `core.remove(index)` to `core.swap_remove(index)` | remove |
    //! | `replace`'s absent-key `core.insert(index, value)` to `core.push(value)` | replace |
    use super::test_support::{
        copies_now, copies_since, observed, span, strictly_increasing, text,
    };
    use super::{Metadata, MetadataKey, MetadataValue};

    /// Deliberately scrambled against the declared key order: ChelisRole is
    /// ninth, PropertySourceId thirteenth, SurfPath twenty-first, Doc
    /// twenty-seventh.
    fn scrambled() -> Vec<MetadataValue> {
        vec![
            MetadataValue::Doc(text("d")),
            MetadataValue::SurfPath(text("p")),
            MetadataValue::ChelisRole(text("r")),
            MetadataValue::PropertySourceId(text("s")),
        ]
    }
    fn expected_order() -> Vec<MetadataKey> {
        vec![
            MetadataKey::ChelisRole,
            MetadataKey::PropertySourceId,
            MetadataKey::SurfPath,
            MetadataKey::Doc,
        ]
    }

    /// `insert` locates against a shared borrow and applies that index after
    /// `Arc::make_mut`, which hands back a different allocation whenever the
    /// storage is shared. Every other order test in this crate builds a
    /// fresh `Metadata` and mutates it, so `make_mut` never clones and that
    /// seam never executes. Here a live co-owner holds the storage shared
    /// across every insert, and the deep-copy count is asserted rather than
    /// assumed, so the test fails loudly if it ever stops exercising the
    /// shared path instead of passing vacuously.
    #[test]
    fn insert_through_a_shared_handle_keeps_key_order_at_every_step() {
        // Seeded, not empty. An insert into `storage: None` allocates a
        // fresh `Arc` at count one, so `make_mut` never clones and the seam
        // is not reached; the copy assertion below caught exactly that when
        // this started from `Metadata::default()`. Destructure is the last
        // declared key, so the scrambled inserts still land at index 0 three
        // times and mid-vector once.
        let mut metadata: Metadata = MetadataValue::Destructure(super::Present::new(span())).into();
        let mut pins = Vec::new();
        for value in scrambled() {
            let pin = metadata.clone();
            let before = observed(&pin).len();

            let start = copies_now();
            metadata.insert(value).expect("each key appears once");
            assert_eq!(
                copies_since(start),
                1,
                "the insert did not run against shared storage, so this test \
                 no longer covers the locate-then-make_mut seam"
            );

            assert!(
                strictly_increasing(&metadata),
                "unsorted after an insert through shared storage: {:?}",
                observed(&metadata)
            );
            assert_eq!(observed(&pin).len(), before, "the co-owner saw the insert");
            pins.push(pin);
        }
        let mut expected = expected_order();
        expected.push(MetadataKey::Destructure);
        assert_eq!(observed(&metadata), expected);
        for pin in &pins {
            assert!(strictly_increasing(pin));
        }
    }

    /// The same seam in `remove`, with the victim taken from every position
    /// in turn so the surviving prefix and suffix are both exercised.
    #[test]
    fn remove_through_a_shared_handle_keeps_key_order() {
        let full = Metadata::try_from_values(scrambled()).expect("distinct keys");
        for victim in expected_order() {
            let mut metadata = full.clone();
            let pin = metadata.clone();

            let start = copies_now();
            assert!(metadata.remove(victim).is_some(), "{victim:?} was present");
            assert_eq!(
                copies_since(start),
                1,
                "the remove did not run against shared storage"
            );

            assert!(
                strictly_increasing(&metadata),
                "unsorted after a remove through shared storage: {:?}",
                observed(&metadata)
            );
            assert_eq!(
                observed(&metadata),
                expected_order()
                    .into_iter()
                    .filter(|key| *key != victim)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                observed(&pin),
                expected_order(),
                "the co-owner saw the removal"
            );
            assert_eq!(
                observed(&full),
                expected_order(),
                "the source saw the removal"
            );
        }
    }

    /// `replace` onto an absent key takes the same insertion path and must
    /// land at the sorted slot in the copy, not at the end of it.
    #[test]
    fn replace_through_a_shared_handle_inserts_at_the_sorted_slot() {
        let partial = Metadata::try_from_values([
            MetadataValue::ChelisRole(text("r")),
            MetadataValue::Doc(text("d")),
        ])
        .expect("distinct keys");
        let mut metadata = partial.clone();
        let pin = metadata.clone();

        let start = copies_now();
        assert!(
            metadata
                .replace(MetadataValue::SurfPath(text("p")))
                .is_none()
        );
        assert_eq!(
            copies_since(start),
            1,
            "the replace did not run against shared storage"
        );

        assert_eq!(
            observed(&metadata),
            vec![
                MetadataKey::ChelisRole,
                MetadataKey::SurfPath,
                MetadataKey::Doc
            ],
            "SurfPath must land between ChelisRole and Doc, not at the end"
        );
        assert!(strictly_increasing(&metadata));
        assert_eq!(
            observed(&pin),
            vec![MetadataKey::ChelisRole, MetadataKey::Doc]
        );
    }

    /// The no-op fast paths must be observational identity, and must leave
    /// the storage usable: a real write afterwards still lands at its sorted
    /// slot. Run against unshared and shared storage in turn, because the
    /// two take different routes through `Arc::make_mut`.
    #[test]
    fn no_op_operations_are_observationally_identity() {
        for shared in [false, true] {
            let base = Metadata::try_from_values(scrambled()).expect("distinct keys");
            let mut metadata = base.clone();
            let pin = shared.then(|| metadata.clone());

            let before = observed(&metadata);
            assert!(metadata.insert(MetadataValue::Doc(text("dup"))).is_err());
            assert!(metadata.remove(MetadataKey::Lin).is_none());
            assert!(metadata.remove(MetadataKey::Span).is_none());

            assert_eq!(observed(&metadata), before, "a no-op changed the key order");
            assert_eq!(metadata, base, "a no-op changed the value");
            assert_eq!(metadata.doc().expect("doc survives").value(), "d");
            assert!(strictly_increasing(&metadata));
            if let Some(pin) = pin {
                assert_eq!(pin, base, "a no-op disturbed the co-owner");
            }

            let mut after = metadata.clone();
            after
                .insert(MetadataValue::Opaque(super::Present::new(span())))
                .expect("Opaque is absent");
            assert!(
                strictly_increasing(&after),
                "a write after the no-ops went astray"
            );
            assert_eq!(
                metadata, base,
                "the later write leaked back into the source"
            );
        }
    }

    /// The rejected-duplicate error must not depend on whether the storage
    /// was shared, which is the only externally visible thing the fast path
    /// could have changed.
    #[test]
    fn rejected_duplicate_reports_the_same_error_shared_or_not() {
        let owned: Metadata = MetadataValue::Doc(text("base")).into();
        let mut unshared = owned.clone();
        drop(owned);
        let from_unshared = unshared
            .insert(MetadataValue::Doc(text("dup")))
            .expect_err("Doc is already present");

        let kept: Metadata = MetadataValue::Doc(text("base")).into();
        let mut shared = kept.clone();
        let from_shared = shared
            .insert(MetadataValue::Doc(text("dup")))
            .expect_err("Doc is already present");

        assert_eq!(format!("{from_unshared:?}"), format!("{from_shared:?}"));
        assert!(
            format!("{from_unshared:?}").contains("doc"),
            "{from_unshared:?}"
        );
        assert_eq!(kept.doc().expect("source intact").value(), "base");
    }
}
