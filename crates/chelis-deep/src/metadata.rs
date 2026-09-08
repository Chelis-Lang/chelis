//! Per-key metadata contracts from spec/03 [03-META-1/2].
//!
//! Raw syntax and programmatic ASTs share these rules. Payload roles matter:
//! binder maps and historical macro arguments are not annotation maps. Node
//! construction checks local invariants inductively; public tree boundaries
//! additionally check parent and sibling placement.
use crate::{Atom, DeepTag, Expr, MetaMap, RawAtom, RawExpr, Span};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataError {
    pub key: String,
    pub span: Span,
    pub expected: &'static str,
    pub detail: Option<String>,
    pub(crate) forbidden_span_char: Option<(usize, u8)>,
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "metadata `{}` requires {} at byte {}; see spec/03 §1.1 [03-META-1/2]",
            self.key, self.expected, self.span.offset
        )?;
        if let Some(detail) = &self.detail {
            write!(f, "; {detail}")?;
        }
        Ok(())
    }
}
impl std::error::Error for MetadataError {}

#[derive(Clone, Copy)]
enum Shape {
    String,
    Span,
    Choices(&'static [&'static str]),
    Names(&'static [&'static str]),
    True,
    PositiveInteger,
    Type,
    Effects,
    Params,
    Expressions,
    Contracts,
    Invariant,
    Bounds,
    Source,
    Loc,
    Wrt,
    Expression,
}
#[derive(Clone, Copy)]
enum Placement {
    Any,
    Declaration,
    Tag(DeepTag),
    Path,
    BindingValue,
    PipeStage,
}

/// How stamping reads a payload, distinct from child-index roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MetadataRole {
    Syntax,
    Expression,
    Type,
    Preserved,
    BinderMap,
}

struct Rule {
    key: &'static str,
    shape: Shape,
    placement: Placement,
    expected: &'static str,
}

macro_rules! rules {
    ($($key:literal => $shape:expr, $placement:expr, $expected:literal;)*) => {
        const RULES: &[Rule] = &[$(Rule { key: $key, shape: $shape, placement: $placement, expected: $expected },)*];
        /// Exact individually registered keys in spec/03 §1.1.
        pub const REGISTERED_METADATA_KEYS: &[&str] = &[$($key,)*];
    };
}
use DeepTag as T;
use Placement as P;
use Shape as S;
rules! {
    "type" => S::Type, P::Any, "a type-expression node";
    "loc" => S::Loc, P::Any, "(loc string integer integer)";
    "eff" => S::Effects, P::Tag(T::TFn), "an effects node of names or (resource {} device) entries on t-fn";
    "dtype_bounds" => S::Bounds, P::Tag(T::Defsig), "a map from distinct binder names to float, int, or numeric on defsig";
    "effects" => S::Effects, P::Tag(T::Fn), "an effects node of names or (resource {} device) entries on fn";
    "source" => S::Source, P::Any, "a preserved structural (macro-name original-arg...) list";
    "wrt" => S::Wrt, P::Tag(T::Grad), "a variable or nonempty tuple of variables on grad";
    "span" => S::Span, P::Any, "a string without ASCII control characters (spec/03 §1.1.1)";
    "chelis_role" => S::String, P::Declaration, "a string on a declaration (property requires def)";
    "property_source_kind" => S::Choices(&["user", "bridge:c-earchin"]), P::Tag(T::Def), "\"user\" or \"bridge:c-earchin\" on def";
    "property_quantifiers" => S::Params, P::Tag(T::Def), "a params node on def";
    "property_preconditions" => S::Expressions, P::Tag(T::Def), "a tuple of expressions on def";
    "property_source_id" => S::String, P::Tag(T::Def), "a string on def";
    "property_tolerance" => S::Expression, P::Tag(T::Def), "an expression on def";
    "property_seed" => S::Expression, P::Tag(T::Def), "an expression on def";
    "property_samples" => S::Expression, P::Tag(T::Def), "an expression on def";
    "property_contracts" => S::Contracts, P::Tag(T::Def), "a tuple of string atoms on def";
    "opaque" => S::True, P::Tag(T::Deftype), "true on deftype";
    "invariant" => S::Invariant, P::Tag(T::Deftype), "a one-binder fn on an opaque deftype";
    "invariant_amenability" => S::Choices(&["linear", "polynomial", "transcendental", "opaque"]), P::Tag(T::Deftype), "a canonical amenability string on an invariant-carrying deftype";
    "surf_path" => S::String, P::Path, "a string on module/import/import-all whose ASCII-lowercased path equals its path child";
    "surf_dim_group_size" => S::PositiveInteger, P::Tag(T::Defdim), "a positive integer on the first member of an adjacent defdim group";
    "surf_pipe_stage" => S::Choices(&["call-first"]), P::PipeStage, "\"call-first\" on an fn at a non-initial pipe stage";
    "surf_literal_style" => S::Choices(&["unsuffixed", "explicit"]), P::Tag(T::Lit), "\"unsuffixed\" or \"explicit\" on lit";
    "surf_binding_type" => S::Choices(&["inferred", "explicit"]), P::BindingValue, "\"inferred\" or \"explicit\" on a bind value";
    "lin" => S::Names(&["once", "borrow", "unrestricted"]), P::Any, "once, borrow, or unrestricted";
    "doc" => S::String, P::Any, "a string";
}

enum KeyClass {
    Registered(&'static Rule),
    Extension,
    Forbidden,
}
fn classify(key: &str) -> KeyClass {
    match RULES.iter().find(|rule| rule.key == key) {
        Some(rule) => KeyClass::Registered(rule),
        None if key.starts_with("surf_") => KeyClass::Forbidden,
        None => KeyClass::Extension,
    }
}
pub(crate) fn role(key: &str) -> MetadataRole {
    match classify(key) {
        KeyClass::Registered(rule) => match rule.shape {
            S::Expression | S::Wrt => MetadataRole::Expression,
            S::Type => MetadataRole::Type,
            S::Source => MetadataRole::Preserved,
            S::Bounds => MetadataRole::BinderMap,
            S::String
            | S::Span
            | S::Choices(_)
            | S::Names(_)
            | S::True
            | S::PositiveInteger
            | S::Effects
            | S::Params
            | S::Expressions
            | S::Contracts
            | S::Invariant
            | S::Loc => MetadataRole::Syntax,
        },
        KeyClass::Extension | KeyClass::Forbidden => MetadataRole::Syntax,
    }
}

/// Borrowed syntax adapters keep one shape implementation for both AST stages.
#[derive(Clone, Copy)]
enum View<'a> {
    Raw(&'a RawExpr),
    Ast(&'a Expr),
}
type Entries<'a> = Vec<(&'a str, View<'a>)>;
struct Parts<'a> {
    tag: Option<DeepTag>,
    meta: Entries<'a>,
    children: Vec<View<'a>>,
}
impl<'a> View<'a> {
    fn span(self) -> Span {
        match self {
            Self::Raw(v) => v.span(),
            Self::Ast(v) => v.span(),
        }
    }
    fn name(self) -> Option<&'a str> {
        match self {
            Self::Raw(RawExpr::Atom(RawAtom::Symbol(v), _))
            | Self::Ast(Expr::Atom(Atom::Name(v), _)) => Some(v),
            _ => None,
        }
    }
    fn string(self) -> Option<&'a str> {
        match self {
            Self::Raw(RawExpr::Atom(RawAtom::Str(v), _))
            | Self::Ast(Expr::Atom(Atom::Str(v), _)) => Some(v),
            _ => None,
        }
    }
    fn integer(self) -> Option<i64> {
        match self {
            Self::Raw(RawExpr::Atom(RawAtom::Int(v), _))
            | Self::Ast(Expr::Atom(Atom::Int(v), _)) => Some(*v),
            _ => None,
        }
    }
    fn is_true(self) -> bool {
        matches!(
            self,
            Self::Raw(RawExpr::Atom(RawAtom::Bool(true), _))
                | Self::Ast(Expr::Atom(Atom::Bool(true), _))
        )
    }
    fn list(self) -> Option<Vec<Self>> {
        match self {
            Self::Raw(RawExpr::List(v, _)) => Some(v.iter().map(Self::Raw).collect()),
            Self::Ast(Expr::List(v, _)) => Some(v.elements.iter().map(Self::Ast).collect()),
            Self::Ast(Expr::BareList(v, _)) => Some(v.iter().map(Self::Ast).collect()),
            _ => None,
        }
    }
    fn map(self) -> Option<Entries<'a>> {
        match self {
            Self::Raw(RawExpr::Map(v, _)) => {
                Some(v.iter().map(|(k, v)| (k.as_str(), Self::Raw(v))).collect())
            }
            Self::Ast(Expr::Map(v, _)) => Some(
                v.entries
                    .iter()
                    .map(|(k, v)| (k.as_str(), Self::Ast(v)))
                    .collect(),
            ),
            _ => None,
        }
    }
    fn parts(self) -> Option<Parts<'a>> {
        match self {
            Self::Ast(Expr::Node(node, _)) => Some(Parts {
                tag: Some(node.tag()),
                meta: entries(node.meta()),
                children: node.children_slice().iter().map(Self::Ast).collect(),
            }),
            Self::Ast(Expr::UnknownForm(data)) => Some(Parts {
                tag: None,
                meta: entries(&data.meta),
                children: data.children.iter().map(Self::Ast).collect(),
            }),
            _ => {
                let list = self.list()?;
                let head = *list.first()?;
                let tag = match head {
                    Self::Ast(Expr::Atom(Atom::Tag(tag), _)) => Some(*tag),
                    _ => DeepTag::parse(head.name()?),
                };
                Some(Parts {
                    tag,
                    meta: list.get(1)?.map()?,
                    children: list[2..].to_vec(),
                })
            }
        }
    }
    fn node(self, tag: DeepTag) -> Option<Parts<'a>> {
        self.parts().filter(|v| v.tag == Some(tag))
    }
}
fn entries(meta: &MetaMap) -> Entries<'_> {
    meta.entries
        .iter()
        .map(|(k, v)| (k.as_str(), View::Ast(v)))
        .collect()
}
fn value<'a>(meta: &Entries<'a>, key: &str) -> Option<View<'a>> {
    meta.iter().find_map(|(k, v)| (*k == key).then_some(*v))
}
fn error(key: &str, v: View<'_>, expected: &'static str) -> MetadataError {
    MetadataError {
        key: key.into(),
        span: v.span(),
        expected,
        detail: None,
        forbidden_span_char: None,
    }
}
fn variable(v: View<'_>) -> bool {
    v.node(T::Var)
        .is_some_and(|n| n.children.len() == 1 && n.children[0].name().is_some())
}
fn expression(v: View<'_>) -> bool {
    match v {
        View::Raw(RawExpr::Atom(
            RawAtom::Int(_) | RawAtom::Float(_) | RawAtom::Bool(_) | RawAtom::Str(_),
            _,
        ))
        | View::Ast(Expr::Atom(Atom::Int(_) | Atom::Float(_) | Atom::Bool(_) | Atom::Str(_), _)) => {
            true
        }
        _ => v.parts().is_some_and(|p| p.tag.is_none_or(runtime_tag)),
    }
}
fn runtime_tag(tag: DeepTag) -> bool {
    match tag {
        T::Fn
        | T::App
        | T::Let
        | T::Match
        | T::If
        | T::Var
        | T::Lit
        | T::Record
        | T::Access
        | T::Pipe
        | T::Block
        | T::Tuple
        | T::TupleGet
        | T::RecordUpdate
        | T::Par
        | T::HandleEffect
        | T::Borrow
        | T::Grad
        | T::Vmap
        | T::Jit
        | T::Realize
        | T::Cast
        | T::Copy
        | T::Quote
        | T::Unquote
        | T::Splice => true,
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
        | T::Resource => false,
    }
}
fn binder(v: View<'_>) -> bool {
    v.name().is_some()
        || v.list()
            .is_some_and(|l| l.len() == 2 && l[0].name().is_some() && l[1].map().is_some())
        || matches!(v, View::Ast(Expr::MetaExpr(m, _)) if matches!(&*m.expr, Expr::Atom(Atom::Name(_), _)))
        || matches!(v, View::Raw(RawExpr::MetaExpr { expr, .. }) if matches!(&**expr, RawExpr::Atom(RawAtom::Symbol(_), _)))
}
fn type_shape_error(root: View<'_>) -> Option<String> {
    use crate::role::{
        AritySpec, TypeSyntaxRole, arity_contract, type_syntax_child_role,
        type_syntax_role_accepts_tag,
    };
    let mut stack = vec![(root, TypeSyntaxRole::Type)];
    while let Some((v, role)) = stack.pop() {
        if role == TypeSyntaxRole::Name {
            if v.name().is_none() {
                return Some(format!("expected {role:?} type syntax"));
            }
            continue;
        }
        if role == TypeSyntaxRole::Integer {
            if v.integer().is_none() {
                return Some(format!("expected {role:?} type syntax"));
            }
            continue;
        }
        let Some(p) = v.parts() else {
            return Some(format!("expected {role:?} type syntax"));
        };
        let Some(tag) = p.tag else {
            return Some(format!("expected {role:?} type syntax"));
        };
        if !type_syntax_role_accepts_tag(role, tag) {
            return Some(format!(
                "expected {role:?} type syntax, got {}",
                tag.as_str()
            ));
        }
        let count = p.children.len();
        let arity_ok = match arity_contract(tag) {
            AritySpec::Fixed(n) => count == n,
            AritySpec::AtLeast(n) => count >= n,
            AritySpec::Range(a, b) => (a..=b).contains(&count),
        };
        if !arity_ok {
            return Some(format!(
                "expected {role:?} type syntax: {} has {count} children, expected {arity:?}",
                tag.as_str(),
                arity = arity_contract(tag)
            ));
        }
        for (i, child) in p.children.into_iter().enumerate() {
            let Some(role) = type_syntax_child_role(tag, i, count) else {
                return Some(format!("expected {role:?} type syntax"));
            };
            stack.push((child, role));
        }
    }
    None
}

fn same_syntax(a: View<'_>, b: View<'_>) -> bool {
    if let Some(name) = a.name() {
        return b.name() == Some(name);
    }
    if let Some(text) = a.string() {
        return b.string() == Some(text);
    }
    if let Some(n) = a.integer() {
        return b.integer() == Some(n);
    }
    if let (Some(a), Some(b)) = (a.parts(), b.parts())
        && a.tag.is_some()
        && b.tag.is_some()
    {
        return a.tag == b.tag
            && same_sequence(&a.children, &b.children)
            && same_entries(&a.meta, &b.meta);
    }
    if let (Some(a), Some(b)) = (a.map(), b.map()) {
        return same_entries(&a, &b);
    }
    if let (Some(a), Some(b)) = (a.list(), b.list()) {
        return same_sequence(&a, &b);
    }
    match (a, b) {
        (
            View::Raw(RawExpr::Atom(RawAtom::Bool(a), _)),
            View::Raw(RawExpr::Atom(RawAtom::Bool(b), _)),
        ) => a == b,
        (View::Ast(Expr::Atom(Atom::Bool(a), _)), View::Ast(Expr::Atom(Atom::Bool(b), _))) => {
            a == b
        }
        _ => false,
    }
}
fn same_sequence(a: &[View<'_>], b: &[View<'_>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_syntax(*a, *b))
}
fn binder_signature(v: View<'_>) -> Option<(&str, Option<View<'_>>)> {
    if let Some(name) = v.name() {
        return Some((name, None));
    }
    let (name, meta) = match v {
        View::Raw(RawExpr::MetaExpr { entries, expr, .. }) => (
            View::Raw(expr).name()?,
            entries
                .iter()
                .map(|(k, v)| (k.as_str(), View::Raw(v)))
                .collect(),
        ),
        View::Ast(Expr::MetaExpr(m, _)) => (
            View::Ast(&m.expr).name()?,
            m.entries
                .iter()
                .map(|(k, v)| (k.as_str(), View::Ast(v)))
                .collect(),
        ),
        _ => {
            let items = v.list()?;
            if items.len() != 2 {
                return None;
            }
            (items[0].name()?, items[1].map()?)
        }
    };
    Some((name, value(&meta, "type")))
}
fn same_params(a: View<'_>, b: View<'_>) -> bool {
    let (Some(a), Some(b)) = (a.node(T::Params), b.node(T::Params)) else {
        return false;
    };
    a.children.len() == b.children.len()
        && a.children.into_iter().zip(b.children).all(|(a, b)| {
            match (binder_signature(a), binder_signature(b)) {
                (Some((a, at)), Some((b, bt))) if a == b => match (at, bt) {
                    (None, None) => true,
                    (Some(a), Some(b)) => same_syntax(a, b),
                    _ => false,
                },
                _ => false,
            }
        })
}
fn same_entries(a: &Entries<'_>, b: &Entries<'_>) -> bool {
    // Source locations are not part of the serialized binder signature.
    let relevant = |key: &str| !matches!(key, "span" | "loc" | "source");
    a.iter().filter(|(k, _)| relevant(k)).count() == b.iter().filter(|(k, _)| relevant(k)).count()
        && a.iter()
            .filter(|(k, _)| relevant(k))
            .all(|(k, v)| value(b, k).is_some_and(|other| same_syntax(*v, other)))
}

fn shape_valid(shape: Shape, v: View<'_>) -> bool {
    match shape {
        S::String | S::Span => v.string().is_some(),
        S::Choices(choices) => v.string().is_some_and(|v| choices.contains(&v)),
        S::Names(choices) => v.name().is_some_and(|v| choices.contains(&v)),
        S::True => v.is_true(),
        S::PositiveInteger => v.integer().is_some_and(|n| n > 0),
        S::Type => type_shape_error(v).is_none(),
        S::Effects => v.node(T::Effects).is_some_and(|n| {
            n.children.iter().all(|v| {
                v.name().is_some() || v.node(T::Resource).is_some_and(|n| n.children.len() == 1)
            })
        }),
        S::Params => v
            .node(T::Params)
            .is_some_and(|n| n.children.into_iter().all(binder)),
        S::Expressions => v
            .node(T::Tuple)
            .is_some_and(|n| n.children.into_iter().all(expression)),
        S::Contracts => v
            .node(T::Tuple)
            .is_some_and(|n| n.children.iter().all(|v| v.string().is_some())),
        S::Invariant => v.node(T::Fn).is_some_and(|n| {
            n.children.len() == 2
                && n.children[0]
                    .node(T::Params)
                    .is_some_and(|p| p.children.len() == 1 && binder(p.children[0]))
                && expression(n.children[1])
        }),
        S::Bounds => v.map().is_some_and(|m| {
            m.iter().enumerate().all(|(i, (k, v))| {
                !m[..i].iter().any(|(prior, _)| prior == k)
                    && v.name()
                        .and_then(crate::DtypeFamily::from_deep_name)
                        .is_some()
            })
        }),
        S::Source => v
            .list()
            .is_some_and(|l| l.first().is_some_and(|v| v.name().is_some())),
        S::Loc => v.list().is_some_and(|l| {
            l.len() == 4
                && l[0].name() == Some("loc")
                && l[1].string().is_some()
                && l[2].integer().is_some()
                && l[3].integer().is_some()
        }),
        S::Wrt => {
            variable(v)
                || v.node(T::Tuple)
                    .is_some_and(|n| !n.children.is_empty() && n.children.into_iter().all(variable))
        }
        S::Expression => expression(v),
    }
}

#[derive(Clone, Copy, Default)]
struct Context {
    binding_value: bool,
    pipe_stage: bool,
}
fn child_context(tag: Option<DeepTag>, index: usize) -> Context {
    Context {
        binding_value: tag == Some(T::Bind) && index % 2 == 1,
        pipe_stage: tag == Some(T::Pipe) && index > 0,
    }
}
fn check_entries(
    meta: &Entries<'_>,
    tag: Option<DeepTag>,
    children: &[View<'_>],
    context: Option<Context>,
) -> Result<(), MetadataError> {
    for (index, (key, v)) in meta.iter().enumerate() {
        let rule = match classify(key) {
            KeyClass::Extension => continue,
            KeyClass::Forbidden => {
                return Err(error(
                    key,
                    *v,
                    "a key in the closed Surf metadata namespace",
                ));
            }
            KeyClass::Registered(rule) => rule,
        };
        if meta[..index].iter().any(|(prior, _)| prior == key) {
            return Err(error(
                key,
                *v,
                "exactly one occurrence of this metadata key",
            ));
        }
        let placement = match rule.placement {
            P::Any => true,
            P::Declaration => {
                tag.is_some_and(crate::role::is_declaration_tag)
                    && (v.string() != Some("property") || tag == Some(T::Def))
            }
            P::Tag(owner) => tag == Some(owner),
            P::Path => {
                matches!(tag, Some(T::Module | T::Import | T::ImportAll))
                    && v.string().is_some_and(|path| {
                        children
                            .first()
                            .and_then(|c| c.name())
                            .is_some_and(|name| path.to_ascii_lowercase() == name)
                    })
            }
            P::BindingValue => {
                tag.is_some_and(runtime_tag) && context.is_none_or(|c| c.binding_value)
            }
            P::PipeStage => tag == Some(T::Fn) && context.is_none_or(|c| c.pipe_stage),
        };
        if !placement || !shape_valid(rule.shape, *v) {
            let mut e = error(key, *v, rule.expected);
            if *key == "dtype_bounds" && tag == Some(T::Def) {
                e.detail = Some(format!(
                    "`{}`: a declaration's `defsig` owns its binders",
                    children.first().and_then(|v| v.name()).unwrap_or("def")
                ));
            }
            if matches!(rule.shape, S::Type) {
                e.detail = type_shape_error(*v);
            }
            if matches!(rule.shape, S::Expression | S::Wrt)
                && let Some(name) = v.name()
            {
                e.detail = Some(format!(
                    "bare name `{name}` at expression slot; use `(var {{}} {name})`"
                ));
            }
            return Err(e);
        }
        if *key == "span"
            && let Some(s) = v.string()
            && let Some((i, b)) = s
                .bytes()
                .enumerate()
                .find(|(_, b)| *b <= 0x1f || *b == 0x7f)
        {
            let mut e = error(key, *v, rule.expected);
            e.detail = Some(format!("forbidden character at byte {i} (U+{b:04X})"));
            e.forbidden_span_char = Some((i, b));
            return Err(e);
        }
        if *key == "invariant" && !value(meta, "opaque").is_some_and(View::is_true) {
            return Err(error(
                key,
                *v,
                "opaque: true on the invariant's deftype (an invariant requires `@opaque`)",
            ));
        }
        if *key == "invariant_amenability" && value(meta, "invariant").is_none() {
            return Err(error(key, *v, "an invariant on the same deftype"));
        }
    }
    if tag == Some(T::Def)
        && let Some(marker) = value(meta, "chelis_role")
        && marker.string() == Some("property")
    {
        for key in [
            "property_source_kind",
            "property_quantifiers",
            "property_preconditions",
        ] {
            if value(meta, key).is_none() {
                return Err(error(key, marker, "the required property metadata field"));
            }
        }
        let quantifiers = value(meta, "property_quantifiers").expect("required above");
        let body = children.get(1).and_then(|v| v.node(T::Fn));
        let params = body.as_ref().and_then(|p| p.children.first()).copied();
        if !params.is_some_and(|p| same_params(quantifiers, p)) {
            return Err(error(
                "property_quantifiers",
                quantifiers,
                "parameter names and types matching the property fn",
            ));
        }
    }
    Ok(())
}
fn check_groups(items: &[View<'_>]) -> Result<(), MetadataError> {
    let mut i = 0;
    while i < items.len() {
        if let Some(parts) = items[i].node(T::Defdim)
            && let Some(v) = value(&parts.meta, "surf_dim_group_size")
        {
            let Some(size) = v
                .integer()
                .and_then(|v| usize::try_from(v).ok())
                .filter(|n| *n > 0 && *n <= items.len() - i)
            else {
                return Err(error(
                    "surf_dim_group_size",
                    v,
                    "a positive size fitting the adjacent defdim group",
                ));
            };
            if items[i + 1..i + size].iter().any(|v| {
                v.node(T::Defdim)
                    .is_none_or(|n| value(&n.meta, "surf_dim_group_size").is_some())
            }) {
                return Err(error(
                    "surf_dim_group_size",
                    v,
                    "adjacent defdim members with a marker only on the first member",
                ));
            }
            i += size;
        } else {
            i += 1;
        }
    }
    Ok(())
}
fn push_metadata<'a>(meta: Entries<'a>, stack: &mut Vec<(View<'a>, Context)>) {
    for (key, v) in meta.into_iter().rev() {
        // These are data roles. Their complete outer shape was checked above.
        if !matches!(role(key), MetadataRole::Preserved | MetadataRole::BinderMap) {
            stack.push((v, Context::default()));
        }
    }
}
fn walk(
    mut stack: Vec<(View<'_>, Context)>,
    descend_nodes: bool,
    groups: bool,
) -> Result<(), MetadataError> {
    while let Some((v, context)) = stack.pop() {
        if let Some(p) = v.parts() {
            check_entries(&p.meta, p.tag, &p.children, Some(context))?;
            if !descend_nodes && matches!(v, View::Ast(Expr::Node(..))) {
                continue;
            }
            if groups && p.tag == Some(T::Module) {
                check_groups(&p.children)?;
            }
            for (i, c) in p.children.into_iter().enumerate().rev() {
                stack.push((c, child_context(p.tag, i)));
            }
            push_metadata(p.meta, &mut stack);
        } else if let Some(meta) = v.map() {
            check_entries(&meta, None, &[], Some(context))?;
            push_metadata(meta, &mut stack);
        } else if let Some(list) = v.list() {
            stack.extend(list.into_iter().rev().map(|v| (v, Context::default())));
        } else {
            let (meta, body) = match v {
                View::Raw(RawExpr::MetaExpr { entries, expr, .. }) => (
                    entries
                        .iter()
                        .map(|(k, v)| (k.as_str(), View::Raw(v)))
                        .collect(),
                    View::Raw(expr),
                ),
                View::Ast(Expr::MetaExpr(m, _)) => (
                    m.entries
                        .iter()
                        .map(|(k, v)| (k.as_str(), View::Ast(v)))
                        .collect(),
                    View::Ast(&m.expr),
                ),
                _ => continue,
            };
            check_entries(&meta, None, &[], Some(context))?;
            stack.push((body, context));
            push_metadata(meta, &mut stack);
        }
    }
    Ok(())
}

pub(crate) fn validate_raw(exprs: &[RawExpr]) -> Result<(), MetadataError> {
    let views: Vec<_> = exprs.iter().map(View::Raw).collect();
    check_groups(&views)?;
    walk(
        views
            .into_iter()
            .rev()
            .map(|v| (v, Context::default()))
            .collect(),
        true,
        true,
    )
}
/// Validate a completed AST, including metadata in legacy and structural carriers.
pub fn validate_metadata(exprs: &[Expr]) -> Result<(), MetadataError> {
    let views: Vec<_> = exprs.iter().map(View::Ast).collect();
    check_groups(&views)?;
    walk(
        views
            .into_iter()
            .rev()
            .map(|v| (v, Context::default()))
            .collect(),
        true,
        true,
    )
}
/// Check a candidate Node without claiming a parent placement for its root.
pub(crate) fn validate_node(
    tag: DeepTag,
    meta: &MetaMap,
    children: &[Expr],
) -> Result<(), MetadataError> {
    let meta = entries(meta);
    let children: Vec<_> = children.iter().map(View::Ast).collect();
    check_entries(&meta, Some(tag), &children, None)?;
    if tag == T::Module {
        check_groups(&children)?;
    }
    let mut stack = Vec::new();
    for (i, c) in children.into_iter().enumerate().rev() {
        stack.push((c, child_context(Some(tag), i)));
    }
    push_metadata(meta, &mut stack);
    walk(stack, false, true)
}
