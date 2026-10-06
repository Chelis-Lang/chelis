use chelis_deep::{DtypeBound, DtypeFamily, Span};
use serde::{Deserialize, Deserializer, Serialize};

/// One entry of a declaration's bracketed binder list
/// (`spec/02-surf-syntax.md` §P4b/§P4c).
///
/// The list is unkinded: a listed name resolves to a dimension variable, a
/// tensor-precision type variable, or a general type variable according to
/// where it occurs. A `bound` narrows the binder to one dtype family
/// (`spec/04-type-system.md` §5.9), which makes it a type binder only.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeBinder {
    /// The declared binder name.
    pub name: String,
    /// The declared dtype family, when the binder carries a bound.
    pub bound: Option<DtypeBound>,
}

impl TypeBinder {
    /// An unbounded binder, the form every `[a, b]` entry had before
    /// dtype-family bounds existed.
    pub fn unbounded(name: impl Into<String>) -> TypeBinder {
        TypeBinder {
            name: name.into(),
            bound: None,
        }
    }

    /// A binder carrying either `spec/02-surf-syntax.md` §P4c bound form.
    pub fn bounded(name: impl Into<String>, bound: DtypeBound) -> TypeBinder {
        TypeBinder {
            name: name.into(),
            bound: Some(bound),
        }
    }

    /// A binder bounded by a dtype family.
    pub fn bounded_by_family(name: impl Into<String>, family: DtypeFamily) -> TypeBinder {
        TypeBinder::bounded(name, DtypeBound::Family(family))
    }

    /// The canonical Surf spelling: `name`, `name: Family`, or
    /// `name: {d1, d2}` with members in §1.1 declaration order (§P4c).
    pub fn render(&self) -> String {
        match &self.bound {
            Some(bound) => format!("{}: {}", self.name, bound.surf_spelling()),
            None => self.name.clone(),
        }
    }
}

// ===== Declarations =====

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Decl {
    Module {
        name: String,
        decls: Vec<Decl>,
        span: Span,
    },
    Import {
        module: String,
        kind: ImportKind,
        span: Span,
    },
    Sig {
        name: String,
        /// Complete `[..]` binder list for every type, dimension, or rank
        /// variable used by the signature (`spec/02-surf-syntax.md` §P4b).
        type_binders: Vec<TypeBinder>,
        ty: TypeExpr,
        effects: Option<Vec<EffectExpr>>,
        span: Span,
    },
    Dim {
        names: Vec<String>,
        span: Span,
    },
    TypeDef {
        name: String,
        params: Vec<String>,
        variants: Vec<Variant>,
        opaque: bool,
        /// Declared invariant for an opaque type (RFC D-SYNTAX). Always
        /// `None` for a non-opaque type; an `@invariant` without
        /// `@opaque` is a parse error.
        invariant: Option<TypeInvariant>,
        span: Span,
    },
    TypeAlias {
        name: String,
        params: Vec<String>,
        ty: TypeExpr,
        span: Span,
    },
    FunDef {
        name: String,
        /// Complete `[a, b]` binder list; unkinded, optionally
        /// dtype-family bounded.
        type_binders: Vec<TypeBinder>,
        params: Vec<Param>,
        ret_ty: Option<TypeExpr>,
        effects: Option<Vec<EffectExpr>>,
        body: Expr,
        span: Span,
    },
    Property {
        name: String,
        /// Complete `[..]` binder list for every type, dimension, or rank
        /// variable used by this property declaration.
        type_binders: Vec<TypeBinder>,
        params: Vec<Param>,
        preconditions: Vec<Expr>,
        body: Expr,
        options: Vec<PropertyOption>,
        span: Span,
    },
    LetDef {
        name: String,
        ty: Option<TypeExpr>,
        value: Expr,
        span: Span,
    },
    MacroDef {
        name: String,
        params: Vec<String>,
        body: Expr,
        span: Span,
    },
    Export {
        names: Vec<String>,
        span: Span,
    },
}

impl Decl {
    /// Byte span of this declaration in the original Surf source.
    pub fn span(&self) -> Span {
        match self {
            Decl::Module { span, .. }
            | Decl::Import { span, .. }
            | Decl::Sig { span, .. }
            | Decl::Dim { span, .. }
            | Decl::TypeDef { span, .. }
            | Decl::TypeAlias { span, .. }
            | Decl::FunDef { span, .. }
            | Decl::Property { span, .. }
            | Decl::LetDef { span, .. }
            | Decl::MacroDef { span, .. }
            | Decl::Export { span, .. } => *span,
        }
    }
}

/// A declared invariant on an opaque type (RFC D-SYNTAX): a boolean
/// predicate over a single binder of the representation, written in Surf
/// at the declaration site. Recorded as the `invariant` metadata key on
/// the Deep `deftype` (RFC D-META).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeInvariant {
    /// The single binder name bound to a representation value.
    pub binder: String,
    /// The predicate body, evaluated with `binder` in scope.
    pub body: Expr,
    /// Byte span of the `@invariant(...) ...` block in the Surf source.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyOption {
    Tolerance(Expr, Span),
    Seed(Expr, Span),
    Samples(Expr, Span),
    Contract(String, Span),
}

impl PropertyOption {
    pub fn span(&self) -> Span {
        match self {
            Self::Tolerance(_, span)
            | Self::Seed(_, span)
            | Self::Samples(_, span)
            | Self::Contract(_, span) => *span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImportKind {
    Qualified,
    All,
    Names(Vec<String>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VariantFields {
    Positional(Vec<TypeExpr>),
    Record(Vec<(String, TypeExpr)>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Variant {
    pub name: String,
    pub fields: VariantFields,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub span: Span,
}

// ===== Expressions =====

/// The cast-ladder rung selector. Defined in `chelis-deep` (the Deep
/// node shape owns it) and re-exported so Surf consumers see one type.
pub use chelis_deep::CastMode;

/// Surf-only stage syntax. The expression preserves source spelling and spans;
/// the syntax discriminator never travels into Deep or depends on binder names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PipeStage {
    pub expression: Expr,
    pub syntax: PipeStageSyntax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipeStageSyntax {
    Callable,
    CallFirst,
    Cast(CastMode),
    Copy,
    Realize,
}

impl std::ops::Deref for PipeStage {
    type Target = Expr;
    fn deref(&self) -> &Expr {
        &self.expression
    }
}
impl std::ops::DerefMut for PipeStage {
    fn deref_mut(&mut self) -> &mut Expr {
        &mut self.expression
    }
}
impl From<Expr> for PipeStage {
    fn from(expression: Expr) -> Self {
        let syntax = if matches!(&expression, Expr::Apply(..))
            || matches!(&expression, Expr::Accumulate(call, _, _) if matches!(call.as_ref(), Expr::Apply(..)))
        {
            PipeStageSyntax::CallFirst
        } else {
            PipeStageSyntax::Callable
        };
        Self { expression, syntax }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Lit(Literal, Span),
    Var(String, Span),
    Constructor(String, Span),         // Uppercase name
    Apply(Box<Expr>, Vec<Expr>, Span), // f(x, y) or f x
    /// A call with an explicit accumulator dtype, `sum(x, 0i32,
    /// accumulator=f64)`: the call (always an `Apply`) and the dtype
    /// spelling (spec/02 `CallArgs`, spec/04 §5.7).
    Accumulate(Box<Expr>, String, Span),
    List(Vec<Expr>, Span),
    Record(String, Vec<(String, Expr)>, Span),
    RecordUpdate(Box<Expr>, Vec<(String, Expr)>, Span),
    Access(Box<Expr>, String, Span),
    TupleGet(Box<Expr>, i64, Span),
    Binary(BinOp, Box<Expr>, Box<Expr>, Span),
    Unary(UnaryOp, Box<Expr>, Span),
    Pipe(Box<Expr>, Vec<PipeStage>, Span), // x |> f |> g (Surf only)
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),
    Match(Box<Expr>, Vec<MatchArm>, Span),
    Lambda(Vec<Param>, Box<Expr>, Span), // fn (x, y) -> body
    Tuple(Vec<Expr>, Span),
    Cast(Box<Expr>, String, CastMode, Span), // cast(x, f64) / cast_trunc(x, i32)
    Grad(Box<Expr>, Option<Vec<String>>, Span),
    Vmap(Box<Expr>, Option<i64>, Span),
    Jit(Box<Expr>, Span),
    Realize(Box<Expr>, Span),
    Copy(Box<Expr>, Span),
    Borrow(Box<Expr>, Span),
    WithDevice(Box<Expr>, Box<Expr>, Span),
    Par(Vec<Expr>, Span),                    // par { e1; e2; ... }
    Do(Vec<Expr>, Span),                     // do { e1; e2; ... }
    Quote(Box<Expr>, Span),                  // quote(e)
    Unquote(Box<Expr>, Span),                // unquote(e)
    Splice(Box<Expr>, Span),                 // splice(e)
    Annotate(Box<Expr>, TypeExpr, Span),     // expr : Type
    Block(Vec<LetBinding>, Box<Expr>, Span), // { x = ...; expr }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LetBinding {
    pub pattern: LetPattern,
    pub ty: Option<TypeExpr>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LetPattern {
    Var(String, Span),
    Wildcard(Span),
    Tuple(Vec<LetPattern>, Span),
}

/// Closed set of literal-suffix dtypes per `spec/02-surf-syntax.md` §P10a.
/// Re-exported from `chelis_deep::lexer::LiteralSuffix` so the surf and
/// deep grammars share the same enum.
pub use chelis_deep::LiteralSuffix;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    /// Bare integer literal. Defaults to `i32` per spec §5.3 unless a
    /// construct that directly contains it states a dtype (spec/04 §5.6).
    Int(i64),
    /// Bare float literal. Defaults to `f32` per spec §5.3, with the same
    /// exception.
    Float(f64),
    /// Integer literal carrying an explicit precision suffix per spec §5.5.
    /// Suffix is part of the literal token; binds at exactly that
    /// precision with no inference, no widening, no narrowing.
    TypedInt(i64, LiteralSuffix),
    /// Float literal carrying an explicit precision suffix per spec §5.5.
    TypedFloat(f64, LiteralSuffix),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    Wildcard(Span),
    Var(String, Span),
    Lit(Literal, Span),
    Constructor(String, Vec<Pattern>, Span), // Some x, None, Pair a b
    Tuple(Vec<Pattern>, Span),
    Record(String, Vec<(String, Pattern)>, Span), // Ctor { field1, field2 }
    As(String, Box<Pattern>, Span),               // x @ Pattern
}

// ===== Type Expressions =====

/// A concrete dimension literal in a nominal type application.
///
/// The numeric carrier is private so an integer cannot be manufactured as an
/// ordinary type expression by downstream crates. The parser is the sole
/// source of this node; consumers may render it through [`Display`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DimensionLiteral(i64);

impl DimensionLiteral {
    pub(crate) fn new(value: i64) -> Self {
        Self(value)
    }

    pub(crate) fn value(self) -> i64 {
        self.0
    }
}

impl std::fmt::Display for DimensionLiteral {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// A tensor precision spelling with the exact token span that authored it.
///
/// Direct human-readable Serde of [`TypeExpr`] accepts the former string field
/// as a compatibility input. The compiler API's separate `surf_ast` wire type
/// remains string-valued and is not this Rust AST representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorPrecision {
    name: String,
    span: Span,
}

impl TensorPrecision {
    pub fn new(name: impl Into<String>, span: Span) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }

    pub fn as_str(&self) -> &str {
        &self.name
    }

    pub fn span(&self) -> Span {
        self.span
    }
}

impl<'de> Deserialize<'de> for TensorPrecision {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Current {
            name: String,
            span: Span,
        }

        if !deserializer.is_human_readable() {
            let Current { name, span } = Current::deserialize(deserializer)?;
            return Ok(Self { name, span });
        }

        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Representation {
            Current(Current),
            Legacy(String),
        }

        match Representation::deserialize(deserializer)? {
            Representation::Current(Current { name, span }) => Ok(Self { name, span }),
            Representation::Legacy(name) => Ok(Self {
                name,
                // The former human-readable field carried only the spelling,
                // so no exact token location can be reconstructed honestly.
                span: Span::new(0, 0),
            }),
        }
    }
}

impl std::fmt::Display for TensorPrecision {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.name.fmt(formatter)
    }
}

impl std::ops::Deref for TensorPrecision {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl PartialEq<str> for TensorPrecision {
    fn eq(&self, other: &str) -> bool {
        self.name == other
    }
}

impl PartialEq<&str> for TensorPrecision {
    fn eq(&self, other: &&str) -> bool {
        self.name == *other
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExpr {
    Named(String, Span),                          // f32, bool, MyType
    DimensionLiteral(DimensionLiteral, Span),     // integer only in Name[...]
    Tensor(Vec<TypeExpr>, TensorPrecision, Span), // tensor[batch, hidden, f32]
    Arrow(Vec<TypeExpr>, Box<TypeExpr>, Span),    // A -> B -> C (flat)
    Ref(Box<TypeExpr>, Span),                     // &T
    App(String, Vec<TypeExpr>, Span),             // Option f32
    Tuple(Vec<TypeExpr>, Span),                   // (f32, f32)
    Infer(Span),                                  // _
    RankSpread(String, Span),                     // ..r (rank variable; whole-shape spread)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EffectExpr {
    Diff(Span),
    Accum(Span),
    Io(Span),
    Test(Span),
    Resource(String, Span),
}

impl EffectExpr {
    pub fn span(&self) -> Span {
        match self {
            Self::Diff(span)
            | Self::Accum(span)
            | Self::Io(span)
            | Self::Test(span)
            | Self::Resource(_, span) => *span,
        }
    }
}
