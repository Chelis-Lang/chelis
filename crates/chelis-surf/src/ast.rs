use chelis_deep::Span;
use serde::{Deserialize, Serialize};

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
        chelis_opaque: bool,
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
        dim_params: Vec<String>, // [a, b] dimension parameters
        params: Vec<Param>,
        ret_ty: Option<TypeExpr>,
        effects: Option<Vec<EffectExpr>>,
        body: Expr,
        span: Span,
    },
    Property {
        name: String,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PropertyOption {
    Tolerance(Expr, Span),
    Seed(Expr, Span),
    Samples(Expr, Span),
}

impl PropertyOption {
    pub fn span(&self) -> Span {
        match self {
            Self::Tolerance(_, span) | Self::Seed(_, span) | Self::Samples(_, span) => *span,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    Lit(Literal, Span),
    Var(String, Span),
    Constructor(String, Span),         // Uppercase name
    Apply(Box<Expr>, Vec<Expr>, Span), // f(x, y) or f x
    List(Vec<Expr>, Span),
    Record(String, Vec<(String, Expr)>, Span),
    Access(Box<Expr>, String, Span),
    TupleGet(Box<Expr>, i64, Span),
    Binary(BinOp, Box<Expr>, Box<Expr>, Span),
    Unary(UnaryOp, Box<Expr>, Span),
    Pipe(Box<Expr>, Vec<Expr>, Span), // x |> f |> g
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),
    Match(Box<Expr>, Vec<MatchArm>, Span),
    Lambda(Vec<Param>, Box<Expr>, Span), // fn (x, y) -> body
    Tuple(Vec<Expr>, Span),
    Cast(Box<Expr>, String, Span), // cast(x, f64)
    Grad(Box<Expr>, Option<Vec<String>>, Span),
    Vmap(Box<Expr>, Option<i64>, Span),
    Jit(Box<Expr>, Span),
    Realize(Box<Expr>, Span),
    Copy(Box<Expr>, Span),
    Borrow(Box<Expr>, Span),
    WithSeed(Box<Expr>, Box<Expr>, Span),
    WithDevice(Box<Expr>, Box<Expr>, Span),
    Par(Vec<Expr>, Span),                    // par { e1; e2; ... }
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
    /// Bare integer literal. Defaults to `int32` per spec §5.3 unless
    /// disambiguated by a suffix variant or surrounding context.
    Int(i64),
    /// Bare float literal. Defaults to `f32` per spec §5.3.
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExpr {
    Named(String, Span),                       // f32, bool, MyType
    Tensor(Vec<TypeExpr>, String, Span),       // tensor[batch, hidden, f32]
    Arrow(Vec<TypeExpr>, Box<TypeExpr>, Span), // A -> B -> C (flat)
    Ref(Box<TypeExpr>, Span),                  // &T
    App(String, Vec<TypeExpr>, Span),          // Option f32
    Tuple(Vec<TypeExpr>, Span),                // (f32, f32)
    Infer(Span),                               // _
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EffectExpr {
    Diff(Span),
    Random(Span),
    Accum(Span),
    Io(Span),
    Test(Span),
    Resource(String, Span),
}

impl EffectExpr {
    pub fn span(&self) -> Span {
        match self {
            Self::Diff(span)
            | Self::Random(span)
            | Self::Accum(span)
            | Self::Io(span)
            | Self::Test(span)
            | Self::Resource(_, span) => *span,
        }
    }
}
