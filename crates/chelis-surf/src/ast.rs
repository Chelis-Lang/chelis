use chelis_deep::Span;

// ===== Declarations =====

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Module {
        name: String,
        decls: Vec<Decl>,
        span: Span,
    },
    Import {
        module: String,
        names: Option<Vec<String>>,
        span: Span,
    },
    TypeDef {
        name: String,
        params: Vec<String>,
        variants: Vec<Variant>,
        span: Span,
    },
    FunDef {
        name: String,
        dim_params: Vec<String>, // [a, b] dimension parameters
        params: Vec<Param>,
        ret_ty: Option<TypeExpr>,
        body: Expr,
        span: Span,
    },
    LetDef {
        name: String,
        ty: Option<TypeExpr>,
        value: Expr,
        span: Span,
    },
    Export {
        names: Vec<String>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum VariantFields {
    Positional(Vec<TypeExpr>),
    Record(Vec<(String, TypeExpr)>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variant {
    pub name: String,
    pub fields: VariantFields,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub span: Span,
}

// ===== Expressions =====

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Lit(Literal, Span),
    Var(String, Span),
    Constructor(String, Span),         // Uppercase name
    Apply(Box<Expr>, Vec<Expr>, Span), // f(x, y) or f x
    Binary(BinOp, Box<Expr>, Box<Expr>, Span),
    Unary(UnaryOp, Box<Expr>, Span),
    Pipe(Box<Expr>, Vec<Expr>, Span), // x |> f |> g
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),
    Match(Box<Expr>, Vec<MatchArm>, Span),
    Let(Vec<LetBinding>, Box<Expr>, Span), // let ... in ...
    Lambda(Vec<Param>, Box<Expr>, Span),   // fn (x, y) -> body
    Tuple(Vec<Expr>, Span),
    Cast(Box<Expr>, String, Span), // cast(x, f64)
    Grad(Box<Expr>, Span),
    Vmap(Box<Expr>, Option<i64>, Span),
    Jit(Box<Expr>, Span),
    Annotate(Box<Expr>, TypeExpr, Span), // expr : Type
    Block(Vec<Decl>, Box<Expr>, Span),   // { decls; expr }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetBinding {
    pub name: String,
    pub ty: Option<TypeExpr>,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
}

#[derive(Debug, Clone, Copy, PartialEq)]
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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
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

#[derive(Debug, Clone, PartialEq)]
pub enum TypeExpr {
    Named(String, Span),                       // f32, bool, MyType
    Tensor(Vec<TypeExpr>, String, Span),       // tensor[batch, hidden, f32]
    Arrow(Vec<TypeExpr>, Box<TypeExpr>, Span), // A -> B -> C (flat)
    App(String, Vec<TypeExpr>, Span),          // Option f32
    Tuple(Vec<TypeExpr>, Span),                // (f32, f32)
    Infer(Span),                               // _
}
