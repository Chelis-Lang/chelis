use chelis_deep::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    Def,
    Let,
    In,
    Type,
    Match,
    With,
    Fn,
    Module,
    Import,
    If,
    Then,
    Else,
    Grad,
    Vmap,
    Jit,
    Tensor,
    Cast,
    Export,

    // Punctuation & Delimiters
    LParen,     // (
    RParen,     // )
    LBrace,     // {
    RBrace,     // }
    LBracket,   // [
    RBracket,   // ]
    Comma,      // ,
    Colon,      // :
    Arrow,      // ->
    Pipe,       // |>
    Bar,        // |
    Eq,         // =
    Dot,        // .
    Underscore, // _

    // Arithmetic operators
    Plus,    // +
    Minus,   // -
    Star,    // *
    Slash,   // /
    Percent, // %

    // Comparison operators
    EqEq,   // ==
    BangEq, // !=
    Lt,     // <
    Gt,     // >
    LtEq,   // <=
    GtEq,   // >=

    // Logical operators
    AmpAmp,   // &&
    PipePipe, // ||
    Bang,     // !

    // Literals
    Int(i64),
    Float(f64),
    Str(String),
    True,
    False,

    // Identifiers
    Ident(String),     // lowercase start or _
    TypeIdent(String), // uppercase start

    Eof,
}
