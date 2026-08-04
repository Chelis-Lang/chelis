pub use chelis_deep::LiteralSuffix;
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
    Sig,
    Type,
    Dim,
    Macro,
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
    Realize,
    Copy,
    Tensor,
    Cast,
    CastTrunc,
    Export,
    Par,

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
    FatArrow,   // =>
    Pipe,       // |>
    Bar,        // |
    Eq,         // =
    Dot,        // .
    DotDot,     // .. (rank-variable spread marker)
    At,         // @
    Amp,        // &
    Semicolon,  // ;
    Newline,    // \n
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
    /// Numeric literal carrying an explicit precision suffix per
    /// `spec/02-surf-syntax.md` §P10a / `spec/04-type-system.md` §5.5.
    /// Emitted by the lexer when the suffix immediately follows the
    /// digit sequence with no intervening whitespace.
    TypedInt(i64, LiteralSuffix),
    TypedFloat(f64, LiteralSuffix),
    Str(String),
    True,
    False,

    // Identifiers
    Ident(String),     // lowercase start or _
    TypeIdent(String), // uppercase start

    Eof,
}
