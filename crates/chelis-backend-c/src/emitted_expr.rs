//! Structured C-expression vocabulary for chelis#730 Phase 2.
//!
//! The host emitter may compose identifiers, literals, calls, and operators;
//! it cannot inject an arbitrary C string and call it an expression.  Open-set
//! dispatch therefore has only two outcomes: construct this closed AST, or
//! return `Err(Unsupported)`.  Rendering happens after construction.
//!
//! The representation is intentionally opaque outside this crate:
//!
//! ```compile_fail
//! use chelis_backend_c::emitted_expr::EmittedExpr;
//! let _: EmittedExpr = "arbitrary raw C text".to_string().into();
//! ```

use std::fmt;

/// A validated C identifier used by an expression node.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CIdentifier(String);

impl CIdentifier {
    fn new(identifier: impl Into<String>) -> Self {
        let identifier = identifier.into();
        assert!(
            is_c_identifier(&identifier),
            "emitted C identifier is not lexical C: {identifier:?}"
        );
        Self(identifier)
    }
}

fn is_c_identifier(identifier: &str) -> bool {
    let mut chars = identifier.chars();
    chars
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Less,
    Greater,
    GreaterEqual,
    LessEqual,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
}

impl BinaryOperator {
    fn spelling(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Remainder => "%",
            Self::Less => "<",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::LessEqual => "<=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::LogicalAnd => "&&",
            Self::LogicalOr => "||",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryOperator {
    Negate,
    LogicalNot,
}

impl UnaryOperator {
    fn spelling(self) -> &'static str {
        match self {
            Self::Negate => "-",
            Self::LogicalNot => "!",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CExpression {
    Identifier(CIdentifier),
    Integer(i64),
    Call {
        function: CIdentifier,
        args: Vec<EmittedExpr>,
    },
    Binary {
        operator: BinaryOperator,
        lhs: Box<EmittedExpr>,
        rhs: Box<EmittedExpr>,
    },
    Unary {
        operator: UnaryOperator,
        operand: Box<EmittedExpr>,
    },
    Conditional {
        condition: Box<EmittedExpr>,
        then_expr: Box<EmittedExpr>,
        else_expr: Box<EmittedExpr>,
    },
}

/// A C expression that was constructed from the closed node vocabulary.
///
/// There is deliberately no `raw`, `new(String)`, `From<String>`, or
/// `Default` implementation.  An unsupported dispatch arm cannot construct a
/// plausible expression payload ([05-UNS-1], chelis#730 C3/C4.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmittedExpr(CExpression);

impl EmittedExpr {
    pub(crate) fn identifier(identifier: impl Into<String>) -> Self {
        Self(CExpression::Identifier(CIdentifier::new(identifier)))
    }

    pub(crate) fn integer(value: i64) -> Self {
        Self(CExpression::Integer(value))
    }

    pub(crate) fn call(function: &'static str, args: impl IntoIterator<Item = Self>) -> Self {
        Self(CExpression::Call {
            function: CIdentifier::new(function),
            args: args.into_iter().collect(),
        })
    }

    pub(crate) fn binary(operator: BinaryOperator, lhs: Self, rhs: Self) -> Self {
        Self(CExpression::Binary {
            operator,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        })
    }

    pub(crate) fn unary(operator: UnaryOperator, operand: Self) -> Self {
        Self(CExpression::Unary {
            operator,
            operand: Box::new(operand),
        })
    }

    pub(crate) fn conditional(condition: Self, then_expr: Self, else_expr: Self) -> Self {
        Self(CExpression::Conditional {
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        })
    }

    pub(crate) fn as_c(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for EmittedExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            CExpression::Identifier(identifier) => f.write_str(&identifier.0),
            CExpression::Integer(value) => write!(f, "{value}"),
            CExpression::Call { function, args } => {
                write!(f, "{}(", function.0)?;
                for (index, arg) in args.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                f.write_str(")")
            }
            CExpression::Binary { operator, lhs, rhs } => {
                write!(f, "({lhs} {} {rhs})", operator.spelling())
            }
            CExpression::Unary { operator, operand } => {
                write!(f, "({}{operand})", operator.spelling())
            }
            CExpression::Conditional {
                condition,
                then_expr,
                else_expr,
            } => write!(f, "({condition} ? {then_expr} : {else_expr})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_tree_renders_only_after_construction() {
        let expr = EmittedExpr::call(
            "floor",
            [EmittedExpr::binary(
                BinaryOperator::Divide,
                EmittedExpr::identifier("a"),
                EmittedExpr::identifier("b"),
            )],
        );
        assert_eq!(expr.as_c(), "floor((a / b))");
    }

    #[test]
    fn operator_precedence_is_explicit_in_rendering() {
        let expr = EmittedExpr::binary(
            BinaryOperator::Multiply,
            EmittedExpr::binary(
                BinaryOperator::Add,
                EmittedExpr::identifier("a"),
                EmittedExpr::identifier("b"),
            ),
            EmittedExpr::identifier("c"),
        );
        assert_eq!(expr.as_c(), "((a + b) * c)");
    }

    #[test]
    #[should_panic(expected = "not lexical C")]
    fn arbitrary_c_text_cannot_launder_through_identifier_node() {
        let raw = ["/* unsupported */ ", "0"].concat();
        let _ = EmittedExpr::identifier(raw);
    }

    #[test]
    fn structured_vocabulary_has_no_raw_expression_node_or_constructor() {
        let source = include_str!("emitted_expr.rs");
        assert!(!source.contains(&["CExpression::", "Raw"].concat()));
        assert!(!source.contains(&["fn ", "raw("].concat()));
        assert!(!source.contains(&["From<", "String> for EmittedExpr"].concat()));
    }
}
