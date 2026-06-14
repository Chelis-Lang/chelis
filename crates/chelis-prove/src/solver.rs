//! Solver trait abstraction for SMT backends.
//!
//! The `Solver` trait allows swapping between cvc5 library binding and
//! alternative backends (easy-smt subprocess, future Z3) without architectural
//! change.

/// Result of a satisfiability check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatResult {
    Sat,
    Unsat,
    Unknown,
}

/// A counterexample extracted from a SAT model.
#[derive(Debug, Clone)]
pub struct Model {
    pub bindings: Vec<(String, f64)>,
}

/// Abstract SMT solver interface.
///
/// Implementations must handle:
/// - Formula assertion
/// - Satisfiability checking with timeout
/// - Model extraction on SAT
/// - Scope management (push/pop)
pub trait Solver {
    type Error: std::fmt::Display;

    fn set_logic(&mut self, logic: &str) -> Result<(), Self::Error>;
    fn set_timeout(&mut self, ms: u64) -> Result<(), Self::Error>;
    fn assert(&mut self, expr: &SmtExpr) -> Result<(), Self::Error>;
    fn check_sat(&mut self) -> Result<SatResult, Self::Error>;
    fn get_model(&self) -> Result<Model, Self::Error>;
    fn push(&mut self) -> Result<(), Self::Error>;
    fn pop(&mut self) -> Result<(), Self::Error>;
}

/// Solver-agnostic SMT expression tree.
///
/// Produced by the predicate lowering pass (`lower.rs`), consumed by
/// solver-specific backends.
#[derive(Debug, Clone, PartialEq)]
pub enum SmtExpr {
    /// Real-valued variable.
    Var(String),
    /// Real literal.
    RealLit(f64),
    /// Integer literal.
    IntLit(i64),
    /// Boolean literal.
    BoolLit(bool),
    /// Arithmetic binary operation.
    Arith(ArithOp, Box<SmtExpr>, Box<SmtExpr>),
    /// Comparison.
    Cmp(CmpOp, Box<SmtExpr>, Box<SmtExpr>),
    /// Boolean connective.
    Bool(BoolOp, Vec<SmtExpr>),
    /// Negation.
    Not(Box<SmtExpr>),
    /// Universal quantifier: forall (vars with sorts) . body
    Forall(Vec<(String, SmtSort)>, Box<SmtExpr>),
    /// Existential quantifier.
    Exists(Vec<(String, SmtSort)>, Box<SmtExpr>),
    /// Uninterpreted function application.
    Apply(String, Vec<SmtExpr>),
    /// If-then-else.
    Ite(Box<SmtExpr>, Box<SmtExpr>, Box<SmtExpr>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Neg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmpOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
    Implies,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmtSort {
    Real,
    Int,
    Bool,
}
