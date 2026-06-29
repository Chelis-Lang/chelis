//! WI-12 (WS-5): the Z3 NRA discharge engine.
//!
//! This is a second SMT [`DischargeEngine`] over the SAME [`GoalShape::Smt`]
//! the cvc5 engine ([`crate::discharge::Cvc5Engine`]) fits, backed by Z3's
//! nonlinear-real-arithmetic decision procedure. It is a RE-IMPLEMENTATION of
//! the lowering against the current main, not a rebase of the old `ws8-z3`
//! branch (which predates the #445 honesty-taxonomy restructure).
//!
//! # Why this mirrors `tier_b::lower_to_cvc5`
//!
//! The whole point of a second engine is that it reasons about the SAME goals
//! as cvc5 and agrees with it. So the Z3 lowering mirrors
//! [`crate::tier_b::lower_to_cvc5`] one-for-one:
//!
//! - The SAME [`SmtExpr`] tree is the input; nothing about the predicate
//!   surface changes.
//! - A [`SmtExpr::RealLit`] lowers to its EXACT f64 VALUE as a rational `n/d`
//!   ([`num_rational::BigRational::from_float`]), the SAME lowering the #444
//!   cvc5 RealLit fix uses ([`crate::tier_b::lower_to_cvc5`]'s `RealLit` arm),
//!   so cvc5 and Z3 reason about the IDENTICAL number the runtime f64 evaluator
//!   uses -- not the literal's exact decimal. A cross-engine oracle can then
//!   require the two engines AGREE.
//! - The lowering is TOTAL with respect to a panic / process abort: every
//!   operand-sort and arity requirement is checked and returns `Err` (routed to
//!   a clean [`TierBResult::Error`]) before any term is built. Z3's typed Rust
//!   bindings already return errors / panic rather than aborting the process
//!   the way cvc5's `mk_term` does, but keeping the same pre-checks gives
//!   byte-for-byte parity on WHICH inputs route to Tier C, and it never relies
//!   on a panic to enforce a contract.
//!
//! # The algebraic / transcendental capability split (honesty-critical)
//!
//! Z3 has a decision procedure for nonlinear POLYNOMIAL real arithmetic (NRA)
//! and the algebraic intrinsics `abs`/`min`/`max` (which lower to ITE +
//! comparison). It has NO built-in real transcendental function kind: unlike
//! cvc5 (which has `Z3_mk`-less EXPONENTIAL/SINE/COSINE/SQRT kinds and attempts
//! `QF_NRAT`), z3-sys exposes no `Z3_mk_sin`/`cos`/`exp`/`sqrt` over `Real`. So
//! a goal containing `exp`/`sqrt`/`sin`/`cos` lowers to a clean
//! [`TierBResult::Error`] here -- an HONEST capability boundary, not a fake
//! `unknown`. This is the cvc5->Z3 capability split the try-until-discharge
//! dispatcher exploits in BOTH directions: a polynomial goal cvc5 returns
//! `Unknown` on can fall through to Z3, and a transcendental goal Z3 cannot
//! lower stays a non-verdict so the dispatcher keeps looking.

use crate::discharge::{
    Discharge, DischargeEngine, Goal, GoalShape, QualifierSet, Soundness, classify_smt_outcome,
};
use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr, SmtSort};
use crate::tier_b::{SmtProperty, TierBResult};
use std::collections::HashMap;
use z3::ast::{Ast, Bool, Int, Real};
use z3::{Params, SatResult, Solver};

/// Maximum [`SmtExpr`] nesting depth the Z3 lowering will descend, mirroring
/// [`crate::tier_b`]'s `MAX_SMT_EXPR_DEPTH`. A property deeper than this routes
/// to Tier C rather than risking a stack overflow in the recursive lowering
/// (a stack overflow aborts the PROCESS, it does not return). The bound is the
/// SAME as the cvc5 path so the two engines route identically deep trees.
const MAX_SMT_EXPR_DEPTH: usize = 256;

/// The cvc5-lowerable algebraic intrinsics Z3 ALSO supports, paired with arity.
/// This is the Z3 analogue of [`crate::tier_b`]'s `CVC5_LOWERABLE`, MINUS the
/// transcendentals (`exp`/`sqrt`/`sin`/`cos`): Z3 has no real-transcendental
/// kind, so those are an honest Unsupported (see [`unsupported_apply_reason`]).
const Z3_LOWERABLE: &[(&str, usize)] = &[("abs", 1), ("min", 2), ("max", 2)];

/// Required arity for a Z3-lowerable intrinsic, or `None` if the name is not in
/// [`Z3_LOWERABLE`].
fn z3_lowerable_arity(name: &str) -> Option<usize> {
    Z3_LOWERABLE
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, a)| *a)
}

/// The reason an `Apply` of `name` cannot lower to a Z3 term. A predicate-
/// grammar transcendental (`exp`/`sqrt`/`sin`/`cos`/`log`) has no Z3 real kind,
/// so the Z3 NRA engine is honestly Unsupported on it -- a capability boundary,
/// not a bug. Distinguishing it from a truly-unknown symbol keeps the
/// diagnostic honest. Mirrors [`crate::tier_b`]'s `unsupported_apply_reason`
/// shape (names the function, frames the boundary), adapted for Z3's
/// polynomial-only fragment.
fn unsupported_apply_reason(name: &str) -> String {
    if chelis_pred::TRANSCENDENTAL_WHITELIST.contains(&name) && z3_lowerable_arity(name).is_none() {
        return format!(
            "transcendental function `{name}` is not supported by the Z3 NRA engine \
             (Z3 has no real-transcendental kind for it); the goal is Unsupported. \
             cvc5's QF_NRAT or an out-of-tree envelope engine (WS-7 Beacon) is the \
             faithful discharge"
        );
    }
    format!("unsupported function `{name}` in Z3 lowering (routes to Tier C)")
}

/// A lowered Z3 term tagged with the [`SmtSort`] Z3 assigns it. Carrying the
/// sort alongside the typed AST lets the parent node check operand sorts and
/// arity (the SAME checks the cvc5 path makes) without re-querying Z3.
enum Z3Term {
    Real(Real),
    Int(Int),
    Bool(Bool),
}

impl Z3Term {
    fn sort(&self) -> SmtSort {
        match self {
            Z3Term::Real(_) => SmtSort::Real,
            Z3Term::Int(_) => SmtSort::Int,
            Z3Term::Bool(_) => SmtSort::Bool,
        }
    }

    /// The term as a [`Bool`], or `Err` if it is not Bool-sorted. Used wherever
    /// Z3 requires a Bool operand (assert, NOT, boolean connectives, ITE cond).
    fn expect_bool(self, what: &str) -> Result<Bool, String> {
        match self {
            Z3Term::Bool(b) => Ok(b),
            other => Err(format!(
                "{what} has sort {:?}, expected Bool (routes to Tier C)",
                other.sort()
            )),
        }
    }

    /// The term coerced to a [`Real`], or `Err` if it is Bool. An `Int` is
    /// promoted with `Int::to_real`; this is how `min`/`max`/`abs`/division
    /// normalise a numeric operand to a single sort.
    fn as_real_numeric(&self, what: &str) -> Result<Real, String> {
        match self {
            Z3Term::Real(r) => Ok(r.clone()),
            Z3Term::Int(i) => Ok(i.to_real()),
            Z3Term::Bool(_) => Err(format!("{what} is Bool, not numeric (routes to Tier C)")),
        }
    }
}

/// Whether `expr` nests deeper than `max`, computed ITERATIVELY (an explicit
/// work stack) so the bound check itself cannot overflow the call stack on a
/// pathologically deep tree -- the whole point of running it before the
/// recursive lowering. Mirrors [`crate::tier_b`]'s `smt_expr_exceeds_depth`.
fn smt_expr_exceeds_depth(expr: &SmtExpr, max: usize) -> bool {
    let mut stack: Vec<(&SmtExpr, usize)> = vec![(expr, 1)];
    while let Some((node, depth)) = stack.pop() {
        if depth > max {
            return true;
        }
        let next = depth + 1;
        match node {
            SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {}
            SmtExpr::Not(inner) => stack.push((&**inner, next)),
            SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => stack.push((&**body, next)),
            SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
                stack.push((&**l, next));
                stack.push((&**r, next));
            }
            SmtExpr::Bool(_, children) | SmtExpr::Apply(_, children) => {
                for c in children {
                    stack.push((c, next));
                }
            }
            SmtExpr::Ite(c, t, e) => {
                stack.push((&**c, next));
                stack.push((&**t, next));
                stack.push((&**e, next));
            }
        }
    }
    false
}

/// Whether any part of a property nests past [`MAX_SMT_EXPR_DEPTH`]. Mirrors
/// [`crate::tier_b`]'s `property_exceeds_smt_depth`.
fn property_exceeds_smt_depth(property: &SmtProperty) -> bool {
    smt_expr_exceeds_depth(&property.postcondition, MAX_SMT_EXPR_DEPTH)
        || property
            .preconditions
            .iter()
            .any(|p| smt_expr_exceeds_depth(p, MAX_SMT_EXPR_DEPTH))
}

/// Lower a [`SmtExpr`] to a [`Z3Term`] (the typed Z3 AST plus its [`SmtSort`]).
///
/// TOTAL with respect to panics / aborts: every operand-sort and arity
/// requirement is checked and returns `Err` (routed to Tier C) before a term is
/// built, mirroring [`crate::tier_b::lower_to_cvc5`]. `vars` holds each declared
/// variable's Z3 term and is extended with quantifier bound vars on the way
/// down; `sorts` carries each variable's declared sort.
fn lower_to_z3(
    expr: &SmtExpr,
    vars: &HashMap<String, Z3Var>,
    sorts: &HashMap<String, SmtSort>,
) -> Result<Z3Term, String> {
    match expr {
        SmtExpr::Var(name) => {
            let var = vars
                .get(name)
                .ok_or_else(|| format!("variable `{name}` has no declared Z3 term"))?;
            let _ = sorts; // sorts mirrors vars; the Z3Var already carries the sort.
            Ok(var.term())
        }
        SmtExpr::RealLit(value) => {
            // Mirror the cvc5 RealLit fix (#444): lower to the literal's EXACT
            // f64 VALUE as a rational `n/d`, NOT its decimal spelling. cvc5 and
            // Z3 must reason about the SAME number the runtime f64 evaluator
            // uses (`0.1_f64` == 0.1000000000000000055..., not 1/10), so the
            // cross-engine oracle compares like for like and neither solver
            // proves a goal that is false at runtime by reading "0.1" as the
            // exact decimal. `from_float` returns `Some` for every finite f64
            // (only `None` on inf/NaN, rejected just above).
            if !value.is_finite() {
                return Err(format!(
                    "non-finite real literal `{value}` cannot lower to Z3 (routes to Tier C)"
                ));
            }
            let exact = num_rational::BigRational::from_float(*value).ok_or_else(|| {
                format!("real literal `{value}` has no exact rational (routes to Tier C)")
            })?;
            let lit =
                Real::from_rational_str(&exact.numer().to_string(), &exact.denom().to_string())
                    .ok_or_else(|| {
                        format!("Z3 rejected the exact rational for `{value}` (routes to Tier C)")
                    })?;
            Ok(Z3Term::Real(lit))
        }
        SmtExpr::IntLit(value) => Ok(Z3Term::Int(Int::from_i64(*value))),
        SmtExpr::BoolLit(value) => Ok(Z3Term::Bool(Bool::from_bool(*value))),
        // Unary `neg` is represented as `Arith(Neg, x, <placeholder>)`; lower
        // ONLY the real operand (mirroring the cvc5 path).
        SmtExpr::Arith(ArithOp::Neg, operand, _placeholder) => {
            match lower_to_z3(operand, vars, sorts)? {
                Z3Term::Real(r) => Ok(Z3Term::Real(r.unary_minus())),
                Z3Term::Int(i) => Ok(Z3Term::Int(i.unary_minus())),
                Z3Term::Bool(_) => {
                    Err("neg operand is Bool, not numeric (routes to Tier C)".to_string())
                }
            }
        }
        SmtExpr::Arith(op, left, right) => {
            let l = lower_to_z3(left, vars, sorts)?;
            let r = lower_to_z3(right, vars, sorts)?;
            lower_arith(*op, l, r)
        }
        SmtExpr::Cmp(op, left, right) => {
            let l = lower_to_z3(left, vars, sorts)?;
            let r = lower_to_z3(right, vars, sorts)?;
            lower_cmp(*op, l, r)
        }
        SmtExpr::Bool(op, children) => {
            let lowered: Vec<Bool> = children
                .iter()
                .map(|c| {
                    lower_to_z3(c, vars, sorts)
                        .and_then(|t| t.expect_bool("boolean connective operand"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            lower_bool(*op, lowered)
        }
        SmtExpr::Not(inner) => {
            let b = lower_to_z3(inner, vars, sorts)?.expect_bool("`not` operand")?;
            Ok(Z3Term::Bool(b.not()))
        }
        SmtExpr::Forall(bindings, body) | SmtExpr::Exists(bindings, body) => {
            let is_forall = matches!(expr, SmtExpr::Forall(_, _));
            lower_quantifier(is_forall, bindings, body, vars, sorts)
        }
        SmtExpr::Apply(name, args) => lower_apply(name, args, vars, sorts),
        SmtExpr::Ite(cond, then_expr, else_expr) => {
            let c = lower_to_z3(cond, vars, sorts)?.expect_bool("if-then-else condition")?;
            let t = lower_to_z3(then_expr, vars, sorts)?;
            let e = lower_to_z3(else_expr, vars, sorts)?;
            lower_ite(c, t, e)
        }
    }
}

/// Lower a binary arithmetic op. Mirrors the cvc5 path's sort discipline:
/// operands must be numeric; a mixed Int/Real pair (or division) normalises to
/// Real. Critically, `Div` ALWAYS yields a Real even over two Int operands --
/// cvc5's `/` is real division (it reports `result_sort = Real`), so to keep
/// the two engines reasoning about the same value Z3 must also do real division
/// for `Div`, NOT Z3's integer `Int::div`.
fn lower_arith(op: ArithOp, left: Z3Term, right: Z3Term) -> Result<Z3Term, String> {
    if matches!(op, ArithOp::Neg) {
        unreachable!("Neg handled in the dedicated arm in lower_to_z3");
    }
    // Real division, or any operand already Real, forces a Real result (cvc5
    // parity). Otherwise two Int operands stay Int.
    let both_int = matches!((&left, &right), (Z3Term::Int(_), Z3Term::Int(_)));
    if both_int && !matches!(op, ArithOp::Div) {
        let (Z3Term::Int(l), Z3Term::Int(r)) = (left, right) else {
            unreachable!("both_int checked above")
        };
        let term = match op {
            ArithOp::Add => Int::add(&[l, r]),
            ArithOp::Sub => Int::sub(&[l, r]),
            ArithOp::Mul => Int::mul(&[l, r]),
            ArithOp::Div | ArithOp::Neg => unreachable!("Div/Neg handled elsewhere"),
        };
        return Ok(Z3Term::Int(term));
    }
    // Real result: promote both operands to Real (an Int via `to_real`), reject
    // a Bool operand. This is the cvc5-Div-is-Real parity path.
    let l = left.as_real_numeric("arithmetic operand")?;
    let r = right.as_real_numeric("arithmetic operand")?;
    let term = match op {
        ArithOp::Add => Real::add(&[l, r]),
        ArithOp::Sub => Real::sub(&[l, r]),
        ArithOp::Mul => Real::mul(&[l, r]),
        ArithOp::Div => l.div(&r),
        ArithOp::Neg => unreachable!("Neg handled in lower_to_z3"),
    };
    Ok(Z3Term::Real(term))
}

/// Lower a comparison. Both operands must share one sort (a mixed Int/Real pair
/// is promoted to Real for the comparison, matching the value the cvc5 path
/// compares); a numeric comparison (Lt/Le/Gt/Ge) over Bool operands is an Err,
/// while Eq/Ne admit any equal sort (including Bool == Bool).
fn lower_cmp(op: CmpOp, left: Z3Term, right: Z3Term) -> Result<Z3Term, String> {
    let is_numeric_cmp = matches!(op, CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge);

    // Eq/Ne over two Bools: compare as Bool (cvc5 admits Bool == Bool).
    if let (Z3Term::Bool(l), Z3Term::Bool(r)) = (&left, &right) {
        if is_numeric_cmp {
            return Err("numeric comparison over a Bool operand (routes to Tier C)".to_string());
        }
        let eq = l.eq(r);
        return Ok(Z3Term::Bool(match op {
            CmpOp::Eq => eq,
            CmpOp::Ne => eq.not(),
            _ => unreachable!("numeric cmp over Bool rejected above"),
        }));
    }

    // Eq/Ne over two Ints: compare as Int (avoid a needless Real promotion so
    // the model and the comparison stay in the integer sort).
    if !is_numeric_cmp {
        if let (Z3Term::Int(l), Z3Term::Int(r)) = (&left, &right) {
            let eq = l.eq(r);
            return Ok(Z3Term::Bool(match op {
                CmpOp::Eq => eq,
                CmpOp::Ne => eq.not(),
                _ => unreachable!(),
            }));
        }
    } else if let (Z3Term::Int(l), Z3Term::Int(r)) = (&left, &right) {
        // Numeric comparison over two Ints: compare in the integer sort.
        let b = match op {
            CmpOp::Lt => l.lt(r),
            CmpOp::Le => l.le(r),
            CmpOp::Gt => l.gt(r),
            CmpOp::Ge => l.ge(r),
            _ => unreachable!(),
        };
        return Ok(Z3Term::Bool(b));
    }

    // Any remaining numeric pair (Real-vs-Real or mixed Int/Real): promote both
    // to Real and compare. A Bool here is rejected by `as_real_numeric`.
    let l = left.as_real_numeric("comparison operand")?;
    let r = right.as_real_numeric("comparison operand")?;
    let b = match op {
        CmpOp::Lt => l.lt(&r),
        CmpOp::Le => l.le(&r),
        CmpOp::Gt => l.gt(&r),
        CmpOp::Ge => l.ge(&r),
        CmpOp::Eq => l.eq(&r),
        CmpOp::Ne => l.eq(&r).not(),
    };
    Ok(Z3Term::Bool(b))
}

/// Lower a boolean connective over already-Bool children. Z3's `and`/`or`
/// accept any arity (0 -> true / false); `implies` requires exactly 2. Mirrors
/// the cvc5 path's normalisation of the degenerate arities so a single-conjunct
/// invariant still lowers rather than erroring.
fn lower_bool(op: BoolOp, children: Vec<Bool>) -> Result<Z3Term, String> {
    let term = match op {
        BoolOp::And => match children.len() {
            0 => Bool::from_bool(true),
            1 => children.into_iter().next().expect("len == 1"),
            _ => Bool::and(&children),
        },
        BoolOp::Or => match children.len() {
            0 => Bool::from_bool(false),
            1 => children.into_iter().next().expect("len == 1"),
            _ => Bool::or(&children),
        },
        BoolOp::Implies => {
            if children.len() != 2 {
                return Err(format!(
                    "`implies` requires exactly 2 operands, got {} (routes to Tier C)",
                    children.len()
                ));
            }
            let mut it = children.into_iter();
            let a = it.next().expect("len == 2");
            let b = it.next().expect("len == 2");
            a.implies(&b)
        }
    };
    Ok(Z3Term::Bool(term))
}

/// Lower an if-then-else. The condition is Bool; the branches must share a
/// sort. A mixed Int/Real branch pair is promoted to Real (matching the value
/// the cvc5 path produces); a Bool/numeric mix is an Err.
fn lower_ite(cond: Bool, then_term: Z3Term, else_term: Z3Term) -> Result<Z3Term, String> {
    match (then_term, else_term) {
        (Z3Term::Bool(t), Z3Term::Bool(e)) => Ok(Z3Term::Bool(cond.ite(&t, &e))),
        (Z3Term::Int(t), Z3Term::Int(e)) => Ok(Z3Term::Int(cond.ite(&t, &e))),
        (Z3Term::Real(t), Z3Term::Real(e)) => Ok(Z3Term::Real(cond.ite(&t, &e))),
        // Mixed numeric branches -> promote both to Real.
        (t @ (Z3Term::Int(_) | Z3Term::Real(_)), e @ (Z3Term::Int(_) | Z3Term::Real(_))) => {
            let t = t.as_real_numeric("if-then-else branch")?;
            let e = e.as_real_numeric("if-then-else branch")?;
            Ok(Z3Term::Real(cond.ite(&t, &e)))
        }
        (t, e) => Err(format!(
            "if-then-else branches have differing sorts {:?} vs {:?} (routes to Tier C)",
            t.sort(),
            e.sort()
        )),
    }
}

/// Lower an intrinsic application. Z3's lowerable set is the ALGEBRAIC subset
/// (`abs`/`min`/`max`); a transcendental (`exp`/`sqrt`/`sin`/`cos`/`log`) has no
/// Z3 real kind and is an honest Unsupported `Err` (see
/// [`unsupported_apply_reason`]). `min`/`max` lower to ITE + comparison and
/// `abs` to ITE on the sign, exactly as the cvc5 path builds them.
fn lower_apply(
    name: &str,
    args: &[SmtExpr],
    vars: &HashMap<String, Z3Var>,
    sorts: &HashMap<String, SmtSort>,
) -> Result<Z3Term, String> {
    let arity = z3_lowerable_arity(name).ok_or_else(|| unsupported_apply_reason(name))?;
    if args.len() != arity {
        return Err(format!(
            "intrinsic `{name}` expects {arity} argument(s), got {} (routes to Tier C)",
            args.len()
        ));
    }
    let lowered: Vec<Z3Term> = args
        .iter()
        .map(|a| lower_to_z3(a, vars, sorts))
        .collect::<Result<Vec<_>, _>>()?;

    match name {
        // abs(x) = ite(x < 0, -x, x), preserving the operand's numeric sort.
        "abs" => match &lowered[0] {
            Z3Term::Real(r) => {
                let zero = Real::from_rational_str("0", "1").expect("0/1 is a valid rational");
                let cond = r.lt(&zero);
                Ok(Z3Term::Real(cond.ite(&r.unary_minus(), r)))
            }
            Z3Term::Int(i) => {
                let zero = Int::from_i64(0);
                let cond = i.lt(&zero);
                Ok(Z3Term::Int(cond.ite(&i.unary_minus(), i)))
            }
            Z3Term::Bool(_) => {
                Err("abs operand is Bool, not numeric (routes to Tier C)".to_string())
            }
        },
        // min(a,b) = ite(a < b, a, b); max(a,b) = ite(a > b, a, b). Both
        // operands normalise to one sort (a mixed pair promotes to Real),
        // matching the value the cvc5 path's min/max produces.
        "min" | "max" => {
            let both_int = matches!((&lowered[0], &lowered[1]), (Z3Term::Int(_), Z3Term::Int(_)));
            if both_int {
                let (Z3Term::Int(a), Z3Term::Int(b)) = (&lowered[0], &lowered[1]) else {
                    unreachable!("both_int checked")
                };
                let cond = if name == "min" { a.lt(b) } else { a.gt(b) };
                Ok(Z3Term::Int(cond.ite(a, b)))
            } else {
                let a = lowered[0].as_real_numeric("min/max operand")?;
                let b = lowered[1].as_real_numeric("min/max operand")?;
                let cond = if name == "min" { a.lt(&b) } else { a.gt(&b) };
                Ok(Z3Term::Real(cond.ite(&a, &b)))
            }
        }
        // Unreachable after the z3_lowerable_arity guard; Err rather than panic
        // if Z3_LOWERABLE ever gains a name with no match arm.
        other => Err(unsupported_apply_reason(other)),
    }
}

/// Lower a quantifier. cvc5 normalises an empty binder list to the body; Z3's
/// `forall_const`/`exists_const` likewise just return the body when given no
/// bound consts, but we normalise explicitly for parity and clarity. Each bound
/// variable is declared as a fresh Z3 const of its sort and added to an extended
/// scope before the body is lowered.
fn lower_quantifier(
    is_forall: bool,
    bindings: &[(String, SmtSort)],
    body: &SmtExpr,
    vars: &HashMap<String, Z3Var>,
    sorts: &HashMap<String, SmtSort>,
) -> Result<Z3Term, String> {
    let mut extended_vars = vars.clone();
    let mut extended_sorts = sorts.clone();
    let mut bound_consts: Vec<Z3Var> = Vec::with_capacity(bindings.len());
    for (name, sort) in bindings {
        let var = Z3Var::new(name, *sort);
        bound_consts.push(var.clone());
        extended_vars.insert(name.clone(), var);
        extended_sorts.insert(name.clone(), *sort);
    }
    let body_bool =
        lower_to_z3(body, &extended_vars, &extended_sorts)?.expect_bool("quantifier body")?;

    // A quantifier over no variables is degenerate (`forall (). P` == `P`);
    // return the body directly rather than building an empty bound list.
    if bound_consts.is_empty() {
        return Ok(Z3Term::Bool(body_bool));
    }

    // `forall_const`/`exists_const` take the bound consts as `&[&dyn Ast]`.
    let dyn_bounds: Vec<&dyn Ast> = bound_consts.iter().map(Z3Var::as_dyn_ast).collect();
    let quantified = if is_forall {
        z3::ast::forall_const(&dyn_bounds, &[], &body_bool)
    } else {
        z3::ast::exists_const(&dyn_bounds, &[], &body_bool)
    };
    Ok(Z3Term::Bool(quantified))
}

/// A declared Z3 variable: the typed constant plus its [`SmtSort`]. Cloning a
/// `Z3Var` clones the underlying Z3 AST handle (a refcounted pointer), so an
/// extended quantifier scope shares the same constant.
#[derive(Clone)]
enum Z3Var {
    Real(Real),
    Int(Int),
    Bool(Bool),
}

impl Z3Var {
    /// Declare a fresh Z3 constant named `name` of `sort`.
    fn new(name: &str, sort: SmtSort) -> Self {
        match sort {
            SmtSort::Real => Z3Var::Real(Real::new_const(name)),
            SmtSort::Int => Z3Var::Int(Int::new_const(name)),
            SmtSort::Bool => Z3Var::Bool(Bool::new_const(name)),
        }
    }

    /// A fresh [`Z3Term`] referring to this variable (clones the AST handle).
    fn term(&self) -> Z3Term {
        match self {
            Z3Var::Real(r) => Z3Term::Real(r.clone()),
            Z3Var::Int(i) => Z3Term::Int(i.clone()),
            Z3Var::Bool(b) => Z3Term::Bool(b.clone()),
        }
    }

    /// The variable as a `&dyn Ast` for `forall_const`/`exists_const`.
    fn as_dyn_ast(&self) -> &dyn Ast {
        match self {
            Z3Var::Real(r) => r,
            Z3Var::Int(i) => i,
            Z3Var::Bool(b) => b,
        }
    }
}

/// Solve a structured SMT property with Z3. Mirrors
/// [`crate::tier_b::solve_property_cvc5`]'s control flow: depth-guard first,
/// declare variables, assert preconditions (each must lower to Bool), assert
/// the NEGATED postcondition, then `check`. UNSAT -> [`TierBResult::Proved`];
/// SAT -> [`TierBResult::Disproved`] with the model; unknown -> Unknown; a
/// lowering error or timeout -> the corresponding non-verdict.
pub fn solve_property_z3(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
    if property_exceeds_smt_depth(property) {
        return TierBResult::Error(format!(
            "property nests deeper than {MAX_SMT_EXPR_DEPTH} levels (routes to Tier C)"
        ));
    }

    let solver = Solver::new();
    let mut params = Params::new();
    // Z3 reads the per-check wall-clock budget from the `timeout` param (ms).
    params.set_u32("timeout", timeout_ms.min(u32::MAX as u64) as u32);
    solver.set_params(&params);

    // 1. Declare variables.
    let mut vars: HashMap<String, Z3Var> = HashMap::new();
    let mut sorts: HashMap<String, SmtSort> = HashMap::new();
    for (name, sort) in &property.variables {
        vars.insert(name.clone(), Z3Var::new(name, *sort));
        sorts.insert(name.clone(), *sort);
    }

    // 2. Assert preconditions (each must be Bool).
    for pre in &property.preconditions {
        match lower_to_z3(pre, &vars, &sorts) {
            Ok(term) => match term.expect_bool("precondition") {
                Ok(b) => solver.assert(&b),
                Err(reason) => return TierBResult::Error(reason),
            },
            Err(reason) => return TierBResult::Error(reason),
        }
    }

    // 3. Assert the NEGATED postcondition (UNSAT of the negation == proved).
    let post = match lower_to_z3(&property.postcondition, &vars, &sorts) {
        Ok(term) => match term.expect_bool("postcondition") {
            Ok(b) => b,
            Err(reason) => return TierBResult::Error(reason),
        },
        Err(reason) => return TierBResult::Error(reason),
    };
    solver.assert(post.not());

    // 4. Check.
    match solver.check() {
        SatResult::Unsat => TierBResult::Proved,
        SatResult::Sat => {
            let model = solver.get_model();
            let bindings = match model {
                Some(m) => {
                    let mut map = serde_json::Map::new();
                    for (name, var) in &vars {
                        let rendered = render_model_value(&m, var);
                        map.insert(name.clone(), serde_json::Value::String(rendered));
                    }
                    serde_json::Value::Object(map)
                }
                // SAT with no extractable model: still a counterexample exists.
                None => serde_json::json!({ "__model": "unavailable" }),
            };
            TierBResult::Disproved(bindings)
        }
        SatResult::Unknown => TierBResult::Unknown,
    }
}

/// Render a variable's value in a SAT model as a string, mirroring the cvc5
/// path's `val.to_string()` counterexample binding. A Real is rendered as a
/// decimal approximation (Z3's exact rationals do not always fit `as_rational`);
/// an Int as its integer; a Bool as `true`/`false`.
fn render_model_value(model: &z3::Model, var: &Z3Var) -> String {
    match var {
        Z3Var::Real(r) => model
            .eval(r, true)
            .map(|v| v.approx(17))
            .unwrap_or_else(|| "?".to_string()),
        Z3Var::Int(i) => model
            .eval(i, true)
            .and_then(|v| v.as_i64())
            .map(|n| n.to_string())
            .unwrap_or_else(|| "?".to_string()),
        Z3Var::Bool(b) => model
            .eval(b, true)
            .and_then(|v| v.as_bool())
            .map(|x| x.to_string())
            .unwrap_or_else(|| "?".to_string()),
    }
}

/// The Z3 NRA discharge engine. Discharges [`GoalShape::Smt`] goals through Z3,
/// classifying the outcome via the SHARED, engine-agnostic
/// [`classify_smt_outcome`] -- so a Z3 `Proved`/`Disproved` carries byte-
/// identical `(soundness, qualifier_set)` to a cvc5 one and projects through
/// the verdict algebra to the same `proven_modulo_real_arithmetic` /
/// `disproved_modulo_real_arithmetic`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Z3Engine;

impl Z3Engine {
    pub const fn new() -> Self {
        Self
    }
}

impl DischargeEngine for Z3Engine {
    fn name(&self) -> &'static str {
        "z3"
    }

    fn fitness(&self, goal: &Goal) -> bool {
        // Z3 discharges structured-SMT goals, the SAME shape cvc5 fits. The
        // WI-5 box/output-range shape is Beacon's; Z3 does not own it.
        matches!(goal.shape, GoalShape::Smt(_))
    }

    fn discharge(&self, goal: &Goal, timeout_ms: u64) -> Discharge {
        let result = match goal.as_smt() {
            Some(property) => solve_property_z3(property, timeout_ms),
            None => {
                TierBResult::Error("z3 engine cannot discharge a non-SMT goal shape".to_string())
            }
        };
        let (soundness, qualifier_set) = classify_smt_outcome(&result);
        // chelis#496: the canonical attribution is the top-level `"engine"` key
        // (`"z3"`); the z3 engine carries no extra backend detail on the normal
        // path. The shared classification only pairs `real_arithmetic` with
        // `SoundApproximate` (its floor) or returns an empty set at `Untrusted`,
        // so this constructor cannot fail here; surfacing the error as a
        // non-proof discharge keeps the seam total (mirrors the cvc5 engine).
        Discharge::with_engine_attribution(
            "z3",
            soundness,
            qualifier_set,
            result,
            serde_json::Value::Null,
        )
        .unwrap_or_else(|err| {
            Discharge::with_engine_attribution(
                "z3",
                Soundness::Untrusted,
                QualifierSet::new(),
                TierBResult::Error(err.to_string()),
                serde_json::json!({ "internal_error": err.to_string() }),
            )
            .expect("untrusted discharge with empty qualifier set is always valid")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discharge::{IntervalBox, OutputRange, Qualifier};

    fn real_var_prop(post: SmtExpr) -> SmtProperty {
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: post,
        }
    }

    fn cmp(op: CmpOp, l: SmtExpr, r: SmtExpr) -> SmtExpr {
        SmtExpr::Cmp(op, Box::new(l), Box::new(r))
    }

    fn var(name: &str) -> SmtExpr {
        SmtExpr::Var(name.to_string())
    }

    fn mul(l: SmtExpr, r: SmtExpr) -> SmtExpr {
        SmtExpr::Arith(ArithOp::Mul, Box::new(l), Box::new(r))
    }

    // --- lowering / solve happy path ---

    #[test]
    fn proves_x_squared_non_negative() {
        // forall x: x*x >= 0 -- a polynomial NRA fact Z3 decides.
        let prop = real_var_prop(cmp(
            CmpOp::Ge,
            mul(var("x"), var("x")),
            SmtExpr::RealLit(0.0),
        ));
        assert_eq!(solve_property_z3(&prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn disproves_x_always_positive_with_a_model() {
        // forall x: x > 0 -- false (x = 0 is a counterexample). SAT with model.
        let prop = real_var_prop(cmp(CmpOp::Gt, var("x"), SmtExpr::RealLit(0.0)));
        match solve_property_z3(&prop, 5000) {
            TierBResult::Disproved(model) => {
                assert!(model.is_object(), "model is a JSON object: {model:?}");
                assert!(model.get("x").is_some(), "model binds x: {model:?}");
            }
            other => panic!("expected Disproved, got {other:?}"),
        }
    }

    #[test]
    fn proves_with_precondition() {
        // forall x where x > 0: x >= 0
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![cmp(CmpOp::Gt, var("x"), SmtExpr::RealLit(0.0))],
            postcondition: cmp(CmpOp::Ge, var("x"), SmtExpr::RealLit(0.0)),
        };
        assert_eq!(solve_property_z3(&prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn proves_intrinsic_value_non_negative_via_ite() {
        // intrinsic_value(S, K) = if S > K then S - K else 0; the value is >= 0.
        let prop = SmtProperty {
            variables: vec![
                ("S".to_string(), SmtSort::Real),
                ("K".to_string(), SmtSort::Real),
            ],
            preconditions: vec![],
            postcondition: cmp(
                CmpOp::Ge,
                SmtExpr::Ite(
                    Box::new(cmp(CmpOp::Gt, var("S"), var("K"))),
                    Box::new(SmtExpr::Arith(
                        ArithOp::Sub,
                        Box::new(var("S")),
                        Box::new(var("K")),
                    )),
                    Box::new(SmtExpr::RealLit(0.0)),
                ),
                SmtExpr::RealLit(0.0),
            ),
        };
        assert_eq!(solve_property_z3(&prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn proves_max_with_zero_is_non_negative() {
        // max(x, 0) >= 0 -- exercises the min/max ITE lowering.
        let prop = real_var_prop(cmp(
            CmpOp::Ge,
            SmtExpr::Apply("max".to_string(), vec![var("x"), SmtExpr::RealLit(0.0)]),
            SmtExpr::RealLit(0.0),
        ));
        assert_eq!(solve_property_z3(&prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn proves_abs_is_non_negative() {
        // abs(x) >= 0 -- exercises the abs ITE lowering.
        let prop = real_var_prop(cmp(
            CmpOp::Ge,
            SmtExpr::Apply("abs".to_string(), vec![var("x")]),
            SmtExpr::RealLit(0.0),
        ));
        assert_eq!(solve_property_z3(&prop, 5000), TierBResult::Proved);
    }

    // --- exact-f64 literal lowering (the #444 parity, positive + negative) ---

    #[test]
    fn exact_f64_literal_lowering_disproves_a_decimal_false_goal() {
        // 0.1_f64 + 0.2_f64 == 0.3 holds in EXACT REALS but is FALSE at runtime
        // f64 (the sum is 0.30000000000000004). Because RealLit lowers to the
        // EXACT f64 VALUE (not the decimal 1/10 + 2/10 == 3/10), Z3 sees three
        // DISTINCT exact rationals and must DISPROVE the equality -- the same
        // verdict the runtime f64 evaluator gives. A decimal lowering would
        // wrongly PROVE it.
        let prop = SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: cmp(
                CmpOp::Eq,
                SmtExpr::Arith(
                    ArithOp::Add,
                    Box::new(SmtExpr::RealLit(0.1)),
                    Box::new(SmtExpr::RealLit(0.2)),
                ),
                SmtExpr::RealLit(0.3),
            ),
        };
        match solve_property_z3(&prop, 5000) {
            TierBResult::Disproved(_) => {}
            other => panic!(
                "0.1+0.2==0.3 over EXACT f64 literals must be Disproved (it is false \
                 at f64); a decimal lowering would wrongly prove it. Got {other:?}"
            ),
        }
    }

    #[test]
    fn exact_f64_literal_lowering_proves_a_dyadic_true_goal() {
        // 0.5_f64 + 0.25_f64 == 0.75 is EXACT in f64 (all dyadic), so it holds
        // at runtime AND over the exact rationals -- Z3 proves it. Negative
        // parity to the decimal-false case above: the exact lowering does not
        // make every float equality false, only the rounding-divergent ones.
        let prop = SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: cmp(
                CmpOp::Eq,
                SmtExpr::Arith(
                    ArithOp::Add,
                    Box::new(SmtExpr::RealLit(0.5)),
                    Box::new(SmtExpr::RealLit(0.25)),
                ),
                SmtExpr::RealLit(0.75),
            ),
        };
        assert_eq!(solve_property_z3(&prop, 5000), TierBResult::Proved);
    }

    // --- the algebraic / transcendental capability split (honesty) ---

    #[test]
    fn transcendental_exp_is_honest_unsupported_not_unknown() {
        // Z3 has no real-transcendental kind, so exp(x) >= 0 lowers to a clean
        // Error (Unsupported), NOT a fake unknown and NEVER a proof. This is the
        // cvc5->Z3 capability split: cvc5 attempts QF_NRAT here; Z3 declines.
        let prop = real_var_prop(cmp(
            CmpOp::Ge,
            SmtExpr::Apply("exp".to_string(), vec![var("x")]),
            SmtExpr::RealLit(0.0),
        ));
        match solve_property_z3(&prop, 5000) {
            TierBResult::Error(reason) => {
                assert!(reason.contains("exp"), "names the intrinsic: {reason}");
                assert!(
                    reason.contains("transcendental"),
                    "frames the boundary: {reason}"
                );
                assert!(
                    reason.contains("not supported by the Z3 NRA engine"),
                    "states the capability boundary: {reason}"
                );
            }
            other => panic!("expected honest Unsupported Error for exp, got {other:?}"),
        }
    }

    #[test]
    fn every_transcendental_is_honest_unsupported() {
        // sqrt/exp/log/sin/cos all have no Z3 real kind. Each must be a clean
        // Error naming the function -- so adding one to the predicate grammar
        // cannot silently regress to a fake unknown.
        for name in chelis_pred::TRANSCENDENTAL_WHITELIST {
            let prop = real_var_prop(cmp(
                CmpOp::Ge,
                SmtExpr::Apply(name.to_string(), vec![var("x")]),
                SmtExpr::RealLit(0.0),
            ));
            match solve_property_z3(&prop, 5000) {
                TierBResult::Error(reason) => {
                    assert!(reason.contains(name), "names {name}: {reason}");
                }
                other => panic!("transcendental `{name}` must be a clean Error, got {other:?}"),
            }
        }
    }

    #[test]
    fn unknown_symbol_keeps_generic_unsupported_reason() {
        // A truly-unknown callee keeps the generic reason; the honest-
        // transcendental framing must not launder an arbitrary symbol.
        let prop = real_var_prop(cmp(
            CmpOp::Ge,
            SmtExpr::Apply("mystery_fn".to_string(), vec![var("x")]),
            SmtExpr::RealLit(0.0),
        ));
        match solve_property_z3(&prop, 5000) {
            TierBResult::Error(reason) => {
                assert!(reason.contains("mystery_fn"), "names the fn: {reason}");
                assert!(
                    reason.contains("unsupported function"),
                    "generic reason: {reason}"
                );
                assert!(
                    !reason.contains("transcendental"),
                    "must not mislabel unknown as transcendental: {reason}"
                );
            }
            other => panic!("expected generic Unsupported Error, got {other:?}"),
        }
    }

    #[test]
    fn wrong_arity_intrinsic_is_a_clean_error() {
        // abs takes 1 arg; abs(x, y) is a clean Error, never a panic/abort.
        let prop = real_var_prop(cmp(
            CmpOp::Ge,
            SmtExpr::Apply("abs".to_string(), vec![var("x"), SmtExpr::RealLit(1.0)]),
            SmtExpr::RealLit(0.0),
        ));
        match solve_property_z3(&prop, 5000) {
            TierBResult::Error(reason) => assert!(reason.contains("abs"), "{reason}"),
            other => panic!("expected Error for two-arg abs, got {other:?}"),
        }
    }

    #[test]
    fn deeply_nested_property_routes_to_error_not_overflow() {
        // A property deeper than MAX_SMT_EXPR_DEPTH routes to a clean Error
        // (the depth guard), never a stack overflow / process abort.
        let mut e = var("x");
        for _ in 0..(MAX_SMT_EXPR_DEPTH + 10) {
            e = SmtExpr::Arith(ArithOp::Add, Box::new(e), Box::new(SmtExpr::RealLit(1.0)));
        }
        let prop = real_var_prop(cmp(CmpOp::Ge, e, SmtExpr::RealLit(0.0)));
        match solve_property_z3(&prop, 5000) {
            TierBResult::Error(reason) => assert!(reason.contains("deeper than"), "{reason}"),
            other => panic!("expected depth Error, got {other:?}"),
        }
    }

    #[test]
    fn non_finite_real_literal_is_a_clean_error() {
        let prop = real_var_prop(cmp(CmpOp::Ge, var("x"), SmtExpr::RealLit(f64::INFINITY)));
        match solve_property_z3(&prop, 5000) {
            TierBResult::Error(reason) => assert!(reason.contains("non-finite"), "{reason}"),
            other => panic!("expected non-finite Error, got {other:?}"),
        }
    }

    // --- the Z3Engine DischargeEngine surface ---

    #[test]
    fn z3_engine_name_is_z3() {
        assert_eq!(Z3Engine::new().name(), "z3");
    }

    #[test]
    fn z3_engine_fitness_accepts_smt_rejects_box_range() {
        let engine = Z3Engine::new();
        let smt_goal = Goal::smt(real_var_prop(cmp(
            CmpOp::Ge,
            mul(var("x"), var("x")),
            SmtExpr::RealLit(0.0),
        )));
        assert!(engine.fitness(&smt_goal));

        let box_goal = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 0.0, 100.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 50.0,
            },
        )
        .expect("well-formed box goal");
        assert!(!engine.fitness(&box_goal));
    }

    #[test]
    fn z3_engine_proves_a_polynomial_goal_over_reals_not_exact() {
        // chelis#422: a Z3 proof is over the REALS, so the discharge is
        // SoundApproximate carrying RealArith -- the SAME badge a cvc5 proof
        // gets -- never the exact-machine badge.
        let engine = Z3Engine::new();
        let goal = Goal::smt(real_var_prop(cmp(
            CmpOp::Ge,
            mul(var("x"), var("x")),
            SmtExpr::RealLit(0.0),
        )));
        let discharge = engine.discharge(&goal, 5000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
        assert!(discharge.qualifier_set().contains(Qualifier::RealArith));
        assert!(
            !discharge.qualifier_set().contains(Qualifier::Exact),
            "an over-reals proof must not claim exact machine soundness"
        );
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("z3"),
            "chelis#496: canonical top-level attribution key is `engine`"
        );
    }

    #[test]
    fn z3_engine_disproves_a_false_goal_over_reals() {
        let engine = Z3Engine::new();
        let goal = Goal::smt(real_var_prop(cmp(
            CmpOp::Gt,
            var("x"),
            SmtExpr::RealLit(0.0),
        )));
        let discharge = engine.discharge(&goal, 5000);
        assert!(matches!(discharge.result(), TierBResult::Disproved(_)));
        assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
        assert!(discharge.qualifier_set().contains(Qualifier::RealArith));
    }

    #[test]
    fn z3_engine_refuses_box_range_goal_as_untrusted() {
        let engine = Z3Engine::new();
        let goal = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 0.0, 1.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 1.0,
            },
        )
        .expect("well-formed box goal");
        let discharge = engine.discharge(&goal, 5000);
        assert!(matches!(discharge.result(), TierBResult::Error(_)));
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
    }

    #[test]
    fn z3_engine_transcendental_goal_is_untrusted_not_a_proof() {
        // A transcendental SMT goal: Z3 lowers it to Error (Unsupported), which
        // the shared classify maps to Untrusted/empty -- a non-verdict the
        // dispatcher falls through, never a fabricated proof.
        let engine = Z3Engine::new();
        let goal = Goal::smt(real_var_prop(cmp(
            CmpOp::Ge,
            SmtExpr::Apply("exp".to_string(), vec![var("x")]),
            SmtExpr::RealLit(0.0),
        )));
        let discharge = engine.discharge(&goal, 5000);
        assert!(matches!(discharge.result(), TierBResult::Error(_)));
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
    }
}
