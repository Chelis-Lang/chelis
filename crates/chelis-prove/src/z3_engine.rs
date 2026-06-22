//! The Z3 discharge engine (WI-12, Wave 3).
//!
//! Z3 is a second SMT/NRA backend, a sibling to [`crate::discharge::Cvc5Engine`].
//! It plugs into the same WI-4 discharge seam: a [`Z3Engine`] is a
//! [`DischargeEngine`] that claims [`GoalShape::Smt`] goals, lowers the
//! [`SmtProperty`] to a Z3 formula, asserts the NEGATION of the postcondition
//! (under the preconditions), and reads the satisfiability result back:
//!
//! - `Unsat` -> the negation is unsatisfiable -> the property is `Proved`.
//! - `Sat`   -> a counterexample exists -> the property is `Disproved(model)`.
//! - `Unknown` -> the solver could not decide -> `Unknown` (never a proof).
//!
//! A `Proved`/`Disproved` Z3 result is an EXACT decision over the chosen logic,
//! so it carries [`Soundness::Exact`] + the [`Qualifier::Exact`] badge, exactly
//! like cvc5; everything else is [`Soundness::Untrusted`] with no qualifier. The
//! discharge is built through [`Discharge::new`], so the per-qualifier soundness
//! guard applies: a Z3 result can never wear a badge its soundness does not back.
//!
//! # Role split vs cvc5 (the master plan's WI-12)
//!
//! Both [`Cvc5Engine`](crate::discharge::Cvc5Engine) and [`Z3Engine`] fit
//! [`GoalShape::Smt`]. Registration ORDER decides priority (see
//! [`crate::engine_registry`]): the first fitting engine in registration order
//! wins. The master plan positions Z3 as the polynomial-NRA backend, where cvc5
//! is weaker, so [`crate::engine_registry::DischargeRegistry::with_builtin_engines`]
//! registers Z3 AHEAD of cvc5 when BOTH feature lanes are active, making the
//! split explicit rather than letting Z3 silently shadow cvc5. When only one of
//! the two solver lanes is built, that lane's engine is the sole SMT engine.
//!
//! # Polynomial NRA, not transcendental NRAT
//!
//! The high-level `z3` crate (0.20) exposes the polynomial real-arithmetic
//! surface (`+ - * / neg`, comparisons, the boolean connectives, quantifiers)
//! but NOT the transcendental kinds (`exp`/`sqrt`/`sin`/`cos`) that cvc5-rs
//! exposes. Rather than fabricate an unsound encoding, a property whose lowering
//! reaches a transcendental (or any other unsupported) intrinsic returns a clean
//! [`TierBResult::Error`] -> [`Soundness::Untrusted`], routed exactly like the
//! cvc5 path routes a term it cannot build. This is sound: Z3 NEVER greens a
//! goal it cannot encode; cvc5 retains the transcendental NRAT lane.
//!
//! # Thread-local context safety
//!
//! The `z3` crate (0.20) builds every AST in a PER-THREAD default
//! [`z3::Context`]. [`Z3Engine::discharge`] performs the whole lowering + solve
//! synchronously on its calling thread, so all terms share that one thread's
//! context and nothing crosses a thread boundary. Concurrent test threads each
//! get an independent context, so the engine is safe under a parallel test
//! runner.

use std::collections::HashMap;

use crate::discharge::{
    Discharge, DischargeEngine, Goal, GoalShape, Qualifier, QualifierSet, Soundness,
};
use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr, SmtSort};
use crate::tier_b::{SmtProperty, TierBResult};

/// The Z3 discharge engine. Mirrors [`crate::discharge::Cvc5Engine`]'s shape:
/// it claims structured-SMT goals and keeps its `SmtProperty -> z3 AST -> Solver`
/// lowering self-contained, so the z3-specific term construction stays internal.
#[derive(Debug, Clone, Copy, Default)]
pub struct Z3Engine;

impl Z3Engine {
    pub const fn new() -> Self {
        Self
    }

    /// Stable engine identifier, also used in evidence and as the
    /// registration-priority diagnostic name.
    pub const ENGINE_NAME: &'static str = "z3";

    /// Map a tier-B outcome to its `(soundness, qualifier_set)`. Identical
    /// classification to cvc5: a proved/disproved Z3 result is an EXACT decision
    /// over the chosen logic; a timeout, unknown, or lowering error is untrusted
    /// (never a proof) and carries no qualifier. This shared classification is
    /// what makes cross-engine agreement meaningful: the two engines attach the
    /// same `(soundness, qualifier)` to the same outcome.
    fn classify(result: &TierBResult) -> (Soundness, QualifierSet) {
        match result {
            TierBResult::Proved | TierBResult::Disproved(_) => (
                Soundness::Exact,
                QualifierSet::from_iter_kinds([Qualifier::Exact]),
            ),
            TierBResult::Timeout | TierBResult::Unknown | TierBResult::Error(_) => {
                (Soundness::Untrusted, QualifierSet::new())
            }
        }
    }
}

impl DischargeEngine for Z3Engine {
    fn name(&self) -> &'static str {
        Self::ENGINE_NAME
    }

    fn fitness(&self, goal: &Goal) -> bool {
        // Z3 discharges structured-SMT goals, like cvc5. The WI-5 box/output
        // range shape is Beacon's native form; Z3 does not own it.
        matches!(goal.shape, GoalShape::Smt(_))
    }

    fn discharge(&self, goal: &Goal, timeout_ms: u64) -> Discharge {
        let result = match goal.as_smt() {
            Some(property) => solve_property_z3(property, timeout_ms),
            None => {
                TierBResult::Error("z3 engine cannot discharge a non-SMT goal shape".to_string())
            }
        };
        let (soundness, qualifier_set) = Self::classify(&result);
        let evidence = serde_json::json!({ "solver": Self::ENGINE_NAME });
        // The classification only ever pairs the `exact` qualifier with
        // `Soundness::Exact`, so this cannot fail here; surfacing the error as a
        // non-proof discharge keeps the seam total (mirrors the cvc5 path).
        Discharge::new(soundness, qualifier_set, result, evidence).unwrap_or_else(|err| {
            Discharge::new(
                Soundness::Untrusted,
                QualifierSet::new(),
                TierBResult::Error(err.to_string()),
                serde_json::json!({ "solver": Self::ENGINE_NAME, "internal_error": err.to_string() }),
            )
            .expect("untrusted discharge with empty qualifier set is always valid")
        })
    }
}

/// A lowered Z3 term, tagged with the [`SmtSort`] it carries. The sort is
/// tracked alongside the term so a parent node can reject a sort-mismatched
/// child (e.g. a numeric comparison over a Bool, or differing operand sorts)
/// BEFORE it builds a malformed Z3 term, exactly as the cvc5 lowering does.
#[derive(Clone)]
enum Z3Term {
    Real(z3::ast::Real),
    Int(z3::ast::Int),
    Bool(z3::ast::Bool),
}

impl Z3Term {
    fn sort(&self) -> SmtSort {
        match self {
            Z3Term::Real(_) => SmtSort::Real,
            Z3Term::Int(_) => SmtSort::Int,
            Z3Term::Bool(_) => SmtSort::Bool,
        }
    }

    fn as_bool(&self) -> Result<&z3::ast::Bool, String> {
        match self {
            Z3Term::Bool(b) => Ok(b),
            other => Err(format!(
                "expected a Bool-sorted term, got {:?} (routes to Tier C)",
                other.sort()
            )),
        }
    }
}

/// Solve a structured SMT property with Z3 (in-process). Mirrors
/// [`crate::tier_b::solve_property_cvc5`]: choose the logic from the property's
/// content, declare the variables, assert the preconditions, assert the negated
/// postcondition, and read the satisfiability result back.
///
/// Unlike the cvc5 path, a malformed Z3 term does not abort the process: the Z3
/// C API reports errors through its context error handler and the `z3` crate
/// surfaces them as Rust-level results, so this lowering does not need cvc5's
/// abort-avoidance contract. It still validates sorts so the formula it builds
/// has the intended meaning (a sort-confused term would otherwise silently
/// change what is being proved), routing any mismatch to a clean
/// [`TierBResult::Error`].
///
/// The ONE process-abort vector the z3 lowering DOES share with the cvc5 path is
/// the depth of the recursive [`lower_to_z3`] walk: a pathologically deep
/// `SmtExpr` overflows the Rust call stack (which aborts the process, it does
/// not return). So the property's nesting depth is screened FIRST, with an
/// iterative check that cannot itself overflow, exactly as
/// [`crate::tier_b::solve_property_cvc5`] does -- a tree past the bound routes to
/// a clean [`TierBResult::Error`] instead of recursing.
fn solve_property_z3(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
    use z3::SatResult;

    if property_exceeds_z3_depth(property) {
        return TierBResult::Error(format!(
            "property nests deeper than {MAX_Z3_EXPR_DEPTH} levels (routes to Tier C)"
        ));
    }

    let solver = z3::Solver::new();

    // Timeout: Z3's `timeout` parameter is the per-check wall-clock budget in
    // milliseconds. Z3 returns `Unknown` (not a panic) when it expires, which
    // classifies as Untrusted -- never a proof.
    let mut params = z3::Params::new();
    let clamped = u32::try_from(timeout_ms).unwrap_or(u32::MAX);
    params.set_u32("timeout", clamped);
    solver.set_params(&params);

    // Declare variables. `sorts` mirrors `vars` so the lowering knows each
    // variable's sort without re-deriving it, and an undeclared variable is a
    // clean Err rather than a confusing term.
    let mut vars: HashMap<String, Z3Term> = HashMap::new();
    let mut sorts: HashMap<String, SmtSort> = HashMap::new();
    for (name, sort) in &property.variables {
        let term = match sort {
            SmtSort::Real => Z3Term::Real(z3::ast::Real::new_const(name.as_str())),
            SmtSort::Int => Z3Term::Int(z3::ast::Int::new_const(name.as_str())),
            SmtSort::Bool => Z3Term::Bool(z3::ast::Bool::new_const(name.as_str())),
        };
        vars.insert(name.clone(), term);
        sorts.insert(name.clone(), *sort);
    }

    // Assert preconditions. Each must lower to a Bool; a non-Bool precondition
    // routes to Tier C rather than asserting a nonsensical formula.
    for pre in &property.preconditions {
        match lower_to_z3(pre, &vars, &sorts) {
            Ok(term) => match term.as_bool() {
                Ok(b) => solver.assert(b),
                Err(reason) => {
                    return TierBResult::Error(format!("precondition: {reason}"));
                }
            },
            Err(reason) => return TierBResult::Error(reason),
        }
    }

    // Assert the negation of the postcondition. The postcondition must lower to
    // a Bool; we prove the property by checking that its negation is Unsat.
    let post = match lower_to_z3(&property.postcondition, &vars, &sorts) {
        Ok(term) => match term.as_bool() {
            Ok(b) => b.clone(),
            Err(reason) => {
                return TierBResult::Error(format!("postcondition: {reason}"));
            }
        },
        Err(reason) => return TierBResult::Error(reason),
    };
    solver.assert(post.not());

    match solver.check() {
        SatResult::Unsat => TierBResult::Proved,
        SatResult::Sat => {
            // Extract a counterexample model, mirroring the cvc5 path's
            // `Disproved(Object{name: String(value)})` shape so the two engines'
            // disproof evidence is structurally comparable.
            let bindings = match solver.get_model() {
                Some(model) => model_bindings(&model, &vars),
                None => serde_json::Map::new(),
            };
            TierBResult::Disproved(serde_json::Value::Object(bindings))
        }
        SatResult::Unknown => TierBResult::Unknown,
    }
}

/// Read each declared variable's value out of a SAT model as a string, mirroring
/// the cvc5 path's `Object{name -> String(value)}` counterexample shape. A
/// variable the model does not interpret is omitted (the same effect the cvc5
/// path's value query has when the model is silent on a name).
fn model_bindings(
    model: &z3::Model,
    vars: &HashMap<String, Z3Term>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut bindings = serde_json::Map::new();
    for (name, term) in vars {
        let rendered = match term {
            Z3Term::Real(r) => model.get_const_interp(r).map(|v| v.to_string()),
            Z3Term::Int(i) => model.get_const_interp(i).map(|v| v.to_string()),
            Z3Term::Bool(b) => model.get_const_interp(b).map(|v| v.to_string()),
        };
        if let Some(value) = rendered {
            bindings.insert(name.clone(), serde_json::Value::String(value));
        }
    }
    bindings
}

/// Require a numeric (Real or Int) sort, else a clean Err.
fn require_numeric_sort(sort: SmtSort, ctx: &str) -> Result<(), String> {
    match sort {
        SmtSort::Real | SmtSort::Int => Ok(()),
        SmtSort::Bool => Err(format!(
            "{ctx} requires a numeric sort, got Bool (routes to Tier C)"
        )),
    }
}

/// Maximum `SmtExpr` nesting depth the z3 lowering will recurse into. A property
/// deeper than this routes to Tier C rather than risking a stack overflow in the
/// recursive [`lower_to_z3`] walk (a stack overflow aborts the PROCESS, it does
/// not return). This mirrors `crate::tier_b`'s `MAX_SMT_EXPR_DEPTH`; it is
/// re-declared here because that constant is `#[cfg(feature = "smt")]`-gated and
/// so is absent from a z3-only build.
const MAX_Z3_EXPR_DEPTH: usize = 256;

/// Whether any part of a property (the postcondition or a precondition) nests
/// past [`MAX_Z3_EXPR_DEPTH`], computed ITERATIVELY (an explicit work stack) so
/// the bound check itself cannot overflow the call stack on a pathologically
/// deep tree -- which is the whole point of running it BEFORE the recursive
/// [`lower_to_z3`] walk.
fn property_exceeds_z3_depth(property: &SmtProperty) -> bool {
    smt_expr_exceeds_depth(&property.postcondition, MAX_Z3_EXPR_DEPTH)
        || property
            .preconditions
            .iter()
            .any(|pre| smt_expr_exceeds_depth(pre, MAX_Z3_EXPR_DEPTH))
}

/// Whether `expr` nests deeper than `max`, with an explicit work stack so the
/// check cannot itself overflow.
fn smt_expr_exceeds_depth(expr: &SmtExpr, max: usize) -> bool {
    let mut stack: Vec<(&SmtExpr, usize)> = vec![(expr, 1)];
    while let Some((node, depth)) = stack.pop() {
        if depth > max {
            return true;
        }
        let next = depth + 1;
        match node {
            SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {}
            SmtExpr::Not(inner) => stack.push((inner, next)),
            SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => stack.push((body, next)),
            SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
                stack.push((l, next));
                stack.push((r, next));
            }
            SmtExpr::Bool(_, children) | SmtExpr::Apply(_, children) => {
                for c in children {
                    stack.push((c, next));
                }
            }
            SmtExpr::Ite(c, t, e) => {
                stack.push((c, next));
                stack.push((t, next));
                stack.push((e, next));
            }
        }
    }
    false
}

/// Lower an [`SmtExpr`] to a [`Z3Term`]. Recursive on `SmtExpr` depth; the caller
/// ([`solve_property_z3`]) screens the depth with [`property_exceeds_z3_depth`]
/// FIRST so this walk cannot overflow the stack. The recursion mirrors the cvc5
/// lowering's structure so the two engines stay in lockstep on supported forms.
fn lower_to_z3(
    expr: &SmtExpr,
    vars: &HashMap<String, Z3Term>,
    sorts: &HashMap<String, SmtSort>,
) -> Result<Z3Term, String> {
    match expr {
        SmtExpr::Var(name) => vars
            .get(name)
            .cloned()
            .ok_or_else(|| format!("variable `{name}` has no declared z3 term")),
        SmtExpr::RealLit(value) => {
            if !value.is_finite() {
                return Err(format!(
                    "non-finite real literal `{value}` cannot lower to z3 (routes to Tier C)"
                ));
            }
            // Build the rational exactly from the f64 so no decimal rounding is
            // introduced: an f64 is a dyadic rational, so numerator/2^k is exact.
            Ok(Z3Term::Real(real_from_f64(*value)))
        }
        SmtExpr::IntLit(value) => Ok(Z3Term::Int(z3::ast::Int::from_i64(*value))),
        SmtExpr::BoolLit(value) => Ok(Z3Term::Bool(z3::ast::Bool::from_bool(*value))),
        // Unary `neg` is represented as `Arith(Neg, x, <placeholder>)`; lower
        // only the real operand (mirrors the cvc5 path).
        SmtExpr::Arith(ArithOp::Neg, left, _placeholder) => {
            let l = lower_to_z3(left, vars, sorts)?;
            require_numeric_sort(l.sort(), "neg operand")?;
            match l {
                Z3Term::Real(r) => Ok(Z3Term::Real(r.unary_minus())),
                Z3Term::Int(i) => Ok(Z3Term::Int(i.unary_minus())),
                Z3Term::Bool(_) => unreachable!("numeric sort checked above"),
            }
        }
        SmtExpr::Arith(op, left, right) => {
            let l = lower_to_z3(left, vars, sorts)?;
            let r = lower_to_z3(right, vars, sorts)?;
            require_numeric_sort(l.sort(), "arithmetic operand")?;
            require_numeric_sort(r.sort(), "arithmetic operand")?;
            if l.sort() != r.sort() {
                return Err(format!(
                    "arithmetic operands have differing sorts {:?} vs {:?} (routes to Tier C)",
                    l.sort(),
                    r.sort()
                ));
            }
            lower_arith(*op, &l, &r)
        }
        SmtExpr::Cmp(op, left, right) => {
            let l = lower_to_z3(left, vars, sorts)?;
            let r = lower_to_z3(right, vars, sorts)?;
            if l.sort() != r.sort() {
                return Err(format!(
                    "comparison operands have differing sorts {:?} vs {:?} (routes to Tier C)",
                    l.sort(),
                    r.sort()
                ));
            }
            let is_numeric_cmp = matches!(op, CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge);
            if is_numeric_cmp && l.sort() == SmtSort::Bool {
                return Err("numeric comparison over a Bool operand (routes to Tier C)".to_string());
            }
            lower_cmp(*op, &l, &r)
        }
        SmtExpr::Bool(op, children) => {
            let lowered: Vec<Z3Term> = children
                .iter()
                .map(|c| lower_to_z3(c, vars, sorts))
                .collect::<Result<Vec<_>, _>>()?;
            let mut bools: Vec<z3::ast::Bool> = Vec::with_capacity(lowered.len());
            for child in &lowered {
                let b = child
                    .as_bool()
                    .map_err(|reason| format!("boolean connective operand: {reason}"))?;
                bools.push(b.clone());
            }
            lower_bool_connective(*op, &bools)
        }
        SmtExpr::Not(inner) => {
            let t = lower_to_z3(inner, vars, sorts)?;
            let b = t
                .as_bool()
                .map_err(|reason| format!("`not` operand: {reason}"))?;
            Ok(Z3Term::Bool(b.not()))
        }
        SmtExpr::Forall(bindings, body) | SmtExpr::Exists(bindings, body) => {
            // A quantifier over no variables is degenerate -- `forall (). P` and
            // `exists (). P` both mean `P` -- so normalize to the body, matching
            // the cvc5 path which avoids an empty binder list.
            let mut extended_vars = vars.clone();
            let mut extended_sorts = sorts.clone();
            let mut bound_terms: Vec<Z3Term> = Vec::with_capacity(bindings.len());
            for (name, sort) in bindings {
                let term = match sort {
                    SmtSort::Real => Z3Term::Real(z3::ast::Real::fresh_const(name)),
                    SmtSort::Int => Z3Term::Int(z3::ast::Int::fresh_const(name)),
                    SmtSort::Bool => Z3Term::Bool(z3::ast::Bool::fresh_const(name)),
                };
                extended_vars.insert(name.clone(), term.clone());
                extended_sorts.insert(name.clone(), *sort);
                bound_terms.push(term);
            }
            let body_term = lower_to_z3(body, &extended_vars, &extended_sorts)?;
            let body_bool = body_term
                .as_bool()
                .map_err(|reason| format!("quantifier body: {reason}"))?
                .clone();
            if bound_terms.is_empty() {
                return Ok(Z3Term::Bool(body_bool));
            }
            let bounds: Vec<&dyn z3::ast::Ast> = bound_terms.iter().map(z3_ast_ref).collect();
            let quantified = if matches!(expr, SmtExpr::Forall(_, _)) {
                z3::ast::forall_const(&bounds, &[], &body_bool)
            } else {
                z3::ast::exists_const(&bounds, &[], &body_bool)
            };
            Ok(Z3Term::Bool(quantified))
        }
        SmtExpr::Apply(name, args) => lower_apply(name, args, vars, sorts),
        SmtExpr::Ite(cond, then_expr, else_expr) => {
            let c = lower_to_z3(cond, vars, sorts)?;
            let cond_bool = c
                .as_bool()
                .map_err(|reason| format!("ite condition: {reason}"))?;
            let t = lower_to_z3(then_expr, vars, sorts)?;
            let e = lower_to_z3(else_expr, vars, sorts)?;
            if t.sort() != e.sort() {
                return Err(format!(
                    "if-then-else branches have differing sorts {:?} vs {:?} (routes to Tier C)",
                    t.sort(),
                    e.sort()
                ));
            }
            match (t, e) {
                (Z3Term::Real(tr), Z3Term::Real(er)) => Ok(Z3Term::Real(cond_bool.ite(&tr, &er))),
                (Z3Term::Int(ti), Z3Term::Int(ei)) => Ok(Z3Term::Int(cond_bool.ite(&ti, &ei))),
                (Z3Term::Bool(tb), Z3Term::Bool(eb)) => Ok(Z3Term::Bool(cond_bool.ite(&tb, &eb))),
                _ => unreachable!("branch sorts checked equal above"),
            }
        }
    }
}

/// Build an `f64` as an exact Z3 [`z3::ast::Real`]. An f64 is a dyadic rational
/// `mantissa * 2^exp`, so it converts to a numerator/denominator pair with no
/// decimal rounding -- the value Z3 reasons over is byte-for-byte the literal,
/// not a re-parsed decimal approximation.
fn real_from_f64(value: f64) -> z3::ast::Real {
    debug_assert!(value.is_finite(), "caller rejects non-finite literals");
    let ratio =
        num_rational::BigRational::from_float(value).expect("a finite f64 is a dyadic rational");
    z3::ast::Real::from_big_rational(&ratio)
}

fn lower_arith(op: ArithOp, l: &Z3Term, r: &Z3Term) -> Result<Z3Term, String> {
    match (l, r) {
        (Z3Term::Real(a), Z3Term::Real(b)) => Ok(Z3Term::Real(match op {
            ArithOp::Add => z3::ast::Real::add(&[a.clone(), b.clone()]),
            ArithOp::Sub => z3::ast::Real::sub(&[a.clone(), b.clone()]),
            ArithOp::Mul => z3::ast::Real::mul(&[a.clone(), b.clone()]),
            ArithOp::Div => a.div(b),
            ArithOp::Neg => unreachable!("Neg handled before lower_arith"),
        })),
        (Z3Term::Int(a), Z3Term::Int(b)) => match op {
            // Integer `/` is Z3's `Z3_mk_div` (Int division), whose result is
            // Int. cvc5's `/` is REAL division and promotes Int operands to
            // Real; to keep the two engines computing the SAME function, lower
            // Int `/` to REAL division here too (promote both operands to Real),
            // so the result sort is Real -- matching the cvc5 path exactly.
            ArithOp::Div => {
                let ar = a.to_real();
                let br = b.to_real();
                Ok(Z3Term::Real(ar.div(&br)))
            }
            ArithOp::Add => Ok(Z3Term::Int(z3::ast::Int::add(&[a.clone(), b.clone()]))),
            ArithOp::Sub => Ok(Z3Term::Int(z3::ast::Int::sub(&[a.clone(), b.clone()]))),
            ArithOp::Mul => Ok(Z3Term::Int(z3::ast::Int::mul(&[a.clone(), b.clone()]))),
            ArithOp::Neg => unreachable!("Neg handled before lower_arith"),
        },
        _ => Err(
            "arithmetic operands have non-numeric or mixed sorts (routes to Tier C)".to_string(),
        ),
    }
}

fn lower_cmp(op: CmpOp, l: &Z3Term, r: &Z3Term) -> Result<Z3Term, String> {
    // Eq/Ne admit any equal sort (including Bool == Bool); the numeric
    // comparisons require numeric operands (already gated by the caller).
    let bool_term = match (l, r) {
        (Z3Term::Real(a), Z3Term::Real(b)) => cmp_numeric_real(op, a, b),
        (Z3Term::Int(a), Z3Term::Int(b)) => cmp_numeric_int(op, a, b),
        (Z3Term::Bool(a), Z3Term::Bool(b)) => match op {
            CmpOp::Eq => a.iff(b),
            CmpOp::Ne => a.iff(b).not(),
            _ => {
                return Err("numeric comparison over Bool operands (routes to Tier C)".to_string());
            }
        },
        _ => {
            return Err("comparison operands have mixed sorts (routes to Tier C)".to_string());
        }
    };
    Ok(Z3Term::Bool(bool_term))
}

fn cmp_numeric_real(op: CmpOp, a: &z3::ast::Real, b: &z3::ast::Real) -> z3::ast::Bool {
    match op {
        CmpOp::Lt => a.lt(b),
        CmpOp::Le => a.le(b),
        CmpOp::Gt => a.gt(b),
        CmpOp::Ge => a.ge(b),
        CmpOp::Eq => a.eq(b),
        CmpOp::Ne => a.eq(b).not(),
    }
}

fn cmp_numeric_int(op: CmpOp, a: &z3::ast::Int, b: &z3::ast::Int) -> z3::ast::Bool {
    match op {
        CmpOp::Lt => a.lt(b),
        CmpOp::Le => a.le(b),
        CmpOp::Gt => a.gt(b),
        CmpOp::Ge => a.ge(b),
        CmpOp::Eq => a.eq(b),
        CmpOp::Ne => a.eq(b).not(),
    }
}

fn lower_bool_connective(op: BoolOp, bools: &[z3::ast::Bool]) -> Result<Z3Term, String> {
    let term = match op {
        // Z3's `and`/`or` accept any arity (including 0 and 1), so unlike the
        // cvc5 path no degenerate-arity normalization is needed: an empty `and`
        // is `true`, an empty `or` is `false`, a singleton is itself -- which is
        // exactly the logical meaning the cvc5 path hand-normalizes to.
        BoolOp::And => z3::ast::Bool::and(bools),
        BoolOp::Or => z3::ast::Bool::or(bools),
        BoolOp::Implies => {
            if bools.len() != 2 {
                return Err(format!(
                    "`implies` requires exactly 2 operands, got {} (routes to Tier C)",
                    bools.len()
                ));
            }
            bools[0].implies(&bools[1])
        }
    };
    Ok(Z3Term::Bool(term))
}

/// Lower an intrinsic application. The high-level `z3` crate (0.20) does NOT
/// expose the transcendental kinds (`exp`/`sqrt`/`sin`/`cos`), so those route to
/// a clean Error: Z3 is the POLYNOMIAL-NRA backend (the master plan's role),
/// while cvc5 keeps the transcendental NRAT lane. `abs`, `min`, and `max` are
/// algebraic and ARE supported -- lowered to the same `ite`/comparison encoding
/// the cvc5 path uses for `min`/`max`, so the two engines decide them
/// identically.
fn lower_apply(
    name: &str,
    args: &[SmtExpr],
    vars: &HashMap<String, Z3Term>,
    sorts: &HashMap<String, SmtSort>,
) -> Result<Z3Term, String> {
    let lowered: Vec<Z3Term> = args
        .iter()
        .map(|a| lower_to_z3(a, vars, sorts))
        .collect::<Result<Vec<_>, _>>()?;
    match name {
        "exp" | "sqrt" | "sin" | "cos" => Err(format!(
            "transcendental intrinsic `{name}` is unsupported by the z3 engine \
             (polynomial NRA only; routes to Tier C / cvc5 NRAT lane)"
        )),
        "abs" => {
            if lowered.len() != 1 {
                return Err(format!(
                    "intrinsic `abs` expects 1 argument, got {} (routes to Tier C)",
                    lowered.len()
                ));
            }
            require_numeric_sort(lowered[0].sort(), "abs operand")?;
            // abs(x) = ite(x < 0, -x, x), built per numeric sort.
            match &lowered[0] {
                Z3Term::Real(x) => {
                    let zero = z3::ast::Real::from_rational(0, 1);
                    let neg = x.unary_minus();
                    Ok(Z3Term::Real(x.lt(&zero).ite(&neg, x)))
                }
                Z3Term::Int(x) => {
                    let zero = z3::ast::Int::from_i64(0);
                    let neg = x.unary_minus();
                    Ok(Z3Term::Int(x.lt(&zero).ite(&neg, x)))
                }
                Z3Term::Bool(_) => unreachable!("numeric sort checked above"),
            }
        }
        "min" | "max" => {
            if lowered.len() != 2 {
                return Err(format!(
                    "intrinsic `{name}` expects 2 arguments, got {} (routes to Tier C)",
                    lowered.len()
                ));
            }
            require_numeric_sort(lowered[0].sort(), "min/max operand")?;
            require_numeric_sort(lowered[1].sort(), "min/max operand")?;
            if lowered[0].sort() != lowered[1].sort() {
                return Err(format!(
                    "`{name}` operands have differing sorts {:?} vs {:?} (routes to Tier C)",
                    lowered[0].sort(),
                    lowered[1].sort()
                ));
            }
            // min(a,b) = ite(a < b, a, b); max(a,b) = ite(a > b, a, b) -- the
            // same encoding the cvc5 path uses.
            let pick_first_when_lt = name == "min";
            match (&lowered[0], &lowered[1]) {
                (Z3Term::Real(a), Z3Term::Real(b)) => {
                    let cond = if pick_first_when_lt { a.lt(b) } else { a.gt(b) };
                    Ok(Z3Term::Real(cond.ite(a, b)))
                }
                (Z3Term::Int(a), Z3Term::Int(b)) => {
                    let cond = if pick_first_when_lt { a.lt(b) } else { a.gt(b) };
                    Ok(Z3Term::Int(cond.ite(a, b)))
                }
                _ => unreachable!("operand sorts checked equal and numeric above"),
            }
        }
        other => Err(format!(
            "unsupported function `{other}` in z3 lowering (routes to Tier C)"
        )),
    }
}

/// Borrow a [`Z3Term`] as a `&dyn z3::ast::Ast` for the quantifier bound list.
fn z3_ast_ref(term: &Z3Term) -> &dyn z3::ast::Ast {
    match term {
        Z3Term::Real(r) => r,
        Z3Term::Int(i) => i,
        Z3Term::Bool(b) => b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discharge::{IntervalBox, OutputRange};

    // --- property builders shared with the cvc5 engine tests ---

    fn trivially_true_property() -> SmtProperty {
        // forall x: Real . x == x -- proved by asserting the negation UNSAT.
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Eq,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::Var("x".to_string())),
            ),
        }
    }

    fn false_property() -> SmtProperty {
        // postcondition 1.0 < 0.0 is false: Z3 disproves it.
        SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Lt,
                Box::new(SmtExpr::RealLit(1.0)),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        }
    }

    /// A polynomial-NRA property: for all real x, x*x >= 0. This is the kind of
    /// nonlinear real-arithmetic goal the master plan positions Z3 for.
    fn nonneg_square_property() -> SmtProperty {
        // forall x: Real . x * x >= 0
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::Var("x".to_string())),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        }
    }

    /// A FALSE polynomial property: for all real x, x*x >= 1 (false at x = 0).
    fn false_square_property() -> SmtProperty {
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::Var("x".to_string())),
                )),
                Box::new(SmtExpr::RealLit(1.0)),
            ),
        }
    }

    fn smt_goal(property: SmtProperty) -> Goal {
        Goal::smt(property)
    }

    fn box_range_goal() -> Goal {
        Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 0.0, 100.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 50.0,
            },
        )
        .expect("well-formed box goal")
    }

    // --- fitness: Z3 claims Smt goals, refuses BoxRange ---

    #[test]
    fn z3_engine_fitness_accepts_smt_goal_rejects_box_range() {
        let engine = Z3Engine::new();
        assert!(engine.fitness(&smt_goal(trivially_true_property())));
        assert!(!engine.fitness(&box_range_goal()));
    }

    #[test]
    fn z3_engine_name_is_stable() {
        assert_eq!(Z3Engine::new().name(), "z3");
    }

    // --- happy path: trivially-true proves Exact ---

    #[test]
    fn z3_engine_proves_a_trivial_goal_as_exact() {
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&smt_goal(trivially_true_property()), 5_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::Exact);
        assert!(discharge.qualifier_set().contains(Qualifier::Exact));
    }

    #[test]
    fn z3_engine_proves_nonneg_square_as_exact() {
        // The NRA goal: forall x. x*x >= 0. Z3's NRA decision procedure proves
        // it exactly.
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&smt_goal(nonneg_square_property()), 5_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::Exact);
        assert!(discharge.qualifier_set().contains(Qualifier::Exact));
    }

    // --- negative twin: false properties disprove Exact ---

    #[test]
    fn z3_engine_disproves_a_false_goal_as_exact() {
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&smt_goal(false_property()), 5_000);
        assert!(matches!(discharge.result(), TierBResult::Disproved(_)));
        assert_eq!(discharge.soundness(), Soundness::Exact);
        assert!(discharge.qualifier_set().contains(Qualifier::Exact));
    }

    #[test]
    fn z3_engine_disproves_false_square_with_counterexample() {
        // forall x. x*x >= 1 is false (x = 0 is a counterexample). Z3 must
        // disprove it and surface a model binding for x.
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&smt_goal(false_square_property()), 5_000);
        match discharge.result() {
            TierBResult::Disproved(model) => {
                assert!(
                    model.get("x").is_some(),
                    "the counterexample must bind the free variable x, got {model:?}"
                );
            }
            other => panic!("expected Disproved with a model, got {other:?}"),
        }
        assert_eq!(discharge.soundness(), Soundness::Exact);
    }

    // --- Z3 refuses a BoxRange goal as untrusted (no fabricated proof) ---

    #[test]
    fn z3_engine_refuses_box_range_goal_as_untrusted() {
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&box_range_goal(), 5_000);
        assert!(matches!(discharge.result(), TierBResult::Error(_)));
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
    }

    // --- transcendentals route to a clean untrusted Error, never a proof ---

    #[test]
    fn z3_engine_routes_transcendental_to_untrusted_error() {
        // sqrt(x) >= 0 -- a true fact, but the z3 engine does not encode
        // transcendentals, so it must NOT fabricate a proof: it returns an
        // untrusted Error (cvc5's NRAT lane owns this).
        let property = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(
                    "sqrt".to_string(),
                    vec![SmtExpr::Var("x".to_string())],
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&smt_goal(property), 5_000);
        assert!(
            matches!(discharge.result(), TierBResult::Error(_)),
            "a transcendental must route to an Error, never a proof"
        );
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
    }

    // --- no laundering: a Z3 disproof cannot wear a stronger badge ---

    #[test]
    fn z3_disproof_carries_exactly_exact_no_extra_badge() {
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&smt_goal(false_property()), 5_000);
        // Exactly the exact badge: no SoundOverApproximation, no Fuzz, no Axiom.
        assert!(discharge.qualifier_set().contains(Qualifier::Exact));
        assert!(
            !discharge
                .qualifier_set()
                .contains(Qualifier::SoundOverApproximation)
        );
        assert!(!discharge.qualifier_set().contains(Qualifier::Fuzz));
        assert!(!discharge.qualifier_set().contains(Qualifier::Axiom));
    }

    // --- depth guard: a pathologically deep tree routes to Error, never aborts ---

    #[test]
    fn z3_engine_routes_an_over_deep_property_to_untrusted_error() {
        // Build a postcondition far deeper than MAX_Z3_EXPR_DEPTH. Without the
        // iterative depth screen, the recursive lowering would overflow the
        // stack and ABORT the process; with it, this is a clean untrusted Error.
        let mut expr = SmtExpr::Var("x".to_string());
        for _ in 0..(MAX_Z3_EXPR_DEPTH + 50) {
            expr = SmtExpr::Not(Box::new(expr));
        }
        // Wrap the (now Bool-ish via repeated Not over a Real var) deep tree in a
        // Bool postcondition shape; the depth screen fires before any lowering,
        // so the inner sort never matters.
        let property = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Bool)],
            preconditions: vec![],
            postcondition: expr,
        };
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&smt_goal(property), 5_000);
        assert!(
            matches!(discharge.result(), TierBResult::Error(_)),
            "an over-deep property must route to a clean Error, never a proof or an abort"
        );
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
    }

    #[test]
    fn z3_depth_screen_admits_a_shallow_property() {
        // Negative twin: a property well within the bound is NOT screened out --
        // it lowers and solves normally (here a true shallow goal proves).
        let property = trivially_true_property();
        assert!(
            !property_exceeds_z3_depth(&property),
            "a shallow property must pass the depth screen"
        );
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&smt_goal(property), 5_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
    }

    // --- a min/max algebraic intrinsic is supported (parity with cvc5) ---

    #[test]
    fn z3_engine_proves_max_dominates_each_argument() {
        // forall a b: Real . max(a, b) >= a
        let property = SmtProperty {
            variables: vec![
                ("a".to_string(), SmtSort::Real),
                ("b".to_string(), SmtSort::Real),
            ],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(
                    "max".to_string(),
                    vec![SmtExpr::Var("a".to_string()), SmtExpr::Var("b".to_string())],
                )),
                Box::new(SmtExpr::Var("a".to_string())),
            ),
        };
        let engine = Z3Engine::new();
        let discharge = engine.discharge(&smt_goal(property), 5_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::Exact);
    }
}
