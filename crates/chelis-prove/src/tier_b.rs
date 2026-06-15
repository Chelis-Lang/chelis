//! Tier B: SMT solving via cvc5.
//!
//! Lowers property predicates to cvc5 terms and solves.
//! Returns Proved (UNSAT), Disproved (SAT with model), or Inconclusive
//! (timeout/unknown).

use serde_json::Value;

use crate::solver::{SmtExpr, SmtSort};

#[cfg(feature = "smt")]
use crate::solver::{ArithOp, BoolOp, CmpOp};

/// Tier B outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum TierBResult {
    /// Property proved (negation is UNSAT).
    Proved,
    /// Property disproved with counterexample.
    Disproved(Value),
    /// Solver timed out.
    Timeout,
    /// Solver returned unknown.
    Unknown,
    /// The property could not be lowered to a valid SMT term (e.g. a
    /// wrong-arity intrinsic application). Carries a human-facing reason.
    /// Routed to the same not-provable handling as Timeout/Unknown, never
    /// allowed to reach cvc5 as an invalid term (which would abort the
    /// solver with empty stdout).
    Error(String),
}

/// Structured property input for SMT solving.
#[derive(Debug, Clone)]
pub struct SmtProperty {
    /// Variable names and their sorts.
    pub variables: Vec<(String, SmtSort)>,
    /// Preconditions (all must hold).
    pub preconditions: Vec<SmtExpr>,
    /// Postcondition to prove (we assert its negation).
    pub postcondition: SmtExpr,
}

/// Run Tier B SMT check on a property source string.
///
/// Without the `smt` feature, returns Timeout (forces Tier C fallback).
pub fn solve(_property_source: &str, _property_name: &str, _timeout_ms: u64) -> TierBResult {
    #[cfg(feature = "smt")]
    {
        // Without a structured SmtProperty, we can't solve from raw source.
        // Use solve_property() with a structured input instead.
        TierBResult::Timeout
    }
    #[cfg(not(feature = "smt"))]
    {
        TierBResult::Timeout
    }
}

/// Solve a structured SMT property.
///
/// This is the primary entry point for Tier B when the caller has already
/// parsed the property into SmtExpr form.
pub fn solve_property(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
    #[cfg(feature = "smt")]
    {
        solve_property_cvc5(property, timeout_ms)
    }
    #[cfg(not(feature = "smt"))]
    {
        let _ = (property, timeout_ms);
        TierBResult::Timeout
    }
}

/// The single source of truth for the intrinsics that [`lower_to_cvc5`]
/// can actually build a cvc5 term for, paired with their required arity.
///
/// CR2-1: three lists used to drift -- the arity guard, the lowering
/// match arms, and the transcendental-logic classifier. They are now all
/// derived from this constant (and [`CVC5_TRANSCENDENTAL`]) so a function
/// admitted by one cannot silently diverge from another.
///
/// `log` is deliberately ABSENT: cvc5 has no LOG kind, so `lower_to_cvc5`
/// cannot build a term for it. `log` is still a valid predicate intrinsic
/// (it is in [`chelis_pred::INTRINSIC_WHITELIST`]) and works at Tier C via
/// the concrete evaluator; the arity guard routes it to a clean
/// [`TierBResult::Error`] so dispatch falls through to Tier C rather than
/// reaching the lowering and panicking.
#[cfg(feature = "smt")]
const CVC5_LOWERABLE: &[(&str, usize)] = &[
    ("exp", 1),
    ("sqrt", 1),
    ("sin", 1),
    ("cos", 1),
    ("abs", 1),
    ("min", 2),
    ("max", 2),
];

/// The transcendental subset of [`CVC5_LOWERABLE`] whose presence selects
/// the `QF_NRAT` logic (vs `QF_NRA`). `abs`/`min`/`max` are algebraic and
/// stay in `QF_NRA`.
#[cfg(feature = "smt")]
const CVC5_TRANSCENDENTAL: &[&str] = &["exp", "sqrt", "sin", "cos"];

/// Required arity for a cvc5-lowerable intrinsic, or `None` if the name is
/// not in [`CVC5_LOWERABLE`].
#[cfg(feature = "smt")]
fn cvc5_lowerable_arity(name: &str) -> Option<usize> {
    CVC5_LOWERABLE
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, a)| *a)
}

/// The WHITELIST gate (review 5): Tier B lowers ONLY provably-cvc5-safe
/// terms. The earlier pre-checks BLACKLISTED known aborting shapes
/// (Int-vs-Real comparisons, mixed-sort min/max, ITE branches) and kept
/// missing siblings (transcendental-over-Int, Bool-vs-Int, quantifier
/// bound-var sort mismatch). The invariant is inverted: a property is
/// lowerable ONLY IF every node is provably safe; the DEFAULT is
/// NOT-lowerable, so any shape no review enumerated (a sort the tracker
/// cannot determine, an unrecognized op) routes to Tier C automatically,
/// never an abort.
///
/// Returns `Ok(())` if the whole property (every precondition + the
/// postcondition) is provably cvc5-lowerable, else `Err(reason)`.
#[cfg(feature = "smt")]
fn is_cvc5_lowerable(prop: &SmtProperty) -> Result<(), String> {
    let mut env: std::collections::HashMap<String, SmtSort> = prop
        .variables
        .iter()
        .map(|(n, s)| (n.clone(), *s))
        .collect();
    for pre in &prop.preconditions {
        require_bool(pre, &mut env)?;
    }
    require_bool(&prop.postcondition, &mut env)
}

/// The cvc5 sort a node lowers to, or `Err` if the node is not provably
/// lowerable (an unknown var sort, an unrecognized op, a sort mismatch, or
/// a function applied to an argument of the wrong cvc5 sort). The DEFAULT
/// is reject: every admitted shape is enumerated explicitly.
///
/// `env` carries the declared variable sorts, extended with quantifier
/// bound-var sorts as we descend into a `Forall`/`Exists` body.
#[cfg(feature = "smt")]
fn lowerable_sort(
    expr: &SmtExpr,
    env: &mut std::collections::HashMap<String, SmtSort>,
) -> Result<SmtSort, String> {
    match expr {
        SmtExpr::IntLit(_) => Ok(SmtSort::Int),
        SmtExpr::RealLit(_) => Ok(SmtSort::Real),
        SmtExpr::BoolLit(_) => Ok(SmtSort::Bool),
        SmtExpr::Var(name) => env
            .get(name)
            .copied()
            .ok_or_else(|| format!("variable `{name}` has no known sort (routes to Tier C)")),
        // A boolean-shaped node is Bool; verify its structure recursively.
        SmtExpr::Not(inner) => {
            require_bool(inner, env)?;
            Ok(SmtSort::Bool)
        }
        SmtExpr::Bool(_, children) => {
            for c in children {
                require_bool(c, env)?;
            }
            Ok(SmtSort::Bool)
        }
        SmtExpr::Cmp(op, l, r) => {
            let ls = lowerable_sort(l, env)?;
            let rs = lowerable_sort(r, env)?;
            // Both operands must lower to the SAME known sort. A numeric
            // comparison (Lt/Le/Gt/Ge) additionally requires a numeric sort
            // (a Bool operand aborts cvc5). Eq/Ne admit any equal sort
            // (including Bool == Bool).
            if ls != rs {
                return Err(format!(
                    "comparison operands have differing sorts {ls:?} vs {rs:?} (routes to Tier C)"
                ));
            }
            let is_numeric_cmp = matches!(op, CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge);
            if is_numeric_cmp && ls == SmtSort::Bool {
                return Err("numeric comparison over a Bool operand (routes to Tier C)".to_string());
            }
            Ok(SmtSort::Bool)
        }
        // Unary `neg` is represented as `Arith(Neg, x, <placeholder>)`; cvc5
        // lowers NEG as unary and ignores the placeholder, so check ONLY the
        // real operand and require a numeric sort.
        SmtExpr::Arith(ArithOp::Neg, l, _placeholder) => {
            let ls = lowerable_sort(l, env)?;
            require_numeric(ls, "neg operand")?;
            Ok(ls)
        }
        SmtExpr::Arith(_, l, r) => {
            let ls = lowerable_sort(l, env)?;
            let rs = lowerable_sort(r, env)?;
            require_numeric(ls, "arithmetic operand")?;
            require_numeric(rs, "arithmetic operand")?;
            if ls != rs {
                return Err(format!(
                    "arithmetic operands have differing sorts {ls:?} vs {rs:?} (routes to Tier C)"
                ));
            }
            Ok(ls)
        }
        SmtExpr::Ite(c, t, e) => {
            require_bool(c, env)?;
            let ts = lowerable_sort(t, env)?;
            let es = lowerable_sort(e, env)?;
            if ts != es {
                return Err(format!(
                    "if-then-else branches have differing sorts {ts:?} vs {es:?} (routes to Tier C)"
                ));
            }
            Ok(ts)
        }
        SmtExpr::Apply(name, args) => apply_lowerable_sort(name, args, env),
        SmtExpr::Forall(bindings, body) | SmtExpr::Exists(bindings, body) => {
            // Seed the bound vars into the env, then verify the body. Restore
            // any shadowed outer bindings afterward so a later sibling node
            // does not see the bound var. (This admits a consistent
            // quantifier and rejects a bound-var sort mismatch in the body.)
            let mut shadowed: Vec<(String, Option<SmtSort>)> = Vec::new();
            for (name, sort) in bindings {
                shadowed.push((name.clone(), env.insert(name.clone(), *sort)));
            }
            let result = require_bool(body, env);
            for (name, prev) in shadowed.into_iter().rev() {
                match prev {
                    Some(s) => {
                        env.insert(name, s);
                    }
                    None => {
                        env.remove(&name);
                    }
                }
            }
            result?;
            Ok(SmtSort::Bool)
        }
    }
}

/// Require a node to lower to `Bool` (a precondition / postcondition / the
/// operands of `and`/`or`/`not` / the condition of an `ite` / a quantifier
/// body must be boolean).
#[cfg(feature = "smt")]
fn require_bool(
    expr: &SmtExpr,
    env: &mut std::collections::HashMap<String, SmtSort>,
) -> Result<(), String> {
    match lowerable_sort(expr, env)? {
        SmtSort::Bool => Ok(()),
        other => Err(format!(
            "expected a boolean-sorted term, got {other:?} (routes to Tier C)"
        )),
    }
}

/// Require a sort to be numeric (`Int` or `Real`); a `Bool` in an
/// arithmetic position aborts cvc5.
#[cfg(feature = "smt")]
fn require_numeric(sort: SmtSort, what: &str) -> Result<(), String> {
    match sort {
        SmtSort::Int | SmtSort::Real => Ok(()),
        SmtSort::Bool => Err(format!("{what} is Bool, not numeric (routes to Tier C)")),
    }
}

/// The lowerable sort of an intrinsic application, enforcing each cvc5
/// function's argument-sort requirement. The DEFAULT is reject: a name not
/// in [`CVC5_LOWERABLE`], a wrong arity, or an argument of the wrong sort
/// makes the whole property not lowerable.
#[cfg(feature = "smt")]
fn apply_lowerable_sort(
    name: &str,
    args: &[SmtExpr],
    env: &mut std::collections::HashMap<String, SmtSort>,
) -> Result<SmtSort, String> {
    let arity = cvc5_lowerable_arity(name).ok_or_else(|| {
        format!("unsupported function `{name}` in cvc5 lowering (routes to Tier C)")
    })?;
    if args.len() != arity {
        return Err(format!(
            "intrinsic `{name}` expects {arity} argument(s), got {} (routes to Tier C)",
            args.len()
        ));
    }
    // Every argument must itself be lowerable.
    let arg_sorts: Vec<SmtSort> = args
        .iter()
        .map(|a| lowerable_sort(a, env))
        .collect::<Result<Vec<_>, _>>()?;
    match name {
        // cvc5 SQRT/EXP/SINE/COSINE require a REAL argument; an Int argument
        // aborts the process. (`log` is not in CVC5_LOWERABLE and is rejected
        // above, so it never reaches here.)
        "sqrt" | "exp" | "sin" | "cos" => {
            if arg_sorts[0] != SmtSort::Real {
                return Err(format!(
                    "`{name}` requires a Real argument, got {:?} (routes to Tier C)",
                    arg_sorts[0]
                ));
            }
            Ok(SmtSort::Real)
        }
        // `abs` preserves a numeric operand sort.
        "abs" => {
            require_numeric(arg_sorts[0], "abs operand")?;
            Ok(arg_sorts[0])
        }
        // `min`/`max` lower to ITE(cmp(a,b), a, b); both args must share one
        // known numeric sort (a mixed pair aborts the inner ITE/cmp).
        "min" | "max" => {
            require_numeric(arg_sorts[0], "min/max operand")?;
            require_numeric(arg_sorts[1], "min/max operand")?;
            if arg_sorts[0] != arg_sorts[1] {
                return Err(format!(
                    "`{name}` operands have differing sorts {:?} vs {:?} (routes to Tier C)",
                    arg_sorts[0], arg_sorts[1]
                ));
            }
            Ok(arg_sorts[0])
        }
        // Defense in depth: any other name (should be unreachable after the
        // CVC5_LOWERABLE arity check) is rejected.
        other => Err(format!(
            "unsupported function `{other}` in cvc5 lowering (routes to Tier C)"
        )),
    }
}

#[cfg(feature = "smt")]
fn solve_property_cvc5(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
    use cvc5_rs::{Kind, Solver, TermManager};
    use std::collections::HashMap;

    // The SOLE pre-lowering gate (review 5 WHITELIST): build a cvc5 term
    // ONLY for a property every node of which is PROVABLY cvc5-safe. cvc5's
    // `mk_term` ABORTS THE PROCESS on a sort-mismatched / wrong-sort term
    // (e.g. Int-vs-Real, a Bool in a numeric comparison, sqrt over an Int, a
    // quantifier bound-var sort clash), surfacing to a JSON consumer as an
    // empty-stdout bare exit -- a machine-contract violation. The whitelist's
    // default is NOT-lowerable, so any shape it cannot prove safe (including
    // ones no review enumerated) routes to a clean Tier C result here, BEFORE
    // any cvc5 term is built. `lower_to_cvc5` is therefore only ever called on
    // a provably-safe property; its Result-not-panic arms are defense in depth.
    if let Err(reason) = is_cvc5_lowerable(property) {
        return TierBResult::Error(reason);
    }

    let tm = TermManager::new();
    let mut solver = Solver::new(&tm);

    // Choose logic based on expression content. A transcendental call needs
    // the `T` (transcendental) extension; a quantifier needs a NON-`QF_`
    // (quantified) logic -- a `forall`/`exists` under a `QF_` logic aborts
    // cvc5 ("doesn't include THEORY_QUANTIFIERS"), which the whitelist admits
    // (the bound-var-seeded body is sort-consistent) so the LOGIC must
    // support it. The `QF_` prefix is dropped when a quantifier is present.
    let has_transcendentals = contains_transcendental(&property.postcondition)
        || property.preconditions.iter().any(contains_transcendental);
    let has_quantifier = contains_quantifier(&property.postcondition)
        || property.preconditions.iter().any(contains_quantifier);
    let base = if has_transcendentals { "NRAT" } else { "NRA" };
    let logic = if has_quantifier {
        base.to_string()
    } else {
        format!("QF_{base}")
    };
    solver.set_logic(&logic);
    solver.set_option("produce-models", "true");
    solver.set_option("tlimit-per", &timeout_ms.to_string());

    // 1. Declare variables
    let mut vars: HashMap<String, cvc5_rs::Term> = HashMap::new();
    for (name, sort) in &property.variables {
        let cvc5_sort = match sort {
            SmtSort::Real => tm.real_sort(),
            SmtSort::Int => tm.integer_sort(),
            SmtSort::Bool => tm.boolean_sort(),
        };
        let var = tm.mk_const(cvc5_sort, name);
        vars.insert(name.clone(), var);
    }

    // 2. Assert preconditions
    for pre in &property.preconditions {
        let term = match lower_to_cvc5(&tm, pre, &vars) {
            Ok(t) => t,
            Err(reason) => return TierBResult::Error(reason),
        };
        solver.assert_formula(term);
    }

    // 3. Assert negation of postcondition
    let post_term = match lower_to_cvc5(&tm, &property.postcondition, &vars) {
        Ok(t) => t,
        Err(reason) => return TierBResult::Error(reason),
    };
    let negated = tm.mk_term(Kind::CVC5_KIND_NOT, &[post_term]);
    solver.assert_formula(negated);

    // 4. Check satisfiability
    let result = solver.check_sat();
    if result.is_unsat() {
        // No counterexample exists → property proved
        TierBResult::Proved
    } else if result.is_sat() {
        // Counterexample found → extract model
        let mut bindings = serde_json::Map::new();
        for (name, var) in &vars {
            let val = solver.get_value(var.clone());
            bindings.insert(name.clone(), Value::String(val.to_string()));
        }
        TierBResult::Disproved(Value::Object(bindings))
    } else {
        // Unknown or timeout
        TierBResult::Unknown
    }
}

/// Build the cvc5 BOUND variables for a quantifier's binder list. cvc5's
/// `VARIABLE_LIST` requires `mk_var` bound variables, NOT `mk_const`
/// constants (a constant in the bound list aborts: "argument of bound var
/// list is not bound variable"). The body lowering substitutes these for
/// the bound names via the extended `vars` map.
#[cfg(feature = "smt")]
fn quantifier_bound_vars(
    tm: &cvc5_rs::TermManager,
    bindings: &[(String, SmtSort)],
) -> Vec<cvc5_rs::Term> {
    bindings
        .iter()
        .map(|(name, sort)| {
            let s = match sort {
                SmtSort::Real => tm.real_sort(),
                SmtSort::Int => tm.integer_sort(),
                SmtSort::Bool => tm.boolean_sort(),
            };
            tm.mk_var(s, name)
        })
        .collect()
}

/// Lower an SmtExpr to a cvc5 Term.
///
/// Returns `Err` (rather than panicking) for any application that the
/// whitelist gate ([`is_cvc5_lowerable`]) would also reject -- a name with
/// no cvc5 kind (e.g. `log`, CR2-1) or a wrong arity. The whitelist gate
/// already runs before lowering in [`solve_property_cvc5`], so every term
/// reaching here is provably cvc5-safe; these arms are defense in depth so
/// this function can never build a malformed term, panic, or hand cvc5 an
/// aborting term.
#[cfg(feature = "smt")]
pub fn lower_to_cvc5(
    tm: &cvc5_rs::TermManager,
    expr: &SmtExpr,
    vars: &std::collections::HashMap<String, cvc5_rs::Term>,
) -> Result<cvc5_rs::Term, String> {
    use cvc5_rs::Kind;

    Ok(match expr {
        // Defense in depth: an undeclared var is rejected by the whitelist
        // gate before lowering, so this is unreachable; return Err rather
        // than panicking on a missing key if it is ever hit.
        SmtExpr::Var(name) => vars
            .get(name)
            .cloned()
            .ok_or_else(|| format!("variable `{name}` has no declared cvc5 sort"))?,
        SmtExpr::RealLit(value) => {
            // Use rational string representation for cvc5
            tm.mk_real_from_str(&format!("{value}"))
        }
        SmtExpr::IntLit(value) => tm.mk_integer(*value),
        SmtExpr::BoolLit(value) => {
            if *value {
                tm.mk_true()
            } else {
                tm.mk_false()
            }
        }
        SmtExpr::Arith(op, left, right) => {
            let l = lower_to_cvc5(tm, left, vars)?;
            let r = lower_to_cvc5(tm, right, vars)?;
            match op {
                ArithOp::Add => tm.mk_term(Kind::CVC5_KIND_ADD, &[l, r]),
                ArithOp::Sub => tm.mk_term(Kind::CVC5_KIND_SUB, &[l, r]),
                ArithOp::Mul => tm.mk_term(Kind::CVC5_KIND_MULT, &[l, r]),
                ArithOp::Div => tm.mk_term(Kind::CVC5_KIND_DIVISION, &[l, r]),
                ArithOp::Neg => tm.mk_term(Kind::CVC5_KIND_NEG, &[l]),
            }
        }
        SmtExpr::Cmp(op, left, right) => {
            let l = lower_to_cvc5(tm, left, vars)?;
            let r = lower_to_cvc5(tm, right, vars)?;
            match op {
                CmpOp::Lt => tm.mk_term(Kind::CVC5_KIND_LT, &[l, r]),
                CmpOp::Le => tm.mk_term(Kind::CVC5_KIND_LEQ, &[l, r]),
                CmpOp::Gt => tm.mk_term(Kind::CVC5_KIND_GT, &[l, r]),
                CmpOp::Ge => tm.mk_term(Kind::CVC5_KIND_GEQ, &[l, r]),
                CmpOp::Eq => tm.mk_term(Kind::CVC5_KIND_EQUAL, &[l, r]),
                CmpOp::Ne => {
                    let eq = tm.mk_term(Kind::CVC5_KIND_EQUAL, &[l, r]);
                    tm.mk_term(Kind::CVC5_KIND_NOT, &[eq])
                }
            }
        }
        SmtExpr::Bool(op, children) => {
            let terms: Vec<_> = children
                .iter()
                .map(|c| lower_to_cvc5(tm, c, vars))
                .collect::<Result<Vec<_>, _>>()?;
            match op {
                BoolOp::And => tm.mk_term(Kind::CVC5_KIND_AND, &terms),
                BoolOp::Or => tm.mk_term(Kind::CVC5_KIND_OR, &terms),
                BoolOp::Implies => tm.mk_term(Kind::CVC5_KIND_IMPLIES, &terms),
            }
        }
        SmtExpr::Not(inner) => {
            let t = lower_to_cvc5(tm, inner, vars)?;
            tm.mk_term(Kind::CVC5_KIND_NOT, &[t])
        }
        SmtExpr::Forall(bindings, body) => {
            let bound_vars = quantifier_bound_vars(tm, bindings);
            let mut extended_vars = vars.clone();
            for (i, (name, _)) in bindings.iter().enumerate() {
                extended_vars.insert(name.clone(), bound_vars[i].clone());
            }
            let body_term = lower_to_cvc5(tm, body, &extended_vars)?;
            let bound_list = tm.mk_term(Kind::CVC5_KIND_VARIABLE_LIST, &bound_vars);
            tm.mk_term(Kind::CVC5_KIND_FORALL, &[bound_list, body_term])
        }
        SmtExpr::Exists(bindings, body) => {
            let bound_vars = quantifier_bound_vars(tm, bindings);
            let mut extended_vars = vars.clone();
            for (i, (name, _)) in bindings.iter().enumerate() {
                extended_vars.insert(name.clone(), bound_vars[i].clone());
            }
            let body_term = lower_to_cvc5(tm, body, &extended_vars)?;
            let bound_list = tm.mk_term(Kind::CVC5_KIND_VARIABLE_LIST, &bound_vars);
            tm.mk_term(Kind::CVC5_KIND_EXISTS, &[bound_list, body_term])
        }
        SmtExpr::Apply(name, args) => {
            // Defense in depth: arity/lowerability is already enforced by
            // validate_smt_arity before any lowering runs, but re-check here
            // so this function can never build a malformed term or panic
            // (CR2-1). A non-lowerable name (e.g. `log`) returns Err.
            match cvc5_lowerable_arity(name) {
                Some(n) if args.len() == n => {}
                Some(n) => {
                    return Err(format!(
                        "intrinsic `{name}` expects {n} argument(s), got {}",
                        args.len()
                    ));
                }
                None => {
                    return Err(format!(
                        "unsupported function `{name}` in cvc5 lowering (routes to Tier C)"
                    ));
                }
            }
            let lowered_args: Vec<_> = args
                .iter()
                .map(|a| lower_to_cvc5(tm, a, vars))
                .collect::<Result<Vec<_>, _>>()?;
            match name.as_str() {
                "exp" => tm.mk_term(Kind::CVC5_KIND_EXPONENTIAL, &lowered_args),
                "sqrt" => tm.mk_term(Kind::CVC5_KIND_SQRT, &lowered_args),
                "sin" => tm.mk_term(Kind::CVC5_KIND_SINE, &lowered_args),
                "cos" => tm.mk_term(Kind::CVC5_KIND_COSINE, &lowered_args),
                "abs" => tm.mk_term(Kind::CVC5_KIND_ABS, &lowered_args),
                "min" if lowered_args.len() == 2 => {
                    let cond = tm.mk_term(
                        Kind::CVC5_KIND_LT,
                        &[lowered_args[0].clone(), lowered_args[1].clone()],
                    );
                    tm.mk_term(
                        Kind::CVC5_KIND_ITE,
                        &[cond, lowered_args[0].clone(), lowered_args[1].clone()],
                    )
                }
                "max" if lowered_args.len() == 2 => {
                    let cond = tm.mk_term(
                        Kind::CVC5_KIND_GT,
                        &[lowered_args[0].clone(), lowered_args[1].clone()],
                    );
                    tm.mk_term(
                        Kind::CVC5_KIND_ITE,
                        &[cond, lowered_args[0].clone(), lowered_args[1].clone()],
                    )
                }
                // Unreachable after the cvc5_lowerable_arity guard above;
                // returns Err rather than panicking if it is ever hit
                // (e.g. CVC5_LOWERABLE gains a name with no match arm).
                other => {
                    return Err(format!(
                        "unsupported function `{other}` in cvc5 lowering (routes to Tier C)"
                    ));
                }
            }
        }
        SmtExpr::Ite(cond, then_expr, else_expr) => {
            let c = lower_to_cvc5(tm, cond, vars)?;
            let t = lower_to_cvc5(tm, then_expr, vars)?;
            let e = lower_to_cvc5(tm, else_expr, vars)?;
            tm.mk_term(Kind::CVC5_KIND_ITE, &[c, t, e])
        }
    })
}

/// Check if an SmtExpr contains a cvc5 transcendental call (exp, sin, cos,
/// sqrt) -- the subset of [`CVC5_LOWERABLE`] that selects the `QF_NRAT`
/// logic. Derived from [`CVC5_TRANSCENDENTAL`] so it cannot drift from the
/// lowering (CR2-1). `abs`/`min`/`max` are algebraic and stay in `QF_NRA`.
/// Whether an SmtExpr contains a quantifier (`Forall`/`Exists`). A
/// quantifier requires a NON-`QF_` cvc5 logic; under a `QF_` logic cvc5
/// aborts on any quantified atom, so the logic selection must know.
#[cfg(feature = "smt")]
fn contains_quantifier(expr: &SmtExpr) -> bool {
    match expr {
        SmtExpr::Forall(_, _) | SmtExpr::Exists(_, _) => true,
        SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
            contains_quantifier(l) || contains_quantifier(r)
        }
        SmtExpr::Bool(_, children) => children.iter().any(contains_quantifier),
        SmtExpr::Not(inner) => contains_quantifier(inner),
        SmtExpr::Apply(_, args) => args.iter().any(contains_quantifier),
        SmtExpr::Ite(c, t, e) => {
            contains_quantifier(c) || contains_quantifier(t) || contains_quantifier(e)
        }
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => false,
    }
}

#[cfg(feature = "smt")]
fn contains_transcendental(expr: &SmtExpr) -> bool {
    match expr {
        SmtExpr::Apply(name, args) => {
            CVC5_TRANSCENDENTAL.contains(&name.as_str()) || args.iter().any(contains_transcendental)
        }
        SmtExpr::Arith(_, l, r) => contains_transcendental(l) || contains_transcendental(r),
        SmtExpr::Cmp(_, l, r) => contains_transcendental(l) || contains_transcendental(r),
        SmtExpr::Bool(_, children) => children.iter().any(contains_transcendental),
        SmtExpr::Not(inner) => contains_transcendental(inner),
        SmtExpr::Ite(c, t, e) => {
            contains_transcendental(c) || contains_transcendental(t) || contains_transcendental(e)
        }
        SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => contains_transcendental(body),
        _ => false,
    }
}

#[cfg(all(test, feature = "smt"))]
mod tests {
    use super::*;
    use crate::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};

    #[test]
    fn proves_x_squared_non_negative() {
        // forall x: x*x >= 0
        let prop = SmtProperty {
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
        };
        let result = solve_property(&prop, 5000);
        assert_eq!(result, TierBResult::Proved);
    }

    #[test]
    fn disproves_x_always_positive() {
        // forall x: x > 0  (false — x=0 is a counterexample)
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let result = solve_property(&prop, 5000);
        match result {
            TierBResult::Disproved(model) => {
                assert!(model.is_object());
                assert!(model.get("x").is_some());
            }
            other => panic!("expected Disproved, got {other:?}"),
        }
    }

    #[test]
    fn proves_with_precondition() {
        // forall x where x > 0: x >= 0
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let result = solve_property(&prop, 5000);
        assert_eq!(result, TierBResult::Proved);
    }

    #[test]
    fn proves_intrinsic_value_non_negative() {
        // intrinsic_value(S, K) = if S > K then S - K else 0.0
        // Property: intrinsic_value(S, K) >= 0
        let prop = SmtProperty {
            variables: vec![
                ("S".to_string(), SmtSort::Real),
                ("K".to_string(), SmtSort::Real),
            ],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Ite(
                    Box::new(SmtExpr::Cmp(
                        CmpOp::Gt,
                        Box::new(SmtExpr::Var("S".to_string())),
                        Box::new(SmtExpr::Var("K".to_string())),
                    )),
                    Box::new(SmtExpr::Arith(
                        ArithOp::Sub,
                        Box::new(SmtExpr::Var("S".to_string())),
                        Box::new(SmtExpr::Var("K".to_string())),
                    )),
                    Box::new(SmtExpr::RealLit(0.0)),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let result = solve_property(&prop, 5000);
        assert_eq!(result, TierBResult::Proved);
    }

    fn intrinsic_ge_zero(name: &str, args: Vec<SmtExpr>) -> SmtProperty {
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(name.to_string(), args)),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        }
    }

    #[test]
    fn rt5_f1_zero_arg_transcendental_is_a_clean_error_not_a_cvc5_abort() {
        // RT5-F1: a zero-arg `exp()` previously built an invalid cvc5
        // EXPONENTIAL term (cvc5 aborts: empty stdout, bare exit 1). It
        // must lower to a clean TierBResult::Error, never an abort.
        let prop = intrinsic_ge_zero("exp", vec![]);
        match solve_property(&prop, 5000) {
            TierBResult::Error(reason) => {
                assert!(reason.contains("exp"), "names the bad intrinsic: {reason}");
            }
            other => panic!("expected Error for zero-arg exp(), got {other:?}"),
        }
    }

    #[test]
    fn rt5_f1_two_arg_transcendental_is_a_clean_error() {
        // The checker treats `exp` as variadic, so a two-arg `exp(a, b)`
        // reaches cvc5 for a "valid-looking" call. It must be a clean Error.
        let prop = intrinsic_ge_zero(
            "exp",
            vec![SmtExpr::Var("x".to_string()), SmtExpr::RealLit(1.0)],
        );
        match solve_property(&prop, 5000) {
            TierBResult::Error(reason) => assert!(reason.contains("exp"), "{reason}"),
            other => panic!("expected Error for two-arg exp(a,b), got {other:?}"),
        }
    }

    #[test]
    fn rt5_f1_wrong_arity_in_each_unary_transcendental_is_a_clean_error() {
        for name in ["exp", "log", "sqrt", "sin", "cos", "abs"] {
            // Zero args.
            assert!(
                matches!(
                    solve_property(&intrinsic_ge_zero(name, vec![]), 5000),
                    TierBResult::Error(_)
                ),
                "zero-arg {name}() must be a clean Error"
            );
            // Two args.
            let two =
                intrinsic_ge_zero(name, vec![SmtExpr::Var("x".into()), SmtExpr::RealLit(1.0)]);
            assert!(
                matches!(solve_property(&two, 5000), TierBResult::Error(_)),
                "two-arg {name}(a,b) must be a clean Error"
            );
        }
    }

    #[test]
    fn rt5_f1_correct_arity_transcendental_still_lowers_and_proves() {
        // Negative parity: a correct unary `exp(x) >= 0` still lowers and
        // proves (exp is always positive over the reals).
        let prop = intrinsic_ge_zero("exp", vec![SmtExpr::Var("x".to_string())]);
        assert_eq!(solve_property(&prop, 5000), TierBResult::Proved);
        // min/max keep their existing two-arg support.
        let max_prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(
                    "max".to_string(),
                    vec![SmtExpr::Var("x".to_string()), SmtExpr::RealLit(0.0)],
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        assert_eq!(solve_property(&max_prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn cr2_1_log_is_a_clean_unsupported_not_a_panic() {
        // CR2-1 CRITICAL: `log` is in the predicate grammar / chelis_pred
        // whitelist and works at Tier C, but cvc5 has no LOG kind, so
        // lower_to_cvc5 panics on it. A correct-arity `log(x)` must lower to
        // a clean Error (so dispatch falls to Tier C), never panic the
        // prove process. The old test suite only ever exercised exp(x).
        let prop = intrinsic_ge_zero("log", vec![SmtExpr::Var("x".to_string())]);
        match solve_property(&prop, 5000) {
            TierBResult::Error(reason) => {
                assert!(reason.contains("log"), "names the unsupported fn: {reason}");
            }
            other => panic!("expected Error for log(x), got {other:?}"),
        }
    }

    #[test]
    fn cr2_1_every_whitelisted_intrinsic_at_correct_arity_does_not_panic() {
        // Loop over EVERY whitelisted intrinsic at its correct arity so no
        // intrinsic is left untested -- that gap hid `log` twice. Each must
        // produce a determinate, panic-free outcome: a cvc5-lowerable fn
        // proves/disproves/times-out; a whitelisted-but-not-lowerable fn
        // (log) is a clean Error.
        for &name in chelis_pred::INTRINSIC_WHITELIST {
            let arity = if name == "min" || name == "max" { 2 } else { 1 };
            let args: Vec<SmtExpr> = (0..arity).map(|_| SmtExpr::Var("x".into())).collect();
            let prop = intrinsic_ge_zero(name, args);
            // The point is the absence of a panic and a determinate result.
            let result = solve_property(&prop, 5000);
            assert!(
                matches!(
                    result,
                    TierBResult::Proved
                        | TierBResult::Disproved(_)
                        | TierBResult::Timeout
                        | TierBResult::Unknown
                        | TierBResult::Error(_)
                ),
                "intrinsic `{name}` at arity {arity} must yield a determinate result, got {result:?}"
            );
        }
    }

    #[test]
    fn cr2_1_cvc5_lowerable_set_matches_lowering_and_transcendental_classification() {
        // The single source of truth (CVC5_LOWERABLE) must (a) NOT contain
        // log (cvc5 has no LOG), (b) be a subset of the predicate-grammar
        // whitelist, and (c) drive contains_transcendental so the three
        // lists cannot drift again.
        assert!(
            !CVC5_LOWERABLE.iter().any(|(n, _)| *n == "log"),
            "log is not cvc5-lowerable"
        );
        for &(f, _) in CVC5_LOWERABLE {
            assert!(
                chelis_pred::INTRINSIC_WHITELIST.contains(&f),
                "cvc5-lowerable `{f}` must be in the predicate grammar"
            );
        }
        // CVC5_TRANSCENDENTAL is a strict subset of the lowerable names.
        for &t in CVC5_TRANSCENDENTAL {
            assert!(
                CVC5_LOWERABLE.iter().any(|(n, _)| *n == t),
                "transcendental `{t}` must be cvc5-lowerable"
            );
        }
        // contains_transcendental agrees with CVC5_TRANSCENDENTAL membership
        // for every lowerable intrinsic (abs/min/max are algebraic).
        for &(f, _) in CVC5_LOWERABLE {
            let is_transcendental = CVC5_TRANSCENDENTAL.contains(&f);
            let expr = SmtExpr::Apply(f.to_string(), vec![SmtExpr::Var("x".into())]);
            assert_eq!(
                contains_transcendental(&expr),
                is_transcendental,
                "contains_transcendental disagrees on `{f}`"
            );
        }
    }

    // --- U3: operand-sort mismatch is a clean Error, not a cvc5 abort ---

    /// An Int-sorted var compared against a Real literal. cvc5's
    /// `mk_term(GEQ, [Int, Real])` ABORTS the process; the U3 pre-check must
    /// turn it into a clean `TierBResult::Error` so dispatch routes to Tier C.
    #[test]
    fn u3_int_vs_real_comparison_is_a_clean_error_not_a_cvc5_abort() {
        let prop = SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("n".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        match solve_property(&prop, 5000) {
            TierBResult::Error(reason) => {
                assert!(
                    reason.contains("sort mismatch") || reason.to_lowercase().contains("int"),
                    "Error names the sort clash: {reason}"
                );
            }
            other => panic!("expected a clean Error for Int-vs-Real comparison, got {other:?}"),
        }
    }

    /// The mismatch is also caught when it sits inside an arithmetic term
    /// (`Int_var + Real_lit`), and when it is buried in a precondition.
    #[test]
    fn u3_int_vs_real_in_arith_and_precondition_is_a_clean_error() {
        // Int var + Real literal inside the postcondition.
        let arith = SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Add,
                    Box::new(SmtExpr::Var("n".to_string())),
                    Box::new(SmtExpr::RealLit(1.5)),
                )),
                Box::new(SmtExpr::IntLit(0)),
            ),
        };
        assert!(
            matches!(solve_property(&arith, 5000), TierBResult::Error(_)),
            "Int+Real arithmetic is a clean Error"
        );
        // Mismatch buried in a precondition.
        let pre = SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Var("n".to_string())),
                Box::new(SmtExpr::RealLit(10.0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("n".to_string())),
                Box::new(SmtExpr::IntLit(0)),
            ),
        };
        assert!(
            matches!(solve_property(&pre, 5000), TierBResult::Error(_)),
            "Int-vs-Real in a precondition is a clean Error"
        );
    }

    /// Negative parity: consistent sorts still solve. An all-Int property
    /// proves, and an all-Real property proves -- the pre-check must not
    /// reject a sound term.
    #[test]
    fn u3_consistent_sorts_still_solve() {
        let int_prop = SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("n".to_string())),
                Box::new(SmtExpr::IntLit(0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("n".to_string())),
                Box::new(SmtExpr::IntLit(0)),
            ),
        };
        assert_eq!(solve_property(&int_prop, 5000), TierBResult::Proved);

        let real_prop = SmtProperty {
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
        };
        assert_eq!(solve_property(&real_prop, 5000), TierBResult::Proved);
    }

    // --- F1 (review 4): min/max over a mixed-sort pair must NOT abort cvc5 ---
    //
    // cvc5 lowers min/max to an internal ITE(LT(a,b),a,b), which ABORTS the
    // process ("Subexpressions must have the same type: Int/Real") on an
    // Int-vs-Real operand pair. The sort pre-check only ever clash-checked a
    // binary op's two operands against each other and recursed into an
    // Apply's args INDIVIDUALLY, never against EACH OTHER, so min(int, real)
    // still aborted. The pre-check must reject the whole property to Tier C.

    fn min_max_prop_cmp(name: &str, a: SmtExpr, b: SmtExpr, rhs: SmtExpr) -> SmtProperty {
        // (name(a, b)) >= rhs, with n declared Int.
        SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(name.to_string(), vec![a, b])),
                Box::new(rhs),
            ),
        }
    }

    fn min_max_prop(name: &str, a: SmtExpr, b: SmtExpr) -> SmtProperty {
        // A mixed-sort min/max: the comparison rhs sort is irrelevant because
        // the abort is inside the intrinsic's own operand pair.
        min_max_prop_cmp(name, a, b, SmtExpr::RealLit(0.0))
    }

    #[test]
    fn f1_min_max_mixed_sort_is_clean_error_not_a_cvc5_abort() {
        // min/max over (Int var, Real lit) and (Real lit, Int var): the
        // intrinsic lowers to a sort-mismatched ITE in cvc5. Must be a clean
        // TierBResult::Error (route to Tier C), NEVER a process abort.
        for name in ["min", "max"] {
            let a = min_max_prop(name, SmtExpr::Var("n".to_string()), SmtExpr::RealLit(1.5));
            assert!(
                matches!(solve_property(&a, 5000), TierBResult::Error(_)),
                "{name}(int, real) must be a clean Error, not a cvc5 abort"
            );
            let b = min_max_prop(name, SmtExpr::RealLit(1.5), SmtExpr::Var("n".to_string()));
            assert!(
                matches!(solve_property(&b, 5000), TierBResult::Error(_)),
                "{name}(real, int) must be a clean Error, not a cvc5 abort"
            );
        }
    }

    #[test]
    fn f1_min_max_consistent_sort_still_proves() {
        // Negative parity: an all-Int and an all-Real min/max still lower and
        // prove -- the conservative route-out must not reject a sound term.
        // max(n, 0) >= 0 with n: Int (the comparison rhs is also Int, so the
        // whole term is Int-sorted and cvc5 accepts it).
        let int_prop = min_max_prop_cmp(
            "max",
            SmtExpr::Var("n".to_string()),
            SmtExpr::IntLit(0),
            SmtExpr::IntLit(0),
        );
        assert_eq!(
            solve_property(&int_prop, 5000),
            TierBResult::Proved,
            "all-Int max proves"
        );
        // max(x, 0.0) >= 0.0 with x: Real.
        let real_prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(
                    "max".to_string(),
                    vec![SmtExpr::Var("x".to_string()), SmtExpr::RealLit(0.0)],
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        assert_eq!(
            solve_property(&real_prop, 5000),
            TierBResult::Proved,
            "all-Real max proves"
        );
    }

    #[test]
    fn f1_comparison_against_min_max_result_with_mixed_operands_is_clean_error() {
        // A comparison whose operand is a min/max with a mixed-sort argument
        // pair must also route out, not abort. min(n, 1.5) is itself the
        // abort source; comparing it against an Int literal must still be a
        // clean Error.
        let prop = SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(
                    "min".to_string(),
                    vec![SmtExpr::Var("n".to_string()), SmtExpr::RealLit(1.5)],
                )),
                Box::new(SmtExpr::IntLit(0)),
            ),
        };
        assert!(
            matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
            "comparison against a mixed-sort min result is a clean Error"
        );
    }

    // --- F2 (review 4): unary neg must not be spuriously rejected ---
    //
    // neg is represented as Arith(Neg, x, RealLit(0.0)) with a DUMMY second
    // operand. The pre-check treated it as binary and clash-checked x against
    // the RealLit(0.0) placeholder, so neg(int) was spuriously rejected to
    // Tier C even though cvc5 lowers NEG as unary and ignores the placeholder.

    #[test]
    fn f2_neg_of_int_proves_at_tier_b_not_spuriously_rejected() {
        // neg(n) <= 0 with n: Int, n >= 0. The placeholder RealLit(0.0) must
        // not trip the Int-vs-Real clash; this proves at Tier B.
        let prop = SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("n".to_string())),
                Box::new(SmtExpr::IntLit(0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Arith(
                    ArithOp::Neg,
                    Box::new(SmtExpr::Var("n".to_string())),
                    Box::new(SmtExpr::RealLit(0.0)),
                )),
                Box::new(SmtExpr::IntLit(0)),
            ),
        };
        assert_eq!(
            solve_property(&prop, 5000),
            TierBResult::Proved,
            "neg(int) >= ... proves at Tier B, not spuriously routed to Tier C"
        );
    }

    #[test]
    fn f2_neg_of_real_still_proves() {
        // Negative parity: neg of a real also proves (the placeholder is the
        // same sort here, so this never regressed, but lock it).
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Arith(
                    ArithOp::Neg,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(0.0)),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        assert_eq!(solve_property(&prop, 5000), TierBResult::Proved);
    }

    // ===================================================================
    // Review 5: whitelist gate. The pre-lowering gate admits ONLY
    // provably-cvc5-safe terms; everything else routes to Tier C. These
    // red tests exercise abort shapes the blacklist missed.
    // ===================================================================

    /// A transcendental over an INTEGER argument. cvc5's SQRT/EXP/SINE/COSINE
    /// require a Real argument; an Int argument aborts the process. The
    /// whitelist must require a Real arg and route this to Tier C.
    #[test]
    fn w5_transcendental_over_int_arg_is_clean_error_not_abort() {
        for name in ["sqrt", "exp", "sin", "cos"] {
            let prop = SmtProperty {
                variables: vec![("n".to_string(), SmtSort::Int)],
                preconditions: vec![],
                postcondition: SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Apply(
                        name.to_string(),
                        vec![SmtExpr::Var("n".to_string())],
                    )),
                    Box::new(SmtExpr::RealLit(0.0)),
                ),
            };
            assert!(
                matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
                "{name}(int) must be a clean Error (Real arg required), not a cvc5 abort"
            );
        }
    }

    /// A Bool operand inside a numeric comparison. cvc5 aborts comparing a
    /// Bool against an Int/Real. The whitelist must reject a Bool operand in
    /// a numeric comparison and route to Tier C.
    #[test]
    fn w5_bool_vs_int_comparison_is_clean_error_not_abort() {
        let prop = SmtProperty {
            variables: vec![
                ("b".to_string(), SmtSort::Bool),
                ("n".to_string(), SmtSort::Int),
            ],
            preconditions: vec![],
            // b >= n : Bool compared against Int.
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("b".to_string())),
                Box::new(SmtExpr::Var("n".to_string())),
            ),
        };
        assert!(
            matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
            "a Bool-vs-Int comparison must be a clean Error, not a cvc5 abort"
        );
    }

    /// A quantifier whose bound var sort differs from a literal it is
    /// compared against. The whitelist must seed the bound var into the sort
    /// env, then reject the sort mismatch in the body -> Tier C, never abort.
    #[test]
    fn w5_quantifier_bound_var_sort_mismatch_is_clean_error_not_abort() {
        // forall (k: Int) . k >= 0.5   -- Int bound var vs Real literal.
        let prop = SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: SmtExpr::Forall(
                vec![("k".to_string(), SmtSort::Int)],
                Box::new(SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Var("k".to_string())),
                    Box::new(SmtExpr::RealLit(0.5)),
                )),
            ),
        };
        assert!(
            matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
            "a quantifier bound-var sort mismatch must be a clean Error, not a cvc5 abort"
        );
    }

    /// A quantifier whose bound var IS used consistently is ADMITTED by the
    /// whitelist (the bound var sort is seeded into the env, so the body
    /// lowers cleanly) and does NOT abort cvc5. The prove layer uses a
    /// quantifier-free logic (no production path emits a top-level
    /// quantifier; they are routed to Tier C by the fuzzability classifier),
    /// so cvc5 returns a clean determinate result, never a process abort.
    #[test]
    fn w5_quantifier_bound_var_consistent_lowers_without_abort() {
        let prop = SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: SmtExpr::Forall(
                vec![("k".to_string(), SmtSort::Real)],
                Box::new(SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Arith(
                        ArithOp::Mul,
                        Box::new(SmtExpr::Var("k".to_string())),
                        Box::new(SmtExpr::Var("k".to_string())),
                    )),
                    Box::new(SmtExpr::RealLit(0.0)),
                )),
            ),
        };
        // The whitelist must ADMIT this (env seeding makes the body lowerable).
        assert!(
            is_cvc5_lowerable(&prop).is_ok(),
            "a consistent Real-bound quantifier is admitted by the whitelist"
        );
        // A quantifier selects a quantified (non-QF) logic, so cvc5 can prove
        // it (forall k: Real . k*k >= 0) -- and never aborts.
        assert_eq!(
            solve_property(&prop, 5000),
            TierBResult::Proved,
            "a consistent Real-bound quantifier proves under the quantified logic"
        );
    }

    /// An unknown-sort var (not declared in `variables`) must be NOT
    /// lowerable (the whitelist default is reject), routing to Tier C rather
    /// than panicking on a missing var in lowering.
    #[test]
    fn w5_unknown_var_sort_is_clean_error_not_panic() {
        let prop = SmtProperty {
            variables: vec![], // `z` is undeclared.
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("z".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        assert!(
            matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
            "an undeclared var sort must be a clean Error, not a panic"
        );
    }

    // --- flagship cases the whitelist MUST still admit (prove at smt) ---

    #[test]
    fn w5_flagship_real_arith_and_comparison_still_proves() {
        // f32 scalar arithmetic + comparison: (x - 1.0) <= x  always.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Arith(
                    ArithOp::Sub,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(1.0)),
                )),
                Box::new(SmtExpr::Var("x".to_string())),
            ),
        };
        assert_eq!(solve_property(&prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn w5_flagship_int32_comparison_still_proves() {
        // int32 comparison with a precondition: n >= 0 => n + 1 >= 1.
        let prop = SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("n".to_string())),
                Box::new(SmtExpr::IntLit(0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Add,
                    Box::new(SmtExpr::Var("n".to_string())),
                    Box::new(SmtExpr::IntLit(1)),
                )),
                Box::new(SmtExpr::IntLit(1)),
            ),
        };
        assert_eq!(solve_property(&prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn w5_flagship_sqrt_of_real_field_still_proves() {
        // sqrt over a Real arg is admitted: sqrt(x) >= 0 with x >= 0.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply(
                    "sqrt".to_string(),
                    vec![SmtExpr::Var("x".to_string())],
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        assert_eq!(solve_property(&prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn w5_flagship_min_real_real_still_proves() {
        // min(x, 0.0) <= 0.0 always (both Real).
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Le,
                Box::new(SmtExpr::Apply(
                    "min".to_string(),
                    vec![SmtExpr::Var("x".to_string()), SmtExpr::RealLit(0.0)],
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        assert_eq!(solve_property(&prop, 5000), TierBResult::Proved);
    }

    #[test]
    fn w5_flagship_bool_equality_still_proves() {
        // Equality of two Bool operands is admitted: b == b always.
        let prop = SmtProperty {
            variables: vec![("b".to_string(), SmtSort::Bool)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Eq,
                Box::new(SmtExpr::Var("b".to_string())),
                Box::new(SmtExpr::Var("b".to_string())),
            ),
        };
        assert_eq!(solve_property(&prop, 5000), TierBResult::Proved);
    }
}
