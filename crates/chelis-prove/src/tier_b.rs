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

/// Require a sort to be numeric (`Int` or `Real`); a `Bool` in an
/// arithmetic or numeric-comparison position aborts cvc5, so it routes to
/// Tier C instead.
#[cfg(feature = "smt")]
fn require_numeric_sort(sort: SmtSort, what: &str) -> Result<(), String> {
    match sort {
        SmtSort::Int | SmtSort::Real => Ok(()),
        SmtSort::Bool => Err(format!("{what} is Bool, not numeric (routes to Tier C)")),
    }
}

/// Maximum `SmtExpr` nesting depth Tier B will lower. A property deeper than
/// this routes to Tier C rather than risking a stack overflow in the
/// recursive lowering / logic-selection walks (a stack overflow aborts the
/// PROCESS, it does not return). Real invariant predicates are a few levels
/// deep; this bound is far above any realistic predicate and far below the
/// stack-overflow threshold observed even on small test-worker stacks (RT6
/// round-2).
#[cfg(feature = "smt")]
const MAX_SMT_EXPR_DEPTH: usize = 256;

/// Whether `expr` nests deeper than `max`, computed ITERATIVELY (an explicit
/// work stack) so the bound check itself cannot overflow the call stack on a
/// pathologically deep tree -- which is the whole point of running it before
/// any recursive walk.
#[cfg(feature = "smt")]
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

#[cfg(feature = "smt")]
fn solve_property_cvc5(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
    use cvc5_rs::{Kind, Solver, TermManager};
    use std::collections::HashMap;

    // SAFETY MODEL (review 6 -- TOTAL LOWERING): `lower_to_cvc5` is the SOLE
    // authority on cvc5-safety, and it is TOTAL -- every `mk_term` call site
    // first verifies cvc5's requirement for that kind (operand sorts AND
    // arity), and any violation returns `Err` (routed to a clean Tier C
    // result) BEFORE `mk_term` is reached. cvc5's `mk_term` ABORTS THE PROCESS
    // on a malformed term (a sort-mismatched comparison, a zero/one-child
    // `and`/`or`, a non-binary `implies`, sqrt over an Int, a non-finite
    // literal), surfacing to a JSON consumer as an empty-stdout bare exit -- a
    // machine-contract violation. Review 5 used a SEPARATE pre-lowering sort
    // gate; it admitted terms (it checked sorts but not arity) the builder
    // then aborted on (RT6). Folding the gate INTO the builder makes the
    // check and the construction one bottom-up pass that returns the cvc5
    // sort it builds, so the two can no longer diverge: a term that reaches
    // `mk_term` is provably well-formed by construction.
    //
    // The bottom-up lowering and the `contains_*` logic-selection walks are
    // RECURSIVE on `SmtExpr` depth; a pathologically deep tree overflows the
    // stack, which aborts the process (SIGABRT) the same way a malformed term
    // does -- and it happens on the way DOWN, before any `mk_term` check runs.
    // Bound the depth FIRST, with an iterative check that cannot itself
    // overflow, so every later recursive walk runs on a bounded tree (RT6
    // round-2).
    if smt_expr_exceeds_depth(&property.postcondition, MAX_SMT_EXPR_DEPTH)
        || property
            .preconditions
            .iter()
            .any(|p| smt_expr_exceeds_depth(p, MAX_SMT_EXPR_DEPTH))
    {
        return TierBResult::Error(format!(
            "property nests deeper than {MAX_SMT_EXPR_DEPTH} levels (routes to Tier C)"
        ));
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

    // 1. Declare variables. `sorts` mirrors `vars` so the lowering knows each
    //    variable's cvc5 sort without re-querying cvc5 (and so a var absent
    //    from the declared set is a clean Err, never a panic or an abort).
    let mut vars: HashMap<String, cvc5_rs::Term> = HashMap::new();
    let mut sorts: HashMap<String, SmtSort> = HashMap::new();
    for (name, sort) in &property.variables {
        let cvc5_sort = match sort {
            SmtSort::Real => tm.real_sort(),
            SmtSort::Int => tm.integer_sort(),
            SmtSort::Bool => tm.boolean_sort(),
        };
        let var = tm.mk_const(cvc5_sort, name);
        vars.insert(name.clone(), var);
        sorts.insert(name.clone(), *sort);
    }

    // 2. Assert preconditions. cvc5's `assert_formula` requires a Bool-sorted
    //    term; asserting a non-Bool aborts ("Expected term with sort Bool"),
    //    so a precondition that lowers to a non-Bool sort routes to Tier C.
    for pre in &property.preconditions {
        let term = match lower_to_cvc5(&tm, pre, &vars, &sorts) {
            Ok((t, SmtSort::Bool)) => t,
            Ok((_, other)) => {
                return TierBResult::Error(format!(
                    "precondition lowers to sort {other:?}, expected Bool (routes to Tier C)"
                ));
            }
            Err(reason) => return TierBResult::Error(reason),
        };
        solver.assert_formula(term);
    }

    // 3. Assert negation of postcondition. cvc5's NOT requires a Bool operand;
    //    `NOT(non-bool)` aborts ("expecting a Boolean subexpression"), so a
    //    postcondition that lowers to a non-Bool sort routes to Tier C.
    let post_term = match lower_to_cvc5(&tm, &property.postcondition, &vars, &sorts) {
        Ok((t, SmtSort::Bool)) => t,
        Ok((_, other)) => {
            return TierBResult::Error(format!(
                "postcondition lowers to sort {other:?}, expected Bool (routes to Tier C)"
            ));
        }
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

/// Lower an SmtExpr to a cvc5 `(Term, SmtSort)`, where the returned sort is
/// EXACTLY the sort cvc5 assigns to the built term.
///
/// This function is TOTAL with respect to aborts: every `mk_term` call site
/// first verifies cvc5's requirement for that kind -- operand sorts AND
/// arity -- and returns `Err` (routed to Tier C by the caller) rather than
/// handing cvc5 a term it would abort on. cvc5's `mk_term` aborts the
/// PROCESS on a malformed term (a sort-mismatched comparison, a zero/one-
/// child `and`/`or`, a non-binary `implies`, a transcendental over an Int, a
/// non-finite literal), so the safety of the whole prove pipeline rests on
/// no aborting term ever reaching `mk_term`. Folding the old separate sort
/// gate INTO this builder (review 6) is what guarantees that: the sort this
/// returns is the sort cvc5 builds, so a parent node's arity/sort check sees
/// the truth (e.g. integer `/` is reported `Real` because cvc5 promotes it),
/// and the check can no longer diverge from the construction.
///
/// `sorts` carries each variable's declared sort, extended with quantifier
/// bound-var sorts as we descend into a `Forall`/`Exists` body.
#[cfg(feature = "smt")]
pub fn lower_to_cvc5(
    tm: &cvc5_rs::TermManager,
    expr: &SmtExpr,
    vars: &std::collections::HashMap<String, cvc5_rs::Term>,
    sorts: &std::collections::HashMap<String, SmtSort>,
) -> Result<(cvc5_rs::Term, SmtSort), String> {
    use cvc5_rs::Kind;

    Ok(match expr {
        SmtExpr::Var(name) => {
            let term = vars
                .get(name)
                .cloned()
                .ok_or_else(|| format!("variable `{name}` has no declared cvc5 term"))?;
            let sort = sorts
                .get(name)
                .copied()
                .ok_or_else(|| format!("variable `{name}` has no known sort (routes to Tier C)"))?;
            (term, sort)
        }
        SmtExpr::RealLit(value) => {
            // cvc5's `mk_real_from_str` aborts on a non-finite token
            // (`inf`/`-inf`/`NaN`); a non-finite invariant literal routes to
            // Tier C instead (RT6 #13).
            if !value.is_finite() {
                return Err(format!(
                    "non-finite real literal `{value}` cannot lower to cvc5 (routes to Tier C)"
                ));
            }
            (tm.mk_real_from_str(&format!("{value}")), SmtSort::Real)
        }
        SmtExpr::IntLit(value) => (tm.mk_integer(*value), SmtSort::Int),
        SmtExpr::BoolLit(value) => {
            let t = if *value { tm.mk_true() } else { tm.mk_false() };
            (t, SmtSort::Bool)
        }
        // Unary `neg` is represented as `Arith(Neg, x, <placeholder>)`; cvc5
        // lowers NEG as unary and ignores the placeholder, so lower ONLY the
        // real operand.
        SmtExpr::Arith(ArithOp::Neg, left, _placeholder) => {
            let (l, ls) = lower_to_cvc5(tm, left, vars, sorts)?;
            require_numeric_sort(ls, "neg operand")?;
            (tm.mk_term(Kind::CVC5_KIND_NEG, &[l]), ls)
        }
        SmtExpr::Arith(op, left, right) => {
            let (l, ls) = lower_to_cvc5(tm, left, vars, sorts)?;
            let (r, rs) = lower_to_cvc5(tm, right, vars, sorts)?;
            require_numeric_sort(ls, "arithmetic operand")?;
            require_numeric_sort(rs, "arithmetic operand")?;
            if ls != rs {
                return Err(format!(
                    "arithmetic operands have differing sorts {ls:?} vs {rs:?} (routes to Tier C)"
                ));
            }
            let term = match op {
                ArithOp::Add => tm.mk_term(Kind::CVC5_KIND_ADD, &[l, r]),
                ArithOp::Sub => tm.mk_term(Kind::CVC5_KIND_SUB, &[l, r]),
                ArithOp::Mul => tm.mk_term(Kind::CVC5_KIND_MULT, &[l, r]),
                ArithOp::Div => tm.mk_term(Kind::CVC5_KIND_DIVISION, &[l, r]),
                ArithOp::Neg => unreachable!("Neg handled in the arm above"),
            };
            // cvc5's `/` (CVC5_KIND_DIVISION) is REAL division: its result is
            // Real even over two Int operands (cvc5 promotes them). Reporting
            // the operand sort (Int) here was the divergence that let a parent
            // Int comparison see Int-vs-Int, pass its own check, then abort
            // cvc5 on a Real-vs-Int term (RT6 #4/#12).
            let result_sort = if matches!(op, ArithOp::Div) {
                SmtSort::Real
            } else {
                ls
            };
            (term, result_sort)
        }
        SmtExpr::Cmp(op, left, right) => {
            let (l, ls) = lower_to_cvc5(tm, left, vars, sorts)?;
            let (r, rs) = lower_to_cvc5(tm, right, vars, sorts)?;
            // Both operands must lower to the SAME cvc5 sort; a mixed pair
            // aborts cvc5. (cvc5 would coerce Int->Real for some mixes, but
            // routing the mix to Tier C is sound -- the property is still
            // decided there, never aborted.)
            if ls != rs {
                return Err(format!(
                    "comparison operands have differing sorts {ls:?} vs {rs:?} (routes to Tier C)"
                ));
            }
            // A numeric comparison (Lt/Le/Gt/Ge) over Bool operands aborts
            // cvc5; Eq/Ne admit any equal sort (including Bool == Bool).
            let is_numeric_cmp = matches!(op, CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge);
            if is_numeric_cmp && ls == SmtSort::Bool {
                return Err("numeric comparison over a Bool operand (routes to Tier C)".to_string());
            }
            let term = match op {
                CmpOp::Lt => tm.mk_term(Kind::CVC5_KIND_LT, &[l, r]),
                CmpOp::Le => tm.mk_term(Kind::CVC5_KIND_LEQ, &[l, r]),
                CmpOp::Gt => tm.mk_term(Kind::CVC5_KIND_GT, &[l, r]),
                CmpOp::Ge => tm.mk_term(Kind::CVC5_KIND_GEQ, &[l, r]),
                CmpOp::Eq => tm.mk_term(Kind::CVC5_KIND_EQUAL, &[l, r]),
                CmpOp::Ne => {
                    let eq = tm.mk_term(Kind::CVC5_KIND_EQUAL, &[l, r]);
                    tm.mk_term(Kind::CVC5_KIND_NOT, &[eq])
                }
            };
            (term, SmtSort::Bool)
        }
        SmtExpr::Bool(op, children) => {
            let lowered: Vec<(cvc5_rs::Term, SmtSort)> = children
                .iter()
                .map(|c| lower_to_cvc5(tm, c, vars, sorts))
                .collect::<Result<Vec<_>, _>>()?;
            for (_, s) in &lowered {
                if *s != SmtSort::Bool {
                    return Err(format!(
                        "boolean connective operand has sort {s:?}, expected Bool (routes to Tier C)"
                    ));
                }
            }
            let terms: Vec<cvc5_rs::Term> = lowered.into_iter().map(|(t, _)| t).collect();
            // cvc5's AND/OR require AT LEAST 2 children and IMPLIES EXACTLY 2;
            // a degenerate arity aborts the process. Normalize the boundary
            // cases to their logical meaning so a single-conjunct invariant
            // still proves at the SMT tier rather than aborting (RT6 #1/#2/
            // #5/#6/#7/#8/#11/#15).
            let term = match op {
                BoolOp::And => match terms.len() {
                    0 => tm.mk_true(),
                    1 => terms.into_iter().next().expect("len checked == 1"),
                    _ => tm.mk_term(Kind::CVC5_KIND_AND, &terms),
                },
                BoolOp::Or => match terms.len() {
                    0 => tm.mk_false(),
                    1 => terms.into_iter().next().expect("len checked == 1"),
                    _ => tm.mk_term(Kind::CVC5_KIND_OR, &terms),
                },
                BoolOp::Implies => {
                    if terms.len() != 2 {
                        return Err(format!(
                            "`implies` requires exactly 2 operands, got {} (routes to Tier C)",
                            terms.len()
                        ));
                    }
                    tm.mk_term(Kind::CVC5_KIND_IMPLIES, &terms)
                }
            };
            (term, SmtSort::Bool)
        }
        SmtExpr::Not(inner) => {
            let (t, s) = lower_to_cvc5(tm, inner, vars, sorts)?;
            if s != SmtSort::Bool {
                return Err(format!(
                    "`not` operand has sort {s:?}, expected Bool (routes to Tier C)"
                ));
            }
            (tm.mk_term(Kind::CVC5_KIND_NOT, &[t]), SmtSort::Bool)
        }
        SmtExpr::Forall(bindings, body) | SmtExpr::Exists(bindings, body) => {
            let bound_vars = quantifier_bound_vars(tm, bindings);
            let mut extended_vars = vars.clone();
            let mut extended_sorts = sorts.clone();
            for (i, (name, sort)) in bindings.iter().enumerate() {
                extended_vars.insert(name.clone(), bound_vars[i].clone());
                extended_sorts.insert(name.clone(), *sort);
            }
            let (body_term, body_sort) = lower_to_cvc5(tm, body, &extended_vars, &extended_sorts)?;
            if body_sort != SmtSort::Bool {
                return Err(format!(
                    "quantifier body has sort {body_sort:?}, expected Bool (routes to Tier C)"
                ));
            }
            // cvc5's VARIABLE_LIST requires >= 1 bound var; an empty binder
            // list aborts the process (RT6 round-2). A quantifier over no
            // variables is degenerate -- `forall (). P` and `exists (). P`
            // both mean `P` -- so normalize it to the body directly rather
            // than building an empty VARIABLE_LIST.
            if bound_vars.is_empty() {
                return Ok((body_term, SmtSort::Bool));
            }
            let bound_list = tm.mk_term(Kind::CVC5_KIND_VARIABLE_LIST, &bound_vars);
            let kind = if matches!(expr, SmtExpr::Forall(_, _)) {
                Kind::CVC5_KIND_FORALL
            } else {
                Kind::CVC5_KIND_EXISTS
            };
            (tm.mk_term(kind, &[bound_list, body_term]), SmtSort::Bool)
        }
        SmtExpr::Apply(name, args) => {
            let arity = cvc5_lowerable_arity(name).ok_or_else(|| {
                format!("unsupported function `{name}` in cvc5 lowering (routes to Tier C)")
            })?;
            if args.len() != arity {
                return Err(format!(
                    "intrinsic `{name}` expects {arity} argument(s), got {} (routes to Tier C)",
                    args.len()
                ));
            }
            let lowered: Vec<(cvc5_rs::Term, SmtSort)> = args
                .iter()
                .map(|a| lower_to_cvc5(tm, a, vars, sorts))
                .collect::<Result<Vec<_>, _>>()?;
            let arg_sorts: Vec<SmtSort> = lowered.iter().map(|(_, s)| *s).collect();
            let lowered_args: Vec<cvc5_rs::Term> = lowered.iter().map(|(t, _)| t.clone()).collect();
            match name.as_str() {
                // cvc5 SQRT/EXP/SINE/COSINE require a REAL argument; an Int
                // argument aborts the process, so it routes to Tier C.
                "exp" | "sqrt" | "sin" | "cos" => {
                    if arg_sorts[0] != SmtSort::Real {
                        return Err(format!(
                            "`{name}` requires a Real argument, got {:?} (routes to Tier C)",
                            arg_sorts[0]
                        ));
                    }
                    let kind = match name.as_str() {
                        "exp" => Kind::CVC5_KIND_EXPONENTIAL,
                        "sqrt" => Kind::CVC5_KIND_SQRT,
                        "sin" => Kind::CVC5_KIND_SINE,
                        "cos" => Kind::CVC5_KIND_COSINE,
                        _ => unreachable!("matched above"),
                    };
                    (tm.mk_term(kind, &lowered_args), SmtSort::Real)
                }
                "abs" => {
                    require_numeric_sort(arg_sorts[0], "abs operand")?;
                    (tm.mk_term(Kind::CVC5_KIND_ABS, &lowered_args), arg_sorts[0])
                }
                // `min`/`max` lower to ITE(cmp(a,b), a, b); both args must
                // share one numeric sort (a mixed pair aborts the inner cmp).
                "min" | "max" => {
                    require_numeric_sort(arg_sorts[0], "min/max operand")?;
                    require_numeric_sort(arg_sorts[1], "min/max operand")?;
                    if arg_sorts[0] != arg_sorts[1] {
                        return Err(format!(
                            "`{name}` operands have differing sorts {:?} vs {:?} (routes to Tier C)",
                            arg_sorts[0], arg_sorts[1]
                        ));
                    }
                    let cmp_kind = if name == "min" {
                        Kind::CVC5_KIND_LT
                    } else {
                        Kind::CVC5_KIND_GT
                    };
                    let cond = tm.mk_term(
                        cmp_kind,
                        &[lowered_args[0].clone(), lowered_args[1].clone()],
                    );
                    let ite = tm.mk_term(
                        Kind::CVC5_KIND_ITE,
                        &[cond, lowered_args[0].clone(), lowered_args[1].clone()],
                    );
                    (ite, arg_sorts[0])
                }
                // Unreachable after the cvc5_lowerable_arity guard above;
                // returns Err rather than panicking if it is ever hit (e.g.
                // CVC5_LOWERABLE gains a name with no match arm).
                other => {
                    return Err(format!(
                        "unsupported function `{other}` in cvc5 lowering (routes to Tier C)"
                    ));
                }
            }
        }
        SmtExpr::Ite(cond, then_expr, else_expr) => {
            let (c, cs) = lower_to_cvc5(tm, cond, vars, sorts)?;
            if cs != SmtSort::Bool {
                return Err(format!(
                    "if-then-else condition has sort {cs:?}, expected Bool (routes to Tier C)"
                ));
            }
            let (t, ts) = lower_to_cvc5(tm, then_expr, vars, sorts)?;
            let (e, es) = lower_to_cvc5(tm, else_expr, vars, sorts)?;
            if ts != es {
                return Err(format!(
                    "if-then-else branches have differing sorts {ts:?} vs {es:?} (routes to Tier C)"
                ));
            }
            (tm.mk_term(Kind::CVC5_KIND_ITE, &[c, t, e]), ts)
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

    /// A quantifier whose bound var IS used consistently lowers cleanly (the
    /// bound var sort is seeded into the lowering, so the body builds) and
    /// does NOT abort cvc5. A quantifier selects a quantified (non-QF) logic,
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
        // The total lowering must build this (bound-var sort seeding makes the
        // body lowerable) and cvc5 proves it (forall k: Real . k*k >= 0) --
        // never aborts.
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

    // ===================================================================
    // Review 6: TOTAL lowering. The review-5 separate sort gate ADMITTED
    // terms (it checked operand sorts but not cvc5's arity / kind-domain
    // requirements) that the builder then aborted cvc5 on. Folding the
    // gate into the builder makes every `mk_term` call site verify cvc5's
    // requirement first. Each test below drives a shape that ABORTED cvc5
    // under review 5; reaching the assertion at all proves no process
    // abort (an abort would crash the test binary with empty output).
    // ===================================================================

    /// Empty `and`/`or` connectives. cvc5's AND/OR require >= 2 children, so
    /// a zero-child connective aborts the process. The total lowering
    /// normalizes them to their logical identity (empty `and` is true, empty
    /// `or` is false), so they yield a determinate result, never an abort.
    #[test]
    fn rt6_empty_and_or_do_not_abort() {
        let empty_and = SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: SmtExpr::Bool(BoolOp::And, vec![]),
        };
        // empty `and` == true, which is trivially provable.
        assert_eq!(solve_property(&empty_and, 5000), TierBResult::Proved);

        let empty_or = SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: SmtExpr::Bool(BoolOp::Or, vec![]),
        };
        // empty `or` == false, which is disproved by the empty model.
        assert!(
            matches!(solve_property(&empty_or, 5000), TierBResult::Disproved(_)),
            "empty `or` (false) must be cleanly disproved, not abort cvc5"
        );
    }

    /// A single-child `and`/`or`. cvc5's AND/OR require >= 2 children, so a
    /// one-child connective aborts. The total lowering unwraps it to the
    /// child so a SINGLE-CONJUNCT invariant still PROVES at the SMT tier
    /// (the over-conservative alternative -- routing to Tier C -- would be a
    /// flagship regression). RT6 #2.
    #[test]
    fn rt6_single_child_and_or_still_proves_at_smt() {
        // and(x*x >= 0) -- one conjunct, always true over the reals.
        let child = SmtExpr::Cmp(
            CmpOp::Ge,
            Box::new(SmtExpr::Arith(
                ArithOp::Mul,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::Var("x".to_string())),
            )),
            Box::new(SmtExpr::RealLit(0.0)),
        );
        for op in [BoolOp::And, BoolOp::Or] {
            let prop = SmtProperty {
                variables: vec![("x".to_string(), SmtSort::Real)],
                preconditions: vec![],
                postcondition: SmtExpr::Bool(op, vec![child.clone()]),
            };
            assert_eq!(
                solve_property(&prop, 5000),
                TierBResult::Proved,
                "a single-conjunct `{op:?}` must prove at the SMT tier, not abort or route to Tier C"
            );
        }
    }

    /// A non-binary `implies`. cvc5's IMPLIES requires EXACTLY 2 children, so
    /// a 1-child (or >2-child) implies aborts. The total lowering returns a
    /// clean Error (routes to Tier C); a correct binary implies still proves.
    /// RT6 #5/#8/#11.
    #[test]
    fn rt6_implies_arity_one_is_clean_error_two_proves() {
        let one_child = SmtProperty {
            variables: vec![("b".to_string(), SmtSort::Bool)],
            preconditions: vec![],
            postcondition: SmtExpr::Bool(BoolOp::Implies, vec![SmtExpr::Var("b".to_string())]),
        };
        assert!(
            matches!(solve_property(&one_child, 5000), TierBResult::Error(_)),
            "a 1-child `implies` must be a clean Error, not a cvc5 abort"
        );

        // x >= 1 ==> x >= 0 is valid (binary implies).
        let two_child = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Bool(
                BoolOp::Implies,
                vec![
                    SmtExpr::Cmp(
                        CmpOp::Ge,
                        Box::new(SmtExpr::Var("x".to_string())),
                        Box::new(SmtExpr::RealLit(1.0)),
                    ),
                    SmtExpr::Cmp(
                        CmpOp::Ge,
                        Box::new(SmtExpr::Var("x".to_string())),
                        Box::new(SmtExpr::RealLit(0.0)),
                    ),
                ],
            ),
        };
        assert_eq!(solve_property(&two_child, 5000), TierBResult::Proved);
    }

    /// Integer `/` feeding a comparison. cvc5's `/` (DIVISION) is REAL
    /// division: its result is Real even over Int operands. The review-5
    /// gate reported the division as Int, so a parent `Eq(int_div, int_var)`
    /// passed its same-sort check, then cvc5 aborted on the Real-vs-Int
    /// equality. The total lowering reports the division as Real, so the
    /// parent comparison sees the mismatch and routes to a clean Error
    /// (Tier C), never an abort. RT6 #4/#12.
    #[test]
    fn rt6_int_division_feeding_comparison_is_clean_error_not_abort() {
        let prop = SmtProperty {
            variables: vec![
                ("a".to_string(), SmtSort::Int),
                ("b".to_string(), SmtSort::Int),
                ("c".to_string(), SmtSort::Int),
            ],
            preconditions: vec![],
            // (a / b) == c : the division is Real, c is Int -> mismatch.
            postcondition: SmtExpr::Cmp(
                CmpOp::Eq,
                Box::new(SmtExpr::Arith(
                    ArithOp::Div,
                    Box::new(SmtExpr::Var("a".to_string())),
                    Box::new(SmtExpr::Var("b".to_string())),
                )),
                Box::new(SmtExpr::Var("c".to_string())),
            ),
        };
        assert!(
            matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
            "an Int `/` feeding an Int comparison must be a clean Error, not a cvc5 abort"
        );
    }

    /// A non-finite real literal. cvc5's `mk_real_from_str` aborts on an
    /// `inf`/`-inf`/`NaN` token. The total lowering returns a clean Error
    /// (routes to Tier C) before reaching cvc5. RT6 #13.
    #[test]
    fn rt6_non_finite_real_literal_is_clean_error_not_abort() {
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let prop = SmtProperty {
                variables: vec![("x".to_string(), SmtSort::Real)],
                preconditions: vec![],
                postcondition: SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(bad)),
                ),
            };
            assert!(
                matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
                "a non-finite real literal ({bad}) must be a clean Error, not a cvc5 abort"
            );
        }
    }

    // --- RT6 round-2: holes the fresh red-team / self-audit found after the
    //     first total-lowering pass. Each is the SAME class (an mk_term /
    //     assert_formula requirement not checked at its call site). ---

    /// An empty quantifier binder list. cvc5's VARIABLE_LIST requires >= 1
    /// bound var; `mk_term(VARIABLE_LIST, &[])` aborts. The lowering
    /// normalizes `forall (). P` / `exists (). P` to `P` (both mean `P`), so
    /// it decides cleanly instead of aborting.
    #[test]
    fn rt6_empty_quantifier_binder_does_not_abort() {
        for forall in [true, false] {
            // body: x*x >= 0 (always true over the reals), x is a free var.
            let body = SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::Var("x".to_string())),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            );
            let q = if forall {
                SmtExpr::Forall(vec![], Box::new(body))
            } else {
                SmtExpr::Exists(vec![], Box::new(body))
            };
            let prop = SmtProperty {
                variables: vec![("x".to_string(), SmtSort::Real)],
                preconditions: vec![],
                postcondition: q,
            };
            // `forall (). x*x >= 0` == `x*x >= 0`, which is valid -> Proved.
            assert_eq!(
                solve_property(&prop, 5000),
                TierBResult::Proved,
                "an empty-binder quantifier (forall={forall}) must normalize to its body and prove, not abort"
            );
        }
    }

    /// A postcondition that lowers to a non-Bool sort. `solve_property_cvc5`
    /// negates the postcondition with cvc5 NOT, which aborts on a non-Bool
    /// operand ("expecting a Boolean subexpression"). It must route to a
    /// clean Error instead.
    #[test]
    fn rt6_non_bool_postcondition_is_clean_error_not_abort() {
        for bad_post in [
            SmtExpr::Var("x".to_string()),
            SmtExpr::IntLit(5),
            SmtExpr::Arith(
                ArithOp::Add,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(1.0)),
            ),
        ] {
            let prop = SmtProperty {
                variables: vec![("x".to_string(), SmtSort::Real)],
                preconditions: vec![],
                postcondition: bad_post,
            };
            assert!(
                matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
                "a non-Bool postcondition must be a clean Error, not a cvc5 abort"
            );
        }
    }

    /// A precondition that lowers to a non-Bool sort. `solve_property_cvc5`
    /// asserts each precondition, and cvc5's `assert_formula` aborts on a
    /// non-Bool term ("Expected term with sort Bool"). It must route to a
    /// clean Error instead.
    #[test]
    fn rt6_non_bool_precondition_is_clean_error_not_abort() {
        let prop = SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![SmtExpr::Arith(
                ArithOp::Add,
                Box::new(SmtExpr::Var("n".to_string())),
                Box::new(SmtExpr::IntLit(1)),
            )],
            postcondition: SmtExpr::BoolLit(true),
        };
        assert!(
            matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
            "a non-Bool precondition must be a clean Error, not a cvc5 abort"
        );
    }

    /// A pathologically deep property. The recursive lowering and the
    /// `contains_*` logic-selection walks would overflow the stack (SIGABRT,
    /// not a returned result) on a deep-enough tree -- an abort on the way
    /// DOWN, before any mk_term check runs. The iterative depth bound routes
    /// it to a clean Error. The shallow twin still proves, confirming the
    /// bound is far above realistic predicates (RT6 round-2).
    #[test]
    fn rt6_deeply_nested_property_is_clean_error_not_stack_overflow() {
        // 5000 nested `not`, far past the depth bound and the overflow point.
        let mut deep = SmtExpr::Cmp(
            CmpOp::Ge,
            Box::new(SmtExpr::Arith(
                ArithOp::Mul,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::Var("x".to_string())),
            )),
            Box::new(SmtExpr::RealLit(0.0)),
        );
        for _ in 0..5000 {
            deep = SmtExpr::Not(Box::new(deep));
        }
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: deep,
        };
        // Reaching this assertion at all proves no stack overflow occurred.
        assert!(
            matches!(solve_property(&prop, 5000), TierBResult::Error(_)),
            "a property past the depth bound must be a clean Error, not a stack-overflow abort"
        );

        // A shallow nesting (8 `not`) is well under the bound and still proves
        // (not(not(...(x*x >= 0))) with an even count is x*x >= 0).
        let mut shallow = SmtExpr::Cmp(
            CmpOp::Ge,
            Box::new(SmtExpr::Arith(
                ArithOp::Mul,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::Var("x".to_string())),
            )),
            Box::new(SmtExpr::RealLit(0.0)),
        );
        for _ in 0..8 {
            shallow = SmtExpr::Not(Box::new(shallow));
        }
        let prop_shallow = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: shallow,
        };
        assert_eq!(
            solve_property(&prop_shallow, 5000),
            TierBResult::Proved,
            "a shallow nesting must still prove at the SMT tier"
        );
    }
}
