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

/// Satisfiability of a property's assumptions alone.
#[derive(Debug, Clone, PartialEq)]
pub enum AssumptionSatisfiability {
    /// The assumptions are satisfiable; the proof is non-vacuous.
    Sat(Value),
    /// The assumptions are contradictory; a green proof would be invalid.
    Unsat,
    /// Solver timed out.
    Timeout,
    /// Solver returned unknown.
    Unknown,
    /// The assumption formula could not be lowered safely.
    Error(String),
}

/// Structured property input for SMT solving.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
    if let Some(forced) = forced_smt_result_from_env() {
        return forced;
    }
    #[cfg(feature = "smt")]
    {
        // When a production host has enabled isolation (only the `chelis`
        // binary does, via `chelis_prove::enable_isolation`), run cvc5 in a
        // short-lived CHILD process so that ANY way the solve can take the
        // process down -- a cvc5 C++ abort on a term the in-process guards
        // somehow still admit, a cvc5-internal assertion on a well-formed
        // formula, a stack overflow, an OOM kill, a panic -- becomes a clean
        // Tier C result in the parent instead of a bare process exit. Tests
        // do not enable isolation, so they solve in-process (no spawn).
        if crate::worker::isolation_enabled() {
            crate::worker::solve_property_isolated(property, timeout_ms)
        } else {
            solve_property_cvc5(property, timeout_ms)
        }
    }
    #[cfg(not(feature = "smt"))]
    {
        let _ = (property, timeout_ms);
        TierBResult::Timeout
    }
}

/// Check only the assumptions/preconditions for satisfiability. This is the
/// non-vacuity oracle: SAT establishes that the assumed domain is inhabited;
/// UNSAT invalidates any claimed green proof under those assumptions; unknown
/// and timeout are unsupported, never failed.
pub fn check_assumptions_satisfiable(
    property: &SmtProperty,
    timeout_ms: u64,
) -> AssumptionSatisfiability {
    let assumptions_as_property = SmtProperty {
        variables: property.variables.clone(),
        preconditions: property.preconditions.clone(),
        postcondition: SmtExpr::BoolLit(false),
    };
    match solve_property(&assumptions_as_property, timeout_ms) {
        TierBResult::Proved => AssumptionSatisfiability::Unsat,
        TierBResult::Disproved(model) => AssumptionSatisfiability::Sat(model),
        TierBResult::Timeout => AssumptionSatisfiability::Timeout,
        TierBResult::Unknown => AssumptionSatisfiability::Unknown,
        TierBResult::Error(reason) => AssumptionSatisfiability::Error(reason),
    }
}

/// Test-only surface for integration tests that need deterministic
/// timeout/unknown classification without relying on host cvc5 timing.
fn forced_smt_result_from_env() -> Option<TierBResult> {
    match std::env::var("CHELIS_PROVE_TEST_FORCE_SMT_RESULT")
        .ok()?
        .as_str()
    {
        "proved" => Some(TierBResult::Proved),
        "disproved" => Some(TierBResult::Disproved(serde_json::json!({
            "__forced": true
        }))),
        "timeout" => Some(TierBResult::Timeout),
        "unknown" => Some(TierBResult::Unknown),
        value => Some(TierBResult::Error(format!(
            "invalid CHELIS_PROVE_TEST_FORCE_SMT_RESULT value `{value}`"
        ))),
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

/// The reason an `Apply` of `name` cannot lower to a cvc5 term (chelis#434).
///
/// A function reaches the lowering only after [`classify_inlineability`]
/// admits it, which uses the BROADER predicate grammar whitelist
/// ([`chelis_pred::INTRINSIC_WHITELIST`], includes `log`) -- not the
/// narrower cvc5-buildable set ([`CVC5_LOWERABLE`], excludes `log`). So a
/// transcendental that is a legitimate Tier-C predicate intrinsic but has no
/// cvc5 kind (today only `log`) lands here. Distinguishing it from a
/// truly-unknown symbol turns the smt-only diagnostic from an internal-
/// looking generic "unsupported function `log`" into an HONEST capability
/// boundary: the SMT tier does not model this transcendental, so the goal is
/// Unsupported, not broken.
///
/// SEAM (WS-7 / Beacon): the faithful discharge of a transcendental finance
/// goal (e.g. Black-Scholes positivity through `normal_cdf`/`exp`/`log`) is
/// an envelope-plus-polynomial bound (Sollya-generated) discharged by the
/// out-of-tree Beacon engine, NOT in-tree cvc5 NRA. Until that lands a
/// transcendental cvc5 cannot lower is honestly Unsupported here.
#[cfg(feature = "smt")]
fn unsupported_apply_reason(name: &str) -> String {
    // A predicate-grammar transcendental with no cvc5 kind (today `log`): a
    // capability boundary, not a bug. Keep `log` in the text -- the
    // CR2-1 regression test pins that the reason names the function.
    if chelis_pred::TRANSCENDENTAL_WHITELIST.contains(&name) && cvc5_lowerable_arity(name).is_none()
    {
        return format!(
            "transcendental function `{name}` is not supported by the SMT tier \
             (cvc5 has no kind for it); the goal is Unsupported. A faithful \
             discharge via a Sollya envelope-plus-polynomial bound is tracked \
             by WS-7 (Beacon engine)"
        );
    }
    format!("unsupported function `{name}` in cvc5 lowering (routes to Tier C)")
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

/// Whether any part of a property (the postcondition or a precondition) nests
/// past [`MAX_SMT_EXPR_DEPTH`]. Run this BEFORE any recursive walk OR recursive
/// clone/serialize of the property -- not only the lowering but `SmtExpr`'s
/// derived `Clone` and `Serialize` recurse on depth, so the isolation parent
/// must screen depth before it `clone`s / bincode-serializes the property to
/// the worker, or a deep property overflows the PARENT (which is not isolated
/// from itself) before any child is spawned.
#[cfg(feature = "smt")]
pub(crate) fn property_exceeds_smt_depth(property: &SmtProperty) -> bool {
    smt_expr_exceeds_depth(&property.postcondition, MAX_SMT_EXPR_DEPTH)
        || property
            .preconditions
            .iter()
            .any(|p| smt_expr_exceeds_depth(p, MAX_SMT_EXPR_DEPTH))
}

/// Whether a variable/binder name is safe to hand to cvc5's `mk_const` /
/// `mk_var`. `cvc5-rs` builds a `CString` from the name and `unwrap()`s it,
/// so an interior NUL byte PANICS out of `solve_property` (a recoverable
/// unwind under the default profile, a hard abort under `panic=abort`) --
/// either way it fails to return a `TierBResult`. A name with an interior
/// NUL routes to Tier C instead (RT6 round-2). cvc5 tolerates every other
/// byte sequence (reserved words, whitespace, parens, unicode) as an opaque
/// symbol.
#[cfg(feature = "smt")]
fn smt_name_is_cvc5_safe(name: &str) -> bool {
    !name.as_bytes().contains(&0)
}

/// Collect every distinct `sqrt` argument in deterministic expression order.
///
/// cvc5's real `SQRT` is partial: at a negative argument its value is
/// underspecified. A nested or quantified occurrence is kept out of the
/// domain-proof fragment deliberately. Supporting either shape would require
/// proving a scoped obligation rather than the single free-variable
/// implication built by [`authorize_sqrt_domains`].
#[cfg(feature = "smt")]
fn collect_sqrt_arguments(
    expr: &SmtExpr,
    under_quantifier: bool,
    arguments: &mut Vec<SmtExpr>,
) -> Result<(), String> {
    match expr {
        SmtExpr::Apply(name, args) if name == "sqrt" && args.len() == 1 => {
            let argument = &args[0];
            if under_quantifier {
                return Err(sqrt_domain_error(
                    "a quantified `sqrt` needs a scoped domain proof",
                ));
            }
            if contains_sqrt(argument) {
                return Err(sqrt_domain_error(
                    "a nested `sqrt` argument is outside the domain-proof fragment",
                ));
            }
            if !is_total_algebraic_numeric_expr(argument) {
                return Err(sqrt_domain_error(
                    "its argument is outside the total algebraic domain-proof fragment",
                ));
            }
            if !arguments.iter().any(|known| known == argument) {
                arguments.push(argument.clone());
            }
            Ok(())
        }
        SmtExpr::Apply(_, args) => {
            for argument in args {
                collect_sqrt_arguments(argument, under_quantifier, arguments)?;
            }
            Ok(())
        }
        SmtExpr::Arith(_, left, right) | SmtExpr::Cmp(_, left, right) => {
            collect_sqrt_arguments(left, under_quantifier, arguments)?;
            collect_sqrt_arguments(right, under_quantifier, arguments)
        }
        SmtExpr::Bool(_, children) => {
            for child in children {
                collect_sqrt_arguments(child, under_quantifier, arguments)?;
            }
            Ok(())
        }
        SmtExpr::Not(inner) => collect_sqrt_arguments(inner, under_quantifier, arguments),
        SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => {
            collect_sqrt_arguments(body, true, arguments)
        }
        SmtExpr::Ite(condition, then_branch, else_branch) => {
            collect_sqrt_arguments(condition, under_quantifier, arguments)?;
            collect_sqrt_arguments(then_branch, under_quantifier, arguments)?;
            collect_sqrt_arguments(else_branch, under_quantifier, arguments)
        }
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => Ok(()),
    }
}

#[cfg(feature = "smt")]
fn contains_sqrt(expr: &SmtExpr) -> bool {
    match expr {
        SmtExpr::Apply(name, args) => name == "sqrt" || args.iter().any(contains_sqrt),
        SmtExpr::Arith(_, left, right) | SmtExpr::Cmp(_, left, right) => {
            contains_sqrt(left) || contains_sqrt(right)
        }
        SmtExpr::Bool(_, children) => children.iter().any(contains_sqrt),
        SmtExpr::Not(inner) => contains_sqrt(inner),
        SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => contains_sqrt(body),
        SmtExpr::Ite(condition, then_branch, else_branch) => {
            contains_sqrt(condition) || contains_sqrt(then_branch) || contains_sqrt(else_branch)
        }
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => false,
    }
}

/// The deliberately small numeric fragment used by the auxiliary domain
/// proof. Every operation here is total over cvc5's Int/Real sorts. Division,
/// conditionals, applications, and quantifiers stay out: using a partial or
/// scoped term to authorize another partial term would only move the
/// soundness hole.
#[cfg(feature = "smt")]
fn is_total_algebraic_numeric_expr(expr: &SmtExpr) -> bool {
    match expr {
        SmtExpr::Var(_) | SmtExpr::IntLit(_) => true,
        SmtExpr::RealLit(value) => value.is_finite(),
        SmtExpr::Arith(ArithOp::Neg, operand, _) => is_total_algebraic_numeric_expr(operand),
        SmtExpr::Arith(ArithOp::Add | ArithOp::Sub | ArithOp::Mul, left, right) => {
            is_total_algebraic_numeric_expr(left) && is_total_algebraic_numeric_expr(right)
        }
        SmtExpr::Arith(ArithOp::Div, _, _)
        | SmtExpr::BoolLit(_)
        | SmtExpr::Cmp(_, _, _)
        | SmtExpr::Bool(_, _)
        | SmtExpr::Not(_)
        | SmtExpr::Forall(_, _)
        | SmtExpr::Exists(_, _)
        | SmtExpr::Apply(_, _)
        | SmtExpr::Ite(_, _, _) => false,
    }
}

/// Flatten only top-level conjunctions and retain atomic comparisons whose
/// operands belong to the total algebraic fragment. In particular, a
/// precondition containing `sqrt` can never authorize its own argument.
#[cfg(feature = "smt")]
fn collect_sqrt_domain_evidence(expr: &SmtExpr, evidence: &mut Vec<SmtExpr>) {
    match expr {
        SmtExpr::Bool(BoolOp::And, children) => {
            for child in children {
                collect_sqrt_domain_evidence(child, evidence);
            }
        }
        SmtExpr::Cmp(_, left, right)
            if is_total_algebraic_numeric_expr(left) && is_total_algebraic_numeric_expr(right) =>
        {
            evidence.push(expr.clone());
        }
        SmtExpr::BoolLit(_) => evidence.push(expr.clone()),
        _ => {}
    }
}

#[cfg(feature = "smt")]
fn sqrt_domain_error(detail: &str) -> String {
    format!(
        "cannot prove every `sqrt` argument non-negative from the user's other total, \
         sqrt-free conjunctive preconditions: {detail} (chelis#1475; routes to Tier C)"
    )
}

#[cfg(feature = "smt")]
fn require_proved_sqrt_domain(result: TierBResult) -> Result<(), String> {
    match result {
        TierBResult::Proved => Ok(()),
        TierBResult::Disproved(_) => Err(sqrt_domain_error("the domain obligation is false")),
        TierBResult::Timeout => Err(sqrt_domain_error("the domain obligation timed out")),
        TierBResult::Unknown => Err(sqrt_domain_error("the domain obligation is unknown")),
        TierBResult::Error(reason) => Err(sqrt_domain_error(&format!(
            "the domain obligation could not be lowered safely: {reason}"
        ))),
    }
}

/// Prove, without assuming any new user-visible facts, that every exact
/// `sqrt` argument in the property is non-negative.
///
/// The one auxiliary implication uses only independent, sqrt-free conjuncts
/// from the user's preconditions. Only UNSAT of the negated conjunction
/// (`TierBResult::Proved`) authorizes lowering. SAT, timeout, unknown, and any
/// lowering error all fail closed.
#[cfg(feature = "smt")]
fn authorize_sqrt_domains(property: &SmtProperty, timeout_ms: u64) -> Result<Vec<SmtExpr>, String> {
    let mut arguments = Vec::new();
    for precondition in &property.preconditions {
        collect_sqrt_arguments(precondition, false, &mut arguments)?;
    }
    collect_sqrt_arguments(&property.postcondition, false, &mut arguments)?;
    if arguments.is_empty() {
        return Ok(arguments);
    }

    let mut evidence = Vec::new();
    for precondition in &property.preconditions {
        collect_sqrt_domain_evidence(precondition, &mut evidence);
    }
    let obligations = arguments
        .iter()
        .cloned()
        .map(|argument| {
            SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(argument),
                Box::new(SmtExpr::RealLit(0.0)),
            )
        })
        .collect();
    let obligation = SmtProperty {
        variables: property.variables.clone(),
        preconditions: evidence,
        postcondition: SmtExpr::Bool(BoolOp::And, obligations),
    };
    require_proved_sqrt_domain(solve_property_cvc5(&obligation, timeout_ms))?;
    Ok(arguments)
}

/// Solve a property IN-PROCESS with cvc5. This is the function the isolation
/// worker child actually runs; the parent reaches it only when isolation is
/// disabled (every test, and any non-`chelis` host that does not opt in).
#[cfg(feature = "smt")]
pub(crate) fn solve_property_cvc5(property: &SmtProperty, timeout_ms: u64) -> TierBResult {
    use cvc5_rs::{Kind, Solver, TermManager};
    use std::collections::BTreeMap;

    // SAFETY MODEL (review 6 -- TOTAL LOWERING): `lower_to_cvc5` is the sole
    // authority on cvc5 term-construction safety, and it is TOTAL -- every
    // `mk_term` call site first verifies cvc5's requirement for that kind
    // (operand sorts AND arity), and any violation returns `Err` (routed to a
    // clean Tier C result) BEFORE `mk_term` is reached. The sqrt-domain
    // preflight above is separately authoritative for whether cvc5's partial
    // SQRT kind may be constructed at all. cvc5's `mk_term` ABORTS THE PROCESS
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
    if property_exceeds_smt_depth(property) {
        return TierBResult::Error(format!(
            "property nests deeper than {MAX_SMT_EXPR_DEPTH} levels (routes to Tier C)"
        ));
    }

    let authorized_sqrt_arguments = match authorize_sqrt_domains(property, timeout_ms) {
        Ok(arguments) => arguments,
        Err(reason) => return TierBResult::Error(reason),
    };

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
    let mut vars: BTreeMap<String, cvc5_rs::Term> = BTreeMap::new();
    let mut sorts: chelis_unord::UnordMap<String, SmtSort> = chelis_unord::UnordMap::new();
    for (name, sort) in &property.variables {
        if !smt_name_is_cvc5_safe(name) {
            return TierBResult::Error(
                "variable name contains an interior NUL byte, cannot lower to cvc5 (routes to Tier C)"
                    .to_string(),
            );
        }
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
    //
    //    Each asserted term is retained so step 4 can ask the solver to
    //    evaluate it under the solver's OWN returned model (chelis#1224). An
    //    assumption that never reached the assertion stack is otherwise
    //    invisible: the model simply comes back unconstrained by it.
    let mut precondition_terms: Vec<cvc5_rs::Term> =
        Vec::with_capacity(property.preconditions.len());
    for pre in &property.preconditions {
        let term = match lower_to_cvc5_with_sqrt_domains(
            &tm,
            pre,
            &vars,
            &sorts,
            &authorized_sqrt_arguments,
        ) {
            Ok((t, SmtSort::Bool)) => t,
            Ok((_, other)) => {
                return TierBResult::Error(format!(
                    "precondition lowers to sort {other:?}, expected Bool (routes to Tier C)"
                ));
            }
            Err(reason) => return TierBResult::Error(reason),
        };
        solver.assert_formula(term.clone());
        precondition_terms.push(term);
    }

    // The auxiliary proof established these facts from the user's own
    // assumptions. Re-asserting them here is semantically redundant, but it
    // makes cvc5's partial-SQRT domain explicit in the main query rather than
    // relying on the solver to rediscover the implication while evaluating
    // SQRT.
    for argument in &authorized_sqrt_arguments {
        let (argument_term, argument_sort) = match lower_to_cvc5_with_sqrt_domains(
            &tm,
            argument,
            &vars,
            &sorts,
            &authorized_sqrt_arguments,
        ) {
            Ok(lowered) => lowered,
            Err(reason) => return TierBResult::Error(reason),
        };
        if argument_sort != SmtSort::Real {
            return TierBResult::Error(sqrt_domain_error(&format!(
                "a proved argument lowers to {argument_sort:?}, expected Real"
            )));
        }
        let zero = tm.mk_real(0);
        solver.assert_formula(tm.mk_term(Kind::CVC5_KIND_GEQ, &[argument_term, zero]));
    }

    // 3. Assert negation of postcondition. cvc5's NOT requires a Bool operand;
    //    `NOT(non-bool)` aborts ("expecting a Boolean subexpression"), so a
    //    postcondition that lowers to a non-Bool sort routes to Tier C.
    let post_term = match lower_to_cvc5_with_sqrt_domains(
        &tm,
        &property.postcondition,
        &vars,
        &sorts,
        &authorized_sqrt_arguments,
    ) {
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
        // chelis#1224: a returned model MUST satisfy every stated assumption.
        // A model that violates one is not a counterexample to this property,
        // it is evidence that the query the solver answered was not the query
        // we built. Report that as a typed error so the disproof is never
        // laundered into a `Failed` verdict.
        if let Err(reason) = validate_model_satisfies_preconditions(
            property,
            &bindings,
            &solver,
            &precondition_terms,
        ) {
            return TierBResult::Error(reason);
        }
        TierBResult::Disproved(Value::Object(bindings))
    } else {
        // Unknown or timeout
        TierBResult::Unknown
    }
}

/// Reject a `sat` model that does not satisfy the property's own assumptions
/// (chelis#1224).
///
/// The observed failure was a model with `d = -4.0` returned for a property
/// whose `where` clause states `d > 0.5`. Such a model is a sound
/// counterexample to the goal *without* that assumption, so reporting it as a
/// disproof of the stated property is a wrong answer, not a weak one.
///
/// Two checks run, and both must pass:
///
/// 1. **Solver-side.** Ask cvc5 to evaluate each retained precondition term
///    under its own model. This is exact and needs no value parsing, but it
///    shares whatever state produced the model.
/// 2. **Independent.** Parse the model back into exact rationals and
///    re-evaluate the preconditions without consulting the solver, which is
///    the point: if the solver's own state is the thing that went wrong, only
///    an independent evaluation can see it.
///
/// Leg 2 evaluates in `BigRational`, not `f64`. cvc5 decides in exact rational
/// arithmetic and does not model IEEE-754 rounding (see the `SOUNDNESS:` /
/// `KNOWN LIMITATION` note in `lower_to_cvc5`'s `SmtExpr::RealLit` arm), so an
/// `f64` re-evaluation disagrees with the solver
/// wherever a witness is not representable: cvc5 answers `x > 1e17` with
/// `100000000000000001.0`, which rounds to exactly `1e17` in `f64` and reads as
/// violating its own bound. Every strict bound past 2^53 would lose its
/// disproof that way.
///
/// Leg 2 is also **one-sided**: it rejects only when it has exactly decided a
/// precondition to be false, and abstains whenever it cannot decide one
/// (a quantifier, an uninterpreted application, a value form it cannot parse,
/// division by zero). A guard that cannot decide must not discard a
/// counterexample, because the cost of a false rejection is a silently weaker
/// prover: `TierBResult::Error` degrades the disproof to Tier C fuzzing under
/// `auto`, and to `Unsupported` under `smt-only`.
///
/// Abstention is still silent by design. Partial `sqrt` cannot exploit that
/// silence: before any SQRT term is built, [`authorize_sqrt_domains`] proves
/// each exact argument non-negative from independent total preconditions and
/// otherwise routes the property to Tier C (chelis#1475). Widening this model
/// guard to reject on any non-`true` would instead reintroduce the
/// false-rejection class that made the first version of this check unsound in
/// the rejecting direction.
#[cfg(feature = "smt")]
fn validate_model_satisfies_preconditions(
    property: &SmtProperty,
    bindings: &serde_json::Map<String, Value>,
    solver: &cvc5_rs::Solver,
    precondition_terms: &[cvc5_rs::Term],
) -> Result<(), String> {
    // Leg 1: the solver's own evaluation of each assumption under its model.
    // A term cvc5 cannot fully evaluate comes back as a residual expression
    // rather than `true`/`false`; that is not a decided violation, so only an
    // explicit `false` rejects here.
    for (index, term) in precondition_terms.iter().enumerate() {
        if solver.get_value(term.clone()).to_string().trim() == "false" {
            return Err(format!(
                "cvc5 returned a model that does not satisfy precondition {index}: the solver \
                 evaluates that assumption to `false` under its own model, so the model is not a \
                 counterexample to the stated property (chelis#1224); routing to Tier C"
            ));
        }
    }

    // Leg 2: independent re-evaluation, without consulting the solver.
    validate_model_independently(property, bindings)
}

/// The solver-free half of [`validate_model_satisfies_preconditions`].
///
/// Split out so the check that matters most can be unit-tested against a
/// planted model without standing up cvc5: if the solver's own state is what
/// went wrong, this is the leg that catches it.
///
/// One-sided by construction: a variable this function cannot read, or a
/// precondition it cannot decide exactly, yields no verdict rather than a
/// rejection.
#[cfg(feature = "smt")]
fn validate_model_independently(
    property: &SmtProperty,
    bindings: &serde_json::Map<String, Value>,
) -> Result<(), String> {
    let mut env: std::collections::BTreeMap<String, ExactValue> = std::collections::BTreeMap::new();
    for (name, sort) in &property.variables {
        // A variable this function cannot read is simply left out of the
        // environment. `eval_exact_bool` then abstains on any precondition
        // that mentions it, while preconditions over the readable variables
        // are still decided. Returning early here instead would switch the
        // whole leg off because ONE unrelated variable came back as, say, an
        // algebraic number, which is routine in the NRA logic this file
        // selects.
        let Some(value) = bindings
            .get(name)
            .and_then(Value::as_str)
            .and_then(|raw| parse_smt_model_value(raw, *sort))
        else {
            continue;
        };
        env.insert(name.clone(), value);
    }
    for (index, pre) in property.preconditions.iter().enumerate() {
        if eval_exact_bool(pre, &env, 0) == Some(false) {
            return Err(format!(
                "cvc5 returned a model that exactly violates precondition {index} under \
                 independent rational re-evaluation, so it is not a counterexample to the stated \
                 property (chelis#1224); routing to Tier C"
            ));
        }
    }
    Ok(())
}

/// A model value read back exactly. Reals and integers share one exact
/// rational representation; there is no `f64` anywhere on this path.
#[cfg(feature = "smt")]
#[derive(Debug, Clone, PartialEq)]
enum ExactValue {
    Num(num_rational::BigRational),
    Bool(bool),
}

/// Bound on the nesting this evaluator and the model-value parser will walk.
///
/// `solve_property_cvc5` already refuses a property deeper than
/// `MAX_SMT_EXPR_DEPTH` because an unbounded recursive walk overflows the stack
/// and aborts the process. These helpers run on solver output rather than on
/// the checked property, so they carry their own bound rather than relying on
/// that one.
#[cfg(feature = "smt")]
const MAX_MODEL_VALUE_DEPTH: usize = 64;

/// Bound on the raw model-value string this parser will read.
///
/// Reading a decimal exactly is quadratic in its digit count, and the guard
/// runs AFTER `check_sat`, so `tlimit-per` does not bound it: cvc5's budget
/// covers the solve, not the revalidation. A property built by chained
/// squaring makes each witness roughly twice as wide as the last, so a few
/// levels reach tens of thousands of digits and the guard costs more than the
/// solve it is checking. Past this width leg 2 abstains, which is the same
/// answer it already gives for a value form it cannot parse.
#[cfg(feature = "smt")]
const MAX_MODEL_VALUE_CHARS: usize = 4096;

/// Exactly evaluate a boolean-shaped [`SmtExpr`] under a model.
///
/// `None` means "cannot decide exactly", never "false". Connectives use
/// three-valued (Kleene) semantics so a decided operand can still settle the
/// result: `false && undecided` is `false`, `true || undecided` is `true`.
#[cfg(feature = "smt")]
fn eval_exact_bool(
    expr: &SmtExpr,
    env: &std::collections::BTreeMap<String, ExactValue>,
    depth: usize,
) -> Option<bool> {
    if depth > MAX_MODEL_VALUE_DEPTH {
        return None;
    }
    match expr {
        SmtExpr::BoolLit(value) => Some(*value),
        SmtExpr::Var(name) => match env.get(name) {
            Some(ExactValue::Bool(value)) => Some(*value),
            _ => None,
        },
        SmtExpr::Not(inner) => eval_exact_bool(inner, env, depth + 1).map(|value| !value),
        SmtExpr::Cmp(op, left, right) => {
            let left = eval_exact_num(left, env, depth + 1)?;
            let right = eval_exact_num(right, env, depth + 1)?;
            Some(match op {
                CmpOp::Lt => left < right,
                CmpOp::Le => left <= right,
                CmpOp::Gt => left > right,
                CmpOp::Ge => left >= right,
                CmpOp::Eq => left == right,
                CmpOp::Ne => left != right,
            })
        }
        SmtExpr::Bool(op, operands) => {
            let evaluated: Vec<Option<bool>> = operands
                .iter()
                .map(|operand| eval_exact_bool(operand, env, depth + 1))
                .collect();
            match op {
                BoolOp::And => kleene_and(evaluated.into_iter()),
                BoolOp::Or => kleene_or(evaluated.into_iter()),
                BoolOp::Implies => {
                    // `a => b` is `!a || b`; anything else is not a shape this
                    // evaluator claims to decide.
                    let [antecedent, consequent] = evaluated.as_slice() else {
                        return None;
                    };
                    kleene_or([antecedent.map(|value| !value), *consequent].into_iter())
                }
            }
        }
        SmtExpr::Ite(condition, then_branch, else_branch) => {
            match eval_exact_bool(condition, env, depth + 1)? {
                true => eval_exact_bool(then_branch, env, depth + 1),
                false => eval_exact_bool(else_branch, env, depth + 1),
            }
        }
        // A quantifier or an uninterpreted application is not something this
        // evaluator decides. Abstain rather than reading it as false.
        SmtExpr::Forall(_, _)
        | SmtExpr::Exists(_, _)
        | SmtExpr::Apply(_, _)
        | SmtExpr::RealLit(_)
        | SmtExpr::IntLit(_)
        | SmtExpr::Arith(_, _, _) => None,
    }
}

#[cfg(feature = "smt")]
fn kleene_and(mut values: impl Iterator<Item = Option<bool>>) -> Option<bool> {
    let mut undecided = false;
    let decided_false = values.any(|value| match value {
        Some(false) => true,
        Some(true) => false,
        None => {
            undecided = true;
            false
        }
    });
    if decided_false {
        return Some(false);
    }
    if undecided { None } else { Some(true) }
}

#[cfg(feature = "smt")]
fn kleene_or(mut values: impl Iterator<Item = Option<bool>>) -> Option<bool> {
    let mut undecided = false;
    let decided_true = values.any(|value| match value {
        Some(true) => true,
        Some(false) => false,
        None => {
            undecided = true;
            false
        }
    });
    if decided_true {
        return Some(true);
    }
    if undecided { None } else { Some(false) }
}

/// Exactly evaluate a numeric [`SmtExpr`] under a model.
///
/// A `RealLit` is converted the same way the lowering converts it for cvc5
/// (`BigRational::from_float`), so the two sides reason about the identical
/// number rather than about the literal's decimal spelling.
#[cfg(feature = "smt")]
fn eval_exact_num(
    expr: &SmtExpr,
    env: &std::collections::BTreeMap<String, ExactValue>,
    depth: usize,
) -> Option<num_rational::BigRational> {
    use num_rational::BigRational;

    if depth > MAX_MODEL_VALUE_DEPTH {
        return None;
    }
    match expr {
        SmtExpr::Var(name) => match env.get(name) {
            Some(ExactValue::Num(value)) => Some(value.clone()),
            _ => None,
        },
        SmtExpr::RealLit(value) => BigRational::from_float(*value),
        SmtExpr::IntLit(value) => exact_from_decimal(&value.to_string()),
        SmtExpr::Arith(op, left, right) => {
            let left = eval_exact_num(left, env, depth + 1)?;
            if matches!(op, ArithOp::Neg) {
                return Some(-left);
            }
            let right = eval_exact_num(right, env, depth + 1)?;
            Some(match op {
                ArithOp::Add => left + right,
                ArithOp::Sub => left - right,
                ArithOp::Mul => left * right,
                ArithOp::Div => {
                    if right == exact_zero() {
                        return None;
                    }
                    left / right
                }
                ArithOp::Neg => unreachable!("handled above"),
            })
        }
        SmtExpr::Ite(condition, then_branch, else_branch) => {
            match eval_exact_bool(condition, env, depth + 1)? {
                true => eval_exact_num(then_branch, env, depth + 1),
                false => eval_exact_num(else_branch, env, depth + 1),
            }
        }
        SmtExpr::BoolLit(_)
        | SmtExpr::Not(_)
        | SmtExpr::Cmp(_, _, _)
        | SmtExpr::Bool(_, _)
        | SmtExpr::Forall(_, _)
        | SmtExpr::Exists(_, _)
        | SmtExpr::Apply(_, _) => None,
    }
}

/// Parse one SMT-LIB model value into an exact value.
///
/// cvc5 renders model values as SMT-LIB terms rather than plain numerals:
/// negatives are `(- 4.0)`, exact rationals are `(/ 3.0 2.0)`, and an
/// irrational witness comes back as `(_ real_algebraic_number <...>)`. Anything
/// this function does not recognise yields `None`, which makes the caller
/// abstain rather than reject.
#[cfg(feature = "smt")]
fn parse_smt_model_value(raw: &str, sort: SmtSort) -> Option<ExactValue> {
    match sort {
        SmtSort::Bool => match raw.trim() {
            "true" => Some(ExactValue::Bool(true)),
            "false" => Some(ExactValue::Bool(false)),
            _ => None,
        },
        SmtSort::Int | SmtSort::Real => parse_smt_rational(raw, 0).map(ExactValue::Num),
    }
}

/// Evaluate an SMT-LIB numeric model term to an exact rational.
///
/// Decimals are read digit-wise rather than through `f64`, so a value cvc5
/// chose because it is the least integer above a bound does not collapse onto
/// that bound on the way back in.
#[cfg(feature = "smt")]
fn parse_smt_rational(raw: &str, depth: usize) -> Option<num_rational::BigRational> {
    if depth > MAX_MODEL_VALUE_DEPTH || raw.len() > MAX_MODEL_VALUE_CHARS {
        return None;
    }
    let text = raw.trim();
    if let Some(inner) = text.strip_prefix('(').and_then(|t| t.strip_suffix(')')) {
        let inner = inner.trim();
        if let Some(rest) = inner.strip_prefix("- ") {
            // Unary negation only. A binary `(- a b)` has a second top-level
            // operand and is not a shape this parser claims.
            if split_smt_operands(rest.trim()).is_some() {
                return None;
            }
            return Some(-parse_smt_rational(rest, depth + 1)?);
        }
        if let Some(rest) = inner.strip_prefix("/ ") {
            let (numerator, denominator) = split_smt_operands(rest.trim())?;
            let denominator = parse_smt_rational(&denominator, depth + 1)?;
            if denominator == exact_zero() {
                return None;
            }
            return Some(parse_smt_rational(&numerator, depth + 1)? / denominator);
        }
        return None;
    }
    if text.is_empty() {
        return None;
    }
    exact_from_decimal(text)
}

/// Exact zero, built without a direct `num-bigint` dependency edge.
#[cfg(feature = "smt")]
fn exact_zero() -> num_rational::BigRational {
    use std::str::FromStr;
    num_rational::BigRational::from_str("0").expect("`0` is a valid rational")
}

/// Read a bare numeral or decimal exactly, digit-wise.
///
/// `"100000000000000001.0"` must come back as that integer, not as the `f64`
/// it rounds to. Parsing goes through `BigRational`'s `FromStr` over the digit
/// string and an exact power of ten, so no `f64` is involved at any point.
#[cfg(feature = "smt")]
fn exact_from_decimal(text: &str) -> Option<num_rational::BigRational> {
    use num_rational::BigRational;
    use std::str::FromStr;

    let (negative, digits) = match text.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let (integer_part, fraction_part) = match digits.split_once('.') {
        Some((integer_part, fraction_part)) => (integer_part, fraction_part),
        None => (digits, ""),
    };
    if integer_part.is_empty() && fraction_part.is_empty() {
        return None;
    }
    if !integer_part.bytes().all(|b| b.is_ascii_digit())
        || !fraction_part.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    let combined = format!("{integer_part}{fraction_part}");
    let numerator = BigRational::from_str(&combined).ok()?;
    let ten = BigRational::from_str("10").ok()?;
    let mut denominator = BigRational::from_str("1").ok()?;
    for _ in 0..fraction_part.len() {
        denominator *= ten.clone();
    }
    let value = numerator / denominator;
    Some(if negative { -value } else { value })
}

/// Split `"a b"` into its two top-level operands, respecting nesting.
#[cfg(feature = "smt")]
fn split_smt_operands(text: &str) -> Option<(String, String)> {
    let mut depth = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => depth = depth.checked_sub(1)?,
            ' ' if depth == 0 => {
                let (left, right) = text.split_at(index);
                return Some((left.to_string(), right.trim_start().to_string()));
            }
            _ => {}
        }
    }
    None
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
///
/// This public, context-free entry point deliberately rejects `sqrt`: only
/// [`solve_property_cvc5`] can supply the exact argument authorization created
/// by its domain preflight.
#[cfg(feature = "smt")]
pub fn lower_to_cvc5(
    tm: &cvc5_rs::TermManager,
    expr: &SmtExpr,
    vars: &std::collections::BTreeMap<String, cvc5_rs::Term>,
    sorts: &chelis_unord::UnordMap<String, SmtSort>,
) -> Result<(cvc5_rs::Term, SmtSort), String> {
    lower_to_cvc5_with_sqrt_domains(tm, expr, vars, sorts, &[])
}

/// Internal lowering entry with an exact set of `sqrt` arguments whose
/// non-negativity has already been proved by [`authorize_sqrt_domains`].
#[cfg(feature = "smt")]
fn lower_to_cvc5_with_sqrt_domains(
    tm: &cvc5_rs::TermManager,
    expr: &SmtExpr,
    vars: &std::collections::BTreeMap<String, cvc5_rs::Term>,
    sorts: &chelis_unord::UnordMap<String, SmtSort>,
    authorized_sqrt_arguments: &[SmtExpr],
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
            // SOUNDNESS: lower to the literal's EXACT f64 VALUE, not its decimal
            // spelling. `format!("{value}")` renders the shortest decimal that
            // round-trips (e.g. "0.1"), which cvc5 then parses as the EXACT
            // DECIMAL 1/10 -- a different number from the `f64` the program
            // actually runs (`0.1_f64` == 0.1000000000000000055...). Reasoning
            // over the exact decimal let cvc5 PROVE float goals that are FALSE
            // at runtime (e.g. `0.1 + 0.2 == 0.3` holds in exact reals but
            // `0.1_f64 + 0.2_f64 == 0.30000000000000004 != 0.3`), diverging
            // from the concrete f64 evaluator (`concrete_eval`, the ground
            // truth). Render the f64 as its exact rational `numerator/
            // denominator` (cvc5 parses "n/d" as an exact rational), the same
            // value the runtime evaluator uses. `from_float` returns `Some` for
            // every finite f64 (only `None` on inf/NaN, already rejected
            // above).
            //
            // KNOWN LIMITATION (not closed here): this fixes only the LITERAL
            // representation. cvc5 still performs EXACT-RATIONAL arithmetic over
            // these f64-valued literals; it does NOT model the IEEE-754
            // rounding of each `+`/`-`/`*`/`/` the runtime applies. A goal whose
            // truth depends on operation rounding (not just literal value) can
            // still diverge from the f64 runtime. Full FP-rounding soundness is
            // a separate, larger problem (a bit-precise float theory); this
            // change only makes each literal match runtime, closing the
            // exact-decimal-literal divergence.
            let exact = num_rational::BigRational::from_float(*value).ok_or_else(|| {
                format!("real literal `{value}` has no exact rational (routes to Tier C)")
            })?;
            let rational_str = format!("{}/{}", exact.numer(), exact.denom());
            (tm.mk_real_from_str(&rational_str), SmtSort::Real)
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
            let (l, ls) =
                lower_to_cvc5_with_sqrt_domains(tm, left, vars, sorts, authorized_sqrt_arguments)?;
            require_numeric_sort(ls, "neg operand")?;
            (tm.mk_term(Kind::CVC5_KIND_NEG, &[l]), ls)
        }
        SmtExpr::Arith(op, left, right) => {
            let (l, ls) =
                lower_to_cvc5_with_sqrt_domains(tm, left, vars, sorts, authorized_sqrt_arguments)?;
            let (r, rs) =
                lower_to_cvc5_with_sqrt_domains(tm, right, vars, sorts, authorized_sqrt_arguments)?;
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
            let (l, ls) =
                lower_to_cvc5_with_sqrt_domains(tm, left, vars, sorts, authorized_sqrt_arguments)?;
            let (r, rs) =
                lower_to_cvc5_with_sqrt_domains(tm, right, vars, sorts, authorized_sqrt_arguments)?;
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
                .map(|c| {
                    lower_to_cvc5_with_sqrt_domains(tm, c, vars, sorts, authorized_sqrt_arguments)
                })
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
            let (t, s) =
                lower_to_cvc5_with_sqrt_domains(tm, inner, vars, sorts, authorized_sqrt_arguments)?;
            if s != SmtSort::Bool {
                return Err(format!(
                    "`not` operand has sort {s:?}, expected Bool (routes to Tier C)"
                ));
            }
            (tm.mk_term(Kind::CVC5_KIND_NOT, &[t]), SmtSort::Bool)
        }
        SmtExpr::Forall(bindings, body) | SmtExpr::Exists(bindings, body) => {
            for (name, _) in bindings {
                if !smt_name_is_cvc5_safe(name) {
                    return Err(
                        "quantifier bound-variable name contains an interior NUL byte (routes to Tier C)"
                            .to_string(),
                    );
                }
            }
            let bound_vars = quantifier_bound_vars(tm, bindings);
            let mut extended_vars = vars.clone();
            let mut extended_sorts = sorts.clone();
            for (i, (name, sort)) in bindings.iter().enumerate() {
                extended_vars.insert(name.clone(), bound_vars[i].clone());
                extended_sorts.insert(name.clone(), *sort);
            }
            let (body_term, body_sort) = lower_to_cvc5_with_sqrt_domains(
                tm,
                body,
                &extended_vars,
                &extended_sorts,
                authorized_sqrt_arguments,
            )?;
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
            let arity = cvc5_lowerable_arity(name).ok_or_else(|| unsupported_apply_reason(name))?;
            if args.len() != arity {
                return Err(format!(
                    "intrinsic `{name}` expects {arity} argument(s), got {} (routes to Tier C)",
                    args.len()
                ));
            }
            if name == "sqrt"
                && !authorized_sqrt_arguments
                    .iter()
                    .any(|argument| argument == &args[0])
            {
                return Err(sqrt_domain_error(
                    "this exact argument was not authorized by the domain preflight",
                ));
            }
            let lowered: Vec<(cvc5_rs::Term, SmtSort)> = args
                .iter()
                .map(|a| {
                    lower_to_cvc5_with_sqrt_domains(tm, a, vars, sorts, authorized_sqrt_arguments)
                })
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
                    return Err(unsupported_apply_reason(other));
                }
            }
        }
        SmtExpr::Ite(cond, then_expr, else_expr) => {
            let (c, cs) =
                lower_to_cvc5_with_sqrt_domains(tm, cond, vars, sorts, authorized_sqrt_arguments)?;
            if cs != SmtSort::Bool {
                return Err(format!(
                    "if-then-else condition has sort {cs:?}, expected Bool (routes to Tier C)"
                ));
            }
            let (t, ts) = lower_to_cvc5_with_sqrt_domains(
                tm,
                then_expr,
                vars,
                sorts,
                authorized_sqrt_arguments,
            )?;
            let (e, es) = lower_to_cvc5_with_sqrt_domains(
                tm,
                else_expr,
                vars,
                sorts,
                authorized_sqrt_arguments,
            )?;
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
    fn issue1475_only_a_proved_domain_obligation_authorizes_sqrt() {
        assert_eq!(require_proved_sqrt_domain(TierBResult::Proved), Ok(()));
        for result in [
            TierBResult::Disproved(serde_json::json!({})),
            TierBResult::Timeout,
            TierBResult::Unknown,
            TierBResult::Error("planted lowering failure".to_string()),
        ] {
            let reason = require_proved_sqrt_domain(result)
                .expect_err("every non-proof domain result must fail closed");
            assert!(reason.contains("sqrt"), "reason names sqrt: {reason}");
            assert!(
                reason.contains("Tier C"),
                "reason names the safe fallback: {reason}"
            );
        }
    }

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
    fn non_vacuity_sat_assumptions_are_established_with_a_model() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            )],
            postcondition: SmtExpr::BoolLit(true),
        };
        let result = check_assumptions_satisfiable(&prop, 5000);
        match result {
            AssumptionSatisfiability::Sat(model) => assert!(model.get("x").is_some()),
            other => panic!("expected satisfiable assumptions, got {other:?}"),
        }
    }

    #[test]
    fn non_vacuity_unsat_assumptions_are_invalid_not_green() {
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![
                SmtExpr::Cmp(
                    CmpOp::Gt,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(0.0)),
                ),
                SmtExpr::Cmp(
                    CmpOp::Lt,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(0.0)),
                ),
            ],
            postcondition: SmtExpr::BoolLit(true),
        };
        let result = check_assumptions_satisfiable(&prop, 5000);
        assert_eq!(result, AssumptionSatisfiability::Unsat);
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

    // chelis#434: a transcendental that is a legitimate predicate intrinsic
    // (in chelis_pred::TRANSCENDENTAL_WHITELIST) but has no cvc5 kind (today
    // `log`) must produce an HONEST capability-boundary reason -- it names the
    // function, says "transcendental ... not supported by the SMT tier", says
    // the goal is Unsupported, and cites the WS-7/Beacon faithful-discharge
    // seam -- NOT the generic internal-looking "unsupported function" text.
    #[test]
    fn issue434_transcendental_log_reason_is_an_honest_capability_boundary() {
        let reason = unsupported_apply_reason("log");
        assert!(reason.contains("log"), "names the fn: {reason}");
        assert!(
            reason.contains("transcendental"),
            "frames it as a transcendental: {reason}"
        );
        assert!(
            reason.contains("not supported by the SMT tier"),
            "states the capability boundary: {reason}"
        );
        assert!(
            reason.contains("Unsupported"),
            "states the goal is Unsupported, not broken: {reason}"
        );
        assert!(
            reason.contains("WS-7") && reason.contains("Beacon"),
            "cites the faithful-discharge seam: {reason}"
        );
    }

    // Negative parity: a truly-unknown symbol (NOT a whitelisted
    // transcendental) keeps the generic "unsupported function" reason -- the
    // honest-transcendental framing must NOT launder an arbitrary unknown
    // callee into a "transcendental" capability boundary.
    #[test]
    fn issue434_unknown_symbol_keeps_generic_unsupported_reason() {
        let reason = unsupported_apply_reason("mystery_fn");
        assert!(reason.contains("mystery_fn"), "names the fn: {reason}");
        assert!(
            reason.contains("unsupported function"),
            "generic unsupported-function reason: {reason}"
        );
        assert!(
            !reason.contains("transcendental"),
            "must NOT mislabel an unknown symbol as a transcendental: {reason}"
        );
    }

    // Every whitelisted-but-not-cvc5-lowerable transcendental (the set
    // difference TRANSCENDENTAL_WHITELIST minus CVC5_LOWERABLE) gets the
    // honest capability-boundary reason -- so adding e.g. `tan`/`atan` to the
    // predicate grammar later cannot silently regress to the generic text.
    #[test]
    fn issue434_all_non_lowerable_transcendentals_are_honest() {
        let non_lowerable: Vec<&&str> = chelis_pred::TRANSCENDENTAL_WHITELIST
            .iter()
            .filter(|name| cvc5_lowerable_arity(name).is_none())
            .collect();
        // Today this is exactly {log}; the assertion locks the invariant, not
        // the cardinality, so it survives whitelist growth.
        assert!(
            non_lowerable.iter().any(|n| ***n == *"log"),
            "log is the known non-lowerable transcendental"
        );
        for name in non_lowerable {
            let reason = unsupported_apply_reason(name);
            assert!(
                reason.contains("transcendental")
                    && reason.contains("not supported by the SMT tier")
                    && reason.contains("Unsupported"),
                "transcendental `{name}` must get the honest reason: {reason}"
            );
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

    /// A variable / bound-variable name with an interior NUL byte. cvc5-rs
    /// builds a CString from the name and unwraps it, so an interior NUL
    /// PANICS out of solve_property (and aborts under panic=abort) rather
    /// than returning a TierBResult. The name guard routes it to a clean
    /// Error instead. RT6 round-2.
    #[test]
    fn rt6_nul_byte_variable_name_is_clean_error_not_panic() {
        // Declared variable name with an interior NUL.
        let declared = SmtProperty {
            variables: vec![("x\0y".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Var("x\0y".to_string())),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        assert!(
            matches!(solve_property(&declared, 5000), TierBResult::Error(_)),
            "a NUL-containing declared variable name must be a clean Error, not a panic"
        );

        // Quantifier bound-variable name with an interior NUL.
        let bound = SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: SmtExpr::Forall(
                vec![("k\0z".to_string(), SmtSort::Real)],
                Box::new(SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Var("k\0z".to_string())),
                    Box::new(SmtExpr::RealLit(0.0)),
                )),
            ),
        };
        assert!(
            matches!(solve_property(&bound, 5000), TierBResult::Error(_)),
            "a NUL-containing bound-variable name must be a clean Error, not a panic"
        );
    }

    // ---- chelis#1224: a counterexample must satisfy the stated assumptions ----

    fn exact(text: &str) -> num_rational::BigRational {
        parse_smt_rational(text, 0).expect("test value parses")
    }

    /// The property from the observed failure: `d > 0.5, r > g, r < 9.5`.
    /// Only the assumptions matter here; the postcondition is never consulted
    /// by the validator.
    fn quotient_grad_property() -> SmtProperty {
        let var = |name: &str| Box::new(SmtExpr::Var(name.to_string()));
        SmtProperty {
            variables: vec![
                ("d".to_string(), SmtSort::Real),
                ("r".to_string(), SmtSort::Real),
                ("g".to_string(), SmtSort::Real),
            ],
            preconditions: vec![
                SmtExpr::Cmp(CmpOp::Gt, var("d"), Box::new(SmtExpr::RealLit(0.5))),
                SmtExpr::Cmp(CmpOp::Gt, var("r"), var("g")),
                SmtExpr::Cmp(CmpOp::Lt, var("r"), Box::new(SmtExpr::RealLit(9.5))),
            ],
            postcondition: SmtExpr::BoolLit(true),
        }
    }

    fn model(pairs: &[(&str, &str)]) -> serde_json::Map<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), Value::String((*v).to_string())))
            .collect()
    }

    #[test]
    fn model_violating_a_precondition_is_rejected() {
        // The exact model CI run 31554449486 reported as a counterexample.
        // `d = -4.0` violates `d > 0.5`, so this is a counterexample to the
        // UNGUARDED goal and must never be reported as a disproof.
        let reason = validate_model_independently(
            &quotient_grad_property(),
            &model(&[("d", "(- 4.0)"), ("r", "4.0"), ("g", "2.0")]),
        )
        .expect_err("a precondition-violating model must be rejected");
        assert!(
            reason.contains("exactly violates precondition 0"),
            "the diagnostic must name which assumption failed: {reason}"
        );
    }

    #[test]
    fn model_satisfying_every_precondition_is_accepted() {
        // Negative parity: the guard must not reject legitimate counterexamples,
        // or every disproof would degrade to Tier C. This is the model the
        // named-`def` polarity probe returns.
        assert!(
            validate_model_independently(
                &quotient_grad_property(),
                &model(&[("d", "2.0"), ("r", "5.0"), ("g", "1.0")]),
            )
            .is_ok(),
            "a model satisfying every stated assumption is a valid counterexample"
        );
    }

    /// The regression that made the first version of this guard unsound in the
    /// rejecting direction: cvc5 decides in exact rationals, so it answers
    /// `x > 1e17` with the least integer above the bound. Read back through
    /// `f64` that value collapses onto the bound and the disproof was thrown
    /// away. Every strict bound past 2^53 was affected.
    #[test]
    fn witnesses_beyond_f64_precision_are_still_accepted() {
        let above = |bound: f64, witness: &str| {
            let property = SmtProperty {
                variables: vec![("x".to_string(), SmtSort::Real)],
                preconditions: vec![SmtExpr::Cmp(
                    CmpOp::Gt,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(bound)),
                )],
                postcondition: SmtExpr::BoolLit(true),
            };
            validate_model_independently(&property, &model(&[("x", witness)]))
        };
        assert!(above(1e17, "100000000000000001.0").is_ok(), "2^53 boundary");
        assert!(above(9.1e15, "9100000000000001.0").is_ok());
        assert!(
            above(1e300, "1000000000000000052504760255204420248704468581108159154915854115511802457988908195786371375080447864043704443832883878176942523235360430575644792184786706982848387200926575803737830233794788090059368953234970799945081119038967640880074652742780142494579258788820056842838115669472135067360170731089224008034192946103522494276477283076570639713657624277070295118594349217453200873511175487.0").is_ok(),
            "1e300 witness"
        );
        // A witness that genuinely violates the bound is still rejected, at any
        // magnitude: the fix must not turn the guard off.
        assert!(
            above(1e17, "100000000000000000.0").is_err(),
            "equal is not >"
        );
    }

    #[test]
    fn integer_witnesses_beyond_i64_are_not_truncated() {
        let property = SmtProperty {
            variables: vec![("n".to_string(), SmtSort::Int)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("n".to_string())),
                Box::new(SmtExpr::RealLit(1.6e19)),
            )],
            postcondition: SmtExpr::BoolLit(true),
        };
        assert!(
            validate_model_independently(&property, &model(&[("n", "16000000000000000001")]))
                .is_ok(),
            "an integer past i64::MAX must not saturate into a false rejection"
        );
    }

    #[test]
    fn an_undecidable_precondition_abstains_rather_than_rejecting() {
        // A quantifier is not something this evaluator decides. Reading it as
        // `false` would discard every counterexample of any property whose
        // assumptions mention one.
        let property = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Forall(
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
            )],
            postcondition: SmtExpr::BoolLit(true),
        };
        assert!(
            validate_model_independently(&property, &model(&[("x", "0.0")])).is_ok(),
            "an undecidable assumption must abstain, not reject"
        );
    }

    #[test]
    fn an_unreadable_model_abstains_rather_than_rejecting() {
        // Both an absent value and a value form the parser does not recognise
        // leave the guard with no verdict. cvc5 emits an irrational witness as
        // `(_ real_algebraic_number <...>)`; that is honestly unusable here,
        // but discarding the disproof over it would be a false rejection.
        assert!(
            validate_model_independently(
                &quotient_grad_property(),
                &model(&[("d", "2.0"), ("r", "5.0")]),
            )
            .is_ok(),
            "a model missing a declared variable cannot be revalidated"
        );
        assert!(
            validate_model_independently(
                &quotient_grad_property(),
                &model(&[
                    ("d", "(_ real_algebraic_number <1*x^2 + (-2), (5/4, 3/2)>)"),
                    ("r", "5.0"),
                    ("g", "1.0"),
                ]),
            )
            .is_ok(),
            "an algebraic-number witness cannot be revalidated"
        );
    }

    #[test]
    fn kleene_connectives_settle_on_a_decided_operand() {
        // `false && undecided` is false, so a decided violation inside a
        // conjunction still rejects; `true || undecided` is true.
        let undecidable = SmtExpr::Apply("mystery".to_string(), vec![]);
        let property = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Bool(
                BoolOp::And,
                vec![
                    SmtExpr::Cmp(
                        CmpOp::Gt,
                        Box::new(SmtExpr::Var("x".to_string())),
                        Box::new(SmtExpr::RealLit(10.0)),
                    ),
                    undecidable.clone(),
                ],
            )],
            postcondition: SmtExpr::BoolLit(true),
        };
        assert!(
            validate_model_independently(&property, &model(&[("x", "0.0")])).is_err(),
            "a decided-false conjunct settles the conjunction"
        );
        let disjunction = SmtProperty {
            preconditions: vec![SmtExpr::Bool(
                BoolOp::Or,
                vec![
                    SmtExpr::Cmp(
                        CmpOp::Lt,
                        Box::new(SmtExpr::Var("x".to_string())),
                        Box::new(SmtExpr::RealLit(10.0)),
                    ),
                    undecidable,
                ],
            )],
            ..property
        };
        assert!(
            validate_model_independently(&disjunction, &model(&[("x", "0.0")])).is_ok(),
            "a decided-true disjunct settles the disjunction"
        );
    }

    #[test]
    fn smt_model_values_parse_exactly() {
        assert_eq!(parse_smt_rational("4.0", 0), Some(exact("4")));
        assert_eq!(parse_smt_rational("(- 4.0)", 0), Some(-exact("4")));
        assert_eq!(
            parse_smt_rational("(/ 3.0 2.0)", 0),
            Some(exact("3") / exact("2"))
        );
        assert_eq!(
            parse_smt_rational("(- (/ 3.0 2.0))", 0),
            Some(-(exact("3") / exact("2")))
        );
        assert_eq!(
            parse_smt_rational("(/ (- 3.0) 2.0)", 0),
            Some(-(exact("3") / exact("2")))
        );
        // Exactness past f64: this must NOT equal 1e17.
        assert_ne!(
            parse_smt_rational("100000000000000001.0", 0),
            parse_smt_rational("100000000000000000.0", 0),
            "a decimal must not round through f64"
        );
        // Fail closed on division by zero, binary minus, and unrecognised forms.
        assert_eq!(parse_smt_rational("(/ 1.0 0.0)", 0), None);
        assert_eq!(
            parse_smt_rational("(- 1 2)", 0),
            None,
            "binary minus is not unary negation"
        );
        assert_eq!(parse_smt_rational("(_ real_algebraic_number <x>)", 0), None);
        assert_eq!(parse_smt_rational("(- )", 0), None);
        assert_eq!(parse_smt_rational("", 0), None);
        assert_eq!(
            parse_smt_rational("1e5", 0),
            None,
            "exponent notation is not claimed"
        );
        assert_eq!(parse_smt_rational("0x10", 0), None);
    }

    #[test]
    fn deeply_nested_model_values_are_bounded_not_a_stack_overflow() {
        // The value comes from the solver rather than from the checked
        // property, so it carries its own depth bound: an unbounded walk here
        // would abort the process the way `property_exceeds_smt_depth` exists
        // to prevent.
        let deep = format!(
            "{}1.0{}",
            "(- ".repeat(MAX_MODEL_VALUE_DEPTH + 10),
            ")".repeat(MAX_MODEL_VALUE_DEPTH + 10)
        );
        assert_eq!(
            parse_smt_rational(&deep, 0),
            None,
            "past the bound, abstain"
        );
        let shallow = format!("{}1.0{}", "(- ".repeat(4), ")".repeat(4));
        assert_eq!(parse_smt_rational(&shallow, 0), Some(exact("1")));
    }

    #[test]
    fn parsed_model_values_carry_their_declared_sort() {
        assert_eq!(
            parse_smt_model_value("(- 4.0)", SmtSort::Real),
            Some(ExactValue::Num(-exact("4")))
        );
        assert_eq!(
            parse_smt_model_value("7", SmtSort::Int),
            Some(ExactValue::Num(exact("7")))
        );
        assert_eq!(
            parse_smt_model_value("true", SmtSort::Bool),
            Some(ExactValue::Bool(true))
        );
        assert_eq!(parse_smt_model_value("4.0", SmtSort::Bool), None);
    }

    #[test]
    fn a_real_disproof_still_reports_a_counterexample_end_to_end() {
        // The whole guard, through cvc5: a genuinely false property under its
        // own preconditions must still come back Disproved, not Error.
        let var = |name: &str| Box::new(SmtExpr::Var(name.to_string()));
        let prop = SmtProperty {
            variables: vec![
                ("d".to_string(), SmtSort::Real),
                ("r".to_string(), SmtSort::Real),
            ],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Gt,
                var("d"),
                Box::new(SmtExpr::RealLit(0.5)),
            )],
            // False under `d > 0.5`: nothing forces d < 0.
            postcondition: SmtExpr::Cmp(CmpOp::Lt, var("d"), Box::new(SmtExpr::RealLit(0.0))),
        };
        match solve_property(&prop, 5000) {
            TierBResult::Disproved(model) => {
                let raw = model["d"].as_str().expect("model carries d");
                let value = parse_smt_rational(raw, 0).expect("model value parses");
                assert!(
                    value > exact("1") / exact("2"),
                    "the reported counterexample must satisfy the stated precondition: {raw}"
                );
            }
            other => panic!("expected a counterexample, got {other:?}"),
        }
    }

    /// Round-2 F1: one unreadable variable must not switch off the whole leg.
    ///
    /// An algebraic-number witness for an unrelated variable is routine in the
    /// NRA logic this file selects. Before the fix, its presence made the
    /// independent leg return early, so a precondition over a perfectly
    /// readable variable went unchecked.
    #[test]
    fn an_unreadable_sibling_does_not_disable_the_readable_checks() {
        let property = SmtProperty {
            variables: vec![
                ("d".to_string(), SmtSort::Real),
                ("z".to_string(), SmtSort::Real),
            ],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("d".to_string())),
                Box::new(SmtExpr::RealLit(0.5)),
            )],
            postcondition: SmtExpr::BoolLit(true),
        };
        // `d = 0.0` violates `d > 0.5`, and it must still be caught even though
        // `z` came back as an algebraic number this parser cannot read.
        let reason = validate_model_independently(
            &property,
            &model(&[
                ("d", "0.0"),
                ("z", "(_ real_algebraic_number <1*x^2 + (-2), (5/4, 3/2)>)"),
            ]),
        )
        .expect_err("a readable violation must be caught beside an unreadable sibling");
        assert!(
            reason.contains("exactly violates precondition 0"),
            "{reason}"
        );
        // A precondition that mentions the unreadable variable still abstains.
        let over_z = SmtProperty {
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Var("z".to_string())),
                Box::new(SmtExpr::RealLit(1e9)),
            )],
            ..property
        };
        assert!(
            validate_model_independently(
                &over_z,
                &model(&[
                    ("d", "2.0"),
                    ("z", "(_ real_algebraic_number <1*x^2 + (-2), (5/4, 3/2)>)"),
                ]),
            )
            .is_ok(),
            "a precondition over the unreadable variable must abstain"
        );
    }

    /// Round-2 F2: reading a decimal exactly is quadratic, and the guard runs
    /// after `check_sat`, where cvc5's `tlimit-per` no longer applies. Past the
    /// width bound leg 2 abstains rather than spending unbounded time.
    #[test]
    fn an_oversized_model_value_abstains_rather_than_parsing() {
        let wide = format!("{}.0", "9".repeat(MAX_MODEL_VALUE_CHARS + 1));
        assert!(wide.len() > MAX_MODEL_VALUE_CHARS);
        assert_eq!(parse_smt_rational(&wide, 0), None, "past the width bound");
        let at_bound = "9".repeat(MAX_MODEL_VALUE_CHARS);
        assert!(
            parse_smt_rational(&at_bound, 0).is_some(),
            "a value at the bound is still read exactly"
        );
    }

    /// Round-1's ulp-tight shape: adjacent f64s, so every real strictly between
    /// them rounds to an endpoint that violates one bound. Distinct from the
    /// magnitude family - this is about interval width, not size.
    #[test]
    fn an_ulp_tight_interval_still_disproves() {
        let hi = f64::from_bits(1.0f64.to_bits() + 1);
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![
                SmtExpr::Cmp(
                    CmpOp::Gt,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(1.0)),
                ),
                SmtExpr::Cmp(
                    CmpOp::Lt,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(hi)),
                ),
            ],
            postcondition: SmtExpr::BoolLit(false),
        };
        assert!(
            matches!(solve_property(&prop, 5000), TierBResult::Disproved(_)),
            "a witness inside a one-ulp interval must survive the guard"
        );
    }

    /// A Bool-sorted variable read straight out of the model, and the
    /// empty-precondition no-op.
    #[test]
    fn bool_sorted_variables_and_empty_preconditions() {
        let property = SmtProperty {
            variables: vec![("b".to_string(), SmtSort::Bool)],
            preconditions: vec![SmtExpr::Var("b".to_string())],
            postcondition: SmtExpr::BoolLit(true),
        };
        assert!(
            validate_model_independently(&property, &model(&[("b", "false")])).is_err(),
            "a Bool precondition false under the model is a decided violation"
        );
        assert!(
            validate_model_independently(&property, &model(&[("b", "true")])).is_ok(),
            "a Bool precondition true under the model is satisfied"
        );
        let none = SmtProperty {
            preconditions: vec![],
            ..property
        };
        assert!(
            validate_model_independently(&none, &model(&[("b", "false")])).is_ok(),
            "no preconditions means nothing to violate"
        );
    }

    /// The measured chelis#1224 false-positive family, end to end through cvc5:
    /// a strict bound past 2^53 must still yield a counterexample.
    #[test]
    fn large_magnitude_disproofs_survive_the_guard_end_to_end() {
        for bound in [1e0, 1e12, 9.1e15, 1e17, 1e30] {
            let prop = SmtProperty {
                variables: vec![("x".to_string(), SmtSort::Real)],
                preconditions: vec![SmtExpr::Cmp(
                    CmpOp::Gt,
                    Box::new(SmtExpr::Var("x".to_string())),
                    Box::new(SmtExpr::RealLit(bound)),
                )],
                postcondition: SmtExpr::BoolLit(false),
            };
            assert!(
                matches!(solve_property(&prop, 5000), TierBResult::Disproved(_)),
                "x > {bound} must still disprove; the guard must not eat it"
            );
        }
    }
}
