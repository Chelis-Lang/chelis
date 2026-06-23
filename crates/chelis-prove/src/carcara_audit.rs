//! WI-16 Carcara auditability: independent in-process re-checking of cvc5's
//! Alethe proofs.
//!
//! Carcara is a *proof auditor*, not a goal-discharger: it does not decide a
//! [`crate::discharge::Goal`], it re-checks the Alethe proof cvc5 produced for
//! an already-`Proved` goal. So it attaches as a post-discharge AUDIT STEP
//! inside the cvc5 engine ([`crate::discharge::Cvc5Engine::discharge`]) rather
//! than as a separate [`crate::discharge::DischargeEngine`]. See the WI-16
//! integration-shape contract in
//! `spec/design/verification_stack_master_plan.md`.
//!
//! The audit covers exactly the fragment cvc5 emits an Alethe proof for, which
//! for the goals this crate lowers is equality-with-uninterpreted-functions and
//! linear arithmetic (the `SmtExpr`/`SmtSort` surface has no bit-vector
//! representation, so the bit-vector slice of Alethe is not reachable here). A
//! nonlinear-real `Proved` (the `QF_NRA`/`NRA` core) has no Alethe proof to
//! re-check, so the audit is reported ABSENT ([`CarcaraAudit::Unavailable`])
//! there, never as a failure.
//!
//! What Carcara actually re-checks vs trusts. Carcara verifies most of the
//! proof step by step (assume against the problem premises, resolution,
//! subproof discharge, congruence, transitivity, ...), but it accepts THREE
//! kinds of step as TRUSTED HOLES with zero checking (carcara
//! `checker/shared.rs`): `hole` (cvc5's `TRUST_THEORY_REWRITE` /
//! `ARITH_POLY_NORM_REL` trust steps), `lia_generic` (cvc5's linear-arithmetic
//! macro step), and any rule in the configured allowed-rules set (we add
//! `rare_rewrite`, cvc5's RARE/DSL rewrite leaf, because checking it needs
//! cvc5's compiled RARE database the in-process auditor lacks). A real cvc5
//! Alethe proof of even a simple linear goal contains `hole` and `lia_generic`
//! steps carrying genuine arithmetic content, so the audit's guarantee is "the
//! proof structure re-checks, MODULO these trusted leaves", not a full
//! independent re-derivation. To keep that honest, a proof that leans on any
//! trusted leaf is reported [`CarcaraAudit::ConfirmedModuloRewrites`] (carrying
//! the exact set and counts of trusted-rule kinds hit), distinct from a fully
//! hole-free [`CarcaraAudit::Confirmed`]. The trusted leaves are themselves
//! SOUND (genuine cvc5 inferences, matching cvc5's own trust posture); this is a
//! disclosure boundary, not an unsoundness.
//!
//! The audit result is folded into the discharge as an independent-audit
//! EVIDENCE DIMENSION; it never changes a discharge's soundness or qualifier.
//! Under the chelis#422 honesty taxonomy a cvc5 `Proved` is classified
//! `Soundness::SoundApproximate` carrying `Qualifier::RealArith` (a decision
//! over the reals, not exact machine arithmetic); the audit keeps that badge
//! unchanged. A re-check that FAILS is recorded as a [`CarcaraAudit::Failed`]
//! auditor
//! disagreement in the evidence and is never folded into a confirmed state.
//! NOTE: today this evidence dimension is WRITE-ONLY -- no production consumer
//! reads `evidence["carcara_audit"]`, and a `Failed` audit does not change the
//! discharge's `result()`/`soundness()`. Wiring a consumer that acts on (or
//! serializes) a `Failed` audit is a tracked follow-up; until then the audit is
//! an in-memory evidence dimension, not a surfaced gate.

use crate::solver::{ArithOp, BoolOp, CmpOp, SmtExpr, SmtSort};
use crate::tier_b::SmtProperty;

/// The Alethe step rules Carcara accepts as TRUSTED HOLES (zero checking), as
/// implemented in carcara `checker/shared.rs`:
///
/// - `hole`: a native always-valid hole; cvc5 emits its `TRUST_THEORY_REWRITE`
///   and `ARITH_POLY_NORM_REL` trust steps as `hole`.
/// - `lia_generic`: a native step Carcara explicitly ignores (logs a warning,
///   returns Ok); cvc5's linear-integer/real-arithmetic macro lemmas.
/// - `rare_rewrite`: NOT native-trusted; trusted only because we add it to the
///   checker's allowed-rules set ([`run_carcara_check`]), since checking it
///   needs cvc5's compiled RARE rewrite database the in-process auditor lacks.
///
/// This is the SINGLE SOURCE OF TRUTH for "what the audit trusts without
/// checking": the allowed-rules config, the proof scan that enumerates trusted
/// leaves, and the disclosure test all derive from it, so the three cannot
/// drift. `rare_rewrite` must be first because [`run_carcara_check`] also feeds
/// the allowlisted subset (currently just `rare_rewrite`) to Carcara's config.
#[cfg(feature = "carcara")]
const TRUSTED_HOLE_RULES: &[&str] = &["rare_rewrite", "hole", "lia_generic"];

/// The subset of [`TRUSTED_HOLE_RULES`] that is trusted only via the checker's
/// allowed-rules config (vs the rules Carcara treats as holes natively). `hole`
/// and `lia_generic` are native holes and must NOT be allowlisted (that would be
/// redundant and misleading); only `rare_rewrite` is ours to allow.
#[cfg(feature = "carcara")]
const ALLOWLISTED_HOLE_RULES: &[&str] = &["rare_rewrite"];

/// One trusted-rule kind and how many times it appeared in the audited proof.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TrustedRuleCount {
    /// The Alethe rule name (one of [`TRUSTED_HOLE_RULES`]).
    pub rule: String,
    /// How many steps in the proof used this rule.
    pub count: usize,
}

/// The outcome of routing a cvc5 Alethe proof through Carcara.
///
/// This is the evidence dimension folded into the cvc5 discharge. It is
/// deliberately a multi-way outcome, not a bool: "the proof was not audited"
/// (absent / unavailable) is a distinct, honest state from "the proof was
/// audited and disagreed" (a surfaced failure), and a fully re-checked proof is
/// distinguished from one re-checked modulo trusted leaves.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "detail")]
pub enum CarcaraAudit {
    /// Carcara independently re-checked cvc5's Alethe proof end to end with no
    /// holes: every step, including all rewrite leaves, was verified.
    Confirmed,
    /// Carcara re-checked the proof and CONFIRMED it, but the proof leans on one
    /// or more steps Carcara accepts as TRUSTED HOLES with zero checking. Three
    /// step kinds are trusted (carcara `checker/shared.rs`): `hole` (cvc5's
    /// `TRUST_THEORY_REWRITE` / `ARITH_POLY_NORM_REL` trust steps),
    /// `lia_generic` (cvc5's linear-arithmetic macro step), and `rare_rewrite`
    /// (cvc5's RARE/DSL rewrite leaf, allowlisted because checking it needs
    /// cvc5's compiled RARE database the in-process auditor lacks). Every OTHER
    /// step kind (assume, resolution, subproof, congruence, transitivity, ...)
    /// was checked. This is strictly weaker than [`CarcaraAudit::Confirmed`] and
    /// is its own state, never reported as a full check. The trusted leaves are
    /// sound cvc5 inferences (matching cvc5's own trust posture); this is a
    /// disclosure boundary, not an unsoundness. Carries the exact set of
    /// trusted-rule kinds that appeared in the proof, each with its count, so the
    /// disclosure reflects the actual proof and cannot silently drift.
    ConfirmedModuloRewrites(Vec<TrustedRuleCount>),
    /// Carcara re-checked the proof and it FAILED (the proof does not check, or
    /// the problem/proof did not parse). A cvc5-vs-auditor disagreement, recorded
    /// in the evidence and never folded into a confirmed state. Carries the
    /// auditor's reason. (NB: this evidence is write-only today -- no production
    /// consumer reads it yet; see the module docs.)
    Failed(String),
    /// No Alethe proof was available to audit (e.g. cvc5 produced none for this
    /// logic fragment, such as the nonlinear-real core). The cvc5 result is
    /// unchanged; the audit dimension is simply absent. Carries why.
    Unavailable(String),
}

impl CarcaraAudit {
    /// Whether Carcara re-verified the proof (fully, or modulo trusted rewrite
    /// leaves). Both are an independent confirmation that the proof structure
    /// re-checks; they differ only in whether the rewrite leaves were verified
    /// or trusted.
    pub fn is_confirmed(&self) -> bool {
        matches!(
            self,
            CarcaraAudit::Confirmed | CarcaraAudit::ConfirmedModuloRewrites(_)
        )
    }

    /// Whether the auditor DISAGREED with cvc5 (a surfaced failure). This is the
    /// signal a consumer must never ignore.
    pub fn is_failed(&self) -> bool {
        matches!(self, CarcaraAudit::Failed(_))
    }
}

/// Render the SMT-LIB-2 sort keyword for an [`SmtSort`]. Mirrors the cvc5 sort
/// selection in `tier_b::lower_to_cvc5` so the rendered problem declares each
/// symbol with the SAME sort cvc5 used.
fn smtlib_sort(sort: SmtSort) -> &'static str {
    match sort {
        SmtSort::Real => "Real",
        SmtSort::Int => "Int",
        SmtSort::Bool => "Bool",
    }
}

/// Whether the rendered SMT-LIB problem can faithfully represent `expr`.
///
/// The renderer covers exactly the linear / EUF / Boolean fragment Carcara
/// audits. A transcendental or nonlinear-real construct (which lives in the
/// `NRA`/`NRAT` core cvc5 has no Alethe proof for anyway) is reported as
/// out-of-fragment so the audit reports [`CarcaraAudit::Unavailable`] rather
/// than rendering an SMT-LIB problem the proof was never about. Computed
/// iteratively so a deep tree cannot overflow the auditor's stack.
fn expr_in_audit_fragment(expr: &SmtExpr) -> Result<(), String> {
    let mut stack = vec![expr];
    while let Some(node) = stack.pop() {
        match node {
            SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {}
            SmtExpr::Arith(ArithOp::Mul, l, r) => {
                // A product of two non-constant terms is nonlinear; that lives
                // in the NRA core cvc5 emits no Alethe proof for. A literal
                // coefficient (`c * x`) stays linear.
                if !is_numeric_literal(l) && !is_numeric_literal(r) {
                    return Err("nonlinear multiplication is outside the Carcara-audited \
                                (linear / EUF) fragment"
                        .to_string());
                }
                stack.push(l);
                stack.push(r);
            }
            SmtExpr::Arith(ArithOp::Div, l, r) => {
                // Division by a non-literal is nonlinear.
                if !is_numeric_literal(r) {
                    return Err("division by a non-constant is outside the Carcara-audited \
                                fragment"
                        .to_string());
                }
                stack.push(l);
                stack.push(r);
            }
            SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
                stack.push(l);
                stack.push(r);
            }
            SmtExpr::Not(inner) => stack.push(inner),
            SmtExpr::Forall(_, body) | SmtExpr::Exists(_, body) => stack.push(body),
            SmtExpr::Bool(_, children) => stack.extend(children.iter()),
            SmtExpr::Ite(c, t, e) => {
                stack.push(c);
                stack.push(t);
                stack.push(e);
            }
            // Every `Apply` in the cvc5 lowering is a transcendental or
            // algebraic special function (`exp`/`sqrt`/`sin`/`cos`/`abs`/
            // `min`/`max`); those are the NRA/NRAT core. None are in the
            // linear/EUF fragment Carcara audits.
            SmtExpr::Apply(name, _) => {
                return Err(format!(
                    "intrinsic `{name}` is outside the Carcara-audited (linear / EUF) fragment"
                ));
            }
        }
    }
    Ok(())
}

/// Whether `expr` is a numeric literal (a constant coefficient), so that
/// `lit * x` is recognized as linear.
fn is_numeric_literal(expr: &SmtExpr) -> bool {
    matches!(expr, SmtExpr::RealLit(_) | SmtExpr::IntLit(_))
}

/// Whether the whole property is in the fragment Carcara can audit. Checks the
/// postcondition and every precondition.
pub fn property_in_audit_fragment(property: &SmtProperty) -> Result<(), String> {
    expr_in_audit_fragment(&property.postcondition)?;
    for pre in &property.preconditions {
        expr_in_audit_fragment(pre)?;
    }
    Ok(())
}

/// Render an [`SmtExpr`] as an SMT-LIB-2 term string.
///
/// This mirrors the operator mapping in `tier_b::lower_to_cvc5` so the rendered
/// SMT-LIB problem is the same problem cvc5 lowered and proved: the auditor
/// re-checks the cvc5 proof against THIS problem, so a divergence here would
/// make Carcara reject a sound cvc5 proof. The renderer is only ever called
/// after [`property_in_audit_fragment`] has accepted the property, so the
/// transcendental / nonlinear `Apply` and nonlinear-`Mul` cases are
/// unreachable; they render to an explicit `(_ outside-fragment ...)` marker
/// (never silently) so any future caller that skips the gate produces an
/// obviously-malformed problem Carcara rejects, rather than a plausible-looking
/// wrong one.
fn render_term(expr: &SmtExpr) -> String {
    match expr {
        SmtExpr::Var(name) => name.clone(),
        // SMT-LIB renders a negative real as `(- 1.0)`; a bare `-1.0` token is
        // not a valid SMT-LIB numeral.
        SmtExpr::RealLit(v) => render_real_lit(*v),
        SmtExpr::IntLit(v) => {
            if *v < 0 {
                format!("(- {})", v.unsigned_abs())
            } else {
                v.to_string()
            }
        }
        SmtExpr::BoolLit(b) => if *b { "true" } else { "false" }.to_string(),
        SmtExpr::Arith(ArithOp::Neg, inner, _placeholder) => {
            format!("(- {})", render_term(inner))
        }
        SmtExpr::Arith(op, l, r) => {
            let sym = match op {
                ArithOp::Add => "+",
                ArithOp::Sub => "-",
                ArithOp::Mul => "*",
                ArithOp::Div => "/",
                ArithOp::Neg => unreachable!("Neg handled above"),
            };
            format!("({sym} {} {})", render_term(l), render_term(r))
        }
        SmtExpr::Cmp(op, l, r) => match op {
            CmpOp::Lt => format!("(< {} {})", render_term(l), render_term(r)),
            CmpOp::Le => format!("(<= {} {})", render_term(l), render_term(r)),
            CmpOp::Gt => format!("(> {} {})", render_term(l), render_term(r)),
            CmpOp::Ge => format!("(>= {} {})", render_term(l), render_term(r)),
            CmpOp::Eq => format!("(= {} {})", render_term(l), render_term(r)),
            CmpOp::Ne => format!("(not (= {} {}))", render_term(l), render_term(r)),
        },
        SmtExpr::Bool(op, children) => {
            let sym = match op {
                BoolOp::And => "and",
                BoolOp::Or => "or",
                BoolOp::Implies => "=>",
            };
            let rendered: Vec<String> = children.iter().map(render_term).collect();
            format!("({sym} {})", rendered.join(" "))
        }
        SmtExpr::Not(inner) => format!("(not {})", render_term(inner)),
        SmtExpr::Forall(bindings, body) | SmtExpr::Exists(bindings, body) => {
            let kw = if matches!(expr, SmtExpr::Forall(_, _)) {
                "forall"
            } else {
                "exists"
            };
            let binders: Vec<String> = bindings
                .iter()
                .map(|(n, s)| format!("({n} {})", smtlib_sort(*s)))
                .collect();
            format!("({kw} ({}) {})", binders.join(" "), render_term(body))
        }
        SmtExpr::Ite(c, t, e) => {
            format!(
                "(ite {} {} {})",
                render_term(c),
                render_term(t),
                render_term(e)
            )
        }
        SmtExpr::Apply(name, _) => format!("(_ outside-fragment {name})"),
    }
}

/// Render a finite f64 as an SMT-LIB-2 Real term.
///
/// Under the `smt` feature this renders the literal's EXACT f64 VALUE as the
/// rational `numerator/denominator` (via `num_rational::BigRational::from_float`,
/// the SAME lowering `tier_b::lower_to_cvc5` uses for a `RealLit` since
/// chelis#444). This is load-bearing for the audit: cvc5's Alethe proof terms
/// carry the exact-f64 rational (e.g. `0.1_f64` is `3602879701896397/2^55`, not
/// `1/10`), so a decimal rendering here would NOT polyeq-match the proof's term
/// for a non-representable literal and Carcara would spuriously reject a sound
/// proof. The rational is emitted as SMT-LIB-2 `(/ n.0 d.0)` (or a bare `n.0`
/// when the denominator is 1), with a negative value wrapped in `(- ...)` since
/// SMT-LIB has no negative-numeral token.
///
/// Without the `smt` feature (the renderer's shape unit tests, no cvc5 in play)
/// `num-rational` is absent, so it falls back to the shortest round-tripping
/// decimal. That path never feeds a live cvc5 proof, so the exact-vs-decimal
/// distinction is moot there.
#[cfg(feature = "smt")]
fn render_real_lit(v: f64) -> String {
    // `from_float` returns Some for every finite f64 (None only on inf/NaN,
    // which the caller has already screened out of the fragment). The fallback
    // to the decimal form keeps this total if that ever changes.
    let Some(exact) = num_rational::BigRational::from_float(v) else {
        return render_real_lit_decimal(v);
    };
    // `BigRational` normalizes to a positive denominator with the sign carried
    // on the numerator, and `BigInt`'s `Display` prints a leading `-` for a
    // negative value. Inspect the printed numerator's sign so the SMT-LIB term
    // wraps the magnitude in `(- ...)` (SMT-LIB has no negative numeral), then
    // emit `(/ n.0 d.0)` for a true fraction or a bare `n.0` for an integer.
    let numer = exact.numer().to_string();
    let denom = exact.denom().to_string();
    let (negative, abs_numer) = match numer.strip_prefix('-') {
        Some(rest) => (true, rest.to_string()),
        None => (false, numer),
    };
    let body = if denom == "1" {
        format!("{abs_numer}.0")
    } else {
        format!("(/ {abs_numer}.0 {denom}.0)")
    };
    if negative {
        format!("(- {body})")
    } else {
        body
    }
}

/// The decimal fallback for the renderer's shape unit tests (no `smt` feature,
/// so no `num-rational` and no live cvc5 proof to match). SMT-LIB has no
/// negative-numeral token, so a negative value is wrapped in `(- ...)`; the
/// magnitude is rendered with a decimal point so it parses as a Real, not an Int.
#[cfg(not(feature = "smt"))]
fn render_real_lit(v: f64) -> String {
    render_real_lit_decimal(v)
}

/// The shortest-round-tripping-decimal rendering. Used as the no-`smt` renderer
/// and as the `smt`-path fallback if `from_float` ever returns `None`.
fn render_real_lit_decimal(v: f64) -> String {
    let mag = format!("{}", v.abs());
    let mag = if mag.contains('.') || mag.contains('e') || mag.contains('E') {
        mag
    } else {
        format!("{mag}.0")
    };
    if v.is_sign_negative() && v != 0.0 {
        format!("(- {mag})")
    } else {
        mag
    }
}

/// Render the SMT-LIB-2 logic name for a property, mirroring
/// `tier_b::solve_property_cvc5`'s logic selection for the audited fragment.
/// Inside the linear/EUF fragment there are no transcendentals, so the base is
/// always `LRA`/`LIRA`-style linear arithmetic; we render `QF_LIRA` (or its
/// quantified form) to cover mixed Int/Real linear arithmetic, which cvc5's
/// Alethe output targets.
fn smtlib_logic(property: &SmtProperty) -> &'static str {
    let has_quantifier = expr_has_quantifier(&property.postcondition)
        || property.preconditions.iter().any(expr_has_quantifier);
    if has_quantifier { "LIRA" } else { "QF_LIRA" }
}

/// Whether an expression contains a quantifier (iterative).
fn expr_has_quantifier(expr: &SmtExpr) -> bool {
    let mut stack = vec![expr];
    while let Some(node) = stack.pop() {
        match node {
            SmtExpr::Forall(_, _) | SmtExpr::Exists(_, _) => return true,
            SmtExpr::Arith(_, l, r) | SmtExpr::Cmp(_, l, r) => {
                stack.push(l);
                stack.push(r);
            }
            SmtExpr::Not(inner) => stack.push(inner),
            SmtExpr::Bool(_, children) => stack.extend(children.iter()),
            SmtExpr::Apply(_, args) => stack.extend(args.iter()),
            SmtExpr::Ite(c, t, e) => {
                stack.push(c);
                stack.push(t);
                stack.push(e);
            }
            SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {}
        }
    }
    false
}

/// Render the full SMT-LIB-2 PROBLEM text for a property: the logic, the
/// constant declarations, an `(assert ...)` for each precondition, and the
/// asserted NEGATION of the postcondition. This is exactly cvc5's assertion
/// stack (`tier_b::solve_property_cvc5` asserts each precondition then the
/// negated postcondition), rendered as text so Carcara can re-check cvc5's
/// proof against it.
///
/// Returns `Err` if the property is outside the audited fragment (so the caller
/// reports [`CarcaraAudit::Unavailable`] rather than rendering a problem the
/// proof is not about).
pub fn render_smtlib_problem(property: &SmtProperty) -> Result<String, String> {
    property_in_audit_fragment(property)?;

    let mut out = String::new();
    out.push_str(&format!("(set-logic {})\n", smtlib_logic(property)));
    for (name, sort) in &property.variables {
        out.push_str(&format!("(declare-const {name} {})\n", smtlib_sort(*sort)));
    }
    for pre in &property.preconditions {
        out.push_str(&format!("(assert {})\n", render_term(pre)));
    }
    out.push_str(&format!(
        "(assert (not {}))\n",
        render_term(&property.postcondition)
    ));
    out.push_str("(check-sat)\n");
    Ok(out)
}

/// Re-check a cvc5 `Proved` result by routing cvc5's Alethe proof through
/// Carcara.
///
/// This is the WI-16 acceptance path: cvc5 has just decided `property` UNSAT
/// (i.e. `Proved`); we ask cvc5 for the Alethe proof of that UNSAT, render the
/// originating SMT-LIB problem, and have Carcara independently re-check the
/// proof against the problem. The return value is the evidence dimension:
///
/// - [`CarcaraAudit::Confirmed`] / [`CarcaraAudit::ConfirmedModuloRewrites`]:
///   Carcara re-checked the cvc5 proof (fully, or modulo trusted holes).
/// - [`CarcaraAudit::Failed`]: Carcara REJECTED the proof (or the
///   problem/proof did not parse). A cvc5-vs-auditor disagreement.
/// - [`CarcaraAudit::Unavailable`]: the property is outside the audited
///   fragment, or cvc5 emitted no Alethe proof, so there was nothing to audit.
///
/// This builds its own short-lived cvc5 solver (with `produce-proofs` enabled,
/// which the production solve path does NOT set) using the SAME
/// [`crate::tier_b::lower_to_cvc5`] lowering the production solve used, so the
/// audited assertions are the assertions cvc5 proved. It is invoked only on the
/// `Proved` path, only under the `carcara` feature.
///
/// ISOLATION NOTE (latent, inert today): this re-solve runs cvc5 IN-PROCESS,
/// even when the production host has enabled the worker-isolation path
/// (`crate::worker`). It is inert today because no shipping host turns on both
/// `--features carcara` and isolation (the `chelis` CLI does not enable
/// `carcara`). But a future host that does would run this audit re-solve in the
/// non-isolated parent, defeating the isolation guarantee for the audit's cvc5
/// call. This must be escalated and routed through the isolation worker BEFORE
/// any such host ships; it is not a routine fix.
#[cfg(feature = "carcara")]
pub fn audit_cvc5_proof(property: &SmtProperty, timeout_ms: u64) -> CarcaraAudit {
    let (problem, alethe) = match capture_alethe_proof(property, timeout_ms) {
        Ok(pair) => pair,
        // Anything that means "there is no proof to re-check" is Unavailable,
        // not a Failed audit.
        Err(audit) => return audit,
    };
    run_carcara_check(&problem, &alethe)
}

/// Solve `property` with cvc5 under `produce-proofs`, and on UNSAT capture both
/// the rendered SMT-LIB problem and the cvc5 Alethe proof string.
///
/// Returns `Ok((problem, alethe))` when there is a proof to re-check, or
/// `Err(CarcaraAudit::Unavailable(..))` when there is not (out of fragment, no
/// reproduced UNSAT, empty proof). Factored out of [`audit_cvc5_proof`] so the
/// captured pair is inspectable.
#[cfg(feature = "carcara")]
fn capture_alethe_proof(
    property: &SmtProperty,
    timeout_ms: u64,
) -> Result<(String, String), CarcaraAudit> {
    use crate::tier_b::lower_to_cvc5;
    use cvc5_rs::{Kind, ProofComponent, ProofFormat, Solver, TermManager};
    use std::collections::HashMap;

    // Only the linear / EUF fragment has a cvc5 Alethe proof to re-check.
    // Outside it, there is nothing to audit (and rendering the SMT-LIB problem
    // would be meaningless), so report Unavailable, never Failed.
    let problem = match render_smtlib_problem(property) {
        Ok(p) => p,
        Err(reason) => return Err(CarcaraAudit::Unavailable(reason)),
    };

    let tm = TermManager::new();
    let mut solver = Solver::new(&tm);

    // The audited fragment is linear (no transcendentals); choose the linear
    // logic matching the rendered problem so cvc5 produces an Alethe proof in
    // that theory.
    let logic = smtlib_logic(property);
    solver.set_logic(logic);
    // produce-proofs is what makes get_proof return a proof on UNSAT; the
    // production solve path deliberately does not enable it.
    solver.set_option("produce-proofs", "true");
    solver.set_option("proof-format-mode", "alethe");
    // Finest granularity: cvc5 emits the most detailed Alethe proof, so Carcara
    // re-checks the maximum structure. The named RARE/DSL rewrite leaves cvc5
    // emits (`rare_rewrite (... "evaluate")`) cannot be checked in-process
    // without cvc5's compiled rewrite database, so they are trusted as holes by
    // `run_carcara_check` (reported as ConfirmedModuloRewrites). Coarser
    // granularities (`macro`/`rewrite`/`theory-rewrite`) do NOT eliminate the
    // `rare_rewrite` leaves, so they would only reduce what Carcara verifies for
    // no benefit.
    solver.set_option("proof-granularity", "dsl-rewrite");
    solver.set_option("tlimit-per", &timeout_ms.to_string());

    // Declare variables exactly as the production lowering does.
    let mut vars: HashMap<String, cvc5_rs::Term> = HashMap::new();
    let mut sorts: HashMap<String, SmtSort> = HashMap::new();
    for (name, sort) in &property.variables {
        let cvc5_sort = match sort {
            SmtSort::Real => tm.real_sort(),
            SmtSort::Int => tm.integer_sort(),
            SmtSort::Bool => tm.boolean_sort(),
        };
        vars.insert(name.clone(), tm.mk_const(cvc5_sort, name));
        sorts.insert(name.clone(), *sort);
    }

    // Assert preconditions, then the negated postcondition, mirroring
    // tier_b::solve_property_cvc5. Any lowering Err here means the fragment
    // check admitted something cvc5 cannot build; report Unavailable (there is
    // no proof) rather than a false Failed.
    let mut assertions: Vec<cvc5_rs::Term> = Vec::new();
    for pre in &property.preconditions {
        match lower_to_cvc5(&tm, pre, &vars, &sorts) {
            Ok((t, SmtSort::Bool)) => {
                solver.assert_formula(t.clone());
                assertions.push(t);
            }
            _ => {
                return Err(CarcaraAudit::Unavailable(
                    "a precondition did not lower to a Bool term for the audit".to_string(),
                ));
            }
        }
    }
    let post = match lower_to_cvc5(&tm, &property.postcondition, &vars, &sorts) {
        Ok((t, SmtSort::Bool)) => t,
        _ => {
            return Err(CarcaraAudit::Unavailable(
                "the postcondition did not lower to a Bool term for the audit".to_string(),
            ));
        }
    };
    let negated = tm.mk_term(Kind::CVC5_KIND_NOT, &[post]);
    solver.assert_formula(negated.clone());
    assertions.push(negated);

    // Re-decide. We only audit a Proved (UNSAT) goal; if this audit solve does
    // not reproduce UNSAT (timeout under produce-proofs, or a divergence), there
    // is no proof to re-check -> Unavailable, not Failed.
    let result = solver.check_sat();
    if !result.is_unsat() {
        return Err(CarcaraAudit::Unavailable(
            "cvc5 did not reproduce UNSAT under produce-proofs, so no Alethe proof was \
             available to audit"
                .to_string(),
        ));
    }

    let proofs = solver.get_proof(ProofComponent::CVC5_PROOF_COMPONENT_FULL);
    let Some(proof) = proofs.into_iter().next() else {
        return Err(CarcaraAudit::Unavailable(
            "cvc5 returned no proof for the UNSAT result".to_string(),
        ));
    };

    // Name the assertions a0, a1, ... so cvc5 can reference them in the Alethe
    // proof's assume steps.
    let names: Vec<String> = (0..assertions.len()).map(|i| format!("a{i}")).collect();
    let name_refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let alethe = solver.proof_to_string(
        proof,
        ProofFormat::CVC5_PROOF_FORMAT_ALETHE,
        &assertions,
        &name_refs,
    );
    if alethe.trim().is_empty() {
        return Err(CarcaraAudit::Unavailable(
            "cvc5 produced an empty Alethe proof string".to_string(),
        ));
    }

    Ok((problem, alethe))
}

/// Count how many steps in an Alethe proof use each of the [`TRUSTED_HOLE_RULES`].
///
/// Alethe renders a step's rule as the token immediately after `:rule`, e.g.
/// `(step t1 (cl ...) :rule hole ...)`. We scan for `:rule <name>` occurrences
/// of each trusted kind. This is a disclosure aid (it powers
/// [`CarcaraAudit::ConfirmedModuloRewrites`]), not a checker: the actual
/// trust decision is Carcara's (native `hole`/`lia_generic` plus our
/// allowlisted `rare_rewrite`); this scan reports WHICH of those Carcara hit.
/// Only kinds with a non-zero count are returned, sorted by the canonical order
/// in [`TRUSTED_HOLE_RULES`].
#[cfg(feature = "carcara")]
fn count_trusted_hole_rules(proof: &str) -> Vec<TrustedRuleCount> {
    TRUSTED_HOLE_RULES
        .iter()
        .filter_map(|&rule| {
            // Match `:rule <name>` where <name> is followed by whitespace or a
            // delimiter, so `hole` does not match a longer rule that happens to
            // start with "hole".
            let needle = format!(":rule {rule}");
            let count = proof
                .match_indices(&needle)
                .filter(|(idx, _)| {
                    let after = idx + needle.len();
                    proof[after..]
                        .chars()
                        .next()
                        .is_none_or(|c| c.is_whitespace() || c == ')' || c == '(')
                })
                .count();
            (count > 0).then_some(TrustedRuleCount {
                rule: rule.to_string(),
                count,
            })
        })
        .collect()
}

/// Run Carcara on an explicit `(problem, proof)` pair and map the result to a
/// [`CarcaraAudit`] outcome.
///
/// Non-elaborated checker config so assume-step matching uses polyeq (mod
/// reordering / n-ary), which tolerates cvc5's term normalization vs our
/// rendered SMT-LIB. The parser enables `allow_int_real_subtyping`: cvc5's
/// Alethe output for a mixed-arithmetic (LIRA) proof freely mixes Int literals
/// (`0`, `-1`) and rational/Real literals (`0/1`, `1/1`) in the same arithmetic
/// term, so without Int-Real subtyping Carcara rejects a SOUND proof on a sort
/// clash (e.g. `expected 'Int', got 'Real'`). The [`ALLOWLISTED_HOLE_RULES`]
/// (`rare_rewrite`) are added to the checker's allowed-rules; `hole` and
/// `lia_generic` are trusted natively by Carcara without being allowlisted.
///
/// `Ok(false)` = fully re-checked, no holes -> [`CarcaraAudit::Confirmed`].
/// `Ok(true)` = re-checked but at least one step was trusted as a hole ->
/// [`CarcaraAudit::ConfirmedModuloRewrites`] carrying the actual trusted-rule
/// kinds + counts scanned from the proof. `Err` = a surfaced failure (a rejected
/// proof, or a parse error), never folded into a confirmed state.
#[cfg(feature = "carcara")]
fn run_carcara_check(problem: &str, proof: &str) -> CarcaraAudit {
    let mut parser_config = carcara::parser::Config::new();
    parser_config.allow_int_real_subtyping = true;

    let mut checker_config = carcara::checker::Config::new();
    for &rule in ALLOWLISTED_HOLE_RULES {
        checker_config.allowed_rules.insert(rule.to_string());
    }

    match carcara::check(
        problem.as_bytes(),
        proof.as_bytes(),
        None,
        parser_config,
        checker_config,
        false,
    ) {
        Ok(false) => CarcaraAudit::Confirmed,
        Ok(true) => {
            // Carcara reported the proof as holey; enumerate which trusted-rule
            // kinds it actually contains so the disclosure is exact. If the scan
            // finds none (Carcara saw a hole our scan does not recognize), say so
            // explicitly via a sentinel rather than reporting an empty, falsely
            // reassuring set.
            let mut trusted = count_trusted_hole_rules(proof);
            if trusted.is_empty() {
                trusted.push(TrustedRuleCount {
                    rule: "unrecognized_hole".to_string(),
                    count: 0,
                });
            }
            CarcaraAudit::ConfirmedModuloRewrites(trusted)
        }
        Err(e) => CarcaraAudit::Failed(format!("Carcara rejected cvc5's Alethe proof: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real_var(name: &str) -> SmtExpr {
        SmtExpr::Var(name.to_string())
    }

    // --- SMT-LIB renderer: shape (no cvc5/carcara needed) ---

    #[test]
    fn renders_linear_real_problem_with_negated_postcondition() {
        // forall-free: x >= 0 |- x + 1 > 0, with x: Real.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(real_var("x")),
                Box::new(SmtExpr::RealLit(0.0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Arith(
                    ArithOp::Add,
                    Box::new(real_var("x")),
                    Box::new(SmtExpr::RealLit(1.0)),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let smt = render_smtlib_problem(&prop).expect("linear real problem renders");
        assert!(smt.contains("(set-logic QF_LIRA)"), "logic: {smt}");
        assert!(smt.contains("(declare-const x Real)"), "decl: {smt}");
        assert!(smt.contains("(assert (>= x 0.0))"), "precond: {smt}");
        // The postcondition is asserted NEGATED (cvc5 proves by refuting it).
        assert!(
            smt.contains("(assert (not (> (+ x 1.0) 0.0)))"),
            "negated postcond: {smt}"
        );
        assert!(smt.contains("(check-sat)"), "check-sat: {smt}");
    }

    #[test]
    fn renders_negative_real_literal_in_smtlib_minus_form() {
        // A bare negative numeral is not valid SMT-LIB; a negative real must be
        // wrapped in `(- ...)`. -1.5 is exactly representable (-3/2), so under
        // the `smt` feature it renders as the exact rational `(- (/ 3.0 2.0))`
        // matching chelis#444's lowering, and without it as the decimal
        // `(- 1.5)`. Either way, no bare negative numeral appears.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(real_var("x")),
                Box::new(SmtExpr::RealLit(-1.5)),
            ),
        };
        let smt = render_smtlib_problem(&prop).expect("renders");
        assert!(!smt.contains("-1.5"), "no bare negative numeral: {smt}");
        #[cfg(feature = "smt")]
        assert!(
            smt.contains("(- (/ 3.0 2.0))"),
            "exact rational of -1.5 is -3/2: {smt}"
        );
        #[cfg(not(feature = "smt"))]
        assert!(smt.contains("(- 1.5)"), "decimal fallback for -1.5: {smt}");
    }

    /// chelis#444 fidelity (smt feature): a NON-representable decimal literal
    /// like 0.1 must render as its EXACT f64 rational, not the decimal `0.1`,
    /// so the rendered problem matches the exact-f64 term cvc5 puts in the
    /// Alethe proof (else polyeq would fail and Carcara would spuriously reject
    /// a sound proof). 0.1_f64 == 3602879701896397 / 2^55.
    #[cfg(feature = "smt")]
    #[test]
    fn renders_non_representable_decimal_as_its_exact_f64_rational() {
        let rendered = render_real_lit(0.1);
        // The exact f64 value of 0.1, matching num_rational::BigRational::from_float.
        let expected = {
            let exact = num_rational::BigRational::from_float(0.1).unwrap();
            format!("(/ {}.0 {}.0)", exact.numer(), exact.denom())
        };
        assert_eq!(
            rendered, expected,
            "0.1 must render as its exact f64 rational"
        );
        // It must NOT be the misleading decimal `0.1` (the bug this guards).
        assert_ne!(rendered, "0.1");
        assert!(
            rendered.contains('/'),
            "a non-integer f64 renders as a fraction"
        );
    }

    #[test]
    fn linear_coefficient_multiply_is_in_fragment() {
        // 2.0 * x is linear (constant coefficient), so it renders.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(SmtExpr::RealLit(2.0)),
                    Box::new(real_var("x")),
                )),
                Box::new(real_var("x")),
            ),
        };
        assert!(
            render_smtlib_problem(&prop).is_ok(),
            "constant-coefficient multiply is linear"
        );
    }

    // --- Negative parity: out-of-fragment is reported, never mis-rendered ---

    #[test]
    fn nonlinear_multiply_is_outside_the_audited_fragment() {
        // x * x is nonlinear; it lives in the NRA core cvc5 has no Alethe
        // proof for. The renderer must reject it (so the audit is Unavailable),
        // not silently emit an SMT-LIB problem.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(real_var("x")),
                    Box::new(real_var("x")),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let err = render_smtlib_problem(&prop).expect_err("x*x is nonlinear");
        assert!(err.contains("nonlinear"), "names nonlinearity: {err}");
    }

    #[test]
    fn transcendental_apply_is_outside_the_audited_fragment() {
        // exp(x) is a transcendental: the NRAT core, no Alethe proof.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Apply("exp".to_string(), vec![real_var("x")])),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let err = render_smtlib_problem(&prop).expect_err("exp is transcendental");
        assert!(err.contains("exp"), "names the intrinsic: {err}");
    }

    #[test]
    fn quantified_problem_selects_a_non_qf_logic() {
        // A quantifier needs a non-QF logic, mirroring the cvc5 lowering.
        let prop = SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: SmtExpr::Forall(
                vec![("k".to_string(), SmtSort::Int)],
                Box::new(SmtExpr::Cmp(
                    CmpOp::Ge,
                    Box::new(SmtExpr::Var("k".to_string())),
                    Box::new(SmtExpr::IntLit(0)),
                )),
            ),
        };
        let smt = render_smtlib_problem(&prop).expect("renders");
        assert!(smt.contains("(set-logic LIRA)"), "non-QF logic: {smt}");
        assert!(smt.contains("(forall ((k Int))"), "quantifier: {smt}");
    }

    #[test]
    fn audit_outcome_distinguishes_confirmed_failed_unavailable() {
        // The evidence dimension is a real multi-way, not a bool: a surfaced
        // failure is distinct from "no proof to audit", and a full check is
        // distinct from a check that trusted rewrite leaves.
        assert!(CarcaraAudit::Confirmed.is_confirmed());
        assert!(!CarcaraAudit::Confirmed.is_failed());

        // ConfirmedModuloRewrites is a confirmation, but a weaker one; it is
        // never a failure and never silently equal to a full Confirmed.
        let modulo = CarcaraAudit::ConfirmedModuloRewrites(vec![TrustedRuleCount {
            rule: "hole".to_string(),
            count: 2,
        }]);
        assert!(modulo.is_confirmed());
        assert!(!modulo.is_failed());
        assert_ne!(modulo, CarcaraAudit::Confirmed);

        let failed = CarcaraAudit::Failed("proof step t5 does not check".to_string());
        assert!(failed.is_failed());
        assert!(!failed.is_confirmed());

        let unavailable = CarcaraAudit::Unavailable("nonlinear: no Alethe proof".to_string());
        assert!(!unavailable.is_confirmed());
        assert!(!unavailable.is_failed());
    }

    // --- WI-16 acceptance: the live cvc5 -> Alethe -> Carcara round-trip ---
    // (carcara feature: needs cvc5 linked AND carcara linked.)

    /// The goal whose REAL cvc5 Alethe proof the round-trip / disclosure tests
    /// share: a linear, provable goal `x >= 0 |- x + 1 > 0` over the reals.
    #[cfg(feature = "carcara")]
    fn linear_acceptance_goal() -> SmtProperty {
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(real_var("x")),
                Box::new(SmtExpr::RealLit(0.0)),
            )],
            postcondition: SmtExpr::Cmp(
                CmpOp::Gt,
                Box::new(SmtExpr::Arith(
                    ArithOp::Add,
                    Box::new(real_var("x")),
                    Box::new(SmtExpr::RealLit(1.0)),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        }
    }

    #[cfg(feature = "carcara")]
    #[test]
    fn valid_cvc5_proof_re_checks_as_confirmed_modulo_trusted_leaves() {
        // cvc5 proves the goal (UNSAT on the negation) and emits an Alethe
        // proof; Carcara re-checks it. The pinned outcome is
        // ConfirmedModuloRewrites: Carcara checks most of the proof step by
        // step, but the REAL cvc5 proof of even this simple linear goal leans on
        // trusted holes (`hole` carrying ARITH_POLY_NORM_REL arithmetic, and
        // `lia_generic`), so the audit is NOT a full hole-free re-derivation.
        // This is the honest, achievable guarantee -- a real independent
        // re-verification of the proof structure modulo cvc5's sound trusted
        // leaves -- not a false full check.
        let audit = audit_cvc5_proof(&linear_acceptance_goal(), 10_000);
        let CarcaraAudit::ConfirmedModuloRewrites(trusted) = &audit else {
            panic!("expected ConfirmedModuloRewrites, got {audit:?}");
        };
        assert!(
            audit.is_confirmed(),
            "modulo-rewrites is still a confirmation"
        );
        assert!(!audit.is_failed(), "a re-checked proof is not a failure");
        // The disclosure must be non-empty (the proof really does lean on
        // trusted leaves), every disclosed kind must be a known trusted rule
        // with a positive count, and the sentinel must NOT appear (every hole in
        // a real cvc5 proof is one of the known kinds).
        assert!(
            !trusted.is_empty(),
            "a modulo-rewrites outcome must disclose at least one trusted rule"
        );
        for t in trusted {
            assert!(
                TRUSTED_HOLE_RULES.contains(&t.rule.as_str()),
                "disclosed rule `{}` must be a known trusted-hole rule",
                t.rule
            );
            assert!(t.count > 0, "disclosed rule `{}` has a zero count", t.rule);
            assert_ne!(
                t.rule, "unrecognized_hole",
                "a real cvc5 proof must not contain a hole the scan cannot name"
            );
        }
    }

    /// DRIFT GUARD for the disclosure: the trusted-rule set the audit reports
    /// must EXACTLY match an independent scan of the captured Alethe proof. If
    /// the enumeration ever drifts from what the proof actually contains (or from
    /// what Carcara trusts), this fails. It also confirms -- by checking the real
    /// proof text -- that the proof genuinely contains `hole` steps, so the
    /// earlier false claim ("only rare_rewrite trusted; every other rule fully
    /// checked") cannot silently return.
    #[cfg(feature = "carcara")]
    #[test]
    fn modulo_rewrites_payload_exactly_matches_an_independent_proof_scan() {
        let goal = linear_acceptance_goal();
        let (_problem, alethe) =
            capture_alethe_proof(&goal, 10_000).expect("cvc5 produces a proof for the linear goal");

        // Independent scan of the captured proof.
        let scanned = count_trusted_hole_rules(&alethe);

        // The audit's disclosed set must equal the independent scan.
        let audit = audit_cvc5_proof(&goal, 10_000);
        let CarcaraAudit::ConfirmedModuloRewrites(disclosed) = audit else {
            panic!("expected ConfirmedModuloRewrites");
        };
        assert_eq!(
            disclosed, scanned,
            "the disclosed trusted-rule set must match an independent proof scan"
        );

        // The real proof must contain `hole` steps with genuine content -- this
        // is the fact the earlier docstring falsely denied.
        assert!(
            alethe.contains(":rule hole"),
            "cvc5's Alethe proof of this goal contains `hole` (trusted-as-unchecked) steps; \
             the audit must disclose them, not claim every non-rare_rewrite rule is checked"
        );
        assert!(
            scanned.iter().any(|t| t.rule == "hole" && t.count > 0),
            "the scan must count the `hole` steps the proof contains"
        );
    }

    #[cfg(feature = "carcara")]
    #[test]
    fn nonlinear_proved_goal_is_unavailable_not_failed() {
        // x*x >= 0 is provable by cvc5 (in QF_NRA) but is OUTSIDE the Alethe
        // fragment, so there is no proof for Carcara to re-check. The audit
        // dimension must be Unavailable (a clean cvc5 proof is NOT discredited),
        // never a false Failed.
        let prop = SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Ge,
                Box::new(SmtExpr::Arith(
                    ArithOp::Mul,
                    Box::new(real_var("x")),
                    Box::new(real_var("x")),
                )),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        };
        let audit = audit_cvc5_proof(&prop, 10_000);
        assert!(
            matches!(audit, CarcaraAudit::Unavailable(_)),
            "a nonlinear Proved goal has no Alethe proof to audit -> Unavailable, got {audit:?}"
        );
    }

    #[cfg(feature = "carcara")]
    #[test]
    fn tampered_alethe_proof_is_surfaced_as_a_failure() {
        // The negative twin of the round-trip: a proof that does NOT establish
        // the empty clause for the given problem must be SURFACED as a Failed
        // audit, never read as Confirmed. We hand Carcara a well-formed problem
        // and a proof whose only step is a `hole` that does not reach the empty
        // clause from the actual premises -- i.e. a proof that does not check.
        let problem = "(declare-const x Real)\n(assert (>= x 0.0))\n(assert (not (> (+ x 1.0) 0.0)))\n(check-sat)\n";
        // A bogus single step asserting an unrelated clause via a rule that does
        // not derive it from the premises: Carcara rejects it.
        let tampered = "(step t1 (cl (> x 0.0)) :rule resolution)\n";
        let audit = run_carcara_check(problem, tampered);
        assert!(
            audit.is_failed(),
            "an invalid/tampered Alethe proof must be a surfaced Failure, got {audit:?}"
        );
    }

    #[cfg(feature = "carcara")]
    #[test]
    fn run_carcara_check_fully_confirms_a_hole_free_proof() {
        // Positive parity for the FULL-check path (Ok(false) -> Confirmed): a
        // minimal, hole-free Alethe proof of a contradictory problem. `x` and
        // `(not x)` resolve to the empty clause; `resolution` is fully checked
        // by Carcara, so there are no trusted leaves and the outcome is the
        // strong Confirmed, not ConfirmedModuloRewrites.
        let problem = "(declare-const x Bool)\n(assert x)\n(assert (not x))\n(check-sat)\n";
        let proof = "(assume a0 x)\n\
                     (assume a1 (not x))\n\
                     (step t1 (cl) :rule resolution :premises (a0 a1) :args (x true))\n";
        let audit = run_carcara_check(problem, proof);
        assert_eq!(
            audit,
            CarcaraAudit::Confirmed,
            "a hole-free empty-clause proof is fully confirmed, got {audit:?}"
        );
    }

    #[cfg(feature = "carcara")]
    #[test]
    fn run_carcara_check_reports_modulo_rewrites_when_a_leaf_is_trusted() {
        // Positive parity for the MODULO path (Ok(true) -> ConfirmedModuloRewrites):
        // a proof whose closing step is a trusted `rare_rewrite` leaf re-checks
        // as holey, so it is reported as the weaker confirmation, never as a
        // full Confirmed.
        let problem = "(declare-const x Int)\n";
        let proof = "(step t1 (cl) :rule rare_rewrite :args (\"evaluate\"))\n";
        let audit = run_carcara_check(problem, proof);
        assert!(
            matches!(audit, CarcaraAudit::ConfirmedModuloRewrites(_)),
            "a proof leaning on a trusted rewrite leaf is ConfirmedModuloRewrites, got {audit:?}"
        );
    }
}
