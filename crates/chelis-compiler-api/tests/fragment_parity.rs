//! The keystone differential gate for fragment-scoped body replacement.
//!
//! For each case this runs the fragment check
//! ([`chelis_compiler_api::check_body_replacement`]) AND full `chelis check`
//! of the rewritten module, and asserts the two verdicts AGREE: full check
//! rejects the rewritten module IFF the fragment check rejects the body. A
//! fragment check that greens a body full check would reject is the cardinal
//! failure this gate exists to catch.
//!
//! ## Agreement definition
//!
//! Both paths run the same passes in the same pinned order TYPE -> EFFECTS ->
//! LINEARITY and short-circuit on the first failing pass. Agreement is
//! primarily the ACCEPT/REJECT boundary. The failing-pass identity (Type vs
//! Effect vs Linearity) is asserted ONLY where both paths agree on a single
//! unambiguous failing pass; a body wrong two ways may differ in first-failing
//! pass between the whole-module walk and the fragment walk and must not flake
//! the gate, provided both reject.
//!
//! Each case also pins its OWN expected accept/reject, so a behavior change in
//! either path fails loudly rather than silently keeping the two paths in
//! lockstep on the wrong answer.
//!
//! ## Full-check shape: module-wrapped and flattened agree
//!
//! The full path `cmd_check_one_deep` runs `check_ir_fitness` (the
//! whole-module type/structural pass, including the cross-def
//! non-termination and binding-cycle detectors) FIRST, THEN
//! `check_typed_program -> check_program -> check_linearity` on the
//! module-WRAPPED rewritten program. [`full_check_verdict`] mirrors that order.
//! The production fragment-authoring path (`compile_new_source_in_context`)
//! FLATTENS module wrappers before the effect and linearity passes. These two
//! shapes now AGREE on every case: the whole-program effect validators
//! (`validate_unhandled_random_roots`, `validate_declared_vs_inferred`) descend
//! into a `(module ...)` wrapper, so a declared-pure function whose body
//! performs `Random`/`Io` is rejected under both shapes. The per-body linearity
//! walk and the type pass already recursed into module-wrapped bodies, so they
//! agreed before; with the effect descent the two shapes are fully aligned.
//!
//! Each case pins both the module-wrapped verdict (`expected_module_wrapped`)
//! and the flattened verdict (`expected_flattened`) so a behavior change in
//! either is loud, and asserts the fragment agrees with the module-wrapped
//! oracle. The two pinned verdicts are equal for every case here.

use chelis_compiler_api::{ReplacementError, ReplacementReport, check_body_replacement};
use chelis_deep::{Atom, Expr};

// ── Surf -> Deep rendering, mirroring `chelis deep` (non-annotated) ──────

/// Render Surf source to canonical Deep `Expr`s, exactly as the `chelis deep`
/// CLI does for a `.ch` file (`parse_str -> desugar_program -> expand_program`).
fn render_deep(surf_source: &str) -> Vec<Expr> {
    let decls = chelis_surf::parser::parse_str(surf_source).expect("surf parse");
    let deep = chelis_surf::desugar::desugar_program(&decls);
    chelis_macros::expand_program(&deep, &chelis_macros::ExpansionOptions::default())
        .expect("macro expand")
        .into_exprs()
}

/// Render a single-function module and return only the resolved function's
/// body subtree. New bodies for a replacement are authored as a tiny
/// well-formed Surf function and the body subtree is extracted from its Deep.
fn render_body(surf_source: &str, function_name: &str) -> Expr {
    let module = render_deep(surf_source);
    let resolved = chelis_deep::resolve_function(&module, function_name).expect("resolve body fn");
    let def = module
        .iter()
        .find_map(|expr| module_decl(expr, resolved.decl_index))
        .unwrap_or_else(|| panic!("could not locate resolved def node for `{function_name}`"))
        .clone();
    chelis_deep::function_body(&def)
        .expect("body function has a body slot")
        .clone()
}

/// Return the `decl_index`-th declaration inside the single `(module ...)`
/// node, borrowed from the program slice.
fn module_decl(expr: &Expr, decl_index: usize) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if !is_tag(list, "module") {
        return None;
    }
    // module.elements: [tag, meta, name, decls...]; decls start at index 3.
    list.elements.get(3 + decl_index)
}

fn is_tag(list: &chelis_deep::List, tag: &str) -> bool {
    matches!(list.elements.first(), Some(Expr::Atom(Atom::Symbol(t), _)) if t == tag)
}

/// Flatten `(module ...)` wrappers into a bare decl list, mirroring
/// `compile_new_source_in_context`'s `flatten_module_decls`.
fn flatten_modules(exprs: &[Expr]) -> Vec<Expr> {
    let mut out = Vec::new();
    for expr in exprs {
        if let Expr::List(list, _) = expr
            && is_tag(list, "module")
        {
            out.extend(flatten_modules(&list.elements[3..]));
            continue;
        }
        out.push(expr.clone());
    }
    out
}

// ── Verdicts and the full-check paths ────────────────────────────────────

/// Which pass first rejected a program, or `Accept` if all passes accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailingPass {
    Type,
    Effect,
    Linearity,
}

/// A check verdict: accept, or reject naming the first failing pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Accept,
    Reject(FailingPass),
}

impl Verdict {
    fn is_accept(self) -> bool {
        matches!(self, Verdict::Accept)
    }
}

/// Run full whole-program `chelis check` on `module`, mirroring
/// `cmd_check_one_deep` exactly: `check_ir_fitness` (the whole-module
/// type/structural pass, including the cross-def `detect_trivial_non_terminating_fns`
/// and `detect_top_level_binding_cycles` detectors) FIRST, THEN
/// `check_typed_program` -> `check_program` (effects) -> `check_linearity`,
/// short-circuiting on the first failing pass and naming it.
///
/// `cmd_check_one_deep`'s overall verdict is "ok iff every reported error list
/// is empty"; its first list comes from `check_ir_fitness`. Omitting that pass
/// would blind this gate to a cross-def structural rejection that the fragment
/// must agree with, so it runs first here. A fitness rejection is named
/// `FailingPass::Type` because the fragment surfaces the same structural
/// rejection as a `ReplacementError::Type`.
fn full_check_verdict(module: &[Expr]) -> Verdict {
    let fitness = chelis_types::check_ir_fitness(module);
    if !fitness.errors.is_empty() {
        return Verdict::Reject(FailingPass::Type);
    }
    let typed = match chelis_types::check_typed_program(module) {
        Ok(checked) => checked,
        Err(_) => return Verdict::Reject(FailingPass::Type),
    };
    let effects = match chelis_effects::check_program(&typed) {
        Ok(checked) => checked,
        Err(_) => return Verdict::Reject(FailingPass::Effect),
    };
    match chelis_types::check_linearity(&effects) {
        Ok(_) => Verdict::Accept,
        Err(_) => Verdict::Reject(FailingPass::Linearity),
    }
}

/// Map a fragment-check result to a [`Verdict`]. A name-resolution miss is a
/// gate-author error (the target must resolve), so it panics rather than
/// becoming a verdict.
fn fragment_verdict(result: &Result<ReplacementReport, ReplacementError>) -> Verdict {
    match result {
        Ok(_) => Verdict::Accept,
        Err(ReplacementError::Type { .. }) => Verdict::Reject(FailingPass::Type),
        Err(ReplacementError::Effect { .. }) => Verdict::Reject(FailingPass::Effect),
        Err(ReplacementError::Linearity { .. }) => Verdict::Reject(FailingPass::Linearity),
        Err(ReplacementError::NameResolution { message, .. }) => {
            panic!("fragment target failed to resolve (gate-author error): {message}")
        }
        Err(ReplacementError::UndeclaredSignature { message, .. }) => {
            // Every parity case targets rendered Deep, which always emits a
            // defsig, so a defsig-less target here is a gate-author error, not
            // a pass verdict.
            panic!("fragment target has no defsig (gate-author error): {message}")
        }
    }
}

// ── The differential assertion ───────────────────────────────────────────

/// Run the fragment check and both full-check shapes for one case, pin each
/// shape's own verdict, and assert the fragment agrees with the module-wrapped
/// oracle.
///
/// The fragment is defined to track `cmd_check_one_deep` (the module-wrapped
/// full check). `expected_module_wrapped` / `expected_flattened` pin each full
/// path's own accept/reject so a behavior change in either is loud; the two are
/// equal for every case here now that the effect pass descends into the module
/// wrapper. `assert_failing_pass` is set only where the module-wrapped full
/// check and the fragment both report a single unambiguous failing pass.
fn assert_parity(
    case: &str,
    module: &[Expr],
    target: &str,
    new_body: &Expr,
    expected_module_wrapped: Verdict,
    expected_flattened: Verdict,
    assert_failing_pass: bool,
) {
    let frag_result = check_body_replacement(module, target, new_body);
    let frag = fragment_verdict(&frag_result);

    // Both full paths check the SAME rewritten module the fragment is defined
    // to agree with. Build it via the same splice the fragment uses.
    let rewritten = chelis_deep::splice_function_body(module, target, new_body.clone())
        .expect("splice for full-check path");
    let module_wrapped = full_check_verdict(&rewritten);
    let flattened = full_check_verdict(&flatten_modules(&rewritten));

    // Pin each full path's own verdict so a behavior change is loud.
    assert_eq!(
        module_wrapped, expected_module_wrapped,
        "[{case}] module-wrapped full-check {module_wrapped:?} != pinned {expected_module_wrapped:?}",
    );
    assert_eq!(
        flattened, expected_flattened,
        "[{case}] flattened full-check {flattened:?} != pinned {expected_flattened:?}",
    );

    // The cardinal invariant: the fragment's accept/reject boundary matches
    // the brief's module-wrapped oracle.
    assert_eq!(
        frag.is_accept(),
        module_wrapped.is_accept(),
        "[{case}] ACCEPT/REJECT disagreement vs module-wrapped: fragment={frag:?} full={module_wrapped:?}; \
         fragment error (if any): {:?}",
        frag_result
            .as_ref()
            .err()
            .map(|e| (e.stage(), e.message().to_string())),
    );

    if assert_failing_pass && !frag.is_accept() {
        assert_eq!(
            frag, module_wrapped,
            "[{case}] failing-pass disagreement vs module-wrapped: fragment={frag:?} full={module_wrapped:?}",
        );
    }
}

/// Shorthand for the common case: both full-check shapes share one verdict.
fn assert_parity_mw(
    case: &str,
    module: &[Expr],
    target: &str,
    new_body: &Expr,
    expected: Verdict,
    assert_failing_pass: bool,
) {
    assert_parity(
        case,
        module,
        target,
        new_body,
        expected,
        expected,
        assert_failing_pass,
    );
}

// ── Fixtures ───────────────────────────────────────────────────────────

/// Real: economoist Gordon growth (`src/growth.ch`). Single real function
/// `gordon_pv(d, r, g) = d / (r - g)`. Checks clean standalone.
const ECONOMOIST_GROWTH: &str = r#"module Economoist.Growth
export (gordon_pv)
def gordon_pv(d: f32, r: f32, g: f32) -> f32 = (d / (r - g))
"#;

/// Real: the shoals f64 Black-Scholes chain
/// (`bs_call_scalar -> bs_call_f64 -> n_cdf64 -> erf64`), copied from
/// `shoals/src/pricing.ch` with the `mc_call_price` Monte-Carlo path (which
/// imports `Nautilus.Distributions.normal_sample`) omitted so the module
/// checks clean standalone with no reef context. The chain is the real
/// cross-def type-and-effect propagation path the brief names.
const SHOALS_CHAIN: &str = r#"module Shoals.Pricing
export (bs_call_scalar)
def abs_f64(x: f64) -> f64 = if lt(x, cast(0.0, f64)) then neg(x) else x
def erf64(x: f64) -> f64 = {
  a1 = cast(0.254829592, f64)
  a2 = cast(-0.284496736, f64)
  a3 = cast(1.421413741, f64)
  a4 = cast(-1.453152027, f64)
  a5 = cast(1.061405429, f64)
  p = cast(0.3275911, f64)
  one = cast(1.0, f64)
  ax = abs_f64(x)
  small = cast(0.00001, f64)
  if lt(ax, small) then mul(x, cast(1.1283791670955126, f64)) else {
    t = div(one, add(one, mul(p, ax)))
    poly = mul(t, add(a1, mul(t, add(a2, mul(t, add(a3, mul(t, add(a4, mul(t, a5)))))))))
    e = exp(neg(mul(ax, ax)))
    y = sub(one, mul(poly, e))
    if lt(x, cast(0.0, f64)) then neg(y) else y
  }
}
def n_cdf64(x: f64) -> f64 = {
  inv_sqrt_2 = cast(0.7071067811865476, f64)
  mul(cast(0.5, f64), sub(cast(1.0, f64), erf64(neg(mul(x, inv_sqrt_2)))))
}
def d1_64(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 = {
  num = add(log(div(s, k)), mul(add(r, mul(cast(0.5, f64), mul(sigma, sigma))), t))
  div(num, mul(sigma, sqrt(t)))
}
def d2_64(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 = sub(d1_64(s, k, r, sigma, t), mul(sigma, sqrt(t)))
def bs_call_f64(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 = {
  nd1 = n_cdf64(d1_64(s, k, r, sigma, t))
  nd2 = n_cdf64(d2_64(s, k, r, sigma, t))
  disc = exp(neg(mul(r, t)))
  sub(mul(s, nd1), mul(k, mul(disc, nd2)))
}
def bs_call_scalar(s: f32, k: f32, r: f32, sigma: f32, t: f32) -> f32 = cast(bs_call_f64(cast(s, f64), cast(k, f64), cast(r, f64), cast(sigma, f64), cast(t, f64)), f32)
"#;

/// Constructed: a recursive integer countdown. The new body of `count_down`
/// references `count_down` itself, so it must resolve against the function's
/// `defsig` in the held context (the old body is excluded). Checks clean
/// standalone.
const RECURSIVE_MODULE: &str = r#"module Frag.Recursive
export (count_down)
def count_down(n: int32) -> int32 = if eq(n, 0) then 0 else count_down(sub(n, 1))
"#;

/// Constructed: a mutually-recursive `is_even` / `is_odd` pair. A new body for
/// `is_even` references `is_odd` (a sibling) and vice versa; both must resolve
/// against their sibling's `defsig` in the held context. Checks clean
/// standalone.
const MUTUAL_RECURSION_MODULE: &str = r#"module Frag.Mutual
export (is_even, is_odd)
def is_even(n: int32) -> bool = if eq(n, 0) then true else is_odd(sub(n, 1))
def is_odd(n: int32) -> bool = if eq(n, 0) then false else is_even(sub(n, 1))
"#;

/// Constructed: an effect-propagation module. `entry` is declared pure
/// (`! { }`) and currently calls only the pure `pure_sibling`. `noisy`
/// performs `Random` via the `dropout` builtin; `logger` performs `Io` via
/// `debug`. Splicing a `noisy`- or `logger`-calling body into `entry` performs
/// an effect under a pure signature; full check rejects it under both the
/// module-wrapped and flattened shapes (the declared-vs-inferred validator
/// descends into the module wrapper). Checks clean standalone.
const EFFECT_MODULE: &str = r#"module Frag.Effect
export (entry)
def noisy(x: tensor[8, f32]) -> tensor[8, f32] = dropout(x, 0.5)
def logger(x: tensor[8, f32]) -> tensor[8, f32] = debug(x)
def pure_sibling(x: tensor[8, f32]) -> tensor[8, f32] = add(x, x)
def entry(x: tensor[8, f32]) -> tensor[8, f32] ! { } = pure_sibling(add(x, x))
"#;

/// Constructed: a `ping` / `pong` mutually-recursive pair where `ping` holds
/// the sole base case (`if eq(n, 0) then 0 ...`) and `pong` is unconditional.
/// Splicing `ping`'s body to drop the base case (new body `pong(sub(n, 1))`)
/// closes the recursion group with no base case anywhere, which the whole-module
/// `detect_trivial_non_terminating_fns` detector flags. The detector is a
/// recursion-group property, so it fires only when the full rewritten module is
/// analyzed; the 2-decl fragment never sees `pong`. Checks clean standalone (the
/// base case is present).
const PINGPONG_MODULE: &str = r#"module Frag.PingPong
export (ping, pong)
def ping(n: int32) -> int32 = if eq(n, 0) then 0 else pong(sub(n, 1))
def pong(n: int32) -> int32 = ping(sub(n, 1))
"#;

/// Constructed: a module carrying a top-level value-binding cycle
/// (`a` reads `b`, `b` reads `a`), which the whole-module
/// `detect_top_level_binding_cycles` detector flags. This is the SECOND
/// cross-def structural detector the fitness pass runs. The body-replacement
/// API only targets functions with a declared signature, so a value-binding
/// cycle cannot be introduced through a fragment splice; this fixture verifies
/// the oracle ([`full_check_verdict`]) now mirrors `cmd_check_one_deep` by
/// running `check_ir_fitness` first and so REJECTS the cycle. Before that
/// change the oracle ran only `check_typed_program`, whose `infer_program`
/// pass does not run `validate_ir_program` and so missed this detector.
const BINDING_CYCLE_MODULE: &str = r#"module Frag.BindingCycle
export (a, b)
a: int32 = add(b, 1)
b: int32 = add(a, 1)
"#;

/// Constructed: a linearity module. `target(x)` takes an owned tensor
/// parameter. A body that consumes `x` (via the consuming `realize`) and then
/// uses it again is a `UseAfterConsume` violation; a body that consumes it
/// exactly once is clean. `consumer` is a sibling that references `target` so
/// the held context retains a real call site. Checks clean standalone.
const LINEARITY_MODULE: &str = r#"module Frag.Linearity
export (target)
def target(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)
def consumer(y: tensor[4, f32]) -> tensor[4, f32] = target(y)
"#;

// ── Real-module cases ─────────────────────────────────────────────────────

#[test]
fn economoist_growth_well_typed_body_agrees_accept() {
    let module = render_deep(ECONOMOIST_GROWTH);
    let body = render_body(
        "module M\ndef f(d: f32, r: f32, g: f32) -> f32 = (d / (r + g))\n",
        "f",
    );
    assert_parity_mw(
        "economoist_growth/well_typed",
        &module,
        "gordon_pv",
        &body,
        Verdict::Accept,
        true,
    );
}

#[test]
fn economoist_growth_precision_mismatch_body_agrees_reject() {
    let module = render_deep(ECONOMOIST_GROWTH);
    // f64 arithmetic returned from an f32-declared function: precision
    // mismatch (no implicit promotion), a TYPE rejection.
    let body = render_body(
        "module M\ndef f(d: f32, r: f32, g: f32) -> f32 = div(cast(d, f64), cast(r, f64))\n",
        "f",
    );
    assert_parity_mw(
        "economoist_growth/precision_mismatch",
        &module,
        "gordon_pv",
        &body,
        Verdict::Reject(FailingPass::Type),
        true,
    );
}

#[test]
fn shoals_chain_well_typed_body_agrees_accept() {
    let module = render_deep(SHOALS_CHAIN);
    // Re-derive bs_call_f64 through the same n_cdf64 chain, reorganized but
    // type-equivalent. Exercises type propagation up the real chain.
    let body = render_body(
        r#"module M
def f(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 = {
  nd1 = n_cdf64(d1_64(s, k, r, sigma, t))
  nd2 = n_cdf64(d2_64(s, k, r, sigma, t))
  disc = exp(neg(mul(r, t)))
  add(mul(s, nd1), neg(mul(k, mul(disc, nd2))))
}
"#,
        "f",
    );
    assert_parity_mw(
        "shoals_chain/well_typed",
        &module,
        "bs_call_f64",
        &body,
        Verdict::Accept,
        true,
    );
}

#[test]
fn shoals_chain_wrong_return_precision_agrees_reject() {
    let module = render_deep(SHOALS_CHAIN);
    // bs_call_f64 declared to return f64; this body returns f32. Precision
    // mismatch against the declared signature: TYPE reject.
    let body = render_body(
        "module M\ndef f(s: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 = cast(s, f32)\n",
        "f",
    );
    assert_parity_mw(
        "shoals_chain/wrong_return_precision",
        &module,
        "bs_call_f64",
        &body,
        Verdict::Reject(FailingPass::Type),
        true,
    );
}

// ── Recursion ──────────────────────────────────────────────────────────────

#[test]
fn recursive_body_resolves_against_signature_agrees_accept() {
    let module = render_deep(RECURSIVE_MODULE);
    // A new recursive body must resolve `count_down` against its defsig, not
    // the excluded old body.
    let body = render_body(
        "module M\ndef f(n: int32) -> int32 = if lt(n, 1) then 0 else add(1, count_down(sub(n, 1)))\n",
        "f",
    );
    assert_parity_mw(
        "recursive/well_typed",
        &module,
        "count_down",
        &body,
        Verdict::Accept,
        true,
    );
}

#[test]
fn recursive_body_wrong_return_type_agrees_reject() {
    let module = render_deep(RECURSIVE_MODULE);
    // The recursive call resolves against the defsig, but its int32 result is
    // cast to f32 and returned under an int32 signature: precision mismatch
    // against the declared return. TYPE reject.
    let body = render_body(
        "module M\ndef f(n: int32) -> int32 = cast(count_down(n), f32)\n",
        "f",
    );
    assert_parity_mw(
        "recursive/wrong_return",
        &module,
        "count_down",
        &body,
        Verdict::Reject(FailingPass::Type),
        true,
    );
}

// ── Mutual recursion ─────────────────────────────────────────────────────

#[test]
fn mutual_recursion_body_resolves_sibling_agrees_accept() {
    let module = render_deep(MUTUAL_RECURSION_MODULE);
    // New body for is_even referencing the sibling is_odd: resolves against
    // is_odd's defsig in the held context.
    let body = render_body(
        "module M\ndef f(n: int32) -> bool = if lt(n, 1) then true else is_odd(sub(n, 1))\n",
        "f",
    );
    assert_parity_mw(
        "mutual_recursion/well_typed",
        &module,
        "is_even",
        &body,
        Verdict::Accept,
        true,
    );
}

#[test]
fn mutual_recursion_body_wrong_type_agrees_reject() {
    let module = render_deep(MUTUAL_RECURSION_MODULE);
    // is_even declared to return bool; the then-branch returns the int32
    // argument while the else-branch returns int32: TYPE reject even though
    // the sibling reference resolves.
    let body = render_body(
        "module M\ndef f(n: int32) -> bool = if lt(n, 1) then n else 0\n",
        "f",
    );
    assert_parity_mw(
        "mutual_recursion/wrong_type",
        &module,
        "is_even",
        &body,
        Verdict::Reject(FailingPass::Type),
        true,
    );
}

// ── Effect propagation across defs ──────────────────────────────────────────

#[test]
fn effect_pure_body_agrees_accept() {
    let module = render_deep(EFFECT_MODULE);
    // entry stays pure: calls only pure_sibling. Accepted by both full shapes
    // and the fragment.
    let body = render_body(
        "module M\ndef f(x: tensor[8, f32]) -> tensor[8, f32] = pure_sibling(add(x, x))\n",
        "f",
    );
    assert_parity_mw(
        "effect/pure_body",
        &module,
        "entry",
        &body,
        Verdict::Accept,
        true,
    );
}

#[test]
fn effect_declared_pure_body_introduces_random() {
    // `entry` is declared pure (`! { }`); the new body calls `noisy`, which
    // performs `Random`. Both full-check shapes REJECT on the effect pass: the
    // declared-vs-inferred validator descends into the `(module ...)` wrapper
    // and sees the nested declared-pure defsig against the Random-performing
    // body. The fragment carries `entry`'s defsig alongside the new def, so its
    // `check_effects_with_context` runs the same declared-vs-inferred check
    // against the new body and also REJECTS: clean accept/reject agreement.
    let module = render_deep(EFFECT_MODULE);
    let body = render_body(
        "module M\ndef f(x: tensor[8, f32]) -> tensor[8, f32] = noisy(x)\n",
        "f",
    );
    assert_parity_mw(
        "effect/declared_pure_introduces_random",
        &module,
        "entry",
        &body,
        Verdict::Reject(FailingPass::Effect),
        true,
    );
}

#[test]
fn effect_declared_pure_body_introduces_io() {
    // The IO counterpart: `entry` is declared pure (`! { }`); the new body
    // calls `logger`, which performs `Io` via `debug`. Both full-check shapes
    // and the fragment REJECT on the effect pass, the same as the Random case.
    let module = render_deep(EFFECT_MODULE);
    let body = render_body(
        "module M\ndef f(x: tensor[8, f32]) -> tensor[8, f32] = logger(x)\n",
        "f",
    );
    assert_parity_mw(
        "effect/declared_pure_introduces_io",
        &module,
        "entry",
        &body,
        Verdict::Reject(FailingPass::Effect),
        true,
    );
}

// ── Linearity ──────────────────────────────────────────────────────────────

#[test]
fn linearity_consume_once_body_agrees_accept() {
    let module = render_deep(LINEARITY_MODULE);
    // Consume `x` exactly once (relu borrows-then-returns its result chain);
    // linearity-clean.
    let body = render_body(
        "module M\ndef f(x: tensor[4, f32]) -> tensor[4, f32] = relu(add(x, x))\n",
        "f",
    );
    assert_parity_mw(
        "linearity/consume_once",
        &module,
        "target",
        &body,
        Verdict::Accept,
        true,
    );
}

#[test]
fn linearity_use_after_consume_agrees_reject() {
    let module = render_deep(LINEARITY_MODULE);
    // `realize(x)` consumes `x`; the later `add(x, y)` uses the consumed `x`:
    // UseAfterConsume. The per-body linearity walk recurses into module-wrapped
    // bodies, so both full shapes and the fragment reject on linearity.
    let body = render_body(
        r#"module M
def f(x: tensor[4, f32]) -> tensor[4, f32] = {
  y = realize(x)
  add(x, y)
}
"#,
        "f",
    );
    assert_parity_mw(
        "linearity/use_after_consume",
        &module,
        "target",
        &body,
        Verdict::Reject(FailingPass::Linearity),
        true,
    );
}

// ── Cross-def structural fitness (whole-module detectors) ───────────────────

#[test]
fn pingpong_drop_base_case_agrees_reject() {
    // `ping` holds the sole base case of the `ping`/`pong` group. Splicing its
    // body to call `pong` unconditionally removes the only base case, so the
    // whole-module `detect_trivial_non_terminating_fns` detector flags both
    // `ping` and `pong` as trivially non-terminating. That detector is a
    // recursion-group property and only fires when the FULL rewritten module is
    // analyzed: the 2-decl fragment `[ping defsig, ping def]` never sees `pong`.
    // The fragment's whole-module fitness gate (added to `check_body_replacement`)
    // runs `check_ir_fitness` on the rewritten module and so REJECTS, matching
    // `cmd_check_one_deep`. Reject identity is a structural Type rejection on
    // both paths.
    let module = render_deep(PINGPONG_MODULE);
    let body = render_body(
        "module M\ndef f(n: int32) -> int32 = pong(sub(n, 1))\n",
        "f",
    );
    assert_parity_mw(
        "pingpong/drop_base_case",
        &module,
        "ping",
        &body,
        Verdict::Reject(FailingPass::Type),
        true,
    );
}

#[test]
fn binding_cycle_oracle_rejects() {
    // The oracle must mirror `cmd_check_one_deep`, which runs `check_ir_fitness`
    // (the whole-module `detect_top_level_binding_cycles` detector) first. A
    // value-binding cycle (`a` reads `b`, `b` reads `a`) is rejected only by
    // that pass; `check_typed_program`'s `infer_program` does not run
    // `validate_ir_program`. A value binding has no declared signature, so it
    // cannot be a `check_body_replacement` target and this detector is exercised
    // at the oracle level only. Before Part 1 the oracle (which ran only
    // `check_typed_program`) ACCEPTED this module; now it REJECTS.
    let module = render_deep(BINDING_CYCLE_MODULE);
    assert_eq!(
        full_check_verdict(&module),
        Verdict::Reject(FailingPass::Type),
        "oracle must reject a top-level binding cycle via the fitness pass",
    );
    assert_eq!(
        full_check_verdict(&flatten_modules(&module)),
        Verdict::Reject(FailingPass::Type),
        "flattened oracle must also reject the binding cycle",
    );
}

// ── Name-resolution surface ─────────────────────────────────────────────────

#[test]
fn name_resolution_miss_is_tagged() {
    let module = render_deep(ECONOMOIST_GROWTH);
    let body = render_body("module M\ndef f(d: f32, r: f32, g: f32) -> f32 = d\n", "f");
    let err = check_body_replacement(&module, "no_such_function", &body)
        .expect_err("unknown target must error");
    assert!(
        matches!(err, ReplacementError::NameResolution { .. }),
        "expected NameResolution, got {err:?}",
    );
}

#[test]
fn value_binding_target_is_name_resolution_error() {
    // A module with a top-level value binding (not a function): resolving it as
    // a function is a NotAFunction -> tagged NameResolution error.
    let module = render_deep(
        "module Frag.Value\nexport (target)\nshared: tensor[4, f32] = to_tensor([1.0, 2.0, 3.0, 4.0])\ndef target(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)\n",
    );
    let body = render_body(
        "module M\ndef f(x: tensor[4, f32]) -> tensor[4, f32] = relu(x)\n",
        "f",
    );
    let err = check_body_replacement(&module, "shared", &body)
        .expect_err("value-binding target must error");
    assert!(
        matches!(err, ReplacementError::NameResolution { .. }),
        "expected NameResolution for value binding, got {err:?}",
    );
}
